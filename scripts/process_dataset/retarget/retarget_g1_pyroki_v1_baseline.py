"""[LEGACY BASELINE — NOT CALLED BY ANYTHING]

Frozen copy of the retargeting script as it was before the 2026-08-12 fixes, kept only so the
old numbers can be reproduced for comparison. The live script is `retarget_g1_pyroki.py`;
its docstring lists what changed and what each change was measured to do.

Known defects preserved here on purpose:
  - hand correspondence off by one joint (degenerate ffknuckle/ffproximal pair)
  - grasp contact cost active in stage 1, distorting the grounding solve
  - no wrist orientation target
Writes the SAME output path as the live script, so always run it with W_OUTSUFFIX set.
"""

"""G1 + bimanual Shadow retargeting with PyRoki (Phase 2: body + hands, contact-aware).

Whole-trajectory batched least-squares (jaxls) retargeting of ParaHome SMPL-X body+hand
keypoints onto the COMPOSITE G1+Shadow robot (65-DOF), ported from pyroki examples 12/11.

Robot: the composite `urdf_pyroki/g1_shadow.urdf`, exported from our G1_shadow.usd via NVIDIA
UsdToUrdf (joint frames patched for consistency; see export_g1_shadow_urdf step). It carries our
EXACT 65 action joints (29 body + 36 Shadow) + collision meshes, so the solve output maps 1:1 to
our action order and self-collision / (Phase-2b) object-contact work.

Costs: local alignment (relative joint-vector + angle, learned per-joint scale) + global alignment
over 54 body+hand correspondences · floor contact · foot skating · self-collision · smoothness ·
rest (coupled J0 held ~0) · joint limits.  [object-contact grasp cost = Phase 2b]

Runs in env_pyroki (numpy>=2 + jax + pyroki), NOT env_isaaclab:
    /home/peunsu/anaconda3/envs/env_pyroki/bin/python scripts/process_dataset/retarget/retarget_g1_pyroki.py \
        --clip s100_seg00_pan
→ data/processed/parahome/g1_shadow/<class>/<clip>/0/trajectory_pyroki.npz
Render in env_isaaclab:  render_retarget.py --clip <clip> --variant pyroki
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import trimesh
import yourdfpy
from pyroki.collision import collide, colldist_from_sdf

sys.path.insert(0, str(Path("/home/peunsu/workspace/pyroki/examples")))
from retarget_helpers._utils import create_conn_tree  # noqa: E402

_ROOT = Path("/home/peunsu/workspace/robotis_sh5/source/robotis_sh5")
_PROC = _ROOT / "data" / "processed" / "parahome"
_URDF = _ROOT / "data" / "robots" / "G1" / "urdf_pyroki" / "g1_shadow.urdf"
_ORDER = json.load(open(_ROOT / "data" / "robots" / "G1" / "g1_shadow_joint_order.json"))

# ---- ParaHome joint idx → composite-URDF link name ----------------------------------------
# body: our BODY_KPTS correspondence (minus ball-foot dups 18/22 → ankle_roll used at 17/21)
_BODY = [
    (0, "pelvis"), (4, "torso_link"),
    (8, "right_shoulder_pitch_link"), (9, "right_elbow_link"), (10, "right_wrist_yaw_link"),
    (12, "left_shoulder_pitch_link"), (13, "left_elbow_link"), (14, "left_wrist_yaw_link"),
    (15, "right_hip_pitch_link"), (16, "right_knee_link"), (17, "right_ankle_roll_link"),
    (19, "left_hip_pitch_link"), (20, "left_knee_link"), (21, "left_ankle_roll_link"),
]
# hands: HAND_CHAIN (ParaHome hand-local idx → Shadow body); left block +23, right +48
_HAND_CHAIN = {
    "wrist":  ([0], ["palm"]),
    "index":  ([18, 19, 20, 21], ["ffknuckle", "ffproximal", "ffmiddle", "ffdistal"]),
    "middle": ([14, 15, 16, 17], ["mfknuckle", "mfproximal", "mfmiddle", "mfdistal"]),
    "ring":   ([10, 11, 12, 13], ["rfknuckle", "rfproximal", "rfmiddle", "rfdistal"]),
    "pinky":  ([6, 7, 8, 9], ["lfknuckle", "lfproximal", "lfmiddle", "lfdistal"]),
    "thumb":  ([22, 23, 24], ["thproximal", "thmiddle", "thdistal"]),
}


def _build_correspondence():
    pairs = list(_BODY)
    for side, off in (("l", 23), ("r", 48)):
        for local, shadow in _HAND_CHAIN.values():
            for pl, sh in zip(local, shadow):
                pairs.append((off + pl, f"robot0_{side}_{sh}"))
    return pairs   # list of (parahome_global_idx, urdf_link_name)


_FOOT_PLANT_H, _FOOT_PLANT_VZ, _FPS = 0.06, 0.15, 30.0
_PARA_BALL_L, _PARA_BALL_R = 22, 18
# G1 ankle_roll_link origin sits this far ABOVE the foot sole (URDF foot-corner contact spheres at
# z=-0.031, r=0.005 → sole at -0.036). floor_contact targets the ANKLE, so its z target must be this
# offset (not 0) — else pulling the ankle to z=0 drives the sole ~3.6 cm INTO the floor (feet penetrate
# at high W_FLOOR) or, when weak, the balance leaves the foot floating. Target = sole-on-floor.
_ANKLE_SOLE_OFF = 0.036
_MOVE_LESS = ["left_hip_yaw_joint", "right_hip_yaw_joint", "waist_yaw_joint"]
# tendon-coupled distal joints our robot doesn't actuate (absorbed) → hold near 0 in the solve
_HOLD_ZERO = [f"robot0_{s}_{f}J0" for s in "lr" for f in ("FF", "MF", "RF", "LF")]

# ---- Phase 2b object-contact grasp: fingertip PAD (distal body + offset) ↔ object surface ----------
_RAW_SCAN = _ROOT / "data" / "raw" / "parahome" / "data" / "scan"
# fingertip_pad_pos order (CLAUDE.md): LEFT[th,ff,mf,rf,lf] then RIGHT[th,ff,mf,rf,lf].
# each entry = (human_pad_idx, side, distal-finger)
_FT_PADS = [(base + j, side, fg) for side, base in (("l", 0), ("r", 5))
            for j, fg in enumerate(("th", "ff", "mf", "rf", "lf"))]
# distal-local pad offset (grasp-env FINGERTIP_OFFSETS): right verbatim, left non-thumb Y-mirrored
_FT_OFF_R = {"th": [-0.0085, 0.0, 0.02], "ff": [0.0, -0.006, 0.0175], "mf": [0.0, -0.006, 0.0175],
             "rf": [0.0, -0.006, 0.0175], "lf": [0.0, -0.006, 0.0175]}
# Wrap-link (palm/proximal/middle) PALMAR-surface offset. The contact cost pulls these links to the
# object surface, but at offset 0 it pulls the link ORIGIN (proximal joint, behind the palmar surface),
# so for palm/thumb-base-dominated grasps the hand ends up offset from the object. Push the contact
# point from the origin toward the palmar surface along the link's palmar normal (fingers -Y, thumb -X,
# palm -Y; left Y-mirrored, thumb Y=0 invariant), magnitude = env W_WRAP_OFFSET (palm ×2, it is thicker).
# W_WRAP_OFFSET=0 reproduces the old origin behavior.  [EXPERIMENT KNOB]
_WRAP_PALMAR_R = {"palm": [0.0, -1.0, 0.0],
                  **{f"{fg}{sg}": [0.0, -1.0, 0.0] for fg in ("ff", "mf", "lf", "rf")
                     for sg in ("proximal", "middle")},
                  **{f"th{sg}": [-1.0, 0.0, 0.0] for sg in ("proximal", "middle")}}


def _wrap_offset(link_name, mag):
    body = link_name.split("_", 2)[2]; side = link_name.split("_")[1]
    n = _WRAP_PALMAR_R.get(body, [0.0, 0.0, 0.0])
    if side == "l":
        n = [n[0], -n[1], n[2]]                                  # left hand = Y-reflection
    s = mag * (2.0 if body == "palm" else 1.0)                   # palm is thicker → larger offset
    return [n[0] * s, n[1] * s, n[2] * s]
_OBJ_LINVEL_TH, _OBJ_ANGVEL_TH, _CENTROID_TH = 0.05, 0.25, 0.05
# Contact gate for RETARGETING uses an ABSOLUTE fingertip→surface distance (not the RL env's tight
# relative "within 1.5 cm of the closest finger" gate) — the thumb opposes the fingers across the handle
# so it sits ~2.6 cm from the surface while ff/mf/rf are ~0.4 cm; the relative gate wrongly drops it.
_CONTACT_ABS_TH = 0.035


def _quat2R(wxyz):
    w, x, y, z = wxyz
    return onp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], onp.float64)


def _contact_signal(ftpad, obj_base, obj_name):
    """Per-frame/fingertip grasp-contact mask (F,10), matching the RL env: (object being manipulated:
    linvel/angvel over threshold) AND (fingertip pad near the object surface). Uses the ParaHome scan
    mesh nearest-vertex relative gate; falls back to object-centroid distance if the mesh is absent."""
    F = len(ftpad)
    op = obj_base[:, :3]; oq = obj_base[:, 3:7]
    lv = onp.zeros_like(op); lv[:-1] = (op[1:] - op[:-1]) * _FPS
    spd = onp.linalg.norm(lv, axis=-1)
    dotq = onp.abs((oq[:-1] * oq[1:]).sum(-1)).clip(0, 1)
    ang = onp.zeros(F); ang[:-1] = 2 * onp.arccos(dotq) * _FPS
    vel = (spd > _OBJ_LINVEL_TH) | (ang > _OBJ_ANGVEL_TH)
    src = _RAW_SCAN / obj_name / "simplified" / "base.obj"
    if src.exists():
        m = trimesh.load(str(src), process=False, force="mesh")
        V = onp.asarray(m.vertices, onp.float64)
        mask = onp.zeros((F, 10), onp.float32); pen = []
        for t in range(F):
            R = _quat2R(oq[t])
            ftl = (ftpad[t] - op[t]) @ R                       # fingertip pads in object-local
            cvd = onp.linalg.norm(V[:, None, :] - ftl[None, :, :], axis=-1).min(0)   # (10,) nearest-vertex dist
            near = cvd < _CONTACT_ABS_TH                       # absolute surface-proximity gate
            mask[t] = vel[t] & near; pen.append(cvd)
        pen = onp.array(pen)
        c = mask.astype(bool)
        print(f"[pyroki-retarget] contact frames/finger: {mask.sum(0).astype(int).tolist()}  "
              f"pad→surface on contact: mean {pen[c].mean()*100:.1f} cm" if c.any() else
              "[pyroki-retarget] no contact frames detected")
        return mask
    dist = onp.linalg.norm(ftpad - op[:, None, :], axis=-1)
    return (vel[:, None] & (dist < _CENTROID_TH)).astype(onp.float32)


def _foot_contact(jp, ball_idx):
    z = jp[:, ball_idx, 2]
    vz = onp.zeros_like(z); vz[1:] = (z[1:] - z[:-1]) * _FPS
    return ((z < _FOOT_PLANT_H) & (onp.abs(vz) < _FOOT_PLANT_VZ)).astype(onp.float32)


def _pelvis_target_R(jp):
    """Per-frame target pelvis orientation (world) from ParaHome body keypoints, in the G1 pelvis-link
    convention X=forward, Y=left, Z=up.  up = midshoulder→ (spine), left = right→left hip, forward = left×up.
    Constraining the free root to this removes the spurious roll that saturates hip_roll/waist_roll."""
    p_rs, p_ls = jp[:, 8], jp[:, 12]      # right / left shoulder
    p_rh, p_lh = jp[:, 15], jp[:, 19]     # right / left hip
    z = 0.5 * (p_ls + p_rs) - 0.5 * (p_lh + p_rh)          # up (hips → shoulders)
    z /= onp.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    y = p_lh - p_rh                                        # left (right hip → left hip)
    y = y - onp.sum(y * z, axis=1, keepdims=True) * z      # orthogonalize against up
    y /= onp.linalg.norm(y, axis=1, keepdims=True) + 1e-9
    x = onp.cross(y, z); x /= onp.linalg.norm(x, axis=1, keepdims=True) + 1e-9   # forward = left × up
    y = onp.cross(z, x)                                    # re-orthonormalize (up × forward)
    return onp.stack([x, y, z], axis=-1).astype(onp.float32)   # (T,3,3) columns = frame axes in world


def _flat_heightmap(jp, bins=48, margin=0.6):
    """Flat floor (z=0) heightmap spanning the clip's body-keypoint xy extent — the ParaHome scene has
    no terrain, so world_collision just prevents the robot sinking below the floor (and lifts it up)."""
    xy = jp[:, :23, :2].reshape(-1, 2)
    lo = xy.min(0) - margin; hi = xy.max(0) + margin
    cx, cy = (lo + hi) / 2.0; dx, dy = (hi - lo)
    box = trimesh.creation.box(extents=[float(dx), float(dy), 0.1])
    box.apply_translation([float(cx), float(cy), -0.05])          # top face at z=0
    return pk.collision.Heightmap.from_trimesh(box, x_bins=bins, y_bins=bins)


# [SCENE-COLLISION] ---------------------------------------------------------------------------
def _context_boxes(sm, obj_key, jp_cloud, radius=1.0, support_radius=1.5,
                   reach=float(os.environ.get("W_SCENEREACH", 0.2)),
                   max_boxes=int(os.environ.get("W_SCENEMAXBOX", 64))):
    """Boxes approximating the fixed scene the env spawns, for the world_collision cost.

    The floor heightmap above was the ONLY thing the retarget collided against, so the counter, sink
    and board simply were not in the problem. The solve happily put the LEFT hand inside the
    countertop, and the sim then threw it out at up to 5.47 m/s (470 of 501 frames) — every bit of
    that impulse lands on the robot, because the context spawns kinematic.

    Geometry choice, measured on s101_seg12_knife (78 robot capsules x 301 frames):

        one OBB per object     max penetration 36.6 cm, 21493 pairs — mostly the OBB's own slack
                               (the sink mesh fills 31% of its OBB), and it traps the LEGS
        per-hull OBBs          max  6.0 cm,  1813 pairs, and the deepest links are the LEFT HAND —
                               which is the defect we are actually chasing
        heightmap              max 117 cm, deepest links the ANKLES: a height field is solid all the
                               way down, so a robot standing at a counter reads as buried in it

    So: convex-decompose, then take each hull's OBB. pyroki has no hull primitive, but Capsule-Box
    is native and `capsule_box` uses a real box SDF (inside, the depth is to the NEAREST FACE), which
    is what keeps a pelvis grazing the counter edge at ~4 cm instead of the ~80 cm a height field
    would report.

    Hull count is NOT controllable here: trimesh delegates to `pyVHACD.compute_vhacd`, which takes no
    keyword arguments, so the decomposition runs at its own default (~64 hulls on the sink). The
    simulator converts with `max_convex_hulls=16`, so the two differ — the retarget sees a somewhat
    tighter scene than PhysX does. Taking the OBB of each hull inflates it back, which pushes the
    other way; the net is verified by measurement, not assumed. Aligning them properly means
    re-converting the context USDs at a matching hull count.

    Selection mirrors the env (proximity of the frame-0 centroid to the active object's swept path
    + the below-object support), so both see the same scene.
    """
    act = sm[obj_key].astype(onp.float64)
    act_xy, act0 = act[:, :2], act[0]
    cands = []
    for k in (kk for kk in sm.files if kk.startswith("ctx__") and kk.endswith("__base")):
        pose0 = sm[k][0].astype(onp.float64)
        dmin = float(onp.linalg.norm(act_xy - pose0[None, :2], axis=1).min())
        cands.append((k.split("__")[1], pose0, dmin))
    keep = {n for n, _, dm in cands if dm < radius}
    below = [(float(onp.linalg.norm(act0[:2] - q[:2])), n) for n, q, _ in cands
             if q[2] < act0[2] and float(onp.linalg.norm(act0[:2] - q[:2])) < support_radius]
    if below:
        keep.add(min(below)[1])

    centres, extents = [], []
    for name, q, _ in cands:
        if name not in keep:
            continue
        src = _RAW_SCAN / name / "simplified" / "base.obj"
        if not src.exists():
            print(f"[pyroki-retarget] scene: no scan mesh for {name}, skipped")
            continue
        mesh = trimesh.load(str(src), process=False, force="mesh")
        rot = trimesh.transformations.quaternion_matrix([q[3], q[4], q[5], q[6]])[:3, :3]
        mesh.vertices = mesh.vertices @ rot.T + q[:3]
        parts = mesh.convex_decomposition()                       # pyVHACD; takes no kwargs
        parts = parts if isinstance(parts, list) else [parts]
        if len(parts) <= 1:
            # A single hull means the object collapsed to its own bounding volume, which for a
            # counter is 3x its real volume and would trap the robot's legs. Loud on purpose: the
            # earlier silent fallback to one OBB per object is what made the first run a no-op.
            print(f"[pyroki-retarget] scene: WARNING {name} decomposed to {len(parts)} hull — its OBB "
                  f"will be far larger than the object")
        for prt in parts:
            v = onp.asarray(prt.vertices)
            lo, hi = v.min(0), v.max(0)
            if (hi - lo).min() < 1e-4:                            # degenerate sliver
                continue
            centres.append((lo + hi) / 2.0)
            extents.append(hi - lo)
    if not centres:
        return None
    centres, extents = onp.asarray(centres), onp.asarray(extents)
    # Prune to what the body can plausibly reach. This is not an optimisation nicety: the residual is
    # (frames x capsules x boxes) and jaxls differentiates through all of it, so the unpruned 576
    # boxes asked for 40 GB against a 27 GB device and the solve died. Distance is measured from the
    # HUMAN body-keypoint cloud (the robot tracks it) to each box's surface, so a box is kept only if
    # some body point passes within `reach` of it. At 0.3 m this keeps 179 of 576 — the cutting board
    # and counter the hands work on — and drops the fridge/microwave the robot never approaches.
    # The budget is tight because jaxls differentiates the whole (frames x capsules x boxes)
    # block: 576 boxes needed 40 GB, 180 needed 21.6 GB, against a 32 GB device. 64 boxes at
    # 0.2 m is the cutting board + countertop, which is exactly where the left hand sinks in;
    # the legs that a larger set would also constrain were measured to be FALSE positives of
    # the OBB slack, and grounding is already owned by the floor heightmap.
    kp = jp_cloud.reshape(-1, 3)
    gap = onp.abs(kp[:, None, :] - centres[None, :, :]) - extents[None, :, :] / 2.0
    d = onp.linalg.norm(onp.clip(gap, 0.0, None), axis=-1).min(0)         # (K,) surface distance
    m = d < reach
    if m.sum() > max_boxes:                                              # keep the closest
        m = onp.zeros_like(m); m[onp.argsort(d)[:max_boxes]] = True
    centres, extents = centres[m], extents[m]
    print(f"[pyroki-retarget] scene collision: {len(centres)} boxes (of {len(d)}) from {sorted(keep)} "
          f"(kept within {reach} m of the body keypoints)")
    return pk.collision.Box.from_extent(
        extent=jnp.array(extents, jnp.float32),
        position=jnp.array(centres, jnp.float32),
        wxyz=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0], jnp.float32), (len(centres), 1)))
# [/SCENE-COLLISION] --------------------------------------------------------------------------


# [OBJECT-COLLISION] ---------------------------------------------------------------------------
def _object_boxes(obj_name, max_boxes=int(os.environ.get("W_OBJCOLLMAXBOX", 64))):
    """Boxes approximating the MANIPULATED object, expressed in the OBJECT'S OWN frame.

    The manipulated object was never in the retarget problem — only the floor (and optionally the
    context furniture). Nothing stopped the solve from putting a finger INSIDE the knife, and it did:
    measured on the reference poses of s101_seg12_knife, `robot0_r_ffmiddle` sits inside the knife
    mesh in 60.9% of frames at 4.8 mm mean depth (every other right-hand link is under 1%). The sim
    then has to resolve that overlap at episode start, which is the "hand stuck in the object" the
    training runs show.

    Local frame, not world: unlike the context furniture this object MOVES every frame, so the boxes
    are built once here in the object's own frame and transformed per frame inside the cost, where
    the frame index is already the vmapped axis.

    Same geometry recipe as `_context_boxes` (convex-decompose, then take each hull's box) for the
    same reason — one box per object is far larger than the object, and pyroki has no hull primitive.
    The knife decomposes to 64 hulls of 8.5 mm median thickness, so the per-hull boxes stay thin
    enough that a fingertip can rest ON the blade without the box claiming it is inside.
    """
    src = _RAW_SCAN / obj_name / "simplified" / "base.obj"
    if not src.exists():
        print(f"[pyroki-retarget] object collision: no scan mesh for {obj_name}, term disabled")
        return None
    mesh = trimesh.load(str(src), process=False, force="mesh")
    parts = mesh.convex_decomposition()                       # pyVHACD; takes no kwargs
    parts = parts if isinstance(parts, list) else [parts]
    centres, extents, vols = [], [], []
    for prt in parts:
        v = onp.asarray(prt.vertices)
        lo, hi = v.min(0), v.max(0)
        if (hi - lo).min() < 1e-4:                            # degenerate sliver
            continue
        centres.append((lo + hi) / 2.0)
        extents.append(hi - lo)
        vols.append(float(onp.prod(hi - lo)))
    if not centres:
        return None
    centres, extents, vols = onp.asarray(centres), onp.asarray(extents), onp.asarray(vols)
    if len(centres) > max_boxes:                              # keep the bulkiest hulls
        k = onp.argsort(-vols)[:max_boxes]
        centres, extents = centres[k], extents[k]
    inflate = float(extents.prod(1).sum() / max(sum(p.volume for p in parts), 1e-9))
    print(f"[pyroki-retarget] object collision: {len(centres)} boxes for '{obj_name}' "
          f"(hulls {len(parts)}, box volume {inflate:.1f}x the hulls')")
    return onp.asarray(centres, onp.float32), onp.asarray(extents, onp.float32)
# [/OBJECT-COLLISION] --------------------------------------------------------------------------


def _give_hands_collision_geometry(urdf):
    """Copy each hand link's VISUAL mesh into its (empty) collision slot, in place.

    Without this the object-collision term below is aimed at nothing. The Shadow hand links in this
    URDF carry no <collision> element at all, so pyroki fits them a ZERO-radius capsule — measured:
    every `robot0_*` finger link has radius 0.0 mm, and the whole hand is represented by a single
    65.2 mm-radius capsule on the palm (whose real thickness is 34 mm). Constraining that stand-in
    made penetration WORSE, not better: it pushed the oversized palm ball clear of the knife and
    drove the index finger — which the problem could not see — 4.6 mm -> 6.0 mm deeper.

    The palm's own two boxes are left alone; its capsule is a poor fit either way (a capsule around
    a flat palm is inflated no matter what it is fitted to), and the fingers are what penetrate.
    """
    n = 0
    for name, link in urdf.link_map.items():
        if not name.startswith("robot0_") or link.collisions or not link.visuals:
            continue
        for v in link.visuals:
            if v.geometry is None or v.geometry.mesh is None:
                continue
            link.collisions.append(yourdfpy.Collision(name=f"{name}_coll", origin=v.origin,
                                                      geometry=v.geometry))
            n += 1
    print(f"[pyroki-retarget] hand collision: filled {n} empty collision slots from the visual meshes")
    return urdf


def solve(robot, robot_coll, heightmap, scene_boxes, keypoints, b_para, b_link, b_mask, a_para, a_link, gw, lw,
          l_contact, r_contact, l_foot_kp, r_foot_kp, left_foot_idx, right_foot_idx,
          left_knee_idx, right_knee_idx, root_R_target, root_z_target, ft_idx, ft_off, ft_margin, ft_target,
          ft_mask, rest_w, weights,
          obj_boxes_local=None, obj_pose=None, obj_capsule_idx=None,
          s2_joints=None, s2_root=None, s2_offset=None, s2_lower_mask=None, s2_w=0.0):
    # STAGE 2 (s2_w>0): freeze the LOWER body (s2_lower_mask=1 joints) + root + offset at the stage-1
    # solution (s2_joints/s2_root/s2_offset) and warm-start from it, so only the UPPER body (waist+arms+
    # hands) moves to reach the hand keypoints — fixes the embodiment "hands below" without un-grounding.
    T = keypoints.shape[0]
    nb = len(b_para)                              # local-alignment set = BODY only (small NxN + scale)
    b_para = jnp.array(b_para); b_link = jnp.array(b_link)
    a_para = jnp.array(a_para); a_link = jnp.array(a_link)   # global-alignment set = body + hands

    class ScaleVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((nb, nb))): ...
    class OffsetVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros((3,))): ...

    var_joints = robot.joint_var_cls(jnp.arange(T))
    var_root = jaxls.SE3Var(jnp.arange(T))
    var_scale = ScaleVar(jnp.zeros(T))
    var_offset = OffsetVar(jnp.arange(T))         # per-frame world translation (feet-on-floor placement)

    # per-node local weight `lw` (nb,) → pairwise lw[i]*lw[j]: weakening the ARM nodes frees the arm's
    # relative structure so the contact cost can bend it UP to reach the (absolute) object once the body
    # is grounded lower; legs/torso keep full local (natural grounded pose + proportion handling).
    lw_pair = lw[:, None] * lw[None, :]

    @jaxls.Cost.factory
    def local_align(vv, v_root: jaxls.SE3Var, v_cfg, v_scale: ScaleVar, kp):
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        s_pos = kp[b_para]; r_pos = T_wl.translation()[b_link]
        d_s = s_pos[:, None] - s_pos[None, :]; d_r = r_pos[:, None] - r_pos[None, :]
        scale = vv[v_scale][..., None]
        pw = (1 - jnp.eye(nb)) * b_mask * lw_pair
        res_pos = (d_s - d_r * scale) * pw[..., None]
        ds_n = d_s / jnp.linalg.norm(d_s + 1e-6, axis=-1, keepdims=True)
        dr_n = d_r / jnp.linalg.norm(d_r + 1e-6, axis=-1, keepdims=True)
        res_ang = (1 - (ds_n * dr_n).sum(-1)) * pw
        return jnp.concatenate([res_pos.flatten(), res_ang.flatten()]) * weights["local_alignment"]

    @jaxls.Cost.factory
    def scale_reg(vv, v_scale: ScaleVar):
        s = vv[v_scale]
        return jnp.concatenate([(s - 1.0).flatten() * 1.0, (s - s.T).flatten() * 100.0,
                                jnp.clip(-s, min=0).flatten() * 100.0])

    # per-correspondence, PER-AXIS global weight `gw` (n_corr,3) built in main by body part: the only
    # ABSOLUTE-position anchors are hands (object grasp) + pelvis XY + feet (floor); torso/arm/leg and
    # pelvis-Z are weak/free so LOCAL (relative, scaled) align preserves the motion shape while the legs
    # ground and the arms bend to reach the object (pyroki's joint-relationship philosophy + our grasp).
    @jaxls.Cost.factory
    def global_align(vv, v_root: jaxls.SE3Var, v_cfg, kp):
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        return ((T_wl.translation()[a_link] - kp[a_para]) * gw).flatten()

    @jaxls.Cost.factory
    def floor_contact(vv, v_root: jaxls.SE3Var, v_cfg, v_off: OffsetVar, lc, rc, lkp, rkp):
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        off = vv[v_off]
        lpos = T_wl.translation()[left_foot_idx] + off; rpos = T_wl.translation()[right_foot_idx] + off
        lz = T_wl.rotation().as_matrix()[left_foot_idx][2, 2]
        rz = T_wl.rotation().as_matrix()[right_foot_idx][2, 2]
        return jnp.concatenate([
            (lc * (lpos - lkp)).flatten(), (rc * (rpos - rkp)).flatten(),
            jnp.where(lc > 0.5, lz - 1, 0.0).flatten(), jnp.where(rc > 0.5, rz - 1, 0.0).flatten(),
        ]) * weights["floor_contact"]

    @jaxls.Cost.factory
    def contact_grasp(vv, v_root: jaxls.SE3Var, v_cfg, tgt, msk):
        # Pull each in-contact robot HAND POINT onto its object contact target, gated by msk so free/
        # reaching parts are untouched, with a per-point margin (so it stops once close). Points are the
        # full grasp set: 10 fingertip PADS (distal FK ∘ local pad offset → human fingertip pad) + the
        # wrap links palm/proximal/middle (link ORIGIN, offset 0 → object-surface contact from the hand
        # mesh). Together they reproduce a power grasp (fingers wrap + thumb opposes + palm), not a pinch.
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        pt = jax.vmap(lambda i, o: T_wl.translation()[i]
                      + T_wl.rotation().as_matrix()[i] @ o)(ft_idx, ft_off)     # (P,3) world contact point
        res = jnp.maximum(jnp.abs(pt - tgt) - ft_margin[:, None], 0.0)          # (P,3)
        return (res * msk[:, None]).flatten() * weights["contact"]

    @jaxls.Cost.factory
    def knee_separation(vv, v_root: jaxls.SE3Var, v_cfg):
        # Paper: absolute-position matching pulls the legs/knees together (narrow, unnatural stance).
        # Penalize the two knees getting closer than knee_min → keeps a natural stance width.
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        d = jnp.linalg.norm(T_wl.translation()[left_knee_idx] - T_wl.translation()[right_knee_idx] + 1e-6)
        return jnp.maximum(weights["knee_min"] - d, 0.0).reshape(1) * weights["knee_separation"]

    @jaxls.Cost.factory
    def root_orient(vv, v_root: jaxls.SE3Var, R_tgt):
        # Pull the free root orientation to the keypoint-derived pelvis frame → kills the spurious
        # roll that otherwise saturates hip_roll/waist_roll (the visible pelvis twist).
        R_rob = vv[v_root].rotation()
        return (R_rob.inverse() @ jaxlie.SO3.from_matrix(R_tgt)).log().flatten() * weights["root_orientation"]

    @jaxls.Cost.factory
    def root_height(vv, v_root: jaxls.SE3Var, z_tgt):
        # Track pelvis HEIGHT at human_pelvis_z × leg_ratio (per frame): grounds the feet across ALL
        # motions (squat/step/bob preserved proportionally) while keeping the body upright — no crouch
        # collapse (free-z failure) and no floating (human-height tracking failure). z_tgt: scalar/frame.
        return ((vv[v_root].translation()[2] - z_tgt) * weights["root_height"]).reshape(1)

    @jaxls.Cost.factory
    def root_smooth(vv, v_root: jaxls.SE3Var, v_prev: jaxls.SE3Var):
        return (vv[v_root].inverse() @ vv[v_prev]).log().flatten() * weights["root_smoothness"]

    @jaxls.Cost.factory
    def skating(vv, v_root: jaxls.SE3Var, v_cfg, v_off: OffsetVar,
                v_root_p: jaxls.SE3Var, v_cfg_p, v_off_p: OffsetVar, lc, rc):
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        T_wl_p = vv[v_root_p] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg_p]))
        off = vv[v_off]; off_p = vv[v_off_p]
        lsk = lc * ((T_wl.translation()[left_foot_idx] + off) - (T_wl_p.translation()[left_foot_idx] + off_p))
        rsk = rc * ((T_wl.translation()[right_foot_idx] + off) - (T_wl_p.translation()[right_foot_idx] + off_p))
        return jnp.stack([lsk, rsk]) * weights["foot_skating"]

    @jaxls.Cost.factory
    def offset_reg(vv, v_off: OffsetVar):
        # Pin the HORIZONTAL offset (xy) strongly so the free offset can't drift the whole robot off the
        # keypoints (root xy already places the robot horizontally); leave Z governed by offset_reg so
        # world_collision can lift the feet onto the floor (per-pose grounding, squat-preserving) when
        # offset_reg (z) is small. offset_xy default high; set offset_reg=0 to free vertical grounding.
        o = vv[v_off]
        return jnp.concatenate([(o[..., :2] * weights["offset_xy"]).flatten(),
                                (o[..., 2:3] * weights["offset_reg"]).flatten()])

    @jaxls.Cost.factory
    def world_collision(vv, v_root: jaxls.SE3Var, v_cfg, v_off: OffsetVar):
        # Low weight: high enough to lift the robot up off the floor, low enough not to fight retargeting.
        transform = jaxlie.SE3.from_translation(vv[v_off]) @ vv[v_root]
        coll = robot_coll.at_config(robot, vv[v_cfg]).transform(transform)
        res = [colldist_from_sdf(collide(coll, heightmap), activation_dist=0.005).flatten()]
        # [SCENE-COLLISION] the context objects the env spawns kinematic. Without this term they are
        # absent from the problem entirely, and the solve puts the left hand inside the countertop.
        # coll is (T,B) capsules, scene_boxes is (K,) — broadcast to (T,B,K) via a trailing axis.
        if scene_boxes is not None:
            d = collide(coll.reshape(coll.get_batch_axes() + (1,)), scene_boxes)
            res.append(colldist_from_sdf(d, activation_dist=0.005).flatten()
                       * weights["scene_collision"])
        return jnp.concatenate(res) * weights["world_collision"]

    # [OBJECT-COLLISION] the object the hand is grasping. `o_pose` is (7,) per frame (wxyz + xyz), so
    # the boxes are rebuilt in world coordinates inside the vmapped frame axis.
    if obj_boxes_local is not None:
        _oc_ctr = jnp.asarray(obj_boxes_local[0])                      # (K,3) object frame
        _oc_ext = jnp.asarray(obj_boxes_local[1])                      # (K,3)
        _oc_idx = jnp.asarray(obj_capsule_idx)                         # capsules we constrain
        _oc_tol = weights["object_collision_tol"]

    @jaxls.Cost.factory
    def object_collision(vv, v_root: jaxls.SE3Var, v_cfg, v_off: OffsetVar, o_pose):
        transform = jaxlie.SE3.from_translation(vv[v_off]) @ vv[v_root]
        coll = robot_coll.at_config(robot, vv[v_cfg]).transform(transform)
        # Only the hand capsules are constrained. The residual is (capsules x boxes) per frame and
        # jaxls differentiates all of it (the scene term needed 21.6 GB at 78 capsules x 180 boxes);
        # nothing but a hand is ever inside a hand-held object, so the rest is memory for no term.
        coll = jax.tree.map(lambda x: x[_oc_idx] if getattr(x, "ndim", 0) >= 1
                            and x.shape[0] == robot_coll.num_links else x, coll)
        o_wxyz, o_xyz = o_pose[:4], o_pose[4:]
        R = jaxlie.SO3(o_wxyz).as_matrix()                             # object frame -> world
        boxes = pk.collision.Box.from_extent(
            extent=_oc_ext, position=_oc_ctr @ R.T + o_xyz,
            wxyz=jnp.broadcast_to(o_wxyz, (_oc_ctr.shape[0], 4)))
        d = collide(coll.reshape(coll.get_batch_axes() + (1,)), boxes)
        # One-sided: `colldist_from_sdf(., 0)` is min(d, 0), so a link that merely TOUCHES the object
        # costs exactly nothing and only overlap is charged. That is what lets this term carry a big
        # weight — the scene-collision term had to stay at 1.0 because its 5 mm activation margin
        # pushed on hands that were correctly resting on the surface, and it lost to the keypoint
        # costs anyway. `tol` is the overlap we are willing to leave (the sim resolves that much
        # itself through its contact offset).
        return colldist_from_sdf(d + _oc_tol, activation_dist=0.0).flatten() * weights["object_collision"]
    # [/OBJECT-COLLISION]

    costs = [
        local_align(var_root, var_joints, var_scale, keypoints),
        scale_reg(var_scale),
        global_align(var_root, var_joints, keypoints),
        floor_contact(var_root, var_joints, var_offset, l_contact, r_contact, l_foot_kp, r_foot_kp),
        root_orient(var_root, root_R_target),
        root_height(var_root, root_z_target),
        knee_separation(var_root, var_joints),
        contact_grasp(var_root, var_joints, ft_target, ft_mask),
        root_smooth(jaxls.SE3Var(jnp.arange(1, T)), jaxls.SE3Var(jnp.arange(0, T - 1))),
        skating(jaxls.SE3Var(jnp.arange(1, T)), robot.joint_var_cls(jnp.arange(1, T)),
                OffsetVar(jnp.arange(1, T)), jaxls.SE3Var(jnp.arange(0, T - 1)),
                robot.joint_var_cls(jnp.arange(0, T - 1)), OffsetVar(jnp.arange(0, T - 1)),
                l_contact[:-1], r_contact[:-1]),
        offset_reg(var_offset),
        pk.costs.smoothness_cost(robot.joint_var_cls(jnp.arange(1, T)),
                                 robot.joint_var_cls(jnp.arange(0, T - 1)),
                                 jnp.array([weights["joint_smoothness"]])),
        pk.costs.rest_cost(var_joints, var_joints.default_factory()[None], rest_w[None]),
        # NOTE: self_collision_cost dropped for the 65-DOF composite — 2926 pairs × 151 frames
        # blows up to ~42 GB / int32-overflow. Re-add with a restricted pair set (hand↔body only) later.
        pk.costs.limit_constraint(jax.tree.map(lambda x: x[None], robot), var_joints),
    ]
    # world_collision + var_offset (feet-on-floor) are ON by default. Naively grounding a single-stage solve
    # drops the whole robot ~10 cm to plant feet and corrupts body+hand keypoint tracking 2–3× (8→19 cm) for
    # our tall-human→short-G1 embodiment — but STAGE 2 resolves this: stage 1 grounds the lower body, then the
    # frozen-legs upper-body re-solve re-reaches the hand keypoints, so grounding no longer costs hand accuracy.
    # offset_xy pins the horizontal offset (no drift); offset_reg (z) is free so feet plant. W_WORLDCOLL=0 →
    # faithful-tracking baseline (no grounding). To disable grounding entirely: W_WORLDCOLL=0.0.
    if weights["world_collision"] > 0:
        costs.append(world_collision(var_root, var_joints, var_offset))
    if obj_boxes_local is not None and weights["object_collision"] > 0:
        costs.append(object_collision(var_root, var_joints, var_offset, jnp.asarray(obj_pose)))

    init_vals = None
    if s2_w > 0.0:
        # STAGE 2: strong pin holding the lower body (mask) + root + offset at the stage-1 pose; the
        # up-weighted hand global_align (gw hands) then drives the free UPPER body to the hand keypoints.
        _pin_j = jnp.asarray(s2_joints)                       # (T, nJ)
        _pin_root = jnp.asarray(s2_root)                      # (T, 7) wxyz_xyz (stage-1 baked root)
        _pin_off = jnp.asarray(s2_offset)                     # (T, 3)
        _lmask = jnp.asarray(s2_lower_mask)                   # (nJ,) 1=freeze, 0=free

        @jaxls.Cost.factory
        def stage2_pin(vv, v_cfg, v_root: jaxls.SE3Var, v_off: OffsetVar, pin_j, pin_root_arr, pin_off):
            dj = (vv[v_cfg] - pin_j) * _lmask                 # freeze lower-body joints
            droot = (vv[v_root].inverse() @ jaxlie.SE3(pin_root_arr)).log()   # freeze root
            doff = vv[v_off] - pin_off                        # freeze offset
            return jnp.concatenate([dj.flatten(), droot.flatten(), doff.flatten()]) * s2_w

        costs.append(stage2_pin(var_joints, var_root, var_offset, _pin_j, _pin_root, _pin_off))
        init_vals = jaxls.VarValues.make([
            var_joints.with_value(_pin_j),
            var_root.with_value(jaxlie.SE3(_pin_root)),
            var_scale.with_value(jnp.ones((T, nb, nb))),
            var_offset.with_value(_pin_off),
        ])

    prob = jaxls.LeastSquaresProblem(
        costs=costs, variables=[var_joints, var_root, var_scale, var_offset]).analyze()
    sol = prob.solve() if init_vals is None else prob.solve(initial_vals=init_vals)
    root = jaxlie.SE3.from_translation(sol[var_offset]) @ sol[var_root]   # bake floor offset into root
    return root, sol[var_joints]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="s100_seg00_pan")
    ap.add_argument("--class", dest="cls", default="single_rigid")
    args = ap.parse_args()

    urdf = yourdfpy.URDF.load(str(_URDF))
    robot = pk.Robot.from_urdf(urdf)
    # [OBJECT-COLLISION] the hand needs real collision shapes before anything can be kept out of the
    # object; only pay for them when a term actually uses them.
    if float(os.environ.get("W_OBJCOLL", 0.0)) > 0:
        _give_hands_collision_geometry(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    sm = onp.load(_PROC / "smplx" / args.cls / args.clip / "0" / "trajectory.npz", allow_pickle=True)
    jp = sm["joint_positions"].astype(onp.float32)
    F = jp.shape[0]

    pairs = _build_correspondence()
    a_para = [p for p, _ in pairs]                                   # global-align: body + hands (54)
    a_link = [robot.links.names.index(l) for _, l in pairs]
    b_para = [p for p, _ in _BODY]                                   # local-align + scale: body only (14)
    b_link = [robot.links.names.index(l) for _, l in _BODY]
    print(f"[pyroki-retarget] clip={args.clip} F={F} correspondences: global {len(pairs)} "
          f"(body {len(_BODY)} + hands {len(pairs)-len(_BODY)}), local+scale {len(_BODY)} (body)")
    b_mask = create_conn_tree(robot, jnp.array(b_link))

    l_c = _foot_contact(jp, _PARA_BALL_L); r_c = _foot_contact(jp, _PARA_BALL_R)
    l_kp = jp[:, _PARA_BALL_L, :].copy(); l_kp[:, 2] = _ANKLE_SOLE_OFF   # target ANKLE at sole-on-floor height
    r_kp = jp[:, _PARA_BALL_R, :].copy(); r_kp[:, 2] = _ANKLE_SOLE_OFF
    left_foot_idx = robot.links.names.index("left_ankle_roll_link")
    right_foot_idx = robot.links.names.index("right_ankle_roll_link")
    left_knee_idx = robot.links.names.index("left_knee_link")
    right_knee_idx = robot.links.names.index("right_knee_link")

    # object-contact grasp point set (concatenated, all fed through one contact cost):
    #  (A) 10 fingertip PADS: distal link + local pad offset → human fingertip pad, gated by mesh proximity.
    #  (B) wrap links (palm + proximal + middle per finger): link ORIGIN → object-surface contact target
    #      from the full hand-mesh↔object precompute (hand_contact.npz), so the grasp WRAPS (not a pinch).
    ft_idx = [robot.links.names.index(f"robot0_{s}_{f}distal") for _, s, f in _FT_PADS]
    ft_off = [(_FT_OFF_R[f] if not (s == "l" and f != "th")
               else [_FT_OFF_R[f][0], -_FT_OFF_R[f][1], _FT_OFF_R[f][2]]) for _, s, f in _FT_PADS]
    ft_pad = sm["fingertip_pad_pos"].astype(onp.float32)                 # (F,10,3) human pads (fallback target)
    ft_margin = [0.005] * len(_FT_PADS)                                  # tips: tight (precise pad)
    bk = [k for k in sm.files if k.startswith("obj__") and k.endswith("__base")]
    obj_name = bk[0].split("__")[1] if bk else ""

    # Load the Option-A per-link contact map ONCE — used for BOTH the fingertips (distal) and the wrap
    # links, so the retarget's contact source matches the RL env exactly (single hand_contact.npz).
    hc_path = _PROC / "smplx" / args.cls / args.clip / "0" / "hand_contact.npz"
    hc = onp.load(hc_path, allow_pickle=True) if hc_path.exists() else None
    hc_names = [str(n) for n in hc["link_names"]] if hc is not None else []

    # fingertip (distal) contact target + mask: UNIFIED with the env → the Option-A object-SURFACE target +
    # mask from hand_contact.npz (same source as the wrap links), NOT the off-surface human pad. The human
    # pad (fingertip_pad_pos) sits ~1.7 cm off a thin handle, which left the retargeted fingers floating;
    # snapping to the Option-A surface point puts the fingertips ON the object. Fallback = human pad + gate.
    if hc is not None:
        _di = [hc_names.index(f"robot0_{s}_{f}distal") for _, s, f in _FT_PADS]   # distal col per fingertip
        ft_target = hc["target"][:, _di].astype(onp.float32)             # (F,10,3) object surface (Option A)
        ft_mask = hc["mask"][:, _di].astype(onp.float32)                 # (F,10) Option-A distal mask
    else:
        ft_target = ft_pad                                               # fallback: human pad
        ft_mask = (_contact_signal(ft_pad.astype(onp.float64), sm[bk[0]].astype(onp.float64), obj_name)
                   if obj_name else onp.zeros((F, 10), onp.float32))
    c_idx, c_off, c_margin = list(ft_idx), list(ft_off), list(ft_margin)
    c_target = [ft_target]; c_mask = [ft_mask]                           # lists of (F,·) blocks, concat later
    c_group = ["tip"] * len(ft_idx)                                      # [DIAGNOSTIC] per-point group: tip/palm/finger

    if hc is not None:
        wrap = [n for n in hc_names if not n.endswith("distal")]        # palm + proximal + middle (tips above)
        _woff = float(os.environ.get("W_WRAP_OFFSET", "0.0"))           # [EXPERIMENT] palmar surface offset (m)
        _wmarg = float(os.environ.get("W_WRAP_MARGIN", "0.012"))        # [EXPERIMENT] wrap contact margin (m)
        for n in wrap:
            j = hc_names.index(n)
            c_idx.append(robot.links.names.index(n)); c_off.append(_wrap_offset(n, _woff))
            c_margin.append(_wmarg)                                     # wrap: link-radius margin (origin→surface)
            c_target.append(hc["target"][:, j][:, None, :])             # (F,1,3)
            c_mask.append(hc["mask"][:, j][:, None])                    # (F,1)
            c_group.append("palm" if n.endswith("palm") else "finger")  # palm vs finger proximal/middle
        print(f"[pyroki-retarget] contact points: {len(_FT_PADS)} fingertip pads (Option-A surface) "
              f"+ {len(wrap)} wrap links (W_WRAP_OFFSET={_woff} W_WRAP_MARGIN={_wmarg})")
    else:
        print("[pyroki-retarget] hand_contact.npz absent → fingertip human-pad contact only "
              "(run parahome_hand_contact.py for full hand-mesh grasp)")
    ft_idx = onp.array(c_idx, onp.int32)
    ft_off = onp.array(c_off, onp.float32)
    ft_margin = onp.array(c_margin, onp.float32)
    ft_pad = onp.concatenate(c_target, axis=1).astype(onp.float32)      # (F,P,3)
    ft_mask = onp.concatenate(c_mask, axis=1).astype(onp.float32)       # (F,P)

    # rest weights: default 0.2, move-less joints 2.0, coupled-J0 5.0 (hold ~0 → solve≈output)
    an = robot.joints.actuated_names
    rest_w = onp.full(len(an), 0.2, onp.float32)
    for nm in _MOVE_LESS:
        if nm in an: rest_w[an.index(nm)] = 2.0
    for nm in _HOLD_ZERO:
        if nm in an: rest_w[an.index(nm)] = 5.0

    _w = lambda k, d: float(os.environ.get(k, d))
    # DEFAULTS = the VALIDATED production recipe (feet grounded + 2-stage upper-body hand reach). Feet are
    # planted onto a flat floor (world_collision + free offset-z), the lower body is frozen after stage 1 and
    # the arms/hands re-reach the human hand keypoints in stage 2, and contact weight is low (2.0) so the hand
    # keypoint tracking wins over object-surface pull (validated: left thumb ~0.12, hand mean ~0.04 on pan).
    # Every weight stays env-overridable for experiments (e.g. W_STAGE2=0 → single-stage; W_WORLDCOLL=0 →
    # no grounding / faithful-tracking baseline). See the sweep notes for the local/contact trade-off.
    weights = dict(local_alignment=_w("W_LOCAL", 2.0), global_alignment=_w("W_GLOBAL", 1.0),
                   hand_alignment=_w("W_HAND", 1.0), floor_contact=_w("W_FLOOR", 3.0),
                   world_collision=_w("W_WORLDCOLL", 1.0), root_orientation=_w("W_ROOTORI", 5.0),
                   root_smoothness=_w("W_ROOTSMOOTH", 1.0), foot_skating=_w("W_SKATE", 1.0),
                   offset_reg=_w("W_OFFSETREG", 0.0), offset_xy=_w("W_OFFSETXY", 1000.0),
                   contact=_w("W_CONTACT", 2.0), contact_margin=_w("W_CONTACTMARGIN", 0.005),
                   knee_separation=_w("W_KNEESEP", 5.0), knee_min=_w("W_KNEEMIN", 0.14),
                   root_height=_w("W_ROOTHEIGHT", 0.0), joint_smoothness=_w("W_SMOOTH", 1.0),
                   # [SCENE-COLLISION] OFF by default. Tried on s101_seg12_knife: the term is added and the solve
                   # converges, but it does not move the result — penetration against its OWN box set went
                   # 1813 -> 1963 pairs (6.03 -> 6.38 cm), and in sim the left hand got WORSE (5.47 -> 6.66
                   # m/s). It is outweighed by the keypoint-tracking costs, which target human hand
                   # positions that sit ON the surface, so matching them necessarily buries the thicker
                   # robot hand. Raising the weight or changing the box set does not address that.
                   # W_SCENECOLL=1 re-enables it; see _context_boxes for the geometry study.
                   scene_collision=_w("W_SCENECOLL", 0.0),
                   # [OBJECT-COLLISION] OFF by default until measured; W_OBJCOLL=50 to enable. The
                   # weight can be large because the residual is one-sided (0 unless links overlap
                   # the object), so at the pose we want the term contributes nothing at all.
                   object_collision=_w("W_OBJCOLL", 0.0),
                   object_collision_tol=_w("W_OBJCOLLTOL", 0.002))

    # per-axis, per-body-part global weights (n_corr,3): body order = pelvis(0) torso(1) arms(2-7)
    # legs(8-13) hands(14+). Default = uniform full tracking; the low-weight decoupling is opt-in via env.
    w_pxy, w_pz = _w("W_PELVISXY", 1.0), _w("W_PELVISZ", 0.0)   # pelvis-Z tracked by root_height (scaled), not global
    w_trunk, w_arm, w_leg = _w("W_TRUNK", 1.0), _w("W_ARM", 1.0), _w("W_LEG", 1.0)
    gw = onp.zeros((len(a_para), 3), onp.float32)
    gw[0] = [w_pxy, w_pxy, w_pz]                        # pelvis: xy anchor, z free (grounding sets height)
    gw[1] = w_trunk * weights["global_alignment"]       # torso
    gw[2:8] = w_arm * weights["global_alignment"]       # arms — weak (free to reach the object)
    gw[8:14] = w_leg * weights["global_alignment"]      # legs — weak (grounded by floor + local, not absolute)
    gw[14:] = weights["hand_alignment"]                 # hands — strong absolute (object grasp)

    # per-node LOCAL weight (nb=14 body): arms weak so the arm's relative structure yields to the contact
    # cost (reach the object from the grounded-lower body); pelvis/torso/legs keep full local.
    w_armlocal = _w("W_ARMLOCAL", 1.0)                   # default full; lower (env) frees the arm for grounding
    lw = onp.ones(len(b_para), onp.float32)
    lw[2:8] = w_armlocal                                 # _BODY idx 2-7 = r/l shoulder,elbow,wrist

    root_R_target = _pelvis_target_R(jp)     # (F,3,3) keypoint-derived pelvis orientation target
    # scaled pelvis-height target: human pelvis z × leg_ratio (per frame). Grounds feet (feet_z≈0 scales to
    # ~0) while preserving vertical dynamics (squat/step/bob) — a per-frame track, NOT a fixed height.
    leg_ratio = _w("W_LEGRATIO", 0.86)
    root_z_tgt = (jp[:, 0, 2] * leg_ratio).astype(onp.float32)     # (F,) pelvis keypoint z × ratio
    heightmap = _flat_heightmap(jp)          # flat z=0 floor spanning the clip
    scene_boxes = (_context_boxes(sm, bk[0], jp[:, :23, :]) if weights["scene_collision"] > 0 else None)

    # [OBJECT-COLLISION] keep the hands out of the object they are grasping. The stored object pose is
    # [xyz, wxyz]; the solve uses jaxlie's wxyz-then-xyz order everywhere else, so reorder here.
    obj_boxes_local, obj_pose, obj_caps = None, None, None
    if weights["object_collision"] > 0 and obj_name:
        obj_boxes_local = _object_boxes(obj_name)
        if obj_boxes_local is not None:
            op = sm[bk[0]].astype(onp.float32)                      # (F,7) xyz + wxyz
            obj_pose = onp.concatenate([op[:, 3:7], op[:, :3]], axis=1)
            g2l = onp.asarray(robot_coll._geom_to_link_idx)
            lnames = list(robot_coll.link_names)
            # The palm is EXCLUDED by default. Its capsule has a 65.2 mm radius against a real palm
            # thickness of 34 mm, so keeping that ball out of the knife shoves the whole hand away —
            # and the palm barely penetrates to begin with (measured 6.3% of frames at 1.8 mm, versus
            # 89% at 4.7 mm for the index finger). Constraining the well-fitted finger capsules and
            # leaving the palm out buys the penetration fix at a far smaller tracking cost.
            # W_OBJCOLLSKIP="" constrains every hand link including the palm.
            skip = os.environ.get("W_OBJCOLLSKIP", "palm")
            skip_parts = [s for s in skip.split(",") if s]
            obj_caps = onp.asarray([i for i, g in enumerate(g2l)
                                    if lnames[g].startswith("robot0_")
                                    and not any(s in lnames[g] for s in skip_parts)], onp.int32)
            print(f"[pyroki-retarget] object collision: {len(obj_caps)} hand capsules "
                  f"x {len(obj_boxes_local[0])} boxes x {len(obj_pose)} frames, "
                  f"weight {weights['object_collision']}, tolerance "
                  f"{weights['object_collision_tol'] * 1000:.0f} mm")

    t0 = time.time()
    Ts_root, joints = solve(robot, robot_coll, heightmap, scene_boxes, jnp.array(jp), b_para, b_link, b_mask, a_para,
                            a_link, jnp.array(gw), jnp.array(lw), jnp.array(l_c), jnp.array(r_c),
                            jnp.array(l_kp), jnp.array(r_kp), left_foot_idx, right_foot_idx,
                            left_knee_idx, right_knee_idx,
                            jnp.array(root_R_target), jnp.array(root_z_tgt), jnp.array(ft_idx), jnp.array(ft_off),
                            jnp.array(ft_margin), jnp.array(ft_pad), jnp.array(ft_mask),
                            jnp.array(rest_w), weights,
                            obj_boxes_local=obj_boxes_local, obj_pose=obj_pose, obj_capsule_idx=obj_caps)
    joints = onp.array(joints); root = onp.array(Ts_root.wxyz_xyz)
    print(f"[pyroki-retarget] stage-1 solved in {time.time()-t0:.1f}s")

    # ---- STAGE 2 (W_STAGE2>0 = pin weight; ON by default): freeze LOWER body (legs) + root + offset at the
    # stage-1 solution and re-solve the UPPER body (waist+arms+hands) with STRONG hand keypoint alignment, so
    # the arms reach UP to the human hand keypoints without un-grounding the (stage-1) feet. W_STAGE2=0 → skip
    # (single-stage baseline). ----
    w_stage2 = _w("W_STAGE2", 100.0)
    if w_stage2 > 0.0:
        lower_mask = onp.array([1.0 if any(k in nm for k in ("hip", "knee", "ankle")) else 0.0
                                for nm in an], onp.float32)      # freeze legs; waist+arms+fingers stay free
        gw2 = gw.copy()
        gw2[0] = 0.0                                             # pelvis pinned → don't track
        gw2[8:14] = 0.0                                          # legs pinned → don't track
        gw2[2:8] = _w("W_STAGE2ARM", 1.0)                       # arms incl. WRIST_yaw_link (anchor hand base)
        gw2[14:] = _w("W_STAGE2HAND", 5.0)                      # hands STRONG (pull arms up to keypoints)
        s2_off = onp.zeros((F, 3), onp.float32)                 # stage-1 root already baked → pin offset→0
        t1 = time.time()
        Ts_root, joints2 = solve(robot, robot_coll, heightmap, scene_boxes, jnp.array(jp), b_para, b_link, b_mask, a_para,
                                 a_link, jnp.array(gw2), jnp.array(lw), jnp.array(l_c), jnp.array(r_c),
                                 jnp.array(l_kp), jnp.array(r_kp), left_foot_idx, right_foot_idx,
                                 left_knee_idx, right_knee_idx,
                                 jnp.array(root_R_target), jnp.array(root_z_tgt), jnp.array(ft_idx),
                                 jnp.array(ft_off), jnp.array(ft_margin), jnp.array(ft_pad), jnp.array(ft_mask),
                                 jnp.array(rest_w), weights,
                                 obj_boxes_local=obj_boxes_local, obj_pose=obj_pose, obj_capsule_idx=obj_caps,
                                 s2_joints=jnp.array(joints), s2_root=jnp.array(root),
                                 s2_offset=jnp.array(s2_off), s2_lower_mask=jnp.array(lower_mask), s2_w=w_stage2)
        joints = onp.array(joints2); root = onp.array(Ts_root.wxyz_xyz)
        print(f"[pyroki-retarget] STAGE 2 (freeze lower + reach hands) solved in {time.time()-t1:.1f}s")

    solved = {name: joints[:, i] for i, name in enumerate(an)}
    act = _ORDER["action_joint_names"]
    g1_joint_pos = onp.zeros((F, len(act)), onp.float32)
    nmap = 0
    for j, name in enumerate(act):
        if name in solved:
            g1_joint_pos[:, j] = solved[name]; nmap += 1
    g1_root_pose = onp.concatenate([root[:, 4:7], root[:, 0:4]], axis=1)

    # ---- post-solve contact residual [DIAGNOSTIC]: how far each in-contact robot contact point ends up
    # from its target after the solve. Tests whether the contact cost actually reaches its targets (small)
    # or is overridden by the keypoint anchors (large). Split fingertip pads vs wrap links (palm+base). ----
    try:
        _fk = jax.vmap(robot.forward_kinematics)(jnp.array(joints))              # (F, nlinks, 7) local wxyz_xyz
        _Tf = jaxlie.SE3(_fk)
        _lp = onp.array(_Tf.translation()); _lR = onp.array(_Tf.rotation().as_matrix())     # (F,nl,3),(F,nl,3,3)
        _Tr = jaxlie.SE3(jnp.array(root))
        _rR = onp.array(_Tr.rotation().as_matrix()); _rp = onp.array(_Tr.translation())     # (F,3,3),(F,3)
        _pos = onp.einsum("fij,flj->fli", _rR, _lp) + _rp[:, None, :]            # (F,nl,3) world link pos
        _Rm = onp.einsum("fij,fljk->flik", _rR, _lR)                            # (F,nl,3,3) world link rot
        _cp = _pos[:, ft_idx] + onp.einsum("flij,lj->fli", _Rm[:, ft_idx], ft_off)   # (F,P,3) robot contact pts
        _gap = onp.linalg.norm(_cp - ft_pad, axis=-1)                            # (F,P) to target
        _m = ft_mask > 0.5; _grp = onp.array(c_group)
        parts = []
        for g in ("palm", "finger", "tip"):
            sel = _grp == g
            gg = _gap[:, sel][_m[:, sel]]
            parts.append(f"{g} mean {gg.mean()*100:.1f} cm max {gg.max()*100:.1f} cm (n={gg.size})"
                         if gg.size else f"{g} none")
        print("[metric] post-solve contact gap (in-contact):  " + "  |  ".join(parts))
    except Exception as _e:
        print(f"[metric] skipped ({type(_e).__name__}: {_e})")

    out = _PROC / "g1_shadow" / args.cls / args.clip / "0" / f"trajectory_pyroki{os.environ.get('W_OUTSUFFIX','')}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Record the column layout with the data. g1_joint_pos is written in `act` order, and the env
    # reads column k into its OWN k-th action joint — an agreement nothing enforced, since `act`
    # comes from a static json dump of the robot's PhysX DOF order. Rebuilding G1_shadow.usd
    # repermutes the env side and silently crosses the columns (it did: 24 of 65, all hands, the
    # middle finger driven by the thumb). With the names alongside, the env matches by name.
    onp.savez(out, g1_joint_pos=g1_joint_pos.astype(onp.float32),
              g1_root_pose=g1_root_pose.astype(onp.float32),
              joint_names=onp.array(act, dtype=object))
    print(f"[pyroki-retarget] wrote {out}  ({nmap}/{len(act)} action joints solved)")


if __name__ == "__main__":
    main()
