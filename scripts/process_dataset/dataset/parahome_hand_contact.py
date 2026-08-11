"""Precompute HAND-MESH ↔ OBJECT contact for a ParaHome clip → hand_contact.npz (sidecar of trajectory.npz).

The fingertip-only contact used in retarget_g1_pyroki misses the palm + finger phalanges that WRAP the
object in a power grasp. This derives the wrap contact the DexMachina way (A.4 Contact Approximation,
MandiZhao/dexmachina retargeting/map_contacts.py), adapted to be INDEPENDENT of any robot-hand retargeting
(Option A — assign via the HUMAN hand, not the retargeted robot pose):

  1. SMPL-X FK → full hand-mesh vertices (world frame, same as joint_positions/fingertip_pad_pos).
  2. Static hand-vertex→robot-link map from the SMPL-X `lbs_weights` argmax (wrist→palm; fingerN joint
     1/2/3 → proximal/middle/distal). Computed ONCE, no robot pose needed.
  3. Per frame — CONTACTS LIVE ON THE OBJECT (DexMachina): for each OBJECT mesh vertex find its nearest
     HAND vertex; keep it as a contact if that distance < gamma. Farthest-point-subsample to <= num_contacts
     for spatial diversity. (Purely GEOMETRIC — no object-velocity gate, so static holds are captured too;
     re-enable the old gate with --use-velocity-gate.)
  4. Assign each object contact to a robot link via its NEAREST HAND vertex's link (step-2 map) — the
     robot-pose-free stand-in for DexMachina's nearest-robot-link assignment. Aggregate per link → mask (F,L)
     + target (F,L,3) = mean world position of that link's OBJECT contacts (on the object surface, i.e. WHERE
     the robot link should touch). The retarget pulls each in-contact robot link there.

Runs in env_isaaclab (needs smplx + torch + trimesh):
    /home/peunsu/anaconda3/envs/env_isaaclab/bin/python scripts/process_dataset/dataset/parahome_hand_contact.py \
        --clip s100_seg00_pan [--class single_rigid] [--gamma 0.015] [--num-contacts 50]
→ writes .../smplx/<class>/<clip>/0/hand_contact.npz  (link_names, mask, target)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

import smplx

_SCRIPT_DIR = Path(__file__).resolve().parents[3]
_DATA = _SCRIPT_DIR / "source" / "robotis_sh5" / "data"
_PROC = _DATA / "processed" / "parahome"
_RAW_SCAN = _DATA / "raw" / "parahome" / "data" / "scan"
_SMPLX_MODEL_DIR = _SCRIPT_DIR / "models_smplx_v1_1" / "models"

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_FPS = 30.0
_OBJ_LINVEL_TH, _OBJ_ANGVEL_TH = 0.05, 0.25


def _farthest_point_sample(pts: np.ndarray, n: int) -> np.ndarray:
    """Greedy farthest-point sampling → indices of n well-spread points (all if <= n). Deterministic
    (starts at index 0), matching DexMachina's spatial-diversity subsampling of the raw contacts."""
    m = len(pts)
    if m <= n:
        return np.arange(m)
    sel = np.zeros(n, np.int64)
    d = np.full(m, np.inf, np.float64)
    for i in range(1, n):
        d = np.minimum(d, np.linalg.norm(pts - pts[sel[i - 1]], axis=1))
        sel[i] = int(d.argmax())
    return sel

# SMPL-X hand joint id → Shadow link (per side). fingerN joint 1/2/3 → proximal/middle/distal.
# SMPL-X left-hand joint order: index(25-27) middle(28-30) pinky(31-33) ring(34-36) thumb(37-39); right +15.
_FINGERS = [("ff", 0), ("mf", 3), ("lf", 6), ("rf", 9), ("th", 12)]   # (shadow name, offset within hand)
_SEGS = ["proximal", "middle", "distal"]


def _joint2link():
    m = {20: "robot0_l_palm", 21: "robot0_r_palm"}
    for side, base in (("l", 25), ("r", 40)):
        for fg, off in _FINGERS:
            for s, seg in enumerate(_SEGS):
                m[base + off + s] = f"robot0_{side}_{fg}{seg}"
    return m


def _quat2R(wxyz):
    w, x, y, z = wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="s100_seg00_pan")
    ap.add_argument("--class", dest="cls", default="single_rigid")
    ap.add_argument("--gamma", type=float, default=0.015, help="object-vertex→nearest-hand-vertex contact dist (m)")
    ap.add_argument("--num-contacts", type=int, default=50, help="farthest-point-subsample cap per frame")
    ap.add_argument("--normal-source", choices=("surface", "to-hand"), default="surface",
                    help="접촉 방향을 물체 메시의 표면 법선에서 얻을지(surface, 기본), "
                         "표면점→손 정점 방향에서 얻을지(to-hand, 이전 동작).")
    ap.add_argument("--use-velocity-gate", action="store_true",
                    help="also require the object to be moving (old behavior); default OFF (DexMachina geometric-only)")
    args = ap.parse_args()

    clip_dir = _PROC / "smplx" / args.cls / args.clip
    ti = json.load(open(clip_dir / "task_info.json"))
    gender = str(ti.get("gender", "neutral"))
    sm = np.load(clip_dir / "0" / "trajectory.npz", allow_pickle=True)
    F = sm["smplx_body_pose"].shape[0]

    model = smplx.create(str(_SMPLX_MODEL_DIR), model_type="smplx", gender=gender, use_pca=False,
                         flat_hand_mean=True, num_betas=20, num_expression_coeffs=10, ext="npz",
                         batch_size=1).to(_DEV)

    # static vertex → robot-link assignment via lbs skinning argmax
    arg = model.lbs_weights.detach().cpu().numpy().argmax(1)      # (V,)
    j2l = _joint2link()
    link_names = ([f"robot0_l_palm"] + [f"robot0_l_{fg}{seg}" for fg, _ in _FINGERS for seg in _SEGS]
                  + [f"robot0_r_palm"] + [f"robot0_r_{fg}{seg}" for fg, _ in _FINGERS for seg in _SEGS])
    link_idx = {n: i for i, n in enumerate(link_names)}
    vtx_link = np.full(arg.shape, -1, np.int32)                  # vertex → link-index (or -1)
    for v, j in enumerate(arg):
        if j in j2l:
            vtx_link[v] = link_idx[j2l[j]]
    hand_v = np.where(vtx_link >= 0)[0]
    hand_v_link = vtx_link[hand_v]

    # SMPL-X FK → full vertices (world frame == joint_positions/fingertip_pad_pos)
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=_DEV)  # noqa: E731
    go = T(sm["smplx_global_orient"]); bp = T(sm["smplx_body_pose"]); hp = T(sm["smplx_hand_pose"])
    tr = T(sm["smplx_transl"]); betas = T(sm["smplx_betas"]).reshape(1, -1)
    verts = np.empty((F, len(hand_v), 3), np.float32)
    z = lambda n, d: torch.zeros(n, d, device=_DEV)  # noqa: E731
    with torch.no_grad():
        for s in range(0, F, 2048):
            e = min(s + 2048, F); n = e - s
            o = model(betas=betas.expand(n, -1), global_orient=go[s:e], body_pose=bp[s:e],
                      left_hand_pose=hp[s:e, :45], right_hand_pose=hp[s:e, 45:], transl=tr[s:e],
                      expression=z(n, 10), jaw_pose=z(n, 3), leye_pose=z(n, 3), reye_pose=z(n, 3))
            verts[s:e] = o.vertices[:, hand_v, :].detach().cpu().numpy()

    # object motion gate + scan mesh
    bk = [k for k in sm.files if k.startswith("obj__") and k.endswith("__base")]
    obj_name = bk[0].split("__")[1] if bk else ""
    ob = sm[bk[0]].astype(np.float64); op = ob[:, :3]; oq = ob[:, 3:7]
    lv = np.zeros_like(op); lv[:-1] = (op[1:] - op[:-1]) * _FPS
    spd = np.linalg.norm(lv, axis=-1)
    dotq = np.abs((oq[:-1] * oq[1:]).sum(-1)).clip(0, 1)
    ang = np.zeros(F); ang[:-1] = 2 * np.arccos(dotq) * _FPS
    vel = (spd > _OBJ_LINVEL_TH) | (ang > _OBJ_ANGVEL_TH)
    mesh = trimesh.load(str(_RAW_SCAN / obj_name / "simplified" / "base.obj"), process=False, force="mesh")
    V = np.asarray(mesh.vertices, np.float64)
    VN = np.asarray(mesh.vertex_normals, np.float64)                # (n_objv,3) object-LOCAL outward normals

    from scipy.spatial import cKDTree
    L = len(link_names)
    mask = np.zeros((F, L), np.float32); target = np.zeros((F, L, 3), np.float32)
    normal = np.zeros((F, L, 3), np.float32)                        # object-LOCAL contact reaction dir (surface→hand)
    n_contacts_log = np.zeros(F, np.int32)                          # #object contacts kept per frame
    for t in range(F):
        if args.use_velocity_gate and not vel[t]:
            continue
        R = _quat2R(oq[t])                                         # object local→world rotation
        Vw = V @ R.T + op[t]                                       # (n_objv,3) OBJECT verts in WORLD
        # DexMachina step 1 (cKDTree-accelerated): for each OBJECT vertex, nearest HAND vertex + distance.
        d_obj, paired = cKDTree(verts[t]).query(Vw, k=1)           # (n_objv,) dist + nearest hand-vert idx
        keep = np.where(d_obj < args.gamma)[0]                     # object vertices in contact
        if keep.size == 0:
            continue
        cw = Vw[keep]                                              # (M,3) world contact points (object surface)
        clink = hand_v_link[paired[keep]]                          # (M,) link via the paired HAND vertex
        # per-contact direction, object-LOCAL (pose-invariant), pointing OUT of the object surface.
        # "surface": the mesh's own outward normal at the contact vertex. This is the axis of the
        #   friction cone, which is what the contact-wrench reward needs, and it is already in object
        #   coordinates so no rotation is involved.
        # "to-hand": the previous behaviour — direction from the object surface point to the paired
        #   HAND vertex. It approximates the surface normal only while the hand sits directly above
        #   the surface; where the nearest hand vertex is off to the side it tilts away from it.
        # Both point outward, so downstream consumers see the same sign convention either way.
        if args.normal_source == "surface":
            nrm_l = VN[keep]
        else:
            nrm_w = verts[t][paired[keep]] - cw                    # (M,3) world, surface→hand
            nrm_w /= np.clip(np.linalg.norm(nrm_w, axis=1, keepdims=True), 1e-9, None)
            nrm_l = nrm_w @ R                                      # (M,3) object-local (world→local dir)
        # DexMachina step 2 (FPS): spatially subsample to <= num_contacts (keep link/normal aligned).
        fps = _farthest_point_sample(cw, args.num_contacts)
        cw = cw[fps]; clink = clink[fps]; nrm_l = nrm_l[fps]
        n_contacts_log[t] = len(cw)
        # aggregate per link: target = mean OBJECT contact pos (world); normal = mean reaction dir (local, renorm).
        for li in range(L):
            sel = clink == li
            if sel.any():
                mask[t, li] = 1.0
                target[t, li] = cw[sel].mean(0)
                nl = nrm_l[sel].mean(0)
                normal[t, li] = nl / max(float(np.linalg.norm(nl)), 1e-9)

    out = clip_dir / "0" / "hand_contact.npz"
    # normal_source is recorded so a consumer can tell which convention a file was written with.
    np.savez(out, link_names=np.array(link_names), mask=mask,
             target=target.astype(np.float32), normal=normal.astype(np.float32),
             normal_source=np.array(args.normal_source))
    nfire = (mask.sum(0) > 0)
    active = n_contacts_log[n_contacts_log > 0]
    print(f"[hand-contact/OptionA] {args.clip}: obj={obj_name}  gamma={args.gamma}  vel_gate={args.use_velocity_gate}")
    print(f"[hand-contact] frames with contact: {int((n_contacts_log>0).sum())}/{F}  "
          f"obj-contacts/frame(when any): mean {active.mean():.0f} max {int(active.max()) if active.size else 0} "
          f"(cap {args.num_contacts})" if active.size else "[hand-contact] NO contact frames — loosen --gamma")
    print(f"[hand-contact] links with ANY contact ({int(nfire.sum())}/{L}):")
    for i, n in enumerate(link_names):
        if mask[:, i].sum() > 0:
            print(f"    {n:22s} {int(mask[:, i].sum()):3d} frames")
    print(f"[hand-contact] wrote {out}")


if __name__ == "__main__":
    main()
