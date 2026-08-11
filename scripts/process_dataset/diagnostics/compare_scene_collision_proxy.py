"""Box-OBB vs heightmap as the scene proxy for the retarget's world_collision cost.

The retarget currently collides the robot against a flat floor only — the context objects the env
spawns (counter, sink, board, ...) are not in the cost at all, which is why the retargeted LEFT hand
sits inside the countertop (measured: 470/501 frames, peak 5.47 m/s expulsion in sim).

Adding them needs a geometry pyroki can collide a Capsule against. Two are native:

    Heightmap   vertical distance to the top surface  ->  everything below the countertop reads as
                penetrating by its full depth, so the robot's own legs and pelvis, standing at the
                counter, are "inside" by up to the counter height. That fights the retarget.
    Box (OBB)   proper signed distance; inside, the depth is the distance to the NEAREST FACE, so a
                pelvis grazing the counter edge reads as a few cm, not a metre.

This runs both against every robot collision capsule over the whole clip and prints the two
penetration distributions side by side, so the choice is made on numbers rather than on the argument
above. Run in env_pyroki.

    /home/peunsu/anaconda3/envs/env_pyroki/bin/python \
        scripts/process_dataset/diagnostics/compare_scene_collision_proxy.py --clip s101_seg12_knife
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import numpy as onp
import pyroki as pk
import trimesh
import yourdfpy
from pyroki.collision import collide
from scipy.spatial.transform import Rotation as R

_ROOT = Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data"
_URDF = _ROOT / "robots" / "G1" / "urdf_pyroki" / "g1_shadow.urdf"
_ORDER = json.load(open(_ROOT / "robots" / "G1" / "g1_shadow_joint_order.json"))
_PROC = _ROOT / "processed" / "parahome"
_SCAN = _ROOT / "raw" / "parahome" / "data" / "scan"

p = argparse.ArgumentParser()
p.add_argument("--clip", default="s101_seg12_knife")
p.add_argument("--cls", default="single_rigid")
p.add_argument("--context_radius", type=float, default=1.0)
p.add_argument("--context_support_radius", type=float, default=1.5)
args = p.parse_args()

sm = onp.load(_PROC / "smplx" / args.cls / args.clip / "0" / "trajectory.npz", allow_pickle=True)
rt = onp.load(_PROC / "g1_shadow" / args.cls / args.clip / "0" / "trajectory_pyroki.npz")

# ---- context selection, mirroring the env exactly (proximity to the active object's swept path) ----
obj_key = [k for k in sm.files if k.startswith("obj__") and k.endswith("__base")][0]
act = sm[obj_key].astype(onp.float64)
act_xy, act0 = act[:, :2], act[0]
cands = []
for k in (kk for kk in sm.files if kk.startswith("ctx__") and kk.endswith("__base")):
    pose0 = sm[k][0].astype(onp.float64)
    dmin = float(onp.linalg.norm(act_xy - pose0[None, :2], axis=1).min())
    cands.append((k.split("__")[1], pose0, dmin))
keep = {n for n, _, dm in cands if dm < args.context_radius}
below = [(float(onp.linalg.norm(act0[:2] - q[:2])), n) for n, q, _ in cands
         if q[2] < act0[2] and float(onp.linalg.norm(act0[:2] - q[:2])) < args.context_support_radius]
if below:
    keep.add(min(below)[1])
sel = [(n, q) for n, q, _ in cands if n in keep]
print(f"context ({len(sel)}): {sorted(n for n, _ in sel)}")

# ---- robot capsules over the clip ----
urdf = yourdfpy.URDF.load(str(_URDF))
robot = pk.Robot.from_urdf(urdf)
rcoll = pk.collision.RobotCollision.from_urdf(urdf)
an = robot.joints.actuated_names
act_names = _ORDER["action_joint_names"]
jp_act = rt["g1_joint_pos"].astype(onp.float32)                   # (F, 65) in act_names order
col = {n: i for i, n in enumerate(act_names)}
cfg = onp.zeros((jp_act.shape[0], len(an)), onp.float32)
for i, n in enumerate(an):
    if n in col:
        cfg[:, i] = jp_act[:, col[n]]
root = rt["g1_root_pose"].astype(onp.float32)                     # (F,7) pos + quat wxyz
F = cfg.shape[0]
T_root = jaxlie.SE3(jnp.concatenate([jnp.array(root[:, 3:7]), jnp.array(root[:, :3])], axis=-1))
caps = rcoll.at_config(robot, jnp.array(cfg))                     # (F, B)
caps = caps.transform(jaxlie.SE3(T_root.wxyz_xyz[:, None, :]))
B = caps.get_batch_axes()[-1]
print(f"frames {F}, capsules {B}")

box_d = onp.full((F, B, len(sel)), onp.inf, onp.float32)
hm_d = onp.full((F, B, len(sel)), onp.inf, onp.float32)
cap_c = onp.asarray(caps.pose.translation())                      # (F,B,3) capsule centres
for j, (name, q) in enumerate(sel):
    mesh = trimesh.load(str(_SCAN / name / "simplified" / "base.obj"), process=False, force="mesh")
    Rm = R.from_quat(q[[4, 5, 6, 3]]).as_matrix()
    v = mesh.vertices @ Rm.T + q[:3]
    lo, hi = v.min(0), v.max(0)
    ctr, ext = (lo + hi) / 2.0, (hi - lo)
    box = pk.collision.Box.from_extent(
        extent=jnp.array(ext, jnp.float32),
        position=jnp.array(ctr, jnp.float32),
        wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]))
    box_d[:, :, j] = onp.asarray(collide(caps, box.broadcast_to((F, B))))
    # heightmap equivalent: the column is solid up to the top face over the whole xy footprint
    inside_xy = ((cap_c[..., 0] >= lo[0]) & (cap_c[..., 0] <= hi[0])
                 & (cap_c[..., 1] >= lo[1]) & (cap_c[..., 1] <= hi[1]))
    hm_d[:, :, j] = onp.where(inside_xy, cap_c[..., 2] - hi[2], onp.inf)

names = [str(n) for n in rcoll.link_names] if hasattr(rcoll, "link_names") else [f"link{i}" for i in range(B)]
bx, hm = box_d.min(-1), hm_d.min(-1)                              # (F,B) worst object per capsule


def report(tag, d):
    pen = d[onp.isfinite(d)]
    neg = pen[pen < 0]
    print(f"\n── {tag}")
    print(f"   penetrating (capsule,frame) pairs : {neg.size} / {pen.size}")
    if neg.size:
        print(f"   depth cm   mean {-neg.mean() * 100:6.2f}   p95 {-onp.percentile(neg, 5) * 100:6.2f}"
              f"   max {-neg.min() * 100:7.2f}")
    worst = onp.where(onp.isfinite(d), d, onp.inf).min(0)
    o = onp.argsort(worst)
    print("   deepest links:")
    for i in o[:6]:
        if worst[i] >= 0:
            break
        print(f"     {names[i]:30s} {-worst[i] * 100:7.2f} cm")


report("Box (single OBB per object)", bx)
report("Heightmap (solid column to top face)", hm)
print("\nlower-body links only (the ones a solid column would falsely trap):")
low = [i for i, n in enumerate(names)
       if any(s in n for s in ("pelvis", "hip", "knee", "ankle", "waist", "torso"))]
for tag, d in (("Box", bx), ("Heightmap", hm)):
    w = onp.where(onp.isfinite(d[:, low]), d[:, low], onp.inf)
    neg = w[w < 0]
    print(f"   {tag:10s} penetrating {neg.size:6d} pairs   "
          f"max depth {(-neg.min() * 100) if neg.size else 0:7.2f} cm")
