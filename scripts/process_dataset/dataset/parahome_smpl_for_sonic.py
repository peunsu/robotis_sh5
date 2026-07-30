"""Offline: ParaHome SMPL-X params -> SONIC SMPL(24) encoder arrays, resampled 30->50 fps.

Milestone 1 (pan clip). Pure numpy/torch (needs gear_sonic + torch; NO Isaac Sim).
Reproduces the SHIPPED SONIC recipe (pico_manager_thread_server.process_smpl_joints):
  root_q_z  = smpl_root_ytoz_up( aa2quat(global_orient) )        # Y-up -> Z-up, base NOT removed
  joints24  = compute_human_joints(body_pose[:, :63], global_orient=aa(root_q_z))  # (F,24,3) [0..21,39,54], z-up
  root_q_zb = remove_smpl_base_rot(root_q_z)                     # == smpl_root_quat_w
  smpl_joints_local = quat_apply(quat_inv(root_q_zb), joints24)  # per-frame own-root canonicalize; NO joint-0 subtraction

Outputs -> <g1_shadow>/<class>/<clip>/0/sonic_smpl_50fps.npz:
  smpl_joints_local (N,72) f32  | root_q_zb (N,4) wxyz f32  | wrist_ref (N,6) f32 (retarget L/R wrist roll/pitch/yaw) | fps=50

Run:
  /home/peunsu/anaconda3/envs/env_isaaclab/bin/python scripts/process_dataset/dataset/parahome_smpl_for_sonic.py --clip s100_seg00_pan
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
)
from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up

_PROC = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/processed/parahome"
_HJINFO = "/home/peunsu/workspace/GR00T-WholeBodyControl/gear_sonic/data/human/human_joints_info.pkl"
SRC_FPS = 30.0
TGT_FPS = 50.0
# ParaHome SMPL-X global_orient is ALREADY Z-up (verified: remove_base(raw) gives upright
# roll~2/pitch~9/yaw-117 == body_global_transform == retarget root). SONIC's smpl_root_ytoz_up
# assumes Y-up (Bones-SEED/VR); applying it to ParaHome DOUBLE-rotates -> 143deg garbage root ->
# SONIC over-torques and falls. So DO NOT apply ytoz for ParaHome.
SMPL_Y_UP = False
# our g1_joint_pos wrist cols (action_joint_names): L/R wrist roll, pitch, yaw = SONIC G1_ISAACLab idx 23..28
WRIST_COLS = [23, 24, 25, 26, 27, 28]


def _slerp_wxyz(q_wxyz: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """Slerp a (F,4) wxyz quaternion series from t_src onto t_tgt. Uses scipy (xyzw internally)."""
    from scipy.spatial.transform import Rotation, Slerp

    r = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]])  # wxyz -> xyzw
    out = Slerp(t_src, r)(t_tgt).as_quat()  # xyzw
    return out[:, [3, 0, 1, 2]].astype(np.float32)  # -> wxyz


def _lerp(x: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """Linear interp of a (F, D) array along time."""
    D = x.shape[1]
    return np.stack([np.interp(t_tgt, t_src, x[:, d]) for d in range(D)], axis=1).astype(np.float32)


def process_clip(clip: str, cls: str, overwrite: bool = False):
    smplx_npz = os.path.join(_PROC, "smplx", cls, clip, "0", "trajectory.npz")
    retgt_npz = os.path.join(_PROC, "g1_shadow", cls, clip, "0", "trajectory_pyroki.npz")
    out_npz = os.path.join(_PROC, "g1_shadow", cls, clip, "0", "sonic_smpl_50fps.npz")
    if os.path.exists(out_npz) and not overwrite:
        print(f"[skip] {out_npz} exists (use --overwrite)")
        return out_npz

    d = np.load(smplx_npz, allow_pickle=True)
    body = torch.tensor(np.asarray(d["smplx_body_pose"]), dtype=torch.float32)      # (F,63)
    root_aa = torch.tensor(np.asarray(d["smplx_global_orient"]), dtype=torch.float32)  # (F,3) Y-up aa
    F = body.shape[0]
    print(f"[clip] {clip} F={F} @ {SRC_FPS}fps")

    # ---- SHIPPED SONIC SMPL recipe (per frame), ytoz GATED (ParaHome is Z-up -> skip) ----
    root_q = angle_axis_to_quaternion(root_aa)               # (F,4) wxyz
    if SMPL_Y_UP:
        root_q = smpl_root_ytoz_up(root_q)                   # Y-up -> Z-up (Bones-SEED/VR only)
    root_aa_fk = quaternion_to_angle_axis(root_q)            # (F,3) (== root_aa when not ytoz'd)
    joints24 = compute_human_joints(body[:, :63], global_orient=root_aa_fk,
                                    human_joints_info_path=_HJINFO)  # (F,24,3) [0..21,39,54]
    root_q_zb = remove_smpl_base_rot(root_q, w_last=False)   # (F,4) == smpl_root_quat_w (now upright Z-up)
    qinv = quat_inv(root_q_zb).unsqueeze(1).expand(F, 24, 4)  # (F,24,4)
    smpl_joints_local = quat_apply(qinv, joints24)           # (F,24,3) own-root canonicalized (global-orient-invariant)
    smpl_joints_local = smpl_joints_local.reshape(F, 72).cpu().numpy().astype(np.float32)
    root_q_zb = root_q_zb.cpu().numpy().astype(np.float32)

    # ---- wrist reference + spawn-alignment root quat from retarget ----
    g1_root_quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # fallback identity (wxyz)
    if os.path.exists(retgt_npz):
        rt = np.load(retgt_npz, allow_pickle=True)
        g = rt["g1_joint_pos"]  # (F,65)
        assert g.shape[0] == F, f"retarget F {g.shape[0]} != smplx F {F}"
        wrist_ref = g[:, WRIST_COLS].astype(np.float32)           # (F,6) L/R wrist roll/pitch/yaw
        if "g1_root_pose" in rt.files:
            g1_root_quat0 = rt["g1_root_pose"][0, 3:7].astype(np.float32)  # wxyz, upright robot-frame root@f0
    else:
        print(f"[warn] no retarget {retgt_npz} -> wrist_ref = zeros (degraded wrist tracking)")
        wrist_ref = np.zeros((F, 6), dtype=np.float32)

    # ---- resample 30 -> 50 fps ----
    dur = (F - 1) / SRC_FPS
    N = int(round(dur * TGT_FPS)) + 1
    t_src = np.arange(F) / SRC_FPS
    t_tgt = np.linspace(0.0, dur, N)
    sj50 = _lerp(smpl_joints_local, t_src, t_tgt)   # (N,72) linear
    rq50 = _slerp_wxyz(root_q_zb, t_src, t_tgt)     # (N,4) slerp
    wr50 = _lerp(wrist_ref, t_src, t_tgt)           # (N,6) linear
    print(f"[resample] {F} -> {N} frames @ {TGT_FPS}fps")

    # ---- sanity ----
    jn = np.linalg.norm(sj50.reshape(N, 24, 3), axis=-1)
    qn = np.linalg.norm(rq50, axis=-1)
    print(f"[sanity] joint-local dist: max={jn.max():.3f} mean={jn.mean():.3f} m  (expect limbs <~0.8m)")
    print(f"[sanity] root quat |q|: min={qn.min():.4f} max={qn.max():.4f}  (expect ~1.0)")
    print(f"[sanity] wrist_ref range: [{wr50.min():.3f}, {wr50.max():.3f}] rad")

    # report the (now-fixed) root orientation at frame 0 for sanity
    from scipy.spatial.transform import Rotation as _R
    e0 = np.round(_R.from_quat(rq50[0][[1, 2, 3, 0]]).as_euler("xyz", degrees=True), 1)
    print(f"[sanity] root_q_zb[0] euler xyz={e0} (expect ~upright: roll~2 pitch~9 yaw~-117; NOT roll-105/pitch62)")

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, smpl_joints_local=sj50, root_q_zb=rq50, wrist_ref=wr50,
             g1_root_quat0=g1_root_quat0, fps=np.float32(TGT_FPS))
    print(f"[write] {out_npz}  (smpl_joints_local {sj50.shape}, root_q_zb {rq50.shape}, "
          f"wrist_ref {wr50.shape}, g1_root_quat0={np.round(g1_root_quat0,3)})")
    return out_npz


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="s100_seg00_pan")
    p.add_argument("--class", dest="cls", default="single_rigid")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    process_clip(a.clip, a.cls, a.overwrite)
