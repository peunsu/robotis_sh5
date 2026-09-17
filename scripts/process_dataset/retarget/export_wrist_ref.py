"""Derive the retargeted WRIST pose (position + orientation) from `trajectory_pyroki.npz`.

Why this exists
---------------
The floating-hand pretrain roots each hand at `robot0_{s}_palm` and needs that link's world pose to
seed RSI resets. (It was `robot0_{s}_wrist` until 2026-09-07, when that zero-DOF link was dropped
from the asset. Regenerate this sidecar for EVERY clip after any such frame change — otherwise the
reset places the hand one link off, silently.) The retarget npz only stores `g1_joint_pos` (73) + `g1_root_pose` (pelvis) +
`joint_names` — the wrist pose is implied by forward kinematics but never written out. An earlier
attempt to read `g1_palm_quat` failed: that key exists in the env's loader for older assets but is
absent from every clip we regenerated.

So: run pinocchio FK over the retargeted trajectory and dump the two wrist poses per frame. This is
CPU-only, so it can run while the GPU is busy.

Stage 2 wants the same array (to A/B the RL hand trajectory against the retargeted one), so the file
is written next to the retarget npz rather than kept inside the env.

    <clip_dir>/wrist_ref.npz
        wrist_pose_l (F, 7)   world, pos + quat wxyz
        wrist_pose_r (F, 7)
        link_names   (2,)     the frames the poses were read from, recorded explicitly

Usage
    <env_isaaclab python> scripts/process_dataset/retarget/export_wrist_ref.py            # all clips
    <env_isaaclab python> scripts/process_dataset/retarget/export_wrist_ref.py --clip s101_seg12_knife
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin

_ROOT = Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data"
_PROC = _ROOT / "processed" / "parahome"
# The inequality-tendon (nomimic) URDF is the current default in retarget_g1_pyroki.py, and it is the
# one whose joint set matches the 73-column npz. The mimic URDF would drop the 8 J0 columns.
_URDF = _ROOT / "robots" / "G1" / "urdf_pyroki" / "g1_shadow_nomimic.urdf"
# 떠 있는 손의 articulation root = palm. 키 이름(wrist_pose_*)은 하위 호환으로 유지하지만
# 실제로 담기는 것은 PALM pose 이고, 실제 프레임 이름은 link_names 열에 기록됩니다.
_WRIST = ("robot0_l_palm", "robot0_r_palm")


def export(clip: str, cls: str = "single_rigid", overwrite: bool = False) -> str:
    clip_dir = _PROC / "g1_shadow" / cls / clip / "0"
    src = clip_dir / "trajectory_pyroki.npz"
    dst = clip_dir / "wrist_ref.npz"
    if not src.exists():
        return f"{clip}: trajectory_pyroki.npz 없음 — 건너뜀"
    if dst.exists() and not overwrite:
        return f"{clip}: 이미 존재 (--overwrite 로 갱신)"

    model = pin.buildModelFromUrdf(str(_URDF), pin.JointModelFreeFlyer())
    data = model.createData()
    qidx = {model.names[j]: model.idx_qs[j] for j in range(1, model.njoints) if model.nqs[j] == 1}
    fids = [model.getFrameId(n) for n in _WRIST]

    d = np.load(src, allow_pickle=True)
    qp, rt = d["g1_joint_pos"], d["g1_root_pose"]
    jn = [str(x) for x in d["joint_names"]]
    # Name-based, like the env's `_remap_ref_joints`: the npz column order is not guaranteed to match
    # the URDF's joint order, and assuming it did is what caused the 24-slot hand permutation bug.
    cols = [(qidx[n], i) for i, n in enumerate(jn) if n in qidx]
    missing = [n for n in jn if n not in qidx]
    if missing:
        print(f"  [{clip}] URDF 에 없는 npz 관절 {len(missing)}개 (무시): {missing[:4]}")

    F = len(qp)
    out = np.zeros((2, F, 7), np.float32)
    for f in range(F):
        q = pin.neutral(model)
        q[0:3] = rt[f, :3]
        w, x, y, z = rt[f, 3:7]                      # npz wxyz -> pinocchio xyzw
        q[3:7] = [x, y, z, w]
        for qi, ci in cols:
            q[qi] = qp[f, ci]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for k, fid in enumerate(fids):
            T = data.oMf[fid]
            out[k, f, :3] = T.translation
            qu = pin.Quaternion(T.rotation).coeffs()   # xyzw
            out[k, f, 3:7] = [qu[3], qu[0], qu[1], qu[2]]   # -> wxyz

    np.savez(dst, wrist_pose_l=out[0], wrist_pose_r=out[1],
             link_names=np.array(_WRIST))
    span = np.linalg.norm(out[:, :, :3].reshape(-1, 3).max(0) - out[:, :, :3].reshape(-1, 3).min(0))
    return (f"{clip}: F={F}  좌손목z중앙 {np.median(out[0, :, 2]) * 100:.1f}cm  "
            f"우손목z중앙 {np.median(out[1, :, 2]) * 100:.1f}cm  작업공간대각 {span * 100:.0f}cm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="all")
    ap.add_argument("--class", dest="cls", default="single_rigid")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    clips = ([p.name for p in sorted((_PROC / "g1_shadow" / a.cls).iterdir()) if p.is_dir()]
             if a.clip == "all" else [a.clip])
    for c in clips:
        print("  " + export(c, a.cls, a.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
