"""Convert the retargeted palm pose into the 6-DoF wrist-joint values of the `wrist6` asset.

The `wrist6` hand (shadow_float6_{l,r}.usd, built by build_shadow_floating_usd.py --wrist6) drives
the palm through a joint chain instead of a free base:

    anchor -> tx -> ty -> tz -> rot1(Y) -> rot2(Z) -> rot3(X) -> (fixed) -> palm

All joint frames are identity, so the composed transform is
    T(q) = Trans(tx,ty,tz) . Ry(rot1) . Rz(rot2) . Rx(rot3)
and therefore palm position = (tx,ty,tz) exactly, palm orientation = the intrinsic YZX triple.
The forward map was verified in sim to 0.0005 mm position / 4e-7 on the rotation matrix.

So this file only has to invert that: copy the position, decompose the quaternion.

WHY YZX AND NOT ZXY
-------------------
The chain used to be ZXY (DexMachina's order). Measured over 16 clips / both hands / 5638 frames,
ZXY is the WORST of the 12 Euler conventions for this data. det(J) is the angular-Jacobian
determinant = control authority along the degenerate direction:

    ZXY  det_min 0.0267  det<0.2 in 8.1% of frames   <- worst of 12
    ZXZ  det_min 0.1032  det<0.2 in 0.8%             lowest rates, but singular at middle=0,
                                                      which is near a hand's neutral pose
    YZX  det_min 0.1615  det<0.2 in 0.1%             <- chosen

On s101_seg12_knife's right hand (which diverged in sim under ZXY), the palm's physical angular
velocity peaks at 7.51 rad/s but ZXY demands 46.59 rad/s of the joints — a 6.21x amplification,
because near the singularity two joints must counter-swing 105 deg each to turn the palm 3 deg
(measured, 50 fps frames 70-76). YZX demands 7.63 rad/s: amplification 1.02x, det 0.946, 0.0%
exposure. Unwrapping cannot fix that amplification — those are not branch wraps but the joint
motion the chart genuinely requires.

Judge singularity proximity by det(J), NEVER by the middle angle in degrees: |middle| of 88.5 vs
84.1 deg looks like a 5% difference but cos gives 0.026 vs 0.105, a 4x difference in authority.
Re-audit any time with diagnostics/audit_euler_convention.py.

BRANCH SELECTION (replaces np.unwrap)
-------------------------------------
An Euler decomposition has two exact solutions per frame -- (a, b, c) and, for Tait-Bryan,
(a+pi, pi-b, c+pi) -- each modulo 2pi per component. Picking per-frame with atan2 lets the sequence
jump between them. `np.unwrap` fixes the 2pi wraps (the -178 -> +171 case) but a branch flip is NOT
a multiple of 360 deg, so unwrap can mis-correct it: a flip shows up as ~180 deg on two components
and subtracting 360 makes it worse. This exporter instead enumerates both branches with their 2pi
shifts and takes whichever is closest to the previous frame's solution. That is exact (orientation
error measured at 1e-13 deg), deterministic, and provably the best any unwrap can do. Values still
accumulate past +-180 deg, which is why the asset's revolute limits are +-720 deg.

    <clip_dir>/wrist_dof6.npz
        wrist_dof_l   (F, 6)  tx,ty,tz [m], rot1,rot2,rot3 [rad], branch-continuous
        wrist_dof_r   (F, 6)
        joint_names   (6,)    ['tx','ty','tz','rot1','rot2','rot3'] — the suffixes of
                              robot0_{s}_wrist_<name>, so a consumer matches BY NAME
        rot_seq       str     'YZX' — the intrinsic rotation order these values assume
        det_l, det_r  (F,)    |det(J)| per frame. The singularity detector; watch for < 0.2
        max_jump      (F,)    per-frame max |diff| over the 6 DoF (rad/m mixed), both hands
        recon_err_deg scalar  max orientation error of the forward map — sanity check, ~1e-13
        fps           scalar  inherited from wrist_ref.npz (30, the source rate)

Usage
    <env_isaaclab python> scripts/process_dataset/retarget/export_wrist_dof6.py --overwrite
    <env_isaaclab python> scripts/process_dataset/retarget/export_wrist_dof6.py --clip s101_seg12_knife --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

_PROC = (Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data"
         / "processed" / "parahome")
# build_shadow_floating_usd.py 의 _W6_ROT_SEQ 와 반드시 일치해야 합니다. 불일치하면 pose 가
# 조용히 틀어지므로, 아래 _verify_against_asset() 이 에셋에서 읽어 대조합니다.
_ROT_SEQ = "YZX"
_NAMES = np.array(["tx", "ty", "tz", "rot1", "rot2", "rot3"])
_AX = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}


def _det_j(seq: str, ang: np.ndarray) -> np.ndarray:
    """(F,3) -> (F,) |det| of the angular Jacobian for R = R_a1(t1) R_a2(t2) R_a3(t3)."""
    a1, a2, a3 = [_AX[c] for c in seq]
    R1 = Rot.from_rotvec(ang[:, [0]] * a1[None, :]).as_matrix()
    R2 = Rot.from_rotvec(ang[:, [1]] * a2[None, :]).as_matrix()
    J = np.stack([np.broadcast_to(a1, (len(ang), 3)), R1 @ a2, (R1 @ R2) @ a3], -1)
    return np.abs(np.linalg.det(J))


def _decompose(seq: str, R: Rot) -> tuple[np.ndarray, float]:
    """Branch-continuous Euler triples. Returns ((F,3), max reconstruction error in deg)."""
    a = R.as_euler(seq)                                    # uppercase = intrinsic
    proper = seq[0] == seq[2]
    b = np.stack([a[:, 0] + np.pi,
                  (-a[:, 1] if proper else np.pi - a[:, 1]),
                  a[:, 2] + np.pi], 1)
    prev = a[0].copy()
    out = [prev]
    for i in range(1, len(a)):
        cands = [x + 2 * np.pi * np.round((prev - x) / (2 * np.pi)) for x in (a[i], b[i])]
        prev = min(cands, key=lambda t: np.abs(t - prev).max())
        out.append(prev)
    T = np.stack(out)
    err = float(np.degrees((Rot.from_euler(seq, T).inv() * R).magnitude().max()))
    return T, err


def _verify_against_asset() -> str:
    """에셋의 실제 축 순서를 읽어 _ROT_SEQ 와 대조. 불일치는 조용한 pose 오류로 이어집니다."""
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        return "pxr 없음 — 에셋 대조를 건너뜁니다 (isaaclab 환경에서 실행하십시오)"
    p = (_PROC.parents[1] / "robots" / "G1" / "shadow_float6_r.usd")
    if not p.exists():
        return f"{p.name} 없음 — 에셋 대조를 건너뜁니다"
    st = Usd.Stage.Open(str(p))
    axes = {}
    for pr in st.Traverse():
        if pr.IsA(UsdPhysics.RevoluteJoint) and "_wrist_rot" in pr.GetName():
            a = pr.GetAttribute("physics:axis")
            axes[pr.GetName()] = a.Get() if a and a.HasAuthoredValue() else None
    seq = "".join(axes[k] for k in sorted(axes))           # rot1, rot2, rot3
    if seq != _ROT_SEQ:
        raise SystemExit(f"[export-wrist-dof6] 축 순서 불일치: 에셋 {seq} vs 이 스크립트 "
                         f"{_ROT_SEQ}. build_shadow_floating_usd.py --wrist6 --rot_seq "
                         f"{_ROT_SEQ} 로 에셋을 다시 만들거나 _ROT_SEQ 를 맞추십시오.")
    return f"에셋 축 순서 {seq} 와 일치"


def convert(clip: str, cls: str = "single_rigid", overwrite: bool = False) -> str:
    cd = _PROC / "g1_shadow" / cls / clip / "0"
    src, dst = cd / "wrist_ref.npz", cd / "wrist_dof6.npz"
    if not src.exists():
        return f"{clip}: wrist_ref.npz 없음 — export_wrist_ref.py 를 먼저 실행하세요"
    if dst.exists() and not overwrite:
        return f"{clip}: 이미 존재 (--overwrite 로 갱신)"
    d = np.load(src, allow_pickle=True)
    ln = [str(x) for x in d["link_names"]]
    if not all(n.endswith("_palm") for n in ln):
        return (f"{clip}: wrist_ref.npz 가 palm 기준이 아닙니다 ({ln}) — "
                f"export_wrist_ref.py 를 --overwrite 로 다시 실행하세요")

    out: dict[str, np.ndarray] = {}
    jumps = None
    err_max = 0.0
    for side in ("l", "r"):
        W = d[f"wrist_pose_{side}"].astype(np.float64)          # (F,7) pos + quat wxyz
        q = W[:, 3:7] / np.linalg.norm(W[:, 3:7], axis=1, keepdims=True)
        T, err = _decompose(_ROT_SEQ, Rot.from_quat(q[:, [1, 2, 3, 0]]))
        err_max = max(err_max, err)
        E = np.concatenate([W[:, :3], T], 1)
        out[f"wrist_dof_{side}"] = E.astype(np.float32)
        out[f"det_{side}"] = _det_j(_ROT_SEQ, T).astype(np.float32)
        j = np.zeros(len(E))
        j[1:] = np.abs(np.diff(E, axis=0)).max(axis=1)
        jumps = j if jumps is None else np.maximum(jumps, j)

    np.savez(dst, joint_names=_NAMES, rot_seq=_ROT_SEQ, max_jump=jumps.astype(np.float32),
             recon_err_deg=np.float32(err_max), fps=np.float32(30.0), **out)
    dl, dr = out["det_l"], out["det_r"]
    return (f"{clip}: F={len(jumps)}  점프 중앙 {np.degrees(np.median(jumps[1:])):5.2f}도 "
            f"최대 {np.degrees(jumps.max()):6.1f}도  |  det최소 l {dl.min():.4f} r {dr.min():.4f}  "
            f"det<0.2 l {(dl < 0.2).mean() * 100:4.1f}% r {(dr < 0.2).mean() * 100:4.1f}%  |  "
            f"재구성 {err_max:.1e}도")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="", help="비우면 모든 클립")
    ap.add_argument("--class", dest="cls", default="", help="비우면 모든 클래스")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    print(f"[export-wrist-dof6] 회전 규약 {_ROT_SEQ}, {_verify_against_asset()}")
    if a.clip:
        cls = a.cls or next((p.parent.parent.name for p in
                             _PROC.glob(f"g1_shadow/*/{a.clip}/0/wrist_ref.npz")), "single_rigid")
        print("[export-wrist-dof6] " + convert(a.clip, cls, a.overwrite))
        return 0
    n = 0
    for f in sorted(_PROC.glob("g1_shadow/*/*/0/wrist_ref.npz")):
        clip, cls = f.parents[1].name, f.parents[2].name
        if a.cls and cls != a.cls:
            continue
        print("[export-wrist-dof6] " + convert(clip, cls, a.overwrite))
        n += 1
    print(f"[export-wrist-dof6] {n}개 클립 처리")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
