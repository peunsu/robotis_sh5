#!/usr/bin/env python3
"""리타게팅 결과를 버전 간에 같은 잣대로 비교합니다.

`retarget_g1_pyroki*.py` 가 찍는 접촉 간극은 그 스크립트가 최적화한 목표에 대한 값이라, 대응표가
다른 두 버전을 비교하는 데 그대로 쓰면 서로 다른 자를 대는 셈이 됩니다. 여기서는 버전과 무관하게
정의되는 값만 잽니다.

  접지    발바닥 높이, 접촉 중 발 미끄러짐  <- 1단계를 건드렸을 때 회귀했는지 보는 값
  손      손끝 패드가 사람 손끝에서 얼마나 떨어져 있나 (대응표와 무관한 공통 기준)
  손목    손 키포인트로 만든 랜드마크 프레임 대비 회전 오차

    <env_pyroki python> scripts/process_dataset/retarget/metrics_retarget.py --tag "" --tag _v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yourdfpy

import pyroki as pk

_ROOT = Path("/home/peunsu/workspace/robotis_sh5/source/robotis_sh5")
_PROC = _ROOT / "data" / "processed" / "parahome"
_URDF = _ROOT / "data" / "robots" / "G1" / "urdf_pyroki" / "g1_shadow.urdf"

_ANKLE_SOLE_OFF = 0.036                       # 발목 링크 원점에서 발바닥까지 (원본 스크립트와 동일)
# ── [ROLLBACK MARKER: smplx-kpts] ParaHome joint_positions → SMPL-X (2026-09-04) ────────────
# 리타게팅(retarget_g1_pyroki.py)이 SMPL-X 로 넘어갔는데 이 스크립트만 ParaHome 스트림에 남아
# 있어서, 같은 판정식을 써도 발 접지 기준선이 달랐습니다 (ParaHome 발끝 z 중앙 1.4 cm vs
# SMPL-X 2.4~3.5 cm). 인덱스는 리타게팅의 _BODY/_HAND_CHAIN/_PALM/_PAD_BASE 와 같은 규약입니다.
_BALL_L, _BALL_R = 10, 11                     # SMPL-X left_foot / right_foot (발 앞꿈치)
# [foot-plant-3d] height 0.05, vel 1.00 (3D 노름) — 리타게팅의 _FOOT_PLANT_H/_V 와 같아야 합니다.
_FOOT_PLANT_H, _FOOT_PLANT_V, _FPS = 0.05, 1.00, 30.0
# 사람 손끝(TIP)과 로봇 distal 링크 + 패드 오프셋. 대응표가 바뀌어도 이 짝은 고정입니다.
# 손끝은 smplx_joints(55) 뒤에 fingertip_pad_pos(10) 를 이어붙인 자리에서 읽습니다 — 순서는
# LEFT[th,ff,mf,rf,lf] 다음 RIGHT[th,ff,mf,rf,lf] 로 리타게팅의 _FT_OFF_R 순회와 동일합니다.
_PAD_BASE = 55
_TIPS = {"th": 0, "ff": 1, "mf": 2, "rf": 3, "lf": 4}
_FT_OFF_R = {"th": [-0.0085, 0.0, 0.02], "ff": [0.0, -0.006, 0.0175], "mf": [0.0, -0.006, 0.0175],
             "rf": [0.0, -0.006, 0.0175], "lf": [0.0, -0.006, 0.0175]}
# 손목(SMPL-X wrist 20/21)과 손 관절 시작(25/40, 손가락 순서 index,middle,pinky,ring,thumb).
# 손목 랜드마크 프레임에 쓰는 index MCP = hand+0, middle MCP = hand+3.
_SIDES = {"l": {"wrist": 20, "hand": 25, "pad": _PAD_BASE},
          "r": {"wrist": 21, "hand": 40, "pad": _PAD_BASE + 5}}


def quat2R(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                    -1).reshape(q.shape[:-1] + (3, 3))


def landmark_R(wrist, idx_mcp, mid_mcp):
    """손 키포인트 3개로 손 자세 프레임을 만듭니다 (환경의 _landmark_frame 과 동일 구성).

    회전 목표가 리타게팅에 없어서(retarget npz 에 손바닥 쿼터니언이 없음) 사람과 로봇 양쪽에서
    같은 방식으로 만들어 비교합니다. 좌표 규약에 의존하지 않는 것이 요점입니다.
    """
    z = mid_mcp - wrist
    z = z / np.linalg.norm(z, axis=-1, keepdims=True).clip(1e-9)
    x = np.cross(z, idx_mcp - wrist)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)
    return np.stack([x, np.cross(z, x), z], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="s101_seg12_knife")
    ap.add_argument("--class", dest="cls", default="single_rigid")
    ap.add_argument("--tag", action="append", default=None, help='출력 접미사. 예: --tag "" --tag _v2')
    args = ap.parse_args()
    tags = args.tag if args.tag else ["", "_v2"]

    d = _PROC / "smplx" / args.cls / args.clip / "0"
    sm = np.load(d / "trajectory.npz", allow_pickle=True)
    if "smplx_joints" not in sm.files:
        raise KeyError("[smplx-kpts] smplx_joints 없음 — parahome.py --overwrite 로 재생성하세요")
    jp = np.concatenate([sm["smplx_joints"].astype(np.float64),
                         sm["fingertip_pad_pos"].astype(np.float64)], axis=1)   # (F,65,3)

    urdf = yourdfpy.URDF.load(str(_URDF))
    robot = pk.Robot.from_urdf(urdf)
    lnames = list(robot.links.names)
    an = list(robot.joints.actuated_names)
    import jax.numpy as jnp

    # 발 접지 구간 (원본 스크립트의 _foot_contact 와 같은 판정)
    def plant(idx):
        p = jp[:, idx]
        v = np.zeros(len(p))
        v[1:] = np.linalg.norm(p[1:] - p[:-1], axis=-1) * _FPS   # [foot-plant-3d] 3D 노름
        return (p[:, 2] < _FOOT_PLANT_H) & (v < _FOOT_PLANT_V)
    c_l, c_r = plant(_BALL_L), plant(_BALL_R)

    print(f"클립 {args.clip}\n")
    hdr = (f"{'':<26}" + "".join(f"{('v1' if t == '' else t):>14}" for t in tags))
    print(hdr)

    rows: dict[str, list] = {}
    for tag in tags:
        f = _PROC / "g1_shadow" / args.cls / args.clip / "0" / f"trajectory_pyroki{tag}.npz"
        z = np.load(f, allow_pickle=True)
        col = {str(s): k for k, s in enumerate(z["joint_names"])}
        jpos, rp = z["g1_joint_pos"].astype(np.float64), z["g1_root_pose"].astype(np.float64)
        cfg = np.stack([jpos[:, col[n]] if n in col else np.zeros(len(jpos)) for n in an], axis=1)
        fk = np.asarray(robot.forward_kinematics(cfg=jnp.asarray(cfg)))    # (F,L,7) 루트 기준
        rR, rt = quat2R(rp[:, 3:7]), rp[:, :3]

        def world(link, off=None):
            i = lnames.index(link)
            lR, lt = quat2R(fk[:, i, :4]), fk[:, i, 4:7]
            p = lt if off is None else np.einsum("fij,j->fi", lR, np.asarray(off)) + lt
            return np.einsum("fij,fj->fi", rR, p) + rt

        out = {}
        # --- 접지 ---
        sole = np.minimum(world("left_ankle_roll_link")[:, 2], world("right_ankle_roll_link")[:, 2]) - _ANKLE_SOLE_OFF
        out["발바닥 높이 중앙(mm)"] = float(np.median(sole)) * 1000
        out["발바닥 높이 p90(mm)"] = float(np.quantile(sole, 0.9)) * 1000
        sk = []
        for link, con in (("left_ankle_roll_link", c_l), ("right_ankle_roll_link", c_r)):
            w = world(link)
            step = np.linalg.norm(w[1:, :2] - w[:-1, :2], axis=-1)
            m = con[1:] & con[:-1]
            if m.any():
                sk.append(step[m])
        sk = np.concatenate(sk) if sk else np.zeros(1)
        out["발미끄러짐 중앙(mm/f)"] = float(np.median(sk)) * 1000
        out["발미끄러짐 p90(mm/f)"] = float(np.quantile(sk, 0.9)) * 1000

        # --- 손끝: 사람 TIP 과의 거리 (대응표와 무관) ---
        err = []
        for side, sd in _SIDES.items():
            for fg, ti in _TIPS.items():
                o = _FT_OFF_R[fg] if not (side == "l" and fg != "th") else \
                    [_FT_OFF_R[fg][0], -_FT_OFF_R[fg][1], _FT_OFF_R[fg][2]]
                err.append(np.linalg.norm(world(f"robot0_{side}_{fg}distal", o)
                                          - jp[:, sd["pad"] + ti], axis=-1))
        err = np.stack(err)                                            # (10,F)
        out["손끝-사람 거리 중앙(mm)"] = float(np.median(err)) * 1000
        out["손끝-사람 거리 p90(mm)"] = float(np.quantile(err, 0.9)) * 1000

        # --- 손목 회전: 사람/로봇 각각 랜드마크 프레임을 만들어 각도 차 ---
        wr = []
        for side, sd in _SIDES.items():
            # [smplx-kpts] wrist / index MCP / middle MCP
            Rh = landmark_R(jp[:, sd["wrist"]], jp[:, sd["hand"] + 0], jp[:, sd["hand"] + 3])
            Rr = landmark_R(world(f"robot0_{side}_palm"),
                            world(f"robot0_{side}_ffknuckle"), world(f"robot0_{side}_mfknuckle"))
            dR = np.einsum("fji,fjk->fik", Rh, Rr)
            tr = dR[:, 0, 0] + dR[:, 1, 1] + dR[:, 2, 2]
            wr.append(np.degrees(np.arccos(np.clip((tr - 1) * 0.5, -1, 1))))
        wr = np.concatenate(wr)
        out["손목 회전오차 중앙(도)"] = float(np.median(wr))
        out["손목 회전오차 p90(도)"] = float(np.quantile(wr, 0.9))
        rows[tag] = out

    for k in rows[tags[0]]:
        print(f"{k:<26}" + "".join(f"{rows[t][k]:>14.1f}" for t in tags))


if __name__ == "__main__":
    main()
