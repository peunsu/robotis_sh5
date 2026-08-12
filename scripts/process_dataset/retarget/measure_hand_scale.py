#!/usr/bin/env python3
"""0단계 게이트: 사람 손과 Shadow 손의 마디 길이가 얼마나 다른지 잽니다.

리타게팅 2단계에 손 스케일 행렬을 넣을지 말지를 이 숫자로 정합니다.

  손가락마다 비율이 크게 다르면  -> 스칼라 하나로는 못 맞추므로 20x20 행렬이 필요합니다.
  비율이 전부 1 근처면          -> 크기는 문제가 아니므로 스케일은 넣을 이유가 없고,
                                  접촉이 안 닿는 원인을 다른 데서 찾아야 합니다.

사람 쪽은 ParaHome의 관절 위치(프레임마다 재고 중앙값), 로봇 쪽은 URDF 순기구학의 기본 자세
링크 위치입니다. 둘 다 `retarget_g1_pyroki.py`의 _HAND_CHAIN 대응을 그대로 씁니다 — 리타게팅이
실제로 짝지어 최적화하는 그 노드들이라야 의미가 있습니다.

    <env_pyroki python> scripts/process_dataset/retarget/measure_hand_scale.py [--clip ...]
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

# retarget_g1_pyroki.py 와 동일 (사람 국소 인덱스, Shadow 링크 접미사)
_HAND_CHAIN = {
    "wrist":  ([0], ["palm"]),
    "index":  ([18, 19, 20, 21], ["ffknuckle", "ffproximal", "ffmiddle", "ffdistal"]),
    "middle": ([14, 15, 16, 17], ["mfknuckle", "mfproximal", "mfmiddle", "mfdistal"]),
    "ring":   ([10, 11, 12, 13], ["rfknuckle", "rfproximal", "rfmiddle", "rfdistal"]),
    "pinky":  ([6, 7, 8, 9], ["lfknuckle", "lfproximal", "lfmiddle", "lfdistal"]),
    "thumb":  ([22, 23, 24], ["thproximal", "thmiddle", "thdistal"]),
}
_SIDE_OFF = {"l": 23, "r": 48}          # ParaHome 손 블록 시작 인덱스


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="s101_seg12_knife")
    ap.add_argument("--class", dest="cls", default="single_rigid")
    args = ap.parse_args()

    sm = np.load(_PROC / "smplx" / args.cls / args.clip / "0" / "trajectory.npz", allow_pickle=True)
    jp = sm["joint_positions"].astype(np.float64)                    # (F,73,3)

    urdf = yourdfpy.URDF.load(str(_URDF))
    robot = pk.Robot.from_urdf(urdf)
    import jax.numpy as jnp
    fk = np.asarray(robot.forward_kinematics(cfg=jnp.zeros(robot.joints.num_actuated_joints)))
    lpos = {n: fk[i, 4:7] for i, n in enumerate(robot.links.names)}   # 기본 자세 링크 위치

    print(f"클립 {args.clip}  프레임 {len(jp)}\n")
    print(f"{'손가락':<8}{'마디':<26}{'사람(mm)':>10}{'로봇(mm)':>10}{'비율':>8}")
    rows = []
    for side in ("l", "r"):
        off = _SIDE_OFF[side]
        wrist_h = jp[:, off + _HAND_CHAIN["wrist"][0][0]]
        wrist_r = lpos[f"robot0_{side}_palm"]
        for fname, (loc, sh) in _HAND_CHAIN.items():
            if fname == "wrist":
                continue
            # 손목 -> 첫 마디, 그다음 마디끼리 연속
            hs = [wrist_h] + [jp[:, off + i] for i in loc]
            rs = [wrist_r] + [lpos[f"robot0_{side}_{s}"] for s in sh]
            for k in range(len(rs) - 1):
                lh = float(np.median(np.linalg.norm(hs[k + 1] - hs[k], axis=-1)))
                lr = float(np.linalg.norm(rs[k + 1] - rs[k]))
                name = ("손목→" + sh[0]) if k == 0 else (sh[k - 1] + "→" + sh[k])
                rows.append((side, fname, name, lh, lr, lr / max(lh, 1e-9)))
                if side == "r":                                       # 표는 오른손만 (좌우 대칭)
                    print(f"{fname:<8}{name:<26}{lh*1000:>9.1f}{lr*1000:>10.1f}{lr/max(lh,1e-9):>8.2f}")

    r = np.array([x[5] for x in rows if x[0] == "r"])
    l = np.array([x[5] for x in rows if x[0] == "l"])
    print(f"\n오른손 비율: 중앙 {np.median(r):.2f}  범위 {r.min():.2f}~{r.max():.2f}  "
          f"표준편차 {r.std():.3f}")
    print(f"왼손  비율: 중앙 {np.median(l):.2f}  범위 {l.min():.2f}~{l.max():.2f}  "
          f"표준편차 {l.std():.3f}")
    print(f"좌우 비율 차이(최대) {np.abs(r - l).max():.3f}")

    spread = float(r.max() - r.min())
    print("\n=== 게이트 판정 ===")
    print(f"손가락 간 비율 폭 {spread:.2f}")
    if spread > 0.25:
        print("-> 스칼라 하나로는 못 맞춥니다. 20x20 스케일 행렬을 넣을 근거가 있습니다.")
    elif np.abs(np.median(r) - 1.0) > 0.1:
        print("-> 폭은 좁지만 전체가 치우쳐 있습니다. 스칼라 1개(또는 대각) 스케일이면 충분합니다.")
    else:
        print("-> 크기는 이미 맞습니다. 스케일은 넣을 이유가 없고 접촉 표현 쪽을 봐야 합니다.")


if __name__ == "__main__":
    main()
