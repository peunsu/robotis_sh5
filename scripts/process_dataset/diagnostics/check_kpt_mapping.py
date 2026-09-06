#!/usr/bin/env python3
"""SMPL(ParaHome) 키포인트 ↔ 로봇 링크 대응을 눈으로 검증합니다.

리타게팅(retarget_g1_pyroki.py)의 _BODY 대응표가 기하학적으로 성립하는지 봅니다. 대응이
어긋나 있으면 최적화가 만족시킬 수 없는 목표를 좇게 되고, 그 잔차가 골반 높이나 상체 기울기로
새어 나옵니다 — waist_pitch 중력 모멘트가 그렇게 커집니다.

만드는 것
  1) <prefix>_pairs.png    측면(XZ)/정면(YZ) 투영에 사람 키포인트(초록)와 로봇 링크 원점(시안)을
                           라벨과 함께 찍고 대응 쌍을 선으로 연결. 선 길이가 곧 대응 오차입니다.
  2) <prefix>_dist.png     쌍별 거리(중앙/p95) 막대. 클립 여러 개를 나란히.

로봇 링크 위치는 리타게팅 결과(trajectory_pyroki.npz 의 g1_root_pose + g1_joint_pos)에
pinocchio FK 를 걸어 얻습니다 — 즉 "리타게팅이 실제로 만든 자세"입니다.

사용:
  python check_kpt_mapping.py --clip s101_seg29_pot --clip s152_seg21_pot --out_prefix /tmp/kpt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from matplotlib import font_manager as _fm
for _f in ("Noto Sans CJK JP", "Noto Sans CJK KR", "Noto Sans CJK HK", "NanumGothic"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

_ROOT = Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data"
URDF = _ROOT / "robots" / "G1" / "urdf_pyroki" / "g1_shadow.urdf"
PROC = _ROOT / "processed" / "parahome"

# retarget_g1_pyroki.py 의 _BODY 와 동일해야 합니다 (여기서 검증하려는 대상).
BODY = [
    (0, "pelvis"), (4, "torso_link"),
    (8, "right_shoulder_pitch_link"), (9, "right_elbow_link"), (10, "right_wrist_yaw_link"),
    (12, "left_shoulder_pitch_link"), (13, "left_elbow_link"), (14, "left_wrist_yaw_link"),
    (15, "right_hip_pitch_link"), (16, "right_knee_link"), (17, "right_ankle_roll_link"),
    (19, "left_hip_pitch_link"), (20, "left_knee_link"), (21, "left_ankle_roll_link"),
]


def load(clip: str, cls: str):
    rp = PROC / "g1_shadow" / cls / clip / "0" / "trajectory_pyroki.npz"
    hp = PROC / "smplx" / cls / clip / "0" / "trajectory.npz"
    r, h = np.load(rp, allow_pickle=True), np.load(hp, allow_pickle=True)
    return r["g1_joint_pos"], r["g1_root_pose"], [str(x) for x in r["joint_names"]], \
        h["joint_positions"].astype(np.float64)


def robot_positions(model, data, qidx, jn, qpos, root, links):
    """(F, n_links, 3) 리타게팅 자세에서의 링크 원점 위치 (world)."""
    fids = [model.getFrameId(l) for l in links]
    cols = [(qidx[n], i) for i, n in enumerate(jn) if n in qidx]
    F = len(qpos)
    out = np.zeros((F, len(links), 3))
    for f in range(F):
        q = pin.neutral(model)
        q[0:3] = root[f, :3]
        w, x, y, z = root[f, 3:7]                       # npz wxyz -> pinocchio xyzw
        q[3:7] = [x, y, z, w]
        for qi, ci in cols:
            q[qi] = qpos[f, ci]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for k, fid in enumerate(fids):
            out[f, k] = data.oMf[fid].translation
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", required=True)
    ap.add_argument("--class", dest="cls", default="single_rigid")
    ap.add_argument("--out_prefix", default="kpt_mapping")
    ap.add_argument("--frames", type=int, default=3, help="쌍 그림에 넣을 프레임 수")
    ap.add_argument("--dpi", type=int, default=160)
    a = ap.parse_args()

    model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
    data = model.createData()
    qidx = {model.names[j]: model.idx_qs[j] for j in range(1, model.njoints) if model.nqs[j] == 1}
    links = [l for _, l in BODY]
    hidx = [i for i, _ in BODY]

    store = {}
    for clip in a.clip:
        qpos, root, jn, jp = load(clip, a.cls)
        n = min(len(qpos), len(jp))
        rob = robot_positions(model, data, qidx, jn, qpos[:n], root[:n], links)
        hum = jp[:n][:, hidx]                            # (F, 14, 3)
        d = np.linalg.norm(rob - hum, axis=-1) * 100     # cm
        store[clip] = (hum, rob, d)
        print(f"\n[{clip}] F={n}")
        print(f"  {'사람idx -> 로봇링크':44s}{'거리 중앙cm':>12s}{'p95':>8s}{'최대':>8s}")
        for k, (i, l) in enumerate(BODY):
            print(f"    {i:2d} -> {l:36s}{np.median(d[:, k]):11.1f}{np.percentile(d[:, k], 95):8.1f}"
                  f"{d[:, k].max():8.1f}")

    # ── 그림 1: 대응 쌍 투영 ──────────────────────────────────────────────────────────────
    clip0 = a.clip[0]
    hum, rob, d = store[clip0]
    F = len(hum)
    fsel = np.linspace(F // 6, F - F // 6, a.frames, dtype=int)
    fig, axes = plt.subplots(2, len(fsel), figsize=(5.0 * len(fsel), 9.5))
    axes = np.atleast_2d(axes)
    for c, f in enumerate(fsel):
        for r, (ax, (h_i, v_i), nm) in enumerate(
                zip(axes[:, c], ((0, 2), (1, 2)), ("측면 (X-Z)", "정면 (Y-Z)"))):
            ax.plot(hum[f, :, h_i], hum[f, :, v_i], "o", ms=7, color="#27ae60", label="사람 키포인트")
            ax.plot(rob[f, :, h_i], rob[f, :, v_i], "s", ms=6, color="#1f6f8b", label="로봇 링크 원점")
            for k, (i, l) in enumerate(BODY):
                err = d[f, k]
                ax.plot([hum[f, k, h_i], rob[f, k, h_i]], [hum[f, k, v_i], rob[f, k, v_i]],
                        "-", lw=0.8 + 2.2 * min(err / 30.0, 1.0),
                        color="#c0392b" if err > 10 else "#95a5a6", zorder=0)
                if r == 0 and err > 8:
                    ax.annotate(f"{i}→{l.replace('_joint','').replace('_link','')}\n{err:.0f}cm",
                                (hum[f, k, h_i], hum[f, k, v_i]), fontsize=6,
                                xytext=(4, 4), textcoords="offset points", color="#c0392b")
            ax.set_aspect("equal")
            ax.grid(alpha=0.25, lw=0.4)
            ax.set_title(f"frame {f}  {nm}\n최대 오차 {d[f].max():.0f}cm", fontsize=9)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{clip0} — SMPL 키포인트 ↔ 로봇 링크 대응 (빨간 선 = 오차 10cm 초과)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p1 = f"{a.out_prefix}_pairs.png"
    fig.savefig(p1, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)

    # ── 그림 2: 쌍별 거리 막대 ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6.5))
    idx = np.arange(len(BODY))
    w = 0.8 / len(store)
    for s, (clip, (_, _, d)) in enumerate(store.items()):
        ax.bar(idx + s * w - 0.4 + w / 2, np.median(d, axis=0), width=w * 0.9,
               label=f"{clip} (중앙)")
        ax.plot(idx + s * w - 0.4 + w / 2, np.percentile(d, 95, axis=0), "k_", ms=10)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{i}→{l.replace('_link','')}" for i, l in BODY],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("사람 키포인트 ↔ 로봇 링크 거리 (cm)", fontsize=10)
    ax.axhline(10, color="#c0392b", ls=":", lw=1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("대응 쌍별 잔차 (막대=중앙, 검은 눈금=p95, 점선=10cm)", fontsize=12)
    fig.tight_layout()
    p2 = f"{a.out_prefix}_dist.png"
    fig.savefig(p2, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\n-> {p1}\n-> {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
