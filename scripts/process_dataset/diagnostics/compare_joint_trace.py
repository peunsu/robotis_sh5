#!/usr/bin/env python3
"""joint_trace.npz 두 개를 나란히 비교합니다 (예: 전체 정책 vs z_res=0 순수 SONIC).

용도: SONIC 디코더 출력이 관절 한계를 넘는 원인이 (a) 프리어 입력이 분포 밖이라 프리어 자체가
극단값을 내는 것인지, (b) z_res 섭동이 pre-quantization latent 를 밀어낸 것인지 가르는 진단.

  a_sonic 포화 = a_sonic 이 "관절 한계가 함의하는 액션 범위"
                 [(q_lo - default)/scale, (q_hi - default)/scale] 를 벗어난 스텝 비율.
                 벗어나면 그 스텝의 관절 PD 타겟이 한계에 clamp 됩니다.

z_res=0 에서도 포화가 남으면 (a), 사라지면 (b) 입니다.

사용:
  python compare_joint_trace.py <A.npz> <B.npz> --label_a 전체정책 --label_b zres0 \\
      --out_prefix <경로/이름>
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager as _fm
for _f in ("Noto Sans CJK JP", "Noto Sans CJK KR", "Noto Sans CJK HK", "NanumGothic"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False


def sonic_range(d):
    """(a_lo, a_hi) — 관절 한계가 함의하는 a_sonic 범위. (29,) 각각."""
    lo, hi = d["ctrl_lower"][:29], d["ctrl_upper"][:29]
    sc, df = d["sonic_scale"], d["sonic_default"]
    nz = np.abs(sc) < 1e-9
    e1 = np.where(nz, np.inf, (lo - df) / np.where(nz, 1.0, sc))
    e2 = np.where(nz, np.inf, (hi - df) / np.where(nz, 1.0, sc))
    return np.minimum(e1, e2), np.maximum(e1, e2)


def summarize(d, tag):
    names = [str(x) for x in d["joint_names"]]
    asn = d["a_sonic"]
    a_lo, a_hi = sonic_range(d)
    sat = np.mean((asn < a_lo) | (asn > a_hi), axis=0)
    print(f"\n[{tag}] 스텝 {len(asn)}  a_sonic 포화 평균 {sat.mean() * 100:.1f}%  "
          f"|a_sonic| 최대 {np.abs(asn).max():.2f}")
    return names, asn, a_lo, a_hi, sat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=str)
    ap.add_argument("b", type=str)
    ap.add_argument("--label_a", type=str, default="A")
    ap.add_argument("--label_b", type=str, default="B")
    ap.add_argument("--out_prefix", type=str, default="cmp")
    ap.add_argument("--top", type=int, default=9, help="시계열로 그릴 관절 수")
    ap.add_argument("--dpi", type=int, default=160)
    a = ap.parse_args()

    A, B = np.load(a.a, allow_pickle=True), np.load(a.b, allow_pickle=True)
    for tag, d in ((a.label_a, A), (a.label_b, B)):
        if "a_sonic" not in d:
            print(f"[error] {tag}: a_sonic 없음 (rollout.py --dump_joints 로 다시 받아야 합니다)")
            return 1
    nmA, asA, loA, hiA, satA = summarize(A, a.label_a)
    nmB, asB, loB, hiB, satB = summarize(B, a.label_b)
    assert nmA[:29] == nmB[:29], "관절 순서가 다릅니다"
    names = nmA

    # ── 표: 포화율 변화 ────────────────────────────────────────────────────────────────────
    order = np.argsort(-np.maximum(satA, satB))
    print(f"\n{'joint':28s}{'한계함의범위':>18s}"
          f"{a.label_a + ' 포화':>14s}{a.label_b + ' 포화':>14s}"
          f"{a.label_a + ' |max|':>14s}{a.label_b + ' |max|':>14s}")
    for j in order[: a.top * 2]:
        print(f"  {names[j]:26s}[{loA[j]:+7.2f},{hiA[j]:+7.2f}]"
              f"{satA[j] * 100:13.1f}%{satB[j] * 100:13.1f}%"
              f"{np.abs(asA[:, j]).max():14.2f}{np.abs(asB[:, j]).max():14.2f}")

    # ── 그림: 포화 상위 관절의 a_sonic 시계열 겹쳐 그리기 ────────────────────────────────
    sel = list(order[: a.top])
    ncol = 3
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.5 * nrow))
    axf = np.atleast_1d(axes).ravel()
    fpsA, fpsB = float(A["control_fps"]), float(B["control_fps"])
    for ax, j in zip(axf, sel):
        tA, tB = np.arange(len(asA)) / fpsA, np.arange(len(asB)) / fpsB
        ax.plot(tA, asA[:, j], lw=0.9, color="#c0392b", label=a.label_a)
        ax.plot(tB, asB[:, j], lw=0.9, color="#1f6f8b", label=a.label_b)
        L, H = loA[j], hiA[j]
        if np.isfinite(L) and np.isfinite(H):
            ax.axhline(H, color="k", ls=":", lw=0.9)
            ax.axhline(L, color="k", ls=":", lw=0.9)
            ys = np.r_[asA[:, j], asB[:, j], L, H]
        else:
            ys = np.r_[asA[:, j], asB[:, j]]
        m = 0.04 * max(ys.max() - ys.min(), 1e-6)
        ax.set_ylim(ys.min() - m, ys.max() + m)
        ax.set_title(f"{names[j]}\n범위 [{L:+.2f}, {H:+.2f}]  포화 "
                     f"{a.label_a} {satA[j]:.0%} / {a.label_b} {satB[j]:.0%}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_xlabel("t (s)", fontsize=8)
    for ax in axf[len(sel):]:
        ax.axis("off")
    axf[0].legend(fontsize=8)
    fig.suptitle(f"SONIC 디코더 출력 a_sonic — {a.label_a} vs {a.label_b} "
                 f"(점선 = 관절 한계가 함의하는 액션 범위)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p1 = f"{a.out_prefix}_sonic_cmp.png"
    fig.savefig(p1, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)

    # ── 그림: 29관절 포화율 막대 비교 ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    idx = np.arange(29)
    ax.barh(idx - 0.2, satA * 100, height=0.4, color="#c0392b", label=a.label_a)
    ax.barh(idx + 0.2, satB * 100, height=0.4, color="#1f6f8b", label=a.label_b)
    ax.set_yticks(idx)
    ax.set_yticklabels(names[:29], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("a_sonic 이 관절 한계 함의 범위를 벗어난 스텝 비율 (%)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("SONIC 출력 포화율 — 몸통 29관절", fontsize=12)
    fig.tight_layout()
    p2 = f"{a.out_prefix}_sat_bar.png"
    fig.savefig(p2, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\n-> {p1}\n-> {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
