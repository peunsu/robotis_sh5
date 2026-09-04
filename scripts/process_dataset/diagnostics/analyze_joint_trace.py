#!/usr/bin/env python3
"""rollout.py --dump_joints 가 남긴 joint_trace.npz 로 bang-bang 여부를 판정하고 그림을 냅니다.

bang-bang(온-오프 채터링) 판정 지표. 모두 PD 타겟(_residual_target)에 대해 계산합니다 —
타겟이 곧 정책이 내리는 명령이고, 실측 관절은 PD 게인에 눌려 항상 더 매끄럽기 때문입니다.

  reversal_hz   타겟 증분의 부호가 뒤집히는 횟수 / 초. 제어율의 절반(= 나이퀴스트,
                50 Hz 제어면 25 Hz)에 가까우면 매 스텝 방향을 바꾸는 것이고 이것이 채터링입니다.
  sat_frac      타겟이 관절 한계(ctrl_lower/upper)의 2% 안에 있는 스텝 비율. bang-bang 은
                보통 두 한계를 왕복하므로 이 값이 큽니다.
  bimodality    타겟을 정규화(한계 기준 0~1)했을 때 중앙 밴드(0.25~0.75)를 벗어난 비율.
                양 극단에만 머무르면 1 에 가깝고, 연속적으로 움직이면 낮습니다.
  step_p95      한 스텝 타겟 변화량의 95 퍼센타일 / 관절 가동범위. 계단형 명령의 크기.

레퍼런스(ref_joints)가 있으면 같은 지표를 사람 궤적에도 계산해 기준선으로 함께 냅니다 —
"관절이 자주 방향을 바꾼다"는 것 자체는 원래 동작에도 있으므로, 초과분만 문제입니다.

사용:
  python analyze_joint_trace.py <joint_trace.npz> --out_prefix <경로/이름>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 한글 라벨이 두부(□)로 깨지지 않도록 CJK 폰트를 우선 지정합니다. 없으면 기본값으로 둡니다.
from matplotlib import font_manager as _fm
for _f in ("Noto Sans CJK JP", "Noto Sans CJK KR", "Noto Sans CJK HK", "NanumGothic"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False


def metrics(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, fps: float) -> dict:
    """x (T,J) 궤적 -> 관절별 지표 dict of (J,)."""
    d = np.diff(x, axis=0)                                       # (T-1,J)
    rng = np.maximum(hi - lo, 1e-6)
    # 부호 반전: 유의미한 움직임만 센다 (가동범위의 0.1% 미만 증분은 잡음).
    sig = np.where(np.abs(d) > 1e-3 * rng, np.sign(d), 0.0)
    rev = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        s = sig[:, j]
        s = s[s != 0]
        if len(s) > 1:
            rev[j] = int(np.sum(s[1:] != s[:-1]))
    dur = max((len(x) - 1) / fps, 1e-6)
    nrm = np.clip((x - lo) / rng, 0.0, 1.0)
    return dict(
        reversal_hz=rev / dur,
        sat_frac=np.mean((nrm < 0.02) | (nrm > 0.98), axis=0),
        bimodality=np.mean((nrm < 0.25) | (nrm > 0.75), axis=0),
        step_p95=np.percentile(np.abs(d), 95, axis=0) / rng,
        travel=np.sum(np.abs(d), axis=0) / rng,                  # 총 이동거리 / 가동범위
    )


def _ylim_range_union(ax, y, lo_r, hi_r):
    """y축 기준 스케일 = 액션 범위(lo_r, hi_r). 데이터가 그 범위를 넘으면 축을 넓혀 숨기지 않습니다.

    범위로만 고정하면 포화가 심한 차원(예: waist_pitch 94%)의 패널이 통째로 비어 버립니다.
    경계는 점선으로 남으므로 "범위 대비 얼마나 넘었나"는 그대로 읽힙니다.
    """
    if not (np.isfinite(lo_r) and np.isfinite(hi_r)):
        lo_r, hi_r = -3.0, 3.0
    lo_y, hi_y = float(np.min(y)), float(np.max(y))
    L, H = min(lo_r, lo_y), max(hi_r, hi_y)
    m = 0.04 * max(H - L, 1e-6)
    ax.set_ylim(L - m, H + m)
    ax.axhline(hi_r, color="#c0392b", ls=":", lw=0.9)
    ax.axhline(lo_r, color="#c0392b", ls=":", lw=0.9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=str)
    ap.add_argument("--out_prefix", type=str, default="joint_trace")
    ap.add_argument("--top", type=int, default=16, help="표에 출력할 관절 수")
    ap.add_argument("--body_n", type=int, default=6, help="관절 그림에 넣을 몸통 관절 수")
    ap.add_argument("--hand_n", type=int, default=6, help="관절 그림에 넣을 손 관절 수")
    ap.add_argument("--act_n", type=int, default=6, help="액션 그림에 넣을 블록별 차원 수")
    ap.add_argument("--sonic_n", type=int, default=6,
                    help="SONIC 출력 그림에 넣을 관절 수 (포화 상위 + 변동 상위 각각)")
    ap.add_argument("--zres_clip", type=float, default=5.0,
                    help="z_res 액션 범위 (cfg.sonic_z_res_clip)")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--include", type=str, default="",
                    help="관절 그림에 반드시 넣을 관절 이름 (쉼표 구분). 채터링 순위와 무관하게 앞에 옵니다.")
    a = ap.parse_args()

    d = np.load(a.trace, allow_pickle=True)
    names = [str(x) for x in d["joint_names"]]
    tgt, qpos = d["target"], d["qpos"]
    lo, hi = d["ctrl_lower"], d["ctrl_upper"]
    fps = float(d["control_fps"])
    T, J = tgt.shape
    print(f"클립={str(d['clip'])}  스텝={T} ({T / fps:.2f}s @ {fps:.0f}Hz)  관절={J}")
    print(f"나이퀴스트 한계 = {fps / 2:.1f} Hz (매 스텝 방향 전환)")

    mt = metrics(tgt, lo, hi, fps)
    mq = metrics(qpos, lo, hi, fps)
    ref = None
    if "ref_joints" in d:
        rj = d["ref_joints"]
        fr = np.clip(d["frame"], 0, len(rj) - 1)
        if rj.shape[1] >= J:
            ref = metrics(rj[fr][:, :J], lo, hi, fps)

    # ── 표: 타겟 reversal_hz 상위 ────────────────────────────────────────────────────────
    order = np.argsort(-mt["reversal_hz"])
    print(f"\n{'joint':30s}{'rev_hz(타겟)':>13s}{'rev_hz(실측)':>13s}"
          f"{'rev_hz(ref)':>12s}{'sat':>7s}{'bimod':>7s}{'step_p95':>9s}")
    for j in order[: a.top]:
        r = f"{ref['reversal_hz'][j]:12.2f}" if ref is not None else f"{'-':>12s}"
        print(f"  {names[j]:28s}{mt['reversal_hz'][j]:13.2f}{mq['reversal_hz'][j]:13.2f}{r}"
              f"{mt['sat_frac'][j]:7.2f}{mt['bimodality'][j]:7.2f}{mt['step_p95'][j]:9.3f}")

    grp = {"body(29)": slice(0, 29), "hands(36)": slice(29, J)}
    print("\n그룹 평균")
    for g, sl in grp.items():
        r = f"{ref['reversal_hz'][sl].mean():8.2f}" if ref is not None else f"{'-':>8s}"
        print(f"  {g:12s} rev_hz 타겟 {mt['reversal_hz'][sl].mean():6.2f} / 실측 "
              f"{mq['reversal_hz'][sl].mean():6.2f} / ref {r}   "
              f"sat {mt['sat_frac'][sl].mean():.2f}  bimod {mt['bimodality'][sl].mean():.2f}  "
              f"step_p95 {mt['step_p95'][sl].mean():.3f}")

    # ── 그림 1: 관절 PD 타겟 vs 실측 (몸통 + 손 선별) ──────────────────────────────────
    # y축은 관절 한계(ctrl_lower/upper)로 고정합니다 — 패널마다 축이 달라지면 "얼마나 크게
    # 움직이는가"를 눈으로 비교할 수 없습니다. 레퍼런스는 그리지 않습니다(요청).
    t = np.arange(T) / fps
    forced = [names.index(x) for x in a.include.split(",") if x.strip() and x.strip() in names]
    b_sel = [j for j in order if j < 29 and j not in forced][: max(0, a.body_n - sum(1 for j in forced if j < 29))]
    h_sel = [j for j in order if j >= 29 and j not in forced][: max(0, a.hand_n - sum(1 for j in forced if j >= 29))]
    sel = ([j for j in forced if j < 29] + b_sel
           + [j for j in forced if j >= 29] + h_sel)
    ncol = 3
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.5 * nrow), sharex=True)
    axf = np.atleast_1d(axes).ravel()
    for ax, j in zip(axf, sel):
        band = 0.02 * (hi[j] - lo[j])
        ax.axhspan(lo[j], lo[j] + band, color="0.88", zorder=0)
        ax.axhspan(hi[j] - band, hi[j], color="0.88", zorder=0)
        ax.plot(t, tgt[:, j], lw=0.9, color="#c0392b", label="PD 타겟")
        ax.plot(t, qpos[:, j], lw=1.3, color="#2c3e50", label="실측")
        ax.set_ylim(lo[j], hi[j])                       # 관절 한계로 스케일링
        grp = "몸통" if j < 29 else "손"
        ax.set_title(f"[{grp}] {names[j]}\nrev {mt['reversal_hz'][j]:.1f} Hz  "
                     f"sat {mt['sat_frac'][j]:.2f}  한계 [{lo[j]:+.2f}, {hi[j]:+.2f}]", fontsize=8)
        ax.set_ylabel("rad", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axf[len(sel):]:
        ax.axis("off")
    axf[0].legend(fontsize=8, loc="lower right", framealpha=0.9)
    for ax in axf[max(0, len(sel) - ncol):len(sel)]:
        ax.set_xlabel("t (s)", fontsize=8)
    fig.suptitle(f"{str(d['clip'])} — 관절 PD 타겟 vs 실측 (y축 = 관절 한계)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p1 = f"{a.out_prefix}_joints.png"
    fig.savefig(p1, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)

    # ── 그림 2: 액션 출력 ─────────────────────────────────────────────────────────────────
    # 레이아웃: [0:64] z_res (환경 클립 ±sonic_z_res_clip), [64:100] 손 (클램프 ±1).
    # 기록값은 _cur_policy_action = 원시(클립 전) 액션이라 범위를 넘을 수 있습니다. y축은 요청대로
    # 액션 범위로 고정하고, 넘긴 비율은 패널 제목에 적어 정보가 숨지 않게 합니다.
    act = d["action"]
    p2 = None
    if act.size:
        NZ = act.shape[1] - (J - 29)                    # z_res 폭 (100-36=64)
        rngs = np.r_[np.full(NZ, a.zres_clip), np.full(act.shape[1] - NZ, 1.0)]
        # 선별: 블록별로 변동이 큰 순
        sd = act.std(axis=0)
        z_sel = list(np.argsort(-sd[:NZ])[: a.act_n])
        h_sel_a = list(NZ + np.argsort(-sd[NZ:])[: a.act_n])
        asel = z_sel + h_sel_a
        nrow = int(np.ceil(len(asel) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.5 * nrow), sharex=True)
        axf = np.atleast_1d(axes).ravel()
        for ax, i in zip(axf, asel):
            R = rngs[i]
            ax.plot(t, act[:, i], lw=0.9, color="#1f6f8b")
            _ylim_range_union(ax, act[:, i], -R, R)     # 기준 = 액션 범위 ±R (점선)
            over = float(np.mean(np.abs(act[:, i]) > R))
            if i < NZ:
                lbl = f"[z_res] a[{i}]   범위 ±{R:.0f}"
            else:
                jn = names[29 + (i - NZ)] if 29 + (i - NZ) < J else f"hand[{i - NZ}]"
                lbl = f"[손] a[{i}] → {jn}   범위 ±{R:.0f}"
            ax.set_title(f"{lbl}\nstd {sd[i]:.3f}  |max| {np.abs(act[:, i]).max():.2f}"
                         + (f"  범위초과 {over:.0%}" if over > 0 else ""), fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, lw=0.4)
        for ax in axf[len(asel):]:
            ax.axis("off")
        for ax in axf[max(0, len(asel) - ncol):len(asel)]:
            ax.set_xlabel("t (s)", fontsize=8)
        fig.suptitle(f"{str(d['clip'])} — 정책 액션 출력 (y축 = 액션 범위, 점선 = 환경 클립)",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        p2 = f"{a.out_prefix}_actions.png"
        fig.savefig(p2, dpi=a.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"\n액션 블록별 범위초과 비율: z_res(±{a.zres_clip:.0f}) "
              f"{np.mean(np.abs(act[:, :NZ]) > a.zres_clip):.3f}   "
              f"손(±1) {np.mean(np.abs(act[:, NZ:]) > 1.0):.3f}")

    # ── 그림 2b: SONIC 디코더 출력 (몸통 29-D 액션) ──────────────────────────────────────
    # a_sonic 은 env 에서 클립되지 않습니다. 대신 body_target = sonic_default + sonic_scale·a_sonic
    # 뒤 관절 한계로 clamp 되므로, "관절 한계가 함의하는 액션 범위"
    #   [(lo - default)/scale, (hi - default)/scale]
    # 를 넘어가면 그 스텝의 관절 타겟이 한계에 포화합니다. y축을 그 범위로 잡습니다(요청).
    p2b = None
    if "a_sonic" in d and "sonic_scale" in d:
        asn = d["a_sonic"]
        sc, df = d["sonic_scale"], d["sonic_default"]
        nz = np.abs(sc) < 1e-9
        e1 = np.where(nz, np.inf, (lo[:29] - df) / np.where(nz, 1.0, sc))
        e2 = np.where(nz, np.inf, (hi[:29] - df) / np.where(nz, 1.0, sc))
        a_lo, a_hi = np.minimum(e1, e2), np.maximum(e1, e2)
        # 포화 비율 = 액션이 한계 함의 범위를 벗어난 스텝 비율
        satf = np.mean((asn < a_lo) | (asn > a_hi), axis=0)
        rev_s = metrics(asn, a_lo, a_hi, fps)["reversal_hz"]
        print("\n=== SONIC 디코더 출력 (29-D 몸통 액션) ===")
        print(f"{'joint':30s}{'한계함의범위':>18s}{'a_sonic 범위':>18s}{'포화%':>8s}{'rev_hz':>8s}")
        for j in np.argsort(-satf)[: a.sonic_n * 2]:
            print(f"  {names[j]:28s}[{a_lo[j]:+7.2f},{a_hi[j]:+7.2f}]"
                  f"[{asn[:, j].min():+7.2f},{asn[:, j].max():+7.2f}]{satf[j] * 100:7.1f}%{rev_s[j]:8.1f}")
        print(f"  전체: 포화 평균 {satf.mean() * 100:.1f}%   "
              f"|a_sonic| 최대 {np.abs(asn).max():.2f}   rev_hz 평균 {rev_s.mean():.1f}")

        s_sel = list(np.argsort(-satf)[: a.sonic_n])
        s_sel += [j for j in np.argsort(-asn.std(0)) if j not in s_sel][: a.sonic_n]
        nrow = int(np.ceil(len(s_sel) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.5 * nrow), sharex=True)
        axf = np.atleast_1d(axes).ravel()
        for ax, j in zip(axf, s_sel):
            L, H = a_lo[j], a_hi[j]
            ax.plot(t, asn[:, j], lw=0.9, color="#7d3c98")
            _ylim_range_union(ax, asn[:, j], L, H)       # 기준 = 한계 함의 액션 범위 (점선)
            ax.set_title(f"[SONIC] {names[j]}\n범위 [{L:+.2f}, {H:+.2f}]  "
                         f"포화 {satf[j]:.0%}  rev {rev_s[j]:.1f} Hz  "
                         f"|max| {np.abs(asn[:, j]).max():.2f}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, lw=0.4)
        for ax in axf[len(s_sel):]:
            ax.axis("off")
        for ax in axf[max(0, len(s_sel) - ncol):len(s_sel)]:
            ax.set_xlabel("t (s)", fontsize=8)
        fig.suptitle(f"{str(d['clip'])} — SONIC 디코더 출력 a_sonic "
                     f"(y축 = 관절 한계가 함의하는 액션 범위, 점선 = 그 경계)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        p2b = f"{a.out_prefix}_sonic.png"
        fig.savefig(p2b, dpi=a.dpi, bbox_inches="tight")
        plt.close(fig)

    # ── 그림 3: 관절별 지표 (레퍼런스 제외) ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, max(7, J * 0.19)))
    idx = np.arange(J)
    for ax, key, lbl in zip(axes,
                            ("reversal_hz", "sat_frac", "step_p95"),
                            (f"방향 전환 (Hz)  — 나이퀴스트 {fps / 2:.0f}",
                             "한계 포화 비율", "스텝 변화량 p95 / 가동범위")):
        ax.barh(idx - 0.2, mt[key], height=0.4, color="#c0392b", label="타겟")
        ax.barh(idx + 0.2, mq[key], height=0.4, color="#2c3e50", label="실측")
        if key == "reversal_hz":
            ax.axvline(fps / 2, color="k", ls=":", lw=1)
        ax.axhline(28.5, color="#27ae60", lw=1.2)        # 몸통 29 / 손 36 경계
        ax.set_yticks(idx)
        ax.set_yticklabels(names, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel(lbl, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{str(d['clip'])} — 관절별 지표 (초록선 위 = 몸통 29, 아래 = 손 36)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    p3 = f"{a.out_prefix}_metrics.png"
    fig.savefig(p3, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\n-> {p1}")
    if p2:
        print(f"-> {p2}")
    if p2b:
        print(f"-> {p2b}")
    print(f"-> {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
