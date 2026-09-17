"""Audit which Euler convention the wrist6 chain actually uses, and measure all 12 alternatives.

Three parts:
  1. Read the ACTUAL chain from `shadow_float6_{l,r}.usd` — parent/child links and each joint's
     `physics:axis` — so the convention is read off the asset, not assumed from variable names.
  2. Read the convention the exporter used, by checking which of the 12 decompositions reproduces
     the stored `wrist_dof6.npz` values from the reference pose in `wrist_ref.npz`.
  3. For each of the 12 conventions, measure what the reference would DEMAND of the joints:
       * det(J) — the angular Jacobian determinant = control authority along the degenerate
         direction. Judge singularity proximity by THIS, never by the middle angle in degrees:
         88.5 deg vs 84.1 deg looks like 5% but cos gives 0.026 vs 0.105, a 4x difference.
       * required joint rate — computed after exact nearest-branch selection (the theoretical best
         any unwrap can do), so the number reflects the chart's demand, not a conversion artifact.
       * physical |omega| from the quaternions, chart-free, as the baseline the demand is
         compared against. demand/physical is the chart's amplification factor.

    <env_isaaclab python> scripts/process_dataset/diagnostics/audit_euler_convention.py
"""

import argparse
import glob

import numpy as np
from scipy.spatial.transform import Rotation as Rot

_PROC = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/processed/parahome"
_ROBOTS = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/robots/G1"
REF_FPS = 30.0
AX = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}
TB = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]                # Tait-Bryan: det = cos(middle)
PE = ["XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ"]                # proper Euler: det = sin(middle)
SEQS = TB + PE

parser = argparse.ArgumentParser()
parser.add_argument("--focus", default="s101_seg12_knife", help="추가로 따로 보고할 클립")
parser.add_argument("--side", default="r", choices=("l", "r"))
args = parser.parse_args()


def read_chain(side):
    """USD 에서 실제 체인을 읽는다: 관절 순서(parent->child 추적)와 각 관절의 physics:axis."""
    from pxr import Usd, UsdPhysics
    st = Usd.Stage.Open(f"{_ROBOTS}/shadow_float6_{side}.usd")
    joints = {}
    for pr in st.Traverse():
        if not pr.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(pr)
        b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
        b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
        axa = pr.GetAttribute("physics:axis")
        joints[pr.GetName()] = dict(
            parent=b0[0].split("/")[-1] if b0 else None,
            child=b1[0].split("/")[-1] if b1 else None,
            axis=(axa.Get() if axa and axa.HasAuthoredValue() else None),
            type=str(pr.GetTypeName()))
    # 루트(anchor)에서 palm 까지 parent->child 로 걸어간다
    byparent = {}
    for n, d in joints.items():
        byparent.setdefault(d["parent"], []).append((n, d))
    cur, chain, seen = f"robot0_{side}_anchor", [], set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        nxt = None
        for n, d in byparent.get(cur, []):
            chain.append((n, d))
            nxt = d["child"]
            break
        cur = nxt
        if cur and "palm" in cur:
            break
    return chain


def det_of(seq, ang):
    """R = R_a1(t1) R_a2(t2) R_a3(t3) (내재 회전) 의 각속도 야코비안 행렬식."""
    a1, a2, a3 = [AX[c] for c in seq]
    R1 = Rot.from_rotvec(ang[:, [0]] * a1[None, :]).as_matrix()
    R2 = Rot.from_rotvec(ang[:, [1]] * a2[None, :]).as_matrix()
    J = np.stack([np.broadcast_to(a1, (len(ang), 3)), R1 @ a2, (R1 @ R2) @ a3], -1)
    return np.abs(np.linalg.det(J))


def perfect(seq, R):
    """정확한 두 branch x 2pi 이동 중 이전 프레임 최근접. 어떤 unwrap 도 이보다 나을 수 없다."""
    a = R.as_euler(seq)
    prop = seq[0] == seq[2]
    b = np.stack([a[:, 0] + np.pi, (-a[:, 1] if prop else np.pi - a[:, 1]), a[:, 2] + np.pi], 1)
    prev, out = a[0].copy(), [a[0].copy()]
    for i in range(1, len(a)):
        cs = [x + 2 * np.pi * np.round((prev - x) / (2 * np.pi)) for x in (a[i], b[i])]
        prev = min(cs, key=lambda t: np.abs(t - prev).max())
        out.append(prev)
    T = np.stack(out)
    err = np.degrees((Rot.from_euler(seq, T).inv() * R).magnitude()).max()
    return T, err


def load_refs():
    """(clip, side, Rot, quat) 목록."""
    out = []
    for f in sorted(glob.glob(f"{_PROC}/g1_shadow/*/*/0/wrist_ref.npz")):
        d = np.load(f, allow_pickle=True)
        clip = f.split("/")[-3]
        for s in "lr":
            q = d[f"wrist_pose_{s}"][:, 3:7].astype(np.float64)
            q = q[np.isfinite(q).all(1)]
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            out.append((clip, s, Rot.from_quat(q[:, [1, 2, 3, 0]])))
    return out


def main():
    # ── 1. 에셋의 실제 체인 ────────────────────────────────────────────────────────────
    print("=" * 78)
    print("1. 에셋에서 읽은 실제 체인 (shadow_float6_%s.usd)" % args.side)
    print("=" * 78)
    chain = read_chain(args.side)
    rot_axes = []
    for n, d in chain:
        t = d["type"].replace("Physics", "").replace("Joint", "")
        ax = d["axis"]
        print(f"  {n:26s} {t:10s} axis={str(ax):5s}  {d['parent']} -> {d['child']}")
        if t == "Revolute" and ax in ("X", "Y", "Z"):
            rot_axes.append(ax)
    asset_seq = "".join(rot_axes)
    print(f"\n  => 회전 축 순서 = {asset_seq}   (내재 회전 R = R_{asset_seq[0]} R_{asset_seq[1]} R_{asset_seq[2]})")

    # ── 2. exporter 가 쓴 규약 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("2. 저장된 wrist_dof6.npz 가 어느 규약으로 만들어졌나 (재구성 오차로 판정)")
    print("=" * 78)
    fr = glob.glob(f"{_PROC}/g1_shadow/*/{args.focus}/0/wrist_ref.npz")[0]
    fd = fr.replace("wrist_ref.npz", "wrist_dof6.npz")
    dref = np.load(fr, allow_pickle=True)
    ddof = np.load(fd, allow_pickle=True)
    q = dref[f"wrist_pose_{args.side}"][:, 3:7].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    Rr = Rot.from_quat(q[:, [1, 2, 3, 0]])
    stored = ddof[f"wrist_dof_{args.side}"][:, 3:6].astype(np.float64)
    print(f"  {'규약':>6s} | {'저장값을 그 규약으로 해석했을 때 pose 재구성 오차 (deg)':>52s}")
    best = None
    for seq in SEQS:
        e = np.degrees((Rot.from_euler(seq, stored).inv() * Rr).magnitude())
        mark = ""
        if best is None or e.max() < best[0]:
            best = (e.max(), seq)
        print(f"  {seq:>6s} | 중앙 {np.median(e):9.4f}   최대 {e.max():9.4f}{mark}")
    print(f"\n  => exporter 규약 = {best[1]}  (재구성 오차 최대 {best[0]:.2e} deg)")
    print(f"  => 에셋({asset_seq}) 과 exporter({best[1]}) 일치: "
          f"{'예' if asset_seq == best[1] else '아니오 ***불일치***'}")

    # ── 3. 12 규약 측정 ───────────────────────────────────────────────────────────────
    refs = load_refs()
    nF = sum(len(R) for _, _, R in refs)
    print("\n" + "=" * 78)
    print(f"3. 12 규약 측정 — 코퍼스 {len(refs)//2}클립 양손 {nF}프레임")
    print("=" * 78)
    wall = np.concatenate([(R[1:] * R[:-1].inv()).magnitude() * REF_FPS for _, _, R in refs])
    print(f"  물리 각속도 |w| (차트 무관): 중앙 {np.median(wall):.2f}  "
          f"p99 {np.quantile(wall,.99):.2f}  최대 {wall.max():.2f} rad/s\n")
    print(f"  {'규약':>6s} {'종류':>4s} | {'det최소':>8s} {'det<0.2':>8s} | "
          f"{'요구속도 p99':>11s} {'p99.9':>8s} {'최대':>8s} | {'재구성오차':>10s}")
    rows = []
    for seq in SEQS:
        dmin, dlow, rates, emax = 1.0, [], [], 0.0
        for _, _, R in refs:
            T, e = perfect(seq, R)
            emax = max(emax, e)
            c = det_of(seq, T)
            dmin = min(dmin, c.min())
            dlow.append(c < 0.2)
            rates.append(np.abs(np.diff(T, axis=0)).max(1) * REF_FPS)
        dlow = np.concatenate(dlow)
        rt = np.concatenate(rates)
        rows.append((seq, dmin, dlow.mean() * 100, np.quantile(rt, .99),
                     np.quantile(rt, .999), rt.max(), emax))
        print(f"  {seq:>6s} {'TB' if seq in TB else 'PE':>4s} | {dmin:8.4f} {dlow.mean()*100:7.1f}% | "
              f"{np.quantile(rt,.99):11.2f} {np.quantile(rt,.999):8.2f} {rt.max():8.2f} | "
              f"{emax:10.2e}")

    # ── 4. focus 클립 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"4. {args.focus} — 발산했던 케이스")
    print("=" * 78)
    for _, s, R in [r for r in refs if r[0] == args.focus]:
        w = ((R[1:] * R[:-1].inv()).magnitude() * REF_FPS).max()
        print(f"\n  {s}손  물리 |w| 최대 {w:.2f} rad/s")
        print(f"  {'규약':>6s} | {'det최소':>8s} {'det<0.2':>8s} | {'요구속도 최대':>12s} | {'증폭':>7s}")
        for seq in SEQS:
            T, _ = perfect(seq, R)
            c = det_of(seq, T)
            rt = (np.abs(np.diff(T, axis=0)).max(1) * REF_FPS).max()
            print(f"  {seq:>6s} | {c.min():8.4f} {(c<0.2).mean()*100:7.1f}% | {rt:12.2f} | "
                  f"{rt/max(w,1e-9):6.2f}x")

    print("\n" + "=" * 78)
    best_rate = min(rows, key=lambda r: r[5])
    best_det = max(rows, key=lambda r: r[1])
    print(f"  코퍼스 최선(det최소 기준): {best_det[0]}  det최소 {best_det[1]:.4f}, 노출 {best_det[2]:.1f}%")
    print(f"  코퍼스 최선(요구속도 기준): {best_rate[0]}  최대 {best_rate[5]:.2f} rad/s")
    print("=" * 78)


if __name__ == "__main__":
    main()
