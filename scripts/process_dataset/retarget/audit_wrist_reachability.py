"""Audit whether the G1 ARM can reach a wrist trajectory — the gate that replaces wrist anchoring.

Why this exists
---------------
Stage 1 lets each hand float free: the policy pushes the wrist with an impedance controller and
nothing constrains the wrist to stay where a G1 arm could put it (the user's decision — no
anchoring). Stage 2 then asks the full-body policy to TRACK that wrist trajectory with a real
7-DoF arm. If stage 1 wandered outside the arm's reachable set, stage 2 is being handed an
impossible target and no amount of training fixes it.

So: solve arm-only IK against the wrist trajectory, per frame, and report the residual. A large
residual fraction means the free-floating wrist needs a bound on `pos_offset` after all.

What is held fixed
------------------
Everything except the 7 arm joints of the side under test is pinned to the RETARGETED pose for
that frame — pelvis, legs, WAIST, and the other arm. That is exactly stage 2's situation: SONIC
owns the body and the residual policy moves the arm.

Pinning the waist matters and is easy to get wrong. `robot0_{s}_palm` hangs off the torso, so its
pose depends on `waist_{yaw,roll,pitch}_joint` as well as the 7 arm joints. Leaving the waist at
NEUTRAL while the retargeting had the torso twisted made a provably-reachable trajectory look 92%
unreachable (measured on `s100_seg02_kettle`) — the residual was the torso displacement, not an
arm limit. Random IK restarts did not fix it (92.2% -> 87.7%), which is what identified the cause:
a bad seed is recoverable, a wrong kinematic chain is not.

Two trajectories can be audited:
    --source retarget   <clip>/0/wrist_ref.npz   (the retargeted wrist — a sanity floor, since
                        the retargeting solved the whole body together, so it must be reachable)
    --source rl         <clip>/0/hand_rl.npz     (stage 1's actual output — the real question;
                        needs a trained checkpoint + export_hand_rl.py first)

Usage (CPU only)
    <env_isaaclab python> scripts/process_dataset/retarget/audit_wrist_reachability.py
    <env_isaaclab python> scripts/process_dataset/retarget/audit_wrist_reachability.py \
        --source rl --clip s101_seg12_knife
"""

from __future__ import annotations

import argparse
from pathlib import Path

import math

import numpy as np
import pinocchio as pin

_ROOT = Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data"
_PROC = _ROOT / "processed" / "parahome"
_URDF = _ROOT / "robots" / "G1" / "urdf_pyroki" / "g1_shadow_nomimic.urdf"

_ARM = {s: [f"{s}_shoulder_pitch_joint", f"{s}_shoulder_roll_joint", f"{s}_shoulder_yaw_joint",
            f"{s}_elbow_joint", f"{s}_wrist_roll_joint", f"{s}_wrist_pitch_joint",
            f"{s}_wrist_yaw_joint"]
        for s in ("left", "right")}
_SIDE = {"l": "left", "r": "right"}
# Position tolerance for "reachable". 2 cm is the scale the stage-2 tracking reward operates at
# (rew_ee_kpts on the palm keypoint); anything the arm cannot close to within 2 cm is a target the
# full-body policy provably cannot hit.
_TOL_M = 0.02
_ITERS = 120
_DAMP = 1e-6
# 재시작 횟수. 감쇠최소자승 IK 는 국소최소에 빠지고, 프레임을 이어 풀면 그 나쁜 해가 씨앗으로
# 계속 전파됩니다. 리타게팅 손목 궤적은 정의상 도달 가능(같은 팔로 풀린 해)이므로, 그 경로에서
# "도달 불가" 가 나오면 그건 궤적이 아니라 풀이의 실패입니다 — 실측: 재시작 없이 s100_seg02_kettle
# 오른손이 92.2% 초과로 나왔는데, 아래 재시작을 넣으면 0.0% 가 됩니다.
_RESTARTS = 8


class _Solver:
    """Damped-least-squares position+orientation IK on the arm joints only."""

    def __init__(self, side: str, joint_names: list[str]):
        self.model = pin.buildModelFromUrdf(str(_URDF), pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        m = self.model
        # 떠 있는 손의 root 가 palm 이라 목표 프레임도 palm 입니다 (2026-09-07 wrist 제거).
        self.fid = m.getFrameId(f"robot0_{side}_palm")
        # npz 열 -> q 인덱스. 팔 7개를 뺀 나머지 전부를 리타게팅 값으로 고정하는 데 씁니다
        # (허리 3개가 여기 포함되는 것이 핵심 — 손목이 몸통에 달려 있습니다).
        _qidx = {m.names[j]: m.idx_qs[j] for j in range(1, m.njoints) if m.nqs[j] == 1}
        _free = set(_ARM[_SIDE[side]])
        self.pin_cols = [(_qidx[n], i) for i, n in enumerate(joint_names)
                         if n in _qidx and n not in _free]
        # q index + velocity index per arm joint: the Jacobian columns live in v-space, the
        # configuration entries in q-space, and for a free-flyer model the two differ (7 vs 6).
        self.qi, self.vi = [], []
        for n in _ARM[_SIDE[side]]:
            j = m.getJointId(n)
            assert m.nqs[j] == 1, f"{n} 은 1-DoF 가 아닙니다"
            self.qi.append(m.idx_qs[j])
            self.vi.append(m.idx_vs[j])
        self.qi = np.array(self.qi)
        self.vi = np.array(self.vi)
        self.lo = np.array([m.lowerPositionLimit[i] for i in self.qi])
        self.hi = np.array([m.upperPositionLimit[i] for i in self.qi])

    def solve(self, root_pose: np.ndarray, target: np.ndarray, q_body: np.ndarray,
              q_seed: np.ndarray | None, rng: np.random.Generator):
        """root_pose (7,) pos+quat wxyz; target (7,) pos+quat wxyz. Returns (pos_err, rot_err, q).

        Tries the previous frame's solution first (continuity), then neutral, then random
        configurations, and keeps the best. Stops as soon as one lands inside tolerance.
        """
        best = (float("nan"), float("nan"), None)
        seeds = [q_seed, None] + [rng.uniform(self.lo, self.hi) for _ in range(_RESTARTS)]
        for sd in seeds:
            pe, re, q = self._solve_one(root_pose, target, q_body, sd)
            # NaN 은 비교에서 항상 False 라 `pe < best[0]` 만 쓰면 best 가 갱신되지 않고
            # 초기 센티넬이 그대로 반환됩니다. 예전 구현은 센티넬이 1e9 여서 그게
            # "1e12 mm 오차" 로 인쇄됐습니다 — 풀이 실패가 거대한 거리로 위장된 셈입니다.
            # 이제 NaN 을 그대로 흘려보내고 호출부가 따로 집계합니다.
            if math.isfinite(pe) and (not math.isfinite(best[0]) or pe < best[0]):
                best = (pe, re, q)
            if math.isfinite(pe) and pe < _TOL_M * 0.5:   # 충분히 좋으면 더 시도하지 않습니다
                break
        return best

    def _solve_one(self, root_pose: np.ndarray, target: np.ndarray, q_body: np.ndarray, q_seed):
        m, d = self.model, self.data
        q = pin.neutral(m)
        for qi, ci in self.pin_cols:                 # 팔 외 전신을 리타게팅 자세로 고정
            q[qi] = float(q_body[ci])
        q[0:3] = np.asarray(root_pose[:3], dtype=np.float64)
        w, x, y, z = (float(v) for v in root_pose[3:7])
        q[3:7] = [x, y, z, w]
        if q_seed is not None:
            q[self.qi] = q_seed
        # pinocchio 의 boost.python 바인딩은 float32 를 받지 않습니다 (npz 는 float32).
        tw, tx, ty, tz = (float(v) for v in target[3:7])
        oMdes = pin.SE3(pin.Quaternion(tw, tx, ty, tz).matrix(),
                        np.asarray(target[:3], dtype=np.float64))
        for _ in range(_ITERS):
            pin.forwardKinematics(m, d, q)
            pin.updateFramePlacements(m, d)
            iMd = d.oMf[self.fid].actInv(oMdes)
            err = pin.log(iMd).vector                                     # (6,) in the frame
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                break
            J = pin.computeFrameJacobian(m, d, q, self.fid, pin.LOCAL)     # (6,nv)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)[:, self.vi]           # (6,7) arm columns
            dv = -J.T @ np.linalg.solve(J @ J.T + _DAMP * np.eye(6), err)
            q[self.qi] = np.clip(q[self.qi] + dv, self.lo, self.hi)
        pin.forwardKinematics(m, d, q)
        pin.updateFramePlacements(m, d)
        T = d.oMf[self.fid]
        pos_err = float(np.linalg.norm(T.translation - oMdes.translation))
        rot_err = float(np.linalg.norm(pin.log3(T.rotation.T @ oMdes.rotation)))
        return pos_err, rot_err, q[self.qi].copy()


def audit(clip: str, cls: str, source: str) -> dict | None:
    cd = _PROC / "g1_shadow" / cls / clip / "0"
    fn = "wrist_ref.npz" if source == "retarget" else "hand_rl.npz"
    src, rt = cd / fn, cd / "trajectory_pyroki.npz"
    if not src.exists() or not rt.exists():
        print(f"  {clip}: {fn} 또는 trajectory_pyroki.npz 없음 — 건너뜀")
        return None
    w = np.load(src, allow_pickle=True)
    _d = np.load(rt, allow_pickle=True)
    root = _d["g1_root_pose"]                                            # (F,7) pelvis
    qbody = _d["g1_joint_pos"]                                           # (F,73) 리타게팅 전신
    jnames = [str(x) for x in _d["joint_names"]]
    # 프레임 레이트 정렬. hand_rl.npz 는 env 의 control_fps(50 Hz)로 나오고 리타게팅 사이드카는
    # 소스 30 Hz 입니다 (501 vs 301). min() 으로 자르면 두 궤적이 시간축에서 어긋납니다 —
    # 실측: 정렬만으로 프레임 100 에서 13 cm 차이. 손목 프레임 f 를 몸통 프레임
    # round(f x (F_body-1)/(F_wrist-1)) 에 대응시킵니다.
    _n_body = len(root)
    out = {"clip": clip}
    for sd in ("l", "r"):
        wp = w[f"wrist_pose_{sd}"]                                       # (F,7)
        F = len(wp)
        # 손목 프레임 -> 몸통 프레임 대응 (레이트가 같으면 항등)
        bi = np.round(np.arange(F) * (_n_body - 1) / max(1, F - 1)).astype(int)
        S = _Solver(sd, jnames)
        pe = np.full(F, np.nan); re = np.full(F, np.nan); seed = None
        rng = np.random.default_rng(0)                 # 재현 가능하도록 고정 시드
        for f in range(F):
            pe[f], re[f], seed = S.solve(root[bi[f]], wp[f], qbody[bi[f]], seed, rng)
        ok = np.isfinite(pe)
        if not ok.any():
            out[sd] = dict(n=F, frac_out=float("nan"), p50=float("nan"), p90=float("nan"),
                           max=float("nan"), rot_p90=float("nan"), n_fail=int((~ok).sum()))
            continue
        # 풀이 실패(NaN)는 "도달 불가" 와 다른 사건이라 따로 셉니다. 통계는 성공한 프레임만.
        out[sd] = dict(n=F, frac_out=float((pe[ok] > _TOL_M).mean()),
                       p50=float(np.median(pe[ok])), p90=float(np.quantile(pe[ok], 0.9)),
                       max=float(pe[ok].max()), rot_p90=float(np.quantile(re[ok], 0.9)),
                       n_fail=int((~ok).sum()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="all")
    ap.add_argument("--class", dest="cls", default="single_rigid")
    ap.add_argument("--source", choices=("retarget", "rl"), default="retarget")
    a = ap.parse_args()
    root = _PROC / "g1_shadow" / a.cls
    clips = ([p.name for p in sorted(root.iterdir()) if p.is_dir()]
             if a.clip == "all" else [a.clip])
    print(f"[reach-audit] source={a.source}  허용오차={_TOL_M * 100:.0f}cm  클립 {len(clips)}개\n")
    print(f"{'clip':22s} {'손':3s} {'F':>5s} {'초과%':>7s} {'p50mm':>7s} "
          f"{'p90mm':>7s} {'maxmm':>7s} {'회전p90°':>9s} {'풀이실패':>9s}")
    rows = []
    for c in clips:
        r = audit(c, a.cls, a.source)
        if r is None:
            continue
        rows.append(r)
        for sd in ("l", "r"):
            v = r[sd]
            print(f"{c:22s} {sd:3s} {v['n']:5d} {v['frac_out'] * 100:6.1f}% "
                  f"{v['p50'] * 1000:7.1f} {v['p90'] * 1000:7.1f} {v['max'] * 1000:7.1f} "
                  f"{np.degrees(v['rot_p90']):9.1f} {v.get('n_fail', 0):9d}")
    if rows:
        fo = np.array([r[sd]["frac_out"] for r in rows for sd in ("l", "r")])
        p90 = np.array([r[sd]["p90"] for r in rows for sd in ("l", "r")])
        nf = sum(r[sd].get("n_fail", 0) for r in rows for sd in ("l", "r"))
        _m = np.isfinite(fo)
        if _m.any():
            print(f"\n[reach-audit] 전체: 초과 프레임 비율 평균 {fo[_m].mean() * 100:.1f}% "
                  f"(최악 {fo[_m].max() * 100:.1f}%), p90 오차 중앙 "
                  f"{np.median(p90[np.isfinite(p90)]) * 1000:.1f} mm")
        if nf:
            print(f"[reach-audit] IK 풀이 실패 {nf} 프레임 — 도달 불가와 다른 사건입니다. "
                  f"대상 pose 가 비정상(발산/비유한)인지 먼저 확인하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
