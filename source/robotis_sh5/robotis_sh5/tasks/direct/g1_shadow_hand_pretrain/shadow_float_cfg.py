"""ArticulationCfg for the two floating Shadow hands (stage-1 dexterous pretrain, hand_pretrain env).

Each hand hangs from a fixed anchor at the env origin through six wrist joints
(tx, ty, tz, rot1, rot2, rot3 — YZX) and is driven by joint PD, like DexMachina.
Assets: `scripts/process_dataset/assets/build_shadow_floating_usd.py --side both --wrist6`
→ `shadow_float6_{l,r}.usd`. The G1 `robot0_{s}_wrist` link (zero-DOF, 0.3 kg) is not part of
the hand; the wrist joints attach directly to `robot0_{s}_palm`.

Tendon prerequisite (do not change one without the other)
---------------------------------------------------------
`fixed_tendons_props` supplies the J0<->J1 loopback coupling force. It only works because the
source asset had the tendon-axis joint drives REMOVED (`fix_tendon_axis_drives.py`) — the extracted
USDs inherit that: J0 carries 0 drive attributes and only `physxTendon:*:gearing`, while J1 keeps
its 6 drive attributes. With the drive present the tendon fights it and the fingers diverge
(measured on the G1 asset: divergence at step 67, RFJ1 reaching 10.26 rad). Finger PD gains and
limits are copied from the G1 build's `shadow_fingers` actuator so stage-1 and stage-2 hands have
identical dynamics.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ROBOTS = Path(__file__).resolve().parents[4] / "data" / "robots" / "G1"

# Actuated finger joints, 18 per hand. Same regex family as the G1 build's `shadow_fingers`, minus
# the (l|r) alternation because each hand is its own articulation here.
_FINGER_EXPR = [
    "robot0_{s}_(FF|MF|RF|LF|TH)J[1-3]",
    "robot0_{s}_LFJ4",
    "robot0_{s}_THJ4",
    "robot0_{s}_THJ0",
]


# ══ [wrist6] 6-DoF 관절 손목 ═══════════════════════════════════════════════════════════════
# palm 앞에 관절 6개를 직렬로 붙이고 anchor 를 월드에 고정한다.
# 에셋: build_shadow_floating_usd.py --wrist6 (shadow_float6_{l,r}.usd, 링크 30 / 관절 29)
# 체인: anchor → tx(X) → ty(Y) → tz(Z) → rot1(Y) → rot2(Z) → rot3(X) → (fixed) → palm
#
# 회전 축 순서가 YZX 인 이유. 12개 오일러 규약을 16클립 양손 5638프레임으로 측정했더니
# DexMachina 가 쓰는 ZXY 가 **가장 나빴다** (det(J)최소 0.0267, det<0.2 노출 8.1%).
# YZX 는 0.1615 / 0.1% 다. s101_seg12_knife 오른손에서 ZXY 는 물리 각속도 7.51 rad/s 를
# 표현하려고 관절에 46.59 rad/s 를 요구하는데(6.21배 증폭) YZX 는 7.63 rad/s 다(1.02배).
# 재감사: scripts/process_dataset/diagnostics/audit_euler_convention.py
#
# 게인은 물체 없이 knife 클립으로 쓸어서 정한 값이다 (2026-09-08). 측정치:
#   양손 최대 13.55 / 12.40 mm, 2.61° / 3.86°, 손목 힘 최대 10.9 N / 3.7 N·m, effort 포화 0%
#   → (이미 제거된) 외력 방식의 학습 중 실측(Error/wrist_kpts 중앙 26 mm, Error/wrist_rot 중앙 24.4°) 대비
#     위치 2배 / 회전 6배 정확하고, 종료 게이트(150 mm / 43°) 에 12배 / 11배 여유
# damping 은 임계값 2√(kp·I) 의 0.54배(과소감쇠)다. 임계나 그 이상으로 올리면 나빠진다 —
# 실측: kd_t 를 임계의 1.4배로 두면 최대 오차가 6.09 → 8.86 mm.
# 속도 목표(set_joint_velocity_target)는 쓰지 않는다. 넣어도 5.72 vs 5.91 mm 로 차이가 없고
# (DexMachina 도 안 쓴다), 빼면 정책이 위치 목표 하나만 다루면 된다.
_WRIST6_TRANS_EXPR = "robot0_{s}_wrist_t[xyz]"
_WRIST6_ROT_EXPR = "robot0_{s}_wrist_rot[123]"
# 탐험(exploration)을 고려한 재튜닝 (2026-09-08). 이전 값은
# 1400/33, 80/1.37, effort 200/50 (.pre_w6gain.bak). 근거 — 1-DoF 모델(질량 0.681 kg, 관성
# 0.02 kg·m², 16 클립 레퍼런스 50 Hz, PPO 초기 std 0.37 의 가우시안 잔차, EMA 0.2, 200 Hz PD):
#   * 잔차 잡음이 만드는 손목 흔들림(std ≈ 4.7 mm / 3.4°)은 게인과 무관하다 — 관절이 EMA 보다
#     빨라 평활된 목표를 그대로 따라간다. 게인이 정하는 것은 그 흔들림이 만드는 **힘**이다.
#     막힌 손바닥이 탐험만으로 쌓을 수 있는 힘(kp·3σ): 1400 → 20 N, 1000 → 14 N, 700 → 9.6 N,
#     회전 80 → 14 N·m, 40 → 7 N·m. 물체 무게(1~15 N)와 비교하면 모두 크므로 힘 한계도 같이 내린다.
#   * 레퍼런스 추종에 필요한 힘은 관성뿐이다(중력 꺼짐): m·a p99 = 4.7 N, I·α p99 = 0.8 N·m.
#     물체(≤1.5 kg) 를 들 때 +15 N / +3 N·m. → effort 60 N / 10 N·m 면 포화 0% (std 1.0 에서도
#     병진 0%, 회전 1.2%) 이면서 탐험·레퍼런스 관통이 만드는 접촉력을 3~5배 낮게 막는다.
#   * kp_t 1400 → 1000: 결정론 추종 rms 2.5 → 3.5 mm (EMA 지연 제외), 잡음 힘 p95 8.5 → 6.3 N.
#     kp_r 80 → 40: 0.5 → 1.2°, 잡음 토크 p95 4.8 → 2.6 N·m. 1 kg 물체 처짐 10 mm / 2°(잔차로 보상).
#   * 감쇠비 0.55 유지 (임계 이상은 지연 kd·v/kp 가 커져 나빠진다 — 아래 실측과 일치).
#   위 수치는 EMA 0.2 기준(그때 값 1000/30, 40/1.0). **EMA 0.5 기준 재조정** (2026-09-08 사용자 요청):
#   평활이 약해져 같은 σ 에서 손목 흔들림이 4.7 → 7 mm / 3.4° → 5° 로 커지고 힘은 kp 에 비례해
#   약 2배가 되므로, 탐험 힘을 EMA 0.2 때 수준(p95 ≈ 10 N / 4 N·m, 막힌 손바닥 ≈ 14 N / 5.5 N·m)
#   으로 맞추려면 kp 를 700 / 25 로 내린다. 결정론 추종 rms 8.5 mm / 2.7° (EMA 지연 1 스텝 포함;
#   kp 1400 이어도 6.3 mm 라 게인 차이는 작다). 1 kg 물체 처짐 14 mm / 3.4°(잔차 ±40 mm / ±28° 로
#   보상). 회전은 ζ 0.8 (잡음 토크 −10%, 오차 +0.5°), 병진은 ζ 0.8 이 오차 8.5 → 11.4 mm 로 나빠져
#   0.55 유지. effort 는 사용자가 둔 100/100 그대로 (회전 100 N·m 은 사실상 무제한: 필요 토크
#   p99 0.8 N·m, kp 25 의 탐험 토크 최대 ≈ 13 N·m).
#   **EMA 1.0(평활 없음) 기준 재조정** (2026-09-08 사용자 요청, 잡음 시드 3개 평균·표준편차 ≤0.1):
#   목표 잡음이 백색(σ=0.37 → 15 mm / 10.6° 매 스텝)이라 관절 자체 동역학만 걸러 준다. 힘 ≈ kp×잡음:
#   병진 kp 1400 → p95 44 N(막힌 손바닥 40 N), 700 → 21.5 N, 500 → 15 N(11 N), 350 → 10.7 N(6.9 N).
#   결정론 rms 는 EMA 지연이 없어 1400/700/500/350 = 2.5/4.7/6.0/7.6 mm. 1 kg 처짐 500 → 20 mm,
#   350 → 28 mm(잔차 ±40 mm 의 70%) 라 500 을 택했다(EMA 0.5 때 700 보다 힘 p95 +4.5 N, 오차 −2.5 mm).
#   회전 kp 80 → p95 32 N·m, 25 → 9.2, 15 → 5.4 N·m(막힌 손 3.4), 10 → 3.6. 결정론 rms 25/15/10 =
#   1.8/2.4/3.0°, 1 kg 팬(0.15 m) 처짐 15 → 5.7°(잔차 ±28°). 15 · ζ 0.8 을 택했다.
#   병진 ζ 0.8 은 rms 6.0 → 9.3 mm 로 나빠져(지연 kd·v/kp) 0.55 유지, 회전은 ζ 0.8 이 잡음 −15%.
#   **EMA 0.2 를 잔차에만(wrist6_ema_on_residual=True) 기준 재조정** (2026-09-08 사용자 요청,
#   "처짐이 심하지 않게", 액션 ±1 clamp 반영, 시드 3개 평균·표준편차 ≤0.1):
#   레퍼런스가 지연 없이 통과하므로 결정론 rms 는 EMA 1.0 과 같고(1400/1000/700/500 =
#   2.5/3.5/4.7/6.0 mm), 잡음은 EMA 0.2 로 걸러져 흔들림 4.6 mm / 3.3°(σ 0.37) 로 kp 와 무관하다.
#   → 막힌 손바닥 힘(kp×3×흔들림) 만 kp 에 비례: 병진 1400/1000/700 = 19.6/13.7/9.5 N,
#     회전 80/60/40/25 = 14.2/10.1/6.9/4.3 N·m. std 가 2.1 까지 커져도 clamp 때문에
#     1400 → 47 N, 40 → 17 N·m 에서 묶인다(effort 100 의 절반 이하).
#   처짐(정적 하중/kp): 병진은 물체 무게가 그대로 실려 1 kg 당 1400 → 7.0 mm, 1000 → 9.8, 700 → 14,
#   500 → 19.6 mm 라 처짐을 우선해 1400 을 택했다(탐험 힘 +6 N 은 물체 무게 수준, 효과 100 N 여유).
#   회전은 1 kg·0.15 m = 1.5 N·m 에 80/60/40/25 = 1.1/1.4/2.1/3.4° 로 처짐이 이미 작으므로,
#   탐험 토크가 물체 토크의 10배(14 N·m)가 되는 80 대신 40(6.9 N·m, ζ 0.55) 을 택했다.
#   ζ: 병진 0.8 은 rms 2.5 → 4.7 mm 로 나빠지고 힘은 −8% 뿐이라 0.53 유지(kd 33 = 이전 값).
#   클립별 확인(knife / s100 pan / s101 pot / s101 bowl): knife 는 |v| p99 0.40 m/s 로 가장 느리고
#   나머지는 0.92~1.06 m/s 라 결정론 p95 가 700 에서 5.5 vs 12~13 mm, 1400 에서 3.0 vs 6.3~6.8 mm 로
#   벌어진다 — 빠른 클립이 1400 을 더 지지한다. 탐험 힘은 클립과 무관(막힌 손바닥 19.6/13.7/9.5 N).
#   물체 USD 질량은 네 개 모두 0.5 kg 이라 실제 처짐은 1400 에서 3.5 mm, 회전 40 에서 1.1°.
WRIST6_KP_TRANS, WRIST6_KD_TRANS = 1400.0, 33.0     # N/m, N·s/m   (임계 2√(kp·0.681)=61.8 의 0.53배)
WRIST6_KP_ROT, WRIST6_KD_ROT = 40.0, 1.0            # N·m/rad, N·m·s/rad (임계 2√(kp·0.02)=1.79 의 0.56배)
WRIST6_EFFORT_TRANS, WRIST6_EFFORT_ROT = 100.0, 100.0     # 접촉력 상한 겸 안전벽. 200/50 이던 때
# 백색잡음 잔차에서 178 N / 71.6 N·m (포화 11.2%), EMA 0.2 에서 28 N / 9.1 N·m 를 실측했다.


def shadow_float6_cfg(side: str, prim_path: str) -> ArticulationCfg:
    """[wrist6] 6-DoF 관절로 구동하는 Shadow 손. `side` 는 "l" 또는 "r"."""
    if side not in ("l", "r"):
        raise ValueError(f"side must be 'l' or 'r', got {side!r}")
    usd = _ROBOTS / f"shadow_float6_{side}.usd"
    if not usd.exists():
        raise FileNotFoundError(
            f"{usd} 없음 — scripts/process_dataset/assets/build_shadow_floating_usd.py "
            f"--side both --wrist6 를 먼저 실행하세요")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                # [wrist6] linear/angular_damping 을 주지 않는다. 관절 PD 가 자체 damping 을 갖고,
                # 링크 감쇠를 더하면 모든 링크에 속도 비례 항력이 걸려 관절 제어기가 그것과 싸운다.
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                # [wrist6] anchor 를 월드에 고정한다. 관절 6개는 anchor **기준 상대** 자세만
                # 정의하므로, (tx,ty,tz) 가 월드 위치를 뜻하려면 anchor 가 알려진 곳에 박혀
                # 있어야 한다. 자유로 두면 6-DoF 자세에 자유도가 12개가 되어 잉여가 되고 관절
                # 목표가 palm 자세를 결정하지 못한다. DexMachina 도 베이스를 고정한다.
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.2),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # anchor 가 env 원점에 박히고, 손은 관절값으로 실려 간다. 실제 초기 관절값은 env 가
            # 리타게팅 손목(wrist_dof6.npz)으로 덮어쓴다.
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "wrist_trans": ImplicitActuatorCfg(
                joint_names_expr=[_WRIST6_TRANS_EXPR.format(s=side)],
                stiffness=WRIST6_KP_TRANS,
                damping=WRIST6_KD_TRANS,
                effort_limit_sim=WRIST6_EFFORT_TRANS,
                velocity_limit_sim=5.0,
            ),
            "wrist_rot": ImplicitActuatorCfg(
                joint_names_expr=[_WRIST6_ROT_EXPR.format(s=side)],
                stiffness=WRIST6_KP_ROT,
                damping=WRIST6_KD_ROT,
                effort_limit_sim=WRIST6_EFFORT_ROT,
                velocity_limit_sim=20.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[e.format(s=side) for e in _FINGER_EXPR],
                velocity_limit_sim=15.0,
                effort_limit_sim=3.09,
                stiffness=1.0,
                damping=0.2,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


SHADOW_FLOAT6_L_CFG = shadow_float6_cfg("l", "/World/envs/env_.*/HandL")
SHADOW_FLOAT6_R_CFG = shadow_float6_cfg("r", "/World/envs/env_.*/HandR")
