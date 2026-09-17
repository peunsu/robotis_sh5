"""Config for the floating-hand dexterous pretrain (stage 1 of the two-stage loco-manip plan).

Relationship to the train cfg
-----------------------------
This SUBCLASSES `G1ShadowSonicResidualEnvCfg` instead of forking its 1415 lines. Two reasons:

* The user's brief was "copy the train env and remove only the full-body parts, do not redesign
  rewards/observations". For the ~31 body-related cfg FIELDS, setting the weight to 0.0 removes the
  term's effect exactly, while keeping the attribute present so none of the 3493 lines of env code
  that reads it raises AttributeError. Deleting the fields would force edits at every read site —
  more churn, more risk, same behaviour.
* The train cfg is the file the RUNNING training job is configured from. Subclassing leaves it
  untouched.

What is neutralised here (the "body" terms)
-------------------------------------------
    rew_body_kpts            9 core body keypoints (pelvis/shoulders/elbows/hips/knees)
    rew_root_pos/_ori        G1 pelvis pose tracking — no pelvis exists in a floating hand
    rew_com_support          whole-body balance over the foot support polygon
    rew_feet_contact_match   foot contact schedule matching
    term_root_pos/rot_err    fall/tilt substitute gates keyed on the pelvis
    <foot terminations>      foot-based gates

What is deliberately KEPT
-------------------------
    alive, link_kpts, hand_kpts, fingertip, obj_pos, obj_rot, contact_force, contact_cws,
    action_reg, pose_reg_hands, action_rate, and `rew_ee_kpts` — but `rew_ee_kpts` averages
    wrist×2 + ankle×2, so the ENV must narrow `_ee_kpt_idx` to the two wrists. That is an env-side
    index change, not a cfg weight, and is listed in the Phase-B change list.

    The foot-contact SCHEDULE (`foot_plant_h`/`foot_plant_v` → `_ref_foot_contact`) is left alone:
    it is computed from the SMPL-X foot joints into its own array and feeds only
    `rew_feet_contact_match` (now 0.0) and two obs slots, so it costs nothing to leave defined.

Not yet wired
-------------
`robot_cfg` still points at the composite G1. The floating-hand env replaces the single `Robot`
articulation with TWO articulations (`HandL`, `HandR`) built from `shadow_float_cfg.py`; that is the
env-side change that touches the 76 `self.robot` call sites and cannot be validated without the
GPU. See `PHASE_B_CHANGES.md` in this package.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..g1_shadow_sonic_residual.g1_shadow_sonic_residual_env_cfg import (
    G1ShadowSonicResidualEnvCfg,
)
from .shadow_float_cfg import (
    SHADOW_FLOAT6_L_CFG,
    SHADOW_FLOAT6_R_CFG,
    SHADOW_FLOAT_L_CFG,
    SHADOW_FLOAT_R_CFG,
)


# [wrist6] 관측에서 직전 액션을 뺀 나머지. observation_space = _OBS_BASE + action_space.
#   126(A 자기수용감각) + 252(B 추종) + 126(C 물체/접촉, 직전액션 제외) + 1(방향 비트) = 505
_OBS_BASE = 505
# 손별 액션 폭: wrench = pos3 + rot6d6 + finger18 = 27,  joint6 = wrist6 + finger18 = 24
_ACT_WRENCH = 2 * (3 + 6 + 18)      # 54
_ACT_JOINT6 = 2 * (6 + 18)          # 48


@configclass
class G1ShadowHandPretrainEnvCfg(G1ShadowSonicResidualEnvCfg):
    # ---- floating-hand assets (two articulations, one per hand) --------------------------------
    hand_l_cfg = SHADOW_FLOAT_L_CFG
    hand_r_cfg = SHADOW_FLOAT_R_CFG

    # ---- wrist impedance control (workspaceTJ shadow_hand_wristfree / gr_env) ------------------
    # F = K_pos * pos_offset * action_dt, tau = K_rot * axis_angle(R_6d(rot_offset)) * action_dt,
    # then a moving average; the D term is the rigid body's damping (100.0) in shadow_float_cfg.
    # K_pos / K_rot / the EMA alpha are TJ's values verbatim.
    wrist_k_pos: float = 4000.0
    wrist_k_rot: float = 160.0
    wrist_force_ema: float = 0.2          # α on the FORCE/TORQUE (TJ: global_moving_average)

    # action_dt is the CONTROL PERIOD, not a tuned constant: TJ's cfg defines it as
    # `action_dt = 1 / action_fps` with `action_fps = 30`. Our loop runs at 50 Hz
    # (sim.dt 1/200 x decimation 4), so the faithful port is 1/50 — hardcoding TJ's 1/30 would
    # be the deviation, not the match. Derived from decimation so it cannot drift if the control
    # rate changes: 1 / (200 * ... ) is expressed via the parent's sim.dt and decimation in
    # __post_init__ below.
    #
    # Consequence of 1/30 -> 1/50: peak force per axis at |action|=1 drops 4000/30 = 133.3 N to
    # 4000/50 = 80.0 N. The hand weighs 0.911 kg (24 links, measured from shadow_float_l.usd),
    # i.e. 8.94 N of gravity, so authority goes 14.9x -> 8.9x gravity. Still ample, and the policy
    # absorbs a gain change by emitting larger offsets; it would only bind at saturation.
    #
    # NOTE a third reading exists. If you wanted to preserve TJ's per-control-step DISPLACEMENT
    # (v = F/damping, so displacement per step = K*action_dt*offset/damping/fps), matching TJ at
    # 50 Hz needs action_dt = 1/18, not 1/50 — TJ's 0.0444 m per unit offset per step becomes
    # 0.0160 m at 1/50. We take the literal definition (action_dt = control period) because that
    # is what TJ's own cfg says the symbol means.
    wrist_action_dt: float = 1.0 / 50.0   # __post_init__ overwrites from sim.dt x decimation

    # ── 6D 회전 항등 오프셋 ────────────────────────────────────────────────────────────────
    # TJ 는 액션의 6열을 그대로 rotation_6d_to_matrix 에 넣습니다. 그 사상은 0 근방에서
    # 퇴화·불연속입니다 (측정, pytorch3d 0.7.9):
    #     [0,0,0,0,0,0]        -> det=0  (회전행렬이 아님), 축각 0
    #     [1e-6]*6             -> det=0, 축각 0.676 rad
    #     a[0]=+1e-5 -> 축각 0.0000 rad / a[0]=-1e-5 -> 축각 2.2214 rad
    # 마지막 줄이 문제입니다: 성분 하나의 부호가 바뀌는 것만으로 토크 지령이
    # 2.2214 * K_rot * action_dt = 11.8 N*m 만큼 튑니다. 정책의 "평균" 액션은 0 에서
    # 출발하고 export_hand_rl.py 의 결정론적 롤아웃은 바로 그 평균을 씁니다 — 즉 stage 2 가
    # 받을 궤적이 이 불연속점 위에서 만들어질 수 있습니다.
    # 그래서 변환 전에 항등 6D [1,0,0,0,1,0] 을 더합니다. 액션 0 -> 정확히 항등회전 -> 토크 0
    # 이 되고, 0 근방이 well-conditioned 해집니다(det=1). 나머지는 TJ 와 동일합니다.
    # 되돌리기: False (TJ 원문 그대로. 위 불연속을 그대로 물려받습니다).
    wrist_rot6d_identity: bool = True

    # ── [wrist6] 손목 구동 방식 ────────────────────────────────────────────────────────────
    # "wrench" = 기존. 자유 베이스(fix_root_link=False)에 외력·토크를 인가한다. 정책이 손목
    #            위치 오프셋 3 + 6D 회전 6 을 내고, env 가 F = K_pos·a·dt / τ = K_rot·axis_angle·dt
    #            로 바꿔 permanent_wrench_composer 로 넣는다. D 항은 강체의 linear/angular_damping.
    # "joint6" = palm 앞에 관절 6개(tx,ty,tz,rot1,rot2,rot3)를 직렬로 붙이고 anchor 를 고정한다
    #            (shadow_float6_{l,r}.usd). 정책이 관절 위치 잔차 6 을 내고 PD 가 구동한다.
    #            회전 규약은 YZX — 12규약 측정에서 ZXY(DexMachina)가 가장 나빴고 YZX 가 최선이다
    #            (det(J)최소 0.0267 → 0.1615, 노출 8.1% → 0.1%).
    # joint6 이 얻는 것: (a) effort_limit 이라는 하드 상한 — 외력 경로에는 상한이 없어 긴 rollout
    # 에서 발산했다. (b) 추종 정확도 — 물체 없이 측정한 knife 클립에서 최대 12.4 mm / 3.86° 로,
    # 외력 방식의 학습 중 실측(26 mm / 24.4°) 대비 위치 2배 / 회전 6배. (c) pytorch3d 의존성 제거
    # — rotation_6d_to_matrix / matrix_to_axis_angle 이 손목 법칙 한 곳에만 쓰인다.
    # 기본값 = "joint6" (2026-09-08 사용자 결정. 캐시 물체 열 수정 [cache-obj-layout] 이후 전환). 외력 경로는 전부 남아 있고
    # `--wrist_mode wrench` 로 언제든 되돌릴 수 있다.
    #
    # 롤백 시점의 측정 상태 — joint6 이 나빴다는 증거는 사실 없었다:
    #   손목 추종:  위치 19.9 vs 26.6 mm,  회전 0.277 vs 0.429 rad   (joint6 이 더 정확)
    #   obj_pos 게이트 누적 초과(무액션 200스텝): 18.8% vs 40.6%      (joint6 이 절반 이하)
    #   obj_rot 게이트:                          99.6% vs 87.5%      (joint6 이 약간 나쁨)
    # 다만 4000스텝 학습에서 death_frac 이 0.0576 vs 0.0037, 보상 23.8 vs 230.4 로 크게 나빴다.
    # 미규명 원인 두 개가 남아 있었다:
    #   (1) 콜드 캐시. joint6 캐시는 174, wrench 는 176 으로 호환되지 않아 joint6 학습은 빈
    #       캐시로 시작했다. RSI 가 "실제 방문한 물체 자세" 대신 레퍼런스 자세에서 출발하면
    #       물체가 정합하지 않은 상태로 시작한다. 검증 미실시.
    #   (2) 손가락 잔차 전환을 손목 변경과 **동시에** 했다. wrench+잔차 A/B 미실시.
    # 또한 term_obj_rot_err=0.75 는 무액션에서 25스텝에 47%(joint6)/37%(wrench)가 넘는다 —
    # 정책이 아무것도 안 해도 넘는 임계라 두 모드 공통의 학습 신호 문제다.
    wrist_mode: str = "joint6"

    # [delta-off] 2026-09-08 사용자 결정. 부모 기본값이 True 라 상속되면 델타 적분기가
    # 활성이 된다 (부팅 후 실측: _delta_cfg['hands'] = (True, 0.4, 1.0)). 명시적으로 끈다.
    # residual_action 도 False 이므로 손가락 경로는 **절대 액션 + EMA** 가 된다:
    #     smoothed = alpha*a + (1-alpha)*smoothed        (alpha = JOINT_GROUPS['hands'].ema_alpha = 0.5)
    #     target   = _scale(smoothed)                    (a=[-1,1] → 관절 범위로 선형 사상)
    # 이는 grasp env 의 절대 손가락 경로 및 sonic 경로의 기본(sonic_hand_delta=False)과 같은 방식이다.
    # 액션 의미가 델타(증분) → 절대(위치)로 바뀌므로 재사전학습이 필요하다.
    hand_delta_action: bool = False

    # [residual] 2026-09-08 롤백: residual_action override 를 제거해 부모 기본값으로 되돌렸다.
    # 잔차 구현 자체는 env 에 남아 있고 `residual_action=True` 로 되살릴 수 있다
    # (손가락 마진 스케일 = finger_residual_mode, 손목 고정 상한 = wrist6_res_*).
    # 액션 의미가 바뀌므로 되살릴 때는 재사전학습이 필요하다.
    # [ROLLBACK MARKER: fres] 2026-09-08 사용자 결정: 손가락도 잔차로 되살린다 (부모 기본 False 를
    # 덮어씀). 위 "절대 액션 + EMA" 설명은 이 줄이 False 일 때만 해당한다.
    residual_action: bool = True
    # 손가락 EMA(α = JOINT_GROUPS['hands'].ema_alpha = 0.5) 를 최종 목표가 아니라 **잔차에만** 건다
    # (손목 wrist6_ema_on_residual 과 같은 배치). 정규화 액션을 평활한 뒤 현재 프레임의 마진으로
    # 매핑하므로 레퍼런스 지연(최종 목표 EMA 면 1 스텝 = 20 ms)이 없다. False = 기존 최종 목표 EMA.
    finger_ema_on_residual: bool = True
    # [/ROLLBACK MARKER: fres]

    # [wrist6] 손목 관절 잔차 스케일. 정책 액션 [-1,1] 이 곱해지는 값이다 (DexMachina 와 동일).
    # 측정(knife, 물체 없음): 이 스케일 + EMA 0.2 로 손목 힘 최대 28 N / 9.1 N·m, effort 포화 0%.
    # 평활을 빼면(백색잡음) 178 N / 71.6 N·m 로 6배 뛰고 포화 11.2% 가 된다 — 힘은 kp·잔차가
    # 아니라 kp·추종오차이고, 잔차가 관절 응답(11 ms)보다 빠르면 오차가 잔차만큼 벌어진다.
    wrist6_res_pos: float = 0.04     # m
    wrist6_res_rot: float = 0.5      # rad

    # ── [residual] 액션 = 레퍼런스 기준 잔차 ─────────────────────────────────────────────────
    # 손가락을 델타 적분기(hand_delta_action) 에서 **잔차** 로 전환한다. DexMachina 본체가 쓰는
    # 방식이고(action_mode="residual"), 델타 적분기와 달리 목표가 매 프레임 레퍼런스에 재고정되어
    # 표류하지 않는다.
    #
    # finger_residual_mode:
    #   "margin" = DexMachina 본체. 관절별·프레임별 **남은 여유**로 스케일한다.
    #                m_up = upper - ref;  m_lo = ref - lower
    #                target = ref + where(a>=0, a*m_up, a*m_lo)
    #              a=+1 이면 그 관절의 상한에 정확히 도달하므로 한계를 넘을 수 없고 clamp 도
    #              windup 도 필요 없다. Shadow 손가락 한계는 **실제 물리 범위**라 이 스케일이
    #              의미를 가진다.
    #   "scale"  = 고정 배율 (기존 residual_scale_hands). 비교용으로 남긴다.
    #
    # 손목은 마진 스케일을 쓰지 않는다. 손목 관절 한계는 물리 범위가 아니라 **안전벽**이고
    # (병진 ±2.0 m, 회전 ±720°), 마진 스케일이면 a=±1 에서 목표가 2 m 밖이 되어 kp_t=1400 에
    # 곱하면 2800 N 을 지령하고 effort_limit 200 N 에 상시 포화한다 (측정: 그 구간에서 힘
    # 178 N / 토크 71.6 N·m, 포화 11.2%). DexMachina 에서 문제가 안 되는 건 kp_t=350 이라 힘
    # 상한이 실질 제한자로 작동하기 때문이다. 그래서 손목만 고정 상한(wrist6_res_*) 을 쓴다.
    finger_residual_mode: str = "margin"

    # EMA 를 **최종 목표**(레퍼런스 + 잔차)에 적용한다 (2026-09-08 사용자 결정). DexMachina 도
    # 켤 때는 최종 목표에 건다 (robot.py: new_targets = avg*target + (1-avg)*curr_targets).
    # 다만 그들의 기본값은 action_moving_avg=1.0 = 평활 없음이다. 우리는 kp_t 가 그들의 4배라
    # 같은 잔차가 더 큰 힘을 만들므로 평활을 켜 두고 학습에서 튜닝한다.
    # 손가락은 _group_alpha["hands"](=0.5), 손목은 아래 값을 쓴다 — 독립 튜닝을 위해 분리.
    # [w6gain] 0.2 → 0.5 → 1.0 → 다시 0.2 (2026-09-08 사용자 결정). 단 아래 wrist6_ema_on_residual=True
    # 라 이 EMA 는 **잔차에만** 걸린다 — wrench 모드의 힘 EMA(wrist_force_ema) 와 같은 배치로,
    # 레퍼런스에는 지연이 없다. 게인(shadow_float_cfg.py)은 이 조합 기준으로 맞췄다.
    wrist6_target_ema: float = 0.2
    # [ROLLBACK MARKER: w6gain] EMA 를 잔차에만 걸고 레퍼런스는 지연 없이 통과시키는 선택지.
    # 기본 False = 위의 결정(최종 목표에 EMA) 그대로. 1-DoF 모델 추정: 최종 목표 EMA 는 레퍼런스에
    # 4 스텝(80 ms) 지연을 넣어 결정론 추종 rms 17 mm / 3.9° (p95 40 mm / 8°) 가 되고, 빠른 구간
    # (p99 0.93 m/s → 74 mm 지연) 에서는 잔차 범위 ±40 mm 로 보상이 불가능하다. 잔차에만 걸면
    # 잡음 평활은 동일하게 유지하면서 rms 2.5~3.5 mm / 1.2° 로 내려간다. 기존 정책과 호환 안 됨.
    # True (2026-09-08 사용자 결정): EMA 0.2 + 잔차 전용. 게인 1400/33, 40/1.0 은 이 조합 기준.
    wrist6_ema_on_residual: bool = True
    # [/ROLLBACK MARKER: w6gain]

    # Action layout per hand: 3 position offset + 6 rotation (6D) + 18 finger joints = 27.
    # Bimanual → 54. Overrides the parent's 100 (z_res 64 + hand 36).
    # [wrist6] joint6 에서는 손별 6(손목 관절) + 18(손가락) = 24 → 48. __post_init__ 이 바꿉니다.
    action_space: int = 2 * (3 + 6 + 18)  # 54

    # 관측 조립부가 유일한 권위입니다(766->772 사건). 아래 값은 손 전용 조립부에서 실측했습니다
    # (2 env 부팅, s101_seg12_knife). 내역:
    #   A 자기수용감각 126 = 관절위치36 + 관절속도36 + palm자세6D12 + palm선속도6 + palm각속도6
    #                       + 손끝선속도30
    #   B 추종           252 = 손키포인트42x3 + 델타42x3
    #   C 물체/접촉/이력 180 = 물체pose+속도15 + 물체델타9 + 물체기준손끝오프셋30 + 관절DOF8
    #                       + 링크접촉마스크32 + 링크접촉력32 + 직전액션54
    #   방향 비트          1
    observation_space: int = 559

    # 몸통 키포인트 자체가 없으므로 리타게팅 FK 교체도 의미가 없습니다. 부모는 2026-09-06 부터
    # 기본 True 지만(학습에 반영됨) 여기서는 반드시 False 여야 합니다 — 켜면 존재하지 않는 몸통
    # 링크에 FK 를 걸다 형상 불일치로 터집니다.
    body_kpt_from_retarget_fk: bool = False

    # ---- body terms neutralised (weight 0 == term removed) ------------------------------------
    rew_body_kpts: float = 0.0
    rew_root_pos: float = 0.0
    rew_root_ori: float = 0.0
    rew_com_support: float = 0.0
    rew_feet_contact_match: float = 0.0

    # ---- RSI 상태 캐시: 손 전용 176 레이아웃으로 활성 --------------------------------------
    # 부모 캐시는 222 = 보상1 + 루트13 + 물체13 + jpos65 + jvel65 + smoothed65 이고, "루트 하나 +
    # 관절 65열" 이라는 전제가 슬라이스 오프셋에 박혀 있습니다. 손 전용은 루트가 둘(손목)이고
    # 구동 관절이 36열이라 176 으로 다시 잡았습니다 (레이아웃은 env 의 _STATE_DIM 주석 참조):
    #   보상1 + 손목(pose7+vel6)x2 + 물체(pose7+vel6) + jpos36 + jvel36 + J0pos8 + J0vel8
    #   + smoothed36 + 손목힘EMA6 + 손목토크EMA6 = 176
    # 부모에 없던 두 블록을 넣었습니다:
    #   J0(텐던 축) — 구동 36열에 J0 가 없어서, 캐시 히트에서도 J0 를 레퍼런스에서 재구성하면
    #     캐시된 J1 과 다른 프레임의 J0 가 섞여 텐던 제약(q_J0 <= q_J1)이 깨집니다.
    #   손목 힘/토크 EMA — 손목 임피던스 컨트롤러의 적분 상태입니다. 0 으로 리셋하면 손을 들고
    #     있던 힘이 사라져 복원 직후 손이 처집니다(무액션 침강 실측 0.038 m/s).
    use_state_cache: bool = True
    # ---- 캐시 쓰기 게이트의 몸통/루트 bar 비활성 ---------------------------------------------
    # _save_state_cache 의 품질 게이트는 부모에서 다섯 항입니다:
    #     good = (ft < enough_ft_threshold) AND (물체 3단계 조건)
    #            AND (body < cache_body_bar) AND (root_pos < ...) AND (root_rot < ...)
    # 뒤의 세 항은 떠 있는 손에서 의미가 없습니다.
    #   root_pos / root_rot — 골반이 없어 env 가 이 오차를 상수 0 으로 둡니다. bar 를 남겨두면
    #     "항상 통과"라 무해하지만, 왜 통과하는지가 코드에 드러나지 않습니다.
    #   body — env 가 e["body"] 를 palm 키포인트 2개의 오차로 재정의했습니다. 값은 살아 있지만
    #     term_wrist_pos_err=0.15 가 먼저 종료시키므로 0.30 bar 에는 도달할 수 없습니다. 즉
    #     설계로 무효가 아니라 임계값 순서 때문에 우연히 무효인 상태였고, 나중에
    #     term_wrist_pos_err 를 완화하면 이 항이 갑자기 살아나 캐시가 조용히 막힙니다.
    # inf 로 명시해 "손 env 는 이 게이트를 쓰지 않는다"를 cfg 에 남깁니다. 게이트는 grasp 의
    # 원래 구성인 손끝 + 물체 두 항으로 줄어듭니다 (부모 주석의 "grasp never gated on those").
    # Error / body_kpts 로그는 그대로 유지되므로 palm 추종 오차는 계속 볼 수 있습니다.
    cache_body_bar: float = float("inf")
    cache_root_pos_bar: float = float("inf")
    cache_root_rot_bar: float = float("inf")

    # ---- body terminations disabled -----------------------------------------------------------
    # Large sentinels rather than a flag: the gates are `err > threshold`, so an unreachable
    # threshold disables them without touching the env's termination assembly.
    term_root_pos_err: float = 1.0e9
    term_root_rot_err: float = 1.0e9

    # ---- SONIC prior off (no body DOF to drive) -----------------------------------------------
    # The 29 body joints do not exist here, so the frozen SONIC decoder has nothing to act on.
    # The env reads this as `getattr(cfg, "use_sonic", True)` at two sites (env.py:338 loads the
    # sonic_smpl npz, env.py:1425 builds the decoder), so defining it False disables both.
    use_sonic: bool = False

    def __post_init__(self):
        # 부모(G1ShadowSonicResidualEnvCfg)에는 __post_init__ 이 없으므로 super() 호출은
        # configclass 가 만든 기본 구현으로 갑니다 — 있으면 부르고 없으면 통과하도록 getattr.
        _sup = getattr(super(), "__post_init__", None)
        if _sup is not None:
            _sup()
        # action_dt = 제어 주기. sim.dt x decimation 에서 유도하므로 제어율을 바꿔도 따라옵니다.
        self.wrist_action_dt = float(self.sim.dt) * int(self.decimation)

        # ── [wrist6] 모드별 차원 ──────────────────────────────────────────────────────────
        # 관측에서 액션에 의존하는 항은 `_prev_policy_action` 하나뿐이므로 (조립부 실측)
        #     observation_space = _OBS_BASE + action_space
        # 가 두 모드 모두에 성립한다. _OBS_BASE = 126(A) + 252(B) + 126(C에서 직전액션 제외)
        #                                        + 1(방향 비트) = 505.
        # 이렇게 쓰면 액션 레이아웃을 바꿀 때 관측 상수를 따로 고칠 필요가 없다 — 766→772 사건이
        # 그 불일치에서 나왔다.
        # 멱등하게 만든다 — train.py 가 --wrist_mode 를 적용한 뒤 다시 호출하므로, 이전 값을
        # assert 하면 두 번째 호출에서 깨진다. 관측 조립부와의 정합은 env 의 런타임 assert
        # (`obs.shape[-1] == c.observation_space`) 가 실제로 보장한다.
        if self.wrist_mode not in ("wrench", "joint6"):
            raise ValueError(f"wrist_mode must be 'wrench' or 'joint6', got {self.wrist_mode!r}")
        self.action_space = _ACT_JOINT6 if self.wrist_mode == "joint6" else _ACT_WRENCH
        self.observation_space = _OBS_BASE + int(self.action_space)
        if self.wrist_mode == "joint6":
            # 에셋도 교체한다. 클래스 기본값은 외력 방식이므로 여기서만 바꾼다.
            self.hand_l_cfg = SHADOW_FLOAT6_L_CFG
            self.hand_r_cfg = SHADOW_FLOAT6_R_CFG
        print(f"[wrist6] mode={self.wrist_mode}  action={self.action_space}  "
              f"obs={self.observation_space}  "
              f"asset={'shadow_float6' if self.wrist_mode == 'joint6' else 'shadow_float'}")

    # NOTE: the object physics path stays ON. The existing kinematic pretrain env forces
    # `_object_cfg=None`; stage 1 needs the opposite — manipulation requires a real object.
