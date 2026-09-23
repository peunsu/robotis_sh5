"""Stage 1: 떠 있는 양손 Shadow 로 손 동작을 물리 시뮬레이션에서 다듬는 hand_pretrain 설정.

부모(G1ShadowSonicResidualEnvCfg)를 상속하고, 몸이 없으므로 몸 관련 보상·종료와 SONIC 을 끈다.
손목은 palm 앞 6-DoF 관절(joint6)로, 손가락은 레퍼런스 기준 margin 잔차로 구동한다.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..g1_shadow_sonic_residual.g1_shadow_sonic_residual_env_cfg import (
    G1ShadowSonicResidualEnvCfg,
)
from .shadow_float_cfg import SHADOW_FLOAT6_L_CFG, SHADOW_FLOAT6_R_CFG

_ACT = 2 * (6 + 18)  # 손별 [손목 관절 6 | 손가락 18]
_OBS_BASE = 505  # 관측 중 직전 액션을 뺀 부분. observation_space = _OBS_BASE + action_space


@configclass
class G1ShadowHandPretrainEnvCfg(G1ShadowSonicResidualEnvCfg):
    # ---- 떠 있는 손 에셋 ----
    hand_l_cfg = SHADOW_FLOAT6_L_CFG  # palm 앞에 손목 관절 6개가 달린 떠 있는 손 (anchor 고정)
    hand_r_cfg = SHADOW_FLOAT6_R_CFG

    # ---- 손가락 액션 ----
    residual_action: bool = True  # True = 레퍼런스 기준 margin 잔차, False = 절대 액션 + EMA (바꾸면 재학습)
    finger_ema_on_residual: bool = True  # EMA 를 최종 목표가 아니라 잔차에만 건다

    # ---- joint6 손목 잔차 ----
    wrist6_res_pos: float = 0.04  # m, 손목 병진 잔차 상한
    wrist6_res_rot: float = 0.5  # rad, 손목 회전 잔차 상한
    wrist6_target_ema: float = 0.2  # 손목 EMA α
    wrist6_ema_on_residual: bool = True  # 손목 EMA 를 잔차에만 건다

    # ---- 차원 ----
    action_space: int = _ACT  # 48
    observation_space: int = _OBS_BASE + _ACT  # 553

    # ---- 몸이 없어 끄는 항목 ----
    body_kpt_from_retarget_fk: bool = False
    rew_body_kpts: float = 0.0
    rew_root_pos: float = 0.0
    rew_root_ori: float = 0.0
    rew_com_support: float = 0.0
    rew_feet_contact_match: float = 0.0

    # ---- RSI 상태 캐시 ----
    use_state_cache: bool = True  # 손 전용 캐시 레이아웃 (174)
    enough_body_threshold: float = float("inf")  # 몸·루트 기준은 끈다
    enough_root_pos_threshold: float = float("inf")
    enough_root_rot_threshold: float = float("inf")
    term_root_pos_err: float = 1.0e9  # 몸 종료 조건은 끈다
    term_root_rot_err: float = 1.0e9

    # ---- SONIC ----
    use_sonic: bool = False  # 몸이 없어 SONIC 을 쓰지 않는다

