"""Stage 1: 떠 있는 양손 Shadow 로 손 동작을 물리 시뮬레이션에서 다듬는 hand_pretrain 설정.

부모(G1ShadowSonicResidualEnvCfg)를 상속한다. env 에 몸과 SONIC 이 없어 몸 관련 cfg 항목은 읽히지 않는다.
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

    # ---- RSI 상태 캐시 ----
    use_state_cache: bool = True  # 손 전용 캐시 레이아웃 (174)

