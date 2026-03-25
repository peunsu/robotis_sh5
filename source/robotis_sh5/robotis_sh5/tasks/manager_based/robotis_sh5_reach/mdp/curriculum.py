# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def fade_in_reward_weight(env, env_ids, old_value, target_weight, grace_period, fade_in_steps):
    """
    grace_period 동안은 NO_CHANGE, 그 후 fade_in_steps 동안 target_weight까지 선형 증가.
    """
    current_step = env.common_step_counter
    
    # 1. Grace Period 단계
    if current_step <= grace_period:
        # 초기값이 0이 아닐 수도 있으니, 0으로 강제 세팅하고 싶다면 0.0 반환
        return 0.0
    
    # 2. Fade-in 단계
    elif current_step <= grace_period + fade_in_steps:
        progress = (current_step - grace_period) / fade_in_steps
        return progress * target_weight
    
    # 3. Target 도달 이후 (변화가 없으면 NO_CHANGE를 보내서 연산 절약)
    if old_value == target_weight:
        return mdp.modify_env_param.NO_CHANGE
        
    return target_weight