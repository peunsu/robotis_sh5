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

def fix_hand_command_curriculum(env: ManagerBasedRLEnv, env_ids, old_value, fix_hand_command, num_step):
    """
    커리큘럼 학습을 위해, fix_hand_command가 True인 경우 일정 단계까지 손가락 관절값을 고정된 값으로 설정하는 커리큘럼 함수.
    """
    
    if env.common_step_counter > num_step:
        return fix_hand_command
    
    return mdp.modify_env_param.NO_CHANGE

def fade_in_reward_weight(env, env_ids, old_value, initial_weight, target_weight, grace_period, fade_in_steps):
    """
    grace_period 동안은 NO_CHANGE, 그 후 fade_in_steps 동안 target_weight까지 선형 증가.
    """
    current_step = env.common_step_counter
    
    # 1. Grace Period 단계
    if current_step <= grace_period:
        # 초기값이 0이 아닐 수도 있으니, 0으로 강제 세팅하고 싶다면 0.0 반환
        return initial_weight
    
    # 2. Fade-in 단계
    elif current_step <= grace_period + fade_in_steps:
        progress = (current_step - grace_period) / fade_in_steps
        return progress * target_weight + (1 - progress) * initial_weight
    
    # 3. Target 도달 이후 (변화가 없으면 NO_CHANGE를 보내서 연산 절약)
    if old_value == target_weight:
        return mdp.modify_env_param.NO_CHANGE
        
    return target_weight