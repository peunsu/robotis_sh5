# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# 보상 (Rewards) 구현
##

def goal_distance_reward(env: ManagerBasedRLEnv, std: float = 0.5) -> torch.Tensor:
    """목표 지점까지의 거리에 따른 보상 (Exponential)."""
    # 1. 로봇 위치 (Articulation이라서 .data 사용 가능)
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, :2]
    
    # 2. 목표 지점 위치 (XformPrimView라서 .get_world_poses() 사용)
    # --- 수정된 부분 ---
    goal_pos_w, _ = env.scene["goal_marker"].get_world_poses()
    goal_pos_w = goal_pos_w[:, :2]
    
    # 3. 거리 계산 및 보상 반환
    distance = torch.norm(goal_pos_w - robot_pos_w, dim=-1)
    return torch.exp(-distance / (2.0 * std**2))

def base_orientation_l2(env: ManagerBasedRLEnv, target_quat: tuple = (1.0, 0.0, 0.0, 0.0)):
    """로봇 몸체의 방향 오차를 계산하여 페널티 부여."""
    # 현재 로봇의 쿼터니언 (w, x, y, z)
    current_quat = env.scene["robot"].data.root_quat_w
    
    # 목표 쿼터니언을 텐서로 변환
    target_quat_tensor = torch.tensor(target_quat, device=env.device).repeat(env.num_envs, 1)
    
    # 두 쿼터니언 사이의 오차(각도 차이) 계산
    # quat_error_magnitude 대신 아래 방식을 주로 사용해
    error = quat_error_magnitude(current_quat, target_quat_tensor)
    
    return torch.square(error)