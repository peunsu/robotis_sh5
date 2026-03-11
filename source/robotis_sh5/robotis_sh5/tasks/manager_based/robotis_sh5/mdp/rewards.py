# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.utils.math import quat_error_magnitude, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def goal_distance_reward(env: ManagerBasedRLEnv, std: float = 1.0) -> torch.Tensor:
    """가까워질수록 보상이 확실히 증가하는 거리 보상"""
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, :2]
    goal_pos_w, _ = env.scene["goal_marker"].get_world_poses()
    goal_pos_w = goal_pos_w[:, :2]
    
    distance = torch.norm(goal_pos_w - robot_pos_w, dim=-1)
    
    # Gaussian Kernel: 멀리서는 완만하고 가까울수록 가파르게 보상 (정규화된 형태)
    return torch.exp(-torch.square(distance) / (2.0 * std**2))

def heading_alignment_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """로봇이 목표를 향하는 정도 (Swerve의 유연성을 위해 Cosine 값 그대로 사용)"""
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, :2]
    goal_pos_w, _ = env.scene["goal_marker"].get_world_poses()
    goal_pos_w = goal_pos_w[:, :2]

    to_goal_vec = goal_pos_w - robot_pos_w
    # 방향 벡터 정규화 (Zero division 방지)
    to_goal_unit = to_goal_vec / (torch.norm(to_goal_vec, dim=-1, keepdim=True) + 1e-6)
    
    # 로봇의 전방 벡터 (World 기준)
    _, _, yaw = euler_xyz_from_quat(env.scene["robot"].data.root_quat_w)
    forward_vec = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    
    # 내적을 통한 유사도 측정 (-1.0 ~ 1.0)
    return torch.sum(forward_vec * to_goal_unit, dim=-1)

def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

def base_orientation_l2(env: ManagerBasedRLEnv, target_quat: tuple = (1.0, 0.0, 0.0, 0.0)):
    """기울어짐 페널티 (안정적인 L2 오차)"""
    current_quat = env.scene["robot"].data.root_quat_w
    target_quat_tensor = torch.tensor(target_quat, device=env.device).expand(env.num_envs, -1)
    error = quat_error_magnitude(current_quat, target_quat_tensor)
    return torch.square(error)

def joint_limits_penalty_l2(env: ManagerBasedRLEnv):
    """
    조인트 위치가 물리적 한계(Soft Limits)를 벗어날 때 부여하는 페널티.
    """
    asset = env.scene["robot"]
    # 하드웨어 보호를 위한 Soft limits 기준
    low_limit = asset.data.soft_joint_pos_limits[..., 0]
    high_limit = asset.data.soft_joint_pos_limits[..., 1]
    current_pos = asset.data.joint_pos
    
    # 한계의 98% 지점을 넘어서면 페널티 발생 시작
    too_low = torch.clamp(low_limit * 0.98 - current_pos, min=0.0)
    too_high = torch.clamp(current_pos - high_limit * 0.98, min=0.0)
    
    # L2 Norm을 사용하여 한계를 많이 벗어날수록 페널티가 기하급수적으로 증가
    return torch.sum(torch.square(too_low + too_high), dim=-1)