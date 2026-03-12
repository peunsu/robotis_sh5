# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def is_near_goal(env: ManagerBasedRLEnv, threshold: float = 0.2) -> torch.Tensor:
    """로봇이 목표 지점의 임계치 안에 들어왔는지 확인 (성공 종료)"""
    # 로봇 현재 위치 (XY)
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, :2]
    
    # 목표 지점 위치 (XY)
    goal_pos_w, _ = env.scene["goal_marker"].get_world_poses()
    goal_pos_w = goal_pos_w[:, :2]
    
    # 거리 계산
    distance = torch.norm(goal_pos_w - robot_pos_w, dim=-1)
    
    # 임계치 이내이면 True 반환
    return distance < threshold

def bad_orientation(env: ManagerBasedRLEnv, threshold: float = 0.5) -> torch.Tensor:
    """
    로봇이 일정 각도 이상 기울어지면 True 반환.
    - 서 있을 때 projected_gravity_b[:, 2]는 -1.0에 가까움.
    - threshold가 0.5라면, -0.5보다 커지는 순간(0.0에 가까워지는 순간) 리셋.
    """
    # 중력의 Z성분이 -0.5보다 크다 = 약 60도 이상 기울어졌다
    return env.scene["robot"].data.projected_gravity_b[:, 2] > -threshold

def root_pos_distance_from_env_origin(env: ManagerBasedRLEnv, threshold: float = 4.5) -> torch.Tensor:
    """로봇이 환경 원점으로부터 너무 멀리 벗어났는지 확인"""
    # 환경 원점 대비 상대 위치 (Local Pos)
    relative_pos = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    distance = torch.norm(relative_pos[:, :2], dim=-1)
    
    return distance > threshold