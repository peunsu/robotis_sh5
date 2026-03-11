# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# 종료 조건 (Terminations) 구현
##
def root_pos_distance_from_origin(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """원점으로부터의 거리가 threshold를 넘으면 종료."""
    # 로봇의 위치 (x, y, z)
    pos = env.scene["robot"].data.root_pos_w
    # 평면 거리 계산 (x, y 만 사용)
    distance = torch.norm(pos[:, :2], dim=-1)
    return distance > threshold

def is_near_goal(env: ManagerBasedRLEnv, threshold: float):
    """로봇이 목표 지점에 충분히 가까워졌는지 확인 (종료 조건 용)."""
    # 로봇 위치 (Articulation이라서 .data 사용 가능)
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, :2]
    
    # --- 수정 부분 ---
    # goal_marker가 XformPrimView이므로 .get_world_poses() 사용
    goal_pos_w, _ = env.scene["goal_marker"].get_world_poses()
    goal_pos_w = goal_pos_w[:, :2]
    
    # 거리 계산 후 threshold 이내인지 확인
    distance = torch.norm(goal_pos_w - robot_pos_w, dim=-1)
    return distance < threshold

def bad_orientation(env: ManagerBasedRLEnv, threshold: float = 0.5) -> torch.Tensor:
    """로봇이 일정 각도 이상 기울어졌는지 판단 (넘어짐 종료)."""
    up_v = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    robot_up_v = quat_apply_inverse(env.scene["robot"].data.root_quat_w, up_v)
    
    # z성분이 threshold(cos(angle))보다 낮으면 너무 많이 기울어진 것
    return robot_up_v[:, 2] < threshold