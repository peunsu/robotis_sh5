# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
def get_rel_pos_to_current_waypoint(env) -> torch.Tensor:
    """로봇 베이스를 기준으로 현재 타겟 웨이포인트까지의 상대적 위치 (x, y, z)"""
    # env에 할당된 waypoint_manager 가져오기
    wm = getattr(env, "waypoint_manager", None)
    if wm is None:
        # 매니저가 없을 경우를 대비한 예외 처리 (0 벡터 반환)
        return torch.zeros((env.num_envs, 3), device=env.device)

    # 로봇의 현재 세계 좌표 위치
    root_pos_w = env.scene["robot"].data.root_pos_w
    
    # 각 환경별 현재 타겟의 세계 좌표 추출 (N_envs, 3)
    # wm.target_indices는 각 환경이 몇 번째 목표를 보고 있는지 저장함
    current_target_w = wm.waypoints[torch.arange(env.num_envs), wm.target_indices]
    
    # 상대 위치 계산 (Target - Robot)
    rel_pos_w = current_target_w - root_pos_w
    
    # [선택 사항] 만약 로봇의 로컬 좌표계(Body Frame) 기준의 상대 위치가 필요하다면
    # 아래 주석을 해제해서 회전 변환을 적용해 (보통 관측값으로는 이게 더 좋아)
    root_quat_w = env.scene["robot"].data.root_quat_w
    rel_pos_b = math_utils.quat_apply_inverse(root_quat_w, rel_pos_w)
    return rel_pos_b

    # return rel_pos_w

def get_target_waypoint_index(env) -> torch.Tensor:
    """전체 웨이포인트 중 현재 몇 번째 타겟인지 정규화된 인덱스 반환 [0, 1]"""
    wm = getattr(env, "waypoint_manager", None)
    if wm is None:
        return torch.zeros((env.num_envs, 1), device=env.device)

    # 현재 인덱스 (0 ~ num_waypoints-1)를 실수형으로 변환
    # (N_envs, 1) 형태로 반환하여 다른 관측값들과 결합하기 쉽게 만듦
    normalized_idx = wm.target_indices.unsqueeze(1).float() / (wm.num_waypoints - 1)
    
    return normalized_idx

def get_waypoint_heading_error_sin_cos(env):
    """타겟과의 헤딩 오차를 sin, cos 값으로 반환 (2-dim)"""
    wm = getattr(env, "waypoint_manager", None)
    if wm is None: return torch.zeros((env.num_envs, 2), device=env.device)

    root_pos_w = env.scene["robot"].data.root_pos_w[:, :2]
    current_target_w = wm.waypoints[torch.arange(env.num_envs), wm.target_indices, :2]
    
    # 로봇의 현재 Yaw 추출
    _, _, robot_yaw = math_utils.euler_xyz_from_quat(env.scene["robot"].data.root_quat_w)
    
    # 타겟 방향 각도
    target_heading_w = torch.atan2(
        current_target_w[:, 1] - root_pos_w[:, 1],
        current_target_w[:, 0] - root_pos_w[:, 0]
    )
    
    # 오차 정규화 및 sin/cos 계산
    error = torch.atan2(torch.sin(target_heading_w - robot_yaw), torch.cos(target_heading_w - robot_yaw))
    return torch.stack([torch.cos(error), torch.sin(error)], dim=-1)