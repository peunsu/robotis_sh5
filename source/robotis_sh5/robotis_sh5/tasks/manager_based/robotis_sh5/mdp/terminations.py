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

def bad_orientation(env: "ManagerBasedRLEnv", threshold: float) -> torch.Tensor:
    """로봇이 일정 각도 이상 기울어지면 종료 (넘어짐 감지)."""
    # 로봇의 위쪽 방향 벡터 (Local Z-axis) 추출
    # Isaac Lab의 project_gravity 함수를 쓰면 세계 좌표계 기준의 중력 방향(내림) 벡터를 얻을 수 있어.
    root_quat_w = env.scene["robot"].data.root_quat_w
    
    # 로봇의 로컬 Z축(0,0,1)이 세계 좌표계에서 어디를 향하는지 계산
    up_vec_w = math_utils.quat_apply(
        root_quat_w, 
        torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    )
    
    # 세계 좌표계의 Z축(0,0,1)과의 내적을 통해 기울기 확인
    # 1.0이면 똑바로 선 상태, 0.0이면 옆으로 누운 상태, -1.0이면 뒤집힌 상태
    z_dot = up_vec_w[:, 2]
    
    # 설정한 문턱값(threshold)보다 낮아지면(많이 기울면) True 반환
    return z_dot < threshold

def all_waypoints_reached(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """모든 웨이포인트를 순차적으로 통과했는지 확인."""
    wm = getattr(env, "waypoint_manager", None)
    if wm is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 1. 현재 인덱스가 마지막 인덱스인지 확인 (num_waypoints - 1)
    is_last_waypoint = (wm.target_indices == wm.num_waypoints - 1)
    
    # 2. 마지막 타겟과의 거리 측정
    root_pos_w = env.scene["robot"].data.root_pos_w[:, :2]
    final_target_w = wm.waypoints[torch.arange(env.num_envs), -1, :2]
    dist_to_final = torch.norm(final_target_w - root_pos_w, dim=-1)
    
    # 마지막 인덱스이면서 거리도 충분히 가깝다면 '성공' 종료
    # threshold는 보통 0.2 ~ 0.3 정도로 잡아주면 적당해
    success = is_last_waypoint & (dist_to_final < 0.2)
    
    return success