# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# 이벤트 (Events/Resets) 구현
##

def reset_goal_position(env: ManagerBasedRLEnv, env_ids: torch.Tensor, pos_range: dict, asset_name: str = "goal_marker"):
    """목표 지점 마커의 위치를 랜덤하게 리셋."""
    
    num_resets = len(env_ids)
    if num_resets == 0: return
    
    view = env.scene[asset_name]
    
    if view.count < env.num_envs:
        # 마커가 하나뿐인 경우: 0번 인덱스만 사용
        indices = torch.tensor([0], device=env.device, dtype=torch.long)
        num_to_calc = 1
    else:
        # 마커가 환경마다 있는 경우: 들어온 env_ids 그대로 사용
        indices = env_ids
        num_to_calc = num_resets

    # --- 수정된 안전 로직 ---
    # 실제 존재하는 마커 개수만큼만 인덱스를 준비함
    if view.count < env.num_envs:
        # 마커가 하나뿐인 경우: 0번 인덱스만 사용
        indices = torch.tensor([0], device=env.device, dtype=torch.long)
        num_to_calc = 1
    else:
        # 마커가 환경마다 있는 경우: 들어온 env_ids 그대로 사용
        indices = env_ids
        num_to_calc = num_resets

    # 랜덤 위치 생성 (계산해야 할 개수만큼만)
    random_pos = torch.zeros((num_to_calc, 3), device=env.device)
    random_pos[:, 0] = torch.rand(num_to_calc, device=env.device) * (pos_range["x"][1] - pos_range["x"][0]) + pos_range["x"][0]
    random_pos[:, 1] = torch.rand(num_to_calc, device=env.device) * (pos_range["y"][1] - pos_range["y"][0]) + pos_range["y"][0]
    random_pos[:, 2] = 0.1 

    # 위치 적용 (indices가 위치 개수와 맞아야 함)
    view.set_world_poses(positions=random_pos, indices=indices)

def reset_root_around_goal_2d(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor, 
    min_dist: float = 1.0, 
    max_dist: float = 4.0, 
    yaw_range: tuple = (-math.pi, math.pi)
):
    num_resets = len(env_ids)
    if num_resets == 0: return

    # 1. 로컬 좌표계에서의 랜덤 위치 계산 (원점 기준)
    r = torch.sqrt(torch.rand(num_resets, device=env.device) * (max_dist**2 - min_dist**2) + min_dist**2)
    theta = torch.rand(num_resets, device=env.device) * 2 * math.pi
    
    local_pos = torch.zeros((num_resets, 3), device=env.device)
    local_pos[:, 0] = r * torch.cos(theta)
    local_pos[:, 1] = r * torch.sin(theta)
    local_pos[:, 2] = 0.1 

    # --- 핵심 수정 사항: 각 환경의 월드 원점 더하기 ---
    # env.scene.env_origins는 각 환경 인덱스에 해당하는 월드 좌표상의 원점을 가지고 있어.
    env_origins = env.scene.env_origins[env_ids]
    world_pos = local_pos + env_origins # 로컬 랜덤 위치를 월드 위치로 변환

    # 2. 방향(Yaw) 계산
    random_yaw = torch.rand(num_resets, device=env.device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    random_quat = torch.zeros((num_resets, 4), device=env.device)
    random_quat[:, 0] = torch.cos(random_yaw / 2)
    random_quat[:, 3] = torch.sin(random_yaw / 2)

    # 3. 월드 좌표계 기준으로 포즈 업데이트
    env.scene["robot"].write_root_pose_to_sim(torch.cat([world_pos, random_quat], dim=-1), env_ids)