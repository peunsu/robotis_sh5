# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

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

    # 랜덤 위치 생성
    random_pos = torch.zeros((num_resets, 3), device=env.device)
    random_pos[:, 0] = torch.rand(num_resets, device=env.device) * (pos_range["x"][1] - pos_range["x"][0]) + pos_range["x"][0]
    random_pos[:, 1] = torch.rand(num_resets, device=env.device) * (pos_range["y"][1] - pos_range["y"][0]) + pos_range["y"][0]
    random_pos[:, 2] = 0.1 

    # --- 수정 부분 ---
    # env_ids 대신 indices를 사용해야 해!
    env.scene[asset_name].set_world_poses(positions=random_pos, indices=env_ids)

def reset_root_at_random_pos_2d(env: ManagerBasedRLEnv, env_ids: torch.Tensor, pos_range: dict, yaw_range: tuple):
    """에피소드 리셋 시 로봇을 2D 평면 상의 랜덤한 위치와 방향으로 배치."""
    num_resets = len(env_ids)
    if num_resets == 0: return

    # 위치 랜덤 샘플링 (x, y)
    ranges = [
        (pos_range["x"][0], pos_range["x"][1]),
        (pos_range["y"][0], pos_range["y"][1]),
    ]
    random_pos = torch.zeros((num_resets, 3), device=env.device)
    for i in range(2):
        random_pos[:, i] = torch.rand(num_resets, device=env.device) * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
    random_pos[:, 2] = 0.1 # 지면 높이

    # 방향(Yaw) 랜덤 샘플링
    random_yaw = torch.rand(num_resets, device=env.device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    
    # 쿼터니언으로 변환 (간단한 yaw-to-quat 수식 적용)
    random_quat = torch.zeros((num_resets, 4), device=env.device)
    random_quat[:, 0] = torch.cos(random_yaw / 2)
    random_quat[:, 3] = torch.sin(random_yaw / 2)

    # 로봇의 물리 상태 업데이트
    env.scene["robot"].write_root_pose_to_sim(torch.cat([random_pos, random_quat], dim=-1), env_ids)