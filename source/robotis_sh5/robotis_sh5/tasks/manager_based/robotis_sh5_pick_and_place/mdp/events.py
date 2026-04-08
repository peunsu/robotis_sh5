# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, combine_frame_transforms
from .utils import get_trajectory_data # 아까 만든 함수 임포트

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_object_to_tray_pose(
    env, 
    env_ids, 
    asset_cfg: SceneEntityCfg, 
    file_path: str, 
    frame_idx: int, 
    table_height: float = 1.0,
    pos_range_xy: tuple[float, float] = (-0.1, 0.1), # 랜덤 범위 추가
    rot_range_z: tuple[float, float] = (-0.2, 0.2)   # 랜덤 범위 추가
):
    # 1. npy 데이터 가져오기
    traj_data = get_trajectory_data(env, file_path, frame_idx=frame_idx)
    num_resets = len(env_ids)
    
    # 2. 랜덤 오프셋 생성 (XY 위치 & Z축 회전)
    p_min, p_max = pos_range_xy
    r_min, r_max = rot_range_z
    
    # 위치 오프셋 (N, 3) -> Z는 0 고정
    pos_offset = torch.zeros(num_resets, 3, device=env.device)
    pos_offset[:, :2] = p_min + torch.rand(num_resets, 2, device=env.device) * (p_max - p_min)
    
    # 회전 오프셋 (Yaw -> Quat)
    yaw_offset = r_min + torch.rand(num_resets, device=env.device) * (r_max - r_min)
    quat_offset = quat_from_euler_xyz(
        torch.zeros_like(yaw_offset), 
        torch.zeros_like(yaw_offset), 
        yaw_offset
    )

    # 3. 좌표 변환 및 배치 (World Pos + Env Origin + Table + Offset)
    env_origins = env.scene.env_origins[env_ids]
    obj_base_pos = traj_data["obj_pos"].view(1, 3).expand(num_resets, 3)
    obj_base_quat = traj_data["obj_quat"].view(1, 4).expand(num_resets, 4)

    # 최종 위치 계산
    target_pos_w = obj_base_pos + env_origins + torch.tensor([0.0, 0.0, table_height], device=env.device) + pos_offset
    
    # 최종 회전 계산 (원본 회전 * 랜덤 회전)
    _, target_quat_w = combine_frame_transforms(
        torch.zeros_like(target_pos_w), obj_base_quat,
        torch.zeros_like(target_pos_w), quat_offset
    )
    
    # 4. Root State 구성 (정지 상태 vel=0)
    vel = torch.zeros((num_resets, 6), device=env.device)
    root_state = torch.cat([target_pos_w, target_quat_w, vel], dim=-1)

    # 5. 시뮬레이션에 적용
    asset = env.scene[asset_cfg.name]
    asset.write_root_state_to_sim(root_state, env_ids)