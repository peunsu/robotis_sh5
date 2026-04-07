# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from .utils import get_trajectory_data # 아까 만든 함수 임포트

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_object_to_tray_pose(env, env_ids, asset_cfg: SceneEntityCfg, file_path: str, frame_idx: int, table_height: float = 1.0):
    # 1. npy 데이터 가져오기 (이 데이터는 0,0,0 기준의 World 좌표임)
    traj_data = get_trajectory_data(env, file_path, frame_idx=frame_idx)
    
    # 2. 환경별 원점(Origins) 가져오기
    # env_origins shape: (num_resets, 3)
    env_origins = env.scene.env_origins[env_ids]
    num_resets = len(env_ids)
    
    table_offset = torch.tensor([0.0, 0.0, table_height], device=env.device)
    
    # 3. 좌표 변환 (NPY World Pos + Env Origin)
    # traj_data["obj_pos"]는 (3,) 이므로 (num_resets, 3)으로 확장 후 더함
    target_pos_w = traj_data["obj_pos"].view(1, 3).expand(num_resets, 3) + env_origins + table_offset
    target_quat_w = traj_data["obj_quat"].view(1, 4).expand(num_resets, 4) # 회전은 원점 영향 안 받음
    
    # 4. Root State 구성 [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    # 속도는 정지 상태(0)로 초기화
    vel = torch.zeros((num_resets, 6), device=env.device)
    
    root_state = torch.cat([
        target_pos_w, 
        target_quat_w, 
        vel
    ], dim=-1)

    # 5. 시뮬레이션에 쓰기
    asset = env.scene[asset_cfg.name]
    asset.write_root_state_to_sim(root_state, env_ids)