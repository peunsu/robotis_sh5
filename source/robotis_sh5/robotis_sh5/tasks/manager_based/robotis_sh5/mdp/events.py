# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from .waypoint_manager import get_or_create_waypoint_manager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_random_waypoints(
    env, 
    env_ids: torch.Tensor, # Isaac Lab이 자동으로 넣어주는 인자
    num_waypoints: int, 
    distance_range: tuple[float, float]
):
    # 1. 매니저 가져오기
    wm = get_or_create_waypoint_manager(env, num_waypoints)
    
    # 2. 인자로 받은 env_ids 사용 (더 이상 asset_cfg.env_ids라고 쓰지 않아!)
    num_resets = len(env_ids)
    
    # 해당 환경들의 타겟 인덱스 초기화
    wm.target_indices[env_ids] = 0
    
    # 웨이포인트 생성 로직
    fps = torch.zeros((num_resets, num_waypoints, 3), device=env.device)
    for i in range(num_waypoints):
        # x축 방향 전진 (랜덤 범위 적용)
        low, high = distance_range
        fps[:, i, 0] = (i + 1) * torch.empty(num_resets, device=env.device).uniform_(low, high)
        # y축 랜덤 (좌우)
        fps[:, i, 1] = torch.randn(num_resets, device=env.device) * 1.5
        fps[:, i, 2] = 0.2
        
    # 절대 좌표로 변환하여 저장
    wm.waypoints[env_ids] = fps + env.scene.env_origins[env_ids].unsqueeze(1)
    
    # 초기 거리 업데이트 (Progress 보상용)
    root_pos = env.scene["robot"].data.root_pos_w[env_ids]
    current_target = wm.waypoints[env_ids, 0]
    wm.prev_dist[env_ids] = torch.norm(current_target - root_pos, dim=-1)
    
    # 시각화 갱신
    wm.update_visuals()
    
def reset_root_at_origin(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """로봇을 각 환경의 원점(Origin)으로 리셋하고 속도를 0으로 초기화합니다."""
    
    # 1. 대상 에셋(로봇) 가져오기
    asset = env.scene[asset_cfg.name]
    
    # 2. 로봇의 기본 상태(default state) 가져오기
    # 보통 ArticulationCfg에서 설정한 init_state 값을 바탕으로 함
    default_root_state = asset.data.default_root_state[env_ids].clone()
    
    # 3. 위치를 환경의 원점으로 설정
    # default_root_state는 로봇 좌표계 기준이므로, 실제 세계 좌표(World)로 변환하기 위해 
    # 해당 환경의 원점(env_origins)을 더해줌
    default_root_state[:, :3] += env.scene.env_origins[env_ids]
    
    # [선택] 필요하다면 높이(z)나 방향(quat)을 여기서 살짝 랜덤하게 흔들 수도 있어
    # 예: default_root_state[:, 0:2] += torch.empty(len(env_ids), 2, device=env.device).uniform_(-0.5, 0.5)

    # 4. 시뮬레이션에 상태 쓰기
    # 위치/방향(pose)과 선속도/각속도(velocity)를 모두 초기화
    asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
    
    # 5. 조인트 상태(Joint positions/velocities)도 기본값으로 리셋
    default_joint_pos = asset.data.default_joint_pos[env_ids]
    default_joint_vel = asset.data.default_joint_vel[env_ids]
    asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)
    
def update_waypoint_status(env, env_ids, threshold: float):
    """
    Isaac Lab EventManager는 env와 env_ids를 기본으로 넘겨줘.
    그 뒤에 params에 적은 threshold가 키워드 인자로 들어와.
    """
    wm = getattr(env, "waypoint_manager", None)
    if wm is not None:
        # wm.update가 내부적으로 모든 환경을 체크한다면 env_ids는 무시해도 되지만, 
        # 성능을 위해선 해당 env_ids만 업데이트하는 게 좋아.
        wm.update(threshold=threshold)