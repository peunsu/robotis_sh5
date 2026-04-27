# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .utils import get_virtual_link_poses

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

markers = VisualizationMarkers(VisualizationMarkersCfg(
    prim_path="/Visual/Markers",
    markers={
        "fingertips": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "palm": sim_utils.SphereCfg(
            radius=0.015,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
))

def object_distance_reward(env: ManagerBasedRLEnv, fingertip_names: list, palm_name: str) -> torch.Tensor:
    # 1. 물체 중심 위치
    obj_pos = env.scene["object"].data.root_pos_w
    
    
    # 2. 가상 위치 계산
    v_fingertip_pos, v_palm_pos = get_virtual_link_poses(env, fingertip_names, palm_name)
    
    # 3. 거리 계산
    dist_fingertips = torch.stack([torch.norm(pos - obj_pos, dim=1) for pos in v_fingertip_pos], dim=1).mean(dim=1)
    dist_palm = torch.norm(v_palm_pos - obj_pos, dim=1)
    
    # -------- Visualization for debugging --------
    # global markers
    # fingertips_tensor = torch.cat(v_fingertip_pos, dim=0) 
    
    # fingertip_indices = torch.zeros(fingertips_tensor.shape[0], dtype=torch.long, device=env.device)
    # palm_indices = torch.ones(v_palm_pos.shape[0], dtype=torch.long, device=env.device)

    # markers.visualize(translations=fingertips_tensor, marker_indices=fingertip_indices)
    
    # all_positions = torch.cat([fingertips_tensor, v_palm_pos], dim=0)
    # all_indices = torch.cat([fingertip_indices, palm_indices], dim=0)
    # markers.visualize(translations=all_positions, marker_indices=all_indices)
    # ---------------------------------------------
    
    # print(f"Object distance reward shape: {dist_fingertips.shape}")
    # print(f"Object distance reward shape: {dist_palm.shape}")
    # print(f"Env 0 object position: {obj_pos[0].cpu().numpy()}, avg fingertip distance: {dist_fingertips[0].cpu().numpy()}, palm distance: {dist_palm[0].cpu().numpy()}")
    
    return -2.0 * dist_fingertips - dist_palm

def object_distance_reward_tanh(env: ManagerBasedRLEnv, std: float, fingertip_names: list, palm_name: str) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    v_fingertip_pos, v_palm_pos = get_virtual_link_poses(env, fingertip_names, palm_name)
    
    dist_fingertips = torch.stack([torch.norm(pos - obj_pos, dim=1) for pos in v_fingertip_pos], dim=1).mean(dim=1)
    dist_palm = torch.norm(v_palm_pos - obj_pos, dim=1)
    
    dist_sum = 2.0 * dist_fingertips + dist_palm
    return 1 - torch.tanh(dist_sum / std)

def object_height_reward(
    env: ManagerBasedRLEnv, 
    fingertip_names: list, 
    palm_name: str,
    table_height: float = 1.0,
    target_lift_height: float = 0.6
) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    h = obj_pos[:, 2]
    target_h = table_height + target_lift_height

    v_fingertip_pos, v_palm_pos = get_virtual_link_poses(env, fingertip_names, palm_name)
    
    avg_dist_fingertips = torch.stack([torch.norm(pos - obj_pos, dim=1) for pos in v_fingertip_pos], dim=1).mean(dim=1)
    dist_palm = torch.norm(v_palm_pos - obj_pos, dim=1)

    # 논문 조건: 손이 너무 멀면 보상 0
    out_of_reach = (avg_dist_fingertips >= 0.12) & (dist_palm >= 0.15)

    diff = h - target_h
    abs_diff = torch.abs(diff)
    
    reward = 0.9 + (-2.0 * abs_diff) + diff + (1.0 / (abs_diff + 1.0))
    
    # print(f"Object height reward shape: {reward.shape}")
    # print(f"Env 0 object height: {h[0].cpu().numpy()}, target height: {target_h}, height reward: {reward[0].cpu().numpy()}, out of reach: {out_of_reach[0].cpu().numpy()}")

    return torch.where(out_of_reach, torch.zeros_like(reward), reward)

def object_horizontal_displacement_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    env_origins = env.scene.env_origins
    obj_pos = env.scene["object"].data.root_pos_w - env_origins
    obj_init_pos = env.scene["object"].data.default_root_state[:, :3]
    displacement_xy = torch.norm(obj_pos[:, :2] - obj_init_pos[:, :2], dim=1)
    
    # print(f"Object horizontal displacement reward shape: {displacement_xy.shape}")
    # print(f"Env 0 object position: {obj_pos[0].cpu().numpy()}, initial position: {obj_init_pos[0].cpu().numpy()}, displacement: {displacement_xy[0].cpu().numpy()}")
    
    return -0.3 * displacement_xy

def success_reward(
    env: ManagerBasedRLEnv, 
    fingertip_names: list, 
    palm_name: str,
    table_height: float = 1.0,
    target_lift_height: float = 0.6, 
    threshold: float = 0.05,
    required_steps: int = 30  # 유지해야 하는 스텝 수
) -> torch.Tensor:
    # 1. 환경에 카운터 버퍼가 없는 경우 초기화 (최초 1회)
    if not hasattr(env, "success_consecutive_steps"):
        env.success_consecutive_steps = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # 2. 현재 상태의 성공 여부 계산 (기존 로직)
    obj_pos = env.scene["object"].data.root_pos_w
    h = obj_pos[:, 2]
    target_h = table_height + target_lift_height
    height_condition = torch.abs(h - target_h) <= threshold

    v_fingertip_pos, v_palm_pos = get_virtual_link_poses(env, fingertip_names, palm_name)
    avg_dist_fingertips = torch.stack([torch.norm(pos - obj_pos, dim=1) for pos in v_fingertip_pos], dim=1).mean(dim=1)
    dist_palm = torch.norm(v_palm_pos - obj_pos, dim=1)

    grasp_condition = (avg_dist_fingertips <= 0.12) | (dist_palm <= 0.15)
    is_now_success = height_condition & grasp_condition

    # 3. 카운터 업데이트
    # 성공 조건을 만족하면 +1, 만족하지 못하면 바로 0으로 리셋
    env.success_consecutive_steps = torch.where(
        is_now_success, 
        env.success_consecutive_steps + 1, 
        torch.zeros_like(env.success_consecutive_steps)
    )

    # 4. 환경이 리셋(termination)된 경우 카운터도 리셋해줘야 해
    # env.reset_buf는 리셋되어야 하는 환경의 인덱스가 1(True)로 표시됨
    env.success_consecutive_steps[env.reset_buf > 0] = 0

    # 5. 30 step 이상 유지되었을 때만 보상 지급
    reward = torch.where(
        env.success_consecutive_steps >= required_steps,
        torch.tensor(200.0, device=env.device),
        torch.tensor(0.0, device=env.device)
    )

    return reward