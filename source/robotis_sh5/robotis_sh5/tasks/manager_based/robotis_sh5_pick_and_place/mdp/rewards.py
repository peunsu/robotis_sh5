# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def get_virtual_link_poses(env: ManagerBasedRLEnv, fingertip_names: list, palm_name: str):
    robot = env.scene["robot"]
    num_envs = env.num_envs  # 현재 16개 환경

    virtual_fingertip_pos = []
    for name in fingertip_names:
        body_indices = robot.find_bodies(name)[0]
        p_pos = robot.data.body_pos_w[:, body_indices]
        p_quat = robot.data.body_quat_w[:, body_indices]
        
        if p_pos.ndim == 3 and p_pos.shape[1] == 1:
            p_pos = p_pos.squeeze(1)
            p_quat = p_quat.squeeze(1)
        
        is_left = "_l_" in name
        if "link4" in name or "link1" in name:
            offset_y = -0.03975 if is_left else 0.03975
            offset = [0.0, offset_y, 0.0]
        else:
            offset = [0.0, 0.0, 0.02425]
            
        # (1, 3) 텐서를 만든 후 현재 환경 개수만큼 확장
        offset_tensor = torch.tensor(offset, device=env.device).view(1, 3).expand(num_envs, 3)
        
        # 회전 적용 후 더하기 -> 결과: (num_envs, 3)
        v_pos = p_pos + quat_apply(p_quat, offset_tensor)
        virtual_fingertip_pos.append(v_pos)
    
    # Palm 포지션 계산 -> 결과: (num_envs, 3)
    palm_idx = robot.find_bodies(palm_name)[0]
    p_pos_palm = robot.data.body_pos_w[:, palm_idx]
    p_quat_palm = robot.data.body_quat_w[:, palm_idx]
    
    if p_pos_palm.ndim == 3 and p_pos_palm.shape[1] == 1:
        p_pos_palm = p_pos_palm.squeeze(1)
        p_quat_palm = p_quat_palm.squeeze(1)
    
    palm_offset = torch.tensor([0.01, 0.0, 0.06], device=env.device).view(1, 3).expand(num_envs, 3)
    virtual_palm_pos = p_pos_palm + quat_apply(p_quat_palm, palm_offset)
    
    # print(f"Virtual fingertip positions shape: {virtual_fingertip_pos[0].shape}")
    # print(f"Virtual palm position shape: {virtual_palm_pos.shape}")
    
    return virtual_fingertip_pos, virtual_palm_pos

def object_distance_reward(env: ManagerBasedRLEnv, fingertip_names: list, palm_name: str) -> torch.Tensor:
    # 1. 물체 중심 위치
    obj_pos = env.scene["object"].data.root_pos_w
    
    # 2. 가상 위치 계산
    v_fingertip_pos, v_palm_pos = get_virtual_link_poses(env, fingertip_names, palm_name)
    
    # 3. 거리 계산
    dist_fingertips = torch.stack([torch.norm(pos - obj_pos, dim=1) for pos in v_fingertip_pos], dim=1).mean(dim=1)
    dist_palm = torch.norm(v_palm_pos - obj_pos, dim=1)
    
    # print(f"Object distance reward shape: {dist_fingertips.shape}")
    # print(f"Object distance reward shape: {dist_palm.shape}")
    
    return -2.0 * dist_fingertips - dist_palm

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

    return torch.where(out_of_reach, torch.zeros_like(reward), reward)

def object_horizontal_displacement_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    obj_init_pos = env.scene["object"].data.default_root_state[:, :3]
    displacement_xy = torch.norm(obj_pos[:, :2] - obj_init_pos[:, :2], dim=1)
    # print(f"Object horizontal displacement reward shape: {displacement_xy.shape}")
    return -0.3 * displacement_xy

def success_reward(
    env: ManagerBasedRLEnv, 
    fingertip_names: list, 
    palm_name: str,
    table_height: float = 1.0,
    target_lift_height: float = 0.6, 
    threshold: float = 0.05
) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    h = obj_pos[:, 2]
    target_h = table_height + target_lift_height

    height_condition = torch.abs(h - target_h) <= threshold

    v_fingertip_pos, v_palm_pos = get_virtual_link_poses(env, fingertip_names, palm_name)
    
    avg_dist_fingertips = torch.stack([torch.norm(pos - obj_pos, dim=1) for pos in v_fingertip_pos], dim=1).mean(dim=1)
    dist_palm = torch.norm(v_palm_pos - obj_pos, dim=1)

    # 논문 성공 조건 반영
    grasp_condition = (avg_dist_fingertips <= 0.12) | (dist_palm <= 0.15)
    is_success = height_condition & grasp_condition

    # print(f"Success reward shape: {is_success.shape}")
    return torch.where(is_success, torch.tensor(200.0, device=env.device), torch.tensor(0.0, device=env.device))