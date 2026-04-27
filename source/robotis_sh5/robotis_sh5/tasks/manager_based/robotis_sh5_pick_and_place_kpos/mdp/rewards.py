# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .utils import get_virtual_link_poses

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_error_magnitude, quat_mul
from .utils import (
    get_wrist_acc, get_grasping_flags, get_object_acc,
    compute_finger_qpos_error, compute_finger_kpos_error, compute_hand_pos_error, compute_hand_rot_error
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_angle_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    
    finger_qpos_error = compute_finger_qpos_error(env, command, command_term)
    return finger_qpos_error

def joint_position_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    
    finger_kpos_error = compute_finger_kpos_error(env, command, command_term)
    return finger_kpos_error

def root_translation_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """수식 두 번째 항: ||th_obj - th_g||2"""
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    hand_pos_error = compute_hand_pos_error(env, command, asset_cfg, asset_cfg.body_names[0])
    return hand_pos_error

def root_rotation_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """수식 세 번째 항: L_rot (Geodesic distance)"""
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    hand_rot_error = compute_hand_rot_error(env, command, asset_cfg, asset_cfg.body_names[0])
    return hand_rot_error

def reaching_reward(
    env: ManagerBasedRLEnv, 
    fingertip_names: list, 
    wrist_link_name: str, 
    object_name: str
) -> torch.Tensor:
    """
    수식 (7): r_reach = -sum ||x_finger - x_obj||2
    각 손가락 끝(및 손바닥)과 오브젝트 사이의 거리 페널티를 계산해.
    """
    # 1. 가상 링크 포즈 가져오기 (이미 구현한 함수 호출)
    # virtual_fingertip_pos: list of (num_envs, 3)
    # virtual_palm_pos: (num_envs, 3)
    virtual_fingertip_pos, virtual_palm_pos = get_virtual_link_poses(env, fingertip_names, wrist_link_name)
    
    # 2. 오브젝트 현재 위치 (World Frame)
    obj = env.scene[object_name]
    obj_pos_w = obj.data.root_pos_w  # (num_envs, 3)
    
    # 3. 각 손가락 끝과 오브젝트 사이의 거리 계산
    total_dist = torch.zeros(env.num_envs, device=env.device)
    
    # 모든 손가락 끝에 대해 거리 합산
    for finger_pos in virtual_fingertip_pos:
        total_dist += torch.norm(finger_pos - obj_pos_w, p=2, dim=-1)
        
    # 손바닥(Palm)과의 거리도 포함할지 선택 가능 (수식에 따라 포함)
    total_dist += 2 * torch.norm(virtual_palm_pos - obj_pos_w, p=2, dim=-1)
    
    # 가중치 제외하고 거리 합산값 반환 (RewTerm에서 weight로 -wr 적용)
    return total_dist

def lifting_reward_fullbody(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg,
    object_name: str,
    sensor_name: str,
    fingertip_names: list,
    wrist_link_name: str,
    wrist_joint_name: str,
    thresholds: dict = {"lambda_fingertip": 0.60, "lambda_palm": 0.12, "lambda_d_obj": 0.05}
) -> torch.Tensor:
    # 1. 플래그 유틸리티 호출
    flags = get_grasping_flags(env, command_name, asset_cfg, object_name, fingertip_names, wrist_link_name, thresholds)
    
    # f_z = -torch.sum(torch.mean(env.scene[sensor_name].data.force_matrix_w_history[:, :, 0, :, 2], dim=1), dim=-1)  # Z축 방향 힘
    # f_z = torch.clip(f_z, min=0.0)  # 음수는 보상으로 계산하지 않음 (들어올리는 힘만 보상)
    
    # 2. 손목 가속도(a_z) 계산
    # a_z = get_wrist_acc(env, wrist_joint_name)
    a_z = get_object_acc(env, object_name)[:, 2]  # 오브젝트의 수직 가속도 사용 (lifting 성공 여부에 더 직접적일 수 있음)
    a_z = torch.clip(a_z, min=-1.0)  # 음수 가속도는 패널티로 계산 (떨어지는 경우), 양수 가속도는 보상으로 계산 (들어올리는 경우)
    
    # 3. f == 3 (모든 조건 만족)일 때만 보상 지급
    reward = torch.where(flags["is_f1"] + flags["is_f2"] == 2, (1.0 + a_z), torch.zeros_like(a_z))
    # reward = torch.where(flags["is_f1"] + flags["is_f2"] == 2, 0.1 * (1.0 + f_z), torch.zeros_like(f_z))
    # reward = torch.where(flags["is_f1"] + flags["is_f2"] + flags["is_f3"] == 3, 0.2, reward)
    
    return reward

def moving_reward(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg, # 추가
    object_name: str,
    fingertip_names: list,     # 추가
    wrist_link_name: str,
    weight_m: float = 2.0,
    weight_b: float = 10.0,
    thresholds: dict = {"lambda_fingertip": 0.60, "lambda_palm": 0.12, "lambda_d_obj": 0.05}
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    
    # 1. 플래그 유틸리티 호출 (d_obj만 가져와서 사용)
    flags = get_grasping_flags(env, command_name, asset_cfg, object_name, fingertip_names, wrist_link_name, thresholds)
    d_obj = flags["d_obj"]
    
    target_flag = sum([
        (compute_hand_pos_error(env, command, asset_cfg, wrist_link_name) < 0.4).int(),  # 0.4
        (compute_hand_rot_error(env, command, asset_cfg, wrist_link_name) < 0.5).int(),  # 1.0
        (compute_finger_kpos_error(env, command, command_term) < 0.5).int()  # 0.1
    ])

    # 2. 기본 거리 페널티
    # reward = weight_m * (0.3 - d_obj)  # 부호가 잘못되어 있었음
    reward = torch.where(flags["is_f1"] + flags["is_f2"] + target_flag == 5, torch.clip(weight_m * (0.3 - d_obj), min=0.05), torch.zeros_like(d_obj))
    # reward = torch.where(flags["is_f1"] + flags["is_f2"] == 2, torch.clip(weight_m * (0.3 - d_obj), min=0.0), torch.zeros_like(d_obj))

    # 3. 보너스 조건 (d_obj < lambda_d_obj)
    # bonus = 1.0 / (1.0 + weight_b * d_obj)
    # reward = torch.where(flags["is_f1"] + flags["is_f2"] + flags["is_f3"] == 3, reward + bonus, reward)

    return reward

def contact_forces_reward(
    env: ManagerBasedRLEnv,
    command_name: str, 
    asset_cfg: SceneEntityCfg,
    wrist_link_name: str,
    sensor_names: list,
    threshold: float = 1.0
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)

    num_sensors = len(sensor_names)
    forces = torch.zeros((env.num_envs, num_sensors), device=env.device)
    contacted_flags = {}
    
    for i, name in enumerate(sensor_names):
        sensor: ContactSensor = env.scene[name]
        
        # 원래 텐서 (num_envs, time_history, num_bodies, num_filters, 3)
        # 필터링된 힘 합산 (num_envs, num_bodies, 3)
        filtered_net_forces = torch.mean(torch.sum(sensor.data.force_matrix_w_history, dim=3), dim=1)
        
        # 각 바디별 힘의 크기 계산 (L2 Norm)
        force_mag = torch.norm(filtered_net_forces[:, 0, :], dim=-1)
        
        forces[:, i] = force_mag
        contacted_flags[name] = (force_mag >= threshold)

    good_contact = contacted_flags[sensor_names[0]] & (
        contacted_flags[sensor_names[1]] | contacted_flags[sensor_names[2]]
        | contacted_flags[sensor_names[3]] | contacted_flags[sensor_names[4]]
    )

    target_flag = sum([
        (compute_hand_pos_error(env, command, asset_cfg, wrist_link_name) < 0.4).int(),  # 0.4
        (compute_hand_rot_error(env, command, asset_cfg, wrist_link_name) < 0.5).int(),  # 1.0
        (compute_finger_kpos_error(env, command, command_term) < 0.5).int()  # 0.1
    ])

    reward = torch.where(good_contact & (target_flag == 3), 1.0, 0.0)  # 모든 조건 만족 시 보상 1.0, 그렇지 않으면 0.0

    return reward