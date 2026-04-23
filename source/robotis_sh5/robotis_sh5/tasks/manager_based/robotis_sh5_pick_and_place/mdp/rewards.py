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
    compute_finger_qpos_error, compute_hand_pos_error, compute_hand_rot_error
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_angle_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    
    finger_qpos_error = compute_finger_qpos_error(env, command, command_term)
    return finger_qpos_error
    
    # 1. command에서 손가락 부분만 추출 (중간에 딱 손가락 개수만큼 있음)
    # 7번 인덱스부터 (7 + 손가락 개수)까지
    num_fingers = len(command_term.robot_finger_indices)
    target_finger_qpos = command[:, 7:7+num_fingers]
    
    # 2. 로봇에서도 미리 찾아둔 손가락 인덱스만 슬라이싱
    current_finger_qpos = robot.data.joint_pos[:, command_term.robot_finger_indices]
    
    # print(f"Env 0 Joint 1 desired position: {target_finger_qpos[0, 0]}")
    # print(f"Env 0 Joint 1 position: {robot.data.joint_pos[0, command_term.robot_finger_indices[0]]}")
    # print(f"Joint angle error: {torch.sum(torch.abs(current_finger_qpos - target_finger_qpos), dim=-1)}")
    
    # 3. 오차 계산 (훨씬 가벼움!)
    return torch.sum(torch.abs(current_finger_qpos - target_finger_qpos), dim=-1)

def root_translation_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """수식 두 번째 항: ||th_obj - th_g||2"""
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    hand_pos_error = compute_hand_pos_error(env, command, asset_cfg, asset_cfg.body_names[0])
    return hand_pos_error
    
    # 1. 커맨드에 저장된 목표 위치 (Base 기준 상대 좌표)
    # DexYCBCommandTerm의 pose_command_b 부분이 0:3에 해당함
    target_pos_b = command[:, :3]
    
    # 2. 현재 End-effector의 Base 기준 상대 위치 계산
    # EE_pos_w - Robot_root_pos_w 를 하면 베이스 기준 상대 위치가 나옴
    # (더 정확하게는 subtract_frame_transforms를 써야 하지만, 
    #  단순 거리 비교라면 베이스 기준 좌표계 변환만으로도 충분해)
    ee_pos_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    
    # 월드 EE 좌표를 로봇 베이스 좌표계로 변환
    curr_pos_b, _ = subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, robot.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    )
    
    # 3. 목표 상대 위치와 현재 상대 위치 사이의 L2 Norm
    return torch.norm(curr_pos_b - target_pos_b, dim=-1)

def root_rotation_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """수식 세 번째 항: L_rot (Geodesic distance)"""
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    hand_rot_error = compute_hand_rot_error(env, command, asset_cfg, asset_cfg.body_names[0])
    return hand_rot_error
    
    # 1. 목표 회전 (쿼터니언, 인덱스 3:7)
    target_quat_b = command[:, 3:7]
    
    # 2. 현재 End-effector의 Base 기준 상대 회전 계산
    ee_quat_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    root_quat_w = robot.data.root_quat_w
    
    # 베이스 기준 현재 EE 회전 (q_rel = q_root_inv * q_ee)
    _, curr_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, root_quat_w, 
        robot.data.body_state_w[:, asset_cfg.body_ids[0], :3], ee_quat_w
    )
    
    # 3. 두 쿼터니언 사이의 각도 차이 (라디안) 계산
    return quat_error_magnitude(curr_quat_b, target_quat_b)

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
    
    # target_flag = sum([
    #     (compute_hand_pos_error(env, command, asset_cfg, wrist_link_name) < 0.4).int(),
    #     (compute_hand_rot_error(env, command, asset_cfg, wrist_link_name) < 1.0).int(),
    #     (compute_finger_qpos_error(env, command, command_term) < 6.0).int()
    # ])

    # 2. 기본 거리 페널티
    # reward = weight_m * (0.3 - d_obj)  # 부호가 잘못되어 있었음
    # reward = torch.where(flags["is_f1"] + flags["is_f2"] + target_flag == 5, 0.9 - weight_m * d_obj, torch.zeros_like(d_obj))
    reward = torch.where(flags["is_f1"] + flags["is_f2"] == 2, torch.clip(weight_m * (0.3 - d_obj), min=0.0), torch.zeros_like(d_obj))

    # 3. 보너스 조건 (d_obj < lambda_d_obj)
    bonus = 1.0 / (1.0 + weight_b * d_obj)
    reward = torch.where(flags["is_f1"] + flags["is_f2"] + flags["is_f3"] == 3, reward + bonus, reward)

    return reward

def grasp_contact_reward(env, sensor_names: list, threshold: float = 1.0) -> torch.Tensor:
    total_reward = torch.zeros(env.num_envs, device=env.device)
    
    for name in sensor_names:
        sensor: ContactSensor = env.scene[name]
        # 각 센서가 임계값 이상의 힘을 감지하지 못한 경우 패널티 -1, 감지한 경우 0 보상
        not_contacted = (torch.norm(sensor.data.net_forces_w[:, 0, :], dim=-1) < threshold)
        total_reward += not_contacted.float()
        
    return total_reward # 손가락마다 접촉이 안 된 경우 -1, 모두 접촉한 경우 0 보상

def contact_forces_reward(env: ManagerBasedRLEnv, sensor_names: list, threshold: float = 1.0) -> torch.Tensor:      
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
        contacted_flags[name] = (force_mag > threshold)

    good_contact = contacted_flags[sensor_names[0]] & (
        contacted_flags[sensor_names[1]] | contacted_flags[sensor_names[2]]
        | contacted_flags[sensor_names[3]] | contacted_flags[sensor_names[4]]
    )

    return good_contact