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
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_error_magnitude, quat_mul
from .utils import get_scaled_wrist_force

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_angle_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    
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
    palm_name: str, 
    object_name: str
) -> torch.Tensor:
    """
    수식 (7): r_reach = -sum ||x_finger - x_obj||2
    각 손가락 끝(및 손바닥)과 오브젝트 사이의 거리 페널티를 계산해.
    """
    # 1. 가상 링크 포즈 가져오기 (이미 구현한 함수 호출)
    # virtual_fingertip_pos: list of (num_envs, 3)
    # virtual_palm_pos: (num_envs, 3)
    virtual_fingertip_pos, virtual_palm_pos = get_virtual_link_poses(env, fingertip_names, palm_name)
    
    # 2. 오브젝트 현재 위치 (World Frame)
    obj = env.scene[object_name]
    obj_pos_w = obj.data.root_pos_w  # (num_envs, 3)
    
    # 3. 각 손가락 끝과 오브젝트 사이의 거리 계산
    total_dist = torch.zeros(env.num_envs, device=env.device)
    
    # 모든 손가락 끝에 대해 거리 합산
    for finger_pos in virtual_fingertip_pos:
        total_dist += torch.norm(finger_pos - obj_pos_w, p=2, dim=-1)
        
    # 손바닥(Palm)과의 거리도 포함할지 선택 가능 (수식에 따라 포함)
    # total_dist += torch.norm(virtual_palm_pos - obj_pos_w, p=2, dim=-1)
    
    # 가중치 제외하고 거리 합산값 반환 (RewTerm에서 weight로 -wr 적용)
    return total_dist

def lifting_reward_fullbody(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg,
    object_name: str,
    fingertip_names: list,
    palm_name: str,
    wrist_link_name: str, # 손목 링크 이름 추가
    thresholds: dict = {
        "lambda_f1": 0.12,
        "lambda_f2": 0.6,
        "lambda_0": 0.05
    }
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    obj = env.scene[object_name]
    
    # --- 1. f 계산 (Distance Terms) ---
    # f1: 손가락 관절만 비교 (target_qpos에서 손가락 부분만 슬라이싱)
    num_fingers = len(command_term.robot_finger_indices)
    target_qpos = command[:, 7:7+num_fingers]
    # 전체 관절 중 데이터셋과 매핑된(손가락) 부분만 오차 계산
    # (command_term에서 만든 finger_mask를 활용하면 더 좋아)
    f1_dist = torch.sum(torch.abs(robot.data.joint_pos[:, command_term.robot_finger_indices] - target_qpos), dim=-1)

    # f2: 가상 손가락 끝 - 물체 거리 (구현한 함수 활용)
    virtual_fingertip_pos, virtual_palm_pos = get_virtual_link_poses(env, fingertip_names, palm_name)
    obj_pos_w = obj.data.root_pos_w
    f2_dist = torch.zeros(env.num_envs, device=env.device)
    for finger_pos in virtual_fingertip_pos:
        f2_dist += torch.norm(finger_pos - obj_pos_w, p=2, dim=-1)
    f2_dist += torch.norm(virtual_palm_pos - obj_pos_w, p=2, dim=-1)

    # d_obj: 물체 - 타겟(30cm 위) 거리
    target_pos_w = command[:, -3:]
    d_obj = torch.norm(obj_pos_w - target_pos_w, p=2, dim=-1)

    # --- 2. 조건 판단 (f = 3) ---
    is_f1 = (f1_dist < thresholds["lambda_f1"]).float()
    is_f2 = (f2_dist < thresholds["lambda_f2"]).float()
    is_f3 = (d_obj > thresholds["lambda_0"]).float()
    f = is_f1 + is_f2 + is_f3

    # --- 3. az 계산 (Measured Joint Force 사용) ---
    wrist_idx = robot.find_bodies(wrist_link_name)[0][0]
    a_z = get_scaled_wrist_force(robot, wrist_idx)

    # --- 4. 최종 리워드 ---
    reward = torch.where(f >= 3.0, 1.0 * (1.0 + a_z), torch.zeros_like(f))
    
    return reward

def moving_reward(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    object_name: str,
    weight_m: float = 2.0,
    weight_b: float = 10.0,
    lambda_0: float = 0.05
) -> torch.Tensor:
    """
    수식 (9): r_move 계산
    물체가 목표 지점(target_pos_w)에 가까워질수록 보상을 주며, 
    lambda_0 이내일 때 보너스 항을 추가함.
    """
    command = env.command_manager.get_command(command_name)
    obj = env.scene[object_name]

    # 1. d_obj 계산: 현재 오브젝트 위치와 목표 위치(command의 마지막 3차원) 사이의 거리
    target_pos_w = command[:, -3:]
    curr_obj_pos_w = obj.data.root_pos_w
    d_obj = torch.norm(curr_obj_pos_w - target_pos_w, p=2, dim=-1)

    # 2. 기본 페널티 항: -wm * d_obj
    reward = -weight_m * d_obj

    # 3. 보너스 조건 판단: d_obj < lambda_0
    # lambda_0 이내일 때 보너스 term인 1 / (1 + wb * d_obj) 를 더해줌
    bonus = 1.0 / (1.0 + weight_b * d_obj)
    
    # torch.where를 사용하여 조건부 리워드 적용
    reward = torch.where(d_obj < lambda_0, reward + bonus, reward)

    return reward