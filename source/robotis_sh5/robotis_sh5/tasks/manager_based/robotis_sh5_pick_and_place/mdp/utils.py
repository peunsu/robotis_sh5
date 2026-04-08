# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_inv, quat_apply, quat_mul, subtract_frame_transforms, quat_error_magnitude

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

def get_trajectory_data(env: ManagerBasedRLEnv, file_path: str, frame_idx: int = 0, obj_idx: int = 0):
    """
    npy 파일에서 특정 프레임의 데이터를 추출하여 
    오브젝트 포즈, 오브젝트 기준 핸드 포즈, 관절 각도, 관절 이름을 반환합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")
    
    # 1. 데이터 로드
    data = np.load(file_path, allow_pickle=True).item()
    
    # --- World Frame 데이터 추출 ---
    obj_pos_w = torch.tensor(data["obj_poses"][obj_idx][frame_idx], device=env.device, dtype=torch.float32)
    obj_quat_w = torch.tensor(data["obj_quats"][obj_idx][frame_idx], device=env.device, dtype=torch.float32)
    
    hand_pos_w = torch.tensor(data["root_pos"][frame_idx], device=env.device, dtype=torch.float32)
    hand_quat_w = torch.tensor(data["root_quat"][frame_idx], device=env.device, dtype=torch.float32)

    # 2. Object Frame 기준의 Hand Pose 계산 (Relative Pose)
    obj_quat_inv = quat_inv(obj_quat_w)
    
    # 상대 위치: R_inv * (hand_pos_w - obj_pos_w)
    relative_hand_pos = quat_apply(obj_quat_inv, (hand_pos_w - obj_pos_w).view(1, 3)).squeeze(0)
    
    # 상대 회전: obj_quat_inv * hand_quat_w
    relative_hand_quat = quat_mul(obj_quat_inv, hand_quat_w)

    # 3. 추가 정보 추출 (qpos 및 joint_names)
    # qpos shape: (num_frames, num_joints)
    qpos = torch.tensor(data["qpos"][frame_idx], device=env.device, dtype=torch.float32)
    joint_names = data["joint_names"]

    # 결과 반환
    return {
        "obj_pos": obj_pos_w,               # World 기준 오브젝트 위치
        "obj_quat": obj_quat_w,             # World 기준 오브젝트 회전
        "root_pos": relative_hand_pos,      # 오브젝트 기준 핸드(Root) 상대 위치
        "root_quat": relative_hand_quat,    # 오브젝트 기준 핸드(Root) 상대 회전
        "qpos": qpos,                       # 로봇 손가락 관절 각도
        "joint_names": joint_names          # 관절 이름 리스트
    }
    
def get_scaled_wrist_force(robot: Articulation, wrist_link_idx: int) -> torch.Tensor:
    """
    robot: Articulation 객체
    wrist_link_idx: 힘을 측정할 손목 링크(Link)의 인덱스
    반환값: Z축 방향으로 작용하는 Force (scaled, torch.Tensor)
    """
    forces = robot.root_physx_view.get_link_incoming_joint_force()

    # 2. 특정 손목 링크의 Z축 Force 추출 (index 2)
    # forces shape: (num_envs, num_links, 6) -> (Fx, Fy, Fz, Tx, Ty, Tz)
    wrist_force_z = forces[:, wrist_link_idx, 2]

    # 3. 수치 안정성을 위한 스케일링
    return wrist_force_z # * 0.001

def get_grasping_flags(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg, 
    object_name: str, 
    fingertip_names: list, 
    palm_name: str,
    thresholds: dict = {"lambda_fingertip": 0.60, "lambda_palm": 0.12, "lambda_d_obj": 0.05}
) -> dict:
    """
    현재 에피소드의 파지 상태 플래그(f1, f2, f3)를 계산하여 딕셔너리로 반환합니다.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    obj = env.scene[object_name]
    
    virtual_fingertip_pos, virtual_palm_pos = get_virtual_link_poses(env, fingertip_names, palm_name)
    obj_pos_w = obj.data.root_pos_w
    
    # --- 1. f1: Joint Angle Error (손가락 관절 일치도) ---
    # num_fingers = len(command_term.robot_finger_indices)
    # target_qpos = command[:, 7:7+num_fingers]
    # current_qpos = robot.data.joint_pos[:, command_term.robot_finger_indices]
    # f1_dist = torch.sum(torch.abs(current_qpos - target_qpos), dim=-1)
    # is_f1 = (f1_dist < thresholds["lambda_fingertip"]).float()
    
    # --- 1. f1: Fingertip Distance (손가락 끝 - 물체 거리) ---
    f1_dist = torch.zeros(env.num_envs, device=env.device)
    for finger_pos in virtual_fingertip_pos:
        f1_dist += torch.norm(finger_pos - obj_pos_w, p=2, dim=-1)
    f1_dist += torch.norm(virtual_palm_pos - obj_pos_w, p=2, dim=-1)
    is_f1 = (f1_dist < thresholds["lambda_fingertip"]).int()
    
    # --- 2. f2: Palm Distance (손바닥과 물체의 거리) ---
    f2_dist = torch.norm(virtual_palm_pos - obj_pos_w, p=2, dim=-1)
    is_f2 = (f2_dist < thresholds["lambda_palm"]).int()

    # --- 3. f3: Lifting Distance (물체 - 타겟 높이 거리) ---
    target_pos_w = command[:, -3:]
    d_obj = torch.norm(obj_pos_w - target_pos_w, p=2, dim=-1)
    is_f3 = (d_obj < thresholds["lambda_d_obj"]).int() # 0.05m 이내로 들어오면 성공으로 간주

    return {
        "is_f1": is_f1, # 관절 각도 조건 만족
        "is_f2": is_f2, # 손가락 위치 조건 만족
        "is_f3": is_f3, # 들어올리기 성공
        "d_obj": d_obj, # 물체-목표 거리 (moving_reward용)
        "f_total": is_f1 + is_f2 + is_f3 # 총합 (3일 때 완전 성공)
    }

# [1] 손의 위치 오차 (L2 Norm)
def compute_hand_pos_error(env, command, asset_cfg, ee_link_name):
    robot = env.scene[asset_cfg.name]
    ee_link_id = robot.find_bodies(ee_link_name)[0][0]
    ee_pos_w = robot.data.body_state_w[:, ee_link_id, :3]
    ee_quat_w = robot.data.body_state_w[:, ee_link_id, 3:7]
    
    # Base 기준 상대 좌표 변환
    curr_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    
    target_pos_b = command[:, :3]
    delta_pos = target_pos_b - curr_pos_b
    return torch.norm(delta_pos, p=2, dim=-1)

# [2] 손의 회전 오차 (Geodesic Distance)
def compute_hand_rot_error(env, command, asset_cfg, ee_link_name):    
    robot = env.scene[asset_cfg.name]
    ee_link_id = robot.find_bodies(ee_link_name)[0][0]
    
    target_quat_b = command[:, 3:7]
    
    # 2. 현재 End-effector의 Base 기준 상대 회전 계산
    ee_quat_w = robot.data.body_state_w[:, ee_link_id, 3:7]
    root_quat_w = robot.data.root_quat_w
    
    # 베이스 기준 현재 EE 회전 (q_rel = q_root_inv * q_ee)
    _, curr_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, root_quat_w, 
        robot.data.body_state_w[:, ee_link_id, :3], ee_quat_w
    )
    
    # 3. 두 쿼터니언 사이의 각도 차이 (라디안) 계산
    return quat_error_magnitude(curr_quat_b, target_quat_b)

# [3] 손가락 관절 오차 (L1 Norm)
def compute_finger_qpos_error(env, command, command_term):
    robot = env.scene[command_term.cfg.asset_name]
    
    # command에서 손가락 부분 추출
    num_fingers = len(command_term.robot_finger_indices)
    target_qpos = command[:, 7:7+num_fingers]
    
    # 현재 로봇의 손가락 관절 값
    current_qpos = robot.data.joint_pos[:, command_term.robot_finger_indices]
    
    delta_qpos = target_qpos - current_qpos
    return torch.norm(delta_qpos, p=1, dim=-1)