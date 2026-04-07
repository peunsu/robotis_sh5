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
from isaaclab.utils.math import quat_inv, quat_apply, quat_mul

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
    return wrist_force_z * 0.001