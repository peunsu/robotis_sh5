# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .utils import get_virtual_link_poses

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# 전역 변수로 마커 객체를 하나 만들어둠 (처음 호출될 때 초기화)
_FINGERTIP_VISUALIZER = None
_PALM_VISUALIZER = None

def visual_marker_obs(env, fingertip_names, palm_name):
    global _FINGERTIP_VISUALIZER, _PALM_VISUALIZER
    
    # 1. 마커가 없으면 여기서 딱 한 번 생성
    if _FINGERTIP_VISUALIZER is None:
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/FingertipMarkers",
            markers={"s": sim_utils.SphereCfg(radius=0.01, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))}
        )
        _FINGERTIP_VISUALIZER = VisualizationMarkers(cfg)
        
        palm_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PalmMarker",
            markers={"s": sim_utils.SphereCfg(radius=0.02, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))}
        )
        _PALM_VISUALIZER = VisualizationMarkers(palm_cfg)

    # 2. 은수가 만든 함수로 위치 계산
    v_fingertips, v_palm = get_virtual_link_poses(env, fingertip_names, palm_name)

    # 3. 마커 업데이트
    fingertip_pos_stack = torch.stack(v_fingertips, dim=1).view(-1, 3)
    _FINGERTIP_VISUALIZER.visualize(translations=fingertip_pos_stack)
    _PALM_VISUALIZER.visualize(translations=v_palm)

    # Observation 함수는 텐서를 리턴해야 하니까 빈 값을 줘 (학습엔 영향 없음)
    return torch.zeros((env.num_envs, 0), device=env.device)

def object_obs(
    env: ManagerBasedRLEnv,
    # left_eef_link_name: str,
    right_eef_link_name: str,
) -> torch.Tensor:
    """
    Object observations (in world frame):
        object pos,
        object quat,
        left_eef to object,
        right_eef to object,
    """
    # body_pos_w = env.scene["robot"].data.body_pos_w
    # left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    # right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    # left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    # right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["object"].data.root_quat_w

    # left_eef_to_object = object_pos - left_eef_pos
    # right_eef_to_object = object_pos - right_eef_pos
    
    # print(f"Object position: {object_pos}, Object quaternion: {object_quat}")
    # print(f"Right EEF position: {right_eef_pos}")

    return torch.cat(
        (
            object_pos,
            object_quat,
            #left_eef_to_object,
            #right_eef_to_object,
        ),
        dim=1,
    )


def body_pose_relative_to_env(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """각 환경의 origin을 기준으로 한 body의 [pos, quat]를 반환해."""
    
    # 1. 자산(robot 등) 추출
    asset = env.scene[asset_cfg.name]
    
    # 2. 월드 기준 포즈 데이터 가져오기 (num_envs, num_bodies, 3/4)
    # body_ids가 지정되어 있으면 해당되는 바디들만 가져옴
    body_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids, :3]
    body_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids, 3:7]
    
    # 3. 각 환경의 Origin 가져오기 (num_envs, 3)
    env_origins = env.scene.env_origins # 각 환경의 (x, y, z) 원점
    
    # 4. 좌표 변환 (World -> Env Relative)
    # 위치(pos)는 단순히 환경 원점을 빼주면 되고, 
    # 회전(quat)은 환경 자체가 회전되어 있지 않다면 월드와 동일해.
    # 만약 환경 자체가 회전되어 있다면 subtract_frame_transforms를 쓰는 게 안전해.
    
    # 각 환경 원점을 body 개수만큼 확장 (num_envs, 1, 3) -> (num_envs, num_bodies, 3)
    num_bodies = body_pos_w.shape[1]
    env_origins_expanded = env_origins.unsqueeze(1).expand(-1, num_bodies, -1)
    
    # 상대 위치 계산
    relative_pos = body_pos_w - env_origins_expanded
    
    # 5. 데이터 합치기 및 Flatten (num_envs, num_bodies * 7)
    # 각 body당 [x, y, z, qw, qx, qy, qz] 순서로 맞춤
    relative_poses = torch.cat([relative_pos, body_quat_w], dim=-1)
    
    return relative_poses.view(env.num_envs, -1)

def body_position_relative_to_env(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """각 환경의 origin을 기준으로 한 body의 pos를 반환해."""
    
    # 1. 자산(robot 등) 추출
    asset = env.scene[asset_cfg.name]
    
    # 2. 월드 기준 포즈 데이터 가져오기 (num_envs, num_bodies, 3/4)
    # body_ids가 지정되어 있으면 해당되는 바디들만 가져옴
    body_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids, :3]
    
    # 3. 각 환경의 Origin 가져오기 (num_envs, 3)
    env_origins = env.scene.env_origins # 각 환경의 (x, y, z) 원점
    
    # 각 환경 원점을 body 개수만큼 확장 (num_envs, 1, 3) -> (num_envs, num_bodies, 3)
    num_bodies = body_pos_w.shape[1]
    env_origins_expanded = env_origins.unsqueeze(1).expand(-1, num_bodies, -1)
    
    # 상대 위치 계산
    relative_pos = body_pos_w - env_origins_expanded
    
    return relative_pos.view(env.num_envs, -1)

def contact_forces(env, sensor_names: list):
    """필터링된 대상과의 접촉력 행렬(force_matrix_w)을 사용하여 크기를 계산해."""
    
    all_forces = []
    
    for name in sensor_names:
        # 1. Scene에서 각 센서 객체 호출
        sensor: ContactSensor = env.scene[name]
        
        # 2. 접촉력 행렬 데이터 가져오기 
        # Shape: (num_envs, num_bodies, num_filters, 3)
        force_matrix = sensor.data.force_matrix_w
        
        # 3. 필터링된 모든 대상으로부터 오는 힘을 합산 (num_filters 차원 합산)
        # 결과 Shape: (num_envs, num_bodies, 3)
        # 특정 물체들과의 접촉력만 남고, 바닥 등 필터링되지 않은 힘은 제외돼.
        filtered_net_forces = torch.sum(force_matrix, dim=2)
        
        # 4. 차원을 조정하여 (num_envs, num_bodies * 3) 형태로 변환
        # 결과 Shape: (num_envs, num_bodies * 3)
        force_concat = filtered_net_forces.view(env.num_envs, -1)
        
        all_forces.append(force_concat)
    
    # 5. 모든 센서 데이터를 하나의 텐서로 결합
    # 결과 Shape: (num_envs, total_bodies * 3 * num_sensors)
    combined_forces = torch.cat(all_forces, dim=-1)
    
    # 6. 최종 접촉력 관측값 반환
    return combined_forces