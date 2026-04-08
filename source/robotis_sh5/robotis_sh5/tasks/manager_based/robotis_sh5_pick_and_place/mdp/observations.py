# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.sim as sim_utils
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


def get_eef_pos(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins

    return left_eef_pos


def get_eef_quat(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_quat = body_quat_w[:, left_eef_idx]

    return left_eef_quat

def contact_forces_norm(env, sensor_name: str):
    """오른손가락들의 접촉력 크기(Norm)를 observation으로 반환해."""
    # 1. Scene에서 센서 객체 호출
    sensor: ContactSensor = env.scene[sensor_name]
    
    # 2. 전체 접촉력 데이터 (env, body_id, 3) 가져오기
    # net_forces_w_history를 쓰면 history_length에 따른 과거 데이터도 볼 수 있지만, 
    # 여기서는 최신 값인 net_forces_w를 쓸게.
    net_forces = sensor.data.net_forces_w
    
    # 3. 각 링크별 힘의 크기 계산 (L2 Norm)
    force_magnitudes = torch.norm(net_forces, p=2, dim=-1)
    
    # 에이전트가 학습하기 좋게 적절한 스케일로 조정 (보통 0.01 ~ 0.1 사이)
    return force_magnitudes * 0.05