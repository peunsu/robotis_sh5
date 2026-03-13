# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class WaypointManager:
    def __init__(self, env: "ManagerBasedRLEnv", num_waypoints: int):
        self.env = env
        self.num_envs = env.num_envs
        self.num_waypoints = num_waypoints
        self.device = env.device
        
        # 데이터 버퍼
        self.target_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.waypoints = torch.zeros((self.num_envs, self.num_waypoints, 3), device=self.device)
        self.prev_dist = torch.zeros(self.num_envs, device=self.device)
        
        self._marker_visualizer = None

    def update_visuals(self):
        if self._marker_visualizer is None:
            self._marker_visualizer = getattr(self.env.scene, "waypoint_markers", None)

        if self._marker_visualizer is not None:
            # 1. 모든 위치 데이터 (N_envs * num_waypoints, 3)
            all_positions = self.waypoints.view(-1, 3)
            
            # 2. 마커 인덱스 생성
            # 모든 마커를 일단 1번('future', 초록색)으로 초기화
            marker_indices = torch.ones((self.num_envs, self.num_waypoints), 
                                    dtype=torch.int32, device=self.device)
            
            # 각 환경의 '현재 타겟'만 0번('current', 빨간색)으로 변경
            marker_indices[torch.arange(self.num_envs), self.target_indices] = 0
            
            # 3. 평탄화 (Flatten)
            marker_indices = marker_indices.view(-1)

            # [핵심] 리스트로 바꾸지 말고 텐서 그대로 전달하되, 
            # 데이터가 비어있지 않은지 확인 루틴 추가
            if all_positions.shape[0] > 0:
                self._marker_visualizer.visualize(
                    translations=all_positions, 
                    marker_indices=marker_indices
                )
    
    def update(self, threshold: float):
        # 모든 환경의 root_pos와 current_target을 비교
        root_pos = self.env.scene["robot"].data.root_pos_w[:, :2]
        current_target = self.waypoints[torch.arange(self.num_envs), self.target_indices, :2]
        
        current_dist = torch.norm(current_target - root_pos, dim=-1)
        goal_reached = current_dist < threshold
        
        if torch.any(goal_reached):
            # 도달한 환경들만 인덱스 증가
            self.target_indices[:] = torch.clamp(
                self.target_indices + goal_reached.long(), 
                max=self.num_waypoints - 1
            )
            self.update_visuals()
            
def get_or_create_waypoint_manager(env: "ManagerBasedRLEnv", num_waypoints: int = 10) -> "WaypointManager":
    if not hasattr(env, "waypoint_manager"):
        # 1. 매니저 생성
        env.waypoint_manager = WaypointManager(env, num_waypoints)
        
        # 2. [핵심] 실제 시각화 마커 객체를 여기서 생성해서 scene에 수동 등록!
        # env.cfg.waypoint_marker_cfg에 담아둔 설정을 사용함
        if hasattr(env.cfg, "waypoint_marker_cfg"):
            env.scene.waypoint_markers = VisualizationMarkers(env.cfg.waypoint_marker_cfg)
            
    return env.waypoint_manager