# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms, compute_pose_error
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from .utils import get_trajectory_data

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class DexYCBCommandTerm(CommandTerm):
    def __init__(self, cfg: DexYCBCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # 자산 및 EE 인덱스
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object = env.scene[cfg.object_name]
        self.ee_body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        
        # [Buffers]
        self.rel_hand_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.rel_hand_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # 29번째 프레임 데이터를 한 번 로드해서 관절 순서 매핑 생성
        self.traj = get_trajectory_data(self._env, self.cfg.file_path, frame_idx=self.cfg.frame_idx)
        dataset_joints = self.traj["joint_names"] # 여기서 가져온 이름 리스트 사용
        
        robot_finger_indices = []
        dataset_finger_indices = []
        
        for i, name in enumerate(self.robot.joint_names):
            if name in dataset_joints:
                robot_finger_indices.append(i)
                dataset_finger_indices.append(dataset_joints.index(name))
        
        # 텐서로 변환 (long 타입)
        self.robot_finger_indices = torch.tensor(robot_finger_indices, device=self.device, dtype=torch.long)
        self.dataset_finger_indices = torch.tensor(dataset_finger_indices, device=self.device, dtype=torch.long)

        # 2. 버퍼 크기를 '진짜 사용할 손가락 개수'로만 설정 (63 -> 손가락 개수)
        num_finger_joints = len(robot_finger_indices)
        self.target_qpos = torch.zeros(self.num_envs, num_finger_joints, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """리셋 시 29번 프레임 데이터 샘플링 및 관절값 매핑"""
        # 1. 상대 포즈 저장
        self.rel_hand_pos[env_ids] = self.traj["root_pos"]
        self.rel_hand_quat[env_ids] = self.traj["root_quat"]
        
        # 2. 관절값 매핑 적용
        raw_qpos = self.traj["qpos"]
        
        self.target_qpos[env_ids] = raw_qpos[self.dataset_finger_indices]

        # [핵심] Lifting Target Position 계산 (World 좌표 변환)
        # npy의 obj_pos + env_origin + table_height + offset(0.3m)
        env_origins = self._env.scene.env_origins[env_ids]
        table_offset = torch.tensor([0.0, 0.0, self.cfg.table_height], device=self.device)
        lift_offset = torch.tensor([0.0, 0.0, 0.3], device=self.device)

        # 데이터셋의 오브젝트 위치를 현재 환경 위치로 오프셋 처리
        # traj["obj_pos"]는 (3,) 이므로 환경 개수만큼 expand해서 계산
        obj_base_pos = self.traj["obj_pos"].view(1, 3).expand(len(env_ids), 3)
        
        # 최종 타겟: (데이터셋 위치 + 환경 원점 + 테이블 높이) + 0.3m 들어올리기
        self.target_pos_w[env_ids] = obj_base_pos + env_origins + table_offset + lift_offset
        
        self._update_command()

    def _update_command(self):
        """오브젝트 위치 변화를 반영하여 실시간 목표 갱신"""
        obj_pos_w = self.object.data.root_pos_w
        obj_quat_w = self.object.data.root_quat_w

        # T_world_target = T_world_obj * T_obj_hand
        goal_pos_w, goal_quat_w = combine_frame_transforms(
            obj_pos_w, obj_quat_w, self.rel_hand_pos, self.rel_hand_quat
        )
        
        self.pose_command_w[:, :3] = goal_pos_w
        self.pose_command_w[:, 3:] = goal_quat_w

        # Robot Base 기준 좌표 (Policy 학습용)
        goal_pos_b, goal_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w,
            goal_pos_w, goal_quat_w
        )
        self.pose_command_b[:, :3] = goal_pos_b
        self.pose_command_b[:, 3:] = goal_quat_b

    def _update_metrics(self):
        """현재 EE와 목표 지점 간 오차 계산"""
        ee_pos_w = self.robot.data.body_state_w[:, self.ee_body_idx, :3]
        ee_quat_w = self.robot.data.body_state_w[:, self.ee_body_idx, 3:7]

        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3], self.pose_command_w[:, 3:],
            ee_pos_w, ee_quat_w
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    @property
    def command(self) -> torch.Tensor:
        # 7(Pose) + 손가락 개수 + 3(Target Pos)
        return torch.cat([self.pose_command_b, self.target_qpos, self.target_pos_w], dim=-1)

    # --- Debug Visualizer ---
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            if not hasattr(self, "current_pose_visualizer"):
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            if not hasattr(self, "target_pos_visualizer"):
                self.target_pos_visualizer = VisualizationMarkers(self.cfg.target_pos_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
            self.target_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)
                self.target_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized: return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        self.current_pose_visualizer.visualize(
            self.robot.data.body_state_w[:, self.ee_body_idx, :3],
            self.robot.data.body_state_w[:, self.ee_body_idx, 3:7]
        )
        self.target_pos_visualizer.visualize(
            self.target_pos_w, 
            torch.tensor([1, 0, 0, 0], device=self.device).repeat(self.num_envs, 1)
        )

@configclass
class DexYCBCommandTermCfg(CommandTermCfg):
    class_type: type = DexYCBCommandTerm
    file_path: str = MISSING
    frame_idx: int = 0
    table_height: float = 1.0
    asset_name: str = "robot"
    object_name: str = "object"
    body_name: str = "hx5_d20_right_base"
    
    # [Debug Visualizer Settings]
    # 1. 목표 포즈 마커 (목표 지점)
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    # 2. 현재 EE 포즈 마커 (로봇 손 위치)
    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/current_ee_pose"
    )
    
    target_pos_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/target_pos"
    )

    # 마커 크기 조절 (0.1m 정도로 작게 설정)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    target_pos_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)