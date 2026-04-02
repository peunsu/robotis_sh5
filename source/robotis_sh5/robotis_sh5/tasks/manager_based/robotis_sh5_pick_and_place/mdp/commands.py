# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import numpy as np
import os
import glob
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms, compute_pose_error
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from ..retargeting.dataset import DexYCBVideoDataset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class DexYCBCommandTerm(CommandTerm):
    """DexYCB 데이터셋 기반의 포즈 및 관절각 커맨드 생성기"""

    def __init__(self, cfg: DexYCBCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.dataset = DexYCBVideoDataset("/home/peunsu/workspace/robotis_sh5/retargeting/DexYCB", hand_type="right")
        self.dataset_len = len(self.dataset)
        
        # 로봇 및 추적할 바디 설정
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        
        # 1. 샘플 데이터를 하나 불러와서 데이터셋의 joint_names 확인
        sample_path = os.path.join(self.cfg.dataset_dir, f"{self.dataset[0]['capture_name']}.npy")
        sample_data = np.load(sample_path, allow_pickle=True).item()
        dataset_joint_names = sample_data["joint_names"]

        # 2. 로봇의 관절 순서에 맞춰 데이터셋 인덱스 맵 생성
        # robot.joint_names는 Isaac Lab이 URDF/USD로부터 읽어들인 순서임
        self.retargeting_to_isaac = [
            dataset_joint_names.index(name) for name in self.robot.joint_names 
            if name in dataset_joint_names
        ]
        
        # [Buffers]
        # pose_command_b: 로봇 베이스 기준 (pos_x, y, z, qw, qx, qy, qz) -> 7차원
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0 # 기본 쿼터니언 (w=1)
        
        # world 좌표계 저장용 (Metrics 및 시각화용)
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        
        # qpos_command: 손가락 관절 목표값 (20차원)
        self.target_qpos = torch.zeros(self.num_envs, 20, device=self.device)

        # [Metrics] 예시 코드처럼 에러 지표 추가
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """데이터셋에서 샘플링한 월드 좌표를 로봇 베이스 기준 상대 좌표로 변환하여 저장"""
        
        env_origins = self._env.scene.env_origins[env_ids]

        for i, env_id in enumerate(env_ids):
            # 1. 데이터셋에서 무작위 샘플링
            random_idx = np.random.randint(self.dataset_len)
            
            data_path = os.path.join(self.cfg.dataset_dir, f"{self.dataset[random_idx]['capture_name']}.npy")
            data = np.load(data_path, allow_pickle=True).item()
            
            random_frame = np.random.randint(len(data["qpos"]))
            
            # 데이터셋의 좌표 (해당 env 원점 기준)
            raw_pos = torch.tensor(data["root_pos"][random_frame], device=self.device).float()
            raw_pos[2] += 1.0 # 테이블 높이 보정
            raw_quat = torch.tensor(data["root_quat"][random_frame], device=self.device).float()

            # 1. 월드 좌표(w)로 먼저 변환 (env_origin + 데이터셋 좌표)
            # 회전은 env 자체가 회전되어 있지 않다면 raw_quat을 그대로 써도 되지만, 
            # 위치는 반드시 origin을 더해줘야 실제 시뮬레이션상의 월드 좌표가 됨.
            w_pos = env_origins[i] + raw_pos
            w_quat = raw_quat 

            # 2. 로봇 베이스 기준 상대 좌표(_b) 계산
            # 이제 로봇의 실제 월드 포즈와 위에서 구한 월드 목표 지점을 비교
            target_pos_b, target_quat_b = subtract_frame_transforms(
                self.robot.data.root_pos_w[env_id:env_id+1], 
                self.robot.data.root_quat_w[env_id:env_id+1], 
                w_pos.unsqueeze(0), 
                w_quat.unsqueeze(0)
            )

            # 버퍼 저장
            self.pose_command_w[env_id, :3] = w_pos
            self.pose_command_w[env_id, 3:] = w_quat
            self.pose_command_b[env_id, :3] = target_pos_b.squeeze(0)
            self.pose_command_b[env_id, 3:] = target_quat_b.squeeze(0)
            
            # 5. 손가락 관절 목표값 저장 (이건 절대값이므로 그대로 유지)
            #self.target_qpos[env_id] = torch.tensor(data["qpos"][-1], device=self.device).float()
            
            qpos_all = data["qpos"][random_frame]
            mapped_qpos = torch.tensor(qpos_all[self.retargeting_to_isaac], device=self.device).float()
            
            # 현재 로봇의 관절 순서에 맞게 버퍼에 저장
            self.target_qpos[env_id] = mapped_qpos

    def _update_command(self):
        """매 스텝마다 Base 기준 커맨드 유지 (필요 시 여기서 동적 갱신 가능)"""
        pass

    def _update_metrics(self):
        """현재 로봇 상태와 목표 포즈 간의 에러 계산"""
        # Base 커맨드를 다시 World로 변환 (로봇이 움직이니까 매번 갱신)
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        
        # 실제 End-effector(body_idx) 포즈와 비교
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    @property
    def command(self) -> torch.Tensor:
        """최종적으로 Policy에 전달될 관찰값 (Pose 7 + Qpos 20 = 27차원)"""
        return torch.cat([self.pose_command_b, self.target_qpos], dim=-1)
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """디버그 시각화 마커 생성 및 가시성 설정"""
        if debug_vis:
            # 마커가 아직 생성되지 않았다면 새로 생성
            if not hasattr(self, "goal_pose_visualizer"):
                # 1. 목표 포즈 시각화 (예: 반투명한 축 또는 구체)
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)

            # 가시성 활성화
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            # 가시성 비활성화
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """매 프레임마다 마커 위치를 업데이트하는 콜백"""
        # 로봇이 아직 초기화되지 않았다면 중단
        if not self.robot.is_initialized:
            return

        # 1. 목표 포즈 업데이트 (World 좌표계 기준)
        # pose_command_w는 _update_metrics에서 이미 갱신됨
        self.goal_pose_visualizer.visualize(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:]
        )

        # 2. 현재 로봇의 End-effector 포즈 업데이트
        ee_state_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(
            ee_state_w[:, :3], 
            ee_state_w[:, 3:7]
        )

@configclass
class DexYCBCommandTermCfg(CommandTermCfg):
    class_type: type = DexYCBCommandTerm
    dataset_dir: str = MISSING
    asset_name: str = "robot"
    body_name: str = MISSING  # 오차 계산을 위한 End-effector 바디 이름
    make_quat_unique: bool = True
    
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)