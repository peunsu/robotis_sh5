# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from .swerve_controller import SwerveController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class SwerveDriveAction(ActionTerm):
    def __init__(self, cfg: SwerveDriveActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # 1. 조인트 인덱스 추출
        self.steer_joint_ids, _ = self._asset.find_joints(cfg.joint_names[0]) # ".*_steer"
        self.drive_joint_ids, _ = self._asset.find_joints(cfg.joint_names[1]) # ".*_drive"
        
        # 2. Vectorized 컨트롤러 생성
        # 이제 단 하나의 컨트롤러 인스턴스가 모든 환경을 한 번에 처리해.
        self.controller = SwerveController(num_envs=self.num_envs, device=self.device)
        
        # 3. 설정 및 버퍼
        self.dt = env.physics_dt * env.cfg.decimation
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        
        # scale 텐서화
        self._scale = torch.tensor(cfg.scale, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3 # [vx, vy, w]

    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # 에이전트 출력에 scale 적용 (모두 텐서 연산)
        self._processed_actions[:] = actions * self._scale

    def apply_actions(self):
        # 현재 조향 각도 (이미 텐서)
        curr_steer = self._asset.data.joint_pos[:, self.steer_joint_ids]
        
        # 컨트롤러의 vectorized forward 호출
        # 더 이상 루프나 numpy 변환 없이 GPU에서 한 번에 계산됨!
        drive_vels, steer_pos = self.controller.forward(
            self._processed_actions, 
            curr_steer, 
            self.dt
        )

        # 실제 시뮬레이션에 명령 전달
        self._asset.set_joint_velocity_target(drive_vels, joint_ids=self.drive_joint_ids)
        self._asset.set_joint_position_target(steer_pos, joint_ids=self.steer_joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 컨트롤러 내부의 이전 상태 값들도 해당 환경 ID만 골라서 리셋
        self.controller._prev_drive_vels[env_ids] = 0.0
        self.controller._prev_steer_pos[env_ids] = 0.0
        
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

@configclass
class SwerveDriveActionCfg(ActionTermCfg):
    """Swerve Drive 액션 설정"""
    class_type: type = SwerveDriveAction
    asset_name: str = "robot"
    # 조인트 이름은 리스트 형태로 전달 [steer_pattern, drive_pattern]
    joint_names: list[str] = MISSING  
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)