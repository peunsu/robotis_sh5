# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from isaaclab.managers.action_manager import ActionTerm
from . import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class JointPositionLowPassAction(JointPositionAction):
    def __init__(self, cfg: JointPositionLowPassActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        alpha = 1.0 - math.exp(-2.0 * math.pi * cfg.f_c / cfg.f_s)
        self._weights = [alpha, 1.0 - alpha]
        self._prev_model_output = torch.zeros_like(self._raw_actions)
  
    def process_actions(self, actions: torch.Tensor):
        filtered = self._weights[0] * actions + self._weights[1] * self._prev_model_output
        self._prev_model_output[:] = actions.clone()
        super().process_actions(filtered)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._prev_model_output[env_ids] = 0.0
        super().reset(env_ids)
        
@configclass
class JointPositionLowPassActionCfg(JointPositionActionCfg):
    """Low-pass filtered joint position action 설정을 위한 클래스."""

    # 이 설정이 연결될 실제 액션 클래스 타입을 지정해
    # (앞서 정의한 JointPositionLowPassAction 클래스 이름을 넣으면 돼)
    class_type: type[ActionTerm] = JointPositionLowPassAction

    # --- LPF 전용 파라미터 추가 ---
    
    f_c: float = 5.0
    """Cut-off frequency (Hz). 값이 낮을수록 더 부드럽지만 반응은 느려져."""
    
    f_s: float = 60.0
    """Sampling frequency (Hz). 시뮬레이션의 제어 주기(Control Loop)와 맞춰야 해."""