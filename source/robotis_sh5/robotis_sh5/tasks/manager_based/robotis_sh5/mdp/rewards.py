# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def position_progress_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    wm = getattr(env, "waypoint_manager", None)
    if wm is None: return torch.zeros(env.num_envs, device=env.device)
    
    root_pos = env.scene["robot"].data.root_pos_w[:, :2]
    current_target = wm.waypoints[torch.arange(env.num_envs), wm.target_indices, :2]
    current_dist = torch.norm(current_target - root_pos, dim=-1)
    
    # 얼마나 가까워졌는가?
    progress = wm.prev_dist - current_dist
    wm.prev_dist[:] = current_dist # 다음 프레임을 위해 업데이트
    
    return progress

def heading_alignment_reward(env: "ManagerBasedRLEnv", sigma: float = 0.25) -> torch.Tensor:
    wm = getattr(env, "waypoint_manager", None)
    if wm is None: return torch.zeros(env.num_envs, device=env.device)
    
    # 방향 벡터 계산
    root_pos = env.scene["robot"].data.root_pos_w[:, :2]
    current_target = wm.waypoints[torch.arange(env.num_envs), wm.target_indices, :2]
    target_vec = torch.nn.functional.normalize(current_target - root_pos, dim=-1)
    
    # 로봇 정면 벡터
    root_quat = env.scene["robot"].data.root_quat_w
    
    forward_vec = math_utils.quat_apply(root_quat, torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1))
    forward_dir = torch.nn.functional.normalize(forward_vec[:, :2], dim=-1)
    
    dot_prod = (forward_dir * target_vec).sum(dim=-1)
    heading_error = torch.acos(torch.clamp(dot_prod, -1.0, 1.0))
    
    return torch.exp(-heading_error / sigma)

def goal_reached_reward(env: "ManagerBasedRLEnv", threshold: float) -> torch.Tensor:
    wm = getattr(env, "waypoint_manager", None)
    if wm is None: return torch.zeros(env.num_envs, device=env.device)
    
    root_pos = env.scene["robot"].data.root_pos_w[:, :2]
    current_target = wm.waypoints[torch.arange(env.num_envs), wm.target_indices, :2]
    current_dist = torch.norm(current_target - root_pos, dim=-1)
    
    goal_reached = current_dist < threshold
        
    return goal_reached.float()