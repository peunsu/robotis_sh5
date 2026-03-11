# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# 관측 (Observations) 구현
##

def get_rel_pos_to_goal(env: ManagerBasedRLEnv):
    """로봇과 목표 지점 사이의 상대 위치를 계산."""
    # 로봇은 Articulation이므로 .data를 사용 가능
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, :2]
    
    # goal_marker가 XformPrimView일 경우 .get_world_poses() 사용
    goal_pos_w, _ = env.scene["goal_marker"].get_world_poses()
    goal_pos_w = goal_pos_w[:, :2]
    
    return goal_pos_w - robot_pos_w