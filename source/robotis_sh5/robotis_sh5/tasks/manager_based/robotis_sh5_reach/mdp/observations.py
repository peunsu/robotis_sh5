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
    
def get_rel_pos_to_current_waypoint(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Calculate the relative position from the robot to the current target waypoint in the robot's local frame.

    Args:
        env (ManagerBasedRLEnv): The environment instance, which should have a waypoint manager with the current target and the robot's state information.

    Returns:
        torch.Tensor: The relative position from the robot to the current target waypoint in the robot's local frame.
    """
    
    wm = getattr(env, "waypoint_manager", None)
    if wm is None:
        return torch.zeros((env.num_envs, 3), device=env.device)

    # Calculate the relative position in the world frame
    root_pos_w = env.scene["robot"].data.root_pos_w
    current_target_w = wm.waypoints[torch.arange(env.num_envs), wm.target_indices]
    rel_pos_w = current_target_w - root_pos_w
    
    # Transform the relative position to the robot's local frame using the inverse of the robot's orientation
    root_quat_w = env.scene["robot"].data.root_quat_w
    rel_pos_b = math_utils.quat_apply_inverse(root_quat_w, rel_pos_w)
    
    return rel_pos_b

def get_target_waypoint_index(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Calculate the normalized index of the current target waypoint.

    Args:
        env (ManagerBasedRLEnv): The environment instance, which should have a waypoint manager with the current target index and total number of waypoints.

    Returns:
        torch.Tensor: The normalized index of the current target waypoint, scaled to the range [0, 1].
    """
    
    wm = getattr(env, "waypoint_manager", None)
    if wm is None:
        return torch.zeros((env.num_envs, 1), device=env.device)

    # Normalize the target index by the total number of waypoints to get a value in the range [0, 1]
    normalized_idx = wm.target_indices.unsqueeze(1).float() / (wm.num_waypoints - 1)
    
    return normalized_idx

def get_waypoint_heading_error_sin_cos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Calculate the sine and cosine of the heading error between the robot's current orientation and the direction to the current target waypoint.

    Args:
        env (ManagerBasedRLEnv): The environment instance, which should have a waypoint manager with the current target and the robot's state information.

    Returns:
        torch.Tensor: The sine and cosine of the heading error, shape (num_envs, 2).
    """
    
    wm = getattr(env, "waypoint_manager", None)
    if wm is None: return torch.zeros((env.num_envs, 2), device=env.device)
    
    root_pos_w = env.scene["robot"].data.root_pos_w[:, :2]
    current_target_w = wm.waypoints[torch.arange(env.num_envs), wm.target_indices, :2]
    
    # Calculate the robot's yaw angle from its orientation quaternion
    _, _, robot_yaw = math_utils.euler_xyz_from_quat(env.scene["robot"].data.root_quat_w)
    
    # Calculate the target heading angle in the world frame using atan2
    target_heading_w = torch.atan2(
        current_target_w[:, 1] - root_pos_w[:, 1],
        current_target_w[:, 0] - root_pos_w[:, 0]
    )
    
    # Calculate the heading error as the difference between the target heading and the robot's yaw, wrapped to the range [-pi, pi]
    error = torch.atan2(torch.sin(target_heading_w - robot_yaw), torch.cos(target_heading_w - robot_yaw))
    
    return torch.stack([torch.cos(error), torch.sin(error)], dim=-1)