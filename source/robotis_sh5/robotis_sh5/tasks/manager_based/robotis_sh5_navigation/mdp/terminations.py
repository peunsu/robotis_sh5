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

def bad_orientation(env: "ManagerBasedRLEnv", threshold: float) -> torch.Tensor:
    """
    Return a boolean tensor indicating whether each environment has a bad orientation (e.g., flipped over), which can be used for episode termination.

    Args:
        env (ManagerBasedRLEnv): The environment instance containing the robot state information.
        threshold (float): The threshold for determining bad orientation.

    Returns:
        torch.Tensor: A boolean tensor of shape (num_envs,) where True indicates that the corresponding environment has a bad orientation.
    """
    
    # Get the robot's root orientation quaternion (w component) to determine how much it's tilted.
    root_quat_w = env.scene["robot"].data.root_quat_w
    
    # Calculate the up vector in world coordinates by applying the root quaternion to the local up vector (0, 0, 1).
    up_vec_w = math_utils.quat_apply(
        root_quat_w, 
        torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    )
    
    # Calculate the dot product between the up vector and the world Z-axis (0, 0, 1) to check the tilt.
    # 1.0 means upright, 0.0 means lying on its side, -1.0 means flipped over.
    z_dot = up_vec_w[:, 2]
    
    # Return True if the robot is tilted beyond the threshold
    return z_dot < threshold

def all_waypoints_reached(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Return a boolean tensor indicating whether each environment has reached all waypoints, which can be used for episode termination.

    Args:
        env (ManagerBasedRLEnv): The environment instance containing the waypoint manager and robot state information.

    Returns:
        torch.Tensor: A boolean tensor of shape (num_envs,) where True indicates that the corresponding environment has reached all waypoints.
    """
    
    wm = getattr(env, "waypoint_manager", None)
    if wm is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Check if the current target index is the last waypoint for each environment
    is_last_waypoint = (wm.target_indices == wm.num_waypoints - 1)
    
    # Calculate the distance from the robot's current position to the final target waypoint
    root_pos_w = env.scene["robot"].data.root_pos_w[:, :2]
    final_target_w = wm.waypoints[torch.arange(env.num_envs), -1, :2]
    dist_to_final = torch.norm(final_target_w - root_pos_w, dim=-1)
    
    # Consider the task completed if the robot is at the last waypoint and within a certain distance threshold to it
    success = is_last_waypoint & (dist_to_final < 0.2)
    
    return success