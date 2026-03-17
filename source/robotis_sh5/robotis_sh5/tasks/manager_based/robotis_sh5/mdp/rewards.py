# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from collections import deque
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def position_progress_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Calculate a reward based on the progress towards the current target position.
    The reward is positive if the agent has moved closer to the target since the last step, and negative if it has moved further away.
    The waypoint manager is expected to maintain the previous distance to the target for this calculation.

    Args:
        env (ManagerBasedRLEnv): The environment instance, which should have a waypoint manager with the current target and previous distance.

    Returns:
        torch.Tensor: A tensor of shape (num_envs,) containing the progress reward for each environment instance.
    """
        
    wm = getattr(env, "waypoint_manager", None)
    if wm is None: return torch.zeros(env.num_envs, device=env.device)
    
    # Calculate the current distance to the target
    root_pos = env.scene["robot"].data.root_pos_w[:, :2]
    current_target = wm.waypoints[torch.arange(env.num_envs), wm.target_indices, :2]
    current_dist = torch.norm(current_target - root_pos, dim=-1)
    
    progress = wm.prev_dist - current_dist # Positive if closer, negative if further
    wm.prev_dist[:] = current_dist # Update previous distance for the next step
    
    return progress

def heading_alignment_reward(env: "ManagerBasedRLEnv", sigma: float = 0.25) -> torch.Tensor:
    """
    Calculate a reward based on the alignment of the robot's heading with the direction to the current target waypoint.
    The reward is higher when the robot is facing towards the target and decreases as the heading error

    Args:
        env (ManagerBasedRLEnv): The environment instance, which should have a waypoint manager with the current target and previous distance.
        sigma (float, optional): The standard deviation for the exponential decay of the heading error. Defaults to 0.25.

    Returns:
        torch.Tensor: A tensor of shape (num_envs,) containing the heading alignment reward for each environment instance.
    """
    
    wm = getattr(env, "waypoint_manager", None)
    if wm is None: return torch.zeros(env.num_envs, device=env.device)
    
    # Calculate the vector from the robot to the current target
    root_pos = env.scene["robot"].data.root_pos_w[:, :2]
    current_target = wm.waypoints[torch.arange(env.num_envs), wm.target_indices, :2]
    target_vec = torch.nn.functional.normalize(current_target - root_pos, dim=-1)
    
    # Calculate the robot's forward direction in the world frame
    root_quat = env.scene["robot"].data.root_quat_w
    forward_vec = math_utils.quat_apply(root_quat, torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1))
    forward_dir = torch.nn.functional.normalize(forward_vec[:, :2], dim=-1)
    
    # Calculate the heading error as the angle between the forward direction and the target vector
    dot_prod = (forward_dir * target_vec).sum(dim=-1)
    heading_error = torch.acos(torch.clamp(dot_prod, -1.0, 1.0))
    
    # Reward is higher when heading error is smaller, using an exponential decay based on the error
    return torch.exp(-heading_error / sigma)

def goal_reached_reward(env: "ManagerBasedRLEnv", threshold: float) -> torch.Tensor:
    """
    Calculate a reward for reaching the goal (current waypoint).
    The reward is given when the robot is within a certain threshold distance from the target waypoint.
    Also updates the success rate metric in the environment's extras when an episode ends.

    Args:
        env (ManagerBasedRLEnv): The environment instance, which should have a waypoint manager with the current target and previous distance.
        threshold (float): The distance threshold within which the robot is considered to have reached the goal.

    Returns:
        torch.Tensor: A tensor of shape (num_envs,) containing the goal reached reward for each environment instance (1.0 if reached, 0.0 otherwise).
    """
    
    wm = getattr(env, "waypoint_manager", None)
    if wm is None: return torch.zeros(env.num_envs, device=env.device)
    
    # Ensure the metrics dictionary exists in env.extras to store success rate
    if "metrics" not in env.extras:
        env.extras["metrics"] = {"success_rate": 0.0}
    if "success_buffer" not in env.extras:
        env.extras["success_buffer"] = deque(maxlen=1000) # Buffer to store recent success outcomes for calculating success rate
    
    # Calculate the current distance to the target
    root_pos = env.scene["robot"].data.root_pos_w[:, :2]
    current_target = wm.waypoints[torch.arange(env.num_envs), wm.target_indices, :2]
    current_dist = torch.norm(current_target - root_pos, dim=-1)
    
    # Reward is 1.0 if the current distance is less than the threshold, otherwise 0.0
    goal_reached = current_dist < threshold
    goal_reached_float = goal_reached.float()
    
    # The ids of the environments that have reached the final goal (last waypoint) at this step
    reset_ids = env.reset_buf.nonzero(as_tuple=False).flatten()
    
    if len(reset_ids) > 0:        
        # Consider it a success if the robot has reached the goal and it is the final waypoint
        is_last = (wm.target_indices == wm.num_waypoints - 1)
        success_all_envs = (is_last & goal_reached)
        
        # Get the success outcomes for the environments that are resetting at this step
        success_at_reset = success_all_envs[reset_ids]
        
        # Update the success buffer     
        env.extras["success_buffer"].extend(success_at_reset.tolist())
        
        # Update the success rate metric in env.extras
        if len(env.extras["success_buffer"]) > 0:
            env.extras["metrics"]["success_rate"] = (
                sum(env.extras["success_buffer"]) / len(env.extras["success_buffer"])
            )
    
    return goal_reached_float