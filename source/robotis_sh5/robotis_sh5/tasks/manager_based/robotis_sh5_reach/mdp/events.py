# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
import logging

from isaaclab.managers import SceneEntityCfg
from .waypoint_manager import get_or_create_waypoint_manager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
logger = logging.getLogger(__name__)

def reset_random_waypoints(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    num_waypoints: int,
    waypoint_params: tuple[float, float, float]
):
    """
    Reset the waypoints for the specified environments with random positions.

    Args:
        env (ManagerBasedRLEnv): The environment instance containing the robot state information and scene.
        env_ids (torch.Tensor): The indices of the environments to reset.
        num_waypoints (int): The number of waypoints to generate.
        waypoint_params (tuple[float, float, float]): The parameters for waypoint generation. (min_dist, max_dist, angle_limit)
    """
    
    wm = get_or_create_waypoint_manager(env, num_waypoints)
    num_resets = len(env_ids) # Number of environments to reset
    wm.target_indices[env_ids] = 0 # Reset target index to the first waypoint for the specified environments
    
    # Get the reference position for the specified environments (usually the environment origin)
    reference_pos = env.scene.env_origins[env_ids]
    
    # Initialize a tensor to store the generated waypoint positions for all resets and waypoints
    fps = torch.zeros((num_resets, num_waypoints, 3), device=env.device)
    
    # Unpack the distance range into minimum and maximum distance values for waypoint generation
    min_dist, max_dist, angle_limit = waypoint_params[0], waypoint_params[1], waypoint_params[2]
    
    # Initialize a tensor to keep track of the previous position for each environment,
    # which will be used to generate waypoints
    prev_pos = torch.zeros((num_resets, 2), device=env.device) 

    for i in range(num_waypoints):
        # Generate random distances for the waypoints within the specified distance range
        r = torch.empty(num_resets, device=env.device).uniform_(min_dist, max_dist)
        
        # Generate random angles for the waypoints within the specified angle limit
        theta = torch.empty(num_resets, device=env.device).uniform_(-angle_limit, angle_limit)
        
        # Calculate the waypoint positions in front of the robot based on the random distances and angles
        prev_pos[:, 0] += r * torch.cos(theta)
        prev_pos[:, 1] += r * torch.sin(theta)
        
        # Store the generated waypoint positions in the fps tensor, with a fixed height of 0.2
        fps[:, i, 0] = prev_pos[:, 0]
        fps[:, i, 1] = prev_pos[:, 1]
        fps[:, i, 2] = 0.2
        
    # Set the waypoint positions in the WaypointManager by adding the reference position to the generated waypoints
    wm.waypoints[env_ids] = fps + reference_pos.unsqueeze(1)
    
    # The current position of the robot's root
    root_pos = env.scene["robot"].data.root_pos_w[env_ids]
    
    # Update the previous distance to the first waypoint for reward calculation
    current_target = wm.waypoints[env_ids, 0]
    wm.prev_dist[env_ids] = torch.norm(current_target - root_pos, dim=-1)
    
    # Update the visuals of the waypoints in the scene
    wm.update_visuals()
    
def reset_root_at_origin(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Reset the root state of the specified asset (default is "robot") to the origin for the given environment IDs.

    Args:
        env (ManagerBasedRLEnv): The environment instance containing the robot state information and scene.
        env_ids (torch.Tensor): The indices of the environments to reset.
        asset_cfg (SceneEntityCfg, optional): The configuration for the asset to reset. Defaults to SceneEntityCfg("robot").
    """
    
    # Get the asset instance from the scene using the provided asset configuration
    asset = env.scene[asset_cfg.name]
    
    # Get the default root state for the specified environments
    default_root_state = asset.data.default_root_state[env_ids].clone()
    
    # Add the environment origin to the default root position to place the robot at the correct location in the world
    default_root_state[:, :3] += env.scene.env_origins[env_ids]
    
    # Write the root pose and velocity to the simulation for the specified environments
    asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
    
    # Reset and write the joint states to the simulation for the specified environments
    default_joint_pos = asset.data.default_joint_pos[env_ids]
    default_joint_vel = asset.data.default_joint_vel[env_ids]
    asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)
    
def update_waypoint_status(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, threshold: float):
    """
    Update the status of waypoints for the specified environments based on the robot's proximity to the current target waypoint.

    Args:
        env (ManagerBasedRLEnv): The environment instance containing the robot state information and scene.
        env_ids (torch.Tensor): The indices of the environments for which to update waypoint status.
        threshold (float): The distance threshold for determining waypoint proximity.
    """
    
    wm = getattr(env, "waypoint_manager", None)
    if wm is not None:
        wm.update(threshold=threshold)

def adaptive_distance_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    old_value: tuple,
    waypoint_params: tuple[float, float, float],
    grace_period: int = 12000,
    fade_in_steps: int = 24000
) -> tuple:
    """
    Adaptively modify the waypoint generation parameters based on the agent's success rate to create a curriculum learning effect.

    Args:
        env (ManagerBasedRLEnv): The environment instance containing the robot state information and scene.
        env_ids (torch.Tensor): The indices of the environments for which to update the distance range.
        old_value (tuple): The initial curriculum parameter values.
        waypoint_params (tuple[float, float, float]): The parameters for waypoint generation. (min_dist, max_dist, angle_limit)
        grace_period (int): The number of steps to wait before starting to update the curriculum based on success rate.
        fade_in_steps (int): The number of steps over which to fade in the curriculum updates

    Returns:
        tuple: The updated values ((min_dist, max_dist), angle_limit) for the curriculum learning.
    """
    
    # Get the current success rate and current step 
    success_rate = env.extras.get("metrics", {}).get("success_rate", 0.0)
    current_step = env.common_step_counter
    
    # Initial and target distance range values for the curriculum learning
    start_min_dist, start_max_dist, start_angle_limit = 1.0, 2.0, (torch.pi / 2)
    target_min_dist, target_max_dist, target_angle_limit = waypoint_params[0], waypoint_params[1], waypoint_params[2]
    
    # During the grace period, return the initial distance range values without updating based on success rate
    if current_step < grace_period:
        return (start_min_dist, start_max_dist, start_angle_limit)
    
    # Calculate the time scale for fading in the curriculum updates based on the current step and fade-in duration
    time_scale = min(1.0, (current_step - grace_period) / fade_in_steps) if fade_in_steps > 0 else 1.0
    
    # Calculate the final scale for the curriculum updates by combining the time scale and success rate
    final_scale = time_scale * success_rate

    # Interpolate the distance range and angle limit based on the final scale
    new_min_dist = start_min_dist + (target_min_dist - start_min_dist) * final_scale
    new_max_dist = start_max_dist + (target_max_dist - start_max_dist) * final_scale
    new_angle_limit = start_angle_limit + (target_angle_limit - start_angle_limit) * final_scale
    
    return (new_min_dist, new_max_dist, new_angle_limit)