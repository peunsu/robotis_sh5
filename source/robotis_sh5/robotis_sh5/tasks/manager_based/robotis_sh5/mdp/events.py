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

def reset_random_waypoints(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, num_waypoints: int, distance_range: tuple[float, float, float]):
    """
    Reset the waypoints for the specified environments with random positions.

    Args:
        env (ManagerBasedRLEnv): The environment instance containing the robot state information and scene.
        env_ids (torch.Tensor): The indices of the environments to reset.
        num_waypoints (int): The number of waypoints to generate.
        distance_range (tuple[float, float, float]): The range of distances for the waypoints. (low_dist, high_dist, lateral_dist)
    """
    
    wm = get_or_create_waypoint_manager(env, num_waypoints)
    num_resets = len(env_ids) # Number of environments to reset
    wm.target_indices[env_ids] = 0 # Reset target index to the first waypoint for the specified environments
    
    # Get the reference position for the specified environments (usually the environment origin)
    reference_pos = env.scene.env_origins[env_ids]
    
    
    # Generate random waypoints in front of the robot within the specified distance range
    fps = torch.zeros((num_resets, num_waypoints, 3), device=env.device)
    low_dist, high_dist, lateral_dist = distance_range[0], distance_range[1], distance_range[2]
    
    # To create a path of waypoints that extends forward from the robot,
    # we can generate random distances along the x-axis (forward direction) for each waypoint,
    # while adding some random lateral offset on the y-axis.
    # The z-axis can be set to a fixed height for all waypoints.
    # This will create a more natural and navigable path for the robot to follow.
    cumulative_x = torch.zeros(num_resets, device=env.device)
    for i in range(num_waypoints):
        # Generate a random step distance along the x-axis for each waypoint,
        # ensuring that waypoints are spaced out in front of the robot
        dist_step = torch.empty(num_resets, device=env.device).uniform_(low_dist, high_dist)
        cumulative_x += dist_step
        
        # Random distance along the x-axis (forward direction)
        fps[:, i, 0] = cumulative_x
        
        # Random lateral offset along the y-axis (sideways direction)
        fps[:, i, 1] = torch.empty(num_resets, device=env.device).uniform_(-lateral_dist, lateral_dist)
        
        # Fixed height (z-axis)
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

def linear_curriculum_distance(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, old_value: tuple, max_distance_range: tuple, max_steps: int) -> tuple:
    """
    Update the distance range for the curriculum learning based on the current step.

    Args:
        env (ManagerBasedRLEnv): The environment instance containing the robot state information and scene.
        env_ids (torch.Tensor): The indices of the environments for which to update the distance range.
        old_value (tuple): The initial distance range values (low_dist, high_dist, lateral_dist).
        max_distance_range (tuple): The maximum distance range values (low_dist, high_dist, lateral_dist).
        max_steps (int): The maximum number of steps for the curriculum learning.

    Returns:
        tuple: The updated distance range values (low_dist, high_dist, lateral_dist).
    """
    
    current_step = env.common_step_counter
    
    # Calculate the progress of the curriculum learning as a value between 0 and 1
    progress = min(current_step / max_steps, 1.0)
    
    # Linearly interpolate the distance range values based on the progress of the curriculum learning
    start_low, start_high, start_lateral = 1.0, 2.0, 1.5
    target_low, target_high, target_lateral = max_distance_range
    
    new_low = start_low + (target_low - start_low) * progress
    new_high = start_high + (target_high - start_high) * progress
    new_lateral = start_lateral + (target_lateral - start_lateral) * progress
    
    logger.info(f"Curriculum update - Step: {current_step}, Progress: {progress:.2%}, Distance Range: ({new_low:.2f}, {new_high:.2f}, {new_lateral:.2f})")
    
    return (new_low, new_high, new_lateral)

def adaptive_distance_curriculum(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, old_value: tuple, max_distance_range: tuple) -> tuple:
    """
    Update the distance range for the curriculum learning based on the recent success rate of the agent.

    Args:
        env (ManagerBasedRLEnv): The environment instance containing the robot state information and scene.
        env_ids (torch.Tensor): The indices of the environments for which to update the distance range.
        old_value (tuple): The initial distance range values (low_dist, high_dist, lateral_dist).
        max_distance_range (tuple): The maximum distance range values (low_dist, high_dist, lateral_dist).

    Returns:
        tuple: The updated distance range values (low_dist, high_dist, lateral_dist).
    """
    
    # Get the recent success rate from the environment's metrics
    success_rate = env.extras.get("metrics", {}).get("success_rate", 0.0)
    
    # Initial and target distance range values for the curriculum learning
    start_low, start_high, start_lateral = 1.0, 2.0, 1.5
    target_low, target_high, target_lateral = max_distance_range
    
    # Interpolate the distance range values based on the recent success rate,
    # allowing the curriculum to adapt to the agent's performance
    new_low = start_low + (target_low - start_low) * success_rate
    new_high = start_high + (target_high - start_high) * success_rate
    new_lateral = start_lateral + (target_lateral - start_lateral) * success_rate
    
    return (new_low, new_high, new_lateral)