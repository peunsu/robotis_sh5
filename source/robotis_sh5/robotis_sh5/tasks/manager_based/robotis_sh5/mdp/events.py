# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from .waypoint_manager import get_or_create_waypoint_manager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_random_waypoints(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, num_waypoints: int, distance_range: tuple[float, float]):
    """
    Reset the waypoints for the specified environments with random positions.

    Args:
        env (ManagerBasedRLEnv): The environment instance containing the robot state information and scene.
        env_ids (torch.Tensor): The indices of the environments to reset.
        num_waypoints (int): The number of waypoints to generate.
        distance_range (tuple[float, float]): The range of distances for the waypoints.
    """
    
    wm = get_or_create_waypoint_manager(env, num_waypoints)
    num_resets = len(env_ids) # Number of environments to reset
    wm.target_indices[env_ids] = 0 # Reset target index to the first waypoint for the specified environments
    
    # Generate random waypoints in front of the robot within the specified distance range
    fps = torch.zeros((num_resets, num_waypoints, 3), device=env.device)
    for i in range(num_waypoints):
        # Random distance along the x-axis (forward direction)
        low, high = distance_range
        fps[:, i, 0] = (i + 1) * torch.empty(num_resets, device=env.device).uniform_(low, high)
        
        # Random lateral offset (y-axis)
        fps[:, i, 1] = torch.randn(num_resets, device=env.device) * 1.5
        
        # Fixed height (z-axis)
        fps[:, i, 2] = 0.2
        
    # Transform the waypoints from the robot's local frame to the world frame by adding the robot's current position
    wm.waypoints[env_ids] = fps + env.scene.env_origins[env_ids].unsqueeze(1)
    
    # Update the previous distance to the first waypoint for reward calculation
    root_pos = env.scene["robot"].data.root_pos_w[env_ids]
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