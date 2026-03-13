# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class WaypointManager:
    """Manages waypoints for the robot in the environment."""
    
    def __init__(self, env: "ManagerBasedRLEnv", num_waypoints: int):
        """
        Initialize the WaypointManager.

        Args:
            env (ManagerBasedRLEnv): The environment instance containing the robot state information and scene.
            num_waypoints (int): The total number of waypoints to manage for each environment.
        """
        
        self.env = env
        self.num_envs = env.num_envs
        self.num_waypoints = num_waypoints
        self.device = env.device
        
        # Store the current target waypoint index, the waypoint positions, and the previous distance to the target for each environment
        self.target_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.waypoints = torch.zeros((self.num_envs, self.num_waypoints, 3), device=self.device)
        self.prev_dist = torch.zeros(self.num_envs, device=self.device)
        
        self._marker_visualizer = None

    def update_visuals(self):
        """Update the visual markers for the waypoints in the scene."""
        
        # If we haven't cached the marker visualizer yet, try to get it from the scene
        if self._marker_visualizer is None:
            self._marker_visualizer = getattr(self.env.scene, "waypoint_markers", None)

        # If we have a marker visualizer, update the marker positions and colors based on the waypoints and target indices
        if self._marker_visualizer is not None:
            # All waypoint positions (flattened across environments)
            all_positions = self.waypoints.view(-1, 3)
            
            # Create a marker index tensor where all waypoints are initially set to 1 (future, green)
            marker_indices = torch.ones((self.num_envs, self.num_waypoints), 
                                    dtype=torch.int32, device=self.device)
            
            # Set the current target waypoint for each environment to 0 (current, red)
            marker_indices[torch.arange(self.num_envs), self.target_indices] = 0
            
            # Flatten the marker indices to match the shape of all_positions
            marker_indices = marker_indices.view(-1)

            # Update the visualizer with the new positions and marker indices
            if all_positions.shape[0] > 0:
                self._marker_visualizer.visualize(
                    translations=all_positions, 
                    marker_indices=marker_indices
                )
    
    def update(self, threshold: float):
        """
        Update the waypoint manager by checking if the robot has reached the current target waypoint and updating the target indices accordingly.

        Args:
            threshold (float): The distance threshold for determining if the robot has reached a waypoint.
        """
        
        # Calculate the distance from the robot's current position to the current target waypoint for each environment (x, y only)
        root_pos = self.env.scene["robot"].data.root_pos_w[:, :2]
        current_target = self.waypoints[torch.arange(self.num_envs), self.target_indices, :2]
        current_dist = torch.norm(current_target - root_pos, dim=-1)
        
        # Check if the robot has reached the current target waypoint (within the threshold distance)
        goal_reached = current_dist < threshold
        
        # If the goal is reached, increment the target index for that environment (but do not exceed num_waypoints - 1)
        if torch.any(goal_reached):
            self.target_indices[:] = torch.clamp(
                self.target_indices + goal_reached.long(),
                max=self.num_waypoints - 1
            )
            self.update_visuals()
            
def get_or_create_waypoint_manager(env: "ManagerBasedRLEnv", num_waypoints: int = 10) -> "WaypointManager":
    """
    Get the existing WaypointManager from the environment or create a new one if it doesn't exist.

    Args:
        env (ManagerBasedRLEnv): The environment instance containing the robot state information and scene.
        num_waypoints (int, optional): The number of waypoints to initialize. Defaults to 10.

    Returns:
        WaypointManager: The waypoint manager instance.
    """
    
    if not hasattr(env, "waypoint_manager"):
        # Initialize the waypoint manager and attach it to the environment for later use
        env.waypoint_manager = WaypointManager(env, num_waypoints)
        
        # If the environment has a configuration for waypoint markers,
        # create a VisualizationMarkers instance and attach it to the scene for visualizing the waypoints
        if hasattr(env.cfg, "waypoint_marker_cfg"):
            env.scene.waypoint_markers = VisualizationMarkers(env.cfg.waypoint_marker_cfg)
            
    return env.waypoint_manager