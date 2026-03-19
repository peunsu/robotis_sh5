# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from .swerve_controller import SwerveController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class SwerveDriveAction(ActionTerm):
    """
    Action term for controlling a swerve drive robot.
    This term processes high-level velocity commands from the agent and translates them into joint position and velocity targets for the robot's steering and drive joints.
    """
    
    def __init__(self, cfg: SwerveDriveActionCfg, env: ManagerBasedRLEnv):
        """
        Initialize the SwerveDriveAction.

        Args:
            cfg (SwerveDriveActionCfg): The configuration for the swerve drive action term, including joint names and scaling factors.
            env (ManagerBasedRLEnv): The environment instance containing the robot state information and scene.
        """
        
        super().__init__(cfg, env)
        
        # Extract joint IDs for the steering and drive joints based on the provided joint name patterns in the configuration
        self.steer_joint_ids, _ = self._asset.find_joints(cfg.joint_names[0]) # ".*_steer"
        self.drive_joint_ids, _ = self._asset.find_joints(cfg.joint_names[1]) # ".*_drive"
        
        # Initialize the swerve controller which will handle the kinematics and control logic for the swerve drive
        self.controller = SwerveController(num_envs=self.num_envs, device=self.device)
        
        # Initialize tensors for raw and processed actions, and store the time step for control updates
        self.dt = env.physics_dt * env.cfg.decimation
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        
        # Store the scaling factors for the actions as a tensor for efficient GPU computation
        self._scale = torch.tensor(cfg.scale, device=self.device)

    @property
    def action_dim(self) -> int:
        """
        Get the dimension of the action space.

        Returns:
            int: The dimension of the action space.
        """
        return 3 # [vx, vy, w]

    def raw_actions(self) -> torch.Tensor:
        """
        Get the raw actions.

        Returns:
            torch.Tensor: The raw actions tensor.
        """
        return self._raw_actions

    def processed_actions(self) -> torch.Tensor:
        """
        Get the processed actions after scaling.

        Returns:
            torch.Tensor: The processed actions tensor.
        """
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """
        Process the raw actions from the agent by applying scaling and preparing them for the swerve controller.

        Args:
            actions (torch.Tensor): The raw actions output by the agent, expected to be in the form [vx, vy, w] for each environment.
        """
        
        self._raw_actions[:] = actions
        
        # Scale the actions according to the configuration.
        self._processed_actions[:] = actions * self._scale

    def apply_actions(self):
        """Apply the processed actions to the robot."""
        
        # Current steering joint positions
        curr_steer = self._asset.data.joint_pos[:, self.steer_joint_ids]
        
        # Use the swerve controller to compute the desired drive velocities and steering positions
        # based on the processed actions and current steering state
        drive_vels, steer_pos = self.controller.forward(
            self._processed_actions, 
            curr_steer, 
            self.dt
        )

        # Send the computed drive velocities and steering positions to the robot's joints
        self._asset.set_joint_velocity_target(drive_vels, joint_ids=self.drive_joint_ids)
        self._asset.set_joint_position_target(steer_pos, joint_ids=self.steer_joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        """
        Reset the action term for the specified environment IDs.
        This typically involves resetting any internal state of the controller and clearing the action buffers.

        Args:
            env_ids (torch.Tensor | None, optional): The environment IDs to reset. If None, all environments will be reset. Defaults to None.
        """
        
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset the previous drive velocities and steering positions in the controller for the specified environment IDs
        self.controller._prev_drive_vels[env_ids] = 0.0
        self.controller._prev_steer_pos[env_ids] = 0.0
        
        # Reset the raw and processed action buffers to zero for the specified environment IDs
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

@configclass
class SwerveDriveActionCfg(ActionTermCfg):
    """Configuration for the SwerveDriveAction."""
        
    class_type: type = SwerveDriveAction
    asset_name: str = "robot"
    joint_names: list[str] = MISSING  
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)