import torch
import numpy as np
from isaacsim.core.api.controllers import BaseController

class SwerveController(BaseController):
    """3-wheel Swerve Drive Controller for Robotis FFW-SH5"""
    
    def __init__(self, num_envs: int, device: str):
        """
        Initialize the SwerveController.

        Args:
            num_envs (int): The number of environments.
            device (str): The device to run the controller on.
        """
        
        super().__init__(name="swerve_controller")
        self.num_envs = num_envs
        self.device = device
        
        # The radius of the wheels (in meters)
        self._wheel_radius = 0.0825
        
        # The relative positions of the three wheels in the robot frame (x, y)
        self._wheel_positions = torch.tensor([
            [0.1371, 0.2554], 
            [0.1371, -0.2554], 
            [-0.2899, 0.0]
        ], device=self.device, dtype=torch.float32)
        
        # Save previous drive velocities and steer angles for slew rate limiting
        self._prev_drive_vels = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_steer_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Control parameters
        self._max_drive_accel = 50.0
        self._max_steer_vel = 15.0
        self._angle_threshold = np.deg2rad(3.0)

    def _normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Normalize an angle to the range [-pi, pi).

        Args:
            angle (torch.Tensor): The angle to normalize.

        Returns:
            torch.Tensor: The normalized angle.
        """
        
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi

    def forward(self, command: torch.Tensor, current_steer_angles: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the target drive velocities and steer angles for the swerve drive.

        Args:
            command (torch.Tensor): The command tensor of shape (num_envs, 3) containing [vx, vy, w].
            current_steer_angles (torch.Tensor): The current steer angles of shape (num_envs, 3).
            dt (float): The time step.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the target drive velocities and steer angles.
        """
        
        vx = command[:, 0:1]
        vy = command[:, 1:2]
        w = command[:, 2:3] * 1.2 
        
        # Lists to store the computed target drive velocities and steer angles for each wheel
        all_target_drive = []
        all_target_steer = []

        # Compute target angles and speeds for each wheel based on the desired robot velocity and rotation
        for i in range(3):
            pos = self._wheel_positions[i]
            # vix = vx - w * pos_y, viy = vy + w * pos_x
            vix = vx - w * pos[1]
            viy = vy + w * pos[0]
            speed = torch.sqrt(vix**2 + viy**2)
            
            # Initially set desired angle to current angle and speed to 0 (for stationary wheels)
            desired_angle = self._prev_steer_pos[:, i:i+1].clone()
            desired_speed = torch.zeros_like(speed)
            
            # Calculate desired angle and speed only for wheels that are moving above a small threshold
            moving_mask = (speed > 1e-3).squeeze(-1)
            if moving_mask.any():
                desired_angle[moving_mask] = torch.atan2(viy[moving_mask], vix[moving_mask])
                desired_speed[moving_mask] = speed[moving_mask] / self._wheel_radius
                
                # Calculate the angle difference to determine if we should reverse the wheel direction
                diff = self._normalize_angle(desired_angle - self._prev_steer_pos[:, i:i+1])
                rev_mask = (torch.abs(diff) > torch.pi / 2).squeeze(-1)
                
                # Reverse the wheel direction and adjust the desired angle by 180 degrees for wheels that need to reverse
                final_rev_mask = moving_mask & rev_mask
                desired_angle[final_rev_mask] = self._normalize_angle(desired_angle[final_rev_mask] + torch.pi)
                desired_speed[final_rev_mask] *= -1.0
            
            all_target_drive.append(desired_speed)
            all_target_steer.append(desired_angle)

        final_target_drive = torch.cat(all_target_drive, dim=-1) # (num_envs, 3)
        final_target_steer = torch.cat(all_target_steer, dim=-1) # (num_envs, 3)

        # Rate limit the steer angle changes (Steer Slew Rate Limiting)
        steer_diff = self._normalize_angle(final_target_steer - self._prev_steer_pos)
        steer_step = torch.clamp(steer_diff, -self._max_steer_vel * dt, self._max_steer_vel * dt)
        limited_steer_pos = self._normalize_angle(self._prev_steer_pos + steer_step)

        # Calculate the error between the target steer angles and current steer angles to determine the drive speed scaling
        real_diff = torch.abs(self._normalize_angle(final_target_steer - current_steer_angles))
        # The more the wheel is misaligned with the target angle, the more we reduce the drive speed to prevent excessive slipping.
        error_scale = torch.clamp(1.0 - (real_diff / (torch.pi / 2)), 0.0, 1.0)
        
        # Drive speed scaling based on steer angle error, with a minimum scale (x0.15) to ensure some movement even when misaligned
        drive_scale = torch.maximum(error_scale, torch.tensor(0.15, device=self.device))
        actual_target_drive = final_target_drive * drive_scale
        
        # Rate limit the drive velocity changes (Drive Slew Rate Limiting)
        drive_diff = actual_target_drive - self._prev_drive_vels
        drive_step = torch.clamp(drive_diff, -self._max_drive_accel * dt, self._max_drive_accel * dt)
        limited_drive_vels = self._prev_drive_vels + drive_step

        # Update previous drive velocities and steer angles for the next iteration
        self._prev_drive_vels = limited_drive_vels.clone()
        self._prev_steer_pos = limited_steer_pos.clone()
        
        return limited_drive_vels, limited_steer_pos