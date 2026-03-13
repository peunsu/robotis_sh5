import torch
import numpy as np
from isaacsim.core.api.controllers import BaseController

class SwerveController(BaseController):
    def __init__(self, num_envs: int, device: str):
        super().__init__(name="swerve_controller")
        self.num_envs = num_envs
        self.device = device
        
        self._wheel_radius = 0.0825
        # 바퀴 위치를 (3, 2) 텐서로 정의
        self._wheel_positions = torch.tensor([
            [0.1371, 0.2554], 
            [0.1371, -0.2554], 
            [-0.2899, 0.0]
        ], device=self.device, dtype=torch.float32)
        
        # 이전 상태 저장을 위한 버퍼 (num_envs, 3)
        self._prev_drive_vels = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_steer_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        self._max_drive_accel = 50.0
        self._max_steer_vel = 15.0
        self._angle_threshold = np.deg2rad(3.0)

    def _normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi

    def forward(self, command: torch.Tensor, current_steer_angles: torch.Tensor, dt: float):
        """
        command: (num_envs, 3) -> [vx, vy, w]
        current_steer_angles: (num_envs, 3)
        """
        vx = command[:, 0:1]
        vy = command[:, 1:2]
        w = command[:, 2:3] * 1.2 
        
        # 결과 저장을 위한 리스트 (나중에 cat으로 합침)
        all_target_drive = []
        all_target_steer = []

        # 1. 목표 각도 및 속도 계산 (IK) - 3개의 바퀴에 대해 반복
        for i in range(3):
            pos = self._wheel_positions[i]
            # vix = vx - w * pos_y, viy = vy + w * pos_x
            vix = vx - w * pos[1]
            viy = vy + w * pos[0]
            speed = torch.sqrt(vix**2 + viy**2)
            
            # 기본값 (속도가 낮을 때 이전 각도 유지)
            desired_angle = self._prev_steer_pos[:, i:i+1].clone()
            desired_speed = torch.zeros_like(speed)
            
            # 움직임이 있는 환경에 대해서만 계산 (Masking)
            moving_mask = (speed > 1e-3).squeeze(-1)
            if moving_mask.any():
                desired_angle[moving_mask] = torch.atan2(viy[moving_mask], vix[moving_mask])
                desired_speed[moving_mask] = speed[moving_mask] / self._wheel_radius
                
                # Wheel Reversal 적용
                diff = self._normalize_angle(desired_angle - self._prev_steer_pos[:, i:i+1])
                rev_mask = (torch.abs(diff) > torch.pi / 2).squeeze(-1)
                
                # 실제 반전이 필요한 인덱스 (움직임이 있고 + 90도 이상 차이남)
                final_rev_mask = moving_mask & rev_mask
                desired_angle[final_rev_mask] = self._normalize_angle(desired_angle[final_rev_mask] + torch.pi)
                desired_speed[final_rev_mask] *= -1.0
            
            all_target_drive.append(desired_speed)
            all_target_steer.append(desired_angle)

        final_target_drive = torch.cat(all_target_drive, dim=-1) # (num_envs, 3)
        final_target_steer = torch.cat(all_target_steer, dim=-1) # (num_envs, 3)

        # 2. Steer 제어 (Slew Rate 적용)
        steer_diff = self._normalize_angle(final_target_steer - self._prev_steer_pos)
        steer_step = torch.clamp(steer_diff, -self._max_steer_vel * dt, self._max_steer_vel * dt)
        limited_steer_pos = self._normalize_angle(self._prev_steer_pos + steer_step)

        # 3. Drive 제어 (정렬 오차에 따른 감속 적용)
        real_diff = torch.abs(self._normalize_angle(final_target_steer - current_steer_angles))
        # 90도 이상이면 0.0, 정렬되면 1.0 (clamping 0~1)
        error_scale = torch.clamp(1.0 - (real_diff / (torch.pi / 2)), 0.0, 1.0)
        
        # 최소 0.15 속도 보장 (Creeping)
        drive_scale = torch.maximum(error_scale, torch.tensor(0.15, device=self.device))
        actual_target_drive = final_target_drive * drive_scale
        
        # 가속도 제한 (Drive Slew Rate)
        drive_diff = actual_target_drive - self._prev_drive_vels
        drive_step = torch.clamp(drive_diff, -self._max_drive_accel * dt, self._max_drive_accel * dt)
        limited_drive_vels = self._prev_drive_vels + drive_step

        # 상태 업데이트
        self._prev_drive_vels = limited_drive_vels.clone()
        self._prev_steer_pos
        
        return limited_drive_vels, limited_steer_pos