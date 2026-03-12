# import numpy as np
# from isaacsim.core.api.controllers import BaseController

# class SwerveController(BaseController):
#     def __init__(self):
#         super().__init__(name="swerve_controller")
#         self._wheel_radius = 0.0825
#         self._wheel_positions = [
#             np.array([0.1371, 0.2554]), 
#             np.array([0.1371, -0.2554]), 
#             np.array([-0.2899, 0.0])
#         ]
        
#         self._prev_drive_vels = np.zeros(3)
#         self._prev_steer_pos = np.zeros(3)
        
#         self._max_drive_accel = 50.0
#         self._max_steer_vel = 15.0
#         self._angle_threshold = np.deg2rad(3.0)

#     def _normalize_angle(self, angle: float) -> float:
#         return (angle + np.pi) % (2 * np.pi) - np.pi

#     def forward(self, command, current_steer_angles, dt):
#         vx, vy, w = command
#         w = w * 1.2 
        
#         target_drive_vels = []
#         target_steer_pos = []

#         # 1. 목표 각도 및 속도 계산 (IK)
#         for i, pos in enumerate(self._wheel_positions):
#             vix = vx - w * pos[1]
#             viy = vy + w * pos[0]
#             speed = np.sqrt(vix**2 + viy**2)
            
#             if speed > 1e-3:
#                 desired_angle = np.arctan2(viy, vix)
#                 desired_speed = speed / self._wheel_radius
                
#                 # Wheel Reversal 적용
#                 diff = self._normalize_angle(desired_angle - self._prev_steer_pos[i])
#                 if abs(diff) > np.pi / 2:
#                     desired_angle = self._normalize_angle(desired_angle + np.pi)
#                     desired_speed *= -1.0
                
#                 target_drive_vels.append(desired_speed)
#                 target_steer_pos.append(desired_angle)
#             else:
#                 target_drive_vels.append(0.0)
#                 target_steer_pos.append(self._prev_steer_pos[i])

#         # 2. 이동 중 조향을 위해 '선 정지' 로직 제거
#         # 대신, 조향 오차가 너무 크면(예: 45도 이상) 구동 속도를 줄여 기계적 무리를 방지해.
#         final_target_drive = np.array(target_drive_vels)
#         final_target_steer = np.array(target_steer_pos)

#         limited_steer_pos = []
#         limited_drive_vels = []
#         all_aligned = True

#         # 3. Steer 제어 (Slew Rate 적용)
#         for i in range(3):
#             steer_diff = self._normalize_angle(final_target_steer[i] - self._prev_steer_pos[i])
#             steer_step = np.clip(steer_diff, -self._max_steer_vel * dt, self._max_steer_vel * dt)
#             new_steer = self._normalize_angle(self._prev_steer_pos[i] + steer_step)
#             limited_steer_pos.append(new_steer)

#             # 실제 각도와 목표 각도의 차이 확인
#             real_diff = abs(self._normalize_angle(final_target_steer[i] - current_steer_angles[i]))
#             if real_diff > self._angle_threshold:
#                 all_aligned = False

#         # 4. Drive 제어 (가속도 50.0 반영)
#         for i in range(3):
#             # 조향 오차에 따른 속도 스케일링 (완전 정지 대신 감속 사용)
#             # 90도 이상 틀어져 있으면 속도를 거의 0으로, 정렬될수록 100%에 가깝게.
#             real_diff = abs(self._normalize_angle(final_target_steer[i] - current_steer_angles[i]))
#             error_scale = np.clip(1.0 - (real_diff / (np.pi / 2)), 0.0, 1.0)
            
#             # Creeping 효과를 포함하여 최소한의 움직임 보장
#             drive_scale = max(error_scale, 0.15) 
#             actual_target = final_target_drive[i] * drive_scale
            
#             drive_diff = actual_target - self._prev_drive_vels[i]
#             drive_step = np.clip(drive_diff, -self._max_drive_accel * dt, self._max_drive_accel * dt)
#             limited_drive_vels.append(self._prev_drive_vels[i] + drive_step)

#         self._prev_drive_vels = np.array(limited_drive_vels)
#         self._prev_steer_pos = np.array(limited_steer_pos)

#         return limited_drive_vels, limited_steer_pos

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