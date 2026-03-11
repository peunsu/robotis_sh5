# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaacsim import SimulationApp

# 1. 앱 초기화 (렌더링 설정)
simulation_app = SimulationApp({"headless": False})

import os
import omni.kit.commands
import omni.ui as ui
import numpy as np
from omni.isaac.core import World
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.api.controllers import BaseController

# --- 기존 SwerveController 클래스 ---
class SwerveController(BaseController):
    def __init__(self):
        super().__init__(name="swerve_controller")
        self._wheel_radius = 0.0825
        self._wheel_positions = [
            np.array([0.1371, 0.2554]), 
            np.array([0.1371, -0.2554]), 
            np.array([-0.2899, 0.0])
        ]
        
        self._prev_drive_vels = np.zeros(3)
        self._prev_steer_pos = np.zeros(3)
        
        self._max_drive_accel = 50.0
        self._max_steer_vel = 15.0
        self._angle_threshold = np.deg2rad(3.0)

    def _normalize_angle(self, angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def forward(self, command, current_steer_angles, dt):
        vx, vy, w = command
        w = w * 1.2 
        
        target_drive_vels = []
        target_steer_pos = []

        # 1. 목표 각도 및 속도 계산 (IK)
        for i, pos in enumerate(self._wheel_positions):
            vix = vx - w * pos[1]
            viy = vy + w * pos[0]
            speed = np.sqrt(vix**2 + viy**2)
            
            if speed > 1e-3:
                desired_angle = np.arctan2(viy, vix)
                desired_speed = speed / self._wheel_radius
                
                # Wheel Reversal 적용
                diff = self._normalize_angle(desired_angle - self._prev_steer_pos[i])
                if abs(diff) > np.pi / 2:
                    desired_angle = self._normalize_angle(desired_angle + np.pi)
                    desired_speed *= -1.0
                
                target_drive_vels.append(desired_speed)
                target_steer_pos.append(desired_angle)
            else:
                target_drive_vels.append(0.0)
                target_steer_pos.append(self._prev_steer_pos[i])

        # 2. 이동 중 조향을 위해 '선 정지' 로직 제거
        # 대신, 조향 오차가 너무 크면(예: 45도 이상) 구동 속도를 줄여 기계적 무리를 방지해.
        final_target_drive = np.array(target_drive_vels)
        final_target_steer = np.array(target_steer_pos)

        limited_steer_pos = []
        limited_drive_vels = []
        all_aligned = True

        # 3. Steer 제어 (Slew Rate 적용)
        for i in range(3):
            steer_diff = self._normalize_angle(final_target_steer[i] - self._prev_steer_pos[i])
            steer_step = np.clip(steer_diff, -self._max_steer_vel * dt, self._max_steer_vel * dt)
            new_steer = self._normalize_angle(self._prev_steer_pos[i] + steer_step)
            limited_steer_pos.append(new_steer)

            # 실제 각도와 목표 각도의 차이 확인
            real_diff = abs(self._normalize_angle(final_target_steer[i] - current_steer_angles[i]))
            if real_diff > self._angle_threshold:
                all_aligned = False

        # 4. Drive 제어 (가속도 50.0 반영)
        for i in range(3):
            # 조향 오차에 따른 속도 스케일링 (완전 정지 대신 감속 사용)
            # 90도 이상 틀어져 있으면 속도를 거의 0으로, 정렬될수록 100%에 가깝게.
            real_diff = abs(self._normalize_angle(final_target_steer[i] - current_steer_angles[i]))
            error_scale = np.clip(1.0 - (real_diff / (np.pi / 2)), 0.0, 1.0)
            
            # Creeping 효과를 포함하여 최소한의 움직임 보장
            drive_scale = max(error_scale, 0.15) 
            actual_target = final_target_drive[i] * drive_scale
            
            drive_diff = actual_target - self._prev_drive_vels[i]
            drive_step = np.clip(drive_diff, -self._max_drive_accel * dt, self._max_drive_accel * dt)
            limited_drive_vels.append(self._prev_drive_vels[i] + drive_step)

        self._prev_drive_vels = np.array(limited_drive_vels)
        self._prev_steer_pos = np.array(limited_steer_pos)

        return limited_drive_vels, limited_steer_pos

# --- 메인 시뮬레이션 클래스 ---
class SwerveApp:
    def __init__(self):
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()
        
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        
        self._robot_name = "FFW_SH5"
        self._robot_prim_path = f"/World/{self._robot_name}/base_link"
        self._robot_asset_path = f"{current_script_path}/../source/robotis_sh5/data/robots/FFW/FFW_SH5.usd"
        
        self._command = np.zeros(3)
        self._speed = 0.5
        self._ang_speed = 1.0
        
        self._setup_scene()
        self._setup_ui()
        
    def _setup_scene(self):
        # 재질 설정
        self._wheel_material = PhysicsMaterial(
            prim_path="/World/Materials/WheelMaterial",
            static_friction=1.5,
            dynamic_friction=1.2,
            restitution=0.0
        )
        
        self._wheel_dof_names = [
            "left_wheel_drive", "right_wheel_drive", "rear_wheel_drive",
            "left_wheel_steer", "right_wheel_steer", "rear_wheel_steer"
        ]

        self._bot = self._world.scene.add(
            WheeledRobot(
                prim_path=self._robot_prim_path,
                name=self._robot_name,
                wheel_dof_names=self._wheel_dof_names,
                create_robot=True,
                usd_path=self._robot_asset_path,
            )
        )

    def _setup_ui(self):
        self._window = ui.Window("Robot UI Controller", width=350, height=450)
        with self._window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Swerve 8-Way Motion Control", alignment=ui.Alignment.CENTER, style={"font_size": 16})
                
                # 속도 설정 영역
                with ui.HStack(height=20):
                    ui.Label("Linear Speed:", width=100)
                    speed_slider = ui.FloatSlider(min=0.1, max=2.0)
                    speed_slider.model.set_value(self._speed)
                    speed_slider.model.add_value_changed_fn(lambda m: setattr(self, '_speed', m.get_value_as_float()))

                ui.Spacer(height=5)

                # 3x3 그리드 레이아웃
                diag = 0.707
                with ui.VStack(spacing=5):
                    # 1행: 전좌측, 전진, 전우측
                    with ui.HStack(spacing=5):
                        ui.Button("Front Left", clicked_fn=lambda: self._set_cmd(self._speed * diag, self._speed * diag, 0))
                        ui.Button("FORWARD", clicked_fn=lambda: self._set_cmd(self._speed, 0, 0))
                        ui.Button("Front Right", clicked_fn=lambda: self._set_cmd(self._speed * diag, -self._speed * diag, 0))
                    
                    # 2행: 좌측, 정지, 우측
                    with ui.HStack(spacing=5):
                        ui.Button("LEFT", clicked_fn=lambda: self._set_cmd(0, self._speed, 0))
                        ui.Button("STOP", clicked_fn=lambda: self._set_cmd(0, 0, 0), 
                                  style={"background_color": 0xFF0000AA, "color": 0xFFFFFFFF, "font_weight": "bold"})
                        ui.Button("RIGHT", clicked_fn=lambda: self._set_cmd(0, -self._speed, 0))
                    
                    # 3행: 후좌측, 후진, 후우측
                    with ui.HStack(spacing=5):
                        ui.Button("Back Left", clicked_fn=lambda: self._set_cmd(-self._speed * diag, self._speed * diag, 0))
                        ui.Button("BACKWARD", clicked_fn=lambda: self._set_cmd(-self._speed, 0, 0))
                        ui.Button("Back Right", clicked_fn=lambda: self._set_cmd(-self._speed * diag, -self._speed * diag, 0))

                ui.Spacer(height=10)

                # 회전 제어 영역
                with ui.HStack(spacing=5, height=40):
                    ui.Button("Rotate Counter-Clockwise", clicked_fn=lambda: self._set_cmd(0, 0, self._ang_speed))
                    ui.Button("Rotate Clockwise", clicked_fn=lambda: self._set_cmd(0, 0, -self._ang_speed))

    def _set_cmd(self, vx, vy, wz):
        self._command = np.array([vx, vy, wz])

    def run(self):
        self._world.reset()
        self._world.get_physics_context().set_solver_type("TGS")
        
        # 바퀴 재질 바인딩
        wheel_links = [f"{self._robot_prim_path}/{n}_link" for n in ["left_wheel_drive", "right_wheel_drive", "rear_wheel_drive"]]
        for link_path in wheel_links:
            omni.kit.commands.execute("BindMaterialCommand", prim_path=[link_path], material_path=self._wheel_material.prim_path, strength="stronger")

        controller = SwerveController()
        dof_indices = [self._bot.get_dof_index(name) for name in self._wheel_dof_names]
        
        # 게인 설정
        num_dof = self._bot.num_dof
        kps, kds = [1e6] * num_dof, [1e4] * num_dof
        for i in range(3):
            kps[dof_indices[i]], kds[dof_indices[i]] = 0.0, 1e5
        self._bot.get_articulation_controller().set_gains(kps=kps, kds=kds)

        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_playing():
                step_size = self._world.get_physics_dt()
                current_joint_positions = self._bot.get_joint_positions()
                current_steer_angles = [current_joint_positions[idx] for idx in dof_indices[3:6]]

                drive_vels, steer_pos = controller.forward(self._command, current_steer_angles, step_size)

                all_pos, all_vel = [None] * num_dof, [None] * num_dof
                for i, idx in enumerate(dof_indices):
                    if i < 3:
                        all_vel[idx] = drive_vels[i]
                    else:
                        all_pos[idx] = steer_pos[i-3]
                        all_vel[idx] = 0.0

                self._bot.apply_action(ArticulationAction(joint_positions=all_pos, joint_velocities=all_vel))

        simulation_app.close()

if __name__ == "__main__":
    app = SwerveApp()
    app.run()