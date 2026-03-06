import numpy as np
from isaacsim import SimulationApp

# 1. 시뮬레이션 앱 초기화
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.kit.commands
import omni.ui as ui
from pxr import UsdPhysics

# 업로드한 파일에서 SwerveDriveController 클래스 로드
from OgnSwerveDriveController import SwerveDriveController

class FFW_SG2_SimpleManager:
    def __init__(self, usd_path, prim_path):
        self._world = World(stage_units_in_meters=1.0)
        ground_plane = self._world.scene.add_default_ground_plane()
        
        # 로봇 모델 로드
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            
        self._robot = Robot(prim_path=prim_path, name="swerve_robot")
        
        # 제공된 로봇 기하학적 파라미터
        self.wheel_radius = np.array([0.0865, 0.0865, 0.0865]) 
        self.wheel_positions = np.array([
            [ 0.1371,  0.2554, 0.0865], # Left
            [ 0.1371, -0.2554, 0.0865], # Right
            [-0.2899,  0.0   , 0.0865]  # Rear
        ])

        # Swerve 컨트롤러 초기화
        self.controller = SwerveDriveController(
            name="ffw_swerve_logic",
            wheel_radius=self.wheel_radius,
            wheel_positions=self.wheel_positions,
            optimize_steering=False # 조향 최적화 사용
        )

        # 조인트 이름 설정
        self.steer_names = ["left_wheel_steer", "right_wheel_steer", "rear_wheel_steer"]
        self.drive_names = ["left_wheel_drive", "right_wheel_drive", "rear_wheel_drive"]
        
        self.cmd = np.array([0.0, 0.0, 0.0]) # [vx, vy, omega]
        self.build_ui()

    def build_ui(self):
        self._window = ui.Window("Swerve Logic Only", width=300, height=180)
        with self._window.frame:
            with ui.VStack(spacing=5):
                ui.Label("vx")
                ui.FloatSlider(min=-1.0, max=1.0).model.add_value_changed_fn(lambda m: self._set(0, m))
                ui.Label("vy")
                ui.FloatSlider(min=-1.0, max=1.0).model.add_value_changed_fn(lambda m: self._set(1, m))
                ui.Label("omega")
                ui.FloatSlider(min=-2.0, max=2.0).model.add_value_changed_fn(lambda m: self._set(2, m))

    def _set(self, i, m): self.cmd[i] = m.get_value_as_float()

    def run(self):        
        self._world.reset()
        self._robot.initialize()
        
        dof_names = self._robot.dof_names
        print(f"✅ 인식된 DOF 이름들: {dof_names}")
        
        if not dof_names:
            print("❌ 여전히 DOF를 못 찾았어. 경로가 정확한지 다시 확인해봐!")
            return
        
        # 1. 인덱스 확보
        steer_idx = [self._robot.get_dof_index(n) for n in self.steer_names]
        drive_idx = [self._robot.get_dof_index(n) for n in self.drive_names]
        
        # 전체 제어 인덱스 합치기 (총 6개)
        combined_indices = steer_idx + drive_idx
        
        art_controller = self._robot.get_articulation_controller()

        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_playing():
                front_rx = 0.1371
                rear_rx = abs(-0.2899)
                ratio = front_rx / rear_rx
                
                # 2. 스워브 로직 계산 (각각 3개씩 반환됨)
                drive_vels, steer_angles = self.controller.forward(self.cmd)
                
                # 3. [에러 해결 핵심] 6개짜리 명령 배열 생성
                # 위치 명령: [s1, s2, s3, d1, d2, d3] 순서인데 drive는 위치 제어 안 하니까 None
                pos_cmd = np.array([steer_angles[0], steer_angles[1], steer_angles[2], 0, 0, 0], dtype=np.float32)
                # 속도 명령: [s1, s2, s3, d1, d2, d3] 순서인데 steer는 속도 제어 안 하니까 0 (또는 생략)
                vel_cmd = np.array([0, 0, 0, drive_vels[0], drive_vels[1], drive_vels[2]], dtype=np.float32)

                # 4. ArticulationAction 생성
                # joint_indices의 개수(6개)와 각 cmd의 개수(6개)를 일치시킴
                action = ArticulationAction(
                    joint_positions=pos_cmd,
                    joint_velocities=vel_cmd,
                    joint_indices=np.array(combined_indices, dtype=np.int32)
                )
                
                art_controller.apply_action(action)

        simulation_app.close()

if __name__ == "__main__":
    FFW_SG2_SimpleManager(
        "D:/robotis/robots/FFW/FFW_SG2_real_final_final.usd", 
        "/World/FFW_SG2_real_final_final"
    ).run()