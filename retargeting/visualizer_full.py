import argparse
import os
import numpy as np
import torch

from isaaclab.app import AppLauncher

# 1. Launcher 설정
parser = argparse.ArgumentParser(description="Isaac Lab Full-body IK Visualizer - SceneEntityCfg Style")
parser.add_argument("--hand_type", type=str, default="right", choices=["right", "left"], help="Hand type.")
parser.add_argument("--seq_idx", type=int, default=0, help="Sequence index.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.sim import SimulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg # 추가
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from pxr import Gf, UsdGeom

from dataset import DexYCBVideoDataset

@configclass
class FullBodySceneCfg(InteractiveSceneCfg):
    """씬 구성 및 로봇/환경 설정 클래스"""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    lights = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

    table = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            scale=(1.0, 1.0, 1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.3, 0.0), rot=(0.0, 0.0, 0.0, 1.0))
    )

    robot = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/robots/FFW/FFW_SH5_simplified.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, fix_root_link=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.75, 1.0, 0.0),
            rot=(0.70711, 0.0, 0.0, -0.70711),
            joint_pos={
                # # Swerve base joints
                "left_wheel_drive": 0.0, "left_wheel_steer": 0.0,
                "right_wheel_drive": 0.0, "right_wheel_steer": 0.0,
                "rear_wheel_drive": 0.0, "rear_wheel_steer": 0.0,

                # Left arm joints
                **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
                # Right arm joints
                **{f"arm_r_joint{i + 1}": 0.0 for i in range(7)},
                
                # Left and right finger joints
                **{f"finger_l_joint{i + 1}": 0.0 for i in range(20)},
                **{f"finger_r_joint{i + 1}": 0.0 for i in range(20)},

                # Head joints
                "head_joint1": 0.0,
                "head_joint2": 0.0,

                # Lift joint
                "lift_joint": 0.0,
            }
        ),
        actuators={
            # Actuators for swerve base
            "base": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_wheel_drive", "left_wheel_steer",
                    "right_wheel_drive", "right_wheel_steer",
                    "rear_wheel_drive", "rear_wheel_steer",
                ],
                velocity_limit_sim=30.0,
                effort_limit_sim=100000.0,
                stiffness=10000.0,
                damping=100.0,
            ),

            # Actuator for vertical lift joint
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["lift_joint"],
                velocity_limit_sim=0.2,
                effort_limit_sim=1000000.0,
                stiffness=10000.0,
                damping=100.0,
            ),

            # Actuators for both arms
            "DY_80": ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint[1-2]",
                    "arm_r_joint[1-2]",
                ],
                velocity_limit_sim=15.0,
                effort_limit_sim=1000.0,  # 61.4
                stiffness=600.0,
                damping=30.0,
            ),
            "DY_70": ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint[3-6]",
                    "arm_r_joint[3-6]",
                ],
                velocity_limit_sim=15.0,
                effort_limit_sim=1000.0,  # 31.7
                stiffness=600.0,
                damping=20.0,
            ),
            "DP-42" : ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint7",
                    "arm_r_joint7",
                ],
                velocity_limit_sim=6.0,
                effort_limit_sim=1000.0,  # 5.1
                stiffness=200.0,
                damping=3.0,
            ),

            # Actuators for hands
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["finger_l_joint[1-20]", "finger_r_joint[1-20]"],
                velocity_limit_sim=2.2,
                effort_limit_sim=1000.0,  # 20.0
                stiffness=2.0,
                damping=0.5,
            ),

            # Actuators for head joints
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_joint1", "head_joint2"],
                velocity_limit_sim=2.0,
                effort_limit_sim=1000.0,  # 30.0
                stiffness=150.0,
                damping=3.0,
            ),
        },
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, data: dict):
    """시뮬레이션 메인 루프"""
    robot: Articulation = scene["robot"]
    TABLE_HEIGHT = 1.00
    
    # 1. SceneEntityCfg 정의 (튜토리얼 방식)
    # 팔(Arm) 엔티티: IK 계산에 필요한 Joint들과 End-effector Body 지정
    arm_entity_cfg = SceneEntityCfg(
        "robot", 
        joint_names=[f"arm_r_joint{i + 1}" for i in range(7)], 
        body_names=["hx5_d20_right_base"]
    )
    arm_entity_cfg.resolve(scene)

    # 손(Hand) 엔티티: 데이터셋 qpos를 직접 주입할 관절들
    hand_entity_cfg = SceneEntityCfg(
        "robot", 
        joint_names=[f"finger_r_joint{i + 1}" for i in range(20)]
    )
    hand_entity_cfg.resolve(scene)
    
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/IKTarget",
        markers={
            "target": sim_utils.SphereCfg(radius=0.025, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
        }
    )
    markers = VisualizationMarkers(marker_cfg)

    # 2. IK 컨트롤러 및 인덱스 설정
    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05})
    ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Fixed base 로봇의 Jacobian 인덱스 보정 (튜토리얼 로직 반영)
    ee_jacobi_idx = arm_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else arm_entity_cfg.body_ids[0]

    # 데이터셋 매핑 (SceneEntity에서 찾은 실제 관절 이름 기반)
    retargeting_joint_names = data["joint_names"]
    retargeting_to_isaac = [retargeting_joint_names.index(name) for name in hand_entity_cfg.joint_names]

    # 객체 Xform 설정
    obj_prims = [UsdGeom.Xformable(sim.stage.GetPrimAtPath(f"/World/Object_{i}")) for i in range(len(data.get("object_names", [])))]
    for xform in obj_prims:
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(); xform.AddOrientOp()

    frame_idx = 0
    num_frames = len(data["qpos"])
    sim_dt = sim.get_physics_dt()
    
    t = 0.0

    while simulation_app.is_running():
        if frame_idx == 0:
            scene.reset()
            ik_controller.reset()

        # [A] 목표 및 현재 상태 추출 (SceneEntity 활용)
        target_pos_w = torch.tensor(data["root_pos"][frame_idx], device=sim.device).unsqueeze(0)
        target_pos_w[:, 2] += TABLE_HEIGHT
        target_quat_w = torch.tensor(data["root_quat"][frame_idx], device=sim.device).unsqueeze(0)
        
        markers.visualize(translations=target_pos_w, orientations=target_quat_w)

        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        
        # World -> Local 변환
        target_pos_b, target_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, target_quat_w)
        
        # 현재 EE Pose (arm_entity_cfg.body_ids[0] 사용)
        ee_pose_w = robot.data.body_pose_w[:, arm_entity_cfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])

        # [B] IK 계산
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_entity_cfg.joint_ids]
        curr_arm_q = robot.data.joint_pos[:, arm_entity_cfg.joint_ids]
    
        ik_controller.set_command(torch.cat([target_pos_w, target_quat_w], dim=-1))
        arm_next_q = ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, curr_arm_q)

        # [C] 명령 적용 (SceneEntity의 joint_ids 활용)
        robot.set_joint_position_target(arm_next_q, joint_ids=arm_entity_cfg.joint_ids)
        
        hand_q = torch.tensor(data["qpos"][frame_idx][retargeting_to_isaac], device=sim.device, dtype=torch.float32).unsqueeze(0)
        robot.set_joint_position_target(hand_q, joint_ids=hand_entity_cfg.joint_ids)

        # [D] 객체 업데이트
        for i, xform in enumerate(obj_prims):
            p, q = data["obj_poses"][i][frame_idx], data["obj_quats"][i][frame_idx]
            xform.GetOrderedXformOps()[0].Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2] + TABLE_HEIGHT)))
            xform.GetOrderedXformOps()[1].Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        
        t += 1
        if t % 10 == 0:
            frame_idx = (frame_idx + 1) % num_frames
            

# main 함수 내부, sim.reset() 이후에 추가해
def debug_robot_info(scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    
    print("-" * 50)
    print(f"[DEBUG] Robot Name: {robot.cfg.prim_path}")
    print(f"Number of Bodies: {robot.num_bodies}")
    print(f"Fixed Base: {robot.is_fixed_base}")
    print("-" * 50)

    # 1. 모든 바디(Link) 이름과 인덱스 출력
    print("ID | Body Name")
    print("-" * 30)
    for i, name in enumerate(robot.data.body_names):
        print(f"{i:2d} | {name}")
    
    print("-" * 50)

    # 2. 모든 관절(Joint) 이름과 인덱스 출력
    print("ID | Joint Name")
    print("-" * 30)
    for i, name in enumerate(robot.data.joint_names):
        print(f"{i:2d} | {name}")
    print("-" * 50)

def main():
    sim = sim_utils.SimulationContext(SimulationCfg(device="cuda", dt=0.01))
    
    dataset = DexYCBVideoDataset("DexYCB", hand_type=args_cli.hand_type)
    data = np.load(f"trajectories/{dataset[args_cli.seq_idx]['capture_name']}.npy", allow_pickle=True).item()

    scene = InteractiveScene(FullBodySceneCfg(num_envs=1, env_spacing=2.0))
    
    for i, name in enumerate(data.get("object_names", [])):
        sim_utils.spawn_from_usd(f"/World/Object_{i}", sim_utils.UsdFileCfg(usd_path=os.path.join("DexYCB/models", name, "textured.obj")))

    sim.reset()
    # debug_robot_info(scene)  # 로봇 정보 디버그 출력
    run_simulator(sim, scene, data)

if __name__ == "__main__":
    main()
    simulation_app.close()