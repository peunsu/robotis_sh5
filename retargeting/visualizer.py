import argparse
import os
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab Stable Visualizer (No Drift)")
parser.add_argument("--trajectory_path", type=str, required=True)
parser.add_argument("--robot_urdf", type=str, required=True)
parser.add_argument("--dexycb_models_dir", type=str, default="DexYCB/models")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from pxr import Gf, UsdGeom


def main():
    sim = sim_utils.SimulationContext(SimulationCfg(device="cuda", dt=0.01))

    sim_utils.spawn_ground_plane("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.spawn_light("/World/Light/Dome", sim_utils.DomeLightCfg(intensity=2000.0))

    # -------------------------
    # Load data
    # -------------------------
    data = np.load(args_cli.trajectory_path, allow_pickle=True).item()

    object_names = data.get("object_names", [])
    obj_poses = data["obj_poses"]
    obj_quats = data["obj_quats"]
    qpos_raw = data["qpos"]
    root_pos_raw = data["root_pos"]
    root_quat_raw = data["root_quat"]
    retargeting_joint_names = data["joint_names"]

    # -------------------------
    # Spawn objects
    # -------------------------
    obj_prims = []
    for i, name in enumerate(object_names):
        prim_path = f"/World/Object_{i}"
        obj_path = os.path.join(args_cli.dexycb_models_dir, name, "textured.obj")
        sim_utils.spawn_from_usd(prim_path, sim_utils.UsdFileCfg(usd_path=obj_path))
        obj_prims.append(prim_path)

    # -------------------------
    # Robot
    # -------------------------
    joint_drive_cfg = UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness={".*": 1e6},
            damping={".*": 1e4},
        )
    )

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=args_cli.robot_urdf,
            fix_base=False,
            joint_drive=joint_drive_cfg,
        ),
        actuators={
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=1e6,
                damping=1e4,
            )
        },
    )

    robot = Articulation(robot_cfg)

    sim.reset()
    robot.reset()

    stage = sim.stage
    env_ids = torch.tensor([0], device=sim.device)

    retargeting_to_isaac = [
        retargeting_joint_names.index(n) for n in robot.joint_names
    ]

    # -------------------------
    # USD transform 초기화
    # -------------------------
    xform_utils = []
    for prim_path in obj_prims:
        prim = stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp()
        xform.AddOrientOp()
        xform_utils.append(xform)

    frame_idx = 0
    num_frames = len(qpos_raw)

    # -------------------------
    # Main loop
    # -------------------------
    while simulation_app.is_running():

        # 🔥 루프 처음에서 완전 초기화 (drift 방지 핵심)
        if frame_idx == 0:
            sim.reset()
            robot.reset()

        # -------------------------
        # Data
        # -------------------------
        current_qpos = torch.tensor(
            qpos_raw[frame_idx][retargeting_to_isaac],
            device=sim.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        curr_root_pos = torch.tensor(
            root_pos_raw[frame_idx],
            device=sim.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        curr_root_quat = torch.tensor(
            root_quat_raw[frame_idx],
            device=sim.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        robot_pose = torch.cat([curr_root_pos, curr_root_quat], dim=-1)

        # -------------------------
        # Object update
        # -------------------------
        for i, xform in enumerate(xform_utils):
            p = obj_poses[i][frame_idx]
            q = obj_quats[i][frame_idx]

            xform.GetOrderedXformOps()[0].Set(
                Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
            )
            xform.GetOrderedXformOps()[1].Set(
                Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            )

        # -------------------------
        # 🔥 Physics 완전 무력화
        # -------------------------
        zero_root_vel = torch.zeros((1, 6), device=sim.device)
        zero_joint_vel = torch.zeros_like(current_qpos)

        robot.write_root_velocity_to_sim(zero_root_vel, env_ids=env_ids)
        robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)
        robot.write_joint_state_to_sim(current_qpos, zero_joint_vel, env_ids=env_ids)

        # -------------------------
        # Step
        # -------------------------
        sim.step(render=True)

        frame_idx = (frame_idx + 1) % num_frames

    simulation_app.close()


if __name__ == "__main__":
    main()