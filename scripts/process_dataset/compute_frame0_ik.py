"""Precompute arm IK solutions for trajectory frame 0 wrist positions.

For each trajectory, solves DLS IK to find the arm_r_joint angles that place the
wrist (hx5_d20_right_base) at the canonicalized frame-0 reference position/orientation.
Result is saved as frame0_arm_joint_pos.npy in each trajectory directory.

Run once before training:
    python scripts/process_dataset/compute_frame0_ik.py
    python scripts/process_dataset/compute_frame0_ik.py --task A01001-0001-0000 --data_id 0
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Precompute arm IK for trajectory frame 0.")
parser.add_argument("--object_id", type=str, default="", help="OakInk object ID to process (empty = all objects).")
parser.add_argument("--task", type=str, default="", help="Specific task directory (empty = all tasks).")
parser.add_argument("--data_id", type=int, default=-1, help="Specific data ID (-1 = all).")
parser.add_argument("--num_iter", type=int, default=300, help="IK iteration count.")
parser.add_argument("--overwrite", action="store_true", help="Re-run even if output file already exists.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_mul

_SCRIPT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPT_DIR / "source" / "robotis_sh5"))
from robotis_sh5.tasks.direct.robotis_sh5_grasp.robotis_sh5_grasp_env_cfg import FFW_SH5_DEX_CFG

_DATA_DIR = _SCRIPT_DIR / "source" / "robotis_sh5" / "data"
_OAKINK_DIR = _DATA_DIR / "processed" / "oakink"

# Must match RobotisSh5GraspPretrainEnvCfg defaults
_TABLE_POS = (0.3, 0.0, 0.0)
_TABLE_SIZE = (0.6, 0.6, 1.0)
_ROBOT_POS = (0.65, 0.55, 0.0)


# ---------------------------------------------------------------------------
# Canonicalization helpers (mirrors _load_reference_trajectories logic)
# ---------------------------------------------------------------------------

def _canonicalize_frame0(traj_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return canonicalized frame-0 wrist (pos, quat-wxyz) in env-local coords."""
    data = np.load(str(traj_path))
    wp = data["qpos_wrist_right"][:, :3].astype(np.float32)
    wq = data["qpos_wrist_right"][:, 3:].astype(np.float32)   # wxyz
    op = data["qpos_obj_right"][:, :3].astype(np.float32)

    # Infer object_id from directory structure: .../mano/right/<task>/<id>/...
    object_id = traj_path.parent.parent.name.split("-")[0]
    mesh_path = _OAKINK_DIR / "assets" / "objects" / object_id / "visual.obj"
    if mesh_path.exists():
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        mesh_z_min = float(mesh.vertices[:, 2].min())
    else:
        mesh_z_min = 0.0
        print(f"[warn] mesh not found at {mesh_path}; using mesh_z_min=0")

    table_surface_z = float(_TABLE_SIZE[2])
    target_centroid_z = table_surface_z - mesh_z_min
    table_target = np.array([_TABLE_POS[0], _TABLE_POS[1], target_centroid_z], dtype=np.float32)
    offset = table_target - op[0]
    op = op + offset
    wp = wp + offset

    # XY canonicalization toward table edge facing robot
    tx, ty = float(_TABLE_POS[0]), float(_TABLE_POS[1])
    hx, hy = float(_TABLE_SIZE[0]) / 2.0, float(_TABLE_SIZE[1]) / 2.0
    dx = float(_ROBOT_POS[0]) - tx
    dy = float(_ROBOT_POS[1]) - ty
    if abs(dy) >= abs(dx):
        ref_xy = np.array([tx, ty + hy * np.sign(dy)], dtype=np.float32)
    else:
        ref_xy = np.array([tx + hx * np.sign(dx), ty], dtype=np.float32)

    canonical_dir = ref_xy - np.array([tx, ty], dtype=np.float32)
    if np.linalg.norm(canonical_dir) < 1e-6:
        return wp[0], wq[0]
    canonical_dir /= np.linalg.norm(canonical_dir)

    o0 = op[0]
    wrist_dir = wp[0, :2] - o0[:2]
    if np.linalg.norm(wrist_dir) < 1e-4:
        return wp[0], wq[0]
    wrist_dir /= np.linalg.norm(wrist_dir)

    cos_a = float(np.clip(np.dot(wrist_dir, canonical_dir), -1.0, 1.0))
    sin_a = float(wrist_dir[0] * canonical_dir[1] - wrist_dir[1] * canonical_dir[0])
    angle = float(np.arctan2(sin_a, cos_a))
    c, s = float(np.cos(angle)), float(np.sin(angle))
    R2 = np.array([[c, -s], [s, c]], dtype=np.float32)
    ox, oy = float(o0[0]), float(o0[1])

    # Rotate wrist position
    wp_rot = wp.copy()
    wp_rot[:, 0] -= ox; wp_rot[:, 1] -= oy
    wp_rot[:, :2] = wp_rot[:, :2] @ R2.T
    wp_rot[:, 0] += ox; wp_rot[:, 1] += oy

    # Rotate wrist quaternion: q_rot = [cos(a/2), 0, 0, sin(a/2)] in wxyz
    hw = angle / 2.0
    qw_r, qz_r = float(np.cos(hw)), float(np.sin(hw))
    w2, x2, y2, z2 = wq[:, 0], wq[:, 1], wq[:, 2], wq[:, 3]
    wq_rot = np.stack([
        qw_r * w2 - qz_r * z2,
        qw_r * x2 - qz_r * y2,
        qw_r * y2 + qz_r * x2,
        qw_r * z2 + qz_r * w2,
    ], axis=-1).astype(np.float32)

    return wp_rot[0], wq_rot[0]


# ---------------------------------------------------------------------------
# IK runner
# ---------------------------------------------------------------------------

def _run_ik(
    robot: Articulation,
    ik_ctrl: DifferentialIKController,
    arm_joint_ids: list[int],
    arm_ids_t: torch.Tensor,
    jac_body_idx: int,
    wrist_body_id: int,
    target_pos: torch.Tensor,   # (1, 3) world frame
    target_quat: torch.Tensor,  # (1, 4) wxyz world frame
    num_iter: int,
    sim: SimulationContext,
) -> torch.Tensor:
    """Run DLS IK and return converged arm joint angles (7,)."""
    # Reset robot to default joint positions
    default_jp = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(default_jp, torch.zeros_like(default_jp))
    sim.step()
    robot.update(sim.get_physics_dt())

    ik_ctrl.reset()
    ik_ctrl.set_command(torch.cat([target_pos, target_quat], dim=-1))

    full_jp = robot.data.joint_pos.clone()
    for _ in range(num_iter):
        ee_pos = robot.data.body_pos_w[:, wrist_body_id, :]    # (1, 3)
        ee_quat = robot.data.body_quat_w[:, wrist_body_id, :]  # (1, 4)

        # Jacobian: (1, num_bodies-1, 6, num_dofs) for fixed-base
        J = robot.root_physx_view.get_jacobians()
        J_arm = J[:, jac_body_idx, :, arm_ids_t]               # (1, 6, 7)

        arm_jp = robot.data.joint_pos[:, arm_ids_t]             # (1, 7)
        targets = ik_ctrl.compute(ee_pos, ee_quat, J_arm, arm_jp)

        # Write directly to sim (bypass actuator dynamics for fast convergence)
        full_jp[:, arm_ids_t] = targets
        robot.write_joint_state_to_sim(full_jp, torch.zeros_like(full_jp))
        sim.step()
        robot.update(sim.get_physics_dt())

    final = robot.data.joint_pos[:, arm_ids_t].squeeze(0)      # (7,)
    err = torch.norm(robot.data.body_pos_w[:, wrist_body_id, :] - target_pos)
    return final, float(err.item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sim_cfg = SimulationCfg(dt=1.0 / 120.0, render_interval=1)
    sim = SimulationContext(sim_cfg)

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    # Use single env (env_origin = (0,0,0))
    robot_cfg = FFW_SH5_DEX_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"  # no env_ wildcard for standalone script
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())

    device = sim.device

    arm_joint_ids, _ = robot.find_joints("arm_r_joint.*")
    arm_ids_t = torch.tensor(arm_joint_ids, dtype=torch.long, device=device)
    wrist_ids, _ = robot.find_bodies("hx5_d20_right_base")
    wrist_body_id = wrist_ids[0]
    # For fixed-base robots, Jacobian rows are indexed body_id - 1 (base excluded)
    jac_body_idx = wrist_body_id - 1

    print(f"Arm joint IDs : {arm_joint_ids}")
    print(f"Wrist body ID : {wrist_body_id}  →  Jacobian row {jac_body_idx}")

    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )
    ik_ctrl = DifferentialIKController(ik_cfg, num_envs=1, device=device)

    # Collect trajectory paths
    mano_dir = _OAKINK_DIR / "mano" / "right"
    if args_cli.task:
        task_dirs = [mano_dir / args_cli.task]
    else:
        task_dirs = sorted(
            d for d in mano_dir.iterdir()
            if d.is_dir() and (not args_cli.object_id or args_cli.object_id in d.name)
        )

    processed = skipped = errors = 0
    for task_dir in task_dirs:
        if args_cli.data_id >= 0:
            data_dirs = [task_dir / str(args_cli.data_id)]
        else:
            data_dirs = sorted(d for d in task_dir.iterdir() if d.is_dir())

        for data_dir in data_dirs:
            traj_path = data_dir / "trajectory_keypoints.npz"
            if not traj_path.exists():
                continue

            out_path = data_dir / "frame0_arm_joint_pos.npy"
            if out_path.exists() and not args_cli.overwrite:
                print(f"[skip] {out_path.relative_to(_OAKINK_DIR)} already exists")
                skipped += 1
                continue

            result = _canonicalize_frame0(traj_path)
            if result is None:
                errors += 1
                continue
            wrist_pos_np, wrist_quat_np = result

            target_pos = torch.tensor(wrist_pos_np, dtype=torch.float32, device=device).unsqueeze(0)
            target_quat = torch.tensor(wrist_quat_np, dtype=torch.float32, device=device).unsqueeze(0)

            arm_jp, pos_err = _run_ik(
                robot, ik_ctrl, arm_joint_ids, arm_ids_t, jac_body_idx,
                wrist_body_id, target_pos, target_quat, args_cli.num_iter, sim,
            )

            np.save(str(out_path), arm_jp.cpu().numpy())
            rel = traj_path.parent.relative_to(_OAKINK_DIR)
            print(f"[ok] {rel}  pos_err={pos_err:.4f}m  → {out_path.name}")
            processed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped, {errors} errors.")
    import os; os._exit(0)


if __name__ == "__main__":
    main()
