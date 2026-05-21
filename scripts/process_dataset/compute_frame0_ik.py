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
parser.add_argument("--dataset", type=str, default="oakink", choices=["oakink", "hocap"], help="Dataset to process.")
parser.add_argument("--object_id", type=str, default="", help="Object ID to process (empty = all objects).")
parser.add_argument("--task", type=str, default="", help="Specific task directory (empty = all tasks).")
parser.add_argument("--data_id", type=int, default=-1, help="Specific data ID (-1 = all).")
parser.add_argument("--num_iter", type=int, default=300, help="IK iteration count.")
parser.add_argument("--overwrite", action="store_true", help="Re-run even if output file already exists.")
parser.add_argument("--max_lift_iter", type=int, default=10,
                    help="Max outer iterations to lift wrist target z so all 21 hand keypoints have z >= table_top.")
parser.add_argument("--lift_margin", type=float, default=0.005,
                    help="Safety margin (m) added to wrist lift each outer iter.")
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
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

_SCRIPT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPT_DIR / "source" / "robotis_sh5"))
from robotis_sh5.tasks.direct.robotis_sh5_grasp.robotis_sh5_grasp_env_cfg import FFW_SH5_DEX_CFG
from robotis_sh5.tasks.direct.robotis_sh5_grasp.robotis_sh5_grasp_env import (
    _FINGERTIP_OFFSETS, _MANO_NON_FT_BODY_NAMES,
)

_DATA_DIR = _SCRIPT_DIR / "source" / "robotis_sh5" / "data"
_OAKINK_DIR = _DATA_DIR / "processed" / "oakink"
_HOCAP_DIR = _DATA_DIR / "processed" / "hocap"

# Must match RobotisSh5GraspPretrainEnvCfg defaults
_TABLE_POS = (0.3, 0.0, 0.0)
_TABLE_SIZE = (0.6, 0.6, 1.0)
_ROBOT_POS = (0.65, 0.65, 0.0)   # must match FFW_SH5_DEX_CFG.init_state.pos in env_cfg


# ---------------------------------------------------------------------------
# Canonicalization helpers (mirrors _load_reference_trajectories logic)
# ---------------------------------------------------------------------------

def _canonicalize_frame0(traj_path: Path, dataset_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return canonicalized frame-0 wrist (pos, quat-wxyz) in env-local coords."""
    data = np.load(str(traj_path))
    wp = data["qpos_wrist_right"][:, :3].astype(np.float32)
    wq = data["qpos_wrist_right"][:, 3:].astype(np.float32)   # wxyz
    op = data["qpos_obj_right"][:, :3].astype(np.float32)

    # Resolve mesh path. Prefer task_info.json's `right_object_mesh_dir` (HO-Cap stores
    # the actual object id there; e.g. "hocap/assets/objects/G09_4" — splitting the task
    # name on '-' gives wrong "subject_2"). Fall back to dash-split heuristic for OakInk
    # (task naming convention: "<object_id>-XXXX-XXXX").
    task_dir = traj_path.parent.parent   # .../mano/right/<task>/
    info_path = task_dir / "task_info.json"
    object_id = None
    if info_path.exists():
        try:
            import json
            info = json.loads(info_path.read_text())
            mesh_dir_rel = info.get("right_object_mesh_dir", "")
            if mesh_dir_rel:
                object_id = Path(mesh_dir_rel).name   # last path component
        except Exception:
            object_id = None
    if object_id is None:
        object_id = task_dir.name.split("-")[0]

    oq0 = data["qpos_obj_right"][0, 3:].astype(np.float32)   # wxyz
    mesh_path = dataset_dir / "assets" / "objects" / object_id / "visual.obj"
    if mesh_path.exists():
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        verts = np.array(mesh.vertices, dtype=np.float32)
        w, x, y, z = float(oq0[0]), float(oq0[1]), float(oq0[2]), float(oq0[3])
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1-2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x + y*y)],
        ], dtype=np.float32)
        mesh_z_min = float((verts @ R.T)[:, 2].min())
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
    ee_pos = robot.data.body_pos_w[:, wrist_body_id, :]
    ee_quat = robot.data.body_quat_w[:, wrist_body_id, :]
    pos_err = torch.norm(ee_pos - target_pos)
    # Rotation error: geodesic angle (rad) between current and target unit quats
    qa = ee_quat / ee_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    qb = target_quat / target_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dot = (qa * qb).sum(dim=-1).abs().clamp(0.0, 1.0)
    rot_err = 2.0 * torch.acos(dot)
    return final, float(pos_err.item()), float(rot_err.item())


def _compute_kpts_min_z(
    robot: Articulation,
    kpt_body_ids_t: torch.Tensor,      # (16,) non-fingertip body ids
    ft_body_ids_t: torch.Tensor,       # (5,)  fingertip body ids
    ft_offsets_t: torch.Tensor,        # (5, 3) local-frame offsets
) -> float:
    """Compute min z over all 21 MANO keypoints — mirrors `_compute_hand_kpts_pos` in env."""
    # Non-fingertip: link origins (16,)
    non_ft_z = robot.data.body_pos_w[:, kpt_body_ids_t, 2]            # (1, 16)
    # Fingertip: link origin + quat_apply(link_quat, local_offset)
    link_pos = robot.data.body_pos_w[:, ft_body_ids_t, :]             # (1, 5, 3)
    link_quat = robot.data.body_quat_w[:, ft_body_ids_t, :]           # (1, 5, 4)
    offsets = ft_offsets_t.unsqueeze(0).expand(1, -1, -1)             # (1, 5, 3)
    rotated = quat_apply(link_quat.reshape(-1, 4), offsets.reshape(-1, 3)).reshape(1, 5, 3)
    ft_pos = link_pos + rotated                                       # (1, 5, 3)
    ft_z = ft_pos[..., 2]                                             # (1, 5)
    all_z = torch.cat([non_ft_z, ft_z], dim=-1)                       # (1, 21)
    return float(all_z.min().item())


def _run_ik_with_kpt_constraint(
    robot: Articulation,
    ik_ctrl: DifferentialIKController,
    arm_joint_ids: list[int],
    arm_ids_t: torch.Tensor,
    jac_body_idx: int,
    wrist_body_id: int,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    num_iter: int,
    sim: SimulationContext,
    table_height: float,
    kpt_body_ids_t: torch.Tensor,
    ft_body_ids_t: torch.Tensor,
    ft_offsets_t: torch.Tensor,
    max_lift_iter: int,
    lift_margin: float,
) -> tuple[torch.Tensor, float, float, float]:
    """DLS IK + outer loop: lift wrist target z until all 21 hand keypoints have z >= table_height.

    Returns (arm_joint_angles, pos_err, rot_err, total_lift_amount).
    """
    target_pos = target_pos.clone()
    orig_z = float(target_pos[0, 2].item())

    arm_jp = None
    pos_err = float("nan")
    rot_err = float("nan")
    for outer in range(max_lift_iter):
        arm_jp, pos_err, rot_err = _run_ik(
            robot, ik_ctrl, arm_joint_ids, arm_ids_t, jac_body_idx,
            wrist_body_id, target_pos, target_quat, num_iter, sim,
        )
        min_z = _compute_kpts_min_z(robot, kpt_body_ids_t, ft_body_ids_t, ft_offsets_t)
        if min_z >= table_height - 1e-4:
            break
        deficit = table_height - min_z + lift_margin
        target_pos = target_pos.clone()
        target_pos[0, 2] += deficit

    lift = float(target_pos[0, 2].item()) - orig_z
    return arm_jp, pos_err, rot_err, lift


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

    # ── Resolve fingertip + non-fingertip MANO keypoint body IDs (mirrors env code) ──
    fingertip_names = ["finger_r_link4", "finger_r_link8", "finger_r_link12",
                       "finger_r_link16", "finger_r_link20"]
    ft_body_ids = []
    for name in fingertip_names:
        found, _ = robot.find_bodies(name)
        if not found:
            raise RuntimeError(f"Fingertip body '{name}' not found in robot.")
        ft_body_ids.append(found[0])
    ft_body_ids_t = torch.tensor(ft_body_ids, dtype=torch.long, device=device)
    ft_offsets_t = torch.tensor(
        [_FINGERTIP_OFFSETS[n] for n in fingertip_names],
        dtype=torch.float32, device=device,
    )  # (5, 3)

    kpt_body_ids = []
    for _, body_name in _MANO_NON_FT_BODY_NAMES:
        found, _ = robot.find_bodies(body_name)
        if not found:
            raise RuntimeError(f"MANO body '{body_name}' not found in robot.")
        kpt_body_ids.append(found[0])
    kpt_body_ids_t = torch.tensor(kpt_body_ids, dtype=torch.long, device=device)

    table_height = float(_TABLE_SIZE[2])

    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )
    ik_ctrl = DifferentialIKController(ik_cfg, num_envs=1, device=device)

    # Collect trajectory paths
    _dataset_dir = _HOCAP_DIR if args_cli.dataset == "hocap" else _OAKINK_DIR
    mano_dir = _dataset_dir / "mano" / "right"
    if args_cli.task:
        task_dirs = [mano_dir / args_cli.task]
    else:
        task_dirs = sorted(
            d for d in mano_dir.iterdir()
            if d.is_dir() and (not args_cli.object_id or args_cli.object_id in d.name)
        )

    # Collect all (data_dir, traj_path) pairs upfront so we can show [i/N] progress.
    targets: list[tuple[Path, Path]] = []
    for task_dir in task_dirs:
        if args_cli.data_id >= 0:
            data_dirs = [task_dir / str(args_cli.data_id)]
        else:
            data_dirs = sorted(d for d in task_dir.iterdir() if d.is_dir())
        for data_dir in data_dirs:
            traj_path = data_dir / "trajectory_keypoints.npz"
            if traj_path.exists():
                targets.append((data_dir, traj_path))
    total = len(targets)
    print(f"\nProcessing {total} trajectory directories...\n")

    processed = skipped = errors = 0
    for i, (data_dir, traj_path) in enumerate(targets, start=1):
        prefix = f"[{i}/{total}]"

        out_path = data_dir / "frame0_arm_joint_pos.npy"
        if out_path.exists() and not args_cli.overwrite:
            print(f"{prefix} skip — {out_path.relative_to(_dataset_dir)} already exists")
            skipped += 1
            continue

        result = _canonicalize_frame0(traj_path, _dataset_dir)
        if result is None:
            print(f"{prefix} error — canonicalize failed for {traj_path.relative_to(_dataset_dir)}")
            errors += 1
            continue
        wrist_pos_np, wrist_quat_np = result

        target_pos = torch.tensor(wrist_pos_np, dtype=torch.float32, device=device).unsqueeze(0)
        target_quat = torch.tensor(wrist_quat_np, dtype=torch.float32, device=device).unsqueeze(0)

        arm_jp, pos_err, rot_err, lift = _run_ik_with_kpt_constraint(
            robot, ik_ctrl, arm_joint_ids, arm_ids_t, jac_body_idx,
            wrist_body_id, target_pos, target_quat, args_cli.num_iter, sim,
            table_height=table_height,
            kpt_body_ids_t=kpt_body_ids_t,
            ft_body_ids_t=ft_body_ids_t,
            ft_offsets_t=ft_offsets_t,
            max_lift_iter=args_cli.max_lift_iter,
            lift_margin=args_cli.lift_margin,
        )

        np.save(str(out_path), arm_jp.cpu().numpy())
        rel = traj_path.parent.relative_to(_dataset_dir)
        print(f"{prefix} {rel}  pos_err={pos_err:.4f}m  rot_err={np.degrees(rot_err):.2f}°  "
              f"lift={lift*100:.2f}cm  → {out_path.name}")
        processed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped, {errors} errors.")
    import os; os._exit(0)


if __name__ == "__main__":
    main()
