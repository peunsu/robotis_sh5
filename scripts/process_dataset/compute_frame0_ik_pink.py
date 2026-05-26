"""Precompute arm IK for trajectory frame 0 — pink (QP-based) variant.

Alternative to `compute_frame0_ik.py` (Isaac Lab DLS). Uses Stéphane Caron's pink
library (https://github.com/stephane-caron/pink) on top of pinocchio for differential
IK with proper joint-limit handling via QP.

Why pink:
  - Joint limits as inequality constraints (DLS only clamps post-hoc).
  - Multi-task formulation: FrameTask (wrist) + PostureTask (regularize toward default).
    The posture task suppresses null-space wandering of the 7-DOF arm.
  - No Isaac sim spawn → faster startup (~ms vs ~30s).

Result file format matches DLS variant: `frame0_arm_joint_pos.npy` shape (7,).

Run:
    python scripts/process_dataset/compute_frame0_ik_pink.py
    python scripts/process_dataset/compute_frame0_ik_pink.py --dataset hocap --overwrite

Notes:
  - Coordinates: canonicalization is done in env-local frame; we transform the resulting
    wrist target into the robot base_link frame (which is what pinocchio uses) using the
    robot's spawn pose from `FFW_SH5_DEX_CFG.init_state`.
  - All non-arm_r joints are locked at the env's default joint pose via `buildReducedModel`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import pink
import trimesh
from pink.tasks import FrameTask, PostureTask


# ────────────────────────────────────────────────────────────────────────────
# Paths and constants (must mirror env_cfg defaults)
# ────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parents[2]
_ROBOT_DIR = _SCRIPT_DIR / "source" / "robotis_sh5" / "data" / "robots" / "FFW"
_URDF_PATH = _ROBOT_DIR / "urdf" / "ffw_sh5_follower_copy.urdf"
_MESH_DIR = _ROBOT_DIR / "meshes"

_DATA_DIR = _SCRIPT_DIR / "source" / "robotis_sh5" / "data"
_OAKINK_DIR = _DATA_DIR / "processed" / "oakink"
_HOCAP_DIR = _DATA_DIR / "processed" / "hocap"

# Must match env_cfg defaults (and DLS variant constants)
_TABLE_POS = (0.3, 0.0, 0.0)
_TABLE_SIZE = (0.6, 0.6, 1.0)
_ROBOT_POS_ENV = np.array([0.65, 0.65, 0.0], dtype=np.float64)
# wxyz quat for spawn rotation: −90° about Z
_ROBOT_QUAT_ENV_WXYZ = np.array([0.70711, 0.0, 0.0, -0.70711], dtype=np.float64)

_WRIST_FRAME = "hx5_d20_right_base"
_ARM_R_JOINT_NAMES = [f"arm_r_joint{i}" for i in range(1, 8)]

# Default joint positions (env's pre-grasp pose). Joints not listed → 0.
_DEFAULT_JOINT_POS: dict[str, float] = {
    "arm_r_joint1": 0.0,
    "arm_r_joint2": -1.162,
    "arm_r_joint3": 0.291,
    "arm_r_joint4": -1.876,
    "arm_r_joint5": -0.609,
    "arm_r_joint6": 0.335,
    "arm_r_joint7": -0.368,
    "lift_joint": 0.0,
}

# Fingertip and MANO non-fingertip body names — kept identical to env code so the
# table-height constraint check uses the same 21 keypoints.
_FINGERTIP_OFFSETS: dict[str, list[float]] = {
    "finger_r_link4":  [0.0,   0.03975, 0.012],
    "finger_r_link8":  [0.012, 0.0,     0.02425],
    "finger_r_link12": [0.012, 0.0,     0.02425],
    "finger_r_link16": [0.012, 0.0,     0.02425],
    "finger_r_link20": [0.012, 0.0,     0.02425],
}
_FINGERTIP_NAMES = list(_FINGERTIP_OFFSETS.keys())
_MANO_NON_FT_BODY_NAMES_ORDERED = [
    "hx5_d20_right_base",
    "finger_r_link2", "finger_r_link6", "finger_r_link10",
    "finger_r_link14", "finger_r_link18",
    "finger_r_link3", "finger_r_link7", "finger_r_link11",
    "finger_r_link15", "finger_r_link19",
    "finger_r_link4", "finger_r_link8", "finger_r_link12",
    "finger_r_link16", "finger_r_link20",
]


# ────────────────────────────────────────────────────────────────────────────
# Quaternion / SE3 helpers
# ────────────────────────────────────────────────────────────────────────────

def _quat_wxyz_to_R(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _quat_wxyz_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse (conjugate) of unit quaternion wxyz."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_wxyz_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions: q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def _quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by wxyz quaternion q."""
    return _quat_wxyz_to_R(q) @ v


# Robot spawn-pose constants
_R_BASE_ENV = _quat_wxyz_to_R(_ROBOT_QUAT_ENV_WXYZ)     # base frame expressed in env frame
_R_ENV_BASE = _R_BASE_ENV.T                            # env frame expressed in base frame
_Q_ENV_BASE = _quat_wxyz_inverse(_ROBOT_QUAT_ENV_WXYZ) # rotation env→base as quat


def _env_pose_to_base(pos_env: np.ndarray, quat_env_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform a pose from env-local frame to robot base_link frame."""
    pos_base = _R_ENV_BASE @ (pos_env - _ROBOT_POS_ENV)
    quat_base = _quat_wxyz_mul(_Q_ENV_BASE, quat_env_wxyz)
    return pos_base, quat_base


def _base_pos_to_env(pos_base: np.ndarray) -> np.ndarray:
    """Transform a position from base_link frame back to env-local frame."""
    return _R_BASE_ENV @ pos_base + _ROBOT_POS_ENV


# ────────────────────────────────────────────────────────────────────────────
# Trajectory canonicalization (mirrors env / DLS variant)
# ────────────────────────────────────────────────────────────────────────────

def _canonicalize_frame0(traj_path: Path, dataset_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return frame-0 wrist (pos, quat-wxyz) in ENV-LOCAL coords after table-anchor +
    canonical Z-rotation. Identical math to compute_frame0_ik.py's version."""
    data = np.load(str(traj_path))
    wp = data["qpos_wrist_right"][:, :3].astype(np.float32)
    wq = data["qpos_wrist_right"][:, 3:].astype(np.float32)   # wxyz
    op = data["qpos_obj_right"][:, :3].astype(np.float32)

    # Mesh path (object id from task_info.json or dash-split fallback)
    task_dir = traj_path.parent.parent
    info_path = task_dir / "task_info.json"
    object_id = None
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            mesh_dir_rel = info.get("right_object_mesh_dir", "")
            if mesh_dir_rel:
                object_id = Path(mesh_dir_rel).name
        except Exception:
            object_id = None
    if object_id is None:
        object_id = task_dir.name.split("-")[0]

    oq0 = data["qpos_obj_right"][0, 3:].astype(np.float32)
    mesh_path = dataset_dir / "assets" / "objects" / object_id / "visual.obj"
    if mesh_path.exists():
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        verts = np.array(mesh.vertices, dtype=np.float32)
        R = _quat_wxyz_to_R(oq0.astype(np.float64)).astype(np.float32)
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

    # XY canonicalization (rotate around object frame-0 xy so wrist→object dir matches robot-side dir)
    tx, ty = float(_TABLE_POS[0]), float(_TABLE_POS[1])
    hx, hy = float(_TABLE_SIZE[0]) / 2.0, float(_TABLE_SIZE[1]) / 2.0
    dx = float(_ROBOT_POS_ENV[0]) - tx
    dy = float(_ROBOT_POS_ENV[1]) - ty
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

    wp_rot = wp.copy()
    wp_rot[:, 0] -= ox; wp_rot[:, 1] -= oy
    wp_rot[:, :2] = wp_rot[:, :2] @ R2.T
    wp_rot[:, 0] += ox; wp_rot[:, 1] += oy

    hw = angle / 2.0
    qw_r, qz_r = float(np.cos(hw)), float(np.sin(hw))
    w2, x2, y2, z2 = wq[:, 0], wq[:, 1], wq[:, 2], wq[:, 3]
    wq_rot = np.stack([
        qw_r * w2 - qz_r * z2,
        qw_r * x2 - qz_r * y2,
        qw_r * y2 + qz_r * x2,
        qw_r * z2 + qz_r * w2,
    ], axis=-1).astype(np.float32)

    return wp_rot[0].astype(np.float64), wq_rot[0].astype(np.float64)


# ────────────────────────────────────────────────────────────────────────────
# Robot model setup (load URDF + lock non-arm joints)
# ────────────────────────────────────────────────────────────────────────────

def _build_arm_only_model() -> tuple[pin.Model, np.ndarray, list[int], list[int], dict[str, int]]:
    """Load URDF and reduce to an arm-only model (all other joints locked at defaults).

    Returns:
        model:           reduced pinocchio Model (only arm_r_joint1..7 movable)
        q_default:       default joint config of the FULL model (used for locking)
        arm_q_idx:       indices in REDUCED model's q vector for arm_r_joint1..7 (in order)
        arm_v_idx:       indices in REDUCED model's v vector for arm_r_joint1..7
        body_frame_ids:  reduced-model frame ids for {wrist + fingertips + non-fingertip kpts}
    """
    # Only the kinematic model is needed for IK; geometry (collision/visual meshes)
    # is not used and skipping it avoids mesh-path resolution issues.
    full_model = pin.buildModelFromUrdf(str(_URDF_PATH))

    # Build default q for full model
    q_default = pin.neutral(full_model)
    for joint_name, joint_val in _DEFAULT_JOINT_POS.items():
        if not full_model.existJointName(joint_name):
            continue
        jid = full_model.getJointId(joint_name)
        q_default[full_model.joints[jid].idx_q] = joint_val

    # Joints to lock = all except arm_r_joint1..7
    joints_to_lock = []
    for jid in range(1, full_model.njoints):   # skip universe (id 0)
        jname = full_model.names[jid]
        if jname not in _ARM_R_JOINT_NAMES:
            joints_to_lock.append(jid)

    reduced_model = pin.buildReducedModel(full_model, joints_to_lock, q_default)

    # Reduced model has only arm_r joints. Find their idx_q.
    arm_q_idx = []
    arm_v_idx = []
    for jn in _ARM_R_JOINT_NAMES:
        jid = reduced_model.getJointId(jn)
        arm_q_idx.append(reduced_model.joints[jid].idx_q)
        arm_v_idx.append(reduced_model.joints[jid].idx_v)

    return reduced_model, q_default, arm_q_idx, arm_v_idx


def _resolve_frame_ids(model: pin.Model) -> dict[str, int]:
    """Map our body names to pinocchio frame ids on the reduced model."""
    name_to_id: dict[str, int] = {}
    needed = [_WRIST_FRAME] + _FINGERTIP_NAMES + _MANO_NON_FT_BODY_NAMES_ORDERED
    for name in needed:
        if not model.existFrame(name):
            raise RuntimeError(f"Frame '{name}' not found in reduced URDF model.")
        name_to_id[name] = model.getFrameId(name)
    return name_to_id


# ────────────────────────────────────────────────────────────────────────────
# Keypoint min-z (in env frame) — for table-height constraint
# ────────────────────────────────────────────────────────────────────────────

def _compute_kpts_min_z_env(
    model: pin.Model,
    data: pin.Data,
    frame_ids: dict[str, int],
    q: np.ndarray,
) -> float:
    """Compute min z over all 21 MANO keypoints in ENV frame after IK update.

    Non-fingertip kpts: link origins (frame placement translation).
    Fingertip kpts: link origin + rotated local offset (matches env's _compute_fingertip_positions).
    Then transform each from base frame to env frame and return min z.
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    z_values: list[float] = []
    # Non-fingertip (uses link origin = frame translation)
    for name in _MANO_NON_FT_BODY_NAMES_ORDERED:
        T = data.oMf[frame_ids[name]]
        pos_env = _base_pos_to_env(T.translation)
        z_values.append(float(pos_env[2]))
    # Fingertip (+ local offset rotated by link orientation)
    for name in _FINGERTIP_NAMES:
        T = data.oMf[frame_ids[name]]
        off_local = np.asarray(_FINGERTIP_OFFSETS[name], dtype=np.float64)
        tip_base = T.translation + T.rotation @ off_local
        pos_env = _base_pos_to_env(tip_base)
        z_values.append(float(pos_env[2]))
    return min(z_values)


def _compute_wrist_pose_base(
    model: pin.Model,
    data: pin.Data,
    wrist_frame_id: int,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return wrist (pos, quat-wxyz) in base_link frame."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T = data.oMf[wrist_frame_id]
    pos = np.array(T.translation, dtype=np.float64)
    # Convert R to wxyz quaternion (pinocchio uses xyzw via Quaternion, take care)
    quat_xyzw = pin.Quaternion(T.rotation).coeffs()  # (x, y, z, w)
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
    return pos, quat_wxyz


# ────────────────────────────────────────────────────────────────────────────
# IK with table-height constraint (lift loop)
# ────────────────────────────────────────────────────────────────────────────

def _solve_ik_with_kpt_constraint(
    model: pin.Model,
    data: pin.Data,
    frame_ids: dict[str, int],
    arm_q_idx: list[int],
    target_pos_env: np.ndarray,
    target_quat_env_wxyz: np.ndarray,
    table_height: float,
    num_iter: int,
    max_lift_iter: int,
    lift_margin: float,
    dt: float = 1.0 / 60.0,
    solver: str = "quadprog",
) -> tuple[np.ndarray, float, float, float]:
    """Solve IK with pink (FrameTask + PostureTask) + outer lift loop.

    Returns:
        arm_q:    (7,) converged arm_r_joint angles
        pos_err:  wrist position error (m)
        rot_err:  wrist rotation error (rad)
        lift:     total z added to wrist target (m)
    """
    # Pinocchio uses unit quaternions internally; convert env target to base frame.
    target_pos_env = np.asarray(target_pos_env, dtype=np.float64).copy()
    orig_z_env = float(target_pos_env[2])

    arm_q_final = None
    pos_err_final = float("nan")
    rot_err_final = float("nan")

    for outer in range(max_lift_iter):
        # Transform current (possibly lifted) target into base frame
        target_pos_base, target_quat_base = _env_pose_to_base(target_pos_env, target_quat_env_wxyz)
        R_target = _quat_wxyz_to_R(target_quat_base)
        target_SE3 = pin.SE3(R_target, target_pos_base)

        # Reset config to default (q0 = neutral for reduced model — arm_r joints at our defaults,
        # but reduced model neutral may differ. Use explicit defaults below.)
        q0 = pin.neutral(model)
        for i, jn in enumerate(_ARM_R_JOINT_NAMES):
            q0[arm_q_idx[i]] = _DEFAULT_JOINT_POS[jn]

        config = pink.Configuration(model, data, q0)

        ee_task = FrameTask(_WRIST_FRAME, position_cost=1.0, orientation_cost=1.0)
        ee_task.set_target(target_SE3)
        posture_task = PostureTask(cost=1e-3)
        posture_task.set_target_from_configuration(config)
        tasks = [ee_task, posture_task]

        for _ in range(num_iter):
            try:
                velocity = pink.solve_ik(config, tasks, dt, solver=solver)
            except Exception as e:
                # Fall back: pad with zero velocity (effectively pause IK this step)
                velocity = np.zeros(model.nv)
                if "no solution" in str(e).lower():
                    break
            q_new = config.integrate(velocity, dt)
            config.update(q_new)

        # Evaluate convergence + check keypoint constraint
        pos_w, quat_w_wxyz = _compute_wrist_pose_base(model, data, frame_ids[_WRIST_FRAME], config.q)
        pos_err = float(np.linalg.norm(pos_w - target_pos_base))
        dot = float(np.clip(np.abs(np.dot(quat_w_wxyz, target_quat_base)), 0.0, 1.0))
        rot_err = float(2.0 * np.arccos(dot))

        arm_q_final = np.array([config.q[arm_q_idx[i]] for i in range(7)], dtype=np.float64)
        pos_err_final = pos_err
        rot_err_final = rot_err

        min_z_env = _compute_kpts_min_z_env(model, data, frame_ids, config.q)
        if min_z_env >= table_height - 1e-4:
            break

        deficit = table_height - min_z_env + lift_margin
        target_pos_env[2] += deficit

    lift = float(target_pos_env[2]) - orig_z_env
    return arm_q_final, pos_err_final, rot_err_final, lift


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute arm IK (pink variant).")
    parser.add_argument("--dataset", type=str, default="oakink", choices=["oakink", "hocap"])
    parser.add_argument("--object_id", type=str, default="")
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--data_id", type=int, default=-1)
    parser.add_argument("--num_iter", type=int, default=200, help="Pink IK iterations per outer attempt.")
    parser.add_argument("--max_lift_iter", type=int, default=10)
    parser.add_argument("--lift_margin", type=float, default=0.005)
    parser.add_argument("--solver", type=str, default="quadprog",
                        help="QP solver backend (quadprog, daqp, osqp, ...).")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Build reduced model once
    print(f"\nLoading URDF: {_URDF_PATH}")
    model, _, arm_q_idx, _ = _build_arm_only_model()
    data = model.createData()
    frame_ids = _resolve_frame_ids(model)
    print(f"  reduced model nq={model.nq}, nv={model.nv}, njoints={model.njoints}")
    print(f"  arm_r_joint idx_q: {arm_q_idx}")
    print(f"  wrist frame_id  : {frame_ids[_WRIST_FRAME]}")

    table_height = float(_TABLE_SIZE[2])

    # Collect targets
    dataset_dir = _HOCAP_DIR if args.dataset == "hocap" else _OAKINK_DIR
    mano_dir = dataset_dir / "mano" / "right"
    if args.task:
        task_dirs = [mano_dir / args.task]
    else:
        task_dirs = sorted(
            d for d in mano_dir.iterdir()
            if d.is_dir() and (not args.object_id or args.object_id in d.name)
        )

    targets: list[tuple[Path, Path]] = []
    for task_dir in task_dirs:
        if args.data_id >= 0:
            data_dirs = [task_dir / str(args.data_id)]
        else:
            data_dirs = sorted(d for d in task_dir.iterdir() if d.is_dir())
        for d in data_dirs:
            t = d / "trajectory_keypoints.npz"
            if t.exists():
                targets.append((d, t))
    total = len(targets)
    print(f"\nProcessing {total} trajectories...\n")

    processed = skipped = errors = 0
    for i, (out_dir, traj_path) in enumerate(targets, start=1):
        prefix = f"[{i}/{total}]"
        out_path = out_dir / "frame0_arm_joint_pos.npy"

        if out_path.exists() and not args.overwrite:
            print(f"{prefix} skip — {out_path.relative_to(dataset_dir)} already exists")
            skipped += 1
            continue

        result = _canonicalize_frame0(traj_path, dataset_dir)
        if result is None:
            print(f"{prefix} error — canonicalize failed for {traj_path.relative_to(dataset_dir)}")
            errors += 1
            continue
        wrist_pos_env, wrist_quat_env_wxyz = result

        arm_q, pos_err, rot_err, lift = _solve_ik_with_kpt_constraint(
            model, data, frame_ids, arm_q_idx,
            wrist_pos_env, wrist_quat_env_wxyz,
            table_height=table_height,
            num_iter=args.num_iter,
            max_lift_iter=args.max_lift_iter,
            lift_margin=args.lift_margin,
            dt=args.dt,
            solver=args.solver,
        )

        np.save(str(out_path), arm_q.astype(np.float32))
        rel = traj_path.parent.relative_to(dataset_dir)
        print(f"{prefix} {rel}  pos_err={pos_err:.4f}m  rot_err={np.degrees(rot_err):.2f}°  "
              f"lift={lift*100:.2f}cm  → {out_path.name}")
        processed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped, {errors} errors.")


if __name__ == "__main__":
    main()
