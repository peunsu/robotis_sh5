"""Unified per-trajectory arm-reference pipeline.

For each trajectory under `<dataset>/mano/right/<task>/<data_id>/`:
  1. Canonicalize wrist + object trajectories (table-anchor + canonical Z-rotation).
  2. VPoser-IK (batched, all N frames) → SMPL elbow positions in env frame.
  3. Pink IK per-frame with warm-start (wrist FrameTask + elbow FrameTask soft prior +
     21 MANO keypoint barriers for z ≥ table_height) → robot arm_r joint angles.
     FK after IK convergence also extracts arm_r_link7 origin position.
  4. Render an mp4 visualization (SMPL mesh + skeleton + robot arm chain per frame).

Outputs written to each trajectory directory:
    arm_keypoints.npz   {elbow_pos: (N,3), link7_pos: (N,3)} float32
                        — Arm-side reference keypoints in RAW frame (same convention
                          as `mano_kpts_right`). `elbow_pos` is from SMPL/VPoser fit;
                          `link7_pos` is the robot's arm_r_link7 origin from IK FK
                          (the last revolute link before the FIXED wrist mount —
                          tracking it adds a positional constraint that indirectly
                          constrains wrist orientation along its local Z axis).
    arm_joint_pos.npy   shape (N, 7) float32  — robot arm_r joint angles per frame
    vposer_ik_video.mp4                        — visualization

Run:
    /home/peunsu/anaconda3/envs/env_isaaclab/bin/python scripts/process_dataset/process_arm_pipeline.py \\
        --dataset hocap --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pinocchio as pin
import pink
import pyrender
import smplx
import torch
import trimesh
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from pink.barriers import PositionBarrier
from pink.tasks import FrameTask, PostureTask
from smplx.lbs import batch_rodrigues

_SCRIPT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPT_DIR / "scripts" / "process_dataset"))

# Reuse stable helpers (transforms + reduced model builder)
from compute_frame0_ik_pink import (
    _env_pose_to_base, _quat_wxyz_to_R,
    _build_arm_only_model, _resolve_frame_ids,
    _R_BASE_ENV, _ROBOT_POS_ENV, _ARM_R_JOINT_NAMES,
    _TABLE_POS, _TABLE_SIZE, _OAKINK_DIR, _HOCAP_DIR,
    _WRIST_FRAME, _FINGERTIP_OFFSETS, _FINGERTIP_NAMES,
    _KPT_21_FRAME_NAMES, _DEFAULT_JOINT_POS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Canonicalization (wrist + MANO 21 keypoints) — same transform as
# ``compute_frame0_ik_pink._canonicalize_trajectory`` extended to mano_kpts.
# ──────────────────────────────────────────────────────────────────────────────

def _canonicalize_full(
    traj_path: Path, dataset_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Canonicalize wrist + mano_kpts + object trajectories to env frame.

    Returns:
        wp_all (N, 3) — canonical wrist position
        wq_all (N, 4 wxyz) — canonical wrist quaternion
        mano_kpts (N, 21, 3) — canonical 21 MANO kpts
        offset (3,) — translation applied (canonical = raw + offset)
        angle (float) — XY rotation angle around `pivot_xy` (canonical = R(angle) on raw+offset)
        pivot_xy (2,) — rotation pivot (= canonical object frame-0 XY = table center XY)

    The returned (offset, angle, pivot_xy) lets callers invert the transform via
    ``_uncanonicalize_pos`` to recover the original raw frame.
    """
    raw = np.load(str(traj_path))
    wp = raw["qpos_wrist_right"][:, :3].astype(np.float32)
    wq = raw["qpos_wrist_right"][:, 3:].astype(np.float32)
    op = raw["qpos_obj_right"][:, :3].astype(np.float32)
    kpts = raw["mano_kpts_right"].astype(np.float32)   # (N, 21, 3)

    # Object id (mesh) for z-anchor at table top
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

    oq0 = raw["qpos_obj_right"][0, 3:].astype(np.float32)
    mesh_path = dataset_dir / "assets" / "objects" / object_id / "visual.obj"
    if mesh_path.exists():
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        verts = np.array(mesh.vertices, dtype=np.float32)
        Rm = _quat_wxyz_to_R(oq0.astype(np.float64)).astype(np.float32)
        mesh_z_min = float((verts @ Rm.T)[:, 2].min())
    else:
        mesh_z_min = 0.0

    # Translation: anchor object frame-0 at table center XY, z = table_top - mesh_z_min
    table_target = np.array(
        [_TABLE_POS[0], _TABLE_POS[1], float(_TABLE_SIZE[2]) - mesh_z_min],
        dtype=np.float32,
    )
    offset = table_target - op[0]
    op = op + offset
    wp = wp + offset
    kpts = kpts + offset                              # broadcast: (N, 21, 3) + (3,)

    # XY canonicalization angle: align frame-0 wrist→object dir with table edge nearest robot
    tx, ty = float(_TABLE_POS[0]), float(_TABLE_POS[1])
    hx, hy = float(_TABLE_SIZE[0]) / 2.0, float(_TABLE_SIZE[1]) / 2.0
    dx = float(_ROBOT_POS_ENV[0]) - tx
    dy = float(_ROBOT_POS_ENV[1]) - ty
    if abs(dy) >= abs(dx):
        ref_xy = np.array([tx, ty + hy * np.sign(dy)], dtype=np.float32)
    else:
        ref_xy = np.array([tx + hx * np.sign(dx), ty], dtype=np.float32)
    canonical_dir = ref_xy - np.array([tx, ty], dtype=np.float32)
    # Default pivot = table center (matches env convention); angle 0 if direction degenerate.
    pivot_xy = np.array([tx, ty], dtype=np.float64)
    if np.linalg.norm(canonical_dir) < 1e-6:
        return (wp.astype(np.float64), wq.astype(np.float64), kpts.astype(np.float64),
                offset.astype(np.float64), 0.0, pivot_xy)
    canonical_dir /= np.linalg.norm(canonical_dir)

    o0 = op[0]
    wrist_dir = wp[0, :2] - o0[:2]
    if np.linalg.norm(wrist_dir) < 1e-4:
        return (wp.astype(np.float64), wq.astype(np.float64), kpts.astype(np.float64),
                offset.astype(np.float64), 0.0, pivot_xy)
    wrist_dir /= np.linalg.norm(wrist_dir)
    cos_a = float(np.clip(np.dot(wrist_dir, canonical_dir), -1.0, 1.0))
    sin_a = float(wrist_dir[0] * canonical_dir[1] - wrist_dir[1] * canonical_dir[0])
    angle = float(np.arctan2(sin_a, cos_a))
    c, s = float(np.cos(angle)), float(np.sin(angle))
    R2 = np.array([[c, -s], [s, c]], dtype=np.float32)
    ox, oy = float(o0[0]), float(o0[1])
    pivot_xy = np.array([ox, oy], dtype=np.float64)

    # Rotate XY of wrist position
    wp_rot = wp.copy()
    wp_rot[:, 0] -= ox; wp_rot[:, 1] -= oy
    wp_rot[:, :2] = wp_rot[:, :2] @ R2.T
    wp_rot[:, 0] += ox; wp_rot[:, 1] += oy

    # Rotate XY of mano_kpts (all 21 points × all N frames)
    kpts_rot = kpts.copy()
    kpts_rot[..., 0] -= ox; kpts_rot[..., 1] -= oy
    kpts_rot[..., :2] = kpts_rot[..., :2] @ R2.T
    kpts_rot[..., 0] += ox; kpts_rot[..., 1] += oy

    # Rotate wrist quaternion around Z by angle
    hw = angle / 2.0
    qw_r, qz_r = float(np.cos(hw)), float(np.sin(hw))
    w2, x2, y2, z2 = wq[:, 0], wq[:, 1], wq[:, 2], wq[:, 3]
    wq_rot = np.stack([
        qw_r * w2 - qz_r * z2,
        qw_r * x2 - qz_r * y2,
        qw_r * y2 + qz_r * x2,
        qw_r * z2 + qz_r * w2,
    ], axis=-1).astype(np.float32)

    return (wp_rot.astype(np.float64), wq_rot.astype(np.float64), kpts_rot.astype(np.float64),
            offset.astype(np.float64), angle, pivot_xy)


def _uncanonicalize_pos(p_canon: np.ndarray, offset: np.ndarray, angle: float,
                        pivot_xy: np.ndarray) -> np.ndarray:
    """Inverse of canonicalization for positions.

    Canonicalization applied (a) translation by ``+offset`` then (b) XY rotation by
    ``+angle`` around ``pivot_xy``. This function reverses (b) then (a):
        p_canon  →  rotate XY by −angle around pivot_xy  →  subtract offset  →  p_raw
    """
    p = np.asarray(p_canon, dtype=np.float64).copy()
    c, s = float(np.cos(-angle)), float(np.sin(-angle))
    R_inv = np.array([[c, -s], [s, c]], dtype=np.float64)
    p_xy = p[..., :2] - pivot_xy
    p[..., :2] = p_xy @ R_inv.T + pivot_xy
    p -= offset
    return p

# ──────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────────────────────────────────────

_SMPLX_MODEL_DIR = "/home/peunsu/workspace/human_body_prior/support_data/dowloads/models/"
_VPOSER_CKPT = "/home/peunsu/workspace/human_body_prior/support_data/dowloads/V02_05"
_BETA_PATH = _SCRIPT_DIR / "source" / "robotis_sh5" / "data" / "smpl_fit" / "robot_beta.npy"

_URDF_PATH = _SCRIPT_DIR / "source" / "robotis_sh5" / "data" / "robots" / "FFW" / "urdf" / "ffw_sh5_follower_copy.urdf"

# Robot landmark frames
_ROBOT_SHOULDER_FRAME = "arm_r_link1"
_ROBOT_ELBOW_FRAME    = "arm_r_link4"
_ROBOT_WRIST_FRAME    = "hx5_d20_right_base"   # SH5 default

# Per-robot wrist mount transform (from arm_r_link7 to the wrist body the IK targets).
#   SH5: hx5_d20_right_base at (0, 0, -0.078) with rpy=[0, π, -π/2] from arm_r_link7
#   Shadow: robot0_palm at (0, -0.010, -0.112) with R = 180° around X axis (composed
#           through hx5_d20_right_joint → robot0_WRJ0 in FFW_SH5_shadow_flat.usd)
_WRIST_MOUNT_TRANSFORMS = {
    "sh5": {
        "frame_name": "hx5_d20_right_base",   # already exists in URDF
        # offset/quat used when registering a virtual frame (unused for sh5 — frame
        # already exists in URDF; kept for code symmetry).
        "offset_in_link7": np.array([0.0, 0.0, -0.078], dtype=np.float64),
        # q wxyz of R(rpy=[0, π, -π/2])
        "quat_in_link7_wxyz": np.array([0.0, 0.7071068, 0.7071068, 0.0], dtype=np.float64),
    },
    "shadow": {
        "frame_name": "robot0_palm_virtual",   # added at IK setup time
        # Position: where robot0_palm body sits relative to arm_r_link7 (from
        # FFW_SH5_shadow_flat.usd composed chain arm_r_link7 → robot0_wrist → robot0_palm).
        "offset_in_link7": np.array([0.0, -0.010, -0.112], dtype=np.float64),
        # Rotation: composition R_link7_to_palm × R_palm_to_landmark.
        # R_link7_to_palm = 180° around X (USD-extracted).
        # R_palm_to_landmark = static rotation aligning robot0_palm body axes with
        # the HOcap landmark frame (computed via extract_wrist_rotation on Shadow
        # Hand knuckle positions at neutral pose; ≈ +90° around palm Z with a small
        # ~5° tilt from middle finger MCP offset).
        # Resulting matrix is very close to SH5's R(rpy=[0, π, -π/2]) so the IK
        # behaves analogously to the sh5 wrist task on robot0_palm.
        "quat_in_link7_wxyz": np.array(
            [0.039103, 0.706025, 0.706025, 0.039103], dtype=np.float64
        ),
    },
}


# Shadow Hand 21 MANO keypoint SE3 placements relative to arm_r_link7 (at default
# pose: arm_r_joint7's frame, all Shadow Hand joints at 0). Pre-computed offline by
# composing USD joint chains from FFW_SH5_shadow_flat.usd. These are registered as
# OP_FRAMEs in the Pinocchio model so the IK barrier (z-floor) and any kpt-tracking
# constraints use the ACTUAL Shadow Hand body locations — not the SH5 finger phantoms
# that would otherwise come from the URDF.
#
# Ordering matches MANO 21-keypoint indices:
#   0  = wrist (palm)
#   1-4 = thumb (CMC, MCP, DIP, TIP)
#   5-8 = index (MCP, PIP, DIP, TIP)
#   9-12 = middle (MCP, PIP, DIP, TIP)
#   13-16 = ring (MCP, PIP, DIP, TIP)
#   17-20 = pinky (MCP, PIP, DIP, TIP)
_SHADOW_KPT_PLACEMENTS: list[tuple[str, np.ndarray, np.ndarray]] = [
    # name, position (3,), quat_wxyz (4,)
    ("robot0_palm",         np.array([0.0, -0.010001, -0.112]),  np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_thbase",       np.array([0.034, -0.001001, -0.141]),  np.array([0.0, 0.923956, 0.0, 0.382499])),
    ("robot0_thmiddle",     np.array([0.060859, -0.001001, -0.167881]), np.array([0.0, 0.923956, 0.0, 0.382499])),
    ("robot0_thdistal",     np.array([0.083478, -0.001001, -0.190517]), np.array([0.0, 0.923956, 0.0, 0.382499])),
    ("robot0_thdistal_tip", np.array([0.091602, -0.001001, -0.210673]), np.array([1.0, 0.0, 0.0, 0.0])),
    ("robot0_ffknuckle",    np.array([0.033, -0.010001, -0.207]),  np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_ffmiddle",     np.array([0.033, -0.010002, -0.252]),  np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_ffdistal",     np.array([0.033, -0.010002, -0.277]),  np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_ffdistal_tip", np.array([0.033, -0.004002, -0.2945]), np.array([1.0, 0.0, 0.0, 0.0])),
    ("robot0_mfknuckle",    np.array([0.011, -0.010001, -0.211]),  np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_mfmiddle",     np.array([0.011, -0.010002, -0.256]),  np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_mfdistal",     np.array([0.011, -0.010002, -0.281]),  np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_mfdistal_tip", np.array([0.011, -0.004002, -0.2985]), np.array([1.0, 0.0, 0.0, 0.0])),
    ("robot0_rfknuckle",    np.array([-0.011, -0.010001, -0.207]), np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_rfmiddle",     np.array([-0.011, -0.010002, -0.252]), np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_rfdistal",     np.array([-0.011, -0.010002, -0.277]), np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_rfdistal_tip", np.array([-0.011, -0.004002, -0.2945]),np.array([1.0, 0.0, 0.0, 0.0])),
    ("robot0_lfknuckle",    np.array([-0.034, -0.010001, -0.2]),   np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_lfmiddle",     np.array([-0.034, -0.010002, -0.245]), np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_lfdistal",     np.array([-0.034, -0.010002, -0.27]),  np.array([0.0, 1.0, 0.0, 0.0])),
    ("robot0_lfdistal_tip", np.array([-0.034, -0.004002, -0.2875]),np.array([1.0, 0.0, 0.0, 0.0])),
]
# Ordered list of 21 kpt frame names (used as barrier targets in shadow IK).
_SHADOW_KPT_FRAME_NAMES: list[str] = [name for name, _, _ in _SHADOW_KPT_PLACEMENTS]

# URDF fixed-joint transform from wrist (`hx5_d20_right_base`) to its parent `arm_r_link7`.
# In wrist's LOCAL frame, the arm_r_link7 origin is at [0, 0, -0.078] (derived from
# `-R(rpy=[0,π,-π/2])^T @ [0,0,-0.078]`, which happens to equal the input by the symmetric
# form of R^T). This lets us compute the link7 reference position analytically from any
# wrist pose (position + quaternion) — no IK required.
_LINK7_OFFSET_IN_WRIST_LOCAL = np.array([0.0, 0.0, -0.078], dtype=np.float64)


def _compute_link7_from_wrist(
    wrist_pos: np.ndarray,         # (N, 3) wrist position in env frame
    wrist_quat_wxyz: np.ndarray,   # (N, 4) wrist orientation (wxyz) in env frame
) -> np.ndarray:
    """Compute arm_r_link7 origin from wrist pose using the fixed URDF transform.

    `link7_pos_env = wrist_pos_env + R(wrist_quat) @ _LINK7_OFFSET_IN_WRIST_LOCAL`.
    Identical to what IK FK would yield (up to IK convergence residual), but independent
    of the IK solve — keeps the saved reference consistent with the human side.
    """
    wp = np.asarray(wrist_pos, dtype=np.float64)
    wq = np.asarray(wrist_quat_wxyz, dtype=np.float64)
    N = wp.shape[0]
    out = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        R_w = _quat_wxyz_to_R(wq[i])
        out[i] = wp[i] + R_w @ _LINK7_OFFSET_IN_WRIST_LOCAL
    return out

# SMPL-X joint indices
_R_SHOULDER, _R_ELBOW, _R_WRIST = 17, 19, 21

# SMPL → env axis rotation (body upright facing -Y in env)
_R_SMPL_TO_ENV = np.array([
    [1.0, 0.0,  0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
], dtype=np.float64)

# Right-arm kinematic chain in body_pose indexing (body_pose = joints 1..21, so index = joint_id - 1).
# Path pelvis → spine1(3) → spine2(6) → spine3(9) → right_collar(14) → right_shoulder(17)
#       → right_elbow(19) → right_wrist(21). Pelvis is implicit (global_orient default = identity).
_RIGHT_ARM_CHAIN_BP_IDX = [2, 5, 8, 13, 16, 18, 20]

# Palm-cloud keypoints (wrist + 5 MCP joints — flexion-invariant landmarks defining wrist pose).
# Order: wrist, thumb_MCP, index_MCP, middle_MCP, ring_MCP, pinky_MCP.
_SMPLX_PALM_IDX = [21, 52, 40, 43, 49, 46]   # SMPL-X joint indices
_MANO_PALM_IDX  = [0,  1,  5,  9,  13, 17]    # MANO 21-keypoint indices


def _rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → axis-angle (3,). Returns zero vector for identity."""
    R = np.asarray(R, dtype=np.float64)
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(c))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    axis /= (2.0 * np.sin(angle))
    return axis * angle


def _quat_wxyz_to_R_torch(q: torch.Tensor) -> torch.Tensor:
    """Batched wxyz quaternion → 3x3 rotation matrix. q: (..., 4) → R: (..., 3, 3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (x * y - w * z),       2 * (x * z + w * y)], dim=-1),
        torch.stack([2 * (x * y + w * z),       1 - 2 * (xx + zz), 2 * (y * z - w * x)], dim=-1),
        torch.stack([2 * (x * z - w * y),       2 * (y * z + w * x),       1 - 2 * (xx + yy)], dim=-1),
    ], dim=-2)
    return R


def _smpl_right_wrist_world_R(pose_body: torch.Tensor) -> torch.Tensor:
    """Compute SMPL right_wrist world rotation by chaining axis-angle joint rotations.

    pose_body: (N, 21, 3) axis-angle per body joint (= VPoser decode output).
    global_orient is assumed = 0 (identity) — body is upright at SMPL world rest.

    Returns: (N, 3, 3) SMPL-world wrist rotation matrix.
    """
    N = pose_body.shape[0]
    # Convert all joints to rotation matrices in one shot
    R_local = batch_rodrigues(pose_body.reshape(-1, 3)).reshape(N, 21, 3, 3)
    # Chain through right arm: pelvis (I) @ spine1 @ spine2 @ spine3 @ r_collar @ r_shoulder @ r_elbow @ r_wrist
    W = torch.eye(3, device=pose_body.device, dtype=pose_body.dtype).expand(N, 3, 3).contiguous()
    for idx in _RIGHT_ARM_CHAIN_BP_IDX:
        W = torch.bmm(W, R_local[:, idx])
    return W   # (N, 3, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Robot anchor (one-time)
# ──────────────────────────────────────────────────────────────────────────────

def _add_shadow_kpt_frames(model: pin.Model) -> list[str]:
    """Register Shadow Hand 21 MANO keypoint OP_FRAMEs on the Pinocchio model.

    All frames are attached to arm_r_link7's parent joint (arm_r_joint7) with
    static SE3 placements taken from ``_SHADOW_KPT_PLACEMENTS`` (default-pose
    body positions computed offline from FFW_SH5_shadow_flat.usd).

    The resulting frames represent Shadow Hand body locations as they appear at
    runtime in the env, so the IK barrier z-floor and any kpt-tracking
    constraints actually correspond to the deployed robot — replacing the SH5
    finger phantoms that come from the URDF.

    Returns: the list of registered frame names (matches MANO 21-kpt ordering).
    """
    link7_fid = model.getFrameId("arm_r_link7")
    link7_frame = model.frames[link7_fid]
    parent_joint = link7_frame.parent
    link7_placement = link7_frame.placement
    for name, t, q_wxyz in _SHADOW_KPT_PLACEMENTS:
        if model.existFrame(name):
            continue
        R_mat = _quat_wxyz_to_R(q_wxyz)
        offset_se3 = pin.SE3(R_mat, np.asarray(t, dtype=np.float64))
        # Placement relative to the parent joint = link7_placement * offset.
        new_placement = link7_placement * offset_se3
        model.addFrame(
            pin.Frame(name, parent_joint, link7_fid, new_placement, pin.FrameType.OP_FRAME)
        )
    return _SHADOW_KPT_FRAME_NAMES


def _add_virtual_wrist_frame(model: pin.Model, robot_kind: str) -> str:
    """Add a virtual frame to the Pinocchio model representing the wrist mount of
    the chosen robot variant. Returns the frame name to use for IK.

    For ``sh5``: returns the URDF-defined ``hx5_d20_right_base`` (no-op).
    For ``shadow``: adds a frame at ``arm_r_link7 + offset`` from
        ``_WRIST_MOUNT_TRANSFORMS["shadow"]`` and returns its name.

    The virtual frame is attached to ``arm_r_link7``'s joint so it moves
    rigidly with the arm under inverse kinematics.
    """
    transform = _WRIST_MOUNT_TRANSFORMS[robot_kind]
    frame_name = transform["frame_name"]
    if model.existFrame(frame_name):
        return frame_name
    # Build SE3 placement: position offset + rotation in arm_r_link7 frame
    R_offset = _quat_wxyz_to_R(transform["quat_in_link7_wxyz"])
    placement = pin.SE3(R_offset, transform["offset_in_link7"].copy())
    parent_link7_frame_id = model.getFrameId("arm_r_link7")
    parent_joint = model.frames[parent_link7_frame_id].parent
    # Stack on top of arm_r_link7's existing local placement.
    M_link7 = model.frames[parent_link7_frame_id].placement
    new_placement = M_link7 * placement
    new_frame = pin.Frame(
        frame_name,
        parent_joint,
        parent_link7_frame_id,
        new_placement,
        pin.FrameType.OP_FRAME,
    )
    model.addFrame(new_frame)
    return frame_name


def _compute_robot_anchors() -> tuple[np.ndarray, float, float]:
    """Return (robot right shoulder in env frame, upper-arm length, elbow→link7 length).

    The "forearm" length used by bone rescaling is now the elbow→arm_r_link7 distance,
    NOT elbow→wrist. The wrist (hx5_d20_right_base) is reached afterward by adding a
    FIXED 7.8cm offset rotated by wrist orientation — this matches the URDF kinematic
    chain where joints 5/6/7 rotate the elbow→link7 segment and the link7→wrist mount
    is a rigid attachment.
    """
    model = pin.buildModelFromUrdf(str(_URDF_PATH))
    data = model.createData()
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    sh_base = np.asarray(data.oMf[model.getFrameId(_ROBOT_SHOULDER_FRAME)].translation, dtype=np.float64)
    el_base = np.asarray(data.oMf[model.getFrameId(_ROBOT_ELBOW_FRAME)].translation, dtype=np.float64)
    link7_base = np.asarray(data.oMf[model.getFrameId("arm_r_link7")].translation, dtype=np.float64)
    upper_arm = float(np.linalg.norm(el_base - sh_base))
    forearm_to_link7 = float(np.linalg.norm(link7_base - el_base))   # actual "forearm bone" length
    sh_env = _R_BASE_ENV @ sh_base + _ROBOT_POS_ENV
    return sh_env, upper_arm, forearm_to_link7




# ──────────────────────────────────────────────────────────────────────────────
# VPoser elbow extraction (batched, returns z_traj for downstream rendering)
# ──────────────────────────────────────────────────────────────────────────────

class VPoserPipeline:
    """Holds SMPL-X + VPoser + fitted β; provides batched elbow extraction."""

    def __init__(self, device: torch.device, robot_upper_arm: float, robot_forearm_to_link7: float):
        """`robot_forearm_to_link7` is the robot's elbow → arm_r_link7 distance. The
        wrist (palm mount) is computed by adding a fixed link7→wrist offset rotated by
        MANO wrist quaternion — see `_rescale_arm` inside `extract_batched`.
        """
        self.device = device
        self.robot_upper_arm = float(robot_upper_arm)
        self.robot_forearm_to_link7 = float(robot_forearm_to_link7)
        beta_np = np.load(str(_BETA_PATH)).astype(np.float32)
        self.beta = torch.from_numpy(beta_np).unsqueeze(0).to(device)

        self.smpl_1 = smplx.create(
            _SMPLX_MODEL_DIR, model_type="smplx", gender="neutral",
            use_pca=False, batch_size=1, ext="npz",
        ).to(device)
        self.vp, _ = load_model(
            _VPOSER_CKPT, model_code=VPoser,
            remove_words_in_model_weights="vp_model.",
            disable_grad=True,
        )
        self.vp = self.vp.to(device).eval()

        self.R_smpl_env = torch.from_numpy(_R_SMPL_TO_ENV.astype(np.float32)).to(device)

        # Cache T-pose pelvis→right-shoulder offset (for anchoring pelvis in env)
        with torch.no_grad():
            J0 = self.smpl_1(betas=self.beta).joints[0].cpu().numpy().astype(np.float64)
        self._pelvis_to_shoulder_env = _R_SMPL_TO_ENV @ (J0[_R_SHOULDER] - J0[0])
        self._smpl_faces = self.smpl_1.faces  # (20908, 3) for rendering

    def extract_batched(
        self,
        wrist_targets_env: np.ndarray,         # (N, 3)
        wrist_quats_env: np.ndarray,           # (N, 4) wxyz — MANO wrist orientation in env frame
        mano_kpts_env: np.ndarray,             # (N, 21, 3) MANO 21 hand keypoints in env frame
        shoulder_anchor_env: np.ndarray,       # (3,)
        num_iter: int = 300,
        lr: float = 0.05,
        z_reg: float = 0.001,
        smooth_reg: float = 0.05,
        shoulder_anchor_weight: float = 1.0,
        palm_weight: float = 1.0,
        rel_tol: float = 1e-5,
        min_iter: int = 50,
        patience: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, np.ndarray]:
        """VPoser-IK with palm position + palm-cloud alignment + shoulder anchor.

        VPoser decodes 21 body-pose axis-angles INCLUDING the wrist (joint 20). Wrist
        orientation is driven by 3D palm-cloud alignment (wrist + 5 MCP joints, root-aligned
        at the wrist) — Cartesian constraint at natural reaching/grasping poses, stays inside
        VPoser's AMASS distribution.

        Bone rescaling (kinematics-faithful, 2-segment chain):
          (a) shoulder → elbow → link7 — driven by SMPL elbow/forearm rotation; rescaled to
              robot's `upper_arm` and `forearm_to_link7` lengths.
          (b) link7 → wrist (palm) — FIXED 7.8cm mount in wrist's local frame, rotated by
              MANO wrist orientation (NOT by SMPL elbow rotation). Matches URDF
              `hx5_d20_right_joint` (fixed type).
        The optimizer matches `wr_rescaled` = SMPL palm position (end of the 2-segment
        chain) to MANO wrist target — palm-to-palm correspondence preserved. The mount
        geometry is decoupled from elbow rotation so the elbow joint isn't forced to
        absorb wrist-mount displacement.

        Returns:
            elbow_envs:           (N, 3) SMPL elbow positions in env frame (robot-rescaled)
            wrist_pos_residuals:  (N,)   ‖SMPL palm − MANO wrist‖ (m)
            palm_residuals:       (N,)   mean ‖SMPL MCP − MANO MCP‖ per frame (m)
            z_traj:               (N, 32) optimized VPoser latents
            pelvis_env:           (3,)   anchored pelvis position in env frame
        """
        device = self.device
        N = int(wrist_targets_env.shape[0])

        smpl_b = smplx.create(
            _SMPLX_MODEL_DIR, model_type="smplx", gender="neutral",
            use_pca=False, batch_size=N, ext="npz",
        ).to(device)
        beta_N = self.beta.expand(N, -1)

        pelvis_env = shoulder_anchor_env - self._pelvis_to_shoulder_env
        pelvis_env_t = torch.from_numpy(pelvis_env.astype(np.float32)).to(device)
        wrist_targets_t = torch.from_numpy(wrist_targets_env.astype(np.float32)).to(device)
        wrist_quats_t = torch.from_numpy(wrist_quats_env.astype(np.float32)).to(device)         # (N, 4) wxyz
        shoulder_anchor_t = torch.from_numpy(shoulder_anchor_env.astype(np.float32)).to(device)
        mano_kpts_t = torch.from_numpy(mano_kpts_env.astype(np.float32)).to(device)    # (N, 21, 3)

        mano_palm_rel = mano_kpts_t[:, _MANO_PALM_IDX] - mano_kpts_t[:, _MANO_PALM_IDX[0:1]]

        ua_robot = self.robot_upper_arm
        fa_robot = self.robot_forearm_to_link7

        # Pre-compute the world-frame link7→wrist mount vector from MANO wrist quat.
        # In wrist's local frame, link7 sits at `_LINK7_OFFSET_IN_WRIST_LOCAL` (constant from URDF).
        # So wrist = link7 + (− R(wrist_quat) @ offset_in_wrist_local) in world frame.
        # This vector is fixed per frame (does not depend on z).
        offset_local_t = torch.tensor(
            _LINK7_OFFSET_IN_WRIST_LOCAL, dtype=torch.float32, device=device
        )                                                                  # (3,)
        R_wrist_world = _quat_wxyz_to_R_torch(wrist_quats_t)                # (N, 3, 3)
        # World-frame displacement from link7 to wrist (palm mount).
        link7_to_wrist_world = -(R_wrist_world @ offset_local_t)            # (N, 3)

        z = torch.zeros(N, self.vp.latentD, device=device, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)

        def _rescale_arm(J_env_):
            """Two-segment rescaling that matches the physical robot kinematics:
              (1) shoulder → elbow → link7 — driven by SMPL elbow/forearm rotation,
                  rescaled to robot's `upper_arm` and `forearm_to_link7` lengths.
              (2) link7 → wrist (palm) — FIXED 7.8cm mount in wrist's local frame,
                  rotated by MANO wrist orientation (NOT by SMPL elbow rotation).
            The chain ends at `wr_new` = SMPL palm position (matched to MANO wrist target).
            Returns (sh, el_rescaled, link7_rescaled, wr_rescaled), each (N, 3).
            """
            sh_ = J_env_[:, _R_SHOULDER]
            el_ = J_env_[:, _R_ELBOW]
            wr_smpl_ = J_env_[:, _R_WRIST]
            u_vec = el_ - sh_
            u_len = u_vec.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            el_new = sh_ + u_vec / u_len * ua_robot
            # Use SMPL forearm direction (elbow → SMPL wrist joint) for the elbow→link7 segment.
            f_vec = wr_smpl_ - el_
            f_len = f_vec.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            link7_new = el_new + f_vec / f_len * fa_robot
            # Palm (wrist body) = link7 + fixed mount vector (constant per frame).
            wr_new = link7_new + link7_to_wrist_world
            return sh_, el_new, link7_new, wr_new

        prev_loss = float("inf")
        plateau = 0
        for i in range(num_iter):
            pose_body = self.vp.decode(z)["pose_body"]                        # (N, 21, 3)
            pose_body_flat = pose_body.reshape(N, 63)
            out = smpl_b(betas=beta_N, body_pose=pose_body_flat)
            J_world_smpl = out.joints                                          # (N, 127, 3)
            J_env = ((J_world_smpl - J_world_smpl[:, 0:1]) @ self.R_smpl_env.T) + pelvis_env_t

            sh_env, el_rescaled, link7_rescaled, wr_rescaled = _rescale_arm(J_env)

            # Palm (= SMPL wrist) position matched to MANO wrist target. The two-segment
            # chain (elbow→link7 driven by SMPL forearm rotation + link7→wrist fixed mount
            # driven by MANO wrist quat) decouples elbow rotation from the wrist mount.
            wrist_pos_loss = (wr_rescaled - wrist_targets_t).pow(2).sum(-1).mean()
            shoulder_loss = (sh_env - shoulder_anchor_t).pow(2).sum(-1).mean()

            # Palm cloud: root-aligned at wrist (in SMPL world). Hand size = SMPL hand size
            # (the wrist→MCP offsets are tiny vs arm so robot scaling doesn't apply here).
            smpl_palm_smpl = J_world_smpl[:, _SMPLX_PALM_IDX]
            smpl_palm_rel_smpl = smpl_palm_smpl - smpl_palm_smpl[:, 0:1]
            smpl_palm_rel_env = smpl_palm_rel_smpl @ self.R_smpl_env.T
            palm_loss = (smpl_palm_rel_env[:, 1:] - mano_palm_rel[:, 1:]).pow(2).sum((-1, -2)).mean()

            z_reg_loss = z_reg * z.pow(2).mean()
            smooth_loss = (smooth_reg * (z[1:] - z[:-1]).pow(2).mean()
                           if N > 1 else torch.zeros((), device=device))

            loss = (wrist_pos_loss
                    + palm_weight * palm_loss
                    + shoulder_anchor_weight * shoulder_loss
                    + z_reg_loss + smooth_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Early termination: relative loss change below rel_tol for `patience` consecutive
            # steps, after a min number of iterations.
            cur = loss.item()
            if i >= min_iter:
                rel = abs(prev_loss - cur) / max(abs(cur), 1e-8)
                if rel < rel_tol:
                    plateau += 1
                    if plateau >= patience:
                        break
                else:
                    plateau = 0
            prev_loss = cur

        with torch.no_grad():
            pose_body = self.vp.decode(z)["pose_body"]
            pose_body_flat = pose_body.reshape(N, 63)
            out = smpl_b(betas=beta_N, body_pose=pose_body_flat)
            J_world_smpl = out.joints
            J_env = ((J_world_smpl - J_world_smpl[:, 0:1]) @ self.R_smpl_env.T) + pelvis_env_t
            sh_env, el_rescaled, _, wr_rescaled = _rescale_arm(J_env)
            elbow_envs = el_rescaled.cpu().numpy().astype(np.float64)             # robot-bone-rescaled
            wrist_env_np = wr_rescaled.cpu().numpy().astype(np.float64)
            smpl_palm_rel_env = (J_world_smpl[:, _SMPLX_PALM_IDX] - J_world_smpl[:, _SMPLX_PALM_IDX[0:1]]) @ self.R_smpl_env.T
            palm_resids_per_frame = (smpl_palm_rel_env[:, 1:] - mano_palm_rel[:, 1:]).norm(dim=-1).mean(dim=-1)
            palm_resids = palm_resids_per_frame.cpu().numpy().astype(np.float64)
        wrist_pos_resids = np.linalg.norm(wrist_env_np - wrist_targets_env, axis=-1)
        return (elbow_envs, wrist_pos_resids, palm_resids, z.detach(), pelvis_env)

    def render_smpl_per_frame(
        self,
        z_traj: torch.Tensor,    # (N, 32)
        pelvis_env: np.ndarray,  # (3,)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (verts_env (N, V, 3), joints_env (N, 127, 3)) for the trajectory."""
        device = self.device
        N = int(z_traj.shape[0])
        smpl_b = smplx.create(
            _SMPLX_MODEL_DIR, model_type="smplx", gender="neutral",
            use_pca=False, batch_size=N, ext="npz",
        ).to(device)
        beta_N = self.beta.expand(N, -1)
        with torch.no_grad():
            pose_body = self.vp.decode(z_traj)["pose_body"].reshape(N, 63)
            out = smpl_b(betas=beta_N, body_pose=pose_body)
            J_smpl = out.joints                              # (N, 127, 3)
            V_smpl = out.vertices                             # (N, 10475, 3)
            pelvis = J_smpl[:, 0:1]                           # (N, 1, 3)
            J_env = (J_smpl - pelvis) @ self.R_smpl_env.T + torch.from_numpy(
                pelvis_env.astype(np.float32)
            ).to(device)
            V_env = (V_smpl - pelvis) @ self.R_smpl_env.T + torch.from_numpy(
                pelvis_env.astype(np.float32)
            ).to(device)
        return V_env.cpu().numpy().astype(np.float64), J_env.cpu().numpy().astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Pink IK per-frame with warm-start
# ──────────────────────────────────────────────────────────────────────────────

def _solve_ik_per_frame(
    model: pin.Model,
    data: pin.Data,
    arm_q_idx: list[int],
    wrist_pos_env_all: np.ndarray,   # (N, 3)
    wrist_quat_env_wxyz_all: np.ndarray,   # (N, 4)
    elbow_target_env_all: np.ndarray | None,   # (N, 3) or None — soft elbow target
    link7_target_env_all: np.ndarray | None = None,   # (N, 3) or None — soft link7 target
    num_iter: int = 100,
    barrier_gain: float = 10.0,
    barrier_margin: float = 0.0,
    elbow_cost: float = 0.3,
    link7_cost: float = 0.3,
    dt: float = 1.0 / 60.0,
    solver: str = "quadprog",
    v_tol: float = 1e-3,
    wrist_frame_name: str = _WRIST_FRAME,   # override for Shadow Hand variant
    barrier_frame_names: list[str] | None = None,  # override for Shadow Hand variant
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame pink IK with warm-start q from previous frame.

    Tasks:
      - wrist (`hx5_d20_right_base`) HARD: position_cost=1.0, orientation_cost=1.0.
      - elbow (`arm_r_link4`) SOFT (if provided): position_cost=`elbow_cost` (default 0.3).
      - link7 (`arm_r_link7`) SOFT (if provided): position_cost=`link7_cost` (default 0.3).
        Redundant with wrist (link7 = wrist + R(wrist_quat) @ const), but serves as a
        position-space surrogate for wrist orientation — numerically more stable and
        explicitly constrains the elbow→link7 forearm direction together with the elbow task.
      - posture: cost=1e-3 regularization.
      - 21 MANO-kpt z-floor PositionBarriers (no penetrating the table).

    Early termination: stops iterating once ``‖v‖ < v_tol``. Warm-started frames typically
    converge in <20 iterations vs the 100-iteration cap.

    Returns:
        arm_qs:  (N, 7) robot arm_r joint angles (rad)
        pos_err: (N,)   wrist position error (m) per frame
        rot_err: (N,)   wrist orientation error (rad) per frame
    """
    N = int(wrist_pos_env_all.shape[0])
    table_height = float(_TABLE_SIZE[2])

    # Initial q: default arm_r joint values from env config
    q_warm = pin.neutral(model)
    for i, jn in enumerate(_ARM_R_JOINT_NAMES):
        q_warm[arm_q_idx[i]] = _DEFAULT_JOINT_POS[jn]

    arm_qs = np.zeros((N, 7), dtype=np.float64)
    pos_errs = np.zeros(N, dtype=np.float64)
    rot_errs = np.zeros(N, dtype=np.float64)

    # Pre-build PositionBarriers once (reused per frame; pure inequality constraint).
    # For SH5: uses the SH5 finger/MANO link frames added via _resolve_frame_ids.
    # For Shadow Hand: pass shadow-specific kpt frame names so the z-floor
    # barrier constrains the ACTUAL Shadow Hand bodies (registered via
    # _add_shadow_kpt_frames) rather than the SH5 finger phantoms.
    p_min_z = float(table_height + barrier_margin)
    barrier_names = barrier_frame_names if barrier_frame_names is not None else _KPT_21_FRAME_NAMES
    barriers = [
        PositionBarrier(
            fname, indices=[2], p_min=np.array([p_min_z]),
            gain=barrier_gain, safe_displacement_gain=0.0,
        )
        for fname in barrier_names
    ]

    wrist_frame_id = model.getFrameId(wrist_frame_name)

    for i in range(N):
        target_pos_base, target_quat_base = _env_pose_to_base(
            wrist_pos_env_all[i], wrist_quat_env_wxyz_all[i]
        )
        target_SE3 = pin.SE3(_quat_wxyz_to_R(target_quat_base), target_pos_base)

        config = pink.Configuration(model, data, q_warm.copy())

        ee_task = FrameTask(wrist_frame_name, position_cost=1.0, orientation_cost=1.0)
        ee_task.set_target(target_SE3)
        posture_task = PostureTask(cost=1e-3)
        posture_task.set_target_from_configuration(config)
        tasks = [ee_task, posture_task]

        if elbow_target_env_all is not None:
            elbow_pos_base, _ = _env_pose_to_base(
                elbow_target_env_all[i], np.array([1.0, 0.0, 0.0, 0.0])
            )
            elbow_task = FrameTask("arm_r_link4", position_cost=elbow_cost, orientation_cost=0.0)
            elbow_task.set_target(pin.SE3(np.eye(3), elbow_pos_base))
            tasks.append(elbow_task)

        if link7_target_env_all is not None:
            link7_pos_base, _ = _env_pose_to_base(
                link7_target_env_all[i], np.array([1.0, 0.0, 0.0, 0.0])
            )
            link7_task = FrameTask("arm_r_link7", position_cost=link7_cost, orientation_cost=0.0)
            link7_task.set_target(pin.SE3(np.eye(3), link7_pos_base))
            tasks.append(link7_task)

        for _ in range(num_iter):
            try:
                v = pink.solve_ik(config, tasks, dt, solver=solver,
                                  barriers=barriers, safety_break=False)
            except Exception:
                v = np.zeros(model.nv)
            config.update(config.integrate(v, dt))
            if float(np.linalg.norm(v)) < v_tol:
                break

        # Errors
        pin.forwardKinematics(model, data, config.q)
        pin.updateFramePlacements(model, data)
        T = data.oMf[wrist_frame_id]
        pos_w = np.asarray(T.translation, dtype=np.float64)
        q_xyzw = pin.Quaternion(T.rotation).coeffs()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
        pos_errs[i] = float(np.linalg.norm(pos_w - target_pos_base))
        dot = float(np.clip(np.abs(np.dot(q_wxyz, target_quat_base)), 0.0, 1.0))
        rot_errs[i] = float(2.0 * np.arccos(dot))

        arm_qs[i] = np.array([config.q[arm_q_idx[j]] for j in range(7)])
        q_warm = config.q.copy()   # warm-start next frame

    return arm_qs, pos_errs, rot_errs


# ──────────────────────────────────────────────────────────────────────────────
# Video rendering (trimesh per frame → mp4)
# ──────────────────────────────────────────────────────────────────────────────

def _build_camera_transform(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0., 0., 1.]),
) -> np.ndarray:
    """OpenGL camera transform (4x4) — look from `eye` at `target`, env Z up.

    Camera local convention: -Z forward, +Y up, +X right.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)

    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = cam_up
    T[:3, 2] = -forward   # camera looks along -Z
    T[:3, 3] = eye
    return T


def _make_sphere(center, color_rgba, radius=0.018):
    s = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    s.apply_translation(center)
    s.visual.face_colors = color_rgba
    return s


def _make_line(a, b, color_rgba, radius=0.003):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    v = b - a
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return None
    cyl = trimesh.creation.cylinder(radius=radius, height=L, sections=12)
    z = np.array([0.0, 0.0, 1.0])
    u = v / L
    axis = np.cross(z, u)
    s = float(np.linalg.norm(axis))
    if s < 1e-6:
        R = np.eye(3) if u[2] > 0 else np.diag([1, 1, -1])
    else:
        axis /= s
        c = float(np.dot(z, u))
        ang = np.arctan2(s, c)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (a + b) / 2.0
    cyl.apply_transform(T)
    cyl.visual.face_colors = color_rgba
    return cyl


def _render_video(
    out_path: Path,
    smpl_verts_env: np.ndarray,    # (N, 10475, 3)
    smpl_joints_env: np.ndarray,   # (N, 127, 3)
    smpl_faces: np.ndarray,        # (F, 3)
    wrist_targets_env: np.ndarray, # (N, 3)  MANO wrist target (env reference)
    arm_qs: np.ndarray,            # (N, 7)  unused now — kept for API compat
    model: pin.Model,               # unused now — kept for API compat
    data: pin.Data,                 # unused now — kept for API compat
    arm_q_idx: list[int],           # unused now — kept for API compat
    elbow_envs: np.ndarray,        # (N, 3)  rescaled SMPL elbow (env reference)
    link7_envs: np.ndarray,        # (N, 3)  analytic link7 (env reference)
    shoulder_env: np.ndarray,      # (3,)    robot shoulder anchor (env, fixed)
    fps: int = 30,
    width: int = 960,
    height: int = 720,
) -> None:
    """Render per-frame pyrender scene → mp4 video (true offscreen via EGL).

    Each frame shows two uniformly-colored chains plus the SMPL body mesh:
      • SMPL body mesh (light gray, semi-transparent)
      • PINK   — SMPL skeleton chain (VPoser fit, raw): shoulder → elbow → wrist
      • YELLOW — Env reference chain (what the env actually tracks):
                 robot shoulder → rescaled elbow → analytic link7 → MANO wrist
    """
    N = int(smpl_verts_env.shape[0])

    # Fixed camera pose (env frame, Z-up)
    cam_T = _build_camera_transform(
        eye=np.array([1.7, -0.5, 1.9]),
        target=np.array([0.45, 0.55, 1.15]),
        up=np.array([0.0, 0.0, 1.0]),
    )

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0), aspectRatio=width / height)
    light_directional = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    light_ambient_color = np.array([0.35, 0.35, 0.35])

    # SMPL body: PBR material (smooth shading + alpha — face_colors conflicts with smooth=True)
    body_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.7, 0.7, 0.55],
        metallicFactor=0.0, roughnessFactor=0.7,
        alphaMode="BLEND",
    )

    def _pyr_primitive(tm: trimesh.Trimesh) -> pyrender.Mesh:
        # Primitives (spheres, cylinders): face_colors + flat shading.
        return pyrender.Mesh.from_trimesh(tm, smooth=False)

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=7)
    try:
        for i in range(N):
            scene = pyrender.Scene(
                bg_color=np.array([1.0, 1.0, 1.0, 1.0]),
                ambient_light=light_ambient_color,
            )
            # SMPL body mesh (smooth + alpha via PBR material)
            body_tm = trimesh.Trimesh(vertices=smpl_verts_env[i], faces=smpl_faces, process=False)
            scene.add(pyrender.Mesh.from_trimesh(body_tm, smooth=True, material=body_material))

            # Two chains, each in a single color:
            #   (1) SMPL skeleton (VPoser fit, raw)   — PINK
            #   (2) Env reference chain (what env tracks) — YELLOW
            SMPL_COLOR = [230, 100, 180, 255]
            REF_COLOR  = [255, 220,   0, 255]

            # (1) SMPL skeleton (raw VPoser output): shoulder → elbow → wrist
            sm_sh = smpl_joints_env[i, _R_SHOULDER]
            sm_el = smpl_joints_env[i, _R_ELBOW]
            sm_wr = smpl_joints_env[i, _R_WRIST]
            for center in (sm_sh, sm_el, sm_wr):
                scene.add(_pyr_primitive(_make_sphere(center, SMPL_COLOR)))
            for a, b in [(sm_sh, sm_el), (sm_el, sm_wr)]:
                c = _make_line(a, b, SMPL_COLOR, radius=0.004)
                if c is not None:
                    scene.add(_pyr_primitive(c))

            # (2) Env reference chain (the exact positions the env uses for arm tracking):
            #     robot shoulder (anchor) → rescaled elbow → analytic link7 → MANO wrist
            ref_pts = [shoulder_env, elbow_envs[i], link7_envs[i], wrist_targets_env[i]]
            for center in ref_pts:
                scene.add(_pyr_primitive(_make_sphere(center, REF_COLOR)))
            for a, b in zip(ref_pts[:-1], ref_pts[1:]):
                c = _make_line(a, b, REF_COLOR, radius=0.004)
                if c is not None:
                    scene.add(_pyr_primitive(c))

            # Env Z-axis cue at origin
            zc = _make_line(np.zeros(3), np.array([0.0, 0.0, 0.15]), [0, 0, 255, 255], radius=0.004)
            if zc is not None:
                scene.add(_pyr_primitive(zc))

            # Camera + light
            scene.add(camera, pose=cam_T)
            scene.add(light_directional, pose=cam_T)

            color, _ = renderer.render(scene)   # (H, W, 3) uint8 RGB
            writer.append_data(color)
    finally:
        writer.close()
        renderer.delete()


# ──────────────────────────────────────────────────────────────────────────────
# Per-trajectory orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def _process_trajectory(
    traj_path: Path,
    out_dir: Path,
    dataset_dir: Path,
    pipeline: VPoserPipeline,
    robot_model: pin.Model,
    robot_data: pin.Data,
    arm_q_idx: list[int],
    shoulder_env: np.ndarray,
    args: argparse.Namespace,
) -> tuple[int, str]:
    """Run the full pipeline for one trajectory. Returns (status_code, message)."""
    # Output: arm_keypoints.npz holds reference positions for arm-side tracking
    #   - elbow_pos : SMPL elbow position (from VPoser fit) per frame
    #   - link7_pos : robot arm_r_link7 position from IK FK per frame.
    # Per-robot variants get a suffix so sh5 and shadow can coexist.
    suffix = "" if args.robot == "sh5" else f"_{args.robot}"
    arm_keypoints_path = out_dir / f"arm_keypoints{suffix}.npz"
    arm_path = out_dir / f"arm_joint_pos{suffix}.npy"
    video_path = out_dir / f"vposer_ik_video{suffix}.mp4"

    # Skip if everything exists and not overwriting
    if not args.overwrite and arm_keypoints_path.exists() and arm_path.exists() and video_path.exists():
        return 0, "skip (all outputs exist)"

    # 1. Canonicalize (wrist + MANO 21 keypoints) and capture transform params
    res = _canonicalize_full(traj_path, dataset_dir)
    if res is None:
        return -1, "canonicalize failed"
    wp_all, wq_all, mano_kpts_canon, canon_offset, canon_angle, canon_pivot = res

    # 2. VPoser elbow extraction (palm-to-palm match via 2-segment chain with fixed wrist mount)
    (elbow_envs, wrist_pos_resids, palm_resids,
     z_traj, pelvis_env) = pipeline.extract_batched(
        wp_all, wq_all, mano_kpts_canon, shoulder_env,
        num_iter=args.vposer_iter, lr=args.vposer_lr, z_reg=args.z_reg,
        smooth_reg=args.smooth_reg,
        shoulder_anchor_weight=args.shoulder_anchor_weight,
        palm_weight=args.palm_weight,
        rel_tol=args.vposer_rel_tol,
        min_iter=args.vposer_min_iter,
        patience=args.vposer_patience,
    )
    # Compute link7 reference analytically from MANO wrist pose using the fixed
    # URDF wrist→link7 transform. Used as both (a) a soft IK task and (b) the saved
    # reference position used by the RL env. Both elbow_envs and link7_envs are
    # human-side references (independent of IK convergence quality).
    link7_envs = _compute_link7_from_wrist(wp_all, wq_all)

    # Save elbow + link7 in RAW frame (same convention as `mano_kpts_right` in
    # trajectory_keypoints.npz) so the env's existing canonicalization handles them
    # uniformly with mano_kpts. The VPoser/IK steps run in canonical frame; we invert
    # the canonicalization before saving.
    elbow_raw = _uncanonicalize_pos(elbow_envs, canon_offset, canon_angle, canon_pivot)
    link7_raw = _uncanonicalize_pos(link7_envs, canon_offset, canon_angle, canon_pivot)

    # 3. Pink IK per-frame (with elbow + link7 soft targets)
    wrist_frame_name = _WRIST_MOUNT_TRANSFORMS[args.robot]["frame_name"]
    # For shadow, use Shadow Hand kpt frames (registered earlier via
    # _add_shadow_kpt_frames). For sh5, pass None to fall back to default SH5
    # finger phantom frames.
    barrier_frame_names = _SHADOW_KPT_FRAME_NAMES if args.robot == "shadow" else None
    arm_qs, pos_errs, rot_errs = _solve_ik_per_frame(
        robot_model, robot_data, arm_q_idx,
        wp_all, wq_all, elbow_envs,
        link7_target_env_all=link7_envs,
        num_iter=args.ik_iter,
        barrier_gain=args.barrier_gain,
        barrier_margin=args.barrier_margin,
        elbow_cost=args.elbow_cost,
        link7_cost=args.link7_cost,
        v_tol=args.ik_v_tol,
        wrist_frame_name=wrist_frame_name,
        barrier_frame_names=barrier_frame_names,
    )
    np.save(str(arm_path), arm_qs.astype(np.float32))

    # Save both arm-reference keypoints in one npz file.
    np.savez(
        str(arm_keypoints_path),
        elbow_pos=elbow_raw.astype(np.float32),   # (N, 3)  SMPL elbow (env→raw)
        link7_pos=link7_raw.astype(np.float32),   # (N, 3)  arm_r_link7 from MANO wrist (env→raw)
    )

    # 4. Render video (env reference chain + SMPL skeleton, no IK FK overlay).
    smpl_verts_env, smpl_joints_env = pipeline.render_smpl_per_frame(z_traj, pelvis_env)
    _render_video(
        video_path,
        smpl_verts_env, smpl_joints_env, pipeline._smpl_faces,
        wp_all, arm_qs,
        robot_model, robot_data, arm_q_idx,
        elbow_envs=elbow_envs,
        link7_envs=link7_envs,
        shoulder_env=shoulder_env,
        fps=args.fps, width=args.video_w, height=args.video_h,
    )

    msg = (f"N={len(wp_all):4d}  "
           f"smpl_pos={wrist_pos_resids.mean()*100:.2f}cm  "
           f"palm={palm_resids.mean()*100:.2f}cm  "
           f"ik_pos={pos_errs.mean()*100:.3f}cm rot={np.degrees(rot_errs).mean():.2f}°")
    return 1, msg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified arm-reference pipeline.")
    parser.add_argument("--dataset", type=str, default="hocap", choices=["oakink", "hocap"])
    parser.add_argument("--object_id", type=str, default="")
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--data_id", type=int, default=-1)
    parser.add_argument(
        "--robot", type=str, default="sh5", choices=["sh5", "shadow"],
        help="Which robot variant to solve IK for. 'sh5' (default) targets the "
             "URDF body 'hx5_d20_right_base' (FFW-SH5 native wrist). 'shadow' "
             "registers a virtual frame at the position+orientation of "
             "robot0_palm relative to arm_r_link7 (extracted from "
             "FFW_SH5_shadow_flat.usd) and targets that. Output files get a "
             "'_shadow' suffix to coexist with sh5 outputs.",
    )
    # VPoser params
    parser.add_argument("--vposer_iter", type=int, default=300)
    parser.add_argument("--vposer_lr", type=float, default=0.05)
    parser.add_argument("--z_reg", type=float, default=0.001)
    parser.add_argument("--smooth_reg", type=float, default=0.05)
    parser.add_argument("--shoulder_anchor_weight", type=float, default=1.0)
    parser.add_argument("--palm_weight", type=float, default=1.0,
                        help="Palm-cloud (wrist+5 MCPs, wrist-root-aligned) alignment loss weight; "
                             "drives wrist orientation through VPoser.")
    parser.add_argument("--vposer_rel_tol", type=float, default=1e-5,
                        help="Relative loss change threshold for early termination of VPoser fit.")
    parser.add_argument("--vposer_patience", type=int, default=10,
                        help="Consecutive steps below rel_tol required to break VPoser fit.")
    parser.add_argument("--vposer_min_iter", type=int, default=50,
                        help="Minimum iterations before VPoser early-termination is considered.")
    # Pink IK params
    parser.add_argument("--ik_iter", type=int, default=100, help="Pink IK iters per frame (warm-start init).")
    parser.add_argument("--ik_v_tol", type=float, default=1e-3,
                        help="Pink IK velocity-norm threshold for early termination per frame.")
    parser.add_argument("--barrier_gain", type=float, default=10.0)
    parser.add_argument("--barrier_margin", type=float, default=0.0)
    parser.add_argument("--elbow_cost", type=float, default=0.3)
    parser.add_argument("--link7_cost", type=float, default=0.3,
                        help="Pink FrameTask position_cost for the arm_r_link7 soft target. "
                             "Redundant with wrist FrameTask (link7 = wrist + R(wrist_quat) @ const) "
                             "but adds a position-space surrogate that stabilizes wrist-orientation "
                             "matching and constrains the elbow→link7 forearm direction.")
    # Video params
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video_w", type=int, default=640)
    parser.add_argument("--video_h", type=int, default=480)
    # IO
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n[setup] device={device}")

    print(f"[setup] Loading robot URDF + reducing to arm_r (robot={args.robot})...")
    robot_model, _, arm_q_idx, _ = _build_arm_only_model()
    robot_data = robot_model.createData()
    _ = _resolve_frame_ids(robot_model)   # registers fingertip OP_FRAMES on the model
    # For shadow variant, register additional OP_FRAMEs:
    #   (1) virtual wrist frame at arm_r_link7 + offset representing robot0_palm pose
    #       — IK targets this instead of hx5_d20_right_base
    #   (2) 21 MANO keypoint frames at Shadow Hand body default-pose locations
    #       — IK barrier z-floor constrains THESE (not the SH5 finger phantoms)
    shadow_barrier_frame_names: list[str] | None = None
    if args.robot == "shadow":
        wrist_frame_name = _add_virtual_wrist_frame(robot_model, "shadow")
        shadow_barrier_frame_names = _add_shadow_kpt_frames(robot_model)
        robot_data = robot_model.createData()   # recreate after model mutation
        print(f"  registered virtual wrist frame '{wrist_frame_name}' "
              f"+ {len(shadow_barrier_frame_names)} Shadow Hand kpt frames for IK")
    print(f"  reduced model nq={robot_model.nq}, arm_r idx_q={arm_q_idx}")

    shoulder_env, upper_arm_robot, forearm_to_link7_robot = _compute_robot_anchors()
    print(f"[setup] Robot right shoulder (env): {shoulder_env}")
    print(f"[setup] Robot upper_arm = {upper_arm_robot:.4f} m, "
          f"forearm→link7 = {forearm_to_link7_robot:.4f} m  "
          f"(wrist mount adds fixed 7.8cm via MANO wrist quat)\n")

    print(f"[setup] Loading VPoser + SMPL-X + β (arm rescaled to robot bone lengths)...")
    pipeline = VPoserPipeline(
        device,
        robot_upper_arm=upper_arm_robot,
        robot_forearm_to_link7=forearm_to_link7_robot,
    )

    dataset_dir = _HOCAP_DIR if args.dataset == "hocap" else _OAKINK_DIR
    mano_dir = dataset_dir / "mano" / "right"
    task_dirs = (
        [mano_dir / args.task] if args.task
        else sorted(d for d in mano_dir.iterdir()
                    if d.is_dir() and (not args.object_id or args.object_id in d.name))
    )
    targets: list[tuple[Path, Path]] = []
    for td in task_dirs:
        dds = ([td / str(args.data_id)] if args.data_id >= 0
               else sorted(d for d in td.iterdir() if d.is_dir()))
        for d in dds:
            t = d / "trajectory_keypoints.npz"
            if t.exists():
                targets.append((d, t))
    total = len(targets)
    print(f"Processing {total} trajectories...\n")

    processed = skipped = errors = 0
    for i, (out_dir, traj_path) in enumerate(targets, start=1):
        prefix = f"[{i}/{total}]"
        try:
            status, msg = _process_trajectory(
                traj_path, out_dir, dataset_dir, pipeline,
                robot_model, robot_data, arm_q_idx, shoulder_env, args,
            )
        except Exception as e:
            print(f"{prefix} ERROR — {e}")
            errors += 1
            continue
        rel = traj_path.parent.relative_to(dataset_dir)
        if status == 0:
            print(f"{prefix} {rel}  {msg}")
            skipped += 1
        elif status == 1:
            print(f"{prefix} {rel}  {msg}")
            processed += 1
        else:
            print(f"{prefix} {rel}  error — {msg}")
            errors += 1

    print(f"\nDone: {processed} processed, {skipped} skipped, {errors} errors.")


if __name__ == "__main__":
    main()
