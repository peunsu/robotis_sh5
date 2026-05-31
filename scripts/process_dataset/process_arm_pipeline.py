"""Unified per-trajectory arm-reference pipeline.

For each trajectory under `<dataset>/mano/right/<task>/<data_id>/`:
  1. Canonicalize wrist + object trajectories (table-anchor + canonical Z-rotation).
  2. VPoser-IK (batched, all N frames) → SMPL elbow positions in env frame.
  3. Pink IK per-frame with warm-start (wrist FrameTask + elbow FrameTask soft prior +
     21 MANO keypoint barriers for z ≥ table_height) → robot arm_r joint angles.
  4. Render an mp4 visualization (SMPL mesh + skeleton + robot arm chain per frame).

Outputs written to each trajectory directory:
    elbow_joint_pos.npy   shape (N, 3) float32  — SMPL elbow per frame (env frame)
    arm_joint_pos.npy     shape (N, 7) float32  — robot arm_r joint angles per frame
    vposer_ik_video.mp4                          — visualization

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
_ROBOT_WRIST_FRAME    = "hx5_d20_right_base"

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

def _compute_robot_anchors() -> tuple[np.ndarray, float, float]:
    """Return (robot right shoulder in env frame, robot upper-arm length, robot forearm length)."""
    model = pin.buildModelFromUrdf(str(_URDF_PATH))
    data = model.createData()
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    sh_base = np.asarray(data.oMf[model.getFrameId(_ROBOT_SHOULDER_FRAME)].translation, dtype=np.float64)
    el_base = np.asarray(data.oMf[model.getFrameId(_ROBOT_ELBOW_FRAME)].translation, dtype=np.float64)
    wr_base = np.asarray(data.oMf[model.getFrameId(_ROBOT_WRIST_FRAME)].translation, dtype=np.float64)
    upper_arm = float(np.linalg.norm(el_base - sh_base))
    forearm = float(np.linalg.norm(wr_base - el_base))
    sh_env = _R_BASE_ENV @ sh_base + _ROBOT_POS_ENV
    return sh_env, upper_arm, forearm




# ──────────────────────────────────────────────────────────────────────────────
# VPoser elbow extraction (batched, returns z_traj for downstream rendering)
# ──────────────────────────────────────────────────────────────────────────────

class VPoserPipeline:
    """Holds SMPL-X + VPoser + fitted β; provides batched elbow extraction."""

    def __init__(self, device: torch.device, robot_upper_arm: float, robot_forearm: float):
        self.device = device
        self.robot_upper_arm = float(robot_upper_arm)
        self.robot_forearm = float(robot_forearm)
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
        """VPoser-IK with wrist position + palm-cloud alignment + shoulder anchor.

        VPoser decodes 21 body-pose axis-angles INCLUDING the wrist (joint 20). Wrist
        orientation is driven by 3D palm-cloud alignment (wrist + 5 MCP joints, root-aligned
        at the wrist) — since this is a Cartesian constraint at natural reaching/grasping
        poses, it stays inside VPoser's training distribution (AMASS) and fits cleanly.

        Bone rescaling: SMPL FK output's shoulder→elbow→wrist segments are stretched in
        env frame to robot bone lengths (directions preserved). The rescaled wrist is what
        gets matched to ``wrist_targets_env`` — this anchors SMPL's elbow on the robot's
        elbow swivel circle, making the elbow target IK-feasible.

        Returns:
            elbow_envs:           (N, 3) SMPL elbow positions in env frame (robot-rescaled)
            wrist_pos_residuals:  (N,)   ‖SMPL wrist − wrist target‖ (m)
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
        shoulder_anchor_t = torch.from_numpy(shoulder_anchor_env.astype(np.float32)).to(device)
        mano_kpts_t = torch.from_numpy(mano_kpts_env.astype(np.float32)).to(device)    # (N, 21, 3)

        mano_palm_rel = mano_kpts_t[:, _MANO_PALM_IDX] - mano_kpts_t[:, _MANO_PALM_IDX[0:1]]

        ua_robot = self.robot_upper_arm
        fa_robot = self.robot_forearm

        z = torch.zeros(N, self.vp.latentD, device=device, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)

        def _rescale_arm(J_env_):
            """Rescale shoulder→elbow→wrist segments to robot bone lengths (directions preserved).
            Returns (sh, el_rescaled, wr_rescaled), each (N, 3)."""
            sh_ = J_env_[:, _R_SHOULDER]
            el_ = J_env_[:, _R_ELBOW]
            wr_ = J_env_[:, _R_WRIST]
            u_vec = el_ - sh_
            u_len = u_vec.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            el_new = sh_ + u_vec / u_len * ua_robot
            f_vec = wr_ - el_
            f_len = f_vec.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            wr_new = el_new + f_vec / f_len * fa_robot
            return sh_, el_new, wr_new

        prev_loss = float("inf")
        plateau = 0
        for i in range(num_iter):
            pose_body = self.vp.decode(z)["pose_body"]                        # (N, 21, 3)
            pose_body_flat = pose_body.reshape(N, 63)
            out = smpl_b(betas=beta_N, body_pose=pose_body_flat)
            J_world_smpl = out.joints                                          # (N, 127, 3)
            J_env = ((J_world_smpl - J_world_smpl[:, 0:1]) @ self.R_smpl_env.T) + pelvis_env_t

            sh_env, el_rescaled, wr_rescaled = _rescale_arm(J_env)

            # Wrist position loss uses RESCALED wrist (robot bone lengths)
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
            sh_env, el_rescaled, wr_rescaled = _rescale_arm(J_env)
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
    elbow_target_env_all: np.ndarray | None,   # (N, 3) or None
    num_iter: int = 100,
    barrier_gain: float = 10.0,
    barrier_margin: float = 0.0,
    elbow_cost: float = 0.3,
    dt: float = 1.0 / 60.0,
    solver: str = "quadprog",
    v_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame pink IK with warm-start q from previous frame.

    Early termination: stops iterating once ``‖v‖ < v_tol`` (the QP-returned joint velocity
    is small, i.e. the configuration has converged on the QP optimum). Warm-started frames
    typically converge in <20 iterations vs the 100-iteration cap.

    Returns:
        arm_qs:  (N, 7) robot arm_r joint angles (rad)
        pos_err: (N,) wrist position error (m) per frame
        rot_err: (N,) wrist orientation error (rad) per frame
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

    # Pre-build PositionBarriers once (reused per frame; pure inequality constraint)
    p_min_z = float(table_height + barrier_margin)
    barriers = [
        PositionBarrier(
            fname, indices=[2], p_min=np.array([p_min_z]),
            gain=barrier_gain, safe_displacement_gain=0.0,
        )
        for fname in _KPT_21_FRAME_NAMES
    ]

    wrist_frame_id = model.getFrameId(_WRIST_FRAME)

    for i in range(N):
        target_pos_base, target_quat_base = _env_pose_to_base(
            wrist_pos_env_all[i], wrist_quat_env_wxyz_all[i]
        )
        target_SE3 = pin.SE3(_quat_wxyz_to_R(target_quat_base), target_pos_base)

        config = pink.Configuration(model, data, q_warm.copy())

        ee_task = FrameTask(_WRIST_FRAME, position_cost=1.0, orientation_cost=1.0)
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
    wrist_targets_env: np.ndarray, # (N, 3)
    arm_qs: np.ndarray,            # (N, 7)
    model: pin.Model,
    data: pin.Data,
    arm_q_idx: list[int],
    fps: int = 30,
    width: int = 960,
    height: int = 720,
) -> None:
    """Render per-frame pyrender scene → mp4 video (true offscreen via EGL).

    Each frame shows:
      • SMPL body mesh (light gray, semi-transparent)
      • SMPL right arm chain (red shoulder → green elbow → blue wrist)
      • Wrist target (yellow sphere)
      • Robot right arm chain at IK-solved pose (cyan spheres + lines)
    """
    N = int(smpl_verts_env.shape[0])
    wrist_frame_id = model.getFrameId(_WRIST_FRAME)
    sh_frame_id = model.getFrameId(_ROBOT_SHOULDER_FRAME)
    el_frame_id = model.getFrameId(_ROBOT_ELBOW_FRAME)

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

            sm_sh = smpl_joints_env[i, _R_SHOULDER]
            sm_el = smpl_joints_env[i, _R_ELBOW]
            sm_wr = smpl_joints_env[i, _R_WRIST]

            # SMPL right arm chain markers
            for center, rgba in [
                (sm_sh, [255, 50,  50,  255]),
                (sm_el, [50,  255, 50,  255]),
                (sm_wr, [50,  100, 255, 255]),
                (wrist_targets_env[i], [255, 230, 0, 255]),
            ]:
                scene.add(_pyr_primitive(_make_sphere(center, rgba)))
            for a, b, col in [
                (sm_sh, sm_el, [255, 180, 180, 255]),
                (sm_el, sm_wr, [180, 255, 180, 255]),
            ]:
                c = _make_line(a, b, col)
                if c is not None:
                    scene.add(_pyr_primitive(c))

            # Robot arm landmarks via pinocchio FK at IK-solved q
            q = pin.neutral(model)
            for j in range(7):
                q[arm_q_idx[j]] = float(arm_qs[i, j])
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            sh_env = _R_BASE_ENV @ np.asarray(data.oMf[sh_frame_id].translation) + _ROBOT_POS_ENV
            el_env = _R_BASE_ENV @ np.asarray(data.oMf[el_frame_id].translation) + _ROBOT_POS_ENV
            wr_env = _R_BASE_ENV @ np.asarray(data.oMf[wrist_frame_id].translation) + _ROBOT_POS_ENV
            for center in (sh_env, el_env, wr_env):
                scene.add(_pyr_primitive(_make_sphere(center, [50, 220, 220, 255])))
            for a, b in [(sh_env, el_env), (el_env, wr_env)]:
                c = _make_line(a, b, [180, 240, 240, 255], radius=0.004)
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
    elbow_path = out_dir / "elbow_joint_pos.npy"
    arm_path = out_dir / "arm_joint_pos.npy"
    video_path = out_dir / "vposer_ik_video.mp4"

    # Skip if everything exists and not overwriting
    if not args.overwrite and elbow_path.exists() and arm_path.exists() and video_path.exists():
        return 0, "skip (all outputs exist)"

    # 1. Canonicalize (wrist + MANO 21 keypoints) and capture transform params
    res = _canonicalize_full(traj_path, dataset_dir)
    if res is None:
        return -1, "canonicalize failed"
    wp_all, wq_all, mano_kpts_canon, canon_offset, canon_angle, canon_pivot = res

    # 2. VPoser elbow extraction (palm cloud alignment)
    (elbow_envs, wrist_pos_resids, palm_resids,
     z_traj, pelvis_env) = pipeline.extract_batched(
        wp_all, mano_kpts_canon, shoulder_env,
        num_iter=args.vposer_iter, lr=args.vposer_lr, z_reg=args.z_reg,
        smooth_reg=args.smooth_reg,
        shoulder_anchor_weight=args.shoulder_anchor_weight,
        palm_weight=args.palm_weight,
        rel_tol=args.vposer_rel_tol,
        min_iter=args.vposer_min_iter,
        patience=args.vposer_patience,
    )
    # Save elbow in RAW frame (same convention as `mano_kpts_right` in trajectory_keypoints.npz)
    # so the env's existing canonicalization handles it uniformly with mano_kpts. The VPoser
    # fit ran in canonical frame, so we invert the canonicalization before saving.
    elbow_raw = _uncanonicalize_pos(elbow_envs, canon_offset, canon_angle, canon_pivot)
    np.save(str(elbow_path), elbow_raw.astype(np.float32))

    # 3. Pink IK per-frame
    arm_qs, pos_errs, rot_errs = _solve_ik_per_frame(
        robot_model, robot_data, arm_q_idx,
        wp_all, wq_all, elbow_envs,
        num_iter=args.ik_iter,
        barrier_gain=args.barrier_gain,
        barrier_margin=args.barrier_margin,
        elbow_cost=args.elbow_cost,
        v_tol=args.ik_v_tol,
    )
    np.save(str(arm_path), arm_qs.astype(np.float32))

    # 4. Render video
    smpl_verts_env, smpl_joints_env = pipeline.render_smpl_per_frame(z_traj, pelvis_env)
    _render_video(
        video_path,
        smpl_verts_env, smpl_joints_env, pipeline._smpl_faces,
        wp_all, arm_qs,
        robot_model, robot_data, arm_q_idx,
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

    print(f"[setup] Loading robot URDF + reducing to arm_r...")
    robot_model, _, arm_q_idx, _ = _build_arm_only_model()
    robot_data = robot_model.createData()
    _ = _resolve_frame_ids(robot_model)   # registers fingertip OP_FRAMES on the model
    print(f"  reduced model nq={robot_model.nq}, arm_r idx_q={arm_q_idx}")

    shoulder_env, upper_arm_robot, forearm_robot = _compute_robot_anchors()
    print(f"[setup] Robot right shoulder (env): {shoulder_env}")
    print(f"[setup] Robot upper_arm = {upper_arm_robot:.4f} m, forearm = {forearm_robot:.4f} m\n")

    print(f"[setup] Loading VPoser + SMPL-X + β (arm rescaled to robot bone lengths)...")
    pipeline = VPoserPipeline(device, robot_upper_arm=upper_arm_robot, robot_forearm=forearm_robot)

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
