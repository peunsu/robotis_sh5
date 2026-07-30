"""OakInk dexterous grasping environment — Isaac Lab Direct **MARL** (MAPPO).

Multi-Agent variant of `RobotisSh5GraspEnv`:
  - Arm agent (7D action): arm_r_joint1..7, drives global palm transport
  - Hand agent (20D action): finger_r_joint1..20, drives local manipulation
  - Lift held at fixed_lift_target via PD (not in action)
  - Mass-as-action removed entirely (uses cfg.object_mass directly)
  - Hand observation in palm-local frame (paper convention)
  - Centralized critic via state_space = -1 (auto-flatten arm + hand obs)
  - Sequential forward coupling handled by SequentialMAPPO patch in train_marl.py

Inherits the same scene/object/reference-loading/PD-control primitives from the
single-agent env (largely copy-verbatim with targeted MARL modifications).
"""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

from .robotis_sh5_grasp_marl_env_cfg import RobotisSh5GraspMarlEnvCfg


def quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternion to orthonormalized 6D continuous rotation rep (Zhou et al. 2019).

    Mirrors TJ's `quat_to_6d` (Gram-Schmidt on first two rows of R), but with wxyz input order
    to match isaaclab convention. Returns shape (..., 6).
    """
    q = torch.nn.functional.normalize(quat, dim=-1)
    w, x, y, z = q.unbind(-1)
    r0 = torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y - w * z),
        2 * (x * z + w * y),
    ], dim=-1)
    r1 = torch.stack([
        2 * (x * y + w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z - w * x),
    ], dim=-1)
    a1 = torch.nn.functional.normalize(r0, dim=-1)
    a2 = r1 - (a1 * r1).sum(-1, keepdim=True) * a1
    a2 = torch.nn.functional.normalize(a2, dim=-1)
    return torch.cat([a1, a2], dim=-1)


# Local-frame offsets from link origin to actual fingertip contact point.
# Derived from get_virtual_link_poses() in manager_based pick_and_place utils.py.
# finger_r_link4  (thumb):                +Y offset
# finger_r_link8/12/16/20 (index~little): +Z offset
_FINGERTIP_OFFSETS: dict[str, list[float]] = {
    "finger_r_link4":  [0.0,    0.03975, 0.012],
    "finger_r_link8":  [0.012,  0.0,     0.02425],
    "finger_r_link12": [0.012,  0.0,     0.02425],
    "finger_r_link16": [0.012,  0.0,     0.02425],
    "finger_r_link20": [0.012,  0.0,     0.02425],
}

# Pad-outward normals in each fingertip link's LOCAL frame (mirrors train env;
# identified via scripts/process_dataset/visualize_fingertip_normals.py).
_FINGERTIP_PAD_NORMALS: dict[str, list[float]] = {
    "finger_r_link4":  [0.0, 0.0, 1.0],   # thumb: +Z (palm-outward)
    "finger_r_link8":  [1.0, 0.0, 0.0],   # index: +X (palm-outward)
    "finger_r_link12": [1.0, 0.0, 0.0],
    "finger_r_link16": [1.0, 0.0, 0.0],
    "finger_r_link20": [1.0, 0.0, 0.0],
}

# Elbow position offset in arm_r_link3's local frame (URDF arm_r_joint4 origin in link3).
# See robotis_sh5_grasp_env.py for the full rationale — link3 + offset stays stable
# under elbow rotation, unlike `body_pos_w[arm_r_link4]`.
_ELBOW_OFFSET_IN_LINK3_LOCAL: list[float] = [0.041004, 0.0, -0.135]

# ── [THUMB-RADIUS-FILTER] ───────────────────────────────────────────────
# Thumb fingertip link (`finger_r_link4`) is unusually long compared to the
# other fingers, so PhysX attributes mid-link contacts to the body and
# inflates the "fingertip force" used by rew_fingertip_force. We gate the
# thumb's force contribution by the distance between the AVG contact point
# (sensor.data.contact_pos_w) and the actual thumb-tip world position
# (link origin + _FINGERTIP_OFFSETS["finger_r_link4"]). Only contacts whose
# avg position falls within this radius are kept. Other fingers are not
# affected (their tip link is short enough).
#
# Activated by `track_contact_points=True` on each ContactSensorCfg (see
# `_setup_scene`). If that flag is removed, `contact_pos_w` becomes None
# and the gate in `_get_fingertip_forces` becomes a no-op → original
# unfiltered behavior is restored.
#
# To revert the entire filter:
#   1) Remove `track_contact_points=True` from ContactSensorCfg in
#      `_setup_scene`.
#   2) Remove the [THUMB-RADIUS-FILTER] block inside `_get_fingertip_forces`.
#   3) (Optional) Delete this constant + comment block.
# ────────────────────────────────────────────────────────────────────────
_THUMB_CONTACT_RADIUS_M: float = 0.02
# [THUMB-RADIUS-FILTER] Master switch — toggle to enable/disable the gate
# without touching any other code. False = unfiltered behavior (original).
# When False, monitoring keys `Force / thumb_*` reflect pre-filter values.
_THUMB_FILTER_ENABLED: bool = False

# MANO keypoint index → SH5 body name mapping for non-fingertip joints.
# MANO layout: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
# (each finger: MCP→PIP→DIP→tip; tip indices are handled by _compute_fingertip_positions)
# Index 21 (extension beyond canonical MANO 21): right elbow — handled separately
# via `arm_r_link3 + _ELBOW_OFFSET_IN_LINK3_LOCAL` in `_compute_hand_kpts_pos`.
# Index 22 (extension): arm_r_link7 — last revolute link before the FIXED wrist mount;
# tracking it adds a positional constraint that indirectly constrains wrist orientation.
_MANO_NON_FT_BODY_NAMES: list[tuple[int, str]] = [
    (0,  "hx5_d20_right_base"),  # wrist
    (1,  "finger_r_link2"),      (5,  "finger_r_link6"),   (9,  "finger_r_link10"),
    (13, "finger_r_link14"),     (17, "finger_r_link18"),  # MCP joints
    (2,  "finger_r_link3"),      (6,  "finger_r_link7"),   (10, "finger_r_link11"),
    (14, "finger_r_link15"),     (18, "finger_r_link19"),  # PIP joints
    (3,  "finger_r_link4"),      (7,  "finger_r_link8"),   (11, "finger_r_link12"),
    (15, "finger_r_link16"),     (19, "finger_r_link20"),  # DIP joints (body origin, no offset)
    # NOTE: kpt 21 (elbow) is handled separately — see _compute_hand_kpts_pos.
    (22, "arm_r_link7"),                                    # last arm link (23rd kpt extension)
]
_MANO_FT_INDICES = [4, 8, 12, 16, 20]  # tip MANO indices → ft_pos[:, 0:5]
_NUM_KPTS = 23  # 21 MANO + elbow (21) + arm_r_link7 (22)


class RobotisSh5GraspMarlEnv(DirectMARLEnv):
    """Dexterous grasping with FFW-SH5 using OakInk kinematic references — MARL variant.

    Two agents: ``arm`` (7D) and ``hand`` (20D). See module docstring for details.
    """

    cfg: RobotisSh5GraspMarlEnvCfg

    def __init__(self, cfg: RobotisSh5GraspMarlEnvCfg, render_mode: str | None = None, **kwargs):
        self._load_reference_trajectories(cfg)
        # Mass-as-action removed: still call mass loader to honor JSON-based object_mass
        # but disable any per-episode sampling (handled via _build_object_cfg only).
        self._apply_object_mass_from_json(cfg)
        self._object_cfg = self._build_object_cfg(cfg)
        # Episode length follows TJ's split (gr_env_cfg.py vs gr_env_cfg_play.py).
        # TJ computes: num_frame_chunk = min(action_fps * episode_length_s, episode_length).
        # We do the same: derive num_frame_chunk from cfg.episode_length_s (default 5.0s).
        #   - Training (adaptive_sampling=True): chunked at num_frame_chunk frames.
        #   - Rollout / play (adaptive_sampling=False): no chunking; full trajectory.
        action_fps = round(1.0 / (cfg.sim.dt * cfg.decimation))
        chunk_from_cfg = round(cfg.episode_length_s * action_fps)
        if cfg.adaptive_sampling:
            self._num_frame_chunk = min(chunk_from_cfg, self._max_traj_len)
        else:
            self._num_frame_chunk = self._max_traj_len
        cfg.episode_length_s = self._num_frame_chunk / action_fps
        super().__init__(cfg, render_mode, **kwargs)
        self._post_init_buffers()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_reference_trajectories(self, cfg: RobotisSh5GraspEnvCfg) -> None:
        """Load trajectory_keypoints.npz file(s) for the configured object and trajectory."""
        _data_root = Path(cfg.hocap_data_dir if cfg.dataset == "hocap" else cfg.oakink_data_dir)
        data_dir = _data_root / "mano" / "right"

        if cfg.trajectory_task:
            # Load a single specific trajectory
            traj_path = data_dir / cfg.trajectory_task / str(cfg.trajectory_data_id) / "trajectory_keypoints.npz"
            if not traj_path.exists():
                raise FileNotFoundError(
                    f"Trajectory not found: {traj_path}\n"
                    f"Check trajectory_task='{cfg.trajectory_task}' and trajectory_data_id={cfg.trajectory_data_id}."
                )
            traj_files = [traj_path]
        else:
            # Fall back: load all trajectories matching object_id
            traj_files = sorted(data_dir.glob("*/*/trajectory_keypoints.npz"))
            traj_files = [p for p in traj_files if cfg.object_id in p.parent.parent.name]

        if not traj_files:
            raise FileNotFoundError(
                f"No trajectories found for object '{cfg.object_id}' in {data_dir}. "
                "Run scripts/process_dataset/dataset/oakink.py first."
            )

        wrist_pos_list, wrist_quat_list = [], []
        ft_pos_list = []
        obj_pos_list, obj_quat_list = [], []
        future_contact_list = []
        mano_kpts_list = []

        for path in traj_files:
            data = np.load(str(path))
            wp = data["qpos_wrist_right"][:, :3].astype(np.float32)
            wq = data["qpos_wrist_right"][:, 3:].astype(np.float32)   # wxyz
            fp = data["qpos_finger_right"][:, :, :3].astype(np.float32)  # (N, 5, 3)
            op = data["qpos_obj_right"][:, :3].astype(np.float32)
            oq = data["qpos_obj_right"][:, 3:].astype(np.float32)     # wxyz
            if "mano_kpts_right" in data:
                kp_mano = data["mano_kpts_right"].astype(np.float32)  # (N, 21, 3)
            else:
                kp_mano = np.zeros((wp.shape[0], 21, 3), dtype=np.float32)
                print(f"[warn] mano_kpts_right missing in {path}. Re-run oakink.py --overwrite.")

            # 22nd keypoint: right elbow from process_arm_pipeline.py — saved in the same
            # RAW frame as mano_kpts_right, so it goes through canonicalization uniformly.
            # 22nd, 23rd keypoints: right elbow + arm_r_link7 from process_arm_pipeline.py
            # (`arm_keypoints.npz` with `elbow_pos`, `link7_pos`).
            arm_kp_path = path.parent / "arm_keypoints.npz"
            if arm_kp_path.exists():
                arm_kp = np.load(str(arm_kp_path))
                kp_elbow = arm_kp["elbow_pos"].astype(np.float32).reshape(-1, 1, 3)
                kp_link7 = arm_kp["link7_pos"].astype(np.float32).reshape(-1, 1, 3)
                if kp_elbow.shape[0] != kp_mano.shape[0] or kp_link7.shape[0] != kp_mano.shape[0]:
                    raise ValueError(
                        f"arm_keypoints.npz lengths (elbow={kp_elbow.shape[0]}, link7={kp_link7.shape[0]}) "
                        f"!= mano_kpts length {kp_mano.shape[0]} in {path.parent}"
                    )
            else:
                kp_elbow = np.zeros((wp.shape[0], 1, 3), dtype=np.float32)
                kp_link7 = np.zeros((wp.shape[0], 1, 3), dtype=np.float32)
                print(f"[warn] arm_keypoints.npz missing at {arm_kp_path}; using zeros. "
                      "Run scripts/process_dataset/retarget/process_arm_pipeline.py first.")
            kp = np.concatenate([kp_mano, kp_elbow, kp_link7], axis=1)   # (N, 23, 3)

            # future_contact mirrors GR env is_contact:
            #   1) object is being moved (linvel > 0.05 m/s)
            #   2) fingertip is near the object surface (dist < contact_dist_threshold)
            # Object linear velocity via finite differences of reference positions.
            action_fps = round(1.0 / (cfg.sim.dt * cfg.decimation))
            obj_linvel = np.concatenate(
                [(op[1:] - op[:-1]) * action_fps, np.zeros((1, 3), dtype=np.float32)], axis=0
            )  # (N, 3)
            obj_speed = np.linalg.norm(obj_linvel, axis=-1)           # (N,) m/s
            # Actual angular velocity magnitude (rad/s) via quaternion geodesic angle.
            _dot_q = np.abs((oq[:-1] * oq[1:]).sum(axis=-1)).clip(0.0, 1.0)
            obj_angspeed = np.concatenate(
                [2.0 * np.arccos(_dot_q) * action_fps, np.array([0.0], dtype=np.float32)]
            )  # (N,) rad/s
            # GR env: contact if (linvel > 0.05 m/s OR angvel > 0.25 rad/s) AND fingertip near object
            velocity_condition = (obj_speed > 0.05) | (obj_angspeed > 0.25)  # (N,)
            dist = np.linalg.norm(fp - op[:, None, :], axis=-1)       # (N, 5)
            near_object = dist < cfg.contact_dist_threshold            # (N, 5)
            fc = (velocity_condition[:, None] & near_object).astype(np.float32)

            wrist_pos_list.append(wp)
            wrist_quat_list.append(wq)
            ft_pos_list.append(fp)
            obj_pos_list.append(op)
            obj_quat_list.append(oq)
            future_contact_list.append(fc)
            mano_kpts_list.append(kp)

        # Normalize all trajectories so that the object bottom rests on the table surface.
        # We need the mesh Z minimum *after applying frame-0 rotation* to offset the centroid
        # above the table. Using the canonical (un-rotated) mesh_z_min is wrong when frame 0
        # has the object pre-rotated — e.g., C22001-0001-0010 starts with the object lying
        # on its side, which floats ~13 cm if we use canonical mesh_z_min.
        # Shift wrist and fingertip positions by the same 3D offset to preserve
        # the relative geometry between hand and object.
        mesh_path = _data_root / "assets" / "objects" / cfg.object_id / "visual.obj"
        if mesh_path.exists():
            _mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
            _mesh_verts = np.array(_mesh.vertices, dtype=np.float32)  # (V, 3) canonical
        else:
            _mesh_verts = None
            print(f"[warn] Centered mesh not found at {mesh_path}; assuming mesh_z_min=0.")

        table_surface_z = float(cfg.table_size[2])

        def _rotate_verts_by_quat(verts: np.ndarray, q: np.ndarray) -> np.ndarray:
            """Rotate (V,3) verts by quaternion q (wxyz, Hamilton)."""
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1-2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x + y*y)],
            ], dtype=np.float32)
            return verts @ R.T

        for i in range(len(obj_pos_list)):
            if _mesh_verts is not None:
                # Use the rotated mesh's z_min so the actual bottom rests on the table.
                rotated_z_min = float(_rotate_verts_by_quat(_mesh_verts, obj_quat_list[i][0])[:, 2].min())
            else:
                rotated_z_min = 0.0
            target_centroid_z = table_surface_z - rotated_z_min
            table_target = np.array(
                [cfg.table_pos_env[0], cfg.table_pos_env[1], target_centroid_z],
                dtype=np.float32,
            )
            offset = table_target - obj_pos_list[i][0]
            obj_pos_list[i] = obj_pos_list[i] + offset
            wrist_pos_list[i] = wrist_pos_list[i] + offset
            ft_pos_list[i] = ft_pos_list[i] + offset
            mano_kpts_list[i] = mano_kpts_list[i] + offset
            if i == 0:
                print(f"[grasp] traj[0] rotated_z_min={rotated_z_min:.4f}, "
                      f"table_surface_z={table_surface_z:.4f}, target_centroid_z={target_centroid_z:.4f}")

        # Canonicalize approach direction:
        # Different OakInk sessions define their world frame with arbitrary XY orientation,
        # so after position-only normalization the reference wrist can point anywhere in XY.
        # Rotate each trajectory around Z (at the object's XY center) so the wrist direction
        # aligns with the robot's approach direction (object → robot in XY).
        if cfg.canonical_ref_pos_env is not None:
            ref_xy = np.array(cfg.canonical_ref_pos_env[:2], dtype=np.float32)
        else:
            # Use the center of the table edge facing the robot.
            tx, ty = float(cfg.table_pos_env[0]), float(cfg.table_pos_env[1])
            hx, hy = float(cfg.table_size[0]) / 2.0, float(cfg.table_size[1]) / 2.0
            dx = float(cfg.robot_cfg.init_state.pos[0]) - tx
            dy = float(cfg.robot_cfg.init_state.pos[1]) - ty
            if abs(dy) >= abs(dx):
                ref_xy = np.array([tx, ty + hy * np.sign(dy)], dtype=np.float32)
            else:
                ref_xy = np.array([tx + hx * np.sign(dx), ty], dtype=np.float32)
        obj_xy_ref = np.array([cfg.table_pos_env[0], cfg.table_pos_env[1]], dtype=np.float32)
        canonical_dir = ref_xy - obj_xy_ref   # direction from object toward reference in XY
        print(f"[grasp] canonical ref XY: {ref_xy}, dir: {canonical_dir}")
        canonical_norm = np.linalg.norm(canonical_dir)
        if canonical_norm > 1e-6:   
            canonical_dir /= canonical_norm

            for i in range(len(obj_pos_list)):
                o0 = obj_pos_list[i][0]
                w0 = wrist_pos_list[i][0]
                wrist_dir = w0[:2] - o0[:2]            # object → wrist in XY
                wrist_norm = np.linalg.norm(wrist_dir)
                if wrist_norm < 1e-4:
                    continue
                wrist_dir /= wrist_norm

                # Signed angle to rotate wrist_dir → canonical_dir (counter-clockwise positive)
                cos_a = float(np.clip(np.dot(wrist_dir, canonical_dir), -1.0, 1.0))
                sin_a = float(wrist_dir[0] * canonical_dir[1] - wrist_dir[1] * canonical_dir[0])
                angle = float(np.arctan2(sin_a, cos_a))

                c, s = float(np.cos(angle)), float(np.sin(angle))
                R2 = np.array([[c, -s], [s, c]], dtype=np.float32)
                ox, oy = float(o0[0]), float(o0[1])

                def _rot_pos(arr: np.ndarray, _R: np.ndarray = R2, _ox: float = ox, _oy: float = oy) -> np.ndarray:
                    flat = arr.reshape(-1, 3).copy()
                    flat[:, 0] -= _ox
                    flat[:, 1] -= _oy
                    flat[:, :2] = flat[:, :2] @ _R.T
                    flat[:, 0] += _ox
                    flat[:, 1] += _oy
                    return flat.reshape(arr.shape)

                # Rotation quaternion for R_z(angle): [cos(a/2), 0, 0, sin(a/2)] in wxyz
                hw = angle / 2.0
                q_rw = float(np.cos(hw))
                q_rz = float(np.sin(hw))

                def _rot_quat(q: np.ndarray, _w1: float = q_rw, _z1: float = q_rz) -> np.ndarray:
                    # q_rot * q_old  (Hamilton product, wxyz convention)
                    # q_rot = [_w1, 0, 0, _z1]
                    w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                    return np.stack([
                        _w1 * w2 - _z1 * z2,
                        _w1 * x2 - _z1 * y2,
                        _w1 * y2 + _z1 * x2,
                        _w1 * z2 + _z1 * w2,
                    ], axis=-1).astype(np.float32)

                obj_pos_list[i] = _rot_pos(obj_pos_list[i])
                wrist_pos_list[i] = _rot_pos(wrist_pos_list[i])
                ft_pos_list[i] = _rot_pos(ft_pos_list[i].reshape(-1, 3)).reshape(ft_pos_list[i].shape)
                mano_kpts_list[i] = _rot_pos(mano_kpts_list[i].reshape(-1, 3)).reshape(mano_kpts_list[i].shape)
                obj_quat_list[i] = _rot_quat(obj_quat_list[i])
                wrist_quat_list[i] = _rot_quat(wrist_quat_list[i])

        # --- Nearest mesh vertex per fingertip per frame (GR-style contact) ---
        # For each reference (frame, fingertip) pair, find the nearest vertex on the
        # object mesh (in object-local frame). Used for:
        #   (a) future_contact: relative distance threshold instead of absolute centroid dist
        #   (b) delta_ft_obj obs: contact fingers target nearest vertex (not ref fingertip)
        #   (c) fingertip reward: contact fingers target nearest vertex
        contact_vertex_local_list: list[np.ndarray] = []
        _action_fps_cv = round(1.0 / (cfg.sim.dt * cfg.decimation))
        if mesh_path.exists():
            _verts = np.array(_mesh.vertices, dtype=np.float32)  # (V, 3) object-local frame
            for i in range(len(obj_pos_list)):
                N_i = obj_pos_list[i].shape[0]
                cv_local = np.zeros((N_i, 5, 3), dtype=np.float32)
                cv_dist  = np.zeros((N_i, 5),    dtype=np.float32)
                for t in range(N_i):
                    op = obj_pos_list[i][t]    # (3,) world
                    oq = obj_quat_list[i][t]   # (4,) wxyz
                    fp = ft_pos_list[i][t]     # (5, 3) world
                    w, x, y, z = float(oq[0]), float(oq[1]), float(oq[2]), float(oq[3])
                    # Rotation matrix local→world (column-vector convention).
                    # For row-vector numpy: fp_local = (fp - op) @ R
                    R = np.array([
                        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
                    ], dtype=np.float32)
                    fp_local = (fp - op) @ R           # (5, 3) fingertips in obj-local
                    diff  = _verts[:, None, :] - fp_local[None, :, :]  # (V, 5, 3)
                    dists = np.linalg.norm(diff, axis=-1)               # (V, 5)
                    idxs  = np.argmin(dists, axis=0)                    # (5,)
                    cv_local[t] = _verts[idxs]                          # (5, 3)
                    cv_dist[t]  = dists[idxs, np.arange(5)]            # (5,)
                # Recompute future_contact using relative vertex distance (GR method):
                #   contact iff (linvel>0.05 m/s OR angvel>0.25 rad/s) AND (cv_dist - min_cv_dist < 0.015)
                op_arr = obj_pos_list[i]
                oq_arr = obj_quat_list[i]
                _lv = np.concatenate([(op_arr[1:]-op_arr[:-1])*_action_fps_cv,
                                      np.zeros((1,3),dtype=np.float32)], axis=0)
                # Actual angular velocity magnitude (rad/s) via quaternion geodesic angle.
                _dot_q = np.abs((oq_arr[:-1] * oq_arr[1:]).sum(axis=-1)).clip(0.0, 1.0)
                _angspeed = np.concatenate(
                    [2.0 * np.arccos(_dot_q) * _action_fps_cv, np.array([0.0], dtype=np.float32)]
                )
                vel_cond = (np.linalg.norm(_lv, axis=-1) > 0.05) | (_angspeed > 0.25)
                cv_min   = cv_dist.min(axis=-1, keepdims=True)            # (N, 1)
                near_vtx = (cv_dist - cv_min) < 0.015                    # (N, 5)
                future_contact_list[i] = (vel_cond[:, None] & near_vtx).astype(np.float32)
                contact_vertex_local_list.append(cv_local)
        else:
            print("[grasp] visual.obj not found; contact vertex data unavailable; using centroid fallback.")
            for i in range(len(obj_pos_list)):
                contact_vertex_local_list.append(
                    np.zeros((obj_pos_list[i].shape[0], 5, 3), dtype=np.float32)
                )

        max_len = max(a.shape[0] for a in wrist_pos_list)
        n_traj = len(wrist_pos_list)

        def _pad(arr_list: list[np.ndarray], shape_suffix: tuple) -> np.ndarray:
            out = np.zeros((n_traj, max_len, *shape_suffix), dtype=np.float32)
            for i, a in enumerate(arr_list):
                T = a.shape[0]
                out[i, :T] = a
                out[i, T:] = a[-1]
            return out

        self._ref_wrist_pos = torch.from_numpy(_pad(wrist_pos_list, (3,)))
        self._ref_wrist_quat = torch.from_numpy(_pad(wrist_quat_list, (4,)))
        self._ref_ft_pos = torch.from_numpy(_pad(ft_pos_list, (5, 3)))
        self._ref_obj_pos = torch.from_numpy(_pad(obj_pos_list, (3,)))
        self._ref_obj_quat = torch.from_numpy(_pad(obj_quat_list, (4,)))
        self._future_contact = torch.from_numpy(_pad(future_contact_list, (5,)))
        self._ref_contact_vertex_local = torch.from_numpy(_pad(contact_vertex_local_list, (5, 3)))
        self._ref_mano_kpts = torch.from_numpy(_pad(mano_kpts_list, (_NUM_KPTS, 3)))
        self._traj_lengths = torch.tensor([a.shape[0] for a in wrist_pos_list], dtype=torch.long)
        self._max_traj_len = max_len
        self._n_trajs = n_traj

        # Load frame-0 arm pose from full-trajectory IK (`arm_joint_pos.npy[0]`,
        # generated by scripts/process_dataset/retarget/process_arm_pipeline.py).
        frame0_arm_list = []
        for path in traj_files:
            ik_path = path.parent / "arm_joint_pos.npy"
            if ik_path.exists():
                frame0_arm_list.append(np.load(str(ik_path))[0].astype(np.float32))
            else:
                frame0_arm_list.append(None)

        if all(x is not None for x in frame0_arm_list):
            self._frame0_arm_joint_pos = torch.from_numpy(np.stack(frame0_arm_list, axis=0))  # (n_trajs, 7)
            print(f"[grasp] Loaded frame-0 arm IK from arm_joint_pos.npy for {n_traj} trajectories.")
        else:
            self._frame0_arm_joint_pos = None
            missing = sum(1 for x in frame0_arm_list if x is None)
            print(f"[grasp] {missing}/{n_traj} arm_joint_pos.npy files missing; arm stays at default on cache miss.")

        print(f"[grasp] Loaded {n_traj} trajectories for '{cfg.object_id}', max_len={max_len}")

    @staticmethod
    def _apply_object_mass_from_json(cfg: RobotisSh5GraspMarlEnvCfg) -> None:
        """Load [lo, hi] mass range from the per-object JSON if available.

        Always populates `cfg.object_mass` (= midpoint, used as static fallback) AND
        `cfg.object_mass_min` / `cfg.object_mass_max` (used by mass-in-the-loop sampling
        when `cfg.enable_mass_in_loop=True`).
        """
        _data_root = Path(cfg.hocap_data_dir if cfg.dataset == "hocap" else cfg.oakink_data_dir)
        if cfg.object_mass_json:
            json_path = Path(cfg.object_mass_json)
        else:
            json_path = _data_root / "object_mass.json"
        if not json_path.exists():
            return
        try:
            with open(json_path) as f:
                mass_table: dict = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[grasp-marl] WARNING: Could not load object_mass_json ({json_path}): {e}")
            return

        entry = mass_table.get(cfg.object_id)
        if entry is None:
            print(f"[grasp-marl] object_mass: {cfg.object_id} not in JSON → keep cfg default {cfg.object_mass:.3f} kg")
            return
        if entry[0] is None or entry[1] is None:
            print(f"[grasp-marl] object_mass: {cfg.object_id} has null in JSON → keep cfg default {cfg.object_mass:.3f} kg")
            return
        lo, hi = float(entry[0]), float(entry[1])
        cfg.object_mass = 0.5 * (lo + hi)
        cfg.object_mass_min = lo
        cfg.object_mass_max = hi
        loop_note = " (mass-in-loop sampling range)" if cfg.enable_mass_in_loop else ""
        print(f"[grasp-marl] object_mass from JSON: {cfg.object_id} → "
              f"static {cfg.object_mass:.3f} kg / range [{lo:.3f}, {hi:.3f}]{loop_note}")

    def _build_object_cfg(self, cfg: RobotisSh5GraspEnvCfg) -> RigidObjectCfg:
        _data_root = Path(cfg.hocap_data_dir if cfg.dataset == "hocap" else cfg.oakink_data_dir)
        usd_path = _data_root / "assets" / "objects" / cfg.object_id / "visual.usd"
        if not usd_path.exists():
            raise FileNotFoundError(
                f"Object USD not found: {usd_path}\n"
                f"Run: isaaclab.sh -p scripts/process_dataset/assets/convert_obj_to_usd.py "
                f"--dataset {cfg.dataset} --object-id {cfg.object_id}"
            )
        return RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=cfg.object_mass),
                # collision_props omitted: geometry is an instanced prim in the USD,
                # so Isaac Lab cannot override it here — collision is defined in the USD itself.
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(cfg.table_pos_env[0], cfg.table_pos_env[1], cfg.table_size[2] + 0.15),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

    @property
    def is_reached_end(self) -> bool:
        """True when the policy has reached near the end of the trajectory (mirrors GR: reached_frame >= episode_length - 3)."""
        return self._reached_frame >= self._max_traj_len - 3

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Static table: two stacked cuboids — base body + tabletop slab (overhangs +y
        # toward robot so lift-and-place trajectories stay over the surface; base body
        # keeps original footprint so robot torso fits under the overhang).
        table_w, table_d, table_h = self.cfg.table_size
        table_x, table_y, _ = self.cfg.table_pos_env
        thickness = float(self.cfg.tabletop_thickness)
        overhang_y = float(self.cfg.tabletop_overhang_y_pos)
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.38, 0.18))

        # Kinematic rigid-body table prims (required for ContactSensor GPU filter — see env.py for rationale).
        _table_rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        _table_mass_props = sim_utils.MassPropertiesCfg(mass=1.0)
        base_h = table_h - thickness
        base_spawner = sim_utils.CuboidCfg(
            size=(table_w, table_d, base_h),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=_table_rigid_props,
            mass_props=_table_mass_props,
            visual_material=mat,
        )
        base_spawner.func(
            "/World/envs/env_.*/TableBase",
            base_spawner,
            translation=(table_x, table_y, base_h / 2),
        )

        top_d = table_d + overhang_y
        top_y = table_y + overhang_y / 2.0
        top_spawner = sim_utils.CuboidCfg(
            size=(table_w, top_d, thickness),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=_table_rigid_props,
            mass_props=_table_mass_props,
            visual_material=mat,
        )
        top_spawner.func(
            "/World/envs/env_.*/TableTop",
            top_spawner,
            translation=(table_x, top_y, table_h - thickness / 2),
        )

        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self._object_cfg)

        # Contact sensors on right-hand fingertip links only
        contact_cfgs = {}
        for link_name in self.cfg.fingertip_body_names:
            contact_cfgs[link_name] = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                update_period=0.0,
                history_length=3,
                debug_vis=False,
                track_pose=False,
                track_air_time=False,
                # ── [THUMB-RADIUS-FILTER] ───────────────────────────────
                # Enables `contact_pos_w` (avg contact-point position per
                # (body, filter) pair) — required by the thumb-only spatial
                # gate in `_get_fingertip_forces`. The flag applies to all
                # fingertip sensors but the gating logic is thumb-specific
                # (other fingers' tip link is short, no filtering needed).
                #
                # `max_contact_data_count_per_prim` is intentionally left at
                # the cfg default (4); we only consume the averaged position.
                #
                # To revert: remove this line (default=False → contact_pos_w
                # becomes None and the gate in `_get_fingertip_forces` is
                # bypassed → original unfiltered force behavior restored).
                # ────────────────────────────────────────────────────────
                track_contact_points=True,
            )

        self._contact_sensors: dict[str, ContactSensor] = {}
        for name, cfg in contact_cfgs.items():
            sensor = ContactSensor(cfg)
            self._contact_sensors[name] = sensor
            self.scene.sensors[f"contact_{name}"] = sensor

        # ── Arm ↔ Table contact sensors (anti-cheating; see `rew_arm_contact`) ──
        # Track arm_r_link3..link7 vs table; MAX magnitude drives penalty + termination.
        _ARM_CONTACT_LINK_NAMES = [
            "arm_r_link3", "arm_r_link4", "arm_r_link5", "arm_r_link6", "arm_r_link7",
        ]
        self._arm_contact_sensors: dict[str, ContactSensor] = {}
        for link_name in _ARM_CONTACT_LINK_NAMES:
            arm_contact_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                filter_prim_paths_expr=[
                    "/World/envs/env_.*/TableBase",
                    "/World/envs/env_.*/TableTop",
                ],
                update_period=0.0,
                history_length=3,
                debug_vis=False,
                track_pose=False,
                track_air_time=False,
            )
            sensor = ContactSensor(arm_contact_cfg)
            self._arm_contact_sensors[link_name] = sensor
            self.scene.sensors[f"contact_{link_name}"] = sensor

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.cfg.debug_vis:
            self._setup_debug_vis()

    def _post_init_buffers(self) -> None:
        """Resolve joint/body indices and allocate per-env buffers after scene is ready."""
        for attr in (
            "_ref_wrist_pos", "_ref_wrist_quat", "_ref_ft_pos",
            "_ref_obj_pos", "_ref_obj_quat", "_future_contact",
            "_ref_contact_vertex_local", "_ref_mano_kpts", "_traj_lengths",
        ):
            setattr(self, attr, getattr(self, attr).to(self.device))

        if self._frame0_arm_joint_pos is not None:
            self._frame0_arm_joint_pos = self._frame0_arm_joint_pos.to(self.device)

        # Joint indices for each controlled group
        self._finger_joint_ids, _ = self.robot.find_joints(self.cfg.finger_joint_names)
        self._arm_r_joint_ids, _ = self.robot.find_joints(self.cfg.arm_r_joint_names)
        self._lift_joint_ids, _ = self.robot.find_joints(self.cfg.lift_joint_name)

        if len(self._finger_joint_ids) != self.cfg.num_hand_dofs:
            raise ValueError(
                f"Found {len(self._finger_joint_ids)} finger joints matching '{self.cfg.finger_joint_names}', "
                f"expected cfg.num_hand_dofs={self.cfg.num_hand_dofs}."
            )
        if len(self._arm_r_joint_ids) != self.cfg.num_arm_r_dofs:
            raise ValueError(
                f"Found {len(self._arm_r_joint_ids)} arm joints matching '{self.cfg.arm_r_joint_names}', "
                f"expected cfg.num_arm_r_dofs={self.cfg.num_arm_r_dofs}."
            )
        if len(self._lift_joint_ids) != self.cfg.num_lift_dofs:
            raise ValueError(
                f"Found {len(self._lift_joint_ids)} lift joints matching '{self.cfg.lift_joint_name}', "
                f"expected cfg.num_lift_dofs={self.cfg.num_lift_dofs}."
            )

        # Fingertip body indices
        self._ft_body_ids = self._resolve_fingertip_ids()
        if len(self._ft_body_ids) != 5:
            print(
                f"[warn] Found {len(self._ft_body_ids)} fingertip bodies; expected 5. "
                "Check cfg.fingertip_body_names."
            )

        # Pre-build per-fingertip local-frame offset tensors (shape: [5, 3])
        # Order matches cfg.fingertip_body_names.
        offsets, pad_normals = [], []
        for name in self.cfg.fingertip_body_names:
            offsets.append(_FINGERTIP_OFFSETS.get(name, [0.0, 0.0, 0.0]))
            pad_normals.append(_FINGERTIP_PAD_NORMALS.get(name, [0.0, 0.0, 1.0]))
        # (5, 3) tip-position offset and pad-outward unit normal, both in link local frame
        self._ft_offsets = torch.tensor(offsets, dtype=torch.float32, device=self.device)
        self._ft_pad_normals = torch.tensor(pad_normals, dtype=torch.float32, device=self.device)

        # Wrist body index (needed for rotation tracking)
        wrist_ids, _ = self.robot.find_bodies(self.cfg.wrist_body_name)
        self._wrist_body_id: int | None = wrist_ids[0] if wrist_ids else None
        if self._wrist_body_id is None:
            print(f"[warn] Wrist body '{self.cfg.wrist_body_name}' not found; wrist rotation tracking disabled.")

        # Elbow position: arm_r_link3 + URDF joint4 offset (stable across joint4 rotation).
        link3_ids, _ = self.robot.find_bodies("arm_r_link3")
        self._link3_body_id: int | None = link3_ids[0] if link3_ids else None
        if self._link3_body_id is None:
            print("[warn] arm_r_link3 not found; elbow position fallback to zero.")
        self._elbow_offset_local = torch.tensor(
            _ELBOW_OFFSET_IN_LINK3_LOCAL, dtype=torch.float32, device=self.device
        )

        # Resolve body IDs for all 16 non-fingertip MANO keypoints
        _kpt_mano_indices, _kpt_body_ids = [], []
        for mano_idx, body_name in _MANO_NON_FT_BODY_NAMES:
            body_ids, _ = self.robot.find_bodies(body_name)
            if body_ids:
                _kpt_mano_indices.append(mano_idx)
                _kpt_body_ids.append(body_ids[0])
            else:
                print(f"[warn] MANO kpt {mano_idx}: body '{body_name}' not found, skipped.")
        self._kpt_body_ids_t = torch.tensor(_kpt_body_ids, dtype=torch.long, device=self.device)
        self._kpt_mano_indices_t = torch.tensor(_kpt_mano_indices, dtype=torch.long, device=self.device)
        self._kpt_ft_mano_indices_t = torch.tensor(_MANO_FT_INDICES, dtype=torch.long, device=self.device)

        # Joint ID groups:
        # - `_all_joint_ids` (28): used for observation (joint_pos/joint_vel include lift)
        #   and state-cache restoration (must capture full controlled state including lift).
        # - `_action_joint_ids` (27): subset used by the action — lift is excluded and held
        #   at `cfg.fixed_lift_target` by the PD controller.
        self._all_joint_ids = self._finger_joint_ids + self._arm_r_joint_ids + self._lift_joint_ids
        self._action_joint_ids = self._finger_joint_ids + self._arm_r_joint_ids

        # Joint limits for normalization (GR env style: scale [-1,1] → [lower, upper]).
        # Two variants:
        #   _ctrl_lower/_ctrl_upper       (27,) — action joints only; used by _scale/_unscale
        #                                          for action targets and obs[joint_pos] slice
        #                                          covering action dims.
        #   _ctrl_lower_all/_ctrl_upper_all (28,) — full controlled-joint set incl. lift;
        #                                          used by _unscale_all for the 28D joint_pos
        #                                          slice of the observation.
        dof_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        action_ids_t = torch.tensor(self._action_joint_ids, dtype=torch.long, device=self.device)
        all_ids_t = torch.tensor(self._all_joint_ids, dtype=torch.long, device=self.device)
        self._ctrl_lower = dof_limits[0, action_ids_t, 0]  # (27,)
        self._ctrl_upper = dof_limits[0, action_ids_t, 1]  # (27,)
        self._ctrl_lower_all = dof_limits[0, all_ids_t, 0]  # (28,)
        self._ctrl_upper_all = dof_limits[0, all_ids_t, 1]  # (28,)

        # Lift target tensor (broadcast each step in _apply_action).
        self._lift_joint_ids_t = torch.tensor(self._lift_joint_ids, dtype=torch.long, device=self.device)
        self._lift_target = torch.full(
            (self.num_envs, len(self._lift_joint_ids)), float(self.cfg.fixed_lift_target),
            device=self.device,
        )
        # Zero velocity for lift (used by write_joint_state_to_sim to kill any
        # accumulated joint vel from PD integration — see _apply_action).
        self._lift_zero_vel = torch.zeros_like(self._lift_target)

        # Per-env buffers
        B = self.num_envs
        self._traj_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        self._frame_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        # Per-agent previous actions (used in obs + arm action-rate penalty).
        self._prev_arm_action = torch.zeros(B, self.cfg.num_arm_r_dofs, device=self.device)
        self._prev_hand_action = torch.zeros(B, self.cfg.num_hand_dofs, device=self.device)

        # EMA action smoothing buffer (actioned joints only, 27D); initialize at normalized default pose
        default_ctrl = torch.cat([
            self.robot.data.default_joint_pos[:1, self._finger_joint_ids],
            self.robot.data.default_joint_pos[:1, self._arm_r_joint_ids],
        ], dim=-1).squeeze(0)
        default_normalized = self._unscale(default_ctrl)
        self._smoothed_actions = default_normalized.unsqueeze(0).expand(B, -1).clone()

        # Adaptive rollout sampling: per-frame EMA failure count (start at zero)
        self._failure_count = torch.zeros(self._max_traj_len, device=self.device)
        # Rewind window in frames — derived from action_fps × adaptive_back_seconds (mirrors TJ).
        _action_fps = round(1.0 / (self.cfg.sim.dt * self.cfg.decimation))
        self._adaptive_back_frames: int = int(_action_fps * self.cfg.adaptive_back_seconds)

        # State cache: stores simulation state at each trajectory frame for physical curriculum.
        # Layout (97-dim): reward(1) + obj_pos_local(3) + obj_quat(4) + obj_linvel(3) + obj_angvel(3)
        #                  + joint_pos(28) + joint_vel(28) + smoothed_act(27, lift excluded)
        _STATE_DIM = 97
        self._state_cache = torch.zeros(self._max_traj_len, _STATE_DIM, device=self.device)
        self._state_cache[:, 0] = -float("inf")  # reward col: any real reward beats -inf
        self._init_flg = torch.ones(self._max_traj_len, dtype=torch.bool, device=self.device)  # True = ref data
        self._reached_frame: int = 0  # furthest frame with sustained good tracking
        # TJ-style: force-save frame 0 cache once on first reset so subsequent resets reuse the IK-lifted pose.
        self._init_save_done: bool = False

        # Effort-saturation diagnostic buffers (cfg-gated).
        self._sat_acc = torch.zeros(self.robot.num_joints, device=self.device)
        self._sat_step_count: int = 0

        # Per-episode tracking quality (for enough_idx and reached_frame update)
        self._enough_continued = torch.ones(B, dtype=torch.bool, device=self.device)
        self._enough_idx = torch.zeros(B, dtype=torch.long, device=self.device)

        # Episode-done flag (set in _get_dones, cleared in _reset_idx)
        self._done_env = torch.zeros(B, dtype=torch.bool, device=self.device)

        # Mass-in-the-loop distribution module (registered by train_marl.py if enabled).
        # When None, env uses static cfg.object_mass loaded into the object spawn.
        self._mass_dist = None
        # Per-step snapshots of (mass_action, log_prob_old) — captured at the START
        # of each step in _pre_physics_step so they reflect the mass USED during
        # this step (resampling happens at end-of-step in _reset_idx).
        # Allocated lazily in _pre_physics_step once a mass_dist is registered.
        self._mass_action_step: torch.Tensor | None = None
        self._mass_log_prob_old_step: torch.Tensor | None = None

        # ── WARMUP ────────────────────────────────────────────────────────────
        # Per-env warm-up flag. True = hand has not yet reached start-frame target.
        # During warm-up: _frame_idx is frozen and early termination is disabled.
        # Initialized to True so the very first episode always goes through warm-up.
        # To restore original behavior: remove this line and all [WARMUP] blocks below.
        self._is_warming_up = torch.ones(B, dtype=torch.bool, device=self.device)
        # ── END WARMUP ────────────────────────────────────────────────────────

        # Shared buffers between _get_rewards() and _get_dones() to avoid double computation.
        self._early_terminate_buf = torch.zeros(B, dtype=torch.bool, device=self.device)
        self._last_ft_mean_err = torch.zeros(B, device=self.device)
        self._last_wrist_err = torch.zeros(B, device=self.device)
        self._last_wrist_rot_err = torch.zeros(B, device=self.device)
        self._last_obj_pos_err = torch.zeros(B, device=self.device)
        self._last_obj_rot_err = torch.zeros(B, device=self.device)  # rad; for evaluation E_r metric
        self._last_kpts_err = torch.zeros(B, device=self.device)  # drift-compensated; used for warmup/termination
        self._last_kpts_err_raw = torch.zeros(B, device=self.device)  # raw ref vs robot kpts (m); for evaluation E_j metric
        self._last_ft_raw_err = torch.zeros(B, device=self.device)  # raw ref vs robot fingertips (m); for evaluation E_ft metric

    def _log_effort_saturation(self) -> None:
        """Diagnostic: print joints whose applied torque hits ≥99 % of effort_limit."""
        if not self.cfg.log_effort_saturation:
            return
        applied = self.robot.data.applied_torque
        limits = self.robot.data.joint_effort_limits
        sat_mask = applied.abs() >= 0.99 * limits.clamp(min=1e-6)
        self._sat_acc += sat_mask.float().mean(dim=0)
        self._sat_step_count += 1
        if self._sat_step_count >= self.cfg.effort_saturation_log_interval:
            avg_ratio = self._sat_acc / self._sat_step_count
            high_idx = (avg_ratio > 0.01).nonzero(as_tuple=True)[0]
            if len(high_idx) > 0:
                print(f"\n[effort_saturation] over last {self._sat_step_count} steps "
                      f"(joints with >1% env-avg saturation):")
                names = self.robot.joint_names
                for idx in high_idx.tolist():
                    print(f"  {names[idx]:30s}  {avg_ratio[idx].item()*100:5.1f}%  "
                          f"(effort_limit={limits[0, idx].item():6.2f} N·m)")
            self._sat_acc.zero_()
            self._sat_step_count = 0

    def _resolve_fingertip_ids(self) -> torch.Tensor:
        ids = []
        for name in self.cfg.fingertip_body_names:
            found, _ = self.robot.find_bodies(name)
            if found:
                ids.append(found[0])
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalized actions [-1, 1] to action-joint positions [lower, upper] (27D)."""
        return 0.5 * (x + 1.0) * (self._ctrl_upper - self._ctrl_lower) + self._ctrl_lower

    def _unscale(self, q: torch.Tensor) -> torch.Tensor:
        """Map action-joint positions [lower, upper] to normalized [-1, 1] (27D)."""
        return (2.0 * q - self._ctrl_upper - self._ctrl_lower) / (self._ctrl_upper - self._ctrl_lower).clamp(min=1e-6)

    def _unscale_all(self, q: torch.Tensor) -> torch.Tensor:
        """Map full controlled-joint positions (incl. lift) [lower, upper] to [-1, 1] (28D)."""
        return (2.0 * q - self._ctrl_upper_all - self._ctrl_lower_all) / (self._ctrl_upper_all - self._ctrl_lower_all).clamp(min=1e-6)

    def _compute_fingertip_positions(self) -> torch.Tensor:
        """Compute world-space virtual fingertip positions by applying local offsets.

        Each finger link's prim origin does not coincide with the actual contact tip.
        We rotate the pre-measured local offset by the link's world orientation and
        add it to the link origin position, mirroring get_virtual_link_poses() from
        the pick_and_place mdp utils.

        Returns:
            ft_pos_w: (B, 5, 3) world-space fingertip positions.
        """
        B = self.num_envs
        if len(self._ft_body_ids) != 5:
            return torch.zeros(B, 5, 3, device=self.device)

        # link_pos: (B, 5, 3),  link_quat: (B, 5, 4)
        link_pos = self.robot.data.body_pos_w[:, self._ft_body_ids, :]   # (B, 5, 3)
        link_quat = self.robot.data.body_quat_w[:, self._ft_body_ids, :] # (B, 5, 4)

        # Expand offsets to (B, 5, 3) and rotate into world frame
        offsets = self._ft_offsets.unsqueeze(0).expand(B, -1, -1)        # (B, 5, 3)
        rotated = quat_apply(
            link_quat.reshape(B * 5, 4),
            offsets.reshape(B * 5, 3),
        ).reshape(B, 5, 3)

        return link_pos + rotated

    def _compute_fingertip_pad_normals_w(self) -> torch.Tensor:
        """Compute world-space pad-outward unit normals for each fingertip link.

        Local-frame pad normals are stored in `self._ft_pad_normals` (5, 3) —
        identified via `scripts/process_dataset/visualize_fingertip_normals.py`:
            thumb (link4)        : -Z local
            others (link8/12/16/20): -X local

        Used for contact-force projection (mirrors TJ):
            force_along_pad = (force_w * -pad_normal_w).sum(-1).clamp_min(0)

        Returns:
            pad_normals_w: (B, 5, 3) unit vectors in world frame.
        """
        B = self.num_envs
        if len(self._ft_body_ids) != 5:
            return torch.zeros(B, 5, 3, device=self.device)

        link_quat = self.robot.data.body_quat_w[:, self._ft_body_ids, :]      # (B, 5, 4)
        pad_local = self._ft_pad_normals.unsqueeze(0).expand(B, -1, -1)       # (B, 5, 3)
        return quat_apply(
            link_quat.reshape(B * 5, 4),
            pad_local.reshape(B * 5, 3),
        ).reshape(B, 5, 3)

    def _compute_hand_kpts_pos(self) -> torch.Tensor:
        """Compute world-space positions for all 23 keypoints
        (21 MANO + elbow at idx 21 + arm_r_link7 at idx 22).

        - MANO non-fingertip joints + link7: body link origins (direct `body_pos_w`).
        - Fingertips (MANO 4, 8, 12, 16, 20): link origin + local offset.
        - Elbow (MANO 21): arm_r_link3 origin + URDF joint4 offset, rotated — stays
          stable under joint4 rotation (unlike `body_pos_w[arm_r_link4]`).

        Returns:
            hand_kpts_pos: (B, 23, 3) world-space keypoint positions.
        """
        B = self.num_envs
        hand_kpts_pos = torch.zeros(B, _NUM_KPTS, 3, device=self.device)
        # Non-fingertip (excl. elbow): body link origins (no offset)
        body_pos = self.robot.data.body_pos_w[:, self._kpt_body_ids_t, :]
        hand_kpts_pos[:, self._kpt_mano_indices_t, :] = body_pos
        # Fingertip: link origin + local offset
        ft_pos = self._compute_fingertip_positions()
        hand_kpts_pos[:, self._kpt_ft_mano_indices_t, :] = ft_pos
        # Elbow (kpt 21): arm_r_link3 origin + URDF joint4 offset, rotated.
        if self._link3_body_id is not None:
            link3_pos = self.robot.data.body_pos_w[:, self._link3_body_id, :]
            link3_quat = self.robot.data.body_quat_w[:, self._link3_body_id, :]
            elbow_offset_b = self._elbow_offset_local.unsqueeze(0).expand(B, -1)
            hand_kpts_pos[:, 21, :] = link3_pos + quat_apply(link3_quat, elbow_offset_b)
        return hand_kpts_pos

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def _setup_debug_vis(self) -> None:
        """Create VisualizationMarkers for reference fingertips, wrist, and elbow."""
        n = min(self.cfg.debug_vis_num_envs, self.num_envs)

        def _sphere_marker_cfg(prim_path: str, radius: float, color: tuple) -> VisualizationMarkersCfg:
            return VisualizationMarkersCfg(
                prim_path=prim_path,
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    )
                },
            )

        # Reference fingertip positions — green spheres
        self._vis_ref_ft = VisualizationMarkers(
            _sphere_marker_cfg("/Visuals/debug/ref_fingertips", 0.012, (0.0, 1.0, 0.0))
        )
        # Actual (virtual) fingertip positions — cyan spheres
        self._vis_actual_ft = VisualizationMarkers(
            _sphere_marker_cfg("/Visuals/debug/actual_fingertips", 0.010, (0.0, 0.8, 1.0))
        )
        # Reference wrist position — magenta sphere
        self._vis_ref_wrist = VisualizationMarkers(
            _sphere_marker_cfg("/Visuals/debug/ref_wrist", 0.020, (1.0, 0.0, 1.0))
        )
        # Reference elbow position — orange sphere
        self._vis_ref_elbow = VisualizationMarkers(
            _sphere_marker_cfg("/Visuals/debug/ref_elbow", 0.020, (1.0, 0.5, 0.0))
        )
        # Reference arm_r_link7 position — yellow sphere
        self._vis_ref_link7 = VisualizationMarkers(
            _sphere_marker_cfg("/Visuals/debug/ref_link7", 0.020, (1.0, 1.0, 0.0))
        )

        self._debug_vis_n = n
        print(f"[grasp] Debug vis enabled for first {n} envs.")

    def _update_debug_vis(
        self,
        ref_ft_pos: torch.Tensor,
        ft_pos: torch.Tensor,
        ref_wrist_pos: torch.Tensor,
        ref_elbow_pos: torch.Tensor,
        ref_link7_pos: torch.Tensor,
    ) -> None:
        """Update debug markers every observation step.

        Args:
            ref_ft_pos:    (B, 5, 3) reference fingertip world positions.
            ft_pos:        (B, 5, 3) actual virtual fingertip world positions.
            ref_wrist_pos: (B, 3)   reference wrist world position.
            ref_elbow_pos: (B, 3)   reference elbow world position (kpt index 21).
            ref_link7_pos: (B, 3)   reference arm_r_link7 world position (kpt index 22).
        """
        n = self._debug_vis_n

        # Reference fingertips: (n*5, 3)
        self._vis_ref_ft.visualize(translations=ref_ft_pos[:n].reshape(n * 5, 3))

        # Actual fingertips: (n*5, 3)
        self._vis_actual_ft.visualize(translations=ft_pos[:n].reshape(n * 5, 3))

        # Reference wrist position: n magenta spheres
        self._vis_ref_wrist.visualize(translations=ref_wrist_pos[:n])

        # Reference elbow position: n orange spheres
        self._vis_ref_elbow.visualize(translations=ref_elbow_pos[:n])

        # Reference arm_r_link7 position: n yellow spheres
        self._vis_ref_link7.visualize(translations=ref_link7_pos[:n])

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """MARL: actions is a dict {"arm": (B,7), "hand": (B,20)}.

        Combines into 27D joint action layout [fingers(20) | arm_r(7)] for EMA smoothing
        and downstream `_apply_action`. Stores prev_arm/prev_hand for next obs.
        """
        # Arm action scaled by cfg.arm_action_scale BEFORE clamp.
        # With σ≈0.10 the raw rarely exceeds ±1, so the clamp is mostly a safety net;
        # the practical effect is to halve (or whatever scale) the arm joint range
        # AND exploration noise in joint space (jitter mitigation).
        arm_a = (actions["arm"].clone() * self.cfg.arm_action_scale).clamp(-1.0, 1.0)  # (B, 7)
        hand_a = actions["hand"].clone().clamp(-1.0, 1.0)  # (B, 20) — not scaled

        # Combine in single-agent layout: fingers first, then arm_r
        joint_actions = torch.cat([hand_a, arm_a], dim=-1)  # (B, 27)

        # Cache combined 27D action for use by _get_rewards and _get_observations.
        # NOTE: Do NOT overwrite `self.actions` — DirectMARLEnv initializes it as a
        # per-agent dict in __init__ (line 643 of direct_marl_env.py) and may rely on
        # that during reset/first-step observation. Keep the parent's dict intact.
        self._joint_actions = joint_actions

        # ── WARMUP ────────────────────────────────────────────────────────────
        new_frame = (self._frame_idx + 1).clamp(max=self._max_traj_len - 1)
        if self.cfg.enable_warmup:
            self._frame_idx = torch.where(self._is_warming_up, self._frame_idx, new_frame)
        else:
            self._frame_idx = new_frame
        # ── END WARMUP ────────────────────────────────────────────────────────

        # EMA smoothing on joint actions (27D, lift excluded — held separately).
        # TJ/rl_games convention: alpha = weight on the new (raw) action.
        # Split α: hand uses action_smoothing, arm uses arm_action_smoothing (stronger smoothing → less wrist tremor).
        a_h = self.cfg.action_smoothing
        a_a = self.cfg.arm_action_smoothing
        self._smoothed_actions[:, :20] = a_h * joint_actions[:, :20] + (1.0 - a_h) * self._smoothed_actions[:, :20]
        self._smoothed_actions[:, 20:] = a_a * joint_actions[:, 20:] + (1.0 - a_a) * self._smoothed_actions[:, 20:]
        # Per-agent prev actions are updated at the END of _get_observations (matches
        # single-agent semantics; used in `prev_action` obs slot, not reward).

        # Mass-in-the-loop: snapshot per-env (action, log_prob_old) AT STEP START.
        # _reset_idx runs at END of step and may resample mass for some envs, so
        # we cache the values that were ACTUALLY used during this step now.
        # Read by train_marl's record_transition patch.
        if self._mass_dist is not None:
            self._mass_action_step = self._mass_dist.current_mass_action.detach().clone()
            self._mass_log_prob_old_step = self._mass_dist.current_log_prob_old.detach().clone()

    def _apply_action(self) -> None:
        N_f = self.cfg.num_hand_dofs
        N_a = self.cfg.num_arm_r_dofs
        # Scale EMA-smoothed normalized actions to full joint range (action joints only).
        targets = self._scale(self._smoothed_actions).clamp(self._ctrl_lower, self._ctrl_upper)
        self.robot.set_joint_position_target(targets[:, :N_f],          joint_ids=self._finger_joint_ids)
        self.robot.set_joint_position_target(targets[:, N_f:N_f+N_a],  joint_ids=self._arm_r_joint_ids)
        # Lift is held at a fixed target every step (NOT in action). PD target alone
        # leaves residual trembling under reaction forces from arm/hand motion, so we
        # ALSO forcibly write joint state every physics sub-step → lift effectively
        # kinematic (pos=fixed_target, vel=0) at the start of each sim.step().
        self.robot.set_joint_position_target(self._lift_target, joint_ids=self._lift_joint_ids)
        self.robot.write_joint_state_to_sim(
            self._lift_target, self._lift_zero_vel, joint_ids=self._lift_joint_ids,
        )

    # ------------------------------------------------------------------
    # Palm-local frame transform helpers
    # ------------------------------------------------------------------

    def _to_palm_local_pos(self, p_world: torch.Tensor, palm_pos: torch.Tensor, palm_quat_inv: torch.Tensor) -> torch.Tensor:
        """Transform (B, N, 3) world positions into palm-local frame.

        p_local = quat_apply_inverse(palm_quat, p_world - palm_pos), but we already
        have palm_quat_inv (== conjugate of palm_quat for unit quats), so we use quat_apply.
        """
        B, N, _ = p_world.shape
        rel = p_world - palm_pos.unsqueeze(1)
        q_exp = palm_quat_inv.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 4)
        return quat_apply(q_exp, rel.reshape(B * N, 3)).reshape(B, N, 3)

    def _to_palm_local_vec(self, v_world: torch.Tensor, palm_quat_inv: torch.Tensor) -> torch.Tensor:
        """Transform (B, N, 3) world-frame free vectors (velocity, delta) into palm-local frame.

        No translation — pure rotation.
        """
        B, N, _ = v_world.shape
        q_exp = palm_quat_inv.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 4)
        return quat_apply(q_exp, v_world.reshape(B * N, 3)).reshape(B, N, 3)

    # ------------------------------------------------------------------
    # MARL observation
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return per-agent observation dict for canonical MAPPO.

        - "arm" ( 82D): own joints + current palm pose + palm lin/ang vel +
                        wrist target delta + prev_arm_action +
                        object pose/velocity/deltas + wrist-to-object offset +
                        current_hand_action slot (filled by SequentialMAPPO
                        patch — hand → arm injection at [62:82]).
        - "hand" (276D): single-agent grasping obs minus mass and lift —
                         22 kpts world (21 MANO + elbow) + palm state + fingertip vel +
                         full 27D joint state (no lift) + object state +
                         reference deltas + future_contact + prev_action(27)
                         + fingertip_forces.

        Sequential conditioning: hand acts first; its action is injected into
        arm's `current_hand_action` slot before arm forwards.

        Also caches `self._shared_state` (292D = hand_obs 289 + delta_wrist_rot 3) for
        `_get_states()` — explicit non-redundant centralized critic input.
        """
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        B = self.num_envs
        N_f = self.cfg.num_hand_dofs
        N_a = self.cfg.num_arm_r_dofs

        # Controlled-action joint state (no lift): [fingers(20) | arm_r(7)] = 27D,
        # normalized via _unscale (action-joint scales).
        jp_no_lift_raw = torch.cat([
            self.robot.data.joint_pos[:, self._finger_joint_ids],
            self.robot.data.joint_pos[:, self._arm_r_joint_ids],
        ], dim=-1)  # (B, 27)
        full_jp_norm = self._unscale(jp_no_lift_raw)  # (B, 27)
        full_jv = torch.cat([
            self.robot.data.joint_vel[:, self._finger_joint_ids],
            self.robot.data.joint_vel[:, self._arm_r_joint_ids],
        ], dim=-1)  # (B, 27)
        jp_arm = full_jp_norm[:, N_f:N_f+N_a]                # (B, 7)
        jv_arm = full_jv[:, N_f:N_f+N_a]                     # (B, 7)

        # MANO keypoints (world frame) + fingertip velocities
        hand_kpts_pos = self._compute_hand_kpts_pos()                           # (B, 23, 3) world
        ft_pos = hand_kpts_pos[:, self._kpt_ft_mano_indices_t, :]              # (B, 5, 3)
        if len(self._ft_body_ids) == 5:
            ft_vel = self.robot.data.body_lin_vel_w[:, self._ft_body_ids, :]   # (B, 5, 3)
        else:
            ft_vel = torch.zeros(B, 5, 3, device=self.device)

        # Wrist (palm) world-frame state
        if self._wrist_body_id is not None:
            wrist_pos_w  = self.robot.data.body_pos_w[:, self._wrist_body_id, :]
            wrist_quat_w = self.robot.data.body_quat_w[:, self._wrist_body_id, :]
            wrist_linvel = self.robot.data.body_lin_vel_w[:, self._wrist_body_id, :]
            wrist_angvel = self.robot.data.body_ang_vel_w[:, self._wrist_body_id, :]
        else:
            wrist_pos_w  = torch.zeros(B, 3, device=self.device)
            wrist_quat_w = torch.zeros(B, 4, device=self.device); wrist_quat_w[:, 0] = 1.0
            wrist_linvel = torch.zeros(B, 3, device=self.device)
            wrist_angvel = torch.zeros(B, 3, device=self.device)

        # Object state (world)
        obj_pos    = self.object.data.root_pos_w
        obj_quat   = self.object.data.root_quat_w
        obj_linvel = self.object.data.root_lin_vel_w
        obj_angvel = self.object.data.root_ang_vel_w

        # Reference look-ahead (next frame)
        env_orig = self.scene.env_origins
        next_frame = (frame + 1).clamp(max=self._max_traj_len - 1)
        ref_kpts_world_next = self._ref_mano_kpts[traj, next_frame] + env_orig.unsqueeze(1)  # (B, 23, 3)
        ref_ft_pos_next     = self._ref_ft_pos[traj, next_frame]   + env_orig.unsqueeze(1)   # (B, 5, 3)
        ref_obj_pos         = self._ref_obj_pos[traj, next_frame]  + env_orig                # (B, 3)
        ref_obj_quat        = self._ref_obj_quat[traj, next_frame]                           # (B, 4)
        ref_wrist_quat      = self._ref_wrist_quat[traj, next_frame]                         # (B, 4)

        # delta_ft_obj (contact-conditioned, object-local frame)
        ref_vertex_local  = self._ref_contact_vertex_local[traj, next_frame]   # (B, 5, 3)
        contact_flag_next = self._future_contact[traj, next_frame]              # (B, 5)
        obj_quat_exp = obj_quat.unsqueeze(1).expand(-1, 5, -1).reshape(B * 5, 4)
        ft_in_obj = quat_apply_inverse(
            obj_quat_exp,
            (ft_pos - obj_pos.unsqueeze(1)).reshape(B * 5, 3),
        ).reshape(B, 5, 3)
        ref_ft_in_obj = quat_apply_inverse(
            obj_quat_exp,
            (ref_ft_pos_next - obj_pos.unsqueeze(1)).reshape(B * 5, 3),
        ).reshape(B, 5, 3)
        target_in_obj = torch.where(contact_flag_next.unsqueeze(-1).bool(), ref_vertex_local, ref_ft_in_obj)
        delta_ft_obj = ft_in_obj - target_in_obj                                # (B, 5, 3)

        # Object delta toward next-frame reference (TJ: 6D rotation representation)
        delta_obj_pos = obj_pos - ref_obj_pos
        q_err_obj = quat_mul(obj_quat, quat_conjugate(ref_obj_quat))
        delta_obj_rot_6d = quat_to_6d(q_err_obj)
        # Wrist delta (kpt[0]) and rotation delta — used by arm obs
        delta_wrist_pos = hand_kpts_pos[:, 0, :] - ref_kpts_world_next[:, 0, :] # (B, 3)
        q_err_wrist = quat_mul(wrist_quat_w, quat_conjugate(ref_wrist_quat))
        # Canonicalize quaternion (force w >= 0) → shortest-path representation.
        # Without this, q and -q encode the same rotation but flip the axis-angle
        # vector sign → obs discontinuity → policy treats it as a sudden rotation
        # change and induces wrist tremor.
        q_err_wrist = torch.where(q_err_wrist[:, 0:1] < 0, -q_err_wrist, q_err_wrist)
        delta_wrist_rot = 2.0 * q_err_wrist[:, 1:]                              # axis-angle approx
        # All-21 kpts delta (world frame) — used by hand obs (single-agent style)
        delta_kpts_world = hand_kpts_pos - ref_kpts_world_next                  # (B, 23, 3)

        future_contact   = self._future_contact[traj, frame]                    # (B, 5)
        fingertip_forces = self._get_fingertip_forces()                         # (B, 5)
        wrist_pos_env    = wrist_pos_w - env_orig                               # (B, 3)

        # Joint-action history (27D combined: [fingers(20) | arm(7)]).
        prev_action_27 = torch.cat([self._prev_hand_action, self._prev_arm_action], dim=-1)  # (B, 27)

        # Object signals for arm obs — wrist-to-object distance is the direct
        # grasp-reach signal; obj pose/vel let arm anticipate object motion.
        obj_pos_env       = obj_pos - env_orig                                     # (B, 3)
        delta_wrist_obj   = wrist_pos_env - obj_pos_env                            # (B, 3)

        # current_hand_action placeholder — filled by SequentialMAPPO patch
        # (hand decides first, action injected into arm obs slot [69:89]).
        current_hand_placeholder = torch.zeros(B, N_f, device=self.device)

        # TJ-style velocity scaling
        vs = self.cfg.vel_obs_scale

        # ── Arm obs (89D) — wrist-pose follower + object context + hand action ─
        arm_obs = torch.cat([
            jp_arm,                       # 7    own arm joints (normalized)
            vs * jv_arm,                  # 7    joint velocities (TJ scaled)
            wrist_pos_env,                # 3    current palm position (env-relative)
            quat_to_6d(wrist_quat_w),     # 6    current palm orientation (6D)
            wrist_linvel,                 # 3    current palm linear velocity (world)
            vs * wrist_angvel,            # 3    current palm angular velocity (TJ scaled)
            delta_wrist_pos,              # 3    next-frame wrist target delta (world)
            delta_wrist_rot,              # 3    axis-angle delta
            self._prev_arm_action,        # 7    own previous action
            obj_pos_env,                  # 3    object position (env-relative)
            quat_to_6d(obj_quat),         # 6    object orientation (6D)
            obj_linvel,                   # 3    object linear velocity (world)
            vs * obj_angvel,              # 3    object angular velocity (TJ scaled)
            delta_wrist_obj,              # 3    wrist - object position (env-relative)
            delta_obj_pos,                # 3    object current - reference (next frame)
            delta_obj_rot_6d,             # 6    object rotation delta (6D)
            current_hand_placeholder,     # 20   ← SequentialMAPPO overwrite slot [69:89]
        ], dim=-1)
        # = 7+7+3+6+3+3+3+3+7 + 3+6+3+3+3+3+6 + 20 = 89

        # ── Hand obs (295D) — single-agent grasping obs minus mass & lift ────
        hand_obs = torch.cat([
            hand_kpts_pos[:, :21].reshape(B, 63),     # 63  21 MANO kpts (world)
            hand_kpts_pos[:, 21],                     # 3   right elbow position
            hand_kpts_pos[:, 22],                     # 3   arm_r_link7 position
            quat_to_6d(wrist_quat_w),       # 6   wrist rotation (6D)
            wrist_linvel,                   # 3   wrist linear vel (world)
            vs * wrist_angvel,              # 3   wrist angular vel (TJ scaled)
            ft_vel.reshape(B, 15),          # 15  fingertip velocities (world)
            full_jp_norm,                   # 27  controlled joints (finger+arm, no lift)
            vs * full_jv,                   # 27  joint velocities (TJ scaled)
            obj_pos,                        # 3
            quat_to_6d(obj_quat),           # 6   object rotation (6D)
            obj_linvel,                     # 3
            vs * obj_angvel,                # 3   (TJ scaled)
            delta_kpts_world[:, :21].reshape(B, 63),  # 63  21 MANO kpts delta
            delta_kpts_world[:, 21],                  # 3   right elbow delta
            delta_kpts_world[:, 22],                  # 3   arm_r_link7 delta
            delta_ft_obj.reshape(B, 15),    # 15  obj-local contact-conditioned delta
            delta_obj_pos,                  # 3
            delta_obj_rot_6d,               # 6   object rotation delta (6D)
            future_contact,                 # 5
            prev_action_27,                 # 27  combined action history (no mass)
            fingertip_forces,               # 5
        ], dim=-1)
        # = 63+3+3+6+3+3+15+27+27+3+6+3+3+63+3+3+15+3+6+5+27+5 = 295 (21 MANO + elbow + link7 separated)

        # ── Shared state (298D) — non-redundant centralized critic input ──────
        # hand_obs already contains 99% of unique info; only delta_wrist_rot is
        # NOT derivable from hand_obs (requires ref_wrist_quat which isn't there).
        # Returned via _get_states() when cfg.state_space > 0.
        self._shared_state = torch.cat([hand_obs, delta_wrist_rot], dim=-1)  # (B, 298)

        if self.cfg.debug_vis:
            ref_wrist_pos = self._ref_wrist_pos[traj, frame] + env_orig
            ref_elbow_pos = self._ref_mano_kpts[traj, frame, 21] + env_orig  # kpt 21 = elbow
            ref_link7_pos = self._ref_mano_kpts[traj, frame, 22] + env_orig  # kpt 22 = arm_r_link7
            self._update_debug_vis(ref_ft_pos_next, ft_pos, ref_wrist_pos, ref_elbow_pos, ref_link7_pos)

        # Update per-agent prev_action AFTER building obs (matches single-agent
        # semantics; used as `prev_action` obs slot in next step).
        if hasattr(self, "_joint_actions"):
            self._prev_arm_action = self._joint_actions[:, N_f:N_f+N_a].clone()
            self._prev_hand_action = self._joint_actions[:, :N_f].clone()

        return {"arm": arm_obs, "hand": hand_obs}

    def _get_states(self) -> torch.Tensor:
        """Return the centralized critic input. Non-redundant 292D shared state
        computed and cached during the most recent `_get_observations()` call.
        """
        return self._shared_state

    def _get_fingertip_forces(self) -> torch.Tensor:
        """Return per-fingertip compressive contact force (N), projected onto pad-inward direction.

        Uses `force_matrix_w` (per-filter-object contact force) — only counts contact
        with the Object filter, NOT self-collision or table contacts.
        Mirrors TJ's projection:
            force_along_pad = (force_w * -pad_normal_w).sum(-1).clamp_min(0)

        [THUMB-RADIUS-FILTER]: the thumb's force is additionally gated by the
        distance between the avg contact position and the actual thumb-tip
        world pos — see the marked block below. Other fingers are unaffected.
        """
        B = self.num_envs
        forces = torch.zeros(B, 5, device=self.device)
        pad_normals_w = self._compute_fingertip_pad_normals_w()   # (B, 5, 3) pad-OUTWARD
        # [THUMB-RADIUS-FILTER] Precompute world-frame tip positions for the
        # thumb-only spatial gate below. Cheap (one quat_apply over 5 links).
        ft_pos_w = self._compute_fingertip_positions()             # (B, 5, 3)

        for i, name in enumerate(self.cfg.fingertip_body_names):
            sensor = self._contact_sensors.get(name)
            if sensor is None:
                continue
            try:
                fmat = sensor.data.force_matrix_w     # (B, 1, 1, 3) — Object-only force
                force_vec = fmat[:, 0, 0, :]          # (B, 3)
                inward = -pad_normals_w[:, i, :]      # (B, 3) pad-inward
                f = (force_vec * inward).sum(dim=-1).clamp(min=0.0)   # (B,)

                # ── [THUMB-RADIUS-FILTER] ─────────────────────────────────
                # Thumb tip link is much longer than the other fingers — PhysX
                # attributes mid-link contacts to the body and inflates the
                # reported force. Gate: only count thumb force when the avg
                # contact position (`contact_pos_w`) is within
                # `_THUMB_CONTACT_RADIUS_M` of the actual thumb-tip world pos.
                #
                # Prereq: `track_contact_points=True` on the ContactSensorCfg
                # (set above in _setup_scene). If that flag is removed,
                # `contact_pos_w` is None and this entire block is bypassed
                # → original unfiltered behavior is restored.
                #
                # `contact_pos_w` is NaN when no contact exists for that
                # (body, filter) pair; we mask with `~isnan` so NaN never
                # propagates into `f`. (force is also 0 in that case, so the
                # gate is effectively a no-op there — explicit masking just
                # avoids any chance of NaN arithmetic in the dist compare.)
                #
                # To revert: delete this block (the unfiltered `f` above is
                # what `forces[:, i]` will then receive directly).
                # ──────────────────────────────────────────────────────────
                if _THUMB_FILTER_ENABLED and name == "finger_r_link4" and sensor.data.contact_pos_w is not None:
                    cpos = sensor.data.contact_pos_w[:, 0, 0, :]                       # (B, 3)
                    valid = ~torch.isnan(cpos).any(dim=-1)                              # (B,)
                    dist = torch.norm(cpos - ft_pos_w[:, i, :], dim=-1)                # (B,)
                    gate = (valid & (dist < _THUMB_CONTACT_RADIUS_M)).float()
                    f = f * gate
                # ── END [THUMB-RADIUS-FILTER] ─────────────────────────────

                forces[:, i] = f
            except Exception:
                pass
        return forces

    def _get_arm_table_force(self) -> torch.Tensor:
        """(B,) MAX magnitude of contact force across arm_r_link3..link7 vs table prims."""
        B = self.num_envs
        sensors = getattr(self, "_arm_contact_sensors", None)
        if not sensors:
            return torch.zeros(B, device=self.device)
        per_link_force = []
        for sensor in sensors.values():
            try:
                fmat = sensor.data.force_matrix_w                # (B, 1, 2, 3)
                force_per_filter = fmat[:, 0, :, :]              # (B, 2, 3)
                per_link_force.append(force_per_filter.norm(dim=-1).sum(dim=-1))   # (B,)
            except Exception:
                per_link_force.append(torch.zeros(B, device=self.device))
        return torch.stack(per_link_force, dim=-1).max(dim=-1).values               # (B,)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Per-agent reward dict.

        Splits the single-agent reward terms into arm-exclusive and hand-exclusive
        signals (paper convention):
          - arm gets object tracking + wrist keypoint (world, Z-weighted) +
            arm regularization terms.
          - hand gets finger keypoints (palm-local, no Z-weighting since palm-local
            Z is not the gravity direction) + fingertip + contact force + hand
            regularization. No object tracking signal — paper Section 3.2.

        Shared "alive" reward goes to both. Termination buf is computed from the
        same global thresholds and stored in self._early_terminate_buf for _get_dones.
        """
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        env_orig = self.scene.env_origins

        obj_pos = self.object.data.root_pos_w
        obj_quat = self.object.data.root_quat_w

        ref_obj_pos = self._ref_obj_pos[traj, frame] + env_orig
        ref_obj_quat = self._ref_obj_quat[traj, frame]

        # Object tracking errors.
        delta_obj_pos = obj_pos - ref_obj_pos  # (B, 3)
        obj_pos_err = torch.norm(delta_obj_pos, dim=-1)  # unweighted — termination
        delta_obj_pos_w = delta_obj_pos.clone()
        delta_obj_pos_w[:, 2] *= 1.5
        obj_pos_err_w = torch.norm(delta_obj_pos_w, dim=-1)  # Z-weighted — reward
        q_err = quat_mul(obj_quat, quat_conjugate(ref_obj_quat))
        # arcsin(||vec||) form for numerical precision (matches TJ).
        obj_rot_err = 2.0 * torch.asin(torch.norm(q_err[:, 1:4], dim=-1).clamp(max=1.0))

        # All 22 keypoints (wrist + MCP/PIP/DIP + fingertips + right elbow).
        B = self.num_envs
        hand_kpts_pos = self._compute_hand_kpts_pos()                      # (B, 23, 3)
        ft_pos = hand_kpts_pos[:, self._kpt_ft_mano_indices_t, :]         # (B, 5, 3)

        # Reference keypoints in world frame (TJ-style: NO drift compensation).
        ref_kpts_local = self._ref_mano_kpts[traj, frame]                 # (B, 23, 3) env-local
        ref_kpts_world = ref_kpts_local + env_orig.unsqueeze(1)            # (B, 23, 3) world

        # Keypoint tracking: 21 MANO kpts (Z-weighted) for rew_kpts; elbow handled separately.
        delta_kpts = hand_kpts_pos - ref_kpts_world                       # (B, 23, 3)
        delta_kpts_w = delta_kpts.clone()
        delta_kpts_w[:, :, 2] *= 1.5                                       # Z-weighted (paper S4.2: gravity emphasis)
        delta_kpts_mano = delta_kpts[:, :21]                              # (B, 21, 3) MANO kpts only, unweighted
        delta_kpts_mano_w = delta_kpts_w[:, :21]                          # (B, 21, 3) MANO kpts only, Z-weighted
        kpts_err_w = torch.norm(delta_kpts_mano_w, dim=-1).mean(dim=-1)   # (B,) 21 MANO Z-weighted
        self._last_kpts_err = torch.norm(delta_kpts_mano, dim=-1).mean(dim=-1)  # 21 MANO, for term/warmup
        self._last_kpts_err_raw = self._last_kpts_err

        # Wrist error from keypoint 0 (unweighted): termination + monitoring (TJ-style raw L2).
        wrist_err = torch.norm(delta_kpts_mano[:, 0, :], dim=-1)          # (B,) unweighted

        # Arm error: mean over wrist (kpt 0) + elbow (kpt 21) + arm_r_link7 (kpt 22).
        # - `arm_pos_err` (unweighted mean): logged as `Error / arm_pos_m`.
        # - `arm_err_w`   (Z-weighted mean): used by `rew_arm_pos`.
        arm_kpt_idx = [0, 21, 22]
        arm_pos_err = torch.norm(delta_kpts[:, arm_kpt_idx, :], dim=-1).mean(dim=-1)   # (B,)
        arm_err_w = torch.norm(delta_kpts_w[:, arm_kpt_idx, :], dim=-1).mean(dim=-1)   # (B,) Z-weighted

        # Wrist rotation error (used by arm reward + termination).
        wrist_rot_err = torch.zeros(B, device=self.device)
        if self._wrist_body_id is not None:
            wrist_quat_w_cur = self.robot.data.body_quat_w[:, self._wrist_body_id, :]
            ref_wrist_quat = self._ref_wrist_quat[traj, frame]
            q_err = quat_mul(wrist_quat_w_cur, quat_conjugate(ref_wrist_quat))
            wrist_rot_err = 2.0 * torch.asin(torch.clamp(torch.norm(q_err[:, 1:4], dim=-1), max=1.0))

        # Fingertip contact tracking (identical to single-agent).
        ref_ft = self._ref_ft_pos[traj, frame] + env_orig.unsqueeze(1)    # (B, 5, 3)
        contact_flag = self._future_contact[traj, frame]                   # (B, 5)

        ref_vertex_local_r = self._ref_contact_vertex_local[traj, frame]  # (B, 5, 3) obj-local
        obj_quat_exp_r = obj_quat.unsqueeze(1).expand(-1, 5, -1).reshape(B * 5, 4)
        ref_vertex_world = quat_apply(
            obj_quat_exp_r, ref_vertex_local_r.reshape(B * 5, 3),
        ).reshape(B, 5, 3) + obj_pos.unsqueeze(1)

        # Non-contact target: drift-compensated ref fingertip (TJ `fingertip_pos_ref_rel`).
        # Place ref ft relative to current obj pose: q_offset * (ref_ft - ref_obj_pos) + obj_pos.
        ref_obj_pos_world_r = self._ref_obj_pos[traj, frame] + env_orig
        ref_obj_quat_r = self._ref_obj_quat[traj, frame]
        ref_ft_local_to_ref_obj = ref_ft - ref_obj_pos_world_r.unsqueeze(1)
        ref_obj_quat_exp = ref_obj_quat_r.unsqueeze(1).expand(-1, 5, -1).reshape(B * 5, 4)
        ref_ft_in_obj_canon = quat_apply_inverse(
            ref_obj_quat_exp, ref_ft_local_to_ref_obj.reshape(B * 5, 3)
        )
        ref_ft_drift = quat_apply(obj_quat_exp_r, ref_ft_in_obj_canon).reshape(B, 5, 3) + obj_pos.unsqueeze(1)

        contact_flag_gated = contact_flag
        # REWARD target: drift-compensated, contact-conditioned (TJ delta_fingertip_pos_rel_new).
        ft_target = torch.where(contact_flag_gated.unsqueeze(-1).bool(), ref_vertex_world, ref_ft_drift)
        ft_err_per_finger = torch.norm(ft_pos - ft_target, dim=-1)
        ft_err = ft_err_per_finger.mean(dim=-1)                            # used for reward

        # TERMINATION: RAW ref ft (TJ delta_fingertip_pos, no drift comp).
        ft_err_raw_per_finger = torch.norm(ft_pos - ref_ft, dim=-1)
        ft_err_raw = ft_err_raw_per_finger.mean(dim=-1)                    # used for termination
        self._last_ft_raw_err = ft_err_raw

        # Contact force reward.
        raw_forces = self._get_fingertip_forces()
        contact_condition = (ft_err_per_finger < 0.03).float()
        fforce_contact = raw_forces * contact_flag_gated * contact_condition
        n_contacts = contact_flag_gated.sum(dim=-1, keepdim=True)
        clamped = torch.clamp(fforce_contact, 0.0, 0.5) / (n_contacts + 1e-6) / 1.5
        force_rew = clamped.sum(dim=-1)   # TJ-exact: 1.0× coef

        # Regularization. Action layout (MARL, no mass): [fingers(20) | arm_r(7)].
        N_f = self.cfg.num_hand_dofs
        N_a = self.cfg.num_arm_r_dofs
        hand_action_reg = (self._joint_actions[:, :N_f] ** 2).sum(dim=-1)
        arm_action_reg  = (self._joint_actions[:, N_f:N_f+N_a] ** 2).sum(dim=-1)
        jp = self.robot.data.joint_pos
        dp = self.robot.data.default_joint_pos
        hand_pose_reg = ((jp[:, self._finger_joint_ids] - dp[:, self._finger_joint_ids]) ** 2).sum(dim=-1)
        arm_pose_reg  = ((jp[:, self._arm_r_joint_ids]  - dp[:, self._arm_r_joint_ids])  ** 2).sum(dim=-1)

        # Arm-table contact (anti-cheating, link3..link7). MAX across links → penalty + termination.
        # Per-step penalty auto-clamped at `rew_arm_contact × max_arm_contact_force`.
        arm_table_force = self._get_arm_table_force()                                     # (B,) N
        arm_penalty = (self.cfg.rew_arm_contact * arm_table_force).clamp(
            min=self.cfg.rew_arm_contact * self.cfg.max_arm_contact_force
        )                                                                                  # (B,) ≤ 0
        arm_strong_press = arm_table_force > self.cfg.max_arm_contact_force
        self._last_arm_table_force = arm_table_force                                       # for logging
        # Termination (shared across agents).
        pos_err_large = obj_pos_err > self.cfg.max_obj_pos_err
        rot_err_large = obj_rot_err > self.cfg.max_obj_rot_err
        ft_err_large = ft_err_raw > self.cfg.max_ft_mean_err  # termination uses RAW (TJ-style)
        wrist_err_large = wrist_err > self.cfg.max_wrist_pos_err
        wrist_rot_err_large = wrist_rot_err > self.cfg.max_wrist_rot_err
        # elbow termination disabled — elbow is soft guidance only (cfg.max_elbow_pos_err kept for future use)
        early_terminate = (
            pos_err_large | rot_err_large | ft_err_large
            | wrist_err_large | wrist_rot_err_large
            | arm_strong_press
        )
        if not self.cfg.termination:
            early_terminate = torch.zeros_like(early_terminate)
        if self.cfg.enable_warmup:
            early_terminate = early_terminate & ~self._is_warming_up
        # Grace period: suppress early termination for the first N steps of each episode.
        if self.cfg.early_termination_grace_frames > 0:
            in_grace = self.episode_length_buf < self.cfg.early_termination_grace_frames
            early_terminate = early_terminate & ~in_grace
        self._early_terminate_buf = early_terminate
        self._last_ft_mean_err = ft_err
        self._last_wrist_err = wrist_err
        self._last_wrist_rot_err = wrist_rot_err
        self._last_obj_pos_err = obj_pos_err
        self._last_obj_rot_err = obj_rot_err

        alive = (~early_terminate).float()

        # ── Single team reward (canonical MAPPO) ─────────────────────────────
        # Identical formula to single-agent reward. Both agents receive the
        # same scalar; the shared centralized critic V(s) is trained on this.
        # `rew_kpts` averages 21 MANO kpts; `rew_arm_pos` supervises the 3 arm
        # endpoints (wrist + elbow + arm_r_link7) under a single weight.
        tracking_penalty = (
            self.cfg.rew_kpts * kpts_err_w               # mean over 21 MANO kpts, Z-weighted
            + self.cfg.rew_arm_pos * arm_err_w           # Z-weighted mean of (wrist + elbow + link7) L2
            + self.cfg.rew_obj_pos * obj_pos_err_w       # object position, Z-weighted
            + self.cfg.rew_obj_rot * obj_rot_err
            + self.cfg.rew_fingertip * ft_err            # contact-conditioned fingertip
        ).clamp(min=-self.cfg.rew_alive)
        team_reward = (
            self.cfg.rew_alive * alive
            + tracking_penalty
            + self.cfg.rew_fingertip_force * force_rew
            + arm_penalty                                       # soft penalty (auto-clamped at rew_arm_contact × max_arm_contact_force)
            + self.cfg.rew_hand_action_reg * hand_action_reg
            + self.cfg.rew_arm_action_reg  * arm_action_reg
            + self.cfg.rew_hand_pose_reg   * hand_pose_reg
            + self.cfg.rew_arm_pose_reg    * arm_pose_reg
        ).clamp(min=0.0)

        # State cache uses the team reward (same as single-agent env).
        self._save_state_cache(team_reward, ft_err, obj_pos_err, obj_rot_err)
        self._log_effort_saturation()

        # Logging grouped by top-level tab. Keys use ` / ` (space-slash-space) so
        # they form proper Tensorboard groups when `train_marl.py` strips skrl's
        # automatic "Info / " prefix and calls agent.track_data() directly.
        self.extras["log"] = {
            # Tracking errors
            "Error / kpts_mean_m":      torch.norm(delta_kpts_mano, dim=-1).mean(),
            "Error / wrist_pos_m":      wrist_err.mean(),
            "Error / wrist_rot_deg":    torch.rad2deg(wrist_rot_err).mean(),
            "Error / arm_pos_m":        arm_pos_err.mean(),     # mean over (wrist, elbow, link7); matches reward kpts
            "Error / obj_pos_m":        obj_pos_err.mean(),
            "Error / obj_rot_deg":      torch.rad2deg(obj_rot_err).mean(),
            "Error / ft_mean_m":        ft_err.mean(),
            "Force / arm_table_N":      arm_table_force.mean(),
            "Force / arm_press_rate":   arm_strong_press.float().mean(),
            # Team reward decomposed components
            "Episode_Reward / alive":            (self.cfg.rew_alive * alive).mean(),
            "Episode_Reward / kpts":             (self.cfg.rew_kpts * kpts_err_w).mean(),
            "Episode_Reward / arm_pos":          (self.cfg.rew_arm_pos * arm_err_w).mean(),
            "Episode_Reward / obj_pos":          (self.cfg.rew_obj_pos * obj_pos_err_w).mean(),
            "Episode_Reward / obj_rot":          (self.cfg.rew_obj_rot * obj_rot_err).mean(),
            "Episode_Reward / fingertip":        (self.cfg.rew_fingertip * ft_err).mean(),
            "Episode_Reward / arm_contact":      arm_penalty.mean(),
            "Episode_Reward / fingertip_force":  (self.cfg.rew_fingertip_force * force_rew).mean(),
            "Episode_Reward / hand_action_reg":  (self.cfg.rew_hand_action_reg * hand_action_reg).mean(),
            "Episode_Reward / arm_action_reg":   (self.cfg.rew_arm_action_reg  * arm_action_reg).mean(),
            "Episode_Reward / hand_pose_reg":    (self.cfg.rew_hand_pose_reg   * hand_pose_reg).mean(),
            "Episode_Reward / arm_pose_reg":     (self.cfg.rew_arm_pose_reg    * arm_pose_reg).mean(),
            "Episode_Reward / team_total":       team_reward.mean(),
            # ── [THUMB-RADIUS-FILTER] Monitoring ──────────────────────────
            # `raw_forces[:, 0]` is the thumb force AFTER the distance gate.
            #   thumb_N         — mean compressive force on thumb pad (N)
            #   thumb_active    — fraction of envs where thumb force > 0
            #                     (proxy for "gate passed AND contact existed")
            # To revert: delete these two lines.
            # ──────────────────────────────────────────────────────────────
            "Force / thumb_N":       raw_forces[:, 0].mean(),
            "Force / thumb_active":  (raw_forces[:, 0] > 0.0).float().mean(),
            # Curriculum
            "Curriculum / reached_frame": torch.tensor(float(self._reached_frame), device=self.device),
            "Curriculum / warmup_ratio":  self._is_warming_up.float().mean(),
        }

        # Both agents receive the same team reward (canonical MAPPO).
        return {"arm": team_reward, "hand": team_reward}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Per-agent (terminated, time_out) dicts. Both agents see identical signals —
        one agent's failure ends the episode for both (cooperative coupling).

        IMPORTANT: DirectMARLEnv computes reset_buf as math.prod(terminated_dict.values()) |
        math.prod(time_out_dict.values()), which uses tensor PRODUCT — for bool tensors
        this is logical AND. Broadcasting the same signal ensures the product equals
        the broadcast signal (i.e. episode resets when the signal is True).
        """
        terminated = self._early_terminate_buf

        # ── WARMUP ────────────────────────────────────────────────────────────
        if self.cfg.enable_warmup and self._is_warming_up.any():
            warmup_done = (
                (self._last_ft_mean_err < self.cfg.warmup_ft_threshold)
                & (self._last_wrist_err < self.cfg.warmup_wrist_threshold)
                & (self._last_wrist_rot_err < self.cfg.warmup_wrist_rot_threshold)
            )
            self._is_warming_up = self._is_warming_up & ~warmup_done
        # ── END WARMUP ────────────────────────────────────────────────────────

        # TJ-style fixed-length episode: time-out fires when the framework step counter
        # reaches max_episode_length (= _num_frame_chunk). Adaptive sampling clamps
        # `start_frame` so episodes run exactly `_num_frame_chunk` steps regardless of
        # per-env trajectory length.
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self.extras.setdefault("log", {})["Curriculum / success_rate"] = (
            (~terminated & ~time_out & (self._last_obj_pos_err < 0.03))
            .float().mean()
        )

        self._done_env = terminated | time_out
        return (
            {"arm": terminated, "hand": terminated},
            {"arm": time_out, "hand": time_out},
        )

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        n = len(env_ids)
        super()._reset_idx(env_ids)

        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # --- Mass-in-the-loop: resample mass per-env at episode start. ---
        # Caches new action and (sampling-time) log_prob_old in mass_dist for PPO ratio.
        # The mass currently in use during this step's transition (about to be
        # recorded by train_marl) was already snapshotted at _pre_physics_step.
        if self._mass_dist is not None and self.cfg.enable_mass_in_loop:
            new_masses_kg = self._mass_dist.sample_for_envs(env_ids_t)
            # PhysX mass write — set_masses() expects (num_envs, num_rigid_bodies);
            # we update only env_ids' slice.
            try:
                all_masses = self.object.root_physx_view.get_masses().clone()
                cpu_dev = all_masses.device
                all_masses[env_ids_t.to(cpu_dev), 0] = new_masses_kg.to(cpu_dev)
                all_indices = torch.arange(self.num_envs, dtype=torch.long, device=cpu_dev)
                self.object.root_physx_view.set_masses(all_masses, all_indices)
            except Exception as e:
                if not hasattr(self, "_mass_set_warned"):
                    print(f"[grasp-marl] WARNING: object mass set_masses failed ({e!r}); "
                          f"continuing with previous mass.")
                    self._mass_set_warned = True

        # --- Adaptive sampling: EMA failure count update using _enough_idx ---
        # Mirrors GR env: bincount the failure frames, then EMA-update the full count vector.
        # NOTE: DirectMARLEnv does NOT populate `self.reset_terminated` (that is a
        # DirectRLEnv-only attribute). We must derive the early-termination mask from
        # the per-agent terminated_dict instead (both agents share the same signal
        # via broadcast in _get_dones, so reading "arm" is sufficient).
        if self.cfg.adaptive_sampling and hasattr(self, "terminated_dict") and self.terminated_dict:
            term_full = next(iter(self.terminated_dict.values()))  # (B,) bool
            is_terminated = term_full[env_ids]
            if is_terminated.any():
                term_env_ids = env_ids_t[is_terminated]
                failure_frames = self._enough_idx[term_env_ids].clamp(0, self._max_traj_len - 1)
                counts = torch.bincount(failure_frames, minlength=self._max_traj_len).float()
                alpha = self.cfg.adaptive_alpha
                self._failure_count = alpha * counts + (1.0 - alpha) * self._failure_count

        # --- Trajectory assignment ---
        if self._n_trajs == 1:
            self._traj_idx[env_ids] = 0
        else:
            self._traj_idx[env_ids] = torch.randint(0, self._n_trajs, (n,), device=self.device)

        # --- Start frame sampling ---
        if self.cfg.adaptive_sampling and self._reached_frame > 0:
            valid_len = min(self._reached_frame + 1, self._max_traj_len)
            if self.cfg.failure_weighted_sampling:
                # TJ formula: p = (fail_probs + ur/N) / (1 + ur) — add uniform then renormalize.
                valid_counts = self._failure_count[:valid_len]
                ur = self.cfg.adaptive_uniform_ratio
                fail_probs = valid_counts / (valid_counts.sum() + 1e-8)
                probs = (fail_probs + ur / valid_len) / (1.0 + ur)         # (valid_len,)
                sampled = torch.multinomial(probs.unsqueeze(0).expand(n, -1), 1).squeeze(-1)
            else:
                # Pure uniform sampling within [0, _reached_frame] — no failure weighting.
                sampled = torch.randint(0, valid_len, (n,), device=self.device)
            start_frames = (sampled - self._adaptive_back_frames).clamp(min=0)
            # TJ upper bound: keep at least `num_frame_chunk` frames after start. Mirrors
            # ``clamp(0, min(max(0, episode_length - num_frame_chunk),
            #               max(reached_frame - back, 0)))``.
            # When traj_len ≤ num_frame_chunk, upper_a = 0 → start_frame = 0 → adaptive
            # sampling effectively disabled (matches TJ behavior on short trajs).
            upper_a = max(0, self._max_traj_len - self._num_frame_chunk)
            upper_b = max(self._reached_frame - self._adaptive_back_frames, 0)
            upper = min(upper_a, upper_b)
            start_frames = start_frames.clamp(max=upper)
        else:
            start_frames = torch.zeros(n, dtype=torch.long, device=self.device)

        self._frame_idx[env_ids] = start_frames
        self._prev_arm_action[env_ids] = 0.0
        self._prev_hand_action[env_ids] = 0.0
        self._done_env[env_ids] = False

        # Reset per-episode tracking quality
        self._enough_continued[env_ids] = True
        self._enough_idx[env_ids] = start_frames

        # --- Robot state reset (with state cache restore if available) ---
        traj = self._traj_idx[env_ids]
        env_orig = self.scene.env_origins[env_ids]

        cached = self._state_cache[start_frames]          # (n, 98) — zeros for unpopulated frames
        has_cache = ~self._init_flg[start_frames]         # (n,) bool
        cache_mask = has_cache.unsqueeze(-1)              # (n, 1) for broadcasting

        # ── WARMUP ────────────────────────────────────────────────────────────
        # Warm-up is needed only when there is no cached state for start_frame.
        # When a cached state exists, the robot is initialized to a physically
        # reachable state near the target, so warm-up is unnecessary.
        # To restore original behavior: remove this block.
        if self.cfg.enable_warmup:
            self._is_warming_up[env_ids] = ~has_cache
        else:
            self._is_warming_up[env_ids] = False
        # ── END WARMUP ────────────────────────────────────────────────────────

        # Joint state: restore controlled joints from cache, others stay at default
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos_reset = default_joint_pos.clone()
        joint_vel_reset = torch.zeros_like(default_joint_pos)

        # cached[:, 14:42] = joint_pos (28 controlled), cached[:, 42:70] = joint_vel (28),
        # cached[:, 70:97] = smoothed_act (27, lift excluded)
        cached_jp = cached[:, 14:42]
        cached_jv = cached[:, 42:70]
        cached_sa = cached[:, 70:97]

        joint_pos_reset[:, self._all_joint_ids] = torch.where(
            cache_mask, cached_jp, default_joint_pos[:, self._all_joint_ids]
        )
        joint_vel_reset[:, self._all_joint_ids] = torch.where(
            cache_mask, cached_jv, torch.zeros(n, len(self._all_joint_ids), device=self.device)
        )

        # When no cached state, apply frame-0 IK for the arm (mirrors GR train first-reset + pretrain).
        if self._frame0_arm_joint_pos is not None:
            arm_ik = self._frame0_arm_joint_pos[traj]   # (n, 7)
            no_cache_arm = (~has_cache).unsqueeze(-1).expand(-1, len(self._arm_r_joint_ids))
            joint_pos_reset[:, self._arm_r_joint_ids] = torch.where(
                no_cache_arm, arm_ik, joint_pos_reset[:, self._arm_r_joint_ids]
            )

        # Force lift joint to the fixed target on every reset (not controlled by policy).
        joint_pos_reset[:, self._lift_joint_ids] = self.cfg.fixed_lift_target
        joint_vel_reset[:, self._lift_joint_ids] = 0.0

        # Build smoothed_actions for cache-miss envs using actual initial joint positions
        # (action joints only — 20 fingers + 7 arm = 27D; lift is excluded).
        default_ctrl = torch.cat([
            default_joint_pos[:, self._finger_joint_ids],
            joint_pos_reset[:, self._arm_r_joint_ids],   # IK if available and no cache, else default
        ], dim=-1)
        default_normalized = self._unscale(default_ctrl)
        self._smoothed_actions[env_ids] = torch.where(cache_mask, cached_sa, default_normalized)

        self.robot.write_joint_state_to_sim(joint_pos_reset, joint_vel_reset, None, env_ids)

        # Robot root pose (fixed-base robot: always reset to default initial state)
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += env_orig
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # --- Object reset: use cached state if available, else reference trajectory ---
        ref_obj_pos = self._ref_obj_pos[traj, start_frames] + env_orig   # (n, 3)
        ref_obj_quat = self._ref_obj_quat[traj, start_frames]             # (n, 4)

        # cached[:, 1:4] = obj_pos env-local, cached[:, 4:8] = obj_quat,
        # cached[:, 8:11] = obj_linvel, cached[:, 11:14] = obj_angvel
        cached_obj_pos = cached[:, 1:4] + env_orig   # env-local → world
        cached_obj_quat = cached[:, 4:8]
        cached_obj_vel = cached[:, 8:14]              # linvel(3) + angvel(3)

        obj_pos_reset = torch.where(cache_mask, cached_obj_pos, ref_obj_pos)
        obj_quat_reset = torch.where(cache_mask, cached_obj_quat, ref_obj_quat)
        obj_vel_reset = torch.where(cache_mask, cached_obj_vel, torch.zeros(n, 6, device=self.device))

        self.object.write_root_pose_to_sim(torch.cat([obj_pos_reset, obj_quat_reset], dim=-1), env_ids)
        self.object.write_root_velocity_to_sim(obj_vel_reset, env_ids)

        # --- TJ-style init save: force-write frame 0 cache once so subsequent resets at
        # frame 0 reuse the IK-lifted pose instead of re-applying IK. ---
        if not self._init_save_done:
            init_state = torch.cat([
                torch.zeros(1, 1, device=self.device),                       # reward placeholder
                obj_pos_reset[0:1] - env_orig[0:1],                          # (1, 3)  obj_pos env-local
                obj_quat_reset[0:1],                                         # (1, 4)
                obj_vel_reset[0:1, :3],                                      # (1, 3)  obj_linvel
                obj_vel_reset[0:1, 3:],                                      # (1, 3)  obj_angvel
                joint_pos_reset[0:1, self._all_joint_ids],                   # (1, 28)
                joint_vel_reset[0:1, self._all_joint_ids],                   # (1, 28)
                self._smoothed_actions[env_ids_t[0:1]],                      # (1, 27)
            ], dim=-1).squeeze(0)                                            # (97,)
            self._state_cache[0] = init_state
            self._init_flg[0] = False
            self._init_save_done = True

    def _save_state_cache(
        self,
        reward: torch.Tensor,
        ft_err: torch.Tensor,
        obj_pos_err: torch.Tensor,
        obj_rot_err: torch.Tensor,
    ) -> None:
        """Save current simulation state to the state cache and update tracking quality.

        Mirrors GR env _save_states: only writes to cache when (a) enough_continued is True
        and (b) the new reward beats the previously cached reward at that frame. For frames
        with multiple eligible envs, the highest-reward env's state is written.
        """
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)  # (B,)
        env_orig = self.scene.env_origins  # (B, 3)

        obj_pos_local = self.object.data.root_pos_w - env_orig   # env-local (B, 3)
        obj_quat = self.object.data.root_quat_w                   # (B, 4)
        obj_linvel = self.object.data.root_lin_vel_w              # (B, 3)
        obj_angvel = self.object.data.root_ang_vel_w              # (B, 3)
        joint_pos = self.robot.data.joint_pos[:, self._all_joint_ids]  # (B, 28)
        joint_vel = self.robot.data.joint_vel[:, self._all_joint_ids]  # (B, 28)

        # Build 97-dim state vector per env (smoothed_act is 27D — lift excluded from action)
        state = torch.cat([
            reward.unsqueeze(-1),   # (B, 1)  — index 0
            obj_pos_local,           # (B, 3)  — indices 1:4
            obj_quat,                # (B, 4)  — indices 4:8
            obj_linvel,              # (B, 3)  — indices 8:11
            obj_angvel,              # (B, 3)  — indices 11:14
            joint_pos,               # (B, 28) — indices 14:42
            joint_vel,               # (B, 28) — indices 42:70
            self._smoothed_actions,  # (B, 27) — indices 70:97
        ], dim=-1)  # (B, 97)

        # Update per-episode tracking quality — GR/TJ 3-phase condition:
        #   start: loose thresholds for the first ~20 frames so the cache
        #          gets populated early in training before the policy has learned to track well.
        #   early: tighter thresholds once initial frames are in cache (not yet reached end).
        #   late : even tighter thresholds after the policy has reached the trajectory end —
        #          encourages refinement of cache quality.
        action_fps = round(1.0 / (self.cfg.sim.dt * self.cfg.decimation))
        start_frame_cutoff = action_fps * 2 // 3  # ~20 frames at 30 Hz (matches GR: fps/1.5)
        reached_end = self.is_reached_end   # python bool — broadcasts to (B,) implicitly
        start_condition = (obj_pos_err < 0.10) & (obj_rot_err < 0.50) & (frame <= start_frame_cutoff)
        early_condition = (
            (obj_pos_err < self.cfg.enough_obj_threshold)
            & (obj_rot_err < self.cfg.enough_obj_rot_threshold)
            & (not reached_end)
        )
        late_condition = (
            (obj_pos_err < self.cfg.enough_obj_threshold_late)
            & (obj_rot_err < self.cfg.enough_obj_rot_threshold_late)
            & reached_end
        )
        good = (
            (ft_err < self.cfg.enough_ft_threshold)
            & (start_condition | early_condition | late_condition)
        )
        still_good = self._enough_continued & good
        # Advance _enough_idx while tracking is still good (captures last "good" frame)
        self._enough_idx = torch.where(still_good, frame, self._enough_idx)
        self._enough_continued = still_good

        # Cache update: GR env pattern — only write if enough_continued AND reward > cached reward.
        # For frames with multiple eligible envs, write the one with highest reward.
        better_reward = reward > self._state_cache[frame, 0]   # compare vs stored reward
        update_mask = self._enough_continued & better_reward    # (B,) bool

        if update_mask.any():
            unique_frames = torch.unique(frame[update_mask])
            for uf in unique_frames:
                mask_at_frame = (frame == uf) & update_mask
                # Select the env with the highest reward at this frame
                best_local = reward[mask_at_frame].argmax()
                best_env = mask_at_frame.nonzero(as_tuple=True)[0][best_local]
                self._state_cache[uf] = state[best_env]
                self._init_flg[uf] = False
                self._reached_frame = max(self._reached_frame, int(uf.item()))
