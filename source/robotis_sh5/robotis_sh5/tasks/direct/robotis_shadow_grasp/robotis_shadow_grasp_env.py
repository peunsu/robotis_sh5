"""Shadow-Hand-on-Robotis dexterous grasping environment — Isaac Lab Direct RL.

Robot: FFW-SH5 arm + Shadow Dexterous Hand mounted on arm_r_link7 (fix_root_link=True).
Policy controls 22 Shadow Hand joints (FFJ0-3 / MFJ0-3 / RFJ0-3 / LFJ0-4 / THJ0-4) +
right arm (7) = 29 joint DOFs, plus 1 mass parameter dim = 30D action total.
Lift joint is NOT in the action; it is held at `cfg.fixed_lift_target` (0.0 = fully up).

Tendon coupling: the USD physics couple J0 distal joints (FFJ0/MFJ0/RFJ0/LFJ0) to
their J1 parents via fixed tendons. The policy still issues targets to all 22 joints;
tendon constraints enforce coupled motion in the simulator.

Joint/body naming follows the standard Shadow Hand convention (`robot0_*`).
Fingertip mapping for MANO 21-keypoint tracking follows TJ's gr_env mapping.
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
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

from .robotis_shadow_grasp_env_cfg import RobotisShadowGraspEnvCfg


def quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternion to orthonormalized 6D continuous rotation rep (Zhou et al. 2019).

    Mirrors TJ's `quat_to_6d` (Gram-Schmidt on first two rows of R), but with wxyz input order
    to match isaaclab convention. Returns shape (..., 6).
    """
    q = torch.nn.functional.normalize(quat, dim=-1)
    w, x, y, z = q.unbind(-1)
    # First two rows of rotation matrix (wxyz convention)
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


# Shadow Hand fingertip local-frame offsets from distal link origin to the tip
# contact point. Values copied exactly from TJ gr_env.py:204-208.
# Distal link local frame: Z = finger axial (forward), Y = pad side (negative pad).
#   thumb (special):  small -X (-8.5mm) + +Z (20mm forward)
#   others (FF/MF/RF/LF): small -Y (-6mm toward pad) + +Z (17.5mm forward)
_FINGERTIP_OFFSETS: dict[str, list[float]] = {
    "robot0_thdistal": [-0.0085, 0.0,    0.02],
    "robot0_ffdistal": [ 0.0,   -0.006,  0.0175],
    "robot0_mfdistal": [ 0.0,   -0.006,  0.0175],
    "robot0_rfdistal": [ 0.0,   -0.006,  0.0175],
    "robot0_lfdistal": [ 0.0,   -0.006,  0.0175],
}

# Pad-outward normals in each fingertip link's LOCAL frame (unit vectors).
# Both TJ's gr_env.py and sh5 use the SAME convention:
#   force_into_pad = (force_w * -pad_normal_w).sum(-1).clamp_min(0)
# i.e., pad_normal points OUTWARD from the pad; the (-pad_normal) factor flips
# it to the inward direction so compressive contact yields a positive value.
# Matches TJ exactly (gr_env.py: fingertip_normal[:,0,0]=-1, [:,1:,1]=-1).
_FINGERTIP_PAD_NORMALS: dict[str, list[float]] = {
    "robot0_thdistal": [-1.0, 0.0, 0.0],   # thumb: pad outward = -X local
    "robot0_ffdistal": [0.0, -1.0, 0.0],   # index: pad outward = -Y local
    "robot0_mfdistal": [0.0, -1.0, 0.0],   # middle
    "robot0_rfdistal": [0.0, -1.0, 0.0],   # ring
    "robot0_lfdistal": [0.0, -1.0, 0.0],   # little
}

# Elbow position offset in arm_r_link3's local frame.
# URDF `arm_r_joint4` (the elbow joint) has origin xyz=[0.041004, 0, -0.135] in
# arm_r_link3, so this offset locates the actual joint pivot. Reading `body_pos_w`
# of arm_r_link4 directly returns the body pose (CoM-like) which shifts as joint4
# rotates — using link3 + this offset gives a STABLE elbow position regardless of
# elbow joint angle, matching pinocchio FK semantics used in Stage 2 bone rescaling.
_ELBOW_OFFSET_IN_LINK3_LOCAL: list[float] = [0.041004, 0.0, -0.135]

# ── [THUMB-RADIUS-FILTER] ───────────────────────────────────────────────
# Same mechanism as the FFW-SH5 env: gate the thumb's force contribution by
# distance from the actual thumb-tip world position. Shadow Hand's thumb
# distal link is much shorter than FFW-SH5's, so the filter is less critical
# but is kept available for symmetry. Disabled by default.
# ────────────────────────────────────────────────────────────────────────
_THUMB_CONTACT_RADIUS_M: float = 0.02
_THUMB_FILTER_ENABLED: bool = False

# MANO keypoint index → Shadow Hand body name mapping for non-fingertip joints.
# Follows TJ gr_env's body_to_kpts_except_fingertips ordering. MANO layout:
#   0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
# (each finger: MCP→PIP→DIP→tip; tip indices handled by _compute_fingertip_positions)
#
# Shadow Hand mapping (TJ-style):
#   MANO 0 (wrist)   → robot0_palm
#   MANO 1/2/3 (thumb CMC/MCP/DIP) → robot0_thbase / robot0_thmiddle / robot0_thdistal
#   MANO 5/6/7 (index)             → robot0_ffknuckle / robot0_ffmiddle / robot0_ffdistal
#   MANO 9/10/11 (middle)          → robot0_mfknuckle / robot0_mfmiddle / robot0_mfdistal
#   MANO 13/14/15 (ring)           → robot0_rfknuckle / robot0_rfmiddle / robot0_rfdistal
#   MANO 17/18/19 (pinky)          → robot0_lfknuckle / robot0_lfmiddle / robot0_lfdistal
# (TJ uses the distal body for MANO 3/7/11/15/19; the fingertip MANO indices
#  4/8/12/16/20 are computed via the fingertip offsets in _FINGERTIP_OFFSETS.)
#
# Extensions beyond MANO 21:
#   21 (elbow): arm_r_link3 + _ELBOW_OFFSET_IN_LINK3_LOCAL (same as FFW-SH5)
#   22: arm_r_link7 — last revolute link before the fixed wrist mount
_MANO_NON_FT_BODY_NAMES: list[tuple[int, str]] = [
    (0,  "robot0_palm"),                                              # wrist
    (1,  "robot0_thbase"),       (5,  "robot0_ffknuckle"),  (9,  "robot0_mfknuckle"),
    (13, "robot0_rfknuckle"),    (17, "robot0_lfknuckle"),  # thumb-CMC / fingers-MCP
    (2,  "robot0_thmiddle"),     (6,  "robot0_ffmiddle"),   (10, "robot0_mfmiddle"),
    (14, "robot0_rfmiddle"),     (18, "robot0_lfmiddle"),   # PIP joints
    (3,  "robot0_thdistal"),     (7,  "robot0_ffdistal"),   (11, "robot0_mfdistal"),
    (15, "robot0_rfdistal"),     (19, "robot0_lfdistal"),   # DIP joints (= distal body origin)
    # NOTE: kpt 21 (elbow) handled separately in _compute_hand_kpts_pos.
    (22, "arm_r_link7"),                                              # last arm link
]
_MANO_FT_INDICES = [4, 8, 12, 16, 20]  # tip MANO indices → ft_pos[:, 0:5]
_NUM_KPTS = 23  # 21 MANO + elbow (21) + arm_r_link7 (22)


class RobotisShadowGraspEnv(DirectRLEnv):
    """Dexterous grasping with FFW-SH5 using OakInk kinematic references."""

    cfg: RobotisShadowGraspEnvCfg

    def __init__(self, cfg: RobotisShadowGraspEnvCfg, render_mode: str | None = None, **kwargs):
        self._load_reference_trajectories(cfg)
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

    def _load_reference_trajectories(self, cfg: RobotisShadowGraspEnvCfg) -> None:
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
                "Run scripts/process_dataset/oakink.py first."
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

            # 22nd, 23rd keypoints: right elbow + arm_r_link7 from process_arm_pipeline.py
            # (`arm_keypoints.npz` with `elbow_pos`, `link7_pos`). Both saved in the same
            # RAW frame as mano_kpts_right, so they go through canonicalization uniformly.
            # link7 is the last revolute arm link before the FIXED wrist mount; tracking
            # its position adds a positional constraint along the wrist's Z-axis world
            # direction (compensates for missing rew_wrist_rot signal).
            arm_kp_path = path.parent / "arm_keypoints_shadow.npz"
            if arm_kp_path.exists():
                arm_kp = np.load(str(arm_kp_path))
                kp_elbow = arm_kp["elbow_pos"].astype(np.float32).reshape(-1, 1, 3)  # (N, 1, 3)
                kp_link7 = arm_kp["link7_pos"].astype(np.float32).reshape(-1, 1, 3)  # (N, 1, 3)
                if kp_elbow.shape[0] != kp_mano.shape[0] or kp_link7.shape[0] != kp_mano.shape[0]:
                    raise ValueError(
                        f"arm_keypoints.npz lengths (elbow={kp_elbow.shape[0]}, link7={kp_link7.shape[0]}) "
                        f"!= mano_kpts length {kp_mano.shape[0]} in {path.parent}"
                    )
            else:
                kp_elbow = np.zeros((wp.shape[0], 1, 3), dtype=np.float32)
                kp_link7 = np.zeros((wp.shape[0], 1, 3), dtype=np.float32)
                print(f"[warn] arm_keypoints.npz missing at {arm_kp_path}; using zeros. "
                      "Run scripts/process_dataset/process_arm_pipeline.py first.")
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
            # Matches TJ's `obj_angvel_seq` semantics (precomputed actual angvel).
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
        # generated by scripts/process_dataset/process_arm_pipeline.py).
        frame0_arm_list = []
        for path in traj_files:
            ik_path = path.parent / "arm_joint_pos_shadow.npy"
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
    def _apply_object_mass_from_json(cfg: RobotisShadowGraspEnvCfg) -> None:
        """Override cfg.object_mass_min/max from the per-object mass JSON if available."""
        # Resolve dataset-aware path: each dataset uses its own
        # data/processed/<dataset>/object_mass.json by default. ``cfg.object_mass_json``
        # acts as an explicit override when non-empty.
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
            print(f"[grasp] WARNING: Could not load object_mass_json ({json_path}): {e}")
            return
        _DEFAULT_MIN, _DEFAULT_MAX = 0.05, 0.20

        entry = mass_table.get(cfg.object_id)
        if entry is None:
            lo, hi = _DEFAULT_MIN, _DEFAULT_MAX
            print(f"[grasp] object_mass: {cfg.object_id} not in JSON → default [{lo:.3f}, {hi:.3f}] kg")
        elif entry[0] is None or entry[1] is None:
            lo, hi = _DEFAULT_MIN, _DEFAULT_MAX
            print(f"[grasp] object_mass: {cfg.object_id} has null in JSON → default [{lo:.3f}, {hi:.3f}] kg")
        else:
            lo, hi = float(entry[0]), float(entry[1])
            print(f"[grasp] object_mass from JSON: {cfg.object_id} → [{lo:.3f}, {hi:.3f}] kg")
        cfg.object_mass_min = lo
        cfg.object_mass_max = hi

    def _build_object_cfg(self, cfg: RobotisShadowGraspEnvCfg) -> RigidObjectCfg:
        _data_root = Path(cfg.hocap_data_dir if cfg.dataset == "hocap" else cfg.oakink_data_dir)
        usd_path = _data_root / "assets" / "objects" / cfg.object_id / "visual.usd"
        if not usd_path.exists():
            raise FileNotFoundError(
                f"Object USD not found: {usd_path}\n"
                f"Run: isaaclab.sh -p scripts/process_dataset/convert_obj_to_usd.py "
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

        # Static table: two stacked cuboids — base body + tabletop slab.
        # The slab overhangs in +y (toward robot) by `tabletop_overhang_y_pos`, while the
        # base body keeps the original footprint so the robot's torso can fit under the
        # overhang without colliding with the table legs/body.
        table_w, table_d, table_h = self.cfg.table_size
        table_x, table_y, _ = self.cfg.table_pos_env
        thickness = float(self.cfg.tabletop_thickness)
        overhang_y = float(self.cfg.tabletop_overhang_y_pos)
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.38, 0.18))

        # Table prims are spawned as KINEMATIC rigid bodies (immovable, no forces affect
        # them) so PhysX GPU contact filter can target them — required by the link7-table
        # ContactSensor (`rew_arm_contact`). Plain collision-only Cuboids would emit
        # "GPU contact filter for collider ... is not supported" warnings and produce
        # zero force_matrix_w.
        _table_rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        _table_mass_props = sim_utils.MassPropertiesCfg(mass=1.0)   # required to apply rigid-body API
        # Base body — original footprint, z ∈ [0, table_h - thickness]
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

        # Tabletop slab — extends in +y by overhang_y, z ∈ [table_h - thickness, table_h]
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
        # `arm_r_link3..link7` together cover the forearm + wrist mount (link7 carries
        # the wrist camera). Any of them resting on the table for grasp stability is
        # cheating. We track each link's contact with the table (TableBase + TableTop)
        # and use the MAX across all 5 links for (1) soft per-N penalty and
        # (2) early termination on strong press.
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

        # Static rotation from robot0_palm body frame to HOcap landmark frame.
        # HOcap stores `qpos_wrist_right` quat in landmark frame (z=wrist→middle MCP,
        # x=palm normal). Shadow Hand's robot0_palm body frame uses a different
        # axis convention (palm normal = +Y, forward = +Z, width = +X). To compare
        # robot0_palm world quat with MANO landmark quat, we right-multiply the
        # body quat by this static palm→landmark rotation. Quat values mirror the
        # `R_palm_to_landmark` rotation computed offline from Shadow Hand knuckle
        # positions at neutral pose (≈ +90° around palm Z with a small ~5° tilt
        # from middle MCP forward offset).
        self._palm_to_landmark_quat = torch.tensor(
            [-0.706025, 0.039103, -0.039103, 0.706025],
            dtype=torch.float32, device=self.device,
        )

        # Elbow position: computed from arm_r_link3 + URDF-defined joint4 offset.
        # See `_ELBOW_OFFSET_IN_LINK3_LOCAL` comment for the rationale.
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
        self._prev_action = torch.zeros(B, self.cfg.action_space, device=self.device)

        # EMA action smoothing buffer (actioned joints only, 27D); initialize at normalized default pose
        default_ctrl = torch.cat([
            self.robot.data.default_joint_pos[:1, self._finger_joint_ids],
            self.robot.data.default_joint_pos[:1, self._arm_r_joint_ids],
        ], dim=-1).squeeze(0)
        default_normalized = self._unscale(default_ctrl)
        self._smoothed_actions = default_normalized.unsqueeze(0).expand(B, -1).clone()

        # Mass-as-an-action: stores the last mass action dim per env
        # Initialize to mu_m = -0.25 (paper Section 3.2: corresponds to ≈ 0.4 × mmax).
        self._current_mass_action = torch.full((B,), -0.25, device=self.device)
        # Envs that reset on the previous step; mass update deferred to next _pre_physics_step
        # so the NEW episode's mass action (from policy cache) is applied before first physics.
        self._just_reset_env_ids: torch.Tensor | None = None

        # Adaptive rollout sampling: per-frame EMA failure count (start at zero)
        self._failure_count = torch.zeros(self._max_traj_len, device=self.device)
        # Rewind window in frames — derived from action_fps × adaptive_back_seconds (mirrors TJ).
        _action_fps = round(1.0 / (self.cfg.sim.dt * self.cfg.decimation))
        self._adaptive_back_frames: int = int(_action_fps * self.cfg.adaptive_back_seconds)

        # State cache (Shadow Hand variant): stores simulation state at each trajectory frame.
        # Layout (91-dim): reward(1) + obj_pos_local(3) + obj_quat(4) + obj_linvel(3) + obj_angvel(3)
        #                  + joint_pos(26: 18 fingers + 7 arm + 1 lift)
        #                  + joint_vel(26) + smoothed_act(25: 18+7, lift excluded)
        _STATE_DIM = 91
        self._state_cache = torch.zeros(self._max_traj_len, _STATE_DIM, device=self.device)
        self._state_cache[:, 0] = -float("inf")  # reward col: any real reward beats -inf
        self._init_flg = torch.ones(self._max_traj_len, dtype=torch.bool, device=self.device)  # True = ref data
        self._reached_frame: int = 0  # furthest frame with sustained good tracking
        # TJ-style: force-save frame 0 cache once on first reset so subsequent resets reuse the IK-lifted pose.
        self._init_save_done: bool = False

        # Per-episode tracking quality (for enough_idx and reached_frame update)
        self._enough_continued = torch.ones(B, dtype=torch.bool, device=self.device)
        self._enough_idx = torch.zeros(B, dtype=torch.long, device=self.device)

        # Episode-done flag (set in _get_dones, cleared in _reset_idx)
        self._done_env = torch.zeros(B, dtype=torch.bool, device=self.device)

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

        # Effort-saturation diagnostic buffers (cfg-gated).
        self._sat_acc = torch.zeros(self.robot.num_joints, device=self.device)
        self._sat_step_count: int = 0

    def _log_effort_saturation(self) -> None:
        """Diagnostic: print joints whose applied torque hits ≥99 % of effort_limit.

        Called every step but only emits a summary every `effort_saturation_log_interval`
        steps. Toggle via `cfg.log_effort_saturation = True`.
        """
        if not self.cfg.log_effort_saturation:
            return
        applied = self.robot.data.applied_torque                       # (B, N_joints)
        limits = self.robot.data.joint_effort_limits                   # (B, N_joints)
        sat_mask = applied.abs() >= 0.99 * limits.clamp(min=1e-6)
        self._sat_acc += sat_mask.float().mean(dim=0)                  # accumulate per-joint env mean
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
        - Fingertips (MANO indices 4, 8, 12, 16, 20): link origin + local offset.
        - Elbow (MANO idx 21): `arm_r_link3` origin + `_ELBOW_OFFSET_IN_LINK3_LOCAL`,
          rotated by link3's world quaternion. This locates the actual joint pivot and
          stays stable under joint4 rotation (unlike `body_pos_w[arm_r_link4]`).

        Returns:
            hand_kpts_pos: (B, 23, 3) world-space keypoint positions.
        """
        B = self.num_envs
        hand_kpts_pos = torch.zeros(B, _NUM_KPTS, 3, device=self.device)
        # Non-fingertip (excl. elbow): body link origins (no offset)
        body_pos = self.robot.data.body_pos_w[:, self._kpt_body_ids_t, :]  # (B, K, 3)
        hand_kpts_pos[:, self._kpt_mano_indices_t, :] = body_pos
        # Fingertip: link origin + local offset
        ft_pos = self._compute_fingertip_positions()  # (B, 5, 3)
        hand_kpts_pos[:, self._kpt_ft_mano_indices_t, :] = ft_pos
        # Elbow (kpt 21): arm_r_link3 origin + URDF-defined joint4 offset, rotated.
        if self._link3_body_id is not None:
            link3_pos = self.robot.data.body_pos_w[:, self._link3_body_id, :]    # (B, 3)
            link3_quat = self.robot.data.body_quat_w[:, self._link3_body_id, :]  # (B, 4)
            elbow_offset_b = self._elbow_offset_local.unsqueeze(0).expand(B, -1)  # (B, 3)
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # Update mass action from the current step's actions first (mirrors GR:
        # mass_actions = mass_scale(raw_actions[:,0]) at the top of _pre_physics_step).
        # This ensures reset envs receive the NEW episode's mass (sampled by the policy
        # for this step), not the previous episode's stale value.
        self._current_mass_action = self.actions[:, -1].clone()

        # Apply mass for envs that reset on the PREVIOUS step, now that the policy has
        # returned the new episode's mass action (from _cache_action in MassDexMimicPolicy).
        # This ensures the first physics simulation of each new episode uses the correct mass.
        if self._just_reset_env_ids is not None and len(self._just_reset_env_ids) > 0:
            try:
                all_masses = self.object.root_physx_view.get_masses().clone()
                cpu_dev = all_masses.device
                reset_ids = self._just_reset_env_ids
                t = (self._current_mass_action[reset_ids].clamp(-1.0, 1.0) + 1.0) / 2.0
                new_masses = self.cfg.object_mass_min + t * (self.cfg.object_mass_max - self.cfg.object_mass_min)
                all_masses[torch.as_tensor(reset_ids, dtype=torch.long, device=cpu_dev), 0] = new_masses.to(cpu_dev)
                all_indices = torch.arange(self.num_envs, dtype=torch.long, device=cpu_dev)
                self.object.root_physx_view.set_masses(all_masses, all_indices)
            except Exception:
                pass
            self._just_reset_env_ids = None

        # ── WARMUP ────────────────────────────────────────────────────────────
        # Only advance frame for envs that have exited warm-up.
        # Warming-up envs stay frozen at their start_frame so the target doesn't
        # move away while the hand is still trying to reach it.
        # To restore original behavior: replace with the single commented line below.
        # self._frame_idx = (self._frame_idx + 1).clamp(max=self._max_traj_len - 1)
        new_frame = (self._frame_idx + 1).clamp(max=self._max_traj_len - 1)
        if self.cfg.enable_warmup:
            self._frame_idx = torch.where(self._is_warming_up, self._frame_idx, new_frame)
        else:
            self._frame_idx = new_frame
        # ── END WARMUP ────────────────────────────────────────────────────────

        # EMA smoothing on joint actions (dims 0–26 = 20 fingers + 7 arm);
        # mass dim (27) is not smoothed. Lift is NOT in the action.
        # TJ/rl_games convention: alpha = weight on the new (raw) action.
        # Split α: hand uses action_smoothing, arm uses arm_action_smoothing (stronger smoothing → less wrist tremor).
        joint_actions = self.actions[:, :-1]  # (B, 25 for Shadow Hand: 18 fingers + 7 arm)
        a_h = self.cfg.action_smoothing
        a_a = self.cfg.arm_action_smoothing
        N_f = self.cfg.num_hand_dofs  # 18 (Shadow Hand)
        self._smoothed_actions[:, :N_f] = a_h * joint_actions[:, :N_f] + (1.0 - a_h) * self._smoothed_actions[:, :N_f]
        self._smoothed_actions[:, N_f:] = a_a * joint_actions[:, N_f:] + (1.0 - a_a) * self._smoothed_actions[:, N_f:]

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

    def _get_observations(self) -> dict:
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        B = self.num_envs

        # Robot state: all controlled joints [fingers | arm_r | lift] — 28D (lift in obs for
        # state awareness even though it is not actioned).
        joint_pos = torch.cat([
            self.robot.data.joint_pos[:, self._finger_joint_ids],
            self.robot.data.joint_pos[:, self._arm_r_joint_ids],
            self.robot.data.joint_pos[:, self._lift_joint_ids],
        ], dim=-1)  # (B, 28)
        joint_pos = self._unscale_all(joint_pos)   # normalize to [-1, 1] using 28D scales
        joint_vel = torch.cat([
            self.robot.data.joint_vel[:, self._finger_joint_ids],
            self.robot.data.joint_vel[:, self._arm_r_joint_ids],
            self.robot.data.joint_vel[:, self._lift_joint_ids],
        ], dim=-1)  # (B, 28)

        # All 22 keypoints (21 MANO + elbow) in world frame + fingertip velocities
        hand_kpts_pos = self._compute_hand_kpts_pos()                          # (B, 22, 3)
        ft_pos = hand_kpts_pos[:, self._kpt_ft_mano_indices_t, :]             # (B, 5, 3)
        if len(self._ft_body_ids) == 5:
            ft_vel = self.robot.data.body_lin_vel_w[:, self._ft_body_ids, :]  # (B, 5, 3)
        else:
            ft_vel = torch.zeros(B, 5, 3, device=self.device)

        # Wrist global state: rotation, linear and angular velocity (position in hand_kpts_pos[0])
        # robot0_palm body quat is in palm-body frame; right-multiply by palm→landmark
        # quat to express the orientation in HOcap landmark frame (matches MANO ref).
        if self._wrist_body_id is not None:
            wrist_quat_body  = self.robot.data.body_quat_w[:, self._wrist_body_id, :]    # (B, 4)
            wrist_quat_obs   = quat_mul(
                wrist_quat_body,
                self._palm_to_landmark_quat.unsqueeze(0).expand(B, -1),
            )
            wrist_linvel_obs = self.robot.data.body_lin_vel_w[:, self._wrist_body_id, :] # (B, 3)
            wrist_angvel_obs = self.robot.data.body_ang_vel_w[:, self._wrist_body_id, :] # (B, 3)
        else:
            wrist_quat_obs   = torch.zeros(B, 4, device=self.device)
            wrist_linvel_obs = torch.zeros(B, 3, device=self.device)
            wrist_angvel_obs = torch.zeros(B, 3, device=self.device)

        # Object state
        obj_pos    = self.object.data.root_pos_w      # (B, 3)
        obj_quat   = self.object.data.root_quat_w     # (B, 4)
        obj_linvel = self.object.data.root_lin_vel_w  # (B, 3)
        obj_angvel = self.object.data.root_ang_vel_w  # (B, 3)

        # Reference look-ahead: next-frame reference positions
        env_orig = self.scene.env_origins
        next_frame = (frame + 1).clamp(max=self._max_traj_len - 1)
        ref_ft_pos   = self._ref_ft_pos[traj, next_frame]   + env_orig.unsqueeze(1)  # (B, 5, 3)
        ref_obj_pos  = self._ref_obj_pos[traj, next_frame]  + env_orig               # (B, 3)
        ref_obj_quat = self._ref_obj_quat[traj, next_frame]                          # (B, 4)

        # Delta keypoints in world frame: 22 keypoints (21 MANO + elbow) vs raw next-frame reference
        ref_kpts_nf = self._ref_mano_kpts[traj, next_frame] + env_orig.unsqueeze(1)  # (B, 22, 3)
        delta_kpts_world = hand_kpts_pos - ref_kpts_nf                               # (B, 22, 3)

        # Delta fingertip in object local frame (contact-conditioned, paper S4.1)
        # Contact fingers: target nearest mesh vertex; Non-contact: target ref fingertip
        ref_vertex_local = self._ref_contact_vertex_local[traj, next_frame]  # (B, 5, 3) obj-local
        contact_flag_next = self._future_contact[traj, next_frame]            # (B, 5)
        obj_quat_exp = obj_quat.unsqueeze(1).expand(-1, 5, -1).reshape(B * 5, 4)
        ft_in_obj = quat_apply_inverse(
            obj_quat_exp,
            (ft_pos - obj_pos.unsqueeze(1)).reshape(B * 5, 3),
        ).reshape(B, 5, 3)
        ref_ft_in_obj = quat_apply_inverse(
            obj_quat_exp,
            (ref_ft_pos - obj_pos.unsqueeze(1)).reshape(B * 5, 3),
        ).reshape(B, 5, 3)
        target_in_obj = torch.where(contact_flag_next.unsqueeze(-1).bool(), ref_vertex_local, ref_ft_in_obj)
        delta_ft_obj = ft_in_obj - target_in_obj                              # (B, 5, 3)

        # Delta object pose toward next reference frame
        delta_obj_pos = obj_pos - ref_obj_pos
        q_err = quat_mul(obj_quat, quat_conjugate(ref_obj_quat))
        delta_obj_rot_6d = quat_to_6d(q_err)  # TJ uses 6D rotation representation

        future_contact   = self._future_contact[traj, frame]
        fingertip_forces = self._get_fingertip_forces()

        # TJ-style scaling on angular and joint velocities
        vs = self.cfg.vel_obs_scale

        obs = torch.cat([
            # Hand keypoint positions: 21 MANO keypoints (kpts 0–20); elbow + link7 separated below
            hand_kpts_pos[:, :21].reshape(B, 63),       # [63]
            hand_kpts_pos[:, 21],                       # [3]  right elbow position
            hand_kpts_pos[:, 22],                       # [3]  arm_r_link7 position
            # Hand global state (TJ: 6D rotation rep + vel_obs_scale on angvel)
            quat_to_6d(wrist_quat_obs),     # [6]
            wrist_linvel_obs,               # [3]
            vs * wrist_angvel_obs,          # [3]
            # Fingertip velocities
            ft_vel.reshape(B, 15),          # [15]
            # Hand DOF (TJ: vel_obs_scale on joint vel)
            joint_pos,                      # [28]
            vs * joint_vel,                 # [28]
            # Object state (TJ: 6D rot + vel_obs_scale on angvel)
            obj_pos,                        # [3]
            quat_to_6d(obj_quat),           # [6]
            obj_linvel,                     # [3]
            vs * obj_angvel,                # [3]
            # Delta targets (next-frame look-ahead, TJ: 6D for delta_obj_rot)
            delta_kpts_world[:, :21].reshape(B, 63),    # [63]  21 MANO kpts delta
            delta_kpts_world[:, 21],                    # [3]   right elbow delta
            delta_kpts_world[:, 22],                    # [3]   arm_r_link7 delta
            delta_ft_obj.reshape(B, 15),    # [15]
            delta_obj_pos,                  # [3]
            delta_obj_rot_6d,               # [6]
            # Contact + history + forces (TJ: mass excluded from obs)
            future_contact,                 # [5]
            self._prev_action[:, :-1],      # [27] joint prev-action only; mass excluded
            fingertip_forces,               # [5]
        ], dim=-1)
        # Total: 63+3+3+6+3+3+15+28+28+3+6+3+3+63+3+3+15+3+6+5+27+5 = 297 (same as pretrain → ckpt-compatible)
        # New +6 vs prior 291: elbow_pos +3 → elbow+link7 (3+3); delta_elbow +3 → delta_elbow+delta_link7 (3+3).

        self._prev_action = self.actions.clone()

        if self.cfg.debug_vis:
            ref_wrist_pos = self._ref_wrist_pos[traj, frame] + env_orig         # (B, 3)
            ref_elbow_pos = self._ref_mano_kpts[traj, frame, 21] + env_orig     # (B, 3) kpt 21 = elbow
            ref_link7_pos = self._ref_mano_kpts[traj, frame, 22] + env_orig     # (B, 3) kpt 22 = arm_r_link7
            self._update_debug_vis(ref_ft_pos, ft_pos, ref_wrist_pos, ref_elbow_pos, ref_link7_pos)

        return {"policy": obs}

    def _get_fingertip_forces(self) -> torch.Tensor:
        """Return per-fingertip compressive contact force (N), projected onto pad-inward direction.

        Uses `force_matrix_w` (per-filter-object contact force) — only counts contact
        with the Object (filter target), NOT self-collision or table contacts.
        Mirrors TJ's projection:
            force_along_pad = (force_w * -pad_normal_w).sum(-1).clamp_min(0)
        where `pad_normal_w` is the pad-OUTWARD unit normal in world frame.

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
                # force_matrix_w: (B, num_bodies=1, num_filter=1, 3) — contact force with Object only.
                fmat = sensor.data.force_matrix_w
                force_vec = fmat[:, 0, 0, :]          # (B, 3) force on fingertip from object
                inward = -pad_normals_w[:, i, :]      # (B, 3) pad-inward (into finger body)
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
                if _THUMB_FILTER_ENABLED and name == "robot0_thdistal" and sensor.data.contact_pos_w is not None:
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
        """Return per-env contact force magnitude (N) between any arm_r_link3..link7 and the table.

        For each tracked link: sums per-table-prim force magnitudes (any contact contributes).
        Then takes the MAX across the 5 links — a single bad link (e.g. forearm or wrist
        pressed against the table) drives the penalty/termination. Used by both
        `rew_arm_contact` (soft penalty) and the arm-strong-press termination check.
        """
        B = self.num_envs
        sensors = getattr(self, "_arm_contact_sensors", None)
        if not sensors:
            return torch.zeros(B, device=self.device)
        per_link_force = []
        for sensor in sensors.values():
            try:
                # force_matrix_w: (B, num_bodies=1, num_filter=2, 3) — per table prim.
                fmat = sensor.data.force_matrix_w
                force_per_filter = fmat[:, 0, :, :]                    # (B, 2, 3)
                per_link_force.append(force_per_filter.norm(dim=-1).sum(dim=-1))   # (B,)
            except Exception:
                per_link_force.append(torch.zeros(B, device=self.device))
        # Stack to (B, L) then take max across L tracked links.
        return torch.stack(per_link_force, dim=-1).max(dim=-1).values            # (B,)

    def _get_rewards(self) -> torch.Tensor:
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        env_orig = self.scene.env_origins

        obj_pos = self.object.data.root_pos_w
        obj_quat = self.object.data.root_quat_w

        ref_obj_pos = self._ref_obj_pos[traj, frame] + env_orig
        ref_obj_quat = self._ref_obj_quat[traj, frame]

        # Object tracking errors.
        # Unweighted error is used for termination checks and state-cache thresholds.
        # Z-weighted error (paper S4.2: "higher weights to gravity direction") is used for reward.
        delta_obj_pos = obj_pos - ref_obj_pos  # (B, 3)
        obj_pos_err = torch.norm(delta_obj_pos, dim=-1)  # unweighted — termination
        delta_obj_pos_w = delta_obj_pos.clone()
        delta_obj_pos_w[:, 2] *= 1.5
        obj_pos_err_w = torch.norm(delta_obj_pos_w, dim=-1)  # Z-weighted — reward
        q_err = quat_mul(obj_quat, quat_conjugate(ref_obj_quat))
        # arcsin(||vec||) form for numerical precision (matches TJ wrist_rot_err / hand_rot_value).
        obj_rot_err = 2.0 * torch.asin(torch.norm(q_err[:, 1:4], dim=-1).clamp(max=1.0))

        # All 22 keypoints (wrist + MCP/PIP/DIP + fingertips + right elbow).
        B = self.num_envs
        hand_kpts_pos = self._compute_hand_kpts_pos()                      # (B, 22, 3)
        ft_pos = hand_kpts_pos[:, self._kpt_ft_mano_indices_t, :]         # (B, 5, 3)

        # Reference keypoints in world frame (TJ-style: NO drift compensation — track original human motion).
        ref_kpts_local = self._ref_mano_kpts[traj, frame]                 # (B, 23, 3) env-local
        ref_kpts_world = ref_kpts_local + env_orig.unsqueeze(1)            # (B, 23, 3) world

        # Keypoint tracking error: 21 MANO kpts (Z-weighted) for rew_kpts; arm (elbow+link7) handled separately.
        delta_kpts = hand_kpts_pos - ref_kpts_world                       # (B, 23, 3)
        delta_kpts_w = delta_kpts.clone()
        delta_kpts_w[:, :, 2] *= 1.5                                       # Z-weighted (paper S4.2: gravity emphasis)
        delta_kpts_mano = delta_kpts[:, :21]                              # (B, 21, 3)  MANO kpts only, unweighted
        delta_kpts_mano_w = delta_kpts_w[:, :21]                          # (B, 21, 3)  MANO kpts only, Z-weighted
        kpts_err_w = torch.norm(delta_kpts_mano_w, dim=-1).mean(dim=-1)   # (B,) mean over 21 MANO
        self._last_kpts_err = torch.norm(delta_kpts_mano, dim=-1).mean(dim=-1)  # (B,) 21 MANO, for warmup/termination
        self._last_kpts_err_raw = self._last_kpts_err

        # Wrist error from keypoint 0 (unweighted): termination + monitoring (TJ-style raw L2).
        wrist_err = torch.norm(delta_kpts_mano[:, 0, :], dim=-1)          # (B,)

        # Arm error: wrist (kpt 0) + elbow (kpt 21) + arm_r_link7 (kpt 22), averaged. Used by `rew_arm_pos`.
        # Three "arm endpoints" tracked jointly under a single weight.
        #   - wrist (kpt 0)      → also in `kpts_err_w` mean (which averages 21 MANO kpts).
        #   - elbow (kpt 21)     → arm_r_link3 origin + URDF joint4 offset
        #                          (= true joint pivot, stable under joint4 rotation).
        #                          Reference comes from VPoser SMPL fit (Stage 2 of pipeline).
        #   - link7 (kpt 22)     → arm_r_link7 origin — last revolute link before the
        #                          FIXED wrist mount; adds positional constraint that
        #                          indirectly constrains wrist orientation.
        # - `arm_pos_err` (unweighted mean over wrist+elbow+link7): logged as `Error / arm_pos_m`.
        # - `arm_err_w`   (Z-weighted mean): used by `rew_arm_pos`.
        arm_kpt_idx = [0, 21, 22]
        # Unweighted per-arm-kpt errors used for logging aggregate `Error / arm_pos_m`.
        arm_pos_err = torch.norm(delta_kpts[:, arm_kpt_idx, :], dim=-1).mean(dim=-1)   # (B,) mean over (wrist, elbow, link7)
        arm_err_w = torch.norm(delta_kpts_w[:, arm_kpt_idx, :], dim=-1).mean(dim=-1)   # (B,) Z-weighted mean

        # Wrist rotation (needed for termination; can't derive from positions alone).
        # Convert robot0_palm body quat into HOcap landmark frame before comparing
        # against ref_wrist_quat (which is itself in landmark frame).
        wrist_rot_err = torch.zeros(B, device=self.device)
        if self._wrist_body_id is not None:
            wrist_quat_body = self.robot.data.body_quat_w[:, self._wrist_body_id, :]
            wrist_quat = quat_mul(
                wrist_quat_body,
                self._palm_to_landmark_quat.unsqueeze(0).expand(B, -1),
            )
            ref_wrist_quat = self._ref_wrist_quat[traj, frame]
            q_err = quat_mul(wrist_quat, quat_conjugate(ref_wrist_quat))
            wrist_rot_err = 2.0 * torch.asin(torch.clamp(torch.norm(q_err[:, 1:4], dim=-1), max=1.0))

        # Fingertip contact tracking (GR env style — drift-compensated `_rel` targets).
        ref_ft = self._ref_ft_pos[traj, frame] + env_orig.unsqueeze(1)    # (B, 5, 3) raw world
        contact_flag = self._future_contact[traj, frame]                   # (B, 5)

        # Contact vertices on the actual object surface (world frame, drift-comp via current obj pose).
        ref_vertex_local_r = self._ref_contact_vertex_local[traj, frame]  # (B, 5, 3) obj-local
        obj_quat_exp_r = obj_quat.unsqueeze(1).expand(-1, 5, -1).reshape(B * 5, 4)
        ref_vertex_world = quat_apply(
            obj_quat_exp_r,
            ref_vertex_local_r.reshape(B * 5, 3),
        ).reshape(B, 5, 3) + obj_pos.unsqueeze(1)

        # Non-contact target: drift-compensated ref fingertip (TJ `fingertip_pos_ref_rel`).
        # Place ref ft relative to current obj pose: q_offset * (ref_ft - ref_obj_pos) + obj_pos
        # where q_offset = q_curr * q_ref^-1 (un-rotate by ref obj quat, then rotate by current).
        ref_obj_pos_world_r = self._ref_obj_pos[traj, frame] + env_orig          # (B, 3)
        ref_obj_quat_r = self._ref_obj_quat[traj, frame]                          # (B, 4)
        ref_ft_local_to_ref_obj = ref_ft - ref_obj_pos_world_r.unsqueeze(1)       # (B, 5, 3)
        ref_obj_quat_exp = ref_obj_quat_r.unsqueeze(1).expand(-1, 5, -1).reshape(B * 5, 4)
        ref_ft_in_obj_canon = quat_apply_inverse(
            ref_obj_quat_exp, ref_ft_local_to_ref_obj.reshape(B * 5, 3)
        )
        ref_ft_drift = quat_apply(obj_quat_exp_r, ref_ft_in_obj_canon).reshape(B, 5, 3) + obj_pos.unsqueeze(1)

        contact_flag_gated = contact_flag                                  # (B, 5)
        # REWARD target: drift-compensated, contact-conditioned (TJ delta_fingertip_pos_rel_new).
        ft_target = torch.where(contact_flag_gated.unsqueeze(-1).bool(), ref_vertex_world, ref_ft_drift)
        ft_err_per_finger = torch.norm(ft_pos - ft_target, dim=-1)        # (B, 5)
        ft_err = ft_err_per_finger.mean(dim=-1)                            # (B,) — used for reward

        # TERMINATION + raw E_ft metric: RAW ref ft (TJ delta_fingertip_pos, no drift comp).
        ft_err_raw_per_finger = torch.norm(ft_pos - ref_ft, dim=-1)       # (B, 5)
        ft_err_raw = ft_err_raw_per_finger.mean(dim=-1)                    # (B,) — used for termination
        self._last_ft_raw_err = ft_err_raw

        # Contact force reward (mirrors GR train env).
        raw_forces = self._get_fingertip_forces()                              # (B, 5)
        contact_condition = (ft_err_per_finger < 0.03).float()                # (B, 5)
        fforce_contact = raw_forces * contact_flag_gated * contact_condition   # (B, 5)
        n_contacts = contact_flag_gated.sum(dim=-1, keepdim=True)             # (B, 1)
        clamped = torch.clamp(fforce_contact, 0.0, 0.5) / (n_contacts + 1e-6) / 1.5
        force_rew = clamped.sum(dim=-1)                                       # (B,)  TJ-exact: 1.0× coef

        # Regularization split by region. Shadow Hand action layout: [fingers(18) | arm_r(7) | mass(1)].
        # Pose excludes lift (PD-held → contributes noise only).
        N_f = self.cfg.num_hand_dofs   # 18 (Shadow Hand)
        N_a = self.cfg.num_arm_r_dofs  # 7
        hand_action_reg = (self.actions[:, :N_f] ** 2).sum(dim=-1)
        arm_action_reg  = (self.actions[:, N_f:N_f + N_a] ** 2).sum(dim=-1)
        jp = self.robot.data.joint_pos
        dp = self.robot.data.default_joint_pos
        hand_pose_reg = ((jp[:, self._finger_joint_ids] - dp[:, self._finger_joint_ids]) ** 2).sum(dim=-1)
        arm_pose_reg  = ((jp[:, self._arm_r_joint_ids]  - dp[:, self._arm_r_joint_ids])  ** 2).sum(dim=-1)
        # Arm-table contact (anti-cheating, link3..link7). Computed once; used for both
        # penalty and termination. `arm_table_force` is the MAX across the 5 tracked
        # arm links of (sum of per-table-prim force magnitudes) — non-negative by
        # construction, used directly without deadband. The per-step penalty is clamped
        # at `rew_arm_contact × max_arm_contact_force` (= penalty value at the termination
        # threshold) so a single above-threshold step can't dominate the reward.
        arm_table_force = self._get_arm_table_force()                                     # (B,) N
        arm_penalty = (self.cfg.rew_arm_contact * arm_table_force).clamp(
            min=self.cfg.rew_arm_contact * self.cfg.max_arm_contact_force
        )                                                                                  # (B,) ≤ 0
        arm_strong_press = arm_table_force > self.cfg.max_arm_contact_force
        self._last_arm_table_force = arm_table_force                                       # for logging
        # Compute alive signal: 0 on terminated steps, 1 otherwise (GR env pattern).
        # All termination thresholds use unweighted errors (GR env: raw L2 distances).
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
        # Absorbs the IK-lift offset and open-vs-curled finger mismatch at t=0 — same window
        # as the loose `start_condition` in `_save_state_cache`.
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

        # `rew_kpts` averages over 21 MANO kpts; `rew_arm_pos` supervises the 3 arm
        # endpoints (wrist + elbow + arm_r_link7) under a single weight.
        tracking_penalty = (
            self.cfg.rew_kpts * kpts_err_w           # Z-weighted mean over 21 MANO keypoints
            + self.cfg.rew_arm_pos * arm_err_w       # Z-weighted mean of (wrist + elbow + link7) L2
            + self.cfg.rew_obj_pos * obj_pos_err_w   # Z-weighted (paper S4.2: gravity emphasis)
            + self.cfg.rew_obj_rot * obj_rot_err
            + self.cfg.rew_fingertip * ft_err        # contact-conditioned fingertip tracking
        ).clamp(min=-self.cfg.rew_alive)

        reward = (
            self.cfg.rew_alive * alive
            + tracking_penalty
            + self.cfg.rew_fingertip_force * force_rew
            + arm_penalty                                       # soft penalty (auto-clamped at rew_arm_contact × max_arm_contact_force)
            + self.cfg.rew_hand_action_reg * hand_action_reg
            + self.cfg.rew_arm_action_reg  * arm_action_reg
            + self.cfg.rew_hand_pose_reg   * hand_pose_reg
            + self.cfg.rew_arm_pose_reg    * arm_pose_reg
        )
        reward = reward.clamp(min=0.0)

        self._save_state_cache(reward, ft_err, obj_pos_err, obj_rot_err)
        self._log_effort_saturation()

        # skrl SequentialTrainer only logs values that are torch.Tensor with numel()==1.
        # Python floats/ints are silently ignored. Use .mean() (0-dim tensor) not .mean().item().
        self.extras["log"] = {
            # Tracking errors (unweighted, for monitoring)
            "Error / kpts_mean_m":     torch.norm(delta_kpts_mano, dim=-1).mean(),
            "Error / wrist_pos_m":     wrist_err.mean(),
            "Error / wrist_rot_deg":   torch.rad2deg(wrist_rot_err).mean(),
            "Error / arm_pos_m":       arm_pos_err.mean(),     # mean over (wrist, elbow, link7); matches reward kpts
            "Error / obj_pos_m":       obj_pos_err.mean(),
            "Error / obj_rot_deg":     torch.rad2deg(obj_rot_err).mean(),
            "Error / ft_mean_m":       ft_err.mean(),
            "Force / arm_table_N":     arm_table_force.mean(),
            "Force / arm_press_rate":  arm_strong_press.float().mean(),
            # Per-component rewards (weighted values match what the optimizer sees)
            "Episode_Reward / alive":             (self.cfg.rew_alive * alive).mean(),
            "Episode_Reward / kpts":              (self.cfg.rew_kpts * kpts_err_w).mean(),
            "Episode_Reward / arm_pos":           (self.cfg.rew_arm_pos * arm_err_w).mean(),
            "Episode_Reward / obj_pos":           (self.cfg.rew_obj_pos * obj_pos_err_w).mean(),
            "Episode_Reward / obj_rot":           (self.cfg.rew_obj_rot * obj_rot_err).mean(),
            "Episode_Reward / fingertip":         (self.cfg.rew_fingertip * ft_err).mean(),
            "Episode_Reward / fingertip_force":   (self.cfg.rew_fingertip_force * force_rew).mean(),
            "Episode_Reward / arm_contact":       arm_penalty.mean(),
            "Episode_Reward / hand_action_reg":   (self.cfg.rew_hand_action_reg * hand_action_reg).mean(),
            "Episode_Reward / arm_action_reg":    (self.cfg.rew_arm_action_reg  * arm_action_reg).mean(),
            "Episode_Reward / hand_pose_reg":     (self.cfg.rew_hand_pose_reg   * hand_pose_reg).mean(),
            "Episode_Reward / arm_pose_reg":      (self.cfg.rew_arm_pose_reg    * arm_pose_reg).mean(),
            "Episode_Reward / total":             reward.mean(),
            # ── [THUMB-RADIUS-FILTER] Monitoring ──────────────────────────
            # `raw_forces[:, 0]` is the thumb force AFTER the distance gate.
            #   thumb_N         — mean compressive force on thumb pad (N)
            #   thumb_active    — fraction of envs where thumb force > 0
            #                     (proxy for "gate passed AND contact existed")
            # To revert: delete these two lines.
            # ──────────────────────────────────────────────────────────────
            "Force / thumb_N":       raw_forces[:, 0].mean(),
            "Force / thumb_active":  (raw_forces[:, 0] > 0.0).float().mean(),
            # Curriculum state
            "Curriculum / reached_frame":  torch.tensor(float(self._reached_frame), device=self.device),
            "Curriculum / warmup_ratio":   self._is_warming_up.float().mean(),
            # Mass-as-action: actual object mass in kg (unnormalized from cached mass action)
            "mass/mean":  self._current_mass_action.mean(),
            "mass/std":   self._current_mass_action.std(),
            "mass/kg_mean": (
                self.cfg.object_mass_min
                + (self._current_mass_action.clamp(-1.0, 1.0) + 1.0) / 2.0
                * (self.cfg.object_mass_max - self.cfg.object_mass_min)
            ).mean(),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Reuse early_terminate precomputed in _get_rewards() (errors already stored).
        terminated = self._early_terminate_buf

        # ── WARMUP ────────────────────────────────────────────────────────────
        # Update warm-up exit state using errors cached from _get_rewards().
        # early_terminate already has the warmup mask applied, so just update the flag.
        if self.cfg.enable_warmup and self._is_warming_up.any():
            warmup_done = (
                (self._last_ft_mean_err < self.cfg.warmup_ft_threshold)
                & (self._last_wrist_err < self.cfg.warmup_wrist_threshold)
                & (self._last_wrist_rot_err < self.cfg.warmup_wrist_rot_threshold)
            )
            self._is_warming_up = self._is_warming_up & ~warmup_done
        # ── END WARMUP ────────────────────────────────────────────────────────

        # TJ-style fixed-length episode: time-out fires when the framework step counter
        # reaches max_episode_length (= _num_frame_chunk = episode_length_s * action_fps).
        # Equivalent to the old `_frame_idx >= traj_end - 1` check because adaptive sampling
        # clamps `start_frame` so every episode runs exactly `_num_frame_chunk` steps; this
        # form just matches TJ (`episode_length_buf` based) and decouples from per-env
        # trajectory length.
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self.extras.setdefault("log", {})["success_rate"] = (
            (~terminated & ~time_out & (self._last_obj_pos_err < 0.03))
            .float().mean()
        )

        self._done_env = terminated | time_out
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        n = len(env_ids)
        super()._reset_idx(env_ids)

        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # --- Adaptive sampling: EMA failure count update using _enough_idx ---
        # Mirrors GR env: bincount the failure frames, then EMA-update the full count vector.
        # This correctly (a) weights by how many envs failed at each frame and
        # (b) decays frames that had no failures this batch (count = 0).
        if self.cfg.adaptive_sampling and hasattr(self, "reset_terminated"):
            is_terminated = self.reset_terminated[env_ids]
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
        self._prev_action[env_ids] = 0.0
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

        # Shadow Hand cache: jp(26) at [14:40], jv(26) at [40:66], sa(25) at [66:91].
        cached_jp = cached[:, 14:40]
        cached_jv = cached[:, 40:66]
        cached_sa = cached[:, 66:91]

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

        # --- TJ-style init save: on the very first reset, force-write frame 0 cache so
        # subsequent resets at frame 0 reuse the IK-lifted pose instead of re-applying IK. ---
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

        # --- Mass-as-an-action: deferred to next _pre_physics_step ---
        # We store the reset env ids here. By the time _pre_physics_step is called next,
        # the policy (MassDexMimicPolicy) will have sampled the new episode's mass and
        # stored it as the action's last dim. Applying mass there ensures the first physics
        # step of the new episode uses the correct mass (matches rl_games GR env timing).
        if isinstance(env_ids, torch.Tensor):
            self._just_reset_env_ids = env_ids.to(self.device)
        else:
            self._just_reset_env_ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

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
