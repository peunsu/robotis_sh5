"""OakInk dexterous grasping **pretrain** environment — Isaac Lab Direct RL.

Pretrain phase: the object is entirely removed from the physics simulation (paper Sec. 3.3).
The policy learns hand motion tracking (fingertips + wrist) using kinematic reference data.
All object-related inputs (position, rotation, contact targets) come from the preloaded
reference trajectory — no physics object exists in the scene. Object tracking rewards are
excluded; only fingertip and wrist tracking rewards are used.

Once pretrain is complete, the checkpoint transfers to RobotisSh5GraspEnv for full
dexterous manipulation with real object physics (observation space: 279 → 280D).
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

from .robotis_sh5_grasp_pretrain_env_cfg import RobotisSh5GraspPretrainEnvCfg

# Pretrain state cache dim (no object; 1 reward + 28 jp + 28 jv + 27 smoothed_act = 84).
_STATE_DIM_PRETRAIN = 84


# Local-frame offsets from link origin to actual fingertip contact point.
_FINGERTIP_OFFSETS: dict[str, list[float]] = {
    "finger_r_link4":  [0.0,    0.03975, 0.012],
    "finger_r_link8":  [0.012,  0.0,     0.02425],
    "finger_r_link12": [0.012,  0.0,     0.02425],
    "finger_r_link16": [0.012,  0.0,     0.02425],
    "finger_r_link20": [0.012,  0.0,     0.02425],
}


class RobotisSh5GraspPretrainEnv(DirectRLEnv):
    """Pretrain environment: track reference hand pose with a frozen (teleported) object."""

    cfg: RobotisSh5GraspPretrainEnvCfg

    def __init__(self, cfg: RobotisSh5GraspPretrainEnvCfg, render_mode: str | None = None, **kwargs):
        self._load_reference_trajectories(cfg)
        # Pretrain runs the FULL reference trajectory in each episode (no chunking).
        # Episode length matches max_traj_len so the policy learns keypoint tracking
        # across the entire sequence. This differs from TJ's 150-frame chunking but
        # avoids needing per-frame arm IK to start at arbitrary trajectory points.
        action_fps = round(1.0 / (cfg.sim.dt * cfg.decimation))
        self._num_frame_chunk = self._max_traj_len
        cfg.episode_length_s = self._num_frame_chunk / action_fps
        super().__init__(cfg, render_mode, **kwargs)
        self._post_init_buffers()

    # ------------------------------------------------------------------
    # Data loading  (identical to main grasp env)
    # ------------------------------------------------------------------

    def _load_reference_trajectories(self, cfg: RobotisSh5GraspPretrainEnvCfg) -> None:
        _data_root = Path(cfg.hocap_data_dir if cfg.dataset == "hocap" else cfg.oakink_data_dir)
        data_dir = _data_root / "mano" / "right"

        if cfg.trajectory_task:
            traj_path = (
                data_dir / cfg.trajectory_task / str(cfg.trajectory_data_id) / "trajectory_keypoints.npz"
            )
            if not traj_path.exists():
                raise FileNotFoundError(
                    f"Trajectory not found: {traj_path}\n"
                    f"Check trajectory_task='{cfg.trajectory_task}' and "
                    f"trajectory_data_id={cfg.trajectory_data_id}."
                )
            traj_files = [traj_path]
        else:
            traj_files = sorted(data_dir.glob("*/*/trajectory_keypoints.npz"))
            traj_files = [p for p in traj_files if cfg.object_id in p.parent.parent.name]

        if not traj_files:
            raise FileNotFoundError(
                f"No trajectories found for object '{cfg.object_id}' in {data_dir}."
            )

        wrist_pos_list, wrist_quat_list = [], []
        ft_pos_list = []
        obj_pos_list, obj_quat_list = [], []
        future_contact_list = []
        mano_kpts_list = []

        for path in traj_files:
            data = np.load(str(path))
            wp = data["qpos_wrist_right"][:, :3].astype(np.float32)
            wq = data["qpos_wrist_right"][:, 3:].astype(np.float32)
            fp = data["qpos_finger_right"][:, :, :3].astype(np.float32)
            op = data["qpos_obj_right"][:, :3].astype(np.float32)
            oq = data["qpos_obj_right"][:, 3:].astype(np.float32)
            if "mano_kpts_right" in data:
                kp = data["mano_kpts_right"].astype(np.float32)       # (N, 21, 3)
            else:
                kp = np.zeros((wp.shape[0], 21, 3), dtype=np.float32)
                print(f"[warn] mano_kpts_right missing in {path}. Re-run oakink.py --overwrite.")

            N = wp.shape[0]  # noqa: F841
            # future_contact mirrors GR env is_contact:
            #   1) object is being moved (linvel > 0.05 m/s)
            #   2) fingertip is near the object surface (dist < contact_dist_threshold)
            action_fps = round(1.0 / (cfg.sim.dt * cfg.decimation))
            obj_linvel = np.concatenate(
                [(op[1:] - op[:-1]) * action_fps, np.zeros((1, 3), dtype=np.float32)], axis=0
            )  # (N, 3)
            obj_angvel_ref = np.concatenate(
                [(oq[1:] - oq[:-1]) * action_fps, np.zeros((1, 4), dtype=np.float32)], axis=0
            )  # crude finite-diff of quaternion components as a proxy for angular speed
            obj_speed = np.linalg.norm(obj_linvel, axis=-1)           # (N,)
            obj_angspeed = np.linalg.norm(obj_angvel_ref, axis=-1)    # (N,) proxy for angvel
            # GR env: contact if (linvel > 0.05 OR angvel > 0.25) AND fingertip near object
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

        # Use frame-0 rotated mesh z_min so the object's actual bottom rests on the table
        # (canonical mesh_z_min is wrong when frame 0 has the object pre-rotated, e.g.
        # C22001-0001-0010 lying on its side floats ~13 cm above the table otherwise).
        mesh_path = _data_root / "assets" / "objects" / cfg.object_id / "visual.obj"
        if mesh_path.exists():
            _mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
            _mesh_verts = np.array(_mesh.vertices, dtype=np.float32)
        else:
            _mesh_verts = None
            print(f"[warn] Centered mesh not found at {mesh_path}; assuming mesh_z_min=0.")

        table_surface_z = float(cfg.table_size[2])

        def _rotate_verts_by_quat(verts: np.ndarray, q: np.ndarray) -> np.ndarray:
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1-2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x + y*y)],
            ], dtype=np.float32)
            return verts @ R.T

        for i in range(len(obj_pos_list)):
            if _mesh_verts is not None:
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
                print(f"[pretrain] traj[0] rotated_z_min={rotated_z_min:.4f}, "
                      f"target_centroid_z={target_centroid_z:.4f}")

        if cfg.canonical_ref_pos_env is not None:
            ref_xy = np.array(cfg.canonical_ref_pos_env[:2], dtype=np.float32)
        else:
            # Use the center of the table edge facing the robot.
            # This is more stable than the robot body position and directly encodes
            # the approach direction from which the arm enters the table workspace.
            tx, ty = float(cfg.table_pos_env[0]), float(cfg.table_pos_env[1])
            hx, hy = float(cfg.table_size[0]) / 2.0, float(cfg.table_size[1]) / 2.0
            dx = float(cfg.robot_cfg.init_state.pos[0]) - tx
            dy = float(cfg.robot_cfg.init_state.pos[1]) - ty
            if abs(dy) >= abs(dx):
                ref_xy = np.array([tx, ty + hy * np.sign(dy)], dtype=np.float32)
            else:
                ref_xy = np.array([tx + hx * np.sign(dx), ty], dtype=np.float32)
        obj_xy_ref = np.array([cfg.table_pos_env[0], cfg.table_pos_env[1]], dtype=np.float32)
        canonical_dir = ref_xy - obj_xy_ref
        print(f"[pretrain] canonical ref XY: {ref_xy}, dir: {canonical_dir}")
        canonical_norm = np.linalg.norm(canonical_dir)
        if canonical_norm > 1e-6:
            canonical_dir /= canonical_norm

            for i in range(len(obj_pos_list)):
                o0 = obj_pos_list[i][0]
                w0 = wrist_pos_list[i][0]
                wrist_dir = w0[:2] - o0[:2]
                wrist_norm = np.linalg.norm(wrist_dir)
                if wrist_norm < 1e-4:
                    continue
                wrist_dir /= wrist_norm

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

                hw = angle / 2.0
                q_rw = float(np.cos(hw))
                q_rz = float(np.sin(hw))

                def _rot_quat(q: np.ndarray, _w1: float = q_rw, _z1: float = q_rz) -> np.ndarray:
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
                #   contact iff (linvel>0.05 OR angvel>0.25) AND (cv_dist - min_cv_dist < 0.015)
                op_arr = obj_pos_list[i]
                oq_arr = obj_quat_list[i]
                _lv = np.concatenate([(op_arr[1:]-op_arr[:-1])*_action_fps_cv,
                                      np.zeros((1,3),dtype=np.float32)], axis=0)
                _av = np.concatenate([(oq_arr[1:]-oq_arr[:-1])*_action_fps_cv,
                                      np.zeros((1,4),dtype=np.float32)], axis=0)
                vel_cond = (np.linalg.norm(_lv,axis=-1)>0.05) | (np.linalg.norm(_av,axis=-1)>0.25)
                cv_min   = cv_dist.min(axis=-1, keepdims=True)            # (N, 1)
                near_vtx = (cv_dist - cv_min) < 0.015                    # (N, 5)
                future_contact_list[i] = (vel_cond[:, None] & near_vtx).astype(np.float32)
                contact_vertex_local_list.append(cv_local)
        else:
            print("[pretrain] visual.obj not found; contact vertex data unavailable; using centroid fallback.")
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
        self._ref_mano_kpts = torch.from_numpy(_pad(mano_kpts_list, (21, 3)))
        self._traj_lengths = torch.tensor([a.shape[0] for a in wrist_pos_list], dtype=torch.long)
        self._max_traj_len = max_len
        self._n_trajs = n_traj

        # Load precomputed frame-0 arm IK solutions (7 arm joints per trajectory).
        # Generated by: python scripts/process_dataset/compute_frame0_ik.py
        frame0_arm_list = []
        for path in traj_files:
            ik_path = path.parent / "frame0_arm_joint_pos.npy"
            if ik_path.exists():
                frame0_arm_list.append(np.load(str(ik_path)).astype(np.float32))
            else:
                frame0_arm_list.append(None)

        if all(x is not None for x in frame0_arm_list):
            self._frame0_arm_joint_pos = torch.from_numpy(
                np.stack(frame0_arm_list, axis=0)  # (n_trajs, 7)
            )
            print(f"[pretrain] Loaded precomputed arm IK for {n_traj} trajectories.")
        else:
            self._frame0_arm_joint_pos = None
            missing = sum(1 for x in frame0_arm_list if x is None)
            print(
                f"[pretrain] {missing}/{n_traj} trajectories missing frame0_arm_joint_pos.npy; "
                "using default arm pose. Run scripts/process_dataset/compute_frame0_ik.py first."
            )

        print(f"[pretrain] Loaded {n_traj} trajectories for '{cfg.object_id}', max_len={max_len}")

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Static table: two stacked cuboids — base body + tabletop slab (overhangs +y).
        table_w, table_d, table_h = self.cfg.table_size
        table_x, table_y, _ = self.cfg.table_pos_env
        thickness = float(self.cfg.tabletop_thickness)
        overhang_y = float(self.cfg.tabletop_overhang_y_pos)
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.38, 0.18))

        base_h = table_h - thickness
        base_spawner = sim_utils.CuboidCfg(
            size=(table_w, table_d, base_h),
            collision_props=sim_utils.CollisionPropertiesCfg(),
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
            visual_material=mat,
        )
        top_spawner.func(
            "/World/envs/env_.*/TableTop",
            top_spawner,
            translation=(table_x, top_y, table_h - thickness / 2),
        )

        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensors: dict = {}  # no object in pretrain; forces always zero

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.cfg.debug_vis:
            self._setup_debug_vis()

    def _post_init_buffers(self) -> None:
        for attr in (
            "_ref_wrist_pos", "_ref_wrist_quat", "_ref_ft_pos",
            "_ref_obj_pos", "_ref_obj_quat", "_future_contact",
            "_ref_contact_vertex_local", "_ref_mano_kpts", "_traj_lengths",
        ):
            setattr(self, attr, getattr(self, attr).to(self.device))

        if self._frame0_arm_joint_pos is not None:
            self._frame0_arm_joint_pos = self._frame0_arm_joint_pos.to(self.device)

        self._finger_joint_ids, _ = self.robot.find_joints(self.cfg.finger_joint_names)
        self._arm_r_joint_ids, _ = self.robot.find_joints(self.cfg.arm_r_joint_names)
        self._lift_joint_ids, _ = self.robot.find_joints(self.cfg.lift_joint_name)

        self._ft_body_ids = self._resolve_fingertip_ids()

        offsets = []
        for name in self.cfg.fingertip_body_names:
            off = _FINGERTIP_OFFSETS.get(name, [0.0, 0.0, 0.0])
            offsets.append(off)
        self._ft_offsets = torch.tensor(offsets, dtype=torch.float32, device=self.device)

        wrist_ids, _ = self.robot.find_bodies(self.cfg.wrist_body_name)
        self._wrist_body_id: int | None = wrist_ids[0] if wrist_ids else None
        if self._wrist_body_id is None:
            print(f"[warn] Wrist body '{self.cfg.wrist_body_name}' not found; wrist rotation tracking disabled.")

        # Resolve body IDs for all 16 non-fingertip MANO keypoints
        from .robotis_sh5_grasp_env import _MANO_NON_FT_BODY_NAMES, _MANO_FT_INDICES
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

        # `_all_joint_ids` (28) — for observation (joint_pos/joint_vel include lift).
        # `_action_joint_ids` (27) — subset used by the action; lift held at fixed_lift_target.
        self._all_joint_ids = self._finger_joint_ids + self._arm_r_joint_ids + self._lift_joint_ids
        self._action_joint_ids = self._finger_joint_ids + self._arm_r_joint_ids

        # Joint limits for normalization. Two variants:
        #   _ctrl_lower/_ctrl_upper       (27,) — action joints only; used by _scale/_unscale.
        #   _ctrl_lower_all/_ctrl_upper_all (28,) — full controlled set incl. lift; used by
        #                                          _unscale_all for the 28D joint_pos obs slice.
        # Actions are mapped from [-1, 1] to the joint range via:
        #   target = 0.5 * (action + 1) * (upper - lower) + lower
        dof_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        action_ids_t = torch.tensor(self._action_joint_ids, dtype=torch.long, device=self.device)
        all_ids_t = torch.tensor(self._all_joint_ids, dtype=torch.long, device=self.device)
        self._ctrl_lower = dof_limits[0, action_ids_t, 0]  # (27,)
        self._ctrl_upper = dof_limits[0, action_ids_t, 1]  # (27,)
        self._ctrl_lower_all = dof_limits[0, all_ids_t, 0]  # (28,)
        self._ctrl_upper_all = dof_limits[0, all_ids_t, 1]  # (28,)

        # Fixed lift target broadcast to envs each step.
        self._lift_target = torch.full(
            (self.num_envs, len(self._lift_joint_ids)), float(self.cfg.fixed_lift_target),
            device=self.device,
        )

        B = self.num_envs
        self._traj_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        self._frame_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        self._prev_action = torch.zeros(B, self.cfg.action_space, device=self.device)
        # Initialize smoothed actions at default pose (action joints only — 27D in [-1, 1]).
        default_ctrl = torch.cat([
            self.robot.data.default_joint_pos[:1, self._finger_joint_ids],
            self.robot.data.default_joint_pos[:1, self._arm_r_joint_ids],
        ], dim=-1).squeeze(0)  # (27,)
        default_normalized = self._unscale(default_ctrl)  # (27,) in [-1, 1]
        self._smoothed_actions = default_normalized.unsqueeze(0).expand(B, -1).clone()

        # ── WARMUP ────────────────────────────────────────────────────────────
        # Per-env warm-up flag. Pretrain has no state cache, so the robot always
        # starts from the default pose and always needs warm-up to reach frame 0.
        # To restore original behavior: remove this line and all [WARMUP] blocks below.
        self._is_warming_up = torch.ones(B, dtype=torch.bool, device=self.device)
        # ── END WARMUP ────────────────────────────────────────────────────────

        # Shared buffers between _get_rewards() and _get_dones() to avoid double computation.
        self._early_terminate_buf = torch.zeros(B, dtype=torch.bool, device=self.device)
        self._last_ft_mean_err = torch.zeros(B, device=self.device)
        self._last_wrist_err = torch.zeros(B, device=self.device)
        self._last_wrist_rot_err = torch.zeros(B, device=self.device)

        # ── Adaptive sampling state (matches train env semantics) ─────────────
        # `_failure_count` is the EMA of per-frame failure counts (updated on reset).
        # `_enough_idx[env]` tracks the latest frame at which tracking was "good".
        # `_reached_frame` is the curriculum frontier (max frame any env reached well).
        self._failure_count = torch.zeros(self._max_traj_len, device=self.device)
        # Rewind window in frames — derived from action_fps × adaptive_back_seconds (mirrors TJ).
        _action_fps = round(1.0 / (self.cfg.sim.dt * self.cfg.decimation))
        self._adaptive_back_frames: int = int(_action_fps * self.cfg.adaptive_back_seconds)
        self._reached_frame: int = 0
        self._enough_continued = torch.ones(B, dtype=torch.bool, device=self.device)
        self._enough_idx = torch.zeros(B, dtype=torch.long, device=self.device)

        # ── State cache (pretrain: no object, 84D) ────────────────────────────
        # Per-frame "best so far" sim state. Layout:
        #   [0]      reward
        #   [1:29]   joint_pos (28 = fingers 20 + arm_r 7 + lift 1)
        #   [29:57]  joint_vel (28)
        #   [57:84]  smoothed_action (27, lift excluded)
        # On reset with adaptive_sampling, cache-hit frames restore this state
        # → robot starts in a previously-good pose, no warmup needed.
        self._state_cache = torch.zeros(self._max_traj_len, _STATE_DIM_PRETRAIN, device=self.device)
        self._state_cache[:, 0] = -float("inf")  # reward col: any real reward beats -inf
        self._init_flg = torch.ones(self._max_traj_len, dtype=torch.bool, device=self.device)
        # TJ-style: force-save frame 0 cache once on first reset so subsequent resets reuse the IK-lifted pose.
        self._init_save_done: bool = False

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalized actions [-1, 1] → action-joint positions [lower, upper] (27D)."""
        return 0.5 * (x + 1.0) * (self._ctrl_upper - self._ctrl_lower) + self._ctrl_lower

    def _unscale(self, q: torch.Tensor) -> torch.Tensor:
        """Map action-joint positions [lower, upper] → normalized [-1, 1] (27D)."""
        return (2.0 * q - self._ctrl_upper - self._ctrl_lower) / (self._ctrl_upper - self._ctrl_lower).clamp(min=1e-6)

    def _unscale_all(self, q: torch.Tensor) -> torch.Tensor:
        """Map full controlled-joint positions (incl. lift) → normalized [-1, 1] (28D)."""
        return (2.0 * q - self._ctrl_upper_all - self._ctrl_lower_all) / (self._ctrl_upper_all - self._ctrl_lower_all).clamp(min=1e-6)

    def _resolve_fingertip_ids(self) -> torch.Tensor:
        ids = []
        for name in self.cfg.fingertip_body_names:
            found, _ = self.robot.find_bodies(name)
            if found:
                ids.append(found[0])
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _compute_fingertip_positions(self) -> torch.Tensor:
        B = self.num_envs
        if len(self._ft_body_ids) != 5:
            return torch.zeros(B, 5, 3, device=self.device)

        link_pos = self.robot.data.body_pos_w[:, self._ft_body_ids, :]
        link_quat = self.robot.data.body_quat_w[:, self._ft_body_ids, :]
        offsets = self._ft_offsets.unsqueeze(0).expand(B, -1, -1)
        rotated = quat_apply(
            link_quat.reshape(B * 5, 4),
            offsets.reshape(B * 5, 3),
        ).reshape(B, 5, 3)
        return link_pos + rotated

    def _compute_fingertip_normals(self) -> torch.Tensor:
        B = self.num_envs
        if len(self._ft_body_ids) != 5:
            return torch.zeros(B, 5, 3, device=self.device)

        link_quat = self.robot.data.body_quat_w[:, self._ft_body_ids, :]
        offsets = self._ft_offsets.unsqueeze(0).expand(B, -1, -1)
        normals = quat_apply(
            link_quat.reshape(B * 5, 4),
            offsets.reshape(B * 5, 3),
        ).reshape(B, 5, 3)
        return normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def _compute_hand_kpts_pos(self) -> torch.Tensor:
        """Compute world-space positions for all 21 MANO keypoints from robot state."""
        B = self.num_envs
        hand_kpts_pos = torch.zeros(B, 21, 3, device=self.device)
        body_pos = self.robot.data.body_pos_w[:, self._kpt_body_ids_t, :]  # (B, 16, 3)
        hand_kpts_pos[:, self._kpt_mano_indices_t, :] = body_pos
        ft_pos = self._compute_fingertip_positions()  # (B, 5, 3)
        hand_kpts_pos[:, self._kpt_ft_mano_indices_t, :] = ft_pos
        return hand_kpts_pos

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def _setup_debug_vis(self) -> None:
        n = min(self.cfg.debug_vis_num_envs, self.num_envs)

        def _sphere_cfg(prim_path: str, radius: float, color: tuple) -> VisualizationMarkersCfg:
            return VisualizationMarkersCfg(
                prim_path=prim_path,
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    )
                },
            )

        self._vis_ref_ft = VisualizationMarkers(
            _sphere_cfg("/Visuals/pretrain/ref_fingertips", 0.012, (0.0, 1.0, 0.0))
        )
        self._vis_actual_ft = VisualizationMarkers(
            _sphere_cfg("/Visuals/pretrain/actual_fingertips", 0.010, (0.0, 0.8, 1.0))
        )
        self._vis_ref_wrist = VisualizationMarkers(
            _sphere_cfg("/Visuals/pretrain/ref_wrist", 0.020, (1.0, 0.0, 1.0))
        )
        self._debug_vis_n = n

    def _update_debug_vis(
        self,
        ref_ft_pos: torch.Tensor,
        ft_pos: torch.Tensor,
        ref_wrist_pos: torch.Tensor,
    ) -> None:
        n = self._debug_vis_n
        self._vis_ref_ft.visualize(translations=ref_ft_pos[:n].reshape(n * 5, 3))
        self._vis_actual_ft.visualize(translations=ft_pos[:n].reshape(n * 5, 3))
        self._vis_ref_wrist.visualize(translations=ref_wrist_pos[:n])

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # ── WARMUP ────────────────────────────────────────────────────────────
        # Only advance frame for envs that have exited warm-up.
        # Warming-up envs stay frozen at frame 0 so the target doesn't move
        # away while the hand is still reaching the initial reference position.
        # To restore original behavior: replace with the single commented line below.
        # self._frame_idx = (self._frame_idx + 1).clamp(max=self._max_traj_len - 1)
        new_frame = (self._frame_idx + 1).clamp(max=self._max_traj_len - 1)
        if self.cfg.enable_warmup:
            self._frame_idx = torch.where(self._is_warming_up, self._frame_idx, new_frame)
        else:
            self._frame_idx = new_frame
        # ── END WARMUP ────────────────────────────────────────────────────────

        # EMA smoothing (27 actioned joint dims; lift is NOT in action).
        # TJ/rl_games convention: alpha = weight on the new (raw) action.
        # Split α: hand uses action_smoothing, arm uses arm_action_smoothing (stronger smoothing → less wrist tremor).
        a_h = self.cfg.action_smoothing
        a_a = self.cfg.arm_action_smoothing
        self._smoothed_actions[:, :20] = a_h * self.actions[:, :20] + (1.0 - a_h) * self._smoothed_actions[:, :20]
        self._smoothed_actions[:, 20:] = a_a * self.actions[:, 20:] + (1.0 - a_a) * self._smoothed_actions[:, 20:]

    def _apply_action(self) -> None:
        N_f = self.cfg.num_hand_dofs   # 20
        N_a = self.cfg.num_arm_r_dofs  # 7

        # Map smoothed actions from [-1, 1] to full joint limit range (action joints only).
        targets = self._scale(self._smoothed_actions).clamp(self._ctrl_lower, self._ctrl_upper)

        self.robot.set_joint_position_target(targets[:, :N_f],         joint_ids=self._finger_joint_ids)
        self.robot.set_joint_position_target(targets[:, N_f:N_f+N_a],  joint_ids=self._arm_r_joint_ids)
        # Lift held at fixed target every step (NOT in action).
        self.robot.set_joint_position_target(self._lift_target, joint_ids=self._lift_joint_ids)


    def _get_observations(self) -> dict:
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        B = self.num_envs
        env_orig = self.scene.env_origins

        # Robot joint state — 28D including lift (lift in obs for state awareness even
        # though it is not actioned).
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

        # All 21 MANO keypoints (world frame) + fingertip velocities
        hand_kpts_pos = self._compute_hand_kpts_pos()                         # (B, 21, 3)
        ft_pos = hand_kpts_pos[:, self._kpt_ft_mano_indices_t, :]            # (B, 5, 3)
        if len(self._ft_body_ids) == 5:
            ft_vel = self.robot.data.body_lin_vel_w[:, self._ft_body_ids, :]
        else:
            ft_vel = torch.zeros(B, 5, 3, device=self.device)

        # Wrist global state: rotation, linear and angular velocity (position in hand_kpts_pos[0])
        if self._wrist_body_id is not None:
            wrist_quat   = self.robot.data.body_quat_w[:, self._wrist_body_id, :]
            wrist_linvel = self.robot.data.body_lin_vel_w[:, self._wrist_body_id, :]
            wrist_angvel = self.robot.data.body_ang_vel_w[:, self._wrist_body_id, :]
        else:
            wrist_quat   = torch.zeros(B, 4, device=self.device)
            wrist_linvel = torch.zeros(B, 3, device=self.device)
            wrist_angvel = torch.zeros(B, 3, device=self.device)

        # Reference object (teleported = reference trajectory)
        ref_obj_pos_local = self._ref_obj_pos[traj, frame]             # (B, 3) env-local
        ref_obj_quat      = self._ref_obj_quat[traj, frame]            # (B, 4)
        ref_obj_pos_world = ref_obj_pos_local + env_orig               # (B, 3) world

        # Object velocities: zeros (object is teleported every step)
        obj_linvel = torch.zeros(B, 3, device=self.device)
        obj_angvel = torch.zeros(B, 3, device=self.device)

        # Next-frame reference for look-ahead deltas
        next_frame  = (frame + 1).clamp(max=self._max_traj_len - 1)
        ref_ft_next = self._ref_ft_pos[traj, next_frame] + env_orig.unsqueeze(1)  # (B, 5, 3)

        # Delta keypoints in world frame: all 21 MANO keypoints vs raw next-frame reference
        ref_kpts_nf = self._ref_mano_kpts[traj, next_frame] + env_orig.unsqueeze(1)  # (B, 21, 3)
        delta_kpts_world = hand_kpts_pos - ref_kpts_nf                               # (B, 21, 3)

        # Delta fingertip in object local frame (contact-conditioned, paper S4.1)
        # Uses reference object orientation since object is teleported = reference
        ref_vertex_local  = self._ref_contact_vertex_local[traj, next_frame]  # (B, 5, 3) obj-local
        contact_flag_next = self._future_contact[traj, next_frame]             # (B, 5)
        ref_obj_quat_exp  = ref_obj_quat.unsqueeze(1).expand(-1, 5, -1).reshape(B * 5, 4)
        ft_in_obj = quat_apply_inverse(
            ref_obj_quat_exp,
            (ft_pos - ref_obj_pos_world.unsqueeze(1)).reshape(B * 5, 3),
        ).reshape(B, 5, 3)
        ref_ft_in_obj = quat_apply_inverse(
            ref_obj_quat_exp,
            (ref_ft_next - ref_obj_pos_world.unsqueeze(1)).reshape(B * 5, 3),
        ).reshape(B, 5, 3)
        target_in_obj = torch.where(contact_flag_next.unsqueeze(-1).bool(), ref_vertex_local, ref_ft_in_obj)
        delta_ft_obj = ft_in_obj - target_in_obj         # (B, 5, 3)

        # Delta object pose: reference current frame vs reference next frame.
        # No physics object in pretrain; delta captures reference motion dynamics so
        # the policy can learn object trajectory context even without real object physics.
        ref_obj_pos_next_w = self._ref_obj_pos[traj, next_frame] + env_orig  # (B, 3)
        ref_obj_quat_next  = self._ref_obj_quat[traj, next_frame]             # (B, 4)
        delta_obj_pos = ref_obj_pos_world - ref_obj_pos_next_w
        q_err_obj = quat_mul(ref_obj_quat, quat_conjugate(ref_obj_quat_next))
        delta_obj_rot = 2.0 * q_err_obj[:, 1:]

        future_contact   = self._future_contact[traj, frame]  # (B, 5)
        fingertip_forces = self._get_fingertip_forces()         # (B, 5)

        obs = torch.cat([
            # Hand keypoint positions: all 21 MANO keypoints in world frame (GR: hand_kpts_pos)
            hand_kpts_pos.reshape(B, 63),    # [63] all 21 keypoints (incl. wrist at idx 0)
            # Hand global state
            wrist_quat,                      # [4]  wrist rotation (wxyz)
            wrist_linvel,                    # [3]  wrist linear velocity
            wrist_angvel,                    # [3]  wrist angular velocity
            # Fingertip velocities
            ft_vel.reshape(B, 15),           # [15]
            # Hand DOF
            joint_pos,                       # [28]
            joint_vel,                       # [28]
            # Object state (reference position since object is teleported)
            ref_obj_pos_world,               # [3]  world-frame object position (matches train env)
            ref_obj_quat,                    # [4]  object rotation
            obj_linvel,                      # [3]  zeros (teleported)
            obj_angvel,                      # [3]  zeros (teleported)
            # Delta targets (next-frame look-ahead)
            delta_kpts_world.reshape(B, 63), # [63] world-frame delta for all 21 keypoints
            delta_ft_obj.reshape(B, 15),     # [15] obj-local contact-conditioned delta
            delta_obj_pos,                   # [3]
            delta_obj_rot,                   # [3]
            # Contact + history + forces
            future_contact,                  # [5]
            self._prev_action,               # [28]
            fingertip_forces,                # [5]
        ], dim=-1)
        # Total: 63+4+3+3+15+28+28+3+4+3+3+63+15+3+3+5+28+5 = 279

        self._prev_action = self.actions.clone()

        if self.cfg.debug_vis:
            ref_wrist_pos_vis = self._ref_wrist_pos[traj, frame] + env_orig
            self._update_debug_vis(ref_ft_next, ft_pos, ref_wrist_pos_vis)

        return {"policy": obs}

    def _get_fingertip_forces(self) -> torch.Tensor:
        B = self.num_envs
        forces = torch.zeros(B, 5, device=self.device)
        normals = self._compute_fingertip_normals()

        for i, name in enumerate(self.cfg.fingertip_body_names):
            sensor = self._contact_sensors.get(name)
            if sensor is None:
                continue
            try:
                net_f = sensor.data.net_forces_w
                force_vec = net_f[:, 0, :]
                normal = normals[:, i, :]
                forces[:, i] = (force_vec * normal).sum(dim=-1).clamp(min=0.0)
            except Exception:
                pass
        return forces

    def _get_rewards(self) -> torch.Tensor:
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        env_orig = self.scene.env_origins

        B = self.num_envs

        # All 21 MANO keypoints (wrist + MCP/PIP/DIP + fingertips).
        hand_kpts_pos = self._compute_hand_kpts_pos()                    # (B, 21, 3)
        ft_pos = hand_kpts_pos[:, self._kpt_ft_mano_indices_t, :]       # (B, 5, 3)

        # Reference keypoints (no drift adjustment — object is teleported = reference).
        ref_kpts_world = self._ref_mano_kpts[traj, frame] + env_orig.unsqueeze(1)  # (B, 21, 3)

        # Keypoint tracking error with Z-weighting (paper S4.2).
        delta_kpts = hand_kpts_pos - ref_kpts_world                     # (B, 21, 3)
        delta_kpts_w = delta_kpts.clone()
        delta_kpts_w[:, :, 2] *= 1.5
        kpts_err_w = torch.norm(delta_kpts_w, dim=-1).mean(dim=-1)      # (B,)

        # Wrist error from keypoint 0 (unweighted, for termination check).
        wrist_err = torch.norm(delta_kpts[:, 0, :], dim=-1)             # (B,)

        # Wrist rotation (needed for termination and delta_condition gating).
        wrist_rot_err = torch.zeros(B, device=self.device)
        if self._wrist_body_id is not None:
            wrist_quat = self.robot.data.body_quat_w[:, self._wrist_body_id, :]
            ref_wrist_quat = self._ref_wrist_quat[traj, frame]
            q_err = quat_mul(wrist_quat, quat_conjugate(ref_wrist_quat))
            wrist_rot_err = 2.0 * torch.asin(torch.clamp(torch.norm(q_err[:, 1:4], dim=-1), max=1.0))

        # delta_condition: gate contact-conditioned targets by wrist tracking quality
        # (GR pretrain: delta_condition = (hand_pos_err < 0.2) & (hand_rot_err < 0.75))
        delta_condition = (wrist_err < 0.2) & (wrist_rot_err < 0.75)
        contact_flag = self._future_contact[traj, frame]
        contact_flag_gated = contact_flag * delta_condition.unsqueeze(-1).float()

        # Fingertip contact tracking (object is teleported; ref_obj_pos == ref position).
        ref_ft = self._ref_ft_pos[traj, frame] + env_orig.unsqueeze(1)
        ref_obj_pos_w = self._ref_obj_pos[traj, frame] + env_orig
        ref_obj_quat = self._ref_obj_quat[traj, frame]
        ref_vertex_local_p = self._ref_contact_vertex_local[traj, frame]  # (B, 5, 3) obj-local
        ref_obj_quat_exp = ref_obj_quat.unsqueeze(1).expand(-1, 5, -1).reshape(B * 5, 4)
        ref_vertex_world = quat_apply(
            ref_obj_quat_exp,
            ref_vertex_local_p.reshape(B * 5, 3),
        ).reshape(B, 5, 3) + ref_obj_pos_w.unsqueeze(1)

        # Fingertip tracking: raw error (unweighted, no contact conditioning) — for termination
        ft_err_raw = torch.norm(ft_pos - ref_ft, dim=-1).mean(dim=-1)  # (B,)

        # Fingertip tracking: contact-conditioned — for reward (no Z-weighting for pretrain)
        ft_target = torch.where(contact_flag_gated.unsqueeze(-1).bool(), ref_vertex_world, ref_ft)
        ft_err = torch.norm(ft_pos - ft_target, dim=-1).mean(dim=-1)   # (B,)

        force_rew = self._get_fingertip_forces().sum(dim=-1)

        # Regularization split by region. Action layout (pretrain, no mass):
        # [fingers(20) | arm_r(7)]. Pose excludes lift (PD-held → contributes noise only).
        hand_action_reg = (self.actions[:, :20] ** 2).sum(dim=-1)
        arm_action_reg  = (self.actions[:, 20:27] ** 2).sum(dim=-1)
        jp = self.robot.data.joint_pos
        dp = self.robot.data.default_joint_pos
        hand_pose_reg = ((jp[:, self._finger_joint_ids] - dp[:, self._finger_joint_ids]) ** 2).sum(dim=-1)
        arm_pose_reg  = ((jp[:, self._arm_r_joint_ids]  - dp[:, self._arm_r_joint_ids])  ** 2).sum(dim=-1)

        # Compute alive signal using unweighted errors (termination check).
        # Mirrors GR pretrain: hand_far_apart = pos>0.15 | rot>0.75 | ft_mean>0.1
        ft_err_large = ft_err_raw > self.cfg.max_ft_mean_err
        wrist_err_large = wrist_err > self.cfg.max_wrist_pos_err
        wrist_rot_err_large = wrist_rot_err > self.cfg.max_wrist_rot_err
        early_terminate = ft_err_large | wrist_err_large | wrist_rot_err_large
        if not self.cfg.termination:
            early_terminate = torch.zeros_like(early_terminate)
        if self.cfg.enable_warmup:
            early_terminate = early_terminate & ~self._is_warming_up
        # Grace period: suppress early termination for the first N steps of each episode.
        if self.cfg.early_termination_grace_frames > 0:
            in_grace = self.episode_length_buf < self.cfg.early_termination_grace_frames
            early_terminate = early_terminate & ~in_grace
        self._early_terminate_buf = early_terminate
        self._last_ft_mean_err = ft_err_raw   # unweighted raw error for termination/warmup checks
        self._last_wrist_err = wrist_err

        self._last_wrist_rot_err = wrist_rot_err

        alive = (~early_terminate).float()

        # GR pretrain formula: 1.5*alive - clamp(1.76*kpts + 12.5*ft, 1.5) + reg
        # Fingertip term NOT Z-weighted (GR pretrain); keypoints include Z-weighted wrist
        tracking_penalty = (
            self.cfg.rew_kpts * kpts_err_w + self.cfg.rew_fingertip * ft_err
        ).clamp(min=-self.cfg.rew_alive)

        reward = (
            self.cfg.rew_alive * alive
            + tracking_penalty
            + self.cfg.rew_fingertip_force * force_rew
            + self.cfg.rew_hand_action_reg * hand_action_reg
            + self.cfg.rew_arm_action_reg  * arm_action_reg
            + self.cfg.rew_hand_pose_reg   * hand_pose_reg
            + self.cfg.rew_arm_pose_reg    * arm_pose_reg
        )
        reward = reward.clamp(min=0.0)

        # ── State cache update + adaptive sampling tracking ─────────────────
        if self.cfg.adaptive_sampling:
            self._save_state_cache(reward, ft_err_raw, wrist_err, wrist_rot_err)

        # skrl SequentialTrainer only logs values that are torch.Tensor with numel()==1.
        # Python floats/ints are silently ignored. Use .mean() (0-dim tensor) not .mean().item().
        self.extras["log"] = {
            # Tracking errors (unweighted, for interpretability)
            "Error / kpts_mean_m":     torch.norm(delta_kpts, dim=-1).mean(),
            "Error / wrist_pos_m":     wrist_err.mean(),
            "Error / wrist_rot_deg":   torch.rad2deg(wrist_rot_err).mean(),
            "Error / ft_mean_m":       ft_err_raw.mean(),
            # Per-component rewards
            "Episode_Reward / alive":             (self.cfg.rew_alive * alive).mean(),
            "Episode_Reward / kpts":              (self.cfg.rew_kpts * kpts_err_w).mean(),
            "Episode_Reward / fingertip":         (self.cfg.rew_fingertip * ft_err).mean(),
            "Episode_Reward / hand_action_reg":   (self.cfg.rew_hand_action_reg * hand_action_reg).mean(),
            "Episode_Reward / arm_action_reg":    (self.cfg.rew_arm_action_reg  * arm_action_reg).mean(),
            "Episode_Reward / hand_pose_reg":     (self.cfg.rew_hand_pose_reg   * hand_pose_reg).mean(),
            "Episode_Reward / arm_pose_reg":      (self.cfg.rew_arm_pose_reg    * arm_pose_reg).mean(),
            "Episode_Reward / total":             reward.mean(),
            # Curriculum state
            "Curriculum / reached_frame":  torch.tensor(float(self._reached_frame), device=self.device),
            "Curriculum / warmup_ratio":   self._is_warming_up.float().mean(),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Reuse early_terminate precomputed in _get_rewards() (errors already stored).
        early_terminate = self._early_terminate_buf

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

        traj_end = self._traj_lengths[self._traj_idx]  # (B,) per-env trajectory length
        time_out = self._frame_idx >= traj_end - 1
        return early_terminate, time_out

    def _save_state_cache(
        self,
        reward: torch.Tensor,
        ft_err: torch.Tensor,
        wrist_err: torch.Tensor,
        wrist_rot_err: torch.Tensor,
    ) -> None:
        """Pretrain state cache (no object). Mirrors train env semantics but uses
        wrist+ft tracking quality as the "good" criterion. Layout:
            [0]      reward
            [1:29]   joint_pos (28 = all controlled joints incl. lift)
            [29:57]  joint_vel (28)
            [57:84]  smoothed_action (27, lift excluded)
        """
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        joint_pos = self.robot.data.joint_pos[:, self._all_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._all_joint_ids]

        state = torch.cat([
            reward.unsqueeze(-1),
            joint_pos,
            joint_vel,
            self._smoothed_actions,
        ], dim=-1)  # (B, 84)

        # "good" tracking criterion for pretrain — same thresholds as warmup exit.
        # During warmup, we don't accumulate _enough_idx (the policy is still
        # converging the wrist, not actually executing the trajectory).
        good = (
            (wrist_err < self.cfg.warmup_wrist_threshold)
            & (wrist_rot_err < self.cfg.warmup_wrist_rot_threshold)
            & (ft_err < self.cfg.warmup_ft_threshold)
        )
        if self.cfg.enable_warmup:
            good = good & ~self._is_warming_up

        still_good = self._enough_continued & good
        self._enough_idx = torch.where(still_good, frame, self._enough_idx)
        self._enough_continued = still_good

        # Cache write: only if continuous-good streak AND reward beats cached reward
        # at this frame. For frames with multiple eligible envs, write the highest-reward one.
        better_reward = reward > self._state_cache[frame, 0]
        update_mask = self._enough_continued & better_reward

        if update_mask.any():
            unique_frames = torch.unique(frame[update_mask])
            for uf in unique_frames:
                mask_at_frame = (frame == uf) & update_mask
                best_local = reward[mask_at_frame].argmax()
                best_env = mask_at_frame.nonzero(as_tuple=True)[0][best_local]
                self._state_cache[uf] = state[best_env]
                self._init_flg[uf] = False
                self._reached_frame = max(self._reached_frame, int(uf.item()))

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        n = len(env_ids)
        super()._reset_idx(env_ids)

        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # ── Adaptive sampling: EMA failure count update from terminated envs ──
        if self.cfg.adaptive_sampling and hasattr(self, "reset_terminated"):
            is_terminated = self.reset_terminated[env_ids]
            if is_terminated.any():
                term_env_ids = env_ids_t[is_terminated]
                failure_frames = self._enough_idx[term_env_ids].clamp(0, self._max_traj_len - 1)
                counts = torch.bincount(failure_frames, minlength=self._max_traj_len).float()
                alpha = self.cfg.adaptive_alpha
                self._failure_count = alpha * counts + (1.0 - alpha) * self._failure_count

        # Trajectory assignment
        if self._n_trajs == 1:
            self._traj_idx[env_ids] = 0
        else:
            self._traj_idx[env_ids] = torch.randint(0, self._n_trajs, (n,), device=self.device)

        # ── Start frame sampling ──────────────────────────────────────────────
        # Match train env's two-part upper bound:
        #   upper_a = max(0, max_traj_len - num_frame_chunk)
        #     ensures episode has at least num_frame_chunk frames after start.
        #     If traj <= num_frame_chunk, upper_a = 0 → start = 0 (effectively
        #     disables adaptive sampling for short trajectories).
        #   upper_b = max(_reached_frame - adaptive_back_frames, 0)
        #     never start past the curriculum frontier.
        if self.cfg.adaptive_sampling and self._reached_frame > 0:
            valid_len = min(self._reached_frame + 1, self._max_traj_len)
            valid_counts = self._failure_count[:valid_len]
            ur = self.cfg.adaptive_uniform_ratio
            fail_probs = valid_counts / (valid_counts.sum() + 1e-8)
            probs = (1.0 - ur) * fail_probs + ur / valid_len
            sampled = torch.multinomial(probs.unsqueeze(0).expand(n, -1), 1).squeeze(-1)
            start_frames = (sampled - self._adaptive_back_frames).clamp(min=0)
            upper_a = max(0, self._max_traj_len - self._num_frame_chunk)
            upper_b = max(self._reached_frame - self._adaptive_back_frames, 0)
            upper = min(upper_a, upper_b)
            start_frames = start_frames.clamp(max=upper)
        else:
            start_frames = torch.zeros(n, dtype=torch.long, device=self.device)

        self._frame_idx[env_ids] = start_frames
        self._prev_action[env_ids] = 0.0
        # Reset per-episode tracking quality (per-env)
        self._enough_continued[env_ids] = True
        self._enough_idx[env_ids] = start_frames

        env_orig = self.scene.env_origins[env_ids]
        traj = self._traj_idx[env_ids]

        # ── Robot state restore (state cache when available, else default + IK) ──
        cached = self._state_cache[start_frames]              # (n, 84)
        has_cache = ~self._init_flg[start_frames]             # (n,) bool
        cache_mask = has_cache.unsqueeze(-1)                  # (n, 1) for broadcasting

        # cached[:, 1:29]   = joint_pos (28)
        # cached[:, 29:57]  = joint_vel (28)
        # cached[:, 57:84]  = smoothed_action (27)
        cached_jp = cached[:, 1:29]
        cached_jv = cached[:, 29:57]
        cached_sa = cached[:, 57:84]

        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos_reset = default_joint_pos.clone()
        joint_vel_reset = torch.zeros_like(default_joint_pos)

        # Apply cached joint state to controlled joints (cache hit); default elsewhere.
        joint_pos_reset[:, self._all_joint_ids] = torch.where(
            cache_mask, cached_jp, default_joint_pos[:, self._all_joint_ids]
        )
        joint_vel_reset[:, self._all_joint_ids] = torch.where(
            cache_mask, cached_jv, torch.zeros(n, len(self._all_joint_ids), device=self.device)
        )

        # When no cached state, apply frame-0 IK for the arm (default for fingers / lift).
        if self._frame0_arm_joint_pos is not None:
            arm_ik = self._frame0_arm_joint_pos[traj]   # (n, 7)
            no_cache_arm = (~has_cache).unsqueeze(-1).expand(-1, len(self._arm_r_joint_ids))
            joint_pos_reset[:, self._arm_r_joint_ids] = torch.where(
                no_cache_arm, arm_ik, joint_pos_reset[:, self._arm_r_joint_ids]
            )

        # Force lift joint to the fixed target on every reset.
        joint_pos_reset[:, self._lift_joint_ids] = self.cfg.fixed_lift_target
        joint_vel_reset[:, self._lift_joint_ids] = 0.0

        # Build smoothed_actions: cached if available, else from default + IK (27D).
        default_ctrl = torch.cat([
            default_joint_pos[:, self._finger_joint_ids],
            joint_pos_reset[:, self._arm_r_joint_ids],
        ], dim=-1)
        default_normalized = self._unscale(default_ctrl)
        self._smoothed_actions[env_ids] = torch.where(cache_mask, cached_sa, default_normalized)

        # ── WARMUP ────────────────────────────────────────────────────────────
        # Cache-hit envs are already in a physically reasonable state → no warmup.
        # Cache-miss envs (or start_frame=0 always) → warmup ON until wrist converges.
        if self.cfg.enable_warmup:
            self._is_warming_up[env_ids] = ~has_cache
        else:
            self._is_warming_up[env_ids] = False
        # ── END WARMUP ────────────────────────────────────────────────────────

        self.robot.write_joint_state_to_sim(
            joint_pos_reset, joint_vel_reset, None, env_ids
        )
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += env_orig
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # --- TJ-style init save: force-write frame 0 cache once so subsequent resets at
        # frame 0 reuse the IK-lifted pose (no object in pretrain → 84D layout). ---
        if not self._init_save_done:
            init_state = torch.cat([
                torch.zeros(1, 1, device=self.device),                      # reward placeholder
                joint_pos_reset[0:1, self._all_joint_ids],                  # (1, 28)
                joint_vel_reset[0:1, self._all_joint_ids],                  # (1, 28)
                self._smoothed_actions[env_ids_t[0:1]],                     # (1, 27)
            ], dim=-1).squeeze(0)                                           # (84,)
            self._state_cache[0] = init_state
            self._init_flg[0] = False
            self._init_save_done = True
