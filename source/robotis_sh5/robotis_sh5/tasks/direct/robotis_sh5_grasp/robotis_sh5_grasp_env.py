"""OakInk dexterous grasping environment — Isaac Lab Direct RL.

Robot: FFW-SH5 full-body (fix_root_link=True).
Policy controls right-hand fingers (20) + right arm (7) + lift (1) = 28 joint DOFs,
plus 1 mass parameter dim = 29D action total.

Incorporates MassDexMimic (NeurIPS 2026) and GR env techniques:
  - Mass-as-an-action: policy dim 28 sets object mass per episode
  - EMA action smoothing: smoothed joint commands sent to the robot
  - Next-frame look-ahead: delta observations target next reference frame
  - Wrist tracking reward: keeps wrist on the reference trajectory
  - Contact-conditioned fingertip reward: targets object center when in contact
  - Adaptive rollout sampling: resample start frames weighted by failure count
"""

from __future__ import annotations

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

from .robotis_sh5_grasp_env_cfg import RobotisSh5GraspEnvCfg

# Local-frame offsets from link origin to actual fingertip contact point.
# Derived from get_virtual_link_poses() in manager_based pick_and_place utils.py.
# finger_r_link4  (thumb):                +Y offset
# finger_r_link8/12/16/20 (index~little): +Z offset
_FINGERTIP_OFFSETS: dict[str, list[float]] = {
    "finger_r_link4":  [0.0,  0.03975, 0.0],
    "finger_r_link8":  [0.0,  0.0,     0.02425],
    "finger_r_link12": [0.0,  0.0,     0.02425],
    "finger_r_link16": [0.0,  0.0,     0.02425],
    "finger_r_link20": [0.0,  0.0,     0.02425],
}


class RobotisSh5GraspEnv(DirectRLEnv):
    """Dexterous grasping with FFW-SH5 using OakInk kinematic references."""

    cfg: RobotisSh5GraspEnvCfg

    def __init__(self, cfg: RobotisSh5GraspEnvCfg, render_mode: str | None = None, **kwargs):
        self._load_reference_trajectories(cfg)
        self._object_cfg = self._build_object_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        self._post_init_buffers()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_reference_trajectories(self, cfg: RobotisSh5GraspEnvCfg) -> None:
        """Load trajectory_keypoints.npz file(s) for the configured object and trajectory."""
        data_dir = Path(cfg.oakink_data_dir) / "mano" / "right"

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

        for path in traj_files:
            data = np.load(str(path))
            wp = data["qpos_wrist_right"][:, :3].astype(np.float32)
            wq = data["qpos_wrist_right"][:, 3:].astype(np.float32)   # wxyz
            fp = data["qpos_finger_right"][:, :, :3].astype(np.float32)  # (N, 5, 3)
            op = data["qpos_obj_right"][:, :3].astype(np.float32)
            oq = data["qpos_obj_right"][:, 3:].astype(np.float32)     # wxyz

            # future_contact: fingertip within contact_dist of object center at next frame
            N = wp.shape[0]
            next_fp = np.concatenate([fp[1:], fp[-1:]], axis=0)
            next_op = np.concatenate([op[1:], op[-1:]], axis=0)
            dist = np.linalg.norm(next_fp - next_op[:, None, :], axis=-1)  # (N, 5)
            fc = (dist < cfg.contact_dist_threshold).astype(np.float32)

            wrist_pos_list.append(wp)
            wrist_quat_list.append(wq)
            ft_pos_list.append(fp)
            obj_pos_list.append(op)
            obj_quat_list.append(oq)
            future_contact_list.append(fc)

        # Normalize all trajectories so that the object bottom rests on the table surface.
        # We need the mesh Z minimum (in the centered mesh) to offset the centroid above the table.
        # Shift wrist and fingertip positions by the same 3D offset to preserve
        # the relative geometry between hand and object.
        mesh_path = Path(cfg.oakink_data_dir) / "assets" / "objects" / cfg.object_id / "visual.obj"
        if mesh_path.exists():
            _mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
            mesh_z_min = float(_mesh.vertices[:, 2].min())
        else:
            mesh_z_min = 0.0
            print(f"[warn] Centered mesh not found at {mesh_path}; assuming mesh_z_min=0.")

        table_surface_z = float(cfg.table_size[2])
        # target centroid Z: place mesh bottom exactly at table surface
        target_centroid_z = table_surface_z - mesh_z_min
        table_target = np.array(
            [cfg.table_pos_env[0], cfg.table_pos_env[1], target_centroid_z],
            dtype=np.float32,
        )
        print(f"[grasp] mesh_z_min={mesh_z_min:.4f}, table_surface_z={table_surface_z:.4f}, "
              f"target_centroid_z={target_centroid_z:.4f}")

        for i in range(len(obj_pos_list)):
            offset = table_target - obj_pos_list[i][0]
            obj_pos_list[i] = obj_pos_list[i] + offset
            wrist_pos_list[i] = wrist_pos_list[i] + offset
            ft_pos_list[i] = ft_pos_list[i] + offset

        # Canonicalize approach direction:
        # Different OakInk sessions define their world frame with arbitrary XY orientation,
        # so after position-only normalization the reference wrist can point anywhere in XY.
        # Rotate each trajectory around Z (at the object's XY center) so the wrist direction
        # aligns with the robot's approach direction (object → robot in XY).
        robot_xy = np.array(
            [cfg.robot_cfg.init_state.pos[0], cfg.robot_cfg.init_state.pos[1]],
            dtype=np.float32,
        )
        obj_xy_ref = np.array([cfg.table_pos_env[0], cfg.table_pos_env[1]], dtype=np.float32)
        canonical_dir = robot_xy - obj_xy_ref   # direction from object toward robot in XY
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
                obj_quat_list[i] = _rot_quat(obj_quat_list[i])
                wrist_quat_list[i] = _rot_quat(wrist_quat_list[i])

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
        self._traj_lengths = torch.tensor([a.shape[0] for a in wrist_pos_list], dtype=torch.long)
        self._max_traj_len = max_len
        self._n_trajs = n_traj

        print(f"[grasp] Loaded {n_traj} trajectories for '{cfg.object_id}', max_len={max_len}")

    def _build_object_cfg(self, cfg: RobotisSh5GraspEnvCfg) -> RigidObjectCfg:
        usd_path = (
            Path(cfg.oakink_data_dir)
            / "assets" / "objects" / cfg.object_id / "visual.usd"
        )
        if not usd_path.exists():
            raise FileNotFoundError(
                f"Object USD not found: {usd_path}\n"
                "Run: python scripts/process_dataset/convert_oakink_to_usd.py"
            )
        return RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
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

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Static table (cuboid): center placed at half-height so bottom sits on ground
        table_w, table_d, table_h = self.cfg.table_size
        table_x, table_y, _ = self.cfg.table_pos_env
        table_spawner = sim_utils.CuboidCfg(
            size=(table_w, table_d, table_h),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.38, 0.18)),
        )
        table_spawner.func(
            "/World/envs/env_.*/Table",
            table_spawner,
            translation=(table_x, table_y, table_h / 2),
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
            )

        self._contact_sensors: dict[str, ContactSensor] = {}
        for name, cfg in contact_cfgs.items():
            sensor = ContactSensor(cfg)
            self._contact_sensors[name] = sensor
            self.scene.sensors[f"contact_{name}"] = sensor

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
            "_ref_obj_pos", "_ref_obj_quat", "_future_contact", "_traj_lengths",
        ):
            setattr(self, attr, getattr(self, attr).to(self.device))

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
        offsets = []
        for name in self.cfg.fingertip_body_names:
            off = _FINGERTIP_OFFSETS.get(name, [0.0, 0.0, 0.0])
            offsets.append(off)
        self._ft_offsets = torch.tensor(offsets, dtype=torch.float32, device=self.device)  # (5, 3)

        # Wrist body index for wrist tracking reward
        wrist_ids, _ = self.robot.find_bodies(self.cfg.wrist_body_name)
        self._wrist_body_id: int | None = wrist_ids[0] if wrist_ids else None
        if self._wrist_body_id is None:
            print(f"[warn] Wrist body '{self.cfg.wrist_body_name}' not found; wrist reward disabled.")

        # Precompute combined joint ID list for reward/obs (find_joints returns lists, not tensors)
        self._all_joint_ids = self._finger_joint_ids + self._arm_r_joint_ids + self._lift_joint_ids

        # Per-env buffers
        B = self.num_envs
        self._traj_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        self._frame_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        self._prev_action = torch.zeros(B, self.cfg.action_space, device=self.device)

        # EMA action smoothing buffer (joint dims only, 28D)
        self._smoothed_actions = torch.zeros(B, self.cfg.action_space - 1, device=self.device)

        # Mass-as-an-action: stores the last mass action dim per env
        self._current_mass_action = torch.zeros(B, device=self.device)

        # Adaptive rollout sampling: per-frame EMA failure count (start at zero)
        self._failure_count = torch.zeros(self._max_traj_len, device=self.device)

        # State cache: stores simulation state at each trajectory frame for physical curriculum.
        # Layout (98-dim): reward(1) + obj_pos_local(3) + obj_quat(4) + obj_linvel(3) + obj_angvel(3)
        #                  + joint_pos(28) + joint_vel(28) + smoothed_act(28)
        _STATE_DIM = 98
        self._state_cache = torch.zeros(self._max_traj_len, _STATE_DIM, device=self.device)
        self._state_cache[:, 0] = -float("inf")  # reward col: any real reward beats -inf
        self._init_flg = torch.ones(self._max_traj_len, dtype=torch.bool, device=self.device)  # True = ref data
        self._reached_frame: int = 0  # furthest frame with sustained good tracking

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

    def _resolve_fingertip_ids(self) -> torch.Tensor:
        ids = []
        for name in self.cfg.fingertip_body_names:
            found, _ = self.robot.find_bodies(name)
            if found:
                ids.append(found[0])
        return torch.tensor(ids, dtype=torch.long, device=self.device)

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

    def _compute_fingertip_normals(self) -> torch.Tensor:
        """Compute world-space unit normal vectors at each fingertip.

        The normal is defined as the approach direction of the fingertip — i.e.,
        the direction from the link origin toward the virtual tip, expressed in
        world frame. Used to project contact forces onto the contact normal.

        Returns:
            normals: (B, 5, 3) unit vectors in world frame.
        """
        B = self.num_envs
        if len(self._ft_body_ids) != 5:
            return torch.zeros(B, 5, 3, device=self.device)

        link_quat = self.robot.data.body_quat_w[:, self._ft_body_ids, :]  # (B, 5, 4)
        offsets = self._ft_offsets.unsqueeze(0).expand(B, -1, -1)         # (B, 5, 3)

        normals = quat_apply(
            link_quat.reshape(B * 5, 4),
            offsets.reshape(B * 5, 3),
        ).reshape(B, 5, 3)  # world-frame offset vectors (not yet unit)

        return normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def _setup_debug_vis(self) -> None:
        """Create VisualizationMarkers for reference fingertips and wrist."""
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

        self._debug_vis_n = n
        print(f"[grasp] Debug vis enabled for first {n} envs.")

    def _update_debug_vis(
        self,
        ref_ft_pos: torch.Tensor,
        ft_pos: torch.Tensor,
        ref_wrist_pos: torch.Tensor,
    ) -> None:
        """Update debug markers every observation step.

        Args:
            ref_ft_pos:    (B, 5, 3) reference fingertip world positions.
            ft_pos:        (B, 5, 3) actual virtual fingertip world positions.
            ref_wrist_pos: (B, 3)   reference wrist world position.
        """
        n = self._debug_vis_n

        # Reference fingertips: (n*5, 3)
        self._vis_ref_ft.visualize(translations=ref_ft_pos[:n].reshape(n * 5, 3))

        # Actual fingertips: (n*5, 3)
        self._vis_actual_ft.visualize(translations=ft_pos[:n].reshape(n * 5, 3))

        # Reference wrist position: n magenta spheres
        self._vis_ref_wrist.visualize(translations=ref_wrist_pos[:n])

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

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

        # EMA smoothing on joint actions (dims 0–27); mass dim (28) is not smoothed
        joint_actions = self.actions[:, :-1]  # (B, 28)
        alpha = self.cfg.action_smoothing
        self._smoothed_actions = alpha * self._smoothed_actions + (1.0 - alpha) * joint_actions

        # Cache mass action for application at next episode reset
        self._current_mass_action = self.actions[:, -1].clone()

    def _apply_action(self) -> None:
        N_f = self.cfg.num_hand_dofs
        N_a = self.cfg.num_arm_r_dofs
        # Use EMA-smoothed joint actions (28D); mass dim is excluded from joint control
        finger_act = self._smoothed_actions[:, :N_f]
        arm_act = self._smoothed_actions[:, N_f : N_f + N_a]
        lift_act = self._smoothed_actions[:, N_f + N_a : N_f + N_a + 1]

        default = self.robot.data.default_joint_pos
        self.robot.set_joint_position_target(
            default[:, self._finger_joint_ids] + finger_act * self.cfg.action_scale,
            joint_ids=self._finger_joint_ids,
        )
        self.robot.set_joint_position_target(
            default[:, self._arm_r_joint_ids] + arm_act * self.cfg.arm_action_scale,
            joint_ids=self._arm_r_joint_ids,
        )
        self.robot.set_joint_position_target(
            default[:, self._lift_joint_ids] + lift_act * self.cfg.lift_action_scale,
            joint_ids=self._lift_joint_ids,
        )

    def _get_observations(self) -> dict:
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        B = self.num_envs

        # Robot state: all controlled joints [fingers | arm_r | lift]
        joint_pos = torch.cat([
            self.robot.data.joint_pos[:, self._finger_joint_ids],
            self.robot.data.joint_pos[:, self._arm_r_joint_ids],
            self.robot.data.joint_pos[:, self._lift_joint_ids],
        ], dim=-1)  # (B, 28)
        joint_vel = torch.cat([
            self.robot.data.joint_vel[:, self._finger_joint_ids],
            self.robot.data.joint_vel[:, self._arm_r_joint_ids],
            self.robot.data.joint_vel[:, self._lift_joint_ids],
        ], dim=-1)  # (B, 28)

        # Fingertip positions (virtual tips) + velocities
        ft_pos = self._compute_fingertip_positions()                           # (B, 5, 3)
        if len(self._ft_body_ids) == 5:
            ft_vel = self.robot.data.body_lin_vel_w[:, self._ft_body_ids, :]  # (B, 5, 3)
        else:
            ft_vel = torch.zeros(B, 5, 3, device=self.device)

        # Object state
        obj_pos = self.object.data.root_pos_w    # (B, 3)
        obj_quat = self.object.data.root_quat_w  # (B, 4)
        obj_linvel = self.object.data.root_lin_vel_w
        obj_angvel = self.object.data.root_ang_vel_w

        # Reference (env-local coords → add env_origins for world frame)
        env_orig = self.scene.env_origins

        # Use next-frame reference for delta computation (look-ahead signal à la GR env)
        next_frame = (frame + 1).clamp(max=self._max_traj_len - 1)
        ref_ft_pos = self._ref_ft_pos[traj, next_frame] + env_orig.unsqueeze(1)   # (B, 5, 3)
        ref_obj_pos = self._ref_obj_pos[traj, next_frame] + env_orig               # (B, 3)
        ref_obj_quat = self._ref_obj_quat[traj, next_frame]                        # (B, 4)

        # Delta fingertip in object local frame
        delta_ft_world = ft_pos - ref_ft_pos                                  # (B, 5, 3)
        delta_ft_obj = quat_apply_inverse(
            obj_quat.unsqueeze(1).expand(-1, 5, -1).reshape(B * 5, 4),
            delta_ft_world.reshape(B * 5, 3),
        ).reshape(B, 5, 3)

        # Delta object position & rotation toward next reference frame
        delta_obj_pos = obj_pos - ref_obj_pos
        q_err = quat_mul(obj_quat, quat_conjugate(ref_obj_quat))
        delta_obj_rot = 2.0 * q_err[:, 1:]  # axis-angle approximation

        future_contact = self._future_contact[traj, frame]
        fingertip_forces = self._get_fingertip_forces()

        obs = torch.cat([
            joint_pos,                    # [N]
            joint_vel,                    # [N]
            ft_pos.reshape(B, 15),        # [15]
            ft_vel.reshape(B, 15),        # [15]
            obj_pos,                      # [3]
            obj_quat,                     # [4]
            obj_linvel,                   # [3]
            obj_angvel,                   # [3]
            delta_ft_obj.reshape(B, 15),  # [15]
            delta_obj_pos,                # [3]
            delta_obj_rot,                # [3]
            future_contact,               # [5]
            self._prev_action,            # [N]
            fingertip_forces,             # [5]
        ], dim=-1)

        self._prev_action = self.actions.clone()

        if self.cfg.debug_vis:
            ref_wrist_pos = self._ref_wrist_pos[traj, frame] + env_orig   # (B, 3)
            self._update_debug_vis(ref_ft_pos, ft_pos, ref_wrist_pos)

        return {"policy": obs}

    def _get_fingertip_forces(self) -> torch.Tensor:
        """Return per-fingertip normal contact force (N), projected onto fingertip approach direction.

        The contact force vector from each sensor is projected onto the fingertip's
        world-frame normal (approach direction). Only positive (compressive) components
        are kept, i.e., force pushing in the direction the finger is pointing.
        """
        B = self.num_envs
        forces = torch.zeros(B, 5, device=self.device)
        normals = self._compute_fingertip_normals()  # (B, 5, 3)

        for i, name in enumerate(self.cfg.fingertip_body_names):
            sensor = self._contact_sensors.get(name)
            if sensor is None:
                continue
            try:
                net_f = sensor.data.net_forces_w    # (B, 1, 3)
                force_vec = net_f[:, 0, :]           # (B, 3)
                normal = normals[:, i, :]            # (B, 3)
                # Scalar projection onto fingertip approach normal; clamp to ≥ 0
                forces[:, i] = (force_vec * normal).sum(dim=-1).clamp(min=0.0)
            except Exception:
                pass
        return forces

    def _get_rewards(self) -> torch.Tensor:
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        env_orig = self.scene.env_origins

        obj_pos = self.object.data.root_pos_w
        obj_quat = self.object.data.root_quat_w

        ref_obj_pos = self._ref_obj_pos[traj, frame] + env_orig
        ref_obj_quat = self._ref_obj_quat[traj, frame]

        # Object tracking
        obj_pos_err = torch.norm(obj_pos - ref_obj_pos, dim=-1)
        q_err = quat_mul(obj_quat, quat_conjugate(ref_obj_quat))
        obj_rot_err = 2.0 * torch.acos(q_err[:, 0].clamp(-1.0, 1.0))

        # Fingertip tracking with contact conditioning:
        # When a finger is predicted to be in contact (future_contact==1), target the object
        # center instead of the fixed reference position to encourage maintained contact force.
        ft_pos = self._compute_fingertip_positions()                       # (B, 5, 3)
        ref_ft = self._ref_ft_pos[traj, frame] + env_orig.unsqueeze(1)    # (B, 5, 3)
        contact_flag = self._future_contact[traj, frame]                   # (B, 5)
        ft_target = torch.where(
            contact_flag.unsqueeze(-1).bool(),
            obj_pos.unsqueeze(1).expand(-1, 5, -1),   # object center when in contact
            ref_ft,                                    # reference position otherwise
        )
        ft_err = torch.norm(ft_pos - ft_target, dim=-1).mean(dim=-1)      # (B,)

        force_rew = self._get_fingertip_forces().sum(dim=-1)

        # Wrist position tracking (MassDexMimic: keep wrist on reference trajectory)
        wrist_err = torch.zeros(self.num_envs, device=self.device)
        if self._wrist_body_id is not None:
            wrist_pos = self.robot.data.body_pos_w[:, self._wrist_body_id, :]  # (B, 3)
            ref_wrist_pos = self._ref_wrist_pos[traj, frame] + env_orig         # (B, 3)
            wrist_err = torch.norm(wrist_pos - ref_wrist_pos, dim=-1)           # (B,)

        # Regularization (joint actions only, excluding mass dim)
        joint_pos = self.robot.data.joint_pos[:, self._all_joint_ids]
        default_pos = self.robot.data.default_joint_pos[:, self._all_joint_ids]
        action_reg = (self.actions[:, :-1] ** 2).sum(dim=-1)   # exclude mass dim
        pose_reg = ((joint_pos - default_pos) ** 2).sum(dim=-1)

        reward = (
            self.cfg.rew_alive
            + self.cfg.rew_obj_pos * obj_pos_err
            + self.cfg.rew_obj_rot * obj_rot_err
            + self.cfg.rew_fingertip * ft_err
            + self.cfg.rew_fingertip_force * force_rew
            + self.cfg.rew_wrist * wrist_err
            + self.cfg.rew_action_reg * action_reg
            + self.cfg.rew_pose_reg * pose_reg
        )

        self._save_state_cache(reward, ft_err, obj_pos_err, obj_rot_err)

        # skrl SequentialTrainer only logs values that are torch.Tensor with numel()==1.
        # Python floats/ints are silently ignored. Use .mean() (0-dim tensor) not .mean().item().
        self.extras["log"] = {
            # Tracking errors
            "error/obj_pos_m":       obj_pos_err.mean(),
            "error/obj_rot_deg":     torch.rad2deg(obj_rot_err).mean(),
            "error/ft_mean_m":       ft_err.mean(),
            "error/wrist_m":         wrist_err.mean(),
            # Per-component rewards
            "rew/alive":             torch.tensor(self.cfg.rew_alive, device=self.device),
            "rew/obj_pos":           (self.cfg.rew_obj_pos * obj_pos_err).mean(),
            "rew/obj_rot":           (self.cfg.rew_obj_rot * obj_rot_err).mean(),
            "rew/fingertip":         (self.cfg.rew_fingertip * ft_err).mean(),
            "rew/fingertip_force":   (self.cfg.rew_fingertip_force * force_rew).mean(),
            "rew/wrist":             (self.cfg.rew_wrist * wrist_err).mean(),
            "rew/action_reg":        (self.cfg.rew_action_reg * action_reg).mean(),
            "rew/pose_reg":          (self.cfg.rew_pose_reg * pose_reg).mean(),
            "rew/total":             reward.mean(),
            # Curriculum state
            "curriculum/reached_frame":  torch.tensor(float(self._reached_frame), device=self.device),
            "curriculum/warmup_ratio":   self._is_warming_up.float().mean(),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        env_orig = self.scene.env_origins

        obj_pos = self.object.data.root_pos_w
        obj_quat = self.object.data.root_quat_w
        ref_obj_pos = self._ref_obj_pos[traj, frame] + env_orig
        ref_obj_quat = self._ref_obj_quat[traj, frame]

        # Object fell below table surface
        obj_fell = obj_pos[:, 2] < self.cfg.obj_fall_z

        # Object position tracking error too large (GR env: 0.15 m)
        obj_pos_err = torch.norm(obj_pos - ref_obj_pos, dim=-1)
        pos_err_large = obj_pos_err > self.cfg.max_obj_pos_err

        # Object rotation tracking error too large (GR env: 0.75 rad)
        q_err = quat_mul(obj_quat, quat_conjugate(ref_obj_quat))
        obj_rot_err = 2.0 * torch.asin(torch.clamp(torch.norm(q_err[:, 1:4], dim=-1), max=1.0))
        rot_err_large = obj_rot_err > self.cfg.max_obj_rot_err

        # Mean fingertip tracking error too large (GR env: 0.15 m)
        ft_pos = self._compute_fingertip_positions()                     # (B, 5, 3)
        ref_ft = self._ref_ft_pos[traj, frame] + env_orig.unsqueeze(1)  # (B, 5, 3)
        ft_mean_err = torch.norm(ft_pos - ref_ft, dim=-1).mean(dim=-1)  # (B,)
        ft_err_large = ft_mean_err > self.cfg.max_ft_mean_err

        # Wrist position tracking error too large (GR env: 0.15 m)
        # ── WARMUP ──: keep raw wrist_err tensor (needed for warm-up exit check below).
        # To restore original: change back to wrist_err_large only (no wrist_err variable).
        wrist_err = torch.zeros(self.num_envs, device=self.device)
        wrist_err_large = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._wrist_body_id is not None:
            wrist_pos = self.robot.data.body_pos_w[:, self._wrist_body_id, :]
            ref_wrist_pos = self._ref_wrist_pos[traj, frame] + env_orig
            wrist_err = torch.norm(wrist_pos - ref_wrist_pos, dim=-1)
            wrist_err_large = wrist_err > self.cfg.max_wrist_pos_err

        early_terminate = obj_fell | pos_err_large | rot_err_large | ft_err_large | wrist_err_large
        # GR env: termination flag allows disabling early termination for warm-up.
        # When termination=False, only timeout resets occur so the policy sees full episodes
        # and can learn even when far from the reference (avoids infinite-termination loop).
        if not self.cfg.termination:
            early_terminate = torch.zeros_like(early_terminate)

        # ── WARMUP ────────────────────────────────────────────────────────────
        # Exit warm-up for envs whose hand has reached the start-frame target.
        # Disable early termination for envs still in warm-up.
        # To restore original behavior: remove this entire block.
        if self.cfg.enable_warmup and self._is_warming_up.any():
            warmup_done = (
                (ft_mean_err < self.cfg.warmup_ft_threshold)
                & (wrist_err < self.cfg.warmup_wrist_threshold)
            )
            self._is_warming_up = self._is_warming_up & ~warmup_done
            early_terminate = early_terminate & ~self._is_warming_up
        # ── END WARMUP ────────────────────────────────────────────────────────

        terminated = early_terminate
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self.extras.setdefault("log", {})["success_rate"] = (
            (~terminated & ~time_out & (obj_pos_err < 0.03))
            .float().mean().item()
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
        if self.cfg.adaptive_sampling and hasattr(self, "reset_terminated"):
            is_terminated = self.reset_terminated[env_ids]
            if is_terminated.any():
                term_env_ids = env_ids_t[is_terminated]
                # Use last "good" frame (_enough_idx) rather than current frame as failure location
                failure_frames = self._enough_idx[term_env_ids].clamp(0, self._max_traj_len - 1)
                alpha = self.cfg.adaptive_alpha
                for f in failure_frames.unique():
                    self._failure_count[f] = alpha + (1.0 - alpha) * self._failure_count[f]

        # --- Trajectory assignment ---
        if self._n_trajs == 1:
            self._traj_idx[env_ids] = 0
        else:
            self._traj_idx[env_ids] = torch.randint(0, self._n_trajs, (n,), device=self.device)

        # --- Start frame sampling ---
        if self.cfg.adaptive_sampling and self._reached_frame > 0:
            valid_len = min(self._reached_frame + 1, self._max_traj_len)
            valid_counts = self._failure_count[:valid_len]
            ur = self.cfg.adaptive_uniform_ratio
            fail_probs = valid_counts / (valid_counts.sum() + 1e-8)
            probs = (1.0 - ur) * fail_probs + ur / valid_len          # (valid_len,)
            sampled = torch.multinomial(probs.unsqueeze(0).expand(n, -1), 1).squeeze(-1)
            start_frames = (sampled - self.cfg.adaptive_back_frames).clamp(min=0)
        else:
            start_frames = torch.zeros(n, dtype=torch.long, device=self.device)

        self._frame_idx[env_ids] = start_frames
        self._prev_action[env_ids] = 0.0
        self._done_env[env_ids] = False

        # Reset per-episode tracking quality
        self._enough_continued[env_ids] = True
        self._enough_idx[env_ids] = start_frames

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

        # --- Robot state reset (with state cache restore if available) ---
        traj = self._traj_idx[env_ids]
        env_orig = self.scene.env_origins[env_ids]

        cached = self._state_cache[start_frames]          # (n, 98) — zeros for unpopulated frames
        has_cache = ~self._init_flg[start_frames]         # (n,) bool
        cache_mask = has_cache.unsqueeze(-1)              # (n, 1) for broadcasting

        # Joint state: restore controlled joints from cache, others stay at default
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos_reset = default_joint_pos.clone()
        joint_vel_reset = torch.zeros_like(default_joint_pos)

        # cached[:, 14:42] = joint_pos (28 controlled), cached[:, 42:70] = joint_vel (28)
        cached_jp = cached[:, 14:42]
        cached_jv = cached[:, 42:70]
        cached_sa = cached[:, 70:98]

        joint_pos_reset[:, self._all_joint_ids] = torch.where(
            cache_mask, cached_jp, default_joint_pos[:, self._all_joint_ids]
        )
        joint_vel_reset[:, self._all_joint_ids] = torch.where(
            cache_mask, cached_jv, torch.zeros(n, len(self._all_joint_ids), device=self.device)
        )
        self._smoothed_actions[env_ids] = torch.where(cache_mask, cached_sa,
                                                       torch.zeros(n, self.cfg.action_space - 1, device=self.device))

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

        # --- Mass-as-an-action (GR env pattern) ---
        try:
            all_masses = self.object.root_physx_view.get_masses().clone()   # (num_envs, 1), CPU
            cpu_dev = all_masses.device
            t = (self._current_mass_action[env_ids].clamp(-1.0, 1.0) + 1.0) / 2.0
            new_masses = self.cfg.object_mass_min + t * (self.cfg.object_mass_max - self.cfg.object_mass_min)
            all_masses[torch.as_tensor(env_ids, dtype=torch.long, device=cpu_dev), 0] = new_masses.to(cpu_dev)
            all_indices = torch.arange(self.num_envs, dtype=torch.long, device=cpu_dev)
            self.object.root_physx_view.set_masses(all_masses, all_indices)
        except Exception:
            pass  # PhysX mass API unavailable in this Isaac Lab build; mass stays at object_mass

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

        # Build 98-dim state vector per env
        state = torch.cat([
            reward.unsqueeze(-1),   # (B, 1)  — index 0
            obj_pos_local,           # (B, 3)  — indices 1:4
            obj_quat,                # (B, 4)  — indices 4:8
            obj_linvel,              # (B, 3)  — indices 8:11
            obj_angvel,              # (B, 3)  — indices 11:14
            joint_pos,               # (B, 28) — indices 14:42
            joint_vel,               # (B, 28) — indices 42:70
            self._smoothed_actions,  # (B, 28) — indices 70:98
        ], dim=-1)  # (B, 98)

        # Update per-episode tracking quality (GR env: early_condition thresholds)
        good = (
            (ft_err < self.cfg.enough_ft_threshold)
            & (obj_pos_err < self.cfg.enough_obj_threshold)
            & (obj_rot_err < self.cfg.enough_obj_rot_threshold)
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
