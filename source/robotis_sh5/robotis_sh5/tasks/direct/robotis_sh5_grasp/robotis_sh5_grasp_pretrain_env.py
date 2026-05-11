"""OakInk dexterous grasping **pretrain** environment — Isaac Lab Direct RL.

Pretrain phase: the object is teleported to the reference trajectory position
at every physics sub-step (decimation loop). The policy learns to track the
reference hand pose (fingertips + wrist) without manipulating a real physics
object. This mirrors the gr_env_pretrain design.

Once pretrain is complete, the checkpoint can be transferred to the main
RobotisSh5GraspEnv for full dexterous manipulation with real object physics.
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
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from .robotis_sh5_grasp_pretrain_env_cfg import RobotisSh5GraspPretrainEnvCfg

# Local-frame offsets from link origin to actual fingertip contact point.
_FINGERTIP_OFFSETS: dict[str, list[float]] = {
    "finger_r_link4":  [0.0,  0.03975, 0.0],
    "finger_r_link8":  [0.0,  0.0,     0.02425],
    "finger_r_link12": [0.0,  0.0,     0.02425],
    "finger_r_link16": [0.0,  0.0,     0.02425],
    "finger_r_link20": [0.0,  0.0,     0.02425],
}


class RobotisSh5GraspPretrainEnv(DirectRLEnv):
    """Pretrain environment: track reference hand pose with a frozen (teleported) object."""

    cfg: RobotisSh5GraspPretrainEnvCfg

    def __init__(self, cfg: RobotisSh5GraspPretrainEnvCfg, render_mode: str | None = None, **kwargs):
        self._load_reference_trajectories(cfg)
        self._object_cfg = self._build_object_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        self._post_init_buffers()

    # ------------------------------------------------------------------
    # Data loading  (identical to main grasp env)
    # ------------------------------------------------------------------

    def _load_reference_trajectories(self, cfg: RobotisSh5GraspPretrainEnvCfg) -> None:
        data_dir = Path(cfg.oakink_data_dir) / "mano" / "right"

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

        for path in traj_files:
            data = np.load(str(path))
            wp = data["qpos_wrist_right"][:, :3].astype(np.float32)
            wq = data["qpos_wrist_right"][:, 3:].astype(np.float32)
            fp = data["qpos_finger_right"][:, :, :3].astype(np.float32)
            op = data["qpos_obj_right"][:, :3].astype(np.float32)
            oq = data["qpos_obj_right"][:, 3:].astype(np.float32)

            N = wp.shape[0]
            next_fp = np.concatenate([fp[1:], fp[-1:]], axis=0)
            next_op = np.concatenate([op[1:], op[-1:]], axis=0)
            dist = np.linalg.norm(next_fp - next_op[:, None, :], axis=-1)
            fc = (dist < cfg.contact_dist_threshold).astype(np.float32)

            wrist_pos_list.append(wp)
            wrist_quat_list.append(wq)
            ft_pos_list.append(fp)
            obj_pos_list.append(op)
            obj_quat_list.append(oq)
            future_contact_list.append(fc)

        mesh_path = Path(cfg.oakink_data_dir) / "assets" / "objects" / cfg.object_id / "visual.obj"
        if mesh_path.exists():
            _mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
            mesh_z_min = float(_mesh.vertices[:, 2].min())
        else:
            mesh_z_min = 0.0
            print(f"[warn] Centered mesh not found at {mesh_path}; assuming mesh_z_min=0.")

        table_surface_z = float(cfg.table_size[2])
        target_centroid_z = table_surface_z - mesh_z_min
        table_target = np.array(
            [cfg.table_pos_env[0], cfg.table_pos_env[1], target_centroid_z],
            dtype=np.float32,
        )
        print(f"[pretrain] mesh_z_min={mesh_z_min:.4f}, target_centroid_z={target_centroid_z:.4f}")

        for i in range(len(obj_pos_list)):
            offset = table_target - obj_pos_list[i][0]
            obj_pos_list[i] = obj_pos_list[i] + offset
            wrist_pos_list[i] = wrist_pos_list[i] + offset
            ft_pos_list[i] = ft_pos_list[i] + offset

        robot_xy = np.array(
            [cfg.robot_cfg.init_state.pos[0], cfg.robot_cfg.init_state.pos[1]],
            dtype=np.float32,
        )
        obj_xy_ref = np.array([cfg.table_pos_env[0], cfg.table_pos_env[1]], dtype=np.float32)
        canonical_dir = robot_xy - obj_xy_ref
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

        print(f"[pretrain] Loaded {n_traj} trajectories for '{cfg.object_id}', max_len={max_len}")

    def _build_object_cfg(self, cfg: RobotisSh5GraspPretrainEnvCfg) -> RigidObjectCfg:
        usd_path = (
            Path(cfg.oakink_data_dir)
            / "assets" / "objects" / cfg.object_id / "visual.usd"
        )
        if not usd_path.exists():
            raise FileNotFoundError(f"Object USD not found: {usd_path}")
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
                    disable_gravity=True,   # pretrain: object is teleported, gravity irrelevant
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
        for attr in (
            "_ref_wrist_pos", "_ref_wrist_quat", "_ref_ft_pos",
            "_ref_obj_pos", "_ref_obj_quat", "_future_contact", "_traj_lengths",
        ):
            setattr(self, attr, getattr(self, attr).to(self.device))

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
            print(f"[warn] Wrist body '{self.cfg.wrist_body_name}' not found; wrist tracking disabled.")

        self._all_joint_ids = self._finger_joint_ids + self._arm_r_joint_ids + self._lift_joint_ids

        B = self.num_envs
        self._traj_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        self._frame_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        self._prev_action = torch.zeros(B, self.cfg.action_space, device=self.device)
        self._smoothed_actions = torch.zeros(B, self.cfg.action_space, device=self.device)

        # ── WARMUP ────────────────────────────────────────────────────────────
        # Per-env warm-up flag. Pretrain has no state cache, so the robot always
        # starts from the default pose and always needs warm-up to reach frame 0.
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

        # EMA smoothing (all 28 joint dims)
        alpha = self.cfg.action_smoothing
        self._smoothed_actions = alpha * self._smoothed_actions + (1.0 - alpha) * self.actions

    def _apply_action(self) -> None:
        N_f = self.cfg.num_hand_dofs
        N_a = self.cfg.num_arm_r_dofs

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

        # Teleport object to reference position every decimation sub-step.
        # Since gravity is disabled and the object is kinematically frozen,
        # this keeps the visual and contact geometry on the reference trajectory.
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        env_orig = self.scene.env_origins
        ref_pos = self._ref_obj_pos[self._traj_idx, frame] + env_orig
        ref_quat = self._ref_obj_quat[self._traj_idx, frame]
        self.object.write_root_pose_to_sim(torch.cat([ref_pos, ref_quat], dim=-1))
        self.object.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, device=self.device))

    def _get_observations(self) -> dict:
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        B = self.num_envs
        env_orig = self.scene.env_origins

        # Robot joint state
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

        # Fingertip state
        ft_pos = self._compute_fingertip_positions()  # (B, 5, 3)
        if len(self._ft_body_ids) == 5:
            ft_vel = self.robot.data.body_lin_vel_w[:, self._ft_body_ids, :]
        else:
            ft_vel = torch.zeros(B, 5, 3, device=self.device)

        # Current reference object (frozen = reference)
        ref_obj_pos_world = self._ref_obj_pos[traj, frame] + env_orig  # (B, 3)
        ref_obj_pos_local = self._ref_obj_pos[traj, frame]              # (B, 3) env-local
        ref_obj_quat = self._ref_obj_quat[traj, frame]                  # (B, 4)

        # Next-frame reference for look-ahead delta
        next_frame = (frame + 1).clamp(max=self._max_traj_len - 1)
        ref_ft_next = self._ref_ft_pos[traj, next_frame] + env_orig.unsqueeze(1)  # (B, 5, 3)
        delta_ft = ft_pos - ref_ft_next  # (B, 5, 3)

        # Wrist tracking error (current frame)
        delta_wrist_pos = torch.zeros(B, 3, device=self.device)
        delta_wrist_rot = torch.zeros(B, 3, device=self.device)
        if self._wrist_body_id is not None:
            wrist_pos = self.robot.data.body_pos_w[:, self._wrist_body_id, :]
            wrist_quat = self.robot.data.body_quat_w[:, self._wrist_body_id, :]
            ref_wrist_pos = self._ref_wrist_pos[traj, frame] + env_orig
            ref_wrist_quat = self._ref_wrist_quat[traj, frame]
            delta_wrist_pos = wrist_pos - ref_wrist_pos
            q_err = quat_mul(wrist_quat, quat_conjugate(ref_wrist_quat))
            delta_wrist_rot = 2.0 * q_err[:, 1:]  # axis-angle approximation

        future_contact = self._future_contact[traj, frame]  # (B, 5)
        fingertip_forces = self._get_fingertip_forces()      # (B, 5)

        obs = torch.cat([
            joint_pos,                    # [28]
            joint_vel,                    # [28]
            ft_pos.reshape(B, 15),        # [15]
            ft_vel.reshape(B, 15),        # [15]
            ref_obj_pos_local,            # [3]  env-local; informs policy where object is
            ref_obj_quat,                 # [4]
            delta_ft.reshape(B, 15),      # [15]
            delta_wrist_pos,              # [3]
            delta_wrist_rot,              # [3]
            future_contact,               # [5]
            self._prev_action,            # [28]
            fingertip_forces,             # [5]
        ], dim=-1)
        # Total: 28+28+15+15+3+4+15+3+3+5+28+5 = 152

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

        # Fingertip tracking with contact conditioning:
        # When predicted in contact, target object center instead of reference fingertip position.
        ft_pos = self._compute_fingertip_positions()
        ref_ft = self._ref_ft_pos[traj, frame] + env_orig.unsqueeze(1)
        contact_flag = self._future_contact[traj, frame]
        ref_obj_center = (self._ref_obj_pos[traj, frame] + env_orig).unsqueeze(1).expand(-1, 5, -1)
        ft_target = torch.where(contact_flag.unsqueeze(-1).bool(), ref_obj_center, ref_ft)
        ft_err = torch.norm(ft_pos - ft_target, dim=-1).mean(dim=-1)  # (B,)

        # Wrist position tracking
        wrist_err = torch.zeros(self.num_envs, device=self.device)
        if self._wrist_body_id is not None:
            wrist_pos = self.robot.data.body_pos_w[:, self._wrist_body_id, :]
            ref_wrist_pos = self._ref_wrist_pos[traj, frame] + env_orig
            wrist_err = torch.norm(wrist_pos - ref_wrist_pos, dim=-1)

        force_rew = self._get_fingertip_forces().sum(dim=-1)
        action_reg = (self.actions ** 2).sum(dim=-1)

        reward = (
            self.cfg.rew_alive
            + self.cfg.rew_fingertip * ft_err
            + self.cfg.rew_fingertip_force * force_rew
            + self.cfg.rew_wrist * wrist_err
            + self.cfg.rew_action_reg * action_reg
        )

        # skrl SequentialTrainer only logs values that are torch.Tensor with numel()==1.
        # Python floats/ints are silently ignored. Use .mean() (0-dim tensor) not .mean().item().
        self.extras["log"] = {
            # Tracking errors
            "error/ft_mean_m":       ft_err.mean(),
            "error/wrist_m":         wrist_err.mean(),
            # Per-component rewards
            "rew/alive":             torch.tensor(self.cfg.rew_alive, device=self.device),
            "rew/fingertip":         (self.cfg.rew_fingertip * ft_err).mean(),
            "rew/fingertip_force":   (self.cfg.rew_fingertip_force * force_rew).mean(),
            "rew/wrist":             (self.cfg.rew_wrist * wrist_err).mean(),
            "rew/action_reg":        (self.cfg.rew_action_reg * action_reg).mean(),
            "rew/total":             reward.mean(),
            # Curriculum state
            "curriculum/warmup_ratio":   self._is_warming_up.float().mean(),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        env_orig = self.scene.env_origins

        # Fingertip mean tracking error
        ft_pos = self._compute_fingertip_positions()
        ref_ft = self._ref_ft_pos[traj, frame] + env_orig.unsqueeze(1)
        ft_mean_err = torch.norm(ft_pos - ref_ft, dim=-1).mean(dim=-1)
        ft_err_large = ft_mean_err > self.cfg.max_ft_mean_err

        # Wrist position tracking error
        # ── WARMUP ──: keep raw wrist_err tensor (needed for warm-up exit check below).
        # To restore original: change back to wrist_err_large only (no wrist_err variable).
        wrist_err = torch.zeros(self.num_envs, device=self.device)
        wrist_err_large = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._wrist_body_id is not None:
            wrist_pos = self.robot.data.body_pos_w[:, self._wrist_body_id, :]
            ref_wrist_pos = self._ref_wrist_pos[traj, frame] + env_orig
            wrist_err = torch.norm(wrist_pos - ref_wrist_pos, dim=-1)
            wrist_err_large = wrist_err > self.cfg.max_wrist_pos_err

        early_terminate = ft_err_large | wrist_err_large
        if not self.cfg.termination:
            early_terminate = torch.zeros_like(early_terminate)

        # ── WARMUP ────────────────────────────────────────────────────────────
        # Exit warm-up for envs whose hand has reached the frame-0 target.
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

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return early_terminate, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        n = len(env_ids)
        super()._reset_idx(env_ids)

        # Trajectory assignment
        if self._n_trajs == 1:
            self._traj_idx[env_ids] = 0
        else:
            self._traj_idx[env_ids] = torch.randint(0, self._n_trajs, (n,), device=self.device)

        # Pretrain: always start from frame 0 (no adaptive sampling)
        self._frame_idx[env_ids] = 0
        self._prev_action[env_ids] = 0.0
        self._smoothed_actions[env_ids] = 0.0

        # ── WARMUP ────────────────────────────────────────────────────────────
        # Pretrain has no state cache, so the robot always starts from the default
        # pose and always needs warm-up to reach the frame-0 reference position.
        # To restore original behavior: remove this block.
        if self.cfg.enable_warmup:
            self._is_warming_up[env_ids] = True
        else:
            self._is_warming_up[env_ids] = False
        # ── END WARMUP ────────────────────────────────────────────────────────

        env_orig = self.scene.env_origins[env_ids]
        traj = self._traj_idx[env_ids]

        # Robot: reset to default pose
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        default_joint_vel = torch.zeros_like(default_joint_pos)
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += env_orig
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Object: reset to reference position at frame 0
        ref_obj_pos = self._ref_obj_pos[traj, 0] + env_orig
        ref_obj_quat = self._ref_obj_quat[traj, 0]
        self.object.write_root_pose_to_sim(
            torch.cat([ref_obj_pos, ref_obj_quat], dim=-1), env_ids
        )
        self.object.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids
        )
