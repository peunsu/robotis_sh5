"""Pretrain MARL environment (no physics object).

Inherits from :class:`RobotisSh5GraspMarlEnv` and reuses every helper
(``_load_reference_trajectories``, ``_compute_hand_kpts_pos``,
``_to_palm_local_pos``, ``_get_fingertip_forces`` (returns zeros here),
``_scale`` / ``_unscale``, etc.). Overrides only lifecycle methods that
touch ``self.object`` or the object reward terms:

  * ``__init__``      — skip object cfg / mass-json
  * ``_setup_scene``  — no object, no contact sensor
  * ``_get_observations`` — substitute object signals with reference (zeros for vel)
  * ``_get_rewards``  — drop object tracking, no force reward
  * ``_get_dones``    — only wrist + ft early-terminations
  * ``_reset_idx``    — no object write, no state cache restore
  * ``_save_state_cache`` — disabled (no cache in pretrain)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from .robotis_sh5_grasp_marl_env import RobotisSh5GraspMarlEnv
from .robotis_sh5_grasp_marl_pretrain_env_cfg import RobotisSh5GraspMarlPretrainEnvCfg


def quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternion to orthonormalized 6D rotation rep (TJ Zhou et al. 2019 style)."""
    q = torch.nn.functional.normalize(quat, dim=-1)
    w, x, y, z = q.unbind(-1)
    r0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1)
    r1 = torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1)
    a1 = torch.nn.functional.normalize(r0, dim=-1)
    a2 = r1 - (a1 * r1).sum(-1, keepdim=True) * a1
    a2 = torch.nn.functional.normalize(a2, dim=-1)
    return torch.cat([a1, a2], dim=-1)


class RobotisSh5GraspMarlPretrainEnv(RobotisSh5GraspMarlEnv):
    """Multi-agent pretrain env — kinematic-only reference tracking, no object."""

    cfg: RobotisSh5GraspMarlPretrainEnvCfg

    def __init__(self, cfg: RobotisSh5GraspMarlPretrainEnvCfg, render_mode: str | None = None, **kwargs):
        # Reuse parent's reference loading; skip object mass JSON + object cfg build.
        self._load_reference_trajectories(cfg)
        self._object_cfg = None

        action_fps = round(1.0 / (cfg.sim.dt * cfg.decimation))
        chunk_from_cfg = round(cfg.episode_length_s * action_fps)
        # Pretrain has no adaptive sampling — use the full trajectory chunk size.
        self._num_frame_chunk = min(chunk_from_cfg, self._max_traj_len)
        cfg.episode_length_s = self._num_frame_chunk / action_fps

        # Skip RobotisSh5GraspMarlEnv.__init__'s object mass path (call DirectMARLEnv directly
        # via the grandparent). super(grandparent) explicit to bypass middle layer's logic.
        from isaaclab.envs import DirectMARLEnv
        DirectMARLEnv.__init__(self, cfg, render_mode, **kwargs)
        self._post_init_buffers()

        # Re-allocate _state_cache as 84D (pretrain has no object — see SA pretrain).
        # MARL train's _post_init_buffers allocated 97D; we shrink it here for pretrain.
        # Layout: [0]=reward, [1:29]=joint_pos(28), [29:57]=joint_vel(28), [57:84]=smoothed_act(27)
        from .robotis_sh5_grasp_pretrain_env import _STATE_DIM_PRETRAIN
        self._state_cache = torch.zeros(self._max_traj_len, _STATE_DIM_PRETRAIN, device=self.device)
        self._state_cache[:, 0] = -float("inf")
        self._init_flg = torch.ones(self._max_traj_len, dtype=torch.bool, device=self.device)
        # Override parent class' _init_save_done (set False so pretrain's 84D save fires).
        self._init_save_done: bool = False

    # ------------------------------------------------------------------
    # Scene — no object, no contact sensor
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
        self._contact_sensors: dict = {}  # no object → all fingertip forces are zero

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.cfg.debug_vis:
            self._setup_debug_vis()

    # ------------------------------------------------------------------
    # Observation — substitute object signals with reference
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Same shape as train env (arm:60, hand:276) for ckpt transfer.

        Object-related signals substituted with reference:
          - obj_pos = ref_obj_pos[current_frame], obj_quat = ref_obj_quat
          - obj_linvel/angvel = zeros
          - delta_obj_pos/rot = zeros (ref vs ref)
          - delta_ft_obj computed against reference object pose
          - fingertip_forces = zeros (no physics object)
        """
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        B = self.num_envs
        N_f = self.cfg.num_hand_dofs
        N_a = self.cfg.num_arm_r_dofs

        jp_no_lift_raw = torch.cat([
            self.robot.data.joint_pos[:, self._finger_joint_ids],
            self.robot.data.joint_pos[:, self._arm_r_joint_ids],
        ], dim=-1)  # (B, 27)
        full_jp_norm = self._unscale(jp_no_lift_raw)
        full_jv = torch.cat([
            self.robot.data.joint_vel[:, self._finger_joint_ids],
            self.robot.data.joint_vel[:, self._arm_r_joint_ids],
        ], dim=-1)  # (B, 27)
        jp_arm = full_jp_norm[:, N_f:N_f+N_a]
        jv_arm = full_jv[:, N_f:N_f+N_a]

        hand_kpts_pos = self._compute_hand_kpts_pos()
        ft_pos = hand_kpts_pos[:, self._kpt_ft_mano_indices_t, :]
        if len(self._ft_body_ids) == 5:
            ft_vel = self.robot.data.body_lin_vel_w[:, self._ft_body_ids, :]
        else:
            ft_vel = torch.zeros(B, 5, 3, device=self.device)

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

        env_orig = self.scene.env_origins

        # ── Object signals: reference, not physics ───────────────────────────
        obj_pos    = self._ref_obj_pos[traj, frame]  + env_orig
        obj_quat   = self._ref_obj_quat[traj, frame]
        obj_linvel = torch.zeros(B, 3, device=self.device)
        obj_angvel = torch.zeros(B, 3, device=self.device)

        # Reference look-ahead (next frame).
        next_frame = (frame + 1).clamp(max=self._max_traj_len - 1)
        ref_kpts_world_next = self._ref_mano_kpts[traj, next_frame] + env_orig.unsqueeze(1)
        ref_ft_pos_next     = self._ref_ft_pos[traj, next_frame]   + env_orig.unsqueeze(1)
        ref_wrist_quat      = self._ref_wrist_quat[traj, next_frame]

        # delta_ft_obj — contact-conditioned, object-local frame. Object pose from reference.
        ref_vertex_local  = self._ref_contact_vertex_local[traj, next_frame]
        contact_flag_next = self._future_contact[traj, next_frame]
        obj_quat_exp = obj_quat.unsqueeze(1).expand(-1, 5, -1).reshape(B * 5, 4)
        from isaaclab.utils.math import quat_apply_inverse
        ft_in_obj = quat_apply_inverse(
            obj_quat_exp, (ft_pos - obj_pos.unsqueeze(1)).reshape(B * 5, 3),
        ).reshape(B, 5, 3)
        ref_ft_in_obj = quat_apply_inverse(
            obj_quat_exp, (ref_ft_pos_next - obj_pos.unsqueeze(1)).reshape(B * 5, 3),
        ).reshape(B, 5, 3)
        target_in_obj = torch.where(contact_flag_next.unsqueeze(-1).bool(), ref_vertex_local, ref_ft_in_obj)
        delta_ft_obj = ft_in_obj - target_in_obj                                # (B, 5, 3)

        # Object deltas: current-frame reference vs next-frame reference (matches
        # single-agent pretrain). Even without a physics object, this signal gives
        # the policy information about how the reference object will move in 1 frame
        # — useful "trajectory dynamics" context. Sign convention matches SA pretrain:
        # delta = current - next  (NEGATIVE direction when ref is moving forward).
        ref_obj_pos_next_w = self._ref_obj_pos[traj, next_frame] + env_orig          # (B, 3)
        ref_obj_quat_next  = self._ref_obj_quat[traj, next_frame]                    # (B, 4)
        delta_obj_pos = obj_pos - ref_obj_pos_next_w                                  # (B, 3)
        q_err_obj = quat_mul(obj_quat, quat_conjugate(ref_obj_quat_next))            # (B, 4)
        delta_obj_rot_6d = quat_to_6d(q_err_obj)                                      # (B, 6)
        # Wrist delta (kpt[0]) and rotation delta — used by arm obs
        delta_wrist_pos = hand_kpts_pos[:, 0, :] - ref_kpts_world_next[:, 0, :]
        q_err_wrist = quat_mul(wrist_quat_w, quat_conjugate(ref_wrist_quat))
        # Canonicalize quaternion (force w >= 0) for shortest-path rep — see train env.
        q_err_wrist = torch.where(q_err_wrist[:, 0:1] < 0, -q_err_wrist, q_err_wrist)
        delta_wrist_rot = 2.0 * q_err_wrist[:, 1:]
        # All-21 kpts delta (world frame) — used by hand obs
        delta_kpts_world = hand_kpts_pos - ref_kpts_world_next

        future_contact   = self._future_contact[traj, frame]
        fingertip_forces = torch.zeros(B, 5, device=self.device)   # no contact in pretrain
        wrist_pos_env    = wrist_pos_w - env_orig

        prev_action_27 = torch.cat([self._prev_hand_action, self._prev_arm_action], dim=-1)  # (B, 27)
        current_hand_placeholder = torch.zeros(B, N_f, device=self.device)

        # Object context for arm obs (pretrain: object = reference; no physics velocity).
        obj_pos_env     = obj_pos - env_orig                           # (B, 3) — ref-derived
        delta_wrist_obj = wrist_pos_env - obj_pos_env                  # (B, 3)

        vs = self.cfg.vel_obs_scale

        # ── Arm obs (89D) — wrist-pose follower + object context + hand action ─
        arm_obs = torch.cat([
            jp_arm,                       # 7
            vs * jv_arm,                  # 7  (TJ scaled)
            wrist_pos_env,                # 3
            quat_to_6d(wrist_quat_w),     # 6  wrist rotation (6D)
            wrist_linvel,                 # 3
            vs * wrist_angvel,            # 3  (TJ scaled)
            delta_wrist_pos,              # 3
            delta_wrist_rot,              # 3
            self._prev_arm_action,        # 7
            obj_pos_env,                  # 3   reference object pos (env-relative)
            quat_to_6d(obj_quat),         # 6   reference object orientation (6D)
            obj_linvel,                   # 3   zeros (no physics)
            vs * obj_angvel,              # 3   zeros (no physics)
            delta_wrist_obj,              # 3
            delta_obj_pos,                # 3
            delta_obj_rot_6d,             # 6   (6D)
            current_hand_placeholder,     # 20  ← SequentialMAPPO slot [69:89]
        ], dim=-1)
        # = 89

        # ── Hand obs (283D) — single-agent style (no mass, no lift) ──────────
        hand_obs = torch.cat([
            hand_kpts_pos.reshape(B, 63),   # 63
            quat_to_6d(wrist_quat_w),       # 6   (6D)
            wrist_linvel,                   # 3
            vs * wrist_angvel,              # 3   (TJ scaled)
            ft_vel.reshape(B, 15),          # 15
            full_jp_norm,                   # 27
            vs * full_jv,                   # 27  (TJ scaled)
            obj_pos,                        # 3
            quat_to_6d(obj_quat),           # 6   (6D)
            obj_linvel,                     # 3
            vs * obj_angvel,                # 3
            delta_kpts_world.reshape(B, 63),# 63
            delta_ft_obj.reshape(B, 15),    # 15
            delta_obj_pos,                  # 3
            delta_obj_rot_6d,               # 6   (6D)
            future_contact,                 # 5
            prev_action_27,                 # 27
            fingertip_forces,               # 5
        ], dim=-1)
        # = 283

        # ── Shared state (286D) — non-redundant ──────────────────────────────
        self._shared_state = torch.cat([hand_obs, delta_wrist_rot], dim=-1)

        if self.cfg.debug_vis:
            ref_wrist_pos = self._ref_wrist_pos[traj, frame] + env_orig
            self._update_debug_vis(ref_ft_pos_next, ft_pos, ref_wrist_pos)

        if hasattr(self, "_joint_actions"):
            self._prev_arm_action = self._joint_actions[:, N_f:N_f+N_a].clone()
            self._prev_hand_action = self._joint_actions[:, :N_f].clone()

        return {"arm": arm_obs, "hand": hand_obs}

    def _get_states(self) -> torch.Tensor:
        """Non-redundant centralized critic input (279D), cached in _get_observations."""
        return self._shared_state

    # ------------------------------------------------------------------
    # Reward — drop object terms, no force, only wrist/finger/ft tracking
    # ------------------------------------------------------------------

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        frame = self._frame_idx.clamp(max=self._max_traj_len - 1)
        traj = self._traj_idx
        env_orig = self.scene.env_origins
        B = self.num_envs

        hand_kpts_pos = self._compute_hand_kpts_pos()
        ft_pos = hand_kpts_pos[:, self._kpt_ft_mano_indices_t, :]

        # No physics object → reference keypoints in world frame (no drift compensation).
        ref_kpts_world = self._ref_mano_kpts[traj, frame] + env_orig.unsqueeze(1)  # (B, 21, 3)
        delta_kpts = hand_kpts_pos - ref_kpts_world
        # Z-weighted kpts error (matches single-agent pretrain reward formulation)
        delta_kpts_w = delta_kpts.clone()
        delta_kpts_w[:, :, 2] *= 1.5
        kpts_err_w = torch.norm(delta_kpts_w, dim=-1).mean(dim=-1)
        self._last_kpts_err = torch.norm(delta_kpts, dim=-1).mean(dim=-1)
        self._last_kpts_err_raw = self._last_kpts_err.clone()

        # Arm reward signals: wrist (kpt 0) world-frame, Z-weighted
        delta_wrist_kpt = delta_kpts[:, 0, :].clone()
        delta_wrist_kpt[:, 2] *= 1.5
        wrist_pos_err_w = torch.norm(delta_wrist_kpt, dim=-1)
        wrist_err = torch.norm(delta_kpts[:, 0, :], dim=-1)

        # Wrist rotation
        wrist_rot_err = torch.zeros(B, device=self.device)
        if self._wrist_body_id is not None:
            wrist_quat_w_cur = self.robot.data.body_quat_w[:, self._wrist_body_id, :]
            ref_wrist_quat = self._ref_wrist_quat[traj, frame]
            q_err = quat_mul(wrist_quat_w_cur, quat_conjugate(ref_wrist_quat))
            wrist_rot_err = 2.0 * torch.asin(torch.clamp(torch.norm(q_err[:, 1:4], dim=-1), max=1.0))

        # Fingertip tracking — contact-conditioned target (matches single-agent pretrain).
        # `ref_ft` is the reference fingertip position. For frames where MANO predicts
        # the fingertip is contacting the object, the target is the reference object
        # vertex (ref_vertex_world) instead — even though we have no physics object,
        # the reference object pose + per-frame contact vertices are available from the
        # MANO trajectory data.
        #
        # Use precomputed contact mask directly (TJ-style); `_future_contact` already encodes
        # (obj velocity moving) AND (fingertip near object) from preprocessing.
        ref_ft = self._ref_ft_pos[traj, frame] + env_orig.unsqueeze(1)              # (B, 5, 3)
        ref_obj_pos_w = self._ref_obj_pos[traj, frame] + env_orig                   # (B, 3)
        ref_obj_quat = self._ref_obj_quat[traj, frame]                              # (B, 4)
        ref_vertex_local = self._ref_contact_vertex_local[traj, frame]              # (B, 5, 3) obj-local
        ref_obj_quat_exp = ref_obj_quat.unsqueeze(1).expand(-1, 5, -1).reshape(B * 5, 4)
        ref_vertex_world = quat_apply(
            ref_obj_quat_exp,
            ref_vertex_local.reshape(B * 5, 3),
        ).reshape(B, 5, 3) + ref_obj_pos_w.unsqueeze(1)
        contact_flag = self._future_contact[traj, frame]                            # (B, 5)
        contact_flag_gated = contact_flag                                            # (B, 5)

        # Two separate ft errors (matches SA pretrain L780-786):
        #   ft_err_raw  : unweighted, NO contact conditioning — used for TERMINATION
        #                 (a misbehaving hand must terminate regardless of contact state).
        #   ft_err      : contact-conditioned via gated flag — used for the REWARD tracking
        #                 penalty (lets contact-frames target the object vertex instead of
        #                 raw ref fingertip once the wrist has converged).
        ft_err_raw = torch.norm(ft_pos - ref_ft, dim=-1).mean(dim=-1)
        ft_target = torch.where(contact_flag_gated.unsqueeze(-1).bool(), ref_vertex_world, ref_ft)
        ft_err = torch.norm(ft_pos - ft_target, dim=-1).mean(dim=-1)
        self._last_ft_raw_err = ft_err_raw.clone()

        # Regularization
        N_f = self.cfg.num_hand_dofs
        N_a = self.cfg.num_arm_r_dofs
        hand_action_reg = (self._joint_actions[:, :N_f] ** 2).sum(dim=-1)
        arm_action_reg  = (self._joint_actions[:, N_f:N_f+N_a] ** 2).sum(dim=-1)
        jp = self.robot.data.joint_pos
        dp = self.robot.data.default_joint_pos
        hand_pose_reg = ((jp[:, self._finger_joint_ids] - dp[:, self._finger_joint_ids]) ** 2).sum(dim=-1)
        arm_pose_reg  = ((jp[:, self._arm_r_joint_ids]  - dp[:, self._arm_r_joint_ids])  ** 2).sum(dim=-1)

        # Termination: only wrist + ft (no object). Use ft_err_raw (unweighted, no
        # contact conditioning) so a misbehaving hand pose triggers termination
        # regardless of contact-target switching. Matches SA pretrain L802.
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
        # Use raw (unconditioned) ft error for warmup / termination buffers — matches
        # SA pretrain L811. The contact-conditioned `ft_err` lives in the reward only.
        self._last_ft_mean_err = ft_err_raw
        self._last_wrist_err = wrist_err

        self._last_wrist_rot_err = wrist_rot_err
        self._last_obj_pos_err = torch.zeros(B, device=self.device)
        self._last_obj_rot_err = torch.zeros(B, device=self.device)

        alive = (~early_terminate).float()

        # ── Single team reward (matches single-agent pretrain reward) ────────
        # No obj_pos/rot/force terms (no physics object); strong rew_fingertip
        # (no contact gating to dampen the signal).
        tracking_penalty = (
            self.cfg.rew_kpts * kpts_err_w
            + self.cfg.rew_fingertip * ft_err
        ).clamp(min=-self.cfg.rew_alive)
        team_reward = (
            self.cfg.rew_alive * alive
            + tracking_penalty
            + self.cfg.rew_hand_action_reg * hand_action_reg
            + self.cfg.rew_arm_action_reg  * arm_action_reg
            + self.cfg.rew_hand_pose_reg   * hand_pose_reg
            + self.cfg.rew_arm_pose_reg    * arm_pose_reg
        ).clamp(min=0.0)

        # ── State cache update + adaptive sampling tracking ─────────────────
        if self.cfg.adaptive_sampling:
            self._save_state_cache(team_reward, ft_err_raw, wrist_err, wrist_rot_err)
        self._log_effort_saturation()

        self.extras["log"] = {
            "Error / kpts_mean_m":       torch.norm(delta_kpts, dim=-1).mean(),
            "Error / wrist_pos_m":       wrist_err.mean(),
            "Error / wrist_rot_deg":     torch.rad2deg(wrist_rot_err).mean(),
            "Error / ft_mean_m":         ft_err.mean(),
            "Episode_Reward / alive":            (self.cfg.rew_alive * alive).mean(),
            "Episode_Reward / kpts":             (self.cfg.rew_kpts * kpts_err_w).mean(),
            "Episode_Reward / fingertip":        (self.cfg.rew_fingertip * ft_err).mean(),
            "Episode_Reward / hand_action_reg":  (self.cfg.rew_hand_action_reg * hand_action_reg).mean(),
            "Episode_Reward / arm_action_reg":   (self.cfg.rew_arm_action_reg  * arm_action_reg).mean(),
            "Episode_Reward / hand_pose_reg":    (self.cfg.rew_hand_pose_reg   * hand_pose_reg).mean(),
            "Episode_Reward / arm_pose_reg":     (self.cfg.rew_arm_pose_reg    * arm_pose_reg).mean(),
            "Episode_Reward / team_total":       team_reward.mean(),
            "Curriculum / reached_frame": torch.tensor(float(self._reached_frame), device=self.device),
            "Curriculum / warmup_ratio": self._is_warming_up.float().mean(),
        }

        # Both agents receive the same team reward (canonical MAPPO).
        return {"arm": team_reward, "hand": team_reward}

    def _save_state_cache(
        self,
        reward: torch.Tensor,
        ft_err: torch.Tensor,
        wrist_err: torch.Tensor,
        wrist_rot_err: torch.Tensor,
    ) -> None:
        """Pretrain state cache (no object, 84D). Mirrors SA pretrain — uses
        wrist+ft tracking as the "good" criterion since there's no object.
        Layout: [0]=reward, [1:29]=joint_pos, [29:57]=joint_vel, [57:84]=smoothed_action.
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

    # ------------------------------------------------------------------
    # Reset — no object write, no state cache restore (always start from default)
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        n = len(env_ids)
        # Skip RobotisSh5GraspMarlEnv._reset_idx (which writes object). Use DirectMARLEnv directly.
        from isaaclab.envs import DirectMARLEnv
        DirectMARLEnv._reset_idx(self, env_ids)

        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # ── Adaptive sampling: EMA failure count update from terminated envs ──
        # DirectMARLEnv lacks `reset_terminated`; use `terminated_dict["arm"]`
        # (broadcast — both agents share the team signal).
        if self.cfg.adaptive_sampling and hasattr(self, "terminated_dict") and self.terminated_dict:
            is_terminated = self.terminated_dict["arm"][env_ids]
            if is_terminated.any():
                term_env_ids = env_ids_t[is_terminated]
                failure_frames = self._enough_idx[term_env_ids].clamp(0, self._max_traj_len - 1)
                counts = torch.bincount(failure_frames, minlength=self._max_traj_len).float()
                alpha = self.cfg.adaptive_alpha
                self._failure_count = alpha * counts + (1.0 - alpha) * self._failure_count

        if self._n_trajs == 1:
            self._traj_idx[env_ids] = 0
        else:
            self._traj_idx[env_ids] = torch.randint(0, self._n_trajs, (n,), device=self.device)

        # ── Start frame sampling (with train-style upper_a + upper_b) ─────────
        if self.cfg.adaptive_sampling and self._reached_frame > 0:
            valid_len = min(self._reached_frame + 1, self._max_traj_len)
            valid_counts = self._failure_count[:valid_len]
            ur = self.cfg.adaptive_uniform_ratio
            fail_probs = valid_counts / (valid_counts.sum() + 1e-8)
            probs = (fail_probs + ur / valid_len) / (1.0 + ur)  # TJ: add uniform then renormalize
            sampled = torch.multinomial(probs.unsqueeze(0).expand(n, -1), 1).squeeze(-1)
            start_frames = (sampled - self._adaptive_back_frames).clamp(min=0)
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

        # Reset per-episode tracking quality (per-env).
        self._enough_continued[env_ids] = True
        self._enough_idx[env_ids] = start_frames

        traj = self._traj_idx[env_ids]
        env_orig = self.scene.env_origins[env_ids]

        # ── Robot state restore (state cache when available) ──────────────────
        cached = self._state_cache[start_frames]              # (n, 84)
        has_cache = ~self._init_flg[start_frames]             # (n,) bool
        cache_mask = has_cache.unsqueeze(-1)

        cached_jp = cached[:, 1:29]
        cached_jv = cached[:, 29:57]
        cached_sa = cached[:, 57:84]

        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos_reset = default_joint_pos.clone()
        joint_vel_reset = torch.zeros_like(default_joint_pos)

        joint_pos_reset[:, self._all_joint_ids] = torch.where(
            cache_mask, cached_jp, default_joint_pos[:, self._all_joint_ids]
        )
        joint_vel_reset[:, self._all_joint_ids] = torch.where(
            cache_mask, cached_jv, torch.zeros(n, len(self._all_joint_ids), device=self.device)
        )

        # Frame-0 IK for cache-miss envs only.
        if self._frame0_arm_joint_pos is not None:
            arm_ik = self._frame0_arm_joint_pos[traj]
            no_cache_arm = (~has_cache).unsqueeze(-1).expand(-1, len(self._arm_r_joint_ids))
            joint_pos_reset[:, self._arm_r_joint_ids] = torch.where(
                no_cache_arm, arm_ik, joint_pos_reset[:, self._arm_r_joint_ids]
            )

        joint_pos_reset[:, self._lift_joint_ids] = self.cfg.fixed_lift_target
        joint_vel_reset[:, self._lift_joint_ids] = 0.0

        # Smoothed_actions: cached if available, else default + IK normalized.
        default_ctrl = torch.cat([
            default_joint_pos[:, self._finger_joint_ids],
            joint_pos_reset[:, self._arm_r_joint_ids],
        ], dim=-1)
        default_normalized = self._unscale(default_ctrl)
        self._smoothed_actions[env_ids] = torch.where(cache_mask, cached_sa, default_normalized)

        # Warmup only when cache miss (cache hit envs are already in a good state).
        if self.cfg.enable_warmup:
            self._is_warming_up[env_ids] = ~has_cache
        else:
            self._is_warming_up[env_ids] = False

        self.robot.write_joint_state_to_sim(joint_pos_reset, joint_vel_reset, None, env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += env_orig
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # No object reset — no object.

        # --- TJ-style init save: force-write frame 0 cache once so subsequent resets at
        # frame 0 reuse the IK-lifted pose (84D layout, no object). ---
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
