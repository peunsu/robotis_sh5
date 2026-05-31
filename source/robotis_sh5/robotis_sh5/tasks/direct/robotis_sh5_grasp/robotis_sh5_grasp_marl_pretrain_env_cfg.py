"""Pretrain MARL environment configuration (no physics object).

Mirrors `RobotisSh5GraspMarlEnvCfg` but:
  - No physics object in scene (pure kinematic reference tracking)
  - Object-related obs signals substituted with reference values (zeros for vel/delta)
  - **Observation dimensions identical to train cfg** so pretrain → train ckpts
    transfer without network resizing
  - Reward terms involving the object removed (rew_obj_pos/rot/force = 0)
  - Termination thresholds removed for object errors (kept for wrist + ft)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .robotis_sh5_grasp_env_cfg import (
    FFW_SH5_DEX_CFG,
    _OAKINK_DATA_DIR,
    _HOCAP_DATA_DIR,
)


@configclass
class RobotisSh5GraspMarlPretrainEnvCfg(DirectMARLEnvCfg):
    """Pretrain configuration (no physics object) — same per-agent obs dims as train."""

    # ── Agent / space definitions (same shapes as train for ckpt transfer) ───
    possible_agents: list = ["arm", "hand"]
    action_spaces: dict = {"arm": 7, "hand": 20}
    observation_spaces: dict = {"arm": 89, "hand": 289}   # hand: 21 MANO kpts + elbow_pos (separated)
    state_space: int = 292   # explicit non-redundant shared state via _get_states() — hand_obs + delta_wrist_rot
    vel_obs_scale: float = 0.2  # TJ: 0.2 — applied to angular velocities and joint velocities

    # ── Viewer ───────────────────────────────────────────────────────────────
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.2, 0.15, 2.2),
        lookat=(-0.2, 0.5, 2.0),
        resolution=(1280, 720),
    )

    # ── Simulation ───────────────────────────────────────────────────────────
    decimation: int = 4
    episode_length_s: float = 5.0

    # ── DOF counts ───────────────────────────────────────────────────────────
    num_hand_dofs: int = 20
    num_arm_r_dofs: int = 7
    num_lift_dofs: int = 1
    fixed_lift_target: float = 0.0

    # ── Physics ──────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.80665),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            friction_correlation_distance=0.00625,
            friction_offset_threshold=0.04,
            bounce_threshold_velocity=0.01,
            gpu_max_rigid_patch_count=4096 * 4096,
        ),
    )

    # ── Scene ────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # ── Robot ────────────────────────────────────────────────────────────────
    robot_cfg: ArticulationCfg = FFW_SH5_DEX_CFG

    # ── Joint / body names ───────────────────────────────────────────────────
    finger_joint_names: str = "finger_r_joint.*"
    arm_r_joint_names: str = "arm_r_joint.*"
    lift_joint_name: str = "lift_joint"
    fingertip_body_names: list = [
        "finger_r_link4",
        "finger_r_link8",
        "finger_r_link12",
        "finger_r_link16",
        "finger_r_link20",
    ]
    wrist_body_name: str = "hx5_d20_right_base"

    canonical_ref_pos_env: tuple | None = None

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset: str = "oakink"
    oakink_data_dir: str = _OAKINK_DATA_DIR
    hocap_data_dir: str = _HOCAP_DATA_DIR
    object_id: str = "C11001"
    trajectory_task: str = "C11001-0001-0007"
    trajectory_data_id: int = 0

    # ── Table (visual only — no object in pretrain) ──────────────────────────
    table_pos_env: tuple = (0.3, 0.0, 0.0)
    table_size: tuple = (0.6, 0.6, 1.0)
    # Tabletop overhang: thin top slab extends in +y (toward robot) — see train cfg comments.
    tabletop_thickness: float = 0.2
    tabletop_overhang_y_pos: float = 0.25

    # ── Contact threshold (used only for future_contact ref consistency) ─────
    contact_dist_threshold: float = 0.05

    # ── Action smoothing ─────────────────────────────────────────────────────
    action_smoothing: float = 0.5   # finger EMA α — higher = more responsive
    # Arm-only EMA alpha (overrides action_smoothing for arm_r slice [20:27]).
    arm_action_smoothing: float = 0.2

    # ── Action scale (arm only — wrist jitter mitigation, see train cfg) ─────
    arm_action_scale: float = 1.0   # disabled (back to full range)

    # ── Reward weights ───────────────────────────────────────────────────────
    # Canonical MAPPO: single team reward (matches single-agent PRETRAIN cfg —
    # no obj_pos/rot, no force, stronger rew_fingertip). `rew_kpts` averages
    # over 21 MANO keypoints ONLY (elbow handled separately).
    rew_alive: float = 1.8
    rew_kpts: float = -1.76               # mean over 21 MANO kpts (elbow excluded), Z-weighted
    rew_wrist_pos: float = -1.5           # wrist emphasis
    rew_elbow_pos: float = -0.3           # elbow guidance (pretrain stronger)
    rew_obj_pos: float = 0.0              # disabled — no object in pretrain
    rew_obj_rot: float = 0.0
    rew_fingertip: float = -14.5          # boosted to maintain signal after wrist/elbow added
    rew_fingertip_force: float = 0.0      # no contact in pretrain
    rew_hand_action_reg: float = -0.004
    rew_arm_action_reg: float = -0.004    # uniform with hand
    rew_hand_pose_reg: float = -0.001
    rew_arm_pose_reg: float = -0.001      # uniform with hand

    # ── Termination ──────────────────────────────────────────────────────────
    termination: bool = True
    max_wrist_pos_err: float = 0.15
    max_wrist_rot_err: float = 0.75
    max_ft_mean_err: float = 0.1      # matches TJ pretrain: delta_fingertip_pos_value_mean > 0.1
    # elbow position tracking error (m); loose threshold since elbow is soft guidance only
    max_elbow_pos_err: float = 0.2
    # Grace period: disable early termination for the first N frames of each episode. 0 = disabled.
    early_termination_grace_frames: int = 0

    # Diagnostic: log joints that saturate at effort_limit.
    log_effort_saturation: bool = False
    effort_saturation_log_interval: int = 500

    # ── Debug visualization ──────────────────────────────────────────────────
    debug_vis: bool = True
    debug_vis_num_envs: int = 16

    # ── Warmup (matches single-agent pretrain) ─────────────────────────────
    enable_warmup: bool = False   # disabled — rely on state cache for adaptive_sampling
    warmup_ft_threshold: float = 0.15
    warmup_wrist_threshold: float = 0.15
    warmup_wrist_rot_threshold: float = 0.75

    # ── Adaptive sampling (pretrain — state cache enabled) ────────────────
    # Same semantics as single-agent pretrain.
    adaptive_sampling: bool = True
    adaptive_alpha: float = 0.001
    adaptive_uniform_ratio: float = 0.1
    adaptive_back_seconds: float = 1.2   # rewind window (s); frames = int(action_fps × this). Matches TJ.
