"""Configuration for the OakInk dexterous grasping **MARL** environment.

Multi-Agent variant of `RobotisSh5GraspEnvCfg`:
  - Action split: arm (7D) and hand (20D) as separate agents (no mass action)
  - Object tracking signals → arm only; finger keypoint tracking → hand only
  - Hand observations expressed in palm-local frame (paper convention)
  - Centralized critic via `state_space = -1` (auto-flatten arm + hand obs)
  - Sequential forward at training time (handled in train_marl.py SequentialMAPPO patch)

Inherits the same scene/robot/dataset/reward primitives from the single-agent cfg,
only redefines spaces and adds arm/hand-split reward fields.
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
class RobotisSh5GraspMarlEnvCfg(DirectMARLEnvCfg):
    """Configuration for OakInk dexterous grasping under canonical MAPPO.

    **Canonical MAPPO** — single critic V(s_global) + shared team reward:
      - **hand agent (20D)**: 20 finger joints. Sees full single-agent grasping
        context (276D — 21 MANO kpts + palm state + full joint state (no lift) +
        object state + reference deltas + future_contact + prev_action + ft_forces).
      - **arm agent (7D)**: arm_r joints. Sees minimal obs (54D — own joints +
        current palm pose + next-frame target palm pose + prev_arm_action +
        current_hand_action slot). Hand decides first; its action is injected
        into arm's obs (sequential conditioning hand → arm).

    Both agents receive the SAME team reward (identical to single-agent reward
    formula: alive + 21-kpt tracking + obj_pos/rot + fingertip + force + regs).
    The centralized critic V(s_global) is shared across agents (patched in
    `train_marl.py` via `_share_value_critic`).

    Shared state (centralized critic input): EXPLICIT non-redundant 279D —
    hand_obs (276D) + delta_wrist_rot (3D). Computed by `_get_states()`. No
    duplication with arm_obs features (jp_arm ⊂ full_jp, wrist_quat_w shared, etc.).

    Coordination is implicit: shared reward + shared critic give both agents
    the same advantage signal; arm sees hand's current action via sequential
    injection, so arm can anticipate finger motion during grasp.

    Observation spaces (per-agent dict):
      - arm  ( 34D): jp_arm(7) + jv_arm(7) + wrist_pos_env(3) + wrist_quat_w(4)
                     + delta_wrist_pos(3) + delta_wrist_rot(3) + prev_arm_action(7)
      - hand (278D): single-agent obs minus mass — 21 MANO kpts world + palm state +
                     fingertip vel + full 28D joint state + object state + reference
                     deltas + future_contact + prev_action(27, no mass) + ft_forces
    Shared state (centralized critic): state_space=-1 → auto-flatten = 34+278 = 312D.

    Action spaces:
      - arm: 7D (arm_r joints), [-1, 1] → [lower, upper] via _scale()
      - hand: 20D (finger joints), [-1, 1] → [lower, upper] via _scale()
    """

    # ── Agent / space definitions ─────────────────────────────────────────────
    possible_agents: list = ["arm", "hand"]
    action_spaces: dict = {"arm": 7, "hand": 20}
    observation_spaces: dict = {"arm": 89, "hand": 283}
    state_space: int = 286   # explicit non-redundant shared state via _get_states()
    vel_obs_scale: float = 0.2  # TJ: 0.2 — applied to angular velocities and joint velocities

    # ── Viewer (identical to single-agent) ────────────────────────────────────
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.2, 0.15, 2.2),
        lookat=(-0.2, 0.5, 2.0),
        resolution=(1280, 720),
    )

    # ── Simulation ────────────────────────────────────────────────────────────
    decimation: int = 4
    episode_length_s: float = 5.0

    # ── DOF counts ────────────────────────────────────────────────────────────
    num_hand_dofs: int = 20
    num_arm_r_dofs: int = 7
    num_lift_dofs: int = 1   # not in action, held at fixed_lift_target
    fixed_lift_target: float = 0.0

    # ── Physics ───────────────────────────────────────────────────────────────
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

    # ── Scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # ── Robot ─────────────────────────────────────────────────────────────────
    robot_cfg: ArticulationCfg = FFW_SH5_DEX_CFG

    # ── Joint / body names ────────────────────────────────────────────────────
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

    # ── Canonical reference XY for trajectory alignment ──────────────────────
    canonical_ref_pos_env: tuple | None = None

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset: str = "oakink"
    oakink_data_dir: str = _OAKINK_DATA_DIR
    hocap_data_dir: str = _HOCAP_DATA_DIR
    object_id: str = "C11001"
    object_mass_json: str = ""
    trajectory_task: str = "C11001-0001-0007"
    trajectory_data_id: int = 0

    # ── Table ─────────────────────────────────────────────────────────────────
    table_pos_env: tuple = (0.3, 0.0, 0.0)
    table_size: tuple = (0.6, 0.6, 1.0)
    # Tabletop overhang: thin top slab extends in +y (toward robot) — see train cfg comments.
    tabletop_thickness: float = 0.2
    tabletop_overhang_y_pos: float = 0.25

    # ── Object physics ────────────────────────────────────────────────────────
    # `object_mass` is used as the static mass when mass-in-loop is disabled OR
    # when the JSON has no [lo, hi] entry for the object. When mass-in-loop is
    # enabled and JSON has a range, masses are sampled per-episode from a learned
    # distribution N(mu_mass, exp(log_std_mass)) affine-mapped to [lo, hi].
    object_mass: float = 0.2              # default static mass (kg) fallback
    object_mass_min: float = 0.04         # used when mass-in-loop enabled (overridden by JSON)
    object_mass_max: float = 0.20         # used when mass-in-loop enabled (overridden by JSON)
    object_static_friction: float = 1.0
    object_dynamic_friction: float = 1.0
    object_restitution: float = 0.1

    # ── Mass-in-the-loop optimization ────────────────────────────────────────
    # When enabled, mu_mass / log_std_mass are standalone learnable parameters
    # (not part of the action vector) optimized via REINFORCE on episode returns.
    # See `agents/mass_distribution.py` and `train_marl.py::_setup_mass_in_loop`.
    enable_mass_in_loop: bool = True
    mass_mu_init: float = -0.25            # μ_mass init (matches single-agent MassDexMimic)
    mass_log_std_init: float = -1.25       # log σ init → σ ≈ 0.286 (action space)
    mass_lr_scale: float = 33.333          # μ_mass LR multiplier over base LR
                                            # log_std_mass uses base LR (no boost)

    # ── Contact threshold for future_contact precomputation ──────────────────
    contact_dist_threshold: float = 0.05  # m

    # ── Action smoothing (EMA, combined 27D) ─────────────────────────────────
    # alpha = weight on NEW action (TJ convention). Higher = more responsive.
    # Increased from 0.3 → 0.4 to reduce smoothing lag (which can cause the
    # policy to overcorrect → wrist tremor).
    action_smoothing: float = 0.5   # finger EMA α — higher = more responsive
    # Arm-only EMA alpha (overrides action_smoothing for arm_r slice [20:27]).
    # Lower α = stronger smoothing → less wrist tremor; hand keeps action_smoothing.
    arm_action_smoothing: float = 0.2

    # ── Action scale (arm only) ───────────────────────────────────────────────
    # Per-step multiplier on arm raw action BEFORE clamp. Hand action is NOT scaled.
    # Effect: arm joint targets restricted to the middle [scale·100]% of joint range
    # (centered at midpoint); exploration σ in joint space scaled accordingly.
    # 1.0 = disabled (full range). Lower values restrict reachable arm pose.
    arm_action_scale: float = 1.0

    # ── Reward weights ───────────────────────────────────────────────────────
    # Canonical MAPPO: SINGLE team reward shared across all agents. Identical
    # formula and coefficients to single-agent train cfg.
    rew_alive: float = 1.5
    rew_kpts: float = -1.76               # mean over all 21 MANO kpts, Z-weighted, world
    rew_obj_pos: float = -4.26            # object position tracking (Z-weighted)
    rew_obj_rot: float = -1.0             # object rotation tracking (rad)
    rew_fingertip: float = -5.2           # contact-conditioned fingertip tracking
    rew_fingertip_force: float = 1.0
    rew_hand_action_reg: float = -0.004
    rew_arm_action_reg: float = -0.008   # 2× hand — penalize arm null-space wandering
    rew_hand_pose_reg: float = -0.001
    rew_arm_pose_reg: float = -0.002     # 2× hand — pull arm toward default pose

    # ── Termination (shared between agents) ──────────────────────────────────
    termination: bool = True
    obj_fall_z: float = 0.95
    max_obj_pos_err: float = 0.15
    max_obj_rot_err: float = 0.75
    max_wrist_pos_err: float = 0.15
    max_wrist_rot_err: float = 0.75
    max_ft_mean_err: float = 0.15   # synced with TJ — absorbs open-vs-curled finger mismatch at frame 0
    # Grace period: disable early termination for the first N frames of each episode. 0 = disabled.
    early_termination_grace_frames: int = 0

    # Diagnostic: log joints that saturate at effort_limit.
    log_effort_saturation: bool = False
    effort_saturation_log_interval: int = 500

    # ── Adaptive sampling (kept same as single-agent) ────────────────────────
    adaptive_sampling: bool = True
    adaptive_alpha: float = 0.001
    adaptive_uniform_ratio: float = 0.1
    adaptive_back_seconds: float = 1.2   # rewind window (s); frames = int(action_fps × this). Matches TJ.

    # ── State cache quality thresholds (kept same as single-agent) ───────────
    enough_ft_threshold: float = 0.10
    enough_obj_threshold: float = 0.085
    enough_obj_rot_threshold: float = 0.425
    enough_obj_threshold_late: float = 0.05
    enough_obj_rot_threshold_late: float = 0.25

    # ── Debug visualization ──────────────────────────────────────────────────
    debug_vis: bool = True
    debug_vis_num_envs: int = 16

    # ── Warmup (kept same; off by default — matches single-agent) ────────────
    enable_warmup: bool = False
    warmup_ft_threshold: float = 0.10
    warmup_wrist_threshold: float = 0.10
    warmup_wrist_rot_threshold: float = 0.75
