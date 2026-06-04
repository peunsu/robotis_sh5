"""Configuration for the Shadow-Hand-on-Robotis dexterous grasping **pretrain** env.

Pretrain phase: the object is entirely removed from the physics simulation.
The policy learns to track hand keypoints (fingertips + wrist) with all object-related
inputs provided directly from the kinematic reference trajectory. Object tracking rewards
are excluded.

Robot: FFW-SH5 arm + Shadow Dexterous Hand (24 hand joints, 18 actuated).
Policy controls: Shadow Hand actuated (22) + right arm (7) + lift (1) = 30 DOF
(action=29: 22 shadow fingers + 7 arm). No mass action (object removed).
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

from .robotis_shadow_grasp_env_cfg import FFW_SH5_DEX_CFG

_DATA_DIR = Path(__file__).resolve().parents[4] / "data"
_OAKINK_DATA_DIR = str(_DATA_DIR / "processed" / "oakink")
_HOCAP_DATA_DIR = str(_DATA_DIR / "processed" / "hocap")


@configclass
class RobotisShadowGraspPretrainEnvCfg(DirectRLEnvCfg):
    """Configuration for Shadow-Hand grasping pretrain (kinematic-only, no object).

    Robot: FFW-SH5 arm + Shadow Dexterous Hand (fix_root_link=True).
    Policy controls: Shadow Hand commanded (22) + right arm (7) = 29 DOF (no mass).
    Object is entirely removed from the physics simulation; all object-related inputs
    come directly from the kinematic reference. Object tracking rewards are excluded.

    Observation space (total=291, matches train env layout for checkpoint transfer):
        hand_kpts_pos      [21*3]   21 MANO keypoints in world frame
        elbow_pos          [3]      right elbow position in world frame
        wrist_quat_6d      [6]      wrist global orientation (6D continuous rotation rep)
        wrist_linvel       [3]      wrist global linear velocity
        wrist_angvel       [3]      wrist global angular velocity
        fingertip_vel      [5*3]    fingertip linear velocities
        joint_pos          [28]     controlled joint angles (normalized)
        joint_vel          [28]     controlled joint velocities
        ref_obj_pos        [3]      reference object position (world frame, from kinematic ref)
        ref_obj_quat_6d    [6]      reference object orientation (6D rot, from kinematic ref)
        obj_linvel         [3]      zeros (no physics object)
        obj_angvel         [3]      zeros (no physics object)
        delta_kpts_world   [21*3]   next-frame delta for 21 MANO keypoints
        delta_elbow_world  [3]      next-frame delta for right elbow
        delta_ft_obj       [5*3]    next-frame fingertip error in object frame (contact-cond.)
        delta_obj_pos      [3]      next-frame ref obj position delta (ref traj dynamics)
        delta_obj_rot_6d   [6]      next-frame ref obj rotation delta (6D rot)
        future_contact     [5]      predicted contact flag per fingertip
        prev_action        [27]     previous joint action (fingers + arm; lift NOT actioned)
        fingertip_forces   [5]      normal contact force per fingertip

    Action space (28): [fingers(20) | arm_r(7) | lift(1)] delta from default pose.
    """

    # Viewer: same as train env (front-left elevated, env-local). See train cfg comment.
    viewer: ViewerCfg = ViewerCfg(
        eye=(-0.4, -0.6, 1.8),
        lookat=(0.4, 0.2, 1.0),
        resolution=(1280, 720),
        origin_type="env",
        env_index=0,
    )

    # Simulation
    decimation: int = 4
    episode_length_s: float = 5.0

    # DOF counts (Shadow Hand: 18 actuated; J0 for FF/MF/RF/LF tendon-absorbed)
    num_hand_dofs: int = 18
    num_arm_r_dofs: int = 7
    num_lift_dofs: int = 1            # lift_joint (NOT in action — held at fixed_lift_target)
    action_space: int = 25           # shadow fingers(18) + arm_r(7); lift excluded, no mass
    # joint_pos/joint_vel: 26 (18+7+1), prev_action: 25 (18+7). Matches train env for ckpt transfer.
    observation_space: int = 291
    state_space: int = 0
    vel_obs_scale: float = 0.2  # TJ: 0.2 — applied to angular velocities and joint velocities
    # Lift target (joint position). 0.0 = URDF upper limit (fully up).
    fixed_lift_target: float = 0.0

    # Physics
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

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # Robot
    robot_cfg: ArticulationCfg = FFW_SH5_DEX_CFG

    # 18 actuated Shadow Hand joints (J0 of FF/MF/RF/LF excluded — tendon-coupled).
    finger_joint_names: str = "robot0_(?:(?:FF|MF|RF|LF|TH)J[1-3]|LFJ4|THJ[04])"
    arm_r_joint_names: str = "arm_r_joint.*"
    lift_joint_name: str = "lift_joint"

    # Shadow Hand fingertip bodies (thumb, index, middle, ring, little)
    fingertip_body_names: list = [
        "robot0_thdistal",
        "robot0_ffdistal",
        "robot0_mfdistal",
        "robot0_rfdistal",
        "robot0_lfdistal",
    ]

    # Wrist body — Shadow Hand palm acts as the wrist/end-effector root
    wrist_body_name: str = "robot0_palm"

    # Canonical reference XY (env-local) for trajectory orientation alignment.
    # Trajectories are rotated in XY so the reference wrist approaches the object
    # from the same direction as (canonical_ref_pos_env → object).
    # None → robot body origin (cfg.robot_cfg.init_state.pos[:2]).
    # Set to the right-arm shoulder XY so alignment is relative to the arm, not the torso.
    canonical_ref_pos_env: tuple | None = None  # set to shoulder XY after running env

    # Dataset
    dataset: str = "oakink"   # "oakink" | "hocap"
    oakink_data_dir: str = _OAKINK_DATA_DIR
    hocap_data_dir: str = _HOCAP_DATA_DIR
    object_id: str = "C11001"
    trajectory_task: str = "C11001-0001-0007" # "A01001-0001-0000"
    trajectory_data_id: int = 0

    # Table
    table_pos_env: tuple = (0.3, 0.0, 0.0)
    table_size: tuple = (0.6, 0.6, 1.0)
    # Tabletop overhang: thin top slab extends in +y (toward robot) — see train cfg comments.
    tabletop_thickness: float = 0.2
    tabletop_overhang_y_pos: float = 0.25

    # Object physics disabled
    object_mass: float = 0.2
    object_static_friction: float = 1.0
    object_dynamic_friction: float = 1.0
    object_restitution: float = 0.1

    # Contact threshold for future_contact precomputation
    contact_dist_threshold: float = 0.05

    # Action: [-1, 1] → full joint range via scale(). EMA smoothing in normalized space.
    # TJ/rl_games convention: smoothed = alpha * current + (1 - alpha) * prev
    # alpha = weight on the new (raw) action. alpha=1.0 → no smoothing; lower alpha = smoother.
    # alpha=0.3 ≡ legacy (alpha-prev convention) value 0.7, same behavior.
    action_smoothing: float = 0.5   # finger EMA α — higher = more responsive
    # Arm-only EMA alpha (overrides action_smoothing for arm_r slice [20:27]).
    # Lower α = stronger smoothing → less wrist tremor; hand keeps action_smoothing.
    arm_action_smoothing: float = 0.2

    # Reward scales (GR pretrain: 1.5*alive - clamp(1.76*kpts + 12.5*ft, 1.5) + reg)
    # `rew_kpts` averages over 21 MANO keypoints (includes wrist as kpt 0).
    # `rew_arm_pos` supervises the 3 arm endpoints (wrist + elbow + arm_r_link7) under
    # a single weight. link7 indirectly constrains wrist orientation (compensates for
    # the missing rew_wrist_rot signal).
    rew_alive: float = 1.6
    rew_kpts: float = -1.76           # mean Z-weighted L2 over 21 MANO keypoints
    rew_arm_pos: float = -0.44          # mean Z-weighted L2 over (wrist, elbow, link7)
    rew_fingertip: float = -12.5      # GR pretrain: 12.5
    rew_fingertip_force: float = 0.0
    # Arm table-contact penalty (anti-cheating: arm_r_link3..link7).
    # Hybrid: soft per-N penalty (auto-clamped at rew_arm_contact × max_arm_contact_force)
    # + hard termination on strong press. Force used = MAX across the 5 tracked arm links.
    rew_arm_contact: float = -0.1               # penalty weight per N of (max) arm-link contact force
    max_arm_contact_force: float = 5.0          # termination threshold (N): episode ends if max link force exceeds this
    # Action/pose regularization (uniform weights across hand and arm).
    # Action layout (pretrain, no mass): [fingers(20) | arm_r(7)]; pose excludes lift.
    rew_hand_action_reg: float = -0.004
    rew_arm_action_reg:  float = -0.004
    rew_hand_pose_reg:   float = -0.001
    rew_arm_pose_reg:    float = -0.001

    # Termination (fingertip + wrist only — no object-based termination)
    termination: bool = True
    max_wrist_pos_err: float = 0.15
    max_wrist_rot_err: float = 0.75   # GR pretrain: delta_hand_rot_value > 0.75
    max_ft_mean_err: float = 0.1      # matches TJ pretrain: delta_fingertip_pos_value_mean > 0.1
    # elbow position tracking error (m); loose threshold since elbow is soft guidance only
    max_elbow_pos_err: float = 0.2
    # Grace period: disable early termination for the first N frames of each episode. 0 = disabled.
    early_termination_grace_frames: int = 0

    # Diagnostic: log joints that saturate at effort_limit.
    log_effort_saturation: bool = False
    effort_saturation_log_interval: int = 500

    # Debug visualization
    debug_vis: bool = True
    debug_vis_num_envs: int = 4096

    # ── WARMUP ────────────────────────────────────────────────────────────────
    # Warm-up mechanism: freeze the target at frame 0 and disable early
    # termination until the hand reaches the reference start position.
    # Pretrain always needs warm-up (no state cache; robot starts from default pose).
    #
    # To disable entirely: set enable_warmup=False.
    # To restore original behavior (no warm-up logic): set enable_warmup=False
    # and remove the [WARMUP] blocks in robotis_sh5_grasp_pretrain_env.py.
    # ── END WARMUP ────────────────────────────────────────────────────────────
    enable_warmup: bool = False                # disabled — rely on state cache for adaptive_sampling
    warmup_ft_threshold: float = 0.15          # exit warm-up when ft mean err < this (m)
    warmup_wrist_threshold: float = 0.15       # exit warm-up when wrist pos err < this (m)
    warmup_wrist_rot_threshold: float = 0.75   # exit warm-up when wrist rot err < this (rad)

    # ── Adaptive sampling (pretrain — state cache enabled) ─────────────────
    # _reset_idx samples start_frame from failure-weighted EMA. With
    # enable_warmup=False, this only works once the state cache has populated
    # frames (cache miss → robot at frame-0 IK + start_frame > 0 → fast fail).
    # Initial episodes always start at frame 0 (until _reached_frame > 0).
    adaptive_sampling: bool = True
    # Pretrain: uniform sampling within [0, _reached_frame] (no failure weighting).
    # Curriculum behavior is preserved (start frame bounded by reached frontier),
    # but no failure EMA weighting — every reached frame equally likely.
    failure_weighted_sampling: bool = False
    adaptive_alpha: float = 0.001              # EMA coefficient (unused when failure_weighted_sampling=False)
    adaptive_uniform_ratio: float = 0.1        # unused when failure_weighted_sampling=False
    adaptive_back_seconds: float = 1.2         # rewind window (s); frames = int(action_fps × this). Matches TJ.
