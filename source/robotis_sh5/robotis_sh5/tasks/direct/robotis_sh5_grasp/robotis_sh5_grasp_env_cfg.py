"""Configuration for the OakInk dexterous grasping environment."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

_DATA_DIR = Path(__file__).resolve().parents[4] / "data"
_ROBOT_USD = str(_DATA_DIR / "robots" / "FFW" / "FFW_SH5_simplified_dex_instanced.usd")
_OAKINK_DATA_DIR = str(_DATA_DIR / "processed" / "oakink")
_HOCAP_DATA_DIR = str(_DATA_DIR / "processed" / "hocap")


##
# Full-body FFW-SH5 robot config.
# fix_root_link=True: robot base is fixed to the ground.
# Policy controls only finger_r_joint1-20 (20 DOFs).
# Arm joints are initialized to a pre-grasp pose and held fixed by high stiffness.
##
FFW_SH5_DEX_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ROBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,   # position control; gravity compensation handled by stiffness
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=True,   # full-body robot: base is fixed
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.65, 0.65, 0.0),
        rot=(0.70711, 0.0, 0.0, -0.70711),
        joint_pos={
            # Swerve base
            "left_wheel_drive": 0.0, "left_wheel_steer": 0.0,
            "right_wheel_drive": 0.0, "right_wheel_steer": 0.0,
            "rear_wheel_drive": 0.0, "rear_wheel_steer": 0.0,
            # Lift
            "lift_joint": 0.0,
            # Left arm (unused, zero)
            **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
            # Right arm: pre-grasp pose matching pick_and_place env
            "arm_r_joint1": 0.0,
            "arm_r_joint2": -1.162,
            "arm_r_joint3": 0.291,
            "arm_r_joint4": -1.876,
            "arm_r_joint5": -0.609,
            "arm_r_joint6": 0.335,
            "arm_r_joint7": -0.368,
            # Fingers (both hands, zero = open)
            **{f"finger_l_joint{i + 1}": 0.0 for i in range(20)},
            **{f"finger_r_joint{i + 1}": 0.0 for i in range(20)},
            # Head
            "head_joint1": 0.0,
            "head_joint2": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_wheel_drive", "left_wheel_steer",
                "right_wheel_drive", "right_wheel_steer",
                "rear_wheel_drive", "rear_wheel_steer",
            ],
            velocity_limit_sim=30.0,
            effort_limit_sim=100000.0,
            stiffness=10000.0,
            damping=1000.0, # default: 100.0
        ),
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            velocity_limit_sim=0.2,
            effort_limit_sim=1000000.0,
            stiffness=10000.0,
            damping=1000.0, # default: 100.0
        ),
        "DY_80": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_joint[1-3]", "arm_r_joint[1-3]"],
            velocity_limit_sim=15.0,
            effort_limit_sim=1e6,  # 61.4,
            stiffness=600.0,
            damping=90.0  # default: 30.0
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_joint[4-6]", "arm_r_joint[4-6]"],
            velocity_limit_sim=15.0,
            effort_limit_sim=1e6,  # 31.7,
            stiffness=600.0,
            damping=60.0  # default: 20.0
        ),
        "DP_42": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_joint7", "arm_r_joint7"],
            velocity_limit_sim=6.0,
            effort_limit_sim=1e6,  # 5.1,
            stiffness=200.0,
            damping=25.0  # default: 3.0
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["finger_l_joint.*", "finger_r_joint.*"],
            velocity_limit_sim=2.2,
            effort_limit_sim=1e6,  # 2.0,
            stiffness=20.0,
            damping=1.0,  # default: 1.0
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint1", "head_joint2"],
            velocity_limit_sim=2.0,
            effort_limit_sim=30.0,
            stiffness=150.0,
            damping=3.0,
        ),
    },
)


@configclass
class RobotisSh5GraspEnvCfg(DirectRLEnvCfg):
    """Configuration for OakInk dexterous grasping with kinematic reference tracking.

    Robot: FFW-SH5 full-body (fix_root_link=True).
    Policy controls: right-hand fingers (20) + right arm (7) = 27 DOF total.
    Lift joint is excluded from the action and held at `fixed_lift_target` (0.0 = fully up)
    by the PD controller; lift remains in joint_pos/joint_vel observations for state awareness.
    Additional output: 1D normalized object mass parameter (MassDexMimic).

    Observation space (total=279, paper S4.1 layout):
        hand_kpts_pos      [21*3]   all 21 MANO keypoints in world frame (GR: hand_kpts_pos)
        wrist_quat         [4]      wrist global orientation (wxyz)
        wrist_linvel       [3]      wrist global linear velocity
        wrist_angvel       [3]      wrist global angular velocity
        fingertip_vel      [5*3]    fingertip linear velocities
        joint_pos          [28]     controlled joint angles (normalized)
        joint_vel          [28]     controlled joint velocities
        obj_pos            [3]      object position
        obj_quat           [4]      object orientation (wxyz)
        obj_linvel         [3]      object linear velocity
        obj_angvel         [3]      object angular velocity
        delta_kpts_world   [21*3]   next-frame delta for all 21 keypoints in world frame
        delta_ft_obj       [5*3]    next-frame fingertip error in object frame (contact-cond.)
        delta_obj_pos      [3]      next-frame obj position error
        delta_obj_rot      [3]      next-frame rotation error (axis-angle approximation)
        future_contact     [5]      predicted contact flag per fingertip
        prev_action        [28]     previous action (27 joints + 1 mass; lift NOT actioned)
        fingertip_forces   [5]      normal contact force per fingertip

    Action space (28): [fingers(20) | arm_r(7) | mass(1)] from default pose.
        Lift is NOT in the action — it is fixed at `fixed_lift_target` (0.0 = URDF upper limit).
        mass dim [-1,1] → [object_mass_min, object_mass_max] applied at episode start.
    """

    # Viewer: positioned in front of and to the right of the table, elevated, looking
    # at the table surface (z=1.0) where the robot hand approaches from behind.
    # Robot base is at env-local (0.65, 0.60); table top is at (0.3, 0.0, 1.0).
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.2, 0.15, 2.2),
        lookat=(-0.2, 0.5, 2.0),
        resolution=(1280, 720),
    )

    # Simulation
    decimation: int = 4  # control at 30 Hz (120 / 4)
    episode_length_s: float = 5.0

    # DOF counts
    num_hand_dofs: int = 20   # finger_r_joint1-20
    num_arm_r_dofs: int = 7   # arm_r_joint1-7
    num_lift_dofs: int = 1    # lift_joint (NOT in action — held at fixed_lift_target via PD ctrl)
    action_space: int = 28    # 20(fingers) + 7(arm) + 1(mass); lift excluded
    observation_space: int = 285  # 63+6+3+3+15+28+28+3+6+3+3+63+15+3+6+5+27+5  (6D rot, prev_action=27 joint-only; matches pretrain for ckpt transfer)
    state_space: int = 0
    vel_obs_scale: float = 0.2  # TJ: 0.2 — applied to angular velocities and joint velocities
    # Lift target (joint position in radians/meters depending on joint type).
    # 0.0 = URDF upper limit (fully up). Held by PD controller every step.
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

    # Controlled joint name patterns (action order: fingers → arm_r → lift)
    finger_joint_names: str = "finger_r_joint.*"
    arm_r_joint_names: str = "arm_r_joint.*"
    lift_joint_name: str = "lift_joint"

    # Actual fingertip body names in FFW_SH5_simplified_dex.usd
    # Corresponds to: thumb, index, middle, ring, little (in finger order 4,8,12,16,20)
    fingertip_body_names: list = [
        "finger_r_link4",
        "finger_r_link8",
        "finger_r_link12",
        "finger_r_link16",
        "finger_r_link20",
    ]

    # Right-hand wrist link name (end-effector base)
    wrist_body_name: str = "hx5_d20_right_base"

    # Canonical reference XY (env-local) for trajectory orientation alignment.
    # Trajectories are rotated in XY so the reference wrist approaches the object
    # from the same direction as (canonical_ref_pos_env → object).
    # None → robot body origin (cfg.robot_cfg.init_state.pos[:2]).
    # Set to the right-arm shoulder XY so alignment is relative to the arm, not the torso.
    canonical_ref_pos_env: tuple | None = None

    # Dataset
    dataset: str = "oakink"   # "oakink" | "hocap"
    oakink_data_dir: str = _OAKINK_DATA_DIR
    hocap_data_dir: str = _HOCAP_DATA_DIR
    object_id: str = "C11001" # "A01001"  # OakInk object to use (must have pre-converted USD)
    # Path to per-object mass JSON. If empty, the env resolves it at runtime to
    # data/processed/<dataset>/object_mass.json (so OakInk and HO-Cap each use
    # their own per-object mass table). Set explicitly to override.
    object_mass_json: str = ""

    # Trajectory selection: which specific trajectory to train on.
    # task: directory name under data/processed/oakink/mano/right/  (e.g. "A01001-0001-0000")
    # data_id: sub-index within that task directory (0, 1, 2, ...)
    # If task is empty, all trajectories matching object_id are loaded and assigned randomly.
    trajectory_task: str = "C11001-0001-0007" # "A01001-0001-0000"
    trajectory_data_id: int = 0

    # Table (static cuboid): bottom center at table_pos_env, top surface at z = table_size[2]
    table_pos_env: tuple = (0.3, 0.0, 0.0)   # env-local XYZ of table bottom center
    table_size: tuple = (0.6, 0.6, 1.0)       # X × Y × Z dimensions in meters
    # Tabletop overhang: thin top slab extends in +y (toward robot) so lift-and-place
    # trajectories that bring the object toward the robot side stay over the table surface.
    # Implementation: two cuboids — base body (`table_size` minus thickness) + top slab
    # (table_size.y + overhang_y_pos by thickness). Robot base sits under the overhang.
    tabletop_thickness: float = 0.2          # thickness of the overhanging top slab (m)
    tabletop_overhang_y_pos: float = 0.25    # extra +y extension of the top slab (m); 0 = original single-cuboid behavior

    # Object physics
    object_mass: float = 0.2              # default mass (kg) used if mass-as-action is disabled
    object_mass_min: float = 0.04         # minimum object mass for mass-as-action sampling (kg)
    object_mass_max: float = 0.20         # maximum object mass for mass-as-action sampling (kg)
    object_static_friction: float = 1.0
    object_dynamic_friction: float = 1.0
    object_restitution: float = 0.1

    # Contact threshold for future_contact precomputation
    contact_dist_threshold: float = 0.05  # m

    # Action smoothing (EMA, TJ/rl_games convention):
    #     smoothed = alpha * current + (1 - alpha) * prev
    # alpha is the weight on the new (raw) action; 1-alpha is the weight on the previous
    # smoothed value. So alpha=1.0 means no smoothing (use raw action as-is); lower alpha
    # means stronger smoothing (more lag, less trembling).
    # alpha=0.3 ≡ legacy (alpha-prev convention) value 0.7, same behavior.
    action_smoothing: float = 0.5   # finger EMA α — higher = more responsive
    # Arm-only EMA alpha (overrides action_smoothing for arm_r slice [20:27]).
    # Lower α = stronger smoothing → less wrist tremor; hand keeps action_smoothing.
    arm_action_smoothing: float = 0.2

    # Reward scales (GR env: 1.5*alive - clamp(4.26*pos + 1.0*rot + 5.2*ft + 1.76*kpts, 1.5) + force + reg)
    rew_alive: float = 1.5
    rew_kpts: float = -1.76           # GR env: 1.76 (mean Z-weighted L2 over all 21 MANO keypoints)
    rew_obj_pos: float = -4.26        # GR env: 4.26
    rew_obj_rot: float = -1.0         # GR env: 1.0
    rew_fingertip: float = -5.2       # GR env: 5.2
    rew_fingertip_force: float = 1.0  # GR env: 1.125 (slightly different normalization)
    # Action/pose regularization split by region (3× stronger on arm than hand).
    # Action layout: [fingers(20) | arm_r(7) | mass(1)]; pose excludes lift (PD-fixed).
    rew_hand_action_reg: float = -0.004   # GR baseline kept for fingers
    rew_arm_action_reg:  float = -0.008   # 2× hand — penalize arm null-space wandering
    rew_hand_pose_reg:   float = -0.001
    rew_arm_pose_reg:    float = -0.002   # 2× hand — pull arm toward default pose; lift excluded
    # Termination
    # termination=False disables early termination entirely (only timeout) — use for warm-up.
    # GR env uses this flag to avoid infinite termination loops early in training when the
    # robot cannot yet track the reference trajectory.
    termination: bool = True
    obj_fall_z: float = 0.95          # object fell if world z < this (table surface at 1.0 m)
    max_obj_pos_err: float = 0.15     # object position tracking error (m) — matches GR env
    max_obj_rot_err: float = 0.75     # object rotation tracking error (rad)
    max_wrist_pos_err: float = 0.15   # wrist (arm end-effector) position tracking error (m)
    max_wrist_rot_err: float = 0.75   # wrist rotation tracking error (rad) — matches GR env
    max_ft_mean_err: float = 0.15     # mean fingertip tracking error (m) — synced with TJ (0.15) to absorb open-vs-curled finger mismatch at frame 0
    # Grace period: disable early termination for the first N frames of each episode.
    # Helps absorb the IK-lift offset and open-vs-curled finger mismatch at t=0.
    # 0 = disabled (no grace; standard TJ behavior).
    early_termination_grace_frames: int = 0

    # Diagnostic: log joints that saturate at effort_limit. Prints to stdout periodically.
    log_effort_saturation: bool = False             # set True to enable
    effort_saturation_log_interval: int = 500       # steps between summary prints

    # Adaptive rollout sampling (GR/TJ env style)
    adaptive_sampling: bool = True
    adaptive_alpha: float = 0.001            # EMA coefficient — matches GR env (slow, stable update)
    adaptive_uniform_ratio: float = 0.1      # uniform mixing ratio — matches GR env
    adaptive_back_seconds: float = 1.2       # rewind window in seconds; frames = int(action_fps × this). Matches TJ `int(fps × 1.2)`.
    # Note: episode chunk length (TJ's ``num_frame_chunk``) is auto-derived at runtime as
    # ``round(episode_length_s * action_fps)`` — see env __init__. With default
    # ``episode_length_s = 5.0`` and ``action_fps = 30`` this gives 150 frames (TJ exact).
    # The chunk length controls (a) the upper clamp on adaptive start_frame and (b) the
    # actual episode time_out length:
    #   - traj_len <  num_frame_chunk      : episode = traj_len (no chunking)
    #   - traj_len >= num_frame_chunk      : adaptive sampling moves start across
    #                                         [0, traj_len - num_frame_chunk]; each episode
    #                                         is exactly num_frame_chunk frames.
    #   - adaptive_sampling = False (rollout): no chunking; full trajectory per episode.

    # State cache quality thresholds. Three phases matching GR/TJ env:
    #   start (frame <= ~20)      : loosest — bootstraps cache early in training
    #   early (not is_reached_end): early_condition values below
    #   late (is_reached_end)     : tighter, encourages refinement once full traj is reached
    # `enough` requires (ft_err < ft_threshold) AND any phase condition.
    enough_ft_threshold: float = 0.10           # max mean fingertip tracking error (m)
    enough_obj_threshold: float = 0.085         # early-phase obj pos err (m)
    enough_obj_rot_threshold: float = 0.425     # early-phase obj rot err (rad)
    enough_obj_threshold_late: float = 0.05     # late-phase obj pos err (m) — matches GR/TJ
    enough_obj_rot_threshold_late: float = 0.25 # late-phase obj rot err (rad) — matches GR/TJ

    # Debug visualization (requires GUI — do not use with --headless)
    debug_vis: bool = True
    debug_vis_num_envs: int = 2048            # show markers for first N environments only

    # ── WARMUP ────────────────────────────────────────────────────────────────
    # Warm-up mechanism: freeze the target at start_frame and disable early
    # termination until the hand reaches the start-frame reference position.
    # Prevents infinite-termination loops early in training when the robot
    # starts far from the reference (no cached state yet).
    #
    # To disable entirely: set enable_warmup=False.
    # To restore original behavior (no warm-up logic): set enable_warmup=False
    # and remove the [WARMUP] blocks in robotis_sh5_grasp_env.py.
    # ── END WARMUP ────────────────────────────────────────────────────────────
    enable_warmup: bool = False
    warmup_ft_threshold: float = 0.10          # exit warm-up when ft mean err < this (m)
    warmup_wrist_threshold: float = 0.10       # exit warm-up when wrist pos err < this (m)
    warmup_wrist_rot_threshold: float = 0.75   # exit warm-up when wrist rot err < this (rad)
