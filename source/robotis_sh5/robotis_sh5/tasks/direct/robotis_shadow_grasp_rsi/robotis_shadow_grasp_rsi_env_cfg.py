"""Configuration for the Shadow-Hand-on-Robotis-arm dexterous grasping environment.

Replaces FFW-SH5's native 20-DOF hand with a Shadow Dexterous Hand (24 joints,
18 actuated; the 4 distal "J0" joints are coupled to "J1" via fixed tendons in
the USD). Joint/body naming follows the standard Shadow Hand convention
(`robot0_*`), mounted on the existing `arm_r_link7` of the FFW-SH5 arm.
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

_DATA_DIR = Path(__file__).resolve().parents[4] / "data"
_ROBOT_USD = str(_DATA_DIR / "robots" / "FFW" / "FFW_SH5_shadow_instanced.usd")
_OAKINK_DATA_DIR = str(_DATA_DIR / "processed" / "oakink")
_HOCAP_DATA_DIR = str(_DATA_DIR / "processed" / "hocap")


##
# Full-body FFW-SH5 + Shadow Hand robot config.
# fix_root_link=True: robot base is fixed to the ground.
# Policy controls only the Shadow Hand actuated joints (18) + arm_r (7) = 25 DOF
# (plus 1 mass action). The 4 unactuated distal joints (robot0_FFJ0/MFJ0/RFJ0)
# are coupled to J1 via tendons in the USD. Left arm/hand remain at zero pose.
##
FFW_SH5_DEX_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ROBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,   # position control; gravity compensation handled by stiffness
            linear_damping=2.0,
            angular_damping=4.0,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            # Use Isaac Sim / PhysX default collision offsets (commented out to revert
            # the custom 0.005 / 0.0 values). NOTE: this robot USD is instanced, so
            # spawn-time collision_props may be a no-op — the effective offsets live in
            # FFW_SH5_shadow_instanced.usd (baked by make_robot_usd_instanceable.py,
            # which uses PhysX defaults unless --contact_offset/--rest_offset are passed).
            # contact_offset=0.005,
            # rest_offset=0.0,
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
            # Right arm: pre-grasp pose. Frame-0 IK from `arm_joint_pos_shadow.npy[0]`
            # overrides these during _reset_idx; this spawn pose is used before the
            # first reset and as a fallback when no IK reference is available.
            "arm_r_joint1": 0.00,
            "arm_r_joint2": -1.13,
            "arm_r_joint3": 0.03,
            "arm_r_joint4": -2.1,
            "arm_r_joint5": -1.44,
            "arm_r_joint6": 0.43,
            "arm_r_joint7": -0.65,
            # Left fingers (unused, zero — kept from FFW-SH5 base)
            **{f"finger_l_joint{i + 1}": 0.0 for i in range(20)},
            # Right hand = Shadow Hand (zero = open, neutral pose)
            "robot0_FFJ3": 0.0, "robot0_FFJ2": 0.0, "robot0_FFJ1": 0.0, "robot0_FFJ0": 0.0,
            "robot0_MFJ3": 0.0, "robot0_MFJ2": 0.0, "robot0_MFJ1": 0.0, "robot0_MFJ0": 0.0,
            "robot0_RFJ3": 0.0, "robot0_RFJ2": 0.0, "robot0_RFJ1": 0.0, "robot0_RFJ0": 0.0,
            "robot0_LFJ4": 0.0, "robot0_LFJ3": 0.0, "robot0_LFJ2": 0.0, "robot0_LFJ1": 0.0, "robot0_LFJ0": 0.0,
            "robot0_THJ4": 0.0, "robot0_THJ3": 0.0, "robot0_THJ2": 0.0, "robot0_THJ1": 0.0, "robot0_THJ0": 0.0,
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
            effort_limit_sim=61.4,  # 61.4,
            stiffness=200.0,  # 600.0
            damping=40.0  # default: 30.0
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_joint[4-6]", "arm_r_joint[4-6]"],
            velocity_limit_sim=15.0,
            effort_limit_sim=31.7,  # 31.7,
            stiffness=160.0,  # 600.0
            damping=30.0  # default: 20.0
        ),
        "DP_42": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_joint7", "arm_r_joint7"],
            velocity_limit_sim=6.0,
            effort_limit_sim=5.1,  # 5.1,
            stiffness=30.0,  # 200.0
            damping=5.0  # default: 3.0
        ),
        # Left hand: FFW-SH5 native fingers (kept, unused — zero pose)
        "hand_l": ImplicitActuatorCfg(
            joint_names_expr=["finger_l_joint.*"],
            velocity_limit_sim=15.0,
            effort_limit_sim=3.09,
            stiffness=1.0,  # 500.0
            damping=0.2,  # 3.0
        ),
        # Right hand: Shadow Dexterous Hand (18 actuated DOF, matching TJ's actuated_joint_names).
        # FFJ0/MFJ0/RFJ0/LFJ0 are tendon-coupled to J1 (not independently actuated) and
        # therefore excluded from this regex; PhysX absorbs them into the tendon constraint
        # rather than exposing them as separate articulation DOFs.
        "shadow_fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "robot0_(FF|MF|RF|LF|TH)J[1-3]",   # J1/J2/J3 for all 5 fingers (15 joints)
                "robot0_LFJ4",                      # little finger metacarpal (1)
                "robot0_THJ4",                      # thumb base (1)
                "robot0_THJ0",                      # thumb distal — independently actuated (1)
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=3.09,
            stiffness={
                "robot0_(FF|MF|RF|LF|TH)J[1-3]": 1.0,
                "robot0_LFJ4": 1.0,
                "robot0_THJ4": 1.0,
                "robot0_THJ0": 1.0,
            },
            damping={
                "robot0_(FF|MF|RF|LF|TH)J[1-3]": 0.2,
                "robot0_LFJ4": 0.2,
                "robot0_THJ4": 0.2,
                "robot0_THJ0": 0.2,
            },
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
class RobotisShadowGraspRsiEnvCfg(DirectRLEnvCfg):
    """Configuration for dexterous grasping with Shadow Hand mounted on FFW-SH5 arm.

    Robot: FFW-SH5 full-body with Shadow Dexterous Hand replacing the right hand
    (fix_root_link=True).
    Policy controls: Shadow Hand actuated joints (18) + right arm (7) = 25 DOF total.
    The 4 distal "J0" joints (robot0_FFJ0/MFJ0/RFJ0/LFJ0) are coupled to "J1" via
    fixed tendons in the USD; the policy only commands the 18 actuated DOFs.
    Lift joint is excluded from the action and held at `fixed_lift_target` (0.0 = fully up).
    Additional output: 1D normalized object mass parameter (MassDexMimic).

    Observation space (total=291):
        hand_kpts_pos      [21*3]   21 MANO keypoints in world frame
        elbow_pos          [3]      right elbow position in world frame
        wrist_quat_6d      [6]      wrist global orientation (6D continuous rotation rep)
        wrist_linvel       [3]      wrist global linear velocity
        wrist_angvel       [3]      wrist global angular velocity
        fingertip_vel      [5*3]    fingertip linear velocities
        joint_pos          [28]     controlled joint angles (normalized)
        joint_vel          [28]     controlled joint velocities
        obj_pos            [3]      object position
        obj_quat_6d        [6]      object orientation (6D continuous rotation rep)
        obj_linvel         [3]      object linear velocity
        obj_angvel         [3]      object angular velocity
        delta_kpts_world   [21*3]   next-frame delta for 21 MANO keypoints
        delta_elbow_world  [3]      next-frame delta for right elbow
        delta_ft_obj       [5*3]    next-frame fingertip error in object frame (contact-cond.)
        delta_obj_pos      [3]      next-frame obj position error
        delta_obj_rot_6d   [6]      next-frame rotation error (6D continuous rotation rep)
        future_contact     [5]      predicted contact flag per fingertip
        prev_action        [27]     previous joint action (mass excluded; lift NOT actioned)
        fingertip_forces   [5]      normal contact force per fingertip

    Action space (28): [fingers(20) | arm_r(7) | mass(1)] from default pose.
        Lift is NOT in the action — it is fixed at `fixed_lift_target` (0.0 = URDF upper limit).
        mass dim [-1,1] → [object_mass_min, object_mass_max] applied at episode start.
    """

    # Viewer: front-left/elevated angled view of the table with the robot reaching
    # in from the back-right. Camera coords are ENV-LOCAL (origin_type="env")
    # so the camera stays anchored to env 0's contents regardless of num_envs
    # (default origin_type="world" caused the camera-to-table relationship to
    # shift when the GridCloner placed env 0 at different world positions for
    # different num_envs).
    # Scene layout (env-local):
    #   Table top center: (0.3, 0.0, 1.0); table extends ±0.3 m in X/Y, height 1.0
    #   Robot base:       (0.65, 0.65, 0.0); right arm reaches toward the table
    viewer: ViewerCfg = ViewerCfg(
        eye=(-0.4, -0.6, 1.8),       # front-left of the table, elevated
        lookat=(0.4, 0.2, 1.0),      # table top, slightly biased toward robot side
        resolution=(1280, 720),
        origin_type="env",
        env_index=0,
    )

    # Simulation
    decimation: int = 4  # control at 30 Hz (120 / 4)
    episode_length_s: float = 5.0

    # DOF counts (Shadow Hand: 18 actuated; J0 for FF/MF/RF/LF are tendon-absorbed).
    num_hand_dofs: int = 18         # actuated Shadow Hand joints (commanded by policy)
    num_arm_r_dofs: int = 7         # arm_r_joint1-7
    num_lift_dofs: int = 1          # lift_joint (NOT in action — held at fixed_lift_target via PD ctrl)
    action_space: int = 26          # 18(shadow fingers) + 7(arm) + 1(mass); lift excluded
    # Adjusted from the FFW-SH5 297 layout by:
    #   joint_pos/joint_vel: 28 (sh5) → 26 (shadow 18 actuated + 7 arm + 1 lift)
    #   prev_action:        27 (sh5) → 25 (shadow 18 + 7, mass excluded)
    # 21 MANO kpts + elbow + link7 keypoints retained as 23-keypoint reference.
    observation_space: int = 291
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
            # friction_correlation_distance=0.00625,
            # friction_offset_threshold=0.04,
            # bounce_threshold_velocity=0.01,
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

    # Controlled joint name patterns (action order: shadow fingers → arm_r → lift)
    # 18 actuated Shadow Hand joints (J0 of FF/MF/RF/LF excluded — tendon-coupled).
    finger_joint_names: str = "robot0_(?:(?:FF|MF|RF|LF|TH)J[1-3]|LFJ4|THJ[04])"
    arm_r_joint_names: str = "arm_r_joint.*"
    lift_joint_name: str = "lift_joint"

    # Shadow Hand fingertip bodies (distal links)
    # Order: thumb, index, middle, ring, little — matches TJ's gr_env_cfg.fingertip_body_names.
    fingertip_body_names: list = [
        "robot0_thdistal",
        "robot0_ffdistal",
        "robot0_mfdistal",
        "robot0_rfdistal",
        "robot0_lfdistal",
    ]

    # Right-hand wrist link name (end-effector base). For Shadow Hand the palm body
    # acts as the wrist/end-effector anchor — TJ's gr_env uses 'robot0_palm' as root_body.
    wrist_body_name: str = "robot0_palm"

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

    # ── OBJECT FRICTION CURRICULUM [ROLLBACK MARKER: friction-curriculum] ──────
    # Per-episode object friction is sampled from [friction_min, friction_max(t)],
    # where friction_max(t) decays LINEARLY from friction_max_init to friction_min
    # over `friction_decay_steps` control steps. Easy→hard: high friction early
    # (objects don't slip) → anneal to the realistic min (firm-pinch required).
    # After decay_steps the range collapses to [friction_min, friction_min] (fixed).
    # Static = dynamic = sampled value (set on the OBJECT material at reset via
    # root_physx_view.set_material_properties). Set friction_curriculum=False to
    # leave object friction at its baked value (no per-episode sampling).
    friction_curriculum: bool = True
    friction_min: float = 1.0            # range lower bound + final fixed value
    friction_max_init: float = 3.0       # initial range upper bound (decays to friction_min)
    friction_decay_steps: int = 24000    # control steps over which friction_max → friction_min
    # Object dynamic friction is set EQUAL to static friction (no separate ratio).
    # ── END OBJECT FRICTION CURRICULUM ────────────────────────────────────────

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
    # NOTE: ignored for the arm while arm_delta_action is True (see below).
    arm_action_smoothing: float = 0.2

    # ── ARM DELTA-ACTION [ROLLBACK MARKER: arm-delta] ─────────────────────────
    # When True, the ARM action output is interpreted as a per-control-step DELTA
    # (residual) instead of an absolute target. Pipeline (per control step):
    #     raw_arm ∈ [-1,1]  →  delta_cmd = raw_arm * arm_delta_scale            (rad)
    #     delta_ema = α·delta_cmd + (1-α)·delta_ema   (EMA on the VELOCITY command)
    #     arm_target = clamp(arm_target + delta_ema, joint_limits)   (integrate + windup clamp)
    # Integration acts as a low-pass filter on action noise → smoother wrist motion
    # (de-tremor). Fingers + mass are UNAFFECTED (still absolute EMA). While True,
    # arm_action_smoothing is unused. Set arm_delta_action = False to fully restore
    # the original absolute-action arm (no other change needed).
    arm_delta_action: bool = True
    arm_delta_scale: float = 0.25      # rad — max |delta| per control step at raw=±1
    arm_delta_smoothing: float = 1.0   # EMA α on the delta command (higher = more responsive)
    # ── END ARM DELTA-ACTION ──────────────────────────────────────────────────

    # ── HAND DELTA-ACTION [ROLLBACK MARKER: hand-delta] ───────────────────────
    # When True, the FINGER action output is interpreted as a per-control-step DELTA
    # (residual) instead of an absolute target — identical pipeline to the arm delta
    # above, on the finger slice. Motivation: make the action semantics CONSISTENT
    # (arm + fingers both velocity/residual), and low-pass-filter finger action noise.
    #     raw_hand ∈ [-1,1]  →  delta_cmd = raw_hand * hand_delta_scale            (rad)
    #     delta_ema = α·delta_cmd + (1-α)·delta_ema   (EMA on the VELOCITY command)
    #     hand_target = clamp(hand_target + delta_ema, joint_limits)   (integrate + windup clamp)
    # NOTE: when True, `rew_hand_action_reg` (sum of squared raw finger actions) becomes
    # a finger VELOCITY penalty rather than a flexion penalty — tune accordingly.
    # Also note delta=0 means HOLD (not "go to mid-range"), and zero-mean action noise
    # random-walks the target around the open reset pose, so contact-force exploration is
    # weaker than absolute mode — may need lower rew_hand_action_reg / higher σ.
    # While True, action_smoothing is unused for the finger slice. Set
    # hand_delta_action = False to fully restore the original absolute-action fingers.
    # Changing this flips finger action semantics → BOTH phases must be re-pretrained.
    hand_delta_action: bool = False
    hand_delta_scale: float = 0.40      # rad — max |delta| per control step at raw=±1
    hand_delta_smoothing: float = 1.0  # EMA α on the delta command (higher = more responsive)
    # ── END HAND DELTA-ACTION ─────────────────────────────────────────────────

    # Reward scales (GR env: 1.5*alive - clamp(4.26*pos + 1.0*rot + 5.2*ft + 1.76*kpts, 1.5) + force + reg)
    # `rew_kpts` averages over 21 MANO keypoints (includes wrist as kpt 0).
    # `rew_arm_pos`: single weight on the MEAN over the 3 arm endpoints — wrist (kpt 0),
    #   elbow (kpt 21), arm_r_link7 (kpt 22). link7 is the last revolute arm link before
    #   the FIXED wrist mount; tracking it adds a positional constraint that indirectly
    #   constrains wrist orientation (compensates for no rew_wrist_rot signal).
    rew_alive: float = 1.5
    rew_kpts: float = -1.76           # mean Z-weighted L2 over 21 MANO keypoints
    rew_arm_pos: float = -0.44          # mean Z-weighted L2 over (wrist, elbow, link7)
    rew_obj_pos: float = -4.26        # GR env: 4.26
    rew_obj_rot: float = -1.0         # GR env: 1.0
    rew_fingertip: float = -5.2       # GR env: 5.2
    rew_fingertip_force: float = 1.0  # GR env: 1.0 (slightly different normalization)
    # ── CONTACT-MAP REWARD [ROLLBACK MARKER: contact-map-reward] ──────────────
    # Two upgrades to the fingertip-force reward (both toggleable for A/B vs the
    # original fingertip-position-gated, pad-normal-projected force):
    #   use_contact_point_gate: gate the force by the distance between the ACTUAL
    #     contact point (contact_pos_w) and the prescribed contact vertex, instead
    #     of by fingertip-position proximity (ft_err < 0.03). contact_match_dist is
    #     the threshold (m); surface↔surface so smaller than the 3cm fingertip gate.
    #   use_grounded_normal: project the contact force onto the OBJECT-SURFACE NORMAL
    #     at the contact vertex (the outward surface normal, which already points the
    #     same way as the sensed reaction force, so it is used directly), instead of
    #     the fixed finger pad normal. Falls back to pad-inward where the surface
    #     normal is degenerate (no mesh).
    use_contact_point_gate: bool = True
    contact_match_dist: float = 0.02
    use_grounded_normal: bool = True
    # Grounded direction source (only used when use_grounded_normal=True):
    #   True  → (reference fingertip − nearest contact vertex): points surface→finger,
    #           ≈ outward normal, AUTO-correct sign (finger is outside the object), no
    #           dependence on mesh vertex normals; falls back to the mesh normal where
    #           the fingertip sits on the surface (degenerate).
    #   False → object mesh vertex normal at the contact vertex (sign-aligned at runtime).
    use_fingertip_to_vertex_dir: bool = True
    # ── END CONTACT-MAP REWARD ────────────────────────────────────────────────
    # Arm table-contact penalty (anti-cheating: arm_r_link3..link7 — including the wrist
    # camera at link7 — are otherwise free to rest on the table for grasp stability).
    # Soft per-N penalty only (auto-clamped); strong press is NOT a termination condition.
    # Force used = MAX contact-force magnitude across the 5 tracked arm links.
    # Per-step penalty is clamped at `rew_arm_contact × max_arm_contact_force` so a single
    # above-threshold contact step can't dominate the reward.
    rew_arm_contact: float = -0.05               # penalty weight per N of (max) arm-link contact force
    max_arm_contact_force: float = 10.0          # penalty saturation point (N): clamps the per-step penalty magnitude
    # Action/pose regularization (uniform weights across hand and arm).
    # Action layout: [fingers(20) | arm_r(7) | mass(1)]; pose excludes lift (PD-fixed).
    rew_hand_action_reg: float = -0.004
    rew_arm_action_reg:  float = -0.004
    rew_hand_pose_reg:   float = -0.001
    rew_arm_pose_reg:    float = -0.001
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
    # elbow position tracking error (m); loose threshold since elbow is soft guidance only
    max_elbow_pos_err: float = 0.2
    # Grace period: disable early termination for the first N frames of each episode.
    # Helps absorb the IK-lift offset and open-vs-curled finger mismatch at t=0.
    # 0 = disabled (no grace; standard TJ behavior).
    early_termination_grace_frames: int = 0

    # Diagnostic: log joints that saturate at effort_limit. Prints to stdout periodically.
    log_effort_saturation: bool = False             # set True to enable
    effort_saturation_log_interval: int = 500       # steps between summary prints

    # Adaptive rollout sampling (GR/TJ env style)
    adaptive_sampling: bool = True
    # When True: TJ failure-weighted start-frame sampling within [0, _reached_frame].
    # When False: uniform sampling within [0, _reached_frame] (skips multinomial entirely).
    failure_weighted_sampling: bool = True
    adaptive_alpha: float = 0.001            # EMA coefficient — matches GR env (slow, stable update)
    adaptive_uniform_ratio: float = 0.1      # uniform mixing ratio (used only if failure_weighted_sampling=True)
    adaptive_back_seconds: float = 1.2       # rewind window in seconds; frames = int(action_fps × this). Matches TJ `int(fps × 1.2)`.

    # ── PRETRAIN-CACHE WARM-START [ROLLBACK MARKER: pretrain-cache-warmstart] ──
    # Warm-start the train RSI state cache from the pretrain phase's state cache.
    # The pretrain cache (saved to disk as a sibling of pretrain.pt) visited ALL
    # frames under physics, so its per-frame robot states are physically valid init
    # poses (object-free). With it loaded, every reset:
    #   - samples a start frame ONLY among frames present in the train OR pretrain
    #     cache (so a reset never lands on a frame with no saved state),
    #   - restores the robot from the TRAIN cache if that frame has one, else from
    #     the PRETRAIN cache (object comes from the reference trajectory on a train
    #     cache-miss).
    # The train cache is read AND written from the first step; as it fills, the
    # pretrain fallback is used less and less (self-deprecating). While the pretrain
    # cache is loaded, cache writes require episode length >= 3 to filter out
    # immediately-failing (penetration / unstable) pretrain-initialized poses.
    # Set pretrain_cache_warmstart = False to fully restore the original (no-warm-
    # start) behavior. Loaded by scripts/skrl/train.py from the checkpoint sibling.
    pretrain_cache_warmstart: bool = True
    # Pure-uniform start-frame sampling for the first `uniform_sampling_steps` control
    # steps (only on the warm-start path). Failure-weighting becomes ~91% of the
    # sampling mass as soon as ANY per-frame failures are recorded (fail_probs is
    # normalized), which would prematurely concentrate before the train cache is
    # broadly populated. Forcing uniform first lets the cache fill across the whole
    # trajectory and failure_count stabilize before concentrating. 0 = disable.
    uniform_sampling_steps: int = 2000
    # ── END PRETRAIN-CACHE WARM-START ─────────────────────────────────────────

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
