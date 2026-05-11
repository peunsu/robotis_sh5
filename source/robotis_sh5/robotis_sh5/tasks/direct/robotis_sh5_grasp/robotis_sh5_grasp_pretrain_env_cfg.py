"""Configuration for the OakInk dexterous grasping **pretrain** environment.

Pretrain phase: object is teleported to the reference trajectory every step.
The policy learns to track hand keypoints (fingertips + wrist) without needing
to manipulate a real physics object. This mirrors the gr_env_pretrain design.

Key differences from the main grasp env:
  - Action space: 28D (no mass dim)
  - Observation space: 152D (no obj linvel/angvel; wrist rot delta instead of obj rot delta)
  - No state cache, no adaptive sampling
  - Reward: fingertip + wrist tracking only (no object physics reward)
  - Termination: fingertip + wrist error only (no object-based termination)
  - Object has gravity disabled and is teleported each step
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

from .robotis_sh5_grasp_env_cfg import FFW_SH5_DEX_CFG

_DATA_DIR = Path(__file__).resolve().parents[4] / "data"
_OAKINK_DATA_DIR = str(_DATA_DIR / "processed" / "oakink")


@configclass
class RobotisSh5GraspPretrainEnvCfg(DirectRLEnvCfg):
    """Configuration for OakInk dexterous grasping pretrain.

    Robot: FFW-SH5 full-body (fix_root_link=True).
    Policy controls: right-hand fingers (20) + right arm (7) + lift (1) = 28 DOF.
    Object is teleported to the reference trajectory every physics step (no real physics).

    Observation space (total=152):
        joint_pos          [28]     controlled joint angles (fingers + arm + lift)
        joint_vel          [28]     controlled joint velocities
        fingertip_pos      [5*3]    fingertip positions in world
        fingertip_vel      [5*3]    fingertip linear velocities
        ref_obj_pos        [3]      reference object position (env-local)
        ref_obj_quat       [4]      reference object orientation (wxyz)
        delta_fingertip    [5*3]    next-frame fingertip error (look-ahead)
        delta_wrist_pos    [3]      current-frame wrist position error
        delta_wrist_rot    [3]      current-frame wrist rotation error (axis-angle)
        future_contact     [5]      predicted contact flag per fingertip
        prev_action        [28]     previous action
        fingertip_forces   [5]      normal contact force per fingertip

    Action space (28): [fingers(20) | arm_r(7) | lift(1)] delta from default pose.
    """

    # Simulation
    decimation: int = 4
    episode_length_s: float = 5.0

    # DOF counts
    num_hand_dofs: int = 20
    num_arm_r_dofs: int = 7
    num_lift_dofs: int = 1
    action_space: int = 28           # no mass dim in pretrain
    observation_space: int = 152     # 28+28+15+15+3+4+15+3+3+5+28+5
    state_space: int = 0

    # Physics
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
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

    # Controlled joint name patterns
    finger_joint_names: str = "finger_r_joint.*"
    arm_r_joint_names: str = "arm_r_joint.*"
    lift_joint_name: str = "lift_joint"

    # Fingertip body names
    fingertip_body_names: list = [
        "finger_r_link4",
        "finger_r_link8",
        "finger_r_link12",
        "finger_r_link16",
        "finger_r_link20",
    ]

    # Wrist link name
    wrist_body_name: str = "hx5_d20_right_base"

    # Dataset
    oakink_data_dir: str = _OAKINK_DATA_DIR
    object_id: str = "A01001"
    trajectory_task: str = "A01001-0001-0000"
    trajectory_data_id: int = 0

    # Table
    table_pos_env: tuple = (0.3, 0.0, 0.0)
    table_size: tuple = (0.8, 0.8, 1.0)

    # Object physics (pretrain: gravity disabled, object is teleported)
    object_mass: float = 0.2
    object_static_friction: float = 0.8
    object_dynamic_friction: float = 0.6
    object_restitution: float = 0.1

    # Contact threshold for future_contact precomputation
    contact_dist_threshold: float = 0.05

    # Action scales
    action_scale: float = 0.5
    arm_action_scale: float = 0.5
    lift_action_scale: float = 0.05

    # Action smoothing (EMA)
    action_smoothing: float = 0.5

    # Reward scales (no object physics rewards in pretrain)
    rew_alive: float = 0.5
    rew_fingertip: float = -5.0
    rew_fingertip_force: float = 0.0
    rew_wrist: float = -2.0
    rew_action_reg: float = -0.004

    # Termination (fingertip + wrist only — no object-based termination)
    termination: bool = True
    max_wrist_pos_err: float = 0.15
    max_ft_mean_err: float = 0.15

    # Debug visualization
    debug_vis: bool = True
    debug_vis_num_envs: int = 4

    # ── WARMUP ────────────────────────────────────────────────────────────────
    # Warm-up mechanism: freeze the target at frame 0 and disable early
    # termination until the hand reaches the reference start position.
    # Pretrain always needs warm-up (no state cache; robot starts from default pose).
    #
    # To disable entirely: set enable_warmup=False.
    # To restore original behavior (no warm-up logic): set enable_warmup=False
    # and remove the [WARMUP] blocks in robotis_sh5_grasp_pretrain_env.py.
    # ── END WARMUP ────────────────────────────────────────────────────────────
    enable_warmup: bool = True
    warmup_ft_threshold: float = 0.10    # exit warm-up when ft mean err < this (m)
    warmup_wrist_threshold: float = 0.10  # exit warm-up when wrist err < this (m)
