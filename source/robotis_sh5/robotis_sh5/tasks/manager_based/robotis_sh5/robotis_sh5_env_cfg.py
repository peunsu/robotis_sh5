# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


##
# Scene definition
##

current_script_path = os.path.dirname(os.path.abspath(__file__))


@configclass
class RobotisSh5SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{current_script_path}/../../../../data/robots/FFW/FFW_SH5.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
        # --- 여기서 Damping / Stiffness 설정 ---
        actuators={
            "steer_actuator": ImplicitActuatorCfg(
                joint_names_expr=[".*_steer"],
                effort_limit_sim=100.0,
                velocity_limit_sim=15.0,
                stiffness=1e6,  # Stiffness (KP)
                damping=1e4,    # Damping (KD)
            ),
            "drive_actuator": ImplicitActuatorCfg(
                joint_names_expr=[".*_drive"],
                effort_limit_sim=400.0,
                velocity_limit_sim=20.0,
                stiffness=0.0,  # 속도 제어를 위해 0으로 설정
                damping=1e5,    # Damping (KD) - 속도 추종 성능 결정
            ),
        },
    )

    # Goal Marker (Visual only)
    goal_marker = AssetBaseCfg(
        prim_path="/World/Visuals/Goal",
        spawn=sim_utils.SphereCfg(
            radius=0.15, 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        ),
    )
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 조향(Steer) 조인트 제어 (3개)
    steer_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*_steer"], 
        scale=1.0
    )
    
    # 구동(Drive) 조인트 제어 (3개)
    drive_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", 
        joint_names=[".*_drive"], 
        scale=10.0
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 로봇의 현재 상태
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        
        # 목표 지점까지의 상대적 위치 (mdp 폴더에 구현 필요)
        rel_goal_pos = ObsTerm(func=mdp.get_rel_pos_to_goal)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    
    # 로봇 위치 리셋 (중앙 부근 랜덤)
    reset_robot_position = EventTerm(
        func=mdp.reset_root_at_random_pos_2d,
        mode="reset",
        params={
            "pos_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            "yaw_range": (-math.pi, math.pi),
        },
    )

    # 목표 지점 위치 리셋 (랜덤한 곳으로 이동)
    reset_goal_position = EventTerm(
        func=mdp.reset_goal_position,
        mode="reset",
        params={
            "asset_name": "goal_marker", # 함수 인자 이름과 같아야 함
            "pos_range": {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ## --- Task Rewards (잘하면 주는 상) ---
    # 1. 목표와의 거리 기반 (Exponential이 Linear보다 학습 초기에 수렴이 빨라)
    reaching_goal = RewTerm(
        func=mdp.goal_distance_reward, 
        weight=2.0,
        params={"std": 0.5} # 거리에 따른 감쇄 정도 조절
    )
    
    # 2. 목표 도달 보너스 (충분히 가까워졌을 때 한 번에 크게!)
    target_reached_bonus = RewTerm(
        func=mdp.is_terminated, 
        weight=50.0
    )

    # --- Constraint & Penalty (못하면 주는 벌) ---
    # 3. 넘어짐 방지 (Orientation Penalty)
    # 로봇의 Up-vector(z축)가 하늘을 보지 않으면 페널티
    tilt_penalty = RewTerm(
        func=mdp.base_orientation_l2,
        weight=-1.0,
        params={"target_quat": (1.0, 0.0, 0.0, 0.0)} # 똑바로 서 있는 상태
    )

    # 4. 급격한 제어 입력 페널티 (모터 보호 및 부드러운 주행)
    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.05
    )

    # 5. 조인트 한계 임계점 페널티
    # Steer 각도가 물리적 한계에 가까워지면 페널티를 줘서 무리한 조향 방지
    joint_limit_penalty = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 1. 시간 초과 (최대 에피소드 길이 도달)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 2. 로봇이 넘어졌을 때 (Roll 또는 Pitch가 일정 각도 이상일 때)
    # 예: 로봇의 z축과 세계 좌표 z축 사이의 각도가 45도 이상 벌어지면 종료
    robot_fell = DoneTerm(
        func=mdp.bad_orientation,
        params={"threshold": 0.7} # cos(45도) 정도의 값으로 튜닝 필요
    )

    # 3. 맵 밖으로 너무 멀리 나갔을 때 (Out of Bounds)
    # 중앙(0,0)에서 반경 10m 이상 벗어나면 리셋
    out_of_track = DoneTerm(
        func=mdp.root_pos_distance_from_origin,
        params={"threshold": 10.0}
    )

    # 4. 목표 도달 성공 (성공 리셋)
    goal_reached = DoneTerm(
        func=mdp.is_near_goal, 
        params={"threshold": 0.2}
    )


##
# Environment configuration
##


@configclass
class RobotisSh5EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RobotisSh5SceneCfg = RobotisSh5SceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10
        # Viewer settings
        self.viewer.eye = (10.0, 0.0, 8.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation