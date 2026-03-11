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
            usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/../../../../data/robots/FFW/FFW_SH5.usd",
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
        prim_path="{ENV_REGEX_NS}/Goal", 
        spawn=sim_utils.SphereCfg(
            radius=0.15, 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)), # 중앙 고정
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
        func=mdp.reset_root_around_goal_2d,
        mode="reset",
        params={
            "min_dist": 1.0,
            "max_dist": 2.0,
            "yaw_range": (-math.pi, math.pi),
        },
    )

    # 목표 지점 위치 리셋 (랜덤한 곳으로 이동)
    # reset_goal_position = EventTerm(
    #     func=mdp.reset_goal_position,
    #     mode="reset",
    #     params={
    #         "asset_name": "goal_marker", # 함수 인자 이름과 같아야 함
    #         "pos_range": {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
    #     },
    # )


@configclass
class RewardsCfg:
    """경험적으로 최적화된 Swerve Reach Task 가중치 구성"""

    # --- [1. Primary Task: 목표 도달] ---
    # 가장 높은 비중을 두되, 학습 전반에 걸쳐 일관된 가이드를 제공
    reaching_goal = RewTerm(
        func=mdp.goal_distance_reward, 
        weight=2.5,  # 20.0 -> 2.5로 하향 (안정적 수렴 유도)
        params={"std": 1.0} 
    )
    
    # 목표 도달 성공 시 보너스 (종료 시 1회성)
    # 너무 크면 '도박'을 하고, 너무 작으면 목표를 무시함. 10.0 정도가 적당해.
    target_reached_bonus = RewTerm(
        func=mdp.is_near_goal, # 별도 성공 체크 함수 사용 권장
        weight=10.0 
    )

    # --- [2. Shaping: 주행 가이드] ---
    # Swerve의 장점을 살리기 위해 방향 정렬은 '힌트' 정도로만 제공
    heading_alignment = RewTerm(
        func=mdp.heading_alignment_reward,
        weight=0.2  # 5.0 -> 0.2로 대폭 하향 (옆으로 가는 기동 허용)
    )

    # 생존 보상: 로봇이 죽지 않고 탐험하도록 유도 (상수값)
    alive = RewTerm(func=mdp.is_alive, weight=0.5)

    # --- [3. Regularization: 안정성 및 에너지 효율] ---
    # 이 항목들은 보상 총합을 갉아먹지 않을 정도로 미세하게 설정 (0.01 ~ 1.0)
    
    # 넘어짐 페널티: 로봇이 서 있는 것의 가치를 목표만큼 중요하게 설정
    tilt_penalty = RewTerm(
        func=mdp.base_orientation_l2,
        weight=-1.0, # 밸런스 유지
        params={"target_quat": (1.0, 0.0, 0.0, 0.0)}
    )

    # 조인트 한계 페널티: '벽'에 부딪히는 느낌만 주도록 설정
    joint_limit_penalty = RewTerm(
        func=mdp.joint_limits_penalty_l2,
        weight=-1.0 
    )

    # 부드러운 제어: 하드웨어 진동 방지 (가장 작은 가중치)
    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.01 
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
        params={"threshold": 0.5} # cos(45도) 정도의 값으로 튜닝 필요
    )

    # 3. 맵 밖으로 너무 멀리 나갔을 때 (Out of Bounds)
    # 중앙(0,0)에서 반경 2m 이상 벗어나면 리셋
    out_of_track = DoneTerm(
        func=mdp.root_pos_distance_from_env_origin,
        params={"threshold": 2.0}
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