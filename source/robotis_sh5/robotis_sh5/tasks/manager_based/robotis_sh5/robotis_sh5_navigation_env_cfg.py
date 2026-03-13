# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from . import mdp

##
# Scene definition
##


@configclass
class RobotisSh5NavigationSceneCfg(InteractiveSceneCfg):
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
    """Swerve Drive 베이스 제어 전용 액션 설정"""

    # 에이전트는 [vx, vy, w] 총 3차원의 액션을 출력함
    base_velocity = mdp.SwerveDriveActionCfg(
        asset_name="robot",
        joint_names=[".*_steer", ".*_drive"],
        scale=(1.0, 1.0, 2.0)  # vx, vy 스케일은 1.0, 회전(w) 스케일은 2.0
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 6개의 바퀴 조인트 상태 (Position)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*drive", ".*steer"])}
        )
        
        # 6개의 바퀴 조인트 상태 (Velocity)
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*drive", ".*steer"])}
        )
        
        # 로봇 베이스의 선속도 및 각속도
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        
        # 현재 타겟 인덱스와 목표 오차 벡터
        rel_goal_pos = ObsTerm(func=mdp.get_rel_pos_to_current_waypoint) 
        # target_index = ObsTerm(func=mdp.get_target_waypoint_index) # 현재 몇 번째 목표인지
        
        # 헤딩 오차 sin, cos (2차원)
        target_heading = ObsTerm(func=mdp.get_waypoint_heading_error_sin_cos)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # 에피소드 리셋 시 10개의 웨이포인트를 랜덤하게 생성하는 로직 필요
    reset_waypoint_positions = EventTerm(
        func=mdp.reset_random_waypoints,
        mode="reset",
        params={"num_waypoints": 10, "distance_range": (1.0, 2.0)}
    )
    
    # 로봇 위치 초기화
    reset_robot_position = EventTerm(func=mdp.reset_root_at_origin, mode="reset")
    
    # 매 스텝(Interval)마다 웨이포인트 상태 업데이트 실행
    update_waypoints = EventTerm(
        func=mdp.update_waypoint_status,
        mode="interval",
        is_global_time=False,
        interval_range_s=(0.001, 0.001), 
        params={"threshold": 0.2}
    )

# cfg.py

@configclass
class RewardsCfg:
    # [수정] params 안의 weight 대신 RewTerm 자체의 weight를 사용해
    progress = RewTerm(
        func=mdp.position_progress_reward,
        weight=2.0  # 여기가 비어있어서 에러가 났던 거야!
    )
    
    heading = RewTerm(
        func=mdp.heading_alignment_reward,
        weight=0.5,
        params={"sigma": 0.25} # sigma(지수 감쇄 계수)는 params로 유지
    )
    
    goal = RewTerm(
        func=mdp.goal_reached_reward,
        weight=15.0,
        params={"threshold": 0.2}
    )
    
    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.01
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_fell = DoneTerm(func=mdp.bad_orientation, params={"threshold": 0.5})
    
    # 2번째 코드의 핵심 종료 조건: 모든 웨이포인트를 다 통과했을 때
    task_completed = DoneTerm(func=mdp.all_waypoints_reached)


##
# Environment configuration
##


@configclass
class RobotisSh5NavigationEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RobotisSh5NavigationSceneCfg = RobotisSh5NavigationSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # Goal Marker를 여러 개 시각화하기 위해 VisualizationMarkers 사용 권장
    # 여기서는 간단하게 기존 goal_marker 구조를 유지하되, 
    # 실제 위치 업데이트는 mdp에서 수행하도록 설계해.
    waypoint_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/WaypointMarkers",
        markers={
            "current": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), # 현재 타겟: 빨강
            ),
            "future": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), # 남은 타겟: 초록
            ),
        },
    )

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30
        # Viewer settings
        self.viewer.eye = (10.0, 0.0, 8.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation