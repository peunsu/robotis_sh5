# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    SceneEntityCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from . import mdp

@configclass
class RobotisSh5ReachSceneCfg(InteractiveSceneCfg):    
    """Scene configuration for the Robotis SH5 Reach task."""
    
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
            rigid_props=sim_utils.RigidPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            joint_pos={
                # # Swerve base joints
                # "left_wheel_drive": 0.0, "left_wheel_steer": 0.0,
                # "right_wheel_drive": 0.0, "right_wheel_steer": 0.0,
                # "rear_wheel_drive": 0.0, "rear_wheel_steer": 0.0,

                # Left arm joints
                **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
                # Right arm joints
                **{f"arm_r_joint{i + 1}": 0.0 for i in range(7)},

                # Left and right gripper joints
                # **{f"gripper_l_joint{i + 1}": 0.0 for i in range(4)},
                # **{f"gripper_r_joint{i + 1}": 0.0 for i in range(4)},
                
                # Wrist joints
                "hx5_d20_left_joint": 0.0,
                "hx5_d20_right_joint": 0.0,
                
                # Left and right finger joints
                **{f"finger_l_joint{i + 1}": 0.0 for i in range(20)},
                **{f"finger_r_joint{i + 1}": 0.0 for i in range(20)},

                # Head joints
                "head_joint1": 0.0,
                "head_joint2": 0.0,

                # Lift joint
                "lift_joint": 0.0,
            }
        ),
        actuators={
            # Actuators for swerve base
            # "base": ImplicitActuatorCfg(
            #     joint_names_expr=[
            #         "left_wheel_drive", "left_wheel_steer",
            #         "right_wheel_drive", "right_wheel_steer",
            #         "rear_wheel_drive", "rear_wheel_steer",
            #     ],
            #     velocity_limit_sim=30.0,
            #     effort_limit_sim=100000.0,
            #     stiffness=10000.0,
            #     damping=100.0,
            # ),

            # Actuator for vertical lift joint
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["lift_joint"],
                velocity_limit_sim=0.2,
                effort_limit_sim=1000000.0,
                stiffness=10000.0,
                damping=100.0,
            ),

            # Actuators for both arms
            "DY_80": ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint[1-2]",
                    "arm_r_joint[1-2]",
                ],
                velocity_limit_sim=15.0,
                effort_limit_sim=61.4,
                stiffness=600.0,
                damping=30.0,
            ),
            "DY_70": ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint[3-6]",
                    "arm_r_joint[3-6]",
                ],
                velocity_limit_sim=15.0,
                effort_limit_sim=31.7,
                stiffness=600.0,
                damping=20.0,
            ),
            "DP-42" : ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint7",
                    "arm_r_joint7",
                ],
                velocity_limit_sim=6.0,
                effort_limit_sim=5.1,
                stiffness=200.0,
                damping=3.0,
            ),

            # Actuators for grippers
            # "gripper_master": ImplicitActuatorCfg(
            #     joint_names_expr=["gripper_l_joint1", "gripper_r_joint1"],
            #     velocity_limit_sim=2.2,
            #     effort_limit_sim=30.0,
            #     stiffness=100.0,
            #     damping=4.0,
            # ),
            # "gripper_slave": ImplicitActuatorCfg(
            #     joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
            #     effort_limit_sim=20.0,
            #     stiffness=2.0,
            #     damping=0.5,
            # ),
            
            # Actuators for hands
            "hand_master": ImplicitActuatorCfg(
                joint_names_expr=["hx5_d20_left_joint", "hx5_d20_right_joint"],
                velocity_limit_sim=2.2,
                effort_limit_sim=30.0,
                stiffness=100.0,
                damping=4.0,
            ),
            # "hand_slave": ImplicitActuatorCfg(
            #     joint_names_expr=["finger_l_joint[1-20]", "finger_r_joint[1-20]"],
            #     effort_limit_sim=20.0,
            #     stiffness=2.0,
            #     damping=0.5,
            # ),

            # Actuators for head joints
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_joint1", "head_joint2"],
                velocity_limit_sim=2.0,
                effort_limit_sim=30.0,
                stiffness=150.0,
                damping=3.0,
            ),
        },
    )
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

@configclass
class ActionsCfg:
    """Action specifications."""

    # SwerveDriveAction outputs [linear_vel_x, linear_vel_y, angular_vel_z]
    base_velocity = mdp.SwerveDriveActionCfg(
        asset_name="robot",
        joint_names=[".*_steer", ".*_drive"],
        scale=(1.0, 1.0, 2.0)  # vx, vy 스케일은 1.0, 회전(w) 스케일은 2.0
    )


@configclass
class ObservationsCfg:
    """Observation specifications."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        # Wheel joint positions with respect to the robot base (6 joints: 3 drive + 3 steer)
        joint_pos_rel = ObservationTermCfg(
            func=mdp.joint_pos_rel, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*drive", ".*steer"])}
        )
        
        # Wheel joint velocities with respect to the robot base (6 joints: 3 drive + 3 steer)
        joint_vel_rel = ObservationTermCfg(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*drive", ".*steer"])}
        )
        
        # Root linear and angular velocity in the robot's local frame (3 linear + 3 angular)
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        
        # Relative position to the current target waypoint (x, y in the robot's local frame)
        rel_goal_pos = ObservationTermCfg(func=mdp.get_rel_pos_to_current_waypoint)
        
        # The index of the current target waypoint (scalar)
        # target_index = ObservationTermCfg(func=mdp.get_target_waypoint_index)
        
        # Relative heading to the current target waypoint, represented as sin and cos (2 values)
        target_heading = ObservationTermCfg(func=mdp.get_waypoint_heading_error_sin_cos)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event specifications."""
    
    # Reset waypoint positions at the start of each episode, with randomization
    reset_waypoint_positions = EventTermCfg(
        func=mdp.reset_random_waypoints,
        mode="reset",
        params={"num_waypoints": 10, "waypoint_params": (1.0, 2.0, torch.pi / 2)}
    )
    
    # Reset robot position at the start of each episode
    reset_robot_position = EventTermCfg(func=mdp.reset_root_at_origin, mode="reset")
    
    # Update waypoint status every step (Interval)
    update_waypoints = EventTermCfg(
        func=mdp.update_waypoint_status,
        mode="interval",
        is_global_time=False,
        interval_range_s=(0.001, 0.001), 
        params={"threshold": 0.2}
    )

@configclass
class RewardsCfg:
    """Reward specifications."""
    
    # The difference in distance to the target waypoint since the last step (progress reward)
    progress = RewardTermCfg(
        func=mdp.position_progress_reward,
        weight=2.0
    )
    
    # Alignment of the robot's heading with the direction to the target waypoint (encourages facing the target)
    # heading = RewardTermCfg(
    #     func=mdp.heading_alignment_reward,
    #     weight=0.5,
    #     params={"sigma": 0.25}
    # )
    
    # A reward for reaching the goal (current waypoint), given when the robot is within a certain threshold distance
    goal = RewardTermCfg(
        func=mdp.goal_reached_reward,
        weight=15.0,
        params={"threshold": 0.2}
    )
    
    # A small penalty on action to encourage smoother control
    action_rate = RewardTermCfg(
        func=mdp.action_rate_l2, 
        weight=-0.01
    )

@configclass
class TerminationsCfg:
    """Termination specifications."""
    
    # Termination if the episode exceeds the maximum time limit
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    
    # Termination if the robot's orientation is too tilted (e.g., fallen over)
    robot_fell = TerminationTermCfg(func=mdp.bad_orientation, params={"threshold": 0.5})
    
    # Termination if all waypoints have been reached (task completion)
    task_completed = TerminationTermCfg(func=mdp.all_waypoints_reached)
    
@configclass
class CurriculumCfg:
    """Curriculum specifications."""

    waypoint_distance_curriculum = CurriculumTermCfg(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.reset_waypoint_positions.params.waypoint_params",
            "modify_fn": mdp.adaptive_distance_curriculum,
            "modify_params": {
                "waypoint_params": (3.0, 5.0, torch.pi),
                "grace_period": 0,
                "fade_in_steps": 0,
            },
        },
    )

@configclass
class RobotisSh5ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for the Robotis SH5 Reach task."""
    

    scene: RobotisSh5ReachSceneCfg = RobotisSh5ReachSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    waypoint_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/WaypointMarkers",
        # Current target waypoint: larger red sphere, Future waypoints: smaller green spheres
        markers={
            "current": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "future": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        
        self.decimation = 4
        self.episode_length_s = 30
        
        self.viewer.eye = (25.0, 0.0, 8.0)
        self.viewer.lookat = (10.0, 0.0, 0.0)
        
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation