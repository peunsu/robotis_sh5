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
from isaaclab.utils import configclass, AdditiveUniformNoiseCfg

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
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    )

@configclass
class ActionsCfg:
    """Action specifications."""
    
    lift_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["lift_joint"],
        scale=0.5,
    )
    arm_l_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_l_joint[1-7]"],
        scale=0.5,
    )
    arm_r_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_r_joint[1-7]"],
        scale=0.5,
    )


@configclass
class ObservationsCfg:
    """Observation specifications."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=AdditiveUniformNoiseCfg(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm_l_joint[1-7]", "arm_r_joint[1-7]", "lift_joint"])}
        )
        
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=AdditiveUniformNoiseCfg(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm_l_joint[1-7]", "arm_r_joint[1-7]", "lift_joint"])}
        )
        
        pose_command_l = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose_l"}
        )
        
        pose_command_r = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose_r"}
        )
        
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event specifications."""
    
    pass

@configclass
class RewardsCfg:
    """Reward specifications."""
    
    pass

@configclass
class TerminationsCfg:
    """Termination specifications."""
    
    pass
    
@configclass
class CurriculumCfg:
    """Curriculum specifications."""

    pass

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