# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from dataclasses import MISSING

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
from isaaclab.utils.noise import UniformNoiseCfg

from . import mdp

@configclass
class RobotisSh5ReachSceneCfg(InteractiveSceneCfg):    
    """Scene configuration for the Robotis SH5 Reach task."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/../../../../data/robots/FFW/FFW_SH5_simplified.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                # # Swerve base joints
                "left_wheel_drive": 0.0, "left_wheel_steer": 0.0,
                "right_wheel_drive": 0.0, "right_wheel_steer": 0.0,
                "rear_wheel_drive": 0.0, "rear_wheel_steer": 0.0,

                # Left arm joints
                **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
                # Right arm joints
                **{f"arm_r_joint{i + 1}": 0.0 for i in range(7)},

                # Left and right gripper joints
                # **{f"gripper_l_joint{i + 1}": 0.0 for i in range(4)},
                # **{f"gripper_r_joint{i + 1}": 0.0 for i in range(4)},
                
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
            "base": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_wheel_drive", "left_wheel_steer",
                    "right_wheel_drive", "right_wheel_steer",
                    "rear_wheel_drive", "rear_wheel_steer",
                ],
                velocity_limit_sim=30.0,
                effort_limit_sim=100000.0,
                stiffness=10000.0,
                damping=100.0,
            ),

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
                effort_limit_sim=1000.0,  # 61.4
                stiffness=600.0,
                damping=30.0,
            ),
            "DY_70": ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint[3-6]",
                    "arm_r_joint[3-6]",
                ],
                velocity_limit_sim=15.0,
                effort_limit_sim=1000.0,  # 31.7
                stiffness=600.0,
                damping=20.0,
            ),
            "DP-42" : ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint7",
                    "arm_r_joint7",
                ],
                velocity_limit_sim=6.0,
                effort_limit_sim=1000.0,  # 5.1
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
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["finger_l_joint[1-20]", "finger_r_joint[1-20]"],
                velocity_limit_sim=2.2,
                effort_limit_sim=1000.0,  # 20.0
                stiffness=2.0,
                damping=0.5,
            ),

            # Actuators for head joints
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_joint1", "head_joint2"],
                velocity_limit_sim=2.0,
                effort_limit_sim=1000.0,  # 30.0
                stiffness=150.0,
                damping=3.0,
            ),
        },
    )
    
    # Light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    )
    
@configclass
class CommandsCfg:
    """Command specifications."""
    
    # End-effector pose commands for both arms,
    # which will be generated and updated at random intervals to create a dynamic reaching task.
    ee_pose_l = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.65),
            pos_y=(0.1, 0.3),
            pos_z=(0.8, 1.6),
            roll=(torch.pi / 2 - torch.pi / 8, torch.pi / 2 + torch.pi / 8),
            pitch=(torch.pi - torch.pi / 8, torch.pi + torch.pi / 8),
            yaw=(torch.pi / 2 - torch.pi / 8, torch.pi / 2 + torch.pi / 8),
        ),
    )
    ee_pose_r = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.65),
            pos_y=(-0.3, -0.1),
            pos_z=(0.8, 1.6),
            roll=(- torch.pi / 2 - torch.pi / 8, - torch.pi / 2 + torch.pi / 8),
            pitch=(torch.pi - torch.pi / 8, torch.pi + torch.pi / 8),
            yaw=(- torch.pi / 2 - torch.pi / 8, - torch.pi / 2 + torch.pi / 8),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications."""
    
    # Actions for controlling the joints of the robot
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

        # Relative joint positions and velocities with respect to the default pose
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm_l_joint[1-7]", "arm_r_joint[1-7]", "lift_joint"])}
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm_l_joint[1-7]", "arm_r_joint[1-7]", "lift_joint"])}
        )
        
        # End-effector pose command for both arms
        pose_command_l = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose_l"}
        )
        pose_command_r = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose_r"}
        )
        
        # The last input action
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event specifications."""
    
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    """Reward specifications."""
    
    # Reward left end-effector tracking
    end_effector_position_tracking_left = RewardTermCfg(
        func=mdp.position_command_error,
        weight=-0.30,  # default: -0.25
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose_l"
        },
    )
    end_effector_position_tracking_fine_grained_left = RewardTermCfg(
        func=mdp.position_command_error_tanh,
        weight=0.18,  # default: 0.12
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "command_name": "ee_pose_l"
        },
    )
    end_effector_orientation_tracking_left = RewardTermCfg(
        func=mdp.orientation_command_error,
        weight=-0.12,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose_l"
        },
    )

    # Reward right end-effector tracking
    end_effector_position_tracking_right = RewardTermCfg(
        func=mdp.position_command_error,
        weight=-0.30,  # default: -0.25  
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose_r"
        },
    )
    end_effector_position_tracking_fine_grained_right = RewardTermCfg(
        func=mdp.position_command_error_tanh,
        weight=0.18,  # default: 0.12
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "command_name": "ee_pose_r"
        },
    )
    end_effector_orientation_tracking_right = RewardTermCfg(
        func=mdp.orientation_command_error,
        weight=-0.12,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose_r"
        },
    )

    # Penalty on action rate and joint velocity to encourage smoother motions
    action_rate = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=0.0,  # default: -0.0001
    )
    joint_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=0.0,  # default: -0.0001
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Experimental: a penalty on the difference between the left and right end-effector errors to encourage more balanced bimanual coordination
    # bimanual_diff_penalty = RewardTermCfg(
    #     func=mdp.bimanual_error_difference_penalty,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg_l": SceneEntityCfg("robot", body_names=["hx5_d20_left_base"]),
    #         "asset_cfg_r": SceneEntityCfg("robot", body_names=["hx5_d20_right_base"]),
    #         "command_name_l": "ee_pose_l",
    #         "command_name_r": "ee_pose_r",
    #     },
    # )

@configclass
class TerminationsCfg:
    """Termination specifications."""
    
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    
@configclass
class CurriculumCfg:
    """Curriculum learning configuration."""

    # action_rate = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.0001, "num_steps": 10000}
    # )
    # joint_vel = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.0001, "num_steps": 10000}
    # )
    
    action_rate_curriculum = CurriculumTermCfg(
        func=mdp.modify_env_param,
        params={
            "address": "reward_manager.cfg.action_rate.weight", 
            "modify_fn": mdp.fade_in_reward_weight,
            "modify_params": {
                "target_weight": -0.0001,
                "grace_period": 10000,
                "fade_in_steps": 20000,
            }
        }
    )

    joint_vel_curriculum = CurriculumTermCfg(
        func=mdp.modify_env_param,
        params={
            "address": "reward_manager.cfg.joint_vel.weight",
            "modify_fn": mdp.fade_in_reward_weight,
            "modify_params": {
                "target_weight": -0.0001,
                "grace_period": 10000,
                "fade_in_steps": 20000,
            }
        }
    )

@configclass
class RobotisSh5ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for the Robotis SH5 Reach task."""
    

    scene: RobotisSh5ReachSceneCfg = RobotisSh5ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        
        self.sim.physx.gpu_max_rigid_patch_count = 4096 * 4096
        
        self.decimation = 2
        self.episode_length_s = 12.0
        
        self.viewer.eye = (3.5, 3.5, 3.5)
        #self.viewer.lookat = (0.0, 0.0, 0.0)
        
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        
        ee_link_l = "hx5_d20_left_base"
        ee_link_r = "hx5_d20_right_base"
        
        self.commands.ee_pose_l.body_name = ee_link_l
        self.commands.ee_pose_r.body_name = ee_link_r
        
        self.rewards.end_effector_position_tracking_left.params["asset_cfg"].body_names = [ee_link_l]
        self.rewards.end_effector_position_tracking_fine_grained_left.params["asset_cfg"].body_names = [ee_link_l]
        self.rewards.end_effector_orientation_tracking_left.params["asset_cfg"].body_names = [ee_link_l]
        
        self.rewards.end_effector_position_tracking_right.params["asset_cfg"].body_names = [ee_link_r]
        self.rewards.end_effector_position_tracking_fine_grained_right.params["asset_cfg"].body_names = [ee_link_r]
        self.rewards.end_effector_orientation_tracking_right.params["asset_cfg"].body_names = [ee_link_r]
        
        
    
@configclass
class RobotisSh5ReachEnv_PLAY(RobotisSh5ReachEnvCfg):
    """Environment configuration for the Robotis SH5 Reach task (PLAY mode)."""
    
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False  # Disable observation corruption for evaluation