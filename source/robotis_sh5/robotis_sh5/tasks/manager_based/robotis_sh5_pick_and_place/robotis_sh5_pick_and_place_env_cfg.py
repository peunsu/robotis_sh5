# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

@configclass
class RobotisSh5PickAndPlaceSceneCfg(InteractiveSceneCfg):    
    """Scene configuration for the Robotis SH5 Pick and Place task."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.3, 0.0), rot=(0.0, 0.0, 0.0, 1.0))
    )
    
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.3, 1.1), rot=(0.0, 0.0, 0.0, 1.0))
    )

    # Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/../../../../data/robots/FFW/FFW_SH5_simplified_dex_2.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.75, 1.0, 0.0),
            rot=(0.70711, 0.0, 0.0, -0.70711),
            joint_pos={
                # # Swerve base joints
                "left_wheel_drive": 0.0, "left_wheel_steer": 0.0,
                "right_wheel_drive": 0.0, "right_wheel_steer": 0.0,
                "rear_wheel_drive": 0.0, "rear_wheel_steer": 0.0,

                # Left arm joints
                **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
                # Right arm joints
                #**{f"arm_r_joint{i + 1}": 0.0 for i in range(7)},
                "arm_r_joint2": -1.13,
                "arm_r_joint3": 0.03,
                "arm_r_joint4": -2.1,
                "arm_r_joint5": -1.44,
                "arm_r_joint6": 0.43,
                "arm_r_joint7": -0.65,
                
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
            
            # Actuators for hands
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["finger_l_joint.*", "finger_r_joint.*"],
                velocity_limit_sim=2.2,
                effort_limit_sim=1000.0,  # 20.0
                stiffness=20.0,
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
    
    pass

@configclass
class ActionsCfg:
    """Action specifications."""
    
    # Actions for controlling the joints of the robot
    lift_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["lift_joint"],
        scale=0.5,
    )
    arm_r_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_r_joint[1-7]"],
        scale=0.5,
    )
    hand_r_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_r_joint.*"],
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
            #noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger_r_joint.*", "arm_r_joint[1-7]", "lift_joint"])}
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            #noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger_r_joint.*", "arm_r_joint[1-7]", "lift_joint"])}
        )
        
        # The position and orientation of the end-effectors (in the world frame)
        # left_eef_pos = ObservationTermCfg(
        #     func=mdp.get_eef_pos,
        #     params={"link_name": "hx5_d20_left_base"}
        # )
        # left_eef_quat = ObservationTermCfg(
        #     func=mdp.get_eef_quat,
        #     params={"link_name": "hx5_d20_left_base"}
        # )
        right_eef_pos = ObservationTermCfg(
            func=mdp.get_eef_pos,
            params={"link_name": "hx5_d20_right_base"}
        )
        right_eef_quat = ObservationTermCfg(
            func=mdp.get_eef_quat,
            params={"link_name": "hx5_d20_right_base"}
        )
        
        # The position and orientation of the object, and the relative position of the end-effectors to the object (in the world frame)
        object = ObservationTermCfg(
            func=mdp.object_obs,
             #noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            params={
                # "left_eef_link_name": "hx5_d20_left_base",
                "right_eef_link_name": "hx5_d20_right_base",
            }
        )
        
        # The last input action
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event specifications."""
    
    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    
    reset_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

@configclass
class RewardsCfg:
    """Reward specifications."""
    
    # 1. Distance Reward
    dist_reward = RewardTermCfg(
        func=mdp.object_distance_reward,
        weight=1.0,
        params={
            "fingertip_names": MISSING,
            "palm_name": MISSING
        }
    )
    
    # 2. Height Reward (CrossDex 논문 수식 반영)
    height_reward = RewardTermCfg(
        func=mdp.object_height_reward,
        weight=1.0,
        params={
            "fingertip_names": MISSING,
            "palm_name": MISSING,
            "table_height": MISSING,
            "target_lift_height": MISSING
        }
    )
    
    # 3. XY Displacement Penalty
    xy_penalty = RewardTermCfg(
        func=mdp.object_horizontal_displacement_reward,
        weight=1.0
    )
    
    # 4. Success Reward (CrossDex 성공 조건: 높이 + 파지 여부)
    success_bonus = RewardTermCfg(
        func=mdp.success_reward,
        weight=1.0,
        params={
            "fingertip_names": MISSING,
            "palm_name": MISSING,
            "table_height": MISSING,
            "target_lift_height": MISSING,
            "threshold": 0.05
        }
    )
    
    # Penalty on action rate and joint velocity to encourage smoother motions
    action_rate = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.0001,
    )
    # joint_vel = RewardTermCfg(
    #     func=mdp.joint_vel_l2,
    #     weight=-2.5e-6,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

@configclass
class TerminationsCfg:
    """Termination specifications."""
    
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    
    object_dropping = TerminationTermCfg(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = TerminationTermCfg(func=mdp.task_done_pick_place, params={
        "fingertip_names": MISSING,
        "palm_name": MISSING,
        "table_height": MISSING,
        "target_lift_height": MISSING,
        "threshold": 0.05
    })
    
@configclass
class CurriculumCfg:
    """Curriculum learning configuration."""
    
    # action_rate_curriculum = CurriculumTermCfg(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "reward_manager.cfg.action_rate.weight", 
    #         "modify_fn": mdp.fade_in_reward_weight,
    #         "modify_params": {
    #             "initial_weight": -0.0001,
    #             "target_weight": -0.005,
    #             "grace_period": 8000,
    #             "fade_in_steps": 2000,
    #         }
    #     }
    # )

    # joint_vel_curriculum = CurriculumTermCfg(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "reward_manager.cfg.joint_vel.weight",
    #         "modify_fn": mdp.fade_in_reward_weight,
    #         "modify_params": {
    #             "initial_weight": 0.0,
    #             "target_weight": -0.00001,
    #             "grace_period": 8000,
    #             "fade_in_steps": 2000,
    #         }
    #     }
    # )
    
    pass

@configclass
class RobotisSh5PickAndPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for the Robotis SH5 Pick and Place task."""
    

    scene: RobotisSh5PickAndPlaceSceneCfg = RobotisSh5PickAndPlaceSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)
    #commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    #curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 4096 * 4096
        
        self.decimation = 6
        self.episode_length_s = 20.0
        
        self.viewer.eye = (3.5, 3.5, 3.5)
        #self.viewer.lookat = (0.0, 0.0, 0.0)
        
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = 2
        
        table_height = 1.0
        target_lift_height = 0.3
        
        ee_link_l = "hx5_d20_left_base"
        ee_link_r = "hx5_d20_right_base"
        
        fingertip_links_l = [
            "finger_l_link4",
            "finger_l_link8",
            "finger_l_link12",
            "finger_l_link16",
            "finger_l_link20"
        ]
        fingertip_links_r = [
            "finger_r_link4",
            "finger_r_link8",
            "finger_r_link12",
            "finger_r_link16",
            "finger_r_link20"
        ]
        
        palm_link_l = "hx5_d20_left_base"
        palm_link_r = "hx5_d20_right_base"
        
        self.rewards.dist_reward.params["fingertip_names"] = fingertip_links_r
        self.rewards.dist_reward.params["palm_name"] = palm_link_r
        
        self.rewards.height_reward.params["fingertip_names"] = fingertip_links_r
        self.rewards.height_reward.params["palm_name"] = palm_link_r
        self.rewards.height_reward.params["table_height"] = table_height
        self.rewards.height_reward.params["target_lift_height"] = target_lift_height
        
        self.rewards.success_bonus.params["fingertip_names"] = fingertip_links_r
        self.rewards.success_bonus.params["palm_name"] = palm_link_r
        self.rewards.success_bonus.params["table_height"] = table_height
        self.rewards.success_bonus.params["target_lift_height"] = target_lift_height
        
        self.terminations.success.params["fingertip_names"] = fingertip_links_r
        self.terminations.success.params["palm_name"] = palm_link_r
        self.terminations.success.params["table_height"] = table_height
        self.terminations.success.params["target_lift_height"] = target_lift_height
        
    
@configclass
class RobotisSh5PickAndPlaceEnv_PLAY(RobotisSh5PickAndPlaceEnvCfg):
    """Environment configuration for the Robotis SH5 Pick and Place task (PLAY mode)."""
    
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False  # Disable observation corruption for evaluation