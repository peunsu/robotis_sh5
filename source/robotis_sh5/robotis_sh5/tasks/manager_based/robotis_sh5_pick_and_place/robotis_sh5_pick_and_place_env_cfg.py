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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.35, 0.0), rot=(0.0, 0.0, 0.0, 1.0))
    )
    
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            #usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            usd_path="/home/peunsu/workspace/robotis_sh5/retargeting/DexYCB/models/006_mustard_bottle/textured.usd",
            scale=(1.00, 1.00, 1.00),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.3, 1.1), rot=(0.0, 0.0, 0.0, 1.0))
    )

    # Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/../../../../data/robots/FFW/FFW_SH5_simplified_dex.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.00, 0.95, 0.0),
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
                effort_limit_sim=61.4,  # 61.4
                stiffness=600.0,
                damping=30.0,
            ),
            "DY_70": ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint[3-6]",
                    "arm_r_joint[3-6]",
                ],
                velocity_limit_sim=15.0,
                effort_limit_sim=31.7,  # 31.7
                stiffness=600.0,
                damping=20.0,
            ),
            "DP-42" : ImplicitActuatorCfg(
                joint_names_expr=[
                    "arm_l_joint7",
                    "arm_r_joint7",
                ],
                velocity_limit_sim=6.0,
                effort_limit_sim=5.1,  # 5.1
                stiffness=200.0,
                damping=3.0,
            ),
            
            # Actuators for hands
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["finger_l_joint.*", "finger_r_joint.*"],
                velocity_limit_sim=2.2,
                effort_limit_sim=3.0,  # 20.0
                stiffness=20.0,
                damping=0.5,
            ),

            # Actuators for head joints
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_joint1", "head_joint2"],
                velocity_limit_sim=2.0,
                effort_limit_sim=30.0,  # 30.0
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
    
    hand_pose_r = mdp.DexYCBCommandTermCfg(
        asset_name="robot",
        file_path="/home/peunsu/workspace/robotis_sh5/retargeting/trajectories/20200709_143257.npy",
        frame_idx=29,
        table_height=MISSING,
        body_name=MISSING,
        object_name="object",
        fix_hand_command=True, # Whether to fix the hand pose in the command (for curriculum learning)
        resampling_time_range=(20.0, 20.0), # same as episode length to sample only once at reset
        debug_vis=True
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
        joint_effort = ObservationTermCfg(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        right_hand_pose = ObservationTermCfg(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["finger_r_link.*"])}
        )
        right_eef_pose = ObservationTermCfg(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["hx5_d20_right_base"])}
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
        
        hand_pose_command_r = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "hand_pose_r"},
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
    
    reset_object_from_data = EventTermCfg(
        func=mdp.reset_object_to_tray_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "file_path": "/home/peunsu/workspace/robotis_sh5/retargeting/trajectories/20200709_143257.npy",
            "frame_idx": 29, # 원하는 프레임 번호
            "table_height": MISSING
        },
    )
    
    # reset_object = EventTermCfg(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": [-0.01, 0.01],
    #             "y": [-0.01, 0.01],
    #         },
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )

@configclass
class RewardsCfg:
    """Reward specifications for the DexYCB Pick-and-Place task."""

    # ---------------------------------------------------------
    # 1. Imitation / Reaching Rewards (기본 자세 및 접근)
    # ---------------------------------------------------------
    
    joint_pos_imitation = RewardTermCfg(
        func=mdp.joint_angle_error,
        weight=-0.05,  # 페널티이므로 음수
        params={
            "command_name": "hand_pose_r",
            "asset_cfg": SceneEntityCfg("robot")
        },
    )

    root_translation = RewardTermCfg(
        func=mdp.root_translation_error,
        weight=-0.6,
        params={
            "command_name": "hand_pose_r",
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING)
        },
    )

    root_rotation = RewardTermCfg(
        func=mdp.root_rotation_error,
        weight=-0.1,
        params={
            "command_name": "hand_pose_r",
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING)
        },
    )

    # 수식 (7): 각 손가락이 물체에 가까워지도록 유도
    fingertip_reaching = RewardTermCfg(
        func=mdp.reaching_reward,
        weight=0.0, # wr 가중치
        params={
            "fingertip_names": MISSING,
            "palm_name": MISSING,
            "object_name": "object"
        },
    )

    # ---------------------------------------------------------
    # 2. Task Rewards (Lifting & Moving)
    # ---------------------------------------------------------

    # 수식 (8): 파지 성공 후 들어올리기 보상
    object_lifting = RewardTermCfg(
        func=mdp.lifting_reward_fullbody,
        weight=0.0, # 성공 보상이므로 큰 양수 가중치
        params={
            "command_name": "hand_pose_r",
            "asset_cfg": SceneEntityCfg("robot"),
            "object_name": "object",
            "fingertip_names": MISSING,
            "palm_name": MISSING,
            "wrist_link_name": MISSING, # 실제 로봇의 손목 링크 이름
            "thresholds": {
                "lambda_f1": 0.12,  # 관절 오차 허용치
                "lambda_f2": 0.6,   # 손가락-물체 거리 허용치
                "lambda_0": 0.05    # 목표 도달 판단 기준 (5cm)
            }
        },
    )

    # 수식 (9): 물체를 목표 지점으로 이동
    object_moving = RewardTermCfg(
        func=mdp.moving_reward,
        weight=0.0,
        params={
            "command_name": "hand_pose_r",
            "object_name": "object",
            "weight_m": 2.0,   # 거리 페널티 wm
            "weight_b": 10.0,  # 보너스 가중치 wb
            "lambda_0": 0.05   # 보너스 구간 threshold
        },
    )

    # ---------------------------------------------------------
    # 3. Regularization (규제 항)
    # ---------------------------------------------------------
    
    action_rate = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.0001,
    )

@configclass
class TerminationsCfg:
    """Termination specifications."""
    
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    
    object_dropping = TerminationTermCfg(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = TerminationTermCfg(func=mdp.task_done_pick_place, params={
        "command_name": "hand_pose_r",
        "threshold": 0.05
    })
    
@configclass
class CurriculumCfg:
    """Curriculum learning configuration."""
    
    hand_pose_command_curriculum = CurriculumTermCfg(
        func=mdp.modify_env_param,
        params={
            "address": "command_manager.cfg.hand_pose_r.fix_hand_command",
            "modify_fn": mdp.fix_hand_command_curriculum,  # 단순히 fix_hand_command를 True로 설정
            "modify_params": {
                "fix_hand_command": False,
                "num_step": 5000
            }
        }
    )
    
    # joint_pos_imitation_reward_schedule = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "joint_pos_imitation",
    #         "weight": -0.1,
    #         "num_steps": 5000,
    #     }
    # )
    
    fingertip_reaching_reward_schedule = CurriculumTermCfg(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "fingertip_reaching",
            "weight": -0.5,
            "num_steps": 10000,
        }
    )
        
    object_lifting_reward_schedule = CurriculumTermCfg(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "object_lifting",
            "weight": 0.1,
            "num_steps": 10000,
        }
    )
    
    object_moving_reward_schedule = CurriculumTermCfg(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "object_moving",
            "weight": 1.0,
            "num_steps": 10000,
        }
    )

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
    

    scene: RobotisSh5PickAndPlaceSceneCfg = RobotisSh5PickAndPlaceSceneCfg(num_envs=16, env_spacing=3.0, replicate_physics=True)
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
        
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 4096 * 4096
        
        self.decimation = 2
        self.episode_length_s = 10.0
        
        self.viewer.eye = (3.5, 3.5, 3.5)
        #self.viewer.lookat = (0.0, 0.0, 0.0)
        
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        
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
        
        # self.rewards.dist_reward.params["fingertip_names"] = fingertip_links_r
        # self.rewards.dist_reward.params["palm_name"] = palm_link_r
        
        # self.rewards.height_reward.params["fingertip_names"] = fingertip_links_r
        # self.rewards.height_reward.params["palm_name"] = palm_link_r
        # self.rewards.height_reward.params["table_height"] = table_height
        # self.rewards.height_reward.params["target_lift_height"] = target_lift_height
        
        # self.rewards.success_bonus.params["fingertip_names"] = fingertip_links_r
        # self.rewards.success_bonus.params["palm_name"] = palm_link_r
        # self.rewards.success_bonus.params["table_height"] = table_height
        # self.rewards.success_bonus.params["target_lift_height"] = target_lift_height
        
        self.rewards.root_translation.params["asset_cfg"].body_names = [ee_link_r]
        self.rewards.root_rotation.params["asset_cfg"].body_names = [ee_link_r]
        
        self.rewards.fingertip_reaching.params["fingertip_names"] = fingertip_links_r
        self.rewards.fingertip_reaching.params["palm_name"] = palm_link_r
        
        self.rewards.object_lifting.params["fingertip_names"] = fingertip_links_r
        self.rewards.object_lifting.params["palm_name"] = palm_link_r
        self.rewards.object_lifting.params["wrist_link_name"] = palm_link_r
        
        self.events.reset_object_from_data.params["table_height"] = table_height
        
        self.commands.hand_pose_r.body_name = ee_link_r
        self.commands.hand_pose_r.table_height = table_height
        
    
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