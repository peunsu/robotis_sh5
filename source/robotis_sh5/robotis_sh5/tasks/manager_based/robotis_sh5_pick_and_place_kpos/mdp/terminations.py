# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg

from .utils import get_virtual_link_poses

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height

def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, float] = {"x": 0.2, "z": 0.5}, # 예: 초기 위치서 20cm 이상 벗어나면 탈락
) -> torch.Tensor:
    # 객체 가져오기
    object: RigidObject = env.scene[asset_cfg.name]
    
    # 1. 현재 로컬 위치 계산
    object_pos_local = object.data.root_pos_w - env.scene.env_origins
    
    # 2. 저장해둔 초기 위치 가져오기 (없을 경우를 대비해 예외 처리)
    if hasattr(env, "object_initial_pos_b"):
        initial_pos = env.object_initial_pos_b
    else:
        # 아직 리셋이 한 번도 안 일어났다면 현재 위치를 기준으로 함
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 3. 초기 위치와의 차이 계산
    diff = torch.abs(object_pos_local - initial_pos)
    
    # 4. 설정된 범위를 벗어났는지 확인
    # 각 축별로 설정된 threshold를 넘어가면 True
    out_x = diff[:, 0] > in_bound_range.get("x", 1.0)
    out_z = diff[:, 2] > in_bound_range.get("z", 1.0)

    return out_x | out_z

def task_done_pick_place(
    env: ManagerBasedRLEnv, 
    command_name: str,
    threshold: float = 0.05
) -> torch.Tensor:
    """
    물체가 목표 위치(30cm 위)에 lambda_0 이내로 들어오면 즉시 성공으로 판단해.
    """
    # 1. 커맨드 매니저에서 목표 위치(World Frame) 가져오기
    # DexYCBCommandTerm의 command 마지막 3차원이 target_pos_w임
    command = env.command_manager.get_command(command_name)
    target_pos_w = command[:, -3:]
    
    # 2. 오브젝트의 현재 월드 좌표
    obj_pos_w = env.scene["object"].data.root_pos_w
    
    # 3. 거리 계산 (L2 Norm)
    d_obj = torch.norm(obj_pos_w - target_pos_w, p=2, dim=-1)

    # 4. 성공 여부 판정 (0.05m 이내면 True)
    is_success = d_obj < threshold

    return is_success

def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action"""
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 2)).any(dim=1)