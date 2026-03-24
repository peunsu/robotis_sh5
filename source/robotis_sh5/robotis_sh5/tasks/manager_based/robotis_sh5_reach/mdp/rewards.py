# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def position_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to use for the position error calculation.
        asset_cfg (SceneEntityCfg): The configuration for the asset.

    Returns:
        torch.Tensor: The position error between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        std (float): The standard deviation for the tanh kernel.
        command_name (str): The name of the command to use for the position error calculation.
        asset_cfg (SceneEntityCfg): The configuration for the asset.

    Returns:
        torch.Tensor: The reward for tracking the position.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to use for the orientation error calculation.
        asset_cfg (SceneEntityCfg): The configuration for the asset.

    Returns:
        torch.Tensor: The orientation error between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def bimanual_error_difference_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg_l: SceneEntityCfg,
    asset_cfg_r: SceneEntityCfg,
    command_name_l: str,
    command_name_r: str
) -> torch.Tensor:
    """
    좌우 팔의 위치 에러 차이를 계산하여 불균형 학습을 방지하는 페널티 함수.
    
    계산 방식: |Error_Left - Error_Right|
    """
    # 1. 에셋 및 커맨드 데이터 추출
    asset: RigidObject = env.scene[asset_cfg_l.name]
    command_l = env.command_manager.get_command(command_name_l)
    command_r = env.command_manager.get_command(command_name_r)
    
    # 2. 로봇의 현재 Root 상태 (World frame 변환용)
    root_pos = asset.data.root_state_w[:, :3]
    root_quat = asset.data.root_state_w[:, 3:7]
    
    # 3. 좌우 목표 지점 계산 (Base -> World)
    des_pos_l_w, _ = combine_frame_transforms(root_pos, root_quat, command_l[:, :3])
    des_pos_r_w, _ = combine_frame_transforms(root_pos, root_quat, command_r[:, :3])
    
    # 4. 좌우 현재 EE 위치 추출
    # asset_cfg_l과 asset_cfg_r에서 각각 지정된 body_names의 ID를 사용
    curr_pos_l_w = asset.data.body_state_w[:, asset_cfg_l.body_ids[0], :3]
    curr_pos_r_w = asset.data.body_state_w[:, asset_cfg_r.body_ids[0], :3]
    
    # 5. 각 팔의 개별 L2 에러 계산
    error_l = torch.norm(curr_pos_l_w - des_pos_l_w, dim=1)
    error_r = torch.norm(curr_pos_r_w - des_pos_r_w, dim=1)
    
    # 6. 두 에러의 차이(절대값) 반환
    # 이 값이 0에 가까울수록 양팔이 균일하게 학습되고 있다는 뜻이야.
    return torch.abs(error_l - error_r)