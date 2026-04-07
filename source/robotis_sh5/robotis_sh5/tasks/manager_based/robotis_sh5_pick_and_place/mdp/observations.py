# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_obs(
    env: ManagerBasedRLEnv,
    # left_eef_link_name: str,
    right_eef_link_name: str,
) -> torch.Tensor:
    """
    Object observations (in world frame):
        object pos,
        object quat,
        left_eef to object,
        right_eef to object,
    """

    # body_pos_w = env.scene["robot"].data.body_pos_w
    # left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    # right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    # left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    # right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["object"].data.root_quat_w

    # left_eef_to_object = object_pos - left_eef_pos
    # right_eef_to_object = object_pos - right_eef_pos
    
    # print(f"Object position: {object_pos}, Object quaternion: {object_quat}")
    # print(f"Right EEF position: {right_eef_pos}")

    return torch.cat(
        (
            object_pos,
            object_quat,
            #left_eef_to_object,
            #right_eef_to_object,
        ),
        dim=1,
    )


def get_eef_pos(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins

    return left_eef_pos


def get_eef_quat(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_quat = body_quat_w[:, left_eef_idx]

    return left_eef_quat