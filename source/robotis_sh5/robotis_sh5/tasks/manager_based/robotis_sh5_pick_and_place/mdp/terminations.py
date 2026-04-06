# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
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

def task_done_pick_place(
    env: ManagerBasedRLEnv, 
    fingertip_names: list, 
    palm_name: str,
    table_height: float = 1.0,
    target_lift_height: float = 0.6,
    threshold: float = 0.05
) -> torch.Tensor:
    obj_pos = env.scene["object"].data.root_pos_w
    h = obj_pos[:, 2]
    target_h = table_height + target_lift_height

    height_condition = torch.abs(h - target_h) <= threshold

    v_fingertip_pos, v_palm_pos = get_virtual_link_poses(env, fingertip_names, palm_name)
    
    avg_dist_fingertips = torch.stack([torch.norm(pos - obj_pos, dim=1) for pos in v_fingertip_pos], dim=1).mean(dim=1)
    dist_palm = torch.norm(v_palm_pos - obj_pos, dim=1)

    # 논문 성공 조건 반영
    grasp_condition = (avg_dist_fingertips <= 0.12) | (dist_palm <= 0.15)
    is_success = height_condition & grasp_condition

    return is_success