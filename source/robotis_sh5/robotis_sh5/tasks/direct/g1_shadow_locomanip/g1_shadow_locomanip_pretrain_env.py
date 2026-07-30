"""Kinematic-only pretrain env for the G1+Shadow loco-manip task.

Thin subclass of the train env: the ONLY functional change is that no object is ever spawned
(`_build_object_cfg` forces `_object_cfg=None`), so `_has_object` is False throughout and every
object obs/reward/termination/cache term is inert (the train env already runs the robot-only
kinematic path when no object USD is present). The policy learns whole-body keypoint tracking
(body + hand + fingertip + foot) purely from the reference.

At the end of pretraining, `train.py` dumps this env's per-frame state cache — remapped to the
209-D layout the TRAIN env's `_reset_idx` pretrain branch reads — as `pretrain_state_cache.npz`
for RSI warm-start (`dump_pretrain_cache_209`). Obs/action dims (621/65) match the train env so
`pretrain.pt` transfers by shape.
"""

from __future__ import annotations

import torch

from .g1_shadow_locomanip_env import G1ShadowLocomanipEnv
from .g1_shadow_locomanip_pretrain_env_cfg import G1ShadowLocomanipPretrainEnvCfg


class G1ShadowLocomanipPretrainEnv(G1ShadowLocomanipEnv):
    cfg: G1ShadowLocomanipPretrainEnvCfg

    def _build_object_cfg(self, cfg) -> None:
        # Kinematic pretrain: never spawn an object regardless of whether a converted USD exists.
        # Reference object tensors (_ref_obj_*) are already built in _load_reference_trajectories,
        # so block-C obs stay 621-D (identical to the train env's robot-only path); _has_object=False
        # makes every LIVE object obs/reward/termination/cache term inert.
        self._object_cfg = None

    def dump_pretrain_cache_209(self) -> torch.Tensor:
        """Remap the 222-D train cache (populated object-free → object cols are 0) to the 209-D
        pretrain layout the train env's `_reset_idx` pretrain branch reads:
            [0]reward [1:14]root(13) [14:79]jpos(65) [79:144]jvel(65) [144:209]smoothed(65).
        (Train layout drops the [14:27] object block that pretrain never uses.)"""
        c = self._state_cache                                    # (F,222)
        out = torch.empty(self._ref_len, 209, device=self.device)
        out[:, 0] = c[:, 0]                                      # reward
        out[:, 1:14] = c[:, 1:14]                               # root (13)
        out[:, 14:79] = c[:, 27:92]                             # jpos (65)  ← train[27:92]
        out[:, 79:144] = c[:, 92:157]                           # jvel (65)  ← train[92:157]
        out[:, 144:209] = c[:, 157:222]                         # smoothed (65) ← train[157:222]
        return out
