"""Floating-hand dexterous pretrain — stage 1 of the two-stage loco-manip plan.

Stage 1 learns bimanual manipulation with two FREE-FLOATING Shadow hands (no G1 arm, no body), then
exports its wrist + finger trajectory for the full-body stage to track. The premise, from the
professor's note, is that the kinematic capture's OBJECT pose is trustworthy while the retargeted
HAND pose is not — so the hand pose is what gets refined under physics.

Package contents
----------------
    build target        scripts/process_dataset/assets/build_shadow_floating_usd.py  (writes the USDs)
    shadow_float_cfg    ArticulationCfg per hand: fix_root_link=False, damping 100, fingers-only
    ..._env_cfg         subclass of the train cfg with the body terms neutralised
    PHASE_B_CHANGES.md  the env-body edit list, which needs the GPU to validate

Task id: `Robotis-G1-Shadow-HandPretrain-Direct-v0`. The agents yaml is a COPY of the
sonic-residual one with its own `experiment.directory`, so stage-1 tensorboard runs land in
`logs/skrl/g1_shadow_hand_pretrain/` instead of mixing into the full-body training tree. The MLP
shape comes from the obs/action spaces, not the yaml, so the copy needs no size edits.
"""

import gymnasium as gym

from . import agents

from . import shadow_float_cfg  # noqa: F401

gym.register(
    id="Robotis-G1-Shadow-HandPretrain-Direct-v0",
    entry_point=f"{__name__}.g1_shadow_hand_pretrain_env:G1ShadowHandPretrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_shadow_hand_pretrain_env_cfg:G1ShadowHandPretrainEnvCfg"),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
