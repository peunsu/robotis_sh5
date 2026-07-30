import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0",
    entry_point=f"{__name__}.g1_shadow_sonic_residual_env:G1ShadowSonicResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shadow_sonic_residual_env_cfg:G1ShadowSonicResidualEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Robotis-G1-Shadow-Locomanip-SonicResidual-Pretrain-Direct-v0",
    entry_point=f"{__name__}.g1_shadow_sonic_residual_pretrain_env:G1ShadowSonicResidualPretrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shadow_sonic_residual_pretrain_env_cfg:G1ShadowSonicResidualPretrainEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_pretrain_cfg.yaml",
    },
)
