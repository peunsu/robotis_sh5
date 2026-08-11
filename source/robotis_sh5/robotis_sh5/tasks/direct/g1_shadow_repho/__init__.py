import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Robotis-G1-Shadow-Locomanip-Repho-Direct-v0",
    entry_point=f"{__name__}.g1_shadow_repho_env:G1ShadowRephoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shadow_repho_env_cfg:G1ShadowRephoEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Robotis-G1-Shadow-Locomanip-Repho-Pretrain-Direct-v0",
    entry_point=f"{__name__}.g1_shadow_repho_pretrain_env:G1ShadowRephoPretrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shadow_repho_pretrain_env_cfg:G1ShadowRephoPretrainEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_pretrain_cfg.yaml",
    },
)
