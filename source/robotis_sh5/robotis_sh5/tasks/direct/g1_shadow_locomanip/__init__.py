import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Robotis-G1-Shadow-Locomanip-Direct-v0",
    entry_point=f"{__name__}.g1_shadow_locomanip_env:G1ShadowLocomanipEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shadow_locomanip_env_cfg:G1ShadowLocomanipEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Robotis-G1-Shadow-Locomanip-Pretrain-Direct-v0",
    entry_point=f"{__name__}.g1_shadow_locomanip_pretrain_env:G1ShadowLocomanipPretrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shadow_locomanip_pretrain_env_cfg:G1ShadowLocomanipPretrainEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_pretrain_cfg.yaml",
    },
)
