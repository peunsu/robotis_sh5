import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Robotis-Shadow-Grasp-Rsi-Direct-v0",
    entry_point=f"{__name__}.robotis_shadow_grasp_rsi_env:RobotisShadowGraspRsiEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotis_shadow_grasp_rsi_env_cfg:RobotisShadowGraspRsiEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Robotis-Shadow-Grasp-Rsi-Pretrain-Direct-v0",
    entry_point=f"{__name__}.robotis_shadow_grasp_rsi_pretrain_env:RobotisShadowGraspRsiPretrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotis_shadow_grasp_rsi_pretrain_env_cfg:RobotisShadowGraspRsiPretrainEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_pretrain_cfg.yaml",
    },
)
