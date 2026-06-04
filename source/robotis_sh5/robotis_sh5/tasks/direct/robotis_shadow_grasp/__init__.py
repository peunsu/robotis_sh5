import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Robotis-Shadow-Grasp-Direct-v0",
    entry_point=f"{__name__}.robotis_shadow_grasp_env:RobotisShadowGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotis_shadow_grasp_env_cfg:RobotisShadowGraspEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Robotis-Shadow-Grasp-Pretrain-Direct-v0",
    entry_point=f"{__name__}.robotis_shadow_grasp_pretrain_env:RobotisShadowGraspPretrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotis_shadow_grasp_pretrain_env_cfg:RobotisShadowGraspPretrainEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_pretrain_cfg.yaml",
    },
)
