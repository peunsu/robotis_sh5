import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Robotis-Sh5-Grasp-Direct-v0",
    entry_point=f"{__name__}.robotis_sh5_grasp_env:RobotisSh5GraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotis_sh5_grasp_env_cfg:RobotisSh5GraspEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Robotis-Sh5-Grasp-Pretrain-Direct-v0",
    entry_point=f"{__name__}.robotis_sh5_grasp_pretrain_env:RobotisSh5GraspPretrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotis_sh5_grasp_pretrain_env_cfg:RobotisSh5GraspPretrainEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_pretrain_cfg.yaml",
    },
)

gym.register(
    id="Robotis-Sh5-Grasp-Marl-Direct-v0",
    entry_point=f"{__name__}.robotis_sh5_grasp_marl_env:RobotisSh5GraspMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotis_sh5_grasp_marl_env_cfg:RobotisSh5GraspMarlEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Robotis-Sh5-Grasp-Marl-Pretrain-Direct-v0",
    entry_point=f"{__name__}.robotis_sh5_grasp_marl_pretrain_env:RobotisSh5GraspMarlPretrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotis_sh5_grasp_marl_pretrain_env_cfg:RobotisSh5GraspMarlPretrainEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_pretrain_cfg.yaml",
    },
)
