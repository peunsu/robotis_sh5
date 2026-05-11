# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab extension for the FFW-SH5 (Fully Functional Wheeled - Super Hand 5) robotic platform. Implements RL-based manipulation tasks with hand pose retargeting from the DexYCB dataset using the MANO hand model.

## Commands

**Installation** (requires Isaac Lab Python environment):
```bash
python -m pip install -e source/robotis_sh5
```

**Verify environment registration:**
```bash
python scripts/list_envs.py
```

**Test environments with dummy agents:**
```bash
python scripts/zero_agent.py --task=<TASK_NAME>
python scripts/random_agent.py --task=<TASK_NAME>
```

**Train with RL frameworks:**
```bash
python scripts/rsl_rl/train.py --task=<TASK_NAME> --num_envs=4096
python scripts/rl_games/train.py --task=<TASK_NAME>
python scripts/skrl/train.py --task=<TASK_NAME>
python scripts/sb3/train.py --task=<TASK_NAME>
```

**Play/inference:**
```bash
python scripts/rsl_rl/play.py --task=<TASK_NAME> --checkpoint=<CHECKPOINT_PATH>
```

**Code formatting:**
```bash
pip install pre-commit
pre-commit run --all-files
```

## Registered Environments

**Direct RL tasks** (`source/robotis_sh5/robotis_sh5/tasks/direct/`):
- `Template-Robotis-Sh5-Direct-v0` — single-agent direct RL template
- `Template-Robotis-Sh5-Marl-Direct-v0` — multi-agent variant
- `Robotis-Sh5-Grasp-Direct-v0` — grasp task (in development)

**Manager-based tasks** (`source/robotis_sh5/robotis_sh5/tasks/manager_based/`):
- `Robotis-SH5-Pick-and-Place-v0` — primary dexterous manipulation task
- `Robotis-SH5-Reach-v0` — bimanual reaching
- `Robotis-SH5-Navigation-v0` — wheeled base navigation

## Architecture

### Source Layout

```
source/robotis_sh5/robotis_sh5/
├── tasks/
│   ├── direct/          # DirectRLEnv subclasses
│   └── manager_based/   # ManagerBasedRLEnv with modular managers
├── assets/              # Python robot asset configs
└── data/
    ├── robots/FFW/      # USD robot models (FFW_SH5*.usd)
    ├── object/          # YCB object meshes with textures
    └── raw/, processed/ # Dataset storage

retargeting/             # MANO hand pose retargeting pipeline
scripts/                 # Training/evaluation entry points per RL framework
```

### Two Environment Paradigms

**Direct RL** (`direct/*/`): Each task subclasses `DirectRLEnv`. The env file contains the full `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx` logic. Config is a `*EnvCfg` dataclass. Use for custom reward/obs pipelines that don't fit manager decomposition.

**Manager-Based** (`manager_based/*/`): Uses Isaac Lab's modular manager system. The entry point is always `isaaclab.envs:ManagerBasedRLEnv`. Config hierarchy:
- `SceneCfg` — robot, objects, sensors with prim paths and actuator groups
- `CommandsCfg` — command generators (e.g., `UniformPoseCommandCfg` for EE targets)
- `ActionsCfg` — action terms (e.g., `JointPositionActionCfg`, custom `JointPositionLowPassAction`)
- `ObservationsCfg` — groups of `ObservationTermCfg` entries that concatenate into policy input
- `EventCfg` — reset and domain randomization events
- `RewardsCfg` — weighted sum of `RewardTermCfg` entries
- `TerminationsCfg` — done conditions
- `CurriculumCfg` — curriculum schedule via direct config address modification

### Manager-Based MDP Module Structure

Each manager-based task has an `mdp/` package with these modules (all exported via `__init__.py` alongside `from isaaclab.envs.mdp import *`):
- `actions.py` — custom action classes (e.g., low-pass filtered joint position)
- `commands.py` — custom command generators
- `observations.py` — custom observation functions; may include visualization helpers
- `rewards.py` — reward functions keyed by `(env, ...)` signature
- `terminations.py` — done condition functions
- `events.py` — reset/randomization event functions
- `curriculum.py` — curriculum schedule functions (e.g., `fade_in_reward_weight`)
- `utils.py` — shared helpers (e.g., `get_virtual_link_poses`, `compute_hand_pos_error`)

### Key Config Conventions

**`@configclass` and `__post_init__`**: All configs use `@configclass` (Isaac Lab's dataclass wrapper). Dynamic setup (e.g., filling `MISSING` body names from the robot model, enabling/disabling observation normalization) goes in `__post_init__()`.

**`MISSING` fields**: Fields that depend on runtime context (e.g., EE body name) are set to `MISSING` in the base class and filled in `__post_init__()`.

**Curriculum via config address**: Curriculum terms use `mdp.modify_env_param` with a dot-separated `address` string pointing into the live config (e.g., `"reward_manager.cfg.action_rate.weight"`).

**Actuator groups use regex**: Joint names in `ImplicitActuatorCfg` use regex expressions like `["arm_l_joint[1-7]", "arm_r_joint[1-7]"]`.

**`SceneEntityCfg` for asset selection**: Observation and reward functions receive an `asset_cfg: SceneEntityCfg("robot", joint_names=[...])` parameter that selects specific joints from the scene.

### Environment Registration

Each task's `__init__.py` calls `gym.register()` with:
- `env_cfg_entry_point` → config class
- `rsl_rl_cfg_entry_point` → `agents/rsl_rl_ppo_cfg.py:PPORunnerCfg`
- `rl_games_cfg_entry_point` → `agents/rl_games_ppo_cfg.yaml`
- `skrl_cfg_entry_point` → `agents/skrl_ppo_cfg.yaml`
- `sb3_cfg_entry_point` → `agents/sb3_ppo_cfg.yaml`

### Robot Platform: FFW-SH5

**Joint groups and DOF:**
- Base: 6 swerve joints (`*_wheel_drive`, `*_wheel_steer` for left/right/rear)
- Lift: 1 DOF
- Arms: 7×2 = 14 DOF (`arm_l/r_joint[1-7]`), actuated by DY_80/DY_70/DP-42 motors
- Hands: 20×2 = 40 DOF (`finger_l/r_joint[1-20]`), low stiffness (2.0), low effort (1kN)
- Head: 2 DOF

**USD model variants:**
- `FFW_SH5.usd` — full fidelity (evaluation/visualization)
- `FFW_SH5_simplified.usd` — lighter collision geometry (standard training)
- `FFW_SH5_simplified_dex.usd` — enhanced hand model for dexterous manipulation tasks

### Hand Retargeting Pipeline

`retargeting/` processes DexYCB dataset recordings into robot-ready `.npy` trajectories:
- `dataset.py` — `DexYCBVideoDataset`; defines `YCB_CLASSES` and 21-point MANO joint layout
- `mano_single_object.py` — `WorldTrajectoryGenerator`: loads subject/object data, applies MANO to get hand geometry, transforms camera→world coordinates, detects motion onset, saves per-frame `joints_world`, `wrist_pos/quat_world`, `obj_pos/quat_world`
- `mano_trajectories/` — pre-computed `.npy` files organized by YCB object ID
- `manopth/` — PyTorch MANO layer implementation

## Code Style

- **Ruff** for linting/formatting: line-length 120, Python 3.10 target
- **Pyright** in basic mode for type checking
- Import order: `omniverse-extensions` → `isaaclab` → `isaaclab-*` → first-party
- `__init__.py` files may have unused imports (F401 ignored)

## Key Simulation Parameters

- Physics timestep: `dt = 1/120s`
- Default parallel envs: 4096 (GPU-accelerated)
- Robot models: `FFW_SH5_simplified.usd` for training, `FFW_SH5.usd` for full fidelity
