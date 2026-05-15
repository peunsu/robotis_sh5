# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab extension for the FFW-SH5 robotic platform implementing RL-based dexterous grasping.
The primary task (`Robotis-Sh5-Grasp-Direct-v0`) trains a policy to retarget human hand motion
from the OakInk or HO-Cap dataset onto the robot's right hand and arm.

## Commands

**Installation** (requires Isaac Lab Python environment):
```bash
python -m pip install -e source/robotis_sh5
```

**Verify environment registration:**
```bash
python scripts/list_envs.py
```

**Test with dummy agents (sanity-check obs/reward shapes before training):**
```bash
python scripts/zero_agent.py --task=Robotis-Sh5-Grasp-Direct-v0
python scripts/random_agent.py --task=Robotis-Sh5-Grasp-Direct-v0
```

**Train (SKRL is primary framework):**
```bash
python scripts/skrl/train.py --task=Robotis-Sh5-Grasp-Direct-v0 --num_envs=2048
python scripts/skrl/train.py --task=Robotis-Sh5-Grasp-Direct-v0 --num_envs=2048 --headless \
    --dataset oakink --object_id C11001 \
    --trajectory_task C11001-0001-0007 --trajectory_data_id 0 \
    --timesteps 10000 --checkpoint <PRETRAIN_CKPT>
```

**Rollout / evaluation:**
```bash
python scripts/skrl/rollout.py \
    --task Robotis-Sh5-Grasp-Direct-v0 \
    --checkpoint <CKPT> --output_dir <DIR> --n_rollouts 32 --headless \
    --dataset oakink --object_id C11001 \
    --trajectory_task C11001-0001-0007 --trajectory_data_id 0
```

**Full benchmark pipeline (train all sequences then evaluate):**
```bash
bash scripts/benchmark/train_sequences.sh       # set DATASET, SEQUENCES at top of file
FORCE=1 bash scripts/benchmark/train_sequences.sh
bash scripts/benchmark/evaluate_sequences.sh
bash scripts/benchmark/evaluate.bash source/robotis_sh5/data/processed/oakink
```

**Dataset preprocessing:**
```bash
python scripts/process_dataset/oakink.py
python scripts/process_dataset/hocap.py
isaaclab.sh -p scripts/process_dataset/convert_oakink_to_usd.py   # requires Isaac Lab env
python scripts/process_dataset/compute_frame0_ik.py --dataset oakink  # or hocap
```

**Code formatting:**
```bash
pre-commit run --all-files
```

## Registered Environments

| Task ID | Description |
|---|---|
| `Robotis-Sh5-Grasp-Direct-v0` | Full grasping task: 29D action (fingers+arm+lift+mass), 280D obs |
| `Robotis-Sh5-Grasp-Pretrain-Direct-v0` | Pretraining without physics object: 28D action, 189D obs |
| `Template-Robotis-Sh5-Direct-v0` | Single-agent direct RL template |
| `Template-Robotis-Sh5-Marl-Direct-v0` | Multi-agent variant |
| `Robotis-SH5-Pick-and-Place-v0` | Manager-based dexterous manipulation |

## Architecture

### Source Layout

```
source/robotis_sh5/robotis_sh5/
├── tasks/direct/robotis_sh5_grasp/   ← primary task (see below)
├── tasks/direct/robotis_sh5*/        ← template tasks
├── tasks/manager_based/              ← pick-and-place, reach, navigation
└── data/
    ├── robots/FFW/                   ← USD robot models
    └── processed/{oakink,hocap}/     ← SPIDER-format trajectories + assets

scripts/
├── skrl/train.py                     ← primary training entry point
├── skrl/rollout.py                   ← evaluation / metrics generation
├── benchmark/train_sequences.sh      ← per-sequence pipeline (IK → pretrain → train)
├── benchmark/evaluate_sequences.sh   ← per-sequence rollout + aggregation
├── benchmark/evaluate.bash           ← CSV aggregation only
└── process_dataset/                  ← data conversion scripts
```

### Grasp Task: Two-Phase Training

**Phase 1 — Pretrain** (`Robotis-Sh5-Grasp-Pretrain-Direct-v0`):
No physics object in the scene. Policy learns hand/wrist keypoint tracking purely from kinematic
reference data. Saves `pretrain.pt` under `data/processed/<dataset>/ffw_sh5/right/<task>/<id>/`.

**Phase 2 — Train** (`Robotis-Sh5-Grasp-Direct-v0`):
Full grasping with a physics object. Loads `pretrain.pt` via `--checkpoint`. Saves `agent.pt`
and `task_info.json` to the same directory.

### Grasp Environment Internals (`robotis_sh5_grasp_env.py`)

`RobotisSh5GraspEnv.__init__` call order:
1. `_load_reference_trajectories(cfg)` — loads `trajectory_keypoints.npz`, computes normalization
   (table placement, XY canonicalization toward robot approach direction), pads to max length
2. `_apply_object_mass_from_json(cfg)` — overrides `cfg.object_mass_min/max` from
   `data/processed/<dataset>/object_mass.json` if the object ID is present
3. `_build_object_cfg(cfg)` — resolves dataset-aware USD path:
   `data/processed/<dataset>/assets/objects/<object_id>/visual.usd`
4. `super().__init__()` — Isaac Lab scene setup
5. `_post_init_buffers()` — allocates GPU tensors for state cache, adaptive sampling weights, EMA

Key design patterns:
- **Mass-as-action** (MassDexMimic): action dim 29 = 28 joints + 1 mass. The mass dim maps
  `[-1,1] → [object_mass_min, object_mass_max]` and is applied at episode reset.
- **Adaptive rollout sampling**: failure-weighted start-frame sampling; EMA tracks per-frame
  failure counts; `adaptive_alpha=0.001`, `adaptive_uniform_ratio=0.1`.
- **Frame-0 arm IK**: precomputed wrist pose for frame 0 is loaded from
  `frame0_arm_joint_pos.npy` in each trajectory directory (generated by `compute_frame0_ik.py`).
- **EMA action smoothing**: `action_smoothing=0.7` by default.

### train.py Patches (applied after runner construction)

Two monkey-patches are applied for the grasp train task only:

**`_patch_mass_policy`**: Replaces the skrl Gaussian policy with `MassGaussianModel`
(`agents/mass_gaussian_model.py`), which uses a separate head with a lower initial mean for the
mass dimension.

**`_patch_entropy_flip`** (GR-faithful entropy scheduling):
- Adds `sigma_grad_flg` tensor to skrl rollout memory.
- Patches `record_transition` to write `is_reached_end` (whether the episode reached end-of-trajectory)
  into `sigma_grad_flg` at each rollout step.
- Replaces `agent._update` entirely with a custom function that reads `sampled_sigma[0].item()`
  per mini-batch and applies:
  `entropy_scale = base_entropy_coef * (1 - sw) - 0.002 * sw`

### PPO Hyperparameters (skrl_ppo_cfg.yaml)

| Parameter | Value | Note |
|---|---|---|
| `rollouts` | 8 | horizon length (= 1 PPO update per 8 timesteps) |
| `learning_epochs` | 5 | passes over the rollout buffer per update |
| `mini_batches` | 4 | buffer of 8×num_envs samples split into 4 |
| `discount_factor` | 0.96 | |
| `learning_rate` | 3e-4 | KLAdaptiveLR scheduler, kl_threshold=0.016 |
| `entropy_loss_scale` | 0.004 | base value; modified per mini-batch by `_patch_entropy_flip` |

Epoch relationship: `10000 timesteps / 8 rollouts = 1250 PPO updates`.

### Data Directory Layout

```
data/processed/<dataset>/              # oakink or hocap
├── object_mass.json                   # {object_id: [min_kg, max_kg]}
├── mano/right/<task>/<id>/
│   ├── trajectory_keypoints.npz       # wrist, fingertip, obj poses + mano keypoints
│   └── frame0_arm_joint_pos.npy       # precomputed IK for arm init
└── ffw_sh5/right/<task>/<id>/
    ├── pretrain.pt
    ├── agent.pt
    ├── task_info.json                  # training metadata (checkpoint path, timesteps, seed, …)
    └── evaluation_ep_le_<N>/metrics.csv
```

### Robot Platform: FFW-SH5

USD variants (in `data/robots/FFW/`):
- `FFW_SH5_simplified_dex.usd` — used for grasp tasks (enhanced hand collision)
- `FFW_SH5_simplified.usd` — standard training
- `FFW_SH5.usd` — full fidelity

Joint groups:
- Base: 6 swerve (`*_wheel_drive/steer`)
- Lift: 1 (`lift_joint`)
- Arms: 7×2 (`arm_l/r_joint[1-7]`), actuated by DY_80/DY_70/DP-42
- Hands: 20×2 (`finger_l/r_joint[1-20]`), stiffness=20, damping=0.5

The grasp task fixes the robot base (`fix_root_link=True`) and controls only the right side:
20 fingers + 7 arm_r + 1 lift.

## Key Config Fields (`RobotisSh5GraspEnvCfg`)

Fields most likely to need adjustment:

| Field | Default | Purpose |
|---|---|---|
| `dataset` | `"oakink"` | `"oakink"` or `"hocap"` |
| `object_id` | `"C11001"` | must have a converted USD under `assets/objects/` |
| `trajectory_task` | `"C11001-0001-0007"` | mano/right subdirectory name |
| `trajectory_data_id` | `0` | sub-index within trajectory_task |
| `object_mass_json` | oakink path | override to point at a different JSON |
| `object_mass_min/max` | 0.04 / 0.10 | overridden from JSON at init if available |
| `action_smoothing` | `0.7` | EMA coefficient for action smoothing |
| `adaptive_sampling` | `True` | failure-weighted start-frame resampling |
| `termination` | `True` | set False during warm-up / pretrain |
| `debug_vis` | `False` | enables fingertip/wrist marker visualization |

## Code Style

- **Ruff**: line-length 120, Python 3.10 target
- **Pyright** in basic mode
- Import order: omniverse-extensions → isaaclab → isaaclab-* → first-party
- `__init__.py` files may have unused imports (F401 ignored)
- Physics timestep: `dt = 1/120s`, control at 30 Hz (`decimation=4`)
