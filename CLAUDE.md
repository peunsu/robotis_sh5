# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab extension for the FFW-SH5 robotic platform implementing RL-based dexterous grasping.
The primary task (`Robotis-Sh5-Grasp-Direct-v0`) trains a policy to retarget human hand motion
from the OakInk or HO-Cap dataset onto the robot's right hand and arm. A MARL variant
(`Robotis-Sh5-Grasp-Marl-Direct-v0`) decomposes the policy into separate arm and hand agents
trained via HAPPO (Kuba et al. 2022): single shared critic + team reward + sequential actor
updates with recursive advantage M and hand → arm sequential conditioning.

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

**Train — single-agent PPO:**
```bash
python scripts/skrl/train.py --task=Robotis-Sh5-Grasp-Direct-v0 --num_envs=2048
python scripts/skrl/train.py --task=Robotis-Sh5-Grasp-Direct-v0 --num_envs=2048 --headless \
    --dataset oakink --object_id C11001 \
    --trajectory_task C11001-0001-0007 --trajectory_data_id 0 \
    --timesteps 10000 --checkpoint <PRETRAIN_CKPT>
```

**Train — MARL (HAPPO):**
```bash
python scripts/skrl/train_marl.py --task=Robotis-Sh5-Grasp-Marl-Direct-v0 --num_envs=2048 --headless \
    --dataset oakink --object_id C11001 \
    --trajectory_task C11001-0001-0007 --trajectory_data_id 0 \
    --checkpoint <PRETRAIN_CKPT>
```
- The MARL task is gated: `--task` must contain `"Marl"` or `train_marl.py` exits.
- Use `--freeze-arm-from <CKPT>` / `--freeze-hand-from <CKPT>` to load one sub-agent
  from an external ckpt and freeze its parameters (curriculum / sim-to-real workflows).

**Rollout / evaluation:**
```bash
# Single-agent
python scripts/skrl/rollout.py --task Robotis-Sh5-Grasp-Direct-v0 --checkpoint <CKPT> \
    --output_dir <DIR> --n_rollouts 32 --headless \
    --dataset oakink --object_id C11001 \
    --trajectory_task C11001-0001-0007 --trajectory_data_id 0

# MARL
python scripts/skrl/rollout_marl.py --task Robotis-Sh5-Grasp-Marl-Direct-v0 --checkpoint <CKPT> \
    --output_dir <DIR> --n_rollouts 32 --headless \
    --dataset oakink --object_id C11001 \
    --trajectory_task C11001-0001-0007 --trajectory_data_id 0
```

**Full benchmark pipeline (per-sequence pretrain → train → evaluate):**
```bash
# Single-agent (results: data/processed/<ds>/ffw_sh5/)
bash scripts/benchmark/oakink/train_sequences.sh
bash scripts/benchmark/oakink/evaluate_sequences.sh

# MARL (results: data/processed/<ds>/ffw_sh5_marl/ — separate tree, evaluate.bash
# picks up both ffw_sh5/ and ffw_sh5_marl/ for side-by-side comparison)
bash scripts/benchmark/oakink/train_sequences_marl.sh
bash scripts/benchmark/oakink/evaluate_sequences_marl.sh

# Force re-run all steps
FORCE=1 bash scripts/benchmark/oakink/train_sequences.sh

# Standalone aggregator (dataset-agnostic — produces <model>_method{1,2,3}.csv per model dir)
bash scripts/benchmark/evaluate.bash source/robotis_sh5/data/processed/oakink
```

HO-Cap mirrors OakInk: `scripts/benchmark/hocap/train_sequences{,_marl}.sh`, etc.

**Dataset preprocessing:**
```bash
python scripts/process_dataset/oakink.py
python scripts/process_dataset/hocap.py
isaaclab.sh -p scripts/process_dataset/convert_obj_to_usd.py [--dataset hocap]   # requires Isaac Lab env
python scripts/process_dataset/compute_frame0_ik.py --dataset oakink   # or hocap
```

**Code formatting:**
```bash
pre-commit run --all-files
```

## Registered Environments

| Task ID | Action | Obs | Description |
|---|---|---|---|
| `Robotis-Sh5-Grasp-Direct-v0` | 28D | 279D | Full single-agent grasping (fingers+arm+mass) |
| `Robotis-Sh5-Grasp-Pretrain-Direct-v0` | 28D | 279D | Kinematic-only pretrain (no physics object) |
| `Robotis-Sh5-Grasp-Marl-Direct-v0` | dict{arm:7, hand:20} | dict{arm:60, hand:276}, state:279 | HAPPO (single shared critic + team reward + sequential actor updates with recursive M + hand→arm forward conditioning) |
| `Robotis-Sh5-Grasp-Marl-Pretrain-Direct-v0` | same shapes | same shapes | MARL pretrain (no object) — ckpt shape-compatible with train |
| `Template-Robotis-Sh5-Direct-v0` | — | — | Single-agent direct RL template |
| `Template-Robotis-Sh5-Marl-Direct-v0` | — | — | Multi-agent variant |
| `Robotis-SH5-Pick-and-Place-v0` | — | — | Manager-based dexterous manipulation |

## Architecture

### Source Layout

```
source/robotis_sh5/robotis_sh5/
├── tasks/direct/robotis_sh5_grasp/   ← primary task: single-agent + MARL
│   ├── robotis_sh5_grasp_env.py / _cfg.py                        ← single-agent train
│   ├── robotis_sh5_grasp_pretrain_env.py / _cfg.py               ← single-agent pretrain
│   ├── robotis_sh5_grasp_marl_env.py / _cfg.py                   ← MARL train (DirectMARLEnv)
│   ├── robotis_sh5_grasp_marl_pretrain_env.py / _cfg.py          ← MARL pretrain
│   └── agents/
│       ├── mass_gaussian_model.py        ← single-agent: mass-as-action via MassDexMimicPolicy
│       ├── mass_distribution.py          ← MARL: standalone mu/log_std nn.Parameters + PPO surrogate
│       ├── skrl_ppo_cfg.yaml / _pretrain ← single-agent PPO config
│       └── skrl_mappo_cfg.yaml / _pretrain ← MARL HAPPO config (1024×1024×512×512 ELU)
│                                              (yaml uses skrl's MAPPO class; HAPPO behavior
│                                              is layered on via patches in train_marl.py)
├── tasks/direct/robotis_sh5*/        ← template tasks
├── tasks/manager_based/              ← pick-and-place, reach, navigation
└── data/
    ├── robots/FFW/                   ← USD robot models
    └── processed/{oakink,hocap}/
        ├── mano/right/...            ← reference trajectories (SPIDER format)
        ├── object_mass.json          ← per-object [min_kg, max_kg]
        ├── ffw_sh5/right/...         ← single-agent checkpoints + metrics
        └── ffw_sh5_marl/right/...    ← MARL checkpoints + metrics (separate tree)

scripts/
├── skrl/train.py        / rollout.py      / play.py       ← single-agent
├── skrl/train_marl.py   / rollout_marl.py / play_marl.py  ← MARL
└── benchmark/{oakink,hocap}/
    ├── train_sequences.sh        / evaluate_sequences.sh        ← single-agent pipeline
    └── train_sequences_marl.sh   / evaluate_sequences_marl.sh   ← MARL pipeline
```

### Grasp Task: Two-Phase Training

**Phase 1 — Pretrain**: no physics object; policy learns hand/wrist keypoint tracking purely from
kinematic reference data. Saves `pretrain.pt`.

**Phase 2 — Train**: full grasping with physics object. Loads `pretrain.pt` via `--checkpoint`.
Saves `agent.pt` and `task_info.json`.

Both phases share obs/action dimensions so checkpoints transfer.

### Single-Agent Internals (`robotis_sh5_grasp_env.py`)

`__init__` order: `_load_reference_trajectories` → `_apply_object_mass_from_json` (overrides
`object_mass_min/max` from JSON) → `_build_object_cfg` (dataset-aware USD path) → `super().__init__`
→ `_post_init_buffers` (state cache, adaptive sampling weights, EMA buffers).

Key design patterns:
- **Mass-as-action (MassDexMimic)**: action dim 28 = 27 joints + 1 mass. Mass action's
  `[-1,1] → [object_mass_min, object_mass_max]` mapping is applied via `set_masses()` at
  episode reset. Mass is held constant within an episode.
- **Adaptive rollout sampling**: failure-weighted start-frame sampling. EMA tracks per-frame
  failure counts (`adaptive_alpha=0.001`, `adaptive_uniform_ratio=0.1`).
- **State cache (physical curriculum)**: `_save_state_cache` stores per-frame "best" sim state
  (97D = reward + obj pose/vel + 28D joint state + 27D smoothed action). On reset, cache-hit
  envs restore from the saved state, cache-miss envs use default pose + frame-0 IK.
- **Frame-0 arm IK**: precomputed wrist pose for frame 0 loaded from `frame0_arm_joint_pos.npy`.
- **EMA action smoothing** (TJ/rl_games convention, `alpha = new action weight`):
  `smoothed = alpha * raw + (1 - alpha) * prev_smoothed`. Default `0.3` for single-agent,
  `0.4` for MARL.

### train.py (single-agent) Patches

Applied after Runner construction for grasp train task only:

**`_patch_mass_policy`**: Replaces skrl Gaussian policy with `MassDexMimicPolicy` —
separate `mu_mass` / `log_std_mass` nn.Parameters, mass dim assembled from these.
mu_mass gets 33.333× LR via separate optimizer group. Per-env caches of episode-start μ/σ
make the PPO ratio non-trivial for mass (clipping restrains σ growth).

**`_patch_entropy_flip`** (GR-faithful entropy scheduling):
- Adds `sigma_grad_flg` tensor to skrl rollout memory.
- Patches `record_transition` to write `is_reached_end` into `sigma_grad_flg`.
- Replaces `agent._update` to apply per-mini-batch: `entropy_scale = base * (1 - sw) - 0.002 * sw`
  where `sw = sampled_sigma[0].item()`.

### MARL Architecture (`robotis_sh5_grasp_marl_env.py` + `train_marl.py`)

HAPPO (Kuba et al. 2022, "Trust Region Policy Optimisation in Multi-Agent RL", Algorithm 4)
implemented as a layer on top of skrl's MAPPO class via monkey-patches:

- **Hand agent** sees full single-agent grasping context (276D obs minus mass+lift):
  21 MANO kpts (world) + palm state + fingertip vel + full 27D joint state + object state +
  reference deltas + `future_contact` + 27D combined prev_action + fingertip forces.
- **Arm agent** sees a minimal "wrist-pose follower" obs (60D):
  jp_arm + jv_arm + wrist pose (env-pos + world-quat) + wrist linvel + wrist angvel +
  delta_wrist_pos + delta_wrist_rot + prev_arm_action +
  **current_hand_action slot at [40:60]** (filled by forward sequential conditioning).
- **Single shared centralized critic** V(s_global) with explicit non-redundant 279D state
  from `_get_states()` = hand_obs (276D) + delta_wrist_rot (3D). State is NOT auto-flattened
  from the obs dict — avoids duplication of jp_arm / wrist_quat / prev_arm_action across agents.

Four monkey-patches are applied in train_marl.py after Runner construction (in this order):

**`_setup_happo_optimizers`** — HAPPO optimizer layout (Algorithm 4):
- `agent.values[uid]` for ALL uid → same shared_value instance (anchored under "arm" in
  checkpoint slot for skrl save/load compatibility).
- Dedicated `agent._critic_optimizer = Adam(shared_value.parameters(), lr)`, separate from
  any actor optimizer. Saved under `checkpoint_modules["arm"]["critic_optimizer"]`.
- Per-actor `agent.optimizers[uid] = Adam(policies[uid].parameters(), lr)` — policy params only.
- KLAdaptiveLR scheduler attached per actor; critic LR is constant (HAPPO convention).

**`_patch_sequential_act`** — hand → arm sequential FORWARD conditioning:
1. Hand acts on its raw obs.
2. Hand action is `.detach()`'d and injected into arm obs slot [40:60].
3. Arm acts on the injected obs.
4. `record_transition` is patched to store `states["arm"]` as the **injected** version
   (PPO ratio consistency between rollout-time `log_pi_old` and training-time `log_pi_new`).
5. Shared state is NOT affected (computed by env's `_get_states()`, independent of obs).
6. Manual env-info logging runs here — bypasses skrl trainer's hard-coded `"Info / "` prefix
   so env's keys (`"Error / X"`, `"Reward / X"`, `"Mass / X"`, `"Curriculum / X"`) become
   separate Tensorboard tabs (the trainer's auto-log is disabled by
   `agent_cfg["trainer"]["environment_info"] = "__disabled__"`).

**`_patch_happo_update`** — replaces skrl's `_update` with HAPPO Algorithm 4:
1. **Single GAE compute** on team reward + shared critic V(s_global). Â is normalized
   (mean=0, std=1). Stored as initial M_{1:1}.
2. **Sequential actor update** in fixed `HAPPO_ACTOR_ORDER = ["hand", "arm"]` (matches
   forward order — no random permutation since 2 agents are asymmetric):
   - For each agent m in order: run `learning_epochs × mini_batches` PPO-clip updates
     on actor m using M_{1:m} as the advantage. KL-adaptive LR step per actor.
   - **Recursive advantage update**: after actor m's inner loop, compute
     `ratio_full_m = exp(log π^m_new(a^m|o^m) - log π^m_old(a^m|o^m))` over ALL stored
     transitions, then `M_{1:m+1} = ratio_full_m · M_{1:m}`. (Skip after the last agent.)
3. **Critic update** (separate optimization step using `_critic_optimizer`):
   `learning_epochs × mini_batches` of MSE(V_φ(s_global), R̂_t) with optional
   value-clip and grad-norm clip.
4. Restores `memory["arm"].advantages = Â` at end so mass-in-loop reads the pristine
   normalized advantage estimate.

**`_setup_mass_in_loop`** — mass-in-the-loop PPO update (mirrors single-agent MassDexMimic
behavior but keeps mass OUT of any agent's action space):
- `MassDistribution` (`agents/mass_distribution.py`) holds `mu_mass`, `log_std_mass` as
  standalone `nn.Parameter`s + per-env `current_mass_action`, `current_log_prob_old` buffers.
- `env._reset_idx` calls `mass_dist.sample_for_envs(env_ids)` — samples action,
  caches log_prob under sample-time μ/σ, applies mass to PhysX via `set_masses()`.
- `env._pre_physics_step` snapshots `(action, log_prob_old)` into env buffers BEFORE
  `_reset_idx` runs at end-of-step (preserves the values that were USED during the step).
- The patched `record_transition` appends snapshots to per-rollout buffers on the agent.
- The patched `_update` runs a mass mini-PPO loop after the main PPO update: reads per-step
  advantages from memory (already normalized) and runs `learning_epochs × mini_batches` of
  PPO surrogate updates with `ratio_clip` clipping. Optimizers: `mu_mass` at
  `base_lr × mass_lr_scale` (33.333), `log_std_mass` at `base_lr`. Mass is excluded from
  entropy regularization (matches single-agent).

**Partial checkpoint loader (`_load_marl_partial_checkpoint`)**: only policy/value weights are
loaded from pretrain.pt — optimizer / state_preprocessor / shared_state_preprocessor /
value_preprocessor are deliberately reset to avoid stale normalization stats from the
no-object pretrain distribution. `log_std_parameter` is also skipped (TJ-style σ reset).

### PPO / HAPPO Hyperparameters

| Parameter | Single-agent PPO | MARL HAPPO | Note |
|---|---|---|---|
| `rollouts` | 8 | 8 | horizon length |
| `learning_epochs` | 5 | 5 | passes per update |
| `mini_batches` | 4 | 4 | |
| `discount_factor` | 0.96 | 0.96 | |
| `learning_rate` | 3e-4 | 3e-4 | KLAdaptiveLR, kl_threshold=0.016 |
| `entropy_loss_scale` | 0.004 (modified by entropy_flip) | 0.004 | |
| `action_smoothing` (env cfg) | 0.3 | **0.4** | higher = more responsive (less wrist tremor) |
| `rew_arm_action_rate` (env cfg) | -0.05 | **-0.01** | weakened to reduce overcorrection |

Epoch relationship: `10000 timesteps / 8 rollouts = 1250 PPO updates`.

### Tensorboard Groups

Logged keys are grouped by top-level prefix (separated by ` / `):
- **`Error /`** — tracking metrics (kpts_mean, wrist_pos, wrist_rot, obj_pos, obj_rot, ft_mean)
- **`Reward /`** — decomposed components (alive, kpts, obj_pos/rot, fingertip, force, regs, total)
- **`Mass /`** — mu_action, std_action, mu_kg, std_kg, loss, approx_kl, n_samples (MARL only)
- **`Curriculum /`** — reached_frame, warmup_ratio, success_rate
- **`Loss /`, `Policy /`, `Reward /` (Instantaneous/Total)** — skrl built-ins

For MARL, the env emits these keys directly. skrl's automatic `"Info / "` prefix is disabled
by setting `agent_cfg["trainer"]["environment_info"] = "__disabled__"`. The patched
`record_transition` calls `agent.track_data(k, v)` with keys as-is.

### Data Directory Layout

```
data/processed/<dataset>/              # oakink or hocap
├── object_mass.json                   # {object_id: [min_kg, max_kg]}
├── mano/right/<task>/<id>/
│   ├── trajectory_keypoints.npz       # wrist, fingertip, obj poses + mano keypoints
│   └── frame0_arm_joint_pos.npy       # precomputed IK for arm init
├── ffw_sh5/right/<task>/<id>/         # single-agent checkpoints
│   ├── pretrain.pt
│   ├── agent.pt
│   ├── task_info.json
│   └── evaluation_ep_le_<N>/metrics.csv
└── ffw_sh5_marl/right/<task>/<id>/    # MARL checkpoints (parallel tree)
    └── (same structure)
```

`evaluate.bash` automatically iterates over all `ffw_sh5*/` subdirectories under `data/processed/<dataset>/`
and produces per-model aggregate CSVs (`ffw_sh5_method{1,2,3}.csv`, `ffw_sh5_marl_method{1,2,3}.csv`).

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
20 fingers + 7 arm_r + 1 lift. The lift joint is held at `cfg.fixed_lift_target=0.0` by the PD
controller every step (not part of the action vector in either single-agent or MARL).

## Key Config Fields

### Common (`RobotisSh5GraspEnvCfg` and MARL variants)

| Field | Default | Purpose |
|---|---|---|
| `dataset` | `"oakink"` | `"oakink"` or `"hocap"` |
| `object_id` | `"C11001"` | must have a converted USD under `assets/objects/` |
| `trajectory_task` | `"C11001-0001-0007"` | mano/right subdirectory name |
| `trajectory_data_id` | `0` | sub-index within trajectory_task |
| `object_mass_json` | "" (defaults to dataset path) | per-object JSON override |
| `adaptive_sampling` | `True` | failure-weighted start-frame resampling |
| `termination` | `True` | set False during evaluation/warm-up |
| `debug_vis` | `False`/`True` | enables fingertip/wrist marker visualization |

### MARL-only (`RobotisSh5GraspMarlEnvCfg`)

| Field | Default | Purpose |
|---|---|---|
| `enable_mass_in_loop` | `True` | sample mass per-episode from learned `MassDistribution` |
| `mass_mu_init` | `-0.25` | initial μ_mass (≈ 0.4 × max_mass after affine map) |
| `mass_log_std_init` | `-1.25` | initial log σ → σ ≈ 0.286 (action space) |
| `mass_lr_scale` | `33.333` | μ_mass LR multiplier vs base LR; log_std_mass uses base LR |
| `object_mass_min/max` | from JSON (or 0.04/0.20) | mass-in-loop sampling range |

## Important Gotchas

- **DirectMARLEnv has no `reset_terminated`** (it's a DirectRLEnv attribute). MARL env's
  `_reset_idx` reads `self.terminated_dict["arm"]` for adaptive-sampling failure count updates.
- **`self.actions` in MARL** is a **per-agent dict** (initialized by DirectMARLEnv.__init__),
  not a tensor. The combined 27D action is kept in `self._joint_actions` to avoid clobbering
  the parent's dict.
- **MARL `prev_arm_action` / `prev_hand_action`** are updated at the END of `_get_observations`
  (matches single-agent semantics — `arm_action_rate` in `_get_rewards` needs the actual
  previous-step action, not the just-decided one).
- **delta_wrist_rot quaternion sign**: canonicalized via `where(q_err.w < 0, -q_err, q_err)`
  to avoid double-cover discontinuity in `delta_wrist_rot` obs (was causing wrist tremor before
  the fix; single-agent's `quat_error_magnitude` is sign-invariant so single-agent didn't show this).

## Code Style

- **Ruff**: line-length 120, Python 3.10 target
- **Pyright** in basic mode
- Import order: omniverse-extensions → isaaclab → isaaclab-* → first-party
- `__init__.py` files may have unused imports (F401 ignored)
- Physics timestep: `dt = 1/120s`, control at 30 Hz (`decimation=4`)
