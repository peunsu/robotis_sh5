# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab extension for the FFW-SH5 robotic platform implementing RL-based dexterous grasping.
The primary task (`Robotis-Sh5-Grasp-Direct-v0`) trains a policy to retarget human hand motion
from the OakInk or HO-Cap dataset onto the robot's right hand and arm. A MARL variant
(`Robotis-Sh5-Grasp-Marl-Direct-v0`) decomposes the policy into separate arm and hand agents
trained via HAPPO (Kuba et al. 2022): single shared critic + team reward + sequential actor
updates with recursive advantage M and hand → arm sequential conditioning.

A **Shadow Hand variant** (`Robotis-Shadow-Grasp-Direct-v0`) replaces FFW-SH5's native 20-DOF
right hand with a Shadow Dexterous Hand (24 joints, 18 actuated) mounted on the same arm. Joint /
body / reference-keypoint conventions follow TJ's `gr` reference (`/home/peunsu/workspaceTJ/gr`).
The variant uses a separate set of data assets (`FFW_SH5_shadow_instanced.usd`, `arm_joint_pos_shadow.npy`,
`arm_keypoints_shadow.npz`) so it can coexist with the native sh5 pipeline.

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

**Train — Shadow Hand variant:**
```bash
# Generate Shadow-Hand-specific arm references first (writes _shadow-suffixed files)
python scripts/process_dataset/process_arm_pipeline.py --dataset hocap --robot shadow --overwrite

# Pretrain → train
python scripts/skrl/train.py --task=Robotis-Shadow-Grasp-Pretrain-Direct-v0 --num_envs=4096 --headless \
    --dataset hocap --object_id G10_1 --trajectory_task subject_1-20231025_170231-G10_1
python scripts/skrl/train.py --task=Robotis-Shadow-Grasp-Direct-v0 --num_envs=2048 --headless \
    --dataset hocap --object_id G10_1 --trajectory_task subject_1-20231025_170231-G10_1 \
    --checkpoint <PRETRAIN_CKPT>

# Per-sequence benchmark (parallel to sh5; results: data/processed/<ds>/ffw_shadow/)
VIDEO=1 bash scripts/benchmark/hocap/train_sequences_shadow.sh
```

**Train / evaluate — Shadow Hand RSI warm-start variant:**
```bash
# Same data assets as the shadow variant; results land in a SEPARATE ffw_shadow_rsi tree.
# Pretrain saves pretrain_state_cache.npz next to pretrain.pt; train loads it as a sibling.
bash scripts/benchmark/hocap/train_sequences_shadow_rsi.sh
bash scripts/benchmark/hocap/evaluate_sequences_shadow_rsi.sh   # → ffw_shadow_rsi_method{1,2,3}.csv

# Manual single sequence (pretrain → train):
python scripts/skrl/train.py --task=Robotis-Shadow-Grasp-Rsi-Pretrain-Direct-v0 --num_envs=4096 --headless \
    --dataset hocap --object_id G10_1 --trajectory_task subject_1-20231025_170231-G10_1
python scripts/skrl/train.py --task=Robotis-Shadow-Grasp-Rsi-Direct-v0 --num_envs=2048 --headless \
    --dataset hocap --object_id G10_1 --trajectory_task subject_1-20231025_170231-G10_1 \
    --checkpoint <PRETRAIN_CKPT>   # sibling pretrain_state_cache.npz is auto-loaded
```

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

# Full-trajectory arm reference pipeline (PRIMARY entry point).
# Canonicalize + VPoser SMPL fit with robot-bone rescaling + per-frame pink IK + mp4 visualization.
# Outputs per trajectory: arm_keypoints.npz, arm_joint_pos.npy (N,7), vposer_ik_video.mp4.
# --robot sh5 (default) targets hx5_d20_right_base; --robot shadow targets a virtual
# robot0_palm frame registered at the Shadow Hand mount offset, and writes *_shadow.npy/.npz/.mp4.
python scripts/process_dataset/process_arm_pipeline.py --dataset hocap --robot {sh5|shadow} --overwrite

# (standalone) Frame-0-only arm IK — pink QP on pinocchio. Not used by env/benchmark anymore;
# kept as a diagnostic / single-frame solver, and provides helpers imported by process_arm_pipeline.
python scripts/process_dataset/compute_frame0_ik_pink.py --dataset oakink --overwrite
```

**Code formatting:**
```bash
pre-commit run --all-files
```

## Registered Environments

| Task ID | Action | Obs | Description |
|---|---|---|---|
| `Robotis-Sh5-Grasp-Direct-v0` | 28D | 285D | Full single-agent grasping (fingers+arm+mass) — 6D rot rep + vel_obs_scale (TJ); mass excluded from obs for pretrain ckpt compat |
| `Robotis-Sh5-Grasp-Pretrain-Direct-v0` | 27D | 285D | Kinematic-only pretrain (no physics object) |
| `Robotis-Sh5-Grasp-Marl-Direct-v0` | dict{arm:7, hand:20} | dict{arm:89, hand:283}, state:286 | HAPPO (single shared critic + team reward + sequential actor updates with recursive M + hand→arm forward conditioning) |
| `Robotis-Sh5-Grasp-Marl-Pretrain-Direct-v0` | same shapes | same shapes | MARL pretrain (no object) — ckpt shape-compatible with train |
| `Robotis-Shadow-Grasp-Direct-v0` | 26D | 291D | Same scheme as sh5 but with Shadow Hand (18 actuated fingers + 7 arm + 1 mass). USD: `FFW_SH5_shadow_instanced.usd` |
| `Robotis-Shadow-Grasp-Pretrain-Direct-v0` | 25D | 291D | Shadow Hand kinematic-only pretrain (no mass) |
| `Robotis-Shadow-Grasp-Rsi-Direct-v0` | 26D | 291D | Shadow Hand + **pretrain-cache RSI warm-start** (copy of shadow grasp; see "Shadow Hand RSI Warm-Start Variant"). Separate `ffw_shadow_rsi` tree |
| `Robotis-Shadow-Grasp-Rsi-Pretrain-Direct-v0` | 25D | 291D | RSI variant pretrain — dumps `pretrain_state_cache.npz` for the train phase to warm-start from |
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
├── tasks/direct/robotis_shadow_grasp/  ← Shadow Hand variant (mirrors sh5 grasp, single-agent only)
│   ├── robotis_shadow_grasp_env.py / _cfg.py             ← train env (action 26D, obs 291D)
│   ├── robotis_shadow_grasp_pretrain_env.py / _cfg.py    ← pretrain env (action 25D, obs 291D)
│   └── agents/                                            ← reuses MassDexMimicPolicy + skrl PPO yaml
├── tasks/direct/robotis_shadow_grasp_rsi/  ← Shadow Hand + pretrain-cache RSI warm-start
│   ├── robotis_shadow_grasp_rsi_env.py / _cfg.py          ← copy of shadow grasp + warm-start logic
│   ├── robotis_shadow_grasp_rsi_pretrain_env.py / _cfg.py ← saves state cache at end of pretrain
│   └── agents/                                            ← own log tree (robotis_shadow_grasp_rsi{,_pretrain})
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
- **Frame-0 arm IK**: precomputed wrist pose for frame 0 loaded from `arm_joint_pos.npy[0]`
  (full-trajectory IK from `process_arm_pipeline.py`).
- **EMA action smoothing** (TJ/rl_games convention, `alpha = new action weight`):
  `smoothed = alpha * raw + (1 - alpha) * prev_smoothed`. Split α: fingers use
  `action_smoothing=0.5` (responsive grasping); arm slice [20:27] uses `arm_action_smoothing=0.17`
  (stronger smoothing → wrist tremor suppression). Applied in `_pre_physics_step`.
- **Contact force projection** (`_get_fingertip_forces`, all 3 train envs): uses
  `sensor.data.force_matrix_w[:, 0, 0, :]` (per-filter-object: Object only — excludes
  self-collision / table contacts) projected onto `-pad_normal_w` (mirrors TJ). Local-frame
  pad-outward unit normals defined in `_FINGERTIP_PAD_NORMALS`:
  thumb (link4) `+Z`, other fingers (link8/12/16/20) `+X`. Distinct from `_FINGERTIP_OFFSETS`
  (tip-position offset for FK / contact target computation). The two are independent and must
  NOT be conflated — earlier bug projected force onto tip-position vector, zeroing out almost
  all compressive contact. See `scripts/process_dataset/visualize_fingertip_normals.py` to
  re-verify visually for either FFW-SH5 or Shadow Hand (TJ reference: thumb `-X`, others `-Y`).
- **Wrist position reward** (`rew_wrist_pos = -1.04` train / `-2.5` pretrain): raw
  `‖wrist_pos − ref_wrist_pos‖`, no Z-weighting, no drift comp. Weight = `rew_fingertip / 5`
  so per-keypoint wrist tracking weight equals per-fingertip weight (5 fingertips → mean).

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

**`_load_partial_checkpoint`** (pretrain → train transfer; always used for grasp-train):
copies policy/value network weights by shape (partial top-left copy on dim mismatch) and
skips `log_std_parameter` (TJ-style σ reset). Preprocessors (`observation_preprocessor` /
`value_preprocessor` RunningStandardScaler) are **deliberately NOT transferred** — the scaler
re-learns the train obs distribution from scratch. Transferring them is actively harmful: the
pretrain stats are frozen (`current_count` ~3e7 → train batches carry ~0 weight) and have
near-zero variance on the object velocity/force/contact dims (those obs are zero in the
no-object pretrain), so real train physics values get divided by ~√1e-8 → exploding inputs that
never correct. Same rationale as the MARL loader's deliberate reset.

### MARL Architecture (`robotis_sh5_grasp_marl_env.py` + `train_marl.py`)

HAPPO (Kuba et al. 2022, "Trust Region Policy Optimisation in Multi-Agent RL", Algorithm 4)
implemented as a layer on top of skrl's MAPPO class via monkey-patches:

- **Hand agent** sees full single-agent grasping context (283D obs minus mass+lift):
  21 MANO kpts (world) + palm state + fingertip vel + full 27D joint state + object state +
  reference deltas + `future_contact` + 27D combined prev_action + fingertip forces.
- **Arm agent** sees a minimal "wrist-pose follower" obs (60D):
  jp_arm + jv_arm + wrist pose (env-pos + world-quat) + wrist linvel + wrist angvel +
  delta_wrist_pos + delta_wrist_rot + prev_arm_action +
  **current_hand_action slot at [40:60]** (filled by forward sequential conditioning).
- **Single shared centralized critic** V(s_global) with explicit non-redundant 286D state
  from `_get_states()` = hand_obs (283D) + delta_wrist_rot (3D). State is NOT auto-flattened
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
| `action_smoothing` (env cfg, finger EMA α) | 0.5 | 0.5 | finger: prev 50% + raw 50% |
| `arm_action_smoothing` (env cfg, arm EMA α) | 0.2 | 0.2 | arm: prev 80% + raw 20% (stronger smoothing → less wrist tremor) |
| `rew_arm_action_reg` / `rew_arm_pose_reg` | -0.008 / -0.002 | -0.008 / -0.002 | 2× hand — penalize arm null-space wandering |

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
│   ├── arm_keypoints.npz              # SH5: {elbow_pos, link7_pos}; env uses both as ref
│   ├── arm_joint_pos.npy              # SH5: (N, 7) per-frame arm IK; env uses [0] for frame-0 init
│   ├── arm_keypoints_shadow.npz       # Shadow Hand variant: same schema
│   ├── arm_joint_pos_shadow.npy       # Shadow Hand variant: arm IK targeting robot0_palm
│   ├── vposer_ik_video.mp4            # SH5 IK visualization
│   └── vposer_ik_video_shadow.mp4     # Shadow Hand IK visualization
├── ffw_sh5/right/<task>/<id>/         # single-agent checkpoints
│   ├── pretrain.pt
│   ├── agent.pt
│   ├── task_info.json
│   └── evaluation_ep_le_<N>/metrics.csv
├── ffw_sh5_marl/right/<task>/<id>/    # MARL checkpoints (parallel tree)
│   └── (same structure)
├── ffw_shadow/right/<task>/<id>/      # Shadow Hand variant (parallel tree)
│   └── (same structure)
└── ffw_shadow_rsi/right/<task>/<id>/  # Shadow Hand RSI warm-start variant (parallel tree)
    └── (same structure) + pretrain_state_cache.npz   # saved at pretrain, loaded at train
```

`evaluate.bash` automatically iterates over every subdirectory under `data/processed/<dataset>/` that
contains a `metrics.csv` (e.g. `ffw_sh5/`, `ffw_sh5_marl/`, `ffw_shadow/`, `ffw_shadow_rsi/`) and
produces per-model aggregate CSVs (`<model>_method{1,2,3}.csv`).

### Robot Platform: FFW-SH5

USD variants (in `data/robots/FFW/`):
- `FFW_SH5_simplified_dex.usd` — used for grasp tasks (enhanced hand collision)
- `FFW_SH5_simplified.usd` — standard training
- `FFW_SH5.usd` — full fidelity
- `FFW_SH5_shadow_instanced.usd` — Shadow Hand replaces right hand (Shadow Hand variant)
  - Created via: flatten `FFW_SH5_shadow.usd` (remove `/Root` and `/shadow_hand` wrappers, merge
    joint scopes, remap all relationship paths) → run `make_robot_usd_instanceable.py`
  - All bodies must live directly under the defaultPrim (e.g. `/FFW_SH5_simplified_dex/<body>`)
    so Isaac Lab's `ContactSensorCfg` paths like `/World/envs/env_*/Robot/<body>` resolve.

Joint groups:
- Base: 6 swerve (`*_wheel_drive/steer`)
- Lift: 1 (`lift_joint`)
- Arms: 7×2 (`arm_l/r_joint[1-7]`), actuated by DY_80/DY_70/DP-42
- Hands (sh5): 20×2 (`finger_l/r_joint[1-20]`), current cfg stiffness=200, damping=5, effort=10
- Hands (shadow, right only): 22 USD joints (FFJ0-3, MFJ0-3, RFJ0-3, LFJ0-4, THJ0-4); only
  18 actuated — FFJ0/MFJ0/RFJ0/LFJ0 are tendon-coupled to J1 (absorbed by PhysX). Cfg uses
  TJ-style PD (stiffness=1.0, damping=0.1) with `effort_limit_sim=10` uniform.

The grasp task fixes the robot base (`fix_root_link=True`) and controls only the right side:
20 fingers (sh5) or 18 fingers (shadow) + 7 arm_r + 1 lift. The lift joint is held at
`cfg.fixed_lift_target=0.0` by the PD controller every step (not part of the action vector).

### Shadow Hand Variant: Key Mechanics

The Shadow Hand variant lives in `tasks/direct/robotis_shadow_grasp/` and shares the same
two-phase pretrain → train workflow as sh5. Notable differences:

- **Frame conventions**: HOcap stores `qpos_wrist_right` quat in a LANDMARK frame derived from
  21 MANO keypoints (z = wrist→middle MCP, x = palm normal). Shadow Hand's `robot0_palm` body
  frame uses a different axis convention. The env code multiplies `body_quat_w[robot0_palm]`
  by a static `_palm_to_landmark_quat` ≈ +90° around palm Z + small tilt before comparing to
  the MANO landmark quat — fix applied both in observation (`wrist_quat_obs`) and in
  termination-check rotation error (`wrist_rot_err`).

- **IK virtual frame**: `process_arm_pipeline.py --robot shadow` registers a virtual frame
  `robot0_palm_virtual` on the (still sh5-URDF) Pinocchio model at the static SE3 from
  `arm_r_link7` to robot0_palm (composed from `FFW_SH5_shadow_flat.usd` joint chain), with the
  static `R_palm_to_landmark` baked in so the IK target quat is interpreted in landmark frame.

- **IK barriers**: For `--robot shadow`, 21 Shadow Hand MANO-keypoint OP_FRAMEs are registered
  on the Pinocchio model (from precomputed default-pose offsets in `_SHADOW_KPT_PLACEMENTS`)
  and used as PositionBarrier targets — replacing the SH5 finger phantom frames. Without this
  the IK would constrain the WRONG fingers (the URDF's sh5 fingers, not what runs in env).

- **Fingertip offsets / pad normals**: mirror TJ exactly (`gr/source/gr/gr/asset/shadow_hand.py`
  + `gr/source/gr/gr/tasks/direct/gr/gr_env.py:204-208`). thumb tip offset `[-0.0085, 0, 0.02]`
  with pad outward `[-1, 0, 0]`; other 4 fingers `[0, -0.006, 0.0175]` with `[0, -1, 0]`.
  Force projection identical to sh5: `(force * -pad_normal).sum().clamp_min(0)`.

- **Tendon coupling**: actuator regex must match exactly the 18 actuated joints (not all 22)
  — use `robot0_(FF|MF|RF|LF|TH)J[1-3]` + `LFJ4` + `THJ[04]`. Otherwise Pinocchio reports
  "actuated joints != joints available" and PhysX silently drops the unmatched ones.

- **State cache dim**: shadow's `_STATE_DIM` (91 for train, 78 for pretrain) replaces sh5's
  (97 / 84) since num_hand_dofs is 18 vs 20. Slicing offsets in `_reset_idx` and the
  EMA-smoothing finger/arm splits use `cfg.num_hand_dofs` (no longer hardcoded `:20`).

- **Viewer**: both shadow cfgs use `origin_type="env"`, `env_index=0` so camera coords are
  env-local and the framing is stable regardless of `num_envs` (default `"world"` made the
  GridCloner shift env 0's world position with grid size).

- **Arm delta-action** (`arm_delta_action=True`, both train + pretrain env/cfg; finger + mass
  unaffected): the ARM action is a per-control-step *residual* rather than an absolute target.
  Per control step in `_pre_physics_step`: `delta_cmd = raw_arm * arm_delta_scale` (rad, raw ∈
  [-1,1]) → `delta_ema = α·delta_cmd + (1-α)·delta_ema` (EMA α=`arm_delta_smoothing` on the
  *delta/velocity*, **not** the integrated target — smoothing the target degenerates to
  `prev + α·delta`) → `arm_target = clamp(arm_target + delta_ema, joint_limits)` (accumulator
  clamped → no integral windup). `_apply_action` then writes `_arm_target` directly for the arm.
  Integration low-pass-filters action noise → suppresses wrist tremor that the absolute-action
  scheme + `arm_action_smoothing` could not. Buffers `_arm_target`/`_arm_delta_ema` are seeded in
  `_reset_idx` (target = actual reset arm pose, delta EMA = 0). **Obs**: the arm slice of the
  action-history obs (`_prev_action[:, N_f:N_f+N_a]`) is overwritten with the *normalized
  integrated target* (not the raw delta) so it carries the same absolute-target meaning as the
  finger slice and exposes the integrator's hidden state; done at the `_prev_action` capture site
  so obs dim, lag-1 timing, and reset-to-zero stay identical to the fingers. Defaults: `arm_delta_scale=0.25`,
  `arm_delta_smoothing=0.5`; `arm_action_smoothing` is unused while enabled. **Rollback**: set
  `arm_delta_action=False` (restores the absolute EMA arm with zero other changes). All code sites
  are bracketed by `[ROLLBACK MARKER: arm-delta]` comment blocks. Mirror values across BOTH cfgs
  so pretrain→train checkpoint transfer keeps matching action dynamics.
- **Hand delta-action** (`hand_delta_action=True`, both train + pretrain env/cfg; arm + mass
  unaffected): identical residual scheme applied to the FINGER slice, for action-semantics
  consistency (arm + fingers both velocity/residual). Per control step: `delta_cmd =
  raw_hand * hand_delta_scale` → `delta_ema = α·delta_cmd + (1-α)·delta_ema`
  (α=`hand_delta_smoothing`) → `hand_target = clamp(hand_target + delta_ema, joint_limits)`.
  `_apply_action` writes `_hand_target` directly for fingers. Buffers `_hand_target`/
  `_hand_delta_ema` seeded in `_reset_idx` (target = actual reset finger pose, EMA = 0).
  **Obs**: `_prev_action[:, :N_f]` is overwritten with the *normalized integrated finger target*
  (same treatment as the arm slice). Defaults: `hand_delta_scale=0.5`, `hand_delta_smoothing=1.0`;
  `action_smoothing` is unused for the finger slice while enabled. **Caveats**: (1) in delta mode
  `rew_hand_action_reg` (Σ raw finger action²) becomes a finger *velocity* penalty, not a flexion
  penalty; (2) delta=0 means HOLD (not "go to mid-range") and zero-mean action noise random-walks
  the target around the open reset pose, so contact-force exploration is weaker than absolute mode
  (observed: fingertip forces drop) — may need lower `rew_hand_action_reg` / higher exploration σ.
  **Re-pretrain required** when toggling (finger action semantics change). **Rollback**: set
  `hand_delta_action=False` (restores absolute EMA fingers). All code sites are bracketed by
  `[ROLLBACK MARKER: hand-delta]` comment blocks. Mirror values across BOTH cfgs for ckpt transfer.

### Shadow Hand RSI Warm-Start Variant (`tasks/direct/robotis_shadow_grasp_rsi/`)

A copy of the Shadow Hand task (`Robotis-Shadow-Grasp-Rsi-{,Pretrain-}Direct-v0`) that warm-starts
the train phase's Reference-State-Initialization (RSI) cache from the **pretrain phase's state
cache**. Motivation: the train cache normally starts EMPTY, so `_reached_frame` crawls from 0 and
the policy can only start near frame 0 until the cache fills (cold start). The pretrain phase already
visited ALL frames under physics, so its per-frame robot states are physically-valid init poses
(object-free) — reusing them lets the train phase sample/start across the whole trajectory from step 0.
All code sites are bracketed by `[ROLLBACK MARKER: pretrain-cache-warmstart]`.

- **Cache save (pretrain)**: `scripts/skrl/train.py`, after `runner.run()` for an RSI *pretrain* task,
  dumps `env.unwrapped._state_cache` + `_init_flg` + `_reached_frame` to `pretrain_state_cache.npz`
  in BOTH the run `log_dir` and the data-tree `_ckpt_dir`. The benchmark script also copies it next
  to `pretrain.pt`.
- **Cache load (train)**: `train.py` looks for `pretrain_state_cache.npz` as a **sibling of
  `--checkpoint`**; if present, calls `env.unwrapped.set_pretrain_cache(path)` (validates trajectory
  length). No cfg path field — convention only.
- **Concurrent read+write RSI** (`_reset_idx`): with the pretrain cache loaded, every reset
  (1) samples a start frame ONLY among frames present in the **train OR pretrain** cache (drops the
  `_reached_frame` gate; failure-weighted, ≈uniform early; rewound `_adaptive_back_frames` for run-up,
  with a safeguard snapping back to the sampled frame if the rewound one is uncovered), and
  (2) restores the robot from the **train cache if that frame has one, else the pretrain cache**
  (object always from the reference trajectory on a train cache-miss). The "fixed home pose +
  frame-0 IK" cold fallback is therefore never reached on this path. The train cache is read AND
  written from step 0; as it fills, the pretrain fallback self-deprecates (watch
  `Curriculum / pretrain_fallback_ratio` → 0, `Curriculum / cache_coverage` → 1).
- **Cache write gate**: while the pretrain cache is loaded, `_save_state_cache` requires
  `episode_length_buf >= 3` before writing (pretrain poses are object-free → restoring with the object
  can interpenetrate and terminate in 1–2 steps; this keeps those bad states out of the train cache).
  The reference is NEVER rewritten and the pretrain reward column is never compared (pretrain cache is
  read-only) — no reward-scale mixing.
- **init-save fix**: the TJ-style first-reset frame-0 seed (`_state_cache[0]`) now copies from an env
  that *actually* started at frame 0 (was a bug: warm-start samples arbitrary start frames, so the old
  hard-coded local-env-0 wrote a grasp-frame state into the frame-0 slot → grasp-pose-vs-table-reference
  mismatch → spurious early termination).
- **Separate tree**: `train.py` routes any task containing `"Rsi"` to `ffw_shadow_rsi` (checkpoints,
  task_info, cache, metrics) and the agents YAML log dir to `robotis_shadow_grasp_rsi{,_pretrain}` so
  RSI runs never clobber plain-Shadow results. `evaluate.bash` auto-discovers the new tree.
- **Rollback / baseline**: set `pretrain_cache_warmstart=False` (cfg) → all warm-start logic is
  gated off and the env behaves IDENTICALLY to the original `robotis_shadow_grasp` (empty train cache,
  `_reached_frame` gate, fixed-home-pose fallback). Clean A/B baseline. `rollout.py` never arms the
  warm-start, so evaluation runs the same protocol as the other variants.
- Benchmark: `scripts/benchmark/hocap/{train,evaluate}_sequences_shadow_rsi.sh`.

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
- **Shadow Hand wrist quat conversion is REQUIRED**: env code must apply `_palm_to_landmark_quat`
  before comparing `robot0_palm` body quat to MANO ref. Skipping this adds a constant ~90°
  offset to `wrist_rot_err` and instantly trips `max_wrist_rot_err` termination at frame 0.
- **Shadow USD layout matters**: defaultPrim must be a root-level prim (not nested). Bodies
  must live directly under `/<defaultPrim>/<body>` — Isaac Lab `ContactSensorCfg.prim_path`
  uses pattern `/World/envs/env_*/Robot/<body>` and won't find bodies under sub-prim wrappers.
- **Per-robot output files**: `process_arm_pipeline.py --robot shadow` writes `*_shadow.npy`
  / `*_shadow.npz` / `*_shadow.mp4` so sh5 and shadow references coexist. The Shadow env
  loads the `_shadow`-suffixed files at trajectory load time.

## Code Style

- **Ruff**: line-length 120, Python 3.10 target
- **Pyright** in basic mode
- Import order: omniverse-extensions → isaaclab → isaaclab-* → first-party
- `__init__.py` files may have unused imports (F401 ignored)
- Physics timestep: `dt = 1/120s`, control at 30 Hz (`decimation=4`)
