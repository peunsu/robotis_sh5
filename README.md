# robotis_sh5

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/robotis_sh5

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/robotis_sh5/robotis_sh5/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/robotis_sh5"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```

---

## Robotis-Sh5-Grasp: Dexterous Grasping

### Overview

`Robotis-Sh5-Grasp-Direct-v0` is a dexterous grasping environment for the FFW-SH5 full-body robot.
The policy controls 27 joints (20 right-hand fingers + 7 right arm) and outputs one additional
mass parameter (MassDexMimic), totalling a **28-dimensional action space**. The lift joint is
held at a fixed target (URDF upper limit) by the PD controller every step and is **not part of
the action**, but remains in the observation for state awareness.
Reference trajectories come from the [OakInk-Image](https://oakink.net/) dataset or the
[HO-Cap](https://github.com/NVlabs/HO-Cap) dataset, both stored in SPIDER format.

### Data Preparation

#### OakInk dataset

##### 1. Download OakInk-Image dataset

Place the raw dataset at:

```
source/robotis_sh5/data/raw/oakink/image/
```

##### 2. Process OakInk → SPIDER format

```bash
python scripts/process_dataset/oakink.py
```

Output:
- `data/processed/oakink/mano/right/{task}/{data_id}/trajectory_keypoints.npz` — reference trajectories
- `data/processed/oakink/assets/objects/{obj_id}/visual.obj` — centered object meshes

##### 3. Convert object meshes to USD

Must be run inside the Isaac Lab Python environment:

```bash
# Convert a single object (default --dataset oakink)
isaaclab.sh -p scripts/process_dataset/convert_obj_to_usd.py --object-id A01001

# Convert all objects
isaaclab.sh -p scripts/process_dataset/convert_obj_to_usd.py

# Re-convert (overwrite existing USD)
isaaclab.sh -p scripts/process_dataset/convert_obj_to_usd.py --overwrite

# Custom mass / friction
isaaclab.sh -p scripts/process_dataset/convert_obj_to_usd.py --mass 0.15 --friction 0.9
```

Converted USD files are written to `data/processed/oakink/assets/objects/{obj_id}/visual.usd`.

##### 4. Per-object mass ranges

A pre-built mass table is provided at:

```
data/processed/oakink/object_mass.json
```

Format: `{"A01001": [min_kg, max_kg], ...}`

This file is loaded automatically at training time to set `object_mass_min` / `object_mass_max`
per object. If an object is missing from the JSON or has `null` values, the default
`[0.05, 0.20] kg` is used as fallback.

To regenerate the table using the Claude Vision API (requires `pip install anthropic`):

```bash
# Estimate mass for all objects (skips already-processed entries)
python scripts/process_dataset/estimate_object_mass.py --resume

# Force re-estimation of all objects
python scripts/process_dataset/estimate_object_mass.py
```

#### HO-Cap dataset

##### 1. Download HO-Cap dataset

Place the raw dataset at:

```
source/robotis_sh5/data/raw/hocap/
```

##### 2. Process HO-Cap → SPIDER format

```bash
python scripts/process_dataset/hocap.py
```

Output mirrors the OakInk layout under `data/processed/hocap/`.

##### 3. Convert object meshes to USD

```bash
isaaclab.sh -p scripts/process_dataset/convert_obj_to_usd.py --dataset hocap
```

Place or symlink the HO-Cap object meshes under `data/processed/hocap/assets/objects/` first.

##### 4. Per-object mass ranges

Place a mass table at `data/processed/hocap/object_mass.json` (same format as OakInk).
If absent, the default `[0.05, 0.20] kg` is used for all HO-Cap objects.

#### Robot USD instancing (memory optimization)

The robot USD imported from URDF stores all mesh data inline, which causes per-env duplication
when spawning many environments (e.g., `num_envs=2048`) — leading to GPU/CPU memory overflow.
To enable USD-level instancing, run the post-processing script once to extract each link's
geometry into separate layers and add `instanceable=True` references back to the main USD:

```bash
python scripts/process_dataset/make_robot_usd_instanceable.py \
    source/robotis_sh5/data/robots/FFW/FFW_SH5_simplified_dex.usd \
    source/robotis_sh5/data/robots/FFW/FFW_SH5_simplified_dex_instanced.usd
```

The resulting USD is referenced from `robotis_sh5_grasp_env_cfg.py` and shared across all
envs at spawn time. The original USD is preserved for comparison / rollback.

### Running the Environment

#### Verify registration

```bash
python scripts/list_envs.py
```

#### Sanity-check with dummy agents

```bash
# Zero-action agent (checks env reset / obs / reward shapes)
python scripts/zero_agent.py --task=Robotis-Sh5-Grasp-Direct-v0

# Random-action agent
python scripts/random_agent.py --task=Robotis-Sh5-Grasp-Direct-v0
```

#### Train with SKRL (recommended)

```bash
python scripts/skrl/train.py --task=Robotis-Sh5-Grasp-Direct-v0 --num_envs=2048
```

Add `--video --video_length 200 --video_interval 1000` to record an mp4 every 1000 env-steps
into `logs/skrl/<task>/<run>/videos/train/`. Camera rendering needs significant VRAM —
typically reduce `--num_envs` to 128–512 when recording video.

#### Train with other RL frameworks

```bash
python scripts/rsl_rl/train.py  --task=Robotis-Sh5-Grasp-Direct-v0 --num_envs=2048
python scripts/rl_games/train.py --task=Robotis-Sh5-Grasp-Direct-v0
python scripts/sb3/train.py      --task=Robotis-Sh5-Grasp-Direct-v0
```

#### Play / inference

```bash
python scripts/skrl/play.py --task=Robotis-Sh5-Grasp-Direct-v0 --checkpoint=<CHECKPOINT_PATH>
```

### Benchmark: Multi-sequence Training & Evaluation

Scripts under `scripts/benchmark/` provide an end-to-end pipeline for training one checkpoint per sequence
and evaluating with standardised metrics (E_t, E_r, E_j, E_ft) compatible with the
[ManipTrans / DexMachina evaluation protocol](https://github.com).

#### Directory layout produced

Results are stored per dataset so OakInk and HO-Cap outputs never collide.
The `ffw_sh5/` tree mirrors the `mano/` tree exactly.

```
source/robotis_sh5/data/processed/<dataset>/   # e.g. oakink/ or hocap/
├── mano/right/<trajectory_task>/<data_id>/    ← reference trajectories (source)
├── object_mass.json                           ← per-object mass ranges
├── ffw_sh5/right/<trajectory_task>/<data_id>/
│   ├── pretrain.pt              ← saved after pretrain
│   ├── agent.pt                 ← saved after train
│   ├── task_info.json           ← training metadata (checkpoint path, timesteps, seed, …)
│   └── evaluation_ep_le_<N>/
│       └── metrics.csv          ← 32 rollout rows per sequence
├── ffw_sh5_method1.csv          ← ManipTrans aggregate (all 4 metrics)
├── ffw_sh5_method2.csv          ← DexMachina aggregate (E_t + E_r only)
└── ffw_sh5_method3.csv          ← Mean over all rollouts
```

`task_info.json` is written automatically after each training run and records:

| Field | Description |
|---|---|
| `task` | Isaac Lab task name |
| `dataset` | `"oakink"` or `"hocap"` |
| `object_id` | Object identifier |
| `trajectory_task` | Trajectory directory name |
| `trajectory_data_id` | Sub-index within the trajectory directory |
| `num_envs` | Number of parallel environments |
| `timesteps` | Total training timesteps |
| `seed` | Random seed |
| `checkpoint` | Absolute path to the resume checkpoint (pretrain.pt for train step) |
| `log_dir` | skrl experiment log directory |
| `skrl_version` | skrl library version |
| `training_time_s` | Wall-clock training time in seconds |
| `trained_at` | ISO-8601 timestamp |

Per-dataset scripts live under `scripts/benchmark/oakink/` and `scripts/benchmark/hocap/`.
The two trees mirror each other (same flags, same env-var contract).

#### Step 1 — Configure sequences

Each entry in `SEQUENCES` matches the corresponding `mano/right/` folder name:
`{OBJECT_ID}-{SEQ}-{GESTURE}` (OakInk) or `subject_{N}-{DATE_TIME}-{G_OBJECT_ID}` (HO-Cap).

```bash
# scripts/benchmark/oakink/train_sequences.sh
SEQUENCES=(
    "C11001-0001-0007"   # → object_id=C11001, task=C11001-0001-0007
    "A01001-0001-0000"
)
```

#### Step 2 — Train (dataset → IK → pretrain → train)

```bash
# Run the full pipeline for all sequences; skip steps already completed.
bash scripts/benchmark/oakink/train_sequences.sh
bash scripts/benchmark/hocap/train_sequences.sh

# Force re-running all steps even if outputs already exist.
FORCE=1 bash scripts/benchmark/oakink/train_sequences.sh

# Override num_envs (useful when enabling video → camera VRAM overhead).
NUM_ENVS=256 PRETRAIN_NUM_ENVS=512 bash scripts/benchmark/oakink/train_sequences.sh

# Record training mp4 every 1000 env-steps into logs/skrl/<task>/<run>/videos/train/
VIDEO=1 NUM_ENVS=256 PRETRAIN_NUM_ENVS=512 \
    bash scripts/benchmark/oakink/train_sequences.sh

# Custom video length / interval
VIDEO=1 VIDEO_LENGTH=300 VIDEO_INTERVAL=2000 NUM_ENVS=256 \
    bash scripts/benchmark/oakink/train_sequences.sh
```

`train_sequences.sh` runs **four steps per sequence** in order:

| Step | Action | Skip condition |
|---|---|---|
| 1. Data check | Verify `mano/right/<task>/` exists | Error if missing |
| 2. Frame-0 arm IK | `scripts/process_dataset/compute_frame0_ik.py --dataset <DATASET>` | `frame0_arm_joint_pos.npy` exists |
| 3. Pretrain | `Robotis-Sh5-Grasp-Pretrain-Direct-v0` | `pretrain.pt` exists |
| 4. Train | `Robotis-Sh5-Grasp-Direct-v0` | `agent.pt` exists |

Key settings at the top of the script and supported env-var overrides:

| Variable | Default | Type | Meaning |
|---|---|---|---|
| `DATASET` | `oakink` / `hocap` | fixed | Determined by script subdir |
| `TASK_PRETRAIN` | `Robotis-Sh5-Grasp-Pretrain-Direct-v0` | fixed | Pretrain task name |
| `TASK` | `Robotis-Sh5-Grasp-Direct-v0` | fixed | Train task name |
| `PRETRAIN_NUM_ENVS` | `4096` | env-var | Parallel environments for pretrain |
| `NUM_ENVS` | `2048` | env-var | Parallel environments for train |
| `PRETRAIN_TIMESTEPS` | `3000` (oakink) / `10000` (hocap) | fixed | Pretrain env steps |
| `TIMESTEPS` | `10000` (oakink) / `40000` (hocap) | fixed | Train env steps |
| `FORCE` | `0` | env-var | Re-run even if outputs exist |
| `VIDEO` | `0` | env-var | Record training videos (`--video`) |
| `VIDEO_LENGTH` | `200` | env-var | Frames per recording |
| `VIDEO_INTERVAL` | `1000` | env-var | Env-steps between recordings |

Checkpoints and `task_info.json` are saved to
`source/robotis_sh5/data/processed/<dataset>/ffw_sh5/right/<trajectory_task>/<data_id>/`.

You can also run individual steps manually for a single sequence:

```bash
# Step 1: dataset processing (run once before training)
python scripts/process_dataset/oakink.py \
    --seq-id-ts "C11001_0001_0007/2021-10-03-14-32-01"

# Step 2: frame-0 arm IK
python scripts/process_dataset/compute_frame0_ik.py \
    --dataset oakink --task C11001-0001-0007 --data_id 0

# Step 3: pretrain
python scripts/skrl/train.py \
    --task Robotis-Sh5-Grasp-Pretrain-Direct-v0 \
    --num_envs 4096 --timesteps 3000 --headless \
    --dataset oakink --object_id C11001 \
    --trajectory_task C11001-0001-0007 --trajectory_data_id 0

# Step 4: train (pass pretrain checkpoint via --checkpoint)
python scripts/skrl/train.py \
    --task Robotis-Sh5-Grasp-Direct-v0 \
    --num_envs 2048 --timesteps 10000 --headless \
    --checkpoint source/robotis_sh5/data/processed/oakink/ffw_sh5/right/C11001-0001-0007/0/pretrain.pt \
    --dataset oakink --object_id C11001 \
    --trajectory_task C11001-0001-0007 --trajectory_data_id 0

# With video recording (reduce num_envs to fit GPU VRAM)
python scripts/skrl/train.py \
    --task Robotis-Sh5-Grasp-Direct-v0 \
    --num_envs 256 --timesteps 10000 --headless \
    --video --video_length 200 --video_interval 1000 \
    --checkpoint source/robotis_sh5/data/processed/oakink/ffw_sh5/right/C11001-0001-0007/0/pretrain.pt \
    --dataset oakink --object_id C11001 \
    --trajectory_task C11001-0001-0007 --trajectory_data_id 0
```

#### Step 3 — Evaluate

```bash
# Run rollouts + produce aggregate CSVs for all sequences.
bash scripts/benchmark/oakink/evaluate_sequences.sh
bash scripts/benchmark/hocap/evaluate_sequences.sh

# Force re-running rollouts even if metrics.csv already exists.
FORCE=1 bash scripts/benchmark/oakink/evaluate_sequences.sh

# Record an mp4 of env 0's rollout into <OUT_DIR>/videos/ for each sequence.
VIDEO=1 bash scripts/benchmark/oakink/evaluate_sequences.sh

# Custom video length (default 300 steps)
VIDEO=1 VIDEO_LENGTH=500 bash scripts/benchmark/oakink/evaluate_sequences.sh
```

Key settings and supported env-var overrides for `evaluate_sequences.sh`:

| Variable | Default | Type | Meaning |
|---|---|---|---|
| `DATASET` | `oakink` / `hocap` | fixed | Determined by script subdir |
| `N_ROLLOUTS` | `32` | fixed | Parallel rollout episodes per sequence |
| `TIMESTEPS` | `10000` | fixed | Used only for output directory naming (must match training) |
| `FORCE` | `0` | env-var | Re-run rollouts even if `metrics.csv` exists |
| `VIDEO` | `0` | env-var | Record video of env 0's rollout |
| `VIDEO_LENGTH` | `300` | env-var | Frames per recording |

Rollouts are **deterministic by default** (mean action) — matches the rl-games
`player.deterministic=True` convention used in TJ/GR. Pass `--stochastic` to
sample from the policy Gaussian instead.

You can also run rollout on a single sequence manually:

```bash
python scripts/skrl/rollout.py \
    --task Robotis-Sh5-Grasp-Direct-v0 \
    --checkpoint source/robotis_sh5/data/processed/oakink/ffw_sh5/right/C11001-0001-0007/0/agent.pt \
    --output_dir  source/robotis_sh5/data/processed/oakink/ffw_sh5/right/C11001-0001-0007/0/evaluation_ep_le_10000 \
    --n_rollouts 32 \
    --headless \
    --dataset oakink --object_id C11001 \
    --trajectory_task C11001-0001-0007 --trajectory_data_id 0

# With video recording
python scripts/skrl/rollout.py ... --video --video_length 300
```

#### Step 4 — Aggregate metrics only

If rollouts are already done and you only want to recompute the CSV summaries:

```bash
# Aggregate one dataset
bash scripts/benchmark/evaluate.bash source/robotis_sh5/data/processed/oakink
```

#### Evaluation metrics

| Metric | Column | Unit | M1 threshold | M2 threshold |
|---|---|---|---|---|
| E_t — object translation error vs ref | `e_t_cm` | cm | < 3.0 | < 10.0 |
| E_r — object rotation error vs ref | `e_r` | deg | < 30.0 | < 28.6 (0.5 rad) |
| E_j — all 21 keypoint error vs MANO ref | `e_j_cm` | cm | < 8.0 | — |
| E_ft — fingertip error vs MANO ref | `e_ft_cm` | cm | < 6.0 | — |

**Method 1 (ManipTrans):** success if all four metrics are below M1 thresholds.  
**Method 2 (DexMachina):** success if E_t < 10 cm **and** E_r < 0.5 rad.  
**Method 3:** mean over all rollouts regardless of success.

---

#### Debug visualization

Enable reference fingertip and wrist 6D-pose markers in the sim:

```python
# In robotis_sh5_grasp_env_cfg.py
debug_vis: bool = True
debug_vis_num_envs: int = 16  # show markers for first N envs
```

Or pass at launch (if your runner supports it):

```bash
python scripts/zero_agent.py --task=Robotis-Sh5-Grasp-Direct-v0 --enable_cameras
```

### Environment Details

| Item | Value |
|---|---|
| Action space | 28 (fingers 20 + arm_r 7 + mass 1); lift held fixed at top |
| Observation space | 279 |
| Control frequency | 30 Hz (decimation=4 @ 120 Hz physics) |
| Episode length | matched to trajectory length |
| Default num_envs | 2048 |
| Default object | `C11001` |
| Supported datasets | `oakink`, `hocap` |

**Action breakdown (28-dim):**

| Group | Dim | Description |
|---|---|---|
| `fingers` | 20 | Right finger joint position deltas (`finger_r_joint1-20`) |
| `arm_r` | 7 | Right arm joint position deltas (`arm_r_joint1-7`) |
| `mass` | 1 | Normalised object mass parameter in `[-1, 1]` → `[mass_min, mass_max]` |

Lift joint is **not in the action**. It is held at `cfg.fixed_lift_target` (default `0.0` = URDF
upper limit, fully up) by the PD controller every step. To allow the policy to control lift,
re-add it to `_action_joint_ids` in `robotis_sh5_grasp_env.py` and bump `action_space` by 1.

**Observation breakdown (279-dim):**

| Group | Dim | Description |
|---|---|---|
| `hand_kpts_pos` | 63 | All 21 MANO keypoints in world frame (21×3) |
| `wrist_quat` | 4 | Wrist global orientation (wxyz) |
| `wrist_linvel` | 3 | Wrist global linear velocity |
| `wrist_angvel` | 3 | Wrist global angular velocity |
| `fingertip_vel` | 15 | Fingertip linear velocities (5×3) |
| `joint_pos` | 28 | Controlled joint angles, normalised (includes lift for state awareness) |
| `joint_vel` | 28 | Controlled joint velocities (includes lift) |
| `obj_pos` | 3 | Object position |
| `obj_quat` | 4 | Object orientation (wxyz) |
| `obj_linvel` | 3 | Object linear velocity |
| `obj_angvel` | 3 | Object angular velocity |
| `delta_kpts_world` | 63 | Next-frame delta for all 21 keypoints in world frame |
| `delta_ft_obj` | 15 | Next-frame fingertip error in object frame |
| `delta_obj_pos` | 3 | Next-frame object position error |
| `delta_obj_rot` | 3 | Next-frame rotation error (axis-angle) |
| `future_contact` | 5 | Predicted contact flag per fingertip |
| `prev_action` | 28 | Previous action (27 joints + 1 mass) |
| `fingertip_forces` | 5 | Normal contact force per fingertip |

---

## Acknowledgement
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [robotis_lab](https://github.com/ROBOTIS-GIT/robotis_lab)
- [ai_worker (feature-sh5-release)](https://github.com/ROBOTIS-GIT/ai_worker/tree/feature-sh5-release)
