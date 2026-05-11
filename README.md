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

## Robotis-Sh5-Grasp: OakInk Dexterous Grasping

### Overview

`Robotis-Sh5-Grasp-Direct-v0` is a dexterous grasping environment for the FFW-SH5 full-body robot.
The policy controls 20 right-hand finger joints while the arm is held at a fixed pre-grasp pose.
Reference trajectories come from the [OakInk-Image](https://oakink.net/) dataset (SPIDER format).

### Data Preparation

#### 1. Download OakInk-Image dataset

Place the raw dataset at:

```
source/robotis_sh5/data/raw/oakink/image/
```

#### 2. Process OakInk → SPIDER format

```bash
python scripts/process_dataset/oakink.py
```

Output:
- `data/processed/oakink/mano/right/{task}/{data_id}/trajectory_keypoints.npz` — reference trajectories
- `data/processed/oakink/assets/objects/{obj_id}/visual.obj` — centered object meshes

#### 3. Convert object meshes to USD

Must be run inside the Isaac Lab Python environment:

```bash
# Convert a single object
isaaclab.sh -p scripts/process_dataset/convert_oakink_to_usd.py --object-id A01001

# Convert all objects
isaaclab.sh -p scripts/process_dataset/convert_oakink_to_usd.py

# Re-convert (overwrite existing USD)
isaaclab.sh -p scripts/process_dataset/convert_oakink_to_usd.py --overwrite

# Custom mass / friction
isaaclab.sh -p scripts/process_dataset/convert_oakink_to_usd.py --mass 0.15 --friction 0.9
```

Converted USD files are written to `data/processed/oakink/assets/objects/{obj_id}/visual.usd`.

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
| Action space | 20 (right finger joint deltas) |
| Observation space | 134 |
| Control frequency | 30 Hz (decimation=4 @ 120 Hz physics) |
| Episode length | 5 s |
| Default num_envs | 2048 |
| Default object | `A01001` |

**Observation breakdown (134-dim):**

| Group | Dim | Description |
|---|---|---|
| `joint_pos` | 20 | Right finger joint angles |
| `joint_vel` | 20 | Right finger joint velocities |
| `fingertip_pos` | 15 | Fingertip world positions (5×3) |
| `fingertip_vel` | 15 | Fingertip linear velocities (5×3) |
| `obj_pos` | 3 | Object position |
| `obj_quat` | 4 | Object orientation (wxyz) |
| `obj_linvel` | 3 | Object linear velocity |
| `obj_angvel` | 3 | Object angular velocity |
| `delta_fingertip` | 15 | Fingertip − ref_fingertip (object frame) |
| `delta_obj_pos` | 3 | obj_pos − ref_obj_pos |
| `delta_obj_rot` | 3 | Rotation error (axis-angle) |
| `future_contact` | 5 | Predicted contact flag per fingertip |
| `prev_action` | 20 | Previous action |
| `fingertip_forces` | 5 | Normal contact force per fingertip |

---

## Acknowledgement
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [robotis_lab](https://github.com/ROBOTIS-GIT/robotis_lab)
- [ai_worker (feature-sh5-release)](https://github.com/ROBOTIS-GIT/ai_worker/tree/feature-sh5-release)