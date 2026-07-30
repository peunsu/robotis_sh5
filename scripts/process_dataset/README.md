# `scripts/process_dataset/`

Offline data / asset preparation. Nothing here is imported by the training package at import time —
the single runtime dependency is `sonic/sonic_prior.py`, which the SONIC-residual env loads via
`sys.path` at `_post_init_buffers`.

Scripts are run **from the repo root** (`python scripts/process_dataset/<group>/<script>.py …`);
several of them resolve data paths as `Path(__file__).resolve().parents[3]`, so keep the
two-levels-under-`scripts/` depth if you move anything.

## Layout

| Dir | Role | Scripts |
|---|---|---|
| `dataset/` | raw dataset → per-trajectory npz | `oakink.py`, `hocap.py`, `parahome.py`, `parahome_hand_contact.py`, `parahome_smpl_for_sonic.py`, `estimate_object_mass.py` |
| `retarget/` | human motion → robot joint references | `process_arm_pipeline.py` (+ `compute_frame0_ik_pink.py` lib), `retarget_g1_pyroki.py`, `retarget_g1_shadow_pink.py` (+ `shadow_usd_model.py` lib) |
| `assets/` | meshes / robots → USD + URDF | `convert_obj_to_usd.py`, `parahome_convert_obj_to_usd.py`, `parahome_build_articulation_spec.py`, `make_robot_usd_instanceable.py`, `build_g1_shadow_usd.py`, `graft_shadow_onto_g1.py`, `extract_g1_urdf_from_usd.py`, `export_g1_shadow_urdf.py` |
| `sonic/` | frozen SONIC whole-body prior | `sonic_prior.py` (runtime lib), `sonic_playback.py` |
| `diagnostics/` | inspection / validation / video | `inspect_g1_asset.py`, `dump_g1_joint_order.py`, `spawn_test_g1_shadow.py`, `render_retarget.py` |

## Which interpreter

| Env | Scripts | Why |
|---|---|---|
| `env_isaaclab` (default `python`) | everything except the two below | torch / smplx / gear_sonic / pinocchio |
| `env_isaaclab` **+ Isaac Sim** (`isaaclab.sh -p …`, or plain python — the app boots itself) | `assets/convert_obj_to_usd.py`, `assets/parahome_convert_obj_to_usd.py`, `diagnostics/*` except `inspect_g1_asset.py`ᵃ | `SimulationApp` / `AppLauncher` |
| **`env_pyroki`** (`/home/peunsu/anaconda3/envs/env_pyroki/bin/python`) | `retarget/retarget_g1_pyroki.py` | jax needs numpy ≥ 2; `import jax/pyroki/pink` **fails** inside `env_isaaclab` (numpy 1.26 pin) |

ᵃ `inspect_g1_asset.py` also boots Isaac Sim (it reads the remote G1 USD).

## Pipelines

**P1/P2 — OakInk · HO-Cap → FFW-SH5 (± Shadow hand)**
```
dataset/{oakink,hocap}.py → dataset/estimate_object_mass.py → assets/convert_obj_to_usd.py
  → retarget/process_arm_pipeline.py [--robot shadow]        # ← called by scripts/benchmark/{oakink,hocap}/*.sh
```

**P3 — ParaHome → G1+Shadow locomanip (legacy, pink)**
```
dataset/parahome.py → assets/parahome_build_articulation_spec.py → assets/parahome_convert_obj_to_usd.py
  → retarget/retarget_g1_shadow_pink.py                      # writes trajectory.npz
```
Superseded by P4. Note the env cfgs default `retarget_file="trajectory_pyroki.npz"`, so this branch's
output is not what the envs read any more.

**P4 — ParaHome → G1+Shadow SONIC residual (current)**
```
dataset/parahome.py → assets/parahome_{build_articulation_spec,convert_obj_to_usd}.py
  → dataset/parahome_hand_contact.py
  → retarget/retarget_g1_pyroki.py        [env_pyroki]       # writes trajectory_pyroki.npz
  → dataset/parahome_smpl_for_sonic.py                       # writes sonic_smpl_50fps.npz (train hard-fails without it)
```

**P5 — robot assets (one-shot; outputs live in the gitignored `data/robots/`)**
```
FFW:  <flattened shadow USD> → assets/make_robot_usd_instanceable.py → FFW_SH5_shadow_instanced.usd
G1:   assets/build_g1_shadow_usd.py (flatten + strip Dex3) → G1_shadow_stripped.usd
      → assets/graft_shadow_onto_g1.py                      → G1_shadow.usd          ← the sim asset
      → diagnostics/dump_g1_joint_order.py                  → g1_shadow_joint_order.json
      → assets/export_g1_shadow_urdf.py                     → urdf_pyroki/g1_shadow.urdf   (PyRoki)
      → assets/extract_g1_urdf_from_usd.py                  → urdf/g1_from_usd.urdf        (pink, legacy)
```
Keep these even though nothing calls them: `data/robots/` is gitignored, so they are the only recipe
for regenerating the assets.
