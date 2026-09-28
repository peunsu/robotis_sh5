"""Process the GRAB dataset (Taheri et al., ECCV 2020) into per-sequence clips in the ParaHome layout.

GRAB: 10 subjects x 51 objects, 1335 sequences @120 FPS. Whole-body SMPL-X (per-subject template mesh,
PCA-24 hands + the full 165-D `fullpose`), one rigid object and a table top per sequence, the subject
standing at the table. Each sequence becomes ONE clip of class `single_rigid`.

Written so every consumer of parahome.py output reads it unchanged (env: dataset_root -> processed/grab;
per-clip scripts: --dataset grab). Differences from ParaHome and how they are handled:
  * 120 FPS -> every 4th frame (30 FPS). The envs and per-clip scripts assume a 30 FPS source.
  * Body shape is a subject template mesh, not betas -> stored as `smplx_v_template` (10475,3).
    `smplx_betas` is NOT written: a zero-betas stand-in would silently give a generic body to any
    consumer that ignores the template. Hands: fullpose[75:165] with flat_hand_mean=True, use_pca=False
    reproduces GRAB's own PCA-24 model to 3.6e-7 m — the ParaHome SMPL-X convention. The repo's
    SMPL-X v1.1 and GRAB's bundled v1.0 give identical FK with a template (0.0 m).
  * GRAB's ObjectModel computes `v @ R + t`, so the world rotation of the object and table is
    R(global_orient)^T, not R.
  * No ParaHome Xsens stream: `joint_positions` is not written (nothing downstream reads it).
    `body_global_transform` = SMPL-X pelvis joint + remove_smpl_base_rot(global_orient). ParaHome's
    transform has exactly that rotation (0.0 deg on s100_seg00_pan) but its Xsens pelvis position, 7-11 cm
    from the SMPL-X pelvis joint. The envs use it only as the default root (the retarget root replaces
    it) and for camera aiming.
  * The table is a thin top plate (no legs), stored as the only context object `ctx__table__base`.
  * The whole scene is turned WORLD_YAW_DEG about the world z axis (see the constant for why).
  * Each clip is trimmed to [first hand contact - 0.5 s, last hand contact + 0.5 s] (see _contact_window).

Input : data/raw/grab/data/grab/sN/<seq>.npz
        data/raw/grab/data/tools/{object_meshes/contact_meshes,subject_meshes}/...
Output: data/processed/grab/smplx/single_rigid/<sN>_<seq>/0/trajectory.npz (+ ../task_info.json)
        data/processed/grab/smplx/{index.json, splits.json}   (splits = GRAB's official object split)
        data/processed/grab/assets/objects/<obj>/mesh/<obj>.obj  (object-frame mesh in m, as is — the GRAB
        contact meshes are 20-54k vertices, already lighter than ParaHome's "simplified" scans)

Run (env_isaaclab: smplx + torch + gear_sonic + trimesh):
    python scripts/process_dataset/dataset/grab.py                 # all 1335 sequences
    python scripts/process_dataset/dataset/grab.py --subjects s1   # subset
    python scripts/process_dataset/dataset/grab.py --limit 3       # smoke test
    python scripts/process_dataset/dataset/grab.py --no_trim       # keep the whole capture (T-pose to T-pose)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot
from gear_sonic.trl.utils.torch_transform import angle_axis_to_quaternion
from scipy.spatial.transform import Rotation

import dataset_paths
from parahome import N_SMPLX_JOINTS, SMPLX_FINGERTIP_VIDS, _ensure_quat_continuity  # same vertices/joints

SRC_FPS = 120.0
STRIDE = 4
REF_DT = STRIDE / SRC_FPS   # 1/30 s
CLASS = "single_rigid"
# Every clip is turned this much about the world z axis through the origin (human, object and table together;
# gravity is along z, so the scene is physically the same). GRAB subjects all face nearly the same way
# (retargeted robot root yaw -74..-91 deg), which puts a forward-reaching hand right at the singularity of the
# floating hand's YZX wrist chain (middle angle +-90 deg): 12.9% of hand-frames had det(J) < 0.2 on the 8
# benchmark clips, vs 0.1% on ParaHome. At -45 deg it is 0.4% (measured 2026-09-25 on the retargeted palm poses).
WORLD_YAW_DEG = -45.0
_RZ = Rotation.from_euler("z", WORLD_YAW_DEG, degrees=True)
# GRAB captures start in a T-pose, reach, act, return and (mostly) end in a T-pose: before the first and after
# the last hand contact there is 1.5 s / 2.0 s of no interaction (median, 42% of a clip). Clips are trimmed to
# the hand-contact window plus TRIM_MARGIN_S on each side, from GRAB's own per-vertex object contact labels
# (value = touching body part, tools/utils.py contact_ids). Measured on all 1335 (2026-09-26): non-hand contact
# (lips when drinking, ...) never extends the window, the 0.1 s persistence changes it in 4% of clips, the object
# never moves before the trimmed start, and a T-pose is left at the start of 1.3% / end of 3.8% of clips.
HAND_CONTACT_IDS = np.array([21, 22] + list(range(26, 56)))   # L/R_Hand + every finger segment
CONTACT_PERSIST = 12        # 120 FPS frames (0.1 s) a contact must last to count
TRIM_MARGIN_S = 0.5
# GRAB's official object split (grab/grab_preprocessing.py); every other object is train.
SPLIT_OBJECTS = {"test": ["mug", "wineglass", "camera", "binoculars", "fryingpan", "toothpaste"],
                 "val": ["apple", "toothbrush", "elephant", "hand"]}
SMPLX_CFG = {
    "model_type": "smplx",
    "flat_hand_mean": True,
    "use_pca": False,
    "shape": "smplx_v_template (subject template mesh, zero betas); no smplx_betas",
    "num_expression_coeffs": 10,
    "hand_pose_layout": "left15_then_right15 (axis-angle, [:45]=left [45:]=right)",
    "model_dir": "models_smplx_v1_1/models",  # relative to repo root
}

_RAW = dataset_paths.DATA_DIR / "raw" / "grab" / "data"
_OUT = dataset_paths.processed_root("grab") / "smplx"
_ASSET_REL = "grab/assets/objects"  # relative to processed/, referenced by task_info
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_models: dict[str, smplx.SMPLX] = {}
_rest_pelvis: dict[str, np.ndarray] = {}   # subject -> rest-pose pelvis joint J0 (3,)


def _split_of(obj: str) -> str:
    return next((s for s, objs in SPLIT_OBJECTS.items() if obj in objs), "train")


def _model(subject: str, gender: str, vtemp_rel: str):
    """SMPL-X with the subject's template mesh (one per subject; gender is fixed per subject)."""
    if subject not in _models:
        vt = np.asarray(trimesh.load(str(_RAW / vtemp_rel), process=False).vertices, np.float32)
        _models[subject] = smplx.create(
            str(dataset_paths.SMPLX_MODEL_DIR), model_type="smplx", gender=gender, use_pca=False,
            flat_hand_mean=True, num_betas=20, num_expression_coeffs=10, ext="npz", batch_size=1,
            v_template=vt).to(_DEV)
        with torch.no_grad():   # zero pose, zero transl -> joint 0 is the rest pelvis J0 the root rotates about
            _rest_pelvis[subject] = _models[subject]().joints[0, 0].cpu().numpy().astype(np.float64)
    return _models[subject]


def _fk(model, fullpose: np.ndarray, transl: np.ndarray):
    """SMPL-X FK -> (fingertip pads (F,10,3), joints (F,55,3)). Face/expression zeroed as in parahome.py."""
    F = len(fullpose)
    pads = np.empty((F, 10, 3), np.float32)
    jts = np.empty((F, N_SMPLX_JOINTS, 3), np.float32)
    z = lambda n, d: torch.zeros(n, d, device=_DEV)  # noqa: E731
    with torch.no_grad():
        for s in range(0, F, 2048):
            e = min(s + 2048, F)
            n = e - s
            fp = torch.as_tensor(fullpose[s:e], device=_DEV)
            o = model(betas=z(n, 20), global_orient=fp[:, :3], body_pose=fp[:, 3:66],
                      left_hand_pose=fp[:, 75:120], right_hand_pose=fp[:, 120:165],
                      transl=torch.as_tensor(transl[s:e], device=_DEV),
                      expression=z(n, 10), jaw_pose=z(n, 3), leye_pose=z(n, 3), reye_pose=z(n, 3))
            pads[s:e] = o.vertices[:, SMPLX_FINGERTIP_VIDS].cpu().numpy()
            jts[s:e] = o.joints[:, :N_SMPLX_JOINTS].cpu().numpy()
    return pads, jts


def _pose7(transl: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """(F,7) pos + quat wxyz of a GRAB ObjectModel pose, turned by _RZ. GRAB computes v @ R + t -> world R^T."""
    q = (_RZ * Rotation.from_rotvec(rotvec).inv()).as_quat()  # xyzw
    return _ensure_quat_continuity(np.concatenate([_RZ.apply(transl), q[:, [3, 0, 1, 2]]], axis=1))


def _contact_window(obj_contact: np.ndarray, margin_s: float):
    """(first, last, start, end) 120 FPS frames of the hand-contact window, or None without hand contact."""
    hand = np.isin(obj_contact, HAND_CONTACT_IDS).any(axis=1)
    run = np.convolve(hand.astype(np.int32), np.ones(CONTACT_PERSIST, np.int32), "valid") == CONTACT_PERSIST
    if not run.any():
        return None
    first = int(np.argmax(run))
    last = int(len(run) - 1 - np.argmax(run[::-1])) + CONTACT_PERSIST - 1
    m = int(round(margin_s * SRC_FPS))
    return first, last, max(first - m, 0), min(last + m, len(hand) - 1)


def _root_transform(global_orient: np.ndarray, pelvis: np.ndarray) -> np.ndarray:
    """(F,4,4) pelvis pose in the ParaHome body_global_transform convention (upright, SMPL base removed)."""
    q = remove_smpl_base_rot(angle_axis_to_quaternion(torch.as_tensor(global_orient)), w_last=False).numpy()
    T = np.tile(np.eye(4, dtype=np.float32), (len(q), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    T[:, :3, 3] = pelvis
    return T


def _export_mesh(name: str, src_rel: str) -> None:
    dst = dataset_paths.object_mesh("grab", name)
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    trimesh.load(str(_RAW / src_rel), process=False).export(str(dst))


def process_sequence(path: Path, overwrite: bool, margin_s: float | None) -> dict:
    d = np.load(path, allow_pickle=True)
    subject, seq, obj = str(d["sbj_id"]), path.stem, str(d["obj_name"])
    intent, gender = str(d["motion_intent"]), str(d["gender"])
    clip = f"{subject}_{seq}"
    n_src = int(d["n_frames"])
    win = _contact_window(d["contact"].item()["object"], margin_s) if margin_s is not None else None
    start, end = (win[2], win[3]) if win else (0, n_src - 1)
    fidx = np.arange(start, end + 1, STRIDE)
    entry = {"clip": clip, "class": CLASS, "seq": seq, "subject": subject, "seg_idx": 0,
             "frame_range": [start, end], "num_frames": len(fidx), "action_text": intent,
             "objects": [obj], "articulated": [], "rigid": [obj], "split": _split_of(obj)}
    out_dir = _OUT / CLASS / clip / "0"
    if (out_dir / "trajectory.npz").exists() and not overwrite:
        return entry

    body, ob, tb = d["body"].item(), d["object"].item(), d["table"].item()
    fullpose = body["params"]["fullpose"][fidx].astype(np.float32)   # (F,165)
    transl = body["params"]["transl"][fidx].astype(np.float32)
    model = _model(subject, gender, body["vtemp"])
    # Turn the body by _RZ about the world origin. SMPL-X rotates the root about the rest pelvis J0:
    #   world = R (v - J0) + J0 + t  ->  R' = Rz R,  t' = Rz (J0 + t) - J0
    j0 = _rest_pelvis[subject]
    fullpose[:, :3] = (_RZ * Rotation.from_rotvec(fullpose[:, :3])).as_rotvec().astype(np.float32)
    transl = (_RZ.apply(j0 + transl) - j0).astype(np.float32)
    pads, joints = _fk(model, fullpose, transl)
    _export_mesh(obj, ob["object_mesh"])
    _export_mesh("table", tb["table_mesh"])

    arrays = {
        "smplx_body_pose": fullpose[:, 3:66],                     # (F,63)
        "smplx_global_orient": fullpose[:, :3],                   # (F,3)
        "smplx_transl": transl,                                   # (F,3)
        "smplx_hand_pose": fullpose[:, 75:165],                   # (F,90) left45 + right45
        "smplx_v_template": model.v_template.cpu().numpy().astype(np.float32),   # (10475,3)
        "smplx_joints": joints,                                   # (F,55,3)
        "fingertip_pad_pos": pads,                                # (F,10,3) L[th,ff,mf,rf,lf] + R[...]
        "body_global_transform": _root_transform(fullpose[:, :3], joints[:, 0]),   # (F,4,4)
        "root_transl": joints[:, 0].copy(),                       # (F,3) pelvis world
        "frame_indices": fidx,                                    # (F,) 120 FPS source frames
        f"obj__{obj}__base": _pose7(ob["params"]["transl"][fidx], ob["params"]["global_orient"][fidx]),
        "ctx__table__base": _pose7(tb["params"]["transl"][fidx], tb["params"]["global_orient"][fidx]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_dir / "trajectory.npz"), **arrays)
    task_info = {
        "task": clip, "dataset_name": "grab", "source_repr": "smplx",
        "class": CLASS, "seq": seq, "subject": subject, "seg_idx": 0,
        "frame_range": [start, end], "num_frames": len(fidx), "source_fps": SRC_FPS, "stride": STRIDE,
        "source_num_frames": n_src,
        "trim": ({"contact_first": win[0], "contact_last": win[1], "margin_s": margin_s,
                  "source": "GRAB object contact labels, hand parts, >= 0.1 s"} if win else None),
        "action_text": intent, "motion_intent": intent, "gender": gender, "ref_dt": REF_DT,
        "world_yaw_deg": WORLD_YAW_DEG,
        "smplx_cfg": SMPLX_CFG,
        "manip_objects": [{"name": obj, "is_articulated": False, "parts": []}],
        "context_objects": ["table"],
        "object_asset_dir": {obj: f"{_ASSET_REL}/{obj}", "table": f"{_ASSET_REL}/table"},
        "split": _split_of(obj),
    }
    with open(out_dir.parent / "task_info.json", "w") as f:
        json.dump(task_info, f, indent=2)
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Process GRAB into per-sequence clips (ParaHome layout).")
    ap.add_argument("--subjects", nargs="*", default=None, help="Subset of subjects (e.g. s1 s2).")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N sequences.")
    ap.add_argument("--overwrite", action="store_true", help="Re-process existing clips.")
    ap.add_argument("--trim_margin", type=float, default=TRIM_MARGIN_S,
                    help="Seconds kept before the first / after the last hand contact.")
    ap.add_argument("--no_trim", action="store_true", help="Keep the whole capture.")
    args = ap.parse_args()

    subjects = sorted((p.name for p in (_RAW / "grab").glob("s*")), key=lambda s: int(s[1:]))
    if args.subjects:
        subjects = [s for s in subjects if s in set(args.subjects)]
    paths = [p for s in subjects for p in sorted((_RAW / "grab" / s).glob("*.npz"))]
    if args.limit > 0:
        paths = paths[: args.limit]

    _OUT.mkdir(parents=True, exist_ok=True)
    manifest, n_fail = [], 0
    for i, p in enumerate(paths, 1):
        try:
            manifest.append(process_sequence(p, args.overwrite, None if args.no_trim else args.trim_margin))
            if i % 50 == 0 or i == len(paths):
                print(f"[ok] {i}/{len(paths)} {p.parent.name}/{p.stem}", flush=True)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[error] {p.parent.name}/{p.stem}: {e}", flush=True)

    split_counts = Counter(e["split"] for e in manifest)
    with open(_OUT / "index.json", "w") as f:
        json.dump({"clips": manifest, "class_counts": dict(Counter(e["class"] for e in manifest))}, f, indent=2)
    with open(_OUT / "splits.json", "w") as f:
        json.dump({"clip2split": {e["clip"]: e["split"] for e in manifest},
                   "object2split": {o: _split_of(o) for o in sorted({e["objects"][0] for e in manifest})},
                   "source": "GRAB official object split (grab/grab_preprocessing.py)"}, f, indent=2)
    print(f"\nDone. clips={len(manifest)} fail={n_fail}  split counts: {dict(split_counts)}")


if __name__ == "__main__":
    main()
