"""Process the ParaHome dataset (CVPR 2025) into per-action-segment clips for
whole-body loco-manipulation.

ParaHome sequences are long (min 1149, max 6614 frames @30 FPS) multi-object
activities. `object_in_scene.json` marks almost every object present in every
sequence, so it is useless for classification. Instead we segment each sequence
by its `text_annotation.json` action intervals and classify each *segment* by the
objects it actually MANIPULATES (detected from motion), which is the only level at
which the single/multi and articulated/rigid distinctions are meaningful.

Human representation:  SMPL-X params (data/smplx_seq/sN/) — the standard input for
retargeting to a parametric humanoid. Custom global joints (joint_positions.pkl)
are ALSO saved as auxiliary tracking targets. Object motion / articulation comes
from the custom stream (object_transformations, joint_states, joint_info) — the
only source, since SMPL-X carries no objects.

Input (symlink data/raw/parahome -> ParaHome repo):
    data/raw/parahome/data/seq/sN/{text_annotation,object_transformations,
        joint_states,joint_positions,body_global_transform}.{json,pkl}
    data/raw/parahome/data/smplx_seq/sN/{smplx_pose,smplx_params}.pkl
    data/raw/parahome/data/{joint_info.pkl, metadata.json}

Output:
    data/processed/parahome/smplx/{class}/{clip}/0/trajectory.npz
    data/processed/parahome/smplx/{class}/{clip}/task_info.json
    data/processed/parahome/smplx/index.json      (manifest of all clips)
    data/processed/parahome/smplx/splits.json     (subject-disjoint train/val/test)
    (object meshes -> data/processed/parahome/assets/objects/ handled by
     parahome_convert_obj_to_usd.py, Phase B.)

Run:
    python scripts/process_dataset/dataset/parahome.py                 # all 207 sequences
    python scripts/process_dataset/dataset/parahome.py --seq s1 s2     # subset
    python scripts/process_dataset/dataset/parahome.py --limit 3       # first 3 (smoke test)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REF_DT = 1.0 / 30.0  # ParaHome is captured at 30 FPS

# Articulation model constants (from data/scan + joint_info.pkl; see dataset CLAUDE.md).
ARTICULATED_OBJECTS = {
    "drawer", "sink", "refrigerator", "gasstove",
    "laptop", "microwave", "trashbin", "washingmachine",
}
PRISMATIC_PARTS = {"drawer_part1", "drawer_part2"}  # all other movable parts are revolute

# Motion-detection thresholds for "was this object manipulated in this segment?"
RIGID_DISP_THRESH = 0.05    # m — rigid base displacement over the segment
REVOLUTE_THRESH = 0.15      # rad — revolute joint-state change
PRISMATIC_THRESH = 0.02     # m — prismatic joint-state change

# SMPL-X fit config used by ParaHome (recorded into task_info for downstream FK).
SMPLX_CFG = {
    "model_type": "smplx",
    "flat_hand_mean": True,
    "use_pca": False,
    "num_betas": 20,
    "num_expression_coeffs": 10,
    "hand_pose_layout": "left15_then_right15 (axis-angle, [:45]=left [45:]=right)",
    "model_dir": "models_smplx_v1_1/models",  # relative to repo root
}

_SCRIPT_DIR = Path(__file__).resolve().parents[3]
_DATA_DIR = _SCRIPT_DIR / "source" / "robotis_sh5" / "data"
_RAW = _DATA_DIR / "raw" / "parahome" / "data"
_OUT = _DATA_DIR / "processed" / "parahome" / "smplx"
_ASSET_REL = "parahome/assets/objects"  # relative to processed/, referenced by task_info

# --- SMPL-X fingertip PAD vertices (contact-accurate fingertip targets) ------------
# smplx built-in fingertip landmark vertex ids on the 10475-vert SMPL-X mesh, ordered to
# match the robot fingertip_body_names: LEFT [thumb,index,middle,ring,little] then RIGHT.
# VERIFIED: these mesh vertices coincide with the ParaHome hand fingertips to ~4mm and are
# in the SAME world frame as joint_positions (so they mix cleanly with the body keypoints).
# Requires the `smplx` package + torch (runs FK); gracefully skipped if unavailable.
SMPLX_FINGERTIP_VIDS = [5361, 4933, 5058, 5169, 5286,   # left  thumb,index,middle,ring,pinky
                        8079, 7669, 7794, 7905, 8022]   # right thumb,index,middle,ring,pinky
_SMPLX_MODEL_DIR = _SCRIPT_DIR / "models_smplx_v1_1" / "models"

try:
    import smplx as _smplx
    _HAS_SMPLX = True
except Exception:
    _HAS_SMPLX = False
_FK_DEVICE = "cuda" if (_HAS_SMPLX and torch.cuda.is_available()) else "cpu"
_smplx_model_cache: dict = {}


def _get_smplx_model(gender: str):
    if gender not in _smplx_model_cache:
        _smplx_model_cache[gender] = _smplx.create(
            str(_SMPLX_MODEL_DIR), model_type="smplx", gender=gender, use_pca=False,
            flat_hand_mean=True, num_betas=20, num_expression_coeffs=10, ext="npz",
            batch_size=1,
        ).to(_FK_DEVICE)
    return _smplx_model_cache[gender]


def _seq_fingertip_pad_pos(smplx_pose: dict, betas_np: np.ndarray, gender: str):
    """Per-frame world positions of the 10 SMPL-X fingertip pad vertices → (F,10,3), or
    None if smplx is unavailable (pipeline still runs, just omits fingertip_pad_pos)."""
    if not _HAS_SMPLX:
        return None
    model = _get_smplx_model(gender)
    go = smplx_pose["global_orient"].to(_FK_DEVICE).float()
    bp = smplx_pose["body_pose"].to(_FK_DEVICE).float()
    hp = smplx_pose["hand_pose"].to(_FK_DEVICE).float()
    tr = smplx_pose["transl"].to(_FK_DEVICE).float()
    F = go.shape[0]
    betas = torch.as_tensor(betas_np, dtype=torch.float32, device=_FK_DEVICE).reshape(1, -1)
    out = np.empty((F, 10, 3), dtype=np.float32)
    CHUNK = 2048
    z = lambda n, d: torch.zeros(n, d, device=_FK_DEVICE)  # noqa: E731
    with torch.no_grad():
        for s in range(0, F, CHUNK):
            e = min(s + CHUNK, F)
            n = e - s
            # Pass face/expression components explicitly (model built with batch_size=1, so
            # its defaults would mismatch the chunk batch dim).
            o = model(betas=betas.expand(n, -1), global_orient=go[s:e], body_pose=bp[s:e],
                      left_hand_pose=hp[s:e, :45], right_hand_pose=hp[s:e, 45:], transl=tr[s:e],
                      expression=z(n, 10), jaw_pose=z(n, 3), leye_pose=z(n, 3), reye_pose=z(n, 3))
            out[s:e] = o.vertices[:, SMPLX_FINGERTIP_VIDS, :].detach().cpu().numpy()
    return out


# ----------------------------------------------------------------------------- #
# IO helpers
# ----------------------------------------------------------------------------- #
def _load_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _mat_to_pos_quat(M: np.ndarray) -> np.ndarray:
    """(4,4) SE(3) -> (7,) [pos(3), quat wxyz(4)]."""
    pos = M[:3, 3]
    q_xyzw = Rotation.from_matrix(M[:3, :3]).as_quat()
    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    return np.concatenate([pos, q_wxyz])


def _ensure_quat_continuity(qpos: np.ndarray) -> np.ndarray:
    """qpos: (F,7) with quat in cols 3:7 (wxyz). Flip sign to avoid double-cover jumps."""
    out = qpos.copy()
    for i in range(1, len(out)):
        if np.dot(out[i, 3:], out[i - 1, 3:]) < 0:
            out[i, 3:] = -out[i, 3:]
    return out


def _obj_traj(obj_transf: dict, frames: list[int], instance: str) -> np.ndarray:
    """Per-frame (F,7) pos+quat(wxyz) for one instance ('{object}_{base|part1|part2}')."""
    out = np.zeros((len(frames), 7), dtype=np.float64)
    for i, f in enumerate(frames):
        out[i] = _mat_to_pos_quat(np.asarray(obj_transf[f][instance], dtype=np.float64))
    return _ensure_quat_continuity(out)


# ----------------------------------------------------------------------------- #
# Classification (which objects are manipulated in a frame window)
# ----------------------------------------------------------------------------- #
def _manipulated_objects(
    obj_transf: dict, joint_states: dict, frames: list[int]
) -> set[str]:
    """Object names manipulated within `frames` (rigid base moved OR articulated DOF changed)."""
    if len(frames) < 2:
        return set()
    samp = frames if len(frames) <= 12 else [frames[int(i)] for i in np.linspace(0, len(frames) - 1, 12)]
    moved: set[str] = set()

    # Base translation over the window = manipulation. Checked for BOTH rigid objects
    # AND articulated objects (a carried laptop moves its base without its lid DOF
    # changing — fixtures like sink/fridge never translate >RIGID_DISP_THRESH in the data,
    # so this adds no false positives).
    pos: dict[str, list] = {}
    for f in samp:
        for inst, M in obj_transf[f].items():
            _, part = inst.rsplit("_", 1)
            if part == "base":
                pos.setdefault(inst, []).append(np.asarray(M)[:3, 3])
    for inst, ps in pos.items():
        ps = np.array(ps)
        if np.linalg.norm(ps.max(0) - ps.min(0)) > RIGID_DISP_THRESH:
            moved.add(inst.rsplit("_", 1)[0])

    # Articulated: joint-state DOF change over the window.
    for part, d in joint_states.items():
        vals = [d[f] for f in samp if f in d]
        if len(vals) < 2:
            continue
        thr = PRISMATIC_THRESH if part in PRISMATIC_PARTS else REVOLUTE_THRESH
        if (max(vals) - min(vals)) > thr:
            moved.add(part.rsplit("_", 1)[0])
    return moved


def _classify(moved: set[str]) -> str:
    if not moved:
        return "loco"
    artic = any(o in ARTICULATED_OBJECTS for o in moved)
    rigid = any(o not in ARTICULATED_OBJECTS for o in moved)
    if len(moved) == 1:
        return "single_articulated" if artic else "single_rigid"
    if artic and rigid:
        return "multi_mixed"
    return "multi_articulated" if artic else "multi_rigid"


# ----------------------------------------------------------------------------- #
# Per-sequence processing
# ----------------------------------------------------------------------------- #
def process_sequence(seq: str, joint_info: dict, overwrite: bool = False) -> list[dict]:
    """Process one sequence into action-segment clips. Returns manifest entries."""
    sdir = _RAW / "seq" / seq
    smdir = _RAW / "smplx_seq" / seq
    if not sdir.is_dir() or not smdir.is_dir():
        print(f"[warn] {seq}: missing seq/ or smplx_seq/ dir; skip")
        return []

    text_annot = json.load(open(sdir / "text_annotation.json"))
    obj_transf = _load_pkl(sdir / "object_transformations.pkl")
    joint_states = _load_pkl(sdir / "joint_states.pkl")
    joint_positions = _to_np(_load_pkl(sdir / "joint_positions.pkl"))          # (F,73,3)
    body_gt = _to_np(_load_pkl(sdir / "body_global_transform.pkl"))            # (F,4,4)
    smplx_pose = _load_pkl(smdir / "smplx_pose.pkl")
    smplx_params = _load_pkl(smdir / "smplx_params.pkl")

    body_pose = _to_np(smplx_pose["body_pose"])        # (F,63)
    global_orient = _to_np(smplx_pose["global_orient"])  # (F,3)
    transl = _to_np(smplx_pose["transl"])              # (F,3)
    hand_pose = _to_np(smplx_pose["hand_pose"])        # (F,90)
    betas = _to_np(smplx_params["beta"]).reshape(-1).astype(np.float32)  # (20,)
    gender = str(smplx_params["gender"])
    fingertip_pad_seq = _seq_fingertip_pad_pos(smplx_pose, betas, gender)  # (F,10,3) | None

    # Frame alignment: object_transformations keys are contiguous ints from 0 and
    # SMPL-X / joint_positions are frame-aligned. Use the common valid length.
    n_valid = min(len(body_pose), len(joint_positions), len(body_gt),
                  max(obj_transf.keys()) + 1)

    entries: list[dict] = []
    for seg_idx, (key, desc) in enumerate(sorted(text_annot.items(), key=lambda kv: int(kv[0].split()[0]))):
        a, b = (int(x) for x in key.split())
        b = min(b, n_valid - 1)
        frames = [f for f in range(a, b + 1) if f in obj_transf and f < n_valid]
        if len(frames) < 2:
            continue
        fidx = np.array(frames, dtype=np.int64)

        moved = _manipulated_objects(obj_transf, joint_states, frames)
        cls = _classify(moved)
        objs = sorted(moved)
        obj_tag = "-".join(objs) if objs else "loco"
        clip = f"{seq}_seg{seg_idx:02d}_{obj_tag}"
        out_dir = _OUT / cls / clip / "0"
        out_npz = out_dir / "trajectory.npz"
        if out_npz.exists() and not overwrite:
            entries.append(_manifest_entry(seq, seg_idx, cls, clip, a, b, len(frames), desc, objs, joint_info))
            continue

        # --- SMPL-X (sliced to the clip's frames) ---
        arrays: dict[str, np.ndarray] = dict(
            smplx_body_pose=body_pose[fidx].astype(np.float32),        # (F,63)
            smplx_global_orient=global_orient[fidx].astype(np.float32),  # (F,3)
            smplx_transl=transl[fidx].astype(np.float32),              # (F,3)
            smplx_hand_pose=hand_pose[fidx].astype(np.float32),        # (F,90)
            smplx_betas=betas,                                         # (20,)
            # aux custom streams
            joint_positions=joint_positions[fidx].astype(np.float32),  # (F,73,3)
            body_global_transform=body_gt[fidx].astype(np.float32),    # (F,4,4)
            root_transl=body_gt[fidx, :3, 3].astype(np.float32),       # (F,3) pelvis world
            frame_indices=fidx,                                        # (F,)
        )
        if fingertip_pad_seq is not None:
            # (F,10,3) SMPL-X fingertip pad vertices — LEFT[th,ff,mf,rf,lf] then RIGHT.
            arrays["fingertip_pad_pos"] = fingertip_pad_seq[fidx].astype(np.float32)

        # --- Manipulated objects (6-DoF trajectories + articulation) ---
        obj_records = []
        for obj in objs:
            is_art = obj in ARTICULATED_OBJECTS
            base_inst = f"{obj}_base"
            if base_inst not in obj_transf[frames[0]]:
                continue
            arrays[f"obj__{obj}__base"] = _obj_traj(obj_transf, frames, base_inst)  # (F,7)
            rec = {"name": obj, "is_articulated": is_art, "parts": []}
            if is_art:
                for part in sorted(joint_info.get(obj, {}).keys()):
                    inst = f"{obj}_{part}"
                    if inst not in obj_transf[frames[0]]:
                        continue
                    arrays[f"obj__{obj}__{part}"] = _obj_traj(obj_transf, frames, inst)  # (F,7)
                    # joint DOF trajectory (radian for revolute, meter for prismatic)
                    js = joint_states.get(inst, {})
                    dof = np.array([js.get(f, np.nan) for f in frames], dtype=np.float64)
                    arrays[f"dof__{obj}__{part}"] = dof                       # (F,)
                    info = joint_info[obj][part]
                    piv = np.asarray(info["pivot"], dtype=np.float64).reshape(-1)
                    ax = np.asarray(info["axis"], dtype=np.float64).reshape(-1)
                    ax = ax / (np.linalg.norm(ax) + 1e-12)  # joint_info axes are NOT all unit
                    jtype = "prismatic" if inst in PRISMATIC_PARTS else "revolute"
                    rec["parts"].append({
                        "part": part, "joint_type": jtype,
                        "axis": ax.tolist(),
                        # prismatic pivot is a shape-(1,) TYPE MARKER, not a 3D point → null
                        "pivot": (piv.tolist() if (jtype == "revolute" and piv.shape[0] == 3) else None),
                    })
            obj_records.append(rec)

        # --- Context / support objects (non-manipulated scene objects) ---
        # Base pose (F,7) of every scene object present in the clip that is NOT manipulated.
        # These are the static furniture / props (support surfaces, placement targets, collision
        # geometry) the active object rests on or is handled near. The env spawns a subset of them
        # (by XY proximity to the active object's trajectory) as KINEMATIC-frozen colliders so the
        # dynamic active object has something to rest on instead of falling to the floor. Base only
        # (articulation DOF is meaningful only for the manipulated set); the instance set is constant
        # across frames, so frames[0] enumerates all. Stored as ctx__<obj>__base to keep them cleanly
        # separate from the active obj__<obj>__base (so the env's "first obj__ = active" stays valid).
        ctx_names = []
        for inst in sorted(obj_transf[frames[0]].keys()):
            obj_c, part_c = inst.rsplit("_", 1)
            if part_c != "base" or obj_c in moved:
                continue
            arrays[f"ctx__{obj_c}__base"] = _obj_traj(obj_transf, frames, inst)   # (F,7)
            ctx_names.append(obj_c)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_npz), **arrays)

        task_info = {
            "task": clip, "dataset_name": "parahome", "source_repr": "smplx",
            "class": cls, "seq": seq, "seg_idx": seg_idx,
            "frame_range": [a, b], "num_frames": len(frames),
            "action_text": desc, "gender": gender, "ref_dt": REF_DT,
            "smplx_cfg": SMPLX_CFG,
            "manip_objects": obj_records,
            "context_objects": ctx_names,
            "object_asset_dir": {
                **{r["name"]: f"{_ASSET_REL}/{r['name']}" for r in obj_records},
                **{n: f"{_ASSET_REL}/{n}" for n in ctx_names},
            },
        }
        with open(out_dir.parent / "task_info.json", "w") as f:
            json.dump(task_info, f, indent=2)

        entries.append(_manifest_entry(seq, seg_idx, cls, clip, a, b, len(frames), desc, objs, joint_info))

    return entries


def _manifest_entry(seq, seg_idx, cls, clip, a, b, nf, desc, objs, joint_info) -> dict:
    return {
        "clip": clip, "class": cls, "seq": seq, "seg_idx": seg_idx,
        "frame_range": [a, b], "num_frames": nf, "action_text": desc,
        "objects": objs,
        "articulated": [o for o in objs if o in ARTICULATED_OBJECTS],
        "rigid": [o for o in objs if o not in ARTICULATED_OBJECTS],
    }


# ----------------------------------------------------------------------------- #
# Splits (subject-disjoint, from metadata.json)
# ----------------------------------------------------------------------------- #
def build_splits() -> dict:
    """Subject-disjoint train/val/test (≈80/10/10). Returns {seq: split} + subject map."""
    meta = json.load(open(_RAW / "metadata.json"))  # {subject: [seq,...]}
    subjects = sorted(meta.keys(), key=lambda p: int(p[1:]))
    seq2split, subj2split = {}, {}
    for i, subj in enumerate(subjects):
        # deterministic interleave: every 10th subject -> test, next -> val, else train
        r = i % 10
        split = "test" if r == 0 else ("val" if r == 1 else "train")
        subj2split[subj] = split
        for s in meta[subj]:
            seq2split[s] = split
    return {"seq2split": seq2split, "subject2split": subj2split}


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Process ParaHome into action-segment clips.")
    ap.add_argument("--seq", nargs="*", default=None, help="Subset of sequences (e.g. s1 s2).")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N sequences.")
    ap.add_argument("--overwrite", action="store_true", help="Re-process existing clips.")
    args = ap.parse_args()

    joint_info = _load_pkl(_RAW / "joint_info.pkl")
    splits = build_splits()

    all_seqs = sorted((_RAW / "seq").glob("s*"), key=lambda p: int(p.name[1:]))
    seq_names = [p.name for p in all_seqs]
    if args.seq:
        seq_names = [s for s in seq_names if s in set(args.seq)]
    if args.limit > 0:
        seq_names = seq_names[: args.limit]

    _OUT.mkdir(parents=True, exist_ok=True)
    manifest, n_ok, n_fail = [], 0, 0
    for seq in seq_names:
        try:
            entries = process_sequence(seq, joint_info, overwrite=args.overwrite)
            for e in entries:
                e["split"] = splits["seq2split"].get(seq, "train")
            manifest.extend(entries)
            n_ok += 1
            print(f"[ok] {seq}: {len(entries)} clips")
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[error] {seq}: {e}")
            import traceback; traceback.print_exc()

    # Manifest + splits (only rewrite when processing the full set or explicitly)
    from collections import Counter
    cls_counts = Counter(e["class"] for e in manifest)
    with open(_OUT / "index.json", "w") as f:
        json.dump({"clips": manifest, "class_counts": dict(cls_counts)}, f, indent=2)
    with open(_OUT / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nDone. sequences ok={n_ok} fail={n_fail}, clips={len(manifest)}")
    print("class counts:", dict(cls_counts))


if __name__ == "__main__":
    main()
