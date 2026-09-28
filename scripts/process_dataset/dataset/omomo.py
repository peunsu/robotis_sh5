"""Process the OMOMO dataset (Li et al., SIGGRAPH Asia 2023) into per-sequence clips in the ParaHome layout.

OMOMO: 17 subjects x 15 large objects (boxes, tables, chairs, lamp, mop, ...), 5882 sequences @30 FPS, the
subject walking, lifting, carrying and putting down ONE object on a flat floor (z = 0). Each sequence becomes
ONE clip. The data is the authors' released `*_diffusion_manip_seq_joints24.p` (the same files the xGMR
convert_omomo_to_smplx.py reads): SMPL-X body params + per-frame object scale / rotation / translation.

Written so every consumer of parahome.py output reads it unchanged (env: dataset_root -> processed/omomo;
per-clip scripts: --dataset omomo). Differences from ParaHome and how they are handled:
  * Fingers were not captured -> `smplx_hand_pose` is all zeros (flat hands: flat_hand_mean=True, as the
    OMOMO BodyModel, which adds no hand mean). `fingertip_pad_pos` is therefore the pad of a flat hand.
  * 16 betas (OMOMO's num_betas) -> `smplx_betas` is padded with zeros to the ParaHome 20 (the extra shape
    directions get weight 0, so FK is identical; consumers build num_betas=20 models). With the repo's
    SMPL-X v1.1 the rest pelvis matches OMOMO's trans2joint and the rest joint offsets match rest_offsets
    to < 1 mm (checked on sub16/sub17, 2026-09-28).
  * Object vertices are `s_t * R_t @ v + t_t` with a per-frame scale s_t on a raw scan in arbitrary units.
    The scale is baked into the exported mesh as ONE value per object part (median over every frame of the
    dataset; each sequence's median is within +-2% of it, 99% within +-1%) and the mesh is re-centred on its
    vertex centroid, so `obj__<obj>__base` = (s_t R_t c + t_t, R_t) is OMOMO's own obj_com_pos + rotation
    and the residual scale error is a symmetric shrink/grow about the centre. Per clip the worst frame is
    recorded as object_scale_max_rel_dev (median 0.35%, p99 2%, a few scale glitches up to 12%); vertices
    match OMOMO's formula to 1-3 mm (median of per-clip max, 400 clips).
  * OMOMO's own object poses sink the mesh into the floor at frame 0 (lowest vertex median -1.3 cm, 24% of
    clips below -2 cm, worst -4.9 cm). Recorded as object_min_z_frame0, NOT corrected here.
  * Frames with scale 0 are tracking drop-outs (identity rotation, zero translation). The 39 sequences that
    have any in any part (22 whole-object, 17 more mop/vacuum bottom; 2-105 frames each) are skipped and
    listed in index.json.
  * mop and vacuum are two scanned parts (handle = `base`, floor head = `bottom`) joined at a point that
    stays fixed in both parts to ~1-2 mm; the vacuum's relative rotation is essentially one hinge axis, the
    mop's two. They are class `single_articulated` with both part poses and the measured joint in
    task_info, but no `dof__` trajectory (not a single revolute/prismatic DOF) and no articulated USD yet.
  * No ParaHome Xsens stream: `joint_positions` is not written. `body_global_transform` is the SMPL-X pelvis
    joint + remove_smpl_base_rot(global_orient), as in grab.py.
  * No context objects (the only support is the floor). Clips are not trimmed: OMOMO sequences are already
    segmented around one interaction (median 0.6 s before the object first moves, 1.3 s after it stops) and
    there are no contact labels. Subjects face every direction, so no scene rotation (cf. grab.py).
  * Splits: OMOMO's subject split (train file = sub1-15, test file = sub16-17) as `split`, and its
    unseen-object protocol (hand_foot_dataset.py train_objects / test_objects) as `object_split`.
  * `action_text` is the sequence's OMOMO text annotation (omomo_text_anno.zip, 4913 of 5882) or "".

Input : data/raw/omomo/data/{train,test}_diffusion_manip_seq_joints24.p
        data/raw/omomo/data/captured_objects/<obj>_cleaned_simplified[_top|_bottom].obj
        data/raw/omomo/omomo_text_anno.zip
Output: data/processed/omomo/smplx/{single_rigid,single_articulated}/<subN>_<obj>_<idx>/0/trajectory.npz
        (+ ../task_info.json), data/processed/omomo/smplx/{index.json, splits.json}
        data/processed/omomo/assets/objects/<obj>/mesh/<obj>.obj   (object frame, m; two-part: the handle)
        data/processed/omomo/assets/objects/<obj>/mesh/<obj>_bottom.obj   (two-part only)

Run (env_isaaclab: smplx + torch + gear_sonic + trimesh + joblib):
    python scripts/process_dataset/dataset/omomo.py                        # all sequences
    python scripts/process_dataset/dataset/omomo.py --subjects sub16       # subset
    python scripts/process_dataset/dataset/omomo.py --objects largebox     # subset
    python scripts/process_dataset/dataset/omomo.py --limit 3              # smoke test
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation

import dataset_paths
from grab import _root_transform  # SMPL-X pelvis -> ParaHome body_global_transform convention
from parahome import N_SMPLX_JOINTS, SMPLX_FINGERTIP_VIDS, _ensure_quat_continuity  # same vertices/joints

SRC_FPS = 30.0
REF_DT = 1.0 / SRC_FPS
N_BETAS_SRC = 16          # OMOMO BodyModel num_betas
N_BETAS_OUT = 20          # ParaHome layout
TWO_PART = ("mop", "vacuum")
# OMOMO's unseen-object protocol (manip/data/hand_foot_dataset.py).
OBJECT_SPLIT = {"train": ["largetable", "woodchair", "plasticbox", "largebox", "smallbox", "trashcan", "monitor",
                          "floorlamp", "clothesstand", "vacuum"],
                "test": ["smalltable", "whitechair", "suitcase", "tripod", "mop"]}
SMPLX_CFG = {
    "model_type": "smplx",
    "flat_hand_mean": True,
    "use_pca": False,
    "num_betas": N_BETAS_OUT,
    "shape": f"smplx_betas: OMOMO's {N_BETAS_SRC} betas zero-padded to {N_BETAS_OUT}",
    "hand_pose": "not captured: zeros (flat hands)",
    "num_expression_coeffs": 10,
    "hand_pose_layout": "left15_then_right15 (axis-angle, [:45]=left [45:]=right)",
    "model_dir": "models_smplx_v1_1/models",  # relative to repo root
}

_RAW = dataset_paths.DATA_DIR / "raw" / "omomo"
_OUT = dataset_paths.processed_root("omomo") / "smplx"
_ASSET_REL = "omomo/assets/objects"  # relative to processed/, referenced by task_info
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_models: dict[str, smplx.SMPLX] = {}


def _object_split_of(obj: str) -> str:
    return next(s for s, objs in OBJECT_SPLIT.items() if obj in objs)


def _model(gender: str):
    if gender not in _models:
        _models[gender] = smplx.create(
            str(dataset_paths.SMPLX_MODEL_DIR), model_type="smplx", gender=gender, use_pca=False,
            flat_hand_mean=True, num_betas=N_BETAS_OUT, num_expression_coeffs=10, ext="npz",
            batch_size=1).to(_DEV)
    return _models[gender]


def _fk(model, betas: np.ndarray, go: np.ndarray, bp: np.ndarray, transl: np.ndarray):
    """SMPL-X FK -> (fingertip pads (F,10,3), joints (F,55,3)). Hands/face/expression zeroed."""
    F = len(go)
    z = lambda d: torch.zeros(F, d, device=_DEV)  # noqa: E731
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=_DEV)  # noqa: E731
    with torch.no_grad():   # OMOMO sequences are <= 652 frames: one batch
        o = model(betas=T(betas)[None].expand(F, -1), global_orient=T(go), body_pose=T(bp), transl=T(transl),
                  left_hand_pose=z(45), right_hand_pose=z(45),
                  expression=z(10), jaw_pose=z(3), leye_pose=z(3), reye_pose=z(3))
    return (o.vertices[:, SMPLX_FINGERTIP_VIDS].cpu().numpy(),
            o.joints[:, :N_SMPLX_JOINTS].cpu().numpy())


def _parts(obj: str) -> dict[str, tuple[str, str]]:
    """part -> (raw mesh file suffix, sequence-key prefix)."""
    if obj in TWO_PART:
        return {"base": ("_top", "obj_"), "bottom": ("_bottom", "obj_bottom_")}
    return {"base": ("", "obj_")}


def _raw_mesh(obj: str, suffix: str) -> trimesh.Trimesh:
    return trimesh.load(str(_RAW / "data" / "captured_objects" / f"{obj}_cleaned_simplified{suffix}.obj"),
                        process=False, force="mesh")


def _valid(s: dict) -> bool:
    keys = ["obj_scale"] + (["obj_bottom_scale"] if "obj_bottom_scale" in s else [])
    return all((s[k] > 1e-6).all() for k in keys)


class ObjectGeometry:
    """Per object part: the dataset-median scale and the raw vertex centroid (the exported mesh's origin)."""

    def __init__(self, seqs: list[dict]):
        scales: dict[tuple[str, str], list] = {}
        self._seqs_of: dict[str, list[dict]] = {}
        for s in seqs:
            obj = s["seq_name"].split("_")[1]
            self._seqs_of.setdefault(obj, []).append(s)
            for part, (_, pre) in _parts(obj).items():
                scales.setdefault((obj, part), []).append(s[f"{pre}scale"])
        self.scale = {k: float(np.median(np.concatenate(v))) for k, v in scales.items()}
        self.centroid = {(o, p): _raw_mesh(o, suf).vertices.mean(0)
                         for o in self._seqs_of for p, (suf, _) in _parts(o).items()}
        self._joint: dict[str, dict] = {}

    def pose7(self, s: dict, obj: str, part: str) -> np.ndarray:
        """(F,7) pos + quat wxyz of the part's centred frame: (s R c + t, R)."""
        pre = _parts(obj)[part][1]
        R, t, sc = s[f"{pre}rot"].astype(np.float64), s[f"{pre}trans"][..., 0], s[f"{pre}scale"]
        pos = sc[:, None] * (R @ self.centroid[(obj, part)]) + t
        q = Rotation.from_matrix(R).as_quat()   # xyzw
        return _ensure_quat_continuity(np.concatenate([pos, q[:, [3, 0, 1, 2]]], axis=1))

    def export(self, obj: str, overwrite: bool) -> None:
        for part, (suf, _) in _parts(obj).items():
            dst = dataset_paths.object_mesh("omomo", obj)
            if part != "base":
                dst = dst.with_name(f"{obj}_{part}.obj")
            if dst.exists() and not overwrite:
                continue
            m = _raw_mesh(obj, suf)
            m.vertices = (m.vertices - self.centroid[(obj, part)]) * self.scale[(obj, part)]
            dst.parent.mkdir(parents=True, exist_ok=True)
            m.export(str(dst))

    def joint(self, obj: str) -> dict:
        """The base-bottom connection of a two-part object, measured over every valid frame of the dataset.

        pivot: the point fixed in both parts (least squares R_b u + p_b = R_h w + p_h), in each part frame (m).
        rel_rot_std_rad: principal spreads of the bottom's rotation relative to the handle about their mean
        (one large value = a hinge; two = a universal joint), axis_base = the first principal axis.
        """
        if obj not in self._joint:
            P, Rm = {}, {}
            for part in ("base", "bottom"):
                pr = [self.pose7(s, obj, part) for s in self._seqs_of[obj] if _valid(s)]
                pr = np.concatenate(pr)[::3]
                P[part] = pr[:, :3]
                Rm[part] = Rotation.from_quat(pr[:, [4, 5, 6, 3]])
            A = np.concatenate([Rm["base"].as_matrix(), -Rm["bottom"].as_matrix()], axis=2).reshape(-1, 6)
            b = (P["bottom"] - P["base"]).reshape(-1)
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            res = np.linalg.norm((A @ x - b).reshape(-1, 3), axis=1)
            rel = Rm["base"].inv() * Rm["bottom"]
            mean = rel.mean()
            dv = (mean.inv() * rel).as_rotvec()
            _, sv, vt = np.linalg.svd(dv - dv.mean(0), full_matrices=False)
            self._joint[obj] = {
                "part": "bottom", "joint_type": None, "axis": None, "pivot": x[:3].tolist(),
                "measured": {"pivot_base": x[:3].tolist(), "pivot_bottom": x[3:].tolist(),
                             "pivot_residual_mm": {"median": float(np.median(res) * 1e3),
                                                   "p99": float(np.percentile(res, 99) * 1e3)},
                             "rel_rot_std_rad": (sv / np.sqrt(len(dv))).tolist(),
                             "axis_base": mean.apply(vt[0]).tolist(),
                             "note": "no single DOF: no dof__ trajectory and no articulated USD"},
            }
        return self._joint[obj]


def _text_annotations() -> dict[str, str]:
    out = {}
    with zipfile.ZipFile(_RAW / "omomo_text_anno.zip") as z:
        for n in z.namelist():
            if n.endswith(".json"):
                out.update(json.loads(z.read(n)))
    return out


def process_sequence(s: dict, split: str, geo: ObjectGeometry, text: dict, overwrite: bool) -> dict:
    clip = s["seq_name"]
    subject, obj = clip.split("_")[:2]
    two = obj in TWO_PART
    cls = "single_articulated" if two else "single_rigid"
    F = len(s["trans"])
    entry = {"clip": clip, "class": cls, "seq": clip, "subject": subject, "seg_idx": 0,
             "frame_range": [0, F - 1], "num_frames": F, "action_text": text.get(clip, ""),
             "objects": [obj], "articulated": [obj] if two else [], "rigid": [] if two else [obj],
             "split": split, "object_split": _object_split_of(obj)}
    out_dir = _OUT / cls / clip / "0"
    if (out_dir / "trajectory.npz").exists() and not overwrite:
        return entry

    gender = str(s["gender"])
    betas = np.zeros(N_BETAS_OUT, np.float32)
    betas[:N_BETAS_SRC] = s["betas"].reshape(-1)
    go = s["root_orient"].astype(np.float32)
    bp = s["pose_body"].astype(np.float32)
    transl = s["trans"].astype(np.float32)
    pads, joints = _fk(_model(gender), betas, go, bp, transl)

    arrays = {
        "smplx_body_pose": bp,                                    # (F,63)
        "smplx_global_orient": go,                                # (F,3)
        "smplx_transl": transl,                                   # (F,3)
        "smplx_hand_pose": np.zeros((F, 90), np.float32),         # (F,90) not captured
        "smplx_betas": betas,                                     # (20,) 16 + zero padding
        "smplx_joints": joints,                                   # (F,55,3)
        "fingertip_pad_pos": pads,                                # (F,10,3) L[th,ff,mf,rf,lf] + R[...]
        "body_global_transform": _root_transform(go, joints[:, 0]),   # (F,4,4)
        "root_transl": joints[:, 0].copy(),                       # (F,3) pelvis world
        "frame_indices": np.arange(F),                            # (F,)
    }
    for part in _parts(obj):
        arrays[f"obj__{obj}__{part}"] = geo.pose7(s, obj, part)   # (F,7)
    geo.export(obj, overwrite=False)

    # Diagnostics: residual scale error and where the object mesh sits against the floor at frame 0.
    scale_dev = {p: float(np.abs(s[f"{pre}scale"] / geo.scale[(obj, p)] - 1).max())
                 for p, (_, pre) in _parts(obj).items()}
    z0 = []
    for part in _parts(obj):
        v = trimesh.load(str(dataset_paths.object_mesh("omomo", obj) if part == "base" else
                             dataset_paths.object_mesh("omomo", obj).with_name(f"{obj}_{part}.obj")),
                         process=False, force="mesh").vertices
        p0 = arrays[f"obj__{obj}__{part}"][0]
        z0.append(float((Rotation.from_quat(p0[[4, 5, 6, 3]]).apply(v) + p0[:3])[:, 2].min()))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_dir / "trajectory.npz"), **arrays)
    task_info = {
        "task": clip, "dataset_name": "omomo", "source_repr": "smplx",
        "class": cls, "seq": clip, "subject": subject, "seg_idx": 0,
        "frame_range": [0, F - 1], "num_frames": F, "source_fps": SRC_FPS,
        "action_text": text.get(clip, ""), "gender": gender, "ref_dt": REF_DT,
        "smplx_cfg": SMPLX_CFG,
        "manip_objects": [{"name": obj, "is_articulated": two, "parts": [geo.joint(obj)] if two else []}],
        "context_objects": [],
        "object_asset_dir": {obj: f"{_ASSET_REL}/{obj}"},
        "object_scale": {p: geo.scale[(obj, p)] for p in _parts(obj)},
        "object_scale_max_rel_dev": scale_dev,
        "object_min_z_frame0": min(z0),
        "split": split, "object_split": _object_split_of(obj),
    }
    with open(out_dir.parent / "task_info.json", "w") as f:
        json.dump(task_info, f, indent=2)
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Process OMOMO into per-sequence clips (ParaHome layout).")
    ap.add_argument("--subjects", nargs="*", default=None, help="Subset of subjects (e.g. sub16 sub17).")
    ap.add_argument("--objects", nargs="*", default=None, help="Subset of objects (e.g. largebox).")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N sequences.")
    ap.add_argument("--overwrite", action="store_true", help="Re-process existing clips and meshes.")
    args = ap.parse_args()

    seqs, split_of = [], {}
    for split in ("train", "test"):
        d = joblib.load(_RAW / "data" / f"{split}_diffusion_manip_seq_joints24.p")
        for s in d.values():
            seqs.append(s)
            split_of[s["seq_name"]] = split
    # Scales and joints come from the whole dataset, so a subset run writes the same clips.
    geo = ObjectGeometry([s for s in seqs if _valid(s)])
    if args.overwrite:
        for obj in {s["seq_name"].split("_")[1] for s in seqs}:
            geo.export(obj, overwrite=True)
    text = _text_annotations()

    skipped = sorted(s["seq_name"] for s in seqs if not _valid(s))
    todo = [s for s in seqs if _valid(s)
            and (not args.subjects or s["seq_name"].split("_")[0] in set(args.subjects))
            and (not args.objects or s["seq_name"].split("_")[1] in set(args.objects))]
    if args.limit > 0:
        todo = todo[: args.limit]

    _OUT.mkdir(parents=True, exist_ok=True)
    manifest, n_fail = [], 0
    for i, s in enumerate(todo, 1):
        try:
            manifest.append(process_sequence(s, split_of[s["seq_name"]], geo, text, args.overwrite))
            if i % 200 == 0 or i == len(todo):
                print(f"[ok] {i}/{len(todo)} {s['seq_name']}", flush=True)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[error] {s['seq_name']}: {e}", flush=True)

    with open(_OUT / "index.json", "w") as f:
        json.dump({"clips": manifest, "class_counts": dict(Counter(e["class"] for e in manifest)),
                   "skipped": {"zero_scale_frames": skipped}}, f, indent=2)
    with open(_OUT / "splits.json", "w") as f:
        json.dump({"clip2split": {e["clip"]: e["split"] for e in manifest},
                   "clip2object_split": {e["clip"]: e["object_split"] for e in manifest},
                   "object2split": {o: _object_split_of(o) for objs in OBJECT_SPLIT.values() for o in objs},
                   "source": "split: OMOMO train/test files (sub1-15 / sub16-17); object_split: OMOMO "
                             "hand_foot_dataset.py train_objects / test_objects"}, f, indent=2)
    print(f"\nDone. clips={len(manifest)} fail={n_fail} skipped(zero scale)={len(skipped)}  "
          f"class counts: {dict(Counter(e['class'] for e in manifest))}  "
          f"split counts: {dict(Counter(e['split'] for e in manifest))}")


if __name__ == "__main__":
    main()
