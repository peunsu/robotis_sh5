"""Build the articulation assembly spec for ParaHome objects (Phase B, step 1).

Runs in the base data env (numpy/scipy only — NO Isaac Sim). Produces a JSON that
the Isaac-Sim USD converter (parahome_convert_obj_to_usd.py) consumes to author
articulated USDs without needing to re-derive joint frames.

For each articulated object it records, per movable part:
  - joint_type: revolute | prismatic
  - axis: UNIT axis in the base(canonical) frame  (joint_info axes are NOT all unit)
  - pivot: a point on the joint axis in the base frame (revolute only; prismatic=null)
  - rest_T_part_in_base: (4,4) rest pose of the part in the base frame, taken at the
    frame where |joint DOF| is minimal (part most "closed") in a reference sequence.

VERIFIED (scripts/process_dataset — empirical check): joint_info axis/pivot are in
the base frame; the empirical rotation axis of `inv(T_base) @ T_part` matches the
(normalized) joint_info axis with |dot|~1.0 for sink/microwave/drawer.

Output: data/processed/parahome/assets/objects/articulation_spec.json
Run:    python scripts/process_dataset/assets/parahome_build_articulation_spec.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ARTICULATED_OBJECTS = {
    "drawer", "sink", "refrigerator", "gasstove",
    "laptop", "microwave", "trashbin", "washingmachine",
}
PRISMATIC_PARTS = {"drawer_part1", "drawer_part2"}

_SCRIPT_DIR = Path(__file__).resolve().parents[3]
_DATA_DIR = _SCRIPT_DIR / "source" / "robotis_sh5" / "data"
_RAW = _DATA_DIR / "raw" / "parahome" / "data"
_ASSET_OUT = _DATA_DIR / "processed" / "parahome" / "assets" / "objects"


def _load(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def _find_rest(obj: str, part: str):
    """Find a reference (sequence, frame) where |joint DOF| for {obj}_{part} is minimal,
    and return inv(T_base) @ T_part at that frame (part-in-base rest transform)."""
    inst = f"{obj}_{part}"
    best = None  # (abs_dof, seq, frame)
    seqs = sorted((_RAW / "seq").glob("s*"), key=lambda p: int(p.name[1:]))
    for sd in seqs:
        js_p = sd / "joint_states.pkl"
        if not js_p.exists():
            continue
        js = _load(js_p)
        if inst not in js:
            continue
        # frame with smallest |dof| (most closed / rest)
        f_rest = min(js[inst], key=lambda f: abs(js[inst][f]))
        v = abs(js[inst][f_rest])
        if best is None or v < best[0]:
            best = (v, sd.name, f_rest)
        if v < 1e-3:  # good enough rest
            break
    if best is None:
        return None
    _, seq, frame = best
    ot = _load(_RAW / "seq" / seq / "object_transformations.pkl")
    if frame not in ot or f"{obj}_base" not in ot[frame] or inst not in ot[frame]:
        return None
    Tb = np.asarray(ot[frame][f"{obj}_base"], dtype=np.float64)
    Tp = np.asarray(ot[frame][inst], dtype=np.float64)
    return (np.linalg.inv(Tb) @ Tp), seq, int(frame)


def main() -> None:
    joint_info = _load(_RAW / "joint_info.pkl")
    spec: dict = {}
    for obj in sorted(ARTICULATED_OBJECTS):
        parts_info = joint_info.get(obj, {})
        parts = []
        for part in sorted(parts_info.keys()):
            info = parts_info[part]
            axis = np.asarray(info["axis"], dtype=np.float64).reshape(-1)
            axis = axis / (np.linalg.norm(axis) + 1e-12)  # NORMALIZE (some are not unit)
            piv = np.asarray(info["pivot"], dtype=np.float64).reshape(-1)
            jtype = "prismatic" if f"{obj}_{part}" in PRISMATIC_PARTS else "revolute"
            rest = _find_rest(obj, part)
            entry = {
                "part": part,
                "joint_type": jtype,
                "axis": axis.tolist(),
                "pivot": (piv.tolist() if jtype == "revolute" and piv.shape[0] == 3 else None),
            }
            # Canonical rest is IDENTITY: ParaHome authors part ≡ base at joint DOF 0
            # (verified: 11/12 parts have inv(T_base)@T_part within 0.04deg / 0.35mm of
            # identity at min|dof|). Authoring identity bakes in NO residual DOF, and
            # corrects laptop_part1 (never observed near DOF 0 → min|dof|≈0.15 rad = 8.8deg
            # offset if the observed transform were used).
            entry["rest_T_part_in_base"] = np.eye(4).tolist()
            if rest is not None:
                T, seq, frame = rest
                dev_rot = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
                dev_t = float(np.linalg.norm(T[:3, 3]))
                entry["rest_ref"] = {"seq": seq, "frame": frame,
                                     "observed_dev_deg": round(dev_rot, 3),
                                     "observed_dev_m": round(dev_t, 4)}
                if dev_rot > 1.0 or dev_t > 0.01:
                    print(f"[warn] {obj}/{part}: observed rest deviates from identity "
                          f"({dev_rot:.2f}deg, {dev_t*1000:.1f}mm) — never near DOF 0; "
                          f"authoring identity canonical rest.")
            else:
                entry["rest_ref"] = None
                print(f"[warn] {obj}/{part}: no rest frame found; using identity")
            parts.append(entry)
        spec[obj] = {"parts": parts}
        print(f"[ok] {obj}: {len(parts)} parts "
              f"({', '.join(p['part']+':'+p['joint_type'] for p in parts)})")

    _ASSET_OUT.mkdir(parents=True, exist_ok=True)
    out = _ASSET_OUT / "articulation_spec.json"
    with open(out, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"\nWrote {out.relative_to(_DATA_DIR)}")


if __name__ == "__main__":
    main()
