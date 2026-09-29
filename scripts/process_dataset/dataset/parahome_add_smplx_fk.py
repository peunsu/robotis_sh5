"""Add smplx_joints / fingertip_pad_pos to one ParaHome clip processed without smplx.

parahome.py writes these two keys only when `smplx` imports; clips processed in an env without it lack
them, and the retarget, the MPPI refine and the envs all read them. Same FK as parahome.py
(_seq_smplx_fk over the clip's whole sequence, sliced by frame_indices; the same as monodex
tools/parahome_smplx_fk.py, per clip). Every other key of trajectory.npz is kept as is; the file is
replaced atomically.

    <env_isaaclab python> scripts/process_dataset/dataset/parahome_add_smplx_fk.py --clip s12_seg07_book
    ... --check s100_seg00_pan     # a clip that has the keys: recompute and compare, write nothing
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parahome as P  # noqa: E402


def seq_fk(seq: str):
    smdir = P._RAW / "smplx_seq" / seq
    pose = P._load_pkl(smdir / "smplx_pose.pkl")
    params = P._load_pkl(smdir / "smplx_params.pkl")
    betas = P._to_np(params["beta"]).reshape(-1).astype(np.float32)
    return P._seq_smplx_fk(pose, betas, str(params["gender"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", default="single_rigid")
    ap.add_argument("--clip", default="")
    ap.add_argument("--check", default="", help="clip that already has the keys: compare, do not write")
    a = ap.parse_args()
    if not P._HAS_SMPLX:
        sys.exit("needs the smplx package (run in env_isaaclab)")
    clip = a.check or a.clip
    if not clip:
        sys.exit("--clip or --check is required")
    path = P._OUT / a.cls / clip / "0" / "trajectory.npz"
    z = dict(np.load(path, allow_pickle=True))
    has = {"smplx_joints", "fingertip_pad_pos"} <= set(z)
    if not a.check and has:
        print(f"[smplx-fk] {clip}: smplx_joints / fingertip_pad_pos already present — nothing to do")
        return
    pads, jts = seq_fk(clip.split("_")[0])
    fi = z["frame_indices"]
    if a.check:
        if not has:
            sys.exit(f"--check {clip}: the clip has no keys to compare against")
        print(f"[smplx-fk] check {clip}: joints max diff {float(np.abs(jts[fi] - z['smplx_joints']).max()):.2e} m, "
              f"pads max diff {float(np.abs(pads[fi] - z['fingertip_pad_pos']).max()):.2e} m")
        return
    z["smplx_joints"] = jts[fi].astype(np.float32)
    z["fingertip_pad_pos"] = pads[fi].astype(np.float32)
    tmp = str(path)[:-4] + ".tmp.npz"
    np.savez(tmp, **z)
    os.replace(tmp, path)
    print(f"[smplx-fk] {clip}: added smplx_joints {z['smplx_joints'].shape}, fingertip_pad_pos "
          f"{z['fingertip_pad_pos'].shape} -> {path}")


if __name__ == "__main__":
    main()
