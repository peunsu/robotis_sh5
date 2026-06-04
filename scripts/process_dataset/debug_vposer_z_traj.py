"""Diagnostic: re-run VPoser fit on one trajectory and dump z_traj statistics.

Goal: identify which frames see latent-z jumps that don't correspond to MANO source jumps,
to localize the cause of SMPL elbow oscillation in G06_2.

Usage:
    python scripts/process_dataset/debug_vposer_z_traj.py \\
        --dataset hocap \\
        --trajectory subject_6-20231025_111357-G06_2/0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPT_DIR / "scripts" / "process_dataset"))

from process_arm_pipeline import (  # noqa: E402
    VPoserPipeline,
    _canonicalize_full,
    _HOCAP_DIR,
    _OAKINK_DIR,
    _compute_robot_anchors,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hocap", choices=["oakink", "hocap"])
    ap.add_argument("--trajectory", required=True,
                    help="e.g. subject_6-20231025_111357-G06_2/0")
    ap.add_argument("--smooth_reg", type=float, default=0.05)
    ap.add_argument("--palm_weight", type=float, default=1.0)
    ap.add_argument("--shoulder_anchor_weight", type=float, default=1.0)
    ap.add_argument("--num_iter", type=int, default=300)
    ap.add_argument("--out_npz", type=str, default=None)
    args = ap.parse_args()

    ds_dir = _HOCAP_DIR if args.dataset == "hocap" else _OAKINK_DIR
    traj_dir = ds_dir / "mano" / "right" / args.trajectory
    traj_npz = traj_dir / "trajectory_keypoints.npz"
    assert traj_npz.exists(), f"missing: {traj_npz}"

    import torch
    shoulder_env, upper_arm, forearm_to_link7 = _compute_robot_anchors()
    print(f"[setup] upper_arm={upper_arm:.4f}m forearm_to_link7={forearm_to_link7:.4f}m "
          f"shoulder_env={shoulder_env}")

    pipeline = VPoserPipeline(
        device=torch.device("cuda"),
        robot_upper_arm=float(upper_arm),
        robot_forearm_to_link7=float(forearm_to_link7),
    )

    res = _canonicalize_full(traj_npz, ds_dir)
    assert res is not None, "canonicalize failed"
    wp_all, wq_all, mano_kpts_canon, _, _, _ = res
    N = wp_all.shape[0]
    print(f"[data] N={N} frames")

    elbow_envs, wrist_resids, palm_resids, z_traj, pelvis_env = pipeline.extract_batched(
        wp_all, wq_all, mano_kpts_canon, shoulder_env,
        num_iter=args.num_iter,
        smooth_reg=args.smooth_reg,
        palm_weight=args.palm_weight,
        shoulder_anchor_weight=args.shoulder_anchor_weight,
    )

    smpl_verts_env, smpl_joints_env = pipeline.render_smpl_per_frame(z_traj, pelvis_env)
    # SMPL skeleton joints (env frame)
    sh_smpl = smpl_joints_env[:, 17]
    el_smpl = smpl_joints_env[:, 19]
    wr_smpl = smpl_joints_env[:, 21]

    z_np = z_traj.detach().cpu().numpy()                                # (N, 32)
    dz = np.linalg.norm(np.diff(z_np, axis=0), axis=-1)                 # (N-1,)
    dsh = np.linalg.norm(np.diff(sh_smpl, axis=0), axis=-1)
    del_ = np.linalg.norm(np.diff(el_smpl, axis=0), axis=-1)
    dwr = np.linalg.norm(np.diff(wr_smpl, axis=0), axis=-1)
    de_rescaled = np.linalg.norm(np.diff(elbow_envs, axis=0), axis=-1)
    dwp = np.linalg.norm(np.diff(wp_all, axis=0), axis=-1)              # MANO wrist input

    def _stats(name, arr, scale=1.0, unit=""):
        print(f"  {name:32s}: max={arr.max()*scale:.3f}{unit} @ {int(arr.argmax()):3d}  "
              f"p95={np.percentile(arr,95)*scale:.3f}{unit}  mean={arr.mean()*scale:.3f}{unit}")

    print("\n[per-frame deltas]")
    _stats("z latent L2 (32D)",       dz,           1.0,    "")
    _stats("MANO wrist input pos",    dwp,          1000.0, "mm")
    _stats("SMPL skel wrist (raw)",   dwr,          1000.0, "mm")
    _stats("SMPL skel elbow (raw)",   del_,         1000.0, "mm")
    _stats("SMPL skel shoulder",      dsh,          1000.0, "mm")
    _stats("Rescaled elbow (saved)",  de_rescaled,  1000.0, "mm")

    print(f"\n[residuals]")
    print(f"  wrist_pos_resid mean={wrist_resids.mean()*1000:.2f}mm  max={wrist_resids.max()*1000:.2f}mm @ {int(wrist_resids.argmax())}")
    print(f"  palm_resid      mean={palm_resids.mean()*1000:.2f}mm  max={palm_resids.max()*1000:.2f}mm @ {int(palm_resids.argmax())}")

    # Find frames where z jump is large AND MANO wrist jump is small (= VPoser basin flip)
    # Normalize by per-trajectory median to identify outliers
    dz_med = np.median(dz)
    dwp_med = np.median(dwp)
    flip_score = (dz / max(dz_med, 1e-6)) - (dwp / max(dwp_med, 1e-6))
    worst = np.argsort(flip_score)[-15:][::-1]
    print("\n[basin-flip suspects]  frames where z jumps >> MANO wrist jumps")
    print(f"  {'frame':>5}  {'dz':>8}  {'dwp_mm':>8}  {'d_el_smpl_mm':>13}  {'d_el_resc_mm':>13}")
    for i in worst:
        print(f"  {int(i):5d}  {dz[i]:8.4f}  {dwp[i]*1000:8.2f}  {del_[i]*1000:13.2f}  {de_rescaled[i]*1000:13.2f}")

    # Save raw arrays for offline plotting if requested
    if args.out_npz:
        out_path = Path(args.out_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(out_path),
            z_traj=z_np,
            sh_smpl=sh_smpl, el_smpl=el_smpl, wr_smpl=wr_smpl,
            elbow_envs=elbow_envs,
            wrist_targets=wp_all,
            wrist_resids=wrist_resids,
            palm_resids=palm_resids,
        )
        print(f"\n[dump] saved -> {out_path}")


if __name__ == "__main__":
    main()
