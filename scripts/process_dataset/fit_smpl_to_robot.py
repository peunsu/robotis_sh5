"""Fit SMPL-X shape β to roughly match FFW-SH5 robot body proportions.

Matches multiple PAIRWISE distances between corresponding SMPL/robot landmarks
(both arms + shoulder width + torso). Pairwise distances are coordinate-frame
invariant so no rigid alignment is needed.

The fit is "rough" by design — SMPL β can only span human-shape variation, while
the robot's per-link kinematics (esp. wrist gimbal) differ structurally. The
goal is to land in a sensible β so that downstream VPoser-IK produces
reasonable arm poses; later we use only the elbow DIRECTION (unit vector)
projected onto the robot's actual upper-arm length to side-step the residual gap.

Run:
    python scripts/process_dataset/fit_smpl_to_robot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin
import smplx
import torch

_SCRIPT_DIR = Path(__file__).resolve().parents[2]
_URDF_PATH = _SCRIPT_DIR / "source" / "robotis_sh5" / "data" / "robots" / "FFW" / "urdf" / "ffw_sh5_follower_copy.urdf"
_SMPLX_MODEL_DIR = "/home/peunsu/workspace/human_body_prior/support_data/dowloads/models/"
_OUT_DIR = _SCRIPT_DIR / "source" / "robotis_sh5" / "data" / "smpl_fit"

# Landmark pairs: (label, smpl_joint_a, smpl_joint_b, robot_frame_a, robot_frame_b)
# SMPL-X indices: l_shoulder=16, r_shoulder=17, l_elbow=18, r_elbow=19,
#                 l_wrist=20, r_wrist=21.
# Robot is wheeled (no pelvis/spine), so we restrict to upper-body landmarks
# that exist on both: both arms + shoulder width.
_PAIRS: list[tuple[str, int, int, str, str]] = [
    # Right arm
    ("upper_arm_R", 17, 19, "arm_r_link1", "arm_r_link4"),
    ("forearm_R",   19, 21, "arm_r_link4", "hx5_d20_right_base"),
    # Left arm (mirror — robot is symmetric)
    ("upper_arm_L", 16, 18, "arm_l_link1", "arm_l_link4"),
    ("forearm_L",   18, 20, "arm_l_link4", "hx5_d20_left_base"),
    # Shoulder width
    ("shoulder_W",  17, 16, "arm_r_link1", "arm_l_link1"),
    # Neck + head (robot has head_joint1 pan + head_joint2 tilt on the lift mast)
    ("shoulder→neck_R", 17, 12, "arm_r_link1", "head_link1"),
    ("shoulder→head_R", 17, 15, "arm_r_link1", "head_link2"),
    ("neck→head",       12, 15, "head_link1",  "head_link2"),
]


def _robot_pair_distances() -> dict[str, float]:
    """FK on URDF at neutral pose → pairwise euclidean distances for each landmark pair."""
    model = pin.buildModelFromUrdf(str(_URDF_PATH))
    data = model.createData()
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    dists: dict[str, float] = {}
    print("Robot pairwise distances (at URDF neutral pose):")
    for label, _sa, _sb, fa, fb in _PAIRS:
        pa = np.asarray(data.oMf[model.getFrameId(fa)].translation, dtype=np.float64)
        pb = np.asarray(data.oMf[model.getFrameId(fb)].translation, dtype=np.float64)
        d = float(np.linalg.norm(pa - pb))
        dists[label] = d
        print(f"  {label:<18s} {fa} ↔ {fb:<24s} d={d:.4f} m")
    return dists


def main():
    parser = argparse.ArgumentParser(description="Fit SMPL-X β to robot body proportions (multi-keypoint).")
    parser.add_argument("--num_iter", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--reg", type=float, default=0.005, help="L2 regularization on β.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    robot_dists = _robot_pair_distances()

    device = torch.device(args.device)
    model = smplx.create(
        _SMPLX_MODEL_DIR,
        model_type="smplx", gender="neutral",
        use_pca=False, batch_size=1, ext="npz",
    ).to(device)

    targets_t = {
        label: torch.tensor(d, device=device, dtype=torch.float32)
        for label, d in robot_dists.items()
    }

    beta = torch.zeros(1, model.num_betas, device=device, requires_grad=True)
    opt = torch.optim.Adam([beta], lr=args.lr)

    print(f"\nOptimizing β ({model.num_betas} dims, {len(_PAIRS)} landmark pairs) on {device}...")
    for it in range(args.num_iter):
        out = model(betas=beta)
        J = out.joints[0]

        loss = args.reg * (beta ** 2).sum()
        per_pair_dist = {}
        for label, sa, sb, _, _ in _PAIRS:
            d = (J[sa] - J[sb]).norm()
            per_pair_dist[label] = d
            loss = loss + (d - targets_t[label]) ** 2

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 500 == 0 or it == args.num_iter - 1:
            line = f"  [it {it:4d}] loss={loss.item():.5f}"
            for label, _, _, _, _ in _PAIRS:
                line += f"  {label}={per_pair_dist[label].item():.3f}"
            print(line)

    beta_np = beta.detach().cpu().numpy()[0]
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "robot_beta.npy"
    np.save(str(out_path), beta_np.astype(np.float32))
    print(f"\nSaved β to {out_path}")
    print(f"β = {beta_np}")

    # Final per-pair residuals
    print("\nFinal residuals (SMPL vs robot, both in meters):")
    with torch.no_grad():
        out = model(betas=beta)
        J = out.joints[0]
        for label, sa, sb, fa, fb in _PAIRS:
            d_smpl = (J[sa] - J[sb]).norm().item()
            d_rob = robot_dists[label]
            gap = d_smpl - d_rob
            print(f"  {label:<18s} smpl={d_smpl:.4f}  robot={d_rob:.4f}  gap={gap:+.4f}")


if __name__ == "__main__":
    main()
