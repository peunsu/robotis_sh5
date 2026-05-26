"""Visualize fingertip-link local axes for either FFW-SH5 or Shadow Hand.

For each fingertip link, draws colored spheres along the ±X / ±Y / ±Z local-frame
axes so you can read off which axis (and sign) corresponds to the pad-outward
normal — the direction the fingertip pad faces AWAY from the palm.

Shadow Hand mode lets you cross-check TJ's pad-normal convention against the
real link frames:

    TJ shadow-hand `fingertip_normal` (gr_env.py:196-198, MANO-order):
        index 0 (thumb)        : (-1, 0, 0)   → pad outward = -X local
        index 1-4 (FF/MF/RF/LF): (0, -1, 0)   → pad outward = -Y local
    Force is projected onto `-pad_normal_w` (i.e. inward into the finger body).

Color convention (matches RViz):
    +X red,    -X dark-red
    +Y green,  -Y dark-green
    +Z blue,   -Z dark-blue

Run (with GUI):
    python scripts/process_dataset/visualize_fingertip_normals.py                       # FFW-SH5 (default)
    python scripts/process_dataset/visualize_fingertip_normals.py --robot shadow_hand   # Shadow Hand
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize fingertip link local axes.")
parser.add_argument("--robot", type=str, default="ffw_sh5",
                    choices=["ffw_sh5", "shadow_hand"],
                    help="Which robot to visualize.")
parser.add_argument("--axis-length", type=float, default=0.04,
                    help="Distance (m) of marker spheres along each axis from the link origin.")
parser.add_argument("--sphere-radius", type=float, default=0.004,
                    help="Marker sphere radius (m).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Keep GUI on (do NOT set headless)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

_SCRIPT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPT_DIR / "source" / "robotis_sh5"))


# ────────────────────────────────────────────────────────────────────────────
# Robot configurations
# ────────────────────────────────────────────────────────────────────────────

def _ffw_sh5_setup():
    from robotis_sh5.tasks.direct.robotis_sh5_grasp.robotis_sh5_grasp_env_cfg import FFW_SH5_DEX_CFG
    fingertip_names = [
        "finger_r_link4",    # thumb
        "finger_r_link8",    # index
        "finger_r_link12",   # middle
        "finger_r_link16",   # ring
        "finger_r_link20",   # little
    ]
    label_map = {
        "finger_r_link4":  "thumb (link4)",
        "finger_r_link8":  "index (link8)",
        "finger_r_link12": "middle (link12)",
        "finger_r_link16": "ring (link16)",
        "finger_r_link20": "little (link20)",
    }
    camera_eye = (0.6, 0.8, 1.4)
    camera_target = (0.65, 0.65, 1.05)
    # TJ pad normals NOT known for FFW-SH5 — visualization is meant to discover them.
    tj_pad_normals = None
    return FFW_SH5_DEX_CFG, fingertip_names, label_map, camera_eye, camera_target, tj_pad_normals


def _shadow_hand_setup():
    from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG
    # TJ's MANO-ordered fingertip list — index 0 = thumb, 1-4 = first/middle/ring/little
    fingertip_names = [
        "robot0_thdistal",   # thumb
        "robot0_ffdistal",   # index (first finger)
        "robot0_mfdistal",   # middle
        "robot0_rfdistal",   # ring
        "robot0_lfdistal",   # little
    ]
    label_map = {
        "robot0_thdistal": "thumb (THdistal)",
        "robot0_ffdistal": "index (FFdistal)",
        "robot0_mfdistal": "middle (MFdistal)",
        "robot0_rfdistal": "ring (RFdistal)",
        "robot0_lfdistal": "little (LFdistal)",
    }
    camera_eye = (0.35, 0.35, 0.65)
    camera_target = (0.0, 0.0, 0.45)
    # TJ's known pad-normal convention (local frame)
    tj_pad_normals = {
        "robot0_thdistal": (-1.0, 0.0, 0.0),
        "robot0_ffdistal": (0.0, -1.0, 0.0),
        "robot0_mfdistal": (0.0, -1.0, 0.0),
        "robot0_rfdistal": (0.0, -1.0, 0.0),
        "robot0_lfdistal": (0.0, -1.0, 0.0),
    }
    return SHADOW_HAND_CFG, fingertip_names, label_map, camera_eye, camera_target, tj_pad_normals


# ────────────────────────────────────────────────────────────────────────────
# Axis marker setup
# ────────────────────────────────────────────────────────────────────────────

# Order: +X, -X, +Y, -Y, +Z, -Z
AXES_LOCAL = torch.tensor([
    [+1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, +1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, +1.0],
    [0.0, 0.0, -1.0],
], dtype=torch.float32)

AXIS_COLORS = [
    (1.00, 0.10, 0.10),   # +X bright red
    (0.50, 0.05, 0.05),   # -X dark red
    (0.10, 1.00, 0.10),   # +Y bright green
    (0.05, 0.50, 0.05),   # -Y dark green
    (0.10, 0.10, 1.00),   # +Z bright blue
    (0.05, 0.05, 0.50),   # -Z dark blue
]
AXIS_LABELS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def _make_axis_marker_cfg(axis_idx: int, axis_label: str, robot: str, radius: float):
    safe = axis_label.replace("+", "p").replace("-", "m")
    return VisualizationMarkersCfg(
        prim_path=f"/Visuals/{robot}_ft_axis_{safe}",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=AXIS_COLORS[axis_idx],
                ),
            )
        },
    )


def _make_tj_pad_marker_cfg(robot: str, radius: float):
    """Large bright-yellow sphere marking TJ's published pad-normal endpoint."""
    return VisualizationMarkersCfg(
        prim_path=f"/Visuals/{robot}_tj_pad_normal",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius * 1.8,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.95, 0.1)),
            )
        },
    )


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    if args_cli.robot == "ffw_sh5":
        robot_cfg_template, fingertip_names, label_map, camera_eye, camera_target, tj_pad_normals = _ffw_sh5_setup()
    else:
        robot_cfg_template, fingertip_names, label_map, camera_eye, camera_target, tj_pad_normals = _shadow_hand_setup()

    print()
    print("=" * 78)
    print(f"Fingertip-link local-axis visualizer  —  robot: {args_cli.robot}")
    print("=" * 78)
    print("Marker color legend (RViz convention):")
    for label, color in zip(AXIS_LABELS, AXIS_COLORS):
        print(f"    {label:>3} : RGB={color}")
    if tj_pad_normals is not None:
        print()
        print("  ★ YELLOW (large) sphere = TJ's published pad-outward normal endpoint.")
        print("    Force projection in TJ: (force_w * -pad_normal_w).sum(-1).clamp_min(0)")
        print()
        print("  TJ shadow-hand `fingertip_normal` (local-frame, gr_env.py:196-198):")
        for name in fingertip_names:
            print(f"    {label_map[name]:<22}: {tj_pad_normals[name]}")
    else:
        print()
        print("  (No published TJ convention for this robot — identify pad axis visually.)")
        print("  TJ shadow-hand reference (for cross-comparison):")
        print("    thumb       : local (-1, 0, 0)")
        print("    other 4     : local ( 0, -1, 0)")
    print("=" * 78)
    print()

    sim_cfg = SimulationCfg(dt=1.0 / 120.0, render_interval=1)
    sim = SimulationContext(sim_cfg)

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    robot_cfg = robot_cfg_template.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(robot_cfg)

    sim.set_camera_view(eye=camera_eye, target=camera_target)

    # Six axis markers (shared across all fingertips)
    axis_markers: list[VisualizationMarkers] = []
    for i, label in enumerate(AXIS_LABELS):
        axis_markers.append(VisualizationMarkers(
            _make_axis_marker_cfg(i, label, args_cli.robot, args_cli.sphere_radius)
        ))

    # Optional: TJ-known pad normal marker
    tj_marker = None
    if tj_pad_normals is not None:
        tj_marker = VisualizationMarkers(_make_tj_pad_marker_cfg(args_cli.robot, args_cli.sphere_radius))

    sim.reset()
    robot.update(sim.get_physics_dt())

    device = sim.device
    axes_local = AXES_LOCAL.to(device)

    body_ids: list[int] = []
    for name in fingertip_names:
        found, _ = robot.find_bodies(name)
        if not found:
            raise RuntimeError(f"Body '{name}' not found in robot.")
        body_ids.append(found[0])

    print(f"Fingertip body IDs: {dict(zip(fingertip_names, body_ids))}")
    print(f"\nAxis length: {args_cli.axis_length*100:.1f} cm  •  sphere radius: {args_cli.sphere_radius*100:.2f} cm")
    print("\nRotate camera in GUI. Identify the axis pointing OUTWARD from each fingertip pad")
    print("(direction the pad face looks away from the palm).")
    if tj_marker is not None:
        print("Yellow sphere shows TJ's published pad-outward endpoint — should overlap the")
        print("correct axis-color sphere of the same direction.")
    print("Ctrl-C in terminal to exit.\n")

    L = args_cli.axis_length

    # TJ pad normal tensor (if available)
    tj_normals_t = None
    if tj_pad_normals is not None:
        tj_normals_t = torch.tensor(
            [tj_pad_normals[n] for n in fingertip_names],
            dtype=torch.float32, device=device,
        )  # (5, 3)

    while simulation_app.is_running():
        sim.step()
        robot.update(sim.get_physics_dt())

        # Gather link poses
        link_pos_w = robot.data.body_pos_w[0, body_ids, :]    # (5, 3)
        link_quat_w = robot.data.body_quat_w[0, body_ids, :]  # (5, 4)

        # Draw 6 axis-spheres per fingertip
        for ax_idx in range(6):
            axis_local = axes_local[ax_idx].unsqueeze(0).expand(len(body_ids), -1)   # (5, 3)
            axis_world = quat_apply(link_quat_w, axis_local)                          # (5, 3)
            translations = link_pos_w + axis_world * L
            axis_markers[ax_idx].visualize(translations=translations)

        # Draw TJ pad-normal yellow sphere
        if tj_marker is not None and tj_normals_t is not None:
            tj_world = quat_apply(link_quat_w, tj_normals_t)   # (5, 3)
            tj_translations = link_pos_w + tj_world * L
            tj_marker.visualize(translations=tj_translations)


if __name__ == "__main__":
    main()
