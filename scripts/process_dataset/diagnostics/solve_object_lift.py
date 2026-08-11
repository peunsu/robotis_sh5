"""Per-frame vertical offset that lets the reference object REST where the capture says it is.

The captured object trajectory is human-capture data; nothing makes it collision-free against the
scene the robot stands in. Where the object is sunk into its support, PhysX depenetrates it on the
first step, and because a thin blade is pushed out about a contact point far from its centre of mass
the damage lands on ORIENTATION, not height. Measured on s101_seg12_knife with the robot parked away:

    raw reference, frames 0-45   settled rotation 16.7 deg,  spread 8.5 - 65.0 deg  (std 13.5)
    lifted 0.75 cm               settled rotation  9.2 deg,  spread 8.4 -  9.8 deg  (std  0.3)

Those 46 frames are effectively ONE pose (the knife does not move until frame 41), so the raw spread
is not frame-to-frame variety — it is the depenetration impulse being chaotic. Every evaluation
rollout starts in that region, so the policy is asked to grip an object whose reset pose is a coin
flip between 8 and 65 degrees. Removing that unpredictability is the point of the correction; the
~9 deg residual is a genuine settled tip (rotation plateaus by 40 steps and holds through 640) and a
pure height offset cannot remove it -- `--correct_quat` writes the settled orientation for that.

The resting height is unique: dropping from 0.00 / 0.75 / 1.50 cm all settle to ref_z + 0.47 cm. So
one drop pass measures it, and the SAME pass classifies the frame, which cannot be assumed:

    supported   comes to rest near the reference height -> `final_z - ref_z` IS the correction, and
                a positive value means the raw reference was penetrating by that much
    airborne    falls away (in the capture the object is held in the air) -> no correction is defined

Only frames that actually penetrate are corrected. A frame already resting at the captured height,
and an airborne frame with nothing to penetrate, both keep the capture untouched — the scene objects
are ground truth and the correction must not drift the reference where it was already right. The
lift ramps to zero over `--taper` frames OUTSIDE each penetrating run, because a hard cut would step
the reference by the full lift in one frame and the finite-differenced reference velocity would hand
an RSI reset on that frame a 0.24 m/s launch that is not in the capture.

Writes obj_settle_correction.npz into the clip directory: `lift` (m, per frame, zero outside the
penetrating runs), `supported`, `penetrating`, `settled_quat`, plus the verification residuals.

    python scripts/process_dataset/diagnostics/solve_object_lift.py \
        --task Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0 \
        --clip_class single_rigid --clip_name s101_seg12_knife --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Solve a per-frame vertical lift for the reference object.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--clip_class", type=str, default=None)
parser.add_argument("--clip_name", type=str, default=None)
parser.add_argument("--drop_cm", type=float, default=2.0,
                    help="Release height above the reference. Must clear the deepest penetration so the\n"
                         "object lands on its support instead of being ejected from inside it; the\n"
                         "resting height it reaches is independent of this (0.00/0.75/1.50 cm all give\n"
                         "the same +0.47 cm on the knife), so this only has to be big enough.")
parser.add_argument("--settle_steps", type=int, default=80,
                    help="Rotation plateaus by 40 steps and is unchanged at 640; 80 is comfortably past it.")
parser.add_argument("--rest_speed", type=float, default=0.05,
                    help="Final |v| (m/s) below which the object counts as AT REST. Resting is the right\n"
                         "test, not 'stayed where I put it': an object released 2 cm up lands 2 cm lower\n"
                         "and is still perfectly supported.")
parser.add_argument("--fall_tol_cm", type=float, default=1.0,
                    help="How far BELOW the reference height a frame may settle and still count as\n"
                         "supported. Beyond this the object left its support and fell.")
parser.add_argument("--min_penetration_cm", type=float, default=0.1,
                    help="Only frames that settle at least this far ABOVE the reference are treated as\n"
                         "penetrating and get corrected. Everything else keeps the captured pose, so the\n"
                         "correction stays confined to the frames that actually need it.")
parser.add_argument("--taper", type=int, default=5,
                    help="Frames over which the lift ramps to zero OUTSIDE each penetrating run. A hard\n"
                         "cut would step the reference by the full lift in one frame; at 50 Hz a 0.47 cm\n"
                         "step is a 0.24 m/s spike in the finite-differenced reference velocity, which an\n"
                         "RSI reset on that frame would hand to the object as real motion. Penetrating\n"
                         "frames themselves keep their full lift — the ramp lives entirely outside them.")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--dry_run", action="store_true", help="Report only; do not write into the clip directory.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import robotis_sh5.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.clip_class:
        env_cfg.clip_class = args_cli.clip_class
    if args_cli.clip_name:
        env_cfg.clip_name = args_cli.clip_name
    env_cfg.scene.num_envs = 256
    env_cfg.termination = False
    env_cfg.debug_vis = False
    # ALWAYS measure the raw capture. Leaving the correction enabled would make the solver read back
    # its own previous output — the reference would already be lifted, penetration would measure ~0,
    # and the npz would be overwritten with an empty correction that undoes the fix.
    env_cfg.object_settle_lift = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    dev, n, org = env.device, env.num_envs, env.scene.env_origins
    F = env._ref_len

    # park the robot out of the scene: the retargeted hand interpenetrates the object on held frames
    # and ejects it, which is a retargeting problem and would swamp the object-vs-scene signal.
    root = torch.zeros(n, 7, device=dev)
    root[:, :3] = org + torch.tensor([0.0, 0.0, 5.0], device=dev)
    root[:, 3] = 1.0
    env.robot.write_root_pose_to_sim(root)
    env.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))

    def settle(frames, dz_cm, quat=None):
        """Place the object at ref(frames) + dz, settle, return (pos, quat, |v|) in env-local frame."""
        m = len(frames)
        op = torch.zeros(n, 7, device=dev)
        op[:m, :3] = env._ref_obj_pos[frames] + org[:m]
        op[:m, 2] += torch.as_tensor(dz_cm, device=dev, dtype=torch.float32) * 0.01
        op[:m, 3:7] = env._ref_obj_quat[frames] if quat is None else quat
        op[m:, 3] = 1.0
        env._object.write_root_pose_to_sim(op)
        env._object.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        env.scene.write_data_to_sim()
        for _ in range(args_cli.settle_steps):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        return (env._object.data.root_pos_w[:m] - org[:m],
                env._object.data.root_quat_w[:m],
                env._object.data.root_lin_vel_w[:m].norm(dim=-1))

    # ---- pass 1: release from a clear height, read where it comes to rest ----
    lift_raw = np.zeros(F, dtype=np.float32)
    settled_q = np.zeros((F, 4), dtype=np.float32)
    supported = np.zeros(F, dtype=bool)
    for base in range(0, F, n):
        fr = torch.arange(base, min(base + n, F), device=dev)
        p, q, v = settle(fr, args_cli.drop_cm)
        dz = (p[:, 2] - env._ref_obj_pos[fr, 2]).cpu().numpy()
        supported[base:base + len(fr)] = ((v.cpu().numpy() < args_cli.rest_speed)
                                          & (dz > -args_cli.fall_tol_cm * 0.01))
        lift_raw[base:base + len(fr)] = dz
        settled_q[base:base + len(fr)] = q.cpu().numpy()
        print(f"  measured frames {base:4d}-{base + len(fr) - 1:4d}")

    # ---- lift curve: correct ONLY the frames that actually penetrate. A frame that already rests
    # at (or above) the captured height needs nothing, and an airborne frame has no support to
    # penetrate — both keep the capture untouched, so the scene objects stay the ground truth they
    # are and the correction cannot drift the reference where it was already right.
    penetrating = supported & (lift_raw > args_cli.min_penetration_cm * 0.01)
    lift = np.where(penetrating, np.clip(lift_raw, 0.0, None), 0.0).astype(np.float32)
    # Ramp to zero outside each penetrating run. Dilating the run by `taper` BEFORE the moving
    # average is what keeps the run itself at full lift: the averaging window over a penetrating
    # frame then sees only full-lift samples, and the ramp falls entirely in the untouched frames.
    t = args_cli.taper
    if t > 0 and penetrating.any():
        ext = lift.copy()
        for _ in range(t):
            ext = np.maximum(ext, np.maximum(np.r_[ext[1:], ext[-1]], np.r_[ext[0], ext[:-1]]))
        k = np.ones(2 * t + 1) / (2 * t + 1)
        lift = np.convolve(np.pad(ext, t, mode="edge"), k, mode="valid")[:F].astype(np.float32)

    # ---- pass 2: does the corrected height hold? place AT ref+lift and settle again ----
    ver_disp = np.zeros(F, dtype=np.float32)
    ver_rot = np.zeros(F, dtype=np.float32)
    for base in range(0, F, n):
        fr = torch.arange(base, min(base + n, F), device=dev)
        m = len(fr)
        p, q, _ = settle(fr, lift[base:base + m] * 100.0)
        tgt = env._ref_obj_pos[fr].clone()
        tgt[:, 2] += torch.from_numpy(lift[base:base + m]).to(dev)
        ver_disp[base:base + m] = (p - tgt).norm(dim=-1).cpu().numpy()
        dot = (q * env._ref_obj_quat[fr]).sum(-1).abs().clamp(max=1.0)
        ver_rot[base:base + m] = torch.rad2deg(2 * torch.arccos(dot)).cpu().numpy()

    clip_dir = env._resolve_clip_dir(env_cfg)
    out = args_cli.out or os.path.join(clip_dir, "obj_settle_correction.npz")
    if not args_cli.dry_run:
        np.savez(out, lift=lift, lift_raw=lift_raw, supported=supported, penetrating=penetrating,
                 settled_quat=settled_q, ver_disp=ver_disp, ver_rot=ver_rot,
                 drop_cm=args_cli.drop_cm, settle_steps=args_cli.settle_steps)

    s = supported
    print(f"\n=== {env_cfg.clip_name}: {F} frames ===")
    print(f"supported (rests at the reference height) : {int(s.sum()):4d}")
    print(f"airborne  (held in the air in the capture): {int((~s).sum()):4d}")
    print(f"  of the supported, actually PENETRATING  : {int(penetrating.sum()):4d}  <- the only frames corrected")
    print(f"untouched (lift < 0.01 cm)                : {int((lift < 1e-4).sum()):4d} / {F}")
    if penetrating.any():
        pen = lift_raw[penetrating] * 100.0
        print(f"\npenetration depth (cm): mean {pen.mean():5.2f}  median {np.median(pen):5.2f}  "
              f"p90 {np.percentile(pen, 90):5.2f}  max {pen.max():5.2f}")
        runs = np.flatnonzero(np.diff(np.r_[0, penetrating.astype(int), 0]) != 0).reshape(-1, 2)
        print(f"penetrating runs: " + ", ".join(f"{a}-{b - 1}" for a, b in runs[:12])
              + (" ..." if len(runs) > 12 else ""))
    print(f"\nverification at the corrected height: displacement {ver_disp[s].mean() * 100:5.2f} cm, "
          f"residual rotation {ver_rot[s].mean():5.2f} deg  (a pure lift cannot remove the tip)")
    print("\nframe   supported   penetrating   lift(cm)   verify disp / rot")
    for f in list(range(0, 60, 5)) + [100, 200, 300, 400, F - 1]:
        if f >= F:
            continue
        print(f"{f:5d}   {str(bool(s[f])):>9s}   {str(bool(penetrating[f])):>11s}   {lift[f] * 100:7.2f}   "
              f"{ver_disp[f] * 100:6.2f} cm  {ver_rot[f]:6.2f} deg")
    print(f"\n{'(dry run, nothing written)' if args_cli.dry_run else 'correction -> ' + out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
