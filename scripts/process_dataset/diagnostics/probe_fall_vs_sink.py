"""Are the robots in the training videos sinking through the floor, or falling over?

Training videos show robots vanishing from view — by step 499 of rl-video-step-0 most of the grid is
empty above the counters. "Below the floor" is one reading, but the measurements so far rule it out:
the reference sole sits +0.63 cm above ground, the RSI cache holds no root below 0.75 m, and a
2500-step run reached only -4 cm on one ankle. What has NOT been measured is ORIENTATION — a robot
that topples disappears behind a 0.86 m counter just as completely as one that sinks, and the two
need opposite fixes.

So record both, per env per step: root height, torso height (what actually clears the counter), and
root tilt (angle between the body's up axis and world up). Falling shows as tilt; sinking shows as
height with tilt near zero.

Actions are Gaussian at the policy's initial sigma (initial_log_std = -1.0 -> 0.368), matching the
step-0 video the report came from.

    python scripts/process_dataset/diagnostics/probe_fall_vs_sink.py \
        --task Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0 \
        --clip_class single_rigid --clip_name s101_seg12_knife --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Distinguish toppling from sinking during training-like stepping.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--clip_class", type=str, default=None)
parser.add_argument("--clip_name", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--std", type=float, default=0.368, help="Action sigma; 0.368 = initial_log_std -1.0.")
parser.add_argument("--counter_top", type=float, default=0.86, help="Counter height the robot must clear.")
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import robotis_sh5.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.clip_class:
        env_cfg.clip_class = args_cli.clip_class
    if args_cli.clip_name:
        env_cfg.clip_name = args_cli.clip_name
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    dev, org = env.device, env.scene.env_origins
    n, adim = env.num_envs, env.cfg.action_space
    up = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    torso = env.robot.find_bodies("torso_link")[0][0]
    ref_z = float(env._ref_root_pos[:, 2].mean())

    rz, tz, tilt, ep = [], [], [], []
    for k in range(args_cli.steps):
        env.step(torch.randn(n, adim, device=dev) * args_cli.std)
        rz.append((env.robot.data.root_pos_w[:, 2] - org[:, 2]).cpu().numpy())
        tz.append((env.robot.data.body_pos_w[:, torso, 2] - org[:, 2]).cpu().numpy())
        zax = math_utils.quat_apply(env.robot.data.root_quat_w, up)
        tilt.append(torch.rad2deg(torch.arccos(zax[:, 2].clamp(-1.0, 1.0))).cpu().numpy())
        ep.append(env.episode_length_buf.cpu().numpy())
        if k % 100 == 0:
            print(f"  step {k:4d}  root p5 {np.percentile(rz[-1], 5):.3f}  tilt p95 {np.percentile(tilt[-1], 95):5.1f}")
    rz, tz, tilt, ep = map(np.array, (rz, tz, tilt, ep))

    out = args_cli.out or f"/tmp/fall_vs_sink_{env_cfg.clip_name}.npz"
    np.savez(out, root_z=rz, torso_z=tz, tilt=tilt, ep_len=ep, ref_root_z=ref_z)

    ct = args_cli.counter_top
    hidden = tz < ct
    print(f"\n=== {env_cfg.clip_name}: {args_cli.steps} steps x {n} envs   (reference root z = {ref_z:.3f} m) ===")
    print(f"root z    p50 {np.median(rz):.3f}  p5 {np.percentile(rz, 5):.3f}  "
          f"p1 {np.percentile(rz, 1):.3f}  min {rz.min():.3f}")
    print(f"torso z   p50 {np.median(tz):.3f}  p5 {np.percentile(tz, 5):.3f}  "
          f"p1 {np.percentile(tz, 1):.3f}  min {tz.min():.3f}")
    print(f"tilt deg  p50 {np.median(tilt):5.1f}  p95 {np.percentile(tilt, 95):5.1f}  "
          f"p99 {np.percentile(tilt, 99):5.1f}  max {tilt.max():5.1f}")
    print("\nhow often each explanation holds (share of all env-steps):")
    for t in (20, 45, 90):
        print(f"  tilt > {t:3d} deg (toppled)            {100 * float((tilt > t).mean()):6.2f}%")
    print(f"  root z < 0.60 m (sunk >19 cm)        {100 * float((rz < 0.60).mean()):6.2f}%")
    print(f"  root z < 0.00 m (below the floor)    {100 * float((rz < 0.0).mean()):6.2f}%")
    print(f"\ntorso below the counter top ({ct} m)   {100 * float(hidden.mean()):6.2f}%  <- what 'vanished' means")
    if hidden.any():
        print(f"   of those, tilt > 45 deg            {100 * float((tilt[hidden] > 45).mean()):6.2f}%")
        print(f"   of those, root z < 0.60 m          {100 * float((rz[hidden] < 0.60).mean()):6.2f}%")
    print(f"\nepisode length mean {ep.mean():.1f}")
    print(f"trace -> {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
