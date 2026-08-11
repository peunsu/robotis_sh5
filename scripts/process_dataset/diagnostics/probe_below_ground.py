"""Do robots actually end up BELOW the floor during training, and where does that state come from?

Training videos keep showing robots under the ground. The reference is not the cause — measured at
the reference pose the sole sits +0.63 cm above ground (median over 501 frames, only 10 below zero) —
and neither is max_depenetration_velocity, since it happens at 1.0 as well. Holding the joint PD at
the reference does drop the root ~48 cm in 0.8 s, but that is the legs buckling with both soles still
ABOVE ground, which is what a floating-base humanoid does on frozen joint targets.

That leaves the loop itself. This runs the env the way training does — RSI resets, real stepping —
and asks three things:

    live      how far below zero any body actually goes, per step, and which body leads
    cache     whether the RSI state cache is STORING sub-floor states, i.e. whether a reset can
              teleport a robot underground. Cache column 3 is root z (row layout: 1:4 = root pos)
    reset     root z immediately after a reset, split by which branch restored it

Actions are drawn from a policy checkpoint when one is given, else zeros — the point is the reset /
cache path, which is action-independent.

    python scripts/process_dataset/diagnostics/probe_below_ground.py \
        --task Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0 \
        --clip_class single_rigid --clip_name s101_seg12_knife --headless --steps 3000
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Track sub-floor robot states and their origin.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--clip_class", type=str, default=None)
parser.add_argument("--clip_name", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=3000, help="Control steps; must exceed the uniform warm-up.")
parser.add_argument("--uniform_steps", type=int, default=None, help="Override cfg.uniform_sampling_steps.")
parser.add_argument("--action_std", type=float, default=0.0,
                    help="Gaussian action scale. 0 = zero actions (the reset/cache path is what matters).")
parser.add_argument("--out", type=str, default=None)
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
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.debug_vis = False
    if args_cli.uniform_steps is not None:
        env_cfg.uniform_sampling_steps = args_cli.uniform_steps

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    dev, n, org = env.device, env.num_envs, env.scene.env_origins
    names = list(env.robot.data.body_names)
    act_dim = env.cfg.action_space

    below_step = np.zeros(args_cli.steps, np.float32)     # deepest body below zero, this step
    frac_step = np.zeros(args_cli.steps, np.float32)      # fraction of envs with any body below zero
    rootz_min = np.zeros(args_cli.steps, np.float32)
    body_hits = np.zeros(len(names), np.int64)
    post_reset_rootz = []

    for k in range(args_cli.steps):
        a = torch.zeros(n, act_dim, device=dev)
        if args_cli.action_std > 0:
            a = torch.randn(n, act_dim, device=dev) * args_cli.action_std
        env.step(a)
        z = env.robot.data.body_pos_w[..., 2] - org[:, 2:3]        # (E,B) env-local body height
        below_step[k] = float(z.min())
        frac_step[k] = float((z.min(dim=1).values < 0).float().mean())
        rootz_min[k] = float((env.robot.data.root_pos_w[:, 2] - org[:, 2]).min())
        hit = (z < 0).any(dim=0)
        body_hits += hit.long().cpu().numpy()
        if k % 200 == 0:
            print(f"  step {k:5d}  deepest {below_step[k] * 100:+7.2f} cm  "
                  f"envs below {frac_step[k] * 100:5.1f}%  root_z min {rootz_min[k]:6.3f}")

    # what the RSI cache is holding: column 3 is root z (row 1:4 = root position, env-local)
    occ = env._slot_occ
    rz = env._state_cache[:, :, 3]
    rz_occ = rz[occ]
    out = args_cli.out or f"/tmp/below_ground_{env_cfg.clip_name}.npz"
    np.savez(out, below_step=below_step, frac_step=frac_step, rootz_min=rootz_min,
             body_hits=body_hits, body_names=np.array(names, dtype=object),
             cache_rootz=rz_occ.cpu().numpy())

    print(f"\n=== {env_cfg.clip_name}: {args_cli.steps} control steps, {n} envs ===")
    print(f"deepest body below ground, over the run : {below_step.min() * 100:+7.2f} cm")
    print(f"steps with ANY env below ground         : {int((frac_step > 0).sum()):5d} / {args_cli.steps}")
    print(f"env-fraction below ground   mean {frac_step.mean() * 100:5.2f}%   max {frac_step.max() * 100:5.2f}%")
    print(f"lowest root z seen                      : {rootz_min.min():6.3f} m")
    print("\nbodies that went below zero (steps in which it happened):")
    order = np.argsort(-body_hits)
    for i in order[:10]:
        if body_hits[i] == 0:
            break
        print(f"  {names[i]:32s} {int(body_hits[i]):5d} / {args_cli.steps}")
    print(f"\nRSI CACHE root z over {int(occ.sum())} occupied slots:")
    print(f"  min {float(rz_occ.min()):6.3f}  p1 {float(rz_occ.quantile(0.01)):6.3f}  "
          f"median {float(rz_occ.median()):6.3f}  max {float(rz_occ.max()):6.3f} m")
    print(f"  slots with root z < 0.60 m : {int((rz_occ < 0.60).sum())}")
    print(f"  slots with root z < 0.00 m : {int((rz_occ < 0.0).sum())}")
    print(f"\ntrace -> {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
