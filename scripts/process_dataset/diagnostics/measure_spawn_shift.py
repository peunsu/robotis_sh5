"""Measure, for every frame of a clip, how far PhysX moves the object away from its REFERENCE pose.

The reference object trajectory is human-capture data. Nothing guarantees it is collision-free
against the scene the robot actually stands in, so on the first physics step the solver pushes the
object out of whatever it interpenetrates. The policy is then asked to reach for a pose the object
does not occupy — and at frame 0 that is the pose every evaluation rollout starts from.

For each frame f this script writes the object to `ref_obj_pos[f] / ref_obj_quat[f]`, lets physics
settle, and records where it ended up. Frames are spread across envs so one settle pass covers many
frames at once. The robot is written to its own reference pose for the same frame, because at frames
where the reference hand already holds the object the hand is part of what determines the rest pose.

    python scripts/process_dataset/diagnostics/measure_spawn_shift.py \
        --task Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0 \
        --clip_class single_rigid --clip_name s101_seg12_knife --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Per-frame object spawn displacement.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--clip_class", type=str, default=None)
parser.add_argument("--clip_name", type=str, default=None)
parser.add_argument("--settle_steps", type=int, default=8, help="Physics steps to let the object rest.")
parser.add_argument("--out", type=str, default=None, help="npz path for the per-frame trace.")
parser.add_argument("--lift_cm", type=float, default=0.0,
                    help="Raise the object by this many cm before settling, keeping its reference\n                         ORIENTATION. A thin blade sunk into its support depenetrates about a\n                         contact point far from its centre of mass, so the solver spins it — the\n                         orientation error is the damaging part, not the ~1 cm of height. If a pure\n                         vertical offset removes the rotation, the reference can keep its captured\n                         orientation and only its height needs correcting.")
parser.add_argument("--no_robot", action="store_true",
                    help="Leave the robot at its DEFAULT pose instead of the reference pose. The\n                         reference hand often interpenetrates the object (retarget mismatch) and\n                         ejects it, which has nothing to do with object-vs-scene penetration.")
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
    env_cfg.termination = False              # nothing may reset mid-measurement
    env_cfg.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    dev = env.device
    F = env._ref_len
    n = env.num_envs
    org = env.scene.env_origins
    aid = env._action_joint_ids_t

    shift = np.full(F, np.nan, dtype=np.float32)
    rot = np.full(F, np.nan, dtype=np.float32)
    dz = np.full(F, np.nan, dtype=np.float32)

    for base in range(0, F, n):
        frames = torch.arange(base, min(base + n, F), device=dev)
        m = len(frames)
        # robot at its reference pose for the same frame (the hand can be part of the rest state)
        root = torch.zeros(n, 7, device=dev)
        root[:m, :3] = env._ref_root_pos[frames] + org[:m]
        root[:m, 3:7] = env._ref_root_quat[frames]
        root[m:, 3] = 1.0
        jp = env.robot.data.default_joint_pos.clone()
        if env._ref_joints is not None and not args_cli.no_robot:
            jp[:m, aid] = env._ref_joints[frames]
        if args_cli.no_robot:
            root[:m, :3] = org[:m] + torch.tensor([0.0, 0.0, 5.0], device=dev)   # park it out of the way
        env.robot.write_root_pose_to_sim(root)
        env.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        # object exactly at the reference pose for that frame
        op = torch.zeros(n, 7, device=dev)
        op[:m, :3] = env._ref_obj_pos[frames] + org[:m]
        op[:m, 2] += args_cli.lift_cm * 0.01
        op[:m, 3:7] = env._ref_obj_quat[frames]
        op[m:, 3] = 1.0
        env._object.write_root_pose_to_sim(op)
        env._object.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        env.scene.write_data_to_sim()

        for _ in range(args_cli.settle_steps):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

        got_p = env._object.data.root_pos_w[:m] - org[:m]
        got_q = env._object.data.root_quat_w[:m]
        want_p = env._ref_obj_pos[frames]
        want_q = env._ref_obj_quat[frames]
        d = (got_p - want_p)
        shift[base:base + m] = d.norm(dim=-1).cpu().numpy()
        dz[base:base + m] = d[:, 2].cpu().numpy()
        dot = (got_q * want_q).sum(-1).abs().clamp(max=1.0)
        rot[base:base + m] = torch.rad2deg(2 * torch.arccos(dot)).cpu().numpy()
        print(f"  frames {base:4d}-{base + m - 1:4d}: mean {shift[base:base + m].mean() * 100:6.2f} cm")

    out = args_cli.out or f"/tmp/spawn_shift_{env_cfg.clip_name}.npz"
    np.savez(out, shift=shift, rot=rot, dz=dz)

    s = shift * 100.0
    print(f"\n=== {env_cfg.clip_name}: {F} frames, settle {args_cli.settle_steps} steps ===")
    print(f"displacement cm   mean {s.mean():6.2f}  median {np.median(s):6.2f}  "
          f"p90 {np.percentile(s, 90):6.2f}  max {s.max():6.2f} (frame {int(s.argmax())})")
    print(f"rotation deg      mean {rot.mean():6.2f}  max {rot.max():6.2f}")
    for thr in (1.0, 2.0, 5.0, 8.5, 15.0):
        print(f"  frames over {thr:4.1f} cm : {int((s > thr).sum()):4d} / {F}  ({(s > thr).mean() * 100:5.1f}%)")
    print("\nfirst 60 frames (the region every evaluation rollout starts in):")
    for f in range(0, 60, 5):
        print(f"   frame {f:3d}: {s[f]:6.2f} cm   dz {dz[f] * 100:+6.2f} cm   rot {rot[f]:6.2f} deg")
    print(f"\ntrace -> {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
