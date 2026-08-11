"""Object-aware retargeting post-processing, PARALLEL (dexmachina/SPIDER appendix A.2).

One env per frame. Same procedure as the serial version — fixate the object at its reference pose,
fixate the robot base at its reference pose, command the retargeted joints as PD targets, step until
settled, keep the achieved joints — but a 501-frame clip becomes 2 batches instead of 501 sequential
solves: 400 physics steps instead of 100,200, ~30 s instead of ~28 min.

Built on the TRAINING env rather than a hand-rolled scene, so the object spawn, the 19 kinematic
context colliders and the collision filtering are exactly what training uses. The only override is
fix_root_link.

Two things learned the hard way and encoded here:

  * The base must be GENUINELY fixed. Holding a floating base by rewriting root pose + zero velocity
    every substep discards the momentum PhysX integrated, and the reaction lands in the joints: the
    left arm ran 1.5-2.4 rad off target at 20-78 rad/s and contact force ended HIGHER than it started
    (33 -> 48 N). fix_root_link is a spawn-time property, hence the cfg override.

  * The settle does not converge at the stock finger velocity limit. shadow_fingers has
    velocity_limit_sim=15.0 and mid-clip frames ended the settle pinned at exactly 15.0 rad/s with
    the deviation still growing (0.021 rad at frame 0 -> 0.122 at frame 350) — a steady-state
    PD-into-object fight, not an equilibrium. More steps do not help; a lower limit does, because it
    caps how hard the PD can drive into the contact. --vel_limit exposes it, and the summary reports
    whether frames actually came to rest so this cannot pass silently again.

    python scripts/process_dataset/retarget/object_aware_parallel.py \
        --clip s101_seg12_knife --vel_limit 2.0 --settle_steps 200
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Parallel object-aware retarget post-processing.")
parser.add_argument("--task", type=str, default="Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0")
parser.add_argument("--clip_class", type=str, default="single_rigid")
parser.add_argument("--clip", type=str, default="s101_seg12_knife")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--settle_steps", type=int, default=200)
parser.add_argument("--vel_limit", type=float, default=2.0,
                    help="shadow_fingers velocity_limit_sim during the solve (stock 15.0). Caps how "
                         "hard the PD can drive into the object, which is what stops the settle from "
                         "converging.")
parser.add_argument("--avg_last", type=int, default=20,
                    help="average the achieved joints over the final N steps, so a frame that is "
                         "still oscillating contributes its mean instead of a random phase snapshot.")
parser.add_argument("--out_dir", type=str, default="/tmp")
parser.add_argument("--save", action="store_true")
parser.add_argument("--no_ctx", action="store_true", help="select zero context objects, to attribute arm runaway")
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
    env_cfg.clip_class = args_cli.clip_class
    env_cfg.clip_name = args_cli.clip
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.termination = False
    env_cfg.debug_vis = False
    env_cfg.obj_guidance = False           # this script IS the fixation
    env_cfg.object_spawn_declear = False   # solve against the raw capture
    env_cfg.robot_cfg.spawn.articulation_props.fix_root_link = True
    if args_cli.no_ctx:
        env_cfg.context_radius = 0.0
        env_cfg.context_support_radius = 0.0
    if "shadow_fingers" in env_cfg.robot_cfg.actuators:
        env_cfg.robot_cfg.actuators["shadow_fingers"].velocity_limit_sim = args_cli.vel_limit

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    dev, n, org = env.device, env.num_envs, env.scene.env_origins
    F, aid = env._ref_len, env._action_joint_ids_t
    if env._ref_joints is None:
        print("[objaware] no pyroki retarget for this clip.")
        env.close()
        return
    z6 = torch.zeros(n, 6, device=dev)
    hsl = env._sonic_hand_slice

    def place(fr, m, jtar):
        rp = torch.zeros(n, 7, device=dev)
        rp[:, 3] = 1.0
        rp[:m, :3] = env._ref_root_pos[fr] + org[:m]
        rp[:m, 3:7] = env._ref_root_quat[fr]
        env.robot.write_root_pose_to_sim(rp)
        env.robot.write_root_velocity_to_sim(z6)
        if env._has_object:
            op = torch.zeros(n, 7, device=dev)
            op[:, 3] = 1.0
            op[:m, :3] = env._ref_obj_pos[fr] + org[:m]
            op[:m, 3:7] = env._ref_obj_quat[fr]
            env._object.write_root_pose_to_sim(op)
            env._object.write_root_velocity_to_sim(z6)
        env.robot.set_joint_position_target(jtar[:, aid], joint_ids=env._action_joint_ids)
        return rp

    achieved = np.zeros((F, env._ref_joints.shape[1]), np.float32)
    jvel_end = np.zeros(F, np.float32)
    root_err = np.zeros(F, np.float32)

    for base in range(0, F, n):
        fr = torch.arange(base, min(base + n, F), device=dev)
        m = len(fr)
        jp = env.robot.data.default_joint_pos.clone()
        jp[:m, aid] = env._ref_joints[fr]
        env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        want = place(fr, m, jp)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        # SELF-CHECK: with fix_root_link the root link is welded to the world, so a written root pose
        # may be ignored or fought. If it does not hold, everything downstream is solved at the wrong
        # place and the numbers would look fine while meaning nothing.
        root_err[base:base + m] = (env.robot.data.root_pos_w[:m] - want[:m, :3]).norm(dim=-1).cpu().numpy()

        acc = torch.zeros(m, env._ref_joints.shape[1], device=dev)
        for k in range(args_cli.settle_steps):
            place(fr, m, jp)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            if k >= args_cli.settle_steps - args_cli.avg_last:
                acc += env.robot.data.joint_pos[:m][:, aid]
        achieved[base:base + m] = (acc / max(args_cli.avg_last, 1)).cpu().numpy()
        jvel_end[base:base + m] = env.robot.data.joint_vel[:m][:, aid].abs().max(dim=-1).values.cpu().numpy()
        print(f"  frames {base:4d}-{base + m - 1:4d}")

    ref = env._ref_joints.cpu().numpy()
    d = np.abs(achieved - ref)
    hi = np.arange(hsl.start, hsl.stop)
    bi = np.arange(0, hsl.start)
    print(f"\n=== {args_cli.clip}: {F} frames | fixed base | vel_limit {args_cli.vel_limit} | "
          f"{args_cli.settle_steps} steps (avg last {args_cli.avg_last}) ===")
    print(f"root placement error (must be ~0, else fix_root_link ignored the written pose):")
    print(f"  mean {root_err.mean() * 1000:.3f} mm   max {root_err.max() * 1000:.3f} mm")
    print(f"\njoint deviation |achieved - retarget| (rad)")
    print(f"  all    mean {d.mean():.4f}  p95 {np.percentile(d, 95):.4f}  max {d.max():.4f}")
    print(f"  hands  mean {d[:, hi].mean():.4f}  p95 {np.percentile(d[:, hi], 95):.4f}  max {d[:, hi].max():.4f}")
    print(f"  body   mean {d[:, bi].mean():.4f}  max {d[:, bi].max():.4f}")
    print(f"\nsettled? joint speed at end of settle (rad/s), limit {args_cli.vel_limit}")
    print(f"  mean {jvel_end.mean():.3f}  p95 {np.percentile(jvel_end, 95):.3f}  max {jvel_end.max():.3f}")
    at_lim = float((jvel_end > 0.9 * args_cli.vel_limit).mean()) * 100
    print(f"  frames still pinned at >90% of the limit: {at_lim:.1f}%   "
          f"({'NOT converged' if at_lim > 10 else 'converged'})")

    out = os.path.join(args_cli.out_dir, f"objaware_par_{args_cli.clip}.npz")
    # export the ACTION-joint order these columns are in. env._ref_joints has already been permuted
    # by _remap_ref_joints (24/65 slots move), so trajectory_pyroki.npz's own joint_names is the WRONG
    # label for them — reading them back with that order scrambles the fingers.
    act_names = np.array([env.robot.data.joint_names[i] for i in env._action_joint_ids], dtype=object)
    np.savez(out, achieved=achieved, ref=ref, jvel_end=jvel_end, root_err=root_err,
             joint_names=act_names)
    names = [env.robot.data.joint_names[i] for i in env._action_joint_ids]
    worst = np.argsort(-d.max(axis=0))[:8]
    print("\nworst joints by max deviation (rad):")
    for j in worst:
        print(f"  {names[j]:28s} max {d[:, j].max():7.3f}  mean {d[:, j].mean():6.3f}")
    print(f"\ntrace -> {out}")
    if args_cli.save:
        g = env._resolve_clip_dir(env.cfg).replace("/smplx/", "/g1_shadow/")
        # achieved is in ACTION order; the env will permute BY NAME on load, so write the names it is
        # actually in rather than copying the source file's order.
        idx = np.round(np.linspace(0, F - 1, int(round((F - 1) * 30.0 / 50.0)) + 1)).astype(int)
        p = os.path.join(g, "trajectory_pyroki_objaware.npz")
        np.savez(p, g1_joint_pos=achieved[idx], joint_names=act_names)
        print(f"wrote {p}  ({len(idx)} frames @ 30 fps)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
