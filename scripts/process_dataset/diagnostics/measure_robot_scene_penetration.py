"""Which robot links start INSIDE the fixed scene, and how hard does PhysX throw them out?

The context objects (counter, cutting board, ...) spawn KINEMATIC — infinite mass — so every bit of
a robot/scene overlap is corrected by moving the ROBOT, and the base is floating with nothing to
absorb it. Training shows the robot reaching 10.4 m/s at the root (p99 1.60) while the reference
root never exceeds 0.92 m/s, i.e. it routinely moves faster than anything the reference asks for.

Reading it off velocity after ONE step makes the test unambiguous. A body at rest and touching
nothing picks up only gravity: g*dt = 9.81 * 0.005 = 0.049 m/s. A body being depenetrated picks up
whatever the solver needs, up to max_depenetration_velocity (1.0 m/s on this robot). So

    speed after one step >> 0.05 m/s   ==>   that body started inside something

and the magnitude is the severity. The joint PD targets are set to the pose being tested, so the
actuators have zero error and cannot be the thing that moves it.

Runs twice — context spawned, and context suppressed (radius 0) — because the ground plane and
gravity act in both. The DIFFERENCE is the robot-vs-fixed-scene contact, isolated.

    python scripts/process_dataset/diagnostics/measure_robot_scene_penetration.py \
        --task Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0 \
        --clip_class single_rigid --clip_name s101_seg12_knife --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Robot vs fixed-scene penetration, per frame and per link.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--clip_class", type=str, default=None)
parser.add_argument("--clip_name", type=str, default=None)
parser.add_argument("--steps", type=int, default=1, help="Physics steps before reading the velocity.")
parser.add_argument("--thresh", type=float, default=0.20,
                    help="Body speed (m/s) above which the body is called PENETRATING. Well clear of the "
                         "0.049 m/s a free body gains from gravity in one 5 ms step.")
parser.add_argument("--no_object", action="store_true", help="Also suppress the manipulated object.")
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


def _measure(env_cfg, task, with_context):
    """Return (F, B) body speed one step after being placed at each frame's reference pose."""
    if not with_context:
        env_cfg.context_radius = 0.0
        env_cfg.context_support_radius = 0.0
    env = gym.make(task, cfg=env_cfg).unwrapped
    env.reset()
    dev, n, org = env.device, env.num_envs, env.scene.env_origins
    F = env._ref_len
    aid = env._action_joint_ids_t
    B = env.robot.num_bodies
    out = np.zeros((F, B), dtype=np.float32)
    root_sp = np.zeros(F, dtype=np.float32)

    for base in range(0, F, n):
        fr = torch.arange(base, min(base + n, F), device=dev)
        m = len(fr)
        root = torch.zeros(n, 7, device=dev)
        root[:m, :3] = env._ref_root_pos[fr] + org[:m]
        root[:m, 3:7] = env._ref_root_quat[fr]
        root[m:, 3] = 1.0
        jp = env.robot.data.default_joint_pos.clone()
        if env._ref_joints is not None:
            jp[:m, aid] = env._ref_joints[fr]
        env.robot.write_root_pose_to_sim(root)
        env.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        # PD target == the pose under test, so any motion is gravity or contact, never tracking error
        env.robot.set_joint_position_target(jp[:, aid], joint_ids=env._action_joint_ids)
        if env._has_object:
            op = torch.zeros(n, 7, device=dev)
            op[:m, :3] = env._ref_obj_pos[fr] + org[:m]
            op[:m, 3:7] = env._ref_obj_quat[fr]
            op[m:, 3] = 1.0
            if args_cli.no_object:
                # park it out of reach: a finger sunk into the KNIFE also gets thrown, and that is a
                # retarget problem against a 0.5 kg dynamic body, not the fixed-scene contact under test
                op[:, 2] += 10.0
            env._object.write_root_pose_to_sim(op)
            env._object.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        env.scene.write_data_to_sim()
        for _ in range(args_cli.steps):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        out[base:base + m] = env.robot.data.body_lin_vel_w[:m].norm(dim=-1).cpu().numpy()
        root_sp[base:base + m] = env.robot.data.root_lin_vel_w[:m].norm(dim=-1).cpu().numpy()
        print(f"  frames {base:4d}-{base + m - 1:4d}")
    names = list(env.robot.data.body_names)
    ctx = [p.split('/')[-1] for p in getattr(env, "_ctx_prims", [])]
    env.close()
    return out, root_sp, names, ctx


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.clip_class:
        env_cfg.clip_class = args_cli.clip_class
    if args_cli.clip_name:
        env_cfg.clip_name = args_cli.clip_name
    env_cfg.scene.num_envs = 256
    env_cfg.termination = False
    env_cfg.debug_vis = False

    print("\n=== WITH context (fixed scene objects spawned) ===")
    sp_on, root_on, names, ctx = _measure(env_cfg, args_cli.task, True)
    print(f"context objects: {ctx}")

    out = args_cli.out or f"/tmp/robot_scene_pen_{env_cfg.clip_name}.npz"
    np.savez(out, speed_ctx=sp_on, root_ctx=root_on, body_names=np.array(names, dtype=object),
             ctx_prims=np.array(ctx, dtype=object), thresh=args_cli.thresh)

    F, B = sp_on.shape
    thr = args_cli.thresh
    hit = sp_on > thr                                     # (F,B)
    per_body = hit.sum(axis=0)
    per_frame = hit.sum(axis=1)
    print(f"\n=== {env_cfg.clip_name}: {F} frames x {B} bodies, one {1000 * 0.005:.0f} ms step ===")
    print(f"free-fall reference: a body touching nothing reaches 0.049 m/s; threshold {thr} m/s\n")
    print(f"frames with ANY body over threshold : {int((per_frame > 0).sum()):4d} / {F}")
    print(f"root speed  mean {root_on.mean():6.3f}  p95 {np.percentile(root_on, 95):6.3f}  max {root_on.max():6.3f} m/s")
    print("\nworst bodies (frames over threshold, and their peak speed):")
    order = np.argsort(-per_body)
    for i in order[:14]:
        if per_body[i] == 0:
            break
        print(f"  {names[i]:32s} {int(per_body[i]):4d}/{F} frames   peak {sp_on[:, i].max():6.3f} m/s")
    print("\nper-frame worst body:")
    for f in list(range(0, 60, 10)) + [100, 200, 300, 400, F - 1]:
        if f >= F:
            continue
        j = int(np.argmax(sp_on[f]))
        print(f"  frame {f:4d}: {names[j]:30s} {sp_on[f, j]:6.3f} m/s   (bodies over thr: {int(hit[f].sum())})")
    print(f"\ntrace -> {out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
