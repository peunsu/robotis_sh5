"""Does the retargeted reference put the robot's feet through the floor, and does it then sink?

Robots were seen dropping below the ground in some envs but not others, at every
max_depenetration_velocity tried — so the cap is not the cause and the reference itself is the
suspect. An earlier offline check said the feet were 6-7 cm under, but that measured pyroki's
COLLISION CAPSULES, which wrap each link in its minimum bounding cylinder: the ankle link's four
5 mm contact spheres became a 7.97 cm-radius capsule, 2.4x the real sole. Useless for this question.

So measure what PhysX actually uses. The USD foot mesh spans z = -0.0354 .. +0.0236 in the ankle
frame, so sole_z = ankle_z - 0.0354, and the retarget's floor_contact aims the ankle origin at
_ANKLE_SOLE_OFF = 0.036 — i.e. the sole should land within a millimetre of zero.

Two things are reported per frame:

    static   sole_z at the reference pose, before any physics. Negative = the reference is asking
             for a foot inside the floor.
    settled  root z drift after N steps with the PD held at the reference pose. Rising means the
             solver is pushing the robot out of the floor; falling means it is sinking.

    python scripts/process_dataset/diagnostics/measure_foot_ground.py \
        --task Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0 \
        --clip_class single_rigid --clip_name s101_seg12_knife --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reference foot height vs the floor, and settle drift.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--clip_class", type=str, default=None)
parser.add_argument("--clip_name", type=str, default=None)
parser.add_argument("--sole_offset", type=float, default=0.0354,
                    help="Ankle-frame depth of the sole, from the USD foot mesh (z_min = -0.0354).")
parser.add_argument("--settle_steps", type=int, default=60)
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
    env_cfg.scene.num_envs = 256
    env_cfg.termination = False
    env_cfg.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    dev, n, org = env.device, env.num_envs, env.scene.env_origins
    F = env._ref_len
    aid = env._action_joint_ids_t
    ank = [env.robot.find_bodies(b)[0][0] for b in ("left_ankle_roll_link", "right_ankle_roll_link")]

    sole0 = np.zeros((F, 2), np.float32)      # sole z at the reference pose, before physics
    sole1 = np.zeros((F, 2), np.float32)      # after one step
    rootz0 = np.zeros(F, np.float32)
    rootzN = np.zeros(F, np.float32)

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
        env.robot.set_joint_position_target(jp[:, aid], joint_ids=env._action_joint_ids)
        env.scene.write_data_to_sim()
        env.scene.update(dt=env.physics_dt)                      # refresh views without stepping physics

        z = env.robot.data.body_pos_w[:m][:, ank, 2] - org[:m, 2:3]
        sole0[base:base + m] = (z - args_cli.sole_offset).cpu().numpy()
        rootz0[base:base + m] = (env.robot.data.root_pos_w[:m, 2] - org[:m, 2]).cpu().numpy()

        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        z = env.robot.data.body_pos_w[:m][:, ank, 2] - org[:m, 2:3]
        sole1[base:base + m] = (z - args_cli.sole_offset).cpu().numpy()

        for _ in range(args_cli.settle_steps - 1):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        rootzN[base:base + m] = (env.robot.data.root_pos_w[:m, 2] - org[:m, 2]).cpu().numpy()
        print(f"  frames {base:4d}-{base + m - 1:4d}")

    out = args_cli.out or f"/tmp/foot_ground_{env_cfg.clip_name}.npz"
    np.savez(out, sole0=sole0, sole1=sole1, rootz0=rootz0, rootzN=rootzN)

    lo = sole0.min(axis=1)
    drift = rootzN - rootz0
    print(f"\n=== {env_cfg.clip_name}: {F} frames  (sole = ankle_z - {args_cli.sole_offset}) ===")
    print("\nSTATIC — sole height at the reference pose, before physics")
    print(f"  lower sole  min {lo.min() * 100:+7.2f}  p5 {np.percentile(lo, 5) * 100:+7.2f}  "
          f"median {np.median(lo) * 100:+7.2f}  max {lo.max() * 100:+7.2f} cm")
    for thr in (0.0, -0.005, -0.01, -0.02):
        print(f"  frames with a sole below {thr * 100:+5.1f} cm : {int((lo < thr).sum()):4d} / {F}")
    print("\nSETTLE — root z drift after "
          f"{args_cli.settle_steps} steps with the PD held at the reference")
    print(f"  drift cm    min {drift.min() * 100:+7.2f}  median {np.median(drift) * 100:+7.2f}  "
          f"max {drift.max() * 100:+7.2f}")
    print(f"  frames that SANK more than 1 cm : {int((drift < -0.01).sum()):4d} / {F}")
    print(f"  frames PUSHED UP more than 1 cm : {int((drift > 0.01).sum()):4d} / {F}")
    print("\nframe   sole L / R (cm)      after 1 step      root drift")
    for f in list(range(0, 60, 10)) + [100, 200, 300, 400, F - 1]:
        if f >= F:
            continue
        print(f"{f:5d}   {sole0[f, 0] * 100:+6.2f} {sole0[f, 1] * 100:+6.2f}   "
              f"{sole1[f, 0] * 100:+6.2f} {sole1[f, 1] * 100:+6.2f}   {drift[f] * 100:+6.2f}")
    print(f"\ntrace -> {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
