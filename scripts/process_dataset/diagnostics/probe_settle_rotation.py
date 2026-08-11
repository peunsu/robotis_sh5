"""Is the residual rotation a genuine tip, or an unsettled drift the measurement window cut short?

`solve_object_lift.py` finds a lift that stops the object's CENTRE from moving (displacement 3 mm,
|v| 0.013 m/s) yet leaves ~9 deg of rotation against the captured orientation on every one of the 46
grasp-region frames, with only 1 deg of spread. A real tip-over onto a stable face would vary from
frame to frame with how far the captured pose sits from that face; a near-constant offset says the
object was still turning when the window closed. The linear-velocity rest test cannot tell these
apart, because a thin object can spin about its contact patch while its centre of mass stays put.

This records the rotation trajectory and the ANGULAR velocity at increasing settle horizons:

    rotation grows with the horizon, |w| still large   -> unsettled drift; measure longer
    rotation plateaus early, |w| -> 0                  -> genuine tip; the lift cannot remove it

    python scripts/process_dataset/diagnostics/probe_settle_rotation.py \
        --task Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0 \
        --clip_class single_rigid --clip_name s101_seg12_knife --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rotation vs settle horizon for the reference object.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--clip_class", type=str, default=None)
parser.add_argument("--clip_name", type=str, default=None)
parser.add_argument("--lifts_cm", type=float, nargs="+", default=[0.0, 0.75, 1.5])
parser.add_argument("--checkpoints", type=int, nargs="+", default=[5, 10, 20, 40, 80, 160, 320, 640])
parser.add_argument("--max_frame", type=int, default=46, help="Only the grasp region by default.")
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
    F = min(args_cli.max_frame, env._ref_len)
    # park the robot out of the scene: hand-object interpenetration is a separate problem
    root = torch.zeros(n, 7, device=dev)
    root[:, :3] = org + torch.tensor([0.0, 0.0, 5.0], device=dev)
    root[:, 3] = 1.0
    env.robot.write_root_pose_to_sim(root)
    env.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))

    cps = sorted(args_cli.checkpoints)
    rot = np.zeros((len(args_cli.lifts_cm), len(cps), F), dtype=np.float32)
    angv = np.zeros_like(rot)
    linv = np.zeros_like(rot)
    dz = np.zeros_like(rot)

    fr = torch.arange(F, device=dev)
    for li, L in enumerate(args_cli.lifts_cm):
        op = torch.zeros(n, 7, device=dev)
        op[:F, :3] = env._ref_obj_pos[fr] + org[:F]
        op[:F, 2] += float(L) * 0.01
        op[:F, 3:7] = env._ref_obj_quat[fr]
        op[F:, 3] = 1.0
        env._object.write_root_pose_to_sim(op)
        env._object.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        env.scene.write_data_to_sim()

        done = 0
        for ci, cp in enumerate(cps):
            for _ in range(cp - done):
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
            done = cp
            q = env._object.data.root_quat_w[:F]
            dot = (q * env._ref_obj_quat[fr]).sum(-1).abs().clamp(max=1.0)
            rot[li, ci] = torch.rad2deg(2 * torch.arccos(dot)).cpu().numpy()
            angv[li, ci] = env._object.data.root_ang_vel_w[:F].norm(dim=-1).cpu().numpy()
            linv[li, ci] = env._object.data.root_lin_vel_w[:F].norm(dim=-1).cpu().numpy()
            dz[li, ci] = (env._object.data.root_pos_w[:F, 2] - org[:F, 2]
                          - env._ref_obj_pos[fr, 2]).cpu().numpy()

    out = args_cli.out or f"/tmp/settle_probe_{env_cfg.clip_name}.npz"
    np.savez(out, rot=rot, angv=angv, linv=linv, dz=dz,
             lifts=np.array(args_cli.lifts_cm), checkpoints=np.array(cps))

    print(f"\n=== {env_cfg.clip_name}: frames 0-{F - 1}, mean over frames ===")
    for li, L in enumerate(args_cli.lifts_cm):
        print(f"\n  lift {L:.2f} cm")
        print(f"    {'steps':>6s}{'rot(deg)':>10s}{'|w|(rad/s)':>12s}{'|v|(m/s)':>11s}{'dz(cm)':>9s}")
        for ci, cp in enumerate(cps):
            print(f"    {cp:6d}{rot[li, ci].mean():10.2f}{angv[li, ci].mean():12.4f}"
                  f"{linv[li, ci].mean():11.4f}{dz[li, ci].mean() * 100:9.2f}")
    print(f"\ntrace -> {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
