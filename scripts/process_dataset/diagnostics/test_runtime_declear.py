"""Runtime penetration clearing at spawn, tested against the precomputed lift it would replace.

The offline correction (obj_settle_correction.npz) solves a per-frame lift once and bakes it into the
reference. It works, but it is solved against ONE scene: raising the context colliders from 16 to 64
convex hulls and turning up friction invalidated it, and the grasp-region spawn spread went from
+-0.3 deg back to +-10 deg. Anything that changes contact geometry silently un-corrects it.

The runtime alternative does the same thing per reset, so it cannot go stale:

    hold the object at the reference pose (rewriting pose + zero velocity every substep = kinematic
    in effect, without touching the kinematic flag, which the GPU pipeline cannot toggle per env)
    -> step once -> if the solver gave the object a velocity, it was overlapping -> raise it 0.5 mm
    -> repeat -> release when the step leaves it at rest

Reading the post-step velocity is what removes the need for a penetration query: Isaac Lab's
ContactSensor exposes forces and contact positions but not separation depth, and "the solver tried to
move a body we pinned" is exactly the same information.

This measures three conditions on the same frames — raw reference, the baked correction, and runtime
clearing — by releasing afterwards and letting each settle, which is what the reset actually does.

    python scripts/process_dataset/diagnostics/test_runtime_declear.py \
        --task Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0 \
        --clip_class single_rigid --clip_name s101_seg12_knife --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Runtime spawn-penetration clearing vs the baked lift.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--clip_class", type=str, default=None)
parser.add_argument("--clip_name", type=str, default=None)
parser.add_argument("--step_mm", type=float, default=0.5, help="Lift increment per clearing iteration.")
parser.add_argument("--max_iters", type=int, default=40, help="Cap on clearing iterations (20 mm at 0.5).")
parser.add_argument("--rest_v", type=float, default=0.09,
                    help="Post-step |v| (m/s) above which the pinned object counts as still overlapping. "
                         "Must sit ABOVE free fall: the pose is written with zero velocity, so one 5 ms "
                         "step gives even a contact-free object 9.81*0.005 = 0.049 m/s. A lower threshold "
                         "marks every AIRBORNE frame as penetrating and lifts it to the cap. Resting on a "
                         "support gives ~0 (the contact constraint cancels gravity), airborne gives 0.049, "
                         "and depenetration gives far more — 0.09 separates all three.")
parser.add_argument("--settle_steps", type=int, default=80)
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
    env_cfg.object_settle_lift = False              # raw reference; the runtime pass does the work

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    dev, n, org = env.device, env.num_envs, env.scene.env_origins
    F = env._ref_len
    # the baked correction, to score condition B without rebuilding the env
    import os
    corr = os.path.join(env._resolve_clip_dir(env_cfg), "obj_settle_correction.npz")
    baked = torch.from_numpy(np.load(corr)["lift"].astype(np.float32)).to(dev) if os.path.exists(corr) \
        else torch.zeros(F, device=dev)

    # Park the robot far above: the reference hand interpenetrates the object by ~3 cm, so leaving the
    # robot at the reference pose would have it shove the object and drown out what is being measured.
    # The base is FLOATING, so parking it once is not enough — it falls back into the scene in ~200
    # steps and starts striking things. That contaminated an earlier run in condition order (raw 5.5 cm,
    # baked 149 cm, runtime 642 cm on the same frames), which reads as the correction failing when it is
    # really the robot landing. So re-pin it every single step.
    _park = torch.zeros(n, 7, device=dev)
    _park[:, :3] = org + torch.tensor([0.0, 0.0, 5.0], device=dev)
    _park[:, 3] = 1.0
    _zero6 = torch.zeros(n, 6, device=dev)

    def _pin_robot():
        env.robot.write_root_pose_to_sim(_park)
        env.robot.write_root_velocity_to_sim(_zero6)

    def pin_and_step(pose):
        """Hold `pose` through one physics step; return the velocity the solver tried to impart."""
        env._object.write_root_pose_to_sim(pose)
        env._object.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        _pin_robot()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        return env._object.data.root_lin_vel_w.clone()

    def settle(pose, steps):
        env._object.write_root_pose_to_sim(pose)
        env._object.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        env.scene.write_data_to_sim()
        for _ in range(steps):
            _pin_robot()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        return env._object.data.root_pos_w.clone(), env._object.data.root_quat_w.clone()

    # disp    = drift away from where it was released (did the release itself disturb it)
    # disp_gt = distance from the UNCORRECTED reference, which is what the reward compares against;
    #           the lift is a spawn-only offset, so this is the price the correction charges
    res = {k: {"disp": np.zeros(F, np.float32), "disp_gt": np.zeros(F, np.float32),
               "rot": np.zeros(F, np.float32)} for k in ("raw", "baked", "rt")}
    rt_lift = np.zeros(F, np.float32)
    rt_iters = np.zeros(F, np.int32)
    rt_exit_v = np.zeros(F, np.float32)

    for base in range(0, F, n):
        fr = torch.arange(base, min(base + n, F), device=dev)
        m = len(fr)
        ref_p = torch.zeros(n, 3, device=dev)
        ref_p[:m] = env._ref_obj_pos[fr] + org[:m]
        ref_q = torch.zeros(n, 4, device=dev); ref_q[:, 0] = 1.0
        ref_q[:m] = env._ref_obj_quat[fr]

        def pose_at(dz):
            p = torch.zeros(n, 7, device=dev); p[:, 3] = 1.0
            p[:, :3] = ref_p; p[:, 2] = p[:, 2] + dz
            p[:, 3:7] = ref_q
            return p

        # --- C: runtime clearing -------------------------------------------------------------
        dz = torch.zeros(n, device=dev)
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        iters = torch.full((n,), -1.0, device=dev)     # -1 = never cleared, distinct from "cleared at 0"
        for k in range(args_cli.max_iters):
            v = pin_and_step(pose_at(dz)).norm(dim=-1)
            clear = v < args_cli.rest_v
            newly = clear & (~done)
            done = done | clear
            iters = torch.where(newly, torch.full_like(iters, float(k)), iters)
            if bool(done.all()):
                break
            dz = torch.where(done, dz, dz + args_cli.step_mm * 1e-3)
        rt_lift[base:base + m] = dz[:m].cpu().numpy()
        rt_iters[base:base + m] = iters[:m].cpu().numpy()
        # exit velocity separates the two ways a frame can be "clear": resting on a support cancels
        # gravity (~0), nothing under it does not (~0.049). Only supported frames make the settle
        # comparison meaningful — releasing an airborne frame just measures free fall.
        rt_exit_v[base:base + m] = pin_and_step(pose_at(dz)).norm(dim=-1)[:m].cpu().numpy()
        if not bool(done[:m].all()):
            print(f"    [warn] {int((~done[:m]).sum())}/{m} frames hit the {args_cli.max_iters}-iteration cap")

        # --- score all three by releasing and settling ---------------------------------------
        for tag, lift in (("raw", torch.zeros(n, device=dev)),
                          ("baked", torch.cat([baked[fr], torch.zeros(n - m, device=dev)])),
                          ("rt", dz)):
            gp, gq = settle(pose_at(lift), args_cli.settle_steps)
            tgt = ref_p.clone(); tgt[:, 2] = tgt[:, 2] + lift
            res[tag]["disp"][base:base + m] = (gp[:m] - tgt[:m]).norm(dim=-1).cpu().numpy()
            res[tag]["disp_gt"][base:base + m] = (gp[:m] - ref_p[:m]).norm(dim=-1).cpu().numpy()
            dot = (gq[:m] * ref_q[:m]).sum(-1).abs().clamp(max=1.0)
            res[tag]["rot"][base:base + m] = torch.rad2deg(2 * torch.arccos(dot)).cpu().numpy()
        print(f"  frames {base:4d}-{base + m - 1:4d}")

    out = args_cli.out or f"/tmp/rt_declear_{env_cfg.clip_name}.npz"
    np.savez(out, rt_lift=rt_lift, rt_iters=rt_iters, rt_exit_v=rt_exit_v,
             grasp_frame=env._grasp_frame,
             **{f"{k}_{q}": res[k][q] for k in res for q in res[k]})

    gf = int(env._grasp_frame)
    ok = rt_iters >= 0
    # A frame counts as penetrating if it needed ANY lift. Classifying by the exit velocity does not
    # work: clearing stops the moment the object leaves contact, so afterwards essentially everything
    # reads as free fall (it labelled 9/501 knife frames "supported" when 31/31 pre-grasp ones overlap).
    pen = rt_iters > 0
    idx = np.arange(F)
    # Releasing the object only measures the spawn when the reference has it resting on something.
    # Past the grasp the reference carries it in the air, so a release there measures where it happens
    # to land — the comparison is not meaningful, and it is reported only for completeness.
    windows = [(f"pre-grasp 0-{max(gf - 1, 0)} (object at rest)", idx < max(gf, 1)),
               ("penetrating frames", pen), ("all frames (incl. airborne)", np.ones(F, bool))]

    print(f"\n=== {env_cfg.clip_name}: {F} frames, grasp frame {gf} ===")
    print(f"clearing: resolved {int(ok.sum())}/{F}   penetrating {int(pen.sum())} "
          f"(pre-grasp {int(pen[:max(gf, 1)].sum())}/{max(gf, 1)})   "
          f"iterations mean {rt_iters[pen].mean():.1f} max {int(rt_iters.max())}")
    if pen.any():
        print(f"lift over penetrating frames: mean {rt_lift[pen].mean() * 100:.2f} cm  "
              f"max {rt_lift[pen].max() * 100:.2f} cm")

    for wname, W in windows:
        if not W.any():
            continue
        print(f"\n-- {wname}  ({int(W.sum())} frames)")
        print(f"{'condition':12s}{'rot mean':>10s}{'rot std':>10s}{'rot min':>10s}{'rot max':>10s}"
              f"{'drift cm':>10s}{'vs GT cm':>10s}")
        for tag, lab in (("raw", "no fix"), ("baked", "baked lift"), ("rt", "runtime")):
            r, d, g = res[tag]["rot"][W], res[tag]["disp"][W], res[tag]["disp_gt"][W]
            print(f"{lab:12s}{r.mean():10.2f}{r.std():10.2f}{r.min():10.2f}{r.max():10.2f}"
                  f"{d.mean() * 100:10.2f}{g.mean() * 100:10.2f}")
    print(f"\ntrace -> {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
