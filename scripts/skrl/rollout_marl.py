# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run MAPPO MARL rollouts and save evaluation metrics to metrics.csv.

Mirrors `rollout.py` for the MARL grasp task. Uses sequential forward
(hand → arm sequential, hand action injected into arm obs) to match training-time
distribution; each agent's RunningStandardScaler is applied separately.

Usage:
    python scripts/skrl/rollout_marl.py \\
        --task Robotis-Sh5-Grasp-Marl-Direct-v0 \\
        --checkpoint <path/to/agent.pt> \\
        --output_dir <path/to/evaluation_ep_le_N/> \\
        --dataset oakink --object_id C11001 \\
        --trajectory_task C11001-0001-0007 --trajectory_data_id 0 \\
        --n_rollouts 32 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run MAPPO MARL rollouts and compute evaluation metrics.")
parser.add_argument("--task", type=str, required=True, help="MARL task name (must contain 'Marl').")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO checkpoint (.pt).")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to write metrics.csv into.")
parser.add_argument("--n_rollouts", type=int, default=32)
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stochastic", action="store_true", default=False,
                    help="Sample stochastic actions (default: deterministic mean).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--object_id", type=str, default=None)
parser.add_argument("--trajectory_task", type=str, default=None)
parser.add_argument("--trajectory_data_id", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

if "Marl" not in args_cli.task:
    raise ValueError(f"--task must be a MARL task (contain 'Marl'); got {args_cli.task!r}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import csv
import math
import os

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import robotis_sh5.tasks  # noqa: F401

_agent_cfg_entry_point = "skrl_mappo_cfg_entry_point"

# ManipTrans (M1) success thresholds
_M1_ET, _M1_ER, _M1_EJ, _M1_EFT = 3.0, 30.0, 8.0, 6.0

# Sequential conditioning: hand → arm. Hand action injected into arm obs slot [62:82].
ARM_HAND_SLOT_START = 62
ARM_HAND_SLOT_LEN = 20


@hydra_task_config(args_cli.task, _agent_cfg_entry_point)
def main(env_cfg: DirectMARLEnvCfg, agent_cfg: dict):
    """Run rollout evaluation for MAPPO MARL."""
    n = args_cli.n_rollouts

    env_cfg.scene.num_envs = n
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed

    # Disable curriculum mechanisms for fair evaluation.
    env_cfg.adaptive_sampling = False
    env_cfg.enable_warmup = False
    env_cfg.debug_vis = False
    env_cfg.termination = False

    if args_cli.dataset is not None:
        env_cfg.dataset = args_cli.dataset
    if args_cli.object_id is not None:
        env_cfg.object_id = args_cli.object_id
    if args_cli.trajectory_task is not None:
        env_cfg.trajectory_task = args_cli.trajectory_task
    if args_cli.trajectory_data_id is not None:
        env_cfg.trajectory_data_id = args_cli.trajectory_data_id

    agent_cfg["seed"] = args_cli.seed
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_folder = os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_folder, exist_ok=True)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[rollout_marl] Recording video to {video_folder} (length={args_cli.video_length}).")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(env, agent_cfg)

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[rollout_marl] Loading MAPPO checkpoint: {checkpoint_path}")
    # HAPPO training restructures optimizers (per-actor policy-only + dedicated
    # critic_optimizer under "arm"). Inference uses skrl's default per-agent
    # (policy+value) optimizers — loading saved state raises ValueError on
    # parameter-group size mismatch. Optimizers are unused at inference; drop.
    for uid in runner.agent.possible_agents:
        runner.agent.checkpoint_modules[uid].pop("optimizer", None)
    runner.agent.load(checkpoint_path)
    runner.agent.set_running_mode("eval")

    agent = runner.agent
    for policy in agent.policies.values():
        if policy is not None:
            policy.eval()

    actual_env = env.unwrapped

    # Per-env accumulators (one episode each — first complete trajectory).
    obj_pos_bufs = [[] for _ in range(n)]
    obj_rot_bufs = [[] for _ in range(n)]
    kpts_bufs    = [[] for _ in range(n)]
    ft_bufs      = [[] for _ in range(n)]
    reward_sums  = [0.0] * n
    episode_done = torch.zeros(n, dtype=torch.bool)

    obs, _ = env.reset()
    ref_start = int(actual_env._frame_idx[0].item())
    n_frames  = int(actual_env._max_traj_len)
    seq_name  = env_cfg.trajectory_task or env_cfg.object_id

    slot_lo = ARM_HAND_SLOT_START
    slot_hi = ARM_HAND_SLOT_START + ARM_HAND_SLOT_LEN

    for _step in range(args_cli.max_steps):
        with torch.no_grad():
            # Sequential inference (hand → arm): hand decides first, action
            # injected into arm obs slot [62:82], then arm forwards (matches training).
            hand_states_pp = agent._state_preprocessor["hand"](obs["hand"])
            hand_action, _, hand_outputs = agent.policies["hand"].act(
                {"states": hand_states_pp}, role="policy"
            )
            if not args_cli.stochastic:
                hand_action = hand_outputs.get("mean_actions", hand_action)

            arm_obs_injected = obs["arm"].clone()
            arm_obs_injected[:, slot_lo:slot_hi] = hand_action.detach()
            arm_states_pp = agent._state_preprocessor["arm"](arm_obs_injected)
            arm_action, _, arm_outputs = agent.policies["arm"].act(
                {"states": arm_states_pp}, role="policy"
            )
            if not args_cli.stochastic:
                arm_action = arm_outputs.get("mean_actions", arm_action)

            actions = {"arm": arm_action, "hand": hand_action}

        obs, rewards, terminated, truncated, _info = env.step(actions)

        # rewards is dict {agent: tensor} for MARL — sum per env across agents
        if isinstance(rewards, dict):
            reward_per_env = sum(rewards.values())
        else:
            reward_per_env = rewards
        if reward_per_env.ndim == 2:
            reward_per_env = reward_per_env.squeeze(-1)

        # done: dict for MARL — take the OR across agents (any terminated/timeout → done)
        if isinstance(terminated, dict):
            done = torch.zeros(n, dtype=torch.bool, device=reward_per_env.device)
            for v in terminated.values():
                d = v.squeeze(-1) if v.ndim == 2 else v
                done = done | d.to(done.device)
            for v in truncated.values():
                d = v.squeeze(-1) if v.ndim == 2 else v
                done = done | d.to(done.device)
        else:
            done = terminated | truncated
            if done.ndim == 2:
                done = done.squeeze(-1)
        done_cpu = done.cpu()

        for i in range(n):
            if not episode_done[i]:
                obj_pos_bufs[i].append(actual_env._last_obj_pos_err[i].item())
                obj_rot_bufs[i].append(actual_env._last_obj_rot_err[i].item())
                kpts_bufs[i].append(actual_env._last_kpts_err_raw[i].item())
                ft_bufs[i].append(actual_env._last_ft_raw_err[i].item())
                reward_sums[i] += float(reward_per_env[i].item())

        episode_done |= done_cpu
        if episode_done.all():
            break

    os.makedirs(args_cli.output_dir, exist_ok=True)
    csv_path = os.path.join(args_cli.output_dir, "metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "name", "n_frames", "ref_start",
            "success", "success_t", "success_r", "success_j", "success_ft",
            "e_t_cm", "e_r", "e_j_cm", "e_ft_cm", "reward_sum",
        ])

        for i in range(n):
            if not obj_pos_bufs[i]:
                writer.writerow(["eval", seq_name, n_frames, ref_start,
                                 0, 0, 0, 0, 0, "999.0", "999.0", "999.0", "999.0", "0.0"])
                continue

            e_t_cm  = float(sum(obj_pos_bufs[i]) / len(obj_pos_bufs[i])) * 100.0
            e_r     = math.degrees(float(sum(obj_rot_bufs[i]) / len(obj_rot_bufs[i])))
            e_j_cm  = float(sum(kpts_bufs[i])    / len(kpts_bufs[i]))    * 100.0
            e_ft_cm = float(sum(ft_bufs[i])      / len(ft_bufs[i]))      * 100.0

            s_t   = int(e_t_cm  < _M1_ET)
            s_r   = int(e_r     < _M1_ER)
            s_j   = int(e_j_cm  < _M1_EJ)
            s_ft  = int(e_ft_cm < _M1_EFT)
            s_all = int(s_t and s_r and s_j and s_ft)

            writer.writerow([
                "eval", seq_name, n_frames, ref_start,
                s_all, s_t, s_r, s_j, s_ft,
                f"{e_t_cm:.6f}", f"{e_r:.6f}", f"{e_j_cm:.6f}", f"{e_ft_cm:.6f}",
                f"{reward_sums[i]:.4f}",
            ])

    n_completed = int(episode_done.sum())
    print(f"[rollout_marl] {n_completed}/{n} episodes completed → {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
