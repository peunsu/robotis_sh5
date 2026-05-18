# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Run inference rollouts and save evaluation metrics (E_t, E_r, E_j, E_ft) to metrics.csv.

Output metrics.csv is compatible with scripts/benchmark/evaluate.bash (same column format
as workspace2/evaluation/evaluate.bash used for the inspire_OAKINK benchmark).

Usage:
    python scripts/skrl/rollout.py \\
        --task Robotis-Sh5-Grasp-Direct-v0 \\
        --checkpoint <path/to/agent.pt> \\
        --output_dir <path/to/evaluation_ep_le_N/> \\
        --dataset oakink --object_id C11001 \\
        --trajectory_task C11001-0001-0007 --trajectory_data_id 0 \\
        --n_rollouts 32 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run rollouts and compute evaluation metrics.")
parser.add_argument("--task", type=str, required=True, help="Task name (e.g. Robotis-Sh5-Grasp-Direct-v0).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt).")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to write metrics.csv into.")
parser.add_argument("--n_rollouts", type=int, default=32, help="Number of parallel rollout episodes.")
parser.add_argument("--max_steps", type=int, default=5000, help="Hard cap on simulation steps per rollout batch.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="Use policy mean actions (no sampling noise).")
# Video recording
parser.add_argument("--video", action="store_true", default=False,
                    help="Record a video of the rollout into <output_dir>/videos/.")
parser.add_argument("--video_length", type=int, default=200,
                    help="Length of the recorded video (in steps).")
# Dataset / sequence overrides
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--object_id", type=str, default=None)
parser.add_argument("--trajectory_task", type=str, default=None)
parser.add_argument("--trajectory_data_id", type=int, default=None)
# Agent config entry point
parser.add_argument(
    "--agent", type=str, default=None,
    help="Agent config entry point key (default: skrl_cfg_entry_point).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import itertools
import math
import os

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectRLEnvCfg, DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import robotis_sh5.tasks  # noqa: F401

_agent_cfg_entry_point = args_cli.agent or "skrl_cfg_entry_point"

_MASS_LR_SCALE = 33.333

# ManipTrans (M1) success thresholds
_M1_ET, _M1_ER, _M1_EJ, _M1_EFT = 3.0, 30.0, 8.0, 6.0


def _patch_mass_policy(agent, policy_cfg: dict, learning_rate: float) -> None:
    """Swap the runner-created policy for MassDexMimicPolicy so the checkpoint loads cleanly."""
    from robotis_sh5.tasks.direct.robotis_sh5_grasp.agents.mass_gaussian_model import MassDexMimicPolicy

    device = agent.device
    model_kwargs = {k: v for k, v in policy_cfg.items() if k not in ("class", "output")}

    new_policy = MassDexMimicPolicy(
        observation_space=agent.observation_space,
        action_space=agent.action_space,
        device=device,
        **model_kwargs,
    ).to(device)

    # Transfer any weights the runner already initialised (rare but safe).
    old_sd = agent.models["policy"].state_dict()
    new_sd = new_policy.state_dict()
    merged = {k: old_sd[k] if k in old_sd and old_sd[k].shape == v.shape else v for k, v in new_sd.items()}
    new_policy.load_state_dict(merged)

    agent.models["policy"] = new_policy
    agent.policy = new_policy
    agent.checkpoint_modules["policy"] = new_policy

    value = agent.models.get("value")
    mass_params = list(new_policy.mass_params())
    base_params = list(itertools.chain(
        new_policy.non_mass_params(),
        value.parameters() if value is not None else [],
    ))
    agent.optimizer = torch.optim.Adam(
        [
            {"params": base_params, "lr": learning_rate},
            {"params": mass_params, "lr": learning_rate * _MASS_LR_SCALE},
        ],
        eps=1e-8,
    )
    agent.checkpoint_modules["optimizer"] = agent.optimizer


@hydra_task_config(args_cli.task, _agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Run rollout evaluation."""
    n = args_cli.n_rollouts

    # ── Env overrides for evaluation ──────────────────────────────────────────
    env_cfg.scene.num_envs = n
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed

    # Disable stochastic curriculum mechanisms for fair evaluation.
    env_cfg.adaptive_sampling = False   # always start at frame 0
    env_cfg.enable_warmup = False
    env_cfg.debug_vis = False
    # Disable early termination so each rollout runs the full trajectory.
    # Paper E_t/E_r/E_j/E_ft are averaged over T (trajectory length); terminating
    # at frame 1-2 would average over near-zero initial errors and report
    # artificially small values.
    env_cfg.termination = False

    # Dataset / sequence overrides
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
    # Disable skrl's experiment logging — rollout only writes metrics.csv to --output_dir
    # (without this, a `./robotis_sh5_grasp/default/` folder with TensorBoard events
    # is created in the CWD every run; mirrors play.py).
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    # ── Create env & runner ───────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording (before skrl wrapper so RecordVideo sees raw gym API)
    if args_cli.video:
        video_folder = os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_folder, exist_ok=True)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[rollout] Recording video to {video_folder} (length={args_cli.video_length}).")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, agent_cfg)

    # Replace policy with MassDexMimicPolicy before loading checkpoint (grasp task only).
    _is_grasp = "Grasp" in args_cli.task and "Pretrain" not in args_cli.task
    if _is_grasp:
        _patch_mass_policy(runner.agent, agent_cfg["models"]["policy"], agent_cfg["agent"]["learning_rate"])

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[rollout] Loading checkpoint: {checkpoint_path}")
    runner.agent.load(checkpoint_path)

    policy = runner.agent.models["policy"]
    policy.eval()

    device = runner.agent.device
    actual_env = env.unwrapped  # RobotisSh5GraspEnv

    # ── Rollout loop ──────────────────────────────────────────────────────────
    # Per-env accumulators: list-of-lists; indexed by env index.
    # Paper definitions (ManipTrans):
    #   E_t  = object translation error  (cm)
    #   E_r  = object rotation error     (deg)
    #   E_j  = ||j_robot - j_human_ref|| over 21 MANO keypoints  (cm)
    #   E_ft = ||t_robot - t_human_ref|| over 5 fingertips       (cm)
    obj_pos_bufs    = [[] for _ in range(n)]   # m   → E_t
    obj_rot_bufs    = [[] for _ in range(n)]   # rad → E_r (converted to deg at save)
    kpts_bufs       = [[] for _ in range(n)]   # m   → E_j (raw ref, no drift compensation)
    ft_bufs         = [[] for _ in range(n)]   # m   → E_ft (raw ref, no contact adjustment)
    reward_sums     = [0.0] * n
    episode_done    = torch.zeros(n, dtype=torch.bool)  # CPU — tracks first-episode completion

    obs, _ = env.reset()

    # Capture trajectory metadata before any stepping.
    ref_start = int(actual_env._frame_idx[0].item())   # 0 with adaptive_sampling=False
    n_frames  = int(actual_env._max_traj_len)
    seq_name  = env_cfg.trajectory_task or env_cfg.object_id

    for _step in range(args_cli.max_steps):
        with torch.no_grad():
            actions, _, outputs = policy.act({"states": obs}, "policy")
            if args_cli.deterministic:
                actions = outputs.get("mean_actions", actions)

        obs, rewards, terminated, truncated, _info = env.step(actions)

        # Normalise done shape to (n,).
        done = (terminated | truncated)
        if done.ndim == 2:
            done = done.squeeze(-1)
        done_cpu = done.cpu()

        # Accumulate per-step errors for envs still in their first episode.
        for i in range(n):
            if not episode_done[i]:
                obj_pos_bufs[i].append(actual_env._last_obj_pos_err[i].item())
                obj_rot_bufs[i].append(actual_env._last_obj_rot_err[i].item())
                kpts_bufs[i].append(actual_env._last_kpts_err_raw[i].item())
                ft_bufs[i].append(actual_env._last_ft_raw_err[i].item())
                r = rewards[i] if rewards.ndim == 1 else rewards[i, 0]
                reward_sums[i] += float(r)

        # Update mass-policy cache for terminated envs.
        if _is_grasp:
            policy.update_mass_terminated(done)

        # Mark envs whose first episode just ended.
        episode_done |= done_cpu

        if episode_done.all():
            break

    # ── Write metrics.csv ─────────────────────────────────────────────────────
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
                # Env never contributed steps — write a failed row.
                writer.writerow(["eval", seq_name, n_frames, ref_start,
                                 0, 0, 0, 0, 0, "999.0", "999.0", "999.0", "999.0", "0.0"])
                continue

            # Convert accumulated per-step values to per-episode mean metrics.
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
    print(f"[rollout] {n_completed}/{n} episodes completed → {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
