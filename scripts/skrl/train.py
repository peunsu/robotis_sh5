# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--timesteps", type=int, default=None, help="Total environment steps to train; overrides agent YAML trainer.timesteps.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# Dataset / sequence overrides — allow per-sequence training from bash scripts.
parser.add_argument("--dataset", type=str, default=None, help="Dataset name (e.g. 'oakink'); overrides env_cfg.dataset.")
parser.add_argument("--object_id", type=str, default=None, help="Object ID; overrides env_cfg.object_id.")
parser.add_argument("--trajectory_task", type=str, default=None, help="Trajectory task directory name; overrides env_cfg.trajectory_task.")
parser.add_argument("--trajectory_data_id", type=int, default=None, help="Trajectory data sub-index; overrides env_cfg.trajectory_data_id.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import robotis_sh5.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


_MASS_LR_SCALE = 33.333  # mass optimizer group: 33.333× higher LR (matches original rl_games config)
_ENTROPY_FLIP_SCALE = -0.002  # entropy coef when is_reached_end=True (GR: -0.002 * sigma_weight)


def _patch_mass_policy(agent, policy_cfg: dict, learning_rate: float) -> None:
    """Replace the Runner-created policy with MassDexMimicPolicy and rebuild the optimizer.

    Must be called AFTER runner = Runner(env, agent_cfg) but BEFORE checkpoint loading.
    This implements Section 3.2 of MassDexMimic:
      - mu_mass and log_std_mass as separate learnable scalars
      - 33.333× higher LR for mass parameters
      - Per-episode mass cache (fixed within an episode, resampled on termination)

    policy_cfg: the ``models.policy`` dict from agent_cfg (YAML), used to pass network
    architecture and hyperparameters through to MassDexMimicPolicy.
    """
    import itertools
    import torch
    from robotis_sh5.tasks.direct.robotis_sh5_grasp.agents.mass_gaussian_model import MassDexMimicPolicy

    old_policy = agent.models["policy"]
    device = agent.device

    # Forward all YAML policy fields to the constructor; 'class' and 'output' are
    # handled internally and must be excluded.
    model_kwargs = {k: v for k, v in policy_cfg.items() if k not in ("class", "output")}

    # Create new policy with mass-specific parameters.
    new_policy = MassDexMimicPolicy(
        observation_space=agent.observation_space,
        action_space=agent.action_space,
        device=device,
        **model_kwargs,
    ).to(device)

    # Transfer overlapping weights from Runner-created policy to new policy.
    old_sd = old_policy.state_dict()
    new_sd = new_policy.state_dict()
    merged = {}
    for k, v in new_sd.items():
        if k in old_sd and old_sd[k].shape == v.shape:
            merged[k] = old_sd[k]
        else:
            merged[k] = v  # keep default init (e.g., mu_mass=-0.25, log_std_mass=-1.25)
    new_policy.load_state_dict(merged)

    # Replace policy in agent.
    agent.models["policy"] = new_policy
    agent.policy = new_policy
    agent.checkpoint_modules["policy"] = new_policy

    # Rebuild optimizer with separate parameter groups.
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

    # Rebuild LR scheduler if configured.
    if agent._learning_rate_scheduler is not None:
        agent.scheduler = agent._learning_rate_scheduler(
            agent.optimizer, **agent.cfg["learning_rate_scheduler_kwargs"]
        )

    # Monkey-patch record_transition to propagate terminated signal to the mass cache.
    _orig_record = agent.record_transition

    def _record_with_mass(states, actions, rewards, next_states, terminated, truncated,
                          infos, timestep, timesteps):
        done = (terminated | truncated).squeeze(-1)
        new_policy.update_mass_terminated(done)
        _orig_record(states, actions, rewards, next_states, terminated, truncated,
                     infos, timestep, timesteps)

    agent.record_transition = _record_with_mass

    print(f"[mass_policy] Patched policy with MassDexMimicPolicy "
          f"(base_lr={learning_rate:.2e}, mass_lr={learning_rate * _MASS_LR_SCALE:.4f})")


def _patch_env_info_log(agent) -> None:
    """Wrap agent.record_transition to manually iterate over `infos["log"]` and call
    agent.track_data(key, value) for each scalar. Bypasses skrl trainer's hard-coded
    "Info / " prefix (when `environment_info="__disabled__"`) so env-emitted keys
    like "Error / X", "Episode_Reward / X", "Curriculum / X" land directly under
    those Tensorboard tabs. Applies unconditionally — for both train and pretrain.
    """
    _orig_record = agent.record_transition

    def _record_with_env_log(states, actions, rewards, next_states, terminated, truncated,
                             infos, timestep, timesteps):
        _orig_record(states, actions, rewards, next_states, terminated, truncated,
                     infos, timestep, timesteps)
        log_dict = infos.get("log") if isinstance(infos, dict) else None
        if log_dict:
            for k, v in log_dict.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    agent.track_data(k, v.item())

    agent.record_transition = _record_with_env_log
    print(f"[env-info-log] Manual env extras['log'] → agent.track_data (bypasses skrl 'Info / ' prefix).")


def _patch_entropy_flip(agent, env_wrapper, base_entropy_scale: float) -> None:
    """GR-faithful entropy scheduling.

    Faithfully mirrors GR (rl_games/a2c_continuous.py):
      1. At each rollout step: store is_reached_end as sigma_grad_flg in the rollout memory.
      2. At training time, per mini-batch:
           sigma_weight = sigma_grad_flg[0].float()   # first element of mini-batch
           entropy_weight = entropy_coef * (1 - sigma_weight) - 0.002 * sigma_weight
           loss -= entropy * entropy_weight

    skrl sign convention: entropy_loss = -entropy_scale * entropy.mean()
      entropy_scale = entropy_coef * (1 - sw) - 0.002 * sw   (same formula as GR)
    """
    import itertools
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from skrl import config
    from skrl.resources.schedulers.torch import KLAdaptiveLR

    # ── Step 1: Add sigma_grad_flg tensor to rollout memory ───────────────────
    agent.memory.create_tensor(name="sigma_grad_flg", size=1, dtype=torch.float32)
    agent._tensors_names.append("sigma_grad_flg")

    # ── Step 2: Store sigma_grad_flg at each rollout step ─────────────────────
    _orig_record = agent.record_transition

    def _record_with_sigma(states, actions, rewards, next_states, terminated, truncated,
                           infos, timestep, timesteps):
        _orig_record(states, actions, rewards, next_states, terminated, truncated,
                     infos, timestep, timesteps)
        # Write is_reached_end into the slot just committed by add_samples.
        # add_samples increments memory_index after writing, so the written slot is index-1.
        prev_idx = (agent.memory.memory_index - 1) % agent.memory.memory_size
        sigma = float(env_wrapper.unwrapped.is_reached_end)
        agent.memory.tensors["sigma_grad_flg"][prev_idx].fill_(sigma)

    agent.record_transition = _record_with_sigma

    # ── Step 3: Replace _update with per-mini-batch entropy weighting ─────────
    def _update_with_per_batch_entropy(timestep: int, timesteps: int) -> None:

        def compute_gae(rewards, dones, values, next_values,
                        discount_factor=0.99, lambda_coefficient=0.95):
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]
            for i in reversed(range(memory_size)):
                next_v = values[i + 1] if i < memory_size - 1 else next_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_v + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        with torch.no_grad(), torch.autocast(device_type=agent._device_type, enabled=agent._mixed_precision):
            agent.value.train(False)
            last_values, _, _ = agent.value.act(
                {"states": agent._state_preprocessor(agent._current_next_states.float())}, role="value"
            )
            agent.value.train(True)
            last_values = agent._value_preprocessor(last_values, inverse=True)

        values = agent.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=agent.memory.get_tensor_by_name("rewards"),
            dones=(agent.memory.get_tensor_by_name("terminated")
                   | agent.memory.get_tensor_by_name("truncated")),
            values=values,
            next_values=last_values,
            discount_factor=agent._discount_factor,
            lambda_coefficient=agent._lambda,
        )
        agent.memory.set_tensor_by_name("values", agent._value_preprocessor(values, train=True))
        agent.memory.set_tensor_by_name("returns", agent._value_preprocessor(returns, train=True))
        agent.memory.set_tensor_by_name("advantages", advantages)

        sampled_batches = agent.memory.sample_all(
            names=agent._tensors_names, mini_batches=agent._mini_batches
        )

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        for epoch in range(agent._learning_epochs):
            kl_divergences = []

            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_sigma,          # sigma_grad_flg: (mini_batch_size, 1)
            ) in sampled_batches:

                with torch.autocast(device_type=agent._device_type, enabled=agent._mixed_precision):

                    sampled_states = agent._state_preprocessor(sampled_states, train=not epoch)

                    _, next_log_prob, _ = agent.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if agent._kl_threshold and kl_divergence > agent._kl_threshold:
                        break

                    # ── GR-faithful per-mini-batch entropy weight ──────────
                    sw = sampled_sigma[0].item()   # first element (scalar), mirrors GR sigma_grad_flg[0]
                    entropy_scale = base_entropy_scale * (1.0 - sw) + _ENTROPY_FLIP_SCALE * sw
                    entropy_loss = -entropy_scale * agent.policy.get_entropy(role="policy").mean()

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - agent._ratio_clip, 1.0 + agent._ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _, _ = agent.value.act({"states": sampled_states}, role="value")
                    if agent._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-agent._value_clip, max=agent._value_clip,
                        )
                    value_loss = agent._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                agent.optimizer.zero_grad()
                agent.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    agent.policy.reduce_parameters()
                    if agent.policy is not agent.value:
                        agent.value.reduce_parameters()

                if agent._grad_norm_clip > 0:
                    agent.scaler.unscale_(agent.optimizer)
                    if agent.policy is agent.value:
                        nn.utils.clip_grad_norm_(agent.policy.parameters(), agent._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(agent.policy.parameters(), agent.value.parameters()),
                            agent._grad_norm_clip,
                        )

                agent.scaler.step(agent.optimizer)
                agent.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                cumulative_entropy_loss += entropy_loss.item()

            if agent._learning_rate_scheduler:
                if isinstance(agent.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=agent.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    agent.scheduler.step(kl.item())
                else:
                    agent.scheduler.step()

        n = agent._learning_epochs * agent._mini_batches
        agent.track_data("Loss / Policy loss", cumulative_policy_loss / n)
        agent.track_data("Loss / Value loss", cumulative_value_loss / n)
        agent.track_data("Loss / Entropy loss", cumulative_entropy_loss / n)
        agent.track_data("Policy / Standard deviation",
                         agent.policy.distribution(role="policy").stddev.mean().item())
        if agent._learning_rate_scheduler:
            agent.track_data("Policy / Learning rate", agent.scheduler.get_last_lr()[0])

    agent._update = _update_with_per_batch_entropy
    print(f"[entropy_flip] GR-faithful: base={base_entropy_scale}, flip={_ENTROPY_FLIP_SCALE}")


# Parameter names to keep at their YAML-initialized values when loading a pretrain
# checkpoint into the train policy (mirrors TJ/rl_games tools/reset_epoch.py, which
# pops `a2c_network.sigma` so train starts with fresh exploration noise).
# `log_std_mass` and `mu_mass` are absent from the pretrain ckpt, so they are
# already freshly initialized via the "not in checkpoint" branch below.
_PRETRAIN_LOAD_SKIP_KEYS = {"log_std_parameter"}


def _load_partial_checkpoint(agent, path: str) -> None:
    """Load a checkpoint with mismatched input/output sizes (pretrain → train transfer).

    For each parameter in the checkpoint:
    - If the param name is in _PRETRAIN_LOAD_SKIP_KEYS: keep YAML-initialized value
      (TJ-style σ reset).
    - If shapes match exactly: copy as-is.
    - If checkpoint is smaller on every dim: copy into the top-left corner of the
      current parameter; the remainder keeps its current (random) initialization.
    - Otherwise: skip with a warning.
    """
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)
    # skrl stores model state dicts nested under module names
    for module_name, module in agent.models.items():
        key = module_name  # e.g. "policy", "value"
        if key not in data:
            continue
        ckpt_sd = data[key]
        cur_sd = module.state_dict()
        updated_sd = {}
        for param_name, cur_tensor in cur_sd.items():
            if param_name in _PRETRAIN_LOAD_SKIP_KEYS:
                print(f"[partial load] {key}.{param_name}: SKIPPED — keeping YAML-initialized value (σ reset)")
                updated_sd[param_name] = cur_tensor
                continue
            if param_name not in ckpt_sd:
                print(f"[partial load] {key}.{param_name}: not in checkpoint, keeping random init")
                updated_sd[param_name] = cur_tensor
                continue
            ckpt_tensor = ckpt_sd[param_name]
            if ckpt_tensor.shape == cur_tensor.shape:
                updated_sd[param_name] = ckpt_tensor
            elif all(c <= t for c, t in zip(ckpt_tensor.shape, cur_tensor.shape)):
                new_tensor = cur_tensor.clone()
                slices = tuple(slice(0, s) for s in ckpt_tensor.shape)
                new_tensor[slices] = ckpt_tensor
                updated_sd[param_name] = new_tensor
                print(f"[partial load] {key}.{param_name}: {list(ckpt_tensor.shape)} → {list(cur_tensor.shape)} (partial copy)")
            else:
                print(f"[partial load] {key}.{param_name}: shape incompatible {list(ckpt_tensor.shape)} vs {list(cur_tensor.shape)}, skipping")
                updated_sd[param_name] = cur_tensor
        module.load_state_dict(updated_sd)
    print("[partial load] Done.")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Dataset / sequence overrides (for per-sequence training via benchmark scripts)
    if args_cli.dataset is not None:
        env_cfg.dataset = args_cli.dataset
    if args_cli.object_id is not None:
        env_cfg.object_id = args_cli.object_id
    if args_cli.trajectory_task is not None:
        env_cfg.trajectory_task = args_cli.trajectory_task
    if args_cli.trajectory_data_id is not None:
        env_cfg.trajectory_data_id = args_cli.trajectory_data_id

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    if args_cli.timesteps:
        agent_cfg["trainer"]["timesteps"] = args_cli.timesteps
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # Disable skrl trainer's auto "Info / " prefix on env extras['log'] — we'll log
    # those keys manually via the patched record_transition so they appear under
    # the env's own group prefixes ("Error /", "Episode_Reward /", "Curriculum /").
    agent_cfg["trainer"]["environment_info"] = "__disabled__"
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # For the grasp train task: replace policy with mass-as-an-action model (Section 3.2).
    _is_grasp_train = "Grasp" in (args_cli.task or "") and "Pretrain" not in (args_cli.task or "")
    _is_grasp_pretrain = "Grasp" in (args_cli.task or "") and "Pretrain" in (args_cli.task or "")
    if _is_grasp_train:
        _patch_mass_policy(runner.agent, agent_cfg["models"]["policy"], agent_cfg["agent"]["learning_rate"])
        _patch_entropy_flip(runner.agent, env, agent_cfg["agent"]["entropy_loss_scale"])

    # Manual env-info → Tensorboard (applies to BOTH train and pretrain — trainer's
    # auto-prefix "Info / " is disabled in main(), so this is the sole logging path
    # for env extras['log'] keys like "Error /", "Episode_Reward /", "Curriculum /").
    if _is_grasp_train or _is_grasp_pretrain:
        _patch_env_info_log(runner.agent)

    # For the grasp pretrain task: freeze log_std_parameter at YAML initial value (σ=0.22)
    # to mirror TJ's `frozen_sigma: True` in rl_games_ppo_cfg_pretrain.yaml. The policy
    # learns only mu during pretrain; exploration scale stays constant.
    if _is_grasp_pretrain:
        _policy = runner.agent.models["policy"]
        if hasattr(_policy, "log_std_parameter"):
            _policy.log_std_parameter.requires_grad_(False)
            _ls = _policy.log_std_parameter.detach()
            print(f"[INFO] Pretrain: log_std_parameter frozen, mean={_ls.mean().item():.4f} "
                  f"(σ_mean={_ls.exp().mean().item():.4f}, n={_ls.numel()}).")

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        if _is_grasp_train:
            # Pretrain → train transfer (and any train→train resume): always use partial
            # loading so we can deterministically (1) skip `log_std_parameter` to reset σ
            # to the YAML initial value, and (2) leave the optimizer untouched. Combined
            # with the fresh optimizer rebuilt by `_patch_mass_policy`, this mirrors TJ's
            # tools/reset_epoch.py — fresh σ + cleared optimizer state + lr back to YAML.
            _load_partial_checkpoint(runner.agent, resume_path)
        else:
            try:
                runner.agent.load(resume_path)
            except RuntimeError as e:
                err = str(e)
                if not any(kw in err for kw in ("size mismatch", "Missing key", "unexpected key")):
                    raise
                print(f"[INFO] Checkpoint mismatch ({err[:80]}...); attempting partial weight loading.")
                _load_partial_checkpoint(runner.agent, resume_path)
    
    #####################
    # Custom callback to log success rate from environment's extras during training
    #####################
    
    # get the agent for logging success rate during training
    agent = runner.agent
    
    # log success rate from environment's extras during training
    def log_success_rate(timestep, timesteps):
        actual_env = env.unwrapped
        
        try:
            success_rate = actual_env.extras.get("metrics", {}).get("success_rate", 0.0)
            if success_rate is not None:
                agent.track_data("Curriculum / success_rate", success_rate)
        except Exception as e:
            print(f"Error while logging success rate: {e}")
        
        agent._original_post_interaction(timestep=timestep, timesteps=timesteps)
    
    # monkey-patch the agent's post_interaction function to log success rate at each step
    agent._original_post_interaction = agent.post_interaction
    agent.post_interaction = log_success_rate
    
    #####################
    # End of custom callback for logging success rate
    #####################

    # run training
    runner.run()

    training_time = round(time.time() - start_time, 2)
    print(f"Training time: {training_time} seconds")

    # Save task_info.json to the processed output directory (only for grasp tasks with sequence info).
    _has_seq = (
        hasattr(env_cfg, "trajectory_task")
        and hasattr(env_cfg, "dataset")
        and env_cfg.trajectory_task
    )
    if _has_seq:
        _data_root = Path(
            env_cfg.hocap_data_dir if env_cfg.dataset == "hocap" else env_cfg.oakink_data_dir
        )
        _ckpt_dir = (
            _data_root / "ffw_sh5" / "right"
            / env_cfg.trajectory_task
            / str(env_cfg.trajectory_data_id)
        )
        _ckpt_dir.mkdir(parents=True, exist_ok=True)
        _task_info = {
            "task": args_cli.task,
            "dataset": env_cfg.dataset,
            "object_id": env_cfg.object_id,
            "trajectory_task": env_cfg.trajectory_task,
            "trajectory_data_id": env_cfg.trajectory_data_id,
            "num_envs": env_cfg.scene.num_envs,
            "timesteps": agent_cfg["trainer"]["timesteps"],
            "seed": agent_cfg["seed"],
            "checkpoint": str(Path(resume_path).resolve()) if resume_path else None,
            "log_dir": log_dir,
            "skrl_version": skrl.__version__,
            "training_time_s": training_time,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        _info_path = _ckpt_dir / "task_info.json"
        with open(_info_path, "w") as _f:
            json.dump(_task_info, _f, indent=2)
        print(f"[INFO] Task info saved → {_info_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
