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
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
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

import logging
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import skrl
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
_ENTROPY_FLIP_SCALE = -0.002  # entropy coef when is_reached_end=True (mirrors GR: -0.002 * sigma_weight)


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


def _patch_entropy_flip(agent, env_wrapper, base_entropy_scale: float) -> None:
    """Flip entropy coefficient sign when is_reached_end (mirrors GR a2c_continuous.py sigma_weight logic).

    GR formula: entropy_weight = entropy_coef * (1 - sigma_weight) - 0.002 * sigma_weight
      sigma_weight = float(is_reached_end)
      → is_reached_end=False: entropy_weight = +entropy_coef  (maximize entropy, std grows)
      → is_reached_end=True:  entropy_weight = -0.002         (minimize entropy, std shrinks)

    skrl uses the same sign convention: loss += -entropy_loss_scale * entropy
    so setting _entropy_loss_scale to negative has the identical effect.
    """
    _orig_update = agent._update

    def _update_with_entropy_flip(timestep: int, timesteps: int) -> None:
        actual_env = env_wrapper.unwrapped
        sigma_weight = float(actual_env.is_reached_end)
        agent._entropy_loss_scale = (
            base_entropy_scale * (1.0 - sigma_weight) + _ENTROPY_FLIP_SCALE * sigma_weight
        )
        _orig_update(timestep, timesteps)

    agent._update = _update_with_entropy_flip
    print(f"[entropy_flip] Patched _update: base={base_entropy_scale}, flip={_ENTROPY_FLIP_SCALE}")


def _load_partial_checkpoint(agent, path: str) -> None:
    """Load a checkpoint with mismatched input/output sizes (pretrain → train transfer).

    For each parameter in the checkpoint:
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

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
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
    if _is_grasp_train:
        _patch_mass_policy(runner.agent, agent_cfg["models"]["policy"], agent_cfg["agent"]["learning_rate"])
        _patch_entropy_flip(runner.agent, env, agent_cfg["agent"]["entropy_loss_scale"])

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
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
                agent.track_data("Metrics/success_rate", success_rate)
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

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
