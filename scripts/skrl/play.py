# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
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
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

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

import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

# ── MassDexMimic helpers ──────────────────────────────────────────────────────


def _patch_mass_policy_for_play(agent, policy_cfg: dict) -> None:
    """Replace the Runner-created policy with MassDexMimicPolicy for evaluation.

    Mirrors _patch_mass_policy() from train.py but skips optimizer / scheduler
    / record_transition patching — none of those are needed during play.

    Must be called AFTER runner = Runner(env, agent_cfg) and BEFORE checkpoint loading.
    """
    from robotis_sh5.tasks.direct.robotis_sh5_grasp.agents.mass_gaussian_model import MassDexMimicPolicy

    device = agent.device
    model_kwargs = {k: v for k, v in policy_cfg.items() if k not in ("class", "output")}

    new_policy = MassDexMimicPolicy(
        observation_space=agent.observation_space,
        action_space=agent.action_space,
        device=device,
        **model_kwargs,
    ).to(device)

    agent.models["policy"] = new_policy
    agent.policy = new_policy
    agent.checkpoint_modules["policy"] = new_policy

    print("[mass_policy] Patched policy with MassDexMimicPolicy for evaluation.")


def _load_partial_checkpoint(agent, path: str) -> None:
    """Load a checkpoint with mismatched sizes (e.g. pretrain → train transfer).

    For each parameter in the checkpoint:
    - Exact shape match → copy as-is.
    - Checkpoint smaller on every dim → copy into the top-left corner; remainder
      keeps current (random) initialization.
    - Otherwise → skip with a warning.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    for module_name, module in agent.checkpoint_modules.items():
        if module_name not in data:
            continue
        # Skip optimizers and schedulers — they are not nn.Modules and their
        # state dicts contain nested dicts, not tensors.
        if not isinstance(module, torch.nn.Module):
            continue
        ckpt_sd = data[module_name]
        cur_sd = module.state_dict()
        updated_sd = {}
        for param_name, cur_tensor in cur_sd.items():
            if param_name not in ckpt_sd:
                print(f"[partial load] {module_name}.{param_name}: not in checkpoint, keeping init")
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
                print(f"[partial load] {module_name}.{param_name}: "
                      f"{list(ckpt_tensor.shape)} → {list(cur_tensor.shape)} (partial copy)")
            else:
                print(f"[partial load] {module_name}.{param_name}: "
                      f"incompatible {list(ckpt_tensor.shape)} vs {list(cur_tensor.shape)}, skipping")
                updated_sd[param_name] = cur_tensor
        module.load_state_dict(updated_sd)
    print("[partial load] Done.")

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
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robotis_sh5.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    # For the grasp train task: replace the Runner-created GaussianMixin with
    # MassDexMimicPolicy so checkpoint shapes match (paper Section 3.2).
    _is_grasp_train = (
        args_cli.task is not None
        and "Grasp" in args_cli.task
        and "Pretrain" not in args_cli.task
    )
    if _is_grasp_train:
        _patch_mass_policy_for_play(runner.agent, experiment_cfg["models"]["policy"])

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    try:
        runner.agent.load(resume_path)
    except (RuntimeError, KeyError, ValueError) as e:
        err = str(e)
        if not any(kw in err for kw in (
            "size mismatch", "Missing key", "unexpected key", "KeyError",
            "parameter groups",   # optimizer group count mismatch (train 2-group vs play 1-group)
        )):
            raise
        print(f"[INFO] Checkpoint load issue ({err[:120]}); loading model weights only.")
        _load_partial_checkpoint(runner.agent, resume_path)

    # set agent to evaluation mode.
    # skrl 2.1.0 replaced `set_running_mode("eval")` with `enable_training_mode(False)`; keep both
    # so this script works against either version.
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)

    # reset environment
    obs, _ = env.reset()
    states = env.state()   # skrl 2.0.0: states are obtained separately from observations
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping — skrl 2.0.0: act(observations, states, ...) → (actions, outputs)
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, terminated, truncated, _ = env.step(actions)
            states = env.state()
            # Propagate episode terminations to MassDexMimicPolicy so mass is
            # re-sampled only at episode boundaries (not every step).
            _policy = getattr(runner.agent, "policy", None)
            if _policy is not None and hasattr(_policy, "update_mass_terminated"):
                _policy.update_mass_terminated(terminated | truncated)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
