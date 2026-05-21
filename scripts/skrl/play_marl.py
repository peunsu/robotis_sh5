# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a MAPPO MARL checkpoint from skrl (arm + hand decomposition).

Mirrors `scripts/skrl/play.py` but for MARL tasks only. Applies the same
sequential forward (hand → arm, hand action injected into arm obs slot at
hand obs slot [65:72]) so distribution at eval matches training time.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a MAPPO MARL checkpoint with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the MARL task (must contain 'Marl').")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to MAPPO checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--object_id", type=str, default=None)
parser.add_argument("--trajectory_task", type=str, default=None)
parser.add_argument("--trajectory_data_id", type=int, default=None)
parser.add_argument(
    "--stochastic", action="store_true", default=False,
    help="Sample actions stochastically (training-style). Default uses mean actions.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

if args_cli.task is None or "Marl" not in args_cli.task:
    raise ValueError(f"--task must be a MARL task (contain 'Marl'); got {args_cli.task!r}")

sys.argv = [sys.argv[0]] + hydra_args
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

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robotis_sh5.tasks  # noqa: F401

agent_cfg_entry_point = "skrl_mappo_cfg_entry_point"

# Sequential conditioning: hand → arm. Hand decides first, action injected
# into arm obs slot [62:82].
ARM_HAND_SLOT_START = 62
ARM_HAND_SLOT_LEN = 20


def _patch_sequential_act(agent) -> None:
    """At eval: same hand → arm sequential forward as training (no record_transition needed)."""
    slot_lo = ARM_HAND_SLOT_START
    slot_hi = ARM_HAND_SLOT_START + ARM_HAND_SLOT_LEN

    def sequential_act(states, timestep, timesteps):
        with torch.no_grad():
            hand_states_pp = agent._state_preprocessor["hand"](states["hand"])
            hand_action, hand_log_prob, hand_outputs = agent.policies["hand"].act(
                {"states": hand_states_pp}, role="policy"
            )
            arm_states_injected = states["arm"].clone()
            arm_states_injected[:, slot_lo:slot_hi] = hand_action.detach()
            arm_states_pp = agent._state_preprocessor["arm"](arm_states_injected)
            arm_action, arm_log_prob, arm_outputs = agent.policies["arm"].act(
                {"states": arm_states_pp}, role="policy"
            )

            actions = {"arm": arm_action, "hand": hand_action}
            log_prob = {"arm": arm_log_prob, "hand": hand_log_prob}
            outputs = {"arm": arm_outputs, "hand": hand_outputs}
        return actions, log_prob, outputs

    agent.act = sequential_act
    print(f"[sequential-mappo-play] hand → arm, inject at arm obs [{slot_lo}:{slot_hi}]")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with MAPPO MARL agent."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.dataset is not None:
        env_cfg.dataset = args_cli.dataset
    if args_cli.object_id is not None:
        env_cfg.object_id = args_cli.object_id
    if args_cli.trajectory_task is not None:
        env_cfg.trajectory_task = args_cli.trajectory_task
    if args_cli.trajectory_data_id is not None:
        env_cfg.trajectory_data_id = args_cli.trajectory_data_id

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=r".*_MAPPO_torch", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if not isinstance(env.unwrapped, DirectMARLEnv):
        raise TypeError(f"Task {args_cli.task} is not a DirectMARLEnv")

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    _patch_sequential_act(runner.agent)

    print(f"[INFO] Loading MAPPO checkpoint from: {resume_path}")
    # See rollout_marl.py: skip optimizer load to avoid param-group size mismatch
    # caused by HAPPO training (actor-only / critic-only optimizer structure).
    for uid in runner.agent.possible_agents:
        runner.agent.checkpoint_modules[uid].pop("optimizer", None)
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # Deterministic by default — use sampled actions only if --stochastic.
            if args_cli.stochastic:
                actions = outputs[0]
            else:
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            obs, _, terminated, truncated, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
