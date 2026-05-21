# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train MAPPO (arm + hand decomposition) agents with skrl.

Mirrors `scripts/skrl/train.py` but stripped to MAPPO + grasp-MARL only:
  - No mass-as-action patch
  - No GR entropy-flip patch (KL-adaptive LR is sufficient)
  - Adds partial pretrain → train checkpoint loader (policy/value weights only;
    optimizer/preprocessors fresh — see `_load_marl_partial_checkpoint`).
  - Adds `_setup_happo_optimizers`: single shared critic V(s_global) with a
    dedicated `critic_optimizer`; per-actor `Adam([policy.params])` only.
  - Adds `_patch_happo_update`: HAPPO (Kuba et al. 2022, Algorithm 4):
      • single GAE compute on team reward + shared critic,
      • sequential actor updates with RANDOM permutation per PPO update,
      • recursive advantage M_{1:m+1} = (π^m_new / π^m_old) · M_{1:m},
      • separate critic optimization step at the end.
  - Adds `_patch_sequential_act`: hand → arm sequential FORWARD conditioning.
    Hand decides first, its action injected into arm obs slot [62:82].

Architecture: arm = wrist-pose follower + object context (82D obs, 7D action),
hand = full grasping policy (276D obs, 20D action). Single shared team reward,
single shared critic, sequential conditioning hand → arm (both forward and update).
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train MAPPO MARL agents (arm + hand) with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the MARL task (must contain 'Marl').")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to MAPPO checkpoint to resume training.")
parser.add_argument("--timesteps", type=int, default=None, help="Overrides agent YAML trainer.timesteps.")
parser.add_argument("--dataset", type=str, default=None, help="Dataset name (e.g. 'oakink'); overrides env_cfg.dataset.")
parser.add_argument("--object_id", type=str, default=None, help="Object ID; overrides env_cfg.object_id.")
parser.add_argument("--trajectory_task", type=str, default=None, help="Trajectory task directory name.")
parser.add_argument("--trajectory_data_id", type=int, default=None, help="Trajectory data sub-index.")
parser.add_argument(
    "--freeze-arm-from", type=str, default=None,
    help="Path to a MAPPO ckpt; load arm policy/value from it and freeze. Hand trains only.",
)
parser.add_argument(
    "--freeze-hand-from", type=str, default=None,
    help="Path to a MAPPO ckpt; load hand policy/value from it and freeze. Arm trains only.",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# Validate task is a MARL task
if args_cli.task is None or "Marl" not in args_cli.task:
    raise ValueError(f"--task must be a MARL task (contain 'Marl'); got {args_cli.task!r}")

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
import math
from pathlib import Path

import gymnasium as gym
import skrl
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from skrl.resources.schedulers.torch import KLAdaptiveLR

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

import robotis_sh5.tasks  # noqa: F401

# MAPPO config is found via this hydra entry point
agent_cfg_entry_point = "skrl_mappo_cfg_entry_point"

# Arm obs layout (see RobotisSh5GraspMarlEnv._get_observations):
#   jp_arm(7) + jv_arm(7) + wrist_pos_env(3) + wrist_quat_w(4)
#   + wrist_linvel(3) + wrist_angvel(3) + delta_wrist_pos(3) + delta_wrist_rot(3)
#   + prev_arm_action(7)                                                         = 40
#   + obj_pos_env(3) + obj_quat(4) + obj_linvel(3) + obj_angvel(3)
#   + delta_wrist_obj(3) + delta_obj_pos(3) + delta_obj_rot(3)                   = 22
#   = 62, then current_hand_action slot of length 20 at [62:82]
ARM_HAND_SLOT_START = 62
ARM_HAND_SLOT_LEN = 20


def _patch_sequential_act(agent) -> None:
    """Sequential conditioning: hand decides first, action injected into arm obs.

    Order is HAND → ARM (reverse of paper convention). Rationale: hand has its own
    sufficient context (full grasping obs); arm needs to anticipate hand's finger
    motion to stabilize the palm during grasp moments.

    Also patches `record_transition` so the stored `states["arm"]` matches the
    injected version (PPO ratio consistency between rollout and training).
    Shared state is NOT affected (env._get_states() is independent of obs).
    """
    slot_lo = ARM_HAND_SLOT_START
    slot_hi = ARM_HAND_SLOT_START + ARM_HAND_SLOT_LEN

    agent._marl_injected_arm_states = None
    _orig_record_transition = agent.record_transition

    def sequential_act(states, timestep, timesteps):
        with torch.autocast(device_type=agent._device_type, enabled=agent._mixed_precision):
            # 1. hand forward (full grasping context, no injection needed)
            hand_states_pp = agent._state_preprocessor["hand"](states["hand"])
            hand_action, hand_log_prob, hand_outputs = agent.policies["hand"].act(
                {"states": hand_states_pp}, role="policy"
            )
            # 2. inject hand action into arm obs slot, then forward arm
            arm_states_injected = states["arm"].clone()
            arm_states_injected[:, slot_lo:slot_hi] = hand_action.detach()
            arm_states_pp = agent._state_preprocessor["arm"](arm_states_injected)
            arm_action, arm_log_prob, arm_outputs = agent.policies["arm"].act(
                {"states": arm_states_pp}, role="policy"
            )

            actions = {"arm": arm_action, "hand": hand_action}
            log_prob = {"arm": arm_log_prob, "hand": hand_log_prob}
            outputs = {"arm": arm_outputs, "hand": hand_outputs}

            agent._current_log_prob = log_prob
            agent._marl_injected_arm_states = arm_states_injected
        return actions, log_prob, outputs

    def record_transition_with_injected(
        states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps,
    ):
        # Replace states["arm"] with the injected version (arm policy actually saw
        # this during act). Without this, PPO ratio = exp(log_pi_new(a|s_raw) -
        # log_pi_old(a|s_injected)) is biased.
        # NOTE: shared_states (centralized critic input) is computed by env._get_states()
        # which is INDEPENDENT of obs dict → no modification needed there.
        if agent._marl_injected_arm_states is not None:
            states = dict(states)
            states["arm"] = agent._marl_injected_arm_states

        # Manual environment-info logging — bypasses skrl trainer's hard-coded
        # "Info / " prefix so that keys grouped as "Error / X", "Reward / X",
        # "Mass / X", "Curriculum / X" become separate Tensorboard tabs.
        # Trainer's environment_info is set to None (see main()), disabling its
        # auto-log loop, so this is the sole logging path for env extras.
        log_dict = infos.get("log")
        if log_dict:
            for k, v in log_dict.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    agent.track_data(k, v.item())

        # Mass-in-the-loop: append per-step (action, log_prob_old) snapshot to buffer.
        # Snapshots are captured by env._pre_physics_step BEFORE _reset_idx resamples,
        # so they correctly represent the mass that was USED during this step.
        # Buffers and env reference are set up by _setup_mass_in_loop (if enabled).
        if hasattr(agent, "_mass_buf_actions") and agent._mass_env_ref is not None:
            env_u = agent._mass_env_ref
            a_step = env_u._mass_action_step
            l_step = env_u._mass_log_prob_old_step
            if a_step is not None and l_step is not None:
                agent._mass_buf_actions.append(a_step.clone())
                agent._mass_buf_lp_old.append(l_step.clone())

        _orig_record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps,
        )

    agent.act = sequential_act
    agent.record_transition = record_transition_with_injected
    print(f"[sequential-mappo] Patched agent.act: hand → arm, injecting hand action at arm obs [{slot_lo}:{slot_hi}]")


# Arm-only initial σ override. The shared yaml uses initial_log_std=-1.5141 (σ=0.22);
# the arm has a low-DoF, smooth tracking objective and benefits from a tighter
# Gaussian. Value is read from `project_overrides.arm_initial_log_std` in the
# MAPPO yaml. max_log_std is left at the yaml default so KL-adaptive LR can
# still breathe if needed.
def _override_arm_log_std(agent, agent_cfg: dict) -> None:
    overrides = agent_cfg.get("project_overrides", {}) or {}
    value = overrides.get("arm_initial_log_std", None)
    if value is None:
        return
    arm_policy = agent.policies.get("arm")
    if arm_policy is None or not hasattr(arm_policy, "log_std_parameter"):
        return
    value = float(value)
    with torch.no_grad():
        arm_policy.log_std_parameter.data.fill_(value)
    print(f"[INFO] arm policy log_std_parameter set to {value:.4f} (σ≈{math.exp(value):.4f}).")


def _setup_happo_optimizers(agent, agent_cfg: dict) -> None:
    """HAPPO optimizer layout (Kuba et al. 2022, Algorithm 4):
      • SINGLE shared critic V_φ(s_global) — one network instance.
      • critic_optimizer  = Adam([V_φ.params])               ← dedicated.
      • actor_optim[uid]  = Adam([π_θ_i.params])             ← policy only.

    Differs from skrl MAPPO default (per-agent value + per-agent Adam([policy+value]))
    and from the previous `_share_value_critic` patch (critic bundled into arm's
    Adam). HAPPO requires a critic-specific optimizer because the critic update
    is a separate optimization step at the end of every iteration.

    KLAdaptiveLR scheduler is attached to each actor optimizer (KL is per-actor).
    The critic uses constant base_lr (standard HAPPO behavior).

    The shared critic preprocessor (`_value_preprocessor`, `_shared_state_preprocessor`)
    are also unified across agents — they were per-agent instances by skrl default.
    """
    base_lr = float(agent_cfg["agent"]["learning_rate"])

    # ── 1. Share critic instance across agents ───────────────────────────────
    # We anchor the shared critic under "arm"'s checkpoint slot (because skrl's
    # save/load only iterates `possible_agents` — custom keys outside this list
    # would not be serialized). "hand" loses its value-related checkpoint entries.
    primary = "arm"
    shared_value = agent.values[primary]
    shared_value_pp = agent._value_preprocessor[primary]
    shared_state_pp = agent._shared_state_preprocessor[primary]
    for uid in agent.possible_agents:
        agent.values[uid] = shared_value
        agent._value_preprocessor[uid] = shared_value_pp
        agent._shared_state_preprocessor[uid] = shared_state_pp
        if "value" in agent.models.get(uid, {}):
            agent.models[uid]["value"] = shared_value
        if uid != primary:
            for key in ("value", "value_preprocessor", "shared_state_preprocessor"):
                agent.checkpoint_modules[uid].pop(key, None)

    # ── 2. Dedicated critic optimizer ────────────────────────────────────────
    # Stored under "arm" with key "critic_optimizer" (not "optimizer" — that
    # name is owned by arm's ACTOR optimizer below). skrl save/load will pick
    # this up via the "arm" agent slot.
    critic_optimizer = torch.optim.Adam(shared_value.parameters(), lr=base_lr)
    agent._critic_optimizer = critic_optimizer
    agent.checkpoint_modules[primary]["critic_optimizer"] = critic_optimizer

    # ── 3. Per-actor optimizers (policy only) ────────────────────────────────
    for uid in agent.possible_agents:
        actor_optim = torch.optim.Adam(agent.policies[uid].parameters(), lr=base_lr)
        agent.optimizers[uid] = actor_optim
        agent.checkpoint_modules[uid]["optimizer"] = actor_optim

    # ── 4. Rebuild LR schedulers (per actor; critic LR is constant) ──────────
    for uid in agent.possible_agents:
        if agent._learning_rate_scheduler.get(uid) is not None:
            agent.schedulers[uid] = agent._learning_rate_scheduler[uid](
                agent.optimizers[uid],
                **agent._learning_rate_scheduler_kwargs[uid],
            )

    n_policy = {uid: sum(p.numel() for p in agent.policies[uid].parameters())
                for uid in agent.possible_agents}
    n_value = sum(p.numel() for p in shared_value.parameters())
    print(f"[HAPPO] Single shared critic V(s_global) + critic-specific optimizer")
    for uid in agent.possible_agents:
        print(f"  actor[{uid}]:  {n_policy[uid]:>9,d} params  (own Adam optimizer)")
    print(f"  shared V:     {n_value:>9,d} params  (own Adam optimizer)")


def _patch_happo_update(agent, agent_cfg: dict) -> None:
    """Override agent._update with the HAPPO algorithm (Algorithm 4, Kuba et al. 2022).

    Differences from skrl MAPPO's default `_update`:
      1. GAE computed ONCE on team reward + shared critic (vs per-agent in skrl).
      2. Actors updated SEQUENTIALLY with RANDOM PERMUTATION (HAPPO Algorithm 4),
         applying recursive advantage M_{1:m+1} = (π^m_new / π^m_old) · M_{1:m}.
         The permutation is resampled EACH PPO update.
      3. Critic update is a SEPARATE optimization step at the end using
         `_critic_optimizer` (Adam on shared_value.parameters only).

    Note on forward vs update order:
      • Forward order is FIXED (hand → arm) — defined by `_patch_sequential_act`,
        which injects hand action into arm obs slot. This is a feature of the
        observation design, NOT the HAPPO update algorithm.
      • Update order is RANDOM (this function) — matches the theoretical
        guarantee of Algorithm 4 (monotonic improvement under random permutation).

    Memory layout & convention:
      • Both agents' memories carry IDENTICAL rewards / shared_states / values
        (team reward + single shared critic V on s_global). We compute GAE once
        from arm.memory (primary).
      • Each agent's memory holds its OWN (states, actions, log_prob).
      • At the end of the HAPPO update we restore each memory's "advantages"
        slot to the original normalized Â — so mass-in-loop (which reads from
        `memories["arm"].get_tensor_by_name("advantages")`) sees the pristine
        advantage estimate.
    """
    primary = "arm"   # memory used for GAE/critic update; rewards/state are agent-invariant
    all_agents = list(agent.possible_agents)   # natural order, will be shuffled per iteration
    cfg = agent_cfg["agent"]

    ratio_clip = float(cfg["ratio_clip"])
    value_clip = float(cfg["value_clip"])
    clip_predicted_values = bool(cfg["clip_predicted_values"])
    entropy_loss_scale = float(cfg["entropy_loss_scale"])
    value_loss_scale = float(cfg["value_loss_scale"])
    kl_threshold = float(cfg["kl_threshold"])
    grad_norm_clip = float(cfg["grad_norm_clip"])
    learning_epochs = int(cfg["learning_epochs"])
    mini_batches = int(cfg["mini_batches"])
    discount_factor = float(cfg["discount_factor"])
    lam = float(cfg["lambda"])

    tensors_names_actor = ["states", "actions", "log_prob", "advantages"]
    tensors_names_critic = ["shared_states", "values", "returns"]

    def _compute_gae(rewards, dones, values, last_values):
        """Standard GAE (mirrors skrl's nested compute_gae). `values`/`last_values`
        are in RAW (denormalized) scale. Returns are raw; advantages are mean/std
        normalized (skrl convention).
        """
        T = rewards.shape[0]
        not_dones = dones.logical_not()
        advantages = torch.zeros_like(rewards)
        adv = 0.0
        for i in reversed(range(T)):
            next_v = values[i + 1] if i < T - 1 else last_values
            adv = (
                rewards[i]
                - values[i]
                + discount_factor * not_dones[i] * (next_v + lam * adv)
            )
            advantages[i] = adv
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _happo_update(timestep: int, timesteps: int) -> None:
        # ─── Phase 0: Single shared GAE on team reward + shared critic ────────
        with torch.no_grad():
            shared_value = agent.values[primary]
            shared_value.train(False)
            last_values, _, _ = shared_value.act(
                {"states": agent._shared_state_preprocessor[primary](
                    agent._current_shared_next_states.float())},
                role="value",
            )
            shared_value.train(True)
            last_values = agent._value_preprocessor[primary](last_values, inverse=True)

        mem_primary = agent.memories[primary]
        rewards = mem_primary.get_tensor_by_name("rewards")
        dones = mem_primary.get_tensor_by_name("terminated") | mem_primary.get_tensor_by_name("truncated")
        values_raw = mem_primary.get_tensor_by_name("values")    # stored denormalized in record_transition
        returns_raw, advantages_hat = _compute_gae(rewards, dones, values_raw, last_values)

        # Store normalized values/returns + Â in BOTH agents' memories (skrl mini-batch
        # sampling requires these tensors present per-memory). The value_preprocessor
        # is SHARED across agents (HAPPO setup) — train it ONCE on primary, then
        # use the already-fitted preprocessor (train=False) for other agents.
        values_norm = agent._value_preprocessor[primary](values_raw, train=True)
        returns_norm = agent._value_preprocessor[primary](returns_raw, train=True)
        for uid in agent.possible_agents:
            mem = agent.memories[uid]
            mem.set_tensor_by_name("values", values_norm)
            mem.set_tensor_by_name("returns", returns_norm)
            mem.set_tensor_by_name("advantages", advantages_hat)

        # ─── Phase 1: Sequential actor updates with recursive M ────────────────
        # Random permutation of agents per HAPPO Algorithm 4 — resampled every PPO
        # update. This is the monotonic-improvement guarantee in the original paper.
        actor_order = list(all_agents)
        random.shuffle(actor_order)
        agent.track_data("Policy / actor_order_first", float(
            all_agents.index(actor_order[0])
        ))

        M = advantages_hat.detach().clone()   # M_{1:1} = Â

        last_kl_per_actor = {}
        cum_policy_loss = {uid: 0.0 for uid in actor_order}
        cum_entropy_loss = {uid: 0.0 for uid in actor_order}

        for m, uid in enumerate(actor_order):
            policy = agent.policies[uid]
            memory = agent.memories[uid]
            actor_optim = agent.optimizers[uid]

            # Plug M into memory.advantages so sample_all picks it up.
            memory.set_tensor_by_name("advantages", M)

            sampled_batches = memory.sample_all(names=tensors_names_actor,
                                                mini_batches=mini_batches)
            sched = agent.schedulers.get(uid)
            kl_last_epoch_mean = torch.tensor(0.0, device=agent.device)
            stop_actor = False

            for epoch in range(learning_epochs):
                kl_this_epoch: list[torch.Tensor] = []   # reset per epoch (skrl PPO/MAPPO convention)

                for (s_states, s_actions, s_log_prob, s_M) in sampled_batches:
                    s_states_pp = agent._state_preprocessor[uid](s_states, train=(epoch == 0))

                    _, next_log_prob, _ = policy.act(
                        {"states": s_states_pp, "taken_actions": s_actions},
                        role="policy",
                    )

                    # KL early stop check (skrl convention)
                    with torch.no_grad():
                        log_ratio_kl = next_log_prob - s_log_prob
                        kl_mb = ((torch.exp(log_ratio_kl) - 1) - log_ratio_kl).mean()
                    kl_this_epoch.append(kl_mb)
                    if kl_threshold > 0 and kl_mb > kl_threshold:
                        stop_actor = True
                        break

                    # PPO-clip surrogate on M
                    ratio = torch.exp(next_log_prob - s_log_prob)
                    surrogate = s_M * ratio
                    surrogate_clipped = s_M * torch.clip(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # Entropy regularization
                    if entropy_loss_scale:
                        entropy_loss = -entropy_loss_scale * policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = torch.tensor(0.0, device=agent.device)

                    actor_optim.zero_grad()
                    (policy_loss + entropy_loss).backward()
                    if grad_norm_clip > 0:
                        nn.utils.clip_grad_norm_(policy.parameters(), grad_norm_clip)
                    actor_optim.step()

                    cum_policy_loss[uid] += float(policy_loss.item())
                    cum_entropy_loss[uid] += float(entropy_loss.item())

                # KL-adaptive LR step PER EPOCH (skrl PPO/MAPPO convention — runs
                # learning_epochs times per PPO update, lets LR adapt within an update).
                if kl_this_epoch:
                    kl_last_epoch_mean = torch.stack(kl_this_epoch).mean()
                    if sched is not None:
                        if isinstance(sched, KLAdaptiveLR):
                            sched.step(kl_last_epoch_mean.item())
                        else:
                            sched.step()

                if stop_actor:
                    break

            last_kl_per_actor[uid] = kl_last_epoch_mean.item()

            # Recursive advantage update for the NEXT actor (skip after last):
            #   M_{1:m+1}(s,a) = (π^m_new(a^m|o^m) / π^m_old(a^m|o^m)) · M_{1:m}(s,a)
            # Ratio is computed over ALL stored transitions for THIS actor under the
            # JUST-UPDATED policy. Memory returns 3D tensors (T, N, dim); skrl's
            # generated `compute()` only flattens "states" via unflatten_tensorized_space
            # but `taken_actions` is passed raw to Normal.log_prob → would broadcast-fail.
            # Flatten everything to 2D (T*N, dim) before policy.act, then reshape M back.
            if m < len(actor_order) - 1:
                with torch.no_grad():
                    states_3d = memory.get_tensor_by_name("states")            # (T, N, obs_dim)
                    actions_3d = memory.get_tensor_by_name("actions")          # (T, N, act_dim)
                    log_prob_old_3d = memory.get_tensor_by_name("log_prob")    # (T, N, 1)
                    T_, N_ = states_3d.shape[:2]

                    states_flat = states_3d.reshape(T_ * N_, -1)
                    actions_flat = actions_3d.reshape(T_ * N_, -1)
                    log_prob_old_flat = log_prob_old_3d.reshape(T_ * N_, -1)

                    states_pp_flat = agent._state_preprocessor[uid](states_flat, train=False)
                    _, log_prob_new_flat, _ = policy.act(
                        {"states": states_pp_flat, "taken_actions": actions_flat},
                        role="policy",
                    )
                    ratio_flat = torch.exp(log_prob_new_flat - log_prob_old_flat)  # (T*N, 1)
                    ratio_3d = ratio_flat.reshape(T_, N_, 1)
                    M = ratio_3d * M

        # ─── Phase 2: Critic update (separate optimizer) ──────────────────────
        critic_batches = mem_primary.sample_all(names=tensors_names_critic,
                                                mini_batches=mini_batches)
        cum_value_loss = 0.0
        n_critic_updates = 0
        for epoch in range(learning_epochs):
            for (s_shared_states, s_values, s_returns) in critic_batches:
                s_shared_states_pp = agent._shared_state_preprocessor[primary](
                    s_shared_states, train=(epoch == 0))

                predicted_values, _, _ = shared_value.act(
                    {"states": s_shared_states_pp}, role="value")

                if clip_predicted_values:
                    predicted_values = s_values + torch.clip(
                        predicted_values - s_values, min=-value_clip, max=value_clip
                    )
                value_loss = value_loss_scale * F.mse_loss(s_returns, predicted_values)

                agent._critic_optimizer.zero_grad()
                value_loss.backward()
                if grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(shared_value.parameters(), grad_norm_clip)
                agent._critic_optimizer.step()

                cum_value_loss += float(value_loss.item())
                n_critic_updates += 1

        # ─── Phase 3: Restore Â in advantages slot (for mass-in-loop) ─────────
        for uid in agent.possible_agents:
            agent.memories[uid].set_tensor_by_name("advantages", advantages_hat)

        # ─── Phase 4: Logging ────────────────────────────────────────────────
        n_updates_per_actor = max(1, learning_epochs * mini_batches)
        for uid in actor_order:
            policy_uid = agent.policies[uid]
            agent.track_data(f"Loss / Policy loss ({uid})",
                             cum_policy_loss[uid] / n_updates_per_actor)
            if entropy_loss_scale:
                agent.track_data(f"Loss / Entropy loss ({uid})",
                                 cum_entropy_loss[uid] / n_updates_per_actor)
            agent.track_data(f"Policy / KL divergence ({uid})", last_kl_per_actor[uid])
            sched = agent.schedulers.get(uid)
            if sched is not None:
                agent.track_data(f"Policy / Learning rate ({uid})",
                                 sched._last_lr[0] if hasattr(sched, "_last_lr") else float(cfg["learning_rate"]))
            # Policy stddev (mirrors skrl PPO's "Policy / Standard deviation" log).
            # Read directly from log_std_parameter — robust against the last-call
            # distribution being from the recursive-M block (3D input).
            if hasattr(policy_uid, "log_std_parameter"):
                log_std = policy_uid.log_std_parameter.detach()
                agent.track_data(f"Policy / Standard deviation ({uid})",
                                 log_std.exp().mean().item())
        if n_critic_updates:
            agent.track_data("Loss / Value loss", cum_value_loss / n_critic_updates)

    agent._update = _happo_update
    print(f"[HAPPO] _update overridden. Actor order: RANDOM permutation of {all_agents} "
          f"resampled each PPO update. Recursive M, separate critic step.")


def _setup_mass_in_loop(agent, env_cfg, agent_cfg: dict, env_wrapper):
    """Create MassDistribution, hook it into env, set up optimizers + PPO-style update.

    **PPO surrogate** (mirrors single-agent MassDexMimicPolicy mechanism — but with
    mass kept OUT of any agent's action space):

      1. At rollout sampling time (env._reset_idx), mass action is sampled and its
         log_prob_old is computed using params AT SAMPLE TIME. Both are cached
         per-env in MassDistribution.
      2. Each env step, env._pre_physics_step snapshots cached (action, log_prob_old)
         into env._mass_action_step / env._mass_log_prob_old_step.
      3. record_transition (patched) reads these snapshots and appends to per-rollout
         buffers (agent._mass_buf_actions, agent._mass_buf_lp_old).
      4. After the original PPO update (which trains policies + shared critic and
         normalizes advantages in memory), a mass mini-PPO loop runs:
           - sample mini-batches of (action, log_prob_old, advantage)
           - ratio = exp(log_prob_live(action; current μ/σ) - log_prob_old)
           - surrogate = min(ratio*adv, clipped_ratio*adv).mean()
           - loss = -surrogate; backprop; step (mu_optim and ls_optim separately).

    Hyperparams (matched to single-agent MassDexMimic convention):
      μ_mass     : Adam(lr=base_lr × env_cfg.mass_lr_scale)   # default 33.333×
      log_σ_mass : Adam(lr=base_lr)                            # no boost
      ratio_clip / learning_epochs / mini_batches : same as PPO cfg

    Returns the created MassDistribution (or None if disabled in cfg).
    """
    from robotis_sh5.tasks.direct.robotis_sh5_grasp.agents.mass_distribution import MassDistribution

    if not getattr(env_cfg, "enable_mass_in_loop", False):
        print("[mass-in-loop] disabled by cfg → skipping")
        return None

    device = agent.device
    num_envs = env_wrapper.unwrapped.num_envs
    mass_dist = MassDistribution(
        num_envs=num_envs,
        mass_min=env_cfg.object_mass_min,
        mass_max=env_cfg.object_mass_max,
        mu_init=env_cfg.mass_mu_init,
        log_std_init=env_cfg.mass_log_std_init,
        device=device,
    ).to(device)

    # Hook into env so _reset_idx can sample + apply mass per-episode.
    env_wrapper.unwrapped._mass_dist = mass_dist

    # Per-rollout snapshot buffers (lists of (B,) tensors, one per step).
    # Cleared after each PPO+mass update.
    agent._mass_buf_actions: list[torch.Tensor] = []
    agent._mass_buf_lp_old: list[torch.Tensor] = []
    # Env reference for the record_transition patch to read step-time snapshots.
    agent._mass_env_ref = env_wrapper.unwrapped

    # Checkpoint integration — save/load mu_mass, log_std_mass via skrl checkpointing.
    agent.checkpoint_modules.setdefault("mass", {})["dist"] = mass_dist

    base_lr = float(agent_cfg["agent"]["learning_rate"])
    lr_scale = float(env_cfg.mass_lr_scale)
    mu_optim = torch.optim.Adam([{"params": [mass_dist.mu_mass], "lr": base_lr * lr_scale}], eps=1e-8)
    ls_optim = torch.optim.Adam([{"params": [mass_dist.log_std_mass], "lr": base_lr}], eps=1e-8)
    agent.checkpoint_modules["mass"]["mu_optim"] = mu_optim
    agent.checkpoint_modules["mass"]["ls_optim"] = ls_optim

    # PPO hyperparameters for the mass mini-update (shared with main PPO).
    ratio_clip = float(agent_cfg["agent"]["ratio_clip"])
    n_epochs = int(agent_cfg["agent"]["learning_epochs"])
    n_minibatches = int(agent_cfg["agent"]["mini_batches"])

    def _mass_ppo_step(advantages_TN1: torch.Tensor) -> None:
        """Run learning_epochs × mini_batches PPO updates on (mu_mass, log_std_mass).

        advantages_TN1: (T, N, 1) — per-step advantages from memory (already normalized
        by the main PPO update — read from memory's "advantages" tensor).
        """
        if not agent._mass_buf_actions:
            return  # nothing collected (shouldn't happen post-rollout, but be safe)

        # Stack per-step snapshots → (T, N)
        actions_TN = torch.stack(agent._mass_buf_actions, dim=0)    # (T, N)
        lp_old_TN = torch.stack(agent._mass_buf_lp_old, dim=0)      # (T, N)

        # Align shapes with advantages: (T, N, 1)
        actions = actions_TN.unsqueeze(-1)
        lp_old = lp_old_TN.unsqueeze(-1)
        advs = advantages_TN1.detach()

        # Flatten across (T, N)
        B = actions.shape[0] * actions.shape[1]
        actions_flat = actions.reshape(B, -1)        # (B, 1)
        lp_old_flat = lp_old.reshape(B, -1)          # (B, 1)
        advs_flat = advs.reshape(B, -1)              # (B, 1)

        batch_size = max(1, B // n_minibatches)

        cum_loss = 0.0
        cum_kl = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            perm = torch.randperm(B, device=actions_flat.device)
            for mb in range(n_minibatches):
                idx = perm[mb * batch_size : (mb + 1) * batch_size]
                if idx.numel() == 0:
                    continue
                a_mb = actions_flat[idx]
                lp_old_mb = lp_old_flat[idx]
                adv_mb = advs_flat[idx]

                lp_new_mb = mass_dist.log_prob_live(a_mb)        # differentiable
                ratio = (lp_new_mb - lp_old_mb).exp()
                surrogate = adv_mb * ratio
                clipped = adv_mb * ratio.clamp(1.0 - ratio_clip, 1.0 + ratio_clip)
                loss = -torch.min(surrogate, clipped).mean()

                mu_optim.zero_grad()
                ls_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [mass_dist.mu_mass, mass_dist.log_std_mass], max_norm=1.0
                )
                mu_optim.step()
                ls_optim.step()

                with torch.no_grad():
                    cum_loss += float(loss.item())
                    cum_kl += float(((lp_old_mb - lp_new_mb).mean()).item())
                    n_updates += 1

        # Tensorboard logging (Mass / *)
        if n_updates > 0:
            agent.track_data("Mass / loss", cum_loss / n_updates)
            agent.track_data("Mass / approx_kl", cum_kl / n_updates)
        agent.track_data("Mass / mu_action", mass_dist.mu_action)
        agent.track_data("Mass / std_action", mass_dist.std_action)
        agent.track_data("Mass / mu_kg", mass_dist.mu_kg)
        agent.track_data("Mass / std_kg", mass_dist.std_kg)
        agent.track_data("Mass / n_samples", float(B))

        agent._mass_buf_actions.clear()
        agent._mass_buf_lp_old.clear()

    # Patch the agent's _update: run normal PPO first, then mass PPO using the
    # advantages already normalized + stored in memory by the main update.
    _orig_update = agent._update

    def _update_with_mass_ppo(timestep: int, timesteps: int) -> None:
        _orig_update(timestep, timesteps)
        # After main update, memory["advantages"] holds normalized advantages.
        # Use the centralized critic's view (any agent's memory works — all are
        # trained on the same team reward / shared critic).
        try:
            advs = agent.memories["arm"].get_tensor_by_name("advantages")
        except Exception as e:
            print(f"[mass-in-loop] WARNING: could not read advantages from memory: {e}")
            return
        _mass_ppo_step(advs)

    agent._update = _update_with_mass_ppo

    print(f"[mass-in-loop] PPO-style update enabled")
    print(f"  μ_mass     optim: Adam(lr={base_lr * lr_scale:.2e})  [= base_lr × {lr_scale}]")
    print(f"  log_σ_mass optim: Adam(lr={base_lr:.2e})")
    print(f"  ratio_clip={ratio_clip}, epochs={n_epochs}, mini_batches={n_minibatches}")
    print(f"  initial: μ={mass_dist.mu_kg:.4f} kg (action={mass_dist.mu_action:+.3f}), "
          f"σ={mass_dist.std_kg:.4f} kg (action={mass_dist.std_action:.3f})")
    print(f"  range:   [{env_cfg.object_mass_min:.3f}, {env_cfg.object_mass_max:.3f}] kg")
    return mass_dist


# Parameter names to keep at their YAML-initialized values when loading a pretrain
# checkpoint into the train policy (mirrors single-agent train.py — pop log_std so
# train starts with fresh exploration noise).
_PRETRAIN_LOAD_SKIP_KEYS = {"log_std_parameter"}


def _load_marl_partial_checkpoint(agent, path: str) -> None:
    """Load ONLY policy/value weights from a pretrain MAPPO ckpt; leave everything else fresh.

    Why partial: skrl's `agent.load()` restores everything saved in `checkpoint_modules`,
    which for MAPPO includes:
      - per-agent `optimizer` state (Adam moments from pretrain)
      - `state_preprocessor` (RunningStandardScaler fitted on pretrain obs distribution
        — no physics object → different from train distribution)
      - `shared_state_preprocessor` (centralized critic input normalization)
      - `value_preprocessor` (value normalization fitted on pretrain rewards which
        lack rew_obj_pos/rot and rew_fingertip_force contributions)

    Inheriting these into train causes the critic and policy to operate on stale
    normalization stats and biased gradient momentum, which (we believe) is the root
    cause of MARL underperformance vs single-agent. Mirrors single-agent train.py's
    `_load_partial_checkpoint`: only `agent.models` (policy/value) are restored,
    with `log_std_parameter` popped to reset exploration noise (TJ tools/reset_epoch.py
    convention).
    """
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)

    # HAPPO checkpoint layout (post-`_setup_happo_optimizers`):
    #   data["arm"]   = {"policy": sd, "optimizer": sd, "state_preprocessor": sd}
    #   data["hand"]  = {"policy": sd, "optimizer": sd, "state_preprocessor": sd}
    #   data["_critic"] = {"value": sd, "value_preprocessor": sd,
    #                      "shared_state_preprocessor": sd, "optimizer": sd}
    # agent.models layout: {uid: {"policy": Model, "value": Model}} where all
    # "value" entries point to the SAME shared instance after HAPPO setup.

    def _load_sd_filtered(module, ckpt_sd, label: str) -> bool:
        cur_sd = module.state_dict()
        updated_sd = {}
        for param_name, cur_tensor in cur_sd.items():
            if param_name in _PRETRAIN_LOAD_SKIP_KEYS:
                print(f"[partial-marl] {label}.{param_name}: SKIPPED (σ reset)")
                updated_sd[param_name] = cur_tensor
                continue
            if param_name not in ckpt_sd:
                print(f"[partial-marl] {label}.{param_name}: not in checkpoint, keeping init")
                updated_sd[param_name] = cur_tensor
                continue
            ckpt_tensor = ckpt_sd[param_name]
            if ckpt_tensor.shape == cur_tensor.shape:
                updated_sd[param_name] = ckpt_tensor
            else:
                print(f"[partial-marl] {label}.{param_name}: shape mismatch "
                      f"{list(ckpt_tensor.shape)} vs {list(cur_tensor.shape)} — keeping init")
                updated_sd[param_name] = cur_tensor
        module.load_state_dict(updated_sd)

    n_loaded = 0

    # 1) Per-agent policy
    for uid in agent.possible_agents:
        if uid not in data:
            print(f"[partial-marl] uid '{uid}' not in checkpoint, skipping")
            continue
        if "policy" not in data[uid]:
            print(f"[partial-marl] {uid}.policy: not in checkpoint, keeping init")
            continue
        _load_sd_filtered(agent.policies[uid], data[uid]["policy"], f"{uid}.policy")
        n_loaded += 1

    # 2) Shared critic (anchored under "arm" in HAPPO ckpts).
    primary = "arm"
    shared_value = agent.values[primary]
    if primary in data and "value" in data[primary]:
        _load_sd_filtered(shared_value, data[primary]["value"], f"{primary}.value")
        n_loaded += 1
    else:
        print(f"[partial-marl] shared critic value: not in checkpoint, keeping init")

    print(f"[partial-marl] Loaded {n_loaded} module(s); optimizer/preprocessors are fresh.")


def _freeze_subagent_from(agent, ckpt_path: str, which: str) -> None:
    """Load arm or hand sub-state from a MAPPO ckpt and freeze that sub-agent.

    Expected ckpt format: skrl MAPPO save, keyed like {"arm": {"policy": ..., "value": ...}, "hand": ...}.
    """
    if which not in ("arm", "hand"):
        raise ValueError(f"`which` must be 'arm' or 'hand'; got {which!r}")
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if which not in data:
        raise KeyError(f"ckpt {ckpt_path} has no '{which}' key (got top-level keys: {list(data.keys())})")
    sub = data[which]
    if "policy" in sub and which in agent.policies and agent.policies[which] is not None:
        agent.policies[which].load_state_dict(sub["policy"])
    if "value" in sub and which in agent.values and agent.values[which] is not None:
        agent.values[which].load_state_dict(sub["value"])
    for p in agent.policies[which].parameters():
        p.requires_grad_(False)
    if agent.values[which] is not agent.policies[which]:
        for p in agent.values[which].parameters():
            p.requires_grad_(False)
    print(f"[freeze] {which} policy/value loaded from {ckpt_path} and frozen.")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: DirectMARLEnvCfg, agent_cfg: dict):
    """Train MAPPO with skrl Runner + sequential forward patch."""
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

    if args_cli.timesteps:
        agent_cfg["trainer"]["timesteps"] = args_cli.timesteps
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # Disable skrl trainer's automatic "Info / {k}" logging — we route env extras
    # manually via patched record_transition (see _patch_sequential_act), so each
    # top-level prefix in env's `extras["log"]` becomes its own Tensorboard tab
    # ("Error / X", "Reward / X", "Mass / X", "Curriculum / X").
    agent_cfg["trainer"]["environment_info"] = "__disabled__"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # Log directory
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_MAPPO_torch"
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    env_cfg.log_dir = log_dir

    # Create env (MARL — no single-agent wrapper)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if not isinstance(env.unwrapped, DirectMARLEnv):
        raise TypeError(f"Task {args_cli.task} is not a DirectMARLEnv (got {type(env.unwrapped).__name__})")

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

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, agent_cfg)

    # HAPPO setup (Kuba et al. 2022, Algorithm 4):
    #   1) Single shared critic V(s_global) + critic-specific optimizer.
    #      Per-actor optimizers contain ONLY policy parameters.
    _setup_happo_optimizers(runner.agent, agent_cfg)

    # Sequential conditioning patch (forward): hand acts first → injected into arm obs.
    _patch_sequential_act(runner.agent)

    #   2) Override `_update` with HAPPO loop: one shared GAE, sequential actor
    #      updates with random permutation (Algorithm 4) and recursive M = ratio · M,
    #      and a separate critic step using critic_optimizer.
    _patch_happo_update(runner.agent, agent_cfg)

    # Mass-in-the-loop: optional, env_cfg-driven. Wraps the HAPPO `_update` so
    # mass mini-PPO runs after actor + critic updates each iteration. Reads
    # advantages back from memory (HAPPO restores Â there at end of _update).
    _setup_mass_in_loop(runner.agent, env_cfg, agent_cfg, env)

    # Optional arm-only initial σ override (value lives in MAPPO yaml under
    # `project_overrides.arm_initial_log_std`). Applied AFTER ckpt-shape patches
    # but BEFORE pretrain freeze / partial ckpt load.
    _override_arm_log_std(runner.agent, agent_cfg)

    # Optional: freeze sub-agent from external ckpt
    if args_cli.freeze_arm_from:
        _freeze_subagent_from(runner.agent, args_cli.freeze_arm_from, which="arm")
    if args_cli.freeze_hand_from:
        _freeze_subagent_from(runner.agent, args_cli.freeze_hand_from, which="hand")

    # Pretrain task: freeze log_std for both policies (mirrors single-agent pretrain σ freeze).
    if "Pretrain" in args_cli.task:
        for uid, policy in runner.agent.policies.items():
            if hasattr(policy, "log_std_parameter"):
                policy.log_std_parameter.requires_grad_(False)
                _ls = policy.log_std_parameter.detach()
                print(f"[INFO] Pretrain {uid}: log_std_parameter frozen, mean={_ls.mean().item():.4f} "
                      f"(σ_mean={_ls.exp().mean().item():.4f}, n={_ls.numel()}).")

    # Load checkpoint (pretrain → train transfer: partial load only).
    # Loads policy/value weights but skips optimizer/preprocessors so train starts
    # with fresh Adam state and RunningStandardScaler stats fitted to the train
    # distribution (with physics object, full reward terms). Mirrors single-agent
    # train.py — without this, MARL inherits pretrain's normalization which is
    # mismatched to the train reward/obs distribution.
    if resume_path:
        print(f"[INFO] Loading MAPPO checkpoint from: {resume_path}")
        _load_marl_partial_checkpoint(runner.agent, resume_path)

    # Custom callback: log per-agent success rate from env extras
    agent = runner.agent

    def log_metrics_callback(timestep, timesteps):
        actual_env = env.unwrapped
        try:
            extras = actual_env.extras
            metrics = extras.get("metrics", {}) if isinstance(extras, dict) else {}
            success_rate = metrics.get("success_rate", None)
            if success_rate is not None:
                agent.track_data("Metrics/success_rate", success_rate)
        except Exception as e:
            print(f"Error while logging metrics: {e}")
        agent._original_post_interaction(timestep=timestep, timesteps=timesteps)

    agent._original_post_interaction = agent.post_interaction
    agent.post_interaction = log_metrics_callback

    runner.run()

    training_time = round(time.time() - start_time, 2)
    print(f"Training time: {training_time} seconds")

    # Save task_info.json (under data/processed/<dataset>/ffw_sh5_marl/right/<task>/<id>/).
    # MARL uses a separate model directory so evaluate.bash can produce per-method
    # aggregates without clobbering single-agent results.
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
            _data_root / "ffw_sh5_marl" / "right"
            / env_cfg.trajectory_task
            / str(env_cfg.trajectory_data_id)
        )
        _ckpt_dir.mkdir(parents=True, exist_ok=True)
        _task_info = {
            "task": args_cli.task,
            "algorithm": "MAPPO",
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

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
