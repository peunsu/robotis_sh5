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
parser.add_argument("--clip_class", type=str, default=None, help="ParaHome clip class (g1 loco-manip); overrides env_cfg.clip_class.")
parser.add_argument("--clip_name", type=str, default=None, help="ParaHome clip name (g1 loco-manip); overrides env_cfg.clip_name.")
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
import math
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
# Feed the KL-adaptive LR scheduler the ANALYTIC Gaussian KL (rl_games/TJ convention) instead of
# skrl's k3 estimator. Set False to restore stock skrl behaviour. See _patch_analytic_kl.
_USE_ANALYTIC_KL = True
# Dual-clip PPO (Ye et al., AAAI 2020, arXiv:1912.09729). Set _DUAL_CLIP_C to None to disable and
# restore stock skrl behaviour. c must be > 1 + ratio_clip (asserted). See _patch_dual_clip.
_DUAL_CLIP_C: float | None = 3.0


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

    # Rebuild LR scheduler if configured. skrl 2.0.0: cfg is a dataclass and
    # learning_rate_scheduler / _kwargs are stored as single-element lists.
    if agent.cfg.learning_rate_scheduler[0] is not None:
        agent.scheduler = agent.cfg.learning_rate_scheduler[0](
            agent.optimizer, **agent.cfg.learning_rate_scheduler_kwargs[0]
        )

    # Monkey-patch record_transition to propagate terminated signal to the mass cache.
    # skrl 2.0.0: record_transition is keyword-only and adds observations/next_observations.
    _orig_record = agent.record_transition

    def _record_with_mass(*, observations, states, actions, rewards, next_observations,
                          next_states, terminated, truncated, infos, timestep, timesteps):
        done = (terminated | truncated).squeeze(-1)
        new_policy.update_mass_terminated(done)
        _orig_record(observations=observations, states=states, actions=actions, rewards=rewards,
                     next_observations=next_observations, next_states=next_states,
                     terminated=terminated, truncated=truncated, infos=infos,
                     timestep=timestep, timesteps=timesteps)

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

    def _record_with_env_log(*, observations, states, actions, rewards, next_observations,
                             next_states, terminated, truncated, infos, timestep, timesteps):
        _orig_record(observations=observations, states=states, actions=actions, rewards=rewards,
                     next_observations=next_observations, next_states=next_states,
                     terminated=terminated, truncated=truncated, infos=infos,
                     timestep=timestep, timesteps=timesteps)
        log_dict = infos.get("log") if isinstance(infos, dict) else None
        if log_dict:
            for k, v in log_dict.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    agent.track_data(k, v.item())
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    # python scalars (e.g. Curriculum/reached_frame, pretrain_fallback) were
                    # silently dropped before — forward them too.
                    agent.track_data(k, float(v))

    agent.record_transition = _record_with_env_log
    print(f"[env-info-log] Manual env extras['log'] → agent.track_data (bypasses skrl 'Info / ' prefix).")


def _patch_kl_logging(agent) -> None:
    """Log the per-update mean approx-KL to Tensorboard as 'Policy / KL'.

    g1 uses skrl's NATIVE PPO `_update`, which computes the mean approx-KL (over the
    update's mini-batches) and feeds it to the KLAdaptiveLR scheduler via
    `scheduler.step(kl)` — but never `track_data`'s it. Rather than fork the whole
    `_update`, wrap the scheduler's `step()` to capture that same kl value and log it.
    `step()` is called once per PPO update from inside `_update`, so the value lands in
    the same write cycle as Loss/Policy etc. Harmless no-op if there is no scheduler or
    a step without a kl arg (e.g. non-KLAdaptive schedulers pass no kl → nothing logged)."""
    sched = getattr(agent, "scheduler", None)
    if sched is None or not hasattr(sched, "step"):
        print("[kl-log] no scheduler.step to wrap; 'Policy / KL' NOT logged.")
        return
    _real_step = sched.step

    def _step_logging_kl(*args, **kwargs):
        kl = args[0] if args else kwargs.get("kl", None)
        if kl is not None:
            try:
                agent.track_data("Policy / KL", float(kl))
            except Exception:
                pass
        return _real_step(*args, **kwargs)

    sched.step = _step_logging_kl
    print("[kl-log] wrapped scheduler.step → track 'Policy / KL' per update.")


def _patch_dual_clip(agent, c: float = 3.0) -> None:
    """Dual-clip PPO (Ye et al., AAAI 2020 — arXiv:1912.09729; they use eps=0.2, c=3).

    WHY
    ---
    skrl's policy loss is `-min(A·r, A·clip(r, 1-eps, 1+eps)).mean()` (`ppo.py:411-417`). `min` picks
    the more negative branch, so the clip binds in ONE direction only:
      * A > 0, r -> inf:  `A·1.2 < A·r`, so `min` takes the CLIPPED branch -> loss saturates at
        -1.2·A and the gradient is exactly 0. Clipping works as intended.
      * A < 0, r -> inf:  `A·r` is far MORE negative than `A·1.2`, so `min` takes the UNCLIPPED
        branch -> loss = |A|·r, growing without bound, and d(loss)/d(logratio) = |A|·r with it.
    Measured on this task: logratio ~= 61 (Diag/kl_k3 2.8e26) -> policy loss 3.1e26. That forward
    value is still FINITE in fp32 (max 3.4e38 = e^88.7); it is the BACKWARD pass that overflows ->
    grad norm inf -> torch's clip_grad_norm_ scales every gradient by max_norm/(inf+1e-6) = 0 -> the
    policy freezes permanently, the analytic KL then reads 0 forever, and KLAdaptiveLR walks the LR up
    to max_lr. This is also why the usual "clamp the log-ratio at 88 before exp()" tip is a no-op
    here: 88 > 61, so it never fires.

    WHAT
    ----
    Adds the published floor `c·A`, on the A < 0 branch only:

        L = -E[ max( min(r·A, clip(r,1-eps,1+eps)·A), c·A ) ]   for A <  0
        L = -E[      min(r·A, clip(r,1-eps,1+eps)·A)        ]   for A >= 0

    For A < 0 this collapses to `L = clamp(r, 1-eps, c)·|A|` (verified against the reference form over
    a grid of A and log-ratio) — i.e. dual-clip is exactly a two-sided TRUNCATION of the importance
    ratio for negative-advantage samples, with c as the upper cut (cf. V-trace's rho-bar, which
    truncates at 1.0, so c=3 is lenient by comparison). The upper cut is the part that matters here:
    it is what turns the unbounded `|A|·r` into `c·|A|`. Two constraints, both asserted:
      * c > 1              else the floor binds at r = 1 and zeroes the gradient of EVERY A<0 sample.
      * c > 1 + ratio_clip else it binds inside the band where the standard clip still wants gradient.

    c = 3 is the paper's empirical value — it carries no ablation there, and OpenDILab's reference
    implementation ships no default at all. So treat it as a SAFETY NET rather than a tuned knob and
    read `Diag / dual_clip_frac`: this run is nominally ON-policy (same rollout, immediate update), so
    the ratio should be ~1 at epoch 0 and a healthy run leaves the floor idle. A persistently nonzero
    fraction does NOT mean "retune c" — it means the ignition (obs-scaler cold-start jump / mu
    blow-up) is still present AND that a chunk of every batch now gets exactly zero gradient, which is
    the same flat-exterior trap one level up.

    Logged per update, all computed on the RAW un-truncated log-ratio. That is deliberate: truncating
    the DIAGNOSTIC too would hide the very tail we use to locate the cause.
      Diag / dual_clip_frac      fraction of ALL samples where the floor binds (A<0 and r>c)
      Diag / dual_clip_frac_neg  the same count as a fraction of the A<0 samples only
      Diag / logratio_max/_min   raw log-ratio extremes (finite even where exp() would overflow)
      Diag / neg_adv_frac        fraction of A<0 samples (context for the two fractions above)

    HOW
    ---
    Replaces `agent.update` with a copy of skrl 2.1.0 `PPO.update` whose policy-loss expression
    carries the floor. Everything else is byte-faithful to upstream, INCLUDING the k3 KL handed to the
    scheduler, so _patch_analytic_kl still overrides that exactly as before. Apply this BEFORE
    _patch_analytic_kl so that wrapper wraps this fork. On a skrl upgrade, diff upstream `update()`
    against this body.
    """
    import itertools

    import torch.nn as nn
    import torch.nn.functional as F
    from skrl import config as _sk_config
    from skrl.agents.torch.ppo.ppo import compute_gae
    from skrl.resources.schedulers.torch import KLAdaptiveLR

    _eps = float(agent.cfg.ratio_clip)
    assert c > 1.0, f"dual-clip c must be > 1.0 (else the floor binds at r=1), got {c}"
    assert c > 1.0 + _eps, f"dual-clip c must be > 1 + ratio_clip ({1.0 + _eps}), got {c}"
    _ln_c = math.log(c)

    def _update(*, timestep: int, timesteps: int) -> None:
        a = agent
        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=a._device_type, enabled=a.cfg.mixed_precision):
            inputs = {
                "observations": a._observation_preprocessor(a._current_next_observations),
                "states": a._state_preprocessor(a._current_next_states),
            }
            a.value.enable_training_mode(False)
            last_values, _ = a.value.act(inputs, role="value")
            a.value.enable_training_mode(True)
            last_values = a._value_preprocessor(last_values, inverse=True)

        values = a.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=a.memory.get_tensor_by_name("rewards"),
            terminated=a.memory.get_tensor_by_name("terminated"),
            truncated=a.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=a.cfg.discount_factor,
            lambda_coefficient=a.cfg.gae_lambda,
            time_limit_bootstrap=a.cfg.time_limit_bootstrap,
        )

        a.memory.set_tensor_by_name("values", a._value_preprocessor(values, train=True))
        a.memory.set_tensor_by_name("returns", a._value_preprocessor(returns, train=True))
        a.memory.set_tensor_by_name("advantages", advantages)

        cumulative_policy_loss = 0
        _rm = []          # [failure-sigma] V1: 첫 에포크의 PPO 비율 중앙값
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        # [DUAL-CLIP] diagnostics, accumulated over every mini-batch of the update. Kept as DEVICE
        # tensors and read out once at the end: a per-mini-batch `.item()` would add a GPU sync per
        # counter per mini-batch (learning_epochs × mini_batches = 40 here), on top of the three
        # upstream already pays for the losses.
        _dev = a.device
        n_tot = 0
        acc_neg = torch.zeros((), dtype=torch.long, device=_dev)
        acc_bind = torch.zeros((), dtype=torch.long, device=_dev)
        acc_lr_max = torch.full((), -math.inf, device=_dev)
        acc_lr_min = torch.full((), math.inf, device=_dev)

        for epoch in range(a.cfg.learning_epochs):
            kl_divergences = []

            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in a.memory.sample(
                names=a._tensors_names, batch_size=len(a.memory), mini_batches=a.cfg.mini_batches
            ):

                with torch.autocast(device_type=a._device_type, enabled=a.cfg.mixed_precision):
                    inputs = {
                        "observations": a._observation_preprocessor(sampled_observations, train=not epoch),
                        "states": a._state_preprocessor(sampled_states, train=not epoch),
                    }

                    _, outputs = a.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    # compute approximate KL divergence (upstream k3; _patch_analytic_kl replaces the
                    # value actually fed to the scheduler, so keep this computation as-is)
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if a.cfg.kl_threshold and kl_divergence > a.cfg.kl_threshold:
                        break

                    # compute entropy loss
                    if a.cfg.entropy_loss_scale:
                        entropy_loss = -a.cfg.entropy_loss_scale * a.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    # [ROLLBACK MARKER: failure-sigma] V1 검증. 롤아웃과 업데이트에 같은 beta 가
                    # 적용되면 첫 에포크의 비율은 1 에서 시작합니다. 어긋나면 여기가 즉시 드러납니다
                    # (beta=1.5, 36차원이면 3.7e-4 까지 떨어져 전부 클립됩니다).
                    if epoch == 0:
                        with torch.no_grad():
                            _rm.append(float(ratio.median()))
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - a.cfg.ratio_clip, 1.0 + a.cfg.ratio_clip
                    )
                    # ── [DUAL-CLIP] the ONLY deviation from upstream ─────────────────────────────
                    _inner = torch.min(surrogate, surrogate_clipped)
                    policy_loss = -torch.where(
                        sampled_advantages < 0, torch.max(_inner, c * sampled_advantages), _inner
                    ).mean()
                    # ── END DUAL-CLIP ───────────────────────────────────────────────────────────

                    # compute value loss
                    predicted_values, _ = a.value.act(inputs, role="value")

                    if a.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-a.cfg.value_clip, max=a.cfg.value_clip
                        )
                    value_loss = a.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # [DUAL-CLIP] diagnostics on the RAW log-ratio (never the truncated one)
                with torch.no_grad():
                    _lr = (next_log_prob - sampled_log_prob).detach().float()
                    _neg = sampled_advantages < 0
                    n_tot += _lr.numel()
                    acc_neg += _neg.sum()
                    acc_bind += (_neg & (_lr > _ln_c)).sum()
                    acc_lr_max = torch.maximum(acc_lr_max, _lr.max())
                    acc_lr_min = torch.minimum(acc_lr_min, _lr.min())

                # optimization step
                a.optimizer.zero_grad()
                a.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if _sk_config.torch.is_distributed:
                    a.policy.reduce_parameters()
                    if a.policy is not a.value:
                        a.value.reduce_parameters()

                if a.cfg.grad_norm_clip > 0:
                    a.scaler.unscale_(a.optimizer)
                    if a.policy is a.value:
                        nn.utils.clip_grad_norm_(a.policy.parameters(), a.cfg.grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(a.policy.parameters(), a.value.parameters()), a.cfg.grad_norm_clip
                        )

                a.scaler.step(a.optimizer)
                a.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if a.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if a.scheduler:
                if isinstance(a.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=a.device).mean()
                    if _sk_config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= _sk_config.torch.world_size
                    a.scheduler.step(kl.item())
                else:
                    a.scheduler.step()

        # record data
        _n = a.cfg.learning_epochs * a.cfg.mini_batches
        if _rm:
            a.track_data("Diag / ppo_ratio_med", sum(_rm) / len(_rm))
        a.track_data("Loss / Policy loss", cumulative_policy_loss / _n)
        a.track_data("Loss / Value loss", cumulative_value_loss / _n)
        if a.cfg.entropy_loss_scale:
            a.track_data("Loss / Entropy loss", cumulative_entropy_loss / _n)

        a.track_data("Policy / Standard deviation", a.policy.distribution(role="policy").stddev.mean().item())

        if a.scheduler:
            a.track_data("Learning / Learning rate", a.scheduler.get_last_lr()[0])

        # [DUAL-CLIP] safety-net diagnostics (single read-out → 4 GPU syncs per update, not per batch)
        if n_tot:
            n_neg = int(acc_neg.item())
            n_bind = int(acc_bind.item())
            a.track_data("Diag / dual_clip_frac", n_bind / n_tot)
            a.track_data("Diag / dual_clip_frac_neg", n_bind / max(n_neg, 1))
            a.track_data("Diag / neg_adv_frac", n_neg / n_tot)
            a.track_data("Diag / logratio_max", float(acc_lr_max.item()))
            a.track_data("Diag / logratio_min", float(acc_lr_min.item()))

    agent.update = _update
    print(f"[dual-clip] forked PPO.update: A<0 branch floored at c·A (c={c}, ratio_clip={_eps}) → "
          f"policy loss ≤ {c}·|A| instead of unbounded |A|·r; track 'Diag / dual_clip_frac'.")


def _patch_grad_norm_logging(agent) -> None:
    """Capture the PRE-clip gradient norm that skrl throws away, to find the exact freeze moment.

    skrl calls `nn.utils.clip_grad_norm_(...)` (`ppo.py:437-442`) and discards its return value — which
    is the total gradient norm BEFORE clipping. That number is the missing link in the observed failure:
    torch computes `clip_coef = max_norm / (total_norm + 1e-6)` and then
    `grad *= clamp(clip_coef, max=1.0)` with `error_if_nonfinite=False` by default
    (`torch/nn/utils/clip_grad.py`), so a single non-finite `total_norm` silently multiplies EVERY
    gradient by 0 → the parameters stop changing for good. Symptoms already measured: Diag/dmu_rms = 0
    exactly, σ frozen to 4 decimals, analytic KL ≡ 0, LR pinned at max_lr.

    Logged per scheduler step (flushed by _patch_analytic_kl's wrapper):
      Diag / grad_norm            mean pre-clip norm over the finite mini-batches
      Diag / grad_norm_max        largest finite pre-clip norm
      Diag / grad_nonfinite_frac  fraction of mini-batches whose norm was inf/NaN  ← the freeze flag
      Diag / grad_clip_scale      mean min(1, max_norm/(norm+1e-6)) actually applied; 0 ⇒ zeroed update

    Patches torch.nn.utils.clip_grad_norm_ process-wide (skrl resolves it at call time). Nothing else
    in this process uses it.
    """
    import torch.nn.utils as _nnu

    _real_clip = _nnu.clip_grad_norm_
    acc: list[float] = []
    agent._grad_norm_acc = acc

    def _clip_logging(parameters, max_norm, *args, **kwargs):
        total_norm = _real_clip(parameters, max_norm, *args, **kwargs)
        try:
            acc.append(float(total_norm))
        except Exception:
            pass
        return total_norm

    _nnu.clip_grad_norm_ = _clip_logging

    # MEASUREMENT ONLY — this patch does not change the update. A non-finite pre-clip norm still
    # reaches torch's clip (`clip_coef = max_norm/inf = 0` → every gradient multiplied by zero →
    # parameters frozen for good → analytic KL ≡ 0 → KLAdaptiveLR walks the LR to max_lr).
    #   An optimizer-step guard that skipped those mini-batches (GradScaler-style) was tried on
    #   2026-07-28 and REVERTED on request: it masks the symptom while the cause — the A<0 branch of
    #   PPO's `-min(A·r, A·clip(r))`, which is unbounded for r ≫ 1+ε and overflows fp32 — keeps
    #   firing. Watch `Diag / grad_nonfinite_frac` (freeze flag) and `Diag / grad_clip_scale` (0 ⇒
    #   zeroed update) instead; fix the log-ratio, not the optimizer step.
    print(f"[grad-norm] wrapped clip_grad_norm_ → track 'Diag / grad_norm*' "
          f"(grad_norm_clip={getattr(agent.cfg, 'grad_norm_clip', None)}). Measurement only: "
          f"non-finite gradients are NOT skipped.")


def _patch_analytic_kl(agent) -> None:
    """Feed the KL-adaptive LR scheduler the ANALYTIC Gaussian KL instead of skrl's k3 estimator.

    WHY
    ---
    skrl computes `ratio = logp_new - logp_old` SUMMED over all action dims and then
    `kl = ((exp(ratio) - 1) - ratio).mean()` (`ppo.py:396-397`) — i.e. the statistic is EXPONENTIAL
    in the joint log-ratio, so with a 100-D action a single well-aligned sample can drive it to 1e14
    while the actual policy change is ordinary. Measured on this task: `Policy / KL` never dropped
    below ~0.6 in ANY run (median 2.5-12 vs `kl_threshold` 0.016) → KLAdaptiveLR was saturated 100 %
    of the time and the LR sat on its floor from the first updates onward, i.e. the controller was
    dead.

    rl_games (TJ's `gr`, and GRAIL's stack) instead uses the closed-form diagonal-Gaussian KL
    (`rl_games/algos_torch/torch_ext.py::policy_kl`), summed over action dims and averaged over
    samples — QUADRATIC in Δμ/σ, so no single sample can dominate:

        KL = Σ_d [ log(σ_new/σ_old) + (σ_old² + (μ_old-μ_new)²) / (2 σ_new²) - 1/2 ]

    HOW (no fork of skrl's `update()`)
    ----------------------------------
    1. `agent.update` is wrapped ONLY to refresh a frozen snapshot of the policy at the start of each
       update — that snapshot is the "old" (rollout-time) policy.
    2. `policy.act` is wrapped; the update-time calls are identified by the presence of
       `taken_actions` in the inputs (the rollout call does not pass it, `ppo.py:209-220` vs `:391`).
       For each such mini-batch we run the frozen snapshot on the SAME already-preprocessed inputs
       and accumulate the analytic KL.
    3. The wrapped `scheduler.step` replaces skrl's k3 value with the mean of those per-mini-batch
       analytic KLs, and logs both (`Policy / KL` = analytic/fed, `Diag / kl_k3` = skrl's original).

    Because both policies see the SAME normalized observations, this measures the policy-parameter
    change only — it is immune to the observation-preprocessor drift that inflates the k3 value
    (skrl refreshes the running scaler INSIDE the update, `ppo.py:387 train=not epoch`).

    Cost: one extra no-grad forward pass per mini-batch (learning_epochs × mini_batches per update)
    plus one policy-sized copy. Watch `Stats / Algorithm update time (ms)`.
    """
    import copy

    sched = getattr(agent, "scheduler", None)
    if sched is None or not hasattr(sched, "step"):
        print("[analytic-kl] no scheduler.step to wrap; falling back to plain KL logging.")
        _patch_kl_logging(agent)
        return

    old_policy = copy.deepcopy(agent.policy).eval()
    for _p in old_policy.parameters():
        _p.requires_grad_(False)
    kl_acc: list[torch.Tensor] = []
    # DIAGNOSTIC accumulators (do NOT feed the scheduler — they only get logged). Purpose: tell apart
    # "the policy genuinely did not move" from "the fp32 formula lost the signal to cancellation".
    # The `(σ² + Δμ²)/(2σ²) − 0.5` form subtracts two nearly-equal numbers, so in float32 any per-dim
    # contribution below ~6e-8 vanishes and the total reads exactly 0 (or even slightly negative).
    dmu_acc: list[torch.Tensor] = []    # RMS over samples+dims of Δμ/σ  → the actual policy movement
    dsig_acc: list[torch.Tensor] = []   # mean |Δσ|/σ                    → separates a σ-only change
    kl64_acc: list[torch.Tensor] = []   # same KL in float64, cancellation-free form (reference)
    lrat_acc: list[torch.Tensor] = []   # max |logp_new - logp_snapshot| (policy-change-only ratio)

    _real_update = agent.update

    def _update_with_snapshot(*args, **kwargs):
        # "old" = the policy that generated this rollout.
        old_policy.load_state_dict(agent.policy.state_dict())
        kl_acc.clear()
        # OBS-PREPROCESSOR DRIFT PROBE. skrl refreshes the RunningStandardScaler INSIDE the update
        # (ppo.py:387 `train=not epoch`), so `sampled_log_prob` (rollout stats) and `next_log_prob`
        # (post-refresh stats) can disagree even with bit-identical weights — the suspected source of
        # the ratio→e^60 / loss→1e27 blow-up. Report the drift in NORMALIZED-INPUT units, which is the
        # quantity that actually moves mu:
        #   x = (o - mean)/sqrt(var)  ⇒  additive shift |Δmean|/sqrt(var),  scale distortion |Δvar|/(2·var)
        _pp = getattr(agent, "_observation_preprocessor", None)
        _snap = None
        if hasattr(_pp, "running_mean"):
            _snap = (_pp.running_mean.clone(), _pp.running_variance.clone(), float(_pp.current_count))
        # STORED-ROLLOUT PROBE — read what is actually in the rollout buffer, which is what skrl's k3
        # compares against. With bit-identical weights AND a settled obs scaler the ratio still reaches
        # e^60, so the remaining candidate is the ABSOLUTE magnitude of the stored actions / log-probs:
        # `clip_actions: False` means PPO scores the raw pre-clip Gaussian sample, and a sample far from
        # mu contributes (a-mu)^2/(2 sigma^2) which is enormous — then any tiny difference explodes.
        try:
            _mem = agent.memory
            _lp = _mem.get_tensor_by_name("log_prob")
            _ac = _mem.get_tensor_by_name("actions")
            agent.track_data("Diag / logp_old_min", float(_lp.min()))
            agent.track_data("Diag / logp_old_mean", float(_lp.mean()))
            agent.track_data("Diag / logp_old_max", float(_lp.max()))
            agent.track_data("Diag / taken_a_absmax", float(_ac.abs().max()))
            agent.track_data("Diag / taken_a_rms", float(_ac.pow(2).mean().sqrt()))
            agent.track_data("Diag / logp_old_nonfinite", float((~torch.isfinite(_lp)).float().mean()))
        except Exception:
            pass
        out = _real_update(*args, **kwargs)
        if _snap is not None:
            try:
                m0, v0, c0 = _snap
                shift = ((_pp.running_mean - m0).abs() / (v0.sqrt() + 1e-12))
                scale = ((_pp.running_variance - v0).abs() / (2.0 * v0 + 1e-12))
                agent.track_data("Diag / scaler_x_shift", float(shift.mean()))
                agent.track_data("Diag / scaler_x_shift_max", float(shift.max()))
                agent.track_data("Diag / scaler_x_scale", float(scale.mean()))
                agent.track_data("Diag / scaler_x_scale_max", float(scale.max()))
                agent.track_data("Diag / scaler_count", float(_pp.current_count))
                agent.track_data("Diag / scaler_count_growth", float(_pp.current_count) / max(c0, 1.0))
            except Exception:
                pass
        return out

    agent.update = _update_with_snapshot

    _real_act = agent.policy.act

    def _act_with_kl(inputs, *, role: str = ""):
        out = _real_act(inputs, role=role)
        if role == "policy" and "taken_actions" in inputs:      # update-time evaluation only
            try:
                with torch.no_grad():
                    mu_new, ls_new = out[1]["mean_actions"], out[1]["log_std"]
                    _, out_old = old_policy.act(inputs, role="policy")
                    mu_old, ls_old = out_old["mean_actions"], out_old["log_std"]
                    # NOTE rl_games writes `log(σn/σo + 1e-5)` and `/(2σn² + 1e-5)`. Those guards are
                    # ADDITIVE and get summed over D dims, so at our scale (D=100, σ≈0.37, KL≈0.02)
                    # they bias the result by −4.2 % — i.e. 6 % of kl_threshold, in the very signal
                    # that drives the LR. Verified numerically. Use a σ floor instead: exact at every
                    # realistic scale, still safe against a degenerate σ→0.
                    s_new = ls_new.exp().clamp_min(1e-12)
                    s_old = ls_old.exp().clamp_min(1e-12)
                    kl = (torch.log(s_new / s_old)
                          + (s_old ** 2 + (mu_old - mu_new) ** 2) / (2.0 * s_new ** 2)
                          - 0.5).sum(dim=-1).mean()
                    kl_acc.append(kl.detach())
                    # ---- diagnostics only (never fed to the scheduler) ----
                    dmu_acc.append((((mu_new - mu_old) / s_new) ** 2).mean().sqrt().detach())
                    dsig_acc.append(((s_new - s_old) / s_old).abs().mean().detach())
                    # policy-change-only log-ratio: the snapshot's log_prob on the SAME stored actions.
                    # Must be ~0 while the policy is frozen; if it is ~0 while skrl's k3 says e^60, the
                    # gap lives in the STORED log_prob (see the rollout probe in the update wrapper).
                    _lp_snap = out_old.get("log_prob")
                    if _lp_snap is not None:
                        lrat_acc.append((out[1]["log_prob"] - _lp_snap).abs().max().detach())
                    _mo, _mn = mu_old.double(), mu_new.double()
                    _so, _sn = s_old.double(), s_new.double()
                    kl64_acc.append((torch.log(_sn / _so)
                                     + (_so ** 2 - _sn ** 2) / (2.0 * _sn ** 2)
                                     + (_mo - _mn) ** 2 / (2.0 * _sn ** 2)).sum(dim=-1).mean().detach())
            except Exception:
                pass
        return out

    agent.policy.act = _act_with_kl

    _real_step = sched.step

    def _step_analytic_kl(*args, **kwargs):
        k3 = args[0] if args else kwargs.get("kl", None)
        if kl_acc:
            kl = float(torch.stack(kl_acc).mean())
            kl_acc.clear()
            try:
                agent.track_data("Policy / KL", kl)                       # analytic — drives the LR
                if k3 is not None:
                    agent.track_data("Diag / kl_k3", float(k3))           # skrl's original, for reference
                if dmu_acc:
                    agent.track_data("Diag / dmu_rms", float(torch.stack(dmu_acc).mean()))
                    agent.track_data("Diag / dsigma_rel", float(torch.stack(dsig_acc).mean()))
                    agent.track_data("Diag / kl_fp64", float(torch.stack(kl64_acc).mean()))
                    if lrat_acc:
                        agent.track_data("Diag / logratio_snap_absmax", float(torch.stack(lrat_acc).max()))
                gn = getattr(agent, "_grad_norm_acc", None)
                if gn:
                    finite = [x for x in gn if math.isfinite(x)]
                    mx = float(agent.cfg.grad_norm_clip)
                    scale = [min(1.0, mx / (x + 1e-6)) if math.isfinite(x) else 0.0 for x in gn]
                    agent.track_data("Diag / grad_nonfinite_frac", 1.0 - len(finite) / len(gn))
                    agent.track_data("Diag / grad_clip_scale", sum(scale) / len(scale))
                    if finite:
                        agent.track_data("Diag / grad_norm", sum(finite) / len(finite))
                        agent.track_data("Diag / grad_norm_max", max(finite))
                    gn.clear()
            except Exception:
                pass
            dmu_acc.clear(); dsig_acc.clear(); kl64_acc.clear(); lrat_acc.clear()
            return _real_step(kl)
        return _real_step(*args, **kwargs)                                # nothing captured → stock path

    sched.step = _step_analytic_kl
    print("[analytic-kl] scheduler now driven by the closed-form Gaussian KL "
          "(rl_games convention); skrl's k3 value is logged as 'Diag / kl_k3'.")


def _patch_failure_sigma(agent, n_action: int, dims: str) -> None:
    """[ROLLBACK MARKER: failure-sigma] 실패 구간에서 sampling sigma 를 beta 배로 키웁니다.

    env 가 관측 마지막 열에 beta 를 실어 보냅니다. 여기서 두 가지를 합니다.

      1. 관측 전처리기를 감싸 beta 열은 정규화하지 않고 통과시킵니다.
         RunningStandardScaler 에 넣으면 학습 초기 beta 가 상수 1.0 이라 분산이 0 으로 수렴하고,
         정규화/역정규화가 0 으로 나누는 꼴이 되어 터집니다.

      2. 정책의 compute 를 감싸 beta 를 log_std 에 더합니다 (log_std += log beta).
         신경망에는 beta 를 뺀 앞부분만 넣으므로 mu 는 beta 를 보지 않습니다 — 탐색 폭만 바뀌고
         행동 자체는 안 바뀝니다.

    PPO 는 건드리지 않습니다. GaussianMixin.act 가 이 log_std 로 분포를 만들고 같은 분포로
    log_prob / 엔트로피를 계산하며, 업데이트 때도 저장된 관측에서 같은 beta 가 나오므로 비율이
    1 에서 시작합니다. (검증: Diag / ppo_ratio_med 가 첫 업데이트에서 1.00 ± 0.01)
    """
    import torch as _t

    policy = agent.models["policy"]

    # ── 1. 전처리기: 마지막 열(beta)은 통과 ────────────────────────────────────
    _pre = agent._observation_preprocessor

    def _pre_keep_last(x, train=False, **kw):
        # 스케일러는 관측 전체 폭(767)으로 만들어져 있으므로 통째로 통과시킨 뒤 beta 열만 원본으로
        # 되돌립니다. 열을 잘라 넣으면 폭이 안 맞아 터집니다. beta 쪽 running stat 은 갱신되지만
        # 그 정규화 값을 쓰지 않으므로 무해합니다.
        out = _pre(x, train=train, **kw)
        return _t.cat([out[..., :-1], x[..., -1:]], dim=-1)

    agent._observation_preprocessor = _pre_keep_last

    # ── 2. 정책: log_std 에 log(beta) ─────────────────────────────────────────
    # 인스턴스가 아니라 CLASS 에 붙입니다. 인스턴스에 붙이면 _patch_analytic_kl 의
    # copy.deepcopy(agent.policy) 스냅샷이 그 클로저를 그대로 복사하는데, 클로저가 원본 정책의
    # 바운드 메서드를 잡고 있어서 스냅샷이 원본 가중치로 계산합니다. 그러면 mu_old == mu_new 가 되어
    # KL 이 항상 정확히 0 → KL 적응형 학습률이 무한정 올라가 정책이 발산합니다
    # (실측: 보상 0.000, logratio_max -7588). 클래스에 붙이면 self 가 호출 시점에 풀리므로
    # 스냅샷은 스냅샷의 가중치로 계산합니다.
    # "hand" 면 뒤쪽 36차원(양손)만, "all" 이면 100차원 전체
    _lo = 0 if dims == "all" else n_action - 36
    _cls = type(policy)
    _orig = _cls.compute

    def _compute_beta(self, inputs, role=""):
        obs = inputs.get("observations", None)
        if obs is None or obs.shape[-1] < 2:
            return _orig(self, inputs, role)
        beta = obs[..., -1:].clamp(min=1e-6)
        # 열을 잘라내면 안 됩니다 — 자동 생성된 compute 가 관측을 원래 폭으로 unflatten 하므로
        # 폭이 바뀌면 터집니다. 상수 0 으로 덮어써서 mu 쪽 신경망이 beta 를 못 보게 합니다.
        obs_masked = obs.clone()
        obs_masked[..., -1] = 0.0
        mean_actions, outputs = _orig(self, {**inputs, "observations": obs_masked}, role)
        ls = outputs["log_std"]
        if ls.dim() == 1:                       # 상태 비의존 파라미터 → 배치로 확장
            ls = ls.unsqueeze(0).expand(beta.shape[0], -1).clone()
        else:
            ls = ls.clone()
        ls[..., _lo:] = ls[..., _lo:] + beta.log()
        outputs["log_std"] = ls
        return mean_actions, outputs

    _cls.compute = _compute_beta
    print(f"[failure-sigma] 정책 패치 완료 — log_std[{_lo}:{n_action}] += log(beta), "
          f"전처리기는 beta 열을 통과시킵니다.")


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
    from skrl.agents.torch.ppo.ppo import compute_gae

    # ── Step 1: Add sigma_grad_flg tensor to rollout memory ───────────────────
    # agent.init() (called by the trainer inside Runner.__init__) already created the
    # standard tensors and populated _tensors_names, so this append is safe here.
    agent.memory.create_tensor(name="sigma_grad_flg", size=1, dtype=torch.float32)
    agent._tensors_names.append("sigma_grad_flg")

    # ── Step 2: Store sigma_grad_flg at each rollout step ─────────────────────
    # skrl 2.0.0: record_transition is keyword-only and adds observations/next_observations.
    _orig_record = agent.record_transition

    def _record_with_sigma(*, observations, states, actions, rewards, next_observations,
                           next_states, terminated, truncated, infos, timestep, timesteps):
        _orig_record(observations=observations, states=states, actions=actions, rewards=rewards,
                     next_observations=next_observations, next_states=next_states,
                     terminated=terminated, truncated=truncated, infos=infos,
                     timestep=timestep, timesteps=timesteps)
        # Write is_reached_end into the slot just committed by add_samples.
        # add_samples increments memory_index after writing, so the written slot is index-1.
        prev_idx = (agent.memory.memory_index - 1) % agent.memory.memory_size
        sigma = float(env_wrapper.unwrapped.is_reached_end)
        agent.memory.tensors["sigma_grad_flg"][prev_idx].fill_(sigma)

    agent.record_transition = _record_with_sigma

    # ── Step 3: Replace update() with per-mini-batch entropy weighting ────────
    # Mirrors skrl 2.0.0 PPO.update() exactly, injecting ONLY the GR sigma-weighted
    # entropy scale. NOTE vs the old (1.4.3) patch: GAE now uses `terminated` alone
    # (not terminated|truncated) — this is the 2.0.0 truncation-bootstrap fix, and the
    # builtin compute_gae already normalizes advantages.
    def _update_with_per_batch_entropy(*, timestep: int, timesteps: int) -> None:
        with torch.no_grad(), torch.autocast(device_type=agent._device_type, enabled=agent.cfg.mixed_precision):
            inputs = {
                "observations": agent._observation_preprocessor(agent._current_next_observations),
                "states": agent._state_preprocessor(agent._current_next_states),
            }
            agent.value.enable_training_mode(False)
            last_values, _ = agent.value.act(inputs, role="value")
            agent.value.enable_training_mode(True)
            last_values = agent._value_preprocessor(last_values, inverse=True)

        values = agent.memory.get_tensor_by_name("values")
        # skrl 2.1.0: compute_gae adds `truncated` + `time_limit_bootstrap` and renames
        # next_values → last_values. With time_limit_bootstrap=False, GAE uses `terminated`
        # alone (unchanged from our 2.0.0 mirror).
        returns, advantages = compute_gae(
            rewards=agent.memory.get_tensor_by_name("rewards"),
            terminated=agent.memory.get_tensor_by_name("terminated"),
            truncated=agent.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=agent.cfg.discount_factor,
            lambda_coefficient=agent.cfg.gae_lambda,
            time_limit_bootstrap=agent.cfg.time_limit_bootstrap,
        )
        agent.memory.set_tensor_by_name("values", agent._value_preprocessor(values, train=True))
        agent.memory.set_tensor_by_name("returns", agent._value_preprocessor(returns, train=True))
        agent.memory.set_tensor_by_name("advantages", advantages)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        for epoch in range(agent.cfg.learning_epochs):
            kl_divergences = []

            # skrl 2.1.0: re-sample mini-batches each epoch (per-epoch shuffling) via
            # memory.sample(...) instead of a single pre-loop sample_all(...).
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_sigma,          # sigma_grad_flg: (mini_batch_size, 1)
            ) in agent.memory.sample(
                names=agent._tensors_names,
                batch_size=len(agent.memory),
                mini_batches=agent.cfg.mini_batches,
            ):

                with torch.autocast(device_type=agent._device_type, enabled=agent.cfg.mixed_precision):

                    inputs = {
                        "observations": agent._observation_preprocessor(sampled_observations, train=not epoch),
                        "states": agent._state_preprocessor(sampled_states, train=not epoch),
                    }

                    _, outputs = agent.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if agent.cfg.kl_threshold and kl_divergence > agent.cfg.kl_threshold:
                        break

                    # ── GR-faithful per-mini-batch entropy weight ──────────
                    sw = sampled_sigma[0].item()   # first element (scalar), mirrors GR sigma_grad_flg[0]
                    entropy_scale = base_entropy_scale * (1.0 - sw) + _ENTROPY_FLIP_SCALE * sw
                    entropy_loss = -entropy_scale * agent.policy.get_entropy(role="policy").mean()

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - agent.cfg.ratio_clip, 1.0 + agent.cfg.ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _ = agent.value.act(inputs, role="value")
                    if agent.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-agent.cfg.value_clip, max=agent.cfg.value_clip,
                        )
                    value_loss = agent.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                agent.optimizer.zero_grad()
                agent.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    agent.policy.reduce_parameters()
                    if agent.policy is not agent.value:
                        agent.value.reduce_parameters()

                if agent.cfg.grad_norm_clip > 0:
                    agent.scaler.unscale_(agent.optimizer)
                    if agent.policy is agent.value:
                        nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.cfg.grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(agent.policy.parameters(), agent.value.parameters()),
                            agent.cfg.grad_norm_clip,
                        )

                agent.scaler.step(agent.optimizer)
                agent.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                cumulative_entropy_loss += entropy_loss.item()

            if agent.scheduler is not None:
                if isinstance(agent.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=agent.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    agent.scheduler.step(kl.item())
                else:
                    agent.scheduler.step()

        n = agent.cfg.learning_epochs * agent.cfg.mini_batches
        agent.track_data("Loss / Policy loss", cumulative_policy_loss / n)
        agent.track_data("Loss / Value loss", cumulative_value_loss / n)
        agent.track_data("Loss / Entropy loss", cumulative_entropy_loss / n)
        agent.track_data("Policy / Standard deviation",
                         agent.policy.distribution(role="policy").stddev.mean().item())
        if agent.scheduler is not None:
            agent.track_data("Policy / Learning rate", agent.scheduler.get_last_lr()[0])

    agent.update = _update_with_per_batch_entropy
    print(f"[entropy_flip] GR-faithful: base={base_entropy_scale}, flip={_ENTROPY_FLIP_SCALE}")


# Parameter names to keep at their YAML-initialized values when loading a pretrain
# checkpoint into the train policy (mirrors TJ/rl_games tools/reset_epoch.py, which
# pops `a2c_network.sigma` so train starts with fresh exploration noise).
# `log_std_mass` and `mu_mass` are absent from the pretrain ckpt, so they are
# already freshly initialized via the "not in checkpoint" branch below.
_PRETRAIN_LOAD_SKIP_KEYS = {"log_std_parameter"}

# Observation-preprocessor transfer (pretrain → train). The loaded policy is calibrated to
# scaler-NORMALIZED obs; leaving the scaler at fresh identity feeds it RAW obs (foot forces ~100s N,
# keypoints ~m) → it diverges from step 0. So transfer the pretrain running stats, but IDENTITY the
# degenerate dims (near-zero pretrain variance = obs that are zero/constant in the no-object pretrain:
# object velocity, delta-object, fingertip forces) to avoid divide-by-~0 explosion, and reset the
# count so the scaler keeps adapting to the TRAIN distribution.
_OBS_PP_VAR_FLOOR = 1.0e-4        # pretrain variance below this = degenerate dim → identity
_OBS_PP_TRANSFER_COUNT = 1.0e4    # reset current_count so transferred stats are a prior, not frozen


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
        if key == "value":
            # Do NOT transfer the value net. The value function is reward-regime-specific (pretrain has
            # no object / different reward), and loading it while the value_preprocessor is RESET creates
            # a scale mismatch: the loaded net outputs pretrain-normalized values (~unit) but the reset
            # preprocessor feeds RAW train returns (~8) → value loss ≈ (8−1)² and the KL-adaptive LR
            # crashes trying to contain it → divergence. Pretrain PROVED a FRESH value net + fresh
            # preprocessor is stable (value loss 1.2→0.02). So keep the value net at its random init and
            # let it co-adapt with the (reset) preprocessor, exactly like pretrain. Only the POLICY
            # transfers (with the obs-scaler transfer below).
            print("[partial load] value: SKIPPED — value net kept fresh (co-adapts with reset preprocessor; "
                  "avoids the loaded-net-vs-reset-preprocessor scale mismatch → value-loss explosion)")
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
    # OBSERVATION preprocessor: TRANSFER the pretrain running stats (see _OBS_PP_* above). Without this
    # the loaded policy is fed RAW obs and diverges immediately (confirmed: with the scaler its wrist
    # tracking stays ~0.36 rad; at identity it blows past the 0.75 termination in ~3 steps). Degenerate
    # dims → identity; count reset so it keeps adapting.
    pp_sd = data.get("observation_preprocessor")
    pp = getattr(agent, "_observation_preprocessor", None)
    if pp_sd is not None and pp is not None and hasattr(pp, "running_variance"):
        dev, dt = pp.running_variance.device, pp.running_variance.dtype
        rm = pp_sd["running_mean"].to(device=dev, dtype=dt)
        rv = pp_sd["running_variance"].to(device=dev, dtype=dt)
        if rm.shape == pp.running_mean.shape:
            degen = rv < _OBS_PP_VAR_FLOOR
            rm = torch.where(degen, torch.zeros_like(rm), rm)
            rv = torch.where(degen, torch.ones_like(rv), rv)
            pp.running_mean.copy_(rm)
            pp.running_variance.copy_(rv)
            pp.current_count.fill_(_OBS_PP_TRANSFER_COUNT)
            print(f"[partial load] observation_preprocessor: TRANSFERRED "
                  f"(identity on {int(degen.sum())}/{degen.numel()} degenerate dims, count→{_OBS_PP_TRANSFER_COUNT:g})")
        else:
            print(f"[partial load] observation_preprocessor: shape {tuple(rm.shape)} != "
                  f"{tuple(pp.running_mean.shape)}, NOT transferred (fresh identity)")
    else:
        print("[partial load] observation_preprocessor: absent/identity, nothing to transfer")
    # VALUE preprocessor: still NOT transferred — train reward/return scale differs from pretrain, it
    # self-corrects, and value normalization is far less critical than the policy's obs normalization.
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
    # ParaHome clip selection (g1 loco-manip cfg uses clip_class/clip_name instead of dataset/traj).
    if args_cli.clip_class is not None and hasattr(env_cfg, "clip_class"):
        env_cfg.clip_class = args_cli.clip_class
    if args_cli.clip_name is not None and hasattr(env_cfg, "clip_name"):
        env_cfg.clip_name = args_cli.clip_name

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
    # G1+Shadow loco-manip (task id contains "Locomanip"; plain GaussianMixin policy — NO
    # mass-as-action / entropy-flip patches, those are grasp-specific). RSI warm-start reuses
    # the same pretrain-cache mechanism as the grasp RSI variant.
    _is_g1_train = "Locomanip" in (args_cli.task or "") and "Pretrain" not in (args_cli.task or "")
    _is_g1_pretrain = "Locomanip" in (args_cli.task or "") and "Pretrain" in (args_cli.task or "")
    if _is_grasp_train:
        _patch_mass_policy(runner.agent, agent_cfg["models"]["policy"], agent_cfg["agent"]["learning_rate"])
        _patch_entropy_flip(runner.agent, env, agent_cfg["agent"]["entropy_loss_scale"])

    # [ROLLBACK MARKER: failure-sigma] env 가 켰을 때만. 관측 마지막 열(beta)로 sampling sigma 조절.
    if _is_g1_train and bool(getattr(env_cfg, "failure_sigma", False)):
        _patch_failure_sigma(runner.agent, int(env_cfg.action_space),
                             str(getattr(env_cfg, "failure_sigma_dims", "all")))

    # Manual env-info → Tensorboard (applies to BOTH train and pretrain — trainer's
    # auto-prefix "Info / " is disabled in main(), so this is the sole logging path
    # for env extras['log'] keys like "Error /", "Episode_Reward /", "Curriculum /").
    if _is_grasp_train or _is_grasp_pretrain or _is_g1_train or _is_g1_pretrain:
        _patch_env_info_log(runner.agent)

    # g1 uses skrl's native PPO update, which computes the mean approx-KL for the KLAdaptiveLR
    # scheduler but never logs it. Track it as 'Policy / KL' — and, with _USE_ANALYTIC_KL, replace
    # the k3 estimator that feeds the scheduler with the closed-form Gaussian KL (rl_games/TJ
    # convention), because k3 is exponential in the 100-D joint log-ratio and saturates the
    # controller (see _patch_analytic_kl).
    if _is_g1_train or _is_g1_pretrain:
        # ORDER MATTERS: _patch_dual_clip REPLACES agent.update, so it must come first — the
        # _patch_analytic_kl wrapper captures whatever agent.update is at wrap time and must end up
        # wrapping the forked version. _patch_grad_norm_logging patches the module-level
        # nn.utils.clip_grad_norm_ (resolved at call time) so it is order-independent.
        if _DUAL_CLIP_C is not None:
            _patch_dual_clip(runner.agent, c=_DUAL_CLIP_C)
        if _USE_ANALYTIC_KL:
            _patch_grad_norm_logging(runner.agent)   # must precede: its accumulator is flushed below
            _patch_analytic_kl(runner.agent)
        else:
            _patch_kl_logging(runner.agent)

    # For the grasp pretrain task: freeze log_std_parameter at YAML initial value (σ=0.22)
    # to mirror TJ's `frozen_sigma: True` in rl_games_ppo_cfg_pretrain.yaml. The policy
    # learns only mu during pretrain; exploration scale stays constant.
    if _is_grasp_pretrain or _is_g1_pretrain:
        _policy = runner.agent.models["policy"]
        if hasattr(_policy, "log_std_parameter"):
            _policy.log_std_parameter.requires_grad_(False)
            _ls = _policy.log_std_parameter.detach()
            print(f"[INFO] Pretrain: log_std_parameter frozen, mean={_ls.mean().item():.4f} "
                  f"(σ_mean={_ls.exp().mean().item():.4f}, n={_ls.numel()}).")

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        if _is_grasp_train or _is_g1_train:
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

    # ── PRETRAIN-CACHE WARM-START [ROLLBACK MARKER: pretrain-cache-warmstart] ──
    # For RSI train tasks: if a `pretrain_state_cache.npz` sits next to the loaded
    # checkpoint (saved at the end of pretrain — see post-run block below), hand it to
    # the env so the first N control steps roll out from the pretrain state cache.
    # No-op for envs without set_pretrain_cache (all non-RSI tasks).
    if resume_path and (_is_grasp_train or _is_g1_train) and hasattr(env.unwrapped, "set_pretrain_cache"):
        _cache_sibling = Path(resume_path).resolve().parent / "pretrain_state_cache.npz"
        if _cache_sibling.is_file():
            env.unwrapped.set_pretrain_cache(str(_cache_sibling))
        else:
            print(f"[pretrain-cache-warmstart] no cache found at {_cache_sibling}; vanilla start.")
    # ── END PRETRAIN-CACHE WARM-START ─────────────────────────────────────────

    #####################
    # Custom callback to log success rate from environment's extras during training
    #####################
    
    # get the agent for logging success rate during training
    agent = runner.agent
    
    # log success rate from environment's extras during training
    # skrl 2.0.0: post_interaction is keyword-only.
    def log_success_rate(*, timestep, timesteps):
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

    # [ROLLBACK MARKER: track-harvest] deterministic actions for the tracking-harvest envs.
    # RePHO harvests the replacement kinematics from a SEPARATE `--test --num_envs 1` process, i.e. a
    # single rollout with NO exploration noise (scripts/train_dual_forward_val.sh). Noise matters here
    # more than anywhere else: a trajectory that only survived because the noise happened to fall well
    # is not one the policy can reproduce, and it would become the thing the reward is measured
    # against. We cannot spawn a second Isaac Lab process cheaply, but we can take the distribution
    # MEAN for the handful of envs that are allowed to promote.
    # The log_prob is recomputed for the action actually taken, so the PPO ratio stays consistent;
    # what remains is that those envs are behaviourally off-policy. With track_harvest_envs at 2 of
    # 2048 that is ~0.1% of the batch.
    _harv = getattr(env.unwrapped, "_harvest", None)
    if _harv is not None and bool(_harv.any()):
        _agent_act = agent.act

        def _act_deterministic_for_harvest(states, timestep, timesteps):
            actions, outputs = _agent_act(states, timestep, timesteps)
            mean = outputs.get("mean_actions") if isinstance(outputs, dict) else None
            if mean is not None:
                actions = torch.where(_harv.unsqueeze(-1), mean, actions)
                dist = agent.policy.distribution(role="policy")
                if dist is not None:
                    lp = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                    outputs["log_prob"] = torch.where(_harv.unsqueeze(-1), lp, outputs["log_prob"])
                    agent._current_log_prob = outputs["log_prob"]
            return actions, outputs

        agent.act = _act_deterministic_for_harvest
        print(f"[track-harvest] deterministic actions for {int(_harv.sum())}/{env.unwrapped.num_envs} "
              f"envs (tracking-target harvest)")

    # run training
    runner.run()

    training_time = round(time.time() - start_time, 2)
    print(f"Training time: {training_time} seconds")

    # ── PRETRAIN-CACHE WARM-START [ROLLBACK MARKER: pretrain-cache-warmstart] ──
    # After an RSI *pretrain* run, dump the per-frame state cache so the subsequent
    # RSI *train* run can roll out from it (warm-start). Saved into the run log dir
    # (mirrors how skrl checkpoints land in logs); the benchmark script copies it next
    # to pretrain.pt, and train.py loads it as a checkpoint sibling. Also written to
    # the data-tree checkpoint dir below (when sequence info is available) for manual
    # runs that point --checkpoint straight at the data tree.
    _pretrain_cache_payload = None
    if (
        (_is_grasp_pretrain and "Rsi" in (args_cli.task or "")) or _is_g1_pretrain
    ) and hasattr(env.unwrapped, "_state_cache"):
        import numpy as _np

        _ue = env.unwrapped
        # g1 dumps the 209-D remap (drops the object block the pretrain env never uses) so it
        # matches the train env's _reset_idx pretrain-branch offsets; grasp dumps its raw cache.
        if _is_g1_pretrain and hasattr(_ue, "dump_pretrain_cache_209"):
            _state_cache_np = _ue.dump_pretrain_cache_209().detach().cpu().numpy()
        else:
            _state_cache_np = _ue._state_cache.detach().cpu().numpy()
        _pretrain_cache_payload = dict(
            state_cache=_state_cache_np,
            init_flg=_ue._init_flg.detach().cpu().numpy(),
            reached_frame=int(_ue._reached_frame),
            # g1 env exposes _ref_len (not _max_traj_len); keep the key for load-time validation.
            max_traj_len=int(getattr(_ue, "_max_traj_len", getattr(_ue, "_ref_len", 0))),
            trajectory_task=str(getattr(env_cfg, "trajectory_task", "")),
        )
        # Save into the run log dir AND its checkpoints subdir — skrl writes checkpoints under
        # log_dir/checkpoints/, so the checkpoints-dir copy lets a manual train run whose
        # --checkpoint points there find the sibling npz (g1 has no benchmark-copy script yet).
        for _dst in (log_dir, os.path.join(log_dir, "checkpoints")):
            os.makedirs(_dst, exist_ok=True)
            _log_cache_path = os.path.join(_dst, "pretrain_state_cache.npz")
            _np.savez(_log_cache_path, **_pretrain_cache_payload)
            print(f"[pretrain-cache-warmstart] saved pretrain state cache → {_log_cache_path}")
    # ── END PRETRAIN-CACHE WARM-START ─────────────────────────────────────────

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
        # Select output tree by task name — keeps Shadow Hand results separate from
        # FFW-SH5 native-hand results (mirrors train_sequences_shadow.sh). The RSI
        # warm-start variant gets its own tree so it doesn't clobber plain-Shadow runs.
        _task_name = args_cli.task or ""
        if "Rsi" in _task_name:
            _robot_tree = "ffw_shadow_rsi"
        elif "Shadow" in _task_name:
            _robot_tree = "ffw_shadow"
        else:
            _robot_tree = "ffw_sh5"
        _ckpt_dir = (
            _data_root / _robot_tree / "right"
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

        # ── PRETRAIN-CACHE WARM-START [ROLLBACK MARKER: pretrain-cache-warmstart] ──
        # Also drop the RSI pretrain cache into the data-tree checkpoint dir so a manual
        # train run with --checkpoint <_ckpt_dir>/pretrain.pt finds the sibling npz.
        if _pretrain_cache_payload is not None:
            import numpy as _np

            _ckpt_cache_path = _ckpt_dir / "pretrain_state_cache.npz"
            _np.savez(_ckpt_cache_path, **_pretrain_cache_payload)
            print(f"[pretrain-cache-warmstart] saved pretrain state cache → {_ckpt_cache_path}")
        # ── END PRETRAIN-CACHE WARM-START ─────────────────────────────────────────

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
