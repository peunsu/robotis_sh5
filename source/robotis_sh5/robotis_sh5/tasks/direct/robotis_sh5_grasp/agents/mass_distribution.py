"""Optimizable mass distribution for episode-level mass-in-the-loop training.

Mirrors single-agent ``MassDexMimicPolicy`` behavior — μ_mass and log_σ_mass are
optimized via **PPO surrogate loss + ratio clipping** — but keeps mass OUT of
the agents' action spaces (per design constraint).

Mechanism (matches MassDexMimic paper):
  1. **Episode start (env._reset_idx)**: sample mass action ~ N(μ, exp(log_σ)),
     clamp to [-1, 1], affine-map to [mass_min, mass_max] kg, apply to PhysX.
     Cache the sampled action AND its log_prob (under params at sample time).
  2. **Per step (env._pre_physics_step)**: snapshot per-env cached
     `(action, log_prob_old)` — these are the values USED during this step.
  3. **PPO update (train_marl._update_mass)**: PPO surrogate with
     ratio = exp(log_prob_live(action; current μ/σ) - log_prob_old(action; cached μ/σ))
     and clip range = ratio_clip. Optimizers:
       μ_mass     : lr × mass_lr_scale  (e.g. 33.333× — single-agent convention)
       log_σ_mass : lr × 1.0            (no boost — keeps σ from blowing up)

Note: the log_prob_old we cache is constant within an episode (same params used
at sample time → same log_prob throughout). PPO mini-batch ratios become
non-trivial (≠ 1) as μ/σ drift over multiple gradient steps within one update.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class MassDistribution(nn.Module):
    """Scalar Gaussian over a per-episode mass action in [-1, 1] (affine → kg)."""

    def __init__(
        self,
        num_envs: int,
        mass_min: float,
        mass_max: float,
        mu_init: float = -0.25,
        log_std_init: float = -1.25,
        log_std_min: float = -5.0,
        log_std_max: float = 0.0,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        device = torch.device(device) if device is not None else torch.device("cpu")
        # Optimizable scalars (single-element tensors).
        self.mu_mass = nn.Parameter(torch.tensor(float(mu_init), device=device))
        self.log_std_mass = nn.Parameter(torch.tensor(float(log_std_init), device=device))

        self.mass_min = float(mass_min)
        self.mass_max = float(mass_max)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Per-env: the mass action SAMPLED at the env's most-recent reset, and the
        # log_prob of that action under the params that produced it (episode-start
        # μ/σ — frozen until next reset for this env). Both used for PPO ratio.
        self.register_buffer("current_mass_action", torch.full((num_envs,), float(mu_init), device=device))
        self.register_buffer("current_log_prob_old", torch.zeros((num_envs,), device=device))

    # ------------------------------------------------------------------
    # Read-only views (for tensorboard / inspection)
    # ------------------------------------------------------------------

    def _clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        return log_std.clamp(min=self.log_std_min, max=self.log_std_max)

    @property
    def mu_action(self) -> float:
        return float(self.mu_mass.detach().item())

    @property
    def std_action(self) -> float:
        return float(self._clamp_log_std(self.log_std_mass.detach()).exp().item())

    @property
    def mu_kg(self) -> float:
        """Mean mass in kg (mu_action mapped through affine [-1,1] → [min, max])."""
        t = (max(-1.0, min(1.0, self.mu_action)) + 1.0) / 2.0
        return self.mass_min + t * (self.mass_max - self.mass_min)

    @property
    def std_kg(self) -> float:
        """1-sigma in kg space ≈ std_action × (max - min) / 2."""
        return self.std_action * (self.mass_max - self.mass_min) / 2.0

    # ------------------------------------------------------------------
    # Sampling + physics-side conversion
    # ------------------------------------------------------------------

    def action_to_kg(self, action: torch.Tensor) -> torch.Tensor:
        """Map a [-1, 1] action to [mass_min, mass_max] kg (affine)."""
        t = (action.clamp(-1.0, 1.0) + 1.0) / 2.0
        return self.mass_min + t * (self.mass_max - self.mass_min)

    @torch.no_grad()
    def sample_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample new mass actions for the given envs.

        Caches the sampled action AND its log_prob under the params USED at sampling
        (which become the "old" log_prob for PPO ratio over the upcoming episode).

        Returns masses in kg (for applying to PhysX).
        """
        n = env_ids.numel()
        log_std = self._clamp_log_std(self.log_std_mass.detach())
        std = log_std.exp().clamp(min=1e-6)
        mu = self.mu_mass.detach()

        eps = torch.randn(n, device=self.mu_mass.device)
        sampled_action = (mu + std * eps).clamp(-1.0, 1.0)

        # log_prob under sampling params — constant for these envs until next reset.
        diff = (sampled_action - mu) / std
        log_prob_old = -0.5 * diff.pow(2) - log_std - 0.5 * math.log(2.0 * math.pi)

        self.current_mass_action[env_ids] = sampled_action
        self.current_log_prob_old[env_ids] = log_prob_old

        return self.action_to_kg(sampled_action)

    # ------------------------------------------------------------------
    # PPO surrogate loss helpers
    # ------------------------------------------------------------------

    def log_prob_live(self, actions: torch.Tensor) -> torch.Tensor:
        """Differentiable log_prob under CURRENT (live) μ/σ.

        Called during PPO mini-batch updates to compute ratio = exp(live - old).
        actions: any shape; returns same shape.
        """
        log_std = self._clamp_log_std(self.log_std_mass)
        std = log_std.exp().clamp(min=1e-6)
        diff = (actions - self.mu_mass) / std
        return -0.5 * diff.pow(2) - log_std - 0.5 * math.log(2.0 * math.pi)
