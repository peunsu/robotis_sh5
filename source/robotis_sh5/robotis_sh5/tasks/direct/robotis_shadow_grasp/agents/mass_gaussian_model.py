"""Custom skrl Gaussian policy for mass-as-optimizable-parameter (Section 3.2, MassDexMimic).

Per paper Section 3.2: mass is NOT a direct output of the policy network.
The network π outputs only 28D (joint targets). Mass distribution parameters
mu_mass and log_std_mass are separate scalar nn.Parameters optimized independently.

  mu_mass      – initial mean   = -0.25  (corresponds to ≈ 0.4 × mmax after unnorm)
  log_std_mass – initial log σ  = -1.25

Only mu_mass lives in the high-LR optimizer group (33.333×); log_std_mass uses normal LR.

During rollout, the sampled mass value is **fixed within each episode** and only
re-drawn when the episode terminates.  The patched agent calls
``policy.update_mass_terminated(done)`` inside ``record_transition`` so the model
knows which envs just ended before the next ``act()`` call.

Key design matching GR (rl_games/models.py ModelA2CContinuousLogStd):
  - Per-env caches store mu_mass and log_std_mass AT EPISODE START (fixed_mass_mu /
    fixed_mass_logstd in GR).  These cached values — not the current global params —
    are used when computing old log_prob for not-done envs during rollout.
  - During training, GaussianMixin evaluates log_prob against the CURRENT global params.
  - This asymmetry makes the PPO ratio for mass non-trivial (r ≠ 1) when sigma drifts
    from its episode-start value, so PPO clipping naturally restrains log_std_mass growth.
  - Mass dim is excluded from the entropy loss (get_entropy returns joint dims only),
    matching GR models.py entropy = distr.entropy()[:,1:].sum(dim=-1).

``net_container`` (not ``net``) is used as the attribute name to stay byte-compatible
with skrl's model_instantiators checkpoint convention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from skrl.models.torch import GaussianMixin, Model


class MassDexMimicPolicy(GaussianMixin, Model):
    """Gaussian actor where the policy network outputs joint dims only; mass uses separate params.

    Network output  (27D):  joint means (20 fingers + 7 arm — lift is excluded, held fixed by env)
    Joint log-std   (27D):  log_std_parameter (learnable, not a network output)
    Mass dim        (1D):   mean = mu_mass, log-std = log_std_mass (both nn.Parameters)

    The full 28D action is assembled as [joint_actions | mass_action] at call time.

    The ``network`` parameter accepts the YAML ``network`` list-of-dicts directly:
      network:
        - name: net
          input: OBSERVATIONS
          layers: [1024, 1024, 512, 512]
          activations: elu   # string or list
    When omitted, the default [1024, 1024, 512, 512] architecture is used.
    """

    _ACT_MAP = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "selu": nn.SELU,
    }

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = -1.5141,
        mass_mu_init: float = -0.25,
        mass_log_std_init: float = -1.25,
        network=None,
        **kwargs,  # absorb 'class', 'output', and any future YAML fields
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        obs_size = observation_space.shape[0]
        act_size = action_space.shape[0]  # 28 (27 joints + 1 mass)
        n_joint = act_size - 1             # 27 — network outputs joint dims only (lift excluded)

        # Network — named ``net_container`` to match skrl's model_instantiators convention
        # so existing checkpoint files (saved by the standard Runner) can be loaded directly.
        # Outputs n_joint (27D) only; mass is NOT a network output (paper Section 3.2).
        if network is not None:
            self.net_container = self._build_net(obs_size, n_joint, network)
        else:
            self.net_container = nn.Sequential(
                nn.Linear(obs_size, 1024), nn.ELU(),
                nn.Linear(1024, 1024),     nn.ELU(),
                nn.Linear(1024, 512),      nn.ELU(),
                nn.Linear(512, 512),       nn.ELU(),
                nn.Linear(512, n_joint),
            )

        # Joint log-std (28D) — ``log_std_parameter`` matches the skrl instantiator key name.
        self.log_std_parameter = nn.Parameter(initial_log_std * torch.ones(n_joint))

        # Mass distribution parameters (paper Section 3.2):
        #   μm = -0.25  →  0.4 × mmax after unnorm;  log σm = -1.25
        self.mu_mass = nn.Parameter(torch.tensor([mass_mu_init]))
        self.log_std_mass = nn.Parameter(torch.tensor([mass_log_std_init]))

        # Per-env rollout cache: mass action and episode-start distribution params.
        # _cache_mu_mass / _cache_log_std_mass mirror GR's fixed_mass_mu / fixed_mass_logstd:
        # they are frozen to the values of mu_mass / log_std_mass at the start of each
        # episode and used (instead of the current global params) when computing old log_prob
        # for not-done envs.  This makes the PPO ratio for mass non-trivial and lets PPO
        # clipping naturally prevent log_std_mass from drifting upward.
        self._prev_terminated: torch.Tensor | None = None   # (num_envs,) bool
        self._cache_action: torch.Tensor | None = None      # (num_envs,) — sampled mass
        self._cache_mu_mass: torch.Tensor | None = None     # (num_envs,) — episode-start μ
        self._cache_log_std_mass: torch.Tensor | None = None  # (num_envs,) — episode-start log σ

    # ------------------------------------------------------------------
    # Network builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_net(obs_size: int, act_size: int, network_cfg) -> nn.Sequential:
        """Build nn.Sequential from the YAML ``network`` list-of-dicts.

        Supports a single-block config:
          network:
            - name: net
              input: OBSERVATIONS
              layers: [1024, 1024, 512, 512]
              activations: elu       # string OR list of strings
        ``layers`` defines hidden sizes; an output Linear(last_hidden, act_size) is appended.
        """
        cfg = network_cfg[0] if isinstance(network_cfg, list) else network_cfg
        layer_sizes = cfg.get("layers", [1024, 1024, 512, 512])
        activations = cfg.get("activations", "elu")

        if isinstance(activations, str):
            act_list = [activations] * len(layer_sizes)
        else:
            act_list = list(activations)

        modules = []
        in_size = obs_size
        for hidden, act_name in zip(layer_sizes, act_list):
            modules.append(nn.Linear(in_size, hidden))
            act_cls = MassDexMimicPolicy._ACT_MAP.get(act_name.lower(), nn.ELU)
            modules.append(act_cls())
            in_size = hidden
        modules.append(nn.Linear(in_size, act_size))
        return nn.Sequential(*modules)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _ensure_cache(self, num_envs: int, device: torch.device) -> None:
        """Lazily allocate all per-env caches; force-resample all envs on first call."""
        if self._cache_action is None or self._cache_action.shape[0] != num_envs:
            log_std_m = self._clamp_log_std(self.log_std_mass)
            init = Normal(self.mu_mass, log_std_m.exp()).sample((num_envs,)).squeeze(-1)
            self._cache_action = init.detach().to(device)
            # Episode-start distribution caches (GR: fixed_mass_mu / fixed_mass_logstd).
            mu_val = self.mu_mass.detach().item()
            lsm_val = log_std_m.detach().item()
            self._cache_mu_mass = torch.full((num_envs,), mu_val, device=device)
            self._cache_log_std_mass = torch.full((num_envs,), lsm_val, device=device)
            # Mark all envs as "just terminated" so they all draw a fresh sample.
            self._prev_terminated = torch.ones(num_envs, device=device, dtype=torch.bool)

    def _clamp_log_std(self, x: torch.Tensor) -> torch.Tensor:
        if self._g_clip_log_std:
            return torch.clamp(x, self._g_log_std_min, self._g_log_std_max)
        return x

    def update_mass_terminated(self, terminated: torch.Tensor) -> None:
        """Store which envs just ended; called by the patched record_transition.

        The NEXT act() call resamples mass only for those envs.
        terminated: (num_envs,) or (num_envs, 1) bool tensor.
        """
        self._prev_terminated = terminated.squeeze(-1).bool().to(self.device)

    # ------------------------------------------------------------------
    # GaussianMixin interface
    # ------------------------------------------------------------------

    def get_entropy(self, role=""):
        """Entropy over joint dims only — mass dim excluded (matches GR models.py [:,1:])."""
        if self._g_distribution is not None:
            return self._g_distribution.entropy()[..., :-1].sum(dim=-1)
        return super().get_entropy(role)

    def compute(self, inputs, role):
        """Called by GaussianMixin.act() during PPO training updates.

        Network outputs 27D (joints only). Mass dim is appended from mu_mass / log_std_mass.
        Returns (mean[28D], log_std[28D], {}).
        Training uses CURRENT global mu_mass / log_std_mass (no per-env cache replacement),
        which makes the PPO ratio for mass non-trivial vs. the rollout old log_prob.
        """
        B = inputs["states"].shape[0]
        net_out = self.net_container(inputs["states"])  # (B, 27)

        # Assemble 28D mean: [joint_mean(27) | mu_mass(1)]
        mean = torch.cat([net_out, self.mu_mass.expand(B, 1)], dim=-1)  # (B, 28)

        # Per-dim log-std: joint dims from log_std_parameter (27D), mass from log_std_mass.
        log_std_j = self.log_std_parameter.expand(B, -1)  # (B, 27)
        log_std_m = self.log_std_mass.expand(B, 1)        # (B, 1)
        log_std = torch.cat([log_std_j, log_std_m], dim=-1)  # (B, 29)

        return mean, log_std, {}

    def act(self, inputs, role):
        """Override: fix mass within rollout episodes; delegate training to parent.

        Rollout log_prob for mass uses PER-ENV EPISODE-START cached mu and log_std
        (not the current global params), mirroring GR's fixed_mass_mu / fixed_mass_logstd
        mechanism.  This creates a non-trivial PPO ratio that prevents log_std_mass from
        drifting upward freely.
        """
        # Training: GaussianMixin evaluates log_prob(taken_actions | current global params).
        if "taken_actions" in inputs:
            return super().act(inputs, role)

        # Rollout: fix mass per episode
        B = inputs["states"].shape[0]
        device = inputs["states"].device
        self._ensure_cache(B, device)

        # For envs that just ended their episode: resample mass action and update the
        # episode-start distribution cache with the CURRENT global params.
        # (GR: done_env branch — fixed_mass_mu/logstd updated with network output.)
        if self._prev_terminated is not None and self._prev_terminated.any():
            done = self._prev_terminated.bool()
            log_std_m_curr = self._clamp_log_std(self.log_std_mass)
            mu_val = self.mu_mass.detach().item()
            lsm_val = log_std_m_curr.detach().item()

            new_samples = Normal(
                self.mu_mass.detach().expand(B),
                log_std_m_curr.exp().detach().expand(B),
            ).sample()
            self._cache_action = self._cache_action.clone()
            self._cache_action[done] = new_samples[done].detach()

            # Cache episode-start mu / log_std for done envs (GR: fixed_mass_mu/logstd update).
            self._cache_mu_mass = self._cache_mu_mass.clone()
            self._cache_log_std_mass = self._cache_log_std_mass.clone()
            self._cache_mu_mass[done] = mu_val
            self._cache_log_std_mass[done] = lsm_val

        # Network forward: outputs joint dims only (27D — lift excluded).
        net_out = self.net_container(inputs["states"])  # (B, 27)
        joint_mean = net_out                            # (B, 27)

        log_std_j = self._clamp_log_std(self.log_std_parameter)  # (27,)
        dist_j = Normal(joint_mean, log_std_j.exp())
        joint_actions = dist_j.rsample()                # (B, 27)

        # Mass: use cached action (fixed for this episode).
        mass_action = self._cache_action                # (B,) — detached

        actions = torch.cat([joint_actions, mass_action.unsqueeze(-1)], dim=-1)

        # Old log_prob for mass: use PER-ENV EPISODE-START cached mu and log_std.
        # (GR: not_done_env branch uses fixed_mass_mu/logstd, done_env uses current output.)
        # This makes old_logprob diverge from the training-time new_logprob as sigma drifts,
        # so PPO clipping naturally constrains log_std_mass.
        dist_m = Normal(self._cache_mu_mass, self._cache_log_std_mass.exp())

        log_prob_j = dist_j.log_prob(joint_actions)    # (B, 27)
        log_prob_m = dist_m.log_prob(mass_action)       # (B,)
        log_prob = (log_prob_j.sum(dim=-1) + log_prob_m).unsqueeze(-1)  # (B, 1)

        # Update GaussianMixin internals for get_entropy() / get_log_std().
        log_std_m_curr = self._clamp_log_std(self.log_std_mass)
        mean_all = torch.cat([joint_mean, self.mu_mass.expand(B, 1)], dim=-1)
        log_std_all = torch.cat([log_std_j.expand(B, -1), log_std_m_curr.expand(B, 1)], dim=-1)
        self._g_distribution = Normal(mean_all, log_std_all.exp())
        self._g_log_std = log_std_all
        self._g_num_samples = B

        return actions, log_prob, {"mean_actions": mean_all}

    # ------------------------------------------------------------------
    # Optimizer group helpers
    # ------------------------------------------------------------------

    def mass_params(self):
        """Yield mass mean parameter for the high-LR optimizer group.

        Only mu_mass gets the 33.333× LR boost — log_std_mass stays in the normal
        group so the entropy bonus doesn't drive std to blow up during training.
        """
        yield self.mu_mass

    def non_mass_params(self):
        """Yield all parameters except mu_mass (normal LR group)."""
        mass_ids = {id(self.mu_mass)}
        for p in self.parameters():
            if id(p) not in mass_ids:
                yield p
