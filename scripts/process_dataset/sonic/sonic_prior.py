"""Frozen SONIC (GEAR universal-token) body prior — thin wrapper for Milestone 1.

Loads the released SONIC Actor (sonic_release/{config.yaml,last.pt}) as a FROZEN,
pure-torch module usable inside env_isaaclab (imports `gear_sonic`, no Isaac Sim needed
to build). SONIC drives the 29 G1 BODY DOF only (no hands, no root).

Pure-torch, so it can be smoke-tested without Isaac Sim:
    /home/peunsu/anaconda3/envs/env_isaaclab/bin/python scripts/process_dataset/sonic/sonic_prior.py --smoke

Verified facts baked in (see MEMORY sonic-residual-integration):
  * build = trl OnlineTrainerState shim + groot->gear_sonic string-replace + strip aux
    losses (avoids open3d) + custom_instantiate + load policy_state_dict (strict) + eval.
  * a_sonic (29) is in SONIC IsaacLab order (G1_ISAACLab_ORDER); target = SONIC_DEFAULT +
    SONIC_SCALE * a_sonic, SONIC_SCALE = 0.25*effort/stiffness (per-joint, wrist ~0.0745).
  * obs = {"actor_obs": (E,930), "tokenizer": (E,TOK)}; the 10-frame history lives in
    actor_obs (Actor keeps 1 frame). tokenizer = 12-term flat concat, encoder_index=[0,0,1]
    for SMPL, only the 3 smpl terms carry real values (rest zero-filled, dims must match).
"""

from __future__ import annotations

import io
import os

import torch

# gear_sonic is pip-installed (editable) in env_isaaclab.
from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER  # 29 body joint names, SONIC order

_SONIC_ROOT = "/home/peunsu/workspace/GR00T-WholeBodyControl"
DEFAULT_CONFIG = os.path.join(_SONIC_ROOT, "sonic_release", "config.yaml")
DEFAULT_CKPT = os.path.join(_SONIC_ROOT, "sonic_release", "last.pt")

# ---- proprioception layout (verified Phase-2): term-major, 10-frame oldest-first ----
PROPRIO_HIST = 10
PROPRIO_PER_FRAME = 93  # base_ang_vel3 + joint_pos_rel29 + joint_vel_rel29 + last_action29 + gravity3
ACTOR_OBS_DIM = PROPRIO_HIST * PROPRIO_PER_FRAME  # 930

# ---- tokenizer 12-term dims (VERIFIED: build_test.py passed strict=True load with these) ----
# NOTE: the 9 non-smpl terms are zero-filled at runtime; their exact dims only need to be
# self-consistent (the encoder-input Linear shapes that strict-load checks are satisfied).
TOK_DIMS = {
    "encoder_index": [3],
    "command_multi_future_nonflat": [10, 58],
    "command_z_multi_future_nonflat": [10, 32],
    "motion_anchor_ori_b_mf_nonflat": [10, 6],
    "command_multi_future_lower_body": [234],
    "vr_3point_local_target": [9],
    "vr_3point_local_orn_target": [18],
    "motion_anchor_ori_b": [6],
    "command_z": [32],
    "smpl_joints_multi_future_local_nonflat": [10, 72],
    "smpl_root_ori_b_multi_future": [10, 6],
    "joint_pos_multi_future_wrist_for_smpl": [10, 6],
}
TOK_NAMES = list(TOK_DIMS.keys())

# SONIC default standing pose (name-keyed, unlisted = 0.0), from g1.py G1_CYLINDER_MODEL_12_DEX_CFG.
_SONIC_DEFAULT_RULES = {
    "hip_pitch": -0.312,
    "knee": 0.669,
    "ankle_pitch": -0.363,
    "elbow": 0.6,
    "left_shoulder_roll": 0.2,
    "right_shoulder_roll": -0.2,
    "left_shoulder_pitch": 0.2,
    "right_shoulder_pitch": 0.2,
}


# SONIC per-joint action scale = 0.25*effort_limit/stiffness (name-keyed), from SONIC's gains
# (NATURAL_FREQ=62.83, zeta=2 -> stiffness=armature*w^2; groups from g1.py G1_CYLINDER_MODEL_12_DEX_CFG).
# Hardcoded (not imported from g1.py) so this module stays free of the isaaclab.actuators/carb import
# and builds in pure torch. Cross-check against g1.G1_MODEL_12_ACTION_SCALE inside Isaac Sim if desired.
_SONIC_SCALE_RULES = {
    "hip_pitch": 0.3507, "hip_roll": 0.3507, "knee": 0.3507,   # 7520_22: k99.10 eff139
    "hip_yaw": 0.5475, "waist_yaw": 0.5475,                     # 7520_14: k40.18 eff88
    "ankle_pitch": 0.4386, "ankle_roll": 0.4386,               # 2x5020:  k28.50 eff50
    "waist_roll": 0.4386, "waist_pitch": 0.4386,
    "shoulder_pitch": 0.4386, "shoulder_roll": 0.4386,          # 5020:    k14.25 eff25
    "shoulder_yaw": 0.4386, "elbow": 0.4386, "wrist_roll": 0.4386,
    "wrist_pitch": 0.0745, "wrist_yaw": 0.0745,                 # 4010:    k16.78 eff5
}


def _resolve_by_substring(rules: dict, joint_name: str, default: float = 0.0) -> float:
    # longest-substring match ("left_shoulder_roll_joint" -> "shoulder_roll", not "roll")
    best = (0, default)
    for key, val in rules.items():
        if key in joint_name and len(key) > best[0]:
            best = (len(key), val)
    return best[1]


def sonic_default_vector(device="cpu") -> torch.Tensor:
    """(29,) SONIC default pose in G1_ISAACLab_ORDER."""
    return torch.tensor([_resolve_by_substring(_SONIC_DEFAULT_RULES, n) for n in G1_ISAACLab_ORDER],
                        dtype=torch.float32, device=device)


def sonic_scale_vector(device="cpu") -> torch.Tensor:
    """(29,) SONIC per-joint action scale (0.25*effort/stiffness) in G1_ISAACLab_ORDER."""
    vals = [_resolve_by_substring(_SONIC_SCALE_RULES, n, default=None) for n in G1_ISAACLab_ORDER]
    assert None not in vals, f"unmapped joint scale: {[n for n, v in zip(G1_ISAACLab_ORDER, vals) if v is None]}"
    return torch.tensor(vals, dtype=torch.float32, device=device)


def crosscheck_scale_against_g1():
    """Optional: verify _SONIC_SCALE_RULES matches g1.G1_MODEL_12_ACTION_SCALE (needs Isaac Sim/carb)."""
    from gear_sonic.envs.manager_env.robots.g1 import G1_MODEL_12_ACTION_SCALE

    ours = sonic_scale_vector()
    ref = torch.tensor([float(G1_MODEL_12_ACTION_SCALE[n]) for n in G1_ISAACLab_ORDER])
    maxdiff = (ours - ref).abs().max().item()
    print(f"[crosscheck] max |scale - g1.G1_MODEL_12_ACTION_SCALE| = {maxdiff:.5f}")
    return maxdiff


def build_body_perm(robot_joint_names: list[str], device="cpu") -> torch.Tensor:
    """perm (29,) with perm[k] = index in `robot_joint_names` of SONIC body joint k.

    Use to GATHER our robot's per-joint quantities into SONIC order:
        q_sonic = robot.data.joint_pos[:, perm]           # (E,29) SONIC order
    and to SCATTER a target back (set_joint_position_target(target29, joint_ids=perm.tolist())).
    """
    perm = [robot_joint_names.index(n) for n in G1_ISAACLab_ORDER]
    return torch.tensor(perm, dtype=torch.long, device=device)


def _make_env_config():
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "obs": {"group_obs_dims": {"tokenizer": {k: list(v) for k, v in TOK_DIMS.items()}},
                "group_obs_names": {"tokenizer": TOK_NAMES}},
        "robot": {"actions_dim": 29, "algo_obs_dim_dict": {"actor_obs": ACTOR_OBS_DIM}},
    })


def build_sonic(config_path: str = DEFAULT_CONFIG, ckpt_path: str = DEFAULT_CKPT, device: str = "cuda:0"):
    """Build + load the frozen SONIC Actor. Returns the Actor (eval, requires_grad_(False))."""
    from omegaconf import OmegaConf
    from gear_sonic.trl.utils import common as trl_common

    # (a) shim the moved trl pickle path so torch.load(last.pt) can unpickle its "state".
    try:
        import trl.trainer.utils as _tu
        from trl.experimental.ppo.ppo_trainer import OnlineTrainerState as _OTS
        _tu.OnlineTrainerState = _OTS
        try:
            from trl.experimental.ppo.ppo_trainer import exact_div as _ed
            _tu.exact_div = _ed
        except Exception:
            pass
    except Exception:
        pass

    # (b) load config, remap legacy groot.rl.* -> gear_sonic.* (verbatim from eval_agent_trl.py).
    raw = open(config_path).read()
    for a, b in [
        ("groot.rl.trl.", "gear_sonic.trl."),
        ("groot.rl.envs.", "gear_sonic.envs."),
        ("groot.rl.utils.", "gear_sonic.utils."),
        ("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module."),
        ("groot.rl.agents.", "gear_sonic.trl."),
        ("groot/rl/data/", "gear_sonic/data/"),
    ]:
        raw = raw.replace(a, b)
    cfg = OmegaConf.load(io.StringIO(raw))

    # (c) drop aux losses so token_losses/motion_lib/open3d are never imported/built.
    OmegaConf.set_struct(cfg, False)
    cfg.algo.config.actor.has_aux_loss = False
    cfg.algo.config.actor.backbone.aux_loss_func = {}
    cfg.algo.config.actor.backbone.aux_loss_coef = {}
    cfg.algo.config.actor.backbone.reencode_smpl_g1_recon = False

    # (d) instantiate exactly like eval_agent_trl.py:401.
    actor = trl_common.custom_instantiate(
        cfg.algo.config.actor,
        env_config=_make_env_config(),
        algo_config=cfg.algo.config,
        module_dim_dict={},
        backbone_kwargs={},
        _resolve=False,
    ).to(device)

    # (e) load frozen weights.
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck.get("actor_model_state_dict", ck.get("policy_state_dict"))
    have = actor.state_dict()
    if "std" in have and "std" not in sd and "log_std" in sd:
        sd = dict(sd)
        sd["std"] = torch.exp(sd.pop("log_std"))
    elif "log_std" in have and "log_std" not in sd and "std" in sd:
        sd = dict(sd)
        sd["log_std"] = torch.log(sd.pop("std"))
    actor.load_state_dict(sd, strict=True)

    # (f) freeze + restrict to the dynamics decoder (skip g1_kin recon every step).
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    try:
        actor.actor_module._active_decoders = {"g1_dyn"}
    except Exception:
        pass
    return actor


def tokenizer_layout(actor) -> tuple[dict, int]:
    """Return ({name: (start, end, dims)}, total_dim) read back from the built module — no hardcoding."""
    specs = actor.actor_module.tokenizer_obs_specs
    layout = {}
    for entry in specs:
        # entry is (name, start, end, dims) per Phase-2; be tolerant of ordering.
        name = entry[0]
        start, end, dims = entry[1], entry[2], entry[3]
        layout[name] = (int(start), int(end), tuple(dims) if hasattr(dims, "__len__") else (int(dims),))
    total = int(getattr(actor.actor_module, "tokenizer_obs_total_dim", max(e[1] for e in layout.values())))
    return layout, total


@torch.no_grad()
def act(actor, actor_obs: torch.Tensor, tokenizer: torch.Tensor) -> torch.Tensor:
    """actor_obs (E,930), tokenizer (E,TOK) -> a_sonic (E,29) deterministic, SONIC order, RAW."""
    actor.init_rollout()
    a = actor.act_inference({"actor_obs": actor_obs, "tokenizer": tokenizer})
    return a


# ---- latent-residual (GRAIL-style) split API: PRE-quantization, stateless -----------------
# GRAIL Eq.6: a_body = G(z + lambda*dz), residual added to the encoder latent BEFORE FSQ.
# Used by the SonicResidual env: policy outputs z_res (64); env does encode -> +lambda*z_res -> FSQ -> decode.
@torch.no_grad()
def encode_latent(actor, tokenizer: torch.Tensor) -> torch.Tensor:
    """(E,TOK) -> PRE-quantization smpl latent (E, max_num_tokens, token_dim) = (E,2,32)."""
    m = actor.actor_module
    parsed = m.parse_tokenizer_obs({"tokenizer": tokenizer.unsqueeze(1)})   # terms (E,1,*dims)
    latent = m.encode("smpl", parsed, encoder_mask=None, no_quantization=True)  # (E,1,2,32) or (E,2,32)
    return latent


@torch.no_grad()
def residual_decode(actor, latent: torch.Tensor, z_res: torch.Tensor, actor_obs: torch.Tensor,
                    scale: float) -> torch.Tensor:
    """PRE-quant residual then frozen g1_dyn decode.
    latent (E,...,2,32), z_res (E,64), actor_obs (E,930) -> a_body (E,29)."""
    m = actor.actor_module
    zr = z_res.view(*latent.shape[:-2], m.max_num_tokens, m.token_dim)   # (E,...,2,32)
    latent_prime = latent + scale * zr
    q_tok, _ = m.quantizer(latent_prime)                                # FSQ on residual-perturbed latent
    # forward_with_external_tokens wants proprio with a seq dim and tokens (E,2,32)
    if q_tok.dim() == 4:            # (E,1,2,32) -> (E,2,32)
        q_tok = q_tok.squeeze(1)
    a = m.forward_with_external_tokens({"actor_obs": actor_obs.unsqueeze(1)}, external_tokens=q_tok)
    return a                                                            # (E,29)


@torch.no_grad()
def act_residual(actor, actor_obs: torch.Tensor, tokenizer: torch.Tensor, z_res: torch.Tensor,
                 scale: float = 0.1) -> torch.Tensor:
    """Full body path with a latent residual (pre-quantization, GRAIL-style)."""
    latent = encode_latent(actor, tokenizer)
    return residual_decode(actor, latent, z_res, actor_obs, scale)


def _smoke():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={dev}")
    actor = build_sonic(device=dev)
    print("[smoke] BUILD OK:", type(actor).__name__)
    print("  num_actions:", getattr(actor, "num_actions", "?"))
    print("  encoders:", list(actor.actor_module.encoders.keys()))
    print("  decoders:", list(actor.actor_module.decoders.keys()))
    layout, TOK = tokenizer_layout(actor)
    print("  TOK total:", TOK)
    for k in ["encoder_index", "smpl_joints_multi_future_local_nonflat",
              "smpl_root_ori_b_multi_future", "joint_pos_multi_future_wrist_for_smpl"]:
        print("   ", k, "->", layout.get(k))
    print("  SONIC_DEFAULT[:6]:", sonic_default_vector()[:6].tolist())
    print("  SONIC_SCALE[:6]:", [round(x, 4) for x in sonic_scale_vector()[:6].tolist()])
    print("  SONIC_SCALE wrist(25:29):", [round(x, 4) for x in sonic_scale_vector()[25:29].tolist()])

    E = 4
    torch.manual_seed(0)
    actor_obs = torch.randn(E, ACTOR_OBS_DIM, device=dev)
    tok = torch.zeros(E, TOK, device=dev)
    s, e, _ = layout["encoder_index"]
    tok[:, s:e] = torch.tensor([0.0, 0.0, 1.0], device=dev)  # smpl mode
    for name in ["smpl_joints_multi_future_local_nonflat", "smpl_root_ori_b_multi_future",
                 "joint_pos_multi_future_wrist_for_smpl"]:
        s, e, _ = layout[name]
        tok[:, s:e] = torch.randn(E, e - s, device=dev) * 0.3
    a = act(actor, actor_obs, tok)
    print("[smoke] fused a_sonic shape:", tuple(a.shape), "finite:", bool(torch.isfinite(a).all()))

    # latent-residual split checks (PRE-quantization)
    z0 = torch.zeros(E, 64, device=dev)
    a_split0 = act_residual(actor, actor_obs, tok, z0, scale=0.1)
    dmax = (a - a_split0).abs().max().item()
    print(f"[smoke] split(z_res=0) vs fused: max|diff|={dmax:.2e}  (expect ~0 -> split API correct)")
    zr = torch.randn(E, 64, device=dev)
    a_splitr = act_residual(actor, actor_obs, tok, zr, scale=0.1)
    print(f"[smoke] split(z_res=randn,scale=0.1): finite={bool(torch.isfinite(a_splitr).all())} "
          f"max|Δaction vs z=0|={ (a_splitr - a_split0).abs().max().item():.3f} (nonzero -> residual has effect)")
    print("[smoke] DONE")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        _smoke()
