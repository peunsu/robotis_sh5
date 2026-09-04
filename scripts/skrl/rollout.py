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
parser.add_argument("--stochastic", action="store_true", default=False,
                    help="Sample actions from the policy Gaussian (default: deterministic, "
                    "matches rl_games player.deterministic=True convention used in TJ/GR).")
# Video recording
parser.add_argument("--video", action="store_true", default=False,
                    help="Record a video of the rollout into <output_dir>/videos/.")
parser.add_argument("--video_length", type=int, default=0,
                    help="Length of the recorded video (in steps). <=0 (default) auto-fits to the "
                         "full sequence length (env.max_episode_length), so the clip covers the "
                         "whole trajectory instead of a fixed duration.")
# Dataset / sequence overrides
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--object_id", type=str, default=None)
parser.add_argument("--trajectory_task", type=str, default=None)
parser.add_argument("--trajectory_data_id", type=int, default=None)
parser.add_argument("--clip_class", type=str, default=None, help="ParaHome clip class (g1 loco-manip).")
parser.add_argument("--clip_name", type=str, default=None, help="ParaHome clip name (g1 loco-manip).")
parser.add_argument("--zero_zres", action="store_true",
                    help="[zres-ablation] 정책 액션의 잠재 블록(앞 sonic_action_dim 차원)을 0 으로 "
                         "덮어써 순수 SONIC 프리어만 몸통을 구동합니다. 손 액션은 그대로 두어 과제가 "
                         "진행되게 합니다. z_res 섭동이 SONIC 출력 포화의 원인인지 가르는 진단.")
parser.add_argument("--dump_joints", action="store_true",
                    help="[joint-dump] env 0 의 PD 타겟/실측 관절 궤적을 joint_trace.npz 로 저장 "
                         "(bang-bang 진단용). _apply_action 이 쓰는 _residual_target 을 그대로 기록.")
parser.add_argument("--debug_vis", action="store_true", help="Draw reference-keypoint markers (needs a viewer / not --headless).")
# ── Recorded-video CAMERA ANGLE (video-only — never touches physics) ─────────────────────────
# The g1 sonic/locomanip env recomputes cfg.viewer.eye/lookat from cfg.viewer_{yaw,elev,look_obj,
# zoom} inside _load_reference_trajectories (g1_shadow_sonic_residual_env.py:361-389), which runs
# BEFORE super().__init__ consumes cfg.viewer. Writing those fields here is the ONLY supported way
# to change the recorded viewpoint: ViewportCameraController deep-copies cfg.viewer at construction
# (isaaclab/envs/ui/viewport_camera_controller.py:54), so post-init cfg edits are dead.
# Deliberately NOT done with an isaaclab.sensors.Camera: constructing one sets the PROCESS-GLOBAL
# carb flag /isaaclab/render/rtx_sensors (camera.py:123), which flips `is_rendering`
# (direct_rl_env.py:367) and moves the sole sim.render() from render() (:464) into the physics loop
# (:381) — i.e. it would silently retime the frame of the pane that is supposed to be unchanged.
parser.add_argument(
    "--cam_preset", type=str, default=None, choices=("current", "old"),
    help="Camera preset for the recorded video. 'old' = the pre-2026-07-22 view (yaw 45 deg / "
         "elev 0 / aim at the ROOT centroid), the formula still live in "
         "g1_shadow_locomanip_env.py:295-296. 'current' (and the default) is a deliberate NO-OP so "
         "the canonical pass always follows whatever the cfg says.")
parser.add_argument("--viewer_yaw", type=float, default=None,
                    help="Override cfg.viewer_yaw (deg azimuth). Applied after --cam_preset.")
parser.add_argument("--viewer_elev", type=float, default=None,
                    help="Override cfg.viewer_elev (deg). NOTE elev<=0 selects the env's "
                         "zoff=0.12*extent branch, not a literal 0 deg pitch.")
parser.add_argument("--viewer_look_obj", type=int, default=None, choices=(0, 1),
                    help="Override cfg.viewer_look_obj: 1 = aim at the object centroid, 0 = the root centroid.")
parser.add_argument("--viewer_zoom", type=float, default=None, help="Override cfg.viewer_zoom.")
parser.add_argument("--video_resolution", type=str, default=None,
                    help="Recording resolution as WxH (sets cfg.viewer.resolution). Default: keep the "
                         "cfg value (1280x720) so the mp4 stays comparable to archived videos. Lower it "
                         "to cut render time and RecordVideo's in-RAM frame list. Keep both dims even.")
parser.add_argument("--video_name_prefix", type=str, default="rl-video",
                    help="RecordVideo name_prefix -> <output_dir>/videos/<prefix>-step-0.mp4. Give a "
                         "second camera pass its own prefix so it does not overwrite the first pass.")
parser.add_argument("--metrics_name", type=str, default="metrics.csv",
                    help="Metrics CSV filename inside --output_dir. Keep the default for the canonical "
                         "pass; a video-only second camera pass should use e.g. metrics_camold.csv. "
                         "scripts/benchmark/evaluate.bash globs exactly '**/metrics.csv' (lines 31/51/99), "
                         "so the extra file is invisible to aggregation, and byte-comparing the two files "
                         "proves the two passes simulated the same rollout.")
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
import json
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
    if hasattr(env_cfg, "enable_warmup"):
        env_cfg.enable_warmup = False   # grasp-only cfg field; g1 has none
    env_cfg.debug_vis = bool(args_cli.debug_vis)   # markers only when explicitly requested (needs a viewer)
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
    # ParaHome clip selection (g1 loco-manip)
    if args_cli.clip_class is not None and hasattr(env_cfg, "clip_class"):
        env_cfg.clip_class = args_cli.clip_class
    if args_cli.clip_name is not None and hasattr(env_cfg, "clip_name"):
        env_cfg.clip_name = args_cli.clip_name

    # ── Recorded-camera overrides (video-only; see the argparse block above) ──
    # hasattr-guarded, mirroring the clip_class/clip_name pattern: a silent no-op on tasks whose cfg
    # has no viewer_* fields (g1_shadow_locomanip has only viewer_zoom — and its hardcoded formula
    # already IS the 'old' view).
    if args_cli.cam_preset == "old":
        for _f, _v in (("viewer_yaw", 45.0), ("viewer_elev", 0.0), ("viewer_look_obj", False)):
            if hasattr(env_cfg, _f):
                setattr(env_cfg, _f, _v)
    # NOTE: cam_preset == "current" (and None) writes NOTHING on purpose.
    for _f, _v in (("viewer_yaw", args_cli.viewer_yaw), ("viewer_elev", args_cli.viewer_elev),
                   ("viewer_zoom", args_cli.viewer_zoom)):
        if _v is not None and hasattr(env_cfg, _f):
            setattr(env_cfg, _f, float(_v))
    if args_cli.viewer_look_obj is not None and hasattr(env_cfg, "viewer_look_obj"):
        env_cfg.viewer_look_obj = bool(args_cli.viewer_look_obj)
    if args_cli.video_resolution is not None and getattr(env_cfg, "viewer", None) is not None:
        _vw, _vh = (int(v) for v in args_cli.video_resolution.lower().split("x"))
        env_cfg.viewer.resolution = (_vw, _vh)   # sole source of the render-product size

    # NO --deterministic FLAG, DELIBERATELY. The two-camera video path re-simulates the same rollout
    # to render it from a second angle, so it is natural to reach for PhysX
    # enable_enhanced_determinism + torch.use_deterministic_algorithms. Both were tried and rejected,
    # MEASURED on this clip (s100_seg00_pan, seed 42, 251 steps):
    #   * Bit-identity is unreachable. Two byte-identical invocations (SAME camera) already differ by
    #     max 1.3e-1 relative at --n_rollouts 32 and 1.2e-3 at --n_rollouts 1. The GPU solver is not
    #     reproducible across processes; enable_enhanced_determinism does not change that (it makes
    #     results independent of OTHER actors in the scene, not of reduction order).
    #   * It is also unnecessary. The camera-change pair diverges LESS (6.8e-2) than the same-camera
    #     pair (1.3e-1), i.e. the camera contributes nothing, and the divergence does not GROW:
    #     frame-wise pixel difference between two same-camera reruns is flat over the whole clip
    #     (visibly-different pixels 5.3% -> 7.6% -> 6.5% across first/mid/last 10 frames at 32 envs;
    #     1.4% -> 1.4% -> 1.0% at 1 env). The panes stay locked; they do not desync.
    #   * torch.use_deterministic_algorithms(True) additionally CRASHES this env: the deterministic
    #     index_put_ kernel does not broadcast a [n,1,D] value into a [n,10,D] masked slice, which is
    #     what g1_shadow_sonic_residual_env.py:1103 (_sonic_hist[k][m] = rows[k][m].unsqueeze(1))
    #     relies on. Fixing that means editing the env, which this video-only feature must not do.
    # compose_side_by_side.py therefore checks a TOLERANCE against that measured noise floor plus the
    # discrete success flags, instead of byte-equality.

    agent_cfg["seed"] = args_cli.seed
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # Disable skrl's experiment logging — rollout only writes metrics.csv to --output_dir
    # (without this, a `./robotis_sh5_grasp/default/` folder with TensorBoard events
    # is created in the CWD every run; mirrors play.py).
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    # ── Create env & runner ───────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Log + persist the RESOLVED viewer pose. env.unwrapped.cfg is the same object the env mutated
    # (DirectRLEnv.__init__ does `self.cfg = cfg`, and gymnasium passes caller kwargs by reference),
    # so this is the clip-adaptive override that _load_reference_trajectories actually computed —
    # the ONLY place it is ever observable (train.py's params/env.yaml is dumped pre-construction
    # and records only the dead cfg literal).
    _vcfg = getattr(env.unwrapped.cfg, "viewer", None)
    if _vcfg is not None:
        _vmeta = {
            "cam_preset": args_cli.cam_preset or "cfg",
            "viewer_yaw": getattr(env.unwrapped.cfg, "viewer_yaw", None),
            "viewer_elev": getattr(env.unwrapped.cfg, "viewer_elev", None),
            "viewer_look_obj": getattr(env.unwrapped.cfg, "viewer_look_obj", None),
            "viewer_zoom": getattr(env.unwrapped.cfg, "viewer_zoom", None),
            "origin_type": _vcfg.origin_type, "env_index": int(_vcfg.env_index),
            "resolution": [int(v) for v in _vcfg.resolution],
            "eye_env_local": [round(float(v), 6) for v in _vcfg.eye],
            "lookat_env_local": [round(float(v), 6) for v in _vcfg.lookat],
            "video": (f"{args_cli.video_name_prefix}-step-0.mp4" if args_cli.video else None),
            "metrics": args_cli.metrics_name, "seed": int(args_cli.seed),
            "clip_class": getattr(env_cfg, "clip_class", None),
            "clip_name": getattr(env_cfg, "clip_name", None),
        }
        print(f"[rollout] viewer: {_vmeta}")
        os.makedirs(args_cli.output_dir, exist_ok=True)
        with open(os.path.join(args_cli.output_dir,
                               f"viewer_{args_cli.video_name_prefix}.json"), "w") as _f:
            json.dump(_vmeta, _f, indent=2)

    # wrap for video recording (before skrl wrapper so RecordVideo sees raw gym API)
    if args_cli.video:
        video_folder = os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_folder, exist_ok=True)
        # Default (video_length <= 0): fit the clip to the full sequence. In eval
        # (adaptive_sampling=False) the episode runs the whole trajectory, so
        # max_episode_length == _max_traj_len == sequence length in control steps.
        # A fixed value (e.g. 300 @ 30 Hz = 10 s) would truncate/pad regardless of
        # sequence length — this makes the video exactly one full sequence.
        if args_cli.video_length > 0:
            video_length = args_cli.video_length
        else:
            video_length = int(env.unwrapped.max_episode_length)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
            "name_prefix": args_cli.video_name_prefix,
        }
        print(f"[rollout] Recording video to {video_folder} (length={video_length} steps"
              f"{' = full sequence' if args_cli.video_length <= 0 else ''}) "
              f"→ {args_cli.video_name_prefix}-step-0.mp4")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, agent_cfg)

    # Replace policy with MassDexMimicPolicy before loading checkpoint (grasp task only).
    _is_grasp = "Grasp" in args_cli.task and "Pretrain" not in args_cli.task
    # g1 loco-manip = plain GaussianMixin (no mass policy); metrics read from the _errs dict.
    _is_g1 = "Locomanip" in (args_cli.task or "")
    if _is_grasp:
        _patch_mass_policy(runner.agent, agent_cfg["models"]["policy"], agent_cfg["agent"]["learning_rate"])

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[rollout] Loading checkpoint: {checkpoint_path}")
    runner.agent.load(checkpoint_path)

    policy = runner.agent.models["policy"]
    policy.eval()

    # IMPORTANT: skrl PPO normalizes observations via _observation_preprocessor before
    # passing them to the policy network (RunningStandardScaler with stats learned
    # during training). We must apply the same normalization here — otherwise the
    # policy sees raw observations far outside its training distribution.
    # (skrl 2.0.0 split obs/state preprocessors; the POLICY input uses observations.)
    observation_preprocessor = runner.agent._observation_preprocessor

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
    n_frames  = int(getattr(actual_env, "_max_traj_len", None) or actual_env._ref_len)   # g1 uses _ref_len
    seq_name  = (getattr(env_cfg, "trajectory_task", None) or getattr(env_cfg, "object_id", None)
                 or getattr(env_cfg, "clip_name", None) or "clip")

    # ── [joint-dump] env 0 의 관절 PD 타겟 vs 실측 ─────────────────────────────────────────
    # env._apply_action 은 set_joint_position_target(_residual_target, _action_joint_ids) 로 씁니다.
    # 그 타겟의 시간축 거동이 bang-bang 판정의 대상입니다. 첫 에피소드 동안만 기록합니다.
    _jrec = None
    if args_cli.dump_joints:
        _jrec = {k: [] for k in ("target", "qpos", "qvel", "tau", "action", "a_sonic",
                                 "root_pos", "root_quat", "ref_root_pos", "frame")}
        print(f"[joint-dump] env 0 기록 시작: {len(actual_env._action_joint_names)}관절")

    for _step in range(args_cli.max_steps):
        with torch.no_grad():
            obs_norm = observation_preprocessor(obs)   # apply training-time normalization stats
            actions, outputs = policy.act({"observations": obs_norm}, role="policy")
            # Default: deterministic (use policy mean). Pass --stochastic to sample.
            if not args_cli.stochastic:
                actions = outputs.get("mean_actions", actions)

        # [zres-ablation] z_res=0 → residual_decode(latent, 0, ...) = 순수 SONIC 디코드.
        if args_cli.zero_zres:
            _nz = int(getattr(env_cfg, "sonic_action_dim", 0))
            if _nz > 0:
                actions = actions.clone()
                actions[:, :_nz] = 0.0

        obs, rewards, terminated, truncated, _info = env.step(actions)

        # Normalise done shape to (n,).
        done = (terminated | truncated)
        if done.ndim == 2:
            done = done.squeeze(-1)
        done_cpu = done.cpu()

        # Accumulate per-step errors for envs still in their first episode.
        # g1 loco-manip exposes errors via the _errs dict (obj_pos/obj_rot/body/ft); grasp exposes
        # them as _last_*_err buffers. e_j (keypoint) uses g1's whole-body error.
        if _is_g1:
            _e = actual_env._errs
            _ope, _ore, _kpe, _fte = _e["obj_pos"], _e["obj_rot"], _e["body"], _e["ft"]
        else:
            _ope, _ore = actual_env._last_obj_pos_err, actual_env._last_obj_rot_err
            _kpe, _fte = actual_env._last_kpts_err_raw, actual_env._last_ft_raw_err
        for i in range(n):
            if not episode_done[i]:
                obj_pos_bufs[i].append(_ope[i].item())
                obj_rot_bufs[i].append(_ore[i].item())
                kpts_bufs[i].append(_kpe[i].item())
                ft_bufs[i].append(_fte[i].item())
                r = rewards[i] if rewards.ndim == 1 else rewards[i, 0]
                reward_sums[i] += float(r)

        # [joint-dump] env 0 이 첫 에피소드를 도는 동안만.
        if _jrec is not None and not episode_done[0]:
            _jid = actual_env._action_joint_ids_t
            _tg = getattr(actual_env, "_residual_target", None)
            _jrec["target"].append((_tg[0] if _tg is not None
                                    else actual_env.robot.data.joint_pos[0, _jid]).clone().cpu())
            _jrec["qpos"].append(actual_env.robot.data.joint_pos[0, _jid].clone().cpu())
            _jrec["qvel"].append(actual_env.robot.data.joint_vel[0, _jid].clone().cpu())
            _jrec["tau"].append(actual_env.robot.data.applied_torque[0, _jid].clone().cpu())
            _jrec["action"].append(actual_env._cur_policy_action[0].clone().cpu())
            # [joint-dump] SONIC 디코더가 결과적으로 내는 29-D 몸통 액션. SONIC 관절 순서로
            # 저장되므로 _sonic_gather 로 action-body 순서(= joint_names[:29])에 맞춥니다.
            _as = getattr(actual_env, "_last_a_sonic", None)
            if _as is not None:
                _g = getattr(actual_env, "_sonic_gather", None)
                _jrec["a_sonic"].append((_as[0, _g] if _g is not None else _as[0]).clone().cpu())
            # [joint-dump] 넘어짐 판정용 루트 상태. rollout 은 termination=False 라 "완주"가
            # 자세 정상을 뜻하지 않습니다 — 골반 높이와 기울기로 직접 봐야 합니다.
            _jrec["root_pos"].append(actual_env.robot.data.root_pos_w[0].clone().cpu())
            _jrec["root_quat"].append(actual_env.robot.data.root_quat_w[0].clone().cpu())
            _rr = getattr(actual_env, "_ref_root_pos", None)
            if _rr is not None:
                _fi = int(actual_env._frame_idx[0].clamp(max=_rr.shape[0] - 1))
                _jrec["ref_root_pos"].append(_rr[_fi].clone().cpu())
            _jrec["frame"].append(int(actual_env._frame_idx[0]))

        # Update mass-policy cache for terminated envs.
        if _is_grasp:
            policy.update_mass_terminated(done)

        # Mark envs whose first episode just ended.
        episode_done |= done_cpu

        if episode_done.all():
            break

    # ── Per-frame error traces ────────────────────────────────────────────────
    # The buffers above already hold every step; only their means reach metrics.csv. Dumping the
    # full traces answers WHEN a rollout diverges, which the per-episode aggregate cannot: an
    # episode that never grasps and one that grasps then drops the object have similar means.
    os.makedirs(args_cli.output_dir, exist_ok=True)
    _L = max(len(b) for b in obj_pos_bufs)
    def _pad(bufs):
        import numpy as _np
        out = _np.full((len(bufs), _L), _np.nan, dtype=_np.float32)
        for _i, _b in enumerate(bufs):
            out[_i, :len(_b)] = _b
        return out
    import numpy as _np
    _np.savez(os.path.join(args_cli.output_dir, "per_frame.npz"),
              obj_pos=_pad(obj_pos_bufs), obj_rot=_pad(obj_rot_bufs),
              kpts=_pad(kpts_bufs), ft=_pad(ft_bufs))
    print(f"[rollout] per-frame traces -> per_frame.npz  ({len(obj_pos_bufs)} x {_L})")

    # ── [joint-dump] 관절 궤적 저장 ────────────────────────────────────────────────────────
    if _jrec is not None and _jrec["qpos"]:
        _jt = os.path.join(args_cli.output_dir, "joint_trace.npz")
        _sv = dict(joint_names=_np.array(actual_env._action_joint_names),
                   ctrl_lower=actual_env._ctrl_lower.cpu().numpy(),
                   ctrl_upper=actual_env._ctrl_upper.cpu().numpy(),
                   frame=_np.asarray(_jrec["frame"], dtype=_np.int64),
                   control_fps=_np.asarray(round(1.0 / (env_cfg.sim.dt * env_cfg.decimation))),
                   clip=_np.asarray(str(seq_name)))
        for _k in ("target", "qpos", "qvel", "tau", "action", "root_pos", "root_quat"):
            _sv[_k] = torch.stack(_jrec[_k]).numpy()
        if _jrec["ref_root_pos"]:
            _sv["ref_root_pos"] = torch.stack(_jrec["ref_root_pos"]).numpy()
        # [joint-dump] SONIC 액션 + 그 액션을 관절 타겟으로 바꾸는 아핀 계수(같은 순서로 gather).
        # 분석 쪽에서 (한계 - default)/scale 로 "관절 한계가 함의하는 액션 범위"를 계산합니다.
        if _jrec["a_sonic"]:
            _sv["a_sonic"] = torch.stack(_jrec["a_sonic"]).numpy()
            _g = getattr(actual_env, "_sonic_gather", None)
            if _g is not None:
                _sv["sonic_default"] = actual_env._sonic_default[0, _g].cpu().numpy()
                _sv["sonic_scale"] = actual_env._sonic_scale[0, _g].cpu().numpy()
        _rj = getattr(actual_env, "_ref_joints", None)
        if _rj is not None:
            _sv["ref_joints"] = _rj.cpu().numpy()
        _np.savez_compressed(_jt, **_sv)
        print(f"[joint-dump] -> joint_trace.npz  스텝 {len(_jrec['qpos'])}  "
              f"관절 {len(actual_env._action_joint_names)}")

    # ── Write metrics.csv ─────────────────────────────────────────────────────
    os.makedirs(args_cli.output_dir, exist_ok=True)
    csv_path = os.path.join(args_cli.output_dir, args_cli.metrics_name)

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
