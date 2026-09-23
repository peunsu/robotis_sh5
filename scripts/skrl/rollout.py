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
parser.add_argument("--keep_termination", action="store_true",
                    help="오차 기반 종료를 켠 채로 rollout 한다 (기본은 끔). train/play 차이가 "
                         "종료 유무에서 오는지 판정하는 스위치 — 학습은 termination=True 로 돌아 "
                         "벗어나는 env 가 리셋되지만 평가는 그걸 끄고 끝까지 굴린다.")
parser.add_argument("--cfg_set", type=str, nargs="*", default=[], metavar="KEY=VAL",
                    help="env_cfg 필드를 직접 덮어쓴다 (예: residual_action=False). 체크포인트가 "
                         "학습된 cfg 와 현재 cfg 가 어긋날 때 맞추기 위한 것 — 액션 의미가 다른 "
                         "cfg 로 rollout 하면 결과가 무의미하다. bool/int/float/str 자동 변환.")
parser.add_argument("--zero_action", action="store_true",
                    help="[wrist-stability] 정책 액션 전체를 0 으로 덮어씁니다. 떠 있는 손에서 "
                         "액션 0 은 손목 잔차 0 (= 레퍼런스 손목 관절값 추종)을 의미하므로, "
                         "손목이 레퍼런스를 따라가는지 = 컨트롤러가 안정한지 보는 진단입니다. "
                         "잔차 손가락 액션도 0 이 되어 손가락은 레퍼런스 자세를 그대로 따릅니다. "
                         "체크포인트는 여전히 필요하지만(Runner 구성용) 결과에 영향은 없습니다.")
parser.add_argument("--dump_joints", action="store_true",
                    help="[joint-dump] env 0 의 PD 타겟/실측 관절 궤적을 joint_trace.npz 로 저장 "
                         "(bang-bang 진단용). _apply_action 이 쓰는 _residual_target 을 그대로 기록.")
# [ROLLBACK MARKER: hand-traj]
parser.add_argument("--dump_hand_traj", action="store_true",
                    help="[hand-traj] (HandPretrain 전용) 모든 rollout 의 손 궤적(손가락 관절 실측/PD 타겟, "
                         "손목 dof6, 손바닥·손끝 위치, 물체 자세·속도, 링크 접촉력, 보상)을 첫 에피소드 동안 "
                         "기록하고, reward_sum 이 가장 큰 rollout 을 골라 <output_dir>/hand_traj_best.npz 로, "
                         "전체를 hand_traj.npz 로 저장합니다. 위치는 env 원점을 뺀 값(= ParaHome 월드 좌표).")
# [/ROLLBACK MARKER: hand-traj]
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
# [ROLLBACK MARKER: viewer-env-index] 녹화 대상 env 를 고릅니다. cfg.viewer.origin_type=="env" 일 때
# 카메라는 이 인덱스의 env 원점 기준으로 놓이고, 그 env 만 화면에 담깁니다. env 코드는 손대지
# 않아도 됩니다 — 카메라 위치 계산(g1_shadow_sonic_residual_env.py:556-)이 레퍼런스 궤적에서
# env-로컬 좌표를 만들기 때문에 인덱스와 무관합니다. 주의: env 마다 리셋/탐색 난수가 달라서
# 다른 인덱스는 "같은 장면의 다른 각도"가 아니라 다른 시행입니다. --n_rollouts 가 이 값보다
# 커야 합니다. 되돌리기: 이 인자와 아래 적용부 삭제 (기본 None 이면 아무것도 안 씁니다).
parser.add_argument("--viewer_env_index", type=int, default=None,
                    help="Override cfg.viewer.env_index (which env the recorded camera follows).")
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
    env_cfg.termination = bool(args_cli.keep_termination)
    if args_cli.keep_termination:
        print("[rollout] termination=True — 학습과 동일한 오차 기반 종료를 유지합니다")

    # [cfg-match] 체크포인트가 학습된 cfg 와 현재 cfg 를 맞추기 위한 직접 덮어쓰기. cfg 가 드리프트
    # 하면(예: residual_action 을 나중에 끄면) 같은 체크포인트가 다른 액션 의미로 해석되어
    # rollout 결과가 무의미해진다. 적용값을 출력해 무엇이 바뀌었는지 기록한다.
    for _kv in args_cli.cfg_set:
        _k, _, _v = _kv.partition("=")
        if not hasattr(env_cfg, _k):
            raise SystemExit(f"[rollout] --cfg_set: env_cfg 에 '{_k}' 필드가 없습니다")
        _cur = getattr(env_cfg, _k)
        if isinstance(_cur, bool):
            _nv = _v.lower() in ("1", "true", "yes", "on")
        elif isinstance(_cur, int):
            _nv = int(_v)
        elif isinstance(_cur, float):
            _nv = float(_v)
        else:
            _nv = _v
        setattr(env_cfg, _k, _nv)
        print(f"[rollout] cfg_set {_k}: {_cur!r} → {_nv!r}")

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
    # [ROLLBACK MARKER: viewer-env-index] cfg.viewer.env_index 는 cfg.viewer_* 와 달리 env 가 읽는
    # 값이 아니라 Isaac Lab ViewportCameraController 가 직접 쓰는 값이라 여기서 바로 꽂습니다.
    if args_cli.viewer_env_index is not None and getattr(env_cfg, "viewer", None) is not None:
        _vei = int(args_cli.viewer_env_index)
        if _vei >= int(args_cli.n_rollouts):
            raise SystemExit(f"[rollout] --viewer_env_index {_vei} >= --n_rollouts {args_cli.n_rollouts}")
        env_cfg.viewer.env_index = _vei
        print(f"[rollout] 녹화 대상 env = {_vei} (origin_type={env_cfg.viewer.origin_type})")
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
    # 떠 있는 손 사전학습(stage 1)도 같은 계보라 _errs 를 같은 키로 노출합니다. 다만 SONIC 관련
    # 코드(--zero_zres, joint-dump 의 a_sonic/골반)에는 걸리지 않아야 하므로 별 플래그로 둡니다.
    #   e_j(키포인트) = _errs["body"] 인데, 손 env 에서는 몸통이 없어 그 항이 palm 2개의
    #   추종 오차로 재정의돼 있습니다 — 즉 손 평가에서 e_j_cm 은 "손목 위치 오차"입니다.
    _is_hand = "HandPretrain" in (args_cli.task or "")
    _has_errs = _is_g1 or _is_hand
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

    # ── [friction-eval] 물체 마찰 커리큘럼을 종료 상태로 고정 ──────────────────────────────
    # _apply_object_friction 은 리셋마다 마찰을 [friction_min, friction_max(t)] 에서 균등
    # 추출하고, 상한이 `_friction_step_count / friction_decay_steps` 로 감쇠합니다. 그 카운터는
    # env 생성 시 0 이고 체크포인트에 저장되지 않으므로, 평가는 커리큘럼 **시작** 조건에서
    # 돌아갑니다 — 정책이 수렴한 조건이 아닙니다.
    #
    # 실측 (41000 step 학습, friction_decay_steps=30000):
    #     학습 종료   friction_max 1.000 → 마찰 1.0 고정   (Curriculum/friction_mean 1.000)
    #     rollout     friction_max 3.000 → 마찰 [1.0,3.0] 무작위 (평균 2.0)
    # 즉 물체를 잡는 과제를 최대 3배 마찰에서 평가하고 있었고, 롤아웃마다 값이 달라
    # 재현성도 없었습니다 (같은 --seed 로도 8개 롤아웃 중 일부만 완주).
    #
    # 카운터를 감쇠 완료 지점 너머로 밀어 fmax == friction_min 이 되게 합니다. cfg 로
    # friction_curriculum=False 를 주는 방법도 있지만(현재 58개 물체 USD 모두 authored
    # 마찰이 1.0 = friction_min 이라 결과가 같습니다), 그건 물체 authoring 에 의존합니다.
    # 여기서는 학습이 끝났을 때와 **같은 코드 경로**로 정확히 friction_min 이 나오게 합니다.
    if getattr(env_cfg, "friction_curriculum", False):
        _fd = int(getattr(env_cfg, "friction_decay_steps", 0))
        actual_env._friction_step_count = _fd + 1
        print(f"[rollout] friction curriculum pinned to terminal state: "
              f"step_count={_fd + 1} → friction = friction_min "
              f"({getattr(env_cfg, 'friction_min', '?')})")
    # ── [/friction-eval] ─────────────────────────────────────────────────────────────────

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
        # [joint-dump] obj_pos/obj_quat: 물리로 잡힌 물체의 실제 자세. figure 렌더가 롤아웃을
        # 그대로 재생할 때 레퍼런스 물체 자세를 쓰면 손과 물체가 어긋나므로 같이 기록합니다.
        # [joint-dump] *_all: 모든 env 의 FULL 관절 벡터(73 = 구동 65 + 텐던 J0 8)와 루트·물체 자세.
        # 구동 65 만 저장하면 재생 시 J0 8개가 articulation 기본값(0 = 손끝 펴짐)으로 남아
        # 손가락이 펴진 그림이 나온다. figure 재생은 *_all 을 써야 한다.
        _jrec = {k: [] for k in ("target", "qpos", "qvel", "tau", "action", "a_sonic",
                                 "root_pos", "root_quat", "ref_root_pos", "frame",
                                 "obj_pos", "obj_quat", "qpos_all", "root_all", "obj_all")}
        print(f"[joint-dump] env 0 기록 시작: {len(actual_env._action_joint_names)}관절")

    # ── [ROLLBACK MARKER: hand-traj] 모든 env 의 손 궤적 기록 ────────────────────────────
    # 스텝마다 (E,…) 텐서를 통째로 쌓고, 각 env 의 첫 에피소드 길이(ep_len)로 마지막에 잘라 씁니다.
    # 상태는 스텝 **전**(= 그 프레임의 상태), PD 타겟·보상은 스텝 **후**(= 그 스텝에 적용된 값)에 기록.
    # done 이 난 env 는 step() 안에서 이미 리셋되어 타겟이 리셋 씨딩 값으로 덮이므로, 그 스텝의 타겟은
    # 직전 값을 그대로 둡니다(마지막 한 스텝만 해당).
    _hrec = None
    if args_cli.dump_hand_traj:
        if not _is_hand:
            print("[hand-traj] HandPretrain 태스크가 아니라 건너뜁니다.")
        else:
            _hrec = {k: [] for k in ("frame", "finger_qpos", "finger_qvel", "joint_pos_all", "wrist_dof6",
                                     "palm_pos", "palm_quat", "palm_linvel", "palm_angvel",
                                     "ft_pos", "ft_pad_inward", "obj_pos", "obj_quat", "obj_linvel", "obj_angvel",
                                     "contact_force_w", "finger_target", "wrist6_target", "reward", "done",
                                     "link_pos", "link_quat")}   # [stage1-contact-map] 32 접촉 링크 바디 자세
            _hrec_ep_len = torch.zeros(n, dtype=torch.long)
            _h_env = actual_env
            _h_hands = (_h_env.hand_l, _h_env.hand_r)
            _h_fids = [getattr(_h_env, f"_finger_joint_ids_{_sd}") for _sd in "lr"]
            _h_wids = [getattr(_h_env, f"_wrist6_joint_ids_{_sd}") for _sd in "lr"]
            _h_sensors = list(getattr(_h_env, "_link_contact_sensors", []) or [])
            print(f"[hand-traj] {n} env 기록 시작: 손가락 {sum(len(f) for f in _h_fids)}관절, "
                  f"손목 관절 {sum(len(w) for w in _h_wids)}, 접촉 링크 {len(_h_sensors)}")

    def _hand_traj_state():
        """스텝 전 상태 (E,…) 를 _hrec 에 추가."""
        _org = _h_env.scene.env_origins
        _cpu = lambda t: t.detach().clone().cpu()
        _hrec["frame"].append(_cpu(_h_env._frame()))
        _hrec["finger_qpos"].append(_cpu(torch.cat([_h.data.joint_pos[:, _f] for _h, _f in zip(_h_hands, _h_fids)], 1)))
        _hrec["finger_qvel"].append(_cpu(torch.cat([_h.data.joint_vel[:, _f] for _h, _f in zip(_h_hands, _h_fids)], 1)))
        _hrec["joint_pos_all"].append(_cpu(torch.stack([_h.data.joint_pos for _h in _h_hands], 1)))
        _hrec["wrist_dof6"].append(_cpu(torch.stack([_h.data.joint_pos[:, _w] for _h, _w in zip(_h_hands, _h_wids)], 1)))
        _hrec["palm_pos"].append(_cpu(_h_env._gather_body(_h_env._palm_sides, _h_env._palm_body_ids, "body_pos_w") - _org[:, None]))
        _hrec["palm_quat"].append(_cpu(_h_env._gather_body(_h_env._palm_sides, _h_env._palm_body_ids, "body_quat_w")))
        _hrec["palm_linvel"].append(_cpu(_h_env._gather_body(_h_env._palm_sides, _h_env._palm_body_ids, "body_lin_vel_w")))
        _hrec["palm_angvel"].append(_cpu(_h_env._gather_body(_h_env._palm_sides, _h_env._palm_body_ids, "body_ang_vel_w")))
        # ── [ROLLBACK MARKER: stage1-contact-map] LINK_CONTACT_NAMES 순서의 32 접촉 링크 바디 자세(env-local 위치, wxyz).
        #    stage1_hand_contact.py 가 이 자세로 Shadow 링크 메시를 놓아 접촉 맵을 만든다 (URDF FK 불필요 = 시뮬레이션과 동일).
        if hasattr(_h_env, "_link_contact_body_ids"):
            _hrec["link_pos"].append(_cpu(_h_env._gather_body(_h_env._lc_sides, _h_env._link_contact_body_ids, "body_pos_w") - _org[:, None]))
            _hrec["link_quat"].append(_cpu(_h_env._gather_body(_h_env._lc_sides, _h_env._link_contact_body_ids, "body_quat_w")))
        # ── [/ROLLBACK MARKER: stage1-contact-map] ──
        _tip, _inw = _h_env._robot_ft_w()
        _hrec["ft_pos"].append(_cpu(_tip - _org[:, None]))
        _hrec["ft_pad_inward"].append(_cpu(_inw))
        _ob = getattr(_h_env, "_object", None)
        if _ob is not None:
            _hrec["obj_pos"].append(_cpu(_ob.data.root_pos_w - _org))
            _hrec["obj_quat"].append(_cpu(_ob.data.root_quat_w))
            _hrec["obj_linvel"].append(_cpu(_ob.data.root_lin_vel_w))
            _hrec["obj_angvel"].append(_cpu(_ob.data.root_ang_vel_w))
        if _h_sensors:
            _fs = []
            for _s in _h_sensors:
                _fm = _s.data.force_matrix_w                            # (E,1,K,3) 물체→링크 힘, 또는 None
                _fs.append(torch.zeros(n, 3, device=_h_env.device) if _fm is None
                           else _fm.reshape(n, -1, 3).sum(1))
            _hrec["contact_force_w"].append(_cpu(torch.stack(_fs, 1)))   # (E,L,3)

    def _hand_traj_post(rewards_t, done_t):
        """스텝 후: 이 스텝에 적용된 PD 타겟 + 보상/종료 (E,…) 를 추가."""
        _cpu = lambda t: t.detach().clone().cpu()
        _tg = getattr(_h_env, "_residual_target", None)
        if _tg is None:
            _tg = getattr(_h_env, "_hand_target_ema", None)
        if _tg is not None:
            _hrec["finger_target"].append(_cpu(_tg))
        _hrec["wrist6_target"].append(_cpu(torch.stack(list(_h_env._wrist6_target), 1)))   # (E,2,6)
        _r = rewards_t if rewards_t.ndim == 1 else rewards_t[:, 0]
        _hrec["reward"].append(_cpu(_r))
        _hrec["done"].append(done_t.clone())
    # [/ROLLBACK MARKER: hand-traj]

    for _step in range(args_cli.max_steps):
        # [ROLLBACK MARKER: hand-traj] 스텝 전 상태 기록
        if _hrec is not None:
            _hand_traj_state()
        # [/ROLLBACK MARKER: hand-traj]
        with torch.no_grad():
            obs_norm = observation_preprocessor(obs)   # apply training-time normalization stats
            actions, outputs = policy.act({"observations": obs_norm}, role="policy")
            # Default: deterministic (use policy mean). Pass --stochastic to sample.
            if not args_cli.stochastic:
                actions = outputs.get("mean_actions", actions)

        # [wrist-stability] 액션 전체 0. 다른 덮어쓰기보다 먼저 적용해 무조건 0 이 되게 합니다.
        if args_cli.zero_action:
            actions = torch.zeros_like(actions)
        # [zres-ablation] z_res=0 → residual_decode(latent, 0, ...) = 순수 SONIC 디코드.
        if args_cli.zero_zres:
            _nz = (int(getattr(env_cfg, "sonic_action_dim", 0))
                   if bool(getattr(env_cfg, "sonic_latent_residual", True)) else 0)
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
        if _has_errs:
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
        # [joint-dump] 손 env 에서는 비활성: _action_joint_ids_t(왼손 18개)와 self.robot(왼손
        # 별칭), _ref_root_pos(골반) 전제가 모두 깨져 조용히 틀린 값을 담게 됩니다.
        if _jrec is not None and _is_hand:
            _jrec = None
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
            _ob = getattr(actual_env, "_object", None)
            if _ob is not None:
                _jrec["obj_pos"].append(_ob.data.root_pos_w[0].clone().cpu())
                _jrec["obj_quat"].append(_ob.data.root_quat_w[0].clone().cpu())
            _jrec["qpos_all"].append(actual_env.robot.data.joint_pos.clone().cpu())
            _jrec["root_all"].append(torch.cat([actual_env.robot.data.root_pos_w,
                                                actual_env.robot.data.root_quat_w], -1).clone().cpu())
            if _ob is not None:
                _jrec["obj_all"].append(torch.cat([_ob.data.root_pos_w,
                                                   _ob.data.root_quat_w], -1).clone().cpu())
            _rr = getattr(actual_env, "_ref_root_pos", None)
            if _rr is not None:
                _fi = int(actual_env._frame_idx[0].clamp(max=_rr.shape[0] - 1))
                _jrec["ref_root_pos"].append(_rr[_fi].clone().cpu())
            _jrec["frame"].append(int(actual_env._frame_idx[0]))

        # Update mass-policy cache for terminated envs.
        if _is_grasp:
            policy.update_mass_terminated(done)

        # [ROLLBACK MARKER: hand-traj]
        if _hrec is not None:
            _hand_traj_post(rewards, done_cpu)
            _hrec_ep_len += (~episode_done).long()          # 이 스텝을 첫 에피소드로 산 env 만 +1
        # [/ROLLBACK MARKER: hand-traj]

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

    # ── [ROLLBACK MARKER: hand-traj] 손 궤적 저장: 전체(hand_traj.npz) + 최고 보상 rollout(hand_traj_best.npz)
    if _hrec is not None and _hrec["frame"]:
        import sys as _sys
        _hm = _sys.modules.get(type(actual_env).__module__)
        _rs = _np.asarray(reward_sums, dtype=_np.float64)
        _best = int(_np.argmax(_rs))
        _T = len(_hrec["frame"])
        _sv = dict(best_idx=_np.asarray(_best), reward_sums=_rs, ep_len=_hrec_ep_len.numpy(),
                   n_frames=_np.asarray(n_frames), clip=_np.asarray(str(seq_name)),
                   clip_class=_np.asarray(str(getattr(env_cfg, "clip_class", ""))),
                   object_name=_np.asarray(str(getattr(actual_env, "_obj_name", ""))),
                   checkpoint=_np.asarray(str(args_cli.checkpoint)),
                   control_fps=_np.asarray(round(1.0 / (env_cfg.sim.dt * env_cfg.decimation))),
                   wrist_mode=_np.asarray("joint6"),   # 파일 형식 호환용 (wrench 모드는 제거됨)
                   stochastic=_np.asarray(bool(args_cli.stochastic)),
                   joint_names=_np.array(actual_env._action_joint_names),
                   joint_names_all_l=_np.array(actual_env.hand_l.joint_names),
                   joint_names_all_r=_np.array(actual_env.hand_r.joint_names),
                   link_contact_names=_np.array(list(getattr(_hm, "LINK_CONTACT_NAMES", []))),
                   fingertip_body_names=_np.array(list(getattr(env_cfg, "fingertip_body_names", []))),
                   ctrl_lower=actual_env._ctrl_lower.cpu().numpy(), ctrl_upper=actual_env._ctrl_upper.cpu().numpy())
        _sv["wrist6_joint_names_l"] = _np.array([actual_env.hand_l.joint_names[i] for i in _h_wids[0].tolist()])
        _sv["wrist6_joint_names_r"] = _np.array([actual_env.hand_r.joint_names[i] for i in _h_wids[1].tolist()])
        # 레퍼런스(같은 50 Hz 프레임 색인) — 비교·변환용
        for _k, _a in (("ref_joints", getattr(actual_env, "_ref_joints", None)),
                       ("ref_wrist_dof6", getattr(actual_env, "_ref_wrist_dof6", None)),
                       ("ref_wrist_pose", getattr(actual_env, "_ref_wrist_pose", None))):
            if _a is not None:
                _sv[_k] = _a.detach().cpu().numpy()
        for _k, _lst in _hrec.items():
            if not _lst:
                continue
            _arr = torch.stack(_lst, 0).numpy()                          # (T,E,…)
            if len(_lst) < _T:                                           # 타겟은 done 스텝에서 빠질 수 있음 → 앞값 유지
                _arr = _np.concatenate([_arr, _np.repeat(_arr[-1:], _T - len(_lst), 0)], 0)
            _sv[_k] = _np.swapaxes(_arr, 0, 1)                           # (E,T,…)
        _np.savez_compressed(os.path.join(args_cli.output_dir, "hand_traj.npz"), **_sv)
        _L = int(_hrec_ep_len[_best])
        _bs = {k: (v[_best, :_L] if (isinstance(v, _np.ndarray) and v.ndim >= 2 and v.shape[0] == n and v.shape[1] == _T)
                   else v) for k, v in _sv.items()}
        _bs["rollout_idx"] = _np.asarray(_best)
        _bs["reward_sum"] = _np.asarray(_rs[_best])
        _np.savez_compressed(os.path.join(args_cli.output_dir, "hand_traj_best.npz"), **_bs)
        _order = _np.argsort(-_rs)
        print(f"[hand-traj] -> hand_traj.npz ({n} x {_T} 스텝), hand_traj_best.npz = rollout {_best} "
              f"(reward_sum {_rs[_best]:.2f}, 길이 {_L}/{n_frames} 프레임, 마지막 프레임 {int(_sv['frame'][_best, _L - 1])})")
        print("[hand-traj] reward_sum 상위 5: " + ", ".join(f"#{i}:{_rs[i]:.1f}(len {int(_hrec_ep_len[i])})" for i in _order[:5]))
    # [/ROLLBACK MARKER: hand-traj]

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
        if _jrec["obj_pos"]:
            _sv["obj_pos"] = torch.stack(_jrec["obj_pos"]).numpy()
            _sv["obj_quat"] = torch.stack(_jrec["obj_quat"]).numpy()
        _sv["joint_names_all"] = _np.array(list(actual_env.robot.joint_names))
        for _k in ("qpos_all", "root_all", "obj_all"):
            if _jrec[_k]:
                _sv[_k] = torch.stack(_jrec[_k]).numpy()          # (T, E, ...)
        _sv["env_origin"] = actual_env.scene.env_origins[0].cpu().numpy()
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
