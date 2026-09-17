"""Export a trained floating-hand policy's trajectory as a stage-2 tracking target.

Stage 1 (floating-hand dexterous pretrain) produces a policy; stage 2 (full-body loco-manip) needs a
per-frame TRAJECTORY. This script bridges them: it runs one DETERMINISTIC rollout per clip (mean
action, no sampling) and writes the wrist pose + finger joints the policy actually achieved.

    <clip_dir>/hand_rl.npz
        wrist_pose_l   (F, 7)   world, pos + quat wxyz
        wrist_pose_r   (F, 7)
        hand_joints_l  (F, 18)  actuated finger joints, in the env's actuated order
        hand_joints_r  (F, 18)
        joint_names_l  (18,)    explicit, so stage 2 can match BY NAME rather than by position
        joint_names_r  (18,)
        meta           dict: checkpoint, wrist gains, clip, n_frames, **fps**

Frame rate
----------
`fps` is in meta and MUST be read. This file is at the env's `control_fps` (50 Hz), NOT at the
30 Hz of the ParaHome source and its sidecars (`wrist_ref.npz`, `trajectory_pyroki.npz` are 301
frames where this is 501). Truncating with `min(len(a), len(b))` instead of resampling silently
time-shifts the two apart — measured 13 cm of pure misalignment at frame 100.

Divergence
----------
The rollout runs with `termination=False` so the whole reference gets rolled, which also removes
the gate that resets a numerically blown-up env. A floating hand CAN diverge (measured: joint
values reaching 2.03e16), and an unguarded export writes those values straight into the stage-2
target — observed 1.8e15 m wrist positions from a lightly-trained checkpoint. So the export
CHECKS every frame and REFUSES to write if any is non-finite or out of range. A partial or
poisoned file that stage 2 consumes silently is worse than no file.

Why names are stored
--------------------
The 766→772 observation mismatch and the `_wrist_frame_idx` off-by-one both came from a hardcoded
count outliving a layout change. Positional joint arrays are the same trap, and we already pay for
it once: the train env has a whole `_remap_ref_joints` machine because `g1_joint_pos` was keyed to a
static json that drifted from the robot's DOF order. So this file records names next to values and
stage 2 matches on them.

Why deterministic
-----------------
The output is a reference trajectory, not a sample. A stochastic rollout would make stage 2's target
depend on the RNG, and the A/B against the retargeted hand would not be reproducible.

Usage (needs the GPU — run when no training job holds the device)
------------------------------------------------------------------
    <env_isaaclab python> scripts/process_dataset/retarget/export_hand_rl.py \
        --checkpoint logs/skrl/g1_shadow_hand_pretrain/<run>/checkpoints/best_agent.pt \
        --clip s101_seg12_knife --class single_rigid
    # --clip all  → every clip that has a hand_contact.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_PROC = (Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data"
         / "processed" / "parahome")

_TASK = "Robotis-G1-Shadow-HandPretrain-Direct-v0"
_N_FINGER = 18


def _clips_with_contact(cls: str) -> list[str]:
    """Clips that have a per-link contact map — the ones stage 1 can actually be trained on.

    `s10_seg03_book` is the current exception (no hand_contact.npz), so it drops out here rather
    than silently training with the contact and CWS rewards inert.
    """
    root = _PROC / "g1_shadow" / cls
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (_PROC / "smplx" / cls / d.name / "0" / "hand_contact.npz").exists():
            out.append(d.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clip", default="all", help='clip name, or "all"')
    ap.add_argument("--class", dest="cls", default="single_rigid")
    ap.add_argument("--num_envs", type=int, default=1,
                    help="1 keeps the rollout deterministic and the device footprint minimal")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--headless", action="store_true", default=True)
    a = ap.parse_args()

    clips = _clips_with_contact(a.cls) if a.clip == "all" else [a.clip]
    print(f"[export-hand-rl] {len(clips)} clip(s): {clips}")

    # Isaac Sim boots here, not at import time, so `--help` and the clip listing stay GPU-free.
    from isaaclab.app import AppLauncher

    app_ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(app_ap)
    app_args = app_ap.parse_args(["--headless"] if a.headless else [])
    app = AppLauncher(app_args).app

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from skrl.utils.runner.torch import Runner

    import robotis_sh5.tasks  # noqa: F401  (registers the task ids)

    written = []
    for clip in clips:
        cfg = parse_env_cfg(_TASK, device="cuda:0", num_envs=a.num_envs)
        cfg.clip_class = a.cls
        cfg.clip_name = clip
        cfg.termination = False          # roll the whole reference, do not cut on error gates
        env = gym.make(_TASK, cfg=cfg)
        u = env.unwrapped
        # skrl 은 래핑된 env 를 요구합니다 (device / num_envs / 텐서 규약). 래핑을 빼면
        # Runner 안에서 `'OrderEnforcing' object has no attribute 'device'` 로 터집니다.
        # train.py / rollout.py 와 같은 래퍼를 씁니다.
        env = SkrlVecEnvWrapper(env, ml_framework="torch")

        out_path = _PROC / "g1_shadow" / a.cls / clip / "0" / "hand_rl.npz"
        if out_path.exists() and not a.overwrite:
            print(f"  [{clip}] 이미 존재 — --overwrite 없이 건너뜀: {out_path.name}")
            env.close()
            continue

        # skrl runner rebuilds the policy from the same yaml the training used, so the
        # architecture cannot drift from the checkpoint.
        # `Runner` needs the PARSED config dict. gym.spec(...).kwargs[...] is only the
        # entry-point STRING ("pkg.agents:skrl_ppo_cfg.yaml"); handing that over fails inside
        # skrl with `AttributeError: 'str' object has no attribute 'get'`, which points at
        # skrl's own cfg check rather than at the real mistake here.
        agent_cfg = load_cfg_from_registry(_TASK, "skrl_cfg_entry_point")
        runner = Runner(env, agent_cfg)
        runner.agent.load(a.checkpoint)
        # skrl 2.1 의 이름은 enable_training_mode 입니다 (set_running_mode/set_training_mode 둘 다 없음). 평가 모드로 두면
        # agent.act 가 value 헤드를 건너뛰고 모델도 eval() 이 됩니다. rollout.py 는
        # models["policy"].eval() 만 부르는데, 여기서는 에이전트까지 함께 내립니다.
        runner.agent.enable_training_mode(False)

        F = int(u._ref_len)
        _fps = float(getattr(cfg, "control_fps", 0.0)) or (1.0 / (cfg.sim.dt * cfg.decimation))
        wl = np.zeros((F, 7), np.float32)
        wr = np.zeros((F, 7), np.float32)
        jl = np.zeros((F, _N_FINGER), np.float32)
        jr = np.zeros((F, _N_FINGER), np.float32)

        obs, _ = env.reset()
        for f in range(F):
            with torch.no_grad():
                # deterministic: take the mean, not a sample. act() is called ONCE and both
                # branches read that one result — calling it twice would advance the policy's
                # RNG and make the "deterministic" export depend on call order.
                # skrl 시그니처: act(observations, states, *, timestep, timesteps)
                #   -> (actions, outputs).  states 는 필수 위치인자이고 단일 에이전트는 None.
                actions, outputs = runner.agent.act(obs, None, timestep=f, timesteps=F)
                act = outputs.get("mean_actions") if isinstance(outputs, dict) else None
                if act is None:                      # policies without a mean head
                    act = actions
            for side, wbuf, jbuf in (("l", wl, jl), ("r", wr, jr)):
                hand = getattr(u, f"hand_{side}")
                root = hand.data.root_state_w[0, :7].detach().cpu().numpy()
                root[:3] -= u.scene.env_origins[0].detach().cpu().numpy()   # env-local
                wbuf[f] = root
                ids = getattr(u, f"_finger_joint_ids_{side}")
                jbuf[f] = hand.data.joint_pos[0, ids].detach().cpu().numpy()
            obs = env.step(act)[0]

        # ── 발산 검사 ──────────────────────────────────────────────────────────────────
        # 손목 위치는 작업공간(수 m) 안, 손가락 관절은 |q| <= ~1.6 rad 이어야 합니다. 넉넉한
        # 상한을 두고, 하나라도 넘으면 쓰지 않습니다 — 이 파일이 stage 2 의 추종 목표입니다.
        _bad = None
        for _nm, _arr, _lim in (("wrist_pose_l", wl, 1.0e2), ("wrist_pose_r", wr, 1.0e2),
                                ("hand_joints_l", jl, 1.0e2), ("hand_joints_r", jr, 1.0e2)):
            _nf = ~np.isfinite(_arr).all(axis=-1)
            _ov = np.abs(np.nan_to_num(_arr, nan=0.0)).max(axis=-1) > _lim
            _b = _nf | _ov
            if _b.any():
                _bad = (_nm, int(np.argmax(_b)), int(_b.sum()))
                break
        if _bad is not None:
            _nm, _f0, _n = _bad
            print(f"  [{clip}] 발산 — 쓰지 않습니다. {_nm} 이 프레임 {_f0} 부터 비정상 "
                  f"(비정상 프레임 {_n}/{F}).")
            print(f"           이 체크포인트로는 stage 2 목표를 만들 수 없습니다. "
                  f"떠 있는 손이 수치적으로 터진 것이므로 더 학습된 체크포인트를 쓰거나, "
                  f"손목 게인/액션 크기를 낮춰야 합니다.")
            env.close()
            continue
        meta = dict(checkpoint=os.path.abspath(a.checkpoint), clip=clip, cls=a.cls,
                    n_frames=F, fps=_fps, wrist_k_pos=float(cfg.wrist_k_pos),
                    wrist_k_rot=float(cfg.wrist_k_rot),
                    wrist_force_ema=float(cfg.wrist_force_ema),
                    wrist_action_dt=float(cfg.wrist_action_dt))
        np.savez(out_path,
                 wrist_pose_l=wl, wrist_pose_r=wr,
                 hand_joints_l=jl, hand_joints_r=jr,
                 # 손별 articulation 에서 각자 읽습니다. 위 루프의 `hand` 를 재사용하면
                 # 루프 종료 시점의 값(= 오른손)이 새어 나와 왼손 인덱스에 오른손 이름이
                 # 붙습니다 — 이 파일이 이름을 저장하는 이유가 바로 그 종류의 버그입니다.
                 joint_names_l=np.array([u.hand_l.data.joint_names[i]
                                         for i in u._finger_joint_ids_l.tolist()]),
                 joint_names_r=np.array([u.hand_r.data.joint_names[i]
                                         for i in u._finger_joint_ids_r.tolist()]),
                 meta=np.array([meta], dtype=object))
        print(f"  [{clip}] wrote {out_path}  F={F}")
        written.append(clip)
        env.close()

    print(f"[export-hand-rl] done: {len(written)}/{len(clips)}")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
