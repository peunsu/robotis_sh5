"""G1 + Shadow-hand full-body loco-manipulation environment (DirectRLEnv).

Ports the core mechanisms of `robotis_shadow_grasp_rsi` (per-group EMA, contact-conditioned
fingertip force with grounded normal, adaptive frame-sampling curriculum, deviation-from-reference
termination, state cache)
from fixed-base single-hand to FLOATING-base bimanual FULL BODY.

SONIC-RESIDUAL VARIANT (this env): the 29 G1 body DOF are driven by a FROZEN SONIC whole-body
decoder — the policy outputs a 64-D latent residual z_res (added to SONIC's FSQ latent PRE-
quantization, GRAIL Eq.6, λ = residual_scale_latent = 0.15) — plus a 36-D ABSOLUTE bimanual hand
action (mapped directly to the Shadow joint range, EMA-smoothed α=0.5). action=100, obs=765
(obs prev_action = the raw 100-D policy action, GRAIL-style). SONIC is required (use_sonic=True); the
non-SONIC residual action path lives in the hand_pretrain env. SONIC is built in _post_init_buffers.

EPISODES ARE VARIABLE LENGTH: each one runs from its RSI start frame to the END of the reference
sequence (or an early deviation termination) — there is no fixed episode_length_s chunk. See
__init__ (horizon), _get_dones (frame-based time-out) and _reset_idx (start clamp).

Reference = ParaHome preprocessed trajectory.npz (SMPLX tree, produced by parahome.py):
world keypoint targets (joint_positions), fingertip pad targets, root SE(3)
(body_global_transform), and per-object 6-DoF + articulation DOF. Per-frame retargeted G1
joint targets are OPTIONAL (retargeting pipeline pending) — if present in the retarget tree
they seed reset poses; otherwise reset falls back to the G1_SHADOW_CFG standing pose and the
policy trains on keypoint tracking. Object entities spawn only when their converted USD
exists; until then the object reward/termination terms are inert and the robot trains on
kinematic keypoint tracking (the "pretrain-style" path).

Load-bearing conventions honored from the grasp precedent (see the extraction workflow):
  * every orientation obs is 6D (never raw quat); every relative quat is canonicalized
    (w<0 → negate) before 6D / arcsin to kill the double-cover discontinuity.
  * Shadow palm→landmark quat conversion applied to BOTH hands before hand-frame comparison.
  * fingertip force = (force · (-pad_normal_w)).clamp_min(0) — pad-INWARD projection; the
    left-hand pad normals/offsets are already Y-mirrored in the cfg (do not re-mirror). The per-LINK
    contact-force reward uses the same scheme over all 32 wrap links (LINK_PAD_NORMALS).
  * per-group EMA α = NEW-action weight; _smoothed_actions seeded at the normalized default.
  * obs prev_action = the RAW 100-D policy action (z_res 64 + a_hand 36), GRAIL-style (SONIC variant).
  * termination is DEVIATION-FROM-REFERENCE (so reference crouch/bend never trips a fall).
"""

from __future__ import annotations

import math
import json
import os
import glob  # [stage1-hand]

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from . import cws as CWS
from .g1_shadow_sonic_residual_env_cfg import (
    _ROBOT_USD,
    BODY_KPTS,
    FINGERTIP_OFFSETS,
    FINGERTIP_PAD_NORMALS,
    HAND_CHAIN,
    JOINT_GROUPS,
    LINK_CONTACT_NAMES,
    LINK_PAD_NORMALS,
    N_BODY_KPTS,
    N_HAND_KPTS_PER_HAND,
    N_LINK_CONTACT,
    G1ShadowSonicResidualEnvCfg,
)

# Fixed-point rounds for the spawn-declear settle (see _solve_spawn_declear). Not a cfg knob:
# 1 round already converges for compact objects and 3 is enough for ones that tip as they
# settle; there is nothing to tune between those.
_DECLEAR_ROUNDS = 3

# ParaHome native reference rate (fps). The env resamples all per-frame references from THIS to
# cfg.control_fps (SONIC's 50 Hz) in _load_reference; must equal parahome_smpl_for_sonic SRC_FPS.
_PARAHOME_FPS = 30.0


def _quat_to_6d(q: torch.Tensor) -> torch.Tensor:
    """6D continuous rotation rep (Zhou et al.): first two columns of R. q:(...,4) wxyz."""
    m = math_utils.matrix_from_quat(q)                       # (...,3,3)
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)   # (...,6)


def _canon(q: torch.Tensor) -> torch.Tensor:
    """Canonicalize quat to the w>=0 hemisphere (kills double-cover discontinuity)."""
    return torch.where(q[..., :1] < 0, -q, q)


# ── [ROLLBACK MARKER: stage1-hand] 1단계 롤아웃(control_fps 행) → 레퍼런스 프레임 시간축 보간 ─────
def _s1_resample(x, k):
    """x (T,...) 를 실수 행 인덱스 k (N,) 에서 선형 보간합니다. 양끝은 첫/마지막 값을 유지합니다."""
    x = np.asarray(x, np.float32)
    i0 = np.clip(np.floor(k).astype(int), 0, len(x) - 1)
    i1 = np.clip(i0 + 1, 0, len(x) - 1)
    w = np.clip(k - i0, 0.0, 1.0).reshape((-1,) + (1,) * (x.ndim - 1)).astype(np.float32)
    return ((1.0 - w) * x[i0] + w * x[i1]).astype(np.float32)


def _s1_nlerp(q, k):
    """(T,...,4) wxyz 사원수: 부호를 맞춘 뒤 선형 보간·정규화 (인접 프레임이라 slerp 와 차이 없음)."""
    q = np.asarray(q, np.float32)
    i0 = np.clip(np.floor(k).astype(int), 0, len(q) - 1)
    i1 = np.clip(i0 + 1, 0, len(q) - 1)
    w = np.clip(k - i0, 0.0, 1.0).reshape((-1,) + (1,) * (q.ndim - 1)).astype(np.float32)
    a, b = q[i0], q[i1]
    b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)
    o = (1.0 - w) * a + w * b
    return (o / np.clip(np.linalg.norm(o, axis=-1, keepdims=True), 1e-9, None)).astype(np.float32)
# ── [/ROLLBACK MARKER: stage1-hand] ──


class G1ShadowSonicResidualEnv(DirectRLEnv):
    cfg: G1ShadowSonicResidualEnvCfg

    # ------------------------------------------------------------------ init
    def __init__(self, cfg: G1ShadowSonicResidualEnvCfg, render_mode: str | None = None, **kwargs):
        # 이 env 는 SONIC 전용이다. 정책 액션 버퍼와 관측이 SONIC 빌드 블록 안에서만 만들어지므로
        # use_sonic=False 로는 돌지 않는다. SONIC 없이 잔차 액션으로 도는 경로는 hand_pretrain env 에 있다.
        if not getattr(cfg, "use_sonic", True):
            raise ValueError("G1ShadowSonicResidualEnv 는 use_sonic=True 가 필요합니다. SONIC 없이 쓰는 잔차 "
                             "액션 경로는 g1_shadow_hand_pretrain env 에 있습니다.")
        # [ROLLBACK MARKER: waist-gain-mult] 허리 3관절 PD 게인만 배수로 올리고 액션 스케일은 그대로 둔다.
        # 얼어 있는 SONIC 의 scale = 0.25*effort/k 규약을 깨지 않기 위해서다. 되돌리기: waist_gain_scale = 1.0
        _wgs = float(getattr(cfg, "waist_gain_scale", 1.0))
        if abs(_wgs - 1.0) > 1e-9:
            import copy as _copy, math as _math
            from isaaclab.actuators import ImplicitActuatorCfg as _IAC
            _rc = _copy.deepcopy(cfg.robot_cfg)
            _acts = dict(_rc.actuators)
            # 허리를 기존 두 그룹에서 떼어낸다 (발목/힙요는 원래 게인 유지)
            _acts["sonic_ankle_waist"].joint_names_expr = [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            _acts["sonic_hipyaw_waistyaw"].joint_names_expr = [".*_hip_yaw_joint"]
            # 원래 그룹 값 x 배수. effort/velocity/armature 는 그대로 — 토크 한계를 바꾸면
            # 위의 "|a|=1 = 25% 토크" 규약이 다시 깨진다.
            for _nm, _joints, _k0, _d0, _eff, _vel, _arm in (
                ("waist_gm", ["waist_roll_joint", "waist_pitch_joint"],
                 28.5013, 1.8143, 50.0, 37.0, 0.00721945),
                ("waist_yaw_gm", ["waist_yaw_joint"],
                 40.1795, 2.5579, 88.0, 32.0, 0.010177520),
            ):
                _k, _d = _k0 * _wgs, _d0 * _wgs
                _acts[_nm] = _IAC(joint_names_expr=_joints, effort_limit_sim=_eff,
                                  velocity_limit_sim=_vel, stiffness=_k, damping=_d, armature=_arm)
                print(f"[waist-gain] {_nm}: k {_k0:.4f} -> {_k:.4f}   d {_d0:.4f} -> {_d:.4f}"
                      f"   w_n {_math.sqrt(_k / _arm) / (2 * _math.pi):.1f} Hz"
                      f"   zeta {_d / (2 * _math.sqrt(_k * _arm)):.2f}")
            _rc.actuators = _acts
            cfg.robot_cfg = _rc
            print(f"[waist-gain] 배수 {_wgs}, 액션 스케일은 변경 없음 (roll/pitch 0.4386, yaw 0.5475)")
        # [/ROLLBACK MARKER: waist-gain-mult]
        # [ROLLBACK MARKER: ankle-gain-mult] 좌우 ankle_pitch PD 게인만 배수로 (SONIC v1.1 --motor-kp-scale 4,10).
        # 4/10 은 하드웨어 모터 인덱스(= ankle_pitch)이지 시뮬 관절 순서가 아니다. 되돌리기: ankle_gain_scale = 1.0
        _ags = float(getattr(cfg, "ankle_gain_scale", 1.0))
        if abs(_ags - 1.0) > 1e-9:
            import copy as _copy, math as _math
            from isaaclab.actuators import ImplicitActuatorCfg as _IAC
            _rc = _copy.deepcopy(cfg.robot_cfg)
            _acts = dict(_rc.actuators)
            # 현재 내용이 무엇이든 ankle_pitch 만 빼낸다 (waist 블록 실행 여부와 무관하게 동작)
            _aw = _acts["sonic_ankle_waist"]
            _aw.joint_names_expr = [n for n in _aw.joint_names_expr if n != ".*_ankle_pitch_joint"]
            _k0, _d0, _arm = 28.5013, 1.8143, 0.00721945
            _k, _d = _k0 * _ags, _d0 * _ags
            _acts["ankle_pitch_gm"] = _IAC(
                joint_names_expr=[".*_ankle_pitch_joint"], effort_limit_sim=50.0,
                velocity_limit_sim=37.0, stiffness=_k, damping=_d, armature=_arm)
            _rc.actuators = _acts
            cfg.robot_cfg = _rc
            print(f"[ankle-gain] ankle_pitch: k {_k0:.4f} -> {_k:.4f}   d {_d0:.4f} -> {_d:.4f}"
                  f"   w_n {_math.sqrt(_k / _arm) / (2 * _math.pi):.1f} Hz"
                  f"   zeta {_d / (2 * _math.sqrt(_k * _arm)):.2f}")
            print(f"[ankle-gain] 배수 {_ags}, 액션 스케일 변경 없음 (0.4386), ankle_roll 은 대상 아님"
                  f"  남은 sonic_ankle_waist={_aw.joint_names_expr}")
        # [/ROLLBACK MARKER: ankle-gain-mult]
        # [ROLLBACK MARKER: joint-residual] 상체 관절 잔차: SONIC 디코더 출력 뒤에 더한다.
        # 액션·관측(prev_action)이 +N 이 되므로 super() 전에 cfg 를 고쳐야 공간과 env.yaml 이 맞는다.
        self._upper_on = (bool(getattr(cfg, "sonic_upper_residual", False)) and bool(getattr(cfg, "use_sonic", True)))
        # "all" = 몸 29관절 전부(다리 12 + 허리 3 + 팔 14, video_to_data 의 ReconBody 방식),
        # "upper" = sonic_upper_residual_joints 의 17관절만(ReconHand 방식). 29 는 액션 관절의
        # 앞 29개가 몸이라는 이 env 의 불변식이다(_action_joint_ids[:29]).
        self._res_mode = str(getattr(cfg, "sonic_residual_joints_mode", "upper"))
        if self._upper_on:
            _n_up = 29 if self._res_mode == "all" else len(list(cfg.sonic_upper_residual_joints))
            cfg.action_space = int(cfg.action_space) + _n_up
            cfg.observation_space = int(cfg.observation_space) + _n_up
            print(f"[joint-residual] ON: 상체 {_n_up}관절 = SONIC 출력 + "
                  f"scale·u (그룹별 scale, SONIC 액션 단위, tanh/EMA 없음); "
                  f"action {cfg.action_space - _n_up}→{cfg.action_space}, "
                  f"obs {cfg.observation_space - _n_up}→{cfg.observation_space}")
        # 잠재 잔차 z_res 를 끄면 액션에서 그 블록을 통째로 뺀다. SONIC 디코더에는 0 벡터를 넘기므로
        # latent + lambda*0 = latent 가 되어 순수 SONIC 디코드가 된다. 몸은 관절 잔차만으로 움직인다.
        self._zres_on = (bool(getattr(cfg, "sonic_latent_residual", True)) and bool(getattr(cfg, "use_sonic", True)))
        self._act_z = int(cfg.sonic_action_dim) if self._zres_on else 0
        if bool(getattr(cfg, "use_sonic", True)) and not self._zres_on:
            cfg.action_space = int(cfg.action_space) - int(cfg.sonic_action_dim)
            cfg.observation_space = int(cfg.observation_space) - int(cfg.sonic_action_dim)
            print(f"[joint-residual] 잠재 잔차 OFF: 액션에서 z_res {cfg.sonic_action_dim}차원 제거 "
                  f"(SONIC 은 순수 디코드); action →{cfg.action_space}, obs →{cfg.observation_space}")
        # ── [/ROLLBACK MARKER: joint-residual] ──
        self._load_reference_trajectories(cfg)          # numpy buffers (pre-super: no device yet) → sets _ref_len
        self._build_object_cfg(cfg)                     # guarded: only if converted USD exists
        # 에피소드 = RSI 시작 프레임부터 레퍼런스 끝(또는 종료)까지라 길이가 가변이다. max_episode_length 는 전체 길이로
        # 두는 안전 상한이고 실제 시간 초과는 _get_dones 의 프레임 기준이 먼저 걸린다. super() 전에 설정해야 한다.
        _action_fps = round(1.0 / (cfg.sim.dt * cfg.decimation))
        cfg.episode_length_s = self._ref_len / _action_fps
        super().__init__(cfg, render_mode, **kwargs)    # calls _setup_scene; consumes cfg.episode_length_s
        self._post_init_buffers()                       # device tensors, index maps, caches
        if cfg.debug_vis:
            self._setup_debug_vis()                     # reference-keypoint markers

    # ------------------------------------------------------- reference loading
    def _resolve_clip_dir(self, cfg) -> str:
        root = os.path.join(cfg.dataset_root, cfg.smplx_subdir, cfg.clip_class)
        name = cfg.clip_name
        if not name:
            if os.path.isdir(root):
                cands = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
                name = cands[0] if cands else ""
        return os.path.join(root, name, "0") if name else ""

    def _load_reference_trajectories(self, cfg) -> None:
        """Load one ParaHome clip into numpy reference buffers (converted to device in
        _post_init_buffers). Keypoint order (56) = BODY_KPTS(16) + left-hand(20) + right-hand(20),
        matching the robot-side keypoint order built in _post_init_buffers."""
        clip_dir = self._resolve_clip_dir(cfg)
        npz_path = os.path.join(clip_dir, "trajectory.npz") if clip_dir else ""
        if not npz_path or not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"ParaHome clip not found (dataset_root={cfg.dataset_root}, class={cfg.clip_class}, "
                f"name={cfg.clip_name or '<auto>'}). Run scripts/process_dataset/dataset/parahome.py first.")
        d = np.load(npz_path, allow_pickle=True)
        # [ROLLBACK MARKER: smplx-kpts] 키포인트 소스를 SMPL-X 로 (2026-09-04, 리타게팅과 같은 스켈레톤).
        # 배열 = [smplx_joints(55) | fingertip_pad_pos(10)] = (F,65,3). SMPL-X 손에는 손끝 관절이 없어 pad 정점을 쓴다.
        if "smplx_joints" not in d.files:
            raise KeyError("[smplx-kpts] smplx_joints 없음 — parahome.py --overwrite 로 재생성하세요")
        jp = np.concatenate([d["smplx_joints"].astype(np.float32),
                             d["fingertip_pad_pos"].astype(np.float32)], axis=1)   # (F,65,3)
        F = jp.shape[0]

        # HAND_CHAIN["parahome"] 값: p>=0 손 블록 로컬 인덱스(왼 25 / 오 40 기준), -10 손목(SMPL-X 20/21),
        # -1..-5 pad 손가락(th, ff, mf, rf, lf) → 55 + side*5 + (-p-1)
        _PAD_BASE = 55
        ref_idx: list[int] = list(BODY_KPTS.keys())            # 13 body (SMPL-X 인덱스)
        for _s, (_hb, _wr, _pb) in enumerate(((25, 20, _PAD_BASE), (40, 21, _PAD_BASE + 5))):
            for spec in HAND_CHAIN.values():
                for p in spec["parahome"]:
                    if p >= 0:
                        ref_idx.append(_hb + p)
                    elif p == -10:
                        ref_idx.append(_wr)
                    else:
                        ref_idx.append(_pb + (-p - 1))
        self._np_ref_kpts = jp[:, ref_idx, :]                  # (F,55,3) = 13 body + 21 hand x2

        # 레퍼런스 발 접지 (F,2)=[L,R]: 리타게팅 _foot_contact 와 같은 규칙(볼 = SMPL-X 10/11, 원본 fps).
        # [ROLLBACK MARKER: foot-plant-3d] 속도는 SONIC foot_detect 처럼 3D 노름 (2026-09-04, 리타게팅과 동일해야 함).
        _bp = jp[:, [10, 11], :]                               # (F,2,3) [L,R] ball position
        _bv = np.zeros(_bp.shape[:2], dtype=np.float32)        # (F,2) 3D speed (m/s)
        _bv[1:] = np.linalg.norm(_bp[1:] - _bp[:-1], axis=-1) * _PARAHOME_FPS
        self._np_ref_foot_contact = ((_bp[..., 2] < cfg.foot_plant_h) & (_bv < cfg.foot_plant_v)).astype(np.float32)
        # ── [/ROLLBACK MARKER: foot-plant-3d] ────────────────────────────────────────────────

        # root SE(3): default from body_global_transform (human pelvis); OVERRIDDEN below by the
        # retargeting's ADJUSTED root (g1_root_pose) if present — the robot stands a little
        # closer/leans so its shorter arms reach the (unchanged) hand/object keypoints.
        fps = _PARAHOME_FPS   # source rate for finite-diff velocities/contact map (NOT cfg.ref_dt,
        #                       which is mutated to the control rate by the resample block below —
        #                       deriving from it would break re-init idempotency, see resample block)
        T = d["body_global_transform"].astype(np.float32)      # (F,4,4)
        self._np_root_pos = T[:, :3, 3]                        # (F,3)
        Rt = torch.from_numpy(T[:, :3, :3])
        self._np_root_quat = math_utils.quat_from_matrix(Rt).numpy().astype(np.float32)  # (F,4) wxyz

        def _recompute_root_vel():
            self._np_root_linvel = np.zeros_like(self._np_root_pos)
            self._np_root_linvel[1:] = (self._np_root_pos[1:] - self._np_root_pos[:-1]) * fps
            q = torch.from_numpy(np.ascontiguousarray(self._np_root_quat))
            dq = math_utils.quat_mul(q[1:], math_utils.quat_conjugate(q[:-1]))
            aa = math_utils.axis_angle_from_quat(_canon(dq)) * fps
            self._np_root_angvel = np.zeros_like(self._np_root_pos)
            self._np_root_angvel[1:] = aa.numpy().astype(np.float32)

        self._recompute_root_vel = _recompute_root_vel
        _recompute_root_vel()

        # fingertip pad targets (F,10,3): L[th,ff,mf,rf,lf] + R[...]
        self._np_ft_pad = d["fingertip_pad_pos"].astype(np.float32)

        # active object (first obj__*__base found). Articulated parts collected in order.
        base_keys = [k for k in d.files if k.startswith("obj__") and k.endswith("__base")]
        self._obj_name = base_keys[0].split("__")[1] if base_keys else ""
        if self._obj_name:
            self._np_obj_base = d[f"obj__{self._obj_name}__base"].astype(np.float32)   # (F,7) pos+quat
            part_keys = sorted(k for k in d.files
                               if k.startswith(f"dof__{self._obj_name}__"))
            self._np_obj_dof = (np.stack([d[k].astype(np.float32) for k in part_keys], axis=-1)
                                if part_keys else np.zeros((F, 0), np.float32))        # (F,P)
        else:
            self._np_obj_base = np.zeros((F, 7), np.float32); self._np_obj_base[:, 3] = 1.0
            self._np_obj_dof = np.zeros((F, 0), np.float32)

        # 맥락 물체(ctx__<obj>__base): 조작 물체 궤적의 XY 근처(< context_radius)에 있는 비조작 물체.
        # 물체를 받치도록 _setup_scene 에서 kinematic 으로 고정 스폰한다. 정적이라 frame-0 자세만 쓴다.
        self._ctx_spawn: list = []                                        # [(name, pose7 wxyz)]
        if self._obj_name and cfg.freeze_inactive_objects:
            act_xy = self._np_obj_base[:, :2]                             # (F,2) active swept path
            act0 = self._np_obj_base[0]                                   # (7,) active frame-0
            cands = []                                                    # (name, pose0, dmin)
            for k in (kk for kk in d.files if kk.startswith("ctx__") and kk.endswith("__base")):
                pose0 = d[k][0].astype(np.float32)                        # (7,) pos+quat wxyz, frame-0
                dmin = float(np.linalg.norm(act_xy - pose0[None, :2], axis=1).min())
                cands.append((k.split("__")[1], pose0, dmin))
            keep = {n for n, p, dm in cands if dm < cfg.context_radius}   # collision neighbours on the path
            # SUPPORT safety net: the nearest object whose frame-0 centroid is BELOW the active object
            # (within context_support_radius) — always included even if its centroid falls outside
            # context_radius (a large support's centroid can be offset from where the object rests).
            below = [(float(np.linalg.norm(act0[:2] - p[:2])), n) for n, p, dm in cands
                     if p[2] < act0[2] and float(np.linalg.norm(act0[:2] - p[:2])) < cfg.context_support_radius]
            if below:
                keep.add(min(below)[1])
            self._ctx_spawn = [(n, p) for n, p, dm in cands if n in keep]

        # 링크별 접촉 맵(hand_contact.npz): 32링크 mask / 법선 / 목표점. force 보상, 손끝 kpt 보상, delta_ft_obj 관측이
        # 모두 이것 하나를 읽는다. 목표점은 물체 로컬로 바꿔 둔다. 파일이 없으면 0 (접촉 항 비활성).
        self._np_link_contact_mask = np.zeros((F, N_LINK_CONTACT), np.float32)
        self._np_link_contact_normal = np.zeros((F, N_LINK_CONTACT, 3), np.float32)   # object-local
        self._np_link_contact_target = np.zeros((F, N_LINK_CONTACT, 3), np.float32)   # object-local
        self._has_link_contact = False
        hc_path = os.path.join(clip_dir, "hand_contact.npz")
        if self._obj_name and os.path.exists(hc_path):
            hc = np.load(hc_path, allow_pickle=True)
            if "normal" in hc.files:                                   # Option-A hand_contact (has normals)
                hln = {str(n): i for i, n in enumerate(hc["link_names"])}
                hmask = hc["mask"]; hnrm = hc["normal"]; htgt = hc["target"]   # target is WORLD
                tgt_w = np.zeros((F, N_LINK_CONTACT, 3), np.float32)
                for j, n in enumerate(LINK_CONTACT_NAMES):
                    k = hln.get(n)
                    if k is None:
                        continue
                    self._np_link_contact_mask[:, j] = hmask[:, k]
                    self._np_link_contact_normal[:, j] = hnrm[:, k]
                    tgt_w[:, j] = htgt[:, k]
                # target WORLD → object-LOCAL (pose-invariant): t_local = (t_world - obj_pos) @ R_obj
                _oqm = math_utils.matrix_from_quat(torch.from_numpy(self._np_obj_base[:, 3:7].astype(np.float32)))  # (F,3,3)
                _diff = torch.from_numpy(tgt_w) - torch.from_numpy(self._np_obj_base[:, :3].astype(np.float32))[:, None, :]
                self._np_link_contact_target = torch.einsum("flj,fjk->flk", _diff, _oqm).numpy().astype(np.float32)
                self._has_link_contact = True

        # optional per-frame retargeted G1 joints (seed reset pose if present)
        rt = os.path.join(cfg.dataset_root, cfg.retarget_subdir, cfg.clip_class,
                          os.path.basename(os.path.dirname(clip_dir)), "0", cfg.retarget_file)
        self._np_ref_joints = None
        self._np_ref_palm_quat = None
        self._ref_joint_names: list[str] | None = None
        if os.path.exists(rt):
            rd = np.load(rt, allow_pickle=True)
            if "g1_joint_pos" in rd.files:
                self._np_ref_joints = rd["g1_joint_pos"].astype(np.float32)           # (F,65)
            # [ROLLBACK MARKER: retarget-joint-order] the column layout g1_joint_pos was WRITTEN in.
            # Newer retarget runs record it in the npz; older ones do not, and for those the layout is
            # whatever g1_shadow_joint_order.json held at solve time. See _post_init_buffers.
            if "joint_names" in rd.files:
                self._ref_joint_names = [str(x) for x in rd["joint_names"]]
            if "g1_root_pose" in rd.files:
                # adjusted robot root (pos + quat wxyz) → becomes the robot's reference root
                rp = rd["g1_root_pose"].astype(np.float32)                            # (F,7)
                self._np_root_pos = rp[:, :3]
                self._np_root_quat = rp[:, 3:7]
                self._recompute_root_vel()
            if "g1_palm_quat" in rd.files:
                # reference palm/wrist orientation per hand [L,R] wxyz (Kabsch palm pose = the
                # robot0_{l,r}_palm body frame) → wrist-rotation termination gate.
                self._np_ref_palm_quat = rd["g1_palm_quat"].astype(np.float32)        # (F,2,4)

        # ---- SONIC SMPL encoder arrays (already resampled to control_fps by parahome_smpl_for_sonic) ----
        # Sibling of the retarget npz. smpl_joints_local(N,72) + root_q_zb(N,4 wxyz) + wrist_ref(N,6).
        # Loaded here (numpy); moved to device + length-asserted in _post_init_buffers.
        self._np_sonic_smpl = None
        if getattr(cfg, "use_sonic", True):
            sonic_npz = os.path.join(os.path.dirname(rt), cfg.sonic_smpl_file)
            if not os.path.exists(sonic_npz):
                raise FileNotFoundError(
                    f"SONIC SMPL arrays not found: {sonic_npz}. Run "
                    f"scripts/process_dataset/dataset/parahome_smpl_for_sonic.py --clip <clip> first.")
            sd = np.load(sonic_npz)
            self._np_sonic_smpl = {k: np.asarray(sd[k], np.float32)
                                   for k in ("smpl_joints_local", "root_q_zb", "wrist_ref")}

        # ---- 프레임별 레퍼런스를 30 fps → control_fps 로 리샘플 (SONIC 50 Hz: 제어 1스텝 = 레퍼런스 1프레임).
        #      위는 원본 fps 로 계산했다. 여기서 루트 속도를 다시 구하고 cfg.ref_dt 도 갱신한다. ----
        src_fps = _PARAHOME_FPS   # ParaHome native rate (constant, NOT cfg.ref_dt — see note above:
        #                           cfg.ref_dt is mutated to 1/tgt_fps at the end, so deriving src
        #                           from it would skip the resample on a re-init with the same cfg).
        tgt_fps = float(getattr(cfg, "control_fps", 1.0 / cfg.ref_dt))
        if abs(tgt_fps - src_fps) > 1e-6:
            dur = (F - 1) / src_fps
            N = int(round(dur * tgt_fps)) + 1
            t_src = np.arange(F) / src_fps
            t_tgt = np.linspace(0.0, dur, N)

            def _rl(x):                                                  # linear (F,...)->(N,...)
                x = np.asarray(x, np.float32)
                o = np.stack([np.interp(t_tgt, t_src, x.reshape(F, -1)[:, d])
                              for d in range(x.reshape(F, -1).shape[1])], axis=1)
                return o.reshape((N,) + x.shape[1:]).astype(np.float32)

            def _rq(q):                                                  # slerp (F,4)/(F,K,4)->(N,...)
                from scipy.spatial.transform import Rotation, Slerp
                q = np.asarray(q, np.float32); qf = q.reshape(F, -1, 4); outs = []
                for k in range(qf.shape[1]):
                    r = Rotation.from_quat(qf[:, k][:, [1, 2, 3, 0]])    # wxyz->xyzw
                    outs.append(Slerp(t_src, r)(t_tgt).as_quat()[:, [3, 0, 1, 2]])
                return np.stack(outs, axis=1).reshape((N,) + q.shape[1:]).astype(np.float32)

            self._np_ref_kpts = _rl(self._np_ref_kpts)
            self._np_root_pos = _rl(self._np_root_pos)
            self._np_root_quat = _rq(self._np_root_quat)
            self._np_ft_pad = _rl(self._np_ft_pad)
            self._np_obj_base = np.concatenate(
                [_rl(self._np_obj_base[:, :3]), _rq(self._np_obj_base[:, 3:7])], axis=1)
            self._np_obj_dof = (_rl(self._np_obj_dof) if self._np_obj_dof.shape[1] > 0
                                else np.zeros((N, 0), np.float32))
            self._np_ref_foot_contact = (_rl(self._np_ref_foot_contact) > 0.5).astype(np.float32)
            if self._has_link_contact:
                self._np_link_contact_mask = (_rl(self._np_link_contact_mask) > 0.5).astype(np.float32)
                ln = _rl(self._np_link_contact_normal)                 # (N,L,3) resampled reaction normal
                lnn = np.linalg.norm(ln, axis=-1, keepdims=True)
                self._np_link_contact_normal = np.where(lnn > 1e-6, ln / np.clip(lnn, 1e-6, None), ln).astype(np.float32)
                self._np_link_contact_target = _rl(self._np_link_contact_target)   # (N,L,3) object-local target
            if self._np_ref_joints is not None:
                self._np_ref_joints = _rl(self._np_ref_joints)
            if self._np_ref_palm_quat is not None:
                self._np_ref_palm_quat = _rq(self._np_ref_palm_quat)
            # root velocity at the NEW rate (finite diff of the resampled root; quat-log for angvel)
            self._np_root_linvel = np.zeros_like(self._np_root_pos)
            self._np_root_linvel[1:] = (self._np_root_pos[1:] - self._np_root_pos[:-1]) * tgt_fps
            _q = torch.from_numpy(np.ascontiguousarray(self._np_root_quat))
            _dq = math_utils.quat_mul(_q[1:], math_utils.quat_conjugate(_q[:-1]))
            self._np_root_angvel = np.zeros_like(self._np_root_pos)
            self._np_root_angvel[1:] = (math_utils.axis_angle_from_quat(_canon(_dq))
                                        * tgt_fps).numpy().astype(np.float32)
            cfg.ref_dt = 1.0 / tgt_fps                                   # runtime rate (foot-contact heuristic, etc.)
            F = N

        self._ref_len = int(F)
        self._n_obj_parts = int(self._np_obj_dof.shape[1])
        self._load_stage1_hand(cfg, clip_dir)      # [ROLLBACK MARKER: stage1-hand]

        # 물체 레퍼런스 속도(리샘플한 _np_obj_base 의 유한차분). RSI 로 중간에서 시작할 때 물체를 이 속도로 움직여 둔다.
        _ofps = 1.0 / cfg.ref_dt
        self._np_obj_linvel = np.zeros((self._ref_len, 3), np.float32)
        self._np_obj_linvel[1:] = (self._np_obj_base[1:, :3] - self._np_obj_base[:-1, :3]) * _ofps
        _oqv = torch.from_numpy(np.ascontiguousarray(self._np_obj_base[:, 3:7]))
        _odq = math_utils.quat_mul(_oqv[1:], math_utils.quat_conjugate(_oqv[:-1]))
        self._np_obj_angvel = np.zeros((self._ref_len, 3), np.float32)
        self._np_obj_angvel[1:] = (math_utils.axis_angle_from_quat(_canon(_odq)) * _ofps).numpy().astype(np.float32)

        # Point the ENV-FIXED viewer (cfg.viewer.origin_type="env") at the reference-root centroid
        # (env-local) so the stable camera frames the robot wherever this clip places it in the
        # ParaHome world. Runs before super().__init__ consumes cfg.viewer, so the override takes.
        if getattr(cfg, "viewer", None) is not None and cfg.viewer.origin_type == "env":
            cx = float(self._np_root_pos[:, 0].mean()); cy = float(self._np_root_pos[:, 1].mean())
            # CLIP-ADAPTIVE vertical framing: fit ground(feet)→highest point of the motion (head, hands,
            # or a LIFTED object) so nothing leaves the top of the env-fixed frame while the feet stay in
            # view. Was fixed at lookat 0.72 which cut off a raised object (e.g. pan lifted to z≈1.4).
            tops = [float(self._np_root_pos[:, 2].max()) + 0.75]                    # head ≈ root + 0.75
            ftp = getattr(self, "_np_ft_pad", None)
            if ftp is not None:
                tops.append(float(np.asarray(ftp)[:, :, 2].max()))                 # highest hand
            objb = getattr(self, "_np_obj_base", None)
            if objb is not None and float(np.asarray(objb)[:, 2].max()) > 0.1:
                tops.append(float(np.asarray(objb)[:, 2].max()))                   # highest object
            z_top = max(tops) + 0.15; extent = z_top
            lookat_z = 0.5 * z_top
            zoom = float(getattr(cfg, "viewer_zoom", 1.0))
            off = max(1.5, extent * 1.25) * zoom
            # camera angle matched to render_retarget.py (same viewpoint for train videos + playbacks):
            # aim at the OBJECT centroid (hands, un-occluded by the torso), azimuth viewer_yaw, elevation
            # viewer_elev. horiz = off·√2 so yaw=45/elev≈0 reproduces the old (cx+off, cy+off) root view.
            objb = getattr(self, "_np_obj_base", None)
            look_obj = bool(getattr(cfg, "viewer_look_obj", False)) and objb is not None \
                and float(np.asarray(objb)[:, 2].max()) > 0.1
            tx, ty = (float(np.asarray(objb)[:, 0].mean()), float(np.asarray(objb)[:, 1].mean())) if look_obj else (cx, cy)
            horiz = off * (2 ** 0.5)
            az = math.radians(float(getattr(cfg, "viewer_yaw", 45.0)))
            elev = float(getattr(cfg, "viewer_elev", 0.0))
            zoff = horiz * math.tan(math.radians(elev)) if elev > 0.0 else 0.12 * extent
            cfg.viewer.lookat = (tx, ty, lookat_z)
            cfg.viewer.eye = (tx + horiz * math.cos(az), ty + horiz * math.sin(az), lookat_z + zoff)

    def _build_object_cfg(self, cfg) -> None:
        """Resolve the active object's converted USD; spawn only if it exists (else robot-only)."""
        self._object_cfg = None
        if not self._obj_name:
            return
        # ParaHome converter writes assets/objects/<obj>/<obj>.usd (parahome_convert_obj_to_usd.py),
        # e.g. objects/pan/pan.usd — resolve on that stem (there is no "object.usd" writer).
        usd = os.path.join(cfg.dataset_root, "assets", "objects", self._obj_name, f"{self._obj_name}.usd")
        if not os.path.exists(usd):
            return   # converted USD not built yet → object stays inert (robot-only kinematic path)
        p0 = self._np_obj_base[0]
        self._object_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd, activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=8, solver_velocity_iteration_count=4,
                    # [ROLLBACK MARKER: obj-depen-vel] 겹침 해소 속도 상한. 0.1 로 낮췄다가(2026-08-14)
                    # 2026-09-17 에 로봇과 함께 1.0 으로 되돌렸다. 물체 USD 의 1.0 은 이 spawn 설정이 덮어쓴다.
                    max_depenetration_velocity=1.0),
                # Recolor the manipulated object a vivid orange so it stands out from the gray robot /
                # scene furniture in the viewer/video. visual_material is created + bound to the loaded
                # USD's geometry (UsdFileCfg/FileCfg feature); does not affect the physics material.
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.35, 0.0), roughness=0.6, metallic=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(float(p0[0]), float(p0[1]), float(p0[2])),
                rot=(float(p0[3]), float(p0[4]), float(p0[5]), float(p0[6]))),
        )

    # ------------------------------------------------------------------ scene
    # ── [ROLLBACK MARKER: cws-rc-mesh] 논문과 같은 물체 크기 정의 (2026-09-02) ──────────────
    def _object_mesh_radius(self) -> float | None:
        """메시 정점 중심에서 가장 먼 정점까지의 거리 (m). 못 읽으면 None.

        논문 공개 구현 scripts/retarget/soma_to_g1.py:142 _compute_mesh_radius 와 같은 정의입니다
        (process_arctic_grab.py:52 도 동일). 렌치의 회전 성분을 이 값으로 나눕니다.

        이전에는 "접촉점 노름의 0.9 분위"를 썼는데, 그건 물체 크기가 아니라 손이 닿는 범위입니다.
        칼 실측으로 논문 정의 13.11 cm vs 0.9 분위 5.60 cm = 2.34 배 차이가 나고, rc 가 토크를
        나누므로 우리 쪽이 그만큼 토크에 민감했습니다. cws_v(=0.1)를 논문 값으로 가져오려면 sigma
        스케일도 논문과 같아야 하므로 정의를 맞춥니다.

        스폰되는 <obj>.usd 를 먼저 보고, 거기서 메시를 못 찾으면 참조 대상인
        Props/instanceable_meshes.usd 를 봅니다 (ParaHome 변환기가 지오메트리를 그쪽에 둡니다).
        """
        try:
            from pxr import Usd, UsdGeom
        except Exception:
            return None
        base = os.path.join(self.cfg.dataset_root, "assets", "objects", self._obj_name)
        for path in (os.path.join(base, f"{self._obj_name}.usd"),
                     os.path.join(base, "Props", "instanceable_meshes.usd")):
            if not os.path.exists(path):
                continue
            try:
                stage = Usd.Stage.Open(path)
                pts_all = []
                # TraverseAll: 참조/인스턴스 프록시 안의 메시까지 봅니다(Traverse 는 건너뜁니다).
                for prim in stage.TraverseAll():
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                    if pts is None or len(pts) == 0:
                        continue
                    arr = np.asarray(pts, dtype=np.float64)
                    try:    # 로컬 xform(스케일 포함)이 있으면 반영
                        M = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default())).T
                        arr = (M[:3, :3] @ arr.T).T + M[:3, 3]
                    except Exception:
                        pass
                    pts_all.append(arr)
                if not pts_all:
                    continue
                V = np.concatenate(pts_all, axis=0)
                return float(np.linalg.norm(V - V.mean(axis=0, keepdims=True), axis=1).max())
            except Exception:
                continue
        return None
    # ── [/ROLLBACK MARKER: cws-rc-mesh] ────────────────────────────────────────────────────

    # ── [ROLLBACK MARKER: stage1-hand] 1단계(떠 있는 양손) 롤아웃 로드 (2026-09-09) ─────────────
    def _load_stage1_hand(self, cfg, clip_dir: str) -> None:
        """hand_traj_best.npz(1단계 롤아웃) 를 레퍼런스 시간축(N=_ref_len)에 맞춰 numpy 로 들고 있습니다.
        cfg 의 hand_kpt_from_hand_pretrain / sonic_hand_residual_base 참조. 파일이 없으면 self._np_s1 = None."""
        self._np_s1 = None
        self._stage1_contact_map_loaded = False   # [stage1-contact-map]
        self._has_link_contact_human = False      # [cws-human-ref] 사람 맵 보관 여부 (1단계 맵으로 교체될 때 채움)
        want_kpt = bool(getattr(cfg, "hand_kpt_from_hand_pretrain", False))
        want_base = (bool(getattr(cfg, "sonic_hand_residual", False))
                     and str(getattr(cfg, "sonic_hand_residual_base", "reference")).startswith("hand_pretrain"))
        if not (want_kpt or want_base):
            return
        clip = os.path.basename(os.path.dirname(clip_dir))
        base_dir = os.path.join(cfg.dataset_root, cfg.hand_pretrain_subdir, cfg.clip_class, clip, "0")
        cands = [os.path.join(base_dir, cfg.hand_pretrain_traj_file)]
        cands += sorted(glob.glob(os.path.join(base_dir, "evaluation_*", cfg.hand_pretrain_traj_file)),
                        key=os.path.getmtime, reverse=True)
        path = next((p for p in cands if os.path.exists(p)), None)
        if path is None:
            print(f"[stage1-hand] 1단계 롤아웃 파일이 없습니다 ({base_dir}/[evaluation_*/]{cfg.hand_pretrain_traj_file}) "
                  f"— 손 목표는 SMPL-X 그대로, 잔차 기준은 레퍼런스로 둡니다.")
            return
        hd = np.load(path, allow_pickle=True)
        N = int(self._ref_len)
        fps_env = 1.0 / float(cfg.ref_dt)
        s1_fps = float(hd["control_fps"])
        fr = np.asarray(hd["frame"], np.float64)                   # 기록된 행마다의 레퍼런스 프레임(롤아웃 시간축)
        assert np.all(np.diff(fr) == 1.0), "[stage1-hand] frame 열이 연속이 아닙니다"
        # 레퍼런스 프레임 i (fps_env) 의 시각 = 롤아웃 프레임 i*s1_fps/fps_env; 행 = 프레임 - fr[0]
        k = np.arange(N) * (s1_fps / fps_env) - fr[0]
        valid = (k >= 0.0) & (k <= fr[-1] - fr[0])
        S = {"path": path, "valid": valid.astype(np.float32)}
        S["palm_pos_raw"] = _s1_resample(hd["palm_pos"], k)         # (N,2,3) [L,R] env-local (ParaHome 좌표)
        S["palm_quat_raw"] = _s1_nlerp(hd["palm_quat"], k)          # (N,2,4) wxyz = robot0_{l,r}_palm 바디 프레임
        # 손 관절 (손목 6자유도 제외, J0 포함) — 이름으로 골라 두고 _apply_stage1_hand_kpts 에서 로봇 관절에 맞춤
        qa = np.asarray(hd["joint_pos_all"], np.float32)            # (T,2,28)
        names_l = [str(n) for n in hd["joint_names_all_l"]]
        names_r = [str(n) for n in hd["joint_names_all_r"]]
        cols_l = [i for i, n in enumerate(names_l) if "wrist" not in n]
        cols_r = [i for i, n in enumerate(names_r) if "wrist" not in n]
        S["hand_q_names"] = [names_l[i] for i in cols_l] + [names_r[i] for i in cols_r]
        S["hand_q"] = _s1_resample(np.concatenate([qa[:, 0, cols_l], qa[:, 1, cols_r]], -1), k)   # (N,44)
        # 잔차 기준 후보 (롤아웃 액션 순서; _setup_stage1_hand 에서 이 env 의 손 액션 순서로 재배열)
        S["base_names"] = [str(n) for n in hd["joint_names"]]
        S["finger_target"] = _s1_resample(hd["finger_target"], k)   # (N,36) 1단계 정책의 손가락 PD 목표
        S["finger_qpos"] = _s1_resample(hd["finger_qpos"], k)       # (N,36) 실측 관절각
        S["ft_names"] = [str(n) for n in hd["fingertip_body_names"]]
        S["ft_pos"] = _s1_resample(hd["ft_pos"], k)                 # (N,10,3) 기록된 손끝(pad) — USD 정합 검사용
        # ── [ROLLBACK MARKER: stage1-vertex] 1단계 pad 를 1단계 물체 좌표로 — 접촉 프레임 손끝 목표용 (2026-09-09).
        #    물체 좌표로 쓰기 때문에 "1단계 손 대비 물체" 관계가 정확하다.
        S["ft_obj_local"] = None
        if bool(getattr(cfg, "hand_pretrain_contact_vertex", False)) and self._obj_name and "obj_pos" in hd.files:
            _op = torch.from_numpy(_s1_resample(hd["obj_pos"], k)).float()                     # (N,3) 1단계 물체
            _oq = torch.from_numpy(_s1_nlerp(hd["obj_quat"], k)).float()                       # (N,4)
            _ft = torch.from_numpy(S["ft_pos"]).float()                                          # (N,10,3)
            _qc = math_utils.quat_conjugate(_oq).unsqueeze(1).expand(-1, _ft.shape[1], -1)
            S["ft_obj_local"] = math_utils.quat_apply(_qc, _ft - _op.unsqueeze(1)).numpy().astype(np.float32)
        # [/ROLLBACK MARKER: stage1-vertex]
        # [ROLLBACK MARKER: stage1-contact-map] 32링크 접촉 맵을 1단계 롤아웃 파일로 교체 (2026-09-09). 선형 혼합 없이
        # 최근접 행을 쓰고, target 이 "world" 면 1단계 물체 자세로 변환한다. 로드되면 hand_pretrain_contact_vertex 는 건너뛴다.
        if bool(getattr(cfg, "hand_pretrain_contact_map", False)) and want_kpt and self._obj_name:
            mp = os.path.join(os.path.dirname(path), str(getattr(cfg, "hand_pretrain_contact_map_file", "hand_contact_stage1.npz")))
            cm = np.load(mp, allow_pickle=True) if os.path.exists(mp) else None
            if cm is None:
                print(f"[stage1-contact-map] 파일이 없습니다 ({mp}) — 사람 접촉 맵(hand_contact.npz)"
                      f"{' + hand_pretrain_contact_vertex' if S.get('ft_obj_local') is not None else ''} 로 둡니다.")
            elif not all(x in cm.files for x in ("link_names", "mask", "normal", "target")):
                print(f"[stage1-contact-map] 스키마 불일치 ({mp}: {list(cm.files)}) — 사람 접촉 맵으로 둡니다.")
            else:
                fr_m = np.asarray(cm["frame"], np.float64) if "frame" in cm.files else fr.copy()
                cmask = np.asarray(cm["mask"], np.float32)                                  # (T_m,L_m)
                cnrm = np.asarray(cm["normal"], np.float32)                                 # (T_m,L_m,3) 물체 로컬
                ctgt = np.asarray(cm["target"], np.float32)                                 # (T_m,L_m,3)
                coord = str(cm["coord"]) if "coord" in cm.files else "world"
                ok = (len(fr_m) == cmask.shape[0] == cnrm.shape[0] == ctgt.shape[0])
                if not ok:
                    print(f"[stage1-contact-map] frame({len(fr_m)}) / mask({cmask.shape[0]}) / normal({cnrm.shape[0]}) / "
                          f"target({ctgt.shape[0]}) 길이가 다릅니다 — 사람 접촉 맵으로 둡니다.")
                elif coord != "object" and "obj_pos" not in hd.files:
                    ok = False
                    print("[stage1-contact-map] target 이 월드 좌표인데 롤아웃에 obj_pos 가 없습니다 — 사람 접촉 맵으로 둡니다.")
                elif coord != "object":
                    hr = np.clip(np.rint(fr_m - fr[0]).astype(int), 0, len(fr) - 1)                # 맵 행 → 롤아웃 행
                    _op = torch.from_numpy(np.asarray(hd["obj_pos"], np.float32)[hr])                # (T_m,3) 1단계 물체
                    _oq = torch.from_numpy(np.asarray(hd["obj_quat"], np.float32)[hr])               # (T_m,4)
                    _qc = math_utils.quat_conjugate(_oq).unsqueeze(1).expand(-1, ctgt.shape[1], -1)
                    ctgt = math_utils.quat_apply(_qc, torch.from_numpy(ctgt) - _op.unsqueeze(1)).numpy()
                    ctgt = np.where(cmask[..., None] > 0.5, ctgt, 0.0).astype(np.float32)            # 비접촉 행은 0 유지
                if ok:
                    k_m = np.arange(N) * (s1_fps / fps_env) - fr_m[0]
                    rows = np.clip(np.rint(k_m).astype(int), 0, len(fr_m) - 1)                      # 최근접 행
                    mln = {str(n): i for i, n in enumerate(cm["link_names"])}
                    new_mask = np.zeros((N, N_LINK_CONTACT), np.float32)
                    new_nrm = np.zeros((N, N_LINK_CONTACT, 3), np.float32)
                    new_tgt = np.zeros((N, N_LINK_CONTACT, 3), np.float32)
                    miss = []
                    for j, n in enumerate(LINK_CONTACT_NAMES):
                        i = mln.get(n)
                        if i is None:
                            miss.append(n)
                            continue
                        new_mask[:, j] = cmask[rows, i]
                        new_nrm[:, j] = cnrm[rows, i]
                        new_tgt[:, j] = ctgt[rows, i]
                    _nn = np.linalg.norm(new_nrm, axis=-1, keepdims=True)
                    new_nrm = np.where(_nn > 1e-6, new_nrm / np.clip(_nn, 1e-6, None), new_nrm).astype(np.float32)
                    # 진단: 사람 맵(이미 N 프레임) 대비 마스크 비교
                    old_mask = (self._np_link_contact_mask > 0.5) if self._has_link_contact else np.zeros(new_mask.shape, bool)
                    if old_mask.shape != new_mask.shape:
                        old_mask = np.zeros(new_mask.shape, bool)
                    nm = new_mask > 0.5
                    _dist = np.array([n.endswith("distal") for n in LINK_CONTACT_NAMES])

                    def _iou(a, b):
                        u = int((a | b).sum())
                        return float((a & b).sum() / u) if u > 0 else float("nan")

                    # [cws-human-ref] 교체 직전의 사람 맵(N 프레임)을 보관 — CWS σ_h 는 이것으로 계산할 수 있다
                    self._np_link_contact_mask_human = self._np_link_contact_mask
                    self._np_link_contact_normal_human = self._np_link_contact_normal
                    self._np_link_contact_target_human = self._np_link_contact_target
                    self._has_link_contact_human = bool(self._has_link_contact) and old_mask.shape == new_mask.shape
                    self._np_link_contact_mask = new_mask
                    self._np_link_contact_normal = new_nrm
                    self._np_link_contact_target = new_tgt
                    self._has_link_contact = True
                    self._stage1_contact_map_loaded = True
                    ns = str(cm["normal_source"]) if "normal_source" in cm.files else "?"
                    print(f"[stage1-contact-map] {mp}\n"
                          f"    {len(fr_m)} 행 @ {s1_fps:.0f} Hz → 레퍼런스 {N} 프레임 (최근접 행, target coord={coord}, "
                          f"normal_source={ns}); 접촉 프레임 {int(nm.any(1).sum())}/{N} (사람 맵 {int(old_mask.any(1).sum())}); "
                          f"프레임당 접촉 링크 {nm.sum(1).mean():.1f} (사람 {old_mask.sum(1).mean():.1f})\n"
                          f"    사람 맵과 마스크 IoU 전체 {_iou(nm, old_mask):.2f} / 손끝 {_iou(nm[:, _dist], old_mask[:, _dist]):.2f}"
                          + (f"; 파일에 없는 링크 {miss}" if miss else ""))
                    if S.get("ft_obj_local") is not None:
                        S["ft_obj_local"] = None
                        print("[stage1-contact-map] hand_pretrain_contact_vertex 는 건너뜁니다 (맵이 손끝 행을 포함).")
        # ── [/ROLLBACK MARKER: stage1-contact-map] ──
        S["palm_pos"], S["palm_quat"] = S["palm_pos_raw"].copy(), S["palm_quat_raw"].copy()
        self._np_s1 = S
        _v = int(valid.sum())
        print(f"[stage1-hand] {path}\n"
              f"    롤아웃 {len(fr)} 행 @ {s1_fps:.0f} Hz → 레퍼런스 {N} 프레임 @ {fps_env:.0f} Hz "
              f"(유효 {_v}, 마지막 값 유지 {N - _v})")
    # ── [/ROLLBACK MARKER: stage1-hand] ──

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self._object = None
        if self._object_cfg is not None:
            self._object = RigidObject(self._object_cfg)
            self.scene.rigid_objects["object"] = self._object

        # 링크별 접촉 센서: 32 wrap 링크마다 하나, 조작 물체로 필터(맥락 물체·자기 충돌 제외). history_length=1.
        # update_period 를 제어 주기로 둬 버퍼 해제를 decimation 배 줄인다(PhysX 는 매 물리 스텝 계산하므로 동작 동일).
        _ctrl_dt = self.cfg.sim.dt * self.cfg.decimation                 # control period (s) = 1/50
        self._link_contact_sensors: list[ContactSensor] = []
        obj_filter = ["/World/envs/env_.*/Object"] if self._object_cfg is not None else []
        for name in LINK_CONTACT_NAMES:
            s = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{name}",
                filter_prim_paths_expr=obj_filter, history_length=1, update_period=_ctrl_dt,
                # [ROLLBACK MARKER: cws-contact] contact_pos_w = 링크-물체 접촉점 평균(월드), CWS 렌치의 모멘트 팔.
                # 접촉이 없는 쌍은 NaN 이므로 반드시 마스크로 거른다.
                track_air_time=False, track_contact_points=bool(self.cfg.track_contact_points),
                # ParaHome objects are CONVEX-DECOMPOSITION colliders (many sub-hulls) → a link can touch
                # several at once → >4 manifold points → raise the contact-data buffer cap (else a HARD
                # device-side assert in ContactSensor._unpack_contact_buffer_data).
                max_contact_data_count_per_prim=self.cfg.ft_max_contact_points))
            self._link_contact_sensors.append(s)
            self.scene.sensors[f"linkc_{name}"] = s

        # foot contact sensors (2) for the feet-contact-match reward. FILTERED on the ground so force_matrix_w
        # reports ONLY the foot↔ground contact force. Ground is a single flat plane → ≤4 manifold points → no
        # buffer-overflow (default cap). history_length=1. Ordered [left, right] to match _ref_foot_contact.
        self._foot_sensors: list[ContactSensor] = []
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            s = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{name}",
                filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],   # actual ground collider prim
                history_length=1, update_period=_ctrl_dt, track_air_time=False, track_contact_points=False))
            self._foot_sensors.append(s)
            self.scene.sensors[f"foot_{name}"] = s

        # 맥락 물체를 clone_environments 전에 kinematic 고정 collider 로 스폰한다(RigidObject 아님 → GPU 뷰 없음).
        # prim 은 env_.* 바로 아래 Ctx_<i>_<name> (중간 scope 는 clone 실패). "Object" 가 아니라 접촉 필터에서 빠진다.
        self._ctx_prims: list = []
        for i, (name, pose0) in enumerate(getattr(self, "_ctx_spawn", [])):
            # Prefer the STATIC-collision context USD (<obj>_ctx.usd, base.obj → single decomp collider,
            # no articulation) if built; fall back to the full <obj>.usd (fine for rigid objects, but for
            # articulated furniture that is a live articulation — build _ctx.usd for those).
            base = os.path.join(self.cfg.dataset_root, "assets", "objects", name)
            # ctx/ subdirectory first: the context collider is cooked there so it does not share
            # Props/instanceable_meshes.usd with the manipulated object's USD (see convert_context).
            # The flat path is the pre-split layout, kept so old asset trees still load.
            usd_ctx = os.path.join(base, "ctx", f"{name}_ctx.usd")
            if not os.path.exists(usd_ctx):
                usd_ctx = os.path.join(base, f"{name}_ctx.usd")
            usd = usd_ctx if os.path.exists(usd_ctx) else os.path.join(base, f"{name}.usd")
            if not os.path.exists(usd):
                continue
            # NOTE: no collision_props override — the converted USD already authors collision on the
            # (instanceable) mesh; overriding it here only emits a benign "modify_collision_properties
            # on an instanced prim" warning and no-ops. Only the kinematic freeze needs overriding.
            ctx_spawn = sim_utils.UsdFileCfg(
                usd_path=usd, activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True))
            prim_path = f"/World/envs/env_.*/Ctx_{i}_{name}"
            ctx_spawn.func(prim_path, ctx_spawn,
                           translation=(float(pose0[0]), float(pose0[1]), float(pose0[2])),
                           orientation=(float(pose0[3]), float(pose0[4]), float(pose0[5]), float(pose0[6])))
            self._ctx_prims.append(prim_path)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # [ROLLBACK MARKER: retarget-joint-order] --------------------------------------------------
    def _remap_ref_joints(self, c) -> None:
        """Put the retargeted joints in the env's live action-joint order, matching BY NAME.

        `g1_joint_pos` is a bare (F,65) array: the retarget writes each solved joint into the column
        that `g1_shadow_joint_order.json` assigned to that NAME, and the env reads column k straight
        into its own k-th action joint. That is only correct while the two orders agree, and nothing
        enforced it — the json is a static dump of the robot's PhysX DOF order, so regenerating the
        robot USD silently repermutes the env side and leaves the json behind. It happened here:
        json dumped 2026-07-07 17:36, G1_shadow.usd rebuilt 2026-07-08 23:21, every retarget npz
        written afterwards. 24 of the 65 slots were crossed, all in the hands (MF<->TH and FF<->RF at
        J1/J2/J3, both hands); legs/waist/arms were untouched, which is why body tracking looked fine
        while the hands never worked. Concretely MFJ1 was being fed THJ2's value (~0.015 = straight)
        while the other fingers curled to ~0.85 — the middle finger stuck out in every rollout.

        _ref_joints is the base of the residual action target, the RSI reset pose, and the state-cache
        seed, so this silently poisoned all three.

        Matching by name removes the coupling entirely: newer retarget npz files carry `joint_names`,
        and for older ones the json is still the layout they were written with. Note the json alone
        cannot be re-dumped as a fix — that would only redefine the layout the OLD npz files are
        already keyed to.
        """
        if self._ref_joints is None:
            return
        src = self._ref_joint_names
        if src is None:                       # legacy npz: the json IS the layout it was written in
            jpath = os.path.join(os.path.dirname(_ROBOT_USD), "g1_shadow_joint_order.json")
            if not os.path.exists(jpath):
                print(f"[ref-joints] no joint_names in the npz and no {jpath} — order NOT verified.")
                return
            with open(jpath) as f:
                src = json.load(f)["action_joint_names"]
        jn = self.robot.data.joint_names
        env_order = [jn[i] for i in self._action_joint_ids]
        missing = [n for n in env_order if n not in src]
        if missing or len(src) != self._ref_joints.shape[1]:
            raise RuntimeError(
                f"[ref-joints] cannot map the retarget columns onto the env's action joints: "
                f"{len(missing)} name(s) absent from the source layout (e.g. {missing[:3]}), "
                f"source width {len(src)} vs g1_joint_pos width {self._ref_joints.shape[1]}. "
                f"Re-run scripts/process_dataset/retarget/retarget_g1_pyroki.py for this clip.")
        # [ROLLBACK MARKER: ref-j0] 리타게팅이 푼 손가락 J0 8개를 리셋에 쓴다 (2026-09-02). 텐던 축 J0 는 액션 관절이
        # 아니어서 이 값이 없으면 리셋이 J0 를 채울 수 없다 → 73열 리타게팅 npz 가 필수.
        _j0n = [f"robot0_{sd}_{fg}J0" for sd in "lr" for fg in ("FF", "MF", "RF", "LF")]
        _miss0 = [n for n in _j0n if n not in src]
        if _miss0:
            raise RuntimeError(f"[ref-j0] 리타게팅 npz 에 손가락 J0 열이 없습니다 (예: {_miss0[:2]}). "
                               f"scripts/process_dataset/retarget/retarget_g1_pyroki.py 로 이 클립을 다시 만드세요.")
        _c = torch.tensor([src.index(n) for n in _j0n], device=self.device, dtype=torch.long)
        self._ref_j0 = self._ref_joints[:, _c].clone()                       # (F,8)
        self._ref_j0_ids = torch.tensor([jn.index(n) for n in _j0n],
                                        device=self.device, dtype=torch.long)
        print(f"[ref-j0] 리타게팅이 푼 J0 8개 보존 — 중앙값 "
              f"{self._ref_j0.median(dim=0).values.cpu().numpy().round(3).tolist()}")
        # [/ROLLBACK MARKER: ref-j0] --------------------------------------------------------
        perm = torch.tensor([src.index(n) for n in env_order], device=self.device, dtype=torch.long)
        n_moved = int((perm != torch.arange(len(perm), device=self.device)).sum())
        self._ref_joints = self._ref_joints[:, perm]
        origin = "npz joint_names" if self._ref_joint_names is not None else "g1_shadow_joint_order.json"
        print(f"[ref-joints] retarget columns remapped by name from {origin}: "
              f"{n_moved}/{len(perm)} slots moved")
    # [/ROLLBACK MARKER: retarget-joint-order] -------------------------------------------------

    # [ROLLBACK MARKER: spawn-declear] ---------------------------------------------------------
    def _solve_spawn_declear(self) -> None:
        """Per-frame spawn lift that clears the object out of whatever it is resting inside.

        Solved against the live scene at startup rather than precomputed, so changing colliders,
        friction or contact offsets cannot leave a stale correction behind. Fills
        `self._obj_spawn_lift` (F,), which `_reset_idx` adds to the reference spawn height only.

        Measures DISPLACEMENT, not velocity. The previous test asked whether the object was moving
        slower than `declear_clear_v` one step after being pinned, and that cannot work: the
        threshold has to sit above free fall (a contact-free object reaches 9.81*dt = 0.049 m/s in
        one step), while a body climbing out of a support reports far less than the motion implies —
        measured on the pan, +5.2 mm in a single 5 ms step (1.04 m/s of travel) alongside a reported
        speed of 0.063 m/s. Depenetration moves the body without depositing the matching velocity, so
        every frame read as "clear" while sitting 19 mm inside its support, and no choice of
        threshold separates resting / floating / climbing out. Raising max_depenetration_velocity to
        20 m/s changed the trajectory by nothing at all (verified applied on the prim), so the cap is
        not the limiter either.

        Placing the object and letting it settle FREELY answers it directly: the height it settles at
        IS the height it should have spawned at, and the sign is unambiguous — positive means it was
        penetrating, ~0 means it was resting, negative means the reference floats above the support
        (not corrected here; lowering the object is a reference problem, not a spawn one). It also
        converges in a handful of steps (the pan is within 0.6 mm of final by step 5), so this
        replaces the old ceiling of 60 pinned probes per frame with a single settle.
        """
        c = self.cfg
        dev, n, F = self.device, self.num_envs, self._ref_len
        self._obj_spawn_lift = torch.zeros(F, device=dev)
        if not getattr(c, "object_spawn_declear", False) or not self._has_object:
            return

        # ---- where the reference holds the object still -------------------------------------
        fps = 1.0 / c.ref_dt
        lin = torch.zeros(F, device=dev)
        lin[1:] = (self._ref_obj_pos[1:] - self._ref_obj_pos[:-1]).norm(dim=-1) * fps
        dq = math_utils.quat_mul(self._ref_obj_quat[1:], math_utils.quat_conjugate(self._ref_obj_quat[:-1]))
        ang = torch.zeros(F, device=dev)
        ang[1:] = math_utils.axis_angle_from_quat(dq).norm(dim=-1) * fps
        lin[0], ang[0] = lin[1], ang[1]          # frame 0 has no predecessor; inherit, do not assume rest
        rest = (lin < c.declear_rest_lin) & (ang < c.declear_rest_ang)
        rest_idx = torch.nonzero(rest, as_tuple=False).flatten()
        if not len(rest_idx):
            print("[spawn-declear] reference never holds the object still — no frame corrected.")
            return

        # ---- 측정 동안 로봇을 치워 둔다(레퍼런스 손이 물체를 파고듦). floating base 라 계속 붙잡아 둔다 ----
        org = self.scene.env_origins
        park = torch.zeros(n, 7, device=dev)
        park[:, :3] = org + torch.tensor([0.0, 0.0, 5.0], device=dev)
        park[:, 3] = 1.0
        zero6 = torch.zeros(n, 6, device=dev)

        def _settle(fr: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
            """Spawn `fr` at reference+dz, let it settle FREELY, return settled height - reference.

            The object is written once and then left alone — pinning it every step is what made the
            old test measure a solver residual instead of the motion. Only the robot is re-parked.
            """
            m = len(fr)
            pose = torch.zeros(n, 7, device=dev)
            pose[:, 3] = 1.0
            pose[:m, :3] = self._ref_obj_pos[fr] + org[:m]
            pose[:m, 3:7] = self._ref_obj_quat[fr]
            z_ref = pose[:, 2].clone()
            pose[:, 2] = pose[:, 2] + dz
            self._object.write_root_pose_to_sim(pose)
            self._object.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
            for _ in range(int(c.declear_settle_steps)):
                self.robot.write_root_pose_to_sim(park)
                self.robot.write_root_velocity_to_sim(zero6)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)
            return (self._object.data.root_pos_w[:, 2] - z_ref)[:m]

        # lift <- settle(lift) 는 고정점 반복이다(누적 아님). 추가 라운드는 settle 이 멱등이 아닌 얇은 물체용.
        # 음수(레퍼런스가 받침 위에 뜸)는 잘라내서 들어 올리기만 한다.
        lift = torch.zeros(F, device=dev)
        for _rnd in range(_DECLEAR_ROUNDS):
            for base in range(0, len(rest_idx), n):
                fr = rest_idx[base:base + n]
                dz = torch.zeros(n, device=dev)
                dz[:len(fr)] = lift[fr]
                d = _settle(fr, dz)
                lift[fr] = d.clamp(min=0.0, max=float(c.declear_max_lift))

        # ---- lift 는 구간 최대가 아니라 프레임별 값: 구간 최대로 올리면 스폰 때 물체가 떨어진다. 구간 수는 로그용 ----
        rest_c = rest.cpu().numpy()
        i, n_seg = 0, 0
        while i < F:
            if not rest_c[i]:
                i += 1
                continue
            j = i
            while j + 1 < F and rest_c[j + 1]:
                j += 1
            n_seg += 1
            i = j + 1
        self._obj_spawn_lift = lift

        # ---- 검증: 보정 높이에서 다시 settle. 잔차 = settle 높이 - 스폰 높이 (+ 는 lift 부족, - 는 여전히 낙하) ----
        resid = []
        for base in range(0, len(rest_idx), n):
            fr = rest_idx[base:base + n]
            dz = torch.zeros(n, device=dev)
            dz[:len(fr)] = lift[fr]
            resid.append(_settle(fr, dz)[:len(fr)] - dz[:len(fr)])   # vs where we SPAWN it, not vs ref
        r = torch.cat(resid) if resid else torch.zeros(1, device=dev)

        nz = lift > 1e-4
        print(f"[spawn-declear] {self._obj_name}: {int(rest.sum())}/{F} frames at rest in "
              f"{n_seg} segment(s); {int(nz.sum())} lifted, mean "
              f"{float(lift[nz].mean()) * 100 if nz.any() else 0:.2f} cm, "
              f"max {float(lift.max()) * 100:.2f} cm "
              f"({int((lift >= float(c.declear_max_lift) - 1e-6).sum())} clipped at declear_max_lift)")
        print(f"[spawn-declear] residual after correction: p50 {float(r.median()) * 1000:+.2f} mm, "
              f"p90 {float(r.quantile(0.9)) * 1000:+.2f} mm, max {float(r.max()) * 1000:+.2f} mm "
              f"({float((r.abs() < 2e-3).float().mean()):.0%} of rest frames within 2 mm)")
    # [/ROLLBACK MARKER: spawn-declear] --------------------------------------------------------

    # [ROLLBACK MARKER: context-z] -------------------------------------------------------------
    def _apply_context_z_auto(self) -> None:
        """Sink the context objects by the lift the declear solve just asked for, then re-solve.

        Why sink the support instead of lifting the object: lifting spawns the object at the height
        it will actually settle at, which is ABOVE the reference — and the reward compares against
        the untouched reference, so the object is marked down on every resting frame for a position
        the support physically BLOCKS it from reaching. Sinking the support moves the settling height
        onto the reference, making the target achievable. Measured 0.3 s after spawn (lift -> sink,
        median over resting frames): pot 49.9 -> 22.4 mm, pan 29.2 -> 5.3 mm, kettle 28.6 -> 7.4 mm,
        knife 5.7 -> 3.4 mm, and the vertical part goes to roughly zero on every clip tested.

        It fixes the VERTICAL error only. The object also slides sideways and tips over on the
        support (pot 9.6 deg, book 21.2 deg even after the fix) because the reference pose is not a
        stable resting pose on the collider — untouched by anything done here.

        The value must be measured, not configured: 5.5 mm on the knife clip, 19.5 mm on the pan,
        24.1 mm on the kettle. And it cannot be known at spawn time, since it comes from settling the
        object ON the contexts — hence moving them here, after the solve, rather than at spawn.

        One number for every context object, because a per-frame or per-object correction has
        nowhere to live: these prims are spawned once and never touched again (no per-env root-state
        view). The few mm one constant cannot cover is what the per-frame object lift is still for,
        which is why object_spawn_declear stays on alongside this.
        """
        c = self.cfg
        self._ctx_z_applied = 0.0
        if not getattr(c, "context_z_auto", False) or not getattr(self, "_ctx_prims", []):
            return
        lift = getattr(self, "_obj_spawn_lift", None)
        if lift is None:
            return
        nz = lift[lift > 1e-5]
        if nz.numel() == 0:                     # 물체가 지지면에 놓이는 프레임이 없으면 할 일이 없습니다
            return
        dz = float(nz.median())
        if dz <= 1e-4:                          # 이미 맞아 있음
            return

        from isaacsim.core.prims import RigidPrim

        moved = 0
        for pat in self._ctx_prims:
            try:
                view = RigidPrim(pat)
                pos, quat = view.get_world_poses()
                pos = pos.clone()
                pos[:, 2] -= dz
                view.set_world_poses(pos, quat)
                moved += 1
            except Exception as e:  # noqa: BLE001
                print(f"[context-z] {pat} 이동 실패: {e}")
        if not moved:
            return
        self.sim.step(render=False)
        self.scene.update(dt=self.cfg.sim.dt)
        self._ctx_z_applied = dz
        print(f"[context-z] 컨텍스트 {moved}개를 {dz * 1000:.1f} mm 내렸습니다 — 남은 보정을 다시 풉니다")
        self._solve_spawn_declear()             # 내린 뒤 남은 보정을 측정 (보통 0에 가깝습니다)
    # [/ROLLBACK MARKER: context-z] ------------------------------------------------------------

    # ---------------------------------------------------------- post-init buffers
    def _post_init_buffers(self) -> None:
        dev = self.device
        c = self.cfg

        # ---- action joint index map (group order legs→waist→arms→hands) ----
        self._action_joint_ids: list[int] = []
        self._group_slices: dict[str, slice] = {}
        off = 0
        for gname, g in JOINT_GROUPS.items():
            ids, _ = self.robot.find_joints(g["expr"])
            assert len(ids) == g["dof"], f"group {gname}: expected {g['dof']} joints, got {len(ids)}"
            self._action_joint_ids += ids
            self._group_slices[gname] = slice(off, off + g["dof"])
            off += g["dof"]
        self._action_joint_ids_t = torch.tensor(self._action_joint_ids, device=dev, dtype=torch.long)
        # 액션 관절의 실제 이름 순서. find_joints 는 정규식 순서가 아니라 articulation 내부 순서로
        # 돌려주므로 밖에서 추측할 수 없다 (rollout.py 가 결과 파일에 함께 저장한다).
        self._action_joint_names = [self.robot.joint_names[i] for i in self._action_joint_ids]
        if bool(os.environ.get("PRINT_ACTION_JOINTS")):
            print("[action-joints] " + " ".join(f"{i}:{n}" for i, n in
                                                enumerate(self._action_joint_names)))
        self._n_act = off                                              # 65
        # [ROLLBACK MARKER: tendon-reset] 텐던 축 J0 8개의 리셋 값. J0 는 액션 관절이 아니어서 따로 쓰지 않으면 0 으로
        # 남아, J1 이 굽은 프레임에서 텐던 제약을 어긴 채 시작해 말단이 튄다. 값은 리타게팅 J0(_ref_j0)에서 온다.
        _tj1n = [f"robot0_{s}_{f}J1" for s in "lr" for f in ("FF", "MF", "RF", "LF")]
        _jn = self.robot.data.joint_names
        _tp = [(_jn.index(a), _jn.index(a[:-1] + "0")) for a in _tj1n
               if a in _jn and (a[:-1] + "0") in _jn]
        self._tendon_j1_ids = torch.tensor([a for a, _b in _tp], device=dev, dtype=torch.long)
        self._tendon_j0_ids = torch.tensor([b for _a, b in _tp], device=dev, dtype=torch.long)
        print(f"[tendon-reset] 텐던 쌍 {len(_tp)}개 — 리셋 J0 = min(리타게팅 J0, J1)")
        self._finger_alpha = float(c.finger_ema_alpha)

        # per-action joint limits (in action order) for scale/unscale
        # ── [ROLLBACK MARKER: soft-limit-hands] 손가락 관절 soft 한계를 soft_joint_pos_limit_factor_hands 로 재계산 ──
        self._soft_hands_info = None
        _fh = float(getattr(c, "soft_joint_pos_limit_factor_hands", 0.0) or 0.0)
        if _fh > 0.0:
            import re as _re
            _hj = torch.tensor([i for i, n in enumerate(self.robot.joint_names)
                                if _re.match(r"robot0_[lr]_(FF|MF|RF|LF|TH)J\d", n)], device=dev, dtype=torch.long)
            _hard = self.robot.data.joint_pos_limits[:, _hj]                              # (E,n,2) PhysX 하드 한계
            _mid = 0.5 * (_hard[..., 0] + _hard[..., 1]); _rng = _hard[..., 1] - _hard[..., 0]
            _old = self.robot.data.soft_joint_pos_limits[0, _hj].clone()
            self.robot.data.soft_joint_pos_limits[:, _hj, 0] = _mid - 0.5 * _rng * _fh
            self.robot.data.soft_joint_pos_limits[:, _hj, 1] = _mid + 0.5 * _rng * _fh
            _new = self.robot.data.soft_joint_pos_limits[0, _hj]
            _fb = getattr(getattr(self.robot, "cfg", None), "soft_joint_pos_limit_factor", None)   # articulation 의 factor (몸에 적용된 값)
            self._soft_hands_info = (_fh, int(_hj.numel()))
            print(f"[soft-limit-hands] Shadow 손가락 {int(_hj.numel())}관절 soft 한계 factor {_fb}→{_fh} 재계산: "
                  f"범위 변화 평균 {float(((_new[:, 1] - _new[:, 0]) - (_old[:, 1] - _old[:, 0])).mean()) * 180 / math.pi:+.2f} deg "
                  f"(몸 관절은 articulation factor {_fb} 유지)")
        # ── [/ROLLBACK MARKER: soft-limit-hands] ──
        lim = self.robot.data.soft_joint_pos_limits[0, self._action_joint_ids_t]       # (65,2)
        self._ctrl_lower = lim[:, 0].clone()
        self._ctrl_upper = lim[:, 1].clone()

        # ---- keypoint body ids + local offsets (56, matching the reference order) ----
        kpt_names: list[str] = list(BODY_KPTS.values())
        kpt_off: list[list[float]] = [[0.0, 0.0, 0.0] for _ in BODY_KPTS]   # 몸통 키포인트 = 링크 원점
        # [ROLLBACK MARKER: hand-kpt-align] 오프셋은 링크 이름이 아니라 키포인트별 `pad` 플래그로 정한다.
        # distal 링크를 두 번 쓰기 때문이다(오프셋 0 = DIP, pad = TIP).
        for side in ("l", "r"):
            for spec in HAND_CHAIN.values():
                for body, use_pad in zip(spec["shadow"], spec["pad"]):
                    full = f"robot0_{side}_{body}"
                    kpt_names.append(full)
                    kpt_off.append(FINGERTIP_OFFSETS.get(full, [0.0, 0.0, 0.0])
                                   if use_pad else [0.0, 0.0, 0.0])
        self._kpt_body_ids = torch.tensor(
            [self.robot.find_bodies(n)[0][0] for n in kpt_names], device=dev, dtype=torch.long)
        self._kpt_offsets = torch.tensor(kpt_off, device=dev, dtype=torch.float32)      # (54,3)
        # 몸 키포인트 보상을 CORE 9개(rew_body_kpts)와 EE = 손목 2 + 발목 2(rew_ee_kpts)로 나눈다(이름 매칭).
        # [ROLLBACK MARKER: ee-split] [ROLLBACK MARKER: ee-torso] 종료 게이트는 13개 전체 평균을 쓴다.
        _body_names = kpt_names[:len(BODY_KPTS)]
        # [ROLLBACK MARKER: wrist-into-ee] 손목이 EE 그룹으로 흡수됐다 (손목2+발목2+몸통1 = 5).
        # [smplx-kpts] torso 키포인트 제거(사용자 결정) → EE 그룹은 손목 2 + 발목 2 = 4개.
        _EE_NAMES = ("wrist", "ankle")            # EE 그룹에 들어가는 링크 이름 조각
        self._body_core_idx = torch.tensor(
            [i for i, n in enumerate(_body_names)
             if not any(t in n for t in _EE_NAMES)],
            device=dev, dtype=torch.long)                                              # (9,) core body
        # [wrist-into-ee] 손목 2개는 이제 보상에서 EE 항에 흡수됐고, 이 인덱스는 오직 전용
        # 손목-POSITION 종료 게이트(term_wrist_pos_err)와 Error / wrist_kpts 로그에만 쓰인다.
        self._wrist_kpt_idx = torch.tensor(
            [i for i, n in enumerate(_body_names) if "wrist" in n], device=dev, dtype=torch.long)  # (2,)
        # [wrist-into-ee][smplx-kpts] 손목 2 + 발목 2 = 4개의 MEAN → rew_ee_kpts.
        self._ee_kpt_idx = torch.tensor(
            [i for i, n in enumerate(_body_names) if any(t in n for t in _EE_NAMES)],
            device=dev, dtype=torch.long)                                    # (4,) wrist×2 + ankle×2
        # 이름 매칭이라 조용히 잘못 묶일 수 있다 — 실제 구성을 한 번 찍어 확인 가능하게 남긴다.
        print(f"[g1] kpt groups: core({len(self._body_core_idx)})="
              f"{[_body_names[i] for i in self._body_core_idx.tolist()]} | "
              f"wrist({len(self._wrist_kpt_idx)})={[_body_names[i] for i in self._wrist_kpt_idx.tolist()]} | "
              f"ee({len(self._ee_kpt_idx)})={[_body_names[i] for i in self._ee_kpt_idx.tolist()]}")
        # [ROLLBACK MARKER: energy] Σ|τ·q̇| 대상 = 허리 3 + 다리 12 = 15관절. 팔·손목은 제외
        # (레퍼런스 파워가 가장 크고 로봇이 이미 레퍼런스보다 느리다).
        _en_pat = ("waist_", "_hip_", "_knee_", "_ankle_")
        self._energy_joint_ids = torch.tensor(
            [i for i, n in enumerate(self.robot.joint_names) if any(p in n for p in _en_pat)],
            device=dev, dtype=torch.long)
        # [ROLLBACK MARKER: ankle-acc] 발목 관절 가속도 벌점 (ResMimic ankle_dof_acc). 급격한 변화에만 대가를 붙인다.
        # ArticulationData.joint_acc 를 그대로 쓴다(서브스텝마다 갱신되고 리셋 때 0 이 되어 불연속이 자동 처리됨).
        self._ankle_acc_joint_ids = None
        if float(getattr(c, "rew_ankle_acc", 0.0)) != 0.0:
            import re as _re_aa
            _aa_pat = list(getattr(c, "ankle_acc_joints", []))
            _aa_ids = [i for i, n in enumerate(self.robot.joint_names)
                       if any(_re_aa.fullmatch(p, n) for p in _aa_pat)]
            assert _aa_ids, f"[ankle-acc] 정규식 {_aa_pat} 에 맞는 관절이 없다"
            self._ankle_acc_joint_ids = torch.tensor(_aa_ids, device=dev, dtype=torch.long)
            self._diag_ankle_acc = torch.zeros((), device=dev)
            print(f"[ankle-acc] ON: {len(_aa_ids)}관절 {[self.robot.joint_names[i] for i in _aa_ids]}, "
                  f"출처 data.joint_acc (물리 dt {float(c.sim.dt) * 1000:.1f} ms 차분), "
                  f"가중치 {float(c.rew_ankle_acc)}")
        # [/ROLLBACK MARKER: ankle-acc]
        # [ROLLBACK MARKER: waist-acc] 허리 관절 가속도 벌점. ankle-acc 와 같은 형태이고,
        # 허리 가속도가 발목보다 커서 가중치를 따로 둔다.
        self._waist_acc_joint_ids = None
        if float(getattr(c, "rew_waist_acc", 0.0)) != 0.0:
            import re as _re_wa
            _wa_pat = list(getattr(c, "waist_acc_joints", []))
            _wa_ids = [i for i, n in enumerate(self.robot.joint_names)
                       if any(_re_wa.fullmatch(_p, n) for _p in _wa_pat)]
            assert _wa_ids, f"[waist-acc] 정규식 {_wa_pat} 에 맞는 관절이 없다"
            self._waist_acc_joint_ids = torch.tensor(_wa_ids, device=dev, dtype=torch.long)
            self._diag_waist_acc = torch.zeros((), device=dev)
            print(f"[waist-acc] ON: {len(_wa_ids)}관절 {[self.robot.joint_names[i] for i in _wa_ids]}, "
                  f"출처 data.joint_acc (물리 dt {float(c.sim.dt) * 1000:.1f} ms 차분), "
                  f"가중치 {float(c.rew_waist_acc)}")
        # ── [/ROLLBACK MARKER: waist-acc] ──
        print(f"[energy] Σ|τ·q̇| 대상 {len(self._energy_joint_ids)}관절: "
              f"{[self.robot.joint_names[i] for i in self._energy_joint_ids.tolist()]}")

        # CoM-over-support balance reward: cache constant link masses + ankle body ids + base +X axis.
        try:
            self._body_masses = self.robot.data.default_mass.to(dev)               # (E, nB) constant link masses
        except Exception:
            self._body_masses = self.robot.root_physx_view.get_masses().to(dev)
        self._use_body_com = hasattr(self.robot.data, "body_com_pos_w")            # per-body world CoM if available
        self._ankle_body_ids = [self.robot.find_bodies(n)[0][0]
                                for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self._x_axis = torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(self.num_envs, 1)  # (E,3) base +X (forward)
        # foot SOLE OUTWARD normal in the ankle_roll_link LOCAL frame (points DOWN, out of the sole) — same
        # for both feet (legs are not Z-mirrored). Used to project the ground contact force to a compressive
        # magnitude (force · sole-INWARD = force · -normal), mirroring the fingertip pad-normal projection.
        self._foot_sole_normal = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], device=dev)  # (2,3) [L,R]

        # ---- fingertip (10) body ids + offsets + pad normals (bimanual) ----
        self._ft_body_ids = torch.tensor(
            [self.robot.find_bodies(n)[0][0] for n in c.fingertip_body_names], device=dev, dtype=torch.long)
        self._ft_offsets = torch.tensor(
            [FINGERTIP_OFFSETS[n] for n in c.fingertip_body_names], device=dev, dtype=torch.float32)   # (10,3)
        self._ft_pad_normals = torch.tensor(
            [FINGERTIP_PAD_NORMALS[n] for n in c.fingertip_body_names], device=dev, dtype=torch.float32)  # (10,3)
        # ── [ROLLBACK MARKER: anti-shake] 각속도 데드존 벌점 대상 링크 (head_link 없으면 torso_link 대체) ──
        self._diag_blowup_frac = torch.zeros((), device=dev)      # [nan-guard]
        self._diag_obs_sanitized = torch.zeros((), device=dev)    # [nan-guard]
        self._anti_shake_body_ids = None
        if float(getattr(c, "rew_anti_shake", 0.0)) != 0.0:
            _ids, _used = [], []
            for _n in list(getattr(c, "anti_shake_bodies", [])):
                _cand = [_n] + (["torso_link"] if _n == "head_link" else [])
                _hit = next((x for x in _cand if x in self.robot.body_names), None)
                if _hit is None:
                    print(f"[anti-shake] 바디 '{_n}' 을 찾지 못해 건너뜁니다")
                    continue
                _ids.append(self.robot.body_names.index(_hit))
                _used.append(_hit if _hit == _n else f"{_n}→{_hit}")
            if _ids:
                self._anti_shake_body_ids = torch.tensor(_ids, device=dev, dtype=torch.long)
                print(f"[anti-shake] ON: 링크 {_used}, θ = {float(c.anti_shake_ang_vel_thresh):.2f} rad/s, 가중치 {float(c.rew_anti_shake)}")
        # ── [/ROLLBACK MARKER: anti-shake] ──
        # ---- palm (wrist) body ids for explicit palm orientation/velocity obs (L,R to match ft order) ----
        self._palm_body_ids = torch.tensor(
            [self.robot.find_bodies(n)[0][0] for n in ("robot0_l_palm", "robot0_r_palm")],
            device=dev, dtype=torch.long)

        # ---- reference tensors (move numpy → device) ----
        def T(a):
            return torch.from_numpy(np.asarray(a)).to(dev)
        self._ref_kpts = T(self._np_ref_kpts)                          # (F,55,3)
        self._ref_root_pos = T(self._np_root_pos)                      # (F,3)
        self._ref_root_quat = _canon(T(self._np_root_quat))            # (F,4)
        self._ref_root_linvel = T(self._np_root_linvel)                # (F,3)
        self._ref_root_angvel = T(self._np_root_angvel)                # (F,3)
        self._ref_ft_pad = T(self._np_ft_pad)                          # (F,10,3)
        self._ref_foot_contact = T(self._np_ref_foot_contact)          # (F,2) [L,R] binary contact schedule
        self._ref_obj_pos = T(self._np_obj_base[:, :3])                # (F,3)
        self._ref_obj_quat = _canon(T(self._np_obj_base[:, 3:7]))      # (F,4)
        self._ref_obj_linvel = T(self._np_obj_linvel)                  # (F,3) reference object linear vel
        self._ref_obj_angvel = T(self._np_obj_angvel)                  # (F,3) reference object angular vel
        # [ROLLBACK MARKER: contact-vel-gate] (F,) 물체-속도 게이트. grasp env 의 future_contact
        # 속도 조건과 같은 식 — 레퍼런스 궤적에서 온 값이라 로봇 상태에 되먹임되지 않는다(살아있는
        # 물체 속도를 쓰면 "떨어뜨려서 빨라짐 → 게이트 열림"이 되어버린다). 비활성이면 전부 1.
        if c.contact_vel_gate:
            _spd = self._ref_obj_linvel.norm(dim=-1)                   # (F,) m/s
            _ang = self._ref_obj_angvel.norm(dim=-1)                   # (F,) rad/s (측지 각속도 크기)
            self._ref_obj_vel_gate = ((_spd > float(c.contact_vel_gate_lin))
                                      | (_ang > float(c.contact_vel_gate_ang))).float()
            if self._ref_len > 1:
                self._ref_obj_vel_gate[0] = self._ref_obj_vel_gate[1]  # 프레임 0 은 유한차분이 0
            print(f"[contact-vel-gate] 통과 {self._ref_obj_vel_gate.mean()*100:.1f}% "
                  f"(lin>{c.contact_vel_gate_lin} m/s | ang>{c.contact_vel_gate_ang} rad/s)  "
                  f"물체 각속도 중앙값 {_ang.median():.3f} rad/s")
        else:
            self._ref_obj_vel_gate = torch.ones(self._ref_len, device=dev)
        self._ref_obj_dof = T(self._np_obj_dof)                        # (F,P)
        self._ref_joints = T(self._np_ref_joints) if self._np_ref_joints is not None else None
        self._remap_ref_joints(c)
        # [ROLLBACK MARKER: ref-reset-jvel] 레퍼런스 관절 속도 (F,65), 액션 관절 순서. 후방차분, 0번 프레임 0.
        # _remap_ref_joints 뒤여야 열이 맞고, 리셋 경로가 쓰므로 SONIC 블록 밖에서 만든다.
        self._ref_joint_vel = None
        # 스위치를 배율로 접어 둡니다 — 꺼져 있으면 0.0이라 _reset_idx의 대입이 정확히 0을 씁니다
        # (jvel 초기값도 0이므로 기존 동작과 비트 단위 동일). 분기를 하나 더 만들지 않기 위한 것.
        self._ref_jvel_scale = float(c.ref_reset_joint_vel_scale) if c.ref_reset_joint_vel else 0.0
        # 성분별 상한을 (1,65)로 펴 둡니다 — 몸통과 손은 자연 스케일이 3배, 액추에이터 한계가
        # 6.7배(100 vs 15) 차이나서 스칼라 하나로는 손 쪽 밸브가 작동하지 않습니다. _group_slices는
        # 위(_post_init_buffers 앞부분)에서 이미 만들어져 있습니다.
        self._ref_jvel_clip = torch.full((1, self._n_act), float(c.ref_reset_joint_vel_clip), device=dev)
        self._ref_jvel_clip[:, self._group_slices["hands"]] = float(c.ref_reset_joint_vel_clip_hands)
        if self._ref_joints is not None:
            self._ref_joint_vel = torch.zeros_like(self._ref_joints)          # (F,65)
            self._ref_joint_vel[1:] = (self._ref_joints[1:] - self._ref_joints[:-1]) * float(c.control_fps)
            # 비유한 값 방어. 배율을 0으로 접어 스위치를 끄는 방식이라 NaN이 들어오면 NaN*0.0=NaN이
            # 되어 꺼진 상태에서도 jvel이 오염됩니다(그러면 SONIC 되먹임을 타고 자기증식). 현재 12클립은
            # 전부 유한함을 확인했지만, 클립이 추가될 때를 대비해 여기서 잘라둡니다.
            self._ref_joint_vel = torch.nan_to_num(self._ref_joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
        # [ROLLBACK MARKER: ref-j0] 손가락 J0 속도도 리타게팅 결과에서 뽑는다(위와 같은 규약).
        self._ref_j0_vel = None
        if getattr(self, "_ref_j0", None) is not None:
            self._ref_j0_vel = torch.zeros_like(self._ref_j0)                 # (F,8)
            self._ref_j0_vel[1:] = (self._ref_j0[1:] - self._ref_j0[:-1]) * float(c.control_fps)
            self._ref_j0_vel = torch.nan_to_num(self._ref_j0_vel, nan=0.0, posinf=0.0, neginf=0.0)
            _p99 = float(self._ref_j0_vel.abs().quantile(0.99))
            print(f"[ref-j0] J0 속도 후방차분 — p99 {_p99:.3f} rad/s "
                  f"(손 클립 {c.ref_reset_joint_vel_clip_hands})")
        print(f"[ref-reset-jvel] {'ON' if c.ref_reset_joint_vel else 'OFF'} "
              f"(scale={self._ref_jvel_scale}, clip body/hands="
              f"{c.ref_reset_joint_vel_clip}/{c.ref_reset_joint_vel_clip_hands} rad/s, "
              f"ref_start_prob={c.ref_start_prob})")
        if self._ref_joint_vel is not None:
            _bv, _hv = self._ref_joint_vel[:, :29].abs(), self._ref_joint_vel[:, 29:].abs()
            _nb = int((_bv > c.ref_reset_joint_vel_clip).sum()) + int((_hv > c.ref_reset_joint_vel_clip_hands).sum())
            print(f"[ref-reset-jvel] clip |v| max body={float(_bv.max()):.2f} hands={float(_hv.max()):.2f} "
                  f"rad/s -> clipped components: {_nb}")
        # [/ROLLBACK MARKER: ref-reset-jvel]
        # 레퍼런스 손바닥 자세 [L,R] (리타게팅에 g1_palm_quat 가 있을 때만). 지금은 어디서도 안 쓰고 로드만 남겨 둔다.
        self._has_palm_ref = self._np_ref_palm_quat is not None
        self._ref_palm_quat = _canon(T(self._np_ref_palm_quat)) if self._has_palm_ref else None  # (F,2,4)
        self._has_object = self._object is not None
        self._RESERVE_ARTIC = 4                                        # reserved obs slots per parts

        # 링크별 접촉(Option A): 레퍼런스 mask + 물체 로컬 법선·목표 + 로봇 body id. _ft_distal_idx 는 손끝 10개를
        # LINK_CONTACT_NAMES 의 distal 링크로 매핑해 손끝 kpt 보상과 delta_ft_obj 가 같은 맵을 읽게 한다.
        self._ref_link_contact_mask = T(self._np_link_contact_mask)            # (F,L)
        self._ref_link_contact_normal_local = T(self._np_link_contact_normal)  # (F,L,3) object-local
        self._ref_link_contact_target_local = T(self._np_link_contact_target)  # (F,L,3) object-local

        # [ROLLBACK MARKER: cws-contact] 접촉을 렌치(물체를 어떻게 움직일 수 있는가)로 비교 (CHORD, arXiv 2607.00033).
        # 링크 위치 매칭과 달리 손바닥 대신 손가락으로 같은 힘을 내도 인정된다. 사람 쪽 σ_h 는 여기서 한 번만 계산한다.
        self._cws_sigma_h = None
        # [ROLLBACK MARKER: cws-human-ref] σ_h 접촉 집합 출처 (2026-09-09). hand_pretrain_contact_map 이 로드돼도 CWS 기준은
        # 사람 맵이다(cws_ref_contact_source="human"). 손끝 목표·관측·force 게이트는 1단계 맵을 쓴다.
        _cws_mask_t, _cws_tgt_t, _cws_nrm_t = (self._ref_link_contact_mask, self._ref_link_contact_target_local,
                                              self._ref_link_contact_normal_local)
        _cws_src_used = "hand_pretrain" if getattr(self, "_stage1_contact_map_loaded", False) else "human"
        if (getattr(self, "_stage1_contact_map_loaded", False)
                and str(getattr(c, "cws_ref_contact_source", "human")) == "human"
                and getattr(self, "_has_link_contact_human", False)):
            _cws_mask_t = T(self._np_link_contact_mask_human)
            _cws_tgt_t = T(self._np_link_contact_target_human)
            _cws_nrm_t = T(self._np_link_contact_normal_human)
            _cws_src_used = "human"
        # ── [/ROLLBACK MARKER: cws-human-ref] ──
        if self._has_link_contact and c.contact_reward_mode in ("cws", "both"):
            _m = _cws_mask_t > 0.5                                             # (F,L)  [cws-human-ref]
            if bool(_m.any()):
                # [ROLLBACK MARKER: cws-rc-mesh] 렌치 회전 성분을 나눌 물체 크기 = 메시 중심에서 최대 거리.
                # 메시를 못 읽으면 접촉점 노름 0.9 분위로 대체. 되돌리기: _rc_mesh = None
                _rc_q90 = float(_cws_tgt_t[_m].norm(dim=-1).quantile(0.9))              # [cws-human-ref]
                _rc_mesh = self._object_mesh_radius() if self._has_object else None
                _rc_ok = _rc_mesh is not None and _rc_mesh > 1e-4
                self._cws_len = _rc_mesh if _rc_ok else _rc_q90
                self._cws_basis = CWS.make_basis(c.cws_n_dir, c.cws_seed, device=dev)
                # [ROLLBACK MARKER: cws-com] 모멘트 팔은 물리 COM 기준이다. 레퍼런스 목표도 COM 프레임으로 옮기고
                # 로봇 쪽도 root_com_* 을 쓴다. COM 을 못 읽으면 양쪽 모두 body 원점.
                self._cws_com_p = None
                self._cws_com_q = None
                try:
                    _cp = self._object.data.com_pos_b.reshape(-1, 3)[0].to(dev)      # (3,)
                    _cq = self._object.data.com_quat_b.reshape(-1, 4)[0].to(dev)     # (4,) wxyz
                    if torch.isfinite(_cp).all() and torch.isfinite(_cq).all() and _cq.norm() > 0.5:
                        self._cws_com_p, self._cws_com_q = _cp, _cq
                except Exception:
                    pass
                _tgt = _cws_tgt_t                                                 # [cws-human-ref]
                _nrm = -_cws_nrm_t                            # 저장 법선은 표면 바깥쪽 -> 안쪽으로
                if self._cws_com_p is not None:
                    _qi = math_utils.quat_conjugate(self._cws_com_q).expand(_tgt.shape[:-1] + (4,))
                    _tgt = math_utils.quat_apply(_qi, _tgt - self._cws_com_p)
                    _nrm = math_utils.quat_apply(_qi, _nrm)
                self._cws_sigma_h = CWS.support(
                    self._cws_basis, _tgt, _nrm, _m,
                    c.cws_mu, self._cws_len, c.cws_n_edge, c.cws_link_chunk)    # (F,n_dir)
                _src = "메시" if _rc_ok else "접촉점0.9분위(메시 읽기 실패)"
                print(f"[cws] 접촉 렌치 보상 포함  물체크기={self._cws_len * 100:.1f}cm({_src}, "
                      f"접촉점0.9분위={_rc_q90 * 100:.1f}cm)  "
                      f"모멘트팔={'COM' if self._cws_com_p is not None else 'body원점'}  "
                      f"방향={c.cws_n_dir}  옆면={c.cws_n_edge}  여유={c.cws_beta}  mu={c.cws_mu}  "
                      f"σ_h 접촉 출처={_cws_src_used}")   # [cws-human-ref]
        self._link_contact_body_ids = torch.tensor(
            [self.robot.find_bodies(n)[0][0] for n in LINK_CONTACT_NAMES], device=dev, dtype=torch.long)  # (L,)
        # 손끝 10개(cfg.fingertip_body_names 순서)의 LINK_CONTACT_NAMES 색인.
        self._ft_distal_idx = torch.tensor(
            [LINK_CONTACT_NAMES.index(n) for n in self.cfg.fingertip_body_names], device=dev, dtype=torch.long)  # (10,)
        # per-link OUTWARD pad/palmar normal (link-local) → the link's own contact FACE. Force is projected on
        # the INWARD (-pad) direction (like the fingertip). VERIFIED on the rest-pose USD (32/32; see cfg
        # LINK_PAD_NORMALS).
        self._link_pad_normals = torch.tensor(
            [LINK_PAD_NORMALS[n] for n in LINK_CONTACT_NAMES], device=dev, dtype=torch.float32)  # (L,3) OUTWARD

        # ---- action / EMA buffers ----
        default_q = self.robot.data.default_joint_pos[:, self._action_joint_ids_t]      # (E,65)
        self._smoothed_actions = self._unscale(default_q).clone()      # (E,65) normalized
        self._residual_target = None       # (E,65) PD target, set every step by _sonic_pre_physics_step

        # ---- per-env trajectory frame index ----
        self._frame_idx = torch.zeros(self.num_envs, device=dev, dtype=torch.long)

        # ---- state cache + RSI (222) ----
        # [0]reward [1:14]root(pos,quat,linvel,angvel) [14:27]obj(같은 순서) [27:92]jpos [92:157]jvel [157:222]smoothed
        self._STATE_DIM = 222
        # [ROLLBACK MARKER: spawn-declear] steps physics; everything it touches (reference arrays,
        # object, robot, scene) already exists by here.
        self._solve_spawn_declear()
        # [ROLLBACK MARKER: context-z] 지지면을 내려 물체가 레퍼런스 높이에 안착하게 합니다.
        # 반드시 declear 뒤에 — 내릴 양을 그 결과에서 읽습니다.
        self._apply_context_z_auto()
        self._apply_body_kpt_fk()             # [ROLLBACK MARKER: body-kpt-fk]
        self._setup_stage1_hand()             # [ROLLBACK MARKER: stage1-hand]
        # [/ROLLBACK MARKER: spawn-declear]
        self._state_cache = torch.zeros(self._ref_len, self._STATE_DIM, device=dev)
        self._state_cache[:, 0] = -float("inf")                        # reward column
        self._init_flg = torch.ones(self._ref_len, device=dev, dtype=torch.bool)   # True = reference (no cache)
        self._reached_frame = 0
        # [ROLLBACK MARKER: late-gate] 캐시 품질 게이트 early→late 전환: 한 에피소드가 클립의
        # late_gate_survival_frac 이상을 살고 끝 3프레임 안에서 끝나야 켠다.
        # [ROLLBACK MARKER: exp-tracking] 지수 추적 가중치 = |rew_*| / exp_tracking_budget. 허용 오차는 σ 가 정한다.
        _lw = {"body": abs(c.rew_body_kpts),
               "ee": abs(c.rew_ee_kpts),                                   # [wrist-into-ee]
               "hand": abs(c.rew_hand_kpts),
               "fingertip": abs(c.rew_fingertip), "root_pos": abs(c.rew_root_pos),
               "root_rot": abs(c.rew_root_ori), "obj_pos": abs(c.rew_obj_pos),
               "obj_rot": abs(c.rew_obj_rot)}
        _tot = sum(_lw.values()) or 1.0
        self._exp_w = {k: float(c.exp_tracking_budget) * v / _tot for k, v in _lw.items()}
        self._exp_s2 = {"body": c.sigma_body ** 2,
                        "ee": c.sigma_ee ** 2,                                  # [wrist-into-ee]
                        "hand": c.sigma_hand ** 2,
                        "fingertip": c.sigma_fingertip ** 2, "root_pos": c.sigma_root_pos ** 2,
                        "root_rot": c.sigma_root_rot ** 2, "obj_pos": c.sigma_obj_pos ** 2,
                        "obj_rot": c.sigma_obj_rot ** 2}
        # 각 항이 읽을 오차 키. [z-weight] hand/obj_pos 는 보상용 z 가중 사본을 읽는다 —
        # 지수 항이 곧 보상이고, z 가중은 "보상에만" 거는 규약이기 때문이다 (게이트는 무가중).
        self._exp_key = {"body": "body_core",
                         "ee": "ee",                                                 # [wrist-into-ee]
                         "hand": "hand_w", "fingertip": "ft_reward",                 # [z-weight]
                         "root_pos": "root_pos", "root_rot": "root_rot",
                         "obj_pos": "obj_pos_w", "obj_rot": "obj_rot"}                # [z-weight]
        print("[exp-tracking] budget=" + f"{c.exp_tracking_budget}  " + "  ".join(
            f"{k}: w={self._exp_w[k]:.3f} s={self._exp_s2[k] ** 0.5:.3g}" for k in _lw))

        # [ROLLBACK MARKER: friction-curriculum] 물체 마찰 커리큘럼(robotis_shadow_grasp_rsi 에서 가져옴): 초반엔 마찰을
        # 높여 미끄러지는 병목 프레임을 통과하는 궤적을 만들고, 학습이 진행되면 실제 값으로 조인다.
        self._friction_step_count: int = 0
        self._last_friction_mean: float = float(c.friction_max_init)
        self._last_friction_max: float = float(c.friction_max_init)
        self._late_gate = False
        self._late_gate_frames = int(round(float(c.late_gate_survival_frac) * self._ref_len))
        # [ROLLBACK MARKER: deferred-cache] 종료 시 일괄 기록용 스테이징 버퍼. 에피소드가 cache_min_episode_length
        # 이상 살고 끝나야 _flush_state_cache 가 _state_cache 에 합친다. 기능이 켜졌을 때만 할당.
        self._pend_n = 0
        if int(getattr(c, "cache_min_episode_length", 0)) > 0:
            self._pend_cap = int(self.max_episode_length)
            self._pend_state = torch.zeros(self.num_envs, self._pend_cap, self._STATE_DIM, device=dev)
            self._pend_frame = torch.zeros(self.num_envs, self._pend_cap, device=dev, dtype=torch.long)
            self._pend_valid = torch.zeros(self.num_envs, self._pend_cap, device=dev, dtype=torch.bool)
            print(f"[cache] deferred (at-termination) commit ON: min_episode_length="
                  f"{c.cache_min_episode_length}, staging buffer "
                  f"{self.num_envs}x{self._pend_cap}x{self._STATE_DIM} fp32 = "
                  f"{self.num_envs * self._pend_cap * self._STATE_DIM * 4 / 1024**2:.0f} MB")
        else:
            self._pend_state = self._pend_frame = self._pend_valid = None
        # [wrist-rot] 손 블록 순서: [0]손목 [1..4]검지 [5..8]중지 [9..12]약지 [13..16]소지 [17..20]엄지
        # [ROLLBACK MARKER: wrist-frame-idx] 손목 프레임 인덱스를 하드코딩(20/40) 대신 상수로 (2026-09-06).
        _nb_k = N_BODY_KPTS                                          # 13
        _nh = N_HAND_KPTS_PER_HAND                                   # 21
        self._wrist_frame_idx = [(_nb_k + o, _nb_k + o + 5, _nb_k + o + 1) for o in (0, _nh)]  # L, R
        assert self._ref_kpts.shape[1] == _nb_k + 2 * _nh, (
            f"[wrist-frame-idx] 키포인트 수 불일치: ref {self._ref_kpts.shape[1]} != "
            f"{_nb_k} + 2x{_nh}")
        # ── [/ROLLBACK MARKER: wrist-frame-idx] ─────────────────────────────────────────────
        self._failure_count = torch.zeros(self._ref_len, device=dev)
        self._adaptive_back_frames = int(round(c.adaptive_back_seconds / c.ref_dt))
        self._adaptive_back_min_frames = int(round(c.adaptive_back_min_seconds / c.ref_dt))
        self._sampling_step_count = 0
        # per-env tracking-quality streak (grasp mechanism): _enough_continued = has tracking been
        # continuously "good enough" since reset; _enough_idx = last good frame (drives cache write
        # gate + failure-weighted sampling). Reset per env in _reset_idx.
        self._enough_continued = torch.ones(self.num_envs, dtype=torch.bool, device=dev)
        self._enough_idx = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        # RSI start frame of the CURRENT episode (diagnostics only — _enough_idx drifts to the last
        # good frame, so it cannot stand in for the start once the episode is running).
        self._episode_start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=dev)

        # ---- FROZEN SONIC body prior (built on device; env_isaaclab + gear_sonic + vector_quantize) ----
        self._sonic = None
        if getattr(c, "use_sonic", True):
            import sys as _sys
            # scripts/process_dataset/sonic/sonic_prior.py 를 불러온다. 이 파일 기준 repo 루트 = parents[6].
            from pathlib import Path as _Path
            _sp_dir = str(_Path(__file__).resolve().parents[6] / "scripts" / "process_dataset" / "sonic")
            if not os.path.isdir(_sp_dir):
                _sp_dir = "/home/peunsu/workspace/robotis_sh5/scripts/process_dataset/sonic"
            if _sp_dir not in _sys.path:
                _sys.path.insert(0, _sp_dir)
            import sonic_prior as _SP
            from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER as _GIO
            from gear_sonic.trl.utils.torch_transform import quat_inv as _qi, quat_mul as _qm
            self._SP = _SP
            self._sonic_qinv, self._sonic_qmul = _qi, _qm
            self._sonic = _SP.build_sonic(config_path=c.sonic_config_path,
                                          ckpt_path=c.sonic_ckpt_path, device=str(dev))
            self._sonic_layout, self._sonic_tok_dim = _SP.tokenizer_layout(self._sonic)
            # [ROLLBACK MARKER: sonic-v11] 참조 루트 자세 관측이 heading 정규화(v1.1)인지 펠비스 전체 자세(release)인지
            # 레이아웃 키로 판별한다. 체크포인트만 바꾸면 전환된다(레이아웃 오프셋은 동일).
            self._sonic_heading = "motion_anchor_ori_heading_mf_nonflat" in self._sonic_layout
            self._sonic_key_ori_g1 = ("motion_anchor_ori_heading_mf_nonflat" if self._sonic_heading
                                      else "motion_anchor_ori_b_mf_nonflat")
            self._sonic_key_ori_smpl = ("smpl_root_ori_heading_multi_future" if self._sonic_heading
                                        else "smpl_root_ori_b_multi_future")
            print(f"[sonic] 참조 자세 정규화 = {'heading(요만, v1.1)' if self._sonic_heading else '전체자세(릴리스)'}"
                  f"  슬롯 g1={self._sonic_key_ori_g1}")
            self._sonic_perm = _SP.build_body_perm(list(self.robot.joint_names), device=dev)  # robot->SONIC (29)
            self._sonic_default = _SP.sonic_default_vector(dev).view(1, -1)                    # (1,29) SONIC order
            # [ROLLBACK MARKER: sonic-encoder-g1] SONIC 인코더 선택.
            # 'smpl' = 사람 SMPL 관절 위치(손목 회전이 거의 안 담김), 'g1' = 로봇 29관절 각도·속도를 직접 준다.
            self._sonic_enc = str(getattr(c, "sonic_encoder", "smpl"))
            if self._sonic_enc == "g1" and self._ref_joints is None:
                print("[sonic-encoder] g1을 요청했지만 이 클립에 리타게팅(_ref_joints)이 없어 smpl로 대체합니다.")
                self._sonic_enc = "smpl"
            print(f"[sonic-encoder] tokenizer encoder = {self._sonic_enc}")
            self._ref_g1_q = self._ref_g1_v = None
            if self._ref_joints is not None:
                # 65개 액션 관절 -> SONIC의 29개 바디 관절 (로봇 관절 동일성 기준)
                _act_of_robot = {int(r): k for k, r in enumerate(self._action_joint_ids)}
                _idx29 = torch.tensor([_act_of_robot[int(r)] for r in self._sonic_perm.tolist()],
                                      device=dev, dtype=torch.long)                       # (29,)
                # g1 토큰은 기본자세 상대가 아니라 ABSOLUTE 관절각을 원합니다(motion_lib.get_dof_pos).
                # 속도는 같은 위치의 차분이라 두 채널이 어긋날 수 없습니다.
                self._ref_g1_q = self._ref_joints[:, _idx29]                              # (F,29)
                self._ref_g1_v = torch.zeros_like(self._ref_g1_q)
                self._ref_g1_v[1:] = (self._ref_g1_q[1:] - self._ref_g1_q[:-1]) * float(c.control_fps)
                # [ROLLBACK MARKER: ref-reset-jvel] 리셋용 _ref_joint_vel 과 여기 29열(SONIC 순서)이 같은지 검사.
                # 어긋나면 시뮬 관절 속도와 SONIC 이 듣는 속도 이력이 조용히 달라진다.
                if self._ref_joint_vel is not None:
                    _dv = (self._ref_joint_vel[:, _idx29] - self._ref_g1_v).abs().max()
                    assert _dv < 1e-3, f"[ref-reset-jvel] reset jvel != SONIC jvr: max diff {_dv:.3e}"
            self._sonic_scale = _SP.sonic_scale_vector(dev).view(1, -1)                        # (1,29) SONIC order
            # [ROLLBACK MARKER: hist-from-reference] 리셋 때 SONIC 관측 창(10프레임)을 채울 레퍼런스 이력.
            # 현재 행 복제로 채우면 "10프레임 정지"가 되어 토크나이저가 보는 레퍼런스 움직임과 어긋난다.
            self._ref_hist = None
            if self._ref_joints is not None:
                _jpr_r = self._ref_joints[:, _idx29] - self._sonic_default                     # (F,29)
                _jvr_r = torch.zeros_like(_jpr_r)
                _jvr_r[1:] = (_jpr_r[1:] - _jpr_r[:-1]) * float(c.control_fps)
                _q_r = self._ref_root_quat                                                     # (F,4)
                _g_r = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self._ref_len, 3)
                self._ref_hist = {
                    "jpr": _jpr_r,
                    "jvr": _jvr_r,
                    # 이 자세를 명령하는 행동 = 디코드의 역: body = default + scale*a  =>  a = jpr/scale
                    "act": _jpr_r / self._sonic_scale,
                    "grav": math_utils.quat_apply(math_utils.quat_conjugate(_q_r), _g_r),
                    "ang": math_utils.quat_apply(math_utils.quat_conjugate(_q_r), self._ref_root_angvel),
                }
            # [/ROLLBACK MARKER: hist-from-reference]
            # SONIC order -> action-body order (first 29 action joints = legs+waist+arms), by NAME
            _ab_names = [self.robot.joint_names[i] for i in self._action_joint_ids[:29]]
            self._sonic_gather = torch.tensor([list(_GIO).index(n) for n in _ab_names],
                                              device=dev, dtype=torch.long)                    # (29,)
            self._sonic_hand_slice = self._group_slices["hands"]
            # ── [ROLLBACK MARKER: joint-residual] 상체 잔차 색인. _upper_act_idx 는 액션-몸 순서(0..28),
            # _upper_sonic_idx 는 같은 관절의 SONIC 순서 색인 — 잔차는 SONIC 순서 벡터 a_sonic 에 더한다.
            if getattr(self, "_upper_on", False):
                _up_names = list(_ab_names) if self._res_mode == "all" else list(c.sonic_upper_residual_joints)
                _miss = [n for n in _up_names if n not in _ab_names]
                assert not _miss, f"[joint-residual] 액션 몸 관절 29개에 없는 이름: {_miss}"
                self._upper_act_idx = torch.tensor([_ab_names.index(n) for n in _up_names],
                                                   device=dev, dtype=torch.long)          # (N,) 액션-몸 순서
                self._upper_sonic_idx = self._sonic_gather[self._upper_act_idx]           # (N,) SONIC 순서
                # 그룹별 scale. _upper_act_idx 의 각 항목이 액션-몸 순서(0..28)이므로 _group_slices
                # 로 어느 그룹인지 정해진다.
                _grp = dict(c.sonic_residual_scale_groups)
                _sv = torch.empty(self._upper_act_idx.numel(), device=dev)
                for _gi, _ai in enumerate(self._upper_act_idx.tolist()):
                    _gn = next(g for g in ("legs", "waist", "arms")
                               if self._group_slices[g].start <= _ai < self._group_slices[g].stop)
                    assert _gn in _grp, f"[joint-residual] sonic_residual_scale_groups 에 '{_gn}' 이 없다"
                    _sv[_gi] = float(_grp[_gn])
                self._upper_res_scale = _sv                                               # (N,) SONIC 액션 단위
                _by_g = {g: sorted({float(_sv[i]) for i, a in enumerate(self._upper_act_idx.tolist())
                                    if self._group_slices[g].start <= a < self._group_slices[g].stop})
                         for g in ("legs", "waist", "arms")}
                print(f"[joint-residual] 그룹별 scale: " + "  ".join(
                    f"{g}({self._group_slices[g].stop - self._group_slices[g].start}관절)="
                    f"{','.join(f'{v:g}' for v in _by_g[g]) or '-'}" for g in ("legs", "waist", "arms")))
                self._diag_upper_res_rad = torch.zeros((), device=dev)
                self._diag_upper_clamp_frac = torch.zeros((), device=dev)
                _rad = self._upper_res_scale * self._sonic_scale[0, self._upper_sonic_idx]
                print(f"[joint-residual] 상체 {len(_up_names)}관절; |u|=1 일 때 관절 변위 "
                      f"min {float(_rad.min()):.4f} / max {float(_rad.max()):.4f} rad "
                      f"(= scale × SONIC 관절 스케일)")
            # ── [/ROLLBACK MARKER: joint-residual] ──
            # 10-frame proprio history (term-major, oldest-first) = playback flat_proprio layout
            self._sonic_hist = {k: torch.zeros(self.num_envs, _SP.PROPRIO_HIST, d0, device=dev)
                                for k, d0 in [("ang", 3), ("jpr", 29), ("jvr", 29), ("act", 29), ("grav", 3)]}
            self._sonic_hist_init = torch.ones(self.num_envs, dtype=torch.bool, device=dev)
            self._last_a_sonic = torch.zeros(self.num_envs, 29, device=dev)
            self._last_z_res = torch.zeros(self.num_envs, int(c.sonic_action_dim), device=dev)
            # RAW 100-D policy action (z_res + a_hand) for the GRAIL-style obs prev_action term AND the
            # action_rate reward. _cur = this step's action (set in _sonic_pre_physics_step); _prev = the
            # previous step's (lag-1, updated at the end of _get_observations). Seeded 0 at reset.
            _pa_dim = int(getattr(self, "_act_z", int(c.sonic_action_dim))) + int(c.hand_action_dim)
            if getattr(self, "_upper_on", False):                        # [joint-residual] + 몸 잔차 N
                _pa_dim += int(self._upper_act_idx.numel())
            self._cur_policy_action = torch.zeros(self.num_envs, _pa_dim, device=dev)
            self._prev_policy_action = torch.zeros(self.num_envs, _pa_dim, device=dev)
            # BOUNDED copy of the same action, used ONLY by rew_action_rate (see _get_rewards).
            # Each block is divided by its own env bound (z_res/z_res_clip, hand/1) so every entry
            # lives in [-1,1] regardless of the clip setting.
            self._cur_policy_action_bnd = torch.zeros(self.num_envs, _pa_dim, device=dev)
            self._prev_policy_action_bnd = torch.zeros(self.num_envs, _pa_dim, device=dev)
            # SONIC SMPL encoder arrays (already control_fps-resampled by parahome_smpl_for_sonic)
            assert self._np_sonic_smpl is not None, "SONIC smpl arrays missing (see _load_reference)"
            self._sonic_smpl_j = torch.from_numpy(self._np_sonic_smpl["smpl_joints_local"]).to(dev)   # (N,72)
            self._sonic_root_q = _canon(torch.from_numpy(self._np_sonic_smpl["root_q_zb"]).to(dev))   # (N,4)
            self._sonic_wrist_ref = torch.from_numpy(self._np_sonic_smpl["wrist_ref"]).to(dev)        # (N,6)
            assert self._sonic_smpl_j.shape[0] == self._ref_len, (
                f"SONIC smpl frames {self._sonic_smpl_j.shape[0]} != resampled ref_len {self._ref_len} "
                "(control_fps must match parahome_smpl_for_sonic TGT_FPS)")

    # ------------------------------------------------- action helpers
    def _unscale(self, q: torch.Tensor) -> torch.Tensor:
        return 2.0 * (q - self._ctrl_lower) / (self._ctrl_upper - self._ctrl_lower) - 1.0

    def _scale(self, a: torch.Tensor) -> torch.Tensor:
        return self._ctrl_lower + 0.5 * (a + 1.0) * (self._ctrl_upper - self._ctrl_lower)

    # ---- action: SONIC latent-residual body + residual hands ----
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # advance the reference frame for this step (reset overrides it at episode start)
        self._frame_idx = (self._frame_idx + 1).clamp(max=self._ref_len - 1)
        self._sonic_pre_physics_step(actions)             # SONIC latent-residual body + hand residual

    def _apply_action(self) -> None:
        target = self._residual_target                                 # (E,65) set by _sonic_pre_physics_step
        self.robot.set_joint_position_target(target, joint_ids=self._action_joint_ids)

    # -------------------------------------------------- frozen SONIC body prior
    def _sonic_proprio(self) -> torch.Tensor:
        """(E,930) SONIC decoder proprioception = flattened 10-frame history (term-major, oldest-
        first): [base_ang_vel 10×3 | joint_pos_rel 10×29 | joint_vel_rel 10×29 | last_action 10×29 |
        gravity_dir 10×3]. Matches sonic_playback.flat_proprio (verified)."""
        h = self._sonic_hist
        E = self.num_envs
        return torch.cat([h["ang"].reshape(E, -1), h["jpr"].reshape(E, -1), h["jvr"].reshape(E, -1),
                          h["act"].reshape(E, -1), h["grav"].reshape(E, -1)], dim=-1)

    def _sonic_anchor_q(self) -> torch.Tensor:
        """(E,4) 참조 자세를 상대화할 로봇 앵커 쿼터니언.

        [sonic-v11] heading 모드면 요만 남깁니다 — get_heading_q (torch_transform.py:391) 와 동일:
        x,y 성분을 0 으로 만들고 재정규화. 그러면 참조 모션의 롤·피치가 중력 기준으로 보존되고
        heading 차이만 제거됩니다. 원본 주석의 한계도 그대로 물려받습니다 — 로봇이 뒤집히면
        heading 이 불연속/미정의가 됩니다(우리는 넘어짐 구간에서 그 상황을 겪습니다).
        """
        q = _canon(self.robot.data.root_quat_w)                         # (E,4) wxyz
        if not getattr(self, "_sonic_heading", False):
            return q
        q = q.clone()
        q[..., 1] = 0.0
        q[..., 2] = 0.0
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    def _sonic_tokenizer(self) -> torch.Tensor:
        """(E,TOK) SONIC tokenizer obs in SMPL mode: encoder_index=[0,0,1] + the 10-frame future
        SMPL reference window (joints local, wrist ref, root orientation RELATIVE to the live
        pelvis). Non-SMPL terms are zero-filled. Matches sonic_playback.build_tok (verified)."""
        E, dev = self.num_envs, self.device
        lay = self._sonic_layout
        K = 10                                                          # SONIC multi-future window
        idx = (self._frame().unsqueeze(1)
               + torch.arange(K, device=dev).unsqueeze(0)).clamp(0, self._ref_len - 1)
        tok = torch.zeros(E, self._sonic_tok_dim, device=dev)
        # [ROLLBACK MARKER: sonic-encoder-g1] encoder_index 열 순서 = m.encoders = ['g1','teleop','smpl'].
        if self._sonic_enc == "g1":
            # [ROLLBACK MARKER: token-frame-skip] 미래 프레임 간격(상류: g1 = 5, 1.0 초 창 / smpl = 1, 0.2 초 창).
            # g1 인코더에 1 을 쓰면 SONIC 이 레퍼런스를 5배 느리게 읽는다. 되돌리기: sonic_token_frame_skip = 1
            _sk = max(1, int(getattr(self.cfg, "sonic_token_frame_skip", 1)))
            if _sk != 1:
                idx = (self._frame().unsqueeze(1)
                       + _sk * torch.arange(K, device=dev).unsqueeze(0)
                       ).clamp(0, self._ref_len - 1)
            # ── [/ROLLBACK MARKER: token-frame-skip] ─────────────────────────────────────
            s_, e_, _ = lay["encoder_index"]; tok[:, s_:e_] = torch.tensor([1.0, 0.0, 0.0], device=dev)
            # 평탄 580 = [pos(f..f+9) 29개] ++ [vel(f..f+9) 29개]. 프레임당 [pos|vel]가 아닙니다
            # (디코더가 푸는 방식과 commands.py의 cat([joint_pos_mf, joint_vel_mf])에서 확인).
            s_, e_, _ = lay["command_multi_future_nonflat"]
            tok[:, s_:e_] = torch.cat([self._ref_g1_q[idx].reshape(E, -1),
                                       self._ref_g1_v[idx].reshape(E, -1)], dim=-1)
            pelvis_q = self._sonic_anchor_q()                           # [sonic-v11] heading 이면 요만
            rq = self._sonic_root_q[idx]
            dif = self._sonic_qmul(self._sonic_qinv(pelvis_q).unsqueeze(1).expand(E, K, 4), rq)
            ori6 = math_utils.matrix_from_quat(dif.reshape(-1, 4))[..., :2].reshape(E, K, 6)
            s_, e_, _ = lay[self._sonic_key_ori_g1]; tok[:, s_:e_] = ori6.reshape(E, -1)
            return tok
        s, e, _ = lay["encoder_index"]; tok[:, s:e] = torch.tensor([0.0, 0.0, 1.0], device=dev)
        s, e, _ = lay["smpl_joints_multi_future_local_nonflat"]; tok[:, s:e] = self._sonic_smpl_j[idx].reshape(E, -1)
        s, e, _ = lay["joint_pos_multi_future_wrist_for_smpl"]; tok[:, s:e] = self._sonic_wrist_ref[idx].reshape(E, -1)
        pelvis_q = self._sonic_anchor_q()                               # [sonic-v11] heading 이면 요만
        rq = self._sonic_root_q[idx]                                    # (E,10,4) reference root (Z-up)
        dif = self._sonic_qmul(self._sonic_qinv(pelvis_q).unsqueeze(1).expand(E, K, 4), rq)  # (E,10,4) robot-relative
        ori6 = math_utils.matrix_from_quat(dif.reshape(-1, 4))[..., :2].reshape(E, K, 6)
        s, e, _ = lay[self._sonic_key_ori_smpl]; tok[:, s:e] = ori6.reshape(E, -1)
        return tok

    def _sonic_pre_physics_step(self, actions: torch.Tensor) -> None:
        """Body(29) = frozen SONIC decoder with a pre-quantization latent residual z_res(64);
        hands(36) = ABSOLUTE action (user-locked) mapped directly to the Shadow joint range,
        per-group EMA-smoothed. Sets the combined 65-D PD target and mirrors it into
        _smoothed_actions so the RSI state cache stays unchanged (kept 65-D)."""
        c = self.cfg
        E, dev = self.num_envs, self.device
        _az = int(getattr(self, "_act_z", int(c.sonic_action_dim)))      # 잠재 블록 폭 (끄면 0)
        z_raw = actions[:, :_az]                                         # (E,64) 또는 (E,0)
        # raw residual, CLIPPED to [-z_res_clip, z_res_clip] (user 2026-07-23, =5.0) so the frozen-SONIC
        # decoder never sees extreme latents (bounds the physical body residual; NOTE this bounds the ENV
        # action, not the PPO log-prob which is on the raw sample).
        z_res = torch.clamp(z_raw, -c.sonic_z_res_clip, c.sonic_z_res_clip)
        if _az == 0:                                                     # [joint-residual] 잠재 잔차 OFF
            z_res = torch.zeros(E, int(c.sonic_action_dim), device=dev)  # 디코더에는 0 → 순수 SONIC
        # [joint-residual] 손 블록은 반드시 닫힌 슬라이스로 — 상체 블록이 뒤에 붙으므로 열린 슬라이스면
        # a_hand 가 53 차원이 된다. _hs = [64,100), 상체 = [100, 117).
        _hs = slice(_az, _az + int(c.hand_action_dim))
        a_hand = actions[:, _hs].clamp(-1.0, 1.0)                        # (E,36)
        # saturation diagnostics: what FRACTION of each block's raw sample is being flattened by the
        # env clamp. Both blocks have a flat exterior (no restoring gradient there), so a rising
        # fraction is the early signature of a mean random-walk. Logged under "Diag /".
        self._diag_hand_clamp_frac = (actions[:, _hs].abs() > 1.0).float().mean()
        self._diag_zres_clip_frac = ((z_raw.abs() > c.sonic_z_res_clip).float().mean() if _az > 0
                                     else torch.zeros((), device=dev))
        # rew_action_reg 의 잠재 블록은 클립 전 raw z_res 를 쓴다.
        # 클립된 값을 벌하면 포화된 뒤 mu 를 안으로 되돌리는 기울기가 없어진다.
        self._last_z_res = z_raw
        self._cur_policy_action = actions                                # raw policy action (100, +17 if joint-residual)
        # 블록별 한계로 [-1,1] 정규화한 실제 적용값(클립된 z_res / 클램프된 손).
        # 지금은 읽는 곳이 없고 action_rate A/B 전환용으로만 유지한다.
        _zb = max(float(c.sonic_z_res_clip), 1e-6)
        self._cur_policy_action_bnd = torch.cat(
            ([z_res / _zb] if _az > 0 else []) + [a_hand]
            + ([actions[:, _hs.stop:].clamp(-1.0, 1.0)] if getattr(self, "_upper_on", False) else []),
            dim=-1)   # [joint-residual] 상체 블록 포함
        # frozen SONIC body: encode SMPL ref -> latent +λ·z_res (pre-quant) -> FSQ -> g1_dyn decode
        proprio = self._sonic_proprio()                                 # (E,930)
        tok = self._sonic_tokenizer()                                   # (E,TOK)
        latent = self._SP.encode_latent(self._sonic, tok, encoder=self._sonic_enc)
        a_sonic = self._SP.residual_decode(self._sonic, latent, z_res, proprio,
                                           float(c.residual_scale_latent))   # (E,29) SONIC order, raw
        self._last_a_sonic = a_sonic
        # [ROLLBACK MARKER: joint-residual] 상체 관절 잔차 a_out[S] += scale·u 는 _sonic_scale 을 곱하기 전
        # SONIC 액션 단위에서 더한다(video_to_data 와 같음). tanh·EMA 없음, S 밖의 다리 12관절은 SONIC 출력 그대로.
        a_out = a_sonic
        if getattr(self, "_upper_on", False):
            _u = actions[:, _hs.stop:]                                  # (E,N) raw, 무제한
            _si = self._upper_sonic_idx
            a_out = a_sonic.clone()
            _rs = self._upper_res_scale                                  # (N,) 그룹별 scale
            a_out[:, _si] = a_out[:, _si] + _rs * _u
            self._diag_upper_res_rad = (_rs * _u.abs() * self._sonic_scale[:, _si]).mean()
        # ── [/ROLLBACK MARKER: joint-residual] ──
        body_sonic = self._sonic_default + self._sonic_scale * a_out    # (E,29) SONIC order absolute target
        body_target = body_sonic[:, self._sonic_gather]                # (E,29) action-body order
        # hands: sonic_hand_residual=True(기본)면 기준(_hand_res_base) 대비 margin 잔차 + 잔차 EMA.
        # False 면 ABSOLUTE — a_hand ∈ [-1,1] 를 Shadow 관절 범위로 직접 사상하고 hands EMA 를 건다
        # (EMA prev = _smoothed_actions[hands], _reset_idx 에서 리셋 자세로 씨딩).
        hsl = self._sonic_hand_slice
        if getattr(c, "sonic_hand_residual", False):
            # [ROLLBACK MARKER: stage1-hand] 잔차 손 액션 — 1단계(hand_pretrain) 와 같은 마진 매핑 + 정규화
            # 잔차 전용 EMA. a=0 이면 기준(_hand_res_base: 1단계 롤아웃 PD 목표 또는 레퍼런스)을 지연 없이
            # 그대로 재생하고, a=±1 은 그 관절의 상/하한에 정확히 닿습니다.
            _b = self._hand_res_base[self._frame()]                                        # (E,36) 관절 단위
            _al = self._finger_alpha
            self._hand_res_ema.mul_(1.0 - _al).add_(_al * a_hand)
            _r = self._hand_res_ema
            hand_target = torch.clamp(_b + torch.where(_r >= 0, _r * (self._ctrl_upper[hsl] - _b),
                                                       _r * (_b - self._ctrl_lower[hsl])),
                                      self._ctrl_lower[hsl], self._ctrl_upper[hsl])
            # [/ROLLBACK MARKER: stage1-hand]
        else:                                                           # ABSOLUTE (default): a_hand → joint range (EMA)
            alpha_h = self._finger_alpha
            smoothed_hand = alpha_h * a_hand + (1.0 - alpha_h) * self._smoothed_actions[:, hsl]
            hand_target = (self._ctrl_lower[hsl]
                           + 0.5 * (smoothed_hand + 1.0) * (self._ctrl_upper[hsl] - self._ctrl_lower[hsl]))
        target = torch.empty(E, self._n_act, device=dev)
        target[:, :hsl.start] = body_target                            # [0:29] = body (legs+waist+arms, SONIC)
        target[:, hsl] = hand_target                                   # [29:65] = bimanual hands (ABSOLUTE)
        self._residual_target = torch.clamp(target, self._ctrl_lower, self._ctrl_upper)
        # [ROLLBACK MARKER: joint-residual] 상체 목표가 soft 관절 한계에 잘린 비율 (video_to_data 는 클램프가
        # 없고 joint_pos_limits 보상으로만 누르지만, 이 env 는 위 한 곳에서 잘라낸다).
        if getattr(self, "_upper_on", False):
            _ui = self._upper_act_idx
            self._diag_upper_clamp_frac = (target[:, _ui] != self._residual_target[:, _ui]).float().mean()
        self._smoothed_actions = self._unscale(self._residual_target)

    # ------------------------------------------------- robot keypoint / fingertip FK
    def _robot_kpts_w(self) -> torch.Tensor:
        """(E,56,3) world keypoint positions = body origin + rotated local offset."""
        p = self.robot.data.body_pos_w[:, self._kpt_body_ids]          # (E,56,3)
        q = self.robot.data.body_quat_w[:, self._kpt_body_ids]         # (E,56,4)
        off = self._kpt_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        return p + math_utils.quat_apply(q, off)

    def _robot_ft_w(self):
        """(E,10,3) fingertip pad positions + (E,10,3) pad-inward world directions."""
        p = self.robot.data.body_pos_w[:, self._ft_body_ids]
        q = self.robot.data.body_quat_w[:, self._ft_body_ids]
        tip = p + math_utils.quat_apply(q, self._ft_offsets.unsqueeze(0).expand(self.num_envs, -1, -1))
        pad_inward = -math_utils.quat_apply(q, self._ft_pad_normals.unsqueeze(0).expand(self.num_envs, -1, -1))
        return tip, pad_inward

    def _link_pad_inward_w(self) -> torch.Tensor:
        """(E,L,3) each wrap link's INWARD pad/palmar unit normal in world = -R(link_quat)·pad_outward_local.
        The direction the object presses when the link touches with its correct (grasping) face — the analog
        of the fingertip `pad_inward`. Used both to project the contact force and for the orientation gate."""
        q = self.robot.data.body_quat_w[:, self._link_contact_body_ids]              # (E,L,4)
        return -math_utils.quat_apply(
            q, self._link_pad_normals.unsqueeze(0).expand(self.num_envs, -1, -1))    # (E,L,3) inward

    def _link_contact_forces(self) -> torch.Tensor:
        """(E,L) per-link COMPRESSIVE contact force with the object (Option A): the object-filtered contact
        force on each wrap link (force_matrix_w) projected on the link's OWN INWARD pad normal (like the
        fingertip `force·(-pad_normal)`), clamped ≥0 — so a link pressed on the WRONG face registers ~0.
        0 where no object / no contact. L = N_LINK_CONTACT. (Was: projected on the object-anchored reference
        reaction normal, which ignored which face of the robot link was touching.)"""
        out = torch.zeros(self.num_envs, N_LINK_CONTACT, device=self.device)
        if not (self._has_object and self._has_link_contact):
            return out
        inward_w = self._link_pad_inward_w()                          # (E,L,3) link face inward (world)
        for i, s in enumerate(self._link_contact_sensors):
            fm = s.data.force_matrix_w                                 # (E,1,1,3) object-filtered, or None
            if fm is None:
                continue
            f = fm.reshape(self.num_envs, -1, 3).sum(dim=1)            # (E,3) object→link force
            out[:, i] = (f * inward_w[:, i]).sum(-1).clamp_min(0.0)    # compressive on the link's own face
        return out

    @property
    def is_reached_end(self) -> bool:
        """Curriculum reached (near) the trajectory end → switch the cache quality gate to the
        tighter 'late' object thresholds (matches grasp).

        [ROLLBACK MARKER: late-gate] set by _reset_idx when one episode both ran for
        late_gate_survival_frac of the clip and finished within 3 frames of the end. `_reached_frame`
        is still maintained, but it only feeds `Curriculum / reached_frame` now — it answers "was a
        late frame ever cached", which RSI makes true immediately and which says nothing about
        whether the policy can actually get there."""
        return bool(getattr(self, "_late_gate", False))

    def _frame(self) -> torch.Tensor:
        return self._frame_idx.clamp(max=self._ref_len - 1)

    def _next_frame(self) -> torch.Tensor:
        return (self._frame_idx + 1).clamp(max=self._ref_len - 1)

    # ------------------------------------------------------------ observation
    def _get_observations(self) -> dict:
        c = self.cfg
        E, vs = self.num_envs, c.vel_obs_scale
        org = self.scene.env_origins                                    # (E,3)
        fr, nfr = self._frame(), self._next_frame()

        # ---- BLOCK A: proprioception ----
        root_pos = self.robot.data.root_pos_w
        root_quat = _canon(self.robot.data.root_quat_w)
        # explicit palm (wrist) state + fingertip velocities (bimanual) — direct manipulation
        # signals (mirrors grasp's wrist quat/linvel/angvel + fingertip velocities; the palm
        # keypoint POSITION alone loses orientation/velocity). Real robot values in BOTH phases.
        palm_quat = _canon(self.robot.data.body_quat_w[:, self._palm_body_ids])     # (E,2,4)
        palm_linvel = self.robot.data.body_lin_vel_w[:, self._palm_body_ids]        # (E,2,3)
        palm_angvel = self.robot.data.body_ang_vel_w[:, self._palm_body_ids]        # (E,2,3)
        ft_vel = self.robot.data.body_lin_vel_w[:, self._ft_body_ids]               # (E,10,3)
        # projected_gravity_b = gravity direction in the base frame → encodes base TILT (roll/pitch), the
        # signal the residual policy needs to perceive & correct balance / forward-fall (added 2026-07-21
        # for the CoM-over-support balance reward; root height/ori6d stay out — recoverable as ref − delta).
        A = [
            self.robot.data.projected_gravity_b,                       # (3) base-frame gravity dir (tilt)
            self.robot.data.root_lin_vel_w,                            # (3)
            self.robot.data.root_ang_vel_w * vs,                       # (3)
            self._unscale(self.robot.data.joint_pos[:, self._action_joint_ids_t]),  # (65)
            self.robot.data.joint_vel[:, self._action_joint_ids_t] * vs,            # (65)
            _quat_to_6d(palm_quat).reshape(E, -1),                     # palm ori 6d ×2 (12)
            palm_linvel.reshape(E, -1),                               # palm linvel ×2 (6)
            (palm_angvel * vs).reshape(E, -1),                        # palm angvel ×2 (6)
            ft_vel.reshape(E, -1),                                     # fingertip linvel ×10 (30)
        ]

        # ---- BLOCK B: reference tracking + look-ahead ----
        kpts = self._robot_kpts_w()                                    # (E,54,3) world (14 body + 40 hand)
        kpts_local = kpts - org[:, None, :]                            # env-local
        delta_kpts = self._ref_kpts[nfr] - kpts_local                  # look-ahead delta
        ref_root_p = self._ref_root_pos[nfr]                           # look-ahead (grasp-parity: obs deltas = next frame)
        ref_root_q = self._ref_root_quat[nfr]
        delta_root_pos = ref_root_p - (root_pos - org)
        droot_q = _canon(math_utils.quat_mul(ref_root_q, math_utils.quat_conjugate(root_quat)))
        # NO phase/time signal (grasp env doesn't use one — progress is conveyed by the next-frame
        # reference deltas / look-ahead below, keeping obs consistent with the existing tasks).
        B = [
            kpts_local.reshape(E, -1),                                 # (54×3=162)
            delta_kpts.reshape(E, -1),                                 # (54×3=162)
            ref_root_p,                                                # ref root pos (3)  [was height-only]
            _quat_to_6d(ref_root_q),                                   # ref root ori 6d (6)
            delta_root_pos,                                            # (3)
            _quat_to_6d(droot_q),                                      # (6)
        ]

        # ---- BLOCK C: object + contact + history ----
        if self._has_object:
            obj_p = self._object.data.root_pos_w - org
            obj_q = _canon(self._object.data.root_quat_w)
            obj_lv = self._object.data.root_lin_vel_w
            obj_av = self._object.data.root_ang_vel_w
        else:
            obj_p = self._ref_obj_pos[fr]
            obj_q = self._ref_obj_quat[fr]
            obj_lv = torch.zeros(E, 3, device=self.device)
            obj_av = torch.zeros(E, 3, device=self.device)
        delta_obj_p = self._ref_obj_pos[nfr] - obj_p                       # look-ahead delta (next-frame ref)
        delta_obj_q = _canon(math_utils.quat_mul(self._ref_obj_quat[nfr], math_utils.quat_conjugate(obj_q)))
        artic = torch.zeros(E, self._RESERVE_ARTIC * 2, device=self.device)
        if self._n_obj_parts > 0:
            p = min(self._n_obj_parts, self._RESERVE_ARTIC)
            artic[:, :p] = self._ref_obj_dof[nfr, :p]                  # ref DOF, look-ahead (next frame)
        _tip = self._robot_ft_w()[0]
        # delta_ft_obj: (손끝 목표 - 손끝)을 물체 좌표계로 표현(물체 자세 불변). 부호는 다른 delta 관측처럼 ref - robot.
        # 목표 = 접촉 예정 손가락은 distal 링크의 물체 표면 접촉점, 나머지는 레퍼런스 pad.
        ref_ft_w = self._ref_ft_pad[nfr] + org[:, None, :]            # (E,10,3) reference pad (world), look-ahead
        oq_exp3 = obj_q[:, None, :].expand(-1, 10, -1)                # (E,10,4)
        if self._has_link_contact:
            obj_pos_w = (obj_p + org)[:, None, :]                     # (E,1,3) world object origin
            dt = self._ref_link_contact_target_local[nfr][:, self._ft_distal_idx]     # (E,10,3) distal target (obj-local)
            ref_vertex_w = math_utils.quat_apply(oq_exp3, dt) + obj_pos_w
            # [contact-vel-gate] 손끝 목표 전환에도 같은 게이트 (grasp env 의 contact_flag_next 와
            # 동일). 물체가 정지한 프레임에서는 목표가 표면 접촉점 → 레퍼런스 패드로 되돌아간다.
            in_contact = (self._ref_link_contact_mask[nfr][:, self._ft_distal_idx]
                          * self._ref_obj_vel_gate[nfr].unsqueeze(-1)).unsqueeze(-1).bool()  # (E,10,1)
            target_w = torch.where(in_contact, ref_vertex_w, ref_ft_w)
        else:
            target_w = ref_ft_w
        delta_ft_obj = math_utils.quat_apply(math_utils.quat_conjugate(oq_exp3), target_w - _tip)  # (E,10,3) ref − current
        C = [
            obj_p, _quat_to_6d(obj_q), obj_lv, obj_av * vs,            # (15)
            delta_obj_p, _quat_to_6d(delta_obj_q),                              # (9)
            delta_ft_obj.reshape(E, -1),                              # obj-local fingertip offset (30)
            artic,                                                    # (8)
            # [contact-vel-gate] 보상과 같은 게이트를 씌운다 — 보상이 0 인 프레임을 관측에서
            # 구분할 수 없으면 정책이 게이트를 학습할 수 없다 (grasp env 도 obs/보상 모두
            # 게이트된 future_contact 를 썼다). 관측 차원은 그대로라 기존 체크포인트 호환.
            self._ref_link_contact_mask[nfr] * self._ref_obj_vel_gate[nfr].unsqueeze(-1),  # FUTURE per-link expected contact (L=32)
            # OBS force CLIP to force_obs_clip (user 2026-07-23, =300N): raw contact/foot forces spike
            # to ~hundreds of N (var ~7e5 measured), destabilizing the obs RunningStandardScaler. Clip
            # the OBS copy only (the reward uses its own contact_force_cap). Keeps scaling stable.
            self._link_contact_forces().clamp(max=c.force_obs_clip),  # current per-link actual contact force (L=32)
            self._ref_foot_contact[nfr],                              # FUTURE (look-ahead) reference foot contact L/R (2)
            self._foot_force().clamp(max=c.force_obs_clip),           # current ACTUAL foot↔ground force L/R (2)
            # 이전 정책 액션(raw, GRAIL 방식). 클립·정규화한 사본을 넣어 봤지만(2026-07-28) 효과가 없어 되돌렸다.
            self._prev_policy_action,
        ]

        # 역방향 롤아웃(제거됨)의 진행 방향 비트가 있던 자리. 값은 항상 0 이지만 관측 차원을
        # 바꾸면 기존 체크포인트를 쓸 수 없으므로 자리를 남긴다.
        C = C + [torch.zeros(E, 1, device=self.device)]
        obs = torch.cat(A + B + C, dim=-1)
        # [ROLLBACK MARKER: nan-guard] (L3) 관측 위생: RunningStandardScaler 는 inf 하나로 영구 오염되므로
        # 비유한 값을 0 으로 바꾸고 크기를 제한한다. Diag / obs_sanitized 가 0 이 아니면 게이트가 놓친 NaN 경로가 있다.
        _oc = float(getattr(c, "nan_guard_obs_clip", 0.0))
        if _oc > 0.0:
            _bad = ~torch.isfinite(obs)
            self._diag_obs_sanitized = _bad.float().mean()
            if _bad.any():
                obs = torch.nan_to_num(obs, nan=0.0, posinf=_oc, neginf=-_oc)
            obs = obs.clamp(-_oc, _oc)
        # ── [/ROLLBACK MARKER: nan-guard] ──
        assert obs.shape[-1] == c.observation_space, (
            f"obs dim {obs.shape[-1]} != cfg.observation_space {c.observation_space} "
            "(block-C dims must be invariant across the has_object flip)")
        # capture prev actions for NEXT step (lag-1). The RAW copy (_*_policy_action) feeds the obs
        # (above) and the drift diagnostics (Diag/zres_absmax, Diag/hand_absmax); the BOUNDED copy
        # (_*_bnd) feeds action_rate only. Both are maintained so either can be swapped in.
        self._prev_policy_action = self._cur_policy_action.clone()
        self._prev_policy_action_bnd = self._cur_policy_action_bnd.clone()

        # ---- SONIC 10-frame proprio history update (POST-step; sonic_playback parity) ----
        # shift-append the newest row; freshly-reset envs get all 10 slots seeded from the current
        # (post-reset) row with last_action=0 → SONIC's first step after reset is in-distribution.
        if self._sonic is not None:
            rows = {
                "ang": self.robot.data.root_ang_vel_b,
                "jpr": self.robot.data.joint_pos[:, self._sonic_perm] - self._sonic_default,
                "jvr": self.robot.data.joint_vel[:, self._sonic_perm],
                "act": self._last_a_sonic,
                "grav": self.robot.data.projected_gravity_b,
            }
            for k, v in rows.items():
                self._sonic_hist[k][:, :-1] = self._sonic_hist[k][:, 1:].clone()
                self._sonic_hist[k][:, -1] = v
            if bool(self._sonic_hist_init.any()):
                m = self._sonic_hist_init
                # [ROLLBACK MARKER: hist-from-reference] 창 10칸 전부를 레퍼런스의 지난 10프레임으로 채운다.
                # 0 에서 자른다. 최근 칸만 실측으로 두면 8·9번 칸 사이 위치 점프가 속도 채널과 모순된다.
                _hist_ref_used = (bool(getattr(c, "sonic_hist_from_reference", True))
                                  and self._ref_hist is not None)
                if _hist_ref_used:
                    _H = self._sonic_hist["jpr"].shape[1]                       # 10
                    _ep = self._frame().unsqueeze(1) - torch.arange(_H - 1, -1, -1, device=self.device)
                    _ep = _ep.clamp(min=0)                                      # (E,H) 레퍼런스 프레임
                    for k in ("jpr", "grav", "act", "jvr", "ang"):
                        self._sonic_hist[k][m] = self._ref_hist[k][_ep][m]
                else:
                    for k in ("ang", "jpr", "jvr", "grav"):
                        self._sonic_hist[k][m] = rows[k][m].unsqueeze(1)
                if not _hist_ref_used:
                    # 행동 이력은 0 으로 둡니다. IsaacLab 의 행동 관리자가 리셋 때 행동을 0 으로 만들므로,
                    # 동결 디코더가 학습 중 실제로 본 값이 0 입니다.
                    self._sonic_hist["act"][m] = 0.0
                # [/ROLLBACK MARKER: hist-from-reference] ------------------------------------------
                self._sonic_hist_init[m] = False

        if c.debug_vis:
            self._update_debug_vis(self._ref_kpts[fr] + org[:, None, :], kpts, ref_ft_w)
        return {"policy": obs}

    # ---------------------------------------------------- debug visualization
    def _setup_debug_vis(self) -> None:
        """Spawn reference-keypoint markers: REFERENCE keypoints (green), robot ACTUAL keypoints
        (cyan), reference fingertip-pad targets (magenta). Only the first debug_vis_num_envs envs."""
        self._debug_vis_n = min(self.cfg.debug_vis_num_envs, self.num_envs)

        def _spheres(path: str, radius: float, color: tuple) -> VisualizationMarkers:
            return VisualizationMarkers(VisualizationMarkersCfg(
                prim_path=path,
                markers={"sphere": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))}))

        self._vis_ref_kpts = _spheres("/Visuals/debug/ref_kpts", 0.010, (0.0, 1.0, 0.0))     # reference target
        self._vis_robot_kpts = _spheres("/Visuals/debug/robot_kpts", 0.008, (0.0, 0.8, 1.0))  # robot actual
        self._vis_ref_ft = _spheres("/Visuals/debug/ref_ft_pad", 0.006, (1.0, 0.0, 1.0))      # ref fingertip pads
        print(f"[g1] Debug vis ON for first {self._debug_vis_n} env(s): "
              "ref kpts=green, robot kpts=cyan, ref fingertip pads=magenta.")

    def _update_debug_vis(self, ref_kpts_w: torch.Tensor, robot_kpts_w: torch.Tensor,
                          ref_ft_w: torch.Tensor) -> None:
        """ref_kpts_w/robot_kpts_w:(E,56,3), ref_ft_w:(E,10,3) — world frame. Draw the first n envs."""
        n = self._debug_vis_n
        self._vis_ref_kpts.visualize(translations=ref_kpts_w[:n].reshape(-1, 3))
        self._vis_robot_kpts.visualize(translations=robot_kpts_w[:n].reshape(-1, 3))
        self._vis_ref_ft.visualize(translations=ref_ft_w[:n].reshape(-1, 3))

    def _com_support_err(self) -> torch.Tensor:
        """(E,) out-of-support excess (m) of the mass-weighted CoM horizontal projection, in the foot frame:
        relu(e_fwd - L_front) + relu(-e_fwd - L_back) + relu(|e_lat| - L_side),  L_side = ½‖aL-aR‖ + halfw.
        0 when the CoM is inside the support box (feasibility → no penalty on the balanced reference)."""
        c = self.cfg
        rd = self.robot.data
        pos = rd.body_com_pos_w if self._use_body_com else rd.body_pos_w          # (E,nB,3) world
        m = self._body_masses                                                    # (E,nB)
        com = (m.unsqueeze(-1) * pos).sum(1) / m.sum(1, keepdim=True)             # (E,3) world CoM
        aL = rd.body_pos_w[:, self._ankle_body_ids[0]]                            # (E,3)
        aR = rd.body_pos_w[:, self._ankle_body_ids[1]]
        cen = 0.5 * (aL + aR)                                                     # (E,3) support center
        bf = math_utils.quat_apply(rd.root_quat_w, self._x_axis)                  # (E,3) base +X in world
        fwd = bf[:, :2] / (bf[:, :2].norm(dim=-1, keepdim=True) + 1e-8)           # (E,2) facing (xy)
        lat = torch.stack([-fwd[:, 1], fwd[:, 0]], dim=-1)                        # (E,2) lateral
        d = com[:, :2] - cen[:, :2]                                               # (E,2)
        e_f = (d * fwd).sum(-1); e_l = (d * lat).sum(-1)                          # (E,)
        l_side = 0.5 * (aL[:, :2] - aR[:, :2]).norm(dim=-1) + c.com_support_foot_halfw
        return ((e_f - c.com_support_l_front).clamp(min=0.0)
                + (-e_f - c.com_support_l_back).clamp(min=0.0)
                + (e_l.abs() - l_side).clamp(min=0.0))                            # (E,) ≥ 0

    def _foot_force(self) -> torch.Tensor:
        """(E,2)=[left,right] COMPRESSIVE foot↔ground force (N). = the ground-filtered contact force
        (force_matrix_w) projected on the foot SOLE-INWARD normal (= -sole normal, ≈ up when flat),
        clamped ≥0 — mirrors the fingertip force computation. Used by the obs + (thresholded) the reward."""
        q = self.robot.data.body_quat_w[:, self._ankle_body_ids]                    # (E,2,4)
        sole_inward = -math_utils.quat_apply(                                        # (E,2,3) into the foot (≈up)
            q, self._foot_sole_normal.unsqueeze(0).expand(self.num_envs, -1, -1))
        comp = torch.zeros(self.num_envs, 2, device=self.device)
        for i, s in enumerate(self._foot_sensors):
            fm = s.data.force_matrix_w                                              # (E,1,1,3) ground-filtered, or None
            if fm is None:
                continue
            f = fm.reshape(self.num_envs, -1, 3).sum(dim=1)                         # (E,3) ground→foot force
            comp[:, i] = (f * sole_inward[:, i]).sum(-1).clamp_min(0.0)             # compressive
        return comp                                                                 # (E,2) N

    def _foot_contact_actual(self) -> torch.Tensor:
        """(E,2)=[left,right] binary ACTUAL ground contact = compressive foot force > threshold."""
        return (self._foot_force() > self.cfg.foot_contact_force_thresh).float()    # (E,2)

    # ── [ROLLBACK MARKER: wrist-rot] 손목 회전: 사람 손과 로봇 손에 같은 키포인트 기하로 좌표계를 세워 비교한다.
    # z = 손목→중지 MCP, x = z × (손목→검지 MCP), y = z × x. 레퍼런스의 고정 차이(오 11.8°, 왼 32.5°)는 보정하지 않는다.
    @staticmethod
    def _landmark_frame(wrist: torch.Tensor, mid_mcp: torch.Tensor, idx_mcp: torch.Tensor) -> torch.Tensor:
        """세 점 (...,3) -> 회전행렬 (...,3,3). 열이 x,y,z 축."""
        z = mid_mcp - wrist
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        x = torch.cross(z, idx_mcp - wrist, dim=-1)
        x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return torch.stack([x, torch.cross(z, x, dim=-1), z], dim=-1)

    def _wrist_rot_err(self, ref: torch.Tensor, kpts: torch.Tensor) -> torch.Tensor:
        """(E,) 양손 평균 손목 회전 오차 (rad). ref/kpts 모두 (E,K,3) env-로컬."""
        errs = []
        for w, m, i2 in self._wrist_frame_idx:                      # 손별 (손목, 중지MCP, 검지MCP)
            Rr = self._landmark_frame(ref[:, w], ref[:, m], ref[:, i2])
            Rb = self._landmark_frame(kpts[:, w], kpts[:, m], kpts[:, i2])
            d = torch.einsum("eji,ejk->eik", Rb, Rr)                # Rb^T Rr
            # 회전행렬 -> 각도: trace = 1 + 2cos(theta)
            tr = d[:, 0, 0] + d[:, 1, 1] + d[:, 2, 2]
            errs.append(torch.arccos(((tr - 1.0) * 0.5).clamp(-1.0, 1.0)))
        return torch.stack(errs, dim=-1).mean(dim=-1)

    # ---------------------------------------------------------------- rewards
    def _compute_errors(self):
        """Shared error terms for reward + termination."""
        org = self.scene.env_origins
        fr = self._frame()
        kpts = self._robot_kpts_w() - org[:, None, :]
        ref = self._ref_kpts[fr]
        dk = ref - kpts
        _nb = len(BODY_KPTS)                                          # 14 body kpts (GRAIL-aligned)
        body_per = dk[:, :_nb].norm(dim=-1)                          # (E,14) per-body-kpt distance
        body_err = body_per.mean(dim=-1)                            # (E,) UNIFORM mean → termination gate
        body_core_err = body_per[:, self._body_core_idx].mean(dim=-1)  # (E,) 9 CORE body kpts (REWARD)
        # [ee-split] wrist and ankle are now separate REWARD groups (they were one 4-kpt "ee" mean).
        wrist_pos_err = body_per[:, self._wrist_kpt_idx].mean(dim=-1)  # (E,) MEAN over both wrists (termination + 로그)
        ee_err = body_per[:, self._ee_kpt_idx].mean(dim=-1)            # (E,) MEAN over ankle×2 + torso (REWARD) [ee-torso]
        hand_err = dk[:, _nb:].norm(dim=-1).mean(dim=-1)            # (E,) 40 hand kpts — 무가중 (로그)
        # [ROLLBACK MARKER: z-weight] 보상용 z 가중 사본. 무가중은 로그/게이트에 그대로 남깁니다.
        _zw = float(self.cfg.z_weight_reward)
        _dkh = dk[:, _nb:].clone()
        _dkh[..., 2] *= _zw
        hand_err_w = _dkh.norm(dim=-1).mean(dim=-1)                 # (E,) 40 hand kpts — 가중 (보상)
        # fingertip pad tracking (contact-conditioned handled in reward; raw here)
        tip, pad_inward = self._robot_ft_w()
        ft_per = (self._ref_ft_pad[fr] - (tip - org[:, None, :])).norm(dim=-1)   # (E,10) = L[5] then R[5]
        # UNIFORM MEAN over all 10 fingertip pads (both hands) — consistent with body / wrist_pos / wrist_rot,
        # all of which use a plain mean for the termination gate (no per-hand worst-of-two-hands max).
        ft_err = ft_per.mean(dim=-1)                                            # (E,)
        # root
        root_pos = self.robot.data.root_pos_w - org
        root_quat = _canon(self.robot.data.root_quat_w)
        root_pos_err = (self._ref_root_pos[fr] - root_pos).norm(dim=-1)
        qerr = _canon(math_utils.quat_mul(self._ref_root_quat[fr], math_utils.quat_conjugate(root_quat)))
        root_rot_err = 2.0 * torch.arcsin(qerr[:, 1:].norm(dim=-1).clamp(max=1.0))
        # object
        if self._has_object:
            obj_pos = self._object.data.root_pos_w - org
            obj_quat = _canon(self._object.data.root_quat_w)
            _dop = self._ref_obj_pos[fr] - obj_pos                              # (E,3)
            obj_pos_err = _dop.norm(dim=-1)                                      # 무가중 (게이트/로그)
            _dopw = _dop.clone(); _dopw[:, 2] *= _zw                             # [z-weight]
            obj_pos_err_w = _dopw.norm(dim=-1)                                   # 가중 (보상)
            oq = _canon(math_utils.quat_mul(self._ref_obj_quat[fr], math_utils.quat_conjugate(obj_quat)))
            obj_rot_err = 2.0 * torch.arcsin(oq[:, 1:].norm(dim=-1).clamp(max=1.0))
            # 손끝 보상 목표(grasp_rsi 방식): 접촉 손가락은 현재 물체 프레임의 접촉 정점, 나머지는 레퍼런스 pad 를
            # 현재 물체 프레임으로 옮긴 값(드리프트 보정). 종료·로그는 위의 raw ft 를 쓴다.
            tip_l = tip - org[:, None, :]                                          # (E,10,3) env-local
            oq_e = obj_quat.unsqueeze(1).expand(-1, tip_l.shape[1], -1)            # (E,10,4)
            roq_e = self._ref_obj_quat[fr].unsqueeze(1).expand(-1, tip_l.shape[1], -1)
            dt = self._ref_link_contact_target_local[fr][:, self._ft_distal_idx]   # (E,10,3) distal target (obj-local)
            ref_vtx_w = math_utils.quat_apply(oq_e, dt) + obj_pos.unsqueeze(1)
            ft_in_refobj = math_utils.quat_apply(math_utils.quat_conjugate(roq_e),
                                                 self._ref_ft_pad[fr] - self._ref_obj_pos[fr].unsqueeze(1))
            ref_ft_drift = math_utils.quat_apply(oq_e, ft_in_refobj) + obj_pos.unsqueeze(1)
            # [contact-vel-gate] 관측(delta_ft_obj)과 같은 게이트 — 둘이 어긋나면 정책이 보는 목표와
            # 보상이 재는 목표가 달라진다. grasp env 도 contact_flag_gated 하나로 둘을 함께 걸었다.
            in_contact = (self._ref_link_contact_mask[fr][:, self._ft_distal_idx]
                          * self._ref_obj_vel_gate[fr].unsqueeze(-1)).unsqueeze(-1).bool()  # (E,10,1)
            ft_target = torch.where(in_contact, ref_vtx_w, ref_ft_drift)          # (E,10,3)
            ft_reward = (ft_target - tip_l).norm(dim=-1).mean(dim=-1)             # (E,) contact-conditioned
        else:
            obj_pos_err = torch.zeros(self.num_envs, device=self.device)
            obj_pos_err_w = obj_pos_err                                          # [z-weight]
            obj_rot_err = torch.zeros(self.num_envs, device=self.device)
            ft_reward = ft_err                                                     # no object → raw pad target
        # 손목 회전 오차(양손 평균). [ROLLBACK MARKER: wrist-rot] 키포인트 기하로 세운 좌표계끼리 비교한다
        # (리타게팅 npz 에 g1_palm_quat 이 없어 예전 경로는 항상 0 이었다).
        wrist_rot_err = self._wrist_rot_err(ref, kpts)
        # [ee-split] `wrist` (reward) and `wrist_pos` (termination) are the SAME tensor, kept under both
        # names so each call site reads the one that matches its intent.
        return dict(body=body_err, body_core=body_core_err,
                    ee=ee_err, hand_w=hand_err_w, obj_pos_w=obj_pos_err_w,   # [z-weight]
                    com_support=self._com_support_err(), wrist_pos=wrist_pos_err, hand=hand_err, ft=ft_err, ft_reward=ft_reward, ft_per=ft_per, tip=tip,
                    pad_inward=pad_inward, root_pos=root_pos_err, root_rot=root_rot_err,
                    obj_pos=obj_pos_err, obj_rot=obj_rot_err, root_quat=root_quat,
                    wrist_rot=wrist_rot_err)

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        e = self._errs                                    # set by _get_dones (runs first each step)
        fr = self._frame()

        # 링크별 접촉력(DexMachina): 32 wrap 링크의 압축 접촉력을 (a) 레퍼런스 접촉 mask 와 (b) 링크가 물체 표면
        # 목표점 근처일 때만 인정한다(엉뚱한 곳을 누르면 보상 없음). 활성 링크에 대해 [0,1] 로 정규화.
        link_force = self._link_contact_forces()                      # (E,L) compressive per link (on own face)
        # [ROLLBACK MARKER: contact-vel-gate] 물체가 정지한 프레임에서는 link_mask 가 전부 0 이 되어
        # 분자(lf)와 분모(n_lc)가 함께 사라진다 → force_rew = 0. grasp env 와 같은 처리다
        # (fforce_contact / n_contacts 를 동일한 게이트 플래그로 걸었다).
        link_mask = self._ref_link_contact_mask[fr] * self._ref_obj_vel_gate[fr].unsqueeze(-1)   # (E,L)
        if self._has_object and self._has_link_contact:
            oqL = self._object.data.root_quat_w[:, None, :].expand(-1, N_LINK_CONTACT, -1)   # (E,L,4) live
            tgt_w = (math_utils.quat_apply(oqL, self._ref_link_contact_target_local[fr])
                     + self._object.data.root_pos_w[:, None, :])                              # (E,L,3) world target
            lp = self.robot.data.body_pos_w[:, self._link_contact_body_ids]                   # (E,L,3) world link pos
            near = ((lp - tgt_w).norm(dim=-1) < c.contact_match_dist).float()                 # (E,L) spatial gate
        else:
            near = torch.zeros_like(link_mask)
        lf = (link_force * link_mask * near).clamp(min=0.0, max=c.contact_force_cap)  # force·mask·near
        n_lc = link_mask.sum(dim=-1).clamp(min=1.0)                   # #active links (≥1 to avoid /0)
        force_rew = lf.sum(dim=-1) / (n_lc * c.contact_force_cap)     # (E,) mean of min(force,cap)/cap ∈ [0,1]

        # [ROLLBACK MARKER: cws-contact] 로봇 쪽 접촉 목록과 점수. 위치·법선은 센서 값(contact_pos_w, force_matrix_w)을
        # 쓴다(링크 원점·고정 법선은 실제 접촉과 크게 어긋난다). 접촉 없는 쌍은 NaN 이라 마스크 후 0.
        cws_rew = torch.zeros(self.num_envs, device=self.device)
        if self._cws_sigma_h is not None and self._has_object:
            cp_w = torch.stack([s_.data.contact_pos_w.reshape(self.num_envs, -1, 3)[:, 0]
                                for s_ in self._link_contact_sensors], dim=1)      # (E,L,3)
            fm_w = torch.stack([s_.data.force_matrix_w.reshape(self.num_envs, -1, 3).sum(1)
                                for s_ in self._link_contact_sensors], dim=1)      # (E,L,3)
            _mag = fm_w.norm(dim=-1)
            _hit = (_mag > c.cws_force_thresh) & torch.isfinite(cp_w).all(dim=-1)
            cp_w = torch.nan_to_num(cp_w, nan=0.0, posinf=0.0, neginf=0.0)
            _n_w = -fm_w / _mag.clamp(min=1e-9).unsqueeze(-1)                      # 손 -> 물체 방향
            # 물체 기준 좌표로 옮긴다(현재 자세). [ROLLBACK MARKER: cws-com] 기준은 body 원점이 아니라 물리 COM
            # (논문 tracking_command.py:1745 와 같고, 레퍼런스 σ_h 도 같은 COM 프레임).
            _use_com = getattr(self, "_cws_com_p", None) is not None
            _oq = (self._object.data.root_com_quat_w if _use_com
                   else self._object.data.root_quat_w)                             # (E,4)
            _op = (self._object.data.root_com_pos_w if _use_com
                   else self._object.data.root_pos_w)                              # (E,3)
            _R = math_utils.matrix_from_quat(_oq)                                  # (E,3,3) 로컬->월드
            _p_o = torch.einsum("eji,ekj->eki", _R, cp_w - _op.unsqueeze(1))
            _n_o = torch.einsum("eji,ekj->eki", _R, _n_w)
            sig_r = CWS.support(self._cws_basis, _p_o, _n_o, _hit,
                                c.cws_mu, self._cws_len, c.cws_n_edge, c.cws_link_chunk)
            cws_rew = CWS.cws_reward(self._cws_sigma_h[fr], sig_r, c.cws_beta, c.cws_v)
            self._diag_cws = float(cws_rew.mean())
            # [ROLLBACK MARKER: cws-diag] 텐서보드에 남길 per-env 점수. 접촉 링크 수도
            # 같이 봐야 "렌치가 부족한 것"과 "애초에 안 닿은 것"을 구분할 수 있습니다.
            self._cws_per_env = cws_rew
            self._cws_nhit = _hit.float().sum(dim=-1)
            # 원 점수(cws_rew)는 v 보정에 민감해 실측상 96%가 정확히 0이었습니다. 커버리지/부족분은
            # v 와 무관하고 방향 수에도 스케일되지 않아 실패들 사이를 구분할 수 있습니다.
            self._cws_cov = CWS.cws_coverage(self._cws_sigma_h[fr], sig_r, c.cws_beta)
            self._cws_def = CWS.cws_deficit(self._cws_sigma_h[fr], sig_r, c.cws_beta)

        # FOOT contact obs/reward REMOVED (2026-07-20, GRAIL-aligned): feet + balance are owned by the
        # frozen SONIC base; the residual policy neither observes nor rewards foot contact/force/flatness.
        # Feet (ankles) are still tracked via the dedicated EE body reward (rew_ee_kpts·e["ee"]). [ee-torso]

        # 관절 물체 DOF 보상 항은 없다(물체는 항상 단일 RigidObject). 관절 물체를 실제로 스폰하면 여기 다시 추가.

        # 루트 선속도·각속도 오차 항은 뺐다. 위치·자세 추적과 중복이고 리셋 직후 각속도 잡음이 벌점을 지배했다.

        # [ROLLBACK MARKER: reg-merge] action_reg = (raw z_res, 나머지 정책 액션)의 제곱합 하나(grasp 관행).
        # 둘 다 클립·클램프 전 값이라 클립 밖에서도 복원 기울기가 남는다.
        hsl = self._group_slices["hands"]
        _nz_reg = int(getattr(self, "_act_z", int(self.cfg.sonic_action_dim)))
        if self._sonic is not None:
            _areg_src = torch.cat([self._last_z_res,
                                   self._cur_policy_action[:, _nz_reg:]], dim=-1)      # (E,100 | 117)
        else:
            _areg_src = self._cur_policy_action
        action_reg = (_areg_src ** 2).sum(-1)                                           # (E,) SUM
        # pose_reg_hands: 손 관절을 기본(중립) 자세 쪽으로 당긴다. 레퍼런스 쪽이 아닌 이유는 손 추적은 이미
        # kpt·손끝 항이 맡기 때문이다. 몸은 SONIC 이 맡으므로 정규화하지 않는다.
        _hand_ids = self._action_joint_ids_t[hsl]
        hand_ref = self.robot.data.default_joint_pos[:, _hand_ids]
        pose_reg_hands = ((self.robot.data.joint_pos[:, _hand_ids] - hand_ref) ** 2).sum(-1)
        # action_rate: raw 정책 액션의 스텝 간 변화 제곱합(GRAIL meta_action_rate_l2). 실제 관절 목표가 아니므로
        # SONIC 자체의 추종은 벌하지 않는다. raw 라 상한이 없으니 Diag / zres_clip_frac 와 함께 볼 것.
        action_rate = ((self._cur_policy_action - self._prev_policy_action) ** 2).sum(-1)
        # [ROLLBACK MARKER: energy] 역학적 파워 Σ|τ·q̇| (허리+다리). applied_torque 는 implicit
        # actuator 가 매 제어 스텝 채우는 값이라 가장 최근 decimation 스텝의 토크를 담는다.
        # 정지 유지에서 0 이므로 중력 지지분이 자동 배제된다 (τ² 와의 결정적 차이).
        _etau = self.robot.data.applied_torque[:, self._energy_joint_ids]           # (E,15) N·m
        _eqd = self.robot.data.joint_vel[:, self._energy_joint_ids]                 # (E,15) rad/s
        energy = (_etau * _eqd).abs().sum(-1)                                       # (E,) W
        self._diag_energy = energy.mean()          # 가중 이전 원값(W) — _log_reward_terms 가 읽는다
        # ── [ROLLBACK MARKER: ankle-acc] Σ_j ((q̇_prev,j − q̇_j)/dt)²  (ResMimic ankle_dof_acc) ──
        if getattr(self, "_ankle_acc_joint_ids", None) is not None:
            _aa = self.robot.data.joint_acc[:, self._ankle_acc_joint_ids]                  # (E,J) rad/s²
            ankle_acc = (_aa * _aa).sum(-1)                                                # (E,) (rad/s²)²
            self._diag_ankle_acc = ankle_acc.mean()    # 가중 이전 원값
        else:
            ankle_acc = torch.zeros(self.num_envs, device=self.device)
        # ── [/ROLLBACK MARKER: ankle-acc] ──
        # ── [ROLLBACK MARKER: waist-acc] Σ_j (data.joint_acc[j])² — 허리 관절 ──
        if getattr(self, "_waist_acc_joint_ids", None) is not None:
            _wa = self.robot.data.joint_acc[:, self._waist_acc_joint_ids]                  # (E,J) rad/s²
            waist_acc = (_wa * _wa).sum(-1)                                                # (E,) (rad/s²)²
            self._diag_waist_acc = waist_acc.mean()    # 가중 이전 원값
        else:
            waist_acc = torch.zeros(self.num_envs, device=self.device)
        # ── [/ROLLBACK MARKER: waist-acc] ──
        # ── [ROLLBACK MARKER: anti-shake] mean_i relu(‖ω_i‖ − θ)² over the configured links (SONIC anti_shake_ang_vel_l2) ──
        if getattr(self, "_anti_shake_body_ids", None) is not None:
            _wb = self.robot.data.body_ang_vel_w[:, self._anti_shake_body_ids]                 # (E,B,3) 월드 각속도
            _ex = (_wb.norm(dim=-1) - float(c.anti_shake_ang_vel_thresh)).clamp(min=0.0)     # (E,B) 데드존 초과분
            anti_shake = (_ex * _ex).mean(dim=-1)                                              # (E,)
        else:
            anti_shake = torch.zeros(self.num_envs, device=self.device)
        self._last_anti_shake = anti_shake
        self._diag_anti_shake_frac = (anti_shake > 0.0).float().mean()
        # ── [/ROLLBACK MARKER: anti-shake] ──
        # CoM-over-support balance penalty (anti-fall): out-of-support excess (m), 0 when the CoM stays
        # over the feet (so it never penalizes the balanced reference) — grows only when the robot tips.
        com_support = e["com_support"]                    # (E,) ≥ 0
        # feet-contact-match (VideoMimic/BSTRO): FRACTION of feet whose actual contact matches the reference
        # schedule (mean over feet → [0,1], so both-match = 1.0 not 2). POSITIVE bonus. c*_ref precomputed
        # (PyRoki rule); c_actual from the ground-filtered foot force sensors.
        feet_match = (self._foot_contact_actual() == self._ref_foot_contact[self._frame()]).float().mean(-1)  # (E,)∈[0,1]

        alive = (~self._died).float()                     # _died set by _get_dones this step

        # [ROLLBACK MARKER: exp-tracking] 추적 항은 항마다 w·exp(-err²/σ²) ∈ [0, w] (형태·σ 는 SONIC rewards.py:442).
        # 항별로 포화해서 나쁜 손끝이 몸 신호를 지우지 않는다. 접촉·정규화 항은 밖에서 더하고 총합은 0 에서 자른다.
        self._exp_terms = {k: self._exp_w[k] * torch.exp(-(e[self._exp_key[k]] ** 2) / self._exp_s2[k])
                           for k in self._exp_w}
        tracking_penalty = sum(self._exp_terms.values())     # POSITIVE, 이미 유계
        _alive_w = c.exp_rew_alive
        reward = (
            _alive_w * alive
            + tracking_penalty
            + (c.rew_contact_force * force_rew if c.contact_reward_mode in ('force', 'both') else 0.0)
            + (c.rew_cws * cws_rew if c.contact_reward_mode in ('cws', 'both') else 0.0)
            + c.rew_action_reg * action_reg + c.rew_pose_reg_hands * pose_reg_hands
            + c.rew_action_rate * action_rate
            + c.rew_energy * energy                       # [energy] 허리+다리 Σ|τ·q̇|
            + c.rew_anti_shake * anti_shake               # [anti-shake] 손목·머리 각속도 데드존 벌점 (SONIC)
            + c.rew_ankle_acc * ankle_acc                 # [ankle-acc] 발목 관절 가속도 벌점 (ResMimic)
            + c.rew_waist_acc * waist_acc                 # [waist-acc] 허리 관절 가속도 벌점
            + c.rew_com_support * com_support             # anti-fall (outside the tracking clamp)
            + c.rew_feet_contact_match * feet_match       # feet-contact-match bonus (positive)
        ).clamp(min=0.0)
        # 비유한 보상은 0 으로 바꾼다. NaN env 는 _get_rewards 뒤에 리셋되므로 그 스텝 보상이 GAE → PPO 를 오염시킨다.
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        self._save_state_cache(reward)                    # per-frame best-state RSI cache
        # 항별 보상 기여(가중치 적용) → Episode_Reward (항마다 그래프 하나). 추적 항은 클램프 전 값이다.
        # 클램프된 그룹 값과 clamp_frac 은 Diag /, 총합은 skrl 의 Reward / Instantaneous reward 와 같아 따로 안 남긴다.
        _track_rew = {"body_kpts": self._exp_terms["body"],       # [ROLLBACK MARKER: exp-tracking]
                      "ee_kpts": self._exp_terms["ee"],
                      "hand_kpts": self._exp_terms["hand"], "fingertip": self._exp_terms["fingertip"],
                      "root_pos": self._exp_terms["root_pos"], "root_ori": self._exp_terms["root_rot"],
                      "obj_pos": self._exp_terms["obj_pos"], "obj_rot": self._exp_terms["obj_rot"]}
        ep_rew = {
            "alive": _alive_w * alive,
            **_track_rew,
            "contact_force": c.rew_contact_force * force_rew,         # per-link (Option A) contact-force reward
            # [ROLLBACK MARKER: cws-diag] Episode_Reward 그룹은 "실제로 보상에 들어간 값"만 담아야
            # 합니다. 진단 전용일 때 여기에 값을 흘리면 보상에 포함된 것처럼 보입니다. 원 점수는
            # 아래 Diag / cws_score 로 나갑니다.
            "contact_cws": (c.rew_cws * cws_rew if c.contact_reward_mode in ('cws', 'both')
                            else torch.zeros_like(cws_rew)),
            "action_reg": c.rew_action_reg * action_reg,        # [reg-merge] 잠재+손 SUM
            "pose_reg_hands": c.rew_pose_reg_hands * pose_reg_hands,
            "action_rate": c.rew_action_rate * action_rate,
            "energy": c.rew_energy * energy,                   # [energy]
            "anti_shake": c.rew_anti_shake * anti_shake,     # [anti-shake]
            "ankle_acc": c.rew_ankle_acc * ankle_acc,        # [ankle-acc]
            "waist_acc": c.rew_waist_acc * waist_acc,        # [waist-acc]
            "com_support": c.rew_com_support * com_support,
            "feet_contact_match": c.rew_feet_contact_match * feet_match,
        }
        self._log_reward_terms(e, tracking_penalty, ep_rew, fr)
        return reward

    def _log_reward_terms(self, e, tracking_penalty, ep_rew, fr):
        log = self.extras.setdefault("log", {})
        log.update({
            "Error / body_kpts": e["body"].mean(),
            "Error / wrist_kpts": e["wrist_pos"].mean(),   # 종료 게이트 대상 (보상에서는 ee 에 흡수)
            "Error / ee_kpts": e["ee"].mean(),
            "Error / com_support": e["com_support"].mean(),
            "Error / hand_kpts": e["hand"].mean(),
            "Error / fingertip": e["ft"].mean(), "Error / root_pos": e["root_pos"].mean(),
            "Error / root_rot": e["root_rot"].mean(), "Error / obj_pos": e["obj_pos"].mean(),
            "Error / wrist_rot": e["wrist_rot"].mean(),
            "Curriculum / reached_frame": float(self._reached_frame),
            "Curriculum / late_gate": float(self._late_gate),
            "Curriculum / friction_max": float(self._last_friction_max),
            "Curriculum / friction_mean": float(self._last_friction_mean),
            "Curriculum / ref_start_frac": float(getattr(self, "_diag_ref_start", 0.0)),
            "Curriculum / cache_coverage": float((~self._init_flg).sum().item()) / self._ref_len,
        })
        # 항별 보상 → Episode_Reward (항마다 그래프 하나. 합계·진단은 넣지 않는다).
        for k, v in ep_rew.items():
            log[f"Episode_Reward / {k}"] = v.mean()
        # reward-shaping diagnostics → separate "Diag /" tab (cfg.log_reward_diag=False drops them).
        # The 9 tracking terms above are logged PRE-clamp, so once the clamp bites their sum no longer
        # equals what entered the reward — these three are the only window on that gap.
        if self.cfg.log_reward_diag:
            log["Diag / tracking_penalty"] = tracking_penalty.mean()
            # [ROLLBACK MARKER: cws-diag] 접촉 렌치 진단. cws_score 는 "지금 파지가 만들 수 있는
            # 렌치가 레퍼런스가 요구하는 렌치를 얼마나 담는가" (1에 가까울수록 충분). cws_nhit 은
            # 실제로 힘을 받고 있는 링크 수 — 점수가 낮을 때 "렌치 부족"인지 "안 닿음"인지 가릅니다.
            if getattr(self, "_cws_per_env", None) is not None:
                log["Diag / cws_score"] = self._cws_per_env.mean()     # 원 점수 (v 보정 확인용)
                log["Diag / cws_coverage"] = self._cws_cov.mean()      # 요구 방향 중 충족 비율 ∈[0,1]
                log["Diag / cws_deficit"] = self._cws_def.mean()       # 방향당 평균 부족분
                log["Diag / cws_nhit"] = self._cws_nhit.mean()
            # [ROLLBACK MARKER: exp-tracking] 포화도: term/w ∈ [0,1]. 1 에 가까우면 σ 가 느슨해 항이
            # 상수라 아무것도 안 가르치고, 0 에 가까우면 너무 조여 항이 죽는다. 0.3~0.7 이 실제로
            # 학습을 이끄는 대역이고, σ 는 이 로그로 튜닝한다.
            for _k, _v in self._exp_terms.items():
                log[f"Sat / {_k}"] = _v.mean() / max(self._exp_w[_k], 1e-9)
        # RSI / 에피소드 길이 진단: 길이가 가변이라 skrl 의 Total reward(에피소드 합)는 설정 간 비교가 안 된다.
        # episode_len = 현재 에피소드 경과 스텝, rsi_start = 시작 프레임.
        if hasattr(self, "_diag_energy"):                # [energy] 가중 이전 원값 (W)
            log["Diag / energy"] = self._diag_energy
            log["Diag / anti_shake_frac"] = self._diag_anti_shake_frac     # [anti-shake] 데드존 초과 env 비율
            if getattr(self, "_ankle_acc_joint_ids", None) is not None:
                log["Diag / ankle_acc"] = self._diag_ankle_acc             # [ankle-acc] 가중 이전 Σ(rad/s²)²
            if getattr(self, "_waist_acc_joint_ids", None) is not None:
                log["Diag / waist_acc"] = self._diag_waist_acc             # [waist-acc] 가중 이전 Σ(rad/s²)²
        if getattr(self, "_upper_on", False):                     # [joint-residual]
            log["Diag / upper_res_rad"] = self._diag_upper_res_rad          # |상체 잔차| 평균, rad
            log["Diag / upper_clamp_frac"] = self._diag_upper_clamp_frac    # 관절 한계에 잘린 비율
        log["Diag / blowup_frac"] = self._diag_blowup_frac        # [nan-guard] 관절 속도 폭주로 리셋된 env 비율
        log["Diag / obs_sanitized"] = self._diag_obs_sanitized    # [nan-guard] 위생 처리된 관측 원소 비율 (0 이어야 정상)
        log["Diag / episode_len_mean"] = self.episode_length_buf.float().mean()
        log["Diag / rsi_start_mean"] = self._episode_start_frame.float().mean()
        if self.cfg.contact_vel_gate:                                  # [contact-vel-gate]
            log["Diag / vel_gate_frac"] = self._ref_obj_vel_gate[self._frame_idx].mean()
        log["Diag / death_frac"] = self._died.float().mean()
        # per-block action saturation (see _sonic_pre_physics_step)
        if hasattr(self, "_diag_hand_clamp_frac"):
            log["Diag / hand_clamp_frac"] = self._diag_hand_clamp_frac
            log["Diag / zres_clip_frac"] = self._diag_zres_clip_frac
        # RAW action magnitude — action_rate no longer sees it, so this is now the ONLY window on a
        # mean random-walk (the failure mode that killed the 2026-07-28 09:32 run: |z_res| reached the
        # ±20 clip on 25% of samples while the bounded terms stayed flat).
        log["Diag / zres_absmax"] = self._cur_policy_action[:, :self.cfg.sonic_action_dim].abs().max()
        log["Diag / hand_absmax"] = self._cur_policy_action[:, self.cfg.sonic_action_dim:].abs().max()
        # per-frame-bucketed tracking error — disambiguates reward-balance from curriculum-mix:
        # if per-bucket error stays flat/falls while the GLOBAL mean rises, the rise is a
        # frame-distribution shift (harder later frames entered the mix), not per-frame regression.
        T = max(1, self._ref_len - 1)
        q = (fr.float() / T * 4.0).clamp(0, 3).long()               # frame quartile per env
        for b in range(4):
            m = q == b
            if bool(m.any()):
                log[f"Error / body_q{b}"] = e["body"][m].mean()
                log[f"Error / root_pos_q{b}"] = e["root_pos"][m].mean()
        # DIAG(term-cause): per-gate share of THIS step's deaths (gates OR → shares can overlap) + obj_rot err.
        log["Error / obj_rot"] = e["obj_rot"].mean()
        dead = self._died
        nd = dead.float().sum()
        if nd > 0:
            cc = self.cfg
            _cause = {"body": e["body"] > cc.term_body_kpt_err,
                      "wrist_pos": e["wrist_pos"] > cc.term_wrist_pos_err, "ft": e["ft"] > cc.term_ft_err,
                      "obj_pos": e["obj_pos"] > cc.term_obj_pos_err, "obj_rot": e["obj_rot"] > cc.term_obj_rot_err,
                      # [wrist-rot-term] _has_palm_ref 조건 제거 (위 종료 게이트 주석 참조)
                      "wrist_rot": e["wrist_rot"] > cc.term_wrist_rot_err}
            for k, mk in _cause.items():
                log[f"Term / {k}"] = (mk & dead).float().sum() / nd

    # ------------------------------------------------------------------ dones
    def _dones_deviation(self, e) -> torch.Tensor:
        """DEVIATION-FROM-REFERENCE termination (frame-0 pink-IK reset matches the reference → no
        grace). NO separate root pos/tilt/height gates: the mean BODY-keypoint error already
        subsumes them — a root translation shifts every keypoint, a tilt rotates the far keypoints
        (torso/arms) away, and a fall drives the foot/pelvis keypoints off. So body_kpt is the
        single body/locomotion gate; the original grasp gates (ft / wrist-pos / wrist-rot / object) add
        their own."""
        c = self.cfg
        d = e["body"] > c.term_body_kpt_err                          # added full-body/locomotion gate
        # 손끝(10 pad 평균)과 손목 위치(양손 평균) 이탈 종료. 손가락 체인 kpt 평균은 보상 전용이다(grasp 와 같음).
        d = d | (e["ft"] > c.term_ft_err) | (e["wrist_pos"] > c.term_wrist_pos_err)
        # 손목 회전 이탈 종료. [wrist-rot-term] 오차 계산에 안 쓰는 _has_palm_ref 조건은 뺐다(2026-09-07).
        # 리타게팅 로봇 자신도 16클립 프레임의 0.2% 가 임계(0.75, 사용자 결정)를 넘어 그 프레임은 허위 종료된다.
        d = d | (e["wrist_rot"] > c.term_wrist_rot_err)
        #  object (mirrors grasp obj_pos + obj_rot) — only when an active object is present.
        if self._has_object:
            d = d | (e["obj_pos"] > c.term_obj_pos_err) | (e["obj_rot"] > c.term_obj_rot_err)
        if not c.termination:
            d = torch.zeros_like(d)
        # 비유한 상태 차단(종료를 꺼도 항상): NaN 은 `err > thresh` 게이트에 안 걸리므로 루트·관절 상태가 비유한인
        # env 를 강제 리셋해 공유 스케일러와 PPO 로 번지지 않게 한다(inf 는 위 게이트가 잡는다).
        rd = self.robot.data
        nonfinite = (
            ~torch.isfinite(rd.root_pos_w).all(-1)
            | ~torch.isfinite(rd.root_quat_w).all(-1)
            | ~torch.isfinite(rd.joint_pos).all(-1)
            | ~torch.isfinite(rd.joint_vel).all(-1)
        )
        # [ROLLBACK MARKER: nan-guard] (L2) 물체 상태 유한성: 위 게이트는 로봇만 보고, 물체 NaN 은 어떤 `>` 비교에도
        # 안 걸려 관측으로 흘러 스케일러를 영구히 망가뜨린다.
        if self._has_object and bool(getattr(c, "nan_guard_object", True)):
            _od = self._object.data
            nonfinite = nonfinite | (
                ~torch.isfinite(_od.root_pos_w).all(-1)
                | ~torch.isfinite(_od.root_quat_w).all(-1)
                | ~torch.isfinite(_od.root_lin_vel_w).all(-1)
                | ~torch.isfinite(_od.root_ang_vel_w).all(-1)
            )
        # (L1) 관절 속도 폭주 게이트. 손가락 J0 는 액추에이터 없이 소프트 텐던에만 묶여 수천 rad/s 까지 튈 수 있고,
        # 이 값이 관측을 통해 스케일러 분산을 부풀린다(정상 드라이브 한계는 15~37 rad/s).
        _vth = float(getattr(c, "nan_guard_joint_vel", 0.0))
        if _vth > 0.0:
            _blow = rd.joint_vel.abs().max(-1).values > _vth
            self._diag_blowup_frac = _blow.float().mean()
            nonfinite = nonfinite | _blow
        # ── [/ROLLBACK MARKER: nan-guard] ──
        return d | nonfinite

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # runs BEFORE _get_rewards each step → compute + cache errors here for both.
        self._errs = self._compute_errors()
        self._died = self._dones_deviation(self._errs)
        # 시간 초과(truncated, 실패 아님): 실패 EMA 에서 빠지고 skrl 이 bootstrap 한다.
        # 첫 항 = 레퍼런스 끝(env 별 프레임), 둘째 항 = 안전 상한. bootstrap 값은 자동 리셋 뒤 새 시작 상태의 V 다.
        time_out = (self._frame_idx >= self._ref_len - 1) | (self.episode_length_buf >= self.max_episode_length - 1)
        return self._died, time_out

    # ── 물체 마찰 커리큘럼 [ROLLBACK MARKER: friction-curriculum] ──────────────────────
    def _apply_object_friction(self, env_ids) -> None:
        """리셋되는 환경들의 물체 마찰을 새로 뽑아 적용합니다.

        [friction_min, friction_max(t)]에서 균등 추출하고, friction_max(t)는
        friction_max_init에서 friction_min까지 friction_decay_steps 제어 스텝에 걸쳐 선형으로
        내려옵니다. 정적 마찰과 동적 마찰을 같은 값으로 둡니다.

        try/except로 감싼 것은 DirectRLEnv.__init__이 물리 뷰가 준비되기 전에 모든 환경을 한 번
        리셋하기 때문입니다(첫 호출만 건너뜁니다).
        """
        if not self.cfg.friction_curriculum or self._object is None:
            return
        fmin = float(self.cfg.friction_min)
        decay = max(1, int(self.cfg.friction_decay_steps))
        frac = min(self._friction_step_count / decay, 1.0)
        fmax = self.cfg.friction_max_init + (fmin - self.cfg.friction_max_init) * frac
        self._last_friction_max = float(fmax)
        try:
            if isinstance(env_ids, torch.Tensor):
                eids = env_ids.detach().to(dtype=torch.long, device="cpu")
            else:
                eids = torch.as_tensor(list(env_ids), dtype=torch.long, device="cpu")
            fr = fmin + (fmax - fmin) * torch.rand(eids.numel())
            mat = self._object.root_physx_view.get_material_properties()   # (E,shapes,3) cpu
            mat[eids, :, 0] = fr.unsqueeze(-1)                             # 정적 마찰
            mat[eids, :, 1] = fr.unsqueeze(-1)                             # 동적 마찰
            self._object.root_physx_view.set_material_properties(mat, eids)
            self._last_friction_mean = float(fr.mean().item())
        except Exception as e:
            print(f"[friction-curriculum] 적용 건너뜀 (뷰 미준비?): {e}")
    # ── END 물체 마찰 커리큘럼 ─────────────────────────────────────────────────────────


    # ── [ROLLBACK MARKER: link-kpt-objframe] ──────────────────────────────────────────
    # ── [ROLLBACK MARKER: body-kpt-fk] 몸통 키포인트 목표를 리타게팅 FK 로 (2026-09-06) ──────
    def _apply_body_kpt_fk(self) -> None:
        """cfg.body_kpt_from_retarget_fk 면 _ref_kpts 의 몸통 13개를 리타게팅 로봇 FK 로 교체합니다.

        기본 목표(SMPL-X 사람 키포인트)는 G1 이 도달할 수 없습니다 — 리타게팅 측정에서 사람 발목이
        로봇 발목보다 5.4~5.8 cm 위, 어깨 잔차 22 cm, 골반 잔차 9.4 cm 입니다. 2026-09-05 run 에서
        Error/body_kpts 가 0.108 m 에 고정되고 Episode_Reward/body_kpts 가 0.0 이었습니다.

        여기서는 로봇을 프레임마다 리타게팅 자세(g1_joint_pos + g1_root_pose)로 세워 body_pos_w 를
        읽습니다. 제거된 _solve_ref_link_local() 과 같은 방식이고, 물리를 진행시킬 필요가 없어
        청크당 한 스텝이면 됩니다(환경 수가 프레임 수 이상이면 한 번에 끝납니다).

        손 키포인트 42개와 손목은 건드리지 않습니다 — 그쪽은 사람 손을 목표로 두는 것이 의도입니다.
        키포인트 오프셋(_kpt_offsets)은 몸통에서 전부 0 이므로
        링크 원점이 곧 키포인트입니다. 그래도 일반성을 위해 오프셋을 적용해 둡니다.
        """
        if not getattr(self.cfg, "body_kpt_from_retarget_fk", False):
            return
        if self._ref_joints is None:
            print("[body-kpt-fk] g1_joint_pos 가 없어 건너뜁니다 (SMPL-X 목표 유지).")
            return
        F, E, dev = self._ref_len, self.num_envs, self.device
        nb = N_BODY_KPTS
        org = self.scene.env_origins
        aid = self._action_joint_ids_t
        body_ids = self._kpt_body_ids[:nb]
        body_off = self._kpt_offsets[:nb]
        keep_q = self.robot.data.joint_pos.clone()
        keep_r = self.robot.data.root_state_w[:, :7].clone()
        out = torch.zeros(F, nb, 3, device=dev)
        for base in range(0, F, E):
            fr = (base + torch.arange(E, device=dev)).clamp(max=F - 1)
            rp = torch.zeros(E, 7, device=dev)
            rp[:, :3] = self._ref_root_pos[fr] + org
            rp[:, 3:7] = self._ref_root_quat[fr]
            jp = self.robot.data.default_joint_pos.clone()
            jp[:, aid] = self._ref_joints[fr]
            if self._ref_j0 is not None and self._ref_j0_ids is not None:
                jp[:, self._ref_j0_ids] = self._ref_j0[fr]
            self.robot.write_root_pose_to_sim(rp)
            # [body-kpt-fk] 루트 속도를 반드시 0 으로 쓴다. write_root_pose_to_sim 은 속도를 남겨서 청크를 도는 동안
            # 중력·접촉으로 누적된 속도만큼 움직인 뒤 위치를 읽게 된다(골반 27.7 cm 오차).
            self.robot.write_root_velocity_to_sim(torch.zeros(E, 6, device=dev))
            self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            p = self.robot.data.body_pos_w[:, body_ids] - org.unsqueeze(1)      # (E,nb,3) env-local
            q = self.robot.data.body_quat_w[:, body_ids]
            kp = p + math_utils.quat_apply(q, body_off.unsqueeze(0).expand(E, -1, -1))
            n = min(E, F - base)
            out[base:base + n] = kp[:n]
        # 로봇을 원래대로 (이 계산이 상태를 남기면 안 됩니다)
        self.robot.write_root_pose_to_sim(keep_r)
        self.robot.write_root_velocity_to_sim(torch.zeros(E, 6, device=dev))
        self.robot.write_joint_state_to_sim(keep_q, torch.zeros_like(keep_q))
        self.scene.write_data_to_sim()
        _d = (out - self._ref_kpts[:, :nb]).norm(dim=-1)                 # (F,nb)
        _names = list(BODY_KPTS.values())
        print(f"[body-kpt-fk] 몸통 {nb}개 목표를 리타게팅 FK 로 교체: {F} 프레임. "
              f"SMPL-X 대비 이동 거리 중앙 {_d.median().item()*100:.1f} cm, "
              f"최대 {_d.max().item()*100:.1f} cm")
        _md = _d.median(dim=0).values
        for _i in torch.argsort(_md, descending=True).tolist():
            print(f"    {_names[_i]:32s} 중앙 {_md[_i].item()*100:6.1f} cm  "
                  f"최대 {_d[:, _i].max().item()*100:6.1f} cm")
        self._ref_kpts[:, :nb] = out

    # ── [ROLLBACK MARKER: stage1-hand] 1단계 롤아웃 → 손 보상 목표 교체 + 잔차 손 액션 기준 (2026-09-09) ──
    def _setup_stage1_hand(self) -> None:
        """_load_stage1_hand 의 numpy 를 써서 (1) hand_kpt_from_hand_pretrain 이면 손·손목 보상 목표를 롤아웃 FK 로
        교체하고 (2) sonic_hand_residual 이면 잔차 기준 _hand_res_base (F,36) 와 EMA 상태를 만듭니다."""
        c, dev = self.cfg, self.device
        S = getattr(self, "_np_s1", None)
        hsl = self._group_slices["hands"]
        if S is not None and bool(getattr(c, "hand_kpt_from_hand_pretrain", False)):
            self._apply_stage1_hand_kpts(S)
        if not bool(getattr(c, "sonic_hand_residual", False)):
            return
        src = str(getattr(c, "sonic_hand_residual_base", "reference"))
        if src.startswith("hand_pretrain") and S is not None:
            key = "finger_qpos" if src == "hand_pretrain_qpos" else "finger_target"
            act_hand_names = [self.robot.joint_names[i] for i in self._action_joint_ids_t[hsl].tolist()]
            perm = [S["base_names"].index(n) for n in act_hand_names]     # 이름으로 재배열 (없으면 ValueError)
            base = torch.from_numpy(np.ascontiguousarray(S[key][:, perm])).to(dev)          # (F,36)
            what = f"1단계 롤아웃 {key}"
        else:
            if src != "reference":
                print(f"[stage1-hand] 잔차 기준 '{src}' 을 쓸 1단계 파일이 없어 레퍼런스(_ref_joints)로 대체합니다.")
            if self._ref_joints is None:
                raise RuntimeError("[stage1-hand] sonic_hand_residual 에는 기준이 필요합니다 — 리타게팅(_ref_joints)도 없습니다.")
            base = self._ref_joints[:, hsl].clone()
            what = "리타게팅 레퍼런스 _ref_joints"
        # 기준을 한계 안으로 — 마진(상한-기준, 기준-하한)이 음수가 되지 않게
        self._hand_res_base = torch.clamp(base, self._ctrl_lower[hsl], self._ctrl_upper[hsl])
        self._hand_res_ema = torch.zeros(self.num_envs, int(hsl.stop - hsl.start), device=dev)
        print(f"[stage1-hand] 잔차 손 액션 ON: 기준 = {what} (F={base.shape[0]}, 한계 클램프 최대 "
              f"{(base - self._hand_res_base).abs().max().item():.3f} rad), EMA α = {self._finger_alpha:.2f} "
              f"(정규화 잔차 전용), 마진 매핑")

    def _apply_stage1_hand_kpts(self, S: dict) -> None:
        """_ref_kpts 손 블록 42 + 손목 2, _ref_ft_pad 10, _ref_palm_quat 을 1단계 롤아웃 FK 로 교체합니다.
        로봇을 프레임마다 리타게팅 자세로 세우되 손 관절만 롤아웃 값으로 바꾸고(_apply_body_kpt_fk 와 같은
        방식, 물리 진행 없음), 각 링크를 손바닥(robot0_{l,r}_palm) 기준으로 읽어 롤아웃 손바닥 자세에 다시
        붙입니다 — 팔 자세가 무엇이든 결과는 같습니다."""
        F, E, dev, c = self._ref_len, self.num_envs, self.device, self.cfg
        nb, nh = N_BODY_KPTS, N_HAND_KPTS_PER_HAND
        org = self.scene.env_origins
        aid = self._action_joint_ids_t
        jn = self.robot.joint_names
        # 롤아웃 손 관절 → 이 로봇 관절 인덱스 (이름). 롤아웃 값이 한계를 살짝 넘는 경우가 있어 클램프.
        s1_jid = torch.tensor([jn.index(n) for n in S["hand_q_names"]], device=dev, dtype=torch.long)
        lim = self.robot.data.soft_joint_pos_limits[0, s1_jid]                                # (44,2)
        s1_q = torch.clamp(torch.from_numpy(S["hand_q"]).to(dev), lim[:, 0], lim[:, 1])        # (F,44)
        pp_s1 = torch.from_numpy(S["palm_pos"]).to(dev)                                       # (F,2,3) 보정 후
        pq_s1 = _canon(torch.from_numpy(S["palm_quat"]).to(dev))                              # (F,2,4)
        pp_raw = torch.from_numpy(S["palm_pos_raw"]).to(dev)                                  # 보정 전 (정합 검사)
        pq_raw = _canon(torch.from_numpy(S["palm_quat_raw"]).to(dev))
        hand_ids, hand_off = self._kpt_body_ids[nb:], self._kpt_offsets[nb:]                  # (42,), (42,3)
        side_h = torch.tensor([0] * nh + [1] * nh, device=dev, dtype=torch.long)              # 손 블록 = 왼손 21, 오른손 21
        side_ft = torch.tensor([0 if "_l_" in n else 1 for n in c.fingertip_body_names], device=dev, dtype=torch.long)
        wr_names = [list(BODY_KPTS.values())[i] for i in self._wrist_kpt_idx.tolist()]
        wrist_ids = self._kpt_body_ids[self._wrist_kpt_idx]
        side_w = torch.tensor([0 if "left" in n else 1 for n in wr_names], device=dev, dtype=torch.long)
        zero_off = torch.zeros(len(wr_names), 3, device=dev)
        keep_q = self.robot.data.joint_pos.clone()
        keep_r = self.robot.data.root_state_w[:, :7].clone()
        out_h = torch.zeros(F, hand_ids.numel(), 3, device=dev)
        out_ft = torch.zeros(F, self._ft_body_ids.numel(), 3, device=dev)
        out_w = torch.zeros(F, wrist_ids.numel(), 3, device=dev)
        chk_ft = torch.zeros_like(out_ft)
        for base in range(0, F, E):
            fr = (base + torch.arange(E, device=dev)).clamp(max=F - 1)
            rp = torch.zeros(E, 7, device=dev)
            rp[:, :3] = self._ref_root_pos[fr] + org
            rp[:, 3:7] = self._ref_root_quat[fr]
            jp = self.robot.data.default_joint_pos.clone()
            if self._ref_joints is not None:
                jp[:, aid] = self._ref_joints[fr]
            jp[:, s1_jid] = s1_q[fr]
            self.robot.write_root_pose_to_sim(rp)
            self.robot.write_root_velocity_to_sim(torch.zeros(E, 6, device=dev))   # 속도 0 — _apply_body_kpt_fk 주석
            self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            bp, bq = self.robot.data.body_pos_w, self.robot.data.body_quat_w
            pp, pq = bp[:, self._palm_body_ids], bq[:, self._palm_body_ids]                  # (E,2,3/4) 이 FK 의 손바닥

            def _anchor(ids, off, side, a_p, a_q):
                # 링크 → 손바닥 기준 상대 자세 → 롤아웃 손바닥 (a_p, a_q) 에 다시 붙임 → 키포인트 오프셋
                p, q = bp[:, ids], bq[:, ids]                                                 # (E,K,3/4) world
                pp_k, pq_k = pp[:, side], pq[:, side]
                rel_p = math_utils.quat_apply(math_utils.quat_conjugate(pq_k), p - pp_k)
                rel_q = math_utils.quat_mul(math_utils.quat_conjugate(pq_k), q)
                ap, aq = a_p[fr][:, side], a_q[fr][:, side]                                   # (E,K,3/4) env-local
                pw = ap + math_utils.quat_apply(aq, rel_p)
                return pw + math_utils.quat_apply(math_utils.quat_mul(aq, rel_q),
                                                  off.unsqueeze(0).expand(E, -1, -1))

            n = min(E, F - base)
            out_h[base:base + n] = _anchor(hand_ids, hand_off, side_h, pp_s1, pq_s1)[:n]
            out_ft[base:base + n] = _anchor(self._ft_body_ids, self._ft_offsets, side_ft, pp_s1, pq_s1)[:n]
            out_w[base:base + n] = _anchor(wrist_ids, zero_off, side_w, pp_s1, pq_s1)[:n]
            chk_ft[base:base + n] = _anchor(self._ft_body_ids, self._ft_offsets, side_ft, pp_raw, pq_raw)[:n]
        # 로봇을 원래대로
        self.robot.write_root_pose_to_sim(keep_r)
        self.robot.write_root_velocity_to_sim(torch.zeros(E, 6, device=dev))
        self.robot.write_joint_state_to_sim(keep_q, torch.zeros_like(keep_q))
        self.scene.write_data_to_sim()
        # 정합 검사: 보정 전 손바닥에 붙인 FK 손끝 vs 롤아웃이 기록한 손끝 (같은 USD 손이면 ~0 mm)
        valid = torch.from_numpy(S["valid"]).to(dev) > 0.5
        perm_ft = [S["ft_names"].index(n) for n in c.fingertip_body_names]
        rec_ft = torch.from_numpy(np.ascontiguousarray(S["ft_pos"][:, perm_ft])).to(dev)
        d_chk = ((chk_ft - rec_ft).norm(dim=-1)[valid] * 1000.0).flatten()
        d_h = (out_h - self._ref_kpts[:, nb:]).norm(dim=-1)[valid] * 100.0
        d_w = (out_w - self._ref_kpts[:, self._wrist_kpt_idx]).norm(dim=-1)[valid] * 100.0
        d_ft = (out_ft - self._ref_ft_pad).norm(dim=-1)[valid] * 100.0
        d_corr = (pp_s1 - pp_raw).norm(dim=-1)[valid] * 100.0
        print(f"[stage1-hand] 손 보상 목표 교체: 손 키포인트 {hand_ids.numel()} + 손목 {wrist_ids.numel()} + 손끝 "
              f"{out_ft.shape[1]} ({F} 프레임, 유효 {int(valid.sum())})\n"
              f"    SMPL-X 대비 이동: 손 kpt 중앙 L {d_h[:, :nh].median().item():.1f} / R {d_h[:, nh:].median().item():.1f} cm "
              f"(최대 {d_h.max().item():.1f}), 손목 중앙 {d_w.median().item():.1f} cm (최대 {d_w.max().item():.1f}), "
              f"손끝 중앙 {d_ft.median().item():.1f} cm (최대 {d_ft.max().item():.1f})\n"
              f"    물체 보정으로 옮긴 손바닥: 중앙 {d_corr.median().item():.2f} / 최대 {d_corr.max().item():.2f} cm\n"
              f"    USD 정합 (FK 손끝 vs 롤아웃 기록 손끝): 중앙 {d_chk.median().item():.2f} / p95 "
              f"{d_chk.kthvalue(max(1, int(0.95 * d_chk.numel()))).values.item():.2f} / 최대 {d_chk.max().item():.2f} mm")
        self._ref_kpts[:, nb:] = out_h
        self._ref_kpts[:, self._wrist_kpt_idx] = out_w
        self._ref_ft_pad = out_ft
        # ── [ROLLBACK MARKER: stage1-vertex] 접촉 프레임 손끝 목표(사람 접촉점, 물체 좌표) → 1단계 pad (1단계 물체 좌표).
        #    rew_fingertip 과 손끝 관측이 접촉/비접촉 모두 "1단계 손 대비 물체" 하나를 보게 된다 — rew_hand_kpts 와
        #    같은 손을 가리킨다. 물체 레퍼런스(_ref_obj_pos/quat)는 그대로. 손끝 외 링크 행은 건드리지 않는다.
        if S.get("ft_obj_local") is not None and self._has_link_contact:
            new_loc = torch.from_numpy(np.ascontiguousarray(S["ft_obj_local"][:, perm_ft])).to(dev)   # (F,10,3)
            old_loc = self._ref_link_contact_target_local[:, self._ft_distal_idx]
            m_c = (self._ref_link_contact_mask[:, self._ft_distal_idx] > 0.5) & valid.unsqueeze(1)    # (F,10) 사람 접촉 프레임
            d_v = (new_loc - old_loc).norm(dim=-1)[m_c] * 100.0
            self._ref_link_contact_target_local[:, self._ft_distal_idx] = new_loc
            if d_v.numel() > 0:
                print(f"[stage1-hand] 접촉 손끝 목표 교체: 사람 접촉점 → 1단계 pad (물체 좌표). 사람 접촉 프레임 {int(m_c.sum())}개에서 "
                      f"중앙 {d_v.median().item():.1f} / p90 "
                      f"{d_v.kthvalue(max(1, int(0.9 * d_v.numel()))).values.item():.1f} / 최대 {d_v.max().item():.1f} cm 이동")
            else:
                print("[stage1-hand] 접촉 손끝 목표 교체: 사람 접촉점 → 1단계 pad (물체 좌표). (사람 접촉 프레임 없음)")
        # ── [/ROLLBACK MARKER: stage1-vertex] ──
        self._ref_palm_quat = pq_s1.clone()
        self._has_palm_ref = True
    # ── [/ROLLBACK MARKER: stage1-hand] ──

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids) -> None:
        # [ROLLBACK MARKER: deferred-cache] super() 전에 실행해야 한다(super() 가 episode_length_buf 를 0 으로 만든다).
        # getattr 인 이유: DirectRLEnv.__init__ 의 첫 리셋은 스테이징 버퍼 할당 전이다.
        _ep_len = self.episode_length_buf[env_ids].clone()
        if getattr(self, "_pend_state", None) is not None:
            self._flush_state_cache(env_ids, _ep_len)
        # [ROLLBACK MARKER: late-gate] `_frame_idx` still holds the frame the ending episode stopped
        # at — it is overwritten with the new start further down. Latched: the quality bars must not
        # oscillate, or the cache ends up holding states admitted under two different standards.
        if hasattr(self, "_late_gate_frames") and not self._late_gate:
            _done_end = self._frame_idx[env_ids] >= self._ref_len - 3
            if bool(((_ep_len >= self._late_gate_frames) & _done_end).any()):
                self._late_gate = True
                print(f"[late-gate] ON at sampling step {self._sampling_step_count}: an episode ran "
                      f"{int(_ep_len.max())} >= {self._late_gate_frames} steps and reached the clip end.")
        super()._reset_idx(env_ids)
        c = self.cfg
        n = len(env_ids)
        dev = self.device

        # ---- 실패 가중 샘플링: 종료된 env 의 마지막 정상 프레임(_enough_idx)에 실패 EMA 를 올린다(시간 초과 제외) ----
        if c.adaptive_sampling and c.failure_weighted_sampling and hasattr(self, "_died"):
            term = (self.reset_terminated[env_ids] if hasattr(self, "reset_terminated")
                    else self._died[env_ids])
            if bool(term.any()):
                fail_frames = self._enough_idx[env_ids][term].clamp(0, self._ref_len - 1)
                counts = torch.bincount(fail_frames, minlength=self._ref_len).float()
                self._failure_count = c.adaptive_alpha * counts + (1.0 - c.adaptive_alpha) * self._failure_count

        # ---- adaptive frame sampling ----
        self._sampling_step_count += 1
        self._friction_step_count += 1   # [ROLLBACK MARKER: friction-curriculum] 감쇠 진행
        have_train = ~self._init_flg
        # RSI 후보 프레임: 리타게팅 레퍼런스(_ref_joints)가 있으면 모든 프레임이 처음부터 시작점이다(캐시는 더 나은
        # 복원 상태를 줄 뿐). 없으면 캐시가 덮은 프레임만(frame 0 은 항상 포함).
        if self._ref_joints is not None:
            candidates = torch.ones(self._ref_len, dtype=torch.bool, device=dev)
        else:
            candidates = have_train.clone()
            if candidates.sum() == 0:
                candidates[0] = True                                 # frame 0 always available
        cand_idx = torch.nonzero(candidates, as_tuple=False).squeeze(-1)

        # pretrain(failure_weighted_sampling=False) → 캐시 프레임에서 균등. train → 처음 uniform_sampling_steps
        # 제어 스텝은 균등, 그 뒤 실패 가중. _sampling_step_count 는 제어 스텝 수라 그대로 비교한다.
        use_uniform = (
            (not c.adaptive_sampling)
            or (not c.failure_weighted_sampling)
            or self._sampling_step_count < c.uniform_sampling_steps
        )
        if use_uniform:
            pick = cand_idx[torch.randint(0, len(cand_idx), (n,), device=dev)]
        else:
            w = self._failure_count[cand_idx]
            probs = w / (w.sum() + 1e-6)
            ur = c.adaptive_uniform_ratio
            probs = (probs + ur / len(cand_idx)) / (1.0 + ur)
            _sel = torch.multinomial(probs, n, replacement=True)
            pick = cand_idx[_sel]
        # run-up 만큼 되감고 시작을 [0, ref_len-1-back] 로 자른다(목표 pick 은 전 구간, 에피소드는 최소 back 스텝).
        # adaptive_sampling=False(rollout/play)면 upper=0 → 항상 frame 0 에서 시작해 클립 전체를 돈다.
        upper = max(0, self._ref_len - 1 - self._adaptive_back_frames) if c.adaptive_sampling else 0
        # [ROLLBACK MARKER: rand-runup] 되감기 길이를 무작위로 뽑는다. 고정 되감기는 시작 분포를 실패 분포의
        # 평행이동으로 만들어 시작-실패 간격이 고착된다. 하한은 너무 빨리 죽어 캐시에 기여 못 하는 것을 막는다.
        if self._adaptive_back_min_frames > 0 and c.adaptive_sampling:
            _lo = min(self._adaptive_back_min_frames, self._adaptive_back_frames)
            _back = torch.randint(_lo, self._adaptive_back_frames + 1, (n,), device=dev)
        else:
            _back = torch.full((n,), self._adaptive_back_frames, device=dev, dtype=torch.long)
        # 상한도 되감기에 맞춰 프레임별로: 되감기가 짧으면 더 뒤에서 시작할 수 있습니다.
        _upper = (self._ref_len - 1 - _back).clamp(min=0) if c.adaptive_sampling else torch.zeros_like(_back)
        start = (pick - _back).clamp(min=0).clamp(max=_upper)
        # safeguard: start must be covered by a cache; snap uncovered → 0 (in [0,upper]; covered via
        # frame-0 init-save, else the restore falls back to reference+default pose).
        bad = ~candidates[start]
        start[bad] = 0
        # ── RSI 마스터 스위치(use_rsi=False): 모든 에피소드를 frame 0 에서 시작. use_rsi=True 경로를 그대로 두려고
        # 샘플링 블록 뒤에 둔다. 아래 짝 게이트가 리셋 상태도 레퍼런스 frame 0 으로 고정한다(캐시 복원 안 함).
        if not c.use_rsi:
            start = torch.zeros_like(start)
        # [ROLLBACK MARKER: friction-curriculum] 에피소드마다 물체 마찰 재추출
        self._apply_object_friction(env_ids)
        self._frame_idx[env_ids] = start
        self._episode_start_frame[env_ids] = start                 # diagnostics (Diag / rsi_start_mean)
        # reset the tracking-quality streak for the reset envs (grasp mechanism)
        self._enough_continued[env_ids] = True
        self._enough_idx[env_ids] = start
        # SONIC: 다음 _get_observations 에서 10프레임 proprio 이력을 리셋 자세로 채우고 last_action / last z_res 를
        # 지워 리셋 직후 첫 SONIC 스텝이 분포 안에 있게 한다.
        if getattr(self, "_sonic", None) is not None:
            self._sonic_hist_init[env_ids] = True
            self._last_a_sonic[env_ids] = 0.0
            self._last_z_res[env_ids] = 0.0
            self._cur_policy_action[env_ids] = 0.0        # prev-action obs AND action_rate = 0 at episode start
            self._prev_policy_action[env_ids] = 0.0
            self._cur_policy_action_bnd[env_ids] = 0.0    # (currently-unread A/B copy — kept in sync)
            self._prev_policy_action_bnd[env_ids] = 0.0

        # ---- restore state (train cache hit → reference+default) ----
        root_pose = torch.zeros(n, 7, device=dev); root_pose[:, 3] = 1.0
        root_vel = torch.zeros(n, 6, device=dev)
        jpos = self.robot.data.default_joint_pos[env_ids].clone()
        jvel = torch.zeros_like(jpos)
        org = self.scene.env_origins[env_ids]

        # 2-way source selection per env (train cache > reference+default),
        # vectorized via boolean masks + 2D advanced-index gathers (no per-env python loop).
        aid = self._action_joint_ids_t                               # (65,) action-joint columns
        train_hit = ~self._init_flg[start]                           # (n,)
        # [ROLLBACK MARKER: ref-start-prob] 낮은 확률로 캐시를 무시하고 레퍼런스에서 시작한다. 한 번 쓰인 나쁜
        # 캐시 상태가 그 프레임을 고착시키는 것을 막는다(레퍼런스 쪽이 나으면 캐시를 교체).
        if c.ref_start_prob > 0.0:
            _use_ref = torch.rand(n, device=dev) < float(c.ref_start_prob)
            train_hit = train_hit & (~_use_ref)
            self._diag_ref_start = float(_use_ref.float().mean())
        # [MASTER RSI SWITCH] use_rsi=False → 캐시를 읽지 않고 모두 레퍼런스 frame 0 자세로 리셋한다.
        # 캐시 기록은 계속하므로 True 로 되돌리면 쌓인 커버리지로 이어간다.
        if not c.use_rsi:
            train_hit = torch.zeros_like(train_hit)
        where_train = train_hit
        where_ref = ~train_hit

        if where_train.any():                                        # train cache: 222-D layout
            idx = where_train.nonzero(as_tuple=True)[0]
            s = self._state_cache[start[idx]]
            root_pose[idx, :3] = s[:, 1:4] + org[idx]; root_pose[idx, 3:7] = s[:, 4:8]
            root_vel[idx, :3] = s[:, 8:11]; root_vel[idx, 3:6] = s[:, 11:14]
            jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 27:92]
            jvel[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 92:157]
            self._smoothed_actions[env_ids[idx]] = s[:, 157:222]
        if where_ref.any():                                          # reference root + default/retargeted joints
            idx = where_ref.nonzero(as_tuple=True)[0]
            fr = start[idx]
            root_pose[idx, :3] = self._ref_root_pos[fr] + org[idx]
            root_pose[idx, 3:7] = self._ref_root_quat[fr]
            root_vel[idx, :3] = self._ref_root_linvel[fr]
            root_vel[idx, 3:6] = self._ref_root_angvel[fr]
            if self._ref_joints is not None:
                jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = self._ref_joints[fr]
                # [ROLLBACK MARKER: ref-reset-jvel] 관절 속도도 루트 속도와 같은 프레임 fr 의 레퍼런스로 채운다.
                # 스위치는 배율(_ref_jvel_scale)에 접어 두었다: 꺼지면 0 을 대입해 기존 동작과 비트 단위로 같다.
                jvel[idx.unsqueeze(1), aid.unsqueeze(0)] = torch.clamp(
                    self._ref_joint_vel[fr] * self._ref_jvel_scale,
                    -self._ref_jvel_clip, self._ref_jvel_clip)          # (1,65) 성분별 상한, 브로드캐스트
            self._smoothed_actions[env_ids[idx]] = self._unscale(jpos[idx][:, aid])


        # [ROLLBACK MARKER: tendon-reset] 텐던 축 J0 는 세 리셋 경로 모두 리타게팅 J0 로 쓴다(캐시도 J0 를 안 담는다).
        # [ROLLBACK MARKER: ref-j0] 캐시 경로에선 J1 이 레퍼런스와 다르므로 min() 으로 q_J0 <= q_J1 을 보장한다.
        # start = 전체 리셋 env 의 시작 프레임 (n,). fr 은 레퍼런스 경로 env 부분집합이라 여기서 쓰면 모양이 안 맞다.
        if getattr(self, "_ref_j0", None) is not None:
            jpos[:, self._ref_j0_ids] = torch.minimum(self._ref_j0[start], jpos[:, self._tendon_j1_ids])
            jvel[:, self._ref_j0_ids] = torch.clamp(
                self._ref_j0_vel[start] * self._ref_jvel_scale,
                -float(c.ref_reset_joint_vel_clip_hands), float(c.ref_reset_joint_vel_clip_hands))

        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)
        # [ankle-acc] 리셋 불연속은 Isaac Lab 이 처리한다 — write_joint_velocity_to_sim 이
        # _previous_joint_vel 을 새 속도로 맞추고 joint_acc 를 0 으로 만든다. 별도 씨딩 불필요.

        # 물체 복원: pretrain·레퍼런스 리셋은 레퍼런스 자세, train 캐시 히트는 캐시 [14:27] 에서 복원해
        # 캐시된 로봇과 물리적으로 맞는 짝을 유지한다(안 그러면 시작 직후 종료될 수 있다).
        if self._has_object:
            f0 = start
            ref_op = self._ref_obj_pos[f0] + org                    # (n,3) world
            # [ROLLBACK MARKER: spawn-declear] 스폰 위치만 받침 위로 올린다(레퍼런스 정지 프레임만). _ref_obj_pos 는
            # 그대로라 보상·관측은 GT 이고, 캐시 히트는 이미 시뮬된 자세라 이 보정을 쓰지 않는다.
            _lift = getattr(self, "_obj_spawn_lift", None)
            if _lift is not None:
                ref_op = ref_op.clone()
                ref_op[:, 2] = ref_op[:, 2] + _lift[f0]
            # [/ROLLBACK MARKER: spawn-declear]
            ref_oq = self._ref_obj_quat[f0]                         # (n,4)
            op = torch.zeros(n, 7, device=dev)
            op[:, :3] = ref_op; op[:, 3:7] = ref_oq
            # reference path: seed the object at its REFERENCE velocity for the sampled frame (mid-motion
            # starts place the object moving, not at rest). Cache-hit envs overwrite from the cache below.
            ovel = torch.zeros(n, 6, device=dev)
            ovel[:, :3] = self._ref_obj_linvel[f0]
            ovel[:, 3:6] = self._ref_obj_angvel[f0]
            if where_train.any():
                sc = self._state_cache[start]                        # (n,222)
                tw = where_train.unsqueeze(-1)
                op[:, :3] = torch.where(tw, sc[:, 14:17] + org, ref_op)
                op[:, 3:7] = torch.where(tw, sc[:, 17:21], ref_oq)
                ovel = torch.where(tw, sc[:, 21:27], ovel)
            self._object.write_root_pose_to_sim(op, env_ids=env_ids)
            self._object.write_root_velocity_to_sim(ovel, env_ids=env_ids)

        if self._sonic is not None:
            if getattr(self.cfg, "sonic_hand_residual", False) and getattr(self, "_hand_res_base", None) is not None:
                # [ROLLBACK MARKER: stage1-hand] 잔차 EMA 씨딩 = (복원된 손 PD 목표 − 그 프레임의 기준) 의 역매핑.
                # 캐시 히트는 담아둔 목표(_smoothed_actions 손 블록 → 관절 단위)를 되살리고, 레퍼런스 리셋은
                # 리셋 자세(리타게팅)에서 기준으로 EMA 가 몇 스텝에 걸쳐 이어집니다 — 1단계 env 의 fres 씨딩.
                _hsl = self._sonic_hand_slice
                _b = self._hand_res_base[start]                                                   # (n,36)
                _d = self._scale(self._smoothed_actions[env_ids])[:, _hsl] - _b
                _r0 = torch.where(_d >= 0, _d / (self._ctrl_upper[_hsl] - _b).clamp_min(1e-6),
                                  _d / (_b - self._ctrl_lower[_hsl]).clamp_min(1e-6))
                self._hand_res_ema[env_ids] = _r0.clamp(-1.0, 1.0)
                # [/ROLLBACK MARKER: stage1-hand]


    # -------------------------------------------------------- state cache write
    def _save_state_cache(self, reward: torch.Tensor) -> None:
        """Store per-frame best (highest-reward) full-body state into the 222-D train cache.

        `reward` is the ACTUAL step reward, exactly as grasp (robotis_sh5_grasp_env.py:1666 passes
        its own `reward.clamp(min=0.0)`) and as TJ's original (gr_env.py:608 compares
        `total_reward > state_cache[current_frame, 0]` and stores it at column 0). This env used to
        recompute a local proxy `-(body + hand + root_pos)` instead — a porting slip, since `reward`
        was already in scope one line above the call site. The proxy dropped `obj_pos`/`obj_rot`,
        i.e. the object term, from the ranking of a loco-MANIPULATION task, and also dropped `ee`,
        `ft_reward`, `root_rot`, contact force and feet-match, and ignored the per-term weights
        (it summed three errors 1:1:1 while the reward weights them separately). The object was
        still gated on (see `good` below), so the proxy only mis-ranked states that were already
        object-acceptable — but `enough_obj_threshold` is loose, so within that band it could not
        prefer the state whose object placement was actually better.

        Vectorized: build the full (E,222) state once, then scatter the highest-reward env
        into each UNIQUE frame it covers (loop is O(unique frames) << O(num_envs))."""
        if not hasattr(self, "_errs"):
            return
        c = self.cfg
        e = self._errs
        org = self.scene.env_origins
        fr = self._frame().clamp(max=self._ref_len - 1)                     # (E,)
        gate = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        # ---- 추적 품질 게이트(grasp 방식): 리셋 이후 계속 기준 안이어야 캐시 대상. 손끝 + 물체 3단계 임계
        # (시작 직후 / early / late) + 몸·루트 기준(enough_body/root_*). 물체가 없으면 물체 단계는 항상 통과. ----
        action_fps = round(1.0 / (c.sim.dt * c.decimation))
        start_cutoff = action_fps * 2 // 3                                  # first 2/3 s = 33 frames @50Hz
        reached_end = self.is_reached_end                                   # python bool
        op, orr = e["obj_pos"], e["obj_rot"]
        start_c = (op < 0.10) & (orr < 0.50) & (fr <= start_cutoff)
        early_c = (op < c.enough_obj_threshold) & (orr < c.enough_obj_rot_threshold) & (not reached_end)
        late_c = (op < c.enough_obj_threshold_late) & (orr < c.enough_obj_rot_threshold_late) & reached_end
        good = (e["ft"] < c.enough_ft_threshold) & (start_c | early_c | late_c)
        # floating-base 몸·루트 품질 기준(inf 면 끔).
        good = good & (e["body"] < c.enough_body_threshold) & (e["root_pos"] < c.enough_root_pos_threshold) \
            & (e["root_rot"] < c.enough_root_rot_threshold)
        still_good = self._enough_continued & good
        self._enough_idx = torch.where(still_good, fr, self._enough_idx)    # last good frame
        self._enough_continued = still_good

        # cache ranking key = the ACTUAL step reward (grasp / TJ convention; see the docstring).
        r = reward                                                          # (E,)
        # [ROLLBACK MARKER: deferred-cache] 유예 모드에선 여기서 `better` 를 비교하지 않는다. 에피소드 동안 캐시가
        # 바뀌므로 비교는 _flush_state_cache 의 커밋 때 하고, 여기서는 프레임별 품질 연속 조건으로만 스테이징한다.
        if getattr(self, "_pend_state", None) is not None:
            stage_mask = gate & self._enough_continued                     # (E,)
            if stage_mask.any():
                slot = self.episode_length_buf.clamp(max=self._pend_cap - 1)   # (E,)
                rows = torch.nonzero(stage_mask, as_tuple=False).squeeze(-1)
                _st = self._build_cache_state(r, org)
                self._pend_state[rows, slot[rows]] = _st[rows]
                self._pend_frame[rows, slot[rows]] = fr[rows]
                self._pend_valid[rows, slot[rows]] = True
            return
        # [/ROLLBACK MARKER: deferred-cache] ------------------------------------------------------

        # write only when tracking is still good AND the new state beats the cached reward.
        # Computed HERE, after the deferral branch above returns, so the per-step fancy-index gather
        # is not paid when the commit is deferred (there it happens once, at flush time).
        better = r > self._state_cache[fr, 0]                              # (E,) fancy-index gather
        update_mask = gate & self._enough_continued & better               # (E,)
        if not update_mask.any():
            return
        state = self._build_cache_state(r, org)
        for uf in torch.unique(fr[update_mask]):
            m = (fr == uf) & update_mask
            best_env = m.nonzero(as_tuple=True)[0][r[m].argmax()]
            self._state_cache[uf] = state[best_env]
            self._init_flg[uf] = False
            self._reached_frame = max(self._reached_frame, int(uf.item()))

    def _build_cache_state(self, r: torch.Tensor, org: torch.Tensor) -> torch.Tensor:
        """(E,222) cache row for every env: [0] = the step reward (ranking key), rest = the full
        restorable sim state. Column 0 must be the SAME quantity the `better` comparison uses."""
        state = torch.empty(self.num_envs, self._STATE_DIM, device=self.device)
        state[:, 0] = r
        state[:, 1:4] = self.robot.data.root_pos_w - org
        state[:, 4:8] = self.robot.data.root_quat_w
        state[:, 8:11] = self.robot.data.root_lin_vel_w
        state[:, 11:14] = self.robot.data.root_ang_vel_w
        if self._has_object:
            state[:, 14:17] = self._object.data.root_pos_w - org
            state[:, 17:21] = self._object.data.root_quat_w
            state[:, 21:24] = self._object.data.root_lin_vel_w
            state[:, 24:27] = self._object.data.root_ang_vel_w
        else:
            state[:, 14:27] = 0.0
        state[:, 27:92] = self.robot.data.joint_pos[:, self._action_joint_ids_t]
        state[:, 92:157] = self.robot.data.joint_vel[:, self._action_joint_ids_t]
        state[:, 157:222] = self._smoothed_actions
        return state

    # [ROLLBACK MARKER: deferred-cache] -----------------------------------------------------------
    def _flush_state_cache(self, env_ids: torch.Tensor, ep_len: torch.Tensor) -> None:
        """Commit the staged states of TERMINATING envs, in bulk (see cfg.cache_min_episode_length).

        Called from _reset_idx BEFORE `super()._reset_idx()` zeroes `episode_length_buf`, so `ep_len`
        must be captured by the caller. Only envs whose episode lasted >= cache_min_episode_length
        contribute anything — that hindsight filter is the reason the commit is deferred at all.
        For every (env, slot) still marked valid we keep the highest-reward candidate per frame and
        write it only if it beats what the cache holds NOW (the cache moved while the episode ran).
        """
        if self._pend_state is None or len(env_ids) == 0:
            return
        keep = ep_len >= int(self.cfg.cache_min_episode_length)
        rows = env_ids[keep]
        if len(rows):
            valid = self._pend_valid[rows]                                   # (R, cap)
            if valid.any():
                sel = torch.nonzero(valid, as_tuple=False)                   # (K,2) [row, slot]
                cand_state = self._pend_state[rows[sel[:, 0]], sel[:, 1]]    # (K,222)
                cand_frame = self._pend_frame[rows[sel[:, 0]], sel[:, 1]]    # (K,)
                cand_r = cand_state[:, 0]
                # per frame: best candidate in this flush, then the usual "only if better" vs cache
                for uf in torch.unique(cand_frame):
                    m = cand_frame == uf
                    j = cand_r[m].argmax()
                    best = cand_state[m][j]
                    if best[0] > self._state_cache[uf, 0]:
                        self._state_cache[uf] = best
                        self._init_flg[uf] = False
                        self._reached_frame = max(self._reached_frame, int(uf.item()))
        # clear staging for ALL terminating envs (kept or dropped) so the next episode starts clean
        self._pend_valid[env_ids] = False
    # [/ROLLBACK MARKER: deferred-cache] ----------------------------------------------------------
