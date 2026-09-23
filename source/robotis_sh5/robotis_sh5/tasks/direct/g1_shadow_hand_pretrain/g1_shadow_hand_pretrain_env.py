"""hand_pretrain(1단계) 환경: 떠 있는 양손 Shadow 로 손 동작을 물리 시뮬레이션에서 다듬는다.

sonic residual env 를 복사해 몸과 SONIC 을 뺀 것이다. 손마다 독립 articulation(env 원점에 고정된 anchor +
손목 관절 6)이고, action 48 = 손별 [손목 관절 잔차 6 | 손가락 margin 잔차 18], obs 553.
residual_action=False 면 손가락은 절대 액션 + EMA 다. 에피소드·레퍼런스·규약은 sonic residual env 와 같다.
"""

from __future__ import annotations

import math
import json
import os

import numpy as np
import torch

# [hand-pretrain] 손목 6D 회전 -> 축각. workspaceTJ 의 gr_env 가 쓰는 것과 같은 구현입니다.

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from ..g1_shadow_sonic_residual import cws as CWS
from ..g1_shadow_sonic_residual.g1_shadow_sonic_residual_env_cfg import (
    _ROBOT_USD,
    FINGERTIP_OFFSETS,
    FINGERTIP_PAD_NORMALS,
    HAND_CHAIN,
    JOINT_GROUPS,
    LINK_CONTACT_NAMES,
    LINK_PAD_NORMALS,
    N_HAND_KPTS_PER_HAND,
    N_LINK_CONTACT,
)
from .g1_shadow_hand_pretrain_env_cfg import G1ShadowHandPretrainEnvCfg

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


class G1ShadowHandPretrainEnv(DirectRLEnv):
    cfg: G1ShadowHandPretrainEnvCfg

    # ------------------------------------------------------------------ init
    def __init__(self, cfg: G1ShadowHandPretrainEnvCfg, render_mode: str | None = None, **kwargs):
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
        _post_init_buffers). Keypoint order (42) = left-hand(21) + right-hand(21),
        matching the robot-side keypoint order built in _post_init_buffers."""
        clip_dir = self._resolve_clip_dir(cfg)
        npz_path = os.path.join(clip_dir, "trajectory.npz") if clip_dir else ""
        if not npz_path or not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"ParaHome clip not found (dataset_root={cfg.dataset_root}, class={cfg.clip_class}, "
                f"name={cfg.clip_name or '<auto>'}). Run scripts/process_dataset/dataset/parahome.py first.")
        d = np.load(npz_path, allow_pickle=True)
        # 키포인트 소스를 SMPL-X 로 (2026-09-04, 리타게팅과 같은 스켈레톤).
        # 배열 = [smplx_joints(55) | fingertip_pad_pos(10)] = (F,65,3). SMPL-X 손에는 손끝 관절이 없어 pad 정점을 쓴다.
        if "smplx_joints" not in d.files:
            raise KeyError("[smplx-kpts] smplx_joints 없음 — parahome.py --overwrite 로 재생성하세요")
        jp = np.concatenate([d["smplx_joints"].astype(np.float32),
                             d["fingertip_pad_pos"].astype(np.float32)], axis=1)   # (F,65,3)
        F = jp.shape[0]

        # HAND_CHAIN["parahome"] 값: p>=0 손 블록 로컬 인덱스(왼 25 / 오 40 기준), -10 손목(SMPL-X 20/21),
        # -1..-5 pad 손가락(th, ff, mf, rf, lf) → 55 + side*5 + (-p-1)
        _PAD_BASE = 55
        ref_idx: list[int] = []                                # 손 21 x2 (몸통 키포인트는 쓰지 않는다)
        for _s, (_hb, _wr, _pb) in enumerate(((25, 20, _PAD_BASE), (40, 21, _PAD_BASE + 5))):
            for spec in HAND_CHAIN.values():
                for p in spec["parahome"]:
                    if p >= 0:
                        ref_idx.append(_hb + p)
                    elif p == -10:
                        ref_idx.append(_wr)
                    else:
                        ref_idx.append(_pb + (-p - 1))
        self._np_ref_kpts = jp[:, ref_idx, :]                  # (F,42,3) = 왼손 21 + 오른손 21

        # 사람 골반 위치 (F,3) — 카메라 조준에만 쓴다. 리타게팅 g1_root_pose 가 있으면 아래에서 그것으로 바꾼다.
        self._np_root_pos = d["body_global_transform"][:, :3, 3].astype(np.float32)

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
        # [hand-state-cache] 리타게팅 산출물 디렉터리를 그대로 보관합니다. wrist_ref.npz 를 읽는
        # 곳과 같은 디렉터리이고, train.py 가 학습 종료 시 상태 캐시를 여기에 씁니다.
        # cfg 필드로 재구성하면 clip_name="" (env 가 자동 선택) 인 경우 어긋납니다.
        self._retarget_dir = os.path.dirname(rt)
        self._np_ref_joints = None
        self._ref_joint_names: list[str] | None = None
        if os.path.exists(rt):
            rd = np.load(rt, allow_pickle=True)
            if "g1_joint_pos" in rd.files:
                self._np_ref_joints = rd["g1_joint_pos"].astype(np.float32)           # (F,65)
            # the column layout g1_joint_pos was WRITTEN in.
            # Newer retarget runs record it in the npz; older ones do not, and for those the layout is
            # whatever g1_shadow_joint_order.json held at solve time. See _post_init_buffers.
            if "joint_names" in rd.files:
                self._ref_joint_names = [str(x) for x in rd["joint_names"]]
            if "g1_root_pose" in rd.files:
                self._np_root_pos = rd["g1_root_pose"][:, :3].astype(np.float32)        # (F,3)
            # [hand-pretrain] 리타게팅 손목 pose(RSI 초기 자세). 떠 있는 손의 루트는 손목이지만 리타게팅 npz 에 없어서
            # export_wrist_ref.py 가 FK 로 뽑아 둔 wrist_ref.npz(리타게팅 산출물 옆)를 읽는다. 없으면 실패시킨다.
            _wr = os.path.join(os.path.dirname(rt), "wrist_ref.npz")
            if not os.path.exists(_wr):
                raise FileNotFoundError(
                    f"{_wr} 없음 — scripts/process_dataset/retarget/export_wrist_ref.py 를 먼저 실행하세요")
            _wd = np.load(_wr, allow_pickle=True)
            self._np_wrist_pose = np.stack([_wd["wrist_pose_l"], _wd["wrist_pose_r"]], axis=0)  # (2,F,7)
            # [wrist6] 손목 목표 = 6-DoF 관절값(export_wrist_dof6.py 가 YZX 로 분해).
            # 저장된 rot_seq 를 에셋 축 순서와 대조한다 — 어긋나면 pose 가 조용히 틀어진다.
            _w6p = os.path.join(os.path.dirname(rt), "wrist_dof6.npz")
            if not os.path.exists(_w6p):
                raise FileNotFoundError(
                    f"{_w6p} 없음 — scripts/process_dataset/retarget/export_wrist_dof6.py 를 먼저 실행하세요")
            _d6 = np.load(_w6p, allow_pickle=True)
            _seq = str(_d6["rot_seq"]) if "rot_seq" in _d6.files else "?"
            if _seq != "YZX":
                raise ValueError(
                    f"{_w6p} 의 rot_seq={_seq!r} 가 에셋(YZX)과 다릅니다 — "
                    f"export_wrist_dof6.py 를 --overwrite 로 다시 실행하세요")
            self._np_wrist_dof6 = np.stack(
                [_d6["wrist_dof_l"], _d6["wrist_dof_r"]], axis=0).astype(np.float32)  # (2,F,6)


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
            self._np_ft_pad = _rl(self._np_ft_pad)
            self._np_obj_base = np.concatenate(
                [_rl(self._np_obj_base[:, :3]), _rq(self._np_obj_base[:, 3:7])], axis=1)
            self._np_obj_dof = (_rl(self._np_obj_dof) if self._np_obj_dof.shape[1] > 0
                                else np.zeros((N, 0), np.float32))
            if self._has_link_contact:
                self._np_link_contact_mask = (_rl(self._np_link_contact_mask) > 0.5).astype(np.float32)
                ln = _rl(self._np_link_contact_normal)                 # (N,L,3) resampled reaction normal
                lnn = np.linalg.norm(ln, axis=-1, keepdims=True)
                self._np_link_contact_normal = np.where(lnn > 1e-6, ln / np.clip(lnn, 1e-6, None), ln).astype(np.float32)
                self._np_link_contact_target = _rl(self._np_link_contact_target)   # (N,L,3) object-local target
            if self._np_ref_joints is not None:
                self._np_ref_joints = _rl(self._np_ref_joints)
            # [hand-pretrain] 손목 pose 도 같은 격자로 — 위치는 선형, 쿼터니언은 slerp
            self._np_wrist_pose = np.stack(
                [np.concatenate([_rl(self._np_wrist_pose[k][:, :3]),
                                 _rq(self._np_wrist_pose[k][:, 3:7])], axis=1) for k in (0, 1)], axis=0)
            # [wrist6] 6-DoF 관절값은 6개 성분 전부 **선형** 보간이다. exporter 가 branch 를
            # 이전 프레임 최근접으로 골라 연속으로 저장하므로(재구성 오차 5e-14도) 선형이
            # 안전하다 — 그게 branch 선택을 넣은 이유다. 각도를 wrap 된 상태로 보간하면 안 된다.
            if self._np_wrist_dof6 is not None:
                self._np_wrist_dof6 = np.stack(
                    [_rl(self._np_wrist_dof6[k]) for k in (0, 1)], axis=0).astype(np.float32)
            cfg.ref_dt = 1.0 / tgt_fps                                   # runtime rate
            F = N

        self._ref_len = int(F)
        self._n_obj_parts = int(self._np_obj_dof.shape[1])

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
                    # 겹침 해소 속도 상한. 0.1 로 낮췄다가(2026-08-14)
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
    # 논문과 같은 물체 크기 정의 (2026-09-02)
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

    def _setup_scene(self) -> None:
        # [hand-pretrain] 양손을 독립 articulation 두 개로 스폰한다(가상 링크로 이으면 양손이 인공적으로 결합된다).
        # self.robot 은 부모 코드 호환용 왼손 별칭이라 양손이 필요한 곳에선 self._hands / hand_l / hand_r 를 쓴다.
        self.hand_l = Articulation(self.cfg.hand_l_cfg)
        self.hand_r = Articulation(self.cfg.hand_r_cfg)
        self._hands = (self.hand_l, self.hand_r)
        self.scene.articulations["hand_l"] = self.hand_l
        self.scene.articulations["hand_r"] = self.hand_r
        self.robot = self.hand_l          # 부모 코드 호환용 별칭 (손 특정 경로에서는 쓰지 말 것)

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
            # [hand-pretrain] articulation 이 둘로 나뉘었으므로 프림 경로만 손별로 바꿉니다.
            # LINK_CONTACT_NAMES 는 robot0_l_* / robot0_r_* 로 좌우를 이미 담고 있고, 리스트
            # 순서가 그대로 유지되므로 _link_contact_forces() 의 위치 인덱싱은 수정 불필요합니다.
            _hp = "HandL" if "_l_" in name else "HandR"
            s = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/{_hp}/{name}",
                filter_prim_paths_expr=obj_filter, history_length=1, update_period=_ctrl_dt,
                # contact_pos_w = 링크-물체 접촉점 평균(월드), CWS 렌치의 모멘트 팔.
                # 접촉이 없는 쌍은 NaN 이므로 반드시 마스크로 거른다.
                track_air_time=False, track_contact_points=bool(self.cfg.track_contact_points),
                # ParaHome objects are CONVEX-DECOMPOSITION colliders (many sub-hulls) → a link can touch
                # several at once → >4 manifold points → raise the contact-data buffer cap (else a HARD
                # device-side assert in ContactSensor._unpack_contact_buffer_data).
                max_contact_data_count_per_prim=self.cfg.ft_max_contact_points))
            self._link_contact_sensors.append(s)
            self.scene.sensors[f"linkc_{name}"] = s

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

    def _remap_ref_joints(self) -> None:
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
        # [hand-pretrain] 리타게팅 열을 손마다 18개씩 이름 기준으로 재배열해 (F,36) 으로 잇는다(왼손 18 → 오른손 18).
        _jn = {sd: self._hands[i].data.joint_names for i, sd in enumerate("lr")}
        _order = {sd: [_jn[sd][i] for i in getattr(self, f"_finger_joint_ids_{sd}").tolist()]
                  for sd in "lr"}
        env_order = _order["l"] + _order["r"]
        missing = [n for n in env_order if n not in src]
        if missing or len(src) != self._ref_joints.shape[1]:
            raise RuntimeError(
                f"[ref-joints] cannot map the retarget columns onto the env's action joints: "
                f"{len(missing)} name(s) absent from the source layout (e.g. {missing[:3]}), "
                f"source width {len(src)} vs g1_joint_pos width {self._ref_joints.shape[1]}. "
                f"Re-run scripts/process_dataset/retarget/retarget_g1_pyroki.py for this clip.")
        # 리타게팅이 푼 손가락 J0 8개를 리셋에 쓴다 (2026-09-02). 텐던 축 J0 는 액션 관절이
        # 아니어서 이 값이 없으면 리셋이 J0 를 채울 수 없다 → 73열 리타게팅 npz 가 필수.
        _j0n = [f"robot0_{sd}_{fg}J0" for sd in "lr" for fg in ("FF", "MF", "RF", "LF")]
        _miss0 = [n for n in _j0n if n not in src]
        if _miss0:
            raise RuntimeError(f"[ref-j0] 리타게팅 npz 에 손가락 J0 열이 없습니다 (예: {_miss0[:2]}). "
                               f"scripts/process_dataset/retarget/retarget_g1_pyroki.py 로 이 클립을 다시 만드세요.")
        _c = torch.tensor([src.index(n) for n in _j0n], device=self.device, dtype=torch.long)
        self._ref_j0 = self._ref_joints[:, _c].clone()                       # (F,8) [l×4, r×4]
        # [hand-pretrain] J0 인덱스도 손별로 — 왼손 4개는 hand_l, 오른손 4개는 hand_r 의
        # DOF 인덱스입니다. _reset_idx 가 손별로 나눠 씁니다.
        self._ref_j0_ids = {
            sd: torch.tensor([_jn[sd].index(n) for n in _j0n if f"_{sd}_" in n],
                             device=self.device, dtype=torch.long) for sd in "lr"}
        print(f"[ref-j0] 리타게팅이 푼 J0 8개 보존 — 중앙값 "
              f"{self._ref_j0.median(dim=0).values.cpu().numpy().round(3).tolist()}")
        perm = torch.tensor([src.index(n) for n in env_order], device=self.device, dtype=torch.long)  # (36,)
        n_moved = int((perm != torch.arange(len(perm), device=self.device)).sum())
        self._ref_joints = self._ref_joints[:, perm]
        origin = "npz joint_names" if self._ref_joint_names is not None else "g1_shadow_joint_order.json"
        print(f"[ref-joints] retarget columns remapped by name from {origin}: "
              f"{n_moved}/{len(perm)} slots moved")

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
                # [hand-pretrain] 두 손 모두 치워야 합니다. 왼손만 치우면 오른손이 그대로 남아
                # 물체 침하량 측정을 오염시킵니다.
                for _hh in self._hands:
                    _hh.write_root_pose_to_sim(park)
                    _hh.write_root_velocity_to_sim(zero6)
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
        print(f"[context-z] 컨텍스트 {moved}개를 {dz * 1000:.1f} mm 내렸습니다 — 남은 보정을 다시 풉니다")
        self._solve_spawn_declear()             # 내린 뒤 남은 보정을 측정 (보통 0에 가깝습니다)

    # ---------------------------------------------------------- post-init buffers
    def _post_init_buffers(self) -> None:
        dev = self.device
        c = self.cfg

        # [hand-pretrain] 액션 관절은 손당 18개. JOINT_GROUPS["hands"] 정규식의 (l|r) 를 그 손으로 고정해 찾는다.
        _hand_expr = JOINT_GROUPS["hands"]["expr"]
        self._finger_joint_ids_l, self._finger_joint_ids_r = None, None
        for _side, _hand in (("l", self.hand_l), ("r", self.hand_r)):
            _e = [x.replace("(l|r)", _side) for x in _hand_expr]
            _ids, _ = _hand.find_joints(_e)
            assert len(_ids) == 18, f"{_side}손 구동 관절 {len(_ids)}개 (기대 18): {_e}"
            setattr(self, f"_finger_joint_ids_{_side}",
                    torch.tensor(_ids, device=dev, dtype=torch.long))
        # 부모 코드가 참조하는 이름들은 왼손 기준으로 채워 둡니다 — 손 특정 경로는
        # _finger_joint_ids_{l,r} 을 쓰고, 아래 두 개는 호환용입니다.
        self._action_joint_ids = self._finger_joint_ids_l.tolist()
        self._group_slices = {"hands": slice(0, 18)}
        self._action_joint_ids_t = self._finger_joint_ids_l
        off = 18
        # 액션 관절의 실제 이름 순서(find_joints 는 articulation 내부 순서로 돌려준다; rollout.py 가 함께 저장).
        # [hand-pretrain] 36열 = 왼손 18 → 오른손 18.
        self._action_joint_names = [self._hands[_h].joint_names[i]
                                    for _h, _sd in enumerate("lr")
                                    for i in getattr(self, f"_finger_joint_ids_{_sd}").tolist()]
        if bool(os.environ.get("PRINT_ACTION_JOINTS")):
            print("[action-joints] " + " ".join(f"{i}:{n}" for i, n in
                                                enumerate(self._action_joint_names)))
        self._n_act = off                                              # 65
        # 텐던 축 J0 8개의 리셋 값. J0 는 액션 관절이 아니어서 따로 쓰지 않으면 0 으로
        # 남아, J1 이 굽은 프레임에서 텐던 제약을 어긴 채 시작해 말단이 튄다. 값은 리타게팅 J0 또는 캐시에서 온다.
        # [hand-pretrain] 텐던 쌍은 손 articulation(22 DOF) 기준 인덱스라 손별 dict.
        self._tendon_j1_ids, _npair = {}, 0
        for _h, _sd in enumerate("lr"):
            _jn2 = self._hands[_h].data.joint_names
            _tj1n = [f"robot0_{_sd}_{f}J1" for f in ("FF", "MF", "RF", "LF")]
            _tp = [(_jn2.index(a), _jn2.index(a[:-1] + "0")) for a in _tj1n
                   if a in _jn2 and (a[:-1] + "0") in _jn2]
            self._tendon_j1_ids[_sd] = torch.tensor([a for a, _b in _tp], device=dev, dtype=torch.long)
            _npair += len(_tp)
        print(f"[tendon-reset] 텐던 쌍 {_npair}개(양손)")
        self._finger_alpha = float(c.finger_ema_alpha)
        # [hand-pretrain] 액션 관절 = 손 36개(왼 18 + 오른 18).
        self._n_act = 36

        # 관절 한계: 손별 articulation 에서 읽어 왼손 → 오른손 순서로 잇는다(_unscale/_scale·리셋 버퍼의 전제).
        # 손가락 soft 한계를 soft_joint_pos_limit_factor_hands 로 다시 계산한다
        # (두 단계가 같은 cfg 값으로 손 범위를 맞춘다. wrist6 관절 제외).
        _fh = float(getattr(self.cfg, "soft_joint_pos_limit_factor_hands", 0.0) or 0.0)
        if _fh > 0.0:
            import re as _re
            _n_done = 0
            for _h in self._hands:
                _fj = torch.tensor([i for i, n in enumerate(_h.joint_names)
                                    if _re.match(r"robot0_[lr]_(FF|MF|RF|LF|TH)J\d", n)], device=self.device, dtype=torch.long)
                _hard = _h.data.joint_pos_limits[:, _fj]
                _mid = 0.5 * (_hard[..., 0] + _hard[..., 1]); _rng = _hard[..., 1] - _hard[..., 0]
                _h.data.soft_joint_pos_limits[:, _fj, 0] = _mid - 0.5 * _rng * _fh
                _h.data.soft_joint_pos_limits[:, _fj, 1] = _mid + 0.5 * _rng * _fh
                _n_done += int(_fj.numel())
            print(f"[soft-limit-hands] 손가락 {_n_done}관절 soft 한계 factor → {_fh} 재계산 (하드 한계 기준, wrist6 제외)")
        _lims = [self._hands[i].data.soft_joint_pos_limits[
                     0, getattr(self, f"_finger_joint_ids_{sd}")]
                 for i, sd in enumerate("lr")]
        lim = torch.cat(_lims, dim=0)                                  # (36,2)
        assert lim.shape == (36, 2), f"관절 한계 형상 {tuple(lim.shape)} != (36,2)"
        self._ctrl_lower = lim[:, 0].clone()
        self._ctrl_upper = lim[:, 1].clone()
        # [residual] 손가락 최종 목표의 EMA 상태 (관절 단위 36열). _smoothed_actions 는 정규화
        # 액션 단위이고 절대 경로가 쓰므로 별 버퍼로 둔다. 리셋에서 레퍼런스로 씨딩한다.
        self._hand_target_ema = torch.zeros(self.num_envs, 36, device=dev)
        # 손가락 잔차 전용 EMA 상태 (정규화 액션 단위, cfg.finger_ema_on_residual
        # 일 때만 사용). 손목의 _wrist6_res_ema 와 같은 배치 — 레퍼런스는 지연 없이 통과, 잔차만 평활.
        # 마진 매핑은 매 스텝 그 프레임의 여유로 다시 계산하므로 |res|≤1 이면 목표가 한계 안이다.
        self._fing_ema_res = bool(getattr(c, "finger_ema_on_residual", False))
        self._hand_res_ema = torch.zeros(self.num_envs, 36, device=dev)
        self._group_slices = {"hands": slice(0, 36)}

        # 정책 액션 사본(관측 prev_action + action_rate 보상). SONIC 경로가 꺼져 있어도
        # 필요하므로 가드 밖에서 할당합니다. 폭은 액션 그대로 48.
        self._cur_policy_action = torch.zeros(self.num_envs, int(c.action_space), device=dev)
        self._prev_policy_action = torch.zeros(self.num_envs, int(c.action_space), device=dev)
        self._cur_policy_action_bnd = torch.zeros(self.num_envs, int(c.action_space), device=dev)
        self._prev_policy_action_bnd = torch.zeros(self.num_envs, int(c.action_space), device=dev)

        # [wrist6] 손별 액션 = [wrist6 | finger18] = 24, 양손 48.
        self._ACT_W = 6                             # 손목 액션 폭 / 손
        self._ACT_PH = self._ACT_W + 18             # 손별 총 폭
        assert int(c.action_space) == 2 * self._ACT_PH, f"action_space {c.action_space} != {2 * self._ACT_PH}"
        _W6N = ["tx", "ty", "tz", "rot1", "rot2", "rot3"]
        for _sd, _hd in (("l", self.hand_l), ("r", self.hand_r)):
            _jn = _hd.data.joint_names
            _ids = [_jn.index(f"robot0_{_sd}_wrist_{n}") for n in _W6N]
            setattr(self, f"_wrist6_joint_ids_{_sd}",
                    torch.tensor(_ids, device=dev, dtype=torch.long))
        # 잔차 스케일. DexMachina 와 같은 (0.04 m, 0.5 rad). 측정: 이 스케일에 EMA 0.2 면
        # 손목 힘 최대 28 N / 9.1 N·m 로 포화 없음. 평활을 빼면(백색잡음) 178 N / 71.6 N·m
        # 까지 가고 포화 11.2% 가 되지만 effort 한계가 걸려 발산은 하지 않았다.
        self._wrist6_res_scale = torch.tensor(
            [float(c.wrist6_res_pos)] * 3 + [float(c.wrist6_res_rot)] * 3, device=dev)
        # EMA 를 최종 목표에 걸면 평활 상태가 곧 목표다. 리셋에서 레퍼런스로 씨딩한다
        # (0 에서 시작하면 초기 몇 스텝이 원점에서 램프업한다).
        self._wrist6_target = [torch.zeros(self.num_envs, 6, device=dev) for _ in range(2)]
        self._wrist6_target_ema = float(c.wrist6_target_ema)
        # 잔차 전용 EMA 상태 (cfg.wrist6_ema_on_residual 일 때만 사용).
        # 목표 = ref[frame] + 이 값. 리셋에서 0, 캐시 복원에서 target − ref[frame] 으로 되살린다.
        self._w6_ema_res = bool(getattr(c, "wrist6_ema_on_residual", False))
        self._wrist6_res_ema = [torch.zeros(self.num_envs, 6, device=dev) for _ in range(2)]
        # 관절 한계 (에셋: 병진 ±2.0 m, 회전 ±720도). 잔차가 이 밖으로 나가지 않게 clamp.
        self._wrist6_lo, self._wrist6_hi = [], []
        for _h, _sd in enumerate("lr"):
            _lim = self._hands[_h].data.soft_joint_pos_limits[
                0, getattr(self, f"_wrist6_joint_ids_{_sd}")]
            self._wrist6_lo.append(_lim[:, 0].clone())
            self._wrist6_hi.append(_lim[:, 1].clone())
        print(f"[wrist6] 관절 구동: 잔차 스케일 pos={c.wrist6_res_pos} m "
              f"rot={c.wrist6_res_rot} rad, 목표 EMA={c.wrist6_target_ema}, "
              f"한계 병진 [{self._wrist6_lo[0][0]:.2f},{self._wrist6_hi[0][0]:.2f}] m "
              f"회전 [{self._wrist6_lo[0][3]:.2f},{self._wrist6_hi[0][3]:.2f}] rad")

        # ---- keypoint body ids + local offsets (56, matching the reference order) ----
        # [hand-pretrain] 몸통 13개 제거 — 손 articulation 에 그 링크들이 없습니다. 남는 것은
        # 손 42개(손당 21)뿐이고, 레퍼런스 쪽 _ref_kpts 도 아래에서 같은 슬라이스로 잘립니다.
        kpt_names: list[str] = []
        kpt_off: list[list[float]] = []
        # 오프셋은 링크 이름이 아니라 키포인트별 `pad` 플래그로 정한다.
        # distal 링크를 두 번 쓰기 때문이다(오프셋 0 = DIP, pad = TIP).
        for side in ("l", "r"):
            for spec in HAND_CHAIN.values():
                for body, use_pad in zip(spec["shadow"], spec["pad"]):
                    full = f"robot0_{side}_{body}"
                    kpt_names.append(full)
                    kpt_off.append(FINGERTIP_OFFSETS.get(full, [0.0, 0.0, 0.0])
                                   if use_pad else [0.0, 0.0, 0.0])
        self._kpt_sides, self._kpt_body_ids = self._find_hand_bodies(kpt_names)
        self._kpt_offsets = torch.tensor(kpt_off, device=dev, dtype=torch.float32)      # (54,3)
        # [hand-pretrain] 몸통이 없으므로 wrist/ee 그룹을 양손 palm 2개(손 블록 첫 항목)로 잡는다.
        self._palm_kpt_idx = torch.tensor([0, N_HAND_KPTS_PER_HAND], device=dev, dtype=torch.long)
        # 아래 둘은 _compute_errors 의 palm_per(= 위 2개를 자른 결과, 폭 2) 안에서의 위치입니다.
        # 42열 색인을 그대로 쓰면 palm_per[:, 21] 을 읽어 CUDA device-side assert 가 납니다.
        _both = torch.tensor([0, 1], device=dev, dtype=torch.long)
        self._wrist_kpt_idx = _both       # term_wrist_pos_err + Error/wrist_kpts
        self._ee_kpt_idx = _both          # rew_ee_kpts (손목 도달)
        print(f"[hand-pretrain] kpt {len(kpt_names)}개(손 전용), palm 인덱스 "
              f"{self._palm_kpt_idx.tolist()} = {[kpt_names[i] for i in self._palm_kpt_idx.tolist()]}")
        # ---- fingertip (10) body ids + offsets + pad normals (bimanual) ----
        self._ft_sides, self._ft_body_ids = self._find_hand_bodies(list(c.fingertip_body_names))
        self._ft_offsets = torch.tensor(
            [FINGERTIP_OFFSETS[n] for n in c.fingertip_body_names], device=dev, dtype=torch.float32)   # (10,3)
        self._ft_pad_normals = torch.tensor(
            [FINGERTIP_PAD_NORMALS[n] for n in c.fingertip_body_names], device=dev, dtype=torch.float32)  # (10,3)
        # ---- palm (wrist) body ids for explicit palm orientation/velocity obs (L,R to match ft order) ----
        self._palm_sides, self._palm_body_ids = self._find_hand_bodies(
            ["robot0_l_palm", "robot0_r_palm"])

        # ---- reference tensors (move numpy → device) ----
        def T(a):
            return torch.from_numpy(np.asarray(a)).to(dev)
        # 로봇 쪽 _kpt_body_ids 도 손 42개라 두 배열의 색인이 1:1 로 맞는다(왼손 21 → 오른손 21).
        self._ref_kpts = T(self._np_ref_kpts)                            # (F,42,3)
        assert self._ref_kpts.shape[1] == 2 * N_HAND_KPTS_PER_HAND, \
            f"레퍼런스 손 키포인트 {self._ref_kpts.shape[1]} != 2x{N_HAND_KPTS_PER_HAND}"
        self._ref_wrist_pose = T(self._np_wrist_pose)                  # [hand-pretrain] (2,F,7) wxyz
        # [wrist6] (2,F,6) = tx,ty,tz[m], rot1,rot2,rot3[rad] (내재 YZX, branch 연속)
        self._ref_wrist_dof6 = (T(self._np_wrist_dof6)
                                if getattr(self, "_np_wrist_dof6", None) is not None else None)
        self._ref_ft_pad = T(self._np_ft_pad)                          # (F,10,3)
        self._ref_obj_pos = T(self._np_obj_base[:, :3])                # (F,3)
        self._ref_obj_quat = _canon(T(self._np_obj_base[:, 3:7]))      # (F,4)
        self._ref_obj_linvel = T(self._np_obj_linvel)                  # (F,3) reference object linear vel
        self._ref_obj_angvel = T(self._np_obj_angvel)                  # (F,3) reference object angular vel
        # (F,) 물체-속도 게이트. grasp env 의 future_contact
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
        self._remap_ref_joints()
        # 레퍼런스 관절 속도 (F,65), 액션 관절 순서. 후방차분, 0번 프레임 0.
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
        # 손가락 J0 속도도 리타게팅 결과에서 뽑는다(위와 같은 규약).
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
        self._has_object = self._object is not None
        self._RESERVE_ARTIC = 4                                        # reserved obs slots per parts

        # 링크별 접촉(Option A): 레퍼런스 mask + 물체 로컬 법선·목표 + 로봇 body id. _ft_distal_idx 는 손끝 10개를
        # LINK_CONTACT_NAMES 의 distal 링크로 매핑해 손끝 kpt 보상과 delta_ft_obj 가 같은 맵을 읽게 한다.
        self._ref_link_contact_mask = T(self._np_link_contact_mask)            # (F,L)
        self._ref_link_contact_normal_local = T(self._np_link_contact_normal)  # (F,L,3) object-local
        self._ref_link_contact_target_local = T(self._np_link_contact_target)  # (F,L,3) object-local

        # 접촉을 렌치(물체를 어떻게 움직일 수 있는가)로 비교 (CHORD, arXiv 2607.00033).
        # 링크 위치 매칭과 달리 손바닥 대신 손가락으로 같은 힘을 내도 인정된다. 사람 쪽 σ_h 는 여기서 한 번만 계산한다.
        self._cws_sigma_h = None
        if self._has_link_contact and c.contact_reward_mode in ("cws", "both"):
            _m = self._ref_link_contact_mask > 0.5                             # (F,L)
            if bool(_m.any()):
                # 렌치 회전 성분을 나눌 물체 크기 = 메시 중심에서 최대 거리.
                # 메시를 못 읽으면 접촉점 노름 0.9 분위로 대체. 되돌리기: _rc_mesh = None
                _rc_q90 = float(self._ref_link_contact_target_local[_m].norm(dim=-1).quantile(0.9))
                _rc_mesh = self._object_mesh_radius() if self._has_object else None
                _rc_ok = _rc_mesh is not None and _rc_mesh > 1e-4
                self._cws_len = _rc_mesh if _rc_ok else _rc_q90
                self._cws_basis = CWS.make_basis(c.cws_n_dir, c.cws_seed, device=dev)
                # 모멘트 팔은 물리 COM 기준이다. 레퍼런스 목표도 COM 프레임으로 옮기고
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
                _tgt = self._ref_link_contact_target_local
                _nrm = -self._ref_link_contact_normal_local   # 저장 법선은 표면 바깥쪽 -> 안쪽으로
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
                      f"방향={c.cws_n_dir}  옆면={c.cws_n_edge}  여유={c.cws_beta}  mu={c.cws_mu}")
        self._lc_sides, self._link_contact_body_ids = self._find_hand_bodies(list(LINK_CONTACT_NAMES))
        # 손끝 10개(cfg.fingertip_body_names 순서)의 LINK_CONTACT_NAMES 색인.
        self._ft_distal_idx = torch.tensor(
            [LINK_CONTACT_NAMES.index(n) for n in self.cfg.fingertip_body_names], device=dev, dtype=torch.long)  # (10,)
        # per-link OUTWARD pad/palmar normal (link-local) → the link's own contact FACE. Force is projected on
        # the INWARD (-pad) direction (like the fingertip). VERIFIED on the rest-pose USD (32/32; see cfg
        # LINK_PAD_NORMALS).
        self._link_pad_normals = torch.tensor(
            [LINK_PAD_NORMALS[n] for n in LINK_CONTACT_NAMES], device=dev, dtype=torch.float32)  # (L,3) OUTWARD

        # ---- action / EMA buffers ----
        # [hand-pretrain] 손별 기본 자세를 왼손→오른손 36열로 (한계 텐서와 같은 순서)
        default_q = torch.cat(
            [self._hands[i].data.default_joint_pos[:, getattr(self, f"_finger_joint_ids_{sd}")]
             for i, sd in enumerate("lr")], dim=1)                                     # (E,36)
        self._smoothed_actions = self._unscale(default_q).clone()      # (E,65) normalized
        self._residual_target = None       # (E,36) 손가락 PD 목표 (residual_action 이면 매 스텝 설정)

        # ---- per-env trajectory frame index ----
        self._frame_idx = torch.zeros(self.num_envs, device=dev, dtype=torch.long)

        # ---- state cache + RSI (손 전용 174, 오프셋은 _CACHE_LAYOUT) ----
        # J0 는 캐시된 J1 과 짝을 맞추려고, 손목 목표 EMA 는 컨트롤러 상태라 함께 담는다.
        self._CL = self._CACHE_LAYOUT
        self._STATE_DIM = int(self._CL["dim"])
        assert self._STATE_DIM == self._CL["ctrl"][1], "레이아웃 dim 과 마지막 블록 끝이 불일치"
        # steps physics; everything it touches (reference arrays,
        # object, robot, scene) already exists by here.
        self._solve_spawn_declear()
        # 지지면을 내려 물체가 레퍼런스 높이에 안착하게 합니다.
        # 반드시 declear 뒤에 — 내릴 양을 그 결과에서 읽습니다.
        self._apply_context_z_auto()
        self._state_cache = torch.zeros(self._ref_len, self._STATE_DIM, device=dev)
        self._state_cache[:, 0] = -float("inf")                        # reward column
        self._init_flg = torch.ones(self._ref_len, device=dev, dtype=torch.bool)   # True = reference (no cache)
        self._reached_frame = 0
        # 캐시 품질 게이트 early→late 전환: 한 에피소드가 클립의
        # late_gate_survival_frac 이상을 살고 끝 3프레임 안에서 끝나야 켠다.
        # 지수 추적 가중치 = |rew_*| / exp_tracking_budget. 허용 오차는 σ 가 정한다.
        _lw = {"ee": abs(c.rew_ee_kpts),
               "hand": abs(c.rew_hand_kpts),
               "fingertip": abs(c.rew_fingertip), "obj_pos": abs(c.rew_obj_pos),
               "obj_rot": abs(c.rew_obj_rot)}
        _tot = sum(_lw.values()) or 1.0
        self._exp_w = {k: float(c.exp_tracking_budget) * v / _tot for k, v in _lw.items()}
        self._exp_s2 = {"ee": c.sigma_ee ** 2,
                        "hand": c.sigma_hand ** 2,
                        "fingertip": c.sigma_fingertip ** 2, "obj_pos": c.sigma_obj_pos ** 2,
                        "obj_rot": c.sigma_obj_rot ** 2}
        # 각 항이 읽을 오차 키. hand/obj_pos 는 보상용 z 가중 사본을 읽는다 —
        # 지수 항이 곧 보상이고, z 가중은 "보상에만" 거는 규약이기 때문이다 (게이트는 무가중).
        self._exp_key = {"ee": "ee",
                         "hand": "hand_w", "fingertip": "ft_reward",
                         "obj_pos": "obj_pos_w", "obj_rot": "obj_rot"}
        print("[exp-tracking] budget=" + f"{c.exp_tracking_budget}  " + "  ".join(
            f"{k}: w={self._exp_w[k]:.3f} s={self._exp_s2[k] ** 0.5:.3g}" for k in _lw))

        # 물체 마찰 커리큘럼(robotis_shadow_grasp_rsi 에서 가져옴): 초반엔 마찰을
        # 높여 미끄러지는 병목 프레임을 통과하는 궤적을 만들고, 학습이 진행되면 실제 값으로 조인다.
        self._friction_step_count: int = 0
        self._last_friction_mean: float = float(c.friction_max_init)
        self._last_friction_max: float = float(c.friction_max_init)
        self._late_gate = False
        self._late_gate_frames = int(round(float(c.late_gate_survival_frac) * self._ref_len))
        # 종료 시 일괄 기록용 스테이징 버퍼. 에피소드가 cache_min_episode_length
        # 이상 살고 끝나야 _flush_state_cache 가 _state_cache 에 합친다. 기능이 켜졌을 때만 할당.
        # [hand-pretrain] use_state_cache 조건 추가: 캐시가 꺼져 있으면 스테이징 버퍼를 만들지 않는다
        # (부모는 cache_min_episode_length 만 봐서 4096 env 에서 1.4 GB 를 쓰지도 않고 할당했다).
        if (int(getattr(c, "cache_min_episode_length", 0)) > 0
                and bool(getattr(c, "use_state_cache", True))):
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
        # 손 블록 순서: [0]손목 [1..4]검지 [5..8]중지 [9..12]약지 [13..16]소지 [17..20]엄지
        # 손목 프레임 인덱스를 하드코딩(20/40) 대신 상수로 (2026-09-06).
        _nb_k = 0            # [hand-pretrain] 몸통 키포인트 없음 → 손 블록이 0 에서 시작
        _nh = N_HAND_KPTS_PER_HAND                                   # 21
        self._wrist_frame_idx = [(_nb_k + o, _nb_k + o + 5, _nb_k + o + 1) for o in (0, _nh)]  # L, R
        assert self._ref_kpts.shape[1] == _nb_k + 2 * _nh, (
            f"[wrist-frame-idx] 키포인트 수 불일치: ref {self._ref_kpts.shape[1]} != "
            f"{_nb_k} + 2x{_nh}")
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


    # ------------------------------------------------- action helpers
    def _unscale(self, q: torch.Tensor) -> torch.Tensor:
        return 2.0 * (q - self._ctrl_lower) / (self._ctrl_upper - self._ctrl_lower) - 1.0

    def _scale(self, a: torch.Tensor) -> torch.Tensor:
        return self._ctrl_lower + 0.5 * (a + 1.0) * (self._ctrl_upper - self._ctrl_lower)

    # ---- action: 손목 관절 6 + 손가락 18 잔차. 48 = 손별 24 x 2(왼손 먼저) = [wrist6 | finger 18] ----
    # 손목은 palm 앞 6-DoF 관절(anchor 는 env 원점 고정)을 레퍼런스 + 잔차 위치 목표로 PD 구동한다.
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._frame_idx = (self._frame_idx + 1).clamp(max=self._ref_len - 1)
        c = self.cfg
        a = actions.clamp(-1.0, 1.0)
        # 정책 액션 사본: 관측의 prev_action 항과 action_rate 보상이 읽습니다. 이미 [-1,1] 이라
        # bnd 사본은 동일합니다(SONIC 경로의 z_res 정규화가 필요 없습니다).
        self._cur_policy_action = a
        self._cur_policy_action_bnd = a

        # ── 손가락 36 = 왼손 18 ++ 오른손 18 (관절 한계 텐서와 같은 순서) ──
        _w, _ph = self._ACT_W, self._ACT_PH
        fing = torch.cat([a[:, _w:_ph], a[:, _ph + _w:2 * _ph]], dim=-1)       # (E,36)
        al = self._finger_alpha
        self._smoothed_actions[:] = al * fing + (1.0 - al) * self._smoothed_actions
        if c.residual_action and self._ref_joints is not None:
            # ── [residual] 손가락: 레퍼런스 기준 margin 잔차 (DexMachina 본체 방식) ──────────────
            # a=+1 이면 그 관절 상한에 정확히 도달하므로 결과가 항상 한계 안이다 — 아래 clamp 는
            # 수치 안전망일 뿐 동작하지 않는다.
            _ref36 = self._ref_joints[self._frame()]                           # (E,36) 관절 단위
            _alh = self._finger_alpha
            if self._fing_ema_res:
                # 잔차 전용 EMA: 정규화 액션을 평활한 뒤 **현재 프레임**의
                # 마진으로 매핑한다. 레퍼런스는 지연 없이 통과한다. _hand_target_ema 는 여기서
                # '현재 목표' 사본이 되어 캐시 저장/복원 코드가 그대로 유효하다.
                _res = self._hand_res_ema
                _res.mul_(1.0 - _alh).add_(_alh * fing)
                _tgt = _ref36 + torch.where(_res >= 0,
                                            _res * (self._ctrl_upper - _ref36),
                                            _res * (_ref36 - self._ctrl_lower))
                self._hand_target_ema.copy_(_tgt).clamp_(self._ctrl_lower, self._ctrl_upper)
            else:
                # EMA 를 최종 목표(레퍼런스 + 잔차)에 건다. 평활 상태가 곧 목표이므로 clamp 를 상태에
                # 걸어 EMA 가 한계 밖으로 표류하지 못하게 한다.
                _raw = _ref36 + torch.where(fing >= 0,
                                            fing * (self._ctrl_upper - _ref36),
                                            fing * (_ref36 - self._ctrl_lower))
                self._hand_target_ema.mul_(1.0 - _alh).add_(_alh * _raw).clamp_(
                    self._ctrl_lower, self._ctrl_upper)
            self._residual_target = self._hand_target_ema
        else:
            # residual_action=False → 절대 액션 + EMA (_apply_action 이 _smoothed_actions 를 관절 범위로 사상)
            self._residual_target = None

        # [wrist6] 손목 관절 잔차 q_des = ref[frame] + scale·EMA(a). 평활이 핵심이다 — 잔차가 관절 응답보다 느려야
        # 추종 오차(= 힘)가 작게 유지된다. 평활을 빼면 힘이 6배로 뛴다.
        _wa = torch.cat([a[:, 0:6], a[:, _ph:_ph + 6]], dim=-1)            # (E,12)
        _fr6 = self._frame()
        _tew = self._wrist6_target_ema
        for _h in range(2):
            # 잔차는 고정 상한(±wrist6_res_pos / ±wrist6_res_rot). 손목 관절 한계는 물리
            # 범위가 아니라 안전벽이라 마진 스케일을 쓰지 않는다.
            # 잔차 전용 EMA: 레퍼런스는 지연 없이 통과, 잡음만 평활.
            if self._w6_ema_res:
                _res = self._wrist6_res_ema[_h]
                _res.mul_(1.0 - _tew).add_(_tew * _wa[:, 6 * _h:6 * (_h + 1)] * self._wrist6_res_scale)
                self._wrist6_target[_h].copy_(self._ref_wrist_dof6[_h, _fr6] + _res).clamp_(
                    self._wrist6_lo[_h], self._wrist6_hi[_h])
                continue
            _raw = (self._ref_wrist_dof6[_h, _fr6]
                    + _wa[:, 6 * _h:6 * (_h + 1)] * self._wrist6_res_scale)
            # EMA 를 최종 목표에. 평활 상태가 곧 목표다.
            self._wrist6_target[_h].mul_(1.0 - _tew).add_(_tew * _raw).clamp_(
                self._wrist6_lo[_h], self._wrist6_hi[_h])

    def _apply_action(self) -> None:
        if self._residual_target is not None:                          # residual: ref[frame] + margin·a
            target = self._residual_target                             # (E,36)
        else:                                                          # residual_action=False → 절대 EMA
            target = self._scale(self._smoothed_actions)
        for _h, _sd in enumerate("lr"):
            _hand = self._hands[_h]
            _sl = slice(0, 18) if _h == 0 else slice(18, 36)
            _hand.set_joint_position_target(
                target[:, _sl], joint_ids=getattr(self, f"_finger_joint_ids_{_sd}").tolist())
            # [wrist6] 손목은 관절 위치 목표로 구동한다. 속도 목표는 주지 않는다 —
            # 넣어도 최대 오차 5.72 vs 5.91 mm 로 차이가 없고(측정), DexMachina 도 쓰지
            # 않는다. 빼면 정책이 위치 목표 하나만 다루면 된다.
            _hand.set_joint_position_target(
                self._wrist6_target[_h],
                joint_ids=getattr(self, f"_wrist6_joint_ids_{_sd}").tolist())


    # ------------------------------------------------- robot keypoint / fingertip FK
    # [hand-pretrain] 두 손이 별개 articulation 이라 body 를 (손, id) 쌍으로 저장하고 읽을 때 합친다.
    def _hand_joint_cat(self, field: str) -> torch.Tensor:
        """(E,36) 두 손의 구동 관절값을 왼손 18 → 오른손 18 순서로 이어붙입니다.
        _ctrl_lower/_upper, 리셋의 36열 버퍼, 관측이 모두 이 순서를 전제합니다."""
        return torch.cat(
            [getattr(self._hands[i].data, field)[:, getattr(self, f"_finger_joint_ids_{sd}")]
             for i, sd in enumerate("lr")], dim=1)

    def _find_hand_bodies(self, names: list[str]):
        """이름 리스트 → (side_idx (N,), body_ids (N,)). side_idx 0=왼손 1=오른손."""
        sides, ids = [], []
        for n in names:
            h = 0 if "_l_" in n else 1
            found = self._hands[h].find_bodies(n)[0]
            if not found:
                raise ValueError(f"[hand-pretrain] {n} 을 {'왼' if h == 0 else '오른'}손에서 찾지 못함")
            sides.append(h)
            ids.append(found[0])
        return (torch.tensor(sides, device=self.device, dtype=torch.long),
                torch.tensor(ids, device=self.device, dtype=torch.long))

    def _gather_body(self, sides: torch.Tensor, ids: torch.Tensor, field: str) -> torch.Tensor:
        """(E,N,D) 두 손의 body_* 배열에서 (side, id) 로 지정된 항목만 모읍니다."""
        outs = []
        for h in (0, 1):
            m = sides == h
            if not bool(m.any()):
                continue
            outs.append((m, getattr(self._hands[h].data, field)[:, ids[m]]))
        E = self.num_envs
        D = outs[0][1].shape[-1]
        res = torch.zeros(E, sides.numel(), D, device=self.device, dtype=outs[0][1].dtype)
        for m, v in outs:
            res[:, m] = v
        return res

    def _robot_kpts_w(self) -> torch.Tensor:
        """(E,56,3) world keypoint positions = body origin + rotated local offset."""
        p = self._gather_body(self._kpt_sides, self._kpt_body_ids, "body_pos_w")   # (E,42,3)
        q = self._gather_body(self._kpt_sides, self._kpt_body_ids, "body_quat_w")  # (E,42,4)
        off = self._kpt_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        return p + math_utils.quat_apply(q, off)

    def _robot_ft_w(self):
        """(E,10,3) fingertip pad positions + (E,10,3) pad-inward world directions."""
        p = self._gather_body(self._ft_sides, self._ft_body_ids, "body_pos_w")
        q = self._gather_body(self._ft_sides, self._ft_body_ids, "body_quat_w")
        tip = p + math_utils.quat_apply(q, self._ft_offsets.unsqueeze(0).expand(self.num_envs, -1, -1))
        pad_inward = -math_utils.quat_apply(q, self._ft_pad_normals.unsqueeze(0).expand(self.num_envs, -1, -1))
        return tip, pad_inward

    def _link_pad_inward_w(self) -> torch.Tensor:
        """(E,L,3) each wrap link's INWARD pad/palmar unit normal in world = -R(link_quat)·pad_outward_local.
        The direction the object presses when the link touches with its correct (grasping) face — the analog
        of the fingertip `pad_inward`. Used both to project the contact force and for the orientation gate."""
        q = self._gather_body(self._lc_sides, self._link_contact_body_ids, "body_quat_w")  # (E,L,4)
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

        Set by _reset_idx when one episode both ran for
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
        # 손바닥(손목) 자세·속도와 손끝 속도(양손): grasp 와 같은 직접 조작 신호. 두 단계 모두 실제 로봇 값.
        palm_quat = _canon(self._gather_body(self._palm_sides, self._palm_body_ids, "body_quat_w"))
        palm_linvel = self._gather_body(self._palm_sides, self._palm_body_ids, "body_lin_vel_w")
        palm_angvel = self._gather_body(self._palm_sides, self._palm_body_ids, "body_ang_vel_w")
        ft_vel = self._gather_body(self._ft_sides, self._ft_body_ids, "body_lin_vel_w")  # (E,10,3)
        # [hand-pretrain] 몸통 항 3개(projected_gravity_b / root_lin_vel_w / root_ang_vel_w)를 뺐다. self.robot 은 왼손
        # 별칭이라 두면 왼손 자세가 골반 신호로 들어간다. 손 자세·속도는 아래 palm 블록이 양손 모두 담는다.
        A = [
            self._unscale(self._hand_joint_cat("joint_pos")),   # [hand-pretrain] (36)
            self._hand_joint_cat("joint_vel") * vs,             # (36)
            _quat_to_6d(palm_quat).reshape(E, -1),                     # palm ori 6d ×2 (12)
            palm_linvel.reshape(E, -1),                               # palm linvel ×2 (6)
            (palm_angvel * vs).reshape(E, -1),                        # palm angvel ×2 (6)
            ft_vel.reshape(E, -1),                                     # fingertip linvel ×10 (30)
        ]

        # ---- BLOCK B: reference tracking + look-ahead ----
        kpts = self._robot_kpts_w()                                    # (E,42,3) world (손 전용)
        kpts_local = kpts - org[:, None, :]                            # env-local
        delta_kpts = self._ref_kpts[nfr] - kpts_local                  # look-ahead delta
        # 위상/시간 신호 없음(grasp 처럼 다음 프레임 레퍼런스 delta 가 진행을 전한다).
        # [hand-pretrain] 골반 레퍼런스 4개는 뺐다. 손목 목표 오차는 delta_kpts 의 palm 두 항(인덱스 0, 21)이 담는다.
        B = [
            kpts_local.reshape(E, -1),                                 # (42×3=126)
            delta_kpts.reshape(E, -1),                                 # (42×3=126)
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
            # 손끝 목표 전환에도 같은 게이트 (grasp env 의 contact_flag_next 와
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
            # 보상과 같은 게이트를 씌운다 — 보상이 0 인 프레임을 관측에서
            # 구분할 수 없으면 정책이 게이트를 학습할 수 없다 (grasp env 도 obs/보상 모두
            # 게이트된 future_contact 를 썼다). 관측 차원은 그대로라 기존 체크포인트 호환.
            self._ref_link_contact_mask[nfr] * self._ref_obj_vel_gate[nfr].unsqueeze(-1),  # FUTURE per-link expected contact (L=32)
            # OBS force CLIP to force_obs_clip (user 2026-07-23, =300N): raw contact/foot forces spike
            # to ~hundreds of N (var ~7e5 measured), destabilizing the obs RunningStandardScaler. Clip
            # the OBS copy only (the reward uses its own contact_force_cap). Keeps scaling stable.
            self._link_contact_forces().clamp(max=c.force_obs_clip),  # current per-link actual contact force (L=32)
            # [hand-pretrain] 발 접촉 스케줄·실측 발 힘은 뺐다(발 없음). 배열은 계속 계산하므로 되살릴 땐 두 줄만 복원.
            # 이전 정책 액션(raw, GRAIL 방식). 클립·정규화한 사본을 넣어 봤지만(2026-07-28) 효과가 없어 되돌렸다.
            self._prev_policy_action,
        ]

        # 역방향 롤아웃(제거됨)의 진행 방향 비트가 있던 자리. 값은 항상 0 이지만 관측 차원을
        # 바꾸면 기존 체크포인트를 쓸 수 없으므로 자리를 남긴다.
        C = C + [torch.zeros(E, 1, device=self.device)]
        obs = torch.cat(A + B + C, dim=-1)
        assert obs.shape[-1] == c.observation_space, (
            f"obs dim {obs.shape[-1]} != cfg.observation_space {c.observation_space} "
            "(block-C dims must be invariant across the has_object flip)")
        # capture prev actions for NEXT step (lag-1). The RAW copy (_*_policy_action) feeds the obs
        # (above) and the drift diagnostics (Diag/zres_absmax, Diag/hand_absmax); the BOUNDED copy
        # (_*_bnd) feeds action_rate only. Both are maintained so either can be swapped in.
        self._prev_policy_action = self._cur_policy_action.clone()
        self._prev_policy_action_bnd = self._cur_policy_action_bnd.clone()


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


    # 손목 회전: 사람 손과 로봇 손에 같은 키포인트 기하로 좌표계를 세워 비교한다.
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
        # [hand-pretrain] 손목 오차 = 양손 palm 2개(떠 있는 손목)의 평균.
        palm_per = dk[:, self._palm_kpt_idx].norm(dim=-1)            # (E,2) palm 거리
        wrist_pos_err = palm_per[:, self._wrist_kpt_idx].mean(dim=-1)  # (E,) 종료 게이트 + 로그
        ee_err = palm_per[:, self._ee_kpt_idx].mean(dim=-1)            # (E,) rew_ee_kpts
        hand_err = dk.norm(dim=-1).mean(dim=-1)                     # (E,) 손 42 kpt — 무가중 (로그)
        # 보상용 z 가중 사본. 무가중은 로그/게이트에 그대로 남깁니다.
        _zw = float(self.cfg.z_weight_reward)
        _dkh = dk.clone()
        _dkh[..., 2] *= _zw
        hand_err_w = _dkh.norm(dim=-1).mean(dim=-1)                 # (E,) 40 hand kpts — 가중 (보상)
        # fingertip pad tracking (contact-conditioned handled in reward; raw here)
        tip, pad_inward = self._robot_ft_w()
        ft_per = (self._ref_ft_pad[fr] - (tip - org[:, None, :])).norm(dim=-1)   # (E,10) = L[5] then R[5]
        # UNIFORM MEAN over all 10 fingertip pads (both hands) — consistent with body / wrist_pos / wrist_rot,
        # all of which use a plain mean for the termination gate (no per-hand worst-of-two-hands max).
        ft_err = ft_per.mean(dim=-1)                                            # (E,)
        # object
        if self._has_object:
            obj_pos = self._object.data.root_pos_w - org
            obj_quat = _canon(self._object.data.root_quat_w)
            _dop = self._ref_obj_pos[fr] - obj_pos                              # (E,3)
            obj_pos_err = _dop.norm(dim=-1)                                      # 무가중 (게이트/로그)
            _dopw = _dop.clone(); _dopw[:, 2] *= _zw
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
            # 관측(delta_ft_obj)과 같은 게이트 — 둘이 어긋나면 정책이 보는 목표와
            # 보상이 재는 목표가 달라진다. grasp env 도 contact_flag_gated 하나로 둘을 함께 걸었다.
            in_contact = (self._ref_link_contact_mask[fr][:, self._ft_distal_idx]
                          * self._ref_obj_vel_gate[fr].unsqueeze(-1)).unsqueeze(-1).bool()  # (E,10,1)
            ft_target = torch.where(in_contact, ref_vtx_w, ref_ft_drift)          # (E,10,3)
            ft_reward = (ft_target - tip_l).norm(dim=-1).mean(dim=-1)             # (E,) contact-conditioned
        else:
            obj_pos_err = torch.zeros(self.num_envs, device=self.device)
            obj_pos_err_w = obj_pos_err
            obj_rot_err = torch.zeros(self.num_envs, device=self.device)
            ft_reward = ft_err                                                     # no object → raw pad target
        # 손목 회전 오차(양손 평균). 키포인트 기하로 세운 좌표계끼리 비교한다.
        wrist_rot_err = self._wrist_rot_err(ref, kpts)
        # `wrist` (reward) and `wrist_pos` (termination) are the SAME tensor, kept under both
        # names so each call site reads the one that matches its intent.
        return dict(ee=ee_err, hand_w=hand_err_w, obj_pos_w=obj_pos_err_w,
                    wrist_pos=wrist_pos_err, hand=hand_err, ft=ft_err, ft_reward=ft_reward, ft_per=ft_per, tip=tip,
                    pad_inward=pad_inward, obj_pos=obj_pos_err, obj_rot=obj_rot_err,
                    wrist_rot=wrist_rot_err)

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        e = self._errs                                    # set by _get_dones (runs first each step)
        fr = self._frame()

        # 링크별 접촉력(DexMachina): 32 wrap 링크의 압축 접촉력을 (a) 레퍼런스 접촉 mask 와 (b) 링크가 물체 표면
        # 목표점 근처일 때만 인정한다(엉뚱한 곳을 누르면 보상 없음). 활성 링크에 대해 [0,1] 로 정규화.
        link_force = self._link_contact_forces()                      # (E,L) compressive per link (on own face)
        # 물체가 정지한 프레임에서는 link_mask 가 전부 0 이 되어
        # 분자(lf)와 분모(n_lc)가 함께 사라진다 → force_rew = 0. grasp env 와 같은 처리다
        # (fforce_contact / n_contacts 를 동일한 게이트 플래그로 걸었다).
        link_mask = self._ref_link_contact_mask[fr] * self._ref_obj_vel_gate[fr].unsqueeze(-1)   # (E,L)
        if self._has_object and self._has_link_contact:
            oqL = self._object.data.root_quat_w[:, None, :].expand(-1, N_LINK_CONTACT, -1)   # (E,L,4) live
            tgt_w = (math_utils.quat_apply(oqL, self._ref_link_contact_target_local[fr])
                     + self._object.data.root_pos_w[:, None, :])                              # (E,L,3) world target
            lp = self._gather_body(self._lc_sides, self._link_contact_body_ids,
                                    "body_pos_w")                                     # [hand-pretrain] (E,L,3)
            near = ((lp - tgt_w).norm(dim=-1) < c.contact_match_dist).float()                 # (E,L) spatial gate
        else:
            near = torch.zeros_like(link_mask)
        lf = (link_force * link_mask * near).clamp(min=0.0, max=c.contact_force_cap)  # force·mask·near
        n_lc = link_mask.sum(dim=-1).clamp(min=1.0)                   # #active links (≥1 to avoid /0)
        force_rew = lf.sum(dim=-1) / (n_lc * c.contact_force_cap)     # (E,) mean of min(force,cap)/cap ∈ [0,1]

        # 로봇 쪽 접촉 목록과 점수. 위치·법선은 센서 값(contact_pos_w, force_matrix_w)을
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
            # 물체 기준 좌표로 옮긴다(현재 자세). 기준은 body 원점이 아니라 물리 COM
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
            # 텐서보드에 남길 per-env 점수. 접촉 링크 수도
            # 같이 봐야 "렌치가 부족한 것"과 "애초에 안 닿은 것"을 구분할 수 있습니다.
            self._cws_per_env = cws_rew
            self._cws_nhit = _hit.float().sum(dim=-1)
            # 원 점수(cws_rew)는 v 보정에 민감해 실측상 96%가 정확히 0이었습니다. 커버리지/부족분은
            # v 와 무관하고 방향 수에도 스케일되지 않아 실패들 사이를 구분할 수 있습니다.
            self._cws_cov = CWS.cws_coverage(self._cws_sigma_h[fr], sig_r, c.cws_beta)
            self._cws_def = CWS.cws_deficit(self._cws_sigma_h[fr], sig_r, c.cws_beta)

        # 관절 물체 DOF 보상 항은 없다(물체는 항상 단일 RigidObject). 관절 물체를 실제로 스폰하면 여기 다시 추가.

        # 루트 선속도·각속도 오차 항은 뺐다. 위치·자세 추적과 중복이고 리셋 직후 각속도 잡음이 벌점을 지배했다.

        # action_reg = 정책 액션(손목 + 손가락)의 제곱합(grasp 관행).
        action_reg = (self._cur_policy_action ** 2).sum(-1)                             # (E,) SUM
        # pose_reg_hands: 손 관절을 기본(중립) 자세 쪽으로 당긴다. 레퍼런스 쪽이 아닌 이유는 손 추적은 이미
        # kpt·손끝 항이 맡기 때문이다. [hand-pretrain] 손 36개 전체가 대상.
        hand_ref = self._hand_joint_cat("default_joint_pos")
        pose_reg_hands = ((self._hand_joint_cat("joint_pos") - hand_ref) ** 2).sum(-1)
        # action_rate: raw 정책 액션의 스텝 간 변화 제곱합(GRAIL meta_action_rate_l2). 실제 관절 목표가 아니므로
        # SONIC 자체의 추종은 벌하지 않는다. raw 라 상한이 없으니 Diag / zres_clip_frac 와 함께 볼 것.
        action_rate = ((self._cur_policy_action - self._prev_policy_action) ** 2).sum(-1)

        alive = (~self._died).float()                     # _died set by _get_dones this step

        # 추적 항은 항마다 w·exp(-err²/σ²) ∈ [0, w] (형태·σ 는 SONIC rewards.py:442).
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
        ).clamp(min=0.0)
        # 비유한 보상은 0 으로 바꾼다. NaN env 는 _get_rewards 뒤에 리셋되므로 그 스텝 보상이 GAE → PPO 를 오염시킨다.
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        self._save_state_cache(reward)                    # per-frame best-state RSI cache
        # 항별 보상 기여(가중치 적용) → Episode_Reward (항마다 그래프 하나). 추적 항은 클램프 전 값이다.
        # 클램프된 그룹 값과 clamp_frac 은 Diag /, 총합은 skrl 의 Reward / Instantaneous reward 와 같아 따로 안 남긴다.
        _track_rew = {"ee_kpts": self._exp_terms["ee"],
                      "hand_kpts": self._exp_terms["hand"], "fingertip": self._exp_terms["fingertip"],
                      "obj_pos": self._exp_terms["obj_pos"], "obj_rot": self._exp_terms["obj_rot"]}
        ep_rew = {
            "alive": _alive_w * alive,
            **_track_rew,
            "contact_force": c.rew_contact_force * force_rew,         # per-link (Option A) contact-force reward
            # Episode_Reward 그룹은 "실제로 보상에 들어간 값"만 담아야
            # 합니다. 진단 전용일 때 여기에 값을 흘리면 보상에 포함된 것처럼 보입니다. 원 점수는
            # 아래 Diag / cws_score 로 나갑니다.
            "contact_cws": (c.rew_cws * cws_rew if c.contact_reward_mode in ('cws', 'both')
                            else torch.zeros_like(cws_rew)),
            "action_reg": c.rew_action_reg * action_reg,  # 잠재+손 SUM
            "pose_reg_hands": c.rew_pose_reg_hands * pose_reg_hands,
            "action_rate": c.rew_action_rate * action_rate,
        }
        self._log_reward_terms(e, tracking_penalty, ep_rew, fr)
        return reward

    def _log_reward_terms(self, e, tracking_penalty, ep_rew, fr):
        log = self.extras.setdefault("log", {})
        log.update({
            "Error / wrist_kpts": e["wrist_pos"].mean(),   # 종료 게이트 대상 (보상에서는 ee 에 흡수)
            "Error / ee_kpts": e["ee"].mean(),
            "Error / hand_kpts": e["hand"].mean(),
            "Error / fingertip": e["ft"].mean(), "Error / obj_pos": e["obj_pos"].mean(),
            "Error / wrist_rot": e["wrist_rot"].mean(),
            "Curriculum / reached_frame": float(self._reached_frame),
            "Curriculum / late_gate": float(self._late_gate),
            "Curriculum / friction_max": float(self._last_friction_max),
            "Curriculum / friction_mean": float(self._last_friction_mean),
            "Curriculum / ref_start_frac": float(getattr(self, "_diag_ref_start", 0.0)),
            "Curriculum / cache_coverage": float((~self._init_flg).sum().item()) / self._ref_len,
            # [hand-pretrain] 유한성/크기 검사로 캐시 쓰기가 거부된 env 비율. 0 이 정상이고,
            # 0 이 아니면 떠 있는 손이 수치적으로 발산하고 있다는 뜻입니다.
            "Diag / cache_reject_frac": float(getattr(self, "_diag_cache_reject", 0.0)),
        })
        # 항별 보상 → Episode_Reward (항마다 그래프 하나. 합계·진단은 넣지 않는다).
        for k, v in ep_rew.items():
            log[f"Episode_Reward / {k}"] = v.mean()
        # reward-shaping diagnostics → separate "Diag /" tab (cfg.log_reward_diag=False drops them).
        # The 9 tracking terms above are logged PRE-clamp, so once the clamp bites their sum no longer
        # equals what entered the reward — these three are the only window on that gap.
        if self.cfg.log_reward_diag:
            log["Diag / tracking_penalty"] = tracking_penalty.mean()
            # 접촉 렌치 진단. cws_score 는 "지금 파지가 만들 수 있는
            # 렌치가 레퍼런스가 요구하는 렌치를 얼마나 담는가" (1에 가까울수록 충분). cws_nhit 은
            # 실제로 힘을 받고 있는 링크 수 — 점수가 낮을 때 "렌치 부족"인지 "안 닿음"인지 가릅니다.
            if getattr(self, "_cws_per_env", None) is not None:
                log["Diag / cws_score"] = self._cws_per_env.mean()     # 원 점수 (v 보정 확인용)
                log["Diag / cws_coverage"] = self._cws_cov.mean()      # 요구 방향 중 충족 비율 ∈[0,1]
                log["Diag / cws_deficit"] = self._cws_def.mean()       # 방향당 평균 부족분
                log["Diag / cws_nhit"] = self._cws_nhit.mean()
            # 포화도: term/w ∈ [0,1]. 1 에 가까우면 σ 가 느슨해 항이
            # 상수라 아무것도 안 가르치고, 0 에 가까우면 너무 조여 항이 죽는다. 0.3~0.7 이 실제로
            # 학습을 이끄는 대역이고, σ 는 이 로그로 튜닝한다.
            for _k, _v in self._exp_terms.items():
                log[f"Sat / {_k}"] = _v.mean() / max(self._exp_w[_k], 1e-9)
        # RSI / 에피소드 길이 진단: 길이가 가변이라 skrl 의 Total reward(에피소드 합)는 설정 간 비교가 안 된다.
        # episode_len = 현재 에피소드 경과 스텝, rsi_start = 시작 프레임.
        log["Diag / episode_len_mean"] = self.episode_length_buf.float().mean()
        log["Diag / rsi_start_mean"] = self._episode_start_frame.float().mean()
        if self.cfg.contact_vel_gate:
            log["Diag / vel_gate_frac"] = self._ref_obj_vel_gate[self._frame_idx].mean()
        log["Diag / death_frac"] = self._died.float().mean()
        # raw 액션 크기: action_rate 가 보지 않으므로 평균 random-walk 를 보는 유일한 창이다.
        # [hand-pretrain] 액션 블록이 손별 (손목 + 손가락)이라 손목·손가락 포화를 따로 본다.
        _a = self._cur_policy_action
        # 손별 폭 = 손목 6 + 손가락 18.
        _w, _ph = self._ACT_W, self._ACT_PH
        _wr = torch.cat([_a[:, 0:_w], _a[:, _ph:_ph + _w]], dim=-1)            # 손목 12
        _fg = torch.cat([_a[:, _w:_ph], _a[:, _ph + _w:2 * _ph]], dim=-1)      # 손가락 36
        log["Diag / wrist_absmax"] = _wr.abs().max()
        log["Diag / hand_absmax"] = _fg.abs().max()
        # per-frame-bucketed tracking error — disambiguates reward-balance from curriculum-mix:
        # if per-bucket error stays flat/falls while the GLOBAL mean rises, the rise is a
        # frame-distribution shift (harder later frames entered the mix), not per-frame regression.
        T = max(1, self._ref_len - 1)
        q = (fr.float() / T * 4.0).clamp(0, 3).long()               # frame quartile per env
        for b in range(4):
            m = q == b
            if bool(m.any()):
                log[f"Error / wrist_q{b}"] = e["wrist_pos"][m].mean()
        # DIAG(term-cause): per-gate share of THIS step's deaths (gates OR → shares can overlap) + obj_rot err.
        log["Error / obj_rot"] = e["obj_rot"].mean()
        dead = self._died
        nd = dead.float().sum()
        if nd > 0:
            cc = self.cfg
            _cause = {"wrist_pos": e["wrist_pos"] > cc.term_wrist_pos_err, "ft": e["ft"] > cc.term_ft_err,
                      "obj_pos": e["obj_pos"] > cc.term_obj_pos_err, "obj_rot": e["obj_rot"] > cc.term_obj_rot_err,
                      "wrist_rot": e["wrist_rot"] > cc.term_wrist_rot_err}
            for k, mk in _cause.items():
                log[f"Term / {k}"] = (mk & dead).float().sum() / nd

    # ------------------------------------------------------------------ dones
    def _dones_deviation(self, e) -> torch.Tensor:
        """레퍼런스 이탈 종료: 손끝 / 손목 위치 / 손목 회전 / 물체 (grasp 와 같은 게이트)."""
        c = self.cfg
        # 손끝(10 pad 평균)과 손목 위치(양손 평균) 이탈 종료. 손가락 체인 kpt 평균은 보상 전용이다(grasp 와 같음).
        d = (e["ft"] > c.term_ft_err) | (e["wrist_pos"] > c.term_wrist_pos_err)
        # 손목 회전 이탈 종료.
        # 리타게팅 로봇 자신도 16클립 프레임의 0.2% 가 임계(0.75, 사용자 결정)를 넘어 그 프레임은 허위 종료된다.
        d = d | (e["wrist_rot"] > c.term_wrist_rot_err)
        #  object (mirrors grasp obj_pos + obj_rot) — only when an active object is present.
        if self._has_object:
            d = d | (e["obj_pos"] > c.term_obj_pos_err) | (e["obj_rot"] > c.term_obj_rot_err)
        if not c.termination:
            d = torch.zeros_like(d)
        # 비유한 상태 차단(종료를 꺼도 항상): NaN 은 `err > thresh` 게이트에 안 걸리므로 루트·관절 상태가 비유한인
        # env 를 강제 리셋한다(inf 는 위 게이트가 잡는다). [hand-pretrain] 두 손 모두 검사한다.
        nonfinite = torch.zeros_like(d)
        for _hh in self._hands:
            rd = _hh.data
            nonfinite = nonfinite | (
                ~torch.isfinite(rd.root_pos_w).all(-1)
                | ~torch.isfinite(rd.root_quat_w).all(-1)
                | ~torch.isfinite(rd.joint_pos).all(-1)
                | ~torch.isfinite(rd.joint_vel).all(-1)
            )
        return d | nonfinite

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # runs BEFORE _get_rewards each step → compute + cache errors here for both.
        self._errs = self._compute_errors()
        self._died = self._dones_deviation(self._errs)
        # 시간 초과(truncated, 실패 아님): 실패 EMA 에서 빠지고 skrl 이 bootstrap 한다.
        # 첫 항 = 레퍼런스 끝(env 별 프레임), 둘째 항 = 안전 상한. bootstrap 값은 자동 리셋 뒤 새 시작 상태의 V 다.
        time_out = (self._frame_idx >= self._ref_len - 1) | (self.episode_length_buf >= self.max_episode_length - 1)
        return self._died, time_out

    # 물체 마찰 커리큘럼
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


    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids) -> None:
        # super 전에 실행해야 한다(super 가 episode_length_buf 를 0 으로 만든다).
        # getattr 인 이유: DirectRLEnv.__init__ 의 첫 리셋은 스테이징 버퍼 할당 전이다.
        _ep_len = self.episode_length_buf[env_ids].clone()
        if getattr(self, "_pend_state", None) is not None:
            self._flush_state_cache(env_ids, _ep_len)
        # `_frame_idx` still holds the frame the ending episode stopped
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
        self._friction_step_count += 1  # 감쇠 진행
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
        # adaptive_sampling=False(rollout/play)면 상한 0 → 항상 frame 0 에서 시작해 클립 전체를 돈다.
        # 되감기 길이를 무작위로 뽑는다. 고정 되감기는 시작 분포를 실패 분포의
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
        # 에피소드마다 물체 마찰 재추출
        self._apply_object_friction(env_ids)
        self._frame_idx[env_ids] = start
        self._episode_start_frame[env_ids] = start                 # diagnostics (Diag / rsi_start_mean)
        # reset the tracking-quality streak for the reset envs (grasp mechanism)
        self._enough_continued[env_ids] = True
        self._enough_idx[env_ids] = start
        # [hand-pretrain] 정책 액션 버퍼 초기화를 SONIC 가드 밖으로 뺐다. 가드 안이면 손 env 에서는 실행되지 않아
        # 첫 관측과 action_rate 가 이전 에피소드의 액션을 들고 시작한다.
        self._cur_policy_action[env_ids] = 0.0        # prev-action obs AND action_rate = 0 at episode start
        self._prev_policy_action[env_ids] = 0.0
        self._cur_policy_action_bnd[env_ids] = 0.0    # (currently-unread A/B copy — kept in sync)
        self._prev_policy_action_bnd[env_ids] = 0.0

        # ---- restore state (train cache hit → reference+default) ----
        # ── [hand-pretrain] 리셋 버퍼는 액션 관절 36열(손당 18, 왼손→오른손)이고, 쓰기 직전에 손별 DOF 벡터
        # (22 = 구동 18 + 텐던 J0 4)로 흩뿌린다. aid 는 36열 안의 위치.
        _nA = 36
        jpos = torch.zeros(len(env_ids), _nA, device=self.device)
        jvel = torch.zeros_like(jpos)
        org = self.scene.env_origins[env_ids]
        aid = torch.arange(_nA, device=self.device)      # [hand-pretrain] 36열 내 위치
        # [wrist6] 손목 상태 = 손별 6-DoF 관절값. 세 리셋 경로가 각자 채운 뒤 맨 아래 쓰기 블록이 한 번에 쓴다.
        wr_dof6 = torch.zeros(2, len(env_ids), 6, device=dev)
        wr_dof6_vel = torch.zeros(2, len(env_ids), 6, device=dev)
        # 텐던 축 J0 8열 (왼 4 -> 오른 4). 레퍼런스 경로는 _ref_j0 를, 캐시 경로는 캐시 값을
        # 씁니다. _use_ref_j0 가 True 인 env 만 아래에서 레퍼런스 J0 로 덮습니다.
        j0pos = torch.zeros(len(env_ids), 8, device=dev)
        j0vel = torch.zeros(len(env_ids), 8, device=dev)
        _use_ref_j0 = torch.ones(len(env_ids), dtype=torch.bool, device=dev)

        # env 별 2갈래 소스 선택(train 캐시 > 레퍼런스+기본값), 불리언 마스크 + 2D 인덱싱으로 벡터화.
        # [hand-pretrain] aid 는 위의 36열 기준이다(부모의 _action_joint_ids_t 재정의는 제거, 두면 브로드캐스트 오류).
        train_hit = ~self._init_flg[start]                           # (n,)
        # 낮은 확률로 캐시를 무시하고 레퍼런스에서 시작한다. 한 번 쓰인 나쁜
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

        if where_train.any():                                        # [hand-pretrain] 174 레이아웃
            idx = where_train.nonzero(as_tuple=True)[0]
            s = self._state_cache[start[idx]]
            L = self._CL
            for _h in range(2):
                _a, _b = L["wristL"] if _h == 0 else L["wristR"]
                # [wrist6] 관절값을 그대로. env-local 오프셋을 더하지 않는다 — anchor 가
                # env 원점에 있고 관절값은 그 기준이다. 속도도 캐시에서 복원한다.
                wr_dof6[_h, idx] = s[:, _a:_a + 6]
                wr_dof6_vel[_h, idx] = s[:, _a + 6:_b]
            jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, L["jpos"][0]:L["jpos"][1]]
            jvel[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, L["jvel"][0]:L["jvel"][1]]
            j0pos[idx] = s[:, L["j0pos"][0]:L["j0pos"][1]]
            j0vel[idx] = s[:, L["j0vel"][0]:L["j0vel"][1]]
            _use_ref_j0[idx] = False              # 캐시된 J1 과 짝이 맞는 J0 를 그대로 씁니다
            _sm = s[:, L["smoothed"][0]:L["smoothed"][1]]
            if self.cfg.residual_action:
                self._hand_target_ema[env_ids[idx]] = _sm                  # 관절 단위
                self._smoothed_actions[env_ids[idx]] = self._unscale(_sm)  # 정규화 사본도 맞춘다
            else:
                self._smoothed_actions[env_ids[idx]] = _sm
            _ca, _cb = L["ctrl"]
            self._wrist6_target[0][env_ids[idx]] = s[:, _ca:_ca + 6]
            self._wrist6_target[1][env_ids[idx]] = s[:, _ca + 6:_cb]
        if where_ref.any():                                          # reference root + default/retargeted joints
            idx = where_ref.nonzero(as_tuple=True)[0]
            fr = start[idx]
            if self._ref_joints is not None:
                jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = self._ref_joints[fr]
                # 관절 속도도 루트 속도와 같은 프레임 fr 의 레퍼런스로 채운다.
                # 스위치는 배율(_ref_jvel_scale)에 접어 두었다: 꺼지면 0 을 대입해 기존 동작과 비트 단위로 같다.
                jvel[idx.unsqueeze(1), aid.unsqueeze(0)] = torch.clamp(
                    self._ref_joint_vel[fr] * self._ref_jvel_scale,
                    -self._ref_jvel_clip, self._ref_jvel_clip)          # (1,65) 성분별 상한, 브로드캐스트
            self._smoothed_actions[env_ids[idx]] = self._unscale(jpos[idx][:, aid])
            # [residual] 손가락 목표 EMA 를 리셋 자세로 씨딩한다. 0 에서 시작하면 초기 몇 스텝이
            # 관절 원점에서 램프업하며 레퍼런스와 크게 벌어진다 (EMA 시간상수 = 1/alpha 스텝).
            self._hand_target_ema[env_ids[idx]] = jpos[idx][:, aid]
            # [wrist6] 손목은 리타게팅 관절값, 속도 0. anchor 가 env 원점에 박혀 있어 org 를 더하지 않는다.
            # 목표 EMA 도 아래 손별 쓰기에서 이 값으로 씨딩된다.
            for _h in range(2):
                wr_dof6[_h, idx] = self._ref_wrist_dof6[_h, fr]


        # [hand-pretrain] 일반 텐던-리셋 블록은 뺐다. _tendon_j{0,1}_ids 는
        # 손 articulation(22 DOF) 기준이라 36열 버퍼에 못 쓰고, J0 는 아래 손별 쓰기가 _ref_j0 / _ref_j0_vel 로 채운다.

        # ── [hand-pretrain] 손별 쓰기. 루트 = 손목(캐시 히트면 방문했던 상태, 레퍼런스면 리타게팅 pose + 속도 0).
        # 관절: 36열을 손별 18개로 흩뿌리고, J0 는 레퍼런스면 _ref_j0, 캐시면 캐시의 J0 (섞으면 텐던 제약이 깨진다).
        _fr_reset = start                                      # (n,) 각 env 의 시작 프레임
        for _h, _sd in enumerate("lr"):
            _hand = self._hands[_h]
            # [wrist6] anchor 를 env 원점에 명시적으로 쓴다: (tx,ty,tz) 가 env-local 좌표가 되고,
            # _spawn_declear 가 +5 m 로 park 한 로봇이 제자리로 돌아온다(안 쓰면 5 m 위에 남는다).
            _anchor = torch.zeros(len(env_ids), 7, device=dev)
            _anchor[:, :3] = self.scene.env_origins[env_ids]
            _anchor[:, 3] = 1.0
            _hand.write_root_pose_to_sim(_anchor, env_ids=env_ids)
            _hand.write_root_velocity_to_sim(
                torch.zeros(len(env_ids), 6, device=dev), env_ids=env_ids)
            _jp = _hand.data.default_joint_pos[env_ids].clone()          # (n,28)
            _jv = torch.zeros_like(_jp)
            _w6id = getattr(self, f"_wrist6_joint_ids_{_sd}")
            _jp[:, _w6id] = wr_dof6[_h]
            # 속도: 레퍼런스 경로는 0(에피소드 시작), 캐시 경로는 담아둔 값. 아래 _jv 에
            # 실어 write_joint_state_to_sim 이 한 번에 쓰게 한다 — 위치만 맞추고 속도를
            # 0 으로 두면 캐시 복원이 실제로 방문했던 상태와 달라진다.
            _jv[:, _w6id] = wr_dof6_vel[_h]
            self._wrist6_target[_h][env_ids] = wr_dof6[_h]
            # 잔차 EMA 상태. 레퍼런스 리셋은 0, 캐시 히트는 (복원된
            # 관절값 − 그 프레임의 레퍼런스) — 위 줄이 목표를 관절값으로 두는 것과 같은 뜻이다.
            self._wrist6_res_ema[_h][env_ids] = wr_dof6[_h] - self._ref_wrist_dof6[_h, _fr_reset]
            _sl = slice(0, 18) if _h == 0 else slice(18, 36)
            _fid = getattr(self, f"_finger_joint_ids_{_sd}")
            _jp[:, _fid] = jpos[:, _sl]
            _jv[:, _fid] = jvel[:, _sl]
            if isinstance(getattr(self, "_ref_j0_ids", None), dict):
                _j0sl = slice(0, 4) if _h == 0 else slice(4, 8)
                _j0id = self._ref_j0_ids[_sd]
                _j1id = self._tendon_j1_ids[_sd]
                # 캐시 경로: 담아둔 J0 그대로. 이미 물리로 정착된 상태라 제약을 만족합니다.
                _q0 = j0pos[:, _j0sl].clone()
                _v0 = j0vel[:, _j0sl].clone()
                # 레퍼런스 경로: 리타게팅이 푼 J0. Shadow 문서의 q_J0 <= q_J1 을 min() 으로
                # 보장합니다 (제약 위반 상태로 리셋하면 말단이 1.7 m/s 로 튑니다 — 실측).
                if getattr(self, "_ref_j0", None) is not None:
                    _r0 = torch.minimum(self._ref_j0[_fr_reset][:, _j0sl], _jp[:, _j1id])
                    _rv = torch.zeros_like(_v0)
                    if getattr(self, "_ref_j0_vel", None) is not None:
                        _rv = torch.clamp(
                            self._ref_j0_vel[_fr_reset][:, _j0sl] * self._ref_jvel_scale,
                            -float(c.ref_reset_joint_vel_clip_hands),
                            float(c.ref_reset_joint_vel_clip_hands))
                    _m = _use_ref_j0.unsqueeze(-1)                       # (n,1)
                    _q0 = torch.where(_m, _r0, _q0)
                    _v0 = torch.where(_m, _rv, _v0)
                _jp[:, _j0id] = _q0
                _jv[:, _j0id] = _v0
            _hand.write_joint_state_to_sim(_jp, _jv, env_ids=env_ids)

        # 손가락 잔차 EMA 상태를 (복원된 목표 − 그 프레임의 레퍼런스) 의
        # 역매핑으로 씨딩한다. 레퍼런스 리셋은 목표 = 리셋 자세 = 레퍼런스라 0, 캐시 히트는 담아둔
        # 목표(_hand_target_ema, 관절 단위)를 정규화 잔차로 되돌린다 — 손목 _wrist6_res_ema 와 같은 뜻.
        if self._fing_ema_res and self._ref_joints is not None:
            _r36 = self._ref_joints[_fr_reset]                                  # (n,36)
            _d36 = self._hand_target_ema[env_ids] - _r36
            _res0 = torch.where(_d36 >= 0,
                                _d36 / (self._ctrl_upper - _r36).clamp_min(1e-6),
                                _d36 / (_r36 - self._ctrl_lower).clamp_min(1e-6))
            self._hand_res_ema[env_ids] = _res0.clamp_(-1.0, 1.0)

        # 물체 복원: pretrain·레퍼런스 리셋은 레퍼런스 자세, train 캐시 히트는 캐시(_CL["obj"])에서 복원해
        # 캐시된 로봇과 물리적으로 맞는 짝을 유지한다(안 그러면 시작 직후 종료될 수 있다).
        if self._has_object:
            f0 = start
            ref_op = self._ref_obj_pos[f0] + org                    # (n,3) world
            # 스폰 위치만 받침 위로 올린다(레퍼런스 정지 프레임만). _ref_obj_pos 는
            # 그대로라 보상·관측은 GT 이고, 캐시 히트는 이미 시뮬된 자세라 이 보정을 쓰지 않는다.
            _lift = getattr(self, "_obj_spawn_lift", None)
            if _lift is not None:
                ref_op = ref_op.clone()
                ref_op[:, 2] = ref_op[:, 2] + _lift[f0]
            ref_oq = self._ref_obj_quat[f0]                         # (n,4)
            op = torch.zeros(n, 7, device=dev)
            op[:, :3] = ref_op; op[:, 3:7] = ref_oq
            # reference path: seed the object at its REFERENCE velocity for the sampled frame (mid-motion
            # starts place the object moving, not at rest). Cache-hit envs overwrite from the cache below.
            ovel = torch.zeros(n, 6, device=dev)
            ovel[:, :3] = self._ref_obj_linvel[f0]
            ovel[:, 3:6] = self._ref_obj_angvel[f0]
            if where_train.any():
                sc = self._state_cache[start]                        # [hand-pretrain] (n,174)
                tw = where_train.unsqueeze(-1)
                # 물체 블록 열 위치를 레이아웃(_CL["obj"])에서 읽는다
                # (지금 25:38. 예전 wrench 레이아웃의 27:40 을 고정해 쓰면 물체가 엉뚱한 자세로 놓였다).
                _oa, _ob = self._CL["obj"]
                op[:, :3] = torch.where(tw, sc[:, _oa:_oa + 3] + org, ref_op)
                op[:, 3:7] = torch.where(tw, sc[:, _oa + 3:_oa + 7], ref_oq)
                ovel = torch.where(tw, sc[:, _oa + 7:_ob], ovel)
            self._object.write_root_pose_to_sim(op, env_ids=env_ids)
            self._object.write_root_velocity_to_sim(ovel, env_ids=env_ids)


    # -------------------------------------------------------- state cache write
    def _save_state_cache(self, reward: torch.Tensor) -> None:
        # [hand-pretrain] use_state_cache=False 면 쓰지 않고, 리셋은 항상 레퍼런스 경로를 탑니다
        # (모든 프레임이 리타게팅 손목 pose 에서 복원 가능하므로 캐시 없이도 RSI 는 돕니다).
        if not getattr(self.cfg, "use_state_cache", True):
            return
        """Store per-frame best (highest-reward) hand state into the 174-D train cache.

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

        Vectorized: build the full (E,174) state once, then scatter the highest-reward env
        into each UNIQUE frame it covers (loop is O(unique frames) << O(num_envs))."""
        if not hasattr(self, "_errs"):
            return
        c = self.cfg
        e = self._errs
        org = self.scene.env_origins
        fr = self._frame().clamp(max=self._ref_len - 1)                     # (E,)
        gate = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        # ── [hand-pretrain] 발산 상태 차단: 떠 있는 손은 수치적으로 터질 수 있고(관절값 2e16 실측), 캐시는 프레임당
        # 최고 보상 1행만 남겨 한 번 오염되면 영구적이다. 그래서 품질 게이트와 별도로 유한성·크기를 검사한다.
        _fin = torch.ones_like(gate)
        for _hh in self._hands:
            _hd = _hh.data
            _fin = _fin & (
                torch.isfinite(_hd.root_pos_w).all(-1)
                & torch.isfinite(_hd.root_quat_w).all(-1)
                & torch.isfinite(_hd.root_lin_vel_w).all(-1)
                & torch.isfinite(_hd.joint_pos).all(-1)
                & torch.isfinite(_hd.joint_vel).all(-1)
                # 손 관절은 |q| <= ~1.6 rad, 속도 한계 15 rad/s. 넉넉히 3배로 잡습니다 —
                # 정상 상태는 절대 넘지 않고 발산은 항상 넘습니다.
                & (_hd.joint_pos.abs().amax(-1) < 3.0 * math.pi)
                & (_hd.joint_vel.abs().amax(-1) < 1.0e3)
                & (_hd.root_lin_vel_w.norm(dim=-1) < 1.0e2)
            )
        if self._has_object:
            _od = self._object.data
            _fin = _fin & torch.isfinite(_od.root_pos_w).all(-1) \
                & torch.isfinite(_od.root_quat_w).all(-1) \
                & torch.isfinite(_od.root_lin_vel_w).all(-1)
        self._diag_cache_reject = float((~_fin).float().mean())
        gate = gate & _fin
        # ── [/hand-pretrain] ─────────────────────────────────────────────────────────────

        # ---- 추적 품질 게이트(grasp 방식): 리셋 이후 계속 기준 안이어야 캐시 대상. 손끝 + 물체 3단계 임계
        # (시작 직후 / early / late). 물체가 없으면 물체 단계는 항상 통과. ----
        action_fps = round(1.0 / (c.sim.dt * c.decimation))
        start_cutoff = action_fps * 2 // 3                                  # first 2/3 s = 33 frames @50Hz
        reached_end = self.is_reached_end                                   # python bool
        op, orr = e["obj_pos"], e["obj_rot"]
        start_c = (op < 0.10) & (orr < 0.50) & (fr <= start_cutoff)
        early_c = (op < c.enough_obj_threshold) & (orr < c.enough_obj_rot_threshold) & (not reached_end)
        late_c = (op < c.enough_obj_threshold_late) & (orr < c.enough_obj_rot_threshold_late) & reached_end
        good = (e["ft"] < c.enough_ft_threshold) & (start_c | early_c | late_c)
        still_good = self._enough_continued & good
        self._enough_idx = torch.where(still_good, fr, self._enough_idx)    # last good frame
        self._enough_continued = still_good

        # cache ranking key = the ACTUAL step reward (grasp / TJ convention; see the docstring).
        r = reward                                                          # (E,)
        # 유예 모드에선 여기서 `better` 를 비교하지 않는다. 에피소드 동안 캐시가
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

    # ── [wrist6] 상태 캐시 레이아웃(이름으로 참조). ctrl = 리셋 때 복원할 손목 목표 EMA(왼6 + 오6).
    _CACHE_LAYOUT = dict(                     # 174
        wristL=(1, 13), wristR=(13, 25),      # 관절 pos6 + vel6 = 12/손
        obj=(25, 38), jpos=(38, 74), jvel=(74, 110),
        j0pos=(110, 118), j0vel=(118, 126), smoothed=(126, 162), ctrl=(162, 174),
        dim=174,
    )

    def _build_cache_state(self, r: torch.Tensor, org: torch.Tensor) -> torch.Tensor:
        """(E,174) cache row for every env: [0] = the step reward (ranking key), rest = the full
        restorable sim state. Column 0 must be the SAME quantity the `better` comparison uses.
        레이아웃은 __init__ 의 _STATE_DIM 주석 참조."""
        L = self._CL
        state = torch.zeros(self.num_envs, self._STATE_DIM, device=self.device)
        state[:, 0] = r
        for _h, _sd in enumerate("lr"):
            _a, _b = L["wristL"] if _h == 0 else L["wristR"]
            _d = self._hands[_h].data
            # [wrist6] 손목 상태 = 관절값. 루트(anchor)는 env 원점에 용접돼 있어 담을 게 없다.
            _w6id = getattr(self, f"_wrist6_joint_ids_{_sd}")
            state[:, _a:_a + 6] = _d.joint_pos[:, _w6id]
            state[:, _a + 6:_b] = _d.joint_vel[:, _w6id]
        _oa, _ob = L["obj"]
        if self._has_object:
            state[:, _oa:_oa + 3] = self._object.data.root_pos_w - org
            state[:, _oa + 3:_oa + 7] = self._object.data.root_quat_w
            state[:, _oa + 7:_oa + 10] = self._object.data.root_lin_vel_w
            state[:, _oa + 10:_ob] = self._object.data.root_ang_vel_w
        else:
            state[:, _oa + 3] = 1.0                           # 사원수 항등 (미기록 행의 NaN 방지)
        state[:, L["jpos"][0]:L["jpos"][1]] = self._hand_joint_cat("joint_pos")
        state[:, L["jvel"][0]:L["jvel"][1]] = self._hand_joint_cat("joint_vel")
        # 텐던 축 J0 8개. _ref_j0_ids 는 손별 dict (손 articulation 내 DOF 인덱스).
        if isinstance(getattr(self, "_ref_j0_ids", None), dict):
            for _h, _sd in enumerate("lr"):
                _j = self._ref_j0_ids[_sd]
                _o = 4 * _h
                state[:, L["j0pos"][0] + _o:L["j0pos"][0] + 4 + _o] = \
                    self._hands[_h].data.joint_pos[:, _j]
                state[:, L["j0vel"][0] + _o:L["j0vel"][0] + 4 + _o] = \
                    self._hands[_h].data.joint_vel[:, _j]
        # [residual] 잔차 모드에서는 손가락 목표 EMA(관절 단위)가 복원해야 하는 상태다.
        # 폭은 36 으로 같아 레이아웃이 바뀌지 않지만 **단위가 다르다** (관절 rad vs 정규화 액션).
        # residual_action 을 바꾸면 캐시를 다시 만들어야 한다.
        state[:, L["smoothed"][0]:L["smoothed"][1]] = (
            self._hand_target_ema if self.cfg.residual_action else self._smoothed_actions)
        _ca, _cb = L["ctrl"]
        # 손목 목표 EMA (평활 상태가 곧 목표). 왼6 ++ 오른6.
        state[:, _ca:_ca + 6] = self._wrist6_target[0]
        state[:, _ca + 6:_cb] = self._wrist6_target[1]
        return state

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
                cand_state = self._pend_state[rows[sel[:, 0]], sel[:, 1]]    # (K,_STATE_DIM)
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
