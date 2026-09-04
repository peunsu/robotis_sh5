"""G1 + Shadow-hand full-body loco-manipulation environment (DirectRLEnv).

Ports the core mechanisms of `robotis_shadow_grasp_rsi` (per-group EMA + optional delta
action, contact-conditioned fingertip force with grounded normal, adaptive frame-sampling
curriculum, pretrain-cache RSI warm-start, deviation-from-reference termination, state cache)
from fixed-base single-hand to FLOATING-base bimanual FULL BODY.

SONIC-RESIDUAL VARIANT (this env): the 29 G1 body DOF are driven by a FROZEN SONIC whole-body
decoder — the policy outputs a 64-D latent residual z_res (added to SONIC's FSQ latent PRE-
quantization, GRAIL Eq.6, λ = residual_scale_latent = 0.15) — plus a 36-D ABSOLUTE bimanual hand
action (mapped directly to the Shadow joint range, EMA-smoothed α=0.5). action=100, obs=765
(obs prev_action = the raw 100-D policy action, GRAIL-style). The inherited per-group EMA / delta /
residual_action machinery below is the NON-SONIC fallback path (inactive while use_sonic=True).
SONIC is built in _post_init_buffers.

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
    contact-force reward uses the same scheme over all 32 wrap links (LINK_PAD_NORMALS), plus an
    orientation gate (link inward normal vs reference reaction normal, ≤ contact_normal_gate_tol) so
    contacting with the wrong face (e.g. back of the palm) yields no reward.
  * per-group EMA α = NEW-action weight; _smoothed_actions seeded at the normalized default.
  * obs prev_action = the RAW 100-D policy action (z_res 64 + a_hand 36), GRAIL-style (SONIC variant);
    the realized-target / delta-integrated prev_action is retained only for the dead fallback path.
  * termination is DEVIATION-FROM-REFERENCE (so reference crouch/bend never trips a fall).
  * state cache write-gate (episode_length>=3) while the pretrain cache is loaded.
"""

from __future__ import annotations

import math
import json
import os

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
    BODY_KPT_OFFSETS,
    BODY_KPTS,
    FINGERTIP_OFFSETS,
    FINGERTIP_PAD_NORMALS,
    HAND_CHAIN,
    JOINT_GROUPS,
    LINK_CONTACT_NAMES,
    LINK_PAD_NORMALS,
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


class G1ShadowSonicResidualEnv(DirectRLEnv):
    cfg: G1ShadowSonicResidualEnvCfg

    # ------------------------------------------------------------------ init
    def __init__(self, cfg: G1ShadowSonicResidualEnvCfg, render_mode: str | None = None, **kwargs):
        # [ROLLBACK MARKER: body-kpt-off] 몸 키포인트 감독 비활성화(실험): 보상·종료·캐시 bar 일괄
        # 적용. super() 이전에 cfg를 고쳐 두므로 exp 가중치 테이블(_post_init_buffers의 _lw 정규화 —
        # body 몫이 나머지 항에 자동 재배분)과 params/env.yaml 덤프 모두 적용된 값을 본다. 관측
        # (54kpt)과 e["body"] 계산·로깅은 그대로 — 감독만 빠진다. 낙상 감지는 루트 게이트가 대신
        # (_dones_deviation). 되돌리기: cfg.body_kpt_supervision=True.
        if not cfg.body_kpt_supervision:
            cfg.rew_body_kpts = 0.0        # 코어 몸 10kpt 보상 제거 (ee/hand/link/fingertip/root/obj 유지)
            cfg.term_body_kpt_err = 1.0e6  # body 종료 게이트 무력화 → 루트 게이트가 낙상 담당
            cfg.cache_body_bar = 1.0e6     # 캐시 body bar 무력화 (root/fingertip bar는 유지)
            print("[body-kpt-off] body keypoint supervision DISABLED: rew_body_kpts=0, "
                  "body termination/cache-bar off, root fall gate on "
                  f"(term_root_pos_err={cfg.term_root_pos_err}, term_root_rot_err={cfg.term_root_rot_err})")
        # [ROLLBACK MARKER: failure-sigma] beta 를 관측 마지막 열로 실어 보내므로 차원이 하나 늘어납니다.
        # super() 이전에 고쳐야 관측 공간과 params/env.yaml 덤프가 함께 맞습니다. 끄면 원래 766 그대로.
        self._fs_on = bool(getattr(cfg, "failure_sigma", False))
        if self._fs_on:
            cfg.observation_space = int(cfg.observation_space) + 1
            print(f"[failure-sigma] ON  beta_max={cfg.failure_sigma_beta_max} "
                  f"gamma={cfg.failure_sigma_gamma} dims={cfg.failure_sigma_dims} "
                  f"global={cfg.failure_sigma_global}  obs {cfg.observation_space - 1}→{cfg.observation_space}")
        self._load_reference_trajectories(cfg)          # numpy buffers (pre-super: no device yet) → sets _ref_len
        self._build_object_cfg(cfg)                     # guarded: only if converted USD exists
        # EPISODE = RSI start frame → END OF THE REFERENCE SEQUENCE (or a termination). The horizon is
        # the trajectory itself, so episodes are VARIABLE length (see _get_dones / _reset_idx).
        #   Previously (grasp/TJ lineage) every episode was a fixed `episode_length_s` chunk
        #   (3.0 s = 150 frames @50 fps) and the RSI start was clamped to [0, ref_len - chunk] so the
        #   chunk always fit. That wasted the tail of every clip: a 251-frame clip could only ever start
        #   in [0, 101], and a MEDIAN ParaHome clip (151 frames after the 30→50 fps resample) collapsed
        #   to [0, 1] — i.e. RSI was effectively dead for half the dataset. Now the start clamp only has
        #   to leave the run-up (_adaptive_back_frames) intact, so the whole clip is reachable.
        # max_episode_length is therefore the FULL sequence: it only acts as a safety cap, because the
        # frame-based time-out in _get_dones fires first (they coincide exactly for a start-at-0 env).
        # Must set cfg.episode_length_s BEFORE super().
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
        jp = d["joint_positions"].astype(np.float32)          # (F,73,3) world keypoints
        F = jp.shape[0]

        # 56 reference-keypoint indices into the 73-joint array (body + L-hand + R-hand).
        ref_idx: list[int] = list(BODY_KPTS.keys())            # 16 body (into [0:23])
        for side_off in (23, 48):                              # left block, right block
            for spec in HAND_CHAIN.values():
                ref_idx += [side_off + p for p in spec["parahome"]]
        self._np_ref_kpts = jp[:, ref_idx, :]                  # (F,54,3) = 14 body + 40 hand

        # reference FOOT-CONTACT schedule (F,2)=[left,right] via the SAME rule PyRoki used for retargeting
        # (_foot_contact): the ball keypoint (ParaHome 22=left, 18=right) is near the floor AND slow in z.
        # Computed at the NATIVE source rate (fps below) then resampled (thresholded) with the other refs.
        _bz = jp[:, [22, 18], 2]                               # (F,2) [L,R] ball height
        _bvz = np.zeros_like(_bz); _bvz[1:] = (_bz[1:] - _bz[:-1]) * _PARAHOME_FPS
        self._np_ref_foot_contact = ((_bz < cfg.foot_plant_h) & (np.abs(_bvz) < cfg.foot_plant_vz)).astype(np.float32)

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

        # context / support objects (ctx__<obj>__base): the non-manipulated scene objects the active
        # object rests on / is handled near. Selected by XY-proximity of the object's (static) frame-0
        # centroid to the active object's SWEPT trajectory (< cfg.context_radius). Spawned kinematic-
        # frozen in _setup_scene so the dynamic active object has support and does not fall to the
        # floor. Static → only the frame-0 pose is kept (no per-frame tensors, no reward/obs/termination).
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

        # (The old fingertip-only contact map — future_contact / contact_vertex / contact_normal, computed
        # here via object-velocity gate + nearest-object-vertex — was REMOVED 2026-07-22. The whole env now
        # uses the single Option-A per-LINK contact map loaded below from hand_contact.npz: force reward,
        # fingertip-keypoint reward, and delta_ft_obj obs all read _ref_link_contact_{mask,target,normal}.)

        # per-LINK contact (Option A, hand_contact.npz sidecar) — the SINGLE contact map for the whole env
        # (force reward + fingertip-keypoint reward + delta_ft_obj obs all read it). Per link (32 wrap links):
        #   mask (which SHOULD touch), object-LOCAL reaction normal (force projection dir), object-LOCAL
        #   target (object-surface contact point). Reordered to the canonical LINK_CONTACT_NAMES. The target
        #   is world in hand_contact → converted to object-LOCAL here (pose-invariant, transformed by the live
        #   object pose at runtime). Absent → zeros (contact terms inert).
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

        # ---- resample all per-frame references 30 fps → control_fps (SONIC runs at 50 Hz, so one
        #      control step == one reference frame). Everything above is computed at the source rate
        #      (correct finite-diff velocities / contact map); here we upsample and recompute root
        #      velocity at the new rate. cfg.ref_dt is updated so the runtime heuristics use it too. ----
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

        # object reference velocity (finite-diff of the FINAL, rate-matched _np_obj_base) — mirrors the
        # root velocity above; seeds the RSI reset object velocity on the reference path so a mid-motion
        # start places the object MOVING at its reference velocity (not at rest). rate = 1/cfg.ref_dt
        # (= tgt_fps after the resample block set it, else the source rate). All-zero when no object.
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
                    # [ROLLBACK MARKER: obj-depen-vel] 1.0 -> 0.1 (2026-08-14). 로봇 쪽은 이미
                    # 0.1로 내려두었는데(cfg의 depen-vel 마커) 물체만 1.0으로 남아 있었습니다.
                    # 겹침 해소 임펄스는 두 물체에 함께 걸리고 상한은 물체별로 적용되므로, 손이
                    # 물체를 파고들면 로봇은 0.1 m/s로 밀려나는데 물체는 1 m/s로 튕겨 나갑니다.
                    # 이 10배 비대칭이 "손이 닿으면 물체가 튄다"의 직접적 후보입니다. 값을 맞추면
                    # 겹침이 여러 스텝에 걸쳐 부드럽게 풀립니다.
                    # 대가는 로봇에서 이미 받아들인 것과 같습니다 — 깊은 겹침이 더 오래 지속되므로
                    # 초반 조기 종료율(term_obj_pos/obj_rot)과 접촉 시 물체 속도를 함께 봐야 합니다.
                    # 되돌리기: 1.0. 참고로 물체 USD 자체에도 1.0이 authored 되어 있는데, 이 spawn
                    # 설정이 그 값을 덮어씁니다(USD를 다시 굽지 않아도 됩니다).
                    max_depenetration_velocity=0.1),
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

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self._object = None
        if self._object_cfg is not None:
            self._object = RigidObject(self._object_cfg)
            self.scene.rigid_objects["object"] = self._object

        # per-LINK contact sensors (Option A): ONE ContactSensor per wrap link (LINK_CONTACT_NAMES, 32),
        # filtered on the active object → force_matrix_w = that link's contact force with the object (not
        # context/self). Used by the per-link contact-force reward + obs (the 10 distal/fingertip links are
        # included). Individual per-link sensors (user decision). history_length=1 (only the current force is
        # read → no history buffer).
        # update the 34 contact sensors once per CONTROL step (not every physics sub-step): the reward/obs
        # read force_matrix only at the control rate, so processing the (expensive, convex-decomp) filtered
        # contact buffers ~decimation× less often is a large speedup with no behavioral change. PhysX still
        # computes contacts every physics step; this only throttles the Isaac-side buffer unpack.
        _ctrl_dt = self.cfg.sim.dt * self.cfg.decimation                 # control period (s) = 1/50
        self._link_contact_sensors: list[ContactSensor] = []
        obj_filter = ["/World/envs/env_.*/Object"] if self._object_cfg is not None else []
        for name in LINK_CONTACT_NAMES:
            s = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{name}",
                filter_prim_paths_expr=obj_filter, history_length=1, update_period=_ctrl_dt,
                # [ROLLBACK MARKER: cws-contact] contact_pos_w = 이 링크와 물체 사이 접촉점들의 평균
                # 위치(월드). 접촉 렌치의 모멘트 팔이 이 위치라서 필요합니다. 링크 원점을 대신 쓰면
                # 실제 닿은 곳에서 링크 크기만큼 떨어진 곳을 기준으로 회전 효과를 계산하게 됩니다.
                # 접촉이 없는 쌍은 NaN으로 나오므로 반드시 마스크로 걸러야 합니다.
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

        # context / support objects: spawn each selected scene object (support surface / nearby
        # furniture) as a KINEMATIC-frozen collider at its reference pose, BEFORE clone_environments
        # so it replicates per-env. Direct .func() spawn — NOT a RigidObject, so no per-env GPU
        # root-state view is allocated for geometry that never moves. kinematic_enabled freezes it
        # (unaffected by gravity/contact) so the dynamic active object rests on it. Leaf is FLAT under
        # env_.* (Ctx_<i>_<name>, like grasp's TableBase/TableTop) — an intermediate scope like
        # env_.*/Context/ fails ("Unable to find source prim path" — the @clone wrapper needs the
        # parent to exist first). Leaf != "Object" so it stays out of the fingertip contact filter
        # (which matches the literal leaf "Object"), leaving ft_max_contact_points valid. USD-guarded.
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
            # [ROLLBACK MARKER: context-z] sink the supports so the reference object rests ON them
            # instead of slightly inside; see the cfg field for the measurements behind it.
            ctx_spawn.func(prim_path, ctx_spawn,
                           translation=(float(pose0[0]), float(pose0[1]),
                                        float(pose0[2]) - float(getattr(self.cfg, "context_z_offset", 0.0))),
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
        already keyed to. Set cfg.remap_ref_joint_order=False for the pre-fix behaviour.
        """
        if self._ref_joints is None or not getattr(c, "remap_ref_joint_order", True):
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
        # ── [ROLLBACK MARKER: ref-j0] 리타게팅이 푼 손가락 J0 를 보존 (2026-09-02) ─────────────
        # 부등식 리타게팅(tendon-ineq)은 J0 8개를 자유 변수로 풀어 73열로 저장합니다. env 액션
        # 관절은 65개(J0 없음, 텐던이 구동)라 아래 perm 이 65열만 골라내고 J0 는 버려졌습니다.
        # 그러면 리셋이 J0 를 J1 x gear 로 재계산하는데, 그 gear(1.14184)는 Shadow 문서의
        # q_J0 <= q_J1 을 위반합니다 — 풀어놓은 값을 버리고 틀린 값을 만드는 셈입니다.
        # 여기서 J0 열을 따로 떠 두고 _reset_idx 가 그걸 씁니다. 65열 npz(등식 mimic)에서는
        # _ref_j0 가 None 이 되어 기존 텐던 결합 경로가 그대로 돕니다.
        _j0n = [f"robot0_{sd}_{fg}J0" for sd in "lr" for fg in ("FF", "MF", "RF", "LF")]
        self._ref_j0 = None
        self._ref_j0_ids = None
        if all(n in src for n in _j0n):
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

        # ---- park the robot for the duration --------------------------------------------------
        # The reference hand interpenetrates the object, so solving with the robot in place would
        # measure the hand shoving it. The base FLOATS, so parking it once is not enough: it falls
        # back into the scene in ~200 steps and starts striking things.
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

        # `_settle` returns settled-minus-reference, which IS the lift that would have put the object
        # where it ends up — so `lift <- settle(lift)` is a fixed-point iteration, no accumulation.
        # One round already lands the pan within a millimetre; the extra rounds are for shapes whose
        # settle is not idempotent (a thin object that tips as it settles moves its own contact set).
        # Negative = the reference floats above the support: clamped away, this correction only lifts.
        lift = torch.zeros(F, device=dev)
        for _rnd in range(_DECLEAR_ROUNDS):
            for base in range(0, len(rest_idx), n):
                fr = rest_idx[base:base + n]
                dz = torch.zeros(n, device=dev)
                dz[:len(fr)] = lift[fr]
                d = _settle(fr, dz)
                lift[fr] = d.clamp(min=0.0, max=float(c.declear_max_lift))

        # ---- per frame, not per segment ---------------------------------------------------------
        # The lift used to be raised to the MAX over each rest segment, so that consecutive start
        # frames spawned at one height and every frame in the segment cleared. That was a stand-in
        # for not knowing the per-frame value: the old probe could only answer "clear / not clear",
        # so the safe move was to over-lift. The settle measures each frame exactly, and over-lifting
        # reintroduces the artifact it is meant to remove — on the knife the segment max was 22.9 mm
        # against a 19.8 mm mean, and the object then FELL 13.2 mm at spawn. Per-frame lift leaves a
        # residual of about a millimetre. Segment counting is kept only for the log.
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

        # ---- verification: re-settle AT the corrected height ------------------------------------
        # Residual = settled height - SPAWN height, which is what the old velocity test could never
        # report. Near zero means the object now spawns where it comes to rest; positive means the
        # lift was too small (or declear_max_lift clipped it), negative means it still drops.
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
        # [ROLLBACK MARKER: failure-dump] 액션 관절의 실제 이름 순서. find_joints 는 정규식 순서가
        # 아니라 아티큘레이션 내부 순서로 돌려주므로 밖에서 추측할 수 없습니다. 덤프를 나중에
        # 해석하려면 반드시 함께 저장해야 합니다.
        self._action_joint_names = [self.robot.joint_names[i] for i in self._action_joint_ids]
        if bool(getattr(c, "failure_dump", False)) or bool(os.environ.get("PRINT_ACTION_JOINTS")):
            print("[action-joints] " + " ".join(f"{i}:{n}" for i, n in
                                                enumerate(self._action_joint_names)))
        self._n_act = off                                              # 65
        # ── [ROLLBACK MARKER: tendon-reset] 텐던 축 관절 J0 의 리셋 값 ────────────────────────
        # 손가락 말단 J0 8개는 액추에이터가 없고 PhysX 고정 텐던으로 J1 에 묶여 있습니다
        # (q_J0 = 1.1418 * q_J1). 액션 관절 65개에 포함되지 않으므로 _reset_idx 의
        # `jpos = default_joint_pos.clone()` 이 항상 0 으로 두는데, 그러면 J1 이 굽은 프레임에서
        # 리셋할 때마다 텐던 제약이 최대 1.14 rad 위반된 상태로 시작합니다.
        # 실측(J1=1.0 rad 로 리셋): J0 가 4스텝(20 ms) 만에 따라잡지만 그 과정에서 말단 링크가
        # 1.705 m/s 로 튑니다 — 리셋 직후 손 속도(0.35~0.57 m/s)의 3배입니다. J0 를 제약에 맞춰
        # 써주면 그 과도응답이 사라집니다(같은 실측에서 J0 시작값이 곧 목표값).
        # 되돌리기: tendon_reset_couple=False.
        _tj1n = [f"robot0_{s}_{f}J1" for s in "lr" for f in ("FF", "MF", "RF", "LF")]
        _jn = self.robot.data.joint_names
        _tp = [(_jn.index(a), _jn.index(a[:-1] + "0")) for a in _tj1n
               if a in _jn and (a[:-1] + "0") in _jn]
        self._tendon_j1_ids = torch.tensor([a for a, _b in _tp], device=dev, dtype=torch.long)
        self._tendon_j0_ids = torch.tensor([b for _a, b in _tp], device=dev, dtype=torch.long)
        self._tendon_gear = float(c.tendon_gear_ratio)
        print(f"[tendon-reset] {'ON' if c.tendon_reset_couple else 'OFF'} — "
              f"텐던 쌍 {len(_tp)}개, gear={self._tendon_gear:.5f}")
        self._group_alpha = {n: float(g["ema_alpha"]) for n, g in JOINT_GROUPS.items()}
        # per-group residual scale, (1,65) so it broadcasts with the (E,65) action: hands wider, body tighter
        _res_scale = torch.full((self._n_act,), float(c.residual_scale_body), device=dev)
        _res_scale[self._group_slices["hands"]] = float(c.residual_scale_hands)
        self._residual_scale_t = _res_scale.unsqueeze(0)               # (1,65)

        # per-action joint limits (in action order) for scale/unscale + delta clamp
        lim = self.robot.data.soft_joint_pos_limits[0, self._action_joint_ids_t]       # (65,2)
        self._ctrl_lower = lim[:, 0].clone()
        self._ctrl_upper = lim[:, 1].clone()

        # ---- keypoint body ids + local offsets (56, matching the reference order) ----
        kpt_names: list[str] = list(BODY_KPTS.values())
        kpt_off: list[list[float]] = [BODY_KPT_OFFSETS.get(i, [0.0, 0.0, 0.0]) for i in BODY_KPTS]
        # [ROLLBACK MARKER: hand-kpt-align] 오프셋은 링크 이름이 아니라 키포인트별 `pad` 플래그로
        # 정합니다. distal 을 두 번 쓰기 때문입니다 — 오프셋 0 이면 DIP, pad 면 TIP. 예전처럼
        # FINGERTIP_OFFSETS.get(링크이름) 으로 조회하면 두 번째(=TIP) 항목까지 pad 를 받거나
        # 반대로 둘 다 0 이 되어 구분이 사라집니다.
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
        # SPLIT body-keypoint reward: the 14 body kpts are partitioned into WRIST (reach/manipulation),
        # ANKLE (foot placement/balance) and CORE (everything else). Each gets its own reward weight
        # (rew_ee_kpts / rew_body_kpts) so the groups can be emphasized independently.
        # Matched by name (robust to reordering). The termination gate still uses the UNIFORM mean over
        # ALL 14 (e["body"]). [ROLLBACK MARKER: ee-split] wrist+ankle used to share one _ee_kpt_idx.
        # [ROLLBACK MARKER: ee-torso] torso_link 이 core(가중치 0)에서 EE 그룹으로 이동했다. 세 그룹은
        # 서로 배타적이다 (9 core + 2 wrist + 3 ee = 14). 되돌리기: 아래 "torso" 조건 2곳을 제거.
        _body_names = kpt_names[:len(BODY_KPTS)]
        # [ROLLBACK MARKER: wrist-into-ee] 손목이 EE 그룹으로 흡수됐다 (손목2+발목2+몸통1 = 5).
        _EE_NAMES = ("wrist", "ankle", "torso")   # EE 그룹에 들어가는 링크 이름 조각
        self._body_core_idx = torch.tensor(
            [i for i, n in enumerate(_body_names)
             if not any(t in n for t in _EE_NAMES)],
            device=dev, dtype=torch.long)                                              # (9,) core body
        # [wrist-into-ee] 손목 2개는 이제 보상에서 EE 항에 흡수됐고, 이 인덱스는 오직 전용
        # 손목-POSITION 종료 게이트(term_wrist_pos_err)와 Error / wrist_kpts 로그에만 쓰인다.
        self._wrist_kpt_idx = torch.tensor(
            [i for i, n in enumerate(_body_names) if "wrist" in n], device=dev, dtype=torch.long)  # (2,)
        # [wrist-into-ee] 손목 2 + 발목 2 + 몸통 1 = 5개의 MEAN → rew_ee_kpts. 몸통은 torso_link
        # 원점이 아니라 BODY_KPT_OFFSETS[4] (TORSO_KPT_OFFSET, 윗가슴)이 더해진 점이다.
        self._ee_kpt_idx = torch.tensor(
            [i for i, n in enumerate(_body_names) if any(t in n for t in _EE_NAMES)],
            device=dev, dtype=torch.long)                                    # (5,) wrist×2 + ankle×2 + torso
        # 이름 매칭이라 조용히 잘못 묶일 수 있다 — 실제 구성을 한 번 찍어 확인 가능하게 남긴다.
        print(f"[g1] kpt groups: core({len(self._body_core_idx)})="
              f"{[_body_names[i] for i in self._body_core_idx.tolist()]} | "
              f"wrist({len(self._wrist_kpt_idx)})={[_body_names[i] for i in self._wrist_kpt_idx.tolist()]} | "
              f"ee({len(self._ee_kpt_idx)})={[_body_names[i] for i in self._ee_kpt_idx.tolist()]}")
        # [ROLLBACK MARKER: energy] Σ|τ·q̇| 대상 = 허리 3 + 다리 12 = 15관절. 팔·손목은 제외
        # (레퍼런스 파워가 가장 크고 로봇이 이미 레퍼런스보다 느리다 — cfg 주석 참조).
        _en_pat = ("waist_", "_hip_", "_knee_", "_ankle_")
        self._energy_joint_ids = torch.tensor(
            [i for i, n in enumerate(self.robot.joint_names) if any(p in n for p in _en_pat)],
            device=dev, dtype=torch.long)
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
        # ---- palm (wrist) body ids for explicit palm orientation/velocity obs (L,R to match ft order) ----
        self._palm_body_ids = torch.tensor(
            [self.robot.find_bodies(n)[0][0] for n in ("robot0_l_palm", "robot0_r_palm")],
            device=dev, dtype=torch.long)

        # ---- reference tensors (move numpy → device) ----
        def T(a):
            return torch.from_numpy(np.asarray(a)).to(dev)
        self._ref_kpts = T(self._np_ref_kpts)                          # (F,56,3)
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
        # ── [ROLLBACK MARKER: ref-reset-jvel] 레퍼런스 관절 속도 (F,65), 액션 관절 순서 ──────────
        # 여기여야 하는 이유가 둘 있습니다. (1) _remap_ref_joints 바로 뒤 — 그 안에서 열이 이름
        # 기준으로 재정렬되므로(_ref_joints[:, perm]), 그 전에 차분하면 조용히 엉뚱한 관절에
        # 속도가 들어갑니다. (2) SONIC 가드 바깥 — 리셋 경로는 use_sonic 여부와 무관하게 이
        # 배열을 씁니다(_ref_g1_v / _ref_hist["jvr"]는 SONIC 초기화 블록 안에 있고 29열뿐이라
        # 재사용할 수 없습니다: 몸통 29개만이고 순서도 SONIC 규약입니다).
        # 공식은 루트 속도와 동일한 규약입니다 — 후방차분, 0번 프레임 0(클립 시작에서 레퍼런스는
        # 실제로 정지). 역방향 부호 반전은 사용처(_reset_idx)에서 루트와 같이 처리합니다.
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
        # [ROLLBACK MARKER: ref-j0] 손가락 J0 속도도 리타게팅 결과에서 뽑습니다 — 위 65열과 같은
        # 규약(후방차분 x control_fps, 0번 프레임 0, nan 정리)입니다. 예전에는 리셋이
        # jvel[J0] = jvel[J1] x 1.14184 로 만들었는데, 그 비율은 Shadow 문서의 q_J0 <= q_J1 을
        # 위반하는 값이라 풀어놓은 궤적을 버리고 틀린 속도를 만드는 셈이었습니다.
        self._ref_j0_vel = None
        if getattr(self, "_ref_j0", None) is not None and self._ref_jvel_scale > 0.0:
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
        # reference palm/wrist orientation per hand [L,R] for the wrist-rotation gate. Sourced ONLY from
        # the retarget g1_palm_quat if the clip provides it; the current PyRoki retarget omits it, so
        # _has_palm_ref is False → wrist_rot is inert (and enable_wrist_rot_termination=False anyway).
        # (An env-side keypoint→frame estimate was tried but the human-hand-frame ↔ robot-palm relation
        #  drifts up to ~90° over a clip — the position-only retarget + Shadow≠human embodiment don't
        #  track the human wrist orientation — so any human-derived reference false-terminates; removed.)
        self._has_palm_ref = self._np_ref_palm_quat is not None
        self._ref_palm_quat = _canon(T(self._np_ref_palm_quat)) if self._has_palm_ref else None  # (F,2,4)
        self._has_object = self._object is not None
        self._RESERVE_ARTIC = 4                                        # reserved obs slots per parts

        # future-contact (F,10) + contact map (F,10,3 vertex/normal, object-local) — GR mechanism
        # computed at load (object velocity + fingertip proximity; contact map zeros until the
        # object mesh is wired into preprocessing → pad-normal fallback in the reward).
        # per-LINK contact (Option A): reference mask + object-local reaction normal + object-local target
        # + robot body ids. _ft_distal_idx maps the 10 fingertips (cfg.fingertip_body_names order) to their
        # distal-link index in LINK_CONTACT_NAMES → lets the fingertip-keypoint reward / delta_ft_obj read the
        # per-link map's distal entries (single unified contact map).
        self._ref_link_contact_mask = T(self._np_link_contact_mask)            # (F,L)
        self._ref_link_contact_normal_local = T(self._np_link_contact_normal)  # (F,L,3) object-local
        self._ref_link_contact_target_local = T(self._np_link_contact_target)  # (F,L,3) object-local

        # [ROLLBACK MARKER: cws-contact] 접촉을 "물체를 어떻게 움직일 수 있는가"로 바꿔 비교합니다
        # (CHORD, arXiv 2607.00033). 링크별 위치 매칭은 사람 손 기준이라 Shadow 손이 도달할 수 없는
        # 접촉을 요구합니다 — 레퍼런스 자세조차 오른손바닥 요구의 5.4%밖에 못 채웠습니다. 렌치로
        # 비교하면 손바닥 대신 손가락으로 같은 밀기/비틀기를 내도 인정되고, 같은 레퍼런스에서
        # 덮은 방향 비율이 66%(중앙 80%)로 올라갑니다.
        # 사람 쪽 목록은 여기서 한 번만 계산합니다. 물체 기준 좌표라 물체 자세와 무관하고, 파일로
        # 빼면 리타게팅을 다시 돌렸을 때 옛 값이 남는 위험만 생깁니다.
        self._cws_sigma_h = None
        # [ROLLBACK MARKER: cws-diag] cws_log_only 면 보상 모드와 무관하게 계산만 켭니다.
        if self._has_link_contact and (c.contact_reward_mode in ("cws", "both")
                                       or bool(getattr(c, "cws_log_only", False))):
            _m = self._ref_link_contact_mask > 0.5                             # (F,L)
            if bool(_m.any()):
                # [ROLLBACK MARKER: cws-rc-mesh] 물체 크기 = 메시 정점 중심에서 최대 거리(논문
                # 정의). 렌치의 회전 성분을 이걸로 나눠 밀기 성분과 같은 자리수로 맞춥니다.
                # 메시를 못 읽으면 옛 추정(접촉점 노름 0.9 분위)으로 떨어집니다 — 되돌리려면
                # 아래 _rc_mesh 를 None 으로 두면 됩니다.
                _rc_q90 = float(self._ref_link_contact_target_local[_m].norm(dim=-1).quantile(0.9))
                _rc_mesh = self._object_mesh_radius() if self._has_object else None
                _rc_ok = _rc_mesh is not None and _rc_mesh > 1e-4
                self._cws_len = _rc_mesh if _rc_ok else _rc_q90
                self._cws_basis = CWS.make_basis(c.cws_n_dir, c.cws_seed, device=dev)
                # [ROLLBACK MARKER: cws-com] 모멘트 팔은 물리 COM 기준이어야 렌치의 회전 성분이
                # 동역학적으로 의미가 있습니다(논문은 body_com_state_w 를 씁니다). 레퍼런스 접촉
                # 목표는 물체 body 프레임 값이라, COM 프레임으로 옮긴 뒤 지지함수를 계산합니다.
                # 로봇 쪽(_compute_rewards)도 root_com_pos_w/root_com_quat_w 를 쓰므로 두 sigma 가
                # 같은 기준을 갖습니다. COM 을 못 읽으면 양쪽 모두 body 원점으로 떨어집니다.
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
                _mode = ("진단 전용(보상 제외)" if c.contact_reward_mode not in ("cws", "both")
                         else "보상 포함")
                _src = "메시" if _rc_ok else "접촉점0.9분위(메시 읽기 실패)"
                print(f"[cws] 접촉 렌치 {_mode}  물체크기={self._cws_len * 100:.1f}cm({_src}, "
                      f"접촉점0.9분위={_rc_q90 * 100:.1f}cm)  "
                      f"모멘트팔={'COM' if self._cws_com_p is not None else 'body원점'}  "
                      f"방향={c.cws_n_dir}  옆면={c.cws_n_edge}  여유={c.cws_beta}  mu={c.cws_mu}")
        self._link_contact_body_ids = torch.tensor(
            [self.robot.find_bodies(n)[0][0] for n in LINK_CONTACT_NAMES], device=dev, dtype=torch.long)  # (L,)
        # [ROLLBACK MARKER: link-kpt] 접촉 목표 보상에 쓸 링크. 손바닥은 기본 제외입니다 —
        # 레퍼런스 자세에서조차 오른손바닥은 요구 프레임의 12.6%만 목표에 닿습니다(측정값).
        # 도달 불가능한 목표를 보상으로 강제하면 정책이 손바닥을 물체로 밀어 넣어 관통을 만듭니다.
        # 손가락 마디만으로도 방향은 충분히 묶입니다.
        _use = torch.ones(N_LINK_CONTACT, dtype=torch.bool, device=dev)
        if not c.link_kpt_include_palm:
            _use[[i for i, n in enumerate(LINK_CONTACT_NAMES) if n.endswith("_palm")]] = False
        # [link-kpt-smpl] SMPL 대응이 없는 링크(thproximal)는 항에서 제외. _build_ref_link_kpt_local
        # 이 _post_init_buffers 뒤에 돌므로 여기서는 곱하지 않고, 사용 시점에 논리곱한다.
        self._link_kpt_use = _use.unsqueeze(0)                          # (1,L) 브로드캐스트
        # [ROLLBACK MARKER: force-fingertip-only] 접촉력 보상에 쓸 링크.
        # 32개 전체를 쓰면 분모(요구 링크 수)에 손바닥처럼 레퍼런스조차 12.6%만 달성하는 링크가
        # 섞여 보상이 구조적으로 눌립니다(프레임당 7개 요구 중 실제 접촉 3개). 손끝 10개로
        # 좁히면 요구와 달성이 같은 대상을 가리킵니다. 링크별 위치 추종은 rew_link_kpts가 담당하고
        # 이 항은 접촉 유무/세기만 봅니다.
        self._ft_distal_idx = torch.tensor(
            [LINK_CONTACT_NAMES.index(n) for n in self.cfg.fingertip_body_names], device=dev, dtype=torch.long)  # (10,)
        _fu = torch.zeros(N_LINK_CONTACT, dtype=torch.bool, device=dev)
        _fu[self._ft_distal_idx] = True
        self._force_link_use = (_fu if self.cfg.contact_force_fingertip_only
                                else torch.ones_like(_fu)).unsqueeze(0)        # (1,L)
        # per-link OUTWARD pad/palmar normal (link-local) → the link's own contact FACE. Force is projected on
        # the INWARD (-pad) direction (like the fingertip), and the orientation gate compares this face to the
        # reference reaction normal. VERIFIED on the rest-pose USD (32/32; see cfg LINK_PAD_NORMALS).
        self._link_pad_normals = torch.tensor(
            [LINK_PAD_NORMALS[n] for n in LINK_CONTACT_NAMES], device=dev, dtype=torch.float32)  # (L,3) OUTWARD
        self._contact_normal_gate_cos = math.cos(self.cfg.contact_normal_gate_tol)  # gate threshold (precomputed)

        # ---- action / EMA / delta buffers ----
        default_q = self.robot.data.default_joint_pos[:, self._action_joint_ids_t]      # (E,65)
        self._smoothed_actions = self._unscale(default_q).clone()      # (E,65) normalized
        self._prev_action = torch.zeros(self.num_envs, self._n_act, device=dev)
        # per-group delta(residual)-action buffers + config (velocity cmd → integrated target). ANY of
        # the 4 groups can be switched to delta mode via cfg {leg,waist,arm,hand}_delta_action; groups
        # with the switch OFF stay on the absolute per-group EMA. Buffers exist for all groups (unused
        # ones just track the reset pose). Mirrors robotis_shadow_grasp_rsi arm/hand delta-action.
        self._delta_target = {g: default_q[:, sl].clone() for g, sl in self._group_slices.items()}
        self._delta_ema = {g: torch.zeros_like(self._delta_target[g]) for g in self._group_slices}
        self._residual_target = None       # (E,65) per-step residual target (ref[frame]+scale·a) when residual_action
        self._delta_cfg: dict[str, tuple] = {}
        for g in self._group_slices:
            pfx = g[:-1] if g.endswith("s") else g          # legs→leg, arms→arm, hands→hand, waist→waist
            self._delta_cfg[g] = (bool(getattr(self.cfg, f"{pfx}_delta_action")),
                                  float(getattr(self.cfg, f"{pfx}_delta_scale")),
                                  float(getattr(self.cfg, f"{pfx}_delta_smoothing")))

        # ---- per-env trajectory frame index ----
        self._frame_idx = torch.zeros(self.num_envs, device=dev, dtype=torch.long)

        # ---- state cache + RSI (train 222) ----
        # layout: reward(1) + root[pos3+quat4+linvel3+angvel3=13] + obj[pos3+quat4+linvel3+angvel3=13]
        #         + jpos(65) + jvel(65) + smoothed(65) = 1+13+13+195 = 222.
        #   [0]=reward [1:4]rootpos [4:8]rootquat [8:11]rootlinvel [11:14]rootangvel
        #   [14:17]objpos [17:21]objquat [21:24]objlinvel [24:27]objangvel
        #   [27:92]jpos [92:157]jvel [157:222]smoothed
        self._STATE_DIM = 222
        # [ROLLBACK MARKER: spawn-declear] steps physics; everything it touches (reference arrays,
        # object, robot, scene) already exists by here.
        self._solve_spawn_declear()
        # [ROLLBACK MARKER: context-z] 지지면을 내려 물체가 레퍼런스 높이에 안착하게 합니다.
        # 반드시 declear 뒤에 — 내릴 양을 그 결과에서 읽습니다.
        self._apply_context_z_auto()
        # [ROLLBACK MARKER: link-kpt-objframe] 레퍼런스 손 자세를 물체 기준 좌표로 미리 계산합니다.
        # rew_link_kpts의 목표를 "물체 표면의 접촉점"에서 "물체 기준으로 표현한 레퍼런스 링크 위치"로
        # 바꾸기 위한 것입니다. 물체가 회전하면 목표도 함께 회전하므로, 스폰 정착으로 물체가 돌아가도
        # (실측: 정지 프레임 중앙값 11.5도, 나쁜 프레임은 56도) 손이 따라가야 할 자세가 물체에 붙어
        # 있습니다. 접촉점과 달리 접촉이 요구되지 않는 프레임에서도 정의됩니다.
        # 로봇을 프레임마다 레퍼런스 자세로 세워 링크 위치를 읽습니다. 물리를 진행시킬 필요가 없어
        # 청크당 한 스텝이면 되고, 환경 수가 프레임 수 이상이면 한 번에 끝납니다.
        self._build_ref_link_kpt_local()      # [ROLLBACK MARKER: link-kpt-smpl]
        # [/ROLLBACK MARKER: spawn-declear]
        # ── [ROLLBACK MARKER: failure-dump] 실패 에피소드 링 버퍼 ─────────────────────────────
        # 한 행 = 한 (환경, 제어 스텝):
        #   [0:222]   _build_cache_state — 복원 가능한 전체 상태 ([0]은 그 스텝의 보상)
        #   [222:322] 원시 정책 액션 100차원 (반사실 섭동의 기준점)
        #   [322:354] 링크별 압축 접촉력 32개 (어느 마디가 실제로 눌렸는가)
        #   [354:360] 레퍼런스 프레임 / 학습 스텝 / 에피소드 길이 / 시작 프레임 / 종료원인 / 환경id
        # 링 버퍼는 GPU에 두고(수 MB), 종료된 환경의 창만 CPU로 꺼내 모았다가 npz로 씁니다.
        self._fd_on = bool(getattr(c, "failure_dump", False))
        if self._fd_on:
            self._FD_DIM = 364      # [cws-diag] 360 score 361 nhit 362 coverage 363 deficit
            self._fd_n = min(int(c.failure_dump_envs), self.num_envs)
            self._fd_w = int(c.failure_dump_window)
            self._fd_ring = torch.zeros(self._fd_n, self._fd_w, self._FD_DIM, device=dev)
            self._fd_ptr = 0                       # 다음에 덮어쓸 슬롯 (전 환경 공통 — 스텝이 같으므로)
            self._fd_filled = 0                    # 링이 몇 칸 찼는지 (초반 부분 창 방지)
            self._fd_buf: list = []                # CPU 로 꺼낸 창들
            self._fd_saved = 0                     # 지금까지 파일에 쓴 에피소드 수
            self._fd_bucket_taken = 0              # 현재 버킷에서 담은 수
            self._fd_bucket_id = -1
            _nb = max(1, int(getattr(c, "failure_dump_total_steps", 41000)) // max(1, int(c.failure_dump_bucket)))
            _bk = max(1, int(c.failure_dump_budget) // _nb)
            self._fd_bucket_cap = _bk              # 버킷당 상한 → 학습 전 구간에 고르게 분포
            import os as _os
            self._fd_dir = str(c.failure_dump_dir) or _os.path.join(_os.getcwd(), "failure_dump")
            _per_ep = self._fd_w * self._FD_DIM * 4 / 1024**2
            print(f"[failure-dump] ON  환경 {self._fd_n}/{self.num_envs}  창 {self._fd_w}스텝  "
                  f"예산 {c.failure_dump_budget} 에피소드 (버킷당 {_bk})  "
                  f"에피소드당 {_per_ep*1024:.1f} KB  →  예상 총 {c.failure_dump_budget * _per_ep:.0f} MB "
                  f"(압축 전)  링 버퍼 {self._fd_n*self._fd_w*self._FD_DIM*4/1024**2:.1f} MB")
            print(f"[failure-dump] 저장 경로: {self._fd_dir}")
        # ── [/ROLLBACK MARKER: failure-dump] ──────────────────────────────────────────────────
        self._state_cache = torch.zeros(self._ref_len, self._STATE_DIM, device=dev)
        self._state_cache[:, 0] = -float("inf")                        # reward column
        self._init_flg = torch.ones(self._ref_len, device=dev, dtype=torch.bool)   # True = reference (no cache)
        self._reached_frame = 0
        # [ROLLBACK MARKER: late-gate] the cache quality gate's early->late switch.
        # It used to read `_reached_frame >= _ref_len - 3`, i.e. "some episode once cached a frame
        # near the end". Under reference-seeded RSI every frame is a legal start, so an episode that
        # begins at frame 480 of 501 satisfies that within ~20 steps and the gate latches on the
        # first control step — `early_c` was dead and the tight `late` bars applied from the start.
        # The fix is a conjunction on ONE episode: it must have run for at least
        # late_gate_survival_frac of the clip AND have finished within 3 frames of the end, which
        # together mean "started inside the first 20% and completed the clip".
        # [ROLLBACK MARKER: exp-tracking] 지수 추적-보상 테이블 (복원 2026-09-02).
        # 항별 가중치는 선형 가중치에서 파생한다: |rew_*| 를 exp_tracking_budget 으로 정규화.
        # 두 형태가 같은 상대 강조를 유지하므로 SHAPE 만 바뀌고, 선형 가중치를 재튜닝하면
        # 지수 가중치도 함께 움직인다. 선형 가중치는 기울기(오차 1m 당 보상)이고 지수 가중치는
        # 그 항의 최대 지급액이라 단위가 다르다 — 이 정규화가 유일하게 방어 가능한 다리이고,
        # "얼마나 오차를 허용하는가"는 가중치가 아니라 σ 가 진다.
        _lw = {"link_kpt": abs(c.rew_link_kpts),
               "body": abs(c.rew_body_kpts),
               "ee": abs(c.rew_ee_kpts),                                   # [wrist-into-ee]
               "hand": abs(c.rew_hand_kpts),
               "fingertip": abs(c.rew_fingertip), "root_pos": abs(c.rew_root_pos),
               "root_rot": abs(c.rew_root_ori), "obj_pos": abs(c.rew_obj_pos),
               "obj_rot": abs(c.rew_obj_rot)}
        _tot = sum(_lw.values()) or 1.0
        self._exp_w = {k: float(c.exp_tracking_budget) * v / _tot for k, v in _lw.items()}
        self._exp_s2 = {"link_kpt": c.sigma_link_kpts ** 2, "body": c.sigma_body ** 2,
                        "ee": c.sigma_ee ** 2,                                  # [wrist-into-ee]
                        "hand": c.sigma_hand ** 2,
                        "fingertip": c.sigma_fingertip ** 2, "root_pos": c.sigma_root_pos ** 2,
                        "root_rot": c.sigma_root_rot ** 2, "obj_pos": c.sigma_obj_pos ** 2,
                        "obj_rot": c.sigma_obj_rot ** 2}
        # 각 항이 읽을 오차 키. [z-weight] hand/obj_pos 는 보상용 z 가중 사본을 읽는다 —
        # 지수 항이 곧 보상이고, z 가중은 "보상에만" 거는 규약이기 때문이다 (게이트는 무가중).
        self._exp_key = {"link_kpt": "link_kpt", "body": "body_core",
                         "ee": "ee",                                                 # [wrist-into-ee]
                         "hand": "hand_w", "fingertip": "ft_reward",                 # [z-weight]
                         "root_pos": "root_pos", "root_rot": "root_rot",
                         "obj_pos": "obj_pos_w", "obj_rot": "obj_rot"}                # [z-weight]
        if c.exp_tracking_reward:
            print("[exp-tracking] ON  budget=" + f"{c.exp_tracking_budget}  " + "  ".join(
                f"{k}: w={self._exp_w[k]:.3f} s={self._exp_s2[k] ** 0.5:.3g}" for k in _lw))

        # ── 물체 마찰 커리큘럼 [ROLLBACK MARKER: friction-curriculum] ──────────────────────
        # robotis_shadow_grasp_rsi에서 가져왔습니다. 초반에는 마찰을 높여 물체가 잘 안 미끄러지게
        # 하고, 학습이 진행되면 실제 값으로 조입니다. 지금 병목은 "잡았는데 미끄러진다"에 가까운데,
        # 그러면 정방향 롤아웃이 병목 프레임을 통과하지 못해 그 프레임의 캐시가 실패 상태로만
        # 채워집니다. 마찰을 올리면 통과한 궤적이 생기고, 캐시가 스스로 좋아집니다.
        # ── 역방향 롤아웃 [ROLLBACK MARKER: backward-dir] ──────────────────────────────────
        # 환경의 일부를 시간 역방향으로 굴립니다. 목적은 병목 프레임의 캐시를 채우는 것입니다 —
        # 정방향 롤아웃이 프레임 f를 통과하지 못하면 f의 캐시가 실패 상태로만 차고, 그 상태로
        # 시작한 에피소드가 또 실패하는 고착이 생깁니다. 뒤에서 거꾸로 내려오면 f에 도달할 수
        # 있고, 거기서 얻은 상태가 정방향 시작점이 됩니다.
        # 분할은 런 내내 고정입니다(에피소드마다 바꾸면 방향 비트가 환경 정체성과 어긋납니다).
        _bwd = float(getattr(c, "backward_ratio", 0.0))
        _n_bwd = int(round(self.num_envs * _bwd)) if _bwd > 0.0 else 0
        self._dir_fwd = torch.arange(self.num_envs, device=dev) < (self.num_envs - _n_bwd)
        self._dir_sign = torch.where(self._dir_fwd, 1.0, -1.0).unsqueeze(-1)   # (E,1) 속도 부호
        self._any_backward = bool(_n_bwd > 0)
        if self._any_backward:
            print(f"[backward-dir] 정방향 {self.num_envs - _n_bwd} / 역방향 {_n_bwd} "
                  f"({_n_bwd / self.num_envs:.0%})")
        self._friction_step_count: int = 0
        self._last_friction_mean: float = float(c.friction_max_init)
        self._last_friction_max: float = float(c.friction_max_init)
        self._late_gate = False
        self._late_gate_frames = int(round(float(c.late_gate_survival_frac) * self._ref_len))
        # [ROLLBACK MARKER: deferred-cache] staging buffer for the at-termination bulk commit.
        # See cfg.cache_min_episode_length. Rows are written per step by _save_state_cache and are
        # only merged into _state_cache by _flush_state_cache when the episode ENDS having lasted at
        # least cache_min_episode_length steps. Sized for a full episode because an episode that runs
        # to the end of the clip must still be committable. Allocated only when the feature is on.
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
        # [wrist-rot] 손 블록 배치: 손당 20개 = [0]손목 [1..4]검지 [5..8]중지 [9..12]약지 ...
        _nb_k = self._ref_kpts.shape[1] - 40                        # 몸 키포인트 수 (14)
        self._wrist_frame_idx = [(_nb_k + o, _nb_k + o + 5, _nb_k + o + 1) for o in (0, 20)]  # L, R
        self._failure_count = torch.zeros(self._ref_len, device=dev)
        # [backward-dir] 각 프레임의 캐시 항목이 역방향에서 왔는지 (마진 값 진단용)
        self._cache_from_bwd = torch.zeros(self._ref_len, dtype=torch.bool, device=dev)
        self._adaptive_back_frames = int(round(c.adaptive_back_seconds / c.ref_dt))
        self._sampling_step_count = 0
        # per-env tracking-quality streak (grasp mechanism): _enough_continued = has tracking been
        # continuously "good enough" since reset; _enough_idx = last good frame (drives cache write
        # gate + failure-weighted sampling). Reset per env in _reset_idx.
        self._enough_continued = torch.ones(self.num_envs, dtype=torch.bool, device=dev)
        self._enough_idx = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        # RSI start frame of the CURRENT episode (diagnostics only — _enough_idx drifts to the last
        # good frame, so it cannot stand in for the start once the episode is running).
        self._episode_start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        # [ROLLBACK MARKER: failure-sigma] 에피소드마다 리셋 때 한 번 정해지는 값들.
        #   _episode_pick = 실패 가중 샘플링이 겨냥한 프레임 (start 는 여기서 되감은 것)
        #   _episode_beta = 그 에피소드의 sigma 배율 (뽑힐 확률에 비례)
        # beta 는 [start, pick] 구간에서만 적용되므로 둘 다 필요합니다.
        self._episode_pick = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._episode_beta = torch.ones(self.num_envs, device=dev)
        # pretrain-cache warm-start (209-D pretrain cache)
        self._pretrain_cache = None
        self._pretrain_init_flg = None
        self._last_pretrain_fallback_ratio = 0.0

        # ---- FROZEN SONIC body prior (built on device; env_isaaclab + gear_sonic + vector_quantize) ----
        self._sonic = None
        if getattr(c, "use_sonic", True):
            import sys as _sys
            # scripts/process_dataset/sonic/sonic_prior.py. This file lives at
            # <repo>/source/robotis_sh5/robotis_sh5/tasks/direct/g1_shadow_sonic_residual/<this>.py
            # → the repo root is parents[6]. (The previous os.path.dirname chain stopped one level
            # short at <repo>/source, so the computed path never existed and the hard-coded fallback
            # below was doing all the work — i.e. it only ran on this machine.)
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
            self._sonic_perm = _SP.build_body_perm(list(self.robot.joint_names), device=dev)  # robot->SONIC (29)
            self._sonic_default = _SP.sonic_default_vector(dev).view(1, -1)                    # (1,29) SONIC order
            # ── [ROLLBACK MARKER: sonic-encoder-g1] ────────────────────────────────────────
            # SONIC의 인코더 중 무엇에게 명령을 줄지. 'smpl'은 사람 SMPL 관절 위치를 주고 SONIC이
            # 속으로 사람->로봇 변환을 합니다. 'g1'은 로봇 자신의 29개 관절 각도와 속도를 직접
            # 줍니다. 손목 회전은 사람 관절 위치에 거의 안 담기는 양이라, smpl 모드에서는 손목 6개를
            # 별도 통로로 넣어야 하고 그런데도 SONIC이 잘 못 따라갑니다(실측: 오른손 오차가 에피소드
            # 나이에 따라 0.33 -> 1.09 rad로 증가). g1 모드는 그 변환 단계 자체가 없습니다.
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
                # [ROLLBACK MARKER: ref-reset-jvel] 리셋에 쓰는 65열 _ref_joint_vel 과 여기(29열,
                # SONIC 순서)가 같은 양임을 못 박습니다. 어긋나면 시뮬레이터의 관절 속도와 SONIC이
                # 듣는 속도 이력이 달라지는데(그게 바로 없애려는 결함), 조용히 어긋나면 못 찾습니다.
                # _ref_hist["jvr"]도 같은 값입니다 — _sonic_default 상수가 차분에서 상쇄되므로.
                if self._ref_joint_vel is not None:
                    _dv = (self._ref_joint_vel[:, _idx29] - self._ref_g1_v).abs().max()
                    assert _dv < 1e-3, f"[ref-reset-jvel] reset jvel != SONIC jvr: max diff {_dv:.3e}"
            self._sonic_scale = _SP.sonic_scale_vector(dev).view(1, -1)                        # (1,29) SONIC order
            # [ROLLBACK MARKER: hist-from-reference] 리셋 때 SONIC의 10프레임 관측 창을 채울 재료.
            # 프레임마다 "레퍼런스가 그 시점에 어떤 상태였는가"를 SONIC이 받는 형식 그대로 미리 만들어
            # 둡니다. 창을 현재 행 복제로 채우면 "로봇은 10프레임 동안 안 움직였다"가 되는데, 같은 순간
            # 토크나이저는 레퍼런스가 움직이는 중이라고 말합니다 — 두 이야기가 어긋납니다. 레퍼런스의
            # 지난 10프레임으로 채우면 과거와 미래가 같은 궤적이 됩니다.
            # jvr은 레퍼런스 관절의 차분이라 0번 프레임이 0인데, 그게 맞습니다 — 클립 시작에서
            # 레퍼런스는 실제로 정지해 있으므로 0으로 잘린 창이 자기모순이 없습니다.
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
            # 10-frame proprio history (term-major, oldest-first) = playback flat_proprio layout
            self._sonic_hist = {k: torch.zeros(self.num_envs, _SP.PROPRIO_HIST, d0, device=dev)
                                for k, d0 in [("ang", 3), ("jpr", 29), ("jvr", 29), ("act", 29), ("grav", 3)]}
            self._sonic_hist_init = torch.ones(self.num_envs, dtype=torch.bool, device=dev)
            self._last_a_sonic = torch.zeros(self.num_envs, 29, device=dev)
            self._last_z_res = torch.zeros(self.num_envs, int(c.sonic_action_dim), device=dev)
            # [DELTA-ACTION] optional per-step INCREMENT integrators (default OFF via cfg). z_res delta →
            # integrated latent residual (clamped ±clip); hand delta → integrated hand JOINT target
            # (clamped to limits). Seeded at reset (z→0, hand→reset pose) so delta=0 holds the reset state.
            self._z_res_int = torch.zeros(self.num_envs, int(c.sonic_action_dim), device=dev)
            self._z_delta_ema = torch.zeros(self.num_envs, int(c.sonic_action_dim), device=dev)
            _hsl = self._sonic_hand_slice
            self._hand_delta_target = default_q[:, _hsl].clone()         # (E,36) hand JOINT target (delta integrator)
            self._hand_delta_ema = torch.zeros(self.num_envs, int(c.hand_action_dim), device=dev)
            # RAW 100-D policy action (z_res + a_hand) for the GRAIL-style obs prev_action term AND the
            # action_rate reward. _cur = this step's action (set in _sonic_pre_physics_step); _prev = the
            # previous step's (lag-1, updated at the end of _get_observations). Seeded 0 at reset.
            _pa_dim = int(c.sonic_action_dim) + int(c.hand_action_dim)   # 100
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

    # ---- action: SONIC latent-residual body + absolute-EMA hands (per-group EMA/delta = fallback) ----
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # advance the reference frame for this step (reset overrides it at episode start)
        self._frame_idx = (self._frame_idx + 1).clamp(max=self._ref_len - 1)
        c = self.cfg
        if self._sonic is not None:                       # SONIC latent-residual body + hand residual
            self._sonic_pre_physics_step(actions)
            return
        a = actions.clamp(-1.0, 1.0)
        # per-group absolute EMA for all groups (baseline; feeds obs prev_action + action regularizers).
        for gname, sl in self._group_slices.items():
            al = self._group_alpha[gname]
            self._smoothed_actions[:, sl] = al * a[:, sl] + (1.0 - al) * self._smoothed_actions[:, sl]

        if c.residual_action and self._ref_joints is not None:
            # PER-STEP RESIDUAL: target = clamp(ref_joints[frame] + residual_scale · a, limits). No integration
            # (a=0 ⟺ exact reference playback); bounded to ±residual_scale about the reference by a∈[-1,1].
            # residual_scale is PER GROUP (body 0.25 / hands 0.50) via _residual_scale_t (1,65).
            # _delta_target mirrors the commanded target so obs prev_action / _apply_action reuse it unchanged.
            self._residual_target = torch.clamp(
                self._ref_joints[self._frame()] + self._residual_scale_t * a, self._ctrl_lower, self._ctrl_upper)
            for gname, sl in self._group_slices.items():
                self._delta_target[gname] = self._residual_target[:, sl]
        else:
            self._residual_target = None
            # rollback (residual_action off / no retarget): per-group free-running delta integrator; groups
            # with the delta switch OFF stay on the absolute EMA baseline computed above.
            for gname, sl in self._group_slices.items():
                on, scale, smooth = self._delta_cfg[gname]
                if not on:
                    continue
                dcmd = a[:, sl] * scale
                self._delta_ema[gname] = smooth * dcmd + (1.0 - smooth) * self._delta_ema[gname]
                self._delta_target[gname] = torch.clamp(self._delta_target[gname] + self._delta_ema[gname],
                                                        self._ctrl_lower[sl], self._ctrl_upper[sl])

    def _apply_action(self) -> None:
        if self._residual_target is not None:                          # per-step residual: ref[frame] + scale·a
            target = self._residual_target
        else:                                                          # rollback: abs EMA + per-group delta
            target = self._scale(self._smoothed_actions)               # (E,65) absolute EMA baseline
            for gname, sl in self._group_slices.items():
                if self._delta_cfg[gname][0]:                          # delta group → use integrated target
                    target[:, sl] = self._delta_target[gname]
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

    def _sonic_tokenizer(self) -> torch.Tensor:
        """(E,TOK) SONIC tokenizer obs in SMPL mode: encoder_index=[0,0,1] + the 10-frame future
        SMPL reference window (joints local, wrist ref, root orientation RELATIVE to the live
        pelvis). Non-SMPL terms are zero-filled. Matches sonic_playback.build_tok (verified)."""
        E, dev = self.num_envs, self.device
        lay = self._sonic_layout
        K = 10                                                          # SONIC multi-future window
        # [ROLLBACK MARKER: backward-dir] "미래"는 에피소드 시간 기준입니다. 역방향 환경에서는
        # 원본 프레임이 감소하는 쪽이 미래이므로 창을 뒤집습니다. 인덱스만 뒤집고 값은 그대로
        # 둡니다 — 관절 위치와 손목 관절값은 자세라서 부호를 바꾸면 안 됩니다.
        _step = self._dir_sign.to(torch.long) if self._any_backward else 1
        idx = (self._rframe().unsqueeze(1)
               + _step * torch.arange(K, device=dev).unsqueeze(0)).clamp(0, self._ref_len - 1)
        tok = torch.zeros(E, self._sonic_tok_dim, device=dev)
        # [ROLLBACK MARKER: sonic-encoder-g1] encoder_index 열 순서 = m.encoders = ['g1','teleop','smpl'].
        if self._sonic_enc == "g1":
            s_, e_, _ = lay["encoder_index"]; tok[:, s_:e_] = torch.tensor([1.0, 0.0, 0.0], device=dev)
            # 평탄 580 = [pos(f..f+9) 29개] ++ [vel(f..f+9) 29개]. 프레임당 [pos|vel]가 아닙니다
            # (디코더가 푸는 방식과 commands.py의 cat([joint_pos_mf, joint_vel_mf])에서 확인).
            s_, e_, _ = lay["command_multi_future_nonflat"]
            tok[:, s_:e_] = torch.cat([self._ref_g1_q[idx].reshape(E, -1),
                                       self._ref_g1_v[idx].reshape(E, -1)], dim=-1)
            pelvis_q = _canon(self.robot.data.root_quat_w)
            rq = self._sonic_root_q[idx]
            dif = self._sonic_qmul(self._sonic_qinv(pelvis_q).unsqueeze(1).expand(E, K, 4), rq)
            ori6 = math_utils.matrix_from_quat(dif.reshape(-1, 4))[..., :2].reshape(E, K, 6)
            s_, e_, _ = lay["motion_anchor_ori_b_mf_nonflat"]; tok[:, s_:e_] = ori6.reshape(E, -1)
            return tok
        s, e, _ = lay["encoder_index"]; tok[:, s:e] = torch.tensor([0.0, 0.0, 1.0], device=dev)
        s, e, _ = lay["smpl_joints_multi_future_local_nonflat"]; tok[:, s:e] = self._sonic_smpl_j[idx].reshape(E, -1)
        s, e, _ = lay["joint_pos_multi_future_wrist_for_smpl"]; tok[:, s:e] = self._sonic_wrist_ref[idx].reshape(E, -1)
        pelvis_q = _canon(self.robot.data.root_quat_w)                  # (E,4) live pelvis
        rq = self._sonic_root_q[idx]                                    # (E,10,4) reference root (Z-up)
        dif = self._sonic_qmul(self._sonic_qinv(pelvis_q).unsqueeze(1).expand(E, K, 4), rq)  # (E,10,4) robot-relative
        ori6 = math_utils.matrix_from_quat(dif.reshape(-1, 4))[..., :2].reshape(E, K, 6)
        s, e, _ = lay["smpl_root_ori_b_multi_future"]; tok[:, s:e] = ori6.reshape(E, -1)
        return tok

    def _sonic_pre_physics_step(self, actions: torch.Tensor) -> None:
        """Body(29) = frozen SONIC decoder with a pre-quantization latent residual z_res(64);
        hands(36) = ABSOLUTE action (user-locked) mapped directly to the Shadow joint range,
        per-group EMA-smoothed. Sets the combined 65-D PD target and mirrors it into
        _smoothed_actions / _delta_target so obs prev_action + the RSI state cache stay unchanged
        (kept 65-D → pretrain→train transfer stays valid)."""
        c = self.cfg
        E, dev = self.num_envs, self.device
        z_raw = actions[:, :c.sonic_action_dim]                          # (E,64) raw policy latent output
        if c.sonic_latent_delta:                                        # [DELTA] integrate latent increments
            self._z_delta_ema = (c.sonic_latent_delta_smoothing * (z_raw * c.sonic_latent_delta_scale)
                                 + (1.0 - c.sonic_latent_delta_smoothing) * self._z_delta_ema)
            self._z_res_int = torch.clamp(self._z_res_int + self._z_delta_ema,
                                          -c.sonic_latent_delta_clip, c.sonic_latent_delta_clip)
            z_res = self._z_res_int
        else:
            # ABSOLUTE (default): raw residual, CLIPPED to [-z_res_clip, z_res_clip] (user 2026-07-23,
            # =5.0) so the frozen-SONIC decoder never sees extreme latents (bounds the physical body
            # residual; NOTE this bounds the ENV action, not the PPO log-prob which is on the raw sample).
            z_res = torch.clamp(z_raw, -c.sonic_z_res_clip, c.sonic_z_res_clip)
        a_hand = actions[:, c.sonic_action_dim:].clamp(-1.0, 1.0)        # (E,36)
        # saturation diagnostics: what FRACTION of each block's raw sample is being flattened by the
        # env clamp. Both blocks have a flat exterior (no restoring gradient there), so a rising
        # fraction is the early signature of a mean random-walk. Logged under "Diag /".
        self._diag_hand_clamp_frac = (actions[:, c.sonic_action_dim:].abs() > 1.0).float().mean()
        self._diag_zres_clip_frac = (z_raw.abs() > c.sonic_z_res_clip).float().mean()
        # rew_action_reg operand (잠재 블록): the RAW (UNCLIPPED) latent residual — pre-2026-07-28
        # form. Penalizing the clipped value leaves the exterior of the clip perfectly flat, so nothing
        # pulls mu back inside once a dim saturates; the raw form keeps that restoring gradient (and
        # makes the term unbounded, which is the accepted cost of the rollback).
        #   DELTA mode exception: there z_raw is a per-step INCREMENT, not a residual magnitude, so
        #   squaring it would silently turn action_reg into a velocity penalty. Keep the integrated
        #   target in that mode (sonic_latent_delta is False by default, so absolute/raw is what runs).
        self._last_z_res = z_res if c.sonic_latent_delta else z_raw
        self._cur_policy_action = actions                                # raw 100-D policy action (obs + action_rate)
        # ENV-EFFECTIVE (clipped z_res / clamped hand) copy, each block normalized by its own bound so
        # every entry is in [-1,1]. Fed rew_action_rate between 2026-07-28 and the same-day rollback;
        # currently UNREAD — kept (and still maintained) as the one-line A/B switch for that term, since
        # this axis is under active comparison. `_diag_*_clamp_frac` above do NOT depend on it.
        _zb = max(float(c.sonic_z_res_clip), 1e-6)
        self._cur_policy_action_bnd = torch.cat([z_res / _zb, a_hand], dim=-1)
        # frozen SONIC body: encode SMPL ref -> latent +λ·z_res (pre-quant) -> FSQ -> g1_dyn decode
        proprio = self._sonic_proprio()                                 # (E,930)
        tok = self._sonic_tokenizer()                                   # (E,TOK)
        latent = self._SP.encode_latent(self._sonic, tok, encoder=self._sonic_enc)
        a_sonic = self._SP.residual_decode(self._sonic, latent, z_res, proprio,
                                           float(c.residual_scale_latent))   # (E,29) SONIC order, raw
        self._last_a_sonic = a_sonic
        body_sonic = self._sonic_default + self._sonic_scale * a_sonic  # (E,29) SONIC order absolute target
        body_target = body_sonic[:, self._sonic_gather]                # (E,29) action-body order
        # hands: ABSOLUTE action (user-locked decision) — a_hand ∈ [-1,1] maps DIRECTLY to the
        # Shadow joint range (NOT residual-on-retarget, NOT delta). Per-group EMA-smoothed like the
        # grasp env's absolute finger path; the EMA prev is _smoothed_actions[hands] (seeded to the
        # reset pose in _reset_idx). residual_scale_hands is UNUSED in this mode.
        hsl = self._sonic_hand_slice
        if c.sonic_hand_delta:                                          # [DELTA] integrate hand JOINT increments
            self._hand_delta_ema = (c.sonic_hand_delta_smoothing * (a_hand * c.sonic_hand_delta_scale)
                                    + (1.0 - c.sonic_hand_delta_smoothing) * self._hand_delta_ema)
            self._hand_delta_target = torch.clamp(self._hand_delta_target + self._hand_delta_ema,
                                                  self._ctrl_lower[hsl], self._ctrl_upper[hsl])
            hand_target = self._hand_delta_target
        else:                                                           # ABSOLUTE (default): a_hand → joint range (EMA)
            alpha_h = self._group_alpha["hands"]
            smoothed_hand = alpha_h * a_hand + (1.0 - alpha_h) * self._smoothed_actions[:, hsl]
            hand_target = (self._ctrl_lower[hsl]
                           + 0.5 * (smoothed_hand + 1.0) * (self._ctrl_upper[hsl] - self._ctrl_lower[hsl]))
        target = torch.empty(E, self._n_act, device=dev)
        target[:, :hsl.start] = body_target                            # [0:29] = body (legs+waist+arms, SONIC)
        target[:, hsl] = hand_target                                   # [29:65] = bimanual hands (ABSOLUTE)
        self._residual_target = torch.clamp(target, self._ctrl_lower, self._ctrl_upper)
        self._smoothed_actions = self._unscale(self._residual_target)
        for gname, sl in self._group_slices.items():
            self._delta_target[gname] = self._residual_target[:, sl]

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

    # ── 역방향 롤아웃 [ROLLBACK MARKER: backward-dir] ──────────────────────────────────
    # _frame()은 에피소드 진행도로, 방향과 무관하게 0부터 올라갑니다. 레퍼런스 배열을 읽을 때는
    # 반드시 _rframe()을 써야 합니다 — 역방향 환경은 원본 프레임을 거꾸로 훑기 때문입니다.
    # 레퍼런스 배열 접근이 40곳 넘지만 프레임을 뽑는 진입점이 여기 둘뿐이라, 호출 지점만 바꾸면
    # 전부 따라옵니다. 역방향 환경이 없으면 항등 함수라 기존 동작과 완전히 같습니다.
    def _rframe(self, f: torch.Tensor | None = None) -> torch.Tensor:
        f = self._frame() if f is None else f
        if not self._any_backward:
            return f
        return torch.where(self._dir_fwd, f, (self._ref_len - 1) - f)

    def _rnext_frame(self) -> torch.Tensor:
        return self._rframe(self._next_frame())

    # ------------------------------------------------------------ observation
    def _get_observations(self) -> dict:
        c = self.cfg
        E, vs = self.num_envs, c.vel_obs_scale
        org = self.scene.env_origins                                    # (E,3)
        fr, nfr = self._rframe(), self._rnext_frame()   # [backward-dir] 원본 프레임

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
        # object-LOCAL target→fingertip offset: fingertip's target relative to the fingertip, expressed in
        # the OBJECT frame → object-pose-invariant contact signal. Sign is ref − current (target_w − _tip),
        # matching every other delta obs in this env (delta_kpts/delta_root/delta_obj = ref − robot); the
        # grasp env uses the opposite current − ref, so we flip here to stay internally consistent. Target =
        # the Option-A DISTAL-link object-surface contact point on expected contact (per-link mask), else the
        # reference fingertip pad. obj_q is LIVE (train) / REFERENCE (pretrain) → same phase as other obj obs.
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
            # PREV policy action (z_res 64 + a_hand 36 = 100), GRAIL-style — the RAW action.
            #   The ENV-EFFECTIVE (clipped, per-block normalized to [-1,1]) copy `_prev_policy_action_bnd`
            #   was tried here on 2026-07-28 on the hypothesis that the unbounded raw action was driving
            #   the RunningStandardScaler statistics and thus the ratio pathology (skrl refreshes that
            #   scaler INSIDE the update — ppo.py:386 `train=not epoch` — so the same raw obs normalizes
            #   differently at rollout time vs update time). MEASURED AND DISPROVED: the frozen-policy
            #   dead state was unchanged (76/80 logged analytic-KL points exactly 0 both before and
            #   after), so the bounded obs bought nothing and was ROLLED BACK to keep the obs faithful
            #   to what the policy emitted. The real chain is the 100-D joint log-ratio feeding the
            #   `A<0` branch of `-min(A·r, A·clip(r))`, which is unbounded for r ≫ 1+ε.
            #   NOTE the bounded copy is still used for the `action_rate` reward (_get_rewards) — that
            #   change was independent and stands; only this obs slot reverted.
            self._prev_policy_action,
        ]

        # [ROLLBACK MARKER: backward-dir] 진행 방향 비트. 시간 역전은 동역학적으로 유효하지 않아
        # (중력은 안 뒤집힘) 두 방향은 사실상 다른 과제입니다. 이 비트가 없으면 정책이 같은 관측에
        # 두 행동을 평균내게 되고, repho 쪽 실험에서 최고 성능이 51.80 -> 21.69로 반토막 났습니다.
        C = C + [(~self._dir_fwd).float().unsqueeze(-1)]
        # [ROLLBACK MARKER: failure-sigma] beta 를 관측 마지막 열로. 신경망 입력에서는 잘라내고
        # (정책 패치가 [:, :-1] 만 씁니다) sigma 배율로만 쓰이므로, mu 는 beta 를 보지 않습니다.
        # 관측에 싣는 이유는 선호가 아니라 구조입니다 — skrl 의 PPO.update 는 미니배치를 고정
        # 7-튜플로 언패킹하고 정책에 observations/states 만 전달하므로, 별도 메모리 텐서를 쓰면
        # update 전체를 복제해야 합니다. 관측은 자동 저장되어 업데이트 때 샘플 정렬이 공짜입니다.
        if self._fs_on:
            if float(c.failure_sigma_global) > 0.0:          # 전역 대조군: 모든 스텝 동일
                beta_now = torch.full((E, 1), float(c.failure_sigma_global), device=self.device)
            else:
                _in = (fr >= self._episode_start_frame) & (fr <= self._episode_pick)
                beta_now = torch.where(_in, self._episode_beta, torch.ones_like(self._episode_beta))
                beta_now = beta_now.unsqueeze(-1)
            self._beta_now = beta_now                        # 로깅/덤프용
            C = C + [beta_now]
        obs = torch.cat(A + B + C, dim=-1)
        assert obs.shape[-1] == c.observation_space, (
            f"obs dim {obs.shape[-1]} != cfg.observation_space {c.observation_space} "
            "(block-C dims must be invariant across the has_object flip)")
        # capture prev actions for NEXT step (lag-1). The RAW copy (_*_policy_action) feeds the obs
        # (above) and the drift diagnostics (Diag/zres_absmax, Diag/hand_absmax); the BOUNDED copy
        # (_*_bnd) feeds action_rate only. Both are maintained so either can be swapped in.
        # _prev_action (realized 65-D) is retained only for the inherited non-SONIC delta rollback
        # path (dead in this SONIC-only env).
        self._prev_policy_action = self._cur_policy_action.clone()
        self._prev_policy_action_bnd = self._cur_policy_action_bnd.clone()
        self._prev_action = self._smoothed_actions.clone()
        for gname, sl in self._group_slices.items():
            if self._delta_cfg[gname][0]:                              # delta group → normalized integrated target
                self._prev_action[:, sl] = self._unscale_slice(gname, self._delta_target[gname])

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
                # [ROLLBACK MARKER: hist-from-reference] -------------------------------------------
                # 창을 레퍼런스의 지난 10프레임으로 채웁니다. 그러면 창 자체가 실재하는 궤적이고,
                # 토크나이저가 미래로 주는 것과 같은 궤적이라 SONIC이 하나의 일관된 이야기를 봅니다.
                # 인덱스는 에피소드 시간으로 만들고 0에서 자릅니다. 이 한 줄이 두 경우를 같이 처리합니다
                # — 클립 시작(10프레임이 없으면 0번을 반복, 거기서는 레퍼런스가 실제로 정지 상태),
                # 그리고 역방향 에피소드(에피소드의 과거가 원본에서는 앞쪽 프레임).
                # 속도 두 채널은 역방향에서 부호를 뒤집습니다(리셋 경로와 동일). 위치·자세는 부호를
                # 바꾸면 안 되므로 그대로 둡니다.
                # 10칸을 전부 레퍼런스로 덮습니다(가장 최근 칸 포함). 최근 칸만 실측으로 남기면 추종
                # 오차가 8번과 9번 칸 사이의 위치 점프로 들어가고, 속도 채널이 그걸 부정합니다 —
                # 여기서 없애려는 결함 그 자체입니다.
                _hist_ref_used = (bool(getattr(c, "sonic_hist_from_reference", True))
                                  and self._ref_hist is not None)
                if _hist_ref_used:
                    _H = self._sonic_hist["jpr"].shape[1]                       # 10
                    _ep = self._frame().unsqueeze(1) - torch.arange(_H - 1, -1, -1, device=self.device)
                    _ep = _ep.clamp(min=0)                                      # (E,H) 에피소드 프레임
                    # 에피소드 프레임 -> 원본 프레임. _rframe과 같은 사상이되 (E,H)로 방송합니다.
                    _of = (_ep if not self._any_backward
                           else torch.where(self._dir_fwd.unsqueeze(1), _ep, (self._ref_len - 1) - _ep))
                    _sg = self._dir_sign.unsqueeze(-1)                          # (E,1,1) 속도 부호
                    for k in ("jpr", "grav", "act"):
                        self._sonic_hist[k][m] = self._ref_hist[k][_of][m]
                    for k in ("jvr", "ang"):
                        self._sonic_hist[k][m] = (self._ref_hist[k][_of] * _sg)[m]
                else:
                    for k in ("ang", "jpr", "jvr", "grav"):
                        self._sonic_hist[k][m] = rows[k][m].unsqueeze(1)
                # 복제 방식일 때만 의미가 있습니다. 위치를 10칸에 얼려놓고 살아있는 속도를 같이 넣으면
                # "10프레임 동안 안 움직였는데 지금 움직이고 있다"가 되어 디코더가 학습 중 본 적 없는
                # 이력이 됩니다. 속도를 0으로 두면 "10프레임 동안 정지"가 되어 자기모순이 없어집니다.
                if not _hist_ref_used:
                    if getattr(c, "sonic_hist_seed_zero_vel", False):
                        self._sonic_hist["jvr"][m] = 0.0
                    # 행동 이력: 0은 "지난 10프레임 동안 기본자세를 명령해 왔다"는 뜻이라 조작 도중
                    # 복원된 로봇과 어긋납니다. 다만 IsaacLab의 행동 관리자가 리셋 때 행동을 0으로
                    # 만들므로, 동결 디코더가 학습 중 실제로 본 값은 0입니다. 그래서 0이 기본입니다.
                    if getattr(c, "sonic_act_seed_from_pose", False):
                        self._sonic_hist["act"][m] = (rows["jpr"][m] / self._sonic_scale).unsqueeze(1)
                    else:
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

    def _unscale_slice(self, gname: str, q: torch.Tensor) -> torch.Tensor:
        sl = self._group_slices[gname]
        lo, hi = self._ctrl_lower[sl], self._ctrl_upper[sl]
        return 2.0 * (q - lo) / (hi - lo) - 1.0

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

    # ── 손목 회전 [ROLLBACK MARKER: wrist-rot] ─────────────────────────────────────────
    # 사람 손과 로봇 손에 "같은 방법으로" 좌표계를 세워 비교합니다. SMPL-X 손목 관절 회전과 Shadow
    # palm 바디 자세를 직접 비교하면 축 규약이 달라 보정값을 손으로 찾아야 하고, 그 값이 틀리면
    # 상수 오차가 생겨 프레임 0에서 종료가 터집니다(grasp 쪽에서 겪은 문제). 대신 키포인트 기하로
    # 좌표계를 정의하면 모델 내부 규약과 무관해집니다:
    #     z = 손목 -> 중지 MCP,  x = z x (손목 -> 검지 MCP) (손바닥 법선),  y = z x x
    # 레퍼런스 자세에서 실측한 양쪽 좌표계의 차이는 오른손 11.8도(퍼짐 p50 5.8), 왼손 32.5도
    # (퍼짐 p50 6.4)입니다. 손 비율 차이에서 오는 고정 회전이고, 보정 없이 쓰기로 했습니다.
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
        fr = self._rframe()             # [backward-dir] 원본 프레임
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
            # contact-conditioned + drift-compensated fingertip REWARD target (ported from grasp_rsi):
            # in-contact fingers → the object-surface contact VERTEX in the LIVE object frame (snaps the
            # tip ONTO the object, not the human pad which sits ~3.6cm beside a thin handle); non-contact
            # fingers → the ref pad re-expressed in the live object frame (drift-comp, follows the object).
            # (rew_fingertip uses this ft_reward; termination/logging keep the raw ft above.)
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
            # ── [ROLLBACK MARKER: link-kpt] 손끝 외 wrap 링크도 같은 방식으로 ─────────────
            # 손끝 10개만으로는 손 방향이 거의 안 묶입니다 — 손목을 60도 돌려도 손끝을 목표 근처에
            # 놓는 자세가 여럿입니다. 반면 뿌리/중간 마디는 한 방향에서만 자기 목표에 닿으므로,
            # 이 항이 실제로 자세의 자유도를 묶습니다. 목표는 로봇 자신의 레퍼런스 자세에서 온
            # 점이라 도달 가능합니다(사람 손 위치를 쓰는 rew_hand_kpts와 달리 물체 안쪽을
            # 가리키지 않습니다).
            # [ROLLBACK MARKER: link-kpt-objframe] 목표 = 물체 기준으로 표현한 레퍼런스 링크 위치를
            # 살아있는 물체 자세로 되돌린 것. 물체가 돌면 손이 따라가야 할 자세도 함께 돕니다.
            # 접촉 요구와 무관하게 매 프레임 정의되므로 접촉 마스크로 걸지 않습니다(손바닥만 제외).
            _lp = self.robot.data.body_pos_w[:, self._link_contact_body_ids] - org[:, None, :]  # (E,L,3)
            _oqL = obj_quat.unsqueeze(1).expand(-1, _lp.shape[1], -1)
            _ltgt = math_utils.quat_apply(_oqL, self._ref_link_kpt_local[fr]) + obj_pos.unsqueeze(1)
            _act = (self._link_kpt_use & self._link_kpt_has_ref.unsqueeze(0)
                    ).expand(self.num_envs, -1)                                     # (E,L) [link-kpt-smpl]
            _d = (_ltgt - _lp).norm(dim=-1) * _act.float()                          # (E,L)
            link_kpt_err = _d.sum(dim=-1) / _act.float().sum(dim=-1).clamp(min=1.0)  # (E,)
        else:
            obj_pos_err = torch.zeros(self.num_envs, device=self.device)
            obj_pos_err_w = obj_pos_err                                          # [z-weight]
            obj_rot_err = torch.zeros(self.num_envs, device=self.device)
            ft_reward = ft_err                                                     # no object → raw pad target
            link_kpt_err = torch.zeros(self.num_envs, device=self.device)   # [link-kpt]
        # palm/wrist rotation deviation (bimanual, MEAN over both hands). Robot palm body quat vs the retarget
        # reference palm quat (same robot0_{l,r}_palm frame → direct compare, no landmark conversion).
        # [ROLLBACK MARKER: wrist-rot] 키포인트 기하로 세운 좌표계끼리 비교합니다. 리타게팅 npz에
        # g1_palm_quat이 없어 기존 경로는 계속 0을 내고 있었습니다(로그 Error/wrist_rot = 0.0000).
        wrist_rot_err = self._wrist_rot_err(ref, kpts)
        # [ee-split] `wrist` (reward) and `wrist_pos` (termination) are the SAME tensor, kept under both
        # names so each call site reads the one that matches its intent.
        return dict(link_kpt=link_kpt_err, body=body_err, body_core=body_core_err,
                    ee=ee_err, hand_w=hand_err_w, obj_pos_w=obj_pos_err_w,   # [z-weight]
                    com_support=self._com_support_err(), wrist_pos=wrist_pos_err, hand=hand_err, ft=ft_err, ft_reward=ft_reward, ft_per=ft_per, tip=tip,
                    pad_inward=pad_inward, root_pos=root_pos_err, root_rot=root_rot_err,
                    obj_pos=obj_pos_err, obj_rot=obj_rot_err, root_quat=root_quat,
                    wrist_rot=wrist_rot_err)

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        e = self._errs                                    # set by _get_dones (runs first each step)
        fr = self._rframe()             # [backward-dir] 원본 프레임

        # per-LINK contact FORCE (Option A / DexMachina): compressive object contact force on each of the 32
        # wrap links (force_matrix · the link's OWN inward pad normal), gated by (a) the reference per-link
        # contact MASK (which links SHOULD touch), (b) a SPATIAL gate — the robot link is near the prescribed
        # object-surface target (so a link touching the object at the WRONG spot is NOT rewarded), and (c) an
        # ORIENTATION gate — the link's inward pad normal is aligned (≤ contact_normal_gate_tol) with the
        # reference reaction normal, so touching with the WRONG face (e.g. back of the palm) is NOT rewarded.
        # Normalized ∈[0,1] over the active links. Full grasp WRAP (palm+phalanges+tips).
        link_force = self._link_contact_forces()                      # (E,L) compressive per link (on own face)
        # [ROLLBACK MARKER: force-fingertip-only] 손끝만 (기본). 마스크를 좁히면 아래의 near /
        # orient / lf / 분모 n_lc 가 전부 따라옵니다.
        # [ROLLBACK MARKER: contact-vel-gate] 물체가 정지한 프레임에서는 link_mask 가 전부 0 이 되어
        # 분자(lf)와 분모(n_lc)가 함께 사라진다 → force_rew = 0. grasp env 와 같은 처리다
        # (fforce_contact / n_contacts 를 동일한 게이트 플래그로 걸었다).
        link_mask = (self._ref_link_contact_mask[fr] * self._force_link_use.float()
                     * self._ref_obj_vel_gate[fr].unsqueeze(-1))                     # (E,L)
        if self._has_object and self._has_link_contact:
            oqL = self._object.data.root_quat_w[:, None, :].expand(-1, N_LINK_CONTACT, -1)   # (E,L,4) live
            tgt_w = (math_utils.quat_apply(oqL, self._ref_link_contact_target_local[fr])
                     + self._object.data.root_pos_w[:, None, :])                              # (E,L,3) world target
            lp = self.robot.data.body_pos_w[:, self._link_contact_body_ids]                   # (E,L,3) world link pos
            near = ((lp - tgt_w).norm(dim=-1) < c.contact_match_dist).float()                 # (E,L) spatial gate
            if c.use_contact_normal_gate:
                inward_w = self._link_pad_inward_w()                                          # (E,L,3) link face
                ref_dir_w = math_utils.quat_apply(oqL, self._ref_link_contact_normal_local[fr])  # (E,L,3) reaction
                cos = (inward_w * ref_dir_w).sum(dim=-1)                                      # (E,L) both unit → cosθ
                orient = (cos >= self._contact_normal_gate_cos).float()                       # (E,L) orientation gate
            else:
                orient = torch.ones_like(near)
        else:
            near = torch.zeros_like(link_mask)
            orient = torch.zeros_like(link_mask)
        lf = (link_force * link_mask * near * orient).clamp(min=0.0, max=c.contact_force_cap)  # force·mask·near·orient
        n_lc = link_mask.sum(dim=-1).clamp(min=1.0)                   # #active links (≥1 to avoid /0)
        force_rew = lf.sum(dim=-1) / (n_lc * c.contact_force_cap)     # (E,) mean of min(force,cap)/cap ∈ [0,1]

        # [ROLLBACK MARKER: cws-contact] 로봇 쪽 목록과 점수.
        # 접촉 위치와 법선을 센서에서 직접 받습니다. contact_pos_w는 이 링크와 물체 사이 접촉점들의
        # 평균 위치(월드), force_matrix_w는 물체가 링크에 가하는 법선 힘(월드)입니다. 링크 원점과
        # 실제 접촉점은 중앙값 2.7 cm 떨어져 있어서(물체 크기의 1/4) 모멘트 팔로 원점을 쓰면 안 되고,
        # 링크에 박아둔 고정 법선은 실제 접촉력 방향과 중앙값 74도 어긋나 원뿔 축으로 못 씁니다.
        # 접촉이 없는 쌍은 두 값 모두 NaN이라 마스크로 거른 뒤 0으로 채웁니다.
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
            # 물체 기준 좌표로. 물체가 회전하므로 반드시 살아있는 자세를 씁니다.
            # [ROLLBACK MARKER: cws-com] 모멘트 팔 기준을 body 원점에서 물리 COM 으로 바꿉니다
            # (논문 tracking_command.py:1745 object_com_position_and_wxyz_w = body_com_state_w).
            # 레퍼런스 sigma_h 도 _post_init_buffers 에서 같은 COM 프레임으로 계산했습니다.
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
            # [ROLLBACK MARKER: cws-diag] 실패 덤프와 텐서보드에 남길 per-env 점수. 접촉 링크 수도
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

        # NOTE: no articulation-DOF reward term. Objects always spawn as a single RigidObject (base
        # only) — no articulated USD is ever loaded — so the error was hard-coded to zeros and the
        # weighted term contributed exactly 0. The reference DOFs are still exposed in the obs
        # (`artic` block, filled from _ref_obj_dof); re-add the reward here together with the live
        # joint read when articulated objects actually spawn.

        # NOTE: root LINEAR/ANGULAR velocity error terms REMOVED from the reward — the proven
        # grasp reward has none (it is fixed-base), they are redundant finite-diffs of the
        # root_pos/root_ori that are already tracked, and the robot's raw angvel is very noisy at
        # reset (zero-action tumbling → ~16 rad/s spikes) which dominated the penalty. The clean
        # reference velocities are still used for RSI reset initial velocity (that is fine).

        # [ROLLBACK MARKER: reg-merge] action_reg: 잠재 64 + 손 36 을 하나의 SUM 으로 통일
        # (grasp 관행 — robotis_shadow_grasp_env.py:1603 `(actions**2).sum(dim=-1)`).
        # 잠재는 _last_z_res (절대 모드에서 _cur_policy_action[:, :64] 와 같은 값이고, delta 모드에서는
        # 적분된 잠재 — raw 증분을 제곱하면 속도 페널티가 되므로 그쪽을 써야 한다). 손은 raw 정책
        # 액션의 손 블록(클램프 이전)이라 클립 밖에서도 복원 기울기가 남는다.
        hsl = self._group_slices["hands"]
        _nz_reg = int(self.cfg.sonic_action_dim)
        if self._sonic is not None:
            _areg_src = torch.cat([self._last_z_res,
                                   self._cur_policy_action[:, _nz_reg:]], dim=-1)      # (E,100)
        else:
            _areg_src = self._cur_policy_action
        action_reg = (_areg_src ** 2).sum(-1)                                           # (E,) SUM
        # pose_reg_hands (HANDS only): pull achieved hand joints toward the DEFAULT (rest/neutral) pose — a
        # task-agnostic regularizer that keeps fingers out of extreme/unnatural configs and damps jitter.
        # NOT toward the retarget reference: the hands are already tracked by rew_hand_kpts + rew_fingertip
        # (inside the tracking_penalty group), so a retarget-target pose_reg_hands would merely duplicate
        # tracking in joint space with zero neutral-pose safety. Body is SONIC-driven so it is NOT regularized
        # here. Matches the grasp ancestor (jp - default_joint_pos) and TJ (dof_pos² toward the rest pose);
        # GRAIL/SONIC carry no pose regularizer at all (tracking owns the reference; only limit/rate/contact).
        _hand_ids = self._action_joint_ids_t[hsl]
        hand_ref = self.robot.data.default_joint_pos[:, _hand_ids]
        pose_reg_hands = ((self.robot.data.joint_pos[:, _hand_ids] - hand_ref) ** 2).sum(-1)
        # action_rate on the RAW 100-D policy action (z_res + a_hand). It is NOT the realized joint
        # target: penalizing that would penalize SONIC's OWN body tracking (the base's job); this
        # penalizes only the policy's residual+hand smoothness (GRAIL meta_action_rate_l2).
        #   The ENV-EFFECTIVE (clipped/clamped, per-block normalized) form ran between 2026-07-28 and
        #   the same-day ROLLBACK. It bounded the term by 4·100 (≤1.6 after the -0.004 weight), but it
        #   also made the exterior of the clip perfectly flat, removing the only restoring gradient on
        #   mu once a dim saturates. Raw is unbounded — a mu excursion measured -12.6 here (vs -0.05
        #   healthy), enough to pin the total onto the clamp(min=0) floor — so watch
        #   `Diag / zres_clip_frac` and `Episode_Reward / action_rate` together: that is the trade
        #   this rollback accepts.
        # action_rate: 100차원 전체의 스텝간 변화 제곱합 (분리 전과 비트 단위로 동일 — 두 가중치가
        # 같았으므로 -0.001·Σ(잠재) + -0.001·Σ(손) == -0.001·Σ(전체)).
        action_rate = ((self._cur_policy_action - self._prev_policy_action) ** 2).sum(-1)
        # [ROLLBACK MARKER: energy] 역학적 파워 Σ|τ·q̇| (허리+다리). applied_torque 는 implicit
        # actuator 가 매 제어 스텝 채우는 값이라 가장 최근 decimation 스텝의 토크를 담는다.
        # 정지 유지에서 0 이므로 중력 지지분이 자동 배제된다 (τ² 와의 결정적 차이).
        _etau = self.robot.data.applied_torque[:, self._energy_joint_ids]           # (E,15) N·m
        _eqd = self.robot.data.joint_vel[:, self._energy_joint_ids]                 # (E,15) rad/s
        energy = (_etau * _eqd).abs().sum(-1)                                       # (E,) W
        self._diag_energy = energy.mean()          # 가중 이전 원값(W) — _log_reward_terms 가 읽는다
        # CoM-over-support balance penalty (anti-fall): out-of-support excess (m), 0 when the CoM stays
        # over the feet (so it never penalizes the balanced reference) — grows only when the robot tips.
        com_support = e["com_support"]                    # (E,) ≥ 0
        # feet-contact-match (VideoMimic/BSTRO): FRACTION of feet whose actual contact matches the reference
        # schedule (mean over feet → [0,1], so both-match = 1.0 not 2). POSITIVE bonus. c*_ref precomputed
        # (PyRoki rule); c_actual from the ground-filtered foot force sensors.
        feet_match = (self._foot_contact_actual() == self._ref_foot_contact[self._frame()]).float().mean(-1)  # (E,)∈[0,1]

        alive = (~self._died).float()                     # _died set by _get_dones this step

        # ── BOUNDED reward (mirrors the proven grasp RSI structure) ──────────────────────
        # Group ALL imitation/tracking penalties and CLAMP the sum at -rew_alive, so an
        # alive-but-poorly-tracking step nets ≤0 after the alive bonus (no free survival reward).
        # fingertip force / regs stay OUTSIDE the clamp (grasp keeps its force+regs outside its
        # tracking_penalty too); total reward floored at 0.
        # [ROLLBACK MARKER: exp-tracking] 같은 추적 항의 두 형태.
        # LINEAR: sum of -w*err, clamped at -rew_alive. 클램프가 물리면 그 샘플의 추적 기울기가
        #   정확히 0 이 되어 12 cm 의 물체 오차와 40 cm 가 같은 점수를 받는다 (`Diag / clamp_frac`).
        # EXPONENTIAL: 항마다 w*exp(-err²/σ²), 각각 [0, w] 로 유계라 클램프가 불필요하고 모든 항이
        #   기울기를 유지한다. 포화가 합 하나가 아니라 항별이라 나쁜 손끝이 몸 신호를 침묵시키지
        #   않는다. 형태와 σ 는 SONIC 을 따른다 (gear_sonic rewards.py:442).
        self._exp_terms = None
        tracking_raw = None                       # exp 모드에는 클램프가 없어 정의되지 않는다
        if c.exp_tracking_reward:
            self._exp_terms = {k: self._exp_w[k] * torch.exp(-(e[self._exp_key[k]] ** 2) / self._exp_s2[k])
                               for k in self._exp_w}
            tracking_penalty = sum(self._exp_terms.values())     # POSITIVE, 이미 유계
            _alive_w = c.exp_rew_alive
        else:
            tracking_raw = (
                c.rew_body_kpts * e["body_core"]               # 9 CORE body kpts
                + c.rew_ee_kpts * e["ee"]                      # WRIST×2 + ANKLE×2 + TORSO  [wrist-into-ee]
                + c.rew_hand_kpts * e["hand_w"] + c.rew_fingertip * e["ft_reward"]   # [z-weight]
                + c.rew_link_kpts * e["link_kpt"]      # [link-kpt]
                + c.rew_root_pos * e["root_pos"] + c.rew_root_ori * e["root_rot"]
                + c.rew_obj_pos * e["obj_pos_w"] + c.rew_obj_rot * e["obj_rot"]      # [z-weight]
            )
            tracking_penalty = tracking_raw.clamp(min=-c.rew_alive)
            _alive_w = c.rew_alive
        reward = (
            _alive_w * alive
            + tracking_penalty
            + (c.rew_contact_force * force_rew if c.contact_reward_mode in ('force', 'both') else 0.0)
            + (c.rew_cws * cws_rew if c.contact_reward_mode in ('cws', 'both') else 0.0)
            + c.rew_action_reg * action_reg + c.rew_pose_reg_hands * pose_reg_hands
            + c.rew_action_rate * action_rate
            + c.rew_energy * energy                       # [energy] 허리+다리 Σ|τ·q̇|
            + c.rew_com_support * com_support             # anti-fall (outside the tracking clamp)
            + c.rew_feet_contact_match * feet_match       # feet-contact-match bonus (positive)
        ).clamp(min=0.0)
        # NON-FINITE CONTAINMENT (reward path): _get_dones runs BEFORE this, and a NaN env is reset
        # AFTER _get_rewards (DirectRLEnv step order) — so its OBS is recomputed clean, but its NaN
        # REWARD is still recorded this step and would poison GAE → PPO gradients → policy weights
        # (all envs blow up next rollout). map any non-finite reward to 0 (neutral) so the reset-on-
        # nonfinite gate can retire the env without leaving a NaN in the buffer. Keep this ON during
        # training; do NOT add an obs nan_to_num (the reset ordering already cleans obs, and masking
        # obs would hide the true NaN origin during localization).
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        self._save_state_cache(reward)                    # per-frame best-state RSI cache
        # per-term reward contributions (weighted) → the Episode_Reward Tensorboard group, which holds
        # ONE graph per reward TERM and nothing else. Tracking terms are logged PRE-clamp (individual
        # insight); the clamped group value + clamp_frac live in the "Diag /" tab, and the TOTAL is not
        # logged here at all (skrl's "Reward / Instantaneous reward (mean)" is the identical value).
        if self._exp_terms is not None:      # [ROLLBACK MARKER: exp-tracking]
            _track_rew = {"link_kpts": self._exp_terms["link_kpt"],
                          "body_kpts": self._exp_terms["body"],
                          "ee_kpts": self._exp_terms["ee"],
                          "hand_kpts": self._exp_terms["hand"], "fingertip": self._exp_terms["fingertip"],
                          "root_pos": self._exp_terms["root_pos"], "root_ori": self._exp_terms["root_rot"],
                          "obj_pos": self._exp_terms["obj_pos"], "obj_rot": self._exp_terms["obj_rot"]}
        else:
            _track_rew = {"link_kpts": c.rew_link_kpts * e["link_kpt"],
                          "body_kpts": c.rew_body_kpts * e["body_core"],
                          "ee_kpts": c.rew_ee_kpts * e["ee"],
                          "hand_kpts": c.rew_hand_kpts * e["hand_w"],
                          "fingertip": c.rew_fingertip * e["ft_reward"],
                          "root_pos": c.rew_root_pos * e["root_pos"],
                          "root_ori": c.rew_root_ori * e["root_rot"],
                          "obj_pos": c.rew_obj_pos * e["obj_pos_w"], "obj_rot": c.rew_obj_rot * e["obj_rot"]}
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
            "com_support": c.rew_com_support * com_support,
            "feet_contact_match": c.rew_feet_contact_match * feet_match,
        }
        self._log_reward_terms(e, tracking_penalty, tracking_raw, ep_rew, fr)
        return reward

    def _log_reward_terms(self, e, tracking_penalty, tracking_raw, ep_rew, fr):
        log = self.extras.setdefault("log", {})
        log.update({
            "Error / link_kpts": e["link_kpt"].mean(),
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
            "Curriculum / cache_bwd_frac": float(
                (self._cache_from_bwd & ~self._init_flg).float().sum()
                / (~self._init_flg).float().sum().clamp(min=1.0)),
            "Curriculum / rsi_start_ceiling": float(getattr(self, "_rsi_ceil_f", self._ref_len - 1)),
            "Curriculum / pretrain_fallback": self._last_pretrain_fallback_ratio,
            "Curriculum / cache_coverage": float((~self._init_flg).sum().item()) / self._ref_len,
        })
        # per-term reward contributions → Episode_Reward group. ONE graph per reward TERM and nothing
        # else: no group aggregate, no total ("Reward / Instantaneous reward (mean)" from skrl is the
        # identical value — its base agent records the env rewards BEFORE rewards_shaper), no
        # diagnostics (those go to "Diag /" below).
        for k, v in ep_rew.items():
            log[f"Episode_Reward / {k}"] = v.mean()
        # reward-shaping diagnostics → separate "Diag /" tab (cfg.log_reward_diag=False drops them).
        # The 9 tracking terms above are logged PRE-clamp, so once the clamp bites their sum no longer
        # equals what entered the reward — these three are the only window on that gap.
        if self.cfg.log_reward_diag:
            log["Diag / tracking_penalty"] = tracking_penalty.mean()
            # [ROLLBACK MARKER: failure-sigma] beta_sq_mean 은 전역 대조군의 beta_G = sqrt(이 값) 에
            # 쓰이므로 반드시 남겨야 합니다. active_frac 은 [start, pick] 구간이 전체 스텝의 몇 %인지.
            if getattr(self, "_beta_now", None) is not None:
                _b = self._beta_now
                log["Diag / beta_mean"] = _b.mean()
                log["Diag / beta_sq_mean"] = (_b ** 2).mean()
                log["Diag / beta_max"] = _b.max()
                log["Diag / beta_active_frac"] = (_b > 1.0).float().mean()
            # [ROLLBACK MARKER: cws-diag] 접촉 렌치 진단. cws_score 는 "지금 파지가 만들 수 있는
            # 렌치가 레퍼런스가 요구하는 렌치를 얼마나 담는가" (1에 가까울수록 충분). cws_nhit 은
            # 실제로 힘을 받고 있는 링크 수 — 점수가 낮을 때 "렌치 부족"인지 "안 닿음"인지 가릅니다.
            if getattr(self, "_cws_per_env", None) is not None:
                log["Diag / cws_score"] = self._cws_per_env.mean()     # 원 점수 (v 보정 확인용)
                log["Diag / cws_coverage"] = self._cws_cov.mean()      # 요구 방향 중 충족 비율 ∈[0,1]
                log["Diag / cws_deficit"] = self._cws_def.mean()       # 방향당 평균 부족분
                log["Diag / cws_nhit"] = self._cws_nhit.mean()
            if self._exp_terms is None:
                # 클램프 '이전' 값으로 잰다. clamp 후의 값을 비교하면 (결과가 -rew_alive 미만일 수
                # 없으므로) clamp_frac 이 구조적으로 항상 0 이 된다 — 2026-09-01 에 고친 버그.
                log["Diag / tracking_penalty_raw"] = tracking_raw.mean()
                log["Diag / clamp_frac"] = (tracking_raw < -self.cfg.rew_alive).float().mean()
            else:
                # [ROLLBACK MARKER: exp-tracking] 잴 클램프가 없다. 대신 중요한 건 포화도다:
                # term/w ∈ [0,1]. 1 에 가까우면 σ 가 느슨해 항이 상수라 아무것도 안 가르치고,
                # 0 에 가까우면 너무 조여 항이 죽는다. 0.3~0.7 이 실제로 학습을 이끄는 대역이고,
                # σ 는 이 로그로 튜닝한다.
                for _k, _v in self._exp_terms.items():
                    log[f"Sat / {_k}"] = _v.mean() / max(self._exp_w[_k], 1e-9)
        # RSI / episode-length diagnostics. Episodes are variable length now (start frame → sequence
        # end), so skrl's "Reward / Total reward" (episode SUM) is length-confounded and no longer
        # comparable across configs — these two make the length distribution itself observable.
        # episode_len = steps elapsed in the CURRENT episode; rsi_start = frame the episode began at.
        if hasattr(self, "_diag_energy"):                # [energy] 가중 이전 원값 (W)
            log["Diag / energy"] = self._diag_energy
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
                      "wrist_rot": ((e["wrist_rot"] > cc.term_wrist_rot_err) if self._has_palm_ref
                                    else torch.zeros_like(dead))}
            # [ROLLBACK MARKER: body-kpt-off] 루트 낙상 게이트가 실제 게이트일 때만 원인으로 집계
            # (기준선에서는 게이트가 아니므로 넣으면 허위 귀속이 된다).
            if not cc.body_kpt_supervision:
                _cause["root_pos"] = e["root_pos"] > cc.term_root_pos_err
                _cause["root_rot"] = e["root_rot"] > cc.term_root_rot_err
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
        # [ROLLBACK MARKER: body-kpt-off] body 게이트를 끈 실험에서는 낙상/루트 이탈을 루트 게이트가
        # 잡는다(기준 = 리타게팅된 로봇 골반 g1_root_pose, e["root_pos"/"root_rot"]은 이미 매 스텝
        # 계산됨). 감독이 켜진 기준선에서는 body 게이트가 이를 포섭하므로(위 도크스트링) 비활성 —
        # 기준선의 거동은 불변.
        if not c.body_kpt_supervision:
            d = d | (e["root_pos"] > c.term_root_pos_err) | (e["root_rot"] > c.term_root_rot_err)
        #  fingertip + wrist-POSITION deviation (ft = mean over all 10 pads; wrist_pos = mean over both
        #  wrists — plain means, matching body). The finger-chain (hand) keypoint mean is NOT a gate — it is
        #  REWARD-only (rew_hand_kpts), matching grasp (which never terminates on its keypoint mean). A
        #  prior misport terminated on the hand-chain mean AND dropped the wrist-position gate; corrected.
        d = d | (e["ft"] > c.term_ft_err) | (e["wrist_pos"] > c.term_wrist_pos_err)
        #  wrist/palm rotation deviation (mirrors grasp max_wrist_rot_err) — only when a retarget
        #  palm-orientation reference is present (else wrist_rot is all-zero → inert).
        if c.enable_wrist_rot_termination and self._has_palm_ref:
            d = d | (e["wrist_rot"] > c.term_wrist_rot_err)
        #  object (mirrors grasp obj_pos + obj_rot) — only when an active object is present.
        if self._has_object:
            d = d | (e["obj_pos"] > c.term_obj_pos_err) | (e["obj_rot"] > c.term_obj_rot_err)
        if not c.termination:
            d = torch.zeros_like(d)
        # GRACE PERIOD: suppress the deviation death for the first termination_grace_frames steps of each
        # episode (episode_length_buf < N), so the policy has time to correct the reset pose before a
        # tracking gate can kill it (born-dead avoidance). Does NOT affect the non-finite gate below.
        if c.enable_termination_grace and c.termination_grace_frames > 0:
            d = d & (self.episode_length_buf >= c.termination_grace_frames)
        # NON-FINITE CONTAINMENT (always on, even when termination is disabled). A NaN robot state
        # (physics blow-up → degenerate/zero quat → matrix_from_quat's 2/‖q‖² → NaN, self-sustaining
        # through the SONIC proprio feedback) is NEVER caught by the `err > thresh` gates above because
        # `NaN > x` is False — so a poisoned env would survive to timeout and leak NaN into the SHARED
        # obs RunningStandardScaler and PPO gradients (whole-run blow-up = "assets disappear at once").
        # Force-reset any env whose root/joint state is non-finite so the poison cannot spread. (`inf`
        # states are already caught by the `>` gates; this specifically closes the NaN leak.) This runs
        # AFTER the termination toggle so a NaN env is reset even during no-termination rollouts.
        rd = self.robot.data
        nonfinite = (
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
        # TIME-OUT (bootstrapped, NOT a failure) — returned in the `truncated` slot, so
        #   (a) the adaptive-sampling failure EMA (keyed on reset_terminated) ignores it, and
        #   (b) skrl bootstraps the cut-off value  (agents/skrl_ppo_cfg.yaml: time_limit_bootstrap: True).
        # FIRST term = the real episode end: the reference sequence ran out. Replaces the old fixed-chunk
        # rule (`episode_length_buf >= max_episode_length - 1`), which only worked because the start was
        # clamped so every episode was exactly num_frame_chunk steps. Episodes are now variable length
        # (start frame → sequence end), so the end must be detected on the PER-ENV frame index.
        # SECOND term = safety cap only; with episode_length_s = ref_len/action_fps the two coincide
        # exactly for an env that started at frame 0, and the cap never fires before the frame check.
        # NOTE the bootstrap value skrl adds is V(next_observations), and IsaacLab auto-resets inside
        # step() → that is the value of the NEW episode's RSI start state, not of the true successor.
        # If end-of-sequence states ever look over-valued, suspect this first.
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
    def _build_ref_link_kpt_local(self) -> None:
        """(F,L,3) 각 wrap 링크의 SMPL-X 목표 키포인트를 물체 기준 좌표로 채웁니다.

        [ROLLBACK MARKER: link-kpt-smpl] 2026-09-01. 이전에는 _solve_ref_link_local() 이 로봇을
        프레임마다 레퍼런스 자세로 세워 FK 로 링크 위치를 읽었다 — 즉 목표가 "리타게팅된 로봇의
        자세"였다. 사용자 의도는 SMPL-X 사람 키포인트 추종이므로 _ref_kpts 의 손 블록(40개)에서
        링크별 대응점을 가져온다. 로봇 FK 가 필요 없어져 init 의 물리 스텝 루프도 사라진다.

        대응 (URDF 실측으로 검증):
          palm            → 손 블록 wrist (HAND_CHAIN 이 wrist→palm 으로 정의)
          {ff,mf,rf,lf}proximal → 같은 손가락 knuckle 의 MCP 점. URDF 상 knuckle→proximal 오프셋이
                            정확히 0.0 mm (2-DOF 너클이 두 링크로 쪼개져 원점이 일치) 이므로 동일 점.
          {ff,mf,rf,lf}middle   → PIP
          {ff,mf,rf,lf}distal   → DIP (Tip 이 아니라 관절 — 링크 원점끼리 비교하므로)
          thmiddle / thdistal   → 엄지 MCP / IP
          thproximal      → 대응 없음 (ParaHome 엄지는 MCP/IP/Tip 3점뿐, thproximal 은 palm 에서
                            45.6 mm 로 CMC 위치). 마스크에서 제외한다.
        물체 기준으로 저장하므로 실행 중에는 살아있는 물체 자세로 되돌려 월드 목표를 만든다 —
        물체가 굴러가면 손이 따라가야 할 자세도 함께 돈다 (이전 형태와 같은 성질).
        """
        F, L, dev = self._ref_len, N_LINK_CONTACT, self.device
        self._ref_link_kpt_local = torch.zeros(F, L, 3, device=dev)
        self._link_kpt_has_ref = torch.zeros(L, dtype=torch.bool, device=dev)
        if not self._has_object:
            return
        # 손 블록(한 손 20개)의 Shadow 바디 이름 = _ref_kpts 손 블록 순서
        _hb = [b for spec in HAND_CHAIN.values() for b in spec["shadow"]]          # 20
        _nb = len(BODY_KPTS)                                                      # 14
        idx: list[int] = []
        for i, full in enumerate(LINK_CONTACT_NAMES):
            side, body = full.split("_")[1], full.split("_", 2)[2]
            want = (f"{body[:2]}knuckle" if body.endswith("proximal") and not body.startswith("th")
                    else body)
            if want not in _hb:
                idx.append(0)                    # 자리만 채움 (마스크가 0 이라 안 읽힘)
                continue
            idx.append(_nb + (0 if side == "l" else len(_hb)) + _hb.index(want))
            self._link_kpt_has_ref[i] = True
        _ti = torch.tensor(idx, device=dev, dtype=torch.long)
        _qc = math_utils.quat_conjugate(self._ref_obj_quat).unsqueeze(1).expand(-1, L, -1)
        _rel = self._ref_kpts[:, _ti, :] - self._ref_obj_pos.unsqueeze(1)         # (F,L,3)
        self._ref_link_kpt_local = math_utils.quat_apply(_qc, _rel)
        _n = int(self._link_kpt_has_ref.sum())
        print(f"[link-kpt-smpl] SMPL 대응 {_n}/{L} 링크 "
              f"(제외: {[n.split('_',2)[2] for i, n in enumerate(LINK_CONTACT_NAMES) if not self._link_kpt_has_ref[i]]})")
        print(f"[link-kpt-objframe] 레퍼런스 링크 위치를 물체 기준으로 계산: {F} 프레임 x {L} 링크")

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids) -> None:
        # [ROLLBACK MARKER: deferred-cache] MUST run before super(), which zeroes episode_length_buf —
        # the whole point of the deferral is to filter on how long the episode actually lasted.
        # getattr, not a bare attribute: DirectRLEnv.__init__ resets every env, and that happens
        # BEFORE _post_init_buffers allocates the staging tensors.
        _ep_len = self.episode_length_buf[env_ids].clone()
        # [ROLLBACK MARKER: failure-dump] 같은 이유로 super() 이전 — episode_length_buf 가 살아 있어야
        # 초기화 여파(min_len)를 걸러낼 수 있고, reset_terminated 도 아직 이번 종료를 가리킵니다.
        if getattr(self, "_fd_on", False):
            self._fd_harvest(env_ids, _ep_len)
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

        # ---- failure-weighted sampling: EMA-update per-frame failure counts ----
        # (was DEAD CODE: _failure_count was init to zeros and never updated → the weighted
        # sampling branch below collapsed to uniform.) Mirrors grasp: up-weight the LAST-GOOD frame
        # (_enough_idx) of TERMINATED envs — i.e. where the policy started failing, not the death
        # frame itself. reset_terminated = deviation-death (not timeout) so successful time-outs
        # don't inflate. Only meaningful when failure_weighted_sampling is on (train), but the EMA
        # is cheap and harmless to keep updated in pretrain (where sampling ignores it).
        if c.adaptive_sampling and c.failure_weighted_sampling and hasattr(self, "_died"):
            term = (self.reset_terminated[env_ids] if hasattr(self, "reset_terminated")
                    else self._died[env_ids])
            # [backward-dir] 역방향 사망은 "거꾸로 하기에 실패"라 정방향 난이도와 무관합니다.
            # 섞으면 시작 프레임 분포가 오염되므로 정방향 환경만 셉니다.
            if self._any_backward:
                term = term & self._dir_fwd[env_ids]
            if bool(term.any()):
                fail_frames = self._enough_idx[env_ids][term].clamp(0, self._ref_len - 1)
                counts = torch.bincount(fail_frames, minlength=self._ref_len).float()
                self._failure_count = c.adaptive_alpha * counts + (1.0 - c.adaptive_alpha) * self._failure_count

        # ---- adaptive frame sampling ----
        self._sampling_step_count += 1
        self._friction_step_count += 1   # [ROLLBACK MARKER: friction-curriculum] 감쇠 진행
        have_train = ~self._init_flg
        have_pre = (~self._pretrain_init_flg) if self._pretrain_init_flg is not None else None
        # RSI candidate frames = every frame the reset can restore. With the PyRoki retarget reference
        # loaded (_ref_joints), the where_ref reset path seeds ANY frame from the retargeted pose, so every
        # frame is a valid start FROM STEP 0 (reference-based RSI, no pretrain cache needed); the train cache
        # still supplies better (physical) restore states for the frames it has since covered. Without a
        # retarget reference, fall back to cache-coverage candidates (train ∪ pretrain, frame 0 always on).
        if self._ref_joints is not None:
            candidates = torch.ones(self._ref_len, dtype=torch.bool, device=dev)
        else:
            candidates = have_train.clone()
            if have_pre is not None:
                candidates = candidates | have_pre
            if candidates.sum() == 0:
                candidates[0] = True                                 # frame 0 always available
        cand_idx = torch.nonzero(candidates, as_tuple=False).squeeze(-1)

        # ── RSI START-FRAME CURRICULUM (cold-start mitigation) ─────────────────────────────
        # Ramp the sampleable start-frame CEILING from ~frame 0 to the full trajectory over the
        # first rsi_curriculum_steps control steps, so a COLD value/policy/obs-scaler (no pretrain
        # warm-start) warms up on a narrow, well-conditioned near-frame-0 distribution before deep
        # frames appear. Diverse-from-step-0 RSI onto cold nets is what ignites the PPO ratio
        # explosion (confirmed via run diff: appeared with episode 5→3 [start range 1→101] +
        # warmstart T→F). Restricts the SHARED candidate pool so it applies to BOTH sampling
        # branches below. `pick` is later rewound by _adaptive_back_frames and clamped to
        # [0, upper], so an early small ceiling keeps start pinned near 0. Reference-seeded RSI
        # only (every frame restorable); with a cache-coverage pool the crawl already gates starts.
        # rsi_curriculum_steps<=0 → disabled (diverse-from-step-0, previous behaviour).
        if (self._ref_joints is not None and getattr(c, "rsi_curriculum_steps", 0)
                and c.rsi_curriculum_steps > 0):
            prog = min(1.0, self._sampling_step_count / float(c.rsi_curriculum_steps))
            ceil_f = max(int(prog * (self._ref_len - 1)), 1)          # target-frame ceiling: 0 → ref_len-1
            self._rsi_ceil_f = ceil_f                                 # for Curriculum/ logging
            cand_idx = cand_idx[cand_idx <= ceil_f]
            if cand_idx.numel() == 0:                                 # safety (frame 0 is always restorable)
                cand_idx = torch.zeros(1, dtype=torch.long, device=dev)

        # PRETRAIN (failure_weighted_sampling=False) → always UNIFORM over cached frames.
        # TRAIN → uniform for the first uniform_sampling_steps CONTROL STEPS, then failure-weighted.
        # (_sampling_step_count increments once per _reset_idx = once per control step, so it is a
        # timestep counter — compare it DIRECTLY, matching the grasp RSI env. The old `* self.num_envs`
        # was a porting bug: with num_envs≥uniform_sampling_steps it tripped on step 1 → no uniform warmup.)
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
            # [ROLLBACK MARKER: failure-sigma] 뽑힐 확률이 높은 프레임일수록 sigma 를 더 키웁니다.
            # 균등 샘플링 구간(위 분기)에서는 이 코드가 실행되지 않으므로 beta 는 1 로 남고,
            # 실패 가중이 켜진 뒤부터 자연스럽게 활성화됩니다.
            if self._fs_on:
                _g = (probs[_sel] / probs.max().clamp(min=1e-12)).clamp(0.0, 1.0) ** float(c.failure_sigma_gamma)
                self._episode_beta[env_ids] = 1.0 + (float(c.failure_sigma_beta_max) - 1.0) * _g
        # rewind for run-up, then clamp START to [0, ref_len - 1 - back]. The target `pick` ranges the
        # FULL trajectory (so failure-weighting can concentrate anywhere); the START only has to leave
        # the run-up itself playable, so every episode runs at least `_adaptive_back_frames` steps
        # (0.8 s @50 Hz) before the sequence-end time-out.
        #   Was `upper = ref_len - num_frame_chunk` (fixed 150-frame chunk): a 251-frame clip could only
        #   start in [0, 101] — frames 102..250 were never RSI starts — and a median 151-frame clip
        #   collapsed to [0, 1]. Now: [0, 210] and [0, 110] respectively.
        # adaptive_sampling=False (rollout/play) → upper=0 → every episode starts at frame 0 and plays
        # the whole clip. That used to fall out of `ref_len - num_frame_chunk` with chunk=ref_len; with
        # the chunk gone it has to be stated explicitly or evaluation would start at random frames.
        upper = max(0, self._ref_len - 1 - self._adaptive_back_frames) if c.adaptive_sampling else 0
        # [ROLLBACK MARKER: rand-runup] 되감기를 프레임별로 무작위로 뽑습니다.
        # 고정 되감기(TJ 1.2초, 우리 0.8초)는 시작 분포를 실패 분포의 평행이동 복사본으로 만듭니다.
        # 실패가 한 지점에 몰리면 시작도 한 곳에 몰려서, 그 사이 프레임들은 캐시가 쌓이기만 하고
        # 시작점으로는 한 번도 쓰이지 않습니다. 또 "실패가 f에 쌓임 -> 시작이 f-back -> 거기서
        # 출발한 에피소드가 f에서 죽음"이 한 간격에 고착됩니다.
        # 되감기를 흩뜨리면 같은 지점을 여러 거리에서 연습하게 되고(짧으면 그 지점의 기술, 길면
        # 도달하는 법), 시작과 실패 지점의 상관이 끊깁니다.
        # 하한을 두는 이유: 되감기가 너무 짧으면 에피소드가 몇 스텝 만에 죽어
        # cache_min_episode_length 필터에 걸려 캐시에 아무것도 기여하지 못합니다.
        if c.runup_rand_min_frames > 0 and c.adaptive_sampling:
            _lo = min(int(c.runup_rand_min_frames), self._adaptive_back_frames)
            _back = torch.randint(_lo, self._adaptive_back_frames + 1, (n,), device=dev)
        else:
            _back = torch.full((n,), self._adaptive_back_frames, device=dev, dtype=torch.long)
        # 상한도 되감기에 맞춰 프레임별로: 되감기가 짧으면 더 뒤에서 시작할 수 있습니다.
        _upper = (self._ref_len - 1 - _back).clamp(min=0) if c.adaptive_sampling else torch.zeros_like(_back)
        start = (pick - _back).clamp(min=0).clamp(max=_upper)
        # [ROLLBACK MARKER: backward-dir] 역방향은 같은 실패 분포를 쓰되 목표의 뒤쪽에서 출발해
        # 거꾸로 내려오며 목표를 지나갑니다. 원본 프레임 pick+run_up 에서 시작한다는 것을 에피소드
        # 프레임으로 환산하면 (ref_len-1) - (pick+run_up) 입니다. 이렇게 해야 역방향 25%가 흩어지지
        # 않고 병목 구간을 훑습니다.
        if self._any_backward and c.adaptive_sampling:
            _bwd_n = ~self._dir_fwd[env_ids]
            if bool(_bwd_n.any()):
                _rs = (pick + self._adaptive_back_frames).clamp(max=self._ref_len - 1)
                _es = ((self._ref_len - 1) - _rs).clamp(min=0)
                _es = torch.minimum(_es, _upper)          # [rand-runup] 프레임별 상한
                start = torch.where(_bwd_n, _es, start)
        # safeguard: start must be covered by a cache; snap uncovered → 0 (in [0,upper]; covered via
        # frame-0 init-save, else the restore falls back to reference+default pose).
        bad = ~candidates[start]
        start[bad] = 0
        # [ROLLBACK MARKER: backward-dir] 이 리셋 묶음의 방향과, 에피소드 프레임 -> 원본 프레임.
        fwd_n = self._dir_fwd[env_ids] if self._any_backward \
            else torch.ones(n, dtype=torch.bool, device=dev)
        dir_sign_n = torch.where(fwd_n, 1.0, -1.0).unsqueeze(-1)     # (n,1)
        rstart = start if not self._any_backward else torch.where(fwd_n, start, (self._ref_len - 1) - start)
        # ── MASTER RSI SWITCH (cfg.use_rsi=False) ──────────────────────────────────────────────
        # Force every episode to begin at frame 0. Placed AFTER the sampling block rather than
        # around it so the block stays byte-identical for the use_rsi=True path (the wasted
        # randint/multinomial is n samples per control step — negligible). The failure-count EMA
        # and `Curriculum / rsi_start_ceiling` keep updating; both are inert bookkeeping here.
        # The companion gate below forces the reference restore path so the reset STATE is
        # frame-0-deterministic too — a frame-0 train-cache hit would otherwise still restore a
        # cached sim state, which is RSI machinery.
        if not c.use_rsi:
            start = torch.zeros_like(start)
        # [ROLLBACK MARKER: friction-curriculum] 에피소드마다 물체 마찰 재추출
        self._apply_object_friction(env_ids)
        self._frame_idx[env_ids] = start
        self._episode_start_frame[env_ids] = start                 # diagnostics (Diag / rsi_start_mean)
        # [ROLLBACK MARKER: failure-sigma] beta 적용 구간의 끝. `pick` 은 되감기 이전의 목표 프레임이라
        # start <= pick 이 항상 성립합니다. 균등 샘플링 분기에서는 위에서 beta 를 안 건드리므로
        # 직전 에피소드 값이 남지 않도록 여기서 1 로 되돌립니다.
        if self._fs_on:
            self._episode_pick[env_ids] = pick
            if use_uniform:
                self._episode_beta[env_ids] = 1.0
        # reset the tracking-quality streak for the reset envs (grasp mechanism)
        self._enough_continued[env_ids] = True
        self._enough_idx[env_ids] = start
        # SONIC: re-seed the 10-frame proprio history from the (post-reset) pose on the next
        # _get_observations, and clear last_action / last z_res so the first post-reset SONIC step
        # is in-distribution (born-dead avoidance). Body/root/hands reset to the retarget reference
        # below (locomanip-identical); the SONIC control base's consistency with that reset pose is
        # the P1 measurement (start-of-episode body jump) — measured in the env smoke, not assumed.
        if getattr(self, "_sonic", None) is not None:
            self._sonic_hist_init[env_ids] = True
            self._last_a_sonic[env_ids] = 0.0
            self._last_z_res[env_ids] = 0.0
            self._cur_policy_action[env_ids] = 0.0        # prev-action obs AND action_rate = 0 at episode start
            self._prev_policy_action[env_ids] = 0.0
            self._cur_policy_action_bnd[env_ids] = 0.0    # (currently-unread A/B copy — kept in sync)
            self._prev_policy_action_bnd[env_ids] = 0.0

        # ---- restore state (train cache hit → pretrain hit → reference+default) ----
        root_pose = torch.zeros(n, 7, device=dev); root_pose[:, 3] = 1.0
        root_vel = torch.zeros(n, 6, device=dev)
        jpos = self.robot.data.default_joint_pos[env_ids].clone()
        jvel = torch.zeros_like(jpos)
        org = self.scene.env_origins[env_ids]

        # 3-way source selection per env (train cache > pretrain cache > reference+default),
        # vectorized via boolean masks + 2D advanced-index gathers (no per-env python loop).
        aid = self._action_joint_ids_t                               # (65,) action-joint columns
        train_hit = ~self._init_flg[rstart]                          # (n,) [backward-dir]
        if self._pretrain_cache is not None:
            pretrain_hit = ~self._pretrain_init_flg[rstart]          # [backward-dir]
        else:
            pretrain_hit = torch.zeros_like(train_hit)
        # [ROLLBACK MARKER: ref-start-prob] 낮은 확률로 캐시를 무시하고 레퍼런스에서 시작합니다.
        # 한 프레임에 캐시가 한 번 쓰이면 그 뒤 모든 리셋이 그 상태를 씁니다. 그 상태가 나쁘면
        # 거기서 시작한 에피소드가 또 실패하고, 더 나은 상태가 그 프레임을 지나갈 일이 없어서
        # 고착됩니다. 레퍼런스를 가끔 섞으면 대안이 생기고, 레퍼런스 쪽 성적이 나으면 그 상태가
        # 캐시를 교체합니다(커밋은 여전히 순간 보상 비교 + 최소 길이 필터를 거칩니다).
        # 레퍼런스는 물리로 정착된 상태가 아니라서 초기에는 더 빨리 죽을 수 있습니다. 그건
        # 정보이지 손해가 아닙니다 — 짧게 죽으면 캐시에 아무것도 안 씁니다.
        if c.ref_start_prob > 0.0:
            _use_ref = torch.rand(n, device=dev) < float(c.ref_start_prob)
            train_hit = train_hit & (~_use_ref)
            pretrain_hit = pretrain_hit & (~_use_ref)
            self._diag_ref_start = float(_use_ref.float().mean())
        # [MASTER RSI SWITCH] use_rsi=False → never READ a state cache; send every env down the
        # where_ref path so the reset state is the frame-0 retarget reference pose, identical every
        # episode. The caches are still WRITTEN (_save_state_cache is untouched), so flipping the
        # switch back to True resumes with whatever coverage training has accumulated.
        if not c.use_rsi:
            train_hit = torch.zeros_like(train_hit)
            pretrain_hit = torch.zeros_like(pretrain_hit)
        where_train = train_hit
        where_pre = (~train_hit) & pretrain_hit
        where_ref = (~train_hit) & (~pretrain_hit)

        if where_train.any():                                        # train cache: 222-D layout
            idx = where_train.nonzero(as_tuple=True)[0]
            # [backward-dir] 캐시는 원본 프레임으로 색인되고 정방향 규약(속도 부호)으로 저장돼
            # 있습니다. train_hit도 rstart로 판정하므로 여기서도 rstart를 써야 합니다 — start를
            # 쓰면 역방향 환경이 판정과 다른 행을 읽고, 그 행이 미기록이면 사원수가 (0,0,0,0)이라
            # 바디 자세가 통째로 NaN이 됩니다.
            s = self._state_cache[rstart[idx]]
            if self._any_backward:
                s = self._flip_cache_vel(s, dir_sign_n[idx])
            root_pose[idx, :3] = s[:, 1:4] + org[idx]; root_pose[idx, 3:7] = s[:, 4:8]
            root_vel[idx, :3] = s[:, 8:11]; root_vel[idx, 3:6] = s[:, 11:14]
            jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 27:92]
            jvel[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 92:157]
            self._smoothed_actions[env_ids[idx]] = s[:, 157:222]
        if where_pre.any():                                          # pretrain cache: 209-D layout
            idx = where_pre.nonzero(as_tuple=True)[0]
            s = self._pretrain_cache[rstart[idx]]                    # [backward-dir]
            root_pose[idx, :3] = s[:, 1:4] + org[idx]; root_pose[idx, 3:7] = s[:, 4:8]
            root_vel[idx, :3] = s[:, 8:11]; root_vel[idx, 3:6] = s[:, 11:14]
            jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 14:79]
            jvel[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 79:144]
            self._smoothed_actions[env_ids[idx]] = s[:, 144:209]
        if where_ref.any():                                          # reference root + default/retargeted joints
            idx = where_ref.nonzero(as_tuple=True)[0]
            # [backward-dir] start는 에피소드 프레임. 레퍼런스는 원본 프레임으로 읽고, 속도는
            # 시간 방향에 따라 부호가 뒤집힙니다(자세는 그대로).
            fr = rstart[idx]
            _sg = dir_sign_n[idx]                                    # (m,1) +1 / -1
            root_pose[idx, :3] = self._ref_root_pos[fr] + org[idx]
            root_pose[idx, 3:7] = self._ref_root_quat[fr]
            root_vel[idx, :3] = self._ref_root_linvel[fr] * _sg
            root_vel[idx, 3:6] = self._ref_root_angvel[fr] * _sg
            if self._ref_joints is not None:
                jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = self._ref_joints[fr]
                # [ROLLBACK MARKER: ref-reset-jvel] 관절 속도도 레퍼런스에서 채웁니다. 바로 위의
                # 루트 속도와 동일한 처리 — 같은 원본 프레임 fr, 같은 방향 부호 _sg.
                # 스위치는 분기가 아니라 배율에 접어 넣었습니다(_ref_jvel_scale). 꺼져 있으면 배율이
                # 0.0이라 대입값이 정확히 0이고, jvel의 초기값도 0이므로 기존 동작과 비트 단위로
                # 동일합니다 — 분기를 하나 더 만들지 않고 같은 보장을 얻습니다.
                jvel[idx.unsqueeze(1), aid.unsqueeze(0)] = torch.clamp(
                    self._ref_joint_vel[fr] * _sg * self._ref_jvel_scale,
                    -self._ref_jvel_clip, self._ref_jvel_clip)          # (1,65) 성분별 상한, 브로드캐스트
            self._smoothed_actions[env_ids[idx]] = self._unscale(jpos[idx][:, aid])

        self._last_pretrain_fallback_ratio = float(where_pre.sum().item()) / max(1, n)

        # [ROLLBACK MARKER: tendon-reset] 텐던 축 J0 를 제약에 맞춰 씁니다 (세 리셋 경로 공통).
        # J0 는 액션 관절 65개에 없어 위에서 항상 default(=0) 로 남는데, 텐던은 1.1418*J1 을
        # 요구하므로 그대로 두면 리셋마다 최대 1.14 rad 위반으로 시작해 말단이 1.7 m/s 로 튑니다.
        # 속도도 같은 비율로 맞춥니다 — 위치만 맞추고 속도를 0 으로 두면 텐던이 다시 속도를 만듭니다.
        # 캐시 경로도 포함해야 합니다: 캐시는 65열만 저장하므로 J0 는 어느 경로에서도 복원되지 않습니다.
        if c.tendon_reset_couple and self._tendon_j0_ids.numel() > 0:
            # [ROLLBACK MARKER: ref-j0] 리타게팅이 J0 를 풀었으면 그 값을 씁니다. 다만 캐시 경로의
            # J1 은 레퍼런스와 다르므로 (레퍼런스 J0 > 실제 J1) 이 되어 제약을 깰 수 있습니다 —
            # min() 으로 잘라 q_J0 <= q_J1 을 항상 만족시킵니다. 65열 npz 면 _ref_j0 가 None 이라
            # 기존 J1 x gear 경로가 그대로 돕니다.
            if getattr(self, "_ref_j0", None) is not None:
                # rstart = 전체 리셋 env 의 원본 프레임 (n,). fr 은 레퍼런스 경로 env 부분집합만
                # 담는 조건부 변수라 여기서 쓰면 모양이 안 맞습니다.
                jpos[:, self._ref_j0_ids] = torch.minimum(
                    self._ref_j0[rstart], jpos[:, self._tendon_j1_ids])
            else:
                jpos[:, self._tendon_j0_ids] = jpos[:, self._tendon_j1_ids] * self._tendon_gear
            if getattr(self, "_ref_j0_vel", None) is not None:
                # 손 클립을 그대로 씁니다 (J0 는 손가락 관절). 부호는 역방향 에피소드 규약(_sg).
                jvel[:, self._ref_j0_ids] = torch.clamp(
                    self._ref_j0_vel[rstart] * dir_sign_n * self._ref_jvel_scale,   # dir_sign_n 은 (n,1)
                    -float(c.ref_reset_joint_vel_clip_hands), float(c.ref_reset_joint_vel_clip_hands))
            else:
                jvel[:, self._tendon_j0_ids] = jvel[:, self._tendon_j1_ids] * self._tendon_gear

        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)

        # object restore if present: reference pose on pretrain/reference resets; on an RSI
        # TRAIN-cache hit restore the object FROM the cache too (blocks [14:27]) so the drifted
        # cached robot and the object stay a physically-consistent pair (a reference object paired
        # with a mid-manip cached robot could trip term_obj_pos_err/term_ft_err at start+1).
        if self._has_object:
            f0 = start
            rf0 = rstart                                            # [backward-dir] 원본 프레임
            ref_op = self._ref_obj_pos[rf0] + org                    # (n,3) world
            # [ROLLBACK MARKER: spawn-declear] lift the SPAWN out of the support. Reference-rest
            # frames only, zero elsewhere. _ref_obj_pos itself is untouched, so reward, observations
            # and the reference velocities stay pure GT. A cache HIT overrides this entirely, which is
            # correct: a cached pose was actually simulated, so it is already clear of the support.
            # getattr: DirectRLEnv.__init__ resets every env before the solve has run.
            _lift = getattr(self, "_obj_spawn_lift", None)
            if _lift is not None:
                ref_op = ref_op.clone()
                ref_op[:, 2] = ref_op[:, 2] + _lift[f0]
            # [/ROLLBACK MARKER: spawn-declear]
            ref_oq = self._ref_obj_quat[rf0]                         # (n,4)
            op = torch.zeros(n, 7, device=dev)
            op[:, :3] = ref_op; op[:, 3:7] = ref_oq
            # reference path: seed the object at its REFERENCE velocity for the sampled frame (mid-motion
            # starts place the object moving, not at rest). Cache-hit envs overwrite from the cache below.
            ovel = torch.zeros(n, 6, device=dev)
            ovel[:, :3] = self._ref_obj_linvel[rf0] * dir_sign_n      # [backward-dir]
            ovel[:, 3:6] = self._ref_obj_angvel[rf0] * dir_sign_n
            if where_train.any():
                # [backward-dir] 캐시는 원본 프레임 색인이고 정방향 규약으로 저장돼 있습니다.
                sc = self._state_cache[rstart]                       # (n,222)
                if self._any_backward:
                    sc = self._flip_cache_vel(sc, dir_sign_n)
                tw = where_train.unsqueeze(-1)
                op[:, :3] = torch.where(tw, sc[:, 14:17] + org, ref_op)
                op[:, 3:7] = torch.where(tw, sc[:, 17:21], ref_oq)
                ovel = torch.where(tw, sc[:, 21:27], ovel)
            self._object.write_root_pose_to_sim(op, env_ids=env_ids)
            self._object.write_root_velocity_to_sim(ovel, env_ids=env_ids)

        # seed delta integrators (all groups) to the actual reset pose; ema → 0
        rj = jpos[:, self._action_joint_ids_t]
        for gname, sl in self._group_slices.items():
            self._delta_target[gname][env_ids] = rj[:, sl]
            self._delta_ema[gname][env_ids] = 0.0
        # [DELTA-ACTION] reset the SONIC delta integrators: z_res → 0 (no accumulated latent residual),
        # hand target → reset hand pose (delta=0 holds the reset grasp). No-op unless the switch is on.
        if self._sonic is not None:
            self._z_res_int[env_ids] = 0.0
            self._z_delta_ema[env_ids] = 0.0
            self._hand_delta_target[env_ids] = rj[:, self._sonic_hand_slice]
            self._hand_delta_ema[env_ids] = 0.0
        self._prev_action[env_ids] = self._smoothed_actions[env_ids]

    # ── [ROLLBACK MARKER: failure-dump] ───────────────────────────────────────────────────────
    def _fd_record(self, state: torch.Tensor, reward: torch.Tensor) -> None:
        """제어 스텝마다 앞쪽 N개 환경의 한 행을 링 버퍼에 씁니다. `state`는 _save_state_cache 가
        이미 만들어 둔 (E,222) 복원 상태라 다시 계산하지 않습니다."""
        n, dev = self._fd_n, self.device
        row = torch.zeros(n, self._FD_DIM, device=dev)
        row[:, 0:222] = state[:n]
        _pa = getattr(self, "_cur_policy_action", None)
        if _pa is not None:
            row[:, 222:222 + min(100, _pa.shape[1])] = _pa[:n, :100]
        try:
            row[:, 322:322 + N_LINK_CONTACT] = self._link_contact_forces()[:n]
        except Exception:
            pass
        row[:, 354] = self._frame_idx[:n].float()                      # 레퍼런스 프레임
        row[:, 355] = float(self._sampling_step_count)                 # 학습 스텝
        row[:, 356] = self.episode_length_buf[:n].float()
        row[:, 357] = self._episode_start_frame[:n].float()
        row[:, 358] = 0.0                                              # 종료 원인 — 수확 때 채웁니다
        row[:, 359] = torch.arange(n, device=dev, dtype=torch.float32)
        # [ROLLBACK MARKER: cws-diag] 실패 창 안에서 접촉 렌치가 어떻게 변하는지 보려면 프레임별로
        # 남겨야 합니다 (텐서보드 평균만으로는 사망 직전 궤적을 못 봅니다).
        if getattr(self, "_cws_per_env", None) is not None:
            row[:, 360] = self._cws_per_env[:n]
            row[:, 361] = self._cws_nhit[:n]
            row[:, 362] = self._cws_cov[:n]
            row[:, 363] = self._cws_def[:n]
        self._fd_ring[:, self._fd_ptr % self._fd_w] = row
        self._fd_ptr += 1
        self._fd_filled = min(self._fd_filled + 1, self._fd_w)

    def _fd_cause_code(self) -> torch.Tensor:
        """종료 원인 비트마스크 (E,). _dones_deviation 이 쓰는 것과 같은 임계값."""
        e, cc = self._errs, self.cfg
        dev = self.device
        code = torch.zeros(self.num_envs, device=dev)
        bits = [(e["obj_pos"] > cc.term_obj_pos_err, 1), (e["obj_rot"] > cc.term_obj_rot_err, 2),
                (e["wrist_pos"] > cc.term_wrist_pos_err, 4), (e["ft"] > cc.term_ft_err, 8),
                (e["root_pos"] > cc.term_root_pos_err, 16), (e["root_rot"] > cc.term_root_rot_err, 32),
                (e["body"] > cc.term_body_kpt_err, 64)]
        for m, b in bits:
            code = code + m.float() * b
        return code

    def _fd_harvest(self, env_ids: torch.Tensor, ep_len: torch.Tensor) -> None:
        """종료(시간초과 아님)한 덤프 대상 환경의 창을 CPU 로 꺼냅니다. _reset_idx 앞부분에서,
        episode_length_buf 가 0 이 되기 전에 호출해야 합니다."""
        if self._fd_saved >= int(self.cfg.failure_dump_budget) or self._fd_filled < self._fd_w:
            return
        # 학습 전 구간에 고르게 담기 위한 버킷 예산
        bk = self._sampling_step_count // max(1, int(self.cfg.failure_dump_bucket))
        if bk != self._fd_bucket_id:
            self._fd_bucket_id, self._fd_bucket_taken = bk, 0
        if self._fd_bucket_taken >= self._fd_bucket_cap:
            return
        sel = env_ids < self._fd_n
        if not bool(sel.any()):
            return
        ids, lens = env_ids[sel], ep_len[sel]
        term = (self.reset_terminated[ids] if hasattr(self, "reset_terminated")
                else getattr(self, "_died", torch.ones_like(ids, dtype=torch.bool))[ids])
        if self._any_backward:                       # 역방향 사망은 정방향 난이도와 무관
            term = term & self._dir_fwd[ids]
        keep = term & (lens >= int(self.cfg.failure_dump_min_len))   # RSI 초기화 여파 차단
        if not bool(keep.any()):
            return
        ids = ids[keep]
        room = min(int(self.cfg.failure_dump_budget) - self._fd_saved,
                   self._fd_bucket_cap - self._fd_bucket_taken, int(ids.numel()))
        ids = ids[:room]
        # 링을 시간순으로 정렬해 꺼냅니다 (_fd_ptr 이 다음에 덮어쓸 자리 = 가장 오래된 칸)
        order = (torch.arange(self._fd_w, device=self.device) + self._fd_ptr) % self._fd_w
        win = self._fd_ring[ids][:, order].clone()                   # (m,W,360)
        win[:, :, 358] = self._fd_cause_code()[ids].unsqueeze(1)     # 종료 원인
        self._fd_buf.append(win.cpu().numpy())
        self._fd_saved += int(ids.numel())
        self._fd_bucket_taken += int(ids.numel())
        if sum(a.shape[0] for a in self._fd_buf) >= 500:
            self._fd_flush()

    def _fd_flush(self) -> None:
        """모아둔 창들을 npz 한 개로 씁니다. 프레임별 실패 히스토그램도 같이 넣습니다."""
        if not self._fd_buf:
            return
        import os

        import numpy as _np
        d = self._fd_dir
        os.makedirs(d, exist_ok=True)
        arr = _np.concatenate(self._fd_buf, axis=0).astype(_np.float32)
        p = os.path.join(d, f"fail_{self._sampling_step_count:07d}.npz")
        _np.savez_compressed(
            p, windows=arr,
            failure_count=self._failure_count.detach().cpu().numpy().astype(_np.float32),
            joint_names=_np.array(self._action_joint_names),      # 27:92 / 92:157 열의 이름 순서
            link_contact_names=_np.array(LINK_CONTACT_NAMES),     # 322:354 열의 이름 순서
            step=_np.int64(self._sampling_step_count), ref_len=_np.int64(self._ref_len),
            layout=_np.array([
                "행 = (에피소드, 창스텝, 360). 창은 시간순(마지막 행 = 종료 직전 스텝).",
                "0:222 cache_state ([0]=reward, [1:4]root_pos, [4:8]root_quat, [8:11]root_linvel,",
                "      [11:14]root_angvel, [14:17]obj_pos, [17:21]obj_quat, [21:24]obj_linvel,",
                "      [24:27]obj_angvel, [27:92]jpos(65), [92:157]jvel(65), [157:222]smoothed(65))",
                "222:322 raw_action(100) | 322:354 link_contact_force(32, LINK_CONTACT_NAMES 순서)",
                "354 ref_frame | 355 train_step | 356 ep_len | 357 start_frame | 358 term_bits | 359 env_id",
                "360 cws_score(원점수, v보정 민감) | 361 cws_nhit | 362 cws_coverage(요구방향 충족비율) | 363 cws_deficit",
                "term_bits: 1=obj_pos 2=obj_rot 4=wrist_pos 8=ft 16=root_pos 32=root_rot 64=body",
                "주의: 링 버퍼라 에피소드가 창보다 짧으면 앞쪽 행은 이전 에피소드입니다.",
                "      유효 행 = 마지막 min(ep_len, W) 개. ep_len = windows[i,-1,356].",
            ]))
        print(f"[failure-dump] {p}  {arr.shape[0]} 에피소드  "
              f"{os.path.getsize(p)/1024**2:.1f} MB  (누적 {self._fd_saved})")
        self._fd_buf.clear()

    def close(self):
        """마지막 500개 미만의 창이 버려지지 않게 종료 시 한 번 더 씁니다."""
        if getattr(self, "_fd_on", False):
            try:
                self._fd_flush()
            except Exception as ex:                       # 종료 경로를 막지 않습니다
                print(f"[failure-dump] 종료 플러시 실패: {ex}")
        return super().close()
    # ── [/ROLLBACK MARKER: failure-dump] ──────────────────────────────────────────────────────

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
        fr = self._rframe().clamp(max=self._ref_len - 1)                    # (E,) [backward-dir]
        # [ROLLBACK MARKER: failure-dump] 아래 게이트들과 무관하게 매 제어 스텝 기록해야 하므로
        # (실패 창은 캐시 자격과 상관없이 필요합니다) 어떤 조기 반환보다도 먼저 호출합니다.
        if getattr(self, "_fd_on", False):
            self._fd_record(self._build_cache_state(reward, org), reward)
        # write-gate while pretrain cache loaded (avoid poisoning with 1-2-step object interpen.)
        gate = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        if self._pretrain_cache is not None:
            gate = self.episode_length_buf >= 3

        # ---- tracking-quality gate (grasp mechanism) ----
        # A frame is cache-eligible only while tracking has been CONTINUOUSLY "good enough" since
        # reset. Matches grasp exactly: FINGERTIP bar + OBJECT 3-phase threshold (start-loose for
        # the first ~20 frames so the cache seeds early / early / tighter 'late' once the curriculum
        # reaches the end) — NO body/hand bars (grasp never gated on those). When has_object is False
        # the obj errs are 0 → the object phase is trivially satisfied → gate reduces to ft only.
        action_fps = round(1.0 / (c.sim.dt * c.decimation))
        start_cutoff = action_fps * 2 // 3                                  # first 2/3 s = 33 frames @50Hz
        reached_end = self.is_reached_end                                   # python bool
        op, orr = e["obj_pos"], e["obj_rot"]
        start_c = (op < 0.10) & (orr < 0.50) & (fr <= start_cutoff)
        early_c = (op < c.enough_obj_threshold) & (orr < c.enough_obj_rot_threshold) & (not reached_end)
        late_c = (op < c.enough_obj_threshold_late) & (orr < c.enough_obj_rot_threshold_late) & reached_end
        good = (e["ft"] < c.enough_ft_threshold) & (start_c | early_c | late_c)
        # floating-base whole-body/root quality bars (default inf = off in train; tightened in
        # pretrain, where the object phase-gate is inert and the gate would otherwise be ft-only).
        good = good & (e["body"] < c.cache_body_bar) & (e["root_pos"] < c.cache_root_pos_bar) \
            & (e["root_rot"] < c.cache_root_rot_bar)
        still_good = self._enough_continued & good
        self._enough_idx = torch.where(still_good, fr, self._enough_idx)    # last good frame
        self._enough_continued = still_good

        # cache ranking key = the ACTUAL step reward (grasp / TJ convention; see the docstring).
        r = reward                                                          # (E,)
        # [ROLLBACK MARKER: deferred-cache] -------------------------------------------------------
        # With the deferral on, `better` is NOT evaluated here — the comparison against the cache
        # happens at commit time in _flush_state_cache, because the cache moves while the episode
        # runs. Staging is gated on the per-frame quality streak only; the episode-length filter is
        # applied at termination, which is the whole point (it needs hindsight).
        if getattr(self, "_pend_state", None) is not None:
            stage_mask = gate & self._enough_continued                     # (E,)
            if stage_mask.any():
                slot = self.episode_length_buf.clamp(max=self._pend_cap - 1)   # (E,)
                rows = torch.nonzero(stage_mask, as_tuple=False).squeeze(-1)
                # [backward-dir] 캐시는 정방향 규약으로 통일해 저장합니다.
                _st = self._build_cache_state(r, org)
                if self._any_backward:
                    _st = self._flip_cache_vel(_st, self._dir_sign)
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

    # ── 역방향 롤아웃 [ROLLBACK MARKER: backward-dir] ──────────────────────────────────
    _CACHE_VEL_SLICES = ((8, 14), (21, 27), (92, 157))   # 루트 6, 물체 6, 관절 65

    def _flip_cache_vel(self, state: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        """캐시 행의 속도 채널에만 부호를 곱합니다. sign (n,1): 정방향 +1 / 역방향 -1.

        캐시는 항상 정방향 규약으로 저장합니다. 역방향 환경이 쓸 때는 정방향으로 바꿔서 넣고,
        읽을 때 다시 뒤집습니다. 자세는 시간 방향과 무관하므로 속도만 건드립니다.
        """
        out = state.clone()
        for a, b in self._CACHE_VEL_SLICES:
            out[:, a:b] = out[:, a:b] * sign
        return out

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
                # [backward-dir] 이 후보가 역방향 환경에서 왔는지 (행 = env_ids[rows] 순서)
                cand_bwd = (~self._dir_fwd[rows[sel[:, 0]]]) if self._any_backward \
                    else torch.zeros(len(sel), dtype=torch.bool, device=cand_r.device)
                for uf in torch.unique(cand_frame):
                    m = cand_frame == uf
                    j = cand_r[m].argmax()
                    best = cand_state[m][j]
                    # [backward-dir] 역방향 상태에는 "정방향 동역학으로 실제 도달했다"는 보증이
                    # 없습니다. 시간을 거꾸로 돌려 닿은 접촉 배치가 정방향으로는 미끄러질 수
                    # 있으므로, 명백히 나을 때만 기존 항목을 밀어내게 합니다. 점수 척도 자체는
                    # 같으므로(둘 다 그 프레임 레퍼런스와의 일치도) 깎지는 않습니다.
                    _need = self._state_cache[uf, 0]
                    if bool(cand_bwd[m][j]):
                        _need = _need * (1.0 + self.cfg.backward_replace_margin)
                    if best[0] > _need:
                        self._state_cache[uf] = best
                        self._init_flg[uf] = False
                        self._cache_from_bwd[uf] = bool(cand_bwd[m][j])
                        self._reached_frame = max(self._reached_frame, int(uf.item()))
        # clear staging for ALL terminating envs (kept or dropped) so the next episode starts clean
        self._pend_valid[env_ids] = False
    # [/ROLLBACK MARKER: deferred-cache] ----------------------------------------------------------

    # ---------------------------------------------------- pretrain-cache warm-start
    def set_pretrain_cache(self, npz_path: str) -> bool:
        """Load the pretrain phase's 209-D read-only state cache for RSI warm-start (layout
        [0]reward [1:14]root [14:79]jpos [79:144]jvel [144:209]smoothed; read by _reset_idx's
        pretrain branch). Gated by cfg.pretrain_cache_warmstart (False → vanilla RSI: empty train
        cache, _reached_frame gate, fixed-home + frame-0-IK fallback). [ROLLBACK MARKER: pretrain-cache-warmstart]"""
        if not self.cfg.pretrain_cache_warmstart:
            print("[pretrain-cache] warm-start DISABLED by cfg (pretrain_cache_warmstart=False); vanilla RSI start.")
            return False
        if not os.path.exists(npz_path):
            return False
        d = np.load(npz_path, allow_pickle=True)
        cache = torch.from_numpy(d["state_cache"].astype(np.float32)).to(self.device)
        flg = torch.from_numpy(d["init_flg"]).to(self.device).bool()
        if cache.shape[0] != self._ref_len:
            print(f"[pretrain-cache] length mismatch {cache.shape[0]} vs {self._ref_len}; ignored")
            return False
        if cache.shape[1] != 209:
            print(f"[pretrain-cache] width {cache.shape[1]} != 209 (expected pretrain layout); ignored")
            return False
        self._pretrain_cache = cache
        self._pretrain_init_flg = flg
        return True
