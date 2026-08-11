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

import json
import math
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from . import rsi_buffer as RB
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .g1_shadow_repho_env_cfg import (
    BODY_KPT_OFFSETS,
    BODY_KPTS,
    FINGERTIP_OFFSETS,
    FINGERTIP_PAD_NORMALS,
    HAND_CHAIN,
    JOINT_GROUPS,
    LINK_CONTACT_NAMES,
    LINK_PAD_NORMALS,
    N_LINK_CONTACT,
    G1ShadowRephoEnvCfg,
    _ROBOT_USD,
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


# [ROLLBACK MARKER: backward-dir] VELOCITY columns of the 222-D state-cache row. Converting a cached
# state between the two time directions negates exactly these and leaves everything else alone (see
# `_flip_cache_vel`). Layout: [0] reward | [1:4] root pos | [4:8] root quat | [8:14] root lin+ang vel
# | [14:17] obj pos | [17:21] obj quat | [21:27] obj lin+ang vel | [27:92] joint pos
# | [92:157] joint vel | [157:222] smoothed actions.
_CACHE_VEL_COLS = list(range(8, 14)) + list(range(21, 27)) + list(range(92, 157))   # 6+6+65 = 77


class G1ShadowRephoEnv(DirectRLEnv):
    cfg: G1ShadowRephoEnvCfg

    # ------------------------------------------------------------------ init
    def __init__(self, cfg: G1ShadowRephoEnvCfg, render_mode: str | None = None, **kwargs):
        self._load_reference_trajectories(cfg)          # numpy buffers (pre-super: no device yet) → sets _ref_len
        self._build_object_cfg(cfg)                     # guarded: only if converted USD exists
        # EPISODE = RSI start frame → END OF THE REFERENCE SEQUENCE (or a termination). The horizon is
        # the trajectory itself, so episodes are VARIABLE length (see _get_dones / _reset_idx).
        #   Previously (grasp/TJ lineage) every episode was a fixed `episode_length_s` chunk
        #   (3.0 s = 150 frames @50 fps) and the RSI start was clamped to [0, ref_len - chunk] so the
        #   chunk always fit. That wasted the tail of every clip: a 251-frame clip could only ever start
        #   in [0, 101], and a MEDIAN ParaHome clip (151 frames after the 30→50 fps resample) collapsed
        #   to [0, 1] — i.e. RSI was effectively dead for half the dataset. Now the start clamp only has
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
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # `size` scales the VISUAL mesh only — the collider under it is a `Plane`, i.e. an infinite
        # half-space, so physics is unaffected either way. The default 100x100 m is smaller than the
        # env grid: 2048 envs at 3 m spacing span +-67.5 m, so 926 of them (45%) sit outside the drawn
        # floor and render as robots hanging over white nothing. That is what the training videos were
        # showing — not a fall, not a depenetration failure (root z never went below 0, and the far
        # envs are physically supported by the infinite collider). 200 m covers the current grid with
        # room to grow; the mesh is 4 vertices, so this costs nothing.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(200.0, 200.0)))

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
                track_air_time=False, track_contact_points=False,
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
        perm = torch.tensor([src.index(n) for n in env_order], device=self.device, dtype=torch.long)
        n_moved = int((perm != torch.arange(len(perm), device=self.device)).sum())
        self._ref_joints = self._ref_joints[:, perm]
        origin = "npz joint_names" if self._ref_joint_names is not None else "g1_shadow_joint_order.json"
        print(f"[ref-joints] retarget columns remapped by name from {origin}: "
              f"{n_moved}/{len(perm)} slots moved")
    # [/ROLLBACK MARKER: retarget-joint-order] -------------------------------------------------

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
        self._n_act = off                                              # 65
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
        for side in ("l", "r"):
            for spec in HAND_CHAIN.values():
                for body in spec["shadow"]:
                    full = f"robot0_{side}_{body}"
                    kpt_names.append(full)
                    kpt_off.append(FINGERTIP_OFFSETS.get(full, [0.0, 0.0, 0.0]))
        self._kpt_body_ids = torch.tensor(
            [self.robot.find_bodies(n)[0][0] for n in kpt_names], device=dev, dtype=torch.long)
        self._kpt_offsets = torch.tensor(kpt_off, device=dev, dtype=torch.float32)      # (54,3)
        # SPLIT body-keypoint reward: the 14 body kpts are partitioned into END-EFFECTOR (both wrists +
        # both ankles = reach + foot placement) and CORE (everything else). Each gets its own reward weight
        # (rew_ee_kpts / rew_body_kpts) so the extremities can be emphasized independently. Matched by name
        # (robust to reordering). The termination gate still uses the UNIFORM mean over ALL 14 (e["body"]).
        _body_names = kpt_names[:len(BODY_KPTS)]
        self._ee_kpt_idx = torch.tensor(
            [i for i, n in enumerate(_body_names) if ("wrist" in n or "ankle" in n)],
            device=dev, dtype=torch.long)                                              # (4,) L/R wrist + L/R ankle
        self._body_core_idx = torch.tensor(
            [i for i, n in enumerate(_body_names) if not ("wrist" in n or "ankle" in n)],
            device=dev, dtype=torch.long)                                              # (10,) core body
        # indices of the two wrist keypoints within the 14-body block → dedicated wrist-POSITION
        # termination gate (MEAN over both wrists).
        self._wrist_kpt_idx = torch.tensor(
            [i for i, n in enumerate(_body_names) if "wrist" in n], device=dev, dtype=torch.long)  # (2,)
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
        self._ref_obj_dof = T(self._np_obj_dof)                        # (F,P)
        self._ref_joints = T(self._np_ref_joints) if self._np_ref_joints is not None else None
        self._remap_ref_joints(c)
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
        self._link_contact_body_ids = torch.tensor(
            [self.robot.find_bodies(n)[0][0] for n in LINK_CONTACT_NAMES], device=dev, dtype=torch.long)  # (L,)
        self._ft_distal_idx = torch.tensor(
            [LINK_CONTACT_NAMES.index(n) for n in self.cfg.fingertip_body_names], device=dev, dtype=torch.long)  # (10,)
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
        # [ROLLBACK MARKER: backward-dir] per-env time direction. Stage 2 leaves every env FORWARD,
        # so `_any_backward` is False and `_rframe` is the identity — the env must behave exactly as
        # before. Stage 3 samples these per reset.
        # [ROLLBACK MARKER: dir-partition] direction is a FIXED property of the env, not a per-reset
        # draw. A partition gives each direction a constant share of the batch, so its buffer fills at
        # a predictable rate and its statistics are not confounded by the share drifting with the
        # weight table; an env's SONIC history, delta integrators and cache reads also never change
        # convention mid-run. Backward takes the LAST envs, so env 0 — which the viewer and most
        # diagnostics follow — is always forward.
        # Same gate the per-reset draw used: rollout/play set adaptive_sampling=False and expect a
        # forward-only, frame-0 protocol. Partitioning regardless would silently make a quarter of
        # every evaluation run backward.
        _bwd_on = (float(getattr(c, "backward_ratio", 0.0)) > 0.0
                   and bool(getattr(c, "use_rsi", True)) and bool(getattr(c, "adaptive_sampling", True)))
        _n_bwd = int(round(self.num_envs * float(getattr(c, "backward_ratio", 0.0)))) if _bwd_on else 0
        self._dir_fwd = torch.arange(self.num_envs, device=dev) < (self.num_envs - _n_bwd)
        self._dir_sign = torch.where(self._dir_fwd, 1.0, -1.0).unsqueeze(-1)   # +1 fwd, -1 bwd (velocity flip)
        self._any_backward = False

        # ---- state cache + RSI (train 222) ----
        # layout: reward(1) + root[pos3+quat4+linvel3+angvel3=13] + obj[pos3+quat4+linvel3+angvel3=13]
        #         + jpos(65) + jvel(65) + smoothed(65) = 1+13+13+195 = 222.
        #   [0]=reward [1:4]rootpos [4:8]rootquat [8:11]rootlinvel [11:14]rootangvel
        #   [14:17]objpos [17:21]objquat [21:24]objlinvel [24:27]objangvel
        #   [27:92]jpos [92:157]jvel [157:222]smoothed
        self._STATE_DIM = 222
        # [ROLLBACK MARKER: slot-cache] -------------------------------------------------------------
        # (D, F, K+1, 222): slot 0 is the retarget reference (state fixed, score live), slots 1..K
        # hold states the policy reached. D=2 is the ROLLOUT DIRECTION — RePHO keeps a separate self
        # buffer per direction because the score means different things in each ("how much further
        # can I go THIS way"), and a state that is a good forward start need not be a good backward
        # one. The STATE is direction-neutral (frames are original-time, velocities forward-signed),
        # which is what lets the cross transfer copy between them.
        self._n_slots = 1 + max(0, int(getattr(c, "cache_num_slots", 0)))
        self._state_cache = torch.zeros(2, self._ref_len, self._n_slots, self._STATE_DIM, device=dev)
        self._state_cache[:, :, :, 0] = -float("inf")                  # score column
        self._slot_occ = torch.zeros(2, self._ref_len, self._n_slots, device=dev, dtype=torch.bool)
        # slot 0 always HOLDS a state — the reference exists at every frame, for both directions.
        self._slot_occ[:, :, 0] = True
        self._state_cache[:, :, 0, 0] = 0.0
        # [ROLLBACK MARKER: repho-cache] the discounted return, kept BESIDE the cache rather than in
        # them: column 0 becomes the survival length under repho_length_score, and the return is the
        # tiebreak, so two numbers per entry are needed. A parallel tensor avoids widening _STATE_DIM,
        # which is tied to the restore slicing.
        self._cache_return = torch.zeros(2, self._ref_len, self._n_slots, device=dev)
        # [ROLLBACK MARKER: cross-buffer] RePHO keeps a SECOND buffer per direction (SupMat Alg 2
        # lines 33-38): the same physical states, scored for the OPPOSITE direction. A rollout from
        # t_start to t_end proves two different things about a frame t it passed through —
        # "from here I can go |t_end - t| further FORWARD" and "from here I can go |t - t_start|
        # further BACKWARD" — and which one matters depends on the direction the next episode will
        # leave that state in. The cross buffer stores the second, carries no return (its update rule
        # is a bare L_new > L_min), and is never sampled from: it is a staging area that feeds the
        # OTHER direction's self buffer through _cross_transfer.
        # Index 0 = written by FORWARD rollouts (feeds backward), 1 = written by BACKWARD.
        self._cross_cache = torch.zeros(2, self._ref_len, self._n_slots, self._STATE_DIM, device=dev)
        self._cross_len = torch.full((2, self._ref_len, self._n_slots), -float("inf"), device=dev)
        self._cross_occ = torch.zeros(2, self._ref_len, self._n_slots, device=dev, dtype=torch.bool)
        self._diag_cross_import = 0.0
        self._n_completions = torch.zeros(2, dtype=torch.long, device=dev)
        # [ROLLBACK MARKER: contact-term] consecutive steps on which the reference asked for object
        # contact and none was delivered. RePHO keeps the same streak with `(acc + new) * new`
        # (intermimic.py:1938) and terminates on it (:1755).
        # 4 channels, RePHO's layout (compute_cg_reward, intermimic.py:2334):
        # [left fingers, right fingers, left palm, right palm]. Independent streaks, because a right
        # hand that has dropped the object must not be masked by a left hand that is still touching.
        self._contact_lost = torch.zeros(self.num_envs, 4, dtype=torch.long, device=dev)
        self._contact_ch_names = ("l_fing", "r_fing", "l_palm", "r_palm")
        _ch = tuple(self.cfg.contact_loss_channels or ())
        self._contact_active = torch.tensor(
            [i for i, n in enumerate(self._contact_ch_names) if n in _ch], device=dev, dtype=torch.long)
        print(f"[contact-term] active channels={list(_ch)}"
              f"{'  (contact termination OFF)' if not _ch else ''}")
        self._diag_clost = [0.0] * 4
        self._diag_contact_lost = 0.0
        # [ROLLBACK MARKER: curriculum-window] RePHO does not let RSI start anywhere from the outset:
        # _init_range_left is the earliest sampleable start, and it only opens to 0 once the policy
        # has repeatedly run the whole clip FROM that boundary (intermimic.py:1787). Frames to its
        # left are not down-weighted, they are absent. Applied in EPISODE time, so it is symmetric:
        # for a backward rollout the same bound excludes the LATE original frames.
        self._init_range_left = int(getattr(c, "init_range_left", 0))
        self._left_completions = torch.zeros(2, dtype=torch.long, device=dev)
        # [ROLLBACK MARKER: curriculum-window] just_update_tar: after the reference is swapped over a
        # segment, RePHO forces sampling onto the SEAM — the 10 frames just before the segment's right
        # edge (intermimic.py:1165-1166), where the new kinematics have to rejoin the original — and
        # nearly freezes the buffer decay (5e-8 vs 5e-4, :1893-1899) so the buffer does not age out
        # while the policy relearns it. Swap and re-drill are one mechanism, not two.
        self._tar_window = None
        self._diag_tar_active = 0.0
        # [ROLLBACK MARKER: track-harvest] RePHO harvests the tracking target from ONE rollout launched
        # from argmax_{t,s} B_self[t][s].L in inference mode (SupMat Alg 3, 57-59), not from whatever
        # the batch happened to do. The inference half is out of reach from inside the env, but the
        # "from the best buffered state" half is not: a dedicated slice of envs is restarted there each
        # interval and only those envs may promote. That is what keeps a mediocre mid-batch rollout out
        # of the tracking target.
        self._harvest = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)
        _nh = int(getattr(c, "track_harvest_envs", 0))
        if _nh > 0 and getattr(c, "track_buffer", False):
            # BOTH directions: the cross path's candidate is the OPPOSITE direction's rollout
            # (intermimic.py:1145-1152), so a forward-only harvest leaves it permanently empty.
            for _m in (self._dir_fwd, ~self._dir_fwd):
                _idx = _m.nonzero(as_tuple=True)[0]
                if _idx.numel():
                    self._harvest[_idx[-min(_nh, _idx.numel()):]] = True
        self._harvest_frame = [None, None]        # per direction, RePHO runs one val per process
        self._pend_kin_valid = None
        self._track_fwd = [True, True]      # travel direction of each held candidate
        self._track_span = [0, 0]
        # [ROLLBACK MARKER: track-buffer] SupMat Alg 3, lines 54-73. RePHO does not only pick WHERE to
        # start from the buffer — it also rewrites WHAT it is tracking, replacing the noisy kinematic
        # reference at frames where a physically successful rollout did better. That is the paper's
        # headline ("Recovering Physically Plausible..."), and our reference has the same problem: it
        # comes from a PyRoki retarget of ParaHome SMPL-X, which is why the object needs a spawn lift
        # and the contact normals sit 40-54 deg off the true surface.
        # OFF by default: this changes what the REWARD compares against, so it must be an explicit
        # choice, and its staging buffer is as large as the state staging.
        self._last_kin = None
        self._track_kin = None
        self._track_occ = None
        self._pend_kin = None
        self._diag_track_promote = 0.0
        self._diag_track_applied = 0.0
        self._ref_orig = None
        if getattr(c, "track_buffer", False):
            kd = 3 * (self._ref_kpts.shape[1] + 10) + 14               # kpts + pads + root + object
            self._track_dim = kd
            # RePHO's candidate is ONE rollout with a contiguous valid span, not a per-frame pick
            # from many episodes (load_ref_traj, intermimic.py:1105-1112). It matters: the swap writes
            # a stretch of the reference, and stitching it from different episodes produces a
            # trajectory no single rollout ever executed — discontinuous exactly where the reward
            # then demands continuity. `_track_occ[d]` marks the candidate's valid span.
            self._track_kin = torch.zeros(2, self._ref_len, kd, device=dev)
            self._track_occ = torch.zeros(2, self._ref_len, device=dev, dtype=torch.bool)
            self._track_span = [0, 0]                                  # candidate length per direction
            print(f"[track] kinematics update ON: buffer 2x{self._ref_len}x{kd}, "
                  f"staging {self.num_envs}x{int(self.max_episode_length)}x{kd} fp32 = "
                  f"{self.num_envs * int(self.max_episode_length) * kd * 4 / 1024**2:.0f} MB")
        self._diag_switch_cov = 0.0
        _protect_slot0 = bool(getattr(c, "repho_protect_slot0", False))
        if False:
            # RePHO seeds slot 0 at 1.0 (intermimic.py:412) and never decays or replaces it. Ours is
            # seeded at 0, which reads as "worst possible" once column 0 means survival length — the
            # reference row would be evicted by the first rollout that survived a single step.
            self._state_cache[:, :, 0, 0] = 1.0
        self._episode_start_slot = torch.zeros(self.num_envs, device=dev, dtype=torch.long)
        self._diag_slot_ref_frac = 0.0
        self._diag_slot_entropy = 0.0
        # [ROLLBACK MARKER: spawn-declear] must precede the slot-0 seed below, which bakes the spawn
        # lift into the reference row. Steps physics; everything it touches (reference arrays, object,
        # robot, scene) already exists by here.
        self._solve_spawn_declear()
        # [/ROLLBACK MARKER: spawn-declear]
        if self._ref_joints is not None:                               # seed slot 0 with the reference
            _af = torch.arange(self._ref_len, device=dev)
            _row = self._ref_cache_row(_af, torch.zeros(self._ref_len, device=dev))
            self._state_cache[:, :, 0] = _row.unsqueeze(0)             # same reference for both directions
        if _protect_slot0:
            # AFTER the reference row: _ref_cache_row writes row[:, 0] = score = 0, so seeding before
            # it silently threw this away. RePHO seeds 1.0 (intermimic.py:412) so the reference is not
            # read as "worst possible" once column 0 means survival length.
            self._state_cache[:, :, 0, 0] = 1.0
        # [/ROLLBACK MARKER: slot-cache] ------------------------------------------------------------
        self._reached_frame = 0
        # [ROLLBACK MARKER: deferred-cache] staging buffer for the at-termination bulk commit.
        # Rows are written per step by _save_state_cache and merged into _state_cache by
        # _flush_state_cache when the episode ENDS, so the score can be hindsight (survival length).
        # Sized for a full episode: one that runs to the end of the clip must still be committable.
        self._pend_n = 0
        if True:
            self._pend_cap = int(self.max_episode_length)
            self._pend_state = torch.zeros(self.num_envs, self._pend_cap, self._STATE_DIM, device=dev)
            self._pend_frame = torch.zeros(self.num_envs, self._pend_cap, device=dev, dtype=torch.long)
            self._pend_valid = torch.zeros(self.num_envs, self._pend_cap, device=dev, dtype=torch.bool)
            if self._track_kin is not None:
                self._pend_kin = torch.zeros(self.num_envs, self._pend_cap, self._track_dim, device=dev)
                # its own validity/frame map: the kinematics are staged unfiltered, the state cache is not
                self._pend_kin_valid = torch.zeros(self.num_envs, self._pend_cap, device=dev, dtype=torch.bool)
                self._pend_kin_frame = torch.zeros(self.num_envs, self._pend_cap, device=dev, dtype=torch.long)
            print(f"[cache] deferred (at-termination) commit: staging buffer "
                  f"{self.num_envs}x{self._pend_cap}x{self._STATE_DIM} fp32 = "
                  f"{self.num_envs * self._pend_cap * self._STATE_DIM * 4 / 1024**2:.0f} MB")
        else:
            self._pend_state = self._pend_frame = self._pend_valid = None
        # [ROLLBACK MARKER: cache-score-rework] ---------------------------------------------------
        # Upper-triangular discount matrix M[j,k] = gamma^(j-k) for j >= k, so the whole
        # return-to-go vector is one matmul: G = rewards @ M, i.e. G[k] = sum_{j>=k} gamma^(j-k) r[j].
        # Built once (cap x cap floats) to avoid a per-flush Python scan over the episode length.
        self._score_gamma_mat = None
        _g = float(getattr(c, "cache_score_gamma", 0.0))
        if _g > 0.0 and self._pend_state is not None:
            _n = self._pend_cap
            _d = torch.arange(_n, device=dev).unsqueeze(1) - torch.arange(_n, device=dev).unsqueeze(0)
            self._score_gamma_mat = torch.where(_d >= 0, _g ** _d.clamp(min=0).float(),
                                                torch.zeros((), device=dev))
        self._diag_cache_overwrite = 0.0
        # [/ROLLBACK MARKER: cache-score-rework] --------------------------------------------------
        # [ROLLBACK MARKER: backward-dir] (2,F): row 0 = forward, row 1 = backward. Episode-frame f
        # is a DIFFERENT place in the clip for the two directions (forward f == original f, backward
        # row 0 is ever touched and the behaviour is the same as the old (F,) table.
        # RePHO bounds the sampleable START range and samples the start DIRECTLY — there is no
        # searchsorted returns is assigned straight to progress_buf and start_times). Its only
        # margin is on the right bound, `_init_range_right = max_episode_length - 24` (:111), so a
        # start cannot land where there is nothing left to roll out. We used to sample a TARGET and
        # rewind to reach the start, which measured the length weight and the finish penalty at one
        # frame while the episode actually began at another.
        self._init_range_right = max(1, self._ref_len - int(getattr(c, "init_range_right_margin", 24)))
        self._sampling_step_count = 0
        # gate + failure-weighted sampling). Reset per env in _reset_idx.
        # good frame, so it cannot stand in for the start once the episode is running).
        self._episode_start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        # pretrain-cache warm-start (209-D pretrain cache)
        self._pretrain_cache = None
        self._pretrain_init_flg = None

        # ---- FROZEN SONIC body prior (built on device; env_isaaclab + gear_sonic + vector_quantize) ----
        self._sonic = None
        if getattr(c, "use_sonic", True):
            import sys as _sys
            # scripts/process_dataset/sonic/sonic_prior.py. This file lives at
            # <repo>/source/robotis_sh5/robotis_sh5/tasks/direct/g1_shadow_repho/<this>.py
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
            # [ROLLBACK MARKER: sonic-encoder-g1] one field drives both encoder_index and the name
            # passed to encode_latent. Falls back to smpl (loudly) when the pyroki retarget that the
            # g1 term needs was never loaded, rather than feeding the encoder a block of zeros.
            self._sonic_enc = str(getattr(c, "sonic_encoder", "smpl"))
            if self._sonic_enc == "g1" and self._ref_joints is None:
                print("[sonic-encoder] g1 requested but no pyroki retarget (_ref_joints) for this "
                      "clip — falling back to smpl.")
                self._sonic_enc = "smpl"
            print(f"[sonic-encoder] tokenizer encoder = {self._sonic_enc}")
            self._sonic_perm = _SP.build_body_perm(list(self.robot.joint_names), device=dev)  # robot->SONIC (29)
            self._sonic_default = _SP.sonic_default_vector(dev).view(1, -1)                    # (1,29) SONIC order
            self._sonic_scale = _SP.sonic_scale_vector(dev).view(1, -1)                        # (1,29) SONIC order
            # SONIC order -> action-body order (first 29 action joints = legs+waist+arms), by NAME
            _ab_names = [self.robot.joint_names[i] for i in self._action_joint_ids[:29]]
            self._sonic_gather = torch.tensor([list(_GIO).index(n) for n in _ab_names],
                                              device=dev, dtype=torch.long)                    # (29,)
            self._sonic_hand_slice = self._group_slices["hands"]
            # 10-frame proprio history (term-major, oldest-first) = playback flat_proprio layout
            self._sonic_hist = {k: torch.zeros(self.num_envs, _SP.PROPRIO_HIST, d0, device=dev)
                                for k, d0 in [("ang", 3), ("jpr", 29), ("jvr", 29), ("act", 29), ("grav", 3)]}
            self._sonic_hist_init = torch.ones(self.num_envs, dtype=torch.bool, device=dev)
            # [ROLLBACK MARKER: hist-from-reference] ---------------------------------------------
            # Precomputed REFERENCE proprio, one row per clip frame, in exactly the layout the 10-frame
            # window wants. Seeding the window from these makes the fabricated past a real trajectory
            # (positions, velocities and orientation all agree), and makes it the SAME trajectory the
            # tokenizer already feeds SONIC as the FUTURE — previously the past was the robot's own
            # (frozen) state while the future was the reference, i.e. two different stories.
            # jvr is a finite difference of the reference joints, so index 0 is 0 — which is correct:
            # at the clip start the reference really is at rest, so a window clamped to frame 0 is
            # self-consistent rather than the usual frozen-position contradiction.
            self._ref_hist = None
            self._ref_g1_q = self._ref_g1_v = None      # [ROLLBACK MARKER: sonic-encoder-g1]
            if self._ref_joints is not None:
                # 65 action joints -> the 29 SONIC body joints, by robot-joint identity
                _act_of_robot = {int(r): k for k, r in enumerate(self._action_joint_ids)}
                _idx = torch.tensor([_act_of_robot[int(r)] for r in self._sonic_perm.tolist()],
                                    device=dev, dtype=torch.long)                                # (29,)
                _jpr = self._ref_joints[:, _idx] - self._sonic_default                           # (F,29)
                _jvr = torch.zeros_like(_jpr)
                _jvr[1:] = (_jpr[1:] - _jpr[:-1]) * float(c.control_fps)
                _q = self._ref_root_quat                                                          # (F,4)
                _g = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self._ref_len, 3)
                # [ROLLBACK MARKER: sonic-encoder-g1] the g1 tokenizer term wants ABSOLUTE joint
                # angles (motion_lib.get_dof_pos), not the default-pose-relative _jpr the proprio
                # history uses. Velocity is a finite difference of these same resampled positions so
                # the two channels cannot disagree.
                self._ref_g1_q = self._ref_joints[:, _idx]                                      # (F,29)
                self._ref_g1_v = torch.zeros_like(self._ref_g1_q)
                self._ref_g1_v[1:] = (self._ref_g1_q[1:] - self._ref_g1_q[:-1]) * float(c.control_fps)
                # [/ROLLBACK MARKER: sonic-encoder-g1]
                self._ref_hist = {
                    "jpr": _jpr,
                    "jvr": _jvr,
                    "act": _jpr / self._sonic_scale,        # action that commands this pose (decode inverse)
                    "grav": math_utils.quat_apply(math_utils.quat_conjugate(_q), _g),
                    "ang": math_utils.quat_apply(math_utils.quat_conjugate(_q), self._ref_root_angvel),
                }
            # [/ROLLBACK MARKER: hist-from-reference] --------------------------------------------
            self._last_a_sonic = torch.zeros(self.num_envs, 29, device=dev)
            self._last_z_res = torch.zeros(self.num_envs, int(c.sonic_action_dim), device=dev)
            # [ROLLBACK MARKER: jerk-diag] buffers for the reset-jerk / FSQ-flip diagnostics. Both are
            # per-step and O(E*64); see _sonic_pre_physics_step for what they measure.
            self._prev_fsq_lvl = torch.zeros(self.num_envs, int(c.sonic_action_dim), device=dev)
            self._prev_body_tgt = torch.zeros(self.num_envs, 29, device=dev)
            self._diag_jerk = {}
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
        # [backward-dir] episode-time future window, then mapped to the reference index. For a
        # backward env this walks BACKWARD through the original clip, which is what the frozen SONIC
        # prior is being asked to drive. Verified viable in closed loop (sonic_playback --reverse:
        # body_err 8.9 cm both directions, robot stays upright).
        idx = (self._frame_idx.unsqueeze(1) + torch.arange(K, device=dev).unsqueeze(0)).clamp(
            max=self._ref_len - 1)                                      # (E,10) episode-time window
        if getattr(self, "_any_backward", False):
            idx = torch.where(self._dir_fwd.unsqueeze(1), idx, (self._ref_len - 1) - idx)
        tok = torch.zeros(E, self._sonic_tok_dim, device=dev)
        pelvis_q = _canon(self.robot.data.root_quat_w)                  # (E,4) live pelvis
        rq = self._sonic_root_q[idx]                                    # (E,10,4) reference root (Z-up)
        dif = self._sonic_qmul(self._sonic_qinv(pelvis_q).unsqueeze(1).expand(E, K, 4), rq)  # (E,10,4) robot-relative
        ori6 = math_utils.matrix_from_quat(dif.reshape(-1, 4))[..., :2].reshape(E, K, 6)
        # [ROLLBACK MARKER: sonic-encoder-g1] ------------------------------------------------------
        if self._sonic_enc == "g1":
            # encoder_index column order is m.encoders = ['g1','teleop','smpl'].
            s, e, _ = lay["encoder_index"]; tok[:, s:e] = torch.tensor([1.0, 0.0, 0.0], device=dev)
            # flat 580 = [pos(f..f+9)] ++ [vel(f..f+9)], each 29, reshaped by SONIC to (10,58).
            # NOT per-frame [pos29|vel29] — recovered from how the decoder unpacks it
            # (gear_sonic/trl/losses/token_losses.py:79-85) and from command_multi_future =
            # cat([joint_pos_multi_future, joint_vel_multi_future]) in commands.py:897.
            # _dir_sign is (E,1): a backward episode walks the clip in reverse, which negates the
            # velocity channel but not the positions. Apply it on the (E,10,29) tensor, before the
            # flatten, so it broadcasts over the frame and joint axes.
            _v = self._ref_g1_v[idx] * self._dir_sign.unsqueeze(-1)      # (E,10,29)
            s, e, _ = lay["command_multi_future_nonflat"]
            tok[:, s:e] = torch.cat([self._ref_g1_q[idx].reshape(E, -1), _v.reshape(E, -1)], dim=-1)
            s, e, _ = lay["motion_anchor_ori_b_mf_nonflat"]; tok[:, s:e] = ori6.reshape(E, -1)
            return tok
        # [/ROLLBACK MARKER: sonic-encoder-g1] -----------------------------------------------------
        s, e, _ = lay["encoder_index"]; tok[:, s:e] = torch.tensor([0.0, 0.0, 1.0], device=dev)
        s, e, _ = lay["smpl_joints_multi_future_local_nonflat"]; tok[:, s:e] = self._sonic_smpl_j[idx].reshape(E, -1)
        s, e, _ = lay["joint_pos_multi_future_wrist_for_smpl"]; tok[:, s:e] = self._sonic_wrist_ref[idx].reshape(E, -1)
        s, e, _ = lay["smpl_root_ori_b_multi_future"]; tok[:, s:e] = ori6.reshape(E, -1)
        return tok

    # [ROLLBACK MARKER: jerk-diag] -----------------------------------------------------------------
    _FSQ_LEVELS = 32          # SONIC config.yaml: num_fsq_levels / fsq_level_list = 32 (verified)

    def _fsq_level(self, z: torch.Tensor) -> torch.Tensor:
        """Integer FSQ level of `z`, mirroring vector_quantize_pytorch.FSQ.bound exactly:
        half_l = (L-1)(1+eps)/2 ; offset = 0.5 (L even) ; round(tanh(z + atanh(offset/half_l))*half_l - offset).
        Bin width near z=0 is 1/((1-tanh(shift)^2)*half_l) = 0.0645 for L=32 — the scale that decides
        whether the policy's per-step noise re-draws the decoder's input."""
        half_l = (self._FSQ_LEVELS - 1) * (1.0 + 1e-3) / 2.0
        shift = math.atanh(0.5 / half_l)
        return torch.round(torch.tanh(z + shift) * half_l - 0.5)

    @torch.no_grad()
    def _diag_jerk_update(self, latent: torch.Tensor, z_res: torch.Tensor, body_target: torch.Tensor) -> None:
        E = self.num_envs
        lp = latent + float(self.cfg.residual_scale_latent) * z_res.view(latent.shape)
        lvl = self._fsq_level(lp).reshape(E, -1)                        # (E,64) quantized levels
        flip = (lvl != self._prev_fsq_lvl).float().mean(dim=1)          # (E,) fraction of dims re-drawn
        jump = (body_target - self._prev_body_tgt).abs().amax(dim=1)    # (E,) rad, largest joint step
        el = self.episode_length_buf
        # `fresh` deliberately EXCLUDES el == 0. `_prev_body_tgt` is not cleared on reset, so at
        # el == 0 the diff compares the LAST target of the previous episode against the FIRST target
        # of the new one — two unrelated points in the clip. That is not a jerk, and because the
        # value is large it dominated the mean (it inflated the earlier reported 4.15x reset/run
        # ratio). el 1..3 are all within one episode and measure the real post-reset transient.
        fresh, run = (el >= 1) & (el <= 3), el >= 10

        def m(t: torch.Tensor, k: torch.Tensor) -> float:
            return float(t[k].mean()) if bool(k.any()) else float("nan")

        self._diag_jerk = {
            "Jerk / fsq_flip_frac": float(flip.mean()),          # 0 = decoder input stable, 1 = fully re-drawn
            "Jerk / fsq_flip_run": m(flip, run),
            "Jerk / tgt_jump_reset": m(jump, fresh),             # rad, first 2 steps after an RSI reset
            "Jerk / tgt_jump_run": m(jump, run),                 # rad, settled steps
            "Jerk / tgt_jump_max": float(jump.max()),
            "Jerk / root_vel_max": float(self.robot.data.root_lin_vel_w.norm(dim=-1).max()),
            "Jerk / root_vel_p99": float(self.robot.data.root_lin_vel_w.norm(dim=-1).quantile(0.99)),
        }
        self._prev_fsq_lvl, self._prev_body_tgt = lvl, body_target.clone()
    # [/ROLLBACK MARKER: jerk-diag] ----------------------------------------------------------------

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
        # rew_latent_reg operand: the RAW (UNCLIPPED) latent residual — restores the pre-2026-07-28
        # form. Penalizing the clipped value leaves the exterior of the clip perfectly flat, so nothing
        # pulls mu back inside once a dim saturates; the raw form keeps that restoring gradient (and
        # makes the term unbounded, which is the accepted cost of the rollback).
        #   DELTA mode exception: there z_raw is a per-step INCREMENT, not a residual magnitude, so
        #   squaring it would silently turn latent_reg into a velocity penalty. Keep the integrated
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
        # [ROLLBACK MARKER: sonic-encoder-g1] encode_latent takes the encoder NAME; the tokenizer's
        # encoder_index alone would not switch it (see the cfg note), so both are set from one field.
        latent = self._SP.encode_latent(self._sonic, tok, encoder=self._sonic_enc)
        a_sonic = self._SP.residual_decode(self._sonic, latent, z_res, proprio,
                                           float(c.residual_scale_latent))   # (E,29) SONIC order, raw
        self._last_a_sonic = a_sonic
        body_sonic = self._sonic_default + self._sonic_scale * a_sonic  # (E,29) SONIC order absolute target
        body_target = body_sonic[:, self._sonic_gather]                # (E,29) action-body order
        # [ROLLBACK MARKER: jerk-diag] -------------------------------------------------------------
        # Separate the two candidate causes of the observed twitching / lurching, on the LIVE sim:
        #   H1 FSQ bin flips     -> the QUANTIZED latent changes every step even in steady state, so
        #                           the decoded body pose is re-drawn at 50 Hz. Measured by fsq_flip.
        #   H2 reset history     -> the fabricated 10-frame proprio history (frozen jpr with non-zero
        #                           jvr, act=0) is out of distribution, so the FIRST steps after an RSI
        #                           reset lurch. Measured by comparing tgt_jump_reset vs tgt_jump_run.
        # If H1 dominates: tgt_jump_run is already large and fsq_flip is high.
        # If H2 dominates: tgt_jump_reset >> tgt_jump_run.
        # They are not exclusive — both can be live at once.
        self._diag_jerk_update(latent, z_res, body_target)
        # [/ROLLBACK MARKER: jerk-diag] ------------------------------------------------------------
        hsl = self._sonic_hand_slice
        self._last_a_hand = a_hand                                      # effective (clamped) residual, for rew
        # [ROLLBACK MARKER: hand-residual] ---------------------------------------------------------
        # RESIDUAL ON THE RETARGET: a_hand=0 is exactly the retargeted hand pose, and the policy may
        # bend it by +-residual_scale_hands rad.
        #
        # The alternative below (ABSOLUTE) maps a_hand straight onto the full joint range with no
        # reference at all, which left the hand as the only part of the robot not anchored to the
        # retarget — the body IS anchored, through SONIC's tokenizer conditioning. The policy then
        # saturated it: Diag/hand_clamp_frac sat at 0.446, i.e. 45% of the 36 finger dims were pinned
        # to a joint limit at any moment, so the hand was effectively bang-bang open/closed and
        # approached the knife already fisted. Anchoring costs nothing in expressiveness that matters:
        # +-0.5 rad is ~32% of a Shadow J1/J2 range, and the retarget's own fingertip residual is
        # 1.4-5.7 cm, so the reference grasp shape is a starting point rather than a cage.
        #
        # STATELESS on purpose (no EMA, unlike the absolute path). The target is rebuilt from
        # (frame, a_hand) every control step, so nothing carries across a reset — which also removes
        # the absolute path's known defect where the cached _smoothed_actions[hands] re-enters CONTROL
        # at 50% weight on the first step after an RSI restore, leading the wrong way for a backward
        # episode. Smoothing is left to the fingers' own soft PD (stiffness 1.0, damping 0.2).
        #
        # This only became usable once the retarget joint order was fixed: before that the reference
        # hand had MFJ1 driven by the thumb's value, so anchoring to it would have anchored to garbage.
        if getattr(c, "sonic_hand_residual", False) and self._ref_joints is not None:
            hand_ref = self._ref_joints[self._frame()][:, hsl]           # (E,36) rad, retarget reference
            hand_target = torch.clamp(hand_ref + float(c.residual_scale_hands) * a_hand,
                                      self._ctrl_lower[hsl], self._ctrl_upper[hsl])
        # [/ROLLBACK MARKER: hand-residual] --------------------------------------------------------
        # hands: ABSOLUTE action — a_hand ∈ [-1,1] maps DIRECTLY to the
        # Shadow joint range (NOT residual-on-retarget, NOT delta). Per-group EMA-smoothed like the
        # grasp env's absolute finger path; the EMA prev is _smoothed_actions[hands] (seeded to the
        # reset pose in _reset_idx). residual_scale_hands is UNUSED in this mode.
        elif c.sonic_hand_delta:                                        # [DELTA] integrate hand JOINT increments
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

    def _frame(self) -> torch.Tensor:
        """(E,) EPISODE progress index: always counts UP from the start frame, both directions.
        Use this for the state cache and for anything that means "how far into the episode"."""
        return self._frame_idx.clamp(max=self._ref_len - 1)

    def _next_frame(self) -> torch.Tensor:
        return (self._frame_idx + 1).clamp(max=self._ref_len - 1)

    # [ROLLBACK MARKER: backward-dir] ------------------------------------------------------------
    # REFERENCE index. Backward envs read the SAME reference arrays mirrored in time, so instead of
    # storing a flipped copy of all 20 reference tensors we map the index:  arr_bwd[f] == arr[T-1-f].
    # That keeps every read site untouched and, critically, keeps all frame ARITHMETIC identical —
    # `_frame_idx` still counts up 0..T-1 for both directions, so `+1`, the SONIC future window,
    # Velocities cannot be handled by the index alone (v_bwd[f] = -v[T-1-f]); the four velocity read
    # sites multiply by `_dir_sign` explicitly.
    def _rframe(self, f: torch.Tensor | None = None) -> torch.Tensor:
        """(E,) index into the reference arrays for each env's own time direction."""
        f = self._frame() if f is None else f
        if not getattr(self, "_any_backward", False):
            return f
        return torch.where(self._dir_fwd, f, (self._ref_len - 1) - f)

    def _rnext_frame(self) -> torch.Tensor:
        return self._rframe(self._next_frame())

    def _canon_frame(self, f: torch.Tensor, fwd: torch.Tensor) -> torch.Tensor:
        """EPISODE frame -> ORIGINAL clip frame (the state cache's index space).

        The cache is shared by both directions and is keyed in ORIGINAL clip time, so a backward
        env's episode frame f — which is physically at original frame T-1-f — must be written to and
        read from that slot. Identical to `_rframe`'s mapping; kept separate because this one is
        about the CACHE index while `_rframe` is about the REFERENCE index (they coincide, but
        conflating them is exactly the kind of mistake that shows up as a silent training regression).
        """
        return torch.where(fwd, f, (self._ref_len - 1) - f)

    def _flip_cache_vel(self, state: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        """Negate the VELOCITY columns of a (N,222) cache row for backward envs.

        A time-reversed trajectory has the same poses and the opposite velocities, so converting a
        state between the two directions is: mirror the frame index (`_canon_frame`) AND negate every
        velocity channel. This mirrors RePHO exactly (intermimic.py:918/1040 negate
        `indices_for_velocity` around a `flip(2)` of the time axis).

        Negated (77 of 222): root lin/ang vel, object lin/ang vel, joint velocities.
        NOT negated: reward, root/object position + orientation, joint positions, and the smoothed
        action targets (those are POSITION targets, not rates — negating them would be wrong).
        `sign` is +1 for forward (no-op) and -1 for backward. The operation is its own inverse, so
        the same call serves both the write and the read direction.
        """
        out = state.clone()
        out[:, _CACHE_VEL_COLS] = out[:, _CACHE_VEL_COLS] * sign
        return out
    # [/ROLLBACK MARKER: backward-dir] -----------------------------------------------------------


    # ------------------------------------------------------------ observation
    def _get_observations(self) -> dict:
        c = self.cfg
        E, vs = self.num_envs, c.vel_obs_scale
        org = self.scene.env_origins                                    # (E,3)
        fr, nfr = self._rframe(), self._rnext_frame()   # [backward-dir] reference index

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
            in_contact = self._ref_link_contact_mask[nfr][:, self._ft_distal_idx].unsqueeze(-1).bool()  # (E,10,1)
            target_w = torch.where(in_contact, ref_vertex_w, ref_ft_w)
        else:
            target_w = ref_ft_w
        delta_ft_obj = math_utils.quat_apply(math_utils.quat_conjugate(oq_exp3), target_w - _tip)  # (E,10,3) ref − current
        C = [
            obj_p, _quat_to_6d(obj_q), obj_lv, obj_av * vs,            # (15)
            delta_obj_p, _quat_to_6d(delta_obj_q),                              # (9)
            delta_ft_obj.reshape(E, -1),                              # obj-local fingertip offset (30)
            artic,                                                    # (8)
            self._ref_link_contact_mask[nfr],                         # FUTURE per-link expected contact (L=32)
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
            # [ROLLBACK MARKER: backward-dir] PHASE BIT: 0 = forward, 1 = backward. One conditioned
            # policy serves both directions instead of RePHO's two separate policies. Constant 0 when
            # backward_ratio=0, so the extra dim is inert (but observation_space must still be 766).
            # TESTED AND KEPT. It was removed once (run 2026-08-02_21-03, obs 765) on the theory that
            # an explicit switch lets the network learn the two directions SEPARATELY and so blocks
            # the transfer that is the whole reason for training them together — the reference-delta
            # terms and the SONIC future window already imply which way time runs, so the bit looked
            # redundant. It was the WORST run of the seven: 21.69 peak against 51.80 with the bit,
            # and the two curves only diverged after ~12k, i.e. exactly when the two directions'
            # behaviours have to specialise. Time reversal is not dynamically valid (gravity does not
            # flip), so these really are two different tasks; the implicit cues are apparently not as
            # usable as an explicit flag, and without it one policy averages two conflicting
            # behaviours.
            (~self._dir_fwd).float().unsqueeze(-1),
        ]

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
        # Shift-append the newest row. Freshly-reset envs have no real history, so all 10 slots are
        # FABRICATED from the current (post-reset) row — see the seeding block below for what that
        # costs and what is done about it.
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
                # [ROLLBACK MARKER: hist-from-reference] ------------------------------------------
                # Fill the window with the reference's own last 10 frames instead of replicating the
                # current row. The window is then a real trajectory, and it is the same trajectory the
                # tokenizer feeds as the future, so SONIC sees one coherent story rather than
                # "frozen robot past + reference future".
                # Indices are built in EPISODE-frame space and clamped at 0, which handles both the
                # clip start (fewer than 10 frames exist -> repeat frame 0, and the reference really is
                # at rest there, so the repeated window is self-consistent) and BACKWARD episodes (an
                # episode-frame past maps through _canon_frame to original frames AHEAD of the start).
                # _dir_sign negates the two velocity terms for backward envs, as the reset path does.
                # Deliberately ALL 10 slots, newest included: leaving the newest as the measured row
                # would put the tracking error as a one-frame position jump between slots 8 and 9,
                # which the velocity channel would contradict — the exact defect being removed here.
                _hist_ref_used = bool(getattr(c, "sonic_hist_from_reference", True)) and self._ref_hist is not None
                if _hist_ref_used:
                    _H = self._sonic_hist["jpr"].shape[1]                       # 10
                    _ep = self._frame().unsqueeze(1) - torch.arange(_H - 1, -1, -1, device=self.device)
                    _ep = _ep.clamp(min=0)                                      # (E,H) episode frames
                    _of = self._canon_frame(_ep, self._dir_fwd.unsqueeze(1))    # (E,H) original frames
                    _sg = self._dir_sign.unsqueeze(-1)                          # (E,1,1) velocity flip
                    for k in ("jpr", "grav", "act"):
                        self._sonic_hist[k][m] = self._ref_hist[k][_of][m]
                    for k in ("jvr", "ang"):
                        self._sonic_hist[k][m] = (self._ref_hist[k][_of] * _sg)[m]
                else:
                    for k in ("ang", "jpr", "jvr", "grav"):
                        self._sonic_hist[k][m] = rows[k][m].unsqueeze(1)
                # [/ROLLBACK MARKER: hist-from-reference] -----------------------------------------
                # [ROLLBACK MARKER: hist-seed-zero-vel] --------------------------------------------
                # Seeding copies ONE row into all 10 slots, which freezes joint POSITION across the
                # window. Copying the live joint VELOCITY alongside it then states something that
                # cannot happen — "the joints did not move for 10 frames, yet they are moving" — and
                # the frozen decoder never saw such a history in training. Warm-up hid it, because
                # the reference reset path leaves jvel at 0 and the two agreed by accident; cache
                # restores in the adaptive phase carry real velocity, and that is where the
                # reset-vs-settled body-target jump ratio climbed (2.29x warm-up -> 2.16x adaptive
                # after the diagnostic fix).
                # Zeroing the seeded velocity makes the fabricated window self-consistent: it now
                # reads as "at rest for 10 frames", a state SONIC has seen. It is a claim about the
                # PAST only — the robot keeps the velocity the cache restored, and the very next step
                # shift-appends the real row, so the lie survives one frame and then decays out.
                # The alternative was back-integrating jpr from jvr (a constant-velocity past, closer
                # to the truth but able to walk positions out of their limits); this is the simpler
                # of the two and the one chosen.
                # JOINT velocity only. `ang` (base angular velocity) keeps the original seeding — the
                # robot's measured root_ang_vel_b — in BOTH phases. It carries the same internal
                # contradiction on paper (`grav`, i.e. base orientation, is frozen across the window
                # while ang is not), but the three candidate fixes disagree there in a way they do not
                # for jvr: measured is the TRUE value, the reference is what warm-up happens to feed,
                # and zero is what the frozen window implies — and in the adaptive phase those are
                # three different numbers. jvr has no such conflict: warm-up's measured value is
                # ALREADY zero (its reset path never writes jvel), so zeroing is simultaneously "what
                # warm-up does" and "what the frozen positions imply". Only that unambiguous case is
                # changed here.
                # Only meaningful on the REPLICATED fallback: with hist_from_reference the window
                # already carries the reference's own velocities, which agree with its positions.
                if _hist_ref_used:
                    pass
                elif getattr(c, "sonic_hist_seed_zero_vel", True):
                    self._sonic_hist["jvr"][m] = 0.0
                # [/ROLLBACK MARKER: hist-seed-zero-vel] -------------------------------------------
                # [ROLLBACK MARKER: act-seed-from-pose] --------------------------------------------
                # Seed the action history with the action that WOULD command the pose the robot is
                # actually in, not zero. Exact inverse of the decode in _sonic_pre_physics_step:
                #     body_sonic = sonic_default + sonic_scale * a_sonic
                #  => a_sonic    = (joint_pos_sonic - sonic_default) / sonic_scale = jpr / sonic_scale
                # Zero meant "I have been commanding the DEFAULT pose for the last 10 frames", which
                # contradicts a robot restored mid-manipulation and is out of distribution for the
                # frozen decoder that produces the very next body target. Measured on the 10k knife
                # run: the reset-vs-settled body-target jump ratio was already 2.29x during WARM-UP,
                # where joint velocities are zero (the reference reset path leaves jvel at 0) and this
                # was therefore the ONLY remaining fabrication — so 2.29x is what this line targets.
                # NOTE the history still freezes jpr across all 10 slots while jvr is non-zero, which
                # is a SEPARATE contradiction (only live once cache restores begin, hence the ratio
                # rising to 4.15x in the adaptive phase). That one needs the velocity-consistent or
                # cache-assembled seed and is NOT addressed here.
                # To roll back: `self._sonic_hist["act"][m] = 0.0`.
                if not _hist_ref_used:
                    # GRAIL/IsaacLab equivalent: at reset the action manager zeroes the action, so the
                    # value CircularBuffer replicates into every slot is 0. The jpr/sonic_scale seed
                    # below was ours, not SONIC's; it is kept documented but disabled so the whole
                    # seeding path matches what the frozen decoder actually saw in training.
                    if getattr(c, "sonic_act_seed_from_pose", False):
                        self._sonic_hist["act"][m] = (rows["jpr"][m] / self._sonic_scale).unsqueeze(1)
                    else:
                        self._sonic_hist["act"][m] = 0.0
                # [/ROLLBACK MARKER: act-seed-from-pose] -------------------------------------------
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

    # ---------------------------------------------------------------- rewards
    def _compute_errors(self):
        """Shared error terms for reward + termination."""
        org = self.scene.env_origins
        fr = self._rframe()                              # [backward-dir] reference index
        kpts = self._robot_kpts_w() - org[:, None, :]
        ref = self._ref_kpts[fr]
        dk = ref - kpts
        _nb = len(BODY_KPTS)                                          # 14 body kpts (GRAIL-aligned)
        body_per = dk[:, :_nb].norm(dim=-1)                          # (E,14) per-body-kpt distance
        body_err = body_per.mean(dim=-1)                            # (E,) UNIFORM mean → termination gate
        body_core_err = body_per[:, self._body_core_idx].mean(dim=-1)  # (E,) 10 CORE body kpts (REWARD)
        ee_err = body_per[:, self._ee_kpt_idx].mean(dim=-1)         # (E,) 4 END-EFFECTOR kpts: L/R wrist+ankle (REWARD)
        wrist_pos_err = body_per[:, self._wrist_kpt_idx].mean(dim=-1)  # (E,) MEAN over both wrists (termination)
        hand_err = dk[:, _nb:].norm(dim=-1).mean(dim=-1)            # (E,) 40 hand kpts (REWARD only)
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
            obj_pos_err = (self._ref_obj_pos[fr] - obj_pos).norm(dim=-1)
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
            in_contact = self._ref_link_contact_mask[fr][:, self._ft_distal_idx].unsqueeze(-1).bool()  # (E,10,1)
            ft_target = torch.where(in_contact, ref_vtx_w, ref_ft_drift)          # (E,10,3)
            ft_reward = (ft_target - tip_l).norm(dim=-1).mean(dim=-1)             # (E,) contact-conditioned
        else:
            obj_pos_err = torch.zeros(self.num_envs, device=self.device)
            obj_rot_err = torch.zeros(self.num_envs, device=self.device)
            ft_reward = ft_err                                                     # no object → raw pad target
        # palm/wrist rotation deviation (bimanual, MEAN over both hands). Robot palm body quat vs the retarget
        # reference palm quat (same robot0_{l,r}_palm frame → direct compare, no landmark conversion).
        if self._has_palm_ref:
            palm_q = _canon(self.robot.data.body_quat_w[:, self._palm_body_ids])      # (E,2,4)
            pqe = _canon(math_utils.quat_mul(self._ref_palm_quat[fr],
                                             math_utils.quat_conjugate(palm_q)))       # (E,2,4)
            wrist_rot_err = 2.0 * torch.arcsin(pqe[..., 1:].norm(dim=-1).clamp(max=1.0)).mean(dim=-1)  # (E,)
        else:
            wrist_rot_err = torch.zeros(self.num_envs, device=self.device)
        # [ROLLBACK MARKER: track-buffer] the ACHIEVED kinematics, in the same quantities and frame
        # convention the reward compares against. Kept so a successful rollout can later replace the
        # retarget reference at frames where it is wrong (SupMat Alg 3, lines 54-73). Cheap when the
        # feature is off: nothing is allocated and this branch never runs.
        if self._track_kin is not None:
            self._last_kin = torch.cat([
                kpts.flatten(1),                                   # (E, K*3) body + hand keypoints
                (tip - org[:, None, :]).flatten(1),                # (E, 30)  fingertip pads
                root_pos, root_quat,                               # (E, 3), (E, 4)
                (self._object.data.root_pos_w - org) if self._has_object else torch.zeros_like(root_pos),
                _canon(self._object.data.root_quat_w) if self._has_object else torch.zeros_like(root_quat),
            ], dim=1)
        return dict(body=body_err, body_core=body_core_err, ee=ee_err, com_support=self._com_support_err(), wrist_pos=wrist_pos_err, hand=hand_err, ft=ft_err, ft_reward=ft_reward, ft_per=ft_per, tip=tip,
                    pad_inward=pad_inward, root_pos=root_pos_err, root_rot=root_rot_err,
                    obj_pos=obj_pos_err, obj_rot=obj_rot_err, root_quat=root_quat,
                    wrist_rot=wrist_rot_err)

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        e = self._errs                                    # set by _get_dones (runs first each step)
        fr = self._rframe()                              # [backward-dir] reference index

        # per-LINK contact FORCE (Option A / DexMachina): compressive object contact force on each of the 32
        # wrap links (force_matrix · the link's OWN inward pad normal), gated by (a) the reference per-link
        # contact MASK (which links SHOULD touch), (b) a SPATIAL gate — the robot link is near the prescribed
        # object-surface target (so a link touching the object at the WRONG spot is NOT rewarded), and (c) an
        # ORIENTATION gate — the link's inward pad normal is aligned (≤ contact_normal_gate_tol) with the
        # reference reaction normal, so touching with the WRONG face (e.g. back of the palm) is NOT rewarded.
        # Normalized ∈[0,1] over the active links. Full grasp WRAP (palm+phalanges+tips).
        link_force = self._link_contact_forces()                      # (E,L) compressive per link (on own face)
        link_mask = self._ref_link_contact_mask[fr]                   # (E,L) which links should touch
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

        # FOOT contact obs/reward REMOVED (2026-07-20, GRAIL-aligned): feet + balance are owned by the
        # frozen SONIC base; the residual policy neither observes nor rewards foot contact/force/flatness.
        # Feet (ankles) are still tracked via the END-EFFECTOR body reward (rew_ee_kpts·e["ee"]).

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

        # action regularization (HANDS only — legs/arms/waist are SONIC-driven, regularized by rew_latent_reg)
        # + hand pose_reg_hands + action rate.
        hsl = self._group_slices["hands"]
        # [ROLLBACK MARKER: hand-residual] in residual mode the meaningful magnitude is the RESIDUAL
        # the policy commanded, not the absolute target: _smoothed_actions is the normalized target,
        # so squaring it asks the fingers to sit near the middle of their range — which penalises the
        # retarget reference itself whenever the reference grasp is a closed hand. Measured at -0.076,
        # 3.6x pose_reg_hands, so it is not a rounding error. a_hand is already clamped to [-1,1].
        _resid = getattr(self.cfg, "sonic_hand_residual", False) and self._ref_joints is not None
        if _resid and hasattr(self, "_last_a_hand"):
            reg_hands = (self._last_a_hand ** 2).sum(-1)
        else:
            reg_hands = (self._smoothed_actions[:, hsl] ** 2).sum(-1)
        # pose_reg_hands (HANDS only): pull achieved hand joints toward the DEFAULT (rest/neutral) pose — a
        # task-agnostic regularizer that keeps fingers out of extreme/unnatural configs and damps jitter.
        # NOT toward the retarget reference: the hands are already tracked by rew_hand_kpts + rew_fingertip
        # (inside the tracking_penalty group), so a retarget-target pose_reg_hands would merely duplicate
        # tracking in joint space with zero neutral-pose safety. Body is SONIC-driven so it is NOT regularized
        # here. Matches the grasp ancestor (jp - default_joint_pos) and TJ (dof_pos² toward the rest pose);
        # GRAIL/SONIC carry no pose regularizer at all (tracking owns the reference; only limit/rate/contact).
        _hand_ids = self._action_joint_ids_t[hsl]
        # [ROLLBACK MARKER: hand-residual] the anchor follows the control parameterisation. Under the
        # ABSOLUTE mapping the target carried no reference, so the only safe anchor was the neutral
        # pose (the note above). Under the RESIDUAL mapping the target IS the reference plus a bounded
        # correction, so the natural regulariser is the REALISED deviation from that reference — it
        # covers the PD tracking error too, which the commanded-residual term (reg_hands) cannot see.
        # Anchoring to `default` here instead would pull the fingers away from every closed-hand
        # reference frame, i.e. fight the thing the residual parameterisation exists to track.
        if _resid:
            hand_ref = self._ref_joints[self._frame()][:, hsl]
        else:
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
        action_rate = ((self._cur_policy_action - self._prev_policy_action) ** 2).sum(-1)
        # SONIC latent-residual L2 penalty (GRAIL LatentL2): keep z_res small so the learned residual
        # does not override the frozen SONIC prior. mean over the 64 latent dims, computed on the RAW
        # (unclipped) residual — see the _last_z_res assignment in _sonic_pre_physics_step for why
        # (a clipped operand has no restoring gradient outside the clip). UNBOUNDED as a result; the
        # clipped form's ceiling was rew_latent_reg · sonic_z_res_clip² (0.25 at clip 5, 4.0 at 20).
        # 0 when SONIC is off.
        latent_reg = ((self._last_z_res ** 2).mean(dim=-1) if self._sonic is not None
                      else torch.zeros(self.num_envs, device=self.device))
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
        tracking_penalty = (
            c.rew_body_kpts * e["body_core"]               # 10 CORE body kpts
            + c.rew_ee_kpts * e["ee"]                      # 4 END-EFFECTOR kpts (L/R wrist + L/R ankle)
            + c.rew_hand_kpts * e["hand"] + c.rew_fingertip * e["ft_reward"]
            + c.rew_root_pos * e["root_pos"] + c.rew_root_ori * e["root_rot"]
            + c.rew_obj_pos * e["obj_pos"] + c.rew_obj_rot * e["obj_rot"]
        )
        reward = (
            c.rew_alive * alive
            + tracking_penalty.clamp(min=-c.rew_alive)
            + c.rew_contact_force * force_rew
            + c.rew_action_reg_hands * reg_hands + c.rew_pose_reg_hands * pose_reg_hands
            + c.rew_action_rate * action_rate
            + c.rew_latent_reg * latent_reg
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
        ep_rew = {
            "alive": c.rew_alive * alive,
            "body_kpts": c.rew_body_kpts * e["body_core"],  # 10 CORE body kpts
            "ee_kpts": c.rew_ee_kpts * e["ee"],             # 4 END-EFFECTOR kpts (L/R wrist + L/R ankle)
            "hand_kpts": c.rew_hand_kpts * e["hand"],
            "fingertip": c.rew_fingertip * e["ft_reward"],
            "root_pos": c.rew_root_pos * e["root_pos"],
            "root_ori": c.rew_root_ori * e["root_rot"],
            "obj_pos": c.rew_obj_pos * e["obj_pos"],
            "obj_rot": c.rew_obj_rot * e["obj_rot"],
            "contact_force": c.rew_contact_force * force_rew,         # per-link (Option A) contact-force reward
            "action_reg": c.rew_action_reg_hands * reg_hands,
            "pose_reg_hands": c.rew_pose_reg_hands * pose_reg_hands,
            "action_rate": c.rew_action_rate * action_rate,
            "latent_reg": c.rew_latent_reg * latent_reg,
            "com_support": c.rew_com_support * com_support,
            "feet_contact_match": c.rew_feet_contact_match * feet_match,
        }
        self._log_reward_terms(e, tracking_penalty, ep_rew, fr)
        return reward

    def _log_reward_terms(self, e, tracking_penalty, ep_rew, fr):
        log = self.extras.setdefault("log", {})
        log.update({
            "Error / body_kpts": e["body"].mean(), "Error / ee_kpts": e["ee"].mean(),
            "Error / com_support": e["com_support"].mean(),
            "Error / hand_kpts": e["hand"].mean(),
            "Error / fingertip": e["ft"].mean(), "Error / root_pos": e["root_pos"].mean(),
            "Error / root_rot": e["root_rot"].mean(), "Error / obj_pos": e["obj_pos"].mean(),
            "Error / wrist_rot": e["wrist_rot"].mean(),
            # [ROLLBACK MARKER: sonic-encoder-g1] wrist POSITION was computed (it is a termination
            # gate, term_wrist_pos_err) but never logged. It is the one metric that separates the
            # smpl and g1 tokenizer encoders — frozen-prior playback puts them at 5.61 vs 8.90 cm
            # while body_kpts ties at 7.2/7.3 — so without it an encoder A/B cannot be read at all.
            "Error / wrist_pos": e["wrist_pos"].mean(),
            "Curriculum / cache_coverage": float(self._slot_usable().any(dim=2).any(dim=0).sum()) / self._ref_len,
            # [ROLLBACK MARKER: rsi-phase-split] pool health. n_candidates should GROW after the
            # warm-up ends (targets reappear as gaps fill); flat = something is not being reached.
            # runup is the mean pick-start distance: ~adaptive_back_frames is healthy, small means
            # the cache is sparse just behind the sampled targets.
            "Curriculum / n_candidates": float(getattr(self, "_diag_n_candidates", self._ref_len)),
            "Curriculum / warmup_active": float(getattr(self, "_in_warmup", True)),
        })
        # [ROLLBACK MARKER: jerk-diag] separates FSQ bin-flip jitter (H1) from reset-history lurch (H2)
        log.update(getattr(self, "_diag_jerk", {}) or {})
        # [ROLLBACK MARKER: cache-score-rework] cache health. `overwrite` is how many frames were
        # replaced by the last flush — if it decays to ~0 the cache has frozen and the decay term is
        # too weak (or the scores have saturated). `score_mean` is the mean over OCCUPIED frames.
        _occ = self._slot_occ
        log["Cache / overwrite"] = float(getattr(self, "_diag_cache_overwrite", 0.0))
        # [ROLLBACK MARKER: spawn-shift] how far does PhysX move the object away from the pose the
        # reset COMMANDED? The reference object pose is human-capture data and can interpenetrate
        # the support it rests on; the solver then pushes it out on the first step, so the object
        # the policy must reach for is not where the reference says it is. Measured on envs in
        # their first two steps, against the reference pose at their own start frame.
        if self._has_object:
            _fresh = self.episode_length_buf <= 2
            if bool(_fresh.any()):
                _sf = self._episode_start_frame[_fresh]
                _rf = torch.where(self._dir_fwd[_fresh], _sf, (self._ref_len - 1) - _sf)
                _d = ((self._object.data.root_pos_w[_fresh] - self.scene.env_origins[_fresh])
                      - self._ref_obj_pos[_rf]).norm(dim=-1)
                log["Diag / obj_spawn_shift_cm"] = float(_d.mean()) * 100.0
                log["Diag / obj_spawn_shift_max_cm"] = float(_d.max()) * 100.0
        log.update(getattr(self, "_diag_grasp", {}) or {})   # [ROLLBACK MARKER: grasp-span]
        _q = getattr(self, "_diag_start_q", None)
        if _q is not None:
            for _k, _v in zip(("p10", "p50", "p90", "frac_grasp", "frac_lt100"), _q):
                log[f"StartHist / {_k}"] = _v
        # [ROLLBACK MARKER: backward-dir] PER-DIRECTION metrics. Aggregates hide a one-sided collapse:
        # "forward improves while backward dies" averages out to "fine". Split the headline tracking
        # error and the episode length so a failing direction is visible on its own curve.
        if getattr(self, "_any_backward", False) and hasattr(self, "_errs"):
            bwd = ~self._dir_fwd
            log["Curriculum / backward_frac"] = float(bwd.float().mean())
            # what the SAMPLER drew this reset (vs backward_frac = share of LIVE envs, which is
            # skewed by how long each direction survives). Drifts away from backward_ratio once the
            # (2,F) failure table takes over in the adaptive phase — that drift IS the mechanism.
            # [ROLLBACK MARKER: dir-decouple] cosine overlap of the two directions' failure profiles
            # AFTER aligning them to original clip time. LOW = the directions die in DIFFERENT places,
            # i.e. backward is covering frames forward cannot reach — the split is doing its job.
            # Rising toward 1 = both die at the same frames, so no direction mix will help there and
            # the fix belongs elsewhere (reward shaping / termination thresholds).
            log["Cache / slot_ref_frac"] = float(getattr(self, "_diag_slot_ref_frac", 1.0))
            log["Cache / slot_entropy"] = float(getattr(self, "_diag_slot_entropy", 0.0))
            # [ROLLBACK MARKER: repho-cache] score_mean is survival length under repho_length_score,
            # so it should climb toward the clip length; if it plateaus low the cache is not finding
            # states worth restarting from. switch_cov is what gates leaving the uniform phase.
            if getattr(self.cfg, "repho_switch", False):
                log["Curriculum / switch_coverage"] = self._diag_switch_cov
            if getattr(self.cfg, "repho_length_score", False):
                _o = self._slot_occ
                if bool(_o.any()):
                    log["Cache / length_mean"] = float(self._state_cache[:, :, :, 0][_o].mean())
                    log["Cache / length_max"] = float(self._state_cache[:, :, :, 0][_o].max())
                    log["Cache / return_mean"] = float(self._cache_return[_o].mean())
            # [ROLLBACK MARKER: failure-rate] does the hazard actually surface the grasp? `visit_frac`
            # is the share of survivor-mass sitting in the pre-grasp window (small by construction —
            for nm in ("body", "obj_pos", "obj_rot", "ft", "root_pos"):
                v = self._errs.get(nm)
                if v is None:
                    continue
                if bool((~bwd).any()):
                    log[f"DirFwd / err_{nm}"] = float(v[~bwd].mean())
                if bool(bwd.any()):
                    log[f"DirBwd / err_{nm}"] = float(v[bwd].mean())
            el = self.episode_length_buf.float()
            if bool((~bwd).any()):
                log["DirFwd / episode_len"] = float(el[~bwd].mean())
            if bool(bwd.any()):
                log["DirBwd / episode_len"] = float(el[bwd].mean())
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
            log["Diag / tracking_penalty"] = tracking_penalty.clamp(min=-self.cfg.rew_alive).mean()
            log["Diag / tracking_penalty_raw"] = tracking_penalty.mean()
            log["Diag / clamp_frac"] = (tracking_penalty < -self.cfg.rew_alive).float().mean()
        # RSI / episode-length diagnostics. Episodes are variable length now (start frame → sequence
        # end), so skrl's "Reward / Total reward" (episode SUM) is length-confounded and no longer
        # comparable across configs — these two make the length distribution itself observable.
        # episode_len = steps elapsed in the CURRENT episode; rsi_start = frame the episode began at.
        log["Diag / stage_frac"] = float(getattr(self, "_diag_stage_frac", 1.0))
        log["Diag / contact_lost_frac"] = float(getattr(self, "_diag_contact_lost", 0.0))
        for _i, _n in enumerate(getattr(self, "_contact_ch_names", ())):
            log[f"Diag / clost_{_n}"] = float(self._diag_clost[_i])
        log["Diag / episode_len_mean"] = self.episode_length_buf.float().mean()
        log["Diag / rsi_start_mean"] = self._episode_start_frame.float().mean()
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
        # [ROLLBACK MARKER: contact-term] SupMat S.6 (ii): required body-object contact lost for
        # over N consecutive frames ends the episode. Only frames where the REFERENCE asks for
        # contact can break the streak, so a no-contact approach phase is unaffected (knife: the
        # first ~31 frames carry no reference contact).
        if self._has_object and self._has_link_contact and c.contact_loss_frames > 0:
            _f = self._link_contact_forces() > c.contact_force_thresh      # (E,L) achieved
            _rf = self._ref_link_contact_mask[self._rframe()] > 0.5        # (E,L) demanded
            _h = N_LINK_CONTACT // 2                                       # links per hand; [0] = palm
            # The DEMAND is per hand ("this hand should be touching something"), matching RePHO's
            # `ref_left_contact_hand_any` over the whole 16-id hand block; the FAILURE is then split
            # into fingers and palm.
            _req_l, _req_r = _rf[:, :_h].any(dim=-1), _rf[:, _h:].any(dim=-1)
            _viol = torch.stack([
                _req_l & ~_f[:, 1:_h].any(dim=-1),
                _req_r & ~_f[:, _h + 1:].any(dim=-1),
                _req_l & ~_f[:, 0],
                _req_r & ~_f[:, _h],
            ], dim=1)                                                      # (E,4)
            self._contact_lost = torch.where(_viol, self._contact_lost + 1,
                                             torch.zeros_like(self._contact_lost))
            # RePHO zeroes it for the first steps after a reset (:1920, `progress > 1 + start_times`):
            # an RSI restore lands with zero joint velocity and the object re-placed, so contact needs
            # a moment to form.
            self._contact_lost *= (self.episode_length_buf >= int(c.contact_loss_grace)).unsqueeze(-1).long()
            if self._contact_active.numel():
                _act = self._contact_lost[:, self._contact_active]
                d = d | (_act > c.contact_loss_frames).any(dim=-1)
                self._diag_contact_lost = float(
                    _viol[:, self._contact_active].any(dim=-1).float().mean())
            self._diag_clost = _viol.float().mean(dim=0).tolist()   # all four, active or not
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
        # (start frame → sequence end), so the end must be detected on the PER-ENV frame index.
        # SECOND term = safety cap only; with episode_length_s = ref_len/action_fps the two coincide
        # exactly for an env that started at frame 0, and the cap never fires before the frame check.
        # NOTE the bootstrap value skrl adds is V(next_observations), and IsaacLab auto-resets inside
        # step() → that is the value of the NEW episode's RSI start state, not of the true successor.
        # If end-of-sequence states ever look over-valued, suspect this first.
        time_out = (self._frame_idx >= self._ref_len - 1) | (self.episode_length_buf >= self.max_episode_length - 1)
        return self._died, time_out

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

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids) -> None:
        # [ROLLBACK MARKER: deferred-cache] MUST run before super(), which zeroes episode_length_buf —
        # the whole point of the deferral is to filter on how long the episode actually lasted.
        # getattr, not a bare attribute: DirectRLEnv.__init__ resets every env, and that happens
        # BEFORE _post_init_buffers allocates the staging tensors.
        if getattr(self, "_pend_state", None) is not None:
            self._flush_state_cache(env_ids, self.episode_length_buf[env_ids].clone())
        # [ROLLBACK MARKER: grasp-span] how do episodes that START inside the grasp window fare
        # against the rest? For s101_seg12_knife the hand reaches the knife at frame 29 and the knife
        # starts moving at 41, so [0,41) is the only part of the clip where the grasp itself is
        # practised — everything later begins with the object already held. The start-frame histogram
        # shows 11-16% of episodes DO begin there (above the 8.9% a uniform draw would give), so the
        # curriculum is not avoiding the grasp; the open question is whether those episodes survive.
        if len(env_ids):
            _st = self._episode_start_frame[env_ids]
            _el = self.episode_length_buf[env_ids].float()
            _g = _st < 41
            def _m(t: torch.Tensor, k: torch.Tensor) -> float:
                return float(t[k].mean()) if bool(k.any()) else float("nan")
            self._diag_grasp = {
                "Grasp / eplen_start_in": _m(_el, _g), "Grasp / eplen_start_out": _m(_el, ~_g),
                "Grasp / frac_of_resets": float(_g.float().mean()),
            }
        super()._reset_idx(env_ids)
        c = self.cfg
        n = len(env_ids)
        dev = self.device

        # [ROLLBACK MARKER: backward-dir] -------------------------------------------------------
        # Draw each resetting env's time direction. MUST come after the cache flush and the failure
        # -count update above (both belong to the episode that just ENDED and need its OLD direction)
        # and before the candidate/sampling block below (the coverage mask is direction-dependent).
        br = float(getattr(c, "backward_ratio", 0.0))
        self._use_backward = br > 0.0 and c.use_rsi and c.adaptive_sampling
        if self._use_backward:
            self._any_backward = True                                  # arms _rframe / cache mapping
        # [/ROLLBACK MARKER: backward-dir] ------------------------------------------------------

        # ---- adaptive frame sampling ----
        self._sampling_step_count += 1
        # [ROLLBACK MARKER: cross-buffer] RePHO runs this every 10 epochs (Alg 3 line 41); with its
        # horizon of 32 that is ~320 control steps. In-memory there is no disk round trip, so the
        # interval is only about how stale an imported state may be.
        _ci = int(getattr(c, "cross_interval", 0))
        if getattr(c, "cross_buffer", False) and _ci > 0 and self._sampling_step_count % _ci == 0:
            self._cross_transfer()
        _ti = int(getattr(c, "track_interval", 0))
        if (self._track_kin is not None and _ti > 0 and self._sampling_step_count % _ti == 0
                and self._sampling_step_count >= int(getattr(c, "track_start_step", 0))):
            # RePHO gates the swap on `curr_epoch >= 54400` against a run that starts at 53001 and
            # ends near 56000 — the reference is untouchable for roughly the first half. An absolute
            # step floor is the same statement in our units; there is no readiness term on THIS path
            # (that one lives in load_ref_traj, its other swap path).
            self._apply_tracking_update()
            # [ROLLBACK MARKER: track-harvest] re-aim the harvest slice at the current best state
            # (SupMat Alg 3 line 57). Only the FRAME is forced: the slot lottery already weights by
            # survival length, so the best slot at that frame is the likely draw anyway.
            if bool(self._harvest.any()) and self._n_slots > 1:
                # RePHO save_run_val:1952-1954 — the val rollout starts at argmax of the FLOORED,
                # SLOT-SUMMED buffer value, but falls back to frame 0 whenever frame 0 is within 10
                # of that best. Starting at 0 whenever it is nearly as good keeps the candidate
                # covering the clip's opening, which is what the swap most needs to be able to fix.
                _fl = float(getattr(c, "repho_sample_floor", 7.0))
                for _d in (0, 1):
                    _Ld = torch.where(self._slot_occ[_d], self._state_cache[_d, :, :, 0],
                                      torch.zeros_like(self._state_cache[_d, :, :, 0]))
                    _rv = (_Ld - _fl).clamp(min=0.0).sum(dim=1)            # (F,) slot sum
                    _cd = int(_rv.argmax())
                    self._harvest_frame[_d] = 0 if float(_rv[0]) > float(_rv[_cd]) - 10.0 else _cd
        have_train = self._slot_usable().any(dim=2).any(dim=0)
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
        cand_idx = cand_idx[(cand_idx >= self._init_range_left) & (cand_idx < self._init_range_right)]
        if cand_idx.numel() == 0:                                      # bounds swallowed the pool
            cand_idx = torch.full((1,), min(self._init_range_left, self._init_range_right - 1),
                                  dtype=torch.long, device=dev)

        # ── RSI START-FRAME CURRICULUM (cold-start mitigation) ─────────────────────────────
        # Ramp the sampleable start-frame CEILING from ~frame 0 to the full trajectory over the
        # first rsi_curriculum_steps control steps, so a COLD value/policy/obs-scaler (no pretrain
        # warm-start) warms up on a narrow, well-conditioned near-frame-0 distribution before deep
        # frames appear. Diverse-from-step-0 RSI onto cold nets is what ignites the PPO ratio
        # explosion (confirmed via run diff: appeared with episode 5→3 [start range 1→101] +
        # warmstart T→F). Restricts the SHARED candidate pool so it applies to BOTH sampling
        # [0, upper], so an early small ceiling keeps start pinned near 0. Reference-seeded RSI
        # only (every frame restorable); with a cache-coverage pool the crawl already gates starts.
        # rsi_curriculum_steps<=0 → disabled (diverse-from-step-0, previous behaviour).
        if (self._ref_joints is not None and getattr(c, "rsi_curriculum_steps", 0)
                and c.rsi_curriculum_steps > 0):
            prog = min(1.0, self._sampling_step_count / float(c.rsi_curriculum_steps))
            ceil_f = max(int(prog * (self._ref_len - 1)), 1)          # target-frame ceiling: 0 → ref_len-1
            cand_idx = cand_idx[cand_idx <= ceil_f]
            if cand_idx.numel() == 0:                                 # safety (frame 0 is always restorable)
                cand_idx = torch.zeros(1, dtype=torch.long, device=dev)

        # PRETRAIN (failure_weighted_sampling=False) → always UNIFORM over cached frames.
        # TRAIN → the repho_switch block below decides it (readiness, not a fixed step count).
        # _sampling_step_count increments once per _reset_idx, i.e. once per control step.
        use_uniform = (
            (not c.adaptive_sampling)
            or (not c.failure_weighted_sampling)
        )
        # [ROLLBACK MARKER: repho-cache] readiness-gated exit instead of a fixed step count. RePHO
        # leaves uniform on sum(L>25)>3 AND epoch>30, with a relaxed sum(L>12)>3 at epoch>150 so a
        # hard clip is not trapped; the step count is only ever its floor. Coverage is the quantity
        # we have that asks the same question. Without this the switch fires whether or not the pool
        # can support it — measured with a short warm-up, usable frames went 501 -> 16 at the boundary.
        # getattr on _state_cache, not a bare attribute: DirectRLEnv.__init__ resets every env and
        # that happens BEFORE _post_init_buffers allocates the cache, so a bare read raises there.
        if (getattr(c, "repho_switch", False) and c.adaptive_sampling and c.failure_weighted_sampling
                and getattr(self, "_state_cache", None) is not None):
            # NOT `n`: that name is bound at the top of _reset_idx to len(env_ids) and is what the
            # start-frame multinomial below draws with. Shadowing it made that draw take
            # _sampling_step_count samples instead of one per resetting env — at the first reset, one
            # frame for all 256 envs, then an out-of-range index deeper in as a device-side assert.
            nstep = self._sampling_step_count
            # RePHO's own test, now that column 0 IS a survival length: count entries that survived
            # past a bar and require more than a handful (intermimic.py:1902-1905). Coverage was tried
            # first and is USELESS here — _slot_occ[:, 0] is pre-set True for every frame at init, so
            # `any usable slot` is 1.0 from step 0 and the condition never bites.
            L = self._state_cache[:, :, :, 0]
            occ = self._slot_occ
            n_hi = int(((L > c.repho_switch_len_hi) & occ).sum())
            n_lo = int(((L > c.repho_switch_len_lo) & occ).sum())
            self._diag_switch_cov = float(n_hi)
            use_uniform = (
                nstep < c.repho_switch_min_steps
                or (n_hi <= c.repho_switch_count and nstep < c.repho_switch_relax_steps)
                or (n_lo <= c.repho_switch_count and nstep < c.repho_switch_max_steps)
            )
        self._in_warmup = bool(use_uniform)   # [ROLLBACK MARKER: rsi-phase-split] read by _flush_state_cache
        # [ROLLBACK MARKER: rsi-phase-split] ------------------------------------------------------
        # target with an empty window is simply not in the pool this reset — it comes back once an
        # episode launched from an earlier target rolls forward and fills the gap.
        # [backward-dir] the coverage mask is DIRECTION-DEPENDENT: the cache is keyed in ORIGINAL clip
        # time, so a backward env asking "is my episode-frame f cached?" must look at original frame
        # T-1-f. In episode-frame space that is simply the flipped mask. Build one pool per direction
        # and sample each group of envs from its own.
        # [ROLLBACK MARKER: slot-cache] freeze the bad-reference verdict at the warm-up boundary.
        self._diag_n_candidates = int(cand_idx.numel())
        # [/ROLLBACK MARKER: rsi-phase-split] -----------------------------------------------------

        # [ROLLBACK MARKER: backward-dir] JOINT (direction, frame) sampling ------------------------
        # In the adaptive phase the DIRECTION is no longer a fixed ratio: it is drawn together with
        # the target frame from the (2,F) failure table, so "which frame, entered which way, is the
        # policy failing at right now" decides both. `backward_ratio` then only sets the warm-up mix
        # (and gates the feature off entirely at 0).
        # Warm-up keeps the simple split: Bernoulli(backward_ratio) x uniform frame.
        allow = torch.zeros(2, self._ref_len, dtype=torch.bool, device=dev)
        allow[0, cand_idx] = True
        if self._use_backward:
            allow[1, cand_idx] = True
        if not bool(allow[0].any()) and not bool(allow[1].any()):      # degenerate: forward pool
            allow[0, cand_idx] = True

        # [ROLLBACK MARKER: slot-cache] DIRECTION FIRST, then each direction draws its own frame from
        # its own budget. Previously both were drawn jointly from a (2,F) failure table, so raising
        # backward's weight at frame f LOWERED forward's there — the two competed for one budget and
        # forward lost exactly the frames it most needed to practise. Splitting the budget removes the
        # competition while still pointing both directions at the same hard frames.
        # [ROLLBACK MARKER: dir-decouple] ONE draw over all (direction, frame) cells. The (2,F) weight
        # table is flattened to 2*F tickets, so `flat // F` is the direction and `flat % F` the frame.
        # Because every cell shares one normalisation, raising backward's weight at a frame LOWERS
        # forward's there — the budgets compete. Per-direction budgets were tried (run "F") and lost
        # (44.18 vs 51.80); the competition is apparently what spreads the two directions over
        # different parts of the clip rather than piling both onto the same frames.
        af = allow.float()
        if use_uniform:
            # Traverse RSI: uniform over each direction's allowed frames. BOTH directions warm up —
            # with a fixed partition the backward envs exist from step 0, and leaving their row at
            # zero would hand them an all-zero distribution to draw from. RePHO warms each direction
            # up in its own process for the same reason.
            w = af / af.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            w = self._adaptive_dir_frame_weights(af)
            # [ROLLBACK MARKER: curriculum-window] x3 on the boundary frame itself (intermimic.py:1336).
            # It is the frame whose completions gate the window opening, and it is also the frame the
            # finish penalty hits first once the policy starts clearing it — without the boost the
            # counter stalls and the window never opens.
            if (self._init_range_left > 0
                    and int(self._left_completions.max()) > int(getattr(c, "left_boost_after", 100))):
                w[:, self._init_range_left] = w[:, self._init_range_left] * 3.0
            # [ROLLBACK MARKER: curriculum-window] and the post-swap seam (intermimic.py:1333): forced
            # to at least the table maximum, so the drill happens regardless of what the weights say.
            if self._tar_window is not None:
                _l, _r = self._tar_window
                if _r > _l:
                    w[:, _l:_r] = (w[:, _l:_r] * 1000.0).clamp(min=float(w.max()))
        # [ROLLBACK MARKER: dir-partition] the direction is fixed per env, so there is nothing to draw
        # for it: each env samples a frame from its OWN direction's row. This replaces the joint draw
        # over 2*F flattened cells, where a single normalisation spanned both rows so raising
        # backward's weight at a frame lowered forward's there.
        w = w.clamp(min=0)                                             # (2,F)
        # a row with no mass falls back to uniform over that direction's allowed frames; an all-zero
        # row makes multinomial return an out-of-range index and faults the cache gather below
        w = torch.where(w.sum(dim=1, keepdim=True) > 0, w, af.clamp(min=1e-12))
        fwd_n = self._dir_fwd[env_ids]                                 # (n,) FIXED, never reassigned
        d_of = (~fwd_n).long()                                         # 0 fwd / 1 bwd
        pick = torch.multinomial(w[d_of], 1).squeeze(1)                # (n,) per-env row draw
        # [/ROLLBACK MARKER: backward-dir] --------------------------------------------------------
        # FULL trajectory (so failure-weighting can concentrate anywhere); the START only has to leave
        # (0.8 s @50 Hz) before the sequence-end time-out.
        #   start in [0, 101] — frames 102..250 were never RSI starts — and a median 151-frame clip
        #   collapsed to [0, 1]. Now: [0, 210] and [0, 110] respectively.
        # adaptive_sampling=False (rollout/play) → upper=0 → every episode starts at frame 0 and plays
        # the chunk gone it has to be stated explicitly or evaluation would start at random frames.
        # RePHO intermimic.py:1401-1402 — the sampled frame IS the start, no offset.
        start = pick if c.adaptive_sampling else torch.zeros_like(pick)
        bad = ~candidates[start]
        start[bad] = 0
        # [ROLLBACK MARKER: track-harvest] the val rollout runs `--stateInit Start`, i.e. it begins
        # candidate span start somewhere else than the frame the buffer nominated.
        if self._harvest_frame[0] is not None and bool(self._harvest[env_ids].any()):
            _hm = self._harvest[env_ids]
            _hf = torch.tensor([self._harvest_frame[0], self._harvest_frame[1]],
                               device=dev, dtype=torch.long)[d_of]
            # chosen from the buffer, which is keyed in ORIGINAL clip time; `start` is EPISODE time
            _hf = torch.where(fwd_n, _hf, (self._ref_len - 1) - _hf)
            start = torch.where(_hm, _hf.clamp(0, self._ref_len - 1), start)
        # ── MASTER RSI SWITCH (cfg.use_rsi=False) ──────────────────────────────────────────────
        # Force every episode to begin at frame 0. Placed AFTER the sampling block rather than
        # around it so the block stays byte-identical for the use_rsi=True path (the wasted
        # randint/multinomial is n samples per control step — negligible). The failure-count EMA
        # The companion gate below forces the reference restore path so the reset STATE is
        # frame-0-deterministic too — a frame-0 train-cache hit would otherwise still restore a
        # cached sim state, which is RSI machinery.
        if not c.use_rsi:
            start = torch.zeros_like(start)
        self._contact_lost[env_ids] = 0
        self._frame_idx[env_ids] = start
        self._episode_start_frame[env_ids] = start                 # diagnostics (Diag / rsi_start_mean)
        # [ROLLBACK MARKER: start-hist] where in the clip do episodes actually BEGIN? The mean
        # alone cannot answer the question that matters — what share of episodes start inside the
        # GRASP window, which for s101_seg12_knife is frames 0-41 (hand reaches the knife at 29,
        # knife starts moving at 41): 8% of the clip. Everything after that is carrying an object
        # the episode was handed, so an RSI start there never practises the grasp itself.
        _s = start.float()
        self._diag_start_q = (float(torch.quantile(_s, 0.1)), float(torch.quantile(_s, 0.5)),
                              float(torch.quantile(_s, 0.9)), float((start < 41).float().mean()),
                              float((start < 100).float().mean()))
        # reset the tracking-quality streak for the reset envs (grasp mechanism)
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
        # [ROLLBACK MARKER: backward-dir] per-reset-env direction subset. `_rframe()` works on the
        # full (E,) buffers, so the reset path (which is (n,) over env_ids) needs its own subset.
        # Stage 2 leaves everything forward, so these are all-True / all-+1 and every use is identity.
        sign_n = self._dir_sign[env_ids]                             # (n,1) +1 fwd / -1 bwd
        root_pose = torch.zeros(n, 7, device=dev); root_pose[:, 3] = 1.0
        root_vel = torch.zeros(n, 6, device=dev)
        jpos = self.robot.data.default_joint_pos[env_ids].clone()
        jvel = torch.zeros_like(jpos)
        org = self.scene.env_origins[env_ids]

        # 3-way source selection per env (train cache > pretrain cache > reference+default),
        # vectorized via boolean masks + 2D advanced-index gathers (no per-env python loop).
        aid = self._action_joint_ids_t                               # (65,) action-joint columns
        # [backward-dir] the cache is keyed in ORIGINAL clip time; a backward env's episode frame
        # `start` lives at original frame T-1-start. Every cache lookup below uses `cstart`.
        cstart = self._canon_frame(start, fwd_n)                     # (n,) cache index
        dsel = (~fwd_n).long()                                       # (n,) 0 fwd / 1 bwd buffer
        # [ROLLBACK MARKER: slot-cache] draw WHICH slot each env restores from. Slot 0 holds the
        # reference row (built by _ref_cache_row, so restoring it reproduces the old where_ref reset
        # bit for bit — original-time pose, forward velocities, jvel 0, smoothed = unscale(ref)), and
        # the backward velocity flip below then applies to it exactly as it did on that path.
        slot = torch.zeros(n, dtype=torch.long, device=dev)
        if self._n_slots > 1 and self._slot_occ[:, :, 0].any():
            slot = torch.multinomial(self._slot_probs(cstart, dsel), 1).squeeze(1)
        train_hit = self._slot_usable()[dsel, cstart, slot]          # (n,) bool
        if self._pretrain_cache is not None:
            pretrain_hit = ~self._pretrain_init_flg[cstart]
        else:
            pretrain_hit = torch.zeros_like(train_hit)
        # [MASTER RSI SWITCH] use_rsi=False → never READ a state cache; send every env down the
        # where_ref path so the reset state is the frame-0 retarget reference pose, identical every
        # episode. The caches are still WRITTEN (_save_state_cache is untouched), so flipping the
        # switch back to True resumes with whatever coverage training has accumulated.
        # [ROLLBACK MARKER: rsi-phase-split] reference_only_warmup — during the uniform warm-up the
        # cache is WRITTEN but never READ, so every warm-up episode starts from the same kind of
        # state (the reference) instead of a mix that shifts as coverage fills in front-to-back.
        if not c.use_rsi or (use_uniform and getattr(c, "reference_only_warmup", False)):
            # [ROLLBACK MARKER: slot-cache] force the REFERENCE slot rather than bypassing the cache:
            # slot 0 IS the reference, so the warm-up restore is unchanged, and this keeps a single
            # restore path. With no retarget reference (slot 0 unoccupied) it degrades to where_ref.
            slot = torch.zeros_like(slot)
            train_hit = self._slot_occ[dsel, cstart, 0]
            pretrain_hit = torch.zeros_like(pretrain_hit)
        self._episode_start_slot[env_ids] = slot
        self._diag_slot_ref_frac = float((slot[train_hit] == 0).float().mean()) if bool(train_hit.any()) else 1.0
        # [ROLLBACK MARKER: slot-cache] entropy of the REALISED slot draw, in nats. This is the number
        # that decides whether the slot machinery is doing its job: every stall this session arrived
        # together with a narrowing start distribution, and it was never measured directly. 0 means one
        # slot has taken over; ln(n_slots) means a flat draw.
        if self._n_slots > 1:
            cnt = torch.bincount(slot, minlength=self._n_slots).float()
            q = cnt / cnt.sum().clamp(min=1)
            self._diag_slot_entropy = float(-(q * torch.log(q.clamp(min=1e-12))).sum())
        where_train = train_hit
        where_pre = (~train_hit) & pretrain_hit
        where_ref = (~train_hit) & (~pretrain_hit)

        if where_train.any():                                        # train cache: 222-D layout
            idx = where_train.nonzero(as_tuple=True)[0]
            # [backward-dir] cached row is in original time with FORWARD velocities; flip the
            # velocity channels back for a backward env (poses are direction-agnostic).
            s = self._flip_cache_vel(self._state_cache[dsel[idx], cstart[idx], slot[idx]], sign_n[idx])
            root_pose[idx, :3] = s[:, 1:4] + org[idx]; root_pose[idx, 3:7] = s[:, 4:8]
            root_vel[idx, :3] = s[:, 8:11]; root_vel[idx, 3:6] = s[:, 11:14]
            jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 27:92]
            jvel[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 92:157]
            # [ROLLBACK MARKER: action-seed-from-pose] RESTORED (was briefly seeded from the pose).
            # The cached commanded target is loaded again, matching TJ (gr_env.py:745 restores
            # `hand_actions` and `dof_actions` from its own cache row).
            # Known and ACCEPTED cost: with `sonic_hand_delta` False the hand runs an absolute EMA
            # whose memory IS `_smoothed_actions[hands]`, so this value enters CONTROL at 50% weight
            # on the first step (alpha_hands = 0.5); a commanded target leads the pose in the
            # direction of travel, so a row written by a FORWARD episode leads the wrong way for a
            # BACKWARD one. Measured magnitude is ~0.05 rad on a finger joint, decaying by half per
            # step — small next to what this column buys. The BODY is unaffected either way: its
            # target comes from the frozen SONIC decoder and never reads `_smoothed_actions`.
            # To seed from the pose instead: `= self._unscale(jpos[idx][:, aid])`.
            self._smoothed_actions[env_ids[idx]] = s[:, 157:222]
        if where_pre.any():                                          # pretrain cache: 209-D layout
            idx = where_pre.nonzero(as_tuple=True)[0]
            s = self._pretrain_cache[start[idx]]
            root_pose[idx, :3] = s[:, 1:4] + org[idx]; root_pose[idx, 3:7] = s[:, 4:8]
            root_vel[idx, :3] = s[:, 8:11]; root_vel[idx, 3:6] = s[:, 11:14]
            jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 14:79]
            jvel[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 79:144]
            # [ROLLBACK MARKER: action-seed-from-pose] RESTORED, same reasoning as the train branch.
            # Inert while pretrain_cache_warmstart is False.
            self._smoothed_actions[env_ids[idx]] = s[:, 144:209]
        if where_ref.any():                                          # reference root + default/retargeted joints
            idx = where_ref.nonzero(as_tuple=True)[0]
            # [backward-dir] mirrored reference index + negated velocities for backward envs
            rfr = torch.where(fwd_n[idx], start[idx], (self._ref_len - 1) - start[idx])
            sgn = sign_n[idx]                                        # (m,1) +1 / -1
            root_pose[idx, :3] = self._ref_root_pos[rfr] + org[idx]
            root_pose[idx, 3:7] = self._ref_root_quat[rfr]
            root_vel[idx, :3] = self._ref_root_linvel[rfr] * sgn
            root_vel[idx, 3:6] = self._ref_root_angvel[rfr] * sgn
            if self._ref_joints is not None:
                jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = self._ref_joints[rfr]
            self._smoothed_actions[env_ids[idx]] = self._unscale(jpos[idx][:, aid])


        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)

        # object restore if present: reference pose on pretrain/reference resets; on an RSI
        # TRAIN-cache hit restore the object FROM the cache too (blocks [14:27]) so the drifted
        # cached robot and the object stay a physically-consistent pair (a reference object paired
        # with a mid-manip cached robot could trip term_obj_pos_err/term_ft_err at start+1).
        if self._has_object:
            f0 = start
            # [backward-dir] f0 is EPISODE-time; mirror it for backward envs (fwd_n/sign_n hoisted above)
            rf0 = torch.where(fwd_n, f0, (self._ref_len - 1) - f0)   # (n,) reference index
            ref_op = self._ref_obj_pos[rf0] + org                    # (n,3) world
            ref_oq = self._ref_obj_quat[rf0]                         # (n,4)
            # [ROLLBACK MARKER: spawn-declear] lift the SPAWN out of the support. Reference-rest
            # frames only, zero elsewhere. Applied to ref_op, so it reaches both the plain reference
            # path and the else-branch of the cache select below — and nothing else: _ref_obj_pos
            # itself is untouched, so reward, observations and the reference velocities stay pure GT.
            # A cache HIT overrides this entirely, which is correct: a cached pose was actually
            # simulated, so it is already clear of the support.
            # getattr: DirectRLEnv.__init__ resets every env before the solve has run.
            _lift = getattr(self, "_obj_spawn_lift", None)
            if _lift is not None:
                ref_op = ref_op.clone()
                ref_op[:, 2] = ref_op[:, 2] + _lift[rf0]
            # [/ROLLBACK MARKER: spawn-declear]
            op = torch.zeros(n, 7, device=dev)
            op[:, :3] = ref_op; op[:, 3:7] = ref_oq
            # reference path: seed the object at its REFERENCE velocity for the sampled frame (mid-motion
            # starts place the object moving, not at rest). Cache-hit envs overwrite from the cache below.
            ovel = torch.zeros(n, 6, device=dev)
            # [backward-dir] v_bwd(f) = -v_fwd(T-1-f): mirrored index AND negated sign. This is the
            # one thing the index mapping alone cannot do, hence the explicit `sign_n`.
            ovel[:, :3] = self._ref_obj_linvel[rf0] * sign_n
            ovel[:, 3:6] = self._ref_obj_angvel[rf0] * sign_n
            if where_train.any():
                sc = self._flip_cache_vel(self._state_cache[dsel, cstart, slot], sign_n)  # [backward-dir]
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
        fr = self._frame().clamp(max=self._ref_len - 1)                     # (E,) long
        # write-gate while pretrain cache loaded (avoid poisoning with 1-2-step object interpen.)
        gate = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        if self._pretrain_cache is not None:
            gate = self.episode_length_buf >= 3

        # [ROLLBACK MARKER: repho-cache] NO per-frame quality filter. RePHO stages nothing and
        # judges the whole episode ONCE, at termination: an episode that is long enough writes its
        # entire visited span in one shot (intermimic.py:1832-1842), a short one writes only its
        # start frame (:1844). What used to be here was a per-frame bar ANDed into a monotone streak
        # rollout that stumbled at step 3 and then tracked cleanly to frame 400 contributed nothing.
        # Since surviving an unbroken streak gets less likely the further into the clip you go, the
        # cache systematically lost its late frames, and a frame with no entry is drawn with
        # probability exactly zero, so it never got one either. The episode-length rules in
        # _flush_state_cache (repho_full_traj_length + the death tail drop) are the whole filter now,
        # and the survival-length score is what sorts good states from bad.

        # cache ranking key = the ACTUAL step reward (grasp / TJ convention; see the docstring).
        r = reward                                                          # (E,)
        # [ROLLBACK MARKER: deferred-cache] -------------------------------------------------------
        # With the deferral on, `better` is NOT evaluated here — the comparison against the cache
        # happens at commit time in _flush_state_cache, because the cache moves while the episode
        # runs. Staging is gated on the per-frame quality bars only; the episode-length filter is
        # applied at termination, which is the whole point (it needs hindsight).
        # `_pend_state` is always allocated, so this is the only path out of the method. The old
        # write-through tail below it (compare against the cache, write slot 0 directly) was
        # unreachable and slot-unaware — it assigned a (222,) row into a (n_slots,222) view, which
        # would have broadcast one state across every slot at that frame.
        # ---- per-frame tracking-quality gate ----
        # Good AT THIS FRAME, not a streak. The previous version ANDed this into a monotone
        # `_enough_continued`, so the first wobble killed staging for the rest of the episode and a
        # rollout that stumbled at step 3 then tracked cleanly to frame 400 contributed nothing —
        # which starved the cache of exactly the late frames, since an unbroken streak gets less
        # likely the further into the clip you go.
        # Two object bars chosen by the SAMPLING PHASE (not a coverage latch, which could only ever
        # tighten): the warm-up seeds from the unsettled reference and needs the loose bar to fill at
        # all; the adaptive phase starts from settled cached states and can afford the tight one.
        _late = not bool(getattr(self, '_in_warmup', True))
        _op_bar = float(c.enough_obj_threshold_late if _late else c.enough_obj_threshold)
        _or_bar = float(c.enough_obj_rot_threshold_late if _late else c.enough_obj_rot_threshold)
        good = ((e['ft'] < c.enough_ft_threshold)
                & (e['obj_pos'] < _op_bar) & (e['obj_rot'] < _or_bar)
                & (e['body'] < c.cache_body_bar)
                & (e['root_pos'] < c.cache_root_pos_bar) & (e['root_rot'] < c.cache_root_rot_bar))
        self._diag_stage_frac = float(good.float().mean())

        if getattr(self, "_pend_state", None) is not None:
            slot = self.episode_length_buf.clamp(max=self._pend_cap - 1)   # (E,)
            stage_mask = gate & good                                       # (E,) frame-local quality
            if stage_mask.any():
                rows = torch.nonzero(stage_mask, as_tuple=False).squeeze(-1)
                self._pend_state[rows, slot[rows]] = self._build_cache_state(r, org)[rows]
                self._pend_frame[rows, slot[rows]] = fr[rows]
                self._pend_valid[rows, slot[rows]] = True
            # [ROLLBACK MARKER: track-harvest] the KINEMATICS staging is NOT filtered. RePHO's val
            # process marks every frame the rollout passed through (`new_hoi_data[left:right, -1] = 1`,
            # intermimic.py:2009) and leaves the quality judgement to the three conjuncts at swap
            # time. Reusing the state cache's quality streak here truncates the candidate at the
            # first wobble and filters on quality twice.
            if self._pend_kin is not None and self._last_kin is not None:
                krows = torch.nonzero(self._harvest, as_tuple=False).squeeze(-1)
                if krows.numel():
                    self._pend_kin[krows, slot[krows]] = self._last_kin[krows]
                    self._pend_kin_valid[krows, slot[krows]] = True
                    self._pend_kin_frame[krows, slot[krows]] = fr[krows]
            return
        # [/ROLLBACK MARKER: deferred-cache] ------------------------------------------------------

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

    # [ROLLBACK MARKER: cache-score-rework] --------------------------------------------------------
    def _ref_cache_row(self, frames: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """(n,222) cache rows built from the REFERENCE trajectory at ORIGINAL-time `frames`.

        Mirrors the `where_ref` restore path in _reset_idx exactly — same arrays, same env-relative
        convention, joint velocities left at ZERO, smoothed actions = unscale(reference joints) — so
        restoring one of these rows reproduces the reference reset bit for bit. Used for the warm-up
        entries, where the episode started from the reference and a mid-rollout snapshot would carry
        no information the reference does not already have. Forward time only: the warm-up is
        forward-only (see the sampling block in _reset_idx), so no mirroring or velocity flip.
        """
        n = frames.shape[0]
        row = torch.zeros(n, self._STATE_DIM, device=self.device)
        row[:, 0] = score
        row[:, 1:4] = self._ref_root_pos[frames]                    # env-relative, as _build_cache_state
        row[:, 4:8] = self._ref_root_quat[frames]
        row[:, 8:11] = self._ref_root_linvel[frames]
        row[:, 11:14] = self._ref_root_angvel[frames]
        if self._has_object:
            row[:, 14:17] = self._ref_obj_pos[frames]
            row[:, 17:21] = self._ref_obj_quat[frames]
            row[:, 21:24] = self._ref_obj_linvel[frames]
            row[:, 24:27] = self._ref_obj_angvel[frames]
            # [ROLLBACK MARKER: spawn-declear] slot 0 IS the reference spawn — _slot_occ[:, 0] is
            # pre-set for every frame, and the uniform warm-up forces slot 0, so a reference reset
            # reaches the sim through THIS row, not through the where_ref branch. The lift has to be
            # applied here or it never fires during warm-up, which is exactly when the cache is empty
            # and every episode starts from the reference. Height only; the reference arrays
            # themselves stay untouched, so reward and observations still see pure GT.
            _lift = getattr(self, "_obj_spawn_lift", None)
            if _lift is not None:
                row[:, 16] = row[:, 16] + _lift[frames]
            # [/ROLLBACK MARKER: spawn-declear]
        if self._ref_joints is not None:
            rj = self._ref_joints[frames]                           # (n,65) action-joint order
            row[:, 27:92] = rj
            row[:, 157:222] = self._unscale(rj)
        # [92:157] joint velocities stay 0 — the reference reset path never writes jvel either.
        return row

    # [ROLLBACK MARKER: dir-decouple] --------------------------------------------------------------
    def _adaptive_dir_frame_weights(self, af: torch.Tensor) -> torch.Tensor:
        """(2,F) joint weights over (direction, frame) for the adaptive phase, in EPISODE-frame space.

        RePHO (intermimic.py:1316-1346) samples a start frame by the SUMMED survival length stored at
        it, then penalises frames the policy already finishes. The raw weight alone would be backwards
        for a curriculum — it prefers whatever already works — so the finish penalty is the mechanism,
        not a refinement:
            completion = mean survival length / frames remaining from here
            completion > lo                       -> x0.5
            completion > hi and few steps left    -> x0.2
        Expressed as a completion FRACTION rather than RePHO's raw (w+14)/2, which bakes in its slot
        count and score convention.

        Direction is an even split over whichever directions are allowed at that frame. The failure
        table drove this before, weighting each direction by the OTHER's failures; that is gone with
        the fixed forward/backward share.

        Frame space: the weights are built in ORIGINAL clip time (where the cache is keyed) and
        flipped back to episode time on the way out, since row 1 is a backward rollout.
        """
        c = self.cfg
        allow = torch.stack([af[0], torch.flip(af[1], [0])])              # allow mask, original time
        anyok = (allow.sum(0) > 0).float()

        occ = self._slot_occ                                              # (D,F,S)
        L = torch.where(occ, self._state_cache[:, :, :, 0], torch.zeros_like(self._state_cache[:, :, :, 0]))
        # The floor belongs to the WEIGHT only. Measuring completion from the floored value understates
        # it by floor/rem per slot, worst exactly where rem is small (the clip end) — so the penalty
        # would stop firing right where it is needed. RePHO adds its +14 back for the same reason.
        _ls = slice(1, None) if self._n_slots > 1 else slice(None)         # LEARNED slots only
        Lm = ((L * occ)[:, :, _ls].sum(dim=2)
              / occ[:, :, _ls].sum(dim=2).clamp(min=1).float())            # (D,F) mean survival length
        w = (L - float(c.repho_sample_floor)).clamp(min=0.0).sum(dim=2)    # (D,F)
        # Frames still ahead FROM f, per direction. Forward has ref_len-f left; BACKWARD runs toward
        # frame 0, so it has f+1. Using the forward figure for both made a backward rollout near the
        # start of the clip — which has almost nothing left to do — look far from finished, so the
        # finish penalty never fired there and the sampler kept paying for frames it had solved.
        _f = torch.arange(self._ref_len, device=w.device)
        rem = torch.stack([(self._ref_len - _f), (_f + 1)]).clamp(min=1).float()   # (2,F)
        comp = Lm / rem                                                    # fraction of what was reachable
        left = rem - Lm                                                    # steps still unaccounted for
        w = torch.where((comp > c.repho_finish_hi) & (left <= c.repho_finish_left),
                        w * float(c.repho_penalty_hi), w)
        w = torch.where(comp > c.repho_finish_lo, w * float(c.repho_penalty_lo), w)
        w = w * allow
        # Each direction's frames are normalised WITHIN that direction, then the two are given an equal
        # share. Previously one normalisation spanned both rows, so raising backward's weight at a frame
        # lowered forward's there — the budgets competed. A fixed split is what the cross buffer needs:
        # backward is not there to win frames off forward, it is there to fill forward's buffer.
        rows = w.sum(dim=1, keepdim=True)
        w = torch.where(rows > 0, w / rows.clamp(min=1e-6),
                        allow / allow.sum(1, keepdim=True).clamp(min=1e-6))
        # The share is over ALLOWED directions, NOT over directions that already have buffer mass.
        # Gating it on mass deadlocks the second direction the moment it is switched on: its buffer
        # holds only the seeded slot 0, whose score sits below repho_sample_floor, so its frame
        # weights sum to zero, so it gets a zero share, so it is never drawn, so its buffer never
        # fills. Measured: with backward_ratio=0.5 the backward buffer stayed at exactly its 501
        # seeded entries over 200 steps and the backward cross buffer never received anything.
        # The uniform fallback on the line above is what carries a direction until it has a buffer.
        allowed = (allow.sum(dim=1, keepdim=True) > 0).float()
        share = torch.where(allowed.sum() > 0, allowed / allowed.sum().clamp(min=1.0),
                            torch.full_like(allowed, 0.5))
        w_orig = w * share
        return torch.stack([w_orig[0], torch.flip(w_orig[1], [0])])        # back to episode time

    # [/ROLLBACK MARKER: dir-decouple] -------------------------------------------------------------

    # [ROLLBACK MARKER: slot-cache] ----------------------------------------------------------------
    def _slot_usable(self) -> torch.Tensor:
        """(D, F, n_slots) bool — slots a reset may restore from, per direction. Plain occupancy."""
        return self._slot_occ.clone()

    def _slot_probs(self, frames: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """(n, n_slots) slot lottery at the given (direction, ORIGINAL-time frame) pairs."""
        return RB.slot_probs(self._state_cache, self._slot_usable(), dirs, frames,
                             float(self.cfg.repho_slot_floor))

    def _episode_returns(self, rows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """(R,cap) discounted return-to-go for each staged slot: G[k] = sum_{j>=k} gamma^(j-k) r[j].

        `rows` are the staged (R,cap,222) states whose column 0 is that step's ACTUAL reward.
        Invalid slots contribute 0, and because a live episode occupies a CONTIGUOUS slot range the
        unconditional matmul is exact for every valid slot.
        """
        rew = rows[:, :, 0] * valid.to(rows.dtype)                  # (R,cap), 0 outside the episode
        return rew @ self._score_gamma_mat                          # (R,cap)
    # [/ROLLBACK MARKER: cache-score-rework] -------------------------------------------------------

    # [ROLLBACK MARKER: deferred-cache] -----------------------------------------------------------
    def _update_cross_buffer(self, dirs, frames, score, state) -> None:
        """Stage candidates into the WRITING direction's cross buffer (SupMat Alg 2, lines 33-38)."""
        RB.cross_write(self._cross_cache, self._cross_len, self._cross_occ, dirs, frames, score, state)

    def _track_promote(self, env_ids, ep_len) -> None:
        """Keep the best candidate TRAJECTORY per direction, RePHO's val-rollout semantics.

        The span is what the rollout passed through, read off the UNFILTERED kinematics staging —
        RePHO marks every such frame (new_hoi_data[left:right, -1] = 1, intermimic.py:2009) and
        leaves quality to the three conjuncts at swap time.

        Two things the frames must survive on the way in:
          - they are staged in EPISODE time while the reference arrays are in ORIGINAL clip time, so
            a BACKWARD candidate has to be mirrored (episode f == original T-1-f) or the swap writes
            a completely different part of the clip.
          - RePHO trims a DIED rollout by 10 frames (right = max(progress-10, left+1),
            intermimic.py:1986): the run-up to a failure is not what the reference should become.
        """
        c = self.cfg
        if not bool(self._harvest.any()):
            return
        hm = self._harvest[env_ids]
        tail = int(getattr(c, "track_death_tail", 10))
        for i in hm.nonzero(as_tuple=True)[0].tolist():
            e = int(env_ids[i])
            v = self._pend_kin_valid[e]
            if not bool(v.any()):
                continue
            fwd = bool(self._dir_fwd[e])
            d = 0 if fwd else 1
            sl = v.nonzero(as_tuple=True)[0]
            fr = self._pend_kin_frame[e, sl]                          # EPISODE frames
            died = bool(self._died[e]) if hasattr(self, "_died") else False
            if died and tail > 0:                                     # drop the run-up to the failure
                keep = fr <= (int(fr.max()) - tail)
                if not bool(keep.any()):
                    continue
                sl, fr = sl[keep], fr[keep]
            ofr = fr if fwd else (self._ref_len - 1) - fr             # -> ORIGINAL clip time
            lo, hi = int(ofr.min()), int(ofr.max())
            ln = hi - lo + 1
            if ln <= self._track_span[d]:
                continue                                              # not better than what we hold
            need = self._ref_link_contact_mask[lo:hi + 1].any(dim=-1).float().mean()
            if float(need) < float(getattr(c, "track_ref_contact_frac", 0.5)):
                continue
            seg = torch.zeros(self._ref_len, dtype=torch.bool, device=self.device)
            seg[lo:hi + 1] = True
            self._track_kin[d] = 0.0
            self._track_kin[d, ofr] = self._pend_kin[e, sl]
            self._track_occ[d] = seg
            self._track_span[d] = ln
            self._track_fwd[d] = fwd                                  # which way the curve runs
        self._diag_track_promote = float(max(self._track_span))

    def _apply_tracking_update(self) -> None:
        """Swap the reference where a candidate clearly beats the buffer (RePHO load_run_val).

        Two paths, and RePHO does NOT treat them the same (intermimic.py:1104-1128 / 1145-1176):

          SELF   candidate = this direction's own rollout
                 vs MAX over learned slots, margin 30, floor 60
                 swap [mask_left, mask_right]           and NO seam drill
          CROSS  candidate = the OPPOSITE direction's rollout
                 vs MIN over learned slots, margin 60, floor 90
                 swap [mask_left, mask_right+10], accepts a short segment at the clip start,
                 and this is the only path that arms the seam drill + decay freeze

        Both add the 3/2 relative-improvement conjunct. The margins and floors are doubled-ish for
        cross while its comparison target is the WORST slot rather than the best — the net bar
        depends on how far apart this direction's slots are, which is the code's way of being more
        open to outside evidence exactly where it is least sure of itself.
        """
        c = self.cfg
        if self._track_occ is None:
            return
        F = self._ref_len
        _L = torch.where(self._slot_occ[0, :, 1:], self._state_cache[0, :, 1:, 0],
                         torch.full_like(self._state_cache[0, :, 1:, 0], -float("inf")))
        ratio = float(getattr(c, "track_ratio", 1.5))
        applied = 0
        self._tar_window = None
        for src, is_cross in ((0, False), (1, True)):
            if not bool(self._track_occ[src].any()):
                continue
            cur = (_L.min(dim=1).values if is_cross else _L.max(dim=1).values).clamp(min=0.0)
            marg = float(getattr(c, "track_margin_cross" if is_cross else "track_margin_self",
                                 60.0 if is_cross else 30.0))
            floor = float(getattr(c, "track_floor_cross" if is_cross else "track_floor_self",
                                  90.0 if is_cross else 60.0))
            span = self._track_occ[src].nonzero(as_tuple=True)[0]
            lo, hi = int(span[0]), int(span[-1])
            # value_candidate[f] = how much further the rollout got FROM f. A forward rollout over
            # original [lo,hi] got hi-f more; a BACKWARD one travelled hi -> lo, so from f it got
            # f-lo more. The forward curve for both makes a backward candidate look best exactly
            # where it was weakest.
            _ar = torch.arange(hi - lo + 1, device=self.device).float()
            cand = torch.zeros(F, device=self.device)
            cand[lo:hi + 1] = _ar.flip(0) if self._track_fwd[src] else _ar
            mask = (cand > cur + marg) & (cand > cur * ratio) & (cand > floor)
            m_lo, m_len = self._longest_run(mask)
            m_min = int(getattr(c, "tar_min_segment", 30))
            ok = (m_len > m_min) or (is_cross and m_len >= 5 and m_lo <= 5)
            if not ok:
                continue
            m_hi = m_lo + m_len - 1
            # cross reaches 10 frames further, so the new kinematics carry past the seam
            end = min(m_hi + 11, F) if is_cross else m_hi + 1
            f = torch.arange(m_lo, end, device=self.device)
            if self._ref_orig is None:
                self._ref_orig = {k: getattr(self, k).clone() for k in
                                  ("_ref_kpts", "_ref_ft_pad", "_ref_root_pos", "_ref_root_quat",
                                   "_ref_obj_pos", "_ref_obj_quat")}
            kin = self._track_kin[src, f]
            nk = self._ref_kpts.shape[1]
            i = 0
            self._ref_kpts[f] = kin[:, i:i + nk * 3].view(-1, nk, 3);      i += nk * 3
            self._ref_ft_pad[f] = kin[:, i:i + 30].view(-1, 10, 3);        i += 30
            self._ref_root_pos[f] = kin[:, i:i + 3];                       i += 3
            self._ref_root_quat[f] = kin[:, i:i + 4];                      i += 4
            if self._has_object:
                self._ref_obj_pos[f] = kin[:, i:i + 3];                    i += 3
                self._ref_obj_quat[f] = kin[:, i:i + 4]
            applied += int(f.numel())
            if is_cross:                                                   # intermimic.py:1168-1172
                self._tar_window = (max(0, m_hi - 15), max(1, m_hi - 5))
            print(f"[track] {'cross' if is_cross else 'self'} swap on frames "
                  f"{m_lo}..{int(f[-1])} ({m_len} matched)")
        self._diag_track_applied = float(applied)
        self._diag_tar_active = float(self._tar_window is not None)

    def _return_ratio(self) -> float:
        """The bar the incumbent's return has to clear, as a multiple of itself (intermimic.py:1868).

        RePHO anneals this to 0 early in its run, leaving survival length alone in charge. We hold it
        at 1.0 instead: RePHO's early termination ends an episode once required body-object contact is
        lost for 10 consecutive frames, so surviving long there REQUIRES doing the task and length is
        a fair proxy for quality. Our terminations are tracking-error only, with bars 2-3.6x above the
        errors actually reached, so a policy that lets the object drift still survives — length and
        quality come apart, and this term is the only thing holding them together.
        Set repho_return_ratio_steps > 0 to restore the annealing."""
        c = self.cfg
        r0 = float(getattr(c, "repho_return_ratio_start", 0.25))
        n = int(getattr(c, "repho_return_ratio_steps", 0))
        if n <= 0:
            return r0
        return max(r0 * (1.0 - self._sampling_step_count / float(n)), 0.0)

    @staticmethod
    def _longest_run(mask: torch.Tensor) -> tuple[int, int]:
        """(start, length) of the longest contiguous True run in a 1-D bool mask; (0, 0) if none."""
        idx = mask.nonzero(as_tuple=True)[0]
        if not idx.numel():
            return 0, 0
        brk = (idx[1:] - idx[:-1] > 1).nonzero(as_tuple=True)[0]
        starts = torch.cat([idx[:1], idx[brk + 1]])
        ends = torch.cat([idx[brk], idx[-1:]])
        lens = ends - starts + 1
        b = int(lens.argmax())
        return int(starts[b]), int(lens[b])

    def _cross_transfer(self) -> None:
        """Inter-direction update (SupMat Alg 3, lines 41-52): each direction imports the OTHER's best
        cross entry into its reserved (last) slot. Inert until backward rollouts actually run — with
        `backward_ratio == 0` the backward cross buffer is never written, so the forward import finds
        no source and the backward buffer has no reader."""
        c = self.cfg
        if self._n_slots < 2:
            return
        self._diag_cross_import = float(RB.cross_import(
            self._state_cache, self._slot_occ, self._cache_return,
            self._cross_cache, self._cross_len, self._cross_occ,
            reserved=self._n_slots - 1,
            margin=float(getattr(c, "cross_margin", 10.0)),
            floor=float(getattr(c, "cross_abs_floor", 40.0)),
            ratio=float(getattr(c, "cross_rel_ratio", 1.25)),
            penalty=float(getattr(c, "cross_penalty", 10.0))))

    def _flush_state_cache(self, env_ids: torch.Tensor, ep_len: torch.Tensor) -> None:
        """Commit the staged states of TERMINATING envs, in bulk.

        Called from _reset_idx BEFORE `super()._reset_idx()` zeroes `episode_length_buf`, so `ep_len`
        is captured by the caller. Deferral exists so the score can be hindsight: a state is worth
        keeping based on what happened AFTER it, which is only known once the episode ends.

        Score (column 0) is the SURVIVAL LENGTH `end - t` in episode time, RePHO 1832/1837. The
        discounted return is demoted to the tiebreak in `_cache_return`.
        """
        if self._pend_state is None or len(env_ids) == 0:
            return
        c = self.cfg
        warm = bool(getattr(self, "_in_warmup", True))
        cap = self._pend_valid.shape[1]
        sl = torch.arange(cap, device=self.device).unsqueeze(0)              # (1,cap)

        # Which staged frames survive into the cache. RePHO writes the whole visited range for a LONG
        # episode (minus a tail if it died) and only the START frame for a short one, valued at the
        # length it reached (1832-1844). A short episode still being recorded is the point: the hardest
        # frames in the clip need to be visible to the sampler scored LOW, not missing.
        full = int(getattr(c, "repho_full_traj_length", 40))
        # SupMat Alg 2 line 23: `if not (AdaptiveMode and t_end - t_start > L_valid_2) then U <- {t_start}`.
        # The WARM-UP contributes only a start frame no matter how long the episode ran — the released
        # code gates on length + contact alone and drops the mode term, but the paper is explicit and
        # the mode is what we follow here. Note `>` L_valid_2 for the full trajectory, so the short
        # test is `<=`, not `<`.
        short = warm | (ep_len <= full)
        # A rollout that reached the END of its direction's clip. Episode time counts up from the
        # start for BOTH directions, so `start + ep_len >= ref_len - 1` is the clip end either way
        # (for a backward env that is original frame 0).
        _reached = (self._episode_start_frame[env_ids].float() + ep_len.float()) >= (self._ref_len - 1)
        # RePHO's to_end_cnt (intermimic.py:1782) counts only completions that ALSO started well
        # before the end — `start_index < max_episode_length - 50`. Without that span condition an
        # episode that begins two frames from the end and times out immediately counts as a
        # completion, which at our env count would clear any threshold within a few steps.
        _span = int(getattr(c, "repho_completion_min_span", 50))
        _d_of = (~self._dir_fwd[env_ids]).long()
        _banked = _reached & (self._episode_start_frame[env_ids] < (self._ref_len - _span))
        for _d in (0, 1):
            self._n_completions[_d] += int((_banked & (_d_of == _d)).sum())
        # [ROLLBACK MARKER: curriculum-window] left_to_end_cnt (intermimic.py:1784) counts only the
        # completions that started AT the boundary — that is what makes it a test of the boundary
        # rather than of the clip as a whole.
        if self._init_range_left > 0:
            _from_left = _banked & (self._episode_start_frame[env_ids] <= self._init_range_left)
            for _d in (0, 1):
                self._left_completions[_d] += int((_from_left & (_d_of == _d)).sum())
            if int(self._left_completions.min()) > int(getattr(c, "left_open_after", 200)):
                print(f"[curriculum] window opens: _init_range_left {self._init_range_left} -> 0 "
                      f"after {self._left_completions.tolist()} boundary completions")
                self._init_range_left = 0

        _tail = int(getattr(c, "repho_drop_tail_on_death", 0))
        if _tail > 0:
            # RePHO keeps the whole visited range ONLY when the rollout reached the clip end AND that
            # direction has banked more than to_end_cnt completions (intermimic.py:1833); everything
            # else loses its last `_tail` frames. Early on it does not trust a completion either.
            # SHORT episodes are exempt: they go down the start-frame fallback below, and cutting them
            # erases them entirely whenever ep_len <= _tail (cut collapses to 0, every staged slot
            # drops, and the fallback then has nothing left to keep).
            _trusted = self._n_completions[_d_of] > int(getattr(c, "repho_trust_completion_after", 50))
            _cut_mask = ~(_reached & _trusted) & ~short
            if bool(_cut_mask.any()):
                cut = (ep_len.long() - _tail).clamp(min=0).unsqueeze(-1)     # (E,1)
                self._pend_valid[env_ids] &= ~(_cut_mask.unsqueeze(-1) & (sl >= cut))
        if bool(short.any()):
            v = self._pend_valid[env_ids]
            # NOT the RSI start frame: _save_state_cache runs after the physics step and
            # _pre_physics_step advances the frame first, so the earliest staged frame is start+1.
            first = torch.where(v.any(dim=1), v.float().argmax(dim=1), torch.zeros_like(ep_len))
            self._pend_valid[env_ids] &= ~(short.unsqueeze(-1) & (sl != first.unsqueeze(-1)))

        # NOT inside the state-cache staging block below: harvest envs are deliberately excluded
        # from `valid` so they cannot pollute the buffers, and leaving the promote call in there made
        # a flush where only harvest envs terminated skip the harvest entirely.
        if self._track_kin is not None and self._pend_kin is not None:
            self._track_promote(env_ids, ep_len)
        n_written = 0
        valid = self._pend_valid[env_ids].clone()                                    # (E, cap)
        if valid.any():
            pend = self._pend_state[env_ids]                                 # (E,cap,222)
            if self._score_gamma_mat is not None:
                ret = self._episode_returns(pend, valid).clamp(min=0.0)      # (E,cap)
            else:
                ret = pend[:, :, 0].clamp(min=0.0)                           # fallback: step reward
            if bool(self._harvest.any()):
                valid = valid & (~self._harvest[env_ids]).unsqueeze(-1)
            sel = torch.nonzero(valid, as_tuple=False)                       # (K,2) [row, slot]
            env_of = env_ids[sel[:, 0]]
            cand_ret = ret[sel[:, 0], sel[:, 1]]                             # (K,) tiebreak
            cand_state = self._pend_state[env_of, sel[:, 1]]                 # (K,222)
            cand_frame = self._pend_frame[env_of, sel[:, 1]]                 # (K,) EPISODE frame
            # _pend_frame counts up FROM the start frame, so the end index is start + ep_len, not
            # ep_len. _episode_start_frame still holds the OLD episode's start (this runs before
            # _reset_idx draws the new one).
            ep_end = self._episode_start_frame[env_of].float() + ep_len[sel[:, 0]].float()
            cand_score = (ep_end - cand_frame.float()).clamp(min=0.0)

            # [backward-dir] into the shared ORIGINAL-time cache: mirror the frame index and negate
            # the velocity channels for backward envs. This is what lets both directions fill ONE
            # cache, and why a backward episode covers early frames a forward one only ever starts at.
            # [ROLLBACK MARKER: cross-buffer] the same states scored for the opposite direction:
            # distance travelled FROM the start rather than distance still to go. Computed before the
            # frame is canonicalised, since both terms are in episode time.
            cross_score = (cand_frame.float()
                           - self._episode_start_frame[env_of].float()).clamp(min=0.0)

            fwd_of = self._dir_fwd[env_of]
            cand_frame = self._canon_frame(cand_frame, fwd_of)
            cand_state = self._flip_cache_vel(cand_state, self._dir_sign[env_of])
            cross_state = cand_state.clone()
            cross_state[:, 0] = cross_score
            cand_state[:, 0] = cand_score
            if getattr(c, "cross_buffer", False):
                # RePHO gates the cross write on `end - start > 60` (intermimic.py:1836/1841): a short
                # rollout barely leaves its start frame, so "this is reachable from the start" is
                # nearly vacuous as evidence for the opposite direction. Episodes below the bar still
                # fill the SELF buffer; they just do not vouch for the other direction.
                _cmin = float(getattr(c, "cross_min_episode_length", 60))
                _long = (ep_len[sel[:, 0]].float() > _cmin)
                if bool(_long.any()):
                    self._update_cross_buffer((~fwd_of[_long]).long(), cand_frame[_long],
                                              cross_score[_long], cross_state[_long])
            dirs_of = (~fwd_of).long()                                   # 0 fwd / 1 bwd buffer
            n_written = RB.self_write(self._state_cache, self._slot_occ, self._cache_return,
                                      dirs_of, cand_frame, cand_score, cand_ret, cand_state,
                                      float(getattr(c, "repho_replace_margin", 10)),
                                      ret_ratio=self._return_ratio())

        # RePHO decays the two scores at DIFFERENT rates and exempts slot 0 from the length decay
        # (intermimic.py:1892-1899): the length is a property of the trajectory and ages slowly, the
        # return is tied to the policy that produced it and is stale ~100x faster. Applied AFTER this
        # flush so a fresh entry is not aged on the step it lands; empty entries sit at -inf and the
        # `where(occ, ...)` shields them.
        dl = float(getattr(c, "repho_decay_length", 0.0))
        if self._tar_window is not None:                                 # intermimic.py:1897-1899
            dl = float(getattr(c, "repho_decay_length_frozen", 5e-8))
        if dl > 0.0:
            lo = 1 if self._n_slots > 1 else 0
            self._state_cache[:, :, lo:, 0] = torch.where(
                self._slot_occ[:, :, lo:], self._state_cache[:, :, lo:, 0] * (1.0 - dl),
                self._state_cache[:, :, lo:, 0])
        if dl > 0.0 and self._cross_len is not None:
            # RePHO decays ref_reward_for_opposite alongside ref_reward (intermimic.py:1893-1896).
            # Without it a high cross entry is permanent: the rule is a bare L_new > L_min, so nothing
            # else can ever lower it and the slot is held for the rest of the run.
            self._cross_len = torch.where(self._cross_occ, self._cross_len * (1.0 - dl), self._cross_len)
        dr = float(getattr(c, "repho_decay_return", 0.0))
        if dr > 0.0:
            self._cache_return *= (1.0 - dr)
        self._diag_cache_overwrite = float(n_written)
        # clear staging for ALL terminating envs so the next episode starts clean
        self._pend_valid[env_ids] = False
        if self._pend_kin_valid is not None:
            self._pend_kin_valid[env_ids] = False
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
