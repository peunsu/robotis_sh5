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

from .g1_shadow_sonic_residual_env_cfg import (
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
        if os.path.exists(rt):
            rd = np.load(rt, allow_pickle=True)
            if "g1_joint_pos" in rd.files:
                self._np_ref_joints = rd["g1_joint_pos"].astype(np.float32)           # (F,65)
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

        # ---- state cache + RSI (train 222) ----
        # layout: reward(1) + root[pos3+quat4+linvel3+angvel3=13] + obj[pos3+quat4+linvel3+angvel3=13]
        #         + jpos(65) + jvel(65) + smoothed(65) = 1+13+13+195 = 222.
        #   [0]=reward [1:4]rootpos [4:8]rootquat [8:11]rootlinvel [11:14]rootangvel
        #   [14:17]objpos [17:21]objquat [21:24]objlinvel [24:27]objangvel
        #   [27:92]jpos [92:157]jvel [157:222]smoothed
        self._STATE_DIM = 222
        self._state_cache = torch.zeros(self._ref_len, self._STATE_DIM, device=dev)
        self._state_cache[:, 0] = -float("inf")                        # reward column
        self._init_flg = torch.ones(self._ref_len, device=dev, dtype=torch.bool)   # True = reference (no cache)
        self._reached_frame = 0
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
        self._failure_count = torch.zeros(self._ref_len, device=dev)
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
        idx = (self._frame_idx.unsqueeze(1) + torch.arange(K, device=dev).unsqueeze(0)).clamp(
            max=self._ref_len - 1)                                      # (E,10) future frames
        tok = torch.zeros(E, self._sonic_tok_dim, device=dev)
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
        latent = self._SP.encode_latent(self._sonic, tok)
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
        tighter 'late' object thresholds (matches grasp)."""
        return self._reached_frame >= self._ref_len - 3

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
                for k in ("ang", "jpr", "jvr", "grav"):
                    self._sonic_hist[k][m] = rows[k][m].unsqueeze(1)
                self._sonic_hist["act"][m] = 0.0
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
        fr = self._frame()
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
        return dict(body=body_err, body_core=body_core_err, ee=ee_err, com_support=self._com_support_err(), wrist_pos=wrist_pos_err, hand=hand_err, ft=ft_err, ft_reward=ft_reward, ft_per=ft_per, tip=tip,
                    pad_inward=pad_inward, root_pos=root_pos_err, root_rot=root_rot_err,
                    obj_pos=obj_pos_err, obj_rot=obj_rot_err, root_quat=root_quat,
                    wrist_rot=wrist_rot_err)

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        e = self._errs                                    # set by _get_dones (runs first each step)
        fr = self._frame()

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
        reg_hands = (self._smoothed_actions[:, hsl] ** 2).sum(-1)
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
            "Curriculum / reached_frame": float(self._reached_frame),
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
            log["Diag / tracking_penalty"] = tracking_penalty.clamp(min=-self.cfg.rew_alive).mean()
            log["Diag / tracking_penalty_raw"] = tracking_penalty.mean()
            log["Diag / clamp_frac"] = (tracking_penalty < -self.cfg.rew_alive).float().mean()
        # RSI / episode-length diagnostics. Episodes are variable length now (start frame → sequence
        # end), so skrl's "Reward / Total reward" (episode SUM) is length-confounded and no longer
        # comparable across configs — these two make the length distribution itself observable.
        # episode_len = steps elapsed in the CURRENT episode; rsi_start = frame the episode began at.
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

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids) -> None:
        # [ROLLBACK MARKER: deferred-cache] MUST run before super(), which zeroes episode_length_buf —
        # the whole point of the deferral is to filter on how long the episode actually lasted.
        # getattr, not a bare attribute: DirectRLEnv.__init__ resets every env, and that happens
        # BEFORE _post_init_buffers allocates the staging tensors.
        if getattr(self, "_pend_state", None) is not None:
            self._flush_state_cache(env_ids, self.episode_length_buf[env_ids].clone())
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
            if bool(term.any()):
                fail_frames = self._enough_idx[env_ids][term].clamp(0, self._ref_len - 1)
                counts = torch.bincount(fail_frames, minlength=self._ref_len).float()
                self._failure_count = c.adaptive_alpha * counts + (1.0 - c.adaptive_alpha) * self._failure_count

        # ---- adaptive frame sampling ----
        self._sampling_step_count += 1
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
            pick = cand_idx[torch.multinomial(probs, n, replacement=True)]
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
        start = (pick - self._adaptive_back_frames).clamp(min=0).clamp(max=upper)
        # safeguard: start must be covered by a cache; snap uncovered → 0 (in [0,upper]; covered via
        # frame-0 init-save, else the restore falls back to reference+default pose).
        bad = ~candidates[start]
        start[bad] = 0
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
        self._frame_idx[env_ids] = start
        self._episode_start_frame[env_ids] = start                 # diagnostics (Diag / rsi_start_mean)
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
        train_hit = ~self._init_flg[start]                           # (n,) bool
        if self._pretrain_cache is not None:
            pretrain_hit = ~self._pretrain_init_flg[start]
        else:
            pretrain_hit = torch.zeros_like(train_hit)
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
            s = self._state_cache[start[idx]]
            root_pose[idx, :3] = s[:, 1:4] + org[idx]; root_pose[idx, 3:7] = s[:, 4:8]
            root_vel[idx, :3] = s[:, 8:11]; root_vel[idx, 3:6] = s[:, 11:14]
            jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 27:92]
            jvel[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 92:157]
            self._smoothed_actions[env_ids[idx]] = s[:, 157:222]
        if where_pre.any():                                          # pretrain cache: 209-D layout
            idx = where_pre.nonzero(as_tuple=True)[0]
            s = self._pretrain_cache[start[idx]]
            root_pose[idx, :3] = s[:, 1:4] + org[idx]; root_pose[idx, 3:7] = s[:, 4:8]
            root_vel[idx, :3] = s[:, 8:11]; root_vel[idx, 3:6] = s[:, 11:14]
            jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 14:79]
            jvel[idx.unsqueeze(1), aid.unsqueeze(0)] = s[:, 79:144]
            self._smoothed_actions[env_ids[idx]] = s[:, 144:209]
        if where_ref.any():                                          # reference root + default/retargeted joints
            idx = where_ref.nonzero(as_tuple=True)[0]
            fr = start[idx]
            root_pose[idx, :3] = self._ref_root_pos[fr] + org[idx]
            root_pose[idx, 3:7] = self._ref_root_quat[fr]
            root_vel[idx, :3] = self._ref_root_linvel[fr]
            root_vel[idx, 3:6] = self._ref_root_angvel[fr]
            if self._ref_joints is not None:
                jpos[idx.unsqueeze(1), aid.unsqueeze(0)] = self._ref_joints[fr]
            self._smoothed_actions[env_ids[idx]] = self._unscale(jpos[idx][:, aid])

        self._last_pretrain_fallback_ratio = float(where_pre.sum().item()) / max(1, n)

        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)

        # object restore if present: reference pose on pretrain/reference resets; on an RSI
        # TRAIN-cache hit restore the object FROM the cache too (blocks [14:27]) so the drifted
        # cached robot and the object stay a physically-consistent pair (a reference object paired
        # with a mid-manip cached robot could trip term_obj_pos_err/term_ft_err at start+1).
        if self._has_object:
            f0 = start
            ref_op = self._ref_obj_pos[f0] + org                     # (n,3) world
            ref_oq = self._ref_obj_quat[f0]                          # (n,4)
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
                self._pend_state[rows, slot[rows]] = self._build_cache_state(r, org)[rows]
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
                    best = cand_state[m][cand_r[m].argmax()]
                    if best[0] > self._state_cache[uf, 0]:
                        self._state_cache[uf] = best
                        self._init_flg[uf] = False
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
