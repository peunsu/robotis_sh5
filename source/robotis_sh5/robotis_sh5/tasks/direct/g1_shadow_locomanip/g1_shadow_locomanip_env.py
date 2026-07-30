"""G1 + Shadow-hand full-body loco-manipulation environment (DirectRLEnv).

Ports the core mechanisms of `robotis_shadow_grasp_rsi` (per-group EMA + optional delta
action, contact-conditioned fingertip force with grounded normal, adaptive frame-sampling
curriculum, pretrain-cache RSI warm-start, deviation-from-reference termination, state cache)
from fixed-base single-hand to FLOATING-base bimanual FULL BODY.

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
    left-hand pad normals/offsets are already Y-mirrored in the cfg (do not re-mirror).
  * per-group EMA α = NEW-action weight; _smoothed_actions seeded at the normalized default.
  * delta groups store the NORMALIZED INTEGRATED TARGET in prev_action (not the raw delta).
  * termination is DEVIATION-FROM-REFERENCE (so reference crouch/bend never trips a fall).
  * state cache write-gate (episode_length>=3) while the pretrain cache is loaded.
"""

from __future__ import annotations

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

from .g1_shadow_locomanip_env_cfg import (
    BODY_KPT_OFFSETS,
    BODY_KPTS,
    FINGERTIP_OFFSETS,
    FINGERTIP_PAD_NORMALS,
    HAND_CHAIN,
    JOINT_GROUPS,
    G1ShadowLocomanipEnvCfg,
)

# Static Shadow palm-body → MANO landmark-frame quat (wxyz), reused from robotis_shadow_grasp.
_PALM_TO_LANDMARK = (-0.706025, 0.039103, -0.039103, 0.706025)


def _quat_to_6d(q: torch.Tensor) -> torch.Tensor:
    """6D continuous rotation rep (Zhou et al.): first two columns of R. q:(...,4) wxyz."""
    m = math_utils.matrix_from_quat(q)                       # (...,3,3)
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)   # (...,6)


def _canon(q: torch.Tensor) -> torch.Tensor:
    """Canonicalize quat to the w>=0 hemisphere (kills double-cover discontinuity)."""
    return torch.where(q[..., :1] < 0, -q, q)


class G1ShadowLocomanipEnv(DirectRLEnv):
    cfg: G1ShadowLocomanipEnvCfg

    # ------------------------------------------------------------------ init
    def __init__(self, cfg: G1ShadowLocomanipEnvCfg, render_mode: str | None = None, **kwargs):
        self._load_reference_trajectories(cfg)          # numpy buffers (pre-super: no device yet) → sets _ref_len
        self._build_object_cfg(cfg)                     # guarded: only if converted USD exists
        # Episode length follows grasp/TJ: chunk the trajectory to _num_frame_chunk frames so every
        # episode runs exactly that many steps and (with the [0, ref_len-chunk] start clamp in
        # _reset_idx) ends AT the reference end — never overshooting/freezing on the last frame.
        # Rollout / no adaptive sampling → full trajectory. Must set cfg.episode_length_s BEFORE super().
        _action_fps = round(1.0 / (cfg.sim.dt * cfg.decimation))
        _chunk = round(cfg.episode_length_s * _action_fps)
        self._num_frame_chunk = min(_chunk, self._ref_len) if cfg.adaptive_sampling else self._ref_len
        cfg.episode_length_s = self._num_frame_chunk / _action_fps
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
        self._np_ref_kpts = jp[:, ref_idx, :]                  # (F,56,3)

        # root SE(3): default from body_global_transform (human pelvis); OVERRIDDEN below by the
        # retargeting's ADJUSTED root (g1_root_pose) if present — the robot stands a little
        # closer/leans so its shorter arms reach the (unchanged) hand/object keypoints.
        fps = 1.0 / cfg.ref_dt
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

        # foot reference keypoints for the planted heuristic: use the BALLFOOT rows (ParaHome
        # 18=R, 22=L), NOT the ankle (17/21). The ballfoot is the actual ground-contact point
        # (z≈0–0.06m when planted); the ankle joint sits 7–16cm up so it NEVER falls below the
        # 0.06m planted threshold → the ankle version reported "never planted" and penalized a
        # correctly-standing robot every step. Locate by position in ref_idx.
        # ORDER MUST be [LEFT, RIGHT] to match cfg.foot_body_names=[left_ankle_roll_link,
        # right_ankle_roll_link] (which drives _foot_body_ids / _foot_contact_flags / foot_vel).
        # ParaHome ballfoot idx: 22=jLeftBallFoot, 18=jRightBallFoot. Previously ordered [18,22]
        # (=[right,left]) → _ref_foot_contact was [R,L] while the robot flags are [L,R], so
        # foot_match / foot_slip multiplied each ref foot against the WRONG robot foot (incentives
        # inverted per-foot during single support) AND the obs paired them mismatched. Fixed to [L,R].
        self._foot_ref_slots = [ref_idx.index(22) if 22 in ref_idx else -1,
                                ref_idx.index(18) if 18 in ref_idx else -1]
        # per-foot keypoint slots into the 56-kpt array (order [LEFT, RIGHT] to match foot_body_names),
        # each = [ankle, ballfoot] → the 2 foot kpts the env already tracks. Used by the foot-contact
        # reward's reference-proximity gate (foot_gate_mode="kpt"): a planted foot is rewarded only when
        # its 2 kpts are, on average, within cfg.foot_kpt_gate of their reference — REPLACING the frozen
        # contact-onset anchor, so the foot may follow the (moving/pivoting) reference during a turn.
        self._foot_kpt_slots = [[ref_idx.index(21), ref_idx.index(22)],   # LEFT  [ankle 21, ball 22]
                                [ref_idx.index(17), ref_idx.index(18)]]   # RIGHT [ankle 17, ball 18]

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

        # future_contact + CONTACT-MAP (matches robotis_shadow_grasp, computed here at load): for each
        # frame/fingertip find the NEAREST vertex on the active object's surface MESH (object-local) and
        # the fingertip→vertex force-projection DIRECTION. Consumed by (a) the contact-force reward
        # (_get_rewards projects force_matrix_w onto quat_apply(obj_q, _ref_contact_normal_local)), (b)
        # the contact-position gate (vs _ref_contact_vertex_local), (c) the delta_ft_obj obs (targets the
        # vertex on expected contact). future_contact = (object linvel>thresh OR angvel>thresh) AND
        # fingertip near the surface — relative nearest-vertex gate with a mesh, else object-centre
        # distance. NOTE the velocity is the OBJECT's (manipulation), NOT the fingertip's.
        self._np_future_contact = np.zeros((F, 10), np.float32)
        self._np_contact_vertex = np.zeros((F, 10, 3), np.float32)   # object-local nearest vertex
        self._np_contact_normal = np.zeros((F, 10, 3), np.float32)   # object-local fingertip→vertex dir
        self._has_contact_map = False
        if self._obj_name:
            op = self._np_obj_base[:, :3]                            # (F,3)
            oq = self._np_obj_base[:, 3:7]                           # (F,4) wxyz
            lv = np.zeros_like(op); lv[:-1] = (op[1:] - op[:-1]) * fps
            obj_speed = np.linalg.norm(lv, axis=-1)                  # (F,)
            dotq = np.abs((oq[:-1] * oq[1:]).sum(-1)).clip(0.0, 1.0)
            angspeed = np.zeros(F, np.float32); angspeed[:-1] = 2.0 * np.arccos(dotq) * fps
            vel_cond = (obj_speed > cfg.obj_contact_linvel_thresh) | (angspeed > cfg.obj_contact_angvel_thresh)
            mesh = self._load_object_mesh(cfg)                       # trimesh (object-local) or None
            if mesh is not None:
                verts = np.asarray(mesh.vertices, np.float32)        # (V,3) object-local
                vnorm = np.asarray(mesh.vertex_normals, np.float32)  # (V,3) mesh surface normals
                cv = np.zeros((F, 10, 3), np.float32); cn = np.zeros((F, 10, 3), np.float32)
                cvd = np.zeros((F, 10), np.float32)
                for t in range(F):
                    w, x, y, z = (float(v) for v in oq[t])
                    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], np.float32)
                    ftl = (self._np_ft_pad[t] - op[t]) @ R           # (10,3) fingertip pads in object-local
                    dd = np.linalg.norm(verts[:, None, :] - ftl[None, :, :], axis=-1)  # (V,10)
                    idx = np.argmin(dd, axis=0)                      # (10,) nearest vertex per fingertip
                    cv[t] = verts[idx]; cvd[t] = dd[idx, np.arange(10)]
                    if cfg.use_fingertip_to_vertex_dir:
                        # (fingertip − nearest vertex) = surface→finger ≈ outward normal, auto-signed to
                        # the finger side; degenerate (finger on the vertex) → mesh vertex normal.
                        dv = ftl - cv[t]                             # (10,3)
                        dn = np.linalg.norm(dv, axis=-1, keepdims=True)
                        cn[t] = np.where(dn > 5e-3, dv / np.clip(dn, 1e-6, None), vnorm[idx])
                    else:
                        cn[t] = vnorm[idx]                           # object mesh surface normal
                near = (cvd - cvd.min(axis=-1, keepdims=True)) < cfg.contact_near_vertex_thresh
                self._np_future_contact = (vel_cond[:, None] & near).astype(np.float32)
                self._np_contact_vertex = cv; self._np_contact_normal = cn
                self._has_contact_map = True
            else:
                dist = np.linalg.norm(self._np_ft_pad - op[:, None, :], axis=-1)  # (F,10) centroid fallback
                self._np_future_contact = (vel_cond[:, None] & (dist < cfg.contact_dist_threshold)).astype(np.float32)

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

        self._ref_len = int(F)
        self._n_obj_parts = int(self._np_obj_dof.shape[1])

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
            cfg.viewer.lookat = (cx, cy, lookat_z)
            cfg.viewer.eye = (cx + off, cy + off, lookat_z + 0.12 * extent)

    def _load_object_mesh(self, cfg):
        """Active object SURFACE mesh (trimesh, object-local frame) from the ParaHome scan, for the
        contact-map (nearest-vertex + fingertip→vertex direction). Uses base.obj (the whole object for
        rigid; the base body for articulated — parts are an accepted approximation for now). None if
        trimesh/mesh absent → the caller falls back to the object-centroid contact gate.
        NOTE: takes `cfg` explicitly — called from _load_reference_trajectories BEFORE super().__init__,
        so self.cfg does not exist yet."""
        if not self._obj_name:
            return None
        try:
            import trimesh
        except Exception:  # noqa: BLE001
            return None
        # dataset_root = .../data/processed/parahome ; scan meshes live under .../data/raw/parahome/data/scan
        data_dir = os.path.dirname(os.path.dirname(cfg.dataset_root))
        src = os.path.join(data_dir, "raw", "parahome", "data", "scan", self._obj_name, "simplified", "base.obj")
        if not os.path.exists(src):
            return None
        try:
            m = trimesh.load(src, process=False, force="mesh")
            return m if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0 else None
        except Exception:  # noqa: BLE001
            return None

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
                    max_depenetration_velocity=5.0),
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

        # fingertip contact sensors (10, bimanual) — filter on the active object only.
        # track_contact_points requires a non-empty filter, so enable it only with an object.
        self._ft_sensors: dict[str, ContactSensor] = {}
        ft_filter = ["/World/envs/env_.*/Object"] if self._object_cfg is not None else []
        for name in self.cfg.fingertip_body_names:
            s = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{name}",
                filter_prim_paths_expr=ft_filter, history_length=3,
                track_air_time=False, track_contact_points=bool(ft_filter),
                # ParaHome objects use CONVEX-DECOMPOSITION colliders (concave shapes → many
                # sub-hulls); a fingertip touches several sub-hulls at once → one ≤4-pt manifold
                # each → >4 total, overflowing the default-4 contact-data buffer → device-side
                # assert in ContactSensor._unpack_contact_buffer_data. (grasp = single ConvexHull
                # → always ≤4, so it never hit this even with flat objects.) Raise the cap.
                max_contact_data_count_per_prim=self.cfg.ft_max_contact_points))
            self._ft_sensors[name] = s
            self.scene.sensors[f"ft_{name}"] = s
        # foot contact sensors (2) — filter on the GROUND ONLY so the measured force is the pure
        # ground reaction (excludes context furniture the foot may brush + self-collision). The
        # non-empty filter enables force_matrix_w (per-filter aggregated force; does NOT need
        # track_contact_points, so no contact-data buffer is allocated). _foot_contact_forces then
        # projects it onto the ground normal (+Z). track_air_time kept for completeness.
        # NB: the filter MUST target the actual ground COLLIDER prim, not the parent Xform. A cube-
        # on-ground test showed filter=/World/ground → force_matrix all-zero, while filter=
        # /World/ground/GroundPlane/CollisionPlane → correct +Z reaction (9.81 N for a 1 kg cube).
        # (GroundPlaneCfg always authors the collider at .../GroundPlane/CollisionPlane.)
        self._foot_sensors: dict[str, ContactSensor] = {}
        for name in self.cfg.foot_body_names:
            s = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{name}",
                filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
                history_length=3, track_air_time=True))
            self._foot_sensors[name] = s
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
        self._kpt_offsets = torch.tensor(kpt_off, device=dev, dtype=torch.float32)      # (56,3)
        self._n_kpt = len(kpt_names)                                                    # 56

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

        # ---- foot + root body ids ----
        self._foot_body_ids = torch.tensor(
            [self.robot.find_bodies(n)[0][0] for n in c.foot_body_names], device=dev, dtype=torch.long)
        self._root_body_id = self.robot.find_bodies(c.root_body_name)[0][0]
        # foot-contact reward: sole normal (foot-up = ankle_roll local +Z). The position gate is either
        # the reference-keypoint proximity (foot_gate_mode="kpt", default → _foot_kpt_slots_t below) or
        # the contact-onset anchor (="anchor" rollback → _foot_anchor_* below).
        self._foot_up_local = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=dev)   # (2,3)
        self._foot_kpt_slots_t = torch.tensor(self._foot_kpt_slots, device=dev, dtype=torch.long)  # (2,2)[L,R]=[ankle,ball]
        # non-foot body kpt slots = the 16 body kpts minus the 4 foot kpts (feet tracked separately,
        # up-weighted, via rew_foot_track). The foot slots are all < 16 (they are body keypoints).
        _foot_flat = {int(s) for pair in self._foot_kpt_slots for s in pair}
        self._body_nonfoot_slots = torch.tensor([i for i in range(16) if i not in _foot_flat],
                                                device=dev, dtype=torch.long)   # (12,)
        # foot-flatness soft-ramp thresholds (sin of the tilt tol/limit angles), precomputed once
        self._foot_flat_sin_tol = float(np.sin(np.deg2rad(c.foot_flat_tol_deg)))
        self._foot_flat_sin_limit = float(np.sin(np.deg2rad(c.foot_flat_limit_deg)))
        self._foot_flat_denom = max(self._foot_flat_sin_limit - self._foot_flat_sin_tol, 1e-6)
        self._foot_anchor_xy = torch.zeros(self.num_envs, 2, 2, device=dev)                  # (E,2,2) anchor-mode only
        self._foot_anchor_init = torch.ones(self.num_envs, dtype=torch.bool, device=dev)     # snap anchor on 1st post-reset step

        self._palm_to_landmark = torch.tensor(_PALM_TO_LANDMARK, device=dev).repeat(self.num_envs, 1)

        # ---- reference tensors (move numpy → device) ----
        def T(a):
            return torch.from_numpy(np.asarray(a)).to(dev)
        self._ref_kpts = T(self._np_ref_kpts)                          # (F,56,3)
        self._ref_root_pos = T(self._np_root_pos)                      # (F,3)
        self._ref_root_quat = _canon(T(self._np_root_quat))            # (F,4)
        self._ref_root_linvel = T(self._np_root_linvel)                # (F,3)
        self._ref_root_angvel = T(self._np_root_angvel)                # (F,3)
        self._ref_ft_pad = T(self._np_ft_pad)                          # (F,10,3)
        self._ref_obj_pos = T(self._np_obj_base[:, :3])                # (F,3)
        self._ref_obj_quat = _canon(T(self._np_obj_base[:, 3:7]))      # (F,4)
        self._ref_obj_dof = T(self._np_obj_dof)                        # (F,P)
        self._ref_joints = T(self._np_ref_joints) if self._np_ref_joints is not None else None
        # reference palm/wrist orientation per hand [L,R] for the wrist-rotation termination gate
        self._has_palm_ref = self._np_ref_palm_quat is not None
        self._ref_palm_quat = _canon(T(self._np_ref_palm_quat)) if self._has_palm_ref else None  # (F,2,4)
        self._has_object = self._object is not None
        self._RESERVE_ARTIC = 4                                        # reserved obs slots per parts

        # future-contact (F,10) + contact map (F,10,3 vertex/normal, object-local) — GR mechanism
        # computed at load (object velocity + fingertip proximity; contact map zeros until the
        # object mesh is wired into preprocessing → pad-normal fallback in the reward).
        self._future_contact = T(self._np_future_contact)             # (F,10)
        self._ref_contact_vertex_local = T(self._np_contact_vertex)   # (F,10,3) object-local
        self._ref_contact_normal_local = T(self._np_contact_normal)   # (F,10,3) object-local outward
        # foot planted-reference heuristic (F,2) from foot kpt height + |vz|
        self._ref_foot_contact = self._compute_ref_foot_contact()

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
        self._init_save_done = False
        self._failure_count = torch.zeros(self._ref_len, device=dev)
        self._adaptive_back_frames = int(round(c.adaptive_back_seconds / c.ref_dt))
        self._sampling_step_count = 0
        # per-env tracking-quality streak (grasp mechanism): _enough_continued = has tracking been
        # continuously "good enough" since reset; _enough_idx = last good frame (drives cache write
        # gate + failure-weighted sampling). Reset per env in _reset_idx.
        self._enough_continued = torch.ones(self.num_envs, dtype=torch.bool, device=dev)
        self._enough_idx = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        # pretrain-cache warm-start (208)
        self._pretrain_cache = None
        self._pretrain_init_flg = None
        self._last_pretrain_fallback_ratio = 0.0

    # ------------------------------------------------- reference-derived heuristics
    def _compute_ref_foot_contact(self) -> torch.Tensor:
        fc = torch.zeros(self._ref_len, 2, device=self.device)
        fps = 1.0 / self.cfg.ref_dt
        for j, slot in enumerate(self._foot_ref_slots):
            if slot < 0:
                continue
            z = self._ref_kpts[:, slot, 2]
            vz = torch.zeros_like(z); vz[1:] = (z[1:] - z[:-1]) * fps
            planted = (z < self.cfg.ref_foot_planted_height) & (vz.abs() < self.cfg.ref_foot_planted_velz)
            fc[:, j] = planted.float()
        return fc

    # ------------------------------------------------- action helpers
    def _unscale(self, q: torch.Tensor) -> torch.Tensor:
        return 2.0 * (q - self._ctrl_lower) / (self._ctrl_upper - self._ctrl_lower) - 1.0

    def _scale(self, a: torch.Tensor) -> torch.Tensor:
        return self._ctrl_lower + 0.5 * (a + 1.0) * (self._ctrl_upper - self._ctrl_lower)

    # -------------------------------------------------- action (per-group EMA + delta)
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # advance the reference frame for this step (reset overrides it at episode start)
        self._frame_idx = (self._frame_idx + 1).clamp(max=self._ref_len - 1)
        a = actions.clamp(-1.0, 1.0)
        c = self.cfg
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

    def _fingertip_forces(self, direction_w: torch.Tensor) -> torch.Tensor:
        """(E,10) compressive force projected on direction_w (per-finger, clamped ≥0)."""
        out = torch.zeros(self.num_envs, 10, device=self.device)
        if not self._has_object:
            return out
        for i, name in enumerate(self.cfg.fingertip_body_names):
            fm = self._ft_sensors[name].data.force_matrix_w                # (E,1,1,3) or None
            if fm is None:
                continue
            f = fm.reshape(self.num_envs, -1, 3).sum(dim=1)                # (E,3)
            out[:, i] = (f * direction_w[:, i]).sum(-1).clamp_min(0.0)
        return out

    def _get_fingertip_contact_pos_w(self) -> torch.Tensor:
        """(E,10,3) actual contact-point world position per fingertip; NaN where no contact
        (requires track_contact_points=True → set only when an object filter exists)."""
        out = torch.full((self.num_envs, 10, 3), float("nan"), device=self.device)
        if not self._has_object:
            return out
        for i, name in enumerate(self.cfg.fingertip_body_names):
            cp = getattr(self._ft_sensors[name].data, "contact_pos_w", None)
            if cp is not None:
                out[:, i] = cp.reshape(self.num_envs, -1, 3)[:, 0]
        return out

    def _foot_contact_forces(self) -> torch.Tensor:
        """(E,2) ground-reaction force per foot projected onto the FOOT SOLE NORMAL (foot-up axis =
        ankle_roll_link local +Z, rotated to world) — the foot analog of _fingertip_forces' pad-normal
        projection. Projecting on the sole normal instead of fixed world +Z rewards FLAT contact: a
        flat foot has sole-up ≈ vertical → the (vertical) reaction projects to ~full magnitude; a
        heel/edge contact tilts the sole-up so the vertical reaction projects to cos(tilt) < 1 → heel-
        only planting is discouraged. force_matrix_w (foot vs /World ground filter) isolates the true
        foot↔ground force from furniture/self contact."""
        out = torch.zeros(self.num_envs, 2, device=self.device)
        q = self.robot.data.body_quat_w[:, self._foot_body_ids]          # (E,2,4) ankle_roll quats
        sole_up = math_utils.quat_apply(q, self._foot_up_local.unsqueeze(0).expand(self.num_envs, -1, -1))  # (E,2,3)
        for j, name in enumerate(self.cfg.foot_body_names):
            fm = self._foot_sensors[name].data.force_matrix_w            # (E,1,1,3) or None (ground filter)
            if fm is None:
                continue
            f = fm.reshape(self.num_envs, -1, 3).sum(dim=1)              # (E,3) foot↔ground contact force
            out[:, j] = (f * sole_up[:, j]).sum(dim=-1).clamp_min(0.0)   # force · sole-normal (flat→max)
        return out

    def _foot_contact_flags(self) -> torch.Tensor:
        """(E,2) contact flag per foot: SOLE-NORMAL-projected ground force > foot_contact_force_thresh."""
        return (self._foot_contact_forces() > self.cfg.foot_contact_force_thresh).float()

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
        A = [
            (root_pos[:, 2:3]),                                         # root height (1)
            _quat_to_6d(root_quat),                                     # root ori 6d (6)
            self.robot.data.root_lin_vel_w,                            # (3)
            self.robot.data.root_ang_vel_w * vs,                       # (3)
            self.robot.data.projected_gravity_b,                       # (3)
            self._unscale(self.robot.data.joint_pos[:, self._action_joint_ids_t]),  # (65)
            self.robot.data.joint_vel[:, self._action_joint_ids_t] * vs,            # (65)
            _quat_to_6d(palm_quat).reshape(E, -1),                     # palm ori 6d ×2 (12)
            palm_linvel.reshape(E, -1),                               # palm linvel ×2 (6)
            (palm_angvel * vs).reshape(E, -1),                        # palm angvel ×2 (6)
            ft_vel.reshape(E, -1),                                     # fingertip linvel ×10 (30)
        ]

        # ---- BLOCK B: reference tracking + look-ahead ----
        kpts = self._robot_kpts_w()                                    # (E,56,3) world
        kpts_local = kpts - org[:, None, :]                            # env-local
        delta_kpts = self._ref_kpts[nfr] - kpts_local                  # look-ahead delta
        ref_root_p = self._ref_root_pos[nfr]                           # look-ahead (grasp-parity: obs deltas = next frame)
        ref_root_q = self._ref_root_quat[nfr]
        delta_root_pos = ref_root_p - (root_pos - org)
        droot_q = _canon(math_utils.quat_mul(ref_root_q, math_utils.quat_conjugate(root_quat)))
        # NO phase/time signal (grasp env doesn't use one — progress is conveyed by the next-frame
        # reference deltas / look-ahead below, keeping obs consistent with the existing tasks).
        B = [
            kpts_local.reshape(E, -1),                                 # (168)
            delta_kpts.reshape(E, -1),                                 # (168)
            ref_root_p[:, 2:3],                                        # ref root height (1)
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
        dobj_p = self._ref_obj_pos[nfr] - obj_p                       # look-ahead delta (next-frame ref)
        dobj_q = _canon(math_utils.quat_mul(self._ref_obj_quat[nfr], math_utils.quat_conjugate(obj_q)))
        artic = torch.zeros(E, self._RESERVE_ARTIC * 2, device=self.device)
        if self._n_obj_parts > 0:
            p = min(self._n_obj_parts, self._RESERVE_ARTIC)
            artic[:, :p] = self._ref_obj_dof[nfr, :p]                  # ref DOF, look-ahead (next frame)
        _tip, pad_inward = self._robot_ft_w()
        # object-LOCAL fingertip→target offset (mirrors grasp delta_ft_obj): fingertip position relative
        # to its target, expressed in the OBJECT frame → object-pose-invariant contact signal. Target =
        # the nearest object-surface VERTEX on expected contact (future_contact), else the reference
        # fingertip pad. obj_q is LIVE (train) / REFERENCE (pretrain) → same phase pattern as other obj obs.
        ref_ft_w = self._ref_ft_pad[nfr] + org[:, None, :]            # (E,10,3) reference pad (world), look-ahead
        oq_exp3 = obj_q[:, None, :].expand(-1, 10, -1)                # (E,10,4)
        if self._has_contact_map:
            obj_pos_w = (obj_p + org)[:, None, :]                     # (E,1,3) world object origin
            ref_vertex_w = math_utils.quat_apply(oq_exp3, self._ref_contact_vertex_local[nfr]) + obj_pos_w
            in_contact = self._future_contact[nfr].unsqueeze(-1).bool()  # (E,10,1) look-ahead
            target_w = torch.where(in_contact, ref_vertex_w, ref_ft_w)
        else:
            target_w = ref_ft_w
        delta_ft_obj = math_utils.quat_apply(math_utils.quat_conjugate(oq_exp3), _tip - target_w)  # (E,10,3)
        C = [
            obj_p, _quat_to_6d(obj_q), obj_lv, obj_av * vs,            # (15)
            dobj_p, _quat_to_6d(dobj_q),                              # (9)
            delta_ft_obj.reshape(E, -1),                              # obj-local fingertip offset (30)
            artic,                                                    # (8)
            self._future_contact[fr],                                 # reference expected fingertip contact (10)
            self._ref_foot_contact[fr],                               # reference expected foot contact = future_foot_contact (2)
            self._fingertip_forces(pad_inward),                       # current robot fingertip contact force (10)
            self._foot_contact_forces(),                              # current robot sole-normal foot force (2)
            self._prev_action,                                        # (65)
        ]

        obs = torch.cat(A + B + C, dim=-1)
        assert obs.shape[-1] == c.observation_space, (
            f"obs dim {obs.shape[-1]} != cfg.observation_space {c.observation_space} "
            "(block-C dims must be invariant across the has_object flip)")
        # capture prev_action = normalized integrated target (delta groups) / smoothed (abs groups)
        self._prev_action = self._smoothed_actions.clone()
        for gname, sl in self._group_slices.items():
            if self._delta_cfg[gname][0]:                              # delta group → normalized integrated target
                self._prev_action[:, sl] = self._unscale_slice(gname, self._delta_target[gname])
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

    # ---------------------------------------------------------------- rewards
    def _compute_errors(self):
        """Shared error terms for reward + termination."""
        org = self.scene.env_origins
        fr = self._frame()
        kpts = self._robot_kpts_w() - org[:, None, :]
        ref = self._ref_kpts[fr]
        dk = ref - kpts
        body_err = dk[:, :16].norm(dim=-1).mean(dim=-1)               # (E,) FULL 16 body (termination gate)
        hand_err = dk[:, 16:].norm(dim=-1).mean(dim=-1)               # (E,)
        # per-foot (ankle+ball) mean kpt error, [L,R] — the foot-contact reward's reference-proximity gate
        foot_kpt_err = dk[:, self._foot_kpt_slots_t].norm(dim=-1).mean(dim=-1)   # (E,2)
        # SPLIT for the reward: 12 non-foot body kpts vs 4 foot kpts (feet up-weighted via rew_foot_track,
        # un-diluted from the 16-kpt mean). Termination still uses the full body_err above.
        body_nonfoot_err = dk[:, self._body_nonfoot_slots].norm(dim=-1).mean(dim=-1)   # (E,) 12 kpts
        foot_track_err = foot_kpt_err.mean(dim=-1)                                     # (E,) 4 foot kpts
        # fingertip pad tracking (contact-conditioned handled in reward; raw here)
        tip, pad_inward = self._robot_ft_w()
        ft_per = (self._ref_ft_pad[fr] - (tip - org[:, None, :])).norm(dim=-1)   # (E,10)
        ft_err = ft_per.mean(dim=-1)
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
            ref_vtx_w = math_utils.quat_apply(oq_e, self._ref_contact_vertex_local[fr]) + obj_pos.unsqueeze(1)
            ft_in_refobj = math_utils.quat_apply(math_utils.quat_conjugate(roq_e),
                                                 self._ref_ft_pad[fr] - self._ref_obj_pos[fr].unsqueeze(1))
            ref_ft_drift = math_utils.quat_apply(oq_e, ft_in_refobj) + obj_pos.unsqueeze(1)
            in_contact = self._future_contact[fr].unsqueeze(-1).bool()            # (E,10,1)
            ft_target = torch.where(in_contact, ref_vtx_w, ref_ft_drift)          # (E,10,3)
            ft_reward = (ft_target - tip_l).norm(dim=-1).mean(dim=-1)             # (E,) contact-conditioned
        else:
            obj_pos_err = torch.zeros(self.num_envs, device=self.device)
            obj_rot_err = torch.zeros(self.num_envs, device=self.device)
            ft_reward = ft_err                                                     # no object → raw pad target
        # palm/wrist rotation deviation (bimanual, worst hand). Robot palm body quat vs the retarget
        # reference palm quat (same robot0_{l,r}_palm frame → direct compare, no landmark conversion).
        if self._has_palm_ref:
            palm_q = _canon(self.robot.data.body_quat_w[:, self._palm_body_ids])      # (E,2,4)
            pqe = _canon(math_utils.quat_mul(self._ref_palm_quat[fr],
                                             math_utils.quat_conjugate(palm_q)))       # (E,2,4)
            wrist_rot_err = 2.0 * torch.arcsin(pqe[..., 1:].norm(dim=-1).clamp(max=1.0)).amax(dim=-1)  # (E,)
        else:
            wrist_rot_err = torch.zeros(self.num_envs, device=self.device)
        return dict(body=body_err, hand=hand_err, ft=ft_err, ft_reward=ft_reward, ft_per=ft_per, tip=tip,
                    pad_inward=pad_inward, root_pos=root_pos_err, root_rot=root_rot_err,
                    obj_pos=obj_pos_err, obj_rot=obj_rot_err, root_quat=root_quat,
                    wrist_rot=wrist_rot_err, foot_kpt=foot_kpt_err,
                    body_nonfoot=body_nonfoot_err, foot_track=foot_track_err)

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        e = self._errs                                    # set by _get_dones (runs first each step)
        fr = self._frame()

        # contact-conditioned fingertip force (matches robotis_shadow_grasp, bimanual 10 fingers):
        #   force projected on the grounded contact-vertex normal (→ pad-inward fallback), gated by
        #   future_contact (expected-contact) AND the ACTUAL contact position near the contact vertex.
        contact_flag = self._future_contact[fr]                       # (E,10)
        pad_inward = e["pad_inward"]                                  # (E,10,3)
        if self._has_object:
            obj_pos = self._object.data.root_pos_w                    # world (incl env origin)
            obj_quat = self._object.data.root_quat_w
            oq_exp = obj_quat[:, None, :].expand(-1, 10, -1)
            gn = math_utils.quat_apply(oq_exp, self._ref_contact_normal_local[fr])    # (E,10,3) world
            gn_norm = gn.norm(dim=-1, keepdim=True)
            gn_valid = gn_norm.squeeze(-1) > 1e-6                     # (E,10) mesh present
            gn_unit = gn / gn_norm.clamp(min=1e-6)
            force_dir = torch.where(gn_norm > 1e-6, gn_unit, pad_inward) if c.use_grounded_normal else pad_inward
            forces = self._fingertip_forces(force_dir)               # (E,10)
            # contact-position gate: actual contact near the prescribed contact vertex (world).
            ref_vertex_world = math_utils.quat_apply(oq_exp, self._ref_contact_vertex_local[fr]) + obj_pos[:, None, :]
            if c.use_contact_point_gate:
                cp = self._get_fingertip_contact_pos_w()             # (E,10,3) NaN where no contact
                delta = cp - ref_vertex_world
                if c.use_contact_normal_gate:
                    d_n = (delta * gn_unit).sum(dim=-1)
                    d_t = (delta - d_n.unsqueeze(-1) * gn_unit).norm(dim=-1)
                    near = (d_t < c.contact_match_dist) & (d_n > -c.contact_normal_tol)
                    iso = delta.norm(dim=-1) < c.contact_match_dist
                    contact_condition = torch.where(gn_valid, near, iso).float()
                else:
                    contact_condition = (delta.norm(dim=-1) < c.contact_match_dist).float()
                contact_condition = torch.nan_to_num(contact_condition, nan=0.0)
            else:
                contact_condition = (e["ft_per"] < c.contact_match_dist).float()
        else:
            forces = torch.zeros(self.num_envs, 10, device=self.device)
            contact_condition = torch.zeros(self.num_envs, 10, device=self.device)
        fforce = forces * contact_flag * contact_condition            # (E,10)
        n_contacts = contact_flag.sum(dim=-1, keepdim=True)
        force_rew = (fforce.clamp(0.0, 0.5) / (n_contacts + 1e-6) / 1.5).sum(dim=-1)

        # FOOT-CONTACT reward (fingertip-force analog). Per foot, quality q ∈ [0,1]:
        #   PLANTED (ref says down): q = flat-force × position-gate, i.e.
        #     (sole-normal force / foot_force_cap, clamped) × GATE. Rewards pressing FLAT (sole-normal
        #     projection) AT THE RIGHT PLACE — heel-only / dragged / wrong-spot contact scores low.
        #   SWING (ref says up): q = (1 − contact) × GATE — reward NOT touching AT the reference spot.
        # GATE (foot_gate_mode):
        #   "kpt"    → foot's (ankle+ball) mean kpt error < foot_kpt_gate of its MOVING reference. The
        #              reference feet pivot during a turn, so the foot may follow instead of being pinned.
        #   "anchor" → foot within foot_anchor_tol of the xy where it FIRST touched down (frozen). Rollback.
        # POSITIVE reward rew_foot_contact·q ∈ [0,w] (added outside the tracking clamp): correct foot
        # behavior (flat plant at the right place, or clean lift on swing) is rewarded, not just the
        # absence of a mistake. Unlike the tracking penalties this can lift an alive step above 0.
        rfc = self._ref_foot_contact[fr]                                      # (E,2) ref planted (height-based)
        foot_force = self._foot_contact_forces()                             # (E,2) sole-normal projected
        contact = (foot_force > c.foot_contact_force_thresh).float()         # (E,2)
        force_n = (foot_force / c.foot_force_cap).clamp(0.0, 1.0)             # (E,2) flat-force, normalized
        if c.foot_gate_mode == "kpt":
            gate = (e["foot_kpt"] < c.foot_kpt_gate).float()                 # (E,2) near the MOVING reference
        else:  # "anchor" (rollback): freeze xy at the 1st post-reset step (body_pos_w stale in _reset_idx),
            #    hold it while in contact, follow the foot while swinging; reward staying near that spot.
            foot_xy = self.robot.data.body_pos_w[:, self._foot_body_ids, :2]  # (E,2,2)
            keep = (~self._foot_anchor_init.view(-1, 1, 1)) & (contact.unsqueeze(-1) > 0.5)
            self._foot_anchor_xy = torch.where(keep, self._foot_anchor_xy, foot_xy)
            self._foot_anchor_init = torch.zeros_like(self._foot_anchor_init)
            gate = ((foot_xy - self._foot_anchor_xy).norm(dim=-1) < c.foot_anchor_tol).float()  # (E,2)
        # FLATNESS soft-ramp: sole tilt from vertical (sin θ = horizontal component of the sole-up axis
        # = ankle_roll local +Z rotated to world). flat_factor = 1 for a flat sole (tilt ≤ tol), ramps
        # linearly to 0 at the limit angle, 0 beyond → a heel-only / toe-only (forefoot) plant earns
        # little/no foot reward. Applied to the PLANTED branch only (force magnitude saturates so it can't
        # encode tilt; this decouples "flat?" from "pressing?").
        sole_up = math_utils.quat_apply(self.robot.data.body_quat_w[:, self._foot_body_ids],
                                        self._foot_up_local.unsqueeze(0).expand(self.num_envs, -1, -1))  # (E,2,3)
        tilt = sole_up[..., :2].norm(dim=-1)                                 # (E,2) sin(tilt): 0=flat
        flat_factor = ((self._foot_flat_sin_limit - tilt) / self._foot_flat_denom).clamp(0.0, 1.0)  # (E,2)
        e["foot_tilt"] = tilt                                                # stash for logging (Error / foot_tilt)
        # position GATE applies in BOTH phases → the foot is rewarded only NEAR its reference kpt,
        # whether planted (force·gate·flat) or swinging ((1−contact)·gate). Without the swing-side gate a
        # foot got full reward for merely being airborne ANYWHERE → free kicking; the gate kills that.
        q_foot = torch.where(rfc > 0.5, force_n * gate * flat_factor, (1.0 - contact) * gate)   # (E,2) per-foot quality
        foot_match = q_foot.mean(dim=-1)                                     # (E,) logged as foot quality
        foot_reward = c.rew_foot_contact * foot_match                        # ≥ 0 POSITIVE reward ∈ [0, w]
        # foot slip: horizontal drag of a foot that is ACTUALLY touching the ground (gated by the
        # robot's real contact flag, NOT the reference rfc) — a foot only "slips" while on the ground.
        # A foot lifted for repositioning (contact=0) is not penalized; a foot dragging on the floor is.
        # Velocity is taken at the ANKLE JOINT (link/actor frame origin = the tracked ankle keypoint),
        # NOT body_lin_vel_w (= CoM). A pivot about ~the leg axis keeps the ankle joint near-stationary
        # (ω×r≈0 on-axis) so it is not penalized, whereas the forward-offset CoM would be — this lets
        # the robot rotate (hip_yaw pivot) without a spurious slip penalty. (body_link_lin_vel_w is the
        # CoM velocity transported to the link origin: v_joint = v_com + ω×(r_joint − r_com).)
        foot_vel = self.robot.data.body_link_lin_vel_w[:, self._foot_body_ids]    # (E,2,3) ankle-joint velocity
        slip = (foot_vel[..., :2].norm(dim=-1) * contact).mean(dim=-1)

        # articulation dof tracking (0 until object joints are read from the sim)
        artic_err = torch.zeros(self.num_envs, device=self.device)

        # NOTE: root LINEAR/ANGULAR velocity error terms REMOVED from the reward — the proven
        # grasp reward has none (it is fixed-base), they are redundant finite-diffs of the
        # root_pos/root_ori that are already tracked, and the robot's raw angvel is very noisy at
        # reset (zero-action tumbling → ~16 rad/s spikes) which dominated the penalty. The clean
        # reference velocities are still used for RSI reset initial velocity (that is fine).

        # per-group action regularization + pose reg + action rate
        act = self._smoothed_actions
        reg_legs = (act[:, self._group_slices["legs"]] ** 2).sum(-1)
        reg_arms = (act[:, self._group_slices["arms"]] ** 2).sum(-1)
        reg_hands = (act[:, self._group_slices["hands"]] ** 2).sum(-1)
        jp = self.robot.data.joint_pos[:, self._action_joint_ids_t]
        # residual mode: regularize toward the REFERENCE joints (retarget), not the robot default pose —
        # else pose_reg fights the reference. This is the residual-magnitude regularizer (keep the learned
        # correction small). Falls back to the default pose when no retarget joints are present.
        pose_ref = (self._ref_joints[self._frame()]
                    if (self.cfg.residual_action and self._ref_joints is not None)
                    else self.robot.data.default_joint_pos[:, self._action_joint_ids_t])
        pose_reg = ((jp - pose_ref) ** 2).sum(-1)
        action_rate = ((act - self._prev_action) ** 2).sum(-1)

        alive = (~self._died).float()                     # _died set by _get_dones this step

        # ── BOUNDED reward (mirrors the proven grasp RSI structure) ──────────────────────
        # Group ALL imitation/tracking penalties and CLAMP the sum at -rew_alive, so an
        # alive-but-poorly-tracking step nets ≤0 after the alive bonus (no free survival reward).
        # foot_reward (positive) / foot_slip / force / regs stay OUTSIDE the clamp (grasp keeps its
        # force+regs outside its tracking_penalty too); total reward floored at 0.
        tracking_penalty = (
            c.rew_body_kpts * e["body_nonfoot"] + c.rew_foot_track * e["foot_track"]
            + c.rew_hand_kpts * e["hand"] + c.rew_fingertip * e["ft_reward"]
            + c.rew_root_pos * e["root_pos"] + c.rew_root_ori * e["root_rot"]
            + c.rew_obj_pos * e["obj_pos"] + c.rew_obj_rot * e["obj_rot"] + c.rew_obj_artic * artic_err
        )
        reward = (
            c.rew_alive * alive
            + tracking_penalty.clamp(min=-c.rew_alive)
            + foot_reward
            + c.rew_foot_slip * slip
            + c.rew_fingertip_force * force_rew
            + c.rew_action_reg_legs * reg_legs + c.rew_action_reg_arms * reg_arms
            + c.rew_action_reg_hands * reg_hands + c.rew_pose_reg * pose_reg
            + c.rew_action_rate * action_rate
        ).clamp(min=0.0)
        self._save_state_cache()                          # per-frame best-state RSI cache
        # per-term reward contributions (weighted) for the Episode_Reward Tensorboard group.
        # tracking terms are logged PRE-clamp (individual insight); "tracking_penalty" is the
        # clamped group value actually added to the reward.
        ep_rew = {
            "alive": c.rew_alive * alive,
            "body_kpts": c.rew_body_kpts * e["body_nonfoot"],
            "foot_track": c.rew_foot_track * e["foot_track"],
            "hand_kpts": c.rew_hand_kpts * e["hand"],
            "fingertip": c.rew_fingertip * e["ft_reward"],
            "root_pos": c.rew_root_pos * e["root_pos"],
            "root_ori": c.rew_root_ori * e["root_rot"],
            "obj_pos": c.rew_obj_pos * e["obj_pos"],
            "obj_rot": c.rew_obj_rot * e["obj_rot"],
            "tracking_penalty": tracking_penalty.clamp(min=-c.rew_alive),
            "foot": foot_reward,
            "foot_slip": c.rew_foot_slip * slip,
            "fingertip_force": c.rew_fingertip_force * force_rew,
            "action_reg": (c.rew_action_reg_legs * reg_legs + c.rew_action_reg_arms * reg_arms
                           + c.rew_action_reg_hands * reg_hands),
            "pose_reg": c.rew_pose_reg * pose_reg,
            "action_rate": c.rew_action_rate * action_rate,
            "total": reward,
        }
        self._log_reward_terms(e, tracking_penalty, ep_rew, fr)
        return reward

    def _log_reward_terms(self, e, tracking_penalty, ep_rew, fr):
        log = self.extras.setdefault("log", {})
        log.update({
            "Error / body_kpts": e["body"].mean(), "Error / hand_kpts": e["hand"].mean(),
            "Error / fingertip": e["ft"].mean(), "Error / root_pos": e["root_pos"].mean(),
            "Error / root_rot": e["root_rot"].mean(), "Error / obj_pos": e["obj_pos"].mean(),
            "Error / wrist_rot": e["wrist_rot"].mean(), "Error / foot_kpt": e["foot_kpt"].mean(),
            "Error / foot_track": e["foot_track"].mean(), "Error / body_nonfoot": e["body_nonfoot"].mean(),
            "Error / foot_tilt_deg": torch.rad2deg(torch.asin(e["foot_tilt"].clamp(max=1.0))).mean(),
            "Curriculum / reached_frame": float(self._reached_frame),
            "Curriculum / pretrain_fallback": self._last_pretrain_fallback_ratio,
            "Curriculum / cache_coverage": float((~self._init_flg).sum().item()) / self._ref_len,
            # reward-shaping diagnostics (kept out of skrl's "Reward /" group):
            "Episode_Reward / tracking_penalty_raw": tracking_penalty.mean(),
            "Episode_Reward / clamp_frac": (tracking_penalty < -self.cfg.rew_alive).float().mean(),
        })
        # per-term reward contributions → Episode_Reward group ("Reward /" holds skrl built-ins).
        for k, v in ep_rew.items():
            log[f"Episode_Reward / {k}"] = v.mean()
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
                      "hand": e["hand"] > cc.term_hand_kpt_err, "ft": e["ft"] > cc.term_ft_err,
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
        single body/locomotion gate; hands + object add their own."""
        c = self.cfg
        d = e["body"] > c.term_body_kpt_err
        #  hands (mirrors grasp ft/wrist; bimanual): fingertip + finger-chain keypoint deviation.
        if c.enable_hand_termination:
            d = d | (e["hand"] > c.term_hand_kpt_err) | (e["ft"] > c.term_ft_err)
        #  wrist/palm rotation deviation (mirrors grasp max_wrist_rot_err) — only when a retarget
        #  palm-orientation reference is present (else wrist_rot is all-zero → inert).
        if c.enable_wrist_rot_termination and self._has_palm_ref:
            d = d | (e["wrist_rot"] > c.term_wrist_rot_err)
        #  object (mirrors grasp obj_pos + obj_rot) — only when an active object is present.
        if self._has_object:
            d = d | (e["obj_pos"] > c.term_obj_pos_err) | (e["obj_rot"] > c.term_obj_rot_err)
        if not c.termination:
            d = torch.zeros_like(d)
        return d

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # runs BEFORE _get_rewards each step → compute + cache errors here for both.
        self._errs = self._compute_errors()
        self._died = self._dones_deviation(self._errs)
        # TIME-OUT (bootstrapped, NOT a failure). Grasp/TJ chunking: episode length = _num_frame_chunk
        # and the adaptive-sampling START is clamped to [0, ref_len - num_frame_chunk] (see _reset_idx),
        # so every episode runs exactly num_frame_chunk steps and ends AT (never past) the reference
        # end — no separate reached-end gate needed and no freeze-on-last-frame.
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._died, time_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids) -> None:
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
        candidates = have_train.clone()
        if have_pre is not None:
            candidates = candidates | have_pre
        if candidates.sum() == 0:
            candidates[0] = True                                     # frame 0 always available
        cand_idx = torch.nonzero(candidates, as_tuple=False).squeeze(-1)

        # PRETRAIN (failure_weighted_sampling=False) → always UNIFORM over cached frames.
        # TRAIN → uniform for the first uniform_sampling_steps, then failure-weighted.
        use_uniform = (
            (not c.adaptive_sampling)
            or (not c.failure_weighted_sampling)
            or self._sampling_step_count * self.num_envs < c.uniform_sampling_steps
        )
        if use_uniform:
            pick = cand_idx[torch.randint(0, len(cand_idx), (n,), device=dev)]
        else:
            w = self._failure_count[cand_idx]
            probs = w / (w.sum() + 1e-6)
            ur = c.adaptive_uniform_ratio
            probs = (probs + ur / len(cand_idx)) / (1.0 + ur)
            pick = cand_idx[torch.multinomial(probs, n, replacement=True)]
        # rewind for run-up, then clamp START to [0, ref_len - num_frame_chunk] (grasp/TJ): the
        # target `pick` ranges the FULL trajectory (so failure-weighting can concentrate anywhere),
        # but the START is capped so the episode (num_frame_chunk steps) ends AT the reference end,
        # never overshooting into a frozen last frame. When num_frame_chunk >= ref_len (short clip /
        # long episode_length_s) upper=0 → start=0 → adaptive sampling effectively off (matches grasp).
        upper = max(0, self._ref_len - self._num_frame_chunk)
        start = (pick - self._adaptive_back_frames).clamp(min=0).clamp(max=upper)
        # safeguard: start must be covered by a cache; snap uncovered → 0 (in [0,upper]; covered via
        # frame-0 init-save, else the restore falls back to reference+default pose).
        bad = ~candidates[start]
        start[bad] = 0
        self._frame_idx[env_ids] = start
        # reset the tracking-quality streak for the reset envs (grasp mechanism)
        self._enough_continued[env_ids] = True
        self._enough_idx[env_ids] = start

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
            ovel = torch.zeros(n, 6, device=dev)
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
        # re-snap the foot contact-onset anchor on the next reward step (body_pos_w is stale here)
        self._foot_anchor_init[env_ids] = True
        self._prev_action[env_ids] = self._smoothed_actions[env_ids]

    # -------------------------------------------------------- state cache write
    def _save_state_cache(self) -> None:
        """Store per-frame best (highest-reward) full-body state into the 222-D train cache.

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
        start_cutoff = action_fps * 2 // 3                                  # ~20 frames @30Hz
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

        # reward proxy: negative total tracking error (higher = better)
        r = -(e["body"] + e["hand"] + e["root_pos"])                        # (E,)
        # write only when tracking is still good AND the new state beats the cached reward
        better = r > self._state_cache[fr, 0]                               # (E,) fancy-index gather
        update_mask = gate & self._enough_continued & better               # (E,)
        if not update_mask.any():
            return
        # build the full (E,222) state row once; only masked rows get written
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
        # per unique frame among updating envs, pick the highest-reward env (order-independent
        # equivalent of the sequential loop's "keep best per frame > pre-existing cache")
        for uf in torch.unique(fr[update_mask]):
            m = (fr == uf) & update_mask
            best_env = m.nonzero(as_tuple=True)[0][r[m].argmax()]
            self._state_cache[uf] = state[best_env]
            self._init_flg[uf] = False
            self._reached_frame = max(self._reached_frame, int(uf.item()))

    # ---------------------------------------------------- pretrain-cache warm-start
    def set_pretrain_cache(self, npz_path: str) -> bool:
        """Load the pretrain phase's 208-D read-only state cache for RSI warm-start.
        Gated by cfg.pretrain_cache_warmstart (False → vanilla RSI: empty train cache, _reached_frame
        gate, fixed-home + frame-0-IK fallback). [ROLLBACK MARKER: pretrain-cache-warmstart]"""
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
        self._pretrain_cache = cache
        self._pretrain_init_flg = flg
        return True
