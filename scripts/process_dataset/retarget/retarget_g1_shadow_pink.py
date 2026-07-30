"""Full-frame pink-IK retargeting of ParaHome keypoints → composite G1+Shadow.

DESIGN (contact-preserving, NO reference rescale): the reference keypoints + object trajectory are
kept EXACTLY as in the dataset. The ROBOT adapts: a FREE-FLYER root is solved (soft-tracked to the
human pelvis) together with the body joints so the (shorter) G1 arms reach the human hand/wrist
keypoints and the (shorter) legs reach the ground-level ankle keypoints — the robot stands closer /
lowers its pelvis. The solved (adjusted) root is written out and becomes the env's reference root.

Four concerns handled by the task setup (see the workflow design):
  * TEMPORAL CONSISTENCY — per-frame warm-start (body + fingers) + moving-average smoothing of the
    Kabsch input points + per-frame velocity/acceleration RATE-LIMITING (post-solve, tangent-space,
    group caps) + a zero-phase Savitzky–Golay pass over the solved joint/root trajectories.
  * FOOT CONTACT — per-frame PLANTED detection (env heuristic on the ballfoot) → the planted foot is
    pinned FLAT (Rz-yaw orientation, roll/pitch 0) at z=ANKLE_FLAT_Z with a frozen (anti-slip) xy;
    swing feet track the human ankle at lower cost.
  * LOWER-BODY STABILITY — knee/hip position costs lowered (mismatched human knee/hip are unreliable)
    and a stronger stance PostureTask regularizes the legs (curbs over-rotation), while the soft root
    track + hard planted feet lower the pelvis so the short legs are not contorted to reach.
  * WRIST-OBJECT — Kabsch palm pose (position+orientation) with a raised wrist orientation cost; the
    fingers warm-start frame-to-frame so the hand rides the object continuously.

Body→hands CASCADE, two pinocchio models:
  MODEL 1  G1 body, FREE-FLYER root (nq=36). USD-derived URDF (g1_from_usd.urdf, sim-consistent).
  MODEL 2  Shadow hand per side (finger DOF, WRJ locked). Palm-relative finger IK after the body solve.

Output per clip → data/processed/parahome/g1_shadow/<class>/<clip>/0/trajectory.npz:
  g1_joint_pos (F,65)   65 actuated joint angles in env _action_joint_ids order (NAME-mapped)
  g1_root_pose (F,7)    adjusted root pose (pos + quat wxyz), world (ParaHome) frame
  g1_palm_quat (F,2,4)  reference palm/wrist world orientation per hand [L,R] (wxyz) = Kabsch palm
                        pose = robot0_{l,r}_palm body frame → env wrist-rotation termination gate

Run (standalone, pinocchio+pink, NO Isaac Sim):
    /home/peunsu/anaconda3/envs/env_isaaclab/bin/python scripts/process_dataset/retarget/retarget_g1_shadow_pink.py \
        [--clip_class single_rigid] [--clip <name>|ALL] [--frame0_only] [--check] [--overwrite] [--no_smooth]
Prereqs: dump_g1_joint_order.py (g1_shadow_joint_order.json) + extract_g1_urdf_from_usd.py (g1_from_usd.urdf).

TODO (next increment): CoM-over-support ComTask, foot/leg self-collision barriers, explicit pelvis
pre-lower, and the g1_ref_kpts FK output stream for a reward that tracks the retargeted (reachable)
keypoints (A-plan Strategy 2).
"""

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shadow_usd_model import build_shadow_from_usd   # sim-consistent Shadow model (from composite USD)

_REPO = Path(__file__).resolve().parents[3]
_DATA = _REPO / "source" / "robotis_sh5" / "data"
_G1_URDF = str(_DATA / "robots" / "G1" / "urdf" / "g1_from_usd.urdf")
_JOINT_ORDER = _DATA / "robots" / "G1" / "g1_shadow_joint_order.json"
_USD_COMPOSITE = str(_DATA / "robots" / "G1" / "G1_shadow.usd")
_PARAHOME = _DATA / "processed" / "parahome"

# Body FrameTasks (position-only): (parahome_idx, link/OP_FRAME, position_cost). The FEET are NOT
# here — they are built per-frame as full-pose (planted-aware) tasks in solve_body_ik. Leg costs
# lowered (human knee/hip are cross-embodiment-mismatched → unreliable; let posture + feet shape the
# legs). ParaHome body indices VERIFIED against env BODY_KPTS.
_BODY_TASKS = [
    (4, "torso_kpt", 10.0),   # OP_FRAME on torso_link at TORSO_KPT_OFFSET (≈ jT9T8 / upper chest)
    (8, "right_shoulder_pitch_link", 8.0), (9, "right_elbow_link", 4.0), (10, "right_wrist_yaw_link", 40.0),
    (12, "left_shoulder_pitch_link", 8.0), (13, "left_elbow_link", 4.0), (14, "left_wrist_yaw_link", 40.0),
    (15, "right_hip_pitch_link", 8.0), (16, "right_knee_link", 5.0),
    (19, "left_hip_pitch_link", 8.0), (20, "left_knee_link", 5.0),
]
# Feet (built per-frame). Ankle link + toe OP_FRAME, planted vs swing costs.
_FOOT = {"l": ("left_ankle_roll_link", "left_toe", 22, 21),      # (ankle_frame, toe_frame, ballfoot_idx, ankle_idx)
         "r": ("right_ankle_roll_link", "right_toe", 18, 17)}
_FOOT_PLANTED_Z = 0.06      # m, ballfoot height gate (matches env ref_foot_planted_height)
_FOOT_PLANTED_VZ = 0.15     # m/s, ballfoot vertical-speed gate (matches env ref_foot_planted_velz)
_ANKLE_FLAT_Z = 0.03        # m, planted ankle_roll_link origin height (= -FOOT_TOE_OFFSET.z → toe at z≈0)
_FOOT_COST_PLANTED = (180.0, 80.0)   # (position[unused when anisotropic below], orientation) — foot FLAT
_FOOT_COST_SWING = (80.0, 3.0)       # swing foot tracks the human ankle, roughly level
_TOE_COST = 5.0
# ANISOTROPIC planted-foot (chosen via knob sweep, foot-slip-fix-sweep): pin z HARD (600) so the planted
# foot stays FLAT ON THE GROUND, while relaxing xy (10) so it shuffles to a reachable point instead of
# floating up at full leg extension. The G1 leg (~0.67 m) is too short to keep a foot fixed while the
# pelvis reaches the counter-height object → residual slip is irreducible; this halves the worst (toe)
# drag (34.7→20.6 cm) + z-lift (17.5→13.9) without hurting hand reach. None → isotropic _FOOT_COST_PLANTED[0].
_FOOT_PLANTED_XY_COST = 10.0
_FOOT_PLANTED_Z_COST = 600.0

_PELVIS_POS_COST = 8.0      # VERTICAL (z) pelvis-track weight
_PELVIS_ORI_COST = 8.0
_WRIST_ORI_COST = 12.0      # raised (8→12): tighter grasp orientation, less contact drift
_POSTURE_COST = 5e-2        # raised (2e-2→5e-2): stronger stance regularizer → curbs leg over-rotation
# reachability / foot-planting knobs (defaults reproduce prior behavior; the G1 leg is ~0.67 m so a
# planted foot cannot stay fixed while the free-flyer root chases the human COM 0.4 m horizontally →
# leg maxes out and the "planted" foot drags + lifts. These give the solver reach margin / de-chase).
_PELVIS_POS_COST_XY = 2.0   # HORIZONTAL (xy) pelvis-track weight (de-chased 8→2 via sweep): pelvis stays more over the feet vs chasing the human COM ±0.4 m → less foot drag
_PELVIS_LOWER = 0.0         # m subtracted from the pelvis target height (pre-lower → bent-knee reach margin)
_LEG_TRACK_SCALE = 1.0      # scale on hip/knee human-keypoint tracking cost (lower → feet+posture shape the legs)
_FOOT_TOE_OFFSET = [0.11, 0.0, -0.03]
_TORSO_KPT_OFFSET = [-0.033, 0.0, 0.274]

# ── per-frame velocity/acceleration limits (temporal) ──────────────────────────────────────────
# The IK loop iterates to convergence PER FRAME (batch Gauss-Newton), so pink's per-iteration
# Velocity/Acceleration limits regulate the SOLVER step, NOT inter-frame motion. To bound the actual
# frame-to-frame velocity + acceleration we solve the unconstrained target, then take a rate-limited
# step from the previous (limited) frame toward it, in the model tangent space (pin.difference /
# pin.integrate -> free-flyer-safe, no quaternion wrap). The limited state feeds the next warm-start,
# so the lag is a bounded, physical "cannot move faster than v_max" -- the correct rate-limiter shape.
_RATE_LIMIT = True
_V_MAX_JOINT = {"leg": 8.0, "waist": 6.0, "arm": 12.0}   # rad/s, body joint groups
_V_MAX_HAND = 20.0                                        # rad/s, Shadow fingers
_V_MAX_ROOT_LIN = 1.5                                     # m/s,  free-flyer translation
_V_MAX_ROOT_ANG = 3.0                                     # rad/s, free-flyer rotation
_ACCEL_RAMP_T = 0.10          # s to reach v_max from rest -> a_max = v_max / _ACCEL_RAMP_T (rad/s^2)

_KABSCH_SMOOTH_WIN = 3      # moving-average window on the human Kabsch input points (temporal)
_SAVGOL_WIN = 7             # post-hoc Savitzky–Golay window (odd)
_SAVGOL_POLY = 2

_HAND_TASKS = [
    (18, "ffknuckle", 1.0), (19, "ffproximal", 1.0), (20, "ffmiddle", 1.0), (21, "ff_pad", 3.0),
    (14, "mfknuckle", 1.0), (15, "mfproximal", 1.0), (16, "mfmiddle", 1.0), (17, "mf_pad", 3.0),
    (10, "rfknuckle", 1.0), (11, "rfproximal", 1.0), (12, "rfmiddle", 1.0), (13, "rf_pad", 3.0),
    (6, "lfknuckle", 1.0), (7, "lfproximal", 1.0), (8, "lfmiddle", 1.0), (9, "lf_pad", 3.0),
    (22, "thproximal", 1.0), (23, "thmiddle", 1.0), (24, "th_pad", 3.0),
]

_KABSCH_HUMAN = [0, 18, 14, 10, 6, 22]   # hand-block idx: wrist + index/middle/ring/pinky/thumb MCP
_KABSCH_SHADOW = ["ffknuckle", "mfknuckle", "rfknuckle", "lfknuckle", "thbase"]
_REF_FPS = 30.0


# ---- pinocchio free-flyer quaternion helpers (pin uses q[3:7]=xyzw) ----
def _mat_to_xyzw(R):
    q = pin.Quaternion(np.asarray(R, float))
    return np.array([q.x, q.y, q.z, q.w])


def _wxyz_from_xyzw(q):
    return np.array([q[3], q[0], q[1], q[2]])


def _rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ============================================================ model builders
def build_body_model():
    model = pin.buildModelFromUrdf(_G1_URDF, pin.JointModelFreeFlyer())   # FREE-FLYER root (nq=36)
    for side, opf in (("left", "left_toe"), ("right", "right_toe")):
        fid = model.getFrameId(f"{side}_ankle_roll_link")
        fr = model.frames[fid]
        model.addFrame(pin.Frame(opf, fr.parentJoint, fid,
                                 pin.SE3(np.eye(3), np.array(_FOOT_TOE_OFFSET)), pin.FrameType.OP_FRAME))
    tid = model.getFrameId("torso_link")
    tf = model.frames[tid]
    model.addFrame(pin.Frame("torso_kpt", tf.parentJoint, tid,
                             pin.SE3(np.eye(3), np.array(_TORSO_KPT_OFFSET)), pin.FrameType.OP_FRAME))
    return model, model.createData()


def build_hand_model(side: str):
    return build_shadow_from_usd(_USD_COMPOSITE, side, floating=False)


def _kabsch(src, dst):
    """Rigid SE3 mapping src(n,3) → dst(n,3) (least-squares). Returns pin.SE3(R, t): R·src+t ≈ dst."""
    sc, dc = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - sc).T @ (dst - dc))
    Rr = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return pin.SE3(Rr, dc - Rr @ sc)


def _shadow_ref_pts(hm, hd):
    pin.framesForwardKinematics(hm, hd, pin.neutral(hm))
    tp = hd.oMf[hm.getFrameId("palm")]
    return np.array([np.zeros(3)] + [tp.inverse().act(hd.oMf[hm.getFrameId(k)].translation)
                                     for k in _KABSCH_SHADOW])


# ============================================================ temporal helpers
def _movavg(arr, win):
    """Centered moving average along axis 0 (edge-clamped). arr: (F, ...)."""
    if win <= 1 or arr.shape[0] < 3:
        return arr
    k = win // 2
    pad = np.concatenate([np.repeat(arr[:1], k, 0), arr, np.repeat(arr[-1:], k, 0)], 0)
    ker = np.ones(win) / win
    out = np.empty_like(arr)
    flat = pad.reshape(pad.shape[0], -1)
    of = np.empty((arr.shape[0], flat.shape[1]), arr.dtype)
    for c in range(flat.shape[1]):
        of[:, c] = np.convolve(flat[:, c], ker, mode="valid")
    return of.reshape(arr.shape)


def _foot_plant_info(jp):
    """Per-side planted flag, foot yaw, and anti-slip hold-xy for the ankle, from RAW ParaHome
    ballfoot/ankle keypoints (matches the env foot-contact reference heuristic)."""
    F = jp.shape[0]
    info = {}
    for s, (_afr, _tfr, bidx, aidx) in _FOOT.items():
        ball = jp[:, bidx]; ankle = jp[:, aidx]
        z = ball[:, 2]
        vz = np.zeros(F); vz[1:] = (z[1:] - z[:-1]) * _REF_FPS
        planted = (z < _FOOT_PLANTED_Z) & (np.abs(vz) < _FOOT_PLANTED_VZ)
        d = _movavg(ball[:, :2] - ankle[:, :2], _KABSCH_SMOOTH_WIN)   # smooth the DIRECTION (wrap-safe)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        # anti-slip: freeze the ankle xy at the first frame of each contact interval
        hold = ankle[:, :2].copy()
        held = None
        for f in range(F):
            if planted[f]:
                if held is None:
                    held = ankle[f, :2].copy()
                hold[f] = held
            else:
                held = None
        info[s] = {"planted": planted, "yaw": yaw, "hold_xy": hold, "ankle": ankle}
    return info


def _savgol(out, root_out):
    """Zero-phase Savitzky–Golay smoothing of the 65 joint columns + root xyz; root quaternion is
    smoothed in the tangent space (log-map vs a running reference) then renormalized. Frame-0 is
    pinned unfiltered (it is the RSI seed). Returns (out, root_out) smoothed."""
    try:
        from scipy.signal import savgol_filter
    except Exception:  # noqa: BLE001
        return out, root_out
    F = out.shape[0]
    if F < _SAVGOL_WIN:
        return out, root_out
    o = out.copy(); r = root_out.copy()
    for c in range(o.shape[1]):
        o[:, c] = savgol_filter(out[:, c], _SAVGOL_WIN, _SAVGOL_POLY)
    for c in range(3):
        r[:, c] = savgol_filter(root_out[:, c], _SAVGOL_WIN, _SAVGOL_POLY)
    # quaternion: smooth in the INCREMENTAL (frame-to-frame) tangent space and reintegrate.
    # A fixed frame-0 reference (log3(q0^-1 * q_f)) is WRONG: pin.log3 has magnitude in [0, pi]
    # and wraps at the pi branch cut, so on a clip where the root turns >180 deg cumulatively the
    # tangent vector jumps discontinuously right where |q0^-1 q_f| crosses 180 deg, and Savgol then
    # smears a spurious ~110 deg/frame root swing across the wrap. The frame-to-frame increments
    # log3(R_{f-1}^T R_f) are each well under pi (≈ few deg here), so filtering them and
    # reintegrating from frame 0 is a proper zero-phase low-pass with no branch-cut artifact.
    q = root_out[:, 3:7].astype(np.float64)   # wxyz (float64 for pin.Quaternion)
    for f in range(1, F):
        if np.dot(q[f], q[f - 1]) < 0:        # adjacent double-cover sign-unwrap
            q[f] = -q[f]
    Rlist = [pin.Quaternion(float(q[f, 0]), float(q[f, 1]), float(q[f, 2]),
                            float(q[f, 3])).toRotationMatrix() for f in range(F)]
    inc = np.zeros((F, 3))                     # inc[f] = log(R_{f-1}^T R_f), |inc[f]| << pi
    for f in range(1, F):
        inc[f] = pin.log3(Rlist[f - 1].T @ Rlist[f])
    for c in range(3):
        inc[:, c] = savgol_filter(inc[:, c], _SAVGOL_WIN, _SAVGOL_POLY)
    Racc = Rlist[0].copy()
    for f in range(1, F):                      # reintegrate; r[0] quat stays the pinned frame-0 seed
        Racc = Racc @ pin.exp3(inc[f])
        qi = pin.Quaternion(Racc)
        qi.normalize()
        r[f, 3:7] = np.array([qi.w, qi.x, qi.y, qi.z])
    o[0] = out[0]; r[0] = root_out[0]   # pin frame-0 (RSI seed)
    return o, r


# ============================================================ IK solvers
def _solve(config, tasks, q_lo, q_hi, iters, dt=1.0 / 30.0):
    for _ in range(iters):
        try:
            v = solve_ik(config, tasks, dt, solver=_SOLVER, damping=1e-2, safety_break=False)
        except Exception:  # noqa: BLE001
            break
        config.integrate_inplace(v, dt)
        if np.linalg.norm(v) < 1e-3:
            break
    return np.clip(config.q.copy(), q_lo, q_hi)


def solve_body_ik(model, data, targets_world, pelvis_world, q_warm, iters, wrist_se3, foot_se3):
    """FREE-FLYER body IK in WORLD. targets_world: {frame:(3,) pos} (position-only body tasks).
    wrist_se3/foot_se3: {frame:(SE3, pos_cost, ori_cost)} full-pose tasks. Returns q(36)=[root7|body29]."""
    config = pink.Configuration(model, data, q_warm.copy())
    tasks = []
    pelvis_t = FrameTask("pelvis",
                         position_cost=[_PELVIS_POS_COST_XY, _PELVIS_POS_COST_XY, _PELVIS_POS_COST],
                         orientation_cost=_PELVIS_ORI_COST, lm_damping=1e-3)
    pt = pelvis_world.copy()
    if _PELVIS_LOWER:
        tt = pt.translation.copy(); tt[2] -= _PELVIS_LOWER; pt.translation = tt
    pelvis_t.set_target(pt)
    tasks.append(pelvis_t)
    for name, cost in _BODY_TASK_NAMES:
        c = cost * (_LEG_TRACK_SCALE if any(k in name for k in ("hip", "knee")) else 1.0)
        t = FrameTask(name, position_cost=c, orientation_cost=0.0, lm_damping=1e-3)
        t.set_target(pin.SE3(np.eye(3), np.asarray(targets_world[name])))
        tasks.append(t)
    for pose_map in (wrist_se3, foot_se3):
        for name, (se3, pc, oc) in pose_map.items():
            pcost = [float(x) for x in pc] if isinstance(pc, (list, tuple)) else float(pc)
            t = FrameTask(name, position_cost=pcost, orientation_cost=float(oc), lm_damping=1e-3)
            t.set_target(se3)
            tasks.append(t)
    posture = PostureTask(cost=_POSTURE_COST)
    posture.set_target(_BODY_POSTURE)
    tasks.append(posture)
    q_lo = model.lowerPositionLimit.copy(); q_hi = model.upperPositionLimit.copy()
    q_lo[:7] = -1e9; q_hi[:7] = 1e9
    return _solve(config, tasks, q_lo, q_hi, iters)


def solve_hand_ik(model, data, targets_palm, q_warm, iters):
    """Fixed-palm finger IK (warm-started from the previous frame's finger config)."""
    config = pink.Configuration(model, data, q_warm.copy())
    tasks = []
    for _i, name, cost in _HAND_TASKS:
        t = FrameTask(name, position_cost=cost, orientation_cost=0.0, lm_damping=1e-3)
        t.set_target(pin.SE3(np.eye(3), np.asarray(targets_palm[name])))
        tasks.append(t)
    posture = PostureTask(cost=1e-3)
    posture.set_target(pin.neutral(model))
    tasks.append(posture)
    return _solve(config, tasks, model.lowerPositionLimit.copy(), model.upperPositionLimit.copy(), iters)


def se3_from_mat(T):
    return pin.SE3(np.asarray(T[:3, :3], float), np.asarray(T[:3, 3], float))


def _quat_pos_to_mat(qp):
    w, x, y, z = qp["quat_wxyz"]
    R = pin.Quaternion(w, x, y, z).toRotationMatrix()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = qp["pos"]
    return T


def _foot_targets(finfo, f):
    """Per-frame foot full-pose tasks: planted → flat (Rz yaw, roll/pitch 0) at held xy + ANKLE_FLAT_Z,
    high cost; swing → human ankle, roughly level, lower cost. + toe position tasks (low cost)."""
    fse3 = {}
    for s, (afr, tfr, bidx, aidx) in _FOOT.items():
        d = finfo[s]
        R = _rz(float(d["yaw"][f]))
        if d["planted"][f]:
            pos = np.array([d["hold_xy"][f, 0], d["hold_xy"][f, 1], _ANKLE_FLAT_Z])
            pc = ([_FOOT_PLANTED_XY_COST, _FOOT_PLANTED_XY_COST, _FOOT_PLANTED_Z_COST]
                  if _FOOT_PLANTED_XY_COST is not None else _FOOT_COST_PLANTED[0])
            fse3[afr] = (pin.SE3(R, pos), pc, _FOOT_COST_PLANTED[1])
        else:
            fse3[afr] = (pin.SE3(R, d["ankle"][f]), _FOOT_COST_SWING[0], _FOOT_COST_SWING[1])
    return fse3


# ============================================================ rate limiting
def _body_rate_caps(model):
    """Per-tangent-DOF velocity + acceleration caps for the free-flyer body model.
    Tangent layout: [0:3]=root linear, [3:6]=root angular, [6:]=body joints (grouped by name)."""
    nv = model.nv
    vmax = np.full(nv, _V_MAX_JOINT["arm"])
    vmax[0:3] = _V_MAX_ROOT_LIN
    vmax[3:6] = _V_MAX_ROOT_ANG
    for j in range(1, model.njoints):
        if model.joints[j].nv != 1:          # skip the 6-DOF free-flyer root joint
            continue
        nm = model.names[j]; iv = model.joints[j].idx_v
        if any(k in nm for k in ("hip", "knee", "ankle")):
            vmax[iv] = _V_MAX_JOINT["leg"]
        elif "waist" in nm:
            vmax[iv] = _V_MAX_JOINT["waist"]
        elif any(k in nm for k in ("shoulder", "elbow", "wrist")):
            vmax[iv] = _V_MAX_JOINT["arm"]
    return vmax, vmax / _ACCEL_RAMP_T


def _hand_rate_caps(model):
    n = model.nv
    return np.full(n, _V_MAX_HAND), np.full(n, _V_MAX_HAND / _ACCEL_RAMP_T)


def _rate_limit(delta, delta_prev, vmax, amax, dt, root=False):
    """Clamp a tangent step: acceleration (|d - d_prev| <= a*dt^2) THEN velocity (|d| <= v*dt).
    root=True -> norm-clamp the [0:3] linear and [3:6] angular blocks (isotropic), per-DOF the rest."""
    d = delta.copy()
    dv = d - delta_prev
    if root:
        for sl in (slice(0, 3), slice(3, 6)):
            lim = amax[sl.start] * dt * dt; n = float(np.linalg.norm(dv[sl]))
            if n > lim:
                dv[sl] *= lim / n
        lim = amax[6:] * dt * dt; dv[6:] = np.clip(dv[6:], -lim, lim)
    else:
        lim = amax * dt * dt; dv = np.clip(dv, -lim, lim)
    d = delta_prev + dv
    if root:
        for sl in (slice(0, 3), slice(3, 6)):
            lim = vmax[sl.start] * dt; n = float(np.linalg.norm(d[sl]))
            if n > lim:
                d[sl] *= lim / n
        lim = vmax[6:] * dt; d[6:] = np.clip(d[6:], -lim, lim)
    else:
        lim = vmax * dt; d = np.clip(d, -lim, lim)
    return d


# ============================================================ direct body scaling (GMR/OmniRetarget)
# Scale the human SMPL-X body keypoints toward the ground point under the pelvis by a single global
# α = h_robot / h_human (robot-to-human height ratio) so the (shorter) robot body/legs become
# reachable-by-construction — while the WRISTS (idx 10,14) + HAND blocks + fingertip pads + object
# stay at their ORIGINAL world positions so object contact and wrist/hand pose are UNCHANGED.
# Feet stay grounded (z→α·z, and standing feet are at z≈0) and the pelvis lowers to robot height.
# Fully rollback-able: OFF unless --scale_body (scale_alpha=0 → identity, no-op).
_SCALE_KEEP_IDX = (10, 14)   # jRightWrist, jLeftWrist — NOT scaled (kept at object with the hands)


def _robot_standing_pelvis_height():
    """G1 pelvis→ground height at the standing posture _BODY_POSTURE (free-flyer root at origin)."""
    m, dat = _BODY_MODEL
    q = _BODY_POSTURE.copy()
    q[:3] = 0.0; q[3:7] = [0.0, 0.0, 0.0, 1.0]               # root at world origin, identity
    pin.framesForwardKinematics(m, dat, q)
    az = min(float(dat.oMf[m.getFrameId("left_ankle_roll_link")].translation[2]),
             float(dat.oMf[m.getFrameId("right_ankle_roll_link")].translation[2]))
    return -az + (-_FOOT_TOE_OFFSET[2])                      # ankle below pelvis + sole offset (0.03)


def _scale_body(jp, Tg, alpha):
    """Scale body keypoints [0:23] EXCEPT the wrists toward the ground point under the pelvis by
    alpha. Hand blocks [23:73], wrists, and fingertip pads are left ORIGINAL (contact-preserving).
    Tg (pelvis root SE3) translation is scaled the same way (rotation kept). Returns (jp, Tg)."""
    jp = jp.copy(); Tg = Tg.copy()
    body_idx = [i for i in range(23) if i not in _SCALE_KEEP_IDX]
    for f in range(jp.shape[0]):
        o = np.array([float(Tg[f, 0, 3]), float(Tg[f, 1, 3]), 0.0])   # ground point under the pelvis
        jp[f, body_idx] = o + alpha * (jp[f, body_idx] - o)
        Tg[f, :3, 3] = o + alpha * (Tg[f, :3, 3] - o)
    return jp, Tg


# ============================================================ per-clip
def retarget_clip(clip_dir: Path, order, frame0_only: bool, smooth: bool,
                  scale_body: bool = False, scale_alpha: float = 0.0):
    d = np.load(clip_dir / "trajectory.npz", allow_pickle=True)
    jp = d["joint_positions"].astype(np.float64)          # (F,73,3) world
    Tg = d["body_global_transform"].astype(np.float64)    # (F,4,4) pelvis world SE3
    ftpad = d["fingertip_pad_pos"].astype(np.float64)     # (F,10,3)
    alpha_used = 0.0
    if scale_body:                                        # DIRECT BODY SCALING (rollback: --scale_body off)
        h_hum = float(np.median(Tg[:, 2, 3]))             # human pelvis height for this clip (robust)
        alpha_used = scale_alpha if scale_alpha > 0.0 else (_robot_standing_pelvis_height() / max(h_hum, 1e-6))
        jp, Tg = _scale_body(jp, Tg, alpha_used)
    F = jp.shape[0]
    frames = [0] if frame0_only else list(range(F))

    bmodel, bdata = _BODY_MODEL
    hmodels = _HAND_MODELS
    action_names = order["action_joint_names"]
    w2p = {s: se3_from_mat(_quat_pos_to_mat(order["wrist_to_palm"][s])) for s in ("r", "l")}

    finfo = _foot_plant_info(jp)
    # temporally smooth the human Kabsch input points (wrist + 5 MCPs) per side (removes wrist jitter)
    kabsch_pts = {}
    for s in ("l", "r"):
        blk = 23 if s == "l" else 48
        pts = np.stack([jp[:, blk + i] for i in _KABSCH_HUMAN], axis=1)   # (F,6,3)
        kabsch_pts[s] = _movavg(pts, _KABSCH_SMOOTH_WIN)

    out = np.zeros((F, 65), np.float32)
    root_out = np.zeros((F, 7), np.float32); root_out[:, 3] = 1.0
    # reference PALM (wrist) world orientation per hand [L,R], wxyz — the Kabsch palm pose IS the
    # robot0_{s}_palm body frame (fit to the human hand landmarks), so the env can compare it
    # DIRECTLY to body_quat_w[palm] for a wrist-rotation termination gate (no landmark conversion).
    palm_quat_out = np.zeros((F, 2, 4), np.float32); palm_quat_out[:, :, 0] = 1.0
    bwarm = _BODY_POSTURE.copy()
    qh_warm = {s: pin.neutral(hmodels[s][0]) for s in ("l", "r")}
    ft_slot = {"l": 0, "r": 5}
    ft_finger = {"th": 0, "ff": 1, "mf": 2, "rf": 3, "lf": 4}
    # rate-limit state: previous LIMITED config + previous tangent step, per model
    dt = 1.0 / _REF_FPS
    qb_prev = None; db_prev = np.zeros(bmodel.nv)
    qh_prev = {"l": None, "r": None}; dh_prev = {s: np.zeros(hmodels[s][0].nv) for s in ("l", "r")}
    if _RATE_LIMIT:
        vcap_b, acap_b = _body_rate_caps(bmodel)
        hcaps = {s: _hand_rate_caps(hmodels[s][0]) for s in ("l", "r")}

    for f in frames:
        pelvis = se3_from_mat(Tg[f])
        if f == frames[0]:
            bwarm[:3] = pelvis.translation
            bwarm[3:7] = _mat_to_xyzw(pelvis.rotation)

        # STAGE 1 — Kabsch palm pose (smoothed input) + warm-started finger IK.
        wrist_se3 = {}
        hand_dict = {}
        for s, wl in (("l", "left_wrist_yaw_link"), ("r", "right_wrist_yaw_link")):
            hm, hd = hmodels[s]
            blk = 23 if s == "l" else 48
            palm_pose = _kabsch(_SHADOW_REF[s], kabsch_pts[s][f])
            palm_quat_out[f, 0 if s == "l" else 1] = _wxyz_from_xyzw(_mat_to_xyzw(palm_pose.rotation))
            wrist_se3[wl] = (palm_pose * w2p[s].inverse(), 40.0, _WRIST_ORI_COST)
            palm_inv = palm_pose.inverse()
            htargets = {}
            for pidx, fname, _c in _HAND_TASKS:
                pw = (ftpad[f, ft_slot[s] + ft_finger[fname[:2]]] if fname.endswith("_pad")
                      else jp[f, blk + pidx])
                htargets[fname] = palm_inv.act(pw)
            qh = solve_hand_ik(hm, hd, htargets, qh_warm[s], iters=(200 if f == frames[0] else 40))
            if _RATE_LIMIT and f != frames[0] and qh_prev[s] is not None:
                vc, ac = hcaps[s]
                step = _rate_limit(pin.difference(hm, qh_prev[s], qh), dh_prev[s], vc, ac, dt)
                qh = pin.integrate(hm, qh_prev[s], step); dh_prev[s] = step
            qh_prev[s] = qh
            qh_warm[s] = qh
            for env_j in _ENV_HAND_JOINTS[s]:
                mj = f"robot0_{s}_{env_j}"
                if hm.existJointName(mj):
                    hand_dict[mj] = float(qh[hm.joints[hm.getJointId(mj)].idx_q])

        # STAGE 2 — body IK: reach/orient the wrists, plant the feet FLAT, regularize the stance.
        btargets = {name: jp[f, pidx] for pidx, name, _c in _BODY_TASKS}
        foot_se3 = _foot_targets(finfo, f)
        qb = solve_body_ik(bmodel, bdata, btargets, pelvis, bwarm,
                           iters=(250 if f == frames[0] else 40), wrist_se3=wrist_se3, foot_se3=foot_se3)
        if _RATE_LIMIT and f != frames[0] and qb_prev is not None:
            step = _rate_limit(pin.difference(bmodel, qb_prev, qb), db_prev, vcap_b, acap_b, dt, root=True)
            qb = pin.integrate(bmodel, qb_prev, step); db_prev = step
        qb_prev = qb
        bwarm = qb
        body_dict = {bmodel.names[j]: float(qb[bmodel.joints[j].idx_q]) for j in range(1, bmodel.njoints)}
        root_out[f, :3] = qb[:3]
        root_out[f, 3:7] = _wxyz_from_xyzw(qb[3:7])

        merged = {**body_dict, **hand_dict}
        out[f] = np.array([merged.get(n, 0.0) for n in action_names], np.float32)
        if f == frames[0]:
            miss = [n for n in action_names if n not in merged]
            if miss:
                print(f"  [warn] {len(miss)} action joints unmapped: {miss[:8]}")

    jitter_raw = _jitter(out) if not frame0_only else 0.0
    if not frame0_only and smooth:
        # Savgol values stay ≈within the window's convex hull (already-clipped inputs); the env
        # re-clamps joint targets to limits at apply time, so no explicit re-clip needed here.
        out, root_out = _savgol(out, root_out)
    if frame0_only:
        out[:] = out[0]; root_out[:] = root_out[0]
    return d, out, root_out, {"jitter_raw": jitter_raw, "jitter": _jitter(out) if not frame0_only else 0.0,
                              "finfo": finfo, "palm_quat": palm_quat_out, "scale_alpha": alpha_used}


def _jitter(out):
    """Mean per-frame ‖Δq‖ over the 65 joints (temporal-consistency metric)."""
    if out.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(out, axis=0), axis=1).mean())


# ============================================================ main / globals
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_class", default="single_rigid")
    ap.add_argument("--clip", default="")
    ap.add_argument("--frame0_only", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_smooth", action="store_true")
    ap.add_argument("--solver", default="daqp")
    ap.add_argument("--scale_body", action="store_true",
                    help="direct-scale SMPL-X body keypoints (except wrists+hands) to robot height "
                         "(GMR/OmniRetarget) so legs/body are reachable; object contact + wrist/hand kept")
    ap.add_argument("--scale_alpha", type=float, default=0.0,
                    help="manual override for the body-scale ratio α (0 = auto = h_robot/h_human)")
    args = ap.parse_args()

    global _SOLVER, _BODY_MODEL, _HAND_MODELS, _BODY_TASK_NAMES, _BODY_POSTURE, _ENV_HAND_JOINTS
    global _SHADOW_REF
    _SOLVER = args.solver
    order = json.load(open(_JOINT_ORDER))
    _BODY_MODEL = build_body_model()
    _HAND_MODELS = {"r": build_hand_model("r"), "l": build_hand_model("l")}
    _SHADOW_REF = {s: _shadow_ref_pts(*_HAND_MODELS[s]) for s in ("r", "l")}
    _BODY_TASK_NAMES = [(name, cost) for (_i, name, cost) in _BODY_TASKS]
    bm = _BODY_MODEL[0]
    _BODY_POSTURE = pin.neutral(bm)
    for jn, val in (("left_hip_pitch_joint", -0.10), ("right_hip_pitch_joint", -0.10),
                    ("left_knee_joint", 0.30), ("right_knee_joint", 0.30),
                    ("left_ankle_pitch_joint", -0.20), ("right_ankle_pitch_joint", -0.20)):
        _BODY_POSTURE[bm.joints[bm.getJointId(jn)].idx_q] = val
    _ENV_HAND_JOINTS = {"r": [], "l": []}
    for n in order["action_joint_names"]:
        if n.startswith("robot0_r_"):
            _ENV_HAND_JOINTS["r"].append(n.split("robot0_r_")[1])
        elif n.startswith("robot0_l_"):
            _ENV_HAND_JOINTS["l"].append(n.split("robot0_l_")[1])
    print(f"[setup] body nq={bm.nq} (free-flyer)  hand nq r={_HAND_MODELS['r'][0].nq}"
          f"  env finger/side={len(_ENV_HAND_JOINTS['r'])}  smooth={not args.no_smooth}")

    smplx_root = _PARAHOME / "smplx" / args.clip_class
    clips = ([args.clip] if args.clip and args.clip != "ALL"
             else sorted(p.name for p in smplx_root.iterdir() if p.is_dir()))
    if args.clip != "ALL" and not args.clip:
        clips = clips[:1]

    n_ok = n_fail = 0
    for ci, clip in enumerate(clips):
        clip_dir = smplx_root / clip / "0"
        out_dir = _PARAHOME / "g1_shadow" / args.clip_class / clip / "0"
        out_npz = out_dir / "trajectory.npz"
        if out_npz.exists() and not args.overwrite and args.clip != clip:
            print(f"[{ci + 1}/{len(clips)}] {clip}: exists (skip; --overwrite to redo)"); n_ok += 1; continue
        try:
            d, out, root_out, meta = retarget_clip(clip_dir, order, args.frame0_only, not args.no_smooth,
                                                   scale_body=args.scale_body, scale_alpha=args.scale_alpha)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(out_npz, g1_joint_pos=out, g1_root_pose=root_out, g1_palm_quat=meta["palm_quat"],
                     scale_alpha=np.float32(meta["scale_alpha"]))
            if args.scale_body:
                print(f"    [scale_body] α={meta['scale_alpha']:.3f} (body keypoints scaled to robot height; wrists+hands kept)")
            print(f"[{ci + 1}/{len(clips)}] {clip}: g1_joint_pos {out.shape} + g1_root_pose {root_out.shape}"
                  f"  jitter(raw→smooth)={meta['jitter_raw']:.4f}→{meta['jitter']:.4f}")
            if args.check:
                _fk_check(clip_dir, out, root_out, order, meta)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[{ci + 1}/{len(clips)}] {clip}: FAILED — {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"[done] ok={n_ok} fail={n_fail} / {len(clips)}")


def _fk_check(clip_dir, out, root_out, order, meta):
    d = np.load(clip_dir / "trajectory.npz", allow_pickle=True)
    jp = d["joint_positions"].astype(np.float64)
    bm, bd = _BODY_MODEL
    anames = order["action_joint_names"]
    q = pin.neutral(bm)
    q[:3] = root_out[0, :3]
    q[3:7] = np.array([root_out[0, 4], root_out[0, 5], root_out[0, 6], root_out[0, 3]])
    for j in range(1, bm.njoints):
        nm = bm.names[j]
        if nm in anames:
            q[bm.joints[j].idx_q] = out[0, anames.index(nm)]
    pin.forwardKinematics(bm, bd, q); pin.updateFramePlacements(bm, bd)
    errs = {}
    for pidx, name, _c in _BODY_TASKS:
        w = bd.oMf[bm.getFrameId(name)].translation
        errs[name] = float(np.linalg.norm(w - jp[0, pidx]))
    # feet: FK ankle + toe, report position err vs human + flatness (sole tilt from vertical)
    foot_msg = []
    for s, (afr, tfr, bidx, aidx) in _FOOT.items():
        ap_ = bd.oMf[bm.getFrameId(afr)]
        ez = ap_.rotation[:, 2]   # foot local +Z in world (sole normal ≈ +Z when flat)
        tilt = np.degrees(np.arccos(np.clip(ez[2], -1, 1)))
        ankle_err = float(np.linalg.norm(ap_.translation - jp[0, aidx]))
        pl = meta["finfo"][s]["planted"][0]
        foot_msg.append(f"{s}:{'P' if pl else 'S'} ankle_z={ap_.translation[2]:.3f} err={ankle_err:.3f} tilt={tilt:.0f}deg")
    wr = np.mean([errs["left_wrist_yaw_link"], errs["right_wrist_yaw_link"]])
    print(f"  [check] frame-0 FK: mean={np.mean(list(errs.values())):.3f} wrists={wr:.3f} "
          f"torso={errs['torso_kpt']:.3f} elbows={np.mean([errs['left_elbow_link'], errs['right_elbow_link']]):.3f} m")
    print(f"  [check] feet: {' | '.join(foot_msg)}  (planted→ankle_z≈0.03, tilt≈0)")
    ftpad = d["fingertip_pad_pos"].astype(np.float64)
    w2p = {s: se3_from_mat(_quat_pos_to_mat(order["wrist_to_palm"][s])) for s in ("r", "l")}
    ft_slot = {"l": 0, "r": 5}; ft_finger = {"th": 0, "ff": 1, "mf": 2, "rf": 3, "lf": 4}
    for s in ("r", "l"):
        wy = bm.getFrameId(f"{'right' if s == 'r' else 'left'}_wrist_yaw_link")
        palm_world = bd.oMf[wy] * w2p[s]
        hm, hd = _HAND_MODELS[s]
        qh = pin.neutral(hm)
        for env_j in _ENV_HAND_JOINTS[s]:
            mj = f"robot0_{s}_{env_j}"
            if hm.existJointName(mj) and mj in anames:
                qh[hm.joints[hm.getJointId(mj)].idx_q] = out[0, anames.index(mj)]
        pin.framesForwardKinematics(hm, hd, qh)
        fte = np.mean([np.linalg.norm(palm_world.act(hd.oMf[hm.getFrameId(f"{fk}_pad")].translation)
                                      - ftpad[0, ft_slot[s] + ft_finger[fk]]) for fk in ft_finger])
        print(f"  [check hand {s}] fingertip world err (env ft): mean={fte:.3f} m (term_ft_err=0.20)")


if __name__ == "__main__":
    main()
