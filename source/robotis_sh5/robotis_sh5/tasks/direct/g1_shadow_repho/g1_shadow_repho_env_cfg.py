"""Configuration for the G1 + Shadow-hand full-body loco-manipulation task.

Unitree G1 (29-DOF body: 12 legs + 3 waist + 14 arms) with its Dex3 3-finger hands
REPLACED by two Shadow Dexterous Hands (18 actuated DOF each) → 65 actuated DOF total,
FLOATING base. The policy imitates retargeted ParaHome SMPL-X whole-body motion
(walking + bimanual object manipulation).

This EXTENDS the fixed-base single-hand grasp task (robotis_shadow_grasp_rsi) to full
body + bimanual + locomotion. Reused mechanisms: adaptive frame-sampling curriculum,
pretrain→train + pretrain-cache RSI warm-start, contact-conditioned fingertip force,
per-group action EMA smoothing. New: root/locomotion tracking, balance + foot contact,
fall (reference-deviation) termination, full-body proprioception.

ASSET STATUS — the composite G1+Shadow USD is BUILT and validated:
  * data/robots/G1/G1_shadow.usd — G1_29DOF flattened, Dex3 stripped, bimanual Shadow
    grafted (right reused + left geometric-mirror), mounted at the wrists. Spawn-tested:
    78 bodies / 73 DOF, actuators cover 65 (8 J0 coupling joints uncoupled by design),
    assembled + bounded under gravity (scripts/process_dataset/diagnostics/spawn_test_g1_shadow.py).
  * G1_SHADOW_CFG below is the validated ArticulationCfg (G1_29DOF body actuators +
    Shadow finger gains). Instancing was assessed and found unnecessary (G1 geometry is
    flatten-instanced; Shadow collision is primitive Cubes → no per-env cook cost).
  * body-link / fingertip-body names are CONFIRMED against the composite USD.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

_DATA_DIR = Path(__file__).resolve().parents[4] / "data"
_ROBOT_USD = str(_DATA_DIR / "robots" / "G1" / "G1_shadow.usd")

# =============================================================================== #
# Keypoint-tracking skeleton (design spec → constants)
# =============================================================================== #
# ParaHome joint_positions layout: [0:23]=body, [23:48]=left hand(25), [48:73]=right hand(25).
# (parahome.py stores the raw (F,73,3) array; blocks split by this layout.)
# Hand-25 order (per hand), VERIFIED against ParaHome visualize/utils.py hand-order dicts:
#   0=Wrist, 1-5=CMC/palm-knuckle row (thumb..pinky — proximal to the finger MCPs, distinct
#   joints), then a 4-point chain per finger MCP/PIP/DIP/Tip in REVERSE finger order:
#   pinky:6-9, ring:10-13, middle:14-17, index:18-21, thumb:22=MCP,23=IP,24=Tip (thumb has 3).

# --- Body: 16 of the 23 ParaHome body joints → G1 body links ------------------------
# The 23-joint ParaHome body skeleton (Xsens-MVN-style rig; VERIFIED against ParaHome
# visualize/utils.py `body_order`, bone_vectors.pkl hierarchy, and frame-0 geometry):
#   0 pHipOrigin* 1 jL5S1  2 jL4L3  3 jL1T12  4 jT9T8*  5 jT1C7  6 jC1Head
#   7 jRightT4Shoulder  8 jRightShoulder* 9 jRightElbow* 10 jRightWrist*
#   11 jLeftT4Shoulder  12 jLeftShoulder* 13 jLeftElbow* 14 jLeftWrist*
#   15 jRightHip* 16 jRightKnee* 17 jRightAnkle* 18 jRightBallFoot*
#   19 jLeftHip*  20 jLeftKnee*  21 jLeftAnkle*  22 jLeftBallFoot*     (* = tracked below)
# UNUSED (7): {1,2,3}=lumbar/thoracic spine segments, 5=jT1C7 (base of neck), 6=jC1Head,
#   {7,11}=T4-anchored clavicle/scapula. None is an INDEPENDENTLY-ARTICULATED G1 body:
#   G1_29DOF's trunk is only the 3 waist joints → torso_link (arms mount at torso_link), and
#   the HEAD DOES EXIST but is RIGIDLY FUSED into torso_link (head_link + d435/mid360 geometry
#   at torso-frame z≈+0.38 m; NO neck joint) — so tracking jC1Head would only re-track the torso
#   AND would wrongly force the torso to chase the human's independent (neck) head motion (common
#   head-down gaze during manipulation). Spine/clavicle likewise have no independent body. Dropped
#   — NOT forced (user-confirmed). pelvis(0)+torso_link(4)+shoulders(8,12)+hips(15,19) already
#   bound trunk pose/orientation. All 16 targets below exist in G1_shadow.usd (CONFIRMED).
BODY_KPTS: dict[int, str] = {
    0:  "pelvis",                     # pHipOrigin (root)
    4:  "torso_link",                 # jT9T8 (upper-trunk / shoulder-girdle anchor)
    8:  "right_shoulder_pitch_link",  # jRightShoulder (arm-root body)
    9:  "right_elbow_link",           # jRightElbow
    10: "right_wrist_yaw_link",       # jRightWrist (distal forearm; arm mount)
    12: "left_shoulder_pitch_link",   # jLeftShoulder (arm-root body)
    13: "left_elbow_link",            # jLeftElbow
    14: "left_wrist_yaw_link",        # jLeftWrist (distal forearm; arm mount)
    15: "right_hip_pitch_link",       # jRightHip (leg-root body)
    16: "right_knee_link",            # jRightKnee
    17: "right_ankle_roll_link",      # jRightAnkle (foot; 1 pt/foot, GRAIL-style)
    19: "left_hip_pitch_link",        # jLeftHip (leg-root body)
    20: "left_knee_link",             # jLeftKnee
    21: "left_ankle_roll_link",       # jLeftAnkle (foot; 1 pt/foot, GRAIL-style)
}
# BODY = 14 links (pelvis, torso, shoulders/elbows/wrists ×2, hips/knees/ankles ×2) — matches GRAIL's
# 14 tracked bodies. The ballfoot toe points (jRightBallFoot 18 / jLeftBallFoot 22) were DROPPED
# (user 2026-07-20, GRAIL-aligned) — feet are now 1 pt/foot (ankle_roll), same as GRAIL; the foot pitch
# they used to add is left to the frozen SONIC base (which owns feet/balance).

# Per-body-keypoint LOCAL offset (in the target G1 link frame), added (rotated) to the body origin
# before tracking. (The FOOT_TOE_OFFSET for the ballfoot keypoints was removed with those keypoints
# 2026-07-20.) jT9T8 (upper spine, idx 4) ≠ torso_link ORIGIN (at the waist, ~0.27 m lower). Track a point on
# G1's UPPER torso (torso_link + this local offset ≈ upper chest) so the correspondence is
# anatomically aligned instead of ~0.27 m off. Measured: jT9T8 in torso_link local frame at a
# standing posture. (env robot-kpt = torso_link origin + offset; retarget targets the same frame.)
TORSO_KPT_OFFSET: list[float] = [-0.033, 0.0, 0.274]
BODY_KPT_OFFSETS: dict[int, list[float]] = {
    4: TORSO_KPT_OFFSET,
}

# --- Hands: 20 keypoints per hand = wrist(1) + 4 fingers×4 + thumb×3. ------------------
# ParaHome hand-local idx (0..24) → Shadow body. Applied to BOTH hands; the left-hand block
# is offset by +23 in joint_positions, right by +48 (block-offset machinery handles this).
# Shadow bodies get the robot0_{l,r}_ prefix at build time. Fingertip (Tip) tracks the pad
# VERTEX (from preprocessing) at the distal body + offset. The WRIST anchors the hand base:
# ParaHome hand-local 0 → Shadow palm (robot0_{l,r}_palm), matching the single-hand grasp
# env's MANO-wrist→palm convention (palm is the hand-root body). It is a body-origin keypoint
# (no pad offset/normal). Complementary to body jRightWrist(10)/jLeftWrist(14)→wrist_yaw_link.
HAND_CHAIN: dict[str, dict] = {
    #             [ParaHome idx]              [Shadow body(ies)]
    "wrist":  {"parahome": [0],              "shadow": ["palm"]},
    #             [MCP, PIP, DIP, Tip]        [knuckle,   proximal,   middle,   distal(+offset)]
    "index":  {"parahome": [18, 19, 20, 21], "shadow": ["ffknuckle", "ffproximal", "ffmiddle", "ffdistal"]},
    "middle": {"parahome": [14, 15, 16, 17], "shadow": ["mfknuckle", "mfproximal", "mfmiddle", "mfdistal"]},
    "ring":   {"parahome": [10, 11, 12, 13], "shadow": ["rfknuckle", "rfproximal", "rfmiddle", "rfdistal"]},
    "pinky":  {"parahome": [6, 7, 8, 9],     "shadow": ["lfknuckle", "lfproximal", "lfmiddle", "lfdistal"]},
    "thumb":  {"parahome": [22, 23, 24],     "shadow": ["thproximal", "thmiddle", "thdistal"]},  # CMC/MCP→palm
}

# Robot-side fingertip = distal body origin + LOCAL offset (rotated), pad faces LOCAL normal.
# Base values (right hand) reused verbatim from the grasp env (Shadow-distal local frame).
# LEFT-hand fingers need the Y component NEGATED: the left Shadow was built by a geometric
# Y-reflection (mesh points Y-negated), so the pad surface moved from local −Y to local +Y.
# VERIFIED on the composite USD (scripts, rest pose): right finger pads → world +Z; left with
# the SAME local normal → world −Z (wrong; the XZ-mirror preserves Z), left with Y-flipped
# normal → world +Z (correct). The THUMB normal/offset have Y=0, so they are mirror-invariant
# and identical on both hands. Keyed by FULL body name (robot0_{l,r}_<body>).
_FT_OFFSET_BASE: dict[str, list[float]] = {
    "thdistal": [-0.0085, 0.0, 0.02],
    "ffdistal": [0.0, -0.006, 0.0175],
    "mfdistal": [0.0, -0.006, 0.0175],
    "rfdistal": [0.0, -0.006, 0.0175],
    "lfdistal": [0.0, -0.006, 0.0175],
}
_FT_NORMAL_BASE: dict[str, list[float]] = {
    "thdistal": [-1.0, 0.0, 0.0],
    "ffdistal": [0.0, -1.0, 0.0],
    "mfdistal": [0.0, -1.0, 0.0],
    "rfdistal": [0.0, -1.0, 0.0],
    "lfdistal": [0.0, -1.0, 0.0],
}


def _mirror_y(v: list[float]) -> list[float]:
    return [v[0], -v[1], v[2]]


def _build_ft(base: dict[str, list[float]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for body, v in base.items():
        out[f"robot0_r_{body}"] = list(v)
        # thumb (Y=0) is mirror-invariant; the four fingers flip Y for the mirrored left hand.
        out[f"robot0_l_{body}"] = list(v) if body == "thdistal" else _mirror_y(v)
    return out


FINGERTIP_OFFSETS: dict[str, list[float]] = _build_ft(_FT_OFFSET_BASE)
FINGERTIP_PAD_NORMALS: dict[str, list[float]] = _build_ft(_FT_NORMAL_BASE)

# Number of tracked keypoints (obs/reward): body 14 (BODY_KPTS entries, GRAIL-aligned) + hands
# (wrist=1, index/mid/ring/pinky=4, thumb=3 → 20 per hand) × 2 = 54.
N_BODY_KPTS = len(BODY_KPTS)                      # 14
N_HAND_KPTS_PER_HAND = sum(len(v["parahome"]) for v in HAND_CHAIN.values())  # 20
N_TRACK_KPTS = N_BODY_KPTS + 2 * N_HAND_KPTS_PER_HAND                          # 54

# --- Per-link CONTACT set for the Option-A per-link contact-FORCE reward ------------------
# The wrap links whose object contact is precomputed in hand_contact.npz (parahome_hand_contact.py) and
# rewarded per-link (force_matrix · reaction-normal, gated by the reference contact mask). Order MUST match
# parahome_hand_contact.py `link_names`: per hand = palm + [proximal,middle,distal]×[ff,mf,lf,rf,th]; LEFT
# block then RIGHT block. Each link gets its OWN ContactSensor (object-filtered).
_LINK_CONTACT_FINGERS = ["ff", "mf", "lf", "rf", "th"]
_LINK_CONTACT_SEGS = ["proximal", "middle", "distal"]
LINK_CONTACT_NAMES: list[str] = [
    f"robot0_{s}_{b}"
    for s in ("l", "r")
    for b in (["palm"] + [f"{fg}{seg}" for fg in _LINK_CONTACT_FINGERS for seg in _LINK_CONTACT_SEGS])
]
N_LINK_CONTACT = len(LINK_CONTACT_NAMES)          # 32 (2 hands × (1 palm + 5 fingers × 3 segs))

# Per-link OUTWARD pad/palmar normal (link-LOCAL), one per wrap link — the "grasping face" of each link,
# used by the per-link contact-force reward: force is projected on the link's OWN inward normal (like the
# fingertip pad, `force·(-pad_normal)`) instead of the object-anchored reference reaction dir, so touching
# with the WRONG face (e.g. back of the palm) yields no compressive force; and an ORIENTATION GATE requires
# the link's inward normal to align with the reference reaction normal within `contact_normal_gate_tol`.
# Right-hand base; the left hand is a geometric Y-reflection so the four fingers + palm flip local Y (thumb
# has Y=0 → mirror-invariant, identical on both hands). VERIFIED on G1_shadow.usd rest pose (2026-07-22):
# every non-thumb face → world +Z (dot>0.7, coplanar open-hand palmar), every thumb face ≤35° from the
# trusted TJ thumb-distal normal [-1,0,0] (thumb opposes the fingers → not +Z). Keyed by short body name.
_LINK_NORMAL_BASE: dict[str, list[float]] = {
    "palm": [0.0, -1.0, 0.0],
    **{f"{fg}{seg}": [0.0, -1.0, 0.0] for fg in ("ff", "mf", "lf", "rf") for seg in _LINK_CONTACT_SEGS},
    **{f"th{seg}": [-1.0, 0.0, 0.0] for seg in _LINK_CONTACT_SEGS},
}


def _build_link_normals(base: dict[str, list[float]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for body, v in base.items():
        out[f"robot0_r_{body}"] = list(v)
        out[f"robot0_l_{body}"] = _mirror_y(v)          # thumb has Y=0 → mirror is identity
    return out


LINK_PAD_NORMALS: dict[str, list[float]] = _build_link_normals(_LINK_NORMAL_BASE)  # 32, full body-name keys


# =============================================================================== #
# Joint groups (action + control), by G1/Shadow joint-name regex
# =============================================================================== #
# G1_29DOF joint naming (isaaclab_assets/robots/unitree.py) + Shadow robot0_* naming.
JOINT_GROUPS: dict[str, dict] = {
    "legs":  {"expr": [".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
                       ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
              "dof": 12, "ema_alpha": 0.3},
    "waist": {"expr": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
              "dof": 3, "ema_alpha": 0.3},
    "arms":  {"expr": [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                       ".*_elbow_joint", ".*_wrist_pitch_joint", ".*_wrist_roll_joint", ".*_wrist_yaw_joint"],
              "dof": 14, "ema_alpha": 0.3},
    # Shadow 18 actuated per hand (J1/J2/J3 ×5 + LFJ4 + THJ4 + THJ0); J0 tendon-coupled.
    "hands": {"expr": ["robot0_(l|r)_(FF|MF|RF|LF|TH)J[1-3]",
                       "robot0_(l|r)_LFJ4", "robot0_(l|r)_THJ4", "robot0_(l|r)_THJ0"],
              "dof": 36, "ema_alpha": 0.5},
}
ACTION_DIM = sum(g["dof"] for g in JOINT_GROUPS.values())   # 65


# =============================================================================== #
# Robot: composite G1 (29-DOF body) + bimanual Shadow (18 actuated DOF/hand) = 65 DOF.
# =============================================================================== #
# FLOATING base (fix_root_link=False) with real gravity — the policy must physically
# balance while tracking the retargeted whole-body trajectory. Actuators reproduce the
# proven layouts: G1_29DOF_CFG (legs/feet DCMotor, waist/arms implicit) for the body and
# the robotis_shadow_grasp Shadow finger PD (stiffness 1.0 / damping 0.2 / effort 3.09)
# for both hands. The 8 J0 coupling joints (FFJ0/MFJ0/RFJ0/LFJ0 ×2) are intentionally left
# to their USD drive (as in the single-hand task). Validated by spawn_test_g1_shadow.py.
G1_SHADOW_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ROBOT_USD,
        activate_contact_sensors=True,   # per-foot (ground) + per-fingertip (object) sensors
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,       # floating base must balance under real gravity
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            # Kept at 1.0. Lowering it (0.5, then 0.1) was tried to damp the impulse produced by the
            # reference's hand-inside-countertop overlap, and it does work for that: spawn-time link
            # speed p99 fell 1.17 -> 0.60 -> 0.18 m/s. It was chasing the wrong fault, though —
            # robots ending up below the floor happen at 1.0 too, and the feet never penetrate.
            # Measured in sim at the reference pose, the sole sits +0.63 cm above ground (median;
            # min -0.55, only 10/501 frames below zero). What looks like sinking is the LEGS BUCKLING:
            # hold the joint PD at the reference and the robot collapses 48 cm in 0.8 s while both
            # soles stay above ground, because a floating-base humanoid cannot stand on frozen joint
            # targets at these gains (hip/knee stiffness 99 Nm/rad). Standing is SONIC's job, done by
            # moving the targets every step. Nothing here is a depenetration problem.
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Match G1_29DOF locomotion setup. Self-collision off (full humanoid + 2 hands is
            # expensive and destabilizing); revisit if hand↔object/body contact needs it.
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Standing default (G1_29DOF pose); RSI overrides root+joints from the reference at reset.
        pos=(0.0, 0.0, 0.75),
        rot=(0.7071, 0.0, 0.0, 0.7071),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ---- G1 body: SONIC actuator gains (GEAR SONIC g1.py G1_CYLINDER_MODEL_12_DEX_CFG).
        #      The frozen SONIC decoder was trained with THESE gains + the per-joint action scale
        #      SONIC_SCALE = 0.25*effort/stiffness, so its a_sonic maps to the intended PD torque
        #      ONLY with this exact stiffness/damping/armature (verified via sonic_playback.py).
        #      effort_limit_sim is kept GENEROUS (SONIC's own 5 Nm wrists saturate under object
        #      load). Do NOT substitute the locomanip locomotion gains here. Groups mirror
        #      sonic_playback.py::_sonic_actuators. Shadow fingers unchanged (SONIC has no hands).
        "sonic_hip_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"],
            effort_limit_sim=300.0, velocity_limit_sim=100.0,
            stiffness=99.0997, damping=6.3088, armature=0.025101925),
        "sonic_hipyaw_waistyaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", "waist_yaw_joint"],
            effort_limit_sim=300.0, velocity_limit_sim=100.0,
            stiffness=40.1795, damping=2.5579, armature=0.010177520),
        "sonic_ankle_waist": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint",
                              "waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=200.0, velocity_limit_sim=100.0,
            stiffness=28.5013, damping=1.8143, armature=0.00721945),
        "sonic_shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                              ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"],
            effort_limit_sim=150.0, velocity_limit_sim=100.0,
            stiffness=14.2506, damping=0.9072, armature=0.003609725),
        "sonic_wrist_pitchyaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit_sim=150.0, velocity_limit_sim=100.0,
            stiffness=16.7783, damping=1.0681, armature=0.00425),
        # ---- Bimanual Shadow fingers (from robotis_shadow_grasp): 18 actuated DOF/hand ----
        "shadow_fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "robot0_(l|r)_(FF|MF|RF|LF|TH)J[1-3]",
                "robot0_(l|r)_LFJ4", "robot0_(l|r)_THJ4", "robot0_(l|r)_THJ0",
            ],
            velocity_limit_sim=15.0, effort_limit_sim=3.09,
            stiffness=1.0, damping=0.2,
        ),
    },
)


@configclass
class G1ShadowRephoEnvCfg(DirectRLEnvCfg):
    # --- Robot -----------------------------------------------------------------
    # Composite G1 + bimanual Shadow (validated). FLOATING base (fix_root_link=False in
    # G1_SHADOW_CFG.spawn.articulation_props) for locomotion.
    robot_cfg: ArticulationCfg = G1_SHADOW_CFG

    # --- Control / sim ---------------------------------------------------------
    decimation: int = 4                 # 50 Hz control (200/4) — SONIC's native control rate
    # DERIVED, not a knob: the env OVERWRITES this in __init__ with ref_len / action_fps, because an
    # episode now runs from its RSI start frame to the END of the reference sequence (variable length).
    # It therefore only sets max_episode_length, which acts as a safety cap behind the frame-based
    # time-out in _get_dones. The value below is a placeholder for cfg-dump readability.
    #   History: this used to be a real 3.0 s chunk (150 frames @50 fps) and the RSI start was clamped
    #   to [0, ref_len-150] so the chunk always fit — which made the tail of every clip unreachable as
    #   a start (251-frame clip → [0,101]; a median 151-frame clip → [0,1], i.e. no RSI at all).
    episode_length_s: float = 3.0
    # ACTION = SONIC latent residual z_res(64) + bimanual Shadow hand action(36) = 100. The 29 body
    # DOF are driven by the FROZEN SONIC decoder (z_res perturbs its FSQ latent pre-quantization,
    # GRAIL Eq.6); the policy never commands body joints directly. Hands = ABSOLUTE action
    # (user-locked): a_hand ∈ [-1,1] maps DIRECTLY to the Shadow joint range (EMA-smoothed), NOT a
    # residual/delta. (Body ACTION_DIM=65 constant kept for the joint-group machinery.)
    sonic_action_dim: int = 64          # z_res latent residual width (SONIC max_num_tokens×token_dim = 2×32)
    # [ROLLBACK MARKER: sonic-encoder-g1] ----------------------------------------------------------
    # Which of the checkpoint's tokenizer encoders conditions the frozen prior. Encoders are
    # ['g1', 'teleop', 'smpl'] and encoder_index (tokenizer slot [0:3]) is the selector.
    #
    #   "smpl"  smpl_joints_multi_future_local_nonflat (10,72) = 24 human joints x 3, pelvis-local
    #           joint_pos_multi_future_wrist_for_smpl  (10,6)   <- WRIST-SPECIFIC input
    #           smpl_root_ori_b_multi_future           (10,6)
    #   "g1"    command_multi_future_nonflat           (10,58) = [dof_pos(29), dof_vel(29)] ABSOLUTE
    #           motion_anchor_ori_b_mf_nonflat         (10,6)
    #           driven from the pyroki retarget, so the human->G1 morphology mapping is already
    #           solved by IK against G1's real kinematics instead of inferred by SONIC.
    #
    # Frozen-prior playback on s101_seg12_knife (scripts/process_dataset/sonic/sonic_playback_g1.py,
    # identical camera/object, --free_base) says g1 is NOT a free win:
    #                 body_err  arm_err  wrist  wrist_L  wrist_R  nearest wrist->object
    #     smpl          7.2 cm   7.5 cm  5.61     4.40     6.81   1.2 cm  (frame 142)
    #     g1            7.3 cm   7.4 cm  8.90    10.40     7.40   5.0 cm  (frame 385)
    # The whole-body means tie; the manipulation-relevant terms are 1.6-4x worse. Per-keypoint, g1
    # tracks the ELBOWS much better (left 8.91 -> 2.57 cm) and the WRISTS much worse — consistent
    # with commanding joint angles (accurate joints, error accumulating down the chain) and with the
    # smpl encoder having a wrist-specific input that the g1 encoder does not.
    #
    # NOTE both the tokenizer's encoder_index AND sonic_prior.encode_latent(encoder=...) must be set
    # together. encode_latent takes the encoder NAME, so flipping only encoder_index would leave the
    # env silently on smpl while looking switched. (sonic_playback uses SP.act, which routes off
    # encoder_index alone — that is why the playback comparison above was valid.)
    sonic_encoder: str = "smpl"         # "smpl" | "g1"
    # [/ROLLBACK MARKER: sonic-encoder-g1] ---------------------------------------------------------

    # [/ROLLBACK MARKER: obj-guidance] -------------------------------------------------------------
    hand_action_dim: int = 36           # bimanual Shadow finger action (ABSOLUTE, EMA-smoothed α=0.5)
    action_space: int = 64 + 36         # 100
    # --- SONIC-mode DELTA-ACTION switches (default OFF = the absolute/raw behavior above) --------------
    # When ON, the policy output is a per-step INCREMENT that is EMA-smoothed and integrated into a
    # clamped target (delta=0 ⟺ HOLD), instead of an absolute value. Independent per channel; enable
    # per experiment. (Distinct from the non-SONIC group *_delta_action switches, which are bypassed here.)
    #   hand:  integrate a_hand·scale (rad/step) into the hand JOINT target, clamped to joint limits.
    #   latent: integrate z_res_raw·scale into the SONIC latent residual, clamped to ±clip (anti-windup).
    # [ROLLBACK MARKER: hand-residual] -----------------------------------------------------------
    # Drive the 36 finger joints as a RESIDUAL on the retargeted hand pose instead of as an absolute
    # command over the full joint range: target = clamp(ref_joints[frame] + residual_scale_hands *
    # a_hand, limits), stateless (no EMA). a_hand=0 reproduces the retarget exactly.
    # Why: under the absolute mapping the hand was the only part of the robot with no reference
    # anchor (the body has one, via SONIC's tokenizer conditioning), and the policy saturated it —
    # Diag/hand_clamp_frac 0.446, i.e. 45% of finger dims pinned to a limit at any instant, so the
    # hand approached the knife already closed. Requires the retarget joint order fix
    # (remap_ref_joint_order): before it, the reference hand had MFJ1 fed by the thumb's value.
    # Also switches two regularisers to match the parameterisation — rew_action_reg_hands to the
    # commanded residual, rew_pose_reg_hands to the realised deviation FROM THE REFERENCE (it
    # anchors to the neutral pose when this is False). False restores the absolute mapping exactly.
    sonic_hand_residual: bool = False
    # [/ROLLBACK MARKER: hand-residual] ----------------------------------------------------------
    # [ROLLBACK MARKER: hist-seed-zero-vel] ------------------------------------------------------
    # At an RSI reset SONIC's 10-frame proprio history does not exist, so one row is copied into
    # all 10 slots — which freezes joint POSITION across the window. Carrying the live joint
    # VELOCITY into that window then asserts a state that cannot occur (positions unchanged while
    # the joints move) and that the frozen decoder never saw. True zeroes the seeded velocity so
    # the fabricated window reads as 'at rest for 10 frames'. It rewrites only the PAST: the robot
    # keeps the velocity the cache restored, and the next step appends the real row.
    # Warm-up was already consistent by accident (the reference reset path leaves jvel at 0), so
    # this only changes the ADAPTIVE phase, where cache restores carry real velocity.
    # [ROLLBACK MARKER: hist-from-reference] -----------------------------------------------------
    # Seed SONIC's 10-frame proprio window from the REFERENCE's own last 10 frames (episode-frame
    # indices clamped at 0, mapped through _canon_frame, velocities sign-flipped for backward)
    # instead of replicating the current measured row. Two things change: the window becomes a
    # real trajectory (positions/velocities/orientation mutually consistent, so the frozen-position
    # contradiction disappears), and it becomes the SAME trajectory the tokenizer already feeds as
    # the FUTURE — the past was the robot's state while the future was the reference, two stories.
    # All 10 slots including the newest: keeping the newest measured would express the tracking
    # error as a one-frame position jump the velocity channel contradicts.
    # False falls back to the replicated row, and then sonic_hist_seed_zero_vel applies.
    #
    # ON since 2026-08-05: this removes the RSI reset discontinuity outright. Measured on
    # s101_seg12_knife, 64 envs, zero action (Jerk/tgt_jump_reset = steps 1-3 of an episode,
    # tgt_jump_run = steps >=10, both rad, largest single-joint step):
    #     replicated row   reset 0.360  run 0.170  ratio 2.13     backward_ratio 0.5
    #     replicated row   reset 0.356  run 0.147  ratio 2.42     backward_ratio 0.0
    #     from reference   reset 0.171  run 0.171  ratio 1.00     backward_ratio 0.5
    #     from reference   reset 0.163  run 0.172  ratio 0.95     backward_ratio 0.0
    # Ratio 1.00 means a just-reset step is indistinguishable from a settled one. Backward rollouts
    # are irrelevant to it (0.5 vs 0.0 agree), so this is the reset seeding alone.
    # NOTE it does NOT lengthen episodes (ep_len 7.16 -> 7.35): those die on the obj_rot gate
    # (79-81% of deaths), which is a separate defect. Judge this flag on reset continuity, not on
    # episode length — an earlier read of root_v99 (0.806 -> 0.733) called it "secondary" because
    # that statistic averages over whole episodes and dilutes a transient that flushes in 10 steps.
    sonic_hist_from_reference: bool = True
    # [/ROLLBACK MARKER: hist-from-reference] ----------------------------------------------------
    sonic_hist_seed_zero_vel: bool = False
    # [ROLLBACK MARKER: act-seed-from-pose] True restores our jpr/sonic_scale action seed ("the
    # action that would command the pose the robot is in"). False is the GRAIL/IsaacLab
    # equivalent: the action manager zeroes actions on reset, so CircularBuffer replicates 0.
    sonic_act_seed_from_pose: bool = False
    # [/ROLLBACK MARKER: hist-seed-zero-vel] -----------------------------------------------------
    sonic_hand_delta: bool = False
    sonic_hand_delta_scale: float = 0.4        # rad/step at raw=1
    sonic_hand_delta_smoothing: float = 1.0    # EMA α on the delta (1.0 = no smoothing)
    sonic_latent_delta: bool = False
    sonic_latent_delta_scale: float = 0.5      # latent-units/step at raw=1
    sonic_latent_delta_smoothing: float = 1.0
    sonic_latent_delta_clip: float = 4.5       # |integrated z_res| bound (anti-windup). Sized from the
    #   trained absolute policy's z_res range (measured on agent_60000/pan: p99 |z|≈2.3, |max|≈3.9, std≈0.87),
    #   so the delta integrator can reach the same latent-residual extent the absolute policy uses (+ margin).
    sonic_z_res_clip: float = 5.0              # ABSOLUTE-mode z_res env clip [-5,5] (user 2026-07-23): the
    #   frozen SONIC decoder never sees extreme latents. Bounds the ENV action (not the PPO log-prob, which
    #   is on the raw Gaussian sample). p99 |z|≈2.3 in the trained policy, so 5.0 rarely bites in practice.
    #   This clip does NOT bound `rew_latent_reg`: that term is computed on the RAW (unclipped) residual
    #   again after the 2026-07-28 rollback, so it stays unbounded and keeps a restoring gradient on mu
    #   outside the clip. (While it briefly ran on the clipped value, its ceiling was
    #   rew_latent_reg · clip² = 0.25 here; widening the clip to 20 raised that to 4.0 vs alive 1.5 and
    #   helped pin the total reward on the clamp(min=0) floor — that widening was also reverted.)
    # obs = A proprio(190) + B reference(342) + C object/contact/history(182) = 714.
    #   A: root_linvel(3)+root_angvel(3)+jpos(65)+jvel(65)+palm_ori6d(12)+palm_linvel(6)+palm_angvel(6)
    #      +fingertip_vel(30)   [bimanual palm state + ft vel]. root absolute height/ori6d + projected_gravity
    #      REMOVED from the policy (user 2026-07-20) — SONIC's decoder still gets gravity/root proprio, and the
    #      policy sees the root only via block-B reference + deltas (robot root = ref − delta).
    #   B: kpts(54×3)+delta(54×3)+ref_root_pos(3)+ref_root_ori6d(6)+delta_root_pos(3)+delta_root_ori6d(6)
    #      (ref_root_pos = full xyz, was height-only; NO phase — look-ahead deltas convey progress)
    #   C: obj(15)+delta_obj(9)+delta_ft_obj(30)+artic_reserved(8)+future_contact(10)+ft_force(10)+prev_action(100)
    #      [delta_ft_obj = object-local ft→target offset; foot contact/force obs REMOVED 2026-07-20]
    #   prev_action = the 100-D policy action (z_res 64 + a_hand 36), GRAIL-style (not the realized 65-D
    #   joint target — the policy sees its own last residual). This is the RAW action. The ENV-EFFECTIVE
    #   (clipped, per-block normalized to [-1,1]) copy was tried on 2026-07-28 and ROLLED BACK the same
    #   day: it did not change the frozen-policy dead state (the analytic KL still read exactly 0 on
    #   76/80 logged points), so it bought nothing and the obs is kept faithful to what the policy
    #   emitted. The bounded copy is still used for the action_rate reward. See the block-C comment in
    #   _get_observations for the measurement, and note the real chain is the 100-D joint log-ratio
    #   feeding the `A<0` branch of PPO's `-min(A·r, A·clip(r))`, unbounded for r ≫ 1+ε.
    observation_space: int = 766        # asserted in _get_observations. Per-link contact (Option A): the
    #   fingertip future_contact(10)+fingertip force(10) obs were REPLACED by per-link mask(32)+force(32) → +44.
    state_space: int = 0

    # Enlarged GPU contact buffers (mirrors the grasp envs). The default buffers overflow for the
    # 78-body G1+bimanual-Shadow robot + object with track_contact_points on the fingertips
    # (ContactSensor._unpack_contact_buffer_data device-side assert) once the fingers press on the
    # object. Larger gpu_max_rigid_patch_count / aggregate-pair capacities fix it.
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,                 # 50 Hz control with decimation 4 (SONIC's native rate)
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.80665),
        # Contact/friction settings copied from robotis_shadow_grasp, which runs 2048 envs stably.
        # This env had NONE of them, so it inherited the IsaacLab defaults, and two of those are
        # actively wrong here:
        #   friction  — neither the robot USD nor the ground plane authors a physics material, so both
        #               fall back to 0.5 and the foot<->ground pair averages to 0.5. The objects DO
        #               author 1.0 (the converter's DEFAULT_FRICTION), so only the feet were slipping.
        #               A floating-base humanoid leaning over a counter on mu=0.5 slides, and the slide
        #               reads as the robot sagging.
        #   bounce_threshold_velocity — default 0.5 m/s means any contact ABOVE that is resolved
        #               elastically. Reset depenetration was measured throwing hand links at up to
        #               5.3 m/s, i.e. far above it, so those contacts bounced, re-touched and bounced
        #               again. 0.01 makes essentially every contact here inelastic.
        #   friction_correlation_distance — default 0.025 m merges friction anchors within 2.5 cm;
        #               Shadow finger links are closer together than that, so separate fingers' contacts
        #               were being merged. 0.00625 keeps them distinct.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_pairs_capacity=1024 * 1024 * 16,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            gpu_max_rigid_patch_count=1024 * 1024 * 16,
            friction_correlation_distance=0.00625,
            friction_offset_threshold=0.04,
            bounce_threshold_velocity=0.01,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0)
    # Viewer / video camera. Default origin_type="world" leaves the camera fixed in world coords, so
    # a FLOATING-BASE robot walks out of frame — video is hard to watch. The grasp/shadow envs use
    # origin_type="env" (env-local origin), which is enough for their FIXED-BASE arm. This robot walks,
    # so use origin_type="asset_root": eye/lookat are offsets from the ROBOT's root (pelvis) center in
    # env_index, i.e. a stable third-person camera that TRACKS the robot as it walks (world-aligned, no
    # spin). For a near-static station-manip clip this is ~identical to "env". To pin the camera to the
    # env origin instead (static frame), set origin_type="env" and drop asset_name.
    # ENV-FIXED camera (origin_type="env"): stable, no per-frame shake. asset_root (follow the robot
    # root) was too shaky. eye/lookat are ENV-LOCAL; the env re-points them at load to the reference-
    # root centroid (env-local) so the fixed camera still frames the robot wherever its clip places it
    # in the ParaHome world (the default here is only a fallback if that override does not run).
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.5, 1.5, 1.0),
        lookat=(0.0, 0.0, 0.8),
        origin_type="env",
        env_index=0,
    )

    # --- Debug visualization ---------------------------------------------------
    # When True, spawn VisualizationMarkers for the REFERENCE keypoints (green), the robot's
    # ACTUAL keypoints (cyan), and the reference fingertip pad targets (magenta) so tracking can
    # be inspected in the viewer. Off by default (no overhead in headless training); toggle for
    # play/rollout with a GUI. debug_vis_num_envs caps how many envs get markers (keep small —
    # each env draws 56+56+10 spheres).
    debug_vis: bool = True
    debug_vis_num_envs: int = 64
    # --- Tensorboard logging ---------------------------------------------------
    # "Episode_Reward /" holds ONE graph per reward TERM (the weighted per-term contribution) and
    # nothing else — no group aggregates, no diagnostics. The reward-shaping diagnostics
    # (tracking_penalty clamped/raw + clamp_frac) go to the "Diag /" tab instead; set False to drop
    # them entirely. NOTE the total reward is deliberately NOT logged by the env: skrl already
    # writes the identical value as "Reward / Instantaneous reward (mean)" (its base agent records
    # the rewards BEFORE rewards_shaper is applied).
    log_reward_diag: bool = True
    # env-fixed video/viewer camera zoom: scales the eye offset from the robot (smaller = closer).
    # 1.0 = the default close framing aimed at chest/hands; set 0.6 to zoom in further on the hands.
    viewer_zoom: float = 1.0
    # env-fixed video/viewer camera ANGLE — matched to render_retarget.py so training videos and the
    # retarget playbacks share one viewpoint. yaw=315° (+X,−Y), 18° elevation, aimed at the OBJECT
    # centroid (where the hands work) so the hands are not occluded by the torso. yaw=45/elev≈0/look_obj=
    # False reproduces the old +X+Y root-aimed view.
    viewer_yaw: float = 315.0        # azimuth around the look target (deg)
    viewer_elev: float = 18.0        # elevation (deg)
    viewer_look_obj: bool = True     # aim at the object centroid (else the root centroid)

    # --- Body links used as tracking anchors / balance / contact ---------------
    # (torso / wrist link names live in BODY_KPTS above — the env resolves every tracking anchor from
    # there, so there are no separate torso_body_name / *_wrist_body_name fields.)
    # Shadow fingertip bodies (bimanual), order thumb,index,middle,ring,little per hand.
    # CONFIRMED against the composite USD (all robot0_{l,r}_*distal bodies present).
    fingertip_body_names: list = [
        "robot0_l_thdistal", "robot0_l_ffdistal", "robot0_l_mfdistal", "robot0_l_rfdistal", "robot0_l_lfdistal",
        "robot0_r_thdistal", "robot0_r_ffdistal", "robot0_r_mfdistal", "robot0_r_rfdistal", "robot0_r_lfdistal",
    ]

    # =========================================================================== #
    # Reward weights (starting values — tune)
    # =========================================================================== #
    # A. Imitation (pose tracking) — extended from grasp
    # These A–D imitation/tracking terms are grouped into a tracking_penalty CLAMPED at -rew_alive
    # (mirrors the proven grasp RSI reward): a poorly-tracking-but-alive step nets ≤0 (no free
    # survival reward). foot_contact/foot_slip/force/regs stay OUTSIDE the clamp (as in grasp).
    # NOTE: all three errors are MEAN per-keypoint distances (.norm().mean()), so the count of
    # keypoints is already normalized out — the weight multiplies a "mean metres" quantity and is
    # directly comparable to grasp's weight on the SAME quantity (do NOT divide by keypoint count).
    # BODY-keypoint reward is SPLIT into two terms (2026-07-21): the 10 CORE body kpts (pelvis, torso,
    # shoulders, elbows, hips, knees) and the 4 END-EFFECTOR kpts (both wrists + both ankles), each with
    # its own weight so the extremities (reach + foot placement) can be emphasized independently. Both are
    # plain means over their group. The termination gate still uses the UNIFORM mean over ALL 14 (e["body"]).
    rew_body_kpts: float = -0.5        # mean over the 10 CORE body kpts (non-end-effector)
    rew_ee_kpts: float = -1.76          # mean over the 4 END-EFFECTOR kpts (L/R wrist + L/R ankle)
    rew_hand_kpts: float = -1.76        # mean over 40 finger-chain kpts (dexterity)
    rew_fingertip: float = -5.2        # mean over 10 bimanual pads ≈ grasp -5.2 (both are means)
    # B. Locomotion / root — root_ori up-weighted over root_pos (GRAIL-aligned: orientation > position).
    rew_root_pos: float = -0.5         # was -2.5 (user 2026-07-20: de-emphasize root position)
    rew_root_ori: float = -0.5         # was -1.0 (user 2026-07-20: emphasize root orientation)
    # C. Balance / feet — REMOVED (user 2026-07-20, GRAIL-aligned): the frozen SONIC base owns feet +
    #   balance, so the residual policy has NO foot-contact obs/reward (no rew_foot_contact/foot_slip,
    #   foot_force_cap, foot_kpt_gate, foot flatness). Feet are still tracked as part of the 14-kpt body reward.
    # D. Object manipulation — reused from grasp (inside the clamped tracking_penalty)
    rew_obj_pos: float = -4.26
    rew_obj_rot: float = -1.0
    # (no rew_obj_artic: objects always spawn as a single rigid base, so the articulation-DOF error was
    #  identically zero — see the NOTE in _get_rewards. Re-add it with the live joint read.)
    # PER-LINK contact-force reward (Option A / DexMachina, extends the old fingertip-only force reward to
    # all wrap links): compressive object force on each of the 32 wrap links, projected on the link's OWN
    # inward pad normal (LINK_PAD_NORMALS, like the fingertip `force·(-pad_normal)`) so the WRONG face (e.g.
    # back of the palm) registers no compressive force. Gated by (a) the reference per-link contact mask
    # (which links SHOULD touch), (b) a spatial gate (link near its object-surface target), and (c) an
    # ORIENTATION gate (link inward normal aligned with the reference reaction normal within
    # contact_normal_gate_tol). Normalized ∈[0,1] over the active links.
    rew_contact_force: float = 0.7     # per-link contact-force reward weight (was rew_fingertip_force)
    contact_force_cap: float = 1.0     # N; per-link compressive force at/above this = full credit (saturates)
    force_obs_clip: float = 300.0      # N; OBS-side clip on per-link + foot contact forces (user 2026-07-23).
    #   Raw forces spike to ~hundreds of N (foot ≈ body weight; measured obs var ~7e5) → destabilize the
    #   RunningStandardScaler. Clip the OBS copy only (reward uses contact_force_cap). Stabilizes scaling.
    use_contact_normal_gate: bool = False   # gate the per-link force reward by link-face ↔ reference-reaction alignment
    # rad (=75°); reward only if angle(link inward normal, ref reaction) ≤ this. 75° (not a tighter value)
    # because the human-derived reference reaction vs the robot Shadow link palmar face is inherently ~55°
    # misaligned even at the clean retarget-reference grasp (embodiment); measured on s100_seg00_pan, a 45°
    # gate passed only 14–50% of genuine reference-contact links (starves the reward), while 75° passes
    # ~62–80% yet still culls clearly-wrong faces. The force is already projected on the link's own inward
    # normal, so back-of-palm contact (reaction ≈ +palmar-outward) yields ≤0 compressive force regardless —
    # this gate is the secondary orientation filter, hence kept lenient.
    contact_normal_gate_tol: float = 1.25
    # E. Alive / regularization
    # rew_alive sets the tracking_penalty clamp floor (-rew_alive). Above the grasp default (1.5)
    # because the grouped full-body/bimanual tracking penalty is larger; kept high enough that good
    # tracking stays unsaturated (else the clamp kills the gradient) while flooring bad steps at 0.
    # User set 2.0 (from 4.0). Watch Reward/tracking_clamp_frac — if it climbs, raise back toward 4.
    rew_alive: float = 1.5
    rew_action_reg_hands: float = -0.004   # action-magnitude reg on the policy-controlled hand JOINTS. legs/
    #   arms/waist are SONIC-driven (regularized by rew_latent_reg on z_res instead).
    rew_pose_reg_hands: float = -0.001   # HANDS-only: pull achieved hand joints toward the DEFAULT (rest/neutral) pose
    #   — a task-agnostic regularizer (grasp/TJ convention), NOT toward the retarget reference (that would just
    #   duplicate rew_hand_kpts/rew_fingertip tracking). NOT applied to the SONIC-driven body.
    rew_action_rate: float = -0.001     # smoothness — on the RAW 100-D policy action (z_res + a_hand),
    #   NOT the realized joint target (user 2026-07-20; GRAIL meta_action_rate_l2). Penalizing the realized
    #   65-D target would penalize SONIC's own body tracking; this penalizes only what the policy controls.
    #   (GRAIL uses -0.1 for the full meta-action rate — tuning knob.)
    # SONIC latent-residual L2 penalty (GRAIL LatentL2, coef 0.01): mean(z_res²) keeps the latent
    # correction small so it does not override the frozen SONIC prior. Outside the tracking clamp.
    rew_latent_reg: float = -0.1

    # CoM-OVER-SUPPORT balance penalty (anti-fall). err = out-of-support excess (m) of the mass-weighted
    # CoM horizontal projection, in the foot-defined frame: relu(e_fwd - L_front) + relu(-e_fwd - L_back)
    # + relu(|e_lat| - L_side), L_side = ½‖ankleL-ankleR‖ + foot_halfw. err=0 when the CoM is inside the
    # support box (feasibility → does NOT fight the reference lean, verified err≈0 on the retarget ref);
    # grows only when the robot tips past the feet. Applied OUTSIDE the tracking clamp (like the regs).
    rew_com_support: float = -0.5        # weight on the out-of-support excess (0 = disable)
    com_support_l_front: float = 0.10    # ankle→toe forward support half-extent (m); ref CoM max ~0.066 → headroom
    com_support_l_back: float = 0.05     # ankle→heel backward support half-extent (m)
    com_support_foot_halfw: float = 0.03 # added to ½‖ankleL-ankleR‖ for the lateral support half-extent (m)

    # FEET-CONTACT-MATCH reward (VideoMimic/BSTRO): fraction of feet whose ACTUAL contact state matches the
    # REFERENCE contact schedule. r = mean_{i∈{L,R}} 𝟙[c_i == c*_i] ∈ [0,1] (NORMALIZED by #feet so both-
    # match = 1.0, not 2 — the weight is the true max bonus), POSITIVE bonus (outside the tracking clamp).
    # c_i = actual foot i in contact (ground-filtered force_matrix projected on the sole normal > force_thresh);
    # c*_i = reference foot i in contact, precomputed at load from the ParaHome ball keypoints via the SAME
    # rule PyRoki used for retargeting: ball_z < foot_plant_h AND |ball_vz| < foot_plant_vz. Rewards correct
    # contact TIMING (planted-when-should-be AND swing-when-should-be) → complements the CoM-over-support net.
    rew_feet_contact_match: float = 0.05   # POSITIVE max bonus (both feet matching → +this; 0 = disable)
    foot_plant_h: float = 0.06            # reference contact: ball height below this = planted (m; = PyRoki)
    foot_plant_vz: float = 0.15           # reference contact: |ball vertical speed| below this (m/s; = PyRoki)
    foot_contact_force_thresh: float = 20.0  # actual contact: compressive foot↔ground force above this = in contact (N)

    # =========================================================================== #
    # Termination — DEVIATION FROM REFERENCE (so reference crouch/bend never triggers)
    # =========================================================================== #
    # All gates compare robot vs the reference pose at that frame. NO separate root pos/tilt/height
    # gates: the mean body-keypoint error subsumes them (a root shift moves every keypoint, a tilt
    # rotates the far keypoints away, a fall drives foot/pelvis keypoints off) — one clean body gate.
    term_body_kpt_err: float = 0.50        # m, mean body-keypoint tracking error (covers root/fall).
    term_obj_pos_err: float = 0.15         # m, active-object position tracking error (= grasp max_obj_pos_err; was loosened to 0.20, now grasp-parity)
    term_obj_rot_err: float = 0.75         # rad, active-object rotation tracking error (= grasp max_obj_rot_err; was 0.80)
    # Fingertip + wrist-POSITION deviation gates (mirror grasp max_ft_mean_err / max_wrist_pos_err;
    # bimanual → worst-of-two-hands). Together with obj + wrist-rot + the added body-kpt gate, these are
    # the full termination set (= original grasp 5 gates + body_kpt). The finger-chain (hand) keypoint
    # mean is REWARD-only (rew_hand_kpts) and is NOT a termination gate — matching grasp, which never
    # terminates on its keypoint mean. (A prior misport terminated on the hand-chain mean AND dropped the
    # wrist-position gate; corrected 2026-07-20.)
    term_ft_err: float = 0.15              # m, mean fingertip (pad) tracking error (= grasp max_ft_mean_err)
    term_wrist_pos_err: float = 0.15       # m, wrist (arm end-effector) POSITION deviation, worst-of-two-hands (= grasp max_wrist_pos_err)
    # Wrist/palm ROTATION deviation gate (mirrors grasp max_wrist_rot_err). DISABLED (2026-07-20): the
    # robot palm orientation is not a reliable target in this embodiment — the position-only PyRoki
    # retarget + Shadow≠human hands don't reproduce the human wrist orientation, and any human-keypoint-
    # derived reference drifts up to ~90° over a clip (measured) → false terminations. So wrist ROTATION
    # is left unconstrained (positions are still gated by body/ft/wrist_pos). Re-enable only with a
    # self-consistent reference (e.g. retarget-FK: the robot's own palm quat FK'd at each retarget frame)
    # AND a retarget that emits g1_palm_quat / that reference in _ref_palm_quat.
    enable_wrist_rot_termination: bool = False
    term_wrist_rot_err: float = 0.75       # rad, per-hand palm-rotation deviation (= grasp max_wrist_rot_err)
    termination: bool = True               # master switch (False during eval/warm-up)
    # GRACE PERIOD: suppress the deviation (tracking) termination for the first N steps of each episode
    # so an episode is not born-dead while the policy has not yet corrected the reset pose (e.g. the
    # absolute hand drifting off the reference in the first ~12 steps). The NON-FINITE containment reset
    # is NOT graced (a NaN env always resets). Applies to train + eval resets alike.
    enable_termination_grace: bool = False  # on/off switch for the grace window below
    termination_grace_frames: int = 10     # steps at episode start with deviation-termination disabled

    # =========================================================================== #
    # Objects — active (dynamic, tracked) vs others (frozen for collision only)
    # =========================================================================== #
    # (no num_active_objects knob: the env always tracks exactly ONE active object — the first
    #  `obj__*__base` key in the clip npz — and freezes the rest as context.)
    freeze_inactive_objects: bool = True   # spawn non-manipulated scene objects as static support/collision
    # Context (support/collision) objects are the non-manipulated scene objects (shelf/table/counter)
    # the active object rests on / is manipulated near. Spawned KINEMATIC-frozen at their reference
    # pose so the dynamic active object has something to rest on (otherwise it falls through to the
    # floor). Selected per clip = {objects within context_radius (XY) of the active object's SWEPT
    # trajectory}  ∪  {the SUPPORT: nearest object whose frame-0 centroid is below the active object,
    # within context_support_radius}. The support term is a safety net because a large-footprint
    # support (a table/counter) can have its CENTROID outside context_radius even though its top is
    # right under the object (centroid distance is an approximation of a proper AABB/footprint test).
    context_radius: float = 1.0            # m, XY centroid distance to the active object's swept path
    context_support_radius: float = 1.5    # m, XY cap for the always-included below-object support

    # [/ROLLBACK MARKER: object-settle-lift] -----------------------------------------------------

    # [ROLLBACK MARKER: spawn-declear] -----------------------------------------------------------
    # Clear the object out of its support at SPAWN, solved once at env startup against whatever
    # colliders and physics are actually loaded.
    #
    # Why this replaces object_settle_lift: that one bakes a precomputed lift into the reference
    # trajectory, so it is solved against ONE scene and silently goes stale. Raising the context
    # colliders from 16 to 64 hulls and friction to 1.0 did exactly that — measured across four
    # clips it now HURTS two of them (pan 3.38 -> 3.58 deg, pot 2.06 -> 2.40 deg settled rotation).
    #
    # Method (see scripts/process_dataset/diagnostics/test_runtime_declear.py, which is where the
    # numbers below come from): pin the object at the reference pose with zero velocity, step once,
    # and read the velocity the solver tried to give it. Nonzero means it was overlapping something.
    # Raise 0.5 mm and repeat. This needs no penetration query — IsaacLab's ContactSensor exposes
    # forces and contact positions but not separation depth, and "the solver pushed a body we
    # pinned" carries the same information.
    #
    # REFERENCE-REST ONLY: applied where the reference object is stationary, and held CONSTANT
    # across each contiguous rest segment. Everywhere else the lift is zero. A moving object is
    # either in flight or being carried, where "release it and see where it settles" is not a
    # spawn question at all, and a per-frame lift there would inject spurious reference velocity.
    #
    # SPAWN ONLY: _np_obj_base / _ref_obj_* stay exactly as captured, so reward, observations and
    # the finite-diff reference velocities all still see pure GT. Only the reset pose moves, and
    # only on the reference path — RSI cache hits restore a genuinely simulated state that is
    # already clear. The object settles back down onto its support, so this does not cost position
    # accuracy: distance from the UNCORRECTED reference after settling drops on every clip tested.
    #
    # Settled spawn rotation over the frames where the reference is at rest, raw -> declear:
    #   s101_seg12_knife  42.04 +- 59.48  ->   9.94 +- 1.83 deg   (and 1.97 -> 0.70 cm from GT)
    #   s101_seg30_bowl   28.02 +- 17.57  ->   2.38 +- 1.02 deg   (penetrating frames)
    #   s100_seg00_pan     3.38 +-  1.45  ->   2.56 +- 0.78 deg
    #   s101_seg29_pot     2.06 +-  0.44  ->   2.13 +- 0.66 deg   (no penetration problem to fix)
    # The spread is what matters: a thin knife or a round bowl tips unpredictably out of a
    # penetrating spawn, and a policy cannot learn a grip whose object pose is a coin flip. Flat
    # wide-based objects (pan, pot) sink into their support just as much but do not topple, so
    # they show little change. The residual ~9 deg on the knife is a genuine settled tip under
    # gravity, not a spawn artifact, and no height offset can remove it.
    object_spawn_declear: bool = True
    declear_settle_steps: int = 12
    declear_max_lift: float = 0.05         # m, safety clip so a pathological frame cannot launch it
    declear_rest_lin: float = 0.05         # m/s   reference speed under which the object counts as at rest
    declear_rest_ang: float = 0.25         # rad/s reference angular speed for the same test
    # [/ROLLBACK MARKER: spawn-declear] ----------------------------------------------------------

    # =========================================================================== #
    # Adaptive frame-sampling curriculum + pretrain-cache RSI (reused from grasp)
    # =========================================================================== #
    # ── MASTER RSI SWITCH ─────────────────────────────────────────────────────────────────────
    # False → NO Reference State Initialization. Every episode starts at FRAME 0 and the robot is
    # reset to the frame-0 retarget-reference pose — deterministic and identical every episode
    # (classic fixed-start imitation baseline). The train / pretrain state caches are never READ
    # (still written, so flipping this back to True mid-run works); the object is seeded at its
    # frame-0 reference pose+velocity. Every knob below (adaptive/failure-weighted sampling,
    # adaptive_back_seconds, uniform_sampling_steps, rsi_curriculum_steps, pretrain_cache_warmstart)
    # is inert while this is False. Episode length is unaffected: frame 0 → end of the sequence.
    # Confirm it is live via `Diag / rsi_start_mean` == 0.
    #   TEMPORARILY DISABLED (2026-07-28) to A/B the PPO ratio explosion — diverse-from-step-0 RSI
    #   onto cold nets is the leading suspect (see the rsi_curriculum_steps note below, and the
    #   A<0 unclipped-surrogate branch that turns one large log-ratio into an fp32 gradient
    #   overflow). Set back to True once that is settled; the pretrain cfg pins it True because
    #   pretrain exists to visit every frame and fill the cache.
    # RePHO's sampleable-start bounds. It samples the start DIRECTLY (intermimic.py:1401-1402) —
    # there is no run-up rewind — and only trims the right end so a start cannot land with nothing
    # left to roll out (`_init_range_right = max_episode_length - 24`, :111).
    init_range_right_margin: int = 24
    # ---- per-frame staging quality bars (see _save_state_cache) ----
    # Two object bars picked by the SAMPLING PHASE: the loose pair during the uniform warm-up, the
    # tight pair once adaptive sampling begins. Frame-local, never latched.
    enough_ft_threshold: float = 0.10
    enough_obj_threshold: float = 0.085          # warm-up object pos bar (m)
    enough_obj_rot_threshold: float = 0.425      # warm-up object rot bar (rad)
    enough_obj_threshold_late: float = 0.05      # adaptive-phase object pos bar
    enough_obj_rot_threshold_late: float = 0.25  # adaptive-phase object rot bar
    cache_body_bar: float = 0.30
    cache_root_pos_bar: float = 0.10
    cache_root_rot_bar: float = 0.30

    # ---- [ROLLBACK MARKER: contact-term] contact-loss termination ----
    # RePHO ends an episode once required body-object contact has been missing for this many
    # consecutive steps (intermimic.py:1755, SupMat S.6 ii). It is what makes survival length a
    # proxy for doing the task — which the cache score, the sampler and the swap gates all assume.
    # 0 disables.
    # 6, not 10: `> 6` is the TRAIN branch (intermimic.py:1753). The `> 10` at :1755 is the eval
    # branch, which also swaps the streak for a non-resetting cumulative counter (:1941).
    contact_loss_frames: int = 6
    contact_loss_grace: int = 2        # steps after a reset during which the streak cannot build
    # Which of RePHO's four streaks may terminate: any subset of the channel names below. EMPTY =
    # contact termination off, which is the shipped default, and the reason is measured rather than
    # cautious. A channel can only teach anything if the retargeted REFERENCE POSE satisfies it: a bar
    # the reference itself trips would kill a perfect tracker and just truncate every episode.
    # Pinning every env to the reference pose and reading the env's own contact path
    # (scratchpad/test_ref_palm.py) gives the longest CONSECUTIVE violation run vs this 6-frame bar:
    #   clip              l_fing      r_fing      l_palm     r_palm
    #   s101_seg12_knife  kill(12)    ok(5)       kill(12)   kill(351)
    #   s100_seg00_pan    no demand   ok(3)       no demand  kill(120)
    #   s101_seg29_pot    ok(5)       ok(3)       kill(11)   kill(277)
    #   s101_seg30_bowl   ok(4)       kill(11)    kill(64)   kill(45)
    # No subset is safe on all four: knife needs l_fing off, bowl needs r_fing off. Raising
    # contact_loss_frames past 12 would make every channel survivable everywhere, but the streak
    # distribution decays ~65%/frame so the rule would then almost never fire. No palm channel
    # survives anywhere — hand_contact.npz derives contacts from the HUMAN SMPL-X hand (wrist->palm
    # via lbs argmax) and demands contact the Shadow palm's geometry cannot reach; the same map also
    # feeds the contact-force and fingertip rewards, so that label is worth fixing on its own.
    # The four violation rates stay visible as `Diag / clost_*` whether or not they terminate.
    contact_loss_channels: tuple[str, ...] = ("l_fing", "r_fing", "l_palm", "r_palm")
    contact_force_thresh: float = 1.0  # N on a required link to count as contact
    use_rsi: bool = True
    adaptive_sampling: bool = True
    failure_weighted_sampling: bool = True   # TRAIN: True (failure-weighted). PRETRAIN overrides → False (uniform).
    # 0.96 -> 0.0 (back to the instantaneous step reward, still clamped at 0). Measured on knife 40k
    # against an otherwise identical run: the discounted return ACCELERATED early learning (reward
    # 12.58 vs 8.74 and episode length 55.5 vs 41.9 over 3-6k) and then FROZE it — reward flat at
    # 14.2-14.9 from 6k through 19k while the baseline climbed 12.6 -> 25.2, and episode length
    # actually fell back 55.9 -> 48.5. `Cache / score_mean` plateaued at ~14.3 at exactly the step
    # the reward stopped moving, and `Cache / overwrite` sat at ~10 of 501 frames = 2% turnover.
    # The mechanism is self-reinforcing: return-to-go is a SHARP estimate of "how well does it go
    # from here", so only high-scoring states survive; RSI then starts almost exclusively from those,
    # the policy only ever sees them, and no fresh candidate can beat an entry scored under a policy
    # that has since specialised to it. The instantaneous reward is a NOISIER key, which keeps the
    # cache churning and the start distribution diverse — the exploration mattered more than the
    # accuracy of the ranking. Sigma was identical in both runs (0.38 -> 0.43), so this is not an
    # action-noise effect. NOTE the clamp at 0 is kept (grasp/TJ do the same) because the decay term
    # below needs non-negative scores.
    # 0.99 (RePHO's gamma). REQUIRED by repho_length_score: _episode_returns is only evaluated when
    # this is > 0, and without it cand_ret falls back to the step reward, so the replacement rule's
    # tiebreak would silently compare instantaneous rewards instead of returns.
    # 0.96 = agents/skrl_ppo_cfg.yaml discount_factor, deliberately NOT RePHO's 0.99. The cache's
    # return must discount on the same horizon the critic does, or the two disagree about what "how
    # well did it go from here" means: at 0.99 the cache ranks a state over ~100 steps of future while
    # PPO values it over ~25, so the states it promotes are ones the policy is not trained to exploit.
    # REQUIRED > 0 by repho_length_score — _episode_returns is only evaluated when it is, and otherwise
    # cand_ret silently falls back to the step reward, making the replacement rule's tiebreak a
    # comparison of instantaneous rewards instead of returns.
    cache_score_gamma: float = 0.96           # 0.0 → instantaneous reward
    cache_score_decay: float = 5e-4          # per flush; 0.0 → no decay
    # [/ROLLBACK MARKER: cache-score-rework] -------------------------------------------------------

    # [ROLLBACK MARKER: rsi-phase-split] ---------------------------------------------------------
    # Split RSI into two phases with DIFFERENT start-state sources, instead of one rule throughout.
    #
    # The problem today: an episode restores from the state cache whenever its start frame happens to
    # be covered, else from the reference. Coverage grows front-to-back, so during the uniform warm-up
    # the early frames already start from cached (settled) states while later frames still start from
    # the reference (kinematic, unsettled) — and that boundary keeps moving as coverage fills in. The
    # warm-up distribution therefore drifts, and "what did the warm-up train on" has no fixed answer.
    #
    # With these on:
    #   WARM-UP (uniform sampling)  reference ONLY. The cache is written but never read, so every
    #                               episode starts from the same kind of state. This also matches
    #                               evaluation, where the cache is empty and rollout always starts
    #                               from the reference.
    #   ADAPTIVE                    start from the cache. The sampled TARGET frame is restricted to
    #                               frames that have a cached frame within `adaptive_back_seconds`
    #                               behind them, and the episode starts at the FURTHEST cached frame
    #                               in that window (longest available run-up).
    #
    # Why a gap in coverage is harmless: targets just past a gap simply drop out of the pool, while
    # targets before it stay in — and episodes launched from those roll forward INTO the gap, filling
    # it from the left (the adaptive phase saves every step, see cache_min_episode_length_adaptive).
    # Once filled, the later targets reappear. The pool is self-healing, so no special handling,
    # no forced frame-0 seed and no out-of-window search are needed.
    reference_only_warmup: bool = True   # warm-up ignores the cache, reference-only start states
    # [/ROLLBACK MARKER: rsi-phase-split] --------------------------------------------------------

    # [ROLLBACK MARKER: backward-dir] ------------------------------------------------------------
    # Fraction of envs that track the reference BACKWARD in time (last frame -> first). 0 disables
    # the whole mechanism and the env is byte-identical to forward-only.
    #
    # Why: the state cache can only grow FORWARD from a cached start, so the frames near the very
    # beginning of a clip are only ever entered as a start, never rolled INTO. A backward episode
    # passes through them at the END of its own trajectory, so it fills exactly the slots the forward
    # direction cannot. Both directions share ONE cache, stored in ORIGINAL (forward) clip time.
    #
    # Feasibility was measured, not assumed: sonic_playback.py --reverse drives the FROZEN SONIC
    # prior on a time-reversed reference in closed-loop physics and tracks it as well as forward
    # (body_err 8.9 cm both ways, robot never falls) — so the prior, which is fed a 10-frame FUTURE
    # window, is not broken by the reversal.
    #
    # A single policy handles both directions, conditioned on a 0/1 phase bit appended to the
    # observation (hence observation_space +1). RePHO trains two separate policies and has them
    # donate states to each other through files; one conditioned policy avoids that machinery.
    #
    # NOT used as a mixing ratio any more. The WARM-UP is forward-only (backward has nothing to
    # extend while every episode starts from the reference), and in the ADAPTIVE phase the (2,F)
    # failure table decides the direction mix on its own. This value now only switches the feature
    # on (>0) or off (0). Backward bootstraps from `adaptive_uniform_ratio`'s floor, which is spread
    # over both rows, so it enters the adaptive phase at a few percent and grows as it accumulates
    # failures — watch `Curriculum / backward_sampled` to confirm it actually ramps.
    # 0 disables backward rollouts entirely: _use_backward goes False, allow[1] is never set, so
    # every draw lands on the forward row and _adaptive_dir_frame_weights degenerates to plain
    # forward failure-RATE frame weighting (failure_rate_normalize below). The backward direction
    # was meant to teach release, on the theory that release-then-reverse teaches grasp — but the
    # policy never released (the clip has no release: reference contact runs frame 31 to 500, so
    # release exists only below frame 31, and with StartHist p50 = 218 and ~39-step episodes a
    # backward episode almost never reaches it). Costing half the samples for a mechanism that
    # never fired is not worth it.
    # FIXED share of resets given to backward (see [dir-fixed-share] in the env). Not a bid that
    # failure statistics win or lose — backward's job is to lay down cache at the frames FORWARD
    # dies on, so the budget is chosen and the target frames come from forward's failure table.
    # 0.0 (2026-08-05, user): FORWARD ONLY, with the existing failure-weighted frame sampling. Setting
    # this to 0 also switches off the whole direction machinery — _use_backward = backward_ratio > 0
    # and use_rsi and adaptive_sampling — so _canon_frame mapping and the velocity sign flip go inert
    # and the (2,F) failure table is driven by its forward row alone. The 25%-fixed-share and
    # opposite-row variants were both tried and neither beat forward-only.
    backward_ratio: float = 0.25   # FIXED partition: the last 25% of envs run backward
    # [ROLLBACK MARKER: slot-cache] ----------------------------------------------------------------
    # K-slot state cache + contribution-driven direction split + bad-reference exclusion.
    # Set cache_num_slots=0 / backward_contrib_ema=0 / cache_exclude_bad_reference=False to get the
    # previous single-slot, failure-driven-direction, no-exclusion behaviour back.
    #
    # WHY SLOTS. Every attempt this session to make the SINGLE slot smarter narrowed the start-state
    # distribution and stalled learning: the discounted-return score froze the run at 6k, the
    # instantaneous-reward score at 12k, and the loosest (oldest) variant went furthest. RePHO can
    # afford a much sharper score than any of those because it keeps K alternatives per frame, draws
    # among them by lottery, protects a reference slot, and decays scores — we ported the sharp score
    # WITHOUT the diversity machinery. Slots restore it structurally.
    #   slot 0      the retarget reference. State FIXED, score LIVE (mirrors RePHO's protected slot 0,
    #               except RePHO pins its score at 1.0 forever; ours tracks reality so it cannot go
    #               stale as the policy improves).
    #   slot 1..K   states the policy actually reached. Lowest-scoring slot is evicted on a write.
    #
    # WHY THE DIRECTION IS NOT FAILURE-DRIVEN. Sampling the direction in proportion to its failure
    # starves whichever direction is WINNING: it survives, so it terminates rarely, so it accumulates
    # no failure mass, so it stops being drawn. Measured on run 2026-07-31_01-47-21: backward
    # episodes lasted 147 steps against forward's 49 while backward's share collapsed 0.44 -> 0.09,
    # and forward — the direction that actually ships — REGRESSED (54.7 -> 48.5 over 35k steps).
    # Backward's only route to helping forward is putting better states in the cache (time reversal
    # is not dynamically valid, so the skill does not transfer), and that contribution is directly
    # measurable, so it drives the split instead. A fixed schedule would also work but has to be
    # retuned per clip; the bandit reads the same curve off the data.
    #
    # WHY BAD FRAMES ARE EXCLUDED PER-SLOT, NOT PER-FRAME. The warm-up measures one thing: "starting
    # from the REFERENCE state at frame f never survived 40 steps while staying good_enough". That
    # condemns the reference row at f, not the frame — a state the policy later reaches at f can be
    # perfectly fine. So the exclusion zeroes slot 0's lottery weight at f and nothing else.
    #
    # HOW A FRAME COMES BACK. _save_state_cache clears _bad_ref[f] whenever it writes a PHYSICS state
    # into slot 0 at f, because the warm-up's verdict was on the reference row that state just
    # replaced. That is what makes this work at cache_num_slots=0, where slot 0 is the ONLY slot:
    # excluded frames cannot be STARTED from, but episodes passing THROUGH f still write states there,
    # and the frame is startable again as soon as one does. Without that clear the exclusion is
    # permanent (_bad_frozen never unfreezes) and one warm-up failure kills the frame for the run.
    #
    # IT DOES APPLY TO BACKWARD. An earlier version of this note claimed otherwise; the code does not
    # agree — the backward row of the sampling pool is `torch.flip(covered0)` and covered0 already has
    # _bad_ref folded in, so an excluded frame is excluded in both directions.
    # 2 (RePHO uses 3 incl. the reference). repho_length_score's rule evicts the worst LEARNED slot
    # (intermimic.py:1870 indexes [1:]), which does not exist at 0 — and with slot 0 protected there
    # would be nowhere to write at all.
    cache_num_slots: int = 3                 # learned slots per frame (total K+1 including the reference)
    # [ROLLBACK MARKER: repho-cache] ---------------------------------------------------------------
    # Port of RePHO/InterMimic's RSI buffer, read from the released code
    # (github.com/dingbang777/RePHO, intermimic/env/tasks/intermimic.py) rather than the paper, which
    # simplifies it. Every field below is OFF/0 by default: each stage can be enabled alone.
    #
    # WHAT RePHO ACTUALLY DOES, by line:
    #   1829-1844  A rollout writes its WHOLE trajectory only if it was long enough (>30 steps with
    #              continuous contact, or >70 with >=70 contact steps). Otherwise it writes ONLY its
    #              START frame, valued at end-start. Neither -> nothing.
    #   1831-1841  A rollout that reached the clip end keeps all its frames; one that DIED drops its
    #              last 20, so the states leading into a failure never enter the buffer.
    #   1832/1837  The stored value is `end - t`: how much longer the episode lived AFTER frame t.
    #              Comparisons are per-frame, so every candidate at t shares the ceiling T-t.
    #   1870       Replace the worst LEARNED slot when  L_new > L_min AND R_new >= R_min*ratio,
    #              or unconditionally when L_new > L_min + 10.
    #   1892-1899  Decay: L *= (1-5e-4) on slots 1.., R *= (1-5e-2) on ALL. Two different rates, and
    #              slot 0 is exempt from the L decay.
    #   412        Slot 0 (the reference) is seeded at 1.0 and never replaced.
    #   1902-1905  Switch to adaptive when sum(L>25)>3 and epoch>30; ALSO a relaxed backup at
    #              epoch>150 with sum(L>12)>3, so a hard clip cannot stay in uniform forever.
    #   1321-1346  Frame sampling: P(t) proportional to summed L, then PENALISED where the clip is
    #              already being finished (x0.5 above 0.8 finish rate, x0.2 above 0.9 with <=15 steps
    #              left). Without that penalty, sampling by L would prefer the frames that already
    #              work — the penalty is what makes it a curriculum.
    #
    # DELIBERATELY NOT PORTED: the contact conditions on the length gates (our contact reward is not
    # trustworthy — see use_contact_normal_gate), the penetration.npy pre-filter (we derive bad frames
    # from the warm-up instead), and the hard-coded epoch constants 53250/54400/58500 (experiment-
    # specific, three orders of magnitude off the 30/150 switch scale).

    repho_switch: bool = True
    repho_switch_min_steps: int = 500        # floor, mirrors RePHO's epoch>30
    repho_switch_len_hi: float = 25.0        # RePHO's sum(L>25)>3
    repho_switch_len_lo: float = 12.0        # RePHO's relaxed sum(L>12)>3
    repho_switch_count: int = 3              # RePHO's ">3"
    repho_switch_relax_steps: int = 2000     # after this, accept the relaxed bar
    repho_switch_max_steps: int = 4000       # hard ceiling: leave uniform regardless

    # Drop the last N staged frames of an episode that DIED, keep everything when it timed out at the
    # clip end (RePHO 1831-1841). Aimed at the states that lead into a failure: today the whole
    # episode is kept or dropped by cache_min_episode_length, which is far blunter.
    repho_drop_tail_on_death: int = 20        # 20 = RePHO

    # Slot 0 holds the retarget reference. RePHO seeds its score at 1.0, exempts it from decay and
    # never replaces its state; ours is seeded at 0, which in a survival-length regime reads as
    # "worst possible" and would be evicted immediately.
    repho_protect_slot0: bool = True

    # --- stage 2: what the cache STORES. Changes the meaning of column 0. ---
    # Score becomes SURVIVAL LENGTH (end - t) with the discounted return kept alongside as the
    # tiebreak, instead of the instantaneous step reward. The step reward answers "did this look good
    # at this instant"; survival length answers "how long did the episode live after being here",
    # which is the only question an RSI start state is ever asked.
    # NOTE cache_score_gamma (return-as-score) was tried and reverted — it accelerated early then
    # froze, because `new > old` is the only write condition and nothing lowered `old`. RePHO avoids
    # that with the two decays below, so this port must keep them on.
    repho_length_score: bool = True
    repho_trust_completion_after: int = 50   # clip-end completions before the tail cut is skipped
    repho_completion_min_span: int = 50      # ... and only starts this far before the end count

    # ── [ROLLBACK MARKER: curriculum-window] RePHO's start-frame window + post-swap seam drill ──
    # init_range_left = 0 disables the window entirely (every frame sampleable from step 0, which is
    # what we ran until now). Set it > 0 to make RSI start no earlier than that frame and open up only
    # once the policy has repeatedly run the clip from the boundary. Applied in EPISODE time, so it is
    # symmetric across directions. The seam drill needs track_buffer on to ever fire.
    init_range_left: int = 0
    left_boost_after: int = 100      # boundary completions before it gets RePHO's x3 (intermimic:1336)
    left_open_after: int = 200       # ... and before the window opens to 0 (intermimic:1787)
    tar_min_segment: int = 30        # a swap must span this many contiguous frames to be drilled
    repho_decay_length_frozen: float = 5e-8   # buffer decay while the seam is being drilled
    # RePHO anneals this to 0 within ~8% of its run (ratio = max((53250-epoch)/1000, 0), epoch
    # starts at 53001), after which the buffer is decided by survival length alone. That is safe
    # THERE because its early termination kills an episode once required body-object contact is
    # lost for over 10 consecutive frames (SupMat S.6 ii) — surviving long REQUIRES holding the
    # object, so length IS quality. We have no contact-loss termination and our error bars sit
    # 2-3.6x above the errors actually reached, so a policy that lets the object drift survives
    # fine: length and quality are decoupled here, and the return term is the only thing tying
    # them together. Measured with the annealing on: Cache/return_mean ended at 0.225 against
    # 2.5-3.0 on the two runs before it, while Cache/length_mean was the HIGHEST of the three
    # (109) — the buffer filled with long-surviving, low-reward states, restarts began from them,
    # and Diag/clamp_frac went 0.24 -> 0.71 with per-step reward collapsing 0.40 -> 0.046.
    # Held at 1.0 (the incumbent's return must not be beaten downward) until a contact-loss
    # termination exists to carry the signal.
    repho_return_ratio_start: float = 1.0
    repho_return_ratio_steps: int = 0        # 0 = no annealing, ratio stays at _start
    repho_replace_margin: int = 10           # m: L_new > L_min + m replaces unconditionally
    repho_decay_length: float = 5e-4         # on slots 1.., per flush
    repho_decay_return: float = 5e-2         # on ALL slots — 100x faster; the return ages with policy
    # Short episodes still contribute their START frame, valued at the length they achieved (RePHO
    # 1843). Only meaningful once the score IS survival length: with an instantaneous-reward score a
    # short episode's start frame can carry a high value and evict a good entry while adding nothing.
    repho_start_frame_fallback: bool = True
    repho_full_traj_length: int = 40         # above this, write every staged frame

    # --- stage 3: how frames are SAMPLED. Replaces the failure-rate hazard. ---
    repho_length_sampling: bool = True
    # RePHO's slot lottery subtracts 6 and clamps at 1 (intermimic.py:1381): the clamp IS the floor,
    # so no separate uniform term is needed and slot_uniform_ratio goes unused under repho_length_score.
    repho_slot_floor: float = 6.0
    repho_sample_floor: float = 7.0          # subtracted before weighting (RePHO 1316)
    repho_finish_hi: float = 0.9             # completion fraction above which weight *= 0.2 ...
    repho_finish_lo: float = 0.8             # ... and above which weight *= 0.5
    repho_finish_left: int = 15              # ... the 0.2 rule also needs <= this many steps left
    repho_penalty_hi: float = 0.2
    repho_penalty_lo: float = 0.5

    # ── [ROLLBACK MARKER: cross-buffer] RePHO inter-direction update (SupMat Alg 2/3) ────────────
    # A rollout proves two things about every frame it passes: how far it still got (self) and how
    # far it had come (cross). The second is what the OPPOSITE direction needs, so it is staged in a
    # per-direction cross buffer and periodically imported into the other direction's reserved slot.
    # Inert while backward_ratio == 0: the backward cross buffer is never written, so nothing to import.
    cross_buffer: bool = True
    cross_interval: int = 80        # 10 RePHO epochs x our PPO rollout length 8
    cross_margin: float = 10.0      # import only if it beats the reserved slot by this (SupMat m)
    cross_abs_floor: float = 40.0   # ... and clears an absolute length bar (intermimic.py:1058)
    cross_rel_ratio: float = 1.25   # ... and beats this direction's OWN best slots by this ratio
    cross_penalty: float = 10.0     # imported score is docked: another direction's evidence is weaker
    cross_min_episode_length: int = 60   # RePHO only vouches for the other direction from LONG rollouts

    # ── [ROLLBACK MARKER: track-buffer] kinematics update (SupMat Alg 3, lines 54-73) ────────────
    # Rewrites the TRACKING TARGET — what the reward compares against — at frames where a physically
    # successful rollout beat the retarget reference. This is the paper's headline mechanism, and our
    # reference has exactly the flaw it addresses (PyRoki retarget of ParaHome SMPL-X: object spawns
    # interpenetrating, contact normals 40-54 deg off the true surface).
    # OFF by default. It changes the reward mid-training, and its staging buffer is as large as the
    # state staging (~900 MB at 2048 envs), so both cost and effect must be an explicit choice.
    track_buffer: bool = False
    track_interval: int = 800       # 100 RePHO epochs x our PPO rollout length 8
    track_margin_self: float = 30.0   # SupMat n1: beat the worst slot of THIS direction by this
    track_ratio: float = 1.5          # RePHO's 3/2 relative-improvement conjunct (:1111)
    track_margin_cross: float = 60.0  # cross path margin (intermimic.py:1152)
    track_floor_self: float = 60.0    # absolute floor, self path (:1111)
    track_floor_cross: float = 90.0   # ... and cross path (:1152)
    track_death_tail: int = 10        # RePHO trims a died rollout by 10 frames (:1986)
    track_ref_contact_frac: float = 0.5
    # NOTE: RePHO has a SECOND swap path (load_ref_traj, intermimic.py:974) which adds a
    # global precondition — bail unless finish_rate > 0.85 over 70% of the clip, then replace
    # only where it is < 0.55. Not ported: the three conjuncts above are the load_run_val
    # path, and they already require the candidate to beat the buffer by margin AND ratio.   # discard unless the REFERENCE contacts over half the span
    track_start_step: int = 19000     # RePHO keeps the swap off for ~47% of its run (:1104)
    track_harvest_envs: int = 1       # RePHO validates with --num_envs 1, one per direction       # RePHO validates with --num_envs 1; noise-free, so a handful is enough
    # [/ROLLBACK MARKER: repho-cache] --------------------------------------------------------------

    # No-pretrain regime: RSI is seeded DIRECTLY from the PyRoki retarget reference (every frame is a valid
    # start via the where_ref reset path), so the pretrain-cache warm-start is OFF. The train cache still
    # supplies better (physical) restore states for frames it covers as training fills it.
    pretrain_cache_warmstart: bool = False   # was True (pretrain→train warm-start); now reference-seeded RSI
    # RSI START-FRAME CURRICULUM (cold-start mitigation). Diverse-from-step-0 RSI onto a COLD
    # value/policy/obs-scaler (no pretrain warm-start) ignites a PPO ratio explosion (policy loss
    # → e12–e21): the cold nets face the whole clip's state range at once → huge early value errors
    # / mis-normalized obs shove μ far in one update (amplified by the UNBOUNDED z_res + FSQ-flat
    # reward + kl_threshold=0). Confirmed via run diff: the explosion appeared exactly when
    # episode_length_s 5→3 (RSI start range upper 1→101 = diverse) + pretrain_cache_warmstart T→F.
    # This ramps the sampleable start-frame CEILING from ~frame 0 to the full trajectory over the
    # first rsi_curriculum_steps CONTROL STEPS, so the cold nets warm up on a narrow, well-conditioned
    # near-frame-0 distribution before deep frames appear. Unlike the old _reached_frame crawl this is
    # TIMESTEP-scheduled (not success-gated) so it ALWAYS reaches full range — no stuck-at-0.
    # 0 → disabled (diverse-from-step-0, previous behaviour). Tune ≳ the early explosion window
    # (~first 2k steps); default warms up then runs full RSI for the rest of training.
    rsi_curriculum_steps: int = 0
    ref_dt: float = 1.0 / 30.0


    # =========================================================================== #
    # Per-LINK contact reward (Option A / DexMachina — the single env contact map, from hand_contact.npz)
    # =========================================================================== #
    # The old fingertip-only contact map (obj-velocity gate + nearest-object-vertex: obj_contact_linvel/
    # angvel_thresh, contact_dist_threshold, contact_near_vertex_thresh, use_grounded_normal,
    # use_fingertip_to_vertex_dir, use_contact_point_gate, use_contact_normal_gate, contact_normal_tol) was
    # REMOVED 2026-07-22 — superseded by the per-link Option-A map (rew_contact_force + contact_force_cap
    # above; LINK_CONTACT_NAMES). Only the SPATIAL-gate tolerance remains, now reused by the per-link force
    # reward (a link's force counts only if the robot link is within this of its object-surface target).
    contact_match_dist: float = 0.03          # m, robot-link ↔ contact-target tolerance for the force gate
    ft_max_contact_points: int = 64           # per-link contact-data buffer cap. DERIVED bound, not a guess:
    #   PhysX reduces a convex-vs-convex manifold to ≤4 points, and parahome_convert_obj_to_usd.py caps the
    #   object convex decomposition at max_convex_hulls=16, so a link can straddle ≤16 sub-hulls → ≤16×4 = 64
    #   contacts. ContactSensor buffer overflow is a HARD device-side assert, so the cap MUST upper-bound the
    #   real count (tied to the hull cap at the source). If max_convex_hulls changes, set this to hulls×4.
    #   RAISED 32 → 64 on 2026-07-30 to satisfy that derivation. It had been 32 (half the bound) while the
    #   14 RIGID object USDs were additionally still cooked from 2026-07-07, i.e. BEFORE the max_convex_hulls
    #   =16 line existed — `maxConvexHulls` was simply NOT AUTHORED on their collider prims, so PhysX used its
    #   schema default of 32 (generatedSchema.usda:577) and the real worst case was 32×4 = 128, four times the
    #   configured cap. Verified by reading the Sdf layer text of every objects/*/Props/instanceable_meshes.usd
    #   (config.yaml is NOT a reliable record: one file per DIRECTORY, so a later <obj>_ctx.usd cook overwrites
    #   the <obj>.usd entry). All 22 objects were re-cooked with `--rigid-only --overwrite` and now author
    #   hulls=16 / verts=64, which makes 16×4 = 64 the true bound again.

    # =========================================================================== #
    # Observation scaling + action smoothing (per-group EMA + optional delta)
    # =========================================================================== #
    vel_obs_scale: float = 0.2             # scale on all angular + joint velocities in obs
    # Per-group EMA alphas live in JOINT_GROUPS[*]["ema_alpha"] (legs/waist/arms 0.3, hands 0.5).
    # NOTE (SONIC mode = this env): the per-group delta switches + residual_action below are BYPASSED.
    # The 29 body DOF come from the FROZEN SONIC decoder (no EMA/delta); the 36 hands use ABSOLUTE action
    # with the hands EMA (α=0.5). These fields govern ONLY the inherited non-SONIC fallback path (dead here).
    # Fallback semantics (kept for that path): delta switch OFF → absolute per-group EMA; ON → integrate
    # raw·scale (EMA-smoothed by *_delta_smoothing) into a clamped target. scale = rad/step (raw∈[-1,1]).
    leg_delta_action: bool = True
    leg_delta_scale: float = 0.25
    leg_delta_smoothing: float = 1.0
    waist_delta_action: bool = True
    waist_delta_scale: float = 0.25
    waist_delta_smoothing: float = 1.0
    arm_delta_action: bool = True
    arm_delta_scale: float = 0.25
    arm_delta_smoothing: float = 1.0
    hand_delta_action: bool = True
    hand_delta_scale: float = 0.4
    hand_delta_smoothing: float = 1.0

    # RESIDUAL-POLICY mode (hybrid with the delta integrator above). When True AND the retarget joint
    # trajectory is present, the per-group delta integrator gains a REFERENCE-FEEDFORWARD term so the
    # commanded target tracks the retargeted joints (target ≈ ref_joints[frame] + accumulated policy
    # correction); the policy then learns only the physics/contact/balance CORRECTION on top of the
    # reference. No offset window — the correction is bounded by joint limits (anti-windup via the delta
    # clamp) and kept small by the delta clamp. Falls back to free-running delta if the
    # retarget joints are absent (guarded). Re-pretrain when toggling (action semantics change).
    residual_action: bool = True   # BYPASSED in SONIC mode (governs only the non-SONIC fallback path).
    # PER-STEP residual: target = clamp(ref_joints[frame] + residual_scale·a, limits), scaled PER GROUP.
    # Body (legs+waist+arms) gets a tighter residual (closer reference tracking); hands get a wider one
    # (grasp adaptation needs more finger authority than the reference provides).
    # BOTH are read ONLY on the use_sonic=False fallback path (_pre_physics_step → residual_action
    # branch), via _residual_scale_t. Under the default use_sonic=True the 29 body DOF come from the
    # frozen SONIC decoder and the hands are an ABSOLUTE action, so neither value affects training —
    # they are not dead code, just inactive in the shipped configuration.
    residual_scale_body: float = 0.50    # legs (12) + waist (3) + arms (14) = 29 DOF
    residual_scale_hands: float = 0.50   # bimanual Shadow fingers (36 DOF)

    # =========================================================================== #
    # Frozen SONIC body prior (GEAR universal-token whole-body controller)
    # =========================================================================== #
    # The 29 G1 body DOF are driven by a FROZEN SONIC decoder; the policy outputs z_res(64) added
    # to SONIC's FSQ latent BEFORE quantization (GRAIL Eq.6, λ=residual_scale_latent) then decoded.
    # SONIC runs at 50 Hz (sim.dt 1/200 × decim 4) and consumes a 10-frame proprio history + a
    # 10-frame future SMPL reference window (sonic_smpl_file, produced by parahome_smpl_for_sonic.py
    # at control_fps). The env resamples ALL per-frame references 30 fps → control_fps at load so
    # one control step == one reference frame. Built in _post_init_buffers (needs env_isaaclab +
    # gear_sonic + vector_quantize_pytorch installed).
    use_sonic: bool = True
    sonic_config_path: str = "/home/peunsu/workspace/GR00T-WholeBodyControl/sonic_release/config.yaml"
    sonic_ckpt_path: str = "/home/peunsu/workspace/GR00T-WholeBodyControl/sonic_release/last.pt"
    # [ROLLBACK MARKER: lambda-vs-fsq-bin] λ on z_res (GRAIL pre-quantization latent residual scale).
    # 0.10 -> 0.03. The SONIC quantizer is FSQ with 32 levels, whose bin width near the operating
    # point is 0.0645 in latent units (verified against the installed vector_quantize_pytorch.FSQ:
    # the env's _fsq_level helper matches it exactly). At λ=0.10 the policy's own exploration noise
    # was λ·σ = 0.10 × 0.673 = 0.067 = 1.04 BINS, i.e. larger than one quantization step, so the
    # decoder's input was re-drawn every control step and the residual's MEAN could not decide which
    # bin was hit. Measured in sim: 47 of 64 latent dims changed level per step (Jerk / fsq_flip_run
    # ≈ 0.74), and the commanded body target moved 0.28 rad/step — 8x the reference motion's mean
    # per-step change and 1.6x its clip-wide MAXIMUM. `Error / body_kpts` was flat across 10k steps
    # on two separate runs: the body was being driven by quantization noise, not by the reference.
    # At λ=0.03 the jitter is 0.31 bins (below one step, so the mean decides the bin) and the reach
    # at the ±5 action clip is 2.32 bins.
    # WHY λ AND NOT σ: KL is computed in ACTION space, so λ (an env-side scaling) leaves it
    # untouched, while shrinking σ inflates KL by 1/σ². That matters here because KLAdaptiveLR is
    # ALREADY saturated — measured LR 1.26e-05 against a configured 3.0e-04 (4%), with kl_fp64
    # ≈ 0.022 sitting permanently above kl_threshold 0.016. Lowering σ would throttle it further,
    # and the LR is shared with the hand action, which is the part that is currently learning.
    # WHY NOT AN EMA ON z_res: PPO would score the sampled z_res while the env applied the smoothed
    # one, the accumulator would be hidden state absent from the obs, it adds ~5 steps of lag, and it
    # needs a reset-time seed — the same class of fabrication that caused the reset-transient bug.
    # TESTED AND REVERTED to 0.10. λ=0.03 did exactly what the analysis predicted mechanically —
    # fsq_flip_run 0.744 -> 0.356 and the commanded per-step body motion 0.284 -> 0.146 rad, i.e.
    # below the reference clip's own MAXIMUM — but the task got much worse, not better:
    #   reward @10k   22.5 -> 5.6      episode_len @10k   68.4 -> 35.2
    #   Error / body_kpts   0.132 -> 0.132  (UNCHANGED)
    # Two conclusions. (1) The jitter was NOT what kept body tracking flat; that hypothesis is dead.
    # (2) In a QUANTIZED latent the bin crossings ARE the exploration — the decoder output is
    # literally constant within a bin — so halving the crossing rate halved the body's exploration
    # and the reward fell with it. The twitching and the exploration are the same mechanism, which
    # is why scaling λ trades one for the other and cannot fix both.
    # If revisiting: λ=0.05 (jitter 0.52 bins, reach 3.87) is the untested middle, but the reward
    # cost is likely to scale the same way. The lever that separates the two would have to make the
    # residual's MEAN move the bin while the per-step noise does not — σ, not λ.
    residual_scale_latent: float = 0.10
    control_fps: float = 50.0            # resample reference 30 fps → this; MUST match parahome_smpl_for_sonic TGT_FPS
    sonic_smpl_file: str = "sonic_smpl_50fps.npz"   # SONIC SMPL encoder arrays (sibling of the retarget npz)


    # =========================================================================== #
    # Data — reference clip (ParaHome). Keypoint targets come from the SMPLX tree
    # (available now); per-frame retargeted G1 joints (if present in the retarget tree)
    # seed reset poses, else fall back to the G1_SHADOW_CFG standing pose.
    # =========================================================================== #
    dataset_root: str = str(_DATA_DIR / "processed" / "parahome")   # absolute (package data dir)
    smplx_subdir: str = "smplx"            # keypoint/object reference tree (produced by parahome.py)
    # [ROLLBACK MARKER: retarget-joint-order] ----------------------------------------------------
    # Match the retargeted joint columns to the env's action joints BY NAME instead of trusting that
    # the two orders line up positionally. They did not: g1_shadow_joint_order.json is a static dump
    # of the robot's PhysX DOF order, G1_shadow.usd was rebuilt a day after that dump, and 24 of the
    # 65 slots ended up crossed — all hands (MF<->TH, FF<->RF at J1/J2/J3, both sides), so the middle
    # finger was driven by the thumb's near-zero joint and stuck out straight in every rollout while
    # legs/waist/arms stayed correct. False restores the raw positional read (pre-fix baseline).
    remap_ref_joint_order: bool = True
    # [/ROLLBACK MARKER: retarget-joint-order] ---------------------------------------------------
    retarget_subdir: str = "g1_shadow"     # per-frame G1 joint refs (produced by retargeting; optional)
    retarget_file: str = "trajectory_pyroki.npz"  # retarget npz filename under the tree (PyRoki output; g1_joint_pos/g1_root_pose)
    clip_class: str = "single_rigid"       # single_rigid | single_articulated | ...
    clip_name: str = ""                    # "" → auto-pick the first available clip in clip_class
    # (clip selection is dataset_root + smplx_subdir/retarget_subdir + clip_class + clip_name; the old
    #  data_subset / dataset_dir aliases were never read by the env and are gone.)
