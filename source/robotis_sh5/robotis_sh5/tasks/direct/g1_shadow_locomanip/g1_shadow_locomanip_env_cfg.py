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
    17: "right_ankle_roll_link",      # jRightAnkle (plant/contact body)
    18: "right_ankle_roll_link",      # jRightBallFoot (+ BODY_KPT_OFFSETS toe offset)
    19: "left_hip_pitch_link",        # jLeftHip (leg-root body)
    20: "left_knee_link",             # jLeftKnee
    21: "left_ankle_roll_link",       # jLeftAnkle (plant/contact body)
    22: "left_ankle_roll_link",       # jLeftBallFoot (+ BODY_KPT_OFFSETS toe offset)
}

# Per-body-keypoint LOCAL offset (in the target G1 link frame), added (rotated) to the body
# origin before tracking. G1 has no toe/ball body, so BallFoot(18/22) share ankle_roll_link
# with Ankle(17/21) — WITHOUT this offset they would be the SAME point and the ball keypoint
# would carry no information. FOOT_TOE_OFFSET measured from the G1 ankle_roll_link foot mesh
# (local +X = forward; toe tip at +0.142 m, sole at z≈−0.03 m → ball ≈ +0.11 m forward, sole).
FOOT_TOE_OFFSET: list[float] = [0.11, 0.0, -0.03]
# jT9T8 (upper spine, idx 4) ≠ torso_link ORIGIN (at the waist, ~0.27 m lower). Track a point on
# G1's UPPER torso (torso_link + this local offset ≈ upper chest) so the correspondence is
# anatomically aligned instead of ~0.27 m off. Measured: jT9T8 in torso_link local frame at a
# standing posture. (env robot-kpt = torso_link origin + offset; retarget targets the same frame.)
TORSO_KPT_OFFSET: list[float] = [-0.033, 0.0, 0.274]
BODY_KPT_OFFSETS: dict[int, list[float]] = {
    4: TORSO_KPT_OFFSET, 18: FOOT_TOE_OFFSET, 22: FOOT_TOE_OFFSET,
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

# Number of tracked keypoints (obs/reward): body 16 (BODY_KPTS entries) + hands
# (wrist=1, index/mid/ring/pinky=4, thumb=3 → 20 per hand) × 2 = 56.
N_BODY_KPTS = len(BODY_KPTS)                      # 16
N_HAND_KPTS_PER_HAND = sum(len(v["parahome"]) for v in HAND_CHAIN.values())  # 20
N_TRACK_KPTS = N_BODY_KPTS + 2 * N_HAND_KPTS_PER_HAND                          # 56


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
            max_depenetration_velocity=1.0,
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
        # ---- G1 body: adopt the canonical IsaacLab G1 locomotion actuator gains
        #      (isaaclab_assets G1_CFG / G1_MINIMAL_CFG), mapped onto the 29-DOF joint names.
        #      Switched legs/feet from DCMotor to ImplicitActuatorCfg to match that proven
        #      locomotion config (drops the DCMotor torque-speed saturation model). The prior
        #      waist=5000 / arms=3000 stiffness were far too stiff for a free-floating balance
        #      task; the reference uses torso=200 / arms=40. Shadow fingers are unchanged (the
        #      reference config does not cover them).
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint",
                              ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit_sim=300.0, velocity_limit_sim=100.0,
            stiffness={".*_hip_yaw_joint": 150.0, ".*_hip_roll_joint": 150.0,
                       ".*_hip_pitch_joint": 200.0, ".*_knee_joint": 200.0},
            damping={".*_hip_yaw_joint": 5.0, ".*_hip_roll_joint": 5.0,
                     ".*_hip_pitch_joint": 5.0, ".*_knee_joint": 5.0},
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=20.0,
            stiffness=20.0, damping=2.0, armature=0.01,
        ),
        # G1_CFG groups torso_joint with legs at stiffness 200 / damping 5; the 29-DOF has 3
        # waist joints — apply the same torso gains to all three.
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim=300.0, velocity_limit_sim=100.0,
            stiffness=200.0, damping=5.0, armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                              ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
            effort_limit_sim=300.0, velocity_limit_sim=100.0,
            stiffness=40.0, damping=10.0, armature=0.01,
        ),
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
class G1ShadowLocomanipEnvCfg(DirectRLEnvCfg):
    # --- Robot -----------------------------------------------------------------
    # Composite G1 + bimanual Shadow (validated). FLOATING base (fix_root_link=False in
    # G1_SHADOW_CFG.spawn.articulation_props) for locomotion.
    robot_cfg: ArticulationCfg = G1_SHADOW_CFG

    # --- Control / sim ---------------------------------------------------------
    decimation: int = 4                 # 30 Hz control (120/4)
    episode_length_s: float = 5.0       # → num_frame_chunk ≈ 150 @30 fps
    action_space: int = ACTION_DIM      # 65
    # obs = A proprio(200) + B reference(352) + C object/contact/history(151) = 703.
    #   A: root_h(1)+root_ori6d(6)+root_linvel(3)+root_angvel(3)+proj_grav(3)+jpos(65)+jvel(65)
    #      +palm_ori6d(12)+palm_linvel(6)+palm_angvel(6)+fingertip_vel(30)  [bimanual palm state + ft vel]
    #   B: kpts(56×3)+delta(56×3)+ref_root_h(1)+ref_root_ori6d(6)+delta_root_pos(3)+delta_root_ori6d(6)
    #      (NO phase — like the grasp env; look-ahead deltas convey progress)
    #   C: obj(15)+dobj(9)+delta_ft_obj(30)+artic_reserved(8)+future_contact(10)+ref_foot_contact(2)
    #      +ft_force(10)+foot_force(2)+prev_action(65)   [delta_ft_obj = object-local ft→target offset]
    #   FOOT CONTACT mirrors fingertips: reference future_foot_contact (ref_foot_contact, C) + current
    #   robot sole-normal foot force (foot_force, C). The duplicate robot foot-contact FLAG that used to
    #   sit in BOTH A and C was removed (was 705 → 703; A lost its foot(2), C's foot flag → foot force).
    observation_space: int = 703        # asserted against the assembled obs in _get_observations
    state_space: int = 0

    # Enlarged GPU contact buffers (mirrors the grasp envs). The default buffers overflow for the
    # 78-body G1+bimanual-Shadow robot + object with track_contact_points on the fingertips
    # (ContactSensor._unpack_contact_buffer_data device-side assert) once the fingers press on the
    # object. Larger gpu_max_rigid_patch_count / aggregate-pair capacities fix it.
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.80665),
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            gpu_max_rigid_patch_count=4096 * 4096,
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
    debug_vis_num_envs: int = 4096
    # env-fixed video/viewer camera zoom: scales the eye offset from the robot (smaller = closer).
    # 1.0 = the default close framing aimed at chest/hands; set 0.6 to zoom in further on the hands.
    viewer_zoom: float = 1.0

    # --- Body links used as tracking anchors / balance / contact ---------------
    root_body_name: str = "pelvis"
    torso_body_name: str = "torso_link"
    foot_body_names: list = ["left_ankle_roll_link", "right_ankle_roll_link"]  # confirmed (inspect_g1_asset.py)
    left_wrist_body_name: str = "left_wrist_yaw_link"
    right_wrist_body_name: str = "right_wrist_yaw_link"
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
    rew_body_kpts: float = -1.5        # mean over 12 NON-FOOT body kpts (feet split to rew_foot_track);
    #   -1.5/12 = -0.125/kpt, same per-kpt weight as the old -2.0/16.
    rew_foot_track: float = -2.0       # mean over 4 FOOT kpts (ankle+ball, L/R) — split out of body_kpts and
    #   UP-weighted (foot tracking is most important for loco-manip; in the 16-kpt mean it was diluted 4/16).
    #   -3.0/4 = -0.75/kpt → the HIGHEST per-kpt tracking weight (above fingertip -5.2/10 = -0.52).
    rew_hand_kpts: float = -1.76        # mean over 40 finger-chain kpts (dexterity)
    rew_fingertip: float = -5.2        # mean over 10 bimanual pads ≈ grasp -5.2 (both are means)
    # B. Locomotion / root — NEW (root_pos/root_ori inside the clamped tracking_penalty)
    rew_root_pos: float = -2.5
    rew_root_ori: float = -1.0
    rew_root_linvel: float = 0.0       # UNUSED — removed from reward (grasp has no root-velocity
    rew_root_angvel: float = 0.0       # terms; redundant finite-diffs of root_pos/ori, and raw
    #                                    robot angvel is very noisy at reset → dominated the penalty)
    # C. Balance / feet — NEW
    rew_upright: float = 0.0           # UNUSED — removed from reward (user: upright not needed;
    #                                    dense body/root keypoints already constrain posture)
    rew_foot_contact: float = 0.2      # weight of the foot-contact QUALITY penalty rew_foot_contact*(q-1) ∈ [-0.5,0].
    #   Per foot q∈[0,1]: PLANTED → (sole-normal force/foot_force_cap, clamped) × (GATE); SWING → (1-contact).
    #   GATE depends on foot_gate_mode. Rewards pressing FLAT at the RIGHT place (fingertip-force analog);
    #   heel-only / dragged / wrong-spot contact scores low even though it touches the ground.
    rew_foot_slip: float = -0.1
    foot_force_cap: float = 100.0       # N, sole-normal force that saturates the foot reward. Set to 50
    #   (observed firm-plant forces ~50-95 N sole-normal) so a solid flat plant reaches full reward while
    #   a light/heel/tilted contact scales down. Tune per training (raise if it saturates too easily on
    #   glancing contact; lower for a firmer requirement).
    # foot-contact position GATE (which "right place" a planted foot is rewarded at):
    #   "kpt"    → within foot_kpt_gate of the foot's MOVING reference keypoints (ankle+ball mean). The
    #              reference feet pivot/re-orient during a turn, so this lets the robot follow instead of
    #              being pinned — fixes "can't rotate while planted". DEFAULT.
    #   "anchor" → within foot_anchor_tol of the xy where the foot FIRST touched down (frozen). Rollback.
    foot_gate_mode: str = "kpt"        # "kpt" | "anchor"
    foot_kpt_gate: float = 0.10        # m, per-foot mean (ankle+ball) kpt error below → planted foot is rewarded.
    #   Data-driven: retarget floor for this quantity is ~7.6 cm mean / 14.7 cm p95 (planted frames), so 0.15
    #   (≈ p95 ≈ 2×mean, = term_ft_err, half of term_body_kpt_err) lets a well-placed foot pass while a foot
    #   >15 cm off its reference target scores 0. Tune like foot_force_cap.
    foot_anchor_tol: float = 0.05      # m (foot_gate_mode="anchor" only): planted foot may drift this far from its contact-onset xy
    # foot FLATNESS soft-ramp factor on the PLANTED foot reward (multiplies force_n·gate). Uses the sole
    # tilt from vertical (sin θ = horizontal component of the sole-up axis). flat_factor = 1 for tilt ≤ tol,
    # ramps linearly to 0 at limit, 0 beyond → a heel-only / toe-only (forefoot) plant earns little/no
    # foot reward. Decouples "flat?" from the force magnitude (force_n saturates so it can't encode tilt).
    foot_flat_tol_deg: float = 10.0    # ≤ this tilt = full flatness factor (deadzone for natural micro-tilt)
    foot_flat_limit_deg: float = 15.0  # ≥ this tilt → flatness factor 0 (foot reward killed)
    # D. Object manipulation — reused from grasp (inside the clamped tracking_penalty)
    rew_obj_pos: float = -4.26
    rew_obj_rot: float = -1.0
    rew_obj_artic: float = -1.0        # articulation DOF tracking (articulated objects)
    rew_fingertip_force: float = 1.0
    # E. Alive / regularization
    # rew_alive sets the tracking_penalty clamp floor (-rew_alive). Above the grasp default (1.5)
    # because the grouped full-body/bimanual tracking penalty is larger; kept high enough that good
    # tracking stays unsaturated (else the clamp kills the gradient) while flooring bad steps at 0.
    # User set 2.0 (from 4.0). Watch Reward/tracking_clamp_frac — if it climbs, raise back toward 4.
    rew_alive: float = 2.0
    rew_action_reg_legs: float = -0.004
    rew_action_reg_arms: float = -0.004
    rew_action_reg_hands: float = -0.004
    rew_pose_reg: float = -0.001
    rew_action_rate: float = -0.01     # smoothness (NEW)

    # =========================================================================== #
    # Termination — DEVIATION FROM REFERENCE (so reference crouch/bend never triggers)
    # =========================================================================== #
    # All gates compare robot vs the reference pose at that frame. NO separate root pos/tilt/height
    # gates: the mean body-keypoint error subsumes them (a root shift moves every keypoint, a tilt
    # rotates the far keypoints away, a fall drives foot/pelvis keypoints off) — one clean body gate.
    term_body_kpt_err: float = 0.50        # m, mean body-keypoint tracking error (covers root/fall).
    term_obj_pos_err: float = 0.15         # m, active-object position tracking error (= grasp max_obj_pos_err; was loosened to 0.20, now grasp-parity)
    term_obj_rot_err: float = 0.75         # rad, active-object rotation tracking error (= grasp max_obj_rot_err; was 0.80)
    # Hand deviation gates (mirror grasp's ft_err / wrist termination; bimanual). Generous vs
    # grasp (0.15) because full-body hands are secondary during locomotion; set huge to disable.
    # Wrist POSITION deviation is caught via the finger-chain keypoints (they hang off the wrist).
    term_hand_kpt_err: float = 0.15        # m, mean finger-chain (hand) keypoint tracking error (grasp-parity: grasp's hand-region tol = max_ft_mean_err/max_wrist_pos_err 0.15; was loosened to 0.25)
    term_ft_err: float = 0.15              # m, mean fingertip (pad) tracking error (= grasp max_ft_mean_err 0.15; was 0.20)
    enable_hand_termination: bool = True   # gate the two hand terms above
    # Wrist/palm ROTATION deviation gate (mirrors grasp max_wrist_rot_err). The dense finger chains
    # constrain palm orientation only weakly via POSITION; an explicit quat gate catches a flipped/
    # twisted palm the fingers can't. Reference palm orientation = retarget g1_palm_quat (Kabsch palm
    # pose = robot0_{l,r}_palm body frame), compared DIRECTLY to body_quat_w[palm] per hand (no
    # landmark conversion — the left palm is a geometric mirror so the grasp _palm_to_landmark does
    # NOT apply). Worst-of-two-hands. Auto-inert if the clip has no retarget palm reference.
    enable_wrist_rot_termination: bool = True
    term_wrist_rot_err: float = 0.75       # rad, per-hand palm-rotation deviation (= grasp max_wrist_rot_err)
    # No grace period: frame-0 reset uses the retargeted (pink-IK) pose that already matches the
    # reference keypoints, so tracking gates are valid from step 0 (RSI restores matching states).
    termination: bool = True               # master switch (False during eval/warm-up)

    # =========================================================================== #
    # Objects — active (dynamic, tracked) vs others (frozen for collision only)
    # =========================================================================== #
    num_active_objects: int = 1            # single-object clips first; other objects kinematic-frozen
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

    # =========================================================================== #
    # Adaptive frame-sampling curriculum + pretrain-cache RSI (reused from grasp)
    # =========================================================================== #
    adaptive_sampling: bool = True
    failure_weighted_sampling: bool = True   # TRAIN: True (failure-weighted). PRETRAIN overrides → False (uniform).
    adaptive_alpha: float = 0.001
    adaptive_uniform_ratio: float = 0.1
    adaptive_back_seconds: float = 1.2
    pretrain_cache_warmstart: bool = True    # Re-enabled: the early-termination cause was the obs-scaler
    #   RESET on pretrain→train transfer (loaded policy fed RAW obs → diverged), NOT the warm-start.
    #   Fixed in train.py _load_partial_checkpoint (floored scaler transfer). Warm-start gives
    #   reference-matching reset poses (lower initial wrist_rot). Set False for vanilla RSI (rollback).
    uniform_sampling_steps: int = 2000
    ref_dt: float = 1.0 / 30.0

    # --- State-cache quality gate (mirrors grasp _save_state_cache): a frame's state is cached
    # only while tracking is CONTINUOUSLY "good enough" since reset (self._enough_continued), so
    # RSI restores from good states, not barely-alive ones. Thresholds are TIGHTER than the
    # termination gates. body+fingertip are the always-present bars; object uses a frame-dependent
    # (start-loose / early / late) phase like grasp. When has_object is False (pretrain / no-object
    # clips) the object conditions are trivially satisfied (obj errs = 0) → gate = body+ft only.
    # enough_* now use the SAME VALUES as grasp (user: do not loosen). Matches grasp's gate exactly:
    # fingertip + object phase only (NO body/hand enough thresholds — grasp never gated on those).
    enough_ft_threshold: float = 0.10          # m, mean fingertip err (= grasp enough_ft_threshold; was loosened to 0.13)
    enough_obj_threshold: float = 0.085        # m, early-phase obj pos err (= grasp; was 0.11)
    enough_obj_rot_threshold: float = 0.425    # rad, early-phase obj rot err (= grasp; was 0.45)
    enough_obj_threshold_late: float = 0.05    # m, late-phase obj pos err (= grasp; was 0.07)
    enough_obj_rot_threshold_late: float = 0.25  # rad, late-phase obj rot err (= grasp; was 0.27)
    # Floating-base cache quality bars (analog of grasp pretrain's fixed-base wrist bars): only
    # cache a frame if the WHOLE BODY / ROOT also track well, not just the fingertips. Needed
    # because with has_object=False the object phase-gate is a tautology → the gate would collapse
    # to fingertip-only, letting a drifted torso with a compensating arm poison the RSI cache.
    # Default inf = OFF (TRAIN unchanged); the PRETRAIN cfg tightens them (obj gate is inert there).
    cache_body_bar: float = 0.30       # < term_body_kpt_err 0.25 (~0.65×); seed body err ~0.066 passes
    cache_root_pos_bar: float = 0.10
    cache_root_rot_bar: float = 0.30

    # =========================================================================== #
    # Contact-map fingertip-force + future-contact (mechanism matches robotis_shadow_grasp)
    # =========================================================================== #
    # future_contact (GR is_contact) per fingertip = the ACTIVE OBJECT is being manipulated
    #   (obj linvel > obj_contact_linvel_thresh OR obj angvel > obj_contact_angvel_thresh)
    #   AND the fingertip is near the object (nearest-vertex gate with mesh, else object-centre
    #   distance < contact_dist_threshold). NOTE: velocity is the OBJECT's, not the fingertip's.
    obj_contact_linvel_thresh: float = 0.05   # m/s
    obj_contact_angvel_thresh: float = 0.25   # rad/s
    contact_dist_threshold: float = 0.05      # m, fingertip→object-centre (centroid fallback / no mesh)
    contact_near_vertex_thresh: float = 0.025  # m, fingertip→nearest-vertex relative gate (with mesh)
    # contact-force reward gating (grounded normal + actual-contact-position gate)
    use_grounded_normal: bool = True          # project force on the contact-vertex force direction
    # force-projection DIRECTION at the contact vertex (matches grasp's default). True = unit(reference
    # fingertip pad − nearest object mesh vertex), i.e. the surface→finger direction (≈ outward normal
    # but auto-pointing to the finger side, no sign ambiguity); False = the object mesh vertex normal.
    use_fingertip_to_vertex_dir: bool = True
    ft_max_contact_points: int = 64           # per fingertip-object contact-data buffer. This is a
    #   DERIVED upper bound, NOT a tuned guess: PhysX reduces a convex-vs-convex manifold to ≤4
    #   points, and parahome_convert_obj_to_usd.py caps the object's convex decomposition at
    #   max_convex_hulls=16, so a fingertip can straddle at most 16 sub-hulls → ≤16×4 = 64 contacts.
    #   (grasp uses a single ConvexHull → ≤4, so its sensors use the PhysX default of 4.) The
    #   ContactSensor buffer overflow is a HARD device-side assert (unchecked gather, no truncation),
    #   so the cap MUST upper-bound the actual contact count — which is why it is tied to the hull cap
    #   at the collider source, not inflated blindly. GPU cost is tiny (~64×10×num_envs×32 B ≈ 80 MiB
    #   at 4096 envs). If the converter's max_convex_hulls changes, change this to hulls×4 to match.
    use_contact_point_gate: bool = True       # gate by ACTUAL contact position near the contact vertex
    use_contact_normal_gate: bool = True      # split displacement normal/tangential, reject back-face
    contact_match_dist: float = 0.03          # m, tangential/isotropic contact-position tolerance
    contact_normal_tol: float = 0.01          # m, allowed behind-surface (normal) offset

    # =========================================================================== #
    # Observation scaling + action smoothing (per-group EMA + optional delta)
    # =========================================================================== #
    vel_obs_scale: float = 0.2             # scale on all angular + joint velocities in obs
    # Per-group EMA alphas live in JOINT_GROUPS[*]["ema_alpha"] (legs/waist 0.3, arms 0.2, hands 0.5).
    # Optional delta(residual)-action switch PER GROUP (velocity command → integrated target). A group
    # with its switch OFF uses the absolute per-group EMA; ON integrates raw·scale (EMA-smoothed by
    # *_delta_smoothing) into a clamped target. Mirrors robotis_shadow_grasp_rsi arm/hand delta-action;
    # now available for ALL 4 groups (legs / waist / arms / hands). Re-pretrain when toggling any switch
    # (action semantics change). scale = rad per step (raw∈[-1,1]); smoothing = EMA α on the velocity cmd (1.0 = none).
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
    # clamp) and kept small by rew_pose_reg (re-based to ref). Falls back to free-running delta if the
    # retarget joints are absent (guarded). Re-pretrain when toggling (action semantics change).
    residual_action: bool = True
    # PER-STEP residual: target = clamp(ref_joints[frame] + residual_scale·a, limits), scaled PER GROUP.
    # Body (legs+waist+arms) gets a tighter residual (closer reference tracking); hands get a wider one
    # (grasp adaptation needs more finger authority than the reference provides).
    residual_scale_body: float = 0.50    # legs (12) + waist (3) + arms (14) = 29 DOF
    residual_scale_hands: float = 0.50   # bimanual Shadow fingers = 36 DOF

    # =========================================================================== #
    # Balance / contact heuristics (reference has no explicit foot-contact labels)
    # =========================================================================== #
    foot_contact_force_thresh: float = 1.0     # N, robot foot in contact (sole-normal-projected ground force)
    ref_foot_planted_height: float = 0.05      # m, ref foot-kpt height below → planted
    ref_foot_planted_velz: float = 0.10        # m/s, ref foot-kpt |vz| below → planted

    # =========================================================================== #
    # Data — reference clip (ParaHome). Keypoint targets come from the SMPLX tree
    # (available now); per-frame retargeted G1 joints (if present in the retarget tree)
    # seed reset poses, else fall back to the G1_SHADOW_CFG standing pose.
    # =========================================================================== #
    dataset_root: str = str(_DATA_DIR / "processed" / "parahome")   # absolute (package data dir)
    smplx_subdir: str = "smplx"            # keypoint/object reference tree (produced by parahome.py)
    retarget_subdir: str = "g1_shadow"     # per-frame G1 joint refs (produced by retargeting; optional)
    retarget_file: str = "trajectory_pyroki.npz"  # retarget npz filename under the tree (PyRoki output; g1_joint_pos/g1_root_pose)
    clip_class: str = "single_rigid"       # single_rigid | single_articulated | ...
    clip_name: str = ""                    # "" → auto-pick the first available clip in clip_class
    data_subset: str = "single_object"     # single_rigid + single_articulated first
    dataset_dir: str = "data/processed/parahome/g1_shadow"  # (legacy alias; retarget tree)
