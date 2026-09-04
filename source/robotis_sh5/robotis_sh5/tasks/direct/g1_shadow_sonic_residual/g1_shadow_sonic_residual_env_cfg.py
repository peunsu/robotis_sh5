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
# ── [ROLLBACK MARKER: hand-kpt-align] 손 키포인트 대응 정정 (2026-08-15) ─────────────────────
# 이전 대응은 사람 관절과 로봇 링크가 한 칸씩 어긋나 있었습니다. 원인은 Shadow 손가락에서
# <finger>knuckle 과 <finger>proximal 의 링크 원점이 **같은 위치**라는 점입니다(측정: 간격
# 0.00 cm — 둘 사이 관절 J2 는 순수 회전이라 병진 오프셋이 없습니다). 그래서 옛 대응은
# 3.67 cm 떨어진 사람의 MCP 와 PIP 를 항상 같은 지점에 있는 두 링크에 요구했고, 원리적으로
# 동시에 만족될 수 없었습니다. 그 어긋남이 뒤로 밀리며 DIP 도 한 칸 틀어졌습니다.
#
# 리타게팅된 레퍼런스에서 잰 키포인트별 오차(왼손, 옛 대응):
#     MCP 1.60 / PIP 4.80 / DIP 2.94 / TIP 2.23 cm   <- 가운데 둘만 2~3배
#
# 로봇 손가락의 구분되는 위치는 3곳(A=knuckle=proximal, B=middle, C=distal)이고, 여기에 pad
# 오프셋을 더하면 4번째 점(C+pad = 손끝)이 생깁니다. 그래서 distal 을 두 번 씁니다 —
# 오프셋 0 이면 DIP, pad 오프셋이면 TIP. 키포인트 수가 20/손으로 유지되므로 관측 차원(54)은
# 그대로입니다. pad 크기 1.85 cm 가 사람 DIP->TIP 1.79 cm 와 일치하는 것이 이 대응의 근거입니다.
#
# 엄지는 사람이 3점(22,23,24=TIP)뿐이라 마디 길이 비율로 후보를 비교해 정했습니다
# (네 손가락의 로봇/사람 비율 1.03~1.26 이 기준):
#     옛 대응 22->thproximal,23->thmiddle : 비율 1.39 / 1.92  편차 0.980
#     채택   22->thmiddle,  23->thdistal  : 비율 1.17 / 0.79  편차 0.388   <- 2.5배 일치
#
# `offset` 은 키포인트별입니다. 예전에는 링크 이름으로 FINGERTIP_OFFSETS 를 조회해서 같은 링크를
# 두 번 쓸 수 없었습니다. ZERO/PAD 는 아래에서 좌우 미러까지 적용해 만듭니다.
# 되돌리기: 이 블록을 git 이전 판으로 (관측 차원은 어느 쪽이든 54라 형태는 호환되지만, 키포인트의
# 의미가 바뀌므로 체크포인트는 재학습해야 합니다).
HAND_CHAIN: dict[str, dict] = {
    #             [ParaHome idx]              [Shadow body]                 [offset key]
    "wrist":  {"parahome": [0],              "shadow": ["palm"],           "pad": [False]},
    #             [MCP, PIP, DIP, Tip]        [knuckle, middle, distal, distal(+pad)]
    "index":  {"parahome": [18, 19, 20, 21], "shadow": ["ffknuckle", "ffmiddle", "ffdistal", "ffdistal"],
               "pad": [False, False, False, True]},
    "middle": {"parahome": [14, 15, 16, 17], "shadow": ["mfknuckle", "mfmiddle", "mfdistal", "mfdistal"],
               "pad": [False, False, False, True]},
    "ring":   {"parahome": [10, 11, 12, 13], "shadow": ["rfknuckle", "rfmiddle", "rfdistal", "rfdistal"],
               "pad": [False, False, False, True]},
    "pinky":  {"parahome": [6, 7, 8, 9],     "shadow": ["lfknuckle", "lfmiddle", "lfdistal", "lfdistal"],
               "pad": [False, False, False, True]},
    #             [MCP, IP, Tip]              [middle, distal, distal(+pad)]
    "thumb":  {"parahome": [22, 23, 24],     "shadow": ["thmiddle", "thdistal", "thdistal"],
               "pad": [False, False, True]},
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
            # [ROLLBACK MARKER: depen-vel] Cap on how fast PhysX pushes an overlap apart. This is NOT
            # a bounce — restitution is 0.0 on the robot, the object and every context collider, so
            # nothing here is elastic. It is the depenetration impulse, and the robot absorbs ALL of
            # it because the supports are spawned kinematic (infinite mass).
            #
            # It fires every reset: the retargeted reference holds the hand a few mm inside whatever
            # it rests on (measured on s101_seg12_knife: cutting board 6.2 mm on 82% of frames, sink
            # 6.4 mm on 72%), and PhysX ejects that overlap on the first step. Measured hand speed one
            # step after a pinned reset: 0.35 m/s with no context objects, 0.57 m/s with them — the
            # 0.22 m/s difference is this ejection, and it pushed the >3 cm displacement frames from
            # 25.9% to 39.7%.
            #
            # 1.0 -> 0.1 lets the overlap resolve over several steps instead of one kick. The cost is
            # that a deep overlap now persists longer, so watch the reset-step hand displacement and
            # the early-episode termination rate together. Raising this value was tried before and did
            # nothing (at 20 the results were byte-identical — the cap was never binding upward);
            # lowering it is the untested direction, which is why it is marked for rollback.
            max_depenetration_velocity=0.1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Match G1_29DOF locomotion setup. Self-collision off (full humanoid + 2 hands is
            # expensive and destabilizing); revisit if hand↔object/body contact needs it.
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        # ── [ROLLBACK MARKER: hand-tendon] Shadow 손 J1<->J0 고정 텐던 (2026-08-15) ──────────
        # 실제 Shadow Hand는 말단 마디(J0)가 중간 마디(J1)에 텐던으로 결합돼 함께 감깁니다. 그게
        # 물체를 감싸 쥐는 마지막 마디입니다. 이 줄이 없으면 J0는 아무 힘도 받지 않습니다 —
        # USD에 baked된 텐던 값이 전부 0이라(stiffness 0 / damping 0 / limitStiffness 미저작)
        # 자산은 "배선"만 갖고 있고 힘은 여기서 공급해야 합니다. 값은 IsaacGymEnvs shadow_hand.py
        # 및 Isaac Lab SHADOW_HAND_CFG와 동일한 정준값(30 / 0.1~0.2)입니다.
        #
        # ★ 자산 쪽 선행 수정이 반드시 함께 가야 합니다 ★
        # scripts/process_dataset/assets/fix_tendon_axis_drives.py 로 텐던 축 관절 8개
        # (robot0_{l,r}_{FF,MF,RF,LF}J0)의 USD 조인트 드라이브를 제거했습니다. 그 드라이브
        # (stiffness=1.0, target=0, maxForce=inf)가 J0를 0에 붙들고 있었고, 액추에이터 정규식
        # (J[1-3])이 J0를 잡지 않아 그대로 살아 있었습니다. 텐던이 그것과 싸우다 발산합니다.
        # NVIDIA 원본과 TJ 자산의 J0에는 드라이브가 없습니다. Isaac Lab도 같은 전제입니다 —
        # sim/schemas/schemas.py:638이 텐던 축 관절의 드라이브 수정을 건너뜁니다.
        # 둘 중 하나만 적용하면 안 됩니다: 자산만 고치면 J0가 자유 관절로 덜렁거리고,
        # 이 줄만 넣으면 발산합니다.
        #
        # 실측(생산 자산, 손가락 J1을 가동범위 60%까지 왕복):
        #   수정 전 + ls=30            발산 step 67 (robot0_{l,r}_RFJ1이 10.26 rad)
        #   수정 후 + 텐던 파라미터 없음  정상, J0 범위 0.000        <- 결합 출처 대조군
        #   수정 후 + ls=30 d=0.2      정상, J1 0.556 -> J0 0.543, 상관 0.909, 기울기 0.913
        #   수정 후 + ls=100 d=0.2     정상, 동일(0.912)            <- 강성 3배에도 무감각
        #   TJ 자산(22-DOF) 참조값                                  기울기 0.912
        # 되돌리기: 이 줄 삭제 + G1_shadow.usd.pre_tendon.bak 복원 (둘 다 해야 함).
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.2),
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
class G1ShadowSonicResidualEnvCfg(DirectRLEnvCfg):
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
    hand_action_dim: int = 36           # bimanual Shadow finger action (ABSOLUTE, EMA-smoothed α=0.5)
    action_space: int = 64 + 36         # 100
    # --- SONIC-mode DELTA-ACTION switches (default OFF = the absolute/raw behavior above) --------------
    # When ON, the policy output is a per-step INCREMENT that is EMA-smoothed and integrated into a
    # clamped target (delta=0 ⟺ HOLD), instead of an absolute value. Independent per channel; enable
    # per experiment. (Distinct from the non-SONIC group *_delta_action switches, which are bypassed here.)
    #   hand:  integrate a_hand·scale (rad/step) into the hand JOINT target, clamped to joint limits.
    #   latent: integrate z_res_raw·scale into the SONIC latent residual, clamped to ±clip (anti-windup).
    sonic_hand_delta: bool = False
    sonic_hand_delta_scale: float = 0.25        # rad/step at raw=1
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
    #   This clip does NOT bound `rew_action_reg`'s latent block: it is computed on the RAW (unclipped)
    #   residual after the 2026-07-28 rollback, so it stays unbounded and keeps a restoring gradient on
    #   mu outside the clip. (While it briefly ran on the clipped value, its ceiling was
    #   weight · clip² ; widening the clip to 20 raised that enough to help pin the total reward on the
    #   clamp(min=0) floor — that widening was also reverted.)
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
    observation_space: int = 766   # [backward-dir] +1 = 진행 방향 비트        # asserted in _get_observations. Per-link contact (Option A): the
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
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            gpu_max_rigid_patch_count=1024 * 1024 * 4,
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
    # The 14 BODY keypoints are split into TWO reward groups, each a plain mean over its own group:
    #   CORE (9)  pelvis, shoulders×2, elbows×2, hips×2, knees×2      → rew_body_kpts
    #   EE   (5)  wrists×2, ankles×2, torso                           → rew_ee_kpts
    # The termination gate still uses the UNIFORM mean over ALL 14 (e["body"]).
    # ── [ROLLBACK MARKER: body-kpt-off] 몸 키포인트 감독 스위치 (2026-08-14 실험) ─────────────
    # 몸 키포인트 목표(_np_ref_kpts의 body 14개)는 SMPL-X 사람 키포인트를 그대로 쓰므로 몸 비율
    # 차이가 영구 잔차로 남는다. False로 두면 env __init__이 (1) rew_body_kpts→0 (2)
    # term_body_kpt_err→1e6 (3) cache_body_bar→1e6 을 일괄 적용해 코어 몸 감독을 빼고, body
    # 게이트가 겸하던 낙상 감지는 아래 term_root_pos/rot_err 루트 게이트(기준 = 리타게팅된 로봇
    # 골반 g1_root_pose — SMPL-X가 아님)가 대신한다. 목적: EE/손/루트 목표만 남겼을 때 SONIC
    # prior가 몸을 얼마나 유지하는지 관찰 (감독 제거이지 정보 제거가 아님 — 관측 54kpt와
    # e["body"] 계산·로깅 Error / body_kpts 는 그대로 남아 진단 지표가 된다).
    # 유지되는 항: rew_ee_kpts(손목2+발목2+몸통1), rew_hand_kpts, rew_link_kpts, rew_fingertip,
    # rew_root_pos/ori, 물체 항, 캐시 root/fingertip bar. 되돌리기: True (아래 값들은 원본 유지).
    body_kpt_supervision: bool = True
    rew_body_kpts: float = -0.5        # mean over the 9 CORE body kpts (pelvis/shoulders/elbows/hips/knees)
    # ── [ROLLBACK MARKER: ee-kpts] 손목2 + 발목2 + 몸통1 을 한 항으로 (2026-09-01) ────────────
    # 몸통은 BODY_KPTS 4번 torso_link + TORSO_KPT_OFFSET(윗가슴, jT9T8 대응)이다. G1 은 목 관절이
    # 없고 head_link 가 torso_link 에 융합돼 있어 독립 머리 키포인트가 존재하지 않는다.
    # 몸통을 넣는 이유: core 는 body_kpt_supervision=False 로 가중치 0 이라, 골반(rew_root_pos/ori)
    # 과 손목·발목 사이의 몸통만 무감독이었고 허리가 자유로워 상체가 앞뒤로 들썩였다
    # (실측: 허리 yaw 진폭 로봇 3.37° vs 레퍼런스 0.47°, roll 3.52° vs 1.12°, pitch 1.37° vs 0.61°).
    # 한 mean 에 섞어도 되는 근거: exp 형태의 기울기 ∂/∂err_i = w·Sat·(-2·mean/σ²)·(1/N) 은 자기
    # 오차 크기와 무관하게 1/N 로 균등 배분된다. 몸통 오차는 mm, 발목은 cm 스케일이지만 발목이
    # mean 을 0 에서 떼어놓아 몸통도 단독 항보다 큰 기울기를 받는다. 키포인트당 배분은 균등하다.
    # 손목 종료 게이트(term_wrist_pos_err)와 Error / wrist_kpts 로그는 그대로 살아 있다 —
    # _wrist_kpt_idx 는 보상에서만 빠졌다.
    # 되돌리기: env 의 _EE_NAMES 에서 빼낼 이름을 지우고 해당 항의 가중치/σ 를 되살린다.
    rew_ee_kpts: float = -3.0 # -1.76         # mean over 5 kpts: WRIST ×2 + ANKLE ×2 + TORSO ×1
    rew_hand_kpts: float = -3.0 # -1.76        # mean over 40 finger-chain kpts (dexterity)
    rew_fingertip: float = -6.0 # -5.2        # mean over 10 bimanual pads ≈ grasp -5.2 (both are means)
    # ── [ROLLBACK MARKER: link-kpt] 손끝 외 wrap 링크의 접촉 목표 추종 ────────────────
    # 손끝 외의 손가락 마디(뿌리/중간)도 제 위치를 지키게 하는 항입니다. 손끝 10개만으로는 손
    # 방향이 거의 안 묶입니다 — 손목을 60도 돌려도 손끝을 목표 근처에 놓는 자세가 여럿이라 옆면
    # 으로 잡는 자세가 허용됩니다. 뿌리/중간 마디는 한 방향에서만 자기 목표에 닿으므로 이 항이
    # 실제로 자세의 자유도를 묶습니다.
    # 목표는 "물체 기준 좌표로 기록해 둔 레퍼런스에서의 링크 위치"를 살아있는 물체 자세로 되돌린
    # 점입니다(_solve_ref_link_local이 시작할 때 한 번 계산). 따라서 물체가 굴러가면 손이 따라가야
    # 할 자세도 같이 돌아갑니다. 물체 표면의 접촉점을 목표로 쓰던 이전 방식은 물체가 회전해도
    # 목표가 그대로여서 회전을 따라가지 못했습니다. 접촉 요구와 무관하게 매 프레임 정의되므로
    # 접촉 마스크로도 걸지 않습니다. 어떤 링크를 셀지는 link_kpt_include_palm이 정합니다.
    rew_link_kpts: float = 0.0 # -1.0 # -3.5
    # 목표가 물체 표면의 접촉점이던 시절에는 손바닥을 뺐습니다. 레퍼런스 자세에서조차 오른손바닥은
    # 요구 프레임의 12.6%만 그 목표에 닿았고(측정값), 도달 불가능한 목표를 강제하면 정책이 손바닥을
    # 물체로 밀어 넣어 관통을 만들기 때문입니다. 목표를 레퍼런스 링크 위치로 바꾼 뒤로는 손바닥
    # 목표도 로봇이 실제로 취했던 자세라 도달 가능하므로 다시 포함합니다.
    link_kpt_include_palm: bool = True
    # B. Locomotion / root — root_ori up-weighted over root_pos (GRAIL-aligned: orientation > position).
    rew_root_pos: float = -0.5         # was -2.5 (user 2026-07-20: de-emphasize root position)
    rew_root_ori: float = -0.5         # was -1.0 (user 2026-07-20: emphasize root orientation)
    # C. Balance / feet — REMOVED (user 2026-07-20, GRAIL-aligned): the frozen SONIC base owns feet +
    #   balance, so the residual policy has NO foot-contact obs/reward (no rew_foot_contact/foot_slip,
    #   foot_force_cap, foot_kpt_gate, foot flatness). Feet are still tracked as part of the 14-kpt body reward.
    # D. Object manipulation — reused from grasp (inside the clamped tracking_penalty)
    # ── [ROLLBACK MARKER: z-weight] 중력 방향 오차 가중 (grasp 규약 포팅, 2026-09-01) ──────────
    # grasp env 는 위치 오차의 z 성분에 ×1.5 를 걸어 보상에 씁니다 (논문 S4.2 "higher weights to
    # gravity direction"): robotis_shadow_grasp_env.py:1498/1516 의 delta_obj_pos_w / delta_kpts_w.
    # 그 규약의 핵심은 **무가중/가중 두 벌을 따로 유지**하는 것입니다 —
    #     무가중 → 종료 게이트(term_*), RSI good 게이트(enough_*), 캐시 bar, Error / 로그
    #     가중   → 보상만
    # 가중값으로 통째로 바꾸면 종료 임계와 실패 판정까지 같이 조여집니다(z 가 오차의 절반이면
    # 노름이 약 1.12배 → term_obj_pos_err 0.15 가 실효 0.134). 그래서 e 딕셔너리에 *_w 키를
    # 병행 추가하고 보상 조립부만 그쪽을 읽습니다.
    # 적용 범위는 hand_kpts 와 obj_pos 둘로 한정합니다 (사용자 지정). grasp 는 여기에 21 MANO
    # 키포인트와 팔 항까지 포함했지만, 우리 쪽 ee/link/fingertip/root 는 무가중으로 둡니다.
    # 1.0 이면 가중이 사라져 이전 거동과 정확히 같습니다.
    z_weight_reward: float = 1.5
    rew_obj_pos: float = -5.0 # -4.26
    rew_obj_rot: float = -1.5 # -1.0
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
    # ── [ROLLBACK MARKER: contact-vel-gate] 접촉력 보상의 물체-속도 게이팅 ─────────────
    # grasp env(robotis_sh5_grasp_env.py:258-261, 출처 TJ GR env `is_contact`)가 쓰던 게이트를
    # 복원한다. 거기서는 future_contact = (물체가 움직임) AND (손끝이 물체 근처) 였고, 접촉력
    # 보상과 손끝 목표 전환을 모두 이 플래그로 걸었다:
    #     fforce_contact = raw_forces * contact_flag_gated * contact_condition
    # sonic_residual 은 2026-07-22 에 Option-A per-link 맵으로 갈아타면서 이 게이트가 빠졌다
    # (parahome_hand_contact.py 에 같은 상수 0.05/0.25 가 --use-velocity-gate 로 남아 있으나
    #  "정적 홀드도 잡으려고" 기본 비활성). 여기서는 hand_contact.npz 자체는 건드리지 않고
    # (CWS 가 같은 맵을 읽는다) 소비 지점 네 곳에만 게이트를 얹는다 — grasp env 와 같은 범위다:
    #   (1) 접촉력 보상 link_mask            → 물체 정지 시 force_rew = 0
    #   (2) 그 관측 (FUTURE per-link mask)   → 보상이 0 인 프레임을 정책이 구분할 수 있게
    #   (3) 손끝 목표 전환 · 보상 (ft_reward)
    #   (4) 손끝 목표 전환 · 관측 (delta_ft_obj)
    #
    # 의미: 물체가 정지해 있으면 "눌러도 보상 없음". 물체를 실제로 옮기는 순간에만 접촉력을
    # 요구하므로, 가만히 놓인 물체를 눌러 점수를 버는 행동이 사라진다.
    #
    # (3)(4) 의 부작용 — 정지 구간에서 손끝 목표가 "물체 표면 접촉점" → "레퍼런스 패드"로 되돌아
    # 간다. 실측상 사람 패드는 얇은 손잡이 옆 약 3.6 cm 에 있으므로, 그 구간 동안 손끝이 물체에서
    # 떨어지는 방향으로 끌린다. 사용자 판단으로 grasp env 와 같은 범위를 택했다 (2026-09-01).
    # 이 절반만 되돌리려면 env 의 in_contact 두 곳에서 _ref_obj_vel_gate 곱을 빼면 된다.
    #
    # 측정 (레퍼런스, env 50Hz 기준 OR 조건 통과율):
    #   칼   s101_seg12 89.7% / s53_seg19 90.7% / s55_seg31 80.8%  ← 자르는 동안 계속 회전
    #        (칼 각속도 중앙값 0.59 rad/s = 34°/s, 임계 0.25 의 2.4배)
    #   팬·볼 s100_seg00 pan / s33_bowl / s71_bowl 은 정적 홀드가 길어 통과율이 크게 낮다
    # 즉 칼에서는 게이트가 약 10% 프레임만 닫히고, 집어-들고-있는 동작에서 크게 작동한다.
    # 임계는 TJ 원값 그대로 (OakInk/HO-Cap 탁상 집기 과제에서 온 값이라 ParaHome 보정은 안 됨).
    # 되돌리기: contact_vel_gate=False (게이트가 전부 1 이 되어 거동이 정확히 이전과 같다).
    contact_vel_gate: bool = True
    contact_vel_gate_lin: float = 0.05   # m/s,   레퍼런스 물체 선속도 임계
    contact_vel_gate_ang: float = 0.25   # rad/s, 레퍼런스 물체 각속도 임계
    contact_force_cap: float = 1.0     # N; per-link compressive force at/above this = full credit (saturates)
    # [ROLLBACK MARKER: force-fingertip-only] 접촉력 보상을 손끝 10개로만 계산합니다. False면
    # wrap 링크 32개 전체를 씁니다. 전체를 쓰면 분모(레퍼런스가 요구하는 링크 수)에 손바닥처럼
    # 달성 불가능한 링크가 섞여 보상이 눌립니다(실측: 프레임당 7개 요구, 실제 접촉 3개).
    contact_force_fingertip_only: bool = False
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
    contact_normal_gate_tol: float = 1.00
    # E. Alive / regularization
    # rew_alive sets the tracking_penalty clamp floor (-rew_alive). Above the grasp default (1.5)
    # because the grouped full-body/bimanual tracking penalty is larger; kept high enough that good
    # tracking stays unsaturated (else the clamp kills the gradient) while flooring bad steps at 0.
    # ── [ROLLBACK MARKER: exp-tracking] tracking-reward SHAPE (복원 2026-09-02) ────────────────
    # False = 선형 (sum of -w*err, clamped at -rew_alive). True = SONIC 형태, 항마다
    # w*exp(-err²/σ²). 이 플래그 하나가 추적 그룹 전체를 바꾸고 나머지 항은 건드리지 않는다.
    # 항별 가중치는 아래 rew_* 에서 파생된다 (|rew_*| 를 exp_tracking_budget 으로 정규화) —
    # 두 형태가 같은 상대 강조를 유지하므로, 결과가 나빠져도 "동시 재밸런싱" 탓으로 돌릴 수 없다.
    #
    # 선형 형태의 문제: clamp 가 물면 그 샘플의 추적 기울기가 EXACTLY 0 이 되어 12cm 의 물체
    # 오차와 40cm 가 같은 점수를 받는다. 복원 직전 측정 `Diag / clamp_frac` = 0.168 (16.8%).
    # exp 형태는 항마다 [0, w] 로 유계라 clamp 가 필요 없고 모든 항이 기울기를 유지한다.
    # 포화가 합 하나가 아니라 항별로 일어나므로 나쁜 손끝이 몸 신호를 침묵시키지 않는다.
    exp_tracking_reward: bool = True
    # 여덟 추적 항이 한 스텝에 지급할 수 있는 총량. 선형 형태의 clamp 크기(rew_alive = 1.5)에
    # 맞춰 둔다 — contact_force 0.7 / com_support -0.5 / action_reg 가 전부 그 스케일에 맞춰
    # 튜닝됐고, SONIC 의 절대 예산(항 합 7.0)을 쓰면 그것들이 값 변경 없이 5배 약해진다.
    exp_tracking_budget: float = 1.50
    exp_rew_alive: float = 0.5   # exp_tracking_reward 가 켜지면 rew_alive 를 대체한다. 훨씬 작은
    # 이유: 지수 항이 전부 양수라 긴 에피소드가 이미 더 큰 할인 수익을 얻고 별도 생존 보너스가
    # 필요 없다 (SONIC 은 alive 항이 아예 없다). 0 이 아닌 이유는 음의 정규화 항들이 총합을 최종
    # clamp(min=0) 아래로 밀어 그 기울기까지 죽이는 걸 막기 위해서다.
    # σ: 그 항이 exp(-1) = 0.37 을 지급하는 오차. SONIC (gear_sonic/config/manager_env/rewards/
    # terms/*.yaml) 에서 대응하는 항의 값을 가져오고, 우리 오차가 더 작게 측정된 곳은 조였다.
    # 직관이 아니라 `Sat /` 로그로 튜닝할 것 (0.3~0.7 대역이 실제로 학습을 이끄는 구간).
    # 복원 시점 예측 Sat (사용자 지정으로 구값 유지): fingertip 0.24 / obj_pos 0.57 / hand 0.56 /
    #   ee 0.25 / obj_rot 0.68 / root_pos 0.97 / root_rot 0.92.
    #   root_pos·root_rot 는 0.9 대라 사실상 상수다 — 대역에 넣으려면 0.045 / 0.10 근처가 필요하고,
    #   ee 는 손목 병합으로 오차가 0.119 로 커져 0.16 근처가 맞다. 지금은 의도적으로 구값 유지.
    sigma_body: float = 0.30        # SONIC tracking_relative_body_pos std
    sigma_ee: float = 0.10          # [wrist-into-ee] 손목2+발목2+몸통1 공통 σ
    sigma_hand: float = 0.075 # 0.10
    sigma_fingertip: float = 0.05  # half of term_ft_err
    # 링크 원점과 표면 접촉점 사이에는 링크 두께만큼의 하한이 있습니다(실측 약 4cm).
    # Sat / link_kpt 이 1에 붙지 않는 게 정상이고, 0.3~0.7 대역에 오도록 맞추세요.
    sigma_link_kpts: float = 0.075 # 0.05
    sigma_root_pos: float = 0.20 # 0.30    # SONIC tracking_anchor_pos std
    sigma_root_rot: float = 0.30 # 0.40    # SONIC tracking_anchor_ori std
    sigma_obj_pos: float = 0.05    # half of term_obj_pos_err
    sigma_obj_rot: float = 0.30    # half of term_obj_rot_err
    # ── [/ROLLBACK MARKER: exp-tracking] ─────────────────────────────────────────────────────
    # 선형 모드에서만 유효: `Diag / clamp_frac` 이 오르면 rew_alive 를 올릴 것.
    rew_alive: float = 1.5
    # ── [ROLLBACK MARKER: reg-merge] action_reg 를 잠재+손 하나로, SUM 으로 통일 (2026-09-01) ──
    # 이전에는 두 항이 서로 다른 관행을 따랐다:
    #   rew_action_reg_hands -0.004  ·  sum(smoothed_hand²)  36차원, EMA 평활  ← grasp 관행
    #   rew_latent_reg       -0.1    ·  mean(z_raw²)         64차원, RAW      ← GRAIL LatentL2 관행
    # 집계(sum/mean)·피연산자(평활/raw)·차원이 모두 달라 차원당 압력이 어긋났다 (실측 차원당 제곱
    # 0.467 vs 1.23). grasp 관행인 SUM 으로 통일한다 — robotis_shadow_grasp_env.py:1603 의
    # `(self.actions[:, :N_f] ** 2).sum(dim=-1)` 와 같은 형태이고 가중치도 같은 -0.004 다.
    # 피연산자: 잠재 64 는 _last_z_res (절대 모드에서 _cur_policy_action[:, :64] 와 동일한 값),
    # 손 36 은 _cur_policy_action 의 손 블록(raw, 클램프 이전). 둘 다 raw 라 클립 밖에서도 복원
    # 기울기가 남는다(클립된 값을 벌하면 외부가 평평해져 mu 가 되돌아오지 않는다). delta 모드에서는
    # _last_z_res 가 적분된 잠재라 그쪽을 써야 한다 — raw 증분을 제곱하면 속도 페널티가 되어버린다.
    #
    # 배점 (실측 차원당 제곱값 기준):
    #   손   36 × 0.467 = 16.8  → -0.004 × 16.8 = -0.067   ← 이전 action_reg 와 정확히 동일
    #   잠재 64 × 1.23  = 78.7  → -0.004 × 78.7 = -0.315   ← 이전 latent_reg -0.123 의 2.6배
    #   합                        -0.382                    ← 이전 두 항 합 -0.190 의 2.0배
    # 즉 손 항은 그대로 보존되고, 잠재 항이 같은 관행을 따르면서 2.6배 강해진다. 참고로 GRAIL
    # LatentL2 는 coef 0.01 · mean(64) = 차원당 1.56e-4 이므로, 차원당 0.004 는 그 약 25배다.
    # z_res 는 정책이 몸에 가하는 유일한 손잡이라, 과하면 z_res→0 으로 수렴해 잔차가 무력해진다.
    # 첫 런에서 Diag / zres_absmax 와 Episode_Reward / action_reg 를 함께 볼 것.
    rew_action_reg: float = -0.002   # SUM over 100 of (z_res 64 + a_hand 36) 제곱
    rew_pose_reg_hands: float = -0.001   # HANDS-only: pull achieved hand joints toward the DEFAULT (rest/neutral) pose
    #   — a task-agnostic regularizer (grasp/TJ convention), NOT toward the retarget reference (that would just
    #   duplicate rew_hand_kpts/rew_fingertip tracking). NOT applied to the SONIC-driven body.
    # action_rate: RAW 100-D 정책 액션의 스텝간 변화 제곱합 (z_res 64 + a_hand 36 한 항).
    # 2026-08-18~09-01 사이 잠재/손으로 분리돼 있었으나 두 가중치가 같아 다시 하나로 합쳤다 —
    # 선형이므로 -0.001·Σ(잠재) + -0.001·Σ(손) == -0.001·Σ(전체), 비트 단위로 동일하다.
    # 실현된 65-D 관절 목표가 아니라 RAW 정책 액션 기준이다 (user 2026-07-20; GRAIL
    # meta_action_rate_l2) — 실현된 목표를 벌하면 SONIC 자신의 몸 추종까지 벌하게 된다.
    # (GRAIL 은 전체 meta-action rate 에 -0.1 을 쓴다 — 튜닝 여지.)
    rew_action_rate: float = -0.001
    # ── [ROLLBACK MARKER: energy] 허리·다리 역학적 파워 페널티 (2026-09-02) ────────────────
    # Σ|τ·q̇| over 허리 3 + 다리 12 = 15관절. τ 는 robot.data.applied_torque (implicit actuator 가
    # computed_effort = k·err_pos + d·err_vel 로 채우고 클립한 값), q̇ 는 joint_vel.
    #
    # 왜 τ² 이 아니라 τ·q̇ 인가: 파워는 정지 유지에서 0 이다(q̇=0). 상체를 중력에 대해 버티는
    # 정적 토크(허리 pitch 약 60 N·m 규모)가 구조적으로 배제되므로, τ² 처럼 정적 부하가 진동
    # 신호를 압도하는 일이 없다. τ² 에서는 진동분이 1.3% 밖에 안 됐다.
    #
    # 왜 상체 들썩임을 겨냥하는가: action_rate 는 차분(고역통과)이라 저주파 대진폭 진동에 둔감하다
    # (실측: 로봇 부호 반전 5회 vs 레퍼런스 16회 — 로봇이 더 느리고 크게 흔들린다. 진폭은 루트
    # 피치 5.7배인데 각속도로는 1.8배로 줄어든다). 파워는 진폭과 주파수를 함께 본다.
    #
    # 왜 팔을 제외하는가: 팔·손목 14관절의 레퍼런스 감쇠 파워 중앙값이 7.54 W 로 세 그룹 중
    # 가장 크고(허리 0.20 / 다리 5.96), 실측상 팔은 로봇이 레퍼런스보다 이미 느리다(가속도 0.6배).
    # 넣으면 이 항의 절반 이상을 차지하면서 정당한 리치 동작을 벌한다.
    #
    # 배점: 절대 형태(레퍼런스 대비 초과분이 아님)라 레퍼런스 기저까지 함께 청구된다. 레퍼런스
    # 감쇠 파워 실측(허리+다리): s101_seg12_knife 0.94 W / s55_seg31_knife 2.09 W / 중앙값 6.15 W /
    # 최대 s66_seg26_pan 10.90 W. 여기에 들썩임 초과분(추정 3~8 W)과 강성분이 더해진다.
    # 칼 클립 운용값을 5~10 W 로 보면 -0.001 은 -0.005 ~ -0.010 — action_rate(-0.028)보다 약하다.
    # 첫 런에서 `Diag / energy` 실측을 보고 조정할 것 (-0.003 ~ -0.01 이 권장 대역).
    # 클립 간 실효 가중치가 4배까지 벌어진다(절대 형태의 대가). 클립 하나씩 학습하면 런 안에서는
    # 일관되지만, train_sequences.sh 로 전 클립을 같은 cfg 로 돌릴 때는 클립별 조정이 필요하다.
    # 0 이면 항이 사라진다.
    rew_energy: float = -0.0001

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
    # [ROLLBACK MARKER: body-kpt-off] 루트 낙상 게이트 — body_kpt_supervision=False일 때만 활성
    # (_dones_deviation). body 게이트가 겸하던 낙상/루트 이탈 감지의 대체물. 기준은 리타게팅된
    # 로봇 골반(g1_root_pose) 대비 e["root_pos"]/e["root_rot"] — 이미 매 스텝 계산되는 값이라
    # 추가 비용 없음. 감독이 켜진 기준선에서는 body 게이트가 이를 포섭하므로 여기는 잠잠하다.
    term_root_pos_err: float = 0.40        # m, root(pelvis) position deviation (fall/drift substitute gate)
    term_root_rot_err: float = 1.0         # rad, root(pelvis) rotation deviation (tilt/fall substitute gate)
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
    enable_wrist_rot_termination: bool = True    # [wrist-rot] 키포인트 좌표계 비교로 되살림
    term_wrist_rot_err: float = 0.75       # rad [wrist-rot]       # rad, per-hand palm-rotation deviation (= grasp max_wrist_rot_err)
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
    # [ROLLBACK MARKER: context-z] Sink every context object at spawn, as an alternative to lifting
    # the object (object_spawn_declear). Both fix the same defect — the reference holds the object a
    # few mm INSIDE whatever it rests on — but they leave the error in different places, and that
    # difference decides the reward.
    #
    # Lifting the object spawns it at the height it will actually settle at, which is ABOVE the
    # reference. The reward compares against the untouched reference, so on every resting frame the
    # object is being marked down for a position the SUPPORT PHYSICALLY BLOCKS it from reaching —
    # unreachable reward, which no amount of training recovers. Sinking the support moves the
    # settling height DOWN onto the reference, so the target becomes achievable.
    #
    # Measured 0.3 s after spawn, median over the resting frames (lift -> sink):
    #     s101_seg29_pot     49.9 -> 22.4 mm      s100_seg02_kettle  28.6 ->  7.4 mm
    #     s100_seg00_pan     29.2 ->  5.3 mm      s101_seg30_bowl    23.1 -> 18.4 mm
    #     s10_seg03_book     70.4 -> 35.9 mm      s101_seg12_knife    5.7 ->  3.4 mm
    # and the vertical component specifically goes from +23..33 mm to -0.4..+4.4 mm on all of them.
    #
    # What it does NOT fix, so expectations stay honest: only the VERTICAL error. The object also
    # slides horizontally (the bowl's z error goes to zero while 18 mm of horizontal error remains)
    # and tips over (pot 9.6 deg, book 21.2 deg, cup 61.8 deg even after the fix) — both come from the
    # reference pose not being a stable resting pose on the collider, which no z shift can address.
    context_z_offset: float = 0.0          # m, how far DOWN to move the context objects at spawn
    # Measure that number instead of hard-coding it: it is per-clip (5.5 mm on s101_seg12_knife,
    # 19.5 mm on s100_seg00_pan, 24.1 mm on s100_seg02_kettle) and cannot be known before the
    # contexts are spawned, because it comes from settling the object ON them. So: spawn, run the
    # declear solve, sink the contexts by the median lift it asked for, re-solve. Costs one extra
    # declear solve at startup. Composes with context_z_offset above — the solve always measures
    # whatever is left over from wherever the contexts currently are. Leave object_spawn_declear ON
    # alongside it: the ~2 mm the single constant cannot cover is what that per-frame lift is for.
    context_z_auto: bool = True
    # [/ROLLBACK MARKER: context-z]

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
    use_rsi: bool = True
    # [ROLLBACK MARKER: retarget-joint-order] map the retarget's g1_joint_pos columns onto the env's
    # action joints BY NAME instead of trusting position. The two orders silently diverged when
    # G1_shadow.usd was rebuilt after g1_shadow_joint_order.json was dumped: 24 of 65 slots were
    # crossed, all in the hands (MF<->TH, FF<->RF at J1/J2/J3), so MFJ1 was fed THJ2's value and the
    # middle finger stuck out in every rollout while body tracking looked fine. _ref_joints is the
    # residual action base, the RSI reset pose and the state-cache seed, so it poisoned all three.
    # False restores the pre-fix positional read.
    remap_ref_joint_order: bool = True

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
    # Steps of FREE settling used to measure where the object actually comes to rest. Convergence is
    # fast (the pan is within 0.6 mm of final by step 5); 12 leaves margin without costing much, since
    # this now runs ONCE per frame instead of once per bisection iteration.
    declear_settle_steps: int = 12
    declear_max_lift: float = 0.05         # m, safety clip so a pathological frame cannot launch it
    declear_rest_lin: float = 0.05         # m/s   reference speed under which the object counts as at rest
    declear_rest_ang: float = 0.25         # rad/s reference angular speed for the same test

    adaptive_sampling: bool = True
    failure_weighted_sampling: bool = True   # TRAIN: True (failure-weighted). PRETRAIN overrides → False (uniform).
    # [ROLLBACK MARKER: deferred-cache] ----------------------------------------------------------
    # Commit the RSI state cache in BULK AT TERMINATION instead of per-step.
    #
    # Today `_save_state_cache` runs inside `_get_rewards`, i.e. EVERY control step, and writes the
    # env's CURRENT frame immediately. The only quality filter is `_enough_continued` (tracking has
    # been continuously good since reset) — which a born-dead episode PASSES for its first few steps,
    # because at reset the state IS the reference/cache so the errors are still small. Such an episode
    # therefore writes states that lead to death into frames whose cache was empty or worse, and those
    # states are then restored by later resets.
    #
    # With this > 0 the writes are held per-env and only committed when the episode ENDS, and only if
    # it lasted at least this many control steps. Episodes that die sooner contribute nothing. The
    # per-frame quality gate and the "keep the higher reward" rule are unchanged — this adds a
    # hindsight filter on top of them.
    #
    # COST: a (num_envs, max_episode_length, 222) fp32 staging buffer. 2048 envs x 251 frames =
    # 442 MB; 4096 x 501 = 1.76 GB. If that is too much, lowering it does NOT shrink the buffer (the
    # buffer must span a whole episode) — cap `episode_length_s` or reduce num_envs instead.
    # 0 disables the deferral entirely and restores the per-step write, byte-identical to before.
    cache_min_episode_length: int = 10
    # [/ROLLBACK MARKER: deferred-cache] ---------------------------------------------------------
    adaptive_alpha: float = 0.001
    adaptive_uniform_ratio: float = 0.1
    adaptive_back_seconds: float = 1.0 # 0.8       # run-up before the sampled target frame (= 50 frames @50 fps)
    # [ROLLBACK MARKER: rand-runup] 되감기를 [이 값, adaptive_back_seconds] 사이에서 프레임마다
    # 무작위로 뽑습니다. 0이면 기존처럼 고정(TJ와 동일한 방식)입니다. 하한은 에피소드가
    # cache_min_episode_length를 넘길 여지를 남기기 위한 것입니다.
    runup_rand_min_frames: int = 30 # 10
    # [ROLLBACK MARKER: ref-start-prob] 이 확률로 캐시 히트를 무시하고 레퍼런스 자세에서 시작합니다.
    # 캐시는 프레임당 슬롯이 하나라, 한 번 들어간 상태가 나쁘면 그 프레임이 영구히 그 상태로
    # 고정됩니다(그 상태에서 시작 -> 또 실패 -> 더 나은 상태가 그 프레임을 지나갈 일 없음).
    # 레퍼런스를 가끔 섞으면 비교 대상이 생겨 교체가 일어날 수 있습니다. 0이면 기존 동작입니다.
    ref_start_prob: float = 0.01
    # ── [ROLLBACK MARKER: ref-reset-jvel] 레퍼런스 리셋에 관절 속도 주입 (2026-08-14) ────────────
    # 지금까지 where_ref 리셋 경로는 관절 위치·루트 자세·루트 선/각속도·물체 선/각속도를 모두
    # 레퍼런스에서 채우면서 관절 속도만 0으로 남겼습니다(jvel = zeros_like(jpos) 뒤로 대입 없음).
    # 그래서 클립 중간에서 시작하면 "몸통과 물체는 움직이는데 관절만 정지"인, 물리적으로 어긋난
    # 조합이 됩니다. 더 중요한 것은 SONIC과의 불일치입니다 — sonic_hist_from_reference=True 이므로
    # 리셋 직후 SONIC의 10프레임 속도 이력은 레퍼런스 속도(_ref_hist["jvr"])로 채워지는데, 정작
    # 시뮬레이터의 관절 속도는 0이었습니다. 즉 프리어는 "지난 10프레임 동안 움직여 왔다"고 듣고
    # 로봇은 멈춰 있었습니다. True로 두면 그 두 이야기가 같아집니다.
    #
    # 값은 _ref_joint_vel (F,65) = 리샘플된 _ref_joints의 후방차분 x control_fps. 루트 속도와 같은
    # 규약(후방차분, 역방향 에피소드에서 부호 반전, 0번 프레임 0)입니다.
    #
    # 12클립 실측(30->50fps 보간 후 차분, |v| rad/s): p99 = 다리 1.18 / 허리 0.81 / 팔 3.11 /
    # 손 0.81, 전체 최대 6.92. 10 rad/s 초과 프레임 0%, max/p99 비율 2.2~4.0 (IK 지터 스파이크
    # 없음). 30->50 보간이 저역통과로 작용해 차분 노이즈가 증폭되지 않습니다. 그래서 평활도
    # 그룹별 축소도 넣지 않았습니다. clip은 지금 데이터에서는 걸리지 않는 안전밸브입니다.
    #
    # 기본값 False인 이유: 지금 12클립 벤치마크 스윕이 진행 중이라, 켜면 아직 시작하지 않은
    # 클립만 새 조건으로 학습해 벤치마크가 반쪽씩 다른 조건이 됩니다. 스윕이 끝난 뒤 True로
    # 뒤집어 새 스윕을 돌리면 깨끗한 A/B가 됩니다. 되돌리기: False.
    # ── [ROLLBACK MARKER: tendon-reset] 리셋 시 텐던 축 J0 를 제약에 맞춰 쓰기 ─────────────────
    # 손가락 말단 J0 8개는 액추에이터가 없고 PhysX 고정 텐던으로 J1 에 묶여 있습니다
    # (q_J0 = gear * q_J1). 액션 관절 65개에 없으므로 _reset_idx 는 항상 default(=0) 로 두는데,
    # 상태 캐시도 65열만 저장하므로 어느 리셋 경로에서도 J0 는 복원되지 않습니다. J1 이 굽은
    # 프레임에서 리셋하면 텐던 제약이 최대 1.14 rad 위반된 채로 시작합니다.
    # 실측(J1=1.0 rad 로 리셋): J0 는 4스텝(20 ms) 만에 따라잡지만 그 과정에서 말단 링크가
    # 1.705 m/s 로 튑니다 — 리셋 직후 손 속도(0.35~0.57 m/s)의 3배입니다. 제약에 맞춰 쓰면
    # 그 과도응답이 사라집니다. 되돌리기: False.
    tendon_reset_couple: bool = True
    # USD physxTendon gearing 비 (root -0.00805 / axis +0.00705). 리타게팅 URDF 의 <mimic>
    # multiplier 와 같은 값이어야 합니다 — 두 곳이 어긋나면 레퍼런스와 시뮬레이터가 다른 손 모양을
    # 갖게 됩니다(scripts/process_dataset/assets/add_urdf_tendon_mimic.py).
    tendon_gear_ratio: float = 0.00805 / 0.00705      # = 1.14184
    ref_reset_joint_vel: bool = True
    ref_reset_joint_vel_scale: float = 1.0    # 주입 배율 (0.5 = 절반 속도로 시작)
    # 성분별 상한(rad/s). 몸통과 손을 나눈 이유: 자연 스케일이 3배, 액추에이터 한계가 6.7배
    # 차이나서 스칼라 하나로는 손 쪽 안전밸브가 작동하지 않습니다. 12클립 3671프레임 실측:
    #   몸통 29개: p99 2.42 / p99.9 4.56 / max 6.92, 액추에이터 한계 100 (max가 한계의 6.9%)
    #   손  36개: p99 0.81 / p99.9 1.67 / max 3.25, 액추에이터 한계  15 (max가 한계의 21.7%)
    # 10.0은 손 p99의 12배라, 손가락이 8 rad/s로 튀는 명백한 이상값(손 액추에이터로 실제 명령
    # 가능한 값)을 그냥 통과시킵니다. 6.0이면 실측 최대의 1.8배로 여유는 남기되 밸브가 됩니다.
    # 두 값 모두 현재 12클립에서는 걸리지 않습니다(몸통은 8 이상, 손은 4 이상이면 비활성) —
    # 잘라낼 이상값이 있어서가 아니라 미래 클립 대비용입니다. 관절별 max/p99가 1.3~2.1로 지금
    # 데이터에는 IK 결함 스파이크가 없습니다(있으면 이 비율이 10을 넘습니다). 상위 관절은 전부
    # 어깨/팔꿈치/손목 — 실제 리치 동작입니다(right_wrist_roll 6.92, right_shoulder_pitch 6.87).
    ref_reset_joint_vel_clip: float = 10.0        # 몸통(다리+허리+팔) 29개
    ref_reset_joint_vel_clip_hands: float = 10.0   # 손 36개
    # No-pretrain regime: RSI is seeded DIRECTLY from the PyRoki retarget reference (every frame is a valid
    # start via the where_ref reset path), so the pretrain-cache warm-start is OFF. The train cache still
    # supplies better (physical) restore states for frames it covers as training fills it.
    pretrain_cache_warmstart: bool = False   # was True (pretrain→train warm-start); now reference-seeded RSI
    #   RESET on pretrain→train transfer (loaded policy fed RAW obs → diverged), NOT the warm-start.
    #   Fixed in train.py _load_partial_checkpoint (floored scaler transfer). Warm-start gives
    #   reference-matching reset poses (lower initial wrist_rot). Set False for vanilla RSI (rollback).
    # [ROLLBACK MARKER: late-gate] fraction of the clip an episode must survive, on top of finishing
    # within 3 frames of the end, before the cache quality gate switches to the tight 'late' object
    # bars. The two together mean "started inside the first 20% and completed the clip". > 1.0 keeps
    # the gate off for good; 0.0 drops the length requirement and the switch reduces to "any episode
    # reached the end", which under reference-seeded RSI fires on the first control step.
    late_gate_survival_frac: float = 0.8

    uniform_sampling_steps: int = 2000
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
    # Per-LINK contact reward (Option A / DexMachina — the single env contact map, from hand_contact.npz)
    # =========================================================================== #
    # The old fingertip-only contact map (obj-velocity gate + nearest-object-vertex: obj_contact_linvel/
    # angvel_thresh, contact_dist_threshold, contact_near_vertex_thresh, use_grounded_normal,
    # use_fingertip_to_vertex_dir, use_contact_point_gate, use_contact_normal_gate, contact_normal_tol) was
    # REMOVED 2026-07-22 — superseded by the per-link Option-A map (rew_contact_force + contact_force_cap
    # above; LINK_CONTACT_NAMES). Only the SPATIAL-gate tolerance remains, now reused by the per-link force
    # reward (a link's force counts only if the robot link is within this of its object-surface target).
    # [ROLLBACK MARKER: cws-contact] 링크별 접촉 센서가 접촉점 위치도 보고하게 합니다. 접촉 렌치의
    # 모멘트 팔에 필요합니다. 켜면 프림당 접촉 데이터 버퍼를 실제로 쓰게 되는데, 이 버퍼는 넘쳐도
    # 잘리지 않고 장치 단언으로 죽으므로 (ft_max_contact_points가 그 상한) 환경 수를 올릴 때 한 번은
    # 확인이 필요합니다. 접촉 렌치 보상을 안 쓰면 꺼서 비용을 아낄 수 있습니다.
    # ── 물체 마찰 커리큘럼 [ROLLBACK MARKER: friction-curriculum] ──────────────────────
    # 에피소드마다 물체 마찰을 [friction_min, friction_max(t)]에서 뽑습니다. friction_max(t)는
    # friction_max_init에서 friction_min까지 friction_decay_steps 제어 스텝에 걸쳐 선형 감소하고,
    # 그 뒤로는 friction_min 고정입니다. 쉬움→어려움: 초반엔 잘 안 미끄러지니 대충 쥐어도 잡히고,
    # 점차 실제 마찰로 조입니다. friction_min은 물체 USD에 구워진 값(DEFAULT_FRICTION=1.0)과
    # 같게 두어, 감쇠가 끝나면 커리큘럼이 없던 것과 동일한 조건이 됩니다.
    # False로 두면 물체 마찰은 USD의 구워진 값 그대로입니다.
    # ── 역방향 롤아웃 [ROLLBACK MARKER: backward-dir] ──────────────────────────────────
    # 이 비율만큼의 환경을 시간 역방향으로 굴립니다. 0.0이면 모든 관련 코드가 항등이 되어 기존
    # 동작과 완전히 같습니다. 목적은 정방향이 통과하지 못하는 병목 프레임의 캐시를 채우는 것.
    backward_ratio: float = 0.00
    # 역방향이 쓴 상태가 정방향 항목을 밀어내려면 보상이 이 비율만큼 더 높아야 합니다. 점수 척도는
    # 같지만(둘 다 그 프레임 레퍼런스와의 일치도) 정방향 항목에는 "정방향 동역학으로 실제 도달했다"
    # 는 보증이 붙는 반면 역방향에는 없습니다. 보상 크기가 1.1~1.5 수준이라 2%는 절대값 0.02~0.03
    # 으로, 프레임 간 흔들림보다는 크고 "쥐었다/아니다"의 차이(0.1 단위)보다는 훨씬 작습니다.
    # `Curriculum / cache_bwd_frac`로 확인하세요 — 0에 붙으면 너무 빡빡, 1로 오르면 너무 느슨,
    # 0.1~0.3이면 의도대로 병목 구간만 채우는 중입니다.
    backward_replace_margin: float = 0.02
    friction_curriculum: bool = True
    friction_min: float = 1.0          # 하한이자 최종 고정값 (물체 USD의 기본 마찰과 동일)
    friction_max_init: float = 3.0     # 초기 상한
    friction_decay_steps: int = 30000  # 이 제어 스텝 동안 상한이 friction_min까지 내려옴
    track_contact_points: bool = True

    # ── [ROLLBACK MARKER: failure-dump] 실패 에피소드 덤프 (2026-08-17) ────────────────────────
    # 목적: 실패의 원인을 기하학적으로 진단하려면 "실패 직전 몇 스텝 동안 무슨 일이 있었는지"와
    # "그 상태로 시뮬레이터를 되돌릴 수 있는 정보"가 함께 필요합니다. 되돌릴 수 있어야 나중에
    # 액션을 조금 흔들어보는 반사실 실험(±d)을 할 수 있습니다. 그래서 RSI 상태 캐시가 쓰는
    # 것과 똑같은 222차원 복원 상태(_build_cache_state)를 그대로 재사용합니다.
    #
    # False면 버퍼조차 만들지 않아 기존 동작과 완전히 같습니다.
    #
    # 용량: 한 스텝 한 환경이 360 float32 = 1.44 KB. 창 20스텝이면 에피소드당 28.8 KB.
    #       기본값(4000 에피소드)에서 약 115 MB. 시작할 때 예상 용량을 출력합니다.
    failure_dump: bool = False
    failure_dump_envs: int = 64          # 앞쪽 N개 환경에서만 기록 (전체를 담으면 수백 GB)
    failure_dump_window: int = 20        # 실패 직전 몇 스텝을 담을지 (원인은 죽은 순간이 아니라 접근 구간)
    failure_dump_budget: int = 4000      # 총 에피소드 상한 — 용량을 정하는 주 손잡이
    failure_dump_bucket: int = 1000      # 이 제어 스텝마다 예산을 균등 배분 (학습 초기에만 쏠리지 않게)
    failure_dump_total_steps: int = 41000  # 예산을 나눌 기준 총 스텝. --timesteps 와 맞춰야 고르게 담깁니다.
    # RSI 복원 직후의 죽음은 정책의 실수가 아니라 초기화 여파입니다(캐시 상태가 물체와 살짝 겹쳐
    # 있으면 물리가 밀어내며 1~2스텝에 종료). 그걸 진단에 넣으면 "물체에서 멀어져라"라는 정반대
    # 방향이 쌓이므로 반드시 걸러야 합니다. 캐시 쓰기 게이트(cache_min_episode_length)보다 깁니다.
    failure_dump_min_len: int = 10
    failure_dump_dir: str = ""           # 빈 문자열이면 로그 디렉터리 아래 failure_dump/

    # ── [ROLLBACK MARKER: failure-sigma] 실패 구간 탐색 확대 (2026-08-18) ─────────────────────
    # 가설: 반복적으로 실패하는 구간에서 정책이 기존 행동 주변을 더 넓게 시도해 보면, 지금 못 찾는
    # 대안을 발견할 수 있다. 보상도 레퍼런스도 PPO 목적함수도 건드리지 않고 SAMPLING 분산만 키운다.
    #
    # RSI 가 이미 매 에피소드마다 "이 실패를 겨냥한다"고 선언한다는 점을 그대로 씁니다:
    #     pick  = 실패 가중 샘플링으로 뽑힌 프레임
    #     start = pick - _back  (10~40 프레임 되감기)
    # 그래서 프레임 전체에 대한 밀도 표를 따로 만들 필요가 없습니다. [start, pick] 구간에서만
    # sigma 를 키우면 되고, 창 폭은 _back 이 이미 무작위로 정해 줍니다.
    #     실측: max 정규화 밀도표 방식은 501프레임 중 3개만 활성 (평균 beta 1.003) → 사실상 무효.
    #           [start, pick] 방식은 전체 스텝의 약 30~40% 가 활성.
    #
    # beta 크기는 그 프레임이 뽑힐 확률에 비례시킵니다 (자주 실패하는 곳일수록 더 넓게):
    #     g    = (p_pick / p_max) ** gamma
    #     beta = 1 + (beta_max - 1) * g
    #   실측(뽑힌 에피소드 기준 g 분포): gamma=1 이면 p10=0.003 p50=0.211 p90=1.000 으로 가장 넓게
    #   구분되고 E[beta]=1.22. 순위 백분위는 샘플이 이미 실패 가중이라 96%가 1.0 으로 포화해 못 씁니다.
    #
    # PPO 는 수정하지 않습니다. beta 를 관측 마지막 열로 실어 보내면 메모리에 자동 저장되어
    # 업데이트 때 미니배치와 정렬이 맞고, GaussianMixin.act 가 같은 분포로 log_prob / 엔트로피 / KL 을
    # 계산하므로 비율이 1 에서 시작합니다. (메모리 텐서로 넣으면 skrl 의 고정 7-튜플 언패킹이 깨져
    # PPO.update 전체를 복제해야 합니다.) 관측에 싣되 mu 쪽 신경망 입력에서는 잘라내므로,
    # beta 는 행동 자체가 아니라 탐색 폭에만 영향을 줍니다.
    failure_sigma: bool = False
    failure_sigma_beta_max: float = 1.5   # 가장 자주 실패하는 프레임에서의 sigma 배율
    failure_sigma_gamma: float = 1.0      # g = (p_pick/p_max)**gamma. 0.5 면 완만해집니다.
    failure_sigma_dims: str = "all"       # "all" = 100차원 전체 | "hand" = 손 36차원만
    # >0 이면 [start, pick] 무시하고 모든 스텝에 이 값을 적용 — 탐색 총량을 맞춘 전역 대조군용.
    # beta_G = sqrt(Diag/beta_sq_mean) 을 temporal 런에서 읽어 넣습니다.
    failure_sigma_global: float = 0.0
    # ── [/ROLLBACK MARKER: failure-sigma] ─────────────────────────────────────────────────────
    # ── [/ROLLBACK MARKER: failure-dump] ──────────────────────────────────────────────────────

    # ── [ROLLBACK MARKER: cws-contact] 접촉 렌치 보상 (CHORD, arXiv 2607.00033) ────────────────
    # "force"  기존 힘 기반만  |  "cws"  접촉 렌치만  |  "both"  둘 다 (기본)
    # 렌치 점수는 접촉의 배치만 보고 세기는 안 봅니다(마찰 원뿔의 대표 방향이 크기 1인 힘이라).
    # 그래서 힘 보상을 없애면 "실제로 눌러라"를 가르치는 항이 사라집니다. 논문도 접촉 보상을 다른
    # 보상들과 더하기로 붙이므로 병행이 기본입니다.
    contact_reward_mode: str = "cws" # "force"
    # 접촉 그룹 총량을 기존 힘 보상과 같은 0.7로 맞춥니다. 나머지 보상(com_support -0.5,
    # latent_reg -0.1 등)이 그 크기를 기준으로 튜닝돼 있어서, 배점을 0.35로 두면 접촉만 절반이
    # 됩니다. "both"로 되돌릴 때는 0.35로 낮추고 rew_contact_force도 0.35로 내려야 총량이 유지됩니다.
    # [ROLLBACK MARKER: cws-diag] 진단 전용 모드 (2026-08-18). True 면 contact_reward_mode 가
    # "force" 여도 접촉 렌치를 계산해 로그와 실패 덤프에만 남기고 보상에는 넣지 않습니다.
    # 목적: 실패 창에서 "지금 파지가 만들 수 있는 렌치가 필요한 렌치를 담고 있는가"를 관찰만 하기.
    # 보상 경로는 비트 단위로 불변이므로 기존 런과 직접 비교할 수 있습니다.
    cws_log_only: bool = False
    rew_cws: float = 0.25
    # 아래 4개는 논문 공개 구현에서 실제로 쓰는 값입니다 (2026-09-02 확인). 논문 본문에는
    # 없어서 예전에는 우리가 정했는데, 저장소 전체를 훑어 오버라이드가 없음을 확인했습니다.
    #   tolerance=0.1, var=0.1  -> g1_sonic_env_cfg.py:644 와 v2d_hand_env_cfg.py (두 태스크 동일)
    #   friction_coefficients=0.1, num_friction_cone_edges=8  -> tracking_command_cfg.py:138-142
    #     (utils.py 함수 시그니처 기본값 0.5 는 cfg 가 항상 덮어써서 쓰이지 않습니다)
    cws_beta: float = 0.1        # 여유 범위. 로봇이 사람의 (1-beta)~(1+beta)배 안이면 만점.
    # 레퍼런스 자세 실측(덮은 방향 비율 평균)은 여유 0.2에서 칼 66% 냄비 71% 그릇 74% 팬 40%,
    # 0.3에서 75/80/82/47% 였습니다. 0.1 로 좁히면 이 비율이 더 내려가지만, 이제 cws_reward 가
    # 방향별 평균이라 덮지 못한 방향이 전부 0 이 아니라 분모에만 비례 감점으로 들어갑니다.
    # 0은 쓰면 안 됩니다 — 접촉점이 정확해지면서 오히려 값이 떨어졌습니다.
    cws_v: float = 0.1            # 벌점 세기. 이제 방향 하나의 (부족²+과잉²)을 나눕니다(합이 아님).
    # sigma 는 힘 성분이 크기 1, 회전 성분이 |p|/rc 라 대략 0~1.4 범위입니다. v=0.1 이면 방향당
    # 부족 0.32 에서 exp=0.37, 0.48 에서 0.10 이라 구분 대역이 맞습니다.
    cws_n_dir: int = 512         # 비교 방향 개수 (논문 num_wrench_space_basis_samples)
    cws_n_edge: int = 8          # 마찰 원뿔 옆면 개수 (논문 num_friction_cone_edges). 여기에
    # 순수 법선 1개가 더해져 생성자는 9개입니다(논문 friction_cone_edges_jit 의 append_normal).
    # 회전 시 값이 달라지는 오차(중앙/최대)는 mu=1.0 에 법선 없던 시절 4개 5.1%/42%, 8개
    # 1.1%/10.2%, 16개 0.3%/2.5% 였습니다. mu=0.1 에서는 옆면이 법선에 거의 붙어 훨씬 작습니다.
    cws_link_chunk: int = 4      # 링크를 몇 개씩 나눠 계산할지. 0이면 한 번에(환경 2048에서 2 GB 초과).
    cws_mu: float = 0.1          # 마찰계수. 논문 값입니다.
    # 주의: 우리 시뮬레이터의 실제 물체 마찰은 DEFAULT_FRICTION=1.0 이고 friction_min 도 1.0 이라
    # 물리적으로는 1.0 이 맞습니다. 논문의 0.1 은 마찰을 거의 없다고 보는 보수적 선택으로, 원뿔이
    # 반각 5.7도로 좁아져 점수가 "마찰을 얼마나 활용하나"보다 "접촉 위치와 법선이 맞나"를 주로
    # 봅니다. sigma_h 와 sigma_r 이 같은 mu 를 쓰므로 비교 자체는 유효합니다. 논문 재현을 우선해
    # 0.1 로 두었고, 시뮬레이터 물리와 맞추려면 1.0 으로 되돌리면 됩니다.
    cws_seed: int = 0            # 비교 방향을 뽑는 시드. 사람/로봇이 같은 방향을 써야 하므로 고정.
    cws_force_thresh: float = 0.1  # N, 접촉으로 칠 최소 법선 힘
    # 논문은 힘 임계를 아예 두지 않습니다. 활성 판정이 접촉점 위치의 노름(> 1e-3)이고, 이는
    # "PhysX 가 접촉점을 보고했는가"와 같습니다(utils_jit.py:wrench_preprocess_jit). 논문
    # ContactSensorCfg 의 force_threshold=0.1 은 렌치와 무관합니다 — IsaacLab 구현에서 그 값은
    # track_air_time 블록의 체공/접촉 시간과 시각화 마커에만 쓰입니다(contact_sensor.py:422,519).
    # 우리는 1.0 N 이었는데, PhysX 접촉 노이즈 하한 정도인 0.1 N 으로 낮춰 논문 거동에 붙입니다.
    # 완전히 같게 하려면 0 으로 두고 isfinite(contact_pos_w) 만 보면 됩니다.
    # ── [/ROLLBACK MARKER: cws-contact] ──────────────────────────────────────────────────────

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
    residual_action: bool = False   # BYPASSED in SONIC mode (governs only the non-SONIC fallback path).
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
    # [ROLLBACK MARKER: sonic-encoder-g1] SONIC에 무엇을 명령으로 줄지.
    #   "smpl" : 사람 SMPL 관절 위치 + 손목 관절 6개 (기존 동작)
    #   "g1"   : 로봇 자신의 29개 관절 각도 + 속도 (사람->로봇 변환 단계가 없음)
    # 리타게팅(_ref_joints)이 없는 클립에서는 자동으로 smpl로 되돌아갑니다.
    # 실측(SONIC 단독, residual 0, 오른손 손목 회전 오차 rad):
    #                     smpl    g1
    #   전체 p50          0.490  0.337
    #   나이 3~10         0.504  0.299
    #   나이 25~60        1.080  0.439   <- smpl은 시간이 갈수록 62도까지 벌어짐(옆면이 물체를 향함)
    # 레퍼런스 고정 시 바닥값이 0.229이므로 SONIC이 만드는 순수 오차는 0.261 -> 0.108로 60% 감소.
    # 에피소드도 훨씬 오래 삽니다(나이 60+ 표본 173 -> 7740).
    sonic_encoder: str = "g1"
    residual_scale_latent: float = 0.10   # λ on z_res (GRAIL pre-quantization latent residual scale)
    control_fps: float = 50.0            # resample reference 30 fps → this; MUST match parahome_smpl_for_sonic TGT_FPS
    sonic_smpl_file: str = "sonic_smpl_50fps.npz"   # SONIC SMPL encoder arrays (sibling of the retarget npz)
    # [ROLLBACK MARKER: hist-from-reference] What SONIC's 10-frame proprioception window is filled
    # with at reset. The window is history — there is none at the first step of an episode, so it has
    # to be fabricated, and the choice decides what the frozen decoder believes just happened.
    #   True  = the REFERENCE's own last 10 frames. The window is then a real trajectory whose
    #           positions, velocities and orientation agree with each other, and it is the SAME
    #           trajectory the tokenizer feeds as the FUTURE.
    #   False = replicate the current row into all 10 slots. That says "the robot has not moved for
    #           10 frames" while the tokenizer says the reference is mid-motion — two different
    #           stories, and a frozen robot past contradicts any non-zero velocity restored from the
    #           state cache.
    # Ported from the RePHO variant, where it was written and measured.
    sonic_hist_from_reference: bool = True
    # The two below only apply to the REPLICATED fallback (from_reference=False); with the reference
    # window the velocities and the action already agree with the positions.
    #   seed_zero_vel   zero the seeded joint velocity so the frozen positions and the velocity stop
    #                   contradicting each other.
    #   act_seed_from_pose  seed last_action with the action that commands the CURRENT pose
    #                   (jpr / sonic_scale, the exact inverse of the decode) instead of 0. Off because
    #                   IsaacLab's action manager zeroes the action at reset, so 0 is what the frozen
    #                   decoder actually saw during its own training.
    sonic_hist_seed_zero_vel: bool = False
    sonic_act_seed_from_pose: bool = False



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
    # (clip selection is dataset_root + smplx_subdir/retarget_subdir + clip_class + clip_name; the old
    #  data_subset / dataset_dir aliases were never read by the env and are gone.)
