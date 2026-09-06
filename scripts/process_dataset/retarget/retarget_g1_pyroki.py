"""G1 + bimanual Shadow retargeting with PyRoki (Phase 2: body + hands, contact-aware).

Whole-trajectory batched least-squares (jaxls) retargeting of ParaHome SMPL-X body+hand
keypoints onto the COMPOSITE G1+Shadow robot (65-DOF), ported from pyroki examples 12/11.

Robot: the composite `urdf_pyroki/g1_shadow.urdf`, exported from our G1_shadow.usd via NVIDIA
UsdToUrdf (joint frames patched for consistency; see export_g1_shadow_urdf step). It carries our
EXACT 65 action joints (29 body + 36 Shadow) + collision meshes, so the solve output maps 1:1 to
our action order and self-collision / (Phase-2b) object-contact work.

Costs: local alignment (relative joint-vector + angle, learned per-joint scale) + global alignment
over 46 body+hand correspondences · floor contact · foot skating · self-collision · smoothness ·
rest (coupled J0 held ~0) · joint limits · object-contact grasp (STAGE 2 only) · wrist orientation.

Four changes were made after the first working version, each kept behind a marked block and each
validated by measurement on s101_seg12_knife. The version they replaced is preserved verbatim as
`retarget_g1_pyroki_v1_baseline.py` (nothing calls it; it exists to reproduce the old numbers).

  [V2: hand-correspondence]  The hand pairing was off by one joint. Shadow's abduction and flexion
        axes meet at a point, so ffknuckle and ffproximal are always COINCIDENT — pairing the human
        MCP and PIP to them demanded two targets 36.7 mm apart at one point, and everything below
        shifted a joint. Dropping the degenerate link and re-pairing took the human/robot bone-length
        ratio from 0.00-2.62 (sd 0.796) to 1.06-1.46 (sd 0.113), reproducible across 4 clips.
  [V2: contact-stage2-only]  The grasp cost now runs only in stage 2. Stage 1's job is grounding, and
        pulling the hand toward the object distorted it for a hand pose stage 2 discards anyway:
        foot skating p90 went 1.4 -> 0.8 mm/frame, hand metrics unchanged or better.
  [V2: wrist-orient]  A wrist ORIENTATION target, which did not exist before (the retarget npz has no
        palm quaternion, so wrist direction was a by-product of the other terms). Built from hand
        keypoints on BOTH the human and robot side, so it needs no shared convention. Palm-direction
        error 12.9 -> 1.7 deg (right hand). Folded into contact_grasp rather than added as its own
        cost: a separate cost re-differentiates forward kinematics for the whole clip and cost +5.3 GB
        of host RAM for 3 residuals per hand.

NOTE the left-hand mirror bug fixed in export_g1_shadow_urdf.py is NOT in this file: the URDF's left
hand had lost its mirrored joint axes, so left fingers curled the wrong way. Both this script and the
baseline read the repaired URDF, so both benefit.

Runs in env_pyroki (numpy>=2 + jax + pyroki), NOT env_isaaclab:
    /home/peunsu/anaconda3/envs/env_pyroki/bin/python scripts/process_dataset/retarget/retarget_g1_pyroki.py \
        --clip s100_seg00_pan
→ data/processed/parahome/g1_shadow/<class>/<clip>/0/trajectory_pyroki.npz
Render in env_isaaclab:  render_retarget.py --clip <clip> --variant pyroki
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import trimesh
import yourdfpy
from pyroki.collision import collide, colldist_from_sdf

sys.path.insert(0, str(Path("/home/peunsu/workspace/pyroki/examples")))
from retarget_helpers._utils import create_conn_tree  # noqa: E402

_ROOT = Path("/home/peunsu/workspace/robotis_sh5/source/robotis_sh5")
_PROC = _ROOT / "data" / "processed" / "parahome"
# [cost-cleanup 2026-09-04] 기본 URDF 를 부등식 모드(nomimic)로 전환했습니다. 등식 mimic
# (J0 = 1.14184 x J1) 은 Shadow 문서의 J0 <= J1 부등식을 J1>0 에서 위반하고, 실측 기울기
# 0.913 이 역수 r2/r1 = 0.87578 에 가까워 multiplier 가 뒤집혔을 가능성이 높습니다.
# 등식 mimic 으로 되돌리려면 W_URDF=<...>/g1_shadow.urdf (그때는 W_TENDONINEQ 가 무효).
_URDF = _ROOT / "data" / "robots" / "G1" / "urdf_pyroki" / "g1_shadow_nomimic.urdf"
# ── [ROLLBACK MARKER: tendon-ineq] J0 를 mimic 등식이 아니라 부등식으로 (2026-09-02) ──────────
# Shadow 공식 문서(md_finger, Loopback tendons and J0 coupling)는 이 결합이 비례 기어가 아니라
# 부등식이라고 명시한다:  joint1 angle ≤ joint2 angle  (Shadow 명명).
# 이름 대응: Shadow J1(distal) = our FFJ0,  Shadow J2(middle) = our FFJ1  →  our J0 ≤ our J1.
# 현재 mimic 은 our J0 = 1.14184 × our J1 이라 J1>0 에서 J0 > J1 이 되어 이 제약을 위반한다.
# 게다가 시뮬 실측 기울기 0.913 은 r2/r1 = 0.87578 에 4% 로 가깝고 r1/r2 = 1.14184 와는 25%
# 차이라, mimic multiplier 가 역수로 들어갔을 가능성이 높다.
# W_URDF 로 pre_mimic URDF 를 지정하면 J0 8개가 다시 자유 변수가 되고(65 → 73), 아래
# tendon_ineq 비용이 relu(q_J0 − gear·q_J1) 로 부등식을 건다. W_TENDONINEQ=0 이면 미적용.
_URDF = Path(os.environ.get("W_URDF", str(_URDF)))
_ORDER = json.load(open(_ROOT / "data" / "robots" / "G1" / "g1_shadow_joint_order.json"))

# ── [ROLLBACK MARKER: smplx-kpts] 키포인트 소스를 ParaHome → SMPL-X 로 전환 (2026-09-04) ──
# 이 파일은 원래 ParaHome 자체 스트림(joint_positions, Xsens 계열 23 몸통 + 25+25 손 = 73)을
# 썼습니다. parahome.py 상단 서술이 SMPL-X 를 "리타게팅 표준 입력"으로 규정하는데 구현이 반대로
# 되어 있었고, 사용자 의도가 SMPL-X 였으므로 전환합니다. 소스는 npz 의 smplx_joints (F,55,3).
#
# 몸통 대응은 PyRoki 원본(examples/retarget_helpers/_utils.py:147 get_humanoid_retarget_indices)
# 의 13쌍과 동일하게 맞췄습니다 — 특히 척추를 매핑하지 않습니다.
#   왜 척추를 빼는가: 사람 척추는 다관절이고 G1 허리는 3-DOF 에 torso_link 원점이 골반에서 4.4 cm
#   뿐입니다. 옛 대응 (4 jT9T8 → torso_link) 은 사람 상흉부(골반 기준 32.0 cm)를 그 4.4 cm 링크에
#   오프셋 0 으로 요구해 27.6 cm 가 구조적으로 만족 불가능했고(실측 잔차 37~51 cm, 전 대응 중 최대),
#   그 잔차가 골반 하강/상체 기울기로 새어 waist_pitch 중력 모멘트를 키웠습니다.
#   SMPL-X 로도 spine1 이 9.7~12.4 cm 라 오프셋 없이는 여전히 안 맞습니다. PyRoki 처럼 제외하면
#   local_alignment(상대 벡터+각도, 학습 스케일)이 어깨-골반 상대 기하로 상체 방향을 잡습니다.
# SMPL-X 관절 순서: 0 pelvis, 1/2 hip, 3 spine1, 4/5 knee, 6 spine2, 7/8 ankle, 9 spine3,
#   10/11 foot, 12 neck, 13/14 collar, 15 head, 16/17 shoulder, 18/19 elbow, 20/21 wrist.
_BODY = [
    (0, "pelvis"),
    (17, "right_shoulder_pitch_link"), (19, "right_elbow_link"), (21, "right_wrist_yaw_link"),
    (16, "left_shoulder_pitch_link"), (18, "left_elbow_link"), (20, "left_wrist_yaw_link"),
    (2, "right_hip_pitch_link"), (5, "right_knee_link"), (8, "right_ankle_roll_link"),
    (1, "left_hip_pitch_link"), (4, "left_knee_link"), (7, "left_ankle_roll_link"),
]
# hands: HAND_CHAIN (ParaHome hand-local idx → Shadow body); left block +23, right +48
# [V2: hand-correspondence] 손 대응표. v1에서 관절 하나만큼 밀려 있던 것을 바로잡았습니다.
#
# Shadow 손은 벌림축과 굽힘축이 한 점에서 만납니다 — URDF의 FFJ2(ffknuckle -> ffproximal) origin이
# (0,0,0)이라 두 링크가 항상 같은 자리에 있고 회전만 다릅니다. v1은 거기에 사람의 MCP와 PIP를 각각
# 짝지었는데, 36.7 mm 떨어진 두 목표를 같은 점에 요구하는 것이라 전역 정렬이 중간에 설 수밖에
# 없었습니다 — 손가락마다 노드 2개가 약 18 mm 오차를 구조적으로 안고 갔고(양손 16/40 노드),
# 그 아래 마디도 전부 한 칸씩 밀렸습니다.
#
# 퇴화한 *proximal을 빼고 사람 인덱스를 한 칸 당기면 마디 길이 비가 이렇게 바뀝니다
# (오른손, s101_seg12_knife, 사람 대비 로봇):
#     v1  1.15 / 0.00 / 2.26 / 1.40    범위 0.00~2.62,  표준편차 0.796
#     v2  1.15 / 1.23 / 1.25           범위 1.06~1.46,  표준편차 0.113
# 4개 클립 x 양손에서 동일하게 재현됩니다. 즉 v1의 "손가락마다 크기 비가 제각각"은 실제 체형
# 차이가 아니라 어긋난 대응이 만든 허상이었고, 바로잡으면 거의 균일한 1.2배 확대입니다.
#
# 사람 손끝(TIP: 21/17/13/9)은 대응에서 빠집니다 — 로봇에는 대응하는 링크 원점이 없고, 손끝은
# _FT_PADS + _FT_OFF_R 로 접촉 비용에서 따로 다룹니다.
# [smplx-kpts] SMPL-X 손 블록 로컬 인덱스 (왼손 base 25 / 오른손 base 40).
# 블록 순서: index1,2,3  middle1,2,3  pinky1,2,3  ring1,2,3  thumb1,2,3  (각 MCP/PIP/DIP)
# 손목은 SMPL-X 손 블록에 없고 몸통 20/21 이라 아래 _PALM 으로 따로 붙입니다.
#
# 엄지는 TJ 의 gr_env_cfg.py:110-113 (MANO↔Shadow, 검증된 매핑)과 같은 [thbase, thmiddle, thdistal]
# 입니다. 로봇 엄지 체인에 퇴화 쌍이 둘 있습니다 — thbase/thproximal 이 좌표 동일(0.00 cm),
# thhub/thmiddle 도 동일. TJ 는 각 쌍에서 하나씩만 취해 비퇴화 3점을 확보합니다. 뼈 길이 비
# 실측 1.32 / 1.29 로 네 손가락 대역(1.08~1.64) 안입니다.
_HAND_CHAIN = {
    "index":  ([0, 1, 2], ["ffknuckle", "ffmiddle", "ffdistal"]),
    "middle": ([3, 4, 5], ["mfknuckle", "mfmiddle", "mfdistal"]),
    "pinky":  ([6, 7, 8], ["lfknuckle", "lfmiddle", "lfdistal"]),
    "ring":   ([9, 10, 11], ["rfknuckle", "rfmiddle", "rfdistal"]),
    # ── [ROLLBACK MARKER: hand-kpt-align] 엄지 대응 정정 (2026-08-15) ────────────────────────
    # 사람 엄지는 22, 23, 24 세 점이고 24 는 손끝입니다(fingertip_pad_pos[0] 과 0.33 cm 일치로 확인).
    # 옛 대응 [22,23,24] -> [thproximal, thmiddle, thdistal] 은 한 칸씩 밀려 있었습니다. 여기서는
    # 오프셋을 쓸 수 없으므로(대응이 링크 원점 기준) 손끝은 빼고 두 관절만 씁니다 — 손끝은
    # _FT_PADS 의 접촉 비용이 pad 오프셋으로 이미 따로 다룹니다.
    #
    # 마디 길이 비율로 후보를 비교했습니다(네 손가락의 로봇/사람 비율 1.03~1.26 이 기준):
    #   옛 대응 22->thproximal, 23->thmiddle : 비율 1.39 / 1.92  편차 0.980
    #   채택   22->thmiddle,   23->thdistal : 비율 1.17 / 0.79  편차 0.388   <- 2.5배 일치
    # 네 손가락 [MCP,PIP,DIP] -> [knuckle, middle, distal] 은 원래 정확해서 그대로 둡니다
    # (로봇 knuckle/middle/distal 원점이 각각 MCP/PIP/DIP 에 있습니다).
    # env 의 HAND_CHAIN 도 같은 대응으로 맞췄습니다 — 두 곳이 어긋나면 리타게팅이 맞춘 것과
    # 보상이 재는 것이 달라집니다.
    "thumb":  ([12, 13, 14], ["thbase", "thmiddle", "thdistal"]),
}


# ── [ROLLBACK MARKER: fingertip-align] 손끝을 대응에 통합 ─────────────────────────────────────
# 손끝은 링크 원점이 아니라 말단 링크에서 pad 오프셋만큼 떨어진 점이라 _HAND_CHAIN 으로는 표현할
# 수 없었습니다. 처음에는 별도 비용(fingertip_align)으로 넣었는데, 그러면 jaxls 가 그 비용을 위해
# 순기구학을 따로 미분해서 프레임당 FK 야코비안이 한 벌 더 생깁니다 — 이 파일의 다른 주석이 같은
# 이유로 손목 방향 잔차를 contact_grasp 안에 합쳤고, 그때 실측이 +5.3 GB 였습니다.
# 그래서 대응 자체에 오프셋을 실어 global_align 하나로 처리합니다. FK 를 나눠 쓰고, 가중치도
# gw 한 곳에서 다른 키포인트와 같은 방식으로 정해집니다.
#
# 목표는 사람 fingertip_pad_pos (F,10,3) 인데 smplx_joints (F,55,3) 와 다른 배열이라,
# main 에서 두 배열을 이어붙여 (F,65,3) 로 만들고 손끝은 뒤쪽 인덱스 55.. 를 가리킵니다.
_PAD_BASE = 55                                   # [smplx-kpts] smplx_joints(55) 뒤에 손끝 10개
# [smplx-kpts] SMPL-X 손 블록에는 손목이 없습니다(몸통 20/21). 옛 _HAND_CHAIN 의 ("wrist",[0]→palm)
# 을 대체합니다 — 같은 사람 관절이 wrist_yaw_link(_BODY)와 palm 두 링크에 대응하는데, 옛 ParaHome
# 대응도 jWrist 를 몸통 10/14 와 손 로컬 0 두 곳에 쓰고 있었으므로 구조가 동일합니다.
_PALM = [(20, "robot0_l_palm"), (21, "robot0_r_palm")]
_PAD_ORDER = ("th", "ff", "mf", "rf", "lf")      # fingertip_pad_pos 의 손별 순서


def _build_correspondence():
    """(parahome_idx, urdf_link_name, distal-local offset) 3-튜플 목록.

    오프셋이 0 이 아닌 것은 손끝 10개뿐입니다(말단 링크 -> pad). 나머지는 링크 원점입니다.
    """
    pairs = [(p, l, [0.0, 0.0, 0.0]) for p, l in _BODY]
    pairs += [(p, l, [0.0, 0.0, 0.0]) for p, l in _PALM]     # [smplx-kpts] 손목 → palm
    for side, off in (("l", 25), ("r", 40)):                 # [smplx-kpts] SMPL-X 손 블록 base
        for local, shadow in _HAND_CHAIN.values():
            for pl, sh in zip(local, shadow):
                pairs.append((off + pl, f"robot0_{side}_{sh}", [0.0, 0.0, 0.0]))
        # 손끝: 왼손은 네 손가락만 Y 반전(엄지는 Y=0 이라 미러 불변) — _FT_OFF_R 과 같은 규약.
        pb = _PAD_BASE + (0 if side == "l" else 5)
        for j, fg in enumerate(_PAD_ORDER):
            o = _FT_OFF_R[fg]
            o = o if (side == "r" or fg == "th") else [o[0], -o[1], o[2]]
            pairs.append((pb + j, f"robot0_{side}_{fg}distal", list(o)))
    return pairs


# ── [ROLLBACK MARKER: foot-plant-3d] 접지 판정을 SONIC 기준으로 (2026-09-04) ────────────────
# SONIC/motionbricks 의 foot_detect_from_pos_and_vel(positions, velocity, skeleton, 0.15, 0.10)
# (motionlib/core/motion_reps/tools/feet.py) 과 동일한 기준: 높이 임계 0.10 m, 속도 임계 0.15 m/s
# 이며 속도는 수직 성분이 아니라 3D 노름입니다. 이전 값은 0.06 + |vz| 였는데, 수직 성분만 보면
# 발을 낮게 둔 채 수평으로 미끄러지는 체중 이동이 전부 "접지"로 판정됩니다 (s101_seg29_pot 에서
# 200 프레임 전부 접지 → floor_contact 가 양발을 클립 내내 바닥에 고정 → 스텝이 발 끌기로 나옴).
# 되돌리려면 이 줄을 0.06 으로 바꾸고 _foot_contact 의 노름을 |Δz| 로 교체하십시오.
# [foot-plant-3d 2026-09-04 갱신] 높이 0.05 m / 속도 1.25 m/s. SONIC 기본값(0.10 / 0.15)에서
# 높이는 조이고 속도는 크게 풀었습니다 — 1.00 m/s 는 사람 발끝 속도 분포의 이봉 사이 골에 있어
# 실질적으로 높이 단독 판정에 가깝고, 낮게 미끄러지는 체중 이동을 접지로 유지하면서 실제로
# 발을 5 cm 이상 드는 구간만 스윙으로 잡습니다. 속도는 여전히 3D 노름입니다.
_FOOT_PLANT_H, _FOOT_PLANT_V, _FPS = 0.05, 1.00, 30.0
# [smplx-kpts] 발 접지 판정에 쓰는 "볼(ball) 발" 키포인트. ParaHome 은 jLeftBallFoot(22)/
# jRightBallFoot(18) 이었고 SMPL-X 는 left_foot(10)/right_foot(11) 이 같은 역할(발가락 볼)입니다.
_PARA_BALL_L, _PARA_BALL_R = 10, 11
# G1 ankle_roll_link origin sits this far ABOVE the foot sole (URDF foot-corner contact spheres at
# z=-0.031, r=0.005 → sole at -0.036). floor_contact targets the ANKLE, so its z target must be this
# offset (not 0) — else pulling the ankle to z=0 drives the sole ~3.6 cm INTO the floor (feet penetrate
# at high W_FLOOR) or, when weak, the balance leaves the foot floating. Target = sole-on-floor.
# [ankle-sole-test 2026-09-04] 예제 12 는 floor_contact 목표 z 를 높이맵 투영값(평지=0)으로 두는데
# 우리는 발목 원점이 발바닥보다 3.6 cm 위라서 이 값으로 덮어씁니다. 예제와 일치시키는(=0) 실험을
# 위해 환경변수로 뺐습니다: W_ANKLESOLE=0 이면 예제와 동일.
_ANKLE_SOLE_OFF = float(os.environ.get("W_ANKLESOLE", 0.036))
_MOVE_LESS = ["left_hip_yaw_joint", "right_hip_yaw_joint", "waist_yaw_joint"]
# ── [ROLLBACK MARKER: tendon-couple] 텐던 결합 J1<->J0 (2026-08-15) ──────────────────────────
# 이 8개는 액추에이터가 없는 텐던 축 관절입니다. 예전에는 시뮬레이터에서 J0가 자체 USD 드라이브에
# 0으로 붙들려 있었으므로 리타게팅도 0으로 푸는 것이 옳았습니다(_HOLD_ZERO + rest 가중치 5.0).
# 자산에서 그 드라이브를 제거하고 cfg에 fixed_tendons_props를 넣으면서 J0는 이제 텐던을 따라
# q_J0 = (0.00805/0.00705) * q_J1 = 1.1418 * q_J1 로 움직입니다(실측 기울기 0.913, 범위비 1.139).
#
# 이 결합은 근사가 아니라 정확한 항등식이므로 최소화할 비용이 아니라 구조로 넣어야 합니다.
# URDF의 <mimic> 태그가 정확히 이런 결합 관절을 위한 것이고 pyroki가 정식 지원합니다
# (_robot_urdf_parser.py: mimic_multiplier/offset/act_indices, FK에서
#  value_multiplied = value_referenced * mimic_multiplier). 적용은
# scripts/process_dataset/assets/add_urdf_tendon_mimic.py 로 했고, 그 결과
# pk.Robot.from_urdf 의 actuated_names 가 73 -> 65 로 줄어 J0가 최적화 변수에서 사라집니다.
#
# 따라서 여기서는 아무 비용도 추가하지 않습니다 — 결합은 FK 자체가 보장합니다.
# 아래 _HOLD_ZERO 는 rest 가중치용으로만 남습니다: mimic 적용 후에는 J0가 actuated_names 에
# 없으므로 자동으로 무효(W_RESTJ0 를 줘도 적용될 관절이 없음)이고, mimic 이전 URDF로 되돌릴 때만
# 의미가 있습니다. 되돌리기: g1_shadow.urdf.pre_mimic.bak 복원 + W_RESTJ0=5.0.
_TENDON_GEAR = 0.00805 / 0.00705            # = 1.14184, USD physxTendon gearing 비 (mimic multiplier)
_HOLD_ZERO = [f"robot0_{s}_{f}J0" for s in "lr" for f in ("FF", "MF", "RF", "LF")]

# ---- Phase 2b object-contact grasp: fingertip PAD (distal body + offset) ↔ object surface ----------
_RAW_SCAN = _ROOT / "data" / "raw" / "parahome" / "data" / "scan"
# fingertip_pad_pos order (CLAUDE.md): LEFT[th,ff,mf,rf,lf] then RIGHT[th,ff,mf,rf,lf].
# each entry = (human_pad_idx, side, distal-finger)
_FT_PADS = [(base + j, side, fg) for side, base in (("l", 0), ("r", 5))
            for j, fg in enumerate(("th", "ff", "mf", "rf", "lf"))]
# distal-local pad offset (grasp-env FINGERTIP_OFFSETS): right verbatim, left non-thumb Y-mirrored
_FT_OFF_R = {"th": [-0.0085, 0.0, 0.02], "ff": [0.0, -0.006, 0.0175], "mf": [0.0, -0.006, 0.0175],
             "rf": [0.0, -0.006, 0.0175], "lf": [0.0, -0.006, 0.0175]}
# Wrap-link (palm/proximal/middle) PALMAR-surface offset. The contact cost pulls these links to the
# object surface, but at offset 0 it pulls the link ORIGIN (proximal joint, behind the palmar surface),
# so for palm/thumb-base-dominated grasps the hand ends up offset from the object. Push the contact
# point from the origin toward the palmar surface along the link's palmar normal (fingers -Y, thumb -X,
# palm -Y; left Y-mirrored, thumb Y=0 invariant), magnitude = env W_WRAP_OFFSET (palm ×2, it is thicker).
# W_WRAP_OFFSET=0 reproduces the old origin behavior.  [EXPERIMENT KNOB]
_WRAP_PALMAR_R = {"palm": [0.0, -1.0, 0.0],
                  **{f"{fg}{sg}": [0.0, -1.0, 0.0] for fg in ("ff", "mf", "lf", "rf")
                     for sg in ("proximal", "middle")},
                  **{f"th{sg}": [-1.0, 0.0, 0.0] for sg in ("proximal", "middle")}}


# ── [ROLLBACK MARKER: wrap-centroid] wrap 접촉점을 링크 원점 -> 메시 중심으로 ──────────────────
# 접촉 비용은 wrap 링크(palm/proximal/middle)를 물체 표면의 접촉 목표로 당기는데, 오프셋 0 이면
# 당겨지는 점이 링크 ORIGIN, 즉 **관절 위치**입니다. 실제 접촉면은 마디 중앙이라 그만큼 어긋납니다.
# 링크별 실측(원점 -> 시각 메시 중심, 링크 로컬):
#     palm 5.78 / ffproximal 2.63 / ffmiddle 1.13 / thproximal 1.62 / thmiddle 1.87 cm
# 그리고 리타게팅의 접촉 gap 이 이 값과 단조로 대응합니다:
#     tip(pad 오프셋 있음) 1.4 cm | finger wrap 3.2 cm | palm 7.5 cm
# 즉 손바닥 gap 7.5 cm 의 주범은 embodiment 격차가 아니라 접촉점을 관절에 찍고 있던 것입니다.
#
# 메시 중심은 방향 가정이 없습니다. 기존 _WRAP_PALMAR_R 은 접촉이 항상 손바닥 쪽 면이라고 가정했는데,
# 저장된 접촉 법선을 재보니 프레임마다 평균 17~33도, 최대 157도까지 흔들립니다(칼 클립은 평균조차
# 27~33도) — 고정 방향 가정이 성립하지 않습니다. palmar 오프셋은 W_WRAP_OFFSET 노브로 남겨두되
# 기본 0 이고, 기준점만 중심으로 옮깁니다.
# 남는 오차는 반경 방향(축 -> 표면, 링크 반경 ~1 cm)이고 그건 저장된 normal 이나 링크 형상까지의
# 거리로 따로 다뤄야 합니다. 되돌리기: W_WRAP_CENTROID=0.
def _link_centroid(urdf, link_name):
    """링크 시각 메시의 중심(링크 로컬). 메시가 없으면 [0,0,0]."""
    import trimesh as _tm
    pts = []
    for v in urdf.link_map[link_name].visuals:
        g = v.geometry
        if g.mesh is None:
            continue
        m = _tm.load(urdf._filename_handler(g.mesh.filename), force="mesh", process=False).copy()
        if g.mesh.scale is not None:
            m.apply_scale(g.mesh.scale)
        if v.origin is not None:
            m.apply_transform(v.origin)
        pts.append(onp.asarray(m.vertices, onp.float64))
    if not pts:
        return [0.0, 0.0, 0.0]
    return onp.concatenate(pts, axis=0).mean(0).tolist()


def _wrap_offset(link_name, mag):
    body = link_name.split("_", 2)[2]; side = link_name.split("_")[1]
    n = _WRAP_PALMAR_R.get(body, [0.0, 0.0, 0.0])
    if side == "l":
        n = [n[0], -n[1], n[2]]                                  # left hand = Y-reflection
    s = mag * (2.0 if body == "palm" else 1.0)                   # palm is thicker → larger offset
    return [n[0] * s, n[1] * s, n[2] * s]
_OBJ_LINVEL_TH, _OBJ_ANGVEL_TH, _CENTROID_TH = 0.05, 0.25, 0.05
# Contact gate for RETARGETING uses an ABSOLUTE fingertip→surface distance (not the RL env's tight
# relative "within 1.5 cm of the closest finger" gate) — the thumb opposes the fingers across the handle
# so it sits ~2.6 cm from the surface while ff/mf/rf are ~0.4 cm; the relative gate wrongly drops it.
_CONTACT_ABS_TH = 0.035


def _quat2R(wxyz):
    w, x, y, z = wxyz
    return onp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], onp.float64)


def _contact_signal(ftpad, obj_base, obj_name):
    """Per-frame/fingertip grasp-contact mask (F,10), matching the RL env: (object being manipulated:
    linvel/angvel over threshold) AND (fingertip pad near the object surface). Uses the ParaHome scan
    mesh nearest-vertex relative gate; falls back to object-centroid distance if the mesh is absent."""
    F = len(ftpad)
    op = obj_base[:, :3]; oq = obj_base[:, 3:7]
    lv = onp.zeros_like(op); lv[:-1] = (op[1:] - op[:-1]) * _FPS
    spd = onp.linalg.norm(lv, axis=-1)
    dotq = onp.abs((oq[:-1] * oq[1:]).sum(-1)).clip(0, 1)
    ang = onp.zeros(F); ang[:-1] = 2 * onp.arccos(dotq) * _FPS
    vel = (spd > _OBJ_LINVEL_TH) | (ang > _OBJ_ANGVEL_TH)
    src = _RAW_SCAN / obj_name / "simplified" / "base.obj"
    if src.exists():
        m = trimesh.load(str(src), process=False, force="mesh")
        V = onp.asarray(m.vertices, onp.float64)
        mask = onp.zeros((F, 10), onp.float32); pen = []
        for t in range(F):
            R = _quat2R(oq[t])
            ftl = (ftpad[t] - op[t]) @ R                       # fingertip pads in object-local
            cvd = onp.linalg.norm(V[:, None, :] - ftl[None, :, :], axis=-1).min(0)   # (10,) nearest-vertex dist
            near = cvd < _CONTACT_ABS_TH                       # absolute surface-proximity gate
            mask[t] = vel[t] & near; pen.append(cvd)
        pen = onp.array(pen)
        c = mask.astype(bool)
        print(f"[pyroki-retarget] contact frames/finger: {mask.sum(0).astype(int).tolist()}  "
              f"pad→surface on contact: mean {pen[c].mean()*100:.1f} cm" if c.any() else
              "[pyroki-retarget] no contact frames detected")
        return mask
    dist = onp.linalg.norm(ftpad - op[:, None, :], axis=-1)
    return (vel[:, None] & (dist < _CENTROID_TH)).astype(onp.float32)


def _foot_contact(jp, ball_idx):
    # [ROLLBACK MARKER: foot-plant-3d] 속도는 3D 노름 (SONIC 기준). |Δz| 로 바꾸면 이전 동작.
    p = jp[:, ball_idx, :]
    v = onp.zeros(len(p), dtype=onp.float32)
    v[1:] = onp.linalg.norm(p[1:] - p[:-1], axis=-1) * _FPS
    return ((p[:, 2] < _FOOT_PLANT_H) & (v < _FOOT_PLANT_V)).astype(onp.float32)


def _pelvis_target_R(jp):
    """Per-frame target pelvis orientation (world) from ParaHome body keypoints, in the G1 pelvis-link
    convention X=forward, Y=left, Z=up.  up = midshoulder→ (spine), left = right→left hip, forward = left×up.
    Constraining the free root to this removes the spurious roll that saturates hip_roll/waist_roll."""
    p_rs, p_ls = jp[:, 17], jp[:, 16]     # [smplx-kpts] right / left shoulder
    p_rh, p_lh = jp[:, 2], jp[:, 1]       # [smplx-kpts] right / left hip
    z = 0.5 * (p_ls + p_rs) - 0.5 * (p_lh + p_rh)          # up (hips → shoulders)
    z /= onp.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    y = p_lh - p_rh                                        # left (right hip → left hip)
    y = y - onp.sum(y * z, axis=1, keepdims=True) * z      # orthogonalize against up
    y /= onp.linalg.norm(y, axis=1, keepdims=True) + 1e-9
    x = onp.cross(y, z); x /= onp.linalg.norm(x, axis=1, keepdims=True) + 1e-9   # forward = left × up
    y = onp.cross(z, x)                                    # re-orthonormalize (up × forward)
    return onp.stack([x, y, z], axis=-1).astype(onp.float32)   # (T,3,3) columns = frame axes in world


def _flat_heightmap(jp, bins=48, margin=0.6):
    """Flat floor (z=0) heightmap spanning the clip's body-keypoint xy extent — the ParaHome scene has
    no terrain, so world_collision just prevents the robot sinking below the floor (and lifts it up)."""
    xy = jp[:, :22, :2].reshape(-1, 2)
    lo = xy.min(0) - margin; hi = xy.max(0) + margin
    cx, cy = (lo + hi) / 2.0; dx, dy = (hi - lo)
    box = trimesh.creation.box(extents=[float(dx), float(dy), 0.1])
    box.apply_translation([float(cx), float(cy), -0.05])          # top face at z=0
    return pk.collision.Heightmap.from_trimesh(box, x_bins=bins, y_bins=bins)


def _hand_sets(robot):
    """[V2: hand-local-scale] 손 하나짜리 국소 정렬에 쓸 대응 집합. side -> (사람 인덱스, 로봇 링크 인덱스).

    몸통과 같은 방식(쌍별 상대 벡터 + 방향)을 손 체인에도 적용하기 위한 것입니다. v1에서 손은
    전역 정렬(절대 위치)만 받았는데, 로봇 손이 사람 손보다 약 1.2배 크므로 절대 위치를 맞추려면
    손가락이 안쪽으로 눌립니다 — 크기 차이를 흡수할 통로가 없었습니다. pyroki 11번 손 예제는
    바로 이 스케일 행렬로 그 차이를 흡수합니다.
    """
    # [smplx-kpts] 오프셋을 SMPL-X 손 블록(왼 25 / 오른 40)으로. 그리고 palm 을 리스트 0번에
    # 명시적으로 넣습니다 — 옛 _HAND_CHAIN 은 ("wrist",[0]->palm) 로 시작해 palm 이 0번이었고,
    # wrist_orient 의 _t=[0,1,4](palm, ffknuckle, mfknuckle) 가 그 순서를 전제합니다. palm 을
    # _PALM 으로 빼내면서 이 전제가 깨져 손목 프레임이 검지 한 줄에서 만들어졌습니다(방향 붕괴).
    out = {}
    _palm_of = {l: p for p, l in _PALM}
    for side, off in (("l", 25), ("r", 40)):
        hp, hl = [], []
        _pl_name = f"robot0_{side}_palm"
        hp.append(_palm_of[_pl_name])                        # 0번 = 손목(사람) ↔ palm(로봇)
        hl.append(robot.links.names.index(_pl_name))
        for local, shadow in _HAND_CHAIN.values():
            for pl, sh in zip(local, shadow):
                hp.append(off + pl)
                hl.append(robot.links.names.index(f"robot0_{side}_{sh}"))
        out[side] = (hp, hl)
    return out


def solve(robot, robot_coll, heightmap, keypoints, b_para, b_link, b_mask, a_para, a_link, a_off, gw, lw,
          l_contact, r_contact, l_foot_kp, r_foot_kp, left_foot_idx, right_foot_idx,
          left_knee_idx, right_knee_idx, root_R_target, ft_idx, ft_off, ft_margin, ft_target,
          ft_mask, rest_w, weights,
          h_sets=None,
          s2_joints=None, s2_root=None, s2_offset=None, s2_lower_mask=None, s2_w=0.0):
    # STAGE 2 (s2_w>0): freeze the LOWER body (s2_lower_mask=1 joints) + root + offset at the stage-1
    # solution (s2_joints/s2_root/s2_offset) and warm-start from it, so only the UPPER body (waist+arms+
    # hands) moves to reach the hand keypoints — fixes the embodiment "hands below" without un-grounding.
    T = keypoints.shape[0]
    nb = len(b_para)                              # local-alignment set = BODY only (small NxN + scale)
    b_para = jnp.array(b_para); b_link = jnp.array(b_link)
    a_para = jnp.array(a_para); a_link = jnp.array(a_link)   # global-alignment set = body + hands
    a_off = jnp.array(a_off)                                 # (n_corr,3) 링크-로컬 오프셋 (손끝만 0 아님)

    class ScaleVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((nb, nb))): ...

    # [V2: hand-local-scale] 손 스케일. 한 손 체인의 노드 수만큼(현재 16) 정사각 행렬이고,
    # 양손과 전 프레임이 하나를 공유합니다 — 같은 하드웨어라 마디 길이가 같고(실측 좌우 차 0.047),
    # 손 크기는 클립 내내 안 변하기 때문입니다.
    class OffsetVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros((3,))): ...

    var_joints = robot.joint_var_cls(jnp.arange(T))
    var_root = jaxls.SE3Var(jnp.arange(T))
    var_scale = ScaleVar(jnp.zeros(T))
    # ── [ROLLBACK MARKER: offset-shared] 오프셋을 클립 상수로 (2026-09-04) ────────────────────
    # pyroki 예제 12 와 동일: OffsetVar(jnp.zeros(T)) — 모든 프레임이 같은 변수 id 라서 클립 전체가
    # 3차원 오프셋 하나를 공유합니다. 예전에는 jnp.arange(T) 로 프레임마다 독립 오프셋을 뒀는데,
    # 그러면 T 개의 자유도가 생기고 이를 묶는 항은 offset_reg(제거됨) 뿐이었습니다. floor_contact/
    # skating 은 접지 프레임에서만 작동하므로 스윙 구간(pot 12% / pan 27%)의 오프셋이 무제약이 되어
    # 최종 루트가 프레임당 118 mm 까지 튀었습니다(global_align/root_smooth 는 오프셋을 못 봅니다).
    # 실측(공유로 전환): 루트 xy 튐 p99  pot 70.0->25.6 mm, pan 93.0->32.3 mm (최대 -70~-74%),
    # 발바닥 높이 표준편차 pot 0.50->0.57 / pan 0.64->0.49 cm, 발슬립 p90 6.00->5.76 / 31.5->29.0 mm.
    # 수직 접지는 var_root 의 z(프레임별)가 담당하므로 오프셋의 프레임별 자유도는 중복이었습니다.
    _floor_sq = int(os.environ.get("W_FLOORSQ", 0))    # [floor-sq-test] 1 = 예제 12 의 제곱 잔차
    _off_ids = onp.zeros(T, onp.int32)
    var_offset = OffsetVar(jnp.asarray(_off_ids))
    # ── [/ROLLBACK MARKER: offset-shared] ────────────────────────────────────────────────────

    # per-node local weight `lw` (nb,) → pairwise lw[i]*lw[j]: weakening the ARM nodes frees the arm's
    # relative structure so the contact cost can bend it UP to reach the (absolute) object once the body
    # is grounded lower; legs/torso keep full local (natural grounded pose + proportion handling).
    lw_pair = lw[:, None] * lw[None, :]

    @jaxls.Cost.factory
    def local_align(vv, v_root: jaxls.SE3Var, v_cfg, v_scale: ScaleVar, kp):
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        s_pos = kp[b_para]; r_pos = T_wl.translation()[b_link]
        d_s = s_pos[:, None] - s_pos[None, :]; d_r = r_pos[:, None] - r_pos[None, :]
        scale = vv[v_scale][..., None]
        pw = (1 - jnp.eye(nb)) * b_mask * lw_pair
        res_pos = (d_s - d_r * scale) * pw[..., None]
        ds_n = d_s / jnp.linalg.norm(d_s + 1e-6, axis=-1, keepdims=True)
        dr_n = d_r / jnp.linalg.norm(d_r + 1e-6, axis=-1, keepdims=True)
        res_ang = (1 - (ds_n * dr_n).sum(-1)) * pw
        return jnp.concatenate([res_pos.flatten(), res_ang.flatten()]) * weights["local_alignment"]

    @jaxls.Cost.factory
    def scale_reg(vv, v_scale: ScaleVar):
        s = vv[v_scale]
        return jnp.concatenate([(s - 1.0).flatten() * 1.0, (s - s.T).flatten() * 100.0,
                                jnp.clip(-s, min=0).flatten() * 100.0])

    # [V2] 손 스케일 정규화 — 몸통과 같은 형태(1 근처, 대칭, 음수 금지).
    # per-correspondence, PER-AXIS global weight `gw` (n_corr,3) built in main by body part: the only
    # ABSOLUTE-position anchors are hands (object grasp) + pelvis XY + feet (floor); torso/arm/leg and
    # pelvis-Z are weak/free so LOCAL (relative, scaled) align preserves the motion shape while the legs
    # ground and the arms bend to reach the object (pyroki's joint-relationship philosophy + our grasp).
    @jaxls.Cost.factory
    def global_align(vv, v_root: jaxls.SE3Var, v_cfg, kp):
        # [ROLLBACK MARKER: fingertip-align] 링크 원점이 아니라 (원점 + 회전 x 로컬 오프셋) 을 씁니다.
        # 손끝 10개만 오프셋이 0 이 아니고 나머지는 0 이라 기존 동작과 같습니다. 손끝을 별도 비용으로
        # 두지 않는 이유는 FK 야코비안이 한 벌 더 생기기 때문입니다(이 파일의 다른 주석 참고).
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        pos = T_wl.translation()[a_link] + jnp.einsum("nij,nj->ni", T_wl.rotation().as_matrix()[a_link], a_off)
        return ((pos - kp[a_para]) * gw).flatten()

    # ── [ROLLBACK MARKER: root-xy] 오프셋을 포함한 골반 xy 를 SMPL-X 골반에 맞춤 (2026-09-04) ──
    # global_align 에 이미 (0,"pelvis") 대응이 있지만 그 비용은 var_offset 을 보지 않으므로
    # var_root 좌표계에서만 만족됩니다. 최종 루트는 SE3.from_translation(offset) @ var_root 이라
    # 오프셋 xy 만큼 실제 골반이 밀려도 아무 항이 관측하지 못합니다. offset_reg/offset_xy(제거됨)가
    # 하던 일이 그것이었고, 여기서는 0 으로 당기는 대신 "실제 골반 xy = 사람 골반 xy" 로 직접
    # 겁니다. z 는 제외합니다 — 다리 길이 차이 때문에 골반 높이는 사람과 같을 수 없고, 접지가
    # floor_contact + world_collision 으로 결정되어야 합니다. W_ROOTXY=0 이면 미적용.
    _pelvis_link_idx = robot.links.names.index("pelvis")

    @jaxls.Cost.factory
    def root_xy(vv, v_root: jaxls.SE3Var, v_cfg, v_off: OffsetVar, kp):
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        pos = T_wl.translation()[_pelvis_link_idx] + vv[v_off]      # 오프셋 포함 = 실제 골반
        return ((pos[:2] - kp[0, :2]) * weights["root_xy"]).flatten()
    # ── [/ROLLBACK MARKER: root-xy] ──────────────────────────────────────────────────────────

    @jaxls.Cost.factory
    def floor_contact(vv, v_root: jaxls.SE3Var, v_cfg, v_off: OffsetVar, lc, rc, lkp, rkp):
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        off = vv[v_off]
        lpos = T_wl.translation()[left_foot_idx] + off; rpos = T_wl.translation()[right_foot_idx] + off
        lz = T_wl.rotation().as_matrix()[left_foot_idx][2, 2]
        rz = T_wl.rotation().as_matrix()[right_foot_idx][2, 2]
        # [floor-sq-test 2026-09-04] 예제 12 는 위치 잔차에 제곱을 넣습니다: is_contact*(pos-kp)**2.
        # 최소제곱이 다시 제곱하므로 실질 4제곱이라 작은 오차는 거의 방치하고 큰 이탈만 강하게
        # 당깁니다(오차 1 cm 에서 선형의 1/100, 10 cm 에서 10배). W_FLOORSQ=1 로 그 형태를 씁니다.
        _dl, _dr = lpos - lkp, rpos - rkp
        if _floor_sq:
            _dl, _dr = _dl ** 2, _dr ** 2
        return jnp.concatenate([
            (lc * _dl).flatten(), (rc * _dr).flatten(),
            jnp.where(lc > 0.5, lz - 1, 0.0).flatten(), jnp.where(rc > 0.5, rz - 1, 0.0).flatten(),
        ]) * weights["floor_contact"]

    @jaxls.Cost.factory
    def contact_grasp(vv, v_root: jaxls.SE3Var, v_cfg, tgt, msk, kp):
        # Pull each in-contact robot HAND POINT onto its object contact target, gated by msk so free/
        # reaching parts are untouched, with a per-point margin (so it stops once close). Points are the
        # full grasp set: 10 fingertip PADS (distal FK ∘ local pad offset → human fingertip pad) + the
        # wrap links palm/proximal/middle (link ORIGIN, offset 0 → object-surface contact from the hand
        # mesh). Together they reproduce a power grasp (fingers wrap + thumb opposes + palm), not a pinch.
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        pt = jax.vmap(lambda i, o: T_wl.translation()[i]
                      + T_wl.rotation().as_matrix()[i] @ o)(ft_idx, ft_off)     # (P,3) world contact point
        res = jnp.maximum(jnp.abs(pt - tgt) - ft_margin[:, None], 0.0)          # (P,3)
        out = [(res * msk[:, None]).flatten() * weights["contact"]]
        # [V2: wrist-orient] 손목 방향 잔차를 여기에 얹습니다. 잔차는 손당 3개뿐인데 별도 비용으로
        # 두면 그 이유만으로 프레임당 FK 야코비안이 한 벌 더 생깁니다(짧은 클립 실측 +5.3 GB).
        # 이 비용은 이미 stage 2에서 T_wl 을 계산하므로 그대로 나눠 씁니다. v1에는 손목 회전 목표가
        # 아예 없어서(retarget npz 에 손바닥 쿼터니언이 없음) 손목 방향이 다른 항들의 부산물이었고,
        # 실측 오차가 25도였습니다. 사람과 로봇 양쪽에서 손 키포인트로 같은 방식의 프레임을 만들어
        # 비교하므로 좌표 규약에 의존하지 않습니다: z = 손목->중지 knuckle, x = 손바닥 법선.
        if weights["wrist_orient"] > 0 and h_sets is not None:
            pos = T_wl.translation()

            def _fr(p):
                z = p[2] - p[0]
                z = z / (jnp.linalg.norm(z) + 1e-9)
                x = jnp.cross(z, p[1] - p[0])
                x = x / (jnp.linalg.norm(x) + 1e-9)
                return jnp.stack([x, jnp.cross(z, x), z], axis=-1)

            for _side in ("l", "r"):
                _hp, _hl = h_sets[_side]
                # h_sets 순서: 0=palm, 1..3=ff, 4..6=mf, ... (_hand_sets 에서 palm 을 0번에 둡니다)
                _t = [0, 1, 4]                      # palm, ffknuckle, mfknuckle → 손 평면
                R_h = _fr(kp[jnp.array([_hp[i] for i in _t])])
                R_r = _fr(pos[jnp.array([_hl[i] for i in _t])])
                out.append(jaxlie.SO3.from_matrix(R_h.T @ R_r).log() * weights["wrist_orient"])
        return jnp.concatenate(out)

    @jaxls.Cost.factory
    def knee_separation(vv, v_root: jaxls.SE3Var, v_cfg):
        # Paper: absolute-position matching pulls the legs/knees together (narrow, unnatural stance).
        # Penalize the two knees getting closer than knee_min → keeps a natural stance width.
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        d = jnp.linalg.norm(T_wl.translation()[left_knee_idx] - T_wl.translation()[right_knee_idx] + 1e-6)
        return jnp.maximum(weights["knee_min"] - d, 0.0).reshape(1) * weights["knee_separation"]

    # [ROLLBACK MARKER: tendon-ineq] 루프백 텐던 부등식.  q_J0 ≤ gear · q_J1 (Shadow 문서).
    # knee_separation 과 같은 hinge 형태 — 위반분만 벌하므로 J0 는 그 아래에서 자유롭고,
    # 손 키포인트 비용이 사람의 distal 굴곡을 따라 J0 를 정한다. 접촉으로 distal 이 멈추고
    # 근위만 더 감기는 감싸쥐기(J0 < J1)가 표현 가능해진다 — 등식 mimic 으로는 불가능했다.
    @jaxls.Cost.factory
    def tendon_ineq(vv, v_cfg, j0i, j1i):
        q = vv[v_cfg]
        viol = jnp.maximum(q[..., j0i] - weights["tendon_gear_ineq"] * q[..., j1i], 0.0)
        return viol.flatten() * weights["tendon_ineq"]

    @jaxls.Cost.factory
    def root_orient(vv, v_root: jaxls.SE3Var, R_tgt):
        # Pull the free root orientation to the keypoint-derived pelvis frame → kills the spurious
        # roll that otherwise saturates hip_roll/waist_roll (the visible pelvis twist).
        R_rob = vv[v_root].rotation()
        return (R_rob.inverse() @ jaxlie.SO3.from_matrix(R_tgt)).log().flatten() * weights["root_orientation"]

    @jaxls.Cost.factory
    def root_smooth(vv, v_root: jaxls.SE3Var, v_prev: jaxls.SE3Var):
        return (vv[v_root].inverse() @ vv[v_prev]).log().flatten() * weights["root_smoothness"]

    @jaxls.Cost.factory
    def skating(vv, v_root: jaxls.SE3Var, v_cfg, v_off: OffsetVar,
                v_root_p: jaxls.SE3Var, v_cfg_p, v_off_p: OffsetVar, lc, rc):
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        T_wl_p = vv[v_root_p] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg_p]))
        off = vv[v_off]; off_p = vv[v_off_p]
        lsk = lc * ((T_wl.translation()[left_foot_idx] + off) - (T_wl_p.translation()[left_foot_idx] + off_p))
        rsk = rc * ((T_wl.translation()[right_foot_idx] + off) - (T_wl_p.translation()[right_foot_idx] + off_p))
        return jnp.stack([lsk, rsk]) * weights["foot_skating"]

    @jaxls.Cost.factory
    def world_collision(vv, v_root: jaxls.SE3Var, v_cfg, v_off: OffsetVar):
        # Low weight: high enough to lift the robot up off the floor, low enough not to fight retargeting.
        transform = jaxlie.SE3.from_translation(vv[v_off]) @ vv[v_root]
        coll = robot_coll.at_config(robot, vv[v_cfg]).transform(transform)
        return (colldist_from_sdf(collide(coll, heightmap), activation_dist=0.005).flatten()
                * weights["world_collision"])

    # [/OBJECT-COLLISION]

    costs = [
        local_align(var_root, var_joints, var_scale, keypoints),
        scale_reg(var_scale),
        global_align(var_root, var_joints, keypoints),
        floor_contact(var_root, var_joints, var_offset, l_contact, r_contact, l_foot_kp, r_foot_kp),
        root_orient(var_root, root_R_target),
        knee_separation(var_root, var_joints),
        # [V2: contact-stage2-only] 접촉 비용은 아래에서 단계별로 붙입니다.
        root_smooth(jaxls.SE3Var(jnp.arange(1, T)), jaxls.SE3Var(jnp.arange(0, T - 1))),
        skating(jaxls.SE3Var(jnp.arange(1, T)), robot.joint_var_cls(jnp.arange(1, T)),
                OffsetVar(jnp.asarray(_off_ids[1:])), jaxls.SE3Var(jnp.arange(0, T - 1)),
                robot.joint_var_cls(jnp.arange(0, T - 1)), OffsetVar(jnp.asarray(_off_ids[:-1])),
                l_contact[:-1], r_contact[:-1]),
        pk.costs.smoothness_cost(robot.joint_var_cls(jnp.arange(1, T)),
                                 robot.joint_var_cls(jnp.arange(0, T - 1)),
                                 jnp.array([weights["joint_smoothness"]])),
        pk.costs.rest_cost(var_joints, var_joints.default_factory()[None], rest_w[None]),
        # NOTE: self_collision_cost dropped for the 65-DOF composite — 2926 pairs × 151 frames
        # blows up to ~42 GB / int32-overflow. Re-add with a restricted pair set (hand↔body only) later.
        pk.costs.limit_constraint(jax.tree.map(lambda x: x[None], robot), var_joints),
    ]
    # world_collision + var_offset (feet-on-floor) are ON by default. Naively grounding a single-stage solve
    # drops the whole robot ~10 cm to plant feet and corrupts body+hand keypoint tracking 2–3× (8→19 cm) for
    # our tall-human→short-G1 embodiment — but STAGE 2 resolves this: stage 1 grounds the lower body, then the
    # frozen-legs upper-body re-solve re-reaches the hand keypoints, so grounding no longer costs hand accuracy.
    # var_offset 은 정규화 없이 자유입니다 (pyroki 예제 12 와 동일 — offset_reg/offset_xy 는 2026-09-04
    # 제거). 접지는 floor_contact + skating + world_collision 이 잡습니다. W_WORLDCOLL=0 →
    # faithful-tracking baseline (no grounding). To disable grounding entirely: W_WORLDCOLL=0.0.
    # [ROLLBACK MARKER: tendon-ineq] J0 가 구동 목록에 있을 때만 (pre_mimic URDF) 의미가 있다.
    if weights["tendon_ineq"] > 0:
        _an = robot.joints.actuated_names
        _pairs = [(_an.index(f"robot0_{sd}_{fg}J0"), _an.index(f"robot0_{sd}_{fg}J1"))
                  for sd in "lr" for fg in ("FF", "MF", "RF", "LF")
                  if f"robot0_{sd}_{fg}J0" in _an and f"robot0_{sd}_{fg}J1" in _an]
        if _pairs:
            _j0 = jnp.array([a for a, _ in _pairs]); _j1 = jnp.array([b for _, b in _pairs])
            # jaxls 는 인자를 배치축으로 봅니다 — (1,8) 로 넘겨 (301,) 배치에 브로드캐스트합니다.
            costs.append(tendon_ineq(var_joints, _j0[None], _j1[None]))
            print(f"[tendon-ineq] ON  쌍 {len(_pairs)}개, gear={weights['tendon_gear_ineq']:.5f}, "
                  f"w={weights['tendon_ineq']:.1f}  (actuated {len(_an)}개)")
        else:
            print("[tendon-ineq] J0 가 구동 목록에 없음 (mimic URDF?) — 비용 미적용")
    if weights["root_xy"] > 0:
        costs.append(root_xy(var_root, var_joints, var_offset, keypoints))
    if weights["world_collision"] > 0:
        costs.append(world_collision(var_root, var_joints, var_offset))
    # [V2: hand-local-scale] 손 체인의 국소 정렬 + 스케일. 몸통 local_align 과 같은 형태이되
    # 대상이 한 손이고, 스케일 행렬을 양손이 공유합니다. 절대 위치가 아니라 마디 사이 상대 벡터를
    # 맞추므로, 로봇 손이 1.2배 커도 손가락을 눌러 접지 않고 "같은 모양"을 만들 수 있습니다.
    # 양손 국소정렬 + 양손 손목방향을 하나의 비용으로 묶습니다. 비용을 4개로 나누면 jaxls가
    # 비용마다 순기구학을 따로 미분해서 프레임당 FK 야코비안이 4벌 생기고(65자유도 x 78링크)
    # 호스트 RAM이 터집니다. FK를 한 번만 계산해 네 잔차가 나눠 쓰면 그 4배가 사라집니다.
    # [V2: contact-stage2-only] 손을 물체로 당기는 비용은 STAGE 2 에만 겁니다.
    # 1단계의 일은 접지입니다(발 평평 / 무릎 / 발미끄러짐 / 바닥 충돌 / 루트 높이). 거기에 손을
    # 물체로 당기는 힘까지 같이 걸면 그 자세를 왜곡하는데, 2단계에서 하반신을 얼리고 상반신을 다시
    # 풀기 때문에 1단계가 만든 손 자세는 어차피 버려집니다. pyroki 예제도 이렇게 나뉘어 있습니다 —
    # 12번(전신)에는 접촉 비용이 없고, 11번(손)에만 있습니다.
    # V2_CONTACT_STAGE2_ONLY=0 이면 v1처럼 두 단계 모두에 겁니다.
    _c_s2_only = os.environ.get("V2_CONTACT_STAGE2_ONLY", "1") == "1"
    if (s2_w > 0.0) or (not _c_s2_only):
        costs.append(contact_grasp(var_root, var_joints, ft_target, ft_mask, keypoints))
    # [V2] 손 관련 항은 STAGE 2 전용입니다 — 1단계는 접지가 일이고 손은 2단계에서 다시 풀립니다.
    # 손목 방향은 contact_grasp 안으로 들어갔습니다(FK 공유).

    init_vals = None
    if s2_w > 0.0:
        # STAGE 2: strong pin holding the lower body (mask) + root + offset at the stage-1 pose; the
        # up-weighted hand global_align (gw hands) then drives the free UPPER body to the hand keypoints.
        _pin_j = jnp.asarray(s2_joints)                       # (T, nJ)
        _pin_root = jnp.asarray(s2_root)                      # (T, 7) wxyz_xyz (stage-1 baked root)
        # [offset-shared] 오프셋 변수가 1개이므로 프레임 0 값만 씁니다.
        _pin_off = jnp.asarray(s2_offset[:1])                 # (1,3)
        _lmask = jnp.asarray(s2_lower_mask)                   # (nJ,) 1=freeze, 0=free

        @jaxls.Cost.factory
        def stage2_pin(vv, v_cfg, v_root: jaxls.SE3Var, v_off: OffsetVar, pin_j, pin_root_arr, pin_off):
            dj = (vv[v_cfg] - pin_j) * _lmask                 # freeze lower-body joints
            droot = (vv[v_root].inverse() @ jaxlie.SE3(pin_root_arr)).log()   # freeze root
            doff = vv[v_off] - pin_off                        # freeze offset
            return jnp.concatenate([dj.flatten(), droot.flatten(), doff.flatten()]) * s2_w

        costs.append(stage2_pin(var_joints, var_root, var_offset, _pin_j, _pin_root, _pin_off))
        init_vals = jaxls.VarValues.make([
            var_joints.with_value(_pin_j),
            var_root.with_value(jaxlie.SE3(_pin_root)),
            var_scale.with_value(jnp.ones((T, nb, nb))),
            var_offset.with_value(_pin_off),
        ])

    prob = jaxls.LeastSquaresProblem(
        costs=costs, variables=[var_joints, var_root, var_scale, var_offset]).analyze()
    sol = prob.solve() if init_vals is None else prob.solve(initial_vals=init_vals)
    root = jaxlie.SE3.from_translation(sol[var_offset]) @ sol[var_root]   # bake floor offset into root
    return root, sol[var_joints]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="s100_seg00_pan")
    ap.add_argument("--class", dest="cls", default="single_rigid")
    args = ap.parse_args()

    urdf = yourdfpy.URDF.load(str(_URDF))
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    sm = onp.load(_PROC / "smplx" / args.cls / args.clip / "0" / "trajectory.npz", allow_pickle=True)
    # [smplx-kpts] SMPL-X 관절(55)을 씁니다. 없으면 parahome.py 를 --overwrite 로 다시 돌려야
    # 합니다(smplx_joints 키 추가). 옛 ParaHome 스트림으로 되돌리려면 joint_positions 로.
    if "smplx_joints" not in sm.files:
        raise KeyError(f"[smplx-kpts] {args.clip}: smplx_joints 없음 — parahome.py --overwrite 필요")
    jp = sm["smplx_joints"].astype(onp.float32)
    F = jp.shape[0]

    pairs = _build_correspondence()
    # global-align 대응 수: body 14 + 손 15x2 = 44. (옛 주석의 54 는 env 의 관측 키포인트 수이고
    # 리타게팅 대응 수와는 다릅니다 — env 는 손끝을 pad 오프셋으로 따로 세어 손당 20개입니다.)
    a_para = [p for p, _, _ in pairs]
    a_link = [robot.links.names.index(l) for _, l, _ in pairs]
    a_off = onp.array([o for _, _, o in pairs], onp.float32)         # (n_corr,3) 손끝만 0 아님
    b_para = [p for p, _ in _BODY]                                   # local-align + scale: body only (14)
    b_link = [robot.links.names.index(l) for _, l in _BODY]
    print(f"[pyroki-retarget] clip={args.clip} F={F} correspondences: global {len(pairs)} "
          f"(body {len(_BODY)} + hands {len(pairs)-len(_BODY)}), local+scale {len(_BODY)} (body)")
    b_mask = create_conn_tree(robot, jnp.array(b_link))
    h_sets = _hand_sets(robot)          # [V2] STAGE 2 손 국소정렬/스케일/손목방향용 대응

    l_c = _foot_contact(jp, _PARA_BALL_L); r_c = _foot_contact(jp, _PARA_BALL_R)
    l_kp = jp[:, _PARA_BALL_L, :].copy(); l_kp[:, 2] = _ANKLE_SOLE_OFF   # target ANKLE at sole-on-floor height
    r_kp = jp[:, _PARA_BALL_R, :].copy(); r_kp[:, 2] = _ANKLE_SOLE_OFF
    left_foot_idx = robot.links.names.index("left_ankle_roll_link")
    right_foot_idx = robot.links.names.index("right_ankle_roll_link")
    left_knee_idx = robot.links.names.index("left_knee_link")
    right_knee_idx = robot.links.names.index("right_knee_link")

    # object-contact grasp point set (concatenated, all fed through one contact cost):
    #  (A) 10 fingertip PADS: distal link + local pad offset → human fingertip pad, gated by mesh proximity.
    #  (B) wrap links (palm + proximal + middle per finger): link ORIGIN → object-surface contact target
    #      from the full hand-mesh↔object precompute (hand_contact.npz), so the grasp WRAPS (not a pinch).
    ft_idx = [robot.links.names.index(f"robot0_{s}_{f}distal") for _, s, f in _FT_PADS]
    ft_off = [(_FT_OFF_R[f] if not (s == "l" and f != "th")
               else [_FT_OFF_R[f][0], -_FT_OFF_R[f][1], _FT_OFF_R[f][2]]) for _, s, f in _FT_PADS]
    ft_pad = sm["fingertip_pad_pos"].astype(onp.float32)                 # (F,10,3) human pads (fallback target)
    # [ROLLBACK MARKER: fingertip-align] 아래에서 ft_pad 는 접촉 목표(물체 표면)로 덮어써지므로
    # [ROLLBACK MARKER: fingertip-align] 대응이 손끝을 인덱스 73.. 로 가리키므로 사람 키포인트
    # 배열 뒤에 fingertip_pad_pos 10개를 이어붙입니다. jp 는 (F,73,3) -> jp_ext 는 (F,83,3).
    # ft_pad 는 아래에서 접촉 목표(물체 표면)로 덮어써지므로 여기서 복사해 둡니다.
    jp_ext = onp.concatenate([jp, ft_pad.copy()], axis=1).astype(onp.float32)
    assert jp.shape[1] == _PAD_BASE, f"_PAD_BASE {_PAD_BASE} != joint_positions {jp.shape[1]}"
    ft_margin = [0.005] * len(_FT_PADS)                                  # tips: tight (precise pad)
    bk = [k for k in sm.files if k.startswith("obj__") and k.endswith("__base")]
    obj_name = bk[0].split("__")[1] if bk else ""

    # Load the Option-A per-link contact map ONCE — used for BOTH the fingertips (distal) and the wrap
    # links, so the retarget's contact source matches the RL env exactly (single hand_contact.npz).
    hc_path = _PROC / "smplx" / args.cls / args.clip / "0" / "hand_contact.npz"
    hc = onp.load(hc_path, allow_pickle=True) if hc_path.exists() else None
    hc_names = [str(n) for n in hc["link_names"]] if hc is not None else []

    # fingertip (distal) contact target + mask: UNIFIED with the env → the Option-A object-SURFACE target +
    # mask from hand_contact.npz (same source as the wrap links), NOT the off-surface human pad. The human
    # pad (fingertip_pad_pos) sits ~1.7 cm off a thin handle, which left the retargeted fingers floating;
    # snapping to the Option-A surface point puts the fingertips ON the object. Fallback = human pad + gate.
    if hc is not None:
        _di = [hc_names.index(f"robot0_{s}_{f}distal") for _, s, f in _FT_PADS]   # distal col per fingertip
        ft_target = hc["target"][:, _di].astype(onp.float32)             # (F,10,3) object surface (Option A)
        ft_mask = hc["mask"][:, _di].astype(onp.float32)                 # (F,10) Option-A distal mask
    else:
        ft_target = ft_pad                                               # fallback: human pad
        ft_mask = (_contact_signal(ft_pad.astype(onp.float64), sm[bk[0]].astype(onp.float64), obj_name)
                   if obj_name else onp.zeros((F, 10), onp.float32))
    c_idx, c_off, c_margin = list(ft_idx), list(ft_off), list(ft_margin)
    c_target = [ft_target]; c_mask = [ft_mask]                           # lists of (F,·) blocks, concat later
    c_group = ["tip"] * len(ft_idx)                                      # [DIAGNOSTIC] per-point group: tip/palm/finger

    if hc is not None:
        wrap = [n for n in hc_names if not n.endswith("distal")]        # palm + proximal + middle (tips above)
        _woff = float(os.environ.get("W_WRAP_OFFSET", "0.0"))           # [EXPERIMENT] palmar surface offset (m)
        _wmarg = float(os.environ.get("W_WRAP_MARGIN", "0.012"))        # [EXPERIMENT] wrap contact margin (m)
        # [ROLLBACK MARKER: wrap-centroid] 기준점 = 메시 중심(축 오차 제거) + 선택적 palmar 밀기.
        _use_ctr = os.environ.get("W_WRAP_CENTROID", "1") == "1"
        for n in wrap:
            j = hc_names.index(n)
            _ctr = _link_centroid(urdf, n) if _use_ctr else [0.0, 0.0, 0.0]
            _pal = _wrap_offset(n, _woff)
            c_idx.append(robot.links.names.index(n))
            c_off.append([_ctr[k] + _pal[k] for k in range(3)])
            c_margin.append(_wmarg)                                     # wrap: link-radius margin (origin→surface)
            c_target.append(hc["target"][:, j][:, None, :])             # (F,1,3)
            c_mask.append(hc["mask"][:, j][:, None])                    # (F,1)
            c_group.append("palm" if n.endswith("palm") else "finger")  # palm vs finger proximal/middle
        print(f"[pyroki-retarget] contact points: {len(_FT_PADS)} fingertip pads (Option-A surface) "
              f"+ {len(wrap)} wrap links (W_WRAP_OFFSET={_woff} W_WRAP_MARGIN={_wmarg})")
    else:
        print("[pyroki-retarget] hand_contact.npz absent → fingertip human-pad contact only "
              "(run parahome_hand_contact.py for full hand-mesh grasp)")
    ft_idx = onp.array(c_idx, onp.int32)
    ft_off = onp.array(c_off, onp.float32)
    ft_margin = onp.array(c_margin, onp.float32)
    ft_pad = onp.concatenate(c_target, axis=1).astype(onp.float32)      # (F,P,3)
    ft_mask = onp.concatenate(c_mask, axis=1).astype(onp.float32)       # (F,P)

    # rest weights: default 0.2, move-less joints 2.0, coupled-J0 5.0 (hold ~0 → solve≈output)
    an = robot.joints.actuated_names
    # ── [ROLLBACK MARKER: rest-hand] rest 가중치를 손만 따로 조절 (2026-08-17 진단) ──────────
    # pyroki 의 rest 목표는 관절 한계의 중간값((lower+upper)/2)이고, rest_cost 와 smoothness_cost
    # 는 65개 관절 전체에 같은 절대 가중치로 걸립니다. 팔은 진폭이 1.6~2.3 rad 라 꿈쩍 않지만
    # 손은 사람 기준 0.46~0.89 rad 라 이 벌점이 지배합니다. 실측: 리타게팅된 오른손 18관절이
    # rest 목표에서 중앙값 0.083 rad 밖에 안 떨어져 있고, 가동범위의 14.5% 만 씁니다.
    # 기본값은 현재 동작과 비트 단위로 동일합니다 (W_REST=0.2, W_RESTHAND=-1 → 미적용).
    rest_w = onp.full(len(an), float(os.environ.get("W_REST", 0.2)), onp.float32)
    for nm in _MOVE_LESS:
        if nm in an: rest_w[an.index(nm)] = 2.0
    _rh = float(os.environ.get("W_RESTHAND", -1.0))
    if _rh >= 0.0:
        _nh = 0
        for _i, _nm in enumerate(an):
            if _nm.startswith("robot0_"):
                rest_w[_i] = _rh; _nh += 1
        print(f"[rest-hand] 손 관절 {_nh}개의 rest 가중치 → {_rh} (나머지는 {os.environ.get('W_REST', 0.2)})")
    # ── [/ROLLBACK MARKER: rest-hand] ──────────────────────────────────────────────────────
    # [ROLLBACK MARKER: tendon-couple] 등식 mimic URDF 에서는 J0 가 구동 목록에서 빠지므로 이 루프가
    # 아무 관절도 못 찾아 무효입니다. mimic 이전 URDF(= 부등식 모드, W_URDF)에서만 의미가 있습니다.
    #
    # ── [ROLLBACK MARKER: rest-j0] 0 을 실제로 넣을 수 있게 (2026-09-02) ──────────────────────
    # 예전 가드가 `> 0.0` 이라 W_RESTJ0=0 을 줘도 아무 일이 없었고(기본 W_REST=0.2 유지), 올리는
    # 방향만 가능했습니다. 부등식 모드에서는 J0 가 구동 목록에 들어와 기본 0.2 가 그대로 걸리는데,
    # pyroki 의 rest 목표는 관절 한계 중간값 (0+1.571)/2 = 0.785 입니다. 실측 결과 J0 중앙값이
    # 0.749~0.875 로 정확히 그 목표에 앉고 가동범위가 0.03~0.10 (등식 파생값의 1/3) 에 그쳤습니다.
    # 즉 rest 가 J0 를 임의의 중간값에 고정하고 있었습니다.
    # 이제 센티넬을 -1 로 바꿔 W_RESTJ0=0 이 "rest 당김 제거" 로 동작합니다. 기본값(-1)은 미적용
    # 이므로 기존 거동과 비트 단위로 동일하고, W_RESTJ0=5.0 같은 기존 사용법도 그대로입니다.
    # 주의: rest 를 떼면 잘못된 앵커는 없어지지만 올바른 앵커가 생기지는 않습니다. J0 를 보는 힘은
    # global_align 손끝(오프셋 1.9cm → J0 0.1rad 당 1.9mm, 레버암이 짧아 약함), contact_grasp
    # (접촉 프레임만, 칼 링크당 22.4%), smoothness(방향은 안 정해줌) 뿐입니다. W_HANDTIP 인상이나
    # 텐던 slack 항과 함께 쓰는 것을 전제로 합니다.
    # [cost-cleanup 2026-09-04] 기본값 -1(미적용) -> 0(rest 당김 제거). tendon_ineq 가 기본 ON 이 되면
    # J0 가 구동 목록에 들어와 기본 rest 0.2 가 J0 를 한계 중간값 0.785 에 묶습니다(위 설명). 둘은 같이
    # 가야 합니다. 예전 거동으로 되돌리려면 W_RESTJ0=-1.
    _rest_j0 = float(os.environ.get("W_RESTJ0", 0.0))
    if _rest_j0 >= 0.0:
        _n = 0
        for nm in _HOLD_ZERO:
            if nm in an: rest_w[an.index(nm)] = _rest_j0; _n += 1
        print(f"[rest-j0] J0 {_n}개의 rest 가중치 → {_rest_j0} (기본 {os.environ.get('W_REST', 0.2)})")

    _w = lambda k, d: float(os.environ.get(k, d))
    # DEFAULTS = the VALIDATED production recipe (feet grounded + 2-stage upper-body hand reach). Feet are
    # planted onto a flat floor (world_collision + free offset-z), the lower body is frozen after stage 1 and
    # the arms/hands re-reach the human hand keypoints in stage 2, and contact weight is low (2.0) so the hand
    # keypoint tracking wins over object-surface pull (validated: left thumb ~0.12, hand mean ~0.04 on pan).
    # Every weight stays env-overridable for experiments (e.g. W_STAGE2=0 → single-stage; W_WORLDCOLL=0 →
    # no grounding / faithful-tracking baseline). See the sweep notes for the local/contact trade-off.
    # [cost-cleanup 2026-09-04] 기본값을 실제 사용 값으로 올렸습니다 (배치 실행 시 매번
    # W_TENDONINEQ=50.0 을 넘기고 있었습니다). 부등식을 끄려면 W_TENDONINEQ=0.
    weights = dict(tendon_ineq=_w("W_TENDONINEQ", 50.0), tendon_gear_ineq=_w("W_TENDONGEAR", 1.0),
                   local_alignment=_w("W_LOCAL", 2.0), global_alignment=_w("W_GLOBAL", 1.0),
                   hand_alignment=_w("W_HAND", 1.0), floor_contact=_w("W_FLOOR", 3.0),
                   world_collision=_w("W_WORLDCOLL", 1.0),
                   # [cost-cleanup 2026-09-04] root_orientation / knee_separation 기본값 5.0 -> 0.0.
                   # s101_seg29_pot / s100_seg00_pan 실측에서 끄는 쪽이 전 지표 우세였습니다
                   # (pot: 허리 모멘트 15.48->12.80 Nm, torso 피치 4.5->0.3도, 손끝 p90 29.6->24.5 mm,
                   #  손목 회전오차 p90 15.6->12.4도, tip 접촉간격 2.1->1.6 cm, 골반 롤 3.4->2.8도).
                   # root_orient 의 근거였던 "허위 골반 롤"은 SMPL-X 대응 수정 후 global_align 만으로
                   # 잡히고, knee_separation 은 무릎 간격 21.7~23.5 cm 로 하한 14 cm 를 한참 웃돌아
                   # 애초에 활성되지 않았습니다. 되살리려면 W_ROOTORI=5 / W_KNEESEP=5.
                   root_orientation=_w("W_ROOTORI", 0.0),
                   root_smoothness=_w("W_ROOTSMOOTH", 1.0), foot_skating=_w("W_SKATE", 1.0),
                   contact=_w("W_CONTACT", 2.0), contact_margin=_w("W_CONTACTMARGIN", 0.005),
                   knee_separation=_w("W_KNEESEP", 0.0), knee_min=_w("W_KNEEMIN", 0.14),
                   # [cost-cleanup 2026-09-04] joint_smoothness = 1.0 (예제 12 는 0.2 이지만 우리는 다름).
                   # 0.2 로 내렸다가 스윕(0.1/0.2/0.5/1.0/2.0)으로 되돌렸습니다. 예상했던
                   # "평활 ↑ → 정확도 ↓" 트레이드오프가 나타나지 않고, 오히려 1.0 에서 정확도가
                   # 계단식으로 좋아집니다 (s101_seg12_knife: 손목 회전오차 p90 17.8->14.8도,
                   # 손끝 거리 p90 23.7->21.6 mm; 2.0 에서 추가 이득 없음). 관절 가속도(2차 차분)
                   # p95 는 0.2 대비 pot 22.4->9.0, knife 22.2->13.2 mrad 로 개선됩니다.
                   # 이유: 절대 위치 대응이 55개(예제는 13개)라 서로 충돌하는데 평활이 그 충돌을
                   # 시간축으로 평균해 더 나은 해를 찾습니다. 대가는 루트 회전 튐 3.18->4.04도.
                   # 2.0 은 발바닥 표준편차가 0.57->0.66 cm 로 나빠져 채택하지 않았습니다.
                   joint_smoothness=_w("W_SMOOTH", 1.0),
                   # [root-xy] 오프셋 포함 골반 xy ↔ SMPL-X 골반 xy. 스윕(0~100, knife/pot) 결과 5.0.
                   # 0 에서는 골반이 사람 골반에서 6.8~7.6 cm 어긋나 있었습니다 — global_align 의
                   # (0,"pelvis") 대응이 var_offset 을 보지 못해 var_root 좌표계에서만 만족되기 때문.
                   # 5.0: 골반 오차 1.7 cm(-77%), knife 손목 회전오차 p90 14.8->9.8도(-34%),
                   #      손끝 p90 21.6->18.1 mm, pot 손끝 p90 24.4->23.2 mm(구간 최소),
                   #      pot 루트 튐 p99 25.1->19.1 mm, knife 부양 8->4%.
                   # 8~20 은 골반 오차를 0.2~0.8 cm 까지 줄이고 pot 루트 튐을 16.1 mm 로 더 내리지만
                   # pot 손목 p90 이 11.6->12.6~13.0 도로 나빠지고, 8 은 knife 에서 국소해에 빠집니다
                   # (손끝 p90 23.0, 손목 p90 12.3 — 4/5/12/16 보다 나쁨).
                   # 100 은 pot 이 붕괴합니다: 오프셋 z -0.094->-0.253 m, 발바닥 -5.34 cm, 관통 96%,
                   # 손목 회전오차 43.4도. 5.0 은 그 지점까지 20배 여유입니다.
                   root_xy=_w("W_ROOTXY", 5.0),
                   # [V2] STAGE 2 전용 손 항목.
                   # hand_local(손 체인 상대벡터 + 공유 스케일)은 실측에서 해로워 2026-09-04 제거했습니다
                   # (s53_seg19_knife: 손목 회전오차 24.1->50.6도, 손끝 거리 39.7->57.2 mm, VRAM 11->19 GB).
                   # 근거였던 "손가락별 크기 비 불균일"은 어긋난 대응표의 허상이었고 대응을 고친 뒤
                   # 비율이 1.06~1.46으로 균일해졌습니다. 복원은 .pre_costcleanup.bak 참고.
                   # wrist_orient 은 채택. v1에는 손목 회전 목표가 아예 없어서(retarget npz 에
                   # 손바닥 쿼터니언이 없음) 손목 방향이 다른 항들의 부산물이었습니다. 사람과 로봇
                   # 양쪽에서 손 키포인트로 같은 방식의 프레임을 만들어 비교하므로 좌표 규약에
                   # 의존하지 않습니다. s53_seg19_knife 실측: 손목 회전오차 24.1도 -> 14.4도,
                   # 손끝-사람 거리 39.7 -> 37.9 mm, VRAM 증가 없음.
                   wrist_orient=_w("W_WRISTORI", 1.0))

    # per-axis, per-body-part global weights (n_corr,3): body order = pelvis(0) torso(1) arms(2-7)
    # legs(8-13) hands(14+). Default = uniform full tracking; the low-weight decoupling is opt-in via env.
    # [cost-cleanup 2026-09-04] 골반을 축별로 나눠 두던 것(W_PELVISXY / W_PELVISZ, Z 만 0.0)을
    # 없애고 다른 부위와 같은 단일 가중치로 통일했습니다. Z 를 0 으로 뺐던 이유는 골반 높이를
    # root_height 가 담당한다는 전제였는데 그 항을 제거했고, 실측에서 Z 를 1.0 으로 켜도 결과가
    # 바뀌지 않습니다 — global_align 은 var_offset 을 보지 않아서(최종 루트 = offset ∘ var_root)
    # 골반 z 목표가 오프셋에 그대로 흡수됩니다. 즉 축별 구분 자체가 의미가 없었습니다.
    w_pelvis = _w("W_PELVIS", 1.0)
    w_trunk, w_arm, w_leg = _w("W_TRUNK", 1.0), _w("W_ARM", 1.0), _w("W_LEG", 1.0)
    # ── [ROLLBACK MARKER: smplx-kpts] 부위 판별을 위치 인덱스 -> 링크 이름으로 ───────────────
    # 옛 코드는 gw[1]=torso, gw[2:8]=arms, gw[8:14]=legs, gw[14:]=hands 처럼 _BODY 의 고정 순서를
    # 전제했습니다. SMPL-X 전환에서 torso 대응을 빼고 순서를 바꾸자 어깨에 몸통 가중치가, 고관절에
    # 팔 가중치가, palm 에 다리 가중치가 걸렸습니다(조용히 통과 — 길이가 우연히 맞았습니다).
    # 이름으로 판별하면 대응표를 고쳐도 가중치가 따라옵니다.
    def _part(link: str) -> str:
        if link.startswith("robot0_"):
            return "hand"
        if "torso" in link:
            return "trunk"
        if any(k in link for k in ("shoulder", "elbow", "wrist")):
            return "arm"
        if any(k in link for k in ("hip", "knee", "ankle")):
            return "leg"
        return "pelvis"
    _a_names = [l for _, l, _ in pairs]
    gw = onp.zeros((len(a_para), 3), onp.float32)
    _gwv = {"pelvis": w_pelvis * weights["global_alignment"],
            "trunk": w_trunk * weights["global_alignment"],
            "arm": w_arm * weights["global_alignment"],
            "leg": w_leg * weights["global_alignment"],
            "hand": weights["hand_alignment"]}
    for _i, _ln in enumerate(_a_names):
        gw[_i] = _gwv[_part(_ln)]
    # [s1-nohand-test 2026-09-04] STAGE 1 의 일은 접지이고, 2단계가 하반신을 얼린 뒤 상반신을 다시
    # 풀기 때문에 1단계가 만든 손 자세는 어차피 버려집니다. 그런데 손 대응 42개(손당 21개)가 각각
    # 사람 손 위치(식탁 높이)를 향해 절대 위치로 당기고 있어서, 접지를 방해하는 상시 상향 압력이
    # 됩니다 — 예제 12 의 절대 정렬은 몸통 13개뿐입니다. 기본 ON 이고 W_S1NOHAND=0 으로 끕니다
    # (gw 의 hand 행 = 0). 손끝 접촉(contact_grasp)은 이미 2단계 전용입니다.
    # 실측(s101_seg29_pot): 손끝 거리 p90 26.3->23.3 mm, 손목 회전오차 p90 12.3->10.8도, 접지·발
    # 미끄러짐 불변. s100_seg00_pan 은 손끝 중앙 14.3->15.0 mm 로 0.7 mm 나빠지고 p90 은 동일.
    # 1단계 절대 정렬 잔차가 165->39 개로 줄어 예제 12 와 같은 구성이 됩니다.
    if int(os.environ.get("W_S1NOHAND", 1)):     # [s1-nohand] 기본 ON (아래 실측 근거)
        _nh_off = 0
        for _i, _ln in enumerate(_a_names):
            if _part(_ln) == "hand":
                gw[_i] = 0.0; _nh_off += 1
        print(f"[s1-nohand] STAGE 1 에서 손 대응 {_nh_off}개 비활성 (몸통 {len(_a_names)-_nh_off}개만)")
    # [ROLLBACK MARKER: fingertip-align] 손끝 10개/손은 각 손 블록의 뒤쪽에 붙어 있습니다.
    # 기본값은 다른 손 키포인트와 동일(균일)입니다. 참고로 env 보상은 손끝을 더 중시합니다
    # (rew_fingertip -6.0 vs rew_hand_kpts -1.5, 둘 다 그룹 평균이라 점당 4:1). 최소제곱에서
    # 중요도는 가중치의 제곱이므로 그 비율을 재현하려면 W_HANDTIP = 2.0 x W_HAND 입니다.
    # 올리면 손끝이 다른 손 키포인트를 밀어낼 수 있으니 스윕으로 확인하고 쓰십시오.
    # [smplx-kpts] 손끝은 위치가 아니라 "링크-로컬 오프셋이 0 이 아닌 대응"으로 식별합니다
    # (_build_correspondence 에서 손끝만 pad 오프셋을 갖습니다) — 블록 경계에 의존하지 않습니다.
    _wtip = _w("W_HANDTIP", 0.0) or weights["hand_alignment"]
    _tip_rows = onp.abs(a_off).sum(axis=1) > 1e-9
    gw[_tip_rows] = _wtip

    # per-node LOCAL weight (nb=14 body): arms weak so the arm's relative structure yields to the contact
    # cost (reach the object from the grounded-lower body); pelvis/torso/legs keep full local.
    w_armlocal = _w("W_ARMLOCAL", 1.0)                   # default full; lower (env) frees the arm for grounding
    lw = onp.ones(len(b_para), onp.float32)
    for _i, (_, _ln) in enumerate(_BODY):                # [smplx-kpts] 이름으로 팔 판별
        if _part(_ln) == "arm":
            lw[_i] = w_armlocal

    root_R_target = _pelvis_target_R(jp)     # (F,3,3) keypoint-derived pelvis orientation target
    heightmap = _flat_heightmap(jp)          # flat z=0 floor spanning the clip

    t0 = time.time()
    Ts_root, joints = solve(robot, robot_coll, heightmap, jnp.array(jp_ext), b_para, b_link, b_mask, a_para,
                            a_link, jnp.array(a_off), jnp.array(gw), jnp.array(lw), jnp.array(l_c), jnp.array(r_c),
                            jnp.array(l_kp), jnp.array(r_kp), left_foot_idx, right_foot_idx,
                            left_knee_idx, right_knee_idx,
                            jnp.array(root_R_target), jnp.array(ft_idx), jnp.array(ft_off),
                            jnp.array(ft_margin), jnp.array(ft_pad), jnp.array(ft_mask),
                            jnp.array(rest_w), weights,
                            h_sets=h_sets)
    joints = onp.array(joints); root = onp.array(Ts_root.wxyz_xyz)
    print(f"[pyroki-retarget] stage-1 solved in {time.time()-t0:.1f}s")

    # ---- STAGE 2 (W_STAGE2>0 = pin weight; ON by default): freeze LOWER body (legs) + root + offset at the
    # stage-1 solution and re-solve the UPPER body (waist+arms+hands) with STRONG hand keypoint alignment, so
    # the arms reach UP to the human hand keypoints without un-grounding the (stage-1) feet. W_STAGE2=0 → skip
    # (single-stage baseline). ----
    w_stage2 = _w("W_STAGE2", 100.0)
    if w_stage2 > 0.0:
        lower_mask = onp.array([1.0 if any(k in nm for k in ("hip", "knee", "ankle")) else 0.0
                                for nm in an], onp.float32)      # freeze legs; waist+arms+fingers stay free
        # [smplx-kpts] 여기도 위치 인덱스 -> 이름 기반. 옛 gw2[8:14]/gw2[14:] 는 _BODY 순서 전제였습니다.
        gw2 = gw.copy()
        _s2arm, _s2hand = _w("W_STAGE2ARM", 1.0), _w("W_STAGE2HAND", 5.0)
        for _i, _ln in enumerate(_a_names):
            _p = _part(_ln)
            if _p in ("pelvis", "leg"):
                gw2[_i] = 0.0                                    # 하체+골반 고정 → 추종 안 함
            elif _p == "arm":
                gw2[_i] = _s2arm                                 # 팔 (wrist_yaw = 손 기준점)
            elif _p == "hand":
                gw2[_i] = _s2hand                                # 손 강하게 (팔을 키포인트로 끌어올림)
        s2_off = onp.zeros((F, 3), onp.float32)                 # stage-1 root already baked → pin offset→0
        t1 = time.time()
        Ts_root, joints2 = solve(robot, robot_coll, heightmap, jnp.array(jp_ext), b_para, b_link, b_mask, a_para,
                                 a_link, jnp.array(a_off), jnp.array(gw2), jnp.array(lw), jnp.array(l_c), jnp.array(r_c),
                                 jnp.array(l_kp), jnp.array(r_kp), left_foot_idx, right_foot_idx,
                                 left_knee_idx, right_knee_idx,
                                 jnp.array(root_R_target), jnp.array(ft_idx),
                                 jnp.array(ft_off), jnp.array(ft_margin), jnp.array(ft_pad), jnp.array(ft_mask),
                                 jnp.array(rest_w), weights,
                                      h_sets=h_sets,
                                 s2_joints=jnp.array(joints), s2_root=jnp.array(root),
                                 s2_offset=jnp.array(s2_off), s2_lower_mask=jnp.array(lower_mask), s2_w=w_stage2)
        joints = onp.array(joints2); root = onp.array(Ts_root.wxyz_xyz)
        print(f"[pyroki-retarget] STAGE 2 (freeze lower + reach hands) solved in {time.time()-t1:.1f}s")

    solved = {name: joints[:, i] for i, name in enumerate(an)}
    act = list(_ORDER["action_joint_names"])
    # [ROLLBACK MARKER: tendon-ineq] 부등식 모드에서는 J0 8개가 자유 변수로 풀립니다. 65열 액션
    # 레이아웃에는 J0 가 없어 그대로 두면 풀린 값이 버려집니다 — 뒤에 덧붙여 저장합니다.
    # (등식 mimic 모드에서는 solved 에 J0 가 없으므로 이 블록이 자동으로 no-op 입니다.)
    _extra = [n for n in (f"robot0_{sd}_{fg}J0" for sd in "lr" for fg in ("FF", "MF", "RF", "LF"))
              if n in solved and n not in act]
    if _extra:
        act = act + _extra
        print(f"[tendon-ineq] 풀린 J0 {len(_extra)}개를 출력에 덧붙임 → g1_joint_pos 폭 {len(act)}")
    g1_joint_pos = onp.zeros((F, len(act)), onp.float32)
    nmap = 0
    for j, name in enumerate(act):
        if name in solved:
            g1_joint_pos[:, j] = solved[name]; nmap += 1
    g1_root_pose = onp.concatenate([root[:, 4:7], root[:, 0:4]], axis=1)

    # ---- post-solve contact residual [DIAGNOSTIC]: how far each in-contact robot contact point ends up
    # from its target after the solve. Tests whether the contact cost actually reaches its targets (small)
    # or is overridden by the keypoint anchors (large). Split fingertip pads vs wrap links (palm+base). ----
    try:
        _fk = jax.vmap(robot.forward_kinematics)(jnp.array(joints))              # (F, nlinks, 7) local wxyz_xyz
        _Tf = jaxlie.SE3(_fk)
        _lp = onp.array(_Tf.translation()); _lR = onp.array(_Tf.rotation().as_matrix())     # (F,nl,3),(F,nl,3,3)
        _Tr = jaxlie.SE3(jnp.array(root))
        _rR = onp.array(_Tr.rotation().as_matrix()); _rp = onp.array(_Tr.translation())     # (F,3,3),(F,3)
        _pos = onp.einsum("fij,flj->fli", _rR, _lp) + _rp[:, None, :]            # (F,nl,3) world link pos
        _Rm = onp.einsum("fij,fljk->flik", _rR, _lR)                            # (F,nl,3,3) world link rot
        _cp = _pos[:, ft_idx] + onp.einsum("flij,lj->fli", _Rm[:, ft_idx], ft_off)   # (F,P,3) robot contact pts
        _gap = onp.linalg.norm(_cp - ft_pad, axis=-1)                            # (F,P) to target
        _m = ft_mask > 0.5; _grp = onp.array(c_group)
        parts = []
        for g in ("palm", "finger", "tip"):
            sel = _grp == g
            gg = _gap[:, sel][_m[:, sel]]
            parts.append(f"{g} mean {gg.mean()*100:.1f} cm max {gg.max()*100:.1f} cm (n={gg.size})"
                         if gg.size else f"{g} none")
        print("[metric] post-solve contact gap (in-contact):  " + "  |  ".join(parts))
    except Exception as _e:
        print(f"[metric] skipped ({type(_e).__name__}: {_e})")

    out = _PROC / "g1_shadow" / args.cls / args.clip / "0" / f"trajectory_pyroki{os.environ.get('W_OUTSUFFIX', '')}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Record the column layout with the data. g1_joint_pos is written in `act` order, and the env
    # reads column k into its OWN k-th action joint — an agreement nothing enforced, since `act`
    # comes from a static json dump of the robot's PhysX DOF order. Rebuilding G1_shadow.usd
    # repermutes the env side and silently crosses the columns (it did: 24 of 65, all hands, the
    # middle finger driven by the thumb). With the names alongside, the env matches by name.
    onp.savez(out, g1_joint_pos=g1_joint_pos.astype(onp.float32),
              g1_root_pose=g1_root_pose.astype(onp.float32),
              joint_names=onp.array(act, dtype=object))
    print(f"[pyroki-retarget] wrote {out}  ({nmap}/{len(act)} action joints solved)")


if __name__ == "__main__":
    main()
