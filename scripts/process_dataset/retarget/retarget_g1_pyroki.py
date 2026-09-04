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
  [V2: hand-local-scale]  Hand local alignment + a shared hand scale matrix. Measured HARMFUL and
        left OFF by default (W_HANDLOCAL=0) — see the weight's comment for the numbers.

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
_URDF = _ROOT / "data" / "robots" / "G1" / "urdf_pyroki" / "g1_shadow.urdf"
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

# ---- ParaHome joint idx → composite-URDF link name ----------------------------------------
# body: our BODY_KPTS correspondence (minus ball-foot dups 18/22 → ankle_roll used at 17/21)
_BODY = [
    (0, "pelvis"), (4, "torso_link"),
    (8, "right_shoulder_pitch_link"), (9, "right_elbow_link"), (10, "right_wrist_yaw_link"),
    (12, "left_shoulder_pitch_link"), (13, "left_elbow_link"), (14, "left_wrist_yaw_link"),
    (15, "right_hip_pitch_link"), (16, "right_knee_link"), (17, "right_ankle_roll_link"),
    (19, "left_hip_pitch_link"), (20, "left_knee_link"), (21, "left_ankle_roll_link"),
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
_HAND_CHAIN = {
    "wrist":  ([0], ["palm"]),
    "index":  ([18, 19, 20], ["ffknuckle", "ffmiddle", "ffdistal"]),
    "middle": ([14, 15, 16], ["mfknuckle", "mfmiddle", "mfdistal"]),
    "ring":   ([10, 11, 12], ["rfknuckle", "rfmiddle", "rfdistal"]),
    "pinky":  ([6, 7, 8], ["lfknuckle", "lfmiddle", "lfdistal"]),
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
    "thumb":  ([22, 23], ["thmiddle", "thdistal"]),
}


# ── [ROLLBACK MARKER: fingertip-align] 손끝을 대응에 통합 ─────────────────────────────────────
# 손끝은 링크 원점이 아니라 말단 링크에서 pad 오프셋만큼 떨어진 점이라 _HAND_CHAIN 으로는 표현할
# 수 없었습니다. 처음에는 별도 비용(fingertip_align)으로 넣었는데, 그러면 jaxls 가 그 비용을 위해
# 순기구학을 따로 미분해서 프레임당 FK 야코비안이 한 벌 더 생깁니다 — 이 파일의 다른 주석이 같은
# 이유로 손목 방향 잔차를 contact_grasp 안에 합쳤고, 그때 실측이 +5.3 GB 였습니다.
# 그래서 대응 자체에 오프셋을 실어 global_align 하나로 처리합니다. FK 를 나눠 쓰고, 가중치도
# gw 한 곳에서 다른 키포인트와 같은 방식으로 정해집니다.
#
# 목표는 사람 fingertip_pad_pos (F,10,3) 인데 joint_positions (F,73,3) 와 다른 배열이라,
# main 에서 두 배열을 이어붙여 (F,83,3) 로 만들고 손끝은 뒤쪽 인덱스 73.. 를 가리킵니다.
_PAD_BASE = 73                                   # jp(73) 뒤에 fingertip_pad_pos 10개를 이어붙인 시작
_PAD_ORDER = ("th", "ff", "mf", "rf", "lf")      # fingertip_pad_pos 의 손별 순서


def _build_correspondence():
    """(parahome_idx, urdf_link_name, distal-local offset) 3-튜플 목록.

    오프셋이 0 이 아닌 것은 손끝 10개뿐입니다(말단 링크 -> pad). 나머지는 링크 원점입니다.
    """
    pairs = [(p, l, [0.0, 0.0, 0.0]) for p, l in _BODY]
    for side, off in (("l", 23), ("r", 48)):
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


_FOOT_PLANT_H, _FOOT_PLANT_VZ, _FPS = 0.06, 0.15, 30.0
_PARA_BALL_L, _PARA_BALL_R = 22, 18
# G1 ankle_roll_link origin sits this far ABOVE the foot sole (URDF foot-corner contact spheres at
# z=-0.031, r=0.005 → sole at -0.036). floor_contact targets the ANKLE, so its z target must be this
# offset (not 0) — else pulling the ankle to z=0 drives the sole ~3.6 cm INTO the floor (feet penetrate
# at high W_FLOOR) or, when weak, the balance leaves the foot floating. Target = sole-on-floor.
_ANKLE_SOLE_OFF = 0.036
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
    z = jp[:, ball_idx, 2]
    vz = onp.zeros_like(z); vz[1:] = (z[1:] - z[:-1]) * _FPS
    return ((z < _FOOT_PLANT_H) & (onp.abs(vz) < _FOOT_PLANT_VZ)).astype(onp.float32)


def _pelvis_target_R(jp):
    """Per-frame target pelvis orientation (world) from ParaHome body keypoints, in the G1 pelvis-link
    convention X=forward, Y=left, Z=up.  up = midshoulder→ (spine), left = right→left hip, forward = left×up.
    Constraining the free root to this removes the spurious roll that saturates hip_roll/waist_roll."""
    p_rs, p_ls = jp[:, 8], jp[:, 12]      # right / left shoulder
    p_rh, p_lh = jp[:, 15], jp[:, 19]     # right / left hip
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
    xy = jp[:, :23, :2].reshape(-1, 2)
    lo = xy.min(0) - margin; hi = xy.max(0) + margin
    cx, cy = (lo + hi) / 2.0; dx, dy = (hi - lo)
    box = trimesh.creation.box(extents=[float(dx), float(dy), 0.1])
    box.apply_translation([float(cx), float(cy), -0.05])          # top face at z=0
    return pk.collision.Heightmap.from_trimesh(box, x_bins=bins, y_bins=bins)


# [SCENE-COLLISION] ---------------------------------------------------------------------------
def _context_boxes(sm, obj_key, jp_cloud, radius=1.0, support_radius=1.5,
                   reach=float(os.environ.get("W_SCENEREACH", 0.2)),
                   max_boxes=int(os.environ.get("W_SCENEMAXBOX", 64))):
    """Boxes approximating the fixed scene the env spawns, for the world_collision cost.

    The floor heightmap above was the ONLY thing the retarget collided against, so the counter, sink
    and board simply were not in the problem. The solve happily put the LEFT hand inside the
    countertop, and the sim then threw it out at up to 5.47 m/s (470 of 501 frames) — every bit of
    that impulse lands on the robot, because the context spawns kinematic.

    Geometry choice, measured on s101_seg12_knife (78 robot capsules x 301 frames):

        one OBB per object     max penetration 36.6 cm, 21493 pairs — mostly the OBB's own slack
                               (the sink mesh fills 31% of its OBB), and it traps the LEGS
        per-hull OBBs          max  6.0 cm,  1813 pairs, and the deepest links are the LEFT HAND —
                               which is the defect we are actually chasing
        heightmap              max 117 cm, deepest links the ANKLES: a height field is solid all the
                               way down, so a robot standing at a counter reads as buried in it

    So: convex-decompose, then take each hull's OBB. pyroki has no hull primitive, but Capsule-Box
    is native and `capsule_box` uses a real box SDF (inside, the depth is to the NEAREST FACE), which
    is what keeps a pelvis grazing the counter edge at ~4 cm instead of the ~80 cm a height field
    would report.

    Hull count is NOT controllable here: trimesh delegates to `pyVHACD.compute_vhacd`, which takes no
    keyword arguments, so the decomposition runs at its own default (~64 hulls on the sink). The
    simulator converts with `max_convex_hulls=16`, so the two differ — the retarget sees a somewhat
    tighter scene than PhysX does. Taking the OBB of each hull inflates it back, which pushes the
    other way; the net is verified by measurement, not assumed. Aligning them properly means
    re-converting the context USDs at a matching hull count.

    Selection mirrors the env (proximity of the frame-0 centroid to the active object's swept path
    + the below-object support), so both see the same scene.
    """
    act = sm[obj_key].astype(onp.float64)
    act_xy, act0 = act[:, :2], act[0]
    cands = []
    for k in (kk for kk in sm.files if kk.startswith("ctx__") and kk.endswith("__base")):
        pose0 = sm[k][0].astype(onp.float64)
        dmin = float(onp.linalg.norm(act_xy - pose0[None, :2], axis=1).min())
        cands.append((k.split("__")[1], pose0, dmin))
    keep = {n for n, _, dm in cands if dm < radius}
    below = [(float(onp.linalg.norm(act0[:2] - q[:2])), n) for n, q, _ in cands
             if q[2] < act0[2] and float(onp.linalg.norm(act0[:2] - q[:2])) < support_radius]
    if below:
        keep.add(min(below)[1])

    centres, extents = [], []
    for name, q, _ in cands:
        if name not in keep:
            continue
        src = _RAW_SCAN / name / "simplified" / "base.obj"
        if not src.exists():
            print(f"[pyroki-retarget] scene: no scan mesh for {name}, skipped")
            continue
        mesh = trimesh.load(str(src), process=False, force="mesh")
        rot = trimesh.transformations.quaternion_matrix([q[3], q[4], q[5], q[6]])[:3, :3]
        mesh.vertices = mesh.vertices @ rot.T + q[:3]
        parts = mesh.convex_decomposition()                       # pyVHACD; takes no kwargs
        parts = parts if isinstance(parts, list) else [parts]
        if len(parts) <= 1:
            # A single hull means the object collapsed to its own bounding volume, which for a
            # counter is 3x its real volume and would trap the robot's legs. Loud on purpose: the
            # earlier silent fallback to one OBB per object is what made the first run a no-op.
            print(f"[pyroki-retarget] scene: WARNING {name} decomposed to {len(parts)} hull — its OBB "
                  f"will be far larger than the object")
        for prt in parts:
            v = onp.asarray(prt.vertices)
            lo, hi = v.min(0), v.max(0)
            if (hi - lo).min() < 1e-4:                            # degenerate sliver
                continue
            centres.append((lo + hi) / 2.0)
            extents.append(hi - lo)
    if not centres:
        return None
    centres, extents = onp.asarray(centres), onp.asarray(extents)
    # Prune to what the body can plausibly reach. This is not an optimisation nicety: the residual is
    # (frames x capsules x boxes) and jaxls differentiates through all of it, so the unpruned 576
    # boxes asked for 40 GB against a 27 GB device and the solve died. Distance is measured from the
    # HUMAN body-keypoint cloud (the robot tracks it) to each box's surface, so a box is kept only if
    # some body point passes within `reach` of it. At 0.3 m this keeps 179 of 576 — the cutting board
    # and counter the hands work on — and drops the fridge/microwave the robot never approaches.
    # The budget is tight because jaxls differentiates the whole (frames x capsules x boxes)
    # block: 576 boxes needed 40 GB, 180 needed 21.6 GB, against a 32 GB device. 64 boxes at
    # 0.2 m is the cutting board + countertop, which is exactly where the left hand sinks in;
    # the legs that a larger set would also constrain were measured to be FALSE positives of
    # the OBB slack, and grounding is already owned by the floor heightmap.
    kp = jp_cloud.reshape(-1, 3)
    gap = onp.abs(kp[:, None, :] - centres[None, :, :]) - extents[None, :, :] / 2.0
    d = onp.linalg.norm(onp.clip(gap, 0.0, None), axis=-1).min(0)         # (K,) surface distance
    m = d < reach
    if m.sum() > max_boxes:                                              # keep the closest
        m = onp.zeros_like(m); m[onp.argsort(d)[:max_boxes]] = True
    centres, extents = centres[m], extents[m]
    print(f"[pyroki-retarget] scene collision: {len(centres)} boxes (of {len(d)}) from {sorted(keep)} "
          f"(kept within {reach} m of the body keypoints)")
    return pk.collision.Box.from_extent(
        extent=jnp.array(extents, jnp.float32),
        position=jnp.array(centres, jnp.float32),
        wxyz=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0], jnp.float32), (len(centres), 1)))
# [/SCENE-COLLISION] --------------------------------------------------------------------------


# [OBJECT-COLLISION] ---------------------------------------------------------------------------
def _object_boxes(obj_name, max_boxes=int(os.environ.get("W_OBJCOLLMAXBOX", 64))):
    """Boxes approximating the MANIPULATED object, expressed in the OBJECT'S OWN frame.

    The manipulated object was never in the retarget problem — only the floor (and optionally the
    context furniture). Nothing stopped the solve from putting a finger INSIDE the knife, and it did:
    measured on the reference poses of s101_seg12_knife, `robot0_r_ffmiddle` sits inside the knife
    mesh in 60.9% of frames at 4.8 mm mean depth (every other right-hand link is under 1%). The sim
    then has to resolve that overlap at episode start, which is the "hand stuck in the object" the
    training runs show.

    Local frame, not world: unlike the context furniture this object MOVES every frame, so the boxes
    are built once here in the object's own frame and transformed per frame inside the cost, where
    the frame index is already the vmapped axis.

    Same geometry recipe as `_context_boxes` (convex-decompose, then take each hull's box) for the
    same reason — one box per object is far larger than the object, and pyroki has no hull primitive.
    The knife decomposes to 64 hulls of 8.5 mm median thickness, so the per-hull boxes stay thin
    enough that a fingertip can rest ON the blade without the box claiming it is inside.
    """
    src = _RAW_SCAN / obj_name / "simplified" / "base.obj"
    if not src.exists():
        print(f"[pyroki-retarget] object collision: no scan mesh for {obj_name}, term disabled")
        return None
    mesh = trimesh.load(str(src), process=False, force="mesh")
    parts = mesh.convex_decomposition()                       # pyVHACD; takes no kwargs
    parts = parts if isinstance(parts, list) else [parts]
    centres, extents, vols = [], [], []
    for prt in parts:
        v = onp.asarray(prt.vertices)
        lo, hi = v.min(0), v.max(0)
        if (hi - lo).min() < 1e-4:                            # degenerate sliver
            continue
        centres.append((lo + hi) / 2.0)
        extents.append(hi - lo)
        vols.append(float(onp.prod(hi - lo)))
    if not centres:
        return None
    centres, extents, vols = onp.asarray(centres), onp.asarray(extents), onp.asarray(vols)
    if len(centres) > max_boxes:                              # keep the bulkiest hulls
        k = onp.argsort(-vols)[:max_boxes]
        centres, extents = centres[k], extents[k]
    inflate = float(extents.prod(1).sum() / max(sum(p.volume for p in parts), 1e-9))
    print(f"[pyroki-retarget] object collision: {len(centres)} boxes for '{obj_name}' "
          f"(hulls {len(parts)}, box volume {inflate:.1f}x the hulls')")
    return onp.asarray(centres, onp.float32), onp.asarray(extents, onp.float32)
# [/OBJECT-COLLISION] --------------------------------------------------------------------------


def _give_hands_collision_geometry(urdf):
    """Copy each hand link's VISUAL mesh into its (empty) collision slot, in place.

    Without this the object-collision term below is aimed at nothing. The Shadow hand links in this
    URDF carry no <collision> element at all, so pyroki fits them a ZERO-radius capsule — measured:
    every `robot0_*` finger link has radius 0.0 mm, and the whole hand is represented by a single
    65.2 mm-radius capsule on the palm (whose real thickness is 34 mm). Constraining that stand-in
    made penetration WORSE, not better: it pushed the oversized palm ball clear of the knife and
    drove the index finger — which the problem could not see — 4.6 mm -> 6.0 mm deeper.

    The palm's own two boxes are left alone; its capsule is a poor fit either way (a capsule around
    a flat palm is inflated no matter what it is fitted to), and the fingers are what penetrate.
    """
    n = 0
    for name, link in urdf.link_map.items():
        if not name.startswith("robot0_") or link.collisions or not link.visuals:
            continue
        for v in link.visuals:
            if v.geometry is None or v.geometry.mesh is None:
                continue
            link.collisions.append(yourdfpy.Collision(name=f"{name}_coll", origin=v.origin,
                                                      geometry=v.geometry))
            n += 1
    print(f"[pyroki-retarget] hand collision: filled {n} empty collision slots from the visual meshes")
    return urdf


def _hand_sets(robot):
    """[V2: hand-local-scale] 손 하나짜리 국소 정렬에 쓸 대응 집합. side -> (사람 인덱스, 로봇 링크 인덱스).

    몸통과 같은 방식(쌍별 상대 벡터 + 방향)을 손 체인에도 적용하기 위한 것입니다. v1에서 손은
    전역 정렬(절대 위치)만 받았는데, 로봇 손이 사람 손보다 약 1.2배 크므로 절대 위치를 맞추려면
    손가락이 안쪽으로 눌립니다 — 크기 차이를 흡수할 통로가 없었습니다. pyroki 11번 손 예제는
    바로 이 스케일 행렬로 그 차이를 흡수합니다.
    """
    out = {}
    for side, off in (("l", 23), ("r", 48)):
        hp, hl = [], []
        for local, shadow in _HAND_CHAIN.values():
            for pl, sh in zip(local, shadow):
                hp.append(off + pl)
                hl.append(robot.links.names.index(f"robot0_{side}_{sh}"))
        out[side] = (hp, hl)
    return out


def solve(robot, robot_coll, heightmap, scene_boxes, keypoints, b_para, b_link, b_mask, a_para, a_link, a_off, gw, lw,
          l_contact, r_contact, l_foot_kp, r_foot_kp, left_foot_idx, right_foot_idx,
          left_knee_idx, right_knee_idx, root_R_target, root_z_target, ft_idx, ft_off, ft_margin, ft_target,
          ft_mask, rest_w, weights,
          obj_boxes_local=None, obj_pose=None, obj_capsule_idx=None, h_sets=None,
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
    nh = len(h_sets["r"][0]) if h_sets else 1
    class HandScaleVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((nh, nh))): ...

    class OffsetVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros((3,))): ...

    var_joints = robot.joint_var_cls(jnp.arange(T))
    var_root = jaxls.SE3Var(jnp.arange(T))
    var_scale = ScaleVar(jnp.zeros(T))
    var_hscale = HandScaleVar(jnp.zeros(T))       # [V2] 전 프레임 공유 (몸통 스케일과 같은 방식)
    var_offset = OffsetVar(jnp.arange(T))         # per-frame world translation (feet-on-floor placement)

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
    @jaxls.Cost.factory
    def hand_scale_reg(vv, v_hs: HandScaleVar):
        s = vv[v_hs]
        return jnp.concatenate([(s - 1.0).flatten() * 1.0, (s - s.T).flatten() * 100.0,
                                jnp.clip(-s, min=0).flatten() * 100.0]) * weights["hand_scale_reg"]

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

    @jaxls.Cost.factory
    def floor_contact(vv, v_root: jaxls.SE3Var, v_cfg, v_off: OffsetVar, lc, rc, lkp, rkp):
        T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
        off = vv[v_off]
        lpos = T_wl.translation()[left_foot_idx] + off; rpos = T_wl.translation()[right_foot_idx] + off
        lz = T_wl.rotation().as_matrix()[left_foot_idx][2, 2]
        rz = T_wl.rotation().as_matrix()[right_foot_idx][2, 2]
        return jnp.concatenate([
            (lc * (lpos - lkp)).flatten(), (rc * (rpos - rkp)).flatten(),
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
                _t = [0, 1, 4]                      # _HAND_CHAIN 순서: palm, ff*3, mf*3, ...
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
    def root_height(vv, v_root: jaxls.SE3Var, z_tgt):
        # Track pelvis HEIGHT at human_pelvis_z × leg_ratio (per frame): grounds the feet across ALL
        # motions (squat/step/bob preserved proportionally) while keeping the body upright — no crouch
        # collapse (free-z failure) and no floating (human-height tracking failure). z_tgt: scalar/frame.
        return ((vv[v_root].translation()[2] - z_tgt) * weights["root_height"]).reshape(1)

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
    def offset_reg(vv, v_off: OffsetVar):
        # Pin the HORIZONTAL offset (xy) strongly so the free offset can't drift the whole robot off the
        # keypoints (root xy already places the robot horizontally); leave Z governed by offset_reg so
        # world_collision can lift the feet onto the floor (per-pose grounding, squat-preserving) when
        # offset_reg (z) is small. offset_xy default high; set offset_reg=0 to free vertical grounding.
        o = vv[v_off]
        return jnp.concatenate([(o[..., :2] * weights["offset_xy"]).flatten(),
                                (o[..., 2:3] * weights["offset_reg"]).flatten()])

    @jaxls.Cost.factory
    def world_collision(vv, v_root: jaxls.SE3Var, v_cfg, v_off: OffsetVar):
        # Low weight: high enough to lift the robot up off the floor, low enough not to fight retargeting.
        transform = jaxlie.SE3.from_translation(vv[v_off]) @ vv[v_root]
        coll = robot_coll.at_config(robot, vv[v_cfg]).transform(transform)
        res = [colldist_from_sdf(collide(coll, heightmap), activation_dist=0.005).flatten()]
        # [SCENE-COLLISION] the context objects the env spawns kinematic. Without this term they are
        # absent from the problem entirely, and the solve puts the left hand inside the countertop.
        # coll is (T,B) capsules, scene_boxes is (K,) — broadcast to (T,B,K) via a trailing axis.
        if scene_boxes is not None:
            d = collide(coll.reshape(coll.get_batch_axes() + (1,)), scene_boxes)
            res.append(colldist_from_sdf(d, activation_dist=0.005).flatten()
                       * weights["scene_collision"])
        return jnp.concatenate(res) * weights["world_collision"]

    # [OBJECT-COLLISION] the object the hand is grasping. `o_pose` is (7,) per frame (wxyz + xyz), so
    # the boxes are rebuilt in world coordinates inside the vmapped frame axis.
    if obj_boxes_local is not None:
        _oc_ctr = jnp.asarray(obj_boxes_local[0])                      # (K,3) object frame
        _oc_ext = jnp.asarray(obj_boxes_local[1])                      # (K,3)
        _oc_idx = jnp.asarray(obj_capsule_idx)                         # capsules we constrain
        _oc_tol = weights["object_collision_tol"]

    @jaxls.Cost.factory
    def object_collision(vv, v_root: jaxls.SE3Var, v_cfg, v_off: OffsetVar, o_pose):
        transform = jaxlie.SE3.from_translation(vv[v_off]) @ vv[v_root]
        coll = robot_coll.at_config(robot, vv[v_cfg]).transform(transform)
        # Only the hand capsules are constrained. The residual is (capsules x boxes) per frame and
        # jaxls differentiates all of it (the scene term needed 21.6 GB at 78 capsules x 180 boxes);
        # nothing but a hand is ever inside a hand-held object, so the rest is memory for no term.
        coll = jax.tree.map(lambda x: x[_oc_idx] if getattr(x, "ndim", 0) >= 1
                            and x.shape[0] == robot_coll.num_links else x, coll)
        o_wxyz, o_xyz = o_pose[:4], o_pose[4:]
        R = jaxlie.SO3(o_wxyz).as_matrix()                             # object frame -> world
        boxes = pk.collision.Box.from_extent(
            extent=_oc_ext, position=_oc_ctr @ R.T + o_xyz,
            wxyz=jnp.broadcast_to(o_wxyz, (_oc_ctr.shape[0], 4)))
        d = collide(coll.reshape(coll.get_batch_axes() + (1,)), boxes)
        # One-sided: `colldist_from_sdf(., 0)` is min(d, 0), so a link that merely TOUCHES the object
        # costs exactly nothing and only overlap is charged. That is what lets this term carry a big
        # weight — the scene-collision term had to stay at 1.0 because its 5 mm activation margin
        # pushed on hands that were correctly resting on the surface, and it lost to the keypoint
        # costs anyway. `tol` is the overlap we are willing to leave (the sim resolves that much
        # itself through its contact offset).
        return colldist_from_sdf(d + _oc_tol, activation_dist=0.0).flatten() * weights["object_collision"]
    # [/OBJECT-COLLISION]

    costs = [
        local_align(var_root, var_joints, var_scale, keypoints),
        scale_reg(var_scale),
        global_align(var_root, var_joints, keypoints),
        floor_contact(var_root, var_joints, var_offset, l_contact, r_contact, l_foot_kp, r_foot_kp),
        root_orient(var_root, root_R_target),
        root_height(var_root, root_z_target),
        knee_separation(var_root, var_joints),
        # [V2: contact-stage2-only] 접촉 비용은 아래에서 단계별로 붙입니다.
        root_smooth(jaxls.SE3Var(jnp.arange(1, T)), jaxls.SE3Var(jnp.arange(0, T - 1))),
        skating(jaxls.SE3Var(jnp.arange(1, T)), robot.joint_var_cls(jnp.arange(1, T)),
                OffsetVar(jnp.arange(1, T)), jaxls.SE3Var(jnp.arange(0, T - 1)),
                robot.joint_var_cls(jnp.arange(0, T - 1)), OffsetVar(jnp.arange(0, T - 1)),
                l_contact[:-1], r_contact[:-1]),
        offset_reg(var_offset),
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
    # offset_xy pins the horizontal offset (no drift); offset_reg (z) is free so feet plant. W_WORLDCOLL=0 →
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
    if weights["world_collision"] > 0:
        costs.append(world_collision(var_root, var_joints, var_offset))
    if obj_boxes_local is not None and weights["object_collision"] > 0:
        costs.append(object_collision(var_root, var_joints, var_offset, jnp.asarray(obj_pose)))
    # [V2: hand-local-scale] 손 체인의 국소 정렬 + 스케일. 몸통 local_align 과 같은 형태이되
    # 대상이 한 손이고, 스케일 행렬을 양손이 공유합니다. 절대 위치가 아니라 마디 사이 상대 벡터를
    # 맞추므로, 로봇 손이 1.2배 커도 손가락을 눌러 접지 않고 "같은 모양"을 만들 수 있습니다.
    # 양손 국소정렬 + 양손 손목방향을 하나의 비용으로 묶습니다. 비용을 4개로 나누면 jaxls가
    # 비용마다 순기구학을 따로 미분해서 프레임당 FK 야코비안이 4벌 생기고(65자유도 x 78링크)
    # 호스트 RAM이 터집니다. FK를 한 번만 계산해 네 잔차가 나눠 쓰면 그 4배가 사라집니다.
    def _make_hand_costs(hs_l, hs_r, mask_l, mask_r, w_local, w_ori):
        idx = {s: (jnp.array(hp), jnp.array(hl)) for s, (hp, hl) in (("l", hs_l), ("r", hs_r))}
        masks = {"l": mask_l, "r": mask_r}
        i_w, i_ff, i_mf = 0, 1, 4          # _HAND_CHAIN 순서: palm, ff*3, mf*3, rf*3, lf*3, th*3

        def _frame(p):                     # p (3,3): 손목 / 검지 knuckle / 중지 knuckle
            z = p[2] - p[0]
            z = z / (jnp.linalg.norm(z) + 1e-9)
            x = jnp.cross(z, p[1] - p[0])
            x = x / (jnp.linalg.norm(x) + 1e-9)
            return jnp.stack([x, jnp.cross(z, x), z], axis=-1)

        @jaxls.Cost.factory
        def hand_costs(vv, v_root: jaxls.SE3Var, v_cfg, v_hs: HandScaleVar, kp):
            T_wl = vv[v_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vv[v_cfg]))
            pos = T_wl.translation()
            scale = vv[v_hs][..., None]
            pw_eye = 1 - jnp.eye(nh)
            res = []
            for side in ("l", "r"):
                hp_j, hl_j = idx[side]
                s_pos, r_pos = kp[hp_j], pos[hl_j]
                if w_local > 0:
                    d_s = s_pos[:, None] - s_pos[None, :]
                    d_r = r_pos[:, None] - r_pos[None, :]
                    pw = pw_eye * masks[side]
                    res_pos = (d_s - d_r * scale) * pw[..., None]
                    ds_n = d_s / jnp.linalg.norm(d_s + 1e-6, axis=-1, keepdims=True)
                    dr_n = d_r / jnp.linalg.norm(d_r + 1e-6, axis=-1, keepdims=True)
                    res_ang = (1 - (ds_n * dr_n).sum(-1)) * pw
                    res += [res_pos.flatten() * w_local, res_ang.flatten() * w_local]
                if w_ori > 0:
                    tri = jnp.array([i_w, i_ff, i_mf])
                    R_h, R_r = _frame(s_pos[tri]), _frame(r_pos[tri])
                    res.append(jaxlie.SO3.from_matrix(R_h.T @ R_r).log() * w_ori)
            return jnp.concatenate(res)

        return hand_costs

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
    # 손목 방향은 contact_grasp 안으로 들어갔습니다(FK 공유). 여기 남은 것은 실험용 hand_local 뿐입니다.
    _w_hl = weights["hand_local"]
    if s2_w > 0.0 and h_sets is not None and _w_hl > 0:
        _m = {s: create_conn_tree(robot, jnp.array(h_sets[s][1])) for s in ("l", "r")}
        costs.append(_make_hand_costs(h_sets["l"], h_sets["r"], _m["l"], _m["r"], _w_hl, 0.0)(
            var_root, var_joints, var_hscale, keypoints))
        costs.append(hand_scale_reg(var_hscale))

    init_vals = None
    if s2_w > 0.0:
        # STAGE 2: strong pin holding the lower body (mask) + root + offset at the stage-1 pose; the
        # up-weighted hand global_align (gw hands) then drives the free UPPER body to the hand keypoints.
        _pin_j = jnp.asarray(s2_joints)                       # (T, nJ)
        _pin_root = jnp.asarray(s2_root)                      # (T, 7) wxyz_xyz (stage-1 baked root)
        _pin_off = jnp.asarray(s2_offset)                     # (T, 3)
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
            var_hscale.with_value(jnp.ones((T, nh, nh))),
            var_offset.with_value(_pin_off),
        ])

    prob = jaxls.LeastSquaresProblem(
        costs=costs, variables=[var_joints, var_root, var_scale, var_hscale, var_offset]).analyze()
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
    # [OBJECT-COLLISION] the hand needs real collision shapes before anything can be kept out of the
    # object; only pay for them when a term actually uses them.
    if float(os.environ.get("W_OBJCOLL", 0.0)) > 0:
        _give_hands_collision_geometry(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    sm = onp.load(_PROC / "smplx" / args.cls / args.clip / "0" / "trajectory.npz", allow_pickle=True)
    jp = sm["joint_positions"].astype(onp.float32)
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
    _rest_j0 = float(os.environ.get("W_RESTJ0", -1.0))
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
    weights = dict(tendon_ineq=_w("W_TENDONINEQ", 0.0), tendon_gear_ineq=_w("W_TENDONGEAR", 1.0),
                   local_alignment=_w("W_LOCAL", 2.0), global_alignment=_w("W_GLOBAL", 1.0),
                   hand_alignment=_w("W_HAND", 1.0), floor_contact=_w("W_FLOOR", 3.0),
                   world_collision=_w("W_WORLDCOLL", 1.0), root_orientation=_w("W_ROOTORI", 5.0),
                   root_smoothness=_w("W_ROOTSMOOTH", 1.0), foot_skating=_w("W_SKATE", 1.0),
                   offset_reg=_w("W_OFFSETREG", 0.0), offset_xy=_w("W_OFFSETXY", 1000.0),
                   contact=_w("W_CONTACT", 2.0), contact_margin=_w("W_CONTACTMARGIN", 0.005),
                   knee_separation=_w("W_KNEESEP", 5.0), knee_min=_w("W_KNEEMIN", 0.14),
                   root_height=_w("W_ROOTHEIGHT", 0.0), joint_smoothness=_w("W_SMOOTH", 1.0),
                   # [SCENE-COLLISION] OFF by default. Tried on s101_seg12_knife: the term is added and the solve
                   # converges, but it does not move the result — penetration against its OWN box set went
                   # 1813 -> 1963 pairs (6.03 -> 6.38 cm), and in sim the left hand got WORSE (5.47 -> 6.66
                   # m/s). It is outweighed by the keypoint-tracking costs, which target human hand
                   # positions that sit ON the surface, so matching them necessarily buries the thicker
                   # robot hand. Raising the weight or changing the box set does not address that.
                   # W_SCENECOLL=1 re-enables it; see _context_boxes for the geometry study.
                   scene_collision=_w("W_SCENECOLL", 0.0),
                   # [OBJECT-COLLISION] OFF by default until measured; W_OBJCOLL=50 to enable. The
                   # weight can be large because the residual is one-sided (0 unless links overlap
                   # the object), so at the pose we want the term contributes nothing at all.
                   object_collision=_w("W_OBJCOLL", 0.0),
                   object_collision_tol=_w("W_OBJCOLLTOL", 0.002),
                   # [V2] STAGE 2 전용 손 항목.
                   #
                   # hand_local(손 체인 상대벡터 + 공유 스케일)은 측정 결과 OFF 입니다. 손이 이미
                   # 전역 정렬(절대 위치)과 접촉 비용(물체 표면)에 묶여 있는데 여기에 "상대 모양"까지
                   # 얹으면 같은 링크를 세 방향으로 당기고, 자유로운 스케일 행렬이 손 전체를 돌려
                   # 버립니다. s53_seg19_knife 실측 (손 항 OFF -> hand_local ON):
                   #     손목 회전오차  24.1도 -> 50.6도      손끝-사람 거리  39.7 -> 57.2 mm
                   #     VRAM 피크      11.0 -> 19.0 GB
                   # 애초에 이 항의 근거였던 "손가락마다 크기 비가 제각각"은 어긋난 대응표가 만든
                   # 허상이었고, 대응을 고친 뒤 비율이 1.06~1.46으로 균일해져 흡수할 것이 거의
                   # 남지 않았습니다. 실험용으로만 W_HANDLOCAL=1 로 켭니다.
                   hand_local=_w("W_HANDLOCAL", 0.0),
                   hand_scale_reg=_w("W_HANDSCALEREG", 1.0),
                   # wrist_orient 은 채택. v1에는 손목 회전 목표가 아예 없어서(retarget npz 에
                   # 손바닥 쿼터니언이 없음) 손목 방향이 다른 항들의 부산물이었습니다. 사람과 로봇
                   # 양쪽에서 손 키포인트로 같은 방식의 프레임을 만들어 비교하므로 좌표 규약에
                   # 의존하지 않습니다. s53_seg19_knife 실측: 손목 회전오차 24.1도 -> 14.4도,
                   # 손끝-사람 거리 39.7 -> 37.9 mm, VRAM 증가 없음.
                   wrist_orient=_w("W_WRISTORI", 1.0))

    # per-axis, per-body-part global weights (n_corr,3): body order = pelvis(0) torso(1) arms(2-7)
    # legs(8-13) hands(14+). Default = uniform full tracking; the low-weight decoupling is opt-in via env.
    w_pxy, w_pz = _w("W_PELVISXY", 1.0), _w("W_PELVISZ", 0.0)   # pelvis-Z tracked by root_height (scaled), not global
    w_trunk, w_arm, w_leg = _w("W_TRUNK", 1.0), _w("W_ARM", 1.0), _w("W_LEG", 1.0)
    gw = onp.zeros((len(a_para), 3), onp.float32)
    gw[0] = [w_pxy, w_pxy, w_pz]                        # pelvis: xy anchor, z free (grounding sets height)
    gw[1] = w_trunk * weights["global_alignment"]       # torso
    gw[2:8] = w_arm * weights["global_alignment"]       # arms — weak (free to reach the object)
    gw[8:14] = w_leg * weights["global_alignment"]      # legs — weak (grounded by floor + local, not absolute)
    gw[14:] = weights["hand_alignment"]                 # hands — strong absolute (object grasp)
    # [ROLLBACK MARKER: fingertip-align] 손끝 10개/손은 각 손 블록의 뒤쪽에 붙어 있습니다.
    # 기본값은 다른 손 키포인트와 동일(균일)입니다. 참고로 env 보상은 손끝을 더 중시합니다
    # (rew_fingertip -6.0 vs rew_hand_kpts -1.5, 둘 다 그룹 평균이라 점당 4:1). 최소제곱에서
    # 중요도는 가중치의 제곱이므로 그 비율을 재현하려면 W_HANDTIP = 2.0 x W_HAND 입니다.
    # 올리면 손끝이 다른 손 키포인트를 밀어낼 수 있으니 스윕으로 확인하고 쓰십시오.
    _n_hand = sum(len(v[0]) for v in _HAND_CHAIN.values())   # 손당 관절 대응 수 15 (손끝 제외)
    _wtip = _w("W_HANDTIP", 0.0) or weights["hand_alignment"]
    for _b in (14, 14 + _n_hand + 5):                   # 왼손 블록 끝 5개, 오른손 블록 끝 5개
        gw[_b + _n_hand: _b + _n_hand + 5] = _wtip

    # per-node LOCAL weight (nb=14 body): arms weak so the arm's relative structure yields to the contact
    # cost (reach the object from the grounded-lower body); pelvis/torso/legs keep full local.
    w_armlocal = _w("W_ARMLOCAL", 1.0)                   # default full; lower (env) frees the arm for grounding
    lw = onp.ones(len(b_para), onp.float32)
    lw[2:8] = w_armlocal                                 # _BODY idx 2-7 = r/l shoulder,elbow,wrist

    root_R_target = _pelvis_target_R(jp)     # (F,3,3) keypoint-derived pelvis orientation target
    # scaled pelvis-height target: human pelvis z × leg_ratio (per frame). Grounds feet (feet_z≈0 scales to
    # ~0) while preserving vertical dynamics (squat/step/bob) — a per-frame track, NOT a fixed height.
    leg_ratio = _w("W_LEGRATIO", 0.86)
    root_z_tgt = (jp[:, 0, 2] * leg_ratio).astype(onp.float32)     # (F,) pelvis keypoint z × ratio
    heightmap = _flat_heightmap(jp)          # flat z=0 floor spanning the clip
    scene_boxes = (_context_boxes(sm, bk[0], jp[:, :23, :]) if weights["scene_collision"] > 0 else None)

    # [OBJECT-COLLISION] keep the hands out of the object they are grasping. The stored object pose is
    # [xyz, wxyz]; the solve uses jaxlie's wxyz-then-xyz order everywhere else, so reorder here.
    obj_boxes_local, obj_pose, obj_caps = None, None, None
    if weights["object_collision"] > 0 and obj_name:
        obj_boxes_local = _object_boxes(obj_name)
        if obj_boxes_local is not None:
            op = sm[bk[0]].astype(onp.float32)                      # (F,7) xyz + wxyz
            obj_pose = onp.concatenate([op[:, 3:7], op[:, :3]], axis=1)
            g2l = onp.asarray(robot_coll._geom_to_link_idx)
            lnames = list(robot_coll.link_names)
            # The palm is EXCLUDED by default. Its capsule has a 65.2 mm radius against a real palm
            # thickness of 34 mm, so keeping that ball out of the knife shoves the whole hand away —
            # and the palm barely penetrates to begin with (measured 6.3% of frames at 1.8 mm, versus
            # 89% at 4.7 mm for the index finger). Constraining the well-fitted finger capsules and
            # leaving the palm out buys the penetration fix at a far smaller tracking cost.
            # W_OBJCOLLSKIP="" constrains every hand link including the palm.
            skip = os.environ.get("W_OBJCOLLSKIP", "palm")
            skip_parts = [s for s in skip.split(",") if s]
            obj_caps = onp.asarray([i for i, g in enumerate(g2l)
                                    if lnames[g].startswith("robot0_")
                                    and not any(s in lnames[g] for s in skip_parts)], onp.int32)
            print(f"[pyroki-retarget] object collision: {len(obj_caps)} hand capsules "
                  f"x {len(obj_boxes_local[0])} boxes x {len(obj_pose)} frames, "
                  f"weight {weights['object_collision']}, tolerance "
                  f"{weights['object_collision_tol'] * 1000:.0f} mm")

    t0 = time.time()
    Ts_root, joints = solve(robot, robot_coll, heightmap, scene_boxes, jnp.array(jp_ext), b_para, b_link, b_mask, a_para,
                            a_link, jnp.array(a_off), jnp.array(gw), jnp.array(lw), jnp.array(l_c), jnp.array(r_c),
                            jnp.array(l_kp), jnp.array(r_kp), left_foot_idx, right_foot_idx,
                            left_knee_idx, right_knee_idx,
                            jnp.array(root_R_target), jnp.array(root_z_tgt), jnp.array(ft_idx), jnp.array(ft_off),
                            jnp.array(ft_margin), jnp.array(ft_pad), jnp.array(ft_mask),
                            jnp.array(rest_w), weights,
                            obj_boxes_local=obj_boxes_local, obj_pose=obj_pose, obj_capsule_idx=obj_caps,
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
        gw2 = gw.copy()
        gw2[0] = 0.0                                             # pelvis pinned → don't track
        gw2[8:14] = 0.0                                          # legs pinned → don't track
        gw2[2:8] = _w("W_STAGE2ARM", 1.0)                       # arms incl. WRIST_yaw_link (anchor hand base)
        gw2[14:] = _w("W_STAGE2HAND", 5.0)                      # hands STRONG (pull arms up to keypoints)
        s2_off = onp.zeros((F, 3), onp.float32)                 # stage-1 root already baked → pin offset→0
        t1 = time.time()
        Ts_root, joints2 = solve(robot, robot_coll, heightmap, scene_boxes, jnp.array(jp_ext), b_para, b_link, b_mask, a_para,
                                 a_link, jnp.array(a_off), jnp.array(gw2), jnp.array(lw), jnp.array(l_c), jnp.array(r_c),
                                 jnp.array(l_kp), jnp.array(r_kp), left_foot_idx, right_foot_idx,
                                 left_knee_idx, right_knee_idx,
                                 jnp.array(root_R_target), jnp.array(root_z_tgt), jnp.array(ft_idx),
                                 jnp.array(ft_off), jnp.array(ft_margin), jnp.array(ft_pad), jnp.array(ft_mask),
                                 jnp.array(rest_w), weights,
                                 obj_boxes_local=obj_boxes_local, obj_pose=obj_pose, obj_capsule_idx=obj_caps,
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
