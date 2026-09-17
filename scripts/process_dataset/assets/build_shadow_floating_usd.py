"""Extract the bimanual Shadow hands from G1_shadow.usd into two FREE-FLOATING hand USDs.

Why
---
The floating-hand pretrain (stage 1 of the two-stage loco-manip plan) learns dexterous
manipulation with a 6-DOF free wrist instead of the G1 arm, then hands its wrist+finger
trajectory to the full-body stage as a tracking target. That needs the hand alone, rooted at
`robot0_{l,r}_wrist`, with `fix_root_link=False`.

Design decisions (and why NOT the alternatives)
-----------------------------------------------
* Built from OUR `G1_shadow.usd`, not from workspaceTJ's `wrist_free_hand_mount.usd`. The link
  NAMES must stay byte-identical to the G1 build because three separate tables key off them:
  `LINK_CONTACT_NAMES` (32 contact links), the `link_names` array inside every clip's
  `hand_contact.npz`, and `HAND_CHAIN`'s Shadow link names. Renaming would force all three to be
  regenerated.
* `pxr` only — no `SimulationApp`. `build_g1_shadow_usd.py` boots Isaac Sim, but
  `make_robot_usd_instanceable.py` shows the layer-level `Sdf` path works standalone, and that
  keeps this script off the GPU (it can run while a training job holds the device).
* Layer-level `Sdf.CopySpec` rather than stage-level copy: it preserves the `physxTendon:*`
  attributes that live ON the joint prims (`robot0_*_FFJ1` carries
  `physxTendon:robot0_T_FFJ1c:{damping,gearing,lowerLimit}`, `robot0_*_FFJ0` carries the paired
  `gearing`), which is how the J0<->J1 loopback coupling is authored. Rebuilding the joints by
  hand would silently drop them.

Source layout (verified on G1_shadow.usd, 2026-09-06)
-----------------------------------------------------
    /g1_29dof_with_hand_rev_1_0            Xform, defaultPrim  (78 Xform links + 2 Scopes)
      <link>                               Xform, flat — collisions/ + visuals/ inline meshes
      joints/<joint>                       Scope of 77 joints; body0/body1 are ABSOLUTE Sdf paths
      Looks/DefaultMaterial{,_0}           Scope
    ArticulationRootAPI lives on /pelvis (NOT the defaultPrim)
    robot0_{s}_mount  (fixed)  {s}_wrist_yaw_link -> robot0_{s}_wrist     <- arm mount
    robot0_{s}_WRJ0   (fixed)  robot0_{s}_wrist   -> robot0_{s}_palm      <- cut here

So: keep the 23 links reachable from `robot0_{s}_palm`, keep every joint whose body0 AND body1
are both in that set (this drops `mount` AND `WRJ0` automatically), and move ArticulationRootAPI
onto the new root link.

Why the cut moved from `wrist` to `palm` (2026-09-07, user's call)
------------------------------------------------------------------
`WRJ0` is a **PhysicsFixedJoint**, so the `wrist` link contributed ZERO degrees of freedom — it was
a 0.300 kg rigid body welded to the palm, inherited from the G1 arm mount. Keeping it cost:

  * mass 0.911 -> 0.611 kg per hand (gravity 8.94 -> 6.00 N, so the wrist impedance gets more
    authority for the same K_pos)
  * one extra rigid body and one extra joint per hand in the solver
  * one more body in every contact/gather path

workspaceTJ's `wrist_free_hand_mount.usd` does keep a base link before the palm, but it is a
0.100 kg `robot0_hand_mount` stub, not a real wrist. Rooting at the palm is closer to TJ than
keeping the G1 wrist was.

CONSEQUENCE: the articulation root is now `robot0_{s}_palm`, so the "wrist pose" that the env
writes at reset and that `export_wrist_ref.py` produces is the PALM pose. Both were updated;
the 16 clips' `wrist_ref.npz` must be regenerated (the frame changed).

Usage
-----
    <env_isaaclab python> scripts/process_dataset/assets/build_shadow_floating_usd.py
    # writes data/robots/G1/shadow_float_l.usd and shadow_float_r.usd, then self-verifies
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdPhysics

_ROOT = Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data" / "robots" / "G1"
_SRC = _ROOT / "G1_shadow.usd"

# ── [wrist6] 6-DoF 손목 관절 체인 (DexMachina, CVPR'25 방식) ──────────────────────────────
# `--wrist6` 로 켜면 palm 앞에 관절 6개 + 더미 링크 7개를 직렬로 붙이고 베이스를 고정합니다.
# 즉 "떠 있는 손"을 자유 베이스 + 외력이 아니라 **관절**로 만듭니다. 그러면 관절 제어가
# 자동으로 제공하는 벽(effort_limit_sim / velocity_limit_sim / 관절 한계)이 생깁니다 —
# 전신 팔에서 손목 발산이 한 번도 없었던 이유가 그것입니다.
#
# 체인:  anchor -> tx -> ty -> tz -> roll(Z) -> pitch(X) -> yaw(Y) -> (fixed) -> palm
#
# 모든 조인트의 local 프레임을 항등으로 두므로 합성 변환이
#     T(q) = Trans(tx,ty,tz) . Rz(roll) . Rx(pitch) . Ry(yaw)
# 가 되고, 병진 3축이 회전 앞에 있으므로 **palm 위치 = (tx,ty,tz)** 그대로입니다.
# 자세는 intrinsic Z-X-Y 오일러라 pose <-> 6-DoF 변환이 정확히 가역입니다 (round-trip 검증).
# 원본 WRJ0 이 localRot 항등이었으므로(측정) palm 프레임 방향을 그대로 물려받습니다.
#
# 회전 축 순서. DexMachina 는 (Z, X, Y) = ZXY 를 쓰지만 **ZXY 는 우리 데이터에서 12개
# 오일러 규약 중 가장 나쁩니다**. 16클립 양손 5638프레임 실측 (det(J) = 퇴화 방향의 제어
# 권한, 낮을수록 특이점에 가까움):
#     ZXY  det최소 0.0267  det<0.2 노출 8.1%   <- 12개 중 최악
#     ZYX  0.0056 / 1.7%      XYZ 0.0200 / 2.7%      XZY 0.0247 / 1.5%
#     YXZ  0.0267 / 4.5%      YXY·YZY 0.0157 / 1.3%  XYX·XZX 0.0134 / 4.5%
#     ZXZ·ZYZ 0.1032 / 0.8%   요구속도는 최저(p99 4.99)지만 특이점이 중간각 0도에 있어
#                             손의 중립 자세에 가깝습니다 — 새 클립에서 걸릴 위험이 큽니다
#     YZX  det최소 0.1615  det<0.2 노출 0.1%   <- 채택
# s101_seg12_knife 오른손(발산했던 케이스)에서 ZXY 는 물리 각속도 7.51 rad/s 를 표현하려고
# 관절에 46.59 rad/s 를 요구합니다(6.21배 증폭, det 0.027, 노출 65.8%). 손바닥을 3도 돌리려고
# roll·yaw 가 반대로 105도씩 휘두르기 때문입니다(50fps 프레임 70~76 실측). YZX 로 바꾸면
# 7.63 rad/s / det 0.946 / 노출 0.0% 로 증폭이 1.02배가 됩니다. 언랩은 이 증폭을 못 고칩니다 —
# 랩이 아니라 차트가 실제로 요구하는 관절 운동이기 때문입니다.
# 특이점 근접도는 **항상 det(J) 로 판정하고 각도로 하지 마십시오**: |중간각| 88.5도 vs 84.1도는
# 5% 차이처럼 보이지만 cos 으로는 0.026 vs 0.105 로 4배입니다.
# 재감사: scripts/process_dataset/diagnostics/audit_euler_convention.py
_W6_ROT_SEQ = "YZX"
# 관절 이름은 서수(rot1/rot2/rot3)로 둡니다. 축 이름을 쓰면 ZXZ 처럼 축이 반복되는 규약에서
# 충돌하고, roll/pitch/yaw 는 순서를 바꾸면 뜻이 어긋납니다. 실제 축은 USD 의 physics:axis 에
# 있고 audit 스크립트가 거기서 읽습니다.


def _make_w6(rot_seq: str) -> list[tuple[str, str, str]]:
    """(이름, 종류, 축) 목록. 병진 3개는 항상 X,Y,Z 순, 회전 3개는 rot_seq 순."""
    if len(rot_seq) != 3 or any(c not in "XYZ" for c in rot_seq):
        raise ValueError(f"rot_seq 는 XYZ 로 이루어진 3글자여야 합니다: {rot_seq!r}")
    return ([("tx", "P", "X"), ("ty", "P", "Y"), ("tz", "P", "Z")]
            + [(f"rot{i + 1}", "R", ax) for i, ax in enumerate(rot_seq)])


_W6 = _make_w6(_W6_ROT_SEQ)
# 관절 한계. 병진은 넉넉하게(리타게팅 palm 범위 실측: x[-1.03,1.47] y[-1.42,1.61] z[0.76,1.41] m),
# 회전은 전 범위. 타이트한 구속은 액션의 잔차 스케일과 term_wrist_pos_err 가 담당하고, 여기 한계는
# **발산 안전벽** 입니다 (DexMachina 도 기본 ±2.0 을 쓰고 클립별로 좁힙니다).
_W6_TRANS_RANGE = (-2.0, 2.0)      # m
# 회전 한계를 ±180 -> ±720 도로 넓혔습니다. 레퍼런스 6-DoF 를 **언랩**해서 쓰기 때문입니다:
# atan2 분기 절단(-179 <-> +179) 때문에 언랩하지 않으면 프레임 간 점프가 최대 359.6 도로
# 튀는데(실측 16클립 5638프레임), 언랩하면 중앙 1.87 도 / p99 18.7 도로 정상화됩니다.
# 대신 언랩 값이 누적되어 최대 |각| 418 도까지 가므로(실측) ±180 한계로는 잘립니다.
# 남는 짐벌 위험: |pitch| 최대 88.5 도로 특이점(90 도)에서 1.5 도까지 접근하고, 그 근방
# 14 프레임에서 언랩 후에도 45 도 이상 점프가 남습니다 — 학습에서 이 구간을 지켜봐야 합니다.
_W6_ROT_RANGE = (-720.0, 720.0)    # deg (USD revolute 한계는 도 단위)
# USD 의 maxForce 상한. cfg 의 effort_limit_sim 이 런타임에 덮어쓰지만, 여기 값이 0 이면
# PhysX 가 구동력을 0 으로 잡을 수 있어 넉넉히 둡니다 (병진 N, 회전 N*m).
_W6_MAX_FORCE = {"P": 500.0, "R": 200.0}
_W6_DUMMY_MASS = 0.01              # kg — DexMachina 와 동일
_W6_DUMMY_INERTIA = 0.01           # kg*m^2 대각

# Expected inventory per hand, from the URDF (g1_shadow_nomimic.urdf):
#   24 links, 23 joints = 18 actuated + 4 tendon J0 (FF/MF/RF/LF) + 1 fixed WRJ0
# palm 부터 잘라내므로 wrist 링크 1개와 WRJ0 고정관절 1개가 빠집니다.
_N_LINKS = 23
_N_JOINTS = 22
_N_ACTUATED = 18


def _reachable_links(stage: Usd.Stage, root_path: str, side: str,
                     exclude: set[str] | None = None) -> set[str]:
    """Link prim NAMES reachable from `root_path` through the /joints graph.

    The USD is flat (every link is a direct child of the defaultPrim), so reachability has to be
    walked through the joint relationships rather than the prim hierarchy.

    `exclude` is required because the walk is UNDIRECTED — the joint graph carries no parent/child
    direction, so a BFS from `robot0_{s}_palm` happily crosses the fixed `WRJ0` backwards and picks
    the wrist link right back up (measured: 24 links instead of 23). Naming the cut explicitly is
    clearer than trying to infer joint direction from body0/body1, which is not reliably
    parent->child in this asset.
    """
    exclude = exclude or set()
    dp = stage.GetDefaultPrim().GetPath()
    joints = stage.GetPrimAtPath(f"{dp}/joints")
    edges: list[tuple[str, str]] = []
    for j in joints.GetChildren():
        b0 = j.GetRelationship("physics:body0")
        b1 = j.GetRelationship("physics:body1")
        if not b0 or not b1:
            continue
        t0, t1 = b0.GetTargets(), b1.GetTargets()
        if not t0 or not t1:
            continue
        edges.append((t0[0].name, t1[0].name))
    # BFS from the root, but only through prims of THIS hand — that stops the walk from crossing
    # `mount` back into the arm (mount's other end is {side}_wrist_yaw_link, not a robot0_ prim).
    pfx = f"robot0_{side}_"
    seen = {root_path}
    frontier = [root_path]
    while frontier:
        cur = frontier.pop()
        for a, b in edges:
            for x, y in ((a, b), (b, a)):
                if x == cur and y.startswith(pfx) and y not in seen and y not in exclude:
                    seen.add(y)
                    frontier.append(y)
    return seen


def _add_wrist6(stage: Usd.Stage, new_root: str, side: str, palm_link: str) -> tuple[int, int]:
    """[wrist6] palm 앞에 6-DoF 관절 체인을 붙이고 anchor 를 새 articulation root 로 만든다.

    반환: (추가 링크 수, 추가 관절 수) = (7, 7)

    모든 조인트의 localPos/localRot 을 항등으로 둔다. 그래야 합성 변환이
    Trans(tx,ty,tz).Rz(roll).Rx(pitch).Ry(yaw) 가 되어 palm 위치가 (tx,ty,tz) 그대로이고
    pose <-> 6-DoF 변환이 정확히 가역이다. 더미 링크는 충돌 형상을 주지 않는다(질량만).
    """
    pfx = f"robot0_{side}"
    ident_q = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    zero_v = Gf.Vec3f(0.0, 0.0, 0.0)

    def _dummy(name: str) -> str:
        p = f"{new_root}/{name}"
        pr = stage.DefinePrim(p, "Xform")
        UsdPhysics.RigidBodyAPI.Apply(pr)
        m = UsdPhysics.MassAPI.Apply(pr)
        m.CreateMassAttr(_W6_DUMMY_MASS)
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*([_W6_DUMMY_INERTIA] * 3)))
        return p

    anchor = _dummy(f"{pfx}_anchor")
    prev = anchor
    n_link, n_joint = 1, 0
    for name, kind, axis in _W6:
        link = _dummy(f"{pfx}_link_{name}")
        n_link += 1
        jp = f"{new_root}/joints/{pfx}_wrist_{name}"
        if kind == "P":
            j = UsdPhysics.PrismaticJoint.Define(stage, jp)
            j.CreateLowerLimitAttr(_W6_TRANS_RANGE[0])
            j.CreateUpperLimitAttr(_W6_TRANS_RANGE[1])
        else:
            j = UsdPhysics.RevoluteJoint.Define(stage, jp)
            j.CreateLowerLimitAttr(_W6_ROT_RANGE[0])
            j.CreateUpperLimitAttr(_W6_ROT_RANGE[1])
        j.CreateAxisAttr(axis)
        # DriveAPI 가 없으면 PhysX 가 그 DOF 를 구동하지 않습니다 — Isaac Lab 이 런타임에
        # stiffness/damping/max_force 를 써도 무시되고 관절이 얼어붙습니다 (실측: 힘 지령이
        # 200 N 한계까지 올라가는데 tz 가 초기값에서 1 mm 도 안 움직임). 기존 손가락 관절은
        # 원본 USD 에서 PhysicsDriveAPI:angular 를 물려받으므로 이 문제가 없었습니다.
        # 값은 여기서 0 으로 두고 cfg 의 ImplicitActuatorCfg 가 실제 게인을 씁니다 —
        # 손가락도 같은 방식(USD 에 1.0/0.1, cfg 가 덮어씀)입니다.
        drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(),
                                        "linear" if kind == "P" else "angular")
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(0.0)
        drv.CreateDampingAttr(0.0)
        drv.CreateTargetPositionAttr(0.0)
        drv.CreateMaxForceAttr(_W6_MAX_FORCE[kind])
        j.CreateBody0Rel().SetTargets([Sdf.Path(prev)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(link)])
        j.CreateLocalPos0Attr(zero_v); j.CreateLocalRot0Attr(ident_q)
        j.CreateLocalPos1Attr(zero_v); j.CreateLocalRot1Attr(ident_q)
        prev = link
        n_joint += 1
    # 체인 끝 -> palm (고정). 항등 프레임이라 palm 이 체인 끝에 정확히 붙는다.
    jf = UsdPhysics.FixedJoint.Define(stage, f"{new_root}/joints/{pfx}_wrist_to_palm")
    jf.CreateBody0Rel().SetTargets([Sdf.Path(prev)])
    jf.CreateBody1Rel().SetTargets([Sdf.Path(f"{new_root}/{palm_link}")])
    jf.CreateLocalPos0Attr(zero_v); jf.CreateLocalRot0Attr(ident_q)
    jf.CreateLocalPos1Attr(zero_v); jf.CreateLocalRot1Attr(ident_q)
    n_joint += 1
    return n_link, n_joint


def build(side: str, out_path: Path, src: Path = _SRC, wrist6: bool = False) -> tuple[int, int]:
    """Write one floating-hand USD. Returns (n_links, n_joints) actually copied."""
    src_stage = Usd.Stage.Open(str(src))
    src_layer = src_stage.GetRootLayer()
    dp = src_stage.GetDefaultPrim().GetPath()
    root_link = f"robot0_{side}_palm"

    # 손목 링크를 명시적으로 배제합니다 — WRJ0 가 고정관절이라 무방향 BFS 가 되돌아갑니다.
    links = _reachable_links(src_stage, root_link, side,
                             exclude={f"robot0_{side}_wrist"})
    if len(links) != _N_LINKS:
        raise RuntimeError(
            f"[{side}] reachable links = {len(links)}, expected {_N_LINKS}. "
            f"Got: {sorted(links)}")

    # Joints to keep: both endpoints inside `links`. This drops robot0_{s}_mount (body0 is an ARM
    # link) and robot0_{s}_WRJ0 (body0 is the now-excluded wrist) without naming either explicitly.
    keep_joints: list[str] = []
    for j in src_stage.GetPrimAtPath(f"{dp}/joints").GetChildren():
        b0 = j.GetRelationship("physics:body0")
        b1 = j.GetRelationship("physics:body1")
        if not b0 or not b1:
            continue
        t0, t1 = b0.GetTargets(), b1.GetTargets()
        if t0 and t1 and t0[0].name in links and t1[0].name in links:
            keep_joints.append(j.GetName())
    if len(keep_joints) != _N_JOINTS:
        raise RuntimeError(
            f"[{side}] kept joints = {len(keep_joints)}, expected {_N_JOINTS}. "
            f"Got: {sorted(keep_joints)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():                    # explicit: never silently merge into a stale layer
        out_path.unlink()
    dst_stage = Usd.Stage.CreateNew(str(out_path))
    dst_layer = dst_stage.GetRootLayer()
    new_root = f"/shadow_hand_{side}"
    Sdf.CreatePrimInLayer(dst_layer, Sdf.Path(new_root))
    dst_stage.DefinePrim(new_root, "Xform")

    # ---- copy link prims (with their inline collisions/ + visuals/ meshes) ----
    for name in sorted(links):
        Sdf.CopySpec(src_layer, Sdf.Path(f"{dp}/{name}"), dst_layer, Sdf.Path(f"{new_root}/{name}"))

    # ---- copy the Looks scope so material bindings still resolve ----
    looks = src_stage.GetPrimAtPath(f"{dp}/Looks")
    if looks:
        Sdf.CopySpec(src_layer, Sdf.Path(f"{dp}/Looks"), dst_layer, Sdf.Path(f"{new_root}/Looks"))

    # ---- copy joints, then retarget their absolute body/material paths onto the new root ----
    Sdf.CreatePrimInLayer(dst_layer, Sdf.Path(f"{new_root}/joints"))
    dst_stage.DefinePrim(f"{new_root}/joints", "Scope")
    for name in keep_joints:
        Sdf.CopySpec(src_layer, Sdf.Path(f"{dp}/joints/{name}"),
                     dst_layer, Sdf.Path(f"{new_root}/joints/{name}"))

    # Rewrite every relationship / path-valued attribute that still points at the OLD root, and
    # DROP targets whose prim no longer exists in this slice.
    #
    # The drop matters. `physics:filteredPairs` on the finger links names every body whose
    # self-collision should be ignored, and in the G1 asset that list includes the wrist. Cutting
    # the wrist left 17 dangling targets per hand (caught by verify()). A dangling filteredPairs
    # entry is not cosmetic now that `enabled_self_collisions=True`: PhysX resolves that list at
    # parse time, so a broken entry means the pair it was supposed to exclude is no longer
    # excluded — neighbouring links would start colliding with each other.
    old_pfx, new_pfx = str(dp), new_root
    kept_paths = {f"{new_root}/{n}" for n in links} | {f"{new_root}/joints/{n}" for n in keep_joints}
    n_dropped = 0
    for prim in dst_stage.Traverse():
        for rel in prim.GetRelationships():
            tgts = rel.GetTargets()
            if not tgts:
                continue
            fixed = []
            for t in tgts:
                p = str(t)
                if p.startswith(old_pfx):
                    p = p.replace(old_pfx, new_pfx, 1)
                # 링크/관절 경로인데 이 슬라이스에 없으면 버립니다. Looks/ 등 다른 경로는 유지.
                base = p.rsplit("/", 1)[0] if "/collisions" in p or "/visuals" in p else p
                is_body_or_joint = p.startswith(f"{new_root}/robot0_") or p.startswith(f"{new_root}/joints/")
                if is_body_or_joint and base.split("/collisions")[0].split("/visuals")[0] not in kept_paths:
                    n_dropped += 1
                    continue
                fixed.append(Sdf.Path(p))
            if fixed != list(tgts):
                rel.SetTargets(fixed)
    if n_dropped:
        print(f"    [{side}] 사라진 프림을 가리키는 관계 대상 {n_dropped}건 제거 "
              f"(대부분 잘려나간 wrist 를 참조하던 physics:filteredPairs)")

    # ---- articulation root ----
    n_extra_l = n_extra_j = 0
    if wrist6:
        # [wrist6] palm 앞에 6-DoF 체인을 붙이고 anchor 가 root 가 된다 (베이스는 cfg 의
        # fix_root_link=True 로 고정). palm 은 더 이상 root 가 아니다.
        n_extra_l, n_extra_j = _add_wrist6(dst_stage, new_root, side, root_link)
        root_name = f"robot0_{side}_anchor"
    else:
        root_name = root_link
    root_prim = dst_stage.GetPrimAtPath(f"{new_root}/{root_name}")
    if not root_prim:
        raise RuntimeError(f"[{side}] root link prim missing after copy: {root_name}")
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)

    dst_stage.SetDefaultPrim(dst_stage.GetPrimAtPath(new_root))
    dst_stage.GetRootLayer().Save()
    return len(links) + n_extra_l, len(keep_joints) + n_extra_j


def verify(path: Path, side: str, wrist6: bool = False) -> None:
    """Re-open the written USD and assert the inventory a floating hand must have."""
    st = Usd.Stage.Open(str(path))
    dp = st.GetDefaultPrim()
    if not dp:
        raise RuntimeError(f"{path.name}: no defaultPrim")
    links = [p for p in dp.GetChildren() if p.GetName().startswith(f"robot0_{side}_")]
    joints = list(st.GetPrimAtPath(f"{dp.GetPath()}/joints").GetChildren())
    revolute = [j for j in joints if "Revolute" in str(j.GetTypeName())]
    prismatic = [j for j in joints if "Prismatic" in str(j.GetTypeName())]
    fixed = [j for j in joints if "Fixed" in str(j.GetTypeName())]
    arts = [p.GetPath() for p in st.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    tendon_j = [j.GetName() for j in joints
                if any("physxTendon" in a.GetName() for a in j.GetAttributes())]
    dangling = []
    for prim in st.Traverse():
        for rel in prim.GetRelationships():
            for t in rel.GetTargets():
                if not st.GetPrimAtPath(t.GetPrimPath()):
                    dangling.append(f"{prim.GetName()}.{rel.GetName()} -> {t}")

    print(f"  [{path.name}]")
    print(f"    defaultPrim      {dp.GetPath()}")
    # [wrist6] 기대치: 링크 +7 (anchor + 더미 6), 관절 +7 (prismatic 3 + revolute 3 + fixed 1)
    exp_l = _N_LINKS + (7 if wrist6 else 0)
    exp_j = _N_JOINTS + (7 if wrist6 else 0)
    exp_root = f"robot0_{side}_anchor" if wrist6 else f"robot0_{side}_palm"
    print(f"    링크             {len(links)} (기대 {exp_l})")
    print(f"    관절             {len(joints)} = revolute {len(revolute)} + "
          f"prismatic {len(prismatic)} + fixed {len(fixed)}  (기대 {exp_j})")
    print(f"    ArticulationRoot {arts}")
    print(f"    텐던 어트리뷰트   {len(tendon_j)}개 관절: {sorted(tendon_j)}")
    print(f"    끊긴 참조         {len(dangling)}건" + (f" — {dangling[:3]}" if dangling else ""))

    errs = []
    if len(links) != exp_l:
        errs.append(f"링크 {len(links)} != {exp_l}")
    if len(joints) != exp_j:
        errs.append(f"관절 {len(joints)} != {exp_j}")
    if len(arts) != 1 or arts[0].name != exp_root:
        errs.append(f"ArticulationRoot 위치 이상: {arts} (기대 {exp_root})")
    if wrist6:
        # 손목 체인이 정확히 prismatic 3 + revolute 3 + fixed 1 인지. 손가락은 revolute 22.
        if len(prismatic) != 3:
            errs.append(f"prismatic {len(prismatic)} != 3")
        if len(revolute) != 22 + 3:
            errs.append(f"revolute {len(revolute)} != 25 (손가락 22 + 손목 3)")
        if len(fixed) != 1:
            errs.append(f"fixed {len(fixed)} != 1 (wrist_to_palm)")
        _w6 = [f"robot0_{side}_wrist_{n}" for n, _, _ in _W6]
        _missing = [n for n in _w6 if not st.GetPrimAtPath(f"{dp.GetPath()}/joints/{n}")]
        if _missing:
            errs.append(f"손목 관절 누락: {_missing}")
    if len(tendon_j) != 8:      # 4 J1 (damping/gearing/limits) + 4 J0 (paired gearing)
        errs.append(f"텐던 관절 {len(tendon_j)} != 8")
    if dangling:
        errs.append(f"끊긴 참조 {len(dangling)}건")
    if errs:
        raise RuntimeError(f"{path.name} 검증 실패: {'; '.join(errs)}")
    print("    → 검증 통과")


def main() -> int:
    global _W6, _W6_DUMMY_MASS, _W6_DUMMY_INERTIA
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(_SRC))
    ap.add_argument("--out_dir", default=str(_ROOT))
    ap.add_argument("--side", choices=["l", "r", "both"], default="both")
    ap.add_argument("--wrist6", action="store_true",
                    help="[wrist6] palm 앞에 6-DoF 관절 체인을 붙여 shadow_float6_{s}.usd 로 "
                         "씁니다 (DexMachina 방식). 기존 shadow_float_{s}.usd 는 그대로 둡니다.")
    ap.add_argument("--dummy_mass", type=float, default=_W6_DUMMY_MASS,
                    help="[wrist6] 더미 링크 질량(kg). 고정 루트에서 손가락 반력이 이 사슬을 통과해 "
                         "접지까지 가므로, 너무 작으면 질량행렬 조건수가 나빠져 TGS 가 수렴하지 못한다.")
    ap.add_argument("--dummy_inertia", type=float, default=_W6_DUMMY_INERTIA,
                    help="[wrist6] 더미 링크 대각 관성(kg*m^2). 같은 이유.")
    ap.add_argument("--rot_seq", default=_W6_ROT_SEQ,
                    help="[wrist6] 회전 3축의 순서 (내재 회전). 기본 %(default)s. "
                         "규약별 특이점 노출은 diagnostics/audit_euler_convention.py 로 재보십시오.")
    a = ap.parse_args()

    if a.rot_seq != _W6_ROT_SEQ:
        _W6 = _make_w6(a.rot_seq)
    _W6_DUMMY_MASS, _W6_DUMMY_INERTIA = a.dummy_mass, a.dummy_inertia
    if a.wrist6:
        print(f"[build-shadow-float] 더미 질량 {a.dummy_mass} kg / 관성 {a.dummy_inertia}")
        print(f"[build-shadow-float] 회전 축 순서 = {a.rot_seq} "
              f"({' -> '.join(f'{n}({ax})' for n, k, ax in _W6 if k == 'R')})")

    sides = ["l", "r"] if a.side == "both" else [a.side]
    for s in sides:
        _stem = "shadow_float6" if a.wrist6 else "shadow_float"
        out = Path(a.out_dir) / f"{_stem}_{s}.usd"
        nl, nj = build(s, out, Path(a.src), wrist6=a.wrist6)
        print(f"[build-shadow-float] {s}: 링크 {nl} / 관절 {nj} → {out}")
        verify(out, s, wrist6=a.wrist6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
