"""Export the composite G1+Shadow USD → a faithful URDF for PyRoki retargeting.

Runs in env_isaaclab (pxr + the bundled NVIDIA UsdToUrdf; NO omni app needed):
    /home/peunsu/anaconda3/envs/env_isaaclab/bin/python scripts/process_dataset/assets/export_g1_shadow_urdf.py

Pipeline:
  1. For EVERY physics joint, rewrite its anchor frames so the URDF child-in-parent equals the TRUE
     neutral relative body pose:  A(localPos0/Rot0) = child_world · inv(parent_world),  B(localPos1/Rot1)
     = identity.  This (a) makes A/B consistent (UsdToUrdf requires it) and (b) — critically — encodes
     the real body offset.  The composite USD's FixedJoint mounts (wrist_yaw → robot0_wrist) had
     localPos0 = 0 while the hand body sits ~27 cm away (PhysX ignores this for fixed joints, using the
     authored pose), so a naive export/patch mislocated the hands by ~24 cm (they floated above the
     object).  Arm joints already have B=identity + true-relative A, so they are unchanged; the axis
     token stays valid in the child frame for them.
  2. Convert the patched stage with nvidia.srl UsdToUrdf.
  3. Post-process the URDF: strip the defaultPrim name prefix the exporter adds to link names, and
     rename the root link (the merged pelvis) → "pelvis".
Output: data/robots/G1/urdf_pyroki/g1_shadow.urdf (+ meshes/).
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

_ROOT = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5"
_SRC = f"{_ROOT}/data/robots/G1/G1_shadow.usd"
_OUT = f"{_ROOT}/data/robots/G1/urdf_pyroki"
_PREBUNDLE = ("/home/peunsu/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/"
              "isaacsim/exts/isaacsim.asset.exporter.urdf/pip_prebundle")


def _m4(pos, quat):
    T = Gf.Matrix4d()
    T.SetRotate(Gf.Quatd(quat.GetReal(), Gf.Vec3d(*quat.GetImaginary())))
    T.SetTranslateOnly(Gf.Vec3d(*pos))
    return T


def main():
    os.makedirs(_OUT, exist_ok=True)
    stage = Usd.Stage.Open(_SRC)
    xc = UsdGeom.XformCache()

    def body_world(path):
        return xc.GetLocalToWorldTransform(stage.GetPrimAtPath(path))

    # STEP 0: re-author the ENTIRE Shadow-hand subtree's body xforms to their JOINT-ANCHOR poses. The
    # composite USD's hand bodies (robot0_*) are AUTHORED at poses inconsistent with the joint anchors
    # (localPos0/Rot0, localPos1/Rot1) — the sim/PhysX places them via the anchors (correct grasp), but
    # UsdToUrdf builds link/joint origins from the BODY xforms, so the exported hand is ~0.25 m off and
    # floats. FK the hand subtree through the ORIGINAL anchors from each wrist_yaw_link and SET each hand
    # body's world xform = inv(B)·A·parent_world. This makes body xforms == anchors (consistent) AND correct.
    dpn = stage.GetDefaultPrim().GetName()
    hand_joints = []
    for prim in stage.Traverse():
        for JT in (UsdPhysics.RevoluteJoint, UsdPhysics.PrismaticJoint, UsdPhysics.FixedJoint):
            if prim.IsA(JT):
                j = JT(prim)
                b0 = j.GetBody0Rel().GetTargets(); b1 = j.GetBody1Rel().GetTargets()
                if b0 and b1 and b1[0].name.startswith("robot0_"):
                    hand_joints.append(dict(child=b1[0].name, parent=b0[0].name,
                                            A=_m4(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
                                            B=_m4(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())))
                break
    known = {}  # body name -> world Gf.Matrix4d; seed with the arm wrist_yaw links (kept as authored)
    for side in ("right", "left"):
        wy = f"{side}_wrist_yaw_link"
        known[wy] = body_world(f"/{dpn}/{wy}")
    pending = list(hand_joints); nfix = 0
    while pending:
        progressed = False
        for J in list(pending):
            if J["parent"] in known:
                child_world = J["B"].GetInverse() * J["A"] * known[J["parent"]]     # inv(B)·A·parent
                xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"/{dpn}/{J['child']}"))
                xf.ClearXformOpOrder(); xf.AddTransformOp().Set(child_world)
                known[J["child"]] = child_world; pending.remove(J); progressed = True; nfix += 1
        if not progressed:
            raise RuntimeError(f"hand FK stuck; unresolved: {[j['child'] for j in pending]}")
    xc.Clear()
    print(f"[export-urdf] re-authored {nfix} hand body xforms from joint anchors")

    npatch = 0
    for prim in stage.Traverse():
        for JT in (UsdPhysics.RevoluteJoint, UsdPhysics.PrismaticJoint, UsdPhysics.FixedJoint):
            if not prim.IsA(JT):
                continue
            j = JT(prim)
            b0 = j.GetBody0Rel().GetTargets()
            b1 = j.GetBody1Rel().GetTargets()
            if not b0 or not b1:
                break
            p_w = body_world(b0[0])
            c_w = body_world(b1[0])
            A = c_w * p_w.GetInverse()          # child-in-parent (row-vec): child_world = A * parent_world
            r = A.ExtractRotationQuat(); t = A.ExtractTranslation()
            j.GetLocalPos0Attr().Set(Gf.Vec3f(*[float(x) for x in t]))
            j.GetLocalRot0Attr().Set(Gf.Quatf(float(r.GetReal()), Gf.Vec3f(*[float(x) for x in r.GetImaginary()])))
            j.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            j.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            npatch += 1
            break
    print(f"[export-urdf] rewrote {npatch} joint anchors to (A=true-relative, B=identity)")

    patched = f"{_OUT}/G1_shadow_patched.usda"
    stage.Export(patched)

    sys.path.insert(0, _PREBUNDLE)
    import nvidia  # noqa: E402
    nvidia.__path__.append(os.path.join(_PREBUNDLE, "nvidia"))
    from nvidia.srl.from_usd.to_urdf import UsdToUrdf  # noqa: E402

    st2 = Usd.Stage.Open(patched)
    u2u = UsdToUrdf(st2, root=st2.GetDefaultPrim().GetPath().pathString)
    urdf_path = f"{_OUT}/g1_shadow.urdf"
    u2u.save_to_file(urdf_output_path=urdf_path, mesh_dir=f"{_OUT}/meshes",
                     mesh_path_prefix="file://", use_uri_file_prefix=True,
                     visualize_collision_meshes=False)
    print(f"[export-urdf] converted → {urdf_path}")

    # post-process link names: strip defaultPrim prefix, rename merged-pelvis root → "pelvis"
    dp = st2.GetDefaultPrim().GetName()          # e.g. g1_29dof_with_hand_rev_1_0
    pfx = dp + "_"
    tree = ET.parse(urdf_path); root = tree.getroot()

    def fix(name):
        if name == dp:
            return "pelvis"
        return name[len(pfx):] if name.startswith(pfx) else name

    for link in root.findall("link"):
        link.set("name", fix(link.get("name", "")))
    for joint in root.findall("joint"):
        for tag in ("parent", "child"):
            e = joint.find(tag)
            if e is not None:
                e.set("link", fix(e.get("link", "")))

    tree.write(urdf_path)
    print(f"[export-urdf] normalized link names (prefix '{pfx}' stripped, root→pelvis). DONE.")


if __name__ == "__main__":
    main()
