"""Build a SIM-CONSISTENT Shadow-hand pinocchio model directly from the composite USD joints
(per side), so the hand-IK model matches the sim's kinematics EXACTLY (unlike the dex-retargeting
URDF, whose palm/joint frame conventions differ → the hand IK failed). This is the hand analog of
g1_from_usd.urdf for the body.

USD joint J (parent body P, child body C): joint frame is (localPos0,localRot0)=A in P and
(localPos1,localRot1)=B in C, coinciding at q=0. Child body pose:
    C(q) = P · A · Rot(axis,q) · B⁻¹
pinocchio mapping (palm = fixed root at identity; joint frame := USD joint frame):
    jointPlacement(J) = B_parent⁻¹ · A_J     (relative to the parent joint frame)
    body-frame(C)     = B_C⁻¹                (fixed frame on joint J → equals the USD C body frame)
Joint names = the USD names (robot0_{s}_FFJ0…) so the env-name mapping is IDENTITY.

Provides build_shadow_from_usd(usd_path, side) → (model, data). Run directly to build + VERIFY the
palm-relative body FK against the composite USD XformCache.
"""

import numpy as np
import pinocchio as pin
from pxr import Usd, UsdGeom, UsdPhysics, Gf

_AXIS = {"X": pin.JointModelRX, "Y": pin.JointModelRY, "Z": pin.JointModelRZ}
_FT_OFFSET = {"th": [-0.0085, 0.0, 0.02], "ff": [0.0, -0.006, 0.0175], "mf": [0.0, -0.006, 0.0175],
              "rf": [0.0, -0.006, 0.0175], "lf": [0.0, -0.006, 0.0175]}
_PAD_TO_DISTAL = {"ff_pad": "ffdistal", "mf_pad": "mfdistal", "rf_pad": "rfdistal",
                  "lf_pad": "lfdistal", "th_pad": "thdistal"}


def _se3(pos, quat):
    w, x, y, z = quat.GetReal(), *quat.GetImaginary()
    return pin.SE3(pin.Quaternion(w, x, y, z).toRotationMatrix(), np.array([pos[0], pos[1], pos[2]], float))


def build_shadow_from_usd(usd_path, side, floating=False):
    """floating=True → the palm is a FREE-FLYER root (its 6-DoF pose is solved by the hand IK from
    the finger keypoints, so the palm orientation is determined by the data, not a guessed frame)."""
    s = Usd.Stage.Open(usd_path)
    pfx = f"robot0_{side}_"
    joints = []
    for prim in s.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName().startswith(pfx):
            j = UsdPhysics.RevoluteJoint(prim)
            b0 = [t.name for t in j.GetBody0Rel().GetTargets()]
            b1 = [t.name for t in j.GetBody1Rel().GetTargets()]
            joints.append(dict(
                name=prim.GetName(), parent=b0[0], child=b1[0], axis=j.GetAxisAttr().Get(),
                A=_se3(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
                B=_se3(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()),
                lo=np.deg2rad(j.GetLowerLimitAttr().Get() if j.GetLowerLimitAttr().Get() is not None else -180.0),
                hi=np.deg2rad(j.GetUpperLimitAttr().Get() if j.GetUpperLimitAttr().Get() is not None else 180.0)))
    root = f"{pfx}palm"
    model = pin.Model(); model.name = f"shadow_{side}"
    inertia = pin.Inertia.Identity()
    if floating:
        palm_jid = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), f"{pfx}palm_ff")
        model.appendBodyToJoint(palm_jid, inertia, pin.SE3.Identity())
        body_jid = {root: palm_jid}
        model.addFrame(pin.Frame("palm", palm_jid, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME))
    else:
        body_jid = {root: 0}
        model.addFrame(pin.Frame("palm", 0, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME))
    body_Binv = {root: pin.SE3.Identity()}
    pending = list(joints)
    while pending:
        progressed = False
        for J in list(pending):
            if J["parent"] not in body_jid:
                continue
            pj = body_jid[J["parent"]]
            placement = body_Binv[J["parent"]] * J["A"]        # B_parent⁻¹ · A
            jid = model.addJoint(pj, _AXIS[J["axis"]](), placement, J["name"])
            model.appendBodyToJoint(jid, inertia, pin.SE3.Identity())
            model.lowerPositionLimit[model.joints[jid].idx_q] = J["lo"]
            model.upperPositionLimit[model.joints[jid].idx_q] = J["hi"]
            Binv = J["B"].inverse()
            body_jid[J["child"]] = jid
            body_Binv[J["child"]] = Binv
            model.addFrame(pin.Frame(J["child"][len(pfx):], jid, 0, Binv, pin.FrameType.OP_FRAME))
            pending.remove(J); progressed = True
        if not progressed:
            raise RuntimeError(f"disconnected Shadow tree, remaining: {[j['name'] for j in pending]}")
    # fingertip-pad OP_FRAMEs on the distal body frames (Y-negated for the LEFT hand mirror)
    for pad, distal in _PAD_TO_DISTAL.items():
        off = list(_FT_OFFSET[pad[:2]])
        if side == "l":
            off[1] = -off[1]
        fid = model.getFrameId(distal)
        fr = model.frames[fid]
        model.addFrame(pin.Frame(pad, fr.parentJoint, fid, fr.placement * pin.SE3(np.eye(3), np.array(off)),
                                 pin.FrameType.OP_FRAME))
    return model, model.createData()


def _verify(usd_path, side):
    """Compare the model's palm-relative body FK (neutral) to the composite USD XformCache."""
    model, data = build_shadow_from_usd(usd_path, side)
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    s = Usd.Stage.Open(usd_path)
    root = "/g1_29dof_with_hand_rev_1_0"
    xc = UsdGeom.XformCache()
    palm_w = xc.GetLocalToWorldTransform(s.GetPrimAtPath(f"{root}/robot0_{side}_palm"))
    palm_inv = palm_w.GetInverse()
    errs = []
    for fr in model.frames:
        nm = fr.name
        if nm in ("universe", "palm") or nm.endswith("_pad"):
            continue
        usd_prim = s.GetPrimAtPath(f"{root}/robot0_{side}_{nm}")
        if not usd_prim:
            continue
        usd_local = palm_inv * xc.GetLocalToWorldTransform(usd_prim)     # USD body in palm frame
        usd_p = np.array([usd_local.ExtractTranslation()[i] for i in range(3)])
        model_p = data.oMf[model.getFrameId(nm)].translation            # model body in palm frame
        errs.append((nm, float(np.linalg.norm(model_p - usd_p))))
    mean = np.mean([e for _, e in errs]); mx = max(e for _, e in errs)
    print(f"[verify {side}] {len(errs)} bodies, palm-relative FK err vs USD: mean={mean:.5f} max={mx:.5f} m")
    worst = sorted(errs, key=lambda x: -x[1])[:4]
    print(f"[verify {side}] worst: {[(n, round(e,4)) for n,e in worst]}")
    return mean < 1e-3


if __name__ == "__main__":
    usd = "source/robotis_sh5/data/robots/G1/G1_shadow.usd"
    ok_r = _verify(usd, "r")
    ok_l = _verify(usd, "l")
    print(f"\n[done] verify r={'OK' if ok_r else 'FAIL'} l={'OK' if ok_l else 'FAIL'}")
