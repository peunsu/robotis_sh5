"""Graft the Shadow hand subtree onto the (Dex3-stripped) G1 composite USD (bimanual).

STANDALONE pxr (env_isaaclab python) — LOCAL USDs → NO Isaac Sim, fast. Rebuilds from the
clean stripped backup `G1_shadow_stripped.usd` (idempotent).

- RIGHT hand: copy the repo Shadow `shadow_hand` subtree → robot0_r_*.
- LEFT hand:  copy → robot0_l_*, then a PROPER geometric mirror across the pelvis XZ-plane
  (negate Y): body transforms M·T·M, mesh points negate-Y + reverse winding, joint frames
  (localPos Y-negate, localRot (w,x,y,z)→(w,-x,y,-z)). A negative-scale "mirror" does NOT
  work in PhysX (the earlier attempt left a right-shaped left hand).
- Shadow JOINT prims are MERGED into G1's existing `/joints` scope (copying the Shadow
  `joints` SCOPE onto G1's clobbers the 29 body joints → body disintegrates).
- mount FixedJoint {side}_wrist_yaw_link → robot0_{r,l}_wrist (rotation tunable via CLI).
- REBUILD isaac:physics:robotJoints/robotLinks from actual prims; prune dangling.

Usage: /home/peunsu/anaconda3/envs/env_isaaclab/bin/python scripts/process_dataset/assets/graft_shadow_onto_g1.py \
           --sides rl --rax y --rdeg 90 --lax y --ldeg 90
"""

import argparse
import math
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

_DATA = Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data"
_STRIPPED = _DATA / "robots" / "G1" / "G1_shadow_stripped.usd"
_COMP = _DATA / "robots" / "G1" / "G1_shadow.usd"
_SHADOW = _DATA / "robots" / "FFW" / "FFW_SH5_shadow.usd"
_ROOT = "/g1_29dof_with_hand_rev_1_0"
_JOINTS = f"{_ROOT}/joints"
_SH_ROOT = "/Root/FFW_SH5_simplified_dex/shadow_hand"
_WRIST = {"r": "right_wrist_yaw_link", "l": "left_wrist_yaw_link"}
_MOUNT_POS = {"r": (0.0, 0.0, 0.0), "l": (0.0, 0.0, 0.0)}
_MOUNT_ROT = {"r": (1.0, 0.0, 0.0, 0.0), "l": (1.0, 0.0, 0.0, 0.0)}  # wxyz, set from CLI


def _axis_quat(axis, deg):
    h = math.radians(deg) / 2.0
    s = math.sin(h)
    v = {"x": (s, 0, 0), "y": (0, s, 0), "z": (0, 0, s)}[axis]
    return (math.cos(h), *v)


def _qmul(a, b):  # wxyz, apply b then a (a∘b)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw)


def _ren(elem, pfx):
    return f"robot0_{pfx}_{elem[len('robot0_'):]}" if elem.startswith("robot0_") else elem


def _remap(path_str, pfx):
    if not path_str.startswith(_SH_ROOT):
        return path_str
    rel = path_str[len(_SH_ROOT):].lstrip("/")
    if not rel:
        return _ROOT
    parts = rel.split("/")
    if parts[0] == "joints":
        return _JOINTS + "/" + "/".join(_ren(e, pfx) for e in parts[1:])
    return _ROOT + "/" + "/".join(_ren(e, pfx) for e in parts)


def graft(comp, cl, sh, sh_flat, pfx):
    sh_root = sh.GetPrimAtPath(_SH_ROOT)
    body_names = [c.GetName() for c in sh_root.GetChildren() if c.GetName() != "joints"]
    for name in body_names:
        Sdf.CopySpec(sh_flat, Sdf.Path(f"{_SH_ROOT}/{name}"), cl, Sdf.Path(f"{_ROOT}/{name}"))
    joint_names = [c.GetName() for c in sh.GetPrimAtPath(f"{_SH_ROOT}/joints").GetChildren()]
    for jn in joint_names:
        Sdf.CopySpec(sh_flat, Sdf.Path(f"{_SH_ROOT}/joints/{jn}"), cl, Sdf.Path(f"{_JOINTS}/{jn}"))
    comp.Save()

    edit = Sdf.BatchNamespaceEdit()
    roots = [f"{_ROOT}/{n}" for n in body_names] + [f"{_JOINTS}/{n}" for n in joint_names]
    paths = []
    for r in roots:
        for prim in Usd.PrimRange(comp.GetPrimAtPath(r)):
            paths.append(prim.GetPath().pathString)
    for p in sorted(set(paths), key=len, reverse=True):
        leaf = p.split("/")[-1]
        if leaf.startswith("robot0_"):
            edit.Add(p, p.rsplit("/", 1)[0] + "/" + _ren(leaf, pfx))
    if not cl.Apply(edit):
        raise RuntimeError(f"[{pfx}] rename failed")
    comp.Save()

    for prim in comp.Traverse():
        for rel in prim.GetRelationships():
            tg = rel.GetTargets()
            new = [Sdf.Path(_remap(t.pathString, pfx)) if t.pathString.startswith(_SH_ROOT) else t for t in tg]
            if new != list(tg):
                rel.SetTargets(new)
    comp.Save()


_M = Gf.Matrix4d(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)  # reflect Y


def mirror_left(comp):
    """Reflect all robot0_l_* prims across the pelvis XZ-plane (proper geometric mirror)."""
    for prim in comp.Traverse():
        path = prim.GetPath().pathString
        if "robot0_l_" not in path:
            continue
        # body/xform transforms: T → M·T·M  (proper SE(3), keeps det +1)
        if prim.IsA(UsdGeom.Xformable) and not prim.IsA(UsdPhysics.Joint):
            xf = UsdGeom.Xformable(prim)
            T = xf.GetLocalTransformation()
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(_M * T * _M)
        # mesh geometry: negate-Y points + reverse winding + reflect normals
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            pts = m.GetPointsAttr().Get()
            if pts:
                m.GetPointsAttr().Set([Gf.Vec3f(p[0], -p[1], p[2]) for p in pts])
            counts = m.GetFaceVertexCountsAttr().Get()
            idx = m.GetFaceVertexIndicesAttr().Get()
            if counts and idx:
                new, i = [], 0
                for c in counts:
                    new += list(idx[i:i + c])[::-1]
                    i += c
                m.GetFaceVertexIndicesAttr().Set(new)
            nrm = m.GetNormalsAttr().Get()
            if nrm:
                m.GetNormalsAttr().Set([Gf.Vec3f(n[0], -n[1], n[2]) for n in nrm])
        # joint frames: localPos Y-negate, localRot (w,x,y,z)→(w,-x,y,-z). For REVOLUTE joints the
        # rotation axis is a PSEUDOVECTOR — a reflection reverses the rotation SENSE, but the USD axis
        # token ("X") is unchanged by the frame mirror. So after mirroring we ADDITIONALLY rotate the
        # joint frame 180° about a perpendicular axis (Z) on BOTH sides: rest pose is preserved (both
        # localRot get the same extra rotation) while the token-X axis direction flips (+X→−X). This
        # makes a +joint angle curl the mirrored finger toward the palm/pad side (matching the human
        # fingertip targets, which sit at +Y in the left palm frame). WITHOUT it the left fingers curl
        # to the WRONG side (−Y) → curling moves AWAY from the target → the finger IK cannot reach the
        # curled human targets and clamps flexion at 0 → the left hand stays OPEN instead of gripping.
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            for a in (j.GetLocalPos0Attr(), j.GetLocalPos1Attr()):
                v = a.Get()
                if v is not None:
                    a.Set(Gf.Vec3f(v[0], -v[1], v[2]))
            axis_flip = prim.IsA(UsdPhysics.RevoluteJoint)
            _flip = Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.0, 1.0))   # 180° about Z → flips the token-X axis
            for a in (j.GetLocalRot0Attr(), j.GetLocalRot1Attr()):
                q = a.Get()
                if q is not None:
                    im = q.GetImaginary()
                    mq = Gf.Quatf(q.GetReal(), -im[0], im[1], -im[2])
                    if axis_flip:
                        mq = mq * _flip
                    a.Set(mq)
    comp.Save()


def add_mount(comp, pfx):
    wrist_body = f"{_ROOT}/robot0_{pfx}_wrist"
    mj = UsdPhysics.FixedJoint.Define(comp, f"{_JOINTS}/robot0_{pfx}_mount")
    mj.CreateBody0Rel().SetTargets([Sdf.Path(f"{_ROOT}/{_WRIST[pfx]}")])
    mj.CreateBody1Rel().SetTargets([Sdf.Path(wrist_body)])
    mj.CreateLocalPos0Attr(Gf.Vec3f(*_MOUNT_POS[pfx]))
    mj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    q = _MOUNT_ROT[pfx]
    mj.CreateLocalRot0Attr(Gf.Quatf(q[0], q[1], q[2], q[3]))
    mj.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    comp.Save()


def cleanup_and_schema(comp):
    for scope in ("left_hand", "right_hand"):
        p = f"{_ROOT}/{scope}"
        if comp.GetPrimAtPath(p):
            comp.RemovePrim(Sdf.Path(p))
    joints = [p.GetPath() for p in comp.Traverse() if p.IsA(UsdPhysics.Joint)]
    links = [p.GetPath() for p in comp.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    root = comp.GetPrimAtPath(_ROOT)
    for rn, vals in (("isaac:physics:robotJoints", joints), ("isaac:physics:robotLinks", links)):
        rel = root.GetRelationship(rn)
        if rel:
            rel.SetTargets(vals)
    for p in comp.Traverse():
        for rel in p.GetRelationships():
            tg = rel.GetTargets()
            keep = [t for t in tg if comp.GetPrimAtPath(t)]
            if len(keep) != len(tg):
                rel.SetTargets(keep)
    comp.Save()
    return len(joints), len(links)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sides", default="rl")
    ap.add_argument("--rax", default="y", choices=["x", "y", "z"])
    ap.add_argument("--rdeg", type=float, default=90.0)
    ap.add_argument("--rax2", default="x", choices=["x", "y", "z"])  # 2nd composed rot (flip)
    ap.add_argument("--rdeg2", type=float, default=0.0)
    ap.add_argument("--lax", default="y", choices=["x", "y", "z"])
    ap.add_argument("--ldeg", type=float, default=90.0)
    ap.add_argument("--lax2", default="x", choices=["x", "y", "z"])
    ap.add_argument("--ldeg2", type=float, default=0.0)
    args = ap.parse_args()
    _MOUNT_ROT["r"] = _qmul(_axis_quat(args.rax2, args.rdeg2), _axis_quat(args.rax, args.rdeg))
    # left mount = MIRROR of the right mount (w,x,y,z)→(w,-x,y,-z), so the two hands are
    # symmetric about the pelvis XZ-plane (matches the geometric mirror of the left hand).
    rw, rx, ry, rz = _MOUNT_ROT["r"]
    _MOUNT_ROT["l"] = (rw, -rx, ry, -rz)

    shutil.copy(_STRIPPED, _COMP)
    comp = Usd.Stage.Open(str(_COMP)); cl = comp.GetRootLayer()
    sh = Usd.Stage.Open(str(_SHADOW)); sh_flat = Usd.Stage.Open(str(_SHADOW)).Flatten()

    for pfx in args.sides:
        graft(comp, cl, sh, sh_flat, pfx)
        if pfx == "l":
            mirror_left(comp)
        add_mount(comp, pfx)
        print(f"[graft {pfx}] done{' (+mirror)' if pfx == 'l' else ''}")
    nj, nl = cleanup_and_schema(comp)

    # sanity: is the left hand mirrored (its palm world-Y ≈ −right palm-Y in pelvis frame)?
    if "r" in args.sides and "l" in args.sides:
        xc = UsdGeom.XformCache()
        rp = comp.GetPrimAtPath(f"{_ROOT}/robot0_r_palm")
        lp = comp.GetPrimAtPath(f"{_ROOT}/robot0_l_palm")
        if rp and lp:
            ry = xc.GetLocalToWorldTransform(rp).ExtractTranslation()[1]
            ly = xc.GetLocalToWorldTransform(lp).ExtractTranslation()[1]
            print(f"[mirror check] palm world-Y: right={ry:.3f} left={ly:.3f} (mirror ⇒ opposite signs)")

    body_j = [p.GetName() for p in comp.Traverse() if p.IsA(UsdPhysics.Joint)
              and not p.GetName().startswith("robot0_")]
    dang = sum(1 for p in comp.Traverse() for rel in p.GetRelationships()
               for t in rel.GetTargets() if not comp.GetPrimAtPath(t))
    print(f"[verify] joints={nj} links={nl}  G1 body joints intact={len(body_j)} (expect 29)  dangling={dang}")
    print(f"[verify] mounts: {[p.GetName() for p in comp.Traverse() if p.GetName().endswith('_mount')]}")
    print("[done]")


if __name__ == "__main__":
    main()
