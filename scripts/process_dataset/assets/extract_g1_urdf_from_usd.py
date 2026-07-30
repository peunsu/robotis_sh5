"""Extract the G1 BODY kinematic tree from the composite USD → a clean URDF whose joint
origins/axes EXACTLY match the sim (so pink-IK in this model is sim-consistent). The composite
USD has localPos1=(0,0,0) / localRot1=identity for every G1 body joint (verified) → child link
frame == joint frame == URDF convention, so a faithful URDF is directly writable. Also compares
the extracted joint origins to the official Unitree URDF to confirm WHY the retargeted seed
mismatched the sim (URDF↔USD joint-frame difference).

Standalone pxr (no Isaac Sim):
    /home/peunsu/anaconda3/envs/env_isaaclab/bin/python scripts/process_dataset/assets/extract_g1_urdf_from_usd.py
Output: source/robotis_sh5/data/robots/G1/urdf/g1_from_usd.urdf
"""

import os
import numpy as np
from pathlib import Path
from pxr import Usd, UsdPhysics

_DATA = Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data"
_USD = str(_DATA / "robots" / "G1" / "G1_shadow.usd")
_OUT = str(_DATA / "robots" / "G1" / "urdf" / "g1_from_usd.urdf")
_OFFICIAL = os.path.expanduser("~/.cache/robot_descriptions/unitree_ros/robots/g1_description/g1_29dof.urdf")


def quat_wxyz_to_rpy(w, x, y, z):
    """ZYX-euler (URDF rpy). Returns (roll, pitch, yaw)."""
    # roll (x)
    sinr = 2 * (w * x + y * z); cosr = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    siny = 2 * (w * z + x * y); cosy = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def main():
    s = Usd.Stage.Open(_USD)
    joints = []
    for prim in s.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        n = prim.GetName()
        if n.startswith("robot0_"):          # skip Shadow hand joints (body only)
            continue
        j = UsdPhysics.RevoluteJoint(prim)
        b0 = [t.name for t in j.GetBody0Rel().GetTargets()]
        b1 = [t.name for t in j.GetBody1Rel().GetTargets()]
        if not b0 or not b1:
            continue
        lp0 = j.GetLocalPos0Attr().Get(); lr0 = j.GetLocalRot0Attr().Get()
        lp1 = j.GetLocalPos1Attr().Get(); lr1 = j.GetLocalRot1Attr().Get()
        axis = j.GetAxisAttr().Get()
        lo = j.GetLowerLimitAttr().Get(); hi = j.GetUpperLimitAttr().Get()
        assert np.linalg.norm(list(lp1)) < 1e-5, f"{n}: localPos1 != 0 (URDF assumption broken)"
        rr = lr1.GetReal(); ii = lr1.GetImaginary()
        assert abs(rr - 1.0) < 1e-4 and np.linalg.norm([ii[0], ii[1], ii[2]]) < 1e-4, \
            f"{n}: localRot1 != identity"
        rpy = quat_wxyz_to_rpy(lr0.GetReal(), *lr0.GetImaginary())
        joints.append(dict(name=n, parent=b0[0], child=b1[0],
                           xyz=tuple(lp0), rpy=rpy, axis={"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis],
                           lo=np.deg2rad(lo) if lo is not None else -3.14,
                           hi=np.deg2rad(hi) if hi is not None else 3.14))
    print(f"[extract] {len(joints)} G1 body revolute joints")

    links = sorted(set([j["parent"] for j in joints] + [j["child"] for j in joints]))
    children = {j["child"] for j in joints}
    roots = [l for l in links if l not in children]
    print(f"[extract] {len(links)} links, root(s)={roots}")

    # write URDF (minimal: dummy inertials, no meshes — kinematics only for IK)
    lines = ['<?xml version="1.0"?>', '<robot name="g1_from_usd">']
    for l in links:
        lines += [f'  <link name="{l}">',
                  '    <inertial><mass value="1.0"/>',
                  '      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>',
                  '    </inertial>', '  </link>']
    for j in joints:
        x, y, z = j["xyz"]; r, p, yw = j["rpy"]; ax = j["axis"]
        lines += [f'  <joint name="{j["name"]}" type="revolute">',
                  f'    <parent link="{j["parent"]}"/>', f'    <child link="{j["child"]}"/>',
                  f'    <origin xyz="{x:.6f} {y:.6f} {z:.6f}" rpy="{r:.6f} {p:.6f} {yw:.6f}"/>',
                  f'    <axis xyz="{ax[0]} {ax[1]} {ax[2]}"/>',
                  f'    <limit lower="{j["lo"]:.4f}" upper="{j["hi"]:.4f}" effort="100" velocity="10"/>',
                  '  </joint>']
    lines.append('</robot>')
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    open(_OUT, "w").write("\n".join(lines))
    print(f"[extract] wrote {_OUT}")

    # verify pinocchio loads + compare joint origins to the official URDF (the WHY)
    import pinocchio as pin
    m = pin.buildModelFromUrdf(_OUT, pin.JointModelFreeFlyer())
    print(f"[verify] pinocchio loads g1_from_usd: nq={m.nq} nv={m.nv}")
    if os.path.exists(_OFFICIAL):
        mo = pin.buildModelFromUrdf(_OFFICIAL, pin.JointModelFreeFlyer())
        print(f"\n[compare vs official URDF] joint origin (xyz) differences:")
        big = 0
        for j in joints:
            jid_u = m.getJointId(j["name"]) if m.existJointName(j["name"]) else None
            jid_o = mo.getJointId(j["name"]) if mo.existJointName(j["name"]) else None
            if jid_u and jid_o:
                pu = m.jointPlacements[jid_u].translation
                po = mo.jointPlacements[jid_o].translation
                d = np.linalg.norm(pu - po)
                Ru = m.jointPlacements[jid_u].rotation
                Ro = mo.jointPlacements[jid_o].rotation
                dR = np.rad2deg(np.arccos(np.clip((np.trace(Ru.T @ Ro) - 1) / 2, -1, 1)))
                if d > 0.01 or dR > 2.0:
                    big += 1
                    print(f"   {j['name']:28s} |Δxyz|={d:.4f}m  Δrot={dR:.1f}°")
        print(f"[compare] {big} joints differ >1cm or >2° between USD-derived and official URDF")
        if big == 0:
            print("   → origins MATCH: the sim/URDF gap is NOT joint origins (look elsewhere)")
        else:
            print("   → origins DIFFER: this is the URDF↔USD kinematic gap (retarget must use the USD-derived model)")
    print("[done]")


if __name__ == "__main__":
    main()
