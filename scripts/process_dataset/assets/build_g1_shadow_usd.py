"""Author the composite G1 + bimanual Shadow-hand USD.

Pipeline (mirrors the FFW+Shadow recipe: flatten → surgery → instance):
  STAGE 1  flatten the (remote) G1_29DOF USD into a local layer.
  STAGE 2  remove the Dex3 hands (14 joints + 16 links + hand↔wrist AssemblerFixedJoints).
  STAGE 3  copy the Shadow `shadow_hand` subtree ×2 (right→robot0_r_*, left→mirror robot0_l_*),
           remapping all prim paths + relationship targets (+ tendons).           [NEXT]
  STAGE 4  mount FixedJoint  {side}_wrist_yaw_link → robot0_{r,l}_wrist  (+ mount xform). [NEXT]
  STAGE 5  make_robot_usd_instanceable.py.                                          [NEXT]

Run in tmux session 1 (Isaac Sim env; G1 USD is remote/S3 so the resolver is needed):
    python -u scripts/process_dataset/assets/build_g1_shadow_usd.py [--stage 12]
USD-layer surgery only (NO Articulation → no PhysX cooking).
"""

import argparse
from pathlib import Path

try:
    import isaacsim  # noqa: F401
except ModuleNotFoundError:
    pass
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import carb  # noqa: E402
_s = carb.settings.get_settings()
if not _s.get("/persistent/isaac/asset_root/cloud"):
    _s.set("/persistent/isaac/asset_root/cloud",
           _s.get("/persistent/isaac/asset_root/default")
           or "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1")

from isaaclab_assets.robots.unitree import G1_29DOF_CFG  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

_DATA = Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data"
_OUT = _DATA / "robots" / "G1" / "G1_shadow.usd"
_ROOT = "/g1_29dof_with_hand_rev_1_0"          # G1 USD root prim (from inspect_g1_asset.py)
_HAND_PREFIXES = ("left_hand_", "right_hand_")  # Dex3 links + joints all start with these


def stage12_flatten_and_strip_dex3():
    g1_usd = G1_29DOF_CFG.spawn.usd_path
    print(f"[stage1] flatten G1: {g1_usd}")
    src = Usd.Stage.Open(g1_usd)
    assert src is not None, "could not open G1 USD"
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    src.Flatten().Export(str(_OUT))
    print(f"[stage1] flattened → {_OUT}")

    stage = Usd.Stage.Open(str(_OUT))
    # pre-surgery counts
    def counts():
        j = sum(1 for p in stage.Traverse() if p.IsA(UsdPhysics.Joint))
        b = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
        return j, b
    j0, b0 = counts()

    # collect Dex3 prim paths: names starting with left_hand_/right_hand_, plus the
    # AssemblerFixedJoints that attach a hand palm to the wrist.
    to_remove = []
    for p in stage.Traverse():
        name = p.GetName()
        if name.startswith(_HAND_PREFIXES):
            to_remove.append(p.GetPath())
        elif name == "AssemblerFixedJoint" and p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            tg = [t.pathString for t in j.GetBody0Rel().GetTargets() + j.GetBody1Rel().GetTargets()]
            if any("hand" in t.lower() for t in tg):
                to_remove.append(p.GetPath())
    # remove deepest-first so parent removal never orphans an already-listed child
    for path in sorted(set(to_remove), key=lambda p: len(p.pathString), reverse=True):
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
    stage.Save()

    j1, b1 = counts()
    print(f"[stage2] removed {len(set(to_remove))} Dex3 prims")
    print(f"[stage2] joints {j0}→{j1}  (expect −16: 14 hand + 2 assembler)")
    print(f"[stage2] bodies {b0}→{b1}  (expect −16: hand links)")
    remaining_hand = [p.GetName() for p in stage.Traverse() if p.GetName().startswith(_HAND_PREFIXES)]
    print(f"[stage2] remaining hand prims (expect none): {remaining_hand}")
    wr = [p.GetName() for p in stage.Traverse()
          if p.HasAPI(UsdPhysics.RigidBodyAPI) and "wrist_yaw" in p.GetName()]
    print(f"[stage2] wrist_yaw mount links present: {wr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="12")
    args = ap.parse_args()
    if "12" in args.stage or args.stage == "all":
        stage12_flatten_and_strip_dex3()
    # STAGE 3-5 to follow (Shadow graft / mirror / mount / instance).
    print("[done]")
    app.close()


if __name__ == "__main__":
    main()
