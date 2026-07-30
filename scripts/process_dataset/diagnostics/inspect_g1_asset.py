"""Inspect the built-in Unitree G1 USD — dump joints/links so the G1+Shadow composite USD
graft (remove Dex3, attach Shadow) can be authored against EXACT names/frames.

Uses USD stage TRAVERSAL (no Articulation / no sim.reset / no PhysX cooking — the earlier
Articulation approach spun for minutes cooking collision meshes). Read-only.

Run in tmux session 1 (env has Isaac Sim python set up):
    python scripts/process_dataset/diagnostics/inspect_g1_asset.py
"""

try:
    import isaacsim  # noqa: F401
except ModuleNotFoundError:
    pass
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

# Isaac Sim 5.1 sets only /persistent/isaac/asset_root/default; Isaac Lab reads /cloud.
# Mirror default→cloud BEFORE importing isaaclab (ISAACLAB_NUCLEUS_DIR baked at import).
import carb  # noqa: E402
_s = carb.settings.get_settings()
if not _s.get("/persistent/isaac/asset_root/cloud"):
    _s.set(
        "/persistent/isaac/asset_root/cloud",
        _s.get("/persistent/isaac/asset_root/default")
        or "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
    )

from isaaclab_assets.robots.unitree import G1_29DOF_CFG  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

_HAND = ("hand", "index", "middle", "thumb", "ring", "pinky", "little", "_dex")


def _is(name, keys):
    return any(k in name.lower() for k in keys)


def main():
    usd_path = G1_29DOF_CFG.spawn.usd_path
    print(f"\nG1 USD: {usd_path}")
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print("[error] could not open stage"); app.close(); return

    joints, links = [], []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joints.append((prim.GetName(), prim.GetTypeName(), prim.GetPath().pathString))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            links.append((prim.GetName(), prim.GetPath().pathString))

    print(f"\n================= JOINTS ({len(joints)}) =================")
    for grp, keys in [("legs", ("hip", "knee", "ankle")), ("waist", ("waist", "torso")),
                      ("arms", ("shoulder", "elbow", "wrist"))]:
        js = [j for j in joints if _is(j[0], keys)]
        print(f"  {grp} ({len(js)}): {[j[0] for j in js]}")
    hj = [j for j in joints if _is(j[0], _HAND)]
    print(f"  HAND/Dex3 — TO REMOVE ({len(hj)}): {[(j[0], j[1]) for j in hj]}")
    other = [j for j in joints if not _is(j[0], ("hip", "knee", "ankle", "waist", "torso",
                                                 "shoulder", "elbow", "wrist") + _HAND)]
    if other:
        print(f"  other ({len(other)}): {[(j[0], j[1]) for j in other]}")

    print(f"\n================= RIGID-BODY LINKS ({len(links)}) =================")
    print("  all:", [l[0] for l in links])
    print("  WRISTS (Shadow mount points):", [l[0] for l in links if "wrist" in l[0].lower()])
    print("  HAND links — TO REMOVE:", [l[0] for l in links if _is(l[0], _HAND)])

    print("\n================= WRIST rest-pose world transforms =================")
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    for name, path in [l for l in links if "wrist" in l[0].lower()]:
        m = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        t = m.ExtractTranslation()
        print(f"  {name} @ {path}: transl=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")

    print("\n[done] use the above to author the composite USD + fill env_cfg TODO(asset).")
    app.close()


if __name__ == "__main__":
    main()
