"""Convert ParaHome object scans to USD for Isaac Sim (Phase B, step 2).

MUST run inside the Isaac Lab python env (launches SimulationApp):
    ~/workspace/IsaacLab/isaaclab.sh -p scripts/process_dataset/assets/parahome_convert_obj_to_usd.py
    ... -p ... --object-id sink        # single object
    ... -p ... --articulated-only      # only the 8 articulated objects

Rigid objects (14: book, bookshelf, bowl, chair, cup, cuttingboard, desk,
diningtable, kettle, knife, pan, pot, potlid, salt) → one USD each with a
convex-DECOMPOSITION collider (household shapes are concave, so a single convex
hull is wrong). Static furniture (desk/table/bookshelf/chair) is left dynamic here;
mark it kinematic at scene-build time as needed.

Articulated objects (8) → one USD each with base + parts as separate rigid bodies
joined by revolute/prismatic joints. Joint axis (unit) / pivot / part rest pose come
from `articulation_spec.json` (built by parahome_build_articulation_spec.py, which
verified that joint_info axis/pivot live in the BASE frame). The base link is fixed
(furniture); parts are driven by the joint DOF (= joint_states trajectory at RL time).

Reads : data/raw/parahome/data/scan/{obj}/simplified/{base,part1,part2}.obj
        data/processed/parahome/assets/objects/articulation_spec.json
Writes: data/processed/parahome/assets/objects/{obj}/{obj}.usd
        (+ per-part {obj}_{part}.usd intermediates for articulated objects)

NOTE: the articulated-USD authoring path needs verification inside Isaac Sim (joint
frames / limits). Rigid path mirrors the proven scripts/process_dataset/assets/convert_obj_to_usd.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data"
_SCAN = _DATA_DIR / "raw" / "parahome" / "data" / "scan"
_ASSET_OUT = _DATA_DIR / "processed" / "parahome" / "assets" / "objects"

ARTICULATED_OBJECTS = [
    "drawer", "sink", "refrigerator", "gasstove",
    "laptop", "microwave", "trashbin", "washingmachine",
]
RIGID_OBJECTS = [
    "book", "bookshelf", "bowl", "chair", "cup", "cuttingboard", "desk",
    "diningtable", "kettle", "knife", "pan", "pot", "potlid", "salt",
]
DEFAULT_MASS = 0.5      # kg (manipulables); furniture is fixed so mass is moot
DEFAULT_FRICTION = 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--object-id", type=str, default="", help="Single object to convert.")
    p.add_argument("--articulated-only", action="store_true")
    p.add_argument("--rigid-only", action="store_true")
    p.add_argument("--context", action="store_true",
                   help="Build STATIC-collision context USDs (<obj>_ctx.usd) from base.obj for "
                        "support/collision use, uniform for rigid AND articulated objects (no joints).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _mesh_to_usd(obj_path: Path, usd_path: Path, mass: float, friction: float,
                 collider: str = "decomposition"):
    """Convert one .obj -> rigid-body USD with a collider + physics material.
    collider: 'decomposition' (concave-safe) | 'hull' | 'none' (visual/static)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas import schemas_cfg
    from pxr import Usd, UsdPhysics, UsdShade

    # Use the approximation-SPECIFIC subclass (it sets mesh_approximation_name + usd/physx
    # funcs). The base MeshCollisionPropertiesCfg has no `collision_approximation` field and
    # leaves usd_func/physx_func MISSING → constructing it directly fails.
    if collider == "decomposition":
        # CAP the sub-hull count at the SOURCE. Left at PhysX defaults, convex decomposition
        # emits up to max_convex_hulls=32 sub-hulls per mesh; a fingertip pressing into a
        # concave feature straddles several at once and PhysX generates one ~4-point manifold
        # PER touched sub-hull → up to ~32×4≈128 simultaneous contacts, which HARD-overflows the
        # ContactSensor per-prim contact buffer (unchecked index_select → device-side assert;
        # the overflow does NOT truncate). Bounding hulls at 16 bounds the worst-case simultaneous
        # fingertip contacts at 16×4 = 64, which is exactly the DERIVED value of the env's
        # ft_max_contact_points (g1_shadow_locomanip_env_cfg.py) — the sensor cap is a proof, not a
        # guess. 16 hulls still preserves handle voids / rims for graspability (2-4 would bridge
        # them). For tighter VRAM use max_convex_hulls=8 with ft_max_contact_points=32.
        # NB: the collider is baked at cook time — re-run with --overwrite after changing this.
        mesh_collision = schemas_cfg.ConvexDecompositionPropertiesCfg(
            max_convex_hulls=16, hull_vertex_limit=64)
    elif collider == "hull":
        mesh_collision = schemas_cfg.ConvexHullPropertiesCfg()
    else:
        mesh_collision = None

    cfg = MeshConverterCfg(
        asset_path=str(obj_path),
        usd_dir=str(usd_path.parent),
        usd_file_name=usd_path.name,
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=mesh_collision,
    )
    MeshConverter(cfg)

    # physics material (friction/restitution). Author it INSIDE the USD's defaultPrim
    # subtree so the binding travels when this part USD is referenced into the articulation
    # (a material at /World/... would fall outside the defaultPrim and dangle).
    try:
        stage = Usd.Stage.Open(str(usd_path))
        default_path = stage.GetDefaultPrim().GetPath()
        mat = UsdShade.Material.Define(stage, default_path.AppendChild("PhysicsMaterial"))
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(friction)
        pm.CreateDynamicFrictionAttr(friction)
        pm.CreateRestitutionAttr(0.0)
        for prim in stage.Traverse():
            if prim.GetTypeName() in ("Mesh",):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    mat, UsdShade.Tokens.strongerThanDescendants, "physics")
        stage.GetRootLayer().Save()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] material bind failed for {usd_path.name}: {e}")


def convert_rigid(obj: str, overwrite: bool):
    src = _SCAN / obj / "simplified" / "base.obj"
    if not src.exists():
        print(f"[skip] {obj}: no base.obj"); return
    out = _ASSET_OUT / obj / f"{obj}.usd"
    if out.exists() and not overwrite:
        print(f"[skip] {obj}: {out.name} exists"); return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[rigid] {obj} -> {out.relative_to(_ASSET_OUT)}")
    _mesh_to_usd(src, out, mass=DEFAULT_MASS, friction=DEFAULT_FRICTION, collider="decomposition")


def convert_context(obj: str, overwrite: bool):
    """STATIC-collision context USD: base.obj -> <obj>/<obj>_ctx.usd, convex-decomposition collider.
    Uniform for rigid AND articulated objects — for articulated furniture (gasstove/sink/...) this
    yields a SINGLE static support body with NO joints (a live articulation is wrong/expensive as a
    fixed support surface). base.obj is the static body the active object rests on. The env spawns it
    kinematic-frozen (overrides rigid props), so the authored rigid body is fine as-is."""
    src = _SCAN / obj / "simplified" / "base.obj"
    if not src.exists():
        print(f"[skip] {obj}: no base.obj"); return
    out = _ASSET_OUT / obj / f"{obj}_ctx.usd"
    if out.exists() and not overwrite:
        print(f"[skip] {obj}: {out.name} exists"); return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[context] {obj} -> {out.relative_to(_ASSET_OUT)}")
    _mesh_to_usd(src, out, mass=DEFAULT_MASS, friction=DEFAULT_FRICTION, collider="decomposition")


def convert_articulated(obj: str, spec: dict, overwrite: bool):
    """Author an articulated USD: fixed base + parts joined by revolute/prismatic joints.

    Uses the base frame convention verified by parahome_build_articulation_spec.py:
    joint axis (unit) and pivot are in the base frame; each part is placed at its
    rest pose `rest_T_part_in_base`.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

    parts_spec = spec.get(obj, {}).get("parts", [])
    out = _ASSET_OUT / obj / f"{obj}.usd"
    if out.exists() and not overwrite:
        print(f"[skip] {obj}: {out.name} exists"); return
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1) Convert base + each part mesh to its own rigid USD (convex-decomposition collider).
    part_usds = {}
    for pn in ["base"] + [p["part"] for p in parts_spec]:
        src = _SCAN / obj / "simplified" / f"{pn}.obj"
        if not src.exists():
            print(f"[warn] {obj}/{pn}: missing mesh"); continue
        pu = _ASSET_OUT / obj / f"{obj}_{pn}.usd"
        if overwrite or not pu.exists():
            _mesh_to_usd(src, pu, mass=DEFAULT_MASS, friction=DEFAULT_FRICTION, collider="decomposition")
        part_usds[pn] = pu

    # 2) Assemble an articulation stage referencing the part USDs + joints.
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    # MeshConverter authors part USDs at metersPerUnit=1.0; the assembly stage defaults to
    # 0.01 (cm) → parts would be 100× too small. Match units explicitly.
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    art = UsdGeom.Xform.Define(stage, f"/World/{obj}")
    UsdPhysics.ArticulationRootAPI.Apply(art.GetPrim())

    def _add_link(name, usd_path, T_in_base):
        link = UsdGeom.Xform.Define(stage, f"/World/{obj}/{name}")
        link.GetPrim().GetReferences().AddReference(str(usd_path))
        M = Gf.Matrix4d(*[float(x) for x in np.asarray(T_in_base).T.reshape(-1)])
        link.AddTransformOp().Set(M)
        return link

    # base link (fixed to the world -> furniture)
    base_link = _add_link("base", part_usds["base"], np.eye(4))
    fixed = UsdPhysics.FixedJoint.Define(stage, f"/World/{obj}/base_fixed")
    fixed.CreateBody1Rel().SetTargets([base_link.GetPath()])

    for p in parts_spec:
        pn = p["part"]
        if pn not in part_usds:
            continue
        T = np.asarray(p["rest_T_part_in_base"], dtype=np.float64)
        _add_link(pn, part_usds[pn], T)
        axis = np.asarray(p["axis"], dtype=np.float64)
        jpath = f"/World/{obj}/{pn}_joint"
        if p["joint_type"] == "prismatic":
            j = UsdPhysics.PrismaticJoint.Define(stage, jpath)
        else:
            j = UsdPhysics.RevoluteJoint.Define(stage, jpath)
        j.CreateBody0Rel().SetTargets([base_link.GetPath()])
        j.CreateBody1Rel().SetTargets([f"/World/{obj}/{pn}"])
        # The oblique `axis` (base frame) is carried on the joint's local rotation (map
        # local +X -> axis), so the principal axis token is "X".
        j.CreateAxisAttr("X")
        # body0=base, body1=part. USD requires the two anchor frames to map to the SAME
        # world point/orientation at rest DOF. T = part pose in base at rest.
        T = np.asarray(p["rest_T_part_in_base"], dtype=np.float64)
        Tinv = np.linalg.inv(T)
        R = _rot_x_to(axis)                       # base-frame rotation: +X -> axis
        if p["joint_type"] == "revolute" and p["pivot"] is not None:
            piv = np.asarray(p["pivot"], dtype=np.float64)
            piv_in_part = Tinv[:3, :3] @ piv + Tinv[:3, 3]   # pivot expressed in PART frame
            j.CreateLocalPos0Attr(Gf.Vec3f(*[float(x) for x in piv]))
            j.CreateLocalPos1Attr(Gf.Vec3f(*[float(x) for x in piv_in_part]))
        else:  # prismatic: anchor arbitrary, but frames must still coincide at rest
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(*[float(x) for x in Tinv[:3, 3]]))
        # LocalRot0 lives in base frame (= R); LocalRot1 lives in PART frame, so it must be
        # R_part_in_base^T @ R for the two joint frames to coincide at rest.
        j.CreateLocalRot0Attr(_mat_to_quatf(R))
        j.CreateLocalRot1Attr(_mat_to_quatf(T[:3, :3].T @ R))

    stage.GetRootLayer().Save()
    print(f"[artic] {obj} -> {out.relative_to(_ASSET_OUT)}  ({len(parts_spec)} joints)")


def _rot_x_to(axis):
    """Rotation matrix mapping local +X to the given unit axis."""
    x = np.array([1.0, 0.0, 0.0])
    a = axis / (np.linalg.norm(axis) + 1e-12)
    v = np.cross(x, a); s = np.linalg.norm(v); c = float(np.dot(x, a))
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _mat_to_quatf(R):
    # USD is row-vector: a column-vector rotation R is authored as its transpose (matches
    # _add_link, which flattens T.T). Build Gf.Matrix3d from R.T before GetQuat().
    from pxr import Gf
    q = Gf.Rotation(Gf.Matrix3d(*[float(x) for x in R.T.reshape(-1)])).GetQuat()
    return Gf.Quatf(float(q.GetReal()), *[float(x) for x in q.GetImaginary()])


def main():
    args = parse_args()

    try:
        import isaacsim  # noqa: F401
    except ModuleNotFoundError:
        pass
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})  # noqa: F841

    spec_path = _ASSET_OUT / "articulation_spec.json"
    spec = json.load(open(spec_path)) if spec_path.exists() else {}

    todo_rigid = RIGID_OBJECTS
    todo_artic = ARTICULATED_OBJECTS
    if args.object_id:
        todo_rigid = [args.object_id] if args.object_id in RIGID_OBJECTS else []
        todo_artic = [args.object_id] if args.object_id in ARTICULATED_OBJECTS else []
    if args.articulated_only:
        todo_rigid = []
    if args.rigid_only:
        todo_artic = []

    if args.context:
        # Static-collision context USDs from base.obj (rigid + articulated alike).
        for obj in todo_rigid + todo_artic:
            try:
                convert_context(obj, args.overwrite)
            except Exception as e:  # noqa: BLE001
                print(f"[error] context {obj}: {e}")
    else:
        for obj in todo_rigid:
            try:
                convert_rigid(obj, args.overwrite)
            except Exception as e:  # noqa: BLE001
                print(f"[error] rigid {obj}: {e}")
        for obj in todo_artic:
            try:
                convert_articulated(obj, spec, args.overwrite)
            except Exception as e:  # noqa: BLE001
                print(f"[error] artic {obj}: {e}")
                import traceback; traceback.print_exc()

    from isaacsim import SimulationApp as _S  # noqa: F401
    app.close()
    print("Done.")


if __name__ == "__main__":
    main()
