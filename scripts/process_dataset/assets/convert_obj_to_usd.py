"""Convert centered object meshes (.obj) to USD for Isaac Sim.

Must be run inside the Isaac Lab Python environment:
    isaaclab.sh -p scripts/process_dataset/assets/convert_obj_to_usd.py [--dataset hocap] [--object-id G10_1]

Reads : data/processed/{dataset}/assets/objects/{obj_id}/visual.obj
Writes: data/processed/{dataset}/assets/objects/{obj_id}/visual.usd
"""

import argparse
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data"

# Per-object collision approximation overrides. Default is ConvexHull.
# Add object IDs that benefit from a simpler / more stable collider here.
_CUBE_COLLISION_OBJECTS: set[str] = {
    "G09_1",   # thin object with bumpy face → toppling; cube gives a stable flat base
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="oakink", choices=["oakink", "hocap"],
                        help="Dataset directory under data/processed/.")
    parser.add_argument("--object-id", type=str, default="", help="Single object ID to convert.")
    parser.add_argument("--mass", type=float, default=0.2, help="Object mass in kg.")
    parser.add_argument("--friction", type=float, default=1.0, help="Friction coefficient.")
    parser.add_argument("--overwrite", action="store_true", help="Re-convert existing USD files.")
    return parser.parse_args()


def convert_object(assets_dir: Path, obj_id: str, mass: float, friction: float) -> bool:
    # Isaac Sim modules imported here — after SimulationApp is running
    import isaaclab.sim as sim_utils
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas import schemas_cfg
    from pxr import Usd, UsdPhysics, UsdShade

    obj_path = assets_dir / obj_id / "visual.obj"
    usd_path = assets_dir / obj_id / "visual.usd"

    if not obj_path.exists():
        print(f"[error] OBJ not found: {obj_path}")
        return False

    if usd_path.exists():
        print(f"[skip] USD already exists: {usd_path}")
        return True

    # Per-object collider override: cube approximation for objects with unstable
    # mesh-derived collision (thin / bumpy faces that prevent stable resting).
    if obj_id in _CUBE_COLLISION_OBJECTS:
        mesh_collider = schemas_cfg.BoundingCubePropertiesCfg()
        collider_label = "BoundingCube"
    else:
        mesh_collider = schemas_cfg.ConvexHullPropertiesCfg()
        collider_label = "ConvexHull"

    print(f"[conv] {obj_path} → {usd_path}  (collider: {collider_label})")

    cfg = MeshConverterCfg(
        asset_path=str(obj_path),
        usd_dir=str(usd_path.parent),
        usd_file_name="visual.usd",
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            # Use Isaac Sim / PhysX default collision offsets (custom 0.005 / 0.0 values
            # commented out). These bake into the object's USD, so re-run this converter
            # with --overwrite for the change to take effect on existing visual.usd files.
            # contact_offset=0.005,
            # rest_offset=0.0,
        ),
        mesh_collision_props=mesh_collider,
    )

    MeshConverter(cfg)  # conversion happens automatically in AssetConverterBase.__init__

    # Apply physics material (friction/restitution) via pxr USD API
    try:
        stage = Usd.Stage.Open(str(usd_path))
        mat_path = "/World/PhysicsMaterial"
        mat = UsdShade.Material.Define(stage, mat_path)
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        phys_mat.CreateStaticFrictionAttr(friction)
        phys_mat.CreateDynamicFrictionAttr(friction)
        phys_mat.CreateRestitutionAttr(0.1)

        for prim in stage.Traverse():
            if prim.GetTypeName() in ("Mesh", "Sphere", "Cube"):
                binding = UsdShade.MaterialBindingAPI.Apply(prim)
                binding.Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")

        stage.GetRootLayer().Save()
        print(f"[phys] Applied friction={friction} to {obj_id}")
    except Exception as e:
        print(f"[warn] Could not apply physics material: {e}")

    print(f"[done] {usd_path}")
    return True


def main():
    args = parse_args()

    # SimulationApp must be created before any Isaac Sim / carb / pxr imports.
    # In workflow (python.sh) installs, PYTHONPATH is populated by python.sh.
    # In pip-package installs, it is populated by `import isaacsim`.
    # Either way, importing carb/pxr before SimulationApp is created will fail.
    try:
        import isaacsim  # noqa: F401 — required for pip-package installs
    except ModuleNotFoundError:
        pass  # workflow install: python.sh already set up PYTHONPATH

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    # --- all Isaac Sim imports happen after this point ---

    assets_dir = DATA_DIR / "processed" / args.dataset / "assets" / "objects"
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory not found: {assets_dir}")

    if args.overwrite:
        for usd in assets_dir.glob("*/visual.usd"):
            usd.unlink()

    obj_ids = [args.object_id] if args.object_id else [
        p.name for p in sorted(assets_dir.iterdir()) if p.is_dir()
    ]

    print(f"[dataset={args.dataset}] {len(obj_ids)} object(s) to convert under {assets_dir}")

    n_ok, n_fail = 0, 0
    for obj_id in obj_ids:
        ok = convert_object(assets_dir, obj_id, mass=args.mass, friction=args.friction)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"Done. converted={n_ok}, failed={n_fail}.")
    simulation_app.close()


if __name__ == "__main__":
    main()
