"""Convert OMOMO object meshes to USD for Isaac Sim (OMOMO counterpart of grab_convert_obj_to_usd.py).

MUST run inside the Isaac Lab python env (launches SimulationApp):
    python scripts/process_dataset/assets/omomo_convert_obj_to_usd.py                 # every rigid object
    python scripts/process_dataset/assets/omomo_convert_obj_to_usd.py --object-id largebox

Rigid objects -> <obj>/<obj>.usd: dynamic, convex-decomposition collider (16 hulls), 0.5 kg, friction 1.0 —
the same _mesh_to_usd settings as ParaHome's and GRAB's rigid objects (OMOMO has no masses either; its boxes,
chairs and tables are really several kg). No context objects: the floor is the env's ground plane.
mop and vacuum are skipped: they are two parts with a non-revolute joint (omomo.py), which has no USD yet.

Reads : data/processed/omomo/assets/objects/<obj>/mesh/<obj>.obj   (omomo.py; object frame, m)
Writes: data/processed/omomo/assets/objects/<obj>/<obj>.usd
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parahome_convert_obj_to_usd import DEFAULT_FRICTION, DEFAULT_MASS, _mesh_to_usd  # noqa: E402  same settings

sys.path.append(str(Path(__file__).resolve().parents[1] / "dataset"))
import dataset_paths  # noqa: E402

_ASSET_OUT = dataset_paths.processed_root("omomo") / "assets" / "objects"
TWO_PART = ("mop", "vacuum")   # omomo.TWO_PART (not imported: omomo.py pulls in smplx/torch)


def convert(obj: str, overwrite: bool) -> None:
    if obj in TWO_PART:
        print(f"[skip] {obj}: two-part object (single_articulated) — no USD")
        return
    src = dataset_paths.object_mesh("omomo", obj)
    if not src.exists():
        print(f"[skip] {obj}: {src} 없음 — omomo.py 를 먼저 실행하세요")
        return
    out = _ASSET_OUT / obj / f"{obj}.usd"
    if out.exists() and not overwrite:
        print(f"[skip] {obj}: {out.name} exists")
        return
    print(f"[rigid] {obj} -> {out.relative_to(_ASSET_OUT)}", flush=True)
    _mesh_to_usd(src, out, mass=DEFAULT_MASS, friction=DEFAULT_FRICTION, collider="decomposition")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-id", type=str, default="", help="Single object to convert.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})  # noqa: F841

    objs = [args.object_id] if args.object_id else sorted(
        p.name for p in _ASSET_OUT.iterdir() if (p / "mesh" / f"{p.name}.obj").exists())
    n_fail = 0
    for obj in objs:
        try:
            convert(obj, args.overwrite)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[error] {obj}: {e}", flush=True)
    app.close()
    print(f"Done. {len(objs)} objects, {n_fail} failed.")


if __name__ == "__main__":
    main()
