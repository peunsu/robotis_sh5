"""Convert GRAB object meshes to USD for Isaac Sim (GRAB counterpart of parahome_convert_obj_to_usd.py).

MUST run inside the Isaac Lab python env (launches SimulationApp):
    python scripts/process_dataset/assets/grab_convert_obj_to_usd.py              # every mesh grab.py exported
    python scripts/process_dataset/assets/grab_convert_obj_to_usd.py --object-id mug

Manipulated objects -> <obj>/<obj>.usd: dynamic, convex-decomposition collider (16 hulls), 0.5 kg,
friction 1.0 — the same _mesh_to_usd settings as ParaHome's rigid objects (GRAB has no masses).
Table top -> table/ctx/table_ctx.usd: the static context collider, same settings as ParaHome's context
USDs, in its own ctx/ folder so it shares no Props/instanceable_meshes.usd with anything.

Reads : data/processed/grab/assets/objects/<obj>/mesh/<obj>.obj   (grab.py)
Writes: data/processed/grab/assets/objects/<obj>/<obj>.usd, table/ctx/table_ctx.usd
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parahome_convert_obj_to_usd import (  # noqa: E402  same collider / mass / material settings
    CONTEXT_COLLIDER, CONTEXT_CONTACT_OFFSET, CONTEXT_MAX_HULLS, CONTEXT_REST_OFFSET,
    DEFAULT_FRICTION, DEFAULT_MASS, _mesh_to_usd,
)

sys.path.append(str(Path(__file__).resolve().parents[1] / "dataset"))
import dataset_paths  # noqa: E402

_ASSET_OUT = dataset_paths.processed_root("grab") / "assets" / "objects"
CONTEXT_OBJECTS = ("table",)


def convert(obj: str, overwrite: bool) -> None:
    src = dataset_paths.object_mesh("grab", obj)
    if not src.exists():
        print(f"[skip] {obj}: {src} 없음 — grab.py 를 먼저 실행하세요")
        return
    if obj in CONTEXT_OBJECTS:
        out = _ASSET_OUT / obj / "ctx" / f"{obj}_ctx.usd"
        kw = dict(collider=CONTEXT_COLLIDER, max_hulls=CONTEXT_MAX_HULLS, kinematic=True,
                  contact_offset=CONTEXT_CONTACT_OFFSET, rest_offset=CONTEXT_REST_OFFSET)
    else:
        out = _ASSET_OUT / obj / f"{obj}.usd"
        kw = dict(collider="decomposition")
    if out.exists() and not overwrite:
        print(f"[skip] {obj}: {out.name} exists")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{'context' if obj in CONTEXT_OBJECTS else 'rigid'}] {obj} -> {out.relative_to(_ASSET_OUT)}", flush=True)
    _mesh_to_usd(src, out, mass=DEFAULT_MASS, friction=DEFAULT_FRICTION, **kw)


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
