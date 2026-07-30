"""Make a robot USD instanceable by extracting per-link geometry subtrees to
separate USD files and replacing them with references with ``instanceable=True``.

Why this script:
    A robot USD imported from URDF often has all mesh data authored directly in
    the file. When that USD is spawned across many envs (Isaac Lab GridCloner),
    each clone holds its own copy of the mesh data, exhausting GPU/CPU memory.
    To enable USD-level instancing, each link's geometry must live in an external
    layer that the main USD references with ``instanceable=True``.

This script preserves all hand-tuned properties of the source USD (collision
meshes, physics materials, joint drives, etc.) — it only restructures the
geometry-bearing subtrees. Re-importing from URDF would regenerate the file
from scratch and would lose any post-import edits.

Usage:
    python scripts/process_dataset/assets/make_robot_usd_instanceable.py \
        <input.usd> <output.usd> [--root-prim /ffw_sh5_follower]

Then update env_cfg.py to point at the new instanceable USD.
"""

import argparse
import os
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom


def _has_mesh(prim: Usd.Prim) -> bool:
    """Return True if the prim subtree contains any UsdGeom.Mesh."""
    for p in Usd.PrimRange(prim):
        if p.GetTypeName() == "Mesh":
            return True
    return False


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def _apply_physx_collision_attrs(
    stage: Usd.Stage,
    contact_offset: float | None,
    rest_offset: float | None,
    torsional_patch_radius: float | None,
    min_torsional_patch_radius: float | None,
) -> int:
    """Author PhysX collision attributes on every Mesh prim in the stage.

    Required because instanced subtrees are read-only at runtime — Isaac Lab's
    ``modify_collision_properties`` call (driven by env_cfg's ``CollisionPropertiesCfg``)
    cannot write these attributes onto an instance. Baking them into the prototype
    layer here keeps the values in effect through instancing.
    """
    n_touched = 0
    pairs = [
        ("physxCollision:contactOffset", contact_offset),
        ("physxCollision:restOffset", rest_offset),
        ("physxCollision:torsionalPatchRadius", torsional_patch_radius),
        ("physxCollision:minTorsionalPatchRadius", min_torsional_patch_radius),
    ]
    for prim in stage.TraverseAll():
        if prim.GetTypeName() != "Mesh":
            continue
        for attr_name, value in pairs:
            if value is None:
                continue
            attr = prim.GetAttribute(attr_name)
            if not attr:
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
            attr.Set(float(value))
        n_touched += 1
    return n_touched


def make_instanceable(
    src_usd: Path,
    dst_usd: Path,
    root_prim_path: str,
    contact_offset: float | None = None,
    rest_offset: float | None = None,
    torsional_patch_radius: float | None = None,
    min_torsional_patch_radius: float | None = None,
) -> tuple[int, int]:
    """Extract each link's geometry subtree to a separate USD with ``instanceable=True``.

    PhysX collision attributes (contact_offset / rest_offset / torsional radii) are
    baked into the prototype layers when provided, so they survive instancing.

    Returns:
        (n_links_visited, n_subtrees_extracted)
    """
    src_usd = Path(src_usd).resolve()
    dst_usd = Path(dst_usd).resolve()
    dst_usd.parent.mkdir(parents=True, exist_ok=True)

    print(f"[copy] {src_usd} → {dst_usd}")
    Usd.Stage.Open(str(src_usd)).Export(str(dst_usd))

    dst_stage = Usd.Stage.Open(str(dst_usd))
    dst_layer = dst_stage.GetRootLayer()
    root_prim = dst_stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        raise SystemExit(f"Root prim not found in destination stage: {root_prim_path}")

    # Place extracted layers in a sibling folder so the main USD has a clean reference
    meshes_dir = dst_usd.parent / f"{dst_usd.stem}_meshes"
    meshes_dir.mkdir(exist_ok=True)

    # Collect prim paths to extract before mutating the stage.
    # We extract each direct child Xform of each link xform whose subtree contains a Mesh.
    to_extract: list[str] = []
    n_links = 0
    for link_prim in root_prim.GetChildren():
        if link_prim.GetTypeName() != "Xform":
            continue
        n_links += 1
        for child in link_prim.GetChildren():
            if child.GetTypeName() != "Xform":
                continue
            if _has_mesh(child):
                to_extract.append(str(child.GetPath()))

    print(f"[plan] {n_links} link xforms scanned; {len(to_extract)} geometry subtrees to extract.")

    n_extracted = 0
    for prim_path_str in to_extract:
        prim_path = Sdf.Path(prim_path_str)
        link_name = prim_path.GetParentPath().name
        child_name = prim_path.name

        # 1) Create a new external layer for this subtree.
        out_filename = f"{_safe_name(link_name)}__{_safe_name(child_name)}.usd"
        out_filepath = meshes_dir / out_filename

        # CreateNew overwrites any existing layer at the path.
        target_layer = Sdf.Layer.CreateNew(str(out_filepath))

        # 2) Copy the subtree spec from dst's root layer into the new layer at "/Geometry".
        target_prim_path = "/Geometry"
        Sdf.CreatePrimInLayer(target_layer, target_prim_path)
        Sdf.CopySpec(dst_layer, prim_path_str, target_layer, target_prim_path)

        # 3) Set stage metadata on the new layer and mark default prim.
        target_stage = Usd.Stage.Open(target_layer)
        UsdGeom.SetStageUpAxis(target_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(target_stage, 1.0)
        target_stage.SetDefaultPrim(target_stage.GetPrimAtPath(target_prim_path))

        # 3b) Bake PhysX collision attributes onto Mesh prims so they survive instancing.
        if any(v is not None for v in (contact_offset, rest_offset,
                                       torsional_patch_radius, min_torsional_patch_radius)):
            _apply_physx_collision_attrs(
                target_stage,
                contact_offset=contact_offset,
                rest_offset=rest_offset,
                torsional_patch_radius=torsional_patch_radius,
                min_torsional_patch_radius=min_torsional_patch_radius,
            )
        target_layer.Save()

        # 4) Remove the in-place spec from the main stage and replace it with a
        #    Xform that references the new layer with ``instanceable=True``.
        dst_stage.RemovePrim(prim_path)
        new_prim = dst_stage.DefinePrim(prim_path, "Xform")
        rel = os.path.relpath(out_filepath, dst_usd.parent)
        new_prim.GetReferences().AddReference("./" + rel.replace(os.sep, "/"),
                                              primPath=target_prim_path)
        new_prim.SetInstanceable(True)
        n_extracted += 1

    dst_layer.Save()
    print(f"[done] {n_extracted} subtrees → {meshes_dir.relative_to(dst_usd.parent)}/")
    return n_links, n_extracted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Path to source USD (non-instanced).")
    parser.add_argument("output", type=str, help="Path to write the instanced USD.")
    parser.add_argument("--root-prim", type=str, default="/ffw_sh5_follower",
                        help="Root xform path inside the USD (default: /ffw_sh5_follower).")
    # PhysX collision properties — match Isaac Lab CollisionPropertiesCfg.
    # Set to None (don't pass the flag) to skip writing that attribute.
    parser.add_argument("--contact-offset", type=float, default=0.005,
                        help="physxCollision:contactOffset baked on every Mesh prim "
                        "(matches env_cfg CollisionPropertiesCfg default).")
    parser.add_argument("--rest-offset", type=float, default=0.0,
                        help="physxCollision:restOffset baked on every Mesh prim.")
    parser.add_argument("--torsional-patch-radius", type=float, default=None)
    parser.add_argument("--min-torsional-patch-radius", type=float, default=None)
    args = parser.parse_args()

    n_links, n_extracted = make_instanceable(
        Path(args.input),
        Path(args.output),
        args.root_prim,
        contact_offset=args.contact_offset,
        rest_offset=args.rest_offset,
        torsional_patch_radius=args.torsional_patch_radius,
        min_torsional_patch_radius=args.min_torsional_patch_radius,
    )
    print(f"\nSummary: {n_links} links visited, {n_extracted} subtrees extracted and "
          f"marked instanceable.")
    if args.contact_offset is not None or args.rest_offset is not None:
        print(f"PhysX collision attrs baked: contact_offset={args.contact_offset}, "
              f"rest_offset={args.rest_offset}")
    print(f"Use this new USD by updating env_cfg.py's robot USD path → {args.output}")


if __name__ == "__main__":
    main()
