"""Build a RENDER-ONLY watertight USD for a ParaHome scan object.

The ParaHome scans are torn open shells (pan: 479 boundary edges, doubleSided=False in USD), so a
camera looking into the object sees straight through the missing faces — fine for physics, ugly in a
paper figure. This script closes the surface morphologically and writes a SEPARATE visual-only USD:

    scan mesh (from <obj>.usd, instance proxies included)
      → drop dust components (< --min_faces faces; the pan scan is 1 real shell + 666 specks)
      → voxelise at --pitch
      → dilate --close / fill enclosed voids / erode --erode      (net wall growth = 2·(close-erode)·pitch)
      → isosurface (vtkFlyingEdges3D) + windowed-sinc smoothing
      → UsdGeom.Mesh, doubleSided=true, kinematic RigidBodyAPI, no collider

Why solidify rather than patch the holes on the surface:
  * `vtkFillHolesFilter` is a no-op here — the scan has non-manifold edges (740 edges with 4 faces,
    263 with 3) so the boundary loops are not fillable.
  * plain morphological CLOSING (dilate k → erode k) cannot remove a through-hole in a thin shell:
    the plug it builds is exactly as thick as the dilation, so the erosion eats it again. Measured:
    genus stayed 54 from k=2 through k=9.
  * growing the wall a little (erode one step less than the dilation) does remove them. Measured on
    the pan at pitch 2 mm: close=5/erode=4 → watertight, genus 0, surface sits 3.0 mm (p50) outside
    the scan — invisible on a 23x41 cm object.

Nothing in the physics pipeline is touched: the collider/rigid USD stays as it is, and the render
script picks the repaired mesh up only when you pass `--obj_usd`.

    python -u scripts/process_dataset/assets/parahome_make_render_mesh.py --obj pan
    → assets/objects/pan/pan_render_watertight.usd   (+ deviation report vs the original surface)
"""

import argparse
import os

import numpy as np
import trimesh
import vtk
from pxr import Usd, UsdGeom, UsdPhysics, Vt
from scipy import ndimage
from vtk.util import numpy_support

_PROC = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/processed/parahome"


def load_usd_mesh(path):
    """Concatenate every Mesh under the asset, with transforms baked (instance proxies included)."""
    stage = Usd.Stage.Open(path)
    parts = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() != "Mesh":
            continue
        m = UsdGeom.Mesh(prim)
        pts = np.array(m.GetPointsAttr().Get(), dtype=np.float64)
        cnt = np.array(m.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        idx = np.array(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if len(cnt) == 0:
            continue
        if not np.all(cnt == 3):                       # fan-triangulate the n-gons
            tris, o = [], 0
            for c in cnt:
                f = idx[o:o + c]
                tris += [[f[0], f[i], f[i + 1]] for i in range(1, c - 1)]
                o += c
            faces = np.array(tris, dtype=np.int64)
        else:
            faces = idx.reshape(-1, 3)
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)).reshape(4, 4)
        pts = pts @ xf[:3, :3] + xf[3, :3]             # USD is row-vector convention
        parts.append(trimesh.Trimesh(vertices=pts, faces=faces, process=False))
    if not parts:
        raise SystemExit(f"no Mesh prim in {path}")
    return trimesh.util.concatenate(parts)


def _to_vtk(mesh):
    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float64), deep=True))
    cells = np.hstack([np.full((len(mesh.faces), 1), 3, np.int64), mesh.faces.astype(np.int64)]).ravel()
    ca = vtk.vtkCellArray()
    ca.SetCells(len(mesh.faces), numpy_support.numpy_to_vtkIdTypeArray(np.ascontiguousarray(cells), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(ca)
    return pd


def _from_vtk(pd):
    v = numpy_support.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
    f = numpy_support.vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    return trimesh.Trimesh(vertices=v, faces=f, process=True)


def solidify(mesh, pitch, close, erode, smooth_iters, passband):
    vg = mesh.voxelized(pitch=pitch)
    vol = np.asarray(vg.matrix, dtype=bool)
    tf = np.array(vg.transform).reshape(4, 4)
    pad = close + 3
    vol = np.pad(vol, pad, mode="constant", constant_values=False)
    origin = tf[:3, 3] - pad * pitch                   # world position of voxel (0,0,0)

    st = ndimage.generate_binary_structure(3, 1)       # 6-neighbourhood
    if close > 0:
        vol = ndimage.binary_dilation(vol, st, iterations=close)
    vol = ndimage.binary_fill_holes(vol)               # enclosed voids only; an open bowl stays open
    if erode > 0:
        vol = ndimage.binary_erosion(vol, st, iterations=erode, border_value=0)

    img = vtk.vtkImageData()
    img.SetDimensions(*vol.shape)
    img.SetSpacing(pitch, pitch, pitch)
    img.SetOrigin(*[float(v) for v in origin])
    arr = numpy_support.numpy_to_vtk(vol.astype(np.uint8).ravel(order="F"), deep=True)
    img.GetPointData().SetScalars(arr)

    mc = vtk.vtkFlyingEdges3D()
    mc.SetInputData(img)
    mc.SetValue(0, 0.5)
    mc.ComputeNormalsOff()
    mc.ComputeGradientsOff()
    mc.Update()
    poly = mc.GetOutput()

    if smooth_iters > 0:                               # take the voxel staircase off
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(poly)
        sm.SetNumberOfIterations(int(smooth_iters))
        sm.SetPassBand(float(passband))
        sm.BoundarySmoothingOff()
        sm.FeatureEdgeSmoothingOff()
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOn()
        sm.Update()
        poly = sm.GetOutput()

    pts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    pol = numpy_support.vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)
    return trimesh.Trimesh(vertices=pts, faces=pol[:, 1:], process=True)


def write_usd(mesh, out, name):
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    gm = UsdGeom.Mesh.Define(stage, f"/{name}/geometry/mesh")
    gm.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
    gm.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.faces.astype(np.int32).ravel()))
    gm.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype=np.int32)))
    gm.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(
        mesh.vertex_normals.astype(np.float32)))
    gm.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    gm.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    gm.CreateDoubleSidedAttr(True)                     # belt and braces: no see-through backfaces
    gm.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(
        np.array(mesh.bounds, dtype=np.float32)))
    # isaaclab's RigidObject resolves the spawned prim through the physics view, so the asset needs a
    # RigidBodyAPI even though this is a visual-only mesh (spawned kinematic, never stepped). No
    # CollisionAPI on purpose: cooking a 120k-triangle collider would cost minutes and buys nothing.
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.5)
    stage.GetRootLayer().Save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True, help="object name under assets/objects/")
    ap.add_argument("--src", default="", help="source USD (default <obj>/<obj>.usd)")
    ap.add_argument("--out", default="", help="output USD (default <obj>/<obj>_render_watertight.usd)")
    ap.add_argument("--pitch", type=float, default=0.002, help="voxel size (m)")
    ap.add_argument("--close", type=int, default=5, help="dilation iterations (seals tunnels < 2*close*pitch)")
    ap.add_argument("--erode", type=int, default=-1,
                    help="erosion iterations; -1 = close-1, i.e. the wall grows by 2*pitch. Equal to "
                         "--close would be a plain closing, which re-opens the tunnels.")
    ap.add_argument("--min_faces", type=int, default=50, help="drop scan components smaller than this")
    ap.add_argument("--smooth", type=int, default=20, help="windowed-sinc iterations")
    ap.add_argument("--passband", type=float, default=0.05)
    ap.add_argument("--probe", type=int, default=20000, help="points sampled for the deviation report")
    args = ap.parse_args()

    base = os.path.join(_PROC, "assets", "objects", args.obj)
    src = args.src or os.path.join(base, f"{args.obj}.usd")
    out = args.out or os.path.join(base, f"{args.obj}_render_watertight.usd")

    raw = load_usd_mesh(src)
    raw = trimesh.Trimesh(raw.vertices, raw.faces, process=True)      # merge split vertices
    comps = raw.split(only_watertight=False)
    keep = [c for c in comps if len(c.faces) >= args.min_faces]
    m0 = trimesh.util.concatenate(keep) if keep else raw
    print(f"[mesh] source {src}")
    print(f"[mesh] components {len(comps)} → kept {len(keep)} (>= {args.min_faces} faces)")
    print(f"[mesh] in : V {len(m0.vertices)} F {len(m0.faces)} watertight={m0.is_watertight} "
          f"components={len(m0.split(only_watertight=False))} "
          f"boundary_edges={len(trimesh.grouping.group_rows(m0.edges_sorted, require_count=1))} "
          f"bounds(cm)={np.round(np.ptp(m0.bounds, axis=0) * 100, 2)}")

    erode = args.close - 1 if args.erode < 0 else args.erode
    m1 = solidify(m0, args.pitch, args.close, erode, args.smooth, args.passband)
    keep1 = [c for c in m1.split(only_watertight=False) if len(c.faces) >= args.min_faces]
    if len(keep1) >= 1:
        m1 = trimesh.util.concatenate(keep1)
    g = (2 - m1.euler_number) // 2
    print(f"[mesh] out: V {len(m1.vertices)} F {len(m1.faces)} watertight={m1.is_watertight} "
          f"components={len(m1.split(only_watertight=False))} "
          f"boundary_edges={len(trimesh.grouping.group_rows(m1.edges_sorted, require_count=1))} "
          f"genus~{g} volume={m1.volume * 1e6:.1f}cm^3 (dilate {args.close} / erode {erode} "
          f"→ wall +{(args.close - erode) * args.pitch * 2000:.0f}mm)")

    if args.probe > 0:                                 # how far the repaired skin sits from the scan
        pts, _ = trimesh.sample.sample_surface(m1, int(args.probe))
        d = np.abs(trimesh.proximity.signed_distance(m0, pts)) * 1000.0 if m0.is_watertight else \
            trimesh.proximity.closest_point(m0, pts)[1] * 1000.0
        print(f"[mesh] deviation vs scan surface (mm): p50 {np.percentile(d, 50):.2f}  "
              f"p95 {np.percentile(d, 95):.2f}  max {d.max():.2f}")

    if os.path.exists(out):
        os.remove(out)
    write_usd(m1, out, args.obj)
    print(f"[mesh] wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
