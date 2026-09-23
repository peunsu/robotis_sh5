"""Export one frame of a ParaHome clip's SMPL-X human as a USD mesh (for figures).

Runs the same SMPL-X configuration the dataset pipeline uses (`scripts/process_dataset/dataset/
parahome.py`: flat_hand_mean=True, use_pca=False, num_betas=20, gender from task_info.json), so the
body comes out in the SAME world frame as the retargeted robot and the object trajectory — the
render script can then draw human and robot from an identical camera.

    python -u scripts/process_dataset/assets/parahome_make_smplx_mesh.py \
        --clip s100_seg00_pan --frame 70
    → assets/smplx_frames/s100_seg00_pan_f0070.usd

Run it with the Isaac Lab python (the `smplx` package + torch live there). No Isaac app needed.
"""

import argparse
import json
import os

import numpy as np
import torch
from pxr import Usd, UsdGeom, Vt

_ROOT = "/home/peunsu/workspace/robotis_sh5"
_PROC = f"{_ROOT}/source/robotis_sh5/data/processed/parahome"
_SMPLX_DIR = f"{_ROOT}/models_smplx_v1_1/models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="s100_seg00_pan")
    ap.add_argument("--class", dest="cls", default="single_rigid")
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import smplx

    cdir = os.path.join(_PROC, "smplx", args.cls, args.clip)
    d = np.load(os.path.join(cdir, "0", "trajectory.npz"), allow_pickle=True)
    gender = json.load(open(os.path.join(cdir, "task_info.json"))).get("gender", "neutral")
    f = args.frame
    F = d["smplx_transl"].shape[0]
    if not 0 <= f < F:
        raise SystemExit(f"frame {f} out of range (clip has {F})")

    model = smplx.create(_SMPLX_DIR, model_type="smplx", gender=gender, use_pca=False,
                         flat_hand_mean=True, num_betas=20, num_expression_coeffs=10,
                         ext="npz", batch_size=1)
    t = lambda a: torch.as_tensor(np.asarray(a)[None], dtype=torch.float32)  # noqa: E731
    z = lambda n: torch.zeros(1, n)                                          # noqa: E731
    with torch.no_grad():
        o = model(betas=t(d["smplx_betas"]), global_orient=t(d["smplx_global_orient"][f]),
                  body_pose=t(d["smplx_body_pose"][f]),
                  left_hand_pose=t(d["smplx_hand_pose"][f][:45]),
                  right_hand_pose=t(d["smplx_hand_pose"][f][45:]),
                  transl=t(d["smplx_transl"][f]),
                  expression=z(10), jaw_pose=z(3), leye_pose=z(3), reye_pose=z(3))
    v = o.vertices[0].numpy().astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int64)
    print(f"[smplx] clip={args.clip} frame={f} gender={gender} V{len(v)} F{len(faces)}")
    print(f"[smplx] world bounds (m) min {np.round(v.min(0), 3)} max {np.round(v.max(0), 3)}")

    # vertex normals (area-weighted face normals accumulated per vertex)
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    vn = np.zeros_like(v)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)

    out = args.out or os.path.join(_PROC, "assets", "smplx_frames", f"{args.clip}_f{f:04d}.usd")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/human")
    stage.SetDefaultPrim(root.GetPrim())
    m = UsdGeom.Mesh.Define(stage, "/human/geometry/mesh")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(vn.astype(np.float32)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(True)
    m.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(
        np.stack([v.min(0), v.max(0)]).astype(np.float32)))
    stage.GetRootLayer().Save()
    print(f"[smplx] wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
