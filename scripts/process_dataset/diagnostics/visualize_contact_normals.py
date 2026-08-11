"""Draw the precomputed per-link contact targets and reaction normals on the object mesh.

These normals are what the contact-force reward's orientation gate compares the robot link's own
inward pad normal against (`cos >= cos(contact_normal_gate_tol)`), so if they point the wrong way the
gate rejects correct grasps and accepts wrong ones. The arrays themselves check out — every active
entry is a unit vector and every inactive one is exactly zero — but that says nothing about direction,
which is what this draws.

A reaction normal should point OUT of the object surface at the contact point: it is the direction the
object pushes back on the link, i.e. anti-parallel to the direction the link presses in. So each arrow
should start on the surface and point away from the object's interior.

No GPU: reads hand_contact.npz + the object's base mesh and renders with matplotlib, so it can run
while training holds the device.

    python scripts/process_dataset/diagnostics/visualize_contact_normals.py \
        --clips s100_seg00_pan s101_seg12_knife
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_PROC = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/processed/parahome"

ap = argparse.ArgumentParser()
ap.add_argument("--clips", nargs="+", default=["s100_seg00_pan", "s101_seg12_knife"])
ap.add_argument("--cls", default="single_rigid")
ap.add_argument("--frames", type=int, default=4, help="frames sampled across the contact phase")
ap.add_argument("--arrow", type=float, default=0.03, help="arrow length (m)")
ap.add_argument("--out_dir", default="/tmp")
args = ap.parse_args()


_SCAN = "/home/peunsu/datasets/ParaHome/data/scan"


def load_mesh(name):
    """Object base mesh. The converted asset dir holds only USD, so read the ORIGINAL ParaHome scan —
    an earlier version of this script looked for base.obj next to the USD, found nothing, silently
    returned None, and drew arrows into an empty void where their direction could not be judged."""
    p = os.path.join(_SCAN, name, "simplified", "base.obj")
    if not os.path.exists(p):
        print(f"   [warn] no scan mesh for {name} at {p}")
        return None
    import trimesh
    m = trimesh.load(p, force="mesh")
    print(f"   mesh {name}: {len(m.vertices)} verts, {len(m.faces)} faces, "
          f"extent {np.round(m.extents, 3).tolist()} m")
    return m


def main():
    for clip in args.clips:
        s = os.path.join(_PROC, "smplx", args.cls, clip, "0")
        hc = np.load(os.path.join(s, "hand_contact.npz"), allow_pickle=True)
        tj = np.load(os.path.join(s, "trajectory.npz"), allow_pickle=True)
        mask, tgt, nrm = hc["mask"].astype(bool), hc["target"], hc["normal"]
        links = [str(x) for x in hc["link_names"]]
        oname = next((k.split("__")[1] for k in tj.files
                      if k.startswith("obj__") and k.endswith("__base")), None)
        mesh = load_mesh(oname) if oname else None
        verts = np.asarray(mesh.vertices, np.float32) if mesh is not None else None

        active = mask.any(axis=1)
        idx = np.where(active)[0]
        pick = idx[np.linspace(0, len(idx) - 1, args.frames).astype(int)] if len(idx) else []

        fig = plt.figure(figsize=(4.6 * len(pick), 5.0))
        fig.suptitle(f"{clip} — object-local contact targets and reaction normals "
                     f"(object: {oname}, {mask.sum()} active link-frames)", fontsize=11)
        for a, f in enumerate(pick):
            ax = fig.add_subplot(1, len(pick), a + 1, projection="3d")
            if verts is not None:
                q = verts[::max(1, len(verts) // 3000)]
                ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=0.6, c="0.82", alpha=0.5, linewidths=0)
            act = np.where(mask[f])[0]
            for j in act:
                p, v = tgt[f, j], nrm[f, j] * args.arrow
                # thumb vs the other fingers vs palm: the thumb opposes, so its arrows should point
                # roughly opposite the finger arrows on a real grasp
                col = "tab:red" if "th" in links[j] else ("tab:blue" if "palm" in links[j] else "tab:green")
                ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], color=col, linewidth=1.4,
                          arrow_length_ratio=0.35)
                ax.scatter(*p, s=9, c=col)
            ax.set_title(f"frame {f}  ({len(act)} links)", fontsize=9)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            # equal aspect around the contact cloud, so arrow directions are not visually sheared
            pts = tgt[f, act] if len(act) else np.zeros((1, 3))
            c0 = pts.mean(0)
            r = max(float(np.abs(pts - c0).max()), args.arrow) * 1.8
            ax.set_xlim(c0[0] - r, c0[0] + r); ax.set_ylim(c0[1] - r, c0[1] + r)
            ax.set_zlim(c0[2] - r, c0[2] + r)
            ax.view_init(elev=22, azim=-58)
        fig.text(0.5, 0.015, "red = thumb   blue = palm   green = other fingers   "
                             "(arrow = direction the OBJECT pushes back on the link)",
                 ha="center", fontsize=9)
        out = os.path.join(args.out_dir, f"contact_normals_{clip}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out}")

        # THE decisive check: each reference normal against the TRUE surface normal of the mesh face
        # nearest its target point. The outward-radial test used before cannot judge these two objects
        # (a pan is concave, a knife is a thin plate) and came out 50/50, which says nothing.
        act = mask
        if mesh is not None:
            import trimesh
            pts = tgt[act]
            _, _, fid = trimesh.proximity.closest_point(mesh, pts)
            fn = np.asarray(mesh.face_normals)[fid]
            v = nrm[act] / np.clip(np.linalg.norm(nrm[act], axis=-1, keepdims=True), 1e-9, None)
            cos = np.clip((v * fn).sum(-1), -1, 1)
            ang = np.degrees(np.arccos(np.abs(cos)))     # abs: face winding sign is not meaningful
            print(f"   vs TRUE surface normal (|angle|, deg): mean {ang.mean():5.1f}  median "
                  f"{np.median(ang):5.1f}  p90 {np.percentile(ang, 90):5.1f}")
            for t in (15, 30, 45, 60):
                print(f"     within {t:2d} deg : {100 * (ang < t).mean():5.1f}%")
            print(f"   surface distance of target points (mm): median "
                  f"{1000 * np.median(trimesh.proximity.closest_point(mesh, pts)[1]):.2f}")
        th = np.array(["th" in links[j] for j in range(len(links))])
        for lab, sel in (("thumb", th), ("fingers+palm", ~th)):
            mm = act & sel[None, :]
            if mm.any():
                v = nrm[mm] / np.clip(np.linalg.norm(nrm[mm], axis=-1, keepdims=True), 1e-9, None)
                print(f"   {lab:12s} mean dir ({v[:, 0].mean():+.3f},{v[:, 1].mean():+.3f},"
                      f"{v[:, 2].mean():+.3f})  n={int(mm.sum())}")


main()
