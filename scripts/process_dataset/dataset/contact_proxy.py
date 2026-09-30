"""접촉 계산용 물체 메시(contact proxy) — artist-made 메시(HUMOTO)용. 만들기(CLI)와 접촉 질의.

HUMOTO 물체는 스캔이 아니라 artist-made 라, parahome_hand_contact.py 의 정점 기준 접촉(물체 정점이 손 정점에서
gamma 안)이 넓은 면·긴 막대에서 접촉을 놓치고(정점이 꼭짓점에만 있다), 면 방향이 뒤집힌 부품의 법선이 반대가 되며,
겹친 부품의 안쪽 면에 접촉이 찍힌다. proxy 는 원본 표면의 바깥 껍질만 남긴 닫힌 메시다:

  1. 2.5 mm 격자에서 원본 표면까지의 정확한 (부호 없는) 거리 udf 를 표면 근처 띠에서만 계산한다.
     표면 샘플은 후보 삼각형을 고르는 데만 쓴다. 샘플 거리를 그대로 쓰면 큰 면의 샘플 틈에서 거리가
     부풀어 띠에 구멍이 생기고, 아래 flood fill 이 닫힌 부품 안으로 샌다.
  2. f = udf - r (r = 1.5 mm). 격자 가장자리에서 f>0 을 flood fill 한 것이 바깥이고, 바깥과 이어지지 않은 f>0 은
     안쪽(빈 공간)이라 f=-r 로 채운다. 정확한 udf 는 1-Lipschitz 라 r >= pitch/2 면 6-이웃 한 칸이 표면을
     건너뛸 수 없어 새지 않는다.
  3. f=0 등위면을 marching cubes 로 뽑는다 (레벨 1e-6: 격자값이 정확히 0 이면 비다양체 모서리가 생긴다).
     결과: 닫힌 한 덩어리, 법선은 전부 바깥, 원본보다 r 바깥. 두께 없는 판은 두께 2r 의 판이 된다.
면 방향은 전혀 쓰지 않으므로 뒤집힌 면·작은 구멍·겹친 부품과 무관하다.

접촉 (frame_contacts; parahome_hand_contact.frame_contacts 와 같은 반환·같은 링크 규칙):
  손 정점마다 proxy 위 최근접점과 부호 있는 거리(부호는 f), 실제 표면까지 거리 sd + r < gamma 면 접촉점.
  접촉점은 r 만큼 안쪽(실제 표면)으로 옮기고, 링크는 기존과 같이 그 접촉점에서 가장 가까운 손 정점의 링크,
  법선은 proxy 바깥 법선(얇은 판에서는 손 쪽 면). 손 정점 기준이라 접촉 밀도가 물체 삼각분할과 무관하다.
  2026-10-01 검증: ParaHome 스캔 15클립에서 기존 정점 방식과 마스크 IoU 93%, 프레임당 접촉 링크 8.7 = 8.7.

만들기 (scikit-image 필요; env_isaaclab 에 없으면 pip install scikit-image 또는 PYTHONPATH 로 추가; GPU 불필요):
    python scripts/process_dataset/dataset/contact_proxy.py --dataset humoto            # 조작 물체 전부
    python scripts/process_dataset/dataset/contact_proxy.py --dataset humoto --object table --overwrite
→ data/processed/<dataset>/assets/objects/<obj>/contact_proxy/{proxy.ply, field.npz, meta.json}
큰 물체(1.2 m 테이블)는 격자 약 4500만 칸, 최대 RSS 약 8 GB, 15 초.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_paths  # noqa: E402

PITCH = 0.0025          # m, 격자 간격
R_OFF = 0.0015          # m, proxy 가 원본 표면에서 떨어진 거리 (>= PITCH/2 여야 flood fill 이 새지 않는다)
K_CAND = 6              # 질의점마다 후보 삼각형 수 (가까운 표면 샘플 K 개의 삼각형)
assert R_OFF >= PITCH / 2
CATS = ("ok", "flipped", "thin", "hidden")


def _field_at(f, lo, pitch, pts):
    return ndimage.map_coordinates(f, ((pts - lo) / pitch).T, order=1, mode="constant", cval=1.0)


def _classify(f, lo, p, n):
    """원본 면 진단: p ± delta·n 이 proxy 바깥인지로 ok / flipped / thin(< ~2.5 mm) / hidden(안쪽 면)."""
    d = R_OFF + PITCH
    out_p = _field_at(f, lo, PITCH, p + d * n) > 0
    out_m = _field_at(f, lo, PITCH, p - d * n) > 0
    c = np.full(len(p), -1)
    c[out_p & ~out_m] = 0
    c[~out_p & out_m] = 1
    c[out_p & out_m] = 2
    c[~out_p & ~out_m] = 3
    return c


def build(mesh: trimesh.Trimesh, max_samples: int = 8_000_000) -> dict:
    """원본 메시 (물체 좌표계, m) → proxy (trimesh), f (격자, float32), lo, 진단 통계."""
    from skimage import measure   # 만들 때만 필요
    m = mesh
    samp, sfid = trimesh.sample.sample_surface(m, int(min(max_samples, max(20000, np.ceil(m.area / (PITCH / 3) ** 2)))), seed=0)
    samp, sfid = np.asarray(samp), np.asarray(sfid)
    lo = m.bounds[0] - (R_OFF + 4 * PITCH)
    dims = np.ceil((m.bounds[1] + (R_OFF + 4 * PITCH) - lo) / PITCH).astype(int) + 1
    ijk = np.clip(np.rint((samp - lo) / PITCH).astype(int), 0, dims - 1)
    band = np.zeros(dims, bool)
    band[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    k = int(np.ceil((R_OFF + 2 * PITCH) / PITCH))
    band = ndimage.binary_dilation(band, structure=np.ones((3, 3, 3), bool), iterations=k)
    bi = np.argwhere(band)
    del band
    tree = cKDTree(samp)
    udf = np.empty(len(bi))
    for s in range(0, len(bi), 200_000):
        c = lo + bi[s:s + 200_000] * PITCH
        ds, ii = tree.query(c, k=K_CAND, workers=8)
        cr = np.repeat(c, K_CAND, axis=0)
        cp = trimesh.triangles.closest_point(m.triangles[sfid[ii].ravel()], cr)
        de = np.nan_to_num(np.linalg.norm(cp - cr, axis=1), nan=np.inf).reshape(-1, K_CAND)
        udf[s:s + 200_000] = np.minimum(de.min(1), ds[:, 0])
    f = np.full(dims, np.float32(k * PITCH - R_OFF))
    f[bi[:, 0], bi[:, 1], bi[:, 2]] = (udf - R_OFF).astype(np.float32)
    del bi, udf
    pos = f > 0
    lab, _ = ndimage.label(pos)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    cavity = pos & ~np.isin(lab, border[border > 0])
    cavity_cm3 = float(cavity.sum()) * PITCH ** 3 * 1e6
    f[cavity] = -R_OFF
    del lab, pos, cavity
    v, fc, _, _ = measure.marching_cubes(f, level=1e-6, spacing=(PITCH,) * 3)
    proxy = trimesh.Trimesh(v + lo, fc, process=True)
    if proxy.volume < 0:
        proxy.invert()
    sub = np.random.default_rng(0).choice(len(samp), min(len(samp), 400_000), replace=False)
    sc = _classify(f, lo, samp[sub], m.face_normals[sfid[sub]])
    j = np.random.default_rng(1).choice(len(sub), min(len(sub), 20000), replace=False)
    d_o2p = trimesh.proximity.closest_point(proxy, samp[sub][j][sc[j] != 3])[1]
    d_p2o = trimesh.proximity.closest_point(m, trimesh.sample.sample_surface(proxy, 20000, seed=2)[0])[1]
    stats = dict(pitch=PITCH, r=R_OFF, grid=dims.tolist(), orig_faces=len(m.faces), proxy_faces=len(proxy.faces),
                 proxy_watertight=bool(proxy.is_watertight), proxy_bodies=int(proxy.body_count),
                 cavity_filled_cm3=round(cavity_cm3, 1),
                 orig_area_frac={CATS[i]: round(float((sc == i).mean()), 4) for i in range(4)},
                 orig_to_proxy_mm_max=round(float(d_o2p.max()) * 1e3, 2), proxy_to_orig_mm_max=round(float(d_p2o.max()) * 1e3, 2))
    return dict(proxy=proxy, f=f, lo=lo, stats=stats)


def save(out_dir: Path, built: dict, src: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    built["proxy"].export(out_dir / "proxy.ply")
    np.savez_compressed(out_dir / "field.npz", f=built["f"].astype(np.float16), lo=built["lo"], pitch=PITCH, r=R_OFF)
    meta = dict(built["stats"], source_mesh=str(src), source_mtime=os.path.getmtime(src),
                built=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


class ContactProxy:
    """proxy 질의: 최근접점·부호 있는 거리·바깥 법선 (모두 물체 좌표계)."""

    def __init__(self, d: Path):
        if not (d / "proxy.ply").exists():
            raise FileNotFoundError(f"contact proxy 가 없습니다: {d} — scripts/process_dataset/dataset/contact_proxy.py 로 먼저 만드세요")
        self.m = trimesh.load(d / "proxy.ply", process=False)
        z = np.load(d / "field.npz")
        self.f, self.lo, self.pitch, self.r = z["f"].astype(np.float32), z["lo"], float(z["pitch"]), float(z["r"])
        sp, self.sf = trimesh.sample.sample_surface(self.m, int(np.ceil(self.m.area / 0.0015 ** 2)), seed=0)
        self.tree = cKDTree(sp)
        self.fn = self.m.face_normals
        self.tri = self.m.triangles

    def query(self, p: np.ndarray, bound: float):
        """p (N,3) → sd (N,) proxy 까지 부호 있는 거리 (안쪽 음수; bound+3 mm 밖은 inf), cp (N,3), n (N,3)."""
        sd = np.full(len(p), np.inf)
        cp = np.zeros_like(p)
        nn = np.zeros_like(p)
        d0, _ = self.tree.query(p, k=1, distance_upper_bound=bound + 0.003, workers=8)
        near = np.where(np.isfinite(d0))[0]
        if not len(near):
            return sd, cp, nn
        q = p[near]
        _, ii = self.tree.query(q, k=K_CAND, workers=8)
        fid = self.sf[ii]
        qq = np.repeat(q, K_CAND, axis=0)
        c = trimesh.triangles.closest_point(self.tri[fid.ravel()], qq)
        d = np.nan_to_num(np.linalg.norm(c - qq, axis=1), nan=np.inf).reshape(-1, K_CAND)
        j = d.argmin(1)
        rows = np.arange(len(q))
        cp[near] = c.reshape(-1, K_CAND, 3)[rows, j]
        nn[near] = self.fn[fid[rows, j]]
        inside = _field_at(self.f, self.lo, self.pitch, q) < 0
        sd[near] = np.where(inside, -d[rows, j], d[rows, j])
        return sd, cp, nn


def frame_contacts(hand_w, hand_link, proxy: ContactProxy, R, op, L, gamma, num_contacts, normal_source, fps_fn):
    """parahome_hand_contact.frame_contacts 와 같은 반환 (mask (L,), target (L,3) 월드, normal (L,3) 물체 로컬 바깥,
    n_kept) 또는 접촉이 없으면 None. fps_fn = parahome_hand_contact._farthest_point_sample."""
    hl = (hand_w - op) @ R                                        # 손 정점, 물체 좌표계
    sd, cp, nl = proxy.query(hl, gamma)
    keep = np.where(sd + proxy.r < gamma)[0]                      # 실제 표면까지 gamma 안 (파고든 정점 포함)
    if keep.size == 0:
        return None
    cl = cp[keep] - proxy.r * nl[keep]                            # proxy → 실제 표면 (두께 없는 판이면 판 위)
    cw = cl @ R.T + op
    _, nh = cKDTree(hand_w).query(cw, k=1)                        # 링크 = 접촉점에서 가장 가까운 손 정점 (기존 규칙)
    clink = hand_link[nh]
    if normal_source == "surface":
        nrm_l = nl[keep]
    else:                                                         # to-hand: 접촉점 → 그 손 정점 방향
        nw = hand_w[nh] - cw
        nrm_l = (nw / np.clip(np.linalg.norm(nw, axis=1, keepdims=True), 1e-9, None)) @ R
    sel = fps_fn(cw, num_contacts)
    cw, clink, nrm_l = cw[sel], clink[sel], nrm_l[sel]
    mask = np.zeros(L, np.float32)
    target = np.zeros((L, 3), np.float32)
    normal = np.zeros((L, 3), np.float32)
    for li in range(L):
        s = clink == li
        if s.any():
            mask[li] = 1.0
            target[li] = cw[s].mean(0)
            v = nrm_l[s].mean(0)
            normal[li] = v / max(float(np.linalg.norm(v)), 1e-9)
    return mask, target, normal, len(cw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=dataset_paths.DATASETS, default="humoto")
    ap.add_argument("--object", default="", help="이 물체만 (기본: 클립들의 조작 물체 전부)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    root = dataset_paths.processed_root(a.dataset) / "smplx" / "single_rigid"
    objs = [a.object] if a.object else sorted({json.load(open(root / c / "task_info.json"))["manip_objects"][0]["name"]
                                               for c in os.listdir(root) if (root / c / "task_info.json").exists()})
    for o in objs:
        out = dataset_paths.contact_proxy_dir(a.dataset, o)
        if (out / "proxy.ply").exists() and not a.overwrite:
            print(f"[contact-proxy] {o}: 이미 있음 — 건너뜀 (--overwrite)")
            continue
        t0 = time.time()
        src = dataset_paths.object_mesh(a.dataset, o)
        b = build(trimesh.load(str(src), force="mesh", process=False))
        save(out, b, src)
        s = b["stats"]
        print(f"[contact-proxy] {o:20s} faces {s['orig_faces']:6d} -> {s['proxy_faces']:7d}  watertight {s['proxy_watertight']!s:5s} "
              f"bodies {s['proxy_bodies']}  cavity {s['cavity_filled_cm3']:7.0f} cm3  {s['orig_area_frac']}  "
              f"orig->proxy max {s['orig_to_proxy_mm_max']} mm  {time.time() - t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
