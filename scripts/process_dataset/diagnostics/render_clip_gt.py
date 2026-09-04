#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Render the GROUND-TRUTH ParaHome scene of a clip to mp4 (`clip_viz.mp4`).

Reproduces the existing `smplx/<class>/<clip>/0/clip_viz.mp4`: the SMPL-X human mesh driven by the
clip's own fit, every context object's scan mesh at its reference pose, and the MANIPULATED object
highlighted in red — i.e. exactly what the reference data says happened, with no robot and no physics.
Use it to eyeball a clip before training on it.

Everything comes from the clip's own `trajectory.npz` + `task_info.json`:
    smplx_{betas,body_pose,global_orient,hand_pose,transl}   -> SMPL-X FK (gender from task_info)
    obj__<name>__base                                        -> the manipulated object (RED)
    ctx__<name>__base                                        -> scene/context objects (GREY)
Object meshes are the raw ParaHome scans (`data/scan/<obj>/simplified/base.obj`) — the SOURCE
geometry, deliberately NOT the convex-decomposition colliders, because this is a picture of the
reference, not of what PhysX sees.

Context objects are drawn at their FRAME-0 pose, matching how the env spawns them (kinematic, frozen
at frame 0 — g1_shadow_sonic_residual_env.py:490-496). The manipulated object is drawn per frame.

CPU/GPU-agnostic offscreen render via pyrender's EGL/OSMesa backend — NO Isaac Sim.
Run with an interpreter that has smplx + torch + pyrender + trimesh:
    /home/peunsu/anaconda3/envs/env_isaaclab/bin/python

    ... render_clip_gt.py --clip s101_seg12_knife
    ... render_clip_gt.py --clip a --clip b --clip c        # several
    ... render_clip_gt.py --class single_rigid --all        # every clip in a class
    ... render_clip_gt.py --clip x --out /tmp/x.mp4 --fps 30 --size 1280x720
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# pyrender needs a headless GL backend chosen BEFORE it is imported.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_ROOT = Path(__file__).resolve().parents[3]
_DATA = _ROOT / "source" / "robotis_sh5" / "data"
_SMPLX_DIR = _ROOT / "models_smplx_v1_1" / "models"      # same location parahome.py uses
_SCAN = _DATA / "raw" / "parahome" / "data" / "scan"
_PROC = _DATA / "processed" / "parahome" / "smplx"

BG = np.array([0.09, 0.10, 0.16])       # dark navy, matching the existing clip_viz.mp4
C_BODY = (0.94, 0.85, 0.76, 1.0)        # skin
C_CTX = (0.72, 0.74, 0.78, 1.0)         # grey furniture
C_ACT = (0.86, 0.22, 0.20, 1.0)         # red — the manipulated object


def quat_to_R(q) -> np.ndarray:
    """(4,) wxyz Hamilton -> (3,3)."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def pose_to_T(pose7) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_R(pose7[3:7])
    T[:3, 3] = pose7[:3]
    return T


def smplx_vertices(npz, gender: str):
    """Run SMPL-X FK for every frame -> (F, V, 3) world vertices, plus the face array."""
    import torch
    import smplx
    F = len(npz["smplx_body_pose"])
    model = smplx.create(str(_SMPLX_DIR), model_type="smplx", gender=gender, use_pca=False,
                         flat_hand_mean=True, num_betas=20, num_expression_coeffs=10,
                         batch_size=F)
    hp = np.asarray(npz["smplx_hand_pose"], np.float32)          # (F,90) = left45 | right45
    with torch.no_grad():
        out = model(
            betas=torch.as_tensor(np.tile(npz["smplx_betas"][None], (F, 1)), dtype=torch.float32),
            global_orient=torch.as_tensor(npz["smplx_global_orient"], dtype=torch.float32),
            body_pose=torch.as_tensor(npz["smplx_body_pose"], dtype=torch.float32),
            left_hand_pose=torch.as_tensor(hp[:, :45], dtype=torch.float32),
            right_hand_pose=torch.as_tensor(hp[:, 45:], dtype=torch.float32),
            transl=torch.as_tensor(npz["smplx_transl"], dtype=torch.float32),
            return_verts=True)
    return out.vertices.numpy(), model.faces.astype(np.int32)


def load_scan(obj: str):
    import trimesh
    p = _SCAN / obj / "simplified" / "base.obj"
    if not p.exists():
        return None
    return trimesh.load(str(p), force="mesh", process=False)


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """Camera-to-world matrix in pyrender's convention (-Z forward, +Y up)."""
    eye, target, up = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-9)
    s = np.cross(f, up)
    s /= max(np.linalg.norm(s), 1e-9)
    u = np.cross(s, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = s, u, -f, eye
    return T


def render_clip(clip: str, klass: str, out_path: Path | None, size, fps: int,
                yaw: float, elev: float, zoom: float,
                eye_override=None, lookat_override=None,
                env_context: bool = False, ctx_radius: float = 1.0,
                ctx_support_radius: float = 1.5,
                follow_obj: bool = False, cam_dist: float = 0.9) -> str:
    import pyrender
    import trimesh
    import imageio.v2 as imageio

    clip_dir = _PROC / klass / clip / "0"
    npz_path = clip_dir / "trajectory.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    d = np.load(npz_path, allow_pickle=True)

    info_path = _PROC / klass / clip / "task_info.json"
    gender = "neutral"
    if info_path.exists():
        gender = json.load(open(info_path)).get("gender", "neutral") or "neutral"

    act_keys = [k for k in d.files if k.startswith("obj__") and k.endswith("__base")]
    ctx_keys = [k for k in d.files if k.startswith("ctx__") and k.endswith("__base")]
    # [env-ctx] --env_context 면 env 가 실제로 스폰하는 것만 남깁니다 (g1_shadow_sonic_residual_env.py
    # _ctx_spawn 과 같은 규칙): 활성 물체의 궤적에서 context_radius 안에 있는 것 + 그 아래에 있는
    # 지지물 하나. 롤아웃 영상과 장면을 맞출 때 씁니다. 기본은 종전대로 전부 그립니다.
    if env_context and act_keys:
        _act = np.asarray(d[act_keys[0]], float)
        _axy, _a0 = _act[:, :2], _act[0]
        _c = [(k, np.asarray(d[k][0], float),
               float(np.linalg.norm(_axy - np.asarray(d[k][0], float)[None, :2], axis=1).min()))
              for k in ctx_keys]
        _keep = {k for k, p0, dm in _c if dm < ctx_radius}
        _below = [(float(np.linalg.norm(_a0[:2] - p0[:2])), k) for k, p0, dm in _c
                  if p0[2] < _a0[2] and float(np.linalg.norm(_a0[:2] - p0[:2])) < ctx_support_radius]
        if _below:
            _keep.add(min(_below)[1])
        ctx_keys = [k for k in ctx_keys if k in _keep]
    F = len(d["smplx_body_pose"])
    print(f"  frames={F} gender={gender} active={[k.split('__')[1] for k in act_keys]} "
          f"context={len(ctx_keys)}")

    verts, faces = smplx_vertices(d, gender)

    # ---- static scene: context objects at their FRAME-0 pose (matches the env's kinematic spawn) --
    scene = pyrender.Scene(bg_color=[*BG, 1.0], ambient_light=[0.35, 0.35, 0.38])
    pts_for_framing = [verts.reshape(-1, 3)]
    for k in ctx_keys:
        m = load_scan(k.split("__")[1])
        if m is None:
            continue
        mm = m.copy()
        mm.apply_transform(pose_to_T(np.asarray(d[k][0], float)))
        scene.add(pyrender.Mesh.from_trimesh(
            mm, material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=C_CTX, metallicFactor=0.05, roughnessFactor=0.85), smooth=False))
        pts_for_framing.append(np.asarray(mm.vertices))

    # ---- dynamic nodes: the human and the manipulated object(s), re-posed per frame ----
    body_tm = trimesh.Trimesh(vertices=verts[0], faces=faces, process=False)
    body_node = scene.add(pyrender.Mesh.from_trimesh(
        body_tm, material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=C_BODY, metallicFactor=0.0, roughnessFactor=0.9), smooth=True))

    act_nodes = []
    for k in act_keys:
        m = load_scan(k.split("__")[1])
        if m is None:
            continue
        node = scene.add(pyrender.Mesh.from_trimesh(
            m, material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=C_ACT, metallicFactor=0.1, roughnessFactor=0.6), smooth=False),
            pose=pose_to_T(np.asarray(d[k][0], float)))
        act_nodes.append((k, node))
        pts_for_framing.append(np.asarray(m.vertices) + np.asarray(d[k][:, :3]).mean(axis=0))

    # ---- framing: fit everything the clip touches, then back off ----
    allp = np.concatenate(pts_for_framing, axis=0)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    centre = 0.5 * (lo + hi)
    extent = float(np.linalg.norm(hi - lo))
    dist = max(1.5, extent * 0.75) * zoom

    # AUTO AZIMUTH: put the camera on the side the manipulated object is on, relative to the body, so
    # the object sits BETWEEN the camera and the person and cannot be hidden behind them. Without this
    # a small object (the knife is 24x5x9 cm, vs the pan's 23x41x16) disappears entirely whenever the
    # subject happens to stand between it and a fixed camera — which is exactly what a fixed yaw=45
    # did for s101_seg12_knife (knife at y=-1.1, body at y=-0.5).
    if yaw is None:
        if act_keys:
            obj_xy = np.mean([np.asarray(d[k][:, :2], float).mean(axis=0) for k in act_keys], axis=0)
        else:
            obj_xy = centre[:2]
        body_xy = np.asarray(d["smplx_transl"][:, :2], float).mean(axis=0)
        v = obj_xy - body_xy
        yaw = float(np.degrees(np.arctan2(v[1], v[0]))) if np.linalg.norm(v) > 1e-3 else 45.0
        print(f"  auto yaw={yaw:.1f} deg (camera placed on the object's side of the body)")
    az, el = np.radians(yaw), np.radians(elev)
    _cam_dir = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    eye = centre + dist * _cam_dir
    # [rollout-cam] 롤아웃 영상과 동일한 시점을 쓰려면 eval 디렉터리의 viewer_*.json 에 저장된
    # eye_env_local / lookat_env_local 을 그대로 넣습니다 (env 가 궤적 통계로 한 번 계산한 값).
    if eye_override is not None:
        eye = np.asarray(eye_override, float)
    if lookat_override is not None:
        centre = np.asarray(lookat_override, float)
    # [ROLLBACK MARKER: follow-obj] --follow_obj 면 매 프레임 물체+손끝 중점을 주시하며 cam_dist
    # 거리에서 따라갑니다. 물체가 크게 이동하는 클립(예: s207_seg06_kettle 2.05 m)에서 고정 시점으로
    # 확대하면 프레임을 벗어나므로, 확대 관찰에는 추종이 필요합니다. 방향(yaw/elev)은 고정입니다.
    _fw_tg = None
    if follow_obj:
        _tg = []
        for _f in range(F):
            _p = []
            if act_keys:
                _p.append(np.mean([np.asarray(d[k][_f][:3], float) for k in act_keys], axis=0))
            if "fingertip_pad_pos" in d:
                _p.append(np.asarray(d["fingertip_pad_pos"][_f], float).mean(axis=0))
            _tg.append(np.mean(_p, axis=0) if _p else centre)
        _tg = np.stack(_tg)
        _k = 9                                   # 이동평균으로 손끝 떨림 억제
        _pad = np.pad(_tg, ((_k // 2, _k // 2), (0, 0)), mode="edge")
        _fw_tg = np.stack([_pad[i:i + _k].mean(axis=0) for i in range(F)])
        eye, centre = _fw_tg[0] + cam_dist * _cam_dir, _fw_tg[0]
        print(f"  follow_obj: 주시점 이동 {np.linalg.norm(_fw_tg[-1] - _fw_tg[0]):.2f} m, "
              f"카메라 거리 {cam_dist:.2f} m")

    cam_pose = look_at(eye, centre)
    _cam_node = scene.add(pyrender.PerspectiveCamera(yfov=np.radians(45.0), znear=0.05, zfar=100.0),
                          pose=cam_pose)
    _key_node = scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0),
                          pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[0.85, 0.88, 1.0], intensity=1.6),
              pose=look_at(centre + np.array([-dist, dist * 0.4, dist * 0.8]), centre))

    W, H = size
    r = pyrender.OffscreenRenderer(W, H)
    out_path = out_path or (clip_dir / "clip_viz.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out_path), fps=fps, macro_block_size=1) as w:
        for f in range(F):
            body_node.mesh = pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(vertices=verts[f], faces=faces, process=False),
                material=pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=C_BODY, metallicFactor=0.0, roughnessFactor=0.9), smooth=True)
            for k, node in act_nodes:
                scene.set_pose(node, pose_to_T(np.asarray(d[k][f], float)))
            if _fw_tg is not None:                                  # [follow-obj]
                _T = look_at(_fw_tg[f] + cam_dist * _cam_dir, _fw_tg[f])
                scene.set_pose(_cam_node, _T)
                scene.set_pose(_key_node, _T)
            color, _ = r.render(scene)
            w.append_data(np.asarray(color)[..., :3])
            if (f + 1) % 50 == 0:
                print(f"    {f + 1}/{F}", flush=True)
    r.delete()
    return str(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append", default=[], help="clip name (repeatable)")
    ap.add_argument("--class", dest="klass", default="single_rigid")
    ap.add_argument("--all", action="store_true", help="every clip in --class")
    ap.add_argument("--out", type=str, default="", help="output mp4 (single clip only)")
    ap.add_argument("--fps", type=int, default=30, help="matches the source 30 fps reference rate")
    ap.add_argument("--size", type=str, default="1280x720")
    ap.add_argument("--yaw", type=float, default=None,
                    help="camera azimuth (deg); default = auto, placed on the object's side of the body")
    ap.add_argument("--elev", type=float, default=22.0, help="camera elevation (deg)")
    ap.add_argument("--zoom", type=float, default=1.0, help="<1 = closer")
    ap.add_argument("--eye", type=float, nargs=3, default=None, help="카메라 위치 직접 지정 (롤아웃과 맞출 때)")
    ap.add_argument("--lookat", type=float, nargs=3, default=None, help="주시점 직접 지정")
    ap.add_argument("--follow_obj", action="store_true",
                    help="[follow-obj] 매 프레임 물체+손끝 중점을 추종 (확대 관찰용)")
    ap.add_argument("--cam_dist", type=float, default=0.9,
                    help="--follow_obj 일 때 주시점까지의 거리 (m). 작을수록 확대")
    ap.add_argument("--env_context", action="store_true",
                    help="env 가 실제로 스폰하는 context 물체만 (롤아웃 장면과 일치)")
    ap.add_argument("--ctx_radius", type=float, default=1.0)
    ap.add_argument("--ctx_support_radius", type=float, default=1.5)
    a = ap.parse_args()

    clips = a.clip
    if a.all:
        clips = sorted(p.parent.parent.name for p in (_PROC / a.klass).glob("*/0/trajectory.npz"))
    if not clips:
        print("[error] give --clip <name> (repeatable) or --all")
        return 1
    W, H = (int(v) for v in a.size.lower().split("x"))

    n_ok = 0
    for i, c in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {a.klass}/{c}")
        try:
            out = Path(a.out) if (a.out and len(clips) == 1) else None
            p = render_clip(c, a.klass, out, (W, H), a.fps, a.yaw, a.elev, a.zoom, a.eye, a.lookat,
                            a.env_context, a.ctx_radius, a.ctx_support_radius,
                            a.follow_obj, a.cam_dist)
            print(f"  -> {p}")
            n_ok += 1
        except Exception as exc:                       # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}: {exc}")
    print(f"\n=== {n_ok}/{len(clips)} rendered ===")
    return 0 if n_ok == len(clips) else 1


if __name__ == "__main__":
    raise SystemExit(main())
