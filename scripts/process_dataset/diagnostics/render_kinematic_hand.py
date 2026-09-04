"""리타게팅 결과를 pyrender 로 렌더링합니다 (Isaac Sim 불필요, 손 클로즈업에 적합).

render_retarget.py 는 Isaac Sim 으로 실제 USD 를 띄우지만, 손을 확대하면 메시가 뭉개지고
GPU 디바이스/DISPLAY 의존성이 있습니다. 이 스크립트는 URDF 메시를 pyrender/EGL 로 직접
올려 손가락 마디까지 선명하게 나옵니다.

    /home/peunsu/anaconda3/envs/env_isaaclab/bin/python \
        scripts/process_dataset/diagnostics/render_kinematic_hand.py \
            --clip s101_seg12_knife --cls single_rigid --nframes 301 --fps 30 \
            --label "" --no_context --follow_link robot0_r_palm \
            --cam_dist 0.55 --cam_yaw2 0 --cam_elev2 25 \
            --retarget <경로>/trajectory_pyroki.npz --out out.mp4

부등식(tendon-ineq) 결과를 렌더할 때는 --urdf 로 mimic 없는 URDF 를 지정해야 풀린 J0 가
반영됩니다 (mimic URDF 는 J0 를 1.14184 x J1 로 재계산해 덮어씁니다):
    --urdf source/robotis_sh5/data/robots/G1/urdf_pyroki/g1_shadow_nomimic.urdf

주요 옵션 (이 파일에서 추가된 것):
    --urdf         URDF 직접 지정
    --follow       프레임마다 물체를 시선 중심으로 추적
    --follow_link  물체 대신 로봇 링크를 추적 (예 robot0_r_palm) — 손 클로즈업에 권장
    --cam_dist     시선 대상으로부터의 거리(m). >0 이면 viewer_zoom 무시
    --cam_yaw2 / --cam_elev2   방위/고도(도) 덮어쓰기
"""
import math
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_DEVICE_ID", "1")

import argparse  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402
import yourdfpy  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5"
SCAN = f"{ROOT}/data/raw/parahome/data/scan"

ap = argparse.ArgumentParser()
ap.add_argument("--clip", default="s101_seg12_knife")
ap.add_argument("--cls", default="single_rigid")
ap.add_argument("--out", required=True)
ap.add_argument("--w", type=int, default=1280)
ap.add_argument("--h", type=int, default=720)
ap.add_argument("--fps", type=int, default=50)
ap.add_argument("--nframes", type=int, default=501)
# env cfg 와 같은 이름 / 같은 기본값
ap.add_argument("--viewer_yaw", type=float, default=225.0)
ap.add_argument("--viewer_elev", type=float, default=22.0)
ap.add_argument("--viewer_zoom", type=float, default=0.55)
ap.add_argument("--viewer_look_obj", type=int, default=1)
ap.add_argument("--context_radius", type=float, default=1.0)
ap.add_argument("--context_support_radius", type=float, default=1.5)
ap.add_argument("--no_context", action="store_true")
ap.add_argument("--label", default="Kinematic reference (retargeted)")
ap.add_argument("--retarget", default="", help="trajectory_pyroki.npz 경로 직접 지정 (빈 값이면 기본 위치)")
ap.add_argument("--urdf", default="", help="URDF 직접 지정 (부등식 결과는 mimic 없는 URDF 가 필요)")
ap.add_argument("--follow", action="store_true", help="프레임마다 물체를 시선 중심으로 추적")
ap.add_argument("--follow_link", default="", help="물체 대신 이 로봇 링크를 시선 중심으로 추적 (예 robot0_r_palm)")
ap.add_argument("--cam_dist", type=float, default=-1.0, help="물체로부터의 카메라 거리(m). >0 이면 viewer_zoom 무시")
ap.add_argument("--cam_yaw2", type=float, default=None, help="방위각(도) 덮어쓰기")
ap.add_argument("--cam_elev2", type=float, default=None, help="고도(도) 덮어쓰기")
args = ap.parse_args()

SKIN = np.array([0.80, 0.81, 0.85, 1.0])
HAND = np.array([0.95, 0.95, 0.97, 1.0])
OBJ = np.array([1.0, 0.38, 0.0, 1.0])
CTX = np.array([0.62, 0.63, 0.66, 1.0])


def mat(c):
    return pyrender.MetallicRoughnessMaterial(baseColorFactor=c, metallicFactor=0.10,
                                              roughnessFactor=0.75, alphaMode="OPAQUE")


def look_at(eye, tgt, up=np.array([0.0, 0.0, 1.0])):
    f = eye - tgt
    f = f / max(np.linalg.norm(f), 1e-9)
    s = np.cross(up, f)
    n = np.linalg.norm(s)
    s = s / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(f, s)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = s, u, f, eye
    return T


def quat_R(wq):
    w, x, y, z = wq
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


urdf = yourdfpy.URDF.load(args.urdf or f"{ROOT}/data/robots/G1/urdf_pyroki/g1_shadow.urdf",
                          load_meshes=True, build_scene_graph=True, load_collision_meshes=False)
UD = set(urdf.actuated_joint_names)


def geom_of(link):
    out = []
    for v in urdf.link_map[link].visuals:
        g = v.geometry
        m = None
        if g.mesh is not None:
            m = trimesh.load(urdf._filename_handler(g.mesh.filename), force="mesh", process=False)
            if g.mesh.scale is not None:
                m = m.copy(); m.apply_scale(g.mesh.scale)
        elif g.box is not None:
            m = trimesh.creation.box(extents=g.box.size)
        elif g.cylinder is not None:
            m = trimesh.creation.cylinder(radius=g.cylinder.radius, height=g.cylinder.length)
        elif g.sphere is not None:
            m = trimesh.creation.icosphere(radius=g.sphere.radius)
        if m is None or len(m.faces) == 0:
            continue
        m = m.copy()
        if v.origin is not None:
            m.apply_transform(v.origin)
        out.append(m)
    return trimesh.util.concatenate(out) if out else None


GEOM = {ln: g for ln in urdf.link_map if (g := geom_of(ln)) is not None}

sm = np.load(f"{ROOT}/data/processed/parahome/smplx/{args.cls}/{args.clip}/0/trajectory.npz",
             allow_pickle=True)
bk = [k for k in sm.files if k.startswith("obj__") and k.endswith("__base")][0]
obj_name = bk.split("__")[1]
otraj = sm[bk].astype(np.float64)
omesh = trimesh.load(f"{SCAN}/{obj_name}/simplified/base.obj", force="mesh", process=False)
ftp = sm["fingertip_pad_pos"].astype(np.float64) if "fingertip_pad_pos" in sm.files else None

# ── context 물체: env(_ctx_spawn) 와 같은 규칙 ────────────────────────────────────
ctx = []
if not args.no_context:
    act_xy, act0 = otraj[:, :2], otraj[0]
    cands = []
    for k in (kk for kk in sm.files if kk.startswith("ctx__") and kk.endswith("__base")):
        p0 = sm[k][0].astype(np.float64)
        cands.append((k.split("__")[1], p0,
                      float(np.linalg.norm(act_xy - p0[None, :2], axis=1).min())))
    keep = {n for n, p, dm in cands if dm < args.context_radius}
    below = [(float(np.linalg.norm(act0[:2] - p[:2])), n) for n, p, dm in cands
             if p[2] < act0[2] and float(np.linalg.norm(act0[:2] - p[:2])) < args.context_support_radius]
    if below:
        keep.add(min(below)[1])
    for n, p0, dm in cands:
        if n not in keep:
            continue
        f = f"{SCAN}/{n}/simplified/base.obj"
        if os.path.exists(f):
            ctx.append((n, trimesh.load(f, force="mesh", process=False), p0))
    print(f"context 물체 {len(ctx)}개: {', '.join(n for n, _, _ in ctx)}")

_rt_path = args.retarget or (f"{ROOT}/data/processed/parahome/g1_shadow/{args.cls}/{args.clip}/0/"
                              "trajectory_pyroki.npz")
print(f"리타게팅 레퍼런스: {_rt_path}")
rt = np.load(_rt_path, allow_pickle=True)
q, root = rt["g1_joint_pos"], rt["g1_root_pose"]
NM = [str(x) for x in rt["joint_names"]]
Fr = len(q)
idx = np.clip((np.arange(args.nframes) * (Fr - 1) / (args.nframes - 1)).round().astype(int), 0, Fr - 1)
oidx = np.clip((np.arange(args.nframes) * (len(otraj) - 1) / (args.nframes - 1)).round().astype(int),
               0, len(otraj) - 1)

# ── 카메라: env.__init__ 과 같은 식으로 한 번 계산하고 고정 ────────────────────────
tops = [float(root[:, 2].max()) + 0.75]
if ftp is not None:
    tops.append(float(ftp[:, :, 2].max()))
if float(otraj[:, 2].max()) > 0.1:
    tops.append(float(otraj[:, 2].max()))
z_top = max(tops) + 0.15
extent, lookat_z = z_top, 0.5 * z_top
off = max(1.5, extent * 1.25) * args.viewer_zoom
look_obj = bool(args.viewer_look_obj) and float(otraj[:, 2].max()) > 0.1
tx, ty = ((float(otraj[:, 0].mean()), float(otraj[:, 1].mean())) if look_obj
          else (float(root[:, 0].mean()), float(root[:, 1].mean())))
horiz = off * (2 ** 0.5)
az = math.radians(args.viewer_yaw)
zoff = horiz * math.tan(math.radians(args.viewer_elev)) if args.viewer_elev > 0.0 else 0.12 * extent
if args.cam_dist > 0:                      # 손 클로즈업: 물체 중심에서 지정 거리
    _yaw = math.radians(args.cam_yaw2 if args.cam_yaw2 is not None else args.viewer_yaw)
    _el = math.radians(args.cam_elev2 if args.cam_elev2 is not None else args.viewer_elev)
    tx, ty = float(otraj[:, 0].mean()), float(otraj[:, 1].mean())
    lookat_z = float(otraj[:, 2].mean())
    horiz = args.cam_dist * math.cos(_el); zoff = args.cam_dist * math.sin(_el)
    az = _yaw
LOOKAT = np.array([tx, ty, lookat_z])
EYE = np.array([tx + horiz * math.cos(az), ty + horiz * math.sin(az), lookat_z + zoff])
print(f"고정 카메라  eye={np.round(EYE,3).tolist()}  lookat={np.round(LOOKAT,3).tolist()}")

sc = pyrender.Scene(bg_color=[0.10, 0.11, 0.13, 1.0], ambient_light=[0.45, 0.45, 0.48])
nodes = {ln: sc.add(pyrender.Mesh.from_trimesh(
    m, material=mat(HAND if ln.startswith("robot0_") else SKIN), smooth=False))
    for ln, m in GEOM.items()}
for n, m, p0 in ctx:                                   # 정적 — 프레임 0 자세로 한 번만
    T = np.eye(4); T[:3, :3] = quat_R(p0[3:7]); T[:3, 3] = p0[:3]
    sc.add(pyrender.Mesh.from_trimesh(m, material=mat(CTX), smooth=False), pose=T)
on = sc.add(pyrender.Mesh.from_trimesh(omesh, material=mat(OBJ), smooth=False))
cam = sc.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.5), pose=look_at(EYE, LOOKAT))
sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.2),
       pose=look_at(LOOKAT + np.array([0.8, 0.4, 2.2]), LOOKAT))
sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
       pose=look_at(EYE + np.array([0.0, 0.0, 0.6]), LOOKAT))
r = pyrender.OffscreenRenderer(args.w, args.h)

wr = imageio.get_writer(args.out, fps=args.fps, quality=8, macro_block_size=1)
for i in range(args.nframes):
    k, ko = idx[i], oidx[i]
    urdf.update_cfg({n: float(q[k, j]) for j, n in enumerate(NM) if n in UD})
    R = quat_R(root[k, 3:7])
    W = np.eye(4); W[:3, :3] = R; W[:3, 3] = root[k, :3]
    for ln, nd in nodes.items():
        sc.set_pose(nd, W @ urdf.get_transform(ln))
    O = np.eye(4); O[:3, :3] = quat_R(otraj[ko, 3:7]); O[:3, 3] = otraj[ko, :3]
    sc.set_pose(on, O)
    if args.follow_link:                   # 프레임별 로봇 링크를 시선 중심으로 (손 클로즈업)
        _L = (W @ urdf.get_transform(args.follow_link))[:3, 3]
        _E = _L + (EYE - LOOKAT)
        sc.set_pose(cam, look_at(_E, _L))
    elif args.follow:                      # 프레임별 물체 위치로 카메라 평행이동
        _c = otraj[ko, :3]
        _L = np.array([_c[0], _c[1], _c[2]])
        _E = _L + (EYE - LOOKAT)
        sc.set_pose(cam, look_at(_E, _L))
    col, _ = r.render(sc)
    if args.label:                       # 빈 문자열이면 상단 라벨 없이 그대로
        im = Image.fromarray(col)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, args.w, 30], fill=(13, 14, 17))
        d.text((10, 8), args.label, fill=(235, 235, 240))
        col = np.asarray(im)
    wr.append_data(col)
    if i % 150 == 0:
        print(f"  {i}/{args.nframes}", flush=True)
wr.close(); r.delete()
print("저장:", args.out)
