"""Teleport-playback of the floating-hand RSI state cache, rendered to mp4.

Why this exists
---------------
`hand_state_cache.npz` (written by train.py at the end of a HandPretrain run) holds, per reference
frame, the highest-reward PHYSICALLY-VISITED hand state — best-of-thousands rather than best-of-N
rollouts. That makes it a candidate stage-2 tracking reference. But its rows come from DIFFERENT
episodes, so consecutive frames are not dynamically consistent: the hand can jump. Measured on
s100_seg00_pan, adjacent-frame wrist jumps were median 8-10 mm against 4.7-4.9 mm of motion implied
by the stored velocity — roughly 2x, with a 10 cm worst case.

Numbers do not tell you whether that is usable. This renders the cache the way a reference would be
consumed — TELEPORT, no controller, no physics integration — so the stitching is visible directly.

Structure mirrors `render_retarget.py` (same camera framing, ctx-object selection, imageio writer);
only the asset and the data source differ: two floating Shadow hands driven from the 176-D cache
instead of the composite G1 driven from the retarget npz.

What is teleported per frame
    wrist (= palm, articulation root) pose x2   cache[1:8], [14:21]
    actuated finger joints 36                  cache[40:76]     (left 18 -> right 18)
    tendon-axis J0 8                           cache[112:120]
    object pose                                cache[27:34]
Velocities are written as ZERO: a teleport has no dynamics, and writing the cached velocities would
make PhysX integrate them between renders, which is not what a reference playback shows.

Frames with valid=False (never cached) hold the LAST valid state rather than snapping to an
uninitialised row — those rows carry reward -inf and garbage pose.

Usage
    <env_isaaclab python> scripts/process_dataset/diagnostics/render_hand_cache.py --clip s101_seg12_knife
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="s101_seg12_knife")
parser.add_argument("--class", dest="cls", default="single_rigid")
parser.add_argument("--res", type=int, nargs=2, default=[1280, 720])
parser.add_argument("--fps", type=int, default=50, help="cache is at control_fps (50), not 30")
parser.add_argument("--out", default="")
parser.add_argument("--no_ctx", action="store_true", help="맥락 물체(ctx__*)를 그리지 않음")
parser.add_argument("--cam_yaw", type=float, default=45.0)
parser.add_argument("--cam_elev", type=float, default=None)
parser.add_argument("--cam_dist_scale", type=float, default=1.0)
parser.add_argument("--cam_min_dist", type=float, default=1.0)
parser.add_argument("--lookat_z", type=float, default=None)
parser.add_argument("--follow_obj", action="store_true",
                    help="프레임별로 물체를 시선 중심에 유지 — 손 클로즈업에 필요")
args = parser.parse_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app

import math  # noqa: E402
import os  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from robotis_sh5.tasks.direct.g1_shadow_hand_pretrain.shadow_float_cfg import (  # noqa: E402
    SHADOW_FLOAT_L_CFG, SHADOW_FLOAT_R_CFG)

_PROC = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/processed/parahome"
# 176 레이아웃 (train.py 가 저장한 layout 설명과 동일)
_SL = dict(wl_pose=(1, 8), wr_pose=(14, 21), obj_pose=(27, 34), jpos=(40, 76), j0=(112, 120))


def main():
    cdir = os.path.join(_PROC, "g1_shadow", args.cls, args.clip, "0")
    cpath = os.path.join(cdir, "hand_state_cache.npz")
    if not os.path.exists(cpath):
        raise FileNotFoundError(f"{cpath} 없음 — HandPretrain 학습을 먼저 완주시키세요")
    ch = np.load(cpath, allow_pickle=True)
    sc = ch["state_cache"].astype(np.float32)
    valid = ch["valid"].astype(bool)
    F = sc.shape[0]
    print(f"[render-cache] {args.clip}  {sc.shape}  valid {int(valid.sum())}/{F} "
          f"({float(ch['coverage'])*100:.1f}%)  fps={float(ch['control_fps'])}")

    # 미기록 프레임은 마지막 유효 상태를 유지 (reward -inf 인 미초기화 행을 그리지 않도록)
    fill = np.arange(F)
    last = -1
    for f in range(F):
        if valid[f]:
            last = f
        fill[f] = last if last >= 0 else int(np.argmax(valid))
    n_held = int((fill != np.arange(F)).sum())
    if n_held:
        print(f"[render-cache] 미기록 {n_held} 프레임은 직전 유효 상태를 유지합니다")

    sm = np.load(os.path.join(_PROC, "smplx", args.cls, args.clip, "0", "trajectory.npz"),
                 allow_pickle=True)
    base_keys = [k for k in sm.files if k.startswith("obj__") and k.endswith("__base")]
    obj_name = base_keys[0].split("__")[1] if base_keys else ""

    # ── ctx 선별: render_retarget.py / env 와 동일 규칙 ──────────────────────────────────
    _CTX_R, _CTX_SUP_R = 1.0, 1.5
    ctx_items = []
    obj_base = sm[f"obj__{obj_name}__base"].astype(np.float32) if obj_name else None
    if not args.no_ctx and obj_base is not None:
        _act_xy, _act0 = obj_base[:, :2], obj_base[0]
        _cands = []
        for k in (kk for kk in sm.files if kk.startswith("ctx__") and kk.endswith("__base")):
            _p0 = sm[k][0].astype(np.float32)
            _cands.append((k.split("__")[1], _p0,
                           float(np.linalg.norm(_act_xy - _p0[None, :2], axis=1).min())))
        _keep = {n for n, p, dm in _cands if dm < _CTX_R}
        _below = [(float(np.linalg.norm(_act0[:2] - p[:2])), n) for n, p, dm in _cands
                  if p[2] < _act0[2] and float(np.linalg.norm(_act0[:2] - p[:2])) < _CTX_SUP_R]
        if _below:
            _keep.add(min(_below)[1])
        ctx_items = [(n, p[None, :]) for n, p, dm in _cands if n in _keep]
        print(f"[render-cache] ctx {len(ctx_items)}/{len(_cands)}개: "
              f"{sorted(n for n, _ in ctx_items)}")

    W, H = args.res
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 50.0, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light",
                                                  sim_utils.DomeLightCfg(intensity=2500.0))

    hands = [Articulation(SHADOW_FLOAT_L_CFG.replace(prim_path="/World/HandL")),
             Articulation(SHADOW_FLOAT_R_CFG.replace(prim_path="/World/HandR"))]

    obj = None
    if obj_name:
        usd = os.path.join(_PROC, "assets", "objects", obj_name, f"{obj_name}.usd")
        if os.path.exists(usd):
            obj = RigidObject(RigidObjectCfg(
                prim_path="/World/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,
                                                                 disable_gravity=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.0),
                                                                roughness=0.6)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(float(v) for v in sc[fill[0], _SL["obj_pose"][0]:_SL["obj_pose"][0] + 3]),
                    rot=tuple(float(v) for v in sc[fill[0], _SL["obj_pose"][0] + 3:_SL["obj_pose"][1]]))))

    for _i, (_cn, _cb) in enumerate(ctx_items):
        _b = os.path.join(_PROC, "assets", "objects", _cn)
        for _cand in (os.path.join(_b, "ctx", f"{_cn}_ctx.usd"),
                      os.path.join(_b, f"{_cn}_ctx.usd"), os.path.join(_b, f"{_cn}.usd")):
            if os.path.exists(_cand):
                _cfg = sim_utils.UsdFileCfg(
                    usd_path=_cand, activate_contact_sensors=False,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,
                                                                 disable_gravity=True))
                _cfg.func(f"/World/Ctx_{_i}_{_cn}", _cfg,
                          translation=tuple(float(v) for v in _cb[0, :3]),
                          orientation=tuple(float(v) for v in _cb[0, 3:7]))
                break

    cam = Camera(CameraCfg(prim_path="/World/Camera", height=H, width=W, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=20.0,
                                                            clipping_range=(0.05, 50.0))))
    sim.reset()

    # 관절 인덱스: 구동 18개(손별) + 텐던 축 J0 4개(손별). 캐시의 36열은 왼손 18 → 오른손 18.
    _FEXPR = ["robot0_{s}_(FF|MF|RF|LF|TH)J[1-3]", "robot0_{s}_LFJ4",
              "robot0_{s}_THJ4", "robot0_{s}_THJ0"]
    fid, j0id = [], []
    for h, sd in enumerate("lr"):
        _ids, _ = hands[h].find_joints([e.format(s=sd) for e in _FEXPR])
        assert len(_ids) == 18, f"{sd} 구동관절 {len(_ids)} != 18"
        fid.append(torch.tensor(_ids, dtype=torch.long, device=sim.device))
        _jn = hands[h].data.joint_names
        j0id.append(torch.tensor([_jn.index(f"robot0_{sd}_{f}J0") for f in ("FF", "MF", "RF", "LF")],
                                 dtype=torch.long, device=sim.device))
    default_q = [h.data.default_joint_pos.clone() for h in hands]

    # ── 카메라: render_retarget.py 와 같은 방식 (수직 범위에 맞춰 뒤로 빼기) ──────────────
    wl = sc[valid, _SL["wl_pose"][0]:_SL["wl_pose"][0] + 3]
    wr = sc[valid, _SL["wr_pose"][0]:_SL["wr_pose"][0] + 3]
    allp = np.concatenate([wl, wr], axis=0)
    z_top = float(allp[:, 2].max()) + 0.25
    if obj_base is not None:
        z_top = max(z_top, float(obj_base[:, 2].max()) + 0.25)
    z_bot = max(0.0, float(allp[:, 2].min()) - 0.25)
    extent = max(0.3, z_top - z_bot)
    lookat_z = 0.5 * (z_top + z_bot)
    off = max(args.cam_min_dist, extent * 1.25) * args.cam_dist_scale
    horiz = off * (2 ** 0.5)
    az = math.radians(args.cam_yaw)
    zoff = (0.12 * extent) if args.cam_elev is None else horiz * math.tan(math.radians(args.cam_elev))
    txy = (obj_base[:, :2].mean(0).astype(np.float32) if obj_base is not None
           else allp[:, :2].mean(0).astype(np.float32))
    if args.lookat_z is not None:
        lookat_z = float(args.lookat_z)
    eye = torch.tensor([[float(txy[0]) + horiz * math.cos(az),
                         float(txy[1]) + horiz * math.sin(az), lookat_z + zoff]],
                       device=sim.device, dtype=torch.float32)
    tgt = torch.tensor([[float(txy[0]), float(txy[1]), lookat_z]],
                       device=sim.device, dtype=torch.float32)
    cam.set_world_poses_from_view(eye, tgt)

    import imageio
    out = args.out or os.path.join(cdir, "hand_cache_playback.mp4")
    writer = imageio.get_writer(out, fps=args.fps, macro_block_size=1)
    dt = sim.get_physics_dt()
    zero6 = torch.zeros(1, 6, device=sim.device)
    for f in range(F):
        row = sc[fill[f]]
        for h in range(2):
            a, b = (_SL["wl_pose"] if h == 0 else _SL["wr_pose"])
            rp = torch.from_numpy(row[a:b]).to(sim.device).unsqueeze(0)      # pos+quat wxyz
            hands[h].write_root_pose_to_sim(rp)
            hands[h].write_root_velocity_to_sim(zero6)                       # teleport → 속도 0
            q = default_q[h].clone()
            _sl = slice(0, 18) if h == 0 else slice(18, 36)
            q[:, fid[h]] = torch.from_numpy(
                row[_SL["jpos"][0]:_SL["jpos"][1]][_sl]).to(sim.device).unsqueeze(0)
            _j0s = slice(0, 4) if h == 0 else slice(4, 8)
            q[:, j0id[h]] = torch.from_numpy(
                row[_SL["j0"][0]:_SL["j0"][1]][_j0s]).to(sim.device).unsqueeze(0)
            hands[h].write_joint_state_to_sim(q, torch.zeros_like(q))
            hands[h].write_data_to_sim()
        if obj is not None:
            op = torch.from_numpy(row[_SL["obj_pose"][0]:_SL["obj_pose"][1]]).to(sim.device).unsqueeze(0)
            obj.write_root_pose_to_sim(op)
            obj.write_data_to_sim()
        if args.follow_obj:
            _t = row[_SL["obj_pose"][0]:_SL["obj_pose"][0] + 3]
            _lz = float(args.lookat_z) if args.lookat_z is not None else float(_t[2])
            cam.set_world_poses_from_view(
                torch.tensor([[float(_t[0]) + horiz * math.cos(az),
                               float(_t[1]) + horiz * math.sin(az), _lz + zoff]],
                             device=sim.device, dtype=torch.float32),
                torch.tensor([[float(_t[0]), float(_t[1]), _lz]],
                             device=sim.device, dtype=torch.float32))
        sim.render()
        cam.update(dt)
        if f % 50 == 0:
            print(f"[render-cache] frame {f}/{F}", flush=True)
        writer.append_data(cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
    writer.close()
    print(f"[render-cache] wrote {out} ({F} frames @ {args.fps} fps)")
    os._exit(0)


if __name__ == "__main__":
    main()
