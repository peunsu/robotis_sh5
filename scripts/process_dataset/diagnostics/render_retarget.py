"""Render a KINEMATIC playback of a retargeted clip: the actual G1+Shadow robot posed frame-by-frame
from the retarget npz (g1_joint_pos + g1_root_pose), with the manipulated object at its reference
pose, captured to mp4. Lets you VISUALLY check the retarget (leg poses / foot-flat / hands-on-object /
over-rotation) with the real robot mesh — no physics (pure kinematic set + render).

Run in the Isaac Sim env (needs cameras):
    python -u scripts/process_dataset/diagnostics/render_retarget.py --clip s100_seg00_pan [--class single_rigid]
→ writes data/processed/parahome/g1_shadow/<class>/<clip>/0/retarget_playback.mp4
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="s100_seg00_pan")
parser.add_argument("--class", dest="cls", default="single_rigid")
parser.add_argument("--res", type=int, nargs=2, default=[1280, 720])
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--out", default="")
parser.add_argument("--variant", default="", help="'' → trajectory.npz (pink); 'pyroki' → trajectory_pyroki.npz")
parser.add_argument("--retarget-file", default="", help="explicit npz name under the clip dir (overrides --variant)")
parser.add_argument("--no_ctx", action="store_true", help="맥락 물체(ctx__*)를 그리지 않음")
parser.add_argument("--no-kpts", action="store_true", help="disable the human(green)/robot(cyan) keypoint overlay")
parser.add_argument("--cam_yaw", type=float, default=45.0, help="camera azimuth around the look target (deg; 45 = default +X+Y)")
parser.add_argument("--cam_elev", type=float, default=None, help="camera elevation (deg); default derives from vertical extent")
parser.add_argument("--cam_dist_scale", type=float, default=1.0, help="pull-back multiplier (>1 = further)")
parser.add_argument("--look_obj", action="store_true", help="aim at the object centroid (where the hands work) instead of the root")
parser.add_argument("--lookat_z", type=float, default=None, help="override the look-at height (m); default = body vertical centre")
parser.add_argument("--cam_min_dist", type=float, default=1.5, help="floor on the pull-back distance before cam_dist_scale")
parser.add_argument("--kpt_radius", type=float, default=0.018, help="human-keypoint marker sphere radius (m)")
parser.add_argument("--follow_obj", action="store_true", help="move the camera every frame to keep the object (hands) centred — lets you zoom much tighter than a static mean-position camera")
AppLauncher.add_app_launcher_args(parser)
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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from robotis_sh5.tasks.direct.g1_shadow_locomanip.g1_shadow_locomanip_env_cfg import G1_SHADOW_CFG  # noqa: E402
from robotis_sh5.tasks.direct.g1_shadow_sonic_residual.g1_shadow_sonic_residual_env_cfg import (  # noqa: E402
    BODY_KPTS, HAND_CHAIN, BODY_KPT_OFFSETS)

_PROC = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/processed/parahome"


def main():
    _rt_name = args.retarget_file or ("trajectory_pyroki.npz" if args.variant == "pyroki" else "trajectory.npz")
    rt = np.load(os.path.join(_PROC, "g1_shadow", args.cls, args.clip, "0", _rt_name))
    jpos = rt["g1_joint_pos"].astype(np.float32)      # (F,65) action-joint order
    root = rt["g1_root_pose"].astype(np.float32)      # (F,7) pos + quat wxyz
    F = jpos.shape[0]
    sm = np.load(os.path.join(_PROC, "smplx", args.cls, args.clip, "0", "trajectory.npz"), allow_pickle=True)
    base_keys = [k for k in sm.files if k.startswith("obj__") and k.endswith("__base")]
    obj_name = base_keys[0].split("__")[1] if base_keys else ""
    obj_base = sm[f"obj__{obj_name}__base"].astype(np.float32) if obj_name else None
    # [ctx-render 2026-09-04] 맥락 물체(ctx__*__base): env 가 kinematic 으로 스폰하는 고정 장면
    # 물체들. 리타게팅 문제에는 들어가지 않으므로(scene_collision 제거) 손이 통과할 수 있는데,
    # 렌더링에 없으면 그게 보이지 않습니다. --no_ctx 로 끕니다.
    # env(g1_shadow_sonic_residual_env.py:247-269)의 선별 규칙과 동일합니다:
    #   1) 각 ctx 물체의 프레임-0 중심이 조작 물체의 SWEPT xy 경로에서 context_radius(1.0 m) 안
    #   2) 조작 물체보다 아래에 있고 context_support_radius(1.5 m) 안인 가장 가까운 것 하나는 항상 포함
    #      (식탁/카운터처럼 바닥면이 넓으면 중심이 1.0 m 밖으로 나갈 수 있어 지지면 안전망)
    #   정적이므로 프레임-0 자세만 씁니다.
    _CTX_R, _CTX_SUP_R = 1.0, 1.5
    ctx_items = []
    if not args.no_ctx and obj_base is not None:
        _act_xy = obj_base[:, :2]                                        # (F,2) 조작 물체 경로
        _act0 = obj_base[0]
        _cands = []
        for k in (kk for kk in sm.files if kk.startswith("ctx__") and kk.endswith("__base")):
            _p0 = sm[k][0].astype(np.float32)
            _dmin = float(np.linalg.norm(_act_xy - _p0[None, :2], axis=1).min())
            _cands.append((k.split("__")[1], _p0, _dmin))
        _keep = {n for n, p, dm in _cands if dm < _CTX_R}
        _below = [(float(np.linalg.norm(_act0[:2] - p[:2])), n) for n, p, dm in _cands
                  if p[2] < _act0[2] and float(np.linalg.norm(_act0[:2] - p[:2])) < _CTX_SUP_R]
        if _below:
            _keep.add(min(_below)[1])
        ctx_items = [(n, p[None, :]) for n, p, dm in _cands if n in _keep]
        print(f"[render] ctx 선별 {len(ctx_items)}/{len(_cands)}개 (radius {_CTX_R} m): "
              f"{sorted(n for n, _ in ctx_items)}")
    # [ctx-render] GREEN 오버레이를 SMPL-X 로 전환. 예전에는 sm["joint_positions"](ParaHome 73관절,
    # 손 오프셋 23/48)을 썼는데 리타게팅이 SMPL-X 를 목표로 하므로 다른 데이터를 그리고 있었습니다.
    # env(g1_shadow_sonic_residual_env.py)의 적재 로직과 같은 구성입니다.
    jp_kp = None
    if not args.no_kpts:
        jp_kp = np.concatenate([sm["smplx_joints"].astype(np.float32),
                                sm["fingertip_pad_pos"].astype(np.float32)], axis=1)   # (F,65,3)
    _PAD_BASE = 55
    ref_idx = list(BODY_KPTS.keys())
    for _s, (_hb, _wr, _pb) in enumerate(((25, 20, _PAD_BASE), (40, 21, _PAD_BASE + 5))):
        for _spec in HAND_CHAIN.values():
            for _p in _spec["parahome"]:
                if _p >= 0:      ref_idx.append(_hb + _p)
                elif _p == -10:  ref_idx.append(_wr)
                else:            ref_idx.append(_pb + (-_p - 1))
    import json
    order = json.load(open("/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/robots/G1/g1_shadow_joint_order.json"))
    act_names = order["action_joint_names"]
    print(f"[render] clip={args.clip} F={F} obj={obj_name}")

    W, H = args.res
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 30.0, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    robot = Articulation(G1_SHADOW_CFG.replace(prim_path="/World/Robot"))

    obj = None
    if obj_name:
        usd = os.path.join(_PROC, "assets", "objects", obj_name, f"{obj_name}.usd")
        if os.path.exists(usd):
            obj = RigidObject(RigidObjectCfg(
                prim_path="/World/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.0), roughness=0.6)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(float(v) for v in obj_base[0, :3]),
                    rot=tuple(float(v) for v in obj_base[0, 3:7]))))

    # [ctx-render] 맥락 물체는 정적 프림으로 스폰합니다 (env 와 동일: 프레임-0 자세 고정).
    # USD 경로 우선순위도 env(g1_shadow_sonic_residual_env.py:612-615)와 같습니다 —
    #   ctx/<name>_ctx.usd  →  <name>_ctx.usd  →  <name>.usd
    # 관절형 가구(sink/refrigerator/microwave/gasstove/washingmachine)는 <name>.usd 가 살아 있는
    # articulation 이라 정적 충돌용 _ctx.usd 가 따로 만들어져 있습니다. 그것만 찾으면 스폰됩니다.
    _ctx_n = 0
    for _i, (_cn, _cb) in enumerate(ctx_items):
        _b = os.path.join(_PROC, "assets", "objects", _cn)
        _usd = os.path.join(_b, "ctx", f"{_cn}_ctx.usd")
        if not os.path.exists(_usd):
            _usd = os.path.join(_b, f"{_cn}_ctx.usd")
        if not os.path.exists(_usd):
            _usd = os.path.join(_b, f"{_cn}.usd")
        if not os.path.exists(_usd):
            print(f"[render] ctx 물체 USD 없음, 건너뜀: {_cn}")
            continue
        _cfg = sim_utils.UsdFileCfg(
            usd_path=_usd, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True))
        _cfg.func(f"/World/Ctx_{_i}_{_cn}", _cfg,
                  translation=tuple(float(v) for v in _cb[0, :3]),
                  orientation=tuple(float(v) for v in _cb[0, 3:7]))
        _ctx_n += 1
    if ctx_items:
        print(f"[render] ctx 물체 {_ctx_n}/{len(ctx_items)}개 스폰")

    cam = Camera(CameraCfg(
        prim_path="/World/Camera", height=H, width=W, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.05, 50.0))))

    sim.reset()

    # action-joint name → articulation joint index
    jn = robot.joint_names
    # [ROLLBACK MARKER: tendon-ineq] npz 가 자기 joint_names 를 들고 있으면 그걸 따릅니다.
    # 부등식 리타게팅은 J0 8개를 자유 변수로 풀어 65 → 73 열로 저장하므로, 고정 65 목록으로
    # 매핑하면 폭이 안 맞고 풀린 J0 도 버려집니다. 이름 기반이면 두 경우 모두 처리됩니다.
    try:
        _npz_names = [str(x) for x in rt["joint_names"]] if "joint_names" in rt.files else None
    except ValueError:                       # object 배열 → allow_pickle 필요
        _npz_names = [str(x) for x in np.load(os.path.join(
            _PROC, "g1_shadow", args.cls, args.clip, "0", _rt_name),
            allow_pickle=True)["joint_names"]]
    _use = _npz_names if (_npz_names and len(_npz_names) == jpos.shape[1]) else act_names
    if _use is not act_names:
        print(f"[render] npz joint_names 사용 ({len(_use)}개, J0 "
              f"{sum(1 for n in _use if n.endswith('J0') and 'TH' not in n)}개 포함)")
    aid = torch.tensor([jn.index(n) for n in _use], dtype=torch.long, device=sim.device)
    default_q = robot.data.default_joint_pos.clone()

    # keypoint overlay: GREEN = human reference (54). Robot pose is shown by its mesh (no cyan → no
    # per-frame FK refresh needed, keeps the render loop the same as the proven original).
    m_human = None
    if not args.no_kpts:
        m_human = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/human", markers={
            "s": sim_utils.SphereCfg(radius=float(args.kpt_radius),
                                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))}))
        human_k = torch.from_numpy(jp_kp[:, ref_idx, :]).to(sim.device)   # (F,54,3) world

    # env-fixed camera framing the robot-root centroid
    c = root[:, :3].mean(0).astype(np.float32)
    # CLIP-ADAPTIVE vertical framing: fit ground(feet)→highest point of the motion so a lifted object
    # or raised hands never leave the top of the (env-fixed) frame while the feet stay in view.
    tops = [float(root[:, 2].max()) + 0.75]                       # head ≈ root + 0.75
    if obj_base is not None:
        tops.append(float(obj_base[:, 2].max()))                 # highest object position
    if "fingertip_pad_pos" in sm.files:
        tops.append(float(sm["fingertip_pad_pos"][:, :, 2].max()))  # highest hand position
    z_top = max(tops) + 0.15
    z_bot = 0.0
    extent = z_top - z_bot
    lookat_z = 0.5 * (z_top + z_bot)
    off = max(args.cam_min_dist, extent * 1.25) * args.cam_dist_scale   # pull back to fit the vertical extent
    horiz = off * (2 ** 0.5)                                     # horizontal cam distance (yaw=45 → default +X+Y)
    az = math.radians(args.cam_yaw)
    zoff = (0.12 * extent) if args.cam_elev is None else horiz * math.tan(math.radians(args.cam_elev))
    # look target: object centroid (where the hands work → un-occludes hands hidden behind the torso) or root
    txy = obj_base[:, :2].mean(0).astype(np.float32) if (args.look_obj and obj_base is not None) else c[:2]
    eye = torch.tensor([[float(txy[0]) + horiz * math.cos(az), float(txy[1]) + horiz * math.sin(az),
                         lookat_z + zoff]], device=sim.device, dtype=torch.float32)
    if args.lookat_z is not None:                      # 손 클로즈업: 시선 높이를 직접 지정
        lookat_z = float(args.lookat_z)
        eye[0, 2] = lookat_z + zoff
    tgt = torch.tensor([[float(txy[0]), float(txy[1]), lookat_z]], device=sim.device, dtype=torch.float32)
    cam.set_world_poses_from_view(eye, tgt)

    import imageio
    _vid = f"retarget_playback_{args.variant}.mp4" if args.variant else "retarget_playback.mp4"
    out = args.out or os.path.join(_PROC, "g1_shadow", args.cls, args.clip, "0", _vid)
    writer = imageio.get_writer(out, fps=args.fps, macro_block_size=1)
    dt = sim.get_physics_dt()
    for f in range(F):
        q = default_q.clone()
        q[:, aid] = torch.from_numpy(jpos[f]).to(sim.device).unsqueeze(0)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        rp = torch.from_numpy(root[f]).to(sim.device).unsqueeze(0)   # pos+quat wxyz
        robot.write_root_pose_to_sim(rp)
        robot.write_data_to_sim()
        if obj is not None and obj_base is not None:
            op = torch.from_numpy(obj_base[f]).to(sim.device).unsqueeze(0)
            obj.write_root_pose_to_sim(op)
            obj.write_data_to_sim()
        if m_human is not None:
            m_human.visualize(translations=human_k[f])                   # GREEN human keypoints
        if args.follow_obj and obj_base is not None:
            # 정적 평균 위치 카메라로는 타이트한 클로즈업에서 손이 프레임을 벗어난다.
            # 프레임별 물체 위치를 시선 중심으로 잡아 따라간다 (거리/방위/고도는 그대로).
            _t = obj_base[f, :3].astype(np.float32)
            _lz = float(args.lookat_z) if args.lookat_z is not None else float(_t[2])
            _e = torch.tensor([[float(_t[0]) + horiz * math.cos(az),
                                float(_t[1]) + horiz * math.sin(az), _lz + zoff]],
                              device=sim.device, dtype=torch.float32)
            _g = torch.tensor([[float(_t[0]), float(_t[1]), _lz]],
                              device=sim.device, dtype=torch.float32)
            cam.set_world_poses_from_view(_e, _g)
        sim.render()
        cam.update(dt)
        if f % 30 == 0:
            print(f"[render] frame {f}/{F}", flush=True)
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        writer.append_data(rgb)
    writer.close()
    print(f"[render] wrote {out} ({F} frames @ {args.fps} fps)")
    os._exit(0)


if __name__ == "__main__":
    main()
