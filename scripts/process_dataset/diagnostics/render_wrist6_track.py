"""Render the `wrist6` 6-DoF wrist chain tracking the retargeted reference, to mp4.

Both floating hands are spawned from `shadow_float6_{l,r}.usd` (root = `robot0_{s}_anchor`, fixed at
the world origin; the tx/ty/tz/rot1/rot2/rot3 joints carry the hand to its pose). The wrist joints
are driven by the joint PD controller toward `wrist_dof6.npz`, fingers are held at the retargeted
pose, and the manipulated object + context objects are teleported along the reference so the motion
is legible in context.

Two things this does that `tune_wrist6_gains.py` did not:
  * the scene is built ONCE per gain setting, so the gains actually reach PhysX. Mutating
    `actuator.stiffness` in place only changes Isaac Lab's Python-side `applied_torque` estimate;
    an implicit actuator's real gains live in PhysX and are written at spawn. That is why the
    earlier 36-setting sweep reported byte-identical tracking error for every setting.
  * a VELOCITY feedforward (`set_joint_velocity_target`) from the reference's finite difference.
    With the default zero velocity target, damping brakes the intended motion: the reference asks
    for 6.1 rad/s at p99, so kd_r=20 alone would demand 120 N*m of pure braking torque.

GREEN spheres mark the reference palm position (the forward map of the target 6-DoF, verified in sim
to 0.0005 mm), so the gap between marker and mesh IS the tracking error.

    <env_isaaclab python> scripts/process_dataset/diagnostics/render_wrist6_track.py \
        --clip s101_seg12_knife
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="s101_seg12_knife")
parser.add_argument("--class", dest="cls", default="single_rigid")
parser.add_argument("--res", type=int, nargs=2, default=[1280, 720])
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--out", default="")
parser.add_argument("--kp_t", type=float, default=2000.0)
parser.add_argument("--kd_t", type=float, default=200.0)
parser.add_argument("--kp_r", type=float, default=200.0)
parser.add_argument("--kd_r", type=float, default=20.0)
parser.add_argument("--eff_t", type=float, default=200.0)
parser.add_argument("--eff_r", type=float, default=50.0)
parser.add_argument("--vlim_r", type=float, default=20.0, help="회전 관절 속도 한계 (rad/s)")
parser.add_argument("--vlim_t", type=float, default=5.0, help="병진 관절 속도 한계 (m/s)")
parser.add_argument("--armature", type=float, default=0.0,
                    help="손목 관절 armature. 더미 링크(0.01kg)가 2kg 손을 지탱하는 200:1 질량비로\n                         articulation 질량행렬이 ill-conditioned 해지는 것을 정칙화한다")
parser.add_argument("--no_vff", action="store_true", help="속도 피드포워드를 끈다 (비교용)")
parser.add_argument("--no_video", action="store_true", help="오차만 측정하고 렌더는 건너뛴다")
parser.add_argument("--no_ctx", action="store_true")
parser.add_argument("--max_depen", type=float, default=0.1,
                    help="max_depenetration_velocity. 실제 env(shadow_float_cfg.py)와 같은 0.1 이 기본.\n                         이전 기본값 1000.0 은 작은 관통을 1000 m/s 사출로 증폭시켜 NaN 을 만들었다")
parser.add_argument("--vel_iters", type=int, default=4,
                    help="solver_velocity_iteration_count. env 와 같은 4 가 기본 (이전 0)")
parser.add_argument("--no_self_collide", action="store_true",
                    help="손 내부 자기충돌을 끈다. 물체/ctx 에 콜라이더가 없고 손이 하나면\n                         남는 접촉은 손가락 자기충돌뿐이다")
parser.add_argument("--only", default="", choices=("", "l", "r"),
                    help="한 손만 스폰한다. 같은 스크립트에서 단손/양손을 이분하기 위한 것")
parser.add_argument("--fing_kd", type=float, default=0.2,
                    help="손가락 damping. 손가락 링크 관성이 1e-5 kg*m^2 이므로 임계값은\n                         2*sqrt(1.0*1e-5)=0.0063 이고 기본 0.2 는 32배 과감쇠다.\n                         kd*dt/I = 100 (>1 이면 수치 불안정)")
parser.add_argument("--fing_armature", type=float, default=0.0,
                    help="손가락 관절 armature. 관성을 키워 kd*dt/I 를 낮춘다")
parser.add_argument("--phys_hz", type=float, default=200.0,
                    help="물리 주기(Hz). 제어는 항상 50Hz 로 유지하고 decimation 을 맞춰 늘린다.\n                         수치 불안정이면 이걸 올리면 사라진다 — 물리적 원인이면 안 변한다")
parser.add_argument("--substep_dump", default="",
                    help="물리 서브스텝(200Hz)마다 손목 관절 pos/vel 을 npz 로 저장. 제어 주기\n                         (50Hz)에서만 읽으면 물리 주기 진동이 에일리어싱되어 안 보인다")
parser.add_argument("--substep_range", type=int, nargs=2, default=[50, 80],
                    help="저장할 제어 프레임 구간")
parser.add_argument("--ref_lpf", type=int, default=0,
                    help="레퍼런스 6-DoF 를 이동평균으로 저역통과(창 크기, 프레임). 요구 속도만\n                         줄이고 나머지는 그대로 두어 원인을 가른다")
parser.add_argument("--ref_from", default="", choices=("", "l", "r"),
                    help="다른 쪽 손의 레퍼런스를 쓴다. 에셋 문제인지 레퍼런스 문제인지 가른다")
parser.add_argument("--res_t", type=float, default=0.0,
                    help="손목 병진 잔차 스케일 [m]. 정책 잔차를 무작위로 흉내내 힘 증폭을 본다.\n                         DexMachina 는 0.04 m 를 쓴다. 잔차는 위치 목표에만 더한다")
parser.add_argument("--res_r", type=float, default=0.0,
                    help="손목 회전 잔차 스케일 [rad]. DexMachina 는 0.5 rad (28.6도)")
parser.add_argument("--res_ema", type=float, default=0.2,
                    help="잔차의 EMA 계수 (새 값 가중). 1.0 이면 백색잡음, 작을수록 시간 상관이 커져\n                         정책 출력에 가까워진다. env 의 wrist_force_ema 와 같은 0.2 가 기본")
parser.add_argument("--res_seed", type=int, default=0)
parser.add_argument("--dump_err", default="", help="프레임별 오차 곡선을 npz 로 저장")
parser.add_argument("--onset_mm", type=float, default=50.0,
                    help="발산 시작점 판정 임계. NaN 은 오버플로 시점이라 원인 시점이 아니다")
parser.add_argument("--no_hand_collide", action="store_true",
                    help="두 손 사이의 충돌만 필터링(각 손의 self-collision 과 지면은 유지).\n                         리타게팅 오차로 양손 레퍼런스가 겹치면 깊은 관통이 생기고, 어떤\n                         depenetration 설정으로도 안전해지지 않는다 — 관통을 막는 쪽이 맞다")
parser.add_argument("--no_obj", action="store_true",
                    help="조작 물체를 스폰하지 않는다. 콜라이더를 끄는 유일하게 확실한 방법 —\n                         Isaac Lab 의 collision_props 는 스폰 루트에 적용을 시도해 no-op 이고,\n                         스폰된 mesh 는 instance proxy 라 속성을 직접 쓸 수도 없다")
parser.add_argument("--no_collide", action="store_true",
                    help="물체/ctx 의 콜라이더를 끈다(시각은 유지). kinematic 물체가 손 안으로\n                         텔레포트되며 생기는 관통 → depenetration 폭주를 배제하는 대조군")
parser.add_argument("--no_markers", action="store_true")
parser.add_argument("--marker_radius", type=float, default=0.015)
parser.add_argument("--cam_yaw", type=float, default=45.0)
parser.add_argument("--cam_elev", type=float, default=None)
parser.add_argument("--cam_dist_scale", type=float, default=1.0)
parser.add_argument("--cam_min_dist", type=float, default=1.0)
parser.add_argument("--lookat_z", type=float, default=None)
args = parser.parse_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

_ROOT = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data"
_PROC = f"{_ROOT}/processed/parahome"
_ROT_SEQ = "YZX"   # build_shadow_floating_usd.py 의 _W6_ROT_SEQ 와 일치해야 한다
_W6 = ["tx", "ty", "tz", "rot1", "rot2", "rot3"]
_FEXPR = ["robot0_{s}_(FF|MF|RF|LF|TH)J[1-3]", "robot0_{s}_LFJ4",
          "robot0_{s}_THJ4", "robot0_{s}_THJ0"]
_J0F = ("FF", "MF", "RF", "LF")
DT = 1.0 / args.phys_hz
DEC = int(round(args.phys_hz / 50.0))   # 제어는 50 Hz 고정
REF_FPS = 30.0


def _resample(a, t_src, t_tgt):
    """(F,K) at t_src -> (len(t_tgt),K). Linear; the 6-DoF values are unwrapped so this is safe."""
    return np.stack([np.interp(t_tgt, t_src, a[:, k]) for k in range(a.shape[1])], 1).astype(np.float32)


def _euler_to_quat(a, b, c):
    """내재 회전 YZX -> quat wxyz. 규약은 export_wrist_dof6.py 의 _ROT_SEQ 와 같아야 한다."""
    from scipy.spatial.transform import Rotation as _R
    _q = _R.from_euler(_ROT_SEQ, np.stack([a, b, c], 1)).as_quat()       # xyzw
    return _q[:, [3, 0, 1, 2]]


def _quat_err(qa, qb):
    d = (qa * qb).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.arccos(d)


def _hand_cfg(side, prim):
    return ArticulationCfg(
        prim_path=prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_ROOT}/robots/G1/shadow_float6_{side}.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=args.max_depen),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=not args.no_self_collide, fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=args.vel_iters),
            fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.2)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.), rot=(1., 0., 0., 0.),
                                                   joint_pos={".*": 0.0}),
        actuators={
            "wrist_trans": ImplicitActuatorCfg(
                joint_names_expr=[f"robot0_{side}_wrist_t[xyz]"],
                stiffness=args.kp_t, damping=args.kd_t,
                effort_limit_sim=args.eff_t, velocity_limit_sim=args.vlim_t),
            "wrist_rot": ImplicitActuatorCfg(
                joint_names_expr=[f"robot0_{side}_wrist_rot[123]"],
                stiffness=args.kp_r, damping=args.kd_r,
                effort_limit_sim=args.eff_r, velocity_limit_sim=args.vlim_r),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[e.format(s=side) for e in _FEXPR],
                stiffness=1.0, damping=args.fing_kd,
                effort_limit_sim=3.09, velocity_limit_sim=15.0)},
        soft_joint_pos_limit_factor=0.9)


def main():
    cdir = f"{_PROC}/g1_shadow/{args.cls}/{args.clip}/0"
    d6 = np.load(f"{cdir}/wrist_dof6.npz", allow_pickle=True)
    rt = np.load(f"{cdir}/trajectory_pyroki.npz", allow_pickle=True)
    sm = np.load(f"{_PROC}/smplx/{args.cls}/{args.clip}/0/trajectory.npz", allow_pickle=True)

    F30 = len(d6["wrist_dof_l"])
    F = int(round((F30 - 1) * args.fps / REF_FPS)) + 1
    t_src, t_tgt = np.arange(F30) / REF_FPS, np.arange(F) / float(args.fps)
    _rsides = [args.only] if args.only else ["l", "r"]
    if args.ref_from:
        _rsides = [args.ref_from] * len(_rsides)
        print(f"[w6] 레퍼런스를 {args.ref_from}손 것으로 대체")
    ref = [_resample(d6[f"wrist_dof_{s}"].astype(np.float32), t_src, t_tgt) for s in _rsides]
    if args.ref_lpf > 1:
        _w = int(args.ref_lpf)
        _k = np.ones(_w, np.float64) / _w
        ref = [np.stack([np.convolve(np.pad(r[:, c], (_w // 2, _w - 1 - _w // 2), mode="edge"),
                                     _k, mode="valid") for c in range(6)], 1).astype(np.float32)
               for r in ref]
        print(f"[w6] 레퍼런스 저역통과 창={_w} 프레임")
    # 속도 목표는 레퍼런스의 차분. 이게 없으면 damping 이 의도된 운동을 제동한다.
    vff = [np.vstack([np.diff(r, axis=0) * args.fps, np.zeros((1, 6), np.float32)]) for r in ref]
    refq = [_euler_to_quat(r[:, 3], r[:, 4], r[:, 5]).astype(np.float32) for r in ref]
    for q in refq:
        q /= np.linalg.norm(q, axis=1, keepdims=True)

    jn_rt = [str(x) for x in rt["joint_names"]]
    obj_name = next((k.split("__")[1] for k in sm.files
                     if k.startswith("obj__") and k.endswith("__base")), "")
    obj_base = sm[f"obj__{obj_name}__base"].astype(np.float32) if obj_name else None
    print(f"[w6] {args.clip}  F={F} @{args.fps}fps  obj={obj_name or '-'}")
    print(f"[w6] gains kp_t={args.kp_t:.0f} kd_t={args.kd_t:.0f} "
          f"kp_r={args.kp_r:.0f} kd_r={args.kd_r:.0f}  vlim_r={args.vlim_r:.0f} "
          f"vff={'off' if args.no_vff else 'on'} "
          f"collide={'off' if args.no_collide else 'on'} "
          f"max_depen={args.max_depen} vel_iters={args.vel_iters} "
          f"fing_kd={args.fing_kd} "
          f"hand_collide={'off' if args.no_hand_collide else 'on'} "
          f"self_collide={'off' if args.no_self_collide else 'on'}")

    # ── ctx 선별: render_retarget.py / env 와 동일 규칙 ──────────────────────────────────
    _CTX_R, _CTX_SUP_R = 1.0, 1.5
    ctx_items = []
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
        ctx_items = [(n, p) for n, p, dm in _cands if n in _keep]
        print(f"[w6] ctx {len(ctx_items)}/{len(_cands)}개: {sorted(n for n, _ in ctx_items)}")

    W, H = args.res
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light",
                                                  sim_utils.DomeLightCfg(intensity=2500.0))

    SIDES = [args.only] if args.only else ["l", "r"]
    hands = [Articulation(_hand_cfg(sd, f"/World/Hand{sd.upper()}")) for sd in SIDES]
    NH = len(hands)
    if args.only:
        print(f"[w6] {args.only}손만 스폰")

    obj = None
    if obj_name and not args.no_obj:
        usd = f"{_PROC}/assets/objects/{obj_name}/{obj_name}.usd"
        if os.path.exists(usd):
            obj = RigidObject(RigidObjectCfg(
                prim_path="/World/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,
                                                                 disable_gravity=True),
                    collision_props=(sim_utils.CollisionPropertiesCfg(collision_enabled=False)
                                     if args.no_collide else None),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.0),
                                                                roughness=0.6)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(float(v) for v in obj_base[0, :3]),
                    rot=tuple(float(v) for v in obj_base[0, 3:7]))))

    for _i, (_cn, _cb) in enumerate([] if args.no_obj else ctx_items):
        _b = f"{_PROC}/assets/objects/{_cn}"
        for _cand in (f"{_b}/ctx/{_cn}_ctx.usd", f"{_b}/{_cn}_ctx.usd", f"{_b}/{_cn}.usd"):
            if os.path.exists(_cand):
                _cfg = sim_utils.UsdFileCfg(
                    usd_path=_cand, activate_contact_sensors=False,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,
                                                                 disable_gravity=True),
                    collision_props=(sim_utils.CollisionPropertiesCfg(collision_enabled=False)
                                     if args.no_collide else None))
                _cfg.func(f"/World/Ctx_{_i}_{_cn}", _cfg,
                          translation=tuple(float(v) for v in _cb[:3]),
                          orientation=tuple(float(v) for v in _cb[3:7]))
                break

    mk = None
    if not args.no_markers and not args.no_video:
        mk = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/wrist_ref", markers={
            "s": sim_utils.SphereCfg(radius=float(args.marker_radius),
                                     visual_material=sim_utils.PreviewSurfaceCfg(
                                         diffuse_color=(0.0, 1.0, 0.0)))}))

    if args.no_hand_collide:
        # 손 두 개를 각자의 CollisionGroup 에 넣고 서로를 필터링한다. sim.reset() 전에 해야 한다.
        import isaacsim.core.utils.stage as stage_utils
        from pxr import UsdPhysics
        _st = stage_utils.get_current_stage()
        _g = {}
        for _n, _pp in (("L", "/World/HandL"), ("R", "/World/HandR")):
            _cg = UsdPhysics.CollisionGroup.Define(_st, f"/World/CG_Hand{_n}")
            _cg.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(_pp)
            _g[_n] = _cg
        _g["L"].CreateFilteredGroupsRel().AddTarget(_g["R"].GetPrim().GetPath())
        _g["R"].CreateFilteredGroupsRel().AddTarget(_g["L"].GetPrim().GetPath())
        _fl = _g["L"].GetFilteredGroupsRel().GetTargets()
        _fr = _g["R"].GetFilteredGroupsRel().GetTargets()
        print(f"[w6] 손-손 충돌 필터링: L->{[str(t) for t in _fl]} R->{[str(t) for t in _fr]}")

    cam = None
    if not args.no_video:
        cam = Camera(CameraCfg(prim_path="/World/Camera", height=H, width=W, data_types=["rgb"],
                               spawn=sim_utils.PinholeCameraCfg(focal_length=20.0,
                                                                clipping_range=(0.05, 50.0))))
    print(f"[w6] 씬 구성: 손 {NH}개, 물체 {'없음' if obj is None else obj_name}, "
          f"ctx {0 if args.no_obj else len(ctx_items)}개, 지면 있음")
    sim.reset()

    # 관절 인덱스 + 손가락/J0 레퍼런스
    wid, fid, j0id, fref, j0ref, pid = [], [], [], [], [], []
    for h, sd in enumerate(SIDES):
        jn = hands[h].data.joint_names
        wid.append(torch.tensor([jn.index(f"robot0_{sd}_wrist_{n}") for n in _W6],
                                dtype=torch.long, device=sim.device))
        _ids, _ = hands[h].find_joints([e.format(s=sd) for e in _FEXPR])
        assert len(_ids) == 18, f"{sd} 구동관절 {len(_ids)} != 18"
        fid.append(torch.tensor(_ids, dtype=torch.long, device=sim.device))
        pid.append(hands[h].find_bodies([f"robot0_{sd}_palm"])[0][0])
        # 구동 18개
        _want = [jn[i] for i in _ids]
        _col = [jn_rt.index(n) for n in _want if n in jn_rt]
        assert len(_col) == len(_want), f"{sd} 손가락 레퍼런스 열 누락"
        fref.append(torch.from_numpy(
            _resample(rt["g1_joint_pos"][:, _col].astype(np.float32), t_src, t_tgt)).to(sim.device))
        # 텐던 축 J0 4개: 액추에이터가 없고 텐던이 J1 에 묶으므로, J1 을 굽힌 채 J0 를 0 으로 두면
        # 제약 위반으로 시작해 말단이 튄다.
        _j0n = [f"robot0_{sd}_{f}J0" for f in _J0F]
        j0id.append(torch.tensor([jn.index(n) for n in _j0n], dtype=torch.long, device=sim.device))
        j0ref.append(torch.from_numpy(_resample(
            rt["g1_joint_pos"][:, [jn_rt.index(n) for n in _j0n]].astype(np.float32),
            t_src, t_tgt)).to(sim.device) if all(n in jn_rt for n in _j0n) else None)

    ref_t = [torch.from_numpy(r).to(sim.device) for r in ref]
    vff_t = [torch.from_numpy(v).to(sim.device) for v in vff]
    refq_t = [torch.from_numpy(q).to(sim.device) for q in refq]

    # 초기 상태 = 레퍼런스 프레임 0. 순서 주의: reset() 이 관절 상태를 기본값(0)으로 되돌리므로
    # write 를 뒤에 해야 한다. 반대로 하면 손이 원점에서 1.4 m 오차로 출발해 포화 폭주한다.
    q0 = []
    for h in range(NH):
        q = torch.zeros(1, hands[h].num_joints, device=sim.device)
        q[0, wid[h]] = ref_t[h][0]
        q[0, fid[h]] = fref[h][0]
        if j0ref[h] is not None:
            q[0, j0id[h]] = j0ref[h][0]
        hands[h].reset()
        hands[h].write_joint_state_to_sim(q, torch.zeros_like(q))
        q0.append(q)

    # ── 카메라: render_hand_cache.py 와 같은 방식 (수직 범위에 맞춰 뒤로 빼기) ──────────────
    allp = np.concatenate([r[:, :3] for r in ref], axis=0)
    z_top = float(allp[:, 2].max()) + 0.25
    z_bot = max(0.0, float(allp[:, 2].min()) - 0.25)
    if obj_base is not None:
        z_top = max(z_top, float(obj_base[:, 2].max()) + 0.25)
    extent = max(0.3, z_top - z_bot)
    lookat_z = float(args.lookat_z) if args.lookat_z is not None else 0.5 * (z_top + z_bot)
    off = max(args.cam_min_dist, extent * 1.25) * args.cam_dist_scale
    horiz = off * (2 ** 0.5)
    az = math.radians(args.cam_yaw)
    zoff = (0.12 * extent) if args.cam_elev is None else horiz * math.tan(math.radians(args.cam_elev))
    txy = (obj_base[:, :2].mean(0) if obj_base is not None else allp[:, :2].mean(0)).astype(np.float32)
    if cam is not None:
        cam.set_world_poses_from_view(
            torch.tensor([[float(txy[0]) + horiz * math.cos(az),
                           float(txy[1]) + horiz * math.sin(az), lookat_z + zoff]],
                         device=sim.device, dtype=torch.float32),
            torch.tensor([[float(txy[0]), float(txy[1]), lookat_z]],
                         device=sim.device, dtype=torch.float32))

    writer, out = None, ""
    if not args.no_video:
        import imageio
        out = args.out or f"{cdir}/wrist6_track.mp4"
        writer = imageio.get_writer(out, fps=args.fps, macro_block_size=1)

    if args.armature > 0.0:
        for h in range(NH):
            av = torch.full((1, len(wid[h])), args.armature, device=sim.device)
            hands[h].write_joint_armature_to_sim(av, joint_ids=wid[h])
        print(f"[w6] armature={args.armature} 적용 (손목 6관절)")
    if args.fing_armature > 0.0:
        for h in range(NH):
            av = torch.full((1, len(fid[h])), args.fing_armature, device=sim.device)
            hands[h].write_joint_armature_to_sim(av, joint_ids=fid[h])
        print(f"[w6] fing_armature={args.fing_armature} 적용 (손가락 18관절)")

    ep, er, sat = [[] for _ in range(NH)], [[] for _ in range(NH)], 0
    nan_at = [-1] * NH
    onset_at = [-1] * NH
    hist = [[] for _ in range(NH)]
    sub = []
    fstat = [[] for _ in range(NH)]
    _rg = torch.Generator(device='cpu').manual_seed(int(args.res_seed))
    _res = [torch.zeros(6) for _ in range(NH)]          # EMA 상태
    _res_scale = torch.tensor([args.res_t] * 3 + [args.res_r] * 3)
    _res_on = bool(args.res_t > 0.0 or args.res_r > 0.0)
    _res_mag = [[] for _ in range(NH)]      # (병진힘 크기, 회전토크 크기) 프레임별          # NaN 발생 직전 상태 이력 (링 버퍼)
    for f in range(F):
        for h in range(NH):
            tgt = q0[h].clone()
            tgt[0, wid[h]] = ref_t[h][f]
            if _res_on:
                _u = torch.rand(6, generator=_rg) * 2.0 - 1.0          # [-1,1]
                _res[h] = (1.0 - args.res_ema) * _res[h] + args.res_ema * _u
                _d = (_res[h] * _res_scale).to(tgt.device)
                tgt[0, wid[h]] = tgt[0, wid[h]] + _d
                _res_mag[h].append((float(_d[:3].norm()), float(_d[3:].norm())))
            tgt[0, fid[h]] = fref[h][f]
            if j0ref[h] is not None:
                tgt[0, j0id[h]] = j0ref[h][f]
            hands[h].set_joint_position_target(tgt)
            if not args.no_vff:
                v = torch.zeros_like(tgt)
                v[0, wid[h]] = vff_t[h][f]
                hands[h].set_joint_velocity_target(v)
        if obj is not None:
            _s = min(F30 - 1, int(round(f * REF_FPS / args.fps)))
            _p = torch.from_numpy(obj_base[_s:_s + 1, :7]).to(sim.device)
            obj.write_root_pose_to_sim(_p)
            obj.write_data_to_sim()
        _lo, _hi = args.substep_range
        _grab = bool(args.substep_dump) and _lo <= f <= _hi
        for _k in range(DEC):
            for h in range(NH):
                hands[h].write_data_to_sim()
            sim.step(render=False)
            for h in range(NH):
                hands[h].update(DT)
            if _grab:
                for h in range(NH):
                    sub.append(np.concatenate([
                        [f, _k, h],
                        hands[h].data.joint_pos[0, wid[h]].detach().cpu().numpy(),
                        hands[h].data.joint_vel[0, wid[h]].detach().cpu().numpy(),
                        hands[h].data.applied_torque[0, wid[h]].detach().cpu().numpy()]))
        if mk is not None:
            mk.visualize(translations=torch.stack([ref_t[i][f, :3] for i in range(NH)]))
        if cam is not None:
            sim.render()
            cam.update(DT * DEC)   # 필수: 없으면 cam.data.output 이 갱신되지 않아 정지화면이 나온다
        for h in range(NH):
            p = hands[h].data.body_pos_w[0, pid[h]]
            q = hands[h].data.body_quat_w[0, pid[h]]
            jp = hands[h].data.joint_pos[0, wid[h]]
            jv = hands[h].data.joint_vel[0, wid[h]]
            tq = hands[h].data.applied_torque[0, wid[h]]
            if bool(torch.isfinite(tq).all()):
                fstat[h].append((float(tq[:3].norm()), float(tq[3:].norm())))
            if nan_at[h] < 0:
                if bool(torch.isfinite(p).all()):
                    _e = float((p - ref_t[h][f, :3]).norm())
                    if onset_at[h] < 0 and _e * 1000.0 > args.onset_mm:
                        onset_at[h] = f
                        print(f"[w6] >> {SIDES[h]}손 발산 시작 f={f} 오차 {_e*1000:.1f}mm "
                              f"vel최대 {float(jv.abs().max()):.2f} "
                              f"|tau|최대 {float(tq.abs().max()):.2f}", flush=True)
                        print(f"[w6]    관절vel {[round(float(x),3) for x in jv]}", flush=True)
                        print(f"[w6]    관절pos {[round(float(x),3) for x in jp]}", flush=True)
                        print(f"[w6]    목표6   {[round(float(x),3) for x in ref_t[h][f]]}", flush=True)
                    hist[h].append((f, jp.clone(), jv.clone(), tq.clone(), _e))
                    if len(hist[h]) > 10:
                        hist[h].pop(0)
                else:
                    nan_at[h] = f
                    print(f"[w6] !! {'lr'[h]}손 NaN 최초 프레임 {f} — 직전 유한 프레임 이력",
                          flush=True)
                    print(f"[w6]    {'f':>4s} {'오차mm':>8s} | {'관절vel 최대':>11s} "
                          f"{'|torque| 최대':>12s} | {'vel(6)':>44s}", flush=True)
                    for (hf, hjp, hjv, htq, he) in hist[h]:
                        print(f"[w6]    {hf:4d} {he*1000:8.2f} | {float(hjv.abs().max()):11.2f} "
                              f"{float(htq.abs().max()):12.2f} | "
                              f"{' '.join(f'{float(x):7.2f}' for x in hjv)}", flush=True)
                    print(f"[w6]    목표6(f={f}) "
                          f"{[round(float(x),3) for x in ref_t[h][f]]}", flush=True)
                    print(f"[w6]    목표6(f={f-1}) "
                          f"{[round(float(x),3) for x in ref_t[h][f-1]]}", flush=True)
            ep[h].append(float((p - ref_t[h][f, :3]).norm()))
            er[h].append(float(_quat_err(q.unsqueeze(0), refq_t[h][f].unsqueeze(0))[0]))
            tau = hands[h].data.applied_torque[0, wid[h]]
            if float(tau[:3].abs().max()) > args.eff_t * 0.99 or \
               float(tau[3:].abs().max()) > args.eff_r * 0.99:
                sat += 1
        if f % 100 == 0:
            print(f"[w6] frame {f}/{F}", flush=True)
        if writer is not None:
            writer.append_data(
                cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
    if writer is not None:
        writer.close()

    print(f"\n[w6] {'손':>4s} | {'pos중앙':>9s} {'pos p99':>9s} {'pos최대':>9s} | "
          f"{'rot중앙':>9s} {'rot p99':>9s} {'rot최대':>9s}")
    for h, sd in enumerate(SIDES):
        a, b = np.array(ep[h]), np.array(er[h])
        print(f"[w6] {sd:>4s} | {np.median(a)*1000:8.2f}mm {np.quantile(a,.99)*1000:8.2f}mm "
              f"{a.max()*1000:8.2f}mm | {np.degrees(np.median(b)):8.2f}° "
              f"{np.degrees(np.quantile(b,.99)):8.2f}° {np.degrees(b.max()):8.2f}°")
    if args.substep_dump and sub:
        np.savez(args.substep_dump, rows=np.stack(sub).astype(np.float64),
                 cols=np.array(["frame", "substep", "hand"]
                                + [f"pos_{n}" for n in _W6] + [f"vel_{n}" for n in _W6]
                                + [f"tau_{n}" for n in _W6]),
                 sides=np.array(SIDES), dt=np.float64(DT))
        print(f"[w6] 서브스텝 {len(sub)}행 저장: {args.substep_dump}")
    if args.dump_err:
        np.savez(args.dump_err, sides=np.array(SIDES),
                 **{f"ep_{SIDES[h]}": np.array(ep[h]) for h in range(NH)},
                 **{f"er_{SIDES[h]}": np.array(er[h]) for h in range(NH)})
        print(f"[w6] 오차 곡선 저장: {args.dump_err}")
    if _res_on:
        print(f"[w6] 잔차 스케일 t={args.res_t} m  r={args.res_r} rad  EMA={args.res_ema} "
              f"seed={args.res_seed}")
        for h in range(NH):
            if _res_mag[h]:
                _rm = np.array(_res_mag[h])
                print(f"[w6] {SIDES[h]}손 실제 잔차: 병진 중앙 {np.median(_rm[:,0])*1000:6.1f} mm "
                      f"최대 {_rm[:,0].max()*1000:6.1f}  |  회전 중앙 "
                      f"{np.degrees(np.median(_rm[:,1])):5.2f}도 최대 {np.degrees(_rm[:,1].max()):5.2f}")
    for h in range(NH):
        if not fstat[h]:
            continue
        _fa = np.array(fstat[h])
        print(f"[w6] {SIDES[h]}손 손목 힘: 병진 중앙 {np.median(_fa[:,0]):6.2f} N "
              f"p99 {np.quantile(_fa[:,0],.99):7.2f} 최대 {_fa[:,0].max():7.2f}  |  "
              f"회전 중앙 {np.median(_fa[:,1]):6.3f} Nm p99 {np.quantile(_fa[:,1],.99):6.3f} "
              f"최대 {_fa[:,1].max():6.3f}")
    print(f"[w6] effort 포화 {sat}/{F*NH} ({sat/(F*NH)*100:.1f}%)")
    print(f"[w6] NaN 최초: {dict(zip(SIDES, nan_at))}  발산시작({args.onset_mm}mm): "
          f"{dict(zip(SIDES, onset_at))}  (-1 = 없음)")
    if out:
        print(f"[w6] wrote {out} ({F} frames @ {args.fps} fps)")
    sys.stdout.flush()          # os._exit 는 stdio 를 flush 하지 않는다
    os._exit(0)


if __name__ == "__main__":
    main()
