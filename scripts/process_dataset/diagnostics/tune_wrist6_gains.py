"""Gain tuning for the `wrist6` 6-DoF wrist chain: drive it along the retargeted reference and
measure tracking error + actuator effort.

Method is DexMachina's `hand_proc/tune_gains.py`: play the reference through the joint controller
and compare the achieved pose against the reference, watching for effort saturation. We measure in
the units that matter downstream — palm POSITION error (m) and palm ORIENTATION error (rad) — not
joint-space error, because those are what the env's rewards and termination gates use.

The reference comes from `wrist_dof6.npz` (export_wrist_dof6.py), which is exact: the forward map
6-DoF -> palm pose was verified in sim to 0.0005 mm / 4e-7. So any error measured here is the
CONTROLLER failing to track, never a conversion artifact. That separation is the point of the tool.

Fingers are held at the retargeted pose (from trajectory_pyroki.npz) so the wrist carries a
realistic inertia; the object is NOT spawned — this isolates the wrist controller from contact.

    <env_isaaclab python> scripts/process_dataset/diagnostics/tune_wrist6_gains.py \
        --clip s101_seg12_knife --sweep
    # --kp_t/--kd_t/--kp_r/--kd_r to evaluate one setting
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="s101_seg12_knife")
parser.add_argument("--class", dest="cls", default="single_rigid")
parser.add_argument("--side", default="l", choices=("l", "r"))
parser.add_argument("--kp_t", type=float, default=2000.0, help="wrist translation stiffness")
parser.add_argument("--kd_t", type=float, default=200.0, help="wrist translation damping")
parser.add_argument("--kp_r", type=float, default=200.0, help="wrist rotation stiffness")
parser.add_argument("--kd_r", type=float, default=20.0, help="wrist rotation damping")
parser.add_argument("--eff_t", type=float, default=200.0, help="translation effort limit (N)")
parser.add_argument("--eff_r", type=float, default=50.0, help="rotation effort limit (N*m)")
parser.add_argument("--sweep", action="store_true", help="sweep a grid instead of one setting")
parser.add_argument("--fingers_ref", action="store_true", default=True,
                    help="hold fingers at the retargeted pose (realistic inertia)")
args = parser.parse_args()

app_launcher = AppLauncher(headless=True)
sim_app = app_launcher.app

import os  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

_ROOT = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data"
_PROC = f"{_ROOT}/processed/parahome"
_ROT_SEQ = "YZX"   # build_shadow_floating_usd.py 의 _W6_ROT_SEQ 와 일치해야 한다
_W6 = ["tx", "ty", "tz", "rot1", "rot2", "rot3"]
_FEXPR = ["robot0_{s}_(FF|MF|RF|LF|TH)J[1-3]", "robot0_{s}_LFJ4",
          "robot0_{s}_THJ4", "robot0_{s}_THJ0"]
DT, DEC = 1.0 / 200.0, 4          # 50 Hz control, same as the env


def _quat_err(qa, qb):
    """(N,4),(N,4) wxyz -> (N,) angle in rad."""
    d = (qa * qb).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.arccos(d)


def main():
    side = args.side
    cd = f"{_PROC}/g1_shadow/{args.cls}/{args.clip}/0"
    d6 = np.load(f"{cd}/wrist_dof6.npz", allow_pickle=True)
    ref6 = d6[f"wrist_dof_{side}"].astype(np.float32)                 # (F,6) 30 fps
    rt = np.load(f"{cd}/trajectory_pyroki.npz", allow_pickle=True)
    F30 = len(ref6)
    # 30 -> 50 fps, matching the env's control rate (linear on the DoF values; they are unwrapped
    # so linear interpolation is safe — that is a reason to unwrap upstream).
    F = int(round((F30 - 1) * 50.0 / 30.0)) + 1
    t_src, t_tgt = np.arange(F30) / 30.0, np.arange(F) / 50.0
    ref = np.stack([np.interp(t_tgt, t_src, ref6[:, k]) for k in range(6)], 1).astype(np.float32)
    # 기대 pose 는 **목표 6-DoF 의 순방향 사상**으로 만든다. wrist_ref 의 pose 를 따로 보간해
    # 비교하면 두 보간(6-DoF 선형 vs pose 선형)이 등가가 아니라 그 불일치가 오차를 지배한다 —
    # 실측: 게인을 500~8000 으로 16배 바꿔도 위치오차가 4.03 mm 로 **완전히 동일**했다. 그건
    # 컨트롤러 성능이 아니라 지표의 결함이었다. 순방향 사상은 sim 에서 0.0005 mm 로 검증됐으므로
    # 여기서 재는 오차는 순수하게 컨트롤러의 추종 실패다.
    def _euler_to_quat(a, b, c):
        """내재 회전 YZX -> quat wxyz. 규약은 export_wrist_dof6.py 의 _ROT_SEQ 와 같아야 한다."""
        from scipy.spatial.transform import Rotation as _R
        _q = _R.from_euler(_ROT_SEQ, np.stack([a, b, c], 1)).as_quat()   # xyzw
        return _q[:, [3, 0, 1, 2]]
    refp = ref[:, :3].copy()
    refq = _euler_to_quat(ref[:, 3], ref[:, 4], ref[:, 5]).astype(np.float32)
    refq /= np.linalg.norm(refq, axis=1, keepdims=True)

    sim = SimulationContext(SimulationCfg(dt=DT, device="cuda:0"))
    grid = ([(kp_t, kd_t, kp_r, kd_r)
             for kp_t in (500.0, 2000.0, 8000.0) for kd_t in (50.0, 200.0)
             for kp_r in (50.0, 200.0, 800.0) for kd_r in (5.0, 20.0)]
            if args.sweep else [(args.kp_t, args.kd_t, args.kp_r, args.kd_r)])

    print(f"\n[tune] clip={args.clip} side={side}  F={F} @50fps  설정 {len(grid)}개")
    print(f"[tune] 레퍼런스 요구: 이동 p99 "
          f"{np.quantile(np.abs(np.diff(ref[:, :3], axis=0)).max(1), .99) * 50:.3f} m/s, "
          f"회전 p99 {np.quantile(np.abs(np.diff(ref[:, 3:], axis=0)).max(1), .99) * 50:.3f} rad/s")
    print(f"\n{'kp_t':>7s} {'kd_t':>6s} {'kp_r':>6s} {'kd_r':>5s} | "
          f"{'pos중앙':>8s} {'pos p99':>8s} {'pos최대':>8s} | "
          f"{'rot중앙':>8s} {'rot p99':>8s} | {'힘max':>8s} {'토크max':>8s} {'포화%':>6s}")

    hand = None
    for (kp_t, kd_t, kp_r, kd_r) in grid:
        cfg = ArticulationCfg(
            prim_path="/World/Hand",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{_ROOT}/robots/G1/shadow_float6_{side}.usd",
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True, max_depenetration_velocity=1000.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, fix_root_link=True,
                    solver_position_iteration_count=8, solver_velocity_iteration_count=0),
                fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0,
                                                                      damping=0.2)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.), rot=(1., 0., 0., 0.),
                                                       joint_pos={".*": 0.0}),
            actuators={
                "wrist_trans": ImplicitActuatorCfg(
                    joint_names_expr=[f"robot0_{side}_wrist_t[xyz]"],
                    stiffness=kp_t, damping=kd_t,
                    effort_limit_sim=args.eff_t, velocity_limit_sim=5.0),
                "wrist_rot": ImplicitActuatorCfg(
                    joint_names_expr=[f"robot0_{side}_wrist_rot[123]"],
                    stiffness=kp_r, damping=kd_r,
                    effort_limit_sim=args.eff_r, velocity_limit_sim=20.0),
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=[e.format(s=side) for e in _FEXPR],
                    stiffness=1.0, damping=0.2,
                    effort_limit_sim=3.09, velocity_limit_sim=15.0)},
            soft_joint_pos_limit_factor=0.9)
        if hand is None:
            hand = Articulation(cfg)
            sim.reset()
            jn = hand.data.joint_names
            wid = torch.tensor([jn.index(f"robot0_{side}_wrist_{n}") for n in _W6],
                               dtype=torch.long, device=sim.device)
            pid = hand.find_bodies([f"robot0_{side}_palm"])[0][0]
            fid, _ = hand.find_joints([e.format(s=side) for e in _FEXPR])
            fid = torch.tensor(fid, dtype=torch.long, device=sim.device)
            # 텐던 축 J0 4개: 액추에이터가 없고 텐던이 J1 에 묶으므로, J1 을 굽힌 채 J0 를 0 으로
            # 두면 제약 위반으로 시작해 말단이 튄다 (env 에서 같은 문제를 겪었다). 리타게팅이 J0 를
            # 풀어 저장했으면 그 값을 쓴다.
            j0_id = torch.tensor(
                [jn.index(f"robot0_{side}_{f}J0") for f in ("FF", "MF", "RF", "LF")],
                dtype=torch.long, device=sim.device)
            j0_t = None
            _j0n = [f"robot0_{side}_{f}J0" for f in ("FF", "MF", "RF", "LF")]
            _jnn = [str(x) for x in rt["joint_names"]]
            if all(n in _jnn for n in _j0n):
                _c = [_jnn.index(n) for n in _j0n]
                _q0 = rt["g1_joint_pos"][:, _c].astype(np.float32)
                j0_t = torch.from_numpy(np.stack(
                    [np.interp(t_tgt, t_src, _q0[:, k]) for k in range(4)], 1).astype(np.float32)
                ).to(sim.device)
            fref = None
            if args.fingers_ref:
                _jn = [str(x) for x in rt["joint_names"]]
                _want = [jn[i] for i in fid.tolist()]
                _col = [_jn.index(n) for n in _want if n in _jn]
                if len(_col) == len(_want):
                    _q = rt["g1_joint_pos"][:, _col].astype(np.float32)
                    fref = np.stack([np.interp(t_tgt, t_src, _q[:, k])
                                     for k in range(_q.shape[1])], 1).astype(np.float32)
            ref_t = torch.from_numpy(ref).to(sim.device)
            refp_t = torch.from_numpy(refp).to(sim.device)
            refq_t = torch.from_numpy(refq).to(sim.device)
            fref_t = torch.from_numpy(fref).to(sim.device) if fref is not None else None
        else:
            # 게인만 갈아끼운다 (씬 재생성 없이) — actuator 의 stiffness/damping 버퍼를 직접 쓴다
            for grp, kp, kd in (("wrist_trans", kp_t, kd_t), ("wrist_rot", kp_r, kd_r)):
                act = hand.actuators[grp]
                act.stiffness[:] = kp
                act.damping[:] = kd

        # 초기 상태 = 레퍼런스 프레임 0.
        # 순서 주의: reset() 이 관절 상태를 기본값(0)으로 되돌리므로 write 를 **뒤에** 해야 한다.
        # 반대로 하면 손이 원점에서 시작해 목표(z~1.03 m)와 1.4 m 오차로 출발하고, 그 오차가
        # kp 를 곱해 effort 한계를 넘겨 포화 폭주한다 (실측: 위치오차 185 mm, 포화 99.2%).
        q0 = torch.zeros(1, hand.num_joints, device=sim.device)
        q0[0, wid] = ref_t[0]
        if fref_t is not None:
            q0[0, fid] = fref_t[0]
        if j0_t is not None:
            q0[0, j0_id] = j0_t[0]
        hand.reset()
        hand.write_joint_state_to_sim(q0, torch.zeros_like(q0))

        ep, er, ef, et, sat = [], [], [], [], 0
        for f in range(F):
            tgt = q0.clone()
            tgt[0, wid] = ref_t[f]
            if fref_t is not None:
                tgt[0, fid] = fref_t[f]
            # J0 는 액추에이터가 없어 목표를 줄 수 없다(텐던이 구동). tgt 의 해당 열은 무시된다.
            hand.set_joint_position_target(tgt)
            for _ in range(DEC):
                hand.write_data_to_sim()
                sim.step(render=False)
                hand.update(DT)
            p = hand.data.body_pos_w[0, pid]
            q = hand.data.body_quat_w[0, pid]
            ep.append(float((p - refp_t[f]).norm()))
            er.append(float(_quat_err(q.unsqueeze(0), refq_t[f].unsqueeze(0))[0]))
            tau = hand.data.applied_torque[0, wid]
            ef.append(float(tau[:3].abs().max()))
            et.append(float(tau[3:].abs().max()))
            if ef[-1] > args.eff_t * 0.99 or et[-1] > args.eff_r * 0.99:
                sat += 1
        ep, er = np.array(ep), np.array(er)
        print(f"{kp_t:7.0f} {kd_t:6.0f} {kp_r:6.0f} {kd_r:5.0f} | "
              f"{np.median(ep)*1000:7.2f}m {np.quantile(ep,.99)*1000:7.2f}m {ep.max()*1000:7.2f}m | "
              f"{np.degrees(np.median(er)):7.2f}d {np.degrees(np.quantile(er,.99)):7.2f}d | "
              f"{max(ef):8.2f} {max(et):8.3f} {sat/F*100:5.1f}%")

    print("\n[tune] 단위: pos = mm, rot = deg, 힘 = N, 토크 = N*m")
    print("[tune] 완료")
    os._exit(0)


if __name__ == "__main__":
    main()
