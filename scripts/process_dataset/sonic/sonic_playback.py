"""MILESTONE 1 — frozen SONIC drives G1_shadow from ParaHome SMPL, closed-loop physics, NO RL.

Multi-clip: reuses ONE Isaac Sim session (spawn once, reset+replay per clip) to avoid re-launching.
Per clip: writes mp4 + a quantitative tracking metric (pelvis-relative body-keypoint error vs the
human ParaHome reference) + an upright check (min root height). Keeps g1_shadow_locomanip files
UNTOUCHED. Fixes applied: SMPL ytoz skipped (ParaHome=Z-up) in parahome_smpl_for_sonic.py, and the
robot spawns facing the reference root (g1_root_quat0) so the initial orientation error ~= 0.

Run (tmux, Isaac Sim env):
  python -u scripts/process_dataset/sonic/sonic_playback.py --free_base --clips s100_seg00_pan s101_seg29_pot ...
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clips", nargs="+", default=[
    "s100_seg00_pan", "s101_seg29_pot", "s10_seg03_book", "s100_seg03_cup",
    "s101_seg12_knife", "s100_seg02_kettle"])
parser.add_argument("--class", dest="cls", default="single_rigid")
parser.add_argument("--res", type=int, nargs=2, default=[960, 540])
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--free_base", action="store_true")
parser.add_argument("--hold", action="store_true", help="freeze SMPL ref at frame 0 (can-it-stand)")
parser.add_argument("--ref_reset", action="store_true",
                    help="reset robot to the retarget reference pose (g1_joint_pos[0]) instead of SONIC standing default (faithful RSI; needed for crouch clips)")
parser.add_argument("--out_dir", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app

import copy  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_apply  # noqa: E402
from gear_sonic.trl.utils.torch_transform import quat_inv as _qinv, quat_mul as _qmul  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import sonic_prior as SP  # noqa: E402
from robotis_sh5.tasks.direct.g1_shadow_locomanip.g1_shadow_locomanip_env_cfg import (  # noqa: E402
    G1_SHADOW_CFG, BODY_KPTS, BODY_KPT_OFFSETS)

_PROC = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/processed/parahome"
_SCRATCH = "/tmp/claude-1000/-home-peunsu-workspace/f4cff77a-4a9a-44a1-84d5-e52903167cd4/scratchpad"
device = "cuda:0"
SRC_FPS, TGT_FPS = 30.0, 50.0

BK_PARA = list(BODY_KPTS.keys())                       # 16 ParaHome joint indices (order)
BK_LINK = [BODY_KPTS[i] for i in BK_PARA]              # G1 link names
ARM_POS = [BK_PARA.index(i) for i in (8, 9, 10, 12, 13, 14)]  # arm kpt slots (shoulder/elbow/wrist L/R)
LEG_POS = [BK_PARA.index(i) for i in (15, 16, 17, 18, 19, 20, 21, 22)]


def _sonic_actuators():
    return {
        "sonic_hip_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"],
            effort_limit_sim=300.0, velocity_limit_sim=100.0, stiffness=99.0997, damping=6.3088, armature=0.025101925),
        "sonic_hipyaw_waistyaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", "waist_yaw_joint"],
            effort_limit_sim=300.0, velocity_limit_sim=100.0, stiffness=40.1795, damping=2.5579, armature=0.010177520),
        "sonic_ankle_waist": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=200.0, velocity_limit_sim=100.0, stiffness=28.5013, damping=1.8143, armature=0.00721945),
        "sonic_shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                              ".*_elbow_joint", ".*_wrist_roll_joint"],
            effort_limit_sim=150.0, velocity_limit_sim=100.0, stiffness=14.2506, damping=0.9072, armature=0.003609725),
        "sonic_wrist_pitchyaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit_sim=150.0, velocity_limit_sim=100.0, stiffness=16.7783, damping=1.0681, armature=0.00425),
        "shadow_fingers": ImplicitActuatorCfg(
            joint_names_expr=["robot0_(l|r)_(FF|MF|RF|LF|TH)J[1-3]", "robot0_(l|r)_LFJ4",
                              "robot0_(l|r)_THJ4", "robot0_(l|r)_THJ0"],
            velocity_limit_sim=15.0, effort_limit_sim=3.09, stiffness=1.0, damping=0.2),
    }


def _lerp_np(x, t_src, t_tgt):
    return np.stack([np.interp(t_tgt, t_src, x[:, d]) for d in range(x.shape[1])], axis=1).astype(np.float32)


def load_clip(clip, cls):
    sm = np.load(os.path.join(_PROC, "g1_shadow", cls, clip, "0", "sonic_smpl_50fps.npz"))
    d = {k: torch.tensor(sm[k], device=device) for k in ("smpl_joints_local", "root_q_zb", "wrist_ref")}
    d["g1_root_quat0"] = tuple(float(x) for x in sm["g1_root_quat0"])
    # retarget reference frame-0 (for --ref_reset RSI): g1_joint_pos (65, action order) + root z
    rt = np.load(os.path.join(_PROC, "g1_shadow", cls, clip, "0", "trajectory_pyroki.npz"))
    d["ref_q"] = torch.tensor(rt["g1_joint_pos"][0], device=device)                 # (65,) action order
    d["ref_root_z"] = float(rt["g1_root_pose"][0, 2]) if "g1_root_pose" in rt.files else 0.80
    N = d["smpl_joints_local"].shape[0]
    # human reference body kpts (ParaHome joint_positions), resampled 30->50fps to align with playback
    jp = np.load(os.path.join(_PROC, "smplx", cls, clip, "0", "trajectory.npz"))["joint_positions"]  # (F,73,3)
    F = jp.shape[0]
    hk = jp[:, BK_PARA, :].reshape(F, -1)                                       # (F,48)
    t_src, t_tgt = np.arange(F) / SRC_FPS, np.linspace(0.0, (F - 1) / SRC_FPS, N)
    d["human_kpts"] = torch.tensor(_lerp_np(hk, t_src, t_tgt).reshape(N, len(BK_PARA), 3), device=device)  # (N,16,3)
    d["N"] = N
    return d


def main():
    W, H = args.res
    sim_dt, decimation = 0.005, 4
    out_dir = args.out_dir or _SCRATCH

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=sim_dt, device=device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    cfg = copy.deepcopy(G1_SHADOW_CFG)
    cfg.prim_path = "/World/Robot"
    cfg.spawn.articulation_props.fix_root_link = not args.free_base
    cfg.init_state.joint_pos = {
        ".*_hip_pitch_joint": -0.312, ".*_knee_joint": 0.669, ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6, "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
        "left_shoulder_pitch_joint": 0.2, "right_shoulder_pitch_joint": 0.2}
    cfg.actuators = _sonic_actuators()
    robot = Articulation(cfg)
    cam = Camera(CameraCfg(prim_path="/World/Camera", height=H, width=W, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.05, 50.0))))
    sim.reset()

    actor = SP.build_sonic(device=device)
    layout, TOK = SP.tokenizer_layout(actor)
    perm = SP.build_body_perm(list(robot.joint_names), device=device)
    sonic_def = SP.sonic_default_vector(device).view(1, -1)
    sonic_scale = SP.sonic_scale_vector(device).view(1, -1)
    E = robot.num_instances
    default_q = robot.data.default_joint_pos.clone()
    dt = sim.get_physics_dt()

    # action_joint_names(65) -> robot articulation joint index, for --ref_reset RSI writes
    import json as _json
    _AJN = _json.load(open("/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/robots/G1/"
                           "g1_shadow_joint_order.json"))["action_joint_names"]
    act_to_robot = torch.tensor([list(robot.joint_names).index(n) for n in _AJN], device=device)  # (65,)

    bk_link_idx = torch.tensor([list(robot.body_names).index(n) for n in BK_LINK], device=device)  # (16,)
    bk_off = torch.zeros(len(BK_PARA), 3, device=device)
    for j, pidx in enumerate(BK_PARA):
        if pidx in BODY_KPT_OFFSETS:
            bk_off[j] = torch.tensor(BODY_KPT_OFFSETS[pidx], device=device)

    cam.set_world_poses_from_view(torch.tensor([[2.2, 2.2, 1.1]], device=device),
                                  torch.tensor([[0.0, 0.0, 0.7]], device=device))

    os.makedirs(out_dir, exist_ok=True)
    HIST = SP.PROPRIO_HIST
    results = []
    for clip in args.clips:
        C = load_clip(clip, args.cls)
        N = C["N"]
        smpl_j, root_q, wr, human_k = C["smpl_joints_local"], C["root_q_zb"], C["wrist_ref"], C["human_kpts"]

        # ---- reset robot: joints = SONIC default OR retarget ref pose (--ref_reset), root faces
        #      the reference, feet grounded. Crouch clips NEED ref_reset (start crouched, not standing). ----
        reset_q = default_q.clone()
        if args.ref_reset:
            reset_q[:, act_to_robot] = C["ref_q"].unsqueeze(0)           # retarget crouch/stand pose
        z0 = C["ref_root_z"] if args.ref_reset else 0.80
        rp = torch.zeros(E, 7, device=device); rp[:, 2] = z0
        rp[:, 3:] = torch.tensor(C["g1_root_quat0"], device=device)
        robot.write_joint_state_to_sim(reset_q, torch.zeros_like(reset_q))
        robot.write_root_pose_to_sim(rp); robot.write_root_velocity_to_sim(torch.zeros(E, 6, device=device))
        robot.write_data_to_sim(); sim.step(render=False); robot.update(dt)
        min_z = robot.data.body_pos_w[0, :, 2].min().item()
        rp[:, 2] += (0.02 - min_z)                                        # ground the feet
        robot.write_root_pose_to_sim(rp); robot.write_joint_state_to_sim(reset_q, torch.zeros_like(reset_q))
        robot.write_root_velocity_to_sim(torch.zeros(E, 6, device=device))
        robot.write_data_to_sim(); robot.update(dt)

        hist = {k: torch.zeros(E, HIST, d0, device=device) for k, d0 in
                [("ang", 3), ("jpr", 29), ("jvr", 29), ("act", 29), ("grav", 3)]}
        last_a = torch.zeros(E, 29, device=device)

        def read_row():
            q = robot.data.joint_pos[:, perm]
            return (robot.data.root_ang_vel_b, q - sonic_def, robot.data.joint_vel[:, perm],
                    robot.data.projected_gravity_b)
        ang0, jpr0, jvr0, grav0 = read_row()
        for k, v in [("ang", ang0), ("jpr", jpr0), ("jvr", jvr0), ("act", last_a), ("grav", grav0)]:
            hist[k][:] = v.unsqueeze(1)

        def flat_proprio():
            return torch.cat([hist["ang"].reshape(E, -1), hist["jpr"].reshape(E, -1), hist["jvr"].reshape(E, -1),
                              hist["act"].reshape(E, -1), hist["grav"].reshape(E, -1)], dim=-1)

        def build_tok(f):
            idx = (torch.zeros(10, dtype=torch.long, device=device) if args.hold
                   else torch.arange(f, f + 10, device=device).clamp(max=N - 1))
            tok = torch.zeros(E, TOK, device=device)
            s, e, _ = layout["encoder_index"]; tok[:, s:e] = torch.tensor([0., 0., 1.], device=device)
            s, e, _ = layout["smpl_joints_multi_future_local_nonflat"]; tok[:, s:e] = smpl_j[idx].reshape(-1)
            s, e, _ = layout["joint_pos_multi_future_wrist_for_smpl"]; tok[:, s:e] = wr[idx].reshape(-1)
            pelvis_q = robot.data.root_quat_w
            rq = root_q[idx].unsqueeze(0).expand(E, 10, 4)
            dif = _qmul(_qinv(pelvis_q).unsqueeze(1).expand(E, 10, 4), rq)
            ori6 = matrix_from_quat(dif.reshape(-1, 4))[..., :2].reshape(E, 10, 6)
            s, e, _ = layout["smpl_root_ori_b_multi_future"]; tok[:, s:e] = ori6.reshape(E, -1)
            return tok

        import imageio
        out = os.path.join(out_dir, f"sonic_{clip}.mp4")
        writer = imageio.get_writer(out, fps=args.fps, macro_block_size=1)
        errs, arm_errs, root_zs = [], [], []
        for f in range(N):
            a_sonic = SP.act(actor, flat_proprio(), build_tok(f))
            target = default_q.clone(); target[:, perm] = sonic_def + sonic_scale * a_sonic
            robot.set_joint_position_target(target)
            for _ in range(decimation):
                robot.write_data_to_sim(); sim.step(render=False); robot.update(sim_dt)
            for k, v in [("ang", robot.data.root_ang_vel_b), ("jpr", robot.data.joint_pos[:, perm] - sonic_def),
                         ("jvr", robot.data.joint_vel[:, perm]), ("act", a_sonic),
                         ("grav", robot.data.projected_gravity_b)]:
                hist[k][:, :-1] = hist[k][:, 1:].clone(); hist[k][:, -1] = v
            last_a = a_sonic
            # ---- tracking metric: pelvis-relative body-kpt error vs human ref ----
            rk = robot.data.body_pos_w[0, bk_link_idx] + quat_apply(robot.data.body_quat_w[0, bk_link_idx], bk_off)
            rk_rel = rk - rk[0]; hk_rel = human_k[f] - human_k[f][0]         # (16,3), pelvis-relative
            e = (rk_rel - hk_rel).norm(dim=-1)                               # (16,)
            errs.append(e.mean().item()); arm_errs.append(e[ARM_POS].mean().item())
            root_zs.append(robot.data.root_pos_w[0, 2].item())
            sim.render(); cam.update(dt)
            writer.append_data(cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
        writer.close()
        # fell = root drops well below its own reset height (crouch clips reset ~0.5, stand ~0.78)
        fell = [i for i, z in enumerate(root_zs) if z < 0.30]
        fell_at = fell[0] if fell else -1
        r = dict(clip=clip, N=N, body_err=float(np.mean(errs)), arm_err=float(np.mean(arm_errs)),
                 min_root_z=float(np.min(root_zs)), end_root_z=root_zs[-1], fell_at=fell_at,
                 upright=bool(fell_at < 0))
        results.append(r)
        print(f"[{clip}] N={N} upright={r['upright']} fell_at={fell_at} min_root_z={r['min_root_z']:.3f} "
              f"end_root_z={r['end_root_z']:.3f} body_err={r['body_err']*100:.1f}cm arm_err={r['arm_err']*100:.1f}cm -> {out}")

    print("\n==== SUMMARY (pelvis-relative body-kpt tracking error) ====")
    print(f"{'clip':22s} {'upright':8s} {'minRootZ':9s} {'bodyErr':8s} {'armErr':8s}")
    for r in results:
        print(f"{r['clip']:22s} {str(r['upright']):8s} {r['min_root_z']:.3f}     "
              f"{r['body_err']*100:5.1f}cm  {r['arm_err']*100:5.1f}cm")
    os._exit(0)


if __name__ == "__main__":
    main()
