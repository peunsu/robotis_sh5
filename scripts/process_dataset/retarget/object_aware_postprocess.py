"""Object-aware retargeting post-processing (dexmachina/SPIDER, paper appendix A.2).

Purely kinematic retargeting solves the hand against fingertip positions with no notion of the
object's volume, so the fingers end up INSIDE it (measured: ~3.2 cm against the true knife mesh).
Two things break as a result — the object is ejected when spawned around such a hand, and the
keypoints used for the imitation reward are positions the robot can never physically reach.

Let the simulator resolve it, per frame:

    fixate the OBJECT at its reference pose
    fixate the ROBOT BASE (fix_root_link=True) at its reference pose
    command the retargeted joint values as POSITION TARGETS
    step until it settles
    record the ACHIEVED joint values

Physics pushes the fingers out of the object while the PD keeps pulling them toward the retarget, so
the equilibrium is the closest penetration-free pose.

WHY THIS IS A STANDALONE SCRIPT and not a pass inside the training env: the env spawns the robot with
a FLOATING base, and holding it in place by rewriting root pose + zero velocity every substep is not
a physical constraint — it discards the momentum PhysX just integrated, and the reaction goes into the
joints. Measured that way, the settle never converged: the LEFT ARM ran 1.5-2.4 rad off target with
joint velocities of 20-78 rad/s, and the contact force ended up HIGHER than it started (33 -> 48 N).
A genuinely fixed base removes that fight, and fix_root_link is a spawn-time property.

    python scripts/process_dataset/retarget/object_aware_postprocess.py \
        --clip s101_seg12_knife --render --save
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Let physics resolve hand-object penetration in the retarget.")
parser.add_argument("--clip", type=str, default="s101_seg12_knife")
parser.add_argument("--cls", type=str, default="single_rigid")
parser.add_argument("--settle_steps", type=int, default=80)
parser.add_argument("--render", action="store_true")
parser.add_argument("--res", type=int, nargs=2, default=[960, 540])
parser.add_argument("--cam", type=float, nargs=3, default=[0.55, -0.55, 0.30])
parser.add_argument("--out_dir", type=str, default="/tmp")
parser.add_argument("--save", action="store_true", help="write trajectory_pyroki_objaware.npz")
args = parser.parse_args()

app_launcher = AppLauncher(headless=True, enable_cameras=args.render)
sim_app = app_launcher.app

import copy  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

from robotis_sh5.tasks.direct.g1_shadow_locomanip.g1_shadow_locomanip_env_cfg import G1_SHADOW_CFG  # noqa: E402

_PROC = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/processed/parahome"
SRC_FPS, TGT_FPS = 30.0, 50.0
dev = "cuda:0"


def _lerp(x, ts, tt):
    return np.stack([np.interp(tt, ts, x[:, d]) for d in range(x.shape[1])], axis=1).astype(np.float32)


def _slerp(q, ts, tt):
    from scipy.spatial.transform import Rotation, Slerp
    r = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    return Slerp(ts, r)(tt).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)


def load(clip, cls):
    g = os.path.join(_PROC, "g1_shadow", cls, clip, "0")
    s = os.path.join(_PROC, "smplx", cls, clip, "0")
    rt = np.load(os.path.join(g, "trajectory_pyroki.npz"), allow_pickle=True)
    tj = np.load(os.path.join(s, "trajectory.npz"), allow_pickle=True)
    q, rp = rt["g1_joint_pos"].astype(np.float32), rt["g1_root_pose"].astype(np.float32)
    F = q.shape[0]
    ts, tt = np.arange(F) / SRC_FPS, np.linspace(0.0, (F - 1) / SRC_FPS, int(round((F - 1) / SRC_FPS * TGT_FPS)) + 1)
    ctx = {k.split("__")[1]: tj[k][0].astype(np.float32)
           for k in tj.files if k.startswith("ctx__") and k.endswith("__base")}
    ok = [k for k in tj.files if k.startswith("obj__") and k.endswith("__base")]
    ob = tj[ok[0]].astype(np.float32) if ok else None
    return dict(
        names=[str(x) for x in rt["joint_names"]], src_dir=g,
        q=_lerp(q, ts, tt), root_p=_lerp(rp[:, :3], ts, tt), root_q=_slerp(rp[:, 3:7], ts, tt),
        ctx=ctx, obj_name=ok[0].split("__")[1] if ok else None,
        obj_p=_lerp(ob[:, :3], ts, tt) if ok else None,
        obj_q=_slerp(ob[:, 3:7], ts, tt) if ok else None,
        N=len(tt))


def main():
    C = load(args.clip, args.cls)
    N, W, H = C["N"], *args.res
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=dev))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    cfg = copy.deepcopy(G1_SHADOW_CFG)
    cfg.prim_path = "/World/Robot"
    cfg.spawn.articulation_props.fix_root_link = True      # the whole point; see the module note
    robot = Articulation(cfg)

    obj = None
    if C["obj_name"]:
        usd = os.path.join(_PROC, "assets", "objects", C["obj_name"], f"{C['obj_name']}.usd")
        if os.path.exists(usd):
            obj = RigidObject(RigidObjectCfg(
                prim_path="/World/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd, rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0))))
    # context: same treatment as the env — <name>_ctx.usd, kinematic_enabled so it is an immovable
    # collider. Without these the fingers have only the knife to collide with, and the penetration
    # against the cutting board / counter that the retarget also contains goes unresolved.
    n_ctx = 0
    for i, (nm, pose0) in enumerate(sorted(C["ctx"].items())):
        base = os.path.join(_PROC, "assets", "objects", nm)
        u = os.path.join(base, f"{nm}_ctx.usd")
        u = u if os.path.exists(u) else os.path.join(base, f"{nm}.usd")
        if not os.path.exists(u):
            continue
        cs = sim_utils.UsdFileCfg(
            usd_path=u, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True))
        cs.func(f"/World/Ctx_{i}_{nm}", cs,
                translation=tuple(float(x) for x in pose0[:3]),
                orientation=tuple(float(x) for x in pose0[3:7]))
        n_ctx += 1
    print(f"[ctx] spawned {n_ctx}/{len(C['ctx'])} context objects (kinematic)")

    cam = Camera(CameraCfg(prim_path="/World/Cam", height=H, width=W, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.02, 40.0)))) \
        if args.render else None
    sim.reset()

    jn = list(robot.joint_names)
    aidx = torch.tensor([jn.index(n) for n in C["names"]], device=dev)          # retarget col -> robot joint
    hand = torch.tensor([i for i, n in enumerate(C["names"]) if n.startswith("robot0_")], device=dev)
    q = torch.from_numpy(C["q"]).to(dev)
    rp = torch.from_numpy(C["root_p"]).to(dev)
    rq = torch.from_numpy(C["root_q"]).to(dev)
    op = torch.from_numpy(C["obj_p"]).to(dev) if obj is not None else None
    oq = torch.from_numpy(C["obj_q"]).to(dev) if obj is not None else None
    z6 = torch.zeros(1, 6, device=dev)
    dt = sim.get_physics_dt()

    achieved = torch.zeros(N, len(C["names"]), device=dev)
    jvel_end = np.zeros(N, np.float32)

    def place(f, jq):
        pose = torch.zeros(1, 7, device=dev)
        pose[0, :3] = rp[f]
        pose[0, 3:7] = rq[f]
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(z6)
        full = robot.data.default_joint_pos.clone()
        full[0, aidx] = jq
        robot.write_joint_state_to_sim(full, torch.zeros_like(full))
        if obj is not None:
            o = torch.zeros(1, 7, device=dev)
            o[0, :3] = op[f]
            o[0, 3:7] = oq[f]
            obj.write_root_pose_to_sim(o)
            obj.write_root_velocity_to_sim(z6)
        return full

    for f in range(N):
        full = place(f, q[f])
        robot.set_joint_position_target(full)
        for _ in range(args.settle_steps):
            if obj is not None:                     # keep the object fixated; it must not be pushed away
                o = torch.zeros(1, 7, device=dev)
                o[0, :3] = op[f]
                o[0, 3:7] = oq[f]
                obj.write_root_pose_to_sim(o)
                obj.write_root_velocity_to_sim(z6)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            if obj is not None:
                obj.update(dt)
        achieved[f] = robot.data.joint_pos[0, aidx]
        jvel_end[f] = float(robot.data.joint_vel[0, aidx].abs().max())
        if f % 50 == 0:
            print(f"  frame {f:4d}/{N}  |dev| {float((achieved[f]-q[f]).abs().mean()):.4f} rad  "
                  f"jvel_end {jvel_end[f]:.3f}")

    ach = achieved.cpu().numpy()
    ref = C["q"]
    d = np.abs(ach - ref)
    hi = hand.cpu().numpy()
    bi = np.array([i for i in range(len(C["names"])) if i not in set(hi.tolist())])
    print(f"\n=== {args.clip}: {N} frames, {args.settle_steps} settle steps, FIXED base ===")
    print(f"joint deviation |achieved - retarget| (rad)")
    print(f"  all      mean {d.mean():.4f}  p95 {np.percentile(d,95):.4f}  max {d.max():.4f}")
    print(f"  hands    mean {d[:,hi].mean():.4f}  p95 {np.percentile(d[:,hi],95):.4f}  max {d[:,hi].max():.4f}")
    print(f"  body     mean {d[:,bi].mean():.4f}  max {d[:,bi].max():.4f}")
    print(f"settled?  joint speed at end of settle: mean {jvel_end.mean():.3f}  max {jvel_end.max():.3f} rad/s")

    np.savez(os.path.join(args.out_dir, f"objaware_{args.clip}.npz"), achieved=ach, ref=ref, jvel_end=jvel_end)
    if args.save:
        p = os.path.join(C["src_dir"], "trajectory_pyroki_objaware.npz")
        # stored at the SOURCE rate so it drops in for trajectory_pyroki.npz; resample back by picking
        # the nearest 50 Hz sample for each 30 Hz frame
        F0 = ref.shape[0] if ref.shape[0] == N else None
        idx = np.round(np.linspace(0, N - 1, int(round((N - 1) * SRC_FPS / TGT_FPS)) + 1)).astype(int)
        np.savez(p, g1_joint_pos=ach[idx], joint_names=np.array(C["names"], dtype=object))
        print(f"wrote {p}  ({len(idx)} frames @ {SRC_FPS} fps)")

    if args.render and cam is not None:
        import imageio
        for tag, src in (("raw", torch.from_numpy(ref).to(dev)), ("objaware", achieved)):
            out = os.path.join(args.out_dir, f"objaware_{args.clip}_{tag}.mp4")
            wr = imageio.get_writer(out, fps=50, macro_block_size=1)
            for f in range(N):
                place(f, src[f])
                robot.write_data_to_sim()
                robot.update(dt)
                tgt = op[f] if obj is not None else rp[f]
                cam.set_world_poses_from_view((tgt + torch.tensor(args.cam, device=dev)).unsqueeze(0),
                                              tgt.unsqueeze(0))
                sim.render()
                cam.update(dt)
                wr.append_data(cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
            wr.close()
            print(f"video -> {out}")


main()
sim_app.close()
