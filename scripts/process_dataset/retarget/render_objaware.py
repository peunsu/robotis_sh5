"""Kinematic playback of an object-aware post-processed retarget, raw vs solved.

Takes the npz written by object_aware_parallel.py and replays both joint trajectories with the object
at its reference pose, so the hand-object penetration before and after can be compared directly.
Fixed base and no settling — this only draws poses, it does not re-solve anything.

    python scripts/process_dataset/retarget/render_objaware.py --npz <trace.npz> --clip s101_seg12_knife
"""
import argparse
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--clip", default="s101_seg12_knife")
ap.add_argument("--cls", default="single_rigid")
ap.add_argument("--res", type=int, nargs=2, default=[960, 540])
ap.add_argument("--cam", type=float, nargs=3, default=[0.45, -0.45, 0.22])
ap.add_argument("--out_dir", default="/tmp")
ap.add_argument("--ctx", action="store_true", help="also draw the context objects")
args = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=True).app

import copy, os                                                   # noqa: E402
import numpy as np, torch                                         # noqa: E402
import isaaclab.sim as sim_utils                                  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg   # noqa: E402
from isaaclab.sensors import Camera, CameraCfg                     # noqa: E402
from robotis_sh5.tasks.direct.g1_shadow_locomanip.g1_shadow_locomanip_env_cfg import G1_SHADOW_CFG  # noqa: E402

P = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/processed/parahome"
dev = "cuda:0"
SRC, TGT = 30.0, 50.0


def lerp(x, ts, tt):
    return np.stack([np.interp(tt, ts, x[:, d]) for d in range(x.shape[1])], 1).astype(np.float32)


def slerp(q, ts, tt):
    from scipy.spatial.transform import Rotation, Slerp
    return Slerp(ts, Rotation.from_quat(q[:, [1, 2, 3, 0]]))(tt).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)


D = np.load(args.npz)
ach, ref = D["achieved"], D["ref"]
g = os.path.join(P, "g1_shadow", args.cls, args.clip, "0")
s = os.path.join(P, "smplx", args.cls, args.clip, "0")
rt = np.load(os.path.join(g, "trajectory_pyroki.npz"), allow_pickle=True)
tj = np.load(os.path.join(s, "trajectory.npz"), allow_pickle=True)
# the trace carries the ACTION-joint order its columns are in; trajectory_pyroki.npz's joint_names is
# a DIFFERENT permutation (the env remaps 24/65 slots on load) and using it scrambles the fingers.
if "joint_names" in D.files:
    names = [str(x) for x in D["joint_names"]]
else:
    raise SystemExit("trace has no joint_names — re-run object_aware_parallel.py to regenerate it")
F0 = rt["g1_root_pose"].shape[0]
ts, tt = np.arange(F0) / SRC, np.linspace(0, (F0 - 1) / SRC, ach.shape[0])
root_p, root_q = lerp(rt["g1_root_pose"][:, :3], ts, tt), slerp(rt["g1_root_pose"][:, 3:7], ts, tt)
ok = [k for k in tj.files if k.startswith("obj__") and k.endswith("__base")]
obj_p = lerp(tj[ok[0]][:, :3], ts, tt) if ok else None
obj_q = slerp(tj[ok[0]][:, 3:7], ts, tt) if ok else None
oname = ok[0].split("__")[1] if ok else None

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=dev))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
cfg = copy.deepcopy(G1_SHADOW_CFG); cfg.prim_path = "/World/Robot"
cfg.spawn.articulation_props.fix_root_link = True
robot = Articulation(cfg)
obj = None
if oname:
    u = os.path.join(P, "assets", "objects", oname, f"{oname}.usd")
    obj = RigidObject(RigidObjectCfg(prim_path="/World/Object", spawn=sim_utils.UsdFileCfg(
        usd_path=u, rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 1))))
if args.ctx:
    for i, k in enumerate(sorted(x for x in tj.files if x.startswith("ctx__"))):
        nm = k.split("__")[1]
        b = os.path.join(P, "assets", "objects", nm)
        u = os.path.join(b, f"{nm}_ctx.usd")
        u = u if os.path.exists(u) else os.path.join(b, f"{nm}.usd")
        if not os.path.exists(u):
            continue
        c = sim_utils.UsdFileCfg(usd_path=u, activate_contact_sensors=False,
                                 rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))
        p0 = tj[k][0]
        c.func(f"/World/Ctx_{i}_{nm}", c, translation=tuple(map(float, p0[:3])),
               orientation=tuple(map(float, p0[3:7])))
W, H = args.res
cam = Camera(CameraCfg(prim_path="/World/Cam", height=H, width=W, data_types=["rgb"],
                       spawn=sim_utils.PinholeCameraCfg(focal_length=28.0, clipping_range=(0.02, 40.0))))
sim.reset()
jn = list(robot.joint_names)
aidx = torch.tensor([jn.index(n) for n in names], device=dev)
z6 = torch.zeros(1, 6, device=dev); dt = sim.get_physics_dt()
rp_t = torch.from_numpy(root_p).to(dev); rq_t = torch.from_numpy(root_q).to(dev)
op_t = torch.from_numpy(obj_p).to(dev) if obj is not None else None
oq_t = torch.from_numpy(obj_q).to(dev) if obj is not None else None

import imageio                                                     # noqa: E402
for tag, src in (("raw", torch.from_numpy(ref).to(dev)), ("objaware", torch.from_numpy(ach).to(dev))):
    out = os.path.join(args.out_dir, f"oa_{args.clip}_{tag}.mp4")
    wr = imageio.get_writer(out, fps=50, macro_block_size=1)
    for f in range(ach.shape[0]):
        pose = torch.zeros(1, 7, device=dev); pose[0, :3] = rp_t[f]; pose[0, 3:7] = rq_t[f]
        full = robot.data.default_joint_pos.clone(); full[0, aidx] = src[f]
        robot.write_root_pose_to_sim(pose); robot.write_root_velocity_to_sim(z6)
        robot.write_joint_state_to_sim(full, torch.zeros_like(full))
        if obj is not None:
            o = torch.zeros(1, 7, device=dev); o[0, :3] = op_t[f]; o[0, 3:7] = oq_t[f]
            obj.write_root_pose_to_sim(o); obj.write_root_velocity_to_sim(z6)
        robot.write_data_to_sim(); robot.update(dt)
        t = op_t[f] if obj is not None else rp_t[f]
        cam.set_world_poses_from_view((t + torch.tensor(args.cam, device=dev)).unsqueeze(0), t.unsqueeze(0))
        sim.render(); cam.update(dt)
        wr.append_data(cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
    wr.close(); print(f"video -> {out}")
app.close()
