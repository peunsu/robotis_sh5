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
parser.add_argument("--no-kpts", action="store_true", help="disable the human(green)/robot(cyan) keypoint overlay")
parser.add_argument("--cam_yaw", type=float, default=45.0, help="camera azimuth around the look target (deg; 45 = default +X+Y)")
parser.add_argument("--cam_elev", type=float, default=None, help="camera elevation (deg); default derives from vertical extent")
parser.add_argument("--cam_dist_scale", type=float, default=1.0, help="pull-back multiplier (>1 = further)")
parser.add_argument("--look_obj", action="store_true", help="aim at the object centroid (where the hands work) instead of the root")
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
    # human reference keypoints (14 body + 40 hand) for the GREEN overlay
    jp_kp = sm["joint_positions"].astype(np.float32) if not args.no_kpts else None
    ref_idx = list(BODY_KPTS.keys())
    for _off in (23, 48):
        for _spec in HAND_CHAIN.values():
            ref_idx += [_off + p for p in _spec["parahome"]]
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

    cam = Camera(CameraCfg(
        prim_path="/World/Camera", height=H, width=W, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.05, 50.0))))

    sim.reset()

    # action-joint name → articulation joint index
    jn = robot.joint_names
    aid = torch.tensor([jn.index(n) for n in act_names], dtype=torch.long, device=sim.device)
    default_q = robot.data.default_joint_pos.clone()

    # keypoint overlay: GREEN = human reference (54). Robot pose is shown by its mesh (no cyan → no
    # per-frame FK refresh needed, keeps the render loop the same as the proven original).
    m_human = None
    if not args.no_kpts:
        m_human = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/human", markers={
            "s": sim_utils.SphereCfg(radius=0.018,
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
    off = max(1.5, extent * 1.25) * args.cam_dist_scale          # pull back to fit the vertical extent
    horiz = off * (2 ** 0.5)                                     # horizontal cam distance (yaw=45 → default +X+Y)
    az = math.radians(args.cam_yaw)
    zoff = (0.12 * extent) if args.cam_elev is None else horiz * math.tan(math.radians(args.cam_elev))
    # look target: object centroid (where the hands work → un-occludes hands hidden behind the torso) or root
    txy = obj_base[:, :2].mean(0).astype(np.float32) if (args.look_obj and obj_base is not None) else c[:2]
    eye = torch.tensor([[float(txy[0]) + horiz * math.cos(az), float(txy[1]) + horiz * math.sin(az),
                         lookat_z + zoff]], device=sim.device, dtype=torch.float32)
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
