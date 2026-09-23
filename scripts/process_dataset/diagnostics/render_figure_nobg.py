"""Render a retargeted clip with NO BACKGROUND for paper figures: only the G1+Shadow robot and the
manipulated object are drawn — no ground plane, no context objects (ctx__*), no keypoint overlay.

Background: the dome light is what the camera sees behind the robot, so `--dome_bg` (default on)
leaves it visible and its intensity is the background brightness (3000 → ~248/255 white). Scene
shading comes from a separate sun + rim DistantLight pair, so you can raise contrast on the white
G1 shell without darkening the backdrop. `--dome_bg` off gives a black backdrop instead (the RTX
`backgroundZeroAlpha` alpha matte is NOT honoured by the rgba annotator on this build — checked).

Frame numbers are burned into the mp4 so you can scrub it and pick the frame for the figure;
`--png_dir` additionally dumps the same frames as unlabelled PNGs (those go into the paper).

Framing is explicit: `--cam_yaw`/`--cam_elev` place the camera on a sphere around the look target
and `--fill_frac` solves the distance so the subject fills that fraction of the frame height.
The robot faces -Y in the ParaHome clips (root yaw ≈ -90°), so `--cam_yaw -90` is a frontal view
and -125 / -55 are the two 3/4 views.

    DISPLAY=:1 python -u scripts/process_dataset/diagnostics/render_figure_nobg.py \
        --clip s101_seg12_knife --variant pyroki --end 120 --fps 10 \
        --cam_yaw -125 --png_dir /tmp/fig_knife_png
→ writes data/processed/parahome/g1_shadow/<class>/<clip>/0/figure_nobg.mp4

`--probe_yaws -160,-125,-90,-55` renders one frame from several azimuths (full body + close-up) and
exits — one Isaac start-up instead of four when you are still choosing the camera.

DISPLAY: this machine's desktop session is :1 and the Claude shell has an empty DISPLAY; without it
Vulkan device creation fails ("No device could be created") and the process spins at 100% CPU.
Always run with `python -u` too: the script ends with os._exit(0), which does not flush stdio.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="s101_seg12_knife")
parser.add_argument("--class", dest="cls", default="single_rigid")
parser.add_argument("--variant", default="pyroki", help="'' → trajectory.npz; 'pyroki' → trajectory_pyroki.npz")
parser.add_argument("--retarget-file", dest="retarget_file", default="",
                    help="explicit npz name under the clip dir (overrides --variant)")
parser.add_argument("--traj_npz", default="",
                    help="absolute npz path to replay instead of the clip's retarget file — e.g. a "
                         "POLICY ROLLOUT converted from rollout.py --dump_joints (needs g1_joint_pos, "
                         "g1_root_pose, joint_names; ParaHome world frame, env origin removed)")
parser.add_argument("--hands_npz", default="",
                    help="render the STAGE-1 floating Shadow hands from a hand_pretrain rollout "
                         "(hand_traj_best.npz): the G1 is not spawned, the two shadow_float6 "
                         "articulations are, and joint_pos_all (6 wrist DOF + 22 finger joints per "
                         "hand, world frame via the fixed anchor) drives them. The object pose comes "
                         "from the same npz.")
parser.add_argument("--obj_from_traj", action="store_true",
                    help="take the object pose from the traj npz's obj_base (the physically grasped "
                         "pose) instead of the SMPL-X reference trajectory")
# frame window
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1, help="exclusive; -1 = end of clip")
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--frames", default="",
                    help="explicit comma-separated frame indices, overriding --start/--end/--step. Use "
                         "it when a 50 Hz rollout and a 30 Hz reference must land on the same instants "
                         "and the two index sets cannot both be uniform.")
parser.add_argument("--src_fps", type=float, default=30.0, help="source frame rate (time label only)")
# output
parser.add_argument("--res", type=int, nargs=2, default=[1080, 1350])
parser.add_argument("--fps", type=int, default=10, help="playback fps of the mp4 (10 = 3x slow, easy to scrub)")
parser.add_argument("--out", default="")
parser.add_argument("--png_dir", default="", help="also dump the frames as unlabelled PNGs")
parser.add_argument("--png_every", type=int, default=1, help="dump every N-th frame as PNG")
# background / lights
parser.add_argument("--floor_radius", type=float, default=0.0,
                    help="draw a matte disc under the subject (m). 0 = no floor. A disc beats an "
                         "infinite plane for a cut-out figure: it gives the contact/scale cue and "
                         "catches the key light's shadow without running into the frame edges.")
parser.add_argument("--floor_color", type=float, nargs=3, default=[0.70, 0.71, 0.75],
                    help="floor albedo — a slightly COOL light gray by default so it separates from "
                         "both the white page and the warm object without competing with either")
parser.add_argument("--floor_z", type=float, default=0.0, help="floor top surface height (m)")
parser.add_argument("--floor_segments", type=int, default=160,
                    help="disc rim tessellation. isaaclab's CylinderCfg mesh is coarse enough that a "
                         "1-2 m disc shows a polygon rim, so the disc is generated here instead.")
parser.add_argument("--dome_bg", type=int, default=1, help="1 = dome light visible = bright backdrop; 0 = black")
parser.add_argument("--bg", default="white",
                    help="colour the mp4 composites the matte over: 'white' | 'black' | 'gray' | 'R,G,B'")
parser.add_argument("--alpha_matte", action="store_true",
                    help="真 transparency: render every frame twice — dome light visible (bright backdrop) "
                         "and hidden (black) — and solve the difference matte. Illumination is identical "
                         "between the two passes because visibleInPrimaryRay only affects primary rays, so "
                         "alpha = 1 - (I_bright - I_black)/(B_bright - B_black) is exact, edges included. "
                         "PNGs then carry real alpha; the mp4 still composites over --bg.")
parser.add_argument("--rtx_alpha", action="store_true",
                    help="take alpha straight from the RTX background-zero-alpha pass (one render per "
                         "frame). Needs outputAlphaInComposite, which the default settings leave off — "
                         "that is why the earlier attempt saw alpha=255 everywhere.")
parser.add_argument("--aa", type=int, default=-1,
                    help="/rtx/post/aa/op: 2 = FXAA (spatial only). DLSS (3, the default) carries "
                         "temporal history and its own auto-exposure between the two matte passes, "
                         "which shows up as a ghost-transparent subject on tight crops.")
parser.add_argument("--matte_settle", type=int, default=6,
                    help="renders after each backdrop toggle before grabbing (RTX temporal history)")
parser.add_argument("--dome", type=float, default=3000.0, help="dome intensity = backdrop brightness")
parser.add_argument("--sun", type=float, default=4500.0, help="key DistantLight intensity (shading contrast)")
parser.add_argument("--rim", type=float, default=1200.0, help="opposite-side DistantLight intensity")
parser.add_argument("--fill", type=float, default=0.0,
                    help="fill DistantLight from the camera azimuth at a low elevation. Use this when a "
                         "floor/scene makes the subject read too dark: at 15 deg it hits the subject "
                         "head-on but grazes the floor (cos 75 = 0.26), so the subject brightens ~4x "
                         "more than the ground and the object colour barely moves.")
# label
parser.add_argument("--no_label", action="store_true")
parser.add_argument("--label_scale", type=float, default=1.0)
parser.add_argument("--no_time_label", action="store_true")
# camera
parser.add_argument("--cam_yaw", type=float, default=-125.0, help="azimuth of the camera around the target (deg)")
parser.add_argument("--yaw_rel", type=float, default=None,
                    help="orbit with the robot: camera azimuth = root yaw + this (deg), recomputed every "
                         "frame, and the key/rim lights follow. -35 reproduces the default 3/4 view. Use "
                         "this on clips where the robot turns (s100_seg00_pan turns ~255 deg in place).")
parser.add_argument("--cam_elev", type=float, default=8.0, help="elevation of the camera (deg)")
parser.add_argument("--fill_frac", type=float, default=0.88, help="fraction of the frame height the subject fills")
parser.add_argument("--closeup", type=float, default=0.0,
                    help="if >0, frame a sphere of this radius (m) around the object instead of the whole robot")
parser.add_argument("--closeup_mix", type=float, default=0.45,
                    help="close-up look target between the object (0) and the robot root (1) in xy — "
                         "at 0 the torso is cut off on the frames where the hands are not on the object yet")
parser.add_argument("--closeup_dz", type=float, default=0.08, help="raise the close-up look target (m)")
parser.add_argument("--follow_obj", action="store_true", help="re-aim every frame at the object (tight close-ups)")
parser.add_argument("--follow_root", action="store_true",
                    help="whole-body view: re-aim every frame at the robot root instead of its mean xy")
parser.add_argument("--lookat_z", type=float, default=None, help="override look-at height (m)")
parser.add_argument("--full_xy", default="",
                    help="'x,y' — pin the whole-body look target instead of deriving it from the "
                         "trajectory's root, so a rollout replay keeps a reference render's camera")
parser.add_argument("--full_ext", type=float, default=0.0,
                    help="pin the whole-body vertical extent (m); same purpose as --full_xy")
parser.add_argument("--focal", type=float, default=24.0)
parser.add_argument("--autofit_frames", type=int, default=5,
                    help="autofit samples this many frames across the window (plus the frame where the "
                         "object is highest) and fits the WORST case, so a lifted object never leaves "
                         "the top of a clip-fixed frame")
parser.add_argument("--autofit", type=int, default=1,
                    help="measure the rendered silhouette and correct distance/centring (the analytic "
                         "FOV of this build runs ~25%% wide); also calibrates the close-up distance")
parser.add_argument("--probe_lights", default="",
                    help="'dome,sun,rim;dome,sun,rim;...' — render the probe frame once per lighting "
                         "setting and exit. The tone map compresses highlights, so a bright light "
                         "level washes a flat diffuse material out to pastel; this is how you find "
                         "the level where the material colour still reads.")
parser.add_argument("--probe_yaws", default="", help="comma-separated azimuths: render one frame each, then exit")
parser.add_argument("--probe_frame", type=int, default=-1, help="frame for --probe_yaws (default: window centre)")
parser.add_argument("--probe_closeup", type=float, default=0.30, help="close-up radius used by the probe")
# object
parser.add_argument("--no_obj", action="store_true")
parser.add_argument("--obj_color", type=float, nargs=3, default=[1.0, 0.35, 0.0])
parser.add_argument("--obj_native", action="store_true", help="keep the object's own USD material")
parser.add_argument("--no_robot", action="store_true",
                    help="hide the G1+Shadow robot from the render (for a human-only figure). The "
                         "articulation is still spawned and posed: a physics scene with no "
                         "articulation crashes the GPU pipeline (CUDA illegal memory access on the "
                         "first camera pose write), so hiding beats not spawning.")
parser.add_argument("--human_usd", default="",
                    help="also spawn this mesh as a static prim at identity — e.g. the SMPL-X body of "
                         "the same frame from parahome_make_smplx_mesh.py (already in world coords)")
parser.add_argument("--human_color", type=float, nargs=3, default=[0.72, 0.74, 0.80])
parser.add_argument("--ctx", default="",
                    help="scene context objects: 'name:r,g,b;name2:r,g,b' — spawned static at their "
                         "frame-0 pose from the clip's ctx__<name>__base, using assets/objects/<name>/"
                         "ctx/<name>_ctx.usd. Colours are yours to pick; keep them desaturated so the "
                         "manipulated object stays the eye's target.")
parser.add_argument("--aim_xyz", default="",
                    help="'x,y,z' — absolute close-up look target, overriding the object/robot mix")
parser.add_argument("--aim_offset", default="",
                    help="'dx,dy,dz' (m, world) added to the FOLLOWED object position. Use it to give a "
                         "tracking video the same relative framing as a still that was aimed with "
                         "--aim_xyz: pass aim_xyz(f) - obj(f) of that frame. Ignored when --aim_xyz "
                         "is set (that one pins the target).")
parser.add_argument("--ref_map", default="",
                    help="'a,b' — also print the reference-clip frame round(a*idx+b) in the time label, "
                         "for replaying a 50 Hz rollout against 30 Hz reference frame numbers")
parser.add_argument("--cal", default="",
                    help="'d,r,u' — reuse another run's autofit result (skips autofit) so two renders "
                         "share the exact same camera")
parser.add_argument("--obj_usd", default="",
                    help="render a DIFFERENT USD for the manipulated object (same local frame) — e.g. the "
                         "watertight visual mesh from parahome_make_render_mesh.py, which closes the torn "
                         "scan shell. Physics assets are untouched; this is render-only.")
# render quality
parser.add_argument("--warmup", type=int, default=24, help="render calls before frame 0 (RTX convergence)")
parser.add_argument("--subframes", type=int, default=2, help="render calls per frame")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.math import quat_from_euler_xyz  # noqa: E402
from robotis_sh5.tasks.direct.g1_shadow_locomanip.g1_shadow_locomanip_env_cfg import G1_SHADOW_CFG  # noqa: E402

_PROC = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/processed/parahome"
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_H_APERTURE = 20.955          # isaaclab PinholeCameraCfg default; vertical = this * H/W


def _parse_bg(t):
    named = {"white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128), "grey": (128, 128, 128)}
    if t in named:
        return np.array(named[t], dtype=np.float32)
    v = [float(x) for x in t.replace(" ", "").split(",")]
    if len(v) != 3:
        raise SystemExit(f"--bg must be a name or 'R,G,B': got {t!r}")
    return np.array(v, dtype=np.float32)


def main():
    bg = _parse_bg(args.bg)
    import carb
    _st = carb.settings.get_settings()
    if args.aa >= 0:
        _st.set("/rtx/post/aa/op", int(args.aa))
        print(f"[fig] aa/op = {_st.get('/rtx/post/aa/op')}", flush=True)
    if args.rtx_alpha:
        for k, v in (("enabled", True), ("backgroundComposite", False),
                     ("outputAlphaInComposite", True), ("premultiplyColorByAlpha", False)):
            _st.set(f"/rtx/post/backgroundZeroAlpha/{k}", v)
        print("[fig] backgroundZeroAlpha: " + ", ".join(
            f"{k}={_st.get('/rtx/post/backgroundZeroAlpha/' + k)}"
            for k in ("enabled", "backgroundComposite", "outputAlphaInComposite")), flush=True)
    _rt_name = args.retarget_file or ("trajectory_pyroki.npz" if args.variant == "pyroki" else "trajectory.npz")
    _rt_path = args.traj_npz or os.path.join(_PROC, "g1_shadow", args.cls, args.clip, "0", _rt_name)
    rt = np.load(_rt_path, allow_pickle=True)
    if args.traj_npz:
        _rt_name = os.path.basename(_rt_path)
        print(f"[fig] trajectory npz: {_rt_path}")
    jpos = rt["g1_joint_pos"].astype(np.float32)      # (F,65|73) action-joint order
    root = rt["g1_root_pose"].astype(np.float32)      # (F,7) pos + quat wxyz
    F = jpos.shape[0]

    sm = np.load(os.path.join(_PROC, "smplx", args.cls, args.clip, "0", "trajectory.npz"), allow_pickle=True)
    base_keys = [k for k in sm.files if k.startswith("obj__") and k.endswith("__base")]
    obj_name = base_keys[0].split("__")[1] if base_keys else ""
    obj_base = sm[f"obj__{obj_name}__base"].astype(np.float32) if (obj_name and not args.no_obj) else None
    if args.obj_from_traj and not args.no_obj:
        if "obj_base" not in rt.files:
            raise SystemExit("[fig] --obj_from_traj: traj npz 에 obj_base 가 없습니다")
        obj_base = rt["obj_base"].astype(np.float32)
        print(f"[fig] object pose from the trajectory npz ({obj_base.shape[0]} frames)")

    hd = None
    if args.hands_npz:
        hd = np.load(args.hands_npz, allow_pickle=True)
        F = int(hd["joint_pos_all"].shape[0])
        if not args.no_obj and "obj_pos" in hd.files:
            obj_base = np.concatenate([hd["obj_pos"], hd["obj_quat"]], 1).astype(np.float32)
        print(f"[fig] stage-1 hands: {args.hands_npz}  F={F}  "
              f"ckpt={os.path.basename(str(hd['checkpoint'])) if 'checkpoint' in hd.files else '?'}",
              flush=True)

    end = F if args.end < 0 else min(args.end, F)
    sel = ([int(v) for v in args.frames.split(",")] if args.frames
           else list(range(max(0, args.start), end, max(1, args.step))))
    sel = [f for f in sel if 0 <= f < F]
    if not sel:
        raise SystemExit(f"empty frame window: start={args.start} end={args.end} F={F}")
    print(f"[fig] clip={args.clip} file={_rt_name} F={F} obj={obj_name or '(none)'} "
          f"window={sel[0]}..{sel[-1]} step={args.step} → {len(sel)} frames", flush=True)

    W, H = args.res
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 30.0, device="cuda:0"))
    # NO GroundPlane: the figure shows the robot and the object only.
    dome = sim_utils.DomeLightCfg(intensity=args.dome, visible_in_primary_ray=bool(args.dome_bg))
    dome.func("/World/light", dome)

    def _light_quat(az_deg, el_deg):
        return quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([math.radians(90.0 - el_deg)]),
                                   torch.tensor([math.radians(az_deg)]))[0]

    def _sun(name, intensity, az_deg, el_deg):
        if intensity <= 0.0:
            return
        cfg = sim_utils.DistantLightCfg(intensity=float(intensity), angle=2.0)
        cfg.func(f"/World/{name}", cfg, orientation=tuple(float(v) for v in _light_quat(az_deg, el_deg)))

    # key light from the camera side (+35° azimuth so the shading is not flat), rim from behind
    _LIGHTS = (("key", args.sun, 35.0, 40.0), ("rim", args.rim, 180.0, 25.0),
               ("fill", args.fill, 0.0, 15.0))
    for _n, _i, _daz, _el in _LIGHTS:
        _sun(_n, _i, args.cam_yaw + _daz, _el)

    # ...and let them orbit with the camera when --yaw_rel is on, otherwise a turning robot spends
    # part of the clip lit from behind.
    def _relight(yaw):
        from pxr import Gf, UsdGeom
        import isaacsim.core.utils.stage as stage_utils
        stage = stage_utils.get_current_stage()
        for n, inten, daz, el in _LIGHTS:
            if inten <= 0.0:
                continue
            prim = stage.GetPrimAtPath(f"/World/{n}")
            if not prim.IsValid():
                continue
            q = _light_quat(yaw + daz, el)
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if "orient" not in op.GetOpName():
                    continue
                qv = [float(v) for v in q]
                op.Set(Gf.Quatd(qv[0], Gf.Vec3d(qv[1], qv[2], qv[3]))
                       if "quatd" in str(op.GetTypeName()).lower()
                       else Gf.Quatf(qv[0], Gf.Vec3f(qv[1], qv[2], qv[3])))

    if args.floor_radius > 0.0:
        _fxy = ([float(v) for v in args.full_xy.split(",")] if args.full_xy
                else [float(root[sel[len(sel) // 2], 0]), float(root[sel[len(sel) // 2], 1])])
        # generated flat disc (smooth rim) — see --floor_segments
        from pxr import UsdGeom as _UG, Vt as _Vt
        import isaacsim.core.utils.stage as _stg
        _N = max(16, int(args.floor_segments))
        _R = float(args.floor_radius)
        _th = np.linspace(0.0, 2.0 * np.pi, _N, endpoint=False)
        _pts = np.stack([np.cos(_th) * _R, np.sin(_th) * _R, np.zeros(_N)], 1)
        _pts = np.concatenate([np.zeros((1, 3)), _pts], 0)          # centre + rim
        _fi = np.stack([np.zeros(_N, int), np.arange(1, _N + 1),
                        np.roll(np.arange(1, _N + 1), -1)], 1)      # triangle fan
        _m = _UG.Mesh.Define(_stg.get_current_stage(), "/World/Floor")
        _m.CreatePointsAttr(_Vt.Vec3fArray.FromNumpy(
            (_pts + np.array([_fxy[0], _fxy[1], args.floor_z])).astype(np.float32)))
        _m.CreateFaceVertexIndicesAttr(_Vt.IntArray.FromNumpy(_fi.astype(np.int32).ravel()))
        _m.CreateFaceVertexCountsAttr(_Vt.IntArray.FromNumpy(np.full(_N, 3, dtype=np.int32)))
        _m.CreateNormalsAttr(_Vt.Vec3fArray.FromNumpy(
            np.tile(np.array([[0, 0, 1]], np.float32), (_N + 1, 1))))
        _m.SetNormalsInterpolation(_UG.Tokens.vertex)
        _m.CreateSubdivisionSchemeAttr(_UG.Tokens.none)
        _m.CreateDoubleSidedAttr(True)
        sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(float(c) for c in args.floor_color), roughness=0.95, metallic=0.0
        ).func("/World/FloorMat", sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(float(c) for c in args.floor_color), roughness=0.95, metallic=0.0))
        sim_utils.bind_visual_material("/World/Floor", "/World/FloorMat")
        print(f"[fig] floor disc r={args.floor_radius} m at ({_fxy[0]:.2f}, {_fxy[1]:.2f}, "
              f"{args.floor_z:.2f}) albedo {args.floor_color}", flush=True)

    robot = None if hd is not None else Articulation(G1_SHADOW_CFG.replace(prim_path="/World/Robot"))
    hands = None
    if hd is not None:
        from robotis_sh5.tasks.direct.g1_shadow_hand_pretrain.shadow_float_cfg import (  # noqa: E402
            SHADOW_FLOAT6_L_CFG, SHADOW_FLOAT6_R_CFG)
        hands = (Articulation(SHADOW_FLOAT6_L_CFG.replace(prim_path="/World/HandL")),
                 Articulation(SHADOW_FLOAT6_R_CFG.replace(prim_path="/World/HandR")))
        print("[fig] spawned shadow_float6 L/R (anchor fixed at the origin; joints carry the pose)",
              flush=True)

    for _spec in [t for t in args.ctx.split(";") if t.strip()]:
        _nm, _, _col = _spec.partition(":")
        _nm = _nm.strip()
        _key = f"ctx__{_nm}__base"
        if _key not in sm.files:
            print(f"[fig] ctx '{_nm}' 없음 — 건너뜀 (clip 에 {_key} 가 없다)")
            continue
        _base = os.path.join(_PROC, "assets", "objects", _nm)
        _u = next((q for q in (os.path.join(_base, "ctx", f"{_nm}_ctx.usd"),
                               os.path.join(_base, f"{_nm}_ctx.usd"),
                               os.path.join(_base, f"{_nm}.usd")) if os.path.exists(q)), None)
        if _u is None:
            print(f"[fig] ctx '{_nm}' USD 없음 — 건너뜀")
            continue
        _p0 = sm[_key][0].astype(np.float32)
        _cfg = sim_utils.UsdFileCfg(usd_path=_u, activate_contact_sensors=False)
        if _col.strip():
            _c = tuple(float(v) for v in _col.split(","))
            _cfg.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=_c, roughness=0.85)
        _cfg.func(f"/World/Ctx_{_nm}", _cfg, translation=tuple(float(v) for v in _p0[:3]),
                  orientation=tuple(float(v) for v in _p0[3:7]))
        print(f"[fig] ctx {_nm} @ {np.round(_p0[:3], 2)}  colour {_col or 'native'}", flush=True)

    human_prims = {}
    if args.human_usd:
        _many = "{f" in args.human_usd   # {f} or {f:04d}
        for _f in (sel if _many else [None]):
            _pth = args.human_usd.format(f=_f) if _many else args.human_usd
            if not os.path.exists(_pth):
                raise SystemExit(f"[fig] human USD 없음: {_pth}")
            hcfg = sim_utils.UsdFileCfg(
                usd_path=_pth,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(float(c) for c in args.human_color), roughness=0.55))
            _prim = f"/World/Human_{_f}" if _many else "/World/Human"
            hcfg.func(_prim, hcfg)                     # static: the mesh is already posed in world
            human_prims[_f] = _prim
        print(f"[fig] human mesh: {len(human_prims)} frame(s) from {args.human_usd}")

    obj = None
    if obj_base is not None:
        usd = args.obj_usd or os.path.join(_PROC, "assets", "objects", obj_name, f"{obj_name}.usd")
        if args.obj_usd:
            print(f"[fig] object USD override: {usd}")
        if not os.path.exists(usd):
            print(f"[fig] object USD 없음: {usd}")
        else:
            spawn = sim_utils.UsdFileCfg(
                usd_path=usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True))
            if not args.obj_native:
                spawn.visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(float(c) for c in args.obj_color), roughness=0.6)
            obj = RigidObject(RigidObjectCfg(
                prim_path="/World/Object", spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(float(v) for v in obj_base[0, :3]),
                    rot=tuple(float(v) for v in obj_base[0, 3:7]))))

    cam = Camera(CameraCfg(
        prim_path="/World/Camera", height=H, width=W, data_types=["rgba"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=args.focal, clipping_range=(0.02, 50.0))))

    sim.reset()

    # action-joint name → articulation joint index (name-based: the tendon-inequality retarget stores
    # 73 columns incl. the 8 free J0, a fixed 65-name list would mis-map it)
    if robot is None:
        aid = default_q = None
        hand_aid = [torch.tensor([h.joint_names.index(n) for n in
                                  [str(x) for x in hd[f"joint_names_all_{sd}"]]],
                                 dtype=torch.long, device=sim.device)
                    for h, sd in zip(hands, "lr")]
        print(f"[fig] hand joint maps: {[len(a) for a in hand_aid]} (of "
              f"{[len(h.joint_names) for h in hands]} articulation DOF)", flush=True)
    jn = robot.joint_names if robot is not None else []
    order = json.load(open("/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/robots/G1/g1_shadow_joint_order.json"))
    act_names = order["action_joint_names"]
    _npz_names = [str(x) for x in rt["joint_names"]] if "joint_names" in rt.files else None
    _use = _npz_names if (_npz_names and len(_npz_names) == jpos.shape[1]) else act_names
    if robot is not None:
        aid = torch.tensor([jn.index(n) for n in _use], dtype=torch.long, device=sim.device)
        default_q = robot.data.default_joint_pos.clone()

    if args.no_robot and robot is not None:           # keep the articulation, drop it from the frame
        from pxr import UsdGeom
        import isaacsim.core.utils.stage as stage_utils
        UsdGeom.Imageable(stage_utils.get_current_stage().GetPrimAtPath("/World/Robot")).MakeInvisible()
        print("[fig] robot hidden from the render", flush=True)
    if robot is not None:
        print(f"[fig] joint columns {jpos.shape[1]} "
              f"({'npz names' if _use is _npz_names else 'action order'})", flush=True)

    # ---- camera placement: explicit (azimuth, elevation, distance-from-fill) ----
    _vfov = 2.0 * math.atan((_H_APERTURE * H / W) * 0.5 / args.focal)
    _cal = {"d": 1.0, "r": 0.0, "u": 0.0}      # autofit: distance multiplier + target shift (m)

    def _basis(yaw):
        az, el = math.radians(yaw), math.radians(args.cam_elev)
        right = (-math.sin(az), math.cos(az), 0.0)
        up = (-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el))
        return az, el, right, up

    def _aim(cx, cy, cz, extent, yaw, fill):
        d = (extent / max(0.05, fill)) / (2.0 * math.tan(0.5 * _vfov)) * _cal["d"]
        az, el, right, up = _basis(yaw)
        cx += right[0] * _cal["r"] + up[0] * _cal["u"]
        cy += right[1] * _cal["r"] + up[1] * _cal["u"]
        cz += right[2] * _cal["r"] + up[2] * _cal["u"]
        eye = torch.tensor([[cx + d * math.cos(el) * math.cos(az), cy + d * math.cos(el) * math.sin(az),
                             cz + d * math.sin(el)]], device=sim.device, dtype=torch.float32)
        tgt = torch.tensor([[cx, cy, cz]], device=sim.device, dtype=torch.float32)
        cam.set_world_poses_from_view(eye, tgt)
        return d

    # whole-robot framing over the SELECTED window (not the whole clip)
    rsel = root[[i for i in sel if i < root.shape[0]] or [0]]
    tops = [float(rsel[:, 2].max()) + 0.78]                      # head ≈ root + 0.78
    if obj_base is not None:
        tops.append(float(obj_base[sel, 2].max()))
    if "fingertip_pad_pos" in sm.files:
        # the SMPL-X reference is the 30 Hz clip; a --traj_npz replay can be longer (50 Hz rollout),
        # so only the frames that exist in the reference contribute to the vertical framing
        _fsel = [i for i in sel if i < sm["fingertip_pad_pos"].shape[0]]
        if _fsel:
            tops.append(float(sm["fingertip_pad_pos"][_fsel][:, :, 2].max()))
    z_top, z_bot = max(tops) + 0.08, 0.0
    full_ext = args.full_ext if args.full_ext > 0.0 else z_top - z_bot
    full_z = float(args.lookat_z) if args.lookat_z is not None else 0.5 * (z_top + z_bot)
    full_xy = (np.array([float(v) for v in args.full_xy.split(",")], dtype=np.float32)
               if args.full_xy else rsel[:, :2].mean(0))
    obj_xy = obj_base[sel, :2].mean(0) if obj_base is not None else full_xy
    obj_z = float(obj_base[sel, 2].mean()) if obj_base is not None else full_z

    _root_yaw = np.degrees(np.arctan2(2.0 * (root[:, 3] * root[:, 6] + root[:, 4] * root[:, 5]),
                                      1.0 - 2.0 * (root[:, 5] ** 2 + root[:, 6] ** 2)))

    def _yaw_of(f):
        return args.cam_yaw if args.yaw_rel is None else float(_root_yaw[f]) + args.yaw_rel

    def _aim_frame(f, yaw=None, closeup=None):
        if yaw is None:
            yaw = _yaw_of(f)
            if args.yaw_rel is not None:
                _relight(yaw)
        closeup = args.closeup if closeup is None else closeup
        if closeup > 0.0:
            if args.follow_obj and obj_base is not None:
                c = obj_base[f, :3]
                cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
                rxy = root[f, :2]
            else:
                cx, cy, cz = float(obj_xy[0]), float(obj_xy[1]), obj_z
                rxy = full_xy
            if args.aim_xyz:
                cx, cy, cz = [float(v) for v in args.aim_xyz.split(",")]
            else:
                w = min(max(args.closeup_mix, 0.0), 1.0)  # slide the target toward the robot
                cx, cy = cx * (1.0 - w) + float(rxy[0]) * w, cy * (1.0 - w) + float(rxy[1]) * w
                cz += args.closeup_dz
                if args.aim_offset:
                    _o = [float(v) for v in args.aim_offset.split(",")]
                    cx, cy, cz = cx + _o[0], cy + _o[1], cz + _o[2]
            if args.lookat_z is not None:
                cz = float(args.lookat_z)
            return _aim(cx, cy, cz, 2.0 * closeup, yaw, args.fill_frac)
        fxy = root[f, :2] if args.follow_root else full_xy
        return _aim(float(fxy[0]), float(fxy[1]), full_z, full_ext, yaw, args.fill_frac)

    def _pose(f):
        if len(human_prims) > 1:
            from pxr import UsdGeom as _UG2
            import isaacsim.core.utils.stage as _stg2
            _st2 = _stg2.get_current_stage()
            for _k, _pp in human_prims.items():
                _im = _UG2.Imageable(_st2.GetPrimAtPath(_pp))
                (_im.MakeVisible() if _k == f else _im.MakeInvisible())
        if hands is not None:
            for h, a, side in zip(hands, hand_aid, range(2)):
                q = h.data.default_joint_pos.clone()
                q[:, a] = torch.from_numpy(hd["joint_pos_all"][f, side]).to(sim.device).unsqueeze(0)
                h.write_joint_state_to_sim(q, torch.zeros_like(q))
                h.write_data_to_sim()
            if obj is not None:
                obj.write_root_pose_to_sim(torch.from_numpy(obj_base[f]).to(sim.device).unsqueeze(0))
                obj.write_data_to_sim()
            return
        q = default_q.clone()
        q[:, aid] = torch.from_numpy(jpos[f]).to(sim.device).unsqueeze(0)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.write_root_pose_to_sim(torch.from_numpy(root[f]).to(sim.device).unsqueeze(0))
        robot.write_data_to_sim()
        if obj is not None:
            obj.write_root_pose_to_sim(torch.from_numpy(obj_base[f]).to(sim.device).unsqueeze(0))
            obj.write_data_to_sim()

    dt = sim.get_physics_dt()

    def _grab4(n=None):
        for _ in range(max(1, args.subframes if n is None else n)):
            sim.render()
        cam.update(dt)
        rgba = cam.data.output["rgba"][0].detach().cpu().numpy()
        if rgba.dtype != np.uint8:
            rgba = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgba

    def _grab(n=None):
        for _ in range(max(1, args.subframes if n is None else n)):
            sim.render()
        cam.update(dt)
        rgba = cam.data.output["rgba"][0].detach().cpu().numpy()
        if rgba.dtype != np.uint8:
            rgba = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgba[..., :3]

    def _dome_visible(flag):
        import isaacsim.core.utils.stage as stage_utils
        a = stage_utils.get_current_stage().GetPrimAtPath("/World/light").GetAttribute("visibleInPrimaryRay")
        if not a.IsValid():
            raise SystemExit("[fig] dome light has no visibleInPrimaryRay attribute — cannot matte")
        a.Set(bool(flag))

    def _grab_matte():
        """RGBA with a real alpha channel from a two-backdrop difference matte."""
        _dome_visible(False)
        ib = _grab(args.matte_settle).astype(np.float32)          # colour over black
        _dome_visible(True)
        iw = _grab(args.matte_settle).astype(np.float32)          # same, over the bright dome
        bb, bw = ib[2, 2].astype(np.float32), iw[2, 2].astype(np.float32)   # measured backdrops
        den = np.maximum(bw - bb, 1.0)
        a = 1.0 - np.clip((iw - ib) / den[None, None, :], 0.0, 1.0).mean(-1)     # (H,W) coverage
        a = np.clip(a, 0.0, 1.0)
        col = np.where(a[..., None] > 0.02, ib / np.maximum(a, 0.02)[..., None], 0.0)   # premult → straight
        return np.concatenate([np.clip(col, 0, 255), (a * 255.0)[..., None]], -1).astype(np.uint8)

    from PIL import Image, ImageDraw, ImageFont
    for _ in range(max(0, args.warmup)):
        sim.render()
    cam.update(dt)

    # ---- autofit: the analytic vFOV of this build renders ~25% wider than the aperture maths says,
    # so measure the silhouette and correct distance + centring instead of trusting the formula.
    def _bbox(rgb):
        bgv = rgb[2, 2].astype(np.int16)
        m = np.abs(rgb.astype(np.int16) - bgv[None, None, :]).max(-1) > 14
        if not m.any():
            return None
        ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        return int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1])

    if args.cal:
        _cal["d"], _cal["r"], _cal["u"] = [float(v) for v in args.cal.split(",")]
        print(f"[fig] camera calibration reused: d={_cal['d']:.4f} r={_cal['r']:+.4f} u={_cal['u']:+.4f}",
              flush=True)

    if args.autofit and not args.cal:
        if args.probe_frame >= 0:
            ff = [args.probe_frame]
        else:
            k = max(1, args.autofit_frames)
            ff = sorted({sel[min(len(sel) - 1, int(round(i * (len(sel) - 1) / max(1, k - 1))))]
                         for i in range(k)})
            if obj_base is not None:                   # the frame with the object at its highest
                ff = sorted(set(ff) | {sel[int(np.argmax(obj_base[sel, 2]))]})
        for it in range(3):
            # UNION of the sampled frames' silhouettes: what has to fit is the envelope of the whole
            # window, not the average frame — a pan lifted over the head only shows up in one frame.
            uni, n = None, 0
            for f0 in ff:
                _pose(f0)
                _aim(float(root[f0, 0] if args.follow_root else full_xy[0]),
                     float(root[f0, 1] if args.follow_root else full_xy[1]),
                     full_z, full_ext, _yaw_of(f0), args.fill_frac)
                bb = _bbox(_grab())
                if bb is None:
                    continue
                n += 1
                uni = bb if uni is None else (min(uni[0], bb[0]), max(uni[1], bb[1]),
                                              min(uni[2], bb[2]), max(uni[3], bb[3]))
            if uni is None:
                print("[fig] autofit: 실루엣 없음 — 건너뜀", flush=True)
                break
            x0, x1, y0, y1 = uni
            clipped = (y0 <= 1 or y1 >= H - 2 or x0 <= 1 or x1 >= W - 2)
            hpx = y1 - y0 + 1
            fh = hpx / H
            mpp = full_ext / max(1, hpx)                                   # metres per pixel
            dxp, dyp = 0.5 * (x0 + x1) - 0.5 * W, 0.5 * (y0 + y1) - 0.5 * H
            print(f"[fig] autofit {it}: union fill={fh:.3f} (target {args.fill_frac:.2f}) "
                  f"off=({dxp:+.0f},{dyp:+.0f})px clipped={clipped} over {n} frames", flush=True)
            _cal["d"] *= max(fh / args.fill_frac, 1.25 if clipped else 0.0)
            _cal["r"] += dxp * mpp
            _cal["u"] -= dyp * mpp
        if args.closeup > 0.0 or args.probe_yaws:
            _cal["r"] = _cal["u"] = 0.0               # close-up/probe centre on the geometric target
        print(f"[fig] autofit → distance x{_cal['d']:.3f}, target shift r={_cal['r']:+.3f} "
              f"u={_cal['u']:+.3f} m", flush=True)

    # ---- probe mode: one frame per lighting setting, then quit ----
    if args.probe_lights:
        import isaacsim.core.utils.stage as stage_utils
        pf = args.probe_frame if args.probe_frame >= 0 else sel[len(sel) // 2]
        pdir = args.png_dir or "/tmp/fig_probe"
        os.makedirs(pdir, exist_ok=True)
        _pose(pf)
        _aim_frame(pf)
        _stage = stage_utils.get_current_stage()

        def _set_i(path, val):
            a = _stage.GetPrimAtPath(path).GetAttribute("inputs:intensity")
            if a.IsValid():
                a.Set(float(val))

        for spec in args.probe_lights.split(";"):
            dv, sv, rv = [float(v) for v in spec.split(",")]
            _set_i("/World/light", dv)
            _set_i("/World/key", sv)
            _set_i("/World/rim", rv)
            rgba = _grab4(args.matte_settle) if args.rtx_alpha else _grab_matte()
            Image.fromarray(rgba, mode="RGBA").save(
                os.path.join(pdir, f"lights_{dv:.0f}_{sv:.0f}_{rv:.0f}.png"))
            print(f"[fig] probe lights dome={dv:.0f} sun={sv:.0f} rim={rv:.0f}", flush=True)
        print(f"[fig] light probes → {pdir}")
        os._exit(0)

    # ---- probe mode: one frame from several azimuths, then quit ----
    if args.probe_yaws:
        pf = args.probe_frame if args.probe_frame >= 0 else sel[len(sel) // 2]
        pdir = args.png_dir or "/tmp/fig_probe"
        os.makedirs(pdir, exist_ok=True)
        _pose(pf)
        for y in [float(v) for v in args.probe_yaws.split(",")]:
            for tag, cu in (("full", 0.0), ("close", args.probe_closeup)):
                _aim_frame(pf, yaw=y, closeup=cu)
                Image.fromarray(_grab()).save(os.path.join(pdir, f"probe_f{pf:04d}_yaw{y:+.0f}_{tag}.png"))
                print(f"[fig] probe yaw={y:+.0f} {tag} → f{pf}", flush=True)
        print(f"[fig] probes → {pdir}")
        os._exit(0)

    # ---- label setup ----
    lab_font = tim_font = None
    if not args.no_label:
        lab_font = ImageFont.truetype(_FONT, max(12, int(H * 0.050 * args.label_scale)))
        tim_font = ImageFont.truetype(_FONT, max(10, int(H * 0.030 * args.label_scale)))
    dark_bg = not args.dome_bg
    fg = (255, 255, 255) if dark_bg else (25, 25, 25)
    stroke = (0, 0, 0) if dark_bg else (255, 255, 255)

    import imageio
    out = args.out or os.path.join(_PROC, "g1_shadow", args.cls, args.clip, "0", "figure_nobg.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)
    writer = imageio.get_writer(out, fps=args.fps, macro_block_size=1, quality=9)

    for i, f in enumerate(sel):
        _pose(f)
        _aim_frame(f)
        if args.rtx_alpha or args.alpha_matte:
            rgba = _grab4() if args.rtx_alpha else _grab_matte()
            al = rgba[..., 3].astype(np.float32) / 255.0
            if i == 0:
                print(f"[fig] {'rtx alpha' if args.rtx_alpha else 'matte'}: "
                      f"transparent={(al < 0.02).mean() * 100:.1f}%  "
                      f"solid={(al > 0.98).mean() * 100:.1f}%  edge={((al >= 0.02) & (al <= 0.98)).mean() * 100:.2f}%",
                      flush=True)
            rgb = np.clip(rgba[..., :3].astype(np.float32) * al[..., None]
                          + bg[None, None, :] * (1.0 - al[..., None]), 0, 255).astype(np.uint8)
            png = Image.fromarray(rgba, mode="RGBA")
        else:
            rgb = _grab()
            png = Image.fromarray(rgb)
        if args.png_dir and (i % max(1, args.png_every) == 0):
            png.save(os.path.join(args.png_dir, f"{args.clip}_f{f:04d}.png"))
        img = Image.fromarray(rgb)
        if lab_font is not None:
            d = ImageDraw.Draw(img)
            m = int(H * 0.028)
            d.text((m, m), f"frame {f}", font=lab_font, fill=fg,
                   stroke_width=max(2, int(H * 0.004)), stroke_fill=stroke)
            if not args.no_time_label:
                _t2 = f"t = {f / args.src_fps:.2f} s"
                if args.ref_map:
                    _a, _b = [float(v) for v in args.ref_map.split(",")]
                    _t2 += f"   ref f {int(round(_a * f + _b))}"
                d.text((m, m + int(H * 0.062)), _t2,
                       font=tim_font, fill=fg, stroke_width=max(1, int(H * 0.003)), stroke_fill=stroke)
        writer.append_data(np.asarray(img))
        if i % 20 == 0:
            print(f"[fig] {i}/{len(sel)} (frame {f})", flush=True)

    writer.close()
    print(f"[fig] wrote {out}  ({len(sel)} frames @ {args.fps} fps)")
    if args.png_dir:
        print(f"[fig] PNGs → {args.png_dir}")
    os._exit(0)


if __name__ == "__main__":
    main()
