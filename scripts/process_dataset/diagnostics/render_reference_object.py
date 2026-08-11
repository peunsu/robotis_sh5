"""Render the reference object trajectory in the actual scene — raw capture vs settle-corrected.

Plays the reference back KINEMATICALLY: every frame the object (and optionally the robot) is written
straight to its reference pose and the viewport is captured, so what you see is the reference itself
rather than what physics does to it. Two passes over the same frames — the raw capture and the same
capture plus the per-frame lift from obj_settle_correction.npz — are stacked side by side, which is
the only way a 0.5 cm correction is legible.

The env is built with the correction DISABLED and the lift is added here, so both panels come from
one scene and one camera and differ by exactly the correction under test.

    python scripts/process_dataset/diagnostics/render_reference_object.py \
        --task Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0 \
        --clip_class single_rigid --clip_name s101_seg12_knife \
        --headless --enable_cameras --out /tmp/ref_object.mp4
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render the reference object trajectory.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--clip_class", type=str, default=None)
parser.add_argument("--clip_name", type=str, default=None)
parser.add_argument("--out", type=str, default="/tmp/ref_object.mp4")
parser.add_argument("--fps", type=int, default=25, help="Reference is 50 Hz; 25 plays at half speed.")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--zoom", type=float, default=None, help="Override cfg.viewer_zoom (smaller = closer).")
parser.add_argument("--yaw", type=float, default=None, help="Override cfg.viewer_yaw (deg).")
parser.add_argument("--elev", type=float, default=None, help="Override cfg.viewer_elev (deg).")
parser.add_argument("--no_robot", action="store_true",
                    help="Leave the robot at its default pose. The reference hand is drawn by default "
                         "because a knife alone gives no sense of scale or of where the grasp is.")
parser.add_argument("--fix_joint_order", action="store_true",
                    help="Permute the retargeted joints BY NAME from g1_shadow_joint_order.json's order "
                         "into the env's live _action_joint_ids order. The npz stores values by the "
                         "json's column layout; if the robot's DOF order has changed since that json "
                         "was dumped, the env silently applies each value to the wrong joint.")
parser.add_argument("--hide_object", action="store_true",
                    help="Park the object far away. The knife occludes and visually merges with the "
                         "fingers wrapped around it, so hand-pose questions need it out of the shot.")
parser.add_argument("--panel", type=str, default="both", choices=["both", "raw", "corrected"],
                    help="'both' stacks raw|corrected side by side (needs a tight --zoom to read a "
                         "0.5 cm difference); a single panel keeps the full width for the trajectory.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import robotis_sh5.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.clip_class:
        env_cfg.clip_class = args_cli.clip_class
    if args_cli.clip_name:
        env_cfg.clip_name = args_cli.clip_name
    env_cfg.scene.num_envs = 1
    env_cfg.termination = False
    env_cfg.debug_vis = False
    env_cfg.object_settle_lift = False          # raw here; the lift is added per-panel below
    for f, v in (("viewer_zoom", args_cli.zoom), ("viewer_yaw", args_cli.yaw), ("viewer_elev", args_cli.elev)):
        if v is not None:
            setattr(env_cfg, f, v)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array").unwrapped
    env.reset()
    dev, org = env.device, env.scene.env_origins
    F = env._ref_len
    end = F if args_cli.end < 0 else min(args_cli.end, F)
    frames = list(range(args_cli.start, end, args_cli.stride))

    clip_dir = env._resolve_clip_dir(env_cfg)
    corr = os.path.join(clip_dir, "obj_settle_correction.npz")
    lift = np.zeros(F, dtype=np.float32)
    pen = np.zeros(F, dtype=bool)
    if os.path.exists(corr):
        d = np.load(corr)
        lift = d["lift"].astype(np.float32)
        pen = d["penetrating"] if "penetrating" in d else np.zeros(F, dtype=bool)
        print(f"[render] correction: {int(pen.sum())} penetrating frames, max {lift.max() * 100:.2f} cm")
    else:
        print(f"[render] no correction found in {clip_dir} — both panels will be identical.")

    aid = env._action_joint_ids_t

    if args_cli.fix_joint_order and env._ref_joints is not None:
        import json
        jp = os.path.join("source/robotis_sh5/data/robots/G1/g1_shadow_joint_order.json")
        js = json.load(open(jp))["action_joint_names"]
        jn = env.robot.data.joint_names
        env_order = [jn[i] for i in env._action_joint_ids]
        perm = torch.tensor([js.index(n) for n in env_order], device=dev, dtype=torch.long)
        n_moved = int((perm != torch.arange(len(perm), device=dev)).sum())
        env._ref_joints = env._ref_joints[:, perm]
        print(f"[render] joint order remapped by name: {n_moved}/{len(perm)} slots moved")

    def draw(f, dz):
        """Write the reference for frame f (object raised by dz metres) and capture the viewport."""
        root = torch.zeros(1, 7, device=dev)
        if args_cli.no_robot:
            root[:, :3] = org + torch.tensor([0.0, 0.0, 5.0], device=dev)
            root[:, 3] = 1.0
            jp = env.robot.data.default_joint_pos.clone()
        else:
            root[:, :3] = env._ref_root_pos[f] + org[0]
            root[:, 3:7] = env._ref_root_quat[f]
            jp = env.robot.data.default_joint_pos.clone()
            if env._ref_joints is not None:
                jp[:, aid] = env._ref_joints[f]
        env.robot.write_root_pose_to_sim(root)
        env.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        op = torch.zeros(1, 7, device=dev)
        op[:, :3] = env._ref_obj_pos[f] + org[0]
        op[:, 2] += dz
        op[:, 3:7] = env._ref_obj_quat[f]
        if args_cli.hide_object:
            op[:, 2] += 10.0
        env._object.write_root_pose_to_sim(op)
        env._object.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        env.scene.write_data_to_sim()
        # one 5 ms step so the render pipeline picks up the written transforms; the pose is rewritten
        # every frame, so the ~0.1 mm of drift inside a single step never accumulates.
        env.sim.step(render=True)
        return np.asarray(env.render())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    def label(img, text, colour):
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 40], fill=(0, 0, 0))
        d.text((12, 8), text, fill=colour, font=font)
        return np.asarray(im)

    # the first render of a fresh stage comes back black — warm the pipeline before capturing
    for _ in range(4):
        draw(frames[0], 0.0)

    out_frames = []
    for i, f in enumerate(frames):
        tag = "PENETRATING" if pen[f] else "untouched"
        panels = []
        if args_cli.panel in ("both", "raw"):
            panels.append(label(draw(f, 0.0), f"RAW capture   frame {f:4d}", (255, 140, 140)))
        if args_cli.panel in ("both", "corrected"):
            panels.append(label(draw(f, float(lift[f])),
                                f"CORRECTED +{lift[f] * 100:4.2f}cm {tag:<11s} frame {f:4d}", (140, 255, 160)))
        out_frames.append(panels[0] if len(panels) == 1 else np.concatenate(panels, axis=1))
        if i % 50 == 0:
            print(f"  frame {f}/{end}")

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)) or ".", exist_ok=True)
    imageio.mimsave(args_cli.out, out_frames, fps=args_cli.fps, quality=8, macro_block_size=1)
    print(f"\n{len(out_frames)} frames -> {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
