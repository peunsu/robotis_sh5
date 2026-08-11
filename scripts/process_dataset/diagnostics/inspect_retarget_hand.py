"""Which finger of the retargeted hand is extended, and is it extended because the target says so?

A finger that looks wrong in a render can be wrong in two different ways, and they need different
fixes: the SOLVE failed to reach its target, or the solve reached a target that is itself wrong. So
for every frame this reports both — where each Shadow fingertip ends up under the retargeted joints
(forward kinematics through the sim), and where the ParaHome human fingertip it was fitted to sits.

    per-finger  reach   = |distal tip - palm|, the plain "how extended is it" number
                residual= |robot tip - human target|, i.e. did the optimiser converge

    python scripts/process_dataset/diagnostics/inspect_retarget_hand.py \
        --task Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0 \
        --clip_class single_rigid --clip_name s101_seg12_knife --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect the retargeted hand pose per finger.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--clip_class", type=str, default=None)
parser.add_argument("--clip_name", type=str, default=None)
parser.add_argument("--frames", type=int, nargs="+", default=[0, 30, 100, 250, 400])
parser.add_argument("--side", type=str, default="r", choices=["l", "r"])
parser.add_argument("--fix_joint_order", action="store_true",
                    help="Permute the retargeted joints BY NAME from the json's column layout into the "
                         "env's live _action_joint_ids order, so the residual measures the SOLVE rather "
                         "than a slot mismatch between the two.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import robotis_sh5.tasks  # noqa: F401,E402
from robotis_sh5.tasks.direct.g1_shadow_sonic_residual.g1_shadow_sonic_residual_env_cfg import (  # noqa: E402
    HAND_CHAIN, N_BODY_KPTS, N_HAND_KPTS_PER_HAND,
)


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.clip_class:
        env_cfg.clip_class = args_cli.clip_class
    if args_cli.clip_name:
        env_cfg.clip_name = args_cli.clip_name
    env_cfg.scene.num_envs = 1
    env_cfg.termination = False
    env_cfg.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    dev, org, s = env.device, env.scene.env_origins, args_cli.side
    aid = env._action_joint_ids_t

    if args_cli.fix_joint_order:
        import json
        js = json.load(open("source/robotis_sh5/data/robots/G1/g1_shadow_joint_order.json"))["action_joint_names"]
        jnames = env.robot.data.joint_names
        perm = torch.tensor([js.index(jnames[i]) for i in env._action_joint_ids], device=dev, dtype=torch.long)
        env._ref_joints = env._ref_joints[:, perm]
        print(f"[inspect] remapped by name: {int((perm != torch.arange(len(perm), device=dev)).sum())} slots moved")

    fingers = [k for k in HAND_CHAIN if k != "wrist"]
    distal = {f: f"robot0_{s}_{HAND_CHAIN[f]['shadow'][-1]}" for f in fingers}
    palm_id = env.robot.find_bodies(f"robot0_{s}_palm")[0][0]
    tip_ids = {f: env.robot.find_bodies(b)[0][0] for f, b in distal.items()}

    # the reference keypoint slot each fingertip was fitted to: 54 = 14 body + 20 L-hand + 20 R-hand,
    # hand block laid out in HAND_CHAIN order (wrist, index, middle, ring, pinky, thumb)
    off = N_BODY_KPTS + (0 if s == "l" else N_HAND_KPTS_PER_HAND)
    kpt_slot, c = {}, 0
    for name, spec in HAND_CHAIN.items():
        for i in range(len(spec["parahome"])):
            if name != "wrist" and i == len(spec["parahome"]) - 1:
                kpt_slot[name] = off + c + i
        c += len(spec["parahome"])

    jn = env.robot.data.joint_names
    act = [jn[i] for i in aid.tolist()]
    print(f"\n{'':6s}" + "".join(f"{f:>10s}" for f in fingers))
    for f in args_cli.frames:
        if f >= env._ref_len:
            continue
        jp = env.robot.data.default_joint_pos.clone()
        jp[:, aid] = env._ref_joints[f]
        env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        root = torch.zeros(1, 7, device=dev)
        root[:, :3] = env._ref_root_pos[f] + org[0]
        root[:, 3:7] = env._ref_root_quat[f]
        env.robot.write_root_pose_to_sim(root)
        env.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

        bp = env.robot.data.body_pos_w[0]
        palm = bp[palm_id]
        reach = {f_: (bp[tip_ids[f_]] - palm).norm().item() for f_ in fingers}
        resid = {f_: (bp[tip_ids[f_]] - (env._ref_kpts[f, kpt_slot[f_]] + org[0])).norm().item()
                 for f_ in fingers}
        print(f"\nframe {f}")
        print(f"  {'reach cm':16s}" + "".join(f"{reach[f_] * 100:10.2f}" for f_ in fingers))
        print(f"  {'residual cm':16s}" + "".join(f"{resid[f_] * 100:10.2f}" for f_ in fingers))
        for jsuf in ("J1", "J2", "J3"):
            row = []
            for f_ in fingers:
                pre = {"index": "FF", "middle": "MF", "ring": "RF", "pinky": "LF", "thumb": "TH"}[f_]
                nm = f"robot0_{s}_{pre}{jsuf}"
                row.append(f"{env._ref_joints[f][act.index(nm)].item():10.3f}" if nm in act else f"{'-':>10s}")
            print(f"  {jsuf:16s}" + "".join(row))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
