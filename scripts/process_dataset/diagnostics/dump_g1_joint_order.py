"""One-time dump of the composite G1+Shadow authoritative joint order + wrist→palm mounts.

The retargeting pipeline must emit g1_joint_pos (F,65) in the EXACT env `_action_joint_ids`
order (PhysX-DOF order filtered per JOINT_GROUPS). We get it straight from the env (which
resolves it) rather than guessing offline. Also dumps the constant SE3 from each
{side}_wrist_yaw_link → robot0_{side}_palm (palm is rigidly mounted via fixed joints), used to
place the hand-IK targets. Writes data/robots/G1/g1_shadow_joint_order.json.

Run in tmux session 1:  python -u scripts/process_dataset/diagnostics/dump_g1_joint_order.py
"""

import json
import os

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

from robotis_sh5.tasks.direct.g1_shadow_locomanip.g1_shadow_locomanip_env import (  # noqa: E402
    G1ShadowLocomanipEnv,
)
from robotis_sh5.tasks.direct.g1_shadow_locomanip.g1_shadow_locomanip_env_cfg import (  # noqa: E402
    G1ShadowLocomanipEnvCfg,
)

_OUT = os.path.abspath("source/robotis_sh5/data/robots/G1/g1_shadow_joint_order.json")


def main():
    cfg = G1ShadowLocomanipEnvCfg()
    cfg.scene.num_envs = 1
    cfg.clip_class = "single_rigid"
    env = G1ShadowLocomanipEnv(cfg)
    env.reset()
    robot = env.robot

    jn = robot.joint_names
    action_joint_names = [jn[i] for i in env._action_joint_ids]
    assert len(action_joint_names) == 65, f"expected 65 action joints, got {len(action_joint_names)}"

    # constant wrist_yaw_link → palm SE3 per side (palm rigidly mounted via fixed joints).
    def body_pose(name):
        bid = robot.find_bodies(name)[0][0]
        p = robot.data.body_pos_w[0, bid]
        q = robot.data.body_quat_w[0, bid]  # wxyz
        return p, q

    wrist_to_palm = {}
    for side in ("r", "l"):
        wy_name = f"{'right' if side == 'r' else 'left'}_wrist_yaw_link"
        wp, wq = body_pose(wy_name)
        pp, pq = body_pose(f"robot0_{side}_palm")
        # T_wy->palm = inv(T_wy_world) * T_palm_world
        wq_inv = math_utils.quat_conjugate(wq.unsqueeze(0))
        rel_p = math_utils.quat_apply(wq_inv, (pp - wp).unsqueeze(0))[0]
        rel_q = math_utils.quat_mul(wq_inv, pq.unsqueeze(0))[0]
        wrist_to_palm[side] = {
            "pos": rel_p.cpu().tolist(),
            "quat_wxyz": rel_q.cpu().tolist(),
            "wrist_link": wy_name,
            "palm_link": f"robot0_{side}_palm",
        }

    # per-group ordered names (for debugging / sanity)
    group_names = {}
    off = 0
    from robotis_sh5.tasks.direct.g1_shadow_locomanip.g1_shadow_locomanip_env_cfg import JOINT_GROUPS
    for g, spec in JOINT_GROUPS.items():
        group_names[g] = action_joint_names[off:off + spec["dof"]]
        off += spec["dof"]

    out = {
        "all_joint_names": list(jn),
        "num_dofs": robot.num_joints,
        "action_joint_names": action_joint_names,   # 65, in _action_joint_ids order
        "group_names": group_names,
        "wrist_to_palm": wrist_to_palm,
    }
    with open(_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[dump] wrote {_OUT}")
    print(f"[dump] 65 action joints (order): {action_joint_names}")
    print(f"[dump] wrist_to_palm r: {wrist_to_palm['r']['pos']}  quat {wrist_to_palm['r']['quat_wxyz']}")
    print(f"[dump] wrist_to_palm l: {wrist_to_palm['l']['pos']}  quat {wrist_to_palm['l']['quat_wxyz']}")
    print("[done]")
    app.close()


if __name__ == "__main__":
    main()
