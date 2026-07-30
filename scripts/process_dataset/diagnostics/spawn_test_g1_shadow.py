"""Spawn/sim integrity test for the composite G1 + bimanual Shadow USD.

Validates that the authored USD parses as ONE PhysX articulation with the expected
DOF/body counts and does NOT disintegrate (bodies flying apart) when stepped. Builds
the *exact* actuator layout intended for `G1_SHADOW_CFG` so this doubles as a cfg check.

Test protocol (integrity, not balance):
  fix_root_link=True + gravity ON + PD-hold every actuated joint at its default pose,
  step ~120 physics steps. A correctly-assembled humanoid keeps a bounded body-position
  AABB (~1.5 m); a broken articulation (link not in the tree / bad joint) explodes.

Run in tmux session 1 (Isaac Sim env):
    python -u scripts/process_dataset/diagnostics/spawn_test_g1_shadow.py [--usd <path>] [--steps 120]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default="source/robotis_sh5/data/robots/G1/G1_shadow.usd")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--float-base", action="store_true", help="fix_root_link=False (falls under gravity)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(headless=True)
sim_app = app_launcher.app

import os  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

_USD = os.path.abspath(args.usd)


# ── the actuator layout intended for G1_SHADOW_CFG (G1 body gains + Shadow finger gains) ──
_ACTUATORS = {
    "legs": DCMotorCfg(
        joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
        effort_limit={".*_hip_yaw_joint": 88.0, ".*_hip_roll_joint": 88.0,
                      ".*_hip_pitch_joint": 88.0, ".*_knee_joint": 139.0},
        velocity_limit={".*_hip_yaw_joint": 32.0, ".*_hip_roll_joint": 32.0,
                        ".*_hip_pitch_joint": 32.0, ".*_knee_joint": 20.0},
        stiffness={".*_hip_yaw_joint": 100.0, ".*_hip_roll_joint": 100.0,
                   ".*_hip_pitch_joint": 100.0, ".*_knee_joint": 200.0},
        damping={".*_hip_yaw_joint": 2.5, ".*_hip_roll_joint": 2.5,
                 ".*_hip_pitch_joint": 2.5, ".*_knee_joint": 5.0},
        armature={".*_hip_.*": 0.03, ".*_knee_joint": 0.03},
        saturation_effort=180.0,
    ),
    "feet": DCMotorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness={".*_ankle_pitch_joint": 20.0, ".*_ankle_roll_joint": 20.0},
        damping={".*_ankle_pitch_joint": 0.2, ".*_ankle_roll_joint": 0.1},
        effort_limit={".*_ankle_pitch_joint": 50.0, ".*_ankle_roll_joint": 50.0},
        velocity_limit={".*_ankle_pitch_joint": 37.0, ".*_ankle_roll_joint": 37.0},
        armature=0.03, saturation_effort=80.0,
    ),
    "waist": ImplicitActuatorCfg(
        joint_names_expr=["waist_.*_joint"],
        effort_limit={"waist_yaw_joint": 88.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0},
        velocity_limit={"waist_yaw_joint": 32.0, "waist_roll_joint": 37.0, "waist_pitch_joint": 37.0},
        stiffness={"waist_yaw_joint": 5000.0, "waist_roll_joint": 5000.0, "waist_pitch_joint": 5000.0},
        damping={"waist_yaw_joint": 5.0, "waist_roll_joint": 5.0, "waist_pitch_joint": 5.0},
        armature=0.001,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                          ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
        effort_limit=300, velocity_limit=100, stiffness=3000.0, damping=10.0,
        armature={".*_shoulder_.*": 0.001, ".*_elbow_.*": 0.001, ".*_wrist_.*_joint": 0.001},
    ),
    # Bimanual Shadow fingers: 18 actuated DOF/hand (J0 coupling joints excluded, as in the
    # robotis_shadow_grasp task). Prefixes robot0_r_ / robot0_l_.
    "shadow_fingers": ImplicitActuatorCfg(
        joint_names_expr=[
            "robot0_(r|l)_(FF|MF|RF|LF|TH)J[1-3]",
            "robot0_(r|l)_LFJ4", "robot0_(r|l)_THJ4", "robot0_(r|l)_THJ0",
        ],
        velocity_limit_sim=15.0, effort_limit_sim=3.09, stiffness=1.0, damping=0.2,
    ),
}

_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, fix_root_link=not args.float_base,
            solver_position_iteration_count=8, solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    actuators=_ACTUATORS,
)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
    # ground + light
    gp = sim_utils.GroundPlaneCfg(); gp.func("/World/ground", gp)
    dl = sim_utils.DomeLightCfg(intensity=2000.0); dl.func("/World/light", dl)

    robot_cfg = _ROBOT_CFG.replace(prim_path="/World/Robot")
    robot_cfg.spawn.func("/World/Robot", robot_cfg.spawn, translation=(0.0, 0.0, 1.0))
    robot = Articulation(robot_cfg)

    sim.reset()
    print("\n" + "=" * 70)
    print(f"USD: {_USD}")
    print(f"fix_root_link={not args.float_base}  num_envs={args.num_envs}")
    print("=" * 70)

    nb, nj = robot.num_bodies, robot.num_joints
    print(f"[articulation] num_bodies={nb}  num_joints(DOF)={nj}   (expect bodies=78, DOF=73)")

    # actuator coverage
    covered = set()
    for name, act in robot.actuators.items():
        jn = [robot.joint_names[i] for i in act.joint_indices]
        covered.update(jn)
        print(f"  actuator '{name}': {len(jn)} joints")
    uncovered = [j for j in robot.joint_names if j not in covered]
    print(f"  covered={len(covered)}  uncovered={len(uncovered)} (expect 8 = J0 coupling ×2 hands)")
    print(f"  uncovered joints: {sorted(uncovered)}")

    # hold default joint pos via PD
    default_q = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(default_q, torch.zeros_like(default_q))

    def body_extent():
        p = robot.data.body_link_pos_w[0]  # (nb, 3), env 0
        return (p.max(0).values - p.min(0).values).tolist(), p

    ext0, p0 = body_extent()
    print(f"\n[step 0]   body AABB extent (m): "
          f"x={ext0[0]:.3f} y={ext0[1]:.3f} z={ext0[2]:.3f}")

    dt = sim.get_physics_dt()
    for _ in range(args.steps):
        robot.set_joint_position_target(default_q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    extN, pN = body_extent()
    root = robot.data.root_link_pos_w[0].tolist()
    has_nan = bool(torch.isnan(pN).any().item())
    drift = (pN - p0).norm(dim=-1).max().item()
    print(f"[step {args.steps}] body AABB extent (m): "
          f"x={extN[0]:.3f} y={extN[1]:.3f} z={extN[2]:.3f}")
    print(f"[step {args.steps}] root pos: {[round(v, 3) for v in root]}")
    print(f"[step {args.steps}] max body drift from start: {drift:.4f} m")
    print(f"[step {args.steps}] NaN in body positions: {has_nan}")

    exploded = max(extN) > 5.0 or has_nan
    verdict = "DISINTEGRATED / BROKEN" if exploded else "OK (assembled, bounded)"
    print(f"\n[VERDICT] {verdict}")
    print("[done]")
    sim_app.close()


if __name__ == "__main__":
    main()
