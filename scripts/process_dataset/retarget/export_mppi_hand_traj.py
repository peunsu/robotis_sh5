"""monodex MPPI hand refine run -> hand_traj_best.npz in the schema the sonic_residual env reads.

The floating-hand MPPI of robotis_sh5_monodex (launch.py stage 5, MuJoCo Warp) replaces the RL hand pretrain
as the stage-1 source. This writes the same keys rollout.py --dump_hand_traj does for the RL policy, from the
executed MuJoCo trajectory sampled at every 30 fps reference frame (sim step warmup + f * ref_dt / sim_dt, as
monodex tools/export_hand_traj.py does), in ParaHome world coordinates (monodex's centering_offset added back):

  frame (F,) 0..F-1, control_fps 30          palm_pos (F,2,3) / palm_quat (F,2,4 wxyz)  [L, R] robot0_*_palm
  joint_names (36) / finger_qpos (F,36)      the 36 actuated finger joints (J0 of FF/MF/RF/LF excluded), qpos
  finger_target (F,36)                       MuJoCo position-actuator ctrl of the same joints
  joint_names_all_{l,r} (28) / joint_pos_all (F,2,28)   6 MuJoCo wrist dofs (robot0_*_wrist_mj_*) + 22 hand joints
  ft_pos (F,10,3), fingertip_body_names      pad = distal link + the env's FINGERTIP_OFFSETS
  link_pos (F,32,3) / link_quat (F,32,4) / link_contact_names   the 32 contact links (stage1_hand_contact.py)
  obj_pos (F,3) / obj_quat (F,4), object_name                   the simulated object
  source "mppi", source_run, couple_j0, reward_sum, rollout_idx 0

Joint values are robot0 values: MJCF value = sign * robot0 value (monodex assets/robots/g1_shadow/
joint_map_bimanual.json). The link poses (and the pads on them) are NOT monodex's MJCF bodies: they are
our URDF's FK (g1_shadow_nomimic.urdf) of those joint values, hung off the simulated palm. The MJCF hand moves
the same way but frames two links differently (measured 2026-09-30, palm -> link at any joint values):
the thumb distal frame is turned 90 deg about its z axis (our thumb pad offset in it lands 11.8 mm off, on
the side of the thumb) and the little-finger chain sits 1.8 mm off; the palm and the other fingers agree
exactly, the thumb chain to 0.4 mm. Using monodex's bodies put the thumb pads and the thumb-distal contact
mesh in the wrong place.
Also writes <out stem>_metrics.json: the object tracking error against the reference, for inspection.

Run in the monodex `retargeting` env (mujoco 3.4, the version that built the scene):
    $PY_RETARGET scripts/process_dataset/retarget/export_mppi_hand_traj.py --run-dir <monodex run dir> --out <npz>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.append(str(Path(__file__).resolve().parents[2]))   # scripts/ (local_paths.py)
import local_paths  # noqa: E402

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
TIP = {"thumb": "th", "index": "ff", "middle": "mf", "ring": "rf", "pinky": "lf"}
SEG = {v: k for k, v in TIP.items()}
WRIST = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z"]
# Must match g1_shadow_sonic_residual_env_cfg.py _FT_OFFSET_BASE (right hand link frame; left = Y mirrored)
# and LINK_CONTACT_NAMES (the env and stage1_hand_contact.py index links by these names).
_FT_OFFSET_BASE = {"th": [-0.0085, 0.0, 0.02], "ff": [0.0, -0.006, 0.0175], "mf": [0.0, -0.006, 0.0175],
                   "rf": [0.0, -0.006, 0.0175], "lf": [0.0, -0.006, 0.0175]}
LINK_CONTACT_NAMES = [f"robot0_{s}_{b}" for s in ("l", "r")
                      for b in ["palm"] + [f"{f}{g}" for f in ("ff", "mf", "lf", "rf", "th")
                                           for g in ("proximal", "middle", "distal")]]
COUPLED_J0 = {"FFJ0", "MFJ0", "RFJ0", "LFJ0"}


class UrdfHand:
    """palm -> link FK of our URDF (fixed + revolute joints, joint values by name)."""

    def __init__(self, path: str | Path):
        import xml.etree.ElementTree as ET
        self.j = {}
        for j in ET.parse(str(path)).getroot().iter("joint"):
            o = j.find("origin")
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in o.get("rpy", "0 0 0").split()]).as_matrix()
                T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            ax = j.find("axis")
            a = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
            self.j[j.find("child").get("link")] = (j.get("name"), j.find("parent").get("link"), j.get("type"), T,
                                                    a / max(np.linalg.norm(a), 1e-12))

    def chain(self, base: str, link: str) -> list:
        out = []
        while link != base:
            if link not in self.j:
                raise KeyError(f"{link} is not below {base} in the URDF")
            out.append(self.j[link])
            link = self.j[link][1]
        return out[::-1]

    @staticmethod
    def fk(chain: list, q: dict) -> np.ndarray:
        T = np.eye(4)
        for name, _, typ, To, a in chain:
            T = T @ To
            if typ in ("revolute", "continuous"):
                Tq = np.eye(4)
                Tq[:3, :3] = Rotation.from_rotvec(a * q.get(name, 0.0)).as_matrix()
                T = T @ Tq
        return T


def _mano_dir(run: Path) -> Path:
    """monodex utils/io.py mano_dir_for_run: drop the variant level, robot -> "mano"."""
    p = run.parts
    return Path(*p[:-6], "mano", *p[-5:-2], p[-1])


def _mj_body(robot0: str) -> str:
    """'robot0_r_ffdistal' -> 'right_index_distal', 'robot0_l_palm' -> 'left_palm' (monodex peunsu_reward)."""
    _, s, rest = robot0.split("_", 2)
    side = {"l": "left", "r": "right"}[s]
    return f"{side}_palm" if rest == "palm" else f"{side}_{SEG[rest[:2]]}_{rest[2:]}"


def _cfg(run: Path) -> dict:
    out = {}
    for line in (run / "config.yaml").read_text().splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="monodex results/shadow_hand/bimanual/<clip>/PARAHOME/<variant>/<id>")
    ap.add_argument("--out", required=True, help="hand_traj_best.npz to write")
    ap.add_argument("--monodex-root", default="", help="default: local_paths MONODEX_ROOT")
    ap.add_argument("--urdf", default=str(Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data" / "robots"
                                          / "G1" / "urdf_pyroki" / "g1_shadow_nomimic.urdf"),
                    help="our robot's URDF (the link frames the env and stage1_hand_contact.py use)")
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    mono = Path(args.monodex_root or local_paths.get("MONODEX_ROOT"))
    jmap = json.loads((mono / "retargeting" / "retargeting" / "assets" / "robots" / "g1_shadow"
                       / "joint_map_bimanual.json").read_text())
    info = json.loads((run.parent / "task_info.json").read_text())
    cfg = _cfg(run)

    model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    d = mujoco.MjData(model)
    res = np.load(run / "trajectory_mjwp.npz")
    q = res["qpos"].reshape(-1, res["qpos"].shape[-1])
    ctrl = res["ctrl"].reshape(-1, res["ctrl"].shape[-1])
    kin = np.load(run / "trajectory_kinematic.npz")
    F = int(kin["qpos"].shape[0])
    step = int(round(float(cfg["ref_dt"]) / float(cfg["sim_dt"])))
    idx = int(cfg["warmup_steps"]) + np.arange(F) * step
    covered = int((idx < len(q)).sum())
    idx = np.clip(idx, 0, len(q) - 1)
    offset = np.asarray(np.load(_mano_dir(run) / "trajectory_keypoints.npz")["centering_offset"], np.float64)

    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    names_all, cols_all, sign_all = {}, {}, {}
    act_names, act_cols, act_sign = [], [], []
    for side in ("left", "right"):
        s = side[0]
        hand = [n for n, m in jmap.items() if n.startswith(f"{side}_") and m.get("pyroki", "").startswith(f"robot0_{s}_")]
        names_all[s] = [f"robot0_{s}_wrist_mj_{w}" for w in WRIST] + [jmap[n]["pyroki"] for n in hand]
        cols_all[s] = [int(model.jnt_qposadr[jid(f"{side}_{w}")]) for w in WRIST] + [int(model.jnt_qposadr[jid(n)]) for n in hand]
        sign_all[s] = np.array([1.0] * 6 + [float(jmap[n]["sign"]) for n in hand])
        for n in hand:
            r0 = jmap[n]["pyroki"]
            if r0.split("_")[-1] in COUPLED_J0:
                continue
            act_names.append(r0)
            act_cols.append(int(model.jnt_qposadr[jid(n)]))
            act_sign.append(float(jmap[n]["sign"]))
    if len(act_names) != 36:
        raise RuntimeError(f"expected 36 actuated finger joints, got {len(act_names)}")
    act_cols, act_sign = np.array(act_cols), np.array(act_sign)
    # ctrl column c drives qpos column c (monodex robot_layout), so the same columns index ctrl.

    palm_ids = [bid(f"{sd}_palm") for sd in ("left", "right")]
    ft_names = [f"robot0_{s}_{TIP[f]}distal" for s in ("l", "r") for f in FINGERS]
    ft_off = np.array([[v[0], (v[1] if n.split("_")[1] == "r" else -v[1]), v[2]]
                       for n in ft_names for v in [_FT_OFFSET_BASE[n.split("_")[2][:2]]]])
    tip_sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{sd}_{f}_tip") for sd in ("left", "right") for f in FINGERS]
    if min(palm_ids) < 0:
        raise RuntimeError("a palm body is missing from the scene")
    urdf = UrdfHand(args.urdf)
    side_of = lambda n: 0 if n.split("_")[1] == "l" else 1  # noqa: E731
    link_chain = [urdf.chain(f"robot0_{n.split('_')[1]}_palm", n) for n in LINK_CONTACT_NAMES]
    ft_chain = [urdf.chain(f"robot0_{n.split('_')[1]}_palm", n) for n in ft_names]
    site_gap = []                                       # our pad vs monodex's own *_tip site (diagnostic)

    jpa = np.zeros((F, 2, 28))
    palm_pos, palm_quat = np.zeros((F, 2, 3)), np.zeros((F, 2, 4))
    link_pos, link_quat = np.zeros((F, 32, 3)), np.zeros((F, 32, 4))
    ft = np.zeros((F, 10, 3))
    for t in range(F):
        d.qpos[:] = q[idx[t]]
        mujoco.mj_kinematics(model, d)
        qd = {}
        for k, s in enumerate(("l", "r")):
            jpa[t, k] = sign_all[s] * q[idx[t], cols_all[s]]
            qd.update(zip(names_all[s][6:], jpa[t, k, 6:]))
        palm_pos[t] = d.xpos[palm_ids] + offset
        palm_quat[t] = d.xquat[palm_ids]
        Tp = []
        for k in range(2):
            T = np.eye(4)
            T[:3, :3] = d.xmat[palm_ids[k]].reshape(3, 3)
            T[:3, 3] = d.xpos[palm_ids[k]]
            Tp.append(T)
        for i, (n, ch) in enumerate(zip(LINK_CONTACT_NAMES, link_chain)):
            T = Tp[side_of(n)] @ urdf.fk(ch, qd)
            link_pos[t, i] = T[:3, 3] + offset
            link_quat[t, i] = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
        for i, (n, ch) in enumerate(zip(ft_names, ft_chain)):
            T = Tp[side_of(n)] @ urdf.fk(ch, qd)
            ft[t, i] = T[:3, 3] + T[:3, :3] @ ft_off[i] + offset
        site_gap.append(np.linalg.norm(ft[t] - offset - d.site_xpos[tip_sites], axis=1))
    for arr in (palm_quat, link_quat):                  # sign-continuous quaternions
        for t in range(1, F):
            flip = (arr[t] * arr[t - 1]).sum(-1) < 0
            arr[t][flip] *= -1

    obj_cols = slice(-14, -7)                           # right_object: the meshed, shared bimanual object
    obj = q[idx][:, obj_cols]
    obj_pos, obj_quat = obj[:, :3] + offset, obj[:, 3:7].copy()
    for t in range(1, F):
        if obj_quat[t] @ obj_quat[t - 1] < 0:
            obj_quat[t] *= -1
    ref = kin["qpos"][:, obj_cols]
    e_pos = np.linalg.norm(obj[:, :3] - ref[:, :3], axis=1)
    e_rot = np.degrees((Rotation.from_quat(obj[:, [4, 5, 6, 3]]) * Rotation.from_quat(ref[:, [4, 5, 6, 3]]).inv()).magnitude())
    metrics = {"clip": info.get("task"), "frames": F, "sim_frames_covered": covered,
               "obj_pos_cm_mean": float(e_pos.mean() * 100), "obj_pos_cm_max": float(e_pos.max() * 100),
               "obj_rot_deg_mean": float(e_rot.mean()), "obj_rot_deg_p90": float(np.percentile(e_rot, 90)),
               "couple_j0": bool(info.get("couple_j0", False)), "source_run": str(run)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, frame=np.arange(F), control_fps=30.0, rollout_idx=0,
             reward_sum=float(np.sum(res["rew_mean"])) if "rew_mean" in res.files else 0.0,
             joint_names=np.array(act_names),
             finger_qpos=(q[idx][:, act_cols] * act_sign).astype(np.float32),
             finger_target=(ctrl[idx][:, act_cols] * act_sign).astype(np.float32),
             joint_names_all_l=np.array(names_all["l"]), joint_names_all_r=np.array(names_all["r"]),
             joint_pos_all=jpa.astype(np.float32),
             palm_pos=palm_pos.astype(np.float32), palm_quat=palm_quat.astype(np.float32),
             ft_pos=ft.astype(np.float32), fingertip_body_names=np.array(ft_names),
             link_contact_names=np.array(LINK_CONTACT_NAMES),
             link_pos=link_pos.astype(np.float32), link_quat=link_quat.astype(np.float32),
             obj_pos=obj_pos.astype(np.float32), obj_quat=obj_quat.astype(np.float32),
             object_name=np.array(info["parahome_object"]), clip=np.array(info.get("task", "")),
             source=np.array("mppi"), source_run=np.array(str(run)),
             couple_j0=np.array(bool(info.get("couple_j0", False))))
    site_gap = np.array(site_gap) * 1000                # (F,10) mm
    metrics["pad_vs_monodex_tip_site_mm_max"] = {n: round(float(v), 2) for n, v in zip(ft_names, site_gap.max(0))}
    mpath = out.with_name(out.stem + "_metrics.json")
    mpath.write_text(json.dumps(metrics, indent=2))
    print(f"[export-mppi] {out}: {F} frames @ 30 fps (sim covers {covered}); object error mean "
          f"{metrics['obj_pos_cm_mean']:.2f} cm / {metrics['obj_rot_deg_mean']:.1f} deg, max "
          f"{metrics['obj_pos_cm_max']:.2f} cm, rot p90 {metrics['obj_rot_deg_p90']:.1f} deg ({mpath.name})")


if __name__ == "__main__":
    main()
