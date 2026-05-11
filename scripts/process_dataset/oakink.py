"""Process OakInk-Image dataset (intent=use/0001) to SPIDER format.

Process:
1. Load 21 hand joints in camera space, transform to world space via cam_extr
2. Extract wrist pose (position + landmark-based rotation) and fingertip positions
3. Extract object world-space pose from obj_anno (T_w_o) in general_info
4. Center object mesh by subtracting centroid, adjust trajectory accordingly
5. Save trajectory as trajectory_keypoints.npz

Input : data/raw/oakink/image/  (OakInk-Image dataset, intent_id=0001 only)
Output: data/processed/oakink/mano/{embodiment_type}/{task}/{data_id}/trajectory_keypoints.npz
        data/processed/oakink/assets/objects/{obj_id}/visual.obj

Reference: workspace2/spider/spider/process_datasets/oakink1.py
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
USE_INTENT_ID = "0001"
REF_DT = 1.0 / 30.0  # OakInk-Image is annotated at 30 FPS
VIEW_ID = 0


def moving_average_filter(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    pad_len = window_size // 2
    padded = np.pad(signal, ((pad_len, pad_len), (0, 0)), mode="edge")
    kernel = np.ones(window_size) / window_size
    smoothed = np.array(
        [np.convolve(padded[:, i], kernel, mode="valid") for i in range(signal.shape[1])]
    ).T
    return smoothed.squeeze()


def compute_mesh_centroid(mesh_path: str) -> np.ndarray:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    return np.mean(mesh.vertices, axis=0)


def center_and_save_mesh(mesh_path: str, centroid: np.ndarray, out_path: str) -> None:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    mesh.vertices -= centroid
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mesh.export(out_path)


def find_obj_mesh(obj_root: Path, obj_id: str) -> Path | None:
    for ext in (".obj", ".ply"):
        p = obj_root / f"{obj_id}{ext}"
        if p.exists():
            return p
    return None


def extract_wrist_rotation(joints_world: np.ndarray) -> np.ndarray:
    """Compute wrist rotation matrix from 21 world-space hand landmarks.

    Uses middle-finger MCP (9), index MCP (5), ring MCP (13) and wrist (0)
    to define a local frame.
    """
    z_axis = joints_world[9] - joints_world[0]
    z_axis /= np.linalg.norm(z_axis) + 1e-8
    y_aux = joints_world[5] - joints_world[13]
    y_aux /= np.linalg.norm(y_aux) + 1e-8
    x_axis = np.cross(y_aux, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-8
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-8
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def load_sequence_frames(anno_path: Path, seq_id_ts: str, sbj_flag: int) -> list[int]:
    with open(anno_path / "seq_all.json") as f:
        seq_all = json.load(f)
    return sorted(
        item[2]
        for item in seq_all
        if item[0] == seq_id_ts and item[3] == VIEW_ID and item[1] == sbj_flag
    )


def get_mesh_dir(data_dir: Path, obj_id: str) -> Path:
    return data_dir / "processed" / "oakink" / "assets" / "objects" / obj_id


def get_processed_data_dir(data_dir: Path, embodiment_type: str, task: str, data_id: int) -> Path:
    return data_dir / "processed" / "oakink" / "mano" / embodiment_type / task / str(data_id)


def process_sequence(
    seq_id_ts: str,
    data_dir: Path,
    data_id: int = 0,
    embodiment_type: str = "right",
    smooth: bool = True,
    skip_existing: bool = True,
) -> bool:
    """Process one OakInk-Image sequence and save SPIDER-format outputs.

    Args:
        seq_id_ts:        e.g. "A01001_0001_0000/2021-09-26-19-59-58"
        data_dir:         root data directory (contains raw/ and processed/)
        data_id:          integer index when multiple timestamps share the same seq_id
        embodiment_type:  "right" (OakInk-Image only contains right-hand data)
        smooth:           apply moving-average smoothing to hand and object trajectories
        skip_existing:    skip if output trajectory_keypoints.npz already exists
    """
    anno_path = data_dir / "raw" / "oakink" / "image" / "anno"
    obj_mesh_root = data_dir / "raw" / "oakink" / "image" / "obj"

    seq_id, ts = seq_id_ts.split("/")
    obj_id = seq_id.split("_")[0]
    sbj_flag = 0  # use-intent is always single subject
    task_name = seq_id.replace("_", "-")

    output_dir = get_processed_data_dir(data_dir, embodiment_type, task_name, data_id)
    out_npz = output_dir / "trajectory_keypoints.npz"

    if skip_existing and out_npz.exists():
        print(f"[skip] {out_npz} already exists.")
        return True

    frames = load_sequence_frames(anno_path, seq_id_ts, sbj_flag)
    if not frames:
        print(f"[warn] No frames found for {seq_id_ts} view={VIEW_ID}, skipping.")
        return False

    obj_mesh_path = find_obj_mesh(obj_mesh_root, obj_id)
    if obj_mesh_path is None:
        print(f"[warn] No mesh file for obj_id={obj_id}, skipping {seq_id}.")
        return False

    N = len(frames)
    wrist_pos_raw = np.zeros((N, 3))
    wrist_rot_raw = np.zeros((N, 3, 3))
    fingertip_raw = np.zeros((N, 5, 3))
    obj_T_w_o = np.zeros((N, 4, 4))
    valid = np.ones(N, dtype=bool)

    for i, frame in enumerate(frames):
        fname = f"{seq_id}__{ts}__{sbj_flag}__{frame}__{VIEW_ID}.pkl"
        hj_path = anno_path / "hand_j" / fname
        gi_path = anno_path / "general_info" / fname

        if not hj_path.exists() or not gi_path.exists():
            valid[i] = False
            continue

        with open(hj_path, "rb") as f:
            hj_cam = pickle.load(f)  # (21, 3) camera space
        with open(gi_path, "rb") as f:
            gi = pickle.load(f)

        T_w_c = np.linalg.inv(gi["cam_extr"].numpy())  # world←camera
        hj_h = np.concatenate([hj_cam, np.ones((21, 1))], axis=1)
        joints_world = (T_w_c @ hj_h.T).T[:, :3]

        wrist_pos_raw[i] = joints_world[0]
        wrist_rot_raw[i] = extract_wrist_rotation(joints_world)
        fingertip_raw[i] = joints_world[FINGERTIP_INDICES]
        obj_T_w_o[i] = gi["obj_anno"].numpy()  # T_w_o (world←obj canonical)

    # Fill missing frames by nearest-neighbour
    if not valid.all():
        n_missing = int((~valid).sum())
        print(f"[warn] {seq_id}: {n_missing}/{N} frames missing, filling by nearest neighbour.")
        valid_idxs = np.where(valid)[0]
        if len(valid_idxs) == 0:
            print(f"[error] {seq_id}: no valid frames at all, skipping.")
            return False
        for m in np.where(~valid)[0]:
            nn = valid_idxs[np.argmin(np.abs(valid_idxs - m))]
            wrist_pos_raw[m] = wrist_pos_raw[nn]
            wrist_rot_raw[m] = wrist_rot_raw[nn]
            fingertip_raw[m] = fingertip_raw[nn]
            obj_T_w_o[m] = obj_T_w_o[nn]

    if smooth:
        wrist_pos_raw = moving_average_filter(wrist_pos_raw, window_size=5)
        fingertip_raw = moving_average_filter(fingertip_raw.reshape(N, 15), window_size=5).reshape(N, 5, 3)
        obj_trans = moving_average_filter(obj_T_w_o[:, :3, 3], window_size=5)
        obj_T_w_o[:, :3, 3] = obj_trans

    qpos_wrist_right = np.zeros((N, 7))
    qpos_finger_right = np.zeros((N, 5, 7))
    qpos_obj_right = np.zeros((N, 7))
    qpos_wrist_left = np.zeros((N, 7))
    qpos_finger_left = np.zeros((N, 5, 7))
    qpos_obj_left = np.zeros((N, 7))

    for i in range(N):
        # Wrist
        qpos_wrist_right[i, :3] = wrist_pos_raw[i]
        xyzw = Rotation.from_matrix(wrist_rot_raw[i]).as_quat()
        qpos_wrist_right[i, 3:] = xyzw[[3, 0, 1, 2]]  # wxyz

        # Fingertips (identity orientation — only position is used)
        for j in range(5):
            qpos_finger_right[i, j, :3] = fingertip_raw[i, j]
            qpos_finger_right[i, j, 3:] = [1, 0, 0, 0]

        # Object
        T = obj_T_w_o[i]
        qpos_obj_right[i, :3] = T[:3, 3]
        xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
        qpos_obj_right[i, 3:] = xyzw[[3, 0, 1, 2]]  # wxyz

    # Centre the mesh and adjust object trajectory by the centroid offset
    mesh_dir = get_mesh_dir(data_dir, obj_id)
    mesh_out_path = str(mesh_dir / "visual.obj")
    mesh_centroid = compute_mesh_centroid(str(obj_mesh_path))
    print(f"[mesh] {obj_id} mesh centroid: {mesh_centroid}")

    if not Path(mesh_out_path).exists():
        center_and_save_mesh(str(obj_mesh_path), mesh_centroid, mesh_out_path)
        print(f"[mesh] Saved centered mesh → {mesh_out_path}")

    mesh_transform = np.eye(4)
    mesh_transform[:3, 3] = mesh_centroid
    for i in range(N):
        q_wxyz = qpos_obj_right[i, 3:]
        R_sim = Rotation.from_quat(q_wxyz[[1, 2, 3, 0]]).as_matrix()
        T_sim = np.eye(4)
        T_sim[:3, :3] = R_sim
        T_sim[:3, 3] = qpos_obj_right[i, :3]
        T_corrected = T_sim @ mesh_transform
        qpos_obj_right[i, :3] = T_corrected[:3, 3]

    # Save trajectory
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        str(out_npz),
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_left=qpos_obj_left,
        contact=np.zeros((N, 10)),
    )

    mesh_dir_rel = str(mesh_dir.relative_to(data_dir))
    task_info = {
        "task": task_name,
        "dataset_name": "oakink",
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "right_object_mesh_dir": mesh_dir_rel,
        "left_object_mesh_dir": None,
        "ref_dt": REF_DT,
    }
    with open(output_dir / ".." / "task_info.json", "w") as f:
        json.dump(task_info, f, indent=2)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump({"seq_id_ts": seq_id_ts}, f, indent=2)

    print(f"[ok] Saved {N} frames → {output_dir}")
    return True


def collect_use_sequences(anno_path: Path) -> dict[str, list[str]]:
    """Return {seq_id: [seq_id_ts, ...]} for all use-intent single-subject sequences."""
    with open(anno_path / "seq_all.json") as f:
        seq_all = json.load(f)

    seen: set[str] = set()
    result: dict[str, list[str]] = defaultdict(list)
    for item in seq_all:
        seq_id_ts = item[0]
        if seq_id_ts in seen:
            continue
        seq_id = seq_id_ts.split("/")[0]
        parts = seq_id.split("_")
        if len(parts) == 3 and parts[1] == USE_INTENT_ID:
            seen.add(seq_id_ts)
            result[seq_id].append(seq_id_ts)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Process OakInk-Image use-intent sequences into SPIDER format."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2] / "source" / "robotis_sh5" / "data"
        ),
        help="Root data directory (contains raw/ and processed/).",
    )
    parser.add_argument(
        "--seq-id-ts",
        type=str,
        default="",
        help=(
            'Process a single sequence, e.g. "A01001_0001_0000/2021-09-26-19-59-58". '
            "If omitted, all use-intent sequences are processed."
        ),
    )
    parser.add_argument(
        "--embodiment-type",
        type=str,
        default="right",
        help="Embodiment type (default: right).",
    )
    parser.add_argument("--no-smooth", action="store_true", help="Disable moving-average smoothing.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-process even if output file already exists."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    smooth = not args.no_smooth
    skip_existing = not args.overwrite
    anno_path = data_dir / "raw" / "oakink" / "image" / "anno"

    if args.seq_id_ts:
        seq_id = args.seq_id_ts.split("/")[0]
        parts = seq_id.split("_")
        if len(parts) != 3 or parts[1] != USE_INTENT_ID:
            print(
                f"[error] seq_id_ts must be a use-intent (0001) single-subject sequence, got: {args.seq_id_ts}"
            )
            return
        process_sequence(
            args.seq_id_ts,
            data_dir,
            data_id=0,
            embodiment_type=args.embodiment_type,
            smooth=smooth,
            skip_existing=skip_existing,
        )
        return

    # Batch mode: process all use-intent sequences
    use_seqs = collect_use_sequences(anno_path)
    total = sum(len(v) for v in use_seqs.values())
    print(f"Found {total} use-intent sequences across {len(use_seqs)} unique seq_ids.")

    n_ok, n_skip = 0, 0
    for seq_id, seq_id_ts_list in sorted(use_seqs.items()):
        for data_id, s in enumerate(sorted(seq_id_ts_list)):
            ok = process_sequence(
                s,
                data_dir,
                data_id=data_id,
                embodiment_type=args.embodiment_type,
                smooth=smooth,
                skip_existing=skip_existing,
            )
            if ok:
                n_ok += 1
            else:
                n_skip += 1

    print(f"Done. processed={n_ok}, skipped={n_skip}.")


if __name__ == "__main__":
    main()
