"""Process HO-Cap dataset into SPIDER format (same as OakInk output).

Input:
    data/raw/hocap/{subject_id}/{timestamp}/mano_params.json
    data/raw/hocap/{subject_id}/{timestamp}/object_pose.json
    data/raw/hocap/models/{obj_id}/textured_mesh.obj
    data/raw/hocap/data_configs.py  (DATACONFIGS list)

Output:
    data/processed/hocap/mano/right/{task}/{data_id}/trajectory_keypoints.npz
    data/processed/hocap/assets/objects/{obj_id}/visual.obj

Task naming: {subject_id}-{timestamp}-{obj_id}
  e.g.  subject_1-20231025_170231-G10_4

Run (all configs):
    python scripts/process_dataset/dataset/hocap.py

Run (single config by index):
    python scripts/process_dataset/dataset/hocap.py --config-index 0

Run (single config by task name):
    python scripts/process_dataset/dataset/hocap.py --task subject_1-20231025_170231-G10_4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
REF_DT = 1.0 / 30.0  # HO-Cap is annotated at 30 FPS

_SCRIPT_DIR = Path(__file__).resolve().parents[3]
_DATA_DIR = _SCRIPT_DIR / "source" / "robotis_sh5" / "data"
_HOCAP_RAW = _DATA_DIR / "raw" / "hocap"
_HOCAP_OUT = _DATA_DIR / "processed" / "hocap"


def _load_dataconfigs():
    sys.path.insert(0, str(_HOCAP_RAW))
    from data_configs import DATACONFIGS  # noqa: PLC0415
    sys.path.pop(0)
    return DATACONFIGS


def _cfg_task_name(cfg: dict) -> str:
    parts = cfg["path"].split("/")
    return f"{parts[0]}-{parts[1]}-{cfg['obj']}"


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


def ensure_quat_continuity(quats: np.ndarray) -> np.ndarray:
    out = quats.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out


def interpolate_object_poses(
    extracted_poses: dict,
    tracked_frames: list[int],
    smooth: bool = True,
) -> tuple[dict, list[int]]:
    """Interpolate sparse object pose observations across the full frame range."""
    trans, rots_xyzw, idxs = [], [], []
    for cid in tracked_frames:
        key = str(cid).zfill(6)
        if key not in extracted_poses:
            continue
        t = np.array(extracted_poses[key]["mesh_translation"], dtype=np.float64).reshape(3)
        q_xyzw = np.array(extracted_poses[key]["mesh_rotation_xyzw"], dtype=np.float64).reshape(4)
        trans.append(t)
        rots_xyzw.append(q_xyzw)
        idxs.append(cid)

    if len(idxs) < 2:
        return {}, []

    trans = np.array(trans)
    rots_xyzw = np.array(rots_xyzw)
    idxs = np.array(idxs)

    full_idx = np.arange(idxs[0], idxs[-1] + 1)
    interp_t = interp1d(idxs, trans, axis=0, kind="linear", fill_value="extrapolate")(full_idx)

    r = Rotation.from_quat(rots_xyzw)  # scipy expects xyzw
    interp_r_xyzw = Slerp(idxs, r)(full_idx).as_quat()
    interp_r_xyzw = ensure_quat_continuity(interp_r_xyzw)

    if smooth:
        interp_t = moving_average_filter(interp_t, window_size=9)
        interp_r_xyzw = moving_average_filter(interp_r_xyzw, window_size=9)
        norms = np.linalg.norm(interp_r_xyzw, axis=1, keepdims=True)
        interp_r_xyzw = interp_r_xyzw / np.where(norms > 1e-6, norms, 1.0)

    result = {
        str(fid).zfill(6): {
            "mesh_translation": interp_t[i].tolist(),
            "mesh_rotation_xyzw": interp_r_xyzw[i].tolist(),
        }
        for i, fid in enumerate(full_idx)
    }
    return result, full_idx.tolist()


def extract_wrist_rotation(kpts: np.ndarray) -> np.ndarray:
    """Landmark-based wrist rotation matrix from 21 MANO keypoints."""
    z_axis = kpts[9] - kpts[0]
    z_axis /= np.linalg.norm(z_axis) + 1e-8
    y_aux = kpts[5] - kpts[13]
    y_aux /= np.linalg.norm(y_aux) + 1e-8
    x_axis = np.cross(y_aux, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-8
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-8
    return np.stack([x_axis, y_axis, z_axis], axis=1)  # (3, 3) column vectors


def process_config(
    cfg: dict,
    smooth: bool = True,
    skip_existing: bool = True,
) -> bool:
    """Process one HO-Cap clip and save SPIDER-format trajectory.

    Args:
        cfg:           One entry from DATACONFIGS.
        smooth:        Apply moving-average smoothing to hand and object trajectories.
        skip_existing: Skip if output trajectory_keypoints.npz already exists.

    Returns:
        True if processed or already existed, False on error.
    """
    rel_parts = cfg["path"].split("/")
    sub_id = rel_parts[0]
    timestamp = rel_parts[1]
    obj_id = cfg["obj"]
    start_f, end_f = int(cfg["start"]), int(cfg["end"])

    task_name = f"{sub_id}-{timestamp}-{obj_id}"
    data_id = 0

    out_dir = _HOCAP_OUT / "mano" / "right" / task_name / str(data_id)
    out_npz = out_dir / "trajectory_keypoints.npz"

    if skip_existing and out_npz.exists():
        print(f"[skip] {out_npz.relative_to(_HOCAP_OUT)} already exists.")
        return True

    traj_dir = _HOCAP_RAW / sub_id / timestamp
    mano_path = traj_dir / "mano_params.json"
    obj_pose_path = traj_dir / "object_pose.json"
    mesh_path = _HOCAP_RAW / "models" / obj_id / "textured_mesh.obj"

    for p in (mano_path, obj_pose_path, mesh_path):
        if not p.exists():
            print(f"[warn] Missing file: {p}")
            return False

    with open(mano_path) as f:
        mano_all = json.load(f)
    with open(obj_pose_path) as f:
        obj_all = json.load(f)

    # --- Object pose: extract sparse observations, interpolate, smooth ---
    extracted_obj: dict = {}
    for f_idx in range(start_f, end_f + 1):
        key = str(f_idx).zfill(6)
        if key in obj_all and obj_id in obj_all[key]:
            pose = obj_all[key][obj_id]
            t = np.array(pose["mesh_translation"], dtype=np.float64).reshape(3)
            # JSON stores wxyz; convert to xyzw for scipy
            q_wxyz = np.array(pose["mesh_rotation"], dtype=np.float64).reshape(4)
            q_xyzw = q_wxyz[[1, 2, 3, 0]]
            extracted_obj[key] = {
                "mesh_translation": t.tolist(),
                "mesh_rotation_xyzw": q_xyzw.tolist(),
            }

    if not extracted_obj:
        print(f"[warn] No object pose data for {task_name}")
        return False

    tracked = sorted(int(k) for k in extracted_obj)
    interp_obj, valid_frames = interpolate_object_poses(extracted_obj, tracked, smooth=smooth)

    N = len(valid_frames)
    if N == 0:
        print(f"[warn] No valid interpolated frames for {task_name}")
        return False

    # --- Hand keypoints (21 × 3 per frame) ---
    kpts_all = np.array(mano_all["right"]["kpts"], dtype=np.float64)  # (total_frames, 21, 3)

    mano_kpts_raw = np.zeros((N, 21, 3), dtype=np.float64)
    for i, f_idx in enumerate(valid_frames):
        if f_idx < len(kpts_all):
            mano_kpts_raw[i] = kpts_all[f_idx]
        else:
            mano_kpts_raw[i] = mano_kpts_raw[i - 1]  # repeat last if out of range

    # Smooth all 21 keypoints jointly (N, 63) then reshape back
    if smooth and N > 1:
        kpts_flat = mano_kpts_raw.reshape(N, 63)
        kpts_flat = moving_average_filter(kpts_flat, window_size=5)
        mano_kpts_right = kpts_flat.reshape(N, 21, 3)
    else:
        mano_kpts_right = mano_kpts_raw.copy()

    # --- Derive wrist and fingertip from (smoothed) keypoints ---
    qpos_wrist_right = np.zeros((N, 7), dtype=np.float64)
    qpos_finger_right = np.zeros((N, 5, 7), dtype=np.float64)

    for i in range(N):
        kpts = mano_kpts_right[i]
        # Wrist position = keypoint 0
        qpos_wrist_right[i, :3] = kpts[0]
        # Wrist rotation from landmark frame → wxyz quaternion
        rot_mat = extract_wrist_rotation(kpts)
        qpos_wrist_right[i, 3:] = Rotation.from_matrix(rot_mat).as_quat()[[3, 0, 1, 2]]  # xyzw→wxyz
        # Fingertip positions; identity quaternion
        for fj, fi in enumerate(FINGERTIP_INDICES):
            qpos_finger_right[i, fj, :3] = kpts[fi]
            qpos_finger_right[i, fj, 3:] = [1.0, 0.0, 0.0, 0.0]  # wxyz identity

    # --- Object trajectory from interpolated poses ---
    qpos_obj_right = np.zeros((N, 7), dtype=np.float64)
    for i, f_idx in enumerate(valid_frames):
        key = str(f_idx).zfill(6)
        pose = interp_obj[key]
        qpos_obj_right[i, :3] = np.array(pose["mesh_translation"])
        q_xyzw = np.array(pose["mesh_rotation_xyzw"])
        qpos_obj_right[i, 3:] = q_xyzw[[3, 0, 1, 2]]  # xyzw → wxyz

    # --- Mesh: compute centroid, correct object positions, center mesh ---
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    centroid = np.mean(mesh.vertices, axis=0)

    # When mesh is centered (vertices -= centroid), the world-space object position must
    # be updated: p_corrected = p + R @ centroid (the centroid point in world frame).
    for i in range(N):
        obj_q_wxyz = qpos_obj_right[i, 3:]
        obj_q_xyzw = obj_q_wxyz[[1, 2, 3, 0]]
        R = Rotation.from_quat(obj_q_xyzw).as_matrix()
        qpos_obj_right[i, :3] += R @ centroid

    mesh_out_dir = _HOCAP_OUT / "assets" / "objects" / obj_id
    mesh_out_path = mesh_out_dir / "visual.obj"
    if not mesh_out_path.exists():
        mesh.vertices -= centroid
        mesh_out_dir.mkdir(parents=True, exist_ok=True)
        mesh.export(str(mesh_out_path))
        print(f"[ok] Mesh → {mesh_out_path.relative_to(_HOCAP_OUT)}")
    else:
        print(f"[skip] Mesh {mesh_out_path.relative_to(_HOCAP_OUT)} already exists.")

    # --- Save trajectory ---
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(out_npz),
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=np.zeros((N, 7), dtype=np.float64),
        qpos_finger_left=np.zeros((N, 5, 7), dtype=np.float64),
        qpos_obj_left=np.zeros((N, 7), dtype=np.float64),
        contact=np.zeros((N, 10), dtype=np.float64),
        mano_kpts_right=mano_kpts_right,
    )
    print(f"[ok] {N} frames → {out_npz.relative_to(_HOCAP_OUT)}")

    # task_info.json (one level up, shared by all data_ids of this task)
    task_info = {
        "task": task_name,
        "dataset_name": "hocap",
        "robot_type": "mano",
        "embodiment_type": "right",
        "data_id": data_id,
        "right_object_mesh_dir": str(mesh_out_dir.relative_to(_DATA_DIR / "processed")),
        "left_object_mesh_dir": None,
        "ref_dt": REF_DT,
    }
    with open(out_dir.parent / "task_info.json", "w") as f:
        json.dump(task_info, f, indent=2)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {"hocap_path": cfg["path"], "start": start_f, "end": end_f, "obj": obj_id},
            f, indent=2,
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process HO-Cap dataset into SPIDER format."
    )
    parser.add_argument(
        "--config-index", type=int, default=-1,
        help="Process a single config entry by 0-based index (-1 = all).",
    )
    parser.add_argument(
        "--task", type=str, default="",
        help="Process configs whose task name contains this string.",
    )
    parser.add_argument("--no-smooth", action="store_true", help="Disable smoothing.")
    parser.add_argument("--overwrite", action="store_true", help="Re-process existing outputs.")
    args = parser.parse_args()

    configs = _load_dataconfigs()
    smooth = not args.no_smooth
    skip_existing = not args.overwrite

    if args.config_index >= 0:
        configs = [configs[args.config_index]]
    elif args.task:
        configs = [c for c in configs if args.task in _cfg_task_name(c)]

    n_ok = n_skip = n_fail = 0
    for cfg in configs:
        task = _cfg_task_name(cfg)
        try:
            ok = process_config(cfg, smooth=smooth, skip_existing=skip_existing)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            print(f"[error] {task}: {e}")
            import traceback; traceback.print_exc()
            n_fail += 1

    print(f"\nDone. processed={n_ok}, skipped={n_skip}, failed={n_fail}.")


if __name__ == "__main__":
    main()
