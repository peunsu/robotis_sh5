import os
import torch
import numpy as np
from tqdm import tqdm
import inspect

from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

from dataset import DexYCBVideoDataset
from mano_layer import MANOLayer

# Python 3.11 대응
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# numpy patch
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.str_


# -----------------------------
# 🔥 Pose composition (SAPIEN 대체)
# -----------------------------
def compose_pose(R1, t1, R2, t2):
    """
    world = (R1, t1) ∘ (R2, t2)
    """
    R = R1 @ R2
    t = R1 @ t2 + t1
    return R, t


def mat_to_quat(R):
    return rotations.quaternion_from_matrix(R)


def quat_to_mat(q):
    return rotations.matrix_from_quaternion(q)


class TrajectoryGenerator:
    def __init__(self, robot_path, config_path, dexycb_dir, hand_type="right"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hand_type = hand_type

        RetargetingConfig.set_default_urdf_dir(robot_path)
        override = dict(add_dummy_free_joint=True)
        self.config = RetargetingConfig.load_from_file(config_path, override=override)
        self.retargeter = self.config.build()

        self.dataset = DexYCBVideoDataset(dexycb_dir, hand_type=hand_type)

    def process_sequence(self, seq_idx=0):
        sample = self.dataset[seq_idx]
        capture_name = sample["capture_name"]

        mesh_path_list = sample["object_mesh_file"]
        if not isinstance(mesh_path_list, list):
            mesh_path_list = [mesh_path_list]
        object_names = [str(p).split('/')[-2] for p in mesh_path_list]

        hand_poses = sample["hand_pose"]
        hand_betas = sample["hand_shape"]
        extrinsic_mat = sample["extrinsics"]
        object_poses_all_frames = sample["object_pose"]

        # -----------------------------
        # 🔥 camera_pose 계산
        # -----------------------------
        R_w2c = extrinsic_mat[:3, :3]
        t_w2c = extrinsic_mat[:3, 3]

        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c

        mano_layer = MANOLayer(side=self.hand_type, betas=hand_betas).to(self.device)

        trajectory, root_positions, root_quaternions = [], [], []
        num_objects = len(object_names)
        multi_obj_pos = [[] for _ in range(num_objects)]
        multi_obj_quat = [[] for _ in range(num_objects)]

        # -----------------------------
        # Warm start
        # -----------------------------
        start_frame = 0
        for i in range(len(hand_poses)):
            if np.abs(hand_poses[i]).sum() > 1e-5:
                start_frame = i
                break

        first_hand = np.array(hand_poses[start_frame]).squeeze()

        p_init = torch.from_numpy(first_hand[:48].astype(np.float32)).to(self.device).reshape(1, 48)
        t_init = torch.from_numpy(first_hand[48:51].astype(np.float32)).to(self.device).reshape(1, 3)

        with torch.no_grad():
            _, joints_init_cam = mano_layer(p_init, t_init)
            joints_init_cam = joints_init_cam.cpu().numpy()[0]

        # 🔥 hand pose (camera)
        wrist_q_cam = rotations.quaternion_from_compact_axis_angle(first_hand[:3])
        wrist_R_cam = quat_to_mat(wrist_q_cam)
        wrist_t_cam = joints_init_cam[0]

        # 🔥 world 변환
        wrist_R_world, wrist_t_world = compose_pose(
            R_c2w, t_c2w,
            wrist_R_cam, wrist_t_cam
        )

        wrist_q_world = mat_to_quat(wrist_R_world)

        self.retargeter.warm_start(
            wrist_t_world,
            wrist_q_world,
            hand_type=self.hand_type
        )

        # -----------------------------
        # Main loop
        # -----------------------------
        for i in tqdm(range(len(hand_poses)), desc=f"Retargeting {capture_name}"):
            curr_hand_raw = hand_poses[i]
            if np.abs(curr_hand_raw).sum() < 1e-5:
                continue

            curr_hand = np.array(curr_hand_raw).squeeze()

            p = torch.from_numpy(curr_hand[:48].astype(np.float32)).to(self.device).reshape(1, 48)
            t = torch.from_numpy(curr_hand[48:51].astype(np.float32)).to(self.device).reshape(1, 3)

            with torch.no_grad():
                _, joints_cam = mano_layer(p, t)
                joints_cam = joints_cam.cpu().numpy()[0]

            indices = self.retargeter.optimizer.target_link_human_indices
            full_qpos = self.retargeter.retarget(joints_cam[indices, :])

            root_euler = full_qpos[3:6]
            qpos = full_qpos[6:].astype(np.float32)

            # 🔥 hand rotation (camera)
            hand_q_cam = rotations.quaternion_from_euler(
                root_euler, 0, 1, 2, extrinsic=False
            )
            hand_R_cam = quat_to_mat(hand_q_cam)
            hand_t_cam = joints_cam[0]

            # 🔥 world 변환
            hand_R_world, hand_t_world = compose_pose(
                R_c2w, t_c2w,
                hand_R_cam, hand_t_cam
            )

            root_q_world = mat_to_quat(hand_R_world).astype(np.float32)

            # -----------------------------
            # Object
            # -----------------------------
            curr_frame_all_objs = object_poses_all_frames[i]

            for obj_idx in range(num_objects):
                pos_quat = curr_frame_all_objs[obj_idx]

                # DexYCB: [qx, qy, qz, qw, tx, ty, tz]
                obj_q_cam = np.array([
                    pos_quat[3],
                    pos_quat[0],
                    pos_quat[1],
                    pos_quat[2]
                ])
                obj_t_cam = pos_quat[4:]

                obj_R_cam = quat_to_mat(obj_q_cam)

                # 🔥 world 변환
                obj_R_world, obj_t_world = compose_pose(
                    R_c2w, t_c2w,
                    obj_R_cam, obj_t_cam
                )

                obj_q_world = mat_to_quat(obj_R_world).astype(np.float32)

                multi_obj_pos[obj_idx].append(obj_t_world.astype(np.float32))
                multi_obj_quat[obj_idx].append(obj_q_world)

            trajectory.append(qpos)
            root_positions.append(hand_t_world.astype(np.float32))
            root_quaternions.append(root_q_world)

        return {
            "qpos": np.array(trajectory),
            "root_pos": np.array(root_positions),
            "root_quat": np.array(root_quaternions),
            "obj_poses": np.array(multi_obj_pos),
            "obj_quats": np.array(multi_obj_quat),
            "object_names": object_names,
            "joint_names": self.retargeter.joint_names[6:],
            "capture_name": capture_name
        }


if __name__ == "__main__":
    generator = TrajectoryGenerator("hands", "config/hx5_d20_hand_right.yml", "DexYCB")
    result = generator.process_sequence(seq_idx=0)

    os.makedirs("trajectories", exist_ok=True)
    save_path = f"trajectories/{result['capture_name']}_multi.npy"
    np.save(save_path, result)

    print(f"\n[Success] Saved to {save_path}")