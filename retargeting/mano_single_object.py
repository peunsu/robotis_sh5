import argparse
import os
import numpy as np
import torch
import inspect
from tqdm import tqdm
from pytransform3d import rotations
from pytransform3d import transformations as pt

from dataset import DexYCBVideoDataset
from mano_layer import MANOLayer

# 환경 호환성을 위한 패치
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.str_

class WorldTrajectoryGenerator:
    def __init__(self, dexycb_dir: str, hand_type: str = "right"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hand_type = hand_type
        
        # 데이터셋 로드
        self.dataset = DexYCBVideoDataset(dexycb_dir, hand_type=hand_type)
        self.dataset_size = len(self.dataset)
        print(f"Loaded DexYCB dataset: {self.dataset_size} sequences ({hand_type})")

    def select_sequence(self, seq_idx: int):
        sample = self.dataset[seq_idx]
        self.capture_name = sample["capture_name"]
        
        # 물체 정보
        self.mesh_path_list = sample["object_mesh_file"]
        if not isinstance(self.mesh_path_list, list):
            self.mesh_path_list = [self.mesh_path_list]
        self.object_names = [str(p).split('/')[-2] for p in self.mesh_path_list]
        self.num_objects = len(self.object_names)
        self.object_poses_all_frames = sample["object_pose"]
        
        # MANO 설정 (손 모양 파라미터 적용)
        self.hand_poses = sample["hand_pose"]
        self.mano_layer = MANOLayer(side=self.hand_type, betas=sample["hand_shape"]).to(self.device)
        
        # Camera to World 변환 행렬
        self.camera_to_world = pt.invert_transform(sample["extrinsics"])

    def compute_hand_geometry(self, hand_pose_frame: np.ndarray):
        """MANO를 통해 Camera space Joint를 구하고 World space로 변환"""
        if np.abs(hand_pose_frame).sum() < 1e-5:
            return None, None
        
        p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32)).to(self.device)
        t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32)).to(self.device)
        
        with torch.no_grad():
            _, joint = self.mano_layer(p, t)
            
        joint_cam = joint.cpu().numpy()[0] # (21, 3)
        
        # World Space로 변환
        joint_world = joint_cam @ self.camera_to_world[:3, :3].T + self.camera_to_world[:3, 3]
        
        # Wrist Orientation (World)
        wrist_q_cam = rotations.quaternion_from_compact_axis_angle(hand_pose_frame[0, :3])
        wrist_R_cam = rotations.matrix_from_quaternion(wrist_q_cam)
        wrist_R_world = self.camera_to_world[:3, :3] @ wrist_R_cam
        wrist_q_world = rotations.quaternion_from_matrix(wrist_R_world)
        
        return joint_world, wrist_q_world

    def process_sequence(self) -> dict:
        # 1. 타겟 물체 선정 (가장 많이 움직인 물체)
        target_obj_idx = 0
        max_dist = -1.0
        
        # 모든 프레임에 대한 물체 포즈 계산 (World 기준)
        total_frames = len(self.hand_poses)
        obj_poses_world = [[] for _ in range(self.num_objects)]
        
        for i in range(total_frames):
            for obj_idx in range(self.num_objects):
                pos_quat = self.object_poses_all_frames[i][obj_idx]
                # DexYCB format: [qx, qy, qz, qw, x, y, z] -> [x, y, z, qw, qx, qy, qz] 변환 후 matrix
                obj_pose_cam = pt.transform_from_pq(np.concatenate([pos_quat[4:], [pos_quat[3], pos_quat[0], pos_quat[1], pos_quat[2]]]))
                obj_pose_world = self.camera_to_world @ obj_pose_cam
                obj_poses_world[obj_idx].append(obj_pose_world)

        # 움직임 거리 기반 타겟 선정
        for obj_idx in range(self.num_objects):
            dist = np.linalg.norm(obj_poses_world[obj_idx][-1][:3, 3] - obj_poses_world[obj_idx][0][:3, 3])
            if dist > max_dist:
                max_dist = dist
                target_obj_idx = obj_idx

        target_obj_trajectory = obj_poses_world[target_obj_idx]
        start_pos = target_obj_trajectory[0][:3, 3]
        motion_threshold = 0.005 # 5mm
        motion_start_frame = 0
        
        for idx, pose in enumerate(target_obj_trajectory):
            if np.linalg.norm(pose[:3, 3] - start_pos) > motion_threshold:
                motion_start_frame = idx
                break

        # 2. 결과 저장용 리스트
        trajectory_data = {
            "joints_world": [],      # (F, 21, 3)
            "wrist_pos_world": [],   # (F, 3)
            "wrist_quat_world": [],  # (F, 4)
            "obj_pos_world": [],     # (F, 3)
            "obj_quat_world": []     # (F, 4)
        }

        for i in range(total_frames):
            joint_world, wrist_q_world = self.compute_hand_geometry(self.hand_poses[i])
            if joint_world is None: continue

            # 손 데이터
            trajectory_data["joints_world"].append(joint_world.astype(np.float32))
            trajectory_data["wrist_pos_world"].append(joint_world[0].astype(np.float32))
            trajectory_data["wrist_quat_world"].append(wrist_q_world.astype(np.float32))
            
            # 물체 데이터 (World 기준 PQ)
            obj_T = obj_poses_world[target_obj_idx][i]
            obj_pq = pt.pq_from_transform(obj_T)
            trajectory_data["obj_pos_world"].append(obj_pq[:3].astype(np.float32))
            trajectory_data["obj_quat_world"].append(obj_pq[3:].astype(np.float32))

        return {
            "kpos": np.array(trajectory_data["joints_world"]),
            "root_pos": np.array(trajectory_data["wrist_pos_world"]),
            "root_quat": np.array(trajectory_data["wrist_quat_world"]),
            "obj_pos": np.array(trajectory_data["obj_pos_world"]),
            "obj_quat": np.array(trajectory_data["obj_quat_world"]),
            "target_object_name": self.object_names[target_obj_idx],
            "motion_start_frame": motion_start_frame,
            "capture_name": self.capture_name
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_type", type=str, default="right")
    args = parser.parse_args()
    
    generator = WorldTrajectoryGenerator("DexYCB", hand_type=args.hand_type)
    save_base_dir = "mano_trajectories"
    
    for seq_idx in tqdm(range(generator.dataset_size), desc="Extracting World Keypoints"):
        generator.select_sequence(seq_idx)
        result = generator.process_sequence()
        
        # 저장 경로 설정
        obj_name = result['target_object_name']
        save_dir = os.path.join(save_base_dir, obj_name)
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{result['capture_name']}.npy")
        np.save(save_path, result)

    print(f"\n[Success] All sequences saved to {save_base_dir}")