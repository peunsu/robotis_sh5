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
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.constants import HandType


# Monkey patch for compatibility with older versions of Python and NumPy
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# Numpy patch
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.str_

class TrajectoryGenerator:
    def __init__(self, robot_path: str, config_path: str, dexycb_dir: str, hand_type: str = "right"):
        """
        Initialize the trajectory generator.

        Args:
            robot_path (str): The path to the robot URDF file.
            config_path (str): The path to the retargeting configuration file.
            dexycb_dir (str): The directory containing the DexYCB dataset.
            hand_type (str, optional): The type of hand ('right' or 'left'). Defaults to "right".

        Raises:
            ValueError: If the hand type is invalid.
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hand_type = hand_type
        if hand_type not in ["right", "left"]:
            raise ValueError(f"Invalid hand type: {hand_type}. Must be 'right' or 'left'.")

        RetargetingConfig.set_default_urdf_dir(robot_path) # Set default URDF directory for retargeting config
        override = dict(add_dummy_free_joint=True) # Add a dummy free joint for root pose retargeting
        self.config = RetargetingConfig.load_from_file(config_path, override=override) # Load retargeting config from YAML file
        self.retargeter = self.config.build() # Build the retargeting module based on the loaded config

        self.dataset = DexYCBVideoDataset(dexycb_dir, hand_type=hand_type) # Load the DexYCB dataset for the specified hand type
        self.dataset_size = len(self.dataset)
        print(f"Loaded DexYCB dataset with {self.dataset_size} sequences for {hand_type} hand.")
    
    def select_sequence(self, seq_idx: int = 0):
        """
        Select a sequence from the dataset and load the relevant data for retargeting.

        Args:
            seq_idx (int, optional): The index of the sequence to select. Defaults to 0.

        Raises:
            ValueError: If the sequence index is out of bounds.
        """
        
        if seq_idx < 0 or seq_idx >= self.dataset_size:
            raise ValueError(f"Sequence index {seq_idx} is out of bounds (0 to {self.dataset_size-1})")
        
        sample = self.dataset[seq_idx] # Load the selected sequence from the dataset
        
        self.capture_name = sample["capture_name"]
        print(f"Selected sequence {seq_idx}: {self.capture_name}")
        
        # Extract object mesh paths, names, and poses from the sample
        self.mesh_path_list = sample["object_mesh_file"]
        if not isinstance(self.mesh_path_list, list):
            self.mesh_path_list = [self.mesh_path_list]
        self.object_names = [str(p).split('/')[-2] for p in self.mesh_path_list]
        self.num_objects = len(self.object_names)
        self.object_poses_all_frames = sample["object_pose"]
        print(f"Objects in this sequence: {self.object_names}")
        
        # Extract hand poses and initialize the MANO layer
        self.hand_poses = sample["hand_pose"]
        self.mano_layer = MANOLayer(side=self.hand_type, betas=sample["hand_shape"]).to(self.device)
        print(f"MANO layer initialized. Shape parameters: {sample['hand_shape']}")
        
        # Compute the camera-to-world transformation from the provided extrinsics
        self.camera_to_world = pt.invert_transform(sample["extrinsics"])
        print(f"Camera extrinsics: {sample['extrinsics']}")
        
        print("Sequence data loaded successfully.")
    
    def compute_hand_geometry(self, hand_pose_frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Compute the hand geometry (vertices and joints) from the given hand pose frame using the MANO layer.

        Args:
            hand_pose_frame (np.ndarray): The hand pose frame to compute geometry for.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: The computed hand geometry (vertices and joints).
        """
        # if np.abs(hand_pose_frame).sum() < 1e-5:
        #     return None, None
        
        # Extract pose parameters and convert to torch tensors
        p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32)).to(self.device)
        t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32)).to(self.device)
        
        # Calculate vertices and joints using the MANO layer
        with torch.no_grad():
            vertex, joint = self.mano_layer(p, t)
            
        vertex = vertex.cpu().numpy()[0]
        joint = joint.cpu().numpy()[0]
        
        # Transform vertices and joints from camera space to world space using the camera-to-world transformation
        vertex = vertex @ self.camera_to_world[:3, :3].T + self.camera_to_world[:3, 3]
        vertex = np.ascontiguousarray(vertex)
        joint = joint @ self.camera_to_world[:3, :3].T + self.camera_to_world[:3, 3]
        joint = np.ascontiguousarray(joint)
        
        return vertex, joint

    def process_sequence(self) -> dict:
        """
        Process the selected sequence to generate the retargeted trajectory for the robot, including root poses and object poses.

        Returns:
            dict: The retargeted trajectory data.
        """
        
        # Initialize lists to store the trajectory data
        trajectory, root_positions, root_quaternions = [], [], []
        multi_obj_pos = [[] for _ in range(self.num_objects)]
        multi_obj_quat = [[] for _ in range(self.num_objects)]

        # Find the first valid hand pose frame to use for warm starting the retargeter
        start_frame = 0
        for i in range(len(self.hand_poses)):
            init_hand_pose_frame = self.hand_poses[i]
            vertex, _ = self.compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break
        
        first_hand_pose_frame = self.hand_poses[start_frame] # Use the first valid hand pose frame for warm starting
        wrist_q_cam = rotations.quaternion_from_compact_axis_angle(first_hand_pose_frame[0, :3])
        
        # Compute the wrist position in world coordinates from the first valid hand pose frame
        _, joint_world = self.compute_hand_geometry(first_hand_pose_frame)
        wrist_t_world = joint_world[0, :]  # The wrist joint is the first joint in the MANO model

        # Warm start the retargeter with the initial wrist pose (position and orientation) in world coordinates
        self.retargeter.warm_start(
            wrist_t_world, # [Important] The wrist_pos should be in world coordinates
            wrist_q_cam, # [Important] The wrist_quat should be in camera coordinates
            hand_type=HandType.right if self.hand_type == "right" else HandType.left,
            is_mano_convention=True
        )

        # Retargeting loop
        for i in range(start_frame, len(self.hand_poses)):
            # Compute the hand geometry (vertices and joints) for the current hand pose frame
            curr_hand_pose_frame = self.hand_poses[i]
            _, joint_world = self.compute_hand_geometry(curr_hand_pose_frame)

            # Retarget the joint positions to the robot's joint space using the retargeter
            indices = self.retargeter.optimizer.target_link_human_indices
            full_qpos = self.retargeter.retarget(joint_world[indices, :])

            # Extract the root translation, rotation (as Euler angles), and joint positions from the full_qpos
            root_trans = full_qpos[:3]
            root_euler = full_qpos[3:6]
            qpos = full_qpos[6:].astype(np.float32)

            # Convert the root rotation from Euler angles to a quaternion
            root_q_world = rotations.quaternion_from_euler(root_euler, 0, 1, 2, extrinsic=False)

            # All objects in the current frame
            curr_frame_all_objs = self.object_poses_all_frames[i]
            for obj_idx in range(self.num_objects):
                # Get the object poses and quaternions for the current frame
                pos_quat = curr_frame_all_objs[obj_idx]
                obj_pose_cam = pt.transform_from_pq(
                    np.concatenate([
                        pos_quat[4:],
                        np.array([pos_quat[3], pos_quat[0], pos_quat[1], pos_quat[2]])
                    ])
                ) # Order: x, y, z, qw, qx, qy, qz
                
                # Transform the object pose from camera space to world space using the camera-to-world transformation
                obj_pose_world = self.camera_to_world @ obj_pose_cam
                obj_R_world = obj_pose_world[:3, :3]
                obj_t_world = obj_pose_world[:3, 3]
                
                # Store the object poses and quaternions in world coordinates for the current frame
                multi_obj_pos[obj_idx].append(obj_t_world.astype(np.float32))
                multi_obj_quat[obj_idx].append(rotations.quaternion_from_matrix(obj_R_world).astype(np.float32))

            # Store the retargeted joint positions, root positions, and root quaternions for the current frame
            trajectory.append(qpos)
            root_positions.append(root_trans.astype(np.float32))
            root_quaternions.append(root_q_world.astype(np.float32))

        return {
            "qpos": np.array(trajectory),
            "root_pos": np.array(root_positions),
            "root_quat": np.array(root_quaternions),
            "obj_poses": np.array(multi_obj_pos),
            "obj_quats": np.array(multi_obj_quat),
            "object_names": self.object_names,
            "joint_names": self.retargeter.joint_names[6:],
            "capture_name": self.capture_name
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate retargeted trajectories for the robot from the DexYCB dataset.")
    parser.add_argument("--hand_type", type=str, default="right", choices=["right", "left"], help="Type of hand to process ('right' or 'left').")
    args_cli = parser.parse_args()
    
    # Initialize the trajectory generator with the paths to the robot URDF, retargeting config, and DexYCB dataset
    generator = TrajectoryGenerator("hands", f"config/hx5_d20_hand_{args_cli.hand_type}.yml", "DexYCB", hand_type=args_cli.hand_type)
    
    # Create the output directory for saving the generated trajectories if it doesn't exist
    os.makedirs("trajectories", exist_ok=True)
    
    for seq_idx in tqdm(range(generator.dataset_size), desc="Processing sequences"):
        # Select the sequence and process it to generate the retargeted trajectory
        generator.select_sequence(seq_idx=seq_idx)
        result = generator.process_sequence()

        # Save the generated trajectory to a .npy file in the trajectories directory
        save_path = f"trajectories/{result['capture_name']}.npy"
        np.save(save_path, result)

        print(f"\n[Success] Saved to {save_path}")
    
    print("\nAll sequences processed and saved successfully.")