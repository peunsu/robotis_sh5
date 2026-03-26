import os
import torch
import numpy as np
from tqdm import tqdm

from dex_retargeting.retargeting_config import RetargetingConfig

from dataset import DexYCBVideoDataset
from mano_layer import MANOLayer

import inspect

# Python 3.11에서 제거된 getargspec을 getfullargspec으로 대체하는 멍키 패치
if not hasattr(inspect, 'getargspec'):    
    inspect.getargspec = inspect.getfullargspec

np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.str_

# --- 1. DexRetargeter 클래스 정의 (통합) ---
class DexRetargeter:
    def __init__(self, config_path: str):
        # YAML 설정 파일 로드
        self.config = RetargetingConfig.load_from_file(config_path)
        # 리타겟팅 엔진 빌드 (SeqRetargeting 객체)
        self.retargeting = self.config.build()
        # 제어 대상 관절 이름 목록
        self.joint_names = self.retargeting.joint_names
        print(f"[Info] Retargeting initialized for joints: {len(self.joint_names)} joints")

    def compute_qpos(self, human_keypoints: np.ndarray) -> np.ndarray:
        """
        human_keypoints: (21, 3) 형태의 numpy array (미터 단위)
        """
        # 설정된 인덱스(4, 8, 12 등)에 맞는 관절 위치 추출
        indices = self.retargeting.optimizer.target_link_human_indices
        ref_value = human_keypoints[indices, :]
        
        # IK 연산 수행 및 qpos 반환
        robot_qpos = self.retargeting.retarget(ref_value)
        return robot_qpos

# --- 2. 메인 실행 루프 ---
def run_dexycb_retargeting(dexycb_dir: str, config_path: str, hand_type: str = "right"):
    # 데이터셋 초기화 (dataset.py의 DexYCBVideoDataset 클래스 사용)
    dataset = DexYCBVideoDataset(dexycb_dir, hand_type=hand_type)
    
    # 리타겟터 초기화
    retargeter = DexRetargeter(config_path)
    
    # 첫 번째 비디오 시퀀스 선택
    sample = dataset[0]
    hand_poses = sample["hand_pose"]      # (Frames, 48)
    hand_betas = sample["hand_shape"]     # (10,)
    capture_name = sample["capture_name"]
    
    # MANO 레이어 초기화 (mano_layer.py의 MANOLayer 클래스 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mano_layer = MANOLayer(side=hand_type, betas=hand_betas).to(device)
    
    print(f"[Start] Processing Sequence: {capture_name} ({len(hand_poses)} frames)")
    
    all_robot_qpos = []

    # 프레임별 처리
    for i in tqdm(range(len(hand_poses))):
        # 1. 데이터를 텐서로 만들고 앞의 48개만 슬라이싱 (51개 중 48개만 사용)
        # 2. reshape(-1, 48)을 통해 [1, 1, 51]을 [1, 48] 행렬로 강제 변환
        curr_pose = torch.tensor(hand_poses[i], dtype=torch.float32, device=device)
        curr_pose = curr_pose.view(-1)[:48].unsqueeze(0) # [1, 48] 확실하게 보장
        
        # t 역시 [1, 3]으로 확실하게 생성
        curr_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)
        
        # 디버깅용 출력 (첫 프레임에서만 확인해봐)
        if i == 0:
            print(f"Fixed Shape - p: {curr_pose.shape}, t: {curr_trans.shape}")

        with torch.no_grad():
            _, joints = mano_layer(curr_pose, curr_trans)
            
        # 2. (21, 3) Numpy 배열로 변환
        human_joints = joints.cpu().numpy()[0]
        
        # 3. 리타겟팅 (Robot qpos 계산)
        qpos = retargeter.compute_qpos(human_joints)
        all_robot_qpos.append(qpos)

    # 결과물 처리
    all_robot_qpos = np.array(all_robot_qpos)
    
    print("\n" + "="*40)
    print(f"Retargeting Completed!")
    print(f"Output Directory: {capture_name}")
    print(f"Total Frames: {all_robot_qpos.shape[0]}")
    print(f"Joint Count: {all_robot_qpos.shape[1]}")
    print("="*40)

    return all_robot_qpos, retargeter.joint_names

if __name__ == "__main__":
    # 사용자 환경에 맞게 경로를 수정해줘
    DEXYCB_PATH = "DexYCB"
    CONFIG_PATH = "config/hx5_d20_hand_right.yml"
    
    # 실행
    qpos_results, joint_list = run_dexycb_retargeting(DEXYCB_PATH, CONFIG_PATH)
    
    # 결과를 npy 파일로 저장하고 싶다면 주석 해제
    # np.save(f"results_qpos.npy", qpos_results)