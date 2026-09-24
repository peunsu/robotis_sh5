"""G1 + 양손 Shadow 전신 loco-manipulation (SONIC 잔차) 환경 설정.

로봇은 Unitree G1 몸 29 DOF 에 Shadow 손 2개(각 18 구동)를 붙인 65 구동 DOF, floating base 다.
정책은 리타게팅된 ParaHome SMPL-X 전신 동작과 물체 궤적을 추종한다.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

_DATA_DIR = Path(__file__).resolve().parents[4] / "data"


def instanced_usd(src: Path) -> Path:
    """링크 형상을 instanceable 로 바꾼 시뮬레이션용 사본 `<이름>_inst.usd` (원본은 오프라인 도구용)."""
    inst = src.with_name(f"{src.stem}_inst{src.suffix}")
    if not inst.exists() or (src.exists() and inst.stat().st_mtime < src.stat().st_mtime):
        raise FileNotFoundError(f"{inst} 이 없거나 {src.name} 보다 오래됐습니다 — "
                                f"scripts/process_dataset/assets/instance_link_geometry.py 를 실행하세요")
    return inst


_ROBOT_USD = str(instanced_usd(_DATA_DIR / "robots" / "G1" / "G1_shadow.usd"))


# 추적 몸 키포인트: ParaHome 관절 인덱스 → G1 링크
BODY_KPTS: dict[int, str] = {
    0:  "pelvis",  # 루트
    17: "right_shoulder_pitch_link",
    19: "right_elbow_link",
    21: "right_wrist_yaw_link",
    16: "left_shoulder_pitch_link",
    18: "left_elbow_link",
    20: "left_wrist_yaw_link",
    2:  "right_hip_pitch_link",
    5:  "right_knee_link",
    8:  "right_ankle_roll_link",
    1:  "left_hip_pitch_link",
    4:  "left_knee_link",
    7:  "left_ankle_roll_link",
}

# 손 키포인트 체인: ParaHome 손 관절 인덱스 ↔ Shadow 링크 (pad=True 면 손끝 pad 오프셋 적용)
HAND_CHAIN: dict[str, dict] = {
    "wrist":  {"parahome": [-10],            "shadow": ["palm"],           "pad": [False]},
    "index":  {"parahome": [0, 1, 2, -2],    "shadow": ["ffknuckle", "ffmiddle", "ffdistal", "ffdistal"],
               "pad": [False, False, False, True]},
    "middle": {"parahome": [3, 4, 5, -3],    "shadow": ["mfknuckle", "mfmiddle", "mfdistal", "mfdistal"],
               "pad": [False, False, False, True]},
    "ring":   {"parahome": [9, 10, 11, -4],  "shadow": ["rfknuckle", "rfmiddle", "rfdistal", "rfdistal"],
               "pad": [False, False, False, True]},
    "pinky":  {"parahome": [6, 7, 8, -5],    "shadow": ["lfknuckle", "lfmiddle", "lfdistal", "lfdistal"],
               "pad": [False, False, False, True]},
    "thumb":  {"parahome": [12, 13, 14, -1], "shadow": ["thbase", "thmiddle", "thdistal", "thdistal"],
               "pad": [False, False, False, True]},
}

# 손끝 pad 오프셋과 pad 바깥 법선 (오른손 링크 좌표, TJ gr 과 같음). 왼손은 Y 대칭으로 만든다.
_FT_OFFSET_BASE: dict[str, list[float]] = {
    "thdistal": [-0.0085, 0.0, 0.02],
    "ffdistal": [0.0, -0.006, 0.0175],
    "mfdistal": [0.0, -0.006, 0.0175],
    "rfdistal": [0.0, -0.006, 0.0175],
    "lfdistal": [0.0, -0.006, 0.0175],
}
_FT_NORMAL_BASE: dict[str, list[float]] = {
    "thdistal": [-1.0, 0.0, 0.0],
    "ffdistal": [0.0, -1.0, 0.0],
    "mfdistal": [0.0, -1.0, 0.0],
    "rfdistal": [0.0, -1.0, 0.0],
    "lfdistal": [0.0, -1.0, 0.0],
}


def _both_hands(base: dict[str, list[float]]) -> dict[str, list[float]]:
    """오른손 링크 좌표 값을 양손 dict 로. 왼손은 Y 대칭."""
    out: dict[str, list[float]] = {}
    for body, v in base.items():
        out[f"robot0_r_{body}"] = list(v)
        out[f"robot0_l_{body}"] = [v[0], -v[1], v[2]]
    return out


FINGERTIP_OFFSETS: dict[str, list[float]] = _both_hands(_FT_OFFSET_BASE)
FINGERTIP_PAD_NORMALS: dict[str, list[float]] = _both_hands(_FT_NORMAL_BASE)

N_BODY_KPTS = len(BODY_KPTS)  # 13
# 21 (손목 1 + 손가락 15 + 손끝 5)
N_HAND_KPTS_PER_HAND = sum(len(v["parahome"]) for v in HAND_CHAIN.values())

# 접촉 보상에 쓰는 wrap 링크: 양손 × (palm + 5손가락 × 3마디) = 32
_LINK_CONTACT_FINGERS = ["ff", "mf", "lf", "rf", "th"]
_LINK_CONTACT_SEGS = ["proximal", "middle", "distal"]
LINK_CONTACT_NAMES: list[str] = [
    f"robot0_{s}_{b}"
    for s in ("l", "r")
    for b in (["palm"] + [f"{fg}{seg}" for fg in _LINK_CONTACT_FINGERS for seg in _LINK_CONTACT_SEGS])
]
N_LINK_CONTACT = len(LINK_CONTACT_NAMES)  # 32

# 링크별 바깥쪽 접촉면 법선 (오른손 링크 좌표). 접촉력은 이 면의 안쪽 방향으로 투영한다.
_LINK_NORMAL_BASE: dict[str, list[float]] = {
    "palm": [0.0, -1.0, 0.0],
    **{f"{fg}{seg}": [0.0, -1.0, 0.0] for fg in ("ff", "mf", "lf", "rf") for seg in _LINK_CONTACT_SEGS},
    **{f"th{seg}": [-1.0, 0.0, 0.0] for seg in _LINK_CONTACT_SEGS},
}
LINK_PAD_NORMALS: dict[str, list[float]] = _both_hands(_LINK_NORMAL_BASE)


# 액션 관절 그룹 (순서 = legs → waist → arms → hands). 몸 29 는 SONIC 이, 손 36 은 정책이 구동한다.
JOINT_GROUPS: dict[str, dict] = {
    "legs":  {"expr": [".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
                       ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
              "dof": 12},
    "waist": {"expr": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
              "dof": 3},
    "arms":  {"expr": [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                       ".*_elbow_joint", ".*_wrist_pitch_joint", ".*_wrist_roll_joint", ".*_wrist_yaw_joint"],
              "dof": 14},
    "hands": {"expr": ["robot0_(l|r)_(FF|MF|RF|LF|TH)J[1-3]",
                       "robot0_(l|r)_LFJ4", "robot0_(l|r)_THJ4", "robot0_(l|r)_THJ0"],
              "dof": 36},
}


# G1 + 양손 Shadow 합성 USD.
G1_SHADOW_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ROBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        # J0-J1 텐던 결합
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.2),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        rot=(0.7071, 0.0, 0.0, 0.7071),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 몸 29관절: SONIC 이 학습된 게인과 같아야 한다 (sonic_prior 값).
        "sonic_hip_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"],
            effort_limit_sim=139.0, velocity_limit_sim=20.0,
            stiffness=99.0997, damping=6.3088, armature=0.025101925),
        "sonic_hipyaw_waistyaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", "waist_yaw_joint"],
            effort_limit_sim=88.0, velocity_limit_sim=32.0,
            stiffness=40.1795, damping=2.5579, armature=0.010177520),
        "sonic_ankle_waist": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint",
                              "waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0, velocity_limit_sim=37.0,
            stiffness=28.5013, damping=1.8143, armature=0.00721945),
        "sonic_shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                              ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"],
            effort_limit_sim=25.0, velocity_limit_sim=37.0,
            stiffness=14.2506, damping=0.9072, armature=0.003609725),
        "sonic_wrist_pitchyaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit_sim=5.0, velocity_limit_sim=22.0,
            stiffness=16.7783, damping=1.0681, armature=0.00425),
        # Shadow 손가락 18 × 2 (J0 는 텐던으로 J1 에 결합)
        "shadow_fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "robot0_(l|r)_(FF|MF|RF|LF|TH)J[1-3]",
                "robot0_(l|r)_LFJ4", "robot0_(l|r)_THJ4", "robot0_(l|r)_THJ0",
            ],
            velocity_limit_sim=15.0, effort_limit_sim=3.09,
            stiffness=1.0, damping=0.2,
        ),
    },
)


@configclass
class G1ShadowSonicResidualEnvCfg(DirectRLEnvCfg):
    # ---- 로봇 / 제어 ----
    robot_cfg: ArticulationCfg = G1_SHADOW_CFG  # G1 + 양손 Shadow, floating base
    waist_gain_scale: float = 1.5  # 허리 3관절 PD 게인 배수 (액션 스케일은 그대로)
    ankle_gain_scale: float = 1.5  # ankle_pitch PD 게인 배수 (SONIC v1.1 배포 튜닝과 같음)
    decimation: int = 4  # 200 Hz 물리 / 4 = 50 Hz 제어 (SONIC 제어 주기)

    # ---- 액션 ----
    sonic_action_dim: int = 64  # z_res 잠재 잔차 폭
    hand_action_dim: int = 36  # 양손 손가락
    finger_ema_alpha: float = 0.5  # 손가락 액션 EMA α (새 액션 가중치)
    soft_joint_pos_limit_factor_hands: float = 1.0  # 손가락 soft 관절 한계 factor (1.0 = PhysX 한계 그대로)
    action_space: int = 64 + 36  # 상체 관절 잔차를 켜면 __init__ 에서 +17

    # ---- SONIC 잔차 (잠재 / 상체 관절 / 손) ----
    sonic_upper_residual: bool = True  # 상체 17관절 잔차. 켜고 끄면 action/obs 가 ±17 → 체크포인트 비호환
    sonic_residual_joints_mode: str = "upper"  # "upper" = 허리+팔 17관절, "all" = 몸 29관절
    sonic_latent_residual: bool = True  # 끄면 액션에서 z_res 64 차원이 빠진다 → 체크포인트 비호환
    # 관절 잔차 scale (SONIC 액션 단위, 그룹별)
    sonic_residual_scale_groups: dict = {"legs": 0.15, "waist": 0.15, "arms": 0.50}
    sonic_upper_residual_joints: list = [  # 허리 3 + 팔 14
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
        "left_shoulder_roll_joint", "right_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
        "left_elbow_joint", "right_elbow_joint",
        "left_wrist_roll_joint", "right_wrist_roll_joint",
        "left_wrist_pitch_joint", "right_wrist_pitch_joint",
        "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    ]
    zero_actor_residual: bool = True  # 정책 마지막 층의 관절 잔차 출력을 0 으로 초기화 (train.py)
    zero_actor_latent: bool = True  # 정책 마지막 층의 z_res 출력을 0 으로 초기화 (train.py)
    sonic_hand_residual: bool = True  # 손 액션 = 기준 자세 대비 margin 잔차
    sonic_hand_residual_base: str = "hand_pretrain_target"  # "hand_pretrain_target" | "hand_pretrain_qpos" | "reference"
    sonic_z_res_clip: float = 5.0  # env 가 z_res 를 [-clip, clip] 으로 자른다 (PPO log-prob 은 자르기 전 값)

    # ---- 관측 / 시뮬레이션 ----
    # 상체 관절 잔차를 켜면 __init__ 에서 +17. 역방향 롤아웃 비트 자리(항상 0) 포함
    observation_space: int = 772
    state_space: int = 0  # 별도 critic 상태 없음
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.80665),
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            gpu_max_rigid_patch_count=1024 * 1024 * 4,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0)
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.5, 1.5, 1.0),
        lookat=(0.0, 0.0, 0.8),
        origin_type="env",
        env_index=0,
    )

    # ---- 시각화 / 로깅 ----
    debug_vis: bool = True  # 레퍼런스/로봇 키포인트 마커
    debug_vis_num_envs: int = 64  # 마커를 그릴 env 수
    log_reward_diag: bool = True  # 항별 보상과 진단 지표를 tensorboard 에 기록
    viewer_zoom: float = 0.6  # 학습 영상 카메라 거리 배율 (작을수록 가까이)
    viewer_yaw: float = 270.0  # deg
    viewer_elev: float = 18.0  # deg
    viewer_look_obj: bool = True  # True 면 물체를, False 면 루트를 조준

    # ---- 추적 링크 ----
    fingertip_body_names: list = [  # 왼손 → 오른손, 엄지·검지·중지·약지·소지
        "robot0_l_thdistal", "robot0_l_ffdistal", "robot0_l_mfdistal", "robot0_l_rfdistal", "robot0_l_lfdistal",
        "robot0_r_thdistal", "robot0_r_ffdistal", "robot0_r_mfdistal", "robot0_r_rfdistal", "robot0_r_lfdistal",
    ]

    # ---- 추적 목표 ----
    body_kpt_from_retarget_fk: bool = True  # 몸통 키포인트 목표를 SMPL-X 대신 리타게팅 로봇 FK 로

    # ---- hand_pretrain(1단계) 결과 사용 ----
    # ablation 은 sonic_hand_residual_base 와 아래 세 플래그를 함께 바꾼다.
    hand_kpt_from_hand_pretrain: bool = True  # 손·손목 목표를 hand_pretrain 롤아웃 키포인트로
    hand_pretrain_subdir: str = "g1_shadow_hand_pretrain"  # data/processed/parahome/ 아래 hand_pretrain 결과 트리
    hand_pretrain_traj_file: str = "hand_traj_best.npz"
    hand_pretrain_contact_vertex: bool = True  # 접촉 프레임 손끝 목표를 hand_pretrain pad 위치로
    hand_pretrain_contact_map: bool = True  # 32링크 접촉 맵을 hand_pretrain 롤아웃 것으로
    hand_pretrain_contact_map_file: str = "hand_contact_stage1.npz"  # 파일 이름은 기존 데이터 그대로
    cws_ref_contact_source: str = "human"  # "human" | "hand_pretrain": CWS 레퍼런스 렌치에 쓸 접촉 맵

    # ---- 보상: 추적 (exp 가중치는 |rew_*| 를 exp_tracking_budget 으로 정규화해 만든다) ----
    rew_body_kpts: float = -1.0  # 몸통 코어 키포인트 (골반·어깨·팔꿈치·엉덩이·무릎)
    rew_ee_kpts: float = -3.0  # 손목 2 + 발목 2
    rew_hand_kpts: float = -3.0  # 손가락 체인 키포인트
    rew_fingertip: float = -6.0  # 손끝 pad 10개
    rew_root_pos: float = -1.0  # 루트(골반) 위치
    rew_root_ori: float = -1.0  # 루트(골반) 방향
    z_weight_reward: float = 1.5  # 손·물체 위치 오차의 z 성분 가중 (보상에만)
    rew_obj_pos: float = -5.0  # 물체 위치
    rew_obj_rot: float = -1.5  # 물체 회전

    # ---- 보상: 링크 접촉력 ----
    rew_contact_force: float = 0.7  # contact_reward_mode 가 "force"/"both" 일 때만 보상에 들어간다
    contact_vel_gate: bool = True  # 레퍼런스 물체가 멈춘 프레임은 접촉력 보상을 끈다
    contact_vel_gate_lin: float = 0.05  # m/s
    contact_vel_gate_ang: float = 0.25  # rad/s
    contact_force_cap: float = 1.0  # N, 링크당 이 힘 이상이면 만점
    force_obs_clip: float = 300.0  # N, 관측의 접촉력 상한 (보상에는 영향 없음)

    # ---- 보상: exp 추적 형태 ----
    exp_tracking_budget: float = 1.50  # 추적 항 전체가 한 스텝에 줄 수 있는 총 보상
    exp_rew_alive: float = 0.5  # 생존 보너스

    # σ = 그 항이 exp(-1) = 0.37 을 지급하는 오차 (m / rad)
    sigma_body: float = 0.10
    sigma_ee: float = 0.075
    sigma_hand: float = 0.075
    sigma_fingertip: float = 0.05
    sigma_root_pos: float = 0.15
    sigma_root_rot: float = 0.30
    sigma_obj_pos: float = 0.05
    sigma_obj_rot: float = 0.30

    # ---- NaN 가드 ----
    nan_guard_joint_vel: float = 100.0  # rad/s, |q̇| 가 넘으면 강제 리셋 (0 = 끔)
    nan_guard_object: bool = True  # 물체 상태 유한성도 리셋 게이트에 포함
    nan_guard_obs_clip: float = 1.0e4  # 관측 비유한값 치환 + 크기 제한 (0 = 끔)

    # ---- 보상: 정규화 / 안정성 ----
    rew_action_reg: float = -0.002  # z_res + 손 액션 제곱합
    rew_pose_reg_hands: float = -0.001  # 손 관절을 기본 자세 쪽으로
    rew_action_rate: float = -0.001  # 액션 변화량
    rew_energy: float = -0.0001  # 허리+다리 Σ|τ·q̇|
    rew_ankle_acc: float = -1e-7  # 발목 관절 가속도
    ankle_acc_joints: list = [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
    rew_waist_acc: float = -1e-7  # 허리 관절 가속도
    waist_acc_joints: list = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    rew_anti_shake: float = -0.005  # 손목·머리 각속도의 데드존 초과분
    anti_shake_ang_vel_thresh: float = 1.5  # rad/s 데드존
    anti_shake_bodies: list = ["left_wrist_yaw_link", "right_wrist_yaw_link", "head_link"]
    rew_com_support: float = -0.5  # CoM 투영이 지지 영역을 벗어난 거리 (0 = 끔)
    com_support_l_front: float = 0.10  # m, 발목 → 발끝 지지 반길이
    com_support_l_back: float = 0.05  # m, 발목 → 뒤꿈치 지지 반길이
    com_support_foot_halfw: float = 0.03  # m, 좌우 지지 반폭에 더하는 값
    rew_feet_contact_match: float = 0.05  # 발 접촉이 레퍼런스와 맞으면 보너스 (0 = 끔)
    foot_plant_h: float = 0.05  # m, 레퍼런스 발 높이가 이보다 낮으면 접지
    foot_plant_v: float = 1.00  # m/s, 레퍼런스 발 속도가 이보다 느리면 접지
    foot_contact_force_thresh: float = 20.0  # N, 실제 발-지면 힘이 이 이상이면 접지

    # ---- 종료 조건 (레퍼런스 이탈) ----
    term_body_kpt_err: float = 0.50  # m, 몸 키포인트 평균 오차 (낙상 포함)
    term_obj_pos_err: float = 0.15  # m, 물체 위치 오차
    term_obj_rot_err: float = 0.75  # rad, 물체 회전 오차
    term_ft_err: float = 0.15  # m, 손끝 평균 오차
    term_wrist_pos_err: float = 0.15  # m, 손목 위치 이탈 (두 손 중 큰 쪽)
    term_wrist_rot_err: float = 0.75  # rad, 손바닥 회전 이탈
    termination: bool = True  # 종료 조건 전체 스위치 (평가 때 False)

    # ---- 물체 / 맥락 물체 ----
    freeze_inactive_objects: bool = True  # 조작하지 않는 물체를 정적 지지물로 스폰
    context_radius: float = 1.0  # m, 물체 궤적 주변에서 맥락 물체를 고르는 XY 반경
    context_support_radius: float = 1.5  # m, 물체 아래 지지물을 찾는 XY 반경
    context_z_auto: bool = True  # 스폰 뒤 측정한 만큼 맥락 물체를 내려 물체가 레퍼런스 높이에 놓이게 한다

    # ---- RSI / 시작 프레임 샘플링 ----
    use_rsi: bool = True  # False 면 항상 frame 0 레퍼런스 자세에서 시작

    # ---- 물체 스폰 관통 해소 ----
    object_spawn_declear: bool = True  # 스폰 때 물체를 지지면에서 띄워 관통을 없앤다
    declear_settle_steps: int = 12  # 물체가 멈추는 위치를 재는 자유 정착 스텝 수
    declear_max_lift: float = 0.05  # m, 띄우는 양 상한
    declear_rest_lin: float = 0.05  # m/s, 이보다 느리면 정지
    declear_rest_ang: float = 0.25  # rad/s, 이보다 느리면 정지

    # ---- 적응 샘플링 / 상태 캐시 ----
    adaptive_sampling: bool = True  # 실패 가중 시작 프레임 샘플링
    failure_weighted_sampling: bool = True  # True = 실패 가중, False = 균등 (pretrain)
    cache_min_episode_length: int = 10  # 이보다 짧게 끝난 에피소드는 캐시에 쓰지 않는다 (종료 때 일괄 기록)
    adaptive_alpha: float = 0.001  # 실패 카운트 EMA
    adaptive_uniform_ratio: float = 0.1  # 균등 샘플링을 섞는 비율
    adaptive_back_seconds: float = 1.0  # s, 샘플한 프레임보다 이만큼 앞에서 시작 (run-up)
    adaptive_back_min_seconds: float = 0.6  # s, run-up 을 [이 값, adaptive_back_seconds] 에서 무작위로 (0 = 고정)
    ref_start_prob: float = 0.01  # 이 확률로 캐시를 무시하고 레퍼런스 자세에서 시작

    # ---- 리셋 상태 ----
    ref_reset_joint_vel: bool = True  # 레퍼런스 리셋에 관절 속도도 넣는다
    ref_reset_joint_vel_scale: float = 1.0  # 넣는 배율
    ref_reset_joint_vel_clip: float = 10.0  # rad/s, 몸 29관절 상한
    ref_reset_joint_vel_clip_hands: float = 10.0  # rad/s, 손 36관절 상한

    # ---- 샘플링 일정 ----
    late_gate_survival_frac: float = 0.8  # late 캐시 기준으로 넘어가려면 클립의 이 비율 이상 생존해야 한다
    uniform_sampling_steps: int = 2000  # 처음 이 제어 스텝 동안은 균등 샘플링
    ref_dt: float = 1.0 / 30.0  # 레퍼런스 원본 프레임 간격 (30 fps)

    # ---- 상태 캐시 품질 게이트 (리셋 뒤 계속 이 기준 안이어야 캐시에 기록) ----
    enough_ft_threshold: float = 0.10  # m, 손끝 평균 오차
    enough_obj_threshold: float = 0.085  # m, 초반 물체 위치 오차
    enough_obj_rot_threshold: float = 0.425  # rad, 초반 물체 회전 오차
    enough_obj_threshold_late: float = 0.05  # m, 후반 물체 위치 오차
    enough_obj_rot_threshold_late: float = 0.25  # rad, 후반 물체 회전 오차
    enough_body_threshold: float = 0.30  # m, 몸 키포인트 평균 오차
    enough_root_pos_threshold: float = 0.10  # m, 루트 위치 오차
    enough_root_rot_threshold: float = 0.30  # rad, 루트 회전 오차

    # ---- 물체 마찰 커리큘럼 ----
    friction_curriculum: bool = True  # 물체 마찰을 높게 시작해 friction_min 까지 낮춘다
    friction_min: float = 1.0  # 최종 마찰 (물체 USD 기본값과 같음)
    friction_max_init: float = 3.0  # 초기 상한
    friction_decay_steps: int = 30000  # 제어 스텝

    # ---- 접촉 센서 ----
    track_contact_points: bool = True  # 센서 접촉점 기록 (CWS 에 필요)


    # ---- CWS 접촉 렌치 보상 ----
    contact_reward_mode: str = "cws"  # "cws" | "force" | "both"
    rew_cws: float = 0.50  # CWS 보상 가중치
    cws_beta: float = 0.1  # 로봇 렌치가 사람의 (1±beta)배 안이면 만점
    cws_v: float = 0.1  # 부족/과잉 벌점 세기
    cws_n_dir: int = 512  # 비교 방향 수 (논문 값)
    cws_n_edge: int = 8  # 마찰 원뿔 옆면 수 (논문 값)
    cws_link_chunk: int = 4  # 링크를 몇 개씩 나눠 계산할지 (0 = 한 번에, 메모리 주의)
    cws_mu: float = 0.1  # 마찰계수 (논문 값)
    cws_seed: int = 0  # 비교 방향 시드 (사람/로봇이 같은 방향을 써야 해서 고정)
    cws_force_thresh: float = 0.1  # N, 접촉으로 볼 최소 법선 힘
    contact_match_dist: float = 0.03  # m, 링크-접촉 목표 거리 허용치 (접촉력 보상 게이트)
    ft_max_contact_points: int = 64  # 링크당 접촉점 버퍼 (≤ 16 hull × 4점)

    # ---- 관측 스케일 ----
    vel_obs_scale: float = 0.2  # 관측의 각속도·관절 속도 배율

    # ---- SONIC 프라이어 ----
    sonic_config_path: str = "/home/peunsu/workspace/GR00T-WholeBodyControl/sonic_v1_1/config.yaml"
    sonic_ckpt_path: str = "/home/peunsu/workspace/GR00T-WholeBodyControl/sonic_v1_1/last.pt"
    sonic_encoder: str = "g1"  # "g1" | "smpl": SONIC 에 줄 명령 인코더
    sonic_token_frame_skip: int = 5  # g1 토큰의 미래 프레임 간격 (5 → 1초 창)
    residual_scale_latent: float = 0.10  # z_res 에 곱하는 λ (GRAIL)
    control_fps: float = 50.0  # 레퍼런스를 이 주기로 리샘플 (parahome_smpl_for_sonic 와 같아야 한다)
    sonic_smpl_file: str = "sonic_smpl_50fps.npz"  # 리타게팅 npz 옆의 SONIC SMPL 입력
    # 리셋 때 SONIC 10프레임 히스토리를 레퍼런스로 채운다 (False = 현재 상태 복제)
    sonic_hist_from_reference: bool = True

    # ---- 데이터 ----
    dataset_root: str = str(_DATA_DIR / "processed" / "parahome")
    smplx_subdir: str = "smplx"  # 키포인트·물체 레퍼런스 트리
    retarget_subdir: str = "g1_shadow"  # 리타게팅 결과 트리
    retarget_file: str = "trajectory_pyroki.npz"
    clip_class: str = "single_rigid"  # single_rigid | single_articulated | ...
    clip_name: str = ""  # "" 이면 clip_class 의 첫 클립
