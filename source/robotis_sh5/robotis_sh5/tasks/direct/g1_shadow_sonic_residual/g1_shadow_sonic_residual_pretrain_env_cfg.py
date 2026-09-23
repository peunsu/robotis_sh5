"""Kinematic pretrain (물체 없음) 설정. 학습 cfg 와 obs/action 차원이 같아 pretrain.pt 를 그대로 옮긴다."""

from __future__ import annotations

from isaaclab.utils import configclass

from .g1_shadow_sonic_residual_env_cfg import G1ShadowSonicResidualEnvCfg


@configclass
class G1ShadowSonicResidualPretrainEnvCfg(G1ShadowSonicResidualEnvCfg):
    use_rsi: bool = True
    failure_weighted_sampling: bool = False  # pretrain 은 균등 샘플링
    rew_fingertip: float = -12.5  # 물체가 없어 손끝 위치 가중을 키운다
    term_body_kpt_err: float = 0.30  # m
    term_obj_pos_err: float = 0.15  # m
    term_obj_rot_err: float = 0.75  # rad
    term_ft_err: float = 0.10  # m
    term_wrist_pos_err: float = 0.15  # m
    term_wrist_rot_err: float = 0.75  # rad
    enough_ft_threshold: float = 0.10  # m
    enough_obj_threshold: float = 0.085  # m
    enough_obj_rot_threshold: float = 0.425  # rad
    enough_obj_threshold_late: float = 0.05  # m
    enough_obj_rot_threshold_late: float = 0.25  # rad
    enough_body_threshold: float = 0.25  # m
    enough_root_pos_threshold: float = 0.10
    enough_root_rot_threshold: float = 0.30
