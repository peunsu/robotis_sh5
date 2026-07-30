"""Pretrain config for the G1+Shadow loco-manip task (kinematic-only warm-start phase).

Subclasses the train cfg — SAME obs/action dims (703/65) so `pretrain.pt` transfers to the
train phase by shape. Object terms (obj_pos/obj_rot reward + fingertip-force reward) become
inert automatically because `_has_object` is False (the env subclass never spawns an object),
so they need no cfg change here. The one PHASE-SPECIFIC reward-weight override mirrors the
proven grasp RSI pretrain (rew_fingertip -5.2→-12.5 there): with no object physics and no
contact-force reward to guide the fingers, the fingertip POSITION weight is the sole finger
signal, so it is boosted so the fingers learn to reach the reference pads kinematically before
the train phase adds contact.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .g1_shadow_locomanip_env_cfg import G1ShadowLocomanipEnvCfg


@configclass
class G1ShadowLocomanipPretrainEnvCfg(G1ShadowLocomanipEnvCfg):
    # Pretrain never consumes a warm-start cache (it dumps one for the train phase).
    pretrain_cache_warmstart: bool = False
    # PRETRAIN sampling = pure UNIFORM over cached frames (no failure weighting), mirroring grasp
    # pretrain. The train phase uses failure-weighted sampling (cfg default True).
    failure_weighted_sampling: bool = False
    # Phase-specific: boost fingertip-position tracking ~2.4× vs train (-5.0), mirroring grasp
    # pretrain (-12.5 vs train -5.2). No contact-force/object shaping in pretrain → position is
    # the only fingertip signal. Still well within the -rew_alive(4.0) clamp floor at good tracking.
    rew_fingertip: float = -12.5

    term_body_kpt_err: float = 0.30        # m, mean body-keypoint tracking error (covers root/fall).
    term_obj_pos_err: float = 0.15         # m, active-object position tracking error (= grasp max_obj_pos_err; was loosened to 0.20, now grasp-parity)
    term_obj_rot_err: float = 0.75         # rad, active-object rotation tracking error (= grasp max_obj_rot_err; was 0.80)
    term_hand_kpt_err: float = 0.15        # m, mean finger-chain (hand) keypoint tracking error (grasp-parity: grasp's hand-region tol = max_ft_mean_err/max_wrist_pos_err 0.15; was loosened to 0.25)
    term_ft_err: float = 0.15
    term_wrist_rot_err: float = 0.75       # rad, per-hand palm-rotation deviation (= grasp max_wrist_rot_err)

    enough_ft_threshold: float = 0.10          # m, mean fingertip err (= grasp enough_ft_threshold; was loosened to 0.13)
    enough_obj_threshold: float = 0.085        # m, early-phase obj pos err (= grasp; was 0.11)
    enough_obj_rot_threshold: float = 0.425    # rad, early-phase obj rot err (= grasp; was 0.45)
    enough_obj_threshold_late: float = 0.05    # m, late-phase obj pos err (= grasp; was 0.07)
    enough_obj_rot_threshold_late: float = 0.25  # rad, late-phase obj rot err (= grasp; was 0.27)
    cache_body_bar: float = 0.25       # < term_body_kpt_err 0.25 (~0.65×); seed body err ~0.066 passes
    cache_root_pos_bar: float = 0.10
    cache_root_rot_bar: float = 0.30