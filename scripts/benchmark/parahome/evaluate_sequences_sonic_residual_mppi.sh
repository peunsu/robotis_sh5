#!/usr/bin/env bash
# =============================================================================
# evaluate_sequences_sonic_residual_mppi.sh — Roll out each SONIC-residual clip trained on the MPPI
# hand refine (train_sequences_sonic_residual_mppi.sh) and aggregate metrics.
#
# evaluate_sequences_sonic_residual.sh for the MPPI stage-1 variant: task
# Robotis-G1-Shadow-Locomanip-SonicResidual-Mppi-Direct-v0 (the env loads the same MPPI hand targets,
# residual base and density object mass as in training) and the g1_shadow_sonic_residual_mppi tree.
# Everything below is unchanged from that script. rollout.py never arms the RSI warm-start, so
# evaluation runs the policy with vanilla frame-0 resets (same protocol as the
# other variants). The frozen SONIC base + the per-clip sonic_smpl_50fps.npz /
# trajectory_pyroki.npz produced during training are consumed by the env at load;
# no extra prep here (evaluation is env_isaaclab only — no PyRoki needed).
#
# For each clip:
#   1. Load agent.pt from the g1_shadow_sonic_residual_mppi tree.
#   2. rollout.py → metrics.csv (obj pos/rot + whole-body kpt + fingertip errors).
#   3. After all clips, evaluate.bash aggregates per-method CSVs.
#
# Output tree (evaluate.bash compatible — clip_name lands at path parts[1]):
#   data/processed/parahome/g1_shadow_sonic_residual_mppi/<clip_class>/<clip_name>/0/
#       agent.pt / pretrain.pt
#       evaluation_ep_le_<TIMESTEPS>/metrics.csv
# Aggregates → data/processed/parahome/g1_shadow_sonic_residual_mppi/method{1,2,3}.csv
#
# Which clips run: edit CLIPS=(...) (comment lines to filter) or CLIPS_OVERRIDE="a b c".
# Keep in sync with train_sequences_sonic_residual_mppi.sh.
# Env vars: FORCE=1 (re-run rollouts), CLIP_CLASS=..., CLIPS_OVERRIDE="a b c",
#   N_ROLLOUTS, TIMESTEPS (dir tag only), VIDEO=1, PY=<env_isaaclab python>,
#   SEED=42, TWO_CAM=1, SBS=1, VIDEO_RESOLUTION=WxH.
#
# TWO-CAMERA VIDEO (TWO_CAM=1, default): the SAME rollout is replayed with --cam_preset old
# (yaw 45 / elev 0 / root-aimed = the pre-2026-07-22 view) and the two mp4s are composed into a
# labelled side-by-side. Pass 2 writes metrics_camold.csv, and compose_side_by_side.py checks it
# against metrics.csv as a PHYSICS RECEIPT.
#
# The replay is trustworthy but NOT bit-reproducible, and that distinction is measured, not assumed
# (s100_seg00_pan, seed 42): two byte-identical invocations with the SAME camera already differ by up
# to 1.3e-1 relative at 32 envs (1.2e-3 at 1 env) because the PhysX GPU solver is not reproducible
# across processes. The actual camera-change pair differed by only 6.8e-2 — BELOW the same-camera
# floor — so the camera angle contributes nothing, and frame-wise pixel divergence between two
# same-camera reruns is FLAT over the clip (it does not accumulate). Hence the receipt is a tolerance
# check plus exact agreement of the discrete success flags, not `cmp`. Lower N_ROLLOUTS (e.g. 1, since
# the video only ever shows env 0) if you want the residual divergence ~100x smaller.
# evaluate.bash globs exactly '**/metrics.csv', so metrics_camold.csv never enters aggregation.
#
# NOTE the two views are produced by two RENDER PASSES, not by two camera sensors: constructing an
# isaaclab.sensors.Camera sets the process-global /isaaclab/render/rtx_sensors flag, which moves the
# sole sim.render() into the physics loop and would retime the "current" pane.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────
TASK="Robotis-G1-Shadow-Locomanip-SonicResidual-Mppi-Direct-v0"
CLIP_CLASS="${CLIP_CLASS:-single_rigid}"
N_ROLLOUTS="${N_ROLLOUTS:-32}"
TIMESTEPS="${TIMESTEPS:-60000}"   # directory-naming tag only (evaluation_ep_le_<TIMESTEPS>)
# PY: 환경변수 → 저장소 루트 local_paths.env (git 제외) 순으로 읽는다.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/local_paths.sh"
require_local_path PY
VIDEO="${VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-0}"  # 0 → rollout.py fits the full sequence
SEED="${SEED:-42}"                 # passed EXPLICITLY to both passes (determinism contract)
TWO_CAM="${TWO_CAM:-1}"            # 0 → single pass, identical to the pre-two-camera script
SBS="${SBS:-1}"                    # 0 → keep the two mp4s separate, skip the compose step
NEW_PREFIX="rl-video"              # pass-1 mp4 stem (unchanged → existing tooling keeps working)
OLD_PREFIX="rl-video-oldcam"       # pass-2 mp4 stem
OLD_METRICS="metrics_camold.csv"   # pass-2 CSV = determinism receipt (invisible to evaluate.bash)
RES_ARGS=(); [[ -n "${VIDEO_RESOLUTION:-}" ]] && RES_ARGS=(--video_resolution "${VIDEO_RESOLUTION}")

# Clip names to evaluate. Comment out lines to filter (like the hocap scripts). Keep this list in
# sync with train_sequences_sonic_residual_mppi.sh. Or override with CLIPS_OVERRIDE="clipA clipB".
CLIPS=(
    # the first 8 (train_sequences_sonic_residual.sh's set, with RL hand-pretrain results to compare against)
    "s100_seg00_pan"
    "s101_seg12_knife"
    "s101_seg29_pot"
    "s101_seg30_bowl"
    "s66_seg26_pan"
    "s53_seg19_knife"
    "s152_seg21_pot"
    "s71_seg27_bowl"
    # added 2026-09-29: three clips per ParaHome single_rigid object
    "s12_seg07_book"
    "s11_seg07_book"
    "s155_seg11_book"
    "s126_seg19_bowl"
    "s109_seg02_cup"
    "s105_seg21_cup"
    "s101_seg00_cup"
    "s197_seg07_cuttingboard"
    "s145_seg22_cuttingboard"
    "s198_seg29_cuttingboard"
    "s117_seg09_kettle"
    "s113_seg01_kettle"
    "s207_seg06_kettle"
    "s125_seg23_knife"
    "s104_seg18_pan"
    "s142_seg00_pot"
    "s44_seg00_potlid"
    "s1_seg06_potlid"
    "s92_seg16_potlid"
    "s123_seg21_salt"
    "s126_seg14_salt"
    "s111_seg20_salt"
)
# Optional: replace the whole list from the environment (space-separated).
[[ -n "${CLIPS_OVERRIDE:-}" ]] && read -ra CLIPS <<< "${CLIPS_OVERRIDE}"

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_BASE="${PROJECT_DIR}/source/robotis_sh5/data/processed/parahome"
CHECKPOINT_BASE="${DATA_BASE}/g1_shadow_sonic_residual_mppi/${CLIP_CLASS}"
FORCE="${FORCE:-0}"

# ── Rollout loop ──────────────────────────────────────────────────────────────
TOTAL="${#CLIPS[@]}"
IDX=0
for clip in "${CLIPS[@]}"; do
    IDX=$(( IDX + 1 ))
    CKPT_FILE="${CHECKPOINT_BASE}/${clip}/0/agent.pt"
    EVAL_TAG="evaluation_ep_le_${TIMESTEPS}"
    OUT_DIR="${CHECKPOINT_BASE}/${clip}/0/${EVAL_TAG}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[eval-sonic-mppi ${IDX}/${TOTAL}] class=${CLIP_CLASS}  clip=${clip}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ ! -f "${CKPT_FILE}" ]]; then
        echo "[eval-sonic-mppi] ERROR: checkpoint not found — run train_sequences_sonic_residual_mppi.sh first.  ${CKPT_FILE}"
        continue
    fi
    # Skip the SIMULATION but still let the compose step run on existing mp4s, so a side-by-side can
    # be (re)built without paying for the rollouts again.
    DO_ROLLOUT=1
    if [[ -f "${OUT_DIR}/metrics.csv" && "${FORCE}" -eq 0 ]]; then
        echo "[eval-sonic-mppi] metrics.csv exists — skipping rollouts.  (FORCE=1 to override)"
        DO_ROLLOUT=0
    fi

    VIDEO_ARGS=()
    if [[ "${VIDEO}" -eq 1 ]]; then
        VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}")
    fi

    cd "${PROJECT_DIR}"
    if [[ "${DO_ROLLOUT}" -eq 1 ]]; then
        # ── PASS 1/2 — CANONICAL: cfg camera + metrics.csv. No --cam_preset, so the cfg's own
        #    viewer_* fields are untouched and this pass is behaviourally identical to before
        #    (only --seed, previously the argparse default 42, is now explicit).
        echo "[eval-sonic-mppi] pass 1/2 — CURRENT camera (cfg viewer_yaw/elev/look_obj) → ${NEW_PREFIX}-step-0.mp4"
        "${PY}" scripts/skrl/rollout.py \
            --task "${TASK}" --checkpoint "${CKPT_FILE}" \
            --output_dir "${OUT_DIR}" --n_rollouts "${N_ROLLOUTS}" --headless \
            --seed "${SEED}" "${VIDEO_ARGS[@]}" "${RES_ARGS[@]}" \
            --clip_class "${CLIP_CLASS}" --clip_name "${clip}" \
            --video_name_prefix "${NEW_PREFIX}"

        # ── PASS 2/2 — SAME rollout, PREVIOUS camera angle. Identical seed / checkpoint / clip and
        #    a deterministic policy, so this re-simulates the same trajectory; only
        #    cfg.viewer.eye/lookat differ. Separate mp4 AND separate CSV.
        if [[ "${VIDEO}" -eq 1 && "${TWO_CAM}" -eq 1 ]]; then
            echo "[eval-sonic-mppi] pass 2/2 — PREVIOUS camera (yaw 45 / elev 0 / root-aimed) → ${OLD_PREFIX}-step-0.mp4"
            "${PY}" scripts/skrl/rollout.py \
                --task "${TASK}" --checkpoint "${CKPT_FILE}" \
                --output_dir "${OUT_DIR}" --n_rollouts "${N_ROLLOUTS}" --headless \
                --seed "${SEED}" "${VIDEO_ARGS[@]}" "${RES_ARGS[@]}" \
                --clip_class "${CLIP_CLASS}" --clip_name "${clip}" \
                --cam_preset old \
                --video_name_prefix "${OLD_PREFIX}" --metrics_name "${OLD_METRICS}"
        fi
    fi

    # ── COMPOSE side-by-side + PHYSICS RECEIPT (CPU only; safe with a busy GPU; rerunnable) ────
    # The receipt lives in compose_side_by_side.py (--left_metrics/--right_metrics), not here: it is a
    # TOLERANCE check against the measured same-camera noise floor, not `cmp`. Byte-equality is
    # unreachable — two identical invocations already differ (PhysX GPU is not reproducible across
    # processes), and the camera-change pair diverges LESS than a same-camera pair does.
    if [[ "${SBS}" -eq 1 && "${VIDEO}" -eq 1 && "${TWO_CAM}" -eq 1 ]]; then
        "${PY}" scripts/benchmark/parahome/compose_side_by_side.py \
            --dir "${OUT_DIR}" --left_prefix "${NEW_PREFIX}" --right_prefix "${OLD_PREFIX}" \
            --left_metrics metrics.csv --right_metrics "${OLD_METRICS}" \
            || echo "[eval-sonic-mppi] WARNING: compose failed; the two single-view mp4s are still on disk."
    fi
done

# ── Aggregate metrics ─────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[eval-sonic-mppi] Aggregating metrics under ${DATA_BASE} ..."
bash "${SCRIPT_DIR}/../evaluate.bash" "${DATA_BASE}"
echo ""
echo "[eval-sonic-mppi] Done.  → ${DATA_BASE}/g1_shadow_sonic_residual_mppi/method{1,2,3}.csv"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
