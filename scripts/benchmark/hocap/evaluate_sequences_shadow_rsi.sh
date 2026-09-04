#!/usr/bin/env bash
# =============================================================================
# evaluate_sequences_shadow_rsi.sh — Run rollouts for each HO-Cap sequence with
# the Shadow Hand RSI warm-start variant and aggregate metrics.
#
# Mirrors evaluate_sequences_shadow.sh, but targets the RSI task
# (Robotis-Shadow-Grasp-Rsi-Direct-v0). Checkpoints + metrics live in a SEPARATE
# tree (ffw_shadow_rsi) written by train_sequences_shadow_rsi.sh, so plain-Shadow
# and FFW-SH5 native-hand results remain untouched.
#
# NOTE: the pretrain-cache warm-start only affects TRAINING (it is armed by
# train.py). rollout.py never arms it, so evaluation runs the policy with vanilla
# resets — identical eval protocol to the other variants.
#
# For each sequence, this script:
#   1. Loads the checkpoint from the ffw_shadow_rsi checkpoints directory.
#   2. Runs rollout.py to produce metrics.csv.
#   3. After all rollouts, calls evaluate.bash to produce per-method CSV files.
#
# HO-Cap key format: subject_{N}-{DATE_TIME}-{G_OBJECT_ID}  e.g. subject_1-20231025_170231-G10_1
#
# Output structure (evaluate.bash compatible):
#   data/processed/hocap/ffw_shadow_rsi/right/<trajectory_task>/<data_id>/
#       pretrain.pt / agent.pt          ← checkpoints (written by train_sequences_shadow_rsi.sh)
#       evaluation_ep_le_<TIMESTEPS>/metrics.csv
#
# Aggregate CSVs are written to:
#   data/processed/hocap/ffw_shadow_rsi/method{1,2,3}.csv
#
# Set FORCE=1 to re-run rollouts even when metrics.csv already exists.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────

TASK="Robotis-Shadow-Grasp-Rsi-Direct-v0"
DATASET="hocap"
N_ROLLOUTS=32
# Must match the timesteps used during training (used for directory naming only).
TIMESTEPS="${TIMESTEPS:-40000}"

# Sequence keys — format matches mano/right folder names: subject_{N}-{DATE_TIME}-{G_OBJECT_ID}
# Default: same subset as train_sequences_shadow_rsi.sh.
SEQUENCES=(
    "subject_1-20231025_170231-G10_1"
    "subject_1-20231025_170231-G10_2"
    "subject_1-20231025_170231-G10_3"
    "subject_1-20231025_170231-G10_4"
    "subject_6-20231025_111357-G06_1"
    "subject_6-20231025_111357-G06_2"
    "subject_6-20231025_111357-G06_3"
    "subject_6-20231025_111357-G06_4"
    "subject_6-20231025_112332-G09_1"
    "subject_6-20231025_112332-G09_2"
    "subject_6-20231025_112332-G09_3"
    "subject_6-20231025_112332-G09_4"
)

# ── Path setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_BASE="${PROJECT_DIR}/source/robotis_sh5/data/processed/${DATASET}"
# RSI checkpoints live in their own tree. evaluate.bash automatically iterates
# over every subdirectory under data/processed/<dataset>/ that has metrics.csv.
CHECKPOINT_BASE="${DATA_BASE}/ffw_shadow_rsi/right"

FORCE="${FORCE:-0}"
# Set VIDEO=1 to record an mp4 of env 0's rollout into <OUT_DIR>/videos/
VIDEO="${VIDEO:-0}"
# 0 (default) → rollout.py auto-fits the clip to the full sequence length
# (max_episode_length). Set a positive value to force a fixed number of steps.
VIDEO_LENGTH="${VIDEO_LENGTH:-0}"

# ── Parse sequence key → (object_id, trajectory_task, data_id) ───────────────
# HO-Cap key format: subject_{N}-{DATE_TIME}-{G_OBJECT_ID}
# Example: subject_1-20231025_170231-G10_1  →  object_id=G10_1
parse_seq() {
    local key="$1"
    IFS='-' read -ra parts <<< "${key}"
    SUBJECT="${parts[0]}"
    DATETIME="${parts[1]}"
    OBJECT_ID="${parts[2]}"
    TRAJECTORY_TASK="${key}"
    DATA_ID="0"
}

# ── Rollout loop ──────────────────────────────────────────────────────────────

TOTAL="${#SEQUENCES[@]}"
IDX=0

for key in "${SEQUENCES[@]}"; do
    IDX=$(( IDX + 1 ))
    parse_seq "${key}"

    CKPT_FILE="${CHECKPOINT_BASE}/${TRAJECTORY_TASK}/${DATA_ID}/agent.pt"
    EVAL_TAG="evaluation_ep_le_${TIMESTEPS}"
    OUT_DIR="${CHECKPOINT_BASE}/${TRAJECTORY_TASK}/${DATA_ID}/${EVAL_TAG}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[eval-shadow-rsi ${IDX}/${TOTAL}] ${key}"
    echo "  object=${OBJECT_ID}  traj=${TRAJECTORY_TASK}  id=${DATA_ID}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ ! -f "${CKPT_FILE}" ]]; then
        echo "[eval-shadow-rsi] ERROR: Checkpoint not found — run train_sequences_shadow_rsi.sh first."
        echo "       ${CKPT_FILE}"
        continue
    fi

    if [[ -f "${OUT_DIR}/metrics.csv" && "${FORCE}" -eq 0 ]]; then
        echo "[eval-shadow-rsi] metrics.csv already exists — skipping rollout.  (set FORCE=1 to override)"
        echo "       ${OUT_DIR}/metrics.csv"
        continue
    fi

    VIDEO_ARGS=()
    if [[ "${VIDEO}" -eq 1 ]]; then
        VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}")
    fi

    cd "${PROJECT_DIR}"
    python scripts/skrl/rollout.py \
        --task        "${TASK}" \
        --checkpoint  "${CKPT_FILE}" \
        --output_dir  "${OUT_DIR}" \
        --n_rollouts  "${N_ROLLOUTS}" \
        --headless \
        "${VIDEO_ARGS[@]}" \
        --dataset             "${DATASET}" \
        --object_id           "${OBJECT_ID}" \
        --trajectory_task     "${TRAJECTORY_TASK}" \
        --trajectory_data_id  "${DATA_ID}"
done

# ── Aggregate metrics ─────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[eval-shadow-rsi] Aggregating metrics for dataset '${DATASET}' ..."
bash "${SCRIPT_DIR}/../evaluate.bash" "${DATA_BASE}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[eval-shadow-rsi] Done.  Results written to:"
echo "       ${DATA_BASE}/ffw_shadow_rsi/method{1,2,3}.csv"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
