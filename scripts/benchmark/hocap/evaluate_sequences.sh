#!/usr/bin/env bash
# =============================================================================
# evaluate_sequences_hocap.sh — Run rollouts for each HO-Cap sequence and
# aggregate metrics.
#
# For each sequence, this script:
#   1. Loads the checkpoint from the checkpoints directory.
#   2. Runs rollout.py to produce metrics.csv.
#   3. After all rollouts, calls evaluate.bash to produce per-method CSV files.
#
# HO-Cap key format differs from OakInk:
#   OakInk : {OBJECT_ID}-{SEQ}-{GESTURE}              e.g. A01005-0001-0000
#   HO-Cap : subject_{N}-{DATE_TIME}-{G_OBJECT_ID}    e.g. subject_1-20231025_170231-G10_1
#
# Output structure (evaluate.bash compatible):
#   data/processed/hocap/ffw_sh5/right/<trajectory_task>/<data_id>/
#       pretrain.pt / agent.pt          ← checkpoints (written by train_sequences.sh)
#       evaluation_ep_le_<TIMESTEPS>/metrics.csv
#
# Aggregate CSVs are written to:
#   data/processed/hocap/ffw_sh5_method{1,2,3}.csv
#
# Set FORCE=1 to re-run rollouts even when metrics.csv already exists.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────

TASK="Robotis-Sh5-Grasp-Direct-v0"
DATASET="hocap"
N_ROLLOUTS=32
# Must match the timesteps used during training (used for directory naming only).
TIMESTEPS=40000

# Sequence keys — format matches mano/right folder names: subject_{N}-{DATE_TIME}-{G_OBJECT_ID}
# TEMP: subject_1 + subject_6 only — uncomment others to restore full set.
SEQUENCES=(
    "subject_1-20231025_170231-G10_1"
    "subject_1-20231025_170231-G10_2"
    "subject_1-20231025_170231-G10_3"
    "subject_1-20231025_170231-G10_4"
    # "subject_2-20231022_201556-G05_1"
    # "subject_2-20231022_201556-G05_2"
    # "subject_2-20231022_203100-G09_2"
    # "subject_2-20231022_203100-G09_4"
    # "subject_2-20231023_164242-G19_1"
    # "subject_2-20231023_164242-G19_2"
    # "subject_2-20231023_164242-G19_4"
    # "subject_2-20231023_164741-G22_2"
    # "subject_2-20231023_164741-G22_3"
    # "subject_2-20231023_164741-G22_4"
    # "subject_3-20231024_154810-G09_1"
    # "subject_3-20231024_154810-G09_2"
    # "subject_3-20231024_154810-G09_4"
    # "subject_4-20231026_162248-G11_1"
    # "subject_4-20231026_162248-G11_2"
    # "subject_4-20231026_164958-G21_1"
    # "subject_4-20231026_164958-G21_3"
    # "subject_4-20231026_164958-G21_4"
    "subject_6-20231025_111357-G06_1"
    "subject_6-20231025_111357-G06_2"
    "subject_6-20231025_111357-G06_3"
    "subject_6-20231025_111357-G06_4"
    "subject_6-20231025_112332-G09_1"
    "subject_6-20231025_112332-G09_2"
    "subject_6-20231025_112332-G09_3"
    "subject_6-20231025_112332-G09_4"
    # "subject_9-20231027_125019-G16_1"
    # "subject_9-20231027_125019-G16_2"
    # "subject_9-20231027_125019-G16_3"
    # "subject_9-20231027_125019-G16_4"
)

# ── Path setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_BASE="${PROJECT_DIR}/source/robotis_sh5/data/processed/${DATASET}"
CHECKPOINT_BASE="${DATA_BASE}/ffw_sh5/right"

FORCE="${FORCE:-0}"
# Set VIDEO=1 to record an mp4 of env 0's rollout into <OUT_DIR>/videos/
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"

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
    echo "[eval ${IDX}/${TOTAL}] ${key}"
    echo "  object=${OBJECT_ID}  traj=${TRAJECTORY_TASK}  id=${DATA_ID}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ ! -f "${CKPT_FILE}" ]]; then
        echo "[eval] ERROR: Checkpoint not found — run train_sequences_hocap.sh first."
        echo "       ${CKPT_FILE}"
        continue
    fi

    if [[ -f "${OUT_DIR}/metrics.csv" && "${FORCE}" -eq 0 ]]; then
        echo "[eval] metrics.csv already exists — skipping rollout.  (set FORCE=1 to override)"
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
echo "[eval] Aggregating metrics for dataset '${DATASET}' ..."
bash "${SCRIPT_DIR}/../evaluate.bash" "${DATA_BASE}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[eval] Done.  Results written to:"
echo "       ${DATA_BASE}/ffw_sh5_method{1,2,3}.csv"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
