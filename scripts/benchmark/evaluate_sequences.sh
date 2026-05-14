#!/usr/bin/env bash
# =============================================================================
# evaluate_sequences.sh — Run rollouts for each sequence and aggregate metrics.
#
# For each sequence, this script:
#   1. Loads the checkpoint from the checkpoints directory.
#   2. Runs rollout.py to produce metrics.csv.
#   3. After all rollouts, calls evaluate.bash to produce per-method CSV files.
#
# Output structure (evaluate.bash compatible):
#   data/processed/<dataset>/ffw_sh5/<object_id>/<trajectory_task>/<data_id>/
#       pretrain.pt / agent.pt          ← checkpoints (written by train_sequences.sh)
#       evaluation_ep_le_<TIMESTEPS>/metrics.csv
#
# Aggregate CSVs are written to:
#   data/processed/<dataset>/ffw_sh5_method{1,2,3}.csv
#
# Set FORCE=1 to re-run rollouts even when metrics.csv already exists.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────

TASK="Robotis-Sh5-Grasp-Direct-v0"
DATASET="oakink"
N_ROLLOUTS=32
# Must match the timesteps used during training (used for directory naming only).
TIMESTEPS=10000

# Sequence keys — format matches mano/right folder names: {OBJECT_ID}-{SEQ}-{GESTURE}
SEQUENCES=(
    "A01005-0001-0000"
    "A01005-0001-0001"
    "A01006-0001-0000"
    "A01026-0001-0000"
    "A01026-0001-0001"
    "A02012-0001-0004"
    "A02014-0001-0005"
    "A02021-0001-0005"
    "A02028-0001-0005"
    "A02031-0001-0005"
    "A15015-0001-0007"
    "A15015-0001-0008"
    "A15027-0001-0007"
    "A16013-0001-0006"
    "A16026-0001-0005"
    "C11001-0001-0007"
    "C13001-0001-0005"
    "C20001-0001-0009"
    "C22001-0001-0010"
    "C25001-0001-0009"
    "C35001-0001-0000"
    "C50001-0001-0000"
    "C91001-0001-0002"
    "O01000-0001-0001"
    "O02001-0001-0005"
    "O03001-0001-0007"
    "O03003-0001-0007"
    "O03003-0001-0008"
    "O21001-0001-0010"
    "O24001-0001-0010"
    "O36002-0001-0002"
    "O50002-0001-0001"
    "O93001-0001-0000"
    "S10002-0001-0000"
    "S10002-0001-0002"
    "S10003-0001-0001"
    "S10007-0001-0001"
    "S10008-0001-0003"
    "S10011-0001-0003"
    "S10012-0001-0003"
    "S10014-0001-0001"
    "S10015-0001-0001"
    "S10016-0001-0002"
    "S10019-0001-0001"
    "S10019-0001-0002"
    "S10019-0001-0003"
    "S10021-0001-0000"
    "S10023-0001-0001"
    "S10024-0001-0001"
    "S10024-0001-0003"
    "S15004-0001-0008"
    "S16001-0001-0005"
    "S16005-0001-0002"
    "S20005-0001-0010"
    "S20021-0001-0010"
    "S20022-0001-0010"
    "Y03006-0001-0007"
    "Y03021-0001-0000"
    "Y27035-0001-0000"
    "Y35037-0001-0002"
)

# ── Path setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_BASE="${PROJECT_DIR}/source/robotis_sh5/data/processed/${DATASET}"
CHECKPOINT_BASE="${DATA_BASE}/ffw_sh5/right"

FORCE="${FORCE:-0}"

# ── Parse sequence key → (object_id, trajectory_task, data_id) ───────────────
# Key format matches mano/right folder names: {OBJECT_ID}-{SEQ}-{GESTURE}
parse_seq() {
    local key="$1"
    IFS='-' read -ra parts <<< "${key}"
    OBJECT_ID="${parts[0]}"
    SEQ="${parts[1]}"
    GESTURE="${parts[2]}"
    TRAJECTORY_TASK="${key}"
    DATA_ID="0"
}

# ── Rollout loop ──────────────────────────────────────────────────────────────

TOTAL="${#SEQUENCES[@]}"
IDX=0
DATASETS_SEEN="${DATASET}"   # single dataset for now; extend if needed

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
        echo "[eval] ERROR: Checkpoint not found — run train_sequences.sh first."
        echo "       ${CKPT_FILE}"
        continue
    fi

    if [[ -f "${OUT_DIR}/metrics.csv" && "${FORCE}" -eq 0 ]]; then
        echo "[eval] metrics.csv already exists — skipping rollout.  (set FORCE=1 to override)"
        echo "       ${OUT_DIR}/metrics.csv"
        continue
    fi

    cd "${PROJECT_DIR}"
    python scripts/skrl/rollout.py \
        --task        "${TASK}" \
        --checkpoint  "${CKPT_FILE}" \
        --output_dir  "${OUT_DIR}" \
        --n_rollouts  "${N_ROLLOUTS}" \
        --headless \
        --dataset             "${DATASET}" \
        --object_id           "${OBJECT_ID}" \
        --trajectory_task     "${TRAJECTORY_TASK}" \
        --trajectory_data_id  "${DATA_ID}"
done

# ── Aggregate metrics ─────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[eval] Aggregating metrics for dataset '${DATASET}' ..."
bash "${SCRIPT_DIR}/evaluate.bash" "${DATA_BASE}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[eval] Done.  Results written to:"
echo "       ${DATA_BASE}/ffw_sh5_method{1,2,3}.csv"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
