#!/usr/bin/env bash
# =============================================================================
# evaluate_sequences_marl.sh — Run MAPPO rollouts and aggregate metrics.
#
# Multi-agent variant of evaluate_sequences.sh. For each sequence:
#   1. Loads the MAPPO checkpoint from data/processed/<dataset>/ffw_sh5_marl/right/...
#   2. Runs rollout_marl.py to produce metrics.csv.
#   3. After all rollouts, calls evaluate.bash on <DATA_BASE> to produce per-method
#      CSVs. Both `ffw_sh5` (single-agent) and `ffw_sh5_marl` are picked up
#      automatically and aggregated into separate `<model>/method{1,2,3}.csv` files.
#
# Set FORCE=1 to re-run rollouts even when metrics.csv already exists.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────

TASK="Robotis-Sh5-Grasp-Marl-Direct-v0"
DATASET="oakink"
N_ROLLOUTS=32
TIMESTEPS=16000

# Sequence keys — must match train_sequences_marl.sh
SEQUENCES=(
    "A02014-0001-0005"
    "A02021-0001-0005"
    "A02028-0001-0005"
    "A15015-0001-0008"
    "C13001-0001-0005"
    "C20001-0001-0009"
    "O50002-0001-0001"
    "S10002-0001-0002"
    "Y27035-0001-0000"
    "A01026-0001-0000"
    "A02012-0001-0004"
    "S10015-0001-0001"
    "A02031-0001-0005"
    "C22001-0001-0010"
    "S10007-0001-0001"
    "C50001-0001-0000"
    "O03001-0001-0007"
    "A01026-0001-0001"
    "O03003-0001-0008"
    "A15027-0001-0007"
    "S10023-0001-0001"
    "C11001-0001-0007"
    "O01000-0001-0001"
    "O36002-0001-0002"
    "A01005-0001-0001"
    "A15015-0001-0007"
    "S10012-0001-0003"
    "A16026-0001-0005"
    "S10003-0001-0001"
    "S10019-0001-0001"
    "S10011-0001-0003"
    "O24001-0001-0010"
    "O02001-0001-0005"
    "S15004-0001-0008"
)

# ── Path setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_BASE="${PROJECT_DIR}/source/robotis_sh5/data/processed/${DATASET}"
CHECKPOINT_BASE="${DATA_BASE}/ffw_sh5_marl/right"

FORCE="${FORCE:-0}"
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"
# Set STOCHASTIC=1 to sample stochastic actions (default: deterministic mean).
STOCHASTIC="${STOCHASTIC:-0}"

# ── Parse sequence key → (object_id, trajectory_task, data_id) ───────────────
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

for key in "${SEQUENCES[@]}"; do
    IDX=$(( IDX + 1 ))
    parse_seq "${key}"

    CKPT_FILE="${CHECKPOINT_BASE}/${TRAJECTORY_TASK}/${DATA_ID}/agent.pt"
    EVAL_TAG="evaluation_ep_le_${TIMESTEPS}"
    OUT_DIR="${CHECKPOINT_BASE}/${TRAJECTORY_TASK}/${DATA_ID}/${EVAL_TAG}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[eval-marl ${IDX}/${TOTAL}] ${key}"
    echo "  object=${OBJECT_ID}  traj=${TRAJECTORY_TASK}  id=${DATA_ID}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ ! -f "${CKPT_FILE}" ]]; then
        echo "[eval-marl] ERROR: MARL checkpoint not found — run train_sequences_marl.sh first."
        echo "            ${CKPT_FILE}"
        continue
    fi

    if [[ -f "${OUT_DIR}/metrics.csv" && "${FORCE}" -eq 0 ]]; then
        echo "[eval-marl] metrics.csv already exists — skipping rollout.  (set FORCE=1 to override)"
        echo "            ${OUT_DIR}/metrics.csv"
        continue
    fi

    VIDEO_ARGS=()
    if [[ "${VIDEO}" -eq 1 ]]; then
        VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}")
    fi

    EXTRA_ARGS=()
    if [[ "${STOCHASTIC}" -eq 1 ]]; then
        EXTRA_ARGS+=(--stochastic)
    fi

    cd "${PROJECT_DIR}"
    python scripts/skrl/rollout_marl.py \
        --task        "${TASK}" \
        --checkpoint  "${CKPT_FILE}" \
        --output_dir  "${OUT_DIR}" \
        --n_rollouts  "${N_ROLLOUTS}" \
        --headless \
        "${VIDEO_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
        --dataset             "${DATASET}" \
        --object_id           "${OBJECT_ID}" \
        --trajectory_task     "${TRAJECTORY_TASK}" \
        --trajectory_data_id  "${DATA_ID}"
done

# ── Aggregate metrics ─────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[eval-marl] Aggregating metrics for dataset '${DATASET}' ..."
bash "${SCRIPT_DIR}/../evaluate.bash" "${DATA_BASE}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[eval-marl] Done.  Results written to:"
echo "            ${DATA_BASE}/ffw_sh5_marl/method{1,2,3}.csv"
echo "            (alongside ${DATA_BASE}/ffw_sh5/method{1,2,3}.csv if present)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
