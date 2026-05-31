#!/usr/bin/env bash
# =============================================================================
# train_sequences.sh — Full pipeline: dataset → IK → pretrain → train.
#
# For each sequence this script:
#   1. Processes the raw OakInk sequence into SPIDER format (skipped if done).
#   2. Computes frame-0 arm IK (skipped if done).
#   3. Pretrains for PRETRAIN_TIMESTEPS steps; saves pretrain.pt.
#   4. Trains for TIMESTEPS steps starting from pretrain.pt; saves agent.pt.
#
# Checkpoints are written alongside evaluation results:
#   data/processed/<dataset>/ffw_sh5/<object_id>/<trajectory_task>/<data_id>/
#       pretrain.pt   ← after pretrain
#       agent.pt      ← after train
#
# Set FORCE=1 to re-run all steps even when outputs already exist.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────

TASK_PRETRAIN="Robotis-Sh5-Grasp-Pretrain-Direct-v0"
TASK="Robotis-Sh5-Grasp-Direct-v0"
DATASET="oakink"
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-4096}"
NUM_ENVS="${NUM_ENVS:-2048}"
PRETRAIN_TIMESTEPS=4000
TIMESTEPS=16000

# Set VIDEO=1 to record training mp4 every VIDEO_INTERVAL steps into
# logs/skrl/<task>/<run>/videos/train/. Reduce {PRETRAIN_,}NUM_ENVS to fit GPU
# memory when enabling video (camera rendering adds significant VRAM overhead).
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-1000}"

# Sequence keys — format matches mano/right folder names: {OBJECT_ID}-{SEQ}-{GESTURE}
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
CHECKPOINT_BASE="${DATA_BASE}/ffw_sh5/right"
LOG_PRETRAIN="${PROJECT_DIR}/logs/skrl/robotis_sh5_grasp_pretrain"
LOG_BASE="${PROJECT_DIR}/logs/skrl/robotis_sh5_grasp"

FORCE="${FORCE:-0}"

# ── Parse sequence key → (object_id, trajectory_task, data_id) ───────────────
# Key format matches mano/right folder names: {OBJECT_ID}-{SEQ}-{GESTURE}
# Example: A01005-0001-0000  →  object_id=A01005, trajectory_task=A01005-0001-0000
parse_seq() {
    local key="$1"
    IFS='-' read -ra parts <<< "${key}"
    OBJECT_ID="${parts[0]}"
    SEQ="${parts[1]}"
    GESTURE="${parts[2]}"
    TRAJECTORY_TASK="${key}"
    DATA_ID="0"
}

# ── Main loop ─────────────────────────────────────────────────────────────────

TOTAL="${#SEQUENCES[@]}"
IDX=0

for key in "${SEQUENCES[@]}"; do
    IDX=$(( IDX + 1 ))
    parse_seq "${key}"

    CKPT_DIR="${CHECKPOINT_BASE}/${TRAJECTORY_TASK}/${DATA_ID}"
    PRETRAIN_FILE="${CKPT_DIR}/pretrain.pt"
    CKPT_FILE="${CKPT_DIR}/agent.pt"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[train ${IDX}/${TOTAL}] ${key}"
    echo "  object=${OBJECT_ID}  traj=${TRAJECTORY_TASK}  id=${DATA_ID}"
    echo "  pretrain=${PRETRAIN_TIMESTEPS} steps  train=${TIMESTEPS} steps"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ -f "${CKPT_FILE}" && "${FORCE}" -eq 0 ]]; then
        echo "[train] Checkpoint already exists — skipping.  (set FORCE=1 to override)"
        echo "        ${CKPT_FILE}"
        continue
    fi

    mkdir -p "${CKPT_DIR}"
    cd "${PROJECT_DIR}"

    # ── Step 1: Dataset processing ────────────────────────────────────────────
    echo ""
    echo "[train] Step 1/4 — Dataset check: ${TRAJECTORY_TASK}"
    MANO_DIR="${DATA_BASE}/mano/right/${TRAJECTORY_TASK}"
    if [[ ! -d "${MANO_DIR}" ]]; then
        echo "[train] ERROR: Mano data not found — run oakink.py or hocap.py first."
        echo "        Expected: ${MANO_DIR}"
        continue
    fi
    echo "[train] Mano data exists — ok."

    # ── Step 2: Arm reference pipeline (elbow + per-frame IK + video) ─────────
    echo ""
    echo "[train] Step 2/4 — Arm pipeline: ${TRAJECTORY_TASK} / ${DATA_ID}"
    python scripts/process_dataset/process_arm_pipeline.py \
        --dataset "${DATASET}" \
        --task    "${TRAJECTORY_TASK}" \
        --data_id "${DATA_ID}"

    # Optional video args (apply to both pretrain and train if VIDEO=1).
    VIDEO_ARGS=()
    if [[ "${VIDEO}" -eq 1 ]]; then
        VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}" --video_interval "${VIDEO_INTERVAL}")
    fi

    # ── Step 3: Pretrain ──────────────────────────────────────────────────────
    if [[ -f "${PRETRAIN_FILE}" && "${FORCE}" -eq 0 ]]; then
        echo ""
        echo "[train] Step 3/4 — Pretrain checkpoint exists — skipping."
        echo "        ${PRETRAIN_FILE}"
    else
        echo ""
        echo "[train] Step 3/4 — Pretraining (${PRETRAIN_TIMESTEPS} steps) ..."
        touch "${CKPT_DIR}/.sentinel_pretrain"

        python scripts/skrl/train.py \
            --task        "${TASK_PRETRAIN}" \
            --num_envs    "${PRETRAIN_NUM_ENVS}" \
            --timesteps   "${PRETRAIN_TIMESTEPS}" \
            --headless \
            "${VIDEO_ARGS[@]}" \
            --dataset             "${DATASET}" \
            --object_id           "${OBJECT_ID}" \
            --trajectory_task     "${TRAJECTORY_TASK}" \
            --trajectory_data_id  "${DATA_ID}"

        # Look for agent_{PRETRAIN_TIMESTEPS}.pt produced after sentinel
        PRETRAIN_CKPT=$(find "${LOG_PRETRAIN}" -name "agent_${PRETRAIN_TIMESTEPS}.pt" \
                        -newer "${CKPT_DIR}/.sentinel_pretrain" \
                        2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)

        if [[ -z "${PRETRAIN_CKPT}" ]]; then
            # Fallback: newest matching checkpoint in the whole pretrain log tree
            PRETRAIN_CKPT=$(find "${LOG_PRETRAIN}" -name "agent_${PRETRAIN_TIMESTEPS}.pt" \
                            2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)
        fi

        if [[ -z "${PRETRAIN_CKPT}" ]]; then
            echo "[train] ERROR: Could not find pretrain checkpoint (agent_${PRETRAIN_TIMESTEPS}.pt)."
            continue
        fi

        cp "${PRETRAIN_CKPT}" "${PRETRAIN_FILE}"
        echo "[train] Pretrain checkpoint saved → ${PRETRAIN_FILE}"
    fi

    # ── Step 4: Train (from pretrain checkpoint) ──────────────────────────────
    echo ""
    echo "[train] Step 4/4 — Training (${TIMESTEPS} steps from pretrain checkpoint) ..."
    touch "${CKPT_DIR}/.sentinel"

    python scripts/skrl/train.py \
        --task        "${TASK}" \
        --num_envs    "${NUM_ENVS}" \
        --timesteps   "${TIMESTEPS}" \
        --headless \
        "${VIDEO_ARGS[@]}" \
        --checkpoint  "${PRETRAIN_FILE}" \
        --dataset             "${DATASET}" \
        --object_id           "${OBJECT_ID}" \
        --trajectory_task     "${TRAJECTORY_TASK}" \
        --trajectory_data_id  "${DATA_ID}"

    LATEST_CKPT=$(find "${LOG_BASE}" -name "agent_*.pt" -newer "${CKPT_DIR}/.sentinel" \
                  2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)

    if [[ -z "${LATEST_CKPT}" ]]; then
        LATEST_CKPT=$(find "${LOG_BASE}" -name "agent_*.pt" 2>/dev/null \
                      | xargs ls -t 2>/dev/null | head -1 || true)
    fi

    if [[ -z "${LATEST_CKPT}" ]]; then
        echo "[train] ERROR: Could not find checkpoint after training."
        continue
    fi

    cp "${LATEST_CKPT}" "${CKPT_FILE}"
    echo "[train] Checkpoint saved → ${CKPT_FILE}"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[train] All ${TOTAL} sequences processed."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
