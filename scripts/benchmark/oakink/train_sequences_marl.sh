#!/usr/bin/env bash
# =============================================================================
# train_sequences_marl.sh — MAPPO pipeline: dataset → IK → pretrain → train.
#
# Multi-agent variant of train_sequences.sh. For each sequence:
#   1. Checks the SPIDER-format mano data is present.
#   2. Computes frame-0 arm IK (idempotent).
#   3. Pretrains MAPPO for PRETRAIN_TIMESTEPS steps → pretrain.pt.
#   4. Trains MAPPO for TIMESTEPS steps from pretrain.pt → agent.pt.
#
# Checkpoint layout (separate `ffw_sh5_marl/` subtree from single-agent):
#   data/processed/<dataset>/ffw_sh5_marl/right/<trajectory_task>/<data_id>/
#       pretrain.pt   ← after MAPPO pretrain
#       agent.pt      ← after MAPPO train
#
# Set FORCE=1 to re-run all steps even when outputs already exist.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────

TASK_PRETRAIN="Robotis-Sh5-Grasp-Marl-Pretrain-Direct-v0"
TASK="Robotis-Sh5-Grasp-Marl-Direct-v0"
DATASET="oakink"
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-4096}"
NUM_ENVS="${NUM_ENVS:-2048}"
PRETRAIN_TIMESTEPS=4000
TIMESTEPS=16000

# Set VIDEO=1 to record training mp4 every VIDEO_INTERVAL steps into
# logs/skrl/<task>/<run>/videos/train/. Reduce NUM_ENVS to fit GPU memory.
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
CHECKPOINT_BASE="${DATA_BASE}/ffw_sh5_marl/right"
LOG_PRETRAIN="${PROJECT_DIR}/logs/skrl/robotis_sh5_grasp_marl_pretrain"
LOG_BASE="${PROJECT_DIR}/logs/skrl/robotis_sh5_grasp_marl"

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
    echo "[train-marl ${IDX}/${TOTAL}] ${key}"
    echo "  object=${OBJECT_ID}  traj=${TRAJECTORY_TASK}  id=${DATA_ID}"
    echo "  pretrain=${PRETRAIN_TIMESTEPS} steps  train=${TIMESTEPS} steps"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ -f "${CKPT_FILE}" && "${FORCE}" -eq 0 ]]; then
        echo "[train-marl] Checkpoint already exists — skipping.  (set FORCE=1 to override)"
        echo "            ${CKPT_FILE}"
        continue
    fi

    mkdir -p "${CKPT_DIR}"
    cd "${PROJECT_DIR}"

    # ── Step 1: Dataset processing ────────────────────────────────────────────
    echo ""
    echo "[train-marl] Step 1/4 — Dataset check: ${TRAJECTORY_TASK}"
    MANO_DIR="${DATA_BASE}/mano/right/${TRAJECTORY_TASK}"
    if [[ ! -d "${MANO_DIR}" ]]; then
        echo "[train-marl] ERROR: Mano data not found — run oakink.py first."
        echo "             Expected: ${MANO_DIR}"
        continue
    fi
    echo "[train-marl] Mano data exists — ok."

    # ── Step 2: Arm reference pipeline (elbow + per-frame IK + video) ─────────
    echo ""
    echo "[train-marl] Step 2/4 — Arm pipeline: ${TRAJECTORY_TASK} / ${DATA_ID}"
    python scripts/process_dataset/retarget/process_arm_pipeline.py \
        --dataset "${DATASET}" \
        --task    "${TRAJECTORY_TASK}" \
        --data_id "${DATA_ID}"

    VIDEO_ARGS=()
    if [[ "${VIDEO}" -eq 1 ]]; then
        VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}" --video_interval "${VIDEO_INTERVAL}")
    fi

    # ── Step 3: Pretrain (MAPPO, no object) ──────────────────────────────────
    if [[ -f "${PRETRAIN_FILE}" && "${FORCE}" -eq 0 ]]; then
        echo ""
        echo "[train-marl] Step 3/4 — Pretrain checkpoint exists — skipping."
        echo "             ${PRETRAIN_FILE}"
    else
        echo ""
        echo "[train-marl] Step 3/4 — Pretraining MAPPO (${PRETRAIN_TIMESTEPS} steps) ..."
        touch "${CKPT_DIR}/.sentinel_pretrain"

        python scripts/skrl/train_marl.py \
            --task        "${TASK_PRETRAIN}" \
            --num_envs    "${PRETRAIN_NUM_ENVS}" \
            --timesteps   "${PRETRAIN_TIMESTEPS}" \
            --headless \
            "${VIDEO_ARGS[@]}" \
            --dataset             "${DATASET}" \
            --object_id           "${OBJECT_ID}" \
            --trajectory_task     "${TRAJECTORY_TASK}" \
            --trajectory_data_id  "${DATA_ID}"

        PRETRAIN_CKPT=$(find "${LOG_PRETRAIN}" -name "agent_${PRETRAIN_TIMESTEPS}.pt" \
                        -newer "${CKPT_DIR}/.sentinel_pretrain" \
                        2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)

        if [[ -z "${PRETRAIN_CKPT}" ]]; then
            PRETRAIN_CKPT=$(find "${LOG_PRETRAIN}" -name "agent_${PRETRAIN_TIMESTEPS}.pt" \
                            2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)
        fi

        if [[ -z "${PRETRAIN_CKPT}" ]]; then
            echo "[train-marl] ERROR: Could not find pretrain checkpoint (agent_${PRETRAIN_TIMESTEPS}.pt)."
            continue
        fi

        cp "${PRETRAIN_CKPT}" "${PRETRAIN_FILE}"
        echo "[train-marl] Pretrain checkpoint saved → ${PRETRAIN_FILE}"
    fi

    # ── Step 4: Train (MAPPO from pretrain checkpoint) ────────────────────────
    echo ""
    echo "[train-marl] Step 4/4 — Training MAPPO (${TIMESTEPS} steps from pretrain) ..."
    touch "${CKPT_DIR}/.sentinel"

    python scripts/skrl/train_marl.py \
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
        echo "[train-marl] ERROR: Could not find checkpoint after training."
        continue
    fi

    cp "${LATEST_CKPT}" "${CKPT_FILE}"
    echo "[train-marl] Checkpoint saved → ${CKPT_FILE}"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[train-marl] All ${TOTAL} sequences processed."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
