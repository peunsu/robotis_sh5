#!/usr/bin/env bash
# =============================================================================
# train_sequences_hocap.sh — Full pipeline for HO-Cap: dataset → IK → pretrain
# → train.
#
# For each HO-Cap sequence this script:
#   1. Checks the processed mano data is present (skipped if done).
#   2. Computes frame-0 arm IK (skipped if done).
#   3. Pretrains for PRETRAIN_TIMESTEPS steps; saves pretrain.pt.
#   4. Trains for TIMESTEPS steps starting from pretrain.pt; saves agent.pt.
#
# HO-Cap key format differs from OakInk:
#   OakInk : {OBJECT_ID}-{SEQ}-{GESTURE}              e.g. A01005-0001-0000
#   HO-Cap : subject_{N}-{DATE_TIME}-{G_OBJECT_ID}    e.g. subject_1-20231025_170231-G10_1
#
# Checkpoints are written alongside evaluation results:
#   data/processed/hocap/ffw_sh5/right/<trajectory_task>/<data_id>/
#       pretrain.pt   ← after pretrain
#       agent.pt      ← after train
#
# Set FORCE=1 to re-run all steps even when outputs already exist.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────

TASK_PRETRAIN="Robotis-Sh5-Grasp-Pretrain-Direct-v0"
TASK="Robotis-Sh5-Grasp-Direct-v0"
DATASET="hocap"
PRETRAIN_NUM_ENVS=4096
NUM_ENVS=2048
PRETRAIN_TIMESTEPS=10000
TIMESTEPS=40000

# Sequence keys — format matches mano/right folder names: subject_{N}-{DATE_TIME}-{G_OBJECT_ID}
SEQUENCES=(
    "subject_1-20231025_170231-G10_1"
    "subject_1-20231025_170231-G10_2"
    "subject_1-20231025_170231-G10_3"
    "subject_1-20231025_170231-G10_4"
    "subject_2-20231022_201556-G05_1"
    "subject_2-20231022_201556-G05_2"
    "subject_2-20231022_203100-G09_2"
    "subject_2-20231022_203100-G09_4"
    "subject_2-20231023_164242-G19_1"
    "subject_2-20231023_164242-G19_2"
    "subject_2-20231023_164242-G19_4"
    "subject_2-20231023_164741-G22_2"
    "subject_2-20231023_164741-G22_3"
    "subject_2-20231023_164741-G22_4"
    "subject_3-20231024_154810-G09_1"
    "subject_3-20231024_154810-G09_2"
    "subject_3-20231024_154810-G09_4"
    "subject_4-20231026_162248-G11_1"
    "subject_4-20231026_162248-G11_2"
    "subject_4-20231026_164958-G21_1"
    "subject_4-20231026_164958-G21_3"
    "subject_4-20231026_164958-G21_4"
    "subject_6-20231025_111357-G06_1"
    "subject_6-20231025_111357-G06_2"
    "subject_6-20231025_111357-G06_3"
    "subject_6-20231025_111357-G06_4"
    "subject_6-20231025_112332-G09_1"
    "subject_6-20231025_112332-G09_2"
    "subject_6-20231025_112332-G09_3"
    "subject_6-20231025_112332-G09_4"
    "subject_9-20231027_125019-G16_1"
    "subject_9-20231027_125019-G16_2"
    "subject_9-20231027_125019-G16_3"
    "subject_9-20231027_125019-G16_4"
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
        echo "[train] ERROR: Mano data not found — run hocap.py first."
        echo "        Expected: ${MANO_DIR}"
        continue
    fi
    echo "[train] Mano data exists — ok."

    # ── Step 2: Frame-0 arm IK ────────────────────────────────────────────────
    echo ""
    echo "[train] Step 2/4 — Frame-0 IK: ${TRAJECTORY_TASK} / ${DATA_ID}"
    python scripts/process_dataset/compute_frame0_ik.py \
        --dataset "${DATASET}" \
        --task    "${TRAJECTORY_TASK}" \
        --data_id "${DATA_ID}"

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
