#!/usr/bin/env bash
# =============================================================================
# train_sequences_shadow.sh — Full pipeline for HO-Cap with the Shadow Hand
# variant: dataset check → arm IK → pretrain → train.
#
# Mirrors train_sequences.sh exactly, but targets the Shadow-Hand-on-Robotis
# environment (Robotis-Shadow-Grasp-*-Direct-v0). Checkpoints + logs are written
# to a SEPARATE tree (ffw_shadow / robotis_shadow_grasp{,_pretrain}) so the
# original FFW-SH5 results remain untouched and the two variants can be
# compared side-by-side.
#
# Per sequence:
#   1. Verify processed mano data exists (skipped if done).
#   2. Compute arm reference (process_arm_pipeline.py — shared with sh5).
#   3. Pretrain for PRETRAIN_TIMESTEPS steps; saves pretrain.pt.
#   4. Train for TIMESTEPS steps starting from pretrain.pt; saves agent.pt.
#
# Set FORCE=1 to re-run all steps even when outputs already exist.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────

TASK_PRETRAIN="Robotis-Shadow-Grasp-Pretrain-Direct-v0"
TASK="Robotis-Shadow-Grasp-Direct-v0"
DATASET="hocap"
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-4096}"
NUM_ENVS="${NUM_ENVS:-2048}"
PRETRAIN_TIMESTEPS="${PRETRAIN_TIMESTEPS:-8000}"
TIMESTEPS="${TIMESTEPS:-40000}"

# Set VIDEO=1 to record training mp4 every VIDEO_INTERVAL steps into
# logs/skrl/<task>/<run>/videos/train/. Reduce {PRETRAIN_,}NUM_ENVS to fit GPU
# memory when enabling video (camera rendering adds significant VRAM overhead).
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-1000}"

# Sequence keys — format matches mano/right folder names: subject_{N}-{DATE_TIME}-{G_OBJECT_ID}
# Default: same subset as train_sequences.sh (subject_1 G10_* + subject_6 G06_*/G09_*).
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
# Shadow checkpoints live in a separate tree so they don't collide with the
# FFW-SH5 native-hand results. evaluate.bash automatically iterates over all
# ffw_* subdirectories under data/processed/<dataset>/ so both variants are
# picked up for side-by-side comparison.
CHECKPOINT_BASE="${DATA_BASE}/ffw_shadow/right"
LOG_PRETRAIN="${PROJECT_DIR}/logs/skrl/robotis_shadow_grasp_pretrain"
LOG_BASE="${PROJECT_DIR}/logs/skrl/robotis_shadow_grasp"

FORCE="${FORCE:-0}"

# ── Parse sequence key → (object_id, trajectory_task, data_id) ───────────────
# HO-Cap key format: subject_{N}-{DATE_TIME}-{G_OBJECT_ID}
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
    echo "[shadow ${IDX}/${TOTAL}] ${key}"
    echo "  object=${OBJECT_ID}  traj=${TRAJECTORY_TASK}  id=${DATA_ID}"
    echo "  pretrain=${PRETRAIN_TIMESTEPS} steps  train=${TIMESTEPS} steps"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ -f "${CKPT_FILE}" && "${FORCE}" -eq 0 ]]; then
        echo "[shadow] Checkpoint already exists — skipping.  (set FORCE=1 to override)"
        echo "         ${CKPT_FILE}"
        continue
    fi

    mkdir -p "${CKPT_DIR}"
    cd "${PROJECT_DIR}"

    # ── Step 1: Dataset processing ────────────────────────────────────────────
    echo ""
    echo "[shadow] Step 1/4 — Dataset check: ${TRAJECTORY_TASK}"
    MANO_DIR="${DATA_BASE}/mano/right/${TRAJECTORY_TASK}"
    if [[ ! -d "${MANO_DIR}" ]]; then
        echo "[shadow] ERROR: Mano data not found — run hocap.py first."
        echo "         Expected: ${MANO_DIR}"
        continue
    fi
    echo "[shadow] Mano data exists — ok."

    # ── Step 2: Arm reference pipeline (Shadow Hand IK target) ────────────────
    echo ""
    echo "[shadow] Step 2/4 — Arm pipeline (shadow): ${TRAJECTORY_TASK} / ${DATA_ID}"
    python scripts/process_dataset/process_arm_pipeline.py \
        --dataset "${DATASET}" \
        --task    "${TRAJECTORY_TASK}" \
        --data_id "${DATA_ID}" \
        --robot   shadow

    # Optional video args (apply to both pretrain and train if VIDEO=1).
    VIDEO_ARGS=()
    if [[ "${VIDEO}" -eq 1 ]]; then
        VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}" --video_interval "${VIDEO_INTERVAL}")
    fi

    # ── Step 3: Pretrain ──────────────────────────────────────────────────────
    if [[ -f "${PRETRAIN_FILE}" && "${FORCE}" -eq 0 ]]; then
        echo ""
        echo "[shadow] Step 3/4 — Pretrain checkpoint exists — skipping."
        echo "         ${PRETRAIN_FILE}"
    else
        echo ""
        echo "[shadow] Step 3/4 — Pretraining (${PRETRAIN_TIMESTEPS} steps) ..."
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
            echo "[shadow] ERROR: Could not find pretrain checkpoint (agent_${PRETRAIN_TIMESTEPS}.pt)."
            continue
        fi

        cp "${PRETRAIN_CKPT}" "${PRETRAIN_FILE}"
        echo "[shadow] Pretrain checkpoint saved → ${PRETRAIN_FILE}"
    fi

    # ── Step 4: Train (from pretrain checkpoint) ──────────────────────────────
    echo ""
    echo "[shadow] Step 4/4 — Training (${TIMESTEPS} steps from pretrain checkpoint) ..."
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
        echo "[shadow] ERROR: Could not find checkpoint after training."
        continue
    fi

    cp "${LATEST_CKPT}" "${CKPT_FILE}"
    echo "[shadow] Checkpoint saved → ${CKPT_FILE}"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[shadow] All ${TOTAL} sequences processed."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
