#!/usr/bin/env bash
# =============================================================================
# train_sequences_shadow_rsi.sh — HO-Cap pipeline for the Shadow Hand RSI
# warm-start variant: dataset check → arm IK → pretrain → train.
#
# Mirrors train_sequences_shadow.sh, but targets the RSI tasks
# (Robotis-Shadow-Grasp-Rsi-*-Direct-v0). The RSI train phase warm-starts its
# reference-state-initialization cache from the pretrain phase's state cache:
#   - pretrain saves `pretrain_state_cache.npz` next to pretrain.pt
#   - train loads it (checkpoint sibling) and, for the first
#     `pretrain_cache_warmup_steps` control steps, rolls out from the pretrain
#     cache while filling its own train cache; then switches over.
#
# Checkpoints + logs use a SEPARATE tree (ffw_shadow_rsi /
# robotis_shadow_grasp_rsi{,_pretrain}) so plain-Shadow and FFW-SH5 results stay
# untouched and all three can be compared side-by-side.
#
# Per sequence:
#   1. Verify processed mano data exists (skipped if done).
#   2. Compute arm reference (process_arm_pipeline.py --robot shadow; shared assets).
#   3. Pretrain for PRETRAIN_TIMESTEPS steps; saves pretrain.pt + pretrain_state_cache.npz.
#   4. Train for TIMESTEPS steps from pretrain.pt (+ cache warm-start); saves agent.pt.
#
# Set FORCE=1 to re-run all steps even when outputs already exist.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────

TASK_PRETRAIN="Robotis-Shadow-Grasp-Rsi-Pretrain-Direct-v0"
TASK="Robotis-Shadow-Grasp-Rsi-Direct-v0"
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
# Default: same subset as train_sequences_shadow.sh.
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
# RSI checkpoints live in their own tree so they don't collide with plain-Shadow
# or FFW-SH5 results. evaluate.bash iterates over all ffw_* subdirectories under
# data/processed/<dataset>/ so every variant is picked up for comparison.
CHECKPOINT_BASE="${DATA_BASE}/ffw_shadow_rsi/right"
LOG_PRETRAIN="${PROJECT_DIR}/logs/skrl/robotis_shadow_grasp_rsi_pretrain"
LOG_BASE="${PROJECT_DIR}/logs/skrl/robotis_shadow_grasp_rsi"

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
    PRETRAIN_CACHE_FILE="${CKPT_DIR}/pretrain_state_cache.npz"
    CKPT_FILE="${CKPT_DIR}/agent.pt"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[shadow-rsi ${IDX}/${TOTAL}] ${key}"
    echo "  object=${OBJECT_ID}  traj=${TRAJECTORY_TASK}  id=${DATA_ID}"
    echo "  pretrain=${PRETRAIN_TIMESTEPS} steps  train=${TIMESTEPS} steps"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ -f "${CKPT_FILE}" && "${FORCE}" -eq 0 ]]; then
        echo "[shadow-rsi] Checkpoint already exists — skipping.  (set FORCE=1 to override)"
        echo "         ${CKPT_FILE}"
        continue
    fi

    mkdir -p "${CKPT_DIR}"
    cd "${PROJECT_DIR}"

    # ── Step 1: Dataset processing ────────────────────────────────────────────
    echo ""
    echo "[shadow-rsi] Step 1/4 — Dataset check: ${TRAJECTORY_TASK}"
    MANO_DIR="${DATA_BASE}/mano/right/${TRAJECTORY_TASK}"
    if [[ ! -d "${MANO_DIR}" ]]; then
        echo "[shadow-rsi] ERROR: Mano data not found — run hocap.py first."
        echo "         Expected: ${MANO_DIR}"
        continue
    fi
    echo "[shadow-rsi] Mano data exists — ok."

    # ── Step 2: Arm reference pipeline (Shadow Hand IK target) ────────────────
    echo ""
    echo "[shadow-rsi] Step 2/4 — Arm pipeline (shadow): ${TRAJECTORY_TASK} / ${DATA_ID}"
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
        echo "[shadow-rsi] Step 3/4 — Pretrain checkpoint exists — skipping."
        echo "         ${PRETRAIN_FILE}"
    else
        echo ""
        echo "[shadow-rsi] Step 3/4 — Pretraining (${PRETRAIN_TIMESTEPS} steps) ..."
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
            echo "[shadow-rsi] ERROR: Could not find pretrain checkpoint (agent_${PRETRAIN_TIMESTEPS}.pt)."
            continue
        fi

        cp "${PRETRAIN_CKPT}" "${PRETRAIN_FILE}"
        echo "[shadow-rsi] Pretrain checkpoint saved → ${PRETRAIN_FILE}"

        # ── RSI warm-start cache ────────────────────────────────────────────
        # train.py writes pretrain_state_cache.npz directly into CKPT_DIR (and the
        # run log dir). If the data-tree copy is missing for any reason, recover it
        # from the pretrain log tree so the train phase can load it as a sibling
        # of pretrain.pt. (No-op when it's already there.)
        if [[ ! -f "${PRETRAIN_CACHE_FILE}" ]]; then
            PRETRAIN_CACHE_SRC=$(find "${LOG_PRETRAIN}" -name "pretrain_state_cache.npz" \
                                 -newer "${CKPT_DIR}/.sentinel_pretrain" \
                                 2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)
            if [[ -z "${PRETRAIN_CACHE_SRC}" ]]; then
                PRETRAIN_CACHE_SRC=$(find "${LOG_PRETRAIN}" -name "pretrain_state_cache.npz" \
                                     2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)
            fi
            if [[ -n "${PRETRAIN_CACHE_SRC}" ]]; then
                cp "${PRETRAIN_CACHE_SRC}" "${PRETRAIN_CACHE_FILE}"
                echo "[shadow-rsi] Pretrain state cache saved → ${PRETRAIN_CACHE_FILE}"
            else
                echo "[shadow-rsi] WARN: pretrain_state_cache.npz not found — train will start vanilla."
            fi
        else
            echo "[shadow-rsi] Pretrain state cache present → ${PRETRAIN_CACHE_FILE}"
        fi
    fi

    # ── Step 4: Train (from pretrain checkpoint + RSI cache warm-start) ───────
    echo ""
    echo "[shadow-rsi] Step 4/4 — Training (${TIMESTEPS} steps from pretrain checkpoint) ..."
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
        echo "[shadow-rsi] ERROR: Could not find checkpoint after training."
        continue
    fi

    cp "${LATEST_CKPT}" "${CKPT_FILE}"
    echo "[shadow-rsi] Checkpoint saved → ${CKPT_FILE}"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[shadow-rsi] All ${TOTAL} sequences processed."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
