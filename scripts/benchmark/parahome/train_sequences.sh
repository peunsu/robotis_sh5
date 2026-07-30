#!/usr/bin/env bash
# =============================================================================
# train_sequences.sh — ParaHome pipeline for the G1 + bimanual Shadow-hand
# whole-body loco-manipulation task: (retarget →) pretrain → train, per clip.
#
# Mirrors scripts/benchmark/hocap/train_sequences_shadow_rsi.sh but targets the
# g1 loco-manip tasks (Robotis-G1-Shadow-Locomanip-{Pretrain-,}Direct-v0) and the
# ParaHome data layout (clip_class / clip_name instead of dataset / trajectory).
#
# The g1 train phase warm-starts its RSI cache from the pretrain phase's state
# cache exactly like the shadow-rsi variant:
#   - pretrain (object-FREE) saves agent_*.pt + pretrain_state_cache.npz (209-D)
#   - train (object spawned when its converted USD exists) loads the cache as a
#     sibling of --checkpoint and warm-starts from it.
#
# Per clip:
#   1. Verify the processed SMPL-X clip exists (smplx/<class>/<clip>/0/trajectory.npz).
#   2. (optional) Frame-0 retarget → g1_joint_pos + g1_root_pose seed (SKIP_RETARGET=1 to skip).
#   3. Pretrain for PRETRAIN_TIMESTEPS steps; copy pretrain.pt + cache into the data tree.
#   4. Train for TIMESTEPS steps from pretrain.pt (+ RSI cache); copy agent.pt into the data tree.
#
# Checkpoints/metrics live in a dedicated tree so evaluate.bash picks them up:
#   data/processed/parahome/g1_shadow_locomanip/<clip_class>/<clip_name>/0/
#
# Which clips run: edit the CLIPS=(...) array below (comment lines out to filter, like the hocap
# scripts), or override the whole list with CLIPS_OVERRIDE="clipA clipB".
# Env vars: FORCE=1 (re-run all), SKIP_RETARGET=1, CLIP_CLASS=..., CLIPS_OVERRIDE="a b c",
#   PRETRAIN_NUM_ENVS / NUM_ENVS / PRETRAIN_TIMESTEPS / TIMESTEPS, VIDEO=1, PY=<python>.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────
TASK_PRETRAIN="Robotis-G1-Shadow-Locomanip-Pretrain-Direct-v0"
TASK="Robotis-G1-Shadow-Locomanip-Direct-v0"
CLIP_CLASS="${CLIP_CLASS:-single_rigid}"
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-4096}"
NUM_ENVS="${NUM_ENVS:-2048}"
PRETRAIN_TIMESTEPS="${PRETRAIN_TIMESTEPS:-20000}"
TIMESTEPS="${TIMESTEPS:-80000}"
PY="${PY:-python}"                       # set to the env_isaaclab python if not on PATH
SKIP_RETARGET="${SKIP_RETARGET:-0}"      # 1 → skip the frame-0 retarget (rely on default standing pose)

# Set VIDEO=1 to record training mp4 every VIDEO_INTERVAL steps.
VIDEO="${VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-1000}"

# Clip names under smplx/${CLIP_CLASS}/ to run. Comment out lines to filter the run (exactly
# like the hocap SEQUENCES arrays). Defaults: single_rigid clips whose objects have converted
# rigid USDs. Alternatively override the WHOLE list from the env: CLIPS_OVERRIDE="clipA clipB".
CLIPS=(
    "s100_seg00_pan"
    # "s100_seg02_kettle"
    # "s100_seg03_cup"
    # "s101_seg30_bowl"
    # "s101_seg29_pot"
    # "s101_seg18_potlid"
    # "s103_seg09_knife"     # ← uncomment / add more clips here
    # "s101_seg23_salt"
)
# Optional: replace the whole list from the environment (space-separated).
[[ -n "${CLIPS_OVERRIDE:-}" ]] && read -ra CLIPS <<< "${CLIPS_OVERRIDE}"

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_BASE="${PROJECT_DIR}/source/robotis_sh5/data/processed/parahome"
CHECKPOINT_BASE="${DATA_BASE}/g1_shadow_locomanip/${CLIP_CLASS}"
LOG_PRETRAIN="${PROJECT_DIR}/logs/skrl/g1_shadow_locomanip_pretrain"
LOG_BASE="${PROJECT_DIR}/logs/skrl/g1_shadow_locomanip"
FORCE="${FORCE:-0}"

# newest agent_*.pt in a log tree, created after a sentinel file (fallback: newest overall)
find_ckpt() {  # $1=log_tree $2=sentinel
    local c
    c=$(find "$1" -name "agent_*.pt" -newer "$2" 2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)
    [[ -z "${c}" ]] && c=$(find "$1" -name "agent_*.pt" 2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)
    echo "${c}"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
TOTAL="${#CLIPS[@]}"
IDX=0
for clip in "${CLIPS[@]}"; do
    IDX=$(( IDX + 1 ))
    CKPT_DIR="${CHECKPOINT_BASE}/${clip}/0"
    PRETRAIN_FILE="${CKPT_DIR}/pretrain.pt"
    PRETRAIN_CACHE_FILE="${CKPT_DIR}/pretrain_state_cache.npz"
    CKPT_FILE="${CKPT_DIR}/agent.pt"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[g1 ${IDX}/${TOTAL}] class=${CLIP_CLASS}  clip=${clip}"
    echo "  pretrain=${PRETRAIN_TIMESTEPS} steps (${PRETRAIN_NUM_ENVS} envs)  train=${TIMESTEPS} steps (${NUM_ENVS} envs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ -f "${CKPT_FILE}" && "${FORCE}" -eq 0 ]]; then
        echo "[g1] Checkpoint exists — skipping.  (FORCE=1 to override)  ${CKPT_FILE}"
        continue
    fi
    mkdir -p "${CKPT_DIR}"
    cd "${PROJECT_DIR}"

    # ── Step 1: clip data check ───────────────────────────────────────────────
    CLIP_NPZ="${DATA_BASE}/smplx/${CLIP_CLASS}/${clip}/0/trajectory.npz"
    if [[ ! -f "${CLIP_NPZ}" ]]; then
        echo "[g1] ERROR: processed clip not found — run parahome.py first."
        echo "       Expected: ${CLIP_NPZ}"
        continue
    fi

    # ── Step 2: frame-0 retarget seed (pinocchio+pink; no Isaac Sim) ──────────
    RETARGET_NPZ="${DATA_BASE}/g1_shadow/${CLIP_CLASS}/${clip}/0/trajectory.npz"
    if [[ "${SKIP_RETARGET}" -eq 1 ]]; then
        echo "[g1] Step 2/4 — retarget SKIPPED (SKIP_RETARGET=1; reset uses the default standing pose)."
    elif [[ -f "${RETARGET_NPZ}" && "${FORCE}" -eq 0 ]]; then
        echo "[g1] Step 2/4 — retarget exists — skipping.  ${RETARGET_NPZ}"
    else
        echo "[g1] Step 2/4 — Retargeting (frame-0 seed): ${clip}"
        "${PY}" scripts/process_dataset/retarget/retarget_g1_shadow_pink.py \
            --clip_class "${CLIP_CLASS}" --clip "${clip}" || \
            echo "[g1] WARN: retarget failed — training will fall back to the default standing pose."
    fi

    VIDEO_ARGS=()
    if [[ "${VIDEO}" -eq 1 ]]; then
        VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}" --video_interval "${VIDEO_INTERVAL}")
    fi

    # ── Step 3: Pretrain (object-free, kinematic) ─────────────────────────────
    if [[ -f "${PRETRAIN_FILE}" && "${FORCE}" -eq 0 ]]; then
        echo "[g1] Step 3/4 — pretrain checkpoint exists — skipping.  ${PRETRAIN_FILE}"
    else
        echo "[g1] Step 3/4 — Pretraining (${PRETRAIN_TIMESTEPS} steps) ..."
        touch "${CKPT_DIR}/.sentinel_pretrain"
        "${PY}" scripts/skrl/train.py \
            --task "${TASK_PRETRAIN}" --num_envs "${PRETRAIN_NUM_ENVS}" \
            --timesteps "${PRETRAIN_TIMESTEPS}" --headless "${VIDEO_ARGS[@]}" \
            --clip_class "${CLIP_CLASS}" --clip_name "${clip}"

        PRETRAIN_CKPT=$(find_ckpt "${LOG_PRETRAIN}" "${CKPT_DIR}/.sentinel_pretrain")
        if [[ -z "${PRETRAIN_CKPT}" ]]; then
            echo "[g1] ERROR: pretrain checkpoint not found in ${LOG_PRETRAIN}."; continue
        fi
        cp "${PRETRAIN_CKPT}" "${PRETRAIN_FILE}"
        echo "[g1] Pretrain checkpoint → ${PRETRAIN_FILE}"

        # RSI warm-start cache (train.py writes it into the run log dir + its checkpoints subdir).
        CACHE_SRC=$(find "${LOG_PRETRAIN}" -name "pretrain_state_cache.npz" \
                    -newer "${CKPT_DIR}/.sentinel_pretrain" 2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)
        [[ -z "${CACHE_SRC}" ]] && CACHE_SRC=$(find "${LOG_PRETRAIN}" -name "pretrain_state_cache.npz" \
                    2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)
        if [[ -n "${CACHE_SRC}" ]]; then
            cp "${CACHE_SRC}" "${PRETRAIN_CACHE_FILE}"
            echo "[g1] Pretrain state cache → ${PRETRAIN_CACHE_FILE}"
        else
            echo "[g1] WARN: pretrain_state_cache.npz not found — train will start without warm-start."
        fi
    fi

    # ── Step 4: Train (from pretrain.pt + RSI cache; object spawned if USD exists) ─
    echo "[g1] Step 4/4 — Training (${TIMESTEPS} steps from pretrain) ..."
    touch "${CKPT_DIR}/.sentinel"
    "${PY}" scripts/skrl/train.py \
        --task "${TASK}" --num_envs "${NUM_ENVS}" \
        --timesteps "${TIMESTEPS}" --headless "${VIDEO_ARGS[@]}" \
        --checkpoint "${PRETRAIN_FILE}" \
        --clip_class "${CLIP_CLASS}" --clip_name "${clip}"

    LATEST_CKPT=$(find_ckpt "${LOG_BASE}" "${CKPT_DIR}/.sentinel")
    if [[ -z "${LATEST_CKPT}" ]]; then
        echo "[g1] ERROR: train checkpoint not found in ${LOG_BASE}."; continue
    fi
    cp "${LATEST_CKPT}" "${CKPT_FILE}"
    echo "[g1] Checkpoint → ${CKPT_FILE}"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[g1] All ${TOTAL} clips processed.  Tree: ${CHECKPOINT_BASE}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
