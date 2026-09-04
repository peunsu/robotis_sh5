#!/usr/bin/env bash
# =============================================================================
# evaluate_sequences.sh — Roll out each trained ParaHome G1+Shadow loco-manip
# clip and aggregate metrics.
#
# Mirrors scripts/benchmark/hocap/evaluate_sequences_shadow_rsi.sh but targets the
# g1 task (Robotis-G1-Shadow-Locomanip-Direct-v0) and the ParaHome tree written by
# train_sequences.sh. rollout.py never arms the RSI warm-start, so evaluation runs
# the policy with vanilla frame-0 resets (same protocol as the other variants).
#
# For each clip:
#   1. Load the checkpoint from the g1_shadow_locomanip tree.
#   2. rollout.py → metrics.csv (obj pos/rot + whole-body kpt + fingertip errors).
#   3. After all clips, evaluate.bash aggregates per-method CSVs.
#
# Output tree (evaluate.bash compatible — clip_name lands at path parts[1]):
#   data/processed/parahome/g1_shadow_locomanip/<clip_class>/<clip_name>/0/
#       agent.pt / pretrain.pt
#       evaluation_ep_le_<TIMESTEPS>/metrics.csv
# Aggregates → data/processed/parahome/g1_shadow_locomanip/method{1,2,3}.csv
#
# Which clips run: edit the CLIPS=(...) array below (comment lines out to filter, like the hocap
# scripts), or override with CLIPS_OVERRIDE="clipA clipB". Keep in sync with train_sequences.sh.
# Env vars: FORCE=1 (re-run rollouts), CLIP_CLASS=..., CLIPS_OVERRIDE="a b c",
#   N_ROLLOUTS, TIMESTEPS (dir tag only), VIDEO=1, PY=<python>.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────
TASK="Robotis-G1-Shadow-Locomanip-Direct-v0"
CLIP_CLASS="${CLIP_CLASS:-single_rigid}"
N_ROLLOUTS="${N_ROLLOUTS:-32}"
TIMESTEPS="${TIMESTEPS:-40000}"   # must match training (directory-naming only)
PY="${PY:-python}"
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-0}"  # 0 → rollout.py fits the full sequence

# Clip names to evaluate. Comment out lines to filter (like the hocap scripts). Keep this list in
# sync with train_sequences.sh. Or override the WHOLE list with CLIPS_OVERRIDE="clipA clipB".
CLIPS=(
    "s100_seg00_pan"
    "s100_seg02_kettle"
    "s100_seg03_cup"
    "s101_seg30_bowl"
    "s101_seg29_pot"
    "s101_seg18_potlid"
    # "s103_seg09_knife"
    # "s101_seg23_salt"
)
# Optional: replace the whole list from the environment (space-separated).
[[ -n "${CLIPS_OVERRIDE:-}" ]] && read -ra CLIPS <<< "${CLIPS_OVERRIDE}"

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_BASE="${PROJECT_DIR}/source/robotis_sh5/data/processed/parahome"
CHECKPOINT_BASE="${DATA_BASE}/g1_shadow_locomanip/${CLIP_CLASS}"
FORCE="${FORCE:-0}"

# ── Rollout loop ──────────────────────────────────────────────────────────────
TOTAL="${#CLIPS[@]}"
IDX=0
for clip in "${CLIPS[@]}"; do
    IDX=$(( IDX + 1 ))
    CKPT_FILE="${CHECKPOINT_BASE}/${clip}/0/agent.pt"
    EVAL_TAG="evaluation_ep_le_${TIMESTEPS}"
    OUT_DIR="${CHECKPOINT_BASE}/${clip}/0/${EVAL_TAG}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[eval-g1 ${IDX}/${TOTAL}] class=${CLIP_CLASS}  clip=${clip}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ ! -f "${CKPT_FILE}" ]]; then
        echo "[eval-g1] ERROR: checkpoint not found — run train_sequences.sh first.  ${CKPT_FILE}"
        continue
    fi
    if [[ -f "${OUT_DIR}/metrics.csv" && "${FORCE}" -eq 0 ]]; then
        echo "[eval-g1] metrics.csv exists — skipping.  (FORCE=1 to override)  ${OUT_DIR}/metrics.csv"
        continue
    fi

    VIDEO_ARGS=()
    if [[ "${VIDEO}" -eq 1 ]]; then
        VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}")
    fi

    cd "${PROJECT_DIR}"
    "${PY}" scripts/skrl/rollout.py \
        --task "${TASK}" --checkpoint "${CKPT_FILE}" \
        --output_dir "${OUT_DIR}" --n_rollouts "${N_ROLLOUTS}" --headless \
        "${VIDEO_ARGS[@]}" \
        --clip_class "${CLIP_CLASS}" --clip_name "${clip}"
done

# ── Aggregate metrics ─────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[eval-g1] Aggregating metrics under ${DATA_BASE} ..."
bash "${SCRIPT_DIR}/../evaluate.bash" "${DATA_BASE}"
echo ""
echo "[eval-g1] Done.  → ${DATA_BASE}/g1_shadow_locomanip/method{1,2,3}.csv"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
