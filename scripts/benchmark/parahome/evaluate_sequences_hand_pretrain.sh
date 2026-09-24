#!/usr/bin/env bash
# =============================================================================
# evaluate_sequences_hand_pretrain.sh — Roll out each trained FLOATING-HAND
# (stage-1) clip and aggregate metrics.
#
# Pairs with train_sequences_hand_pretrain.sh. Task:
#   Robotis-G1-Shadow-HandPretrain-Direct-v0
#
# Evaluation protocol comes from rollout.py and is what makes these numbers
# comparable across clips and runs:
#   * `adaptive_sampling = False` → EVERY episode starts at reference frame 0 and
#     plays the whole clip. No RSI sampling, so metrics are not confounded by
#     which frames a run happened to start at.
#   * `termination = False`       → the clip is rolled to the end even after the
#     policy loses the object, so a failure shows up as a large error rather than
#     a short episode.
#   * the RSI warm-start / state cache is never armed by rollout.py.
#
# What the metrics mean HERE (they are the generic columns, but the hand env
# redefines one of them):
#   e_t_cm   object position error (cm)          — the task signal
#   e_r      object rotation error (deg)
#   e_ft_cm  fingertip pad tracking error (cm)   — the contact signal
#   e_j_cm   `_errs["body"]`, which in the hand env is NOT a body error: the
#            floating hand has no torso, so that term was redefined as the mean
#            tracking error of the two PALM keypoints. Read it as "wrist position
#            error", not "whole-body error". Full-body runs put a 13-keypoint
#            whole-body mean in the same column, so this column is NOT comparable
#            between the two trees.
#
# ROLLOUT SPREAD (why N_ROLLOUTS matters beyond averaging)
# -------------------------------------------------------
# metrics.csv holds ONE ROW PER ROLLOUT (N_ROLLOUTS rows), not just a mean. For
# stage 2 that spread is the useful part: stage 1's trajectory is only worth
# handing over on frames where the policy is CONSISTENT, and the row-to-row
# variance is the cheapest measure of that. Keep N_ROLLOUTS >= 16 if you intend
# to use the run as a stage-2 reference source.
#
# For each clip:
#   1. Load agent.pt from the g1_shadow_hand_pretrain tree.
#   2. rollout.py → metrics.csv (+ per-frame error traces + optional mp4).
#   3. After all clips, evaluate.bash aggregates per-method CSVs.
#
# Output tree (evaluate.bash compatible — clip_name lands at path parts[1]):
#   data/processed/parahome/g1_shadow_hand_pretrain/<clip_class>/<clip_name>/0/
#       agent.pt
#       evaluation_ep_le_<TIMESTEPS>/metrics.csv
# Aggregates → data/processed/parahome/g1_shadow_hand_pretrain/method{1,2,3}.csv
#
# Deliberately single-pass: the sonic-residual script runs a second pass with
# `--cam_preset old` to compose a side-by-side and use it as a physics receipt.
# That existed to compare two camera angles on the full-body view; there is no
# such comparison to make here, and a second pass would double the GPU cost.
#
# Which clips run: keep in sync with train_sequences_hand_pretrain.sh.
# Env vars: FORCE=1 (re-run rollouts), CLIP_CLASS=..., CLIPS_OVERRIDE="a b c",
#   HAND_TRAJ=0 (skip the hand_traj_best.npz dump, see below),
#   CONTACT_MAP=0 (skip the hand_contact_stage1.npz contact map built from the dump),
#   N_ROLLOUTS, TIMESTEPS (dir tag only), VIDEO=1, SEED=42,
#   PY=<env_isaaclab python>, VIDEO_RESOLUTION=WxH.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────
TASK="Robotis-G1-Shadow-HandPretrain-Direct-v0"
CLIP_CLASS="${CLIP_CLASS:-single_rigid}"
N_ROLLOUTS="${N_ROLLOUTS:-32}"
TIMESTEPS="${TIMESTEPS:-41000}"    # directory-naming tag only (evaluation_ep_le_<TIMESTEPS>)
# PY: 환경변수 → 저장소 루트 local_paths.env (git 제외) 순으로 읽는다.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/local_paths.sh"
require_local_path PY
VIDEO="${VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-0}"  # 0 → rollout.py fits the full sequence
SEED="${SEED:-42}"                 # explicit: the determinism contract of the run
RES_ARGS=(); [[ -n "${VIDEO_RESOLUTION:-}" ]] && RES_ARGS=(--video_resolution "${VIDEO_RESOLUTION}")
# 손 궤적 덤프 (2026-09-09)
# 2단계(sonic residual) env 는 hand_kpt_from_hand_pretrain / sonic_hand_residual_base 가 켜져 있으면
#   <clip>/0/hand_traj_best.npz   (보상 합이 가장 큰 rollout 의 손가락 관절·손바닥 자세·지문·접촉력)
# 를 읽어 손·손목 보상 목표와 잔차 손 액션의 기준점으로 쓴다. 이 평가에서 그 파일을 함께 만든다:
# rollout.py 에 --dump_hand_traj 를 붙여 OUT_DIR 에 hand_traj.npz / hand_traj_best.npz 를 쓰고, best 를
# <clip>/0/ 로 복사한다. metrics.csv 와 hand_traj_best.npz 가 모두 있을 때만 건너뛴다 (metrics.csv 만
# 있으면 rollout 을 다시 돌린다 — 두 파일은 같은 rollout 묶음에서 나와야 스프레드 판단과 손 궤적이 일치).
HAND_TRAJ="${HAND_TRAJ:-1}"        # 0 → previous behaviour: no dump, skip on metrics.csv alone
CONTACT_MAP="${CONTACT_MAP:-1}"    # [stage1-contact-map] 1 → hand_traj_best.npz 옆에 hand_contact_stage1.npz 생성 (CPU 전용)

# Must match train_sequences_hand_pretrain.sh (stage 1's output feeds stage 2, so the two stages
# have to cover the same clip set). Selection criteria are documented in the train script.
CLIPS=(
    "s100_seg00_pan"
    "s101_seg12_knife"
    "s101_seg29_pot"
    "s101_seg30_bowl"
    "s207_seg06_kettle"
    "s66_seg26_pan"
    "s55_seg31_knife"
    "s73_seg31_pot"
    "s33_seg18_bowl"
    "s127_seg29_pan"
    "s53_seg19_knife"
    "s152_seg21_pot"
    "s71_seg27_bowl"
    # "s10_seg03_book"     # ← hand_contact.npz 없음
    # "s100_seg02_kettle"
    # "s100_seg03_cup"
)
[[ -n "${CLIPS_OVERRIDE:-}" ]] && read -ra CLIPS <<< "${CLIPS_OVERRIDE}"

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_BASE="${PROJECT_DIR}/source/robotis_sh5/data/processed/parahome"
CHECKPOINT_BASE="${DATA_BASE}/g1_shadow_hand_pretrain/${CLIP_CLASS}"
FORCE="${FORCE:-0}"

# ── Rollout loop ──────────────────────────────────────────────────────────────
TOTAL="${#CLIPS[@]}"
IDX=0
N_OK=0
N_SKIP=0
for clip in "${CLIPS[@]}"; do
    IDX=$(( IDX + 1 ))
    CKPT_FILE="${CHECKPOINT_BASE}/${clip}/0/agent.pt"
    EVAL_TAG="evaluation_ep_le_${TIMESTEPS}"
    OUT_DIR="${CHECKPOINT_BASE}/${clip}/0/${EVAL_TAG}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[eval-hand ${IDX}/${TOTAL}] class=${CLIP_CLASS}  clip=${clip}"
    echo "  frame-0 start, no termination, ${N_ROLLOUTS} rollouts"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ ! -f "${CKPT_FILE}" ]]; then
        echo "[eval-hand] SKIP: checkpoint not found — run train_sequences_hand_pretrain.sh first."
        echo "            ${CKPT_FILE}"
        N_SKIP=$(( N_SKIP + 1 ))
        continue
    fi
    # adopt an earlier dump (same OUT_DIR, or the legacy *_handtraj dir)
    HAND_BEST="${CHECKPOINT_BASE}/${clip}/0/hand_traj_best.npz"
    if [[ "${HAND_TRAJ}" -eq 1 && ! -f "${HAND_BEST}" && "${FORCE}" -eq 0 ]]; then
        for _src in "${OUT_DIR}/hand_traj_best.npz" "${OUT_DIR}_handtraj/hand_traj_best.npz"; do
            if [[ -f "${_src}" ]]; then
                cp -f "${_src}" "${HAND_BEST}"
                echo "[eval-hand] hand_traj_best.npz copied from ${_src}"
                break
            fi
        done
    fi
    if [[ -f "${OUT_DIR}/metrics.csv" && "${FORCE}" -eq 0 ]]; then
        if [[ "${HAND_TRAJ}" -eq 0 || -f "${HAND_BEST}" ]]; then                       # [stage1-hand]
            echo "[eval-hand] metrics.csv exists — skipping.  (FORCE=1 to override)"
            # 덤프는 있는데 접촉 맵만 없으면 맵만 만든다 (CPU 전용, GPU 불필요)
            if [[ "${CONTACT_MAP}" -eq 1 && -f "${HAND_BEST}" && ! -f "$(dirname "${HAND_BEST}")/hand_contact_stage1.npz" ]]; then
                echo "[eval-hand] hand_contact_stage1.npz missing — generating from the existing dump."
                "${PY}" "${PROJECT_DIR}/scripts/process_dataset/dataset/stage1_hand_contact.py" --hand_traj "${HAND_BEST}" \
                    || echo "[eval-hand] WARNING: contact map generation failed for ${clip}."
            fi
            N_SKIP=$(( N_SKIP + 1 ))
            continue
        fi
        echo "[eval-hand] metrics.csv exists but hand_traj_best.npz is missing — re-running the rollout with --dump_hand_traj."
    fi

    VIDEO_ARGS=()
    if [[ "${VIDEO}" -eq 1 ]]; then
        VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}")
    fi
    HAND_ARGS=(); [[ "${HAND_TRAJ}" -eq 1 ]] && HAND_ARGS=(--dump_hand_traj)          # [stage1-hand]

    cd "${PROJECT_DIR}"
    # `|| true` + the metrics.csv existence check below: one clip that crashes (e.g. a missing
    # wrist_ref.npz) must not abort the remaining clips — `set -e` would otherwise kill the loop.
    "${PY}" scripts/skrl/rollout.py \
        --task "${TASK}" --checkpoint "${CKPT_FILE}" \
        --output_dir "${OUT_DIR}" --n_rollouts "${N_ROLLOUTS}" --headless \
        --seed "${SEED}" "${VIDEO_ARGS[@]}" "${RES_ARGS[@]}" "${HAND_ARGS[@]}" \
        --clip_class "${CLIP_CLASS}" --clip_name "${clip}" || true

    if [[ -f "${OUT_DIR}/metrics.csv" ]]; then
        echo "[eval-hand] → ${OUT_DIR}/metrics.csv"
        N_OK=$(( N_OK + 1 ))
        # best rollout → <clip>/0/hand_traj_best.npz (what stage 2 reads)
        if [[ "${HAND_TRAJ}" -eq 1 ]]; then
            if [[ -f "${OUT_DIR}/hand_traj_best.npz" ]]; then
                cp -f "${OUT_DIR}/hand_traj_best.npz" "${HAND_BEST}"
                echo "[eval-hand] → ${HAND_BEST}"
                # 1단계 롤아웃 → 32링크 접촉 맵 (스테이지 2 hand_pretrain_contact_map 의 입력).
                #    사람 맵(parahome_hand_contact.py)과 같은 코어(frame_contacts)로 만들어 같은 폴더에 hand_contact_stage1.npz 로 둔다.
                if [[ "${CONTACT_MAP}" -eq 1 ]]; then
                    "${PY}" "${PROJECT_DIR}/scripts/process_dataset/dataset/stage1_hand_contact.py" --hand_traj "${HAND_BEST}" \
                        || echo "[eval-hand] WARNING: contact map generation failed for ${clip}."
                fi
            else
                echo "[eval-hand] WARNING: rollout produced no hand_traj_best.npz for ${clip}."
            fi
        fi
    else
        echo "[eval-hand] ERROR: rollout produced no metrics.csv for ${clip}."
        N_SKIP=$(( N_SKIP + 1 ))
    fi
done

# ── Aggregate metrics ─────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[eval-hand] rollouts written ${N_OK}, skipped/failed ${N_SKIP}"
echo "[eval-hand] Aggregating metrics under ${DATA_BASE} ..."
bash "${SCRIPT_DIR}/../evaluate.bash" "${DATA_BASE}"
echo ""
echo "[eval-hand] Done.  → ${DATA_BASE}/g1_shadow_hand_pretrain/method{1,2,3}.csv"
echo "  NOTE e_j_cm in this tree = PALM (wrist) position error, not a whole-body"
echo "       keypoint mean — not comparable with the g1_shadow_sonic_residual tree."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
