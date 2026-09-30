#!/usr/bin/env bash
# =============================================================================
# train_sequences_sonic_residual_mppi.sh — ParaHome SONIC-residual pipeline with the
# MPPI hand refine as stage 1 (instead of the RL hand pretrain).
#
# train_sequences_sonic_residual.sh trains stage 2 on the hand trajectory the RL floating-hand
# policy produced (train/evaluate_sequences_hand_pretrain.sh -> g1_shadow_hand_pretrain/). Here that
# trajectory comes from robotis_sh5_monodex instead: sampling-based MPC (MPPI, MuJoCo Warp) makes the
# floating Shadow hands carry the object, starting from the PyRoki hands, with J0 coupled to J1 as the
# Isaac hand's tendon moves it (J0 = 1.142 J1). No RL is trained for stage 1; one MPPI run per clip
# (~7-18 min on an RTX 5090; ParaHome furniture CoACD is cached after the first clip that uses it).
#   Task: Robotis-G1-Shadow-Locomanip-SonicResidual-Mppi-Direct-v0 (G1ShadowSonicResidualMppiEnvCfg:
#   stage-1 tree g1_shadow_hand_mppi, residual base = the MPPI joint angles, object mass fixed at 0.3 kg;
#   MPPI itself keeps monodex's density 1000 kg/m^3).
#
# Per clip:
#   1. Verify the processed SMPL-X clip exists; add smplx_joints / fingertip_pad_pos if it was processed
#      without smplx (parahome_add_smplx_fk.py, parahome.py's FK; other keys untouched).
#   2. Hand contact (env_isaaclab) → smplx/<class>/<clip>/0/hand_contact.npz  (surface normals: the
#      retarget and this env read it). Same as train_sequences_sonic_residual.sh.
#   3. PyRoki retarget (env_pyroki) → g1_shadow/<class>/<clip>/0/trajectory_pyroki.npz
#   4. SONIC SMPL prep (env_isaaclab) → g1_shadow/<class>/<clip>/0/sonic_smpl_50fps.npz
#   5. MPPI hand refine → g1_shadow_hand_mppi/<class>/<clip>/0/
#        a. to-hand contact map (env_isaaclab) → smplx/<class>/<clip>/0/hand_contact_tohand.npz
#           (the MPPI reward was tuned on to-hand normals; the surface file above stays as it is)
#        b. monodex launch.py (conda env retargeting) → g1_shadow_hand_mppi/_runs/... (stages 1-5); its log is
#           shown live and kept in <clip>/0/mppi.log
#        c. export_mppi_hand_traj.py → hand_traj_best.npz + hand_traj_best_metrics.json (object tracking
#           error, for inspection). Every MPPI run is used for stage 2; there is no pass/fail gate.
#        d. VIDEO_MPPI=1: mppi_refine.mp4 (kinematic reference | MPPI rollout, object camera)
#        e. stage1_hand_contact.py (env_isaaclab) → hand_contact_stage1.npz (the MPPI hands' contacts)
#   6. Train stage 2 → g1_shadow_sonic_residual_mppi/<class>/<clip>/0/agent.pt
#
# Trees (evaluate.bash-compatible; clip_name at path parts[1]):
#   data/processed/parahome/g1_shadow_hand_mppi/<clip_class>/<clip_name>/0/      stage 1 (MPPI)
#   data/processed/parahome/g1_shadow_sonic_residual_mppi/<clip_class>/<clip_name>/0/   stage 2
#   logs/skrl/g1_shadow_sonic_residual_mppi/
#
# ONE-TIME PREREQS: the composite PyRoki URDF (as in train_sequences_sonic_residual.sh), the monodex repo and
# its conda env: MONODEX_ROOT / PY_RETARGET in local_paths.env (local_paths.env.example).
#
# Which clips run: edit CLIPS=(...) (comment lines to filter) or CLIPS_OVERRIDE="a b c".
# Env vars: FORCE=1 (re-run all), FORCE_MPPI=1 (re-run only step 5), SKIP_RETARGET=1, SKIP_TRAIN=1 (steps 1-5
#   only), VIDEO_MPPI=0, CLIP_CLASS=..., CLIPS_OVERRIDE="a b c", NUM_ENVS / TIMESTEPS,
#   VIDEO=1, PY / PY_PYROKI / PY_RETARGET / MONODEX_ROOT.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────
TASK="Robotis-G1-Shadow-Locomanip-SonicResidual-Mppi-Direct-v0"
CLIP_CLASS="${CLIP_CLASS:-single_rigid}"
NUM_ENVS="${NUM_ENVS:-2048}"
TIMESTEPS="${TIMESTEPS:-41000}"
# PY / PY_PYROKI / PY_RETARGET / MONODEX_ROOT: 환경변수 → 저장소 루트 local_paths.env (git 제외) 순으로 읽는다.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/local_paths.sh"
require_local_path PY PY_PYROKI PY_RETARGET MONODEX_ROOT
SKIP_RETARGET="${SKIP_RETARGET:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
VIDEO_MPPI="${VIDEO_MPPI:-1}"

# Set VIDEO=1 to record a training mp4 every VIDEO_INTERVAL steps.
VIDEO="${VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"

# 30 ParaHome single_rigid clips. The first 8 are train_sequences_sonic_residual.sh's, so the two stage-1
# sources can be compared on them; 20 of the added clips were processed without smplx and get their
# smplx_joints / fingertip_pad_pos in step 1 (parahome_add_smplx_fk.py).
CLIPS=(
    # the first 8 (train_sequences_sonic_residual.sh's set, with RL hand-pretrain results to compare against)
    "s101_seg12_knife"
    "s100_seg00_pan"
    "s101_seg29_pot"
    "s101_seg30_bowl"
    "s66_seg26_pan"
    "s53_seg19_knife"
    "s152_seg21_pot"
    "s71_seg27_bowl"
    # added 2026-09-29: three clips per ParaHome single_rigid object
    "s12_seg07_book"
    "s11_seg07_book"
    "s155_seg11_book"
    "s126_seg19_bowl"
    "s109_seg02_cup"
    "s105_seg21_cup"
    "s101_seg00_cup"
    "s197_seg07_cuttingboard"
    "s145_seg22_cuttingboard"
    "s198_seg29_cuttingboard"
    "s117_seg09_kettle"
    "s113_seg01_kettle"
    "s207_seg06_kettle"
    "s125_seg23_knife"
    "s104_seg18_pan"
    "s142_seg00_pot"
    "s44_seg00_potlid"
    "s1_seg06_potlid"
    "s92_seg16_potlid"
    "s123_seg21_salt"
    "s126_seg14_salt"
    "s111_seg20_salt"
)
[[ -n "${CLIPS_OVERRIDE:-}" ]] && read -ra CLIPS <<< "${CLIPS_OVERRIDE}"

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${PROJECT_DIR}/source/robotis_sh5/data"
DATA_BASE="${DATA_ROOT}/processed/parahome"
MPPI_BASE="${DATA_BASE}/g1_shadow_hand_mppi"
MPPI_RUNS="${MPPI_BASE}/_runs"          # monodex --output-root-dir (its layout + shared mesh caches)
MPPI_ID="mppi"                          # monodex --data-id
CHECKPOINT_BASE="${DATA_BASE}/g1_shadow_sonic_residual_mppi/${CLIP_CLASS}"
LOG_DIR_NAME="g1_shadow_sonic_residual_mppi"
LOG_BASE="${PROJECT_DIR}/logs/skrl/${LOG_DIR_NAME}"
URDF="${PROJECT_DIR}/source/robotis_sh5/data/robots/G1/urdf_pyroki/g1_shadow.urdf"
FORCE="${FORCE:-0}"
FORCE_MPPI="${FORCE_MPPI:-${FORCE}}"

# newest agent_*.pt in a log tree, created after a sentinel file (fallback: newest overall).
# `xargs -r`: with no match a bare xargs runs `ls -t` on the current directory (see hand_pretrain scripts).
find_ckpt() {  # $1=log_tree $2=sentinel
    local c
    c=$(find "$1" -name "agent_*.pt" -newer "$2" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1 || true)
    [[ -z "${c}" ]] && c=$(find "$1" -name "agent_*.pt" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1 || true)
    [[ -f "${c}" ]] || c=""
    echo "${c}"
}

# ── One-time prereq checks ────────────────────────────────────────────────────
if [[ ! -f "${URDF}" ]]; then
    echo "[sonic-mppi] WARN: composite PyRoki URDF missing: ${URDF}"
    echo "             Build it once (env_isaaclab): python scripts/process_dataset/assets/export_g1_shadow_urdf.py"
fi
if [[ ! -f "${MONODEX_ROOT}/retargeting/launch.py" ]]; then
    echo "[sonic-mppi] ERROR: ${MONODEX_ROOT}/retargeting/launch.py not found (MONODEX_ROOT)."; exit 1
fi

# ── Main loop ─────────────────────────────────────────────────────────────────
TOTAL="${#CLIPS[@]}"
IDX=0
for clip in "${CLIPS[@]}"; do
    IDX=$(( IDX + 1 ))
    CKPT_DIR="${CHECKPOINT_BASE}/${clip}/0"
    CKPT_FILE="${CKPT_DIR}/agent.pt"
    CLIP_DIR="${DATA_BASE}/smplx/${CLIP_CLASS}/${clip}/0"
    G1_DIR="${DATA_BASE}/g1_shadow/${CLIP_CLASS}/${clip}/0"
    S1_DIR="${MPPI_BASE}/${CLIP_CLASS}/${clip}/0"
    S1_TRAJ="${S1_DIR}/hand_traj_best.npz"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[sonic-mppi ${IDX}/${TOTAL}] class=${CLIP_CLASS}  clip=${clip}"
    echo "  stage 1: MPPI hand refine (monodex)   stage 2: train=${TIMESTEPS} steps (${NUM_ENVS} envs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ -f "${CKPT_FILE}" && "${FORCE}" -eq 0 && "${SKIP_TRAIN}" -eq 0 ]]; then
        echo "[sonic-mppi] Checkpoint exists — skipping.  (FORCE=1 to override)  ${CKPT_FILE}"
        continue
    fi
    cd "${PROJECT_DIR}"

    # ── Step 1: clip data check (+ the SMPL-X keys the retarget / MPPI / env read) ─
    if [[ ! -f "${CLIP_DIR}/trajectory.npz" ]]; then
        echo "[sonic-mppi] ERROR: processed clip not found — run parahome.py first.  ${CLIP_DIR}/trajectory.npz"
        continue
    fi
    #   Clips processed without smplx lack smplx_joints / fingertip_pad_pos; add them with parahome.py's FK.
    "${PY}" scripts/process_dataset/dataset/parahome_add_smplx_fk.py --class "${CLIP_CLASS}" --clip "${clip}" \
        || { echo "[sonic-mppi] ERROR: could not add the SMPL-X keys; skipping clip."; continue; }

    # ── Step 2: hand-mesh contact map (surface normals) → hand_contact.npz ────
    if [[ "${SKIP_RETARGET}" -eq 1 ]]; then
        echo "[sonic-mppi] Step 2/6 — hand contact SKIPPED (SKIP_RETARGET=1)."
    elif [[ -f "${CLIP_DIR}/hand_contact.npz" && "${FORCE}" -eq 0 ]]; then
        echo "[sonic-mppi] Step 2/6 — hand contact exists — skipping."
    else
        echo "[sonic-mppi] Step 2/6 — hand-mesh contact map (env_isaaclab): ${clip}"
        "${PY}" scripts/process_dataset/dataset/parahome_hand_contact.py --class "${CLIP_CLASS}" --clip "${clip}" || \
            echo "[sonic-mppi] WARN: hand contact failed — retarget falls back to fingertip-pad contact only."
    fi

    # ── Step 3: PyRoki retarget (env_pyroki) → trajectory_pyroki.npz ─────────
    if [[ "${SKIP_RETARGET}" -eq 1 ]]; then
        echo "[sonic-mppi] Step 3/6 — retarget SKIPPED (SKIP_RETARGET=1)."
    elif [[ -f "${G1_DIR}/trajectory_pyroki.npz" && "${FORCE}" -eq 0 ]]; then
        echo "[sonic-mppi] Step 3/6 — PyRoki retarget exists — skipping."
    else
        echo "[sonic-mppi] Step 3/6 — PyRoki retarget (env_pyroki): ${clip}"
        "${PY_PYROKI}" scripts/process_dataset/retarget/retarget_g1_pyroki.py --class "${CLIP_CLASS}" --clip "${clip}" || \
        { echo "[sonic-mppi] WARN: retarget failed on GPU — retrying once on CPU (JAX_PLATFORMS=cpu)."
          JAX_PLATFORMS=cpu "${PY_PYROKI}" scripts/process_dataset/retarget/retarget_g1_pyroki.py \
              --class "${CLIP_CLASS}" --clip "${clip}"; } || echo "[sonic-mppi] WARN: PyRoki retarget failed."
    fi
    if [[ ! -f "${G1_DIR}/trajectory_pyroki.npz" ]]; then
        echo "[sonic-mppi] ERROR: trajectory_pyroki.npz missing — MPPI starts from the PyRoki hands; skipping clip."
        continue
    fi

    # ── Step 4: SONIC SMPL prep (env_isaaclab) → sonic_smpl_50fps.npz ────────
    if [[ -f "${G1_DIR}/sonic_smpl_50fps.npz" && "${FORCE}" -eq 0 ]]; then
        echo "[sonic-mppi] Step 4/6 — SONIC smpl npz exists — skipping."
    else
        echo "[sonic-mppi] Step 4/6 — SONIC SMPL prep (env_isaaclab): ${clip}"
        "${PY}" scripts/process_dataset/dataset/parahome_smpl_for_sonic.py --class "${CLIP_CLASS}" --clip "${clip}" --overwrite
    fi
    if [[ ! -f "${G1_DIR}/sonic_smpl_50fps.npz" ]]; then
        echo "[sonic-mppi] ERROR: sonic_smpl_50fps.npz missing after prep — the env requires it; skipping clip."
        continue
    fi

    # ── Step 5: MPPI hand refine → g1_shadow_hand_mppi/<class>/<clip>/0/ ─────
    RUN_DIR="${MPPI_RUNS}/shadow_hand/bimanual/${clip}/PARAHOME/default/${MPPI_ID}"
    if [[ -f "${S1_TRAJ}" && "${FORCE_MPPI}" -eq 0 ]]; then
        echo "[sonic-mppi] Step 5/6 — MPPI refine exists — skipping.  ${S1_DIR}"
    else
        mkdir -p "${S1_DIR}"
        rm -f "${S1_TRAJ}" "${S1_DIR}/hand_contact_stage1.npz"
        echo "[sonic-mppi] Step 5/6 a — to-hand contact map (env_isaaclab) → hand_contact_tohand.npz"
        "${PY}" scripts/process_dataset/dataset/parahome_hand_contact.py --class "${CLIP_CLASS}" --clip "${clip}" \
            --normal-source to-hand --out-name hand_contact_tohand.npz
        echo "[sonic-mppi] Step 5/6 b — monodex MPPI (${PY_RETARGET}) → ${RUN_DIR}"
        ( cd "${MONODEX_ROOT}/retargeting" && PARAHOME_SMPLX_PYTHON="${PY}" "${PY_RETARGET}" launch.py \
              --task "${clip}" --data-id "${MPPI_ID}" --no-show-viewer --no-wait-on-finish \
              --parahome-root "${DATA_ROOT}" --output-root-dir "${MPPI_RUNS}" \
              --hand-contact-file hand_contact_tohand.npz ) 2>&1 | tee "${S1_DIR}/mppi.log" \
            || { echo "[sonic-mppi] ERROR: MPPI failed — see ${S1_DIR}/mppi.log; skipping clip."; continue; }
        grep -h "Final object tracking error" "${S1_DIR}/mppi.log" | sed 's/.*| - /[sonic-mppi]   /' || true
        echo "[sonic-mppi] Step 5/6 c — export → ${S1_TRAJ}"
        "${PY_RETARGET}" scripts/process_dataset/retarget/export_mppi_hand_traj.py --run-dir "${RUN_DIR}" --out "${S1_TRAJ}" \
            || { echo "[sonic-mppi] ERROR: export failed; skipping clip."; continue; }
        if [[ "${VIDEO_MPPI}" -eq 1 ]]; then
            echo "[sonic-mppi] Step 5/6 d — MPPI video → ${S1_DIR}/mppi_refine.mp4"
            ( cd "${MONODEX_ROOT}/retargeting" && MUJOCO_GL=egl "${PY_RETARGET}" tools/render_run.py \
                  --run-dir "${RUN_DIR}" --label "MPPI ROLLOUT" --cam object --cam-distance 1.0 \
                  --out "${S1_DIR}/mppi_refine.mp4" ) > "${S1_DIR}/render.log" 2>&1 \
                || echo "[sonic-mppi] WARN: render failed — see ${S1_DIR}/render.log"
        fi
    fi
    if [[ -f "${S1_TRAJ}" && ! -f "${S1_DIR}/hand_contact_stage1.npz" ]]; then
        echo "[sonic-mppi] Step 5/6 e — MPPI contact map (env_isaaclab) → hand_contact_stage1.npz"
        "${PY}" scripts/process_dataset/dataset/stage1_hand_contact.py --dataset parahome --hand_traj "${S1_TRAJ}" || \
            echo "[sonic-mppi] WARN: stage-1 contact map failed — the env keeps the human contact map."
    fi

    if [[ "${SKIP_TRAIN}" -eq 1 ]]; then
        echo "[sonic-mppi] Step 6/6 — training SKIPPED (SKIP_TRAIN=1)."
        continue
    fi

    VIDEO_ARGS=()
    if [[ "${VIDEO}" -eq 1 ]]; then
        VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}" --video_interval "${VIDEO_INTERVAL}")
    fi

    # ── Step 6: Train stage 2 (from scratch; RSI from the retarget reference) ──
    echo "[sonic-mppi] Step 6/6 — Training (${TIMESTEPS} steps, from scratch) ..."
    mkdir -p "${CKPT_DIR}"
    touch "${CKPT_DIR}/.sentinel"
    "${PY}" scripts/skrl/train.py \
        --task "${TASK}" --num_envs "${NUM_ENVS}" \
        --timesteps "${TIMESTEPS}" --headless "${VIDEO_ARGS[@]}" \
        --clip_class "${CLIP_CLASS}" --clip_name "${clip}" \
        agent.agent.experiment.directory="${LOG_DIR_NAME}"

    LATEST_CKPT=$(find_ckpt "${LOG_BASE}" "${CKPT_DIR}/.sentinel")
    if [[ -z "${LATEST_CKPT}" ]]; then
        echo "[sonic-mppi] ERROR: train checkpoint not found in ${LOG_BASE}."; continue
    fi
    cp "${LATEST_CKPT}" "${CKPT_FILE}"
    echo "[sonic-mppi] Checkpoint → ${CKPT_FILE}"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[sonic-mppi] All ${TOTAL} clips processed."
echo "  stage 1 (MPPI) : ${MPPI_BASE}/${CLIP_CLASS}/"
echo "  stage 2        : ${CHECKPOINT_BASE}/   (evaluate_sequences_sonic_residual_mppi.sh)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
