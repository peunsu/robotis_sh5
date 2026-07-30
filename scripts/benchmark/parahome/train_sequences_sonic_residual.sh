#!/usr/bin/env bash
# =============================================================================
# train_sequences_sonic_residual.sh — ParaHome pipeline for the SONIC-RESIDUAL
# variant of the G1 + bimanual Shadow-hand whole-body loco-manipulation task.
#
# This variant drives the 29 G1 BODY DOF with a FROZEN SONIC decoder (the policy
# emits a 64-D latent residual z_res, pre-quantization, GRAIL Eq.6) plus a 36-D
# ABSOLUTE bimanual hand action. Task:
#   Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0
#
# NO-PRETRAIN regime: train runs FROM SCRATCH (no --checkpoint). RSI is seeded
# DIRECTLY from the PyRoki retarget reference — every frame is a valid episode
# start via the env's where_ref reset path (cfg.pretrain_cache_warmstart=False),
# so the old object-free pretrain phase + its warm-start cache are removed.
#
# Mirrors train_sequences.sh (the plain locomanip pipeline) but adds the TWO
# SONIC-specific prep steps and the PyRoki retarget, with per-step conda-env
# switching (retarget needs env_pyroki; everything else env_isaaclab):
#
# Per clip:
#   1. Verify the processed SMPL-X clip exists (smplx/<class>/<clip>/0/trajectory.npz).
#   2. Hand-mesh contact map (env_isaaclab, Option A / DexMachina) → smplx/<class>/<clip>/0/hand_contact.npz
#      (object-surface wrap-link contact targets; retarget input). SKIP_RETARGET=1 to skip.
#   3. PyRoki retarget (env_pyroki) → g1_shadow/<class>/<clip>/0/trajectory_pyroki.npz
#      (g1_joint_pos + g1_root_pose; seeds RSI reset pose + hand base + spawn root). SKIP_RETARGET=1 to skip.
#   4. SONIC SMPL prep (env_isaaclab) → g1_shadow/<class>/<clip>/0/sonic_smpl_50fps.npz
#      (smpl_joints_local + root_q_zb + wrist_ref, resampled 30→50 fps). *** HARD-REQUIRED by the env ***
#      (reads wrist_ref + g1_root_quat0 from the retarget npz → run AFTER step 3).
#   5. Train (object spawned when its converted USD exists; RSI seeded from the retarget reference) → agent.pt.
#
# Checkpoints/metrics tree (evaluate.bash-compatible; clip_name at path parts[1]):
#   data/processed/parahome/g1_shadow_sonic_residual/<clip_class>/<clip_name>/0/
#
# ONE-TIME PREREQ (not per-clip): the composite PyRoki URDF must exist —
#   source/robotis_sh5/data/robots/G1/urdf_pyroki/g1_shadow.urdf
#   build it once with:  <env_isaaclab python> scripts/process_dataset/assets/export_g1_shadow_urdf.py
#
# Which clips run: edit CLIPS=(...) (comment lines to filter) or CLIPS_OVERRIDE="a b c".
# Env vars: FORCE=1 (re-run all), SKIP_RETARGET=1, CLIP_CLASS=..., CLIPS_OVERRIDE="a b c",
#   NUM_ENVS / TIMESTEPS, VIDEO=1, PY=<env_isaaclab python>, PY_PYROKI=<env_pyroki python>.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────
TASK="Robotis-G1-Shadow-Locomanip-SonicResidual-Direct-v0"
CLIP_CLASS="${CLIP_CLASS:-single_rigid}"
NUM_ENVS="${NUM_ENVS:-2048}"
TIMESTEPS="${TIMESTEPS:-41000}"
# Two conda envs: SONIC train/smpl-prep run in env_isaaclab; PyRoki retarget in env_pyroki.
PY="${PY:-/home/peunsu/anaconda3/envs/env_isaaclab/bin/python}"
PY_PYROKI="${PY_PYROKI:-/home/peunsu/anaconda3/envs/env_pyroki/bin/python}"
SKIP_RETARGET="${SKIP_RETARGET:-0}"      # 1 → skip PyRoki retarget (degrades SONIC wrist_ref + RSI reset pose)

# Set VIDEO=1 to record a training mp4 every VIDEO_INTERVAL steps.
VIDEO="${VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"

# Clip names under smplx/${CLIP_CLASS}/ to run. Comment out lines to filter (like the hocap scripts).
# Defaults: single_rigid clips with converted rigid object USDs. Only clips that ALSO have a PyRoki
# retarget (or SKIP_RETARGET builds one) and a SONIC smpl npz will train well. Override the whole list
# from the env with CLIPS_OVERRIDE="clipA clipB".
CLIPS=(
    "s100_seg00_pan"
    "s101_seg12_knife"
    "s101_seg29_pot"
    "s101_seg30_bowl"
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
CHECKPOINT_BASE="${DATA_BASE}/g1_shadow_sonic_residual/${CLIP_CLASS}"
LOG_BASE="${PROJECT_DIR}/logs/skrl/g1_shadow_sonic_residual"
URDF="${PROJECT_DIR}/source/robotis_sh5/data/robots/G1/urdf_pyroki/g1_shadow.urdf"
FORCE="${FORCE:-0}"

# newest agent_*.pt in a log tree, created after a sentinel file (fallback: newest overall)
find_ckpt() {  # $1=log_tree $2=sentinel
    local c
    c=$(find "$1" -name "agent_*.pt" -newer "$2" 2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)
    [[ -z "${c}" ]] && c=$(find "$1" -name "agent_*.pt" 2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)
    echo "${c}"
}

# ── One-time prereq check ──────────────────────────────────────────────────────
if [[ ! -f "${URDF}" ]]; then
    echo "[sonic] WARN: composite PyRoki URDF missing: ${URDF}"
    echo "        Build it once (env_isaaclab): python scripts/process_dataset/assets/export_g1_shadow_urdf.py"
    echo "        Retargeting (step 3) will fail without it."
fi

# ── Main loop ─────────────────────────────────────────────────────────────────
TOTAL="${#CLIPS[@]}"
IDX=0
for clip in "${CLIPS[@]}"; do
    IDX=$(( IDX + 1 ))
    CKPT_DIR="${CHECKPOINT_BASE}/${clip}/0"
    CKPT_FILE="${CKPT_DIR}/agent.pt"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[sonic ${IDX}/${TOTAL}] class=${CLIP_CLASS}  clip=${clip}"
    echo "  train=${TIMESTEPS} steps (${NUM_ENVS} envs)  from scratch (no pretrain; reference-seeded RSI)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ -f "${CKPT_FILE}" && "${FORCE}" -eq 0 ]]; then
        echo "[sonic] Checkpoint exists — skipping.  (FORCE=1 to override)  ${CKPT_FILE}"
        continue
    fi
    mkdir -p "${CKPT_DIR}"
    cd "${PROJECT_DIR}"

    # ── Step 1: clip data check ───────────────────────────────────────────────
    CLIP_NPZ="${DATA_BASE}/smplx/${CLIP_CLASS}/${clip}/0/trajectory.npz"
    if [[ ! -f "${CLIP_NPZ}" ]]; then
        echo "[sonic] ERROR: processed clip not found — run parahome.py first."
        echo "        Expected: ${CLIP_NPZ}"
        continue
    fi

    # ── Step 2: hand-mesh contact map (Option A / DexMachina) → hand_contact.npz ──
    #   Precomputes the wrap-link (palm/phalanges) contact targets the retarget loads for a power grasp:
    #   object surface vertices near the SMPL-X hand → farthest-point-subsampled → assigned to robot links
    #   via the human hand vertex's link (robot-pose-free). MUST run BEFORE step 3 (retarget reads it).
    #   env_isaaclab (smplx + trimesh). Skipped if the retarget is skipped (it is retarget input).
    HANDC_NPZ="${DATA_BASE}/smplx/${CLIP_CLASS}/${clip}/0/hand_contact.npz"
    if [[ "${SKIP_RETARGET}" -eq 1 ]]; then
        echo "[sonic] Step 2/5 — hand contact SKIPPED (SKIP_RETARGET=1)."
    elif [[ -f "${HANDC_NPZ}" && "${FORCE}" -eq 0 ]]; then
        echo "[sonic] Step 2/5 — hand contact exists — skipping.  ${HANDC_NPZ}"
    else
        echo "[sonic] Step 2/5 — hand-mesh contact map (env_isaaclab): ${clip}"
        "${PY}" scripts/process_dataset/dataset/parahome_hand_contact.py \
            --class "${CLIP_CLASS}" --clip "${clip}" || \
            echo "[sonic] WARN: hand contact failed — retarget falls back to fingertip-pad contact only."
    fi

    # ── Step 3: PyRoki retarget (env_pyroki) → trajectory_pyroki.npz ───────────
    RETARGET_NPZ="${DATA_BASE}/g1_shadow/${CLIP_CLASS}/${clip}/0/trajectory_pyroki.npz"
    if [[ "${SKIP_RETARGET}" -eq 1 ]]; then
        echo "[sonic] Step 3/5 — retarget SKIPPED (SKIP_RETARGET=1; degrades SONIC wrist_ref + RSI reset pose)."
    elif [[ -f "${RETARGET_NPZ}" && "${FORCE}" -eq 0 ]]; then
        echo "[sonic] Step 3/5 — PyRoki retarget exists — skipping.  ${RETARGET_NPZ}"
    else
        echo "[sonic] Step 3/5 — PyRoki retarget (env_pyroki): ${clip}"
        "${PY_PYROKI}" scripts/process_dataset/retarget/retarget_g1_pyroki.py \
            --class "${CLIP_CLASS}" --clip "${clip}" || \
            echo "[sonic] WARN: PyRoki retarget failed — SONIC smpl prep will use zero wrist_ref (degraded)."
    fi

    # ── Step 4: SONIC SMPL prep (env_isaaclab) → sonic_smpl_50fps.npz ──────────
    #   HARD-required by the env (_load_reference raises if missing). Reads the retarget npz for
    #   wrist_ref + g1_root_quat0, so it must run AFTER step 3.
    SONIC_NPZ="${DATA_BASE}/g1_shadow/${CLIP_CLASS}/${clip}/0/sonic_smpl_50fps.npz"
    if [[ -f "${SONIC_NPZ}" && "${FORCE}" -eq 0 ]]; then
        echo "[sonic] Step 4/5 — SONIC smpl npz exists — skipping.  ${SONIC_NPZ}"
    else
        echo "[sonic] Step 4/5 — SONIC SMPL prep (env_isaaclab): ${clip}"
        "${PY}" scripts/process_dataset/dataset/parahome_smpl_for_sonic.py \
            --class "${CLIP_CLASS}" --clip "${clip}" --overwrite
    fi
    if [[ ! -f "${SONIC_NPZ}" ]]; then
        echo "[sonic] ERROR: sonic_smpl_50fps.npz missing after prep — the env requires it; skipping clip."
        continue
    fi

    VIDEO_ARGS=()
    if [[ "${VIDEO}" -eq 1 ]]; then
        VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}" --video_interval "${VIDEO_INTERVAL}")
    fi

    # ── Step 5: Train (from scratch; RSI seeded from the retarget reference; object if USD exists) ─
    #   No --checkpoint: the policy trains from random init. RSI reset poses come from the PyRoki
    #   retarget reference (env cfg.pretrain_cache_warmstart=False), so no object-free pretrain is needed.
    echo "[sonic] Step 5/5 — Training (${TIMESTEPS} steps, from scratch) ..."
    touch "${CKPT_DIR}/.sentinel"
    "${PY}" scripts/skrl/train.py \
        --task "${TASK}" --num_envs "${NUM_ENVS}" \
        --timesteps "${TIMESTEPS}" --headless "${VIDEO_ARGS[@]}" \
        --clip_class "${CLIP_CLASS}" --clip_name "${clip}"

    LATEST_CKPT=$(find_ckpt "${LOG_BASE}" "${CKPT_DIR}/.sentinel")
    if [[ -z "${LATEST_CKPT}" ]]; then
        echo "[sonic] ERROR: train checkpoint not found in ${LOG_BASE}."; continue
    fi
    cp "${LATEST_CKPT}" "${CKPT_FILE}"
    echo "[sonic] Checkpoint → ${CKPT_FILE}"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[sonic] All ${TOTAL} clips processed.  Tree: ${CHECKPOINT_BASE}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
