#!/usr/bin/env bash
# =============================================================================
# train_sequences_hand_pretrain.sh — ParaHome pipeline for STAGE 1 of the
# two-stage loco-manip plan: FLOATING-HAND dexterous pretrain.
#
# Two free-floating Shadow hands (no G1 arm, no body, no SONIC) learn bimanual
# manipulation under physics; the wrist + finger trajectory they produce becomes
# the tracking target for the full-body stage. Task:
#   Robotis-G1-Shadow-HandPretrain-Direct-v0
#
# The premise (professor's note): the kinematic capture's OBJECT pose is
# trustworthy while the retargeted HAND pose is not — so the hand pose is what
# gets refined under physics, and the refined version replaces the retargeted
# hand reference in stage 2.
#
# HOW THIS DIFFERS FROM train_sequences_sonic_residual.sh
#   - NO SONIC prep. cfg.use_sonic=False, so sonic_smpl_50fps.npz is never read
#     (the assert that requires it lives inside the SONIC-guarded block).
#   - NEEDS wrist_ref.npz. The floating hands are ROOTED at robot0_{l,r}_palm and
#     the RSI reset writes that root pose directly (it was robot0_{l,r}_wrist until
#     2026-09-07, when that zero-DOF link was dropped from the asset). The retarget npz stores only
#     g1_joint_pos + g1_root_pose (pelvis), so the wrist pose is implied by FK but
#     never written out — export_wrist_ref.py runs pinocchio FK to produce it.
#     The env RAISES FileNotFoundError without it, so this step is HARD-REQUIRED.
#   - EXPORTS a stage-2 target. After training, export_hand_rl.py runs one
#     deterministic rollout per clip → hand_rl.npz. That file IS the deliverable;
#     a trained checkpoint with no export gives stage 2 nothing to track.
#   - AUDITS reachability. audit_wrist_reachability.py checks whether a G1 arm can
#     actually reach the wrist trajectory stage 1 invented. Stage 1 lets the wrist
#     float with no anchor (user's decision), so this is the post-hoc gate that
#     replaces anchoring. The retargeted baseline measures 0.0% unreachable across
#     all 16 clips, so any excess here was produced by stage 1.
#
# Per clip:
#   1. Verify the processed SMPL-X clip exists (smplx/<class>/<clip>/0/trajectory.npz).
#   2. Hand-mesh contact map (env_isaaclab) → smplx/<class>/<clip>/0/hand_contact.npz
#      (retarget input AND the env's contact/CWS reward input). SKIP_RETARGET=1 to skip.
#   3. PyRoki retarget (env_pyroki) → g1_shadow/<class>/<clip>/0/trajectory_pyroki.npz
#      *** HARD-REQUIRED *** — seeds the RSI reference pose and is step 4's input.
#   4. Wrist FK sidecar (env_isaaclab, pinocchio) → g1_shadow/<class>/<clip>/0/wrist_ref.npz
#      *** HARD-REQUIRED *** — the floating-hand root pose. Runs AFTER step 3.
#   5. Train → agent.pt
#   6. Deterministic rollout export → g1_shadow/<class>/<clip>/0/hand_rl.npz  (stage-2 target)
#   7. Reachability audit of the exported wrist trajectory (report only, never fails the clip).
#
# Checkpoints/metrics tree (evaluate.bash-compatible; clip_name at path parts[1]):
#   data/processed/parahome/g1_shadow_hand_pretrain/<clip_class>/<clip_name>/0/
#
# ONE-TIME PREREQS (not per-clip):
#   - composite PyRoki URDF:  source/robotis_sh5/data/robots/G1/urdf_pyroki/g1_shadow_nomimic.urdf
#     built by: <env_isaaclab python> scripts/process_dataset/assets/export_g1_shadow_urdf.py
#   - floating-hand USDs:     source/robotis_sh5/data/robots/G1/shadow_float_{l,r}.usd
#     built by: <env_isaaclab python> scripts/process_dataset/assets/build_shadow_floating_usd.py
#
# Which clips run: edit CLIPS=(...) or CLIPS_OVERRIDE="a b c".
# Env vars: FORCE=1 (re-run all), SKIP_RETARGET=1, SKIP_EXPORT=1, SKIP_AUDIT=1,
#   CLIP_CLASS=..., CLIPS_OVERRIDE="a b c", NUM_ENVS / TIMESTEPS, VIDEO=1,
#   PY=<env_isaaclab python>, PY_PYROKI=<env_pyroki python>.
# =============================================================================
set -euo pipefail

# ── User configuration ────────────────────────────────────────────────────────
TASK="Robotis-G1-Shadow-HandPretrain-Direct-v0"
CLIP_CLASS="${CLIP_CLASS:-single_rigid}"
NUM_ENVS="${NUM_ENVS:-2048}"
TIMESTEPS="${TIMESTEPS:-41000}"
PY="${PY:-/home/peunsu/anaconda3/envs/env_isaaclab/bin/python}"
PY_PYROKI="${PY_PYROKI:-/home/peunsu/anaconda3/envs/env_pyroki/bin/python}"
SKIP_RETARGET="${SKIP_RETARGET:-0}"   # 1 → skip steps 2-3 (env then fails at step 4: no wrist_ref input)
SKIP_EXPORT="${SKIP_EXPORT:-0}"       # 1 → train only, no stage-2 target
SKIP_AUDIT="${SKIP_AUDIT:-0}"         # 1 → no reachability report

# Set VIDEO=1 to record a training mp4 every VIDEO_INTERVAL steps.
VIDEO="${VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"

# Same 13 clips as train_sequences_sonic_residual.sh, deliberately: stage 1's output feeds stage 2,
# so the two stages must cover the same set or the comparison is between different data.
# (Selection criteria are documented in that script — standing / actually grasped / manipulated
# >0.10 m / in place / 120-320 frames / one subject per object.)
CLIPS=(
    # "s100_seg00_pan"
    # "s101_seg12_knife"
    # "s101_seg29_pot"
    # "s101_seg30_bowl"
    "s66_seg26_pan"
    "s53_seg19_knife"
    "s152_seg21_pot"
    "s71_seg27_bowl"
    # "s207_seg06_kettle"
    # "s55_seg31_knife"
    # "s73_seg31_pot"
    # "s33_seg18_bowl"
    # "s127_seg29_pan"
    # "s100_seg02_kettle"
    # "s100_seg03_cup"
    # "s101_seg18_potlid"
    # "s101_seg23_salt"      # ← uncomment / add more clips here
)
[[ -n "${CLIPS_OVERRIDE:-}" ]] && read -ra CLIPS <<< "${CLIPS_OVERRIDE}"

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_BASE="${PROJECT_DIR}/source/robotis_sh5/data/processed/parahome"
CHECKPOINT_BASE="${DATA_BASE}/g1_shadow_hand_pretrain/${CLIP_CLASS}"
LOG_BASE="${PROJECT_DIR}/logs/skrl/g1_shadow_hand_pretrain"
ROBOT_DIR="${PROJECT_DIR}/source/robotis_sh5/data/robots/G1"
URDF="${ROBOT_DIR}/urdf_pyroki/g1_shadow_nomimic.urdf"
FORCE="${FORCE:-0}"

# newest agent_*.pt in a log tree, created after a sentinel file (fallback: newest overall).
#
# NOTE the `xargs -r`. The other benchmark scripts use a bare `xargs ls -t`, and when `find`
# matches nothing xargs still runs `ls -t` with NO arguments — which lists the CURRENT DIRECTORY
# and returns its newest entry (measured: "wandb"). The caller's `[[ -z ... ]]` guard does not
# catch that, so the script goes on to `cp wandb agent.pt` and dies with a confusing
# "omitting directory" error, or worse copies a real-but-wrong file. `-r` suppresses the run on
# empty input; the `[[ -f ]]` check below is the belt-and-braces. This bites whenever training
# ends before the first checkpoint_interval (2000 steps) — i.e. every short/crashed run.
find_ckpt() {  # $1=log_tree $2=sentinel
    local c
    c=$(find "$1" -name "agent_*.pt" -newer "$2" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1 || true)
    [[ -z "${c}" ]] && c=$(find "$1" -name "agent_*.pt" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1 || true)
    [[ -z "${c}" ]] && c=$(find "$1" -name "best_agent.pt" -newer "$2" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1 || true)
    [[ -f "${c}" ]] || c=""
    echo "${c}"
}

# ── One-time prereq checks ────────────────────────────────────────────────────
if [[ ! -f "${URDF}" ]]; then
    echo "[hand 1] WARN: PyRoki URDF missing: ${URDF}"
    echo "         Build once (env_isaaclab): python scripts/process_dataset/assets/export_g1_shadow_urdf.py"
    echo "         Steps 3-4 will fail without it."
fi
for side in l r; do
    if [[ ! -f "${ROBOT_DIR}/shadow_float_${side}.usd" ]]; then
        echo "[hand 1] ERROR: floating-hand USD missing: ${ROBOT_DIR}/shadow_float_${side}.usd"
        echo "         Build once (env_isaaclab): python scripts/process_dataset/assets/build_shadow_floating_usd.py"
        exit 1
    fi
done

# ── Main loop ─────────────────────────────────────────────────────────────────
TOTAL="${#CLIPS[@]}"
IDX=0
for clip in "${CLIPS[@]}"; do
    IDX=$(( IDX + 1 ))
    CKPT_DIR="${CHECKPOINT_BASE}/${clip}/0"
    CKPT_FILE="${CKPT_DIR}/agent.pt"
    G1_DIR="${DATA_BASE}/g1_shadow/${CLIP_CLASS}/${clip}/0"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[hand ${IDX}/${TOTAL}] class=${CLIP_CLASS}  clip=${clip}"
    echo "  stage 1: floating bimanual Shadow hands, no body, no SONIC"
    echo "  train=${TIMESTEPS} steps (${NUM_ENVS} envs)  from scratch (reference-seeded RSI + state cache)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ -f "${CKPT_FILE}" && "${FORCE}" -eq 0 ]]; then
        echo "[hand] Checkpoint exists — skipping.  (FORCE=1 to override)  ${CKPT_FILE}"
        continue
    fi
    mkdir -p "${CKPT_DIR}"
    cd "${PROJECT_DIR}"

    # ── Step 1: clip data check ───────────────────────────────────────────────
    CLIP_NPZ="${DATA_BASE}/smplx/${CLIP_CLASS}/${clip}/0/trajectory.npz"
    if [[ ! -f "${CLIP_NPZ}" ]]; then
        echo "[hand] ERROR: processed clip not found — run parahome.py first."
        echo "        Expected: ${CLIP_NPZ}"
        continue
    fi

    # ── Step 2: hand-mesh contact map → hand_contact.npz ─────────────────────
    #   Feeds BOTH the retarget (step 3) and the env's contact_force / contact_cws rewards.
    #   Without it those rewards go inert — the clip still trains, but on tracking alone.
    HANDC_NPZ="${DATA_BASE}/smplx/${CLIP_CLASS}/${clip}/0/hand_contact.npz"
    if [[ "${SKIP_RETARGET}" -eq 1 ]]; then
        echo "[hand] Step 2/7 — hand contact SKIPPED (SKIP_RETARGET=1)."
    elif [[ -f "${HANDC_NPZ}" && "${FORCE}" -eq 0 ]]; then
        echo "[hand] Step 2/7 — hand contact exists — skipping.  ${HANDC_NPZ}"
    else
        echo "[hand] Step 2/7 — hand-mesh contact map (env_isaaclab): ${clip}"
        "${PY}" scripts/process_dataset/dataset/parahome_hand_contact.py \
            --class "${CLIP_CLASS}" --clip "${clip}" || \
            echo "[hand] WARN: hand contact failed — contact/CWS rewards will be inert for this clip."
    fi

    # ── Step 3: PyRoki retarget (env_pyroki) → trajectory_pyroki.npz ─────────
    #   HARD-REQUIRED: supplies the RSI reference joint pose (_ref_joints) and is step 4's input.
    RETARGET_NPZ="${G1_DIR}/trajectory_pyroki.npz"
    if [[ "${SKIP_RETARGET}" -eq 1 ]]; then
        echo "[hand] Step 3/7 — retarget SKIPPED (SKIP_RETARGET=1)."
    elif [[ -f "${RETARGET_NPZ}" && "${FORCE}" -eq 0 ]]; then
        echo "[hand] Step 3/7 — PyRoki retarget exists — skipping.  ${RETARGET_NPZ}"
    else
        echo "[hand] Step 3/7 — PyRoki retarget (env_pyroki): ${clip}"
        "${PY_PYROKI}" scripts/process_dataset/retarget/retarget_g1_pyroki.py \
            --class "${CLIP_CLASS}" --clip "${clip}" || \
            echo "[hand] WARN: PyRoki retarget failed."
    fi
    if [[ ! -f "${RETARGET_NPZ}" ]]; then
        echo "[hand] ERROR: trajectory_pyroki.npz missing — step 4 has no input; skipping clip."
        continue
    fi

    # ── Step 4: wrist FK sidecar (env_isaaclab, pinocchio) → wrist_ref.npz ───
    #   HARD-REQUIRED: the env raises FileNotFoundError without it (the floating hands are rooted
    #   at robot0_{l,r}_wrist and the reset writes that pose). CPU-only. Runs AFTER step 3.
    WRIST_NPZ="${G1_DIR}/wrist_ref.npz"
    if [[ -f "${WRIST_NPZ}" && "${FORCE}" -eq 0 ]]; then
        echo "[hand] Step 4/7 — wrist_ref exists — skipping.  ${WRIST_NPZ}"
    else
        echo "[hand] Step 4/7 — wrist FK sidecar (env_isaaclab): ${clip}"
        "${PY}" scripts/process_dataset/retarget/export_wrist_ref.py \
            --class "${CLIP_CLASS}" --clip "${clip}" --overwrite
    fi
    if [[ ! -f "${WRIST_NPZ}" ]]; then
        echo "[hand] ERROR: wrist_ref.npz missing after export — the env requires it; skipping clip."
        continue
    fi

    VIDEO_ARGS=()
    if [[ "${VIDEO}" -eq 1 ]]; then
        VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}" --video_interval "${VIDEO_INTERVAL}")
    fi

    # ── Step 5: Train ────────────────────────────────────────────────────────
    #   From scratch (no --checkpoint). RSI start frames span the whole clip from step 0 (every frame
    #   is restorable from the retarget reference) and the 176-D hand state cache fills as training
    #   runs — watch Curriculum/cache_coverage rising and Diag/cache_reject_frac staying at 0.
    echo "[hand] Step 5/7 — Training (${TIMESTEPS} steps, from scratch) ..."
    touch "${CKPT_DIR}/.sentinel"
    "${PY}" scripts/skrl/train.py \
        --task "${TASK}" --num_envs "${NUM_ENVS}" \
        --timesteps "${TIMESTEPS}" --headless "${VIDEO_ARGS[@]}" \
        --clip_class "${CLIP_CLASS}" --clip_name "${clip}"

    LATEST_CKPT=$(find_ckpt "${LOG_BASE}" "${CKPT_DIR}/.sentinel")
    if [[ -z "${LATEST_CKPT}" ]]; then
        echo "[hand] ERROR: train checkpoint not found in ${LOG_BASE}."
        echo "        skrl writes checkpoints every agent yaml 'checkpoint_interval' steps"
        echo "        (currently 2000), so a run shorter than that produces none — raise"
        echo "        TIMESTEPS or lower checkpoint_interval. Steps 6-7 need a checkpoint."
        continue
    fi
    cp "${LATEST_CKPT}" "${CKPT_FILE}"
    echo "[hand] Checkpoint → ${CKPT_FILE}"

    # ── Step 6: deterministic rollout export → hand_rl.npz (stage-2 target) ──
    #   THE deliverable of stage 1. One deterministic rollout (policy mean, not a sample) so the
    #   stage-2 target does not depend on the RNG. Writes wrist pose + finger joints WITH their
    #   joint names, so stage 2 matches by name rather than by column position.
    HANDRL_NPZ="${G1_DIR}/hand_rl.npz"
    if [[ "${SKIP_EXPORT}" -eq 1 ]]; then
        echo "[hand] Step 6/7 — rollout export SKIPPED (SKIP_EXPORT=1; stage 2 gets no target)."
    else
        echo "[hand] Step 6/7 — deterministic rollout export: ${clip}"
        "${PY}" scripts/process_dataset/retarget/export_hand_rl.py \
            --checkpoint "${CKPT_FILE}" --class "${CLIP_CLASS}" --clip "${clip}" --overwrite || \
            echo "[hand] WARN: rollout export failed — stage 2 has no target for this clip."
        [[ -f "${HANDRL_NPZ}" ]] && echo "[hand] Stage-2 target → ${HANDRL_NPZ}"
    fi

    # ── Step 7: reachability audit (report only) ─────────────────────────────
    #   Can a G1 arm reach the wrist trajectory stage 1 produced? The wrist floated with no anchor,
    #   so this is the gate that replaces anchoring. Retargeted baseline = 0.0% unreachable on all
    #   16 clips, so anything above that came from stage 1. Never fails the clip — it is a report.
    if [[ "${SKIP_AUDIT}" -eq 1 ]]; then
        echo "[hand] Step 7/7 — reachability audit SKIPPED (SKIP_AUDIT=1)."
    elif [[ ! -f "${HANDRL_NPZ}" ]]; then
        echo "[hand] Step 7/7 — reachability audit skipped: no hand_rl.npz."
    else
        echo "[hand] Step 7/7 — reachability audit (env_isaaclab, CPU): ${clip}"
        "${PY}" scripts/process_dataset/retarget/audit_wrist_reachability.py \
            --class "${CLIP_CLASS}" --clip "${clip}" --source rl \
            | tee "${CKPT_DIR}/reach_audit_rl.txt" || \
            echo "[hand] WARN: reachability audit failed."
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[hand] All ${TOTAL} clips processed."
echo "  checkpoints : ${CHECKPOINT_BASE}/"
echo "  stage-2 targets : ${DATA_BASE}/g1_shadow/${CLIP_CLASS}/<clip>/0/hand_rl.npz"
echo "  tensorboard : ${LOG_BASE}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
