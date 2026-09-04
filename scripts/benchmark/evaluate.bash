#!/usr/bin/env bash
# =============================================================================
# evaluate.bash — Aggregate rollout metrics.csv files into per-method CSVs.
#
# Compatible with workspace2/evaluation/evaluate.bash (identical column format).
#
# Usage:
#   bash scripts/benchmark/evaluate.bash <BASE_DIR>
#
# BASE_DIR must contain subdirectories representing different systems (models).
# Each model directory is expected to hold:
#   <model>/right/<trajectory_task>/<data_id>/evaluation_*/metrics.csv
#
# Output files are written INSIDE each model directory (2026-08-14; they used to land
# flat in BASE_DIR as <model>_methodN.csv, which buried the run directories under dozens
# of loose CSVs once several runs accumulated):
#   <model>/method1.csv  — ManipTrans thresholds (all 4 metrics)
#   <model>/method2.csv  — DexMachina thresholds (E_t < 10cm, E_r < 0.5 rad)
#   <model>/method3.csv  — Mean over ALL rollouts regardless of success
#
# Nothing in the repo READS these files (they are for humans), so the move is safe; the
# only references were doc lines and the "results written to" echoes, updated with it.
# Aggregates from earlier runs stay where they were — this only changes new writes.
# =============================================================================

BASE_DIR="${1:?Usage: $0 <BASE_DIR>}"
BASE_DIR="$(cd "${BASE_DIR}" && pwd)"

python3 - "${BASE_DIR}" << 'PYEOF'
import sys, os, csv, glob, math, statistics


def _mean_finite(rows, col):
    """유한한 값만으로 평균. (mean|None, n_used, n_skipped).

    롤아웃 중 물리 발산으로 로봇 상태가 NaN 이 되면 그 스텝의 키포인트/손끝 오차가 nan 으로
    기록되고, 에피소드 평균이 nan 이 되어 여기까지 올라온다 (물체 오차는 멀쩡한 채로).
    nan 을 그대로 넘기면 statistics.stdev 가 'float object has no attribute numerator' 로
    죽는다 — CPython 의 정확산술 경로가 비유한값에서 깨지면서 나는 불친절한 예외다.
    """
    vals = []
    skipped = 0
    for r in rows:
        try:
            v = float(r[col])
        except (TypeError, ValueError):
            skipped += 1
            continue
        if math.isfinite(v):
            vals.append(v)
        else:
            skipped += 1
    return (sum(vals) / len(vals) if vals else None), len(vals), skipped

BASE_DIR = sys.argv[1]

MODELS = sorted(
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
    and glob.glob(os.path.join(BASE_DIR, d, "**/metrics.csv"), recursive=True)
)

if not MODELS:
    print(f"[evaluate] No model directories with metrics.csv found in: {BASE_DIR}")
    sys.exit(0)

# ── Thresholds ────────────────────────────────────────────────────────────────
# Method 1 (ManipTrans)
M1_ET, M1_ER, M1_EJ, M1_EFT = 3.0, 30.0, 8.0, 6.0

# Method 2 (DexMachina): wrist position < 10 cm, rotation < 0.5 rad
M2_ET = 10.0
M2_ER = math.degrees(0.5)   # ≈ 28.648 deg


# ── Per-sequence aggregation ──────────────────────────────────────────────────

def compute_sequences(model_dir, success_fn):
    """Return per-sequence metrics; errors are mean over successful rollouts only."""
    pattern = os.path.join(model_dir, "**/metrics.csv")
    csv_files = sorted(glob.glob(pattern, recursive=True))

    results = []
    for fpath in csv_files:
        # Path layout: <model>/right/<trajectory_task>/<data_id>/evaluation_*/metrics.csv
        rel = os.path.relpath(fpath, model_dir)
        parts = rel.split(os.sep)
        task = parts[1]  # e.g. "A01005-0001-0000"

        with open(fpath, newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        data_id  = int(rows[0]["ref_start"])
        n_frames = int(rows[0]["n_frames"])
        n_total  = len(rows)

        successful = [r for r in rows if success_fn(r)]
        n_succ     = len(successful)
        seq_sr     = n_succ / n_total

        if n_succ > 0:
            et,  _, _ = _mean_finite(successful, "e_t_cm")
            er,  _, _ = _mean_finite(successful, "e_r")
            ej,  _, _ = _mean_finite(successful, "e_j_cm")
            eft, _, _ = _mean_finite(successful, "e_ft_cm")
        else:
            et = er = ej = eft = None

        results.append({
            "task":     task,
            "data_id":  data_id,
            "et_cm":    et,
            "er_deg":   er,
            "ej_cm":    ej,
            "eft_cm":   eft,
            "n_frames": n_frames,
            "success":  seq_sr,
        })

    return results


def compute_sequences_all(model_dir):
    """Method 3: mean error over ALL rollouts; SR still uses M1 thresholds."""
    pattern = os.path.join(model_dir, "**/metrics.csv")
    csv_files = sorted(glob.glob(pattern, recursive=True))

    results = []
    for fpath in csv_files:
        rel = os.path.relpath(fpath, model_dir)
        parts = rel.split(os.sep)
        task = parts[1]  # e.g. "A01005-0001-0000"

        with open(fpath, newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        data_id  = int(rows[0]["ref_start"])
        n_frames = int(rows[0]["n_frames"])
        n_total  = len(rows)

        n_succ = sum(1 for r in rows if m1_success(r))
        seq_sr = n_succ / n_total

        et,  _,  s_et  = _mean_finite(rows, "e_t_cm")
        er,  _,  s_er  = _mean_finite(rows, "e_r")
        ej,  _,  s_ej  = _mean_finite(rows, "e_j_cm")
        eft, _,  s_eft = _mean_finite(rows, "e_ft_cm")
        _sk = max(s_et, s_er, s_ej, s_eft)
        if _sk:
            print(f"[warn] {task}: 비유한(nan/inf) 롤아웃 {_sk}/{n_total} 건을 평균에서 제외 "
                  f"(성공률은 전체 {n_total} 건 기준 그대로). 롤아웃 중 물리 발산 의심.",
                  file=sys.stderr)

        results.append({
            "task":     task,
            "data_id":  data_id,
            "et_cm":    et,
            "er_deg":   er,
            "ej_cm":    ej,
            "eft_cm":   eft,
            "n_frames": n_frames,
            "success":  seq_sr,
        })

    return results


# ── CSV writer ────────────────────────────────────────────────────────────────

def write_result_csv(rows, out_path, et_thr, er_thr, ej_thr, eft_thr):
    n = len(rows)
    if n == 0:
        return

    valid   = [r for r in rows
               if all(r[k] is not None and math.isfinite(r[k])
                      for k in ("et_cm", "er_deg", "ej_cm", "eft_cm"))]
    n_valid = len(valid)

    if n_valid > 0:
        et_vals  = [r["et_cm"]  for r in valid]
        er_vals  = [r["er_deg"] for r in valid]
        ej_vals  = [r["ej_cm"]  for r in valid]
        eft_vals = [r["eft_cm"] for r in valid]

        mean_et  = statistics.mean(et_vals)
        mean_er  = statistics.mean(er_vals)
        mean_ej  = statistics.mean(ej_vals)
        mean_eft = statistics.mean(eft_vals)

        std_et   = statistics.stdev(et_vals)  if n_valid > 1 else 0.0
        std_er   = statistics.stdev(er_vals)  if n_valid > 1 else 0.0
        std_ej   = statistics.stdev(ej_vals)  if n_valid > 1 else 0.0
        std_eft  = statistics.stdev(eft_vals) if n_valid > 1 else 0.0

        n_et  = sum(1 for v in et_vals  if v < et_thr)
        n_er  = sum(1 for v in er_vals  if v < er_thr)
        n_ej  = sum(1 for v in ej_vals  if v < ej_thr)
        n_eft = sum(1 for v in eft_vals if v < eft_thr)

        sr_et  = f"{100 * n_et  / n_valid:.1f}%"
        sr_er  = f"{100 * n_er  / n_valid:.1f}%"
        sr_ej  = f"{100 * n_ej  / n_valid:.1f}%"
        sr_eft = f"{100 * n_eft / n_valid:.1f}%"

        mean_row = ["MEAN", "", f"{mean_et:.4f}", f"{mean_er:.4f}", f"{mean_ej:.4f}", f"{mean_eft:.4f}",
                    sum(r["n_frames"] for r in rows), ""]
        std_row  = ["STD",  "", f"{std_et:.4f}",  f"{std_er:.4f}",  f"{std_ej:.4f}",  f"{std_eft:.4f}",  "", ""]
    else:
        sr_et = sr_er = sr_ej = sr_eft = "0.0%"
        mean_row = ["MEAN", "", "", "", "", "", sum(r["n_frames"] for r in rows), ""]
        std_row  = ["STD",  "", "", "", "", "", "", ""]

    overall_sr = statistics.mean(r["success"] for r in rows)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "data_id", "et_cm", "er_deg", "ej_cm", "eft_cm", "n_frames", "success"])
        for r in rows:
            if r["et_cm"] is not None:
                et_s  = f"{r['et_cm']:.4f}"
                er_s  = f"{r['er_deg']:.4f}"
                ej_s  = f"{r['ej_cm']:.4f}"
                eft_s = f"{r['eft_cm']:.4f}"
            else:
                et_s = er_s = ej_s = eft_s = ""
            w.writerow([r["task"], r["data_id"], et_s, er_s, ej_s, eft_s,
                        r["n_frames"], f"{r['success']:.4f}"])
        w.writerow(mean_row)
        w.writerow(std_row)
        w.writerow(["SR", "", sr_et, sr_er, sr_ej, sr_eft, n, f"{100 * overall_sr:.1f}%"])

    print(f"  → {out_path}  ({n} seqs, SR={100*overall_sr:.1f}%)")


# ── Success predicates ────────────────────────────────────────────────────────

def m1_success(r):
    return (float(r["e_t_cm"])  < M1_ET and
            float(r["e_r"])     < M1_ER and
            float(r["e_j_cm"])  < M1_EJ and
            float(r["e_ft_cm"]) < M1_EFT)

def m2_success(r):
    return (float(r["e_t_cm"]) < M2_ET and
            float(r["e_r"])    < M2_ER)


# ── Main ─────────────────────────────────────────────────────────────────────

for model in MODELS:
    model_dir = os.path.join(BASE_DIR, model)

    rows_m1 = compute_sequences(model_dir, m1_success)
    rows_m2 = compute_sequences(model_dir, m2_success)
    rows_m3 = compute_sequences_all(model_dir)

    # 결과는 모델(런) 디렉터리 안에 씁니다. 이름에 모델명을 다시 붙이지 않는 이유: 폴더가 이미
    # 신원을 갖고 있고, 셋이 method1/2/3로 나란히 정렬돼 눈으로 훑기 좋기 때문입니다.
    out_m1 = os.path.join(model_dir, "method1.csv")
    out_m2 = os.path.join(model_dir, "method2.csv")
    out_m3 = os.path.join(model_dir, "method3.csv")

    sr_m1 = statistics.mean(r["success"] for r in rows_m1) * 100 if rows_m1 else 0.0
    sr_m2 = statistics.mean(r["success"] for r in rows_m2) * 100 if rows_m2 else 0.0
    sr_m3 = statistics.mean(r["success"] for r in rows_m3) * 100 if rows_m3 else 0.0
    n_seqs = len(rows_m1)

    print(f"\n[{model}]  {n_seqs} sequences")
    print(f"  M1(ManipTrans)  SR={sr_m1:.1f}%")
    print(f"  M2(DexMachina)  SR={sr_m2:.1f}%")
    print(f"  M3(All rollouts)SR={sr_m3:.1f}%")

    write_result_csv(rows_m1, out_m1, M1_ET, M1_ER, M1_EJ, M1_EFT)
    write_result_csv(rows_m2, out_m2, M2_ET, M2_ER, M1_EJ, M1_EFT)
    write_result_csv(rows_m3, out_m3, M1_ET, M1_ER, M1_EJ, M1_EFT)

PYEOF
