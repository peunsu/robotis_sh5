#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Compose two rollout mp4s (two camera angles of ONE clip) into a labelled side-by-side mp4.

CPU ONLY -- no Isaac Sim, no GPU. Safe to run while a training job owns the GPU, and rerunnable
without re-simulating. Encoding goes through imageio-ffmpeg's bundled ffmpeg; that binary has
`hstack` but NOT `drawtext` (verified), which is why the labels are burned in with PIL.

Also runs the checks that decide whether the composite is trustworthy:
  * PHYSICS RECEIPT -- the two passes' metrics CSVs must agree. NOT byte-equality: the GPU solver is
    not reproducible across processes, so byte-equality can never hold. Measured on s100_seg00_pan /
    seed 42, two byte-identical invocations with the SAME camera differ by up to 1.3e-1 relative at
    32 envs (1.2e-3 at 1 env), while the actual camera-change pair differed by 6.8e-2 -- i.e. LESS
    than the same-camera floor, so the camera angle demonstrably contributes nothing. The check is
    therefore: discrete success flags identical AND continuous errors within that measured floor.
  * frame counts equal -- a mismatch means the two passes recorded a different number of steps;
  * the two sides have the SAME number of leading near-black frames. The first env.render() returns
    the zeros fallback (isaaclab/envs/direct_rl_env.py:495-496) because the RTX pipeline has not
    produced anything yet, so ONE leading black frame on BOTH sides is normal and harmless -- it
    shifts both panes identically. An UNEQUAL count is the real hazard: it offsets the panes by a
    control step while leaving both metrics CSVs byte-identical (those are computed from physics,
    not from frames). By default the common leading black frames are trimmed from both sides, which
    preserves sync exactly (same count removed from each) and keeps the composite from opening on
    black; --no_trim_black keeps the raw frame indexing of the two single-view mp4s.
  * the two sidecar JSONs report DIFFERENT resolved eye positions -- equal eyes mean --cam_preset
    silently did not apply (e.g. a task whose cfg has no viewer_* fields) and both mp4s are the
    same view.
Exit code is 0 unless the composite could not be written; problems are printed as [compose] ERROR.
"""
import argparse
import csv
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np

# Per-env numeric metric columns compared between the two passes (see _check_metrics).
_METRIC_COLS = ("e_t_cm", "e_r", "e_j_cm", "e_ft_cm", "reward_sum")
_FLAG_COLS = ("success", "success_t", "success_r", "success_j", "success_ft")
# Tolerance CALIBRATED against the measured same-camera noise floor, not guessed. On
# s100_seg00_pan / seed 42, two byte-identical invocations (SAME camera, so any difference is pure
# PhysX-GPU non-reproducibility) reach max 1.3e-1 relative at --n_rollouts 32 and 1.2e-3 at
# --n_rollouts 1. The camera-change pair reached 6.8e-2 — i.e. BELOW the same-camera floor, proving
# the camera angle contributes nothing. So anything under this bound is indistinguishable from
# re-running the identical command, and cannot be blamed on the second camera pass.
_METRIC_REL_TOL = 2.0e-1


def _check_metrics(d, left_name, right_name, tol):
    """Compare the two passes' metrics CSVs. Returns True if they agree within tolerance.

    NOT byte-equality: the GPU solver is not reproducible across processes, so byte-equality can
    never hold (measured — see _METRIC_REL_TOL). What must hold is (a) the discrete success flags
    agree exactly, since those are the eval semantics, and (b) the continuous per-env errors agree
    within the same-camera noise floor. Both together say "the two panes show the same rollout to
    within the precision the simulator itself offers".
    """
    pl, pr = os.path.join(d, left_name), os.path.join(d, right_name)
    if not (os.path.isfile(pl) and os.path.isfile(pr)):
        print(f"[compose] WARNING: metrics receipt skipped — {left_name} and/or {right_name} missing.")
        return True
    with open(pl) as f:
        a = list(csv.DictReader(f))
    with open(pr) as f:
        b = list(csv.DictReader(f))
    if len(a) != len(b):
        print(f"[compose] ERROR: metrics row count differs ({len(a)} vs {len(b)}) — the two passes "
              "did not run the same number of episodes.")
        return False
    if not a:
        print("[compose] WARNING: metrics receipt skipped — no rows.")
        return True
    bad_flags = [(i, k) for i, (ra, rb) in enumerate(zip(a, b)) for k in _FLAG_COLS
                 if k in ra and ra[k] != rb[k]]
    worst, wcol, wrow = 0.0, None, -1
    for i, (ra, rb) in enumerate(zip(a, b)):
        for k in _METRIC_COLS:
            if k not in ra:
                continue
            try:
                va, vb = float(ra[k]), float(rb[k])
            except ValueError:
                continue
            rel = abs(va - vb) / max(abs(va), 1e-12)
            if rel > worst:
                worst, wcol, wrow = rel, k, i
    ok = not bad_flags and worst <= tol
    print(f"[compose] metrics receipt: max relative divergence {worst:.3e} "
          f"(col {wcol}, row {wrow}) vs tolerance {tol:.1e}; success flags "
          f"{'AGREE' if not bad_flags else f'DIFFER in {len(bad_flags)} cell(s)'}")
    if ok:
        print("[compose]   → the two camera passes simulated the same rollout to within the "
              "simulator's own run-to-run noise. Panes are frame-synchronous.")
    else:
        if bad_flags:
            print(f"[compose] ERROR: success flags differ (e.g. row {bad_flags[0][0]} "
                  f"col {bad_flags[0][1]}) — the two passes are NOT the same rollout.")
        if worst > tol:
            print(f"[compose] ERROR: divergence {worst:.3e} exceeds the same-camera noise floor "
                  f"({tol:.1e}). The panes may drift apart; do not trust the composite.")
    return ok


def _meta(d, prefix):
    p = os.path.join(d, f"viewer_{prefix}.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="evaluation_ep_le_<N>/ directory")
    ap.add_argument("--left_prefix", default="rl-video")
    ap.add_argument("--right_prefix", default="rl-video-oldcam")
    ap.add_argument("--out_prefix", default="rl-video-sbs")
    # Labels default to "cam 1 (yaw <y> / elev <e>)" / "cam 2 (...)", with the ANGLES READ FROM THE
    # SIDECAR JSON rather than hardcoded, so they cannot go stale if a pass is run with an explicit
    # --viewer_yaw/--viewer_elev instead of --cam_preset. Pass these flags to override outright.
    ap.add_argument("--left_label", default=None, help="Override the left pane label entirely.")
    ap.add_argument("--right_label", default=None, help="Override the right pane label entirely.")
    ap.add_argument("--no_labels", action="store_true")
    ap.add_argument("--black_thresh", type=float, default=4.0,
                    help="mean pixel value below which a frame counts as 'black' (warm-up frame).")
    ap.add_argument("--no_trim_black", action="store_true",
                    help="Keep the leading renderer warm-up frame(s) instead of trimming the count "
                         "common to both sides. Preserves the frame indexing of the single-view mp4s.")
    ap.add_argument("--left_metrics", default="metrics.csv")
    ap.add_argument("--right_metrics", default="metrics_camold.csv")
    ap.add_argument("--metric_rel_tol", type=float, default=_METRIC_REL_TOL,
                    help="Max allowed relative divergence between the two passes' per-env metrics. "
                         f"Default {_METRIC_REL_TOL:.1e} = the MEASURED same-camera noise floor.")
    a = ap.parse_args()

    vdir = os.path.join(a.dir, "videos")
    lp = os.path.join(vdir, f"{a.left_prefix}-step-0.mp4")
    rp = os.path.join(vdir, f"{a.right_prefix}-step-0.mp4")
    op = os.path.join(vdir, f"{a.out_prefix}-step-0.mp4")
    for p in (lp, rp):
        if not os.path.isfile(p):
            print(f"[compose] ERROR: missing input {p}")
            return 1

    # ---- physics receipt: did the two passes simulate the same rollout? ----
    _check_metrics(a.dir, a.left_metrics, a.right_metrics, a.metric_rel_tol)

    # ---- sidecar cross-check: the two passes must have resolved DIFFERENT eyes ----
    ml, mr = _meta(a.dir, a.left_prefix), _meta(a.dir, a.right_prefix)
    if ml and mr:
        el, er = ml.get("eye_env_local"), mr.get("eye_env_local")
        print(f"[compose] left  eye={el} lookat={ml.get('lookat_env_local')} "
              f"(yaw={ml.get('viewer_yaw')} elev={ml.get('viewer_elev')} look_obj={ml.get('viewer_look_obj')})")
        print(f"[compose] right eye={er} lookat={mr.get('lookat_env_local')} "
              f"(yaw={mr.get('viewer_yaw')} elev={mr.get('viewer_elev')} look_obj={mr.get('viewer_look_obj')})")
        if el == er:
            print("[compose] ERROR: both passes resolved the SAME camera eye -- --cam_preset did not "
                  "apply. The side-by-side would show one view twice.")
        if ml.get("resolution") != mr.get("resolution"):
            print(f"[compose] ERROR: resolution mismatch {ml.get('resolution')} vs {mr.get('resolution')}.")
        if ml.get("seed") != mr.get("seed"):
            print(f"[compose] ERROR: seed mismatch {ml.get('seed')} vs {mr.get('seed')} -- the panes "
                  "cannot be frame-synchronous.")
    else:
        print("[compose] WARNING: viewer_*.json sidecar(s) not found -- skipping the pose cross-check.")

    rl, rr = imageio.get_reader(lp), imageio.get_reader(rp)
    fps = float(rl.get_meta_data().get("fps") or 30.0)
    fl = [np.asarray(f)[..., :3] for f in rl.iter_data()]
    fr = [np.asarray(f)[..., :3] for f in rr.iter_data()]
    rl.close()
    rr.close()
    print(f"[compose] frames: left={len(fl)} right={len(fr)}  fps={fps}")
    if len(fl) != len(fr):
        print("[compose] ERROR: frame counts differ -- output is truncated to the shorter input.")
    if not fl or not fr:
        print("[compose] ERROR: an input decoded to 0 frames.")
        return 1
    # ---- leading renderer warm-up frames: EQUAL counts are benign, UNEQUAL offsets the panes ----
    def _lead_black(frames):
        n = 0
        for f in frames:
            if float(f.mean()) < a.black_thresh:
                n += 1
            else:
                break
        return n

    bl, br = _lead_black(fl), _lead_black(fr)
    print(f"[compose] leading near-black frames: left={bl} right={br}")
    if bl != br:
        print(f"[compose] ERROR: UNEQUAL warm-up frames ({bl} vs {br}) -- the panes are offset by "
              f"{abs(bl - br)} control step(s). The metrics receipt cannot see this. Re-run, or "
              "trim manually before trusting the composite.")
    elif bl:
        if a.no_trim_black:
            print(f"[compose] the {bl} warm-up frame(s) are IDENTICAL on both sides (panes stay in "
                  "sync); keeping them (--no_trim_black).")
        else:
            print(f"[compose] trimming the {bl} leading warm-up frame(s) from BOTH sides -- equal "
                  "counts, so sync is preserved exactly. Use --no_trim_black to keep them.")
            fl, fr = fl[bl:], fr[bl:]
            if not fl or not fr:
                print("[compose] ERROR: trimming consumed every frame.")
                return 1
    if fl[0].shape != fr[0].shape:
        print(f"[compose] ERROR: frame shapes differ {fl[0].shape} vs {fr[0].shape}.")
        return 1

    # ---- pane labels: "cam N (yaw <y> / elev <e>)", angles taken from the sidecar when available ----
    def _label(explicit, n, meta):
        if explicit is not None:
            return explicit
        y = (meta or {}).get("viewer_yaw")
        e = (meta or {}).get("viewer_elev")
        if y is None or e is None:
            return f"cam {n}"
        return f"cam {n} (yaw {float(y):g} / elev {float(e):g})"

    left_label = _label(a.left_label, 1, ml)
    right_label = _label(a.right_label, 2, mr)
    print(f"[compose] labels: LEFT '{left_label}'  RIGHT '{right_label}'")

    draw = None
    if not a.no_labels:
        try:
            from PIL import Image, ImageDraw, ImageFont
            px = max(14, fl[0].shape[0] // 28)
            try:
                font = ImageFont.load_default(size=px)      # Pillow >= 10.1
            except TypeError:
                font = ImageFont.load_default()

            def draw(frame):                               # noqa: F811
                im = Image.fromarray(frame)
                d = ImageDraw.Draw(im)
                pw = im.width // 2
                for i, t in enumerate((left_label, right_label)):
                    x0 = i * pw
                    d.rectangle([x0, 0, x0 + pw - 1, px + 10], fill=(0, 0, 0))
                    d.text((x0 + 8, 4), t, fill=(255, 255, 255), font=font)
                return np.asarray(im, dtype=np.uint8)
        except Exception as exc:                           # noqa: BLE001
            print(f"[compose] WARNING: labels skipped ({exc}).")
            draw = None

    with imageio.get_writer(op, fps=fps, macro_block_size=1) as w:
        for i in range(min(len(fl), len(fr))):
            frame = np.hstack((fl[i], fr[i]))
            w.append_data(draw(frame) if draw else frame)
    h, wd = fl[0].shape[:2]
    print(f"[compose] wrote {op}  {2 * wd}x{h} @{fps} fps  "
          f"({min(len(fl), len(fr))} frames)  LEFT={a.left_prefix} RIGHT={a.right_prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
