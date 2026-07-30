"""Estimate object mass for each OakInk object using the Claude Vision API.

For each object, this script:
  1. Finds representative RGB images from the stream_release_v2 directory.
  2. Sends them to the Claude API with a physics-reasoning prompt.
  3. Parses the "Answer: min - kg, max - kg" response.
  4. Saves the results to data/processed/oakink/object_mass.json.

Usage:
    python scripts/process_dataset/estimate_object_mass.py

Requires:
    pip install anthropic

Output:
    source/robotis_sh5/data/processed/oakink/object_mass.json
    Format: {"A01001": [min_kg, max_kg], ...}
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent.parent
_DATA_DIR = _PROJECT_DIR / "source" / "robotis_sh5" / "data"
_STREAM_DIR = _DATA_DIR / "raw" / "oakink" / "image" / "stream_release_v2"
_ASSETS_DIR = _DATA_DIR / "processed" / "oakink" / "assets" / "objects"
_OUTPUT_JSON = _DATA_DIR / "processed" / "oakink" / "object_mass.json"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = "You are a physics and engineering assistant."

_USER_PROMPT = (
    "Estimate the weight of the object from the images. "
    "Reason step by step and finally state your answer in kilograms "
    "like Answer: min - kg, max - kg"
)

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

_CAMERAS = ["north_east_color", "north_west_color", "south_east_color", "south_west_color"]
_MAX_IMAGES_PER_OBJECT = 4   # one per camera direction


def _find_stream_images(object_id: str, n_images: int = _MAX_IMAGES_PER_OBJECT) -> list[Path]:
    """Return up to n_images PNG files from the first available sequence for object_id."""
    if not _STREAM_DIR.exists():
        return []

    # Find any sequence directory belonging to this object
    seq_dirs = sorted(_STREAM_DIR.glob(f"{object_id}_*"))
    if not seq_dirs:
        return []

    seq_dir = seq_dirs[0]
    ts_dirs = sorted(seq_dir.iterdir())
    if not ts_dirs:
        return []

    ts_dir = ts_dirs[0]
    images: list[Path] = []

    # Pick one mid-sequence frame from each camera direction
    for cam in _CAMERAS:
        frames = sorted(ts_dir.glob(f"{cam}_*.png"))
        if not frames:
            continue
        mid = frames[len(frames) // 2]
        images.append(mid)
        if len(images) >= n_images:
            break

    return images


def _find_texture_image(object_id: str) -> Path | None:
    """Fallback: return material_0.png texture for object_id."""
    p = _ASSETS_DIR / object_id / "material_0.png"
    return p if p.exists() else None


def _encode_image(path: Path) -> dict:
    """Encode image as base64 for the Anthropic messages API."""
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(
    r"Answer:\s*([\d.]+)\s*[-–]\s*kg\s*,\s*([\d.]+)\s*[-–]\s*kg",
    re.IGNORECASE,
)


def _parse_mass(text: str) -> tuple[float, float] | None:
    """Extract (min_kg, max_kg) from the model response. Returns None on parse failure."""
    m = _ANSWER_RE.search(text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    # Fallback: look for two numbers near "kg"
    numbers = re.findall(r"([\d]+\.[\d]+|[\d]+)\s*kg", text, re.IGNORECASE)
    if len(numbers) >= 2:
        lo, hi = float(numbers[0]), float(numbers[1])
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    if len(numbers) == 1:
        v = float(numbers[0])
        return v, v

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate OakInk object masses via Claude Vision API.")
    parser.add_argument("--model", default="claude-opus-4-7", help="Claude model ID to use.")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between API calls.")
    parser.add_argument("--output", default=str(_OUTPUT_JSON), help="Output JSON path.")
    parser.add_argument("--resume", action="store_true", help="Skip objects already in output JSON.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    results: dict[str, list[float]] = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"[resume] Loaded {len(results)} existing entries from {output_path}")

    client = anthropic.Anthropic()

    object_ids = sorted(d.name for d in _ASSETS_DIR.iterdir() if d.is_dir())
    print(f"[estimate] Found {len(object_ids)} objects in {_ASSETS_DIR}")

    for idx, object_id in enumerate(object_ids, 1):
        if args.resume and object_id in results:
            print(f"[{idx}/{len(object_ids)}] {object_id} — skip (already in output)")
            continue

        # Gather images
        images = _find_stream_images(object_id)
        if not images:
            tex = _find_texture_image(object_id)
            if tex:
                images = [tex]
        if not images:
            print(f"[{idx}/{len(object_ids)}] {object_id} — WARNING: no images found, using fallback [0.05, 0.50]")
            results[object_id] = [0.05, 0.50]
            continue

        print(f"[{idx}/{len(object_ids)}] {object_id} — {len(images)} image(s): {[p.name for p in images]}")

        content: list = [_encode_image(p) for p in images]
        content.append({"type": "text", "text": _USER_PROMPT})

        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            text = response.content[0].text
            print(f"  Response: {text[:200].replace(chr(10), ' ')}")

            mass = _parse_mass(text)
            if mass is None:
                print(f"  WARNING: Could not parse mass from response — using fallback [0.05, 0.50]")
                mass = (0.05, 0.50)

            results[object_id] = list(mass)
            print(f"  → mass: {mass[0]:.3f} – {mass[1]:.3f} kg")

        except anthropic.APIError as e:
            print(f"  ERROR: API call failed: {e}")
            results[object_id] = [0.05, 0.50]

        # Save after each object so partial results are preserved
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)

        if args.delay > 0:
            time.sleep(args.delay)

    print(f"\n[estimate] Done. Results written to {output_path} ({len(results)} objects)")


if __name__ == "__main__":
    main()
