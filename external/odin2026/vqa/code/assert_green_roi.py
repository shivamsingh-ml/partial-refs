#!/usr/bin/env python3
"""
Step 0 gate: assert every *_sinus_*_detail.png under a directory contains at
least one pixel of the ROI outline color (pure green by default). Run this
after re-rendering with the fixed create_sinus_detail.py, before calling the
model on any of the images.

    python3 code/assert_green_roi.py outputs/sinus_probe/images

Exits non-zero and prints the offending files if any PNG has zero matching
pixels. A side that the renderer skipped for having <10 sinus voxels never
gets a PNG at all — that's a `no_image` case for scoring, not a gate
failure, so this script only checks files that exist; it does not require
every (case, side) pair to have produced one.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def count_color_pixels(png_path: Path, color: tuple, tolerance: int) -> int:
    """
    Count marker pixels by GREENNESS, not by exact colour match.

    create_sinus_detail.py draws the marker at native slice resolution
    (SINUS_MARKER_RGB = (0,255,0), a cross rather than an outline), and the
    tile then goes through _scale_to_width() with Image.LANCZOS. Resampling
    blends the stroke against the dark background, so on a thin stroke there
    may be no pixel left within a tight window of pure green even though the
    marker is plainly visible. A tolerance test around (0,255,0) would then
    fail a perfectly good render.

    "Strongly green" -- G high, R and B both low -- survives the resample and
    still cannot be produced by the grayscale slice (R==G==B) or by any tooth
    tag in TOOTH_COLORS.
    """
    img = Image.open(png_path).convert("RGB")
    arr = np.asarray(img).astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    target_r, target_g, target_b = color
    if (target_r, target_g, target_b) == (0, 255, 0):
        mask = (g > 200) & (r < 80) & (b < 80)
    else:  # non-default --color: fall back to the exact-match window
        lo = np.array(color) - tolerance
        hi = np.array(color) + tolerance
        mask = np.all((arr >= lo) & (arr <= hi), axis=-1)
    return int(mask.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images_dir", type=Path)
    ap.add_argument("--pattern", default="*_sinus_*_detail.png",
                     help="glob for sinus detail images (default: %(default)s)")
    ap.add_argument("--color", default="0,255,0",
                     help="target R,G,B for the ROI outline (default: pure green)")
    ap.add_argument("--tolerance", type=int, default=10,
                     help="per-channel tolerance around --color (default: 10)")
    args = ap.parse_args()

    color = tuple(int(c) for c in args.color.split(","))
    files = sorted(args.images_dir.glob(args.pattern))
    if not files:
        print(f"No files matched {args.pattern} under {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    failures = []
    for f in files:
        n = count_color_pixels(f, color, args.tolerance)
        status = "OK" if n > 0 else "MISSING GREEN ROI"
        print(f"{f.name}: {n} matching pixels — {status}")
        if n == 0:
            failures.append(f)

    print(f"\n{len(files)} files checked, {len(failures)} missing the ROI outline.")
    if failures:
        print("\nFailing files:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
