#!/usr/bin/env python3
"""
uniform-32_selection.py

Implements the fixed 32-slice axial selection protocol (roadmap §4):

    z_i = z_min + (i + 0.5) / 32 * (z_max - z_min),   i = 0 ... 31

z_min/z_max are taken from the jaw-volume bounding box (derived from the
segmentation mask's non-background extent along the z axis). This is the
SAME selection rule used for every model/arm in the comparison -- do not
change it between models (roadmap explicit requirement).

For each case, writes:
    {out_dir}/{case_id}/{case_id}_slice{00..31}.png   -- 8-bit windowed axial slices
    {out_dir}/{case_id}/{case_id}_slices_meta.json     -- per-slice provenance record

Usage:
    python uniform-32_selection.py \
        --volume dataset/images/A004_0000.nii.gz \
        --mask   dataset/predictions/A004.nii.gz \
        --out-dir outputs/uniform32 \
        --case-id A004

NOTE: window-center/width below are placeholders (soft-tissue-ish CBCT window).
Verify against whatever HU window your existing create_panoramic.py /
create_tooth_axial.py scripts use, and pass --window-center/--window-width
to match -- this script does not currently share that constant.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image


def load_canonical(path):
    """Load a NIfTI volume and canonicalize orientation, consistent with the
    orientation-handling convention already used elsewhere in this project
    (nib.as_closest_canonical())."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    affine = img.affine
    spacing = img.header.get_zooms()[:3]
    return data, affine, spacing


def jaw_bounding_box_z(mask_data):
    """Return (z_min_idx, z_max_idx) voxel indices spanning all non-background
    mask labels, i.e. the jaw-volume axial extent."""
    nonzero = np.any(mask_data > 0, axis=(0, 1))
    z_indices = np.where(nonzero)[0]
    if z_indices.size == 0:
        raise ValueError("Mask contains no non-background voxels; cannot derive jaw bounding box.")
    return int(z_indices.min()), int(z_indices.max())


def voxel_index_from_z(z_phys, affine, z_shape):
    """Convert a physical z-coordinate (in the canonical RAS-ish space) back
    to a fractional voxel index along the volume's 3rd axis, by inverting
    the affine on the z-axis only (assumes approx. axis-aligned volumes,
    standard for CBCT after as_closest_canonical)."""
    inv_affine = np.linalg.inv(affine)
    # Solve for voxel z assuming x,y held at volume center (only z row matters
    # for an axis-aligned canonical volume).
    world_pt = np.array([0.0, 0.0, z_phys, 1.0])
    voxel_pt = inv_affine @ world_pt
    vz = voxel_pt[2]
    return float(np.clip(vz, 0, z_shape - 1))


def extract_slice(volume, z_frac, interp="linear"):
    """Extract an axial slice at fractional voxel index z_frac."""
    z_shape = volume.shape[2]
    if interp == "nearest":
        z_idx = int(round(z_frac))
        z_idx = int(np.clip(z_idx, 0, z_shape - 1))
        return volume[:, :, z_idx], z_idx
    elif interp == "linear":
        z0 = int(np.floor(z_frac))
        z1 = min(z0 + 1, z_shape - 1)
        frac = z_frac - z0
        slice_ = (1 - frac) * volume[:, :, z0] + frac * volume[:, :, z1]
        # nearest original index for provenance
        z_idx = z0 if frac < 0.5 else z1
        return slice_, z_idx
    else:
        raise ValueError(f"Unknown interpolation mode: {interp}")


def window_to_uint8(slice_2d, window_center, window_width):
    lo = window_center - window_width / 2
    hi = window_center + window_width / 2
    clipped = np.clip(slice_2d, lo, hi)
    scaled = (clipped - lo) / max(hi - lo, 1e-6) * 255.0
    return scaled.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volume", required=True, help="Path to CBCT volume (.nii.gz)")
    ap.add_argument("--mask", required=True, help="Path to segmentation mask (.nii.gz), used for jaw bounding box")
    ap.add_argument("--out-dir", required=True, help="Output root directory")
    ap.add_argument("--case-id", required=True, help="Case identifier, e.g. A004")
    ap.add_argument("--num-slices", type=int, default=32)
    ap.add_argument("--interp", choices=["nearest", "linear"], default="linear")
    ap.add_argument("--window-center", type=float, default=1000.0,
                     help="HU window center -- VERIFY against existing pipeline's window before real runs")
    ap.add_argument("--window-width", type=float, default=3000.0,
                     help="HU window width -- VERIFY against existing pipeline's window before real runs")
    args = ap.parse_args()

    volume, affine, spacing = load_canonical(args.volume)
    mask, mask_affine, mask_spacing = load_canonical(args.mask)

    if volume.shape != mask.shape:
        print(f"[WARN] volume shape {volume.shape} != mask shape {mask.shape}; "
              f"proceeding but bounding box may be misaligned.", file=sys.stderr)

    z_min_idx, z_max_idx = jaw_bounding_box_z(mask)

    # Convert voxel index bounds to physical z using the (canonical) affine,
    # then re-sample 32 uniform physical z positions per the roadmap formula.
    def voxel_to_phys_z(vz):
        pt = affine @ np.array([0.0, 0.0, vz, 1.0])
        return float(pt[2])

    z_min_phys = voxel_to_phys_z(z_min_idx)
    z_max_phys = voxel_to_phys_z(z_max_idx)

    case_out = Path(args.out_dir) / args.case_id
    case_out.mkdir(parents=True, exist_ok=True)

    records = []
    n = args.num_slices
    for i in range(n):
        z_phys = z_min_phys + (i + 0.5) / n * (z_max_phys - z_min_phys)
        z_frac = voxel_index_from_z(z_phys, affine, volume.shape[2])
        slice_2d, orig_voxel_idx = extract_slice(volume, z_frac, interp=args.interp)
        img_u8 = window_to_uint8(slice_2d, args.window_center, args.window_width)

        fname = f"{args.case_id}_slice{i:02d}.png"
        Image.fromarray(img_u8).save(case_out / fname)

        records.append({
            "slice_index": i,
            "filename": fname,
            "original_voxel_z_index": int(orig_voxel_idx),
            "physical_z": float(z_phys),
            "voxel_spacing": [float(s) for s in spacing],
            "orientation": "canonical_RAS",
            "interpolation": args.interp,
        })

    meta = {
        "case_id": args.case_id,
        "num_slices": n,
        "jaw_bbox_voxel_z": [int(z_min_idx), int(z_max_idx)],
        "jaw_bbox_physical_z": [float(z_min_phys), float(z_max_phys)],
        "window_center": float(args.window_center),
        "window_width": float(args.window_width),
        "slices": records,
    }
    meta_path = case_out / f"{args.case_id}_slices_meta.json"
    tmp_path = meta_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(meta, f, indent=2)
    tmp_path.replace(meta_path)  # atomic on the same filesystem -- a crash
                                  # during json.dump leaves only the .tmp file
                                  # corrupted, never the real meta_path

    print(f"[OK] {args.case_id}: wrote {n} slices + metadata to {case_out}")


if __name__ == "__main__":
    main()