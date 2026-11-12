"""ToothFairy3 manual masks in the voxel grid and label scheme of V2V2R's segmentation output (2026-11-13).

Purpose: the screening check "what does V2V2R's VLM see per tooth" runs the submitted pipeline with the
dataset's manual masks instead of its own segmentation (PRECOMPUTED_MASK in external/odin2026/inference.py),
so the VLM and our per-tooth module work from the same outlines.

Grid: TF4 volumes are voxel-identical to the TF3 image after flipping the TF3 array on axes 1 and 2
(DECISIONS 2026-11-07, job 614578); this is re-checked per case on the image arrays. Output uses the TF4
volume's affine and header.

Labels (V2V2R's segmentation output, `task1_inference.BasePredictor.orig_mapping`): 1-10 as TF3 (jaws,
canals 3/4, sinuses 5/6, pharynx 7, bridge 8, crown 9, implant 10), teeth 11-48 in FDI, incisive canals
51/52 and lingual canal 53 (TF3 103/104/105), pulp 50 (all TF3 pulp labels 111-148 merged).

Left/right: V2V2R flips L-first volumes before segmentation and flips the mask back, so its tooth numbers may
be mirrored relative to TF3's naming (TF3 tooth labels are mirrored relative to the reports, DECISIONS
2026-11-08). `mirror=True` swaps all left/right labels (teeth quadrants 1<->2, 3<->4; canals 3<->4, 51<->52;
sinuses 5<->6). Which variant matches V2V2R is decided empirically with `dice`, against V2V2R's own
predicted mask of the same scans.

    python -m partialref.baseline.v2v2r_masks --cases P001,F006,S0001 --tf3-dir <TF3> --tf4-dir <TF4> \\
        --out-dir <dir> [--mirror both|yes|no] [--workers 4]
    python -m partialref.baseline.v2v2r_masks --dice <pred.nii.gz> <converted.nii.gz>
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from partialref.audit.tf3_link import tf3_paths
from partialref.perception.tooth_features import TEETH
from partialref.perception.tooth_features import mirror as mirror_fdi

KEEP = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
CANALS = {103: 51, 104: 52, 105: 53}
PULP = 50
LR_SWAP = {3: 4, 4: 3, 5: 6, 6: 5, 51: 52, 52: 51}
# Labels outside TF3's label list that are set to background, as V2V2R's segmentation training data did
# (U-Mamba2/scripts/convert_tf3_singlepulp.py: anything not in the mapping -> 0). 100: S0021, 13 pieces
# (169 mm^3) touching only the pharynx and background, no tooth (2026-11-12). Any other unknown label raises.
TO_BACKGROUND = (100,)


def convert_labels(L: np.ndarray, mirror: bool) -> np.ndarray:
    """TF3 label array -> V2V2R label scheme (same array layout). Raises on labels it does not know."""
    known = set(KEEP) | set(TEETH) | set(CANALS) | {100 + t for t in TEETH} | set(TO_BACKGROUND) | {0}
    unknown = sorted(set(np.unique(L).tolist()) - known)
    if unknown:
        raise ValueError(f"unexpected labels {unknown}")
    lut = np.zeros(max(known) + 1, np.uint8)
    for k in KEEP:
        lut[k] = LR_SWAP.get(k, k) if mirror else k
    for t in TEETH:
        lut[t] = mirror_fdi(t) if mirror else t
        lut[100 + t] = PULP
    for k, v in CANALS.items():
        lut[k] = LR_SWAP.get(v, v) if mirror else v
    return lut[L]


def to_tf4_grid(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flip(a, axis=(1, 2)))


def convert_case(case_id: str, tf3_dir: str, tf4_dir: str, out_dir: str, variants: tuple[bool, ...]) -> dict:
    try:
        img_p, lab_p = tf3_paths(case_id, Path(tf3_dir))
        tf4 = nib.load(str(Path(tf4_dir) / case_id / "cbct" / "volume.nii.gz"))
        I3 = np.asarray(nib.load(str(img_p)).dataobj)
        I4 = np.asarray(tf4.dataobj)
        if I3.shape != I4.shape or not np.array_equal(to_tf4_grid(I3), I4):
            return {"case_id": case_id, "error": "TF3 image (flipped y,z) is not voxel-identical to the TF4 volume"}
        L = np.asarray(nib.load(str(lab_p)).dataobj).astype(np.int16)
        counts = {}
        for m in variants:
            out = to_tf4_grid(convert_labels(L, m))
            d = Path(out_dir) / ("mirror" if m else "nomirror")
            d.mkdir(parents=True, exist_ok=True)
            hdr = tf4.header.copy()
            hdr.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(out, tf4.affine, hdr), str(d / f"{case_id}.nii.gz"))
            counts["mirror" if m else "nomirror"] = int((out > 0).sum())
        return {"case_id": case_id, "error": None, "voxels_labelled": counts}
    except Exception as e:                                                   # noqa: BLE001
        return {"case_id": case_id, "error": f"{type(e).__name__}: {e}"[:200]}


def dice(a: np.ndarray, b: np.ndarray, labels: list[int] | None = None) -> dict[int, float]:
    """Per-label Dice between two label arrays of the same shape (labels present in either)."""
    labels = labels or sorted((set(np.unique(a).tolist()) | set(np.unique(b).tolist())) - {0})
    out = {}
    for k in labels:
        x, y = a == k, b == k
        s = int(x.sum() + y.sum())
        out[int(k)] = 2 * int((x & y).sum()) / s if s else np.nan
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", help="comma-separated case ids")
    ap.add_argument("--tf3-dir", type=Path)
    ap.add_argument("--tf4-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--mirror", default="both", choices=["both", "yes", "no"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dice", nargs=2, type=Path, metavar=("PRED", "CONVERTED"),
                    help="print per-label Dice of two masks and the mean over teeth, then exit")
    a = ap.parse_args(argv)
    if a.dice:
        A = np.asarray(nib.load(str(a.dice[0])).dataobj)
        B = np.asarray(nib.load(str(a.dice[1])).dataobj)
        if A.shape != B.shape:
            raise SystemExit(f"shape mismatch {A.shape} vs {B.shape}")
        d = dice(A, B)
        teeth = [v for k, v in d.items() if k in TEETH and np.isfinite(v)]
        print(json.dumps({"mean_dice_teeth": float(np.mean(teeth)) if teeth else None,
                          "per_label": {k: round(v, 3) for k, v in d.items()}}))
        return 0
    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    variants = {"both": (False, True), "yes": (True,), "no": (False,)}[a.mirror]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(convert_case, cases, [str(a.tf3_dir)] * len(cases), [str(a.tf4_dir)] * len(cases),
                          [str(a.out_dir)] * len(cases), [variants] * len(cases)))
    (a.out_dir / "convert_summary.json").write_text(json.dumps(res, indent=2))
    bad = [r for r in res if r["error"]]
    print(json.dumps({"n_cases": len(res), "n_errors": len(bad), "errors": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
