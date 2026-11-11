"""2D slices through each tooth for image embeddings (PAPER_PLAN step 2, candidate 3; 2026-11-12).

For every tooth (tooth label + its pulp label) of a ToothFairy3 case: a local frame from the tooth's main
axis, oriented root -> crown (root end = the end with more jaw bone around it, as in tooth_features), and
three planes resampled from the image (linear interpolation, 0.25 mm pixels, 112 x 112 = 28 mm):
  0: along the axis (axis + first perpendicular), through the tooth centre, crown at the top;
  1: along the axis (axis + second perpendicular), same;
  2: across the tooth at crown height (80 % of the way from root tip to crown end).
Intensities are windowed per scan on the scan's own tissue scale: from below the pulp level to the enamel
level (or dentin + 1.5 x (dentin - pulp) when the scanner clips), so metal saturates and a tooth's contrast
does not depend on whether another tooth in the scan has metal (review 2026-11-12). Stored as uint8.
Truncated teeth are kept and flagged (points outside the volume are 0). Planes 0 and 1 follow the scan's
axes, not mesiodistal/buccolingual directions; the classifier therefore averages their embeddings.

Output (results/perception/<run_id>/): slices.npz with
  slices (n, 3, 112, 112) uint8, case_id (n,), fdi_mask (n,), fdi_report (n,), truncated (n,)
and provenance. Development cases only. No reports are read.

    python -m partialref.perception.tooth_slices --splits configs/splits/v1/splits.csv \\
        --tf3-dir /msc/home/ssingh54/sae_tooth/data/raw/toothfairy3 [--limit N] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from partialref.audit.tf3_link import tf3_paths
from partialref.perception.tooth_features import BONE, MIN_VOXELS, TEETH, mirror, scan_levels
from partialref.provenance import make_run_id, write_provenance

PIX_MM = 0.25
SIZE = 112
CROWN_AT = 0.80


def tooth_frame(mb: np.ndarray, L: np.ndarray, spacing: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Centre (mm, box coordinates), unit vectors (3 x 3: u = root->crown axis, v, w) and the projections of
    the root tip and crown end on u. Root end = the end with more bone (labels 1, 2) within 2 mm."""
    vox = np.argwhere(mb) * spacing
    mean = vox.mean(axis=0)
    _, _, vt = np.linalg.svd(vox - mean, full_matrices=False)
    ax = vt[0]
    proj = (vox - mean) @ ax
    dist_out = ndimage.distance_transform_edt(~mb, sampling=spacing)
    shell = (dist_out > 0) & (dist_out <= 2.0)
    sproj = (np.argwhere(shell) * spacing - mean) @ ax
    length = proj.max() - proj.min()
    sl = L[shell]
    lo, hi = sproj <= proj.min() + 0.25 * length, sproj >= proj.max() - 0.25 * length
    bone_lo = np.isin(sl[lo], BONE).mean() if lo.any() else 0.0
    bone_hi = np.isin(sl[hi], BONE).mean() if hi.any() else 0.0
    u = ax if bone_lo >= bone_hi else -ax                      # points from root to crown
    p = (vox - mean) @ u
    helper = np.eye(3)[int(np.argmin(np.abs(u)))]
    v = helper - (helper @ u) * u
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    return mean, np.stack([u, v, w]), float(p.min()), float(p.max()), float(abs(bone_lo - bone_hi))


def sample_plane(I: np.ndarray, origin_mm: np.ndarray, e1: np.ndarray, e2: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """SIZE x SIZE plane centred on origin_mm; rows run along -e1 (so +e1 is at the top), columns along e2."""
    r = (np.arange(SIZE) - (SIZE - 1) / 2) * PIX_MM
    a, b = np.meshgrid(-r, r, indexing="ij")
    pts = origin_mm[None, None, :] + a[..., None] * e1 + b[..., None] * e2
    coords = (pts / spacing).transpose(2, 0, 1)
    return ndimage.map_coordinates(I, coords, order=1, mode="constant", cval=np.nan)


def window(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    z = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.nan_to_num(z * 255, nan=0).astype(np.uint8)


def case_slices(case_id: str, tf3_dir: str) -> dict:
    try:
        img_p, lab_p = tf3_paths(case_id, Path(tf3_dir))
        I = np.asarray(nib.load(str(img_p)).dataobj, dtype=np.float32)
        lab = nib.load(str(lab_p))
        L = np.asarray(lab.dataobj).astype(np.int16)
        if I.shape != L.shape:
            return {"case_id": case_id, "error": "shape_mismatch", "slices": [], "fdi_mask": [], "truncated": []}
        spacing = np.asarray(lab.header.get_zooms()[:3], dtype=float)
        scan_max = float(I.max())
        lv = scan_levels(I[np.isin(L, TEETH + [100 + t for t in TEETH])], scan_max)
        low, dentin = lv["lvl_low"], lv["lvl_dentin"]
        ok = np.isfinite(low) and np.isfinite(dentin) and dentin > low
        lo = low - 0.5 * (dentin - low) if ok else float(np.percentile(I, 1))
        enamel = lv.get("lvl_enamel", np.nan)
        hi = float(enamel) if np.isfinite(enamel) else (dentin + 1.5 * (dentin - low) if ok else 0.99 * scan_max)
        merged = np.where((L >= 111) & (L <= 148), L - 100, L)
        objs = ndimage.find_objects(np.where(np.isin(merged, TEETH), merged, 0))
        out = {"slices": [], "fdi_mask": [], "truncated": []}
        for fdi in TEETH:
            if fdi - 1 >= len(objs) or objs[fdi - 1] is None:
                continue
            bb = objs[fdi - 1]
            truncated = any(s.start == 0 or s.stop == n for s, n in zip(bb, L.shape))
            pad = int(np.ceil(3.0 / spacing.min())) + 2               # same margin as tooth_features
            box = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(bb, L.shape))
            Lb = L[box]
            mb = (Lb == fdi) | (Lb == 100 + fdi)
            if mb.sum() < MIN_VOXELS:
                continue
            centre_box, (u, v, w), pmin, pmax, _ = tooth_frame(mb, Lb, spacing)
            offset = np.array([s.start for s in box]) * spacing
            centre = centre_box + offset
            crown_pt = centre + (pmin + CROWN_AT * (pmax - pmin)) * u
            planes = [sample_plane(I, centre, u, v, spacing), sample_plane(I, centre, u, w, spacing),
                      sample_plane(I, crown_pt, v, w, spacing)]
            out["slices"].append(np.stack([window(p, lo, hi) for p in planes]))
            out["fdi_mask"].append(fdi)
            out["truncated"].append(truncated)
        return {"case_id": case_id, "error": None, **out}
    except Exception as e:                                                   # noqa: BLE001
        return {"case_id": case_id, "error": f"{type(e).__name__}: {e}"[:200], "slices": [], "fdi_mask": [], "truncated": []}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--tf3-dir", required=True, type=Path)
    ap.add_argument("--role", default="dev", choices=["dev"])
    ap.add_argument("--exclude", default="S0015,P116,P117,P260,P261")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="tooth-slices")
    a = ap.parse_args(argv)
    sp = pd.read_csv(a.splits)
    cases = sorted(sp.loc[(sp["role"] == a.role) & ~sp["case_id"].isin(a.exclude.split(",")), "case_id"])
    if a.limit:
        cases = cases[: a.limit]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(case_slices, cases, [str(a.tf3_dir)] * len(cases)))
    S, cid, fm, tr = [], [], [], []
    for r in res:
        S += r["slices"]
        cid += [r["case_id"]] * len(r["slices"])
        fm += r["fdi_mask"]
        tr += r["truncated"]
    run_id = make_run_id(a.tag + (f"-limit{a.limit}" if a.limit else ""))
    out = a.results_dir / "perception" / run_id
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "slices.npz", slices=np.stack(S) if S else np.zeros((0, 3, SIZE, SIZE), np.uint8),
                        case_id=np.array(cid), fdi_mask=np.array(fm), fdi_report=np.array([mirror(f) for f in fm]),
                        truncated=np.array(tr))
    summ = {"run_id": run_id, "n_cases": len(cases), "n_teeth": len(S),
            "case_errors": {r["case_id"]: r["error"] for r in res if r["error"]}, "pix_mm": PIX_MM, "size": SIZE}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    write_provenance(out, extra={**summ, "splits": str(a.splits), "tf3_dir": str(a.tf3_dir), "role": a.role,
                                 "excluded": a.exclude})
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
