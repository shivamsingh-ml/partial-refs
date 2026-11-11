"""Per-tooth intensity features from CBCT + tooth masks (PAPER_PLAN step 1, candidate 1; 2026-11-12).

For every tooth (FDI 11-48) in a ToothFairy3 label file, on the ToothFairy3 image grid (same grid as the
label): the tooth mask is the tooth label plus its own pulp label (100 + FDI; ToothFairy3 labels pulp
separately). The tooth is split into a crown part and a root part along its main axis, then measured:
- bright material (metal / ceramic) on the crown surface, inside the crown, in the root, in the pulp;
- prosthetic crown / bridge labels (9 / 8, which carry no tooth number) next to the crown end, and the
  implant label (10) next to the root end;
- dark voxels inside the crown (eroded, to avoid partial-volume edges);
- the bone around the root tip (1-3 mm from the tooth, within 4 mm of the apex point), relative to the
  jaw-bone median, as a low percentile and a dark share.
Bright = >= 0.97 x the scan maximum (metal saturates at the maximum); the number of saturated voxels in the
scan and the tooth intensity are recorded so a scan without metal can be recognised.

Crown end vs root end: the end whose 2 mm surroundings contain more jaw bone (labels 1, 2) is the root
end; end_bone_contrast near 0 means the orientation is uncertain (e.g. impacted teeth fully in bone).
Teeth whose bounding box touches the volume border are flagged (truncated).

Tooth numbers: ToothFairy3 tooth labels are mirrored left/right relative to the reports (DECISIONS);
`fdi_report` is the mirrored number. No reports are read here. Development cases only.

    python -m partialref.perception.tooth_features --splits configs/splits/v1/splits.csv \\
        --tf3-dir /msc/home/ssingh54/sae_tooth/data/raw/toothfairy3 [--limit N] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from partialref.audit.tf3_link import tf3_paths
from partialref.provenance import make_run_id, write_provenance

TEETH = [q * 10 + i for q in (1, 2, 3, 4) for i in range(1, 9)]
MIRROR_Q = {1: 2, 2: 1, 3: 4, 4: 3}
BONE = (1, 2)
PROSTHETIC = (8, 9)           # bridge, crown
IMPLANT = 10
BRIGHT_FRAC_OF_MAX = 0.97
MIN_VOXELS = 200
CROWN_SHARE = 0.35
APEX_SHARE = 0.05             # apex point = mean of the outer 5 % of the tooth on the root side
SHELL_MM = (1.0, 3.0)         # bone voxels 1-3 mm from the tooth surface ...
APEX_RADIUS_MM = 4.0          # ... and within 4 mm of the apex point


def mirror(fdi: int) -> int:
    return MIRROR_Q[fdi // 10] * 10 + fdi % 10


def tooth_axis(coords_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = coords_mm.mean(axis=0)
    c = coords_mm - mean
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[0], c @ vt[0], mean


def tooth_row(I: np.ndarray, L: np.ndarray, fdi: int, spacing: np.ndarray, scan_max: float,
              bone_med: float, bbox: tuple[slice, ...] | None = None) -> dict | None:
    """bbox: optional tight slices around tooth + pulp (from ndimage.find_objects)."""
    if bbox is None:
        sel = np.isin(L, (fdi, 100 + fdi))
        if sel.sum() < MIN_VOXELS:
            return None
        idx = np.argwhere(sel)
        bbox = tuple(slice(a, b + 1) for a, b in zip(idx.min(0), idx.max(0)))
    pad = int(np.ceil(SHELL_MM[1] / spacing.min())) + 2
    lo = [max(s.start - pad, 0) for s in bbox]
    hi = [min(s.stop + pad, n) for s, n in zip(bbox, L.shape)]
    truncated = any(s.start == 0 or s.stop == n for s, n in zip(bbox, L.shape))
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    Lb, Ib = L[sl], I[sl].astype(np.float32)
    mb = (Lb == fdi) | (Lb == 100 + fdi)
    n = int(mb.sum())
    if n < MIN_VOXELS:
        return None
    pulp = (Lb == 100 + fdi)[mb]
    vox = np.argwhere(mb) * spacing
    ax, proj, mean = tooth_axis(vox)
    length = float(proj.max() - proj.min())
    dist_out = ndimage.distance_transform_edt(~mb, sampling=spacing)

    def project(mask):
        return (np.argwhere(mask) * spacing - mean) @ ax

    shell2 = (dist_out > 0) & (dist_out <= 2.0)
    sproj, sL = project(shell2), Lb[shell2]
    lo_end, hi_end = sproj <= proj.min() + 0.25 * length, sproj >= proj.max() - 0.25 * length
    bone_lo = float(np.isin(sL[lo_end], BONE).mean()) if lo_end.any() else 0.0
    bone_hi = float(np.isin(sL[hi_end], BONE).mean()) if hi_end.any() else 0.0
    root_is_low = bone_lo >= bone_hi
    crown_end, root_end = (hi_end, lo_end) if root_is_low else (lo_end, hi_end)
    t = (proj - proj.min()) / max(length, 1e-6)
    from_root = t if root_is_low else 1 - t
    crown, root = from_root >= 1 - CROWN_SHARE, from_root < 1 - CROWN_SHARE
    vals = Ib[mb]
    bright = vals >= BRIGHT_FRAC_OF_MAX * scan_max
    tooth_med = float(np.median(vals))
    surf = (mb & ~ndimage.binary_erosion(mb))[mb]
    inner = ndimage.binary_erosion(mb, iterations=2)[mb]
    # around the root tip: voxels 1-3 mm outside the tooth and within 4 mm of the apex point
    apex_pt = vox[from_root <= APEX_SHARE].mean(axis=0)
    ring = (dist_out >= SHELL_MM[0]) & (dist_out <= SHELL_MM[1])
    rv = np.argwhere(ring) * spacing
    near = np.linalg.norm(rv - apex_pt, axis=1) <= APEX_RADIUS_MM
    rI, rL = Ib[ring], Lb[ring]
    keep = near & (np.isin(rL, BONE) | (rL == 0))
    apex_vals = rI[keep]

    def frac(x):
        return float(x.mean()) if len(x) else np.nan

    return {
        "fdi_mask": fdi, "fdi_report": mirror(fdi), "n_vox": n, "pulp_share": frac(pulp), "length_mm": length,
        "truncated": truncated, "end_bone_contrast": abs(bone_lo - bone_hi),
        "tooth_median_rel_max": tooth_med / scan_max,
        "bright_crown": frac(bright[crown]), "bright_root": frac(bright[root]),
        "bright_crown_surface": frac(bright[crown & surf]), "bright_crown_interior": frac(bright[crown & ~surf]),
        "bright_pulp": frac(bright[pulp]), "bright_root_pulp": frac(bright[root & pulp]),
        "bright_root_inner": frac(bright[root & inner]),
        "dark_crown": frac(vals[crown & inner] < 0.6 * tooth_med),
        "prosthetic_label_near_crown": frac(np.isin(sL[crown_end], PROSTHETIC)),
        "implant_label_near_root": frac(sL[root_end] == IMPLANT),
        "apex_n": len(apex_vals),
        "apex_p20_rel_bone": float(np.percentile(apex_vals, 20) / bone_med) if len(apex_vals) and bone_med > 0 else np.nan,
        "apex_dark_share": frac(apex_vals < 0.5 * bone_med) if bone_med > 0 else np.nan,
    }


def case_rows(case_id: str, tf3_dir: str) -> list[dict]:
    try:
        img_p, lab_p = tf3_paths(case_id, Path(tf3_dir))
        img, lab = nib.load(str(img_p)), nib.load(str(lab_p))
        I = np.asarray(img.dataobj, dtype=np.float32)
        L = np.asarray(lab.dataobj).astype(np.int16)
        if I.shape != L.shape:
            return [{"case_id": case_id, "error": "shape_mismatch"}]
        spacing = np.asarray(lab.header.get_zooms()[:3], dtype=float)
        scan_max = float(I.max())
        bone = np.isin(L, BONE)
        bone_med = float(np.median(I[bone])) if bone.any() else np.nan
        case = {"case_id": case_id, "error": None, "scan_max": scan_max, "bone_median": bone_med,
                "n_saturated": int((I >= 0.99 * scan_max).sum()),
                "crown_label_vox": int((L == 9).sum()), "bridge_label_vox": int((L == 8).sum())}
        merged = np.where((L >= 111) & (L <= 148), L - 100, L)                # tooth + its pulp
        objs = ndimage.find_objects(np.where(np.isin(merged, TEETH), merged, 0))
        rows = []
        for fdi in TEETH:
            if fdi - 1 < len(objs) and objs[fdi - 1] is not None:
                r = tooth_row(I, L, fdi, spacing, scan_max, bone_med, objs[fdi - 1])
                if r is not None:
                    rows.append({**case, **r})
        return rows
    except Exception as e:                                                   # noqa: BLE001
        return [{"case_id": case_id, "error": f"{type(e).__name__}: {e}"[:200],
                 "trace": traceback.format_exc()[-400:]}]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--tf3-dir", required=True, type=Path)
    ap.add_argument("--role", default="dev", choices=["dev"], help="step 1 never touches the locked test set")
    ap.add_argument("--exclude", default="S0015,P116,P117,P260,P261")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="tooth-features")
    a = ap.parse_args(argv)
    sp = pd.read_csv(a.splits)
    cases = sorted(sp.loc[(sp["role"] == a.role) & ~sp["case_id"].isin(a.exclude.split(",")), "case_id"])
    if a.limit:
        cases = cases[: a.limit]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(case_rows, cases, [str(a.tf3_dir)] * len(cases)) for r in rs]
    df = pd.DataFrame(rows)
    run_id = make_run_id(a.tag + (f"-limit{a.limit}" if a.limit else ""))
    out = a.results_dir / "perception" / run_id
    out.mkdir(parents=True, exist_ok=False)
    df.to_parquet(out / "tooth_features.parquet", index=False)
    teeth = df[df["error"].isna()] if "error" in df else df
    summ = {"run_id": run_id, "n_cases": len(cases), "n_teeth": len(teeth),
            "case_errors": sorted(set(df.loc[df["error"].notna(), "case_id"])) if "error" in df else [],
            "share_orientation_uncertain": float((teeth["end_bone_contrast"] < 0.2).mean()) if len(teeth) else None,
            "share_truncated": float(teeth["truncated"].mean()) if len(teeth) else None,
            "share_scans_without_saturation": float((teeth.drop_duplicates("case_id")["n_saturated"] < 100).mean()) if len(teeth) else None}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    write_provenance(out, extra={**summ, "splits": str(a.splits), "tf3_dir": str(a.tf3_dir), "role": a.role,
                                 "excluded": a.exclude})
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
