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

Per-scan intensity levels (Shivam's suggestion, 2026-11-12; reviewed): voxels at the scanner ceiling are
counted (clip_share) and set aside; the remaining tooth voxels are clustered into pulp / dentin / enamel
(Gaussian mixture; dentin = largest group; enamel left out when the scanner clips). Every region is put on
the scan's own scale (pulp = 0, dentin = 1). Histogram features per region (deep crown, crown surface, root,
root-tip zone, mid-root band): percentiles 5/25/50/75/95, spread and skew on that scale.

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
from scipy import ndimage, stats
from sklearn.mixture import GaussianMixture

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
DEEP_MM = 2.0                 # "deep" = at least 2 mm inside the tooth surface (below the enamel)
MIDROOT_AT = 0.35             # reference bone: around the point 35 % of the way from the root tip


LEVEL_SAMPLE = 200_000
CLIP_SHARE_MAX = 0.01        # above this share of tooth voxels at the ceiling, the scanner clips enamel
REGIONS = ("crown_deep", "crown_surface", "root", "apex", "midroot")


def scan_levels(tooth_vals: np.ndarray, scan_max: float, seed: int = 0) -> dict[str, float]:
    """Per-scan levels from the tooth voxels. Voxels at the scanner ceiling (>= 0.99 x max) are counted
    (clip_share) and left out; the rest is clustered into 3 groups (Gaussian mixture): pulp/soft (lowest
    mean), dentin (largest weight), enamel (highest mean, only if the scanner does not clip, else NaN)."""
    v = tooth_vals[np.isfinite(tooth_vals)]
    clip = v >= 0.99 * scan_max
    out = {"clip_share": float(clip.mean()) if len(v) else np.nan}
    u = v[~clip]
    if len(u) < 1000:
        return out | {"lvl_low": np.nan, "lvl_dentin": np.nan, "lvl_enamel": np.nan}
    rng = np.random.default_rng(seed)
    s = rng.choice(u, min(LEVEL_SAMPLE, len(u)), replace=False).reshape(-1, 1)
    gm = GaussianMixture(3, random_state=seed).fit(s)
    means, w = gm.means_.ravel(), gm.weights_
    dentin = float(means[np.argmax(w)])
    low = float(means.min())
    enamel = float(means.max()) if out["clip_share"] < CLIP_SHARE_MAX and means.max() > dentin else np.nan
    return out | {"lvl_low": low, "lvl_dentin": dentin, "lvl_enamel": enamel}


def hist_feats(name: str, x: np.ndarray, low: float, dentin: float) -> dict[str, float]:
    """Percentiles, spread and skew of intensities on the scan's own scale: pulp level = 0, dentin = 1."""
    keys = [f"{name}_p{p}" for p in (5, 25, 50, 75, 95)] + [f"{name}_iqr", f"{name}_skew", f"{name}_n"]
    if len(x) < 10 or not (np.isfinite(dentin) and np.isfinite(low)) or dentin - low <= 0:
        return dict.fromkeys(keys, np.nan) | {f"{name}_n": float(len(x))}
    z = (x - low) / (dentin - low)
    q = np.percentile(z, [5, 25, 50, 75, 95])
    return dict(zip(keys, [*map(float, q), float(q[3] - q[1]), float(stats.skew(z)), float(len(x))]))


def mirror(fdi: int) -> int:
    return MIRROR_Q[fdi // 10] * 10 + fdi % 10


def tooth_axis(coords_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = coords_mm.mean(axis=0)
    c = coords_mm - mean
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[0], c @ vt[0], mean


def tooth_row(I: np.ndarray, L: np.ndarray, fdi: int, spacing: np.ndarray, scan_max: float,
              bone_med: float, bbox: tuple[slice, ...] | None = None, levels: dict | None = None) -> dict | None:
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
    dist_in = ndimage.distance_transform_edt(mb, sampling=spacing)
    deep = dist_in[mb] >= DEEP_MM
    # largest connected bright region deep in the crown (fillings, posts, crown cores), in mm3
    deep_bright = np.zeros(mb.shape, bool)
    deep_bright[mb] = crown & deep & bright
    comp, ncomp = ndimage.label(deep_bright)
    largest_deep_bright_mm3 = (float(np.bincount(comp.ravel())[1:].max()) * float(np.prod(spacing))) if ncomp else 0.0
    # around the root tip: voxels 1-3 mm outside the tooth and within 4 mm of the apex point, compared with
    # the same kind of voxels around the mid-root of the same tooth (its own surrounding bone)
    apex_pt = vox[from_root <= APEX_SHARE].mean(axis=0)
    ring = (dist_out >= SHELL_MM[0]) & (dist_out <= SHELL_MM[1])
    rv = np.argwhere(ring) * spacing
    rI, rL = Ib[ring], Lb[ring]
    tissue = np.isin(rL, BONE) | (rL == 0)
    apex_vals = rI[tissue & (np.linalg.norm(rv - apex_pt, axis=1) <= APEX_RADIUS_MM)]
    # reference: a band around the root at mid-root height (any root width), +-1.5 mm along the axis
    mid_proj = proj.min() + (MIDROOT_AT if root_is_low else 1 - MIDROOT_AT) * length
    mid_vals = rI[tissue & (np.abs((rv - mean) @ ax - mid_proj) <= 1.5)]
    local_ref = float(np.median(mid_vals)) if len(mid_vals) else np.nan

    def frac(x):
        return float(x.mean()) if len(x) else np.nan

    lv = levels or {}
    low, dentin = lv.get("lvl_low", np.nan), lv.get("lvl_dentin", np.nan)
    hist = {}
    for name, x in (("crown_deep", vals[crown & deep]), ("crown_surface", vals[crown & surf]), ("root", vals[root]),
                    ("apex", apex_vals), ("midroot", mid_vals)):
        hist |= hist_feats(name, x, low, dentin)
    cd = vals[crown & deep]
    ok = cd.size and np.isfinite(low) and np.isfinite(dentin) and dentin > low
    deep_dense = float(((cd - low) / (dentin - low) > 1.5).mean()) if ok else np.nan
    return {
        **hist, "crown_deep_dense_share": deep_dense, "crown_deep_at_ceiling": frac(cd >= 0.99 * scan_max),
        "fdi_mask": fdi, "fdi_report": mirror(fdi), "n_vox": n, "pulp_share": frac(pulp), "length_mm": length,
        "truncated": truncated, "end_bone_contrast": abs(bone_lo - bone_hi),
        "tooth_median_rel_max": tooth_med / scan_max,
        "bright_crown": frac(bright[crown]), "bright_root": frac(bright[root]),
        "bright_crown_surface": frac(bright[crown & surf]), "bright_crown_interior": frac(bright[crown & ~surf]),
        "bright_pulp": frac(bright[pulp]), "bright_root_pulp": frac(bright[root & pulp]),
        "bright_root_inner": frac(bright[root & inner]),
        "dark_crown": frac(vals[crown & inner] < 0.6 * tooth_med),
        "bright_crown_deep": frac(bright[crown & deep]), "largest_deep_bright_mm3": largest_deep_bright_mm3,
        "prosthetic_label_near_crown": frac(np.isin(sL[crown_end], PROSTHETIC)),
        "implant_label_near_root": frac(sL[root_end] == IMPLANT),
        "apex_n": len(apex_vals),
        "apex_p20_rel_bone": float(np.percentile(apex_vals, 20) / bone_med) if len(apex_vals) and bone_med > 0 else np.nan,
        "apex_dark_share": frac(apex_vals < 0.5 * bone_med) if bone_med > 0 else np.nan,
        "apex_rel_local": float(np.median(apex_vals) / local_ref) if len(apex_vals) and local_ref > 0 else np.nan,
        "apex_dark_share_local": frac(apex_vals < 0.6 * local_ref) if local_ref > 0 else np.nan,
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
        levels = scan_levels(I[np.isin(L, TEETH + [100 + t for t in TEETH])], scan_max)
        case = {"case_id": case_id, "error": None, "scan_max": scan_max, "bone_median": bone_med, **levels,
                "n_saturated": int((I >= 0.99 * scan_max).sum()),
                "crown_label_vox": int((L == 9).sum()), "bridge_label_vox": int((L == 8).sum())}
        merged = np.where((L >= 111) & (L <= 148), L - 100, L)                # tooth + its pulp
        objs = ndimage.find_objects(np.where(np.isin(merged, TEETH), merged, 0))
        rows = []
        for fdi in TEETH:
            if fdi - 1 < len(objs) and objs[fdi - 1] is not None:
                r = tooth_row(I, L, fdi, spacing, scan_max, bone_med, objs[fdi - 1], levels)
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
