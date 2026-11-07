"""Are TF4 P/F/S volumes the same scans as ToothFairy3 images with the same number? (HANDOFF §6.2)

For each TF4 case C<nnn> (C in P/F/S) we look for
    <tf3_dir>/imagesTr/ToothFairy3<C>_<nnn>_0000.nii.gz   and   <tf3_dir>/labelsTr/ToothFairy3<C>_<nnn>.nii.gz
and record: identical file (sha256), header geometry of both, which of the 8 axis flips maps the
TF3 voxel array onto the TF4 array exactly (diagnostic 2026-11-07: flip of axes 1 and 2, voxel-
identical, while the headers only differ in axis 1 → one header is wrong in z), and anatomical
checks with the TF3 labels in each header's world frame (RAS+): upper jaw (2) above lower jaw (1);
left canal (3) at smaller x than right canal (4). Also the canonical-reorientation comparison and
whether the label grid matches the TF3 image grid. Numbers only.

Usage:
    python -m partialref.audit.tf3_link --manifest results/manifest/<run_id> --tf3-dir DIR [--workers 8] [--limit N]
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

from partialref.provenance import make_run_id, sha256_file, write_provenance


def tf3_paths(case_id: str, tf3_dir: Path) -> tuple[Path, Path]:
    # TF3 pads like TF4 does (P/F: 3 digits, S: 4 digits, verified on disk) -> reuse the digits
    c, digits = case_id[0], case_id[1:]
    return (tf3_dir / "imagesTr" / f"ToothFairy3{c}_{digits}_0000.nii.gz",
            tf3_dir / "labelsTr" / f"ToothFairy3{c}_{digits}.nii.gz")


def _geom(img: nib.Nifti1Image) -> dict:
    return {"shape": list(img.shape[:3]), "zooms": [round(float(z), 5) for z in img.header.get_zooms()[:3]],
            "axcodes": "".join(nib.aff2axcodes(img.affine))}


def compare_case(case_id: str, tf4_path: str, tf3_dir: str, tf4_sha: str | None = None) -> dict:
    img_p, lab_p = tf3_paths(case_id, Path(tf3_dir))
    row: dict = {"case_id": case_id, "tf3_image": img_p.is_file(), "tf3_label": lab_p.is_file()}
    if not img_p.is_file():
        return row
    a, b = nib.load(tf4_path), nib.load(str(img_p))
    ga, gb = _geom(a), _geom(b)
    row.update({"tf4_axcodes": ga["axcodes"], "tf3_axcodes": gb["axcodes"],
                "same_shape_raw": ga["shape"] == gb["shape"],
                "same_zooms": ga["zooms"] == gb["zooms"]})
    row["same_file"] = (tf4_sha or sha256_file(tf4_path)) == sha256_file(img_p)
    ca, cb = nib.as_closest_canonical(a), nib.as_closest_canonical(b)
    row["same_shape_canonical"] = ca.shape[:3] == cb.shape[:3]
    if row["same_shape_canonical"]:
        xa = np.asarray(ca.dataobj, dtype=np.float32)
        xb = np.asarray(cb.dataobj, dtype=np.float32)
        d = np.abs(xa - xb)
        row["frac_voxels_equal"] = float((d == 0).mean())
        row["max_abs_diff"] = float(d.max())
        sa, sb = xa.ravel()[::97], xb.ravel()[::97]
        row["corr"] = float(np.corrcoef(sa, sb)[0, 1]) if sa.std() > 0 and sb.std() > 0 else np.nan
        row["affine_close_canonical"] = bool(np.allclose(ca.affine[:3, :3], cb.affine[:3, :3], atol=1e-3))
    flip = None
    if row["same_shape_raw"]:
        xa = np.asarray(a.dataobj, dtype=np.float32)
        xb = np.asarray(b.dataobj, dtype=np.float32)
        for fl in FLIPS:
            if np.array_equal(xa, _flip(xb, fl)):
                flip = fl
                break
        del xa, xb
    row["voxel_identical_flip"] = "".join(map(str, flip)) if flip else None
    if lab_p.is_file():
        lab = nib.load(str(lab_p))
        row["label_matches_tf3_grid"] = (lab.shape[:3] == b.shape[:3]
                                         and bool(np.allclose(lab.affine, b.affine, atol=1e-3)))
        row["label_axcodes"] = "".join(nib.aff2axcodes(lab.affine))
        L = np.asarray(lab.dataobj)
        row.update(_anatomy(L, lab.affine, "tf3"))
        if flip is not None:
            row.update(_anatomy(_flip(L, flip), a.affine, "tf4"))
    return row


FLIPS = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]


def _flip(x: np.ndarray, fl: tuple[int, int, int]) -> np.ndarray:
    for ax, f in enumerate(fl):
        if f:
            x = np.flip(x, ax)
    return x


def _centroid(L: np.ndarray, lbl: int, aff: np.ndarray):
    idx = np.argwhere(L == lbl)
    return None if len(idx) == 0 else (aff @ np.r_[idx.mean(0), 1.0])[:3]


def _anatomy(L: np.ndarray, aff: np.ndarray, tag: str) -> dict:
    lo, up, lc, rc = (_centroid(L, k, aff) for k in (1, 2, 3, 4))
    return {f"{tag}_upper_above_lower": None if up is None or lo is None else bool(up[2] > lo[2]),
            f"{tag}_left_canal_at_smaller_x": None if lc is None or rc is None else bool(lc[0] < rc[0])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--tf3-dir", required=True, type=Path)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="tf3-link")
    a = ap.parse_args(argv)

    cases = pd.read_parquet(a.manifest / "cases.parquet")
    cases = cases[cases["centre"].isin(list("PFS"))].sort_values("case_id")
    if a.limit:
        cases = cases.head(a.limit)
    args = [(r.case_id, r.volume_path, str(a.tf3_dir), getattr(r, "volume_sha256", None))
            for r in cases.itertuples()]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(compare_case, *zip(*args)))
    df = pd.DataFrame(rows)

    run_id = make_run_id(a.tag)
    out = a.results_dir / "audit" / run_id
    out.mkdir(parents=True, exist_ok=False)
    df.to_csv(out / "tf3_link.csv", index=False)

    def count(col: str) -> dict:
        return {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()} if col in df else {}

    summary = {
        "run_id": run_id, "n_cases": int(len(df)),
        **{c: count(c) for c in ["tf3_image", "tf3_label", "same_file", "same_shape_raw",
                                 "same_shape_canonical", "same_zooms", "affine_close_canonical",
                                 "label_matches_tf3_grid", "tf4_axcodes", "tf3_axcodes", "label_axcodes",
                                 "voxel_identical_flip", "tf3_upper_above_lower", "tf4_upper_above_lower",
                                 "tf3_left_canal_at_smaller_x", "tf4_left_canal_at_smaller_x"]},
        "frac_voxels_equal_min": float(df["frac_voxels_equal"].min()) if "frac_voxels_equal" in df else None,
        "corr_min": float(df["corr"].min()) if "corr" in df else None,
        "n_identical_voxels": int((df.get("frac_voxels_equal", pd.Series(dtype=float)) == 1.0).sum()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_provenance(out, extra={"run_id": run_id, "manifest_run": str(a.manifest), "tf3_dir": str(a.tf3_dir)})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
