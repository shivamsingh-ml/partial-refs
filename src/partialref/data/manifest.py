"""Dataset manifest (HANDOFF §6.3), for the ToothFairy4 release layout found on 2026-11-07:

    <data_dir>/<case>/cbct/volume.nii.gz
    <data_dir>/<case>/reports_en/cbct_<X>_<Y>.txt
    <data_dir>/<case>/reports_it/cbct_<X>_<Y>.txt      (same names as reports_en)

Writes, into ``results/manifest/<run_id>/``:
- ``cases.parquet``   one row per case: centre prefix, report counts, volume header features;
- ``reports.parquet`` one row per report file: IDs, language, size, sha256, character count.
**Report text is never stored or printed**; only counts and hashes are computed from it.

``<X>`` is kept as ``report_x`` — it looks like a reader ID but that is unverified (DECISIONS.md).
``centre`` is the case-ID prefix letter, also to be verified against the dataset description.

Usage:
    python -m partialref.data.manifest --data-dir DIR [--limit N | --case-ids A003,P001] [--no-volume-hash]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from partialref.provenance import make_run_id, sha256_file, write_provenance

REPORT_RE = re.compile(r"^cbct_(\d+)_(\d+)\.txt$")
LANGS = ("en", "it")


def list_cases(data_dir: Path) -> list[str]:
    return sorted(p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def volume_features(path: Path) -> dict:
    """Header-only features (no voxel data is loaded)."""
    img = nib.load(str(path))
    hdr = img.header
    shape = tuple(int(s) for s in hdr.get_data_shape()[:3])
    zooms = tuple(float(z) for z in hdr.get_zooms()[:3])
    extent = tuple(s * z for s, z in zip(shape, zooms))
    return {
        "shape_x": shape[0], "shape_y": shape[1], "shape_z": shape[2],
        "spacing_x": zooms[0], "spacing_y": zooms[1], "spacing_z": zooms[2],
        "extent_x_mm": extent[0], "extent_y_mm": extent[1], "extent_z_mm": extent[2],
        "dtype": str(hdr.get_data_dtype()),
        "axcodes": "".join(nib.aff2axcodes(img.affine)),
        "qform_code": int(hdr["qform_code"]), "sform_code": int(hdr["sform_code"]),
        "volume_bytes": path.stat().st_size,
    }


def report_rows(case: str, case_dir: Path) -> list[dict]:
    rows = []
    for lang in LANGS:
        d = case_dir / f"reports_{lang}"
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            m = REPORT_RE.match(f.name)
            raw = f.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            rows.append({
                "case_id": case, "lang": lang, "file": f.name,
                "report_id": f.stem.removeprefix("cbct_") if m else None,
                "report_x": int(m.group(1)) if m else None,
                "report_y": int(m.group(2)) if m else None,
                "name_ok": bool(m),
                "bytes": len(raw), "n_chars": len(text), "n_lines": text.count("\n") + 1,
                "utf8_ok": "�" not in text,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "path": str(f),
            })
    return rows


def build_manifest(data_dir: Path, case_ids: list[str] | None = None, limit: int | None = None,
                   hash_volumes: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = case_ids or list_cases(data_dir)
    if limit:
        cases = cases[:limit]
    case_rows, rep_rows = [], []
    for case in cases:
        cdir = data_dir / case
        vol = cdir / "cbct" / "volume.nii.gz"
        reps = report_rows(case, cdir)
        rep_rows += reps
        en = {r["file"] for r in reps if r["lang"] == "en"}
        it = {r["file"] for r in reps if r["lang"] == "it"}
        row = {
            "case_id": case,
            "centre": case[0],
            "n_reports_en": len(en), "n_reports_it": len(it),
            "en_it_names_match": en == it,
            "report_ids_en": sorted(r["report_id"] for r in reps if r["lang"] == "en"),
            "report_x_en": sorted(r["report_x"] for r in reps if r["lang"] == "en"),
            "volume_present": vol.is_file(),
            "volume_path": str(vol),
        }
        if vol.is_file():
            row.update(volume_features(vol))
            row["volume_sha256"] = sha256_file(vol) if hash_volumes else None
        case_rows.append(row)
    return pd.DataFrame(case_rows), pd.DataFrame(rep_rows)


def manifest_hash(cases: pd.DataFrame, reports: pd.DataFrame) -> str:
    """Order-independent hash of the file identities (paths are excluded, hashes included)."""
    h = hashlib.sha256()
    for cid, vsha in sorted(zip(cases["case_id"], cases.get("volume_sha256", [None] * len(cases)))):
        h.update(f"{cid}:{vsha}\n".encode())
    for row in sorted(zip(reports["case_id"], reports["lang"], reports["file"], reports["sha256"])):
        h.update(":".join(map(str, row)).encode() + b"\n")
    return h.hexdigest()


def summarize(cases: pd.DataFrame) -> dict:
    """Counts only (safe to print)."""
    out = {
        "n_cases": int(len(cases)),
        "cases_per_centre": {str(k): int(v) for k, v in
                             cases["centre"].value_counts().sort_index().items()},
        "en_reports_per_case": {str(k): int(v) for k, v in
                                cases["n_reports_en"].value_counts().sort_index().items()},
        "n_en_reports": int(cases["n_reports_en"].sum()),
        "n_multi_report_cases": int((cases["n_reports_en"] >= 2).sum()),
        "en_it_names_match_all": bool(cases["en_it_names_match"].all()),
        "volumes_missing": int((~cases["volume_present"]).sum()),
    }
    by = cases.groupby(["centre", "n_reports_en"]).size().unstack(fill_value=0)
    out["en_reports_per_case_by_centre"] = {str(c): {int(k): int(v) for k, v in r.items() if v}
                                            for c, r in by.iterrows()}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--case-ids", type=lambda s: [x for x in s.split(",") if x])
    ap.add_argument("--no-volume-hash", action="store_true")
    ap.add_argument("--tag", default="manifest")
    a = ap.parse_args(argv)

    cases, reports = build_manifest(a.data_dir, a.case_ids, a.limit, hash_volumes=not a.no_volume_hash)
    run_id = make_run_id(a.tag)
    out = a.results_dir / "manifest" / run_id
    out.mkdir(parents=True, exist_ok=False)
    cases.to_parquet(out / "cases.parquet", index=False)
    reports.to_parquet(out / "reports.parquet", index=False)
    mhash = manifest_hash(cases, reports)
    summary = summarize(cases)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_provenance(out, extra={"run_id": run_id, "manifest_hash": mhash, "data_dir": str(a.data_dir),
                                 "volume_hash": not a.no_volume_hash, "limit": a.limit,
                                 "case_ids": a.case_ids})
    print(json.dumps({"run_id": run_id, "out": str(out), "manifest_hash": mhash, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
