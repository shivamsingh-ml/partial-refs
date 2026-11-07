"""Inventory and centre characterisation (HANDOFF M1 step 1), from a manifest run.

Inputs: ``results/manifest/<run_id>/{cases,reports}.parquet`` (no report text in them).
Outputs into ``results/audit/<run_id>/``:
- ``centre_summary.csv``      per centre: cases, reports per case, multi-report share, report
                              length (EN chars), scan geometry (shape, spacing, field of view);
- ``reports_per_case.csv``    centre x number of EN reports;
- ``report_x_by_centre.csv``  centre x report_x (candidate reader ID) counts;
- ``report_x_within_case.json`` do the reports of one case carry distinct report_x values?
- ``geometry_vs_reference.csv`` (optional) per-case comparison with an earlier geometry table;
- ``summary.json``, ``provenance.json``.
Everything is counts / numbers; safe to print.

Usage:
    python -m partialref.audit.inventory --manifest results/manifest/<run_id> [--reference-geometry CSV]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from partialref.provenance import make_run_id, write_provenance

GEOM = ["shape_x", "shape_y", "shape_z", "spacing_x", "spacing_y", "spacing_z",
        "extent_x_mm", "extent_y_mm", "extent_z_mm"]
# column names in cbct-vlm-transfer/reports/intensity_percase.csv
REF_MAP = {"size_x": "shape_x", "size_y": "shape_y", "size_z": "shape_z",
           "spacing_x": "spacing_x", "spacing_y": "spacing_y", "spacing_z": "spacing_z",
           "extent_x": "extent_x_mm", "extent_y": "extent_y_mm", "extent_z": "extent_z_mm",
           "orientation": "axcodes"}


def _q(s: pd.Series) -> str:
    s = s.dropna()
    return "" if s.empty else f"{s.median():.3g} [{s.min():.3g}–{s.max():.3g}]"


def centre_summary(cases: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    en = reports[reports["lang"] == "en"]
    chars = en.groupby("case_id")["n_chars"].mean().rename("mean_en_chars")
    c = cases.merge(chars, left_on="case_id", right_index=True, how="left")
    rows = []
    for centre, g in c.groupby("centre"):
        e = en[en["case_id"].isin(g["case_id"])]
        row = {
            "centre": centre, "n_cases": len(g), "n_en_reports": int(g["n_reports_en"].sum()),
            "n_multi_report": int((g["n_reports_en"] >= 2).sum()),
            "share_multi_report": round(float((g["n_reports_en"] >= 2).mean()), 3),
            "en_chars_per_report": _q(e["n_chars"]),
            "orientation": "; ".join(f"{k}:{v}" for k, v in g["axcodes"].value_counts().items())
            if "axcodes" in g else "",
        }
        for col in GEOM:
            if col in g:
                row[col] = _q(g[col])
        rows.append(row)
    return pd.DataFrame(rows)


def report_x_within_case(reports: pd.DataFrame) -> dict:
    en = reports[(reports["lang"] == "en") & reports["report_x"].notna()]
    per_case = en.groupby("case_id")["report_x"].agg(list)
    multi = per_case[per_case.map(len) >= 2]
    distinct = multi.map(lambda xs: len(set(xs)) == len(xs))
    all_same = multi.map(lambda xs: len(set(xs)) == 1)
    return {"n_multi_report_cases": int(len(multi)), "all_distinct": int(distinct.sum()),
            "all_same": int(all_same.sum()),
            "mixed": int(len(multi) - distinct.sum() - all_same.sum()),
            "n_distinct_report_x": int(en["report_x"].nunique())}


def compare_geometry(cases: pd.DataFrame, ref: pd.DataFrame, atol: float = 0.06) -> pd.DataFrame:
    """Per-case comparison with an earlier geometry table (``REF_MAP`` column names).

    Extents are compared with ``atol`` mm (the reference rounds to 0.1 mm), spacings with 1e-3,
    shapes and orientation exactly.
    """
    r = ref.rename(columns=REF_MAP)[["case_id", *[v for v in REF_MAP.values() if v in
                                                  ref.rename(columns=REF_MAP).columns]]]
    m = cases.merge(r, on="case_id", how="outer", suffixes=("", "_ref"), indicator=True)
    out = m[["case_id", "_merge"]].rename(columns={"_merge": "in"}).copy()
    out["in"] = out["in"].map({"both": "both", "left_only": "ours_only", "right_only": "ref_only"})
    for col in REF_MAP.values():
        if col not in m or f"{col}_ref" not in m:
            continue
        a, b = m[col], m[f"{col}_ref"]
        if col == "axcodes":
            ok = a.astype(str) == b.astype(str)
        elif col.startswith("shape"):  # outer merge can turn ints into floats: compare numerically
            ok = a.astype(float) == b.astype(float)
        else:
            tol = atol if col.startswith("extent") else 1e-3
            ok = np.isclose(a.astype(float), b.astype(float), atol=tol)
        out[f"{col}_match"] = np.where(out["in"] == "both", ok, np.nan)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path, help="results/manifest/<run_id>")
    ap.add_argument("--reference-geometry", type=Path)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="inventory")
    a = ap.parse_args(argv)

    cases = pd.read_parquet(a.manifest / "cases.parquet")
    reports = pd.read_parquet(a.manifest / "reports.parquet")
    run_id = make_run_id(a.tag)
    out = a.results_dir / "audit" / run_id
    out.mkdir(parents=True, exist_ok=False)

    cs = centre_summary(cases, reports)
    cs.to_csv(out / "centre_summary.csv", index=False)
    pd.crosstab(cases["centre"], cases["n_reports_en"]).to_csv(out / "reports_per_case.csv")
    en = reports[reports["lang"] == "en"].merge(cases[["case_id", "centre"]], on="case_id")
    pd.crosstab(en["report_x"], en["centre"]).to_csv(out / "report_x_by_centre.csv")
    rx = report_x_within_case(reports)
    (out / "report_x_within_case.json").write_text(json.dumps(rx, indent=2))

    summary = {"run_id": run_id, "n_cases": int(len(cases)), "report_x_within_case": rx}
    if a.reference_geometry:
        cmp_ = compare_geometry(cases, pd.read_csv(a.reference_geometry))
        cmp_.to_csv(out / "geometry_vs_reference.csv", index=False)
        match_cols = [c for c in cmp_.columns if c.endswith("_match")]
        both = cmp_[cmp_["in"] == "both"]
        summary["geometry_vs_reference"] = {
            "reference": str(a.reference_geometry),
            "coverage": cmp_["in"].value_counts().to_dict(),
            "mismatching_cases": int((both[match_cols] == 0).any(axis=1).sum()),
            "mismatches_per_field": {c: int((both[c] == 0).sum()) for c in match_cols},
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    manifest_prov = json.loads((a.manifest / "provenance.json").read_text())
    write_provenance(out, extra={"run_id": run_id, "manifest_run": str(a.manifest),
                                 "manifest_hash": manifest_prov.get("manifest_hash")})
    print(cs.to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
