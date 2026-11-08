"""Which left/right convention do the TF3 side labels (canals, sinuses) follow? Image-only, no reports.

For mirrored masks (--mirror-centres, default P/F/S; centre A masks from run_segmentation are already in
report convention and use standard FDI quadrants) the tooth labels are mirrored relative to the reports (DECISIONS 2026-11-08/09): TF3 label
quadrant 4 is the reports' quadrant 3 (patient's left, lower), TF3 quadrant 1 is the reports' quadrant 2
(patient's left, upper). Using world-x centroids from partialref.audit.visibility_features we test, per
case, whether the structure LABELLED "left" sits on the same side as the teeth the REPORTS call left:

- canals:   sign(cx[canal_left] - cx[canal_right])  vs  sign(cx[TF3 q4 teeth] - cx[TF3 q3 teeth])
- sinuses:  sign(cx[sinus_left] - cx[sinus_right])  vs  sign(cx[TF3 q1 teeth] - cx[TF3 q2 teeth])
If only one member of a pair is present, it is compared with the jaw centroid (mandible / maxilla;
fallback: mean of that arch's tooth centroids). "match" = the side label already follows the report
convention (keep names); "mismatch" = the side names must be swapped for comparison with reports.
Also recorded: whether the reports' left teeth lie at negative world x (RAS left) in the TF4 header,
i.e. whether the TF4 header's left/right agrees with the reports.

Usage:
    python -m partialref.audit.side_geometry --vis results/audit/<visibility run with *_cx columns>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from partialref.provenance import make_run_id, write_provenance

# (structure pair, mask quadrant holding the reports' LEFT teeth, ... RIGHT teeth, jaw), per tooth convention:
# mirrored masks (TF3 labels, P/F/S): reports' left lower = label q4, left upper = label q1;
# report-convention masks (e.g. centre A from run_segmentation): standard FDI, left = q3 / q2.
CHECKS = {True: {"canal": ("canal_left", "canal_right", "4", "3", "mandible"),
                 "sinus": ("sinus_left", "sinus_right", "1", "2", "maxilla")},
          False: {"canal": ("canal_left", "canal_right", "3", "4", "mandible"),
                  "sinus": ("sinus_left", "sinus_right", "2", "1", "maxilla")}}
REPORT_LEFT_QUADRANTS = {True: ["1", "4"], False: ["2", "3"]}


def _num(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x


def _direction(left: float, right: float, mid: float) -> float:
    """+1 if 'left' lies at larger world x than 'right' (or than the midline), -1 if smaller, NaN if unknown."""
    if np.isfinite(left) and np.isfinite(right):
        d = left - right
    elif np.isfinite(left) and np.isfinite(mid):
        d = left - mid
    elif np.isfinite(right) and np.isfinite(mid):
        d = mid - right
    else:
        return np.nan
    return float(np.sign(d)) if d != 0 else np.nan


def case_side_checks(case: dict, teeth: pd.DataFrame, mirrored: bool = True) -> dict:
    """case: one row of case_features; teeth: that case's rows of tooth_features (with cx).
    mirrored: the mask's tooth labels are mirrored relative to the reports (TF3 labels, P/F/S)."""
    pres = teeth[teeth["present"] == True]  # noqa: E712
    q = pres["fdi"].astype(int).astype(str).str[0]
    qcx = pres.groupby(q)["cx"].mean().to_dict() if "cx" in pres else {}
    out: dict = {"case_id": case["case_id"]}
    for name, (sl, sr, ql, qr, jaw) in CHECKS[mirrored].items():
        arch = [ql, qr]
        arch_teeth = pres[q.isin(arch)]["cx"] if "cx" in pres else pd.Series(dtype=float)
        mid = _num(case.get(f"{jaw}_cx"))
        if not np.isfinite(mid) and len(arch_teeth):
            mid = float(arch_teeth.mean())
        tdir = _direction(_num(qcx.get(ql)), _num(qcx.get(qr)), mid)
        sdir = _direction(_num(case.get(f"{sl}_cx")), _num(case.get(f"{sr}_cx")), mid)
        out[f"{name}_teeth_dir"], out[f"{name}_label_dir"] = tdir, sdir
        out[name] = ("undetermined" if not (np.isfinite(tdir) and np.isfinite(sdir))
                     else "match" if tdir == sdir else "mismatch")
    # reports' left teeth at negative world x = RAS left -> header agrees with the reports
    lq = REPORT_LEFT_QUADRANTS[mirrored]
    left = pres[q.isin(lq)]["cx"] if "cx" in pres else pd.Series(dtype=float)
    right = pres[~q.isin(lq)]["cx"] if "cx" in pres else pd.Series(dtype=float)
    out["header_lr_agrees_with_reports"] = (None if not (len(left) and len(right))
                                           else bool(left.mean() < right.mean()))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vis", required=True, type=Path)
    ap.add_argument("--mirror-centres", default="PFS",
                    help="centres whose mask tooth labels are mirrored vs the reports (as in load_visibility)")
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="side-geometry")
    a = ap.parse_args(argv)
    cf = pd.read_parquet(a.vis / "case_features.parquet")
    tf = pd.read_parquet(a.vis / "tooth_features.parquet")
    if "cx" not in tf:
        raise SystemExit("tooth_features has no 'cx' column: re-run visibility_features first")
    by_case = dict(tuple(tf.groupby("case_id")))
    rows = [case_side_checks(r, by_case.get(r["case_id"], tf.iloc[0:0]), mirrored=r["case_id"][0] in a.mirror_centres)
            for r in cf.to_dict("records") if r.get("label_found", False)]
    df = pd.DataFrame(rows)
    df["centre"] = df["case_id"].str[0]
    out = a.results_dir / "audit" / make_run_id(a.tag)
    out.mkdir(parents=True, exist_ok=False)
    df.to_parquet(out / "side_checks.parquet", index=False)
    summary = {"n_cases": len(df),
               "mirror_centres": a.mirror_centres,
               **{name: df.groupby("centre")[name].value_counts().unstack(fill_value=0).to_dict(orient="index")
                  for name in CHECKS[True]},
               "header_lr_agrees_with_reports": df.groupby("centre")["header_lr_agrees_with_reports"]
               .value_counts(dropna=False).unstack(fill_value=0).to_dict(orient="index")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_provenance(out, extra={"vis": str(a.vis), "mirror_centres": a.mirror_centres})
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
