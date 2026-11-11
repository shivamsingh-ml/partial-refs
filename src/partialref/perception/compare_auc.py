"""Paired case-bootstrap intervals for AUC / AP differences between feature sets (PAPER_PLAN step 2; 2026-11-12).

Input: predictions.parquet of a tooth_classifier run (out-of-fold probability per tooth, finding type and
feature set). For each finding type and each pair (A, B) of feature sets on the SAME teeth, cases are resampled
with replacement (teeth of a case stay together, the same resample for A and B), and AUC(A) - AUC(B) and
AP(A) - AP(B) are recomputed. Reports the observed difference and the 2.5 / 97.5 % percentiles.

    python -m partialref.perception.compare_auc --run results/perception/<classifier run> \\
        --pairs "basic+histogram+dino:basic+histogram (same teeth),dino:basic+histogram (same teeth)" [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from partialref.provenance import write_provenance


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """a, b: case_id, fdi_report, y, p for the same teeth. Returns observed differences and percentiles."""
    key = ["case_id", "fdi_report"]
    if a.duplicated(key).any() or b.duplicated(key).any():
        raise ValueError("duplicate teeth within a feature set")
    m = a[key + ["y", "p"]].merge(b[key + ["y", "p"]], on=key, suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b) or not (m["y_a"] == m["y_b"]).all():
        raise ValueError("feature sets are not on the same teeth / labels")
    y, pa, pb = m["y_a"].to_numpy(), m["p_a"].to_numpy(), m["p_b"].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("labels have one class only")
    cases = m["case_id"].to_numpy()
    uc, inv = np.unique(cases, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uc))]
    rng = np.random.default_rng(seed)
    d_auc, d_ap = [], []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(uc), len(uc))])
        if len(np.unique(y[idx])) < 2:
            continue
        d_auc.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
        d_ap.append(average_precision_score(y[idx], pa[idx]) - average_precision_score(y[idx], pb[idx]))
    q = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])] if x else [np.nan, np.nan]
    return {"n_teeth": len(m), "n_cases": len(uc), "n_pos": int(y.sum()),
            "auc_a": roc_auc_score(y, pa), "auc_b": roc_auc_score(y, pb), "d_auc": roc_auc_score(y, pa) - roc_auc_score(y, pb),
            "d_auc_lo": q(d_auc)[0], "d_auc_hi": q(d_auc)[1],
            "d_ap": average_precision_score(y, pa) - average_precision_score(y, pb), "d_ap_lo": q(d_ap)[0], "d_ap_hi": q(d_ap)[1],
            "n_boot_used": len(d_auc)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--pairs", required=True, help="comma-separated A:B pairs of feature-set names")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    P = pd.read_parquet(a.run / "predictions.parquet")
    rows = []
    for pair in a.pairs.split(","):
        A, B = (x.strip() for x in pair.split(":"))
        for t in sorted(P["finding_type"].unique()):
            pa = P[(P.finding_type == t) & (P.feature_set == A)]
            pb = P[(P.finding_type == t) & (P.feature_set == B)]
            if not (len(pa) and len(pb)):
                continue
            if pa["y"].nunique() < 2:
                print(f"skipped {t} ({A} vs {B}): one label class only")
                continue
            try:
                rows.append({"finding_type": t, "a": A, "b": B, **paired_bootstrap(pa, pb, a.n_boot, a.seed)})
            except ValueError as e:
                raise ValueError(f"{t}, '{A}' vs '{B}': {e}") from e
    res = pd.DataFrame(rows)
    res.to_csv(a.run / "paired_differences.csv", index=False)
    write_provenance(a.run / "paired_differences_provenance", extra={"pairs": a.pairs, "n_boot": a.n_boot, "seed": a.seed})
    print(res[["finding_type", "a", "b", "auc_a", "auc_b", "d_auc", "d_auc_lo", "d_auc_hi", "d_ap", "d_ap_lo", "d_ap_hi"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
