"""GT/T4: how much does reference incompleteness move precision? (HANDOFF §7 GT; expected values, no model run)

Takes the T2 estimates (prevalence p and reporting propensity pi per finding type, pooled double-read) and
computes, for described systems, the precision a reference-based evaluation would show:

1. Per finding type, for a system whose TRUE precision is fixed (scenario values): expected raw precision
   against n reference reports (raw - true = the bias the incomplete reference creates), the §2.2 corrected
   precision, and corrected - raw. pi_lo / pi_hi from T2 give the range (p kept at its estimate).
   Note: §2.2 is exact only for a system without skill; for a skilled system it under-corrects, most for
   rare findings (low p), so corrected - raw understates the bias there (HANDOFF L1).
   The range moves pi between its T2 limits with p held at its estimate; p and pi are negatively correlated,
   so this range is approximate (review 2026-11-11: about 0.003-0.017 too narrow at the low end).
   phi = 0 is assumed (T3 measures phi against the masks).
2. Two systems with the SAME true precision and sensitivity on every type they report:
   "complete" reports every supported type, "conservative" stays silent on the types with pi below
   --conservative-min-pi (the kind of silence rule a challenge system adopts). Micro-averaged over types with
   weights = expected number of true findings (p x instances). Any raw-precision gap between them is created
   by the references alone.

The true precision, sensitivity and the silence threshold are ASSUMED scenario values, written to the output.

    python -m partialref.propensity.effect_size --propensity-run results/propensity/<T2 run> \
        [--true-precision 0.6,0.8,0.95] [--n-ref 1,2] [--sensitivity 0.7] [--conservative-min-pi 0.8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from partialref.metrics.corrected import expected_precisions
from partialref.provenance import make_run_id, write_provenance


def by_type(est: pd.DataFrame, true_precisions: list[float], n_refs: list[int]) -> pd.DataFrame:
    """est: finding_type, p_mle, pi_mle, pi_lo, pi_hi (supported types only)."""
    rows = []
    for r in est.itertuples():
        for n in n_refs:
            for tp in true_precisions:
                e = expected_precisions(r.p_mle, r.pi_mle, n, tp)
                lo = expected_precisions(r.p_mle, r.pi_hi, n, tp)      # high pi -> small effect
                hi = expected_precisions(r.p_mle, r.pi_lo, n, tp)
                rows.append({"finding_type": r.finding_type, "n_ref": n, "true_precision": tp, "p": r.p_mle,
                             "pi": r.pi_mle, "raw_precision": e["raw"], "corrected_precision": e["corrected"],
                             "raw_minus_true": e["raw"] - tp, "corrected_minus_raw": e["corrected"] - e["raw"],
                             "corrected_minus_raw_lo": lo["corrected"] - lo["raw"],
                             "corrected_minus_raw_hi": hi["corrected"] - hi["raw"],
                             "corrected_minus_true": e["corrected"] - tp})
    return pd.DataFrame(rows)


def systems(est: pd.DataFrame, true_precision: float, sensitivity: float, n: int, min_pi: float) -> pd.DataFrame:
    """Micro-averaged scores of the complete and the conservative system (same true precision and sensitivity)."""
    w = est["p_mle"] * est["n_instances"]                     # expected true findings per type
    e = [expected_precisions(p, pi, n, true_precision) for p, pi in zip(est["p_mle"], est["pi_mle"])]
    ment = pd.Series([x["mentioned_share"] for x in e], index=est.index)
    q0 = pd.Series([x["q0"] for x in e], index=est.index)
    rows = []
    for name, keep in (("complete", pd.Series(True, index=est.index)), ("conservative", est["pi_mle"] >= min_pi)):
        tp = (w * sensitivity)[keep]                          # true positives per type
        preds = tp / true_precision
        matched = tp * ment[keep]
        raw_p = matched.sum() / preds.sum()
        corr_p = (matched + q0[keep] * (preds - matched)).sum() / preds.sum()
        raw_r = matched.sum() / (w * ment).sum()              # against what the references mention
        true_r = tp.sum() / w.sum()
        f1 = lambda p_, r_: 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        rows.append({"system": name, "n_ref": n, "true_precision": true_precision, "sensitivity": sensitivity,
                     "types_reported": int(keep.sum()), "types_silenced": ",".join(est.loc[~keep, "finding_type"]),
                     "raw_precision": raw_p, "corrected_precision": corr_p, "raw_recall": raw_r,
                     "true_recall": true_r, "raw_f1": f1(raw_p, raw_r), "true_f1": f1(true_precision, true_r),
                     "corrected_f1": f1(corr_p, raw_r)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--propensity-run", required=True, type=Path)
    ap.add_argument("--group", default="pooled_double_read")
    floats = lambda s: [float(x) for x in s.split(",") if x]
    ap.add_argument("--true-precision", type=floats, default=[0.6, 0.8, 0.95])
    ap.add_argument("--n-ref", type=lambda s: [int(x) for x in s.split(",") if x], default=[1, 2])
    ap.add_argument("--headline-true-precision", type=float, default=0.8)
    ap.add_argument("--sensitivity", type=float, default=0.7)
    ap.add_argument("--conservative-min-pi", type=float, default=0.8)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="effect-size-gt")
    a = ap.parse_args(argv)
    if a.headline_true_precision not in a.true_precision:
        ap.error("--headline-true-precision must be one of --true-precision")
    if 1 not in a.n_ref:
        ap.error("--n-ref must include 1 (the headline uses one reference report)")
    t2 = pd.read_csv(a.propensity_run / "propensity_by_type.csv")
    est = t2[(t2["group"] == a.group) & (t2["supported"] == True)].dropna(subset=["pi_mle", "p_mle"])
    est = est.reset_index(drop=True)
    per_type = by_type(est, a.true_precision, a.n_ref)
    sysres = pd.concat([systems(est, tp, a.sensitivity, n, a.conservative_min_pi)
                        for n in a.n_ref for tp in a.true_precision], ignore_index=True)
    head = per_type[(per_type["n_ref"] == 1) & (per_type["true_precision"] == a.headline_true_precision)]
    hs = sysres[(sysres["n_ref"] == 1) & (sysres["true_precision"] == a.headline_true_precision)].set_index("system")
    summary = {"propensity_run": a.propensity_run.name, "group": a.group, "n_types": len(est),
               "assumed": {"true_precision": a.true_precision, "headline_true_precision": a.headline_true_precision,
                           "sensitivity": a.sensitivity, "conservative_min_pi": a.conservative_min_pi},
               "headline": "one reference report, true precision " + str(a.headline_true_precision),
               "types_with_true_minus_raw_ge_0_05": head.loc[-head["raw_minus_true"] >= 0.05, "finding_type"].tolist(),
               "types_with_corrected_minus_raw_ge_0_05": head.loc[head["corrected_minus_raw"] >= 0.05, "finding_type"].tolist(),
               "types_with_lower_bound_ge_0_05": head.loc[head["corrected_minus_raw_lo"] >= 0.05, "finding_type"].tolist(),
               "raw_precision_conservative_minus_complete": float(hs.loc["conservative", "raw_precision"] - hs.loc["complete", "raw_precision"]),
               "corrected_precision_conservative_minus_complete": float(hs.loc["conservative", "corrected_precision"] - hs.loc["complete", "corrected_precision"]),
               "raw_f1_conservative_minus_complete": float(hs.loc["conservative", "raw_f1"] - hs.loc["complete", "raw_f1"]),
               "true_f1_conservative_minus_complete": float(hs.loc["conservative", "true_f1"] - hs.loc["complete", "true_f1"])}
    run_id = make_run_id(a.tag)
    out = a.results_dir / "propensity" / run_id
    out.mkdir(parents=True, exist_ok=False)
    per_type.to_csv(out / "effect_by_type.csv", index=False)
    sysres.to_csv(out / "systems.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"run_id": run_id, **summary}, indent=2))
    write_provenance(out, extra={"run_id": run_id, "propensity_run": str(a.propensity_run)})
    print(head[["finding_type", "p", "pi", "raw_precision", "raw_minus_true", "corrected_precision", "corrected_minus_raw",
                "corrected_minus_raw_lo", "corrected_minus_raw_hi", "corrected_minus_true"]].round(3).to_string(index=False))
    print(sysres[sysres["n_ref"] == 1].drop(columns=["types_silenced"]).round(3).to_string(index=False))
    print(json.dumps({"run_id": run_id, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
