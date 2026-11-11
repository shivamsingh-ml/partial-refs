"""Human reference for the per-tooth classifier (2026-11-12): is the model already as good as a second dentist?

On double-read cases, one report (A) is used as a yes/no "predictor" of another report's (B) per-tooth labels;
every ordered pair of reports of a case counts once (A->B and B->A). Labels per report come from the same
rule as the classifier (tooth_labels with a single report; authors silent on a type give no label). For each
finding type, on the same teeth:
  human: sensitivity = P(A says yes | B says yes), false-positive rate = P(A says yes | B says no);
  model: out-of-fold probability (classifier run, feature set 'basic+histogram' by default) scored against B:
         AUC, and the model's sensitivity at the human's false-positive rate (read off the model's ROC curve).
Difference model - human sensitivity at that rate, with 95 % intervals from resampling cases (the threshold
is re-read in every resample). Each case-tooth has weight 1 in total: a case with k reports gives k(k-1)
ordered pairs, each weighted 1 / (k(k-1)). Parse run, manifest, reader profiles and min_n are read from the
classifier run's provenance, and the pair labels are checked against the classifier's labels.

This measures agreement with a second reader, not accuracy. Caveats (review 2026-11-12), all but the last
favour the human: double-read cases are the harder cases; omissions are probably shared between readers; the
model is trained on union labels (more positives than one report), so it flags teeth B omitted; silent
authors were found from agreement on these same double reads; the ~13 annotators recur, so intervals are
too narrow for "a dentist in general". Reading the model's threshold on the same rows favours the model.

    python -m partialref.perception.human_reference --classifier-run results/perception/<run with predictions> \\
        [--feature-set basic+histogram] [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from partialref.perception.tooth_classifier import EXCLUDE, TYPES, silent_authors, tooth_labels
from partialref.propensity.agreement import select_reports
from partialref.provenance import write_provenance


def pair_table(teeth: pd.DataFrame, reps: pd.DataFrame, inst: pd.DataFrame, silent: set) -> pd.DataFrame:
    """teeth: case_id, fdi_report. reps: case_id, report_id, report_x. One row per (ordered report pair,
    tooth, type) with both labels known: case_id, fdi_report, finding_type, rep_a, rep_b, a, b."""
    per_rep = []
    for rid, r in reps.groupby("report_id"):
        lab = tooth_labels(teeth[teeth["case_id"].isin(r["case_id"])], r, inst, silent)
        per_rep.append(lab.assign(report_id=rid, author=str(r["report_x"].iloc[0])))
    if not per_rep:
        return pd.DataFrame(columns=["case_id", "fdi_report", "finding_type", "rep_a", "rep_b", "a", "b", "w", "same_author"])
    L = pd.concat(per_rep, ignore_index=True).melt(["case_id", "fdi_report", "report_id", "author"], TYPES, "finding_type", "y")
    L = L.dropna(subset=["y"])
    m = L.merge(L, on=["case_id", "fdi_report", "finding_type"], suffixes=("_a", "_b"))
    m = m[m["report_id_a"] != m["report_id_b"]]
    k = m.groupby(["case_id", "fdi_report", "finding_type"])["report_id_a"].transform("nunique")
    m = m.assign(w=1.0 / (k * (k - 1)), same_author=m["author_a"] == m["author_b"])
    return m.rename(columns={"report_id_a": "rep_a", "report_id_b": "rep_b", "y_a": "a", "y_b": "b"})[
        ["case_id", "fdi_report", "finding_type", "rep_a", "rep_b", "a", "b", "w", "same_author"]].reset_index(drop=True)


def compare(a: np.ndarray, b: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict:
    """a: report A's yes/no, b: report B's yes/no (truth here), p: model probability, w: row weights."""
    w = np.ones(len(b)) if w is None else w
    pos, neg = b == 1, b == 0
    tpr_h, fpr_h = float(np.average(a[pos], weights=w[pos])), float(np.average(a[neg], weights=w[neg]))
    fpr, tpr, _ = roc_curve(b, p, sample_weight=w)
    tpr_m = float(np.interp(fpr_h, fpr, tpr))
    return {"auc_model": float(roc_auc_score(b, p, sample_weight=w)), "sens_human": tpr_h, "fpr_human": fpr_h,
            "sens_model_at_fpr_human": tpr_m, "d_sens": tpr_m - tpr_h}


def bootstrap(m: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """m: a, b, p, case_id for one type. Observed comparison and case-resampled 95 % intervals."""
    a, b, p, w = m["a"].to_numpy(), m["b"].to_numpy(), m["p"].to_numpy(), m["w"].to_numpy()
    obs = compare(a, b, p, w)
    uc, inv = np.unique(m["case_id"].to_numpy(), return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uc))]
    rng = np.random.default_rng(seed)
    d, auc = [], []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(uc), len(uc))])
        if len(np.unique(b[idx])) < 2:
            continue
        r = compare(a[idx], b[idx], p[idx], w[idx])
        d.append(r["d_sens"])
        auc.append(r["auc_model"])
    q = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])] if x else [np.nan, np.nan]
    return {**obs, "d_sens_lo": q(d)[0], "d_sens_hi": q(d)[1], "auc_model_lo": q(auc)[0], "auc_model_hi": q(auc)[1],
            "n_boot_used": len(d)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifier-run", required=True, type=Path)
    ap.add_argument("--feature-set", default="basic+histogram")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    prov = json.loads((a.classifier_run / "provenance.json").read_text())
    parse_run, manifest, readers, min_n = Path(prov["parse_run"]), Path(prov["manifest"]), Path(prov["readers"]), int(prov["min_n"])
    P = pd.read_parquet(a.classifier_run / "predictions.parquet")
    P = P[P["feature_set"] == a.feature_set]
    if P.empty or P.duplicated(["case_id", "fdi_report", "finding_type"]).any():
        raise ValueError(f"no or duplicate predictions for feature set {a.feature_set!r}")
    reps, bad = select_reports(manifest, parse_run, "en", "PFS", EXCLUDE)
    reps = reps[~reps["case_id"].isin({c for c, _ in bad}) & reps["case_id"].isin(set(P["case_id"]))]
    reps = reps[["case_id", "report_id", "report_x"]]
    reps = reps[reps.groupby("case_id")["report_id"].transform("nunique") >= 2]          # double-read cases
    teeth = P[["case_id", "fdi_report"]].drop_duplicates()
    inst = pd.read_parquet(parse_run / "instances_en.parquet")
    silent = silent_authors(pd.read_csv(readers / "profiles.csv"), min_n)
    pairs = pair_table(teeth, reps, inst, silent)
    # consistency: where every report of a case gives a label, the classifier's label is the union of them
    k_case = reps.groupby("case_id")["report_id"].nunique()
    full = pairs.groupby(["case_id", "fdi_report", "finding_type"]).agg(n_rep=("rep_a", "nunique"), any_yes=("a", "max"))
    full = full[full["n_rep"] == full.index.get_level_values(0).map(k_case).to_numpy()].reset_index()
    chk = full.merge(P[["case_id", "fdi_report", "finding_type", "y"]], on=["case_id", "fdi_report", "finding_type"])
    n_mismatch = int((chk["any_yes"] != chk["y"]).sum())
    if n_mismatch:
        raise ValueError(f"{n_mismatch} pair labels disagree with the classifier's labels: inputs differ")
    m = pairs.merge(P[["case_id", "fdi_report", "finding_type", "p"]], on=["case_id", "fdi_report", "finding_type"],
                    validate="many_to_one")
    rows = []
    for t in TYPES:
        mt = m[m["finding_type"] == t]
        if mt["b"].nunique() < 2:
            continue
        ct = mt.drop_duplicates(["case_id", "fdi_report"])
        row = {"finding_type": t, "n_case_teeth": len(ct), "n_cases": mt["case_id"].nunique(),
               "n_pos_b_weighted": float((mt["b"] * mt["w"]).sum()), "n_same_author_pairs": int(mt["same_author"].sum()),
               **bootstrap(mt, a.n_boot, a.seed)}
        for c in "FPS":                                                      # human operating point per centre
            mc = mt[mt["case_id"].str[0] == c]
            if mc["b"].nunique() == 2:
                pc = compare(mc["a"].to_numpy(), mc["b"].to_numpy(), mc["p"].to_numpy(), mc["w"].to_numpy())
                row |= {f"sens_human_{c}": pc["sens_human"], f"sens_model_{c}": pc["sens_model_at_fpr_human"],
                        f"n_cases_{c}": mc["case_id"].nunique()}
        rows.append(row)
    res = pd.DataFrame(rows)
    out = a.classifier_run / f"human_reference-{a.feature_set.replace(' ', '_').replace('|', '_')}"
    out.mkdir(exist_ok=False)
    res.to_csv(out / "human_vs_model.csv", index=False)
    summ = {"feature_set": a.feature_set, "n_double_read_cases": int(reps["case_id"].nunique()),
            "n_reports": int(reps["report_id"].nunique()), "n_cases_3plus_reports": int((k_case >= 3).sum()),
            "silent_cells_used": len(silent), "n_label_checks": len(chk), "n_label_mismatch": n_mismatch,
            "pair_rows_without_prediction": int(len(pairs) - len(m))}
    write_provenance(out, extra={**summ, "classifier_run": str(a.classifier_run), "parse_run": str(parse_run),
                                 "manifest": str(manifest), "readers": str(readers), "min_n": min_n,
                                 "n_boot": a.n_boot, "seed": a.seed})
    print(json.dumps(summ))
    cols = ["finding_type", "n_cases", "n_case_teeth", "n_pos_b_weighted", "sens_human", "fpr_human",
            "sens_model_at_fpr_human", "d_sens", "d_sens_lo", "d_sens_hi", "auc_model", "n_boot_used"]
    print(res[cols].round(3).to_string(index=False))
    print(res[["finding_type"] + [c for c in res.columns if c.endswith(("_F", "_P", "_S"))]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
