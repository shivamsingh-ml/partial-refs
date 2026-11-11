"""Do per-tooth intensity features detect fillings, crowns, root canal fillings, root-tip infection and
caries? (PAPER_PLAN step 1, decision D1; 2026-11-12)

Labels come from the reports, per tooth and finding type:
- positive: at least one report of the case, by an author who reports this type, mentions it on this tooth;
- negative: at least one such report exists for the case and none of them mentions it;
- unknown (left out): the case has no report by an author who reports this type.
"Author who reports this type" = not flagged silent in the reader profiles (rate_hi < 0.10 with at least
--min-n partner mentions; partialref.propensity.readers). So a report by an author who never writes about
fillings is never used as a "no filling" example.

Evaluation: the five development folds of the split (cases of fold k are scored by a model trained on the
other folds), per type and per centre: AUC and average precision of (a) gradient boosting on all features +
reference variables, (b) the reference only: tooth position, upper/lower and centre (knows nothing about
the image). Truncated teeth and the default exclusions are left out. The reader profiles must come from
development cases only (partialref.propensity.readers --splits ... --role dev). Labels from reports
are incomplete, so these numbers are lower bounds on what the features can do; that is stated, not fixed.

    python -m partialref.perception.tooth_classifier --features results/perception/<run> \\
        --parse-run results/extract/<run> --manifest results/manifest/<run> --readers results/propensity/<run> \\
        --splits configs/splits/v1/splits.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from partialref.propensity.agreement import select_reports
from partialref.provenance import make_run_id, write_provenance

TYPES = ["ConservativeRestoration", "ProstheticDevice", "EndodonticTreatment", "PeriapicalLesion", "Caries"]
FEATURES = ["n_vox", "pulp_share", "length_mm", "tooth_median_rel_max", "bright_crown", "bright_root",
            "bright_crown_surface", "bright_crown_interior", "bright_pulp", "bright_root_pulp", "bright_root_inner",
            "dark_crown", "prosthetic_label_near_crown", "implant_label_near_root", "apex_p20_rel_bone",
            "apex_dark_share", "apex_rel_local", "apex_dark_share_local", "bright_crown_deep",
            "largest_deep_bright_mm3", "end_bone_contrast", "n_saturated"]
REFERENCE = ["pos_in_quadrant", "upper", "centre_code"]
EXCLUDE = ["P116", "P117", "P260", "P261", "S0015"]


def _author(x) -> str:
    """Author ids as plain strings ('14', not '14.0')."""
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x)


def silent_authors(profiles: pd.DataFrame, min_n: int) -> set[tuple[str, str, str]]:
    """(centre, author, type) that clearly do not report the type."""
    p = profiles[(profiles["both"] + profiles["only_partner"]) >= min_n]
    p = p[p["rate_hi"] < 0.10]
    return {(c, _author(a), t) for c, a, t in p[["centre", "author", "finding_type"]].itertuples(index=False)}


def tooth_labels(teeth: pd.DataFrame, reps: pd.DataFrame, inst: pd.DataFrame, silent: set) -> pd.DataFrame:
    """teeth: case_id, fdi_report. reps: case_id, report_id, report_x. inst: parser instances.
    Returns teeth x TYPES with 1 / 0 / NaN (unknown)."""
    r = reps.assign(author=reps["report_x"].map(_author), centre=reps["case_id"].str[0])
    ment = inst[inst["finding_type"].isin(TYPES)][["case_id", "report_id", "finding_type", "slot"]].drop_duplicates()
    ment = ment[ment["slot"].astype(str).str.fullmatch(r"\d\d")].assign(fdi=lambda x: x["slot"].astype(int))
    ment = ment.merge(r[["case_id", "report_id", "author", "centre"]], on=["case_id", "report_id"])
    out = teeth[["case_id", "fdi_report"]].copy()
    for t in TYPES:
        ok_reports = r.loc[np.array([(c, a, t) not in silent for c, a in zip(r["centre"], r["author"])], dtype=bool)]
        covered = set(ok_reports["case_id"])
        okm = ment[(ment["finding_type"] == t)]
        okm = okm.loc[np.array([(c, a, t) not in silent for c, a in zip(okm["centre"], okm["author"])], dtype=bool)]
        pos = set(zip(okm["case_id"], okm["fdi"]))
        out[t] = [1.0 if (c, f) in pos else (0.0 if c in covered else np.nan)
                  for c, f in zip(out["case_id"], out["fdi_report"])]
    return out


def evaluate(df: pd.DataFrame, folds: dict[str, int], seed: int = 0) -> pd.DataFrame:
    df = df.assign(fold=df["case_id"].map(folds), pos_in_quadrant=df["fdi_report"] % 10,
                   upper=(df["fdi_report"] // 10).isin([1, 2]).astype(int),
                   centre_code=df["case_id"].str[0].map({"F": 0, "P": 1, "S": 2}))
    rows = []
    for t in TYPES:
        d = df[df[t].notna() & df["fold"].notna()]
        if d[t].nunique() < 2:
            continue
        pred_full, pred_pos = np.full(len(d), np.nan), np.full(len(d), np.nan)
        for k in sorted(d["fold"].unique()):
            tr, te = d["fold"] != k, d["fold"] == k
            if d.loc[tr, t].nunique() < 2:
                continue
            for cols, store in ((FEATURES + REFERENCE, pred_full), (REFERENCE, pred_pos)):
                m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
                m.fit(d.loc[tr, cols], d.loc[tr, t])
                store[te.to_numpy()] = m.predict_proba(d.loc[te, cols])[:, 1]
        ok = ~np.isnan(pred_full)
        cen = d["case_id"].str[0].to_numpy()
        for centre in ["all", "F", "P", "S"]:
            sel = ok & ((cen == centre) if centre != "all" else True)
            y = d[t].to_numpy()[sel]
            if len(y) == 0 or len(np.unique(y)) < 2:
                continue
            rows.append({"finding_type": t, "centre": centre, "n_teeth": int(sel.sum()), "n_pos": int(y.sum()),
                         "prevalence": float(y.mean()),
                         "auc_features": roc_auc_score(y, pred_full[sel]), "ap_features": average_precision_score(y, pred_full[sel]),
                         "auc_reference": roc_auc_score(y, pred_pos[sel]), "ap_reference": average_precision_score(y, pred_pos[sel]),
                         "n_unknown_teeth": int(df[t].isna().sum()) if centre == "all" else None})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--parse-run", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--readers", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="tooth-classifier")
    a = ap.parse_args(argv)
    feats = pd.read_parquet(a.features / "tooth_features.parquet")
    feats = feats[feats["error"].isna() & feats["fdi_mask"].notna()].copy()
    feats["fdi_report"] = feats["fdi_report"].astype(int)
    sp = pd.read_csv(a.splits)
    dev = sp[sp["role"] == "dev"]
    feats = feats[feats["case_id"].isin(set(dev["case_id"]))]            # never the locked test set
    reps, bad = select_reports(a.manifest, a.parse_run, "en", "PFS", EXCLUDE)
    bad_cases = {c for c, _ in bad}
    reps = reps[~reps["case_id"].isin(bad_cases)][["case_id", "report_id", "report_x"]]
    feats = feats[~feats["case_id"].isin(bad_cases | set(EXCLUDE)) & ~feats["truncated"].astype(bool)]
    inst = pd.read_parquet(a.parse_run / "instances_en.parquet")
    silent = silent_authors(pd.read_csv(a.readers / "profiles.csv"), a.min_n)
    labels = tooth_labels(feats, reps, inst, silent)
    df = feats.merge(labels, on=["case_id", "fdi_report"])
    res = evaluate(df, dict(zip(dev["case_id"], dev["fold"])))
    run_id = make_run_id(a.tag)
    out = a.results_dir / "perception" / run_id
    out.mkdir(parents=True, exist_ok=False)
    res.to_csv(out / "auc_by_type.csv", index=False)
    labels.to_parquet(out / "tooth_labels.parquet", index=False)
    summ = {"run_id": run_id, "n_teeth": len(df), "n_cases": int(df["case_id"].nunique()), "silent_cells_used": len(silent)}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    write_provenance(out, extra={**summ, "features": str(a.features), "readers": str(a.readers), "splits": str(a.splits),
                                 "parse_run": str(a.parse_run), "manifest": str(a.manifest), "min_n": a.min_n,
                                 "excluded": EXCLUDE})
    print(res.round(3).to_string(index=False))
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
