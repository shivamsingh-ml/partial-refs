"""Does each report belong to its scan? Missing-teeth fingerprint (screening, counts and IDs only).

For every parsed report r and every scan s (permanent teeth 11-48, report FDI convention):
- report says ABSENT:  ToothAbsence, DentalImplant, PostExtractionSocket slots;
- report says PRESENT: slots of tooth-level findings (Caries, Endo, Restoration, TDP, Periapical);
- scan: tooth label present (T) and position assessable (arch in the scan + HANDOFF §2.6 missing-tooth
  rule, as in partialref.propensity.agreement).
agree   = |ABSENT & assessable & ~T| + |PRESENT & T|
contra  = |ABSENT & T|               + |PRESENT & assessable & ~T|
rate    = contra / (agree + contra)  (NaN without evidence).
Each report is scored against its own scan (as used: P/F/S mirrored), its own scan with the opposite
mirroring, and every other scan. Flags are a SCREEN for a human check, not exclusion rules:
- flipped:  the opposite mirroring removes >= --min-contra-drop contradictions and lowers the rate;
- misfiled: own rate >= --min-own-rate with >= --min-contra contradictions, and some other scan with
  enough evidence (>= --min-evidence and >= half the own evidence) has rate <= own rate / 2.
own_rank = rank of the own scan among all scans with enough evidence (1 = best fit; ties share the best rank).
P116/P117/P260/P261 (reports believed misfiled, DECISIONS) are kept in on purpose as positive controls.

Usage:
    python -m partialref.audit.report_scan_match --parse-run results/extract/<run> \
        --vis results/audit/<run> [--vis ...] [--lang en] [--mirror-centres PFS]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from partialref.extract.per_report import report_ok
from partialref.propensity.agreement import NEEDS_TOOTH, PERMANENT, is_visible, load_visibility
from partialref.provenance import make_run_id, write_provenance

ABSENT_TYPES = {"ToothAbsence", "DentalImplant", "PostExtractionSocket"}
IDX = {t: i for i, t in enumerate(PERMANENT)}


def scan_matrices(vis: dict[str, dict]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """cases, T (tooth present) and A (position assessable for an absence call), each n_cases x 32."""
    cases = sorted(vis)
    T = np.zeros((len(cases), 32), bool)
    A = np.zeros((len(cases), 32), bool)
    for i, c in enumerate(cases):
        for t in PERMANENT:
            T[i, IDX[t]] = t in vis[c]["teeth"]
            # previous visibility rule pinned, so earlier report-scan-match results stay reproducible
            A[i, IDX[t]] = T[i, IDX[t]] or is_visible(vis[c], "ToothAbsence", t, upper_gap_rule="teeth_or_bone")
    return cases, T, A


def report_vectors(m: pd.DataFrame, reports: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """ABSENT and PRESENT claims per report (rows in the order of ``reports``)."""
    pos = {r: i for i, r in enumerate(reports)}
    ab = np.zeros((len(reports), 32), bool)
    pr = np.zeros((len(reports), 32), bool)
    m = m[m["slot"].isin(IDX)]
    for c, r, ft, s in m[["case_id", "report_id", "finding_type", "slot"]].itertuples(index=False):
        i = pos.get((c, r))
        if i is None:
            continue
        if ft in ABSENT_TYPES:
            ab[i, IDX[s]] = True
        elif ft in NEEDS_TOOTH:
            pr[i, IDX[s]] = True
    return ab, pr


def score(ab: np.ndarray, pr: np.ndarray, T: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """agree, contra: n_reports x n_scans."""
    ab, pr, T, A = (x.astype(np.int32) for x in (ab, pr, T, A))
    nt = A * (1 - T)                                   # assessable and no tooth
    agree = ab @ nt.T + pr @ T.T
    contra = ab @ T.T + pr @ nt.T
    return agree, contra


def match_reports(m: pd.DataFrame, reports: list[tuple[str, str]], vis: dict, vis_opp: dict,
                  min_evidence: int = 6, min_contra: int = 3, min_own_rate: float = 0.25,
                  min_contra_drop: int = 2) -> pd.DataFrame:
    cases, T, A = scan_matrices(vis)
    ci = {c: i for i, c in enumerate(cases)}
    ab, pr = report_vectors(m, reports)
    agree, contra = score(ab, pr, T, A)
    ev = agree + contra
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(ev > 0, contra / np.maximum(ev, 1), np.nan)
    _, To, Ao = scan_matrices(vis_opp)          # same case order (same keys)
    rows = []
    for i, (c, r) in enumerate(reports):
        if c not in ci:
            continue
        j = ci[c]
        oa, oc = score(ab[i:i + 1], pr[i:i + 1], To[j:j + 1], Ao[j:j + 1])
        oa, oc = int(oa[0, 0]), int(oc[0, 0])
        own_ev, own_rate = int(ev[i, j]), rate[i, j]
        opp_rate = oc / (oa + oc) if oa + oc else np.nan
        ok = (ev[i] >= max(min_evidence, own_ev / 2))
        ok[j] = True
        cand = np.where(ok)[0]
        others = cand[cand != j]
        if len(others) and np.isfinite(rate[i, others]).any():
            k = others[np.nanargmin(rate[i, others])]
            best_case, best_rate, best_ev = cases[k], rate[i, k], int(ev[i, k])
        else:
            best_case, best_rate, best_ev = None, np.nan, 0
        cr = rate[i, cand]
        own_rank = int(np.sum(cr[np.isfinite(cr)] < own_rate) + 1) if np.isfinite(own_rate) else None
        rows.append({
            "case_id": c, "report_id": r, "n_absent_claims": int(ab[i].sum()), "n_present_claims": int(pr[i].sum()),
            "own_agree": int(agree[i, j]), "own_contra": int(contra[i, j]), "own_evidence": own_ev,
            "own_rate": own_rate, "opp_agree": oa, "opp_contra": oc, "opp_rate": opp_rate,
            "best_other_case": best_case, "best_other_rate": best_rate, "best_other_evidence": best_ev,
            "own_rank": own_rank, "n_candidate_scans": int(len(cand)),
            "flag_flipped": bool(int(contra[i, j]) - oc >= min_contra_drop and
                                 np.isfinite(opp_rate) and opp_rate < own_rate),
            "flag_misfiled": bool(own_ev >= min_evidence and int(contra[i, j]) >= min_contra
                                  and own_rate >= min_own_rate and np.isfinite(best_rate)
                                  and best_rate <= own_rate / 2),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse-run", required=True, type=Path)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--vis", required=True, type=Path, action="append")
    ap.add_argument("--mirror-centres", default="PFS")
    ap.add_argument("--min-evidence", type=int, default=6)
    ap.add_argument("--min-contra", type=int, default=3)
    ap.add_argument("--min-own-rate", type=float, default=0.25)
    ap.add_argument("--min-contra-drop", type=int, default=2)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="report-scan-match")
    a = ap.parse_args(argv)

    meta = pd.read_parquet(a.parse_run / f"extractions_{a.lang}.parquet")
    ok = meta[report_ok(meta)]
    reports = sorted(set(map(tuple, ok[["case_id", "report_id"]].to_numpy())))
    m = pd.read_parquet(a.parse_run / f"instances_{a.lang}.parquet")
    vis = load_visibility(a.vis, mirror_centres=a.mirror_centres)
    opp = "".join(sorted(set("PFSA") - set(a.mirror_centres)))   # every centre flipped the other way
    vis_opp = load_visibility(a.vis, mirror_centres=opp)
    res = match_reports(m, reports, vis, vis_opp, a.min_evidence, a.min_contra, a.min_own_rate, a.min_contra_drop)

    out = a.results_dir / "audit" / make_run_id(a.tag)
    out.mkdir(parents=True, exist_ok=False)
    res.to_parquet(out / "report_scan_match.parquet", index=False)
    res["centre"] = res["case_id"].str[0]
    by_case = res.groupby("case_id").agg(n=("report_id", "size"), flipped=("flag_flipped", "sum"),
                                         misfiled=("flag_misfiled", "sum"))
    flagged = res[res["flag_flipped"] | res["flag_misfiled"]]
    summary = {
        "n_reports_scored": len(res), "n_reports_without_scan": len(reports) - len(res),
        "own_rate_median_by_centre": res.groupby("centre")["own_rate"].median().round(3).to_dict(),
        "own_rank_share_top1_by_centre": res.assign(top=res["own_rank"] == 1).groupby("centre")["top"]
        .mean().round(3).to_dict(),
        "flag_flipped_reports": int(res["flag_flipped"].sum()), "flag_misfiled_reports": int(res["flag_misfiled"].sum()),
        "cases_all_reports_flipped": by_case[(by_case["flipped"] == by_case["n"])].index.tolist(),
        "flagged": flagged[["case_id", "report_id", "own_evidence", "own_contra", "own_rate", "opp_rate",
                            "best_other_case", "best_other_rate", "own_rank", "flag_flipped", "flag_misfiled"]]
        .round(3).to_dict(orient="records"),
        "positive_controls": res[res["case_id"].isin(["P116", "P117", "P260", "P261"])]
        [["case_id", "report_id", "own_rate", "best_other_case", "best_other_rate", "own_rank", "flag_misfiled"]]
        .round(3).to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_provenance(out, extra={"parse_run": str(a.parse_run), "lang": a.lang,
                                 "vis": [str(v) for v in a.vis], "mirror_centres": a.mirror_centres,
                                 "thresholds": {k: getattr(a, k) for k in
                                                ("min_evidence", "min_contra", "min_own_rate", "min_contra_drop")}})
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
