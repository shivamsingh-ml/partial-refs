"""Parser check (GT/T1): our per-report findings, merged per case, vs Lumetti's per-patient labels.

Lumetti's labels consolidate all reports of a patient, so we compare the *union* of our per-report
instances per case. Two levels, per finding type: type level (case, type) and slot level
(case, type, slot). Only cases present in both sources are scored.

Usage:
    python -m partialref.extract.score_parser --ours results/extract/<run>/instances.parquet \
        --lumetti results/lumetti/<run>/instances.parquet [--exclude P116,P117]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from partialref.provenance import make_run_id, write_provenance


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (p == p and r == r and p + r) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def score(ours: pd.DataFrame, ref: pd.DataFrame, level: str) -> pd.DataFrame:
    """ours/ref: columns case_id, finding_type, slot. level: 'type' or 'slot'."""
    cases = sorted(set(ours["case_id"]) & set(ref["case_id"]))
    cols = ["case_id", "finding_type"] + (["slot"] if level == "slot" else [])
    o = set(map(tuple, ours[ours.case_id.isin(cases)][cols].drop_duplicates().to_numpy()))
    r = set(map(tuple, ref[ref.case_id.isin(cases)][cols].drop_duplicates().to_numpy()))
    rows = []
    for t in sorted({x[1] for x in o | r}):
        ot, rt = {x for x in o if x[1] == t}, {x for x in r if x[1] == t}
        rows.append({"finding_type": t, "n_ref": len(rt), **prf(len(ot & rt), len(ot - rt), len(rt - ot))})
    rows.append({"finding_type": "ALL (micro)", "n_ref": len(r), **prf(len(o & r), len(o - r), len(r - o))})
    return pd.DataFrame(rows).assign(level=level, n_cases=len(cases))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, type=Path)
    ap.add_argument("--lumetti", required=True, type=Path)
    ap.add_argument("--exclude", type=lambda s: [x for x in s.split(",") if x], default=[])
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="parser-check")
    a = ap.parse_args(argv)
    ours = pd.read_parquet(a.ours)
    ref = pd.read_parquet(a.lumetti)
    ours, ref = ours[~ours.case_id.isin(a.exclude)], ref[~ref.case_id.isin(a.exclude)]
    res = pd.concat([score(ours, ref, "type"), score(ours, ref, "slot")], ignore_index=True)
    run_id = make_run_id(a.tag)
    out = a.results_dir / "extract" / run_id
    out.mkdir(parents=True, exist_ok=False)
    res.to_csv(out / "parser_vs_lumetti.csv", index=False)
    slot = res[res.level == "slot"].set_index("finding_type")
    summary = {"run_id": run_id, "n_cases": int(res["n_cases"].iloc[0]), "excluded": a.exclude,
               "slot_f1": slot["f1"].round(3).to_dict(), "slot_n_ref": slot["n_ref"].to_dict()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_provenance(out, extra={"run_id": run_id, "ours": str(a.ours), "lumetti": str(a.lumetti)})
    print(res.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
