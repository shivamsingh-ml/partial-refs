"""Parser checks (GT/T1).

1. vs Lumetti et al.'s per-patient labels (--lumetti): Lumetti's labels consolidate all reports of a
   patient, so we compare the *union* of our per-report instances per case, at type level (case, type)
   and slot level (case, type, slot).
2. vs another parse of the same reports (--ref-parse; run-to-run or EN-vs-IT): per report
   (case, report, type, slot) — what the propensity model consumes — plus the per-case union.

Scored universe (independent of who found what, so empty outputs count as misses / false positives):
- our cases: every report of the case parsed without error (from extractions_<lang>.parquet next to
  the instances file); cases with a failed report are dropped, as in T2;
- Lumetti cases: every TTL file (cases.parquet of the Lumetti run; fallback: case ids of findings.parquet);
- per-report level: reports parsed without error in both runs.
Lumetti type level uses findings.parquet, so findings without an anchor (no tooth/side/quadrant) still count.

Usage:
    python -m partialref.extract.score_parser --ours results/extract/<run>/instances_en.parquet \
        --lumetti results/lumetti/<run>/instances.parquet [--exclude P116,P117] [--half heldout]
    python -m partialref.extract.score_parser --ours <run>/repeat/instances_en.parquet --ref-parse <run>/instances_en.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from partialref.extract.ontology import REGION_PREFIX
from partialref.extract.per_report import report_ok
from partialref.provenance import make_run_id, write_provenance

LEVEL_COLS = {"type": ["case_id", "finding_type"], "slot": ["case_id", "finding_type", "slot"],
              "report_slot": ["case_id", "report_id", "finding_type", "slot"]}


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (p == p and r == r and p + r) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def parser_half(case_id: str) -> str:
    """Fixed split of cases for parser development (decided 2026-11-08): 'tune' cases may be inspected
    to change the prompt; T1 is reported on 'heldout'. Stable hash, independent of run order."""
    return "tune" if int(hashlib.sha256(f"parser-split:{case_id}".encode()).hexdigest(), 16) % 2 == 0 else "heldout"


def extractions_for(instances_path: Path) -> pd.DataFrame:
    """extractions_<lang>.parquet that sits next to instances_<lang>.parquet."""
    return pd.read_parquet(instances_path.with_name(instances_path.name.replace("instances_", "extractions_", 1)))


def fully_parsed_cases(meta: pd.DataFrame) -> set[str]:
    ok = report_ok(meta).groupby(meta["case_id"]).all()
    return set(ok[ok].index)


def ok_reports(meta: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, meta.loc[report_ok(meta), ["case_id", "report_id"]].to_numpy()))


def lumetti_cases(instances_path: Path) -> set[str]:
    d = instances_path.parent
    if (d / "cases.parquet").exists():
        return set(pd.read_parquet(d / "cases.parquet")["case_id"])
    if (d / "findings.parquet").exists():
        return set(pd.read_parquet(d / "findings.parquet")["case_id"])
    return set(pd.read_parquet(instances_path)["case_id"])     # e.g. another parse passed as --lumetti


def lumetti_types(instances_path: Path) -> pd.DataFrame:
    f = instances_path.parent / "findings.parquet"
    if not f.exists():
        return pd.read_parquet(instances_path)[["case_id", "finding_type"]].drop_duplicates()
    x = pd.read_parquet(f)
    return x[~x["unknown_type"].astype(bool)][["case_id", "finding_type"]].drop_duplicates()


def score(ours: pd.DataFrame, ref: pd.DataFrame, level: str, keep: set) -> pd.DataFrame:
    """level: 'type' | 'slot' (keep = case ids) | 'report_slot' (keep = (case, report) pairs)."""
    cols = LEVEL_COLS[level]

    def rows_of(df: pd.DataFrame) -> set[tuple]:
        if level == "report_slot":
            sel = [k in keep for k in zip(df["case_id"], df["report_id"])]
        else:
            sel = df["case_id"].isin(keep)
        return set(map(tuple, df.loc[sel, cols].drop_duplicates().to_numpy()))

    o, r = rows_of(ours), rows_of(ref)
    ti = cols.index("finding_type")
    rows = []
    for t in sorted({x[ti] for x in o | r}):
        ot, rt = {x for x in o if x[ti] == t}, {x for x in r if x[ti] == t}
        rows.append({"finding_type": t, "n_ref": len(rt), **prf(len(ot & rt), len(ot - rt), len(rt - ot))})
    rows.append({"finding_type": "ALL (micro)", "n_ref": len(r), **prf(len(o & r), len(o - r), len(r - o))})
    n_cases = len({k[0] for k in keep}) if level == "report_slot" else len(keep)
    return pd.DataFrame(rows).assign(level=level, n_cases=n_cases,
                                     n_reports=len(keep) if level == "report_slot" else None)


def scoring_universe(ours_path: Path, lumetti: Path | None = None, ref_parse: Path | None = None,
                     exclude: list[str] = (), half: str = "all") -> tuple[set[str], set[tuple[str, str]], dict]:
    """(cases, reports, info) to score; see module docstring."""
    m_ours = extractions_for(ours_path)
    cases = fully_parsed_cases(m_ours)
    reports = ok_reports(m_ours)
    info = {"our_cases": int(m_ours["case_id"].nunique()), "our_cases_with_failed_report": int(
        m_ours["case_id"].nunique() - len(cases))}
    if lumetti is not None:
        lc = lumetti_cases(lumetti)
        info["our_fully_parsed_cases_without_lumetti_file"] = len(cases - lc)
        cases &= lc
    if ref_parse is not None:
        m_ref = extractions_for(ref_parse)
        cases &= fully_parsed_cases(m_ref)
        reports &= ok_reports(m_ref)
    keep_case = (lambda c: c not in exclude and (half == "all" or parser_half(c) == half))
    cases = {c for c in cases if keep_case(c)}
    reports = {(c, r) for c, r in reports if keep_case(c)}
    return cases, reports, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, type=Path, help="instances_<lang>.parquet of a parse run")
    ref = ap.add_mutually_exclusive_group(required=True)
    ref.add_argument("--lumetti", type=Path, help="Lumetti run instances.parquet (per-patient union)")
    ref.add_argument("--ref-parse", type=Path, help="instances_<lang>.parquet of another parse (per report)")
    ap.add_argument("--exclude", type=lambda s: [x for x in s.split(",") if x], default=[])
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="parser-check")
    ap.add_argument("--half", choices=["all", "tune", "heldout"], default="all")
    a = ap.parse_args(argv)
    ours = pd.read_parquet(a.ours)
    cases, reports, info = scoring_universe(a.ours, a.lumetti, a.ref_parse, a.exclude, a.half)
    if a.lumetti is not None:
        ref_inst = pd.read_parquet(a.lumetti)
        # region-level mentions (no tooth numbers) count at type level only; Lumetti has no such slots
        ours_slots = ours[~ours["slot"].astype(str).str.startswith(REGION_PREFIX)]
        res = pd.concat([score(ours, lumetti_types(a.lumetti), "type", cases),
                         score(ours_slots, ref_inst, "slot", cases)], ignore_index=True)
        headline = "slot"
    else:
        ref_inst = pd.read_parquet(a.ref_parse)
        res = pd.concat([score(ours, ref_inst, "report_slot", reports),
                         score(ours, ref_inst, "slot", cases)], ignore_index=True)
        headline = "report_slot"
    run_id = make_run_id(a.tag)
    out = a.results_dir / "extract" / run_id
    out.mkdir(parents=True, exist_ok=False)
    res.to_csv(out / "parser_vs_lumetti.csv" if a.lumetti is not None else out / "parser_vs_parse.csv", index=False)
    h = res[res.level == headline].set_index("finding_type")
    summary = {"run_id": run_id, "half": a.half, "reference": "lumetti" if a.lumetti is not None else "parse",
               "headline_level": headline, "n_cases": len(cases), "n_reports": len(reports), **info,
               "excluded": a.exclude, f"{headline}_f1": h["f1"].round(3).to_dict(),
               f"{headline}_n_ref": h["n_ref"].to_dict()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_provenance(out, extra={"run_id": run_id, "ours": str(a.ours), "lumetti": str(a.lumetti),
                                 "ref_parse": str(a.ref_parse), "half": a.half})
    print(res.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith(("_f1", "_n_ref"))}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
