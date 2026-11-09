"""Validate a review log brought back from Shivam's laptop before it is used (IDs and codes only).

Checks every record: known (case, report) in the manifest, known parse run, verdict / flag / finding-type
codes from the fixed lists, slots from the fixed slot list, no unexpected fields, no long strings. Valid
records are appended to results/review/<parse run>/review_log.jsonl (duplicates by (case, report, logged_at)
skipped); the summary lists rejected lines by number only.

    python -m partialref.review.import_log results/review/incoming/review_log.jsonl \
        --manifest results/manifest/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from partialref.extract.ontology import FindingType
from partialref.review.data import REPORT_FLAGS, VERDICTS

SLOTS = {f"{q}{i}" for q in (1, 2, 3, 4) for i in range(1, 9)} | {
    "Left", "Right", "QuadrantI", "QuadrantII", "QuadrantIII", "QuadrantIV", "UpperArch", "LowerArch"}
TOP = {"logged_at", "parse_run", "lang", "case_id", "report_id", "half", "verdicts", "missed", "flags"}
TYPES = {t.value for t in FindingType}


def check_record(rec: dict, known: set[tuple[str, str]]) -> list[str]:
    errs = []
    if set(rec) - TOP:
        errs.append(f"unexpected fields {sorted(set(rec) - TOP)}")
    if (rec.get("case_id"), rec.get("report_id")) not in known:
        errs.append("unknown case/report")
    for k in ("logged_at", "parse_run", "lang", "case_id", "report_id", "half"):
        if not isinstance(rec.get(k), str) or len(rec[k]) > 120:
            errs.append(f"bad {k}")
    for v in rec.get("verdicts", []):
        if set(v) - {"idx", "finding_type", "anchor", "verdict", "correct_slot"} or v.get("verdict") not in VERDICTS \
                or v.get("finding_type") not in TYPES or len(str(v.get("anchor", ""))) > 100 \
                or v.get("correct_slot") not in SLOTS | {None}:
            errs.append("bad verdict entry")
    for m in rec.get("missed", []):
        if set(m) - {"finding_type", "slots"} or m.get("finding_type") not in TYPES or not set(m.get("slots", [])) <= SLOTS:
            errs.append("bad missed entry")
    if not set(rec.get("flags", [])) <= set(REPORT_FLAGS):
        errs.append("bad flag")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--review-root", default=Path("results/review"), type=Path)
    a = ap.parse_args(argv)
    known = set(map(tuple, pd.read_parquet(a.manifest / "reports.parquet")[["case_id", "report_id"]].to_numpy()))
    ok, rejected = [], []
    for i, line in enumerate(a.log.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            errs = check_record(rec, known) if isinstance(rec, dict) else ["not an object"]
        except json.JSONDecodeError:
            errs = ["not JSON"]
        (rejected.append({"line": i, "errors": errs}) if errs else ok.append(rec))
    by_run: dict[str, list[dict]] = {}
    for r in ok:
        by_run.setdefault(r["parse_run"], []).append(r)
    added = 0
    for run, recs in by_run.items():
        dst = a.review_root / run / "review_log.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        seen = set()
        if dst.exists():
            seen = {(x["case_id"], x["report_id"], x["logged_at"]) for x in map(json.loads, dst.read_text().splitlines()) if x}
        with dst.open("a") as fh:
            for r in recs:
                if (r["case_id"], r["report_id"], r["logged_at"]) not in seen:
                    fh.write(json.dumps(r) + "\n")
                    added += 1
    print(json.dumps({"valid": len(ok), "added": added, "rejected": rejected, "runs": sorted(by_run)}, indent=2))
    return 0 if not rejected else 1


if __name__ == "__main__":
    sys.exit(main())
