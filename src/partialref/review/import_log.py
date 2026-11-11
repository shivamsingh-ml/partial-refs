"""Validate a review log brought back from Shivam's laptop before it is used (IDs and codes only).

Checks every record strictly: exactly the expected fields (plus an optional "scope": the finding types the
reviewer judged, set by a scoped review list) with the expected types; known (case, report) in
the manifest; parse run = an existing folder under results/extract; lang/half/timestamp in fixed formats;
verdict / flag / finding-type codes and slots from fixed lists; anchors only ids/sides/regions. Rejections
are reported as line numbers and error codes only (content is never echoed). Valid
records are appended to results/review/<parse run>/review_log.jsonl (duplicates by (case, report, logged_at)
skipped); the summary lists rejected lines by number only.

    python -m partialref.review.import_log results/review/incoming/review_log.jsonl \
        --manifest results/manifest/<run>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from partialref.extract.ontology import FindingType
from partialref.review.data import REPORT_FLAGS, VERDICTS

SLOTS = {f"{q}{i}" for q in (1, 2, 3, 4) for i in range(1, 9)} | {
    "Left", "Right", "QuadrantI", "QuadrantII", "QuadrantIII", "QuadrantIV", "UpperArch", "LowerArch"}
TOP = {"logged_at", "parse_run", "lang", "case_id", "report_id", "half", "verdicts", "missed", "flags"}
TYPES = {t.value for t in FindingType}
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,150}$")
# anchors exactly as data.finding_rows builds them: tooth ids, a side, a quadrant or a region
# (v3.2: a region may carry " AllMissing" and the remaining teeth)
_Q = r"Quadrant(?:I|II|III|IV)"
_IDS = r"\d{2}(?:, \d{2}){0,51}"       # at most 32 permanent + 20 primary teeth
ANCHOR_RE = re.compile(rf"^(?:{_IDS}|Left|Right|{_Q}|region (?:UpperArch|LowerArch|{_Q})"
                       rf"(?: AllMissing(?: \(remaining {_IDS}\))?)?)?$")


def _str(x, allowed=None, pattern=None) -> bool:
    return isinstance(x, str) and (allowed is None or x in allowed) and (pattern is None or bool(pattern.match(x)))


def check_record(rec: dict, known: set[tuple[str, str]], known_runs: set[str]) -> list[str]:
    """Error CODES only (never echo content). Strict types: anything unexpected is rejected."""
    errs = []
    if set(rec) not in (TOP, TOP | {"scope"}):
        errs.append("fields")
    sc = rec.get("scope")
    if "scope" in rec:                      # only written by a scoped list: a non-empty list of distinct type codes
        if not (isinstance(sc, list) and sc and all(_str(x, TYPES) for x in sc) and len(set(sc)) == len(sc)):
            errs.append("scope")
        else:                               # nothing judged or entered outside the scope
            inside = set(sc)
            v_out = isinstance(rec.get("verdicts"), list) and any(
                isinstance(e, dict) and e.get("finding_type") not in inside for e in rec["verdicts"])
            m_out = isinstance(rec.get("missed"), list) and any(
                isinstance(e, dict) and e.get("finding_type") not in inside for e in rec["missed"])
            if v_out or m_out:
                errs.append("outside_scope")
    if not (_str(rec.get("case_id")) and _str(rec.get("report_id"))) or (rec["case_id"], rec["report_id"]) not in known:
        errs.append("unknown_report")
    if not _str(rec.get("parse_run"), known_runs, RUN_RE):
        errs.append("unknown_parse_run")
    if not _str(rec.get("lang"), {"en", "it"}):
        errs.append("lang")
    if not _str(rec.get("half"), {"tune", "heldout"}):
        errs.append("half")
    if not _str(rec.get("logged_at"), pattern=TS_RE):
        errs.append("logged_at")
    v = rec.get("verdicts")
    if not isinstance(v, list) or not all(
            isinstance(e, dict) and set(e) == {"idx", "finding_type", "anchor", "verdict", "correct_slot"}
            and isinstance(e["idx"], int) and not isinstance(e["idx"], bool) and 0 <= e["idx"] < 1000
            and _str(e["finding_type"], TYPES) and _str(e["anchor"], pattern=ANCHOR_RE)
            and _str(e["verdict"], set(VERDICTS)) and (e["correct_slot"] is None or _str(e["correct_slot"], SLOTS))
            for e in v):
        errs.append("verdicts")
    m = rec.get("missed")
    if not isinstance(m, list) or not all(
            isinstance(e, dict) and set(e) == {"finding_type", "slots"} and _str(e["finding_type"], TYPES)
            and isinstance(e["slots"], list) and all(_str(x, SLOTS) for x in e["slots"]) for e in m):
        errs.append("missed")
    f = rec.get("flags")
    if not isinstance(f, list) or not all(_str(x, set(REPORT_FLAGS)) for x in f):
        errs.append("flags")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--review-root", default=Path("results/review"), type=Path)
    ap.add_argument("--extract-root", default=Path("results/extract"), type=Path,
                    help="known parse runs = folder names here")
    a = ap.parse_args(argv)
    known = set(map(tuple, pd.read_parquet(a.manifest / "reports.parquet")[["case_id", "report_id"]].to_numpy()))
    known_runs = {d.name for d in a.extract_root.iterdir() if d.is_dir()} if a.extract_root.exists() else set()
    ok, rejected = [], []
    for i, line in enumerate(a.log.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            errs = check_record(rec, known, known_runs) if isinstance(rec, dict) else ["not_object"]
        except json.JSONDecodeError:
            errs = ["not_json"]
        (rejected.append({"line": i, "errors": errs}) if errs else ok.append(rec))
    by_run: dict[str, list[dict]] = {}
    for r in ok:
        by_run.setdefault(r["parse_run"], []).append(r)
    added, root = 0, a.review_root.resolve()
    for run, recs in by_run.items():
        dst = (a.review_root / run / "review_log.jsonl").resolve()
        if root not in dst.parents:                 # belt and braces: RUN_RE already forbids "/" and ".."
            rejected.append({"line": None, "errors": ["path_escape"]})
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        seen = set()
        if dst.exists():
            seen = {(x["case_id"], x["report_id"], x["logged_at"])
                    for x in (json.loads(ln) for ln in dst.read_text().splitlines() if ln.strip())}
        with dst.open("a") as fh:
            for r in recs:
                key = (r["case_id"], r["report_id"], r["logged_at"])
                if key not in seen:
                    fh.write(json.dumps(r) + "\n")
                    seen.add(key)                   # also dedups repeats within the incoming file
                    added += 1
    print(json.dumps({"valid": len(ok), "added": added, "rejected": rejected, "runs": sorted(by_run)}, indent=2))
    return 0 if not rejected else 1


if __name__ == "__main__":
    sys.exit(main())
