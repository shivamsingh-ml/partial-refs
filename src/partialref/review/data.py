"""Data access for the report review app (partialref.review.app). No Streamlit here, so it is testable.

Reads the manifest (report paths), one parse run (per-report JSON files), an optional fixed review list
(partialref.review.sample) and writes OPTIONAL review logs.
Logs contain IDs and codes only, never report text (DUA). They go to results/review/<parse run>/ which is
git-ignored and not matched by the weekly rsync patterns (*.csv / metrics.json / provenance.json).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from partialref.extract.score_parser import parser_half

VERDICTS = ["correct", "wrong_slot", "not_in_report", "unclear"]
REPORT_FLAGS = ["parser_missed_findings", "translation_looks_odd", "abbreviation_or_wording_issue",
                "category_unclear_to_me", "report_seems_to_belong_elsewhere"]


def load_reports(manifest: Path, lang: str) -> pd.DataFrame:
    r = pd.read_parquet(manifest / "reports.parquet")
    r = r[r["lang"] == lang][["case_id", "report_id", "path"]].copy()
    # review bundle: paths are relative to the bundle folder
    r["path"] = [p if Path(p).is_absolute() else str(manifest / p) for p in r["path"]]
    r["centre"] = r["case_id"].str[0]
    r["half"] = r["case_id"].map(parser_half)
    return r.sort_values(["case_id", "report_id"]).reset_index(drop=True)


def load_parse(parse_run: Path, case_id: str, report_id: str, lang: str) -> dict | None:
    f = parse_run / "per_report" / f"{case_id}__{report_id}__{lang}.json"
    return json.loads(f.read_text()) if f.exists() else None


def finding_rows(extraction: dict | None) -> pd.DataFrame:
    """One row per parsed finding, anchor written as the model gave it (v3.2: an AllMissing region is shown
    as the region statement plus the remaining teeth)."""
    rows = []
    for i, f in enumerate((extraction or {}).get("findings", [])):
        if f.get("region") and f.get("region_scope") == "AllMissing":     # tooth_ids were filled in by code
            rem = ", ".join(f.get("remaining_tooth_ids") or [])
            anchor = f"region {f['region']} AllMissing" + (f" (remaining {rem})" if rem else "")
        else:
            anchor = (", ".join(f.get("tooth_ids") or []) or f.get("side") or f.get("quadrant")
                      or (f"region {f['region']}" if f.get("region") else ""))
        quals = "; ".join(f"{k}={'/'.join(v)}" for k, v in (f.get("qualifiers") or {}).items() if v)
        rows.append({"idx": i, "finding_type": f["finding_type"], "anchor": anchor, "qualifiers": quals})
    return pd.DataFrame(rows, columns=["idx", "finding_type", "anchor", "qualifiers"])


def case_order(reports: pd.DataFrame, half: str, centres: list[str], seed: int, only_multi: bool) -> list[str]:
    """Case IDs to step through: filtered, then shuffled with a fixed seed (no cherry-picking)."""
    r = reports[reports["centre"].isin(centres)]
    if half != "all":
        r = r[r["half"] == half]
    n = r.groupby("case_id")["report_id"].nunique()
    cases = sorted(n[n >= 2].index if only_multi else n.index)
    return pd.Series(cases).sample(frac=1, random_state=seed).tolist() if cases else []


def load_review_list(path: Path, reports: pd.DataFrame) -> list[dict]:
    """Entries of a fixed review list (partialref.review.sample), checked strictly: IDs known to the manifest,
    strata and finding types from fixed lists. The list holds IDs and codes only."""
    from partialref.extract.ontology import FindingType
    types = {t.value for t in FindingType}
    known = set(map(tuple, reports[["case_id", "report_id"]].to_numpy()))
    entries = json.loads(Path(path).read_text())["entries"]
    for e in entries:
        jt = e.get("judge_types")
        if (set(e) != {"case_id", "report_id", "stratum", "judge_types"} or (e["case_id"], e["report_id"]) not in known
                or e["stratum"] not in ("base", "extra") or (e["stratum"] == "base") != (jt is None)
                or (jt is not None and not (isinstance(jt, list) and jt and set(jt) <= types))):
            raise ValueError("review list: bad entry (unknown report, stratum or finding type)")
    if len({(e["case_id"], e["report_id"]) for e in entries}) != len(entries):
        raise ValueError("review list: a report is listed twice")
    return entries


def log_review(log_dir: Path, rec: dict) -> Path:
    """Append one review record (IDs and codes only) as a JSON line; returns the file."""
    bad = [k for k, v in rec.items() if isinstance(v, str) and len(v) > 120]
    if bad:                                                   # guard: no long free text can be logged
        raise ValueError(f"refusing to log long text fields: {bad}")
    log_dir.mkdir(parents=True, exist_ok=True)
    f = log_dir / "review_log.jsonl"
    with f.open("a") as fh:
        fh.write(json.dumps({"logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}) + "\n")
    return f


def read_log(log_dir: Path) -> pd.DataFrame:
    f = log_dir / "review_log.jsonl"
    if not f.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(x) for x in f.read_text().splitlines() if x.strip()])
