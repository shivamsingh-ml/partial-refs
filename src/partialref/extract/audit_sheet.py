"""Blinded audit sheet for a human check of suspicious parser instances (no report text is read here).

Builds a stratified random sample of (case, report, finding type, tooth/slot) items from one parse:
- left_right_swap, neighbour_shift: reader-pair mismatches (partialref.propensity.laterality), both sides;
- residual_one_reader, residual_multi_reader: tooth findings on teeth the (mirrored) mask calls absent;
- this_parser_only / other_parser_only: instances only one of two parses produced (needs --other-parse);
- not_in_lumetti: (case, type, slot) absent from Lumetti et al.'s per-patient labels (needs --lumetti).
The reviewer opens each report locally and fills `verdict`:
  report_says         the report states this finding at this tooth/slot
  parser_wrong_slot   the report states the finding, but at another tooth/slot (write it in correct_slot)
  not_in_report       the report does not state this finding here at all
  unclear             wording ambiguous
The sheet does not show why an item was sampled (blinding); the reasons are in key.parquet (an item
picked for several reasons keeps all of them). Left/right pairs appear as two items of the same case.
DUA: no free-text column (verdict is a code, correct_slot an FDI number / Left / Right / QuadrantI-IV, checked
when the sheet is read back); written as .tsv, which the weekly results rsync (*.csv) does not copy.

Usage:
    python -m partialref.extract.audit_sheet --parse-run results/extract/<run> --manifest results/manifest/<run> \
        --vis results/audit/<run> [--other-parse results/extract/<run>] [--lumetti results/lumetti/<run>/instances.parquet] \
        [--per-reason 10] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from partialref.extract.ontology import REGION_PREFIX
from partialref.propensity.agreement import load_visibility, select_reports
from partialref.propensity.laterality import residual_check, swap_check
from partialref.provenance import make_run_id, write_provenance

KEY = ["case_id", "report_id", "finding_type", "slot"]
RESIDUAL_REASON = {"one_of_several_readers": "residual_one_reader", "multiple_readers": "residual_multi_reader"}
VERDICTS = ["report_says", "parser_wrong_slot", "not_in_report", "unclear"]


def candidate_items(m: pd.DataFrame, reports_by_case: dict, vis_mirror: dict,
                    other: pd.DataFrame | None = None, lumetti: pd.DataFrame | None = None) -> pd.DataFrame:
    """All candidate items with their reason. m / other: deduplicated mentions of usable reports."""
    n_by_case = {c: len(set(r)) for c, r in reports_by_case.items()}
    rows: list[dict] = []
    pairs: list[dict] = []
    swap_check(m, reports_by_case, collect=pairs)
    for i, p in enumerate(pairs):
        for side in ("a", "b"):
            rows.append({"case_id": p["case_id"], "report_id": p[f"report_{side}"], "finding_type": p["finding_type"],
                         "slot": p[f"slot_{side}"], "reason": p["kind"], "pair_id": f"pair{i}"})
    res = residual_check(m, n_by_case, vis_mirror)
    if len(res):
        hit = m.merge(res[["case_id", "finding_type", "slot", "category"]], on=["case_id", "finding_type", "slot"])
        hit = hit[hit["category"] != "single_read_case"]
        rows += [{**{k: r[k] for k in KEY}, "reason": RESIDUAL_REASON[r["category"]], "pair_id": None}
                 for r in hit.to_dict("records")]
    if other is not None:
        both = m[KEY].merge(other[KEY], how="outer", indicator=True)
        for flag, reason in (("left_only", "this_parser_only"), ("right_only", "other_parser_only")):
            rows += [{**r, "reason": reason, "pair_id": None}
                     for r in both[both["_merge"] == flag][KEY].to_dict("records")]
    if lumetti is not None:
        lum = lumetti[["case_id", "finding_type", "slot"]].drop_duplicates().assign(_in=True)
        x = m.merge(lum, on=["case_id", "finding_type", "slot"], how="left")
        x = x[x["case_id"].isin(set(lumetti["case_id"])) & x["_in"].isna()   # only cases Lumetti labelled
              & ~x["slot"].astype(str).str.startswith(REGION_PREFIX)]            # Lumetti has no region slots
        rows += [{**r, "reason": "not_in_lumetti", "pair_id": None} for r in x[KEY].to_dict("records")]
    return pd.DataFrame(rows, columns=KEY + ["reason", "pair_id"])


def sample_items(cand: pd.DataFrame, per_reason: int, seed: int) -> pd.DataFrame:
    """Up to per_reason items per reason (pairs are kept whole); an item sampled twice is listed once."""
    rng = np.random.default_rng(seed)
    picked = []
    for reason, g in cand.groupby("reason", sort=True):
        if g["pair_id"].notna().any():
            ids = sorted(g["pair_id"].dropna().unique())
            keep = rng.choice(ids, size=min(per_reason // 2 or 1, len(ids)), replace=False)
            picked.append(g[g["pair_id"].isin(keep)])
        else:
            picked.append(g.iloc[rng.choice(len(g), size=min(per_reason, len(g)), replace=False)])
    s = pd.concat(picked) if picked else cand.iloc[0:0]
    # one item per (case, report, type, slot); keep every reason it was sampled for (review #10)
    s = s.groupby(KEY, as_index=False, sort=True).agg(
        reason=("reason", lambda r: ";".join(sorted(set(r)))),
        pair_id=("pair_id", lambda p: ";".join(sorted({x for x in p if isinstance(x, str)})) or None))
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)          # mix reasons (blinding)
    s.insert(0, "item_id", [f"item{i:03d}" for i in range(len(s))])
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse-run", required=True, type=Path)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--vis", required=True, type=Path, action="append")
    ap.add_argument("--centres", default="PFS")
    ap.add_argument("--mirror-centres", default="PFS")
    ap.add_argument("--exclude", type=lambda s: [x for x in s.split(",") if x],
                    default=["P116", "P117", "P260", "P261", "S0015"])
    ap.add_argument("--other-parse", type=Path, help="second parse run (same language) for disagreements")
    ap.add_argument("--lumetti", type=Path, help="Lumetti et al. instances.parquet (internal check only)")
    ap.add_argument("--per-reason", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="parser-audit-sheet")
    a = ap.parse_args(argv)

    reps, bad_reports = select_reports(a.manifest, a.parse_run, a.lang, a.centres, a.exclude)
    if a.other_parse:
        _, bad2 = select_reports(a.manifest, a.other_parse, a.lang, a.centres, a.exclude)
        bad_reports |= bad2
    vis = load_visibility(a.vis, mirror_centres=a.mirror_centres)
    bad = {c for c, _ in bad_reports}
    rep = reps[~reps["case_id"].isin(bad) & reps["case_id"].isin(vis.keys())]
    by_case = rep.groupby("case_id")["report_id"].agg(list).to_dict()

    def mentions(run: Path) -> pd.DataFrame:
        return pd.read_parquet(run / f"instances_{a.lang}.parquet").merge(rep[["case_id", "report_id"]]) \
            .drop_duplicates(KEY)[KEY]

    m = mentions(a.parse_run)
    other = mentions(a.other_parse) if a.other_parse else None
    lum = pd.read_parquet(a.lumetti) if a.lumetti else None
    cand = candidate_items(m, by_case, vis, other, lum)
    s = sample_items(cand, a.per_reason, a.seed)
    paths = rep.set_index(["case_id", "report_id"])["path"]
    s["report_path"] = [paths.get((c, r)) for c, r in zip(s["case_id"], s["report_id"])]

    out = a.results_dir / "audit" / make_run_id(a.tag)
    out.mkdir(parents=True, exist_ok=False)
    sheet = s[["item_id", "case_id", "report_id", "report_path", "finding_type", "slot"]].assign(
        verdict="", correct_slot="")
    sheet.to_csv(out / "sheet.tsv", index=False, sep="\t")
    s[["item_id", "reason", "pair_id"]].to_parquet(out / "key.parquet", index=False)
    (out / "README.txt").write_text(
        "Open each report_path locally and fill `verdict` with one of: " + ", ".join(VERDICTS) +
        ".\nparser_wrong_slot: put the tooth/slot the report actually names in correct_slot (FDI number, Left, Right "
        "or QuadrantI-IV only).\nNo other text: do not copy report wording into this file. It stays on the cluster.\n")
    summary = {"n_candidates_by_reason": cand["reason"].value_counts().to_dict(),
               "n_sampled_by_reason": s["reason"].str.split(";").explode().value_counts().to_dict(), "n_items": len(s),
               "cases_dropped_failed_parse": len(bad & set(reps["case_id"]))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_provenance(out, extra={"parse_run": str(a.parse_run), "other_parse": str(a.other_parse),
                                 "lumetti": str(a.lumetti), "per_reason": a.per_reason, "seed": a.seed,
                                 "mirror_centres": a.mirror_centres, "excluded": a.exclude})
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
