"""Where does the parser disagree with Lumetti's labels? (GT/T1 error analysis; counts only, no text)

Works on the per-case union of our per-report instances (as the scorer does) vs Lumetti's per-patient
instances, on cases present in both. For each finding type:
- false positives (we have (case, type, slot), they don't): what type(s) do THEY have at that slot?
  is our slot their slot mirrored left/right (FDI quadrant 1<->2, 3<->4) or off by one tooth?
- false negatives (they have it, we don't): what type(s) do WE have at that slot? mirrored / off-by-one?
Also: our condyle findings in P/F/S (where Lumetti has none) vs our 'not_assessable' statements.

Usage:
    python -m partialref.extract.parser_diagnostics --run results/extract/<parse run> --lang en \
        --lumetti results/lumetti/<run>/instances.parquet [--exclude P116,P117,P260,P261]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from partialref.extract.score_parser import scoring_universe
from partialref.provenance import make_run_id, write_provenance


def mirror(slot: str) -> str | None:
    if len(slot) == 2 and slot.isdigit():
        q, p = int(slot[0]), slot[1]
        m = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}.get(q)
        return f"{m}{p}" if m else None
    return {"Left": "Right", "Right": "Left", "QuadrantI": "QuadrantII", "QuadrantII": "QuadrantI",
            "QuadrantIII": "QuadrantIV", "QuadrantIV": "QuadrantIII"}.get(slot)


def neighbours(slot: str) -> list[str]:
    if len(slot) == 2 and slot.isdigit():
        q, p = slot[0], int(slot[1])
        return [f"{q}{p + d}" for d in (-1, 1) if 1 <= p + d <= 8]
    return []


def diagnose(ours: pd.DataFrame, ref: pd.DataFrame, cases: set[str]) -> pd.DataFrame:
    """cases: scored universe from partialref.extract.score_parser.scoring_universe."""
    o = ours[ours.case_id.isin(cases)][["case_id", "finding_type", "slot"]].drop_duplicates()
    r = ref[ref.case_id.isin(cases)][["case_id", "finding_type", "slot"]].drop_duplicates()
    oset, rset = set(map(tuple, o.to_numpy())), set(map(tuple, r.to_numpy()))
    o_at = o.groupby(["case_id", "slot"])["finding_type"].agg(set).to_dict()
    r_at = r.groupby(["case_id", "slot"])["finding_type"].agg(set).to_dict()
    rows = []
    for t in sorted({x[1] for x in oset | rset}):
        fp = [x for x in oset - rset if x[1] == t]
        fn = [x for x in rset - oset if x[1] == t]
        fp_other, fn_other = Counter(), Counter()
        fp_mirror = sum((c, t, mirror(s)) in rset for c, _, s in fp if mirror(s))
        fn_mirror = sum((c, t, mirror(s)) in oset for c, _, s in fn if mirror(s))
        fp_nb = sum(any((c, t, n) in rset for n in neighbours(s)) for c, _, s in fp)
        fn_nb = sum(any((c, t, n) in oset for n in neighbours(s)) for c, _, s in fn)
        for c, _, s in fp:
            fp_other.update(r_at.get((c, s), {"<nothing>"}) - {t} or {"<nothing>"})
        for c, _, s in fn:
            fn_other.update(o_at.get((c, s), {"<nothing>"}) - {t} or {"<nothing>"})
        rows.append({"finding_type": t, "tp": len(oset & {x for x in rset if x[1] == t}),
                     "fp": len(fp), "fp_mirrored": fp_mirror, "fp_off_by_one": fp_nb,
                     "fp_their_type_at_slot": dict(fp_other.most_common(4)),
                     "fn": len(fn), "fn_mirrored": fn_mirror, "fn_off_by_one": fn_nb,
                     "fn_our_type_at_slot": dict(fn_other.most_common(4))})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--lumetti", required=True, type=Path)
    ap.add_argument("--exclude", type=lambda s: [x for x in s.split(",") if x], default=[])
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    a = ap.parse_args(argv)
    ours = pd.read_parquet(a.run / f"instances_{a.lang}.parquet")
    ref = pd.read_parquet(a.lumetti)
    cases, _, _ = scoring_universe(a.run / f"instances_{a.lang}.parquet", lumetti=a.lumetti, exclude=a.exclude)
    d = diagnose(ours, ref, cases)

    # condyle findings outside A vs our not_assessable statements for the same reports
    na = pd.read_parquet(a.run / f"not_assessable_{a.lang}.parquet")
    cond = ours[(ours.finding_type == "MandibularCondyleFinding") & (ours.case_id.str[0] != "A")]
    cond_reports = cond[["case_id", "report_id"]].drop_duplicates()
    na_cond = na[na.structure.str.startswith("condyle")][["case_id", "report_id"]].drop_duplicates()
    extra = {"condyle_findings_outside_A": int(len(cond)),
             "reports_with_condyle_finding_outside_A": int(len(cond_reports)),
             "of_which_also_say_condyle_not_assessable": int(len(cond_reports.merge(na_cond)))}

    out = a.results_dir / "extract" / make_run_id(f"parser-diag-{a.lang}")
    out.mkdir(parents=True, exist_ok=False)
    d.to_csv(out / "diagnostics.csv", index=False)
    (out / "extra.json").write_text(json.dumps(extra, indent=2))
    write_provenance(out, extra={"run": str(a.run), "lumetti": str(a.lumetti), "lang": a.lang})
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(d.to_string(index=False))
    print(json.dumps(extra, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
