"""Coding conventions in Lumetti's labels vs our parser output (GT/T1; counts only, no text).

A. Bridges: for FixedBridge findings, how many span teeth also carry ToothAbsence (pontic coding)?
   Are spans contiguous (abutments + pontics) or abutments only?
B. ToothDevelopmentPathology: which qualifiers/values do findings of this type carry?
C. Side consistency: quadrant vs the quadrant digit of tooth ids (JawAtrophy, PathologicalLesion);
   canal side vs tooth ids (left canal -> 3x teeth, right -> 4x).
D. ToothAbsence disagreements by tooth position (1..8) and concentration in few cases.
E. Mirrored canal disagreements by centre.

Usage:
    python -m partialref.extract.convention_checks --run results/extract/<parse run> --lang en \
        --lumetti-dir results/lumetti/<run> [--exclude ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from partialref.extract.ontology import REGION_PREFIX
from partialref.extract.score_parser import scoring_universe

QUAD_DIGIT = {"QuadrantI": "1", "QuadrantII": "2", "QuadrantIII": "3", "QuadrantIV": "4"}


def ours_findings(run: Path, lang: str) -> pd.DataFrame:
    rows = []
    for f in sorted((run / "per_report").glob(f"*__{lang}.json")):
        d = json.loads(f.read_text())
        if not d["extraction"]:
            continue
        for x in d["extraction"]["findings"]:
            q = {k: v for k, v in (x.get("qualifiers") or {}).items() if v}
            rows.append({"case_id": d["meta"]["case_id"], "report_id": d["meta"]["report_id"],
                         "finding_type": x["finding_type"], "tooth_ids": x.get("tooth_ids") or [],
                         "side": x.get("side"), "quadrant": x.get("quadrant"), "qualifiers": json.dumps(q)})
    return pd.DataFrame(rows)


def bridges(f: pd.DataFrame) -> dict:
    absent = f[f.finding_type == "ToothAbsence"].explode("tooth_ids").groupby("case_id")["tooth_ids"].agg(set)
    br = f[(f.finding_type == "ProstheticDevice") & f.qualifiers.str.contains("FixedBridge")]
    n_teeth = n_abs = contiguous = 0
    for r in br.itertuples():
        t = sorted(r.tooth_ids)
        n_teeth += len(t)
        n_abs += len(set(t) & absent.get(r.case_id, set()))
        nums = sorted(int(x) for x in t)
        same_quad = len({x // 10 for x in nums}) == 1
        contiguous += same_quad and nums == list(range(nums[0], nums[-1] + 1))
    return {"bridge_findings": int(len(br)), "teeth_in_spans": n_teeth,
            "span_teeth_also_ToothAbsence": n_abs, "contiguous_single_quadrant_spans": int(contiguous),
            "median_span_length": float(br.tooth_ids.map(len).median()) if len(br) else None}


def qualifier_values(f: pd.DataFrame, ftype: str) -> dict:
    c = Counter()
    sub = f[f.finding_type == ftype]
    for q in sub.qualifiers:
        for k, vals in json.loads(q).items():
            c.update(f"{k}={v}" for v in vals)
    return {"n": int(len(sub)), "without_qualifiers": int((sub.qualifiers == "{}").sum()),
            "values": dict(c.most_common(12))}


def side_consistency(f: pd.DataFrame) -> dict:
    out = {}
    for t in ("JawAtrophy", "PathologicalLesion"):
        sub = f[(f.finding_type == t) & f.quadrant.notna() & (f.tooth_ids.map(len) > 0)]
        ok = sum(all(x[0] == QUAD_DIGIT.get(q, "?") for x in teeth) for q, teeth in zip(sub.quadrant, sub.tooth_ids))
        mir = sum(all(x[0] == {"1": "2", "2": "1", "3": "4", "4": "3"}.get(QUAD_DIGIT.get(q, "?"), "?") for x in teeth)
                  for q, teeth in zip(sub.quadrant, sub.tooth_ids))
        out[t] = {"with_quadrant_and_teeth": int(len(sub)), "teeth_match_quadrant": int(ok),
                  "teeth_match_MIRRORED_quadrant": int(mir)}
    sub = f[(f.finding_type == "MandibularCanalCourse") & f.side.notna() & (f.tooth_ids.map(len) > 0)]
    exp = {"Left": "3", "Right": "4"}
    ok = sum(all(x[0] == exp[s] for x in teeth) for s, teeth in zip(sub.side, sub.tooth_ids))
    mir = sum(all(x[0] == exp["Right" if s == "Left" else "Left"] for x in teeth) for s, teeth in zip(sub.side, sub.tooth_ids))
    out["MandibularCanalCourse"] = {"with_side_and_teeth": int(len(sub)), "teeth_match_side": int(ok),
                                    "teeth_match_MIRRORED_side": int(mir)}
    return out


def absence_errors(ours_inst: pd.DataFrame, ref_inst: pd.DataFrame, cases: set[str]) -> dict:
    o = {tuple(x) for x in ours_inst[(ours_inst.finding_type == "ToothAbsence") & ours_inst.case_id.isin(cases)]
         [["case_id", "slot"]].to_numpy()}
    r = {tuple(x) for x in ref_inst[(ref_inst.finding_type == "ToothAbsence") & ref_inst.case_id.isin(cases)]
         [["case_id", "slot"]].to_numpy()}
    fp, fn = o - r, r - o

    def by_pos(s):
        return dict(sorted(Counter(x[1][1] for x in s if len(x[1]) == 2).items()))

    def conc(s):
        per = Counter(x[0] for x in s)
        top = sorted(per.values(), reverse=True)
        k = max(1, len(cases) // 10)
        return {"cases_with_errors": len(per), "share_in_top10pct_cases": round(sum(top[:k]) / max(1, len(s)), 3),
                "cases_with_>=8_errors": sum(v >= 8 for v in per.values())}

    return {"fp": len(fp), "fn": len(fn), "fp_by_position": by_pos(fp), "fn_by_position": by_pos(fn),
            "fp_concentration": conc(fp), "fn_concentration": conc(fn)}


def mirrored_canal_by_centre(ours_inst, ref_inst, cases: set[str]) -> dict:
    o = {tuple(x) for x in ours_inst[(ours_inst.finding_type == "MandibularCanalCourse") & ours_inst.case_id.isin(cases)]
         [["case_id", "slot"]].to_numpy()}
    r = {tuple(x) for x in ref_inst[(ref_inst.finding_type == "MandibularCanalCourse") & ref_inst.case_id.isin(cases)]
         [["case_id", "slot"]].to_numpy()}
    mir = {"Left": "Right", "Right": "Left"}
    fn_m = [c for c, s in r - o if (c, mir[s]) in o]
    return {"fn_mirrored_by_centre": dict(Counter(c[0] for c in fn_m)),
            "canal_instances_by_centre": dict(Counter(c[0] for c, _ in r))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--lumetti-dir", required=True, type=Path)
    ap.add_argument("--exclude", type=lambda s: [x for x in s.split(",") if x], default=[])
    a = ap.parse_args(argv)
    lf = pd.read_parquet(a.lumetti_dir / "findings.parquet")
    lf = lf[~lf.unknown_type & ~lf.case_id.isin(a.exclude)].copy()
    lf["tooth_ids"] = lf["tooth_ids"].map(list)
    of = ours_findings(a.run, a.lang)
    cases, _, _ = scoring_universe(a.run / f"instances_{a.lang}.parquet", lumetti=a.lumetti_dir / "instances.parquet",
                                   exclude=a.exclude)
    lf = lf[lf.case_id.isin(cases)]
    of = of[of.case_id.isin(cases)]
    oi = pd.read_parquet(a.run / f"instances_{a.lang}.parquet")
    oi = oi[~oi["slot"].astype(str).str.startswith(REGION_PREFIX)]           # slot-level checks only
    li = pd.read_parquet(a.lumetti_dir / "instances.parquet")
    oi, li = oi[~oi.case_id.isin(a.exclude)], li[~li.case_id.isin(a.exclude)]
    res = {
        "A_bridges": {"lumetti": bridges(lf), "ours": bridges(of)},
        "B_development_pathology_qualifiers": {"lumetti": qualifier_values(lf, "ToothDevelopmentPathology"),
                                               "ours": qualifier_values(of, "ToothDevelopmentPathology")},
        "C_side_consistency": {"lumetti": side_consistency(lf), "ours": side_consistency(of)},
        "D_tooth_absence_errors": absence_errors(oi, li, cases),
        "E_mirrored_canal": mirrored_canal_by_centre(oi, li, cases),
    }
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
