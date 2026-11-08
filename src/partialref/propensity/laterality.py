"""Laterality diagnostics for GT/T2 (DECISIONS 2026-11-08: TF3 tooth masks mirrored vs report FDI numbers).

Counts only; no report text is read (parser instances + visibility features only). Three checks:

1. Per case: tooth-level mentions (Caries, Endo, Restoration, ToothDevelopmentPathology, Periapical) on
   teeth the mask calls absent, with the mask as is vs mirrored (quadrants 1<->2, 3<->4). If the flip is a
   dataset convention, mirrored should be better in (almost) every informative case; cases where the
   unmirrored mask fits clearly better are candidates for individually flipped scans.
2. Residual after mirroring: each (case, type, tooth) mentioned on a mask-absent tooth, split by how many
   of the case's readers mention it. Mentioned by >= 2 readers -> most likely a mask (or visibility-rule)
   problem; by 1 of >= 2 readers -> reader or parser error; single-read case -> undecidable.
3. Side swaps between two readers of the same case: for each finding type with tooth slots, a tooth
   mentioned only by reader A whose left/right mirror (e.g. 36 vs 46) is mentioned only by reader B.
   Chance control: the same count for the up/down counterpart (36 vs 26, same side, other jaw), which has
   the same combinatorics but is not a plausible side confusion. Also reported: neighbour shifts (36 vs 37).
   Position 1 (central incisors) is left out of both swap counts because 11 and 21 are physical neighbours.
   A swap seen here can also be a parser error; compare with the parser check before calling it a reader error.

Usage:
    python -m partialref.propensity.laterality --parse-run results/extract/<run> --manifest results/manifest/<run> \
        --vis results/audit/<run> [--lang en] [--centres PFS]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

from partialref.extract.ontology import FindingType, slot_kind
from partialref.propensity.agreement import (NEEDS_TOOTH, PERMANENT, _mirror_tooth, is_visible, load_visibility,
                                             select_reports)
from partialref.provenance import make_run_id, write_provenance

TOOTH_TYPES = [t.value for t in FindingType if slot_kind(t) == "tooth"]
VERT_Q = {"1": "4", "4": "1", "2": "3", "3": "2"}
PERM = set(PERMANENT)


def _vert(t: str) -> str:
    return VERT_Q[t[0]] + t[1]


def _neighbours(t: str) -> set[str]:
    q, p = t[0], int(t[1])
    return {f"{q}{p + d}" for d in (-1, 1) if 1 <= p + d <= 8}


def per_case_check(m: pd.DataFrame, n_by_case: dict, vis: dict, vis_mirror: dict) -> pd.DataFrame:
    """m: deduplicated mentions (case_id, report_id, finding_type, slot) of usable cases."""
    t = m[m["finding_type"].isin(NEEDS_TOOTH) & m["slot"].isin(PERM)]
    rows = []
    for c, n in n_by_case.items():
        g = t[t["case_id"] == c]
        inv = [sum(not is_visible(v[c], ft, s) for ft, s in zip(g["finding_type"], g["slot"]))
               for v in (vis, vis_mirror)]
        rows.append({"case_id": c, "centre": c[0], "n_reports": n, "tooth_mentions": len(g),
                     "invisible_normal": inv[0], "invisible_mirrored": inv[1]})
    df = pd.DataFrame(rows, columns=["case_id", "centre", "n_reports", "tooth_mentions",
                                     "invisible_normal", "invisible_mirrored"])
    d = df["invisible_normal"] - df["invisible_mirrored"]
    df["verdict"] = "tie_zero"
    df.loc[(d == 0) & (df["invisible_normal"] > 0), "verdict"] = "tie_nonzero"
    df.loc[d > 0, "verdict"] = "mirrored_better"
    df.loc[d < 0, "verdict"] = "normal_better"
    return df


def residual_check(m: pd.DataFrame, n_by_case: dict, vis_mirror: dict) -> pd.DataFrame:
    t = m[m["finding_type"].isin(NEEDS_TOOTH) & m["slot"].isin(PERM)]
    k = t.groupby(["case_id", "finding_type", "slot"])["report_id"].nunique().rename("k").reset_index()
    k = k[[not is_visible(vis_mirror[c], ft, s) for c, ft, s in zip(k["case_id"], k["finding_type"], k["slot"])]]
    k["n"] = k["case_id"].map(n_by_case)
    k["category"] = "single_read_case"
    k.loc[(k["n"] >= 2) & (k["k"] >= 2), "category"] = "multiple_readers"
    k.loc[(k["n"] >= 2) & (k["k"] == 1), "category"] = "one_of_several_readers"
    return k


def swap_check(m: pd.DataFrame, reports_by_case: dict[str, list[str]],
               collect: list | None = None) -> pd.DataFrame:
    """Per finding type, summed over all reader pairs of double-read cases. reports_by_case lists every
    usable report of a case, also ones without any tooth finding (a silent reader is still a reader).
    ``collect``: if a list, one dict per left/right swap and neighbour shift is appended (for audit sheets)."""
    t = m[m["finding_type"].isin(TOOTH_TYPES) & m["slot"].isin(PERM)]
    sets = t.groupby(["case_id", "report_id", "finding_type"])["slot"].agg(set).to_dict()
    cols = ["pairs", "both", "unmatched", "left_right_swaps", "up_down_swaps", "neighbour_shifts"]
    acc = {ft: dict.fromkeys(cols, 0) for ft in TOOTH_TYPES}
    for c, rids in reports_by_case.items():
        for r1, r2 in combinations(sorted(set(rids)), 2):
            for ft in TOOTH_TYPES:
                s1, s2 = sets.get((c, r1, ft), set()), sets.get((c, r2, ft), set())
                if not s1 and not s2:
                    continue
                a = acc[ft]
                o1, o2 = s1 - s2, s2 - s1
                a["pairs"] += 1
                a["both"] += len(s1 & s2)
                a["unmatched"] += len(o1) + len(o2)
                lr = [(s, _mirror_tooth(s)) for s in sorted(o1) if s[1] != "1" and _mirror_tooth(s) in o2]
                nb = [(s, n) for s in sorted(o1) for n in sorted(_neighbours(s) & o2)]   # each adjacent pair once
                a["left_right_swaps"] += len(lr)
                a["up_down_swaps"] += sum(1 for s in o1 if s[1] != "1" and _vert(s) in o2)
                a["neighbour_shifts"] += len(nb)
                if collect is not None:
                    collect += [{"case_id": c, "report_a": r1, "report_b": r2, "finding_type": ft,
                                 "slot_a": sa, "slot_b": sb, "kind": kind}
                                for kind, pairs in (("left_right_swap", lr), ("neighbour_shift", nb))
                                for sa, sb in pairs]
    return pd.DataFrame([{"finding_type": ft, **v} for ft, v in acc.items()])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse-run", required=True, type=Path)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--vis", required=True, type=Path, action="append")
    ap.add_argument("--centres", default="PFS")
    ap.add_argument("--exclude", type=lambda s: [x for x in s.split(",") if x],
                    default=["P116", "P117", "P260", "P261", "S0015"])
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="laterality")
    a = ap.parse_args(argv)

    # same report selection and failed-parse handling as partialref.propensity.agreement
    reps, bad_reports = select_reports(a.manifest, a.parse_run, a.lang, a.centres, a.exclude)
    reps = reps[["case_id", "report_id"]]
    bad = {c for c, _ in bad_reports}
    vis, vis_m = load_visibility(a.vis), load_visibility(a.vis, mirror_centres=a.centres)
    rep = reps[~reps["case_id"].isin(bad) & reps["case_id"].isin(vis.keys())]
    n_by_case = rep.groupby("case_id")["report_id"].nunique().to_dict()
    m = pd.read_parquet(a.parse_run / f"instances_{a.lang}.parquet") \
        .merge(rep, on=["case_id", "report_id"]) \
        .drop_duplicates(["case_id", "report_id", "finding_type", "slot"])

    pc = per_case_check(m, n_by_case, vis, vis_m)
    res = residual_check(m, n_by_case, vis_m)
    sw = swap_check(m, rep.groupby("case_id")["report_id"].agg(list).to_dict())

    out = a.results_dir / "propensity" / make_run_id(a.tag)
    out.mkdir(parents=True, exist_ok=False)
    pc.to_parquet(out / "per_case.parquet", index=False)
    res.to_parquet(out / "residual_instances.parquet", index=False)
    sw.to_csv(out / "swaps_by_type.csv", index=False)
    res_tab = res.pivot_table(index="finding_type", columns="category", values="k", aggfunc="size", fill_value=0)
    res_tab.to_csv(out / "residual_by_type.csv")
    summary = {
        "cases_used": len(n_by_case), "cases_dropped_failed_parse": len(bad & set(reps["case_id"])),
        "per_case_verdict": pc["verdict"].value_counts().to_dict(),
        "per_case_verdict_by_centre": pc.groupby("centre")["verdict"].value_counts().unstack(fill_value=0)
        .to_dict(orient="index"),
        "normal_better_by_2_or_more": pc.loc[pc["invisible_normal"] - pc["invisible_mirrored"] <= -2, "case_id"].tolist(),
        "residual_by_category": res["category"].value_counts().to_dict(),
        "swaps_total": sw[["pairs", "both", "unmatched", "left_right_swaps", "up_down_swaps",
                           "neighbour_shifts"]].sum().astype(int).to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_provenance(out, extra={"parse_run": str(a.parse_run), "manifest": str(a.manifest),
                                 "vis": [str(v) for v in a.vis], "excluded": a.exclude, "centres": a.centres})
    print("residual after mirroring (instances by category):\n" + res_tab.to_string())
    print("\nreader-pair swaps by type:\n" + sw.to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
