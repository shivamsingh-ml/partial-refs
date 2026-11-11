"""Fixed review list for T1: which reports Shivam reviews, drawn once with a recorded seed (no cherry-picking).

Two strata, because they answer different questions (DECISIONS 2026-11-11):
  base   n random reports of the held-out half (double-read P/F/S cases, stratified by centre). Every parsed
         finding is judged and missed findings are entered -> parser precision AND recall per finding type.
  extra  for each finding type that the parser found in fewer than `target` base reports: further random
         held-out reports in which the parser found that type. They are selected on the parser's own output,
         so they count ONLY towards that type's precision (never recall, never another type).

Inputs and output are IDs and codes only (never report text). Output: one JSON file, read by the review app
(REVIEW_LIST) and copied into the review bundle.

    python -m partialref.review.sample --parse-run results/extract/<run> --manifest results/manifest/<run> \
        [--n-base 60] [--target 15] [--seed 20261111] [--out-dir results/review/lists]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from partialref.extract.ontology import FindingType
from partialref.extract.score_parser import parser_half
from partialref.provenance import git_state

DEFAULT_EXCLUDE = ["P116", "P117", "P260", "P261", "S0015"]   # misfiled reports; possibly upside down
TYPES = [t.value for t in FindingType]


def population(reports: pd.DataFrame, half: str = "heldout", centres: str = "PFS",
               exclude: list[str] | None = None) -> pd.DataFrame:
    """Reports (case_id, report_id, centre) of double-read cases in the given parser half, sorted."""
    r = reports[["case_id", "report_id"]].drop_duplicates().copy()
    r["centre"] = r["case_id"].str[0]
    n = r.groupby("case_id")["report_id"].transform("nunique")
    keep = (n >= 2) & r["centre"].isin(list(centres)) & ~r["case_id"].isin(exclude or [])
    if half != "all":
        keep &= r["case_id"].map(parser_half) == half
    return r[keep].sort_values(["case_id", "report_id"]).reset_index(drop=True)


def _allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation with largest remainders (ties: alphabetical); never more than a stratum holds."""
    total = sum(sizes.values())
    n = min(n, total)
    raw = {k: n * v / total for k, v in sizes.items()}
    out = {k: int(np.floor(x)) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: n - sum(out.values())]:
        out[k] += 1
    return out


def draw(pop: pd.DataFrame, instances: pd.DataFrame, n_base: int, target: int, seed: int) -> tuple[list[dict], dict]:
    """Review list entries (base first, then extra; each in random order) and the counts behind them."""
    keys = list(map(tuple, pop[["case_id", "report_id"]].to_numpy()))
    found: dict[tuple[str, str], set[str]] = {k: set() for k in keys}
    for c, r, t in instances[["case_id", "report_id", "finding_type"]].drop_duplicates().itertuples(index=False):
        if (c, r) in found:
            found[(c, r)].add(t)
    base: list[tuple[str, str]] = []
    alloc = _allocate(pop["centre"].value_counts().to_dict(), n_base)
    for i, centre in enumerate(sorted(alloc)):
        ks = [k for k, c in zip(keys, pop["centre"]) if c == centre]
        pick = np.random.default_rng([seed, i]).choice(len(ks), size=alloc[centre], replace=False)
        base += [ks[j] for j in sorted(pick)]
    in_base = set(base)
    base = [base[j] for j in np.random.default_rng([seed, 100]).permutation(len(base))]
    extra_for: dict[tuple[str, str], list[str]] = {}
    counts = {}
    for i, t in enumerate(TYPES):
        n_pos = sum(t in found[k] for k in base)
        pool = [k for k in keys if k not in in_base and t in found[k]]
        need = min(max(0, target - n_pos), len(pool))
        pick = np.random.default_rng([seed, 1000 + i]).choice(len(pool), size=need, replace=False) if need else []
        for j in sorted(pick):
            extra_for.setdefault(pool[j], []).append(t)
        counts[t] = {"base_reports_with_type": n_pos, "extra_reports": need, "pool_outside_base": len(pool)}
    extra = sorted(extra_for)
    extra = [extra[j] for j in np.random.default_rng([seed, 200]).permutation(len(extra))]
    entries = [{"case_id": c, "report_id": r, "stratum": "base", "judge_types": None} for c, r in base]
    entries += [{"case_id": c, "report_id": r, "stratum": "extra", "judge_types": extra_for[(c, r)]} for c, r in extra]
    return entries, {"base_by_centre": alloc, "by_type": counts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse-run", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--half", default="heldout", choices=["heldout", "tune", "all"])
    ap.add_argument("--centres", default="PFS")
    ap.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDE))
    ap.add_argument("--n-base", default=60, type=int)
    ap.add_argument("--target", default=15, type=int, help="wanted reports with the type (base + extra)")
    ap.add_argument("--seed", default=20261111, type=int)
    ap.add_argument("--out-dir", default=Path("results/review/lists"), type=Path)
    a = ap.parse_args(argv)
    reps = pd.read_parquet(a.manifest / "reports.parquet")
    exclude = [x for x in a.exclude.split(",") if x]
    pop = population(reps[reps["lang"] == a.lang], a.half, a.centres, exclude)
    inst = pd.read_parquet(a.parse_run / f"instances_{a.lang}.parquet")
    entries, counts = draw(pop, inst, a.n_base, a.target, a.seed)
    meta = {"parse_run": a.parse_run.name, "lang": a.lang, "half": a.half, "centres": a.centres, "exclude": exclude,
            "n_base": a.n_base, "target": a.target, "seed": a.seed, "n_population_reports": len(pop),
            "n_population_cases": int(pop["case_id"].nunique()), **counts,
            "n_entries": len(entries), "n_extra": sum(e["stratum"] == "extra" for e in entries),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **git_state()}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"review-list-{a.parse_run.name}-s{a.seed}.json"
    out.write_text(json.dumps({"meta": meta, "entries": entries}, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"review list: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
