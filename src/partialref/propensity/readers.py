"""Who reports what: reporting profiles per report author (DECISIONS 2026-11-11/12).

`report_x` (first number of the report file name, e.g. 14 in cbct_14_419.txt) is treated as the report
author; the ODIN 2026 design document says 13 maxillofacial surgeons wrote the reports, which fits about 15
values. Nothing here needs it to be a person: it is a label with a stable reporting profile.

Inputs: manifest (reports, report_x), frozen parser instances, visibility features (manual masks, for the
double- vs single-read comparison). No report text. Cases with a failed parse are dropped (as in T2).

Outputs (results/propensity/<run_id>/):
- profiles.csv: per centre, author and finding type, on cases with exactly two reports by two different
  authors: both / only_me / only_partner over (type, slot) instances mentioned by at least one of the two
  reports, rate = both / (both + only_partner) = "share of what the partner mentions that this author also
  mentions". Instances cluster within cases, so the 95 % interval is a case-level bootstrap. The rate is
  pooled over the author's partners (see pair_tests for per-partner numbers). Also: cases paired, and the
  share of this author's reports (all cases) that mention the type at all.
- pair_tests.csv: per author pair and type, a case-level sign test: in how many cases does A mention more
  instances of the type than B, and vs fewer (ties dropped). Exchangeable readers -> about equal.
- stability.csv: rate per author and type in two fixed halves of the cases (hash of case ID).
- double_vs_single.csv: within each author, their reports in double-read cases vs their reports in
  single-read cases (n_findings, per-type mention share), plus case-level mask teeth per centre, with
  standardised mean differences and non-missing n (ODIN: second reports were made for "harder cases").
  Comparing within authors keeps author habits out of the case-difficulty comparison.
Cases with 3–4 reports and two-report cases written by the same author are not used for pairs; they are
counted in summary.json.

    python -m partialref.propensity.readers --parse-run results/extract/<run> \\
        --manifest results/manifest/<run> --vis results/audit/<run> [--min-n 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from partialref.propensity.agreement import select_reports
from partialref.provenance import make_run_id, write_provenance


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def half(case_id: str) -> str:
    return "A" if int(hashlib.sha256(f"reader-half:{case_id}".encode()).hexdigest(), 16) % 2 == 0 else "B"


def pair_instances(reps: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """One row per (case, author, type, slot) on cases with exactly two reports by two DIFFERENT authors:
    mine / partner mention flags. reps: case_id, report_id, report_x. inst: case_id, report_id, finding_type, slot."""
    n = reps.groupby("case_id")["report_id"].nunique()
    two = reps[reps["case_id"].isin(n[n == 2].index)].copy()
    two["author"] = two["report_x"].astype(str)
    na = two.groupby("case_id")["author"].nunique()
    two = two[two["case_id"].isin(na[na == 2].index)]
    m = inst[["case_id", "report_id", "finding_type", "slot"]].drop_duplicates().merge(
        two[["case_id", "report_id", "author"]], on=["case_id", "report_id"])
    who = m.groupby(["case_id", "finding_type", "slot"])["author"].agg(frozenset).reset_index()
    authors = two.groupby("case_id")["author"].agg(list).to_dict()
    rows = []
    for c, t, s, w in who.itertuples(index=False):
        a, b = authors[c]
        for me, partner in ((a, b), (b, a)):
            rows.append((c, c[0], me, partner, t, s, me in w, partner in w))
    return pd.DataFrame(rows, columns=["case_id", "centre", "author", "partner", "finding_type", "slot", "mine", "theirs"])


def _rate(x: pd.DataFrame) -> float:
    den = int(x["theirs"].sum())
    return float((x["mine"] & x["theirs"]).sum() / den) if den else np.nan


def boot_ci(x: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """Case-level bootstrap of the rate (cases resampled with replacement)."""
    per = x.groupby("case_id").agg(b=("both_i", "sum"), t=("theirs", "sum"))
    if per["t"].sum() == 0:
        return (np.nan, np.nan)
    b, t = per["b"].to_numpy(), per["t"].to_numpy()
    idx = rng.integers(0, len(per), (n_boot, len(per)))
    tb = t[idx].sum(axis=1)
    r = np.where(tb > 0, b[idx].sum(axis=1) / np.maximum(tb, 1), np.nan)
    r = r[~np.isnan(r)]
    return (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))) if len(r) else (np.nan, np.nan)


def profiles(pi: pd.DataFrame, reps_all: pd.DataFrame, inst: pd.DataFrame, n_boot: int = 1000,
             seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pi = pi.assign(both_i=pi["mine"] & pi["theirs"])
    g = pi.groupby(["centre", "author", "finding_type"])
    out = pd.DataFrame({"n_cases": g["case_id"].nunique(), "both": g.apply(lambda x: int((x.mine & x.theirs).sum()), include_groups=False),
                        "only_me": g.apply(lambda x: int((x.mine & ~x.theirs).sum()), include_groups=False),
                        "only_partner": g.apply(lambda x: int((~x.mine & x.theirs).sum()), include_groups=False)}).reset_index()
    den = out["both"] + out["only_partner"]
    out["rate_when_partner_mentions"] = np.where(den > 0, out["both"] / den.where(den > 0, 1), np.nan)
    ci = {k: boot_ci(x, n_boot, rng) for k, x in g}
    keys = list(zip(out["centre"], out["author"], out["finding_type"]))
    out["rate_lo"], out["rate_hi"] = [ci[k][0] for k in keys], [ci[k][1] for k in keys]
    paired = pi.groupby(["centre", "author"])["case_id"].nunique().rename("n_paired_cases_any_mention")
    out = out.merge(paired.reset_index(), on=["centre", "author"], how="left")
    # share of this author's reports (all cases) that mention the type at all
    ra = reps_all.assign(author=reps_all["report_x"].astype(str), centre=reps_all["case_id"].str[0])
    ment = inst[["case_id", "report_id", "finding_type"]].drop_duplicates()
    nrep = ra.groupby(["centre", "author"])["report_id"].count().rename("n_reports_all")
    k = ment.merge(ra[["case_id", "report_id", "author", "centre"]], on=["case_id", "report_id"]) \
        .groupby(["centre", "author", "finding_type"])["report_id"].nunique().rename("reports_mentioning")
    out = out.merge(nrep.reset_index(), on=["centre", "author"], how="left") \
        .merge(k.reset_index(), on=["centre", "author", "finding_type"], how="left")
    out["reports_mentioning"] = out["reports_mentioning"].fillna(0).astype(int)
    out["share_reports_mentioning"] = out["reports_mentioning"] / out["n_reports_all"]
    return out


def pair_tests(pi: pd.DataFrame) -> pd.DataFrame:
    """Case-level sign test per author pair and type (instances within a case are not independent)."""
    p = pi[pi["author"] < pi["partner"]]            # each instance once per unordered pair (ids always differ)
    rows = []
    for (c, a, b, t), g in p.groupby(["centre", "author", "partner", "finding_type"]):
        only_a, only_b = int((g.mine & ~g.theirs).sum()), int((~g.mine & g.theirs).sum())
        both = int((g.mine & g.theirs).sum())
        per = g.assign(da=g.mine & ~g.theirs, db=~g.mine & g.theirs).groupby("case_id")[["da", "db"]].sum()
        more_a, more_b = int((per.da > per.db).sum()), int((per.db > per.da).sum())
        pv = binomtest(more_a, more_a + more_b, 0.5).pvalue if more_a + more_b > 0 else np.nan
        rows.append({"centre": c, "author_a": a, "author_b": b, "finding_type": t, "n_cases": g["case_id"].nunique(),
                     "both": both, "only_a": only_a, "only_b": only_b, "cases_a_more": more_a,
                     "cases_b_more": more_b, "p_sign_cases": pv})
    return pd.DataFrame(rows)


def stability(pi: pd.DataFrame, min_n: int) -> pd.DataFrame:
    pi = pi.assign(half=pi["case_id"].map(half))
    rows = []
    for (c, a, t), g in pi.groupby(["centre", "author", "finding_type"]):
        rec = {"centre": c, "author": a, "finding_type": t}
        for h in ("A", "B"):
            x = g[g["half"] == h]
            den = int(x.theirs.sum())
            rec[f"n_{h}"] = den
            rec[f"rate_{h}"] = (x.mine & x.theirs).sum() / den if den else np.nan
        rec["both_halves_ge_min_n"] = rec["n_A"] >= min_n and rec["n_B"] >= min_n
        rows.append(rec)
    return pd.DataFrame(rows)


def smd(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def double_vs_single(reps: pd.DataFrame, inst: pd.DataFrame, vis: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    """Within each author: their reports in double-read cases vs in single-read cases; plus case-level mask
    teeth per centre. Rows: level ("author" or "case"), centre, author ("" for case level), variable."""
    nrep = reps.groupby("case_id")["report_id"].nunique()
    r = reps.assign(author=reps["report_x"].astype(str), centre=reps["case_id"].str[0],
                    double=reps["case_id"].map(nrep) >= 2)
    ment = inst[["case_id", "report_id", "finding_type", "slot"]].drop_duplicates()
    cnt = ment.groupby(["case_id", "report_id"]).size()
    r["n_findings"] = [int(cnt.get((c, i), 0)) for c, i in zip(r["case_id"], r["report_id"])]
    for t in types:
        has = set(map(tuple, ment.loc[ment.finding_type == t, ["case_id", "report_id"]].to_numpy()))
        r[f"mentions_{t}"] = [(c, i) in has for c, i in zip(r["case_id"], r["report_id"])]
    rows = []

    def add(level, centre, author, var, dd, ss):
        dd, ss = dd.dropna().astype(float), ss.dropna().astype(float)
        rows.append({"level": level, "centre": centre, "author": author, "variable": var,
                     "n_double": len(dd), "n_single": len(ss),
                     "mean_double": float(dd.mean()) if len(dd) else np.nan,
                     "mean_single": float(ss.mean()) if len(ss) else np.nan, "smd": smd(dd, ss)})

    for (c, a), g in r.groupby(["centre", "author"]):
        if g["double"].all() or not g["double"].any():
            continue
        for var in ["n_findings"] + [f"mentions_{t}" for t in types]:
            add("author", c, a, var, g.loc[g["double"], var], g.loc[~g["double"], var])
    cases = r.drop_duplicates("case_id")[["case_id", "centre", "double"]].merge(
        vis[["case_id", "n_upper_teeth", "n_lower_teeth"]], on="case_id", how="left")
    cases["mask_teeth"] = cases["n_upper_teeth"] + cases["n_lower_teeth"]
    for c, g in cases.groupby("centre"):
        add("case", c, "", "mask_teeth", g.loc[g["double"], "mask_teeth"], g.loc[~g["double"], "mask_teeth"])
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse-run", required=True, type=Path)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--vis", required=True, type=Path)
    ap.add_argument("--centres", default="PFS")
    ap.add_argument("--exclude", type=lambda s: [x for x in s.split(",") if x],
                    default=["P116", "P117", "P260", "P261", "S0015"])
    ap.add_argument("--min-n", type=int, default=10, help="instances the partner mentions, per half, for stability")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="readers")
    a = ap.parse_args(argv)

    reps, bad = select_reports(a.manifest, a.parse_run, a.lang, a.centres, a.exclude)
    bad_cases = {c for c, _ in bad}
    reps = reps[~reps["case_id"].isin(bad_cases)][["case_id", "report_id", "report_x"]]
    inst = pd.read_parquet(a.parse_run / f"instances_{a.lang}.parquet")
    inst = inst.merge(reps[["case_id", "report_id"]], on=["case_id", "report_id"])
    vis = pd.read_parquet(a.vis / "case_features.parquet")
    types = sorted(inst["finding_type"].unique())

    pi = pair_instances(reps, inst)
    prof, tests, stab = profiles(pi, reps, inst, a.n_boot, a.seed), pair_tests(pi), stability(pi, a.min_n)
    nrep = reps.groupby("case_id")["report_id"].nunique()
    two = reps[reps["case_id"].isin(nrep[nrep == 2].index)]
    same_author = int((two.groupby("case_id")["report_x"].nunique() == 1).sum())
    dvs = double_vs_single(reps, inst, vis, types)

    supported = prof[(prof["both"] + prof["only_partner"]) >= a.min_n]
    silent = supported[supported["rate_hi"] < 0.10]
    st = stab[stab["both_halves_ge_min_n"]]
    summary = {
        "n_cases_with_two_authors": int(pi["case_id"].nunique()), "n_failed_cases_dropped": len(bad_cases),
        "cases_by_number_of_reports": {str(k): int(v) for k, v in nrep.value_counts().sort_index().items()},
        "two_report_cases_same_author_not_paired": same_author,
        "cases_with_3plus_reports_not_paired": int((nrep >= 3).sum()),
        "authors_per_centre": pi.groupby("centre")["author"].nunique().to_dict(),
        "author_type_cells_with_min_n": len(supported),
        "silent_cells_upper_ci_below_0_10": silent[["centre", "author", "finding_type", "both", "only_partner"]].to_dict("records"),
        "pair_tests_p_below_0_001": int((tests["p_sign_cases"] < 0.001).sum()), "pair_tests_total": len(tests),
        "stability_cells": len(st),
        "stability_corr_rate_A_B": (None if len(st) <= 2 or st[["rate_A", "rate_B"]].nunique().min() < 2
                                    else float(st[["rate_A", "rate_B"]].corr().iloc[0, 1])),
        "min_n": a.min_n}
    run_id = make_run_id(a.tag)
    out = a.results_dir / "propensity" / run_id
    out.mkdir(parents=True, exist_ok=False)
    prof.to_csv(out / "profiles.csv", index=False)
    tests.to_csv(out / "pair_tests.csv", index=False)
    stab.to_csv(out / "stability.csv", index=False)
    dvs.to_csv(out / "double_vs_single.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"run_id": run_id, **summary}, indent=2, default=str))
    write_provenance(out, extra={"run_id": run_id, "parse_run": str(a.parse_run), "manifest": str(a.manifest),
                                 "vis": str(a.vis), "excluded": a.exclude})
    print(json.dumps({"run_id": run_id, **{k: v for k, v in summary.items() if k != "silent_cells_upper_ci_below_0_10"},
                      "n_silent_cells": len(silent)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
