"""Case-level splits (HANDOFF §6.4; review and decisions 2026-11-11): locked test set, 5 development folds,
centre A external. Inputs are case IDs, centre, number of EN reports and manual-mask tooth counts only
(never report text).

Rules:
- P/F/S cases are split inside cells centre x single/double-read. Inside a cell, cases are ordered by jaw
  coverage ("teeth in both jaws" in the manual mask), then total tooth count, then a seeded random number;
  the test set is drawn by systematic sampling over that order (balances coverage and tooth count), the
  rest is dealt into the folds in turn.
- Never in the test set: cases with misfiled reports (P116, P117, P260, P261), S0015 (possibly upside
  down) and the cases with an open visual check. They stay in development. Cases are never moved after
  the split is committed, even if a check clears them later.
- Centre A (predicted masks only) is external and locked: no tuning on its scores.
- Leave-one-centre-out uses development cases only.
- The segmentation training list is NOT written here: it is frozen at stage 2 (after the visual checks)
  from the development cases plus the ToothFairy3-only scans, without unresolved cases.

    python -m partialref.data.splits --manifest results/manifest/<run> --features results/audit/<run> \\
        [--test-fraction 0.20] [--seed 20261110] [--out configs/splits/v1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from partialref.provenance import git_state

K = 5
NEVER_TEST = {"P116": "misfiled reports", "P117": "misfiled reports", "P260": "misfiled reports",
              "P261": "misfiled reports", "S0015": "possibly stored upside down; visual check open",
              "F028": "visual check open", "F055": "visual check open", "P230": "visual check open",
              "S0019": "visual check open", "P192": "visual check open", "P242": "visual check open"}
TF3_ONLY = ["P120", "P121", "P262", "P263", "P378"]          # ToothFairy3 scans without TF4 reports


def prepare(cases: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """One row per case: centre, read (single/multi), coverage (both_jaws / not_both / na for A), teeth."""
    f = features[["case_id", "n_upper_teeth", "n_lower_teeth"]]
    d = cases[["case_id", "centre", "n_reports_en"]].merge(f, on="case_id", how="left")
    pfs = d["centre"] != "A"
    missing = d.loc[pfs & d["n_upper_teeth"].isna(), "case_id"].tolist()
    if missing:
        raise ValueError(f"mask features missing for P/F/S cases: {missing[:5]}")
    if set(TF3_ONLY) & set(d["case_id"]):
        raise ValueError("a ToothFairy3-only scan appears in the manifest")
    d["read"] = np.where(d["n_reports_en"] >= 2, "multi", "single")
    d["teeth"] = d["n_upper_teeth"].fillna(0) + d["n_lower_teeth"].fillna(0)
    d["coverage"] = np.where(~pfs, "na", np.where((d["n_upper_teeth"] > 0) & (d["n_lower_teeth"] > 0),
                                                   "both_jaws", "not_both"))
    return d.sort_values("case_id").reset_index(drop=True)


def build(d: pd.DataFrame, test_fraction: float = 0.20, seed: int = 20261110,
          never_test: dict[str, str] | None = None) -> pd.DataFrame:
    """Roles (test / dev / external_A) and folds (0..K-1 for dev, -1 otherwise)."""
    never = NEVER_TEST if never_test is None else never_test
    d = d.copy()
    d["role"], d["fold"] = "", -1
    d.loc[d["centre"] == "A", "role"] = "external_A"
    rng = np.random.default_rng(seed)
    d["u"] = rng.random(len(d))                       # one random number per case, in case-ID order
    counter = int(rng.integers(K))
    for _, g in d[d["centre"] != "A"].groupby(["centre", "read"], sort=True):
        g = g.sort_values(["coverage", "teeth", "u"])
        elig = g[~g["case_id"].isin(never)]
        m = round(test_fraction * len(g))
        if m > 0:
            start = rng.random()
            pick = np.unique(np.floor((np.arange(m) + start) * len(elig) / m).astype(int))
            d.loc[elig.index[pick], "role"] = "test"
        dev = g[d.loc[g.index, "role"] != "test"]
        d.loc[dev.index, "role"] = "dev"
        d.loc[dev.index, "fold"] = (np.arange(len(dev)) + counter) % K
        counter = (counter + len(dev)) % K
    assert (d["role"] != "").all()
    assert not d.loc[d["role"] == "test", "case_id"].isin(never).any()
    assert (d.loc[d["role"] == "dev", "fold"].between(0, K - 1)).all()
    return d.drop(columns="u")


def summary(d: pd.DataFrame) -> str:
    p = d[d["centre"] != "A"].assign(set=lambda x: np.where(x["role"] == "test", "test", "fold" + x["fold"].astype(str)))
    g = p.groupby("set")
    tab = pd.DataFrame({"cases": g.size(), "F": g["centre"].apply(lambda s: (s == "F").sum()),
                        "P": g["centre"].apply(lambda s: (s == "P").sum()), "S": g["centre"].apply(lambda s: (s == "S").sum()),
                        "double_read": g["read"].apply(lambda s: (s == "multi").sum()),
                        "both_jaws_share": g["coverage"].apply(lambda s: (s == "both_jaws").mean()).round(3),
                        "mean_teeth": g["teeth"].mean().round(1), "en_reports": g["n_reports_en"].sum()})
    return tab.to_string() + f"\nexternal_A: {int((d['role'] == 'external_A').sum())} cases"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path, help="visibility-features run (manual masks)")
    ap.add_argument("--test-fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--out", type=Path, default=Path("configs/splits/v1"))
    a = ap.parse_args(argv)
    if a.out.exists():
        ap.error(f"{a.out} exists: a committed split is never overwritten")
    cases_p, feat_p = a.manifest / "cases.parquet", a.features / "case_features.parquet"
    d = build(prepare(pd.read_parquet(cases_p), pd.read_parquet(feat_p)), a.test_fraction, a.seed)
    a.out.mkdir(parents=True)
    d[["case_id", "centre", "role", "fold"]].to_csv(a.out / "splits.csv", index=False)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta = {"seed": a.seed, "test_fraction": a.test_fraction, "k": K, "manifest": str(a.manifest),
            "manifest_cases_sha256": sha(cases_p), "features": str(a.features), "features_sha256": sha(feat_p),
            "never_test": NEVER_TEST, "tf3_only_for_segmentation": TF3_ONLY,
            "counts": d["role"].value_counts().to_dict(), "numpy": np.__version__, "pandas": pd.__version__,
            **git_state()}
    (a.out / "splits.json").write_text(json.dumps(meta, indent=1))
    (a.out / "summary.txt").write_text(summary(d) + "\n")
    print(summary(d))
    print(f"split written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
