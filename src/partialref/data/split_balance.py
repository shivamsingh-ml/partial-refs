"""Balance of the case split on segmentation-relevant properties (2026-11-17; reviewed).

Purpose: document whether `configs/splits/v1` can serve as the standard ToothFairy3 split for future studies
(segmentation and others): test vs development and the five development folds, overall and per centre.
Inputs: the per-case and per-tooth label features the split was built from (`partialref.audit` visibility
features: numbers from the manual masks; no images, no report text), the split, and the manifest (number of
English reports per case, which defines the split's single/double-read stratum).

Properties. Shares of scans: toothless, upper/lower teeth present, both arches, upper jaw labelled, implant /
crown / bridge label present, any tooth crown cut by the field of view, any root tip cut (toothless scans count
as "not cut"). Means: teeth per scan; per-scan share of present teeth with crown / root tip cut, averaged over
scans with teeth (toothless scans excluded, not 0); axial slices (a slice count; voxel size differs by centre,
so compare within centre).

Design variables: `partialref.data.splits` sorted every centre x single/double-read cell by jaw coverage (both
arches / one arch / none), then tooth count, and sampled the test set evenly along that order. Toothless, teeth
present per arch, both arches and teeth per scan are therefore balanced BY CONSTRUCTION (`design_variable`):
their balance shows the method worked, and their p-values say nothing.

Statistics, test - development: standardised mean difference (SMD; binary: difference / sqrt of the mean of the
two groups' p(1-p); numeric: difference / pooled SD) and a two-sided permutation p-value. The permutation keeps
the split's rules: test labels are shuffled within centre x single/double-read cells, and the `never_test`
cases stay in development. It does not reproduce the sorted systematic sampling, so p-values are conservative
(too large) for properties correlated with the design variables; they are uncorrected for the 14 properties.
Read SMDs first (|SMD| < 0.1 negligible, 0.1-0.2 small).

    python -m partialref.data.split_balance --splits configs/splits/v1 \\
        --features results/audit/<run>-visibility-features [--manifest <run>] [--out results/splits/<run_id>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from partialref.provenance import make_run_id, write_provenance

BINARY = {
    "toothless": lambda c: c.n_teeth == 0,
    "upper teeth present": lambda c: c.n_upper_teeth > 0,
    "lower teeth present": lambda c: c.n_lower_teeth > 0,
    "both arches": lambda c: (c.n_upper_teeth > 0) & (c.n_lower_teeth > 0),
    "upper jaw labelled": lambda c: c.maxilla_vox > 0,
    "implant label": lambda c: c.implant_vox > 0,
    "crown label": lambda c: c.crown_vox > 0,
    "bridge label": lambda c: c.bridge_vox > 0,
    "any crown cut by field of view": lambda c: c.any_crown_cut > 0,
    "any root tip cut by field of view": lambda c: c.any_root_cut > 0,
}
NUMERIC = {"teeth per scan": "n_teeth", "share of teeth crown-cut": "crown_cut", "share of teeth root-cut": "root_cut",
           "axial slices": "z_slices"}
DESIGN = {"toothless", "upper teeth present", "lower teeth present", "both arches", "teeth per scan"}


def case_table(splits: Path, features: Path, manifest: Path) -> pd.DataFrame:
    c = pd.read_parquet(features / "case_features.parquet")
    t = pd.read_parquet(features / "tooth_features.parquet")
    s = pd.read_csv(splits / "splits.csv")
    never = set(json.loads((splits / "splits.json").read_text()).get("never_test", {}))
    m = pd.read_parquet(manifest / "cases.parquet")[["case_id", "n_reports_en"]]
    c = c.merge(s[["case_id", "centre", "role", "fold"]].rename(columns={"centre": "centre_split"}), on="case_id",
                how="left", validate="one_to_one")
    if c["role"].isna().any():
        raise ValueError(f"cases without a split role: {c.loc[c['role'].isna(), 'case_id'].tolist()[:5]}")
    if (c["centre"] != c["centre_split"]).any():
        raise ValueError("centre differs between features and splits.csv")
    c = c.merge(m, on="case_id", how="left", validate="one_to_one")
    if c["n_reports_en"].isna().any():
        raise ValueError("cases missing from the manifest")
    c = c[c["role"].isin(["dev", "test"])].copy()
    c["read"] = np.where(c["n_reports_en"] >= 2, "multi", "single")     # as partialref.data.splits
    c["never_test"] = c["case_id"].isin(never)
    if (c["never_test"] & (c["role"] == "test")).any():
        raise ValueError("a never_test case is in the test set")
    tp = t[t["present"].astype(bool)].copy()
    for col in ("crown_cut", "root_tip_cut"):
        if tp[col].isna().any():
            raise ValueError(f"{col} missing for a present tooth")
        tp[col] = tp[col].astype(bool).astype(float)
    g = tp.groupby("case_id").agg(crown_cut=("crown_cut", "mean"), root_cut=("root_tip_cut", "mean"),
                                  any_crown_cut=("crown_cut", "max"), any_root_cut=("root_tip_cut", "max"))
    c = c.merge(g, left_on="case_id", right_index=True, how="left")
    c[["any_crown_cut", "any_root_cut"]] = c[["any_crown_cut", "any_root_cut"]].fillna(0.0)
    c["n_teeth"] = c["n_upper_teeth"] + c["n_lower_teeth"]
    c["z_slices"] = c["shape"].map(lambda x: int(str(x).split("x")[2]))
    for k, f in BINARY.items():
        c[k] = f(c).astype(float)
    for k, col in NUMERIC.items():
        c[k] = c[col].astype(float)                    # teeth-share columns stay NaN for toothless scans
    c["is_test"] = (c["role"] == "test").astype(int)
    return c


def smd(x: pd.Series, test: pd.Series, binary: bool) -> float:
    a, b = x[test == 0].mean(), x[test == 1].mean()
    if binary:
        sd = np.sqrt((a * (1 - a) + b * (1 - b)) / 2)
    else:
        sd = np.sqrt((x[test == 0].var() + x[test == 1].var()) / 2)
    return float((b - a) / sd) if sd > 0 else 0.0


def perm_p(x: pd.Series, test: pd.Series, cell: pd.Series, fixed_dev: pd.Series, n: int, seed: int) -> float:
    """Two-sided permutation p: test labels shuffled within cells among cases allowed in the test set."""
    ok = x.notna().to_numpy()
    xv, tv, cv, fv = x.to_numpy()[ok], test.to_numpy()[ok], cell.to_numpy()[ok], fixed_dev.to_numpy()[ok]
    obs = abs(xv[tv == 1].mean() - xv[tv == 0].mean())
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero((cv == g) & ~fv) for g in np.unique(cv)]
    hits = 0
    for _ in range(n):
        t = tv.copy()
        for idx in groups:
            t[idx] = rng.permutation(t[idx])
        hits += abs(xv[t == 1].mean() - xv[t == 0].mean()) >= obs - 1e-12
    return (hits + 1) / (n + 1)


def balance(c: pd.DataFrame, n_perm: int = 10_000, seed: int = 0) -> pd.DataFrame:
    cell = c["centre"] + "|" + c["read"]
    rows = []
    for k in list(BINARY) + list(NUMERIC):
        binary = k in BINARY
        row = {"property": k, "kind": "share" if binary else "mean", "design_variable": k in DESIGN,
               "dev": c.loc[c.is_test == 0, k].mean(), "test": c.loc[c.is_test == 1, k].mean(),
               "smd_test_minus_dev": smd(c[k], c["is_test"], binary),
               "perm_p": perm_p(c[k], c["is_test"], cell, c["never_test"], n_perm, seed)}
        for f in range(5):
            row[f"fold{f}"] = c.loc[(c.is_test == 0) & (c.fold == f), k].mean()
        for cen in sorted(c["centre"].unique()):
            m = c["centre"] == cen
            row[f"{cen}_dev"] = c.loc[m & (c.is_test == 0), k].mean()
            row[f"{cen}_test"] = c.loc[m & (c.is_test == 1), k].mean()
            row[f"{cen}_n_test_with"] = int(c.loc[m & (c.is_test == 1), k].sum()) if binary else None
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, help="default: the manifest recorded in splits.json")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    manifest = a.manifest or Path(json.loads((a.splits / "splits.json").read_text())["manifest"])
    c = case_table(a.splits, a.features, manifest)
    b = balance(c, a.n_perm, a.seed)
    out = a.out or Path("results/splits") / make_run_id("split-balance")
    out.mkdir(parents=True, exist_ok=True)
    b.to_csv(out / "balance.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "n_dev": int((c.is_test == 0).sum()), "n_test": int(c.is_test.sum()),
        "counts_by_centre_role": {f"{k[0]}|{k[1]}": int(v) for k, v in c.groupby(["centre", "role"]).size().items()},
        "not_in_features": "5 TF3-only scans (development) and centre A (external) have no visibility features",
        "n_perm": a.n_perm, "seed": a.seed}, indent=2))
    write_provenance(out, {"splits": str(a.splits), "features": str(a.features), "manifest": str(manifest)})
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(b[["property", "design_variable", "dev", "test", "smd_test_minus_dev", "perm_p"]].round(3)
              .to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
