"""GT/T3: reader-based propensity checked against the manual masks (HANDOFF §7 GT; reviewed 2026-11-11).

For findings the ToothFairy3 manual masks can check, the mask says what is in the image, so two things can
be measured directly (no latent model):
  pi_mask  = share of reports that mention the finding where the mask shows it   (reporting propensity)
  phi_mask = share of reports that mention the finding where the mask does not   (false-mention rate)
and compared with the two-reader estimate pi_readers (same MLE as T2, fitted blind to the masks on the same
instances; case-level bootstrap, paired, so the difference has its own interval).

How to read it (independent review 2026-11-11): with phi = 0 and two readers the reader fit estimates
E[pi^2]/E[pi], which equals the mask-based pi only if pi is the same for every finding; if pi varies (an
obvious finding is reported more often than a subtle one) pi_readers is HIGHER than pi_mask, still with pure
omission. So:
  phi_mask small                   -> readers rarely mention what is not there (the T3 question);
  pi_readers ~ pi_mask             -> two-reader pi is unbiased here;
  pi_readers > pi_mask             -> pi varies between findings; the two-reader pi is an upper bound
                                      (HANDOFF L3), corrections based on it are conservative;
  pi_readers < pi_mask             -> not explained by varying pi (false mentions or correlated readers).

Units (what the masks support today):
- ToothAbsence: per tooth position. Universe = the assessable positions of T2 (arch in the scan, §2.6
  missing-tooth rule); mask truth = no tooth label at that position (same left/right handling as T2).
  A reader "mentions" the gap if it codes ToothAbsence, DentalImplant or PostExtractionSocket there (an
  implant site has no tooth label, and the parser codes "implant in 36" as DentalImplant only; bridge
  pontics are already ToothAbsence). The ToothAbsence-only coding, a version without third molars and
  one row per jaw are reported as extra rows. Per jaw to test a hypothesis: centre P's short scans (about
  5 cm) may leave upper tooth positions outside the scan, so a missing upper label may mean "not in the
  scan" rather than "missing tooth" (alternatives: readers skip the upper jaw, or annotators leave partly
  visible teeth unlabelled). If it holds, the cause is `slot_assessable` in agreement.py, which checks the
  side and back faces of the scan but not the top face. Evidence: DECISIONS 2026-11-11.
- DentalImplant, ProstheticDevice: per CASE, because the implant / crown / bridge labels carry no tooth
  number. Mask truth = implant label (crown or bridge label) present with at least --min-mm3; a report
  "mentions" the type if it has at least one finding of that type (ProstheticDevice: only prosthesis_type
  Crown, FixedBridge or SuspendedPontic, read from the per-report parser files: post-and-core, removable
  prostheses etc. have no mask label). This is P(at least one mentioned | at least one present), not the
  per-tooth pi of T2; rows restricted to cases with exactly one mask component are closer to it.

Inputs: parser instances, manifest, visibility features. No report text, no images.

    python -m partialref.propensity.mask_agreement --parse-run results/extract/<run> \
        --manifest results/manifest/<run> --vis results/audit/<run P/F/S> [--n-boot 1000] [--min-mm3 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from partialref.propensity.agreement import (
    _case_tables,
    _fit_table,
    build_instances,
    load_visibility,
    select_reports,
)
from partialref.provenance import make_run_id, write_provenance

CASE_TYPES = {"DentalImplant": ["implant"], "ProstheticDevice": ["crown", "bridge"]}
GAP_TYPES = ("ToothAbsence", "DentalImplant", "PostExtractionSocket")      # a reader saying "no natural tooth here"
FIXED_PROSTHESES = {"Crown", "FixedBridge", "SuspendedPontic"}              # the kinds that have a mask label
GT_TYPES = ["ToothAbsence", "DentalImplant", "ProstheticDevice"]


def jaw_subset(inst: pd.DataFrame, jaw: str) -> pd.DataFrame:
    """Tooth-position instances of one jaw: FDI quadrants 1/2 = upper, 3/4 = lower (the left/right mirroring
    of mask teeth swaps 1<->2 and 3<->4 only, so it never moves a tooth between jaws)."""
    return inst[inst["slot"].str[0].isin({"upper": ["1", "2"], "lower": ["3", "4"]}[jaw])]


def gap_mentions(mentions: pd.DataFrame, types=GAP_TYPES) -> pd.DataFrame:
    """Mentions of the given types, all relabelled as ToothAbsence (one per report and slot after dedup)."""
    m = mentions[mentions["finding_type"].isin(types)].assign(finding_type="ToothAbsence")
    return m.drop_duplicates(["case_id", "report_id", "finding_type", "slot"])


def fixed_prosthesis_mentions(parse_run: Path, lang: str) -> pd.DataFrame:
    """(case_id, report_id, finding_type="ProstheticDevice") for reports with a Crown / FixedBridge /
    SuspendedPontic finding, read from the per-report parser files (codes only)."""
    rows = []
    for f in sorted((parse_run / "per_report").glob(f"*__{lang}.json")):
        ext = (json.loads(f.read_text()).get("extraction") or {})
        for x in ext.get("findings", []):
            if x.get("finding_type") == "ProstheticDevice" and \
                    set((x.get("qualifiers") or {}).get("prosthesis_type") or []) & FIXED_PROSTHESES:
                c, r = f.stem.split("__")[:2]
                rows.append((c, r, "ProstheticDevice", "case"))
                break
    return pd.DataFrame(rows, columns=["case_id", "report_id", "finding_type", "slot"])


def tooth_absence_instances(reps: pd.DataFrame, mentions: pd.DataFrame, vis: dict, bad: set,
                            region_mode: str, upper_gap_rule: str = "bone") -> pd.DataFrame:
    """T2's assessable ToothAbsence positions with the mask truth (absent = no tooth label there).
    ``mentions`` decides what counts as a mention (see gap_mentions)."""
    inst, _ = build_instances(reps, mentions, vis, bad, ["ToothAbsence"], region_mode=region_mode,
                              upper_gap_rule=upper_gap_rule)
    inst["truth"] = [s not in vis[c]["teeth"] for c, s in zip(inst["case_id"], inst["slot"])]
    return inst


def case_level_instances(reps: pd.DataFrame, mentions: pd.DataFrame, case_features: pd.DataFrame, bad: set,
                         ftype: str, labels: list[str], min_mm3: float, single_component: bool = False) -> pd.DataFrame:
    """One instance per case: n = reports of the case, k = reports with >= 1 finding of the type,
    truth = one of the mask labels present (>= min_mm3, and > 0). Cases without a usable label table
    (label not found, shape mismatch, missing volume) are dropped, never counted as mask-negative.
    single_component: keep only cases with no label or exactly one connected component in total."""
    ok = case_features["label_found"].fillna(False).astype(bool)
    if "shape_mismatch" in case_features:
        ok = ok & ~case_features["shape_mismatch"].fillna(False).astype(bool)
    for lab in labels:
        ok &= case_features[f"{lab}_mm3"].notna()
    cf = case_features[ok].set_index("case_id")
    vol = sum(cf[f"{lab}_mm3"] for lab in labels)
    truth = ((vol > 0) & (vol >= min_mm3))
    if single_component:
        ncomp = sum(cf[f"{lab}_ncomp"].fillna(0) for lab in labels)
        truth = truth[(ncomp <= 1) | ~truth]
    truth = truth.to_dict()
    bad_cases = {c for c, _ in bad}
    rep = reps[~reps["case_id"].isin(bad_cases) & reps["case_id"].isin(truth.keys())]
    n = rep.groupby("case_id")["report_id"].nunique()
    m = mentions[mentions["finding_type"] == ftype].merge(rep[["case_id", "report_id"]], on=["case_id", "report_id"])
    k = m.drop_duplicates(["case_id", "report_id"]).groupby("case_id").size()
    out = pd.DataFrame({"case_id": n.index, "n": n.to_numpy(), "k": k.reindex(n.index).fillna(0).astype(int).to_numpy()})
    out["centre"] = out["case_id"].str[0]
    out["finding_type"], out["slot"] = ftype, "case"
    out["truth"] = out["case_id"].map(truth).astype(bool)
    return out


def _rates(s: np.ndarray) -> tuple[float, float, float]:
    """s = summed [k_true, n_true, k_false, n_false, inst_true, inst_all] -> (pi_mask, phi_mask, p_mask)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s[0] / s[1] if s[1] else np.nan, s[2] / s[3] if s[3] else np.nan, s[4] / s[5] if s[5] else np.nan)


def compare(inst: pd.DataFrame, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Mask-based rates and the mask-blind two-reader fit on the same instances, paired case bootstrap."""
    cases, tab = _case_tables(inst, int(inst["n"].max()))
    idx = {c: i for i, c in enumerate(cases)}
    ci = inst["case_id"].map(idx).to_numpy()
    t = inst["truth"].to_numpy(bool)
    sums = np.zeros((len(cases), 6))
    for col, val in enumerate([inst["k"] * t, inst["n"] * t, inst["k"] * ~t, inst["n"] * ~t, t, np.ones(len(inst))]):
        np.add.at(sums[:, col], ci, np.asarray(val, float))
    pi_m, phi_m, p_m = _rates(sums.sum(axis=0))
    p_r, pi_r, _ = _fit_table(tab.sum(axis=0))       # reader fit: mask-blind
    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, 5), np.nan)
    for b in range(n_boot):
        sel = rng.integers(0, len(cases), len(cases))
        bp, bpi, _ = _fit_table(tab[sel].sum(axis=0))
        boot[b] = (*_rates(sums[sel].sum(axis=0)), bpi, bp)

    def ci_(x: np.ndarray) -> tuple[float, float]:
        x = x[np.isfinite(x)]
        return tuple(np.quantile(x, [alpha / 2, 1 - alpha / 2])) if len(x) else (np.nan, np.nan)

    (pm_lo, pm_hi), (ph_lo, ph_hi), (pr_lo, pr_hi) = ci_(boot[:, 0]), ci_(boot[:, 1]), ci_(boot[:, 3])
    d_lo, d_hi = ci_(boot[:, 3] - boot[:, 0])
    tot = sums.sum(axis=0)
    return {"n_cases": len(cases), "n_instances": int(tot[5]), "n_mask_positive": int(tot[4]),
            "reader_slots_mask_positive": int(tot[1]), "reader_slots_mask_negative": int(tot[3]),
            "mentions_mask_positive": int(tot[0]), "mentions_mask_negative": int(tot[2]),
            "pi_mask": pi_m, "pi_mask_lo": pm_lo, "pi_mask_hi": pm_hi,
            "pi_readers": pi_r, "pi_readers_lo": pr_lo, "pi_readers_hi": pr_hi,
            "diff_readers_minus_mask": pi_r - pi_m, "diff_lo": d_lo, "diff_hi": d_hi,
            "ci_overlap": bool(np.isfinite([pm_lo, pr_lo]).all() and pm_lo <= pr_hi and pr_lo <= pm_hi),
            "readers_vs_mask": ("not estimable" if not np.isfinite([d_lo, d_hi]).all() else
                                "readers higher (pi varies; upper bound)" if d_lo > 0 else
                                "readers lower (not explained by varying pi)" if d_hi < 0 else "no difference"),
            "phi_mask": phi_m, "phi_mask_lo": ph_lo, "phi_mask_hi": ph_hi,
            "p_mask": p_m, "p_readers": p_r, "boot_identified_share": float(np.isfinite(boot[:, 3]).mean())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse-run", required=True, type=Path)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--vis", required=True, type=Path, action="append")
    ap.add_argument("--centres", default="PFS")
    ap.add_argument("--exclude", type=lambda s: [x for x in s.split(",") if x],
                    default=["P116", "P117", "P260", "P261", "S0015"])
    ap.add_argument("--mirror-centres", default="PFS")
    ap.add_argument("--swap-side-centres", default="PFS")
    ap.add_argument("--region-mode", choices=["omission", "uninformative"], default="omission")
    ap.add_argument("--upper-gap-rule", choices=["bone", "teeth_or_bone"], default="bone")
    ap.add_argument("--upper-bone-min-mm3", type=float, default=2000.0)
    ap.add_argument("--min-mm3", type=float, default=0.0, help="smallest implant / crown+bridge label volume that counts")
    ap.add_argument("--phi-max", type=float, default=0.05, help="guidance: upper 95%% limit of phi_mask must stay below")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="mask-agreement-gt")
    a = ap.parse_args(argv)

    reps, bad = select_reports(a.manifest, a.parse_run, a.lang, a.centres, a.exclude)
    reps = reps[["case_id", "report_id"]]
    mentions = pd.read_parquet(a.parse_run / f"instances_{a.lang}.parquet")
    cf = pd.concat([pd.read_parquet(d / "case_features.parquet") for d in a.vis], ignore_index=True)
    if "label_source" in cf and (cf["label_source"] != "tf3_manual").any():
        sys.exit("T3 needs manual masks: --vis must hold only label_source == 'tf3_manual' cases (no centre A)")
    vis = load_visibility(a.vis, mirror_centres=a.mirror_centres, swap_side_centres=a.swap_side_centres,
                          upper_bone_min_mm3=a.upper_bone_min_mm3)
    ta = tooth_absence_instances(reps, gap_mentions(mentions), vis, bad, a.region_mode, a.upper_gap_rule)
    ta_strict = tooth_absence_instances(reps, gap_mentions(mentions, ("ToothAbsence",)), vis, bad, a.region_mode,
                                        a.upper_gap_rule)
    sets = {("ToothAbsence", "tooth, all permanent"): ta,
            ("ToothAbsence", "tooth, without third molars"): ta[~ta["slot"].str.endswith("8")],
            ("ToothAbsence", "tooth, ToothAbsence coding only"): ta_strict,
            ("ToothAbsence", "tooth, lower jaw"): jaw_subset(ta, "lower"),
            ("ToothAbsence", "tooth, upper jaw"): jaw_subset(ta, "upper")}
    case_mentions = {"DentalImplant": mentions, "ProstheticDevice": fixed_prosthesis_mentions(a.parse_run, a.lang)}
    for ftype, labels in CASE_TYPES.items():
        sets[(ftype, "case")] = case_level_instances(reps, case_mentions[ftype], cf, bad, ftype, labels, a.min_mm3)
        sets[(ftype, "case, single mask component")] = case_level_instances(
            reps, case_mentions[ftype], cf, bad, ftype, labels, a.min_mm3, single_component=True)
    rows = []
    for (ftype, unit), inst in sets.items():
        for group, g in (("double_read", inst[inst["n"] >= 2]), ("all_cases", inst)):
            if len(g) and g["k"].sum() > 0:
                rows.append({"finding_type": ftype, "unit": unit, "group": group,
                             **compare(g, n_boot=a.n_boot, seed=a.seed)})
    res = pd.DataFrame(rows)
    main_units = {"tooth, all permanent", "case"}
    head = res[(res["group"] == "double_read") & res["unit"].isin(main_units)] if len(res) else res
    complete = set(head["finding_type"]) == set(GT_TYPES) if len(head) else False
    summary = {"main_rows_complete": complete,
               "readers_vs_mask": dict(zip(head["finding_type"], head["readers_vs_mask"])),
               "types_with_ci_overlap": head.loc[head["ci_overlap"], "finding_type"].tolist(),
               "phi_mask": dict(zip(head["finding_type"], head["phi_mask"].round(4))),
               "phi_mask_hi": dict(zip(head["finding_type"], head["phi_mask_hi"].round(4))),
               "guidance_overlap_2_of_3": bool(complete and head["ci_overlap"].sum() >= 2),
               "guidance_phi_hi_le_max": bool(complete and (head["phi_mask_hi"] <= a.phi_max).all()),
               "min_mm3": a.min_mm3, "region_mode": a.region_mode, "upper_gap_rule": a.upper_gap_rule,
               "upper_bone_min_mm3": a.upper_bone_min_mm3, "n_failed_or_missing_reports": len(bad)}
    run_id = make_run_id(a.tag)
    out = a.results_dir / "propensity" / run_id
    out.mkdir(parents=True, exist_ok=False)
    res.to_csv(out / "mask_agreement.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"run_id": run_id, **summary}, indent=2, default=str))
    write_provenance(out, extra={"run_id": run_id, "parse_run": str(a.parse_run), "manifest": str(a.manifest),
                                 "vis": [str(v) for v in a.vis], "n_boot": a.n_boot, "seed": a.seed,
                                 "mirror_centres": a.mirror_centres, "excluded": a.exclude})
    cols = ["finding_type", "unit", "group", "n_cases", "n_mask_positive", "pi_mask", "pi_mask_lo", "pi_mask_hi",
            "pi_readers", "pi_readers_lo", "pi_readers_hi", "diff_lo", "diff_hi", "readers_vs_mask", "phi_mask",
            "phi_mask_hi"]
    print(res[cols].round(3).to_string(index=False))
    print(json.dumps({"run_id": run_id, **summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
