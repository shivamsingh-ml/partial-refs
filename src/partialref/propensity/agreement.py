"""GT/T2: reader-agreement counts and reporting propensity per finding type (HANDOFF §2.1, §2.6, §7 GT).

Inputs (no report text anywhere):
- per-report parser instances (case_id, report_id, finding_type, slot) from partialref.extract.per_report;
- the list of reports per case (manifest) — a report with no findings is still a reader;
- image-only visibility features (partialref.audit.visibility_features) for P/F/S (and optionally A).

For every case read by n >= 1 parsed reports and every VISIBLE candidate instance (finding type x slot),
k = number of reports mentioning it. Candidate slots: permanent teeth 11-48, sides, quadrants (primary
teeth are excluded: too rare). Cases where any report failed to parse are dropped (reader unknown).

Provisional visibility (GT; final thresholds are frozen later in M1; recorded in DECISIONS.md):
- tooth must be present (label): Caries, EndodonticTreatment, ConservativeRestoration,
  ToothDevelopmentPathology, PeriapicalLesion (+ root tip not cut by the scan boundary);
- slot must be assessable (HANDOFF §2.6 missing-tooth rule): ToothAbsence, DentalImplant,
  ProstheticDevice, PostExtractionSocket, RootResidue -> arch in the scan AND (slot lies between present
  teeth of its quadrant OR that quadrant's teeth do not touch a lateral/posterior face); an UPPER slot
  without a tooth in the mask additionally needs the upper jaw bone in the mask: maxilla label of at least
  --upper-bone-min-mm3 (default 2000 mm3, just below the smallest maxilla of the tall centre-F scans, which
  show the whole upper jaw; 2026-11-11: in short scans only the tips of a few upper teeth, or a thin strip
  of bone, may reach into the picture, so "no tooth label" there means "not scanned");
- MandibularCanalCourse: canal label of that side present; MaxillarySinusFinding: sinus label present
  (provisional: the agreed "% of a complete sinus" rule needs a reference volume, still open);
- MandibularCondyleFinding: BOTH mandible halves touch neither the top nor the posterior face (one value
  for both sides until laterality is resolved — canal/sinus sides follow TF3 label names, the mandible
  halves follow world x, and DECISIONS records these as opposite);
- JawAtrophy, PathologicalLesion: the jaw of that quadrant is in the scan.
Headline fit uses double-read instances only (n >= 2); the all-instances fit is a sensitivity row.
pi is reported as not estimable (NaN) when no instance has n >= 2 and k >= 1, also per bootstrap resample.

Fit per finding type (pooled over the selected centres): MLE (phi = 0) of (p, pi) on all instances,
n = 2 closed form as a check, and case-level bootstrap CIs computed from per-case (n, k) count tables.

Usage:
    python -m partialref.propensity.agreement --parse-run results/extract/<run> --lang en \
        --manifest results/manifest/<run> --vis results/audit/<run P/F/S> [--vis results/audit/<run A>] \
        [--centres PFS] [--exclude P116,P117,P260,P261] [--n-boot 1000] [--by-centre] [--mirror-centres PFS]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from partialref.extract.ontology import REGION_PREFIX, REGION_TEETH, TOOTH_PRESENT_TYPES, FindingType, slot_kind
from partialref.extract.per_report import report_ok
from partialref.propensity.closed_form import closed_form_n2
from partialref.propensity.mle import fit_mle
from partialref.provenance import make_run_id, write_provenance

PERMANENT = [f"{q}{i}" for q in (1, 2, 3, 4) for i in range(1, 9)]
UPPER = {t for t in PERMANENT if t[0] in "12"}
QUADRANTS = {"QuadrantI": "upper", "QuadrantII": "upper", "QuadrantIII": "lower", "QuadrantIV": "lower"}
NEEDS_TOOTH = {"Caries", "EndodonticTreatment", "ConservativeRestoration", "ToothDevelopmentPathology",
               "PeriapicalLesion"}
NEEDS_ARCH = {"ToothAbsence", "DentalImplant", "ProstheticDevice", "PostExtractionSocket", "RootResidue"}


# ---------------------------------------------------------------- visibility
MIRROR_Q = {"1": "2", "2": "1", "3": "4", "4": "3"}


def _mirror_tooth(t: str) -> str:
    return MIRROR_Q[t[0]] + t[1]


UPPER_BONE_MIN_MM3 = 2000.0
UPPER_GAP_RULES = ("bone", "teeth_or_bone")


def load_visibility(vis_dirs: list[Path], mirror_centres: str = "", swap_side_centres: str = "",
                    upper_bone_min_mm3: float = UPPER_BONE_MIN_MM3) -> dict[str, dict]:
    """Per case: arches in the scan, teeth present, teeth with cut root tip, quadrant edge contact,
    side structures. Cases of ``mirror_centres`` get left/right FDI quadrants of the mask teeth swapped
    (1<->2, 3<->4): TF3 tooth labels are mirrored relative to the reports (DECISIONS 2026-11-08/09).
    Cases of ``swap_side_centres`` get canal/sinus Left<->Right swapped (set from partialref.audit.side_geometry;
    the CLI default is PFS, DECISIONS 2026-11-09). Condyles use one value for both sides, so no swap is needed."""
    out: dict[str, dict] = {}
    for d in vis_dirs:
        cf = pd.read_parquet(d / "case_features.parquet")
        tf = pd.read_parquet(d / "tooth_features.parquet")
        if tf.empty or "present" not in tf:
            tf = pd.DataFrame(columns=["case_id", "fdi", "present", "root_tip_cut"])
        teeth = tf[tf["present"] == True].assign(fdi=lambda x: x["fdi"].astype(int).astype(str))  # noqa: E712
        flip = teeth["case_id"].str[0].isin(list(mirror_centres))
        teeth.loc[flip, "fdi"] = teeth.loc[flip, "fdi"].map(_mirror_tooth)
        present = teeth.groupby("case_id")["fdi"].agg(set).to_dict()
        cut = teeth[teeth["root_tip_cut"] == True].groupby("case_id")["fdi"].agg(set).to_dict()  # noqa: E712
        for r in cf.to_dict("records"):
            c = r["case_id"]
            if not r.get("label_found", False):
                continue
            pres = present.get(c, set())

            def vox(name: str) -> float:
                v = r.get(f"{name}_vox", 0)
                return 0 if v is None or (isinstance(v, float) and np.isnan(v)) else v

            def false_(key: str) -> bool:      # True only if the flag exists and is False
                return r.get(key) is False

            def jaw_inside(jaw: str) -> bool:   # both halves clear of lateral and posterior faces
                return vox(jaw) > 0 and all(false_(f"{jaw}_{h}_touches_{f}") for h in ("left", "right")
                                            for f in ("lateral", "post"))

            # quadrant with mask teeth: do its teeth touch a lateral/posterior face? Quadrant WITHOUT mask
            # teeth (review 2026-11-09 #4): only assessable if the whole jaw stays clear of those faces
            # (side-independent, so unaffected by the L/R conventions); missing flags -> not assessable.
            edge = {q: (not false_(f"q{q}_teeth_touch_lateral_or_post")) if (r.get(f"q{q}_n_teeth") or 0) > 0
                    else not jaw_inside("maxilla" if q in "12" else "mandible") for q in "1234"}
            if c[0] in mirror_centres:
                edge = {q: edge[MIRROR_Q[q]] for q in "1234"}
            condyles = (vox("mandible") > 0 and all(false_(f"mandible_{h}_touches_sup") and
                                                     false_(f"mandible_{h}_touches_post") for h in ("left", "right")))
            out[c] = {
                "upper": vox("maxilla") > 0 or bool(pres & UPPER),
                "upper_bone": vox("maxilla") > 0 and (r.get("maxilla_mm3") or 0) >= upper_bone_min_mm3,
                "lower": vox("mandible") > 0 or bool(pres - UPPER),
                "teeth": pres, "root_cut": cut.get(c, set()), "quadrant_edge": edge,
                "canal_Left": vox("canal_left") > 0, "canal_Right": vox("canal_right") > 0,
                "sinus_Left": vox("sinus_left") > 0, "sinus_Right": vox("sinus_right") > 0,
                "condyle_Left": condyles, "condyle_Right": condyles,
            }
            if c[0] in swap_side_centres:
                for s in ("canal", "sinus"):
                    out[c][f"{s}_Left"], out[c][f"{s}_Right"] = out[c][f"{s}_Right"], out[c][f"{s}_Left"]
    return out


def is_visible(v: dict, ftype: str, slot: str, upper_gap_rule: str = "bone") -> bool:
    """upper_gap_rule: "bone" (main, 2026-11-11) = an upper slot without a mask tooth also needs the upper
    jaw bone label; "teeth_or_bone" = previous rule (any upper tooth or the bone puts the upper arch in the
    scan). Dicts without "upper_bone" (older callers) fall back to "upper"."""
    if upper_gap_rule not in UPPER_GAP_RULES:
        raise ValueError(f"upper_gap_rule must be one of {UPPER_GAP_RULES}, got {upper_gap_rule!r}")
    kind = slot_kind(FindingType(ftype))
    if kind == "tooth":
        arch = "upper" if slot in UPPER else "lower"
        if ftype in NEEDS_TOOTH:
            ok = slot in v["teeth"]
            if ftype == "PeriapicalLesion":
                ok = ok and slot not in v["root_cut"]
            return ok
        if arch == "upper" and upper_gap_rule == "bone" and slot not in v["teeth"] \
                and not v.get("upper_bone", v["upper"]):
            return False
        return v[arch] and slot_assessable(v, slot)
    if kind == "quadrant":
        return v[QUADRANTS[slot]]
    if ftype == "MandibularCanalCourse":
        return v[f"canal_{slot}"]
    if ftype == "MaxillarySinusFinding":
        return v[f"sinus_{slot}"]
    if ftype == "MandibularCondyleFinding":
        return v[f"condyle_{slot}"]
    return False


def slot_assessable(v: dict, slot: str) -> bool:
    """HANDOFF §2.6 missing-tooth rule: a tooth position is assessable if it lies between present teeth
    of its quadrant, or if that quadrant's teeth do not touch a lateral/posterior face of the scan."""
    q, pos = slot[0], int(slot[1])
    present_pos = [int(t[1]) for t in v["teeth"] if t[0] == q]
    between = bool(present_pos) and min(present_pos) < pos < max(present_pos)
    return between or not v.get("quadrant_edge", {}).get(q, True)


def candidate_slots(ftype: str) -> list[str]:
    kind = slot_kind(FindingType(ftype))
    return PERMANENT if kind == "tooth" else (["Left", "Right"] if kind == "side" else list(QUADRANTS))


# ---------------------------------------------------------------- instances
def build_instances(reports: pd.DataFrame, mentions: pd.DataFrame, vis: dict[str, dict],
                    failed_reports: set[tuple[str, str]], types: list[str],
                    region_mode: str = "omission", upper_gap_rule: str = "bone") -> tuple[pd.DataFrame, dict]:
    """reports: case_id, report_id (all EN reports of the selected cases).
    mentions: case_id, report_id, finding_type, slot (parser output)."""
    bad_cases = {c for c, _ in failed_reports}
    no_vis = set(reports["case_id"]) - set(vis.keys())
    rep = reports[~reports["case_id"].isin(bad_cases) & reports["case_id"].isin(vis.keys())]
    n_by_case = rep.groupby("case_id")["report_id"].nunique().to_dict()
    m = mentions.merge(rep[["case_id", "report_id"]], on=["case_id", "report_id"]) \
        .drop_duplicates(["case_id", "report_id", "finding_type", "slot"])
    k_counts = m.groupby(["case_id", "finding_type", "slot"]).size().to_dict()
    if region_mode not in ("omission", "uninformative"):
        raise ValueError(f"region_mode must be 'omission' or 'uninformative', got {region_mode!r}")
    # region-level mentions (parser v3, e.g. "partial edentulism of the upper arch"): for each tooth of the
    # region that this report does not name, the reader is uninformative (neither mention nor omission)
    # -> that instance's n drops by one for this reader instead of recording a false omission.
    # region_mode (decided by Shivam 2026-11-09): "omission" = MAIN analysis (B): the estimand is tooth-level
    # completeness of the reference, so a vague reader does not have that tooth -> plain n. "uninformative" =
    # secondary (A): reporting propensity of readers who name teeth; the vague reader drops out for that tooth.
    named = set(map(tuple, m[["case_id", "report_id", "finding_type", "slot"]].to_numpy()))
    # a tooth named with a type that implies it is present: that reader is informative ("not missing")
    named_present = set(map(tuple, m.loc[m["finding_type"].isin(TOOTH_PRESENT_TYPES),
                                         ["case_id", "report_id", "slot"]].to_numpy()))
    is_region = m["slot"].astype(str).str.startswith(REGION_PREFIX)
    uninformative = set() if region_mode == "omission" else {
        (c, r, t, tooth) for c, r, t, sl in m.loc[is_region, ["case_id", "report_id", "finding_type",
                                                              "slot"]].itertuples(index=False)
        for tooth in REGION_TEETH.get(sl[len(REGION_PREFIX):], [])
        if (c, r, t, tooth) not in named and (c, r, tooth) not in named_present}
    n_uninf: dict = defaultdict(int)
    for c, _, t, tooth in uninformative:
        n_uninf[(c, t, tooth)] += 1
    rows, invisible_mentions = [], defaultdict(int)
    for (c, t, s), k in k_counts.items():
        if t in types and s in set(candidate_slots(t)) and not is_visible(vis[c], t, s, upper_gap_rule):
            invisible_mentions[t] += int(k)
    for c, n in n_by_case.items():
        v = vis[c]
        for t in types:
            for s in candidate_slots(t):
                n_eff = n - n_uninf.get((c, t, s), 0)
                if n_eff >= 1 and is_visible(v, t, s, upper_gap_rule):
                    rows.append((c, c[0], t, s, n_eff, k_counts.get((c, t, s), 0)))
    inst = pd.DataFrame(rows, columns=["case_id", "centre", "finding_type", "slot", "n", "k"])
    info = {"cases_used": int(inst["case_id"].nunique()) if len(inst) else 0,
            "cases_without_visible_instances": len(n_by_case) - (int(inst["case_id"].nunique()) if len(inst) else 0),
            "cases_dropped_failed_parse": len(bad_cases & set(reports.case_id)),
            "cases_dropped_no_visibility_features": len(no_vis - bad_cases),
            "mentions_on_invisible_slots": dict(invisible_mentions),
            "region_mentions": int(is_region.sum()),
            "reader_slot_pairs_uninformative_by_region": len(uninformative),
            "mentions_total": {t: int(m[m.finding_type == t].shape[0]) for t in types},
            "mentions_total_region_level": m.loc[is_region, "finding_type"].value_counts().to_dict(),
            "region_mode": region_mode, "upper_gap_rule": upper_gap_rule}
    return inst, info


def select_reports(manifest: Path, parse_run: Path, lang: str, centres: str,
                   exclude: list[str]) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """Reports of the selected centres (minus exclusions) and the (case, report) pairs whose parse failed
    or is missing. Callers drop every case with such a report (reader unknown)."""
    reps = pd.read_parquet(manifest / "reports.parquet")
    reps = reps[(reps["lang"] == lang) & reps["case_id"].str[0].isin(list(centres))
                & ~reps["case_id"].isin(exclude)]
    meta = pd.read_parquet(parse_run / f"extractions_{lang}.parquet")
    # failed = parse error or a finding dropped during salvage (report_ok; NaN-safe for pandas >= 3)
    failed = set(map(tuple, meta.loc[~report_ok(meta), ["case_id", "report_id"]].to_numpy()))
    parsed = set(map(tuple, meta[["case_id", "report_id"]].to_numpy()))
    missing = {(c, r) for c, r in reps[["case_id", "report_id"]].itertuples(index=False)} - parsed
    return reps, failed | missing


# ---------------------------------------------------------------- fitting
def _case_tables(inst: pd.DataFrame, n_max: int) -> tuple[list[str], np.ndarray]:
    """Per case a (n_max+1) x (n_max+1) table of instance counts by (n, k)."""
    cases = sorted(inst["case_id"].unique())
    idx = {c: i for i, c in enumerate(cases)}
    tab = np.zeros((len(cases), n_max + 1, n_max + 1))
    np.add.at(tab, (inst["case_id"].map(idx).to_numpy(), inst["n"].to_numpy(), inst["k"].to_numpy()), 1)
    return cases, tab


def _fit_table(t: np.ndarray) -> tuple[float, float, float]:
    """(p, pi, pi_closed_form). NaN when pi is not identifiable (no instance with n >= 2 and k >= 1)."""
    n_idx, k_idx = np.nonzero(t)
    w = t[n_idx, k_idx]
    if len(w) == 0:
        return np.nan, np.nan, np.nan
    res = fit_mle(n_idx, k_idx, w)
    cf = closed_form_n2(n_idx, k_idx, w)
    if not res.identifiable:
        return np.nan, np.nan, cf.pi
    return res.p, res.pi, cf.pi


def fit_type(inst: pd.DataFrame, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
             min_both: int = 10, min_multi_mentioned: int = 30) -> dict:
    """Fit pi, p with case-level bootstrap CIs. GT flags are only set with enough double-read support:
    >= min_both n=2 'both' instances and >= min_multi_mentioned mentioned instances with n >= 2."""
    n_max = int(inst["n"].max())
    cases, tab = _case_tables(inst, n_max)
    total = tab.sum(axis=0)
    p, pi, pi_cf = _fit_table(total)
    two = total[2] if n_max >= 2 else np.zeros(n_max + 1)
    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, 2), np.nan)
    for b in range(n_boot):
        sel = rng.integers(0, len(cases), len(cases))
        bp, bpi, _ = _fit_table(tab[sel].sum(axis=0))
        boot[b] = (bp, bpi)
    ok = np.isfinite(boot[:, 1])
    lo, hi = np.quantile(boot[ok, 1], [alpha / 2, 1 - alpha / 2]) if ok.any() else (np.nan, np.nan)
    plo, phi_ = np.quantile(boot[ok, 0], [alpha / 2, 1 - alpha / 2]) if ok.any() else (np.nan, np.nan)
    multi_mentioned = int(total[2:, 1:].sum()) if n_max >= 2 else 0
    both = int(two[2]) if n_max >= 2 else 0
    supported = bool(np.isfinite(pi) and both >= min_both and multi_mentioned >= min_multi_mentioned
                     and ok.mean() >= 0.95)
    return {"n_cases": len(cases), "n_instances": int(total.sum()),
            "n_mentioned_instances": int(total[:, 1:].sum()), "n_multi_read_mentioned": multi_mentioned,
            "supported": supported, "boot_non_identified_share": float(1 - ok.mean()),
            "two_reader_both": int(two[2]) if n_max >= 2 else 0,
            "two_reader_one": int(two[1]) if n_max >= 2 else 0,
            "pi_mle": pi, "pi_lo": float(lo), "pi_hi": float(hi), "pi_ci_width": float(hi - lo),
            "pi_closed_form_n2": pi_cf, "p_mle": p, "p_lo": float(plo), "p_hi": float(phi_),
            "boot_ok": int(ok.sum())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse-run", required=True, type=Path)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--vis", required=True, type=Path, action="append")
    ap.add_argument("--centres", default="PFS")
    ap.add_argument("--exclude", type=lambda s: [x for x in s.split(",") if x],
                    default=["P116", "P117", "P260", "P261", "S0015"],
                    help="default: 4 cases with likely misfiled reports + S0015 (possibly upside down)")
    ap.add_argument("--mirror-centres", default="PFS",
                    help="centres whose mask tooth IDs are mirrored L/R (TF3 convention, DECISIONS); '' = none")
    ap.add_argument("--swap-side-centres", default="PFS",
                    help="centres whose canal/sinus Left/Right labels are swapped: TF3 side labels follow the "
                         "mirrored tooth convention (side_geometry 499/500, DECISIONS 2026-11-09); '' = none")
    ap.add_argument("--region-mode", choices=["omission", "uninformative"], default="omission",
                    help="region-level mentions without tooth numbers: counted as omitting the unnamed teeth (main, "
                         "tooth-level reference completeness) or reader uninformative for them (secondary)")
    ap.add_argument("--upper-gap-rule", choices=["bone", "teeth_or_bone"], default="bone",
                    help="upper slot without a mask tooth: needs the upper jaw bone label (main, 2026-11-11) or "
                         "only some upper tooth or the bone (previous rule, sensitivity)")
    ap.add_argument("--upper-bone-min-mm3", type=float, default=UPPER_BONE_MIN_MM3)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by-centre", action="store_true")
    ap.add_argument("--min-mentioned", type=int, default=30, help="flag types below this for pooling")
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="propensity-gt")
    a = ap.parse_args(argv)

    reps, bad_reports = select_reports(a.manifest, a.parse_run, a.lang, a.centres, a.exclude)
    reps = reps[["case_id", "report_id"]]
    mentions = pd.read_parquet(a.parse_run / f"instances_{a.lang}.parquet")
    vis = load_visibility(a.vis, mirror_centres=a.mirror_centres, swap_side_centres=a.swap_side_centres,
                          upper_bone_min_mm3=a.upper_bone_min_mm3)
    types = [t.value for t in FindingType]
    inst, info = build_instances(reps, mentions, vis, bad_reports, types, region_mode=a.region_mode,
                                 upper_gap_rule=a.upper_gap_rule)

    rows = []
    multi = inst[inst["n"] >= 2]
    groups = [("pooled_double_read", multi), ("pooled_all_cases", inst)] + \
        ([(f"{c}_double_read", g) for c, g in multi.groupby("centre")] if a.by_centre else [])
    for gname, g in groups:
        for t in types:
            sub = g[g["finding_type"] == t]
            if sub.empty or sub["k"].sum() == 0:
                rows.append({"group": gname, "finding_type": t, "n_instances": int(len(sub)), "note": "no mentions"})
                continue
            r = fit_type(sub, n_boot=a.n_boot, seed=a.seed, min_multi_mentioned=a.min_mentioned)
            r.update(group=gname, finding_type=t,
                     gt_upper_ci_below_0_90=bool(r["supported"] and r["pi_hi"] < 0.90),
                     gt_ci_width_le_0_30=bool(r["supported"] and r["pi_ci_width"] <= 0.30),
                     low_support=not r["supported"])
            rows.append(r)
    res = pd.DataFrame(rows)

    run_id = make_run_id(a.tag)
    out = a.results_dir / "propensity" / run_id
    out.mkdir(parents=True, exist_ok=False)
    res.to_csv(out / "propensity_by_type.csv", index=False)
    inst.to_parquet(out / "instances_nk.parquet", index=False)
    pooled = res[res["group"] == "pooled_double_read"].dropna(subset=["pi_mle"]) if "pi_mle" in res else res.iloc[0:0]
    summary = {"run_id": run_id, "centres": a.centres, "excluded": a.exclude, "mirror_centres": a.mirror_centres,
               "swap_side_centres": a.swap_side_centres, "upper_bone_min_mm3": a.upper_bone_min_mm3, **info,
               "n_failed_or_missing_reports": len(bad_reports),
               "types_passing_both_guidance_checks": pooled[pooled.gt_upper_ci_below_0_90 & pooled.gt_ci_width_le_0_30]
               ["finding_type"].tolist() if len(pooled) else []}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_provenance(out, extra={"run_id": run_id, "parse_run": str(a.parse_run), "manifest": str(a.manifest),
                                 "vis": [str(v) for v in a.vis], "n_boot": a.n_boot, "seed": a.seed,
                                 "mirror_centres": a.mirror_centres, "excluded": a.exclude})
    cols = ["group", "finding_type", "n_cases", "n_multi_read_mentioned", "two_reader_both", "two_reader_one",
            "pi_mle", "pi_lo", "pi_hi", "pi_closed_form_n2", "p_mle", "supported"]
    print(res[[c for c in cols if c in res]].round(3).to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
