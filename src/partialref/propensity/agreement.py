"""GT/T2: reader-agreement counts and reporting propensity per finding type (HANDOFF §2.1, §2.6, §7 GT).

Inputs (no report text anywhere):
- per-report parser instances (case_id, report_id, finding_type, slot) from partialref.extract.per_report;
- the list of reports per case (manifest) — a report with no findings is still a reader;
- image-only visibility features (partialref.audit.visibility_features) for P/F/S (and optionally A).

For every case read by n >= 1 parsed reports and every VISIBLE candidate instance (finding type x slot),
k = number of reports mentioning it. Candidate slots: permanent teeth 11-48, sides, quadrants (primary
teeth are excluded: too rare). Cases where any report failed to parse are dropped (reader unknown).

Provisional visibility (GT; final thresholds are frozen later in M1):
- tooth must be present (label): Caries, EndodonticTreatment, ConservativeRestoration,
  ToothDevelopmentPathology, PeriapicalLesion (+ root tip not cut by the scan boundary);
- arch must be in the scan (jaw label or any tooth of that arch): ToothAbsence, DentalImplant,
  ProstheticDevice, PostExtractionSocket, RootResidue;
- MandibularCanalCourse: canal label of that side present; MaxillarySinusFinding: sinus label present;
- MandibularCondyleFinding: that mandible half does not touch the top of the scan;
- JawAtrophy, PathologicalLesion: the jaw of that quadrant is in the scan.

Fit per finding type (pooled over the selected centres): MLE (phi = 0) of (p, pi) on all instances,
n = 2 closed form as a check, and case-level bootstrap CIs computed from per-case (n, k) count tables.

Usage:
    python -m partialref.propensity.agreement --parse-run results/extract/<run> --lang en \
        --manifest results/manifest/<run> --vis results/audit/<run P/F/S> [--vis results/audit/<run A>] \
        [--centres PFS] [--exclude P116,P117,P260,P261] [--n-boot 1000] [--by-centre]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from partialref.extract.ontology import FindingType, slot_kind
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
def load_visibility(vis_dirs: list[Path]) -> dict[str, dict]:
    """Per case: arches in the scan, teeth present, teeth with cut root tip, side structures."""
    out: dict[str, dict] = {}
    for d in vis_dirs:
        cf = pd.read_parquet(d / "case_features.parquet")
        tf = pd.read_parquet(d / "tooth_features.parquet")
        teeth = tf[tf["present"]].assign(fdi=lambda x: x["fdi"].astype(str))
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

            out[c] = {
                "upper": vox("maxilla") > 0 or bool(pres & UPPER),
                "lower": vox("mandible") > 0 or bool(pres - UPPER),
                "teeth": pres, "root_cut": cut.get(c, set()),
                "canal_Left": vox("canal_left") > 0, "canal_Right": vox("canal_right") > 0,
                "sinus_Left": vox("sinus_left") > 0, "sinus_Right": vox("sinus_right") > 0,
                "condyle_Left": vox("mandible") > 0 and r.get("mandible_left_touches_sup") is False,
                "condyle_Right": vox("mandible") > 0 and r.get("mandible_right_touches_sup") is False,
            }
    return out


def is_visible(v: dict, ftype: str, slot: str) -> bool:
    kind = slot_kind(FindingType(ftype))
    if kind == "tooth":
        arch = "upper" if slot in UPPER else "lower"
        if ftype in NEEDS_TOOTH:
            ok = slot in v["teeth"]
            if ftype == "PeriapicalLesion":
                ok = ok and slot not in v["root_cut"]
            return ok
        return v[arch]
    if kind == "quadrant":
        return v[QUADRANTS[slot]]
    if ftype == "MandibularCanalCourse":
        return v[f"canal_{slot}"]
    if ftype == "MaxillarySinusFinding":
        return v[f"sinus_{slot}"]
    if ftype == "MandibularCondyleFinding":
        return v[f"condyle_{slot}"]
    return False


def candidate_slots(ftype: str) -> list[str]:
    kind = slot_kind(FindingType(ftype))
    return PERMANENT if kind == "tooth" else (["Left", "Right"] if kind == "side" else list(QUADRANTS))


# ---------------------------------------------------------------- instances
def build_instances(reports: pd.DataFrame, mentions: pd.DataFrame, vis: dict[str, dict],
                    failed_reports: set[tuple[str, str]], types: list[str]) -> tuple[pd.DataFrame, dict]:
    """reports: case_id, report_id (all EN reports of the selected cases).
    mentions: case_id, report_id, finding_type, slot (parser output)."""
    bad_cases = {c for c, _ in failed_reports}
    rep = reports[~reports["case_id"].isin(bad_cases) & reports["case_id"].isin(vis.keys())]
    n_by_case = rep.groupby("case_id")["report_id"].nunique().to_dict()
    m = mentions.merge(rep[["case_id", "report_id"]], on=["case_id", "report_id"])
    k_counts = m.drop_duplicates(["case_id", "report_id", "finding_type", "slot"]) \
        .groupby(["case_id", "finding_type", "slot"]).size().to_dict()
    rows, invisible_mentions = [], defaultdict(int)
    for (c, t, s), k in k_counts.items():
        if t in types and s in set(candidate_slots(t)) and not is_visible(vis[c], t, s):
            invisible_mentions[t] += int(k)
    for c, n in n_by_case.items():
        v = vis[c]
        for t in types:
            for s in candidate_slots(t):
                if is_visible(v, t, s):
                    rows.append((c, c[0], t, s, n, k_counts.get((c, t, s), 0)))
    inst = pd.DataFrame(rows, columns=["case_id", "centre", "finding_type", "slot", "n", "k"])
    info = {"cases_used": len(n_by_case), "cases_dropped_failed_parse": len(bad_cases & set(reports.case_id)),
            "mentions_on_invisible_slots": dict(invisible_mentions),
            "mentions_total": {t: int(m[m.finding_type == t].shape[0]) for t in types}}
    return inst, info


# ---------------------------------------------------------------- fitting
def _case_tables(inst: pd.DataFrame, n_max: int) -> tuple[list[str], np.ndarray]:
    """Per case a (n_max+1) x (n_max+1) table of instance counts by (n, k)."""
    cases = sorted(inst["case_id"].unique())
    idx = {c: i for i, c in enumerate(cases)}
    tab = np.zeros((len(cases), n_max + 1, n_max + 1))
    np.add.at(tab, (inst["case_id"].map(idx).to_numpy(), inst["n"].to_numpy(), inst["k"].to_numpy()), 1)
    return cases, tab


def _fit_table(t: np.ndarray) -> tuple[float, float, float]:
    n_idx, k_idx = np.nonzero(t)
    w = t[n_idx, k_idx]
    res = fit_mle(n_idx, k_idx, w)
    cf = closed_form_n2(n_idx, k_idx, w)
    return res.p, res.pi, cf.pi


def fit_type(inst: pd.DataFrame, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
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
    lo, hi = np.nanquantile(boot[:, 1], [alpha / 2, 1 - alpha / 2])
    plo, phi_ = np.nanquantile(boot[:, 0], [alpha / 2, 1 - alpha / 2])
    return {"n_cases": len(cases), "n_instances": int(total.sum()),
            "n_mentioned_instances": int(total[:, 1:].sum()),
            "two_reader_both": int(two[2]) if n_max >= 2 else 0,
            "two_reader_one": int(two[1]) if n_max >= 2 else 0,
            "pi_mle": pi, "pi_lo": float(lo), "pi_hi": float(hi), "pi_ci_width": float(hi - lo),
            "pi_closed_form_n2": pi_cf, "p_mle": p, "p_lo": float(plo), "p_hi": float(phi_),
            "boot_ok": int(np.isfinite(boot[:, 1]).sum())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse-run", required=True, type=Path)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--vis", required=True, type=Path, action="append")
    ap.add_argument("--centres", default="PFS")
    ap.add_argument("--exclude", type=lambda s: [x for x in s.split(",") if x], default=[])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by-centre", action="store_true")
    ap.add_argument("--min-mentioned", type=int, default=30, help="flag types below this for pooling")
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="propensity-gt")
    a = ap.parse_args(argv)

    reps = pd.read_parquet(a.manifest / "reports.parquet")
    reps = reps[(reps["lang"] == a.lang) & reps["case_id"].str[0].isin(list(a.centres))
                & ~reps["case_id"].isin(a.exclude)][["case_id", "report_id"]]
    meta = pd.read_parquet(a.parse_run / f"extractions_{a.lang}.parquet")
    failed = {(r.case_id, r.report_id) for r in meta.itertuples() if r.error is not None}
    parsed = {(r.case_id, r.report_id) for r in meta.itertuples()}
    missing = {(c, r) for c, r in reps.itertuples(index=False)} - parsed   # never parsed = unknown reader
    mentions = pd.read_parquet(a.parse_run / f"instances_{a.lang}.parquet")
    vis = load_visibility(a.vis)
    types = [t.value for t in FindingType]
    inst, info = build_instances(reps, mentions, vis, failed | missing, types)

    rows = []
    groups = [("pooled", inst)] + ([(c, g) for c, g in inst.groupby("centre")] if a.by_centre else [])
    for gname, g in groups:
        for t in types:
            sub = g[g["finding_type"] == t]
            if sub.empty or sub["k"].sum() == 0:
                rows.append({"group": gname, "finding_type": t, "n_instances": int(len(sub)), "note": "no mentions"})
                continue
            r = fit_type(sub, n_boot=a.n_boot, seed=a.seed)
            r.update(group=gname, finding_type=t,
                     gt_upper_ci_below_0_90=bool(r["pi_hi"] < 0.90), gt_ci_width_le_0_30=bool(r["pi_ci_width"] <= 0.30),
                     low_support=bool(r["n_mentioned_instances"] < a.min_mentioned))
            rows.append(r)
    res = pd.DataFrame(rows)

    run_id = make_run_id(a.tag)
    out = a.results_dir / "propensity" / run_id
    out.mkdir(parents=True, exist_ok=False)
    res.to_csv(out / "propensity_by_type.csv", index=False)
    inst.to_parquet(out / "instances_nk.parquet", index=False)
    pooled = res[res["group"] == "pooled"].dropna(subset=["pi_mle"]) if "pi_mle" in res else res.iloc[0:0]
    summary = {"run_id": run_id, "centres": a.centres, "excluded": a.exclude, **info,
               "n_failed_or_missing_reports": len(failed | missing),
               "types_passing_both_guidance_checks": pooled[pooled.gt_upper_ci_below_0_90 & pooled.gt_ci_width_le_0_30]
               ["finding_type"].tolist() if len(pooled) else []}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_provenance(out, extra={"run_id": run_id, "parse_run": str(a.parse_run), "manifest": str(a.manifest),
                                 "vis": [str(v) for v in a.vis], "n_boot": a.n_boot, "seed": a.seed})
    cols = ["group", "finding_type", "n_cases", "n_mentioned_instances", "two_reader_both", "two_reader_one",
            "pi_mle", "pi_lo", "pi_hi", "pi_closed_form_n2", "p_mle", "low_support"]
    print(res[[c for c in cols if c in res]].round(3).to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
