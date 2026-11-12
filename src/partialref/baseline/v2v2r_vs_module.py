"""Screening comparison: V2V2R's released VLM per-tooth answers vs our per-tooth module (2026-11-12).

Question: on the same teeth, outlines and labels, does the VLM already find the per-tooth findings as well as
our module? Not a paper result (released weights were trained on these scans; DECISIONS 2026-11-13 entry).

Inputs
- VLM run (`jobs/v2v2r_vlm_check.sbatch`): `<run>/<case>/<tag>/vqa/predictions/<case>_pred.json`, read
  BEFORE V2V2R's source rules and report writer. Its tooth numbers equal our `fdi_report` (manual masks
  converted with the mirrored variant, chosen by Dice in job 616828; `fdi_report` is the mirrored TF3 label).
- Classifier run (`partialref.perception.tooth_classifier`): out-of-fold `predictions.parquet` (y = report
  label, p = module probability), one feature set.

The VLM gives one yes/no per tooth, so it has one operating point. As in `human_reference`, we compare
sensitivities at the VLM's false-positive rate: d_sens = module sensitivity at the VLM's FPR minus the VLM's
sensitivity; 95 % intervals by case resampling. Teeth in our table without a VLM answer (`detected` not
"yes") count as "no" (V2V2R's report would say nothing) and are counted in `n_unanswered`; a tooth with
`detected == "yes"` but a missing or malformed field counts as "no" and in `n_malformed` (stops above 2 %).

What the VLM's "yes" means per type (reviewed 2026-11-12):
- ProstheticDevice: full crown OR post-and-core (V2V2R files a posted tooth there, never as crown) OR the tooth
  is listed as abutment/pontic of a fixed bridge (`global.fixed_bridges_*`). Our label also covers healing
  abutments, removable prostheses and orthodontic appliances, which V2V2R has no per-tooth field for.
- EndodonticTreatment: `with_endo` OR post-and-core (V2V2R: "a posted tooth reads root canal treatment");
  `n_yes_from_post` counts the teeth this adds.
- PeriapicalLesion: the question says to answer only for root-filled teeth, so "no" on other teeth may follow
  the instruction. Also reported split by our EndodonticTreatment label (rows `PeriapicalLesion|endo=0/1`).

Caveats: these are answers BEFORE V2V2R's source rules (panoramic cross-checks, arch-level additions) and
report writer, i.e. what the VLM sees per tooth, not V2V2R's report. The module's threshold is read off the
ROC curve of the same rows (slightly favours the module), as in `human_reference`.

    python -m partialref.baseline.v2v2r_vs_module --vlm-run <run> --classifier-run <run> [--tag manual]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from partialref.perception.human_reference import bootstrap

# our finding type -> V2V2R (question suffix, field) whose "yes" counts; any one true -> yes
FIELDS = {
    "ConservativeRestoration": [("morphology", "with_fillings")],
    "Caries": [("morphology", "with_caries")],
    "EndodonticTreatment": [("morphology", "with_endo"), ("morphology", "with_post_and_core")],
    "ProstheticDevice": [("morphology", "with_full_crown"), ("morphology", "with_post_and_core")],
    "PeriapicalLesion": [("endodontic_treatment", "periapical_lesion")],
}
MAX_MALFORMED = 0.02


def _bool(v) -> bool | None:
    """V2V2R's answer as True/False (its source rules also accept the string "true"); None if not a yes/no."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


def bridge_teeth(g: dict) -> set[int]:
    out = set()
    for arch in ("maxilla", "mandible"):
        for b in ((g.get(f"fixed_bridges_{arch}") or {}).get("bridges") or []):
            for k in ("abutment_teeth", "pontic_teeth"):
                out |= {int(x) for x in (b.get(k) or []) if str(x).isdigit()}
    return out


def vlm_table(run: Path, tag: str) -> pd.DataFrame:
    """One row per (case, tooth, finding type): a = VLM yes (1) / no (0); answered = tooth detected and the
    primary field is a yes/no; malformed = detected but the primary field missing or not a yes/no;
    from_post = yes only because of post-and-core."""
    rows = []
    for f in sorted(run.glob(f"*/{tag}/vqa/predictions/*_pred.json")):
        d = json.loads(f.read_text())
        bridges = bridge_teeth(d.get("global") or {})
        for key, t in d["teeth"].items():
            fdi = int(key.removeprefix("tooth_"))
            det = t.get("detected") == "yes"
            for ft, fields in FIELDS.items():
                vals = []
                for q, field in fields:
                    ans = t.get(f"{key}_{q}") if det else None
                    vals.append(_bool(ans.get(field)) if isinstance(ans, dict) else None)
                yes = any(v is True for v in vals)
                if ft == "ProstheticDevice" and det and fdi in bridges:
                    yes = True
                rows.append({"case_id": d["case_id"], "fdi_report": fdi, "finding_type": ft, "a": int(yes),
                             "answered": det and vals[0] is not None, "malformed": det and vals[0] is None,
                             "from_post": ft == "EndodonticTreatment" and yes and vals[0] is not True})
    df = pd.DataFrame(rows, columns=["case_id", "fdi_report", "finding_type", "a", "answered", "malformed",
                                     "from_post"])
    if df.duplicated(["case_id", "fdi_report", "finding_type"]).any():
        raise ValueError("duplicate (case, tooth, type) in VLM answers")
    return df


def join(P: pd.DataFrame, V: pd.DataFrame) -> pd.DataFrame:
    """Module rows of the cases the VLM ran on, with the VLM's answer (missing -> no)."""
    P = P[P["case_id"].isin(set(V["case_id"]))]
    m = P.merge(V, on=["case_id", "fdi_report", "finding_type"], how="left", validate="one_to_one")
    for c in ("answered", "malformed", "from_post"):
        m[c] = m[c].fillna(False).astype(bool)
    m["a"] = m["a"].fillna(0).astype(int)
    return m.assign(b=m["y"].astype(int), w=1.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm-run", required=True, type=Path)
    ap.add_argument("--classifier-run", required=True, type=Path)
    ap.add_argument("--feature-set", default="basic+histogram")
    ap.add_argument("--tag", default="manual")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    P = pd.read_parquet(a.classifier_run / "predictions.parquet")
    P = P[P["feature_set"] == a.feature_set]
    if P.empty or P.duplicated(["case_id", "fdi_report", "finding_type"]).any():
        raise ValueError(f"no or duplicate predictions for feature set {a.feature_set!r}")
    V = vlm_table(a.vlm_run, a.tag)
    m = join(P, V)
    share_bad = float(m["malformed"].mean()) if len(m) else 0.0
    if share_bad > MAX_MALFORMED:
        raise SystemExit(f"{share_bad:.1%} of rows have a detected tooth without a yes/no answer (> {MAX_MALFORMED:.0%})")
    endo = m[m["finding_type"] == "EndodonticTreatment"].set_index(["case_id", "fdi_report"])["b"]
    groups = [(ft, g) for ft, g in m.groupby("finding_type")]
    pa = m[m["finding_type"] == "PeriapicalLesion"]
    pa_endo = pd.Series([endo.get((c, f), -1) for c, f in zip(pa["case_id"], pa["fdi_report"])], index=pa.index)
    groups += [(f"PeriapicalLesion|endo={k}", pa[pa_endo == k]) for k in (0, 1)]
    out = []
    for ft, g in groups:
        if g["b"].nunique() < 2:
            continue
        r = bootstrap(g, a.n_boot, a.seed)
        out.append({"finding_type": ft, "n_cases": g["case_id"].nunique(), "n_teeth": len(g),
                    "n_pos": int(g["b"].sum()), "n_vlm_yes": int(g["a"].sum()),
                    "n_unanswered": int((~g["answered"]).sum()), "n_malformed": int(g["malformed"].sum()),
                    "n_yes_from_post": int(g["from_post"].sum()),
                    **{k.replace("human", "vlm"): v for k, v in r.items()}})
    res = pd.DataFrame(out)
    od = a.vlm_run / f"vs_module-{a.feature_set}-{a.tag}"
    od.mkdir(parents=True, exist_ok=True)
    res.to_csv(od / "vlm_vs_module.csv", index=False)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    (od / "provenance.json").write_text(json.dumps({
        "git_sha": sha, "vlm_run": str(a.vlm_run), "classifier_run": str(a.classifier_run),
        "feature_set": a.feature_set, "tag": a.tag, "n_boot": a.n_boot, "seed": a.seed,
        "vlm_cases": int(V["case_id"].nunique()), "cases_scored": int(m["case_id"].nunique()),
        "vlm_cases_without_module_rows": sorted(set(V["case_id"]) - set(m["case_id"]))}, indent=2))
    print(res.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
