"""nnU-Net plans and folds for retraining V2V2R's segmentation network on our training scans (2026-11-17).

Plans: V2V2R's plans file (`nnUNetResEncUNetLPlans_torchres`, Dataset703; identical to the trained model's
`plans.json`) copied with exactly two changes:
- `dataset_name` -> our dataset (Dataset713_TF3SinglePulpDev);
- `foreground_intensity_properties_per_channel` -> the values nnU-Net's fingerprint step computed on OUR 424
  training scans (the original values came from all 532 ToothFairy3 scans, test scans included). CTNormalization
  clips to percentile_00_5 / percentile_99_5 and scales by mean / std of these values.
Everything else (configurations, network, patch size, spacing, resampling, normalization scheme, data identifier)
stays identical; the script checks this key by key and stops otherwise. nnU-Net's own
`nnUNetv2_move_plans_between_datasets` is not used because it keeps the source intensity values.
The fingerprint's median spacing / shape are reported next to the plans' (informational; the configurations fix
spacing and patch size themselves).

Also copies the raw `dataset.json` into the preprocessed folder (nnU-Net's planner would; we skip the planner).

Folds (`splits_final.json`): the five development folds of `configs/splits/v1`, minus the fix table's exclusions
(S0015); the ToothFairy3-only scans are in every training fold and never in a validation fold; no test or centre-A
scan anywhere (checked).

    python -m partialref.baseline.seg_plans --source-plans <U-Mamba2>/nnunet/nnUNet_preprocessed/Dataset703.../nnUNetResEncUNetLPlans_torchres.json \\
        --fingerprint data/nnunet/preprocessed/Dataset713_TF3SinglePulpDev/dataset_fingerprint.json \\
        --dataset Dataset713_TF3SinglePulpDev --splits configs/splits/v1 --fixes configs/seg/label_fixes_v1.json \\
        --raw data/nnunet/raw/Dataset713_TF3SinglePulpDev --out-dir data/nnunet/preprocessed/Dataset713_TF3SinglePulpDev
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHANGED = ("dataset_name", "foreground_intensity_properties_per_channel")
INTENSITY_KEYS = ("max", "mean", "median", "min", "percentile_00_5", "percentile_99_5", "std")


def make_plans(source: dict, fingerprint: dict, dataset: str) -> dict:
    fp = fingerprint["foreground_intensity_properties_per_channel"]
    if set(fp) != set(source["foreground_intensity_properties_per_channel"]):
        raise ValueError(f"channels differ: {sorted(fp)} vs {sorted(source['foreground_intensity_properties_per_channel'])}")
    for ch, v in fp.items():
        missing = [k for k in INTENSITY_KEYS if k not in v]
        if missing or not all(np.isfinite(v[k]) for k in INTENSITY_KEYS):
            raise ValueError(f"channel {ch}: bad intensity properties {v}")
        if not (v["min"] <= v["percentile_00_5"] < v["mean"] < v["percentile_99_5"] <= v["max"]) or v["std"] <= 0:
            raise ValueError(f"channel {ch}: implausible intensity properties {v}")
    out = copy.deepcopy(source)
    out["dataset_name"] = dataset
    out["foreground_intensity_properties_per_channel"] = copy.deepcopy(fp)
    diff = [k for k in set(source) | set(out) if k not in CHANGED and source.get(k) != out.get(k)]
    if diff:
        raise AssertionError(f"plans differ outside {CHANGED}: {diff}")
    return out


def make_folds(splits: Path, fixes: dict, built_cases: set[str]) -> list[dict]:
    s = pd.read_csv(splits / "splits.csv")
    meta = json.loads((splits / "splits.json").read_text())
    excluded = set(fixes.get("exclude", {}))
    name = lambda c: f"ToothFairy3{c[0]}_{c[1:]}"
    dev = s[(s["role"] == "dev") & ~s["case_id"].isin(excluded)]
    extra = [name(c) for c in meta["tf3_only_for_segmentation"] if c not in excluded]
    folds = [{"train": sorted([name(c) for c in dev.loc[dev["fold"] != k, "case_id"]] + extra),
              "val": sorted(name(c) for c in dev.loc[dev["fold"] == k, "case_id"])} for k in range(5)]
    banned = {name(c) for c in s.loc[s["role"] != "dev", "case_id"] if c[0] != "A"} | {name(c) for c in excluded}
    used = set().union(*[set(f["train"]) | set(f["val"]) for f in folds])
    if used & banned:
        raise AssertionError(f"test / excluded scans in folds: {sorted(used & banned)[:5]}")
    if used != built_cases:
        raise AssertionError(f"folds and built dataset differ: {sorted(used ^ built_cases)[:5]}")
    vals = [set(f["val"]) for f in folds]
    if set().union(*vals) & set(extra) or sum(len(v) for v in vals) != len(set().union(*vals)):
        raise AssertionError("validation folds overlap or contain ToothFairy3-only scans")
    for f in folds:
        if set(f["train"]) & set(f["val"]):
            raise AssertionError("a scan is in training and validation of the same fold")
    return folds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-plans", required=True, type=Path)
    ap.add_argument("--fingerprint", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--fixes", required=True, type=Path)
    ap.add_argument("--raw", required=True, type=Path, help="nnU-Net raw dataset folder (case list)")
    ap.add_argument("--out-dir", required=True, type=Path, help="nnU-Net preprocessed dataset folder")
    ap.add_argument("--report", type=Path, help="where to write plans_report.json (default: out-dir)")
    a = ap.parse_args(argv)
    source, fingerprint = json.loads(a.source_plans.read_text()), json.loads(a.fingerprint.read_text())
    plans = make_plans(source, fingerprint, a.dataset)
    fixes = json.loads(a.fixes.read_text())
    built = {p.name.removesuffix(".nii.gz") for p in (a.raw / "labelsTr").glob("*.nii.gz")}
    folds = make_folds(a.splits, fixes, built)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plans_path = a.out_dir / f"{source['plans_name']}.json"
    if plans_path.exists():
        raise FileExistsError(f"{plans_path} exists")
    plans_path.write_text(json.dumps(plans, indent=4))
    (a.out_dir / "splits_final.json").write_text(json.dumps(folds, indent=2))
    # nnUNetv2_preprocess and training read <preprocessed>/<dataset>/dataset.json; normally the planner copies it
    (a.out_dir / "dataset.json").write_text((a.raw / "dataset.json").read_text())
    sp = np.array(fingerprint["spacings"], float)
    sh = np.array(fingerprint["shapes_after_crop"], float)
    report = {"source_plans": str(a.source_plans), "fingerprint": str(a.fingerprint), "plans": str(plans_path),
              "changed_keys": list(CHANGED),
              "intensity_source": source["foreground_intensity_properties_per_channel"],
              "intensity_ours": plans["foreground_intensity_properties_per_channel"],
              "fingerprint_n_cases": len(fingerprint["spacings"]),
              "fingerprint_median_spacing": np.median(sp, 0).tolist(),
              "plans_median_spacing": source["original_median_spacing_after_transp"],
              "fingerprint_median_shape": np.median(sh, 0).tolist(),
              "plans_median_shape": source["original_median_shape_after_transp"],
              "folds": [{"train": len(f["train"]), "val": len(f["val"])} for f in folds]}
    (a.report or a.out_dir / "plans_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("intensity_source", "intensity_ours", "fingerprint_n_cases",
                                             "fingerprint_median_spacing", "plans_median_spacing", "folds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
