"""Scan list for repeating U-Mamba2's self-supervised pretraining on our training scans only (2026-11-17).

Replaces U-Mamba2's `dataset_tf3.json` (all 532 ToothFairy3 images: 478 training + 54 validation, i.e. including our
locked test set) by our segmentation training scans: development cases of `configs/splits/v1` + the ToothFairy3-only
scans, minus the fix table's exclusions (S0015) = the 424 scans of nnU-Net Dataset713. Images only (pretraining uses
no labels); absolute paths to the ToothFairy3 originals. SD-Tooth is not used (not available to us).

Validation (used by their code only for the validation loss and the "best" checkpoint): about 10 % of the scans, as
in their list (54 of 532), taken deterministically per centre: cases sorted by ID, every 10th (offset 5). These
scans stay in the segmentation training set; no test or centre-A scan is in either list (checked).

    python -m partialref.baseline.pretrain_datalist --splits configs/splits/v1 --fixes configs/seg/label_fixes_v1.json \\
        --tf3-dir <TF3> --out data/pretrain/datalist_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from partialref.audit.tf3_link import tf3_paths
from partialref.baseline.seg_train_check import candidates

EVERY, OFFSET = 10, 5


def build(splits: Path, fixes: dict, tf3_dir: Path) -> dict:
    cases = [c for c in candidates(splits) if c not in fixes.get("exclude", {})]
    s = pd.read_csv(splits / "splits.csv")
    banned = set(s.loc[s["role"] != "dev", "case_id"])
    if set(cases) & banned:
        raise AssertionError(f"test / external scans in the list: {sorted(set(cases) & banned)[:5]}")
    val = []
    for centre in sorted({c[0] for c in cases}):
        cc = sorted(c for c in cases if c[0] == centre)
        val += cc[OFFSET::EVERY]
    train = sorted(set(cases) - set(val))
    rows = {}
    for key, lst in (("training", train), ("validation", sorted(val))):
        rows[key] = []
        for c in lst:
            img, _ = tf3_paths(c, tf3_dir)
            if not img.is_file():
                raise FileNotFoundError(img)
            rows[key].append({"image": str(img.resolve()), "case_id": c})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--fixes", required=True, type=Path)
    ap.add_argument("--tf3-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    d = build(a.splits, json.loads(a.fixes.read_text()), a.tf3_dir)
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(d, indent=1))
    by = {k: pd.Series([r["case_id"][0] for r in v]).value_counts().to_dict() for k, v in d.items()}
    print(json.dumps({"n_training": len(d["training"]), "n_validation": len(d["validation"]), "by_centre": by}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
