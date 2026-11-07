#!/usr/bin/env python3
"""
code/build_interrater_pairs.py

Stage the multi-radiologist cases as a (reference, prediction) corpus so the
official metric stack can score the reference reports AGAINST EACH OTHER.

59.7% of this dataset's cases carry more than one report -- two independent
radiologists describing the same scan. Scoring report _2 as if it were a
system output, with report _1 as the reference, gives the HUMAN CEILING for
RadFact Logical F1 / BLEU-4 / METEOR on this data: the score a perfect
radiologist-grade report gets when judged against a different radiologist's
wording. That number is the yardstick for every pipeline arm.

Layout written under --out-dir (default outputs/interrater_agreement/):

    ref/{case}.txt      <- {case}_1.txt   (primary reference, alphabetical)
    pred/{case}.txt     <- {case}_2.txt   (second radiologist, the "prediction")
    manifest.json       one row per pair: split, n_reports, source filenames

Why the files are COPIED and renamed rather than pointing official_ranking.py
at dataset/{split}/reports directly:

  - official_ranking.py's find_all_reference_reports() globs {case}_*.txt and
    takes [0] as primary. Aimed at the raw reports dir it would silently pull
    _2 back in as a reference under --multi-reference, i.e. score the
    prediction against itself.
  - training and validate are separate directories, but case IDs are unique
    across them (asserted below), so one staged corpus covers both splits in
    a single run.
  - Cases with 3-4 reports keep only the _1/_2 pair; the extras are recorded
    in manifest.json's n_reports but not scored, so every case contributes
    exactly one comparison and none is double-weighted.

Usage:
  python3 code/build_interrater_pairs.py
  python3 code/build_interrater_pairs.py --splits validate --out-dir /tmp/ir
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter


def case_id_of(path: Path) -> str:
    """A004_2.txt -> A004 ; A004.txt -> A004 (the _<digits> suffix is the rater)."""
    m = re.match(r"^(.*)_(\d+)$", path.stem)
    return m.group(1) if m else path.stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--splits", nargs="+", default=["training", "validate"])
    ap.add_argument("--out-dir", default="outputs/interrater_agreement")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "ref").mkdir(parents=True, exist_ok=True)
    (out / "pred").mkdir(parents=True, exist_ok=True)

    manifest, seen = [], {}
    for split in args.splits:
        reports_dir = Path(args.dataset_dir) / split / "reports"
        if not reports_dir.is_dir():
            raise SystemExit(f"[FAIL] No such reports dir: {reports_dir}")

        by_case = defaultdict(list)
        for p in sorted(reports_dir.glob("*.txt")):
            by_case[case_id_of(p)].append(p)

        for case, files in sorted(by_case.items()):
            if len(files) < 2:
                continue
            if case in seen:
                raise SystemExit(f"[FAIL] case {case} appears in both {seen[case]} and {split}")
            seen[case] = split

            files = sorted(files)
            r1, r2 = files[0], files[1]
            t1 = r1.read_text(encoding="utf-8").strip()
            t2 = r2.read_text(encoding="utf-8").strip()
            if not t1 or not t2:
                print(f"[SKIP] {case}: empty report ({r1.name} or {r2.name})")
                continue

            (out / "ref" / f"{case}.txt").write_text(t1 + "\n", encoding="utf-8")
            (out / "pred" / f"{case}.txt").write_text(t2 + "\n", encoding="utf-8")
            manifest.append({
                "case_id": case, "split": split, "n_reports": len(files),
                "reference": r1.name, "prediction": r2.name,
                "ref_chars": len(t1), "pred_chars": len(t2),
            })

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] {len(manifest)} paired case(s) -> {out}")
    print("  by split    :", dict(Counter(m["split"] for m in manifest)))
    print("  by prefix   :", dict(Counter(re.match(r"^([A-Za-z]+)", m["case_id"]).group(1)
                                          for m in manifest)))
    print("  by n_reports:", dict(Counter(m["n_reports"] for m in manifest)))


if __name__ == "__main__":
    main()
