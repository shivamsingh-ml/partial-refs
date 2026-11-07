#!/usr/bin/env python3
"""
code/interrater_merge.py

Stitch jobs/interrater_eval.sh's per-shard results back into one official
summary, plus the breakdowns that make the human ceiling interpretable.

The shards exist only because radfact_lite is serial (see the header of
jobs/interrater_eval.sh); they are an execution detail, not a scientific
one. So the aggregate here is NOT an average of shard summaries -- that
would be wrong for corpus-level BLEU-4, which is computed jointly over all
candidate/reference pairs and is not the mean of anything. Instead the
per-case rows are pooled and handed to official_ranking.compute_aggregate(),
the exact function the single-process path uses. Same math, one corpus.

Breakdowns reported alongside the headline:
  - by split  (training / validate) -- is the validate split representative?
  - by prefix (A/F/P/S sub-dataset) -- ODIN2026 pools four sources whose
    reports differ in length and house style, so agreement differs too.

Outputs, under --stage-dir:
  results.jsonl   pooled per-case rows, deduplicated by case_id
  summary.json    headline aggregate + per-split/per-prefix breakdowns

Usage:
  python3 code/interrater_merge.py --stage-dir outputs/interrater_agreement
"""

import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_ranking import compute_aggregate  # noqa: E402


def load_rows(stage_dir: Path) -> dict:
    """Pool every shard's results.jsonl. Later rows win on duplicate case_id --
    a case re-scored after a judge restart should use the fresher number."""
    rows_by_id = {}
    shard_files = sorted((stage_dir / "shards").glob("shard_*/results.jsonl"))
    for f in shard_files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("case_id"):
                rows_by_id[row["case_id"]] = row
    print(f"[INFO] {len(rows_by_id)} case(s) pooled from {len(shard_files)} shard file(s)",
          file=sys.stderr)
    return rows_by_id


def subset_aggregate(case_ids, rows_by_id, pred_dir, ref_dir) -> dict:
    """compute_aggregate over a subset, keeping only the headline fields."""
    ids = [c for c in case_ids if c in rows_by_id]
    if not ids:
        return {}
    agg = compute_aggregate(ids, rows_by_id, pred_dir, ref_dir, multi_reference=False)
    return {k: agg.get(k) for k in ("num_cases", "bleu_4", "meteor", "captioning_score",
                                    "logical_precision", "logical_recall",
                                    "clinical_score", "final_score")}


def fmt(agg: dict, label: str) -> str:
    if not agg:
        return f"  {label:<12} (no cases)"
    p, r = agg.get("logical_precision"), agg.get("logical_recall")
    pr = f"P {p:.4f}  R {r:.4f}  " if p is not None else ""
    return (f"  {label:<12} n={agg['num_cases']:<4} "
            f"BLEU-4 {agg['bleu_4']:.4f}  METEOR {agg['meteor']:.4f}  "
            f"Cap {agg['captioning_score']:.4f}  {pr}"
            f"Clinical {agg['clinical_score']:.4f}  Final {agg['final_score']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", default="outputs/interrater_agreement")
    args = ap.parse_args()

    stage = Path(args.stage_dir)
    pred_dir, ref_dir = stage / "pred", stage / "ref"
    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    split_of = {m["case_id"]: m["split"] for m in manifest}

    rows_by_id = load_rows(stage)
    if not rows_by_id:
        raise SystemExit("[FAIL] No shard results found under {}/shards".format(stage))

    all_ids = sorted(rows_by_id)
    missing = [m["case_id"] for m in manifest if m["case_id"] not in rows_by_id]
    scored_rf = sum(1 for r in rows_by_id.values() if r.get("logical_f1") is not None)

    overall = compute_aggregate(all_ids, rows_by_id, pred_dir, ref_dir, multi_reference=False)

    by_split = defaultdict(list)
    by_prefix = defaultdict(list)
    for cid in all_ids:
        by_split[split_of.get(cid, "unknown")].append(cid)
        by_prefix[re.match(r"^([A-Za-z]+)", cid).group(1)].append(cid)

    summary = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "what": "inter-rater agreement: report _2 scored as prediction vs report _1 as reference",
        "num_paired_cases": len(manifest),
        "num_scored": len(all_ids),
        "num_with_radfact": scored_rf,
        "missing_cases": missing,
        "overall": overall,
        "by_split": {k: subset_aggregate(v, rows_by_id, pred_dir, ref_dir)
                     for k, v in sorted(by_split.items())},
        "by_prefix": {k: subset_aggregate(v, rows_by_id, pred_dir, ref_dir)
                      for k, v in sorted(by_prefix.items())},
    }

    (stage / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (stage / "results.jsonl").open("w", encoding="utf-8") as fh:
        for cid in all_ids:
            fh.write(json.dumps(rows_by_id[cid]) + "\n")

    print("\n" + "=" * 78)
    print("INTER-RATER AGREEMENT  (report _2 as prediction vs report _1 as reference)")
    print("=" * 78)
    print(f"Paired cases: {len(manifest)}   scored: {len(all_ids)}   with RadFact: {scored_rf}")
    if missing:
        print(f"MISSING ({len(missing)}): {' '.join(missing[:20])}"
              + (" ..." if len(missing) > 20 else ""))
    print("")
    print(fmt(overall, "OVERALL"))
    print("\nBy split:")
    for k, v in summary["by_split"].items():
        print(fmt(v, k))
    print("\nBy sub-dataset:")
    for k, v in summary["by_prefix"].items():
        print(fmt(v, k))

    stats = overall.get("per_case_stats", {})
    if stats:
        print("\nPer-case mean +/- sd (the corpus numbers above are NOT these):")
        for k in ("bleu_4", "meteor", "captioning_score", "logical_precision",
                  "logical_recall", "logical_f1", "final_score"):
            if k in stats:
                print(f"  {k:<18} {stats[k]['mean']:.4f} +/- {stats[k]['sd']:.4f}")
    print("=" * 78)
    print(f"\n[INFO] {stage/'summary.json'}\n[INFO] {stage/'results.jsonl'}")


if __name__ == "__main__":
    main()
