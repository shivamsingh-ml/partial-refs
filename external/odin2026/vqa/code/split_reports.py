#!/usr/bin/env python3
"""
code/split_reports.py

Distribute the flat pool of per-case ground-truth reports
(dataset/reports/<case_id>.txt, or <case_id>_<n>.txt when a case has several
annotator reports) into the existing training/validate splits:

    dataset/training/reports/<case_id>[_<n>].txt
    dataset/validate/reports/<case_id>[_<n>].txt

Like code/split_facts.py -- and unlike code/split_dataset.py -- this does NOT
re-draw the partition. It reads the already-recorded partition from
dataset/split_manifest.json, so the validate set stays exactly the one every
existing output was produced against.

Cases present in the report pool but absent from the manifest are assigned to
`--unknown-split` (default: training), which keeps validate pinned at the
manifest's N-per-prefix.

The GT reports get regenerated in place, and the per-case filenames are not
stable across regenerations (a case that used to be `A003_1.txt` may now be
`A003.txt`). Copying alone would therefore leave the superseded file sitting in
the split next to its replacement. `--replace` deletes, for every case in the
pool, any *other* report file of that case already in either split. Without it
those files are only reported as `orphaned`.

Report files in a split whose case has no file in the pool at all are always
left untouched and reported as `stale`.

SAFE BY DEFAULT: dry-run unless --execute is passed.

Usage:
    python3 code/split_reports.py --project-dir $HOME/project_ToothFairy4
    python3 code/split_reports.py --project-dir $HOME/project_ToothFairy4 --replace --execute
"""

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# Trailing "_<n>" marks the n-th report of a case; strip it to get the case ID.
REPORT_INDEX_RE = re.compile(r"_\d+$")


def case_id_of(report_path: Path) -> str:
    return REPORT_INDEX_RE.sub("", report_path.stem)


def load_partition(manifest_path: Path) -> dict:
    """Returns {case_id: split_name} from the split manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    assignment = {}
    for split_name, groups in manifest["partition"].items():
        for case_ids in groups.values():
            for case_id in case_ids:
                assignment[case_id] = split_name
    return assignment


def main():
    ap = argparse.ArgumentParser(
        description="Distribute dataset/reports/*.txt into the training/validate "
                    "splits recorded in dataset/split_manifest.json.")
    ap.add_argument("--project-dir", required=True, help="Project root")
    ap.add_argument("--reports-dir", default=None,
                    help="Flat report pool (default: <project-dir>/dataset/reports)")
    ap.add_argument("--manifest", default=None,
                    help="Split manifest (default: <project-dir>/dataset/split_manifest.json)")
    ap.add_argument("--mode", choices=["copy", "move"], default="copy",
                    help="copy keeps the flat pool intact as the master set (default: copy)")
    ap.add_argument("--unknown-split", choices=["training", "validate", "skip"],
                    default="training",
                    help="Where to put cases not listed in the manifest (default: training)")
    ap.add_argument("--replace", action="store_true",
                    help="Delete superseded report files of pool cases from the splits "
                         "(filenames are not stable across GT regenerations). Without "
                         "this they are only reported as 'orphaned'.")
    ap.add_argument("--execute", action="store_true",
                    help="Actually perform the file operations (default: dry-run)")
    args = ap.parse_args()

    project_dir = Path(args.project_dir)
    dataset_dir = project_dir / "dataset"
    reports_dir = Path(args.reports_dir) if args.reports_dir else dataset_dir / "reports"
    manifest_path = Path(args.manifest) if args.manifest else dataset_dir / "split_manifest.json"
    dry_run = not args.execute

    if not reports_dir.is_dir():
        print(f"[FAIL] Reports dir not found: {reports_dir}", file=sys.stderr)
        sys.exit(1)
    if not manifest_path.is_file():
        print(f"[FAIL] Split manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    assignment = load_partition(manifest_path)
    pool = sorted(reports_dir.glob("*.txt"))
    pool_cases = {case_id_of(p) for p in pool}

    print(f"[INFO] {'DRY-RUN -- no files will be touched' if dry_run else 'EXECUTING'}")
    print(f"[INFO] Report pool: {reports_dir} "
          f"({len(pool)} file(s), {len(pool_cases)} case(s))")
    print(f"[INFO] Partition: {manifest_path} ({len(assignment)} case(s))")
    print(f"[INFO] Mode: {args.mode}  Unknown cases -> {args.unknown_split}  "
          f"Replace superseded: {args.replace}")
    print()

    # Where each pool case is headed, and the filenames it will own there.
    target_split = {}
    expected_names = defaultdict(set)
    unknown = []
    for src in pool:
        case_id = case_id_of(src)
        if case_id in target_split:
            expected_names[case_id].add(src.name)
            continue
        split_name = assignment.get(case_id)
        if split_name is None:
            unknown.append(case_id)
            if args.unknown_split == "skip":
                continue
            split_name = args.unknown_split
        target_split[case_id] = split_name
        expected_names[case_id].add(src.name)

    # Report files of pool cases that the new pool does not account for: either a
    # renamed predecessor in the target split, or a copy left in the other split.
    orphaned = defaultdict(list)
    stale = defaultdict(list)
    for split_name in ("training", "validate"):
        split_reports = dataset_dir / split_name / "reports"
        if not split_reports.is_dir():
            continue
        for existing in sorted(split_reports.glob("*.txt")):
            case_id = case_id_of(existing)
            if case_id not in pool_cases:
                stale[split_name].append(existing.name)
            elif (target_split.get(case_id) != split_name
                  or existing.name not in expected_names[case_id]):
                orphaned[split_name].append(existing.name)

    if args.replace:
        for split_name, names in orphaned.items():
            for name in names:
                victim = dataset_dir / split_name / "reports" / name
                if dry_run:
                    print(f"    [DRY-RUN] delete superseded: {split_name}/reports/{name}")
                    continue
                victim.unlink()
                print(f"    [DELETE] {split_name}/reports/{name}")

    placed = defaultdict(list)
    transfers = []
    for src in pool:
        case_id = case_id_of(src)
        split_name = target_split.get(case_id)
        if split_name is None:
            print(f"    [SKIP] {src.name}: case not in manifest")
            continue
        dst = dataset_dir / split_name / "reports" / src.name
        placed[split_name].append(src.name)
        transfers.append({"case_id": case_id, "split": split_name,
                          "src": str(src), "dst": str(dst)})
        if dry_run:
            print(f"    [DRY-RUN] {args.mode}: {src.name} -> {split_name}/reports/")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "move":
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

    print("\n[INFO] ========== Summary ==========")
    for split_name in ("training", "validate"):
        n_cases = len({case_id_of(Path(n)) for n in placed[split_name]})
        print(f"[INFO] {split_name}: {len(placed[split_name])} report file(s) "
              f"placed across {n_cases} case(s)")
    if unknown:
        print(f"[WARN] {len(unknown)} case(s) not in the manifest "
              f"-> {args.unknown_split}: {sorted(unknown)}")
    for split_name, names in orphaned.items():
        verb = "deleted" if args.replace else "LEFT IN PLACE (re-run with --replace)"
        print(f"[WARN] {split_name}: {len(names)} superseded report file(s) "
              f"whose case is in the pool under other name(s), {verb}: "
              f"{names[:10]}{' ...' if len(names) > 10 else ''}")
    for split_name, names in stale.items():
        print(f"[WARN] {split_name}: {len(names)} report file(s) whose case has no "
              f"counterpart in the pool, left untouched: {names}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "reports_dir": str(reports_dir),
        "manifest": str(manifest_path),
        "mode": args.mode,
        "unknown_split": args.unknown_split,
        "replace": args.replace,
        "dry_run": dry_run,
        "counts": {s: len(placed[s]) for s in ("training", "validate")},
        "case_counts": {s: len({case_id_of(Path(n)) for n in placed[s]})
                        for s in ("training", "validate")},
        "unknown_cases": sorted(unknown),
        "orphaned_in_split": dict(orphaned),
        "stale_in_split": dict(stale),
        "transfers": transfers,
    }
    report_path = dataset_dir / "reports_split_manifest.json"
    if dry_run:
        print(f"\n[DRY-RUN] Report would be written to: {report_path}")
        print(f"[INFO] Re-run with --execute to perform the {args.mode}.")
    else:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[INFO] Report written: {report_path}")


if __name__ == "__main__":
    main()
