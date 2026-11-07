#!/usr/bin/env python3
"""
code/split_facts.py

Distribute a flat pool of per-case files (one file per case, named
<case_id>.<ext>) into the existing training/validate splits:

    dataset/training/<dest-subdir>/<case_id>.<ext>
    dataset/validate/<dest-subdir>/<case_id>.<ext>

Defaults to the fact pool (dataset/facts/*.json -> <split>/facts/), but
--pool-dir/--pattern/--dest-subdir make it work for any single-file-per-case
artifact. Notably the segmentation masks, whose pool is named `predictions`
while the pipeline reads them from `masks`:

    --pool-dir dataset/predictions --pattern '*.nii.gz' --dest-subdir masks

Unlike code/split_dataset.py, this does NOT re-draw the partition. It reads the
already-recorded partition from a split manifest, so the validate set stays
exactly the one every existing output was produced against. (Re-running the
seeded draw would silently reshuffle the split whenever cases are added/removed
from the pool.)

Cases present in the pool but absent from the manifest are assigned to
`--unknown-split` (default: training), which keeps validate pinned at the
manifest's N-per-prefix.

Existing files whose case has no file in the pool are left untouched and
reported as `stale`.

SAFE BY DEFAULT: dry-run unless --execute is passed.

Usage:
    python3 code/split_facts.py --project-dir $HOME/project_ToothFairy4
    python3 code/split_facts.py --project-dir $HOME/project_ToothFairy4 --execute

    # masks, from the differently-named `predictions` pool
    python3 code/split_facts.py --project-dir $HOME/project_ToothFairy4 \
        --pool-dir $HOME/project_ToothFairy4/dataset/predictions \
        --pattern '*.nii.gz' --dest-subdir masks --execute
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


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


def case_id_of(path: Path, suffix: str) -> str:
    """Strip the artifact's full suffix -- Path.stem gets '.nii.gz' wrong."""
    name = path.name
    return name[: -len(suffix)] if suffix and name.endswith(suffix) else path.stem


def main():
    ap = argparse.ArgumentParser(
        description="Distribute a flat per-case artifact pool into the "
                    "training/validate splits recorded in a split manifest.")
    ap.add_argument("--project-dir", required=True, help="Project root")
    ap.add_argument("--facts-dir", "--pool-dir", dest="facts_dir", default=None,
                    help="Flat per-case pool (default: <project-dir>/dataset/facts)")
    ap.add_argument("--pattern", default="*.json",
                    help="Glob for pool files, one per case (default: *.json). "
                         "Use '*.nii.gz' for masks.")
    ap.add_argument("--dest-subdir", default="facts",
                    help="Subdir under <split>/ to fill (default: facts). Use "
                         "'masks' when splitting the predictions pool.")
    ap.add_argument("--report-path", default=None,
                    help="Audit report path (default: "
                         "<dataset>/<dest-subdir>_split_manifest.json)")
    ap.add_argument("--manifest", default=None,
                    help="Split manifest (default: <project-dir>/dataset/split_manifest.json)")
    ap.add_argument("--mode", choices=["copy", "move"], default="copy",
                    help="copy keeps the flat pool intact as the master set (default: copy)")
    ap.add_argument("--unknown-split", choices=["training", "validate", "skip"],
                    default="training",
                    help="Where to put cases not listed in the manifest (default: training)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually perform the file operations (default: dry-run)")
    args = ap.parse_args()

    project_dir = Path(args.project_dir)
    dataset_dir = project_dir / "dataset"
    facts_dir = Path(args.facts_dir) if args.facts_dir else dataset_dir / "facts"
    manifest_path = Path(args.manifest) if args.manifest else dataset_dir / "split_manifest.json"
    dest_subdir = args.dest_subdir
    suffix = args.pattern[1:] if args.pattern.startswith("*") else ""
    dry_run = not args.execute

    if not facts_dir.is_dir():
        print(f"[FAIL] Pool dir not found: {facts_dir}", file=sys.stderr)
        sys.exit(1)
    if not manifest_path.is_file():
        print(f"[FAIL] Split manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    assignment = load_partition(manifest_path)
    pool = sorted(facts_dir.glob(args.pattern))

    print(f"[INFO] {'DRY-RUN -- no files will be touched' if dry_run else 'EXECUTING'}")
    print(f"[INFO] Pool: {facts_dir}/{args.pattern} ({len(pool)} file(s))")
    print(f"[INFO] Dest: <split>/{dest_subdir}/")
    print(f"[INFO] Partition: {manifest_path} ({len(assignment)} case(s))")
    print(f"[INFO] Mode: {args.mode}  Unknown cases -> {args.unknown_split}")
    print()

    placed = defaultdict(list)
    unknown = []
    transfers = []

    for src in pool:
        case_id = case_id_of(src, suffix)
        split_name = assignment.get(case_id)
        if split_name is None:
            unknown.append(case_id)
            if args.unknown_split == "skip":
                print(f"    [SKIP] {case_id}: not in manifest")
                continue
            split_name = args.unknown_split
        dst = dataset_dir / split_name / dest_subdir / src.name
        placed[split_name].append(case_id)
        transfers.append({"case_id": case_id, "split": split_name,
                          "src": str(src), "dst": str(dst)})
        if dry_run:
            print(f"    [DRY-RUN] {args.mode}: {case_id} -> {split_name}/{dest_subdir}/")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "move":
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

    # Files already in a split whose case has no file in the pool.
    pool_ids = {case_id_of(p, suffix) for p in pool}
    stale = defaultdict(list)
    for split_name in ("training", "validate"):
        split_dest = dataset_dir / split_name / dest_subdir
        if not split_dest.is_dir():
            continue
        for existing in sorted(split_dest.glob(args.pattern)):
            case_id = case_id_of(existing, suffix)
            if case_id not in pool_ids:
                stale[split_name].append(case_id)

    print("\n[INFO] ========== Summary ==========")
    for split_name in ("training", "validate"):
        print(f"[INFO] {split_name}: {len(placed[split_name])} "
              f"{dest_subdir} file(s) placed")
    if unknown:
        print(f"[WARN] {len(unknown)} case(s) not in the manifest "
              f"-> {args.unknown_split}: {unknown}")
    for split_name, ids in stale.items():
        print(f"[WARN] {split_name}: {len(ids)} pre-existing {dest_subdir} file(s) "
              f"with no counterpart in the pool, left untouched: {ids}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "facts_dir": str(facts_dir),
        "pattern": args.pattern,
        "dest_subdir": dest_subdir,
        "manifest": str(manifest_path),
        "mode": args.mode,
        "unknown_split": args.unknown_split,
        "dry_run": dry_run,
        "counts": {s: len(placed[s]) for s in ("training", "validate")},
        "unknown_cases": unknown,
        "stale_in_split": dict(stale),
        "transfers": transfers,
    }
    report_path = (Path(args.report_path) if args.report_path
                   else dataset_dir / f"{dest_subdir}_split_manifest.json")
    if dry_run:
        print(f"\n[DRY-RUN] Report would be written to: {report_path}")
        print(f"[INFO] Re-run with --execute to perform the {args.mode}.")
    else:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[INFO] Report written: {report_path}")


if __name__ == "__main__":
    main()
