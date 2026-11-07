#!/usr/bin/env python3
"""
code/split_dataset.py

Partition the full dataset (images + masks + reports + facts, grouped by
case ID) into training/ and validate/ splits, each holding all four
per-case artifact types:

    dataset/training/{images,masks,reports,facts}
    dataset/validate/{images,masks,reports,facts}

Cases are grouped by the case ID's leading alphabetic prefix (A004 -> "A",
S0030 -> "S", handling differing digit-widths across sub-datasets).
`--n-validate-per-dataset` cases are randomly selected (seeded, so
reproducible) per prefix for `validate`; everything else goes to
`training`. This is an exhaustive partition -- every case ends up in
exactly one split, nothing is left in an unsplit pool.

SAFE BY DEFAULT: runs as a dry-run (prints what WOULD happen, moves
nothing) unless --execute is passed. Always writes an audit manifest
(dataset/split_manifest.json) recording exactly what was selected and
where it went.

`facts` and `masks` source directories are optional -- if a case's
facts/mask file isn't found there, that artifact is just skipped for that
case (with a warning), since not every dataset may have all four artifact
types available yet.

Usage:
    # Preview only -- no files touched
    python3 code/split_dataset.py \
        --project-dir $HOME/project_ToothFairy4 \
        --n-validate-per-dataset 10

    # Actually perform the move
    python3 code/split_dataset.py \
        --project-dir $HOME/project_ToothFairy4 \
        --n-validate-per-dataset 10 \
        --execute

    # Copy instead of move (keeps originals in place too)
    python code/split_dataset.py --project-dir ... --mode copy --execute
"""

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


CASE_ID_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def discover_case_ids(images_dir: Path) -> List[str]:
    case_ids = []
    for volume_file in sorted(images_dir.glob("*_0000.nii.gz")):
        case_id = volume_file.name[: -len("_0000.nii.gz")]
        case_ids.append(case_id)
    return case_ids


def group_by_dataset(case_ids: List[str]) -> Dict[str, List[str]]:
    """Group case IDs by their leading alphabetic prefix (A004 -> 'A')."""
    groups: Dict[str, List[str]] = defaultdict(list)
    unmatched = []
    for case_id in case_ids:
        m = CASE_ID_RE.match(case_id)
        if m:
            groups[m.group(1)].append(case_id)
        else:
            unmatched.append(case_id)
    if unmatched:
        print(f"[WARN] {len(unmatched)} case ID(s) didn't match the expected "
              f"<letters><digits> pattern, skipped: {unmatched[:10]}"
              f"{' ...' if len(unmatched) > 10 else ''}", file=sys.stderr)
    return dict(groups)


def partition_cases(groups: Dict[str, List[str]], n_validate: int,
                     seed: int) -> Dict[str, Dict[str, List[str]]]:
    """Returns {"validate": {prefix: [...]}, "training": {prefix: [...]}}."""
    rng = random.Random(seed)
    validate: Dict[str, List[str]] = {}
    training: Dict[str, List[str]] = {}

    for prefix, case_ids in sorted(groups.items()):
        case_ids = sorted(case_ids)
        if len(case_ids) <= n_validate:
            print(f"[WARN] Dataset '{prefix}' has only {len(case_ids)} case(s), "
                  f"<= requested {n_validate} for validate -- ALL go to validate, "
                  f"none remain for training.", file=sys.stderr)
            validate[prefix] = case_ids
            training[prefix] = []
            continue
        validate_ids = sorted(rng.sample(case_ids, n_validate))
        validate_set = set(validate_ids)
        training_ids = sorted([c for c in case_ids if c not in validate_set])
        validate[prefix] = validate_ids
        training[prefix] = training_ids

    return {"validate": validate, "training": training}


def find_case_files(source_dir: Optional[Path], case_id: str,
                     exact_suffixes: List[str]) -> List[Path]:
    """
    Find file(s) belonging to a case in source_dir. Tries exact filenames
    first (case_id + each suffix, e.g. '.nii.gz'), then falls back to any
    file starting with '{case_id}_' or exactly '{case_id}.*' (for reports/
    facts, which may have multiple numbered files per case, or an unknown
    extension for facts).
    """
    if source_dir is None or not source_dir.is_dir():
        return []

    found = []
    for suffix in exact_suffixes:
        candidate = source_dir / f"{case_id}{suffix}"
        if candidate.exists():
            found.append(candidate)
    if found:
        return found

    # Fallback: any file(s) prefixed with this case_id
    found = sorted(source_dir.glob(f"{case_id}_*")) + sorted(source_dir.glob(f"{case_id}.*"))
    return found


def transfer_file(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"    [DRY-RUN] {mode}: {src} -> {dst}")
        return
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "copy":
        shutil.copy2(str(src), str(dst))
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    print(f"    [{mode.upper()}] {src.name} -> {dst}")


def main():
    ap = argparse.ArgumentParser(
        description="Partition the dataset into training/ and validate/ splits "
                    "(images + masks + reports + facts per case).")
    ap.add_argument("--project-dir", required=True,
                     help="Project root")
    ap.add_argument("--images-dir", default=None,
                     help="Source CBCT images dir (default: <project-dir>/dataset/images)")
    ap.add_argument("--masks-dir", default=None,
                     help="Source masks dir (default: <project-dir>/dataset/predictions, "
                          "optional -- skipped per-case if not found)")
    ap.add_argument("--reports-dir", default=None,
                     help="Source reports dir (default: <project-dir>/dataset/reports, "
                          "optional -- skipped per-case if not found)")
    ap.add_argument("--facts-dir", default=None,
                     help="Source facts dir (default: <project-dir>/dataset/facts, "
                          "optional -- skipped per-case if not found; format TBD)")
    ap.add_argument("--n-validate-per-dataset", type=int, default=10,
                     help="Number of cases per sub-dataset to assign to 'validate' "
                          "(default: 10). Everything else goes to 'training'.")
    ap.add_argument("--seed", type=int, default=42,
                     help="Random seed for reproducible selection (default: 42)")
    ap.add_argument("--mode", choices=["move", "copy", "symlink"], default="move",
                     help="How to transfer files (default: move). Use 'copy' to keep "
                          "the original flat dataset/ layout intact too.")
    ap.add_argument("--execute", action="store_true",
                     help="Actually perform the file operations. Without this flag, runs "
                          "as a dry-run: prints what would happen, touches nothing.")

    args = ap.parse_args()

    project_dir = Path(args.project_dir)
    dataset_dir = project_dir / "dataset"
    images_dir = Path(args.images_dir) if args.images_dir else dataset_dir / "images"
    masks_dir = Path(args.masks_dir) if args.masks_dir else dataset_dir / "masks"
    reports_dir = Path(args.reports_dir) if args.reports_dir else dataset_dir / "reports"
    facts_dir = Path(args.facts_dir) if args.facts_dir else dataset_dir / "measurements"

    dry_run = not args.execute

    if not images_dir.is_dir():
        print(f"[FAIL] Images dir not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    for name, d in [("masks", masks_dir), ("reports", reports_dir), ("facts", facts_dir)]:
        if not d.is_dir():
            print(f"[WARN] {name} dir not found: {d} -- {name} will be skipped for all cases",
                  file=sys.stderr)

    print(f"[INFO] {'DRY-RUN -- no files will be touched' if dry_run else 'EXECUTING'}")
    print(f"[INFO] Project dir: {project_dir}")
    print(f"[INFO] Sources: images={images_dir}, masks={masks_dir}, "
          f"reports={reports_dir}, facts={facts_dir}")
    print(f"[INFO] Mode: {args.mode}  Seed: {args.seed}")
    print()

    case_ids = discover_case_ids(images_dir)
    print(f"[INFO] Discovered {len(case_ids)} total case(s) in {images_dir}")

    groups = group_by_dataset(case_ids)
    print(f"[INFO] Found {len(groups)} sub-dataset(s): "
          + ", ".join(f"{k} ({len(v)})" for k, v in sorted(groups.items())))
    print()

    partition = partition_cases(groups, args.n_validate_per_dataset, args.seed)

    for split_name in ("training", "validate"):
        total = sum(len(v) for v in partition[split_name].values())
        print(f"[INFO] {split_name}: {total} case(s) total")
        for prefix, ids in partition[split_name].items():
            print(f"    {prefix}: {len(ids)} case(s)")
    print()

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "project_dir": str(project_dir),
        "sources": {"images": str(images_dir), "masks": str(masks_dir),
                    "reports": str(reports_dir), "facts": str(facts_dir)},
        "mode": args.mode,
        "seed": args.seed,
        "n_validate_per_dataset": args.n_validate_per_dataset,
        "dry_run": dry_run,
        "partition": partition,
        "transfers": [],
        "missing": {"masks": [], "reports": [], "facts": []},
    }

    ARTIFACT_SPECS = [
        # (name, source_dir, dest_subdir, exact_suffixes)
        ("volume", images_dir, "images", ["_0000.nii.gz"]),
        ("mask", masks_dir, "masks", [".nii.gz"]),
        ("report", reports_dir, "reports", []),   # no single exact name -- may be numbered
        ("fact", facts_dir, "facts", []),         # format/naming unknown yet
    ]

    for split_name in ("training", "validate"):
        split_dir = dataset_dir / split_name
        for prefix, case_ids_split in partition[split_name].items():
            for case_id in case_ids_split:
                print(f"[{split_name}/{case_id}] ({prefix})")

                for artifact_name, source_dir, dest_subdir, exact_suffixes in ARTIFACT_SPECS:
                    files = find_case_files(source_dir, case_id, exact_suffixes)
                    if not files:
                        if artifact_name != "volume":
                            print(f"    [WARN] No {artifact_name} file found for {case_id}",
                                  file=sys.stderr)
                            manifest["missing"][dest_subdir].append(case_id)
                        continue
                    for src in files:
                        dst = split_dir / dest_subdir / src.name
                        transfer_file(src, dst, args.mode, dry_run)
                        manifest["transfers"].append({
                            "case_id": case_id, "split": split_name, "type": artifact_name,
                            "src": str(src), "dst": str(dst),
                        })

    manifest_path = dataset_dir / "split_manifest.json"
    if dry_run:
        print(f"\n[DRY-RUN] Manifest would be written to: {manifest_path}")
    else:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n[INFO] Manifest written: {manifest_path}")

    print(f"\n[INFO] ========== Summary ==========")
    for split_name in ("training", "validate"):
        total = sum(len(v) for v in partition[split_name].values())
        print(f"[INFO] {split_name}: {total} case(s)")
    for artifact in ("masks", "reports", "facts"):
        n_missing = len(manifest["missing"][artifact])
        if n_missing:
            print(f"[WARN] {n_missing} case(s) missing {artifact}")
    if dry_run:
        print(f"\n[INFO] This was a DRY-RUN. Re-run with --execute to perform the {args.mode}.")


if __name__ == "__main__":
    main()