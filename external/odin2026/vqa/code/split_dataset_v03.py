#!/usr/bin/env python3
"""
code/split_dataset_v03.py

Split the nested toothfairy4_v03 release into the flat training/validate
layout the rest of the pipeline expects:

    toothfairy4_v03/{case_id}/cbct/volume.nii.gz
    toothfairy4_v03/{case_id}/reports_en/*.txt        (1..4 per case)
        |
        v
    dataset/{training,validate}/images/{case_id}_0000.nii.gz
    dataset/{training,validate}/reports/{case_id}_{n}.txt

Report files are renumbered to the `{case_id}_{radiologist}.txt`
convention parse_reports_to_gt.py already parses -- the original release
filename (cbct_34_3872.txt) is preserved in the manifest so provenance
isn't lost. reports_it/ is left alone; only reports_en/ is split.

SPLIT REUSE: by default the partition is read from an existing
--manifest (dataset/split_manifest.json) rather than re-drawn, so the
validate set stays identical across dataset releases and previously
reported validate metrics remain comparable. Cases present in the new
release but absent from that manifest go to --unknown-split (training,
matching how split_reports.py/split_facts.py handled the same 5 cases).
Without a manifest, falls back to a seeded per-prefix random draw.

Volumes present in the destination but no longer in the release (cases
dropped between versions) are NOT deleted -- they're relocated to
--stale-dir so nothing is lost irreversibly.

SAFE BY DEFAULT: dry-run unless --execute is passed.

Usage:
    # Preview
    python3 code/split_dataset_v03.py --project-dir $HOME/project_ToothFairy4

    # Perform it
    python3 code/split_dataset_v03.py --project-dir $HOME/project_ToothFairy4 \
        --mode move --execute
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
# Release report names look like cbct_34_3872.txt -- sort on the embedded
# numbers so the _1/_2 suffixes we assign are stable across re-runs and
# don't depend on zero-padding.
NUM_RE = re.compile(r"\d+")


def discover_cases(source_dir: Path) -> List[str]:
    """Case dirs holding a cbct/volume.nii.gz."""
    cases = []
    for case_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        if (case_dir / "cbct" / "volume.nii.gz").exists():
            cases.append(case_dir.name)
        else:
            print(f"[WARN] {case_dir.name}: no cbct/volume.nii.gz, skipped",
                  file=sys.stderr)
    return cases


def prefix_of(case_id: str) -> Optional[str]:
    m = CASE_ID_RE.match(case_id)
    return m.group(1) if m else None


def report_sort_key(p: Path):
    return ([int(n) for n in NUM_RE.findall(p.stem)], p.name)


def partition_from_manifest(manifest_path: Path, cases: List[str],
                            unknown_split: str):
    """Reuse a prior split. Returns (partition, new_cases, dropped_cases)."""
    with open(manifest_path) as f:
        prior = json.load(f)["partition"]

    split_of = {}
    for split_name in ("training", "validate"):
        for ids in prior[split_name].values():
            for case_id in ids:
                split_of[case_id] = split_name

    partition = {"training": defaultdict(list), "validate": defaultdict(list)}
    new_cases = []
    case_set = set(cases)
    for case_id in cases:
        split_name = split_of.get(case_id)
        if split_name is None:
            split_name = unknown_split
            new_cases.append(case_id)
        partition[split_name][prefix_of(case_id) or "?"].append(case_id)

    dropped = sorted(c for c in split_of if c not in case_set)
    partition = {k: {p: sorted(v) for p, v in sorted(d.items())}
                 for k, d in partition.items()}
    return partition, new_cases, dropped


def partition_random(cases: List[str], n_validate: int, seed: int):
    """Fallback when no prior manifest is available."""
    rng = random.Random(seed)
    groups: Dict[str, List[str]] = defaultdict(list)
    for case_id in cases:
        groups[prefix_of(case_id) or "?"].append(case_id)

    partition = {"training": {}, "validate": {}}
    for prefix, ids in sorted(groups.items()):
        ids = sorted(ids)
        if len(ids) <= n_validate:
            print(f"[WARN] prefix '{prefix}' has only {len(ids)} case(s) "
                  f"<= {n_validate} -- all go to validate", file=sys.stderr)
            partition["validate"][prefix] = ids
            partition["training"][prefix] = []
            continue
        val = sorted(rng.sample(ids, n_validate))
        partition["validate"][prefix] = val
        partition["training"][prefix] = [c for c in ids if c not in set(val)]
    return partition, [], []


def transfer(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))       # overwrites an existing dst
    elif mode == "copy":
        shutil.copy2(str(src), str(dst))
    elif mode == "symlink":
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())


def main():
    ap = argparse.ArgumentParser(
        description="Split the nested toothfairy4_v03 release into "
                    "dataset/{training,validate}/{images,reports}.")
    ap.add_argument("--project-dir", required=True, help="Project root")
    ap.add_argument("--source-dir", default=None,
                    help="Release root (default: <project-dir>/toothfairy4_v03)")
    ap.add_argument("--dest-dir", default=None,
                    help="Split root (default: <project-dir>/dataset)")
    ap.add_argument("--manifest", default=None,
                    help="Prior split_manifest.json to reuse the partition from "
                         "(default: <dest-dir>/split_manifest.json). Pass "
                         "--no-manifest to draw a fresh random split instead.")
    ap.add_argument("--no-manifest", action="store_true",
                    help="Ignore any prior manifest; draw a fresh seeded split.")
    ap.add_argument("--unknown-split", choices=["training", "validate"],
                    default="training",
                    help="Where cases absent from the prior manifest go "
                         "(default: training)")
    ap.add_argument("--n-validate-per-dataset", type=int, default=10,
                    help="Fresh-split only: validate cases per prefix (default: 10)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Fresh-split only: RNG seed (default: 42)")
    ap.add_argument("--mode", choices=["move", "copy", "symlink"], default="move",
                    help="How to transfer VOLUMES (default: move). Reports are "
                         "always copied -- they're tiny and reports_it/ stays "
                         "next to them in the release tree.")
    ap.add_argument("--stale-dir", default=None,
                    help="Where volumes for cases dropped from the release are "
                         "relocated (default: <dest-dir>/_stale_pre_v03)")
    ap.add_argument("--manifest-out", default=None,
                    help="Audit manifest path (default: <dest-dir>/split_manifest_v03.json)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually perform the file operations (default: dry-run)")
    args = ap.parse_args()

    project_dir = Path(args.project_dir)
    source_dir = Path(args.source_dir) if args.source_dir else project_dir / "toothfairy4_v03"
    dest_dir = Path(args.dest_dir) if args.dest_dir else project_dir / "dataset"
    stale_dir = Path(args.stale_dir) if args.stale_dir else dest_dir / "_stale_pre_v03"
    manifest_out = (Path(args.manifest_out) if args.manifest_out
                    else dest_dir / "split_manifest_v03.json")
    manifest_in = (Path(args.manifest) if args.manifest
                   else dest_dir / "split_manifest.json")
    dry_run = not args.execute

    if not source_dir.is_dir():
        print(f"[FAIL] Source dir not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] {'DRY-RUN -- nothing will be touched' if dry_run else 'EXECUTING'}")
    print(f"[INFO] Source: {source_dir}")
    print(f"[INFO] Dest:   {dest_dir}/{{training,validate}}/{{images,reports}}")
    print(f"[INFO] Volume mode: {args.mode}   Reports: copy")

    cases = discover_cases(source_dir)
    print(f"[INFO] Discovered {len(cases)} case(s) in the release")

    if args.no_manifest or not manifest_in.exists():
        if not args.no_manifest:
            print(f"[WARN] No prior manifest at {manifest_in} -- drawing a fresh "
                  f"seeded split", file=sys.stderr)
        partition, new_cases, dropped = partition_random(
            cases, args.n_validate_per_dataset, args.seed)
        split_source = "fresh-random"
    else:
        partition, new_cases, dropped = partition_from_manifest(
            manifest_in, cases, args.unknown_split)
        split_source = str(manifest_in)
        print(f"[INFO] Partition reused from {manifest_in}")
        if new_cases:
            print(f"[INFO] {len(new_cases)} case(s) new in v03 -> "
                  f"{args.unknown_split}: {new_cases}")
        if dropped:
            print(f"[INFO] {len(dropped)} case(s) in the prior split are gone "
                  f"from v03: {dropped}")

    for split_name in ("training", "validate"):
        total = sum(len(v) for v in partition[split_name].values())
        by_prefix = ", ".join(f"{p} ({len(v)})"
                              for p, v in sorted(partition[split_name].items()))
        print(f"[INFO] {split_name}: {total} case(s)  [{by_prefix}]")

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "project_dir": str(project_dir),
        "source_dir": str(source_dir),
        "dest_dir": str(dest_dir),
        "release": source_dir.name,
        "split_source": split_source,
        "unknown_split": args.unknown_split,
        "volume_mode": args.mode,
        "report_mode": "copy",
        "dry_run": dry_run,
        "partition": partition,
        "new_cases": new_cases,
        "dropped_cases": dropped,
        "counts": {},
        "no_reports": [],
        "report_map": {},     # case_id -> {new_name: original release name}
        "relocated_stale": [],
        "errors": [],
    }

    n_vol = n_rep = 0
    for split_name in ("training", "validate"):
        split_dir = dest_dir / split_name
        for prefix in sorted(partition[split_name]):
            for case_id in partition[split_name][prefix]:
                case_dir = source_dir / case_id

                src_vol = case_dir / "cbct" / "volume.nii.gz"
                dst_vol = split_dir / "images" / f"{case_id}_0000.nii.gz"
                if src_vol.exists():
                    transfer(src_vol, dst_vol, args.mode, dry_run)
                    n_vol += 1
                elif dst_vol.exists() and args.mode == "move":
                    # Already moved by an earlier run -- idempotent re-run.
                    n_vol += 1
                else:
                    manifest["errors"].append(
                        {"case_id": case_id, "error": "missing volume",
                         "path": str(src_vol)})

                reports = sorted((case_dir / "reports_en").glob("*.txt"),
                                 key=report_sort_key)
                if not reports:
                    manifest["no_reports"].append(case_id)
                    continue
                case_map = {}
                for i, src_rep in enumerate(reports, start=1):
                    dst_rep = split_dir / "reports" / f"{case_id}_{i}.txt"
                    transfer(src_rep, dst_rep, "copy", dry_run)
                    case_map[dst_rep.name] = src_rep.name
                    n_rep += 1
                manifest["report_map"][case_id] = case_map

    # Cases that were in the prior split but are gone from v03: get their
    # now-orphaned volumes out of images/ without destroying them.
    for case_id in dropped:
        for split_name in ("training", "validate"):
            old = dest_dir / split_name / "images" / f"{case_id}_0000.nii.gz"
            if old.exists() or old.is_symlink():
                dst = stale_dir / split_name / "images" / old.name
                transfer(old, dst, "move", dry_run)
                manifest["relocated_stale"].append(
                    {"case_id": case_id, "src": str(old), "dst": str(dst)})

    manifest["counts"] = {
        "cases": len(cases),
        "training": sum(len(v) for v in partition["training"].values()),
        "validate": sum(len(v) for v in partition["validate"].values()),
        "volumes": n_vol,
        "reports": n_rep,
    }

    if dry_run:
        print(f"\n[DRY-RUN] Would write manifest: {manifest_out}")
    else:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_out, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n[INFO] Manifest written: {manifest_out}")

    print("\n[INFO] ========== Summary ==========")
    print(f"[INFO] cases={len(cases)}  volumes={n_vol}  reports={n_rep}")
    print(f"[INFO] training={manifest['counts']['training']}  "
          f"validate={manifest['counts']['validate']}")
    if manifest["no_reports"]:
        print(f"[WARN] {len(manifest['no_reports'])} case(s) with no reports_en: "
              f"{manifest['no_reports'][:10]}")
    if manifest["relocated_stale"]:
        print(f"[INFO] {len(manifest['relocated_stale'])} stale volume(s) "
              f"relocated to {stale_dir}")
    if manifest["errors"]:
        print(f"[FAIL] {len(manifest['errors'])} error(s): {manifest['errors'][:5]}")
    if dry_run:
        print(f"\n[INFO] DRY-RUN. Re-run with --execute to perform the {args.mode}.")


if __name__ == "__main__":
    main()
