"""Export a self-contained review bundle for Shivam's own laptop (approved by him 2026-11-09, DUA note in
DECISIONS): report texts (EN + IT), the parse run's per-report findings, and a report table with relative
paths. The bundle stays in results/review/bundles/ (git-ignored, not rsynced), file mode 600; copy it to the
laptop with scp, never via cloud services. Review logs coming back are validated by partialref.review.import_log.

    python -m partialref.review.export_bundle --parse-run results/extract/<run> \
        --manifest results/manifest/<run> [--langs en,it] [--review-list results/review/lists/<list>.json]

With --review-list (partialref.review.sample) the bundle holds ONLY the listed reports and their parse files,
plus the list itself; the local app then steps through that list.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import pandas as pd

from partialref.extract.score_parser import parser_half
from partialref.provenance import git_state
from partialref.review.data import load_review_list


def build_bundle(parse_run: Path, manifest: Path, langs: list[str], out_dir: Path,
                 review_list: Path | None = None) -> Path:
    reps = pd.read_parquet(manifest / "reports.parquet")
    reps = reps[reps["lang"].isin(langs)].copy()
    name = f"review-bundle-{parse_run.name}"
    listed = None
    if review_list is not None:
        listed = {(e["case_id"], e["report_id"]) for e in load_review_list(review_list, reps)}
        reps = reps[[k in listed for k in zip(reps["case_id"], reps["report_id"])]].copy()
        # one name per list (full list vs scoped subset), so bundles never overwrite or unpack into each other
        prefix = f"review-list-{parse_run.name}"
        stem = review_list.stem
        name += "-list" + (stem[len(prefix):] if stem.startswith(prefix) else f"-{stem}")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / name
        rows = []
        for r in reps.itertuples():
            rel = Path("reports") / r.case_id / f"reports_{r.lang}" / Path(r.path).name
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(r.path, root / rel)
            rows.append({"case_id": r.case_id, "lang": r.lang, "report_id": r.report_id, "path": str(rel)})
        pd.DataFrame(rows).to_parquet(root / "reports.parquet", index=False)
        dst = root / "parse" / parse_run.name / "per_report"
        dst.mkdir(parents=True)
        n_parse = 0
        for f in parse_run.glob("per_report/*.json"):
            if f.stem.split("__")[-1] in langs and (listed is None or tuple(f.stem.split("__")[:2]) in listed):
                shutil.copyfile(f, dst / f.name)
                n_parse += 1
        if review_list is not None:
            shutil.copyfile(review_list, root / "review_list.json")
        g = git_state()
        info = {"parse_run": parse_run.name, "langs": langs, "n_reports": len(rows), "n_parse_files": n_parse,
                "n_cases": int(reps["case_id"].nunique()), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "git": (g["git_sha"] or "")[:7] or None, "git_dirty": g["git_dirty"],
                "review_list": review_list.name if review_list is not None else None,
                "halves": reps["case_id"].drop_duplicates().map(parser_half).value_counts().to_dict()}
        (root / "bundle_info.json").write_text(json.dumps(info, indent=2))
        (root / "README.txt").write_text(
            "ToothFairy4 report review bundle (contains report text: keep on your own machine only, no cloud sync).\n"
            "Run:   python -m partialref.review.local <this folder>\n"
            "Logs:  <this folder>/logs/<parse run>/review_log.jsonl  -> scp it back to the cluster into\n"
            "       results/review/incoming/ (then: python -m partialref.review.import_log <file>)\n")
        out_dir.mkdir(parents=True, exist_ok=True)
        tgz = out_dir / f"{name}.tar.gz"
        old = os.umask(0o077)
        try:
            with tarfile.open(tgz, "w:gz") as t:
                t.add(root, arcname=name)
        finally:
            os.umask(old)
        os.chmod(tgz, 0o600)
    return tgz


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse-run", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--langs", default="en,it")
    ap.add_argument("--out-dir", default=Path("results/review/bundles"), type=Path)
    ap.add_argument("--review-list", default=None, type=Path, help="bundle only the reports of this list")
    a = ap.parse_args(argv)
    tgz = build_bundle(a.parse_run, a.manifest, [x for x in a.langs.split(",") if x], a.out_dir, a.review_list)
    print(f"bundle: {tgz} ({tgz.stat().st_size / 1e6:.1f} MB, mode 600)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
