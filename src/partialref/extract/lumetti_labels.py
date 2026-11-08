"""Read Lumetti et al.'s released per-patient labels (Turtle) into our ontology (internal use only).

Each file `ToothFairy3<C>_<id>.ttl` holds the findings consolidated from all reports of one patient
(clinician-validated). We keep type, slot (tooth ids / side / quadrant) and qualifiers. The
`rdfs:comment` (a quote from the report) is **never read into the output**.

Usage:
    python -m partialref.extract.lumetti_labels --ttl-dir ~/Datasets/CBCT-Report/data/ttl_reports
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import rdflib

from partialref.extract.ontology import LUMETTI_PROPERTY, FindingType, slot_kind
from partialref.provenance import make_run_id, write_provenance

NS = rdflib.Namespace("http://example.org/dentro#")
NAME_RE = re.compile(r"^ToothFairy3([A-Z])_(\d+)\.ttl$")


def case_id_from_file(name: str) -> str | None:
    m = NAME_RE.match(name)
    return f"{m.group(1)}{m.group(2)}" if m else None   # keeps TF3 padding = TF4 padding


def _local(term) -> str:
    return str(term).split("#")[-1]


def parse_file(path: Path) -> list[dict]:
    g = rdflib.Graph()
    g.parse(path, format="turtle")
    cid = case_id_from_file(path.name)
    rows = []
    for subj in set(g.subjects(NS.findingType, None)):
        ftype = _local(g.value(subj, NS.findingType))
        if ftype not in FindingType.__members__:
            rows.append({"case_id": cid, "finding_id": _local(subj), "finding_type": ftype,
                         "unknown_type": True})
            continue
        teeth = sorted({str(o) for o in g.objects(subj, NS.hasToothId)})
        side = g.value(subj, NS.onSide)
        quad = g.value(subj, NS.hasQuadrant)
        quals = {}
        for prop, ours in LUMETTI_PROPERTY.items():
            vals = sorted({_local(o) for o in g.objects(subj, NS[prop])})
            if vals:
                quals[ours] = vals
        kind = slot_kind(FindingType(ftype))
        slots = teeth if kind == "tooth" else ([_local(side)] if side is not None and kind == "side"
                                               else [_local(quad)] if quad is not None else [])
        rows.append({"case_id": cid, "finding_id": _local(subj), "finding_type": ftype,
                     "slot_kind": kind, "slots": slots, "tooth_ids": teeth,
                     "side": _local(side) if side is not None else None,
                     "quadrant": _local(quad) if quad is not None else None,
                     "qualifiers": json.dumps(quals, sort_keys=True), "unknown_type": False})
    return rows


def load_dir(ttl_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rows, errors = [], []
    for p in sorted(ttl_dir.glob("*.ttl")):
        try:
            rows += parse_file(p)
        except Exception as e:  # report, keep going
            errors.append({"file": p.name, "error": type(e).__name__})
    findings = pd.DataFrame(rows)
    inst = (findings[~findings["unknown_type"]].explode("slots").dropna(subset=["slots"])
            [["case_id", "finding_type", "slots"]].rename(columns={"slots": "slot"})
            .drop_duplicates().reset_index(drop=True))
    return findings, inst, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttl-dir", required=True, type=Path)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--tag", default="lumetti-labels")
    a = ap.parse_args(argv)
    findings, inst, errors = load_dir(a.ttl_dir)
    run_id = make_run_id(a.tag)
    out = a.results_dir / "lumetti" / run_id
    out.mkdir(parents=True, exist_ok=False)
    findings.to_parquet(out / "findings.parquet", index=False)
    inst.to_parquet(out / "instances.parquet", index=False)
    try:
        src = subprocess.run(["git", "-C", str(a.ttl_dir), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except Exception:
        src = None
    summary = {"run_id": run_id, "n_files": int(findings["case_id"].nunique()),
               "n_findings": int(len(findings)), "n_instances": int(len(inst)),
               "unknown_types": int(findings["unknown_type"].sum()),
               "parse_errors": errors,
               "findings_per_type": findings["finding_type"].value_counts().to_dict(),
               "source_commit": src}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_provenance(out, extra={"run_id": run_id, "ttl_dir": str(a.ttl_dir), "source_commit": src},
                     packages=["pandas", "rdflib"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
