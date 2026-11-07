#!/usr/bin/env python3
"""
apply_triage_decisions.py -- write a reviewer's sheet decisions into stage 1.

build_triage_sheet.py deliberately writes nothing: a drop edits the ground
truth and should be applied by a reviewed command. This is that command.

WHAT EACH VERDICT MEANS depends on which screen raised the row, and getting
that wrong is how a decision becomes a different edit than the one intended:

  not-in-text / canal / sinus / intrasinusal
      drop      remove the FDI from that field. keep changes nothing -- the
                value is already there and the reviewer is affirming it.
  arch range
      keep      the stated range wins, so ADD the tooth to teeth_present (and
                take it out of teeth_absent, which is what it was conflicting
                with). This is the one "keep" that writes.
      drop      the extracted absence wins; the tooth is already out of
                teeth_present, so nothing to do.
  present/absent (a tooth in BOTH lists)
      present   drop it from teeth_absent.
      absent    drop it from teeth_present.
  unsure / unlocate
      never written. unsure records that nobody could tell, and unlocate needs
      a target list this file does not build.

Row ids are rebuilt from the audit rather than parsed, because the id has its
dots replaced by underscores and `sinus_left.intrasinusal_teeth` cannot be
recovered from `sinus_left_intrasinusal_teeth` by splitting.

    python3 code/apply_triage_decisions.py --split training \\
        --decisions decisions.json [--dry-run]

Every touched file keeps a .bak, and the stage-2 replay command is printed
with exactly the cases that changed.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from audit_report_facts import audit_case, report_text_for  # noqa: E402

SCREENS = {"not-in-text", "contradiction", "arch-range", "canal-adjacency",
           "intrasinusal"}


def build_index(split: str):
    """{row id: (file, arch, field, fdi)} -- the same ids the sheet renders."""
    gt = ROOT / f"dataset/{split}/outputs/ground_truth"
    reports = ROOT / f"dataset/{split}/reports"
    index = {}
    for path in sorted(gt.glob("*_report_facts.json")):
        stem = path.name.replace("_report_facts.json", "")
        case = re.match(r"([A-Z]+\d+)", stem).group(1)
        text = report_text_for(stem, case, reports)
        if not text:
            continue
        facts = json.loads(path.read_text())
        for level, screen, message in audit_case(case, facts, text, SCREENS,
                                                 stem=stem):
            if level != "ERROR":
                continue
            m = re.search(r"^(\w+)\.([\w.]+) asserts (\d+)", message)
            m2 = re.search(r"^(\w+): (\d+) is in BOTH", message)
            m3 = re.search(r"^(\w+)\.(canal_\w+|sinus_\w+) (?:claims contact with|puts) (\d+)",
                           message)
            m4 = re.search(r"^(\w+): \".+\" sweeps the arch and teeth_present is "
                           r"missing \[([\d, ]+)\]", message)
            rows = []
            if m:
                rows = [(m.group(1), m.group(2), int(m.group(3)))]
            elif m2:
                rows = [(m2.group(1), "present/absent", int(m2.group(2)))]
            elif m3:
                key = ("adjacent_teeth" if m3.group(2).startswith("canal")
                       else "intrasinusal_teeth")
                rows = [(m3.group(1), f"{m3.group(2)}.{key}", int(m3.group(3)))]
            elif m4:
                rows = [(m4.group(1), "arch range", int(x))
                        for x in m4.group(2).split(",")]
            for arch, field, fdi in rows:
                rid = f"{stem}-{arch}-{field}-{fdi}".replace(".", "_")
                index[rid] = (path.name, arch, field, fdi)
    return index


def edit(block: dict, field: str, fdi: int, verdict: str) -> str | None:
    """Apply one decision to one arch block. Returns a description, or None."""
    if field == "arch range":
        if verdict != "keep":
            return None
        present = [x for x in block.get("teeth_present", [])]
        if fdi in present:
            return None
        block["teeth_present"] = sorted(present + [fdi])
        block["teeth_absent"] = [x for x in block.get("teeth_absent", []) if x != fdi]
        return f"teeth_present += {fdi} (range wins)"

    if field == "present/absent":
        if verdict == "present":
            if fdi not in block.get("teeth_absent", []):
                return None
            block["teeth_absent"] = [x for x in block["teeth_absent"] if x != fdi]
            return f"teeth_absent -= {fdi}"
        if verdict == "absent":
            if fdi not in block.get("teeth_present", []):
                return None
            block["teeth_present"] = [x for x in block["teeth_present"] if x != fdi]
            return f"teeth_present -= {fdi}"
        return None

    if verdict != "drop":
        return None

    if "." in field:                                   # canal_left.adjacent_teeth
        side, key = field.split(".", 1)
        sub = block.get(side)
        if not isinstance(sub, dict) or fdi not in (sub.get(key) or []):
            return None
        sub[key] = [x for x in sub[key] if x != fdi]
        return f"{side}.{key} -= {fdi}"

    value = block.get(field)
    if not isinstance(value, list) or not value:
        return None
    if isinstance(value[0], dict):                     # endodontic / implants
        if not any(e.get("fdi") == fdi for e in value):
            return None
        block[field] = [e for e in value if e.get("fdi") != fdi]
    else:
        if fdi not in value:
            return None
        block[field] = [x for x in value if x != fdi]
    return f"{field} -= {fdi}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="training")
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    decisions = json.loads(Path(args.decisions).read_text())
    index = build_index(args.split)
    gt = ROOT / f"dataset/{args.split}/outputs/ground_truth"

    unknown = [rid for verdict in ("keep", "drop", "present", "absent", "unsure")
               for rid in decisions.get(verdict, []) if rid not in index]
    if unknown:
        print(f"[WARN] {len(unknown)} id(s) not in the current sheet -- the audit "
              f"has moved on since it was rendered:", file=sys.stderr)
        for rid in unknown[:10]:
            print(f"         {rid}", file=sys.stderr)

    loaded, changes, tally = {}, defaultdict(list), Counter()
    for verdict in ("keep", "drop", "present", "absent"):
        for rid in decisions.get(verdict, []):
            if rid not in index:
                continue
            name, arch, field, fdi = index[rid]
            data = loaded.setdefault(name, json.loads((gt / name).read_text()))
            block = data.get(arch)
            if not isinstance(block, dict):
                continue
            done = edit(block, field, fdi, verdict)
            if done:
                changes[name].append(f"{verdict:8} {arch}.{done}")
                tally[verdict] += 1
            else:
                tally[f"{verdict} (no-op)"] += 1

    for name in sorted(changes):
        print(f"{name}")
        for line in changes[name]:
            print(f"    {line}")
    print(f"\n{dict(tally)}")
    print(f"unsure, recorded and not written: {len(decisions.get('unsure', []))}")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return
    for name, data in loaded.items():
        if name not in changes:
            continue
        path = gt / name
        bak = path.with_suffix(".json.bak")
        if not bak.exists():
            bak.write_text(path.read_text())
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    cases = sorted({re.match(r"([A-Z]+\d+)", n).group(1) for n in changes})
    print(f"\nwrote {len(changes)} file(s); originals kept as *.json.bak")
    print("now replay stage 2 (CPU):\n")
    print(f"    python3 code/parse_reports_to_gt.py \\\n"
          f"        --reports-dir dataset/{args.split}/reports "
          f"--schema schema/schema.json \\\n"
          f"        --out-dir dataset/{args.split}/outputs/ground_truth \\\n"
          f"        --from-report-facts --consensus --case-ids {' '.join(cases)}")


if __name__ == "__main__":
    main()
