#!/usr/bin/env python3
"""validate_summary_with_facts.py -- drop summary claims the masks contradict.

The post_process_val arm. postprocess_pred.py already reconciles the VLM's
sources against each other; this stage adds the one source that is not a VLM
read at all -- dataset/$SPLIT/facts/<case>.json, derived from the segmentation
masks -- and DROPS every claim that source contradicts. It never adds a claim
and never rewrites one: a finding either survives untouched or disappears.

That asymmetry is deliberate. The facts files are a high-precision, low-recall
witness (on this validate split their crown list is 0.45 precision but only
0.24 recall against the reports), so they are trustworthy when they say "there
is no tooth there" and untrustworthy when they say nothing. Dropping on
contradiction uses only the trustworthy direction.

Five contradictions are checked, each disableable so the arm can be ablated:

  absent      a per-tooth finding on an FDI the mask lists in teeth_absent.
              The tooth was never segmented, so it got no composite crop and
              nothing the VLM said about it can be backed by an image.
  crown       restoration == "crown" on an FDI missing from structured.crowns.
  implant     an implant at a position missing from structured.implants.
  bridge      any fixed bridge in a case whose facts say bridge_present false.
  canal       a canal-adjacent tooth missing from structured.ian_close_teeth.

Only these five have a counterpart in the masks. Endodontic treatment, posts
and cores, fillings and impaction direction are invisible to a segmentation,
so those claims pass through however wrong they are -- this stage cannot help
the pipeline's two worst precision numbers.

Implants and pontics are exempt from the `absent` rule by construction: they
sit exactly where a tooth is missing, so the mask calling that FDI absent is
agreement, not contradiction.

Usage:
    python3 code/validate_summary_with_facts.py \
        --summary-dir outputs/aksssr_v6_facts_validate/summaries \
        --facts-dir dataset/validate/facts \
        --out-dir outputs/aksssr_v6_postproc_val_validate/summaries \
        --report outputs/aksssr_v6_postproc_val_validate/validation_report.json

    # measure without writing anything
    python3 code/validate_summary_with_facts.py ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

ARCHES = ("mandible", "maxilla")

RULES = ("absent", "crown", "implant", "bridge", "canal")


def as_list(value) -> List:
    return value if isinstance(value, list) else []


def as_dict(value) -> Dict:
    return value if isinstance(value, dict) else {}


def ints(value) -> Set[int]:
    """FDI numbers only. At least one facts file carries a stray None."""
    return {v for v in as_list(value) if isinstance(v, int)}


class Facts:
    """The mask-derived witness for one case."""

    def __init__(self, raw: Dict):
        structured = as_dict(raw.get("structured"))
        self.present = ints(structured.get("teeth_present"))
        # teeth_absent is a positive assertion of absence. Deliberately NOT
        # `everything not in teeth_present`: that would also catch teeth the
        # facts file simply never mentions, which is silence, not denial.
        self.absent = ints(structured.get("teeth_absent"))
        self.crowns = ints(structured.get("crowns"))
        self.implants = ints(structured.get("implants"))
        self.bridge_present = bool(structured.get("bridge_present"))
        self.ian_close = ints(structured.get("ian_close_teeth"))


class Dropper:
    """Applies the rules to one summary, counting every drop it makes."""

    def __init__(self, facts: Facts, enabled: Set[str], case_id: str):
        self.facts = facts
        self.enabled = enabled
        self.case_id = case_id
        self.counts: Counter = Counter()
        self.detail: Dict[str, List[str]] = {r: [] for r in RULES}

    def _note(self, rule: str, what: str) -> None:
        self.counts[rule] += 1
        self.detail[rule].append(what)

    # ── predicates ─────────────────────────────────────────────────────────

    def absent(self, fdi: int) -> bool:
        return "absent" in self.enabled and fdi in self.facts.absent

    def unbacked_crown(self, fdi: int) -> bool:
        return "crown" in self.enabled and fdi not in self.facts.crowns

    def unbacked_implant(self, fdi: int) -> bool:
        return "implant" in self.enabled and fdi not in self.facts.implants

    def unbacked_canal_tooth(self, fdi: int) -> bool:
        return "canal" in self.enabled and fdi not in self.facts.ian_close

    # ── list filters ───────────────────────────────────────────────────────

    def filter_fdi_list(self, fdis: List, where: str) -> List:
        kept = []
        for fdi in fdis:
            if isinstance(fdi, int) and self.absent(fdi):
                self._note("absent", f"{where}:{fdi}")
                continue
            kept.append(fdi)
        return kept

    def filter_records(self, records: List, key: str, where: str) -> List:
        kept = []
        for record in records:
            fdi = as_dict(record).get(key)
            if isinstance(fdi, int) and self.absent(fdi):
                self._note("absent", f"{where}:{fdi}")
                continue
            kept.append(record)
        return kept

    # ── the summary walk ───────────────────────────────────────────────────

    def apply(self, summary: Dict) -> Dict:
        for arch in ARCHES:
            block = as_dict(summary.get(arch))
            if not block:
                continue
            self._arch(arch, block)
        return summary

    def _arch(self, arch: str, block: Dict) -> None:
        w = arch

        # canal-adjacent teeth
        for side, canal in as_dict(block.get("canals")).items():
            canal = as_dict(canal)
            adjacent = as_list(canal.get("adjacent_teeth"))
            if not adjacent:
                continue
            kept = []
            for fdi in adjacent:
                if isinstance(fdi, int) and self.unbacked_canal_tooth(fdi):
                    self._note("canal", f"{w}.canals.{side}:{fdi}")
                    continue
                kept.append(fdi)
            if kept:
                canal["adjacent_teeth"] = kept
            else:
                canal.pop("adjacent_teeth", None)

        # per-tooth finding records
        for key, id_field in (("root_remnants", "fdi"),
                              ("impacted_teeth", "tooth"),
                              ("wisdom_teeth", "tooth")):
            if key in block:
                block[key] = self.filter_records(
                    as_list(block[key]), id_field, f"{w}.{key}")

        # endodontic summary
        endo = as_dict(block.get("endodontic_summary"))
        if endo:
            endo["teeth"] = self.filter_fdi_list(
                as_list(endo.get("teeth")), f"{w}.endodontic_summary.teeth")
            groups = as_dict(endo.get("quality_groups"))
            for quality in list(groups):
                groups[quality] = self.filter_fdi_list(
                    as_list(groups[quality]),
                    f"{w}.endodontic_summary.{quality}")
                if not groups[quality]:
                    groups.pop(quality)
            if "conflicting_sources" in endo:
                endo["conflicting_sources"] = self.filter_fdi_list(
                    as_list(endo.get("conflicting_sources")),
                    f"{w}.endodontic_summary.conflicting")

        # restoration summary -- absent rule for every type, crown rule for crown
        restoration = as_dict(block.get("restoration_summary"))
        groups = as_dict(restoration.get("groups"))
        for rtype in list(groups):
            kept = []
            for fdi in as_list(groups[rtype]):
                if not isinstance(fdi, int):
                    kept.append(fdi)
                    continue
                if self.absent(fdi):
                    self._note("absent", f"{w}.restoration.{rtype}:{fdi}")
                    continue
                if rtype == "crown" and self.unbacked_crown(fdi):
                    self._note("crown", f"{w}.restoration.crown:{fdi}")
                    continue
                kept.append(fdi)
            if kept:
                groups[rtype] = kept
            else:
                groups.pop(rtype)
        conflicting = as_dict(restoration.get("conflicting_sources"))
        for rtype in list(conflicting):
            conflicting[rtype] = self.filter_fdi_list(
                as_list(conflicting[rtype]), f"{w}.restoration.conflicting.{rtype}")
            if not conflicting[rtype]:
                conflicting.pop(rtype)

        # per-tooth findings
        if "tooth_findings" in block:
            block["tooth_findings"] = self._tooth_findings(
                as_list(block["tooth_findings"]), w)

        # intrasinusal roots
        intrasinusal = as_dict(block.get("sinus_intrasinusal_teeth"))
        if intrasinusal:
            intrasinusal["teeth"] = self.filter_fdi_list(
                as_list(intrasinusal.get("teeth")), f"{w}.sinus_intrasinusal")

        # prosthetics: implants and bridges
        prosthetics = as_dict(block.get("prosthetics"))
        if "implants" in prosthetics:
            kept = []
            for implant in as_list(prosthetics["implants"]):
                fdi = as_dict(implant).get("fdi_number")
                if isinstance(fdi, int) and self.unbacked_implant(fdi):
                    self._note("implant", f"{w}.implants:{fdi}")
                    continue
                kept.append(implant)
            if kept:
                prosthetics["implants"] = kept
            else:
                prosthetics.pop("implants")
        if "bridges" in prosthetics and "bridge" in self.enabled \
                and not self.facts.bridge_present:
            for bridge in as_list(prosthetics["bridges"]):
                span = as_dict(bridge)
                self._note("bridge", f"{w}.bridges:"
                                     f"{span.get('span_start')}-{span.get('span_end')}")
            prosthetics.pop("bridges")
        if not prosthetics:
            block.pop("prosthetics", None)

    def _tooth_findings(self, findings: List, where: str) -> List:
        kept = []
        for entry in findings:
            entry = as_dict(entry)
            fdi = entry.get("fdi")
            if isinstance(fdi, int) and self.absent(fdi):
                self._note("absent", f"{where}.tooth_findings:{fdi}")
                continue
            if entry.get("restoration") == "crown" and isinstance(fdi, int) \
                    and self.unbacked_crown(fdi):
                self._note("crown", f"{where}.tooth_findings.crown:{fdi}")
                entry.pop("restoration")
            # An entry carrying nothing but its own FDI is not a finding.
            if set(entry) <= {"fdi"}:
                continue
            kept.append(entry)
        return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-dir", type=Path, required=True,
                    help="postprocess_pred.py output to validate")
    ap.add_argument("--facts-dir", type=Path, required=True,
                    help="dataset/<split>/facts")
    ap.add_argument("--out-dir", type=Path,
                    help="where the validated summaries go (required unless --dry-run)")
    ap.add_argument("--report", type=Path, default=None,
                    help="also write a JSON record of every drop")
    ap.add_argument("--case-ids", nargs="+", default=None)
    ap.add_argument("--disable", nargs="+", default=[], choices=RULES,
                    help="turn a rule off, for ablation")
    ap.add_argument("--dry-run", action="store_true",
                    help="count the drops, write nothing")
    args = ap.parse_args()

    if not args.dry_run and not args.out_dir:
        sys.exit("[FAIL] --out-dir is required unless --dry-run")
    if not args.summary_dir.is_dir():
        sys.exit(f"[FAIL] no summary dir: {args.summary_dir}")
    if not args.facts_dir.is_dir():
        sys.exit(f"[FAIL] no facts dir: {args.facts_dir}")

    enabled = set(RULES) - set(args.disable)
    if not enabled:
        sys.exit("[FAIL] every rule disabled -- nothing to do")

    summaries = sorted(args.summary_dir.glob("*_summary.json"))
    if args.case_ids:
        summaries = [p for p in summaries
                     if p.stem.replace("_summary", "") in args.case_ids]
    if not summaries:
        sys.exit(f"[FAIL] no *_summary.json under {args.summary_dir}")

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] {len(summaries)} summary file(s) from {args.summary_dir}")
    print(f"[INFO] facts: {args.facts_dir}")
    print(f"[INFO] rules enabled: {', '.join(sorted(enabled))}"
          + (f"   disabled: {', '.join(sorted(args.disable))}" if args.disable else ""))
    if args.dry_run:
        print("[INFO] DRY RUN -- nothing will be written")
    print()

    grand = Counter()
    per_case_report = {}
    no_facts = []

    for path in summaries:
        case_id = path.stem.replace("_summary", "")
        summary = json.loads(path.read_text())
        facts_path = args.facts_dir / f"{case_id}.json"
        if not facts_path.exists():
            # No witness means nothing to contradict. Copy through untouched
            # rather than silently emitting a case the arm never validated.
            no_facts.append(case_id)
            if not args.dry_run:
                (args.out_dir / path.name).write_text(json.dumps(summary, indent=2))
            continue

        facts = Facts(json.loads(facts_path.read_text()))
        dropper = Dropper(facts, enabled, case_id)
        validated = dropper.apply(summary)

        if dropper.counts:
            grand.update(dropper.counts)
            per_case_report[case_id] = {
                "counts": dict(dropper.counts),
                "dropped": {r: d for r, d in dropper.detail.items() if d},
            }
            summary_line = "  ".join(f"{r}={dropper.counts[r]}"
                                     for r in RULES if dropper.counts[r])
            print(f"  {case_id:<7} {summary_line}")

        if not args.dry_run:
            (args.out_dir / path.name).write_text(json.dumps(validated, indent=2))

    print()
    if no_facts:
        print(f"[WARN] {len(no_facts)} case(s) had no facts file and passed through "
              f"unvalidated: {', '.join(no_facts)}")
    total = sum(grand.values())
    print(f"[INFO] dropped {total} claim(s) across "
          f"{len(per_case_report)}/{len(summaries)} case(s)")
    for rule in RULES:
        if rule in enabled:
            print(f"         {rule:<8} {grand[rule]:>5}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "summary_dir": str(args.summary_dir),
            "facts_dir": str(args.facts_dir),
            "rules_enabled": sorted(enabled),
            "rules_disabled": sorted(args.disable),
            "cases_without_facts": no_facts,
            "totals": dict(grand),
            "per_case": per_case_report,
        }, indent=2))
        print(f"[PASS] drop record -> {args.report}")

    if not args.dry_run:
        print(f"[PASS] {len(summaries)} validated summary file(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
