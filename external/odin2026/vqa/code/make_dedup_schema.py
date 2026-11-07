#!/usr/bin/env python3
"""
make_dedup_schema.py

Derive schema/schema_dedup.json from schema/schema.json by removing the facts
that are DELIBERATELY redundant in the main pipeline but buy nothing in the
uniform-32 arm.

Why the redundancy exists in the first place
--------------------------------------------
schema.json asks some things twice on purpose, and the two reads land on
DIFFERENT images: dental_arch_findings_{arch} sweeps sixteen teeth in one
panoramic read, while tooth_{fdi}_morphology / _endodontic_treatment /
_eruption read that same tooth on its own composite crop.

postprocess_pred.py then reconciles the two: a finding one source caught and
the other cannot speak to survives, while one the sources actively DISAGREE
about is dropped (see its cross-source agreement gate, plus collect_arch_absent's
multi-source absence derivation). That is worth paying for when the sources
really are independent looks at different pictures -- both the corroboration
and the veto depend on it.

Under uniform-32 they are not. Every fact is asked against the SAME 32 axial
slices, so asking twice produces two reads of one picture: no cross-source
signal, just extra tokens and an extra chance to contradict itself. Hence this
deduplicated copy.

What is removed (2 facts)
-------------------------
  dental_arch_findings_mandible, dental_arch_findings_maxilla
      -> their per-tooth findings map (absent|normal|caries|filling|
         post_and_core|root_canal_treatment|impacted) is covered position by
         position: caries / filling / post_and_core by tooth_{fdi}_morphology's
         with_caries / with_fillings / with_post_and_core flags,
         root_canal_treatment by tooth_{fdi}_endodontic_treatment, absent by
         tooth_{fdi}_eruption's `absent` state, and impacted -- which that fact
         restricts to the third molars by its own definition -- by the four
         wisdom-tooth facts, which are KEPT for exactly that reason.

What is KEPT and why
--------------------
  lower_left_wisdom_tooth, lower_right_wisdom_tooth, upper_right_wisdom_tooth,
  upper_left_wisdom_tooth. The v6.1 plan removed these as duplicates of
  tooth_{18,28,38,48}_eruption + _impaction. That stopped being true at v6.3:
  the composite impaction read was deleted from the schema, and
  build_impacted_teeth now has exactly two sources -- these four facts and
  dental_arch_findings_{arch} (see postprocess_pred.build_impacted_teeth).
  Removing both leaves impaction with NO source, so every "tooth 38 is
  impacted" sentence vanishes from this arm while surviving in every arm it is
  compared against. tooth_{fdi}_eruption carries eruption_state only; the
  `orientation` and `impacted` fields exist nowhere else. Their eruption_state
  IS still duplicated by tooth_{fdi}_eruption -- that residual redundancy is
  the price of keeping impaction, and it is one field on four teeth.

  periodontal_bone_resorption_{arch} looks redundant against
  tooth_{fdi}_periodontal_status but is not: only the arch-level fact carries
  `pattern` (vertical/horizontal), which synthesize_report.py renders and no
  per-tooth field can reconstruct.

Cross-reference scrubbing
-------------------------
Removing a fact is not enough: other facts NAME the removed ones in their
`description`, and those descriptions go straight into the prompt (see
inference_following_schema._describe_field). Left alone, the model would be
told to cross-check against a fact it is never asked. The known references are
scrubbed by exact-substring replacement, and a general net afterwards fails the
run if any removed fact_id still appears anywhere in a surviving fact -- so a
cross-reference added by a FUTURE schema edit breaks this script loudly instead
of leaking into the prompt silently.

schema.json itself is never modified. build_vqa_pairs.py's CALL_PLAN hard-names
the removed fact_ids and resolve_call_plan() raises ValueError if a named fact
is missing, so the main aksssr pipeline keeps working untouched.

Usage:
    python3 code/make_dedup_schema.py                     # write schema/schema_dedup.json
    python3 code/make_dedup_schema.py --dry-run           # show what would change
    python3 code/make_dedup_schema.py --check             # fail if the committed file is stale
"""
import argparse
import json
import sys
from pathlib import Path

# The facts this script drops, each with the facts that already cover it.
# Keyed by fact_id; the value is recorded in the output's provenance block.
REMOVED_FACTS = {
    "dental_arch_findings_mandible":
        "per-tooth findings map covered by tooth_{fdi}_morphology + "
        "_endodontic_treatment + _eruption, and its wisdom-only `impacted` "
        "value by lower_{left,right}_wisdom_tooth",
    "dental_arch_findings_maxilla":
        "per-tooth findings map covered by tooth_{fdi}_morphology + "
        "_endodontic_treatment + _eruption, and its wisdom-only `impacted` "
        "value by upper_{left,right}_wisdom_tooth",
}

# Sections searched for facts. Mirrors inference_following_schema.py's
# GLOBAL_SECTIONS + TOOTH_SECTION.
FACT_SECTIONS = ("mandible", "maxilla", "dental_elements")

# (fact_id, field, exact old substring, replacement).
#
# Exact substrings, not regexes: if schema.json's wording drifts even slightly
# the match fails and the run aborts, which is the point -- a silently-skipped
# scrub would leave a dangling cross-reference in the prompt.
#
# The only surviving facts that name a removed one are the four wisdom-tooth
# reads. Each says its finding is "cross-checked against this tooth's entry in
# dental_arch_findings_{arch}" -- true in the aksssr arm, false here, and
# actively misleading: told its answer will be corroborated, the model has less
# reason to commit. In this arm it is the sole impaction read, so the
# replacement says so.
SCRUBS = [
    (
        fact_id, "description",
        f", cross-checked in postprocessing against this tooth's entry in "
        f"dental_arch_findings_{arch} (panoramic), and its eruption_state also "
        f"against tooth_{fdi}_eruption (composite). There is no composite "
        f"impaction read any more, so the impaction direction here is "
        f"corroborated by the panoramic entry alone.",

        f", whose eruption_state is cross-checked in postprocessing against "
        f"tooth_{fdi}_eruption. This is the ONLY read of this tooth's "
        f"impaction -- nothing else corroborates it, so answer it on what you "
        f"can actually see.",
    )
    for fact_id, fdi, arch in (
        ("lower_left_wisdom_tooth",  38, "mandible"),
        ("lower_right_wisdom_tooth", 48, "mandible"),
        ("upper_right_wisdom_tooth", 18, "maxilla"),
        ("upper_left_wisdom_tooth",  28, "maxilla"),
    )
]


def build_dedup_schema(source: dict) -> dict:
    """Return the deduplicated schema, or raise ValueError on any surprise."""
    out = {}
    seen_fact_ids = set()

    for key, value in source.items():
        if key not in FACT_SECTIONS:
            out[key] = value
            continue
        kept = []
        for fact in value:
            fact_id = fact.get("fact_id")
            seen_fact_ids.add(fact_id)
            if fact_id in REMOVED_FACTS:
                continue
            kept.append(json.loads(json.dumps(fact)))   # deep copy
        out[key] = kept

    missing = sorted(set(REMOVED_FACTS) - seen_fact_ids)
    if missing:
        raise ValueError(
            f"these fact_ids are not in the source schema, so REMOVED_FACTS is "
            f"stale: {missing}")

    _apply_scrubs(out)
    _assert_no_dangling_references(out)

    out["version"] = f"{source.get('version', 'unknown')}-dedup"
    out["_derived_from"] = {
        "source": "schema/schema.json",
        "source_version": source.get("version"),
        "generator": "code/make_dedup_schema.py",
        "why": "uniform-32 asks every fact against the same 32 slices, so the "
               "schema's cross-source redundancy buys no extra signal -- only "
               "tokens and self-contradiction",
        "removed_facts": dict(sorted(REMOVED_FACTS.items())),
        "kept_despite_looking_redundant": {
            "lower_left_wisdom_tooth":
                "since v6.3 there is no tooth_{fdi}_impaction, so with "
                "dental_arch_findings_{arch} removed these four facts are the "
                "ONLY source of impaction; tooth_{fdi}_eruption duplicates "
                "their eruption_state but not `orientation` or `impacted`",
            "lower_right_wisdom_tooth": "same",
            "upper_right_wisdom_tooth": "same",
            "upper_left_wisdom_tooth": "same",
            "periodontal_bone_resorption_mandible":
                "only source of `pattern` (vertical/horizontal), which "
                "synthesize_report.py renders",
            "periodontal_bone_resorption_maxilla": "same",
        },
        "scrubbed_descriptions": sorted({fid for fid, *_ in SCRUBS}),
    }
    return out


def _apply_scrubs(schema: dict) -> None:
    """Rewrite the descriptions that name a removed fact. In place."""
    by_id = {f["fact_id"]: f
             for section in FACT_SECTIONS
             for f in schema.get(section, [])}

    for fact_id, field, old, new in SCRUBS:
        fact = by_id.get(fact_id)
        if fact is None:
            raise ValueError(
                f"scrub target '{fact_id}' is not a surviving fact -- SCRUBS "
                f"is out of sync with REMOVED_FACTS")
        text = fact.get(field)
        if not isinstance(text, str) or old not in text:
            raise ValueError(
                f"scrub for {fact_id}.{field} did not match. The schema wording "
                f"changed; update SCRUBS.\n  looking for: {old!r}\n  found: {text!r}")
        fact[field] = text.replace(old, new).strip()


def _assert_no_dangling_references(schema: dict) -> None:
    """
    The general net: no surviving fact may still mention a removed fact_id
    anywhere. Catches cross-references a future schema edit adds without
    anyone remembering to extend SCRUBS.
    """
    dangling = []
    for section in FACT_SECTIONS:
        for fact in schema.get(section, []):
            blob = json.dumps(fact)
            for removed in REMOVED_FACTS:
                if removed in blob:
                    dangling.append((fact["fact_id"], removed))
    if dangling:
        lines = "\n".join(f"  {fid} still references {ref}" for fid, ref in dangling)
        raise ValueError(
            "removed fact_ids survive in other facts' text -- they would reach "
            "the prompt as instructions to cross-check a fact that is never "
            f"asked. Add a SCRUBS entry for each:\n{lines}")


def _fact_counts(schema: dict) -> str:
    return ", ".join(f"{s}={len(schema.get(s, []))}" for s in FACT_SECTIONS)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="schema/schema.json",
                    help="Full schema to derive from (never modified)")
    ap.add_argument("--out", default="schema/schema_dedup.json",
                    help="Where to write the deduplicated schema")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change; write nothing")
    ap.add_argument("--check", action="store_true",
                    help="Exit non-zero if --out differs from what would be "
                         "generated now. Use this at job start so a stale "
                         "committed file is a hard failure, never a silent one.")
    args = ap.parse_args()

    source_path = Path(args.source)
    if not source_path.is_file():
        sys.exit(f"[FAIL] source schema not found: {source_path}")

    source = json.loads(source_path.read_text())
    try:
        dedup = build_dedup_schema(source)
    except ValueError as e:
        sys.exit(f"[FAIL] {e}")

    text = json.dumps(dedup, indent=2, ensure_ascii=False) + "\n"
    out_path = Path(args.out)

    print(f"[INFO] source : {source_path} (v{source.get('version')}) "
          f"-- {_fact_counts(source)}")
    print(f"[INFO] dedup  : {out_path} (v{dedup['version']}) "
          f"-- {_fact_counts(dedup)}")
    print(f"[INFO] removed: {', '.join(sorted(REMOVED_FACTS))}")

    if args.check:
        if not out_path.is_file():
            sys.exit(f"[FAIL] --check: {out_path} does not exist. Run "
                     f"`python3 code/make_dedup_schema.py` and commit it.")
        if out_path.read_text() != text:
            sys.exit(f"[FAIL] --check: {out_path} is stale relative to "
                     f"{source_path}. Regenerate it and commit the result.")
        print("[PASS] --check: committed dedup schema is up to date")
        return

    if args.dry_run:
        print("[DRY-RUN] nothing written")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
