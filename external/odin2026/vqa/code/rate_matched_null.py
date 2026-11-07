#!/usr/bin/env python3
"""Arm 0 -- the rate-matched null. Suppress baseline claims, no model involved.

`vision_sft_plan_light.md` §4.5, and R1 in the parent plan.

WHAT QUESTION THIS ANSWERS. If arm 2 raises precision, there are two reasons it
could have: it learned to SEE better, or it learned to ASSERT less. Those look
identical in a precision column and they are worth very different amounts --
the second is available for free, on CPU, with no adapter and no change to the
submission container.

So: take the baseline's own predictions, throw away claims until the per-field
claim count equals arm 2's, and score the result on the same path. Whatever
precision that buys is what arm 2's claim-rate change was worth on its own.
Arm 2 has to beat it to have earned anything perceptual.

The few-shot probe is why this is mandatory rather than nice to have: it cut
claims 45%, raised precision 0.129 -> 0.168, and moved the Final Score +0.0081
with a CI straddling zero. A pure operating-point slide along one ROC curve,
and it looked like a result.

THE SUPPRESSION MUST BE BLIND. Anything that picks WHICH claims to drop using
the image, the model's confidence, or its prose would be a perceptual
intervention, and then arm 0 stops being a null. Two orderings, both blind:

  --order random   seeded uniform among a field's positive claims. The
                   strictest null: it knows nothing at all.
  --order prior    drop where the finding is rarest a priori -- per (field,
                   FDI) positive rates read from the TRAINING GT, never from
                   validate (§0). Still blind to this case's pixels, but it
                   knows base rates, so it is the null a careful person would
                   build. If arm 2 cannot beat this one either, the gain is
                   epidemiology.

WHAT COUNTS AS A CLAIM follows survey_facts.py exactly, because arm 0 has to
be scored by it: a `bool` field claims when it is true, a `list` field claims
once per element. Enums and maps are scored as ACCURACY -- right/scored, with
no false-positive column -- so there is no claim to suppress and they are left
alone. That is the same line §3.5 draws for the 6:1 negative cap, and for the
same reason: "negative" is undefined for `eruption_state`.

Runs in `cbct_base` on the login node. No GPU, no model, seconds.

    python3 code/rate_matched_null.py \\
        --pred-dir outputs/aksssr_v7_validate/predictions_awq/predictions_v7.1 \\
        --match-claims outputs/vsft_arm2_validate/predictions \\
        --out-dir outputs/vsft_arm0_validate/predictions

    python3 code/rate_matched_null.py --census --pred-dir <dir>    # count only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from survey_facts import field_kind, schema_facts  # noqa: E402

# survey_facts scores these two kinds with a false-positive column; everything
# else is accuracy, and accuracy cannot be gamed by saying less.
CLAIMABLE = ("bool", "list")


class Claim:
    """One suppressible positive assertion, located precisely enough to undo."""

    __slots__ = ("case", "container", "block", "field", "kind", "item", "fdi")

    def __init__(self, case, container, block, field, kind, item=None, fdi=None):
        self.case, self.container, self.block = case, container, block
        self.field, self.kind, self.item, self.fdi = field, kind, item, fdi

    @property
    def key(self):
        """The per-field bucket claims are matched in, aggregated over teeth."""
        return (self.block_template, self.field)

    @property
    def block_template(self):
        return f"tooth_{{fdi}}_{self.block.split('_', 2)[-1]}" if self.fdi else self.block

    def __repr__(self):
        return f"<{self.case}/{self.container}/{self.block}.{self.field}>"


def iter_blocks(doc, facts):
    """(container, block_name, block_dict, object_fields, fdi) over a pred doc."""
    for _group, entries in facts.items():
        for output_field, section, object_fields in entries:
            if section != "dental_elements":
                block = (doc.get("global") or {}).get(output_field)
                if isinstance(block, dict):
                    yield "global", output_field, block, object_fields, None
                continue
            for tooth_key, tooth in (doc.get("teeth") or {}).items():
                if not isinstance(tooth, dict):
                    continue
                fdi = tooth_key.split("_")[-1]
                name = output_field.replace("{fdi}", fdi)
                block = tooth.get(name)
                if isinstance(block, dict):
                    yield tooth_key, name, block, object_fields, fdi


def collect_claims(pred_dir: Path, facts, case_ids=None):
    """Every suppressible claim in a prediction dir, and the docs they live in."""
    claims, docs = [], {}
    for path in sorted(pred_dir.glob("*_pred.json")):
        case = path.name[: -len("_pred.json")]
        if case_ids and case not in case_ids:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        docs[case] = doc
        for container, block_name, block, object_fields, fdi in iter_blocks(doc, facts):
            for field, spec in object_fields.items():
                kind = field_kind(spec if isinstance(spec, str) else str(spec))
                if kind not in CLAIMABLE:
                    continue
                value = block.get(field)
                if kind == "bool":
                    if str(value).strip().lower() == "true":
                        claims.append(Claim(case, container, block_name, field,
                                            kind, fdi=fdi))
                else:
                    for item in (value if isinstance(value, list) else []):
                        claims.append(Claim(case, container, block_name, field,
                                            kind, item=item, fdi=fdi))
    return claims, docs


def census(claims):
    counts = Counter()
    for c in claims:
        counts[c.key] += 1
    return counts


def load_priors(prior_dir: Path, facts):
    """Per (fact, field, FDI) positive rate from the training GT.

    §0: refuse a validate source outright rather than trusting the caller to
    have passed the right directory. A prior fitted on validate would make
    arm 0 a validate-tuned artifact, and validate is read once per arm at the
    end or it is not held out.
    """
    resolved = prior_dir.resolve()
    if "validate" in resolved.parts:
        raise SystemExit(
            f"[FAIL] --prior-dir is under a validate path ({resolved}).\n"
            f"       Priors come from `training` only (§0).")

    hits, total = Counter(), Counter()
    files = sorted(resolved.glob("*_gt.json"))
    if not files:
        raise SystemExit(f"[FAIL] no *_gt.json in {resolved}")
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        for _container, block_name, block, object_fields, fdi in iter_blocks(doc, facts):
            template = (f"tooth_{{fdi}}_{block_name.split('_', 2)[-1]}"
                        if fdi else block_name)
            for field, spec in object_fields.items():
                kind = field_kind(spec if isinstance(spec, str) else str(spec))
                if kind not in CLAIMABLE:
                    continue
                value = block.get(field)
                if value is None:
                    continue          # GT null: the reader did not answer here
                key = (template, field, fdi)
                total[key] += 1
                if kind == "bool":
                    hits[key] += int(str(value).strip().lower() == "true")
                else:
                    hits[key] += int(bool(value))
    print(f"[INFO] priors from {len(files)} GT file(s) in {resolved.name}")
    return {k: hits[k] / total[k] for k in total if total[k]}


def suppress(doc, claim) -> bool:
    """Undo one claim in place. Returns whether anything changed."""
    block = ((doc.get("global") or {}) if claim.container == "global"
             else (doc.get("teeth") or {}).get(claim.container) or {}).get(claim.block)
    if not isinstance(block, dict):
        return False
    if claim.kind == "bool":
        if str(block.get(claim.field)).strip().lower() != "true":
            return False
        block[claim.field] = False
        return True
    value = block.get(claim.field)
    if not isinstance(value, list) or claim.item not in value:
        return False
    value.remove(claim.item)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True,
                    help="baseline predictions -- the claims to suppress")
    ap.add_argument("--match-claims", type=Path,
                    help="arm 2's predictions -- the per-field target counts")
    ap.add_argument("--out-dir", type=Path, help="where to write arm 0")
    ap.add_argument("--census", action="store_true",
                    help="print --pred-dir's per-field claim counts and stop")
    ap.add_argument("--order", choices=("random", "prior"), default="random",
                    help="which claims to drop first (default: random)")
    ap.add_argument("--prior-dir", type=Path,
                    default=ROOT / "dataset/training/outputs/ground_truth",
                    help="TRAINING GT for --order prior (§0: never validate)")
    ap.add_argument("--schema", type=Path, default=ROOT / "schema/schema.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    facts = schema_facts(args.schema)
    base_claims, docs = collect_claims(args.pred_dir, facts)
    base = census(base_claims)
    print(f"[INFO] {args.pred_dir.name}: {len(docs)} case(s), "
          f"{len(base_claims):,} claim(s) over {len(base)} field(s)")

    if args.census:
        for (block, field), n in sorted(base.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>6}  {block}.{field}")
        return 0

    if not args.match_claims or not args.out_dir:
        raise SystemExit("[FAIL] --match-claims and --out-dir are required "
                         "unless --census")
    if not args.match_claims.exists():
        raise SystemExit(f"[FAIL] {args.match_claims} does not exist. Arm 0 "
                         f"matches ARM 2's counts, so it cannot run before "
                         f"arm 2 has produced predictions (§4.5).")

    # Match on the case set the two dirs share, and say so rather than
    # silently scoring different denominators.
    target_claims, target_docs = collect_claims(args.match_claims, facts)
    shared = sorted(set(docs) & set(target_docs))
    if not shared:
        raise SystemExit("[FAIL] the two prediction dirs share no case ids")
    if len(shared) != len(docs) or len(shared) != len(target_docs):
        print(f"[WARN] case sets differ: baseline {len(docs)}, target "
              f"{len(target_docs)}, shared {len(shared)}. Rate-matching on "
              f"the shared {len(shared)} -- a coverage difference between "
              f"arms is not a finding about them.")
    base_claims = [c for c in base_claims if c.case in set(shared)]
    target = census([c for c in target_claims if c.case in set(shared)])
    base = census(base_claims)

    priors = load_priors(args.prior_dir, facts) if args.order == "prior" else {}
    rng = random.Random(args.seed)

    by_field = defaultdict(list)
    for c in base_claims:
        by_field[c.key].append(c)

    dropped, rows, higher = 0, [], []
    for key in sorted(by_field, key=lambda k: (-base[k], k)):
        want, have = target.get(key, 0), base[key]
        if want >= have:
            if want > have:
                higher.append((key, have, want))
            rows.append((key, have, want, 0))
            continue
        pool = list(by_field[key])
        rng.shuffle(pool)                      # seeded; also breaks prior ties
        if args.order == "prior":
            pool.sort(key=lambda c: priors.get((c.block_template, c.field, c.fdi), 0.0))
        n = have - want
        for claim in pool[:n]:
            dropped += suppress(docs[claim.case], claim)
        rows.append((key, have, want, n))

    print(f"\n[INFO] per-field claim counts ({args.order} order, seed {args.seed}):")
    print(f"  {'field':<52} {'base':>6} {'arm2':>6} {'drop':>6}")
    for (block, field), have, want, n in rows:
        mark = "  <-- arm 2 claims MORE" if want > have else ""
        print(f"  {block + '.' + field:<52} {have:>6} {want:>6} {n:>6}{mark}")

    if higher:
        print(f"\n[NOTE] arm 2 claims MORE than baseline in {len(higher)} "
              f"field(s), which suppression cannot match. Arm 0 is a null for "
              f"the fields where arm 2 said LESS; on these it is the baseline "
              f"unchanged, and the comparison there is arm 2 vs baseline "
              f"directly.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for case in shared:
        (args.out_dir / f"{case}_pred.json").write_text(
            json.dumps(docs[case], indent=2), encoding="utf-8")

    after = census(collect_claims(args.out_dir, facts)[0])
    residual = {k: (after.get(k, 0), target.get(k, 0))
                for k in set(after) | set(target)
                if after.get(k, 0) > target.get(k, 0)}
    if residual:
        raise SystemExit(f"[FAIL] {len(residual)} field(s) still above arm 2's "
                         f"count after suppression, e.g. {next(iter(residual))}. "
                         f"A claim was counted but not undone.")
    print(f"\n[PASS] wrote {len(shared)} case(s) to {args.out_dir}: "
          f"{dropped:,} claim(s) suppressed, {sum(after.values()):,} remain "
          f"(arm 2: {sum(target.values()):,})")

    (args.out_dir.parent / "arm0_meta.json").write_text(json.dumps({
        "pred_dir": str(args.pred_dir), "match_claims": str(args.match_claims),
        "order": args.order, "seed": args.seed, "cases": shared,
        "dropped": dropped,
        "per_field": [{"block": b, "field": f, "base": h, "target": w, "dropped": n}
                      for (b, f), h, w, n in rows],
        "arm2_claims_more": [{"block": b, "field": f, "base": h, "target": w}
                             for (b, f), h, w in higher],
    }, indent=2), encoding="utf-8")

    print(f"\n  Next: score it exactly like any other arm --\n"
          f"    jobs/postprocess_now.sh {args.out_dir.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
