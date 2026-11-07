"""
verify_report_facts.py -- a SECOND OPINION on the audit's open findings.

Read this before trusting an output of it.

The idea is sound and the shape is RadFact's: each open finding is written back
out as the sentence a radiologist would have written, and handed to the judge
against one reader's own sentences, which answers entailment / contradiction /
neutral with the phrase it used. Per reader, because the combination rule turns
on who said what:

    both claim -> positive        one claims, one denies  -> unscore
    both deny  -> negative        one claims, one silent  -> positive

WHAT IT MEASURED, AND WHY IT IS NOT AN AUTHORITY. On 127 training findings it
returned 85 negative, 42 positive, no conflicts. Splitting the negatives by
whether ANY sentence names the tooth positionally:

    56  no positional sentence   the mechanical screens in audit_report_facts
                                 already prove this; the judge adds nothing
    29  positional sentence      needs judgement -- and this is the class it
                                 gets wrong

F008 is the worked example. Its reader 2 writes "The roots of the first,
second and third molars bilaterally protrude into the maxillary sinuses",
which states 16, 17, 26 and 27, and the judge answered neutral for both
readers on all four. Applying that would have deleted correct ground truth.
That is the same ordinal-list-sharing-a-head-noun form that had already broken
the mechanical resolver, and the prompt had been revised twice for exactly
this class before the run.

The prompt has since gained that form too (untested -- the judge server was
gone before it could be re-run). Do not read that as a fix: each revision was
found by a human sampling the output, and the control set is six findings that
had already been adjudicated by hand, so the agreement it shows is a fit to
those six rather than evidence about the next hundred.

So: use it to sort, not to decide. --apply exists and writes the negatives out
of report_facts with a .bak, and it should not be run over a class the tool is
known to misread. The 19 distinct findings in the suspect set are a twenty
minute read for a human with the sentences in front of them, which is what
build_triage_sheet.py renders.

Usage, once a judge is serving (jobs/judge_server.sh):

    python3 code/verify_report_facts.py --split training --out verdicts.json
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from audit_report_facts import (audit_case, report_text_for,  # noqa: E402
                                split_sentences)

# One extracted finding, written back out as the sentence a radiologist would
# have written. The judge compares THIS against the report's own sentences, so
# it has to read like a report and not like a field name.
HYPOTHESIS = {
    "teeth_absent": "Tooth {fdi} is absent.",
    "teeth_present": "Tooth {fdi} is present in the arch.",
    "present/absent": "Tooth {fdi} is absent.",
    "arch range": "Tooth {fdi} is present in the arch.",
    "crowns": "Tooth {fdi} carries a prosthetic crown.",
    "fillings": "Tooth {fdi} has a conservative restoration.",
    "caries": "Tooth {fdi} is carious.",
    "endodontic": "Tooth {fdi} has endodontic treatment.",
    "post_and_core": "Tooth {fdi} is restored with a post and core.",
    "root_remnants": "Only a root remnant is left of tooth {fdi}.",
    "root_fractures": "Tooth {fdi} has a root fracture.",
    "periapical_lesions": "There is a periapical lesion at the apex of tooth {fdi}.",
    "implants": "There is an endosseous implant in position {fdi}.",
    "unerupted": "Tooth {fdi} is impacted or not fully erupted.",
    "primary_teeth": "Tooth {fdi} is a deciduous tooth still in place.",
    "uncertain_teeth": "Tooth {fdi} could not be assessed with certainty.",
    "canal_right": "The mandibular canal is in contact with the roots of tooth {fdi}.",
    "canal_left": "The mandibular canal is in contact with the roots of tooth {fdi}.",
    "sinus_right": "The root of tooth {fdi} protrudes into the maxillary sinus.",
    "sinus_left": "The root of tooth {fdi} protrudes into the maxillary sinus.",
    "bridges": "Tooth {fdi} is part of a fixed bridge.",
}


# RadFact's toothfairy system message, plus the one thing it cannot know.
#
# It is built for report-vs-report comparison, where both sides write tooth
# NUMBERS, and it says so in as many words: "a reference phrase about a
# different tooth number is never valid evidence". Run unchanged against these
# reports it rejected every positionally-named finding in the control set --
# A022's "a root remnant in the canine region of the 2nd quadrant" (tooth 23)
# and F003_1's "the distal teeth of the 1st quadrant have been endodontically
# treated" (16, 17, 18) all came back not_entailment, while the two hypotheses
# whose teeth were written as digits came back entailment. That is not the
# judge being strict, it is the judge being told the wrong thing: in this
# corpus half the evidence names teeth by position.
#
# So the exact-number discipline is kept and the numbering is explained.
_POSITION_RULES = """
NAMING A TOOTH WITHOUT WRITING ITS NUMBER. These reports name teeth by
position at least as often as by number, and a positional phrase IS an exact
match when it resolves to the hypothesis tooth. FDI numbering makes this
mechanical, not a guess: the first digit is the quadrant (1 upper right,
2 upper left, 3 lower left, 4 lower right) and the second is the tooth,
counting from the midline --

  1 central incisor   2 lateral incisor   3 canine        4 first premolar
  5 second premolar   6 first molar       7 second molar  8 third molar

so "the canine region of the 2nd quadrant" is 23, "the fourth tooth of
quadrant III" is 34, and "the third molars bilaterally" in the lower jaw is
38 and 48. A definite claim about a group covers every tooth in it: "the
distal teeth of the 1st quadrant have been endodontically treated" entails
16, 17 and 18. A range runs along the ARCH, not through the numbers: "teeth
from 13 to 23" is 13-12-11-21-22-23.

SEVERAL ORDINALS CAN SHARE ONE HEAD NOUN, and each of them names a tooth.
"The roots of the FIRST, SECOND AND THIRD MOLARS bilaterally protrude into
the maxillary sinuses" is six teeth -- first molar 6, second molar 7, third
molar 8, in both upper quadrants: 16, 17, 18, 26, 27, 28 -- and it entails a
claim about every one of them. "Bilaterally" and "on both sides" mean both
quadrants of the jaw under discussion; without them, a side word restricts it
to one. This form is common in these reports and is NOT a vague region
phrase: it names its teeth by type, one at a time.

A REGION CAN NAME ONE TOOTH. "A root remnant in the canine region of the 2nd
quadrant" identifies tooth 23 and nothing else, because a quadrant has one
canine -- that is an exact match and it entails a claim about 23. The same is
true of "the 15 region" or "the first molar region of quadrant IV".

WHAT STILL IS NOT EVIDENCE. A region that covers SEVERAL teeth, with a finding
that does not say which of them -- "root remnants in the molar region of the
1st quadrant" (three molars, an unstated number of remnants), "restorations in
the posterior sector" -- states that something exists there without saying
where, and entails a claim about no particular tooth. Neither does a phrase
about a neighbouring tooth, however similar its wording.
"""


def build_system_prompt() -> str:
    from radfact_lite.prompts import load_system_prompt
    from radfact_lite.rf_types import ReportType
    return load_system_prompt("entailment", ReportType.TOOTHFAIRY) + "\n" + _POSITION_RULES


# RadFact answers entailment / not_entailment, and the combination rule needs
# more than that: a reader who CONTRADICTS a finding and a reader who simply
# never mentions it are the difference between "both deny" and "only one
# claims", which the rule sends to opposite outcomes. So the status is the
# three-way NLI one, and the extra instruction says where the line falls.
_THREE_WAY = {
    "type": "json_schema",
    "json_schema": {
        "name": "evidenced_phrase",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "phrase": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string",
                           "enum": ["entailment", "contradiction", "neutral"]},
            },
            "required": ["phrase", "evidence", "status"],
            "additionalProperties": False,
        },
    },
}

_THREE_WAY_RULE = """
ANSWER ONE OF THREE, NOT TWO.
  entailment    -- the reference states the hypothesis, or states something
                   that necessarily includes it. Quote the phrase as evidence.
  contradiction -- the reference states something that CANNOT be true at the
                   same time: it calls the tooth absent when the hypothesis
                   gives it a filling, or calls the sinus clear when the
                   hypothesis puts a root in it. Quote that phrase too.
  neutral       -- the reference simply does not address this tooth or this
                   finding. Evidence must be empty.
Silence is neutral, never contradiction. A report describes what is notable
and does not list every normal tooth, so "the report never mentions it" is not
a denial.
"""


def assess(client, reference, hypothesis, system):
    """RadFact's entailment call, three-way, with our system message.

    Same few-shot pairs and the same query shape, so a verdict stays
    comparable with a RadFact one; only the status enum and the instructions
    around it differ.
    """
    import json as _json
    from radfact_lite.clients import Message
    from radfact_lite.entailment import _expand_single_phrase_examples
    from radfact_lite.prompts import load_few_shot_json
    from radfact_lite.rf_types import ReportType

    messages = [Message(role="system", content=system + _THREE_WAY_RULE)]
    for shot in load_few_shot_json("entailment", ReportType.TOOTHFAIRY):
        for pair in _expand_single_phrase_examples(shot):
            messages.append(Message(role="user",
                                    content=_json.dumps(pair["input"], ensure_ascii=False)))
            messages.append(Message(role="assistant",
                                    content=_json.dumps(pair["output"], ensure_ascii=False)))
    messages.append(Message(role="user", content=_json.dumps(
        {"reference": reference, "hypothesis": hypothesis}, ensure_ascii=False)))
    out = client.complete_json(messages=messages, response_format=_THREE_WAY)
    return out["status"], list(out.get("evidence") or [])


# The user's rule, and the reason each line of it is right:
#   both claim              -> positive   corroborated
#   both deny               -> negative   the finding is against the evidence
#   one claims, one denies  -> unscore    a genuine reader disagreement is not
#                                         something a verifier gets to settle
#   one claims, one silent  -> positive   silence is not a denial; a report
#                                         describes what is notable
# A case with a single reader has no second opinion, so a claim stands and
# anything else drops -- there is nothing to corroborate it with.
def combine(statuses):
    claims = sum(s == "entailment" for s in statuses)
    denies = sum(s == "contradiction" for s in statuses)
    if claims and denies:
        return "unscore"
    if claims:
        return "positive"
    if denies:
        return "negative"
    return "negative"


def findings_for(split: str, case_ids=None):
    """[(stem, arch, field, fdi, report_text)] for every ERROR the audit raises."""
    gt = ROOT / f"dataset/{split}/outputs/ground_truth"
    reports = ROOT / f"dataset/{split}/reports"
    out = []
    for path in sorted(gt.glob("*_report_facts.json")):
        stem = path.name.replace("_report_facts.json", "")
        case = re.match(r"([A-Z]+\d+)", stem).group(1)
        if case_ids and case not in case_ids:
            continue
        # EVERY reader's report, separately -- the combination rule turns on
        # who said what, so merging them first would throw the question away.
        per_reader = {r.name: r.read_text(encoding="utf-8")
                      for r in sorted(reports.glob(f"{case}_*.txt"))}
        text = report_text_for(stem, case, reports)
        if not text or not per_reader:
            continue
        facts = json.loads(path.read_text())
        for level, screen, message in audit_case(case, facts, text,
                                                 {"not-in-text", "contradiction",
                                                  "arch-range", "canal-adjacency",
                                                  "intrasinusal"}):
            if level != "ERROR":
                continue
            m = re.search(r"^(\w+)\.([\w.]+) asserts (\d+)", message)
            m2 = re.search(r"^(\w+): (\d+) is in BOTH", message)
            m3 = re.search(r"^(\w+)\.(canal_\w+|sinus_\w+) (?:claims contact with|puts) (\d+)",
                           message)
            m4 = re.search(r"^(\w+): \".+\" sweeps the arch and teeth_present is "
                           r"missing \[([\d, ]+)\]", message)
            if m:
                rows = [(m.group(1), m.group(2).split(".")[0], int(m.group(3)))]
            elif m2:
                rows = [(m2.group(1), "present/absent", int(m2.group(2)))]
            elif m3:
                rows = [(m3.group(1), m3.group(2), int(m3.group(3)))]
            elif m4:
                rows = [(m4.group(1), "arch range", int(x)) for x in m4.group(2).split(",")]
            else:
                continue
            for arch, field, fdi in rows:
                out.append((stem, arch, field, fdi, per_reader))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="training")
    ap.add_argument("--case-ids", nargs="+", default=None)
    ap.add_argument("--vllm-url", default=None,
                    help="defaults to the URL judge_server.sh advertises in .judge/url")
    ap.add_argument("--model", default="qwen3-text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--apply", action="store_true",
                    help="write the not_entailment findings out of report_facts")
    args = ap.parse_args()

    url = args.vllm_url
    if not url:
        advertised = ROOT / ".judge/url"
        if not advertised.exists():
            sys.exit("[FAIL] no --vllm-url and no .judge/url -- start jobs/judge_server.sh")
        url = advertised.read_text().strip()

    from radfact_lite.clients import _ChatJSONClient

    client = _ChatJSONClient(args.model, api_key="EMPTY", base_url=url.rstrip("/"),
                             timeout=120, max_retries=3)
    system = build_system_prompt()

    rows = findings_for(args.split, set(args.case_ids) if args.case_ids else None)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[INFO] {len(rows)} finding(s) to verify against {url}")

    verdicts, tally = [], Counter()
    for i, (stem, arch, field, fdi, per_reader) in enumerate(rows, 1):
        template = HYPOTHESIS.get(field)
        if not template:
            continue
        hypothesis = template.format(fdi=fdi)
        # One call per reader. Each sees that radiologist's sentences and
        # nothing else, so "the other one never mentions it" stays visible
        # instead of being merged into a single reference list.
        by_reader, failed = {}, False
        for name, text in sorted(per_reader.items()):
            try:
                status, evidence = assess(client, split_sentences(text),
                                          hypothesis, system)
            except Exception as exc:                   # noqa: BLE001
                print(f"  [{i}/{len(rows)}] {stem} {field} {fdi}: FAILED {exc}",
                      file=sys.stderr)
                failed = True
                break
            by_reader[name] = {"status": status, "evidence": evidence}
        if failed:
            continue
        verdict = combine([r["status"] for r in by_reader.values()])
        tally[verdict] += 1
        verdicts.append({"file": f"{stem}_report_facts.json", "stem": stem,
                         "arch": arch, "field": field, "fdi": fdi,
                         "hypothesis": hypothesis, "verdict": verdict,
                         "readers": by_reader})
        if i % 20 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] {dict(tally)}")

    Path(args.out).write_text(json.dumps(verdicts, indent=2, ensure_ascii=False))
    print(f"\n[INFO] {dict(tally)} -> {args.out}")

    if args.apply:
        from collections import defaultdict
        drops = defaultdict(list)
        for v in verdicts:
            if v["verdict"] == "negative":
                drops[v["file"]].append(v)
        gt = ROOT / f"dataset/{args.split}/outputs/ground_truth"
        for name, items in sorted(drops.items()):
            path = gt / name
            data = json.loads(path.read_text())
            bak = path.with_suffix(".json.bak")
            if not bak.exists():
                bak.write_text(path.read_text())
            for v in items:
                block, field, fdi = data.get(v["arch"]) or {}, v["field"], v["fdi"]
                if field in ("present/absent",):
                    block["teeth_absent"] = [x for x in block.get("teeth_absent", [])
                                             if x != fdi]
                elif field in ("canal_right", "canal_left"):
                    sub = block.get(field) or {}
                    sub["adjacent_teeth"] = [x for x in sub.get("adjacent_teeth", [])
                                             if x != fdi]
                elif field in ("sinus_right", "sinus_left"):
                    sub = block.get(field) or {}
                    sub["intrasinusal_teeth"] = [x for x in sub.get("intrasinusal_teeth", [])
                                                 if x != fdi]
                elif field in ("endodontic", "unerupted", "implants"):
                    block[field] = [e for e in block.get(field, [])
                                    if e.get("fdi") != fdi]
                elif field == "arch range":
                    continue                      # an ADD, never a drop
                elif field in block:
                    block[field] = [x for x in block[field] if x != fdi]
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        cases = sorted({re.match(r"([A-Z]+\d+)", v["stem"]).group(1)
                        for v in verdicts if v["verdict"] == "negative"})
        print(f"[INFO] wrote {len(drops)} file(s); originals kept as *.json.bak")
        print("[INFO] now replay stage 2 (CPU):\n")
        print(f"    python3 code/parse_reports_to_gt.py \\\n"
              f"        --reports-dir dataset/{args.split}/reports "
              f"--schema schema/schema.json \\\n"
              f"        --out-dir dataset/{args.split}/outputs/ground_truth \\\n"
              f"        --from-report-facts --consensus --case-ids {' '.join(cases)}")


if __name__ == "__main__":
    main()
