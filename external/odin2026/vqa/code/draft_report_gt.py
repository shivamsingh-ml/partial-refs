#!/usr/bin/env python3
"""
draft_report_gt.py -- draft REPORT_GT entries for cases that do not have any.

survey_findings.REPORT_GT is hand-read from dataset/validate/reports and covers
the 40 validate cases only. Development for the few-shot probe runs on the
TRAINING split (validate is held out), so the training cases being scored need
their own entries before survey_findings can measure anything on them.

This does the mechanical half. It never decides ground truth -- it puts the two
independent drafts next to each other and marks where they disagree, because
those are the sentences worth reading:

  KEYWORD  survey_findings.finding_sentences() over the reports. Free, instant,
           no model. Its keyword/strong/noise tables are the same ones the
           survey scores with, so a finding it cannot see is one the survey
           would not have credited anyway.
  LLM      parse_reports_to_gt.py's {case}_gt.json, if present. That file is
           already prediction-shaped, so survey_findings.pred_claims() reads it
           directly -- no second parser, no re-keying.

Then it prints a REPORT_GT-shaped block to paste into survey_findings.py, with
the supporting sentence quoted beside every FDI so the hand-check is a read
rather than a hunt. Categories the two drafts disagree on are marked; nothing
is silently resolved.

HOW GOOD IS THE DRAFT -- measured against the 40 hand-read validate cases,
which is the only place both a draft and a trusted answer exist:

    hand-read findings          290
    drafted but not in GT       157   deleted during review (visible)
    in GT but not drafted        15   MISSED unless you read the report

So: review means deleting about two wrong FDIs for every right one, and the
draft still misses ~5%. IT IS NOT A SUBSTITUTE FOR READING THE REPORT. The
residual misses are not parser gaps and adding regexes will not fix them:

  - anaphora: F043's "BOTH presumably restored with post-and-core" names its
    teeth in the previous sentence;
  - noise-filtered wording: A041's crowns 24 and 26 come from "radiolucent
    crowns in the 24-27 region", which the filter demotes and the hand reader
    accepted -- so weak sentences are now PRINTED, flagged, rather than
    dropped silently;
  - a laterality disagreement between report and GT (F003 names quadrant 1,
    REPORT_GT records quadrant 2).

Span expansion IS handled, and mattered: without it the drafter missed a
further 9 findings, every one of them the interior of a span like A022's
"circular fixed prosthetic rehabilitation" 33-43 or P014's 3.3-4.3.

But ONLY on explicit range words, never on a bare hyphen -- that distinction
cost 48 fabricated FDIs when it was got wrong (205 extras, down to 157). These
reports use the hyphen as a LIST separator: S0043's "Endodontic treatment
involving teeth 12-13-17-21-22-23-27" names seven teeth, and expanding each
adjacent pair invented 11, 14, 15 and 16. Fabricated ground truth is worse
than missing ground truth -- a missing entry understates the score, an
invented one marks correct model silence as a false negative and scores the
arm against findings that were never in the report.

WHY PASTE INTO survey_findings.py rather than write a side file: REPORT_GT is
the single source for these labels (CLAUDE.md -- never re-key them elsewhere),
it is keyed by case_id, and training and validate case IDs do not collide. A
second table would drift.

Usage
─────
    # both drafts, for the 4 smoke-test cases
    python3 code/draft_report_gt.py --cases A011,F007,P020,S0003 \\
        --reports-dir dataset/training/reports \\
        --gt-dir outputs/ground_truth

    # keyword draft only (no LLM pass run yet)
    python3 code/draft_report_gt.py --cases A011 --reports-dir dataset/training/reports
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from survey_findings import (  # noqa: E402
    CATEGORIES,
    KEYWORD_RE,
    STRONG_RE,
    finding_sentences,
    pred_claims,
    report_paths,
)

# "48", "4.8" and "e.d. 34" all appear in these reports; split_sentences()
# already protects the dotted form, so both shapes reach us intact.
FDI_RE = re.compile(r"\b([1-4])\.?([1-8])\b")

# Teeth in ARCH order, which is not numeric order: quadrant 1 runs backwards to
# the midline before quadrant 2 runs outwards again. Needed because reports
# state prosthetic work as a SPAN across the arch.
UPPER_ARCH = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_ARCH = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

# "from 3.3 to 4.3", "from tooth 12 to tooth 21", "23 through 27".
#
# EXPLICIT RANGE WORDS ONLY -- a bare hyphen is NOT a span in these reports,
# it is a list separator: S0043's "Endodontic treatment involving teeth
# 12-13-17-21-22-23-27" names seven teeth, and reading each adjacent pair as a
# range invented 11, 14, 15 and 16, none of which the report states.
# Fabricated ground truth is far worse than missing ground truth -- it turns
# correct model silence into a false negative and scores the arm against
# findings that were never there.
#
# The optional noun ("tooth", "teeth", "e.d.") between the connector and the
# second FDI matters: the reports write both "from tooth 23 to 27" and "from
# tooth 12 to tooth 21", and only the first matched before.
SPAN_RE = re.compile(
    r"\b([1-4])\.?([1-8])\s*(?:to|through|thru|a|al|–|—)\s*"
    r"(?:tooth\s+|teeth\s+|e\.?\s?d\.?\s*)?([1-4])\.?([1-8])\b",
    re.IGNORECASE)


def expand_span(a: int, b: int) -> Set[int]:
    """Every tooth from a to b along the arch, inclusive.

    A span is not a numeric range. "Splinted crowns from 3.3 to 4.3" covers
    33, 32, 31, 41, 42, 43 -- it crosses the midline, and numerically that is
    33..43 which would wrongly sweep in the whole of quadrant 3. Cross-ARCH
    pairs (an upper and a lower tooth) are not a span at all and expand to
    nothing.
    """
    for arch in (UPPER_ARCH, LOWER_ARCH):
        if a in arch and b in arch:
            i, j = sorted((arch.index(a), arch.index(b)))
            return set(arch[i:j + 1])
    return set()


def fdis_in(sentence: str) -> Set[int]:
    """Every FDI a sentence asserts, INCLUDING the interior of any span.

    Without span expansion this misses the teeth between the endpoints, which
    is not a rare edge case: checked against the 40 hand-read validate cases,
    it accounted for every one of the 24 findings the drafter otherwise
    dropped -- A022's "circular fixed prosthetic rehabilitation" 33-43, F003's
    24-27, F043's and P014's 3.3-4.3.
    """
    found = {int(q + t) for q, t in FDI_RE.findall(sentence)}
    for q1, t1, q2, t2 in SPAN_RE.findall(sentence):
        found |= expand_span(int(q1 + t1), int(q2 + t2))
    return found


# "38 mesioverted impacting on the crown of 37, 48 impacting on the distal
# root of 47." Both 37 and 47 are what the impacted tooth PRESSES AGAINST, and
# fdis_in returns every number in the sentence, so the drafter filed them as
# impacted themselves. Five of the eight drafted impaction entries carried a
# victim this way (F018, F033, F055, F064, S0033) -- invisible in review,
# because the sentence really does contain both the finding word and the number.
#
# Only impaction needs this. A restoration sentence names the teeth it
# restores; an impaction sentence names an aggressor and a victim, and the
# grammar is the only thing that tells them apart.
#
# TWO TRAPS, both found by the reviewer on real sentences:
#
#   "The right mandibular canal is in close relationship with impacted tooth
#   48."  Here the victim preposition is present but 48 is the IMPACTED tooth,
#   not the thing impacted upon -- the subject is the canal. So a tail that
#   carries an impaction word for itself is not a victim clause at all. Without
#   this guard the rule deletes the very findings it exists to protect.
#
#   "...48 impacting on the crown of 47 and 38 in relationship with the
#   cervical region of 37."  The clause break before "38" is a bare "and" with
#   no comma, so a tail that runs to the end of the sentence swallows 38 -- an
#   impacted tooth -- as a victim. The split therefore breaks on "and" when a
#   number follows it.
VICTIM_LEAD = re.compile(
    r"(?:impact\w*|imping\w*|press\w*|encroach\w*)\s+(?:on|upon|against)\b"
    r"|in\s+(?:close\s+)?(?:relationship|contact|contiguity)\s+with\b", re.I)
_VICTIM_CLAUSE_END = re.compile(r"[;.]|,\s*(?:and\s+)?(?=\d|tooth|teeth)|\s+and\s+(?=\d)")


def victim_fdis(sentence: str) -> Set[int]:
    """FDIs a sentence names as the thing an impacted tooth acts ON."""
    out: Set[int] = set()
    for m in VICTIM_LEAD.finditer(sentence):
        tail = _VICTIM_CLAUSE_END.split(sentence[m.end():])[0]
        if KEYWORD_RE["impaction"].search(tail):
            continue          # "...with impacted tooth 48" -- that IS the finding
        out |= fdis_in(tail)
    return out


# A sentence often carries TWO findings and both their tooth lists:
#
#   "conservative restorations are present on the crowns of teeth 23, 24, 25,
#   26, 27 and a probable prosthetic crown on tooth 16"
#
# fdis_in gives every number to whichever category matched, so all six teeth
# were filed as crowns AND as fillings. 23-27 are fillings; 16 is the crown.
# Six of the 35 findings deleted in review came from this one shape.
#
# A nearest-preceding-keyword rule was tried and REJECTED on measurement: over
# the 40 hand-read validate cases it cut drafted extras 115 -> 85 but pushed
# misses 17 -> 42, because reports name the tooth first about as often as they
# name the finding first ("12 with endodontic material ... and restored with a
# prosthetic crown" puts 12 before both keywords, so it fell out of both
# categories). Extras are deleted in review and are visible; misses are
# invisible. The trade is the wrong way round.
#
# So the drafter does not guess. It FLAGS: an FDI that one sentence files
# under two categories is printed as contested, which is exactly the shape a
# human settles in seconds and a rule keeps getting wrong.
def contested_fdis(reports_dir: Path, case_id: str) -> Dict[int, Dict[str, List[str]]]:
    """{fdi: {category: [sentences]}} for FDIs one sentence files twice.

    NOT an error report -- a question. Two shapes produce a collision and only
    one is wrong:

      BOTH TRUE   "Prosthetic crown on tooth 36, which has undergone adequate
                  endodontic therapy" -- 36 really is crowned and root-filled.
      TEETH SPLIT "conservative restorations on the crowns of 23, 24, 25, 26,
                  27 and a probable prosthetic crown on 16" -- 23-27 are
                  fillings, 16 is the crown, and every one of the six was
                  filed under both.

    Telling them apart is a reading, not a rule: the proximity rule that tried
    measured worse than printing the question (see the note above).
    """
    per_sentence: Dict[str, Dict[str, Set[int]]] = {}
    for category in CATEGORIES:
        for _name, sentence, strong in finding_sentences(reports_dir, case_id, category):
            if not strong:
                continue
            per_sentence.setdefault(sentence, {})[category] = category_fdis(sentence, category)
    out: Dict[int, Dict[str, List[str]]] = {}
    for sentence, cats in per_sentence.items():
        if len(cats) < 2:
            continue
        for fdi in set.intersection(*cats.values()) if len(cats) > 1 else set():
            for category in cats:
                out.setdefault(fdi, {}).setdefault(category, []).append(sentence)
    return out


def category_fdis(sentence: str, category: str) -> Set[int]:
    """Every FDI a sentence asserts for one category.

    Every number in the sentence, minus the impaction victims -- see
    victim_fdis, which measured clean where proximity did not.
    """
    found = fdis_in(sentence)
    if category == "impaction":
        found -= victim_fdis(sentence)
    return found


def keyword_draft(reports_dir: Path, case_id: str):
    """({category: {fdi: [sentences]}}, {category: [weak sentences]}).

    Strong hits become candidates. WEAK hits -- sentences the noise filter
    demotes -- are returned separately and printed rather than dropped,
    because they are not always wrong: A041's crown findings 24 and 26 live in
    "Presence of radiolucent crowns in the 24-27 region", which the filter
    demotes and the hand reader accepted. Showing them keeps the judgement
    with the human instead of silently deciding it here.
    """
    out: Dict[str, Dict[int, List[str]]] = {c: {} for c in CATEGORIES}
    weak: Dict[str, List[str]] = {c: [] for c in CATEGORIES}
    for category in CATEGORIES:
        for _name, sentence, strong in finding_sentences(reports_dir, case_id, category):
            if not strong:
                if sentence not in weak[category]:
                    weak[category].append(sentence)
                continue
            for fdi in category_fdis(sentence, category):
                out[category].setdefault(fdi, []).append(sentence)
    return out, weak


def per_report_draft(reports_dir: Path, case_id: str):
    """{report name: {category: {fdi}}} -- who asserted what, reader by reader.

    REPORT_GT records the union of a case's reports and does not remember which
    reader contributed which tooth, so migrating to the agreement rule needs
    that attribution recovered from the text.
    """
    out: Dict[str, Dict[str, Set[int]]] = {}
    for path in report_paths(reports_dir, case_id):
        out[path.name] = {c: set() for c in CATEGORIES}
    for category in CATEGORIES:
        for name, sentence, strong in finding_sentences(reports_dir, case_id, category):
            if strong:
                out[name][category] |= category_fdis(sentence, category)
    return out


def audit_agreement(reports_dir: Path, gt_table, unscored_table):
    """Which REPORT_GT findings does only SOME of a case's readers assert?

    Under the agreement rule those belong in UNSCORED -- neither credited nor
    penalised -- rather than in REPORT_GT, where a model that stays silent is
    marked wrong for agreeing with the other reader.

    CAVEAT, and it is why this prints sentences instead of rewriting the table:
    attribution comes from the keyword pass, which misses ~5% of findings. A
    tooth flagged "reader 1 only" may simply be one the keyword tables did not
    catch in reader 2's wording. Every flag needs the quoted sentences read.
    """
    rows = []
    for case_id, cats in sorted(gt_table.items()):
        reports = report_paths(reports_dir, case_id)
        if len(reports) < 2:
            continue                      # nothing to disagree about
        per = per_report_draft(reports_dir, case_id)
        names = sorted(per)
        for category, fdis in sorted(cats.items()):
            already = set(unscored_table.get(case_id, {}).get(category, []))
            for fdi in sorted(set(fdis) - already):
                asserting = [n for n in names if fdi in per[n][category]]
                if not asserting:
                    # No reader's wording matched -- the hand reader saw
                    # something the keyword tables do not model. Not a
                    # disagreement; leave it alone.
                    continue
                if len(asserting) < len(names):
                    # Distinguish SILENCE from DISAGREEMENT. A reader who names
                    # no tooth at all for this category is writing at a coarser
                    # granularity ("partial edentulism of the 3rd quadrant"),
                    # not contradicting a colleague who enumerates. Only a
                    # reader who does enumerate this category, and omits THIS
                    # tooth, is a candidate disagreement.
                    silent = [n for n in names
                              if n not in asserting and not per[n][category]]
                    kind = "silent" if silent else "differs"
                    rows.append((case_id, category, fdi, asserting, names, kind))
    return rows


def llm_draft_per_report(gt_dir: Path, case_id: str):
    """{gt file name: {category: {fdi: ...}}} from parse_reports_to_gt.py.

    That script writes ONE FILE PER RADIOLOGIST for a multi-report case
    ({case}_1_gt.json, {case}_2_gt.json) and a single {case}_gt.json otherwise.
    Keeping them separate rather than merging is deliberate: per-reader claims
    are what the agreement rule needs, so the LLM pass can corroborate or
    contradict the keyword pass reader by reader instead of only in aggregate.
    """
    out = {}
    single = gt_dir / f"{case_id}_gt.json"
    if single.exists():
        out[single.name] = pred_claims(json.loads(single.read_text(encoding="utf-8")))
    for path in sorted(gt_dir.glob(f"{case_id}_*_gt.json")):
        out[path.name] = pred_claims(json.loads(path.read_text(encoding="utf-8")))
    return out or None


def llm_draft(gt_dir: Path, case_id: str):
    """Union across readers -- the candidate set. None if no GT file exists."""
    per = llm_draft_per_report(gt_dir, case_id)
    if not per:
        return None
    merged = {c: {} for c in CATEGORIES}
    for claims in per.values():
        for category in CATEGORIES:
            merged[category].update(claims.get(category, {}))
    return merged


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", help="comma-separated case IDs (drafting mode)")
    ap.add_argument("--reports-dir", type=Path, default=Path("dataset/training/reports"))
    ap.add_argument("--gt-dir", type=Path, default=Path("outputs/ground_truth"),
                    help="where parse_reports_to_gt.py wrote {case}_gt.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the REPORT_GT block here")
    ap.add_argument("--audit-agreement", action="store_true",
                    help="audit the EXISTING REPORT_GT: which findings do only some "
                         "of a case's readers assert? Those belong in UNSCORED.")
    args = ap.parse_args()

    if args.audit_agreement:
        from survey_findings import REPORT_GT, UNSCORED
        rows = audit_agreement(args.reports_dir, REPORT_GT, UNSCORED)
        silent = [r for r in rows if r[5] == "silent"]
        differs = [r for r in rows if r[5] == "differs"]
        print(f"{len(rows)} REPORT_GT finding(s) not asserted by every reader:")
        print(f"  {len(silent):3}  the other reader names NO tooth in that category")
        print(f"       -> granularity, not disagreement. Reader writes regions")
        print(f"          ('partial edentulism of the 3rd quadrant') instead of")
        print(f"          teeth. Silence is not contradiction; leave in REPORT_GT.")
        print(f"  {len(differs):3}  the other reader DOES enumerate that category")
        print(f"       -> real disagreement candidates. These belong in UNSCORED.\n")
        by_case: Dict[str, List] = {}
        for case_id, category, fdi, asserting, names, kind in differs:
            by_case.setdefault(case_id, []).append((category, fdi, asserting, names))
        for case_id, items in sorted(by_case.items()):
            print(f"{case_id}  ({len(report_paths(args.reports_dir, case_id))} reports)")
            for category, fdi, asserting, names in items:
                missing = [n for n in names if n not in asserting]
                print(f"    {category:14} {fdi}   asserted by {asserting}, "
                      f"not by {missing}")
            moves = {}
            for category, fdi, _a, _n in items:
                moves.setdefault(category, []).append(fdi)
            entries = ", ".join(f'"{c}": {sorted(v)}' for c, v in sorted(moves.items()))
            print(f'    -> UNSCORED["{case_id}"] gains {{{entries}}}\n')
        print("Read the reports before applying: attribution comes from the keyword")
        print("pass, which misses ~5% of findings, so a 'reader 1 only' flag can be")
        print("a wording the tables did not catch rather than a real disagreement.")
        return

    if not args.cases:
        ap.error("--cases is required unless using --audit-agreement")
    case_ids = [c.strip() for c in args.cases.split(",")]
    block_lines, any_conflict = [], False

    for case_id in case_ids:
        paths = report_paths(args.reports_dir, case_id)
        if not paths:
            print(f"[FAIL] no report for {case_id} under {args.reports_dir}",
                  file=sys.stderr)
            continue

        kw, weak = keyword_draft(args.reports_dir, case_id)

        contested = contested_fdis(args.reports_dir, case_id)
        llm = llm_draft(args.gt_dir, case_id)
        llm_per = llm_draft_per_report(args.gt_dir, case_id) or {}
        kw_per = per_report_draft(args.reports_dir, case_id)

        print(f"\n{'=' * 78}\n{case_id}   ({', '.join(p.name for p in paths)})\n{'=' * 78}")
        if llm is None:
            print(f"  [note] no {case_id}_gt.json in {args.gt_dir} -- keyword draft only.")

        merged: Dict[str, List[int]] = {}
        for category in CATEGORIES:
            kw_fdis = set(kw[category])
            llm_fdis = set(llm[category]) if llm else set()

            # The union is the CANDIDATE set, deliberately. A keyword-only hit
            # is usually a sentence the LLM paraphrased away; an LLM-only hit
            # is usually an inference from wording the keyword tables do not
            # cover. Both are worth a human look; neither is auto-accepted.
            both = sorted(kw_fdis & llm_fdis)
            kw_only = sorted(kw_fdis - llm_fdis)
            llm_only = sorted(llm_fdis - kw_fdis)
            if llm is not None and (kw_only or llm_only):
                any_conflict = True

            merged[category] = sorted(kw_fdis | llm_fdis)
            if not merged[category] and not weak[category]:
                continue

            print(f"\n  {category}:")
            for sentence in weak[category][:3]:
                print(f"    weak/filtered  \"{sentence[:130]}\"")
                w = sorted(fdis_in(sentence) - set(merged[category]))
                if w:
                    print(f"                   would add {w} if you accept it")
            if not merged[category]:
                continue
            if both:
                print(f"    agreed        {both}")
            if kw_only:
                print(f"    keyword only  {kw_only}   <- check")
            if llm_only:
                print(f"    LLM only      {llm_only}   <- check")
            for fdi in merged[category]:
                # Who asserts this tooth, reader by reader and pass by pass.
                # A tooth only one reader supports is not ground truth under the
                # agreement rule -- it is UNSCORED -- so the attribution matters
                # as much as the tooth.
                kw_who = [n.replace(f"{case_id}_", "").replace(".txt", "")
                          for n in sorted(kw_per) if fdi in kw_per[n][category]]
                llm_who = [n.replace(f"{case_id}_", "").replace("_gt.json", "")
                           .replace("gt.json", "1")
                           for n in sorted(llm_per)
                           if fdi in llm_per[n].get(category, {})]
                print(f"      {fdi}: keyword={kw_who or '-'}  llm={llm_who or '-'}")
                # One sentence, two categories, same tooth. Not decided here --
                # a rule that guesses which one wins measured WORSE than
                # printing it (see contested_fdis); a human settles it in
                # seconds from the quoted sentence.
                also = [c for c in contested.get(fdi, {}) if c != category]
                if also:
                    print(f"          <- SAME SENTENCE also files {fdi} under "
                          f"{', '.join(sorted(also))}: are BOTH true of this "
                          f"tooth, or do the teeth split between them?")
                for sentence in kw[category].get(fdi, [])[:1]:
                    print(f"          \"{sentence[:130]}\"")
                if fdi not in kw[category]:
                    print(f"          (no report sentence matched -- LLM only, "
                          f"verify by hand)")

        entries = ", ".join(f'"{c}": {merged[c]}' for c in CATEGORIES if merged[c])
        block_lines.append(f'    "{case_id}": {{{entries}}},'
                           if entries else
                           f'    "{case_id}": {{}},'
                           f'                    # reports assert none of the five')

    print(f"\n{'=' * 78}")
    print("Paste into REPORT_GT in code/survey_findings.py, AFTER checking every")
    print("line above against the quoted sentence. An empty dict is a real claim:")
    print("it says the reports assert nothing, which is what turns a prediction")
    print("into a false positive.")
    print(f"{'=' * 78}")
    for line in block_lines:
        print(line)

    if any_conflict:
        print("\n[NOTE] The two drafts disagreed somewhere (marked '<- check').")
        print("       Those are the sentences to read closely.")

    if args.out:
        args.out.write_text("\n".join(block_lines) + "\n", encoding="utf-8")
        print(f"\nAlso written to {args.out}")


if __name__ == "__main__":
    main()
