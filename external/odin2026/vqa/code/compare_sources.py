#!/usr/bin/env python3
"""
code/compare_sources.py

Per-SOURCE accuracy against the generated ground truth: who is actually right
about each finding -- the segmentation masks, one of the three image reads, the
union of the reads, or the post-vote summary.

WHY THIS IS NOT survey_facts.py
───────────────────────────────
survey_facts.py answers two questions and neither is this one. Its CALL GROUPS
section scores per SCHEMA FIELD, so `[panoramic] dental_arch_findings_mandible.
findings` and `[detail] tooth_{fdi}_morphology.with_full_crown` appear as two
unrelated rows even though both are a claim about crowns; its OVERALL table
scores per FINDING but collapses every read into one PRED column, which is
exactly the collapse that hides which read earned it.

This module keeps the finding as the row and splits the sources into columns.
The join is already in the codebase: pred_claims() tags every claim with the
source that made it (`composite`, `panoramic`, `3d_render`) and pred_absent()
returns a `by_source` map, so a per-source score needs no new extraction and
cannot drift from what the pipeline consumes. The ground truth is read with
those same prediction extractors -- parse_reports_to_gt.py emits prediction
shape -- so a GT file and a prediction are the same kind of object.

THE FIVE SOURCES
────────────────
  facts      dataset/<split>/facts/<case>.json -- derived from the segmentation
             masks, not read out of an image by anything. The only source that
             is not a VLM read, and the only one whose errors are the mask's.
  3d         the three 3D renders: wisdom-tooth slots and the whole-jaw reads.
  panoramic  the arch survey -- all sixteen positions per arch, coarse vocabulary.
  detail     the per-tooth composite crops -- fine vocabulary, one tooth at a time.
  sinus      the two sinus crops. Speaks only for the sinus axes.
then two derived columns, which are not sources but stages:
  PRED       the union over every read that can assert the finding -- what
             survey_facts.py calls PRED, reproduced here as the reference point.
  SUMMARY    what survived postprocess_pred.py's cross-source vote, i.e. what
             can still reach the synthesized report.

WHY A CELL CAN BE BLANK, AND WHY THAT IS NOT A ZERO
───────────────────────────────────────────────────
Most sources cannot assert most findings, and "did not claim it" and "cannot
claim it" are different facts about a pipeline. A blank cell means the source
is structurally silent on that axis -- the composite has carried no impaction
fact since v6.3, the arch survey's v7.1 vocabulary cannot distinguish a crown
from a filling (both fold to `restoration`), the mask knows nothing about root
canals. Those are read out of SPEAKS below, one line per (axis, source) with
the reason, rather than inferred from an empty column: a source that claimed
nothing because it was wrong about everything would otherwise be printed as
though it had never been asked.

RECALL IS SCORED OVER WHAT THE SOURCE WAS SHOWN
───────────────────────────────────────────────
The sources do not cover the same positions, so scoring every one of them
against all GT positives would report a difference in COVERAGE as a difference
in ACCURACY. The 3D renders answer four wisdom slots; asked for recall over
every impacted tooth in the split they would score near zero for the sole
reason that nobody showed them the other twenty-eight positions. The composites
are expanded only for teeth the segmentation places, so their silence at a
missing tooth is build_vqa_pairs' doing rather than a read.

So each source carries a DOMAIN -- the positions it was actually asked about --
and its recall denominator is the GT positives inside that domain, printed as
`dom` beside it. Precision needs no such adjustment: a claim is either right or
wrong wherever it was made. `N` is the unrestricted GT count, so `dom` < `N` is
itself the readable statement that this source was never shown the rest.

Usage:
    python3 code/compare_sources.py outputs/aksssr_v7_trained_validate
    python3 code/compare_sources.py <run_dir> --split validate
    python3 code/compare_sources.py <run_dir> --case-ids A008 A019
    python3 code/compare_sources.py <run_dir> --no-write     # stdout only

Writes <run_dir>/survey/source_compare_<stamp>.{txt,json} unless --no-write,
under a timestamp, never overwriting a previous one -- the same convention as
survey_facts.py, and a different prefix from it because the two score different
questions.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from survey_findings import (  # noqa: E402
    ALL_TEETH,
    ARCH_TEETH,
    CATEGORIES,
    WISDOM_SLOT,
    _as_dict,
    as_list,
    composite_coverage,
    facts_absent,
    facts_prosthetics,
    pred_absent,
    pred_claims,
    pred_prosthetics,
    uses_legacy_shape,
    summary_absent,
    summary_claims,
    summary_prosthetics,
)
from survey_facts import (  # noqa: E402
    GLOBAL_AXES,
    GROUP_OF_IMAGE,
    SIDE_AXES,
    SIDES,
    SINGLE_AXES,
    _cast,
    _dig,
    global_axis,
    global_axis_summary,
    load_cases,
    score_sets,
)

# The source columns, in pipeline order: the mask first because it is the one
# witness that is not a read, then the three reads, then the two stages.
SOURCES = ("facts", "3d", "sinus", "panoramic", "detail", "pred", "summary")
READS = ("3d", "sinus", "panoramic", "detail")

# pred_claims/pred_absent tag a claim with the source that made it. Those tags
# predate this module; map them onto the column names rather than renaming them.
COLUMN_OF_TAG = {"composite": "detail", "panoramic": "panoramic",
                 "3d_render": "3d", "sinus": "sinus"}

# ---------------------------------------------------------------------------
# Which source can assert which axis, and why not where it cannot.
#
# Read as: axis -> {column: reason it CAN speak}. A column absent from an
# entry prints blank rather than 0.00, and the reason is what stops that blank
# from being re-derived by guesswork the next time the schema moves.
#
# `pred` and `summary` are omitted throughout -- they are stages, not sources,
# and are scored on every axis by construction.
# ---------------------------------------------------------------------------
SPEAKS: Dict[str, Dict[str, str]] = {
    "absent teeth": {
        "facts": "structured.teeth_absent, straight off the mask",
        "panoramic": "the arch survey answers `absent` as an explicit value",
        "3d": "the wisdom renders answer eruption_state `absent`",
        # not detail: build_vqa_pairs expands a composite only for teeth the
        # segmentation already places, so its silence is the pipeline's, not a read.
    },
    "impaction": {
        "3d": "the wisdom renders carry `impacted` + `orientation`",
        "panoramic": "the arch survey's `impacted` value",
        "detail": "tooth_{fdi}_eruption.eruption_state through the schema's own "
                  "_definitions.impacted rule -- AVAILABLE, NOT CONSUMED (see below)",
        # not facts: a mask places a tooth, it does not judge its eruption.
    },
    "endodontic": {
        "detail": "tooth_{fdi}_morphology.with_endo",
        "panoramic": "the arch survey's `root_canal_treatment` value",
        # not facts: canal filling material is not segmented.
    },
    "post_and_core": {
        "detail": "tooth_{fdi}_morphology.with_post_and_core",
        # not panoramic: v7.1 files a posted tooth under root_canal_treatment --
        # a post and a canal filling are one bright canal at that resolution.
        # not facts: not segmented.
    },
    "crown": {
        "facts": "structured.crowns, mask-derived",
        "detail": "tooth_{fdi}_morphology.with_full_crown",
        # not panoramic: v7.1 folds a crown into `restoration`, which cannot be
        # told from a large filling at arch resolution. It votes on `restoration`.
    },
    "fillings": {
        "detail": "tooth_{fdi}_morphology.with_fillings",
        # not panoramic under v7.1: the arch vocabulary has no filling value at
        # all -- a crown and a large filling are one bright capped tooth at that
        # resolution, so both fold to `restoration` and are scored on that row.
        # A v6.x run DID have one; see LEGACY_SPEAKS.
    },
    "restoration": {
        "facts": "a mask-derived crown IS a restoration at this granularity",
        "detail": "the composite's second vote: a crown or filling is a restoration",
        "panoramic": "the arch survey's `restoration` value",
    },
    "implants": {
        "facts": "structured.implants, mask-derived",
        "panoramic": "implants_{arch}.implants",
    },
    "fixed bridges": {
        "facts": "structured.bridge_present",
        "panoramic": "fixed_bridges_{arch}.present",
    },
    "canal-adjacent teeth": {
        "facts": "structured.ian_close_teeth, measured on the mask",
        "3d": "mandible_canal_{side}.adjacent_teeth",
        "detail": "tooth_{fdi}_mandible_canal.adjacent_to_teeth, asked at 36-38/46-48",
    },
    "condyle scope": {
        "facts": "structured.fov.condyles, mapped excluded -> not_included",
        "3d": "mandible_condyle_{side}.scope",
    },
    "maxilla scope": {
        "facts": "structured.fov.maxilla, mapped partial -> included",
        "3d": "maxilla_scope.maxilla_included",
    },
}

# Capability that a v6.x run HAS and a v7.1 run does not. ARCH_VALUE still maps
# the retired `filling` and `post_and_core` arch values, so an old prediction
# carries panoramic claims on two axes where v7.1 is structurally silent -- and
# a blank column on a run that plainly has such claims would be the same lie in
# the other direction. Vintage is read with survey_findings.uses_legacy_shape,
# the codebase's existing test, rather than by sniffing for the values.
LEGACY_SPEAKS: Dict[str, Dict[str, str]] = {
    "fillings": {"panoramic": "the pre-v7.1 arch value `filling` (v6.x run)"},
    "post_and_core": {"panoramic": "the pre-v7.1 arch value `post_and_core` "
                                   "(v6.x run)"},
}

# Facts-file enums do not use the schema's words. Map them once, here, so the
# mapping is visible rather than buried in a comparison. A facts file that omits
# `fov` entirely (5 of the 40 validate cases) is UNANSWERED, not a negative.
FACTS_SCOPE_MAP = {
    "condyle scope": {"excluded": "not_included", "included": "fully_included",
                      "partial": "partially_included"},
    "maxilla scope": {"excluded": "not_included", "included": "included",
                      "partial": "included"},
}

# The set axes, in the order the table prints them: (label, unit, section).
SET_AXES = [
    ("absent teeth", "teeth", "teeth"),
    ("impaction", "teeth", "findings"),
    ("endodontic", "teeth", "findings"),
    ("post_and_core", "teeth", "findings"),
    ("crown", "teeth", "findings"),
    ("fillings", "teeth", "findings"),
    ("restoration", "teeth", "findings"),
    ("implants", "teeth", "prosthetics"),
    ("canal-adjacent teeth", "teeth", "anatomy"),
]

# ---------------------------------------------------------------------------
# Axes where a claim at a position the GT never ANSWERED is not a false
# positive, and both sides are cut down to the positions the GT settles.
#
# Absence is the case where this is not a refinement but the whole result. No
# reference report enumerates all 32 positions -- the generated GT answers a
# mean of 16.9 of them per case -- so a position no report mentions is
# UNANSWERED, not "present". Charged over all 32, every source loses about 0.25
# precision to the label's silence, and it lands hardest on whichever source
# claims most, which inverts the comparison the axis exists to make: the mask
# enumerates every position and is therefore penalised for being complete.
# survey_findings.py meets the same wall and answers it the other way, by
# scoring absence against the mask instead of the reports; that is not open to
# this module, whose entire question is whether the mask is right.
#
# Deliberately NOT applied to the finding axes. There the GT's silence at a
# tooth is a weaker claim -- an arch read that came back `unremarkable` IS an
# answer of "no finding there" -- so the restriction would discard true
# negatives along with the unanswered ones. Those rows carry the unrestricted
# penalty, which depresses precision but does so evenly across sources.
# ---------------------------------------------------------------------------
GT_ANSWERED_ONLY = {"absent teeth"}

# ---------------------------------------------------------------------------
# One column is DERIVED here rather than read from pred_claims: the composite's
# impaction claim.
#
# Everywhere else this module deliberately reuses pred_claims' source tags so
# it cannot report a capability the pipeline does not consume. Impaction is the
# one place where that rule would print a falsehood: the composite carries
# tooth_{fdi}_eruption.eruption_state, which IS half of the schema's own
# _definitions.impacted, so "the composite cannot speak about impaction" is
# untrue -- what is true is that postprocess_pred.py does not currently ask it.
# Blanking the cell would hide an available source; filling it silently would
# imply the pipeline uses it. So it is computed, and its reason line says
# AVAILABLE, NOT CONSUMED.
#
# Only the eruption half is used. `orientation` is the other half of the rule
# and the composite has no such field, so this column is a floor on what the
# composite could contribute, not the full derived rule -- which needs the 3D
# fact's orientation and is scored in "THE RULE -- impaction" instead.
# ---------------------------------------------------------------------------
UNERUPTED = ("complete_bony_inclusion", "partially_erupted")

MANDIBULAR_POSTERIOR = {36, 37, 38, 46, 47, 48}


# ── which call group owns a global fact ─────────────────────────────────────

def group_of_fact(schema_path: Path) -> Dict[str, str]:
    """{output_field: call group} from the schema's own images_needed.

    A global fact is answered from exactly one image set, so "which source said
    this" is not a question for the enum axes -- it is a lookup, and it belongs
    to the schema rather than to a table here.
    """
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for section in ("mandible", "maxilla", "dental_elements"):
        for fact in schema.get(section) or []:
            groups = {GROUP_OF_IMAGE.get(i)
                      for i in (fact.get("images_needed") or [])} - {None}
            if len(groups) == 1:
                out[fact.get("output_field") or "?"] = groups.pop()
    return out


# ── the facts file as a source ──────────────────────────────────────────────

def facts_claims(facts: Dict) -> Dict[str, Set[int]]:
    """{axis: teeth} for every axis a mask-derived facts file can assert."""
    structured = _as_dict(facts.get("structured"))
    absent, _enumerated = facts_absent(facts)
    implants, _bridge = facts_prosthetics(facts)
    crowns = {f for f in as_list(structured.get("crowns")) if isinstance(f, int)}
    return {
        "absent teeth": absent,
        "crown": crowns & ALL_TEETH,
        # A crown is the only restoration a mask carries, so it is also the whole
        # of what this source can say on the arch survey's coarser row.
        "restoration": crowns & ALL_TEETH,
        "implants": implants & ALL_TEETH,
        "canal-adjacent teeth": {f for f in as_list(structured.get("ian_close_teeth"))
                                 if isinstance(f, int)},
    }


def facts_scope(facts: Dict, axis: str) -> Optional[str]:
    """The facts file's answer to a scope enum, in the schema's vocabulary."""
    fov = _as_dict(_as_dict(facts.get("structured")).get("fov"))
    key = "condyles" if axis == "condyle scope" else "maxilla"
    raw = fov.get(key)
    if not isinstance(raw, str):
        return None                       # no fov block: unanswered
    return FACTS_SCOPE_MAP[axis].get(raw.strip().lower())


# ── per-source claim extraction for one case ────────────────────────────────

def by_source(case: Dict) -> Tuple[Dict[str, Dict[str, Set[int]]],
                                   Dict[str, Dict[str, Set[int]]]]:
    """(claims, domains) as {column: {axis: teeth}} for one case.

    A domain is what that source was asked about; see the module docstring.
    An axis missing from a source's domain means "everything", handled by the
    caller rather than by materialising all 32 positions per axis per case.
    """
    pred, summ = case["pred"], case["summary"]
    facts = case.get("facts") or {}

    claims: Dict[str, Dict[str, Set[int]]] = {s: {} for s in SOURCES}
    domains: Dict[str, Dict[str, Set[int]]] = {s: {} for s in SOURCES}

    # -- the reads, straight off the source tags the extractors already carry
    tagged = pred_claims(pred)
    for category in CATEGORIES:
        for fdi, srcs in tagged.get(category, {}).items():
            for tag in srcs:
                column = COLUMN_OF_TAG.get(tag)
                if column:
                    claims[column].setdefault(category, set()).add(fdi)
        claims["pred"].setdefault(category, set()).update(tagged.get(category, {}))

    absent = pred_absent(pred)
    for tag, teeth in absent["by_source"].items():
        claims[COLUMN_OF_TAG[tag]]["absent teeth"] = set(teeth)
    claims["pred"]["absent teeth"] = absent["absent"]

    p_implants, p_bridge = pred_prosthetics(pred)
    claims["pred"]["implants"] = p_implants
    claims["panoramic"]["implants"] = p_implants     # implants_{arch} is a panoramic fact

    # canal-adjacent teeth: three independent readers of the same relation.
    canal_3d: Set[int] = set()
    for side in SIDES:
        canal_3d |= {f for f in as_list(_as_dict(_as_dict(pred.get("global"))
                                                 .get(f"mandible_canal_{side}"))
                                        .get("adjacent_teeth"))
                     if isinstance(f, int)}
    canal_detail: Set[int] = set()
    canal_detail_domain: Set[int] = set()
    for key, tooth in _as_dict(pred.get("teeth")).items():
        fdi = int(key.split("_")[1])
        block = _as_dict(tooth).get(f"tooth_{fdi}_mandible_canal")
        if not isinstance(block, dict):
            continue
        canal_detail_domain.add(fdi)
        near = block.get("adjacent_to_teeth", block.get("close_to_canal"))
        if near is True or str(near).lower() == "true":
            canal_detail.add(fdi)
        canal_detail |= {f for f in as_list(near) if isinstance(f, int)}
    # The composite's impaction claim, derived -- see UNERUPTED above.
    imp_detail: Set[int] = set()
    imp_detail_domain: Set[int] = set()
    for fdi in WISDOM_SLOT.values():
        state = _as_dict(_as_dict(_as_dict(pred.get("teeth")).get(f"tooth_{fdi}"))
                         .get(f"tooth_{fdi}_eruption")).get("eruption_state")
        if not isinstance(state, str):
            continue
        state = state.strip().lower()
        if state == "absent":
            continue                      # absent is explicitly not a claim
        imp_detail_domain.add(fdi)
        if state in UNERUPTED:
            imp_detail.add(fdi)
    claims["detail"]["impaction"] = imp_detail
    domains["detail"]["impaction"] = imp_detail_domain
    # PRED stays the union of what the PIPELINE reads, so the derived column is
    # deliberately not folded into it -- that is the difference the row shows.

    claims["3d"]["canal-adjacent teeth"] = canal_3d
    claims["detail"]["canal-adjacent teeth"] = canal_detail
    claims["pred"]["canal-adjacent teeth"] = canal_3d | canal_detail

    # -- the mask
    if facts:
        for axis, teeth in facts_claims(facts).items():
            claims["facts"][axis] = teeth

    # -- the post-vote stage
    s_claims = summary_claims(summ) if summ else {}
    for category in CATEGORIES:
        claims["summary"][category] = set(s_claims.get(category, {}))
    claims["summary"]["absent teeth"] = (summary_absent(summ)["absent"]
                                         if summ else set())
    s_implants, _s_bridge = summary_prosthetics(summ) if summ else (set(), False)
    claims["summary"]["implants"] = s_implants
    s_canal: Set[int] = set()
    canals = _as_dict(_as_dict(summ.get("mandible")).get("canals")) if summ else {}
    for side in SIDES:
        s_canal |= {f for f in as_list(_as_dict(canals.get(side)).get("adjacent_teeth"))
                    if isinstance(f, int)}
    claims["summary"]["canal-adjacent teeth"] = s_canal

    # -- domains: only where a source was genuinely shown less than the whole
    #    dentition. Everything else is unrestricted and left absent.
    wisdom = set(WISDOM_SLOT.values())
    domains["3d"]["absent teeth"] = wisdom
    domains["3d"]["impaction"] = wisdom
    coverage = composite_coverage(pred)
    for category in ("endodontic", "post_and_core", "crown", "fillings",
                     "restoration"):
        domains["detail"][category] = coverage.get(category, set())
    domains["detail"]["canal-adjacent teeth"] = canal_detail_domain
    # The arch survey answers a position only if it came back as a string.
    arch_read: Set[int] = set()
    for arch, positions in ARCH_TEETH.items():
        findings = _as_dict(_as_dict(_as_dict(pred.get("global"))
                                     .get(f"dental_arch_findings_{arch}")).get("findings"))
        for fdi in positions:
            if isinstance(findings.get(str(fdi), findings.get(fdi)), str):
                arch_read.add(fdi)
    for category in ("absent teeth", "impaction", "endodontic", "fillings",
                     "restoration"):
        domains["panoramic"][category] = arch_read
    # The mask enumerates only the positions it placed or explicitly called absent.
    if facts:
        _absent, enumerated = facts_absent(facts)
        for axis in ("absent teeth", "crown", "restoration", "implants"):
            domains["facts"][axis] = enumerated
    return claims, domains


# ── scoring ─────────────────────────────────────────────────────────────────

def score_axis(cases: List[Dict], axis: str,
               claims_all: List[Tuple[Dict, Dict]],
               reasons: Dict[str, str]) -> Dict[str, Counter]:
    """{column: Counter(tp, fp, fn, dom, n)} for one set-valued axis.

    `reasons` is the row's capability set, and is the SAME object the renderer
    decides blankness from -- scoring one set of columns and printing another
    is how a column ends up reading "claimed nothing" when it was never scored.
    """
    stats = {s: Counter() for s in SOURCES}
    for case, (claims, domains) in zip(cases, claims_all):
        truth = case["truth"].get(axis, set())
        answered = case["gt_answered"] if axis in GT_ANSWERED_ONLY else None
        if answered is not None:
            truth = truth & answered
        for column in SOURCES:
            if column not in ("pred", "summary") and column not in reasons:
                continue
            if column == "facts" and not case.get("facts"):
                continue
            claimed = claims[column].get(axis, set())
            if answered is not None:
                claimed = claimed & answered
            domain = domains[column].get(axis)
            scoped_truth = truth if domain is None else truth & domain
            # A claim outside the domain is still a claim, and still right or
            # wrong on its own terms -- precision is never domain-restricted.
            # Only the recall denominator shrinks to what the source was shown.
            stats[column]["tp"] += len(claimed & truth)
            stats[column]["fp"] += len(claimed - truth)
            stats[column]["fn"] += len(scoped_truth - claimed)
            stats[column]["dom"] += len(scoped_truth)
    for column in SOURCES:
        stats[column]["n"] = sum(len(c["truth"].get(axis, set())) for c in cases)
    return stats


def score_bridges(cases: List[Dict]) -> Dict[str, Counter]:
    """Fixed bridges are one boolean per case, not a tooth set."""
    stats = {s: Counter() for s in SOURCES}
    for case in cases:
        truth = case["truth_bridge"]
        p_i, p_b = pred_prosthetics(case["pred"])
        _s_i, s_b = (summary_prosthetics(case["summary"])
                     if case["summary"] else (set(), False))
        answers = {"pred": p_b, "panoramic": p_b, "summary": s_b}
        if case.get("facts"):
            answers["facts"] = facts_prosthetics(case["facts"])[1]
        for column, claim in answers.items():
            stats[column]["tp" if (claim and truth) else
                          "fp" if claim else "fn" if truth else "tn"] += 1
            stats[column]["dom"] += int(truth)
        for column in SOURCES:
            stats[column]["n"] += int(truth)
    return stats


def owned_by(label: str, group: str, tmpl: str) -> Dict[str, str]:
    """Which columns speak on an enum axis, and why.

    A global fact is answered from one image set, so its owning read is a
    schema lookup rather than a judgement -- SPEAKS only has to carry the
    exceptions, which are the axes the MASK can also answer.
    """
    reasons = dict(SPEAKS.get(label, {}))
    if group and group not in reasons:
        reasons[group] = f"the schema points {tmpl} at this call group alone"
    return reasons


def score_enum_axes(cases: List[Dict], owner: Dict[str, str]) -> List[Dict]:
    """The enum / bool / arch-level axes: one owning read, plus the stages."""
    rows = []

    def blank() -> Dict[str, Counter]:
        return {s: Counter() for s in SOURCES}

    # per-side and single-fact axes
    for label, tmpl, field, path, kind in SIDE_AXES + SINGLE_AXES:
        if label in {a for a, _u, _s in SET_AXES}:
            # canal-adjacent teeth is scored above as a THREE-source axis --
            # the mask, the 3D canal trace and the composite's own canal fact
            # all name the same teeth. Scoring it again here, per side against
            # a per-side truth, would print two rows that disagree because
            # they are two different questions.
            continue
        stats = blank()
        group = owner.get(tmpl.format(side="right"), owner.get(tmpl, ""))
        units = SIDES if "{side}" in tmpl else ("",)
        for case in cases:
            for side in units:
                fact = tmpl.format(side=side)
                truth = _as_dict(_as_dict(case["gt"].get("global")).get(fact)).get(field)
                if truth is None:
                    continue
                n = len(_cast(truth, kind)) if kind == "set" else 1
                for column in SOURCES:
                    stats[column]["n"] += n
                read = _as_dict(_as_dict(case["pred"].get("global")).get(fact)).get(field)
                got = {"pred": read, "summary": _dig(case["summary"] or {}, path, side)}
                if group:
                    got[group] = read
                if label in FACTS_SCOPE_MAP and case.get("facts"):
                    got["facts"] = facts_scope(case["facts"], label)
                for column, value in got.items():
                    if value is None:
                        continue
                    if kind == "set":
                        stats[column] += score_sets(_cast(value, kind),
                                                    _cast(truth, kind))
                        stats[column]["dom"] += len(_cast(truth, kind))
                    else:
                        stats[column]["scored"] += 1
                        stats[column]["right"] += int(_cast(value, kind)
                                                      == _cast(truth, kind))
                        stats[column]["dom"] += 1
        rows.append({"section": "anatomy", "axis": label,
                     "unit": "teeth" if kind == "set" else
                             ("sides" if "{side}" in tmpl else "cases"),
                     "metric": "prec/rec" if kind == "set" else "acc",
                     "reasons": owned_by(label, group, tmpl),
                     "stats": stats})

    # per-arch axes
    for label, tmpl, field, path, kind, default in GLOBAL_AXES:
        stats = blank()
        group = owner.get(tmpl.format(arch="mandible"), "")
        for case in cases:
            truth = global_axis(case["gt"], tmpl, field, kind)
            read = global_axis(case["pred"], tmpl, field, kind)
            summ = global_axis_summary(case["summary"] or {}, path, kind, default)
            n = (sum(len(v) for v in truth.values()) if kind == "set"
                 else len(truth))
            for column in SOURCES:
                stats[column]["n"] += n
            got = {"pred": read, "summary": summ}
            if group:
                got[group] = read
            for column, values in got.items():
                for arch, want in truth.items():
                    if arch not in values:
                        continue
                    if kind == "set":
                        stats[column] += score_sets(values[arch], want)
                        stats[column]["dom"] += len(want)
                    else:
                        stats[column]["scored"] += 1
                        stats[column]["right"] += int(values[arch] == want)
                        stats[column]["dom"] += 1
        rows.append({"section": "global", "axis": label,
                     "unit": "teeth" if kind == "set" else "arches",
                     "metric": "prec/rec" if kind == "set" else "acc",
                     "reasons": owned_by(label, group, tmpl),
                     "stats": stats})
    return rows


# ── rendering ───────────────────────────────────────────────────────────────

def cell(stat: Counter, metric: str, speaks: bool) -> str:
    if not speaks:
        return "   .   "
    if metric == "acc":
        if not stat.get("scored"):
            return "   -   "
        return f" {stat['right'] / stat['scored']:.2f}  "
    tp, fp, fn = stat["tp"], stat["fp"], stat["fn"]
    if not (tp + fp + fn):
        return "   -   "
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return f"{p:.2f}/{r:.2f}"


def f1(stat: Counter, metric: str) -> Optional[float]:
    if metric == "acc":
        return stat["right"] / stat["scored"] if stat.get("scored") else None
    tp, fp, fn = stat["tp"], stat["fp"], stat["fn"]
    if not (tp + fp + fn):
        return None
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def render(rows: List[Dict], cases: List[Dict], run_dir: Path) -> str:
    out: List[str] = []
    out.append(f"Source comparison: {run_dir}")
    out.append("")
    out.append(f"{len(cases)} cases, reference = first *_gt.json per case "
               f"(generated from the reference reports)")
    out.append("")
    header = (f"{'section':11} {'axis':22} {'N':>4} {'unit':6} {'metric':8} " +
              " ".join(f"{s.upper():^7}" for s in SOURCES))
    out.append("=" * len(header))
    out.append("PER-SOURCE ACCURACY -- one row per finding, one column per source")
    out.append("=" * len(header))
    out.append(header)
    out.append("-" * len(header))
    section = None
    for row in rows:
        if row["section"] != section:
            section = row["section"]
        stats, metric, axis = row["stats"], row["metric"], row["axis"]
        speaks = row["reasons"]
        cells = []
        for column in SOURCES:
            can = column in ("pred", "summary") or column in speaks
            cells.append(cell(stats[column], metric, can))
        n = stats["pred"]["n"]
        out.append(f"{row['section']:11} {axis:22} {n:>4} {row['unit']:6} "
                   f"{metric:8} " + " ".join(cells))
    out.append("")
    out.append("prec/rec rows: precision over what that source CLAIMED, recall over")
    out.append("the GT positives inside its DOMAIN (see per-axis detail below).")
    out.append("acc rows: right/scored over the units that source answered.")
    out.append("  .  = the source cannot assert this axis at all -- see WHY, below.")
    out.append("  -  = it can, and claimed nothing here.")
    out.append("PRED = the union over every read. SUMMARY = after the cross-source vote.")
    out.append("")

    out.append("=" * len(header))
    out.append("PER-AXIS DETAIL -- claims, hits and coverage behind each cell")
    out.append("=" * len(header))
    for row in rows:
        stats, metric, axis = row["stats"], row["metric"], row["axis"]
        speaks = row["reasons"]
        n = stats["pred"]["n"]
        out.append("")
        out.append(f"[{row['section']}] {axis}   GT asserts {n} {row['unit']}")
        if metric == "acc":
            out.append(f"  {'source':10} {'answered':>8} {'right':>6} {'acc':>6}")
            for column in SOURCES:
                if not (column in ("pred", "summary") or column in speaks):
                    continue
                s = stats[column]
                acc = f"{s['right'] / s['scored']:.2f}" if s.get("scored") else "  -  "
                out.append(f"  {column:10} {s.get('scored', 0):>8} "
                           f"{s.get('right', 0):>6} {acc:>6}")
        else:
            out.append(f"  {'source':10} {'dom':>5} {'claims':>7} {'tp':>4} "
                       f"{'fp':>4} {'fn':>4} {'prec':>6} {'rec':>6} {'f1':>6}")
            for column in SOURCES:
                if not (column in ("pred", "summary") or column in speaks):
                    continue
                s = stats[column]
                tp, fp, fn = s["tp"], s["fp"], s["fn"]
                p = tp / (tp + fp) if (tp + fp) else 0.0
                r = tp / (tp + fn) if (tp + fn) else 0.0
                score = f1(s, metric)
                out.append(f"  {column:10} {s['dom']:>5} {tp + fp:>7} {tp:>4} "
                           f"{fp:>4} {fn:>4} {p:>6.2f} {r:>6.2f} "
                           + ("     -" if score is None else f"{score:>6.2f}"))
        for column in SOURCES:
            if column in speaks:
                continue
            if column in ("pred", "summary"):
                continue
            out.append(f"  {column:10} .   silent: not a source for this axis")
    out.append("")

    out.append("=" * len(header))
    out.append("WHY A SOURCE SPEAKS -- from SPEAKS and the schema's images_needed")
    out.append("=" * len(header))
    for row in rows:
        speaks = row["reasons"]
        if not speaks:
            continue
        out.append(f"[{row['axis']}]")
        for column, reason in speaks.items():
            out.append(f"  {column:10} {reason}")
    return "\n".join(out)


# ── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    project = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--split", default=None,
                    help="validate|training (default: inferred from the run dir name)")
    ap.add_argument("--gt-dir", type=Path, default=None)
    ap.add_argument("--facts-dir", type=Path, default=None)
    ap.add_argument("--schema", type=Path, default=project / "schema" / "schema.json")
    ap.add_argument("--case-ids", nargs="*", default=None)
    ap.add_argument("--no-write", action="store_true",
                    help="print to stdout only, write no survey files")
    args = ap.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else project / args.run_dir
    split = args.split
    if not split:
        name = run_dir.name
        split = "validate" if "validate" in name else "training" if "train" in name else None
    if not split:
        print(f"[FAIL] cannot infer split from {run_dir.name} -- pass --split",
              file=sys.stderr)
        sys.exit(1)
    gt_dir = args.gt_dir or project / "dataset" / split / "outputs" / "ground_truth"
    facts_dir = args.facts_dir or project / "dataset" / split / "facts"

    cases = load_cases(run_dir, gt_dir, args.case_ids)
    if not cases:
        print(f"[FAIL] no scoreable cases under {run_dir}", file=sys.stderr)
        sys.exit(1)

    missing_facts = 0
    for case in cases:
        path = facts_dir / f"{case['case_id']}.json"
        if path.exists():
            case["facts"] = json.loads(path.read_text(encoding="utf-8"))
        else:
            case["facts"] = {}
            missing_facts += 1
        # The ground truth is read with the PREDICTION extractors -- that is the
        # payoff of parse_reports_to_gt.py emitting prediction shape.
        gt_claims = pred_claims(case["gt"])
        truth = {c: set(gt_claims.get(c, {})) for c in CATEGORIES}
        gt_a = pred_absent(case["gt"])
        truth["absent teeth"] = gt_a["absent"]
        # The positions this case's GT settles either way -- see GT_ANSWERED_ONLY.
        case["gt_answered"] = gt_a["read"]
        gt_i, gt_b = pred_prosthetics(case["gt"])
        truth["implants"] = gt_i
        gt_canal: Set[int] = set()
        for side in SIDES:
            gt_canal |= {f for f in as_list(
                _as_dict(_as_dict(case["gt"].get("global"))
                         .get(f"mandible_canal_{side}")).get("adjacent_teeth"))
                if isinstance(f, int)}
        truth["canal-adjacent teeth"] = gt_canal
        case["truth"], case["truth_bridge"] = truth, gt_b
    if missing_facts:
        print(f"[WARN] {missing_facts} case(s) have no facts file under {facts_dir} "
              f"-- the facts column is scored over the rest", file=sys.stderr)

    claims_all = [by_source(case) for case in cases]
    owner = group_of_fact(args.schema)

    rows: List[Dict] = []
    legacy = any(uses_legacy_shape(c["pred"]) for c in cases)
    if legacy:
        print("[INFO] a v6.x-shaped prediction is present -- the panoramic column "
              "speaks on the two retired arch values too", file=sys.stderr)
    for axis, unit, section in SET_AXES:
        reasons = dict(SPEAKS.get(axis, {}))
        if legacy:
            reasons.update(LEGACY_SPEAKS.get(axis, {}))
        rows.append({"section": section, "axis": axis, "unit": unit,
                     "metric": "prec/rec", "reasons": reasons,
                     "stats": score_axis(cases, axis, claims_all, reasons)})
    rows.append({"section": "prosthetics", "axis": "fixed bridges", "unit": "cases",
                 "metric": "prec/rec",
                 "reasons": dict(SPEAKS["fixed bridges"]),
                 "stats": score_bridges(cases)})
    rows.extend(score_enum_axes(cases, owner))

    text = render(rows, cases, run_dir)
    print(text)

    if args.no_write:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    survey_dir = run_dir / "survey"
    survey_dir.mkdir(parents=True, exist_ok=True)
    (survey_dir / f"source_compare_{stamp}.txt").write_text(text + "\n",
                                                            encoding="utf-8")
    payload = {
        "run_dir": str(run_dir),
        "split": split,
        "cases": [c["case_id"] for c in cases],
        "sources": list(SOURCES),
        "rows": [{"section": r["section"], "axis": r["axis"], "unit": r["unit"],
                  "metric": r["metric"],
                  "n": r["stats"]["pred"]["n"],
                  "speaks": sorted(set(r["reasons"]) | {"pred", "summary"}),
                  "stats": {s: dict(r["stats"][s]) for s in SOURCES},
                  "score": {s: f1(r["stats"][s], r["metric"]) for s in SOURCES}}
                 for r in rows],
    }
    (survey_dir / f"source_compare_{stamp}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[PASS] -> {survey_dir / f'source_compare_{stamp}.txt'}")
    print(f"[PASS] -> {survey_dir / f'source_compare_{stamp}.json'}")


if __name__ == "__main__":
    main()
