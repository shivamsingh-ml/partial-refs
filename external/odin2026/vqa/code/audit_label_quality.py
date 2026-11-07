#!/usr/bin/env python3
"""
audit_label_quality.py -- is this case's *label* good enough to teach from?

Promoted out of the scratch scripts described in fewshot_probe_plan.md Step 3b.
An exemplar is only as good as its label, and in this dataset the reports
frequently contradict the segmentation. Three independent screens, cheapest
first, and every one of them is text-or-mask only -- no GPU, no model:

  --screen report-vs-mask   the report says a tooth is impacted (a PRESENT
                            tooth) that the mask does not contain, or calls a
                            tooth absent that the mask has. IMPOSSIBLE labels.
  --screen mirror           the report's absent-tooth list fits the mask better
                            MIRRORED than as written -- a laterality-flipped
                            report. Decided on the FULL absent list, not on the
                            wisdom teeth: when a report calls both 38 and 48
                            impacted, mirroring that pair is a no-op and the
                            flip is invisible (this is how P205 passed the
                            first version of the screen).
  --screen wisdom-geometry  measure the wisdom teeth in the MASK -- how deep
                            the crown sits relative to the local alveolar crest
                            and how far its long axis is tipped off its
                            neighbour's -- and compare that with what the
                            report claims. This is the only screen that can
                            catch a report whose words are unrenderable:
                            "mucosal inclusion" is invisible in a bone+tooth
                            segmentation, so an exemplar copying it teaches the
                            student to assert what the picture cannot show.

Sources it reads, all of which the pipeline already produces:

  dataset/<split>/facts/<case>.json                       segmentation-derived
  dataset/<split>/masks/<case>.nii.gz                     (wisdom-geometry only)
  dataset/<split>/outputs/ground_truth/<case>[_<r>]_gt.json  prediction-shaped GT
  outputs/gt_quality_<split>/gt_quality_*.json            per-case ranking

Usage:

    python3 code/audit_label_quality.py --cases P493,P543,P342 --split training
    python3 code/audit_label_quality.py --clean-set --screen report-vs-mask,mirror
    python3 code/audit_label_quality.py --cases P493 --screen wisdom-geometry

Exit status is 1 if any requested case FAILS a screen, so it can gate a render.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

ARCH_FDI = {
    "mandible": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38],
    "maxilla":  [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
}
WISDOM = {48: "lower_right_wisdom_tooth", 38: "lower_left_wisdom_tooth",
          18: "upper_right_wisdom_tooth", 28: "upper_left_wisdom_tooth"}
# the second molar each wisdom tooth is measured against
NEIGHBOUR = {48: 47, 38: 37, 18: 17, 28: 27}
MANDIBLE_LABEL = 1
IMPACTION_WORDS = re.compile(r"impact|inclus|disodon", re.I)


def mirror_fdi(fdi: int) -> int:
    """21 <-> 11, 48 <-> 38 ... swap the quadrant within the same jaw."""
    q, t = divmod(fdi, 10)
    return {1: 2, 2: 1, 3: 4, 4: 3}[q] * 10 + t


# ── inputs ────────────────────────────────────────────────────────────────

def load_facts(root: Path, split: str, case: str) -> dict:
    return json.loads((root / "dataset" / split / "facts" / f"{case}.json").read_text())


def load_readers(root: Path, split: str, case: str) -> dict:
    """{reader: prediction-shaped GT}. Per-reader files when they exist."""
    gt_dir = root / "dataset" / split / "outputs" / "ground_truth"
    out = {}
    for p in sorted(gt_dir.glob(f"{case}_*_gt.json")):
        m = re.fullmatch(rf"{case}_(\d+)_gt\.json", p.name)
        if m:
            out[m.group(1)] = json.loads(p.read_text())
    merged = gt_dir / f"{case}_gt.json"
    if not out and merged.exists():
        out["merged"] = json.loads(merged.read_text())
    return out


def load_reports(root: Path, split: str, case: str) -> dict:
    d = root / "dataset" / split / "reports"
    return {p.name: p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(d.glob(f"{case}_*.txt"))}


def load_ranking(root: Path, split: str) -> dict:
    files = sorted((root / "outputs" / f"gt_quality_{split}").glob("gt_quality_*.json"))
    if not files:
        return {}
    q = json.loads(files[-1].read_text())
    return {c["case_id"]: c for c in q.get("cases", [])}


# ── screen 1: report vs mask ──────────────────────────────────────────────

def screen_report_vs_mask(case, facts, readers, **_):
    """
    IMPOSSIBLE labels: a claim the rendered image cannot show.

    Problems are tagged with the arch they are in, and that matters when
    picking exemplars: a case whose maxilla is out of the acquisition volume
    routinely comes back with a whole maxillary arch of phantom "normal"
    values, because the extraction fills what the report never discussed. That
    disqualifies the case for pano_tooth_findings_maxilla and says nothing at
    all about its mandible calls.
    """
    present = {int(t) for t in facts["structured"].get("teeth_present", [])}
    problems = []
    for reader, gt in readers.items():
        g = gt.get("global") or {}
        for fdi, fact in WISDOM.items():
            v = g.get(fact) or {}
            state = v.get("eruption_state")
            if v.get("impacted") and fdi not in present:
                problems.append(f"[{reader}] {fdi} called impacted, mask has no {fdi} "
                                f"-- an impacted tooth is a PRESENT tooth")
            if state not in (None, "absent") and fdi not in present:
                problems.append(f"[{reader}] {fdi} called {state}, mask has no {fdi}")
            if state == "absent" and fdi in present:
                problems.append(f"[{reader}] {fdi} called absent, mask HAS {fdi} "
                                f"-- report silence is not absence")
        for arch, fdis in ARCH_FDI.items():
            f = ((g.get(f"dental_arch_findings_{arch}") or {}).get("findings") or {})
            for k, val in f.items():
                fdi = int(k)
                if val == "absent" and fdi in present:
                    problems.append(f"[{reader}] {arch} arch says {fdi} absent, mask HAS it")
                elif val not in (None, "absent") and fdi not in present:
                    problems.append(f"[{reader}] {arch} arch says {fdi} {val}, mask LACKS it")
    return problems


# ── screen 2: mirror ──────────────────────────────────────────────────────

def screen_mirror(case, facts, readers, ranking=None, **_):
    """
    Laterality-flipped report. Decided on the full absent-tooth list.

    A flip verdict from this screen is per-case and trustworthy only when the
    mask-absent list is SMALL: facts.teeth_absent is the complement of
    everything segmented, so for a partially-scanned arch it holds 16+ teeth,
    and against a set that large a mirrored FDI lands in it almost as often as
    the original by chance. The aggregate flip rate over the split is therefore
    NOT a population estimate -- see fewshot_probe_plan.md Step 3b.
    """
    st = facts["structured"]
    mask_absent = {int(t) for t in st.get("teeth_absent", [])}
    problems = []
    for reader, gt in readers.items():
        g = gt.get("global") or {}
        stated_absent = set()
        for arch in ARCH_FDI:
            f = ((g.get(f"dental_arch_findings_{arch}") or {}).get("findings") or {})
            stated_absent |= {int(k) for k, v in f.items() if v == "absent"}
        if len(stated_absent) < 2:
            continue
        direct = len(stated_absent & mask_absent)
        flipped = len({mirror_fdi(f) for f in stated_absent} & mask_absent)
        if flipped > direct and direct == 0:
            problems.append(
                f"[{reader}] absent list {sorted(stated_absent)} matches the mask "
                f"{flipped}/{len(stated_absent)} MIRRORED vs {direct}/{len(stated_absent)} "
                f"as written -- laterality-flipped report")
        elif flipped > direct:
            problems.append(
                f"[{reader}] absent list leans mirrored ({flipped} vs {direct}) "
                f"-- weak signal, inspect before use")
    # the standing ranking already convicts some cases; report its verdict too
    if ranking and case in ranking and ranking[case].get("mirror_verdict") == "mirrored":
        problems.append("gt_quality ranking convicts this case as mirrored")
    return problems


# ── screen 3: wisdom geometry, measured off the mask ──────────────────────

def _tooth_geometry(mask, spacing, fdi):
    """
    -> dict or None. Everything here is measured, not judged:

      crest_gap_mm  how far the tooth's occlusal tip sits BELOW the local
                    alveolar crest (mandible bone) around it. Positive = the
                    crown tip is under the crest, i.e. bone still covers it.
      neighbour_gap_mm  the same tip measured against the second molar's tip:
                    positive = this tooth stands lower in the arch than its
                    neighbour, which is what a partially erupted third molar
                    looks like.
      tip_deg       angle between this tooth's long axis and its neighbour's.
    """
    import numpy as np

    vox = np.argwhere(mask == fdi)
    if vox.size == 0:
        return None
    nb = np.argwhere(mask == NEIGHBOUR[fdi])
    bone = mask == MANDIBLE_LABEL

    sp = np.asarray(spacing, dtype=float)
    # superior-inferior axis: the one along which the mandible bone is thinnest
    # is never it; use the axis with the largest bone extent as A-P/L-R and
    # take SI as the remaining axis with the smallest bone extent.
    extents = [np.ptp(np.argwhere(bone)[:, i]) * sp[i] for i in range(3)]
    si = int(np.argmin(extents))
    lower = fdi >= 30  # mandibular teeth erupt upward

    def axis(points):
        c = points.mean(0)
        u, s, vt = np.linalg.svd((points - c) * sp, full_matrices=False)
        return vt[0]

    tip = vox[:, si].max() if lower else vox[:, si].min()
    out = {"n_voxels": int(len(vox))}

    # local alveolar crest: bone voxels within 8 mm of the tooth in-plane
    other = [i for i in range(3) if i != si]
    c0, c1 = vox[:, other[0]].mean(), vox[:, other[1]].mean()
    bidx = np.argwhere(bone)
    near = bidx[(np.abs(bidx[:, other[0]] - c0) * sp[other[0]] < 8) &
                (np.abs(bidx[:, other[1]] - c1) * sp[other[1]] < 8)]
    if len(near):
        crest = near[:, si].max() if lower else near[:, si].min()
        out["crest_gap_mm"] = round(float((crest - tip) * sp[si]) * (1 if lower else -1), 1)
    if len(nb):
        ntip = nb[:, si].max() if lower else nb[:, si].min()
        out["neighbour_gap_mm"] = round(float((ntip - tip) * sp[si]) * (1 if lower else -1), 1)
        a, b = axis(vox.astype(float)), axis(nb.astype(float))
        cos = abs(float(np.dot(a, b)))
        out["tip_deg"] = round(float(np.degrees(np.arccos(min(1.0, cos)))), 1)
    return out


def screen_wisdom_geometry(case, facts, readers, root=None, split=None, **_):
    """
    Compare the report's eruption/orientation words with the mask geometry.

    The point is not to overrule the radiologist -- it is that the exemplar's
    answer has to be true OF THE PICTURE. A crown that the mask shows standing
    clear of the crest cannot be described as complete_bony_inclusion, however
    the report phrased it.
    """
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        return ["nibabel/numpy unavailable -- run this screen in cbct_base"]

    p = root / "dataset" / split / "masks" / f"{case}.nii.gz"
    if not p.exists():
        return [f"no mask at {p}"]
    img = nib.load(str(p))
    mask = np.asarray(img.dataobj)
    spacing = img.header.get_zooms()[:3]

    problems = []
    for fdi in (48, 38):
        geo = _tooth_geometry(mask, spacing, fdi)
        claims = {(r, json.dumps({k: v for k, v in ((gt.get("global") or {}).get(WISDOM[fdi]) or {}).items()
                                  if k != "visual_evidence"}, sort_keys=True))
                  for r, gt in readers.items()}
        if geo is None:
            for r, c in claims:
                d = json.loads(c) if c else {}
                if d.get("eruption_state") not in (None, "absent"):
                    problems.append(f"{fdi}: not in the mask but [{r}] says "
                                    f"{d.get('eruption_state')}")
            continue
        print(f"    {case} {fdi} geometry: {geo}")
        for r, c in claims:
            d = json.loads(c) if c else {}
            state, orient = d.get("eruption_state"), d.get("orientation")
            gap = geo.get("crest_gap_mm")
            if state == "complete_bony_inclusion" and gap is not None and gap < 0:
                problems.append(
                    f"{fdi}: [{r}] says complete_bony_inclusion but the crown tip "
                    f"stands {-gap} mm ABOVE the local crest -- the render will "
                    f"show a tooth clear of bone")
            if state == "fully_erupted" and gap is not None and gap > 2:
                problems.append(
                    f"{fdi}: [{r}] says fully_erupted but the crown tip sits {gap} mm "
                    f"BELOW the local crest")
            deg = geo.get("tip_deg")
            if orient == "normal" and deg is not None and deg > 35:
                problems.append(
                    f"{fdi}: [{r}] says orientation normal but its long axis is "
                    f"{deg}deg off {NEIGHBOUR[fdi]}'s")
            if orient not in (None, "normal") and deg is not None and deg < 12:
                problems.append(
                    f"{fdi}: [{r}] says orientation {orient} but its long axis is "
                    f"only {deg}deg off {NEIGHBOUR[fdi]}'s")
    return problems


# ── screen 4: the word actually has to be there ───────────────────────────

def screen_impaction_wording(case, facts, readers, root=None, split=None, **_):
    """
    A candidate offered as an impaction POSITIVE must have an explicit
    impaction word in the sentence that names the tooth. The keyword table
    elsewhere is deliberately broad (it includes "ectopic"), which is how
    A011's "ectopic endodontic material between teeth 47 and 48" once scored
    as "48 impacted".
    """
    reports = load_reports(root, split, case)
    problems = []
    for reader, gt in readers.items():
        g = gt.get("global") or {}
        for fdi, fact in WISDOM.items():
            if not (g.get(fact) or {}).get("impacted"):
                continue
            ok = False
            for text in reports.values():
                for sentence in re.split(r"(?<=[.;])\s+", text):
                    if str(fdi) in sentence and IMPACTION_WORDS.search(sentence):
                        ok = True
            if not ok:
                problems.append(f"[{reader}] {fdi} marked impacted but no sentence "
                                f"names {fdi} with an impaction word")
    return problems


SCREENS = {
    "report-vs-mask": screen_report_vs_mask,
    "mirror": screen_mirror,
    "wisdom-geometry": screen_wisdom_geometry,
    "impaction-wording": screen_impaction_wording,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--split", default="training")
    ap.add_argument("--cases", help="comma-separated case IDs")
    ap.add_argument("--clean-set", action="store_true",
                    help="every case the gt_quality ranking calls clean "
                         "(tier A, no mirror conviction, readers agreeing)")
    ap.add_argument("--screen", default="report-vs-mask,mirror,impaction-wording",
                    help=f"comma-separated: {','.join(SCREENS)} (or 'all')")
    args = ap.parse_args()

    root = args.project_dir
    ranking = load_ranking(root, args.split)

    if args.clean_set:
        cases = [c for c, q in ranking.items()
                 if q.get("score_asis", 0) >= 0.90 and q.get("n_evidence", 0) >= 10
                 and q.get("mirror_verdict") != "mirrored"
                 and (q.get("interreader_score") is None or q["interreader_score"] >= 0.80)]
        cases.sort(key=lambda c: ranking[c]["rank"])
    elif args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        ap.error("give --cases or --clean-set")

    names = list(SCREENS) if args.screen == "all" else \
        [s.strip() for s in args.screen.split(",") if s.strip()]
    for n in names:
        if n not in SCREENS:
            ap.error(f"unknown screen {n!r}; pick from {','.join(SCREENS)}")

    failed = []
    for case in cases:
        facts = load_facts(root, args.split, case)
        readers = load_readers(root, args.split, case)
        q = ranking.get(case, {})
        head = (f"{case}  rank={q.get('rank', '?')} tier={q.get('tier', '?')} "
                f"score={q.get('score_asis', float('nan')):.2f} "
                f"mirror={q.get('mirror_verdict', '?')} readers={len(readers)}"
                if q else f"{case}  (not in the ranking)")
        print("=" * 78)
        print(head)
        if not readers:
            print("  [FAIL] no ground_truth extraction for this case")
            failed.append(case)
            continue
        problems = []
        for n in names:
            problems += [f"{n}: {p}" for p in
                         SCREENS[n](case, facts, readers, root=root,
                                    split=args.split, ranking=ranking)]
        if problems:
            failed.append(case)
            for p in problems:
                print("  [FAIL]", p)
        else:
            print("  [PASS] every requested screen")

    print()
    print(f"{len(cases) - len(failed)}/{len(cases)} case(s) passed "
          f"({','.join(names)})")
    if failed:
        print("failed:", " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
