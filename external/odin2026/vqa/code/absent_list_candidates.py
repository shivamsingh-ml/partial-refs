#!/usr/bin/env python3
"""Which absent-teeth list is the most accurate? Scored on validate-40.

Every candidate is scored only over positions the generated GT actually
answers -- no report enumerates all 32, so a position no report mentions is
unanswered, not "present", and charging it as a false positive punishes
whichever source claims most.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "code")
from survey_findings import (ALL_TEETH, ARCH_TEETH, _as_dict, facts_absent,
                             pred_absent, summary_absent)
from survey_facts import load_cases

RUN = Path("outputs/aksssr_v7_trained_validate")
GT = Path("dataset/validate/outputs/ground_truth")
UPPER = set(ARCH_TEETH["maxilla"])
LOWER = set(ARCH_TEETH["mandible"])

cases = load_cases(RUN, GT, None)
rows = {}


def add(name, claimed, truth, answered):
    tp = len(claimed & truth & answered)
    fp = len((claimed & answered) - truth)
    fn = len((truth & answered) - claimed)
    s = rows.setdefault(name, [0, 0, 0])
    s[0] += tp
    s[1] += fp
    s[2] += fn


for c in cases:
    cid = c["case_id"]
    gt = pred_absent(c["gt"])
    truth, answered = gt["absent"], gt["read"]
    facts = json.loads(Path(f"dataset/validate/facts/{cid}.json").read_text())
    fabs, _ = facts_absent(facts)
    fov = _as_dict(_as_dict(facts.get("structured")).get("fov"))
    max_fov = str(fov.get("maxilla") or "").lower()

    pa = pred_absent(c["pred"])
    pano = pa["by_source"]["panoramic"]
    d3 = pa["by_source"]["3d_render"]
    summ = summary_absent(c["summary"])["absent"] if c["summary"] else set()

    # -- single sources
    add("facts.teeth_absent", fabs, truth, answered)
    add("panoramic arch read", pano, truth, answered)
    add("current SUMMARY (shipped)", summ, truth, answered)

    # -- FOV gating: an upper tooth outside a maxilla that is not fully in the
    #    volume is NOT IMAGED, which is not the same claim as absent.
    add("facts, drop upper if maxilla excluded",
        fabs - (UPPER if max_fov == "excluded" else set()), truth, answered)
    add("facts, drop upper unless maxilla full",
        fabs - (UPPER if max_fov != "included" else set()), truth, answered)

    # -- combinations with the panoramic read
    add("facts AND panoramic", fabs & pano, truth, answered)
    add("facts OR panoramic", fabs | pano, truth, answered)
    add("facts OR (panoramic AND 3d)", fabs | (pano & d3), truth, answered)

    # -- the shape the pipeline already has, with the better mask signal:
    #    mask-absent, split by panoramic confirmation, keeping both halves
    add("facts OR panoramic, FOV-gated",
        (fabs | pano) - (UPPER if max_fov == "excluded" else set()),
        truth, answered)
    add("facts AND panoramic, plus lower facts",
        (fabs & pano) | (fabs & LOWER), truth, answered)

print(f"{'candidate':44} {'claims':>7} {'tp':>5} {'fp':>5} {'fn':>5} "
      f"{'prec':>6} {'rec':>6} {'F1':>6}")
print("-" * 96)
out = []
for name, (tp, fp, fn) in rows.items():
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    out.append((f1, name, tp + fp, tp, fp, fn, p, r))
for f1, name, claims, tp, fp, fn, p, r in sorted(out, reverse=True):
    print(f"{name:44} {claims:>7} {tp:>5} {fp:>5} {fn:>5} "
          f"{p:>6.2f} {r:>6.2f} {f1:>6.2f}")
