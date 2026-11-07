#!/usr/bin/env python3
"""Impaction on 18/28/38/48: which sources, combined how.

Rule (schema _definitions.impacted, and the one asked for):
    impacted = (eruption_state in {complete_bony_inclusion, partially_erupted})
               OR (orientation not in {normal, null})
    absent   -> no claim at all.

Scored against the GT's own `impacted` where the GT commits to one.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "code")
from survey_findings import WISDOM_SLOT, _as_dict
from survey_facts import load_cases

RUN = Path("outputs/aksssr_v7_trained_validate")
cases = load_cases(RUN, Path("dataset/validate/outputs/ground_truth"), None)
UNERUPTED = {"complete_bony_inclusion", "partially_erupted"}


def norm(v):
    return v.strip().lower() if isinstance(v, str) else None


def derive(es, orient):
    """The schema rule. None = no claim (absent, or nothing to go on)."""
    if es == "absent":
        return None
    if es is None and orient is None:
        return None
    if es in UNERUPTED:
        return True
    if orient is not None and orient != "normal":
        return True
    if es == "fully_erupted":
        return False        # orientation normal or unstated
    return None


comp = {k: Counter() for k in ("3d_es", "detail_es", "3d_orient", "3d_bool")}
agree = Counter()
rules = {}


def score(name, pred, truth):
    s = rules.setdefault(name, Counter())
    if truth is None:
        return
    if pred is None:
        s["nocall"] += 1
        s["fn"] += int(truth)
        return
    s["tp"] += int(pred and truth)
    s["fp"] += int(pred and not truth)
    s["fn"] += int((not pred) and truth)
    s["tn"] += int((not pred) and (not truth))


for c in cases:
    g, p = _as_dict(c["gt"].get("global")), _as_dict(c["pred"].get("global"))
    teeth = _as_dict(c["pred"].get("teeth"))
    arch_imp = set()
    for a in ("mandible", "maxilla"):
        f = _as_dict(_as_dict(p.get(f"dental_arch_findings_{a}")).get("findings"))
        for k, v in f.items():
            try:
                k = int(k)
            except (TypeError, ValueError):
                continue
            if norm(v) == "impacted":
                arch_imp.add(k)

    for slot, fdi in WISDOM_SLOT.items():
        gb = _as_dict(g.get(slot))
        pb = _as_dict(p.get(slot))
        truth = gb.get("impacted")
        truth = truth if isinstance(truth, bool) else None

        g_es, g_or = norm(gb.get("eruption_state")), norm(gb.get("orientation"))
        es3, or3 = norm(pb.get("eruption_state")), norm(pb.get("orientation"))
        b3 = pb.get("impacted")
        b3 = b3 if isinstance(b3, bool) else (
            True if str(b3).lower() == "true" else
            False if str(b3).lower() == "false" else None)
        esd = norm(_as_dict(_as_dict(teeth.get(f"tooth_{fdi}"))
                            .get(f"tooth_{fdi}_eruption")).get("eruption_state"))

        # component accuracy, where both sides commit
        if g_es and es3:
            comp["3d_es"]["n"] += 1; comp["3d_es"]["ok"] += g_es == es3
        if g_es and esd:
            comp["detail_es"]["n"] += 1; comp["detail_es"]["ok"] += g_es == esd
        if g_or and or3:
            comp["3d_orient"]["n"] += 1; comp["3d_orient"]["ok"] += g_or == or3
        if truth is not None and b3 is not None:
            comp["3d_bool"]["n"] += 1; comp["3d_bool"]["ok"] += b3 == truth
        if es3 and esd:
            agree["both"] += 1; agree["same"] += es3 == esd
        agree["detail_answered"] += bool(esd)
        agree["3d_answered"] += bool(es3)

        # ---- candidate rules
        score("A  3d `impacted` bool as-is (current)", b3, truth)
        score("B  derive from 3d es+orientation", derive(es3, or3), truth)
        score("C  derive, detail es preferred over 3d es",
              derive(esd or es3, or3), truth)
        score("D  derive, 3d es preferred, detail as fallback",
              derive(es3 or esd, or3), truth)
        score("E  derive from detail es only (no orientation)",
              derive(esd, None), truth)
        score("F  B, or panoramic says impacted",
              True if fdi in arch_imp else derive(es3, or3), truth)
        score("G  B, or 3d bool when B has no call",
              derive(es3, or3) if derive(es3, or3) is not None else b3, truth)
        score("H  D, or panoramic when still no call",
              (derive(es3 or esd, or3) if derive(es3 or esd, or3) is not None
               else (True if fdi in arch_imp else None)), truth)
        score("I  unerupted only (ignore orientation)",
              None if es3 == "absent" else
              (es3 in UNERUPTED) if es3 else None, truth)
        score("J  orientation only (ignore eruption)",
              None if es3 == "absent" else
              (or3 is not None and or3 != "normal") if or3 else None, truth)

print("COMPONENT ACCURACY (where GT and the source both commit)")
for k, v in comp.items():
    print(f"  {k:12} {v['ok']:>3}/{v['n']:<4} {v['ok']/v['n'] if v['n'] else 0:.2f}")
print(f"\n  3d answered eruption on {agree['3d_answered']} slots, "
      f"detail on {agree['detail_answered']}; both on {agree['both']}, "
      f"agreeing {agree['same']}")

print(f"\n{'rule':46} {'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4} {'no call':>8} "
      f"{'prec':>6} {'rec':>6} {'F1':>6} {'acc':>6}")
print("-" * 104)
out = []
for name, s in rules.items():
    tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    out.append((f1, name, tp, fp, fn, tn, s["nocall"], p, r, acc))
for f1, name, tp, fp, fn, tn, nc, p, r, acc in sorted(out, reverse=True):
    print(f"{name:46} {tp:>4} {fp:>4} {fn:>4} {tn:>4} {nc:>8} "
          f"{p:>6.2f} {r:>6.2f} {f1:>6.2f} {acc:>6.2f}")
