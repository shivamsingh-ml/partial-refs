#!/usr/bin/env python3
"""
code/postprocess_pred_LIM.py   --  "Less Is More"

BACK-UP ARM. postprocess_pred.py is the main code and is NOT touched by
this file: everything here is a wrapper around it.

THE IDEA
────────
The v4 official ranking on this split scored RadFact logical
PRECISION 0.2268 against RECALL 0.5309 -- the reports find a fair share of
what is really there, but roughly three quarters of what they assert is not
supported by the reference. Clinical Score is the F1 of those two and
carries 0.8 of the final score, so an unsupported sentence is not free: it
costs precision on every case that emits it.

So: if a fact is easy to get wrong, don't say it at all. This arm keeps the
statements that are structural, few, and hard to be wrong about (was the
arch in the volume, are the condyles there, the canal course, how many teeth
are missing) and deletes the ones that require reading fine detail off a
single tooth composite -- above all the treatment findings (root canal
treatment, crowns, fillings, post-and-core, implants, bridges), which are
also the most numerous sentences per report and therefore the biggest
precision liability.

HOW IT WORKS
────────────
It calls postprocess_pred.postprocess_prediction() unchanged and then
DELETES the silenced keys from the summary. Deliberately not a forked copy
of the 1000-line original: every fix to the main postprocessor (scope
handling, absent-teeth suppression, wisdom impaction, ...) is inherited
automatically, and the diff between the two arms stays readable -- it is
exactly the SILENCE_LEVELS table below.

LEVELS
──────
  treatments (default) -- silences dental treatment only, i.e. the
        instruction as given: RCT, crowns, "and other tooth treatment".
        Pathology findings (caries, fracture, root remnants) still speak.

  strict               -- also silences per-tooth pathology and the
        arch-level periodontal/atrophy grades. What survives is scope,
        condyles, canals, absent teeth, sinus and bone quality: the
        skeleton of a report, almost none of which depends on resolving
        a lesion in a single tooth crop.

Usage (identical flags to postprocess_pred.py, plus --level):
    python3 code/postprocess_pred_LIM.py \
        --pred-dir outputs/aksssr_v4_validate/predictions \
        --out-dir  outputs/aksssr_v4_validate_LIM/summaries \
        --schema   schema/schema.json \
        --level    treatments
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocess_pred as base  # noqa: E402


# ── What each level silences ──────────────────────────────────────────────
#
# Keys are removed from BOTH arch sections. Anything not listed here is
# passed through exactly as the main postprocessor built it.

SILENCE_LEVELS: Dict[str, List[str]] = {
    # The instruction as given: treatment only.
    "treatments": [
        "endodontic_summary",     # root canal treatment + filling quality
        "restoration_summary",    # crowns, fillings, post-and-core
        "prosthetics",            # implants, bridges
    ],
    # Everything above, plus the findings that need fine detail resolved
    # inside one tooth's crop, plus the two graded bone facts (mild/
    # moderate/severe is a judgement call the VLM has no calibration for).
    "strict": [
        "endodontic_summary",
        "restoration_summary",
        "prosthetics",
        "tooth_findings",             # caries, crown defect, root fracture,
                                      # resorption, per-tooth periodontal
        "root_remnants",
        "impacted_teeth",
        "periodontal_bone_resorption",
        "alveolar_bone_atrophy",
    ],
}

DEFAULT_LEVEL = "treatments"


def silence_summary(summary: Dict, level: str = DEFAULT_LEVEL) -> Dict:
    """
    Drop the silenced keys from both arches. Returns a NEW dict; the input
    summary is left alone so a caller can keep the full version too.
    """
    if level not in SILENCE_LEVELS:
        raise ValueError(f"unknown level {level!r}; "
                         f"expected one of {sorted(SILENCE_LEVELS)}")
    drop = SILENCE_LEVELS[level]

    out = dict(summary)
    for arch in ("mandible", "maxilla"):
        section = summary.get(arch)
        if not isinstance(section, dict):
            continue
        out[arch] = {k: v for k, v in section.items() if k not in drop}
    out["_lim_level"] = level
    out["_lim_silenced"] = list(drop)
    return out


def postprocess_prediction_LIM(pred: Dict, level: str = DEFAULT_LEVEL,
                               **kwargs) -> Dict:
    """postprocess_pred.postprocess_prediction(), then silenced."""
    return silence_summary(base.postprocess_prediction(pred, **kwargs), level)


# ── CLI (mirrors postprocess_pred.py) ─────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="'Less is more' postprocessing: postprocess_pred.py with "
                    "the error-prone facts silenced. Writes the same "
                    "{case_id}_summary.json shape, minus the silenced keys.")
    ap.add_argument("--pred-dir", required=True,
                    help="Directory of {case_id}_pred.json files, or a single such file")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for {case_id}_summary.json files")
    ap.add_argument("--case-ids", nargs="+", default=None,
                    help="Filter to specific case IDs")
    ap.add_argument("--schema", default=None,
                    help="Path to schema.json (passed through to normalize_prediction)")
    ap.add_argument("--level", default=DEFAULT_LEVEL, choices=sorted(SILENCE_LEVELS),
                    help=f"How much to silence (default: {DEFAULT_LEVEL})")
    args = ap.parse_args()

    pred_path = Path(args.pred_dir)
    pred_files = ([pred_path] if pred_path.is_file()
                  else sorted(pred_path.glob("*_pred.json")))
    if args.case_ids:
        pred_files = [p for p in pred_files
                      if p.stem.replace("_pred", "") in args.case_ids]
    if not pred_files:
        print(f"[WARN] No prediction files found under {pred_path}")
        return

    print(f"[INFO] LIM level '{args.level}' silences: "
          f"{', '.join(SILENCE_LEVELS[args.level])}")

    os.makedirs(args.out_dir, exist_ok=True)
    for pf in pred_files:
        pred = json.loads(pf.read_text())
        summary = postprocess_prediction_LIM(pred, level=args.level,
                                             schema_path=args.schema, quiet=True)
        case_id = pred.get("case_id", pf.stem.replace("_pred", ""))
        out_path = Path(args.out_dir) / f"{case_id}_summary.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[{case_id}] -> {out_path}")


if __name__ == "__main__":
    main()
