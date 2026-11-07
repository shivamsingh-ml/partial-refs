#!/usr/bin/env python3
"""
code/synthesize_report_LIM.py   --  "Less Is More"

BACK-UP ARM. synthesize_report.py is the main code and is NOT touched by
this file: every sentence template, ordering decision and REAL SNIPPET is
inherited from it. See postprocess_pred_LIM.py for why this arm exists.

WHAT THIS ADDS OVER JUST FEEDING IT A SILENCED SUMMARY
──────────────────────────────────────────────────────
postprocess_pred_LIM.py already deletes the silenced keys, and the main
renderer omits a section whose key is missing -- so pointing plain
synthesize_report.py at a LIM summary would very nearly work. "Very nearly"
is the problem: it would silently depend on which keys happen to be absent,
so a full (non-LIM) summary handed to it by mistake would render the
treatment sentences right back into the report and quietly invalidate the
arm.

This module instead makes the silence a property of the RENDERER: the
suppressed renderers are stubbed out to return nothing at all. Feed it any
summary, LIM or not, and the treatment sentences cannot appear.

Usage (identical flags to synthesize_report.py):
    python3 code/synthesize_report_LIM.py \
        --summary-dir outputs/aksssr_v4_validate_LIM/summaries \
        --out-dir     outputs/aksssr_v4_validate_LIM/synthesized_reports
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthesize_report as base  # noqa: E402
from postprocess_pred_LIM import SILENCE_LEVELS, DEFAULT_LEVEL  # noqa: E402


# ── The silencing itself ──────────────────────────────────────────────────
#
# Each summary key that a level silences maps to the renderer(s) that turn
# it into text. Stubbing the renderer is what makes the silence stick
# regardless of what the summary contains.

_RENDERERS_BY_KEY = {
    "endodontic_summary":          ["render_endodontic_summary"],
    "restoration_summary":         ["render_restoration_summary"],
    "prosthetics":                 ["render_prosthetics"],
    "tooth_findings":              ["render_morphology_findings"],
    "root_remnants":               ["render_root_remnants"],
    "impacted_teeth":              ["render_impacted_teeth"],
    "periodontal_bone_resorption": ["render_periodontal_bone_resorption"],
    "alveolar_bone_atrophy":       ["render_alveolar_bone_atrophy"],
}

# render_* helpers that return a single string rather than a list; their
# stub has to return None (the caller tests truthiness), not [].
_SCALAR_RENDERERS = {"render_periodontal_bone_resorption"}


def apply_silence(level: str = DEFAULT_LEVEL) -> List[str]:
    """
    Monkey-patch the inherited renderers to emit nothing. Module-level and
    process-wide -- this module is an ARM, not a library to mix with the
    normal renderer in the same process.

    Returns the names of the renderers that were stubbed, for logging.
    """
    if level not in SILENCE_LEVELS:
        raise ValueError(f"unknown level {level!r}; "
                         f"expected one of {sorted(SILENCE_LEVELS)}")

    stubbed = []
    for key in SILENCE_LEVELS[level]:
        for fn_name in _RENDERERS_BY_KEY.get(key, []):
            if not hasattr(base, fn_name):
                continue
            if fn_name in _SCALAR_RENDERERS:
                setattr(base, fn_name, lambda *a, **k: None)
            else:
                setattr(base, fn_name, lambda *a, **k: [])
            stubbed.append(fn_name)
    return stubbed


def synthesize_report_LIM(summary: Dict, level: str = DEFAULT_LEVEL) -> str:
    apply_silence(level)
    return base.synthesize_report(summary)


# ── CLI (mirrors synthesize_report.py) ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="'Less is more' report rendering: synthesize_report.py "
                    "with the error-prone sentences silenced at the renderer.")
    ap.add_argument("--summary-dir", required=True,
                    help="Directory of {case_id}_summary.json files, or a single such file")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for {case_id}.txt report files")
    ap.add_argument("--case-ids", nargs="+", default=None)
    ap.add_argument("--level", default=None, choices=sorted(SILENCE_LEVELS),
                    help="Override the level; by default each summary's own "
                         "_lim_level is used, falling back to "
                         f"'{DEFAULT_LEVEL}' for a summary that has none.")
    args = ap.parse_args()

    summary_path = Path(args.summary_dir)
    summary_files = ([summary_path] if summary_path.is_file()
                     else sorted(summary_path.glob("*_summary.json")))
    if args.case_ids:
        summary_files = [p for p in summary_files
                         if p.stem.replace("_summary", "") in args.case_ids]
    if not summary_files:
        print(f"[WARN] No summary files found under {summary_path}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    logged_level = None
    for sf in summary_files:
        summary = json.loads(sf.read_text())
        level = args.level or summary.get("_lim_level") or DEFAULT_LEVEL
        if level != logged_level:
            print(f"[INFO] LIM level '{level}' silences renderers: "
                  f"{', '.join(apply_silence(level))}")
            logged_level = level
        report_text = base.synthesize_report(summary)
        case_id = summary.get("case_id", sf.stem.replace("_summary", ""))
        out_path = Path(args.out_dir) / f"{case_id}.txt"
        out_path.write_text(report_text)
        print(f"[{case_id}] -> {out_path}")


if __name__ == "__main__":
    main()
