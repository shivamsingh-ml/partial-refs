#!/usr/bin/env python3
"""
code/generate_report_from_pred.py

Generates narrative CBCT dental reports using an LLM (Qwen3, text-only --
no images needed at this stage) instead of the old deterministic template
renderer. Two-stage design, per the earlier discussion:

  {case_id}_pred.json  --[postprocess_pred.py's classification logic]-->
  {case_id}_summary.json (compact, pre-classified)  --[THIS SCRIPT]-->
  {case_id}.txt (narrative report)

WHY A SEPARATE TEXT-ONLY LLM STAGE (not the VLM, not the old template):
  - The report-writer never needs to see images -- it's translating
    already-extracted structured findings into prose, so a cheap text
    model works and you don't re-spend image tokens.
  - Precise classification (which absent-teeth pattern, which teeth share
    a restoration quality, whether a normal-side sentence is owed) stays
    in postprocess_pred.py's Python code, where it's testable and exact.
    This script's ONLY job is prose generation from that already-decided
    structure -- it is not asked to re-derive any of the classification
    logic itself.

This script imports postprocess_pred.py directly rather than requiring a
separate postprocessing pass first -- pass raw {case_id}_pred.json files
and it postprocesses them internally. (If you'd rather inspect/version the
intermediate {case_id}_summary.json files, run postprocess_pred.py first
and pass its output directory instead -- both raw predictions and
already-postprocessed summaries are accepted, detected by the presence of
a "global"/"teeth" top-level shape vs. the compact summary shape.)

Usage:
    python code/generate_report_from_pred.py \
        --pred-dir  dataset/training/outputs/predictions \
        --out-dir   dataset/training/outputs/synthesized_reports \
        --vllm-url  http://localhost:8000/v1 \
        --model     qwen3-14b

    LIMIT=5 python code/generate_report_from_pred.py ...     (smoke test)
    python code/generate_report_from_pred.py ... --dry-run   (no vLLM calls)
    python code/generate_report_from_pred.py ... --resume    (skip existing .txt)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess_pred import postprocess_prediction  # noqa: E402


# ── Empty-prediction safety net ──────────────────────────────────────────

def _is_blank(v) -> bool:
    if v in (None, "", [], {}):
        return True
    if isinstance(v, dict):
        return all(_is_blank(x) for x in v.values())
    if isinstance(v, list):
        return len(v) == 0
    return False


def is_empty_prediction(pred: Dict) -> bool:
    """
    Matches the old template's empty_prediction rule: if the model produced
    essentially nothing usable (no global facts, every tooth either marked
    absent, never imaged, or missing entirely), output an empty report
    rather than asking the LLM to write boilerplate around nothing.
    """
    g = pred.get("global", {}) or {}
    if any(not _is_blank(v) for v in g.values()):
        return False
    teeth = pred.get("teeth", {}) or {}
    for t in teeth.values():
        if t.get("detected") not in ("no", "no_image", None):
            return False
    return True


# ── Writing guide (system prompt) ─────────────────────────────────────────
# Condensed from the retired template's metadata/rules/helpers blocks --
# NOT the template's literal branching logic (an LLM shouldn't be handed
# that as instructions -- see postprocess_pred.py's module docstring for
# why), just the style/structure/interpretation rules that stay genuinely
# ambiguous after postprocess_pred.py's classification.

SYSTEM_PROMPT = """\
You are an expert dental/maxillofacial radiologist writing a CBCT report \
from a structured JSON. Write clinical, concise prose -- like a \
radiologist dictating, not a casual description.

The JSON is already filtered and classified (normal items dropped, teeth \
grouped, conflicts resolved) -- just write prose from it. Never mention or \
invent a finding not present in the JSON.

STRUCTURE
─────────
1. Opening sentence only if dentition_type == "mixed": "Pediatric patient \
   with incomplete eruption of the permanent teeth." Otherwise skip.

2. MANDIBLE (with a short "MANDIBLE" label):
   a. Main paragraph, one block: scope -> alveolar bone/periodontal (only \
      if abnormal) -> dental arch (present/absent teeth, wisdom teeth \
      38/48) -> teeth detail (endodontics, restorations, root remnants, \
      impacted teeth, other findings; most confident first). If there's a \
      cyst/lesion finding for the mandible, end the paragraph with it. If \
      normal, say nothing here -- save it for the closing paragraph (4).
   b. Canals paragraph -- always its own paragraph, both sides, even if \
      fully normal.
   c. Condyles paragraph -- always its own paragraph, both sides, even if \
      fully normal or not included.

3. MAXILLA, same shape: main paragraph (scope -> \
   alveolar bone/periodontal -> arch with wisdom teeth 18/28 -> teeth \
   detail -> cyst/lesion if present) -> sinuses paragraph (always) -> \
   nasal cavity, only if abnormal.

4. Closing paragraph, only if BOTH arches' bone quality is clear: one \
   merged negative sentence. Omit if nothing remains to state.

KEY RULES
─────────
- group_status "all_normal" -> one merged sentence. "mixed" -> describe \
  every member individually, including normal ones.
- Absent teeth: "consecutive_run" -> range; "scatter" -> list both absent \
  and present; "quadrant_edentulous_*" -> "Complete edentulism of the \
  {quadrant} quadrant" (+ "except for {remaining}" if nearly-edentulous).
- Impacted teeth and root remnants: assign to the arch matching their FDI \
  quadrant (I/II->maxilla, III/IV->mandible); skip silently if absent.
- Intrasinusal teeth: state_negative true -> explicit negative sentence; \
  otherwise name the root(s) involved.
- Tooth numbering: "Tooth X" singular, "Teeth X and Y" / "Teeth X, Y and \
  Z" for multiple.
- Ignore "conflicting_sources" flags -- just describe using the fuller/ \
  union info already present in the item.
- FDI quadrants: I=11-18, II=21-28, III=31-38, IV=41-48.

EXAMPLE
───────
Given a JSON with: dentition_type "permanent"; mandible partially \
included, mild anterior periodontal bone loss, teeth 31-38/41-48 all \
present, tooth 48 impacted/horizontal, root canal + crown findings on \
several teeth, root remnants at 36/45, mixed canals (right close to roots \
45-47, left regular), condyles partially included (right normal, left not \
visualized); maxilla fully included with sinuses, teeth 11-18/21-28 all \
present, tooth 18 impacted/mesial, endodontic + crown findings, root \
remnant at 56, a periapical radiolucency at 26, mixed sinuses (right \
normal, left with roots of 23/26/27 projecting in) -- the report should \
read:

"MANDIBLE

Mandible is partially included in the scan volume. Mild atrophy of the \
mandibular ridge is noted in the anterior region bilaterally. Diffuse mild \
periodontal bone loss is present, predominantly affecting the mandibular \
anterior teeth, with slightly decreased vertical bone height at tooth 45. \
Teeth 31 to 38 and 41 to 48 are present in the mandibular arch. Tooth 38 \
is fully erupted with normal orientation. Tooth 48 is unerupted, \
horizontally impacted, with lingual orientation. Root canal treatment is \
present at teeth 35, 36, 46 and 47 with adequate filling quality, and at \
tooth 45 with inadequate filling quality. Coronal restorations are \
present at teeth 36 and 45 (crowns) and tooth 47 (filling). Root remnants \
are noted at teeth 36 and 45, without periodontal bone resorption. \
Possible residual bony fragments are seen at the 36 and 45 regions, \
consistent with early post-extraction healing.

The mandibular canals show a mixed course: the right canal follows a \
curved course through the retromolar and molar region, running lingually \
and superiorly, and is closely related to the roots of teeth 45, 46 and \
47; the left canal shows a normal, regular sigmoid course beneath the \
roots of teeth 31 to 38.

The mandibular condyles are partially included in the scan volume. The \
right condyle appears as a smooth, rounded structure articulating \
normally with the temporal bone, with no degenerative changes. The left \
condyle is not clearly visualized for morphological assessment.

MAXILLA

Maxilla is completely included in the scan volume, including both \
maxillary sinuses. Teeth 11 to 18 and 21 to 28 are present in the \
maxillary arch. Tooth 18 is partially erupted, mesially impacted, with \
lingual orientation, causing impaction against tooth 17. Tooth 28 is \
partially erupted with normal orientation and no impaction. Root canal \
treatment with adequate obturation is present at teeth 12, 13, 22, 23 and \
24. Coronal restorations are present at these same teeth (crowns) and at \
tooth 21 (filling). A root remnant is noted at tooth 56, without \
periodontal bone resorption. A periapical radiolucency is present around \
tooth 26, compatible with periapical periodontitis/cyst associated with \
an extensive restorative defect.

The right maxillary sinus is normally aerated with no mucosal thickening, \
fluid, or intrasinusal teeth; the left sinus is likewise normally \
aerated, but the roots of teeth 23, 26 and 27 project into the sinus \
floor/cavity."

(No closing paragraph here, since both arches already report a cyst/ \
lesion finding inline.)

Output ONLY the report text. No markdown, no commentary about your \
process.
"""


# ── vLLM call with retry (text-only, chat-formatted) ──────────────────────

def call_vllm_chat(client, model: str, messages: List[Dict],
                   max_tokens: int = 2048, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            msg = resp.choices[0].message
            content = msg.content
            reasoning = getattr(msg, "reasoning_content", None)
            if content:
                return content
            if reasoning:
                print("    [DEBUG] content empty, using reasoning_content instead", file=sys.stderr)
                return reasoning
            return None
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"    [retry {attempt+1}] {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)


# ── Few-shot examples (optional) ──────────────────────────────────────────

def load_few_shot_examples(examples_dir: Optional[str]) -> List[Dict[str, str]]:
    """
    Load {case_id}_summary.json / {case_id}.txt pairs from a directory to
    use as few-shot examples, if you have any hand-verified (or manually
    corrected) summary->report pairs. Returns a list of
    {"summary": <json str>, "report": <text>} dicts, empty if no directory
    given or none found.

    NOTE: none are bundled with this script by default -- the two example
    reports discussed while designing postprocess_pred.py don't have a
    matching *verified* summary JSON (they were real report text without
    a corresponding real prediction run), so fabricating one to use as a
    few-shot pair risked teaching the model from an input that doesn't
    actually match what postprocess_pred.py would produce. Once you've
    run a few real cases through the pipeline and are happy with (or have
    hand-corrected) their {case_id}_summary.json, drop matching
    {case_id}.txt files next to them and point --examples-dir here.
    """
    if not examples_dir:
        return []
    ex_dir = Path(examples_dir)
    if not ex_dir.exists():
        print(f"[WARN] --examples-dir not found: {examples_dir}", file=sys.stderr)
        return []

    pairs = []
    for summary_file in sorted(ex_dir.glob("*_summary.json")):
        case_id = summary_file.stem.replace("_summary", "")
        report_file = ex_dir / f"{case_id}.txt"
        if not report_file.exists():
            continue
        pairs.append({
            "summary": summary_file.read_text(),
            "report": report_file.read_text().strip(),
        })
    if pairs:
        print(f"[INFO] Loaded {len(pairs)} few-shot example(s) from {examples_dir}", file=sys.stderr)
    return pairs


def build_messages(system: str, few_shot: List[Dict[str, str]], summary_json_str: str) -> List[Dict]:
    """
    system prompt, then each few-shot example as a user/assistant exchange
    (JSON in, report text out), then the real case's JSON as the final
    user turn -- standard few-shot chat formatting.
    """
    messages = [{"role": "system", "content": system}]
    for ex in few_shot:
        messages.append({"role": "user", "content": ex["summary"]})
        messages.append({"role": "assistant", "content": ex["report"]})
    messages.append({"role": "user", "content": summary_json_str})
    return messages


# ── Case processing ────────────────────────────────────────────────────────

def _looks_like_raw_prediction(data: Dict) -> bool:
    """Distinguish a raw {case_id}_pred.json (has 'global'+'teeth') from an
    already-postprocessed {case_id}_summary.json (compact shape)."""
    return "global" in data and "teeth" in data


def process_case(client, model: str, data: Dict, few_shot: List[Dict[str, str]],
                 dry_run: bool = False, debug: bool = True) -> str:
    """
    data: either a raw prediction dict or an already-postprocessed summary
    dict (see _looks_like_raw_prediction). Returns the report text (may be
    "" for an empty prediction).
    """
    if _looks_like_raw_prediction(data):
        if is_empty_prediction(data):
            return ""
        summary = postprocess_prediction(data)
    else:
        summary = data  # already postprocessed

    case_id = summary.get("case_id", "?")
    summary_json_str = json.dumps(summary, indent=2)

    if dry_run:
        return ""

    messages = build_messages(SYSTEM_PROMPT, few_shot, summary_json_str)
    raw = call_vllm_chat(client, model, messages, max_tokens=2048)

    if debug:
        preview = raw[:300] if raw else repr(raw)
        print(f"    [DEBUG] {case_id} raw response ({len(raw) if raw else 0} chars): {preview}",
              file=sys.stderr)

    return (raw or "").strip()


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate narrative CBCT reports with an LLM (Qwen3, text-only) "
                     "from postprocessed prediction summaries.")
    ap.add_argument("--pred-dir", required=True,
                    help="Directory of {case_id}_pred.json (raw) or "
                         "{case_id}_summary.json (postprocessed) files, or a single such file")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for {case_id}.txt report files")
    ap.add_argument("--model", default="qwen3-14b",
                    help="Model name (served-model-name on the vLLM server)")
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1",
                    help="vLLM OpenAI-compatible endpoint")
    ap.add_argument("--examples-dir", default=None,
                    help="Optional directory of {case_id}_summary.json + {case_id}.txt "
                         "pairs to use as few-shot examples (see load_few_shot_examples)")
    ap.add_argument("--case-ids", nargs="+", default=None,
                    help="Filter case IDs")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max cases (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip case if {case_id}.txt already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip vLLM calls (validate structure only)")
    ap.add_argument("--no-debug", action="store_true",
                    help="Suppress raw-response debug logging")
    args = ap.parse_args()

    pred_path = Path(args.pred_dir)
    if pred_path.is_file():
        files = [pred_path]
    else:
        files = sorted(pred_path.glob("*_pred.json")) + sorted(pred_path.glob("*_summary.json"))

    def _case_id_of(p: Path) -> str:
        return p.stem.replace("_pred", "").replace("_summary", "")

    if args.case_ids:
        files = [p for p in files if _case_id_of(p) in args.case_ids]

    if args.limit:
        files = files[: args.limit]

    if not files:
        print(f"[FAIL] No prediction/summary files found under {pred_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] {len(files)} case(s)", file=sys.stderr)

    few_shot = load_few_shot_examples(args.examples_dir)

    if not args.dry_run:
        import openai
        client = openai.OpenAI(base_url=args.vllm_url, api_key="none")
    else:
        client = None
        print("[INFO] DRY-RUN (no vLLM calls)", file=sys.stderr)

    os.makedirs(args.out_dir, exist_ok=True)
    processed, skipped = 0, 0

    for pf in files:
        case_id = _case_id_of(pf)
        out_path = Path(args.out_dir) / f"{case_id}.txt"

        if args.resume and out_path.exists():
            print(f"[skip] {case_id}", file=sys.stderr)
            skipped += 1
            continue

        print(f"[gen] {case_id}", file=sys.stderr)
        try:
            data = json.loads(pf.read_text())
            report = process_case(client, args.model, data, few_shot,
                                  dry_run=args.dry_run, debug=not args.no_debug)
            if not args.dry_run:
                out_path.write_text(report)
                print(f"  -> {out_path} ({len(report)} chars)", file=sys.stderr)
            processed += 1
        except Exception as e:
            print(f"  [FAIL] {case_id}: {e}", file=sys.stderr)

    print(f"\n[INFO] Done: {processed} processed, {skipped} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()