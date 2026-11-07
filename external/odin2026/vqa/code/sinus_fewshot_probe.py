#!/usr/bin/env python3
"""
sinus_fewshot_probe.py — does few-shot cut the sinus over-call?

WHY THIS EXISTS
---------------
The VLM over-calls findings, and `maxilla_sinus_{right,left}` is the cheapest
place to test a fix: one self-contained call, one image, and a hand-curated
label table already in the repo (survey_findings.SINUS_MUCOSA_GT — 14 cases,
26 sides). Measured on the v5 validate predictions, that table gives
17 correct / 4 false-positive / 4 false-negative / 1 unanswered.

This script runs the SAME single sinus call the pipeline runs, with one
difference: `--arm fewshot` prepends a handful of solved sinus cases as
user/assistant turns before the real question. Everything else — system
prompt, user template, caption placement, guided_json — is imported from
run_vqa_inference / build_vqa_pairs rather than restated, so the only variable
between the two arms is the exemplar turns.

THE CONFOUND THIS PROBE HAS TO CLEAR FIRST
------------------------------------------
The sinus caption says the sinus "is outlined in bright green" and to "judge
only what is inside the green outline" — but no image on disk has ever had
that outline. create_sinus_detail.py grew the green ROI in commit cb62085,
while every rendered PNG predates it (the pipeline's per-step skip check saw
the old files and moved on), and the captions were rebuilt afterwards without
re-rendering. So the model was being told to read a marker that is not in the
picture, with the nasal cavity and the mandible sharing the crop.

Hence three arms, not two:
  A0  stale images, zero-shot   — the v5 predictions, already scored
  A1  re-rendered images, zero-shot  — isolates the image fix
  A2  re-rendered images, few-shot   — isolates the exemplars
Point --images-dir at a directory rendered by the CURRENT create_sinus_detail
(see jobs/sinus_fewshot_probe.sh STAGE=render); it never writes there itself.

The image budget matters: 3 exemplars + 1 query = 4 images in one prompt, and
the pipeline's vLLM server is started with --limit-mm-per-prompt {"image": 3}.
The job script raises it. A 4-image prompt that exceeds the server's limit is
rejected outright, so this fails loudly rather than silently dropping images.

Usage
-----
    # exact messages for one side, no model needed (base64 redacted to paths)
    python3 code/sinus_fewshot_probe.py --arm fewshot \
        --exemplars code/sinus_fewshot_examples.json \
        --dump-prompt /tmp/msgs.json

    # arm A1 / A2 (inside the container, against the VL server)
    python3 code/sinus_fewshot_probe.py --arm zeroshot \
        --images-dir /project/outputs/sinus_probe/images \
        --out-dir  /project/outputs/sinus_probe/runs/zeroshot

    # scoring only — no GPU, no model, runs on the login node
    python3 code/sinus_fewshot_probe.py --arm zeroshot --score
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Prompt text and transport, imported (never restated) so this probe cannot
# drift from what the pipeline actually sends.
from run_vqa_inference import (  # noqa: E402
    CATEGORY_SYSTEM, CATEGORY_USER_TEMPLATE,
    build_user_blocks, call_vllm_messages, parse_json,
)
from build_vqa_pairs import (  # noqa: E402
    GLOBAL_SECTIONS, resolve_call_plan, build_questions_block,
)
from survey_findings import SINUS_MUCOSA_GT  # noqa: E402

# CALL_PLAN's key for each side's call, and the fact/image names hanging off
# it. Taken from the plan rather than spelled out, so a rename in
# build_vqa_pairs.CALL_PLAN surfaces here as a KeyError instead of a silently
# empty run.
CALL_KEY = {"right": "sinus_right", "left": "sinus_left"}
IMAGE_KEY = {"right": "sinus_right_detail", "left": "sinus_left_detail"}
FIELD = {"right": "maxilla_sinus_right", "left": "maxilla_sinus_left"}

CAPTION_SIDECAR = "{case}_sinus_captions.json"
IMAGE_FILE = "{case}_sinus_{side}_detail.png"

PROJECT_DIR = Path(__file__).resolve().parent.parent


# ── Prompt assembly ───────────────────────────────────────────────────────

def load_sinus_blocks(schema_path: str) -> Dict[str, dict]:
    """
    Render each side's questions / output_schema / json_schema / guidance
    straight from schema.json, through the same builders build_vqa_pairs.py
    uses. resolve_call_plan() is called for its validation side effect: it
    hard-fails if the sinus facts and their images_needed have drifted apart,
    which is exactly the drift that would make this probe measure nothing.
    """
    with open(schema_path) as f:
        schema = json.load(f)

    definitions = schema.get("_definitions", {}) or {}
    visual_evidence_rule = schema.get("_visual_evidence_rule")
    facts_by_id = {f["fact_id"]: f
                   for section in GLOBAL_SECTIONS
                   for f in schema.get(section, []) or []}

    calls = resolve_call_plan(schema)

    blocks = {}
    for side, call_key in CALL_KEY.items():
        images_tuple, call_facts = calls[call_key]
        if tuple(images_tuple) != (IMAGE_KEY[side],):
            raise ValueError(
                f"call '{call_key}' now needs {list(images_tuple)}, not "
                f"['{IMAGE_KEY[side]}'] -- this probe sends one image per turn")
        blocks[side] = build_questions_block(
            call_facts, definitions=definitions,
            visual_evidence_rule=visual_evidence_rule, facts_by_id=facts_by_id)
    return blocks


def image_path(images_dir: str, case: str, side: str) -> str:
    return os.path.join(images_dir, IMAGE_FILE.format(case=case, side=side))


def load_caption(images_dir: str, case: str, side: str) -> Optional[str]:
    """
    The caption the renderer wrote NEXT TO the image, not a rebuilt one --
    reading the sidecar is what guarantees the caption describes the pixels
    that are actually being sent (the whole point of the confound above).
    """
    sidecar = Path(images_dir) / CAPTION_SIDECAR.format(case=case)
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text()).get(IMAGE_KEY[side])
    except (json.JSONDecodeError, OSError):
        return None


def build_user_turn(images_dir: str, case: str, side: str, block: dict,
                    redact: bool = False) -> Optional[dict]:
    """
    One user turn: [caption text, image, rendered question] -- built by
    run_vqa_inference.build_user_blocks(), the same function infer_call() uses,
    so the turn is byte-identical to a production sinus call.

    Returns None when the side has no image on disk (create_sinus_detail skips
    a side with under 10 sinus voxels), which the caller records as no_image
    rather than as a wrong answer.

    redact=True swaps the base64 payload for the file path so --dump-prompt
    produces something readable.
    """
    path = image_path(images_dir, case, side)
    key = IMAGE_KEY[side]
    caption = load_caption(images_dir, case, side)
    content = build_user_blocks({key: path}, {key: caption or ""},
                                block, CATEGORY_USER_TEMPLATE)
    if content is None:
        return None
    if redact:
        for b in content:
            if b.get("type") == "image_url":
                b["image_url"] = {"url": f"<file: {path}>"}
    return {"role": "user", "content": content}


def build_messages(images_dir: str, case: str, side: str, blocks: Dict[str, dict],
                   exemplars: List[dict], exemplar_images_dir: str,
                   redact: bool = False) -> Optional[List[dict]]:
    """
    system → [exemplar user, exemplar assistant] × N → the real user turn.

    Each exemplar is built by the same build_user_turn() as the real question
    and answered with ITS OWN side's field name, so a right-side exemplar in
    front of a left-side question stays internally consistent. guided_json
    constrains only the generated turn, so the exemplar answers pass through
    verbatim.
    """
    target = build_user_turn(images_dir, case, side, blocks[side], redact=redact)
    if target is None:
        return None

    messages: List[dict] = [{"role": "system", "content": CATEGORY_SYSTEM}]
    for ex in exemplars:
        ex_side = ex["side"]
        turn = build_user_turn(exemplar_images_dir, ex["case"], ex_side,
                               blocks[ex_side], redact=redact)
        if turn is None:
            raise FileNotFoundError(
                f"exemplar {ex['case']} {ex_side}: no image at "
                f"{image_path(exemplar_images_dir, ex['case'], ex_side)} -- "
                f"render it before running the fewshot arm")
        messages.append(turn)
        messages.append({"role": "assistant",
                         "content": json.dumps({FIELD[ex_side]: ex["answer"]}, indent=2)})
    messages.append(target)
    return messages


def load_exemplars(path: Optional[str]) -> List[dict]:
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    exemplars = data["exemplars"] if isinstance(data, dict) else data
    for ex in exemplars:
        missing = {"case", "side", "answer"} - set(ex)
        if missing:
            raise ValueError(f"exemplar {ex!r} is missing {sorted(missing)}")
        if ex["side"] not in CALL_KEY:
            raise ValueError(f"exemplar {ex['case']}: bad side {ex['side']!r}")
    return exemplars


# ── Running one arm ───────────────────────────────────────────────────────

def run_arm(client, model: str, cases: List[str], sides: List[str],
            images_dir: str, blocks: Dict[str, dict], out_dir: str,
            exemplars: List[dict], exemplar_images_dir: str,
            max_tokens: int = 2048, resume: bool = False) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n_done = 0

    for case in cases:
        for side in sides:
            out_path = Path(out_dir) / f"{case}_{side}.json"
            if resume and out_path.exists():
                print(f"  [skip] {case} {side}: already answered", file=sys.stderr)
                continue

            messages = build_messages(images_dir, case, side, blocks,
                                      exemplars, exemplar_images_dir)
            if messages is None:
                out_path.write_text(json.dumps(
                    {"case_id": case, "side": side, "status": "no_image"}, indent=2))
                print(f"  [{case} {side}] no image -- skipped", file=sys.stderr)
                continue

            record = {"case_id": case, "side": side,
                      "n_exemplars": len(exemplars), "status": "ok"}
            try:
                raw = call_vllm_messages(client, model, messages,
                                         max_tokens=max_tokens,
                                         json_schema=blocks[side].get("json_schema"))
                record["raw"] = raw
                record["answer"] = parse_json(raw).get(FIELD[side])
            except Exception as e:
                record["status"] = "error"
                record["error"] = f"{type(e).__name__}: {e}"
                print(f"  [ERROR] {case} {side}: {record['error']}", file=sys.stderr)

            out_path.write_text(json.dumps(record, indent=2))
            n_done += 1
            ans = record.get("answer") or {}
            print(f"  [{case} {side}] mucosa={ans.get('mucosa_state')} "
                  f"scope={ans.get('scope')} content={ans.get('sinus_content')} "
                  f"teeth={ans.get('intrasinusal_teeth')}", file=sys.stderr)

    print(f"[INFO] {n_done} call(s) written -> {out_dir}", file=sys.stderr)


# ── Scoring ───────────────────────────────────────────────────────────────

def score_arm(out_dir: str, cases: List[str], sides: List[str],
              exemplar_cases: Optional[set] = None) -> dict:
    """
    Score against SINUS_MUCOSA_GT, keeping FALSE POSITIVES in their own bucket.
    survey_findings.py folds them into a generic "wrong" count, and a false
    positive is the number this experiment is about: the report gains a whole
    sinus paragraph the patient does not have.
    """
    exemplar_cases = exemplar_cases or set()
    rows, totals = [], {"correct": 0, "false_positive": 0, "false_negative": 0,
                        "unanswered": 0, "no_image": 0, "leaked_exemplar": 0}

    for case in cases:
        gt_sides = SINUS_MUCOSA_GT.get(case, {})
        for side in sides:
            expected = gt_sides.get(side)
            if expected is None:
                continue                      # no hand label for this side
            if case in exemplar_cases:
                totals["leaked_exemplar"] += 1
                rows.append({"case": case, "side": side, "expected": expected,
                             "got": None, "verdict": "leaked_exemplar"})
                continue

            path = Path(out_dir) / f"{case}_{side}.json"
            got, verdict = None, "unanswered"
            if path.exists():
                rec = json.loads(path.read_text())
                if rec.get("status") == "no_image":
                    verdict = "no_image"
                else:
                    got = (rec.get("answer") or {}).get("mucosa_state")
                    if got is None:
                        verdict = "unanswered"
                    elif got == expected:
                        verdict = "correct"
                    else:
                        verdict = ("false_positive" if expected == "normal"
                                   else "false_negative")
            totals[verdict] += 1
            rows.append({"case": case, "side": side, "expected": expected,
                         "got": got, "verdict": verdict})

    scored = totals["correct"] + totals["false_positive"] + totals["false_negative"]
    n_normal = sum(1 for r in rows if r["expected"] == "normal"
                   and r["verdict"] in ("correct", "false_positive"))
    n_abnormal = sum(1 for r in rows if r["expected"] == "thickening"
                     and r["verdict"] in ("correct", "false_negative"))
    return {
        "out_dir": str(out_dir),
        "totals": totals,
        "n_scored": scored,
        "accuracy": round(totals["correct"] / scored, 4) if scored else None,
        # The over-call rate: of the sides that really are normal, how many did
        # the model call thickened.
        "false_positive_rate": (round(totals["false_positive"] / n_normal, 4)
                                if n_normal else None),
        "abnormal_recall": (round((n_abnormal - totals["false_negative"]) / n_abnormal, 4)
                            if n_abnormal else None),
        "rows": rows,
    }


def print_score(score: dict) -> None:
    print(f"\n{'case':6s} {'side':6s} {'expected':10s} {'got':12s} verdict")
    print("-" * 52)
    for r in score["rows"]:
        print(f"{r['case']:6s} {r['side']:6s} {r['expected']:10s} "
              f"{str(r['got']):12s} {r['verdict']}")
    t = score["totals"]
    print("-" * 52)
    print(f"scored {score['n_scored']}  correct {t['correct']}  "
          f"FP {t['false_positive']}  FN {t['false_negative']}  "
          f"unanswered {t['unanswered']}  no_image {t['no_image']}")
    print(f"accuracy {score['accuracy']}  FP-rate(on normal sides) "
          f"{score['false_positive_rate']}  abnormal-recall {score['abnormal_recall']}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["zeroshot", "fewshot"], required=True)
    ap.add_argument("--images-dir", default=str(PROJECT_DIR / "outputs/sinus_probe/images"),
                    help="Sinus images rendered by the CURRENT create_sinus_detail.py")
    ap.add_argument("--exemplar-images-dir", default=None,
                    help="Where the exemplars' images live (default: --images-dir)")
    ap.add_argument("--out-dir", default=None,
                    help="Default: outputs/sinus_probe/runs/{arm}")
    ap.add_argument("--schema", default=str(PROJECT_DIR / "schema/schema.json"))
    ap.add_argument("--exemplars", default=None,
                    help="JSON file of few-shot exemplars (required for --arm fewshot)")
    ap.add_argument("--cases", nargs="+", default=None,
                    help="Default: every case in survey_findings.SINUS_MUCOSA_GT")
    ap.add_argument("--sides", nargs="+", choices=["right", "left"],
                    default=["right", "left"])
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3.5-vl")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dump-prompt", default=None, metavar="PATH",
                    help="Write the exact message list for the first target and "
                         "exit -- no model call, image payloads shown as paths")
    ap.add_argument("--score", action="store_true",
                    help="Score an existing --out-dir and exit (no model needed)")
    args = ap.parse_args()

    out_dir = args.out_dir or str(PROJECT_DIR / "outputs/sinus_probe/runs" / args.arm)
    exemplar_images_dir = args.exemplar_images_dir or args.images_dir
    cases = args.cases or sorted(SINUS_MUCOSA_GT)

    if args.arm == "fewshot" and not args.exemplars:
        ap.error("--arm fewshot needs --exemplars")
    exemplars = load_exemplars(args.exemplars if args.arm == "fewshot" else None)
    exemplar_cases = {ex["case"] for ex in exemplars}

    if args.score:
        score = score_arm(out_dir, cases, args.sides, exemplar_cases)
        print_score(score)
        score_path = Path(out_dir).parent.parent / f"score_{args.arm}.json"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(score, indent=2))
        print(f"\n[INFO] -> {score_path}")
        return

    blocks = load_sinus_blocks(args.schema)
    print(f"[INFO] arm={args.arm}  cases={len(cases)}  sides={args.sides}  "
          f"exemplars={len(exemplars)}", file=sys.stderr)
    print(f"[INFO] images: {args.images_dir}", file=sys.stderr)

    if args.dump_prompt:
        case, side = cases[0], args.sides[0]
        messages = build_messages(args.images_dir, case, side, blocks,
                                  exemplars, exemplar_images_dir, redact=True)
        if messages is None:
            print(f"[FAIL] {case} {side}: no image at "
                  f"{image_path(args.images_dir, case, side)}", file=sys.stderr)
            sys.exit(1)
        n_images = sum(1 for m in messages for b in m["content"]
                       if isinstance(b, dict) and b.get("type") == "image_url")
        Path(args.dump_prompt).write_text(json.dumps(
            {"arm": args.arm, "case": case, "side": side,
             "n_messages": len(messages), "n_images": n_images,
             "json_schema": blocks[side].get("json_schema"),
             "messages": messages}, indent=2))
        print(f"[INFO] {case} {side}: {len(messages)} message(s), {n_images} image(s) "
              f"-> {args.dump_prompt}", file=sys.stderr)
        return

    import openai
    client = openai.OpenAI(base_url=args.vllm_url, api_key="none")
    run_arm(client, args.model, cases, args.sides, args.images_dir, blocks,
            out_dir, exemplars, exemplar_images_dir,
            max_tokens=args.max_tokens, resume=args.resume)

    score = score_arm(out_dir, cases, args.sides, exemplar_cases)
    print_score(score)
    score_path = Path(out_dir).parent.parent / f"score_{args.arm}.json"
    score_path.write_text(json.dumps(score, indent=2))
    print(f"\n[INFO] -> {score_path}")


if __name__ == "__main__":
    main()
