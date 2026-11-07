#!/usr/bin/env python3
"""
inference_free_report.py

Task B baseline: uniform-32 slices -> Qwen3.5-VL (served via vLLM,
OpenAI-compatible endpoint) -> free-text clinical report.

Assumes the vLLM server is already up (see gen_report_baseline1.sh), served
with --served-model-name qwen3.5-vl, matching the existing project convention.

Reads the per-case output of uniform-32_selection.py:
    {slices_dir}/{case_id}/{case_id}_slice{00..31}.png
    {slices_dir}/{case_id}/{case_id}_slices_meta.json

Writes:
    {out_dir}/{case_id}_free_text_report.json
        {
          "case_id": ...,
          "model": "qwen3.5-vl",
          "num_slices": 32,
          "prompt": "...",
          "report": "<free text>",
          "raw_response": {...}   # full API response for audit/debug
        }

Usage (single case):
    python inference_free_report.py --slices-dir outputs/uniform32 --case-id A004 --out-dir outputs/reports_baseline1

Usage (all cases found under slices-dir):
    python inference_free_report.py --slices-dir outputs/uniform32 --all --out-dir outputs/reports_baseline1
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

DEFAULT_PROMPT = (
    "You are a dental radiology assistant.\n"
    "Based on the CBCT images, generate a clinically structured report.\n"
    "Include findings, tooth numbers, anatomical locations, severity, impression,\n"
    "and recommendations when supported by the images.\n"
    "Do not invent unsupported findings."
)


def find_cases(slices_dir):
    slices_dir = Path(slices_dir)
    cases = []
    for meta_path in sorted(slices_dir.glob("*/*_slices_meta.json")):
        case_id = meta_path.stem.replace("_slices_meta", "")
        cases.append((case_id, meta_path.parent))
    return cases


def load_case_images(case_dir, meta):
    """Return list of base64-encoded PNGs, ordered by slice_index (preserves
    axial order -- important since the model has no other way to know order)."""
    b64_images = []
    for rec in sorted(meta["slices"], key=lambda r: r["slice_index"]):
        img_path = case_dir / rec["filename"]
        with open(img_path, "rb") as f:
            b64_images.append(base64.b64encode(f.read()).decode("utf-8"))
    return b64_images


def build_messages(prompt, b64_images):
    content = [{"type": "text", "text": prompt}]
    for b64 in b64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return [{"role": "user", "content": content}]


def call_vllm(api_base, model_name, messages, max_tokens, enable_thinking=False, timeout=300):
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # Current vLLM controls Qwen3-style thinking per-request via
        # chat_template_kwargs, not a server startup flag (the old
        # --enable_thinking flag this vLLM version rejects). Without this,
        # the model burns its whole max_tokens budget on reasoning_content
        # and "content" comes back None -- which is what crashed this job.
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def run_case(case_id, case_dir, args):
    meta_path = case_dir / f"{case_id}_slices_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    b64_images = load_case_images(case_dir, meta)
    if len(b64_images) != meta["num_slices"]:
        print(f"[WARN] {case_id}: expected {meta['num_slices']} slices, found {len(b64_images)}", file=sys.stderr)

    messages = build_messages(args.prompt, b64_images)

    t0 = time.time()
    response = call_vllm(args.api_base, args.model_name, messages, args.max_tokens,
                          enable_thinking=args.enable_thinking)
    elapsed = time.time() - t0

    try:
        message = response["choices"][0]["message"]
        report_text = message["content"]
    except (KeyError, IndexError) as e:
        print(f"[ERROR] {case_id}: could not parse response ({e}): {response}", file=sys.stderr)
        report_text = None
        message = {}

    if report_text is None:
        reasoning = message.get("reasoning_content")
        print(f"[WARN] {case_id}: model returned content=None "
              f"(likely ran out of max_tokens while thinking"
              f"{' -- reasoning_content was present, content was not' if reasoning else ''}). "
              f"Writing empty report; consider raising --max-tokens or checking "
              f"--enable-thinking.", file=sys.stderr)
        report_text = ""

    out = {
        "case_id": case_id,
        "model": args.model_name,
        "num_slices": len(b64_images),
        "prompt": args.prompt,
        "report": report_text,
        "inference_seconds": elapsed,
        "raw_response": response,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}_free_text_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Plain-text candidate report, for official_ranking.py / evaluation.sh,
    # which expect {case_id}.txt (same convention synthesize_report.py uses
    # for the schema baseline) -- NOT the JSON audit trail above.
    if args.txt_out_dir:
        txt_out_dir = Path(args.txt_out_dir)
        txt_out_dir.mkdir(parents=True, exist_ok=True)
        (txt_out_dir / f"{case_id}.txt").write_text(report_text, encoding="utf-8")

    print(f"[OK] {case_id}: report written to {out_path} ({elapsed:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slices-dir", required=True, help="Root dir containing per-case slice folders (uniform-32_selection.py output)")
    ap.add_argument("--out-dir", required=True, help="Where to write free-text report JSONs (audit trail)")
    ap.add_argument("--txt-out-dir", default=None,
                     help="Where to also write plain {case_id}.txt candidate reports, for "
                          "official_ranking.py/evaluation.sh. If omitted, only the JSON audit "
                          "trail in --out-dir is written.")
    ap.add_argument("--case-id", help="Run a single case")
    ap.add_argument("--all", action="store_true", help="Run every case found under --slices-dir")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N cases (smoke testing)")
    ap.add_argument("--api-base", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible API base URL")
    ap.add_argument("--model-name", default="qwen3.5-vl", help="Must match --served-model-name used to start vLLM")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--enable-thinking", action="store_true",
                     help="Let the model use visible reasoning (reasoning_content) before "
                          "answering. Off by default -- with thinking on, a model that runs "
                          "out of --max-tokens mid-reasoning returns content=None.")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    if not args.case_id and not args.all:
        ap.error("Specify either --case-id <id> or --all")

    if args.all:
        cases = find_cases(args.slices_dir)
        if not cases:
            print(f"[ERROR] No cases found under {args.slices_dir}", file=sys.stderr)
            sys.exit(1)
        if args.limit:
            cases = cases[:args.limit]
        for case_id, case_dir in cases:
            run_case(case_id, case_dir, args)
    else:
        case_dir = Path(args.slices_dir) / args.case_id
        run_case(args.case_id, case_dir, args)


if __name__ == "__main__":
    main()