#!/usr/bin/env python3
"""
inference_freetext_renders.py

Arm "report_generation_without_schema": the SAME custom renders the aksssr
schema pipeline reads (panoramic, 3D left/frontal/right, per-tooth composites,
sinus detail), but read as FREE TEXT -- no schema, no enums, no JSON, no
category checklist -- and then merged into one CBCT report.

Two stages, both against one already-running vLLM server:

  1. findings: ONE call per image. Each call gets that image, its caption, and
     "describe what you see here". ~38 calls/case.
  2. merge:    ONE text-only call per case. Input is the concatenated findings;
     output is the report.

What the arm is FOR
-------------------
It is the free-text cell of the image-source x extraction-method design in
experiment_plan.md. Everything the schema arm does to shape the model's
attention -- naming the fact, giving the enum, giving the recognition criteria,
demanding a value for every tooth -- is removed, while image ACCESS is held
identical: the same files, one image per call, in a fixed order. So a score
difference against the schema arm is attributable to the extraction method
rather than to the model having seen different pictures.

GROUND TRUTH MUST NOT REACH THIS SCRIPT
---------------------------------------
The caption sidecars written by the four create_*.py generators normally carry
fragments derived from dataset/<split>/facts/<case>.json, which is derived from
the reference reports: absent teeth, crowns, implants, bridges, canal proximity,
FOV/condyle scope. create_panoramic.py additionally filters the panoramic's
tooth outlines and FDI tags by facts.teeth_present, putting the missing-teeth
answer into the PIXELS.

This script therefore refuses to run against an images dir whose captions still
carry those fragments (--allow-leaky-captions overrides, for deliberate
leakage ablations only). Generate the images with jobs/gen_images_nofacts.sh,
which passes create_panoramic.py --no-facts.

The captions that DO survive are kept, deliberately: static captions are pure
image description (which plane each panel is, which way it faces, what the red
outline does and does not mark), they contain nothing about this
case, and the schema arm receives them too. Dropping them would break the
image-access matching the arm depends on. --no-captions exists as an ablation.

Outputs
-------
    {findings_dir}/{case_id}_findings.json   per-image audit trail
    {findings_dir}/{case_id}_merge.json      merge-call audit trail
    {report_out_dir}/{case_id}.txt           the candidate report

{report_out_dir} holds nothing but .txt, because official_ranking.py globs
*.txt there and intersects the stems with the reference reports. A case with no
.txt is silently dropped from scoring, which would quietly compare this arm on
an easier subset than the others -- so a case that fails still gets an EMPTY
.txt rather than none.

Usage:
    python3 code/inference_freetext_renders.py \
        --images-dir     outputs/report_generation_without_schema_validate/images \
        --findings-dir   outputs/report_generation_without_schema_validate/findings \
        --report-out-dir outputs/report_generation_without_schema_validate/reports \
        --all --resume
"""
import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The single source of truth for filename -> schema image key, shared with
# build_vqa_pairs.py and rebuild_captions.py so the three can never disagree
# about which images a case has. build_vqa_pairs is stdlib-only, so importing
# it here is container-safe.
from build_vqa_pairs import discover_images, discover_captions

# ── Which images this arm reads, and in what order ─────────────────────────
#
# Fixed order so the merge call's notes always arrive in the same anatomical
# sequence, and so two runs of the same case are comparable.
GLOBAL_ORDER = ["panoramic", "3d_left", "3d_frontal", "3d_right",
                "sinus_right_detail", "sinus_left_detail"]

# nasal_cavity_detail is deliberately NOT read. create_sinus_detail.py writes
# it, but schema v6.1 dropped the nasal_cavity fact, so build_vqa_pairs.py
# attaches it to no call and the schema arm never sees it. Giving it to this
# arm would be an image-access difference, which is exactly what this arm is
# designed not to have.
EXCLUDED_KEYS = {"nasal_cavity_detail"}

# Quadrant order, matching report_template.json rules.fdi_order and the FDI
# convention the reference reports use.
TOOTH_FDI_ORDER = (list(range(18, 10, -1)) + list(range(21, 29)) +
                   list(range(38, 30, -1)) + list(range(41, 49)))

_TOOTH_KEY = re.compile(r"^tooth_(?P<fdi>\d{2})_composite$")

# What each image is, in one clause, for the per-image prompt. This is the
# minimum context needed to ask a question at all (the model cannot report on
# "the right maxillary sinus" without being told that is what it is looking
# at) and it names no finding and no finding category.
REGION_BLURB = {
    "panoramic": "a reconstructed panoramic view of the whole dentition",
    "3d_left": "a 3D surface rendering of the jaws, viewed from the patient's left",
    "3d_frontal": "a 3D surface rendering of the jaws, viewed from the front",
    "3d_right": "a 3D surface rendering of the jaws, viewed from the patient's right",
    "sinus_right_detail": "a detail view of the RIGHT maxillary sinus",
    "sinus_left_detail": "a detail view of the LEFT maxillary sinus",
}

# Short label for the notes block handed to the merge call.
NOTE_LABEL = {
    "panoramic": "panoramic",
    "3d_left": "3D view from the left",
    "3d_frontal": "3D view from the front",
    "3d_right": "3D view from the right",
    "sinus_right_detail": "right maxillary sinus",
    "sinus_left_detail": "left maxillary sinus",
}

# ── Prompts ────────────────────────────────────────────────────────────────

FINDINGS_PROMPT = """You are a dental radiologist reading ONE image from a CBCT study.

This image is {blurb}.
{caption_block}{reading_block}
Describe everything clinically notable you can see in THIS image. Where the
image lets you identify a tooth, name it by its FDI number. Report only what is
visible here: say nothing about structures this image does not show, and do not
speculate about the rest of the study. If nothing notable is visible, say so.

Write 1-6 sentences of plain prose. No lists, no headings, no JSON, no labels."""

TOOTH_BLURB = ("a multi-panel CBCT detail view centred on tooth {fdi}, which is "
               "outlined in red")

# Sent with tooth composites only. The captions are pure image description --
# every reading rule lives in the prompt now (see the note above
# create_tooth_detail.TOOTH_STATIC_CAPTION_INTRO) -- and this arm has no
# equivalent of the schema arm's HOW_TO_READ_A_TOOTH, so without this it would
# be told nothing about how to read the tooth at all. Kept to four lines: this
# arm answers in 1-6 sentences of prose, so a full checklist would outweigh
# the question.
TOOTH_READING = """
A natural tooth normally has a bright crown (enamel) and a dark root canal down
the centre of the root; neither is a finding. A BRIGHT area can be enamel or
restorative material, and a DARK one can be the normal root canal or a defect --
tell them apart by brightness, shape and position together.
"""

MERGE_PROMPT = """You are a dental radiologist. Below are your own reading notes from a single
CBCT study, one block per image.

Write the final report for this study.

Use exactly these labels, in this order, one per line, formatted
"<Label>: <findings>". Omit any label your notes say nothing about.
{labels}

Use only what the notes contain. Do not introduce a finding the notes do not
support, do not state the same finding under two labels, and do not add
headings, bullets, numbering, or any closing commentary.

--- NOTES ---
{notes}"""

# The four inline labels code/synthesize_report.py emits, in its order. Taken
# from the corpus, not invented: those are the anatomical block labels the
# reference reports actually use. Matching them keeps this arm's SURFACE FORM
# aligned with the template arm's, so a captioning-metric difference between
# the two reflects content rather than typography.
DEFAULT_SECTION_LABELS = ["Mandible", "Maxilla", "Maxillary sinuses",
                          "Mandibular condyles"]

# Fragments the four generators append to a caption when they are given a facts
# file. Their presence in an images dir means that dir carries ground truth.
# Matched as substrings against every caption; see the module docstring.
LEAK_MARKERS = [
    "listed absent in this case's findings",
    "absent (refer to FDI tags on image)",
    "Crown on tooth",
    "Endosseous implant",
    "endosseous implant",
    "fixed bridge",
    "Fixed bridge present",
    "close to mandibular canal",
    "Acquisition FOV",
    " present; teeth ",
]


def report_section_labels(template_path):
    """
    The section labels for the merge prompt.

    Read from schema/report_template.json so this arm and the template arm
    cannot drift apart silently, but mapped through DEFAULT_SECTION_LABELS
    rather than used verbatim: the template's own section TITLES ("Dentition
    Type", "Periodontal Condition (whole-mouth)") appear in no reference report
    and would cost captioning score, whereas the four labels below are the
    corpus's own. The template read is therefore an assertion that the
    structure still is what this arm assumes -- if someone restructures the
    template, this fails loudly instead of drifting.
    """
    expected = ["dentition_type", "periodontal_condition", "mandible_findings",
                "maxilla_findings", "impacted_teeth_whole_mouth",
                "significant_negative_findings"]
    try:
        template = json.loads(Path(template_path).read_text())
        found = [s.get("id") for s in template.get("sections", [])]
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"[FAIL] could not read report template {template_path}: {e}")
    if found != expected:
        sys.exit(f"[FAIL] {template_path} sections changed since this arm was "
                 f"written.\n  expected: {expected}\n  found:    {found}\n"
                 f"Re-derive DEFAULT_SECTION_LABELS before running the arm, or "
                 f"the two arms' report formats will silently diverge.")
    return list(DEFAULT_SECTION_LABELS)


# ── Image plan ─────────────────────────────────────────────────────────────

def plan_case_images(case_images):
    """
    The ordered [(image_key, path)] this case will be asked about.

    Driven entirely by what is on disk: a case with no panoramic, or a tooth
    with no composite, simply produces no call for it. That matches how the
    schema arm gates its own tooth calls (run_vqa_inference.py's has_image
    check) -- a tooth has no composite only because the segmentation carries no
    label at that FDI, so the two arms skip the same teeth.

    Nothing about a skipped image is mentioned to the model. "This image does
    not exist" is itself a claim about the case, and this arm does not make it.
    """
    plan = []
    for key in GLOBAL_ORDER:
        if key in case_images:
            plan.append((key, case_images[key]))

    by_fdi = {}
    for key, path in case_images.items():
        m = _TOOTH_KEY.match(key)
        if m:
            by_fdi[int(m["fdi"])] = (key, path)
    for fdi in TOOTH_FDI_ORDER:
        if fdi in by_fdi:
            plan.append(by_fdi[fdi])

    return plan


def image_blurb(image_key):
    m = _TOOTH_KEY.match(image_key)
    if m:
        return TOOTH_BLURB.format(fdi=int(m["fdi"]))
    return REGION_BLURB.get(image_key, f"a CBCT view ({image_key})")


def image_reading_block(image_key):
    """TOOTH_READING for a tooth composite, "" for every other image."""
    return TOOTH_READING if _TOOTH_KEY.match(image_key) else ""


def note_label(image_key):
    m = _TOOTH_KEY.match(image_key)
    if m:
        return f"tooth {int(m['fdi'])}"
    return NOTE_LABEL.get(image_key, image_key)


# ── Leakage gate ───────────────────────────────────────────────────────────

def assert_captions_clean(captions_by_case, allow_leaky=False):
    """
    Refuse to run against captions that still carry case findings.

    This is a gate, not a warning: an arm that silently reads leaked ground
    truth produces numbers that look fine and mean nothing, and the leak is
    invisible in every downstream artifact.
    """
    leaks = []
    for case_id, captions in sorted(captions_by_case.items()):
        for image_key, caption in sorted(captions.items()):
            for marker in LEAK_MARKERS:
                if marker in caption:
                    leaks.append((case_id, image_key, marker))
                    break
    if not leaks:
        print(f"[PASS] captions carry no case findings "
              f"({len(captions_by_case)} case(s) checked)")
        return

    print(f"[FAIL] {len(leaks)} caption(s) still carry ground truth derived "
          f"from facts.json:", file=sys.stderr)
    for case_id, image_key, marker in leaks[:10]:
        print(f"         {case_id}/{image_key}: contains {marker!r}", file=sys.stderr)
    if len(leaks) > 10:
        print(f"         ... and {len(leaks) - 10} more", file=sys.stderr)
    print("\n       Regenerate this arm's images facts-free:\n"
          "         sbatch jobs/gen_images_nofacts.sh <split>\n"
          "       or rewrite the sidecars in place:\n"
          "         python3 code/rebuild_captions.py --images-dir <dir> "
          "--static-captions\n"
          "       (--allow-leaky-captions overrides this, for a deliberate "
          "leakage ablation only.)", file=sys.stderr)
    if not allow_leaky:
        sys.exit(1)
    print("[WARN] --allow-leaky-captions given: continuing with leaked ground "
          "truth in the prompts", file=sys.stderr)


# ── vLLM ───────────────────────────────────────────────────────────────────

def call_vllm(api_base, model_name, messages, max_tokens,
              enable_thinking=False, timeout=300):
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # See inference_free_report.py's call_vllm: without this, thinking
        # defaults on and a model that runs out of max_tokens mid-reasoning
        # returns content=None.
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _text_of(response):
    """(content, finish_reason). content is "" when the model returned none."""
    choice = response["choices"][0]
    return (choice["message"].get("content") or ""), choice.get("finish_reason")


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


# ── Stage 1: per-image findings ────────────────────────────────────────────

def run_findings(case_id, plan, captions, args):
    """
    One call per planned image. The findings file is rewritten after EVERY
    call, so a job killed 20 teeth into a case loses one call, not twenty.
    """
    findings_path = Path(args.findings_dir) / f"{case_id}_findings.json"

    record = {
        "case_id": case_id,
        "model": args.model_name,
        "images_dir": str(args.images_dir),
        "image_order": [k for k, _ in plan],
        "captions_used": not args.no_captions,
        "findings": [],
    }
    done = {}
    if args.resume and findings_path.exists():
        try:
            prev = json.loads(findings_path.read_text())
            done = {f["image_key"]: f for f in prev.get("findings", [])
                    if f.get("text") and not f.get("error")}
            record["findings"] = []
        except (OSError, json.JSONDecodeError):
            print(f"  [WARN] {case_id}: unreadable findings file, restarting the case",
                  file=sys.stderr)

    t0 = time.time()
    n_called = 0
    for image_key, image_path in plan:
        if image_key in done:
            record["findings"].append(done[image_key])
            continue

        caption = captions.get(image_key) if not args.no_captions else None
        prompt = FINDINGS_PROMPT.format(
            blurb=image_blurb(image_key),
            caption_block=f"\n{caption}\n" if caption else "",
            reading_block=image_reading_block(image_key))

        entry = {"image_key": image_key, "image_path": str(image_path),
                 "caption": caption, "prompt": prompt,
                 "text": None, "seconds": None, "finish_reason": None,
                 "error": None}

        if args.dry_run:
            entry["dry_run"] = True
            record["findings"].append(entry)
            n_called += 1
            continue

        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_b64(image_path)}"}},
        ]}]

        call_t0 = time.time()
        try:
            response = call_vllm(args.api_base, args.model_name, messages,
                                 args.max_tokens,
                                 enable_thinking=args.enable_thinking,
                                 timeout=args.timeout)
            text, finish = _text_of(response)
            entry["text"] = text.strip()
            entry["finish_reason"] = finish
            if not entry["text"]:
                # Almost always the thinking/max_tokens interaction described
                # in inference_free_report.py -- worth surfacing per image
                # rather than discovering it as an empty report at the end.
                entry["error"] = "empty content"
                print(f"  [WARN] {case_id}/{image_key}: model returned no content "
                      f"(finish_reason={finish}); consider raising --max-tokens",
                      file=sys.stderr)
        except (requests.RequestException, KeyError, IndexError) as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            print(f"  [ERROR] {case_id}/{image_key}: {entry['error']}", file=sys.stderr)
        entry["seconds"] = round(time.time() - call_t0, 2)

        record["findings"].append(entry)
        n_called += 1
        write_json_atomic(findings_path, record)

    record["n_calls"] = n_called
    record["n_resumed"] = len(plan) - n_called
    record["total_seconds"] = round(time.time() - t0, 2)
    write_json_atomic(findings_path, record)
    return record


# ── Stage 2: merge into one report ─────────────────────────────────────────

def build_notes(record, placeholder=False):
    """
    The notes block: one labelled paragraph per image that said something.

    placeholder=True substitutes a stand-in for each image's text, so --dry-run
    can show the real merge prompt (labels, ordering, per-image framing) when
    no findings have actually been generated yet. Without it the dry run would
    silently skip the merge prompt, which is the prompt most worth eyeballing.
    """
    blocks = []
    for f in record.get("findings", []):
        text = (f.get("text") or "").strip()
        if not text:
            if not placeholder:
                continue
            text = "<findings for this image would appear here>"
        blocks.append(f"[{note_label(f['image_key'])}] {text}")
    return "\n".join(blocks)


def run_merge(case_id, record, labels, args):
    notes = build_notes(record, placeholder=args.dry_run)
    report_path = Path(args.report_out_dir) / f"{case_id}.txt"

    if not notes:
        # No image said anything. An empty report scores 0 for this case; a
        # MISSING file would drop the case from official_ranking's intersection
        # and quietly score this arm on an easier subset than the others.
        print(f"  [WARN] {case_id}: no usable findings; writing an empty report",
              file=sys.stderr)
        if not args.dry_run:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("", encoding="utf-8")
        return ""

    prompt = MERGE_PROMPT.format(
        labels="\n".join(f"  {label}" for label in labels), notes=notes)

    merge_record = {"case_id": case_id, "model": args.model_name,
                    "labels": labels, "prompt": prompt,
                    "n_note_blocks": len(notes.splitlines()),
                    "report": None, "seconds": None, "error": None}

    if args.dry_run:
        merge_record["dry_run"] = True
        write_json_atomic(Path(args.findings_dir) / f"{case_id}_merge.json",
                          merge_record)
        head, _, tail = prompt.partition("--- NOTES ---")
        print(f"\n----- merge prompt for {case_id} (notes truncated) -----")
        print(head + "--- NOTES ---")
        print("\n".join(tail.strip().splitlines()[:3]))
        print(f"... [{len(tail.strip().splitlines())} note block(s) total]")
        print("-" * 58)
        return None

    t0 = time.time()
    report = ""
    try:
        response = call_vllm(args.api_base, args.model_name,
                             [{"role": "user", "content": prompt}],
                             args.merge_max_tokens,
                             enable_thinking=args.enable_thinking,
                             timeout=args.timeout)
        text, finish = _text_of(response)
        report = text.strip()
        merge_record["finish_reason"] = finish
        merge_record["raw_response"] = response
        if not report:
            merge_record["error"] = "empty content"
            print(f"  [WARN] {case_id}: merge returned no content "
                  f"(finish_reason={finish})", file=sys.stderr)
    except (requests.RequestException, KeyError, IndexError) as e:
        merge_record["error"] = f"{type(e).__name__}: {e}"
        print(f"  [ERROR] {case_id}: merge failed: {merge_record['error']}",
              file=sys.stderr)

    merge_record["report"] = report
    merge_record["seconds"] = round(time.time() - t0, 2)
    write_json_atomic(Path(args.findings_dir) / f"{case_id}_merge.json", merge_record)

    # Written even when empty -- see the no-notes branch above.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report


# ── Driver ─────────────────────────────────────────────────────────────────

def run_case(case_id, case_images, captions, labels, args):
    plan = plan_case_images(case_images)
    if not plan:
        print(f"[WARN] {case_id}: no readable images under {args.images_dir}; "
              f"writing an empty report", file=sys.stderr)
        if not args.dry_run:
            Path(args.report_out_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.report_out_dir) / f"{case_id}.txt").write_text("", encoding="utf-8")
        return

    n_teeth = sum(1 for k, _ in plan if _TOOTH_KEY.match(k))
    print(f"[{case_id}] {len(plan)} image(s): {len(plan) - n_teeth} region + "
          f"{n_teeth} tooth composite(s)")

    findings_path = Path(args.findings_dir) / f"{case_id}_findings.json"
    if args.stage == "merge":
        if not findings_path.exists():
            print(f"  [SKIP] {case_id}: --stage merge but no {findings_path.name}",
                  file=sys.stderr)
            return
        record = json.loads(findings_path.read_text())
    else:
        record = run_findings(case_id, plan, captions, args)

    if args.stage == "findings":
        return

    t0 = time.time()
    run_merge(case_id, record, labels, args)
    n_ok = sum(1 for f in record.get("findings", []) if (f.get("text") or "").strip())
    print(f"[{case_id}] done: {n_ok}/{len(plan)} image(s) described"
          + (f", report -> {args.report_out_dir}/{case_id}.txt"
             if not args.dry_run else " (dry run)")
          + f" ({time.time() - t0:.1f}s merge)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True,
                    help="This arm's generated images (facts-free -- see the "
                         "module docstring), e.g. outputs/<run>_<split>/images")
    ap.add_argument("--findings-dir", required=True,
                    help="Where the per-image and merge audit JSONs go")
    ap.add_argument("--report-out-dir", required=True,
                    help="Where {case_id}.txt candidate reports go. Keep this "
                         "directory free of any other .txt: official_ranking.py "
                         "globs *.txt here.")
    ap.add_argument("--report-template", default="schema/report_template.json",
                    help="Read only to assert the section structure this arm's "
                         "merge prompt assumes is still current")

    ap.add_argument("--case-id", help="Run a single case")
    ap.add_argument("--case-ids", nargs="+", default=None,
                    help="Run exactly these cases. Prefer this over --all when "
                         "the caller already knows which cases it built.")
    ap.add_argument("--all", action="store_true",
                    help="Run every case with images under --images-dir")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only the first N cases (smoke testing)")

    ap.add_argument("--stage", choices=("findings", "merge", "both"), default="both",
                    help="'merge' re-runs only the report call from existing "
                         "findings JSONs -- prompt iteration with no image calls")
    ap.add_argument("--resume", action="store_true",
                    help="Skip cases with a non-empty {case_id}.txt, and within a "
                         "case skip images already described in its findings JSON")

    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--model-name", default="qwen3.5-vl",
                    help="Must match --served-model-name used to start vLLM")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="Per-image findings call")
    ap.add_argument("--merge-max-tokens", type=int, default=1536,
                    help="The report-writing call")
    ap.add_argument("--timeout", type=int, default=300, help="Per-call HTTP timeout")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Off by default -- see inference_free_report.py's help")

    ap.add_argument("--no-captions", action="store_true",
                    help="Ablation: omit the static caption from the per-image "
                         "prompt. Note the schema arm always receives it, so this "
                         "breaks image-access matching -- for ablations only.")
    ap.add_argument("--allow-leaky-captions", action="store_true",
                    help="Run even when captions still carry facts.json-derived "
                         "findings. For a deliberate leakage ablation only; the "
                         "resulting scores are not a measure of perception.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan every call and print the prompts; make no HTTP "
                         "calls and write no reports")
    args = ap.parse_args()

    if not args.case_id and not args.case_ids and not args.all:
        ap.error("Specify one of --case-id <id>, --case-ids <id> [<id> ...], or --all")

    labels = report_section_labels(args.report_template)

    images_by_case = discover_images(args.images_dir)
    if not images_by_case:
        sys.exit(f"[FAIL] no generated images discovered under {args.images_dir}. "
                 f"Run jobs/gen_images_nofacts.sh <split> first.")
    captions_by_case = discover_captions(args.images_dir) if not args.no_captions else {}

    if args.case_id:
        case_ids = [args.case_id]
    elif args.case_ids:
        case_ids = list(args.case_ids)
    else:
        case_ids = sorted(images_by_case)
    if args.limit:
        case_ids = case_ids[:args.limit]

    missing = [c for c in case_ids if c not in images_by_case]
    if missing:
        print(f"[WARN] no images found for: {missing}", file=sys.stderr)

    if not args.no_captions:
        assert_captions_clean({c: captions_by_case.get(c, {}) for c in case_ids},
                              allow_leaky=args.allow_leaky_captions)

    os.makedirs(args.findings_dir, exist_ok=True)
    os.makedirs(args.report_out_dir, exist_ok=True)

    print(f"[INFO] images    : {args.images_dir}")
    print(f"[INFO] cases     : {len(case_ids)}")
    print(f"[INFO] stage     : {args.stage}"
          + ("  (DRY RUN -- no calls, no reports)" if args.dry_run else ""))
    print(f"[INFO] labels    : {labels}")

    for case_id in case_ids:
        report_path = Path(args.report_out_dir) / f"{case_id}.txt"
        if args.resume and args.stage == "both" and report_path.exists() \
                and report_path.read_text().strip():
            print(f"[SKIP] {case_id}: report already exists (--resume)")
            continue
        try:
            run_case(case_id, images_by_case.get(case_id, {}),
                     captions_by_case.get(case_id, {}), labels, args)
        except Exception as e:
            # One bad case must not cost every later case its report.
            print(f"[FAIL] {case_id}: {type(e).__name__}: {e}", file=sys.stderr)

    n_reports = len(list(Path(args.report_out_dir).glob("*.txt")))
    print(f"\n[INFO] reports in {args.report_out_dir}: {n_reports}")
    if not args.dry_run and n_reports < len(case_ids):
        print(f"[WARN] {len(case_ids) - n_reports} case(s) produced NO report file. "
              f"official_ranking.py drops those cases entirely, so this arm would "
              f"be scored on a different subset than the others. Re-run with "
              f"--resume.", file=sys.stderr)


if __name__ == "__main__":
    main()
