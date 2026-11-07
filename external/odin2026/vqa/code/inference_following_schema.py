#!/usr/bin/env python3
"""
inference_following_schema.py

Task A baseline (comparison roadmap): uniform-32 slices -> Qwen3.5-VL ->
structured predictions built from schema.json, written as {case_id}_pred.json
in the SAME shape run_vqa_inference.py produces:

    {"case_id": "...", "global": {<output_field>: value, ...},
     "teeth": {"tooth_11": {<output_field>: value, ..., "detected": "yes"|"no"},
               "tooth_12": {...}, ...}}

so code/postprocess_pred.py + code/generate_report_from_pred.py (the same
predictions -> postprocess -> LLM report pipeline aksssr_pipeline.sh now
uses) can be run unchanged on this arm's output.

CALL STRUCTURE -- two modes, chosen with --call-mode:

  single (default, unchanged): ONE combined call per case covering every
    mandible/maxilla fact plus every tooth fact for every FDI in --fdi-list.
    The original design, on the reasoning that every fact sees the SAME 32
    uniform slices, so splitting adds API cost without adding signal --
    unlike aksssr_pipeline.sh, where each call is scoped to the images a
    group of facts actually needs.

  split: ONE call for the global (mandible+maxilla) facts, then ONE CALL PER
    FDI for the six dental_elements facts. 33 calls/case, every one of them
    receiving the same full 32-slice stack -- there is still no anatomy-aware
    slice selection, which is the point of this arm.

    The reason to split is not signal, it's completion: a single call has to
    emit ~200 fields, six of which per tooth are free-text visual_evidence
    strings, and that does not fit a sane --max-tokens. A truncated JSON is
    not a weaker answer, it's no answer for every tooth after the cutoff.
    Split mode also puts the images BEFORE the text in each message, so
    vLLM's --enable-prefix-caching can reuse the ~29k-token image prefix
    across all 33 calls of a case; single mode keeps the original text-first
    order so its prompt stays byte-identical.

    Per-call results are written to --partial-dir as they arrive, so
    --resume restarts a half-finished case at the call it died on rather
    than at the case boundary.

schema.json (v6.1) shape this script actually reads:
    {
      "mandible":        [ {"fact_id": "...", "question_text": "...",
                            "description": "...", "type": "...",
                            "output_field": "...", "object_fields": {...}?,
                            "enum_values": [...]?}, ... ],
      "maxilla":         [ ... same shape ... ],
      "dental_elements": [ ... same shape, but fact_id/question_text/output_field
                            all contain a "{fdi}" placeholder filled in once per
                            FDI in --fdi-list, e.g. output_field
                            "tooth_{fdi}_morphology" -> "tooth_11_morphology" ]
    }
(There is no separate "dental_arch" section: v6.1 folded those facts into
mandible/maxilla.) "mandible"/"maxilla" facts become top-level keys in
prediction["global"]; "dental_elements" facts become keys inside
prediction["teeth"]["tooth_<fdi>"] -- exactly mirroring how
run_vqa_inference.py merges its own per-call results, so the SAME
output_field strings schema.json already declares are what
postprocess_pred.py reads on the other end (maxilla_scope,
dental_arch_findings_mandible, tooth_11_morphology, etc.) -- no separate
key-mapping step needed.

Usage (single case):
    python inference_following_schema.py \
        --schema schema/schema.json \
        --slices-dir dataset/validate/outputs/uniform32_slices \
        --case-id A004 \
        --out-dir dataset/validate/outputs/baseline2_uniform32_qwen35_schema/predictions

Usage (all cases):
    python inference_following_schema.py \
        --schema schema/schema.json \
        --slices-dir dataset/validate/outputs/uniform32_slices \
        --out-dir dataset/validate/outputs/baseline2_uniform32_qwen35_schema/predictions \
        --all
"""
import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import requests

FIELD_ID_KEY = "fact_id"
FIELD_DESC_KEY = "question_text"
FIELD_OUTPUT_KEY = "output_field"

# schema.json's whole-case sections (-> prediction["global"]) and its
# per-tooth section (-> prediction["teeth"]["tooth_<fdi>"]). Must match
# build_vqa_pairs.py's GLOBAL_SECTIONS / TOOTH_SECTION -- keep in
# sync if schema.json's top-level keys ever change.
GLOBAL_SECTIONS = ["mandible", "maxilla"]
TOOTH_SECTION = "dental_elements"

# The per-FDI gate on a dental_elements fact ("applies_to_fdi"), imported
# rather than reimplemented: this arm asks the same schema the aksssr arm
# does, and a fact asked here at a position the main arm skips would be
# answering a question the schema does not pose there.
from build_vqa_pairs import fact_applies_to_fdi  # noqa: E402

DEFAULT_FDI_LIST = (
    list(range(11, 19)) + list(range(21, 29)) +
    list(range(31, 39)) + list(range(41, 49))
)  # standard 32-position adult permanent dentition, matching ALL_FDIS in
   # run_vqa_inference.py/postprocess_pred.py. dental_elements facts (crown/
   # root morphology, eruption, endodontic, periodontal, restoration) apply
   # uniformly to all 32 FDIs including the four wisdom teeth -- wisdom-tooth
   # eruption/impaction is ALSO separately asked as its own arch-level fact
   # (lower_left_wisdom_tooth etc.), so wisdom teeth get both, same as in
   # aksssr_pipeline.sh's main pipeline. Override with --fdi-list only if you
   # have a real reason to ask about a different tooth set.

SYSTEM_PREAMBLE = (
    "You are a dental radiology assistant analyzing a CBCT study represented "
    "as 32 uniformly sampled axial slices (ordered from inferior to superior "
    "or as provided). Answer strictly from what is visible in the images. "
    "Do not invent unsupported findings. Respond with a single JSON object "
    "only -- no prose, no markdown code fences -- matching exactly the key "
    "structure given in the schema below."
)


def load_schema(schema_path):
    with open(schema_path) as f:
        return json.load(f)


def find_cases(slices_dir):
    slices_dir = Path(slices_dir)
    cases = []
    for meta_path in sorted(slices_dir.glob("*/*_slices_meta.json")):
        case_id = meta_path.stem.replace("_slices_meta", "")
        cases.append((case_id, meta_path.parent))
    return cases


def load_case_images(case_dir, meta):
    b64_images = []
    for rec in sorted(meta["slices"], key=lambda r: r["slice_index"]):
        img_path = case_dir / rec["filename"]
        with open(img_path, "rb") as f:
            b64_images.append(base64.b64encode(f.read()).decode("utf-8"))
    return b64_images


def _placeholder_for_type(spec):
    """
    A concrete-shaped skeleton value for one fact, so the model has an exact
    key structure to fill in rather than just a bare `null` -- matters most
    for "object" facts (e.g. mandible_scope needs right_condyle_included /
    left_condyle_included / ... as actual keys, not just one opaque blob).
    """
    t = spec.get("type")
    if t == "object":
        return {k: None for k in spec.get("object_fields", {}).keys()}
    if t in ("list[int]", "list[string]", "list[object]"):
        return []
    # bool, enum, int, string, or anything else -- a single scalar the model
    # overwrites directly (true/false, one of enum_values, a number, text).
    return None


def _describe_field(spec, fdi_placeholder=None):
    """One or more human-readable field-guide lines for one fact spec."""
    out_field = spec[FIELD_OUTPUT_KEY]
    question = spec.get(FIELD_DESC_KEY, "")
    if fdi_placeholder is not None:
        out_field = out_field.replace("{fdi}", fdi_placeholder)
        question = question.replace("{fdi}", fdi_placeholder)

    lines = [f"- {out_field}: {question}"]
    # schema v6.1 keeps the recognition criteria in "description" (what a
    # post-and-core looks like versus a crown, what counts as a canal fill).
    # Asking the question without it is a materially harder question -- same
    # reasoning as build_vqa_pairs.build_questions_block.
    description = spec.get("description")
    if description:
        if fdi_placeholder is not None:
            description = description.replace("{fdi}", fdi_placeholder)
        lines.append(f"    how to judge: {description}")
    if spec.get("type") == "object" and spec.get("object_fields"):
        for subkey, subdesc in spec["object_fields"].items():
            lines.append(f"    - {out_field}.{subkey}: {subdesc}")
    if spec.get("type") == "enum" and spec.get("enum_values"):
        lines.append(f"    (one of: {spec['enum_values']})")
    return lines


def build_global_template(schema):
    """
    The {output_field: <skeleton>} JSON and field guide for the whole-case
    (mandible + maxilla) facts only -- split mode's first call.
    """
    template = {}
    field_guide_lines = []
    for section in GLOBAL_SECTIONS:
        for spec in schema.get(section, []):
            template[spec[FIELD_OUTPUT_KEY]] = _placeholder_for_type(spec)
            field_guide_lines.extend(_describe_field(spec))
    return template, "\n".join(field_guide_lines)


def build_tooth_template(schema, fdi):
    """
    The skeleton and field guide for ONE tooth -- split mode's per-FDI calls.

    Flat, with concrete keys (tooth_38_morphology, not tooth_{fdi}_morphology
    inside a "teeth" wrapper): one tooth per call means there is nothing to
    disambiguate, and a flatter shape is one less thing for the model to get
    wrong. assemble_prediction puts it back under teeth["tooth_38"].
    """
    template = {}
    field_guide_lines = []
    for spec in schema.get(TOOTH_SECTION, []):
        if not fact_applies_to_fdi(spec, fdi):
            continue
        out_field = spec[FIELD_OUTPUT_KEY].format(fdi=fdi)
        template[out_field] = _placeholder_for_type(spec)
        field_guide_lines.extend(_describe_field(spec, fdi_placeholder=str(fdi)))
    field_guide_lines.append(
        f"  If there is no tooth at position {fdi}, set every field above to "
        f"null rather than guessing."
    )
    return template, "\n".join(field_guide_lines)


def build_output_template(schema, fdi_list):
    """Build the {output_field: <skeleton>} JSON the model must fill in
    (with a nested "teeth" object for per-tooth facts), plus a
    human-readable field guide for the prompt. Single-call mode."""
    template, global_guide = build_global_template(schema)
    field_guide_lines = global_guide.splitlines()

    tooth_specs = schema.get(TOOTH_SECTION, [])
    if tooth_specs:
        teeth_obj = {}
        for fdi in fdi_list:
            tooth_key = f"tooth_{fdi}"
            tooth_fields = {}
            for spec in tooth_specs:
                if not fact_applies_to_fdi(spec, fdi):
                    continue
                out_field = spec[FIELD_OUTPUT_KEY].format(fdi=fdi)
                tooth_fields[out_field] = _placeholder_for_type(spec)
            teeth_obj[tooth_key] = tooth_fields
        template["teeth"] = teeth_obj

        field_guide_lines.append(
            "- teeth.tooth_<fdi>.*: per-tooth facts, one object per FDI number "
            "(keys below use <fdi> as a placeholder for the real FDI number, "
            "e.g. tooth_11_morphology for FDI 11):"
        )
        for spec in tooth_specs:
            field_guide_lines.extend(_describe_field(spec, fdi_placeholder="<fdi>"))
        # An FDI-gated fact is described once, with <fdi>, like the rest -- but
        # the template above only carries its key under the teeth it applies
        # to. Say so, or the guide reads as if a field went missing.
        for spec in tooth_specs:
            scope = spec.get("applies_to_fdi")
            if scope:
                field_guide_lines.append(
                    f"  {spec[FIELD_OUTPUT_KEY]} appears only under these FDIs: "
                    f"{', '.join(str(f) for f in scope)}. Where the skeleton "
                    f"above does not list it, it is not asked -- do not add it."
                )
        field_guide_lines.append(
            "  If a tooth at a given FDI position is absent/missing, set all of "
            "that tooth's fields to null rather than guessing."
        )

    return template, "\n".join(field_guide_lines)


def build_messages_for_case(b64_images, template, field_guide, scope_note=None,
                            images_first=False):
    """
    One user message: the instructions plus all 32 slices.

    images_first flips the order to [images..., text]. Split mode uses it so
    that all 33 calls in a case share a byte-identical ~29k-token image
    prefix, which is the only thing vLLM's --enable-prefix-caching can
    actually reuse -- with the text first, the prefix diverges at token 0 and
    caching contributes nothing. Single mode keeps the original text-first
    order so its prompt is unchanged.
    """
    prompt = (
        f"{SYSTEM_PREAMBLE}\n\n"
        + (f"{scope_note}\n\n" if scope_note else "")
        + f"Fields to fill in (output_field: what to look for):\n{field_guide}\n\n"
        f"Required JSON key structure (fill in every value, use null where "
        f"not supported by the images):\n{json.dumps(template, indent=2)}"
    )
    text_block = {"type": "text", "text": prompt}
    image_blocks = [{"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    for b64 in b64_images]
    content = (image_blocks + [text_block]) if images_first else ([text_block] + image_blocks)
    return [{"role": "user", "content": content}]


def extract_json(text):
    """Best-effort JSON extraction: try straight parse, then fall back to
    slicing between the first '{' and last '}' (handles models that wrap
    JSON in prose or code fences despite instructions)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _is_deeply_null(v):
    """True if v is None, or a dict/list containing nothing but null/empty
    values all the way down -- needed because object-type facts default to
    {"subfield": None, ...} (a dict of Nones), not a bare None, so a
    shallow check would miss an "all fields null" tooth entirely."""
    if v is None:
        return True
    if isinstance(v, dict):
        return all(_is_deeply_null(x) for x in v.values())
    if isinstance(v, list):
        return len(v) == 0
    return False


def assemble_prediction(case_id, parsed, fdi_list, teeth_raw=None):
    """
    Reshape the model's raw JSON (which mirrors build_output_template's flat
    {output_field: value, "teeth": {"tooth_<fdi>": {...}}} skeleton) into
    the {"case_id", "global", "teeth"} shape postprocess_pred.py expects --
    every non-"teeth" top-level key IS already a valid prediction["global"]
    field name (schema.json's own output_field strings), so this is just a
    wrap + a per-tooth "detected" flag, not a field-by-field remapping.

    teeth_raw lets split mode supply the {"tooth_<fdi>": {...}} map it built
    from its own per-tooth calls, instead of digging it out of `parsed` --
    the only difference between the two modes at this point.
    """
    if teeth_raw is None:
        teeth_raw = parsed.get("teeth", {})
    if not isinstance(teeth_raw, dict):
        teeth_raw = {}
    global_fields = {k: v for k, v in parsed.items() if k != "teeth"}

    teeth_out = {}
    for fdi in fdi_list:
        tooth_key = f"tooth_{fdi}"
        tooth_data = teeth_raw.get(tooth_key, {})
        if not isinstance(tooth_data, dict):
            tooth_data = {}
        # A call that failed outright carries its own detected value already
        # (see run_case_split) -- keep it rather than re-deriving "no" from
        # its emptiness, which would turn a parse failure into a positive
        # claim that this tooth is absent.
        if tooth_data.get("detected") == "error":
            teeth_out[tooth_key] = dict(tooth_data)
            continue
        # Absence signal per the field-guide instruction: "set all of that
        # tooth's fields to null rather than guessing" -- an all-null (or
        # entirely empty) tooth object means the model reported it absent.
        # Checked recursively: object-type facts default to a dict of
        # Nones, not a bare None, so a shallow check would miss them.
        all_null = (len(tooth_data) == 0) or all(_is_deeply_null(v) for v in tooth_data.values())
        tooth_data = dict(tooth_data)
        tooth_data["detected"] = "no" if all_null else "yes"
        teeth_out[tooth_key] = tooth_data

    return {"case_id": case_id, "global": global_fields, "teeth": teeth_out}


def call_vllm(api_base, model_name, messages, max_tokens, enable_thinking=False, timeout=300):
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # See inference_free_report.py's call_vllm for why this is needed:
        # without it, thinking defaults on and a model that runs out of
        # max_tokens mid-reasoning returns content=None.
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def run_case(case_id, case_dir, schema, fdi_list, args):
    meta_path = case_dir / f"{case_id}_slices_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    b64_images = load_case_images(case_dir, meta)
    template, field_guide = build_output_template(schema, fdi_list)
    messages = build_messages_for_case(b64_images, template, field_guide)

    t0 = time.time()
    response = call_vllm(args.api_base, args.model_name, messages, args.max_tokens,
                          enable_thinking=args.enable_thinking, timeout=args.timeout)
    elapsed = time.time() - t0

    try:
        raw_text = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"[ERROR] {case_id}: could not read model response ({e}): {response}", file=sys.stderr)
        raw_text = ""

    parsed = extract_json(raw_text) if raw_text else None
    if not isinstance(parsed, dict):
        print(f"[WARN] {case_id}: failed to parse JSON from model output; "
              f"writing an empty global/teeth skeleton (raw text kept under "
              f"global._unparsed_raw_output for manual inspection).", file=sys.stderr)
        prediction = {
            "case_id": case_id,
            "global": {"_unparsed_raw_output": raw_text},
            "teeth": {f"tooth_{fdi}": {"detected": "no_image"} for fdi in fdi_list},
        }
    else:
        prediction = assemble_prediction(case_id, parsed, fdi_list)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{case_id}_pred.json"
    with open(pred_path, "w") as f:
        json.dump(prediction, f, indent=2)

    print(f"[OK] {case_id}: predictions written to {pred_path} ({elapsed:.1f}s)")


# ── Split mode: one global call + one call per FDI ─────────────────────────

def _partial_path(args, case_id, call_key):
    return Path(args.partial_dir or (Path(args.out_dir) / "_partial")) / case_id / f"{call_key}.json"


def _run_one_call(case_id, call_key, b64_images, template, field_guide,
                  scope_note, max_tokens, args):
    """
    Issue one call and return its parsed dict, or None if it never produced
    parseable JSON. Retries --max-retries times on transport errors and on
    unparseable output alike: both are "the call didn't land", and at
    temperature 0 a retry costs one call for a chance at a clean parse.
    """
    messages = build_messages_for_case(b64_images, template, field_guide,
                                       scope_note=scope_note, images_first=True)
    raw_text = ""
    for attempt in range(args.max_retries + 1):
        try:
            response = call_vllm(args.api_base, args.model_name, messages,
                                 max_tokens, enable_thinking=args.enable_thinking,
                                 timeout=args.timeout)
            raw_text = response["choices"][0]["message"]["content"] or ""
        except (requests.RequestException, KeyError, IndexError) as e:
            print(f"  [WARN] {case_id}/{call_key}: call failed "
                  f"(attempt {attempt + 1}/{args.max_retries + 1}): {e}",
                  file=sys.stderr)
            continue
        parsed = extract_json(raw_text)
        if isinstance(parsed, dict):
            return parsed, raw_text
        print(f"  [WARN] {case_id}/{call_key}: output did not parse as JSON "
              f"(attempt {attempt + 1}/{args.max_retries + 1})", file=sys.stderr)
    return None, raw_text


def run_case_split(case_id, case_dir, schema, fdi_list, args):
    """
    33 calls for one case: one covering every global fact, then one per FDI
    covering that tooth's six facts. Every call sees the same 32 slices --
    loaded once here and reused, so the base64 work happens once per case,
    not once per call.
    """
    meta_path = case_dir / f"{case_id}_slices_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    b64_images = load_case_images(case_dir, meta)

    t0 = time.time()
    n_called = n_cached = 0

    # ── global call ────────────────────────────────────────────────────────
    global_partial = _partial_path(args, case_id, "global")
    if args.resume and global_partial.exists():
        global_fields = json.loads(global_partial.read_text())
        n_cached += 1
    else:
        template, guide = build_global_template(schema)
        parsed, raw_text = _run_one_call(
            case_id, "global", b64_images, template, guide,
            scope_note="This call covers the WHOLE CASE: the mandible and "
                       "maxilla as a whole, not any one tooth.",
            max_tokens=args.max_tokens, args=args)
        n_called += 1
        # A failed global call must not cost this case its 32 tooth answers,
        # so record the failure and carry on -- same shape run_case uses.
        global_fields = parsed if parsed is not None else {"_unparsed_raw_output": raw_text}
        if parsed is None:
            print(f"  [ERROR] {case_id}/global: no parseable JSON; continuing "
                  f"with the per-tooth calls", file=sys.stderr)
        _write_json_atomic(global_partial, global_fields)

    # ── per-tooth calls ────────────────────────────────────────────────────
    teeth_raw = {}
    for fdi in fdi_list:
        call_key = f"tooth_{fdi}"
        partial = _partial_path(args, case_id, call_key)
        if args.resume and partial.exists():
            teeth_raw[call_key] = json.loads(partial.read_text())
            n_cached += 1
            continue
        template, guide = build_tooth_template(schema, fdi)
        parsed, raw_text = _run_one_call(
            case_id, call_key, b64_images, template, guide,
            scope_note=f"This call is about TOOTH {fdi} ONLY. Locate FDI "
                       f"position {fdi} in the slices and answer only about "
                       f"that tooth.",
            max_tokens=args.tooth_max_tokens, args=args)
        n_called += 1
        if parsed is None:
            # NOT "no": postprocess_pred.ABSENT_DETECTED_VALUES is
            # ("no", "no_image"), so "no" would convert a parse failure into a
            # positive claim that this tooth is absent, and that claim reaches
            # the report. "error" is read by nothing, i.e. "no statement about
            # this tooth", which is what actually happened.
            parsed = {"detected": "error", "_unparsed_raw_output": raw_text}
            print(f"  [ERROR] {case_id}/{call_key}: no parseable JSON",
                  file=sys.stderr)
        teeth_raw[call_key] = parsed
        _write_json_atomic(partial, parsed)

    prediction = assemble_prediction(case_id, global_fields, fdi_list,
                                     teeth_raw=teeth_raw)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{case_id}_pred.json"
    _write_json_atomic(pred_path, prediction)

    n_error = sum(1 for t in prediction["teeth"].values() if t.get("detected") == "error")
    elapsed = time.time() - t0
    print(f"[OK] {case_id}: {n_called} call(s)"
          + (f", {n_cached} resumed from partials" if n_cached else "")
          + (f", {n_error} tooth call(s) FAILED" if n_error else "")
          + f" -> {pred_path} ({elapsed:.1f}s)")


def _write_json_atomic(path, obj):
    """Write via a temp file + replace, so a kill mid-write can't leave a
    truncated partial that --resume would then happily load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", required=True, help="Path to schema.json")
    ap.add_argument("--slices-dir", required=True, help="Root dir containing per-case slice folders (uniform-32_selection.py output)")
    ap.add_argument("--out-dir", required=True, help="Where to write {case_id}_pred.json")
    ap.add_argument("--case-id", help="Run a single case")
    ap.add_argument("--case-ids", nargs="+", default=None,
                     help="Run exactly this list of cases. Prefer this over --all "
                          "when the caller already knows which cases it built: "
                          "--slices-dir is shared across baseline arms and prior "
                          "runs, so --all + --limit can silently pick up cases this "
                          "job never touched.")
    ap.add_argument("--all", action="store_true", help="Run every case found under --slices-dir")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N cases (smoke testing)")
    ap.add_argument("--resume", action="store_true",
                     help="Skip cases that already have {case_id}_pred.json in --out-dir")
    ap.add_argument("--call-mode", choices=("single", "split"), default="single",
                     help="single (default): one call per case covering every fact. "
                          "split: one global call + one call per FDI, all seeing the "
                          "same 32 slices. Default stays 'single' so existing callers "
                          "(jobs/gen_report_baseline2.sh) are unchanged -- the arm that "
                          "wants splitting passes --call-mode split explicitly.")
    ap.add_argument("--partial-dir", default=None,
                     help="Where split mode writes each call's result as it arrives "
                          "(default: <out-dir>/_partial). With --resume these let a "
                          "half-finished case restart at the call it died on. Kept "
                          "after success as an audit trail of what each call returned.")
    ap.add_argument("--max-retries", type=int, default=1,
                     help="Split mode: retries per call on a transport error or "
                          "unparseable output (default 1, i.e. up to 2 attempts)")
    ap.add_argument("--timeout", type=int, default=300, help="Per-call HTTP timeout, seconds")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--model-name", default="qwen3.5-vl", help="Must match --served-model-name used to start vLLM")
    ap.add_argument("--max-tokens", type=int, default=4096,
                     help="single mode: the whole case (global+all teeth) in one call. "
                          "split mode: the global call only -- see --tooth-max-tokens.")
    ap.add_argument("--tooth-max-tokens", type=int, default=1024,
                     help="Split mode: budget for one tooth's six facts. Raise it if "
                          "predictions come back with detected == 'error'.")
    ap.add_argument("--enable-thinking", action="store_true",
                     help="Let the model use visible reasoning before answering. Off by "
                          "default -- see inference_free_report.py's --enable-thinking help.")
    ap.add_argument("--fdi-list", type=str, default=None,
                     help="Comma-separated FDI list, e.g. 11,12,13,... Defaults to standard 32-position adult dentition -- verify against your schema's actual dental_elements convention.")
    args = ap.parse_args()

    if not args.case_id and not args.case_ids and not args.all:
        ap.error("Specify one of --case-id <id>, --case-ids <id> [<id> ...], or --all")

    schema = load_schema(args.schema)
    fdi_list = (
        [int(x) for x in args.fdi_list.split(",")] if args.fdi_list else DEFAULT_FDI_LIST
    )

    if args.case_id:
        cases = [(args.case_id, Path(args.slices_dir) / args.case_id)]
    elif args.case_ids:
        cases = [(cid, Path(args.slices_dir) / cid) for cid in args.case_ids]
    else:
        cases = find_cases(args.slices_dir)
        if not cases:
            print(f"[ERROR] No cases found under {args.slices_dir}", file=sys.stderr)
            sys.exit(1)

    if args.limit:
        cases = cases[:args.limit]

    out_dir = Path(args.out_dir)
    runner = run_case_split if args.call_mode == "split" else run_case
    n_calls = (1 + len(fdi_list)) if args.call_mode == "split" else 1
    print(f"[INFO] schema: {args.schema}")
    print(f"[INFO] call mode: {args.call_mode} ({n_calls} call(s)/case, "
          f"{len(cases)} case(s), all seeing the same 32 slices)")

    for case_id, case_dir in cases:
        if args.resume and (out_dir / f"{case_id}_pred.json").exists():
            print(f"[SKIP] {case_id}: prediction already exists (--resume)")
            continue
        # One case's failure must not cost every later case its predictions --
        # split mode is a multi-hour run and dying at case 30 of 40 with
        # nothing written for 31-40 is the expensive failure mode.
        try:
            runner(case_id, case_dir, schema, fdi_list, args)
        except Exception as e:
            print(f"[FAIL] {case_id}: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()