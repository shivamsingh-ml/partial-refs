"""One-image vision check: replay ONE real VQA call against any model.

Uses the pipeline's OWN prompt builder so what the model sees here is exactly
what run_vqa_inference.py would send -- a reconstruction would drift from the
real thing on the next schema edit. Compares the answer against the stored
aksssr_v6 prediction for the same tooth, which was produced by the bf16 model.

Everything lives under a __main__ guard: vLLM 0.19 spawns its EngineCore, which
re-imports this file in the child, and module-level engine construction would
recurse.
"""
import json
import os
import sys

sys.path.insert(0, "/project/code")

CASE = os.environ.get("CHECK_CASE", "A008")
TOOTH = os.environ.get("CHECK_TOOTH", "tooth_11")
QA = "/project/outputs/aksssr_v6_validate/qa_pairs.jsonl"
PRED = ("/project/outputs/aksssr_v6_validate/predictions/"
        f"predictions_v6.4/{CASE}_pred.json")
# Parametrised so the SAME prompt and code path can be replayed against the
# bf16 model as a control -- the int4/bf16 comparison is only worth anything
# if nothing but the weights differs.
MODEL = os.environ.get("CHECK_MODEL", "/models/Qwen3.5-9B-int4-AutoRound")
TOKENIZER = os.environ.get("CHECK_TOKENIZER", "/models/Qwen3.5-9B")
UTIL = float(os.environ.get("CHECK_UTIL", "0.275"))


def main():
    from run_vqa_inference import (CATEGORY_SYSTEM, CATEGORY_USER_TEMPLATE,
                                   build_call_prompt, resolve_record_paths)

    record = None
    with open(QA) as f:
        for line in f:
            r = json.loads(line)
            if r.get("case_id") == CASE:
                record = resolve_record_paths(r, "/project")
                break
    if record is None:
        sys.exit(f"case {CASE} not in {QA}")

    call = record["dental_elements"][TOOTH]
    blocks, schema = build_call_prompt(call, CATEGORY_USER_TEMPLATE)
    if blocks is None:
        sys.exit(f"no image on disk for {CASE}/{TOOTH}")

    n_img = sum(1 for b in blocks if b.get("type") != "text")
    print(f"[check] {CASE}/{TOOTH}: {len(blocks)} blocks, {n_img} image(s) | model={MODEL} util={UTIL}")

    # Which image preprocessing does each path resolve to? If these disagree,
    # the tokenizer override is silently changing how pixels reach the model.
    from transformers import AutoProcessor
    for label, path in (("model   ", MODEL), ("tokenizer", TOKENIZER)):
        try:
            ip = AutoProcessor.from_pretrained(path).image_processor
            print(f"[proc] {label}: patch={getattr(ip,'patch_size',None)} "
                  f"merge={getattr(ip,'merge_size',None)} "
                  f"mean={getattr(ip,'image_mean',None)} size={getattr(ip,'size',None)}")
        except Exception as e:
            print(f"[proc] {label}: FAILED {type(e).__name__}: {e}")

    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, tokenizer=TOKENIZER, max_model_len=8192,
              gpu_memory_utilization=UTIL, limit_mm_per_prompt={"image": 1})

    # vLLM renamed guided_decoding -> structured_outputs; support both.
    try:
        from vllm.sampling_params import StructuredOutputsParams
        sp = SamplingParams(temperature=0.0, max_tokens=1024,
                            structured_outputs=StructuredOutputsParams(json=schema))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        sp = SamplingParams(temperature=0.0, max_tokens=1024,
                            guided_decoding=GuidedDecodingParams(json=schema))

    messages = [{"role": "system", "content": CATEGORY_SYSTEM},
                {"role": "user", "content": blocks}]
    out = llm.chat(messages, sampling_params=sp)
    text = out[0].outputs[0].text

    print("=" * 70)
    print("INT4 ANSWER:")
    print(text)
    print("=" * 70)
    try:
        print("PARSED OK:", json.dumps(json.loads(text), indent=2)[:900])
    except Exception as e:
        print("PARSE FAILED:", e)

    print("=" * 70)
    print("STORED bf16 REFERENCE (predictions_v6.4):")
    pred = json.load(open(PRED))
    ref = (pred.get("teeth") or {}).get(TOOTH) or pred.get(TOOTH)
    print(json.dumps(ref, indent=2)[:900] if ref else "(no stored entry for this tooth)")


if __name__ == "__main__":
    main()
