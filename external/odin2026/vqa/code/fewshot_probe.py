#!/usr/bin/env python3
"""
fewshot_probe.py -- does prepending worked exemplars reduce over-calling?

Targets the four finding groups survey_findings.py measures, on the calls that
actually carry them (see fewshot_probe_plan.md):

    impaction           3d_left, 3d_right      (the four wisdom-tooth facts)
    bridges + implants  pano_tooth_findings_{mandible,maxilla}
    tooth morphology    tooth_{fdi}            (dental_elements, ~28 per case)
    sinus mucosa        sinus_{right,left}     (one fact, one image each)

REPLAYS qa_pairs.jsonl -- it does not rebuild prompts from schema.json.
outputs/<run>/qa_pairs.jsonl already holds, per call, the images, the captions,
the questions, the guidance, the output_schema and the json_schema exactly as
the run sent them. Replaying that and going through
run_vqa_inference.build_call_prompt() means the zero-shot arm is the SAME
prompt the baseline run sent, byte for byte, and the few-shot arm differs from
it in exactly one respect: the exemplar turns. Re-deriving prompts from the
schema instead would let the two arms drift apart on the next schema edit,
which is the one thing this comparison cannot survive.

Arms
────
  --arm zeroshot   [system, query_user]                        -- replay control
  --arm fewshot    [system, ex_user, ex_asst, ..., query_user] -- the treatment

The zero-shot arm is not redundant with the baseline run: it is the control that
shows the probe's own plumbing reproduces the baseline's answers, so a
zeroshot->fewshot difference is attributable to the exemplars rather than to
this file. Run it on a handful of cases before spending GPU time on the rest.

Output is the PIPELINE'S OWN prediction layout --
    {out-dir}/predictions/{case_id}_pred.json
    {"case_id": ..., "global": {...}, "teeth": {"tooth_11": {...}, ...}}
-- with every call the probe did NOT run copied through from --baseline-pred-dir
unchanged. So the result is a complete prediction that survey_findings.py,
postprocess_pred.py and synthesize_report.py all read with no changes at all:

    python3 code/survey_findings.py --run-dir outputs/fewshot_v6_validate

Usage
─────
    # inspect one prompt without calling the model
    python3 code/fewshot_probe.py --qa-jsonl outputs/aksssr_v6_validate/qa_pairs.jsonl \
        --arm fewshot --exemplars code/fewshot_examples \
        --dump-prompt A008/3d_left --out-dir outputs/fewshot_v6_validate

    # arch-level arms (impaction + implants + bridges): ~4 calls/case
    python3 code/fewshot_probe.py --qa-jsonl outputs/aksssr_v6_validate/qa_pairs.jsonl \
        --arm fewshot --exemplars code/fewshot_examples --calls arch \
        --baseline-pred-dir outputs/aksssr_v6_validate/predictions \
        --out-dir outputs/fewshot_v6_validate --vllm-url http://localhost:8000/v1

    # per-tooth morphology: ~28 calls/case, the dominant spend
    ... --calls teeth ...
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_vqa_inference import (  # noqa: E402
    ALL_FDIS,
    CATEGORY_SYSTEM,
    CATEGORY_USER_TEMPLATE,
    HOW_TO_READ_A_TOOTH,
    TOOTH_READING_CALLS,
    TOOTH_SYSTEM,
    TOOTH_USER_TEMPLATE,
    build_call_prompt,
    call_vllm_messages,
    parse_json,
)

# The global calls this probe targets. Deliberately NOT every global call:
# 3d_frontal (maxilla scope, alveolar atrophy) and pano_others_* (primary
# teeth, bone quality, periodontal) carry none of the target findings, and
# running them would spend GPU time on answers the comparison never looks at.
# They are copied through from the baseline instead.
ARCH_CALLS = ["3d_left", "3d_right",
              "pano_tooth_findings_mandible", "pano_tooth_findings_maxilla"]

# One fact and one image per call -- maxilla_sinus_{side} on
# sinus_{side}_detail. Scored separately from the five finding categories:
# survey_findings.py reads the sinus in its ANATOMY section, against
# SINUS_MUCOSA_GT, as right/wrong/unread per side rather than tp/fp. A side
# with no GT entry counts as unread, not as wrong, so an unkeyed case costs
# nothing but tells nothing either.
#
# Kept out of ARCH_CALLS rather than appended to it: the arch calls answer
# from a panoramic or a 3D surface render, these from a coronal+sagittal
# slice pair, and the two stages are worth being able to run and cost
# separately.
SINUS_CALLS = ["sinus_right", "sinus_left"]

# Per-tooth exemplars cannot be per-FDI -- there are 32 of them. Tooth class is
# the useful grouping instead, because what the model has to recognise differs
# by class: root count, canal shape, and what a crown margin looks like on a
# molar are not what they look like on an incisor.
TOOTH_CLASSES = {1: "incisor", 2: "incisor", 3: "canine", 4: "premolar",
                 5: "premolar", 6: "molar", 7: "molar", 8: "molar"}

# How many times to ask again when the answer will not parse. Three attempts
# took arm A's full run from 10 unanswered calls of 652 to zero; see run_case.
PARSE_RETRIES = 3


def tooth_class(fdi: int) -> str:
    return TOOTH_CLASSES[fdi % 10]


# ── Exemplars ─────────────────────────────────────────────────────────────

class ExemplarBundle:
    """
    One exemplar file: the worked examples for a single call type, plus the
    training-split qa_pairs.jsonl their prompts are replayed from.

    File format (code/fewshot_examples/<name>.json):

        {
          "call_key":   "3d_left",          # or "tooth" + "tooth_class"
          "tooth_class": "molar",           # tooth files only
          "split":      "training",
          "qa_jsonl":   "outputs/fewshot_pool_training/qa_pairs.jsonl",
          "exemplars": [
            {"case_id": "A011", "fdi": 46,  # fdi: tooth files only
             "kind": "positive|negative_hard|mixed",
             "why": "what this example is here to teach",
             "answer": { ...the full call object, every fact in the call... }}
          ]
        }

    `answer` must answer EVERY fact in the call, not just the target one. The
    3D calls carry condyle and canal facts as passengers and the panoramic
    calls carry dental_arch_findings; an exemplar that answers only the
    wisdom-tooth fact teaches a truncated output shape, and the model copies
    the shape it is shown.
    """

    def __init__(self, path: Path, project_dir: Path):
        self.path = path
        self._turns = None
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.call_key = spec["call_key"]
        self.tooth_class = spec.get("tooth_class")
        self.split = spec.get("split", "training")
        self.exemplars = spec["exemplars"]

        if self.split != "training":
            raise ValueError(
                f"{path}: split is '{self.split}'. Exemplars must come from the "
                f"training split -- a validate exemplar is leakage into the "
                f"scored set.")
        if not self.exemplars:
            raise ValueError(f"{path}: no exemplars")

        # The pool qa_pairs is loaded lazily (see .rows), so that a leakage
        # check can run on the declared case IDs alone -- before a missing or
        # mistyped pool path gets the chance to fail first with a less
        # important error.
        self.qa_path = Path(spec["qa_jsonl"])
        if not self.qa_path.is_absolute():
            self.qa_path = project_dir / self.qa_path
        self._rows = None

    @property
    def rows(self):
        if self._rows is None:
            self._rows = load_qa_rows(self.qa_path)
        return self._rows

    def case_ids(self):
        return {ex["case_id"] for ex in self.exemplars}

    def validate_answer(self, ex: dict, call_data: dict, where: str):
        """
        Refuse to send a half-written exemplar.

        Two checks, both hard failures, because both fail SILENTLY otherwise --
        the run completes, the numbers look real, and the exemplars taught
        nothing or taught the wrong output shape:

          - placeholder text ("TODO", "TBD", "FIXME", "..."): the previous
            attempt at this experiment shipped an exemplar file that was 100%
            placeholders, down to "case": "TBD". visual_evidence IS the
            teaching signal; a placeholder there teaches noise.
          - top-level keys must be exactly the facts this call asks about,
            read off the call's own json_schema. An exemplar answering only
            the target fact teaches the model to drop the passengers (the
            condyle/canal facts on the 3D calls, dental_arch_findings on the
            panoramic ones), and the model copies the shape it is shown.
        """
        answer = ex.get("answer")
        if not isinstance(answer, dict) or not answer:
            raise ValueError(f"{self.path}: {where} has no 'answer' object")

        blob = json.dumps(answer, ensure_ascii=False)
        for marker in ("TODO", "TBD", "FIXME", "XXX", "<fill", "..."):
            if marker in blob:
                raise ValueError(
                    f"{self.path}: {where}'s answer still contains the "
                    f"placeholder {marker!r}. Write it against the rendered "
                    f"image before running -- an unfinished exemplar is worse "
                    f"than no exemplar.")

        # The placeholder check alone stopped being sufficient once
        # build_fewshot_exemplars.py --draft-visual-evidence could write fluent
        # prose: machine-drafted text passes it trivially, so "nobody ever
        # looked at this" became undetectable. `verified` has to be set by hand
        # and is an affirmative claim that a human read the prose against the
        # image -- which is also where a drafted cue the student cannot
        # actually see gets caught (see --check-perceivable).
        if not ex.get("verified"):
            drafted = ex.get("drafted_by")
            raise ValueError(
                f"{self.path}: {where} is not marked verified"
                + (f" (drafted by {drafted})" if drafted else "")
                + ".\n       Read its visual_evidence against the rendered "
                  "image, run\n       --check-perceivable, then set "
                  "\"verified\": true in the exemplar file.")

        schema = (call_data.get("questions", {}) or {}).get("json_schema") or {}
        expected = set((schema.get("properties") or {}).keys())
        if expected:
            got = set(answer.keys())
            if got != expected:
                raise ValueError(
                    f"{self.path}: {where}'s answer has keys {sorted(got)} but "
                    f"this call asks for {sorted(expected)}. Answer every fact "
                    f"in the call, including the passengers.")

    def turns(self, project_dir: Path) -> List[dict]:
        """
        [user, assistant] x N, built through the SAME prompt path as the query.

        Cached: the turns are identical for every query that uses this bundle,
        and building them base64-encodes the exemplar images. Rebuilding per
        call would re-encode 3 PNGs ~1100 times over a --calls teeth run.
        """
        if self._turns is not None:
            return self._turns

        messages = []
        for ex in self.exemplars:
            case_id = ex["case_id"]
            row = self.rows.get(case_id)
            if row is None:
                raise ValueError(
                    f"{self.path}: exemplar case {case_id} is not in "
                    f"{self.rows.qa_path}. Render it into the exemplar pool "
                    f"first (jobs/fewshot_probe.sh STAGE=render_pool).")

            if self.tooth_class:
                fdi = ex["fdi"]
                call_data = row.get("dental_elements", {}).get(f"tooth_{fdi}")
                where = f"tooth_{fdi}"
            else:
                call_data = row.get("global", {}).get(self.call_key)
                where = self.call_key
            if not call_data:
                raise ValueError(
                    f"{self.path}: exemplar {case_id} has no '{where}' call in "
                    f"its qa_pairs row.")

            self.validate_answer(ex, call_data, f"{case_id}/{where}")

            user_blocks, _ = build_prompt_for(call_data, self.call_key,
                                              ex.get("fdi"), project_dir)
            if user_blocks is None:
                raise ValueError(
                    f"{self.path}: exemplar {case_id}/{where} has no image on "
                    f"disk -- an exemplar with no picture teaches nothing.")

            messages.append({"role": "user", "content": user_blocks})
            messages.append({"role": "assistant",
                             "content": json.dumps(ex["answer"], ensure_ascii=False)})
        self._turns = messages
        return messages


def load_exemplars(exemplars_dir: Path, project_dir: Path,
                   scored_cases) -> Dict[str, ExemplarBundle]:
    """
    Load every exemplar file, keyed by the call it serves: a global call_key
    ("3d_left"), or "tooth:<class>" for the per-tooth files.

    Leakage is checked FIRST, before anything else can fail: an exemplar drawn
    from the scored split voids every number the run produces, so it deserves
    the clearest possible error rather than being reached only after a path or
    a shape check happens to pass.
    """
    bundles = {}
    files = sorted(exemplars_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"[FAIL] no exemplar files in {exemplars_dir}")

    parsed = []
    for f in files:
        try:
            parsed.append(ExemplarBundle(f, project_dir))
        except (ValueError, KeyError) as e:
            raise SystemExit(f"[FAIL] {e}")
    assert_no_leakage(parsed, scored_cases)

    for b in parsed:
        try:
            # Build (and so validate) the turns NOW rather than lazily on the
            # first call that needs them: a malformed exemplar should stop the
            # run before it books GPU time, not 40 minutes into --calls teeth.
            b.turns(project_dir)
        except (ValueError, KeyError) as e:
            raise SystemExit(f"[FAIL] {e}")
        key = f"tooth:{b.tooth_class}" if b.tooth_class else b.call_key
        if key in bundles:
            raise SystemExit(f"[FAIL] two exemplar files claim '{key}': "
                             f"{bundles[key].path} and {b.path}")
        bundles[key] = b
    return bundles


def assert_no_leakage(bundles, scored_cases):
    """
    HARD failure, not a warning. An exemplar that is also a scored case turns
    the probe into a memory test and every number it produces is void -- so
    this is checked before the first call, not audited afterwards.

    `scored_cases` is the SCORED SUBSET, not every case in the qa_pairs.
    Development runs on the training split (validate is held out), so
    exemplars and scored cases are drawn from the same split and appear in the
    same qa_pairs file; presence in that file is not leakage, overlap with the
    scored set is.
    """
    scored = set(scored_cases)
    bad = []
    for b in bundles:
        overlap = b.case_ids() & scored
        if overlap:
            bad.append(f"  {b.path.name}: {sorted(overlap)}")
    if bad:
        raise SystemExit(
            "[FAIL] exemplar cases appear in the scored set -- this is "
            "leakage:\n" + "\n".join(bad))


# ── Prompt construction ───────────────────────────────────────────────────

def build_prompt_for(call_data: dict, call_key: str, fdi: Optional[int],
                     project_dir: Path):
    """
    (user_blocks, json_schema) for one call, through run_vqa_inference's own
    prompt builder -- same system/template/extra_fmt split the pipeline uses:
    tooth calls get TOOTH_USER_TEMPLATE plus HOW_TO_READ_A_TOOTH, the two
    panoramic tooth-finding calls get HOW_TO_READ_A_TOOTH too (they are in
    TOOTH_READING_CALLS), the 3D calls get neither.
    """
    call_data = resolve_image_paths(call_data, project_dir)
    if fdi is not None:
        return build_call_prompt(call_data, TOOTH_USER_TEMPLATE,
                                 extra_fmt={"fdi": fdi,
                                            "tooth_reading": "\n" + HOW_TO_READ_A_TOOTH})
    return build_call_prompt(call_data, CATEGORY_USER_TEMPLATE,
                             extra_fmt={"tooth_reading": (
                                 "\n" + HOW_TO_READ_A_TOOTH
                                 if call_key in TOOTH_READING_CALLS else "")})


def resolve_image_paths(call_data: dict, project_dir: Path) -> dict:
    """
    qa_pairs.jsonl stores image paths relative to --project-dir, so the same
    file resolves on the host and inside the container (see CLAUDE.md). Resolve
    them here rather than chdir'ing, so the probe can be run from anywhere.
    """
    images = call_data.get("images", {})
    resolved = {k: (str(project_dir / p) if p and not os.path.isabs(p) else p)
                for k, p in images.items()}
    return dict(call_data, images=resolved)


def system_for(fdi: Optional[int]) -> str:
    return TOOTH_SYSTEM if fdi is not None else CATEGORY_SYSTEM


def max_tokens_for(fdi: Optional[int]) -> int:
    # Matches run_vqa_inference: 8192 for the multi-fact global calls, 4096 per tooth.
    return 4096 if fdi is not None else 8192


def build_messages(arm: str, call_data: dict, call_key: str, fdi: Optional[int],
                   bundles: Dict[str, ExemplarBundle], project_dir: Path):
    user_blocks, json_schema = build_prompt_for(call_data, call_key, fdi, project_dir)
    if user_blocks is None:
        return None, None

    messages = [{"role": "system", "content": system_for(fdi)}]
    if arm == "fewshot":
        key = f"tooth:{tooth_class(fdi)}" if fdi is not None else call_key
        bundle = bundles.get(key)
        if bundle is None:
            raise SystemExit(
                f"[FAIL] --arm fewshot but no exemplar file for '{key}'. "
                f"Have: {sorted(bundles)}")
        messages.extend(bundle.turns(project_dir))
    messages.append({"role": "user", "content": user_blocks})
    return messages, json_schema


# ── qa_pairs.jsonl ────────────────────────────────────────────────────────

class QARows(dict):
    qa_path: Path


def assert_qa_current(qa_path: Path, schema_path: Path, images_dir: Optional[Path]):
    """
    schema.json is the single source of truth (CLAUDE.md), so a replayed
    qa_pairs.jsonl is only legitimate while it is NEWER than the schema and the
    image captions it was built from. Otherwise the probe would be running an
    older schema's questions under the current schema's name.

    This is not hypothetical. On 2026-08-07 at 11:07 this repo's
    outputs/aksssr_v6_validate held a qa_pairs.jsonl (08-06 18:59) older than
    its own image captions (08-07 02:09), and its frozen 3D caption was missing
    the facts-derived addenda the sidecar had grown -- the wisdom-tooth colour
    key, the acquisition FOV, the IAN proximity note. Rebuilt at 12:41 the same
    day, it matched again. A staleness window that closes on its own is exactly
    the kind that gets replayed by accident.

    Warns rather than exits on the image comparison: build_vqa_pairs.py may
    legitimately be re-run without touching images, and a caption sidecar can
    be rewritten with identical content.
    """
    qa_mtime = qa_path.stat().st_mtime
    if schema_path.exists() and schema_path.stat().st_mtime > qa_mtime:
        raise SystemExit(
            f"[FAIL] {schema_path} is NEWER than {qa_path}.\n"
            f"       The replayed prompts predate the current schema, so this "
            f"run would not be testing it.\n"
            f"       Rebuild first:\n"
            f"         python3 code/build_vqa_pairs.py --schema {schema_path} \\\n"
            f"             --images-dir <run>/images --project-dir . \\\n"
            f"             --cases <ids> --out {qa_path}")

    if images_dir and images_dir.is_dir():
        newer = [p.name for p in images_dir.glob("*caption*.json")
                 if p.stat().st_mtime > qa_mtime]
        if newer:
            print(f"[WARN] {len(newer)} caption sidecar(s) are newer than "
                  f"{qa_path.name} (e.g. {', '.join(sorted(newer)[:3])}).\n"
                  f"       The replayed captions may not describe the current "
                  f"images. Rebuild qa_pairs to be sure.", file=sys.stderr)


def load_qa_rows(qa_path: Path) -> QARows:
    if not qa_path.exists():
        raise SystemExit(f"[FAIL] no qa_pairs.jsonl at {qa_path}")
    rows = QARows()
    rows.qa_path = qa_path
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["case_id"]] = row
    return rows


# ── One case ──────────────────────────────────────────────────────────────

def load_baseline(baseline_dir: Optional[Path], case_id: str) -> dict:
    """
    The baseline run's prediction for this case. Every call the probe does not
    re-run is copied from here, so the output file is a COMPLETE prediction
    rather than a fragment -- that is what lets survey_findings.py read it with
    no changes. Without a baseline dir the untouched calls are simply absent,
    and the survey will read the case as having answered nothing there.
    """
    if not baseline_dir:
        return {}
    path = baseline_dir / f"{case_id}_pred.json"
    if not path.exists():
        print(f"  [WARN] no baseline prediction at {path} -- untouched calls "
              f"will be missing from the output", file=sys.stderr)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_case(client, model: str, arm: str, row: dict, targets: str,
             bundles, project_dir: Path, baseline_dir: Optional[Path],
             out_dir: Path, dry_run: bool, debug: bool) -> dict:
    case_id = row["case_id"]
    baseline = load_baseline(baseline_dir, case_id)

    prediction = {"case_id": case_id,
                  "global": dict(baseline.get("global", {})),
                  "teeth": dict(baseline.get("teeth", {}))}
    stats = {"called": 0, "unanswered": 0, "no_image": 0, "from_baseline": 0}

    jobs = []
    if targets in ("arch", "all"):
        for call_key in ARCH_CALLS:
            call_data = row.get("global", {}).get(call_key)
            if call_data:
                jobs.append((call_key, None, call_data))
    if targets in ("sinus", "all"):
        for call_key in SINUS_CALLS:
            # Absent for a case whose volume clips that antrum -- the image is
            # never rendered, so build_vqa_pairs.py emits no call for it. That
            # is a property of the acquisition, identical in both arms.
            call_data = row.get("global", {}).get(call_key)
            if call_data:
                jobs.append((call_key, None, call_data))
    if targets in ("teeth", "all"):
        tooth_calls = row.get("dental_elements", {})
        for fdi in ALL_FDIS:
            call_data = tooth_calls.get(f"tooth_{fdi}")
            if call_data and call_data.get("images"):
                jobs.append((f"tooth_{fdi}", fdi, call_data))

    for call_key, fdi, call_data in jobs:
        messages, json_schema = build_messages(arm, call_data, call_key, fdi,
                                               bundles, project_dir)
        if messages is None:
            print(f"  [{case_id}] {call_key}: no image on disk, leaving baseline "
                  f"answer in place", file=sys.stderr)
            stats["no_image"] += 1
            continue

        if dry_run:
            stats["called"] += 1
            continue

        # Ask again when the answer will not parse. call_vllm_messages already
        # retries TRANSPORT failures, but a 200 carrying malformed JSON is not
        # one -- and it is the common failure here: arm A's first full run lost
        # 10 of 652 calls that way, every one a COMPLETE object with a missing
        # comma or a raw control character inside a string.
        #
        # A lost call is a difference in coverage between the arms, not a
        # finding about them, and the arms do not lose calls at the same rate --
        # a few-shot prompt is thousands of tokens longer and its exemplars end
        # in '}'. So this is not a quality nicety, it is what keeps the two arms
        # comparable. Nothing is invented: the same prompt is sent again, and
        # sampling is not greedy, so the retry is a genuinely new answer.
        result, raw, last_error = None, None, None
        for attempt in range(PARSE_RETRIES):
            raw = call_vllm_messages(client, model, messages,
                                     max_tokens=max_tokens_for(fdi),
                                     json_schema=json_schema)
            # parse_json RAISES on unparseable output (it never returns None),
            # so a single bad response would otherwise abort the whole run.
            try:
                result = parse_json(raw)
                if attempt:
                    print(f"  [{case_id}] {call_key}: parsed on attempt "
                          f"{attempt + 1}/{PARSE_RETRIES}", file=sys.stderr)
                break
            except Exception as e:
                last_error = e
                if attempt + 1 < PARSE_RETRIES:
                    print(f"  [{case_id}] {call_key}: unparseable ({e}) -- "
                          f"retrying {attempt + 2}/{PARSE_RETRIES}", file=sys.stderr)
        if result is None:
            e = last_error
            # Keep the evidence. An unparseable response used to leave nothing
            # behind but a character offset, and the two explanations -- a
            # malformed answer, or a completion cut short because the prompt
            # plus max_tokens exceeded the server's context -- are told apart
            # by whether the text stops mid-word. A few-shot prompt is several
            # thousand tokens longer than its zero-shot twin, so the arms do
            # not fail at the same rate, and a failure that costs one arm a
            # call is a difference in coverage between them, not a finding.
            dump_dir = out_dir / "unparsed"
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump = dump_dir / f"{case_id}__{call_key}.txt"
            dump.write_text(
                f"# {case_id} {call_key} arm={arm}\n"
                f"# parse error: {e}\n"
                f"# response chars: {len(raw or '')}\n\n{raw or ''}",
                encoding="utf-8")
            print(f"  [{case_id}] {call_key}: unparseable response ({e}) "
                  f"-- raw kept in {dump}", file=sys.stderr)
            stats["unanswered"] += 1
            continue

        stats["called"] += 1
        if fdi is None:
            prediction["global"].update(result)
        else:
            result["detected"] = "yes"
            prediction["teeth"][call_key] = result
        if debug:
            print(f"  [{case_id}] {call_key}: {json.dumps(result)[:200]}",
                  file=sys.stderr)

    # Entries in the finished prediction that this run did not produce -- i.e.
    # what came from the baseline. Not a call count: global facts and tooth
    # entries are both counted, and one global call answers several facts.
    stats["from_baseline"] = (len(prediction["global"])
                              + len(prediction["teeth"]) - stats["called"])

    if not dry_run:
        out_path = out_dir / "predictions" / f"{case_id}_pred.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(prediction, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa-jsonl", type=Path, required=True,
                    help="the baseline run's qa_pairs.jsonl -- replayed, not rebuilt")
    ap.add_argument("--arm", choices=["zeroshot", "fewshot"], default="zeroshot")
    ap.add_argument("--calls", choices=["arch", "sinus", "teeth", "all"],
                    default="arch",
                    help="arch = %s; sinus = %s; teeth = tooth_{fdi} (~28/case)"
                         % (", ".join(ARCH_CALLS), ", ".join(SINUS_CALLS)))
    ap.add_argument("--exemplars", type=Path, default=Path("code/fewshot_examples"),
                    help="directory of exemplar files (required for --arm fewshot)")
    ap.add_argument("--baseline-pred-dir", type=Path, default=None,
                    help="predictions/ of the baseline run; untouched calls are "
                         "copied from here so the output is a complete prediction")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/fewshot_v6_validate"))
    ap.add_argument("--project-dir", type=Path, default=Path("."),
                    help="root the qa_pairs image paths are relative to")
    ap.add_argument("--cases", default=None,
                    help="comma-separated case IDs (default: every case in --qa-jsonl)")
    ap.add_argument("--limit", type=int, default=None, help="first N cases only")
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3.5-vl")
    ap.add_argument("--dump-prompt", metavar="CASE/CALL",
                    help="e.g. A008/3d_left or A008/tooth_46 -- write the exact "
                         "message list to disk and exit, no model call")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every prompt but call nothing (checks images and exemplars)")
    ap.add_argument("--resume", action="store_true",
                    help="skip cases that already have a prediction in --out-dir")
    ap.add_argument("--schema", type=Path, default=Path("schema/schema.json"),
                    help="checked against --qa-jsonl's mtime; the schema is the "
                         "source of truth, so a replay older than it is refused")
    ap.add_argument("--images-dir", type=Path, default=None,
                    help="caption sidecars to staleness-check (default: <qa dir>/images)")
    ap.add_argument("--skip-staleness-check", action="store_true",
                    help="run against a deliberately older qa_pairs (e.g. reproducing "
                         "an archived arm). States what it is doing.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    project_dir = args.project_dir.resolve()

    images_dir = args.images_dir or (args.qa_jsonl.parent / "images")
    if args.skip_staleness_check:
        print("[WARN] --skip-staleness-check: replaying qa_pairs without "
              "checking it against the current schema.", file=sys.stderr)
    elif args.qa_jsonl.exists():
        assert_qa_current(args.qa_jsonl, args.schema, images_dir)

    rows = load_qa_rows(args.qa_jsonl)

    case_ids = ([c.strip() for c in args.cases.split(",")] if args.cases
                else list(rows.keys()))
    missing = [c for c in case_ids if c not in rows]
    if missing:
        raise SystemExit(f"[FAIL] not in {args.qa_jsonl}: {missing}")
    if args.limit:
        case_ids = case_ids[:args.limit]

    # Only the few-shot arm needs exemplars -- a zero-shot run (including
    # --dump-prompt on it) must not fail just because the exemplar directory
    # is empty or absent.
    bundles = {}
    if args.arm == "fewshot":
        if not args.exemplars.is_dir():
            raise SystemExit(f"[FAIL] --exemplars {args.exemplars} is not a directory")
        # Checked against the cases actually SCORED (post --cases/--limit), not
        # every case in the qa_pairs. Once the experiment runs on the training
        # split, exemplars and scored cases come from the same split and
        # therefore the same qa_pairs file -- comparing against rows.keys()
        # would reject every legitimate exemplar for being present in the file
        # it is necessarily built from. What must not overlap is exemplars and
        # the SCORED SET.
        bundles = load_exemplars(args.exemplars, project_dir, case_ids)
        print(f"[INFO] exemplars: {len(bundles)} call type(s) -- "
              f"{', '.join(sorted(bundles))}")

    # ── one prompt, to disk, no model ────────────────────────────────────
    if args.dump_prompt:
        case_id, _, call_key = args.dump_prompt.partition("/")
        row = rows.get(case_id)
        if row is None:
            raise SystemExit(f"[FAIL] {case_id} not in {args.qa_jsonl}")
        fdi = int(call_key.split("_")[1]) if call_key.startswith("tooth_") else None
        call_data = (row.get("dental_elements", {}).get(call_key) if fdi is not None
                     else row.get("global", {}).get(call_key))
        if not call_data:
            raise SystemExit(f"[FAIL] {case_id} has no call '{call_key}'")

        messages, json_schema = build_messages(args.arm, call_data, call_key, fdi,
                                               bundles, project_dir)
        if messages is None:
            raise SystemExit(f"[FAIL] {case_id}/{call_key}: no image on disk")

        # Images are multi-MB base64 -- elide them in the dump, keep the count.
        def strip(m):
            if not isinstance(m["content"], list):
                return m
            content = [b if b.get("type") != "image_url"
                       else {"type": "image_url", "image_url": "<elided base64>"}
                       for b in m["content"]]
            return dict(m, content=content)

        n_images = sum(1 for m in messages if isinstance(m["content"], list)
                       for b in m["content"] if b.get("type") == "image_url")
        out_path = args.out_dir / f"prompt_{args.arm}_{case_id}_{call_key}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {"arm": args.arm, "case_id": case_id, "call": call_key,
             "n_messages": len(messages), "n_images": n_images,
             "roles": [m["role"] for m in messages],
             "guided_json_present": json_schema is not None,
             "system": system_for(fdi)[:200] + "...",
             "messages": [strip(m) for m in messages]},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_path}")
        print(f"  messages: {len(messages)}  ({' '.join(m['role'] for m in messages)})")
        print(f"  images:   {n_images}")
        print(f"  guided_json: {'present' if json_schema else 'MISSING'}")
        return

    # ── the run ──────────────────────────────────────────────────────────
    client = None
    if not args.dry_run:
        from openai import OpenAI
        client = OpenAI(base_url=args.vllm_url, api_key="not-needed")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    totals = {"called": 0, "unanswered": 0, "no_image": 0}
    for i, case_id in enumerate(case_ids, 1):
        out_path = args.out_dir / "predictions" / f"{case_id}_pred.json"
        if args.resume and out_path.exists():
            print(f"[{i}/{len(case_ids)}] {case_id}: exists, skipping")
            continue
        print(f"[{i}/{len(case_ids)}] {case_id} ({args.arm}, {args.calls})...")
        stats = run_case(client, args.model, args.arm, rows[case_id], args.calls,
                         bundles, project_dir, args.baseline_pred_dir,
                         args.out_dir, args.dry_run, debug=not args.quiet)
        for k in totals:
            totals[k] += stats[k]
        print(f"    called {stats['called']}, unanswered {stats['unanswered']}, "
              f"no_image {stats['no_image']}, kept from baseline "
              f"{stats['from_baseline']}")

    print(f"\n[DONE] {args.arm}/{args.calls} over {len(case_ids)} case(s): {totals}")
    if not args.dry_run:
        print(f"Score with:\n  python3 code/survey_findings.py --run-dir {args.out_dir}")


if __name__ == "__main__":
    main()
