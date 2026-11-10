"""Per-report structured findings (no union) with a LOCAL LLM served by vLLM (HANDOFF M1 step 4, GT/T1).

Talks to a vLLM OpenAI-compatible server (started by jobs/parse_reports.sbatch). Output is forced into
our JSON schema (structured output) and validated against partialref.extract.ontology. Report text is
read inside this process only; it is never written to outputs or logs. One JSON file per report makes
runs resumable.

Usage (inside the job, server already up):
    python -m partialref.extract.per_report --manifest results/manifest/<run> --lang en \
        --server http://127.0.0.1:PORT --model qwen3-14b [--limit-cases 50] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from partialref.extract.fewshot import messages
from partialref.extract.ontology import (
    Finding,
    FindingType,
    ReportExtraction,
    absence_conflict_details,
    absence_conflicts,
    instances,
    json_schema,
    slot_kind,
)
from partialref.provenance import make_run_id, write_provenance


def _post(url: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def request_body(model: str, report_text: str, mode: str, repetition_penalty: float | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> dict:
    body = {"model": model, "messages": messages(report_text), "temperature": temperature,
            "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": False}}
    if repetition_penalty:
        body["repetition_penalty"] = repetition_penalty   # vLLM sampling extra; used only on retry
    if seed is not None:
        body["seed"] = seed                                # reproducible sampling on the last retry
    schema = json_schema()
    if mode == "response_format":
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "report_extraction", "schema": schema}}
    else:  # vLLM's own spelling (installed 0.19: structured_outputs; 'guided_json' no longer exists)
        body["structured_outputs"] = {"json": schema}
    return body


DEFAULT_LADDER = "1.1:0,1.3:0,1.05:0.3"


def parse_case_ids(s: str) -> list[str]:
    ids = [x.strip() for x in s.split(",") if x.strip()]
    if not ids:                                       # "--case-ids ''" must not turn into a full run
        raise argparse.ArgumentTypeError("no case id given")
    return ids


def parse_ladder(spec: str) -> list[tuple[float, float]]:
    return [(float(p), float(t)) for p, t in (x.split(":") for x in spec.split(",") if x.strip())]


LEFT_QUADRANT_DIGITS = set("2367")          # FDI: 2 upper left, 3 lower left (+ primary 6, 7)
QUADRANT_OF_DIGIT = {"1": "QuadrantI", "2": "QuadrantII", "3": "QuadrantIII", "4": "QuadrantIV",
                     "5": "QuadrantI", "6": "QuadrantII", "7": "QuadrantIII", "8": "QuadrantIV"}


def _repair_anchor(f: dict) -> list[dict]:
    """A side/quadrant finding without its side/quadrant but with tooth ids: derive the anchor from the
    FDI digits (the prompt's own convention), one finding per distinct side/quadrant. Else unchanged."""
    try:
        kind = slot_kind(FindingType(f.get("finding_type")))
    except ValueError:
        return [f]
    teeth = [str(t) for t in (f.get("tooth_ids") or []) if str(t)[:1].isdigit()]
    if kind == "side" and not f.get("side") and teeth:
        side_of = {t: "Left" if t[0] in LEFT_QUADRANT_DIGITS else "Right" for t in teeth}
        return [{**f, "side": v, "tooth_ids": sorted(t for t in teeth if side_of[t] == v)}
                for v in sorted(set(side_of.values()))]
    if kind == "quadrant" and not f.get("quadrant") and teeth:
        q_of = {t: QUADRANT_OF_DIGIT[t[0]] for t in teeth if t[0] in QUADRANT_OF_DIGIT}
        return [{**f, "quadrant": v, "tooth_ids": sorted(t for t in q_of if q_of[t] == v)}
                for v in sorted(set(q_of.values()))] or [f]
    return [f]


def report_ok(meta: pd.DataFrame) -> pd.Series:
    """Report usable for counting readers: parsed without error AND no finding dropped during salvage
    (a dropped real finding would look like a reader omission; review 2026-11-09 #14)."""
    dropped = pd.to_numeric(meta["dropped_findings"], errors="coerce").fillna(0) if "dropped_findings" in meta \
        else pd.Series(0, index=meta.index)
    return meta["error"].isna() & (dropped == 0)


def parse_extraction(content: str) -> tuple[ReportExtraction, dict]:
    """Validate the model output. If the report as a whole fails validation (schema-valid output can still
    violate the per-type anchor rule, review 2026-11-09 #1), salvage it finding by finding: repair missing
    side/quadrant from tooth ids, drop findings that stay invalid, and count both. Truncated / non-JSON
    output still raises (whole report failed). The model's findings are stored as given; absent/present
    conflicts are only recorded (extract_one), never resolved here, so any convention stays reversible."""
    try:
        return ReportExtraction.model_validate_json(content), {"salvaged": False}
    except ValidationError:
        return _salvage(content)


def _salvage(content: str) -> tuple[ReportExtraction, dict]:
    """Per-finding salvage of an output that failed whole-report validation (see parse_extraction)."""
    data = json.loads(content)                           # JSONDecodeError -> report failed
    if not isinstance(data, dict):
        raise TypeError("model output is not a JSON object")
    good, repaired, dropped, why = [], 0, 0, []
    types = {t.value for t in FindingType}
    for f in data.get("findings") or []:
        fixed = _repair_anchor(f) if isinstance(f, dict) else [f]
        repaired += fixed != [f]
        for g in fixed:
            try:
                good.append(Finding.model_validate(g))
            except ValidationError as e:
                dropped += 1         # why: type (if a known one) + pydantic messages; never the input values
                t = g.get("finding_type") if isinstance(g, dict) else None
                t = t if isinstance(t, str) and t in types else "?"
                why.append(f"{t}: " + "; ".join(sorted({err["msg"] for err in e.errors()})))
    ext = ReportExtraction(findings=good, not_assessable=data.get("not_assessable") or [])
    return ext, {"salvaged": True, "repaired_findings": repaired, "dropped_findings": dropped,
                 "dropped_reasons": why}


def extract_one(server: str, model: str, text: str, mode: str, repetition_penalty: float | None = None,
                temperature: float = 0.0, seed: int | None = None) -> tuple[ReportExtraction | None, dict]:
    t0 = time.time()
    meta: dict = {"repetition_penalty": repetition_penalty, "temperature": temperature, "seed": seed}
    try:
        resp = _post(f"{server}/v1/chat/completions",
                     request_body(model, text, mode, repetition_penalty, temperature, seed))
        content = resp["choices"][0]["message"]["content"]
        meta.update(prompt_tokens=resp.get("usage", {}).get("prompt_tokens"),
                    completion_tokens=resp.get("usage", {}).get("completion_tokens"),
                    finish_reason=resp["choices"][0].get("finish_reason"))
        ext, info = parse_extraction(content)
        meta.update(info)
        meta["absence_conflicts"] = len(absence_conflicts(ext.findings))
        meta["absence_conflict_details"] = absence_conflict_details(ext.findings)
        meta["error"] = None
    except ValidationError as e:
        ext, meta["error"] = None, f"validation: {e.error_count()} errors"
    except json.JSONDecodeError:
        ext, meta["error"] = None, "invalid_json"
    except urllib.error.HTTPError as e:
        ext, meta["error"] = None, f"http {e.code}"
    except Exception as e:  # noqa: BLE001  (record and continue)
        ext, meta["error"] = None, type(e).__name__
    meta["seconds"] = round(time.time() - t0, 2)
    return ext, meta


def detect_mode(server: str, model: str) -> str:
    """Pick the structured-output spelling the server accepts. Success = HTTP 200 and valid JSON
    (our stricter ontology validation is NOT required here, so one odd answer cannot abort a run)."""
    reasons = {}
    for mode in ("response_format", "structured_outputs"):
        try:
            resp = _post(f"{server}/v1/chat/completions",
                         request_body(model, "Implant in position 36. No other findings.", mode), timeout=600)
            json.loads(resp["choices"][0]["message"]["content"])
            return mode
        except urllib.error.HTTPError as e:
            reasons[mode] = f"http {e.code}"
        except Exception as e:  # noqa: BLE001
            reasons[mode] = type(e).__name__
    raise RuntimeError(f"server accepted neither structured-output spelling: {reasons}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--lang", default="en", choices=["en", "it"])
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--model-path", default="", help="recorded in provenance")
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--case-ids", type=parse_case_ids, default=[])
    ap.add_argument("--centres", default="PFSA")
    ap.add_argument("--out-dir", type=Path, help="existing run dir to resume")
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default="parse")
    ap.add_argument("--retry-errors", action="store_true", help="re-parse reports whose stored result failed")
    ap.add_argument("--retry-ladder", type=parse_ladder, default=parse_ladder(DEFAULT_LADDER),
                    help="retries for a failed parse, in order, as penalty:temperature[,...] "
                         f"(default {DEFAULT_LADDER}; temperature > 0 uses seed 0); '' = no retry")
    a = ap.parse_args(argv)

    reps = pd.read_parquet(a.manifest / "reports.parquet")
    reps = reps[(reps["lang"] == a.lang) & reps["case_id"].str[0].isin(list(a.centres))]
    if a.case_ids:
        unknown = sorted(set(a.case_ids) - set(reps["case_id"]))
        if unknown:                                   # a typo must not silently shrink the run
            raise SystemExit(f"--case-ids: {len(unknown)} id(s) without a {a.lang} report, e.g. {unknown[:5]}")
        reps = reps[reps["case_id"].isin(a.case_ids)]
    if a.limit_cases:
        keep = sorted(reps["case_id"].unique())[: a.limit_cases]
        reps = reps[reps["case_id"].isin(keep)]
    reps = reps.sort_values(["case_id", "report_id"])
    if reps.empty:
        raise SystemExit("no reports selected")

    out = a.out_dir or (a.results_dir / "extract" / make_run_id(f"{a.tag}-{a.lang}"))
    (out / "per_report").mkdir(parents=True, exist_ok=True)
    mode = detect_mode(a.server, a.model)
    print(f"structured-output mode: {mode}; reports: {len(reps)}", flush=True)

    def work(row) -> dict:
        dst = out / "per_report" / f"{row.case_id}__{row.report_id}__{a.lang}.json"
        prev = None
        if dst.exists():
            try:
                prev = json.loads(dst.read_text())
            except json.JSONDecodeError:          # partial file from a killed job -> parse again
                prev = None
            prev_ok = prev is not None and prev["extraction"] is not None and not prev["meta"].get("dropped_findings")
            if prev is not None and (prev_ok or not a.retry_errors):   # --retry-errors: redo every not-OK report
                return prev["meta"]
        text = Path(row.path).read_text(encoding="utf-8", errors="replace")
        stored = None                                 # --retry-errors: the stored attempt stays a candidate
        if prev is not None and prev["extraction"] is not None:
            stored = (ReportExtraction.model_validate(prev["extraction"]), prev["meta"])
        ext, meta = extract_one(a.server, a.model, text, mode)
        # failed parses are usually degenerate repetition loops on normal-length reports; a salvaged parse
        # that had to DROP findings also counts as not OK (report_ok). Retry with stronger repetition
        # penalties, last with a little seeded sampling; keep the first fully OK attempt, else the best one.
        def fully_ok(e, m):
            return e is not None and not m.get("dropped_findings")

        def badness(e, m):
            return (e is None, m.get("dropped_findings") or 0)

        def problem(e, m):
            return m.get("error") or f"dropped_findings: {m.get('dropped_findings')}"

        attempts = ([stored] if stored else []) + [(ext, meta)]
        for pen, temp in (a.retry_ladder if not fully_ok(ext, meta) else []):
            attempts.append(extract_one(a.server, a.model, text, mode, pen, temp, 0 if temp > 0 else None))
            if fully_ok(*attempts[-1]):
                break
        kept = min(range(len(attempts)), key=lambda i: (badness(*attempts[i]), i))   # earliest best attempt
        ext, meta = attempts[kept]
        if len(attempts) > 1:                         # problems of every attempt that was NOT kept, in order
            others = [problem(*attempts[i]) for i in range(len(attempts)) if i != kept]
            meta.update(previous_errors=others, first_attempt_error=problem(*attempts[0]) if kept else None,
                        attempts_made=len(attempts), kept_attempt=kept + 1)
        meta.update(case_id=row.case_id, report_id=row.report_id, lang=a.lang,
                    n_findings=len(ext.findings) if ext else None)
        tmp = dst.with_suffix(".json.tmp")           # atomic: a killed job never leaves a partial file
        tmp.write_text(json.dumps({"meta": meta, "extraction": ext.model_dump(mode="json") if ext else None}))
        tmp.replace(dst)
        return meta

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        metas = list(ex.map(work, reps.itertuples()))

    rows_inst, rows_na, all_meta, scopes = [], [], [], {}
    for f in sorted((out / "per_report").glob(f"*__{a.lang}.json")):
        d = json.loads(f.read_text())
        all_meta.append(d["meta"])          # coverage = every per-report file in the run, not this call only
        if not d["extraction"]:
            continue
        ext = ReportExtraction.model_validate(d["extraction"])
        m = d["meta"]
        for fd in ext.findings:              # region-level ToothAbsence by scope (post-run check, v3.2)
            if fd.region:
                scopes[fd.region_scope or "unspecified"] = scopes.get(fd.region_scope or "unspecified", 0) + 1
        rows_inst += [{"case_id": c, "report_id": m["report_id"], "lang": a.lang, "finding_type": t, "slot": s}
                      for c, t, s in instances(m["case_id"], ext.findings)]
        rows_na += [{"case_id": m["case_id"], "report_id": m["report_id"], "structure": s}
                    for s in ext.not_assessable]
    meta_df = pd.DataFrame(all_meta) if all_meta else pd.DataFrame(metas)
    meta_df.to_parquet(out / f"extractions_{a.lang}.parquet", index=False)
    pd.DataFrame(rows_inst, columns=["case_id", "report_id", "lang", "finding_type", "slot"]).to_parquet(
        out / f"instances_{a.lang}.parquet", index=False)
    pd.DataFrame(rows_na, columns=["case_id", "report_id", "structure"]).to_parquet(
        out / f"not_assessable_{a.lang}.parquet", index=False)
    summary = {"out": str(out), "lang": a.lang, "mode": mode, "n_reports": int(len(meta_df)),
               "n_errors": int(meta_df["error"].notna().sum()),
               "n_salvaged": int(meta_df.get("salvaged", pd.Series(dtype=object)).fillna(False).astype(bool).sum()),
               "n_dropped_findings": int(pd.to_numeric(meta_df.get("dropped_findings", pd.Series(dtype=float)),
                                                       errors="coerce").fillna(0).sum()),
               "absence_regions_by_scope": scopes,
               "n_reports_with_absence_conflicts": int((pd.to_numeric(meta_df.get(
                   "absence_conflicts", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()),
               "retry_settings_used": pd.Series([
                   f"{p}:{t}" for p, t, e in zip(meta_df.get("repetition_penalty", [None] * len(meta_df)),
                                                 meta_df.get("temperature", [0.0] * len(meta_df)), meta_df["error"])
                   if pd.notna(p) and p and pd.isna(e)], dtype=object).value_counts().to_dict(),
               "errors": meta_df["error"].value_counts().to_dict(),
               "n_instances": len(rows_inst), "median_seconds": float(pd.to_numeric(meta_df.get("seconds", pd.Series(dtype=float)), errors="coerce").median()),
               "finish_reasons": meta_df.get("finish_reason", pd.Series(dtype=str)).value_counts().to_dict()}
    (out / f"summary_{a.lang}.json").write_text(json.dumps(summary, indent=2, default=str))
    write_provenance(out, extra={"manifest_run": str(a.manifest), "model": a.model, "model_path": a.model_path,
                                 "lang": a.lang, "structured_output_mode": mode})
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
