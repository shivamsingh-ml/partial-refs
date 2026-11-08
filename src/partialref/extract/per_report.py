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
from partialref.extract.ontology import ReportExtraction, instances, json_schema
from partialref.provenance import make_run_id, write_provenance


def _post(url: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def request_body(model: str, report_text: str, mode: str) -> dict:
    body = {"model": model, "messages": messages(report_text), "temperature": 0.0,
            "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": False}}
    schema = json_schema()
    if mode == "response_format":
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "report_extraction", "schema": schema}}
    else:  # older vLLM spelling
        body["guided_json"] = schema
    return body


def extract_one(server: str, model: str, text: str, mode: str) -> tuple[ReportExtraction | None, dict]:
    t0 = time.time()
    meta: dict = {}
    try:
        resp = _post(f"{server}/v1/chat/completions", request_body(model, text, mode))
        content = resp["choices"][0]["message"]["content"]
        meta.update(prompt_tokens=resp.get("usage", {}).get("prompt_tokens"),
                    completion_tokens=resp.get("usage", {}).get("completion_tokens"),
                    finish_reason=resp["choices"][0].get("finish_reason"))
        ext = ReportExtraction.model_validate_json(content)
        meta["error"] = None
    except ValidationError as e:
        ext, meta["error"] = None, f"validation: {e.error_count()} errors"
    except urllib.error.HTTPError as e:
        ext, meta["error"] = None, f"http {e.code}"
    except Exception as e:  # noqa: BLE001  (record and continue)
        ext, meta["error"] = None, type(e).__name__
    meta["seconds"] = round(time.time() - t0, 2)
    return ext, meta


def detect_mode(server: str, model: str) -> str:
    """Use response_format if the server accepts it, else guided_json (version-dependent API)."""
    for mode in ("response_format", "guided_json"):
        ext, meta = extract_one(server, model, "Implant in position 36. No other findings.", mode)
        if ext is not None:
            return mode
    raise RuntimeError("server accepted neither structured-output spelling")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--lang", default="en", choices=["en", "it"])
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--model-path", default="", help="recorded in provenance")
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--case-ids", type=lambda s: [x for x in s.split(",") if x], default=[])
    ap.add_argument("--centres", default="PFSA")
    ap.add_argument("--out-dir", type=Path, help="existing run dir to resume")
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default="parse")
    a = ap.parse_args(argv)

    reps = pd.read_parquet(a.manifest / "reports.parquet")
    reps = reps[(reps["lang"] == a.lang) & reps["case_id"].str[0].isin(list(a.centres))]
    if a.case_ids:
        reps = reps[reps["case_id"].isin(a.case_ids)]
    if a.limit_cases:
        keep = sorted(reps["case_id"].unique())[: a.limit_cases]
        reps = reps[reps["case_id"].isin(keep)]
    reps = reps.sort_values(["case_id", "report_id"])

    out = a.out_dir or (a.results_dir / "extract" / make_run_id(f"{a.tag}-{a.lang}"))
    (out / "per_report").mkdir(parents=True, exist_ok=True)
    mode = detect_mode(a.server, a.model)
    print(f"structured-output mode: {mode}; reports: {len(reps)}", flush=True)

    def work(row) -> dict:
        dst = out / "per_report" / f"{row.case_id}__{row.report_id}__{a.lang}.json"
        if dst.exists():
            return json.loads(dst.read_text())["meta"]
        text = Path(row.path).read_text(encoding="utf-8", errors="replace")
        ext, meta = extract_one(a.server, a.model, text, mode)
        meta.update(case_id=row.case_id, report_id=row.report_id, lang=a.lang,
                    n_findings=len(ext.findings) if ext else None)
        dst.write_text(json.dumps({"meta": meta, "extraction": ext.model_dump(mode="json") if ext else None}))
        return meta

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        metas = list(ex.map(work, reps.itertuples()))

    rows_inst, rows_na = [], []
    for f in sorted((out / "per_report").glob(f"*__{a.lang}.json")):
        d = json.loads(f.read_text())
        if not d["extraction"]:
            continue
        ext = ReportExtraction.model_validate(d["extraction"])
        m = d["meta"]
        rows_inst += [{"case_id": c, "report_id": m["report_id"], "lang": a.lang, "finding_type": t, "slot": s}
                      for c, t, s in instances(m["case_id"], ext.findings)]
        rows_na += [{"case_id": m["case_id"], "report_id": m["report_id"], "structure": s}
                    for s in ext.not_assessable]
    meta_df = pd.DataFrame(metas)
    meta_df.to_parquet(out / f"extractions_{a.lang}.parquet", index=False)
    pd.DataFrame(rows_inst, columns=["case_id", "report_id", "lang", "finding_type", "slot"]).to_parquet(
        out / f"instances_{a.lang}.parquet", index=False)
    pd.DataFrame(rows_na, columns=["case_id", "report_id", "structure"]).to_parquet(
        out / f"not_assessable_{a.lang}.parquet", index=False)
    summary = {"out": str(out), "lang": a.lang, "mode": mode, "n_reports": int(len(meta_df)),
               "n_errors": int(meta_df["error"].notna().sum()),
               "errors": meta_df["error"].value_counts().to_dict(),
               "n_instances": len(rows_inst), "median_seconds": float(meta_df["seconds"].median()),
               "finish_reasons": meta_df.get("finish_reason", pd.Series(dtype=str)).value_counts().to_dict()}
    (out / f"summary_{a.lang}.json").write_text(json.dumps(summary, indent=2, default=str))
    write_provenance(out, extra={"manifest_run": str(a.manifest), "model": a.model, "model_path": a.model_path,
                                 "lang": a.lang, "structured_output_mode": mode})
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
