"""End-to-end test of the parser client against a fake vLLM server (synthetic reports only)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from partialref.extract import per_report
from partialref.extract.score_parser import main as score_main

ANSWER = {"findings": [{"finding_type": "DentalImplant", "tooth_ids": ["36"]},
                       {"finding_type": "Caries", "tooth_ids": ["14", "15"], "qualifiers": {"severity": ["Mild"]}}],
          "not_assessable": ["maxilla"]}


class FakeVLLM(BaseHTTPRequestHandler):
    reject_response_format = True
    seen: list = []

    def log_message(self, *a):  # keep test output quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.seen.append(body)
        if self.reject_response_format and "response_format" in body:
            self.send_response(400); self.end_headers(); return
        assert body["temperature"] == 0.0 and body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "structured_outputs" in body and "json" in body["structured_outputs"]
        out = {"choices": [{"message": {"content": json.dumps(ANSWER)}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), FakeVLLM)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_parser_end_to_end(tmp_path, server):
    man = tmp_path / "manifest"
    man.mkdir()
    rows = []
    for cid, rid in [("P001", "21_100"), ("P001", "26_101"), ("A003", "34_102")]:
        for lang in ("en", "it"):
            p = tmp_path / "data" / cid / f"reports_{lang}" / f"cbct_{rid}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("synthetic placeholder text")
            rows.append({"case_id": cid, "lang": lang, "report_id": rid, "path": str(p)})
    pd.DataFrame(rows).to_parquet(man / "reports.parquet")
    out = tmp_path / "run"
    assert per_report.main(["--manifest", str(man), "--lang", "en", "--server", server, "--model", "fake",
                            "--out-dir", str(out), "--workers", "2"]) == 0
    s = json.loads((out / "summary_en.json").read_text())
    assert s["mode"] == "structured_outputs" and s["n_reports"] == 3 and s["n_errors"] == 0
    assert s["absence_regions_by_scope"] == {}
    inst = pd.read_parquet(out / "instances_en.parquet")
    assert set(inst["finding_type"]) == {"DentalImplant", "Caries"} and len(inst) == 9   # 3 slots x 3 reports
    assert len(pd.read_parquet(out / "not_assessable_en.parquet")) == 3
    assert len(list((out / "per_report").glob("*__en.json"))) == 3
    n_calls = len(FakeVLLM.seen)
    # resume: second run makes no new report calls (only the mode probe)
    per_report.main(["--manifest", str(man), "--lang", "en", "--server", server, "--model", "fake",
                     "--out-dir", str(out), "--workers", "2"])
    assert len(FakeVLLM.seen) - n_calls <= 2
    # --case-ids: ids are stripped; an unknown id or an empty selection stops before any request
    sub_out, n_calls = tmp_path / "sub", len(FakeVLLM.seen)
    base = ["--manifest", str(man), "--lang", "en", "--server", server, "--model", "fake", "--out-dir", str(sub_out)]
    for extra in (["--case-ids", "P001,P999"], ["--case-ids", " , "], ["--centres", "X"]):
        with pytest.raises(SystemExit):
            per_report.main(base + extra)
    assert len(FakeVLLM.seen) == n_calls and not sub_out.exists()
    assert per_report.main(base + ["--case-ids", "P001, A003 "]) == 0
    assert json.loads((sub_out / "summary_en.json").read_text())["n_reports"] == 3
    assert per_report.main(base + ["--case-ids", "A003"]) == 0                 # same dir: summary covers all files
    # no report text in any output file
    for f in out.rglob("*"):
        if f.is_file() and f.suffix in {".json", ".parquet"}:
            assert b"synthetic placeholder" not in f.read_bytes()
    # scoring works on these outputs (self vs self -> F1 = 1)
    assert score_main(["--ours", str(out / "instances_en.parquet"), "--ref-parse", str(out / "instances_en.parquet"),
                       "--results-dir", str(tmp_path / "res")]) == 0


class LoopingVLLM(FakeVLLM):
    """First attempt 'loops' (truncated JSON, finish_reason=length); retry with repetition_penalty works."""
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.seen.append(body)
        if "repetition_penalty" in body:
            content, reason = json.dumps(ANSWER), "stop"
        else:
            content, reason = '{"findings": [{"finding_type": "Caries", "tooth_ids": ["36"', "length"
        data = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": reason}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 4096}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


def test_retry_with_repetition_penalty(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), LoopingVLLM)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        ext, meta = per_report.extract_one(url, "fake", "synthetic", "response_format")
        assert ext is None and meta["finish_reason"] == "length"
        ext2, meta2 = per_report.extract_one(url, "fake", "synthetic", "response_format", repetition_penalty=1.1)
        assert ext2 is not None and meta2["repetition_penalty"] == 1.1
    finally:
        srv.shutdown()


def test_salvage_schema_valid_but_invalid_findings():
    """Review #1: schema-valid output that breaks the per-type anchor rule no longer loses the report."""
    bad = {"findings": [{"finding_type": "MandibularCanalCourse", "tooth_ids": ["38", "48"]},   # no side
                        {"finding_type": "JawAtrophy", "tooth_ids": ["46"]},                        # no quadrant
                        {"finding_type": "ToothAbsence", "tooth_ids": []},                          # unanchorable
                        {"finding_type": "Caries", "tooth_ids": ["36"]}],
           "not_assessable": []}
    ext, info = per_report.parse_extraction(json.dumps(bad))
    slots = {(f.finding_type.value, s) for f in ext.findings for s in f.slots()}
    assert slots == {("MandibularCanalCourse", "Left"), ("MandibularCanalCourse", "Right"),
                     ("JawAtrophy", "QuadrantIV"), ("Caries", "36")}
    assert info == {"salvaged": True, "repaired_findings": 2, "dropped_findings": 1,
                    "dropped_reasons": ["ToothAbsence: Value error, ToothAbsence needs tooth_ids"]}
    canal = {f.side: f.tooth_ids for f in ext.findings if f.finding_type.value == "MandibularCanalCourse"}
    assert canal == {"Left": ["38"], "Right": ["48"]}                 # tooth ids split per derived side
    ok, info_ok = per_report.parse_extraction(json.dumps(ANSWER))
    assert info_ok == {"salvaged": False} and len(ok.findings) == 2
    # a finding_type that is not a string is dropped with reason "?", not a crash; no input value in the reason
    odd = {"findings": [{"finding_type": {"x": "free text"}, "tooth_ids": ["36"]},
                        {"finding_type": "Caries", "tooth_ids": ["36"]}], "not_assessable": []}
    ext2, info2 = per_report.parse_extraction(json.dumps(odd))
    assert len(ext2.findings) == 1 and info2["dropped_findings"] == 1
    assert info2["dropped_reasons"][0].startswith("?: ") and "free text" not in info2["dropped_reasons"][0]
    with pytest.raises(json.JSONDecodeError):
        per_report.parse_extraction('{"findings": [{"finding_type": "Caries", "tooth_ids": ["36"')


def test_resume_survives_partial_file(tmp_path, server):
    man = tmp_path / "manifest"
    man.mkdir()
    p = tmp_path / "data" / "P001" / "reports_en" / "cbct_21_100.txt"
    p.parent.mkdir(parents=True)
    p.write_text("synthetic placeholder text")
    pd.DataFrame([{"case_id": "P001", "lang": "en", "report_id": "21_100", "path": str(p)}]).to_parquet(
        man / "reports.parquet")
    out = tmp_path / "run"
    (out / "per_report").mkdir(parents=True)
    (out / "per_report" / "P001__21_100__en.json").write_text('{"meta": {"case_id": "P0')   # killed mid-write
    assert per_report.main(["--manifest", str(man), "--lang", "en", "--server", server, "--model", "fake",
                            "--out-dir", str(out), "--workers", "1"]) == 0
    d = json.loads((out / "per_report" / "P001__21_100__en.json").read_text())
    assert d["extraction"] is not None and not list((out / "per_report").glob("*.tmp"))


def test_report_ok_treats_dropped_findings_as_failed():
    meta = pd.DataFrame({"error": [None, None, "invalid_json", None],
                         "dropped_findings": [None, 0, None, 2]})
    assert per_report.report_ok(meta).tolist() == [True, True, False, False]
    assert per_report.report_ok(meta.drop(columns="dropped_findings")).tolist() == [True, True, False, True]


class StubbornLoopVLLM(FakeVLLM):
    """Loops at temperature 0 whatever the penalty; only seeded sampling (temperature > 0) succeeds."""
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.seen.append(body)
        probe = "Implant in position 36" in body["messages"][-1]["content"]    # mode probe at startup
        ok = probe or (body.get("temperature", 0) > 0 and body.get("seed") == 0)
        content = json.dumps(ANSWER) if ok else '{"findings": [{"finding_type": "Caries"'
        data = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop" if ok else "length"}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 4096}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


def test_retry_ladder_reaches_seeded_sampling(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), StubbornLoopVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    man = tmp_path / "manifest"
    man.mkdir()
    p = tmp_path / "data" / "P001" / "reports_en" / "cbct_21_100.txt"
    p.parent.mkdir(parents=True)
    p.write_text("synthetic placeholder text")
    pd.DataFrame([{"case_id": "P001", "lang": "en", "report_id": "21_100", "path": str(p)}]).to_parquet(
        man / "reports.parquet")
    out = tmp_path / "run"
    try:
        assert per_report.main(["--manifest", str(man), "--lang", "en", "--server", url, "--model", "fake",
                                "--out-dir", str(out), "--workers", "1"]) == 0
    finally:
        srv.shutdown()
    d = json.loads((out / "per_report" / "P001__21_100__en.json").read_text())
    assert d["extraction"] is not None
    m = d["meta"]
    assert m["attempts_made"] == 4 and m["temperature"] == 0.3 and m["seed"] == 0 and len(m["previous_errors"]) == 3
    assert m["absence_conflicts"] == 0
    s = json.loads((out / "summary_en.json").read_text())
    assert s["n_errors"] == 0 and s["retry_settings_used"] == {"1.05:0.3": 1}
    assert per_report.parse_ladder("1.1:0,1.3:0") == [(1.1, 0.0), (1.3, 0.0)] and per_report.parse_ladder("") == []


class DropThenCleanVLLM(FakeVLLM):
    """First answer is schema-valid but has an unanchorable finding (salvage drops it); retry is clean."""
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.seen.append(body)
        dirty = {"findings": ANSWER["findings"] + [{"finding_type": "ToothAbsence", "tooth_ids": []}],
                 "not_assessable": []}
        content = json.dumps(ANSWER if body.get("repetition_penalty") else dirty)
        data = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 50}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


def test_salvage_with_dropped_findings_is_retried(tmp_path):
    """Review v3: a parse that lost findings in salvage is not OK for T2 -> it must be retried."""
    srv = HTTPServer(("127.0.0.1", 0), DropThenCleanVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    man = tmp_path / "manifest"
    man.mkdir()
    p = tmp_path / "data" / "P001" / "reports_en" / "cbct_21_100.txt"
    p.parent.mkdir(parents=True)
    p.write_text("synthetic placeholder text")
    pd.DataFrame([{"case_id": "P001", "lang": "en", "report_id": "21_100", "path": str(p)}]).to_parquet(
        man / "reports.parquet")
    out = tmp_path / "run"
    try:
        per_report.main(["--manifest", str(man), "--lang", "en", "--server", url, "--model", "fake",
                         "--out-dir", str(out), "--workers", "1"])
    finally:
        srv.shutdown()
    m = json.loads((out / "per_report" / "P001__21_100__en.json").read_text())["meta"]
    assert m["attempts_made"] == 2 and m["first_attempt_error"] == "dropped_findings: 1"
    assert not m.get("dropped_findings") and m["repetition_penalty"] == 1.1
    meta = pd.read_parquet(out / "extractions_en.parquet")
    assert per_report.report_ok(meta).all()


def test_conflicts_recorded_not_resolved_and_optional_convention():
    """v3.1 after review + mask check: parsing stores findings as given and records conflict details;
    the narrowed convention is optional (RootResidue and eruption/development TDP only)."""
    from partialref.extract.ontology import absence_conflict_details, apply_absence_convention
    out = {"findings": [{"finding_type": "ToothAbsence", "tooth_ids": ["36", "38", "46", "35"]},
                        {"finding_type": "RootResidue", "tooth_ids": ["36"]},
                        {"finding_type": "ToothDevelopmentPathology", "tooth_ids": ["38"],
                         "qualifiers": {"eruption_status": ["Impacted"]}},
                        {"finding_type": "ToothDevelopmentPathology", "tooth_ids": ["46"],
                         "qualifiers": {"periodontal_feature": ["VerticalDefect"]}},          # extraction site
                        {"finding_type": "ToothDevelopmentPathology", "tooth_ids": ["35"],
                         "qualifiers": {"development_feature": ["Deciduous"]}},               # agenesis case
                        {"finding_type": "ToothAbsence", "tooth_ids": ["48"], "region": "LowerArch"},
                        {"finding_type": "ToothDevelopmentPathology", "tooth_ids": ["48"],
                         "qualifiers": {"bone_inclusion": ["FullInclusion"]}}],
           "not_assessable": []}
    ext, info = per_report.parse_extraction(json.dumps(out))
    absent = sorted(t for f in ext.findings if f.finding_type.value == "ToothAbsence" for t in f.tooth_ids)
    assert absent == ["35", "36", "38", "46", "48"] and "absence_convention_removed" not in info   # untouched
    kinds = {d["tooth"]: d["with"] for d in absence_conflict_details(ext.findings)}
    assert kinds == {"36": "RootResidue", "38": "TDP_eruption_development", "46": "TDP_periodontal",
                     "35": "TDP_deciduous_supernumerary", "48": "TDP_eruption_development"}
    fixed, n = apply_absence_convention(ext.findings)
    left = sorted(t for f in fixed if f.finding_type.value == "ToothAbsence" for t in f.tooth_ids)
    assert left == ["35", "46"] and n == 3
    assert not any(f.finding_type.value == "ToothAbsence" and f.region for f in fixed)   # no fallback to region


def test_retry_errors_reparses_reports_with_dropped_findings(tmp_path, server):
    man = tmp_path / "manifest"
    man.mkdir()
    p = tmp_path / "data" / "P001" / "reports_en" / "cbct_21_100.txt"
    p.parent.mkdir(parents=True)
    p.write_text("synthetic placeholder text")
    pd.DataFrame([{"case_id": "P001", "lang": "en", "report_id": "21_100", "path": str(p)}]).to_parquet(
        man / "reports.parquet")
    out = tmp_path / "run"
    (out / "per_report").mkdir(parents=True)
    stored = {"meta": {"case_id": "P001", "report_id": "21_100", "lang": "en", "error": None, "dropped_findings": 2},
              "extraction": {"findings": [], "not_assessable": []}}
    f = out / "per_report" / "P001__21_100__en.json"
    f.write_text(json.dumps(stored))
    args = ["--manifest", str(man), "--lang", "en", "--server", server, "--model", "fake", "--out-dir", str(out),
            "--workers", "1"]
    per_report.main(args)                                              # plain resume keeps the stored file
    assert json.loads(f.read_text())["meta"]["dropped_findings"] == 2
    per_report.main(args + ["--retry-errors"])                         # --retry-errors re-parses it
    d = json.loads(f.read_text())
    assert not d["meta"].get("dropped_findings") and len(d["extraction"]["findings"]) == 2
