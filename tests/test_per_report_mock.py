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
    inst = pd.read_parquet(out / "instances_en.parquet")
    assert set(inst["finding_type"]) == {"DentalImplant", "Caries"} and len(inst) == 9   # 3 slots x 3 reports
    assert len(pd.read_parquet(out / "not_assessable_en.parquet")) == 3
    assert len(list((out / "per_report").glob("*__en.json"))) == 3
    n_calls = len(FakeVLLM.seen)
    # resume: second run makes no new report calls (only the mode probe)
    per_report.main(["--manifest", str(man), "--lang", "en", "--server", server, "--model", "fake",
                     "--out-dir", str(out), "--workers", "2"])
    assert len(FakeVLLM.seen) - n_calls <= 2
    # no report text in any output file
    for f in out.rglob("*"):
        if f.is_file() and f.suffix in {".json", ".parquet"}:
            assert b"synthetic placeholder" not in f.read_bytes()
    # scoring works on these outputs (self vs self -> F1 = 1)
    assert score_main(["--ours", str(out / "instances_en.parquet"), "--lumetti", str(out / "instances_en.parquet"),
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
