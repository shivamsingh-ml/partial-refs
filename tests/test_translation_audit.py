"""Translation audit against a fake vLLM server (synthetic report pairs only)."""

import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest
from pydantic import ValidationError

from partialref.extract import translation_audit as ta
from partialref.extract.score_parser import parser_half

IT = "Assenza di e.d. 36.\nCanale mandibolare destro a decorso regolare. Elemento 47 con otturazione."
EN = "Absence of tooth 36.\nRight mandibular canal with a regular course. Tooth 47 with a filling."
EN_WRONG = "Presence of tooth 36.\nRight mandibular canal with a regular course. Tooth 47 with a filling."


def _en_section(body: dict) -> tuple[int, str]:
    user = body["messages"][-1]["content"]
    it, en = user.split("\n\nENGLISH (translation):\n")
    return it.count("\n"), en                      # Italian block: one header line + one line per segment


class FakeAuditor(BaseHTTPRequestHandler):
    """Flags segment 1 as meaning-changed whenever the English differs from the registered clean rendering."""
    clean: set = set()
    seen: list = []
    skip_last_segment_first_try = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeAuditor.seen.append(body)
        if body.get("response_format", {}).get("json_schema", {}).get("name") != "pair_audit":
            content = "{}"                          # the structured-output probe of per_report.detect_mode
        else:
            assert body["temperature"] == 0.0 and body["chat_template_kwargs"] == {"enable_thinking": False}
            n_it, en = _en_section(body)
            bad = [] if en in self.clean else [{"category": "Other", "meaning_changed": True, "tooth_ids": []}]
            segs = [{"it": 1, "problems": bad}] + [{"it": i, "problems": []} for i in range(2, n_it + 1)]
            if self.skip_last_segment_first_try and "repetition_penalty" not in body:
                segs = segs[:-1]
            content = json.dumps({"segments": segs, "english_only": []})
        data = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


@pytest.fixture
def server():
    FakeAuditor.seen, FakeAuditor.skip_last_segment_first_try = [], False
    FakeAuditor.clean = {ta.pair_prompt([], ta.split_segments(EN)).split("ENGLISH (translation):\n")[1]}
    srv = HTTPServer(("127.0.0.1", 0), FakeAuditor)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_fewshot_examples_are_consistent():
    for it, en, out in ta.FEWSHOT:
        audit = ta.PairAudit.model_validate(out)
        n_it, n_en = len(ta.split_segments(it)), len(ta.split_segments(en))
        assert [s.it for s in audit.segments] == list(range(1, n_it + 1))      # exactly one entry per segment
        assert all(1 <= e.en <= n_en for e in audit.english_only)
    msgs = ta.messages(IT, EN)
    assert msgs[0]["role"] == "system" and len(msgs) == 2 + 2 * len(ta.FEWSHOT)
    assert msgs[-1]["content"].startswith("ITALIAN (original):\n[1] Assenza di e.d. 36.\n[2] Canale")


def test_split_segments():
    assert ta.split_segments("Mandibola: e.d. 38 incluso. E.d. 48 in arcata; Canale regolare.\n\nNon lesioni.") == \
        ["Mandibola: e.d. 38 incluso.", "E.d. 48 in arcata;", "Canale regolare.", "Non lesioni."]
    assert ta.split_segments("Presence of no. 4 implants in positions 3.6 and 4.6. normal bone.") == \
        ["Presence of no. 4 implants in positions 3.6 and 4.6. normal bone."]


def test_plant():
    def rng():
        return random.Random(3)

    text, detail = ta.plant(EN, "tooth", rng())
    old, new = detail.split("->")
    assert old in EN and new not in EN and text == EN.replace(old, new) and old[0] == new[0]
    assert ta.plant(EN, "side", rng())[0] == EN.replace("Right", "Left")
    assert ta.plant(EN, "polarity", rng())[0] == EN_WRONG
    dropped, where = ta.plant(EN, "drop", rng())
    assert where.startswith("en_segment") and len(ta.split_segments(dropped)) == 2
    assert ta.plant(EN, "tooth", rng()) == ta.plant(EN, "tooth", rng())            # deterministic
    # numbers that are not teeth are left alone; no place for an error -> None
    assert ta.plant("Lesion of 12 mm, angle 45°, canal 1.45 from the crest, 100 slices.", "tooth", rng()) is None
    for kind in ta.EXPECTED:
        assert ta.plant("No osteolytic lesions.", kind, rng()) is None
    assert ta.plant("THE LEFT SINUS IS ABSENT", "side", rng())[0] == "THE RIGHT SINUS IS ABSENT"
    assert ta.plant("The third molars are missing", "polarity", rng())[0] == "The third molars are present"


def test_clean_audit_and_rows():
    raw = ta.PairAudit.model_validate({
        "segments": [{"it": 1, "problems": []},
                     {"it": 2, "problems": [{"category": "Side", "meaning_changed": True, "tooth_ids": ["48"]}]},
                     {"it": 2, "problems": [{"category": "Untranslated", "meaning_changed": False}]},
                     {"it": 9, "problems": []}],
        "english_only": [{"en": 2, "meaning_changed": True, "tooth_ids": ["47"]}, {"en": 7, "meaning_changed": True}]})
    audit, info = ta.clean_audit(raw, n_it=4, n_en=3)
    assert info == {"n_it": 4, "n_en": 3, "n_unchecked": 2, "n_out_of_range": 2}
    rows = ta.problem_rows(audit)
    assert [(r["where"], r["segment"], r["category"], r["meaning_changed"], r["tooth_ids"]) for r in rows] == [
        ("it", 2, "Side", True, "48"), ("it", 2, "Untranslated", False, ""), ("en", 2, "FindingAdded", True, "47")]
    assert ta.n_meaning(audit.model_dump(mode="json")) == 2 and ta.n_meaning(None) is None
    with pytest.raises(ValidationError):                               # free text is not part of the schema
        ta.Problem.model_validate({"category": "Side", "meaning_changed": True, "note": "x"})
    assert [ta.known_class(v) for v in ("MISTRANSLATION (x)", "minor: y", "abbreviation kept", "faithful")] == \
        ["mistranslation", "minor", "abbreviation_kept", "faithful"]


def _manifest(tmp_path, cases):
    man = tmp_path / "manifest"
    man.mkdir()
    rows = []
    for cid, rid, en in cases:
        for lang, text in (("it", IT), ("en", en)):
            p = tmp_path / "data" / cid / f"reports_{lang}" / f"cbct_{rid}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            rows.append({"case_id": cid, "lang": lang, "report_id": rid, "path": str(p)})
    p = tmp_path / "data" / "P900" / "reports_it" / "cbct_21_9.txt"       # Italian only: not a pair
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(IT)
    rows.append({"case_id": "P900", "lang": "it", "report_id": "21_9", "path": str(p)})
    pd.DataFrame(rows).to_parquet(man / "reports.parquet")
    return man


def test_audit_end_to_end(tmp_path, server):
    ids = [f"P{i:03d}" for i in range(1, 60)]
    tune = [c for c in ids if parser_half(c) == "tune"][:3]
    held = [c for c in ids if parser_half(c) == "heldout"][:2]
    cases = [(tune[0], "30_1", EN_WRONG), (tune[1], "22_2", EN), (tune[2], "22_3", EN),
             (held[0], "30_4", EN), (held[1], "24_5", EN_WRONG)]
    man = _manifest(tmp_path, cases)
    known = tmp_path / "known.csv"
    pd.DataFrame([{"case_id": tune[0], "report_id": "30_1", "extent": "full pair", "verdict": "MISTRANSLATION (x)"},
                  {"case_id": tune[1], "report_id": "22_2", "extent": "full pair", "verdict": "faithful"},
                  {"case_id": held[0], "report_id": "30_4", "extent": "full pair", "verdict": "faithful"}]
                 ).to_csv(known, index=False)
    out = tmp_path / "run"
    base = ["--manifest", str(man), "--server", server, "--model", "fake", "--out-dir", str(out), "--workers", "2"]
    assert ta.main(base + ["--half", "tune", "--plant", "2", "--known", str(known)]) == 0
    pairs = pd.read_csv(out / "pairs.csv", dtype={"report_id": str})
    assert set(pairs["case_id"]) == set(tune) and set(pairs["half"]) == {"tune"}
    assert pairs.set_index("case_id")["n_meaning_changed"].to_dict() == {tune[0]: 1, tune[1]: 0, tune[2]: 0}
    assert pairs.set_index("case_id").loc[tune[0], "it_segments_meaning"] == 1
    planted = pd.read_csv(out / "planted.csv")
    assert len(planted) == 8 and set(planted["kind"]) == set(ta.EXPECTED)
    clean_pairs = planted[planted["case_id"] != tune[0]]
    assert clean_pairs["detected"].all() and not clean_pairs["expected_category"].any()   # fake answers "Other"
    wrong = planted[planted["case_id"] == tune[0]]              # already flagged: count does not go up
    assert not wrong["detected"].any()
    s = json.loads((out / "summary.json").read_text())
    assert s["n_pairs"] == 3 and s["n_errors"] == 0 and s["flagged_meaning_changed_by_half"] == {"tune": 1}
    assert s["known_not_audited"] == 1                          # the held-out pair is not in this run yet
    assert {(k["known"], k["n"], k["flagged"]) for k in s["known_check"]} == {("mistranslation", 1, 1), ("faithful", 1, 0)}
    # resume: nothing is audited twice (only the mode probe is sent again)
    n_calls = len(FakeAuditor.seen)
    assert ta.main(base + ["--half", "tune", "--plant", "2"]) == 0
    assert len(FakeAuditor.seen) - n_calls <= 2
    # extending the same run dir with the held-out half gives one table over both halves; planted files stay
    assert ta.main(base + ["--half", "heldout", "--known", str(known)]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["pairs_by_half"] == {"tune": 3, "heldout": 2} and s["known_not_audited"] == 0
    assert s["flagged_meaning_changed_by_half"] == {"tune": 1, "heldout": 1}
    assert s["flagged_meaning_changed_by_reader"] == {"30": 1, "24": 1}
    assert len(pd.read_csv(out / "planted.csv")) == 8
    # no report text in any output file
    for f in out.rglob("*"):
        if f.is_file():
            blob = f.read_bytes().lower()
            for word in (b"otturazione", b"filling", b"mandibular", b"assenza", b"absence"):
                assert word not in blob, (f.name, word)
    with pytest.raises(SystemExit):
        ta.main(base[:6] + ["--out-dir", str(tmp_path / "none"), "--half", "tune", "--limit-pairs", "0",
                            "--manifest", str(_empty_manifest(tmp_path))])


def _empty_manifest(tmp_path):
    man = tmp_path / "empty"
    man.mkdir()
    pd.DataFrame([{"case_id": "P900", "lang": "it", "report_id": "21_9", "path": "x"},
                  {"case_id": "P901", "lang": "en", "report_id": "21_8", "path": "y"}]).to_parquet(man / "reports.parquet")
    return man


def test_incomplete_answer_is_retried(server):
    FakeAuditor.skip_last_segment_first_try = True
    audit, meta = ta.audit_one(server, "fake", IT, EN, "response_format")
    assert meta["attempt"] == 2 and meta["repetition_penalty"] == 1.1 and meta["n_unchecked"] == 0
    assert meta["n_it"] == 3 and meta["n_en"] == 3 and ta.n_meaning(audit.model_dump(mode="json")) == 0
    audit, meta = ta.audit_one(server, "fake", IT, EN, "response_format", penalties=(None,))
    assert meta["n_unchecked"] == 1 and meta["error"] is None      # kept, but counted as incomplete
    audit, meta = ta.audit_one("http://127.0.0.1:9", "fake", IT, EN, "response_format", penalties=(None,))
    assert audit is None and meta["error"]
