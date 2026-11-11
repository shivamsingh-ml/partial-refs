"""Report review app (synthetic reports only): access gate, display, explore mode saves nothing, logging codes only."""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from partialref.review.data import case_order, finding_rows, load_reports, log_review, read_log

APP = str(Path(__file__).resolve().parents[1] / "src" / "partialref" / "review" / "app.py")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    man = tmp_path / "manifest"
    man.mkdir()
    run = tmp_path / "run"
    (run / "per_report").mkdir(parents=True)
    rows = []
    for cid, rid, findings in [("P001", "21_100", [{"finding_type": "Caries", "tooth_ids": ["36"]}]),
                               ("P001", "26_101", [{"finding_type": "ToothAbsence", "region": "UpperArch"}]),
                               ("F002", "27_102", [])]:
        for lang in ("en", "it"):
            p = tmp_path / "data" / cid / f"reports_{lang}" / f"cbct_{rid}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"SYNTHETIC REPORT {cid} {rid} {lang}")
            rows.append({"case_id": cid, "lang": lang, "report_id": rid, "path": str(p)})
        (run / "per_report" / f"{cid}__{rid}__en.json").write_text(json.dumps(
            {"meta": {"case_id": cid, "report_id": rid, "error": None},
             "extraction": {"findings": findings, "not_assessable": []}}))
    pd.DataFrame(rows).to_parquet(man / "reports.parquet")
    for k, v in {"REVIEW_TOKEN": "secret-token", "REVIEW_PARSE_RUN": str(run), "REVIEW_MANIFEST": str(man),
                 "REVIEW_LOG_ROOT": str(tmp_path / "review")}.items():
        monkeypatch.setenv(k, v)
    return tmp_path, run, man


def test_gate_blocks_without_token(setup):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert len(at.text_input) == 1 and not at.tabs          # only the token box is shown
    assert not any("SYNTHETIC" in t.value for t in at.text)
    at.text_input[0].input("wrong").run()
    assert not any("SYNTHETIC" in t.value for t in at.text)


def test_explore_shows_report_and_findings_and_saves_nothing(setup):
    tmp_path, run, _ = setup
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].input("secret-token").run()
    at.selectbox[0].select("all").run()                      # both halves
    shown = " ".join(t.value for t in at.text)
    assert "SYNTHETIC REPORT" in shown and len(at.dataframe) >= 1
    assert not (tmp_path / "review").exists()                  # explore mode: nothing written
    n_select = len(at.selectbox)
    next(c for c in at.checkbox if c.label == "Enable logging").check().run()
    assert len(at.selectbox) > n_select                        # log form appears ...
    assert not (tmp_path / "review").exists()                  # ... but nothing is saved until "Save"
    save = [b for b in at.button if b.label == "Save this report's decisions"]
    save[0].click().run()
    log = (tmp_path / "review" / run.name / "review_log.jsonl").read_text()
    assert "SYNTHETIC" not in log and json.loads(log.splitlines()[0])["verdicts"] == []


def test_logging_writes_codes_only(tmp_path):
    f = log_review(tmp_path, {"case_id": "P001", "report_id": "21_100", "verdicts": [{"idx": 0, "verdict": "correct"}],
                              "missed": [], "flags": ["translation_looks_odd"]})
    rec = json.loads(f.read_text().splitlines()[0])
    assert rec["case_id"] == "P001" and "logged_at" in rec
    with pytest.raises(ValueError):
        log_review(tmp_path, {"case_id": "P001", "note": "x" * 200})   # no long free text
    assert len(read_log(tmp_path)) == 1


def test_data_helpers(setup):
    _, _, man = setup
    r = load_reports(man, "en")
    assert set(r["case_id"]) == {"P001", "F002"} and set(r["half"]) <= {"tune", "heldout"}
    order = case_order(r, "all", ["P", "F"], 0, only_multi=True)
    assert order == ["P001"]
    assert case_order(r, "all", ["P", "F"], 0, False) == case_order(r, "all", ["P", "F"], 0, False)  # stable
    fr = finding_rows({"findings": [{"finding_type": "ToothAbsence", "region": "UpperArch"},
                                    {"finding_type": "Caries", "tooth_ids": ["36"], "qualifiers": {"severity": ["Mild"]}}]})
    assert fr["anchor"].tolist() == ["region UpperArch", "36"] and fr["qualifiers"].iloc[1] == "severity=Mild"
    # v3.2: an AllMissing region (tooth ids filled in by code) is shown as the statement, and import accepts it
    from partialref.extract.ontology import Finding
    from partialref.review.import_log import ANCHOR_RE
    dumps = [Finding(finding_type="ToothAbsence", region="LowerArch", region_scope="AllMissing",
                     remaining_tooth_ids=r).model_dump(mode="json") for r in ([], ["33", "43"])]
    dumps.append(Finding(finding_type="ToothAbsence", region="QuadrantII", region_scope="SomeMissing").model_dump(mode="json"))
    dumps.append({"finding_type": "ToothAbsence", "tooth_ids": [f"{q}{i}" for q in "1234" for i in range(1, 9)]})
    anchors = finding_rows({"findings": dumps})["anchor"].tolist()
    assert anchors[:3] == ["region LowerArch AllMissing", "region LowerArch AllMissing (remaining 33, 43)",
                           "region QuadrantII"]
    assert all(ANCHOR_RE.match(x) and len(x) <= 210 for x in anchors)
    assert not ANCHOR_RE.match("region LowerArch AllMissing (remaining free text)")


def test_bundle_local_mode_and_import_round_trip(setup, tmp_path, monkeypatch):
    """Export bundle -> unpack -> app in local mode (no token) -> save -> import_log validates."""
    import tarfile

    from partialref.review.export_bundle import build_bundle
    from partialref.review.import_log import main as import_main
    _, run, man = setup
    tgz = build_bundle(run, man, ["en", "it"], tmp_path / "bundles")
    assert oct(tgz.stat().st_mode)[-3:] == "600"
    with tarfile.open(tgz) as t:
        names = t.getnames()
        t.extractall(tmp_path / "laptop", filter="data")
    assert any(n.endswith("reports_it/cbct_21_100.txt") for n in names)
    b = tmp_path / "laptop" / f"review-bundle-{run.name}"
    info = json.loads((b / "bundle_info.json").read_text())
    assert info["n_reports"] == 6 and info["n_parse_files"] == 3
    monkeypatch.delenv("REVIEW_TOKEN")
    from streamlit import config as st_config
    st_config.set_option("server.address", "127.0.0.1")          # what partialref.review.local passes
    for k, v in {"REVIEW_LOCAL": "1", "REVIEW_PARSE_RUN": str(b / "parse" / run.name), "REVIEW_MANIFEST": str(b),
                 "REVIEW_LOG_ROOT": str(b / "logs")}.items():
        monkeypatch.setenv(k, v)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox[0].select("all").run()
    assert "SYNTHETIC REPORT" in " ".join(t.value for t in at.text)       # no token needed locally
    next(c for c in at.checkbox if c.label == "Enable logging").check().run()
    next(b_ for b_ in at.button if b_.label == "Save this report's decisions").click().run()
    log = b / "logs" / run.name / "review_log.jsonl"
    assert log.exists()
    # bring it back: valid line imported, a tampered line rejected
    incoming = tmp_path / "incoming.jsonl"
    good = log.read_text().splitlines()[0]
    bad = json.dumps({**json.loads(good), "note": "free text not allowed"})
    incoming.write_text(good + "\n" + bad + "\n")
    rc = import_main([str(incoming), "--manifest", str(man), "--review-root", str(tmp_path / "res_review"),
                      "--extract-root", str(run.parent)])
    st_config.set_option("server.address", "")
    assert rc == 1                                                            # one rejected line
    imported = (tmp_path / "res_review" / run.name / "review_log.jsonl").read_text().splitlines()
    assert len(imported) == 1 and "SYNTHETIC" not in imported[0]


def test_local_flag_without_loopback_still_needs_token(setup, monkeypatch):
    """Review B2: REVIEW_LOCAL inherited on the cluster (0.0.0.0) must not bypass the token gate."""
    monkeypatch.setenv("REVIEW_LOCAL", "1")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert len(at.text_input) == 1 and not any("SYNTHETIC" in t.value for t in at.text)
    at.text_input[0].input("wrong ünïcode").run()                 # non-ASCII entry: no crash, still locked
    assert not at.exception and not any("SYNTHETIC" in t.value for t in at.text)


def test_import_rejects_crafted_records():
    from partialref.review.import_log import check_record
    known, runs = {("P001", "21_100")}, {"run1"}
    good = {"logged_at": "2026-11-10T10:00:00", "parse_run": "run1", "lang": "en", "case_id": "P001",
            "report_id": "21_100", "half": "tune",
            "verdicts": [{"idx": 0, "finding_type": "Caries", "anchor": "36, 37", "verdict": "wrong_slot",
                          "correct_slot": "46"}],
            "missed": [{"finding_type": "Caries", "slots": ["36"]}], "flags": ["translation_looks_odd"]}
    assert check_record(good, known, runs) == []
    bad = [
        {**good, "verdicts": [{**good["verdicts"][0], "idx": "x" * 50}]},
        {**good, "verdicts": [{**good["verdicts"][0], "idx": {"t": "text"}}]},
        {**good, "verdicts": [""]},
        {**good, "verdicts": [{**good["verdicts"][0], "correct_slot": ["36"]}]},
        {**good, "verdicts": [{**good["verdicts"][0], "anchor": "the report says something"}]},
        {**good, "missed": [{"finding_type": "Caries", "slots": {"36": "text"}}]},
        {**good, "flags": {"parser_missed_findings": "text"}},
        {**good, "half": "free text here"}, {**good, "lang": "xx"}, {**good, "logged_at": "yesterday"},
        {**good, "parse_run": "../../escaped"}, {**good, "parse_run": "unknown_run"},
        {**good, "note": "free"}, {k: v for k, v in good.items() if k != "flags"},
    ]
    for b in bad:
        errs = check_record(b, known, runs)
        assert errs and all(e.isidentifier() for e in errs)       # codes only, never content


# ---- fixed review list (partialref.review.sample): synthetic IDs only -------------------------------------

def _synthetic_population():
    """300 synthetic double-read cases (P/F/S) + single-read and excluded ones; one common and two rare types."""
    from partialref.extract.score_parser import parser_half
    rows, inst = [], []
    for i in range(1, 301):
        cid = f"{'PPPPFS'[i % 6]}{i:04d}"
        for j in (1, 2):
            rid = f"{20 + j}_{i}"
            rows.append({"case_id": cid, "report_id": rid, "lang": "en"})
            inst.append({"case_id": cid, "report_id": rid, "finding_type": "ToothAbsence", "slot": "36"})
            if (i + j) % 9 == 0:
                inst.append({"case_id": cid, "report_id": rid, "finding_type": "Caries", "slot": "16"})
            if i % 150 == 0:
                inst.append({"case_id": cid, "report_id": rid, "finding_type": "MandibularCondyleFinding", "slot": "Left"})
    rows += [{"case_id": "P9001", "report_id": "21_1", "lang": "en"},                       # single-read
             {"case_id": "S0015", "report_id": "21_2", "lang": "en"}, {"case_id": "S0015", "report_id": "22_2", "lang": "en"}]
    return pd.DataFrame(rows), pd.DataFrame(inst), parser_half


def test_review_sample_population_and_strata():
    from partialref.review.sample import TYPES, draw, population
    reps, inst, parser_half = _synthetic_population()
    pop = population(reps, "heldout", "PFS", ["S0015"])
    assert len(pop) and set(pop["case_id"].map(parser_half)) == {"heldout"}
    assert not {"P9001", "S0015"} & set(pop["case_id"]) and (pop.groupby("case_id").size() == 2).all()
    assert set(population(reps, "all", "F", [])["centre"]) == {"F"}
    entries, counts = draw(pop, inst, n_base=40, target=10, seed=1)
    base = [e for e in entries if e["stratum"] == "base"]
    extra = [e for e in entries if e["stratum"] == "extra"]
    keys = [(e["case_id"], e["report_id"]) for e in entries]
    assert len(base) == 40 and entries[:40] == base and len(set(keys)) == len(keys)       # base first, no repeats
    assert all(e["judge_types"] is None for e in base) and all(e["judge_types"] for e in extra)
    by_centre = pd.Series([e["case_id"][0] for e in base]).value_counts().to_dict()
    assert by_centre == {k: v for k, v in counts["base_by_centre"].items() if v}
    share = pop["centre"].value_counts(normalize=True)
    assert all(abs(by_centre.get(c, 0) - 40 * share[c]) < 1 for c in share.index)          # proportional
    found = {(c, r, t) for c, r, t in inst[["case_id", "report_id", "finding_type"]].itertuples(index=False)}
    assert all((e["case_id"], e["report_id"], t) in found for e in extra for t in e["judge_types"])
    c = counts["by_type"]
    assert c["ToothAbsence"] == {"base_reports_with_type": 40, "extra_reports": 0, "pool_outside_base": len(pop) - 40}
    n_caries = sum(("Caries" in (e["judge_types"] or [])) for e in extra)
    assert c["Caries"]["base_reports_with_type"] < 10 and n_caries == c["Caries"]["extra_reports"]
    assert c["Caries"]["base_reports_with_type"] + n_caries == 10                          # filled up to target
    rare = c["MandibularCondyleFinding"]                                                   # pool smaller than need
    assert rare["extra_reports"] == rare["pool_outside_base"] <= 4
    assert set(c) == set(TYPES)
    assert draw(pop, inst, 40, 10, 1)[0] == entries and draw(pop, inst, 40, 10, 2)[0] != entries   # seeded
    assert len(draw(pop, inst, 10 ** 6, 10, 1)[0]) == len(pop)                             # never more than exist


def test_review_list_loader_rejects_bad_entries(setup, tmp_path):
    from partialref.review.data import load_review_list
    _, _, man = setup
    reps = load_reports(man, "en")
    good = [{"case_id": "P001", "report_id": "21_100", "stratum": "base", "judge_types": None},
            {"case_id": "P001", "report_id": "26_101", "stratum": "extra", "judge_types": ["ToothAbsence"]}]
    f = tmp_path / "list.json"
    f.write_text(json.dumps({"meta": {}, "entries": good}))
    assert load_review_list(f, reps) == good
    for bad in ([{**good[0], "case_id": "P999"}], [{**good[0], "stratum": "other"}], [{**good[0], "judge_types": ["Caries"]}],
                [{**good[1], "judge_types": None}], [{**good[1], "judge_types": ["free text"]}], [{**good[1], "judge_types": []}],
                [{**good[0], "note": "x"}], [good[0], good[0]]):
        f.write_text(json.dumps({"entries": bad}))
        with pytest.raises(ValueError):
            load_review_list(f, reps)


def test_list_mode_steps_through_the_list_and_bundle_holds_only_listed_reports(setup, tmp_path, monkeypatch):
    import tarfile

    from partialref.review.export_bundle import build_bundle
    _, run, man = setup
    lst = tmp_path / "review-list.json"
    lst.write_text(json.dumps({"meta": {"seed": 1}, "entries": [
        {"case_id": "P001", "report_id": "21_100", "stratum": "base", "judge_types": None},
        {"case_id": "P001", "report_id": "26_101", "stratum": "extra", "judge_types": ["Caries"]}]}))
    monkeypatch.setenv("REVIEW_LIST", str(lst))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].input("secret-token").run()
    assert not at.exception and not any(s.label == "Cases" for s in at.selectbox)           # no free filter
    shown = " ".join(t.value for t in at.text)
    assert "SYNTHETIC REPORT P001 21_100 en" in shown and "26_101" not in shown and "F002" not in shown
    assert sum(s.label.startswith("type ") for s in at.selectbox) == 6                       # missed rows (base)
    assert any(s.label == "Caries · 36" for s in at.selectbox)
    next(b for b in at.button if b.label == "Save this report's decisions").click().run()    # logging is on
    log = tmp_path / "review" / run.name / "review_log.jsonl"
    assert len(log.read_text().splitlines()) == 1 and any("no verdict" in w.value for w in at.warning)
    next(b for b in at.button if b.label == "Next ▶").click().run()
    shown = " ".join(t.value for t in at.text)
    assert "SYNTHETIC REPORT P001 26_101 en" in shown and "21_100 en" not in shown
    assert any("Extra report for: Caries" in i.value for i in at.info)
    assert not any(s.label.startswith("type ") for s in at.selectbox)                        # extra: no missed rows
    assert not any(s.label.startswith("ToothAbsence") for s in at.selectbox)                 # other types not judged
    next(b for b in at.button if b.label == "Previous" or b.label == "◀ Previous").click().run()
    next(b for b in at.button if b.label == "Next report without a saved decision").click().run()
    assert "SYNTHETIC REPORT P001 26_101 en" in " ".join(t.value for t in at.text)           # first one not saved yet

    tgz = build_bundle(run, man, ["en", "it"], tmp_path / "bundles", review_list=lst)
    with tarfile.open(tgz) as t:
        names = t.getnames()
        info = json.loads(t.extractfile(next(n for n in names if n.endswith("bundle_info.json"))).read())
    assert tgz.name.endswith("-list.tar.gz") and any(n.endswith("review_list.json") for n in names)
    assert info["n_reports"] == 4 and info["n_parse_files"] == 2 and info["review_list"] == lst.name
    assert not any("F002" in n for n in names) and "git_dirty" in info
