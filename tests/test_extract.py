"""Ontology, synthetic few-shot examples, Lumetti-label reader and parser scoring (synthetic only)."""

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from partialref.extract.fewshot import FEWSHOT, SYSTEM_PROMPT, messages
from partialref.extract.lumetti_labels import case_id_from_file, parse_file
from partialref.extract.ontology import Finding, ReportExtraction, instances, json_schema
from partialref.extract.score_parser import score, scoring_universe


def test_fewshot_outputs_are_valid():
    for _, out in FEWSHOT:
        ReportExtraction.model_validate(out)
    m = messages("synthetic text")
    assert m[0]["content"] == SYSTEM_PROMPT and m[-1]["content"].endswith("synthetic text")
    assert len(m) == 2 + 2 * len(FEWSHOT)


def test_finding_validation():
    with pytest.raises(ValidationError):
        Finding(finding_type="Caries")                       # needs tooth ids
    with pytest.raises(ValidationError):
        Finding(finding_type="Caries", tooth_ids=["19"])     # not FDI
    with pytest.raises(ValidationError):
        Finding(finding_type="MandibularCanalCourse")        # needs side
    with pytest.raises(ValidationError):
        Finding(finding_type="Caries", tooth_ids=["36"], qualifiers={"severity": ["Mild", "Severe"]})
    f = Finding(finding_type="ProstheticDevice", tooth_ids=["46", "44", "45"],
                qualifiers={"prosthesis_type": ["FixedBridge"]})
    assert f.slots() == ["44", "45", "46"]
    assert Finding(finding_type="ToothDevelopmentPathology", tooth_ids=["63"]).slots() == ["63"]


def test_instances_dedup_and_schema():
    fs = [Finding(finding_type="Caries", tooth_ids=["36"]), Finding(finding_type="Caries", tooth_ids=["36", "37"]),
          Finding(finding_type="JawAtrophy", quadrant="QuadrantIII")]
    assert instances("P001", fs) == [("P001", "Caries", "36"), ("P001", "Caries", "37"),
                                     ("P001", "JawAtrophy", "QuadrantIII")]
    assert "findings" in json.dumps(json_schema())


TTL = """@prefix :     <http://example.org/dentro#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
:Patient_X__Finding_1 a :Finding ; :findingType :ProstheticDevice ;
    :hasToothId "44", "45", "46" ; :hasProsthesisType :FixedBridge ; rdfs:comment "synthetic" .
:Patient_X__Finding_2 a :Finding ; :findingType :MandibularCanalCourse ; :onSide :Left ;
    :hasCanalTrajectory :PrevalentlyLingual ; rdfs:comment "synthetic" .
:Patient_X__Finding_3 a :Finding ; :findingType :MaxillarySinusFinding ; :hasQuadrant :QuadrantII .
"""


def test_lumetti_reader(tmp_path):
    p = tmp_path / "ToothFairy3S_0007.ttl"
    p.write_text(TTL)
    assert case_id_from_file(p.name) == "S0007"
    rows = {r["finding_type"]: r for r in parse_file(p)}
    assert rows["ProstheticDevice"]["slots"] == ["44", "45", "46"]
    assert rows["MandibularCanalCourse"]["slots"] == ["Left"]
    assert rows["MaxillarySinusFinding"]["slots"] == []      # side type without onSide: no quadrant as slot
    assert "synthetic" not in json.dumps(rows)              # comments never read into output
    assert json.loads(rows["ProstheticDevice"]["qualifiers"]) == {"prosthesis_type": ["FixedBridge"]}


def test_score_parser():
    ref = pd.DataFrame([("P001", "Caries", "36"), ("P001", "Caries", "37"), ("P002", "ToothAbsence", "18")],
                       columns=["case_id", "finding_type", "slot"])
    ours = pd.DataFrame([("P001", "Caries", "36"), ("P001", "Caries", "46"), ("P002", "ToothAbsence", "18"),
                         ("P999", "Caries", "11")], columns=["case_id", "finding_type", "slot"])
    s = score(ours, ref, "slot", {"P001", "P002"}).set_index("finding_type")
    assert (s.loc["Caries", "tp"], s.loc["Caries", "fp"], s.loc["Caries", "fn"]) == (1, 1, 1)
    assert s.loc["ToothAbsence", "f1"] == 1.0 and s["n_cases"].iloc[0] == 2   # P999 outside universe
    t = score(ours, ref, "type", {"P001", "P002"}).set_index("finding_type")
    assert t.loc["Caries", "f1"] == 1.0


def _run(tmp_path, name, meta_rows, inst_rows):
    d = tmp_path / name
    d.mkdir()
    pd.DataFrame(meta_rows, columns=["case_id", "report_id", "error"]).to_parquet(d / "extractions_en.parquet")
    pd.DataFrame(inst_rows, columns=["case_id", "report_id", "finding_type", "slot"]).to_parquet(d / "instances_en.parquet")
    return d / "instances_en.parquet"


def test_scoring_universe_counts_empty_and_drops_failed(tmp_path):
    """Review #2: a fully parsed case with no findings still counts (its Lumetti findings are misses);
    a case with one failed report is dropped instead of being scored on a partial union."""
    ours = _run(tmp_path, "ours",
                [("P001", "a", None), ("P002", "b", None), ("P003", "c", None), ("P003", "d", "validation: 1 errors")],
                [("P001", "a", "Caries", "36"), ("P003", "c", "Caries", "11")])
    lum = tmp_path / "lum"
    lum.mkdir()
    pd.DataFrame({"case_id": ["P001", "P002", "P003"]}).to_parquet(lum / "cases.parquet")
    ref = pd.DataFrame([("P001", "Caries", "36"), ("P002", "Caries", "14"), ("P002", "Caries", "15"),
                        ("P003", "Caries", "11")], columns=["case_id", "finding_type", "slot"])
    ref.to_parquet(lum / "instances.parquet")
    cases, _, info = scoring_universe(ours, lumetti=lum / "instances.parquet")
    assert cases == {"P001", "P002"} and info["our_cases_with_failed_report"] == 1
    s = score(pd.read_parquet(ours), ref, "slot", cases).set_index("finding_type")
    assert (s.loc["Caries", "tp"], s.loc["Caries", "fn"]) == (1, 2)          # P002's two misses count


def test_per_report_level_sees_within_case_changes(tmp_path):
    """Review #3: run 2 loses report b's finding while report a still has it -> union unchanged,
    per-report level shows the miss."""
    meta = [("P001", "a", None), ("P001", "b", None)]
    r1 = _run(tmp_path, "r1", meta, [("P001", "a", "Caries", "36"), ("P001", "b", "Caries", "36")])
    r2 = _run(tmp_path, "r2", meta, [("P001", "a", "Caries", "36")])
    cases, reports, _ = scoring_universe(r2, ref_parse=r1)
    o, r = pd.read_parquet(r2), pd.read_parquet(r1)
    assert score(o, r, "slot", cases).set_index("finding_type").loc["Caries", "f1"] == 1.0
    rep = score(o, r, "report_slot", reports).set_index("finding_type")
    assert (rep.loc["Caries", "tp"], rep.loc["Caries", "fn"]) == (1, 1) and rep["n_reports"].iloc[0] == 2


def test_region_level_absence_and_conflicts():
    from partialref.extract.ontology import absence_conflicts
    r = Finding(finding_type="ToothAbsence", region="UpperArch")
    assert r.slots() == ["region:UpperArch"]
    assert Finding(finding_type="Caries", tooth_ids=["36"], region="LowerArch").region is None  # ignored off-type
    with pytest.raises(ValueError):
        Finding(finding_type="Caries")                                   # tooth type still needs teeth
    both = Finding(finding_type="ToothAbsence", tooth_ids=["17"], region="QuadrantI")
    assert both.slots() == ["17"]                                        # explicit teeth win over region
    f = [Finding(finding_type="ToothAbsence", tooth_ids=["36", "37"]),
         Finding(finding_type="ToothDevelopmentPathology", tooth_ids=["37"]),
         Finding(finding_type="ProstheticDevice", tooth_ids=["36"])]     # pontic: not a conflict
    assert absence_conflicts(f) == ["37"]


def test_region_scope_expanded_by_code():
    """v3.2: a toothless arch / quadrant is stated as region + AllMissing; the tooth list is filled in by code."""
    from partialref.extract.ontology import REGION_TEETH, absence_conflict_details
    full = Finding(finding_type="ToothAbsence", region="LowerArch", region_scope="AllMissing")
    assert full.tooth_ids == sorted(REGION_TEETH["LowerArch"]) and len(full.slots()) == 16
    assert (full.region, full.region_scope) == ("LowerArch", "AllMissing")          # statement kept as given
    exc = Finding(finding_type="ToothAbsence", region="LowerArch", region_scope="AllMissing",
                  remaining_tooth_ids=["43", "33", "11"])                           # 11 is outside: no effect
    assert len(exc.tooth_ids) == 14 and not {"33", "43"} & set(exc.tooth_ids)
    assert Finding(finding_type="ToothAbsence", region="QuadrantII", region_scope="AllMissing").tooth_ids == \
        REGION_TEETH["QuadrantII"]
    again = Finding.model_validate(exc.model_dump(mode="json"))                     # stored result re-read
    assert again.tooth_ids == exc.tooth_ids and again.remaining_tooth_ids == ["11", "33", "43"]
    some = Finding(finding_type="ToothAbsence", region="UpperArch", region_scope="SomeMissing",
                   remaining_tooth_ids=["13"])
    assert some.slots() == ["region:UpperArch"] and some.tooth_ids == []            # never expanded
    assert Finding(finding_type="ToothAbsence", region="UpperArch").slots() == ["region:UpperArch"]   # v3 files
    with pytest.raises(ValidationError):                                           # contradictory statement
        Finding(finding_type="ToothAbsence", region="QuadrantI", region_scope="AllMissing",
                remaining_tooth_ids=REGION_TEETH["QuadrantI"])
    off = Finding(finding_type="Caries", tooth_ids=["36"], region="LowerArch", region_scope="AllMissing",
                  remaining_tooth_ids=["33"])
    assert (off.region, off.region_scope, off.remaining_tooth_ids, off.tooth_ids) == (None, None, [], ["36"])
    lone = Finding(finding_type="ToothAbsence", tooth_ids=["17"], region_scope="AllMissing")   # scope without region
    assert lone.region_scope is None and lone.tooth_ids == ["17"]
    # a residual root the model did not list as remaining is recorded as a conflict, as before
    conf = absence_conflict_details([full, Finding(finding_type="RootResidue", tooth_ids=["31"])])
    assert conf == [{"tooth": "31", "with": "RootResidue"}]
    props = json_schema()["$defs"]["Finding"]["properties"]
    assert "region_scope" in props and "remaining_tooth_ids" in props
    # tooth ids the model adds to an AllMissing region are kept (union), and instances() gives teeth, no region
    both = Finding(finding_type="ToothAbsence", region="QuadrantI", region_scope="AllMissing", tooth_ids=["21"])
    assert both.tooth_ids == sorted(REGION_TEETH["QuadrantI"] + ["21"])
    assert {s for _, _, s in instances("P001", [exc])} == set(exc.tooth_ids)
    # optional convention: a removed tooth stays removed when the stored finding is validated again
    from partialref.extract.ontology import apply_absence_convention
    fixed, n = apply_absence_convention([full, Finding(finding_type="RootResidue", tooth_ids=["31"])])
    assert n == 1 and "31" not in fixed[0].tooth_ids and fixed[0].remaining_tooth_ids == ["31"]
    back = Finding.model_validate(fixed[0].model_dump(mode="json"))
    assert back.tooth_ids == fixed[0].tooth_ids and len(back.tooth_ids) == 15


def test_prompt_v3_rules_present():
    flat = " ".join(SYSTEM_PROMPT.split())                            # independent of line wraps
    for phrase in ('"e.d." NEVER means edentulous', "ARE PRESENT", "through it", "neither is ToothAbsence",
                   'region_scope "AllMissing"', 'region_scope "SomeMissing"', "remaining_tooth_ids",
                   "never list its teeth yourself", '"distal to 45" = 46, 47, 48', "molar sector = q6, q7, q8",
                   "a WHOLE tooth group with its side stated", "only PARTLY edentulous", "has no side stated",
                   "is NOT missing and also goes in remaining_tooth_ids", "implant positions and bridge pontics do not",
                   "describes the SCAN, not the teeth", "coded ONCE", "in the arch", "Never guess numbers"):
        assert phrase in flat, phrase
    outs = [ReportExtraction.model_validate(o) for _, o in FEWSHOT]
    regions = [f for e in outs for f in e.findings if f.region]
    assert all(f.region_scope for f in regions)                       # every example region states its scope
    assert sorted(f.region_scope for f in regions) == ["AllMissing"] * 4 + ["SomeMissing"] * 2
    # no example lets the MODEL write out a whole arch (v3.2: code does it), and the expansion is right
    assert max(len(f.get("tooth_ids", [])) for _, o in FEWSHOT for f in o["findings"]) <= 6
    assert sorted(len(f.tooth_ids) for f in regions if f.region_scope == "AllMissing") == [14, 15, 16, 16]
    # a residual root inside a toothless arch is a remaining tooth; implants there are not: no example conflicts
    from partialref.extract.ontology import absence_conflict_details
    assert any(f.remaining_tooth_ids == ["17"] for f in regions)
    assert all(absence_conflict_details(e.findings) == [] for e in outs)
    # scan-coverage wording next to a dentate arch: that example has no upper-arch region
    cov = [e for (t, _), e in zip(FEWSHOT, outs) if "partially included in the scan" in t]
    assert len(cov) == 1 and not any(f.region == "UpperArch" for f in cov[0].findings)


def test_score_excludes_region_slots_vs_lumetti(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    pd.DataFrame([("P001", "a", None)], columns=["case_id", "report_id", "error"]).to_parquet(d / "extractions_en.parquet")
    pd.DataFrame([("P001", "a", "en", "ToothAbsence", "region:UpperArch"), ("P001", "a", "en", "Caries", "36")],
                 columns=["case_id", "report_id", "lang", "finding_type", "slot"]).to_parquet(d / "instances_en.parquet")
    lum = tmp_path / "lum"
    lum.mkdir()
    pd.DataFrame([("P001", "Caries", "36"), ("P001", "ToothAbsence", "17")],
                 columns=["case_id", "finding_type", "slot"]).to_parquet(lum / "instances.parquet")
    pd.DataFrame({"case_id": ["P001"]}).to_parquet(lum / "cases.parquet")
    from partialref.extract.score_parser import main as score_main
    assert score_main(["--ours", str(d / "instances_en.parquet"), "--lumetti", str(lum / "instances.parquet"),
                       "--results-dir", str(tmp_path / "res")]) == 0
    out = next((tmp_path / "res" / "extract").iterdir())
    res = pd.read_csv(out / "parser_vs_lumetti.csv").set_index(["level", "finding_type"])
    assert res.loc[("type", "ToothAbsence"), "tp"] == 1                     # region counts at type level
    assert (res.loc[("slot", "ToothAbsence"), "fp"], res.loc[("slot", "ToothAbsence"), "fn"]) == (0, 1)
