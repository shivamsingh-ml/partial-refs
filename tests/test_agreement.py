"""GT/T2 agreement counting and propensity fitting (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from partialref.propensity.agreement import (build_instances, candidate_slots, fit_type, is_visible,
                                             load_visibility, slot_assessable)
from partialref.propensity.simulate import SimConfig, simulate


def vis_case(**kw):
    v = {"upper": True, "lower": True, "teeth": {"16", "36", "46"}, "root_cut": {"16"},
         "quadrant_edge": {"1": False, "2": False, "3": False, "4": False},
         "canal_Left": True, "canal_Right": False, "sinus_Left": False, "sinus_Right": True,
         "condyle_Left": False, "condyle_Right": True}
    v.update(kw)
    return v


def test_visibility_rules():
    v = vis_case()
    assert is_visible(v, "Caries", "36") and not is_visible(v, "Caries", "37")          # tooth must be present
    assert not is_visible(v, "PeriapicalLesion", "16") and is_visible(v, "PeriapicalLesion", "46")  # root tip
    assert is_visible(v, "ToothAbsence", "37")                                          # arch suffices
    assert not is_visible(vis_case(upper=False), "ToothAbsence", "17")
    assert is_visible(v, "DentalImplant", "26")
    assert is_visible(v, "MandibularCanalCourse", "Left") and not is_visible(v, "MandibularCanalCourse", "Right")
    assert is_visible(v, "MaxillarySinusFinding", "Right") and not is_visible(v, "MaxillarySinusFinding", "Left")
    assert is_visible(v, "MandibularCondyleFinding", "Right") and not is_visible(v, "MandibularCondyleFinding", "Left")
    assert not is_visible(vis_case(upper=False), "JawAtrophy", "QuadrantI")
    assert is_visible(vis_case(upper=False), "JawAtrophy", "QuadrantIV")
    assert len(candidate_slots("Caries")) == 32 and candidate_slots("JawAtrophy")[0] == "QuadrantI"


def test_build_instances_counts_and_drops():
    reports = pd.DataFrame({"case_id": ["P001", "P001", "P002", "P003", "P003"],
                            "report_id": ["a", "b", "c", "d", "e"]})
    mentions = pd.DataFrame({"case_id": ["P001", "P001", "P001", "P002", "P003"],
                             "report_id": ["a", "b", "a", "c", "d"],
                             "finding_type": ["Caries", "Caries", "Caries", "Caries", "Caries"],
                             "slot": ["36", "36", "37", "46", "36"]})   # P001: 36 by both, 37 invisible
    vis = {c: vis_case() for c in ["P001", "P002", "P003"]}
    inst, info = build_instances(reports, mentions, vis, failed_reports={("P003", "e")}, types=["Caries"])
    assert info["cases_used"] == 2 and info["cases_dropped_failed_parse"] == 1   # P003 dropped
    assert info["mentions_on_invisible_slots"] == {"Caries": 1}                  # tooth 37 not present
    r = inst.set_index(["case_id", "slot"])
    assert (r.loc[("P001", "36"), "n"], r.loc[("P001", "36"), "k"]) == (2, 2)
    assert (r.loc[("P001", "46"), "n"], r.loc[("P001", "46"), "k"]) == (2, 0)
    assert (r.loc[("P002", "46"), "n"], r.loc[("P002", "46"), "k"]) == (1, 1)
    assert set(inst["slot"]) == {"16", "36", "46"}                              # only present teeth


def test_reader_without_findings_counts():
    reports = pd.DataFrame({"case_id": ["P001", "P001"], "report_id": ["a", "b"]})
    mentions = pd.DataFrame({"case_id": ["P001"], "report_id": ["a"], "finding_type": ["Caries"], "slot": ["36"]})
    inst, _ = build_instances(reports, mentions, {"P001": vis_case()}, set(), ["Caries"])
    r = inst.set_index("slot")
    assert (r.loc["36", "n"], r.loc["36", "k"]) == (2, 1)                        # silent reader still counts


def test_load_visibility(tmp_path):
    d = tmp_path / "vis"
    d.mkdir()
    pd.DataFrame([{"case_id": "P001", "label_found": True, "maxilla_vox": 0, "mandible_vox": 100,
                   "canal_left_vox": 5, "canal_right_vox": 0, "sinus_left_vox": 0, "sinus_right_vox": np.nan,
                   "mandible_left_touches_sup": True, "mandible_right_touches_sup": False,
                   "mandible_left_touches_post": False, "mandible_right_touches_post": False,
                   "q1_teeth_touch_lateral_or_post": True, "q2_teeth_touch_lateral_or_post": False,
                   "q1_n_teeth": 1, "q2_n_teeth": 2},
                  {"case_id": "P002", "label_found": False}]).to_parquet(d / "case_features.parquet")
    pd.DataFrame([{"case_id": "P001", "fdi": 16, "present": True, "root_tip_cut": True},
                  {"case_id": "P001", "fdi": 36, "present": True, "root_tip_cut": False},
                  {"case_id": "P001", "fdi": 37, "present": False, "root_tip_cut": None}]).to_parquet(d / "tooth_features.parquet")
    v = load_visibility([d])
    assert set(v) == {"P001"}
    p = v["P001"]
    assert p["upper"] and p["lower"]                       # maxilla label absent but tooth 16 present
    assert p["teeth"] == {"16", "36"} and p["root_cut"] == {"16"}
    assert p["canal_Left"] and not p["canal_Right"] and not p["sinus_Right"]
    assert not p["condyle_Right"] and not p["condyle_Left"]     # one half cut at the top -> neither side
    assert p["quadrant_edge"]["1"] and not p["quadrant_edge"]["2"]
    assert p["quadrant_edge"]["3"]                              # missing flag -> treated as touching
    sw = load_visibility([d], swap_side_centres="P")["P001"]
    assert sw["canal_Right"] and not sw["canal_Left"] and sw["teeth"] == {"16", "36"}   # sides only, not teeth
    m = load_visibility([d], mirror_centres="PFS")["P001"]
    assert load_visibility([d], mirror_centres="A")["P001"]["teeth"] == {"16", "36"}   # other centre: as is
    assert m["teeth"] == {"26", "46"} and m["root_cut"] == {"26"} and not m["quadrant_edge"]["1"]


@pytest.mark.parametrize("pi_true", [0.6, 0.85])
def test_fit_type_recovers_pi(pi_true):
    cfg = SimConfig(n_findings=1, n_centres=1, centre_logit_sd=0.0, n_cases=3000,
                    prevalence_range=(0.3, 0.3), propensity_range=(pi_true, pi_true), seed=5)
    s = simulate(cfg).instances
    inst = pd.DataFrame({"case_id": s.case_id, "centre": "P", "finding_type": "Caries", "slot": "36",
                         "n": s.n_reports, "k": s.k})
    r = fit_type(inst, n_boot=200, seed=1)
    assert r["pi_lo"] <= pi_true <= r["pi_hi"]
    assert abs(r["pi_mle"] - pi_true) < 0.05 and abs(r["pi_closed_form_n2"] - pi_true) < 0.06
    assert r["boot_ok"] == 200 and r["pi_ci_width"] < 0.2


def test_slot_assessable_missing_tooth_rule():
    v = vis_case(teeth={"31", "34", "36"}, quadrant_edge={"1": True, "2": False, "3": True, "4": True})
    assert slot_assessable(v, "33")            # between present 31 and 36
    assert not slot_assessable(v, "38")        # beyond last present tooth, quadrant touches the edge
    assert slot_assessable(v, "28")            # quadrant 2 does not touch the edge
    assert not slot_assessable(v, "17")        # no present teeth in q1 and q1 touches the edge
    assert not is_visible(v, "ToothAbsence", "38") and is_visible(v, "ToothAbsence", "33")


def test_condyle_needs_both_halves_inside(tmp_path):
    d = tmp_path / "v"
    d.mkdir()
    base = {"label_found": True, "mandible_vox": 10, "mandible_left_touches_sup": False,
            "mandible_right_touches_sup": False}
    pd.DataFrame([{**base, "case_id": "A1", "mandible_left_touches_post": False, "mandible_right_touches_post": False},
                  {**base, "case_id": "A2", "mandible_left_touches_post": True, "mandible_right_touches_post": False}]
                 ).to_parquet(d / "case_features.parquet")
    pd.DataFrame(columns=["case_id", "fdi", "present", "root_tip_cut"]).to_parquet(d / "tooth_features.parquet")
    v = load_visibility([d])
    assert v["A1"]["condyle_Left"] and v["A1"]["condyle_Right"]
    assert not v["A2"]["condyle_Left"] and not v["A2"]["condyle_Right"]


def test_failed_reports_detected_after_parquet_roundtrip(tmp_path):
    """Regression (review A1): pandas >= 3 reads missing strings back as NaN, not None."""
    meta = pd.DataFrame({"case_id": ["P001", "P002", "P003"], "report_id": ["a", "b", "c"],
                         "error": [None, None, "validation: 3 errors"]})
    meta.to_parquet(tmp_path / "m.parquet")
    back = pd.read_parquet(tmp_path / "m.parquet")
    failed = set(map(tuple, back.loc[back["error"].notna(), ["case_id", "report_id"]].to_numpy()))
    assert failed == {("P003", "c")}


def test_single_report_only_is_not_estimable():
    """Regression (review A2): no instance with n >= 2 and k >= 1 -> pi NaN, never a passing GT flag."""
    cfg = SimConfig(n_findings=1, n_centres=1, centre_logit_sd=0.0, n_cases=2000, report_count_probs=(1.0,),
                    prevalence_range=(0.3, 0.3), propensity_range=(0.8, 0.8), seed=3)
    s = simulate(cfg).instances
    inst = pd.DataFrame({"case_id": s.case_id, "centre": "F", "finding_type": "Caries", "slot": "36",
                         "n": s.n_reports, "k": s.k})
    r = fit_type(inst, n_boot=50, seed=0)
    assert np.isnan(r["pi_mle"]) and not r["supported"] and r["boot_non_identified_share"] == 1.0


def test_low_double_read_support_not_supported():
    cfg = SimConfig(n_findings=1, n_centres=1, centre_logit_sd=0.0, n_cases=60, report_count_probs=(0.0, 1.0),
                    prevalence_range=(0.1, 0.1), propensity_range=(0.8, 0.8), seed=4)
    s = simulate(cfg).instances
    inst = pd.DataFrame({"case_id": s.case_id, "centre": "P", "finding_type": "Caries", "slot": "36",
                         "n": s.n_reports, "k": s.k})
    r = fit_type(inst, n_boot=50, seed=0)
    assert not r["supported"]                   # far fewer than 30 double-read mentions


def test_empty_quadrant_needs_whole_jaw_inside(tmp_path):
    """Review #4: a quadrant without mask teeth is not assessable just because no tooth touches an edge."""
    d = tmp_path / "v"
    d.mkdir()
    base = {"label_found": True, "mandible_vox": 100, "maxilla_vox": 0, "q3_n_teeth": 7, "q4_n_teeth": 0,
            "q3_teeth_touch_lateral_or_post": False, "q4_teeth_touch_lateral_or_post": False}
    inside = {f"mandible_{h}_touches_{f}": False for h in ("left", "right") for f in ("lateral", "post")}
    pd.DataFrame([{**base, "case_id": "P001"},                                   # jaw-half flags missing
                  {**base, "case_id": "P002", **inside, "mandible_right_touches_lateral": True},
                  {**base, "case_id": "P003", **inside}]).to_parquet(d / "case_features.parquet")
    pd.DataFrame([{"case_id": c, "fdi": 30 + i, "present": True, "root_tip_cut": False}
                  for c in ("P001", "P002", "P003") for i in range(1, 8)]).to_parquet(d / "tooth_features.parquet")
    v = load_visibility([d], mirror_centres="P")          # label q3 teeth -> reports' 41-47; empty label q4 -> 3x
    assert v["P001"]["quadrant_edge"]["3"] and v["P002"]["quadrant_edge"]["3"]
    assert not is_visible(v["P001"], "ToothAbsence", "36") and not is_visible(v["P002"], "ToothAbsence", "36")
    assert not v["P003"]["quadrant_edge"]["3"] and is_visible(v["P003"], "ToothAbsence", "36")
    assert not v["P001"]["quadrant_edge"]["4"]            # quadrant with teeth: unchanged rule


def test_region_absence_makes_reader_uninformative_not_omitting():
    """Parser v3: reader a says 'partial edentulism of the upper arch' (no numbers), reader b names 16.
    For tooth 16, a must not count as an omission: n drops to 1 (only b), k = 1. For tooth 17 neither
    named it: n = 1 (a uninformative), k = 0. Lower teeth: both readers informative (n = 2)."""
    reports = pd.DataFrame({"case_id": ["P001", "P001"], "report_id": ["a", "b"]})
    mentions = pd.DataFrame({"case_id": ["P001", "P001"], "report_id": ["a", "b"],
                             "finding_type": ["ToothAbsence", "ToothAbsence"], "slot": ["region:UpperArch", "16"]})
    v = vis_case(teeth={"11", "36", "46"})
    inst, info = build_instances(reports, mentions, {"P001": v}, set(), ["ToothAbsence"], region_mode="uninformative")
    r = inst.set_index("slot")
    assert (r.loc["16", "n"], r.loc["16", "k"]) == (1, 1)
    assert (r.loc["17", "n"], r.loc["17", "k"]) == (1, 0)
    assert r.loc["37", "n"] == 2
    assert info["region_mentions"] == 1 and info["reader_slot_pairs_uninformative_by_region"] == 16


def test_region_reader_stays_informative_for_teeth_it_calls_present_and_omission_mode():
    reports = pd.DataFrame({"case_id": ["P001", "P001"], "report_id": ["a", "b"]})
    mentions = pd.DataFrame({"case_id": ["P001"] * 3, "report_id": ["a", "a", "b"],
                             "finding_type": ["ToothAbsence", "Caries", "ToothAbsence"],
                             "slot": ["region:UpperArch", "26", "16"]})
    v = vis_case(teeth={"11", "26", "36"})
    inst, info = build_instances(reports, mentions, {"P001": v}, set(), ["ToothAbsence"], region_mode="uninformative")
    r = inst.set_index("slot")
    assert r.loc["26", "n"] == 2                        # a named 26 with caries -> informative (not missing)
    assert r.loc["17", "n"] == 1 and info["reader_slot_pairs_uninformative_by_region"] == 15
    inst2, info2 = build_instances(reports, mentions, {"P001": v}, set(), ["ToothAbsence"])   # default = main (B)
    r2 = inst2.set_index("slot")
    assert (r2.loc["16", "n"], r2.loc["16", "k"]) == (2, 1) and info2["reader_slot_pairs_uninformative_by_region"] == 0
