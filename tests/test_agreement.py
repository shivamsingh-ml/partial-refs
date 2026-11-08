"""GT/T2 agreement counting and propensity fitting (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from partialref.propensity.agreement import (build_instances, candidate_slots, fit_type, is_visible,
                                             load_visibility)
from partialref.propensity.simulate import SimConfig, simulate


def vis_case(**kw):
    v = {"upper": True, "lower": True, "teeth": {"16", "36", "46"}, "root_cut": {"16"},
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
                   "mandible_left_touches_sup": True, "mandible_right_touches_sup": False},
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
    assert p["condyle_Right"] and not p["condyle_Left"]


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
