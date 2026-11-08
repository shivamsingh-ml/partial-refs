"""Side geometry, report-scan fingerprint matching and the audit sheet (synthetic data only)."""

import numpy as np
import pandas as pd

from partialref.audit.report_scan_match import match_reports
from partialref.audit.side_geometry import case_side_checks
from partialref.extract.audit_sheet import candidate_items, sample_items


def teeth_df(case, cx_by_fdi):
    return pd.DataFrame([{"case_id": case, "fdi": f, "present": True, "cx": x} for f, x in cx_by_fdi.items()])


def test_side_geometry_match_and_mismatch():
    # TF3 q4 teeth (reports' left lower) at x = +30, q3 at -30; q1 (reports' left upper) at +30, q2 at -30
    t = teeth_df("P001", {46: 30.0, 36: -30.0, 16: 30.0, 26: -30.0})
    same = {"case_id": "P001", "canal_left_cx": 25.0, "canal_right_cx": -25.0,
            "sinus_left_cx": -20.0, "sinus_right_cx": 20.0, "mandible_cx": 0.0, "maxilla_cx": 0.0}
    r = case_side_checks(same, t)
    assert r["canal"] == "match" and r["sinus"] == "mismatch"
    assert r["header_lr_agrees_with_reports"] is False        # reports' left teeth at +x = RAS right
    # one canal only -> compared with the mandible centroid
    r1 = case_side_checks({"case_id": "P001", "canal_left_cx": 10.0, "mandible_cx": 0.0}, t)
    assert r1["canal"] == "match" and r1["sinus"] == "undetermined"
    # report-convention mask (centre A): standard FDI, reports' left = q3/q2 -> verdicts flip
    ra = case_side_checks(same, t, mirrored=False)
    assert ra["canal"] == "mismatch" and ra["sinus"] == "match" and ra["header_lr_agrees_with_reports"] is True


def vis(teeth, edge=False):
    return {"upper": True, "lower": True, "teeth": set(teeth), "root_cut": set(),
            "quadrant_edge": dict.fromkeys("1234", edge), "canal_Left": True, "canal_Right": True,
            "sinus_Left": True, "sinus_Right": True, "condyle_Left": False, "condyle_Right": False}


ALL = [f"{q}{i}" for q in (1, 2, 3, 4) for i in range(1, 9)]


def test_report_scan_match_flags():
    # scan X: 16, 26, 36 missing; scan Y: 46, 47, 45 missing; scan Z: mirror of X (26/16 swap is symmetric,
    # so use an asymmetric pattern: 17, 36, 37 missing)
    X = vis(set(ALL) - {"17", "36", "37"})
    Y = vis(set(ALL) - {"44", "45", "46", "47"})
    Xm = vis(set(ALL) - {"27", "46", "47"})            # X seen with the opposite mirroring
    v = {"P001": X, "P002": Y, "P003": vis(set(ALL) - {"27", "46", "47"})}
    v_opp = {"P001": Xm, "P002": vis(set(ALL) - {"34", "35", "36", "37"}), "P003": X}
    rows = []
    for c, r, missing, present in [("P001", "a", ["17", "36", "37"], ["16", "26", "46"]),   # fits own scan
                                   ("P002", "b", ["17", "36", "37"], ["16", "26", "46"]),   # belongs to P001
                                   ("P003", "c", ["17", "36", "37"], ["16", "26", "46"])]:  # P003 is flipped
        rows += [(c, r, "ToothAbsence", s) for s in missing] + [(c, r, "Caries", s) for s in present]
    m = pd.DataFrame(rows, columns=["case_id", "report_id", "finding_type", "slot"])
    res = match_reports(m, [("P001", "a"), ("P002", "b"), ("P003", "c")], v, v_opp,
                        min_evidence=4, min_contra=2).set_index("case_id")
    a, b, c = res.loc["P001"], res.loc["P002"], res.loc["P003"]
    assert a["own_contra"] == 0 and a["own_rank"] == 1 and not a["flag_misfiled"] and not a["flag_flipped"]
    assert b["flag_misfiled"] and b["best_other_case"] == "P001" and b["own_rank"] > 1
    assert c["flag_flipped"] and c["opp_contra"] == 0
    assert a["own_agree"] == 6 and a["own_evidence"] == 6


def test_report_scan_match_no_evidence():
    v = {"P001": vis(ALL)}
    m = pd.DataFrame(columns=["case_id", "report_id", "finding_type", "slot"])
    r = match_reports(m, [("P001", "a")], v, v).iloc[0]
    assert r["own_evidence"] == 0 and np.isnan(r["own_rate"]) and r["own_rank"] is None
    assert not r["flag_misfiled"] and not r["flag_flipped"]


def test_audit_sheet_reasons_blinding_and_pairs():
    m = pd.DataFrame([("P001", "a", "Caries", "36"), ("P001", "b", "Caries", "46"),          # L/R swap
                      ("P001", "a", "EndodonticTreatment", "17"),                            # 17 absent in mask
                      ("P001", "a", "DentalImplant", "24")],
                     columns=["case_id", "report_id", "finding_type", "slot"])
    other = m.iloc[:2].copy()                                                                # misses 17, 24
    lum = pd.DataFrame({"case_id": ["P001"], "finding_type": ["Caries"], "slot": ["36"]})
    v = {"P001": vis(set(ALL) - {"17"})}
    cand = candidate_items(m, {"P001": ["a", "b"]}, v, other, lum)
    reasons = cand.groupby("reason").size().to_dict()
    assert reasons["left_right_swap"] == 2 and reasons["residual_one_reader"] == 1
    assert reasons["this_parser_only"] == 2 and reasons["not_in_lumetti"] == 3
    s = sample_items(cand, per_reason=10, seed=0)
    assert s.duplicated(["case_id", "report_id", "finding_type", "slot"]).sum() == 0
    assert set(s["item_id"]) == {f"item{i:03d}" for i in range(len(s))}
    sw = s[s["reason"].str.contains("left_right_swap")]
    assert len(sw) in (0, 2) and sw["pair_id"].nunique() <= 1
    both = s[(s["case_id"] == "P001") & (s["report_id"] == "b") & (s["slot"] == "46")]
    assert both.empty or both["reason"].iloc[0] == "left_right_swap;not_in_lumetti"   # all reasons kept
