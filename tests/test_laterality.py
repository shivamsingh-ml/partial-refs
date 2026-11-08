"""Laterality diagnostics (synthetic data only)."""

import pandas as pd

from partialref.propensity.laterality import per_case_check, residual_check, swap_check


def vis(teeth):
    return {"upper": True, "lower": True, "teeth": set(teeth), "root_cut": set(),
            "quadrant_edge": dict.fromkeys("1234", False), "canal_Left": True, "canal_Right": True,
            "sinus_Left": True, "sinus_Right": True, "condyle_Left": False, "condyle_Right": False}


def mentions(rows):
    return pd.DataFrame(rows, columns=["case_id", "report_id", "finding_type", "slot"])


def test_per_case_verdicts():
    # mask labels say 16/36 present; reports say 26/46 -> mask is mirrored for P001
    v = {"P001": vis({"16", "36"}), "P002": vis({"16", "26"}), "P003": vis({"16"})}
    vm = {"P001": vis({"26", "46"}), "P002": vis({"26", "16"}), "P003": vis({"26"})}
    m = mentions([("P001", "a", "Caries", "26"), ("P001", "a", "EndodonticTreatment", "46"),
                  ("P002", "b", "Caries", "16"),                       # both 16/26 present -> tie at zero
                  ("P003", "c", "Caries", "16"), ("P003", "c", "Caries", "36")])  # normal fits better
    pc = per_case_check(m, {"P001": 1, "P002": 1, "P003": 1}, v, vm).set_index("case_id")
    assert pc.loc["P001", "verdict"] == "mirrored_better" and pc.loc["P001", "invisible_normal"] == 2
    assert pc.loc["P002", "verdict"] == "tie_zero"
    assert pc.loc["P003", "verdict"] == "normal_better"
    assert (pc.loc["P003", "invisible_normal"], pc.loc["P003", "invisible_mirrored"]) == (1, 2)


def test_residual_categories():
    vm = {"P001": vis({"16"}), "P002": vis({"16"})}
    m = mentions([("P001", "a", "Caries", "36"), ("P001", "b", "Caries", "36"),   # both readers: mask suspect
                  ("P001", "a", "Caries", "46"),                                  # one of two readers
                  ("P002", "c", "Caries", "37"),                                  # single-read case
                  ("P001", "a", "ToothAbsence", "36")])                           # not a tooth-presence type
    r = residual_check(m, {"P001": 2, "P002": 1}, vm).set_index(["case_id", "slot"])["category"]
    assert r[("P001", "36")] == "multiple_readers" and r[("P001", "46")] == "one_of_several_readers"
    assert r[("P002", "37")] == "single_read_case" and len(r) == 3


def test_swap_counts_and_controls():
    m = mentions([("P001", "a", "Caries", "36"), ("P001", "b", "Caries", "46"),    # left/right swap
                  ("P001", "a", "DentalImplant", "15"), ("P001", "b", "DentalImplant", "45"),  # up/down
                  ("P001", "a", "ConservativeRestoration", "24"),
                  ("P001", "b", "ConservativeRestoration", "25"),                # neighbour shift
                  ("P001", "a", "EndodonticTreatment", "11"), ("P001", "b", "EndodonticTreatment", "21"),  # midline
                  ("P001", "a", "ToothAbsence", "18"), ("P001", "b", "ToothAbsence", "18"),  # agreement
                  ("P002", "c", "Caries", "36")])                                # single-read case: no pairs
    sw = swap_check(m, {"P001": ["a", "b"], "P002": ["c"]}).set_index("finding_type")
    assert sw.loc["Caries", "left_right_swaps"] == 1 and sw.loc["Caries", "pairs"] == 1
    assert sw.loc["DentalImplant", "up_down_swaps"] == 1 and sw.loc["DentalImplant", "left_right_swaps"] == 0
    assert sw.loc["ConservativeRestoration", "neighbour_shifts"] == 1
    assert sw.loc["EndodonticTreatment", "left_right_swaps"] == 0      # 11 vs 21 excluded (neighbours)
    assert sw.loc["ToothAbsence", "both"] == 1 and sw.loc["ToothAbsence", "unmatched"] == 0


def test_silent_reader_counts_as_pair():
    m = mentions([("P001", "a", "Caries", "36")])
    sw = swap_check(m, {"P001": ["a", "b"]}).set_index("finding_type")
    assert sw.loc["Caries", "pairs"] == 1 and sw.loc["Caries", "unmatched"] == 1
