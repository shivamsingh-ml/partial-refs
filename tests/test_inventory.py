"""Inventory tables on a synthetic manifest (no real data)."""

import pandas as pd
import pytest

from partialref.audit.inventory import centre_summary, compare_geometry, report_x_within_case


@pytest.fixture
def manifest():
    cases = pd.DataFrame({
        "case_id": ["A001", "P002", "P003"], "centre": ["A", "P", "P"],
        "n_reports_en": [1, 2, 2], "axcodes": ["LPS", "LAS", "LAS"],
        "shape_x": [400, 370, 370], "shape_y": [400, 330, 330], "shape_z": [400, 170, 168],
        "spacing_x": [0.3] * 3, "spacing_y": [0.3] * 3, "spacing_z": [0.3] * 3,
        "extent_x_mm": [120.0, 111.0, 111.0], "extent_y_mm": [120.0, 99.0, 99.0],
        "extent_z_mm": [120.0, 51.0, 50.4],
    })
    reports = pd.DataFrame({
        "case_id": ["A001", "P002", "P002", "P003", "P003", "A001"],
        "lang": ["en", "en", "en", "en", "en", "it"],
        "report_x": [21, 14, 24, 25, 25, 21], "n_chars": [500, 300, 320, 280, 260, 520],
    })
    return cases, reports


def test_centre_summary(manifest):
    cs = centre_summary(*manifest).set_index("centre")
    assert cs.loc["P", "n_cases"] == 2 and cs.loc["P", "n_multi_report"] == 2
    assert cs.loc["A", "share_multi_report"] == 0.0
    assert cs.loc["P", "orientation"] == "LAS:2"
    assert cs.loc["P", "extent_z_mm"].startswith("50.7")


def test_report_x_within_case(manifest):
    rx = report_x_within_case(manifest[1])
    assert rx == {"n_multi_report_cases": 2, "all_distinct": 1, "all_same": 1, "mixed": 0,
                  "n_distinct_report_x": 4}


def test_compare_geometry(manifest):
    cases, _ = manifest
    ref = pd.DataFrame({"case_id": ["A001", "P002", "P999"], "size_x": [400, 370, 1],
                        "size_y": [400, 330, 1], "size_z": [400, 171, 1],
                        "spacing_x": [0.3, 0.3, 1], "spacing_y": [0.3, 0.3, 1],
                        "spacing_z": [0.3, 0.3, 1], "extent_x": [120.0, 111.0, 1],
                        "extent_y": [120.0, 99.0, 1], "extent_z": [120.0, 51.0, 1],
                        "orientation": ["LPS", "LAS", "LAS"]})
    c = compare_geometry(cases, ref).set_index("case_id")
    assert c.loc["P999", "in"] == "ref_only" and c.loc["P003", "in"] == "ours_only"
    assert c.loc["A001"].filter(like="_match").all()
    assert c.loc["P002", "shape_z_match"] == 0 and c.loc["P002", "shape_x_match"] == 1


def test_compare_geometry_reference_has_extra_cases(manifest):
    """Regression: ref-only rows turn our int shapes into floats after the outer merge."""
    cases, _ = manifest
    one = cases[cases["case_id"] == "A001"]
    ref = pd.DataFrame({"case_id": ["A001", "Z999"], "size_x": [400, 5], "size_y": [400, 5],
                        "size_z": [400, 5], "orientation": ["LPS", "LAS"]})
    c = compare_geometry(one, ref).set_index("case_id")
    assert c.loc["A001", ["shape_x_match", "shape_y_match", "shape_z_match",
                          "axcodes_match"]].all()
