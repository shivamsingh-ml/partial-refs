"""Manifest on a tiny synthetic dataset (fake volumes, placeholder report text)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from partialref.data.manifest import build_manifest, main, manifest_hash, summarize


def make_case(root: Path, case: str, report_names: list[str], spacing=(0.3, 0.3, 0.4)):
    (root / case / "cbct").mkdir(parents=True)
    affine = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((8, 6, 4), dtype=np.int16), affine),
             str(root / case / "cbct" / "volume.nii.gz"))
    for lang in ("en", "it"):
        d = root / case / f"reports_{lang}"
        d.mkdir()
        for name in report_names:
            (d / name).write_text(f"synthetic placeholder {lang} {name}\nline two\n")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    make_case(tmp_path, "A001", ["cbct_21_100.txt"])
    make_case(tmp_path, "P002", ["cbct_14_101.txt", "cbct_24_102.txt"])
    make_case(tmp_path, "S0003", ["cbct_28_103.txt", "cbct_29_104.txt", "cbct_26_105.txt"])
    return tmp_path


def test_cases_and_reports(dataset):
    cases, reports = build_manifest(dataset)
    assert list(cases["case_id"]) == ["A001", "P002", "S0003"]
    assert list(cases["centre"]) == ["A", "P", "S"]
    assert list(cases["n_reports_en"]) == [1, 2, 3]
    assert cases["en_it_names_match"].all()
    assert len(reports) == 2 * 6
    p2 = cases.set_index("case_id").loc["P002"]
    assert p2["report_x_en"] == [14, 24]
    assert (p2["shape_x"], p2["shape_y"], p2["shape_z"]) == (8, 6, 4)
    assert p2["spacing_z"] == pytest.approx(0.4)
    assert p2["extent_x_mm"] == pytest.approx(8 * 0.3)
    assert p2["axcodes"] == "RAS"
    assert reports["name_ok"].all() and reports["utf8_ok"].all()
    assert "text" not in reports.columns  # report text is never stored


def test_limit_and_case_ids(dataset):
    assert len(build_manifest(dataset, limit=2)[0]) == 2
    assert list(build_manifest(dataset, case_ids=["S0003"])[0]["case_id"]) == ["S0003"]


def test_hash_changes_with_content(dataset):
    c1, r1 = build_manifest(dataset)
    (dataset / "A001" / "reports_en" / "cbct_21_100.txt").write_text("changed")
    c2, r2 = build_manifest(dataset)
    assert manifest_hash(c1, r1) != manifest_hash(c2, r2)


def test_summary_counts_only(dataset):
    s = summarize(build_manifest(dataset)[0])
    assert s["n_cases"] == 3 and s["n_en_reports"] == 6 and s["n_multi_report_cases"] == 2
    assert s["en_reports_per_case_by_centre"]["S"] == {3: 1}


def test_cli_writes_outputs(dataset, tmp_path):
    res = tmp_path / "results"
    assert main(["--data-dir", str(dataset), "--results-dir", str(res), "--no-volume-hash"]) == 0
    (run,) = (res / "manifest").iterdir()
    for f in ("cases.parquet", "reports.parquet", "summary.json", "provenance.json"):
        assert (run / f).is_file()
    assert len(pd.read_parquet(run / "cases.parquet")) == 3
