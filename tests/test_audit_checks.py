"""TF3 link and duplicate checks on synthetic data only."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from partialref.audit.duplicates import score_pairs
from partialref.audit.tf3_link import compare_case


def _save(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


@pytest.fixture
def vol():
    rng = np.random.default_rng(0)
    return rng.normal(size=(6, 5, 4)).astype(np.float32)


def test_tf3_identical_after_reorientation(tmp_path, vol):
    las = np.diag([-0.3, 0.3, 0.3, 1.0])
    ras = np.diag([0.3, 0.3, 0.3, 1.0])
    _save(tmp_path / "tf4" / "P001.nii.gz", vol, las)
    _save(tmp_path / "tf3" / "imagesTr" / "ToothFairy3P_001_0000.nii.gz", vol[::-1].copy(), ras)
    _save(tmp_path / "tf3" / "labelsTr" / "ToothFairy3P_001.nii.gz", np.zeros(vol.shape, np.uint8), ras)
    r = compare_case("P001", str(tmp_path / "tf4" / "P001.nii.gz"), str(tmp_path / "tf3"))
    assert r["tf4_axcodes"] == "LAS" and r["tf3_axcodes"] == "RAS"
    assert not r["same_file"]
    assert r["same_shape_canonical"] and r["frac_voxels_equal"] == 1.0
    assert r["label_matches_tf3_grid"]
    assert r["voxel_identical_flip"] == "100"


def test_tf3_flip_and_anatomy(tmp_path):
    """TF3 array = TF4 array flipped on axes 1,2 with a header that does not record the z flip."""
    L4 = np.zeros((6, 6, 8), np.uint8)
    L4[1:3, 2:4, 5:7] = 2   # upper jaw high z in TF4 voxel space
    L4[1:3, 2:4, 1:3] = 1   # lower jaw low z
    L4[0, 1, 2] = 3; L4[5, 1, 2] = 4          # left canal at low x index, right at high
    v4 = L4.astype(np.float32) * 10 + 1
    aff = np.diag([0.3, 0.3, 0.3, 1.0])       # RAS: x index up = towards Right
    _save(tmp_path / "tf4" / "P003.nii.gz", v4, aff)
    _save(tmp_path / "tf3" / "imagesTr" / "ToothFairy3P_003_0000.nii.gz", v4[:, ::-1, ::-1].copy(), aff)
    _save(tmp_path / "tf3" / "labelsTr" / "ToothFairy3P_003.nii.gz", L4[:, ::-1, ::-1].copy(), aff)
    r = compare_case("P003", str(tmp_path / "tf4" / "P003.nii.gz"), str(tmp_path / "tf3"))
    assert r["voxel_identical_flip"] == "011"
    assert r["tf4_upper_above_lower"] is True and r["tf3_upper_above_lower"] is False
    assert r["tf4_left_canal_at_smaller_x"] is True


def test_tf3_different_scan_detected(tmp_path, vol):
    a = np.diag([0.3, 0.3, 0.3, 1.0])
    _save(tmp_path / "tf4" / "F002.nii.gz", vol, a)
    _save(tmp_path / "tf3" / "imagesTr" / "ToothFairy3F_002_0000.nii.gz", vol * 0 + 1, a)
    r = compare_case("F002", str(tmp_path / "tf4" / "F002.nii.gz"), str(tmp_path / "tf3"))
    assert r["frac_voxels_equal"] < 1.0 and r["max_abs_diff"] > 0


def test_tf3_missing(tmp_path, vol):
    _save(tmp_path / "tf4" / "S0001.nii.gz", vol, np.eye(4))
    r = compare_case("S0001", str(tmp_path / "tf4" / "S0001.nii.gz"), str(tmp_path / "tf3"))
    assert r["tf3_image"] is False


def test_duplicate_scores_and_cross_case_template():
    reps = pd.DataFrame({"case_id": ["C1", "C1", "C2", "C2", "C3"],
                         "report_id": ["1_1", "2_2", "1_3", "2_4", "3_5"],
                         "report_x": [1, 2, 1, 2, 3]})
    template = "no pathological findings in the examined teeth. bone level within normal limits."
    texts = [template, "Tooth 36 shows deep caries; periapical lesion at 46. Implant planned in region 25.",
             template, template.replace("normal", "regular"),
             "Impacted 38 near the mandibular canal; maxillary sinus partially visible."]
    pairs, nearest = score_pairs(reps, texts)
    c2 = pairs[pairs.case_id == "C2"].iloc[0]
    c1 = pairs[pairs.case_id == "C1"].iloc[0]
    assert c2["flagged"] and c2["levenshtein"] > 0.9
    assert not c1["flagged"] and c1["cosine"] < 0.5
    n = nearest.set_index("report_id")
    assert n.loc["1_1", "nn_case_id"] == "C2" and n.loc["1_1", "nn_cosine"] > 0.99
    assert set(pairs.columns) >= {"cosine", "levenshtein", "minhash_jaccard", "flagged"}
    assert "text" not in pairs.columns and "text" not in nearest.columns
