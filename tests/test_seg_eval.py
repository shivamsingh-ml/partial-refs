"""Synthetic test of partialref.baseline.seg_eval scoring (no real data, no network)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd

from partialref.baseline import seg_eval as se


def _scan(tmp_path):
    rng = np.random.default_rng(0)
    L = np.zeros((20, 22, 24), np.uint8)
    L[2:6, 2:6, 2:6] = 1                       # lower jawbone
    L[8:12, 3:7, 3:7] = 16                     # TF3 tooth 16 -> V2V2R scheme 26 (mirror)
    L[9:11, 4:6, 4:6] = 116                    # its pulp -> 50
    L[14:18, 10:14, 10:14] = 31                # TF3 31 -> 41
    L[1:3, 15:18, 15:18] = 3                   # TF3 canal 3 -> 4 (mirror)
    I = rng.integers(-1000, 3000, L.shape).astype(np.int16)
    tf3, tf4 = tmp_path / "tf3", tmp_path / "tf4"
    (tf3 / "imagesTr").mkdir(parents=True)
    (tf3 / "labelsTr").mkdir()
    (tf4 / "P001" / "cbct").mkdir(parents=True)
    aff = np.diag([0.3, 0.3, 0.3, 1.0])
    nib.save(nib.Nifti1Image(I, aff), str(tf3 / "imagesTr" / "ToothFairy3P_001_0000.nii.gz"))
    nib.save(nib.Nifti1Image(L, aff), str(tf3 / "labelsTr" / "ToothFairy3P_001.nii.gz"))
    tf4_img = nib.Nifti1Image(np.ascontiguousarray(np.flip(I, (1, 2))), aff)
    nib.save(tf4_img, str(tf4 / "P001" / "cbct" / "volume.nii.gz"))
    return tf3, tf4, L, aff


def test_reference_is_mirrored_and_on_tf4_grid(tmp_path):
    tf3, tf4, L, _ = _scan(tmp_path)
    ref, _ = se.reference("P001", tf3, tf4)
    back = np.flip(ref, (1, 2))                                   # undo the TF4 flip
    assert set(np.unique(back[L == 16])) == {26} and set(np.unique(back[L == 116])) == {50}
    assert set(np.unique(back[L == 31])) == {41} and set(np.unique(back[L == 3])) == {4}


def test_score_perfect_missed_extra(tmp_path):
    tf3, tf4, L, aff = _scan(tmp_path)
    ref, _ = se.reference("P001", tf3, tf4)
    pred = ref.copy()
    pred[pred == 41] = 0                                          # tooth missed
    pred[0:2, 0:2, 0:2] = 11                                      # extra tooth
    pdir = tmp_path / "pred"
    pdir.mkdir()
    nib.save(nib.Nifti1Image(pred, aff), str(pdir / "P001.nii.gz"))
    rows = se.score_case("P001", str(tf3), str(tf4), str(pdir))
    d = {r["label"]: r["dice"] for r in rows}
    assert all(r["error"] is None for r in rows)
    assert d[26] == 1.0 and d[50] == 1.0 and d[1] == 1.0 and d[4] == 1.0 and d[41] == 0.0 and d[11] == 0.0
    s = se.summarise(pd.DataFrame(rows), {"P001": "P"})
    assert s["teeth_counts"] == {"in_reference": 2, "missed": 1, "extra": 1, "below_0.5": 1}
    assert s["teeth"]["mean_dice"] == 0.5 and s["pulp"]["mean_dice"] == 1.0


def test_wrong_grid_is_an_error(tmp_path):
    tf3, tf4, L, aff = _scan(tmp_path)
    pdir = tmp_path / "pred"
    pdir.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((20, 22, 23), np.uint8), aff), str(pdir / "P001.nii.gz"))
    rows = se.score_case("P001", str(tf3), str(tf4), str(pdir))
    assert rows[0]["error"] and "grid" in rows[0]["error"]
