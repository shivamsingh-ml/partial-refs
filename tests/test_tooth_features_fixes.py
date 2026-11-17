"""Synthetic test: tooth_features applies module-scope label fixes before measuring (no real data)."""

from __future__ import annotations

import nibabel as nib
import numpy as np

from partialref.perception import tooth_features as tf


def _write(tmp_path):
    rng = np.random.default_rng(0)
    L = np.zeros((40, 40, 40), np.int16)
    L[5:35, 5:35, 25:35] = 1                                  # jaw bone below the root end
    L[10:20, 10:20, 8:30] = 28                                # one tooth (wrong quadrant in this "scan")
    L[13:17, 13:17, 12:26] = 128                              # its pulp
    L[15, 15, 9] = 0
    I = rng.normal(300, 30, L.shape)
    I[L == 1] = 900
    I[L == 28] = 1800
    I[L == 128] = 600
    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "labelsTr").mkdir()
    aff = np.diag([0.3, 0.3, 0.3, 1.0])
    nib.save(nib.Nifti1Image(I.astype(np.int16), aff), str(tmp_path / "imagesTr" / "ToothFairy3P_001_0000.nii.gz"))
    nib.save(nib.Nifti1Image(L.astype(np.uint8), aff), str(tmp_path / "labelsTr" / "ToothFairy3P_001.nii.gz"))
    return int((L == 28).sum()), int((L == 128).sum())


def test_case_rows_applies_module_fixes(tmp_path):
    n_tooth, n_pulp = _write(tmp_path)
    plain = tf.case_rows("P001", str(tmp_path))
    assert [r["fdi_mask"] for r in plain] == [28]
    fixes = {"exclude": {}, "fixes": {"P001": [
        {"op": "relabel", "label": 28, "to": 18, "scope": ["train", "module"], "expect_voxels": n_tooth, "why": "t"},
        {"op": "relabel", "label": 128, "to": 118, "scope": ["module"], "expect_voxels": n_pulp, "why": "t"}]}}
    fixed = tf.case_rows("P001", str(tmp_path), fixes)
    assert [r["fdi_mask"] for r in fixed] == [18] and fixed[0]["fdi_report"] == 28   # report number = mirror
    assert abs(fixed[0]["pulp_share"] - plain[0]["pulp_share"]) < 1e-9               # pulp followed its tooth
    other = tf.case_rows("P002", str(tmp_path), fixes)                                 # scan not in the table
    assert other[0]["error"] is not None                                               # (no such file) untouched path
