"""Visibility features on a synthetic label volume (no real data)."""

import nibabel as nib
import numpy as np

from partialref.audit.visibility_features import case_features, face_axes, touches


def test_face_axes_ras_and_las():
    ras = np.diag([0.3, 0.3, 0.3, 1.0])
    f = face_axes(ras)
    assert f["sup"] == (2, 1) and f["inf"] == (2, 0) and f["right"] == (0, 1)
    las = np.diag([-0.3, 0.3, 0.3, 1.0])
    assert face_axes(las)["right"] == (0, 0)


def test_touches():
    m = np.zeros((4, 4, 4), bool)
    m[1, 1, 3] = True
    t = touches(m, face_axes(np.diag([1.0, 1.0, 1.0, 1.0])))
    assert t["sup"] and not t["inf"] and not t["left"]


def test_case_features_root_tip_and_condyle(tmp_path):
    # TF4 grid (RAS): z index up = superior. Upper tooth 16 reaches the top face (root tip cut),
    # lower tooth 36 stays inside. Mandible right half touches the top face (condyle cut), left not.
    L4 = np.zeros((10, 10, 12), np.int16)
    L4[7:9, 4:6, 6:12] = 16      # upper tooth, touches z=11 (sup)
    L4[1:3, 4:6, 2:5] = 36       # lower tooth (left side, x<5), inside
    L4[1:4, 2:8, 1:4] = 1        # mandible left part, low
    L4[6:9, 2:8, 1:12] = np.where(L4[6:9, 2:8, 1:12] == 0, 1, L4[6:9, 2:8, 1:12])  # right ramus to top
    aff = np.diag([0.3, 0.3, 0.3, 1.0])
    tf4 = tmp_path / "tf4" / "P001.nii.gz"
    tf4.parent.mkdir()
    nib.save(nib.Nifti1Image(np.zeros(L4.shape, np.float32), aff), str(tf4))
    lab_dir = tmp_path / "tf3" / "labelsTr"
    lab_dir.mkdir(parents=True)
    nib.save(nib.Nifti1Image(L4[:, ::-1, ::-1].copy(), aff), str(lab_dir / "ToothFairy3P_001.nii.gz"))

    row, teeth = case_features("P001", str(tf4), str(tmp_path / "tf3"))
    t = {r["fdi"]: r for r in teeth}
    assert t[16]["present"] and t[16]["root_tip_cut"] is True
    assert t[36]["present"] and t[36]["root_tip_cut"] is False
    assert not t[11]["present"]
    assert row["mandible_right_touches_sup"] is True and row["mandible_left_touches_sup"] is False
    assert row["n_upper_teeth"] == 1 and row["n_lower_teeth"] == 1
    assert row["q1_n_teeth"] == 1 and row["q3_n_teeth"] == 1

    # predicted-mask path (centre A): mask already on the TF4 grid, no flip, pulp 199 ignored
    pred = tmp_path / "pred"
    pred.mkdir()
    Lp = L4.copy()
    Lp[0, 0, 0] = 199
    nib.save(nib.Nifti1Image(Lp, aff), str(pred / "A001.nii.gz"))
    rowp, teethp = case_features("A001", str(tf4), str(tmp_path / "tf3"), pred_dir=str(pred))
    assert rowp["label_source"] == "umamba2_foldall_pred"
    assert {r["fdi"]: r for r in teethp}[16]["root_tip_cut"] is True
    assert rowp["mandible_right_touches_sup"] is True
