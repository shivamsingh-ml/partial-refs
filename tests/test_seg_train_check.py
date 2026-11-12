"""Synthetic tests for partialref.baseline.seg_train_check: each kind of fault must be flagged, clean cases not."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd

from partialref.baseline import seg_train_check as m

# same layout as V2V2R's mapping_DS119_singlepulp (tests must not depend on the external repo)
MAPPING = {**{i: i for i in range(1, 19)}, **{i: i - 2 for i in range(21, 29)}, **{i: i - 4 for i in range(31, 39)},
           **{i: i - 6 for i in range(41, 49)}, **{i: i - 60 for i in range(103, 106)},
           **{b + k: 46 for b in (111, 121, 131, 141) for k in range(8)}}


def _scan(seed: int, mirror=False, drop_q2=False, label_flip=False, permute=False):
    rng = np.random.default_rng(seed)
    L = np.zeros((48, 40, 40), np.int16)
    # axis 0 = left/right (Q1/Q4 at small index), axis 2 = up/down (upper at large index); slightly asymmetric
    for q, (x0, z0) in {1: (4, 26), 2: (27, 26), 3: (27, 6), 4: (4, 6)}.items():
        if drop_q2 and q == 2:
            continue
        for i, t in enumerate(range(q * 10 + 1, q * 10 + 9)):
            y = 2 + i * 4
            L[x0:x0 + 8 + (q == 1), y:y + 3, z0:z0 + 6] = t
            L[x0 + 3:x0 + 5, y + 1, z0 + 2:z0 + 4] = 100 + t
    L[6:12, 30:38, 1:3] = 4
    L[30:36, 30:38, 1:3] = 3
    I = (rng.normal(0, 50, L.shape) + 1000 * np.isin(L, m.TEETH + m.PULPS) + 3 * np.arange(48)[:, None, None])
    I = I.astype(np.int16)
    if mirror:
        L, I = L[::-1].copy(), I[::-1].copy()
    if permute:
        L, I = L.transpose(2, 1, 0).copy(), I.transpose(2, 1, 0).copy()
    if label_flip:
        L = L[:, ::-1].copy()
    return I, L


def _write(tf3, case, I, L):
    nib.save(nib.Nifti1Image(I, np.eye(4)), str(tf3 / "imagesTr" / f"ToothFairy3P_{case[1:]}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(L, np.eye(4)), str(tf3 / "labelsTr" / f"ToothFairy3P_{case[1:]}.nii.gz"))


def test_seg_train_check_flags_each_fault(tmp_path):
    tf3 = tmp_path / "tf3"
    (tf3 / "imagesTr").mkdir(parents=True)
    (tf3 / "labelsTr").mkdir()
    cases = {"P001": {}, "P002": {}, "P003": {}, "P004": {}, "P005": {"mirror": True}, "P006": {"drop_q2": True},
             "P007": {"label_flip": True}, "P008": {"permute": True}}
    for i, (c, kw) in enumerate(cases.items()):
        _write(tf3, c, *_scan(i, **kw))
    rows = [m.check_case(c, str(tf3), MAPPING) for c in [*cases, "P999"]]
    df = pd.DataFrame(rows)
    df.to_parquet(tmp_path / "x.parquet", index=False)                       # writable as in main()
    s = m.summarise(df)
    assert s["missing_files"] == ["P999"]
    assert s["errors"] == []
    assert s["label_not_on_image_grid"] == []
    assert s["converted_labels_not_ok"] == [] and s["unknown_labels"] == {} and s["dropped_by_conversion"] == {}
    assert s["left_right_axis_is_nnunet_mirror_axis"] and s["left_right_same_axis_and_sign"]
    lr_flagged = {f["case_id"] for x in s["left_right"] for f in x["flagged"]}
    assert "P005" in lr_flagged and "P008" in lr_flagged                      # mirrored scan, permuted axes
    assert not lr_flagged & {"P001", "P002", "P003", "P004", "P006"}
    assert "P006" in s["left_right"][0]["unchecked"] and "P006" not in s["cases_without_left_right_check"]
    assert s["label_not_fitting_image"] == ["P007"]
    assert {f["case_id"] for f in s["single_teeth_wrong_side"]} <= {"P005", "P007", "P008"}
    assert s["pulp"]["correct_share_all"] == 1.0
    json_ok = __import__("json").dumps(s, default=str)
    assert "P005" in json_ok


def test_pulp_reassign_without_tooth_and_wrong_neighbour():
    L = np.zeros((12, 12, 12), np.int16)
    L[2:4, 2:4, 2:4] = 111                                                   # pulp, no tooth at all
    r = m.pulp_reassign(L)
    assert r["pulp_correct_share"] == 0.0 and r["pulp_vox_without_own_tooth"] == 8
    L[5:8, 2:4, 2:4] = 12                                                    # only a neighbouring tooth
    r = m.pulp_reassign(L)
    assert r["pulp_correct_share"] == 0.0 and r["pulps_without_own_tooth"] == "[11]"
