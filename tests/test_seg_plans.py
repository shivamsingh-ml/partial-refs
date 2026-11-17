"""Synthetic tests for partialref.baseline.seg_plans (no real data)."""

from __future__ import annotations

import json

import pytest

from partialref.baseline import seg_plans as sp

SRC = {"dataset_name": "Dataset703_X", "plans_name": "nnUNetResEncUNetLPlans_torchres",
       "original_median_spacing_after_transp": [0.3, 0.3, 0.3], "original_median_shape_after_transp": [169, 349, 371],
       "configurations": {"3d_fullres": {"patch_size": [112, 224, 256], "normalization_schemes": ["ZScoreNormalization"]},
                          "ps160": {"inherits_from": "3d_fullres", "patch_size": [160, 288, 288],
                                    "normalization_schemes": ["CTNormalization"]}},
       "foreground_intensity_properties_per_channel": {"0": {"max": 15383.0, "mean": 810.6, "median": 929.5,
                                                             "min": -1000.0, "percentile_00_5": -997.0,
                                                             "percentile_99_5": 3608.0, "std": 1020.9}}}
FP = {"foreground_intensity_properties_per_channel": {"0": {"max": 9000.0, "mean": 800.0, "median": 920.0,
                                                            "min": -1000.0, "percentile_00_5": -996.0,
                                                            "percentile_99_5": 3500.0, "std": 1000.0}},
      "spacings": [[0.3, 0.3, 0.3]] * 3, "shapes_after_crop": [[170, 350, 370]] * 3}


def test_only_two_keys_change():
    out = sp.make_plans(SRC, FP, "Dataset713_Y")
    assert out["dataset_name"] == "Dataset713_Y"
    assert out["foreground_intensity_properties_per_channel"]["0"]["std"] == 1000.0
    assert out["configurations"] == SRC["configurations"]
    assert SRC["dataset_name"] == "Dataset703_X"                                 # source untouched


def test_bad_intensity_values_stop():
    bad = json.loads(json.dumps(FP))
    bad["foreground_intensity_properties_per_channel"]["0"]["percentile_99_5"] = 100.0   # below the mean
    with pytest.raises(ValueError, match="implausible"):
        sp.make_plans(SRC, bad, "D")
    bad = json.loads(json.dumps(FP))
    bad["foreground_intensity_properties_per_channel"]["1"] = bad["foreground_intensity_properties_per_channel"]["0"]
    with pytest.raises(ValueError, match="channels differ"):
        sp.make_plans(SRC, bad, "D")


def _splits(tmp_path):
    rows = ["case_id,centre,role,fold"] + [f"P{i:03d},P,dev,{i % 5}" for i in range(10)] + \
           ["P100,P,test,-1", "S0015,S,dev,0", "A001,A,external_A,-1"]
    (tmp_path / "splits.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "splits.json").write_text(json.dumps({"tf3_only_for_segmentation": ["P120"]}))
    return tmp_path


def test_folds(tmp_path):
    s = _splits(tmp_path)
    fixes = {"exclude": {"S0015": "x"}, "fixes": {}}
    built = {f"ToothFairy3P_{i:03d}" for i in range(10)} | {"ToothFairy3P_120"}
    folds = sp.make_folds(s, fixes, built)
    assert len(folds) == 5 and all(len(f["val"]) == 2 for f in folds)
    assert all("ToothFairy3P_120" in f["train"] and "ToothFairy3P_120" not in f["val"] for f in folds)
    assert not any("ToothFairy3S_0015" in f["train"] + f["val"] or "ToothFairy3P_100" in f["train"] + f["val"]
                   for f in folds)
    with pytest.raises(AssertionError, match="differ"):                         # built dataset must match the folds
        sp.make_folds(s, fixes, built | {"ToothFairy3P_100"})
