"""Synthetic end-to-end test of partialref.baseline.seg_dataset with V2V2R's real conversion script (no real data)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from partialref.baseline import seg_dataset as sd

UMAMBA2 = Path("/msc/home/ssingh54/U-Mamba2")
SCRIPT = UMAMBA2 / "scripts" / "convert_tf3_singlepulp.py"
TEETH = [t for q in (10, 20, 30, 40) for t in range(q + 1, q + 9)]
IDS = list(range(1, 11)) + TEETH + [103, 104, 105] + [100 + t for t in TEETH]


def _case(tf3: Path, cid: str, seed: int, extra: dict | None = None):
    rng = np.random.default_rng(seed)
    L = np.zeros((16, 18, 20), np.uint8)
    L[1:4, 1:4, 1:4] = 1
    L[5:8, 5:8, 5:8] = 16
    L[6, 6, 6] = 116
    L[10:13, 2:5, 2:5] = 28
    L[2:4, 12:15, 12:15] = 100                                    # unlisted label -> background in V2V2R's script
    for (x, v) in (extra or {}).items():
        L[x] = v
    I = rng.normal(0, 50, L.shape).astype(np.int16)
    aff = np.diag([0.3, 0.3, 0.3, 1.0])
    c, d = cid[0], cid[1:]
    nib.save(nib.Nifti1Image(I, aff), str(tf3 / "imagesTr" / f"ToothFairy3{c}_{d}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(L, aff), str(tf3 / "labelsTr" / f"ToothFairy3{c}_{d}.nii.gz"))


@pytest.mark.skipif(not SCRIPT.is_file(), reason="V2V2R conversion script not on this machine")
def test_build_and_verify(tmp_path):
    tf3 = tmp_path / "tf3"
    (tf3 / "imagesTr").mkdir(parents=True)
    (tf3 / "labelsTr").mkdir()
    (tf3 / "dataset.json").write_text(json.dumps({"labels": {"background": 0, **{f"L{i}": i for i in IDS}},
                                                  "channel_names": {"0": "CBCT"}}))
    for i, cid in enumerate(["P001", "P002", "P003", "P009"]):
        _case(tf3, cid, i)
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "splits.csv").write_text("case_id,centre,role,fold\nP001,P,dev,0\nP002,P,dev,1\nP003,P,dev,2\n"
                                       "P009,P,test,-1\n")
    (splits / "splits.json").write_text(json.dumps({"tf3_only_for_segmentation": []}))
    um = tmp_path / "um"
    (um / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, um / "scripts" / SCRIPT.name)
    fixes = tmp_path / "fixes.json"
    fixes.write_text(json.dumps({"exclude": {"P003": "bad"}, "fixes": {"P002": [
        {"op": "relabel", "label": 28, "to": 18, "scope": ["train", "module"], "expect_voxels": 27, "why": "t"},
        {"op": "relabel", "label": 116, "to": 126, "scope": ["module"], "expect_voxels": 1, "why": "t"}]}}))
    out, raw = tmp_path / "out", tmp_path / "raw"
    rc = sd.main(["--tf3-dir", str(tf3), "--splits", str(splits), "--fixes", str(fixes), "--umamba2-dir", str(um),
                  "--raw-dir", str(raw), "--dataset", "Dataset999_T", "--out", str(out), "--np", "1"])
    s = json.loads((out / "summary.json").read_text())
    assert rc == 0, s
    assert s["n_cases"] == 2 and s["case_list_equal"] and s["n_ok"] == 2      # no test case, P003 excluded
    assert s["fixed_scans"] == {"P002": 27}                                    # module-only op not in training
    ds = raw / "Dataset999_T"
    got = np.asarray(nib.load(str(ds / "labelsTr" / "ToothFairy3P_002.nii.gz")).dataobj)
    assert (got == 18).sum() == 27 and not (got == 26).any()                  # 18 kept as 18; 28 gone
    assert (got == 46).sum() == 1                                              # pulp merged
    orig = np.asarray(nib.load(str(tf3 / "labelsTr" / "ToothFairy3P_002.nii.gz")).dataobj)
    assert (orig == 28).sum() == 27                                            # raw file untouched
    assert not (ds / "labelsTr" / "ToothFairy3P_009.nii.gz").exists()
    # a tampered converted label is caught
    bad = got.copy()
    bad[0, 0, 0] = 5
    nib.save(nib.Nifti1Image(bad, nib.load(str(ds / "labelsTr" / "ToothFairy3P_002.nii.gz")).affine),
             str(ds / "labelsTr" / "ToothFairy3P_002.nii.gz"))
    from partialref.baseline.label_fixes import load_fixes
    from partialref.baseline.seg_train_check import load_mapping
    r = sd.verify_case("P002", str(tf3), str(ds), load_fixes(fixes), load_mapping(um), 27)
    assert not r["label_equal"] and not r["ok"]
    import os
    assert all(Path(os.readlink(f)).parent == tf3 / "imagesTr" for f in (ds / "imagesTr").iterdir())  # direct links
    assert sorted(p.name for p in raw.iterdir()) == ["Dataset999_T"]          # no staging next to the dataset
    with pytest.raises(FileExistsError):                                       # never rebuilt over
        sd.main(["--tf3-dir", str(tf3), "--splits", str(splits), "--fixes", str(fixes), "--umamba2-dir", str(um),
                 "--raw-dir", str(raw), "--dataset", "Dataset999_T", "--out", str(out), "--np", "1"])
