"""Synthetic tests for partialref.baseline.label_fixes (no real data)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from partialref.baseline import label_fixes as lf


def _vol():
    L = np.zeros((20, 20, 20), np.int16)
    L[2:5, 2:5, 2:5] = 26          # 27 voxels
    L[10:12, 2:4, 2:4] = 36        # 8 voxels
    L[2:4, 10:13, 10:12] = 27      # upper piece, 12 voxels
    L[15:17, 15:17, 10:12] = 27    # lower piece touching 37, 8 voxels
    L[15:17, 15:17, 12:15] = 37    # 12 voxels
    L[3, 3, 3] = 126
    return L


def _table(ops, exclude=None):
    return {"exclude": exclude or {}, "fixes": {"P001": ops}}


def test_swap_relabel_component_with_counts():
    L = _vol()
    ops = [{"op": "component_to", "label": 27, "touching": 37, "to": 37, "merge": True, "scope": ["train"],
            "expect_voxels": 8},
           {"op": "swap", "a": 26, "b": 36, "scope": ["train", "module"], "expect_voxels": [26, 8]},
           {"op": "relabel", "label": 126, "to": 136, "scope": ["module"], "expect_voxels": 1}]
    out, log = lf.apply_fixes(L, "P001", _table(ops), "train")
    assert (out == 27).sum() == 12 and (out == 37).sum() == 20          # only the touching piece moved
    assert (out[2:5, 2:5, 2:5] == 36).sum() == 26 and (out[10:12, 2:4, 2:4] == 26).all()
    assert (out == 126).sum() == 1                                      # module-only op skipped in train
    assert [e["changed"] for e in log] == [8, [26, 8]]
    out_m, _ = lf.apply_fixes(L, "P001", _table(ops), "module")
    assert (out_m == 136).sum() == 1 and (out_m == 27).sum() == 20       # train-only op skipped in module
    assert (L == 26).sum() == 26                                         # input untouched


def test_count_mismatch_absent_label_and_duplicates_raise():
    L = _vol()
    with pytest.raises(ValueError, match="expected"):
        lf.apply_fixes(L, "P001", _table([{"op": "relabel", "label": 36, "to": 46, "scope": ["train"],
                                            "expect_voxels": 9}]), "train")
    with pytest.raises(ValueError, match="nothing to change"):
        lf.apply_fixes(L, "P001", _table([{"op": "relabel", "label": 48, "to": 38, "scope": ["train"],
                                            "expect_voxels": 0}]), "train")
    with pytest.raises(ValueError, match="already present"):                 # 26 -> 37 without merge
        lf.apply_fixes(L, "P001", _table([{"op": "relabel", "label": 26, "to": 37, "scope": ["train"],
                                            "expect_voxels": 26}]), "train")
    with pytest.raises(ValueError, match="expect_voxels missing"):
        lf.apply_fixes(L, "P001", _table([{"op": "relabel", "label": 36, "to": 46, "scope": ["train"],
                                            "expect_voxels": None}]), "train")
    with pytest.raises(ValueError, match="excluded"):
        lf.apply_fixes(L, "P001", _table([], exclude={"P001": "bad"}), "train")


def test_component_to_ignores_pieces_not_touching_and_corner_contact():
    L = np.zeros((10, 10, 10), np.int16)
    L[1:3, 1:3, 1:3] = 27
    L[3, 3, 3] = 37                       # touches 27 only at a corner -> counts (26-connectivity)
    L[7:9, 7:9, 7:9] = 27                 # far piece stays
    out, _ = lf.apply_op(L, {"op": "component_to", "label": 27, "touching": 37, "to": 37, "merge": True,
                             "expect_voxels": 8})
    assert (out == 27).sum() == 8 and (out[7:9, 7:9, 7:9] == 27).all()


def test_shipped_table_is_valid_and_complete():
    t = lf.load_fixes("configs/seg/label_fixes_v1.json")
    assert "S0015" in t["exclude"]
    ops = [op for v in t["fixes"].values() for op in v]
    assert all(op.get("expect_voxels") is not None for op in ops), "fill expect_voxels from --dry-run"
    assert all(op.get("why") for op in ops)
    json.dumps(t)


def test_table_field_check_and_merge_without_target():
    import tempfile
    from pathlib import Path
    bad = {"exclude": {}, "fixes": {"P001": [{"op": "swap", "a": 26, "scope": ["train"], "why": "x"}]}}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="lacks"):
            lf.load_fixes(p)
    L = _vol()
    with pytest.raises(ValueError, match="merge set but label 48 is absent"):
        lf.apply_op(L, {"op": "relabel", "label": 36, "to": 48, "merge": True, "expect_voxels": 8})
