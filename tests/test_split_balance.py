"""Synthetic tests for partialref.data.split_balance (no real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from partialref.data import split_balance as sb


def _fake(tmp_path, implant_test_share):
    rng = np.random.default_rng(0)
    n = 400
    ids = [f"P{i:03d}" for i in range(n)]
    test = np.zeros(n, int)
    test[:80] = 1
    implant = np.where(test == 1, rng.random(n) < implant_test_share, rng.random(n) < 0.1)
    c = pd.DataFrame({"case_id": ids, "centre": "P", "shape": "370x352x170", "n_upper_teeth": 10,
                      "n_lower_teeth": 10, "maxilla_vox": 1, "implant_vox": implant.astype(int), "crown_vox": 0,
                      "bridge_vox": 0})
    c.loc[:9, ["n_upper_teeth", "n_lower_teeth"]] = 0                   # a few toothless scans (no tooth rows)
    t = pd.DataFrame([{"case_id": i, "fdi": 11, "present": True, "crown_cut": False, "root_tip_cut": True}
                      for i in ids[10:]] + [{"case_id": i, "fdi": 12, "present": False, "crown_cut": None,
                                             "root_tip_cut": None} for i in ids], dtype=object)  # as the real file
    feat = tmp_path / "f"
    feat.mkdir()
    c.to_parquet(feat / "case_features.parquet")
    t.to_parquet(feat / "tooth_features.parquet")
    sp = tmp_path / "s"
    sp.mkdir()
    pd.DataFrame({"case_id": ids, "centre": "P", "role": np.where(test == 1, "test", "dev"),
                  "fold": np.where(test == 1, -1, np.arange(n) % 5)}).to_csv(sp / "splits.csv", index=False)
    (sp / "splits.json").write_text('{"never_test": {"P399": "x"}}')
    man = tmp_path / "m"
    man.mkdir()
    pd.DataFrame({"case_id": ids, "n_reports_en": np.where(np.arange(n) % 3 == 0, 2, 1)}).to_parquet(
        man / "cases.parquet")
    return sp, feat, man


def test_balance_detects_imbalance_and_not_balance(tmp_path):
    sp, feat, man = _fake(tmp_path, implant_test_share=0.5)
    c = sb.case_table(sp, feat, man)
    assert c.loc[c.case_id == "P399", "never_test"].item()
    b = sb.balance(c, n_perm=999).set_index("property")
    assert b.loc["teeth per scan", "design_variable"] and not b.loc["implant label", "design_variable"]
    assert b.loc["implant label", "smd_test_minus_dev"] > 0.5
    assert b.loc["implant label", "perm_p"] < 0.01
    assert b.loc["teeth per scan", "smd_test_minus_dev"] == b.loc["teeth per scan", "smd_test_minus_dev"]  # not NaN
    assert b.loc["upper jaw labelled", "perm_p"] == 1.0            # identical everywhere
    assert np.isnan(b.loc["share of teeth root-cut", "dev"]) is np.False_         # toothless scans dropped, not 0
    assert abs(b.loc["share of teeth root-cut", "dev"] - 1.0) < 1e-9


def test_balanced_split_gives_large_p(tmp_path):
    sp, feat, man = _fake(tmp_path, implant_test_share=0.1)
    b = sb.balance(sb.case_table(sp, feat, man), n_perm=999).set_index("property")
    assert b.loc["implant label", "perm_p"] > 0.05
