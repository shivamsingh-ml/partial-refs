"""Case-level splits (synthetic case tables only)."""

import numpy as np
import pandas as pd
import pytest

from partialref.data.splits import NEVER_TEST, TF3_ONLY, build, prepare


def _tables(seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"P{i:03d}" for i in range(1, 301)] + [f"F{i:03d}" for i in range(1, 61)] + \
          [f"S{i:04d}" for i in range(1, 51)] + [f"A{i:03d}" for i in range(1, 41)] + ["P242", "S0015"]
    ids = sorted(set(ids) - set(TF3_ONLY))                                   # TF3-only scans have no TF4 case
    cases = pd.DataFrame({"case_id": ids, "centre": [c[0] for c in ids],
                          "n_reports_en": rng.choice([1, 2], len(ids), p=[0.3, 0.7])})
    pfs = [c for c in ids if c[0] != "A"]
    feats = pd.DataFrame({"case_id": pfs, "n_upper_teeth": rng.integers(0, 16, len(pfs)),
                          "n_lower_teeth": rng.integers(0, 16, len(pfs))})
    return cases, feats


def test_split_rules():
    cases, feats = _tables()
    d = build(prepare(cases, feats), 0.2, 7)
    assert set(d.loc[d.centre == "A", "role"]) == {"external_A"} and (d.loc[d.centre == "A", "fold"] == -1).all()
    test = d[d.role == "test"]
    assert not test.case_id.isin(NEVER_TEST).any() and {"P242", "S0015"} <= set(d.loc[d.role == "dev", "case_id"])
    pfs = d[d.centre != "A"]
    for (c, r), g in pfs.groupby(["centre", "read"]):                      # exact per cell
        assert (g.role == "test").sum() == round(0.2 * len(g))
    folds = pfs[pfs.role == "dev"].fold.value_counts()
    assert folds.max() - folds.min() <= 1 and set(folds.index) == set(range(5))
    assert d.equals(build(prepare(cases, feats), 0.2, 7))                    # reproducible
    assert not d.equals(build(prepare(cases, feats), 0.2, 8))


def test_prepare_guards():
    cases, feats = _tables()
    with pytest.raises(ValueError):
        prepare(cases, feats.iloc[1:])                                        # a P/F/S case without features
    with pytest.raises(ValueError):
        prepare(pd.concat([cases, pd.DataFrame([{"case_id": "P120", "centre": "P", "n_reports_en": 1}])]), feats)
    d = prepare(cases, feats)
    assert set(d.loc[d.centre == "A", "coverage"]) == {"na"} and set(d.coverage) <= {"na", "both_jaws", "not_both"}
