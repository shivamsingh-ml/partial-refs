"""Per-tooth features and labels (synthetic volumes and tables only)."""

import numpy as np
import pandas as pd

from partialref.perception.tooth_classifier import (
    FEATURES,
    TYPES,
    evaluate,
    silent_authors,
    tooth_labels,
)
from partialref.perception.tooth_features import mirror, tooth_row

SP = np.array([0.3, 0.3, 0.3])


def _volume(cap=False, canal_fill=False, dark_apex=False, crown_label=False, root_up=False, deep_fill=False):
    """Cylinder tooth 36 along axis 2 (z 10..49), pulp label 136 in its centre, root part (z < 34) in
    bone (label 1). Optional: bright cap on the crown, bright filling in the pulp of the root, dark zone
    just beyond the root tip, a prosthetic crown label (9) over the crown end. root_up flips axis 2."""
    I = np.zeros((70, 70, 60), np.float32)
    L = np.zeros(I.shape, np.int16)
    x, y, z = np.meshgrid(np.arange(70), np.arange(70), np.arange(60), indexing="ij")
    r = np.sqrt((x - 35) ** 2 + (y - 35) ** 2)
    tooth = (r <= 12) & (z >= 10) & (z < 50)                  # 7 mm wide, 12 mm long
    pulp = tooth & (r <= 1.5) & (z < 45)
    bone = (z < 34) & ~tooth
    I[bone], L[bone] = 900, 1
    I[tooth], L[tooth] = 2000, 36
    I[pulp], L[pulp] = 1000, 136
    if cap:
        I[tooth & (z >= 44) & (r >= 11)] = 3000
    if deep_fill:                                              # filling deep in the crown, below the surface
        I[tooth & (z >= 40) & (z < 47) & (r <= 5) & ~pulp] = 3000
    if canal_fill:
        I[pulp & (z < 30)] = 3000
    if dark_apex:                                              # round radiolucency around the root tip
        I[(np.sqrt((x - 35) ** 2 + (y - 35) ** 2 + (z - 10) ** 2) <= 14) & bone] = 200
    if crown_label:                                            # a prosthetic crown wraps the crown end
        L[((z >= 50) & (z < 53) & (r <= 14)) | ((z >= 40) & (z < 50) & (r > 12) & (r <= 14))] = 9
    I[0, 0, 0] = 3000                                          # scan maximum
    if root_up:
        I, L = I[:, :, ::-1].copy(), L[:, :, ::-1].copy()
    return I, L


def _row(**kw):
    I, L = _volume(**kw)
    return tooth_row(I, L, 36, SP, float(I.max()), float(np.median(I[L == 1])))


def test_features_find_crown_canal_filling_apex_and_labels_in_either_orientation():
    plain = _row()
    assert plain["fdi_report"] == 46 and mirror(46) == 36 and 0 < plain["pulp_share"] < 0.2
    assert plain["bright_crown"] == 0 and plain["bright_pulp"] == 0 and plain["end_bone_contrast"] > 0.3
    capped = _row(cap=True)
    assert capped["bright_crown_surface"] > 0.3 and capped["bright_root"] == 0
    filled = _row(canal_fill=True)
    assert filled["bright_root_pulp"] > 0.5 and filled["bright_crown"] == 0
    apex = _row(dark_apex=True)
    assert apex["apex_rel_local"] < 0.5 < plain["apex_rel_local"] and apex["apex_dark_share_local"] > 0.5
    assert plain["apex_dark_share_local"] == 0
    deep = _row(deep_fill=True)
    assert deep["bright_crown_deep"] > 0.1 and deep["largest_deep_bright_mm3"] > 1
    assert capped["bright_crown_deep"] == 0 and capped["largest_deep_bright_mm3"] == 0     # enamel-like shell only
    assert _row(crown_label=True)["prosthetic_label_near_crown"] > 0.1 and plain["prosthetic_label_near_crown"] == 0
    flipped = _row(cap=True, root_up=True)
    assert abs(flipped["bright_crown_surface"] - capped["bright_crown_surface"]) < 0.05 and flipped["bright_root"] == 0
    assert not plain["truncated"]
    I, L = _volume()
    assert tooth_row(I, L, 11, SP, 3000.0, 900.0) is None


def test_labels_skip_silent_authors_and_mark_unknown():
    profiles = pd.DataFrame([{"centre": "P", "author": 14.0, "finding_type": "ConservativeRestoration",
                              "both": 0, "only_partner": 40, "rate_hi": 0.0}])
    silent = silent_authors(profiles, 10)
    assert silent == {("P", "14", "ConservativeRestoration")}                      # '14.0' normalised
    reps = pd.DataFrame({"case_id": ["P001", "P001", "P002", "P003", "P004"],
                         "report_id": ["14_1", "20_1", "14_2", "20_3", "14_4"], "report_x": [14, 20, 14, 20, 14]})
    inst = pd.DataFrame({"case_id": ["P001", "P003", "P003", "P004"], "report_id": ["20_1", "20_3", "20_3", "14_4"],
                         "finding_type": ["ConservativeRestoration", "Caries", "ConservativeRestoration",
                                          "ConservativeRestoration"],
                         "slot": ["36", "36", "Left", "36"]})
    teeth = pd.DataFrame({"case_id": ["P001", "P001", "P002", "P003", "P004"], "fdi_report": [36, 46, 36, 36, 36]})
    lab = tooth_labels(teeth, reps, inst, silent).set_index(["case_id", "fdi_report"])
    assert lab.loc[("P001", 36), "ConservativeRestoration"] == 1
    assert lab.loc[("P001", 46), "ConservativeRestoration"] == 0
    assert np.isnan(lab.loc[("P002", 36), "ConservativeRestoration"])
    assert np.isnan(lab.loc[("P004", 36), "ConservativeRestoration"])              # silent author's mention ignored
    assert lab.loc[("P002", 36), "Caries"] == 0 and lab.loc[("P003", 36), "Caries"] == 1
    assert set(TYPES) <= set(lab.columns)


def test_evaluate_beats_reference_and_reports_per_centre():
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({"case_id": [f"{'PFS'[i % 3]}{i // 10:03d}" for i in range(n)],
                       "fdi_report": rng.choice([16, 26, 36, 46, 11], n)})
    for f in FEATURES:
        df[f] = rng.normal(size=n)
    y = (rng.random(n) < 0.2).astype(float)
    df["bright_root_pulp"] = y * 2 + rng.normal(size=n) * 0.5
    for t in TYPES:
        df[t] = np.nan
    df["EndodonticTreatment"] = y
    folds = {c: i % 5 for i, c in enumerate(sorted(df["case_id"].unique()))}
    res = evaluate(df, folds)
    assert set(res["finding_type"]) == {"EndodonticTreatment"} and set(res["centre"]) == {"all", "F", "P", "S"}
    r = res.set_index("centre").loc["all"]
    assert r["auc_features"] > 0.9 and r["auc_reference"] < 0.6


def test_scan_levels_and_histogram_features_also_when_the_scanner_clips():
    from partialref.perception.tooth_classifier import feature_columns
    from partialref.perception.tooth_features import hist_feats, scan_levels
    rng = np.random.default_rng(0)
    pulp, dentin, enamel = rng.normal(800, 50, 5000), rng.normal(1800, 60, 20000), rng.normal(2500, 50, 8000)
    lv = scan_levels(np.concatenate([pulp, dentin, enamel]), 3000.0)
    assert abs(lv["lvl_dentin"] - 1800) < 60 and abs(lv["lvl_enamel"] - 2500) < 80 and lv["clip_share"] == 0
    clipped = np.concatenate([pulp, dentin, np.full(8000, 2600.0)])          # enamel at the ceiling
    lc = scan_levels(clipped, 2600.0)
    assert lc["clip_share"] > 0.2 and np.isnan(lc["lvl_enamel"]) and abs(lc["lvl_dentin"] - 1800) < 60
    assert abs(lc["lvl_low"] - 800) < 80
    h = hist_feats("apex", rng.normal(1300, 30, 500), 800.0, 1800.0)
    assert abs(h["apex_p50"] - 0.5) < 0.03 and h["apex_iqr"] > 0 and h["apex_n"] == 500
    assert np.isnan(hist_feats("apex", np.array([1.0]), 800.0, 1800.0)["apex_p50"])
    I, L = _volume(deep_fill=True)                                           # deep filling at the ceiling
    r = tooth_row(I, L, 36, SP, float(I.max()), float(np.median(I[L == 1])), levels={"lvl_low": 1000.0, "lvl_dentin": 2000.0})
    assert r["crown_deep_dense_share"] > 0.1 and r["crown_deep_at_ceiling"] > 0.1
    plain = tooth_row(*_volume(), 36, SP, 3000.0, 900.0, levels={"lvl_low": 1000.0, "lvl_dentin": 2000.0})
    assert plain["crown_deep_dense_share"] == 0 and abs(plain["crown_deep_p50"] - 1.0) < 1e-6
    cols = feature_columns(pd.DataFrame(columns=["crown_deep_p50", "apex_n", "lvl_dentin", "clip_share", "fdi_report",
                                                 "crown_deep_dense_share", "bright_root_pulp"]))
    assert set(cols) == {"crown_deep_p50", "crown_deep_dense_share", "bright_root_pulp"}
