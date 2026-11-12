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


def test_tooth_frame_and_slices_put_the_crown_at_the_top():
    from partialref.perception.tooth_slices import SIZE, sample_plane, tooth_frame, window
    for root_up in (False, True):
        I, L = _volume(cap=True, root_up=root_up)
        mb = (L == 36) | (L == 136)
        centre, (u, v, w), _, _, contrast = tooth_frame(mb, L, SP)
        assert contrast > 0.3 and abs(np.linalg.norm(u) - 1) < 1e-9 and abs(u @ v) < 1e-9 and abs(u @ w) < 1e-9
        assert (u[2] > 0.99) != root_up                                    # u points root -> crown
        img = window(sample_plane(I, centre, u, v, SP), 0.0, 3000.0)
        assert img.shape == (SIZE, SIZE) and img.dtype == np.uint8
        assert img[: SIZE // 2].max() == 255 and img[SIZE // 2:].max() < 255  # bright cap in the upper half only


def test_embedding_alignment_and_feature_sets():
    import pytest

    from partialref.perception.tooth_classifier import align_embeddings, evaluate_embeddings
    df = pd.DataFrame({"case_id": ["P001", "P001", "F002"], "fdi_report": [36, 46, 11]})
    emb = np.arange(4 * 6, dtype=float).reshape(4, 6)                       # 3 slices x 2 values per tooth
    z = {"case_id": np.array(["F002", "P001", "P001", "S009"]), "fdi_report": np.array([11, 46, 36, 21]), "emb": emb}
    dfe, E, info = align_embeddings(df, z)                                  # shuffled order, one extra tooth
    assert list(dfe["fdi_report"]) == [36, 46, 11] and info == {"matched": 3, "dropped": 0, "dropped_by_centre": {}}
    assert E.shape == (3, 4) and list(E[0]) == [(12 + 14) / 2, (13 + 15) / 2, 16, 17]   # planes 0/1 averaged
    miss = align_embeddings(pd.concat([df, pd.DataFrame({"case_id": ["S010"], "fdi_report": [21]})]), z)[2]
    assert miss["dropped"] == 1 and miss["dropped_by_centre"] == {"S": 1}
    with pytest.raises(ValueError):
        align_embeddings(df, {**z, "case_id": np.array(["F002", "P001", "P001", "P001"]),
                              "fdi_report": np.array([11, 46, 36, 36])})
    rng = np.random.default_rng(1)
    n = 900
    df = pd.DataFrame({"case_id": [f"{'PFS'[i % 3]}{i // 6:03d}" for i in range(n)], "fdi_report": rng.choice([16, 26, 36], n)})
    for f in FEATURES:
        df[f] = rng.normal(size=n)
    for t in TYPES:
        df[t] = np.nan
    y = (rng.random(n) < 0.15).astype(float)
    df["Caries"] = y
    E = rng.normal(size=(n, 40))
    E[:, 3] += 2.5 * y                                                       # signal only in the embedding
    folds = {c: i % 5 for i, c in enumerate(sorted(df["case_id"].unique()))}
    res = evaluate_embeddings(df, E, folds, FEATURES).set_index("feature_set")
    assert set(res.index) == {"reference (linear)", "dino", "basic+histogram+dino"}
    assert res.loc["dino", "auc_features"] > 0.8 and res.loc["reference (linear)", "auc_features"] < 0.65
    assert res.loc["basic+histogram+dino", "auc_features"] > 0.7


def test_paired_bootstrap_detects_a_real_difference_and_not_a_fake_one():
    import pytest

    from partialref.perception.compare_auc import paired_bootstrap
    rng = np.random.default_rng(0)
    n = 2000
    cases = [f"P{i // 10:03d}" for i in range(n)]
    y = (rng.random(n) < 0.2).astype(float)
    good = y + rng.normal(0, 0.6, n)
    worse = y + rng.normal(0, 1.5, n)
    base = pd.DataFrame({"case_id": cases, "fdi_report": np.tile(np.arange(10), n // 10), "y": y})
    r = paired_bootstrap(base.assign(p=good), base.assign(p=worse), n_boot=200)
    assert r["d_auc"] > 0 and r["d_auc_lo"] > 0 and r["n_cases"] == 200
    equal = paired_bootstrap(base.assign(p=y + rng.normal(0, 1.0, n)), base.assign(p=y + rng.normal(0, 1.0, n)), n_boot=200)
    assert equal["d_auc_lo"] < 0 < equal["d_auc_hi"]                       # two equally good, independent scores
    with pytest.raises(ValueError):                                         # different teeth
        paired_bootstrap(base.assign(p=good), base.iloc[:100].assign(p=good[:100]), n_boot=10)
    with pytest.raises(ValueError):                                         # same teeth, different labels
        paired_bootstrap(base.assign(p=good), base.assign(p=good, y=1 - y), n_boot=10)
    with pytest.raises(ValueError):                                         # a tooth twice
        dup = pd.concat([base, base.iloc[:1]]).assign(p=0.5)
        paired_bootstrap(dup, dup, n_boot=10)


def test_saved_predictions_are_out_of_fold_and_reproduce_the_reported_auc():
    from sklearn.metrics import roc_auc_score

    from partialref.perception.tooth_classifier import evaluate_embeddings
    rng = np.random.default_rng(2)
    n = 900
    df = pd.DataFrame({"case_id": [f"{'PFS'[i % 3]}{i // 6:03d}" for i in range(n)], "fdi_report": np.tile(np.arange(11, 17), n // 6)})
    for f in FEATURES:
        df[f] = rng.normal(size=n)
    for t in TYPES:
        df[t] = np.nan
    y = (rng.random(n) < 0.2).astype(float)
    df["Caries"] = y
    df[FEATURES[0]] += 2 * y
    folds = {c: i % 5 for i, c in enumerate(sorted(df["case_id"].unique()))}
    preds: list = []
    res = evaluate(df, folds, features=FEATURES, preds=preds, set_name="basic")
    E = rng.normal(size=(n, 40))
    E[:, 0] += 2 * y
    res_e = evaluate_embeddings(df, E, folds, FEATURES, preds=preds).set_index("feature_set")
    P = pd.concat(preds, ignore_index=True)
    assert set(P["feature_set"]) == {"basic", "basic|reference", "reference (linear)", "dino", "basic+histogram+dino"}
    assert not P.duplicated(["feature_set", "finding_type", "case_id", "fdi_report"]).any()
    row = res[(res["finding_type"] == "Caries") & (res["centre"] == "all")].iloc[0]
    auc = lambda s: roc_auc_score(P.loc[P.feature_set == s, "y"], P.loc[P.feature_set == s, "p"])
    assert abs(auc("basic") - row["auc_features"]) < 1e-12 and abs(auc("basic|reference") - row["auc_reference"]) < 1e-12
    assert abs(auc("dino") - res_e.loc["dino", "auc_features"]) < 1e-12
    lab = P[P.feature_set == "basic"].merge(df[["case_id", "fdi_report", "Caries"]], on=["case_id", "fdi_report"])
    assert len(lab) == n and (lab["y"] == lab["Caries"]).all()             # each prediction sits on its own tooth


def test_human_reference_pairs_reports_and_compares_on_the_same_teeth():
    from partialref.perception.human_reference import compare, pair_table
    teeth = pd.DataFrame({"case_id": ["P001"] * 3 + ["P002"] * 2 + ["P003"], "fdi_report": [16, 26, 36, 16, 46, 11]})
    reps = pd.DataFrame({"case_id": ["P001", "P001", "P002", "P002", "P003"], "report_id": ["r1", "r2", "r3", "r4", "r5"],
                         "report_x": [1, 2, 1, 14, 1]})
    inst = pd.DataFrame({"case_id": ["P001", "P001", "P001", "P002"], "report_id": ["r1", "r1", "r2", "r3"],
                         "finding_type": ["Caries", "Caries", "Caries", "Caries"], "slot": ["16", "36", "16", "46"]})
    silent = {("P", "14", "Caries")}                                         # author 14 never reports caries
    pt = pair_table(teeth, reps, inst, silent)
    c = pt[pt["finding_type"] == "Caries"].set_index(["rep_a", "rep_b", "fdi_report"])
    assert len(c) == 6                                                        # P001: 3 teeth x 2 directions; P002 silent
    assert c.loc[("r1", "r2", 36), "a"] == 1 and c.loc[("r1", "r2", 36), "b"] == 0
    assert c.loc[("r2", "r1", 16), ["a", "b"]].tolist() == [1, 1]
    assert "P003" not in set(pt["case_id"])                                   # single report: no pair
    assert "P002" not in set(pt.loc[pt["finding_type"] == "Caries", "case_id"])
    assert len(pt[pt["finding_type"] == "ProstheticDevice"]) == 10             # P002 is silent on caries only
    b = np.array([1, 1, 0, 0, 0, 0])
    r = compare(np.array([1, 0, 1, 0, 0, 0]), b, np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4]))
    assert r["sens_human"] == 0.5 and r["fpr_human"] == 0.25 and r["auc_model"] == 1.0 and r["sens_model_at_fpr_human"] == 1.0


def test_human_reference_weights_three_reports_and_interpolates():
    from partialref.perception.human_reference import bootstrap, compare, pair_table
    teeth = pd.DataFrame({"case_id": ["P001", "P002"], "fdi_report": [16, 16]})
    reps = pd.DataFrame({"case_id": ["P001"] * 3 + ["P002"] * 2, "report_id": ["r1", "r2", "r3", "r4", "r5"],
                         "report_x": [1, 2, 3, 1, 14]})
    inst = pd.DataFrame({"case_id": ["P001"], "report_id": ["r1"], "finding_type": ["Caries"], "slot": ["16"]})
    pt = pair_table(teeth, reps, inst, {("P", "14", "Caries")})
    c = pt[pt["finding_type"] == "Caries"]
    assert len(c) == 6 and np.allclose(c["w"], 1 / 6) and set(c["case_id"]) == {"P001"}   # P002: one side silent
    assert not c["same_author"].any()
    b = np.array([1, 1, 0, 0, 0, 0])
    r = compare(np.array([1, 0, 0, 0, 0, 0]), b, np.array([0.9, 0.3, 0.5, 0.2, 0.1, 0.05]))
    assert r["fpr_human"] == 0 and r["sens_model_at_fpr_human"] == 0.5 and r["d_sens"] == 0     # between ROC points
    rt = compare(np.array([1, 0, 0, 0, 0, 0]), b, np.full(6, 0.5))                              # all tied
    assert rt["auc_model"] == 0.5 and rt["sens_model_at_fpr_human"] == 0
    m = pd.DataFrame({"case_id": [f"P{i // 2:03d}" for i in range(40)], "a": np.tile([1, 0], 20), "b": np.tile([1, 0], 20),
                      "p": np.tile([0.9, 0.1], 20), "w": 1.0})
    bs = bootstrap(m, n_boot=50, seed=0)
    assert bs["n_boot_used"] == 50 and bs["d_sens"] == 0 and bs["d_sens_lo"] == bs["d_sens_hi"] == 0


def test_root_canal_filling_is_found_when_one_metal_voxel_sets_the_scan_maximum():
    I, L = _volume(canal_fill=True)                                          # filling 3000, dentin 2000, pulp 1000
    I[0, 0, 0] = 30000.0                                                     # one extreme voxel elsewhere (no clipping)
    lv = {"lvl_low": 1000.0, "lvl_dentin": 2000.0}
    r = tooth_row(I, L, 36, SP, float(I.max()), float(np.median(I[L == 1])), levels=lv)
    assert r["bright_root_pulp"] == 0                                        # the old max-relative rule misses it
    assert r["root_pulp_dense2"] > 0.5 and r["root_pulp_p50"] > 1.9 and r["root_dense2"] > 0
    plain = tooth_row(*_volume(), 36, SP, 30000.0, 900.0, levels=lv)
    assert plain["root_pulp_dense2"] == 0 and plain["crown_deep_dense2"] == 0 and plain["root_dense3"] == 0
    clipped, Lc = _volume(canal_fill=True)                                   # clipping scanner: filling at the ceiling
    rc = tooth_row(clipped, Lc, 36, SP, 3000.0, 900.0, levels={"lvl_low": 1000.0, "lvl_dentin": 2400.0})
    assert rc["root_pulp_dense3"] > 0.5                                      # z < 3 but at the ceiling
