"""V2V2R mask conversion (synthetic arrays only)."""

import numpy as np
import pytest

from partialref.baseline.v2v2r_masks import convert_labels, dice, to_tf4_grid


def test_labels_map_to_v2v2r_scheme_with_and_without_mirror():
    L = np.array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 26, 36, 46, 116, 146, 103, 104, 105]]], np.int16)
    plain = convert_labels(L, mirror=False).ravel().tolist()
    assert plain == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 26, 36, 46, 50, 50, 51, 52, 53]
    mir = convert_labels(L, mirror=True).ravel().tolist()
    assert mir == [0, 1, 2, 4, 3, 6, 5, 7, 8, 9, 10, 26, 16, 46, 36, 50, 50, 52, 51, 53]
    with pytest.raises(ValueError):
        convert_labels(np.array([[[0, 77]]], np.int16), mirror=False)
    assert convert_labels(np.array([[[100, 7]]], np.int16), mirror=False).tolist() == [[[0, 7]]]   # TF3 label 100


def test_grid_flip_and_dice():
    a = np.arange(24).reshape(2, 3, 4)
    f = to_tf4_grid(a)
    assert f[0, 0, 0] == a[0, 2, 3] and np.array_equal(to_tf4_grid(f), a)
    x = np.zeros((4, 4, 4), np.uint8)
    y = x.copy()
    x[:2] = 36
    y[:1] = 36
    y[3] = 46
    d = dice(x, y)
    assert abs(d[36] - 2 * 16 / (32 + 16)) < 1e-12 and d[46] == 0


def _pred_json(d, case, teeth):
    """teeth: {fdi: (detected, fillings_bool)}"""
    out = {"case_id": case, "global": {}, "coverage": {}, "teeth": {}}
    for fdi, (det, fill) in teeth.items():
        k = f"tooth_{fdi}"
        t = {"detected": det}
        if det == "yes":
            t[f"{k}_morphology"] = {"with_fillings": fill, "with_caries": False, "with_endo": False,
                                    "with_full_crown": False}
            t[f"{k}_endodontic_treatment"] = {"periapical_lesion": False}
        out["teeth"][k] = t
    p = d / case / "manual" / "vqa" / "predictions"
    p.mkdir(parents=True)
    (p / f"{case}_pred.json").write_text(__import__("json").dumps(out))


def test_vlm_vs_module_join_and_score(tmp_path):
    import pandas as pd

    from partialref.baseline.v2v2r_vs_module import join, main, vlm_table

    # two cases; VLM says fillings yes on 11 (true pos) and 21 (false pos); 12 positive but not detected
    _pred_json(tmp_path, "C1", {11: ("yes", True), 12: ("no_image", None), 13: ("yes", False)})
    _pred_json(tmp_path, "C2", {21: ("yes", True), 22: ("yes", False), 23: ("yes", False)})
    V = vlm_table(tmp_path, "manual")
    assert len(V) == 6 * 5 and V["answered"].sum() == 5 * 5
    P = pd.DataFrame({"case_id": ["C1", "C1", "C1", "C2", "C2", "C2", "C3"],
                      "fdi_report": [11, 12, 13, 21, 22, 23, 31],
                      "finding_type": "ConservativeRestoration", "feature_set": "fs",
                      "y": [1, 1, 0, 0, 1, 0, 1], "p": [0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.5]})
    m = join(P, V)
    assert set(m["case_id"]) == {"C1", "C2"}                      # C3 has no VLM run
    assert m.set_index("fdi_report").loc[12, ["a", "answered"]].tolist() == [0, False]
    # VLM: sens 1/3, fpr 1/3; module ranks all positives above all negatives -> sens 1 at fpr 1/3
    run = tmp_path / "clf"
    run.mkdir()
    P.to_parquet(run / "predictions.parquet")
    main(["--vlm-run", str(tmp_path), "--classifier-run", str(run), "--feature-set", "fs", "--n-boot", "50"])
    r = pd.read_csv(tmp_path / "vs_module-fs-manual" / "vlm_vs_module.csv").iloc[0]
    assert r["n_teeth"] == 6 and r["n_unanswered"] == 1 and r["n_vlm_yes"] == 2
    assert abs(r["sens_vlm"] - 1 / 3) < 1e-9 and abs(r["fpr_vlm"] - 1 / 3) < 1e-9
    assert abs(r["sens_model_at_fpr_vlm"] - 1) < 1e-9 and abs(r["auc_model"] - 1) < 1e-9


def test_vlm_table_post_bridge_string_malformed(tmp_path):
    import json

    from partialref.baseline.v2v2r_vs_module import vlm_table

    def morph(**kw):
        base = {"with_fillings": False, "with_caries": False, "with_endo": False, "with_full_crown": False,
                "with_post_and_core": False}
        return {**base, **kw}
    teeth = {
        "tooth_11": {"detected": "yes", "tooth_11_morphology": morph(with_post_and_core=True),
                     "tooth_11_endodontic_treatment": {"periapical_lesion": "true"}},
        "tooth_12": {"detected": "yes", "tooth_12_morphology": morph(),
                     "tooth_12_endodontic_treatment": {"periapical_lesion": False}},     # bridge abutment
        "tooth_13": {"detected": "yes", "tooth_13_morphology": {}},                     # failed call
        "tooth_14": {"detected": "no"},
    }
    g = {"fixed_bridges_maxilla": {"present": True, "bridges": [{"abutment_teeth": [12, 14], "pontic_teeth": []}]}}
    p = tmp_path / "C1" / "manual" / "vqa" / "predictions"
    p.mkdir(parents=True)
    (p / "C1_pred.json").write_text(json.dumps({"case_id": "C1", "global": g, "teeth": teeth}))
    V = vlm_table(tmp_path, "manual").set_index(["fdi_report", "finding_type"])
    assert V.loc[(11, "ProstheticDevice"), "a"] == 1 and V.loc[(11, "EndodonticTreatment"), "a"] == 1
    assert V.loc[(11, "EndodonticTreatment"), "from_post"]
    assert V.loc[(11, "PeriapicalLesion"), "a"] == 1                      # string "true"
    assert V.loc[(12, "ProstheticDevice"), "a"] == 1                      # bridge abutment
    assert V.loc[(14, "ProstheticDevice"), "a"] == 0                      # not detected -> no answer
    assert not V.loc[(14, "ProstheticDevice"), "malformed"] and not V.loc[(14, "ProstheticDevice"), "answered"]
    assert V.loc[(13, "Caries"), "malformed"] and V.loc[(13, "Caries"), "a"] == 0
    assert V.loc[(13, "PeriapicalLesion"), "malformed"]                   # question missing


def test_vlm_vs_module_periapical_split_and_malformed_stop(tmp_path):
    import pandas as pd
    import pytest

    from partialref.baseline.v2v2r_vs_module import main

    _pred_json(tmp_path, "C1", {11: ("yes", True), 12: ("yes", False), 13: ("yes", False), 14: ("yes", False)})
    rows = []
    for f, endo, pa, p in [(11, 1, 1, 0.9), (12, 1, 0, 0.2), (13, 0, 1, 0.8), (14, 0, 0, 0.1)]:
        rows += [("EndodonticTreatment", f, endo, 0.5), ("PeriapicalLesion", f, pa, p)]
    P = pd.DataFrame([{"case_id": "C1", "fdi_report": f, "finding_type": t, "feature_set": "fs", "y": y, "p": p}
                      for t, f, y, p in rows])
    run = tmp_path / "clf"
    run.mkdir()
    P.to_parquet(run / "predictions.parquet")
    main(["--vlm-run", str(tmp_path), "--classifier-run", str(run), "--feature-set", "fs", "--n-boot", "20"])
    r = pd.read_csv(tmp_path / "vs_module-fs-manual" / "vlm_vs_module.csv").set_index("finding_type")
    assert {"PeriapicalLesion", "PeriapicalLesion|endo=0", "PeriapicalLesion|endo=1"} <= set(r.index)
    assert r.loc["PeriapicalLesion|endo=1", "n_teeth"] == 2
    # a detected tooth without answers -> malformed share 1/5 rows > 2 % -> stop
    q = tmp_path / "C2" / "manual" / "vqa" / "predictions"
    q.mkdir(parents=True)
    (q / "C2_pred.json").write_text('{"case_id": "C2", "global": {}, "teeth": {"tooth_21": {"detected": "yes"}}}')
    P2 = pd.concat([P, pd.DataFrame([{"case_id": "C2", "fdi_report": 21, "finding_type": "Caries",
                                      "feature_set": "fs", "y": 0, "p": 0.1}])])
    P2.to_parquet(run / "predictions.parquet")
    with pytest.raises(SystemExit):
        main(["--vlm-run", str(tmp_path), "--classifier-run", str(run), "--feature-set", "fs", "--n-boot", "20"])
