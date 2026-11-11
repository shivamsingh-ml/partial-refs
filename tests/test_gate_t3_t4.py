"""GT/T3 (mask agreement) and T4 (effect size): synthetic data only."""

import json

import numpy as np
import pandas as pd

from partialref.metrics.corrected import (
    corrected_precision,
    expected_precisions,
    p_present_given_unmentioned,
)
from partialref.propensity.effect_size import by_type, systems
from partialref.propensity.effect_size import main as effect_main
from partialref.propensity.mask_agreement import (
    case_level_instances,
    compare,
    fixed_prosthesis_mentions,
    gap_mentions,
    tooth_absence_instances,
)


def _instances(n_cases, per_case, p, pi_of_true, phi, seed):
    """Two readers per case; pi_of_true(rng, size) gives each true instance its own mention probability."""
    rng = np.random.default_rng(seed)
    m = n_cases * per_case
    truth = rng.random(m) < p
    pi = np.where(truth, pi_of_true(rng, m), phi)
    k = (rng.random((m, 2)) < pi[:, None]).sum(axis=1)
    return pd.DataFrame({"case_id": np.repeat([f"P{i:04d}" for i in range(n_cases)], per_case), "n": 2, "k": k,
                         "truth": truth})


def test_t3_recovers_pi_and_phi_and_agrees_when_disagreement_is_omission():
    inst = _instances(400, 20, 0.3, lambda r, m: np.full(m, 0.6), 0.0, seed=1)
    r = compare(inst, n_boot=200, seed=0)
    assert abs(r["pi_mask"] - 0.6) < 0.02 and r["phi_mask"] == 0 and abs(r["p_mask"] - 0.3) < 0.02
    # no bias between the two estimates (checked over 30 seeds: mean difference -0.003, sd 0.009, the
    # difference interval covered 0 in 28 of 30), so here only a tolerance and the interval overlap
    assert abs(r["pi_readers"] - 0.6) < 0.04 and r["ci_overlap"] and abs(r["diff_readers_minus_mask"]) < 0.04
    assert r["pi_mask_lo"] < r["pi_mask"] < r["pi_mask_hi"] and r["n_cases"] == 400 and r["n_instances"] == 8000
    assert r["reader_slots_mask_positive"] == 2 * r["n_mask_positive"]
    r2 = compare(_instances(400, 20, 0.3, lambda r, m: np.full(m, 0.6), 0.03, seed=2), n_boot=100)
    assert abs(r2["phi_mask"] - 0.03) < 0.006 and r2["phi_mask_lo"] < 0.03 < r2["phi_mask_hi"]


def test_t3_varying_pi_is_still_omission_and_makes_readers_higher():
    """Pure omission (phi = 0) with pi varying between findings: the two-reader fit estimates
    E[pi^2]/E[pi] > E[pi], so it is read as 'pi varies; upper bound', not as false mentions."""
    inst = _instances(400, 20, 0.3, lambda r, m: np.where(r.random(m) < 0.5, 0.9, 0.0), 0.0, seed=3)
    r = compare(inst, n_boot=200)
    assert abs(r["pi_mask"] - 0.45) < 0.03 and r["pi_readers"] > 0.85 and r["phi_mask"] == 0
    assert not r["ci_overlap"] and r["diff_lo"] > 0.3 and r["readers_vs_mask"].startswith("readers higher")
    mild = _instances(400, 20, 0.3, lambda r, m: r.uniform(0.4, 0.8, m), 0.0, seed=4)   # E[pi^2]/E[pi] = 0.622
    r2 = compare(mild, n_boot=200)
    assert abs(r2["pi_mask"] - 0.6) < 0.02 and abs(r2["pi_readers"] - 0.622) < 0.03


def test_t3_instance_builders():
    vis = {c: {"upper": True, "lower": True, "teeth": {"16", "36", "46"}, "root_cut": set(),
               "quadrant_edge": {q: False for q in "1234"}} for c in ("P001", "P002", "P003")}
    reps = pd.DataFrame({"case_id": ["P001", "P001", "P002", "P003", "P003"], "report_id": list("abcde")})
    ment = pd.DataFrame({"case_id": ["P001", "P001", "P001", "P001", "P002", "P003"],
                         "report_id": ["a", "b", "a", "a", "c", "d"],
                         "finding_type": ["ToothAbsence", "ToothAbsence", "ToothAbsence", "DentalImplant",
                                          "DentalImplant", "ProstheticDevice"],
                         "slot": ["17", "17", "36", "26", "11", "45"]})
    ta = tooth_absence_instances(reps, ment, vis, {("P003", "e")}, "omission")
    assert set(ta["case_id"]) == {"P001", "P002"} and len(ta) == 64             # P003 has a failed report
    row = lambda c, s: ta[(ta.case_id == c) & (ta.slot == s)].iloc[0]
    assert row("P001", "17")[["n", "k", "truth"]].tolist() == [2, 2, True]      # absent in mask, both mention
    assert row("P001", "36")[["n", "k", "truth"]].tolist() == [2, 1, False]     # present in mask: false mention
    assert row("P002", "17")[["n", "k", "truth"]].tolist() == [1, 0, True]
    assert row("P001", "26")[["n", "k", "truth"]].tolist() == [2, 0, True]      # implant site, ToothAbsence coding
    gap = tooth_absence_instances(reps, gap_mentions(ment), vis, {("P003", "e")}, "omission")
    grow = lambda c, s: gap[(gap.case_id == c) & (gap.slot == s)].iloc[0]
    assert grow("P001", "26")[["k", "truth"]].tolist() == [1, True]               # implant counts as "no tooth"
    assert grow("P002", "11")[["k", "truth"]].tolist() == [1, True]
    assert grow("P001", "17")["k"] == 2 and len(gap) == len(ta)
    cf = pd.DataFrame({"case_id": ["P001", "P002", "P003", "P004"], "label_found": [True, True, True, False],
                       "implant_mm3": [40.0, 0.0, 5.0, 99.0], "crown_mm3": [0.0, 0.0, 3.0, 0.0],
                       "bridge_mm3": [0.0, np.nan, 4.0, 0.0]})
    imp = case_level_instances(reps, ment, cf, set(), "DentalImplant", ["implant"], 0.0).set_index("case_id")
    assert imp.loc["P001", ["n", "k", "truth"]].tolist() == [2, 1, True]
    assert imp.loc["P002", ["n", "k", "truth"]].tolist() == [1, 1, False]       # mention without mask label
    assert imp.loc["P003", ["n", "k", "truth"]].tolist() == [2, 0, True] and "P004" not in imp.index
    imp10 = case_level_instances(reps, ment, cf, set(), "DentalImplant", ["implant"], 10.0).set_index("case_id")
    assert not imp10.loc["P003", "truth"] and imp10.loc["P001", "truth"]        # 5 mm3 is below the threshold
    pro = case_level_instances(reps, ment, cf, {("P001", "a")}, "ProstheticDevice", ["crown", "bridge"], 0.0)
    assert set(pro["case_id"]) == {"P003"}             # P001: failed report; P002: bridge volume missing
    assert pro.set_index("case_id").loc["P003", ["k", "truth"]].tolist() == [1, True]
    # shape mismatch: never counted as mask-negative
    bad_cf = cf.assign(shape_mismatch=[False, True, False, False])
    assert "P002" not in set(case_level_instances(reps, ment, bad_cf, set(), "DentalImplant", ["implant"], 0.0)["case_id"])
    # single mask component: P001 has 3 implants in the mask -> dropped; mask-negative P002 kept
    one = cf.assign(implant_ncomp=[3, 0, 1, 1])
    single = case_level_instances(reps, ment, one, set(), "DentalImplant", ["implant"], 0.0, single_component=True)
    assert set(single["case_id"]) == {"P002", "P003"}


def test_t3_fixed_prostheses_only(tmp_path):
    (tmp_path / "per_report").mkdir()
    files = {"P001__a__en.json": [{"finding_type": "ProstheticDevice", "tooth_ids": ["16"],
                                   "qualifiers": {"prosthesis_type": ["Crown"]}}],
             "P001__b__en.json": [{"finding_type": "ProstheticDevice", "tooth_ids": ["16"],
                                   "qualifiers": {"prosthesis_type": ["PostAndCore"]}}],
             "P002__c__en.json": [{"finding_type": "ProstheticDevice", "tooth_ids": ["44", "45", "46"],
                                   "qualifiers": {"prosthesis_type": ["FixedBridge"]}},
                                  {"finding_type": "ProstheticDevice", "tooth_ids": ["44"],
                                   "qualifiers": {"prosthesis_type": ["Crown"]}}],
             "P002__c__it.json": [{"finding_type": "ProstheticDevice", "tooth_ids": ["11"],
                                   "qualifiers": {"prosthesis_type": ["Crown"]}}],
             "P003__d__en.json": None}
    for name, f in files.items():
        (tmp_path / "per_report" / name).write_text(json.dumps({"extraction": None if f is None else {"findings": f}}))
    m = fixed_prosthesis_mentions(tmp_path, "en")
    assert list(map(tuple, m[["case_id", "report_id"]].to_numpy())) == [("P001", "a"), ("P002", "c")]


def test_t4_expected_values_match_a_simulated_scoring():
    p, pi, se, tp = 0.3, 0.5, 0.7, 0.8
    fpr = p * se * (1 - tp) / (tp * (1 - p))                 # false-positive rate that gives true precision tp
    rng = np.random.default_rng(0)
    m = 600_000
    present = rng.random(m) < p
    pred = rng.random(m) < np.where(present, se, fpr)
    for n in (1, 2):
        mentioned = present & (rng.random((m, n)) < pi).any(axis=1)
        e = expected_precisions(p, pi, n, tp)
        q0 = float(p_present_given_unmentioned(p, pi, n))
        raw = (pred & mentioned).sum() / pred.sum()
        corr = corrected_precision((pred & mentioned).sum(), q0 * (pred & ~mentioned).sum(), pred.sum())
        assert abs((pred & present).sum() / pred.sum() - tp) < 0.005
        assert abs(raw - e["raw"]) < 0.005 and abs(corr - e["corrected"]) < 0.005
        assert e["raw"] < e["corrected"] < tp                 # skilled system: corrected is still too low
        assert abs(e["raw"] - tp * (1 - (1 - pi) ** n)) < 1e-12
    noskill = expected_precisions(p, pi, 1, p)               # predictions independent of presence
    assert abs(noskill["corrected"] - p) < 1e-12              # the estimator is exact without skill
    assert p_present_given_unmentioned(0.3, 1.0, 1) == 0 and abs(p_present_given_unmentioned(0.3, 0.0, 2) - 0.3) < 1e-12
    assert np.isnan(corrected_precision(0, 0, 0))


def _est():
    return pd.DataFrame({"finding_type": ["Canal", "Restoration", "Caries"], "p_mle": [0.9, 0.2, 0.05],
                         "pi_mle": [0.9, 0.5, 0.25], "pi_lo": [0.85, 0.4, 0.15], "pi_hi": [0.95, 0.6, 0.35],
                         "n_instances": [700, 9000, 9000], "supported": [True, True, True],
                         "group": "pooled_double_read"})


def test_t4_tables_and_systems(tmp_path):
    est = _est()
    bt = by_type(est, [0.8], [1, 2])
    one = bt[bt.n_ref == 1].set_index("finding_type")
    assert abs(one.loc["Caries", "raw_precision"] - 0.8 * 0.25) < 1e-12
    assert one["raw_minus_true"].is_monotonic_decreasing               # lower pi -> raw precision further off
    # the §2.2 correction credits each unmatched prediction with q0, so it is small for rare findings even
    # when pi is low (Caries here) -- the estimator, not the size of the bias, limits it (HANDOFF L1)
    q0 = p_present_given_unmentioned(one["p"], one["pi"], 1)
    assert np.allclose(one["corrected_minus_raw"], q0 * (1 - one["raw_precision"]))
    assert one.loc["Caries", "corrected_minus_raw"] < 0.05 < -one.loc["Caries", "raw_minus_true"]
    assert (one["corrected_minus_raw_lo"] <= one["corrected_minus_raw"]).all()
    assert (one["corrected_minus_raw"] <= one["corrected_minus_raw_hi"]).all()
    two = bt[bt.n_ref == 2].set_index("finding_type")
    assert (two["raw_precision"] > one["raw_precision"]).all()         # a second report closes part of the gap
    s = systems(est, 0.8, 0.7, 1, 0.8).set_index("system")
    assert s.loc["conservative", "types_reported"] == 1 and s.loc["conservative", "types_silenced"] == "Restoration,Caries"
    assert s.loc["conservative", "raw_precision"] > s.loc["complete", "raw_precision"] + 0.1   # references alone
    assert s.loc["complete", "true_recall"] > s.loc["conservative", "true_recall"]
    assert abs(s.loc["complete", "true_recall"] - 0.7) < 1e-12
    gap_raw = s.loc["conservative", "raw_precision"] - s.loc["complete", "raw_precision"]
    gap_cor = s.loc["conservative", "corrected_precision"] - s.loc["complete", "corrected_precision"]
    # both systems have true precision 0.8, so the true gap is 0. The §2.2 estimator does NOT remove the
    # artefact: it credits unmatched predictions of the frequent type (high p) far more than those of rare
    # types, so the conservative system still looks better (HANDOFF L1: exact only without skill).
    assert gap_raw > 0.1 and gap_cor > 0.1
    run = tmp_path / "t2"
    run.mkdir()
    pd.concat([est, est.assign(group="pooled_all_cases"),
               est.iloc[:1].assign(finding_type="Rare", supported=False)]).to_csv(run / "propensity_by_type.csv", index=False)
    assert effect_main(["--propensity-run", str(run), "--results-dir", str(tmp_path / "res")]) == 0
    out = next((tmp_path / "res" / "propensity").iterdir())
    summ = json.loads((out / "summary.json").read_text())
    assert summ["n_types"] == 3 and summ["types_with_true_minus_raw_ge_0_05"] == ["Canal", "Restoration", "Caries"]
    assert summ["types_with_corrected_minus_raw_ge_0_05"] == ["Canal", "Restoration"]     # §2.2 misses Caries
    assert summ["raw_precision_conservative_minus_complete"] > 0
    assert len(pd.read_csv(out / "effect_by_type.csv")) == 3 * 2 * 3 and (out / "provenance.json").exists()
