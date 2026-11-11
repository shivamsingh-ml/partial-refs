"""Who reports what (synthetic data only)."""

import numpy as np
import pandas as pd

from partialref.propensity.readers import (
    double_vs_single,
    pair_instances,
    pair_tests,
    profiles,
    stability,
    wilson,
)


def _synthetic(seed=0, n_cases=200, a="9", b="14"):
    """Centre P, two authors per case (ids of mixed digit length). Author b never reports fillings; both
    report the canal on both sides; caries reported by each with probability 0.5. Plus 50 single-read cases
    by author a (with 1 canal finding each) and one 3-report case."""
    rng = np.random.default_rng(seed)
    reps, inst = [], []
    for i in range(n_cases):
        c = f"P{i:04d}"
        for x in (a, b):
            rid = f"{x}_{i}"
            reps.append((c, rid, x))
            for side in ("Left", "Right"):
                inst.append((c, rid, "MandibularCanalCourse", side))
            if x == a:
                for t in ("16", "26"):
                    inst.append((c, rid, "ConservativeRestoration", t))
            if rng.random() < 0.5:
                inst.append((c, rid, "Caries", "36"))
    for i in range(n_cases, n_cases + 50):
        c, rid = f"P{i:04d}", f"{a}_{i}"
        reps.append((c, rid, a))
        inst.append((c, rid, "MandibularCanalCourse", "Left"))
    for x in (a, b, "30"):                                            # a 3-report case: not paired
        reps.append(("P8888", f"{x}_8888", x))
        inst.append(("P8888", f"{x}_8888", "MandibularCanalCourse", "Left"))
    reps = pd.DataFrame(reps, columns=["case_id", "report_id", "report_x"])
    inst = pd.DataFrame(inst, columns=["case_id", "report_id", "finding_type", "slot"])
    return reps, inst


def test_profiles_find_the_silent_author_and_the_exchangeable_types():
    reps, inst = _synthetic()
    pi = pair_instances(reps, inst)
    assert pi["case_id"].nunique() == 200 and set(pi["author"]) == {"9", "14"} and "P8888" not in set(pi["case_id"])
    prof = profiles(pi, reps, inst, n_boot=300).set_index(["author", "finding_type"])
    f14 = prof.loc[("14", "ConservativeRestoration")]
    assert f14["both"] == 0 and f14["only_partner"] == 400 and f14["rate_when_partner_mentions"] == 0
    assert f14["rate_hi"] == 0 and f14["share_reports_mentioning"] == 0 and f14["n_paired_cases_any_mention"] == 200
    assert np.isnan(prof.loc[("9", "ConservativeRestoration"), "rate_when_partner_mentions"])   # partner never mentions
    canal = prof.loc[("14", "MandibularCanalCourse")]
    assert canal["rate_when_partner_mentions"] == 1 and canal["only_me"] == 0
    car = prof.loc[("9", "Caries")]
    assert 0.35 < car["rate_when_partner_mentions"] < 0.65 and car["rate_lo"] < car["rate_when_partner_mentions"] < car["rate_hi"]
    tests = pair_tests(pi).set_index("finding_type")
    assert len(tests) == 3 and set(tests["author_a"]) == {"14"}             # each unordered pair once
    fill = tests.loc["ConservativeRestoration"]
    assert fill["cases_b_more"] == 200 and fill["cases_a_more"] == 0 and fill["p_sign_cases"] < 1e-10
    assert tests.loc["Caries", "p_sign_cases"] > 0.01                        # exchangeable


def test_bootstrap_interval_is_wider_than_instance_count_suggests():
    """Clustered instances: whole cases either all-mentioned or not -> case bootstrap wider than Wilson."""
    rows = []
    for i in range(60):
        hit = i % 2 == 0
        for t in [f"1{k}" for k in range(1, 9)]:
            rows.append((f"P{i:03d}", "P", "9", "14", "Caries", t, hit, True))
    pi = pd.DataFrame(rows, columns=["case_id", "centre", "author", "partner", "finding_type", "slot", "mine", "theirs"])
    reps = pd.DataFrame({"case_id": [f"P{i:03d}" for i in range(60)], "report_id": "x", "report_x": "9"})
    prof = profiles(pi, reps, pd.DataFrame(columns=["case_id", "report_id", "finding_type", "slot"]), n_boot=500).iloc[0]
    w_lo, w_hi = wilson(240, 480)
    assert prof["rate_hi"] - prof["rate_lo"] > 2 * (w_hi - w_lo)


def test_pairs_need_two_different_authors_and_stability_replicates():
    reps, inst = _synthetic()
    same = pd.DataFrame([("P9999", "9_a", "9"), ("P9999", "9_b", "9")], columns=reps.columns)
    extra = pd.DataFrame([("P9999", "9_a", "Caries", "36"), ("P9999", "9_b", "Caries", "46")], columns=inst.columns)
    pi = pair_instances(pd.concat([reps, same]), pd.concat([inst, extra]))
    assert "P9999" not in set(pi["case_id"])                                  # has findings, still not paired
    st = stability(pi, 10).set_index(["author", "finding_type"])
    row = st.loc[("14", "ConservativeRestoration")]
    assert row["rate_A"] == 0 and row["rate_B"] == 0 and row["both_halves_ge_min_n"]
    assert 0 < st.loc[("9", "Caries"), "n_A"] < 200


def test_double_vs_single_within_author_and_wilson():
    reps, inst = _synthetic()
    vis = pd.DataFrame({"case_id": reps["case_id"].unique()})
    vis["n_upper_teeth"] = np.where(vis.case_id.str[1:].astype(int) < 200, 10, 14)
    vis["n_lower_teeth"] = 12
    vis = vis[vis.case_id != "P0000"]                                         # one case without mask features
    d = double_vs_single(reps, inst, vis, ["MandibularCanalCourse", "ConservativeRestoration"])
    a9 = d[(d.level == "author") & (d.author == "9")].set_index("variable")
    assert a9.loc["n_findings", "n_double"] == 201 and a9.loc["n_findings", "n_single"] == 50   # incl. the 3-report case
    assert a9.loc["n_findings", "mean_single"] == 1 and a9.loc["n_findings", "mean_double"] > 3.9
    assert a9.loc["mentions_MandibularCanalCourse", "mean_single"] == 1
    assert not (d.author == "14").any()                                       # author 14 has no single-read reports
    case = d[d.level == "case"].set_index("variable").loc["mask_teeth"]
    assert case["n_double"] == 199 + 1 and case["n_single"] == 50             # missing mask features not counted
    assert case["mean_single"] == 26
    lo, hi = wilson(0, 400)
    assert lo == 0 and hi < 0.01 and all(np.isnan(wilson(0, 0)))
