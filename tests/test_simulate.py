import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from partialref.propensity.simulate import SimConfig, simulate


def one_group(**kw) -> SimConfig:
    """Single finding, single centre, no house-style jitter: truth is one (p, pi)."""
    base = dict(n_findings=1, n_centres=1, centre_logit_sd=0.0,
                prevalence_range=(0.4, 0.4), propensity_range=(0.7, 0.7))
    base.update(kw)
    return SimConfig(**base)


def test_reproducible_given_seed():
    a = simulate(SimConfig(seed=1)).instances
    b = simulate(SimConfig(seed=1)).instances
    c = simulate(SimConfig(seed=2)).instances
    pd.testing.assert_frame_equal(a, b)
    assert not a["k"].equals(c["k"])


def test_shapes_and_invariants():
    cfg = SimConfig(n_cases=300, n_findings=7, seed=3, phi=0.02, rho=0.1)
    sim = simulate(cfg)
    df = sim.instances
    assert len(df) == 300 * 7
    assert df.groupby("case_id").size().eq(7).all()
    assert df["n_reports"].between(1, 3).all()
    assert (df["k"] >= 0).all() and (df["k"] <= df["n_reports"]).all()
    m = df[["m1", "m2", "m3"]].to_numpy()
    assert ((m == -1).sum(axis=1) == 3 - df["n_reports"]).all()
    assert ((m == 1).sum(axis=1) == df["k"]).all()
    # one centre and one n_reports per case
    assert df.groupby("case_id")[["centre", "n_reports"]].nunique().eq(1).all().all()
    assert len(sim.truth) == 7 * cfg.n_centres
    assert sim.truth["pi"].between(0.01, 0.99).all()


def test_truth_respects_ranges_without_jitter():
    cfg = SimConfig(centre_logit_sd=0.0, seed=4)
    t = simulate(cfg).truth
    assert t["p"].between(*cfg.prevalence_range).all()
    assert t["pi"].between(*cfg.propensity_range).all()


def test_report_count_distribution():
    df = simulate(SimConfig(n_cases=6000, n_findings=1, seed=5)).instances
    frac_multi = (df["n_reports"] >= 2).mean()
    assert abs(frac_multi - 0.60) < 0.03


def test_phi_zero_means_absent_never_mentioned():
    df = simulate(SimConfig(n_cases=500, seed=6, phi=0.0)).instances
    assert (df.loc[~df["present"], "k"] == 0).all()


def test_phi_positive_gives_false_mentions_at_rate_phi():
    df = simulate(one_group(n_cases=40000, phi=0.05, seed=7)).instances
    absent = df[~df["present"]]
    assert abs((absent["m1"] == 1).mean() - 0.05) < 0.01


@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_marginal_mention_rate_is_pi_for_any_rho(rho):
    df = simulate(one_group(n_cases=40000, rho=rho, seed=8)).instances
    present = df[df["present"]]
    assert abs((present["m1"] == 1).mean() - 0.7) < 0.015


@pytest.mark.parametrize("rho", [0.0, 0.1, 0.3])
def test_reader_correlation_equals_rho(rho):
    df = simulate(one_group(n_cases=60000, rho=rho, report_count_probs=(0.0, 1.0), seed=9)).instances
    present = df[df["present"]]
    r = np.corrcoef(present["m1"], present["m2"])[0, 1]
    assert abs(r - rho) < 0.03


def test_config_validation():
    with pytest.raises(ValidationError):
        SimConfig(report_count_probs=(0.5, 0.4))
    with pytest.raises(ValidationError):
        SimConfig(prevalence_range=(0.6, 0.1))
    with pytest.raises(ValidationError):
        SimConfig(n_centres=2, centre_probs=(1.0,))
    with pytest.raises(ValidationError):
        SimConfig(unknown_field=1)
