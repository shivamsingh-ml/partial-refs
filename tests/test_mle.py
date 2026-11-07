import numpy as np
import pytest

from partialref.propensity.closed_form import closed_form_n2
from partialref.propensity.mle import fit_by_group, fit_mle, sufficient_stats
from partialref.propensity.simulate import SimConfig, simulate

from conftest import expected_counts, stack


def one_group(**kw) -> SimConfig:
    base = dict(n_findings=1, n_centres=1, centre_logit_sd=0.0,
                prevalence_range=(0.3, 0.3), propensity_range=(0.7, 0.7))
    base.update(kw)
    return SimConfig(**base)


def test_sufficient_stats_collapse():
    n, k, w = sufficient_stats([1, 2, 2, 2, 3], [0, 1, 1, 2, 0])
    assert dict(zip(zip(n, k), w)) == {(1, 0): 1, (2, 1): 2, (2, 2): 1, (3, 0): 1}


@pytest.mark.parametrize("p,pi", [(0.3, 0.7), (0.05, 0.4), (0.6, 0.95)])
def test_exact_on_expected_counts_mixed_report_numbers(p, pi):
    data = stack(*(expected_counts(p, pi, n=m, total=t) for m, t in [(1, 400), (2, 450), (3, 150)]))
    res = fit_mle(*data)
    assert res.converged and res.identifiable
    assert res.pi == pytest.approx(pi, abs=1e-4)
    assert res.p == pytest.approx(p, abs=1e-4)
    assert res.phi == 0.0


def test_equals_closed_form_on_two_report_data():
    df = simulate(one_group(n_cases=3000, report_count_probs=(0.0, 1.0), seed=21)).instances
    cf = closed_form_n2(df["n_reports"], df["k"])
    assert 0 < cf.p < 1
    res = fit_mle(df["n_reports"], df["k"])
    assert res.pi == pytest.approx(cf.pi, abs=1e-4)
    assert res.p == pytest.approx(cf.p, abs=1e-4)


def test_recovers_on_simulated_data_rho0_phi0():
    df = simulate(one_group(n_cases=30000, seed=22)).instances
    res = fit_mle(df["n_reports"], df["k"])
    assert res.pi == pytest.approx(0.7, abs=0.02)
    assert res.p == pytest.approx(0.3, abs=0.02)


def test_single_report_only_is_not_identifiable():
    res = fit_mle([1, 1, 1, 1], [1, 0, 0, 1])
    assert not res.identifiable


def test_reader_correlation_inflates_pi():
    """HANDOFF §2.1 known bias: shared omissions make pi look too high (and p too low)."""
    p, pi, rho = 0.3, 0.7, 0.3
    exact = fit_mle(*expected_counts(p, pi, n=2, rho=rho))
    assert exact.pi > pi + 0.05
    assert exact.p < p
    df = simulate(one_group(n_cases=30000, rho=rho, seed=23)).instances
    sim = fit_mle(df["n_reports"], df["k"])
    assert sim.pi > pi + 0.03


def test_false_positives_bias_pi_down_with_two_reports():
    p, pi, phi = 0.3, 0.7, 0.03
    res = fit_mle(*expected_counts(p, pi, n=2, phi=phi))  # model assumes phi = 0
    assert res.pi < pi - 0.03
    assert res.p > p


def test_fit_phi_recovers_with_three_reports():
    p, pi, phi = 0.3, 0.7, 0.03
    res = fit_mle(*expected_counts(p, pi, n=3, phi=phi), fit_phi=True)
    assert res.phi == pytest.approx(phi, abs=1e-3)
    assert res.pi == pytest.approx(pi, abs=1e-3)
    assert res.p == pytest.approx(p, abs=1e-3)


def test_fixed_phi_is_used():
    p, pi, phi = 0.3, 0.7, 0.03
    res = fit_mle(*expected_counts(p, pi, n=2, phi=phi), phi=phi)
    assert res.phi == phi
    assert res.pi == pytest.approx(pi, abs=1e-4)


def test_fit_by_group_matches_truth():
    cfg = SimConfig(n_cases=20000, n_findings=3, n_centres=2, seed=24,
                    prevalence_range=(0.2, 0.5), propensity_range=(0.5, 0.9))
    sim = simulate(cfg)
    est = fit_by_group(sim.instances).merge(sim.truth, on=["finding", "centre"],
                                            suffixes=("_hat", ""))
    assert len(est) == 6
    assert np.max(np.abs(est["pi_hat"] - est["pi"])) < 0.05
    assert np.max(np.abs(est["p_hat"] - est["p"])) < 0.05
