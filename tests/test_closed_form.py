import numpy as np
import pytest

from partialref.propensity.closed_form import closed_form_n2
from partialref.propensity.simulate import SimConfig, simulate

from conftest import expected_counts, stack


@pytest.mark.parametrize("p,pi", [(0.3, 0.7), (0.05, 0.4), (0.6, 0.95), (0.2, 0.2)])
def test_exact_on_expected_counts(p, pi):
    n, k, w = expected_counts(p, pi, n=2)
    est = closed_form_n2(n, k, w)
    assert est.pi == pytest.approx(pi, abs=1e-12)
    assert est.p == pytest.approx(p, abs=1e-12)


def test_integer_example():
    # p = 0.3, pi = 0.7, 1000 instances: 147 both, 126 one, 727 none
    n = np.full(1000, 2)
    k = np.array([2] * 147 + [1] * 126 + [0] * 727)
    est = closed_form_n2(n, k)
    assert est.pi == pytest.approx(0.7)
    assert est.p == pytest.approx(0.3)


def test_ignores_instances_without_two_reports():
    n2, k2, w2 = expected_counts(0.3, 0.7, n=2)
    noise = (np.array([1, 3, 3]), np.array([1, 3, 0]), np.array([50.0, 50.0, 50.0]))
    est = closed_form_n2(*stack((n2, k2, w2), noise))
    assert est.pi == pytest.approx(0.7, abs=1e-12)


def test_no_double_mentions_returns_nan():
    est = closed_form_n2([2, 2, 2], [1, 0, 0])
    assert np.isnan(est.pi) and np.isnan(est.p)
    assert est.n_one == 1


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        closed_form_n2([2], [3])
    with pytest.raises(ValueError):
        closed_form_n2([2, 2], [1])


def test_recovers_on_simulated_data():
    cfg = SimConfig(n_findings=1, n_centres=1, centre_logit_sd=0.0, n_cases=40000,
                    prevalence_range=(0.3, 0.3), propensity_range=(0.7, 0.7), seed=11)
    df = simulate(cfg).instances
    est = closed_form_n2(df["n_reports"], df["k"])
    assert est.pi == pytest.approx(0.7, abs=0.02)
    assert est.p == pytest.approx(0.3, abs=0.02)
