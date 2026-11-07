import pandas as pd

from partialref.propensity.bootstrap import case_bootstrap
from partialref.propensity.mle import fit_mle
from partialref.propensity.simulate import SimConfig, simulate


def mle_estimator(df: pd.DataFrame) -> dict[str, float]:
    r = fit_mle(df["n_reports"].to_numpy(), df["k"].to_numpy())
    return {"p": r.p, "pi": r.pi}


def small_sim(seed: int = 31) -> pd.DataFrame:
    cfg = SimConfig(n_findings=1, n_centres=1, centre_logit_sd=0.0, n_cases=2000,
                    prevalence_range=(0.3, 0.3), propensity_range=(0.7, 0.7), seed=seed)
    return simulate(cfg).instances


def test_ci_contains_truth_and_is_ordered():
    out = case_bootstrap(small_sim(), mle_estimator, n_boot=200, seed=0).set_index("param")
    for name, truth in [("p", 0.3), ("pi", 0.7)]:
        row = out.loc[name]
        assert row.lo <= row.estimate <= row.hi
        assert row.lo <= truth <= row.hi
        assert row.n_ok == 200


def test_deterministic_given_seed():
    df = small_sim()
    a = case_bootstrap(df, mle_estimator, n_boot=30, seed=5)
    b = case_bootstrap(df, mle_estimator, n_boot=30, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_resamples_whole_cases():
    df = simulate(SimConfig(n_cases=50, n_findings=4, seed=32)).instances
    seen = []

    def est(d: pd.DataFrame) -> dict[str, float]:
        seen.append((len(d), d.groupby("case_id").size().mod(4).eq(0).all()))
        return {"rows": float(len(d))}

    case_bootstrap(df, est, n_boot=20, seed=1)
    assert all(n_rows == 200 and whole for n_rows, whole in seen)
