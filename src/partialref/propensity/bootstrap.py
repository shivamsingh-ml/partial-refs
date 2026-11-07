"""Case-level (cluster) bootstrap for propensity estimators (HANDOFF §2.1: >= 1000 resamples).

Instances of the same case share readers, so cases, not instances, are resampled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd


def case_bootstrap(
    df: pd.DataFrame,
    estimator: Callable[[pd.DataFrame], Mapping[str, float]],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    case_col: str = "case_id",
) -> pd.DataFrame:
    """Percentile bootstrap CIs, resampling whole cases with replacement.

    Args:
        df: instance table (one row per case x finding, possibly several rows per case).
        estimator: maps a (resampled) instance table to named scalar estimates.
        n_boot: number of resamples.
        alpha: 1 - confidence level.
        seed: RNG seed.

    Returns:
        One row per estimate name: estimate (on the full data), lo, hi, n_ok (finite resamples).
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    point = dict(estimator(df))
    codes, _ = pd.factorize(df[case_col], sort=False)
    n_cases = int(codes.max()) + 1
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(n_cases + 1))
    rows_of = [order[bounds[i] : bounds[i + 1]] for i in range(n_cases)]

    rng = np.random.default_rng(seed)
    draws = {name: np.full(n_boot, np.nan) for name in point}
    for b in range(n_boot):
        pick = rng.integers(0, n_cases, size=n_cases)
        idx = np.concatenate([rows_of[i] for i in pick])
        est = estimator(df.iloc[idx])
        for name in point:
            draws[name][b] = est.get(name, np.nan)

    out = []
    for name, x in draws.items():
        ok = x[np.isfinite(x)]
        lo, hi = (np.quantile(ok, [alpha / 2, 1 - alpha / 2]) if ok.size else (np.nan, np.nan))
        out.append(
            {"param": name, "estimate": point[name], "lo": float(lo), "hi": float(hi),
             "n_ok": int(ok.size)}
        )
    return pd.DataFrame(out)
