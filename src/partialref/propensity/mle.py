"""Maximum-likelihood propensity estimate over all instances (HANDOFF §2.1).

Per instance with n reports, k of which mention the finding:

    L(p, pi, phi) = p * Binom(k; n, pi) + (1 - p) * Binom(k; n, phi)

Base model phi = 0, where the second term is (1 - p) * [k = 0]. Single-report instances
contribute; identifiability of pi needs instances with n >= 2. With ``fit_phi=True``, phi is
parameterised as pi * sigmoid(z) to remove label switching (see DECISIONS.md); phi is only
identified with n >= 3 instances or external information (e.g. a mask "reader").

The likelihood is evaluated on the sufficient statistics (counts per distinct (n, k)), so fits
are cheap enough for per-group bootstraps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import binom

from partialref.propensity.closed_form import closed_form_n2

_EPS = 1e-6


@dataclass(frozen=True)
class MLEResult:
    p: float
    pi: float
    phi: float
    loglik: float
    n_instances: float
    converged: bool
    identifiable: bool  # False if no instance has n >= 2 with k >= 1 (pi not identified)
    fit_phi: bool

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def sufficient_stats(
    n: ArrayLike, k: ArrayLike, weights: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse instances to distinct (n, k) pairs with summed weights."""
    n = np.asarray(n, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    w = np.ones(n.shape, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if not (n.shape == k.shape == w.shape):
        raise ValueError("n, k and weights must have the same shape")
    if np.any(n < 1) or np.any((k < 0) | (k > n)):
        raise ValueError("need n >= 1 and 0 <= k <= n")
    keys, inv = np.unique(np.stack([n, k], axis=1), axis=0, return_inverse=True)
    counts = np.bincount(inv.ravel(), weights=w, minlength=len(keys))
    keep = counts > 0
    return keys[keep, 0], keys[keep, 1], counts[keep]


def _unpack(theta: np.ndarray, fit_phi: bool, phi_fixed: float) -> tuple[float, float, float]:
    p = float(expit(theta[0]))
    pi = float(expit(theta[1]))
    phi = pi * float(expit(theta[2])) if fit_phi else phi_fixed
    return p, pi, phi


def _loglik(
    theta: np.ndarray, n: np.ndarray, k: np.ndarray, w: np.ndarray, fit_phi: bool, phi_fixed: float
) -> float:
    p, pi, phi = _unpack(theta, fit_phi, phi_fixed)
    with np.errstate(divide="ignore"):
        t1 = np.log(p) + binom.logpmf(k, n, pi)
        t2 = np.log1p(-p) + binom.logpmf(k, n, phi)  # phi = 0 -> log(1-p)*[k=0], else -inf
    return float(np.sum(w * np.logaddexp(t1, t2)))


def fit_mle(
    n: ArrayLike,
    k: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    fit_phi: bool = False,
    phi: float = 0.0,
) -> MLEResult:
    """Fit (p, pi[, phi]) by maximum likelihood.

    Args:
        n, k: reports and mentions per instance.
        weights: optional per-instance weights.
        fit_phi: estimate phi (needs n >= 3 instances to be identified).
        phi: fixed false-positive rate when ``fit_phi`` is False (default 0, the base model).
    """
    if not 0.0 <= phi < 1.0:
        raise ValueError("phi must be in [0, 1)")
    nn, kk, ww = sufficient_stats(n, k, weights)
    total = float(ww.sum())
    identifiable = bool(np.any((nn >= 2) & (kk >= 1)))
    if total == 0:
        nan = float("nan")
        return MLEResult(nan, nan, nan, nan, 0.0, False, False, fit_phi)

    # starting points: closed form if usable, plus a small fixed grid (cheap: 2-3 params)
    starts: list[tuple[float, float]] = [(0.3, 0.7), (0.1, 0.5), (0.6, 0.9)]
    cf = closed_form_n2(nn, kk, ww)
    if np.isfinite(cf.pi) and 0 < cf.p < 1:
        starts.insert(0, (cf.p, cf.pi))

    def clip(x: float) -> float:
        return float(np.clip(x, _EPS, 1 - _EPS))

    best = None
    for p0, pi0 in starts:
        x0 = [logit(clip(p0)), logit(clip(pi0))] + ([logit(0.05)] if fit_phi else [])
        res = minimize(
            lambda th: -_loglik(th, nn, kk, ww, fit_phi, phi) / total,
            np.asarray(x0, dtype=float),
            method="L-BFGS-B",
            bounds=[(-30, 30)] * len(x0),
        )
        if best is None or res.fun < best.fun:
            best = res

    p_hat, pi_hat, phi_hat = _unpack(best.x, fit_phi, phi)
    return MLEResult(
        p=p_hat,
        pi=pi_hat,
        phi=phi_hat,
        loglik=float(-best.fun * total),
        n_instances=total,
        converged=bool(best.success),
        identifiable=identifiable,
        fit_phi=fit_phi,
    )


def fit_by_group(
    df: pd.DataFrame,
    by: Sequence[str] = ("finding", "centre"),
    *,
    n_col: str = "n_reports",
    k_col: str = "k",
    **kwargs,
) -> pd.DataFrame:
    """Fit ``fit_mle`` separately per group; one row per group."""
    rows = []
    for key, g in df.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        res = fit_mle(g[n_col].to_numpy(), g[k_col].to_numpy(), **kwargs)
        rows.append({**dict(zip(by, key)), **res.as_dict()})
    return pd.DataFrame(rows)
