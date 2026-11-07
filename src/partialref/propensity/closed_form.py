"""Closed-form propensity estimate for n = 2, phi = 0 (HANDOFF §2.1).

    P(both) = p pi^2,   P(exactly one) = 2 p pi (1 - pi)
    R = P(exactly one) / P(both)  ->  pi = 2 / (R + 2),   p = P(both) / pi^2

Only instances with exactly two reports are used; this is a sanity check for the MLE, which on
n = 2 data alone is the same estimator whenever p_hat <= 1 (the n = 2, phi = 0 model is saturated).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ClosedFormEstimate:
    p: float
    pi: float
    n_instances: float  # (weighted) number of n == 2 instances
    n_both: float
    n_one: float

    @property
    def p_out_of_range(self) -> bool:
        """True if sampling noise pushed p_hat above 1 (estimate is then not a valid probability)."""
        return bool(np.isfinite(self.p) and self.p > 1.0)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def closed_form_n2(
    n: ArrayLike, k: ArrayLike, weights: ArrayLike | None = None
) -> ClosedFormEstimate:
    """Estimate (p, pi) from the n == 2 instances.

    Args:
        n: number of reports per instance.
        k: number of reports mentioning the finding per instance.
        weights: optional per-instance weights (e.g. expected counts in tests). Default 1.

    Returns NaN for p and pi when no n == 2 instance is mentioned by both readers.
    """
    n = np.asarray(n)
    k = np.asarray(k)
    w = np.ones(n.shape, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if not (n.shape == k.shape == w.shape):
        raise ValueError("n, k and weights must have the same shape")
    if np.any((k < 0) | (k > n)):
        raise ValueError("need 0 <= k <= n")

    sel = n == 2
    total = float(w[sel].sum())
    both = float(w[sel & (k == 2)].sum())
    one = float(w[sel & (k == 1)].sum())
    if total == 0 or both == 0:
        return ClosedFormEstimate(np.nan, np.nan, total, both, one)

    pi = 2.0 * both / (one + 2.0 * both)  # == 2 / (R + 2) with R = one / both
    p = (both / total) / pi**2
    return ClosedFormEstimate(p=p, pi=pi, n_instances=total, n_both=both, n_one=one)
