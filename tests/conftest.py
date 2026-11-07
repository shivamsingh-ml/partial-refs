"""Shared helpers for synthetic-data tests (never real reports)."""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def expected_counts(
    p: float, pi: float, n: int, phi: float = 0.0, rho: float = 0.0, total: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact expected (n, k, weight) for n reports under the simulate.py generative model.

    With rho > 0, given presence: prob rho all readers share one draw, else independent draws.
    """
    k = np.arange(n + 1)
    shared = np.zeros(n + 1)
    shared[0], shared[n] = 1 - pi, pi
    given_present = rho * shared + (1 - rho) * binom.pmf(k, n, pi)
    given_absent = binom.pmf(k, n, phi)
    w = total * (p * given_present + (1 - p) * given_absent)
    return np.full(n + 1, n), k, w


def stack(*parts: tuple[np.ndarray, np.ndarray, np.ndarray]):
    n, k, w = zip(*parts)
    return np.concatenate(n), np.concatenate(k), np.concatenate(w)
