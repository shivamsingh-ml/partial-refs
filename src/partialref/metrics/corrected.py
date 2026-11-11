"""Corrected precision under partial references (HANDOFF §2.2) and its expected value for a described system.

A predicted finding that none of the n reference reports mentions is still correct with probability
    q0 = P(present | 0 of n mention) = p (1-pi)^n / ( p (1-pi)^n + (1-p) (1-phi)^n ).
Corrected precision = (matched predictions + sum of q0 over unmatched predictions) / predictions.
"""

from __future__ import annotations

import numpy as np


def p_present_given_unmentioned(p, pi, n, phi=0.0):
    """q0 above; works on scalars or arrays. Prevalence p, reporting propensity pi, false-mention rate phi."""
    p, pi, n, phi = (np.asarray(x, float) for x in (p, pi, n, phi))
    a = p * (1 - pi) ** n
    return a / (a + (1 - p) * (1 - phi) ** n)


def corrected_precision(n_matched: float, q0_unmatched_sum: float, n_predictions: float) -> float:
    """§2.2 from counts: matched predictions, the summed q0 of the unmatched ones, all predictions."""
    return float("nan") if n_predictions == 0 else (n_matched + q0_unmatched_sum) / n_predictions


def expected_precisions(p: float, pi: float, n: int, true_precision: float) -> dict[str, float]:
    """Expected raw and corrected precision of a system with the given TRUE precision on one finding type,
    scored against n reference reports (phi = 0; the system's hits and the readers' omissions independent).
    Raw precision = true precision x share of true findings that the reference mentions; it does not depend
    on the system's sensitivity. The §2.2 estimator credits every unmatched prediction with q0, which is
    exact for a system without skill and too low for a skilled one (HANDOFF L1)."""
    mentioned = 1 - (1 - pi) ** n
    q0 = float(p_present_given_unmentioned(p, pi, n))
    raw = true_precision * mentioned
    corrected = raw + q0 * (true_precision * (1 - mentioned) + (1 - true_precision))
    return {"mentioned_share": mentioned, "q0": q0, "true": true_precision, "raw": raw, "corrected": corrected}
