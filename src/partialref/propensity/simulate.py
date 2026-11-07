"""Synthetic partial-reference data with known ground truth (HANDOFF §2.1, §7 M0.5).

Generative model, per finding instance (case, finding):

- centre c(case) drawn from ``centre_probs`` (uniform if unset);
- number of reports n(case) drawn from ``report_count_probs`` (index i -> i + 1 reports);
- truly present with probability p[f, c];
- if present, each of the n readers mentions it with marginal probability pi[f, c]. With
  probability ``rho`` all readers of the case share one Bernoulli(pi) draw (shared omission /
  house style), otherwise they draw independently. This keeps the marginal at pi and makes the
  correlation between two readers' mentions (given presence) exactly rho;
- if absent, each reader mentions it independently with probability ``phi`` (false positive).

Centre house style: logit(pi[f, c]) = logit(pi_f) + N(0, centre_logit_sd^2), and likewise for p
with ``centre_prev_logit_sd``; both are clipped to [CLIP_LO, CLIP_HI].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.special import expit, logit

CLIP_LO, CLIP_HI = 0.01, 0.99


class SimConfig(BaseModel):
    """Simulation settings. Defaults follow HANDOFF §7 M0.5."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_cases: int = Field(600, ge=1)
    n_findings: int = Field(12, ge=1)
    n_centres: int = Field(4, ge=1)
    prevalence_range: tuple[float, float] = (0.01, 0.6)
    propensity_range: tuple[float, float] = (0.2, 0.95)
    centre_logit_sd: float = Field(0.5, ge=0.0)
    centre_prev_logit_sd: float = Field(0.0, ge=0.0)
    centre_probs: tuple[float, ...] | None = None
    report_count_probs: tuple[float, ...] = (0.4, 0.45, 0.15)
    rho: float = Field(0.0, ge=0.0, le=1.0)
    phi: float = Field(0.0, ge=0.0, lt=1.0)
    seed: int = 0

    @model_validator(mode="after")
    def _check(self) -> SimConfig:
        for name in ("prevalence_range", "propensity_range"):
            lo, hi = getattr(self, name)
            if not (0.0 < lo <= hi < 1.0):
                raise ValueError(f"{name} must satisfy 0 < lo <= hi < 1, got {(lo, hi)}")
        if not np.isclose(sum(self.report_count_probs), 1.0):
            raise ValueError("report_count_probs must sum to 1")
        if any(q < 0 for q in self.report_count_probs):
            raise ValueError("report_count_probs must be non-negative")
        if self.centre_probs is not None:
            if len(self.centre_probs) != self.n_centres:
                raise ValueError("centre_probs must have n_centres entries")
            if not np.isclose(sum(self.centre_probs), 1.0) or min(self.centre_probs) < 0:
                raise ValueError("centre_probs must be non-negative and sum to 1")
        return self

    @property
    def max_reports(self) -> int:
        return len(self.report_count_probs)


@dataclass(frozen=True)
class SimData:
    """Simulation output.

    ``instances``: one row per (case, finding) with columns
        case_id, centre, finding, n_reports, k, present, m1..m{max_reports}
        (m_j = 1/0 mention by reader j, -1 if the case has fewer than j reports).
    ``truth``: one row per (finding, centre) with the true p, pi and the global phi, rho.
    """

    instances: pd.DataFrame
    truth: pd.DataFrame
    config: SimConfig


def _jitter(base: np.ndarray, sd: float, n_centres: int, rng: np.random.Generator) -> np.ndarray:
    """(K,) base probabilities -> (K, C) centre-specific probabilities via a logit offset."""
    shape = (base.size, n_centres)
    offsets = rng.normal(0.0, sd, size=shape) if sd > 0 else np.zeros(shape)
    return np.clip(expit(logit(base)[:, None] + offsets), CLIP_LO, CLIP_HI)


def simulate(cfg: SimConfig) -> SimData:
    """Draw one synthetic dataset from ``cfg`` (deterministic given ``cfg.seed``)."""
    rng = np.random.default_rng(cfg.seed)
    K, C, N, R = cfg.n_findings, cfg.n_centres, cfg.n_cases, cfg.max_reports

    # --- ground truth per finding x centre
    p_f = rng.uniform(*cfg.prevalence_range, size=K)
    pi_f = rng.uniform(*cfg.propensity_range, size=K)
    p_fc = _jitter(p_f, cfg.centre_prev_logit_sd, C, rng)
    pi_fc = _jitter(pi_f, cfg.centre_logit_sd, C, rng)

    # --- cases
    centre = rng.choice(C, size=N, p=cfg.centre_probs)
    n_rep = rng.choice(np.arange(1, R + 1), size=N, p=cfg.report_count_probs)

    # --- instances: (case, finding) flattened, case-major
    case_idx = np.repeat(np.arange(N), K)
    find_idx = np.tile(np.arange(K), N)
    c_idx = centre[case_idx]
    n_i = n_rep[case_idx]
    p_i = p_fc[find_idx, c_idx]
    pi_i = pi_fc[find_idx, c_idx]
    M = N * K

    present = rng.random(M) < p_i
    indep = rng.random((M, R)) < pi_i[:, None]
    shared_draw = rng.random(M) < pi_i
    use_shared = rng.random(M) < cfg.rho
    mention_if_present = np.where(use_shared[:, None], shared_draw[:, None], indep)
    false_pos = rng.random((M, R)) < cfg.phi
    mention = np.where(present[:, None], mention_if_present, false_pos)

    has_reader = np.arange(R)[None, :] < n_i[:, None]
    m = np.where(has_reader, mention.astype(np.int8), np.int8(-1))
    k = (m == 1).sum(axis=1)

    instances = pd.DataFrame(
        {
            "case_id": np.char.add("S", np.char.zfill(case_idx.astype(str), 5)),
            "centre": np.char.add("C", c_idx.astype(str)),
            "finding": np.char.add("F", np.char.zfill(find_idx.astype(str), 2)),
            "n_reports": n_i.astype(np.int64),
            "k": k.astype(np.int64),
            "present": present,
        }
    )
    for j in range(R):
        instances[f"m{j + 1}"] = m[:, j]

    ff, cc = np.meshgrid(np.arange(K), np.arange(C), indexing="ij")
    truth = pd.DataFrame(
        {
            "finding": np.char.add("F", np.char.zfill(ff.ravel().astype(str), 2)),
            "centre": np.char.add("C", cc.ravel().astype(str)),
            "p": p_fc.ravel(),
            "pi": pi_fc.ravel(),
            "phi": cfg.phi,
            "rho": cfg.rho,
        }
    )
    return SimData(instances=instances, truth=truth, config=cfg)
