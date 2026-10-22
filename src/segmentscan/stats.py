"""Statistics used by segmentscan.

Three pieces live here:

* ``binom_pvalues`` - one-sided exact binomial p-values for many cells at once.
* ``adjust_pvalues`` - multiple-testing adjustment (BH, BY or Holm).
* ``design_effect`` - how much clustered rows shrink the real sample size.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ("bh", "by", "holm")
ALTERNATIVES = ("greater", "less")


class DependenceWarning(UserWarning):
    """Rows look dependent, so row-level p-values are likely too small."""


def binom_pvalues(
    successes: Sequence[int] | np.ndarray,
    n: Sequence[int] | np.ndarray,
    rate: float,
    alternative: str = "greater",
) -> np.ndarray:
    """One-sided exact binomial p-values, computed for many cells at once.

    Let X follow Binomial(n, rate).

    * ``alternative="greater"`` returns P(X >= successes).
    * ``alternative="less"`` returns P(X <= successes).

    These are the same numbers as ``scipy.stats.binomtest(k, n, rate,
    alternative=...).pvalue``, but without a Python loop.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")
    if not 0.0 < rate < 1.0:
        raise ValueError(f"rate must be strictly between 0 and 1, got {rate}")

    k = np.asarray(successes, dtype=float)
    size = np.asarray(n, dtype=float)
    if k.shape != size.shape:
        raise ValueError("successes and n must have the same shape")
    if np.any(k < 0) or np.any(k > size):
        raise ValueError("successes must be between 0 and n")

    if alternative == "greater":
        # P(X >= k) is the survival function evaluated at k - 1.
        pvalues = stats.binom.sf(k - 1, size, rate)
    else:
        pvalues = stats.binom.cdf(k, size, rate)
    return np.clip(pvalues, 0.0, 1.0)


def adjust_pvalues(pvalues: Sequence[float] | np.ndarray, method: str = "bh") -> np.ndarray:
    """Adjust p-values for testing many cells at once.

    A cell is a discovery when its adjusted p-value is at or below ``q``.

    * ``"bh"`` - Benjamini-Hochberg. Controls the false discovery rate (FDR).
    * ``"by"`` - Benjamini-Yekutieli. Controls the FDR under any dependence
      between the tests. It is stricter than BH.
    * ``"holm"`` - Holm. Controls the family-wise error rate (the chance of
      even one false discovery). The strictest of the three.

    BH and BY use ``scipy.stats.false_discovery_control``. That is the full
    step-up rule: find the largest rank k with p_(k) <= k * q / m and accept
    every cell up to rank k. Comparing each p-value only with its own
    k * q / m line is a common bug; it can miss real discoveries.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if p.size == 0:
        return p.copy()
    if np.any(np.isnan(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("pvalues must be numbers between 0 and 1")

    if method == "holm":
        return _holm(p)
    return stats.false_discovery_control(p, method=method)


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values.

    Sort the p-values from small to large. The i-th smallest (counting from
    0) is multiplied by (m - i). A running maximum keeps the adjusted values
    in the same order as the raw ones.
    """
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def design_effect(df: pd.DataFrame, outcome: str, cluster: str) -> float:
    """Kish design effect for a 0/1 outcome measured on clustered rows.

    The design effect says how many times larger the variance of a rate is
    than it would be if every row were independent. A value of 1 means the
    rows behave like independent rows. A value of 4 means 1,000 rows carry
    about as much information as 250 independent rows.

    Formula: ``1 + (m - 1) * icc``. Here ``m`` is the size-weighted mean
    cluster size (sum of squared sizes / number of rows) and ``icc`` is the
    intraclass correlation from a one-way ANOVA, clipped to [0, 1].
    """
    for column in (outcome, cluster):
        if column not in df.columns:
            raise KeyError(f"column {column!r} not found")

    y = df[outcome].astype(float)
    groups = y.groupby(df[cluster], sort=False)
    sizes = groups.size().to_numpy(dtype=float)
    means = groups.mean().to_numpy(dtype=float)

    n_rows = sizes.sum()
    n_clusters = len(sizes)
    if n_clusters < 2 or n_rows <= n_clusters:
        # One cluster, or every cluster has a single row: nothing to estimate.
        return 1.0

    overall = float(y.mean())
    ms_between = np.sum(sizes * (means - overall) ** 2) / (n_clusters - 1)
    # For a 0/1 outcome the within-cluster sum of squares is sum(m_i * p_i * (1 - p_i)).
    ms_within = np.sum(sizes * means * (1 - means)) / (n_rows - n_clusters)
    m0 = (n_rows - np.sum(sizes**2) / n_rows) / (n_clusters - 1)

    denominator = ms_between + (m0 - 1) * ms_within
    if denominator <= 0:
        return 1.0
    icc = (ms_between - ms_within) / denominator
    icc = min(max(icc, 0.0), 1.0)

    weighted_size = np.sum(sizes**2) / n_rows
    return float(1 + (weighted_size - 1) * icc)
