"""Turn numeric columns into categories without leaking the holdout.

``scan(..., bins={"age": 4})`` calls these helpers. The bin edges are fitted
on the explore rows only. The same edges are then applied to every row.
If the edges came from all rows, the holdout would help shape the cells it
is meant to test. That is a quiet leak.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def quantile_edges(values: Sequence[float] | pd.Series | np.ndarray, n_bins: int) -> np.ndarray:
    """Edges that cut ``values`` into about ``n_bins`` bins of equal size.

    The result starts with -inf and ends with +inf. So every future value,
    even one outside the range seen here, falls into some bin. Repeated
    values can merge edges, so you may get fewer than ``n_bins`` bins.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError(f"n_bins must be an integer >= 2, got {n_bins!r}")

    numbers = pd.to_numeric(pd.Series(values), errors="raise").dropna()
    x = numbers.to_numpy(dtype=float)
    if x.size == 0:
        raise ValueError("cannot fit bin edges: no non-missing values")

    cut_points = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    inner = np.unique(np.quantile(x, cut_points))
    return np.concatenate(([-np.inf], inner, [np.inf]))


def edges_from_spec(
    values: Sequence[float] | pd.Series | np.ndarray,
    spec: int | Sequence[float],
) -> np.ndarray:
    """Build bin edges from a bin count or from explicit edges.

    * An integer means "this many quantile bins, fitted on ``values``".
    * A list of numbers is used as the edges. -inf and +inf are added at the
      ends when they are missing.
    """
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        return quantile_edges(values, int(spec))
    if isinstance(spec, (str, bytes, bool, float, np.floating)) or not hasattr(spec, "__iter__"):
        raise ValueError(
            f"bin spec must be a whole number of bins or a list of edges, got {spec!r}"
        )

    edges = np.asarray(list(spec), dtype=float)
    if edges.ndim != 1 or edges.size == 0:
        raise ValueError("explicit bin edges must be a non-empty list of numbers")
    if np.any(np.isnan(edges)):
        raise ValueError("bin edges must not contain NaN")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("bin edges must be strictly increasing")
    if edges[0] != -np.inf:
        edges = np.concatenate(([-np.inf], edges))
    if edges[-1] != np.inf:
        edges = np.concatenate((edges, [np.inf]))
    return edges


def apply_bins(values: Sequence[float] | pd.Series | np.ndarray, edges: np.ndarray) -> pd.Series:
    """Map numbers to bin labels such as ``"[30, 45)"``.

    Bins include their left edge and exclude their right edge. Missing values
    stay missing. The result is an ordered categorical Series, so cells sort
    from low to high.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        raise ValueError("need at least two edges")
    labels = _labels(edges)
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    numbers = pd.to_numeric(series, errors="raise")
    return pd.cut(numbers, bins=edges, right=False, labels=labels, ordered=True)


def _labels(edges: np.ndarray) -> list[str]:
    """Readable, unique labels for each bin."""
    for formatter in (_short_number, repr):
        labels = []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            if low == -np.inf:
                labels.append(f"(-inf, {formatter(float(high))})")
            else:
                labels.append(f"[{formatter(float(low))}, {formatter(float(high))})")
        if len(set(labels)) == len(labels):
            return labels
    raise ValueError("could not build unique bin labels")  # pragma: no cover


def _short_number(x: float) -> str:
    if x == np.inf:
        return "inf"
    return f"{x:.6g}"
