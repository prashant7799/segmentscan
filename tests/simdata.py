"""Synthetic data generators shared by the tests.

Nothing here touches the network or real data.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

# Three independent category columns with uniform levels.
LEVELS = {
    "a": ["a0", "a1", "a2"],
    "b": ["b0", "b1", "b2"],
    "c": ["c0", "c1"],
}
BASE_RATE = 0.30


def planted_rate(point: dict[str, str], planted_rate_value: float) -> float:
    """True rate at one full combination of levels: a0 & b0 is the planted cell."""
    if point["a"] == "a0" and point["b"] == "b0":
        return planted_rate_value
    return BASE_RATE


def make_table(n: int, seed: int, planted: float | None = None) -> pd.DataFrame:
    """Rows with dims a, b, c and a 0/1 outcome ``y``.

    With ``planted=None`` every row has rate BASE_RATE (a global null).
    Otherwise the cell a=a0 & b=b0 has rate ``planted``.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({dim: rng.choice(levels, n) for dim, levels in LEVELS.items()})
    rate = np.full(n, BASE_RATE)
    if planted is not None:
        rate[(df["a"] == "a0").to_numpy() & (df["b"] == "b0").to_numpy()] = planted
    df["y"] = (rng.random(n) < rate).astype(int)
    return df


def true_cell_rate(dims: tuple[str, ...], levels: tuple[str, ...], planted: float | None) -> float:
    """Population rate of a cell, averaging over the full grid of levels.

    Dims are independent and uniform, so every grid point has equal weight.
    """
    rates = []
    for combo in itertools.product(*LEVELS.values()):
        point = dict(zip(LEVELS, combo, strict=True))
        if all(point[dim] == level for dim, level in zip(dims, levels, strict=True)):
            rates.append(BASE_RATE if planted is None else planted_rate(point, planted))
    return float(np.mean(rates))


def make_clustered_null(
    n_clusters: int, rows_per_cluster: int, seed: int, mean: float = BASE_RATE
) -> pd.DataFrame:
    """Clustered data with NO real segment effect.

    Each user has their own rate drawn from a Beta distribution with the
    given mean (intraclass correlation about 1/3). ``region`` and ``plan``
    are fixed per user; ``device`` changes from row to row. No column
    changes the expected rate, so every discovery is false.
    """
    rng = np.random.default_rng(seed)
    beta = 1.4
    alpha = beta * mean / (1 - mean)  # Beta(alpha, beta) has mean alpha / (alpha + beta)
    user_rate = rng.beta(alpha, beta, n_clusters)
    region = rng.choice(["north", "south", "east", "west"], n_clusters)
    plan = rng.choice(["free", "basic", "pro"], n_clusters)

    user = np.repeat(np.arange(n_clusters), rows_per_cluster)
    df = pd.DataFrame(
        {
            "user_id": user,
            "region": region[user],
            "plan": plan[user],
            "device": rng.choice(["ios", "android", "web"], len(user)),
        }
    )
    df["y"] = (rng.random(len(user)) < user_rate[user]).astype(int)
    return df


def make_binary_null(n: int, n_dims: int, seed: int) -> pd.DataFrame:
    """``n_dims`` yes/no dims and an outcome with rate BASE_RATE everywhere.

    With five dims and depth 3 this gives 130 heavily overlapping cells.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"d{i}": rng.choice(["x", "z"], n) for i in range(n_dims)})
    df["y"] = (rng.random(n) < BASE_RATE).astype(int)
    return df


def make_unequal_clusters_null(n_clusters: int, max_rows: int, seed: int) -> pd.DataFrame:
    """Like ``make_clustered_null``, but users have 1 to ``max_rows`` rows each.

    How many rows a user has is unrelated to their rate, so the per-user
    average rate and the pooled row rate are both BASE_RATE.
    """
    rng = np.random.default_rng(seed)
    beta = 1.4
    alpha = beta * BASE_RATE / (1 - BASE_RATE)
    user_rate = rng.beta(alpha, beta, n_clusters)
    sizes = rng.integers(1, max_rows + 1, n_clusters)
    region = rng.choice(["north", "south", "east", "west"], n_clusters)
    user = np.repeat(np.arange(n_clusters), sizes)
    df = pd.DataFrame(
        {
            "user_id": user,
            "region": region[user],
            "device": rng.choice(["ios", "android", "web"], len(user)),
        }
    )
    df["y"] = (rng.random(len(user)) < user_rate[user]).astype(int)
    return df
