import numpy as np
import pandas as pd
import pytest

from segmentscan.split import assign_halves, assign_splits


def _clustered(n_clusters=50, size=4):
    return pd.DataFrame({"user": np.repeat(np.arange(n_clusters), size), "x": 0})


def test_row_split_sizes_follow_the_fractions():
    df = pd.DataFrame({"x": np.zeros(1000)})
    labels = assign_splits(df, None, 0.3, 0.2, np.random.default_rng(0))
    counts = pd.Series(labels).value_counts()
    assert counts["explore"] == 500
    assert counts["holdout"] == 300
    assert counts["confirm"] == 200


def test_clusters_never_cross_splits():
    df = _clustered()
    labels = assign_splits(df, "user", 0.5, 0.2, np.random.default_rng(1))
    labels_per_user = pd.Series(labels).groupby(df["user"]).nunique()
    assert (labels_per_user == 1).all()


def test_halves_keep_clusters_together_and_are_balanced():
    df = _clustered(n_clusters=51)
    halves = assign_halves(df, "user", np.random.default_rng(2))
    assert (pd.Series(halves).groupby(df["user"]).nunique() == 1).all()
    units_per_half = pd.Series(halves).groupby(df["user"]).first().value_counts()
    assert sorted(units_per_half.tolist()) == [25, 26]


def test_same_seed_gives_same_split():
    df = _clustered()
    first = assign_splits(df, "user", 0.5, 0.0, np.random.default_rng(7))
    second = assign_splits(df, "user", 0.5, 0.0, np.random.default_rng(7))
    assert first.tolist() == second.tolist()


def test_too_few_units_is_an_error():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="too few units"):
        assign_splits(df, None, 0.5, 0.0, np.random.default_rng(0))
