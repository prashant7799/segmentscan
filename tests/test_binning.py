import numpy as np
import pandas as pd
import pytest

from segmentscan import apply_bins, edges_from_spec, quantile_edges, scan


def test_quantile_edges_have_open_ends_and_equal_counts():
    values = np.arange(100)
    edges = quantile_edges(values, 4)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert len(edges) == 5
    counts = apply_bins(values, edges).value_counts(sort=False)
    assert counts.tolist() == [25, 25, 25, 25]


def test_repeated_values_merge_edges():
    values = [1, 1, 1, 1, 1, 1, 1, 2, 3, 4]
    edges = quantile_edges(values, 4)
    assert len(edges) < 6  # fewer than 4 bins survive
    assert np.all(np.diff(edges) > 0)


def test_apply_bins_labels_and_missing_values():
    edges = np.array([-np.inf, 30, 45, np.inf])
    binned = apply_bins(pd.Series([5, 30, 44.9, 45, 100, np.nan]), edges)
    assert binned.tolist()[:5] == ["(-inf, 30)", "[30, 45)", "[30, 45)", "[45, inf)", "[45, inf)"]
    assert pd.isna(binned.iloc[5])
    assert binned.cat.ordered


def test_values_outside_the_fitted_range_still_get_a_bin():
    edges = quantile_edges([10, 20, 30, 40], 2)
    binned = apply_bins([-1000, 1000], edges)
    assert binned.notna().all()


def test_explicit_edges_get_infinite_ends():
    edges = edges_from_spec([1, 2, 3], [18, 30, 65])
    assert edges.tolist() == [-np.inf, 18, 30, 65, np.inf]
    with pytest.raises(ValueError):
        edges_from_spec([1], [30, 18])
    with pytest.raises(ValueError):
        edges_from_spec([1], [])


def test_bad_bin_counts_are_rejected():
    with pytest.raises(ValueError):
        quantile_edges([1, 2, 3], 1)
    with pytest.raises(ValueError):
        quantile_edges([np.nan, np.nan], 3)
    with pytest.raises(ValueError):
        quantile_edges([1, 2, 3], True)


def _table_with_age(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "plan": rng.choice(["free", "pro"], n),
            "y": (rng.random(n) < 0.3).astype(int),
        }
    )


def test_scan_fits_bin_edges_on_explore_rows_only():
    df = _table_with_age()
    result = scan(df, ["age", "plan"], "y", 0.3, bins={"age": 4})
    explore_ages = df.loc[result.splits == "explore", "age"]
    assert result.bin_edges["age"] == pytest.approx(quantile_edges(explore_ages, 4))


def test_holdout_values_cannot_move_the_bin_edges():
    """The leak check: change every holdout age wildly; edges must not move."""
    df = _table_with_age()
    first = scan(df, ["age", "plan"], "y", 0.3, bins={"age": 4})

    changed = df.copy()
    holdout = first.splits == "holdout"
    changed.loc[holdout, "age"] = changed.loc[holdout, "age"] * 10 + 500
    second = scan(changed, ["age", "plan"], "y", 0.3, bins={"age": 4})

    assert second.bin_edges["age"].tolist() == first.bin_edges["age"].tolist()


def test_scan_rejects_bins_on_unknown_or_text_columns():
    df = _table_with_age()
    with pytest.raises(ValueError, match="also be listed in dims"):
        scan(df, ["plan"], "y", 0.3, bins={"age": 4})
    with pytest.raises(ValueError, match="numeric"):
        scan(df, ["age", "plan"], "y", 0.3, bins={"plan": 2})
