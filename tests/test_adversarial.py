"""Adversarial checks: compare scan() with slow, obvious code, and try to break it.

Every check here uses synthetic data. The reference code loops over rows
with plain boolean masks, so it is easy to trust and easy to read.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy import stats

from segmentscan import adjust_pvalues, binom_pvalues, scan

# ---------- slow reference code ----------


def cell_mask(rows, dims, levels):
    """Boolean mask of the rows inside one cell."""
    mask = np.ones(len(rows), dtype=bool)
    for dim, level in zip(dims, levels, strict=True):
        mask &= (rows[dim] == level).to_numpy()
    return mask


def all_cells(rows, dims, max_depth):
    """Every (dims, levels) pair that has at least one row, found by brute force."""
    cells = set()
    for depth in range(1, min(max_depth, len(dims)) + 1):
        for combo in itertools.combinations(dims, depth):
            present = rows[list(combo)].dropna()
            for levels in present.itertuples(index=False, name=None):
                cells.add((combo, tuple(levels)))
    return cells


def row_counts(rows, dims, levels, outcome):
    mask = cell_mask(rows, dims, levels)
    return int(mask.sum()), int(rows[outcome].to_numpy()[mask].sum())


def cluster_counts(rows, dims, levels, outcome, cluster):
    """Units in a cell when each cluster counts once, plus bounds on successes.

    scan() keeps one random row per cluster. So its success count must lie
    between "clusters whose rows are all 1" and "clusters with any 1".
    """
    inside = rows[cell_mask(rows, dims, levels)]
    per_cluster = inside.groupby(cluster)[outcome]
    n_units = int(per_cluster.ngroups)
    all_ones = int((per_cluster.min() == 1).sum())
    any_one = int((per_cluster.max() == 1).sum())
    return n_units, all_ones, any_one


def beats(rate, bar, alternative):
    return rate > bar if alternative == "greater" else rate < bar


def random_table(seed, n_rows, n_dims, kinds, n_clusters, missing):
    """A small random table with mixed column types and some missing values."""
    rng = np.random.default_rng(seed)
    columns = {}
    for i in range(n_dims):
        n_levels = int(rng.integers(1, 5))
        if kinds[i] == "text":
            values = rng.choice([f"L{j}" for j in range(n_levels)], n_rows).astype(object)
        elif kinds[i] == "int":
            values = rng.integers(0, n_levels, n_rows).astype(float)
        else:
            values = rng.random(n_rows) < 0.5
            values = values.astype(object)
        if missing:
            values[rng.random(n_rows) < 0.1] = None
        columns[f"d{i}"] = values
    df = pd.DataFrame(columns)
    for i in range(n_dims):
        if kinds[i] == "int" and not missing:
            df[f"d{i}"] = df[f"d{i}"].astype(int)
    df["y"] = (rng.random(n_rows) < rng.uniform(0.1, 0.9)).astype(int)
    if n_clusters:
        df["user"] = rng.integers(0, n_clusters, n_rows)
    return df


# ---------- the main property test ----------


@pytest.mark.filterwarnings("ignore::segmentscan.DependenceWarning")
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 10_000),
    n_rows=st.integers(40, 300),
    n_dims=st.integers(1, 3),
    kinds=st.lists(st.sampled_from(["text", "int", "bool"]), min_size=3, max_size=3),
    missing=st.booleans(),
    clustered=st.sampled_from([None, "aggregate", "rows"]),
    threshold=st.sampled_from([0.2, 0.35, 0.5, 0.65]),
    min_lift=st.sampled_from([0.0, 0.05]),
    alternative=st.sampled_from(["greater", "less"]),
    method=st.sampled_from(["bh", "by", "holm"]),
    min_n=st.integers(1, 12),
    confirm=st.sampled_from([0.0, 0.2]),
    max_depth=st.integers(1, 3),
)
def test_scan_matches_brute_force(
    seed, n_rows, n_dims, kinds, missing, clustered, threshold, min_lift, alternative,
    method, min_n, confirm, max_depth,
):  # fmt: skip
    n_clusters = 25 if clustered else 0
    df = random_table(seed, n_rows, n_dims, kinds, n_clusters, missing)
    dims = [f"d{i}" for i in range(n_dims)]
    cluster = "user" if clustered else None
    result = scan(
        df,
        dims,
        "y",
        threshold,
        min_lift=min_lift,
        alternative=alternative,
        method=method,
        min_n=min_n,
        confirm=confirm,
        max_depth=max_depth,
        cluster=cluster,
        aggregate=clustered != "rows",
        seed=seed,
    )
    bar = result.bar
    explore = df[result.splits == "explore"]
    holdout = df[result.splits == "holdout"]
    confirm_rows = df[result.splits == "confirm"]

    # Clusters never cross splits.
    if cluster:
        assert (result.splits.groupby(df[cluster]).nunique() == 1).all()

    # The explore counts are right, and n_explored counts every non-empty cell.
    assert result.n_explored == len(all_cells(explore, dims, max_depth))

    # The tested cells are the explore candidates with enough holdout units.
    # Counting rows, we can say exactly which cells those are. Counting one
    # random row per cluster, we know bounds: a cell whose worst case passes
    # the screen must be tested, and a cell whose best case fails must not.
    count_clusters = clustered == "aggregate"
    must_test, may_test = set(), set()
    for cell_dims, levels in all_cells(explore, dims, max_depth):
        if count_clusters:
            n, low, high = cluster_counts(explore, cell_dims, levels, "y", cluster)
            holdout_n = cluster_counts(holdout, cell_dims, levels, "y", cluster)[0]
        else:
            n, low = row_counts(explore, cell_dims, levels, "y")
            high = low
            holdout_n = row_counts(holdout, cell_dims, levels, "y")[0]
        if n < min_n or holdout_n < min_n:
            continue
        worst, best = (low, high) if alternative == "greater" else (high, low)
        if beats(worst / n, bar, alternative):
            must_test.add((cell_dims, levels))
        if beats(best / n, bar, alternative):
            may_test.add((cell_dims, levels))
    tested = set(zip(result.cells["dims"], result.cells["levels"], strict=True))
    assert must_test <= tested <= may_test
    if not count_clusters:
        assert tested == must_test

    for row in result.cells.itertuples():
        if count_clusters:
            n, low, high = cluster_counts(explore, row.dims, row.levels, "y", cluster)
            assert row.explore_n == n
            assert low / n - 1e-12 <= row.explore_rate <= high / n + 1e-12
        else:
            n, k = row_counts(explore, row.dims, row.levels, "y")
            assert row.explore_n == n
            assert row.explore_rate == pytest.approx(k / n)

        if count_clusters:
            units, low, high = cluster_counts(holdout, row.dims, row.levels, "y", cluster)
            assert row.holdout_n == units
            assert low <= row.holdout_successes <= high
        else:
            assert (row.holdout_n, row.holdout_successes) == row_counts(
                holdout, row.dims, row.levels, "y"
            )

        # The two halves add up to the whole holdout, in units and in successes.
        assert row.half1_n + row.half2_n == row.holdout_n
        half_successes = 0
        for half_n, half_rate in ((row.half1_n, row.half1_rate), (row.half2_n, row.half2_rate)):
            if half_n > 0:
                half_successes += round(half_rate * half_n)
        assert half_successes == row.holdout_successes

        expected_p = stats.binomtest(
            int(row.holdout_successes), int(row.holdout_n), bar, alternative=alternative
        ).pvalue
        assert row.p_value == pytest.approx(expected_p, rel=1e-9, abs=1e-12)
        assert row.lift == pytest.approx(row.holdout_rate - threshold)

    # Adjusted p-values and discoveries follow from the raw p-values.
    if len(result.cells):
        adjusted = adjust_pvalues(result.cells["p_value"].to_numpy(), method)
        assert result.cells["p_adj"].to_numpy() == pytest.approx(adjusted)
        assert (result.cells["discovery"] == (result.cells["p_adj"] <= result.q)).all()

    # The confirm split re-tests the discoveries only, counted the same way.
    if confirm > 0 and len(result.discoveries) and not count_clusters:
        found = result.discoveries
        confirm_n, confirm_successes = [], []
        for cell_dims, levels in zip(found["dims"], found["levels"], strict=True):
            n, k = row_counts(confirm_rows, cell_dims, levels, "y")
            confirm_n.append(n)
            confirm_successes.append(k)
        assert found["confirm_n"].tolist() == confirm_n
        pvalues = binom_pvalues(confirm_successes, confirm_n, bar, alternative)
        expected_confirmed = adjust_pvalues(pvalues, method) <= result.q
        assert found["confirmed"].tolist() == expected_confirmed.tolist()


# ---------- the holdout and explore splits stay separate ----------


def planted_table(seed, n=4000):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "region": rng.choice(["north", "south", "east"], n),
            "tier": rng.choice(["free", "pro", "team"], n),
            "channel": rng.choice(["web", "app"], n),
            "user": rng.integers(0, 800, n),
        }
    )
    rate = np.where((df["region"] == "north") & (df["tier"] == "pro"), 0.45, 0.25)
    df["y"] = (rng.random(n) < rate).astype(int)
    return df


DIMS = ["region", "tier", "channel"]


@pytest.mark.parametrize("cluster", [None, "user"])
def test_holdout_outcomes_cannot_change_which_cells_are_tested(cluster):
    df = planted_table(0)
    first = scan(df, DIMS, "y", 0.25, cluster=cluster, min_n=5)

    scrambled = df.copy()
    holdout = (first.splits == "holdout").to_numpy()
    rng = np.random.default_rng(1)
    scrambled.loc[holdout, "y"] = rng.integers(0, 2, holdout.sum())
    second = scan(scrambled, DIMS, "y", 0.25, cluster=cluster, min_n=5)

    assert second.splits.equals(first.splits)
    assert set(second.cells["cell"]) == set(first.cells["cell"])
    assert second.n_candidates == first.n_candidates


@pytest.mark.parametrize("cluster", [None, "user"])
def test_explore_outcomes_cannot_change_holdout_counts(cluster):
    df = planted_table(2)
    first = scan(df, DIMS, "y", 0.25, cluster=cluster, min_n=5)

    changed = df.copy()
    explore = (first.splits == "explore").to_numpy()
    rng = np.random.default_rng(3)
    changed.loc[explore, "y"] = (rng.random(explore.sum()) < 0.3).astype(int)
    second = scan(changed, DIMS, "y", 0.25, cluster=cluster, min_n=5)

    # Explore decides WHICH cells are tested, so the two lists differ. But a
    # cell tested in both runs must get the same holdout numbers.
    columns = ["holdout_n", "holdout_successes", "p_value"]
    before = first.cells.set_index("cell")[columns]
    after = second.cells.set_index("cell")[columns]
    shared = before.index.intersection(after.index)
    assert len(shared) > 5
    assert set(before.index) != set(after.index)
    pd.testing.assert_frame_equal(before.loc[shared], after.loc[shared])


def test_confirm_outcomes_cannot_change_discoveries():
    df = planted_table(3, n=6000)
    first = scan(df, DIMS, "y", 0.25, holdout=0.4, confirm=0.2)
    changed = df.copy()
    in_confirm = (first.splits == "confirm").to_numpy()
    changed.loc[in_confirm, "y"] = 0
    second = scan(changed, DIMS, "y", 0.25, holdout=0.4, confirm=0.2)
    assert second.discoveries["cell"].tolist() == first.discoveries["cell"].tolist()
    assert not second.discoveries["confirmed"].any()


# ---------- symmetries ----------


@pytest.mark.parametrize("cluster", [None, "user"])
def test_flipping_the_outcome_mirrors_greater_and_less(cluster):
    """y -> 1 - y with threshold t -> 1 - t turns "greater" into "less" exactly."""
    df = planted_table(4)
    flipped = df.assign(y=1 - df["y"])
    high = scan(df, DIMS, "y", 0.25, min_lift=0.0, cluster=cluster, min_n=5)
    low = scan(flipped, DIMS, "y", 0.75, alternative="less", cluster=cluster, min_n=5)

    a = high.cells.set_index("cell")
    b = low.cells.set_index("cell")
    assert set(a.index) == set(b.index)
    b = b.loc[a.index]
    assert (a["holdout_n"] == b["holdout_n"]).all()
    assert a["holdout_rate"].to_numpy() == pytest.approx(1 - b["holdout_rate"].to_numpy())
    assert a["p_value"].to_numpy() == pytest.approx(b["p_value"].to_numpy(), rel=1e-9)
    assert (a["discovery"] == b["discovery"]).all()
    assert high.discoveries["cell"].tolist() == low.discoveries["cell"].tolist()


@pytest.mark.parametrize("cluster", [None, "user"])
def test_order_of_dims_does_not_matter(cluster):
    df = planted_table(5)
    first = scan(df, ["region", "tier", "channel"], "y", 0.25, cluster=cluster, min_n=5)
    second = scan(df, ["channel", "region", "tier"], "y", 0.25, cluster=cluster, min_n=5)

    def keyed(result):
        table = {}
        for row in result.cells.itertuples():
            key = frozenset(zip(row.dims, row.levels, strict=True))
            table[key] = (row.holdout_n, row.holdout_successes, row.discovery)
        return table

    assert keyed(first) == keyed(second)


def test_index_labels_do_not_matter():
    df = planted_table(6)
    plain = scan(df, DIMS, "y", 0.25)

    odd = df.copy()
    odd.index = [f"row-{i % 50}" for i in range(len(df))]  # strings, and repeated
    shuffled = scan(odd, DIMS, "y", 0.25)

    pd.testing.assert_frame_equal(plain.cells, shuffled.cells)
    assert shuffled.splits.index.equals(odd.index)
    assert (shuffled.splits.to_numpy() == plain.splits.to_numpy()).all()


def test_column_names_that_clash_with_internal_names():
    df = planted_table(7).rename(
        columns={"region": "n", "tier": "levels", "channel": "cell", "y": "successes"}
    )
    df = df.rename(columns={"user": "split"})
    result = scan(df, ["n", "levels", "cell"], "successes", 0.25, cluster="split", min_n=5)
    assert "n=north & levels=pro" in set(result.discoveries["cell"])


def test_integer_bool_and_categorical_dims_match_between_splits():
    rng = np.random.default_rng(8)
    n = 4000
    df = pd.DataFrame(
        {
            "store": rng.integers(1, 4, n),  # numpy int64 values
            "promo": rng.random(n) < 0.5,  # numpy bool values
            "size": pd.Categorical(
                rng.choice(["S", "M", "L"], n), categories=["S", "M", "L", "XL"]
            ),
        }
    )
    rate = np.where((df["store"] == 2) & df["promo"], 0.5, 0.3)
    df["y"] = (rng.random(n) < rate).astype(int)
    result = scan(df, ["store", "promo", "size"], "y", 0.3)
    assert "store=2 & promo=True" in set(result.discoveries["cell"])
    # Every tested cell was found on the holdout too (keys matched).
    assert (result.cells["holdout_n"] > 0).all()
    # The unused category XL never becomes a cell.
    assert not any("XL" in cell for cell in result.cells["cell"])


def test_max_tests_keeps_the_smallest_explore_pvalues():
    df = planted_table(9)
    full = scan(df, DIMS, "y", 0.25, min_n=1)
    capped = scan(df, DIMS, "y", 0.25, min_n=1, max_tests=4)

    explore = df[full.splits == "explore"]
    scored = []
    for row in full.cells.itertuples():
        n, k = row_counts(explore, row.dims, row.levels, "y")
        scored.append((binom_pvalues([k], [n], 0.25)[0], row.cell))
    # Every candidate reached the holdout in the full run (min_n=1), so the
    # four smallest explore p-values among them are the ones to keep.
    assert full.n_tested == full.n_candidates
    expected = {cell for _, cell in sorted(scored)[:4]}
    assert set(capped.cells["cell"]) == expected


# ---------- tiny and odd inputs ----------


def test_two_rows_is_enough_to_run():
    df = pd.DataFrame({"a": ["x", "x"], "y": [1, 0]})
    result = scan(df, ["a"], "y", 0.5, min_n=1)
    assert result.n_units == {"explore": 1, "holdout": 1, "confirm": 0}


def test_constant_outcomes():
    df = pd.DataFrame({"a": ["x", "y"] * 100, "y": 0})
    assert scan(df, ["a"], "y", 0.5).n_candidates == 0
    low = scan(df, ["a"], "y", 0.5, alternative="less")
    assert set(low.discoveries["cell"]) == {"a=x", "a=y"}


def test_a_dim_that_is_always_missing_gives_no_cells():
    df = pd.DataFrame({"a": ["x", "y"] * 100, "b": None, "y": [0, 1] * 100})
    result = scan(df, ["a", "b"], "y", 0.2)
    assert all("b=" not in cell for cell in result.cells["cell"])
    assert result.n_explored == 2


def test_explicit_bin_edges_through_scan():
    rng = np.random.default_rng(10)
    df = pd.DataFrame({"age": rng.integers(18, 80, 3000), "y": 0})
    df.loc[df["age"] < 30, "y"] = (rng.random((df["age"] < 30).sum()) < 0.6).astype(int)
    df.loc[df["age"] >= 30, "y"] = (rng.random((df["age"] >= 30).sum()) < 0.3).astype(int)
    result = scan(df, ["age"], "y", 0.3, bins={"age": [30, 50]})
    assert result.bin_edges["age"].tolist() == [-np.inf, 30, 50, np.inf]
    assert result.discoveries["cell"].tolist() == ["age=(-inf, 30)"]


def test_float_bin_count_gets_a_clear_error():
    df = pd.DataFrame({"age": np.arange(100.0), "y": [0, 1] * 50})
    with pytest.raises(ValueError, match="bin"):
        scan(df, ["age"], "y", 0.3, bins={"age": 4.0})


# ---------- binomial p-values are valid tests ----------


@settings(max_examples=150, deadline=None)
@given(
    n=st.integers(1, 200),
    bar=st.floats(0.02, 0.98),
    shrink=st.floats(0.0, 1.0).map(lambda x: round(x, 6)),  # no subnormal rates
    alpha=st.floats(0.001, 0.5),
    alternative=st.sampled_from(["greater", "less"]),
)
def test_null_pvalues_never_reject_too_often(n, bar, shrink, alpha, alternative):
    """Under any true rate on the null side of the bar, P(p <= alpha) <= alpha.

    The probability is computed exactly from the binomial distribution.
    """
    if alternative == "greater":
        true_rate = bar * shrink  # at or below the bar
    else:
        true_rate = bar + (1 - bar) * (1 - shrink)  # at or above the bar
    k = np.arange(n + 1)
    pvalues = binom_pvalues(k, np.full(n + 1, n), bar, alternative)
    rejection_probability = stats.binom.pmf(k[pvalues <= alpha], n, true_rate).sum()
    assert rejection_probability <= alpha + 1e-9


@settings(max_examples=100, deadline=None)
@given(n=st.integers(1, 500), bar=st.floats(0.01, 0.99))
def test_pvalues_move_the_right_way_as_successes_grow(n, bar):
    k = np.arange(n + 1)
    greater = binom_pvalues(k, np.full(n + 1, n), bar, "greater")
    less = binom_pvalues(k, np.full(n + 1, n), bar, "less")
    assert np.all(np.diff(greater) <= 1e-12)
    assert np.all(np.diff(less) >= -1e-12)


def test_pvalues_stay_accurate_for_large_cells():
    for k in (300_000, 301_000, 302_500):
        ours = binom_pvalues([k], [1_000_000], 0.3)[0]
        expected = stats.binomtest(k, 1_000_000, 0.3, alternative="greater").pvalue
        assert ours == pytest.approx(expected, rel=1e-8)
