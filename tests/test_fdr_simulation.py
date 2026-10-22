"""Calibration by simulation: does the protocol keep its promises?

Each test runs about 200 seeded scans on synthetic data where we know the
truth, so the outcome is the same on every run. A "false discovery" is a
discovered cell whose true rate is at or below the bar.

All three dims overlap (a 1-way cell contains 2-way cells), so these runs
also check BH on the kind of positively dependent tests that scan() makes.
"""

import numpy as np
import pytest
from simdata import (
    make_binary_null,
    make_clustered_null,
    make_table,
    make_unequal_clusters_null,
    true_cell_rate,
)

from segmentscan import scan

N_SIMULATIONS = 200
DIMS = ["a", "b", "c"]


def false_discovery_proportions(method, q, planted, n_rows=2000):
    proportions = []
    planted_found = 0
    for run in range(N_SIMULATIONS):
        df = make_table(n_rows, seed=10_000 + run, planted=planted)
        result = scan(df, DIMS, "y", 0.30, q=q, method=method, seed=run)
        found = result.discoveries
        n_false = 0
        for dims, levels in zip(found["dims"], found["levels"], strict=True):
            if true_cell_rate(dims, levels, planted) <= result.bar:
                n_false += 1
        proportions.append(n_false / max(len(found), 1))
        planted_found += "a=a0 & b=b0" in set(found["cell"])
    return np.array(proportions), planted_found / N_SIMULATIONS


@pytest.mark.parametrize("method", ["bh", "by", "holm"])
@pytest.mark.parametrize("q", [0.05, 0.10])
def test_null_fdr_is_at_or_below_q(method, q):
    """Global null: no cell is special, so any discovery is false.

    Here FDR equals the chance of at least one discovery.
    """
    proportions, _ = false_discovery_proportions(method, q, planted=None)
    assert proportions.mean() <= q


@pytest.mark.parametrize("method", ["bh", "by"])
def test_fdr_with_a_real_signal_is_at_or_below_q(method):
    """One planted cell. Its parents (a=a0, b=b0) are truly above the bar
    too, so they count as true discoveries. Everything else is null."""
    q = 0.10
    proportions, power = false_discovery_proportions(method, q, planted=0.40)
    assert proportions.mean() <= q
    assert power > 0.1  # the scan still finds the planted cell sometimes


def test_clustered_null_inflates_row_level_discoveries():
    """The demo: rows from the same user are not independent.

    Treated as independent rows, a no-effect dataset produces false
    discoveries far more often than q. Counting users (cluster=) fixes it.
    """
    q = 0.10
    runs = 100
    row_level_runs_with_false_finds = 0
    cluster_level_runs_with_false_finds = 0
    for run in range(runs):
        df = make_clustered_null(400, 10, seed=20_000 + run)
        dims = ["region", "plan", "device"]
        by_rows = scan(df, dims, "y", 0.30, q=q, min_n=10, seed=run)
        by_users = scan(df, dims, "y", 0.30, q=q, min_n=10, seed=run, cluster="user_id")
        row_level_runs_with_false_finds += len(by_rows.discoveries) > 0
        cluster_level_runs_with_false_finds += len(by_users.discoveries) > 0

    row_level_fdr = row_level_runs_with_false_finds / runs
    cluster_level_fdr = cluster_level_runs_with_false_finds / runs
    assert row_level_fdr > 3 * q
    assert cluster_level_fdr <= q


def share_of_runs_with_discoveries(make_table_for_run, runs, dims, **scan_options):
    """Under a global null every discovery is false, so FDR = P(any discovery)."""
    runs_with_discoveries = 0
    for run in range(runs):
        result = scan(make_table_for_run(run), dims, "y", seed=run, **scan_options)
        runs_with_discoveries += len(result.discoveries) > 0
    return runs_with_discoveries / runs


def test_min_lift_is_part_of_the_null():
    """Every row has rate 0.30. Threshold 0.25 plus min_lift 0.05 puts the bar
    at 0.30, so every cell sits exactly on the null boundary and any discovery
    is false. If min_lift were only a filter on the observed lift, cells at
    0.30 would pass all the time."""
    q = 0.10
    fdr = share_of_runs_with_discoveries(
        lambda run: make_table(2000, seed=30_000 + run),
        N_SIMULATIONS,
        DIMS,
        threshold=0.25,
        min_lift=0.05,
        q=q,
    )
    assert fdr <= q


def test_null_fdr_for_cells_below_the_bar():
    q = 0.10
    fdr = share_of_runs_with_discoveries(
        lambda run: make_table(2000, seed=40_000 + run),
        N_SIMULATIONS,
        DIMS,
        threshold=0.30,
        alternative="less",
        q=q,
    )
    assert fdr <= q


def test_null_fdr_with_many_overlapping_cells():
    """Five yes/no dims at depth 3: 130 cells, each overlapping many others."""
    q = 0.10
    fdr = share_of_runs_with_discoveries(
        lambda run: make_binary_null(2000, 5, seed=50_000 + run),
        100,
        [f"d{i}" for i in range(5)],
        threshold=0.30,
        q=q,
    )
    assert fdr <= q


def test_null_fdr_with_unequal_cluster_sizes():
    """Users have 1 to 15 rows. One random row per user per cell keeps it valid."""
    q = 0.10
    fdr = share_of_runs_with_discoveries(
        lambda run: make_unequal_clusters_null(500, 15, seed=60_000 + run),
        N_SIMULATIONS,
        ["region", "device"],
        threshold=0.30,
        cluster="user_id",
        min_n=10,
        q=q,
    )
    assert fdr <= q
