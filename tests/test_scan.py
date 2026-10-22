import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from simdata import make_clustered_null, make_table

from segmentscan import DependenceWarning, ScanResult, scan

PLANTED = "a=a0 & b=b0"


def test_planted_cell_is_recovered_and_stable():
    df = make_table(6000, seed=0, planted=0.45)
    result = scan(df, ["a", "b", "c"], "y", 0.30)
    assert isinstance(result, ScanResult)
    found = result.discoveries.set_index("cell")
    assert PLANTED in found.index
    assert bool(found.loc[PLANTED, "stable"])
    assert found.loc[PLANTED, "holdout_rate"] == pytest.approx(0.45, abs=0.06)
    # It is the strongest finding.
    assert result.discoveries.iloc[0]["cell"] == PLANTED


def test_planted_cell_is_found_in_most_runs():
    hits = 0
    for seed in range(20):
        df = make_table(6000, seed=100 + seed, planted=0.45)
        result = scan(df, ["a", "b", "c"], "y", 0.30, seed=seed)
        hits += PLANTED in set(result.discoveries["cell"])
    assert hits >= 18


def test_alternative_less_finds_a_low_cell():
    df = make_table(6000, seed=3, planted=0.15)
    result = scan(df, ["a", "b", "c"], "y", 0.30, alternative="less")
    assert PLANTED in set(result.discoveries["cell"])
    assert (result.discoveries["lift"] < 0).all()


def test_holdout_counts_and_pvalues_match_a_plain_mask():
    """Check the vectorised groupby against the slow, obvious way."""
    df = make_table(3000, seed=4, planted=0.40)
    result = scan(df, ["a", "b", "c"], "y", 0.30, min_lift=0.02)
    holdout = df[result.splits == "holdout"]
    assert len(result.cells) > 5
    for row in result.cells.itertuples():
        mask = np.ones(len(holdout), dtype=bool)
        for dim, level in zip(row.dims, row.levels, strict=True):
            mask &= (holdout[dim] == level).to_numpy()
        n = int(mask.sum())
        k = int(holdout["y"].to_numpy()[mask].sum())
        assert (row.holdout_n, row.holdout_successes) == (n, k)
        expected = stats.binomtest(k, n, 0.32, alternative="greater").pvalue
        assert row.p_value == pytest.approx(expected)


def test_min_lift_raises_the_bar():
    df = make_table(6000, seed=5, planted=0.45)
    plain = scan(df, ["a", "b", "c"], "y", 0.30)
    strict = scan(df, ["a", "b", "c"], "y", 0.30, min_lift=0.10)
    assert strict.bar == pytest.approx(0.40)
    assert len(strict.discoveries) < len(plain.discoveries)
    assert (strict.discoveries["holdout_rate"] > 0.40).all()

    low = scan(df, ["a", "b", "c"], "y", 0.30, min_lift=0.10, alternative="less")
    assert low.bar == pytest.approx(0.20)


def test_every_discovery_passes_q_and_counts_add_up():
    df = make_table(4000, seed=6, planted=0.42)
    result = scan(df, ["a", "b", "c"], "y", 0.30)
    cells = result.cells
    assert (result.discoveries["p_adj"] <= 0.05).all()
    assert (cells["half1_n"] + cells["half2_n"] == cells["holdout_n"]).all()
    assert (cells["holdout_n"] >= 30).all()
    assert result.n_tested <= result.n_candidates <= result.n_explored
    assert cells["p_value"].is_monotonic_increasing


def test_same_seed_same_answer_and_seed_changes_split():
    df = make_table(3000, seed=7, planted=0.40)
    first = scan(df, ["a", "b", "c"], "y", 0.30, seed=11)
    second = scan(df, ["a", "b", "c"], "y", 0.30, seed=11)
    other = scan(df, ["a", "b", "c"], "y", 0.30, seed=12)
    pd.testing.assert_frame_equal(first.cells, second.cells)
    assert not first.splits.equals(other.splits)


def test_max_cells_stops_a_combinatorial_blow_up():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id_like": np.arange(2000), "y": rng.integers(0, 2, 2000)})
    with pytest.raises(ValueError, match="max_cells"):
        scan(df, ["id_like"], "y", 0.5, max_cells=500)


def test_max_tests_caps_the_holdout_tests():
    df = make_table(6000, seed=8, planted=0.45)
    full = scan(df, ["a", "b", "c"], "y", 0.30, min_n=1)
    capped = scan(df, ["a", "b", "c"], "y", 0.30, max_tests=5, min_n=1)
    assert full.n_tested > 5
    assert capped.n_tested == 5
    assert capped.n_candidates == full.n_candidates


def test_confirm_split_retests_discoveries():
    df = make_table(9000, seed=9, planted=0.45)
    result = scan(df, ["a", "b", "c"], "y", 0.30, holdout=0.4, confirm=0.2)
    assert result.n_units["confirm"] == 1800
    cells = result.cells
    assert {"confirm_n", "confirm_rate", "confirm_p_adj", "confirmed"} <= set(cells.columns)
    assert cells.loc[~cells["discovery"], "confirm_n"].isna().all()
    assert not cells.loc[~cells["discovery"], "confirmed"].any()
    planted = cells.set_index("cell").loc[PLANTED]
    assert bool(planted["confirmed"])
    assert "confirmed" in str(result)


def test_cluster_mode_counts_each_cluster_once_per_cell():
    df = make_clustered_null(300, 10, seed=1)
    result = scan(df, ["region", "plan", "device"], "y", 0.30, cluster="user_id", min_n=5)
    holdout_users = df.loc[result.splits == "holdout", "user_id"].nunique()
    assert result.n_units["holdout"] == holdout_users
    assert (result.cells["holdout_n"] <= holdout_users).all()
    # No user is in two splits.
    assert (result.splits.groupby(df["user_id"]).nunique() == 1).all()
    assert result.design_effect is not None and result.design_effect > 2


def test_row_level_tests_on_clustered_rows_warn():
    df = make_clustered_null(300, 10, seed=2)
    with pytest.warns(DependenceWarning, match="design effect"):
        scan(df, ["region", "plan"], "y", 0.30, cluster="user_id", aggregate=False, min_n=5)


def test_no_warning_for_independent_rows_in_clusters():
    df = make_table(3000, seed=10)
    df["user_id"] = np.repeat(np.arange(300), 10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scan(df, ["a", "b"], "y", 0.30, cluster="user_id", aggregate=False)


def test_missing_levels_are_left_out_and_bool_outcomes_work():
    df = make_table(3000, seed=11, planted=0.45)
    df.loc[df.index[:200], "c"] = None
    df["y"] = df["y"].astype(bool)
    result = scan(df, ["a", "b", "c"], "y", 0.30)
    assert not any("c=None" in cell or "c=nan" in cell for cell in result.cells["cell"])
    assert PLANTED in set(result.discoveries["cell"])


def test_no_candidates_gives_an_empty_but_valid_result():
    df = make_table(2000, seed=12)
    result = scan(df, ["a", "b"], "y", 0.90)
    assert result.n_candidates == 0
    assert result.n_tested == 0
    assert result.discoveries.empty
    assert "No discoveries." in str(result)


def test_summary_and_repr():
    df = make_table(6000, seed=13, planted=0.45)
    result = scan(df, ["a", "b", "c"], "y", 0.30, min_lift=0.02)
    text = str(result)
    assert text.startswith("ScanResult:")
    assert "(BH, q=0.05)" in text
    assert "threshold 0.300 + min_lift 0.020" in text
    assert PLANTED in text
    assert repr(result).startswith("<ScanResult explored=")
    assert len(result.discoveries) > 1
    assert "more (see result.discoveries)" in result.summary(max_rows=1)


def test_original_dataframe_is_not_changed():
    df = make_table(2000, seed=14)
    before = df.copy()
    scan(df, ["a", "b", "c"], "y", 0.30, bins=None)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"dims": []}, ValueError),
        ({"dims": ["a", "a"]}, ValueError),
        ({"dims": ["nope"]}, KeyError),
        ({"dims": ["a", "y"]}, ValueError),
        ({"threshold": 0.0}, ValueError),
        ({"threshold": 1.0}, ValueError),
        ({"min_lift": -0.1}, ValueError),
        ({"min_lift": 0.8}, ValueError),
        ({"q": 0.0}, ValueError),
        ({"holdout": 1.0}, ValueError),
        ({"holdout": 0.6, "confirm": 0.4}, ValueError),
        ({"method": "bonferroni"}, ValueError),
        ({"alternative": "two-sided"}, ValueError),
        ({"max_depth": 0}, ValueError),
        ({"min_n": 0}, ValueError),
        ({"max_tests": 0}, ValueError),
        ({"cluster": "a"}, ValueError),
    ],
)
def test_bad_arguments_raise(kwargs, error):
    df = make_table(500, seed=15)
    arguments = {"dims": ["a", "b"], "outcome": "y", "threshold": 0.3}
    arguments.update(kwargs)
    with pytest.raises(error):
        scan(df, **arguments)


def test_bad_outcomes_raise():
    df = make_table(500, seed=16)
    with pytest.raises(ValueError, match="only 0/1"):
        scan(df.assign(y=df["y"] * 2), ["a"], "y", 0.3)
    with pytest.raises(ValueError, match="missing"):
        scan(df.assign(y=df["y"].astype(float).where(df.index > 0)), ["a"], "y", 0.3)
    with pytest.raises(TypeError):
        scan(df.to_dict(), ["a"], "y", 0.3)


def test_summary_says_discovery_for_exactly_one():
    df = make_table(6000, seed=0, planted=0.45)
    result = scan(df, ["a", "b"], "y", 0.30, min_lift=0.05)
    assert len(result.discoveries) == 1
    assert "1 discovery (" in str(result)
