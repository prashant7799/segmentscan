import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from segmentscan import adjust_pvalues, binom_pvalues, design_effect

pvalue_lists = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=40
)


# ---------- reference rules, written straight from the papers ----------


def step_up_rejections(p, q, factor=1.0):
    """BH step-up (BY when factor = sum 1/i): reject ranks 1..k, where k is
    the largest rank with p_(k) <= k * q / (m * factor)."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p, kind="stable")
    largest_rank = 0
    for rank in range(1, m + 1):
        if p[order[rank - 1]] <= rank * q / (m * factor):
            largest_rank = rank
    rejected = np.zeros(m, dtype=bool)
    rejected[order[:largest_rank]] = True
    return rejected


def holm_rejections(p, q):
    """Holm step-down: walk up the sorted p-values, stop at the first failure."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p, kind="stable")
    rejected = np.zeros(m, dtype=bool)
    for i, index in enumerate(order):
        if p[index] <= q / (m - i):
            rejected[index] = True
        else:
            break
    return rejected


def row_wise_rejections(p, q):
    """The buggy rule: compare each p-value only with its own line."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p, kind="stable")
    rejected = np.zeros(m, dtype=bool)
    for rank, index in enumerate(order, start=1):
        rejected[index] = p[index] <= rank * q / m
    return rejected


def assert_between(ours, reference, q):
    """Same rejections as the reference rule, allowing for float rounding.

    A p-value that sits exactly on a line can round either way, so we only
    ask: everything the rule rejects at a slightly smaller q, we reject; and
    we reject nothing that the rule keeps at a slightly larger q.
    """
    surely_rejected = reference(q * (1 - 1e-9))
    maybe_rejected = reference(q * (1 + 1e-9))
    assert np.all(ours[surely_rejected])
    assert not np.any(ours[~maybe_rejected])


# ---------- binomial p-values ----------


@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=300),
    data=st.data(),
    rate=st.floats(min_value=0.01, max_value=0.99),
    alternative=st.sampled_from(["greater", "less"]),
)
def test_binom_pvalues_match_scipy_binomtest(n, data, rate, alternative):
    k = data.draw(st.integers(min_value=0, max_value=n))
    ours = binom_pvalues([k], [n], rate, alternative)[0]
    reference = stats.binomtest(k, n, rate, alternative=alternative).pvalue
    assert ours == pytest.approx(reference, rel=1e-9, abs=1e-12)


def test_binom_pvalues_obvious_cases():
    # 0 successes out of 10 is no evidence of a rate above 0.3.
    assert binom_pvalues([0], [10], 0.3, "greater")[0] == pytest.approx(1.0)
    # 10 out of 10 is strong evidence.
    assert binom_pvalues([10], [10], 0.3, "greater")[0] == pytest.approx(0.3**10)
    # An empty cell gives p = 1 either way.
    assert binom_pvalues([0], [0], 0.3, "greater")[0] == pytest.approx(1.0)
    assert binom_pvalues([0], [0], 0.3, "less")[0] == pytest.approx(1.0)


def test_binom_pvalues_reject_bad_input():
    with pytest.raises(ValueError):
        binom_pvalues([5], [4], 0.3)
    with pytest.raises(ValueError):
        binom_pvalues([1], [4], 1.0)
    with pytest.raises(ValueError):
        binom_pvalues([1], [4], 0.3, alternative="two-sided")


# ---------- BH / BY / Holm ----------


def test_bh_fixes_the_row_wise_bug():
    """The case that broke the old row-by-row rule.

    Lines at q = 0.05 are 0.0167, 0.0333 and 0.05. The row-wise rule passes
    only the last p-value. The real step-up rule passes all three, because
    the largest rank under its line (rank 3) carries ranks 1 and 2 with it.
    """
    p = [0.02, 0.04, 0.045]
    assert row_wise_rejections(p, 0.05).tolist() == [False, False, True]
    adjusted = adjust_pvalues(p, "bh")
    assert adjusted == pytest.approx([0.045, 0.045, 0.045])
    assert (adjusted <= 0.05).tolist() == [True, True, True]


@settings(max_examples=300, deadline=None)
@given(p=pvalue_lists, q=st.floats(min_value=0.001, max_value=0.5))
def test_bh_matches_step_up_rule(p, q):
    ours = adjust_pvalues(p, "bh") <= q
    assert_between(ours, lambda level: step_up_rejections(p, level), q)


@settings(max_examples=300, deadline=None)
@given(p=pvalue_lists, q=st.floats(min_value=0.001, max_value=0.5))
def test_by_matches_step_up_rule_with_harmonic_factor(p, q):
    harmonic = sum(1.0 / i for i in range(1, len(p) + 1))
    ours = adjust_pvalues(p, "by") <= q
    assert_between(ours, lambda level: step_up_rejections(p, level, factor=harmonic), q)


@settings(max_examples=300, deadline=None)
@given(p=pvalue_lists, q=st.floats(min_value=0.001, max_value=0.5))
def test_holm_matches_step_down_rule(p, q):
    ours = adjust_pvalues(p, "holm") <= q
    assert_between(ours, lambda level: holm_rejections(p, level), q)


@settings(max_examples=200, deadline=None)
@given(p=pvalue_lists)
def test_bh_and_by_parity_with_scipy(p):
    for method in ("bh", "by"):
        expected = stats.false_discovery_control(np.asarray(p), method=method)
        assert adjust_pvalues(p, method) == pytest.approx(expected)


@settings(max_examples=200, deadline=None)
@given(p=pvalue_lists, method=st.sampled_from(["bh", "by", "holm"]))
def test_adjusted_pvalues_are_valid_and_keep_order(p, method):
    raw = np.asarray(p)
    adjusted = adjust_pvalues(raw, method)
    assert np.all(adjusted >= raw - 1e-12)
    assert np.all((adjusted >= 0) & (adjusted <= 1))
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_methods_rank_from_least_to_most_strict():
    p = np.array([0.001, 0.008, 0.012, 0.03, 0.04, 0.2, 0.5])
    bh = adjust_pvalues(p, "bh")
    by = adjust_pvalues(p, "by")
    holm = adjust_pvalues(p, "holm")
    assert np.all(bh <= by + 1e-12)
    assert (bh <= 0.05).sum() >= (by <= 0.05).sum()
    assert (bh <= 0.05).sum() >= (holm <= 0.05).sum()


def test_holm_known_values():
    # m = 3: multipliers 3, 2, 1 on the sorted p-values, then a running max.
    adjusted = adjust_pvalues([0.01, 0.04, 0.03], "holm")
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_adjust_pvalues_input_checks():
    assert adjust_pvalues([], "bh").size == 0
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, 1.2], "bh")
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, np.nan], "bh")
    with pytest.raises(ValueError):
        adjust_pvalues([0.1], "bonferroni")
    with pytest.raises(ValueError):
        adjust_pvalues([[0.1, 0.2]], "bh")


# ---------- design effect ----------


def test_design_effect_is_about_one_for_independent_rows():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"cluster": np.repeat(np.arange(500), 10), "y": (rng.random(5000) < 0.3).astype(int)}
    )
    assert design_effect(df, "y", "cluster") == pytest.approx(1.0, abs=0.15)


def test_design_effect_is_large_for_clustered_rows():
    rng = np.random.default_rng(0)
    cluster_rate = rng.beta(0.6, 1.4, 500)  # intraclass correlation 1/3
    cluster = np.repeat(np.arange(500), 10)
    df = pd.DataFrame({"cluster": cluster, "y": (rng.random(5000) < cluster_rate[cluster])})
    # Theory: 1 + (10 - 1) / 3 = 4.
    assert design_effect(df, "y", "cluster") == pytest.approx(4.0, rel=0.2)


def test_design_effect_edge_cases():
    singletons = pd.DataFrame({"cluster": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    assert design_effect(singletons, "y", "cluster") == 1.0
    one_cluster = pd.DataFrame({"cluster": [1, 1, 1], "y": [0, 1, 0]})
    assert design_effect(one_cluster, "y", "cluster") == 1.0
    constant = pd.DataFrame({"cluster": [1, 1, 2, 2], "y": [1, 1, 1, 1]})
    assert design_effect(constant, "y", "cluster") == 1.0
    with pytest.raises(KeyError):
        design_effect(constant, "y", "missing")
