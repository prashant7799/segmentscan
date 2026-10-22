"""The main entry point: ``scan()``.

The protocol, in five steps:

1. Split units (rows, or clusters) at random into explore and holdout.
2. Fit any numeric bins on the explore rows only.
3. Explore: count every 1-, 2- and 3-way cell. Keep cells with enough units
   whose rate beats the bar. These are the candidates.
4. Holdout: test each candidate once, with a one-sided exact binomial test
   against the bar. Adjust the p-values for the number of tests (BH, BY or
   Holm). Cells at or below ``q`` are discoveries.
5. Report split-half stability, and re-test discoveries on an optional
   third split.

The holdout outcomes never help choose what gets tested. (Only the holdout
cell sizes are used, to drop cells smaller than ``min_n``.) That is what
makes the holdout p-values honest.

With ``cluster=`` and ``aggregate=True``, every split is counted the same
way: each cluster counts once per cell, through one random row of that
cluster in the cell.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .binning import apply_bins, edges_from_spec
from .result import ScanResult
from .split import CONFIRM, EXPLORE, HOLDOUT, assign_halves, assign_splits
from .stats import (
    ALTERNATIVES,
    METHODS,
    DependenceWarning,
    adjust_pvalues,
    binom_pvalues,
    design_effect,
)

#: Above this design effect, row-level tests on clustered data get a warning.
DESIGN_EFFECT_WARNING_LEVEL = 1.2

CELL_COLUMNS = [
    "cell",
    "depth",
    "explore_n",
    "explore_rate",
    "holdout_n",
    "holdout_successes",
    "holdout_rate",
    "lift",
    "p_value",
    "p_adj",
    "discovery",
    "half1_n",
    "half1_rate",
    "half2_n",
    "half2_rate",
    "stable",
]
CONFIRM_COLUMNS = ["confirm_n", "confirm_rate", "confirm_p_adj", "confirmed"]


def scan(
    df: pd.DataFrame,
    dims: Sequence[str],
    outcome: str,
    threshold: float,
    *,
    holdout: float = 0.5,
    q: float = 0.05,
    method: str = "bh",
    alternative: str = "greater",
    min_lift: float = 0.0,
    cluster: str | None = None,
    aggregate: bool = True,
    max_depth: int = 3,
    min_n: int = 30,
    max_cells: int = 100_000,
    max_tests: int | None = None,
    bins: Mapping[str, int | Sequence[float]] | None = None,
    confirm: float = 0.0,
    seed: int | None = 0,
) -> ScanResult:
    """Find cells (segments) whose 0/1 rate beats a threshold, with FDR control.

    Parameters
    ----------
    df:
        One row per item. Must hold the ``dims`` columns and the ``outcome``.
    dims:
        Category columns to cut by, for example ``["device", "country"]``.
        Cells combine up to ``max_depth`` of them.
    outcome:
        A 0/1 (or True/False) column with no missing values.
    threshold:
        The rate a cell must beat, for example the overall rate. Choose it
        before you look at the data.
    holdout:
        Share of units held back for testing. Default 0.5.
    q:
        Allowed false-discovery rate (for Holm: the family-wise error rate).
    method:
        ``"bh"`` (Benjamini-Hochberg, default), ``"by"`` (Benjamini-Yekutieli,
        valid under any dependence) or ``"holm"`` (family-wise control).
    alternative:
        ``"greater"`` finds cells above the bar; ``"less"`` finds cells below it.
    min_lift:
        How far past the threshold a cell must be to matter. The test's null
        rate (the bar) is ``threshold + min_lift`` for "greater" and
        ``threshold - min_lift`` for "less".
    cluster:
        Column that groups dependent rows, such as ``"user_id"``. Clusters are
        never split between explore and holdout.
    aggregate:
        Only used with ``cluster``. When True (default), each cluster counts
        once per cell, on every split: one of its rows in that cell is picked
        at random. When False, every row counts and a ``DependenceWarning``
        is issued if the rows look clustered.
    max_depth:
        Largest number of dims combined in one cell. Default 3.
    min_n:
        Smallest cell to consider, in units, on both explore and holdout.
    max_cells:
        Safety cap on the number of explore cells. Exceeding it raises a
        ``ValueError`` instead of running for a very long time.
    max_tests:
        If set, only the ``max_tests`` candidates with the smallest explore
        p-values go to the holdout. Fewer tests means more power per test.
    bins:
        Numeric dims to bin, as ``{column: n_quantile_bins}`` or
        ``{column: [edge, edge, ...]}``. Edges are fitted on explore only.
    confirm:
        Share of units held back as a third split. Discoveries are re-tested
        there and marked ``confirmed``. Default 0 (off).
    seed:
        Seed for the random splits. Default 0, so runs repeat exactly.

    Returns
    -------
    ScanResult
        ``result.discoveries`` holds the cells that passed. ``result.cells``
        holds every tested cell. ``result.splits`` says which split each
        input row went to.
    """
    dims = _check_arguments(
        df,
        dims,
        outcome,
        threshold,
        holdout=holdout,
        q=q,
        method=method,
        alternative=alternative,
        min_lift=min_lift,
        cluster=cluster,
        max_depth=max_depth,
        min_n=min_n,
        max_cells=max_cells,
        max_tests=max_tests,
        bins=bins,
        confirm=confirm,
    )
    bar = _bar(threshold, min_lift, alternative)
    rng = np.random.default_rng(seed)
    # The column that defines a unit when counting cells. None means "every
    # row is a unit". With a cluster column and aggregate=True, each cluster
    # counts once per cell.
    unit_cluster = cluster if (cluster is not None and aggregate) else None

    # Work on a small copy: only the columns we need, with a fresh 0..n-1 index.
    columns = [*dims, outcome] + ([cluster] if cluster is not None else [])
    data = df[columns].reset_index(drop=True).copy()
    data[outcome] = data[outcome].astype(int)

    # Step 1: split units into explore / holdout / confirm.
    split = assign_splits(data, cluster, holdout, confirm, rng)
    is_explore = split == EXPLORE

    # Step 2: bin numeric columns with edges learned from explore rows only.
    bin_edges: dict[str, np.ndarray] = {}
    for column, spec in (bins or {}).items():
        edges = edges_from_spec(data.loc[is_explore, column], spec)
        bin_edges[column] = edges
        data[column] = apply_bins(data[column], edges)

    explore_rows = data[is_explore]
    holdout_rows = data[split == HOLDOUT]
    confirm_rows = data[split == CONFIRM]

    deff = None
    if cluster is not None:
        deff = design_effect(explore_rows, outcome, cluster)
        if not aggregate and deff > DESIGN_EFFECT_WARNING_LEVEL:
            warnings.warn(
                f"Rows within {cluster!r} look correlated (design effect {deff:.2f} on the "
                "explore split). With aggregate=False every row counts as a separate unit, "
                "so p-values will be too small. Use aggregate=True (the default).",
                DependenceWarning,
                stacklevel=2,
            )

    # Step 3: explore. Count every cell and keep the promising ones.
    if unit_cluster is not None:
        # Shuffle once, so that "the first row of each cluster in a cell" is
        # a random row. The holdout and confirm rows are shuffled the same way.
        explore_rows = explore_rows.iloc[rng.permutation(len(explore_rows))]
    combos = dimension_combinations(dims, max_depth)
    candidates, n_explored = _explore(
        explore_rows, combos, outcome, unit_cluster, bar, alternative, min_n, max_cells
    )
    n_candidates = len(candidates)
    if max_tests is not None and n_candidates > max_tests:
        candidates = candidates.sort_values(["explore_p", "cell"], kind="mergesort")
        candidates = candidates.head(max_tests).reset_index(drop=True)

    # Step 4: test candidates on the holdout.
    holdout_rows = holdout_rows.iloc[rng.permutation(len(holdout_rows))]
    halves = assign_halves(holdout_rows, cluster, rng)

    table = candidates.copy()
    parts = [
        ("holdout", holdout_rows),
        ("half1", holdout_rows[halves == 1]),
        ("half2", holdout_rows[halves == 2]),
    ]
    for prefix, rows in parts:
        n, successes = _count_candidates(candidates, rows, outcome, unit_cluster)
        table[f"{prefix}_n"] = n
        table[f"{prefix}_successes"] = successes
    table = table[table["holdout_n"] >= min_n].reset_index(drop=True)

    table["holdout_rate"] = _rate(table["holdout_successes"], table["holdout_n"])
    table["lift"] = table["holdout_rate"] - threshold
    table["p_value"] = binom_pvalues(
        table["holdout_successes"].to_numpy(), table["holdout_n"].to_numpy(), bar, alternative
    )
    table["p_adj"] = adjust_pvalues(table["p_value"].to_numpy(), method)
    table["discovery"] = table["p_adj"] <= q

    # Step 5a: split-half stability. Does the cell beat the bar in both halves?
    table["half1_rate"] = _rate(table["half1_successes"], table["half1_n"])
    table["half2_rate"] = _rate(table["half2_successes"], table["half2_n"])
    table["stable"] = _beats(table["half1_rate"], bar, alternative) & _beats(
        table["half2_rate"], bar, alternative
    )

    # Step 5b: optional third split. Re-test only the discoveries.
    if confirm > 0:
        confirm_rows = confirm_rows.iloc[rng.permutation(len(confirm_rows))]
        table = _add_confirmation(
            table, confirm_rows, outcome, unit_cluster, bar, alternative, method, q
        )

    table = table.sort_values(["p_value", "cell"], kind="mergesort").reset_index(drop=True)
    ordered = CELL_COLUMNS + (CONFIRM_COLUMNS if confirm > 0 else []) + ["dims", "levels"]
    table = table[ordered]

    units = _units_per_split(data, split, cluster)
    rows_per_split = {name: int(np.sum(split == name)) for name in (EXPLORE, HOLDOUT, CONFIRM)}
    return ScanResult(
        cells=table,
        n_explored=n_explored,
        n_candidates=n_candidates,
        threshold=float(threshold),
        min_lift=float(min_lift),
        bar=float(bar),
        q=float(q),
        method=method,
        alternative=alternative,
        cluster=cluster,
        aggregate=bool(aggregate),
        n_units=units,
        n_rows=rows_per_split,
        bin_edges=bin_edges,
        design_effect=deff,
        splits=pd.Series(split, index=df.index, name="split"),
        seed=seed,
    )


def dimension_combinations(dims: Sequence[str], max_depth: int) -> list[tuple[str, ...]]:
    """All 1-way, 2-way, ... up to ``max_depth``-way combinations of dims."""
    combos: list[tuple[str, ...]] = []
    for depth in range(1, min(max_depth, len(dims)) + 1):
        combos.extend(itertools.combinations(dims, depth))
    return combos


def count_cells(rows: pd.DataFrame, combo: Sequence[str], outcome: str) -> pd.DataFrame:
    """Rows and successes in every non-empty cell of one combination of dims.

    Returns columns ``levels`` (a tuple with one value per dim), ``n`` and
    ``successes``. Rows with a missing value in any of these dims are left out.
    """
    if rows.empty:
        return pd.DataFrame(
            {"levels": pd.Series([], dtype=object), "n": [], "successes": []}
        ).astype({"n": int, "successes": int})

    grouped = rows.groupby(list(combo), observed=True, sort=False, dropna=True)[outcome]
    counts = grouped.agg(n="size", successes="sum")

    levels = []
    for key in counts.index:
        parts = key if isinstance(key, tuple) else (key,)
        levels.append(tuple(_plain(value) for value in parts))

    return pd.DataFrame(
        {
            "levels": levels,
            "n": counts["n"].to_numpy(dtype=int),
            "successes": counts["successes"].to_numpy(dtype=int),
        }
    )


def cell_label(dims: Sequence[str], levels: Sequence[object]) -> str:
    """Readable name of a cell, such as ``"device=tablet & country=NZ"``."""
    return " & ".join(f"{dim}={level}" for dim, level in zip(dims, levels, strict=True))


def _one_row_per_unit(
    rows: pd.DataFrame, combo: Sequence[str], cluster: str | None
) -> pd.DataFrame:
    """Rows to count for one combination of dims.

    Without a cluster column every row is a unit, so all rows are kept.
    With one, each cluster keeps only its first row in each cell. The caller
    shuffles ``rows`` beforehand, so that first row is a random one.
    """
    if cluster is None:
        return rows
    return rows.drop_duplicates(subset=[*combo, cluster], keep="first")


def _explore(
    rows: pd.DataFrame,
    combos: list[tuple[str, ...]],
    outcome: str,
    cluster: str | None,
    bar: float,
    alternative: str,
    min_n: int,
    max_cells: int,
) -> tuple[pd.DataFrame, int]:
    """Count all cells on the explore split and keep the promising ones."""
    records = []
    n_explored = 0
    for combo in combos:
        counts = count_cells(_one_row_per_unit(rows, combo, cluster), combo, outcome)
        n_explored += len(counts)
        if n_explored > max_cells:
            raise ValueError(
                f"more than max_cells={max_cells} cells on the explore split. "
                "Use fewer dims, a smaller max_depth, coarser bins, or raise max_cells."
            )
        rates = counts["successes"] / counts["n"]
        promising = (counts["n"] >= min_n) & _beats(rates, bar, alternative)
        kept = counts[promising]
        for levels, n, successes in zip(kept["levels"], kept["n"], kept["successes"], strict=True):
            records.append(
                {
                    "cell": cell_label(combo, levels),
                    "depth": len(combo),
                    "dims": combo,
                    "levels": levels,
                    "explore_n": int(n),
                    "explore_successes": int(successes),
                }
            )

    columns = ["cell", "depth", "dims", "levels", "explore_n", "explore_successes"]
    candidates = pd.DataFrame(records, columns=columns)
    candidates = candidates.astype({"depth": int, "explore_n": int, "explore_successes": int})
    candidates["explore_rate"] = _rate(candidates["explore_successes"], candidates["explore_n"])
    candidates["explore_p"] = binom_pvalues(
        candidates["explore_successes"].to_numpy(),
        candidates["explore_n"].to_numpy(),
        bar,
        alternative,
    )
    return candidates, n_explored


def _count_candidates(
    candidates: pd.DataFrame,
    rows: pd.DataFrame,
    outcome: str,
    cluster: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Units and successes for each candidate cell, counted on ``rows``.

    One groupby per combination of dims, then a dictionary lookup per cell.
    With ``cluster`` set, each cluster keeps one row per cell: the first one,
    which is random because the caller shuffled ``rows``.
    """
    n = np.zeros(len(candidates), dtype=int)
    successes = np.zeros(len(candidates), dtype=int)

    positions_by_combo: dict[tuple[str, ...], list[int]] = {}
    for position, combo in enumerate(candidates["dims"]):
        positions_by_combo.setdefault(combo, []).append(position)
    all_levels = candidates["levels"].tolist()

    for combo, positions in positions_by_combo.items():
        counts = count_cells(_one_row_per_unit(rows, combo, cluster), combo, outcome)
        lookup = {}
        for levels, cell_n, cell_successes in zip(
            counts["levels"], counts["n"], counts["successes"], strict=True
        ):
            lookup[levels] = (cell_n, cell_successes)
        for position in positions:
            n[position], successes[position] = lookup.get(all_levels[position], (0, 0))
    return n, successes


def _add_confirmation(
    table: pd.DataFrame,
    confirm_rows: pd.DataFrame,
    outcome: str,
    cluster: str | None,
    bar: float,
    alternative: str,
    method: str,
    q: float,
) -> pd.DataFrame:
    """Re-test the discoveries on the confirm split and add the results."""
    table = table.copy()
    table["confirm_n"] = pd.array([pd.NA] * len(table), dtype="Int64")
    table["confirm_rate"] = np.nan
    table["confirm_p_adj"] = np.nan
    table["confirmed"] = False

    found = table.index[table["discovery"]]
    if len(found) == 0:
        return table

    n, successes = _count_candidates(table.loc[found], confirm_rows, outcome, cluster)
    pvalues = binom_pvalues(successes, n, bar, alternative)
    adjusted = adjust_pvalues(pvalues, method)
    table.loc[found, "confirm_n"] = n
    table.loc[found, "confirm_rate"] = _rate(pd.Series(successes), pd.Series(n)).to_numpy()
    table.loc[found, "confirm_p_adj"] = adjusted
    table.loc[found, "confirmed"] = adjusted <= q
    return table


def _units_per_split(data: pd.DataFrame, split: np.ndarray, cluster: str | None) -> dict[str, int]:
    units = {}
    for name in (EXPLORE, HOLDOUT, CONFIRM):
        mask = split == name
        if cluster is None:
            units[name] = int(mask.sum())
        else:
            units[name] = int(data.loc[mask, cluster].nunique())
    return units


def _bar(threshold: float, min_lift: float, alternative: str) -> float:
    """The null rate each cell is tested against."""
    if alternative == "greater":
        return threshold + min_lift
    return threshold - min_lift


def _rate(successes: pd.Series, n: pd.Series) -> pd.Series:
    """successes / n, with NaN where n is 0."""
    n_float = n.astype(float).where(n > 0)
    return successes.astype(float) / n_float


def _beats(rates: pd.Series, bar: float, alternative: str) -> pd.Series:
    """True where the rate is past the bar in the tested direction (NaN is False)."""
    if alternative == "greater":
        return rates > bar
    return rates < bar


def _plain(value: object) -> object:
    """Turn numpy scalars into plain Python values, so cell keys always match."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _check_arguments(
    df,
    dims,
    outcome,
    threshold,
    *,
    holdout,
    q,
    method,
    alternative,
    min_lift,
    cluster,
    max_depth,
    min_n,
    max_cells,
    max_tests,
    bins,
    confirm,
) -> list[str]:
    """Raise a clear error for bad inputs. Returns dims as a list."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if isinstance(dims, str):
        dims = [dims]
    dims = list(dims)
    if not dims:
        raise ValueError("dims must name at least one column")
    if len(set(dims)) != len(dims):
        raise ValueError("dims must not repeat a column")

    needed = [*dims, outcome] + ([cluster] if cluster is not None else [])
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise KeyError(f"columns not found in df: {missing}")
    if outcome in dims:
        raise ValueError("outcome must not also be a dim")
    if cluster is not None and (cluster in dims or cluster == outcome):
        raise ValueError("cluster must be a separate column from dims and outcome")

    y = df[outcome]
    if y.isna().any():
        raise ValueError(f"outcome {outcome!r} has missing values")
    if not set(pd.unique(y)) <= {0, 1}:
        raise ValueError(f"outcome {outcome!r} must contain only 0/1 or True/False")
    if cluster is not None and df[cluster].isna().any():
        raise ValueError(f"cluster column {cluster!r} has missing values")

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")
    if min_lift < 0:
        raise ValueError("min_lift must be >= 0")
    bar = _bar(threshold, min_lift, alternative)
    if not 0 < bar < 1:
        raise ValueError(f"threshold and min_lift give a bar of {bar}; it must be in (0, 1)")
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")
    if not 0 < holdout < 1:
        raise ValueError("holdout must be strictly between 0 and 1")
    if not 0 <= confirm < 1 or holdout + confirm >= 1:
        raise ValueError("confirm must be >= 0 and holdout + confirm must be below 1")
    if int(max_depth) != max_depth or max_depth < 1:
        raise ValueError("max_depth must be a whole number >= 1")
    if int(min_n) != min_n or min_n < 1:
        raise ValueError("min_n must be a whole number >= 1")
    if int(max_cells) != max_cells or max_cells < 1:
        raise ValueError("max_cells must be a whole number >= 1")
    if max_tests is not None and (int(max_tests) != max_tests or max_tests < 1):
        raise ValueError("max_tests must be None or a whole number >= 1")

    for column in bins or {}:
        if column not in dims:
            raise ValueError(f"binned column {column!r} must also be listed in dims")
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"binned column {column!r} must be numeric")
    return dims
