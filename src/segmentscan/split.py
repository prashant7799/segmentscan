"""Random splits of units into explore, holdout and (optional) confirm parts.

A unit is one row, or one cluster when ``cluster=`` is given. All rows of a
cluster always land in the same part. Otherwise one user could appear in
both explore and holdout, and the holdout would no longer be fresh data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXPLORE = "explore"
HOLDOUT = "holdout"
CONFIRM = "confirm"


def unit_codes(data: pd.DataFrame, cluster: str | None) -> tuple[np.ndarray, int]:
    """Give each row the integer code of its unit, plus the number of units."""
    if cluster is None:
        return np.arange(len(data)), len(data)
    codes, uniques = pd.factorize(data[cluster], sort=False)
    return codes, len(uniques)


def assign_splits(
    data: pd.DataFrame,
    cluster: str | None,
    holdout: float,
    confirm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Label every row ``"explore"``, ``"holdout"`` or ``"confirm"``.

    ``holdout`` and ``confirm`` are fractions of units. Explore gets the rest.
    """
    codes, n_units = unit_codes(data, cluster)
    n_holdout = int(round(holdout * n_units))
    n_confirm = int(round(confirm * n_units))
    n_explore = n_units - n_holdout - n_confirm

    if n_explore < 1 or n_holdout < 1 or (confirm > 0 and n_confirm < 1):
        raise ValueError(
            f"too few units ({n_units}) to split into explore, holdout"
            + (" and confirm" if confirm > 0 else "")
        )

    shuffled = rng.permutation(n_units)
    label_of_unit = np.empty(n_units, dtype=object)
    label_of_unit[shuffled[:n_explore]] = EXPLORE
    label_of_unit[shuffled[n_explore : n_explore + n_holdout]] = HOLDOUT
    label_of_unit[shuffled[n_explore + n_holdout :]] = CONFIRM
    return label_of_unit[codes]


def assign_halves(data: pd.DataFrame, cluster: str | None, rng: np.random.Generator) -> np.ndarray:
    """Label every row 1 or 2, keeping each unit inside one half.

    Used for split-half stability: does a cell beat the bar in both halves
    of the holdout?
    """
    codes, n_units = unit_codes(data, cluster)
    shuffled = rng.permutation(n_units)
    half_of_unit = np.full(n_units, 2, dtype=int)
    half_of_unit[shuffled[: (n_units + 1) // 2]] = 1
    return half_of_unit[codes]
