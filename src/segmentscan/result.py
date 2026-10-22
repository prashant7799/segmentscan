"""The object that ``scan()`` returns."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True, repr=False)
class ScanResult:
    """Everything ``scan()`` found, plus the settings that produced it.

    The main table is ``cells``: one row per cell that was tested on the
    holdout. ``discoveries`` is the subset that passed the false-discovery
    control. Print the result for a short summary.
    """

    cells: pd.DataFrame
    """One row per cell tested on the holdout, sorted by p-value."""

    n_explored: int
    """Number of non-empty cells counted on the explore split."""

    n_candidates: int
    """Explore cells with at least ``min_n`` rows whose rate beat the bar."""

    threshold: float
    min_lift: float
    bar: float
    """The rate a cell must beat: threshold + min_lift (or minus, for "less")."""

    q: float
    method: str
    alternative: str
    cluster: str | None
    aggregate: bool
    n_units: dict[str, int] = field(default_factory=dict)
    """Units (rows, or clusters) in each split."""

    n_rows: dict[str, int] = field(default_factory=dict)
    """Rows in each split."""

    bin_edges: dict[str, np.ndarray] = field(default_factory=dict)
    """Bin edges for each binned column, fitted on the explore split only."""

    design_effect: float | None = None
    """Kish design effect on the explore split (only when ``cluster`` is given)."""

    splits: pd.Series | None = None
    """For each row of the input, which split it went to: explore, holdout or confirm."""

    seed: int | None = None

    @property
    def n_tested(self) -> int:
        """Number of cells tested on the holdout (the m in the FDR rule)."""
        return len(self.cells)

    @property
    def discoveries(self) -> pd.DataFrame:
        """Cells that passed the false-discovery control, smallest p-value first."""
        return self.cells[self.cells["discovery"]].reset_index(drop=True)

    def to_frame(self) -> pd.DataFrame:
        """A copy of the full table of tested cells."""
        return self.cells.copy()

    def summary(self, max_rows: int = 10) -> str:
        """A short text report: counts, the bar, and the top discoveries."""
        found = self.discoveries
        noun = "discovery" if len(found) == 1 else "discoveries"
        lines = [
            f"ScanResult: {self.n_explored} cells explored, {self.n_candidates} screened, "
            f"{self.n_tested} tested on holdout, {len(found)} {noun} "
            f"({self.method.upper()}, q={self.q:g})",
            f"Bar: holdout rate {self._bar_text()}; units = {self._units_text()}",
        ]
        if found.empty:
            lines.append("No discoveries.")
            return "\n".join(lines)

        has_confirm = "confirmed" in found.columns
        headers = ["cell", "holdout_n", "rate", "lift", "p_adj", "stable"]
        if has_confirm:
            headers.append("confirmed")

        table = []
        for row in found.head(max_rows).itertuples(index=False):
            line = [
                row.cell,
                str(int(row.holdout_n)),
                f"{row.holdout_rate:.3f}",
                f"{row.lift:+.3f}",
                _format_p(row.p_adj),
                "yes" if row.stable else "no",
            ]
            if has_confirm:
                line.append("yes" if row.confirmed else "no")
            table.append(line)

        widths = [len(h) for h in headers]
        for line in table:
            widths = [max(w, len(text)) for w, text in zip(widths, line, strict=True)]

        def render(parts: list[str]) -> str:
            first = parts[0].ljust(widths[0])
            rest = [text.rjust(width) for text, width in zip(parts[1:], widths[1:], strict=True)]
            return "  ".join([first, *rest]).rstrip()

        lines.append(render(headers))
        lines.extend(render(line) for line in table)
        if len(found) > max_rows:
            lines.append(f"... and {len(found) - max_rows} more (see result.discoveries)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"<ScanResult explored={self.n_explored} tested={self.n_tested} "
            f"discoveries={int(self.cells['discovery'].sum())} method={self.method!r} q={self.q:g}>"
        )

    def _bar_text(self) -> str:
        sign = ">" if self.alternative == "greater" else "<"
        plus_minus = "+" if self.alternative == "greater" else "-"
        if self.min_lift == 0:
            return f"{sign} {self.bar:.3f} (threshold)"
        return (
            f"{sign} {self.bar:.3f} (threshold {self.threshold:.3f} "
            f"{plus_minus} min_lift {self.min_lift:.3f})"
        )

    def _units_text(self) -> str:
        if self.cluster is None:
            return "rows"
        if self.aggregate:
            return f"clusters of {self.cluster!r} (one random row per cluster per cell)"
        return f"rows (split by {self.cluster!r})"


def _format_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"
