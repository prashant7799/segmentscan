"""Command line: ``segmentscan data.csv --dims a,b,c --outcome y --threshold 0.5``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import pandas as pd

from . import __version__
from .scan import scan
from .stats import ALTERNATIVES, METHODS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segmentscan",
        description=(
            "Find segments whose 0/1 rate beats a threshold. Explore on one split, "
            "test once on a holdout, and control false discoveries."
        ),
    )
    parser.add_argument("csv", help="input CSV file, one row per item")
    parser.add_argument("--dims", required=True, help="comma-separated category columns")
    parser.add_argument("--outcome", required=True, help="0/1 outcome column")
    parser.add_argument("--threshold", required=True, type=float, help="rate a cell must beat")
    parser.add_argument("--min-lift", type=float, default=0.0, help="extra margin past threshold")
    parser.add_argument("--holdout", type=float, default=0.5, help="share of units for testing")
    parser.add_argument("--confirm", type=float, default=0.0, help="share for a third split")
    parser.add_argument("--q", type=float, default=0.05, help="allowed false-discovery rate")
    parser.add_argument("--method", choices=METHODS, default="bh")
    parser.add_argument("--alternative", choices=ALTERNATIVES, default="greater")
    parser.add_argument("--cluster", help="column that groups dependent rows, e.g. user_id")
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="with --cluster: count every row instead of one row per cluster per cell",
    )
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--max-cells", type=int, default=100_000)
    parser.add_argument("--max-tests", type=int, default=None)
    parser.add_argument(
        "--bins",
        default=None,
        help="numeric dims to bin into quantiles, e.g. age:4,income:5",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=20, help="discoveries to print")
    parser.add_argument("--out", default=None, help="write every tested cell to this CSV")
    parser.add_argument("--version", action="version", version=f"segmentscan {__version__}")
    return parser


def parse_bins(text: str | None) -> dict[str, int] | None:
    """Parse ``"age:4,income:5"`` into ``{"age": 4, "income": 5}``."""
    if not text:
        return None
    bins = {}
    for item in text.split(","):
        column, sep, count = item.partition(":")
        if not sep or not column.strip() or not count.strip().isdigit():
            raise ValueError(f"bad --bins item {item!r}; use column:count, e.g. age:4")
        bins[column.strip()] = int(count)
    return bins


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        df = pd.read_csv(args.csv)
        result = scan(
            df,
            dims=[d.strip() for d in args.dims.split(",") if d.strip()],
            outcome=args.outcome,
            threshold=args.threshold,
            holdout=args.holdout,
            q=args.q,
            method=args.method,
            alternative=args.alternative,
            min_lift=args.min_lift,
            cluster=args.cluster,
            aggregate=not args.no_aggregate,
            max_depth=args.max_depth,
            min_n=args.min_n,
            max_cells=args.max_cells,
            max_tests=args.max_tests,
            bins=parse_bins(args.bins),
            confirm=args.confirm,
            seed=args.seed,
        )
    except (ValueError, KeyError, FileNotFoundError) as error:
        message = str(error)
        if isinstance(error, KeyError) and error.args:
            message = str(error.args[0])  # str(KeyError) adds quotes around the text
        print(f"segmentscan: error: {message}", file=sys.stderr)
        return 2

    print(result.summary(max_rows=args.rows))
    if args.out:
        result.cells.drop(columns=["dims", "levels"]).to_csv(args.out, index=False)
        print(f"Wrote {result.n_tested} tested cells to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
