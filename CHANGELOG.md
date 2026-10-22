# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-22

First release.

### Added

- `scan(df, dims, outcome, threshold, ...)` returns a `ScanResult`. It enumerates 1-, 2- and
  3-way cells on an explore split and tests the candidates once on a holdout split.
- One-sided exact binomial tests against a bar of `threshold + min_lift` (or `threshold - min_lift`
  with `alternative="less"`).
- False discovery control through `scipy.stats.false_discovery_control`: Benjamini-Hochberg (the
  default) and Benjamini-Yekutieli. Holm is available for family-wise control. A regression test
  pins `p = [0.02, 0.04, 0.045]`, where a row-by-row comparison would miss two discoveries.
- `min_lift`, for findings that must clear a practical margin.
- Cluster support: `cluster=` splits by cluster and counts one random row per cluster per cell, on
  the explore, holdout and confirm splits alike.
  `design_effect()` and a `DependenceWarning` for row-level tests on clustered rows
  (`aggregate=False`).
- Split-half stability (`stable`), and an optional third `confirm` split that re-tests
  discoveries.
- Binning helpers (`quantile_edges`, `edges_from_spec`, `apply_bins`). `bins=` fits edges on the
  explore split only, so the holdout cannot shape the cells it tests.
- Vectorised holdout counting: one groupby per combination of dims. `max_cells` caps the
  combinatorics, and `max_tests` caps the number of holdout tests.
- `result.splits` records which split each input row went to.
- Command line tool: `segmentscan data.csv --dims a,b,c --outcome y --threshold 0.5`.
- Tests: planted cell recovered, null FDR at or below q over about 200 simulations for BH, BY and
  Holm (plus harder nulls: a `min_lift` boundary, `alternative="less"`, 130 overlapping cells,
  unequal cluster sizes), BH/BY parity with SciPy and with the textbook step-up rule, a
  clustered-null inflation demo, a brute-force property test on random tables, exact validity of
  the binomial p-values, symmetry checks, and leak checks for bins, splits and outcomes.
- Examples on synthetic data (sign-ups, error rates, a model audit, calibration runs) and a
  bring-your-own-copy script for the UCI Adult dataset (CC BY 4.0).

### Planned

- Release on PyPI.
- An option to hide overlapping cells that add nothing beyond their parent cell.
- Explicit bin edges from the command line (today the CLI takes quantile bin counts only).
