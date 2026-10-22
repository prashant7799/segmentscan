# segmentscan

[![CI](https://github.com/prashant7799/segmentscan/actions/workflows/ci.yml/badge.svg)](https://github.com/prashant7799/segmentscan/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

segmentscan finds the segments of a table where a yes/no outcome happens more often (or less often) than a rate you choose. It then checks those findings on data that played no part in finding them. You give it a table with a few category columns, such as device, country or plan, and one 0/1 outcome, such as "signed up", "had an error" or "model was right". It searches every 1-, 2- and 3-way segment on one random part of the data. It then tests the promising segments once, on the other part, with exact binomial tests. Finally, it controls the false discovery rate across all those tests, at a level you choose. It is for product and growth analysts, QA and reliability engineers, risk analysts, and ML teams who check accuracy segment by segment.

## What it does NOT do

- **It does not prove cause.** A segment can do well for reasons that are not in your columns.
- **It is not an A/B test.** It does not compare two randomised groups. If you can run an experiment, do that.
- **It handles binary outcomes only.** Revenue, time and other numbers are out of scope.
- **It runs one-sided tests only.** You ask "above the bar?" (`alternative="greater"`) or "below the bar?" (`"less"`), not both at once.
- **It does not choose your threshold.** Pick the rate to beat before you look at the data.
- **It does not merge overlapping findings.** If one segment is special, its parents and children often show up too. You read them as a family.
- **It does not look across time.** The split is random, not by date. A segment that is special in this table may not be special next month. Check that on newer data.
- **It does not search for cut points.** Number columns are cut into fixed bins (quantiles or your own edges) before the search.
- **It only tries AND-combinations of single values**, such as `device=tablet & country=IN`. It does not try ORs, such as `country=IN or UK`.
- **It does not use row weights.** Survey weights and importance weights are not supported.
- **It does not detect dependent rows by itself.** If rows come in groups (many visits per user), you must name the group column with `cluster=`.

## Install

```bash
pip install git+https://github.com/prashant7799/segmentscan
```

A PyPI release will come later. You need Python 3.10 or newer. The only dependencies are numpy, pandas and scipy.

## Quickstart (60 seconds)

This builds a synthetic table of 6,000 website visits. Tablet visitors who came from the newsletter sign up 50% of the time. Everyone else signs up 30% of the time.

```python
import numpy as np
import pandas as pd

from segmentscan import scan

# 6,000 synthetic website visits with three category columns.
rng = np.random.default_rng(42)
n = 6000
df = pd.DataFrame(
    {
        "device": rng.choice(["mobile", "desktop", "tablet"], n),
        "country": rng.choice(["IN", "UK", "NZ", "DE"], n),
        "referrer": rng.choice(["search", "social", "newsletter"], n),
    }
)
# Everyone signs up 30% of the time, except tablet visitors from the newsletter: 50%.
special = (df["device"] == "tablet") & (df["referrer"] == "newsletter")
df["signed_up"] = (rng.random(n) < np.where(special, 0.50, 0.30)).astype(int)

result = scan(
    df,
    dims=["device", "country", "referrer"],
    outcome="signed_up",
    threshold=0.30,  # the baseline rate, chosen before looking
    min_lift=0.05,  # only care about segments at least 5 points better
)
print(result)
```

Output:

```text
ScanResult: 79 cells explored, 18 screened, 18 tested on holdout, 4 discoveries (BH, q=0.05)
Bar: holdout rate > 0.350 (threshold 0.300 + min_lift 0.050); units = rows
cell                                              holdout_n   rate    lift    p_adj  stable
device=tablet & referrer=newsletter                     351  0.504  +0.204  4.0e-08     yes
device=tablet & country=IN & referrer=newsletter         93  0.581  +0.281  4.3e-05     yes
country=IN & referrer=newsletter                        244  0.455  +0.155    0.003     yes
device=tablet & country=UK & referrer=newsletter         93  0.495  +0.195    0.013     yes
```

How to read it:

- 79 segments ("cells") were counted on the explore half. 18 of them had at least `min_n` rows and beat the bar there, so 18 were tested on the holdout half.
- The planted segment comes first. Its holdout rate is 0.504, and its adjusted p-value is tiny.
- The other three rows overlap the planted segment, so their true rates are above the bar too. Two of them are pieces of it (true rate 0.50). `country=IN & referrer=newsletter` mixes tablets with other devices: its true rate is about 0.367 (one third tablets at 0.50, two thirds others at 0.30).
- `stable = yes` means the segment beat the bar in both halves of the holdout.

The same script is in [`examples/quickstart.py`](examples/quickstart.py).

### From the command line

```bash
python examples/signup_segments.py --save-csv data/signups.csv
segmentscan data/signups.csv --dims country,plan_page,device,referrer \
    --outcome signed_up --threshold 0.30 --min-lift 0.03 --cluster user_id
```

```text
ScanResult: 248 cells explored, 77 screened, 77 tested on holdout, 3 discoveries (BH, q=0.05)
Bar: holdout rate > 0.330 (threshold 0.300 + min_lift 0.030); units = clusters of 'user_id' (one random row per cluster per cell)
cell                                                   holdout_n   rate    lift    p_adj  stable
device=desktop & referrer=newsletter                         765  0.412  +0.112  1.0e-04     yes
country=UK & device=desktop & referrer=newsletter            177  0.469  +0.169    0.003     yes
plan_page=team & device=desktop & referrer=newsletter        242  0.434  +0.134    0.012     yes
```

Run `segmentscan --help` for every option. Add `--out cells.csv` to save the full table of tested cells.

## Concepts

### The method in five steps

```text
                 your table: one row per item, or rows grouped by --cluster
                                        |
                         random split by unit (seeded)
                 +----------------------+----------------------+
                 |                      |                      |
              EXPLORE                HOLDOUT            CONFIRM (optional)
                 |                      |                      |
   1. count every 1-, 2-, 3-way    3. test each candidate      |
      cell                            ONCE: exact binomial     |
   2. keep cells with n >= min_n      test against the bar     |
      whose rate beats the bar     4. adjust p-values for the  |
      = candidates  ------------->    number of tests          |
                                      (BH, BY or Holm)         |
                                      p_adj <= q = discovery   |
                                   5. split-half stability     |
                                      discoveries ------------> re-tested here
```

1. **Split.** Units go at random into explore and holdout (and confirm, if you ask). A unit is a row, or a whole cluster when you pass `cluster=`.
2. **Explore.** Count every non-empty cell up to `max_depth` dims. A cell is a candidate if it has at least `min_n` units and its explore rate beats the bar.
3. **Test.** Count each candidate on the holdout. Drop cells with fewer than `min_n` units there. Run a one-sided exact binomial test against the bar.
4. **Control false discoveries.** Adjust the p-values for the number of tests. A cell with an adjusted p-value at or below `q` is a discovery.
5. **Check stability.** Split the holdout in two. A cell is `stable` if it beats the bar in both halves. With `confirm > 0`, discoveries are tested again on a third split.

The holdout outcomes never help choose what gets tested. (Only the holdout cell sizes are used, to drop cells below `min_n`.) That is why the holdout p-values are honest. If you choose segments and then test them on the same rows, the test cannot allow for the choice, and the p-values come out too small.

This explore-then-test design is not new. It is data splitting (Cox, 1975), used for pattern search as "holdout evaluation" by Webb (2007). segmentscan packages it for 0/1 outcomes, with exact tests, clusters and leak-free bins.

### Threshold, minimum lift and the bar

- `threshold` is the rate a segment must beat, such as last quarter's sign-up rate or an error budget.
- `min_lift` is the extra margin that makes a difference matter to you.
- The **bar** is `threshold + min_lift` for `alternative="greater"`, or `threshold - min_lift` for `"less"`.

The null hypothesis for each cell is "the true rate is at or below the bar" (or at or above it, for `"less"`). So a discovery means the data show that the segment clears the margin. A rate that merely sits above the threshold is not enough.

### False discovery control

When you test 70 segments at the 5% level, about 3 or 4 will look special by luck. The `method` chooses the protection:

| method | controls | in words |
| --- | --- | --- |
| `"bh"` (default) | false discovery rate | on average, at most a share `q` of your discoveries are false |
| `"by"` | false discovery rate, under any dependence | the same promise with no assumption about how the tests relate; stricter |
| `"holm"` | family-wise error rate | the chance of even one false discovery is at most `q`; strictest |

BH and BY call `scipy.stats.false_discovery_control`, the full step-up rule. Holm is written out in a few lines in [`stats.py`](src/segmentscan/stats.py).

**Why BH is the default.** Cells that share rows give tests that move in the same direction: if `device=tablet` is lucky, so is `device=tablet & country=IN`. Benjamini and Yekutieli (2001) proved that BH controls the FDR under a kind of positive dependence called PRDS. Overlapping cells look like that kind of dependence, but this package does not prove that they meet the condition. So the test suite checks it by simulation on overlapping cells (see [How correctness is tested](#how-correctness-is-tested)). If you need a guarantee under any dependence, use `method="by"`.

### Clustered or repeated observations

Rows are often not independent. One user visits many times. One device sends many sessions. Rows from the same unit tend to agree. If you treat them as independent rows, the p-values come out too small, and false discoveries go up.

Pass `cluster="user_id"` and segmentscan does two things:

1. **The split is done by cluster.** A user is never in both explore and holdout.
2. **Each cluster counts once per cell, on every split.** One of its rows in that cell is chosen at random, so both the explore screen and the binomial test count users, not visits.

Counting one random row per cluster keeps the outcome a per-visit outcome, and it keeps the exact test valid. The cost is power: you test with fewer units. `result.design_effect` reports how strongly rows cluster. A design effect of 2 means 1,000 rows carry about as much information as 500 independent rows.

**Pick the threshold on the same scale.** With `cluster=`, the rate being tested is a per-visit rate *averaged over users*: every user counts the same, however many visits they made. This is not always the pooled rate (all sign-ups divided by all visits). If heavy users sign up less often than light users, the pooled rate is lower than the per-user average. A threshold set to the pooled rate then makes almost every segment look "above the bar". Take the threshold from the same kind of average, for example the mean over users of each user's sign-up rate in earlier data.

If you pass `cluster=` with `aggregate=False`, every row counts again. segmentscan then issues a `DependenceWarning` when the design effect is above 1.2. On small tables the design effect estimate is noisy, so the warning can fire on rows that are in fact independent.

If your question is really per user ("did the user ever sign up?"), make one row per user first. Then you do not need `cluster=`.

### Binning numbers without leakage

Category columns work as they are. Number columns need bins. Pass `bins={"age": 4}` for quartile bins, or `bins={"age": [25, 40, 60]}` for your own edges. Quantile edges are fitted on the **explore rows only**, then used everywhere. If the edges came from all rows, the holdout would help shape the very cells it is meant to test. The edges used are in `result.bin_edges`.

### Choosing parameters

| parameter | default | how to choose |
| --- | --- | --- |
| `threshold` | (required) | A rate fixed in advance: a baseline, a target or a budget. |
| `min_lift` | `0.0` | The smallest difference worth acting on. A larger value gives fewer, stronger findings. |
| `q` | `0.05` | The share of false discoveries you can live with. |
| `method` | `"bh"` | `"by"` for strict FDR under any dependence; `"holm"` when one false finding is costly. |
| `holdout` | `0.5` | Half is a good start. More holdout means more power per test but a noisier explore step. |
| `min_n` | `30` | Small cells rarely pass anyway. A larger value means fewer tests and a smaller penalty. |
| `max_depth` | `3` | 3-way cells get small fast. Use 2 for many dims or small tables. |
| `max_tests` | `None` | Set it (for example to 50) to test only the strongest explore candidates. |
| `confirm` | `0.0` | Set it (for example to 0.2) when you have plenty of data and want a clean re-test. |
| `seed` | `0` | Change it to see how much the result depends on the split. |

## API reference

### `scan()`

```python
scan(
    df, dims, outcome, threshold, *,
    holdout=0.5, q=0.05, method="bh", alternative="greater", min_lift=0.0,
    cluster=None, aggregate=True, max_depth=3, min_n=30, max_cells=100_000,
    max_tests=None, bins=None, confirm=0.0, seed=0,
) -> ScanResult
```

| argument | meaning |
| --- | --- |
| `df` | A pandas DataFrame, one row per item. It is not modified. |
| `dims` | Category columns to cut by. Rows with a missing value in a dim are left out of the cells that use that dim. |
| `outcome` | A 0/1 or True/False column with no missing values. |
| `threshold` | The rate to beat, strictly between 0 and 1. |
| `holdout` | Share of units used for testing. |
| `q` | Allowed false discovery rate (for Holm: the family-wise error rate). |
| `method` | `"bh"`, `"by"` or `"holm"`. |
| `alternative` | `"greater"` (cells above the bar) or `"less"` (cells below it). |
| `min_lift` | Margin added to (or taken from) the threshold to form the bar. |
| `cluster` | Column that groups dependent rows, such as a user ID. |
| `aggregate` | With `cluster`: count one random row per cluster per cell, on every split (default `True`). |
| `max_depth` | Most dims combined in one cell. |
| `min_n` | Smallest cell, in units, on both explore and holdout. A unit is a row, or a cluster with `cluster=` and `aggregate=True`. |
| `max_cells` | Safety cap. Raises `ValueError` if the explore split has more cells than this. |
| `max_tests` | Test only this many candidates, those with the smallest explore p-values. |
| `bins` | `{column: n_bins}` or `{column: [edges]}` for numeric dims. |
| `confirm` | Share of units for an optional third split. |
| `seed` | Seed for all random choices. The same seed gives the same result. |

### `ScanResult`

| attribute | meaning |
| --- | --- |
| `discoveries` | DataFrame of the cells with `p_adj <= q`, smallest p-value first. |
| `cells` | DataFrame of every cell tested on the holdout. |
| `n_explored`, `n_candidates`, `n_tested` | Cells counted on explore, cells that passed the screen, and cells tested. |
| `bar`, `threshold`, `min_lift`, `q`, `method`, `alternative` | The settings used. |
| `n_units`, `n_rows` | Units and rows in each split. |
| `splits` | A Series aligned with `df`: `"explore"`, `"holdout"` or `"confirm"` for each row. |
| `bin_edges` | Edges for each binned column. |
| `design_effect` | Kish design effect on the explore split (only with `cluster`). |
| `summary(max_rows=10)` | The text report. `print(result)` shows the same. |
| `to_frame()` | A copy of `cells`. |

Columns of `cells` and `discoveries`:

| column | meaning |
| --- | --- |
| `cell` | Readable name, such as `device=tablet & country=IN`. |
| `depth` | Number of dims in the cell. |
| `explore_n`, `explore_rate` | Units and rate on the explore split. |
| `holdout_n`, `holdout_successes`, `holdout_rate` | Units, successes and rate on the holdout. |
| `lift` | `holdout_rate - threshold`. Negative for cells below the threshold. |
| `p_value` | One-sided exact binomial p-value against the bar. |
| `p_adj` | Adjusted p-value (BH, BY or Holm). |
| `discovery` | `p_adj <= q`. |
| `half1_n`, `half1_rate`, `half2_n`, `half2_rate` | The two halves of the holdout. |
| `stable` | The cell beats the bar in both halves. |
| `confirm_n`, `confirm_rate`, `confirm_p_adj`, `confirmed` | Only with `confirm > 0`. Filled for discoveries. |
| `dims`, `levels` | The cell as two tuples, for code. |

### Helpers

| function | what it does |
| --- | --- |
| `adjust_pvalues(p, method="bh")` | BH, BY or Holm adjusted p-values. |
| `binom_pvalues(successes, n, rate, alternative)` | One-sided exact binomial p-values for many cells at once. Same numbers as `scipy.stats.binomtest`. |
| `design_effect(df, outcome, cluster)` | Kish design effect, `1 + (m - 1) * icc`. |
| `quantile_edges(values, n_bins)` | Quantile bin edges with open ends (`-inf`, `+inf`). |
| `edges_from_spec(values, spec)` | Edges from a bin count or from explicit edges. |
| `apply_bins(values, edges)` | Map numbers to labels such as `[30, 45)`. |
| `count_cells(rows, combo, outcome)` | Rows and successes for every cell of one combination of dims. |
| `DependenceWarning` | Issued when row-level tests meet clustered rows. |

### Command line

```text
segmentscan CSV --dims a,b,c --outcome y --threshold 0.3
            [--min-lift 0] [--holdout 0.5] [--confirm 0] [--q 0.05]
            [--method bh|by|holm] [--alternative greater|less]
            [--cluster COL] [--no-aggregate] [--max-depth 3] [--min-n 30]
            [--max-cells 100000] [--max-tests N] [--bins age:4,income:5]
            [--seed 0] [--rows 20] [--out cells.csv]
```

`python -m segmentscan` works too. Errors print one line and exit with code 2.

## Examples

Every example uses synthetic data, except the Adult one, which reads a copy you download yourself.

| script | question | shows |
| --- | --- | --- |
| [`quickstart.py`](examples/quickstart.py) | Which visitor segments sign up above 30%? | The basic call (output above). |
| [`signup_segments.py`](examples/signup_segments.py) | The same question, with repeat visits per user. | `cluster=`, the design effect, visit-level vs user-level. |
| [`error_rates.py`](examples/error_rates.py) | Which client/browser/version combos break the 2% error budget? | `method="by"`, `min_lift`, a confirm split. |
| [`model_audit.py`](examples/model_audit.py) | Where is a classifier below 90% accuracy? | `alternative="less"`, leak-free `bins=`. |
| [`uci_adult.py`](examples/uci_adult.py) | Which census groups earn more than $50K? | A real public dataset (UCI Adult, CC BY 4.0). Bring your own copy. |
| [`fdr_simulation.py`](examples/fdr_simulation.py) | Does the FDR promise hold? | The calibration runs (output below). |

`signup_segments.py` runs the same scan twice. Counting visits gives `p_adj = 1.6e-06` for the real segment and 6 discoveries. Counting users gives `1.0e-04` and 3 discoveries. The design effect is 2.05, so the visit-level run is over-confident. In this run both find only the real effect and its sub-segments. On data with no real effect, the visit-level run is the one that makes false discoveries (see the simulation below).

`error_rates.py` plants two problems: Android on version 5.3 (4.0%) and Safari on the web client (3.5%). The scan finds both, plus their sub-segments, and confirms both on the third split.

`model_audit.py` finds the weak spot, young customers on the app, through the age bin `(-inf, 27)` that was fitted on explore rows.

## How correctness is tested

The tests run with `pytest`. They use only synthetic data and no network.

- **p-values.** `binom_pvalues` matches `scipy.stats.binomtest` on hundreds of random cases (Hypothesis).
- **BH, BY, Holm.** Each method is checked against a rule written straight from its paper, on random lists of p-values. BH and BY are also checked against SciPy.
- **The step-up bug.** `p = [0.02, 0.04, 0.045]` at `q = 0.05`. A rule that compares each p-value only with its own line accepts just the last one. The real step-up rule accepts all three. A test pins this.
- **Planted cell recovered.** A planted segment is found, is stable, and ranks first. It is found in at least 18 of 20 seeded datasets.
- **Brute force.** On 60 random small tables (text, number and yes/no columns, missing values, clusters or not, both directions, all three methods, with and without a confirm split), every count, candidate, p-value, adjusted p-value and discovery is checked against slow, obvious loops (Hypothesis). With clusters, the random row per cluster cannot be replayed, so its counts are checked against exact bounds.
- **No leakage.** Bin edges equal the edges fitted on explore rows alone, and changing every holdout value does not move them. Changing every holdout outcome does not change which cells are tested. Changing explore outcomes does not change any tested cell's holdout counts. Changing confirm outcomes does not change the discoveries. Clusters never cross splits.
- **Symmetries.** Flipping the outcome (`y` to `1 - y`, threshold `t` to `1 - t`) turns `"greater"` into `"less"` with the same p-values. The order of `dims` and the index labels of `df` do not change the result.
- **Valid p-values.** For every true rate on the null side of the bar, the exact chance of `p <= alpha` is at most `alpha` (computed from the binomial distribution, not simulated).
- **Null FDR at or below q.** About 200 seeded datasets per setting, for BH, BY and Holm, at `q = 0.05` and `0.10`. Also for harder settings: rates sitting exactly on a `min_lift` bar, `alternative="less"`, 130 heavily overlapping cells, and users with unequal numbers of rows.
- **Clustered-null inflation demo.** On clustered data with no real effect, row-level analysis makes false discoveries in more than 3q of runs. `cluster=` brings it back to q or below.

These simulations are seeded, so they give the same numbers every time. They are still Monte Carlo estimates: with 200 runs, an FDR near 0.10 has a standard error of about 0.02. They support the guarantees; they do not prove them.

Output of `python examples/fdr_simulation.py` (about a minute):

```text
Part 1: independent rows, 200 runs per line, q = 0.1
scenario      method      FDR  any false  planted found
global null   bh        0.080      0.080              -
global null   by        0.005      0.005              -
global null   holm      0.055      0.055              -
one planted   bh        0.053      0.105           0.36
one planted   by        0.005      0.005           0.23
one planted   holm      0.025      0.030           0.33

Part 2: clustered rows, no real effect, 100 runs, q = 0.1
rows treated as independent : FDR 0.550
cluster="user_id"           : FDR 0.050

Part 3: harder settings, no real effect, BH, q = 0.1
rate exactly on the bar (min_lift)  : FDR 0.050 (200 runs)
alternative="less"                  : FDR 0.060 (200 runs)
130 overlapping cells               : FDR 0.080 (100 runs)
unequal clusters, by user           : FDR 0.065 (200 runs)
```

Every FDR is at or below `q = 0.10` when the rows are treated correctly. (In Part 2, the FDR of the row-level run is the share of runs with any discovery, because every discovery is false there.) The planted cell here is small (about 110 holdout rows at 0.40 against 0.30), so power is modest. BY pays for its stronger guarantee with less power.

## Limitations and assumptions

- **Units are independent.** Rows, or clusters when you pass `cluster=`, must not influence each other. Time trends, spillovers and shared shocks break this.
- **The threshold is fixed in advance.** If you pick it after looking at the same data, the p-values are no longer exact.
- **Overlapping cells.** A strong cell lifts its parents and children. They are real discoveries in the statistical sense, but they describe one effect. Read the shortest cell with a large lift first.
- **Winner's curse.** Discoveries are chosen by their holdout p-values, so their holdout rates lean a little high. Use `confirm=` for a fresh estimate.
- **BH under dependence.** BH's guarantee is proven for a narrower kind of dependence than overlapping cells. The simulations support it here. `method="by"` needs no such assumption.
- **Discrete tests are conservative.** Exact binomial tests on small cells rarely reach the nominal level. You lose a little power, not validity.
- **One random row per cluster loses power.** With heavy clustering you test with roughly one unit per cluster per cell. The rate is averaged over clusters, so big and small clusters count the same. Set the threshold on that scale (see [Clustered or repeated observations](#clustered-or-repeated-observations)).
- **The split costs data.** Half of the data (by default) only chooses what to test. If you have few cells and would test all of them anyway, testing every cell on all rows with `adjust_pvalues` can be more powerful.
- **The split is random.** Results change a little with `seed`. A finding that only appears for one seed is weak. With `cluster=`, the random row chosen per cluster also depends on `seed`.
- **Size.** Everything runs in memory with pandas. The examples run in a few seconds on 240,000 rows. Many dims at depth 3 can produce a very large number of cells; `max_cells` stops that early.

## Related tools

- **Subgroup discovery libraries** offer many search algorithms and quality measures, and numeric targets too. They find and rank subgroups on the data you give them; a held-out test and false-discovery control are usually left to you. Use one if you want many search strategies or numeric targets.
- **Slice-finding tools for model debugging** search fast for slices where a model's error is high. They usually score slices by size and error, not with p-values. Use one for quick model debugging.
- **Research methods** for pattern divergence and for statistically significant slices (see the references) are closest in spirit to segmentscan.
- **P-value corrections** (`scipy.stats.false_discovery_control`, or the multiple-testing functions in statsmodels) are all you need if you already have one p-value per segment from fresh data. segmentscan packages the whole protocol around them.

Use segmentscan when you want each segment tested on data that did not choose it, with the false discovery rate controlled at a level you pick (under the assumptions above).

## References and citation

- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B*, 57(1), 289-300.
- Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165-1188.
- Cox, D. R. (1975). A note on data-splitting for the evaluation of significance levels. *Biometrika*, 62(2), 441-444.
- Webb, G. I. (2007). Discovering significant patterns. *Machine Learning*, 68(1), 1-33. (Holdout evaluation for pattern discovery.)
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Hoeffding, W. (1956). On the distribution of the number of successes in independent trials. *Annals of Mathematical Statistics*, 27(3), 713-721. (Why the binomial test stays valid when units have different rates.)
- Kish, L. (1965). *Survey Sampling*. Wiley. (The design effect.)
- Lemmerich, F. and Becker, M. (2018). pysubgroup: Easy-to-use subgroup discovery in Python. *ECML PKDD 2018*.
- Sagadeeva, S. and Boehm, M. (2021). SliceLine: Fast, linear-algebra-based slice finding for ML model debugging. *SIGMOD 2021*.
- Pastor, E., de Alfaro, L. and Baralis, E. (2021). Looking for trouble: analyzing classifier behavior via pattern divergence. *SIGMOD 2021*.
- Chung, Y., Kraska, T., Polyzotis, N., Tae, K. H. and Whang, S. E. (2019). Slice Finder: automated data slicing for model validation. *ICDE 2019*.
- Becker, B. and Kohavi, R. (1996). Adult. UCI Machine Learning Repository. CC BY 4.0.

If segmentscan helps your work, please cite the method papers above and the software:

```bibtex
@software{lohar_segmentscan_2026,
  author  = {Lohar, Prashant},
  title   = {segmentscan: subgroup discovery for binary outcomes with false-discovery control},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/prashant7799/segmentscan}
}
```

## Licence

MIT. Copyright (c) 2026 Prashant Lohar. See [LICENSE](LICENSE).
