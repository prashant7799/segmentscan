"""segmentscan: subgroup discovery for 0/1 outcomes with false-discovery control.

Explore on one split, test once on a holdout with exact binomial tests,
adjust for the number of tests, and report split-half stability.

Example::

    from segmentscan import scan

    result = scan(df, dims=["device", "country"], outcome="converted", threshold=0.10)
    print(result)
"""

from .binning import apply_bins, edges_from_spec, quantile_edges
from .result import ScanResult
from .scan import cell_label, count_cells, dimension_combinations, scan
from .stats import DependenceWarning, adjust_pvalues, binom_pvalues, design_effect

__version__ = "0.1.0"

__all__ = [
    "DependenceWarning",
    "ScanResult",
    "__version__",
    "adjust_pvalues",
    "apply_bins",
    "binom_pvalues",
    "cell_label",
    "count_cells",
    "design_effect",
    "dimension_combinations",
    "edges_from_spec",
    "quantile_edges",
    "scan",
]
