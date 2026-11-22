"""Statistics — effect-sized, multiple-testing-aware verdicts for failure analysis.

``compare`` is the single entry point (never returns a bare p). Building blocks
are exported too: McNemar (paired binary), clustered bootstrap CI, e-values
(anytime-valid), e-BH (FDR under dependence), stratified subset sampling + τ.
"""

from evalrx.stats.api import (
    MultiCompareResult,
    StatResult,
    compare,
    compare_multiple,
    compare_paired_rates,
)
from evalrx.stats.bootstrap import (
    bootstrap_p_value,
    clustered_bootstrap_diff,
    mean_difference_statistic,
    weighted_bootstrap,
)
from evalrx.stats.ebh import ebh
from evalrx.stats.evalue import e_value_test, evalue_bernoulli, evalue_bounded_mean
from evalrx.stats.exact import two_group_exact_p
from evalrx.stats.friedman import chi2_sf, friedman_test, nemenyi_cd, nemenyi_pairs
from evalrx.stats.mcnemar import mcnemar
from evalrx.stats.multiplicity import MultiplicityReport, bh, correct_results
from evalrx.stats.subset_sampling import kendall_tau, stratified_subset

__all__ = [
    "compare",
    "StatResult",
    "compare_multiple",
    "compare_paired_rates",
    "MultiCompareResult",
    "mcnemar",
    "clustered_bootstrap_diff",
    "weighted_bootstrap",
    "bootstrap_p_value",
    "mean_difference_statistic",
    "evalue_bernoulli",
    "e_value_test",
    "evalue_bounded_mean",
    "ebh",
    "two_group_exact_p",
    "friedman_test",
    "nemenyi_cd",
    "nemenyi_pairs",
    "chi2_sf",
    "bh",
    "correct_results",
    "MultiplicityReport",
    "stratified_subset",
    "kendall_tau",
]
