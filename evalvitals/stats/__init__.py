"""Statistics — effect-sized, multiple-testing-aware verdicts for failure analysis.

``compare`` is the single entry point (never returns a bare p). Building blocks
are exported too: McNemar (paired binary), clustered bootstrap CI, e-values
(anytime-valid), e-BH (FDR under dependence), stratified subset sampling + τ.
"""

from evalvitals.stats.api import (
    MultiCompareResult,
    StatResult,
    ab_test,
    compare,
    compare_multiple,
    compare_paired_rates,
)
from evalvitals.stats.bootstrap import clustered_bootstrap_diff
from evalvitals.stats.ebh import ebh
from evalvitals.stats.evalue import e_value_test, evalue_bernoulli, evalue_bounded_mean
from evalvitals.stats.friedman import chi2_sf, friedman_test, nemenyi_cd, nemenyi_pairs
from evalvitals.stats.mcnemar import mcnemar
from evalvitals.stats.multiplicity import MultiplicityReport, bh, correct_results
from evalvitals.stats.subset_sampling import kendall_tau, sample_subset, stratified_subset

__all__ = [
    "compare",
    "StatResult",
    "compare_multiple",
    "compare_paired_rates",
    "MultiCompareResult",
    "ab_test",
    "mcnemar",
    "clustered_bootstrap_diff",
    "evalue_bernoulli",
    "e_value_test",
    "evalue_bounded_mean",
    "ebh",
    "friedman_test",
    "nemenyi_cd",
    "nemenyi_pairs",
    "chi2_sf",
    "bh",
    "correct_results",
    "MultiplicityReport",
    "stratified_subset",
    "kendall_tau",
    "sample_subset",
]
