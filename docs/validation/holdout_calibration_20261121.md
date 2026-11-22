# Held-out comparison designs: calibration (2026-11-21)

**What was checked.** Whether the held-out comparisons in
`evalrx.analysis.holdout_designs` give the error rates they claim, on simulated
data with a known answer. No model and no real dataset is involved.

**How to rerun.** `python tools/check_holdout_calibration.py --runs 400 --n-boot 1000`
(about one minute on one CPU core). Every number below is that command's
output.

## 1. False positives when nothing is going on

Data: 120 groups of 2–10 rows. Rows of a group share a failure tendency
(group effect with standard deviation 1.2 on the logit scale). A flag has no
effect on the outcome. 400 datasets per cell; the target is 0.050 and the
simulation error is about ±0.021.

| The flag varies by | Rows treated as independent | Group column used |
|---|---|---|
| group | **0.223** | 0.075 |
| row | 0.045 | 0.050 |

When the flag is a property of the group — the common case, since a flag
computed from an item's features is shared by the rows cut from it — treating
rows as independent rejects four times as often as it should. Using the group
column brings the rate to 0.075. That is still above 0.050 by more than the
simulation error: the percentile bootstrap is somewhat too narrow at this
number of groups.

When the flag varies row by row within groups, both are on target.

## 2. Interval coverage when the effect is known

Data: 150 groups of 2–8 rows with a group-level shift in the failure rate.
400 datasets per design; the target is 0.950.

| Design | True effect | Mean estimate | Coverage |
|---|---|---|---|
| two-group | 0.200 | 0.200 | 0.932 |
| stratified | 0.100 | 0.101 | 0.963 |
| difference in differences | 0.250 | 0.258 | 0.950 |
| dose-response (per bin step) | 0.120 | 0.120 | 0.950 |

In the stratified data the flag is three times as common in the stratum that
fails more. The unadjusted two-group difference averages 0.274; the stratified
estimate recovers the within-stratum effect of 0.100.

## 3. Keyword recipes

Data: 120 clips of 3–7 rows, a `note` column with one real keyword (present in
60% of failing rows, none of the passing ones) and six filler words that are
independent of the outcome. Keywords are chosen on a 60% exploration half,
split by clip, and re-tested on the rest with the clip as the group column.
200 datasets.

| | |
|---|---|
| Real keyword accepted | 200 / 200 |
| Runs in which any filler word was accepted | 8.0% |
| Mean share of accepted keywords that were filler | 5.6% |

The share of accepted keywords that are filler is the quantity the correction
is meant to hold at 5%. It came out at 5.6%, which is within simulation error
of the target but on the high side, in line with the slightly narrow grouped
interval in part 1. The fillers come in pairs that always occur together, so
one chance acceptance usually brings a second.

## What this does and does not show

It shows the estimators are unbiased on data of the shape they assume and that
the error rates are near their targets. Where they miss, they miss on the
permissive side: 7.5% against 5% for a grouped two-group comparison with 120
groups, 93.2% coverage against 95%, and 5.6% filler keywords against 5%. A
result that only just clears the threshold on grouped data deserves less
confidence than the threshold suggests.

It does not show how the designs behave on real tables, where grouping may be
partial or unrecorded, strata may be sparse, and the outcome threshold may
itself have been tuned on the data. It also says nothing about the quality of
the recipes an explorer proposes — only about how a recipe, once frozen, is
re-tested.
