# Held-out Confirmation

`evalrx explore --holdout-frac F --holdout-confirm` explores one part of a
table and re-tests what it found on the rest. This page covers what makes that
re-test mean something: which rows count as independent, which comparisons can
be re-tested, how many claims are corrected together, and what the report says
about the data it was given.

```bash
evalrx explore ./results \
  --outcome-col label \
  --group-col clip_id \
  --holdout-frac 0.4 --holdout-confirm \
  --data-card card.yaml
```

## Rows that are not independent

Many tables hold several rows per underlying item: frames cut from one clip,
one prompt answered by several models, chunks of one document. Rows of an item
resemble each other. Two things go wrong if they are treated as independent.

- **The split leaks.** A row-level split puts rows of one item on both sides,
  so the held-out half contains near-copies of rows the explorer has seen.
- **The intervals are too narrow.** A thousand rows from fifty items are
  closer to fifty observations than to a thousand.

`--group-col COLUMN` names the item. The split then sends each group wholly to
one side, and every held-out interval resamples whole groups
([`weighted_bootstrap`](https://github.com/evalvitals/evalrx/blob/main/evalrx/stats/bootstrap.py)).
`confirm_report.json` records this under `split_meta.group_overlap` (the share
of held-out rows whose group was explored — zero with a group column) and
`adjudication.clustered_by`.

The size of the problem is measurable. In
[the calibration run](validation/holdout_calibration_20261121.md), a flag with
no effect at all was called significant in 22% of datasets when rows were
treated as independent and in 7.5% when the group column was used; the target
is 5%.

### When no group column is named

The run looks for one. A column is suggested when it has at least 20 distinct
values, at least a fifth of the rows share their value with another row, and
rows that share a value agree on the outcome well beyond chance (intraclass
correlation of 0.1 or more). The warning is printed, written to the report's
caveats, and comes with the measured overlap. It reads like this (the numbers
depend on the table):

```text
WARNING: column 'clip_id' looks like the unit rows were cut from (200 groups;
rows of one group share 51% of the outcome variance) and 91.6% of held-out
rows share a group with an explored row, so the held-out check is not
independent of the exploration. Re-run with --group-col clip_id.
```

Only string columns and integer id columns are considered. A float column that
repeats is a rounded measurement, not an item.

`--time-col COLUMN` names a time column the same way. Without it, a column is
taken as time when its name contains `date`, `time`, `timestamp`, `month`,
`year`, `week` or `quarter` as a whole word, ends in `_at`, or holds ISO dates.

## Comparisons a recipe can ask for

A candidate signal carries a **recipe**: an expression over the table's
columns, fixed during exploration. By default it is compared as two groups —
the outcome among rows the expression flags against the rest. Optional fields
on the recipe ask for a different comparison. Every number in them is fixed
during exploration and reused unchanged.

| Field | Comparison | Use it for |
|---|---|---|
| *(none)* | Flagged rows against unflagged rows. | A threshold or a composite condition. |
| `where` | The same, inside the rows where the condition holds. | A finding about one subgroup. |
| `design: stratified` + `strata` | The difference within each stratum, averaged with the strata's sizes as weights. | A gap that may only reflect a different mix. |
| `design: did` + `modifier` | The recipe's effect where the modifier holds, minus its effect where it does not. | An interaction: "the jump is larger for X". |
| `design: dose_response` + `bins` | The change in the outcome per step up fixed cut points. | A continuous signal. |

```json
{"name": "late_batch_matched", "kind": "expr",
 "expr": "batch == 'late'",
 "where": "task == 'i2v'",
 "design": "stratified", "strata": ["generator", "int(encoder_score * 5)"],
 "expect": {"relation": "abs<", "value": 0.10}}
```

Expressions may use column names, comparisons, `and`/`or`/`not`, arithmetic,
`abs`/`min`/`max`/`float`/`int`/`len`, `has_word(column, "word")` and
`contains(column, "text")`. None may read the outcome column, whatever it is
called.

A continuous recipe with no `bins` cannot be re-tested — there is no frozen
threshold to reuse — and is skipped with a reason that says so. A stratum that
holds only flagged or only unflagged rows contributes nothing and is counted in
`details.n_rows_in_one_sided_strata`.

The outcome may be two-valued (effects are differences of failure rates) or
numeric (differences of means).

## Stating what is expected

Without `expect`, a recipe claims "the effect is not zero". With it, the
recipe states the prediction to be tested:

| `relation` | Claim |
|---|---|
| `">"`, `"<"` | The effect is above / below `value` (use `0` for a direction). |
| `"abs<"` | The effect is smaller than `value` in either direction. |
| `"nonzero"` | The default. |

`"abs<"` is how a **control** is written: a comparison that should come out
small. It holds only when the whole interval lies inside the bound. A control
that shows a real difference of nine points against a bound of ten is
`inconclusive`, not a finding and not a pass.

Each claim ends in one of three states:

- `holds` — it survived the correction described below.
- `contradicted` — the data show the opposite of a bounded claim (not corrected
  for multiplicity; "not zero" has no testable opposite and is never
  contradicted).
- `inconclusive` — neither.

`reject` on a signal means its claim holds and the claim is one of separation;
a control that holds has `reject: false`.

## One correction for every claim

Every adjudicated signal makes one claim, and so does every numeric prediction
a hypothesis carries. All of them are corrected together with
Benjamini-Hochberg at `alpha`. `adjudication.method` names what was applied —
`"BH"`, or `"none"` when no claim could be tested — and `n_in_family` counts
the claims. `raw_reject` on a signal says whether it would have passed alone.

p-values come from the same cluster bootstrap as the intervals. For a
two-group comparison of a two-valued outcome with independent rows, the exact
(Fisher) test is used instead, because a bootstrap of a small, cleanly
separated sample degenerates: four failures against four passes resample to
the same difference every time. `details.p_value_method` records which was
used, and `details.small_sample_warning` appears when the smaller side has
fewer than ten independent units.

Candidates adjudicated during exploration (`sufficient` statistics) follow the
same rule: paired comparisons share an e-BH family and two-group comparisons
share a BH family.

## Hypotheses: what the host checks and what the judge grades

A hypothesis may carry `predictions` — numeric statements about named signals:

```json
{"statement": "Late batches fail more on image-to-video, not text-to-video.",
 "predictions": [
   {"signal": "late_i2v", "relation": ">", "value": 0.10},
   {"signal": "late_t2v", "relation": "abs<", "value": 0.10}],
 "experiment": {
   "data_needed": "200 held-out prompts re-rendered with both encoders",
   "operation": "swap the image encoder, keep everything else",
   "expected_result": "the late-batch gap closes",
   "falsified_by": "the gap stays above 10 points"}}
```

The host checks each prediction against the held-out rows and sets
`host_verdict`: `supported` when all hold, `refuted` when any is contradicted,
`inconclusive` otherwise, `none` when there was nothing to check. This needs
no model. The judge still grades the hypothesis as a whole, is shown the
host's results, and is asked to leave them as given.

`experiment` replaces a free-text test plan with four fields. It describes the
intervention needed when re-counting rows cannot settle the claim; a plan with
no `falsified_by` is visibly incomplete.

`--judge-backend` picks the judge: `same` (default) follows `--backend`, so a
run on Codex does not also need the Claude CLI. `confirm_report.json` records
the judge under `judge`, including `same_as_proposer`. When the judge is, or
may be, the model that proposed the hypotheses, the report says so; pass
`--judge-backend` or `--judge-model` for an independent review. Signal verdicts
and host-checked predictions never depend on the judge.

## Representative cases

For each signal whose claim holds, `representative_cases` lists three rows per
side, chosen nearest the side's median on the outcome and on the columns the
recipe reads. They show what the comparison is mostly made of; the most extreme
rows would show the cases least like the rest. Each entry gives the row's
position in `holdout_records.json` and its id when the rows have one.

## Keywords in a free-text column

`--text-col COLUMN` adds keyword candidates. On the exploration rows, the words
most over-represented on each side of the outcome are found (log-odds ratio
with a prior, counting rows rather than occurrences). Each becomes a recipe
`has_word(column, "word")` whose `expect` is the direction it showed there, and
is then re-tested on the held-out rows in the same family as everything else.

Some word always leans one way. The discipline is in the second step: a word
has to repeat its direction on rows it was not chosen on, after correction.

## What the report says about the data

**Data card.** Four facts decide whether a pattern in a table says anything
about the system that produced it:

| Field | Question |
|---|---|
| `unit` | What is one row? |
| `extraction` | How was a row cut from the raw object? |
| `threshold` | How was the outcome, or its threshold, set? |
| `aggregation` | How are results combined when the system is used? |

Give them with `--data-card FILE` (JSON or YAML) or `--row-unit`,
`--row-extraction`, `--outcome-threshold` and `--deployment-aggregation`. The
answers are printed in the report's caveats and passed to the explorer. Any
left unanswered are listed there too.

**Leakage check.** A column that almost fully determines the outcome is
usually the outcome under another name, or something recorded after it. Such
columns are reported in the caveats: a column with few distinct values whose
values predict the outcome at least 98% of the time, a numeric column whose
rank AUC is within 0.01 of 0 or 1, or — for a numeric outcome — a correlation
of 0.98 or more.

**Row limit.** `--max-rows` caps what the host loads (default 2000). When rows
are dropped the run says how many, and that the loaded rows are taken from the
start of each file and are not a random sample.

## From Python

```python
from evalrx.analysis.holdout import holdout_confirm, split_records

explore, held_out = split_records(rows, 0.4, outcome_col="label", group_col="clip_id")
# ... explore on `explore`, collect candidate signals with recipes ...
confirm = holdout_confirm(report, held_out, outcome_col="label", group_col="clip_id")
```

`evalrx.stats.compare(a, b, paired=False, cluster_by=(groups_a, groups_b))`
gives the clustered two-group comparison on its own, and
`tools/check_holdout_calibration.py` reruns the calibration.

## Limits

- The bootstrap interval is approximate, and on grouped data it errs on the
  permissive side: in the calibration run the false-positive rate with 120
  groups was 7.5% against a 5% target, and interval coverage was 93% against
  95%. With fewer groups expect it to be worse.
- A group column has to be named or noticed. Grouping that is not in the table
  — two rows that came from one source but carry no shared id — cannot be
  detected.
- Stratification removes differences in mix across the strata that were named.
  It says nothing about differences along anything else.
- A claim that holds on held-out rows is a pattern that repeated. Whether it is
  a cause is the question `experiment` is for.
