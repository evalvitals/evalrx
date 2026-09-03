# Reference output — the confound catch

**A committed, unmodified EvalRX run.** Nothing here was hand-edited. Read
it without installing anything; re-create it with `bash run.sh`.

Input: 30 synthetic chemical batches (`generate_data.py`, seeded, no model or
API key) with `temperature`, `pressure`, `catalyst`, and a continuous
`yield_pct` outcome.

This is the smallest example that shows the behaviour the project exists for:
**the run talks itself out of a finding.**

## The catch

Catalyst looks like it matters. Raw group means are A 70.2%, B 65.3%,
C 72.2% — ANOVA p = 0.080, the kind of near-threshold result that gets
reported as a finding.

Then the run checked the design and found the groups were not comparable:

| Catalyst | Mean yield | Mean temperature |
|---|---|---|
| A | 70.2% | 203 |
| B | **65.3%** | **179** |
| C | 72.2% | 200 |

Catalyst B is preferentially run 21 units cooler. Once temperature is in the
model, catalyst adds **0.032** to R². The M3 hypothesis makes this falsifiable:
B's penalty is inherited temperature and *should shrink to under ~2 pct-pt when
temperature is matched by design.*

Nothing here was prompted. The question asked only "what predicts `yield_pct`?"

## What the run found

| | |
|---|---|
| Temperature | r = **0.90** [0.81, 0.95], univariate R² 0.82 |
| Pressure | r = **-0.14** (p = 0.455) — flat |
| Full model | R² 0.862 (adj 0.840), residual SD 2.76 pct-pt, max VIF 1.56 |
| Diagnostics | Shapiro p 0.785, Breusch-Pagan p 0.166 |
| Host adjudication | e-BH, α = 0.05, 4 candidates → **0 rejected** |

**Zero candidate signals cleared the bar.** The run produced 17 figures and 20
tables and confirmed nothing. That is a correct outcome, not a failed run.

## The `do_not_infer` field

Every claim states what it does *not* license:

- **C1** (temperature, the headline): *"Not causal and not confirmed; a single
  observational batch set cannot separate temperature from anything that
  co-varies with it."*
- **C2** (pressure is flat): *"Not evidence of no effect; n=30 gives limited
  power and the observed range is narrow."*
- **C3** (the confound): *"Does not establish that the catalysts are
  equivalent, nor that temperature explains the gap."*

C2 is the one to notice — the run declines to convert a null result into
evidence of absence.

## Files

| Path | What it is |
|---|---|
| `exploratory_report.json` | Observations, claims, candidate signals, hypotheses, adjudication |
| `analysis.py` | The code the agent actually ran — re-runnable |
| `records.json` | The normalized 30-row table |
| `figures/` | 17 charts |
| `tables/` | 20 analysis-ready CSVs |
| `agent_audit.json` | Provider-verified record of which skills the agent invoked |
| `agent_raw_output.txt` | Full agent transcript |
