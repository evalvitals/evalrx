# Reference output — attention-enriched hallucination analysis

**A committed, unmodified EvalRX run.** Nothing here was hand-edited. Read
it without installing anything; re-create it with `bash run_attn.sh`.

Input: `data_attn_full/` — 606 cases across three Qwen3-VL checkpoints
(2B/4B/8B) answering *"Is there a {object} in the image?"* on COCO val2014,
with seven per-case attention-geometry scalars. 126 FAIL, 480 PASS.

## What the run found

| | |
|---|---|
| Strongest separator | Attention focus share, **AUC 0.82** [0.78, 0.87] |
| Redundancy among the 7 signals | max VIF **20.7**; forward AIC retains **1** |
| Failure rate vs. model size | 30% at 4B → **38% at 8B** (non-monotonic) |
| Object identity vs. checkpoint | Cramér's V **0.51** vs **0.07** |
| Host adjudication | e-BH, α = 0.05, 4 candidates → **1 rejected** (`peaked_attention`) |

## What makes it a reference run

Three things happened here that a tool optimizing for a good demo would not do.

**1. It refused the flattering framing.** All 126 failures sit in the
adversarial absent-object probe; none of the 240 present-object rows fail. So
the run restricted the FAIL/PASS contrast to the 366 adversarial cases rather
than reporting a whole-sample effect that would have looked stronger and meant
less.

**2. It marked its own best number as optimistic.** Claim C5 finds a
peaked-attention rule with a 72% hallucination rate vs 24% elsewhere, and
attaches:

> `do_not_infer`: *Thresholds were chosen on these same rows; the rate is
> optimistic.*

**3. It argued against its own second-strongest signal.** The relative-weight
scalars shift hard across checkpoints (ε² = 0.362, medians 1.63/6.52/3.61,
peaking at 4B). The proposed hypothesis is that this is **extraction
geometry** — differing visual-token counts and layer depth — not a change in
failure mechanism, because bounded concentration measures barely move
(ε² = 0.02) and no signal direction flips in any checkpoint.

Every claim in `exploratory_report.json` carries `status`, `interpretation`,
and `do_not_infer`. The two M3 hypotheses are labeled **proposed, not
validated**.

## Files

| Path | What it is |
|---|---|
| `exploratory_report.json` | Observations, claims, candidate signals, hypotheses, adjudication |
| `analysis.py` | The code the agent actually ran — re-runnable |
| `records.json` | The normalized 606-row table the analysis used |
| `figures/` | 23 charts |
| `tables/` | 28 analysis-ready CSVs |
| `agent_audit.json` | Provider-verified record of which skills the agent invoked |
| `agent_raw_output.txt` | Full agent transcript |

## Caveat

`split: in_sample` — thresholds were chosen on the same rows they were scored
on. To fill the **Held-out Verdicts** tab, run with `--holdout-frac` and
`--holdout-confirm`, or use `run_attn_pipeline.sh`.
