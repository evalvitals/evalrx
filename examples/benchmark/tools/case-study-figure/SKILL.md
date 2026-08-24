---
name: case-study-figure
version: 0.1.0
description: >
  Draw the one-page case-study figure for a finished EvalVitals benchmark run —
  the M1→M2→M3→M5→M4 story as a single SVG. Use whenever someone asks for a case
  study figure, a qualitative figure, or "the figure" for a run directory under
  examples/benchmark/*/outputs/. Runs extract_figure_data.py first and draws only
  numbers that come back in its JSON; the run's own qa_flags are binding on what
  the figure is allowed to claim.
---

# Case-Study Figure

One run directory in, one SVG out: the diagnosis pipeline as a reader sees it —
what the probes asked, which signal survived correction, which hypothesis held up
on held-out cases, and what the accepted repair did to accuracy.

## When this applies

The user points at a **finished run root** — the directory holding `summary.json`
and `logs/`, i.e. what `examples/benchmark/_common/runner.py` writes to
`<modality>/<family>/outputs/<model>/<dataset>[.<tag>]/`. If there is no
`summary.json`, there is no figure to draw yet; say so instead of guessing.

## Step 1 — extract the numbers

Never read the run's raw artifacts by hand. Run the extractor:

```bash
python examples/benchmark/tools/extract_figure_data.py <run-root> -o figure_data --format json
```

`figure_data.json` is the only source of numbers for the figure. Its shape is
documented in [`../README.md`](../README.md); the fields each card needs are named
in [`references/figure-spec.md`](references/figure-spec.md).

If the extractor prints `warn:` lines, read them — a missing block means that card
has no data and must be drawn empty rather than filled in from imagination.

## Step 2 — draw the SVG

Follow [`references/figure-spec.md`](references/figure-spec.md) for the layout
coordinates, the palette, and which JSON field goes on which card. Two finished
figures are in `references/` as templates — `casestudy_chartqa.svg` (image task)
and `casestudy_mmau.svg` (audio task) — drawn from the two runs in
[`../samples/`](../samples), so a template can be read side by side with the JSON
that produced it. `references/qualitative_vlm_L2.pdf` is the target style.

Write the SVG to a file and tell the user the path. Do not inline it into chat.

## The rules that make the figure honest

These are not stylistic. A case-study figure is read as evidence, and each of
these is a way a truthful pipeline can produce a lying picture.

**Every number comes from the JSON.** If a number is not in the extract, it does
not go on the figure. Leave the gap and say what is missing. Never round a
`p_value` down to `0` — below 0.001 write `p ≪ .001`.

**`qa_flags` is binding.** Each entry has a mandatory consequence:

| flag | what the figure must do |
| --- | --- |
| `repair_without_supported_hypothesis` | M5 carries an amber callout: no mechanism was confirmed, the gain below is an empirical fix |
| `multiple_signals_survived` | M2 states how many signals survived; never imply there was one |
| `accepted_fix_breaks_cases` | the bottom prints the broken count beside the fixed count |
| `example_case_from_explore` | the last box is *the output shape this repair demands*, not a measured result for that case — never write it up as "fixed" |
| `signal_binned_for_plotting` | the bar chart's axis says these are quartile bins, not raw levels |
| `correction_family_size_differs` | M2 names the phase its forest plot is drawn from — the two splits corrected over different numbers of signals |
| `direction_mismatch` | do not draw that hypothesis as a clean finding; surface the contradiction or leave it out |
| `shared_test_between_hypotheses` | two hypotheses resting on one test are one finding — draw them as one, or say they share a predicate |

**Explore and held-out never mix.** M1/M2/M3 sit in the explore swimlane, M5/M4
in the held-out one. A number measured on explore may not be captioned as a
held-out result, and the split sizes in `run` say which is which.

**Condense, don't paste.** Hypothesis `statement` fields and repair prompts are
too long for a card; boil each down to a line. Keep the meaning, drop the prose.

## Checking your own work

Before handing the file over: every `rect` inside the canvas, no text crossing
the inner edge of its card, every card's numbers traceable to a JSON path, and
every `qa_flags` entry visibly honoured. If a card cannot fit its content, shrink
the type — never drop a hypothesis, a candidate, or a caveat to make room.
