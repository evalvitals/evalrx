# chartqa.chain1 — gemma-4-e2b-it on ChartQA/test_human

**49.2% → 79.7%** on 128 held-out cases · 45 fixed, 6 broken · accepted repair `transcribe_then_refine_zoom` (L2)

> 256 cases, split 128 explore / 128 held-out · baseline 49.2% · repair cap L3a · analyzer selection: pinned

## M1 · Probe — which analyzers ran

- **BLACK-BOX BEHAVIOR** — **selected**
  - `answer_extraction_audit`
  - `selfcheck_consistency`
  - `coverage_verification_gap`
- **INTERNAL / WHITE-BOX** — available, not used
- **MULTIMODAL** — available, not used

**The signal it found:** `coverage_verification_gap.n_unique`

| n_unique | cases | failure rate |
| --- | ---: | :-- |
| 1 | 76 | `████········` 33% |
| 2 | 21 | `███████·····` 62% |
| 3 | 13 | `█████████···` 77% |
| 4 | 6 | `████████████` 100% |
| 5 | 12 | `███████████·` 92% |

## M2 · Statistics — is the pattern real?

- **explore** — 9 signals tested (BH correction), 1 survived: `coverage_verification_gap.n_unique`
- **heldout** — 8 signals tested (BH correction), 1 survived: `coverage_verification_gap.n_unique`

Strongest confirmed effect: **+0.46** extra failure rate when `coverage_verification_gap.n_unique` is high (95% CI +0.30 to +0.61)

## M3 → M5 · Hypotheses, frozen then adjudicated on held-out

**H1 · `computation_slip` — ✓ SUPPORTED**

> Failures concentrate on questions requiring a *derived* quantity (difference, ratio, percent-of-total, sum) because the model composes an arithmetic result from two or more separately-estimated value…

**H2 · `ignored_obs` — ✓ SUPPORTED**

> A second, disjoint failure population is *stable* misperception — the model binds the question's label to the wrong bar/series/axis tick and returns the identical wrong value in all 5 samples (`n_uni…

**H3 · `language_prior_bias` — ○ INCONCLUSIVE**

> On binary/comparison questions the model applies a polarity prior rather than reading the chart, defaulting to "No" for comparison-style prompts regardless of the true relation (503, 1092 both gold-Y…

*An adversarial critic objected to 3 of 3 hypotheses; objections are recorded, not vetoes — adjudication is statistical.*

## M4 · Repair ladder

| tier | | outcome |
| --- | --- | --- |
| L1 | Prompt / instructions | tried, not selected — `visual_grounding` |
| L2 | Scaffold / tools / multi-call | **accepted** — `transcribe_then_refine_zoom` |
| L3 | Internals / read & write | regressed — `coded_pipeline` |
| L4 | Re-training | untouched |

6 candidates were tried on the explore split; the winner was then re-measured on held-out cases.

- `L2` **transcribe_then_refine_zoom** — 44 fixed / 8 broken (fixed) ← accepted
- `L2` **rebind_verify_contrast** — 29 fixed / 16 broken (partial)
- `L2` **grounded_label_binding_upscale** — 27 fixed / 16 broken (partial)
- `L2` **self_consistency_5** — 2 fixed / 0 broken (partial)
- `L1` **visual_grounding** — 5 fixed / 4 broken (partial)
- `L3a` **coded_pipeline** — 2 fixed / 9 broken (unsafe)

## The accepted repair — `transcribe_then_refine_zoom` (L2)

No parameter update; the model itself is unchanged.

1. image ops: crop_salient_region(padding=0.06, min_delta=18), upscale(factor=2.0), enhance_contrast(factor=1.4)
2. First transcribe the chart, then answer from your transcription.
3. TRANSCRIPTION: list every data mark you can see as `label -> value` (one per line), taking labels from the axis ticks and legend text exactly as printed. Include marks that seem irrelevant to the question. If there are multiple panels or series, group the lines under the panel/series name.
4. Then re-check the transcription once: are any two lines swapped, is any value assigned to the wrong label, and did you miss a mark at either end of the axis? Fix the list if so.
5. Finally answer the question using ONLY the corrected list. Match the chart's own format (bare number with the chart's precision, or the exact label text) - no units, no explanation.
6. Question: {prompt}
7. End with exactly one line: Final answer: <short answer>
8. sample 3 times, take the modal answer

<details><summary>full prompt</summary>

```
First transcribe the chart, then answer from your transcription.

TRANSCRIPTION: list every data mark you can see as `label -> value` (one per line), taking labels from the axis ticks and legend text exactly as printed. Include marks that seem irrelevant to the question. If there are multiple panels or series, group the lines under the panel/series name.

Then re-check the transcription once: are any two lines swapped, is any value assigned to the wrong label, and did you miss a mark at either end of the axis? Fix the list if so.

Finally answer the question using ONLY the corrected list. Match the chart's own format (bare number with the chart's precision, or the exact label text) - no units, no explanation.

Question: {prompt}

End with exactly one line:
Final answer: <short answer>
```

</details>

## Held-out validation

| | | |
| --- | :-- | ---: |
| unchanged model | `██████████··········` | 49.2% |
| with the repair | `████████████████····` | 79.7% |

Of 128 paired cases: **45 wrong → right**, **6 right → wrong**, 57 already right, 20 still wrong. The gain is far beyond chance.

## One case

**chartqa-human-1142** (explore split)

> Work out the ratio of the bigger segment to the smaller one? Answer with only the short answer.

- expected: `1.577`
- unchanged model answered: `1.8:1`
- media: `logs/artifacts/case_media/af398fbef42b31e9_chartqa-01142.png`

## Checks worth reading before you draw the figure

- **warning** — H2 (ignored_obs) reuses the test and effect of H1; it has no independent predicate
- **warning** — H2 expects lower_in_fail but the observed effect is +0.459
- **warning** — H3 expects higher_in_fail but the observed effect is -0.524
- note — multiplicity family size differs across phases: {'explore': 9, 'heldout': 8}
- note — the illustrative case comes from the explore split; do not present its treated output as a measured held-out result
- note — the accepted fix broke 6 previously-correct cases; report it alongside the gain
