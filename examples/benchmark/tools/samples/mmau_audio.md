# mmau.chain1 — gemma-4-e2b-it on MMAU/test-mini

**43.0% → 60.2%** on 128 held-out cases · 30 fixed, 8 broken · accepted repair `verify_then_commit` (L2)

> 256 cases, split 128 explore / 128 held-out · baseline 43.0% · repair cap L3a · analyzer selection: pinned

## M1 · Probe — which analyzers ran

- **BLACK-BOX BEHAVIOR** — **selected**
  - `answer_extraction_audit`
  - `termination_audit`
  - `selfcheck_consistency`
  - `format_sensitivity`
  - `self_consistency`
  - `calibration`
  - `coverage_verification_gap`
- **INTERNAL / WHITE-BOX** — **selected**
  - `logprob_entropy`
- **MULTIMODAL** — available, not used

**The signal it found:** `coverage_verification_gap.majority_share` — 1 of 8 signals that survived correction

| majority_share | cases | failure rate |
| --- | ---: | :-- |
| 0.2 | 26 | `███████████·` 88% |
| 0.4 | 31 | `█████████···` 77% |
| 0.6 | 11 | `█████·······` 45% |
| 0.8 | 12 | `██████······` 50% |
| 1.0 | 48 | `████········` 31% |

## M2 · Statistics — is the pattern real?

- **explore** — 15 signals tested (BH correction), 9 survived: `answer_extraction_audit.output_chars`, `coverage_verification_gap.n_unique`, `format_sensitivity.n_unparsed` …
- **heldout** — 15 signals tested (BH correction), 8 survived: `answer_extraction_audit.output_chars`, `coverage_verification_gap.n_unique`, `selfcheck_consistency.n_sentences` …

Strongest confirmed effect: **+0.41** extra failure rate when `selfcheck_consistency.n_sentences` is high (95% CI +0.26 to +0.56)

## M3 → M5 · Hypotheses, frozen then adjudicated on held-out

**H1 · `truncation` — ○ INCONCLUSIVE**

> The leaked reasoning channel is a chat-template defect (`enable_thinking=False` is silently ignored by the gemma-4-e2b-it template) that emits a multi-hundred-token `Thinking Process:` preamble, so t…

**H2 · `answer_extraction` — ○ INCONCLUSIVE**

> The grader's fallback extractor corrupts labels in *both* directions — it assigns FAIL to prose outputs whose gold string is present (`strict_match=1 & labelled_fail=1` on ≥12 cases: `a24ba06b`, `38d…

**H3 · `language_prior_bias` — ✗ REFUTED**

> Restricted to the 43 contract-respecting (`termination_class == 'clean'`) cases, answers are produced from an option-letter/text prior rather than from the audio — the model's own leaked text says so…

## M4 · Repair ladder

| tier | | outcome |
| --- | --- | --- |
| L1 | Prompt / instructions | tried, not selected — `letter_line_then_reasoning` |
| L2 | Scaffold / tools / multi-call | **accepted** — `verify_then_commit` |
| L3 | Internals / read & write | tried, not selected — `coded_pipeline` |
| L4 | Re-training | untouched |

8 candidates were tried on the explore split; the winner was then re-measured on held-out cases.

- `L2` **verify_then_commit** — 36 fixed / 9 broken (fixed) ← accepted
- `L3a` **coded_pipeline** — 27 fixed / 3 broken (fixed)
- `L2` **long_budget_final_tag** — 34 fixed / 14 broken (partial)
- `L2` **no_think_letter_vote** — 27 fixed / 8 broken (fixed)
- `L1` **letter_line_then_reasoning** — 29 fixed / 14 broken (partial)
- `L1` **answer_first_hard_contract** — 15 fixed / 13 broken (partial)
- `L2` **self_consistency_5** — 3 fixed / 7 broken (unsafe)
- `L1` **fewshot_format_anchor** — 15 fixed / 39 broken (regressed)

## The accepted repair — `verify_then_commit` (L2)

No parameter update; the model itself is unchanged.

1. {prompt}
2. Procedure (keep the whole reply under 120 words): 1. In one sentence, state only what you actually hear in the clip (events, instruments, voices, tempo feel, room/background) — do not mention the options yet. 2. In one sentence, check each option against that description and rule out the ones contradicted by it. 3. Then stop reasoning and output the final line in exactly this form: FINAL: <letter> with <letter> one of A, B, C, or D. Guess your most likely option rather than refusing; the FINAL line is mandatory and must be the last line.
3. sample 3 times, take the modal answer

<details><summary>full prompt</summary>

```
{prompt}

Procedure (keep the whole reply under 120 words):
1. In one sentence, state only what you actually hear in the clip (events, instruments, voices, tempo feel, room/background) — do not mention the options yet.
2. In one sentence, check each option against that description and rule out the ones contradicted by it.
3. Then stop reasoning and output the final line in exactly this form:
FINAL: <letter>
with <letter> one of A, B, C, or D. Guess your most likely option rather than refusing; the FINAL line is mandatory and must be the last line.
```

</details>

## Held-out validation

| | | |
| --- | :-- | ---: |
| unchanged model | `█████████···········` | 43.0% |
| with the repair | `████████████········` | 60.2% |

Of 128 paired cases: **30 wrong → right**, **8 right → wrong**, 47 already right, 43 still wrong. The gain is far beyond chance.

## One case

**12b245bb-65b5-4ffc-8743-3e8c4481bfb5** (explore split)

> How many times did the cat meowing sound appear? (A) 1 (B) 2 (C) 3 (D) 4 Listen to the audio and reply with only the option letter (A, B, C, or D).

- expected: `A`
- unchanged model answered: `B`
- media: `logs/artifacts/case_media/8da763f8ed4a6956_12b245bb-65b5-4ffc-8743-3e8c4481bfb5.wav`

## Checks worth reading before you draw the figure

- **warning** — H3 expects higher_in_fail but the observed effect is -0.296
- note — the illustrative case comes from the explore split; do not present its treated output as a measured held-out result
- note — 8 signals survived correction; the figure plots coverage_verification_gap.majority_share and must say so rather than implying a single finding
- **warning** — a repair was accepted although no hypothesis reached 'supported'; present the gain as an empirical fix, not as a validated mechanism
- note — the accepted fix broke 8 previously-correct cases; report it alongside the gain
