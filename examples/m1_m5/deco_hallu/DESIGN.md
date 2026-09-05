# deco_hallu — the POPE *hallucination* subset (the hard slice)

> Third sibling of [`../deco_pope`](../deco_pope) / [`../deco_miss`](../deco_miss).
> Same DeCo scenario family ([arXiv 2410.11779](https://arxiv.org/abs/2410.11779)),
> the **hardest** failure slice: a confident *false Yes*.

## 1. Why this slice is hard

`deco_miss` is the recoverable failure — a present object answered "No". The
object's evidence exists in the image, so amplifying it (describe-first,
zoom, evidence-first decoding) flips misses without cost, and the loop closes to
a **validated fix**.

The **hallucination** is the opposite: the model answers "Yes" for an object
that is **not** in the image. There is no correct latent signal to amplify — the
wrong answer is a positive assertion, not a suppressed truth. So the open
question this example poses to the loop is: *is there any intervention up to
L3b that reduces the false-Yes rate without also destroying the model's correct
"Yes" detections?* The honest answer may well be "no, escalate to L4" — which is
itself a valid, useful outcome (it is what `deco_pope`'s mixed run concluded).

## 2. Cases — the no-free-lunch guard is built into the data

[`build_cases.py`](build_cases.py) re-slices `deco_pope`'s frozen manifest (no
GPU, images shared). The batch is deliberately mixed so a degenerate "always
answer No" cannot win — every case is scored against its **own** gold label:

| group | probe / answer | gold | role |
|---|---|---|---|
| **FAIL** | adversarial-absent → "Yes" | no | the hallucination to reduce |
| PASS | adversarial-absent → "No" | no | correct rejection (control) |
| PASS | present-object → "Yes" | yes | **correct detection — the recall a fix must not break** |

Counts (per size): 2B **41** hallucinations / 4B 35 / 8B 50, plus 80 + 80
controls. The two control types are **interleaved** in document order so the fix
module's stratified validation subset always contains present-detections — the
guard is only meaningful if a recall-breaking fix has present cases to break
(verified: an 80-case subset carries 40 FAIL / 20 reject / 20 present).

A fix is therefore accepted only if, by paired McNemar across the whole mixed
batch, it flips more hallucinations (absent→No) than it breaks detections
(present→No). A skeptical prompt that just biases toward "No" nets out near zero
or negative and is correctly rejected.

## 3. Observation-only protocol

The `ExperimentProtocol.description` states only the observable behaviour — the
model answers "Yes" for an absent object, and present-object questions are
mostly answered "Yes" correctly (so suppressing those is not an improvement). It
names **no mechanism**: no co-occurrence prior, no language prior, no layers, no
suppression, no DeCo. The loop must discover any such explanation itself. (This
follows the same answer-no-leak rule applied to `deco_pope` / `deco_miss`.)

## 4. What "closing the loop" looks like here

- **Detect/analyse**: the loop selects its own analyzers; the interesting M4
  question is whether the hallucinations carry any intermediate "No" signal
  (recoverable, DeCo-style) or are confident all the way through (irrecoverable).
- **Repair**: `run_fix` proposes interventions up to L3b, validated on the mixed
  subset. **Success = a fix with a positive, significant net flip that does NOT
  break present detections. Honest failure = no such fix → escalate to L4.**
  Both are valid; the example tests whether the loop reaches the right verdict
  and resists a degenerate recall-destroying "fix".

Reference analyses live OUTSIDE this repository so the loop's coding agents
cannot read them.

## 4b. Run record — 2026-06-13 (2B, opus-4-8-low judge/coder)

201 cases (41 hallucinations + 80 rejects + 80 present-detections, interleaved),
zero OOM. The loop closed every stage AND the no-free-lunch guard demonstrably
worked.

- **M4 verified 3/4**, converged in 2 cycles (`criteria_met`). Hypotheses were
  self-critical: the top one questioned whether its own layer-25 linear probe
  was just reading answer polarity (a confound), another flagged FAIL-slice
  heterogeneity, a third claimed the false-detections are a recoverable
  decision-threshold problem.
- **M5 surgery ran a real experiment** (returncode 0, sandbox_runs 1) targeting
  the probe-leakage self-critique and **REFUTED** the over-claim (`verdict=0.0`)
  — an honest internal check that its probe signal was not the "model knows it's
  wrong" evidence it first looked like.
- **run_fix found a VALIDATED, guarded fix**: `fixed=True`, best =
  `coded_pipeline` (L3a, opus-written), **22 hallucinations fixed / 1 detection
  broken / +0.26 / significant**. The pipeline is multi-view evidence
  consistency: re-ask under attention-crop + zoom + sharpen and require a
  majority of views to confirm presence — "a hallucinated Yes is inconsistent
  across views; a real object survives magnification". `attention_guided_crop`
  (24/5) and `upscale_verify_evidence` (13/1) also validated.
- **The guard caught the degenerate fixes**: the purely skeptical candidates
  were rejected — `skeptical_majority_vote` flipped **0 hallucinations and broke
  4 detections** (−0.05), `evidence_grounded_skeptical` 2 fixed / 4 broken
  (−0.025). A "say No more often" fix nets negative once present-detections are
  in the scoring, exactly as designed.

**Reading vs the reference**: confident hallucinations carry no intermediate
"No" to recover, so DeCo-style *logit rescoring* would not help — but
*evidence-amplifying* interventions (look closer / demand cross-view
consistency) reduce false-Yes without destroying recall, and the loop found one.
This complements deco_miss (misses fixed by surfacing evidence) and contrasts
with deco_pope's mixed run (no guard, honest L4): on a guarded, focused
hallucination slice the loop produces a real, recall-preserving repair.

## 4c. Re-run — 2026-06-17 (after removing the canned attention_guided_crop primitive)

Re-run unchanged (L3b, single round, 80-case validation) to confirm the
`fix_internals` refactor — which **deleted** the pre-audited
`attention_guided_crop` primitive (an L3a *read*) on the grounds that the agent
can author it itself against the `model_attend()` bridge — did not break this
slice. It did not: the loop still closes to a validated, recall-preserving fix.

- M4 verified 1/3, converged in 1 cycle (`criteria_met`).
- `run_fix`: **fixed=True, recommendation=None**. Two candidates cleared the
  e-value bar; both keep present-detections intact:

  | candidate (tier) | fixed / broke | effect | e-value |
  |---|---|---|---|
  | **negative_framed_vote (L2)** | **13 / 0** | **+0.163** | **585** ✅ best |
  | coded_pipeline (L3a, agent-written) | 13 / 1 | +0.150 | 78 ✅ |

- **The key check passed**: with the canned primitive gone, the **agent-written**
  `coded_pipeline` was handed the read bridge (`enable_attend=True`, tagged L3a)
  and still validated (13/1, e=78) — the self-repair path does not need a
  pre-built attention-crop tool. `attention_guided_crop` appears nowhere in the
  run. The L3b *write* primitive we kept, `visual_embedding_boost`, was tried
  three times and stayed inert/negative (1/2, 0/1, 3/1) — correctly rejected, as
  on a confident hallucination there is no latent signal to amplify.
- The no-free-lunch guard still bit: `evidence_first_zoom` broke more present
  detections than it fixed (7/8, −0.013) and was rejected.

The fixed-count is lower than the 2026-06-13 run (13 vs 22) — expected
run-to-run variance: the coded pipeline is written fresh each run, this run
converged in 1 cycle, and the winner here is actually *cleaner* (0 broken,
e=585). Conclusion unchanged, and the refactor is validated end-to-end: removing
a tool the agent can write itself did not cost the loop its repair.

## 5. Files

```
examples/m1_m5/deco_hallu/
├── DESIGN.md          this file
├── run.py             load -> observation-only protocol -> VLDiagnoseLoop -> run_fix
│                      (also: shared ReplayProbeAgent / FrozenModel for the staged scripts)
├── build_cases.py     offline slice -> hallucination FAIL + interleaved controls
├── run_m1.py          STAGE 1: M1 only, frozen to outputs/m1_state.pkl       [GPU]
├── run_fused.py       STEP 1: explore -> held-out confirm (signals + charts)  [no GPU]
├── run_m2-5.py        one-shot Step 2: M2 -> M3 -> M4 -> M5 -> Fix            [GPU + claude]
├── run_analysis.py    DECOUPLED PHASE 1: M2 stats+charts -> M3 PROPOSE, no M4/Fix [no GPU]
├── run_confirm_fix.py DECOUPLED PHASE 2: M4 confirm (reuses PHASE 1) -> M5 + Fix  [GPU + claude]
├── run_all.sh / run_from_m1.sh   one-shot wrappers (the M2-5 path)
├── run_analysis.sh / run_confirm_fix.sh   decoupled wrappers (analyse → dashboard → confirm+fix)
├── config.yaml        opus-4-8-low judge/coder, L3b fix tier, 80-case validation
└── data/cases/        frozen manifests (committed; images shared w/ deco_pope)
```

## 6. Decoupled analysis vs. confirm+fix (run_analysis / run_confirm_fix)

The default loop confirms hypotheses (M4) before any fix or dashboard. The
**decoupled** path splits that so the **analysis dashboard comes first, without
confirming**: `VLDiagnoseLoop.run_analysis()` runs M1→M2→M3 (rigorous e-BH stats +
charts + *proposed* hypotheses) and stops; the dashboard then tells the
analysis story with the hypotheses marked proposed-not-confirmed. Confirmation
(M4) and repair are deferred to `VLDiagnoseLoop.run_confirm()` + `run_fix`, which
**reuse the analysis artifacts** (`outputs/analysis/analysis_state.pkl` = the
proposed hypotheses + the exact M2 stats report) so the hypotheses confirmed are
the *same* ones the dashboard showed. M2 keeps its full rigor in both paths —
only the M4 hypothesis-confirmation gate moves out of the analysis phase. The
no-free-lunch fix guard (§2) is unchanged; it still runs in PHASE 2.
