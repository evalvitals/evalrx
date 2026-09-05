# deco_miss — the POPE *miss* subset (fixable failure slice)

> Sibling of [`../deco_pope`](../deco_pope) and [`../deco_chair`](../deco_chair).
> Same scenario family (DeCo, [arXiv 2410.11779](https://arxiv.org/abs/2410.11779)),
> a different failure slice chosen so the loop can close all the way to a
> **validated repair**, not just an analysis.

## 1. Why a separate slice

`deco_pope`'s adversarial slice is dominated by **confident hallucination**
(the model asserts "Yes" for an absent object). On Qwen3-VL that failure is a
stable, full-stack commitment — no intermediate signal disagrees with the
output — so no prompt- or decoding-level intervention flips it, and the honest
fix outcome is "escalate to fine-tuning" (run#3 of deco_pope: `verified=0`,
recommend L4). That is correct, but it never exercises the loop's repair path
to a positive result.

The **miss** is the complementary slice: a present object answered "No". Here
the evidence for the object is in the image, so an internals-write intervention
that amplifies visual evidence (or a decoding intervention that surfaces an
intermediate read) has something real to recover. This slice is therefore the
right substrate to test whether the loop can produce and **validate** a fix
(M5/`run_fix` → paired McNemar flip rate), closing detect → analyse → repair.

## 2. Cases (reuse, no re-mining)

[`build_cases.py`](build_cases.py) re-slices `deco_pope`'s frozen manifest to
the **present-object** probes — already mined greedily with their answers and
yes/no token-id sets — and relabels:

| label | probe | meaning |
|---|---|---|
| FAIL | present, answered "No"  | a missed detection (the fixable failure) |
| PASS | present, answered "Yes" | correct, same images + template (control) |

Counts (per size): 2B **59** miss / 4B **70** / 8B **68**, each split
explore/validate. Images are shared from `../deco_pope/data/images` (not
copied). No GPU is needed to build the cases.

## 3. Observation-only protocol (no answer leak)

The `ExperimentProtocol.description` states only **what is observed** — which
answers are wrong and on what inputs — and deliberately names **no mechanism**:
no layers, no "suppression", no "language prior", no "DeCo". Supplying a
suspected cause in the protocol would hand the diagnosis loop the answer and
void the test of whether it can find the cause itself. (This mirrors the
correction applied to `deco_pope`'s protocol: keep observations, drop the
"suspected mechanism" sentence.) The loop must reach any layer/decoding
explanation on its own, via the analyzers and tier-(b) probes it chooses.

## 4. What "closing the loop" looks like here

- **Detect/analyse**: M1 selects white-box analyzers (logit_lens / linear_probe)
  on its own; if the miss is a genuine "seen-but-suppressed" case, the per-layer
  readout shows the correct answer present mid-stack and absent at the output,
  and M4 reaches a *supported* (not just refuted) hypothesis with enough FAILs
  (stratified subsample, ≥ ~30 miss cases available).
- **Repair**: `run_fix` proposes interventions up to L3b. The pre-audited
  internals primitive `visual_embedding_boost` (scale image-token embeddings)
  is the available internals-write lever; a coded pipeline may also re-ask with
  evidence-amplifying scaffolds. A candidate is accepted only if paired McNemar
  on the validation split shows it flips more misses than it breaks correct
  answers.
- **Acceptance**: a validated fix with a positive, significant flip rate — or,
  if none validates, an honest escalation. Either way the repair path runs to a
  decision instead of stalling.

Reference analyses (hand-written probes, expected layer signatures) live
OUTSIDE this repository so the loop's coding agents cannot read them.

## 4b. Run record — 2026-06-12 (2B, opus-4-8-low judge/coder)

**The loop closed detect → analyse → validated repair.** 179 cases (59 miss +
120 correct), zero OOM, observation-only protocol.

- **Analyse** (from its own logit_lens readout): the misses are *bimodal* — a
  hard majority that locks "No" by layer 23–25 at prob≈1.0 (irrecoverable), and
  a late-deciding low-confidence tail (layer 27, prob≈0.75, nonzero late-drop)
  that is recoverable; the "No" reads as an LM-prior token largely independent
  of the image. M4 effects were directionally right but the 32-case analyzer
  subsample kept them under the e-value bar (describe_first e=5.33).
- **Repair** (`run_fix`, paired McNemar on the 60-case validation subset, where
  all 59 misses give real power): **fixed=True, recommendation=None**.

  | candidate (tier) | fixed / broken | effect | e-value | verdict |
  |---|---|---|---|---|
  | **coded_pipeline (L3a, opus-written)** | **12 / 0** | **+0.20** | **315** | ✅ best |
  | describe_then_decide (L1) | 11 / 0 | +0.18 | 171 | ✅ significant |
  | zoom_describe_vote (L2) | 8 / 1 | +0.12 | 5.7 | sub-threshold |
  | reason_step_majority (L2) | 6 / 0 | +0.10 | 9.1 | sub-threshold |
  | visual_embedding_boost (L3b primitive) | 0 / 0 | 0 | 1.0 | ❌ inert |
  | attention_guided_crop (L3a primitive) | 2 / **16** | −0.23 | 90 | ❌ harmful, rejected |

  The winning pipeline (written by the opus coder, unaware of any reference)
  targets the exact mechanism the loop diagnosed: emit reasoning tokens to
  *delay the binary "No" commitment past the layer-23–25 attractor*, then let
  any positive visual evidence override the prior. It recovers 12 misses with
  zero regressions (e=315). The naive internals-write primitive
  (`visual_embedding_boost`) did nothing and the attention crop was actively
  harmful — both correctly rejected, not chosen.

- **Contrast with deco_pope** (adversarial hallucination slice): there the
  honest outcome was `verified=0`, recommend L4 — no prompt/decoding fix flips a
  confident hallucination. Here, on the fixable miss slice, the same machinery
  finds and validates a real repair. The two examples together exercise both
  honest-no-fix and validated-fix paths of the loop.

## 4c. Run record — 2026-06-13 (full M1→M4→M5 closure, after defects 10 & 11)

Re-run after raising the white-box analyzer case caps (defect 10) and fixing the
surgery agent's interpreter PATH (defect 11). The loop now closes **every**
stage to a decision:

- **M4 verified 2/3** (was 0/3): with all ~59 misses in the now-128-cap
  stratified subsample, the per-strategy and per-layer contrasts cleared the
  e-value bar. Top supported hypothesis: *misses are caused by a late-layer
  (24–28) negative-answer prior that overrides an already-correct interim
  representation* (conf 1.0). Loop converged in **1 cycle** (`criteria_met`).
- **M5 surgery RAN and produced a real verdict** (was None / then a 240s
  timeout): with a verified hypothesis, `run_m5` invoked the SurgeryAgent, whose
  CLI agent wrote an 80-line `import evalrx` experiment — `m.final_norm()` +
  per-layer `forward` readout, splitting cases into late-decided (layer ≥26,
  late_drop>0.15) vs early-decided. `returncode=0, sandbox_runs=1, verdict=0.0`
  → it **REFUTED** the over-specific quantitative form (only 6 cases were
  late-decided, and the early-decided group actually failed *more*: 0.72 vs
  0.50). An honest negative — the mechanism is real but the loop's first
  thresholded operationalization was too sharp.
- **run_fix validated a repair**: `fixed=True, recommendation=None`, best =
  `describe_first_cot` (L2, 12 misses fixed / 0 broken / +20% / significant);
  `upscale_describe_first` also validated (10/0). The L3b
  `visual_embedding_boost` primitive was inert again (0/0). Zero OOM.

This is the loop running every module to a decision on the fixable slice:
detect → analyse → **verify** → **surgery experiment (honest refute)** →
**validated fix**.

## 5. Files

```
examples/m1_m5/deco_miss/
├── DESIGN.md          this file
├── build_cases.py     offline slice of deco_pope manifest -> present probes
├── run.py             load -> observation-only protocol -> VLDiagnoseLoop -> run_fix
├── config.yaml        model key, opus-4-8-low judge/coder, L3b fix tier
└── data/cases/        frozen miss manifests (committed; images shared w/ deco_pope)
```
