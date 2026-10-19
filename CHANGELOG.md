# Changelog

All notable changes to EvalRX will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed — triage spends by measurement: chunked steps, no per-analyzer cost prior

Steps run in chunks of `--triage-chunk` cases (default 8, FAIL and PASS
alternating); the budget is checked before every chunk on the cost measured
so far, so a step whose cost nobody knew overshoots by at most one chunk, and
a chunk that raises stops the rest of the step. Cost estimates no longer use
per-call rates or `calls_per_case` (kept for reservations only): a step that
has run is priced at its own measured cost per case, one that has not at the
run's median case (ranking) or its most expensive case (budget checks). A
calibration on Qwen3.5-0.8B showed why: tools' cost ratios hold within a
family on one model (generate tools ~ the number of generate calls, forward
tools within ~15% against `embed_lens`) but not across input sizes
(`patchscope` 3.9-16.7x `embed_lens` between 72 and 985 media tokens), and
per-case cost also grows with batch size for several tools. The scored
scheduler decides one chunk at a time (scoring `mvp3-score-3`): T and D are
earned in proportion to the share of the route's remaining FAIL cases a chunk
covers, and further chunks of a step it already took do not count against
`--triage-max-actions`.

### Added — triage competing explanations: D for the scored scheduler (MVP-3, second part)

`--triage-explanations judge` (with `--triage-scheduler scored`) has the
chain's judge propose, per active route, up to three competing local
explanations of the route's FAIL cases, each predicting `up` / `down` /
`same` / `unknown` for the readings of every tool still to run there (FAIL mean
minus PASS control mean against a band fixed per reading in
`probing/explanations.py`, version `mvp3-explain-1`). The judge is asked once
the sample has settled and again, at most three asks, when the evidence has
decided a prediction and a route with tools left has fewer than two live
explanations. Code checks each prediction on the 95% interval of the difference
(hit inside the predicted region, miss outside, else undetermined); a miss
contradicts the explanation. A candidate's D is the share of pairs of its
route's live explanations whose predictions for it conflict (scoring version
`mvp3-score-2`). The judge sees route summaries and readings only, never a
case or CONFIRM; each call's prompt, reply and cost is stored under
`probing/explanations/`, so a resumed run reads it back. The report is
`triage_report.scored_d.*`; every report now carries `outcomes` (each card
reading's FAIL-vs-control difference per route), and `compare.py` counts the
pairs of the explaining run's explanations each run's evidence tells apart.
Only each open layer's current candidate is offered for prediction. On
Qwen3.5-0.8B POPE (claude-opus-5 judge, two asks, 169 s) D reorders the
follow-ups but changes neither the set that runs nor the pairs of explanations
the final evidence tells apart (4/7 for every scheduler); under a tight budget
the judge's time costs coverage. The re-ask after the first S explanations were
all contradicted proposed one that patchscope then confirmed out of sample
(gold readable at no depth: peak probability FAIL - control -0.85). D does not
yet discount predictions the route's case count cannot decide.

### Changed — triage budgets charge the evidence read

`--triage-budget-sec` / `--triage-budget-usd` are checked against the evidence
the run reads, each observation at its first execution's cost, plus judge
calls (`cost.charged` in the report), not against everything the ledger has
recorded across invocations: a resumed run or a second scheduler on the same
ledger now decides as the first run did.

### Added — triage scored scheduler (MVP-3, first part)

`--triage-scheduler scored` replaces the rules' rounds with one decision at a
time: every feasible candidate (each active route's open or stale layer, the
expansion, the deferred battery steps) gets
`U = q·[D + T + C − R]₊ / (c + ε)` (`probing/scoring.py`, version
`mvp3-score-1`) and the highest wins, ties going to the soft order; taking a
later layer first is recorded as a soft-order exception. D is 0 until
competing explanations with predictions exist; T is 1 for extending a
`reuse` step to a route's new cases or a contradiction, 0.5 for an expansion
that may close a recorded gap; C is the coverage gap over `layer × gold
stratum` units the action removes; R filters an observation that already
exists; q is `(successes+1)/(attempts+2)` over the analyzer's distinct
observations (unsupported is not an attempt); c is the step's own cost per
case, else measured per-call rates, on the primary axis (USD when every
action is priced, else wall seconds). Every 10th decision explores with a
seed; no candidate above the gain floor (0.01) stops the run. While an
expansion can still follow, no batch-dependent candidate is taken (its reading
would be superseded; its layer waits rather than falling back). On
Qwen3.5-0.8B POPE it reaches the rules' follow-ups, readings and call counts
exactly, with twice the decisions (each expansion makes it extend what it ran
on the smaller sample): with D = 0 it only reorders. Decisions read
only the scheduler's own evidence and price candidates as if fresh, so a
shared ledger saves compute without changing choices; the report
(`triage_report.scored.*`) lists every candidate's components per decision.
Both schedulers' reports now carry `cost.evidence` (the first-execution cost
of every observation read), `cost.controller_sec` and `coverage`;
`probing/compare.py` compares two reports. `--triage-budget-usd` caps USD the
way `--triage-budget-sec` caps seconds. Adoption of earlier chunks is limited
to chunks inside the target, so no case enters the evidence without a
decision that asked for it.

### Added — `prompt_contrast(gen_kwargs=...)`; describe-first gets its own output room

`prompt_contrast` takes per-call generation settings for every re-ask
(baseline included); without them it behaves as before (Qwen3.5-0.8B: every
reworded and threshold answer identical to the earlier run). The triage's
describe-first candidates ask for `max_new_tokens=256`, so they run under
POPE's 16-token cap where the backend honours per-call settings; on a
backend that drops them (OpenAI-compatible endpoint) the step runs without
them and is gated on the host's cap. Follow-up readings of a strategy that
asks for text before the answer add `extra_words_*` and `complied_*` (at
least 5 more words than the re-asked question), and the report warns when
most route cases did not comply: Qwen3.5-0.8B answers "No" straight away
(0/17 complied), so its unchanged answers do not test the question;
gemini-3.1-flash-lite describes first (15/15, about 21 extra words).

### Added — triage expansion and rule-chosen follow-ups (MVP-2)

`--m1-policy triage_active` (policy `mvp2-0`) keeps the nine-step battery and
then lets fixed rules decide what runs next (`probing/rules.py`, version
`mvp2-rules-1`). **Expansion** is a controller action: while a gap holds that
more cases can close — a positive FAIL-minus-PASS routing difference whose 95%
Newcombe interval still includes zero, or a gold stratum with fewer than 8 S
or stable PASS cases — the nested sample doubles up to `--triage-max-fail`
(default 32). A step's observations are now chunks over disjoint case sets
(`probing/evidence.py`): steps whose per-case rows do not depend on the batch
(`PlanStep.reuse`; audits, counterfactual swap with partition donors,
knockout, re-sampling, wrappers) run on the new cases only; `embed_lens`
and `patchscope` (question words inferred from the batch) are not read by any
route, so they run once, on the sample the expansions settle on. An observation index by step identity without the cases lets a
later invocation with a larger cap reuse earlier chunks. **Follow-ups**: each
route with at least 3 FAIL cases proposes, per round, the next candidate of
its first open layer (default representative, then fallbacks), on its FAIL
cases plus gold-matched PASS controls; a failed or refused candidate falls
back within its layer, a valid one advances, candidates the model cannot run
are passed over at no cost, and a layer done on fewer cases is extended
first. `--triage-max-actions` (default 12) caps the decisions; `--triage-budget-sec`
also skips an action whose estimate exceeds what is left. Every decision is a
trace event naming its rule, condition and what it passed over; the report
adds sample revisions, decisions, descriptive follow-up readings (route FAIL
vs controls) and every route's layer states. Report keys changed: `steps` is
now `executions`; `next_candidates` and `probing/proposals.py` are gone (the
rules run the candidates instead of listing them). An edit refused on any
chunk refuses the step on every case. The MVP-0 entry point
`run_triage_mvp0` is the same controller.

### Added — triage on API models: provider usage, dollars, next experiments (MVP-1)

The gemini runtime now sums the token counts every response reports
(`state["usage"]`: prompt, completion and thought tokens, plus responses that
reported none), and `APIModel.usage_counters()` exposes them with the request
and retry counts. The triage executor differences two reads around each
action (`cost["api_usage"]`), the runner does the same around Stage 0
(`baseline.json` gains `api_usage` and `usd`), and `usd` comes from a dated
price table (`models/backends/api/prices.py`: only prices read off the
provider's page, checked 2026-10-17; an unlisted model's cost is `None`, never
a guess, and the input part is an upper bound because implicit cache hits are
not reported). The report adds a cost summary for the invocation. On the
gemini backend the U sampling check now runs: the runtime merges per-call
kwargs over its own sampling, so `self_consistency` gets `temperature=1.0`
(the endpoint backend drops per-call kwargs and stays untested). The report
also listed rule-based next candidates for every route with FAIL cases, each
gated on the actual model (superseded in MVP-2, which runs them).

### Fixed — logit margins compared an answer with its own other spelling

`build_margin` (every white-box analyzer's target: knockout, causal trace,
patchscope, gradcam, attribution and path patching, ...) compared the target
token with the strongest OTHER token. A BPE vocabulary holds `"Yes"`,
`" Yes"`, `"yes"` and `" yes"` as different ids, and a model can split one
answer across them: Qwen3.5-0.8B on POPE gives `" Yes"` 0.80 and `"Yes"` 0.20
on a correct case, so the gold `"Yes"` margin compared "Yes" with " Yes" and
came out negative (−1.375) on a correct answer; `target="prediction"` had the
same problem with its top-2. The comparison token is now the strongest token
outside the answer's surface forms (`answer_variants`: with or without a
leading space, lower-case or capitalised for answers longer than one
character, each confirmed by the decoder; without a decoder only the literal
form), the gold target is the form the clean run scores highest, and
`LogitMargin.variants` lists the forms. On that case the gold margin is now
+7.875 (" Yes" vs " No"). Patchscope sums the readout probability over the
forms and counts a layer as reading the answer when any form ranks first.
Knockout and other margins measured with a decoder-backed tokenizer before
this change are not comparable with new ones.

### Added — triage U and S routes (policy `mvp0-2`)

The triage plan now runs five more actions after the knockout. U (an answer
that moves) has two required parts: `self_consistency`, run once per case
(the executor's new `per_case` mode, since the analyzer reads only
`cases[0]`) at the host's sampling settings (`temperature=1.0` plus the run's
`top_p`/`top_k` on local backends; greedy settings leave it untested), and
`prompt_contrast` with the question kept verbatim inside two neutral wrappers
(a wrapper whose answer is unreadable is an invalid control). Answers are
compared with the task's new label-blind `answer_key` (`benchmark.scoring`,
`tasks.answer_key`), never graded. S is a FAIL case on which H, N (cases with
media; N is now `not_applicable` without media) and both U parts ran and none
fired; a missing check leaves it in X, and a PASS case with every check clear
has no route. The S readings on every case are descriptive: a threshold
instruction (`prompt_contrast` "sensitive", yes/no tasks only; others record
the step as `not_applicable`), `embed_lens(layer=0.9)` and
`patchscope(target="gold", media_layers=3)`. Both lenses gain a question-word
reading — prompt words found in at most half of the batch's distinct prompts
(`_internals/words.py`), so template words never count: `el_query_sim`, and
`ps_media_query_hits`/`_share` (media tokens decoded as the case's question
word). Patchscope's patch self-check now raises `FidelityError` (still a
`RuntimeError`) and reports `patch_fidelity`, so the triage records an
unfaithful patch as `unfaithful`. `prompt_contrast` also keeps untruncated
outputs as an `outputs` artifact, and the executor keeps a step's named
result artifacts with its findings (`keep_artifacts`).

### Added — counterfactual swap donors (`modality_ablation(donor_rule="counterfactual")`)

An unchanged answer after a media swap is only evidence when the donor's
media should have changed it. On yes/no and few-option tasks a random donor
often leaves the correct answer unchanged (a random COCO image usually lacks
the object POPE asks about, so "No" stays right). The new rule first looks,
in `donor_pool` (default: the probed batch), for a case asking the identical
prompt about different-content media whose gold differs from the model's own
unablated answer (not from the case's gold: on a failed case those differ, and
a donor whose gold equals the wrong answer makes "unchanged" correct); that
gold is what a correct answer must become, recorded as
`ablation_<slot>_donor_expected`. Otherwise it falls back to the content rule
and records `ablation_<slot>_donor_kind="content"`. `n_counterfactual_<slot>`
counts the former. The default `donor_rule="content"` is unchanged. The triage
controller uses the counterfactual rule with the whole EXPLORE partition as
the pool; on yes/no and letter tasks, N is tested only on counterfactual swaps
(content-only swaps leave it untested with the reason), and the report adds
donor kinds and how often a counterfactual swap moved the answer to the
donor's gold.

### Changed — edit-free self-checks run on every analysed case

`attention_knockout`, `attention_sensitivity`, `attention_temperature` and
`generic_attention` used to check the attention tap (and, on hybrid stacks, the
media-absent linear re-run) on the first analysable case only, and the verdict
was cached on the model — so whether a whole batch was accepted depended on
which case came first. On Qwen3.5-0.8B in bf16 the tap differs from `sdpa` by
0.8–2.0% of the largest logit depending on the input, straddling the 1%
tolerance. A new `FidelityGate` (`analyzers/_internals/edits.py`) now checks
every analysed case before measuring it; the first failing case refuses the
whole batch with a `FidelityError` naming it. `tap_fidelity` reports the worst
case per mechanism, the number of argmax changes and every per-case check.
`check_fidelity` / `check_linear_fidelity` are no longer cached and
`check_fidelity` accepts the caller's native logits (`base=`) to save a
forward. Cost: one extra tapped forward per case (plus one linear re-run on a
hybrid stack). Under `sdpa` these analyzers will now usually refuse on
Qwen3.5; use `--attn-impl eager`, where the tap is exact. The triage
controller's action identity now includes a content hash of the analyzer's
module and `analyzers/_internals`, so a code change re-runs the action instead
of reading back an old observation.

### Added — triage probing controller, MVP-0 (`--m1-policy triage_active`)

An opt-in alternative to the judge-selected M1 (`evalrx/eval_agent/probing/`).
After Stage 0 and the usual EXPLORE/CONFIRM split, a host-side controller runs
a fixed plan — `answer_extraction_audit`, `termination_audit` (no
continuation), `modality_ablation(mode="swap")`, `attention_knockout(target="gold")`
— on a matched, nested FAIL/PASS sample of EXPLORE, exactly as planned (no
backfill, cap or generated probes), and writes `selection_trace.jsonl`,
`cost_ledger.jsonl`, per-action observations and a triage report (H/N/X route
labels traced to case, field, action and fidelity verdict) under
`<run>/probing/`. Outcomes are typed (`ok` / `unsupported` / `failed` /
`unfaithful`) and never written as zeros; a rerun reads finished actions back
instead of re-running them. No judge is called and M2–M5 do not run. The
default `--m1-policy judge` path is unchanged.

### Fixed — `modality_ablation(mode="swap")` could swap in the same media

The donor was the first other case filling the slot, compared by case id only:
on POPE (two questions per image, adjacent rows) the first pair swapped the
same image back in, and every other case shared one donor. Donors are now the
next case in batch order whose media differs in content (file bytes or
pixels), each row records `ablation_<slot>_donor` and the compared answers, and
`n_drop_fallback_<slot>` counts cases that had to fall back to drop. `drop`
mode output is unchanged.

### Changed — fidelity refusals raise `FidelityError`

`require_fidelity` and `require_edit_fidelity` raise
`evalrx.analyzers._internals.attn_tap.FidelityError` (a `RuntimeError`
subclass carrying the check's `verdict`), so callers can tell an untrustworthy
edit from an ordinary failure without parsing the message.

### Fixed — hybrid analyzer compatibility with Transformers 5

Recognize Qwen3.5 / Qwen3-Next linear-attention mixers when Transformers calls
`torch_chunk_gated_delta_rule` at module scope instead of storing a kernel on
each instance. Gradient relevance now taps that call during the target mixer's
forward, restores it on exit or error, and retains the older instance-kernel
path. Regression tests check native logits and gradients, per-layer relevance,
model isolation, and cleanup after a failed forward.

### Added — write-side white-box analyzers (knockout, causal trace, gradients, head attribution)

Four registered stubs are now native implementations, plus three new
analyzers. All explain the next-token logit margin at the end of the prompt
(`target="prediction"`: clean top-1 vs top-2, label-free; or `"gold"`) and emit
`per_case` columns for M2:

- `attention_knockout` (new, `patching/`) — cut text→media attention in all
  layers and per third of the stack (Geva et al. 2023).
- `attention_sensitivity` (new, `patching/`) — rescale text→media attention
  (×0.5 / ×2, renormalised) and report the margin's slope; negative = more
  looking weakens the answer (PAI-style repair signal).
- `causal_trace` (was a stub) — Meng-style corruption of the media tokens
  (noise, incl. Qwen3-VL DeepStack injections; or a same-size gray image) and
  per-layer restoration at the media tokens and at the answer position.
- `gradcam` (was a stub) — Grad-CAM over the decoder's image tokens at a
  fractional depth, mapped onto the patch grid with photo overlays.
- `generic_attention` (was a stub) — Chefer et al. 2021 relevance, propagated
  as one row from the top layer down.
- `head_attribution` (new, `lens/`) — direct logit attribution per attention
  head with the final norm frozen at the clean run.
- `tuned_lens` (was a stub) — per-layer affine translators fit label-free to the
  model's own final distribution; cached per model, optional save/load.

`attention` and `attention_sink` now run over the batch (`max_cases`) and emit
`per_case` columns (`media_mass_mid`, `sink_mass_mean`, …) instead of reading
only the first case.

Plumbing lives in `evalrx/analyzers/_internals/`: an attention function
registered with transformers' `AttentionInterface` (plus the matching mask
function — without it transformers builds no causal mask) that reproduces eager
attention and can block / reweight / record it, verified once per model to
engage on every layer and reproduce the native logits; decoder-layer pre-hooks
for reading/replacing the residual stream; per-head output capture.
Attention-editing analyzers refuse latent attention (hybrid stacks: see below).

### Added — the write-side analyzers run on hybrid stacks (Qwen3.5 / Qwen3-Next)

Qwen3.5 interleaves softmax attention with gated-delta-rule (GatedDeltaNet)
layers that have no attention matrix; the analyzers used to refuse it. New
`_internals/linear_mix.py` covers those layers with the module's own code:

- media edits (`attention_knockout`, `attention_sensitivity`) re-run a linear
  layer on the sequence WITHOUT the media tokens and blend its output at the
  text positions (`w = 0` = knockout); `_internals/edits.py` applies the tap
  and this re-run together, and checks both once per model (the re-run must
  reproduce the outputs before the first media token);
- `generic_attention` materialises each linear layer's token-mixing matrix
  (the delta rule is linear in its values: `o = A·v`) and uses
  `∇A = (∂y/∂o) vᵀ` in Chefer's rule;
- `head_attribution` attributes the linear layers' value heads too
  (`dla_linear_share`, `layer_kinds`, `heads_per_layer`).

`causal_trace`, `gradcam` and `tuned_lens` were already architecture-agnostic.
Verified on Qwen3.5-2B (transformers 5.14): tap 6/6 softmax layers exact,
re-run 18/18 linear layers within 0.24% (bf16). Tests: a tiny Qwen3-Next
(transformers ≥ 4.57) and a tiny Qwen3.5 VLM (transformers 5).

### Added — `attribution_patching`: causal tracing of single tokens, from gradients

New `patching/attribution_patching` estimates the effect of restoring each
single token at each layer — `causal_trace`'s residual-stream site, corruptions
and noise draws — as `(h_clean − h_corrupt) · ∂margin/∂h`, with the gradient
averaged along the corrupt → clean media path (EAP-IG, `ig_steps=5`;
`ig_steps=1` = plain attribution patching, `direction="denoise" | "noise"`).
It reports the answer position's share of the media's effect per layer
(`ap_last_rise_frac`, `ap_last_share_mid`) and the strongest single relay token
(`ap_relay_token`, `ap_relay_share`, `ap_relay_depth_frac`), with a
cross-case tally of relay tokens.

- Each case is checked against exact patches of the sites it reports (the last
  position at `n_verify` layers, the relay token at its peak). Cases whose mean
  gap exceeds `verify_tol` (0.25) are flagged `ap_verified=False` and kept out
  of the aggregates.
- Measured on Qwen3-VL-2B and Qwen3.5-2B (8 deco_pope cases): the media tokens'
  joint patch is not additive, and the sum of their gradient estimates does not
  track it (r < 0.4 under every setting), so it is not reported. Single-token
  estimates do track the exact patch where corrupting the media lowers the
  margin (last position r 0.92–0.96), and fail where it makes the model more
  confident (a saturated margin). With the shipped defaults, 2/6 (Qwen3-VL) and
  4/6 (Qwen3.5) traced cases pass their own check; 19 s / 55 s per 8 cases.
- `causal_trace`: the corrupted-variant construction is now
  `corrupted_variants`, shared with the new analyzer (behaviour unchanged; the
  exact checks reproduce `causal_trace`'s numbers digit for digit).
- `_internals/residual.py`: `keep_grad` also works with frozen weights (the
  captured state becomes a leaf).

### Added — `attention_temperature` and `embed_lens`

- `patching/attention_temperature`: divides the pre-softmax scores of the text
  after the media by a temperature (`sharp` 0.5 / `flat` 2.0, by band) and
  reports the margin's slope in log T. Default `scope="within"` tempers only the
  media keys and keeps each row's total media weight — the model looks at the
  image as much as before, only more focused or more spread out (FOCUS; the
  AMOUNT is `attention_sensitivity`'s). `scope="all"` is plain temperature.
  New tap edit kinds `temp` / `temp_within` (`attn_tap.temperature`), exact to
  float precision (tested: `T=1` no-op, media mass kept, `p^(1/T)` inside,
  `softmax(s/T)`). Softmax layers only: on Qwen3.5 6 of 24 layers are edited.
- `lens/embed_lens` (Fan et al., CVPR 2026): each media token at the decoder
  input, read through its nearest input-embedding words (cosine). Tokens are
  split into recurring (nearest word shared across ≥ `recurring_frac` of the
  batch's images), high-norm (≥ `norm_ratio` × the image's median) and specific
  — the paper's sink / dead / alive, inferred from the batch's DISTINCT images
  instead of checkpoint-specific ids — and each prompt content word gets the
  best cosine any media token reaches to it (grounding). `layer` reads any
  depth (default 0, the decoder input, as in the paper).
- Measured on Qwen3.5-0.8B (POPE adversarial):
  - `attention_temperature` (16 cases; tap exact; within-media mean slope
    +0.33, no flips; plain temperature flips 19% when flattened).
  - `embed_lens` (100 cases): at the input Qwen3.5's image tokens are not in
    vocabulary space — best cosine 0.15, 90% recurring (space, digits,
    `<|endoftext|>`), the queried object's word no closer when present
    (AUROC 0.47). At `layer=0.9` 87% are image-specific (rocky, snow, wooden,
    buses, ...) and object grounding reaches AUROC 0.76 (0.82 at the last layer).
- Routing: `attention_temperature` after `attention_sensitivity` and under
  `attention_sink` / `insufficient_visual_attention` / `mislocalized_attention`;
  `embed_lens` after the gradient maps and under `attention_sink` /
  `visual_information_loss` / `grounding_error`.

### Added — `path_patching` and `patchscope`

- `patching/path_patching` (Wang et al., IOI): which heads carry the media's
  effect directly to the answer. One head's output from the media-corrupted run
  (`causal_trace`'s corruptions), every other token-mixing head frozen at its
  clean output (the value heads of linear-attention layers included), MLPs and
  norms recomputed. Heads are screened by direct logit attribution of their
  media-dependent output and the `top_k` (8) patched exactly; per case the top
  direct-path head, the joint direct / total share of the top-k, the screen's
  rank agreement and (hybrids) the linear-attention share; across cases the
  recurring heads. Patching every head at once reproduces the full effect at the
  answer position — tested as an identity, on hybrids too.
- `lens/patchscope` (Ghandeharioun et al.): each layer's residual at the answer
  position is patched into a token-identity prompt at the same layer and decoded
  by the model: the depth where the answer first ranks first, its peak, and
  `ps_late_drop` (peak minus the output probability; with `target="gold"`, a
  right answer formed then suppressed). `media_layers > 0` also decodes every
  image / audio token into a word ("what it sees").
- Shared plumbing: `_internals/heads.mixer_hooks` (read / overwrite every
  head's output at every position), `heads.readout_direction` (the frozen-norm
  margin direction, now also used by `head_attribution`),
  `residual.batched_patch` (a different residual patch per batch row).
- Measured on Qwen3.5-0.8B (16 POPE-adversarial cases): the screen ranks heads
  like the exact patch (Spearman 0.86); the direct carriers recur in the last
  two layers (`L23.H6` +0.33 of the effect in 8 cases; `L23.H0` −0.24 in 10 —
  it consistently works against the image's effect). Patchscope's top-1 matches
  the model's answer from layer 18 in 14/16 cases; the logit lens only at the
  last layer, 6/16. Image tokens decode to scene words ("snow", "board",
  "skiing", "slope" for a snowboarding photo).
- New spec `qwen3.5-0.8b-vl` (same 24-layer hybrid layout as the 2B; the fast
  real-model check).
- Routing: `path_patching` after `attribution_patching` in the image / audio /
  video lists and under `answer_head_conflict` / `visual_information_loss`;
  `patchscope` after `tuned_lens` in the text list and under `late_suppression`,
  `premature_layer_divergence`, `logit_lens`.

### Changed — routing reaches the white-box analyzers

`StrategyProbe` ranks the new analyzers for white-box image / audio / video
runs (`attention_knockout` right after `modality_ablation`), and M3 failure
modes promote them: `language_prior_bias` and `hallucination` now include the
media interventions, and new tags `visual_neglect`, `audio_neglect`,
`insufficient_visual_attention`, `visual_information_loss`,
`mislocalized_attention`, `late_suppression` (also offered to the M3 prompt)
route to the analyzer that tests them. The video list named the unregistered
`relative_attn` (silently dropped); it is now `relative_attention`, and a test
fails on any unregistered name in the routing tables. `attribution_patching`
follows `causal_trace` in the image / audio / video lists (the video list now
has both) and under `visual_information_loss` / `late_visual_integration`.

### Changed — `hf_local` provides `GRADIENTS`

`HFLocalModel.forward(spec=CaptureSpec(with_grad=True))` now keeps the autograd
graph (it was always `no_grad`, and `with_grad` was ignored), and `hf_local`
declares `Capability.GRADIENTS`, so `gradcam` / `generic_attention` are offered
for local models. The default forward is still `no_grad`. Gradient analyzers use
`torch.autograd.grad` on hooked activations and never write parameter `.grad`.

### Docs — analyzer zoo regenerated from the registries

`docs/analyzers.md` listed 29 of the 46 registered analyzers and 16 of the 57
model specs, and several listed rows had drifted from the code (`rollout` is
registered as `attention_rollout`; `linear_probe` is implemented, not a stub;
`mm_shap` and `relative_attention` cover more than images; two citations pointed
at the wrong paper). The page is now organised by `evalrx/analyzers/` subpackage
and by model group, and its tables are generated by `tools/gen_analyzer_docs.py`
(`--check` exits 1 when the page is stale;
`tests/test_analyzers/test_analyzer_docs.py` runs the same check).

## [0.1.2] — 2026-09-11

First release under the `evalrx` name (formerly published as `evalvitals`
0.1.0/0.1.1) — see "Changed — project renamed EvalVitals → EvalRX" below.
Versioning continues `evalvitals`'s sequence (0.1.0, 0.1.1) rather than
restarting, since `v0.1.0`/`v0.1.1` are already taken as git tags in this
same repository's history.

### Breaking — `RunLogger` (V1) removed; `RunLoggerV2` is the only run logger

`RunContext` no longer takes a `logger_version` argument — there is nothing
to select between. Removed outright: `evalrx/eval_agent/run_logger.py`
(the `RunLogger` class), its own test suite, and every "try V1, fall back to
V2" branch in the report/observability readers (`evalrx/reporting/dynamic.py`,
`html_report.py`, `case_study.py`, `server.py`, `evalrx/observability/tracer.py`,
`evalrx/analysis/dashboard.py`) and in
`examples/dataset_selection/llm_benchmark/run_pipeline.py`'s own confirm-only
reloader. A run written by V1 — `run_log.jsonl`, `manifest.json`,
`README.txt`, `figures/`/`prompts/`/`experiments/`/`tools/`/`workspace/`/
`fixes/` — is no longer readable by anything in this package; re-run it under
`RunLoggerV2` (`run.json` + `M1/log.json`..`M5/log.json`) if you need it in a
current report/dashboard. `log_schema.py`'s `build_schema`/`build_v2_schema`
are merged into one `build_schema()`; `RUN_LOG_SCHEMA_VERSION` bumped **5 → 6**
(the merged envelope now unconditionally requires `event_seq`/`stage`/
`span_id`) and moved from `run_logger.py` to `log_schema.py`. See
`evalrx/eval_agent/RUN_LOGGER_V2.md` for the full layout and migration
history. `examples/benchmark/tools/extract_figure_data.py` (a standalone
tool outside the `evalrx` package) got its own, independent V1→V2 conversion
in the same pass, including two real reads it was still doing from a
sibling-file convention V2 never produces (per-analyzer `.result.json`,
per-phase `_m2_stats_results.json`) — now reads `M1/log.json`'s inlined
per-analyzer results and `M2/log.json`'s inlined `stats_results` instead.
Also fixed along the way: `evalrx/reporting/case_study.py`'s M1 signal-curve
and multiplicity-survivor helpers were calling the V1-only artifact glob
directly instead of the already-V2-aware `_m2_sources()` helper, so they
silently returned nothing for any V2 run — every V2 run's M1 bar chart and
survivor list, not just old-archive-reading.

### Changed — project renamed EvalVitals → EvalRX

Breaking. The import/package name, console scripts, and every on-disk/data
identifier changed together:

- Package: `evalvitals` → `evalrx`; `import evalvitals` no longer works.
- Console scripts: `evalvitals`/`evalvitals-explore` → `evalrx`/`evalrx-explore`.
- Env vars: `EVALVITALS_*` → `EVALRX_*` (`EVALRX_HF_CACHE`,
  `EVALRX_LANGFUSE_MODE`, `EVALRX_DATASET_SELECTION_DIR`,
  `EVALRX_VALIDATE_LOG`, `EVALRX_GIT_COMMIT`, `EVALRX_REPORT__*`) — update
  `.env` files and shell profiles.
- State/cache dirs: `.evalvitals-cache/`, `.evalvitals/langfuse_outbox.sqlite3`,
  `.evalvitals-langfuse-outbox.sqlite3`, `evalvitals_chat_output/`,
  `evalvitals_web_runs/` → the `evalrx` equivalents. Existing run trees under
  the old names are invisible to the new code; move or symlink them if you
  need the old outputs read back.
- `run_log.jsonl`: the `evalvitals_version` field on every event is now
  `evalrx_version`, and `RUN_LOG_SCHEMA_VERSION` bumped **4 → 5** per the
  documented rule (a renamed field is a breaking schema change). The
  published schema's `$id` moved to
  `https://evalrx.dev/schemas/run_log.schema.json`. The schema stays
  permissive on unknown fields, so old logs still parse; a strict
  `evalrx_version`-keyed reader needs the version bump to tell old and new
  logs apart.
- Repo: `github.com/evalvitals/evalvitals` → `github.com/evalvitals/evalrx`
  (org unchanged); the PyPI package is republished as `evalrx` (the old
  `evalvitals` releases on PyPI are not renamed in place).

### Added — the case-study sheet in the served report

A run's evidence was spread across five stage views a reader had to assemble
themselves, which is the wrong shape for the question people actually arrive
with: what happened, start to finish. `evalrx/reporting/case_study.py`
compiles the run into one sheet — what the probes asked in plain language, the
funnel from measurements taken to signals forwarded to M2, the forest plot of
what survived multiplicity correction, the hypotheses, the held-out verdicts,
the L1–L4 repair ladder with every candidate the search tried, the accepted
repair as steps, and the paired before/after — and `CaseStudySheet` renders it
directly under the journey graphic, in the report a dropped `.zip` produces.

It reads the run's own artifacts and nothing else, and accepts either level of a
zipped run (the run directory or the `logs/` inside it, both of which people
zip). It is the reporting-side twin of
`examples/benchmark/tools/extract_figure_data.py`, deliberately sharing its
block and field names; `tests/test_analysis/test_case_study_sheet.py` runs both
over one synthetic run and holds their numbers to each other, because two
readers of the same artifacts drifting apart is how one run acquires two
plausible descriptions.

The caveats travel with the sheet rather than being left to the reader: a repair
accepted with no supported hypothesis, a fix that broke previously-correct
cases, a chart plotting one of several surviving signals, an illustrative case
drawn from the explore split. A run with no probe or stats artifacts yields no
sheet at all — the section is dropped rather than rendered as zeroes.

`REPORT_DATA_VERSION` 15 → 17 and the catalog `evalrx-report@1` → `@2`: a
cached report composed against the old catalog cannot name the new component,
and a cached payload from before the sheet has no probe phrases to render.

### Fixed — `extract_figure_data.py` re-anchors `trial_root` on the run root

The run log records the writer's absolute path — inside a container that is
`/app/work/outputs/<run>/logs/...`, which exists on no host — and relpath
against it emitted a `../../..` chain whose length depended on where the reader
sat. The accepted repair's `trial_root` is now the run-root-relative
`logs/fixes/<NN_name>` whenever that directory exists under the run being read,
with the old behaviour kept for paths that genuinely live elsewhere.

### Added — `examples/benchmark/tools/extract_figure_data.py`, run dir → case-study figure data

A benchmark run's numbers were previously readable only through the dashboard or
by hand-reading `logs/artifacts/`, which is the wrong shape for producing a paper
figure. The new tool takes a run root — the directory `_common/runner.py` writes,
holding `summary.json` and `logs/` — and emits the same numbers in three shapes:
a sectioned JSON document for reading, a flat JSONL stream for plotting code, and
a Markdown write-up of the figure with character-bar charts. Standard library
only; it imports nothing from `evalrx` and reads only run artifacts.

One record per figure block: the header strip, the M1 analyzer families and the
signal bar chart, the M2 correction families and per-test rows (explore and
held-out kept apart), the M3 hypotheses with the critic's objections, the M4
held-out verdicts, the M5 L1–L4 repair ladder, the accepted repair, the paired
held-out validation, and one illustrative case. Every record carries the artifact
path it was read from, so a number in a figure can be traced back.

M1 also ships as a reader sees it rather than as field names: `probe_questions`
lists what the probes actually asked in plain language (keyed by the per-case
field, because one analyzer often asks two questions and one question is often
answered by two analyzers), over a `measurement_inventory` that states why each
measurement was kept or dropped — it saw the answer key, it never varied, it
covered only part of the batch. The funnel's two ends are the numbers M1 and M2
must agree on: `n_measured` probes taken, `n_forwarded` entering the
multiplicity-correction family, which is also the BH denominator and the forest
plot's row count.

The drawing half ships as an Agent Skill, `tools/case-study-figure/`: point it at
a run root and it runs the extractor, then draws the SVG from the JSON —
`SKILL.md` for the workflow, `references/figure-spec.md` for the layout
coordinates, palette and per-card field mapping, plus a target-style PDF and two
finished SVGs drawn from the two runs in `samples/`. Install it with `cp -r` into
`~/.claude/skills/`, or hand it to a coding agent with `explore --skill <dir>`.
The skill makes `qa_flags` binding — an example case drawn from explore may not
be written up as "fixed", a repair accepted with no supported hypothesis must
carry the callout, a fix that broke cases must print the broken count next to the
gain.

Because a figure fails silently — a plausible wrong number gets printed, nothing
raises — each extraction also self-checks into `qa_flags`: two hypotheses sharing
one test and effect, a verdict whose observed sign contradicts its stated
direction, a repair accepted with no supported hypothesis, several surviving
signals where the chart shows one, a continuous signal that was quartile-binned,
an illustrative case drawn from explore, and a fix that broke previously-correct
cases.

`tests/test_examples/test_figure_data_extract.py` pins those numbers on a
synthetic run, including an AST check that the `summary.json` keys the tool reads
are still the ones `_common/runner.py` writes.

### Added — dashboard case-study view for paper-method bench runs (audio + image)

`evalrx dashboard` now recognises a third run shape: a paper-method bench
run under `examples/` (a report with `baseline` + `auto_fix`, e.g.
`examples/agent_loop/qwen2_audio_tcd_mmau/outputs/tcd_mmau.json`). Those
reports record per-case outcomes by id only — the question, options and media
file live in the benchmark manifest beside them — so a repaired case was
previously just a bare uuid with nothing to inspect. The new
`evalrx/analysis/case_studio.py` (Streamlit-free, like `workbench.py`)
joins the two, resolving the manifest from the report's `dataset.manifest`
pointer or, for older reports, by id-coverage across the conventional sibling
locations, and normalising MMAU's `instruction`/`choices`/`audio_path` and the
VLM slices' `question`/`options`/`image` into one case view.

The dashboard renders it as three tabs (Run Overview / Case Study / Repair
Methods). Case Study plays each case's stimulus — an `<audio>` element for
audio benchmarks, the image for visual ones — beside the question and its
options, so a human can answer the item themselves; blind mode (on by default)
hides the correct answer, the model's answer and the repair verdicts until an
answer is locked in, and disables the outcome filter while on so filtering
cannot leak the answer. Human answers are scored against the model on the same
cases, in-session only. Repair Methods describes what each candidate actually
changes (mechanism, prompt template, configuration), its paired McNemar /
e-value verdict, and a per-flipped-case button that jumps to that case.
`qwen2_audio_tcd_mmau/run.py` now writes the `dataset` pointer so its reports
are self-describing.

### Added — L4 fix tier: first executable shape (LoRA on the language model)

`FixAgent`'s L4 (parameter space) tier previously only recorded a
`FinetuneSpec` recipe for a human decision. `run_lora_repair`
(`fix_internals.py`) now executes the one recipe shape this v1 supports —
`method="lora"` on `target="llm"` — end to end: LoRA is spliced into the
model's underlying HF module via `peft`, trained on a caller-supplied
diagnosis-only pool (`FixAgent(finetune_pool=...)`, never the batch being
validated), generated against the validation batch, and unloaded again —
validated through the exact same paired McNemar + e-value machinery as
every other tier, so L4 is no longer a special case in the outcome. Every
other recipe shape (`method="sft"`/`"full"`, `target="vision_encoder"`/
`"projector"`) is still recorded, not executed, with a reason naming what's
missing. `dataset_recipe` (free judge-written text) is deliberately never
interpreted — training data is a fixed default (failing cases as SFT
targets, passing cases as anti-forgetting ballast) rather than another
judge call producing executable code. New optional dependency group
`evalrx[finetune]` (`peft`). See `fix_internals.py`'s module docstring
and `FixAgent.__init__`'s `finetune_pool` parameter.

### Added — S-tier paper-method full-arc experiment on V*Bench and HR-Bench 4K

`examples/vlm_paper_benchmark` gained an HR-Bench 4K data surface (base64
image fallback and lettered-option support in the downloader), endpoint
overrides for `run_autofix.py` (`AUTOFIX_MODEL_ID` / `AUTOFIX_BASE_URL`), and
a recorded five-batch experiment (`experiments_s_tier_2026-08.md`): diagnosis
matched the DyFo/DC² failure mechanisms in every batch and
`detector_visual_search_consensus` led all five disjoint batches
(+40/−12 pooled), but combined anytime-valid evidence stopped at e = 13.85 < 20
— the gate declined to certify, which is the designed honest outcome. The runs
also surfaced that parseable judge proposals displace the paper-inspired
default candidates (workaround: `--paper-methods-only`), filed as an upstream
improvement candidate.

### Added — L2 loop policy VALIDATED by cross-task replication (the loop's first shipped fix)

The pre-registered `L2_loop_policy` arm (block identical repeats + force a
final answer) ran on five further VTC-Bench tasks at 2B — independent batches,
one arm, no peeking. It replicated in every one:

| task | n | baseline→L2 pass | effect | e |
|---|---|---|---|---|
| counting | 85 | 16→20 | +0.047 | 0.48 |
| chart | 100 | 1→23 | +0.220 | 27962 |
| color | 90 | 8→15 | +0.078 | 0.97 |
| math (MC subset) | 66 | 3→14 | +0.167 | 19.5 |
| measure | 105 | 7→26 | +0.181 | 1381 |
| spatial | 44 | 6→17 | +0.250 | 45.0 |

E-values multiply across independent batches: combined e ≈ 1.6×10¹⁰ ≫ 20 —
the anytime-valid bar is cleared and the fix is validated. Two lessons the
process itself taught: the diagnosis task (counting) was the fix's WEAKEST
batch — a single-task read would have under-sold a real repair — and the 2B
failure structure found on counting (loop until the budget dies, answer
nothing) is family-wide: fail rates 0.81–0.99, empty-answer shares up to 80%,
judge modes led by FM-LOOP almost everywhere.

### Added — Escalation results: 2B/4B/8B on vtcbench counting

Rerunning the full M1→M5 arc at 4B and 8B (`run_m1.py --model ... --out
outputs_<tag>`, parametrized `run_explore.sh`, `run_m5.py`) turned the single
diagnosis into a scaling study:

- **The capability wall is flat** — pass 16/16/18 of 85; mean k-rerun success
  rate 0.26/0.25/0.22. Dense counting does not yield to 2B→8B scaling.
- **The failure phenotype migrates**: identical-repeat loops at 2B (has_loop
  54/85, 93 policy-blocked repeats) → varied tool churn that never concludes
  at 4B (has_loop 8/85, zero repeats to block, 30 empty answers) →
  perception/reasoning-limited at 8B. The held-out-confirmed invariant at
  every scale is the budget-exhaustion family.
- **Fix efficacy tracks the phenotype**: the L1 description warning only
  helps where identical repeats exist (+0.024 at 2B, ≤0 after); the L2
  forced-answer policy GROWS with capability — +0.047 → +0.094 → **+0.141 at
  8B (30/85 vs 18, 17 fixed / 5 broken, e = 6.92)** — rescued turns convert
  to correct answers only when the model has actually seen enough.
- **Nothing ships**: e-BH rejects none at any scale (the anytime-valid bar is
  e ≥ 20; the 8B L2 arm would clear a classical McNemar at p ≈ 0.008). The
  recorded recommendation is independent replication — e-values multiply
  across batches.

Operational note: an 8B tool-shap conversation exceeded a 32k serve window;
8B runs use `--max-model-len 49152` and the precompute now stubs failed runs.

### Added — Loop-policy options + the M5 paired fix experiment (inconclusive, correctly)

The two held-out-confirmed causes became deployable configuration:
`Agent(block_repeat_calls=True)` refuses an identical consecutive tool call
(nudge observation, `span["repeat_blocked"]`), and
`Agent(force_final_answer=True)` spends one toolless turn to force an answer
when the budget runs out (`terminated="forced_final"`);
`run_batch(agent_kwargs=...)` passes them through. `run_m5.py` pairs three
fix arms against the recorded baseline with anytime-valid e-values and e-BH
across the family, plus no-free-lunch accounting.

Recorded verdict on Qwen3-VL-2B counting: every arm moves the right way
(16→18/20/21 of 85, monotone dose-response, mechanisms demonstrably engaged
— 37 forced finals, 93+ blocked repeats, 7/22 empty-answer cases recovered)
and **e-BH rejects none of it**. The loop was the symptom; the capability is
the cause. The run ends inconclusive and recommends escalation (bigger
checkpoint, or a scaffold that counts detector boxes itself) instead of
shipping a prompt tweak as a win.

### Added — vtcbench_diagnosis example: the first end-to-end agent diagnosis

`examples/agent_demos/vtcbench_diagnosis/` runs one VTC-Bench task (default:
counting, 85 four-way MC cases) through the whole agent arc: vLLM-served
Qwen3-VL with zoom+detect under a ~1MP per-view budget → MC grading (a run
that never answers IS a failure) → the full M1 probe set → `records.json` →
`evalrx explore` with a 0.6/0.4 held-out confirm.

First recorded run (Qwen3-VL-2B): 81% fail. In-sample, loop-share signals
looked strong (AUC ≈ 0.75–0.80) — held-out kept only 3 of 7 frozen recipes:
*budget_exhausted*/*empty_answer* (every run that used all turns without
answering failed; held-out fail 1.00 vs 0.76) and *single_tool_only* (sign
FLIPPED on held-out — more tool use predicts failure, consistent with the
negative tool-Shapley mass). The tool-description-gap hypothesis came back
`not_testable` with the judge explicitly demanding the intervention test —
the M5 handoff the agent-aware M3 hint exists to produce. Practical notes
captured in the scripts: counting originals exceed a 16k context in vision
tokens (serve with `--mm-processor-kwargs '{"max_pixels": ...}'`), and the
explore CLI's default `--timeout-sec 120` truncates real analyses.

### Added — Agent-aware M2/M3: hypotheses that point at fixable causes

The statistical machinery needed no structural change for agent trajectories
(leak isolation, categorical recipes and the held-out flow all apply as-is —
`failure_mode == "FM-LOOP"` compiles to a testable 0/1 recipe today). What
changed is the guidance layer:

- M3 is now agent-aware: when the exploratory report references
  trajectory-column families, `HypothesisAgent` appends a hint steering
  hypotheses toward INTERVENABLE causes — prompt wording, tool descriptions,
  tool subsets, loop policy (the black-box fix surface) — and spelling out
  the column-family semantics (`shap_outcome_*` = causal attribution,
  `success_rate` = stability not capability, `failure_mode` = judge label to
  verify, `total_*` = cost). TEST lines may propose interventions, not just
  observational splits.
- `trajectory_records.AGENT_QUESTION_TEMPLATE` — the standard M2 question
  for agent records, pinned by test to the flattener's actual columns.
- The interventional probes' caveats now travel with their findings:
  held-out verification must RE-RUN `reliability_probe`/`tool_shap` on the
  held-out cases (never reuse exploration-set values), and
  `tool_shap.baseline_pass` is sanity, not signal.

### Added — Agent scale path, perception tools, and four new M1 information sources

Second wave of agent-under-test diagnosis: run agents at batch scale on any
backend, and make M1 emit *interventional* evidence (cost, reliability, tool
attribution, failure taxonomy) — not just single-run observations.

- Built-in OpenAI-compatible client (`evalrx.models.backends.openai_compat`):
  `openai_runtime(base_url=...)` wires `compose(key, "api")` to any
  OpenAI-compatible endpoint — a local `vllm serve` (verified end-to-end with
  Qwen3-VL + the hermes tool parser), OpenAI, or a gateway. Content-block
  images become data URLs; sampling defaults to `temperature=0`.
- `run_batch(handle, cases, tools_factory=..., concurrency=N)` — the M1-scale
  batch driver: per-case tool binding, threaded for API handles (local
  backends forced sequential), one failed case yields an error-stub
  trajectory instead of killing the batch.
- `GeminiModel.chat()` — native Gemini function calling (deliberately not the
  OpenAI-compat bridge), inline-PNG images, `tool_call_style="native"` routes
  it to the OpenAI codec. `generate()` now actually sends the image.
- New perception tools (`evalrx.models.tools`): `image_ocr` (easyocr
  default engine, region support) and `image_detect` (open-vocabulary
  Grounding DINO; detections come back as fractional boxes PLUS an annotated
  image the model can look at). Engines are injectable and lazily built.
- Cost accounting end-to-end: every agent turn records `latency_ms` /
  `prompt_tokens` / `completion_tokens` on the step span (all three chat
  backends fill usage), totals land in trajectory metrics and as
  `total_*`/`mean_turn_latency_ms` record columns.
- Trajectory→records bridge (`evalrx.analysis.trajectory_records`):
  flattens trajectories into `records.json` rows (loop volume, per-tool call
  counts, error rates, repeated-call structure, images returned, termination,
  cost) that `evalrx explore` and `build_stats_input_from_records`
  consume unchanged; `Step.from_dict` / `Trajectory.from_dict` reload
  persisted runs.
- Three new M1 analyzers (AGENT priority list is now seven deep):
  - `reliability_probe` — re-runs each case k times (injected runner) and
    reports pass@k vs pass^k, flakiness, answer agreement, and trajectory
    variance: capability failures and stability failures are different
    diagnoses.
  - `tool_shap` — AgentSHAP-style Shapley attribution over TOOL SUBSETS, with
    the value function changed for diagnosis: primary = outcome flip under an
    injected grader, secondary = answer similarity to the all-tools baseline.
    Exact for kits of ≤4 tools (all 2^N subsets), Monte-Carlo beyond. Also
    reports `no_tools_pass`/`tools_needed` — whether tools matter at all.
  - `trajectory_rubric` — LLM-judged failure-mode code (compact MAST-inspired
    taxonomy for single-agent tool loops) plus five 0-2 rubric dimensions;
    stamps `Step.failure_mode` at the judged first-error step.
- Analyzer selection is now data-aware: trajectory-carrying cases flip
  `StrategyProbe.detect_kind` to AGENT even for VLM handles (fixes the
  image-wins rule hiding agent analyzers), threaded through ProbeAgent; the
  four existing agent analyzers now declare text+image modalities.

### Added — Multimodal agent tool loop: image cases, image-returning tools, serialized trajectories

First step of agent-under-test diagnosis (a VLM that calls visual tools to
solve a task, with the run captured for trajectory analysis). The existing
backend-agnostic `Agent` loop + `Trajectory` schema stay the skeleton — no
external agent framework is introduced; frameworks remain potential *subjects*
adapted in via `Trajectory.from_records`.

- `Agent.run` is now multimodal end-to-end: `case.inputs.image` enters the
  first user message as a transformers-style content block, and a tool result
  carrying images gets them re-injected as a follow-up user message — the
  model *sees* what its tool produced (the o3 / Qwen3-VL-demo zoom mechanism).
- New `ToolResult(text, images, meta)` (`evalrx.core.tool`): model-visible
  text, re-injectable images, host-only meta recorded on the trajectory step.
  Plain return values keep working; errors keep the standard
  `"[tool error in ...]"` envelope (what `ignored_obs` matches).
- `HFLocalModel.chat()` gains a VLM path: content-block images are collected
  across the whole conversation and routed through the processor, so
  placeholder tokens line up with pixels; tool schemas still render via the
  chat template (Hermes text parsing, no server needed).
- Trajectories serialize: `Step.to_dict()` / `Trajectory.to_dict()`, and
  `FailureCase.to_dict()` now includes the trajectory. Images degrade to
  `"<image WxH>"` descriptors — trajectories reference media, never embed it.
- New `evalrx.models.tools` package (subject-side tools; deterministic,
  model-agnostic schemas): `zoom_in_tool` crops a bbox, upscales it (LANCZOS,
  short side → 672, ≤4x) and returns it as a new image. Accepts fractional
  coordinates but auto-detects pixel and Qwen-style 0-1000-grid boxes,
  recording `coord_mode` on the step — the convention the model actually used
  is evidence, not noise.
- New example `examples/agent_demos/visual_zoom_agent/`: the minimal
  trajectory on a real checkpoint (Qwen3-VL-2B, one COCO image). The first
  recorded run already exhibits a diagnosable failure — three consecutive
  identical zoom calls, flagged by the existing `LoopDetector` when fed the
  reloaded trajectory.

---

Everything below this line was published to PyPI as **`evalvitals`**, before
the rename — see [0.1.0]'s "Changed" entry above. Version numbers restart at
0.1.0 for `evalrx`, so `[0.1.0] — 2026-07-10` below is the *original*
`evalvitals` 0.1.0, not this package's.

## [0.1.1] — 2026-07-26 (evalvitals)

### Added — Plain-language headlines for M2 takeaways and M3 hypotheses

Reader feedback: `exploratory_report.json`'s takeaway/hypothesis headlines
were often unreadable to a non-technical reader (e.g. `"focus_share
separates FAIL from PASS at AUC 0.82 (CI 0.77-0.87)"`). Asking the prompt for
"plain language" alone didn't hold — the model kept reusing the jargon-heavy
technical line, so this adds a host-side check, not just a prompt tweak.

- Each M2 takeaway now carries a required `plain_title` (jargon-free,
  one-sentence headline) alongside the existing `title` (precise technical
  line with the numbers). Each M3 hypothesis carries `plain_statement`
  alongside `statement`.
- New `evalrx.analysis.plain_language.jargon_violation()` — flags a
  missing headline, a verbatim copy of the technical line, banned stats
  jargon/acronyms (AUC, collinear, logistic, p-value, quantile, ...), or
  stray symbols (→, ρ, σ). Shared by both M2 and M3.
- M2: a jargon violation reuses the existing repair-attempt budget
  (`_run_explore_loop`) to ask for a rewrite; if attempts run out the report
  still returns `ok=True` with the violation logged to `critique` rather than
  failing the whole analysis.
- M3 (`HypothesisAgent`): one bounded extra generation call rewrites only the
  flagged `PLAIN:` lines; falls back to the original hypotheses if the
  repair itself doesn't fix it.
- Dashboard (`dashboard_app.py`): takeaway and hypothesis cards render the
  plain headline first, with the technical line demoted to a small secondary
  "Technical detail" line. Falls back to `title`/`statement` for older
  reports that predate this field, so nothing existing breaks.

### Changed — one chart authority across the bundled skills; nature-figure palette synced

Deduplicates the overlapping style/chart rules the skill audit surfaced:

- **eval-chart-style (v0.4.0) is now the single chart authority.** Its policy
  table absorbed the figure kinds outcome-driver-analysis used to prescribe —
  one-variable EDA distributions (histogram / count bars), regression
  odds-ratio forests, predicted-probability curves with CI bands, and ROC +
  calibration plots — and gained an explicit arbitration clause: when another
  installed skill calls for a specific chart, keep its statistical intent but
  render it under this table's chart types and §1's palette. The staged
  `skills_hint` carries the same precedence sentence.
- **outcome-driver-analysis (v0.2.0) keeps the WHAT, defers the HOW**: a new
  "chart types and palettes" clause makes its step-2/3/7 figure prescriptions
  the standalone fallback (used only when no chart-style skill is installed);
  step 3's boxplot wording now names the deference explicitly. Also removes
  the dangling reference to a never-bundled `clear-technical-writing` skill.
- **nature-figure's canonical palette constants synced** (Apache-2.0
  modification, notice added to its README): `PALETTE`/`DEFAULT_COLORS` in
  `references/api.md` + `references/design-theory.md` re-valued to the
  dataviz-validated palette — key names and family semantics unchanged, ramps
  (`#7ec07e→#4ca74b→#008300`, `#ee9999→#e97675→#e34948`) and the 5 chromatic
  series slots re-validated with the dataviz checker; the 6th series slot is
  now muted ink `#898781`, documented as reference/background only. Because
  the host spec renderer (`viz/style.py::load_nature_style`) parses those
  constants at runtime, host-rendered spec PNGs now share the same palette as
  agent figures and host plotly charts — closing the last palette divergence
  (`NATURE_COLORS_FALLBACK` mirrored). Tutorial/example hexes elsewhere in
  the vendored skill are illustrative and were deliberately not rewritten.

### Added — `run_codebase`: run a user's codebase, then explore the results

New standalone entry point bridging the run infrastructure
(`evalrx.agent_runtime`) and the analysis entry (`evalrx.explore`):
given a path to an existing evaluation/inference codebase, a CLI coding agent
runs it inside an isolated copy (the original is never modified), harvests
the per-case results it writes (`records.json`, one row per case with a
`label` plus input/prediction/target fields — one repair turn if nothing
usable was produced), and hands them straight to `explore()` (M2 + M3).

- `evalrx.run_codebase(path, ...)` / `evalrx.analysis.run_codebase`
  (library) — returns a `CodebaseRunResult` (`records`, `ran_ok`, `explore`,
  `error`, ...).
- `evalrx run-codebase ./my_repo [--backend ...] [--out ...] [--dashboard]`
  (CLI) — persists the workspace, harvested records, and explore artifacts
  under `--out`; `--no-explore` skips the M2/M3 step.
- See [quickstart.md#run-a-codebase-then-explore](docs/quickstart.md).

### Added — ProbeLLM-style failure-mode synthesis: failure-aware clustering + hierarchical MCTS probe search (VLM)

Deeper integration of arXiv 2602.12966 ("ProbeLLM"), which already inspired
`eval_agent`'s failure-mode-clustering design (see the 2026-07-10/11 refactor
entry below) — two further pieces from that paper, both additive:

- **`evalrx.analysis.cluster_failures`**: two new opt-in, off-by-default
  parameters. `expected_col`/`error_fn` fold a failure-*mechanism* signal into
  each FAIL case's vectorized text (LLM-described when `judge` is given,
  deterministic `"expected=... got=..."` fallback otherwise), so grouping
  reflects *how* the model failed, not just what the prompt was about.
  `boundary_aware=True` keeps non-FAIL rows as a contrast pool and pairs each
  cluster's peripheral cases with their nearest verified non-failure
  (`FailureMode.boundary_pairs`), surfaced to the LLM namer to describe
  *where* the failure boundary sits. Existing callers see byte-identical
  output — neither parameter is used unless explicitly passed.
- **`evalrx.analysis.probe_search`** (new, standalone): `ProbeSearch`, a
  hierarchical two-level MCTS (Macro = broad topical coverage, Micro = local
  refinement around recurring failures) that adaptively synthesizes and
  evaluates *new* test cases rather than analyzing an already-collected
  dataset, via three injected callables (`generate_macro`/`generate_micro`/
  `verify`) — no model/judge/eval_agent dependency in this module.
- **`evalrx.eval_agent.ProbeSearchAgent`** + **`VLMProbeCandidateGenerator`**
  (new): wire `ProbeSearch` to a real target model and judge for VLM QA —
  `CaseDiscoveryAgent` as the verifier, paraphrase-based Macro/Micro question
  generation over a fixed (image, expected) seed pool as the generators (v1
  scope: paraphrase only, never new imagery or an unverified new gold
  answer). Also exposed as `AgenticDiagnoseLoop`'s new `search_probes` tool
  (host-capped via `max_calls`), reusing the loop's own decision judge/model.
  `result.failure_cases` is a plain `CaseBatch` that feeds directly into
  `cluster_failures` above.
- See [m2_analysis.md#probe-search--hierarchical-mcts-failure-discovery-vlm](docs/m2_analysis.md)
  and `docs/architecture.md`'s package-layout/stage-contracts sections.

### Added — held-out verification for ANY explore run (`analysis.holdout`, workbench modes)

- **`evalrx.analysis.holdout`** (new): the deco_hallu pipeline's phase 2,
  generalized to any dataset the explorer can load. `split_records` carves a
  deterministic, outcome-stratified explore/holdout partition BEFORE
  exploration (wraps the fused pipeline's split); `holdout_confirm`
  re-evaluates the explorer's frozen recipes verbatim on the held-out rows
  (`compile_recipe`, no re-fitting), rebuilds two-group sufficient stats,
  adjudicates with `adjudicate_signals(split_label="held_out")`, and has an
  optional duck-typed LLM judge grade each M3 hypothesis
  (supported/partial/refuted/not_testable + surgery routing; `judge=None` →
  `not_judged`). A generic `failure_indicator` maps arbitrary binary outcome
  columns (fail-like value → boolean/0-1 → minority class, with the rule
  recorded in the report); non-binary outcomes degrade honestly to skipped
  verdicts instead of inventing one.
- **`evalrx explore --holdout-frac F [--holdout-confirm] [--holdout-seed N]
  [--judge-model M]`**: with a fraction the explorer only ever sees the
  explore share (the held-out rows are persisted to `holdout_records.json`);
  with `--holdout-confirm` the question additionally demands FROZEN,
  threshold-explicit recipes and `confirm_report.json` lands next to the
  exploratory report — the dashboard's Held-out Verdicts tab fills in.
- **Workbench "New analysis" modes**: a mode selector — *Explore only
  (M2 + M3)* vs *Explore + held-out verification* — with an adjustable
  explore : verdict split (defaults 1 : 0 and 0.6 : 0.4). In explore-only
  mode a sub-1.0 share reserves the remainder untouched; in verification
  mode it becomes the verdict half. The job record carries
  mode/explore_share/holdout_frac.

### Changed — one fixed five-tab layout for every explore result; the web workbench as the unified surface

- **Fixed tab set** (`dashboard_app.render_explore_report`): every
  explore-shaped result — plain `evalrx explore` run, held-out pipeline
  run, uploaded-zip run — now shows the same five tabs (Problem Setting /
  Exploratory Analysis / Hypotheses / Held-out Verdicts / Fix). Stages a run
  never reached render as greyed **"not available"** placeholders saying what
  the panel would show and which phase produces it (`confirm_report.json` /
  `fix_report.json`), instead of the tabs appearing and disappearing between
  runs. `SKIP_FIX=1` pipeline runs get real verdicts in tab 4 and a greyed
  tab 5. Applies to `evalrx dashboard` and the workbench alike (shared
  renderer).
- **`evalrx web --attach DIR`** (repeatable): existing result directories
  (explore outputs or loop runs) are listed read-only (📁) in the workbench
  sidebar next to uploads, rendered with the same views `evalrx dashboard`
  would use — one page holds "upload a .zip to start a new analysis" AND the
  results other scripts already produced. `run_web.sh` auto-attaches the
  example's `outputs_attn_full` / `outputs_pipeline/1_explore` / `outputs`
  when present (`ATTACH_DIRS` env override).

### Changed — eval-chart-style skill synced to the redesigned theme palette

The dashboard redesign swapped `eval_viz_theme`'s palette for the
dataviz-validated light/dark palette, which left the bundled eval-chart-style
skill teaching the retired colors — agent-drawn PNGs and host-rendered charts
would have disagreed (most visibly PASS: neutral slate vs status green). The
skill's §1 now carries the light-mode values of the same palette: FAIL
`#d03b3b` / PASS `#0ca30c` (reserved status colors), inconclusive `#fab219`
(amber, with the label-not-color-alone rule), leaky `#898781`; the 8-slot
fixed categorical order (`#2a78d6` …) with green/red slots skipped next to
outcome hues; ordinal blue ramp `#86b6ef → #2a78d6 → #104281` (validated);
diverging `#2a78d6 ↔ #f0efec ↔ #e34948` — the categorical red, deliberately
not the FAIL red, so signed heatmaps can't impersonate the outcome.

### Added — `evalrx web`: upload-a-.zip explore workbench

- **`evalrx web [WORKSPACE]`** (new CLI subcommand) serves a Streamlit page
  (`analysis/upload_app.py`) where users upload a **.zip** of results
  (JSON/JSONL/CSV; zipped folders are unwrapped, `__MACOSX`/`.DS_Store` junk
  skipped, zip-slip rejected). Each upload becomes one run directory under the
  workspace and is analysed by the ordinary `evalrx explore` CLI (M2
  exploratory analysis + M3 hypothesis proposal) as a **detached subprocess**
  — closing the browser never kills an analysis; `job.sh`/`job.json`/
  `exit_code` make every run observable and re-runnable by hand. Finished runs
  render in place via the shared `dashboard_app.render_explore_report(...)`
  (extracted from the dashboard's explore view — same tabs, including the
  held-out verdict/fix tabs if pipeline artifacts appear later), and past runs
  stay selectable in the sidebar with live status (🟡 running / 🟢 done /
  🔴 failed / ⚪ stale).
- `examples/m2_statistics/deco_hallu_explore/run_web.sh`: turnkey launcher
  (`PORT`/`WORKSPACE`/`CODER_PROVIDER`/`CODER_MODEL`/`TIMEOUT_SEC` env
  overrides; these only pre-fill the form — each upload can override in the UI).
- Tests: `tests/test_analysis/test_upload_app.py` — zip staging (single-root
  unwrap, junk skip, zip-slip/empty rejection), job argv/record/wrapper, status
  transitions (running/done/failed/stale), and AppTest passes over the upload
  form, a finished run (explore tabs render), and a failed run (log + hint).

### Added — agentic orchestrator, standalone data-analysis package, failure-mode clustering

Four related changes make `eval_agent` more flexible/agentic and make
`evalrx.analysis` genuinely usable without the diagnosis loop:

- **`evalrx.agent_runtime`** (new): the CLI-agent runtime (sandboxing, code
  generation, providers, judge models, skills) shared by `evalrx.analysis`
  and `evalrx.eval_agent` — imports neither, so either can be used standalone.
- **`evalrx.analysis` is now dependency-free of `eval_agent`** — a new AST-based
  guard test locks this in permanently. The M2 stats stack (`StatsAnalysisAgent`,
  `stats_tools`, `stats_tool_agent`, `stats_tool_generator`, `AnalysisModule`) and
  the explorer's prompt templates moved here from `eval_agent`.
- **`evalrx.explore(...)`** (new): a one-call library entry point — path or
  in-memory records in, `ExploreRunResult` out; `out=None` keeps everything in
  memory. `evalrx-explore` CLI behavior is unchanged (thin wrapper over the
  same function).
- **`AgenticDiagnoseLoop`** (new, `evalrx.eval_agent.agentic`): a judge-decided
  alternative to `VLDiagnoseLoop`'s fixed M1→M2→M3→M4 cycle — a CLI judge picks the
  next tool each turn (`run_probe` / `run_stats` / `explore_data` /
  `cluster_failures` / `propose_hypotheses` / `test_hypothesis` / `run_surgery` /
  `run_fix` / `stop`) instead of a hardcoded sequence. The host enforces tool call
  caps, preconditions, and the stop-gate (declaring success requires an
  actually-tested, supported, protocol-consistent hypothesis — a premature
  `stop(resolved=true)` is rejected and fed back to the judge). `VLDiagnoseLoop`
  itself is unchanged. `run_log.jsonl` gains two event types, `agent_decision` and
  `agent_tool` (schema version 2 → 3).
- **`evalrx.analysis.cluster_failures(...)`** (new): groups FAIL cases into
  interpretable failure-mode clusters — pattern discovery over the raw failing
  cases, complementing M2's per-signal EDA. No required dependency (a pure-numpy
  fallback always works); install the new `[cluster]` extra
  (`scikit-learn`, `hdbscan`) for TF-IDF + density-based clustering. Wired into
  `DiagnosisAgent(failure_modes=...)` and as the agentic loop's `cluster_failures`
  tool.

### Breaking — module reorganization (no compatibility shims)

Several modules moved as part of the above. Two facades are kept
(`evalrx.eval_agent.cli_agent` and `.cli_skills`) — everything else below
requires updating the import path:

| Old path | New path |
|---|---|
| `evalrx.eval_agent.cli_types` | `evalrx.agent_runtime.cli_types` |
| `evalrx.eval_agent.cli_runtime` | `evalrx.agent_runtime.cli_runtime` |
| `evalrx.eval_agent.cli_transcript` | `evalrx.agent_runtime.cli_transcript` |
| `evalrx.eval_agent.sandbox` | `evalrx.agent_runtime.sandbox` |
| `evalrx.eval_agent.factory` | `evalrx.agent_runtime.factory` |
| `evalrx.eval_agent.experiment_harness` | `evalrx.agent_runtime.experiment_harness` |
| `evalrx.eval_agent.codegen` | `evalrx.agent_runtime.codegen` |
| `evalrx.eval_agent.providers.*` | `evalrx.agent_runtime.providers.*` |
| `evalrx.eval_agent.models.*` (`AgyModel`, `ClaudeModel`) | `evalrx.agent_runtime.judges.*` |
| `evalrx.eval_agent.skills.*` | `evalrx.agent_runtime.skills.*` |
| `evalrx.eval_agent.stages.stats_tools` | `evalrx.analysis.stats_tools` |
| `evalrx.eval_agent.stages.stats_agent` | `evalrx.analysis.stats_agent` |
| `evalrx.eval_agent.stages.stats_tool_agent` | `evalrx.analysis.stats_tool_agent` |
| `evalrx.eval_agent.stages.stats_tool_generator` | `evalrx.analysis.stats_tool_generator` |
| `evalrx.eval_agent.stages.analysis` (`AnalysisModule`) | `evalrx.analysis.analysis_module` |
| `evalrx.eval_agent.prompts.explorer` | `evalrx.analysis.prompts.explorer` |
| `evalrx.eval_agent.prompts.{stats_agent,stats_tool_generator}` | `evalrx.analysis.prompts.*` |
| `evalrx.eval_agent.loop.AutoDiagnoseLoop` | `evalrx.eval_agent.legacy.AutoDiagnoseLoop` |
| `evalrx.eval_agent.loop.SelfEvolveLoop` | `evalrx.eval_agent.legacy.SelfEvolveLoop` |
| `evalrx.eval_agent.loop.AutoDiagnoseReport` | `evalrx.eval_agent.loop_reports.AutoDiagnoseReport` |
| `evalrx.eval_agent.loop.VLDiagnoseReport` | `evalrx.eval_agent.loop_reports.VLDiagnoseReport` |

`from evalrx.eval_agent import X` (the top-level package) is **unaffected**
for every name above except the ones that moved to `legacy`/`loop_reports` —
those still import the same way from `evalrx.eval_agent`, just backed by a
different implementation module. `evalrx.eval_agent.loop` now contains only
`VLDiagnoseLoop`. `pip install evalrx[cluster]` (or `[all]`) adds
`cluster_failures`'s optional sklearn/hdbscan backends.

### Added — held-out hypothesis pipeline for the explore path (deco_hallu_explore)

- **Four-phase pipeline** (`run_attn_pipeline.sh`): the standalone explore
  path gains the loop's propose→confirm→fix arc with a REAL held-out design.
  Phase 0 `prepare_splits.py` carves the enriched data by its `split` column
  (explore=365 / validate=241). Phase 1 runs `evalrx explore` on the
  explore half only (the prompt tells the agent a validate half exists and
  demands frozen, threshold-explicit recipes). Phase 2 `test_hypotheses.py`
  re-evaluates each candidate recipe VERBATIM on the validate half
  (`operationalize.compile_recipe`, thresholds frozen — no re-fitting),
  rebuilds two-group sufficient statistics, adjudicates with
  `adjudicate_signals(split_label="held_out")` (a REJECT here is a real
  held-out verdict, unlike phase 1's in-sample screen), then an LLM judge
  grades each M3 hypothesis against the held-out table
  (supported/partial/refuted/not_testable + needs_surgery routing). Phase 3
  `run_surgery.py` hands the surviving hypotheses to the diagnosis loop's
  repair machinery — M4 confirm → M5 surgery → tiered fix (L1→L3b) on the
  loop example's frozen M1 batch (GPU) — and distills `fix_report.json`.
  Artifacts (`confirm_report.json`, `fix_report.json`) land next to the
  exploratory report for the dashboard to render.

### Changed — pipeline report: separate tabs + reader-facing repair digest

- **Tab layout** (`analysis/dashboard_app.py`): tab 3 is back to the PURE
  proposal view (identical to a plain explore run — verdict badges no longer
  bleed into it); held-out verdicts get their own tab 4 and the fix its own
  tab 5 (numbering adapts when only one artifact exists).
- **Fix tab digest**: `run_surgery.py` now distills the run logger's fix event
  (the single source of truth) into structured `fix_report.json` fields —
  per-candidate tier/name/repaired/broke/coverage/e-value/verdict, `best`,
  `ebh_survivors`, `refine_signal` — plus a `--distill-only` mode to rebuild
  the report from existing logs without a GPU rerun. The dashboard renders a
  deterministic narrative ("What was repaired, and how well": 🏆 winner with
  paired-flip counts and e-value, family-corrected survivors, a prompt-level
  vs internals-write contrast when the data shows one, and the refine signal)
  above the full candidates table.

### Added — dashboard renders the held-out pipeline (verdicts + fix)

- **Explore dashboard, tab 4** (`analysis/dashboard_app.py`): when
  `confirm_report.json` / `fix_report.json` sit next to the exploratory
  report, the view grows a "Held-out Verdicts & Fix" tab — validate-split
  metrics, the frozen-recipe re-adjudication table (REJECT here is a real
  held-out verdict, distinguished from tab 2's in-sample screen), per-
  hypothesis judge verdict cards, the M4/M5 confirmation table, and the fix
  recommendation. Tab 3's hypothesis cards gain held-out verdict badges
  (supported/partial/refuted/not_testable + surgery routing); without the
  artifacts the view is unchanged (three tabs, proposal-only wording).

### Changed — eval-chart-style: color range for non-outcome dimensions

- The semantic FAIL/PASS lockdown made every figure red/slate/grey, even panels
  sliced by something other than the outcome. §1 now adds: a 6-color
  categorical series order (from nature-figure's palette; the red slot is
  skipped whenever FAIL-red shares the panel) for checkpoint/object/multi-signal
  series, a single-hue luminance ramp rule for ORDERED dimensions (model size
  2B→4B→8B, ordered bins), and heatmap guidance (diverging Red-Blue around 0
  vs single-hue sequential). Semantic role colors still win wherever the
  outcome appears. `_skills_hint` describes the extended palette accordingly.

### Added — outcome-driver-analysis: statistical-method skill for the analysis stage

- **`outcome-driver-analysis` bundled skill** (`agent_assets/skills/…`, MIT,
  vendored from the user's project skill): a disciplined 8-step protocol for
  explaining a binary outcome — explanatory-variable EDA, per-variable tests
  WITH effect sizes, conditioning/Simpson's checks, marginal screening, a
  *justified* regression model (GLM vs mixed-effects reasoned from the actual
  clustering), fit diagnostics (VIF, ROC/AUC, calibration), and result
  visualization. Fills the standalone explore pipeline's biggest gap: marginal
  descriptive contrasts were never adjusted for confounders or the
  checkpoint-clustering structure.
- **Staged skill hint** (`explorer._skills_hint`): the prompt now stages skills
  by function — ANALYSIS METHOD (invoke outcome-driver-analysis BEFORE writing
  any analysis code) then FIGURE STYLING (eval-chart-style/nature-figure BEFORE
  plotting). Guards keep the sandbox contract intact: adopt the methodology,
  not the skill's file layout; infer intake from the data profile (never ask);
  takeaways stay DESCRIPTIVE (effect sizes + CIs; test statistics live in
  tables/artifacts, never phrased as significance verdicts — validity remains
  the confirm phase's job); with very few clusters prefer a fixed effect.
- Env: `statsmodels` + `scikit-learn` added to the venv (the protocol's Python
  path needs real inference; the repo previously had numpy-only stats).

### Added — figure skills on by default across claude/agy/codex

- **`eval-chart-style` bundled skill** (`agent_assets/skills/eval-chart-style/`):
  the chart-type + house-style policy that `analysis/eval_viz_theme.py` codifies
  for host plotly charts, rewritten as an Agent Skill for the sandbox coder —
  distribution-first chart selection (violin/ECDF/heatmap/forest/paired-slope,
  never a mean as a bar), the FAIL `#C0413B` / PASS `#5B7A99` semantic palette,
  human bin/number formatting, and leakage-demotion rules. Complements
  `nature-figure` (journal polish); the two divide chart-TYPE vs styling.
- **Bundled skills are now default-on for every figure-drawing agent flow**
  (`analysis/explorer.py`): `ExploratoryAnalysisAgent` injects the bundled
  skills into any bare `CliAgentConfig` on a skill-capable backend, so the
  fused pipeline and example scripts get them without wiring
  (`use_bundled_skills=False` opts out; explicit `skills=` is respected).
  The prompt hint is no longer opt-in ("you MAY invoke") — agents are told to
  apply the skills BY DEFAULT before plotting, with the non-interactive guard
  kept (never pause to ask; styling only).
- **codex now receives skills too** (`agent_assets/skills.py`,
  `eval_agent/cli_agent.py`): `SKILL_BACKENDS` gains `codex`; since codex has
  no `Skill` tool, `CodexAgent._install_skills` surfaces the vendored
  `.claude/skills/<name>/SKILL.md` files through the workdir's `AGENTS.md`
  (appending, never clobbering), and `_skills_hint` tells codex to read the
  vendored guides rather than invoke a tool. deco_hallu example
  `codegen_timeout_sec` 240→480 to absorb the skill read+apply overhead.

### Changed — M2 explorer adapts to arbitrary outcome shapes, not just FAIL/PASS

- **Outcome-adaptive explorer prompt** (`analysis/explorer.py`, `analysis/profile.py`):
  `ExploratoryAnalysisAgent`'s generated-code prompt used to hardcode a binary FAIL/PASS
  framing (class balance, fail-rate curves, "call the two groups FAIL and PASS")
  regardless of what the data actually was. The host now profiles the outcome
  column via the new `describe_outcome()` (`profile.py`) and classifies it as
  `binary` / `categorical` / `continuous` / `none`, and `_framing_block()` swaps
  in the matching framing + standard chart battery — including a genuinely
  unsupervised-EDA battery (missingness, distributions, correlation structure)
  when there is no recognizable outcome column at all. `profile_records()` gained
  an `outcome_col` override so callers with an arbitrarily-named target (e.g.
  `revenue`, `yield_pct`) can point at it explicitly instead of relying on the
  English name-heuristic list. `explore_records` / `explore_path` / `run_explore`
  / the `evalrx explore` and `evalrx-explore` CLIs all take a new
  `outcome_col` / `--outcome-col`. M1's diagnosis loop is unaffected — its
  records already carry a `label` column the heuristic finds automatically, so
  it still gets the same binary FAIL/PASS framing as before.

### Added/Changed — signal hygiene, descriptive analysis, tensor-level attention

- **Label-leak isolation (the deferred "leak-1" check)** (`eval_agent/stages/stats_tools.py`):
  `label_leak_score` / `isolate_label_leaks` detect per-case signals that
  *reconstruct* the FAIL label (a probe flag equal to the outcome) and route them
  to a new `StatsInput.sanity` lane, out of the tested family / e-BH multiplicity
  / candidate charts / hypothesis seeding. A leak is a **binary flag that ~equals
  the label** (best-split accuracy ≥ 0.985, ≥ 10 cases) — a *continuous* feature
  that perfectly separates the classes is legitimate discovery and is NOT flagged.
  Wired into `build_stats_input` / `build_stats_input_from_records` and the fused
  pipeline's confirm step, so `generated:probe1.false_detection` /
  `explored.probe1_positive` no longer get tested, charted, or "confirmed."

- **Descriptive analysis phase (validity deferred to confirm)**
  (`StatsAnalysisAgent.analyze(confirmatory=…)`, `VLDiagnoseLoop`,
  `reporting/compiler.py`): `run_analysis()` now runs M2 with the e-BH validity
  verdict DEFERRED (`StatsAnalysisReport.descriptive_only=True`, `corrected_rejections`
  marked deferred) — effect sizes + charts only. `run_confirm()` recomputes e-BH
  (`_finalize_confirmatory_stats`) and logs the confirmatory M2, so the dashboard
  shows supported/not-supported claims ONLY after confirmation. The compiler
  demotes every claim to descriptive in analysis-phase mode with a banner; the
  all-in-one `run()` path stays confirmatory and unchanged.

- **Richer per-case attention features + per-case map stack**
  (`analyzers/attention/relative_attn.py`): each case now also emits
  `attention_entropy`, `top1_share`, `center_offset`, `edge_mass` (diffuse-vs-spike,
  peripheral, positional-sink proxies) alongside max/mean/focus, and the full
  per-case spatial maps are stored (float16) in `artifacts["per_case_maps"]`.
  `prompt_contrast` now surfaces a non-tautological per-case `prompt_sensitivity`
  signal (answer instability across prompts, correctness-independent) — the
  tautological `fixed_by_*`/`broken_by_*` flags stay in artifacts.

- **`attention_decoding` — tensor-level omnibus** (`stats_tools.py`): a new M2 tool
  over the FULL per-case attention map (not a scalar reduction), answering "do
  FAIL and PASS attend differently anywhere?" — feature-agnostic, pure numpy,
  robust at features≫samples. Primary test is a two-sample **energy-distance
  permutation** test (more powerful than linear decoding at low n; sensitive to
  nonlinear / distributional differences), with a cross-validated linear-decoder
  AUC reported alongside as an interpretable companion. Reads
  `StatsInput.per_case_vectors`, runs as a mandatory global/omnibus tool in M4,
  and is added to `default_plan` when map vectors exist. On the deco_hallu slice
  the energy test flips the verdict from inconclusive (CV-AUC 0.42) to a real
  finding (energy-distance D=1.88, permutation p=0.018) — the maps differ
  distributionally even though no linear boundary separates them.

### Fixed — analysis-phase dashboard rendering

- **Stale confirm runs no longer leak verdicts into the analysis view**
  (`analysis/dashboard.py`): `load_loop_story` merged events from *every*
  `logs*/` dir, so a directory holding both a descriptive `logs_analysis/` run
  and a stale all-in-one `logs_m2_5/` run would resurrect the old surgeries +
  supported/not-supported verdicts on top of the descriptive run. It now keeps
  the shared M1 probe log plus only the **single most-recent M2+ arc** (by
  mtime), so an analysis-phase directory renders descriptively without a symlink
  workaround.
- **Analysis phase shows candidate signals descriptively** (`analysis/dashboard_app.py`):
  when the loaded run is analysis-only (`_story_is_descriptive`: every M2
  `descriptive_only`, no surgery), evidence cards render a **"Descriptive"**
  badge ranked by |effect| instead of mapping the explorer's `reject` flags to
  "Supported"/"Not supported", the method card drops the e-BH/"tested signals
  survived" framing, and an "Analysis phase" banner orients the reader. The
  explorer's per-recipe `reject` is not confirmation.
- **Scatter tables with real-named columns no longer crash** (`_resolve_scatter_axes`):
  the explorer now writes scatter CSVs with real signal columns
  (`attention_entropy, center_offset, outcome`); the dashboard assumed legacy
  `x`/`y` columns and fell back to literal `"x"`/`"y"`, raising plotly's
  `Value of 'x' is not the name of a column`. Axis resolution now prefers the
  report's recovered names, then the CSV's own non-outcome columns, and skips
  cleanly when two value axes can't be formed.

### Added — decoupled analysis vs. confirm+fix

- **`VLDiagnoseLoop.run_analysis()` + `VLDiagnoseLoop.run_confirm()`**
  (`eval_agent/loop.py`): split the diagnosis pipeline so the analysis
  dashboard can be produced *before* hypotheses are confirmed. `run_analysis()`
  runs **M1 → M2 → M3** (the same rigorous e-BH stats + charts, then *propose*
  hypotheses) and stops — the returned `VLDiagnoseReport` carries
  `all_hypotheses` (proposed, unconfirmed) and `final_stats_report`, with
  `all_test_results` / `verified_hypotheses` empty (`stopped_by="analysis_complete"`).
  `run_confirm(data, hypotheses, stats_report=...)` runs **M4** on those
  hypotheses — typically reloaded via `hypothesis_from_dict` and confirmed
  against the *exact* M2 report the dashboard showed (regenerated from the
  frozen M1 when omitted) — then feeds `run_m5` / `run_fix` as before. The
  shared per-stage helpers (`_do_m1/_do_m2/_do_m3/_do_m4`) are factored out of
  `run()`, whose behavior is unchanged. The dashboard renders proposed
  hypotheses without M4/M5/Fix verdicts and gains them once the confirm phase's
  log dir is present.

- **deco_hallu decoupled scripts** (`examples/diagnosis_loops/deco_hallu/`):
  `run_analysis.py` (GPU-free: replay M1 → M2 stats/charts → M3 propose →
  dashboard, persisting `outputs/analysis/{proposed_hypotheses.json,
  analysis_state.pkl}`) and `run_confirm_fix.py` (reload those artifacts → M4
  confirm → M5 + tiered Fix), with matching `run_analysis.sh` /
  `run_confirm_fix.sh` wrappers. The shared frozen-M1 `ReplayProbeAgent` and a
  GPU-free `FrozenModel` stub now live in `run.py`. The one-shot `run_m2-5.py`
  path is unchanged.

### Added — run output ownership & tiered repair

- **`RunContext`** (`eval_agent/run_context.py`): single owner of a diagnosis
  run's output directory, replacing the old per-example pattern of
  hand-written report files, `RunLogger` buried under a `logs/` subdir, and
  M5 sandboxes living in ephemeral temp dirs deleted on success. Owns
  `report/`, `figures/`, `artifacts/`, `prompts/`, `experiments/`, `tools/`,
  `workspace/`, `fixes/`, plus `manifest.json` and an auto-generated
  `README.txt`. `write_diagnose_report(report, cases, discovery=...)` writes
  the standard `report/` deliverables, duck-typed across `VLDiagnoseReport`
  and `AutoDiagnoseReport`. Migrated all 11 VL examples + `nl_runner`'s
  codegen template off the old `RunLogger(run_dir / "logs")` convention.

- **Per-trial output folders** (`RunContext.new_trial()` / `Trial`): each
  `FixAgent` candidate and M5 `ExperimentWriter` experiment now gets its own
  lazily-created, numbered folder (`fixes/03_widen_crop/`,
  `experiments/01_...`) holding its generated code, the sandbox it ran in,
  judge prompt/output, `record.md`, and `result.json` — instead of scattering
  those across `tools/` / `workspace/` / `fixes/` and re-correlating by
  filename slug. A discarded/deduped candidate leaves no folder on disk; a
  gap in the numbering means "proposed, then discarded."

- **`relative_attention` overlay heatmaps** (`analyzers/attention/relative_attn.py`):
  `RelativeAttentionResult.overlay()` / `.save_overlay()` / `.image_overlays()`
  alpha-blend each spatial map onto its representative case image (CAM-style),
  resolving lazy `Inputs.image` paths/URLs the same way the model's forward
  pass did. `RunLogger` picks this up via a duck-typed `image_overlays()` hook
  on `Result`, so overlay PNGs land in `figures/` alongside the bare heatmaps
  and reach the multimodal judge the same way.

- **`run_log.jsonl` schema_version**: every event now carries a `schema_version`
  (int), bumped only when an existing event's fields are renamed, removed, or
  change meaning, so downstream parsers can detect breaking changes without
  guessing from `evalrx_version`.

- **`run_log.jsonl` schema_version 2 — M2 stats payloads externalized above
  4 KB**: `analysis`'s `stats_tool_results`/`stats_results`/`stats_plan`/
  `corrected_rejections` no longer inline unboundedly — once the serialized
  value exceeds 4 KB it's written to `artifacts/c{cycle}_m2_{field}.json` and
  the JSONL line carries `{"path", "n_items", "bytes"}` instead, matching how
  every other heavy field (judge I/O, M1 artifacts) is already handled.
  Typical small runs are unaffected. Also: `RunLogger._codegen_seq` (the
  per-cycle codegen filename counter) now increments under a lock.

- **`run_start` provenance — dataset + code version**: the first
  `run_log.jsonl` event now also records `data_fingerprint` (an
  order-independent SHA-1 over the case batch, so two runs can be confirmed to
  use the same data) and `label_distribution` (the base PASS/FAIL/UNKNOWN
  counts the whole diagnosis is conditioned on) alongside the existing
  `n_cases`. `git_commit` now falls back to the `EVALRX_GIT_COMMIT` env var
  when the `git` CLI is unavailable, so the code-version provenance is no longer
  silently dropped inside the example Docker images (which ship no git). The
  `eval_agent` compose file forwards `EVALRX_GIT_COMMIT`.

- **Published JSON Schema for `run_log.jsonl`** (`eval_agent/log_schema.py` +
  shipped `run_log.schema.json`): the log event format is now a machine-readable
  contract (Draft 2020-12), not just a docstring + an opaque `schema_version`
  int. `build_schema()` generates it from the stdlib (no new core dependency —
  the light install is preserved); `load_schema()`, `validate_event()` and
  `iter_log_errors()` validate logs (needing the optional `jsonschema` dev dep).
  The schema is permissive (pins the envelope, per-event required fields and
  core types; allows additive fields). A contract test drives every `RunLogger`
  event type and asserts the real output conforms, so the schema can't silently
  drift from the producer. Opt-in `EVALRX_VALIDATE_LOG=1` makes `RunLogger`
  self-check each event and warn (never raise) on a violation.

- **`self_consistency` records its sampling config**: the analyzer's findings
  now include `gen_kwargs` (the kwargs passed to `model.generate`, temperature
  above all). The consistency score is uninterpretable without it — a low score
  at temperature 0 is a real defect, the same score at 1.0 is expected — so the
  parameter the measurement is conditioned on now travels with it into the
  `probe` event. Empty dict means the model's own `generate()` defaults.

- **`FixAgent.max_repair_rounds`** (`eval_agent/stages/fix_agent.py`, from the
  `jiaqiliu` merge): feedback-driven multi-round propose→validate within one
  fix tier. After a round where nothing validates, per-candidate results are
  summarised and fed back to the judge/coder, which proposes a *different*
  strategy — never re-running an identical candidate (candidate dedup via
  `FixAgent._signature`), never raising the tier automatically.

- **`examples/diagnosis_loops/deco_hallu/`** (from the `jiaqiliu` merge): POPE hallucination
  slice example with a no-free-lunch guard.

### Fixed

- **Path-doubling in coded fix/M5 sandboxes**: `RunContext.root`,
  `ExperimentSandbox.workdir`, and `run_coded_pipeline`'s workdir are now
  resolved to absolute paths. A relative `run_dir` (the common case for
  examples) previously made every coded fix/M5 subprocess resolve its own
  script path a second time relative to its new cwd and fail with
  `FileNotFoundError`.
- **`cli_agent.py` venv-PATH fix** (from the `jiaqiliu` merge): spawned CLI
  coding agents now use the same Python interpreter as the loop.

### Changed

- Default `max_cases` raised (32→128) on several white-box analyzers, sized
  for enriched/stratified batches (from the `jiaqiliu` merge).
- Removed the canned `attention_guided_crop` L3a primitive — superseded by
  the L2 coded pipeline's `model_attend()` bridge, since reads need no
  privileged model handle and now go through the agent-authored sandboxed
  path instead.

### Added — experiment infrastructure (ported from AutoResearchClaw)

- **`ExperimentGitManager`** (`eval_agent/git_manager.py`): git-native run versioning.
  Each resolved diagnosis run is committed on branch `eval/{run_id}`; unresolved runs
  are discarded with `git reset --hard HEAD`.  Auto-detected when `run_dir` is inside
  a git repository.

- **`EvolutionStore`** (`eval_agent/evolution.py`): append-only JSONL store for
  cross-run lessons.  Lessons are weighted by a 30-day half-life exponential decay.
  `extract_lessons(report)` derives lessons from `AutoDiagnoseReport` automatically.
  `build_overlay(category)` formats the top-k lessons as a prompt injection string.

- **`JsonlStore`** (`eval_agent/store.py`): durable JSONL-backed implementation of the
  `Store` interface.  Hypotheses are fully round-tripped via `hypothesis_to_dict` /
  `hypothesis_from_dict` and survive process restarts.

- **`create_sandbox` factory** (`eval_agent/factory.py`): `SandboxFactoryConfig(mode=...)`
  dispatches to `ExperimentSandbox` (subprocess, default) or `DockerSandbox` (with
  graceful fallback when Docker is unavailable).

- **`ExperimentSandbox.run_project()`** (`eval_agent/sandbox.py`): multi-file project
  execution.  Path traversal protection (pre-copy syntax check + post-copy symlink
  resolve).  Immutable `experiment_harness.py` injected before execution; projects
  cannot overwrite it.  Numbered `_project_{N}` dirs (thread-safe).  Cleanup-on-success
  policy (failure artefacts preserved for debugging).  `SandboxProtocol` structural type
  for transparent backend substitution.

- **`experiment_harness.py`**: immutable evaluation harness (time budget, NaN-guarded
  metric reporting, `results.json` persistence) injected into every sandbox project.

- **`Hypothesis.to_dict` / `from_dict`** (`eval_agent/hypothesis.py`): serialization
  helpers used by `JsonlStore` and loop checkpointing.

- **Multi-phase `ExperimentWriter`** (`eval_agent/experiment_writer.py`): full port of
  AutoResearchClaw's `CodeAgent`.  Six opt-in phases:
  1. Blueprint — YAML spec with file list, per-file pseudocode, and dependency order.
  2. Sequential generation — files generated in dependency order; each prior file
     summarised by AST-based CodeMem for context injection.
  3. Hard validation — AST parse; critical issues (SyntaxError, missing `__main__` guard,
     unresolvable cross-file imports) trigger targeted repair.
  4. Exec-fix loop — parse traceback to identify failing file/line; targeted ±30-line
     context repair; falls back to full-file repair.
  5. Tree search (opt-in) — explore multiple blueprint variants, score by metrics.
  6. Review dialog (opt-in) — coder-reviewer LLM exchange; reverts if run degrades.
  `result.code` always equals `result.files["main.py"]` for backward compatibility.
  Case images are saved as JPEG files in the sandbox workdir and referenced in the
  codex prompt via `image_path` in `cases.json`.

- **Run-directory infrastructure** (`eval_agent/loop.py`): `AutoDiagnoseLoop` now
  accepts `run_dir`, `git_manager`, and `evolution_store` parameters.
  - Atomic checkpoint writes (temp+rename) to `run_dir/checkpoint.json` after every cycle.
  - Heartbeat writes to `run_dir/heartbeat.json` (pid, last_cycle, timestamp).
  - `AutoDiagnoseLoop.resume(run_dir, model, data)` classmethod reads the checkpoint
    and skips already-completed cycles.
  - `EvolutionStore` auto-created under `run_dir/evolution/` when `run_dir` is set.
  - `ExperimentGitManager` auto-detected from `run_dir` when inside a git repo.

- **VLM image-attention rule** (`eval_agent/analysis.py`): `AnalysisModule` derives
  `image_token_attention_ratio` from `top_attended_tokens` in attention findings.
  Fires a `medium`-severity finding when the ratio is below 0.05 — indicating the VLM
  is ignoring image tokens in favour of structural text tokens.

- **Diagnosis fallback** (`eval_agent/diagnosis.py`): when the LLM judge returns
  `NO_ISSUE` but M2 findings include medium-or-higher severity anomalies, `DiagnosisAgent`
  automatically generates one hypothesis per finding.  Prevents self-diagnosis bias when
  the judge is the same model under test.

- **38 new infrastructure tests** (`tests/test_eval_agent/test_infrastructure.py`):
  git manager, sandbox entry-point validation, `run_project`, harness injection,
  cleanup policy, `JsonlStore` roundtrip, `EvolutionStore` time-decay, `extract_lessons`,
  sandbox factory, `ExperimentWriterResult` backward compat, `AutoDiagnoseLoop` run_dir
  lifecycle, checkpoint/heartbeat/resume.

- **`examples/qwen_loop/`**: end-to-end `AutoDiagnoseLoop` example on Qwen3-VL-4B
  with a real (or synthetic fallback) image.  `VerboseRunLogger` mirrors each M1/M2/M3/M5
  event to stdout as it happens.  Docker Compose with CUDA 12.4 wheels, GPU selection,
  host codex binary mount, and `./outputs/` volume.

### Added
- `ModelSpec` / `Backend` / `compose()` architecture — identity separate from runtime.
- 14 model specs registered: Qwen3/Qwen2.5/Qwen2 (LLM + VLM), DeepSeek-V3, Llama 3.1, Gemma 3, GLM-4, Kimi-VL, Llama-4-Scout, Step-1o.
- Capability enum extended: `LOGPROBS`, `TOOL_CALLS` (split from `LOGITS`).
- `Agent` — backend-agnostic tool-calling loop over any model with `GENERATE + TOOL_CALLS`.
- `ToolCallCodec` — OpenAI native and Qwen/Hermes text codecs.
- `evalrx.wrap()` — captum-style on-ramp for any already-loaded HF model.
- Attention analyzers: `AttentionAnalyzer`, `AttentionRolloutAnalyzer`, `AttentionSinkAnalyzer`, `RelativeAttentionAnalyzer` (arXiv:2502.17422).
- Perturbation analyzers: `RISEAnalyzer`, `MMSHAPAnalyzer` (arXiv:2212.08158), `VLSHAPAnalyzer`.
- Uncertainty analyzers: `TokenEntropyAnalyzer`, `LogprobEntropyAnalyzer`, `SelfConsistencyAnalyzer`, `VerbalizedConfidenceAnalyzer`.
- Hallucination analyzers: `POPEAnalyzer` (arXiv:2305.10355), `CHAIRAnalyzer` (arXiv:1809.02156); stubs for OPERA and VCD.
- Lens analyzers: `LogitLensAnalyzer`; stub for `TunedLensAnalyzer`.
- Attribution stubs: `GradCAMAnalyzer`, `GenericAttentionExplainability`.
- Patching stub: `CausalTraceAnalyzer`.
- Geometry analyzers: `LinearCKAAnalyzer`; stub for `LinearProbeAnalyzer`.
- Agent analyzers: `LoopDetectAnalyzer`, `IgnoredObsAnalyzer`, `FirstErrorJudgeAnalyzer`, `CounterfactualAnalyzer`.
- Datasets: `PureQADataset`, `WebSearchQADataset`, `GUIOSDataset` → `CaseBatch`.
- Stats: `compare()` / `compare_multiple()` — effect size, clustered-bootstrap CI, e-value, BH correction.
- `eval_agent`: `EvalOrchestrator`, `PreregisteredHypothesis`, `SelfEvolveLoop` (interfaces in place, LLM proposer in Stage 2).
- CI: GitHub Actions matrix (Python 3.10/3.11/3.12) with ruff + mypy + pytest.
- PyPI trusted publishing (OIDC) release workflow.

## [0.1.0] — 2026-07-10 (evalvitals)

Initial alpha. Core contracts (`Model`, `Analyzer`, `Result`, `FailureCase`, registry, pipeline, experiment).
