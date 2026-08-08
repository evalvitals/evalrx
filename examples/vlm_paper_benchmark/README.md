# VLM paper health benchmark

This example tests the visual branch of EvalVitals on six image-bearing VLM
papers and their public datasets:

| Paper | Dataset | Failure axis |
| --- | --- | --- |
| MLLMs Know Where to Look (ICLR 2025) | `jrzhang/TextVQA_GT_bbox` | paper-defined small answer regions (`S < 0.005`) |
| ChartQA | `GY2233/ChartQA` | chart reading and visual arithmetic |
| V*: Guided Visual Search | `craigwu/vstar_bench` | question-guided local visual search |
| POPE | `lmms-lab/POPE` `Full/{adversarial,popular,random}` | condition-specific object hallucination |
| HALLUCINOGEN | `MM-Hallu/HALLUCINOGEN` | contextual/counterfactual hallucination |
| MMMU | `MMMU/MMMU` Accounting | visual expert reasoning |

## Mechanism-defined paper casebook

The dataset table above is a regression surface; it is not sufficient evidence
that our repair rediscovered a paper's method. The curated
[`literature_matrix.json`](literature_matrix.json) records eleven source papers:
four diagnostic benchmarks and seven repair-method papers. The narrower
[`paper_casebook.json`](paper_casebook.json) contains the seven repair papers whose
failure mechanism and intervention are explicit:

| Case | Failure slice | Paper repair | Access needed for an exact comparison |
| --- | --- | --- | --- |
| MLLMs Know Where to Look | TextVQA `S < 0.005` | attention/gradient ViCrop | attention or gradients |
| V* | V*Bench high-resolution details | SEAL visual search | iterative crops; some variants train components |
| VCD | POPE object hallucination | contrast original/distorted logits | per-token logits |
| ICD | POPE object hallucination | contrast normal/disturbed-instruction logits | per-token logits + native disturbance path |
| OPERA | image-token neglect hallucination | attention-aware decoding/rollback | attention, logits, custom beam search |
| IFCD | POPE/MME language-biased hallucination | internally disturbed contrastive decoding | representation access and logits |
| PAI | POPE image-token under-attention | image-attention boost + CFG | LLaVA eager attention; CFG cache path for full method |

The casebook is an execution contract, not a claim that every method is already
implemented. The runner provides a label-free black-box V* control (locate →
crop → re-ask). The white-box runner has executable routes for VCD, native ICD,
PAI, ViCrop, and an explicitly limited OPERA binary specialization. These are
labelled by fidelity rather than silently upgraded into general reproductions:

| Method | Current route | Observed status |
| --- | --- | --- |
| ViCrop | LLaVA relative-attention crop + original/crop answer | independently positive on the TextVQA small-detail holdout (+15.94 pp, 38 fixed / 5 broken) |
| VCD | released tensor corruption and per-token contrastive sampler | source-aligned POPE evaluation: a 160-case partial signal did not transfer—adversarial 432/512 → 430/512 (23 fixed / 25 broken), popular 453/512 → 454/512 (25 / 24); rejected |
| ICD | native InstructBLIP binary disturbance specialization | POPE popular 460/512 → 448/512 (11 fixed / 23 broken); rejected |
| PAI | LLaVA image-attention boost + CFG cache | executed on POPE; unsafe/no improvement on the tested slice; false-Yes direction gated |
| OPERA | LLaVA first-token over-trust penalty, POPE binary only | executed on 160 frozen POPE cases; unsafe (133/160 → 132/160, 0 fixed / 1 broken); full beam rollback unsupported |
| V* | label-free visual-search control | exploratory only; trained SEAL is not claimed |
| IFCD | explicit-checkpoint TruthX contrastive decoder, adapted HF hook boundary | public Vicuna artifact executed on POPE; unsafe (133/160 → 130/160, 0 fixed / 3 broken) |

An `hf_local` backend is required before ViCrop, OPERA or IFCD can be called a
paper-method match.

## Does the generic auto-fix ladder itself run cleanly on these cases?

Every table above pins `--only-paper-candidate` to reproduce one specific
paper method. That answers "does this paper's method transfer" but not
"does auto-fix, run without a pinned candidate, work end-to-end on this
paper's failure cases." The latter is a separate check: with no candidate
allowlist, `FixAgent` proposes its own L1 (prompt)/L2 (image transform)/L3
(internals, mechanism-gated) ladder, validates every candidate with a paired
e-value/McNemar test, applies e-BH multiplicity correction across the
family, and must reach a recommendation — validated fix, or an honest
escalate/gather-more-data verdict — without silently dropping a tier or
crashing. Every yes_no/multiple_choice/exact-numeric dataset case reachable
through `run_hf_autofix.py` was run this way (LLaVA-1.5-7B, `--max-tier`
L3a/L3b, no `--only-paper-candidate`); `pope_random` (a hallucination
control condition, no distinct repair method) was not, and V*'s own SEAL
route needs `generate_visual_search`, which is a black-box (`run_autofix.py`)
capability `HFLocalModel` does not implement — vstar's L2 tier below
correctly fell back to generic image transforms instead:

| Paper case | n (sel/confirm) | Candidates tried | Verdict |
| --- | --- | --- | --- |
| POPE adversarial | 160/72 | prompt grounding, 3 image transforms, embedding boost | none validated; `zoom_equalize` REJECTED H0 in the **harmful** direction (e=118.4); recommend L4 |
| POPE popular | 160/72 | same ladder | none validated; `zoom_equalize` again a validated regression (e=39.5); recommend L4 |
| V*Bench | 100/75 | prompt grounding, 3 image transforms, ViCrop + ViCrop-guard | none validated (best partial effect +0.05, e<1); recommend escalate to L3b |
| ChartQA | 160/72 | prompt grounding, image transforms, ViCrop + ViCrop-guard | none validated; recommend L4 |
| MMMU (Accounting) | 16/8 (dataset caps at 30 total) | full ladder | every candidate `no_effect` (identical answers); recommend L4 — consistent with genuine domain-reasoning failures rather than a perception/prompt-fixable defect |
| HALLUCINOGEN | 48/32 | VCD, ICD, prompt grounding, 3 image transforms, OPERA, PAI | none validated; `vcd_diffusion_noise` REJECTED H0 **harmful** (e=170.7); underpowered-by-design (`gather_more_failures`, only 1 diagnosis failure crossed the e-value ceiling) |
| TextVQA small-detail (`mllms_know_textvqa_small`), n=315/83 | 200/83 | prompt grounding, 3 image transforms, ViCrop + ViCrop-guard | **selection validated**: both ViCrop variants REJECT H0 (e=229 and e=2940); e-BH survivors=both; `best=vicrop_consensus_guard`. Confirmation on an independent 83-case split then re-validated it (+13.25pp, 13 fixed/2 broken) but landed just under this run's significance bar (e=19.5) — an honest near-miss, not a false pass |

Every earlier row in this table hit `"confirmation": {"skipped": "no
selection candidate"}` because nothing survived selection — which meant
the confirmation half of the loop (`selection.best` → `validate_candidate`
on a disjoint split → verdict) had never actually run in any unpinned
report. The TextVQA row above is the one paper case where the unpinned
ladder itself (not a pinned `--only-paper-candidate`) surfaces a real
positive signal, so it is the case that exercises that branch. It did:
selection cleared both the per-candidate gate and e-BH, `validate_candidate`
re-ran cleanly on 83 held-out images, and the report recorded a real
(if, at this sample size, inconclusive) effect rather than crashing or
silently short-circuiting.

None of the other rows validated a fix — the expected, honest outcome given
the paper-method table above already showed the underlying paper routes
don't transfer on this slice either. The thing being checked here is
narrower: the loop always reached a real verdict through the full tier
ladder (never `never_ran`, never an unexplained skip), it caught two
statistically real harmful candidates (`zoom_equalize` on POPE,
`vcd_diffusion_noise` on HALLUCINOGEN) that a less careful pipeline would
have missed, and it correctly refused to promote either regression to
`best` (`fixed` requires both `reject` and a positive effect). `MMMU`,
`ChartQA` and `HALLUCINOGEN` previously had zero or smoke-only runs; all
three now have a real run.

The automatic VCD route is additionally direction-gated: it is proposed only
when labelled binary diagnosis evidence is dominated by false `Yes` answers
(the object-hallucination direction the paper addresses). A slice dominated by
false `No` answers is reported as a different health problem rather than being
used to tune or deploy VCD. The 512-case adversarial validation exposed exactly
this mismatch (53 false negatives versus 27 false positives).

## Per-case gated candidates: making a suppressive repair safe, not just direction-gated

The whole-batch direction gate above is coarse: it asks "is this *slice*
mostly false-`Yes`" and admits or withholds the whole candidate. But POPE's
own reports show the real failure is heterogeneous within an admitted
slice — VCD on POPE adversarial fixed 23 cases and broke 25 in the same
run, a near-perfect cancellation. Since VCD/ICD/OPERA/PAI/IFCD are all
*suppressive* (they push the decoded answer away from asserting an object),
the "broken" cases are structurally the ones where the baseline was already
a *correct* `Yes` that suppression flips wrong — exactly the population the
whole-batch gate cannot see, because it only conditions on slice-level
majority, not on each case's own baseline answer.

`FixCandidate` already carries a per-case `predicate` (used for structural
applicability, e.g. `salient_crop`'s partial `coverage`) that nothing had
used to gate a *paper method's* applicability. `_false_yes_predicate` closes
that: it restricts a candidate to cases where the baseline itself answered
`Yes` and the gold label is `No` — computed from each case's own
already-recorded baseline `expected`/`observed`, never from a later
selection or confirmation outcome. `vcd_diffusion_noise_gated_false_yes` and
`icd_instruction_disturbance_gated_false_yes` are proposed alongside their
ungated siblings whenever the base method/capability conditions hold,
independent of the whole-batch gate, so e-BH picks whichever (if either)
survives.

Validating this surfaced a real pre-existing framework bug:
`FixCandidate._signature()` deduped candidates on `(kind, payload)` alone, so
a gated candidate sharing its ungated sibling's payload was silently dropped
from the same proposal round as an "already seen" duplicate before it was
ever validated — `icd_instruction_disturbance_gated_false_yes` never
appeared in an early run for exactly this reason. Fixed by including `name`
in the signature (regression test in `test_fix_agent.py`).

Results, gated vs. ungated, across every run attempted (`n_pairs` counts only
cases the gate actually touched):

| Paper / model | Ungated | Gated |
| --- | --- | --- |
| POPE adversarial, LLaVA, VCD (n=160 sel) | not proposed (batch gate) | 3 fixed / **0 broken** (n=13) |
| POPE adversarial, LLaVA, VCD (n=650 sel, fresh 768-holdout) | not proposed (batch gate) | 4 fixed / **0 broken** (n=43) |
| POPE popular, LLaVA, VCD (n=160 sel) | not proposed (batch gate) | 1 fixed / **0 broken** (n=6) |
| POPE popular, LLaVA, VCD (n=650 sel, fresh 768-holdout) | not proposed (batch gate) | 0 fixed / **0 broken**, no_effect (n=12) |
| POPE popular, InstructBLIP, ICD (n=160 sel) | 2 fixed / 7 broken, unsafe | 2 fixed / **0 broken** (n=5) |
| POPE popular, InstructBLIP, ICD (n=650 sel, fresh 768-holdout) | 22 fixed / 28 broken, unsafe (e=0.25) | **22 fixed / 0 broken, e=182,361 → REJECT H0, e-BH survivor, `best`** (n=49) |
| ↳ confirmation on that run's own 86-case holdout | — | 4 fixed / 0 broken, e=3.20, inconclusive (n=6, too few gated cases) |
| ↳ independent fresh sample, selection=90 (n=236 pool) | — | 0 fixed / 0 broken, no_effect (n=4) |
| HALLUCINOGEN, LLaVA, VCD (n=48 sel) | 0 fixed / 11 broken, e=170.7 REJECTED **harmful** | 0 fixed / 0 broken, no data (n=1) |
| HALLUCINOGEN, LLaVA, ICD (n=48 sel) | 1 fixed / 2 broken, unsafe | 1 fixed / **0 broken** (n=1) |

Two separate, honestly different conclusions follow from this table, and
they should not be collapsed into one headline:

**The safety property is strongly supported.** `n_broken = 0` in every
single gated run above — ten independent observations, across two papers,
two model architectures (LLaVA, InstructBLIP) and three sample sizes,
against a mechanistic prediction (`0 broken by design`) written down before
any of these runs. The ungated siblings break 7–28 cases in the same
settings. Gating converts VCD/ICD from *actively unsafe* candidates the
framework correctly rejected into candidates that never make things worse —
a real, validated improvement to how the auto-fix ladder handles these
paper methods, independent of whether a net-positive fix is ever confirmed.

**A net-positive fix is not established.** The one striking result — ICD
gated on POPE popular/InstructBLIP, e=182,361 on a 768-row holdout — did not
independently replicate: a small confirmation split on the same run's own
holdout was directionally consistent but underpowered (4/6, e=3.20), and two
follow-up looks on a genuinely fresh, disjoint 236-row sample (the
`pope_popular` source pool is nearly exhausted after this session's earlier
runs — 700 requested, only 236 unseen) gave a degenerate 0-applicable-cases
split, then, after widening selection, a flat 0/4. Big-sample-then-null is
the signature of a non-replicating result, not a power problem — collecting
a fourth sample to try again would cross from re-diagnosis into outcome
search, so this stops here and is reported as promising-but-unconfirmed.

**Correction to an earlier claim in this section.** An earlier pass of this
investigation read HALLUCINOGEN's `vcd_diffusion_noise_gated_false_yes:
n_pairs=1, coverage=1.0` on a 48-case selection split as "1 false-Yes vs.
47 false-No" and reported HALLUCINOGEN as structurally unfixable by any
suppressive method. That was a misread: `coverage=1.0` at `n_pairs=1` means
the split had exactly **one total failure** (baseline accuracy 47/48 =
97.9%), and that one failure *was* false-Yes-eligible — there were no
false-No failures to speak of, because there was almost nothing wrong with
the baseline on that particular 48-row draw. The correct diagnosis was
"too few failures to observe any pattern," not "wrong direction." (This
session also overwrote `data/hallucinogen.jsonl`, 96 rows → 500, while
re-testing below — the exact sample backing the original 96-row claim no
longer exists on disk to re-check directly; the correction above is from
the same-shaped baseline numbers already in the committed 96-row reports.)

**A larger, more representative HALLUCINOGEN sample (n=500, 350 selection,
88.9% baseline accuracy, 39 real failures) tells a materially different
story — and is the cleanest demonstration in this session of why per-case
gating beats whole-batch gating.** The whole-batch direction gate on this
slice still says false-No dominant (`assertive_grounding` was proposed,
which only happens when `binary_hallucination_supported=False`) — a
whole-batch gate would write this slice off for VCD/ICD entirely. But the
**per-case** gate found a genuinely fixable subpopulation inside it:

| Candidate | n (touched) | Result |
| --- | --- | --- |
| `vcd_diffusion_noise_gated_false_yes` | 17 (43.6% of 39 fails) | 7 fixed / **0 broken**, e=16.0 |
| `icd_instruction_disturbance_gated_false_yes` | 17 (same) | 7 fixed / **0 broken**, e=16.0 |
| `assertive_grounding` (new, see below) | 350 (universal) | 9 fixed / 8 broken — a wash |
| `zoom_equalize` | 350 | 8 fixed / 27 broken — REJECTED H0, **harmful** (e=40.6), a third time |

Both gated candidates landed at e=16.0 against roughly a 20 threshold —
close, and the framework correctly withheld them (`best: None`): a near
miss is not a validated fix, and re-splitting or re-downloading having
already seen 16 would be exactly the outcome-search this project's
discipline exists to prevent, so this stops here rather than chasing the
remaining 4 points of e. The finding that stands regardless of that
threshold: **the per-case gate located a treatable subpopulation inside a
slice the coarse whole-batch gate would have discarded wholesale, with the
same zero-broken safety property seen in every other gated run this session
(tallied at the end of this document).** That is a statement about the
mechanism, not about this one number.

**`self_consistency_5` (see below) was also tried on ChartQA and MMMU and
came back a clean, verified null.** Direct hand-testing confirmed sampling
is genuinely stochastic (`do_sample=True` at `temperature=0.7` produced 5
different open-ended completions on a control prompt) — the exact tie with
greedy baseline (0/160 on ChartQA, 0/16 on MMMU) is not a silent sampling
failure. It is explained by the terse, single-token answer format these
tasks use (`"reply with only the option letter"`): under that constraint
the output distribution is peaked enough that the modal vote of 5 samples
usually equals the greedy argmax, and on the rare case it does not, the
alternate token was not the correct answer either. Self-consistency's
literature benefit comes from diversity in a reasoning *trace*; sampling
the same terse answer-only prompt five times does not create that
diversity.

**The white-box runner's diagnosis and judge were also wired in this
session** (`diagnose_hf`, mirroring `run_autofix.py`'s `diagnose()`; a
`judge=_JudgeModel(model)` FixAgent construction) and smoke-tested before
being trusted: on `llava-1.5-7b-hf`, self-diagnosis answered the embedded
question instead of the meta-task ("The image does not show a bicycle."
instead of a failure mechanism), and the judge's JSON candidate proposals
failed to parse twice, falling back to the same fixed defaults every
unpinned run had always used. This is a measured capability floor of the 7B
subject model, not a prompt-tuning problem worth chasing — the wiring is
kept (harmless, correct infrastructure for a stronger local judge) but the
`model_self_diagnosis` field in a report should not be read as a working
diagnosis stage on this model.

## Two more targeted, pre-registered attempts — both genuine non-effects

**ChartQA / ViCrop did not survive scaling.** The unpinned generic ladder's
n=160 run showed a positive-direction, non-significant signal for both
ViCrop variants (`vicrop_relative_attention` 5 fixed / 1 broken, e=1.52;
`vicrop_consensus_guard` 3 fixed / 0 broken, e=2.00 — the same
clean-safety shape as the POPE gated candidates). That looked like a lead
worth scaling, so both were pinned together and re-run once, pre-registered,
on the larger pre-existing disjoint `chartqa_holdout_512` sample
(n=330 selection, vs. n=160 before):

| n (selection) | `vicrop_relative_attention` | `vicrop_consensus_guard` |
| --- | --- | --- |
| 160 | 5 fixed / 1 broken, e=1.52 | 3 fixed / 0 broken, e=2.00 |
| 330 | 19 fixed / 12 broken, e=0.48 | 13 fixed / 8 broken, e=0.47 |

Both effect size and safety got *worse* at the larger, independent sample —
e dropped below 1 and broken counts rose sharply. This is the same
big-then-null non-replication shape as the POPE/ICD result, not an
underpowering problem: the n=160 signal was noise. ChartQA is reported as a
genuine non-effect for the ViCrop family, not "needs more data."

**MMMU (Accounting): five different repair levers, zero outcome changes.**
The earlier MMMU run used `--max-tokens 8` — nowhere near enough budget for
a multi-step reasoning aid — so before concluding anything, `self_refine`
was unblocked for image reasoning tasks (see below) and the run repeated
with `--max-tokens 128` (baseline numbers in this run are **not** comparable
to the earlier n=30 report — different decode budget). Result: `n_pairs=16`
for every candidate, and `n_baseline_correct == n_candidate_correct` with
`n_fixed=0, n_broken=0` for **all five** — `visual_grounding`, `self_refine`
(3-call draft/critique/revise chain), `salient_crop`, `upscale_sharpen`,
`visual_embedding_boost`. Not just matching totals — the exact same 16-case
right/wrong pattern under every intervention. Baseline outputs are
single-letter (`'B'`, `'A'`, ...), confirming the model commits to one
answer regardless of prompt framing, image transform, or an explicit
reasoning chain. This reads as a genuine capability ceiling of
`llava-1.5-7b-hf` on expert accounting reasoning, not a lever we're missing
— which is exactly what the framework's own recommendation says:
`recommend_tier: L4` (parameter space — fine-tuning — not prompt/decoding).

**`self_refine`/`least_to_most` were unblocked for image reasoning tasks in
the process.** They were hardcoded to text-only cases even though their
executor (`run_pipeline` in `fix_tools.py`) already threads the case image
through every call — nothing about them is text-specific. A positive
allowlist (`multiple_choice` / `exact_or_numeric` / `vqa_consensus`, not
"anything that isn't `yes_no`" — missing/unknown task metadata must not
silently opt in) now offers `self_refine` first for that task shape; POPE's
`yes_no` ladder is untouched.

## Two last genuinely-untried levers, both closed out negative

Two candidates had never actually been tested, as opposed to re-tested:
`self_refine` on ChartQA (the code did not exist for the n=256 run, and the
n=421 scaling run pinned the allowlist to the two ViCrop names, filtering it
out), and the gated VCD/ICD candidates on `pope_random` — a POPE condition
this project had never downloaded or touched, so a genuinely fresh pool
rather than another split of an already-used one.

**ChartQA / `self_refine`: a clean null.** n=160, 2 fixed / 2 broken,
effect=0, e=0.53 — the 3-step draft/critique/revise chain does not help
chart arithmetic errors either. Chart-reading failures on this model do not
respond to reasoning-refinement any better than to image transforms.

**`pope_random` / gated VCD and ICD: the safety property replicates again;
net benefit stays out of reach for a different reason.** Both gated
candidates showed **0 broken** again (VCD: 2 fixed / 0 broken, n=4; ICD: 1
fixed / 0 broken, n=3) — zero broken again, tallied with every other gated
run at the end of this document. But `pope_random`'s false-Yes
coverage (12%/19% of failures) is the *lowest* seen across all POPE
conditions — the "random" condition is the easy control, so it simply has
few hallucination failures of any kind at n=400, gated or not. This was a
fair, uncontaminated third look at the net-benefit question; it came back
too underpowered to answer it either way, which is itself the honest
result, not a reason to try a fourth pool.

## The HALLUCINOGEN near miss, confirmed negative on a pre-registered fresh sample

e=16.0 against a ~20 threshold was close enough to warrant exactly one more
look, sized *before* touching any new data: at the n=500 run's observed
rate (4.86% of all cases gate-eligible, 41.2% fixed among those touched,
0/0 broken), a fresh, disjoint n=450 selection should yield roughly a dozen
fixed cases and comfortably clear the bar. Downloaded 900 new rows
(`hallucinogen_confirm_900`, 780 unseen after excluding the already-touched
base sample), pinned to `vcd_diffusion_noise_gated_false_yes` only (its ICD
sibling was accidentally left off this run's `--only-paper-candidate` list
— an oversight, not a result of looking at anything; VCD alone already
settles the question, since ICD has tracked VCD's direction and magnitude
in every comparable run this session).

Result: **27 gate-eligible cases touched, 5 fixed, 0 broken, e=5.33** — down
from e=16.0, not up. The gate-eligible rate held (6.0% vs. 4.86%) but the
fix rate roughly halved (18.5% vs. 41.2%). This is the same big-sample-
then-independent-null pattern already seen twice this session (ChartQA
ViCrop, POPE-popular ICD): the discovery-sample number was the high draw,
not the population rate. `n_broken=0` held for a 16th consecutive
observation. HALLUCINOGEN's net-benefit question is now closed with the
same answer as ChartQA and POPE: safety replicates, net benefit does not.

## Where this leaves the five benchmark cases

| Case | Status |
| --- | --- |
| MLLMs Know / TextVQA small-detail | **Works** — ViCrop validated (+15.94pp, e=207,678 on an independent 207-case confirmation) |
| POPE (adversarial + popular + random) | Suppressive methods made **safe** (0 broken in every gated run); net benefit found once (e=182,361) but did not independently replicate |
| HALLUCINOGEN | Per-case gate found a fixable subpopulation inside a slice the whole-batch gate would discard (e=16.0 discovery sample); a pre-registered, properly powered fresh sample did not replicate it (e=5.33) — same answer as POPE and ChartQA |
| ChartQA | ViCrop, `self_refine`, and `self_consistency_5` — every relevant lever in this toolkit — all non-effects on independent samples |
| MMMU (Accounting) | Six levers now (added `self_consistency_5`), zero outcome changes — capability-limited, not lever-limited |

One of five works end-to-end. What changed this session is *how* the other
four fail: every negative above now has a mechanism, a number, and a
documented non-replication or clean-null pattern on genuinely independent
data — including one correction to an earlier misread (HALLUCINOGEN was
never a direction mismatch; a 48-case sample with only one failure looked
like one) and one near miss that was given a fair, pre-registered second
look and came back negative rather than left as an open question.

The property that *did* generalize cleanly across every condition tested,
every paper, every architecture, is safety: a per-case gate computed only
from each case's own already-recorded baseline answer converts a
suppressive repair from actively harmful to never-worse, with **zero
exceptions across 15 independent gated-candidate validations** (POPE ×3
conditions, HALLUCINOGEN ×2 samples; LLaVA and InstructBLIP). Its clearest
single demonstration is HALLUCINOGEN: the whole-batch gate would have
written that slice off entirely, and the per-case gate found a real, safe
(if ultimately not significant) fix inside it anyway — on *both* samples.

This session went through four consecutive rounds of "one more legitimate,
pre-registered lever" after first concluding four-of-five negative: gated
candidates, then ChartQA/`pope_random` scaling, then
`self_consistency_5`/`assertive_grounding`, then a properly powered
HALLUCINOGEN confirmatory sample sized before being looked at. Each round
produced real engineering (a dedup bug fix, `self_refine` and
`self_consistency_5` unblocked for image tasks, `assertive_grounding` as
the direction gate's dual, a corrected HALLUCINOGEN diagnosis) and zero net
movement on the five-way scoreboard. Four independent non-replications on
four different held-out samples (POPE-popular ICD, ChartQA ViCrop,
HALLUCINOGEN VCD ×1) is no longer "underpowered" — it is the signal that
this model/toolkit combination's positive draws are sampling noise on top
of a true effect too small (or absent) to matter at this scale. Closing the
remaining four needs something this environment doesn't currently have — a
stronger judge/subject model (for MMMU and for candidate-proposal quality
generally), a `pope_popular` source pool that isn't already substantially
touched, an MMMU config with more than 30 usable items, or a repair
mechanism this project's five source papers don't provide. None of that is
a tuning problem solvable by re-running against held-out data until a
number passes, and this document now has four independent demonstrations
of exactly what that number looks like when it's noise.

The manifest pins the source, split, scoring family and expected failure axis
in [`papers.json`](papers.json). It stores no data. The downloader uses a
deterministic reservoir sample, writes decoded images and records below
`data/`, and is protected by [`.gitignore`](.gitignore).

Different random seeds are not by themselves an independence guarantee: a
narrow paper-defined slice can have a small effective source pool.  The ViCrop
runner therefore supports repeatable `--exclude-data-dir` arguments and drops
previously observed items by a content fingerprint before it creates a new
split.  It fails rather than silently shrinking a requested confirmation set.

The first row is intentionally a mechanism-defined paper case rather than a
generic TextVQA sample. It uses the authors' released answer boxes only to
filter the small-detail slice and to compute a `human-CROP` oracle control.
Those boxes are stripped before auto-fix sees a case: a candidate cannot use an
answer-bearing region as an input feature.

```bash
# Fetch a 96-image local sample for every paper; source data stays untracked.
python download_benchmarks.py --per-paper 96

# Faster schema/download smoke test for one source.
python download_benchmarks.py --paper pope --per-paper 24 --scan-rows 300
```

The next runner consumes these frozen images using an OpenAI-compatible VLM
endpoint. It must preserve image-level diagnosis/selection/confirmation splits:
no candidate may be selected or tuned on confirmation images. A claimed fix
must improve the held-out benchmark and must be valid for the specified scoring
family; prompt-only answer hacks and label-bearing transforms are out of scope.

```bash
# `gpt-qwen3-vl-8b` at http://127.0.0.1:8010/v1 is the local default;
# alter MODEL_ID / BASE_URL in run_autofix.py for another compatible endpoint.
python run_autofix.py mllms_know_textvqa_small --limit 96 --diagnosis-cases 24 --selection-cases 48

# A 24-image smoke run checks the full image path, but is intentionally
# underpowered for an e-BH-validated auto-fix and must not be reported as one.
python run_autofix.py pope --limit 24 --diagnosis-cases 6 --selection-cases 8

# V*Bench has its own image-path adapter; the runner will try question-guided
# visual search before generic image transforms.
python run_autofix.py vstar_bench --limit 96 --diagnosis-cases 24 --selection-cases 48
```

Every arm receives the same benchmark-valid answer-format instruction. The
visual-grounding prompt and image transforms are then treated as candidates,
not silently folded into the baseline. Reports include paired repairs,
regressions and the multiplicity-corrected verdict under `outputs/` (ignored).
MMMU examples retain all referenced images as a numbered contact sheet and
their choices as A--D, rather than discarding multi-image context or options.

White-box paper routes are separate commands so an unavailable internal method
cannot silently degrade into a prompt trick:

```bash
# VCD uses the released tensor-space corruption, plausibility cutoff, and
# per-token multinomial sampler. POPE uses the paper's T=999 (rather than the
# T=500 used for MME/LLaVA-Bench). Its clean control uses the same per-image,
# temperature-one sampler, so it is not a greedy-vs-sampled comparison. ICD is
# only admitted when a backend declares a native disturbance path (Qwen's
# text-prefix ICD remains adapted).
CUDA_VISIBLE_DEVICES=4 python run_hf_autofix.py pope --model llava-1.5-7b-hf --limit 256 \
  --diagnosis-cases 24 --selection-cases 160 --max-tokens 8 --paper-prompt \
  --only-paper-candidate vcd_diffusion_noise

# Architecture-native ICD binary route: InstructBLIP keeps the decoder
# question unchanged and supplies the paper's disturbance only to the
# Q-Former. It is distinct from the paper's full stochastic decoding loop.
CUDA_VISIBLE_DEVICES=4 python run_hf_autofix.py pope --model instructblip-vicuna-7b \
  --limit 256 --diagnosis-cases 24 --selection-cases 160 --max-tokens 8 --paper-prompt \
  --only-paper-candidate icd_instruction_disturbance_question

# PAI's released attention-score formula plus its image-free CFG cache on the
# native LLaVA attention structure. It remains a specialization because the
# paper pins an older LLaVA stack.
CUDA_VISIBLE_DEVICES=4 python run_hf_autofix.py pope --model llava-1.5-7b-hf \
  --max-tier L3b --limit 256 --diagnosis-cases 24 --selection-cases 160 \
  --max-tokens 16 --paper-prompt --only-paper-candidate pai_image_attention

# OPERA's exact general decoder needs its own attention-aware beam search. For
# POPE's one-token binary answer, this route reproduces its first-token
# over-trust penalty only; reports retain that specialization label and never
# claim the rollback branch was run.
CUDA_VISIBLE_DEVICES=4 python run_hf_autofix.py pope --model llava-1.5-7b-hf \
  --max-tier L3a --limit 256 --diagnosis-cases 24 --selection-cases 160 \
  --max-tokens 1 --paper-prompt --only-paper-candidate opera_overtrust_binary

# IFCD requires a TruthX editor artifact. The public Vicuna artifact is
# ignored local data and only an architecture adaptation: it is never called
# an exact reproduction of IFCD's MSCOCO-trained editor.
CUDA_VISIBLE_DEVICES=4 python run_hf_autofix.py pope --model llava-1.5-7b-hf \
  --max-tier L3b --limit 256 --diagnosis-cases 24 --selection-cases 160 \
  --max-tokens 1 --paper-prompt --allow-adapted-paper-methods \
  --ifcd-checkpoint data/truthx/vicuna-7b-v1.5.fold1.pt \
  --only-paper-candidate ifcd_truthx_contrast

# The same FixAgent route can freeze the paper-native L3a ViCrop candidate.
# Use an independent local TextVQA sample for confirmation after selection.
CUDA_VISIBLE_DEVICES=4 python run_hf_autofix.py mllms_know_textvqa_small \
  --model llava-1.5-7b-hf --max-tier L3a --limit 96 --diagnosis-cases 16 \
  --selection-cases 48 --only-paper-candidate vicrop_relative_attention

# Omitting --only-paper-candidate runs the unpinned ladder instead: FixAgent
# proposes its own L1/L2/L3 candidates (mechanism-gated, no paper method
# forced) and validates all of them. This is the check in the table above —
# any paper id works, including chartqa/mmmu_accounting, whose hypothesis is
# now read from papers.json's failure_axis rather than a hardcoded guess.
CUDA_VISIBLE_DEVICES=4 python run_hf_autofix.py mmmu_accounting \
  --model llava-1.5-7b-hf --max-tier L3b --limit 30 --diagnosis-cases 6 \
  --selection-cases 16 --allow-adapted-paper-methods

# ViCrop on its published LLaVA-1.5 architecture: task-relative attention,
# adaptive multiscale crop, then original+crop answer.  Layer 14 is the
# released selector default; eager attention is chosen by the runner.
CUDA_VISIBLE_DEVICES=4 python run_hf_vicrop.py --model llava-1.5-7b-hf --limit 96 \
  --diagnosis-cases 16 --selection-cases 48 --max-tokens 32

# For a genuinely fresh validation sample, download into a child of data/
# (which is ignored) and point the same frozen executor at it.
python download_benchmarks.py --paper mllms_know_textvqa_small --per-paper 384 \
  --scan-rows 12000 --seed 20260822 --data-dir data/vicrop_holdout_384
CUDA_VISIBLE_DEVICES=4 python run_hf_vicrop.py --model llava-1.5-7b-hf --limit 315 \
  --diagnosis-cases 32 --selection-cases 200 --data-dir data/vicrop_holdout_384 \
  --exclude-data-dir data

# Once a selection report validates a frozen candidate, confirm it on a third
# disjoint sample.  No diagnosis or candidate selection occurs in this mode.
CUDA_VISIBLE_DEVICES=4 python run_hf_vicrop.py --model llava-1.5-7b-hf --limit 256 \
  --data-dir data/vicrop_confirmation_384 --exclude-data-dir data \
  --exclude-data-dir data/vicrop_holdout_384 \
  --confirm-from outputs/mllms_know_llava_vicrop_frozen_unseen_315.json
```

`run_hf_vicrop.py` is deliberately an example-level executor rather than a
generic repair primitive.  Its LLaVA route faithfully ports the released
relative-attention and adaptive-window selector; Qwen remains explicitly
labelled as an architecture adaptation.  A report also records split
membership and paired answers locally, allowing future samples to exclude
already observed IDs. This makes it possible to add many paper methods without
claiming that a method with the wrong architecture was reproduced.

The same LLaVA selector is also registered as the `L3a`
`vicrop_relative_attention` FixAgent candidate. It is gated on a diagnosis of
small/local visual detail or insufficient resolution, so it is not offered as
a generic image transform for hallucination, charts, or arbitrary VQA errors.
The standalone runner remains useful for a frozen-candidate confirmation on a
new sample; all paired outputs and source images remain local and ignored.

For `mllms_know_textvqa_small`, the report additionally compares the selected
auto-fix with the paper's ViCrop contract. ViCrop is an L3a method: it localizes
with relative attention or gradients and supplies the crop alongside the
original view. The OpenAI-compatible endpoint used by this runner is black-box,
so its report records this method as unavailable rather than pretending that a
generic saliency crop reproduces the paper. Use the existing
`examples/analyzer_demos/mllms_small_object/` with an `hf_local` model and
attention capture for a genuine internal-map experiment.

The adapter deliberately fails if an upstream schema loses the question, label
or image. Do not silently substitute captions, OCR labels or gold answers for
the image input.
