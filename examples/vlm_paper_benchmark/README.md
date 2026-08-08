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

The automatic VCD route is additionally direction-gated: it is proposed only
when labelled binary diagnosis evidence is dominated by false `Yes` answers
(the object-hallucination direction the paper addresses). A slice dominated by
false `No` answers is reported as a different health problem rather than being
used to tune or deploy VCD. The 512-case adversarial validation exposed exactly
this mismatch (53 false negatives versus 27 false positives).

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
