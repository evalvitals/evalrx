# audiocaps_hallucination_qwen2_audio

Full M1→M5 diagnosis loop: does the agent independently discover that
Qwen2-Audio-7B-Instruct over-affirms sounds that aren't in the clip
("language priors override audio evidence"), and reach for AAD
(Audio-Aware Decoding, Hsu et al. 2025, arXiv:2506.07233) to fix it — the
same way it already reaches for TCD on `examples/m1_m5/mmau_qwen2_audio`?

## The paper and dataset

AAD is a training-free, inference-time contrastive-decoding fix: at every
decode step it contrasts the model's real-audio logits against the same
prompt with the waveform silenced,
`logits' = (1+α)·logit(with audio) − α·logit(silenced)`. Its own released
code (`github.com/GillbertHsu/Audio-Aware-Decoding`) evaluates on
**Qwen2-Audio-7B-Instruct** — the exact model `mmau_qwen2_audio` already
runs — using stock `transformers.LogitsProcessor` + `Qwen2AudioForConditionalGeneration`,
not a frozen custom fork. (An earlier attempt at AVCD, arXiv:2505.20862, on
VideoLLaMA2 hit a real version-skew bug in that paper's own released code
under current transformers; AAD's much simpler mechanism and stock-API
implementation avoids that class of problem entirely.)

The benchmark itself is Kuan et al. 2024 (Interspeech 2024, arXiv:2406.08402,
"Understanding Sounds, Missing the Questions"): binary yes/no questions
("Is there a sound of X in the audio?") over AudioCaps clips, with
random/popular/adversarial negative-sampling variants. `download_audiohallucination.py`
pulls the questions from `kuanhuggingface/AudioHallucination_AudioCaps-<sampling>`
and the actual audio from a second HF mirror, `OpenSound/AudioCaps` (joined
on AudioCaps' own `youtube_id`), decoding through ffmpeg the same way
`mmau_qwen2_audio/download_mmau.py` does — no `datasets`/torchcodec
dependency.

## Usage

```bash
python download_audiohallucination.py --limit 300 --scan-rows 2000
python run.py --model qwen2-audio-7b-instruct --limit 300
python run.py --smoke-test     # fast wiring check, no GPU/model/judge
```

300 rows split 120 explore (M1-M4 discovery, fix *selection*) / 180 confirm
(held-out M4 + fix *validation*) by `--confirm-split 0.6`; the old 120-row
default left only 48 discovery cases, too few for M2/M4 or the fix gate.

The fix pool is open by default (`candidate_allowlist=None`, `--fix-max-tier L3a`
like `mmau_qwen2_audio`): the judge's L1/L2 candidates, the `self_consistency_5`
floor, a coded L2 pipeline and -- when its admission gate fires -- AAD all
compete, selected on the explore half and confirmed on the untouched confirm
half. `--paper-method-only` restores the AAD-only pool with no coder. AAD is
L3a because it reads and combines logits from paired real/silenced-audio
forwards; `--fix-max-tier L0` now admits runtime-configuration repairs only.

Or via Docker (see `docker-compose.yml` — mirrors `mmau_qwen2_audio`'s setup
exactly, same base model, same judge/audio volume mounts):

```bash
docker compose up
```

## What "fixed" means here

`fix_agent.py`'s admission gate for AAD (`aad_silence_contrast` /
`aad_silence_contrast_gated_false_yes`, both L3a — they read and combine logits
from two `generate()`-compatible forward passes) requires
`case.metadata["task"] == "yes_no"`, `model.generate_aad` to exist, and
`paper_method_fidelity("aad") == "native_silence_contrast"`. Like TCD on
`mmau_qwen2_audio`, this run.py never names AAD or its mechanism anywhere
in the protocol or dataset glue — if it gets proposed and validated, that
has to come from the loop's own M1→M4 discovery and the judge's own
mechanism-match against whatever hypothesis M3 produces, not a hand-supplied
hint.

## Run log: 2026-08-20, open pool (tealab RTX A6000, docker, 55 min)

`docker compose up` with the defaults above (300 rows, judge `sonnet`/`high`,
`--fix-max-tier L3a`, open pool). Fresh greedy baseline **67.0% (201/300)**;
split 120 explore / 180 confirm. Outputs in the llm_benchmark layout:
`outputs/logs/` (run_log.jsonl, artifacts, `figures/m2_effects.png`,
`experiments/post_m5_*`, `fixes/outcome.md`) + `outputs/explore/` (12 PNGs).

| stage | what happened |
|---|---|
| M1 (168 s) | 7 pinned analyzers, 40/120 FAIL. `termination_audit` flags 91% "truncated" -- the model's terse `Yes`/`No` against a 16-token budget; `answer_extraction_audit` 0 suspects. |
| explore (562 s) | 82% of answers are `Yes`; absent-sound questions fail 69% vs 7% for present-sound ones. |
| M2 (59 s) | Only BH survivor is `looks_truncated` (+0.39); M2 cross-checks it against extraction/continuation and dismisses it as the terse-answer artifact, concluding a confidently-wrong `Yes` bias on absent sounds (calibration tau ~ 0, overconfidence gap 0.23). |
| M3 (146 s) | 3 leads: `language_prior_bias`, `brittleness`, `self_correction_failure`. The critic rejected all three (kept as flagged leads). |
| held-out M4 (180 rows) | `brittleness` SUPPORTED (`perturbation_battery.noop_clause_flipped` vs FAIL +0.50, CI +0.21..+0.72, BH p=0.0023, n=10). `language_prior_bias` INCONCLUSIVE: its test design named `answer_extraction_audit.extracted_answer` / `labelled_fail`, categorical fields no analyzer exposes as an M2 signal. `self_correction_failure` inconclusive (conf_verbal -0.24, BH not survived). |
| M5 | Coder experiment on the verified lead: three meaning-preserving paraphrases per prompt, FAIL flip rate 3.0% vs PASS 0% -- below the pre-set 10-point gap -> REFUTED (small n under the 50 s sandbox budget). |
| FIX (31 min) | 12 candidates raced on the explore half (table below); `aad_silence_contrast_gated_false_yes` selected (3 repaired / 0 broken) and re-run on the untouched confirm half: **5 repaired / 0 broken on 52 applicable false-Yes cases** (coverage 88%), effect +0.096, CI +0.019..+0.174, **e=5.33 -> partial, NOT FIXED** (the gate is e >= 20). Overall that is +5/180 = +2.8 pp. |

EXPLORE selection (120 rows, 40 FAIL; selection only, not confirmation evidence):

| tier | candidate | fixed | broken | effect | verdict |
|---|---|---|---|---|---|
| L0 | `aad_silence_contrast` | 6 | 3 | +0.025 | partial |
| L0 | `aad_silence_contrast_gated_false_yes` | 3 | 0 | +0.088 (34 applicable) | partial, **selected** |
| L1 | `skeptical_default_no` | 29 | 26 | +0.025 | partial (just moves the prior) |
| L1 | `evidence_then_answer` | 9 | 42 | -0.275 | regressed |
| L1 | `two_sided_forced_check` | 6 | 5 | +0.008 | partial |
| L2 | `cove_evidence_check` | 14 | 24 | -0.083 | unsafe |
| L2 | `self_refine_antibias` | 21 | 56 | -0.292 | regressed |
| L2 | `debiased_majority_vote` | 4 | 1 | +0.025 | partial |
| L2 | `self_consistency_5` (floor) | 0 | 1 | -0.008 | unsafe |
| L3a | `coded_pipeline` | 0 | 0 | 0 | no_effect (its strict unanimous-override rule never fired) |

Takeaways. (1) The loop found the paper's mechanism on its own (explore + M2)
and the open pool still *selected* the paper method -- but the judge's
prompt-level ideas were mostly harmful on audio (describe/verify scaffolds
-8 to -29 pp), replicating what `mmau_qwen2_audio` saw. (2) AAD at alpha=0.5
on the `Random` negative-sampling slice is real but small here (+2.8 pp
overall, 0 breaks); the e >= 20 certification needs roughly four times that
margin at n=180. (3) Two gaps surfaced: M4 cannot test a directional
hypothesis ("answers Yes on gold-No") because no analyzer emits a per-case
directional-error signal for binary tasks, and the critic rejected that same
hypothesis for "no ground-truth present/absent field in the evidence" -- it
sees the findings JSON, not the case labels explore used. (4) Two VCD
candidates were proposed for this audio batch and `not_executed`; the VCD
gate now requires visual inputs.
