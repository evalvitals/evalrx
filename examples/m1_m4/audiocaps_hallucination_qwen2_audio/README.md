# audiocaps_hallucination_qwen2_audio

Full M1→M4 diagnosis loop: does the agent independently discover that
Qwen2-Audio-7B-Instruct over-affirms sounds that aren't in the clip
("language priors override audio evidence"), and reach for AAD
(Audio-Aware Decoding, Hsu et al. 2025, arXiv:2506.07233) to fix it — the
same way it already reaches for TCD on `examples/m1_m4/mmau_qwen2_audio`?

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

300 rows split 120 explore (M1-M5 discovery, fix *selection*) / 180 confirm
(held-out M5 + fix *validation*) by `--confirm-split 0.6`; the old 120-row
default left only 48 discovery cases, too few for M2/M5 or the fix gate.

The fix pool is open by default (`candidate_allowlist=None`, `--fix-max-tier L3a`
like `mmau_qwen2_audio`): the judge's L1/L2 candidates, the `self_consistency_5`
floor, a coded L2 pipeline and -- when its admission gate fires -- AAD all
compete, selected on the explore half and confirmed on the untouched confirm
half. `--paper-method-only` restores the AAD-only pool with no coder; note that
`--fix-max-tier L0` would shrink the pool back to AAD as well, since every other
family is tiered L1/L2.

Or via Docker (see `docker-compose.yml` — mirrors `mmau_qwen2_audio`'s setup
exactly, same base model, same judge/audio volume mounts):

```bash
docker compose up
```

## What "fixed" means here

`fix_agent.py`'s admission gate for AAD (`aad_silence_contrast` /
`aad_silence_contrast_gated_false_yes`, both L0 — no internals read needed,
just two `generate()`-compatible forward passes) requires
`case.metadata["task"] == "yes_no"`, `model.generate_aad` to exist, and
`paper_method_fidelity("aad") == "native_silence_contrast"`. Like TCD on
`mmau_qwen2_audio`, this run.py never names AAD or its mechanism anywhere
in the protocol or dataset glue — if it gets proposed and validated, that
has to come from the loop's own M1→M5 discovery and the judge's own
mechanism-match against whatever hypothesis M3 produces, not a hand-supplied
hint.
