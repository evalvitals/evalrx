# qwen2_audio_tcd_mmau

Diagnose → propose → validate loop for Temporal Contrastive Decoding (TCD,
Li et al. 2026, arXiv:2604.15383) against its own benchmark: MMAU test-mini
(Sakshi et al. 2024), on the paper's own hyperparameter anchor model,
`Qwen2-Audio-7B-Instruct` (`paper_method_fidelity("tcd") ==
"native_layer_matched_stability"` for this pair — see
`evalvitals/models/backends/hf_local.py`).

This is the paper-method arc used elsewhere in this repo
(`examples/m4/vlm_paper_benchmark/run_hf_autofix.py`), narrowed to one method /
one benchmark: a frozen baseline probe, a diagnosis/selection split,
`FixAgent.propose_and_validate` proposing (only) `tcd_temporal_blur`, and
`validate_candidate` on a held-out confirmation split.

## Run

```bash
docker compose up --build
```

GPU selection: `CUDA_VISIBLE_DEVICES=<idx> docker compose up --build` (defaults
to GPU 0 — override if that's occupied). Runs `download_mmau.py` then
`run.py` in one container; both read/write under `./data` and
`./outputs`, mounted from the host so a re-run doesn't re-download.

To run the two steps independently, override the compose `command`, or run
outside Docker (after `pip install -e ".[local,data]"` and installing
`ffmpeg` on PATH):

```bash
python download_mmau.py --limit 120
python run.py --model qwen2-audio-7b-instruct --limit 120
```

## Case study: listen to the clips, answer them yourself

```bash
pip install -e ".[dashboard]"
evalvitals dashboard examples/agent_loop/qwen2_audio_tcd_mmau
```

The dashboard reads every `outputs/*.json` here and re-joins its case ids to
`data/mmau_test_mini.jsonl` (via the `dataset` pointer `run.py` writes), so each
case shows up with its `.wav` in an audio player, its question and its four
options. Blind mode is on by default: answer the item yourself first, then
unblind to see the correct answer, what Qwen2-Audio answered, and — per repair
candidate — whether it repaired, broke or left that case alone. The *Repair
Methods* tab spells out what each candidate actually changes (`tcd_temporal_blur`'s
blurred-waveform contrast vs. an L1 prompt template vs. an L2 scaffold) and
links each flipped case back into the case browser. Requires `./data` to be
populated — playback needs the audio files `download_mmau.py` fetched.

## What this does and does not claim

- The baseline arm calls `generate_tcd_baseline` (forced `do_sample=False`),
  not the bare `generate()` — Qwen2-Audio-7B-Instruct's own
  `generation_config` defaults to sampling, and TCD's candidate is greedy by
  construction, so a sampled baseline would not be a valid paired comparison.
- `FixAgent`'s admission gate for `tcd_temporal_blur` requires
  `task == "multiple_choice"` and `paper_method_fidelity("tcd")` to be
  `"native_layer_matched_stability"` (or `"adapted_..."` with
  `--allow-adapted-paper-methods`) — see `fix_agent.py`'s `_l3_candidates`.
- `--limit 120` (default) is a small proof run, not test-mini's full 1000
  rows — this establishes the loop runs correctly end to end on real data,
  not a certified accuracy claim. `FixAgent`'s e-value martingale needs
  roughly 25+ discordant pairs at a 4:1 ratio to clear its e≥20 certification
  threshold (see `examples/m4/vlm_paper_benchmark/experiments_s_tier_2026-08.md`
  for what that looks like on a comparable paper-method run); a small run can
  legitimately land on "correct direction, not yet certified" rather than a
  clean win, and that is a valid outcome to report, not a bug to chase away
  by re-splitting the data. Raise `--limit`/`--selection-cases` (up to 1000
  total, after `download_mmau.py --limit 1000 --scan-rows 1000`) for a
  better-powered run.
- MMAU clips over Qwen2-Audio's 30s encoder window are skipped at download
  time (`download_mmau.py`'s `MAX_DURATION_SEC`), with the skip count printed
  — never silently truncated mid-run.

## `--judge --unrestricted`: what does the agent propose on its own?

By default this script disables the LLM-judge proposal path (`judge=None`,
so `FixAgent._ask_judge()` is a no-op) and pins the candidate pool to
`tcd_temporal_blur` (`candidate_allowlist=["tcd_temporal_blur"]`) — TCD is
proposed the same way OPERA/VCD/ICD/PAI/IFCD already are elsewhere in this
repo: an unconditional, structurally-gated Python default in
`fix_agent.py::_l3_candidates`, not something an LLM invents at runtime.

`--judge --unrestricted` turns both restrictions off, wiring the model under
test itself as its own judge (`_JudgeModel`, 900-token decode budget for L1/L2
proposal calls) and letting every admissible candidate — paper defaults AND
whatever the judge proposes — compete on equal footing. `--allow-codegen` is
NOT exposed here (kept off) since a coding backend is out of scope for what
this is testing.

Measured result on the same 120-row / 32-selection-case split as the default
run above: the judge's own JSON proposal for L1 and L2 failed to parse
(`FixAgent: unparseable judge proposal; using defaults` — matches the exact
failure mode `vlm_paper_benchmark/run_hf_autofix.py` already documented for a
7B judge), so what actually competed was `FixAgent`'s generic FALLBACK
defaults (image-oriented prompt/scaffold templates, since they predate any
audio-capable spec) against `tcd_temporal_blur`:

| tier | candidate | n_pairs | fixed | broken | effect | e | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| L1 | `attend_carefully` (generic default, mentions "the image") | 32 | 1 | 11 | −31.2% | 26.3 | **regressed** |
| L2 | `self_refine` (generic multi-call default) | 32 | 2 | 15 | −40.6% | 53.5 | **regressed** |
| L2 | `least_to_most` (generic multi-call default) | 32 | 1 | 16 | −46.9% | 428.3 | **regressed** |
| L3a | `tcd_temporal_blur` (paper default) | 32 | 3 | 0 | +9.4% | 2.0 | partial |

Every generic candidate was actively harmful once let loose on an audio task
they weren't written for (unsurprising — they're vision-era fallbacks, not
audio-aware); `tcd_temporal_blur` was the only one that was safe and
positive. `best` was still `None` — TCD wins the field but doesn't clear
e≥20 at this sample size either way. Reproduce with:

```bash
docker compose run --rm qwen2_audio_tcd_mmau sh -c \
  "python download_mmau.py --limit 120 && \
   python run.py --model qwen2-audio-7b-instruct --limit 120 \
     --judge --unrestricted --output-name tcd_mmau_open"
```
