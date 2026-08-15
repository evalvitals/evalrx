# qwen2_audio_tcd_mmau

Diagnose → propose → validate loop for Temporal Contrastive Decoding (TCD,
Li et al. 2026, arXiv:2604.15383) against its own benchmark: MMAU test-mini
(Sakshi et al. 2024), on the paper's own hyperparameter anchor model,
`Qwen2-Audio-7B-Instruct` (`paper_method_fidelity("tcd") ==
"native_layer_matched_stability"` for this pair — see
`evalvitals/models/backends/hf_local.py`).

This is the paper-method arc used elsewhere in this repo
(`examples/vlm_paper_benchmark/run_hf_autofix.py`), narrowed to one method /
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
  threshold (see `examples/vlm_paper_benchmark/experiments_s_tier_2026-08.md`
  for what that looks like on a comparable paper-method run); a small run can
  legitimately land on "correct direction, not yet certified" rather than a
  clean win, and that is a valid outcome to report, not a bug to chase away
  by re-splitting the data. Raise `--limit`/`--selection-cases` (up to 1000
  total, after `download_mmau.py --limit 1000 --scan-rows 1000`) for a
  better-powered run.
- MMAU clips over Qwen2-Audio's 30s encoder window are skipped at download
  time (`download_mmau.py`'s `MAX_DURATION_SEC`), with the skip count printed
  — never silently truncated mid-run.
