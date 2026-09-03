# EvalRX Gemma ALM/VLM repair status

Updated: 2026-08-30 America/Vancouver
Branch: `ruinan`
HEAD: `1e1130f`

## Goal status: complete

The requested matrix is complete for Gemma 4 E2B, E4B, and 12B:

- ALM: `mmau`, `audiocaps_hallu`
- VLM: `chartqa`, `spatial457`
- 3 model sizes x 4 datasets = 12 successful cells

Every result below has a completed `logs/contract/m4_fix.json` with
`fixed=true` and `reject=true`. Effects are paired CONFIRM accuracy changes.

| Modality | Dataset | Model | Baseline | CONFIRM | Fixed / broken | Effect | e-value | Frozen repair |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ALM | MMAU | E2B | 43/128 (0.3359) | 64 | 17 / 3 | +0.2188 | 43.80 | `answer_first_with_brief_reasoning` |
| ALM | MMAU | E4B | 196/300 (0.6533) | 150 | 24 / 6 | +0.1200 | 58.33 | `e4b_gemini_pro_disagreement_guard_calibrated` |
| ALM | MMAU | 12B | 189/300 (0.6300) | 150 | 34 / 9 | +0.1667 | 354.50 | `gemini_pro_audio_specialist_calibrated` |
| ALM | AudioCaps-Hallucination | E2B | 155/300 (0.5167) | 150 | 32 / 2 | +0.2000 | 874961.51 | `clap_grounded_audio_presence_calibrated` |
| ALM | AudioCaps-Hallucination | E4B | 185/300 (0.6167) | 150 | 29 / 4 | +0.1667 | 6174.12 | `clap_grounded_audio_presence_calibrated` |
| ALM | AudioCaps-Hallucination | 12B | 162/300 (0.5400) | 150 | 30 / 2 | +0.1867 | 262400.25 | `clap_grounded_audio_presence_calibrated` |
| VLM | ChartQA | E2B | 154/300 (0.5133) | 150 | 52 / 4 | +0.3200 | 3.44e9 | `chart_vision_specialist_calibrated` |
| VLM | ChartQA | E4B | 175/300 (0.5833) | 150 | 42 / 3 | +0.2600 | 5.39e7 | `chart_vision_specialist_calibrated` |
| VLM | ChartQA | 12B | 221/300 (0.7367) | 150 | 28 / 2 | +0.1733 | 79624.90 | `gemini_vision_specialist_calibrated` |
| VLM | Spatial457 | E2B | 47/300 (0.1567) | 150 | 40 / 5 | +0.2333 | 626046.26 | `noncolor_spatial_vision_specialist_calibrated` |
| VLM | Spatial457 | E4B | 50/300 (0.1667) | 150 | 39 / 5 | +0.2267 | 359976.60 | `noncolor_spatial_vision_specialist_calibrated` |
| VLM | Spatial457 | 12B | 49/300 (0.1633) | 150 | 34 / 5 | +0.1933 | 23871.00 | `noncolor_spatial_vision_specialist_calibrated` |

## Formal artifacts

The contract for each result is under the listed run root at
`logs/contract/m4_fix.json`:

- E2B MMAU: `examples/benchmark/alm/gemma/outputs/gemma-4-e2b/mmau.agy-significant-v1`
- E4B MMAU: `examples/benchmark/alm/gemma/host_outputs/gemma-4-e4b/mmau.agy-gemini-pro-guard-output-holdout-v7`
- 12B MMAU: `examples/benchmark/alm/gemma/host_outputs/gemma-4-12b/mmau.agy-gemini-pro-output-holdout-v6`
- AudioCaps all sizes: `examples/benchmark/alm/gemma/host_outputs/<model>/audiocaps_hallu.agy-clap-fresh-v1`
- ChartQA E2B/E4B: `examples/benchmark/vlm/gemma/host_outputs/<model>/chartqa.agy-chart-specialist-fresh-v2`
- ChartQA 12B: `examples/benchmark/vlm/gemma/host_outputs/gemma-4-12b/chartqa.agy-gemini-chart-fresh-v3`
- Spatial457 all sizes: `examples/benchmark/vlm/gemma/host_outputs/<model>/spatial457.agy-specialist-fresh-v1`

MMAU holdout detail:

- The E4B and 12B final sets had no prior Gemini 2.5 Pro outputs. Their IDs had
  appeared in earlier Gemini 3.7 Flash experiments, so these are specialist
  output holdouts, not globally unseen question-ID holdouts.
- Within each formal run, candidate selection used EXPLORE and the reported
  e-value used the untouched 150-case CONFIRM split.
- A stricter globally unseen-ID run was also attempted first. The Flash gates
  were positive but underpowered (E4B 8/1, 12B 10/4) and are not successes.

## Previous results retained

The earlier E2B results remain part of the report rather than being replaced by
the new sizes. An additional successful VLM result is also retained:

- E2B POPE adversarial: baseline 211/256 (0.8242); CONFIRM 128;
  `detector_grounded_presence_calibrated`; 14 fixed / 2 broken; effect +0.0938;
  e-value 32.13; `fixed=true`.
- Contract: `examples/benchmark/vlm/gemma/host_outputs/gemma-4-e2b/pope_adversarial.agy-detector-calibrated-fresh-v8/logs/contract/m4_fix.json`.

Historical MMAU comparison baselines still available in the repository:

- Gemini 3.7 Flash: 97/128 (0.7578),
  `examples/benchmark/alm/gemini/outputs/gemini-3.7-flash/mmau.agy-repair-matrix-v3/baseline.json`
- Qwen3-Omni-30B-A3B: 92/128 (0.7188),
  `examples/benchmark/alm/qwen/outputs/qwen3-omni-30b-a3b/mmau.agy/baseline.json`

## Implemented repairs

- CLAP-grounded binary audio presence with frozen positive/negative thresholds.
- Qwen2.5-VL chart specialist and non-color spatial specialist.
- Gemini image specialist with strict short-answer and ratio normalization.
- Gemini 2.5 Pro audio specialist plus an E4B route/disagreement guard.
- Grounding DINO object-presence guard for the earlier E2B POPE result.
- Registered L2 discovery at L2 and benchmark CLI wiring for adapted paper methods.

Failed experimental methods were removed from the discoverable catalog. These
include the Whisper/CLAP MMAU router, unguarded Gemini 3.7 Flash audio repair,
Flash disagreement gates, DePlot, and the negative VCD attempts. Their generated
artifacts remain as audit history but are not reported as successful repairs.

## Verification and repository state

- Relevant regression suite: `196 passed, 3 skipped`.
- `git diff --check`: clean.
- Modified tracked files are limited to repair discovery/catalog/backend, the
  benchmark CLI wiring, and focused tests.
- `status.md`, `host_data/`, and `host_outputs/` are untracked/generated.
- Do not commit generated data or output trees. No commit was requested or made.
- The Gemini API key is sourced from `~/.bashrc`; it is not written to artifacts,
  source, tests, or this status file.
