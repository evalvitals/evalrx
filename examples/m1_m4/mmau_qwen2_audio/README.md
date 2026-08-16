# mmau_qwen2_audio

Full M1→M4 diagnosis loop for `Qwen2-Audio-7B-Instruct` on MMAU test-mini
(Sakshi et al. 2024): `VLDiagnoseLoop` discovers its own hypothesis about why
the model fails audio multiple-choice questions, then `FixAgent` proposes and
validates a repair — including the paper-registered `tcd_temporal_blur`
candidate for Temporal Contrastive Decoding (TCD, Li et al. 2026,
arXiv:2604.15383) on the paper's own hyperparameter anchor model
(`paper_method_fidelity("tcd") == "native_layer_matched_stability"` for this
pair — see `evalvitals/models/backends/hf_local.py`).

This was `examples/m4/qwen2_audio_tcd_mmau`, a `FixAgent`-only run against a
hand-supplied hypothesis. TCD's own framing ("temporal smoothing bias") is no
longer wired in anywhere — the `ExperimentProtocol` below only describes the
task and the measured baseline accuracy; whatever mechanism M3 proposes has
to come out of the loop's own M1/M2 evidence.

## Run

```bash
docker compose up --build
```

GPU selection: `CUDA_VISIBLE_DEVICES=<idx> docker compose up --build` (defaults
to GPU 0). Needs a `claude` CLI on the host (mount path via `$CLAUDE_PATH`,
see `docker-compose.yml`) — it's the M2/M3/M5 judge and FixAgent's L1/L2
proposer, no API key needed (reuses your local OAuth session).

```bash
python download_mmau.py --limit 120
python run.py --model qwen2-audio-7b-instruct --limit 120
python run.py --smoke-test   # wiring check, no GPU/model/judge needed — see below
```

## Pipeline

```
download_mmau.py            frozen manifest + decoded .wav clips (once)
        │
fresh baseline pass          generate_tcd_baseline (forced greedy) on every
                              row, every run -- never cached
        │
ExperimentProtocol            OBSERVATION ONLY: task description + measured
                               baseline accuracy, no mechanism named
        │
VLDiagnoseLoop M1→M5
  M1  ProbeAgent               PINNED static analyzer set (see run.py's
                                PINNED_M1_ANALYZERS) -- not LLM-guided catalog
                                selection; see "Why M1 is pinned" below
  M2  StatsAnalysisAgent       e-BH FDR-corrected stats + Claude-written
                                evidence chain
  M3  DiagnosisAgent           Claude judge proposes hypotheses from M1+M2
  M5  HypothesisTester         statistical test + protocol-consistency check
        │
loop.run_fix                  FixAgent's tiered candidates (default: only
                               tcd_temporal_blur is admitted -- see
                               --unrestricted below), validated on a held-out
                               CONFIRM split the loop's own M1-M5 discovery
                               never saw (VLDiagnoseLoop's confirm_split)
```

Outputs land under `--run-dir` (default `./outputs/`) via the shared
`RunContext` — `run_log.jsonl`, `artifacts/`, `prompts/`, `experiments/` —
same layout as every other `m1_m4/` example.

## Why M1 is pinned, not LLM-guided

`ProbeAgent(judge=<claude>, protocol=...)` would let the judge pick from the
*full* analyzer catalog. Most catalog analyzers declare
`applies_to_modalities={"text","image"}` — since the modality gate is an OR
and every model (including this audio one) has `"text"`, that gate does
**not** actually filter the catalog down to analyzers verified safe for
audio. Auditing every one of ~25 catalog analyzers against a live audio
handle for the first time was out of scope here, so `run.py` pins M1 to
`PINNED_M1_ANALYZERS` instead (`ProbeAgent(judge=None, probe=StrategyProbe(
priority_override=...))` — bypasses LLM-guided selection entirely):

- all eight are read-verified to preserve non-image `Inputs` fields
  (`dataclasses.replace(case.inputs, ...)` or `case.inputs` unmodified) —
  `prompt_contrast` needed exactly this fix
  (`evalvitals/analyzers/perturbation/prompt_contrast.py`) before it could be
  trusted on an audio case: it used to rebuild a bare `Inputs(prompt=...,
  image=...)` for every re-ask, silently dropping `.audio`, so the "prompt
  repairable?" re-ask would have answered the question **without hearing the
  clip**. Fixed and covered by
  `tests/test_analyzers/test_prompt_contrast.py::test_audio_input_preserved_across_strategies`.
- none of the eight touch vision-specific internals (attention maps,
  logit-lens, CKA, mm-shap — all excluded, untested against an audio tower).
- `format_sensitivity` is *built* for this exact task shape (4-way MC
  option-order bias); the rest are black-box output/logprob diagnostics that
  don't care what the input modality was.
- `prompt_contrast` and `cot_faithfulness` stay excluded even after the fix
  above: their *default* strategy templates are image-phrased ("describe
  what you see in the relevant region of the image…"), which is not just
  irrelevant but actively confusing prompt content for an audio model — see
  the measured "generic fallback candidates are actively harmful on an audio
  task" result below.

Raising M1 to full LLM-guided selection (auditing the rest of the catalog,
or writing audio-native analyzer variants) is the natural next step, not
done here.

## Wiring verification

`python run.py --smoke-test` builds a synthetic model + 4 synthetic cases and
runs the **real** `VLDiagnoseLoop` + `FixAgent` code paths (stub M1 probe and
M3 diagnosis, everything else real) — specifically to catch
`tcd_temporal_blur`'s admission gate (`fix_agent.py`'s `_l3_candidates`)
silently never firing, which would otherwise complete a whole run and produce
nothing, with no error anywhere. Passing, as of this rewrite:

```
fix candidates attempted: ['tcd_temporal_blur']
tcd_temporal_blur: fixed=2 broken=0 effect=+0.500
Smoke test passed.
```

This confirms the wiring is correct. It does **not** substitute for a real
GPU run against Qwen2-Audio-7B-Instruct and a real `claude` judge — that
takes GPU + CLI access this rewrite has not yet exercised end-to-end; run
`docker compose up` to do that.

## What this does and does not claim

- The baseline arm calls `generate_tcd_baseline` (forced `do_sample=False`),
  not the bare `generate()` — Qwen2-Audio-7B-Instruct's own
  `generation_config` defaults to sampling, and TCD's candidate is greedy by
  construction, so a sampled baseline would not be a valid paired comparison.
- `FixAgent`'s admission gate for `tcd_temporal_blur` requires
  `task == "multiple_choice"` (exact set equality over every case's
  `metadata["task"]` — `run.py` asserts this right after building the case
  batch, since a silent metadata drift would otherwise mean the gate never
  fires and the run completes having tested nothing) and
  `paper_method_fidelity("tcd")` to be `"native_layer_matched_stability"` (or
  `"adapted_..."` with `--allow-adapted-paper-methods`).
- `--limit 120` (default) is a small proof run, not test-mini's full 1000
  rows — this establishes the loop runs correctly end to end on real data,
  not a certified accuracy claim. `FixAgent`'s e-value martingale needs
  roughly 25+ discordant pairs at a 4:1 ratio to clear its e≥20 certification
  threshold (see `examples/m4/vlm_paper_benchmark/experiments_s_tier_2026-08.md`
  for what that looks like on a comparable paper-method run); a small run can
  legitimately land on "correct direction, not yet certified" rather than a
  clean win, and that is a valid outcome to report, not a bug to chase away
  by re-splitting the data. Raise `--limit` (up to 1000, after
  `download_mmau.py --limit 1000 --scan-rows 1000`) for a better-powered run.
- MMAU clips over Qwen2-Audio's 30s encoder window are skipped at download
  time (`download_mmau.py`'s `MAX_DURATION_SEC`), with the skip count printed
  — never silently truncated mid-run.

## `--unrestricted`: what does the agent propose on its own?

By default `run.py` pins the fix candidate pool to `tcd_temporal_blur`
(`candidate_allowlist=["tcd_temporal_blur"]`) — TCD is proposed the same way
OPERA/VCD/ICD/PAI/IFCD already are elsewhere in this repo: an unconditional,
structurally-gated Python default in `fix_agent.py::_l3_candidates`, not
something an LLM invents at runtime. `--unrestricted` drops that allowlist so
every admissible candidate — paper defaults AND whatever the Claude judge
proposes at L1/L2 — competes on equal footing.

### Prior result (old FixAgent-only script, hand-supplied hypothesis)

Measured on a 120-row / 32-selection-case split with `--judge --unrestricted`
on the *old* `m4/qwen2_audio_tcd_mmau` script — kept here as the reference
point for "generic fallback candidates on an audio task", not reproducible
byte-for-byte with this rewrite (different M1/M4 split, different hypothesis
source):

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
e≥20 at this sample size either way. This is exactly why `prompt_contrast`
and `cot_faithfulness` (whose default templates are the same vision-era
generic style) are excluded from `PINNED_M1_ANALYZERS` above rather than
included and hoped-for.

## Case study: listen to the clips, answer them yourself

```bash
pip install -e ".[dashboard]"
evalvitals dashboard examples/m1_m4/mmau_qwen2_audio
```

The dashboard reads every `outputs/*.json` here and re-joins case ids to
`data/mmau_test_mini.jsonl`, so each case shows up with its `.wav` in an
audio player, its question and its four options. Requires `./data` to be
populated — playback needs the audio files `download_mmau.py` fetched.
