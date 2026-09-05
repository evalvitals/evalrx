# ALM (audio + text) × Gemini (Google Gen AI API)

One leaf of [`examples/benchmark`](../../README.md): the closed-weight Gemini
models, called through Google's official `google-genai` SDK (`--backend
gemini`, forced for the family) on every alm dataset through the shared
[`_common/run.py`](../../_common/run.py). The image is the family's
(`evalrx-bench-gemini`, stage `gemini` of
[`docker/Dockerfile`](../../docker/Dockerfile): the base stack plus
`google-genai`); **no GPU is reserved** — the service extends `bench-core`, not
`bench`. The API key is `GEMINI_API_KEY` in `examples/benchmark/.env` (this
leaf's `.env` links to it).

## Models (services)

Every model id is its own api-only spec; each takes text, image, video and
audio, so the same service names exist in the llm / vlm / alm leaves.

| service / `--model` | spec | thinking floor the runtime sends |
|---|---|---|
| `gemini-3.7-flash` | `gemini-3.7-flash` | low |
| `gemini-3.6-flash` | `gemini-3.6-flash` | minimal |
| `gemini-3.5-flash` | `gemini-3.5-flash` | minimal |
| `gemini-3.5-flash-lite` | `gemini-3.5-flash-lite` | minimal |
| `gemini-3.1-flash-lite` | `gemini-3.1-flash-lite` | minimal |
| `gemini-2.5-flash` | `gemini-2.5-flash` | budget 0 |
| `gemini-2.5-flash-lite` | `gemini-2.5-flash-lite` | budget 0 |
| `gemini-2.5-pro` | `gemini-2.5-pro` | budget 128 (cannot be disabled) |

`--thinking-level {minimal,low,medium,high}` / `--thinking-budget N` override
the floor; `--enable-thinking` leaves the API default. What the family cannot
do: expose internals or logprobs (Gemini returns none for 3.x), so the model
claims `GENERATE` only — `calibration` runs on its verbalized channel,
`logprob_entropy` is skipped — and the fix ladder is **clamped to L2**
(`--fix-tier L3a/L3b` prints the clamp; `summary.json` records the effective
`fix_tier`). The served `model_version` is written to `baseline.json`.

## Datasets (`DATASET=`)

| name | slice | scoring | default rows | source |
|---|---|---|---|---|
| `mmau` | MMAU/test-mini | multiple_choice_letter | 256 | gamma-lab-umd/MMAU-test-mini |
| `mmsu` | MMSU (47 spoken-language tasks) | multiple_choice_letter | 256 | ddwang2000/MMSU |
| `audiocaps_hallu` | AudioCaps-Hallucination/Random | yes_no | 300 | kuanhuggingface/AudioHallucination_AudioCaps-Random + OpenSound/AudioCaps |
| `af_reasoning_mcq` | AF-Reasoning-Eval/AQA-MCQ | multiple_choice_letter | 76 | NVIDIA/audio-flamingo (AQA_MCQ) + gijs/clothoaqa (audio) |

Data is frozen once per modality under [`../_data/`](../_data) (`<dataset>/manifest.json`
+ media), shared by all families of this modality; outputs go to
`outputs/<model>/<dataset>[.<tag>]/`.

## Run

```bash
cd examples/benchmark/alm/gemini
docker compose build                                   # once per family (cached afterwards)
# per-cell smoke: Stage 0 on 8 rows, no judge
EXTRA_ARGS="--baseline-only --limit 8" docker compose run --rm gemini-3.6-flash
# the full chain (M1 -> explore -> M2 -> M3 -> held-out M4 -> M5 -> fix L0..L2), detached
DATASET=mmau CONCURRENCY=8 docker compose run -d --name alm-gemini-3.6-flash-mmau gemini-3.6-flash
docker logs -f alm-gemini-3.6-flash-mmau
```

Every service's command pins the judge and coder to Codex Terra (`--judge-provider
codex --judge-model gpt-5.6-terra --judge-effort medium`). Requests retry with
backoff on 429/5xx (`--request-retries`, default 5; `--request-timeout`, default
300 s); `CONCURRENCY` (default 4) is the number of discovery requests in flight —
size it to the account's rate-limit tier.
