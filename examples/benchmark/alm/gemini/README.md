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
| `gemini-3.1-pro-preview` | `gemini-3.1-pro-preview` | low (cannot be disabled) |
| `gemini-2.5-flash` | `gemini-2.5-flash` | budget 0 |
| `gemini-2.5-flash-lite` | `gemini-2.5-flash-lite` | budget 0 |
| `gemini-2.5-pro` | `gemini-2.5-pro` | budget 128 (cannot be disabled) |

`--thinking-level {minimal,low,medium,high}` / `--thinking-budget N` override
the floor; `--enable-thinking` leaves the API default.

The benchmark automatically probes probability support with a short real request.
Only valid token log probabilities enable `LOGPROBS` analyzers; five top
alternatives are requested when supported. Response evidence and effective
capabilities are saved in `api_capabilities.json`. Authentication, quota and
transport failures remain errors rather than being treated as missing capabilities.

Raw logits and model internals are unavailable. Top-k logprobs cannot be
substituted for logits. Without logprobs, probability-only analyzers are
skipped and calibration can use verbalized confidence. M1–M5 and the L0–L2
repair search still operate on API generation. The repair ceiling is clamped
to L2; CONFIRM evaluates the frozen candidate separately from EXPLORE.
Running all stages does not imply a statistically confirmed improvement.
The served `model_version` is written to `baseline.json`.

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

The default judge/coder is Claude. To run the full chain with the mounted
Antigravity CLI, explicitly set both the model and its required effort:

```bash
EXTRA_ARGS="--limit 32 --run-tag gemini-m1m5 --judge-provider agy --judge-model gemini-3.6-flash --judge-effort low --fix-tier L2" \
  docker compose run --rm gemini-3.6-flash
```

This runs M1–M5 plus the repair search; omit `--baseline-only`, `--skip-fix`
and `--skip-m5` when validating the whole chain. Use a freshly built image
after updating source. Inspect `summary.json` and stage logs for actual
completion and the repair verdict. A 32-case run checks integration; it is
not a powered benchmark for small gains.

Requests retry with backoff on 429/5xx (`--request-retries`, default 5;
`--request-timeout`, default 300 s). `CONCURRENCY` (default 4) controls the
number of discovery requests in flight.

Adaptive probe selection and bounded controller decisions are the default; no mode flag is needed. See [adaptive benchmark diagnosis](../../../../docs/agentic_benchmark.md) for the confirmation boundary, failure handling, and protocol differences.
