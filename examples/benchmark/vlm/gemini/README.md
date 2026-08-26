# VLM (image + text) × Gemini (Google Gen AI API)

One leaf of [`examples/benchmark`](../../README.md): the closed-weight Gemini
models, called through Google's official `google-genai` SDK (`--backend
gemini`, forced for the family) on every vlm dataset through the shared
[`_common/run.py`](../../_common/run.py). The image is the family's
(`evalvitals-bench-gemini`, stage `gemini` of
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
| `chartqa` | ChartQA/test_human | exact_or_numeric | 256 | HuggingFaceM4/ChartQA (test, human-authored) |
| `spatial457` | Spatial457/L5_6d_spatial | exact_or_numeric | 256 | RyanWW/Spatial457 (L5_6d_spatial) |
| `pope_random` | POPE/coco_random | yes_no | 1000 | AoiDragon/POPE coco_pope_random @08d957b9 + COCO val2014 |
| `pope_popular` | POPE/coco_popular | yes_no | 1000 | AoiDragon/POPE coco_pope_popular @08d957b9 + COCO val2014 |
| `pope_adversarial` | POPE/coco_adversarial | yes_no | 1000 | AoiDragon/POPE coco_pope_adversarial @08d957b9 + COCO val2014 |

Data is frozen once per modality under [`../_data/`](../_data) (`<dataset>/manifest.json`
+ media), shared by all families of this modality; outputs go to
`outputs/<model>/<dataset>[.<tag>]/`.

## Run

```bash
cd examples/benchmark/vlm/gemini
docker compose build                                   # once per family (cached afterwards)
# per-cell smoke: Stage 0 on 8 rows, no judge
EXTRA_ARGS="--baseline-only --limit 8" docker compose run --rm gemini-3.6-flash
# the full chain (M1 -> explore -> M2 -> M3 -> held-out M5 -> M4 -> fix L0..L2), detached
DATASET=chartqa CONCURRENCY=8 docker compose run -d --name vlm-gemini-3.6-flash-chartqa gemini-3.6-flash
docker logs -f vlm-gemini-3.6-flash-chartqa
```

Every service's command pins the judge and coder to Codex Terra (`--judge-provider
codex --judge-model gpt-5.6-terra --judge-effort medium`). Requests retry with
backoff on 429/5xx (`--request-retries`, default 5; `--request-timeout`, default
300 s); `CONCURRENCY` (default 4) is the number of discovery requests in flight —
size it to the account's rate-limit tier.
