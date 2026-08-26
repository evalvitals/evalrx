# LLM (text only) × Gemini (Google Gen AI API)

One leaf of [`examples/benchmark`](../../README.md): the closed-weight Gemini
models, called through Google's official `google-genai` SDK (`--backend
gemini`, forced for the family) on every llm dataset through the shared
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
| `cruxeval_output` | cruxeval_output | llm_graded | 256 | examples/dataset_selection/llm_benchmark/datasets.py |
| `bbh_causal_judgement` | bbh_causal_judgement | llm_graded | 256 | examples/dataset_selection/llm_benchmark/datasets.py |
| `supergpqa_economics` | supergpqa_economics | llm_graded | 256 | examples/dataset_selection/llm_benchmark/datasets.py |
| `bbh_word_sorting` | bbh_word_sorting | llm_graded | 256 | examples/dataset_selection/llm_benchmark/datasets.py |
| `bbh_tracking7` | bbh_tracking7 | llm_graded | 256 | examples/dataset_selection/llm_benchmark/datasets.py |
| `bamboogle` | bamboogle | llm_graded | 256 | examples/dataset_selection/llm_benchmark/datasets.py |
| `minervamath` | minervamath | llm_graded | 256 | examples/dataset_selection/llm_benchmark/datasets.py |
| `supergpqa_law` | supergpqa_law | llm_graded | 256 | examples/dataset_selection/llm_benchmark/datasets.py |
| `supergpqa_medicine_hard` | supergpqa_medicine_hard | llm_graded | 256 | examples/dataset_selection/llm_benchmark/datasets.py |
| `hotpotqa_gepa` | GEPA test split (fullwiki/train, seed 1) + the 10-paragraph distractor context | short_answer_em (SQuAD EM) | 300 | hotpotqa/hotpot_qa, arXiv:2507.19457 |
| `gsm8k` | seeded 500-of-1,319 sample of the test split | exact_or_numeric (0 tolerance) | 500 | openai/gsm8k |

Data is frozen once per modality under [`../_data/`](../_data) (`<dataset>/manifest.json`
+ media), shared by all families of this modality; outputs go to
`outputs/<model>/<dataset>[.<tag>]/`.

## Run

```bash
cd examples/benchmark/llm/gemini
docker compose build                                   # once per family (cached afterwards)
# per-cell smoke: Stage 0 on 8 rows, no judge
EXTRA_ARGS="--baseline-only --limit 8" docker compose run --rm gemini-3.6-flash
# the full chain (M1 -> explore -> M2 -> M3 -> held-out M5 -> M4 -> fix L0..L2), detached
DATASET=bbh_causal_judgement CONCURRENCY=8 docker compose run -d --name llm-gemini-3.6-flash-bbh_causal_judgement gemini-3.6-flash
docker logs -f llm-gemini-3.6-flash-bbh_causal_judgement
```

Every service's command pins the judge and coder to Codex Terra (`--judge-provider
codex --judge-model gpt-5.6-terra --judge-effort medium`). Requests retry with
backoff on 429/5xx (`--request-retries`, default 5; `--request-timeout`, default
300 s); `CONCURRENCY` (default 4) is the number of discovery requests in flight —
size it to the account's rate-limit tier.
