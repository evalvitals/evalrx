# VLM (image + text) × Qwen

One leaf of [`examples/benchmark`](../../README.md): the Qwen sizes that
take images as input, run on every vlm dataset through the shared
[`_common/run.py`](../../_common/run.py). The image is the family's
(`evalvitals-bench-qwen`, stage `qwen` of
[`docker/Dockerfile`](../../docker/Dockerfile)); size and dataset are runtime
arguments, never a rebuild.

## Sizes (services)

| service / `--model` | hf_local spec (default) | endpoint spec | GPUs | note |
|---|---|---|---|---|
| `qwen3.5-2b` | `qwen3.5-2b-vl` | — | 1 | same checkpoint: text spec loads the language tower only, -vl adds the vision tower |
| `qwen3.5-4b` | `qwen3.5-4b-vl` | — | 1 | same checkpoint: text spec loads the language tower only, -vl adds the vision tower |
| `qwen3.5-9b` | `qwen3.5-9b-vl` | — | 1 | same checkpoint: text spec loads the language tower only, -vl adds the vision tower |

## Datasets (`DATASET=`)

| name | slice | scoring | default rows | source |
|---|---|---|---|---|
| `chartqa` | ChartQA/test_human | exact_or_numeric | 256 | HuggingFaceM4/ChartQA (test, human-authored) |
| `spatial457` | Spatial457/L5_6d_spatial | exact_or_numeric | 256 | RyanWW/Spatial457 (L5_6d_spatial) |

Data is frozen once per modality under [`../_data/`](../_data) (`<dataset>/manifest.json`
+ media), shared by all families of this modality; outputs go to
`outputs/<model>/<dataset>[.<tag>]/`.

## Run

```bash
cd examples/benchmark/vlm/qwen
docker compose build                                   # once per family (cached afterwards)
# per-cell smoke: load + Stage 0 on 8 rows, no judge
EXTRA_ARGS="--baseline-only --limit 8" docker compose run --rm qwen3.5-2b
# the full chain (M1 -> explore -> M2 -> M3 -> held-out M5 -> M4 -> fix), detached
DATASET=chartqa CUDA_VISIBLE_DEVICES=0 docker compose run -d --name vlm-qwen3.5-2b-chartqa qwen3.5-2b
docker logs -f vlm-qwen3.5-2b-chartqa
```

Every size's command pins the judge and coder to Claude Opus 5 (`--judge-provider
claude --judge-model claude-opus-5 --judge-effort high`); thinking is OFF on every
model (`--enable-thinking` turns it on for one run); the model runs in-process
(`hf_local`; `EXTRA_ARGS="--backend endpoint --base-url http://host.docker.internal:8020/v1"`
talks to a vLLM server instead).
