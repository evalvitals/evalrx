# VLM (image + text) × Gemma 4

One leaf of [`examples/benchmark`](../../README.md): the Gemma 4 sizes that
take images as input, run on every vlm dataset through the shared
[`_common/run.py`](../../_common/run.py). The image is the family's
(`evalvitals-bench-gemma`, stage `gemma` of
[`docker/Dockerfile`](../../docker/Dockerfile)); size and dataset are runtime
arguments, never a rebuild.

## Sizes (services)

| service / `--model` | hf_local spec (default) | endpoint spec | GPUs | note |
|---|---|---|---|---|
| `gemma-4-e2b` | `gemma-4-e2b-it` | — | 1 | natively text+image+audio; one spec serves every modality |
| `gemma-4-e4b` | `gemma-4-e4b-it` | — | 1 | natively text+image+audio; one spec serves every modality |
| `gemma-4-12b` | `gemma-4-12b-it` | — | 1 | natively text+image+audio; one spec serves every modality |

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
cd examples/benchmark/vlm/gemma
docker compose build                                   # once per family (cached afterwards)
# per-cell smoke: load + Stage 0 on 8 rows, no judge
EXTRA_ARGS="--baseline-only --limit 8" docker compose run --rm gemma-4-e2b
# the full chain (M1 -> explore -> M2 -> M3 -> held-out M5 -> M4 -> fix), detached
DATASET=chartqa CUDA_VISIBLE_DEVICES=0 docker compose run -d --name vlm-gemma-4-e2b-chartqa gemma-4-e2b
docker logs -f vlm-gemma-4-e2b-chartqa
```

Every size's command pins the judge and coder to Codex Terra (`--judge-provider
codex --judge-model gpt-5.6-terra --judge-effort medium`); thinking is OFF on every
model (`--enable-thinking` turns it on for one run); the model runs in-process
(`hf_local`; `EXTRA_ARGS="--backend endpoint --base-url http://host.docker.internal:8020/v1"`
talks to a vLLM server instead).
