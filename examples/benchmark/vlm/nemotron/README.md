# VLM (image + text) × Nemotron 3 Nano

One leaf of [`examples/benchmark`](../../README.md): the Nemotron 3 Nano sizes that
take images as input, run on every vlm dataset through the shared
[`_common/run.py`](../../_common/run.py). The image is the family's
(`evalvitals-bench-nemotron`, stage `nemotron` of
[`docker/Dockerfile`](../../docker/Dockerfile)); size and dataset are runtime
arguments, never a rebuild.

## Sizes (services)

| service / `--model` | hf_local spec (default) | endpoint spec | GPUs | note |
|---|---|---|---|---|
| `nemotron-3-nano-omni-30b-a3b` | `nemotron-3-nano-omni-30b-a3b-reasoning` | `nemotron-3-nano-omni-30b-a3b-reasoning-fp8` | 2 | 62 GB BF16: device=auto over two 48 GB cards; FP8 (33 GB) is vLLM-only |

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
cd examples/benchmark/vlm/nemotron
docker compose build                                   # once per family (cached afterwards)
# per-cell smoke: load + Stage 0 on 8 rows, no judge
EXTRA_ARGS="--baseline-only --limit 8" docker compose run --rm nemotron-3-nano-omni-30b-a3b
# the full chain (M1 -> explore -> M2 -> M3 -> held-out M5 -> M4 -> fix), detached
DATASET=chartqa CUDA_VISIBLE_DEVICES=0 docker compose run -d --name vlm-nemotron-3-nano-omni-30b-a3b-chartqa nemotron-3-nano-omni-30b-a3b
docker logs -f vlm-nemotron-3-nano-omni-30b-a3b-chartqa
```

Every size's command pins the judge and coder to Claude Opus 5 (`--judge-provider
claude --judge-model claude-opus-5 --judge-effort high`); thinking is OFF on every
model (`--enable-thinking` turns it on for one run); the model runs in-process
(`hf_local`; `EXTRA_ARGS="--backend endpoint --base-url http://host.docker.internal:8020/v1"`
talks to a vLLM server instead).
