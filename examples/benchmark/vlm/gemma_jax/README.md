# VLM (image + text) × Gemma 4 on JAX

One leaf of [`examples/benchmark`](../../README.md): the JAX twin of
[`../gemma`](../gemma). Same sizes, same specs, same datasets and the same
[`_common/run.py`](../../_common/run.py); the model runs in-process through
`--backend jax_local` (Google DeepMind's `gemma` library on Flax) instead of
transformers. The image is `evalrx-bench-gemma-jax`, stage `gemma_jax` of
[`docker/Dockerfile`](../../docker/Dockerfile). Design and measured facts:
[`docs/design_jax_backend.md`](../../../../docs/design_jax_backend.md).

## Sizes (services)

| service / `--model` | spec | JAX checkpoint | GPUs | note |
|---|---|---|---|---|
| `gemma-4-e2b` | `gemma-4-e2b-it` | `gemma4-e2b-it` | 1 | verified on real weights (CPU) |
| `gemma-4-e4b` | `gemma-4-e4b-it` | `gemma4-e4b-it` | 1 | wired, not run yet |

There is no `gemma-4-12b` service: the `gemma` library has no class for it.
The vision and audio towers load with the language model (17 GB on disk for E2B).

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

Outputs are kept apart from the `hf_local` runs of the same model: this leaf
writes to its own `outputs/`.

## Weights

The specs point at the public bucket (`gs://gemma-data/checkpoints/gemma4-<size>-it`,
anonymous). Reading 17 GB from the bucket on every load is slow, so mirror once
and point `EVALRX_JAX_CKPT` in [`examples/benchmark/.env`](../../.env.example) at
the directory:

```bash
gsutil -m cp -r gs://gemma-data/checkpoints/gemma4-e2b-it /path/to/gemma4/
gsutil cp gs://gemma-data/tokenizers/tokenizer_gemma4.model /path/to/gemma4/
echo 'EVALRX_JAX_CKPT=/path/to/gemma4' >> examples/benchmark/.env
```

The directory is mounted read-only at `/ckpt`. A service passes
`--model-path /ckpt/gemma4-<size>-it` when that mirror exists and falls back to
the bucket when it does not.

## Run

```bash
cd examples/benchmark/vlm/gemma_jax
docker compose build                                   # once (cached afterwards)
# per-cell smoke: load + Stage 0 on 8 rows, no judge
EXTRA_ARGS="--baseline-only --limit 8" CUDA_VISIBLE_DEVICES=0 docker compose run --rm gemma-4-e2b
# the same on CPU, when no card is free
EXTRA_ARGS="--baseline-only --limit 8 --device cpu" docker compose run --rm gemma-4-e2b
# the full chain, detached
DATASET=chartqa CUDA_VISIBLE_DEVICES=0 docker compose run -d --name vlm-gemma-jax-4-e2b-chartqa gemma-4-e2b
docker logs -f vlm-gemma-jax-4-e2b-chartqa
```

## Limits of this backend today

* **Fix ladder stops at L2.** `jax_local` reads internals (hidden states,
  attention, logits, logprobs), but the L3a executors and L3b hooks are still
  `hf_local` methods, so the runner clamps `--fix-tier` as it does for the API
  backends.
* **Thinking stays off.** `--enable-thinking` raises on this backend; the
  template placement of the thinking token is not verified yet.
* **XLA recompiles per input shape.** The first case of each padded length
  bucket pays a compile (about 17 s on CPU).
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set in the image, so a run takes GPU
  memory on demand instead of 75 % of the card at start-up.
