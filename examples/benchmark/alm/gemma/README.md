# ALM (audio + text) × Gemma 4

One leaf of [`examples/benchmark`](../../README.md): the Gemma 4 sizes that
take 16 kHz WAV clips as input, run on every alm dataset through the shared
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
| `mmau` | MMAU/test-mini | multiple_choice_letter | 256 | gamma-lab-umd/MMAU-test-mini |
| `audiocaps_hallu` | AudioCaps-Hallucination/Random | yes_no | 300 | kuanhuggingface/AudioHallucination_AudioCaps-Random + OpenSound/AudioCaps |
| `af_reasoning_mcq` | AF-Reasoning-Eval/AQA-MCQ | multiple_choice_letter | 76 | NVIDIA/audio-flamingo (AQA_MCQ) + gijs/clothoaqa (audio) |

Data is frozen once per modality under [`../_data/`](../_data) (`<dataset>/manifest.json`
+ media), shared by all families of this modality; outputs go to
`outputs/<model>/<dataset>[.<tag>]/`.

## Run

```bash
cd examples/benchmark/alm/gemma
docker compose build                                   # once per family (cached afterwards)
# per-cell smoke: load + Stage 0 on 8 rows, no judge
EXTRA_ARGS="--baseline-only --limit 8" docker compose run --rm gemma-4-e2b
# the full chain (M1 -> explore -> M2 -> M3 -> held-out M5 -> M4 -> fix), detached
DATASET=mmau CUDA_VISIBLE_DEVICES=0 docker compose run -d --name alm-gemma-4-e2b-mmau gemma-4-e2b
docker logs -f alm-gemma-4-e2b-mmau
```

Every size's command pins the judge and coder to Codex Terra (`--judge-provider
codex --judge-model gpt-5.6-terra --judge-effort medium`); thinking is OFF on every
model (`--enable-thinking` turns it on for one run); the model runs in-process
(`hf_local`; `EXTRA_ARGS="--backend endpoint --base-url http://host.docker.internal:8020/v1"`
talks to a vLLM server instead).
