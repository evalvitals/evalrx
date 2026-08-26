# LLM (text only) × Qwen

One leaf of [`examples/benchmark`](../../README.md): the Qwen sizes that
take prompts as input, run on every llm dataset through the shared
[`_common/run.py`](../../_common/run.py). The image is the family's
(`evalvitals-bench-qwen`, stage `qwen` of
[`docker/Dockerfile`](../../docker/Dockerfile)); size and dataset are runtime
arguments, never a rebuild.

## Sizes (services)

| service / `--model` | hf_local spec (`--backend hf_local`) | endpoint spec (default; `—` = the hf_local spec is served) | GPUs | note |
|---|---|---|---|---|
| `qwen3.5-2b` | `qwen3.5-2b` | — | 1 | same checkpoint: text spec loads the language tower only, -vl adds the vision tower |
| `qwen3.5-4b` | `qwen3.5-4b` | — | 1 | same checkpoint: text spec loads the language tower only, -vl adds the vision tower |
| `qwen3.5-9b` | `qwen3.5-9b` | — | 1 | same checkpoint: text spec loads the language tower only, -vl adds the vision tower |

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
cd examples/benchmark/llm/qwen
docker compose build                                   # once per family (cached afterwards)
# per-cell smoke: load + Stage 0 on 8 rows, no judge
EXTRA_ARGS="--baseline-only --limit 8" docker compose run --rm qwen3.5-2b
# the full chain (M1 -> explore -> M2 -> M3 -> held-out M5 -> M4 -> fix), detached
DATASET=bbh_causal_judgement CUDA_VISIBLE_DEVICES=0 docker compose run -d --name llm-qwen3.5-2b-bbh_causal_judgement qwen3.5-2b
docker logs -f llm-qwen3.5-2b-bbh_causal_judgement
```

Every size's command pins the judge and coder to Codex Terra (`--judge-provider
codex --judge-model gpt-5.6-terra --judge-effort medium`); thinking is OFF on every
model (`--enable-thinking` turns it on for one run); **text cells default to
`--backend endpoint`**: the container talks to an OpenAI-compatible server at
`--base-url` (default `http://host.docker.internal:8020/v1` — start a vLLM server
for the size's spec on the host first; `--concurrency N` is honoured there).
`EXTRA_ARGS="--backend hf_local"` runs the model in-process instead (the
service still reserves a GPU for that case).
