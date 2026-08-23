# Benchmark matrix: model family × modality × dataset

The full diagnosis chain — Stage 0 baseline → M1 (pinned analyzers) → explore →
M2 → M3 → held-out M5 → M4 → fix — run as a **matrix** over three model families,
three input modalities and the benchmark datasets of each modality, through one
code path ([`_common/run.py`](_common/run.py)). Each `<modality>/<family>/`
directory is one cell family: a compose file whose services are the sizes, with
the dataset as a runtime variable.

```
examples/benchmark/
├── _common/            the code: run.py (CLI) · models.py (matrix) · tasks/ (datasets) · runner.py (loop wiring)
├── docker/Dockerfile   ONE multi-stage file: base → qwen | gemma | nemotron (docker-compose.build.yml builds all)
├── .env.example        host mount sources (copy to .env; every leaf links to it)
├── vlm/ _data/ {qwen,gemma,nemotron}/     image + text   : chartqa, spatial457
├── llm/ _data/ {qwen,gemma,nemotron}/     text only      : the nine band-located slices of dataset_selection
└── alm/ _data/ {qwen,gemma,nemotron}/     audio + text   : mmau, audiocaps_hallu
```

## The matrix

`--model` is the **size key**; `models.py` maps (size, modality) to the registered
spec. Same checkpoint, different modality = different spec only where the loader
differs (Qwen3.5 text tower vs vision tower); Gemma 4 is one spec for all three.

| size key (`--model`) | family | LLM | VLM | ALM | GPUs | hf_local spec(s) |
|---|---|---|---|---|---|---|
| `qwen3.5-2b` | Qwen | ✓ | ✓ | — | 1 | `qwen3.5-2b` / `qwen3.5-2b-vl` |
| `qwen3.5-4b` | Qwen | ✓ | ✓ | — | 1 | `qwen3.5-4b` / `qwen3.5-4b-vl` |
| `qwen3.5-9b` | Qwen | ✓ | ✓ | — | 1 | `qwen3.5-9b` / `qwen3.5-9b-vl` (the "8B" is the 9B) |
| `qwen3-omni-30b-a3b` | Qwen | — | — | ✓ | 2 | `qwen3-omni-30b-a3b-instruct` |
| `gemma-4-e2b` | Gemma 4 | ✓ | ✓ | ✓ | 1 | `gemma-4-e2b-it` |
| `gemma-4-e4b` | Gemma 4 | ✓ | ✓ | ✓ | 1 | `gemma-4-e4b-it` |
| `gemma-4-12b` | Gemma 4 | ✓ | ✓ | ✓ | 1 | `gemma-4-12b-it` (Unified, encoder-free; audio native) |
| `nemotron-3-nano-4b` | Nemotron 3 | ✓ | — | — | 1 | `nemotron-3-nano-4b` (BF16; `-fp8` for `--backend endpoint`) |
| `nemotron-3-nano-omni-30b-a3b` | Nemotron 3 | — | ✓ | ✓ | 2 | `nemotron-3-nano-omni-30b-a3b-reasoning` (BF16; `-fp8` for endpoint) |

`python -m _common.run --list` prints the same table from the code.

Gemma 4 12B **does** take audio (the model card lists audio on E2B, E4B and 12B;
the 12B is the encoder-free "Unified" variant), so it sits in the ALM row; drop
it from `SIZES["gemma-4-12b"].specs` if it should stay out.

### Datasets

| modality | `--dataset` | slice | scoring | default rows |
|---|---|---|---|---|
| vlm | `chartqa` (default) | ChartQA test, human-authored | normalised exact match, 5 % numeric tolerance | 256 |
| vlm | `spatial457` | Spatial457 L5_6d_spatial | normalised exact match | 256 |
| llm | `bbh_causal_judgement` (default), `bbh_word_sorting`, `bbh_tracking7`, `cruxeval_output`, `bamboogle`, `minervamath`, `supergpqa_law`, `supergpqa_economics`, `supergpqa_medicine_hard` | the band-located slices of [`dataset_selection`](../dataset_selection/llm_benchmark/datasets.py) | each slice's own grader on the extracted answer | 256 |
| alm | `mmau` (default) | MMAU test-mini, 4-way MC | option letter | 256 |
| alm | `audiocaps_hallu` | AudioCaps object hallucination (Random) | Yes/No | 300 |

Every dataset is frozen to `<modality>/_data/<dataset>/manifest.json` (+ `images/`
or `audio/`) the first time a cell of that modality needs it, with the same seeded
sample the `m1_m4` examples use; the manifest protocol is modality-blind
(`prompt`, `image`, `audio`, `answers`, `task`), so one `build_cases` /
`score_case` serves all three.

## Design rules (why the tree looks like this)

* **The image is keyed by the family's runtime stack, never by size or dataset.**
  Size = `--model`, dataset = `--dataset`; both are runtime arguments to the same
  image. `qwen` and `gemma` are the `base` stack (torch 2.13 cu129 + transformers
  5.15). `nemotron` is the first family whose pins differ: its checkpoints need
  the repo's own code (transformers' native `nemotron_h` generated only newline
  tokens on the 4B), that code hard-imports `mamba_ssm`, and the Mamba kernels
  ship prebuilt wheels only up to torch 2.10 (the sdist needs `nvcc`) — so its
  stage is torch 2.10 cu126 + transformers 4.57 + the Dao-AILab/state-spaces
  wheels. One Dockerfile, one stage per family, split on evidence. Two more
  facts about that repo code live in `hf_local` as family shims
  (`HFLocalModel._apply_family_shims`): `generate()` must not pre-build a
  `DynamicCache` (the repo builds its hybrid Mamba cache only when none is
  passed — without the shim it recomputes every step at ~2 tok/s) and generation
  must also stop on the tokenizer's `<|im_end|>` (the template's turn end;
  `generation_config` only lists `</s>`, so every answer padded to the cap).
  The remote class has no SDPA dispatch, so nemotron sizes default to eager.
* **`hf_local` by default for every model** (in-process transformers: white-box
  capture and paper-method fix candidates stay available). `--backend endpoint`
  is the black-box alternative (OpenAI-compatible server; images yes, audio not
  carried). The two 30B-A3B omni models need `--device auto` over two 48 GB cards
  (`CUDA_VISIBLE_DEVICES=a,b`; their services already pass `--device auto`).
* **Thinking OFF on every model.** Every spec in the matrix sends
  `enable_thinking=False` on each template render (Qwen3.5, Gemma 4, Nemotron 3
  all default ON otherwise); `--enable-thinking` flips it for one run.
* **Text prompts go through the chat template.** `hf_local` used to tokenise a
  text-only spec's prompt verbatim (completion mode), so Qwen3.5 opened its own
  `<think>` block and ran 8192 tokens on a causal-judgement item; the runner sets
  `RuntimeConfig(apply_chat_template=True)` (new, opt-in, off by default for
  everyone else), which renders one user turn with the spec's
  `enable_thinking=False`. Multimodal specs always did this — `hf_local` has two
  encode paths chosen by the SPEC, not the task (`_encode` for text specs,
  `_encode_vlm` for any spec with vision/audio, so Gemma 4 / Qwen3.5-VL take the
  processor template even on LLM datasets); a template bug can sit in one path
  only, so check both. `tests/test_models/test_hf_local_chat_template.py` pins
  the two paths to identical template kwargs.
* **Decoding:** short-answer tasks (vlm, alm) are greedy at 64 tokens; the llm
  tasks sample at T=0.6 / top_p 0.95 / top_k 20 with a 2048-token cap and a
  5-sample baseline noise model in the fix stage, like `llm_benchmark`. Serial
  in-process generation is ~an order of magnitude slower than `llm_benchmark`'s
  vLLM endpoint (~29 tok/s on the 2B; ~25 s per causal-judgement item), so the
  llm default is 256 rows, not a census; `--backend endpoint` is the fast path
  when a vLLM server is up.
* **Judge/coder pinned to Claude Opus 5** in every compose command (the CLI
  default stays `agy`).
* **Nemotron's FP8 checkpoints** (`nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8`,
  `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`) are ModelOpt exports:
  transformers has no ModelOpt quantizer and refuses fp8 below compute
  capability 8.9 (the lab's A6000/A100 are 8.6/8.0). They are registered as the
  **endpoint** resolution of their cells (serve with vLLM ≥ 0.20); `hf_local`
  loads the BF16 siblings, which are the same weights before quantization.
* **Thinking leaks are model behaviour, not a harness bug:** Gemma 4's template
  defaults `enable_thinking` to false and the spec sends it explicitly, yet 3/8
  MMAU generations still opened a `thought` channel and hit the 64-token cap.
  `termination_audit` / `answer_extraction_audit` are pinned for exactly that.
* **M1 is pinned per task** (`Task.pinned_m1`, the audited sets of the `m1_m4`
  examples; an 8-analyzer text set for llm). `--m1-selection judge` switches to
  catalog selection, which is modality-gated on the MODEL — on a multimodal spec
  running a text task it can pick image analyzers, hence the pinned default.

## Run

```bash
cp examples/benchmark/.env.example examples/benchmark/.env      # tealab: mount sources on /tealab-data
docker compose -f examples/benchmark/docker/docker-compose.build.yml build   # all three images

cd examples/benchmark/vlm/qwen
EXTRA_ARGS="--baseline-only --limit 8" docker compose run --rm qwen3.5-2b      # per-cell smoke (no judge)
DATASET=spatial457 CUDA_VISIBLE_DEVICES=2 docker compose run -d --name vlm-qwen35-4b-spatial457 qwen3.5-4b
docker logs -f vlm-qwen35-4b-spatial457

cd ../../alm/qwen                                                            # 2-GPU cell
CUDA_VISIBLE_DEVICES=0,2 docker compose run -d --name alm-qwen3omni-mmau qwen3-omni-30b-a3b
```

Useful `EXTRA_ARGS`: `--baseline-only` (download + load + Stage 0, the cheap
per-cell check), `--skip-fix` (M1..M5 only), `--code-only`, `--no-explore`,
`--run-tag smoke`, `--analyzer-max-cases 16`, `--m1-selection judge`,
`--backend endpoint --base-url http://host.docker.internal:8020/v1`.

Outputs: `<modality>/<family>/outputs/<model>/<dataset>[.<tag>]/` with
`baseline.json` (every Stage 0 output + label), `logs/` (run log, artifacts,
README.txt guide), `explore/`, `summary.json`. The dashboard reads the run dir:
`python -m evalvitals.cli dashboard examples/benchmark/vlm/qwen/outputs/qwen3.5-2b/chartqa`.

## Cell status

`baseline` = `--baseline-only` load + Stage 0 succeeded on this cluster; `chain` =
a full M1→fix run completed. Blank = not run yet.

All smokes: `--baseline-only --limit 8`, A6000, 2026-08-21.

| cell | baseline | chain | notes |
|---|---|---|---|
| vlm / qwen3.5-2b / chartqa | ✓ 7/8, 2.0 s/case | | same 256-row sample (seed 5022) as `m1_m4/chartqa_qwen3_5_2b` |
| llm / qwen3.5-2b / bbh_causal_judgement | ✓ 4/8, 25 s/case | | before the chat-template fix: 8/8 at the 2048 cap (greedy) and 4/4 at an 8192 cap (sampled), all starting with `<think>` |
| vlm / gemma-4-e2b / chartqa | ✓ 4/8, 0.8 s/case | | first run 28 s/case = the 10 GB lazy load; the runner now loads before timing |
| llm / gemma-4-e2b / bbh_causal_judgement | ✓ 2/8, 27 s/case | | every output ends in an `Answer:` line |
| alm / gemma-4-e2b / mmau | ✓ 5/8, 2.0 s/case | | audio through the generic hf_local encode; 3 `thought`-channel truncations |
| llm / nemotron-3-nano-4b / bbh_causal_judgement | ✓ 4/8, 6.2 s/case | | remote code on the torch 2.10 stack + the two `hf_local` shims; before them: native 5.15 = newline-only output, remote code without shims = right text then `<|im_end|>` padding to the cap at 2 tok/s (249 s/case) |
| alm / qwen3-omni-30b-a3b / mmau | | | needs two free 48 GB cards |
| vlm+alm / nemotron-3-nano-omni-30b-a3b | | | needs two free 48 GB cards; remote-code processor path unexercised |
