# Gemma 4 on JAX: using this benchmark tree

A walkthrough for the `jax_local` route of `examples/benchmark`: Gemma 4 E2B / E4B
driven by Google DeepMind's `gemma` library instead of transformers. It is in
three parts:

1. [From zero to a running demo](#part-1-from-zero-to-a-running-demo)
2. [Switching model, dataset, modality, validation set and agent](#part-2-switching-things)
3. [Extending the code: new datasets, new models](#part-3-extending-the-code)

The matrix, the design rules and the cell status live in [`README.md`](README.md);
the backend design and the measured facts in
[`docs/design_jax_backend.md`](../../docs/design_jax_backend.md). This file only
tells you what to type and which file to edit.

## Part 1. From zero to a running demo

The demo is one **cell**: one modality, one model size, one dataset, run through
the whole EvalRX chain (Stage 0 baseline, M1 analyzers, explore, M2 statistics,
M3 hypotheses, M4 verification, M5 surgery, fix search with a frozen CONFIRM
gate). The cell used below is `vlm` / `gemma-4-e2b` / `chartqa`, which ran end
to end on 2026-10-02 (256 rows, 2 h 10 min on one shared A100, FIXED +0.211).

### 1.1 What the host needs

| need | why | check |
|---|---|---|
| Docker with the NVIDIA container toolkit, your user in the `docker` group | the cell runs in `evalrx-bench-gemma-jax` | `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` |
| one GPU with about 20 GB free, 35 GB of host RAM | the E2B vlm cell took about 17 GB of the card and peaked at 32.6 GB RSS while loading (the checkpoint is stored float32 and cast after the read); the image sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so the run shares a card | `nvidia-smi`, `free -g` |
| `gsutil` | the checkpoints are public objects in `gs://gemma-data` (no account needed) | `gsutil ls gs://gemma-data/checkpoints/` |
| the `claude` CLI, logged in | judge and repair coder default to Claude (`claude-opus-5`); Part 2.4 has the alternatives | `claude -p "Reply with exactly OK"` |
| 30 GB of disk | 17 GB checkpoint mirror + 11 GB image | `df -h` |

Nothing from HuggingFace is gated for this cell; the ChartQA parquet is public.

### 1.2 Clone and configure

```bash
git clone <repo> evalrx && cd evalrx
cp examples/benchmark/.env.example examples/benchmark/.env
```

Every leaf directory has `.env -> ../../.env`, so this one file configures
them all. Open it and set the host paths the compose files mount. On a laptop
or a workstation the defaults (`~/.cache/huggingface`, `~/.claude`, ...) are
fine and you only need the checkpoint line (next step). On a host whose docker
daemon cannot read your home directory (tealab: root-squashed NFS) point every
variable at a directory the daemon can read; the comments in `.env.example`
name each one.

### 1.3 Weights (once)

```bash
mkdir -p /path/to/gemma4
gsutil -m cp -r gs://gemma-data/checkpoints/gemma4-e2b-it /path/to/gemma4/     # 17 GB
gsutil cp gs://gemma-data/tokenizers/tokenizer_gemma4.model /path/to/gemma4/
echo 'EVALRX_JAX_CKPT=/path/to/gemma4' >> examples/benchmark/.env
```

The directory is mounted read-only at `/ckpt` in the container. A service passes
`--model-path /ckpt/gemma4-<size>-it` when that directory exists and otherwise
reads the bucket on every load (works, slow). The tokenizer must sit beside the
checkpoint directories under the same name.

### 1.4 Build the image

```bash
docker compose -f examples/benchmark/docker/docker-compose.build.yml build gemma_jax
```

This builds the `gemma_jax` stage of [`docker/Dockerfile`](docker/Dockerfile):
python 3.12, `jax[cuda12]`, `gemma==4.0.1`, CPU torch, transformers, the data
loaders. About 15 minutes the first time, 11 GB. Rebuild after any change under
`evalrx/` or `examples/benchmark/_common/` (the image copies the repo).

### 1.5 Smoke the cell (no judge)

```bash
cd examples/benchmark/vlm/gemma_jax
EXTRA_ARGS="--baseline-only --limit 8" CUDA_VISIBLE_DEVICES=0 docker compose run --rm gemma-4-e2b
```

This freezes the dataset (first time only, into `../_data/chartqa/`), loads the
weights, generates 8 answers and scores them. Expect the first case to take a
minute (XLA compiles once per input shape) and a closing line like
`Baseline: PASS=4, FAIL=4, UNKNOWN=0, ...`. `CUDA_VISIBLE_DEVICES` is the
PCI-bus index of the card you want.

### 1.6 Run the full chain

```bash
cd examples/benchmark/vlm/gemma_jax
DATASET=chartqa CUDA_VISIBLE_DEVICES=0 EXTRA_ARGS="--run-tag demo" \
    docker compose run -d --name vlm-gemma-jax-e2b-chartqa gemma-4-e2b
docker logs -f vlm-gemma-jax-e2b-chartqa
```

What happens and how long each step took on the 2026-10-02 run:

| stage | what you see in the log | time |
|---|---|---|
| data + load | `[data] ...`, `[model] ...`, `judge: claude ...` | 2 min |
| Stage 0 baseline | one line per case, then `Baseline: PASS=131, FAIL=125, ...` | 5 min |
| M1 | three pinned analyzers over the EXPLORE half | 5 min |
| explore + M2 | free-form EDA report, statistics | 15 min |
| M3 + M4 | hypotheses, one re-probe each | 3 min |
| M5 + fix | surgery experiment, L1 templates, L2 coded pipelines, CONFIRM gate | 1 h 45 min |

The run is finished when the log prints the verdict (`FIXED` / `NOT FIXED`) and
the container exits. The console log is only in `docker logs`, so save it
(`docker logs vlm-gemma-jax-e2b-chartqa > outputs/chartqa.demo.console.log`)
before `docker rm vlm-gemma-jax-e2b-chartqa`.

### 1.7 Read the results

Outputs land in `outputs/gemma-4-e2b/chartqa.demo/` (the `--run-tag` becomes
the suffix; without one the directory is `chartqa/`):

| path | content |
|---|---|
| `baseline.json` | every Stage 0 prompt, output and PASS / FAIL label |
| `summary.json` | baseline accuracy, cycles, verified hypotheses, the fix verdict and winner, the effective `fix_tier` |
| `logs/run.json` | the run config (model, backend, split mode, judge) and the event trace |
| `logs/M1/` ... `logs/M5/` | one `log.json` per stage plus `artifacts/` with the figures the stage drew |
| `logs/contract/` | the machine-readable stage outputs: `c0.m1.json` ... `c0.m4.json`, `m5_surgery.json`, `m5_fix.json` (every fix candidate, its EXPLORE and CONFIRM counts, the selection) |
| `logs/media/` | the images or clips the explore report refers to |
| `fix_quarantine.json` | what was hidden from the repair coder during the fix stage |

```bash
cd ../../../..          # repo root
.venv/bin/python -m evalrx.cli dashboard examples/benchmark/vlm/gemma_jax/outputs/gemma-4-e2b/chartqa.demo
```

The container runs as root, so the output tree is root-owned. To work on it as
yourself:

```bash
docker run --rm -v "$PWD/outputs:/o" python:3.12-slim chown -R "$(id -u):$(id -g)" /o
```

### 1.8 The same demo without Docker

Build a JAX environment beside the repo's `.venv` (the `gemma` library needs
Python 3.12 or newer) and call the same entry point:

```bash
uv venv .venv-jax --python 3.12
uv pip install --python .venv-jax/bin/python "jax[cuda12]" "gemma==4.0.1" gcsfs
uv pip install --python .venv-jax/bin/python --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv-jax/bin/python -e ".[data,viz]"

cd examples/benchmark
../../.venv-jax/bin/python -m _common.run --modality vlm --model gemma-4-e2b --backend jax_local \
    --dataset chartqa --data-dir vlm/_data --run-dir vlm/gemma_jax/outputs \
    --model-path /path/to/gemma4/gemma4-e2b-it --limit 8 --baseline-only
```

Drop `--baseline-only` for the full chain; `claude` must then be on `PATH`.
`--device cpu` keeps JAX off the GPUs (slow, fine for a 2-row check). The CLI
is the same one the compose files call, so every `EXTRA_ARGS` below is also a
plain argument here.

## Part 2. Switching things

Everything in this part is a runtime argument. No file changes, no rebuild.

### 2.1 Model

**Within JAX.** The leaf has two services, one per size the `gemma` library
ships a class for:

| service / `--model` | spec | mirror directory under `EVALRX_JAX_CKPT` |
|---|---|---|
| `gemma-4-e2b` | `gemma-4-e2b-it` | `gemma4-e2b-it/` (17 GB) |
| `gemma-4-e4b` | `gemma-4-e4b-it` | `gemma4-e4b-it/` |

```bash
gsutil -m cp -r gs://gemma-data/checkpoints/gemma4-e4b-it /path/to/gemma4/
CUDA_VISIBLE_DEVICES=0 docker compose run -d --name vlm-gemma-jax-e4b-chartqa gemma-4-e4b
```

There is no 12B service: the library has no class for the Unified 12B. Adding
a third size is Part 3.2.

**Same model, other runtime.** The same `--model` value runs on four backends.
The leaf directory decides the image, `--backend` decides the code path:

| want | where | command |
|---|---|---|
| Gemma E2B served by vLLM on the host (the matrix default; fix ladder clamped to L2) | `vlm/gemma/` | `EXTRA_ARGS="--base-url http://host.docker.internal:8020/v1" docker compose run --rm gemma-4-e2b` with `vllm serve google/gemma-4-E2B-it --port 8020` running on the host |
| Gemma E2B on transformers, in-process (white-box, full fix ladder) | `vlm/gemma/` | `EXTRA_ARGS="--backend hf_local" docker compose run --rm gemma-4-e2b` |
| Gemma E2B on JAX (white-box reads, fix ladder clamped to L2) | `vlm/gemma_jax/` | as in Part 1; the service already passes `--backend jax_local` |
| a Gemini API model | `vlm/gemini/` | `docker compose run --rm gemini-2.5-flash-lite` (needs `GEMINI_API_KEY` in `.env`) |

The JAX leaves keep their own `outputs/`, so a JAX run never overwrites the
`hf_local` run of the same `--model`.

**A checkpoint from somewhere else.** `--model-path <dir>` replaces the
weights only; the spec still decides the chat template, the modalities and the
thinking-off policy. For JAX the directory is an Orbax checkpoint with
`tokenizer_gemma4.model` beside it (one level up).

### 2.2 Dataset and modality

The modality is the leaf directory (`vlm/`, `llm/`, `alm/`); the dataset is the
`DATASET` variable. A dataset belongs to exactly one modality and the runner
refuses a mismatch (`dataset 'gsm8k' is a llm task, not vlm`).

| modality | leaf | `DATASET=` (default first) | task kind |
|---|---|---|---|
| vlm | `vlm/gemma_jax` | `chartqa`, `spatial457`, `pope_random`, `pope_popular`, `pope_adversarial`, `chair` | exact / numeric, yes-no, caption |
| llm | `llm/gemma_jax` | `bbh_causal_judgement`, `bbh_word_sorting`, `bbh_tracking7`, `cruxeval_output`, `bamboogle`, `minervamath`, `supergpqa_law`, `supergpqa_economics`, `supergpqa_medicine_hard`, `hotpotqa_gepa`, `gsm8k` | each slice's own grader |
| alm | `alm/gemma_jax` | `mmau`, `mmsu`, `audiocaps_hallu`, `af_reasoning_mcq` | letter, yes-no |

```bash
cd examples/benchmark/alm/gemma_jax
DATASET=mmau CUDA_VISIBLE_DEVICES=0 docker compose run -d --name alm-gemma-jax-e2b-mmau gemma-4-e2b

cd ../../llm/gemma_jax
DATASET=gsm8k CUDA_VISIBLE_DEVICES=0 docker compose run -d --name llm-gemma-jax-e2b-gsm8k gemma-4-e2b
```

Three things change with the modality and need no flag:

* **Data.** Each modality freezes its datasets under `<modality>/_data/<dataset>/`
  (`manifest.json` plus `images/` or `audio/`) the first time any family needs
  them, and every family of that modality shares the frozen slice.
* **Towers.** The llm cells load the language model only; vlm and alm also load
  the vision and audio towers.
* **Decoding.** vlm and alm are greedy at 64 tokens; the llm tasks sample at
  T=0.6 with a 2048-token cap and a 5-sample baseline noise model in the fix
  stage. `--temperature`, `--max-new-tokens` override per run.

Row count: `LIMIT=64` (or `--limit 64`) uses the first 64 manifest rows;
`LIMIT=0` uses every row. A fresh manifest takes the task's default size and
seed; `--download-limit` / `--seed` change them, `--no-download` refuses to
freeze anything new. Natively, the equivalent of changing leaf is changing
`--modality`, `--dataset`, `--data-dir <modality>/_data` and `--run-dir`.

### 2.3 Validation set (`--held-out`)

By default a run splits its cases 50/50 into EXPLORE and CONFIRM (stratified by
label, fixed seed): M1 to M4 and the fix search use EXPLORE, the frozen winner is
scored once on CONFIRM. The validation set is a third, disjoint sample that the
default run never reads.

**Use it.**

```bash
EXTRA_ARGS="--held-out" docker compose run -d --name vlm-gemma-jax-e2b-chartqa-ho gemma-4-e2b
EXTRA_ARGS="--held-out --val-limit 64" ...        # only the first 64 validation rows
```

With `--held-out`, M4 verifies each hypothesis once on the validation set and
the fix ladder searches and selects there; CONFIRM is still used exactly once
for the winner. The run config in `logs/run.json` records
`"split_mode": "held_out_val"` (default runs say `explore_confirm`).

**Which datasets have one.** A task declares `val_limit` in its `Task`; the
download freezes `manifest_val.json` beside `manifest.json` as a fresh draw
(seed + 1) from the rows the main sample did not take:

| has a validation manifest | size | none (census or fixed set) |
|---|---|---|
| `chartqa`, `spatial457`, `mmau`, the nine `llm` slices | 128 | `pope_*`, `mmsu`, `af_reasoning_mcq` |
| `gsm8k`, `chair` | 250 | |
| `hotpotqa_gepa`, `audiocaps_hallu` | 150 | |

**Two traps.**

1. A data directory frozen before the task had `val_limit` has no
   `manifest_val.json`, and the runner stops rather than silently writing one:
   delete `<modality>/_data/<dataset>/` and let the next run re-freeze both
   files. The main draw is deterministic and comes back identical. On this
   host `vlm/_data/chartqa/` already has the file, `alm/_data/mmau/` does not.
2. `--held-out` on a task with `val_limit=0` is an error by design; those sets
   have no untouched rows to draw from.

**Give a dataset a validation set** when it has none: set `val_limit=` on its
`Task` and make its `download()` accept `val_limit` and write
`manifest_val.json` from rows the main sample excluded. `chartqa.py` is the
reference implementation (`rows_for` + the `leftover` draw). Then re-freeze.

### 2.4 The agent: Claude, Codex or Antigravity

One flag chooses both the judge (M1 selection, M2 to M5 reasoning) and the
repair coder (the CLI that writes coded pipelines in the fix stage); they are
always the same provider.

| `--judge-provider` | default `--judge-model` | binary in the container | `.env` mounts |
|---|---|---|---|
| `claude` (default) | `claude-opus-5` | `claude` | `CLAUDE_PATH`, `CLAUDE_HOME`, `CLAUDE_JSON` |
| `codex` | `gpt-5.6-terra` | `codex` (`node` + the npm package) | `CODEX_NODE`, `CODEX_PACKAGE`, `CODEX_HOME` |
| `agy` | none, pass one | `agy` (Antigravity CLI) | `AGY_PATH`, `AGY_GEMINI_HOME`, `AGY_CACHE_HOME` |

`--judge-effort` (default `high`) is passed through as `--effort` to Claude and
as `model_reasoning_effort` to Codex. The run opens with one availability probe
(`Reply with exactly OK`) and stops immediately if the provider returns nothing,
so a broken login costs seconds, not hours.

```bash
# Claude (default): nothing to pass
docker compose run -d --name ... gemma-4-e2b

# Codex
EXTRA_ARGS="--judge-provider codex" docker compose run -d --name ... gemma-4-e2b
EXTRA_ARGS="--judge-provider codex --judge-model gpt-5.6-terra --judge-effort medium" ...

# Antigravity, Google sign-in session
EXTRA_ARGS="--judge-provider agy --judge-model gemini-3.6-flash-low" docker compose run -d --name ... gemma-4-e2b
```

What each provider needs on the host:

* **Claude.** The native `claude` binary and its state directory, logged in.
  The OAuth token in `~/.claude/.credentials.json` expires; when the probe
  fails with `OAuth session expired`, log in again on the host (or re-copy the
  file to the directory `CLAUDE_HOME` points at). `IS_SANDBOX=1` in the compose
  environment lets it run as root with permissions skipped.
* **Codex.** The npm package is mounted at `/opt/codex`, Node at
  `/usr/local/bin/node`, the authenticated `~/.codex` at `CODEX_HOME`. The
  `codex` shim in [`docker/codex`](docker/codex) starts it.
* **Antigravity.** Either a Google sign-in session in `~/.gemini`, or API-key
  mode: write `{"modelProvider": "gemini"}` to
  `~/.gemini/antigravity-cli/settings.json` and set `GEMINI_API_KEY` in `.env`.
  In API-key mode `agy models` lists only Gemini 3.x slugs
  (`gemini-3.1-pro-preview`, `gemini-3.5-flash`, `gemini-3.6-flash-{low,medium,high}`,
  `gemini-3.7-flash-{low,medium,high}`); pass one as `--judge-model`. Without
  the flag agy uses its own session default, which is fine for a Google
  sign-in session and undefined in API-key mode, so set it there. An
  interactive `agy` session may rewrite `settings.json`; check it before a
  long run.

The provider does not change what the pipeline does; it changes who writes the
hypotheses and the repair code. Compare runs with `--run-tag claude` /
`--run-tag agy` so the directories stay apart.

## Part 3. Extending the code

Both extensions touch files the image copies, so rebuild (`docker compose build`
in the leaf, or the build file in Part 1.4) and run
`pytest tests/test_examples/test_benchmark_common.py` before a real run. The
tests pin the matrix and the compose files, so an incomplete addition fails
there first.

### 3.1 A new dataset

A dataset is one module in [`_common/tasks/`](_common/tasks/) that exports a
`Task` and is listed in [`_common/tasks/__init__.py`](_common/tasks/__init__.py).
Nothing else in the tree knows dataset names; the compose files take any
registered name through `DATASET=`.

**1. The manifest contract.** `download(out_dir, limit, seed, val_limit=0)`
writes `<out_dir>/manifest.json`: a list of rows

```json
{"id": "mytask-00017", "prompt": "...", "image": "images/00017.png", "audio": null,
 "answers": ["42"], "task": "exact_or_numeric", "numeric_tolerance": 0.05,
 "choices": null, "metadata": {"any": "extra"}}
```

`image` and `audio` are paths relative to the manifest, or `null`; copy the
media under `images/` or `audio/`. `task` is the grader kind and must be one of
`exact_or_numeric`, `multiple_choice_letter` (then set `choices`), `yes_no`,
`short_answer_em`, `llm_graded`, `chair_caption`; the scorers are in
[`_common/scoring.py`](_common/scoring.py). The same contract serves every
modality, which is what lets one `build_cases` / `score_case` run all three.
Return a small summary dict (row counts, paths) for the log.

**2. The protocol.** `protocol(model_label)` returns an `ExperimentProtocol`
(use `_protocol(...)` from `base.py`): a task description, the domain, the
success criteria, and `target_modalities` (`{"text"}`, `{"text","image"}` or
`{"text","audio"}`). The judge reads this at M2 and M3, so say what a failure
looks like on this task.

**3. The `Task`.**

```python
TASK = Task(
    name="mytask", modality="vlm", kind="exact_or_numeric", title="MyTask/test",
    download=download, protocol=protocol,
    pinned_m1=("answer_extraction_audit", "selfcheck_consistency", "coverage_verification_gap"),
    default_limit=256, val_limit=128, default_seed=0, max_new_tokens=64,
    source="<hub id or citation>",
)
```

`pinned_m1` is the static M1 analyzer set. Start from a sibling task of the
same kind and modality (vlm short answers: `chartqa.py`; yes-no: `pope.py`;
letters: `mmau.py`; sampled text: `llm.py`'s eight-analyzer `PINNED_M1`). Names
are the analyzer classes' `name` attributes under `evalrx/analyzers/`;
`--m1-selection judge` lets the judge pick from the catalog instead, useful
once to see what it would choose. `max_new_tokens` is the per-call cap;
`short_answer=False` for free-form outputs (captions). `output_contract` can
carry a dict the prompt-level fixes must respect (see `mmau.py`).

**4. Register.** Import the module in `_common/tasks/__init__.py` and add its
`TASK` to the tuple that fills `TASKS`. Add a row to the dataset table in
[`README.md`](README.md) and to the three leaf READMEs of the modality
(`<modality>/{gemma,gemma_jax,...}/README.md`).

**5. Check.**

```bash
.venv/bin/python -m pytest tests/test_examples/test_benchmark_common.py -q
cd examples/benchmark/vlm/gemma_jax && docker compose build
DATASET=mytask EXTRA_ARGS="--download-only" docker compose run --rm gemma-4-e2b       # freeze, inspect ../_data/mytask/manifest.json
DATASET=mytask EXTRA_ARGS="--baseline-only --limit 8" docker compose run --rm gemma-4-e2b
```

`test_default_tasks_and_pinned_sets` checks every registered task has a pinned
set, a known kind, a download and a protocol. Add a test like
`test_gsm8k_is_registered` and a download test that monkeypatches the hub call
(`test_gsm8k_download_samples_in_test_order_with_numeric_golds` is the pattern)
so the slice stays reproducible without network. A text dataset built from
`examples/dataset_selection` goes through `llm.py`'s `DATASETS` tuple instead
of a new module; `test_llm_task_names_match_the_dataset_selection_catalog`
keeps the two lists equal.

### 3.2 A new model

Two cases, with very different cost.

**A. Another checkpoint for an adapter that exists** (a new Gemma 4 size or
fine-tune the `gemma` library can load). Four edits:

1. `evalrx/specs.py`: add the spec. For a Gemma 4 variant, extend
   `_GEMMA4_JAX` with `key -> (library class, gs:// or local Orbax path)` and
   the key tuple below it; the loop builds a `ModelSpec` whose `jax=JaxSpec(...)`
   names the checkpoint, the tokenizer, `model_class` and
   `adapter="evalrx.models.backends.jax.adapters.gemma:make_adapter"`. A spec without `jax=` is
   refused by `jax_local` with a clear message.
2. [`_common/models.py`](_common/models.py): add a `Size` row (`key` is the
   `--model` value; `specs={modality: spec_key}` for the modalities it serves;
   `family="gemma"` keeps it in the gemma leaves). `test_the_matrix_is_the_one_specified`
   lists every size per modality and family; update it.
3. The compose files: one service per size in each `<modality>/gemma_jax/docker-compose.yml`
   (copy the `gemma-4-e4b` block, change the two names), and the same in the
   `gemma` leaves if the size also runs on hf_local.
   `test_gemma_jax_leaves_run_the_gemma_sizes_on_jax_local` requires the set of
   jax services to equal the set of gemma sizes whose spec has `.jax`.
4. The leaf READMEs' size tables.

Mirror the checkpoint under `EVALRX_JAX_CKPT` as `gemma4-<size>-it/`, rebuild,
smoke with `--baseline-only --limit 8`.

**B. A model from another framework or family.** The backend
[`evalrx/models/backends/jax/backend.py`](../../evalrx/models/backends/jax/backend.py)
is framework-agnostic: it drives any object satisfying
[`evalrx/models/backends/jax/protocol.py`](../../evalrx/models/backends/jax/protocol.py)
`JaxModelAdapter`, and the spec's `JaxSpec.adapter` import string
(`"pkg.module:factory"`, called as `factory(spec, runtime)`) says which one.
Write a new module beside `evalrx/models/backends/jax/adapters/gemma.py` that provides:

| member | what it must do |
|---|---|
| `n_layers`, `modalities`, `reference_attention` | static facts; `modalities` is `{"text"}` or with `"image"` / `"audio"` |
| `load()` | the heavy load; construction must stay cheap and import no `jax` |
| `encode(inputs, chat_template=)`, `render_chat(messages, tools)`, `decode(ids)` | tokenizer and template; return `Encoding` |
| `forward(enc, capture=, layers=)` | one pass; return `ForwardOut` with logits and the requested hidden states / attention |
| `unembed()`, `final_norm_params()` | the real output head and norm, for the lens analyzers |
| `generate(enc, params: SamplingParams)` | honour `max_new_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop`; return `GenerateOut` |

Two facts learned on the Gemma adapter apply to any JAX stack and are already
handled on the backend side: `JaxLocalModel` serialises every adapter call
through one re-entrant lock (the M1 analyzers run in threads, and kauldron's
type-check scope is process-global), and an unseeded sampling call must draw a
fresh seed each time (a constant seed makes every "sample" identical and the
self-consistency analyzers blind). Keep both behaviours when you write
`generate`. `tests/test_models/test_jax_local.py` runs the backend against a
toy adapter; point its fixtures at yours for a cheap first check, then a real
weights check with `--limit 2 --device cpu`.

After the adapter: a `ModelSpec` with `jax=JaxSpec(framework=..., checkpoint=...,
tokenizer=..., adapter="evalrx.models.jax.<module>:make_adapter")`, then steps
2 to 4 of case A. If the model is not Gemma, its runtime stack probably differs
too; the design rule in `README.md` is one Dockerfile stage per stack, so give
it a stage and a leaf of its own rather than growing `gemma_jax`.

### 3.3 Before you call it done

```bash
.venv/bin/python -m pytest tests/test_examples/test_benchmark_common.py tests/test_models -q
.venv/bin/python -m ruff check evalrx examples/benchmark tests
docker compose -f examples/benchmark/docker/docker-compose.build.yml build gemma_jax
cd examples/benchmark/<modality>/gemma_jax && EXTRA_ARGS="--baseline-only --limit 8" docker compose run --rm <service>
```

Then a 16-row chain (`LIMIT=16`, with the judge) before a 256-row one: the two
backend bugs of 2026-10-02 only appeared once M1 ran its analyzers in threads
and M3 compared repeated samples. Record the outcome in the cell-status table
of [`README.md`](README.md).
