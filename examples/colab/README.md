# ChartQA repair on a Colab TPU

This example runs **Gemma 4 E2B-it** through EvalRX's `jax_local` backend on
ChartQA's human-authored test questions. It diagnoses failures, searches for
prompt/inference repairs on EXPLORE, then evaluates one frozen candidate on
CONFIRM. The model weights do not change.

[Open the notebook](chartqa_repair.ipynb), or run the commands below from the
repository root. The separate [TPU backend tutorial](../../docs/colab_tpu.md)
contains text/image/audio smoke checks; those checks are not accuracy repairs.

## Actual result: the proposed accuracy repair did not validate

**Gemma 4 E2B-it, BF16, one Colab TPU v5 lite, ChartQA human test questions.**
The full automatic run completed on 2026-10-05 UTC. It searched 13 L1/L2
candidates on EXPLORE and froze `conditional_arithmetic_gate` for CONFIRM.
This prompt asks for calculations only when the question requires them; it
uses a 256-token generation budget. Model weights are unchanged.

| Split | Baseline | Frozen candidate | Wrong → right | Right → wrong |
|---|---:|---:|---:|---:|
| EXPLORE, 64 questions | 32/64 (50.0%) | 39/64 (60.94%) | 13 | 6 |
| CONFIRM, 64 questions | 32/64 (50.0%) | 32/64 (50.0%) | 9 | 9 |

The paired confirmation verdict is **`no_effect`**, `fixed=false`, e-value
**0.283773**, below the required **20**. The development improvement did not
generalize. This is a completed, runnable repair experiment with a negative
result, not a successful accuracy-repair claim. M4 verified none of its three
proposed explanations, so there is also no confirmed root-cause claim.

The complete 128-question baseline was:

```text
Baseline: PASS=64, FAIL=64, UNKNOWN=0, accuracy=0.500 (439s, 3.4s/case)
```

[reference_output.json](reference_output.json) contains the exact frozen prompt,
all 128 baseline predictions, all 64 CONFIRM predictions, candidate-selection
metrics, pinned dataset revision, image checksums, hardware and source hashes.
The [compressed raw TPU log](reference_output.worker_calls.jsonl.gz) contains
2,645 calls, including the initial backend regression check. All 64 final
CONFIRM responses match that log and completed without execution failures.

Two actual CONFIRM examples illustrate why both directions matter:

| Case | Expected | Baseline output | Candidate output |
|---|---|---|---|
| `chartqa-human-295`, sum of the smallest three bars | `3.4` | `2.4%` | calculation ending in `Answer: 3.4%` |
| `chartqa-human-998` | `light blue` | `Light blue` | `Answer: 15%` |

## Replay the frozen candidate without a judge

Use a fresh Colab TPU runtime with Python 3.12 or newer and this checkout.
For unpublished working-tree files, upload the checkout to `/content/evalrx`;
a Git clone cannot fetch uncommitted files. From the repository root:

```bash
bash tools/colab/setup.sh
python -m pip install -c ~/.cache/evalrx/colab/constraints.txt -e '.[data,contract,viz,stats]'
OMP_NUM_THREADS=2 python -u examples/colab/replay_chartqa_repair.py \
  --reference examples/colab/reference_output.json
```

This downloads the pinned ChartQA images, verifies their checksums, loads Gemma
on the TPU, regenerates the 64 CONFIRM baseline answers, then runs exactly the
frozen candidate. It needs no judge key or SSH tunnel. It writes per-question
outputs and the paired verdict to `examples/colab/outputs/replay.json`.
Replaying these same questions checks reproducibility, not a new independent
test. `--limit 4` performs only a four-question installation check.

Actual console output from a fresh native TPU process, without the HTTP worker:

```text
CONFIRM n=64: baseline=32, repaired=32, fixed=9, broken=9, verdict=no_effect
```

The [native replay JSON](reference_replay.json) and
[console output](reference_replay_console.txt) retain every answer. All 64
baseline answers, all 64 candidate answers, and the confirmation statistics
exactly matched the original search run. The
[replay verification record](../../docs/validation/colab_tpu_chartqa_replay_20261005.json)
records this check and source hashes.

The [notebook](chartqa_repair.ipynb) follows this workflow. Keep model execution
in a subprocess so the notebook kernel does not retain the TPU. Initial model
loading and compilation take minutes. The replay script was executed natively;
the notebook's code cells were syntax-checked, not run as a notebook.

## The backend bug that was actually fixed

A 300-token chart request previously exhausted TPU memory because KV-cache
allocation used the rounded output-buffer size. With a 512-token padded input,
the patch reserves 1024 cache slots instead of 2048, preserving the requested
300-token budget. The same request now completes on TPU; its answer is still
wrong. See the [before/after regression record](../../docs/validation/colab_tpu_cache_regression_20261005.json).
All 128 baseline answers stayed identical after this allocation change.

Larger multi-round pipelines still hit the device's memory limit. The full
search recorded 402 failed calls: 384 at a 512-token budget, 16 at 384 tokens,
and 2 at 400 tokens with accumulated inputs. These affected three L2 candidates,
which had 0, 55 and 62 scored pairs respectively; those are not complete
64-question comparisons. The selected L1 candidate had no such failures.
The earlier interrupted run is retained in [reference_partial.json](reference_partial.json).

## Reproduce the baseline

Use a fresh Colab TPU runtime with Python 3.12 or newer and this checkout. For
unpublished working-tree files, copy the checkout into `/content/evalrx` first;
a Git clone cannot fetch uncommitted tutorial files.

```bash
bash tools/colab/setup.sh
python -m pip install -c ~/.cache/evalrx/colab/constraints.txt -e '.[data,contract,viz,stats]'
OMP_NUM_THREADS=2 python -u examples/colab/run_chartqa_repair.py \
  --baseline-only --run-tag baseline
```

This needs no judge API key. It writes actual predictions to
`examples/colab/outputs/gemma-4-e2b/chartqa.baseline/baseline.json`. The notebook
defaults to frozen-candidate replay; this command reproduces the full baseline
separately.

Run model code in a subprocess. Importing JAX in the notebook kernel first can
keep the TPU occupied. Initial checkpoint loading and compilation take minutes;
warm inference timings do not include that startup cost.

## Run automatic discovery

The reference configuration uses an authenticated Claude CLI for diagnosis and
repair authoring:

```bash
OMP_NUM_THREADS=2 python -u examples/colab/run_chartqa_repair.py \
  --judge-provider claude --judge-model claude-sonnet-4-6 --judge-effort high
```

Changing the judge or sampling can change the selected repair and its result.
No reference repair is supplied to automatic search. Defaults freeze 128 human
ChartQA questions with seed 5022; EXPLORE/CONFIRM use the benchmark's 50/50
baseline-label-stratified split with seed 20260818. Scoring uses normalized
short answers and 5% numeric tolerance (including numeric year answers). This
is a development example using a subset of the public test set, not a full
ChartQA leaderboard evaluation.

The pipeline runs baseline, pinned M1 probes, structured M2/M3, M4, and repair
search through L2. It disables free-form exploration, code generation, and the
separate pre-fix surgery experiment. Only EXPLORE selects candidates. One frozen
candidate reaches CONFIRM; if no candidate improves EXPLORE, CONFIRM is left
untouched. A completed run may report that no repair validated.

Outputs are under `examples/colab/outputs/gemma-4-e2b/chartqa.tpu_repair/`:

- `baseline.json`: all baseline answers and labels.
- `logs/M1` through `logs/M5`: durable stage records and model calls.
- `logs/M5/log.json`: selected repair payload, CONFIRM outputs and verdict.
- `summary.json`: written when the full runner completes.

## Keep a local judge while using a remote TPU

The measured run keeps the authenticated judge on the controller machine. The
optional `tools/colab/serve_jax.py --log-dir /path/to/logs` worker loads Gemma on
the TPU and listens only on `127.0.0.1:18657`. Forward that port with SSH, then
add `--tpu-url http://127.0.0.1:18657` to the discovery or replay command. The
worker exposes generation and token log probabilities, not white-box captures;
all target-model inference still happens on the TPU. Recommendations about
unsupported higher tiers in the reference log reflect this transport
capability limit, not a test of the native backend at those tiers.

When launching through SSH, first restore the TPU environment captured from
the notebook (in this setup, `source /root/colab_env.sh`). A plain SSH shell may
otherwise lack the Colab TPU topology variables.

This transport is optional. The notebook's replay loads Gemma directly on the
TPU and needs no SSH tunnel. Worker `calls.jsonl` records actual responses and
latencies; `hardware.json` records the checked accelerator.

## Export another completed run

```bash
python examples/colab/export_chartqa_result.py \
  --run-dir examples/colab/outputs/gemma-4-e2b/chartqa.tpu_repair \
  --manifest examples/colab/data/chartqa/manifest.json \
  --dataset-revision b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5 \
  --output examples/colab/outputs/my_reference.json
```

The exporter requires a completed run and recomputes paired counts from actual
outputs. Optionally pass `--worker-log /path/to/calls.jsonl` to archive raw calls
and count execution failures. Replay accepts the resulting `--reference` file.
