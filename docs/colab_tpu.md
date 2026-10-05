# Run EvalRX on a Colab TPU

For a dataset-level repair experiment, see the
[ChartQA + Gemma 4 E2B example](https://github.com/evalvitals/evalrx/tree/ruinan/examples/colab).
It includes a completed automatic search, frozen-candidate replay, and actual
per-question outputs. On 64 CONFIRM questions the candidate stayed at 32/64,
with 9 repairs and 9 regressions; no accuracy repair validated. The checks below establish backend functionality; they are not a
model-accuracy repair result.

The recorded smoke checks predate the KV-cache allocation correction found
during the ChartQA run. The corrected 300-token request has now passed on TPU;
the [regression record](validation/colab_tpu_cache_regression_20261005.json)
contains the actual cache sizes and output. This verifies execution, not answer
correctness or a dataset-level accuracy repair.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evalvitals/evalrx/blob/ruinan/examples/colab/tpu_quickstart.ipynb)

Use `jax_local` to run Gemma 4 E2B on a Colab TPU, inspect its hidden states
and attention, and run EvalRX's logit lens analyzer. The
[notebook](https://github.com/evalvitals/evalrx/blob/ruinan/examples/colab/tpu_quickstart.ipynb)
contains the same steps with optional image and audio checks. Upload that file
to Colab, or open it from the repository branch containing the tutorial.

These checks need no SSH tunnel, Hugging Face login, or judge API key. They
validate local inference and analysis. The separate ChartQA example needs a
judge for automatic search. Its frozen-candidate replay and baseline-only
commands need no judge key.

## Choose the runtime

In Colab, select **Runtime → Change runtime type → TPU**. Start with a fresh
runtime, Python 3.12 or newer, and enough host memory for the model and captured
arrays. Start with E2B and BF16 on a single 16 GB TPU; the model's media towers
use float32 and increase memory use.

Colab supplies JAX and libtpu. Preserve that matched set instead of upgrading
JAX independently. On a Cloud TPU VM without a preinstalled runtime, follow the
[JAX TPU installation guide](https://docs.jax.dev/en/latest/installation.html#pip-installation-google-cloud-tpu).
A CPU PyTorch wheel is sufficient: model computation uses JAX, and EvalRX
converts captured arrays to CPU tensors for its existing analyzers.

## Install from the checkout

Run these notebook cells in order. The examples use the `ruinan` development
branch; pin a commit containing this tutorial when reproducing a result.

```python
import subprocess
import sys
from pathlib import Path

repo = Path("/content/evalrx")
if not repo.exists():
    subprocess.run([
        "git", "clone", "--branch", "ruinan",
        "https://github.com/evalvitals/evalrx.git", str(repo),
    ], check=True)
subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True)
subprocess.run(["bash", str(repo / "tools/colab/setup.sh")], check=True)
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
```

The installer uses this checkout, pins Gemma to 4.0.1, and preserves Colab's
JAX, jaxlib, libtpu, torch and fsspec versions. It handles Debian packages
without pip uninstall metadata and the shared import directory used by
TensorFlow distributions. Installer logs are in `~/.cache/evalrx/colab`.

If Gemma or TensorFlow was already imported before installation, restart the
Python session before running model code. Keep JAX out of the notebook kernel
when following the subprocess workflow below, so the kernel does not retain
the TPU while a subprocess needs it.

## Verify text inference and white box access

```python
import os

env = os.environ.copy()
env.update(JAX_PLATFORMS="tpu,cpu", OMP_NUM_THREADS="2", TF_CPP_MIN_LOG_LEVEL="2")
reports = repo / "colab_results"
reports.mkdir(exist_ok=True)
subprocess.run([
    sys.executable, "-u", str(repo / "tools/colab/jax_smoke.py"),
    "--device", "tpu", "--text-only", "--output", str(reports / "text.json"),
], cwd=repo, env=env, check=True)
```

Each new process reads the public Orbax checkpoint from
`gs://gemma-data/checkpoints/gemma4-e2b-it`. It may take several minutes.
`--ckpt /path/to/gemma4-e2b-it` uses a local mirror instead; place
`tokenizer_gemma4.model` beside or inside that checkpoint directory.

The suite checks the actual backend, model loading, cold and warm greedy
inference, sampled generation, token log probabilities, captured tensors and
a logit lens analyzer. It verifies finite values, 36 hidden-state entries,
35 attention layers, and attention row normalization. The short prompts are
functional checks, not an accuracy benchmark.

```python
import json
report = json.loads((reports / "text.json").read_text())
assert report["status"] == "passed"
for step in report["steps"]:
    print(step["name"], step["status"], step["seconds"])
```

Reports are updated after each step. `running` means incomplete, including if
the runtime disconnected or the process was killed. `failed` or a nonzero exit
means at least one check failed. First-call timing includes compilation;
load time includes checkpoint I/O. Download the JSON files before Colab expires.

## Optional image and audio checks

Run each suite in a new process to release model buffers and compiled programs
between modalities. Omit `--text-only` so the towers are loaded.

```python
for suite in ("image", "audio"):
    subprocess.run([
        sys.executable, "-u", str(repo / "tools/colab/jax_smoke.py"),
        "--device", "tpu", "--suite", suite,
        "--output", str(reports / f"{suite}.json"),
    ], cwd=repo, env=env, check=True)
```

The image check uses a solid red image; the audio check uses two seconds of
silence. Both check media-token masks, finite logits and nonempty generation.
These synthetic inputs establish that the media path runs; use your task data
for accuracy evaluation. Different image sizes and audio lengths may trigger
additional XLA compilation.

## Use the model in your own script

Set `JAX_PLATFORMS=tpu,cpu` before starting Python, then use the same public API
as the other EvalRX backends:

```python
from evalrx.models import RuntimeConfig, compose
from evalrx.core.capability import Capability
from evalrx.analyzers.lens.logit_lens import LogitLensAnalyzer

model = compose("gemma-4-e2b-it", "jax_local", RuntimeConfig(
    device="tpu", dtype="bfloat16", apply_chat_template=True,
    max_new_tokens=32, engine_kwargs={"text_only": True},
))
prompt = "What is the capital of France? Answer in one word."
print(model.generate(prompt))
trace = model.forward(prompt, capture={Capability.LOGITS, Capability.HIDDEN_STATES})
print(trace.logits.shape, len(trace.hidden_states), trace.logits.device)
print(LogitLensAnalyzer(top_k=3).run(model, prompt).to_json(indent=2))
```

For media, set `text_only=False` and pass an `Inputs` object with `image` or
`audio` and `prompt`. See the notebook for a subprocess version of this example.

## Measured validation

[Download the validation record](validation/colab_tpu_20261005.json) for the
full step outputs, environment versions, source hashes and notebook execution
record. All 14 steps across the three suites passed. The notebook's seven
code cells also completed in order in default mode on the same Colab runtime,
including installation, the text suite and ZIP export. Its optional branches
were disabled; image and audio were validated separately above. Execution used
a pre-synchronized checkout and Python code cells, not browser UI automation.

The validation on **2026-10-05 UTC** used one Colab device reported as
`TPU v5 lite`, 16.91 GB HBM (15.75 GiB), 47 GiB host RAM, Python 3.13.15,
JAX/jaxlib 0.7.2, libtpu 0.0.21.1, Gemma 4.0.1, Flax 0.11.2,
Kauldron 1.4.4 and CPU PyTorch 2.9.0. The dependency check reported no broken
requirements. The 33 JAX CPU unit tests passed; the two opt-in real-weight
tests were deselected in that CPU test invocation. The TPU suites below loaded
real weights separately.

| Check | Observed result | Time |
|---|---|---|
| Text-only model load | Passed; BF16 language model | 140.8 s |
| First greedy answer | `Paris` | 101.35 s, including compilation |
| Same prompt again | `Paris` | 0.185 s |
| Four sampled answers | Four nonempty, distinct answers | 44.34 s |
| Token log probabilities | Finite values for generated tokens | 36.27 s |
| White-box forward | 36 hidden entries, 35 attention layers; finite BF16 tensors on CPU | 36.05 s |
| Logit lens | 36-layer result; final normalization applied | 37.00 s |
| Image load | Full model with media towers | 119.5 s |
| Image forward and generation | 266 image tokens, 14 × 19 grid, answer `Red` | 153.35 s, including compilation |
| Audio load | Full model with media towers | 79.0 s |
| Audio forward and generation | 50 audio tokens from 2 s silence, answer `No` | 184.54 s, including compilation |

All three suites ran in separate processes. Their reported HBM peaks were
11.17 GB (text), 11.76 GB (image) and 11.34 GB (audio). Peak host RSS was
30.02 GB for the image suite and **46.60 GB for audio**, close to the available
Colab host memory. The audio check is therefore optional; do not assume a
lower-memory Colab runtime can run it. These timings describe these particular short prompts and
include compilation where indicated. They are not steady-state throughput
estimates or task-accuracy measurements.

## Scope and troubleshooting

- The built-in JAX adapter supports Gemma E2B and E4B. This tutorial targets E2B;
  E4B has a different memory requirement.
- `sharding="auto"` selects FSDP on multiple visible TPU devices. One visible
  TPU means single-device placement. Multi-chip and multi-host execution are
  not established by a single-device check.
- JAX gradient analyzers, L3 interventions and LoRA are not implemented. The
  JAX benchmark repair ladder is capped at L2. Thinking mode and tool rendering
  are also not supported by this adapter.
- If the report says `cpu`, confirm Colab's runtime type and start a fresh
  session. `--device tpu` in the smoke explicitly rejects a CPU backend.
- For memory errors, stop other model processes and begin with the text-only
  suite. Do not run suites concurrently. A crash or OOM does not count as a pass.
- If installation fails, inspect `pip_setup.log`. Do not discard the runtime
  constraints to force installation; the preinstalled accelerator versions
  may need a compatible dependency set.

See the [backend design](design_jax_backend.md) for the adapter contract and
remaining implementation work.
