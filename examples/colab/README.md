# Colab notebooks

Select the matching hardware runtime, add `GEMINI_API_KEY` to Colab Secrets, then run the notebook from top to bottom. The notebooks install **Antigravity CLI (`agy`)** and configure API-key authentication; no browser login is needed. Agent code, charts, predictions, and confirmation results remain in the notebook outputs.

| Hardware | Model | Task | Notebook |
|---|---|---|---|
| TPU | Gemma 4 E2B-it | BBH word sorting | [Word sorting](tpu/word_sorting_repair.ipynb) |
| TPU | Gemma 4 E2B-it | CRUXEval output prediction | Execution in progress; notebook pending |
| GPU | Gemma 4 E2B-it | BBH word sorting | Execution in progress; notebook pending |
| GPU | Gemma 4 E2B-it | CRUXEval output prediction | Queued; notebook pending |

The planned TPU/GPU pairs use the same model and dataset, 128 cases, seed 0, a 64/64 EXPLORE/CONFIRM split, a 512-token generation budget, and the same agent and repair settings. The hardware adaptation changes installation, device checks, and `--backend`/`--device`: `jax_local`/`tpu` versus `hf_local`/`cuda`. Both default to bfloat16; use a compatible GPU such as L4 or A100. Backend numerics can still produce different predictions.

The executed TPU word-sorting notebook reports CONFIRM accuracy of **27/64 → 37/64** (13 fixed, 3 broken). Its e-value is **6.884**, below the threshold of 20: the verdict is **partial**, not a validated repair. All six code cells completed. The outputs of the run and result cells are saved; the two setup cells' outputs were removed when setup moved to the self-contained environment (see the notebook header for the original revision and runtime). The remaining three notebooks will be published after execution.

A successful repair must pass the independent CONFIRM check. EXPLORE gains alone are not enough. These are inference repairs around unchanged model weights, not fine-tuning.

Google recommends migrating individual-account Gemini CLI users to agy; API-key Gemini CLI access remains supported. agy API-key authentication requires `modelProvider: "gemini"` in its settings as well as `GEMINI_API_KEY`. See the [official migration announcement](https://github.com/google-gemini/gemini-cli/discussions/28017) and [authentication instructions](https://www.antigravity.google/docs/cli/install/).

## Environment

The install cell does not install into the notebook kernel. [`tools/colab/bootstrap.sh`](../../tools/colab/bootstrap.sh) creates `/content/evalrx-env` with its own Python 3.12 and the exact package versions in [`tools/colab/lock/`](../../tools/colab/lock) (`gpu.txt`: torch 2.13.0+cu129, transformers 5.15.0; `tpu.txt`: jax/jaxlib 0.7.2, libtpu 0.0.21.1, gemma 4.0.1, the set the executed TPU notebook used). Every EvalRX command, including the agent check, runs from that environment. A Colab image update can change the kernel's Python and preinstalled packages without changing the EvalRX environment. The notebooks also fetch a pinned source revision (`EVALRX_REF`), not a moving branch.

Choose a setup method:

| Runtime | Setup |
|---|---|
| Colab with internet access | Run the notebook unchanged. |
| Colab with restricted egress, e.g. Google-internal Colab | Use an offline bundle (below). |
| A GPU machine you control | Run the [local-runtime image](#local-runtime-container) and connect Colab to it. |

Before installing anything, setup checks each download endpoint. If an endpoint is blocked, the error names it and the variable that overrides it:

| Variable | Default |
|---|---|
| `EVALRX_PYPI_INDEX` | pip's configured `index-url`, else `https://pypi.org/simple` |
| `EVALRX_TORCH_INDEX` | `https://download.pytorch.org/whl/cu129` (GPU), `.../whl/cpu` (TPU) |
| `EVALRX_PYTHON_MIRROR` | python-build-standalone on GitHub (uv's `UV_PYTHON_INSTALL_MIRROR`) |
| `EVALRX_AGY_INSTALLER` | `https://antigravity.google/cli/install.sh` (`EVALRX_SKIP_AGY=1` skips agy) |
| `EVALRX_REF`, `EVALRX_BUNDLE`, `EVALRX_ENV`, `EVALRX_REPO` | set in the first cell |

Set these with `os.environ[...] = ...` in a cell before the install cell, or edit the first cell.

### Offline bundle

On any Linux x86_64 machine with internet access, at a committed revision:

```bash
bash tools/colab/build_bundle.sh gpu   # or tpu; writes dist/evalrx-colab-gpu-<commit>.tar
```

The tar contains the source tree, uv, a relocatable Python 3.12, every locked wheel, the agy binary, the two notebook datasets frozen at 128 cases and seed 0, and SHA-256 checksums. The builder installs the bundle into a scratch environment without network access to the package indexes, then freezes the datasets with it. A missing wheel therefore fails the build instead of the notebook. Upload the tar where the runtime can read it, such as a GCS bucket, Google Drive, or an internal share. Then set `EVALRX_BUNDLE` in the first cell to its `gs://`, `https://`, or local path (for example `/content/drive/MyDrive/...`). Setup then downloads nothing from GitHub, PyPI, PyTorch, or the agy server.

The bundle does not include these runtime resources:

- **Gemini API** (`generativelanguage.googleapis.com`) for the agy agent, authenticated with `GEMINI_API_KEY`.
- **Model weights.** GPU uses `google/gemma-4-E2B-it` from the Hugging Face Hub, which is not gated. TPU uses `gs://gemma-data/checkpoints/gemma4-e2b-it`, read anonymously. Where either is unreachable, copy the weights to a reachable location and add `--model-path <dir>` to the `evalrx run` cell.

### Local-runtime container

Hosted Colab, public or Google-internal, cannot start a user-supplied container image. A container can be used only through Colab's [local runtime](https://research.google.com/colaboratory/local-runtimes.html). [`tools/colab/Dockerfile`](../../tools/colab/Dockerfile) extends Google's Colab runtime image (`us-docker.pkg.dev/colab-images/public/runtime`) and runs the same bootstrap at build time:

```bash
docker build -f tools/colab/Dockerfile -t evalrx-colab:gpu .   # repo root or an unpacked bundle
docker run --gpus=all -p 127.0.0.1:9000:8080 evalrx-colab:gpu
```

In Colab, choose **Connect → Connect to a local runtime** and enter the printed `http://127.0.0.1:9000/?token=...` URL. The notebooks find `/content/evalrx` and `/content/evalrx-env` and skip installation. This image is GPU-only. TPUs are available through hosted Colab TPU runtimes.

### Verification status

All runs below used Google's Colab runtime image (`us-docker.pkg.dev/colab-images/public/runtime`). On 2026-10-09 that image already had Python 3.13, torch 2.11, and jax 0.11.1, which differ from the validated jax 0.7.2. The earlier in-kernel installation would have used those versions.

- **GPU, offline bundle, `--network none`, A100:** the install cell built the environment, torch 2.13.0+cu129 detected CUDA, and an 8-case Gemma 4 E2B baseline ran from local weights (`--model-path`). Rerunning the install cell took 0.2 s. The kernel's own packages were unchanged.
- **GPU, online:** the install cell and the agy agent check completed. Model weights were downloaded from the Hugging Face Hub, and a 16-case `evalrx run` completed its baseline and the M1 analyzers without import errors.
- **Local-runtime image:** built from a bundle with `docker build --network none`. Jupyter started, and the notebook's install cell detected the existing environment and skipped installation.
- **TPU environment, online and from a bundle with `--network none`:** installation completed, and jax 0.7.2, libtpu 0.0.21.1, gemma 4.0.1, and TensorFlow imported. These tests ran on CPU.

Not yet verified: TPU execution on TPU hardware from this environment, and Google-internal Colab. If setup fails there, the endpoint check reports the blocked URL.

To read a `gs://` bundle from hosted Colab, first run `from google.colab import auth; auth.authenticate_user()`. The install cell does not run it.
