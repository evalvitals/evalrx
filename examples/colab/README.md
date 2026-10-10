# Colab notebooks

Select the matching hardware runtime, add `GEMINI_API_KEY` to Colab Secrets, then run the notebook from top to bottom. The notebooks install **Antigravity CLI (`agy`)** and configure API-key authentication; no browser login is needed. Agent code, charts, predictions, and confirmation results remain in the notebook outputs.

| Hardware | Model | Task | Notebook |
|---|---|---|---|
| TPU | Gemma 4 E2B-it | BBH word sorting | [Word sorting](tpu/word_sorting_repair.ipynb) |
| TPU | Gemma 4 E2B-it | CRUXEval output prediction | [Code reasoning](tpu/code_reasoning_repair.ipynb) |
| GPU | Gemma 4 E2B-it | BBH word sorting | Execution in progress; notebook pending |
| GPU | Gemma 4 E2B-it | CRUXEval output prediction | Queued; notebook pending |

The planned TPU/GPU pairs use the same model and dataset, 128 cases, seed 0, a 64/64 EXPLORE/CONFIRM split, a 512-token generation budget, and the same agent and repair settings. The hardware adaptation changes installation, device checks, and `--backend`/`--device`: `jax_local`/`tpu` versus `hf_local`/`cuda`. Both default to bfloat16; use a compatible GPU such as L4 or A100. Backend numerics can still produce different predictions.

The executed TPU word-sorting notebook reports CONFIRM accuracy of **27/64 → 37/64** (13 fixed, 3 broken). Its e-value is **6.884**, below the threshold of 20: the verdict is **partial**, not a validated repair. All six code cells completed. The outputs of the run and result cells are saved; the two setup cells' outputs were removed when setup moved to the self-contained environment (see the notebook header for the original revision and runtime). The executed TPU CRUXEval notebook reports CONFIRM accuracy of **32/64 → 41/64** (9 fixed, 0 broken). Its e-value is **51.2**, above the threshold of 20: the verdict is **fixed**, a validated inference repair (the frozen L1 candidate `scratchpad_and_expanded_limit`). It ran end to end, all six code cells, in the self-contained environment on a Colab TPU v5e runtime (image `release-colab-external-images_20261008-060111_RC00`). The remaining two GPU notebooks will be published after execution.

A successful repair must pass the independent CONFIRM check. EXPLORE gains alone are not enough. These are inference repairs around unchanged model weights, not fine-tuning.

Google recommends migrating individual-account Gemini CLI users to agy; API-key Gemini CLI access remains supported. agy API-key authentication requires `modelProvider: "gemini"` in its settings as well as `GEMINI_API_KEY`. See the [official migration announcement](https://github.com/google-gemini/gemini-cli/discussions/28017) and [authentication instructions](https://www.antigravity.google/docs/cli/install/).

## Prebuilt TPU image: install once, launch anywhere

For Google-internal Colab, or any runtime without git or package downloads, use [`tpu/evalrx_tpu.ipynb`](tpu/evalrx_tpu.ipynb). It is a single notebook. Set `EVALRX_STORE` to a folder (Drive, `gs://`, or a local path) and run all cells:

- **First run**, on a Colab TPU runtime with internet access: the store has no image yet. The notebook installs the locked environment, packs it into an image tar and the model weights into a weights tar, saves both to the store, then runs the experiment. Installing takes about 15 minutes.
- **Every later run**, on any Colab TPU runtime: the notebook finds the image in the store and installs nothing. It restores the image by unpacking it, checks the TPU and `agy`, runs the complete workflow on `bbh_word_sorting` or `cruxeval_output`, and shows the result.

For an internal runtime that cannot read the store the image was built into, copy the two tars and their `.sha256` files to a folder it can read, and point `EVALRX_STORE` there. `EVALRX_IMAGE` and `EVALRX_WEIGHTS` can also name each tar directly, including as an `https://` URL.

The image, `evalrx-colab-tpu-image-<commit>.tar` (about 5 GB), holds the installed `/content/evalrx-env` and `/content/evalrx`:

- Python 3.12, every package in `tools/colab/lock/tpu.txt`, and `agy`;
- the EvalRX source and the two datasets, frozen at 128 cases and seed 0;
- a manifest `evalrx-env/.image` that records the commit, the package versions, and the Colab image it was built on.

The weights tar, `evalrx-colab-weights-tpu-gemma-4-e2b.tar` (18 GB), holds the Gemma 4 E2B checkpoint and tokenizer. It does not depend on the EvalRX revision. Each tar has a `.sha256` file; keep it next to the tar. The notebook checks the checksum while it unpacks, and rejects a damaged or partial copy.

Launching does not run pip, uv, or git, and it does not contact GitHub, PyPI, or the model's source. It does not import or change the Colab kernel's packages, so the kernel's Python version does not matter. The only network service a run needs is the Gemini API, for `agy`. The image restores to the fixed paths `/content/evalrx` and `/content/evalrx-env`, which its scripts refer to.

A later Colab image does not affect the image. A new TPU generation can: libtpu 0.0.21.1 supports the TPUs available in Colab as of October 2026. If the TPU check fails on new hardware, update `tools/colab/lock/tpu.txt` and build a new image: set `EVALRX_REBUILD = True` (or remove the image from the store) and run the notebook on a runtime with internet access.

The scripts behind the notebooks also work outside them: `tools/colab/build_image.sh` packs an environment built by `bootstrap.sh`, and `tools/colab/build_weights.sh` packs the weights.

## Environment

The install cell does not install into the notebook kernel. [`tools/colab/bootstrap.sh`](../../tools/colab/bootstrap.sh) creates `/content/evalrx-env` with its own Python 3.12 and the exact package versions in [`tools/colab/lock/`](../../tools/colab/lock) (`gpu.txt`: torch 2.13.0+cu129, transformers 5.15.0; `tpu.txt`: jax/jaxlib 0.7.2, libtpu 0.0.21.1, gemma 4.0.1, the set the executed TPU notebook used). Every EvalRX command, including the agent check, runs from that environment. A Colab image update can change the kernel's Python and preinstalled packages without changing the EvalRX environment. The notebooks also fetch a pinned source revision (`EVALRX_REF`), not a moving branch.

Choose a setup method:

| Runtime | Setup |
|---|---|
| Colab with internet access | Run the notebook unchanged. |
| Colab with restricted egress or no git, e.g. Google-internal Colab | Use an offline bundle, and where the model source is unreachable a weights tar (below). |
| A GPU machine you control | Run the [local-runtime image](#local-runtime-container) and connect Colab to it. |

Before installing anything, setup checks each download endpoint. If an endpoint is blocked, the error names it and the variable that overrides it:

| Variable | Default |
|---|---|
| `EVALRX_PYPI_INDEX` | pip's configured `index-url`, else `https://pypi.org/simple` |
| `EVALRX_TORCH_INDEX` | `https://download.pytorch.org/whl/cu129` (GPU), `.../whl/cpu` (TPU) |
| `EVALRX_PYTHON_MIRROR` | python-build-standalone on GitHub (uv's `UV_PYTHON_INSTALL_MIRROR`) |
| `EVALRX_AGY_INSTALLER` | `https://antigravity.google/cli/install.sh` (`EVALRX_SKIP_AGY=1` skips agy) |
| `EVALRX_REF`, `EVALRX_BUNDLE`, `EVALRX_WEIGHTS`, `EVALRX_ENV`, `EVALRX_REPO` | set in the first cell |

Set these with `os.environ[...] = ...` in a cell before the install cell, or edit the first cell.

### Offline bundle

On any Linux x86_64 machine with internet access, at a committed revision:

```bash
bash tools/colab/build_bundle.sh gpu   # or tpu; writes dist/evalrx-colab-gpu-<commit>.tar
```

The tar contains the source tree, uv, a relocatable Python 3.12, every locked wheel, the agy binary, the two notebook datasets frozen at 128 cases and seed 0, and SHA-256 checksums. The builder installs the bundle into a scratch environment without network access to the package indexes, then freezes the datasets with it. A missing wheel therefore fails the build instead of the notebook. Upload the tar where the runtime can read it, such as a GCS bucket, Google Drive, or an internal share. Then set `EVALRX_BUNDLE` in the first cell to its `gs://`, `https://`, or local path (for example `/content/drive/MyDrive/...`). Setup then downloads nothing from GitHub, PyPI, PyTorch, or the agy server, and does not need git installed.

The bundle does not include the model weights. By default GPU reads `google/gemma-4-E2B-it` from the Hugging Face Hub (not gated), and TPU reads `gs://gemma-data/checkpoints/gemma4-e2b-it` anonymously. Where that source is unreachable, pack the weights once:

```bash
bash tools/colab/build_weights.sh tpu   # or gpu; writes dist/evalrx-colab-weights-tpu-gemma-4-e2b.tar
```

The TPU tar is 17 GB: the Orbax checkpoint plus its tokenizer, downloaded over plain HTTPS. The GPU tar is the Hugging Face snapshot. The weights do not depend on the EvalRX revision, so one tar serves every bundle. Host it next to the bundle and set `EVALRX_WEIGHTS` in the first cell to its path or URL, or to a directory that already holds the unpacked weights. The install cell unpacks the tar to `/content/evalrx-weights` and verifies its checksums, and the run cell passes the weights as `--model-path`.

One runtime resource remains outside both tars: the **Gemini API** (`generativelanguage.googleapis.com`) for the agy agent, authenticated with `GEMINI_API_KEY`.

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
- **GPU, online:** the install cell and the agy agent check completed. Model weights were downloaded from the Hugging Face Hub, and a reduced 16-case `evalrx run` completed baseline through M5 with exit code 0.
- **Local-runtime image:** built from a bundle with `docker build --network none`. Jupyter started, and the notebook's install cell detected the existing environment and skipped installation.
- **TPU, Colab TPU v5e runtime (`release-colab-external-images_20261008-060111_RC00`):** the complete CRUXEval notebook ran in a Jupyter kernel with the Colab kernel's environment. The environment's jax 0.7.2 and libtpu drove the TPU, and the run reached a validated repair (above). The source was copied in instead of fetched by `EVALRX_REF`, and the key came from the environment instead of Colab Secrets.
- **TPU bundle, `--network none`, CPU:** installation completed, and jax 0.7.2, libtpu 0.0.21.1, gemma 4.0.1, and TensorFlow imported.

- **Prebuilt image, Colab TPU v5e runtime (`release-colab-external-images_20261008-060111_RC00`):** `setup.ipynb` installed the environment and wrote the image (4.5 GB) and weights tar (17 GB). With git hidden and PyPI, GitHub, PyTorch, Hugging Face, the agy server and GCS unresolvable, `launch.ipynb` restored and verified the image in about a minute. The runtime was recycled while the weights were restoring, so the TPU run of `launch.ipynb` is still pending.
- **Prebuilt image, Colab runtime image on CPU (local container):** `setup.ipynb` wrote the image. Under the same isolation, `launch.ipynb` restored the image and the 18 GB weights tar in 78 s, both checksums verified. A 2-case CPU baseline then loaded Gemma 4 E2B from the restored weights while GCS was unreachable, and the `agy` check printed `AGENT_READY`.

Not yet verified: Google-internal Colab, and a complete `launch.ipynb` run on a TPU. If setup fails there, the endpoint check reports the blocked URL.

To read a `gs://` bundle or weights tar from a private bucket, first run `from google.colab import auth; auth.authenticate_user()`. The install cell does not run it. Colab TPU runtimes have no Cloud SDK (`gcloud`, `gsutil`), so there the install cell downloads `gs://` paths with the kernel's `google.cloud.storage`, using your credentials or, for a public bucket, anonymous access.
