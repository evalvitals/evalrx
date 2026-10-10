# Colab notebooks

Select the matching hardware runtime, add `GEMINI_API_KEY` to Colab Secrets, then run the notebook from top to bottom. The notebooks provide **Antigravity CLI (`agy`)** and configure API-key authentication; no browser login is needed. Agent code, charts, predictions, and confirmation results remain in the notebook outputs.

| Hardware | Model | Task | Notebook |
|---|---|---|---|
| TPU | Gemma 4 E2B-it | BBH word sorting or CRUXEval output prediction (set `DATASET`) | [`tpu/evalrx_tpu.ipynb`](tpu/evalrx_tpu.ipynb): install once, then launch anywhere (below) |
| GPU | Gemma 4 E2B-it | BBH word sorting | Execution in progress; notebook pending |
| GPU | Gemma 4 E2B-it | CRUXEval output prediction | Queued; notebook pending |

The planned TPU/GPU pairs use the same model and dataset, 128 cases, seed 0, a 64/64 EXPLORE/CONFIRM split, a 512-token generation budget, and the same agent and repair settings. The hardware adaptation changes installation, device checks, and `--backend`/`--device`: `jax_local`/`tpu` versus `hf_local`/`cuda`. Both default to bfloat16; use a compatible GPU such as L4 or A100. Backend numerics can still produce different predictions.

Results on a Colab TPU v5e:

- **CRUXEval, from the prebuilt image (2026-10-10):** CONFIRM accuracy was **32/64 → 48/64** (16 fixed, 0 broken). The e-value was **3855**, above the threshold of 20, so the verdict is **fixed**: a validated repair by the frozen L1 candidate `concise_direct_execution_trace`. See *Verification status* below.
- **Earlier per-task notebooks, installed online (2026-10-09):**
  - Word sorting: **27/64 → 37/64** (13 fixed, 3 broken), e = 6.884, below 20. The verdict is **partial**, not a validated repair.
  - CRUXEval: **32/64 → 41/64** (9 fixed, 0 broken), e = 51.2. The verdict is **fixed**, by the candidate `scratchpad_and_expanded_limit`.
  - `tpu/evalrx_tpu.ipynb` replaces those notebooks. They remain in git history at `f55fbea`, with their outputs.

The remaining two GPU notebooks will be published after execution.

A successful repair must pass the independent CONFIRM check. EXPLORE gains alone are not enough. These are inference repairs around unchanged model weights, not fine-tuning.

Google recommends migrating individual-account Gemini CLI users to agy; API-key Gemini CLI access remains supported. agy API-key authentication requires `modelProvider: "gemini"` in its settings as well as `GEMINI_API_KEY`. See the [official migration announcement](https://github.com/google-gemini/gemini-cli/discussions/28017) and [authentication instructions](https://www.antigravity.google/docs/cli/install/).

## Prebuilt TPU image: install once, launch anywhere

[`tpu/evalrx_tpu.ipynb`](tpu/evalrx_tpu.ipynb) runs EvalRX on any Colab TPU runtime: public Colab, Google-internal Colab, or a later Colab image. To run it:

1. Select a TPU runtime.
2. Add `GEMINI_API_KEY` to Colab Secrets.
3. Run all cells.

The notebook downloads the public prebuilt image from Google Drive. The image is shared with anyone who has the link, and the notebook holds its link and SHA-256 checksum. The notebook unpacks the image, checks the TPU and `agy`, runs the complete workflow on `bbh_word_sorting` or `cruxeval_output`, and shows the result. It installs nothing, and it needs no git, GitHub or PyPI.

The image contains the environment only, not a model. Model weights load from the source that EvalRX registers for `MODEL` in `evalrx/specs.py`. For Gemma 4 E2B and E4B on TPU, that source is Google's public `gs://gemma-data`. Switching models therefore only means setting `MODEL`. A runtime that cannot read the weights source can use a weights tar instead (`EVALRX_WEIGHTS`, see *Offline bundle* below).

| Public file (Google Drive) | Size | SHA-256 |
|---|---|---|
| [`evalrx-colab-tpu-image-e6e8ec4db687.tar`](https://drive.google.com/file/d/1dkVwb02WnR15NBfDy20m5lOeT_oENMdV) (EvalRX commit `e6e8ec4`) | 4.8 GB | `1a2048c8fc8a6e7effe659ffda4c86a065439d557ada54fc79bb918ddeb36f21` |

Google Drive limits how often a shared file can be downloaded in a day. If the notebook reports "quota exceeded", try again later, or keep your own copy. To use your own copy, set `EVALRX_STORE` to a folder (Drive, `gs://`, or a local path):

- If the folder holds an image, the notebook uses the newest one.
- If it holds none, the notebook builds one on a runtime with internet access and saves it there. Building installs the locked environment and packs it, which takes about 5 minutes. With `EVALRX_BUILD_WEIGHTS`, the notebook also packs `MODEL`'s weights into a tar next to the image.

`EVALRX_IMAGE` and `EVALRX_WEIGHTS` can also name a tar directly, as a path, a Drive file link, or an `https://` URL.

The image, `evalrx-colab-tpu-image-<commit>.tar` (about 5 GB), holds the installed `/content/evalrx-env` and `/content/evalrx`:

- Python 3.12, every package in `tools/colab/lock/tpu.txt`, and `agy`;
- the EvalRX source and the two datasets, frozen at 128 cases and seed 0;
- a manifest `evalrx-env/.image` that records the commit, the package versions, and the Colab image it was built on.

A weights tar, `evalrx-colab-weights-tpu-<model>.tar`, holds one model's checkpoint and tokenizer, for example 18 GB for Gemma 4 E2B. It does not depend on the EvalRX revision. Each tar has a `.sha256` file; keep it next to the tar. The notebook checks the checksum while it unpacks, and rejects a damaged or partial copy.

Launching does not run pip, uv, or git, and it does not contact GitHub or PyPI. It does not import or change the Colab kernel's packages, so the kernel's Python version does not matter. A run uses three network services: the image's location (Google Drive by default), the model's weights source (unless a weights tar is given), and the Gemini API, for `agy`. The image restores to the fixed paths `/content/evalrx` and `/content/evalrx-env`, which its scripts refer to.

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
bash tools/colab/build_weights.sh tpu dist gemma-4-e2b   # or gpu; any model in evalrx/benchmark/models.py; writes dist/evalrx-colab-weights-tpu-gemma-4-e2b.tar
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

- **Prebuilt image, end to end, Colab TPU v5e runtime (`release-colab-external-images_20261008-060111_RC00`), 2026-10-10:**
  - A fresh runtime installed the environment from commit `e6e8ec4` and wrote the image (`evalrx-colab-tpu-image-e6e8ec4db687.tar`, 4.8 GB, sha256 `1a2048c8fc8a6e7effe659ffda4c86a065439d557ada54fc79bb918ddeb36f21`) and the weights tar (18.2 GB, sha256 `f7d8a0c983281f2154a92ded5cb4c2b47888fc1f6c26d9e2cc2f8cfde045cdb6`) in about 6 minutes. The manifest records jax 0.7.2, libtpu 0.0.21.1 and Python 3.12.15.
  - `/content/evalrx`, `/content/evalrx-env`, the run directory, `~/.gemini` and the uv cache were then deleted. git was hidden, and PyPI, GitHub, PyTorch, Hugging Face, the agy server and GCS (including `gs://gemma-data`) were made unresolvable; the Gemini API stayed reachable.
  - Under that isolation, the restore, check, run and result cells ran without errors. These cells match sections 3–8 of `tpu/evalrx_tpu.ipynb` (they then lived in a separate `launch.ipynb`).
    - The image restored in 210 s and the weights in 1153 s, both checksums verified, with no install step.
    - jax drove the TPU, and the `agy` check printed `AGENT_READY`.
    - The model loaded from the restored weights, since GCS was unreachable.
  - The full CRUXEval run (128 cases, seed 0) reached a baseline of 63/128. CONFIRM accuracy was **32/64 → 48/64** (16 fixed, 0 broken), with e-value **3855**, above the threshold of 20. The verdict is **fixed**, a validated repair by the frozen L1 candidate `concise_direct_execution_trace`.
  - The test ran on the same runtime that built the image, after the deletions above, not on a second fresh runtime.
- **Fresh runtime, end to end, Colab TPU v5e runtime (same Colab image), 2026-10-10:**
  - A newly allocated runtime ran `tpu/evalrx_tpu.ipynb` with `EVALRX_STORE` set to the Drive folder that held the image and weights tars, under the isolation above. Nothing was on the runtime beforehand; the notebook was the version from `26c4ef8`.
  - Every cell completed without an error.
    - The image restored from Drive in 83 s, and the weights in 621 s. Both checksums verified, with no install step.
    - jax drove the TPU, and the `agy` check printed `AGENT_READY`.
  - The CRUXEval baseline was 63/128. CONFIRM accuracy was **32/64 → 44/64** (12 fixed, 0 broken), with e-value **315**. The verdict is **fixed**, by the frozen L1 candidate `simulated_interpreter_repl`.
- **Reproducibility across the three TPU CRUXEval runs:**
  - The baseline was 63/128 in each run, and CONFIRM started from 32/64 in each.
  - The repaired CONFIRM accuracy was 41, 48 and 44 of 64, each from a different selected candidate. The candidates are written by the Gemini agent, whose output varies between runs.
  - The environment and the model's predictions therefore reproduce exactly. The repair reproduces statistically: each run reached a validated fix, by a different candidate.
- **Public image download, Colab runtime image on CPU (local container), under the same isolation:** with default settings, the notebook downloaded the image anonymously from its public Google Drive link (4.8 GB in 57 s), verified it and restored it. Separately, a stalled connection and a dropped connection were simulated, and the download resumed from the current byte after each.
- **One notebook, Colab runtime image on CPU (local container):** `tpu/evalrx_tpu.ipynb` with an empty local `EVALRX_STORE` installed the environment, saved the image to the store, and then skipped the restore of the environment it had just packed. A second run, under the isolation above, found the image in the store, installed nothing, and restored and verified the image and the 18 GB weights tar.

Not yet verified: Google-internal Colab. If the first install fails on a public runtime, the endpoint check reports the blocked URL.

To read a `gs://` bundle or weights tar from a private bucket, first run `from google.colab import auth; auth.authenticate_user()`. The install cell does not run it. Colab TPU runtimes have no Cloud SDK (`gcloud`, `gsutil`), so there the install cell downloads `gs://` paths with the kernel's `google.cloud.storage`, using your credentials or, for a public bucket, anonymous access.
