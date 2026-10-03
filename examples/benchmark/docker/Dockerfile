# examples/benchmark images — ONE multi-stage file, one stage per model family.
#
#   base      python:3.11-slim + torch 2.13 / torchvision 0.28 (cu129) + ffmpeg +
#             gcc/libc6-dev (Triton JIT) + evalrx[data,viz] + transformers 5.15
#             + the benchmark code. Everything every family needs.
#   qwen      = base (Qwen3.5 text/VL and Qwen3-Omni load with the base stack)
#   gemma     = base (Gemma 4 / 4-Unified are native in transformers 5.15)
#   nemotron  ITS OWN STACK (python:3.11-slim + torch 2.10 cu126 + transformers 4.57 +
#             prebuilt causal-conv1d / mamba-ssm wheels). The Nemotron 3 checkpoints
#             need their repo code (transformers' native nemotron_h produced
#             all-newline generations on the 4B, 2026-08-21), that code hard-imports
#             mamba_ssm, and the kernels only ship prebuilt wheels up to torch 2.10
#             (the sdist needs nvcc). First family whose pins differ from base.
#   gemma_jax ITS OWN STACK (python:3.12-slim + jax[cuda12] 0.11 + flax + Google
#             DeepMind's gemma 4.0.1 + CPU torch 2.14). The JAX twin of the gemma
#             stage for --backend jax_local (docs/design_jax_backend.md): the gemma
#             library needs python >= 3.12, and torch only carries the captured
#             arrays into the analyzers, so the CPU wheel is enough. Not a family
#             of its own: same sizes and specs as gemma, leaves <modality>/gemma_jax.
#
# Size and dataset are NOT build-time: the same image runs every size of its
# family and every modality's datasets. Build context is the repo root:
#   docker compose -f examples/benchmark/docker/docker-compose.build.yml build
# or per leaf: `docker compose build` (target: <family>).
FROM python:3.11-slim AS base

WORKDIR /app
# ffmpeg/ffprobe: audio decode (hf_local._resolve_audio + the audio downloaders).
# gcc + libc headers: torch 2.13 routes some CUDA ops through Triton, whose JIT
# compiles a small C driver module on first use (Qwen3.5 gated-delta layers).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
# cu129 wheels run on the cluster's 12.4 driver; torchvision is REQUIRED by the
# Qwen3.5 processor even for image-only use (its video sub-processor imports it).
RUN pip install --no-cache-dir "torch==2.13.0" "torchvision==0.28.0" \
    --index-url https://download.pytorch.org/whl/cu129

COPY . /evalrx
# transformers 5.15 explicitly: the package's [local] extra pins < 5, but
# Qwen3.5 / Gemma 4 / nemotron_h only exist in 5.x. openai: the endpoint
# backend. requests: the LLM datasets come through the HF datasets-server.
# scipy/soundfile: remote audio processors resample/read through them.
# gemini extra (google-genai): the Gemini API family (--backend gemini).
RUN pip install --no-cache-dir "/evalrx[data,viz,gemini]" pandas requests openai scipy soundfile "matplotlib<3.11" \
    "transformers==5.15.0" "accelerate>=0.26.0"

# The benchmark code + the LLM dataset catalog it imports, at the SAME relative
# paths as in the repo (examples/benchmark/_common -> examples/dataset_selection).
COPY examples/dataset_selection/llm_band_probe /app/examples/dataset_selection/llm_band_probe
COPY examples/dataset_selection/llm_benchmark/datasets.py /app/examples/dataset_selection/llm_benchmark/datasets.py
COPY examples/benchmark/_common /app/examples/benchmark/_common
COPY --chmod=0755 examples/benchmark/docker/codex /usr/local/bin/codex
# CHAIR captioning task (tasks/chair_words.py): nltk tokeniser/tagger/WordNet on
# the default search path, so the in-harness grader is the official chair.py rule.
RUN pip install --no-cache-dir "nltk>=3.9" \
    && python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab averaged_perceptron_tagger_eng wordnet omw-1.4

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/examples/benchmark
WORKDIR /app/work
CMD ["python", "-m", "_common.run", "--list"]

FROM base AS qwen

FROM base AS gemma

# Gemini (Google Gen AI API): the base stack as-is — google-genai is installed
# there; no GPU is used, the image only carries the harness, the judge/coder
# CLIs and the dataset catalog.
FROM base AS gemini

FROM python:3.11-slim AS nemotron
# Not FROM base: the Mamba2 kernels the remote NemotronH code hard-imports exist
# as prebuilt wheels only up to torch 2.10, and their sdist build needs nvcc.
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "torch==2.10.0" "torchvision==0.25.0" \
    --index-url https://download.pytorch.org/whl/cu126
COPY . /evalrx
# [local] pins transformers < 5 -> 4.57.x, the 4.x line the repo code targets
# (the model card tests 4.48.3). einops/packaging: the Omni remote code.
RUN pip install --no-cache-dir "/evalrx[local,data,viz]" pandas requests openai scipy soundfile "matplotlib<3.11" \
    einops packaging "transformers==4.57.6"
# Prebuilt kernels matching torch 2.10 / cu12 / cp311 / cxx11abi=TRUE (Dao-AILab +
# state-spaces release assets): causal_conv1d for the fast path, mamba_ssm for the
# gated RMSNorm + SSD ops the remote modeling_nemotron_h.py imports unconditionally.
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" \
    "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
COPY examples/dataset_selection/llm_band_probe /app/examples/dataset_selection/llm_band_probe
COPY examples/dataset_selection/llm_benchmark/datasets.py /app/examples/dataset_selection/llm_benchmark/datasets.py
COPY examples/benchmark/_common /app/examples/benchmark/_common
COPY --chmod=0755 examples/benchmark/docker/codex /usr/local/bin/codex
# CHAIR captioning task (tasks/chair_words.py): nltk tokeniser/tagger/WordNet on
# the default search path, so the in-harness grader is the official chair.py rule.
RUN pip install --no-cache-dir "nltk>=3.9" \
    && python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab averaged_perceptron_tagger_eng wordnet omw-1.4
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/examples/benchmark
WORKDIR /app/work
CMD ["python", "-m", "_common.run", "--list"]

FROM python:3.12-slim AS gemma_jax
# Not FROM base: the gemma library (Flax Linen) needs python >= 3.12 and the model
# runs on jax[cuda12], whose wheels bring their own CUDA 12 libraries. torch is the
# CPU wheel: jax_local hands captured arrays to the analyzers as torch tensors and
# never runs the model through it.
WORKDIR /app
# ffmpeg: audio decode (evalrx.models._media.resolve_audio + the audio downloaders).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
COPY . /evalrx
# gcsfs: the specs' default checkpoints are gs://gemma-data/... (public, anonymous);
# a local Orbax mirror mounted at /ckpt skips the bucket (see the leaf compose files).
# transformers: no model code here, only the image helpers some analyzers import.
RUN pip install --no-cache-dir "/evalrx[jax,data,viz,gemini]" "jax[cuda12]==0.11.2" "gemma==4.0.1" gcsfs \
    pandas requests openai scipy soundfile "matplotlib<3.11" "transformers==5.15.0"
COPY examples/dataset_selection/llm_band_probe /app/examples/dataset_selection/llm_band_probe
COPY examples/dataset_selection/llm_benchmark/datasets.py /app/examples/dataset_selection/llm_benchmark/datasets.py
COPY examples/benchmark/_common /app/examples/benchmark/_common
COPY --chmod=0755 examples/benchmark/docker/codex /usr/local/bin/codex
RUN pip install --no-cache-dir "nltk>=3.9" \
    && python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab averaged_perceptron_tagger_eng wordnet omw-1.4
# XLA would otherwise reserve 75 % of the card at start-up (shared GPU boxes).
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/examples/benchmark XLA_PYTHON_CLIENT_PREALLOCATE=false
WORKDIR /app/work
CMD ["python", "-m", "_common.run", "--list"]
