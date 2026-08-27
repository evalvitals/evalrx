# examples/benchmark images — ONE multi-stage file, one stage per model family.
#
#   base      python:3.11-slim + torch 2.13 / torchvision 0.28 (cu129) + ffmpeg +
#             gcc/libc6-dev (Triton JIT) + evalvitals[data,viz] + transformers 5.15
#             + the benchmark code. Everything every family needs.
#   qwen      = base (Qwen3.5 text/VL and Qwen3-Omni load with the base stack)
#   gemma     = base (Gemma 4 / 4-Unified are native in transformers 5.15)
#   nemotron  ITS OWN STACK (python:3.11-slim + torch 2.10 cu126 + transformers 4.57 +
#             prebuilt causal-conv1d / mamba-ssm wheels). The Nemotron 3 checkpoints
#             need their repo code (transformers' native nemotron_h produced
#             all-newline generations on the 4B, 2026-08-21), that code hard-imports
#             mamba_ssm, and the kernels only ship prebuilt wheels up to torch 2.10
#             (the sdist needs nvcc). First family whose pins differ from base.
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

COPY . /evalvitals
# transformers 5.15 explicitly: the package's [local] extra pins < 5, but
# Qwen3.5 / Gemma 4 / nemotron_h only exist in 5.x. openai: the endpoint
# backend. requests: the LLM datasets come through the HF datasets-server.
# scipy/soundfile: remote audio processors resample/read through them.
# gemini extra (google-genai): the Gemini API family (--backend gemini).
RUN pip install --no-cache-dir "/evalvitals[data,viz,gemini]" pandas requests openai scipy soundfile "matplotlib<3.11" \
    "transformers==5.15.0" "accelerate>=0.26.0"

# The benchmark code + the LLM dataset catalog it imports, at the SAME relative
# paths as in the repo (examples/benchmark/_common -> examples/dataset_selection).
COPY examples/dataset_selection/llm_band_probe /app/examples/dataset_selection/llm_band_probe
COPY examples/dataset_selection/llm_benchmark/datasets.py /app/examples/dataset_selection/llm_benchmark/datasets.py
COPY examples/benchmark/_common /app/examples/benchmark/_common
COPY --chmod=0755 examples/benchmark/docker/codex /usr/local/bin/codex

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
COPY . /evalvitals
# [local] pins transformers < 5 -> 4.57.x, the 4.x line the repo code targets
# (the model card tests 4.48.3). einops/packaging: the Omni remote code.
RUN pip install --no-cache-dir "/evalvitals[local,data,viz]" pandas requests openai scipy soundfile "matplotlib<3.11" \
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
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/examples/benchmark
WORKDIR /app/work
CMD ["python", "-m", "_common.run", "--list"]
