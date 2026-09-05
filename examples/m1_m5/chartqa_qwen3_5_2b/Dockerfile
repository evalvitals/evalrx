FROM python:3.11-slim

WORKDIR /app
# Qwen3.5 (architecture `qwen3_5`) needs transformers >= 5.15, which is not what
# the package's [local] extra pins (< 5). The cu129 torch wheel is the one the
# cluster's vLLM venv pairs with transformers 5.15 and it runs on a 12.4 driver.
# torchvision: the Qwen3.5 processor builds a video sub-processor that imports
# it even for image-only use (ImportError at AutoProcessor.from_pretrained).
RUN pip install --no-cache-dir "torch==2.13.0" "torchvision==0.28.0" --index-url https://download.pytorch.org/whl/cu129
# gcc: torch 2.13 routes some CUDA ops (torch/_native, e.g. bmm_outer_product,
# hit by the Qwen3.5 gated-delta linear-attention layers) through Triton,
# whose JIT compiles a small C driver module on first use (needs gcc AND the libc
# headers: without them the first generate() dies in the Triton driver build).
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY . /evalrx
# viz: matplotlib for M2 figures + explore chart rendering; pandas for the
# explorer's generated analysis.py. transformers/accelerate installed explicitly
# (see above) instead of via [local].
RUN pip install --no-cache-dir "/evalrx[data,viz]" pandas "transformers==5.15.0" "accelerate>=0.26.0"

WORKDIR /app/example
COPY examples/m1_m5/vlm_benchmark_common.py /app/vlm_benchmark_common.py
COPY examples/m1_m5/chartqa_qwen3_5_2b/ .
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
CMD ["python", "run.py"]
