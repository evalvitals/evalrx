FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install evalrx with local (white-box) dependencies.
# torch is already in the base image — install only the non-torch extras
# to avoid pip overwriting the CUDA wheel with a CPU one from PyPI.
COPY . /evalrx
RUN pip install --no-cache-dir "transformers>=4.51.0,<5.0.0" accelerate sentencepiece tiktoken && \
    pip install --no-cache-dir --no-deps "/evalrx"

WORKDIR /app/example
COPY examples/analyzer_demos/qwen_attention/ .

# Model weights are expected to be mounted at /models or pulled via HF_HOME.
# Set HF_HOME to a mounted volume to avoid re-downloading on every run.
ENV HF_HOME=/models
ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]