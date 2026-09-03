FROM python:3.11-slim

WORKDIR /app

# Install torch with CUDA 12.4 wheels first (must precede evalrx install
# so the [local] extra doesn't pull a CPU-only torch from PyPI)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install evalrx — torch is already present so [local] won't reinstall it
COPY . /evalrx
RUN pip install --no-cache-dir "/evalrx[local,gemini,data,viz]"

WORKDIR /app/example
COPY examples/m1_m3/qwen_video_temporal/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
