FROM python:3.11-slim

WORKDIR /app

# ffmpeg/ffprobe: required by hf_local._resolve_audio (path/URL audio decode)
# AND by download_mmau.py (MMAU's audio ships as embedded file bytes, decoded
# straight to 16kHz mono WAV without pulling in torchcodec).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CUDA torch first so the [local] extra doesn't pull a CPU-only wheel
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY . /evalvitals
RUN pip install --no-cache-dir "/evalvitals[local,data]"

WORKDIR /app/example
COPY examples/m1_m4/mmau_qwen2_audio/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
