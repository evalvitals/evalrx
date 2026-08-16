FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY . /evalvitals
RUN pip install --no-cache-dir "/evalvitals[local,data]" PyYAML

# VideoLLaMA2's `audio_visual` branch ships the trust_remote_code model class
# + mm_infer() this example needs; it is NOT on PyPI (see videollama2_model.py
# docstring). `--no-deps` avoids its pinned transformers==4.42.3 (which would
# downgrade below evalvitals' >=4.51 requirement) — only the lightweight
# extras it actually imports are installed explicitly.
RUN git clone --branch audio_visual --depth 1 \
    https://github.com/DAMO-NLP-SG/VideoLLaMA2.git /videollama2_av \
    && pip install --no-cache-dir --no-deps -e /videollama2_av \
    && pip install --no-cache-dir decord pytorchvideo iopath portalocker \
       opencv-python-headless moviepy librosa timm

WORKDIR /app/example
COPY examples/m1_m4/musicavqa_videollama2/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
