FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY . /evalrx
RUN pip install --no-cache-dir "/evalrx[local,data]" PyYAML

# VideoLLaMA2's `audio_visual` branch ships the trust_remote_code model class
# + mm_infer() this example needs; it is NOT on PyPI (see videollama2_model.py
# docstring). `--no-deps` avoids its pinned transformers==4.42.3 (which would
# downgrade below evalrx' >=4.51 requirement) — only the lightweight
# extras it actually imports at inference time are installed explicitly
# (skips gradio/wandb/tensorboard/deepspeed/bitsandbytes/peft — training +
# serving-UI deps from upstream's requirements.txt this example never
# exercises). einops/einops-exts/imageio/imageio-ffmpeg/scenedetect/pysubs2/
# sentencepiece/shortuuid were missing here the first time this example was
# actually run end-to-end against real weights (caught live, not caught by
# the Dockerfile build itself since `import` failures are load-time, not
# build-time, for a --no-deps editable install).
#
# (An AVCD (arXiv:2505.20862) integration was tried and reverted here: its
# released kaistmm/AVCD fork of this same package produces degenerate
# garbage output under our transformers>=4.51 -- a real version-skew bug in
# their frozen custom attention patch, not something this example's wiring
# caused. See project chat history; abandoned in favor of AAD on
# Qwen2-Audio, examples/m1_m4/audiocaps_hallucination_qwen2_audio/.)
RUN git clone --branch audio_visual --depth 1 \
    https://github.com/DAMO-NLP-SG/VideoLLaMA2.git /videollama2_av \
    && pip install --no-cache-dir --no-deps -e /videollama2_av \
    && pip install --no-cache-dir decord pytorchvideo iopath portalocker \
       opencv-python-headless "moviepy==1.0.3" librosa timm \
       einops einops-exts imageio imageio-ffmpeg scenedetect pysubs2 \
       sentencepiece shortuuid

WORKDIR /app/example
COPY examples/m1_m4/musicavqa_videollama2/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
