"""ffmpeg helpers shared by the audio tasks (from the m1_m5 audio downloaders)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

#: Conservative clip cap: Qwen2-Audio's Whisper window is 30 s and Gemma 4's
#: audio tokeniser caps at 750 tokens x 40 ms = 30 s; the margin absorbs a clip
#: a few frames over after resampling. hf_local re-checks per processor.
MAX_DURATION_SEC = 29.5


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg/ffprobe must be on PATH to decode the embedded audio")


def duration_seconds(wav_path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def decode_to_wav(raw_bytes: bytes, wav_path: Path) -> float:
    """Embedded file bytes -> mono float32 16 kHz WAV (the pipeline's sample-rate contract)."""
    with tempfile.NamedTemporaryFile(suffix=".src", delete=False) as raw:
        raw.write(raw_bytes)
        raw_path = raw.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", raw_path,
             "-ac", "1", "-ar", "16000", "-f", "wav", str(wav_path)],
            check=True, capture_output=True,
        )
    finally:
        Path(raw_path).unlink(missing_ok=True)
    return duration_seconds(wav_path)
