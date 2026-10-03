"""Media resolution shared by the local backends (torch-free).

``Inputs.image`` / ``Inputs.audio`` hold either a decoded object (PIL image,
mono waveform) or a path / URL the backend resolves. Both ``hf_local`` and
``jax_local`` need the same resolution, so it lives here rather than in either
backend module. ``hf_local`` re-exports the historical private names.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Sampling rate contract for ``Inputs.audio``: an ndarray is expected to already
#: be mono float32 at this rate (matches the WhisperFeatureExtractor every
#: audio-capable torch spec uses and the gemma library's audio front end). A
#: path/URL is decoded to it here, so callers never have to think about
#: resampling.
AUDIO_SAMPLE_RATE = 16000


def resolve_image(obj: Any) -> Any:
    """Return a PIL image for *obj* (PIL passes through; str/Path is opened)."""
    if hasattr(obj, "size") and hasattr(obj, "mode"):  # already PIL-like
        return obj
    from PIL import Image

    return Image.open(obj).convert("RGB")


def resolve_audio(obj: Any) -> Any:
    """Return a mono float32 waveform (numpy array) at :data:`AUDIO_SAMPLE_RATE`.

    An already-decoded array passes through UNCHANGED — the caller is on the
    hook for it being mono float32 @ 16 kHz; there is no signal here to detect
    or fix a mismatched sample rate, so getting this wrong fails silently
    downstream (the model just hears audio sped up/slowed down). A str/Path is
    decoded via ``ffmpeg`` (already assumed present elsewhere in this codebase,
    e.g. ``analysis/workbench.py``'s duration probing) rather than adding a new
    audio-decoding dependency.
    """
    if hasattr(obj, "dtype") and hasattr(obj, "shape"):  # already a numpy array
        import numpy as np

        arr = np.asarray(obj, dtype=np.float32)
        if arr.ndim > 1:
            # Silently flattening a (channels, samples) array would interleave
            # channels into one stream -- audio at the wrong speed with channel
            # aliasing, no error anywhere downstream. Mono is the documented
            # contract; enforce it instead of guessing which axis is channels.
            raise ValueError(
                f"Inputs.audio must be a 1-D mono waveform, got shape {arr.shape}; "
                "mix down to mono before passing it in"
            )
        return arr.reshape(-1)

    import shutil
    import subprocess

    import numpy as np

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "decoding audio from a path/URL requires the 'ffmpeg' binary on PATH; "
            "install it, or pass Inputs.audio as an already-decoded mono float32 "
            f"numpy array at {AUDIO_SAMPLE_RATE} Hz"
        )
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(obj),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        logger.warning("ffmpeg failed to decode audio %r: %s", obj, proc.stderr.decode(errors="replace"))
        raise RuntimeError(
            f"ffmpeg failed to decode audio {obj!r}: {proc.stderr.decode(errors='replace')}"
        )
    wav = np.frombuffer(proc.stdout, dtype="<f4").copy()
    logger.debug("decoded audio %r via ffmpeg: %.1fs @ %dHz", obj, len(wav) / AUDIO_SAMPLE_RATE, AUDIO_SAMPLE_RATE)
    return wav


def media_lists(inputs: Any, *, want_image: bool = True, want_audio: bool = True) -> tuple[list, list]:
    """``(images, audios)`` resolved from an :class:`~evalrx.core.case.Inputs`.

    ``inputs.video`` (a list of frames) takes priority over ``inputs.image`` and
    becomes one image per frame, exactly as ``hf_local`` does; a single image
    or a list of images is accepted; audio is one clip or a list of clips.
    """
    images: list = []
    audios: list = []
    if want_image:
        video = getattr(inputs, "video", None)
        image = getattr(inputs, "image", None)
        if video is not None:
            images = [resolve_image(f) for f in video]
        elif image is not None:
            raw = list(image) if isinstance(image, (list, tuple)) else [image]
            images = [resolve_image(i) for i in raw]
    if want_audio:
        audio = getattr(inputs, "audio", None)
        if audio is not None:
            raw = list(audio) if isinstance(audio, (list, tuple)) else [audio]
            audios = [resolve_audio(a) for a in raw]
    return images, audios
