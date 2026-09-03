"""Integration tests: real audio processors, no weights (network, no GPU).

Verifies the two things Phase 0 of the audio white-box pipeline (see
_resolve_audio / _check_audio_duration in hf_local.py) exists to get right:

  * audio-placeholder token count scales with clip duration (the encode path
    actually reaches the processor's audio= kwarg), and
  * the duration guard raises before a checkpoint's WhisperFeatureExtractor
    window would silently truncate a longer clip, rather than degrading
    quietly (empirically verified: Qwen2.5-Omni's window is 300s, Qwen2-Audio's
    is 30s — both read live off the processor here, never hard-coded).

Downloads only tokenizer/preprocessor configs (tens of MB, no model weights),
so this needs network but not a GPU or cached checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from evalrx.models.backends.hf_local import (
    AUDIO_SAMPLE_RATE,
    _check_audio_duration,
    _resolve_audio,
)

pytestmark = pytest.mark.network

transformers = pytest.importorskip("transformers")


def _sine(duration_sec: float) -> np.ndarray:
    n = int(duration_sec * AUDIO_SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / AUDIO_SAMPLE_RATE
    return (0.01 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _audio_token_count(proc, wav: np.ndarray) -> int:
    tok = proc.tokenizer
    content = [{"type": "audio"}, {"type": "text", "text": "What do you hear?"}]
    text = proc.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    enc = proc(text=[text], audio=[wav], return_tensors="pt", sampling_rate=AUDIO_SAMPLE_RATE)
    audio_tok_id = tok.convert_tokens_to_ids(proc.audio_token)
    return int((enc["input_ids"][0] == audio_tok_id).sum())


def test_resolve_audio_passes_ndarray_through_unchanged():
    wav = _sine(1.0)
    out = _resolve_audio(wav)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, wav)


@pytest.fixture(scope="module")
def qwen25_omni_processor():
    return transformers.Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")


@pytest.fixture(scope="module")
def qwen2_audio_processor():
    return transformers.Qwen2AudioProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")


def test_qwen25_omni_audio_token_count_scales_with_duration(qwen25_omni_processor):
    short = _audio_token_count(qwen25_omni_processor, _sine(3.0))
    long = _audio_token_count(qwen25_omni_processor, _sine(10.0))
    assert long > short
    assert long == pytest.approx(short * 10 / 3, rel=0.05)


def test_qwen2_audio_token_count_scales_with_duration(qwen2_audio_processor):
    # This is the encode path that actually matters: our _encode_vlm extension
    # emits a bare {"type": "audio"} block with no "audio_url"/"audio" payload
    # key. Qwen2-Audio's own cookbook examples set one; if its chat template
    # requires that key to emit <|audio_bos|><|AUDIO|>...<|audio_eos|>, a bare
    # block would silently produce zero placeholder tokens against real audio
    # features. This is the paper's actual TCD hyperparameter anchor (Appendix
    # A), so it gets the same real-encode check as Qwen2.5-Omni, not just the
    # pure-function duration-guard check below.
    short = _audio_token_count(qwen2_audio_processor, _sine(3.0))
    long = _audio_token_count(qwen2_audio_processor, _sine(10.0))
    assert short > 0
    assert long > short
    assert long == pytest.approx(short * 10 / 3, rel=0.05)


def test_qwen25_omni_duration_guard_matches_live_chunk_length(qwen25_omni_processor):
    chunk_length = qwen25_omni_processor.feature_extractor.chunk_length
    assert chunk_length == 300  # documented in the spec's caveats; pinned here so drift is caught
    _check_audio_duration([_sine(chunk_length - 1)], qwen25_omni_processor, "qwen2.5-omni-7b")
    with pytest.raises(ValueError, match="silently truncated"):
        _check_audio_duration([_sine(chunk_length + 1)], qwen25_omni_processor, "qwen2.5-omni-7b")


def test_qwen2_audio_duration_guard_matches_live_chunk_length(qwen2_audio_processor):
    chunk_length = qwen2_audio_processor.feature_extractor.chunk_length
    assert chunk_length == 30
    _check_audio_duration([_sine(chunk_length - 1)], qwen2_audio_processor, "qwen2-audio-7b-instruct")
    with pytest.raises(ValueError, match="silently truncated"):
        _check_audio_duration(
            [_sine(chunk_length + 1)], qwen2_audio_processor, "qwen2-audio-7b-instruct"
        )
