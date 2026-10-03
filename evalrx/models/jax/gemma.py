"""Gemma 4 on JAX: the reference adapter for the ``jax_local`` backend.

Runs the Gemma 4 E2B / E4B checkpoints through Google DeepMind's ``gemma``
library (Flax Linen) behind the :class:`~evalrx.models.jax.protocol.JaxModelAdapter`
contract: text, image and audio inputs, read access (logits, hidden states,
attention probabilities), greedy / sampled generation, teacher-forced logprobs
(phases 1 and 3 of ``docs/design_jax_backend.md``; the L3a executors, sites and
LoRA are still open).

Facts this file depends on, verified against gemma 4.0.1 / flax 0.12.10 on
2026-09-26 (text) and 2026-09-28 (image, audio); design doc, section 4:

* blocks are ``layer_{i}`` (35 on E2B); ``final_norm`` is a plain RMSNorm,
  ``x * rsqrt(mean(x^2) + 1e-6) * scale``; logits are
  ``x @ embedder.input_embedding.T`` followed by a tanh soft-cap at 30;
* attention is a materialised softmax whose probabilities pass through an
  identity module named ``attention_weights`` inside each block's ``attn``, so
  Flax ``capture_intermediates`` reads them without touching the model code;
  the same mechanism captures the root's ``_encode_and_get_inputs`` (Flax wraps
  private methods too), whose ``embeddings`` are the block-0 input WITH the
  image / audio embeddings merged in, i.e. HF's ``hidden_states[0]``;
* the library's ``Transformer.__call__`` is wrapped in ``nn.jit``, whose cached
  trace bakes in the FIRST capture filter it sees (a second call with another
  filter silently returns the first filter's intermediates). This adapter
  therefore calls the layer beneath that wrapper through its own
  per-configuration ``jax.jit``;
* with images, ``__call__`` (``return_last_only=False``) additionally runs a
  legacy ``remove_mm_logits`` step written for fixed-count Gemma 3 images; on
  the variable-count Gemma 4 tokens it garbles the sequence axis, so the media
  forward here calls the library's ``_encode_and_get_inputs`` +
  ``_apply_attention`` + ``embedder.decode`` + soft-cap directly (the same
  code path, minus that step);
* images: the text carries one ``<|image|>`` (id 258880) per image; before the
  forward it is expanded to ``\\n\\n <|image> P*n <image|> \\n\\n`` with ``n``
  soft tokens per image (``P`` = the library's internal -2 placeholder). The
  vision encoder resizes each image to a multiple of 48 px per side keeping
  the aspect ratio, so ``n = (H/48) * (W/48)`` exactly and the pooled tokens
  come out row-major, which gives ``TokenTypeMap.grids`` for free. ``n`` is
  bounded by ``config.vision_encoder.num_mm_tokens_per_image`` (280 on E2B / E4B):
  the library's ``Gemma4Sampler`` DEFAULTS (``max_soft_tokens=1120``) do not
  match that encoder (it then reserves ~1100 slots for ~270 pooled tokens), so
  every image constant here is read from the model config;
* audio: 16 kHz mono float32; one ``<|audio|>`` (id 258881) per clip, expanded
  to ``<|audio> A*m <audio|>`` with ``m`` = mel frames (20 ms / 10 ms hop) after
  two stride-2 subsamplings, capped at 750 tokens (~30 s; longer clips raise
  here like hf_local's ``_check_audio_duration``). The conformer runs on the raw
  waveform (``audio_encoder``); ``audio_lengths`` masks padding;
* the multimodal towers and their projections are kept float32 by the library
  (``initialize_param_with_dtype`` excludes them); the bf16 cast below skips the
  same paths;
* right-padding with PAD (0) leaves valid positions' outputs unchanged (max
  abs diff 1e-7 on a random model), so lengths are bucketed to bound recompiles;
* chat format (``dialog.Format.GEMMA4``): ``<|turn>user\\n...<turn|>\\n<|turn>model\\n``,
  media placeholders precede the text inside the user turn (hf_local's block
  order: audio, images, text). Thinking is switched on by the ``<|think|>``
  control token (id 98); the specs' ``enable_thinking=False`` therefore renders
  none of it. The exact placement the HF ``chat_template.jinja`` uses for
  ``enable_thinking=True`` was not verifiable here (gated repo), so that
  setting raises for now;
* checkpoints: public ``gs://gemma-data/checkpoints/gemma4-{e2b,e4b}-it``
  (Orbax; 17 GB for E2B including the 167 M-param vision and 305 M-param audio
  towers, which ``text_only=True`` skips); tokenizer
  ``gs://gemma-data/tokenizers/tokenizer_gemma4.model`` (SentencePiece, 4.5 MB).
  Both read anonymously; a local mirror is picked up through
  ``RuntimeConfig.engine_kwargs["checkpoint"]`` (a ``tokenizer_gemma4.model``
  next to it, or in it, is used automatically).

``engine_kwargs`` understood: ``checkpoint``, ``tokenizer``, ``model_class``,
``pad_buckets``, ``text_only`` (default: False when the spec declares vision or
audio, True otherwise), ``audio_seq_length`` (750).

jax and gemma are imported lazily inside ``load()``.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
import warnings
from typing import Any

from evalrx.core.case import Inputs
from evalrx.models._media import AUDIO_SAMPLE_RATE, media_lists
from evalrx.models.jax.protocol import (
    CAPTURE_ATTN,
    CAPTURE_HIDDEN,
    CAPTURE_LOGITS,
    Encoding,
    ForwardOut,
    GenerateOut,
    NormParams,
    SamplingParams,
)

logger = logging.getLogger(__name__)

PAD_ID = 0
DEFAULT_BUCKETS = (128, 256, 512, 1024, 2048, 4096)
#: generation output-buffer sizes (static under jit; see ``generate``)
OUT_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
TOKENIZER_FILENAME = "tokenizer_gemma4.model"
#: the library's internal placeholders for merged media embeddings (never real ids)
IMAGE_SOFT_PLACEHOLDER = -2
AUDIO_SOFT_PLACEHOLDER = -4
#: what the tokenizer emits for one image / one clip in the text; also the id the
#: Trace reports at every soft-token position (hf_local's image_token_id role)
IMAGE_PLACEHOLDER_ID = 258880      # <|image|>
AUDIO_PLACEHOLDER_ID = 258881      # <|audio|>
DEFAULT_AUDIO_SEQ_LENGTH = 750     # soft tokens; ~30 s at 16 kHz (Gemma4Sampler default)
#: parameter sub-trees the library keeps float32 even under a bf16 model dtype
_MM_FLOAT32_PREFIXES = (
    "vision_encoder", "audio_encoder",
    "embedder/mm_input_projection", "embedder/mm_pre_projection_norm",
    "embedder/audio_input_projection", "embedder/audio_soft_embedding_norm",
)
_DTYPES = {
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float32": "float32", "fp32": "float32",
    "float16": "float16", "fp16": "float16",
}
_IMAGE_BLOCKS = ("image", "image_url", "video")
_AUDIO_BLOCKS = ("audio", "input_audio")


def make_adapter(spec, runtime) -> "GemmaJaxAdapter":
    """``JaxSpec.adapter`` factory."""
    return GemmaJaxAdapter(spec, runtime)


def bucket_length(n: int, buckets=DEFAULT_BUCKETS) -> int:
    """Smallest bucket >= *n*; past the last bucket, round up to its multiple."""
    for b in buckets:
        if n <= b:
            return int(b)
    step = int(buckets[-1])
    return int(math.ceil(n / step) * step)


def audio_soft_token_count(n_samples: int, *, sample_rate: int = AUDIO_SAMPLE_RATE) -> int:
    """Soft tokens the audio tower emits for a clip of *n_samples* (uncapped).

    Mirrors ``Gemma4Sampler.sample``: 20 ms frames on a 10 ms hop (the extra
    +1 sample is the library's unfold quirk), then two stride-2 subsampling
    stages (kernel 3, padding 1).
    """
    frame = int(round(sample_rate * 20.0 / 1000.0))
    hop = int(round(sample_rate * 10.0 / 1000.0))
    t = (int(n_samples) - (frame + 1)) // hop + 1
    for _ in range(2):
        t = (t + 2 - 3) // 2 + 1
    return int(t)


def image_grid(height: int, width: int, *, patch_size: int, max_soft_tokens: int,
               pooling_kernel_size: int) -> tuple[int, int]:
    """``(rows, cols)`` of pooled soft tokens for an image of *height* x *width*.

    The encoder resizes to multiples of ``patch_size * pooling_kernel_size`` per
    side (aspect ratio kept, total patches <= budget), then pools k x k patches
    row-major, so the soft-token count is exactly ``rows * cols``.
    """
    from gemma.gm.nn.gemma4.vision import _preprocessing as vp

    th, tw = vp.get_target_dimensions(
        int(height), int(width), patch_size=patch_size,
        max_patches=max_soft_tokens * pooling_kernel_size**2,
        pooling_kernel_size=pooling_kernel_size,
    )
    side = patch_size * pooling_kernel_size
    return int(th // side), int(tw // side)


def _content_blocks(content: Any) -> tuple[str, list, list]:
    """``(text, images, audios)`` of an OpenAI / transformers-style message content.

    Media blocks render as the Gemma 4 placeholders in the order they appear;
    ``{"type": "image", "image": <PIL | path>}`` / ``{"type": "audio", "audio": ...}``
    blocks also hand back their payloads (``None`` payloads are placeholders
    only, the way hf_local's ``_encode_vlm`` builds its content).
    """
    if isinstance(content, str):
        return content, [], []
    if content is None:
        return "", [], []
    if not isinstance(content, list):
        return str(content), [], []
    parts: list[str] = []
    images: list = []
    audios: list = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        kind = block.get("type")
        if kind in _IMAGE_BLOCKS:
            parts.append("<|image|>")
            payload = block.get("image", block.get("image_url", block.get("video")))
            if payload is not None and not isinstance(payload, dict):
                images.append(payload)
        elif kind in _AUDIO_BLOCKS:
            parts.append("<|audio|>")
            payload = block.get("audio", block.get("input_audio"))
            if payload is not None and not isinstance(payload, dict):
                audios.append(payload)
        else:
            parts.append(str(block.get("text", "")))
    return "".join(parts), images, audios


def _below_flax(fn):
    """The first wrapper level under flax's ``nn.jit`` on a module method.

    ``Transformer.__call__`` is ``nn.jit(flatten_unflatten_batch_dim(typechecked(body)))``;
    this returns the ``flatten_unflatten_batch_dim`` level, keeping the library's
    own batch handling and type checks while escaping the shared jit cache.
    """
    f = fn
    while f is not None:
        code = getattr(f, "__code__", None)
        if code is not None and f"{os.sep}flax{os.sep}" not in code.co_filename:
            return f
        f = getattr(f, "__wrapped__", None)
    raise RuntimeError("could not find the un-jitted Transformer.__call__ beneath flax's wrapper")


def _media_forward(mdl, tokens, images, audio, audio_lengths, audio_soft_token_counts):
    """``Transformer.__call__`` for expanded multimodal tokens, minus the legacy
    ``remove_mm_logits`` step (see the module docstring). Returns the full-sequence
    soft-capped logits and the merged block-0 input embeddings."""
    import jax.numpy as jnp

    inputs = mdl._encode_and_get_inputs(  # noqa: SLF001 - the library's own building blocks
        tokens=tokens, images=images, audio=audio, audio_lengths=audio_lengths,
        audio_soft_token_counts=audio_soft_token_counts,
    )
    x, _ = mdl._apply_attention(inputs, None)  # noqa: SLF001
    logits = mdl.embedder.decode(x)
    cap = mdl.config.final_logit_softcap
    if cap is not None:
        logits = jnp.tanh(logits / cap) * cap
    return logits, inputs.embeddings


class GemmaJaxAdapter:
    """Gemma 4 E2B / E4B through the ``gemma`` library: text, image and audio."""

    def __init__(self, spec, runtime) -> None:
        js = getattr(spec, "jax", None)
        if js is None or js.framework != "gemma":
            raise ValueError(
                f"{getattr(spec, 'key', spec)!r}: GemmaJaxAdapter needs ModelSpec.jax with framework='gemma'"
            )
        kw = dict(getattr(runtime, "engine_kwargs", None) or {})
        self.spec = spec
        self.runtime = runtime
        self.checkpoint = str(kw.pop("checkpoint", js.checkpoint))
        self.tokenizer_path = str(
            kw.pop("tokenizer", None) or self._sibling_tokenizer(self.checkpoint) or js.tokenizer
        )
        self.model_class = str(kw.pop("model_class", None) or js.model_class or "Gemma4_E2B")
        self.buckets = tuple(sorted(int(b) for b in kw.pop("pad_buckets", DEFAULT_BUCKETS)))
        self.audio_seq_length = int(kw.pop("audio_seq_length", DEFAULT_AUDIO_SEQ_LENGTH))
        spec_mods = set(getattr(spec, "modalities", None) or {"text"})
        has_media = bool(spec_mods - {"text"})
        text_only = kw.pop("text_only", None)
        # the towers load unless the spec is text-only or the caller opts out
        # (the llm benchmark cells do: ~2 GB float32 and load time saved)
        self.text_only = bool(text_only) if text_only is not None else not has_media
        self.modalities = frozenset({"text"}) if self.text_only else frozenset(spec_mods | {"text"})
        # fused kernels never materialise probabilities; the library's softmax path
        # does, so ATTENTION is on unless the spec or attn_impl says otherwise
        self.reference_attention = bool(js.reference_attention) and (
            getattr(runtime, "attn_impl", None) in (None, "eager")
        )
        if kw:
            logger.warning("GemmaJaxAdapter: ignoring engine_kwargs %s", sorted(kw))
        self._model = None
        self._params = None
        self._tok = None
        self._pieces: list[str] = []
        self._special_ids: set[int] = set()
        self._raw_call = None
        self._n_layers: int | None = None
        self._forward_cache: dict = {}
        # fresh sampler seeds for calls without SamplingParams.seed (hf_local's
        # sampled generates differ call to call through torch's global RNG)
        self._seed_source = random.Random()

    # -- construction helpers --------------------------------------------
    @staticmethod
    def _sibling_tokenizer(checkpoint: str) -> str | None:
        if "://" in checkpoint:
            return None
        d = os.path.abspath(checkpoint)
        for cand in (os.path.join(d, TOKENIZER_FILENAME), os.path.join(os.path.dirname(d), TOKENIZER_FILENAME)):
            if os.path.isfile(cand):
                return cand
        return None

    def _model_cls(self):
        from gemma import gm

        try:
            return getattr(gm.nn, self.model_class)
        except AttributeError as exc:
            raise ValueError(
                f"the gemma library has no model class {self.model_class!r} (JaxSpec.model_class)"
            ) from exc

    @property
    def n_layers(self) -> int:
        if self._n_layers is None:
            self._n_layers = int(self._model_cls().config.num_layers)
        return self._n_layers

    @property
    def loaded(self) -> bool:
        return self._params is not None

    def _ensure_loaded(self) -> None:
        if not self.loaded:
            self.load()

    # -- load ---------------------------------------------------------------
    def load(self) -> None:
        import jax
        import jax.numpy as jnp
        from flax.traverse_util import flatten_dict, unflatten_dict
        from gemma import gm

        t0 = time.monotonic()
        dtype_name = _DTYPES.get(str(getattr(self.runtime, "dtype", "bfloat16")).lower(), "bfloat16")
        dtype = getattr(jnp, dtype_name)
        cls = self._model_cls()
        self._model = cls(text_only=self.text_only, dtype=dtype)
        self._n_layers = int(self._model.config.num_layers)
        self._raw_call = _below_flax(type(self._model).__call__)
        self._tok = gm.text.Gemma4Tokenizer(path=self.tokenizer_path)
        self._pieces = list(self._tok.tokens)
        self._special_ids = self._collect_special_ids(self._tok, self._pieces)
        # restore on the host and cast there: the public checkpoints are float32
        # (20.5 GB for E2B with its towers) and load_params' default sharding
        # replicates them as stored onto every device, which peaks at 29.8 GB on
        # the accelerator before the cast below (a 16 GB TPU v5e chip cannot load)
        try:
            host = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])
        except RuntimeError:  # JAX_PLATFORMS set without cpu by the caller
            host = None
            logger.warning("GemmaJaxAdapter: no jax cpu backend (JAX_PLATFORMS=%r); the float32 "
                           "checkpoint is restored straight onto the accelerator",
                           os.environ.get("JAX_PLATFORMS"))
        params = gm.ckpts.load_params(self.checkpoint, text_only=self.text_only, sharding=host)
        if not self.text_only:
            # the library scatters float32 tower outputs into the bf16 text
            # embeddings (merge_flat_embeddings); jax warns about the implicit
            # cast on every media forward. The cast is the library's design
            # (towers float32, residual stream in the model dtype), so silence it.
            warnings.filterwarnings(
                "ignore", message="scatter inputs have incompatible types", category=FutureWarning,
            )
        # the public checkpoints store float32 (19.8 GB for text-only E2B); honour
        # RuntimeConfig.dtype the way hf_local's torch_dtype does (bf16 = 9.9 GB),
        # except for the media towers and projections the library keeps float32
        flat = flatten_dict(params)
        del params  # each float32 leaf is freed as soon as its cast replaces it
        kept = 0
        for path in list(flat):
            a = flat[path]
            if not (jnp.issubdtype(a.dtype, jnp.floating) and a.dtype != dtype):
                continue
            joined = "/".join(str(p) for p in path)
            if joined.startswith(_MM_FLOAT32_PREFIXES):
                kept += 1
                continue
            flat[path] = a.astype(dtype)
            del a
        # one host-to-device copy of the cast tree (11.3 GB peak for E2B)
        params = jax.device_put(unflatten_dict(flat), jax.devices()[0])
        del flat
        self._params = params
        self._forward_cache.clear()
        n_bytes = sum(int(a.nbytes) for a in jax.tree.leaves(params))
        logger.info(
            "GemmaJaxAdapter: %s loaded from %s in %.0fs (%.1f GB params, %s, %d layers, "
            "modalities=%s, %d float32 media arrays, reference_attention=%s, jax backend=%s)",
            self.model_class, self.checkpoint, time.monotonic() - t0, n_bytes / 1e9, dtype_name,
            self._n_layers, sorted(self.modalities), kept, self.reference_attention, jax.default_backend(),
        )

    @staticmethod
    def _collect_special_ids(tok, pieces: list[str]) -> set[int]:
        """Ids ``decode`` skips, the way ``skip_special_tokens=True`` would: the
        enum specials plus the low-id control pieces (``<|turn>``, ``<|channel>``,
        ``<|think|>``, ``<unusedNN>``...)."""
        ids = {int(v) for v in tok.special_tokens.__members__.values()}
        for i, piece in enumerate(pieces[:512]):
            if piece.startswith("<") and piece.endswith(">"):
                ids.add(i)
        return ids

    # -- media constants (from the model config, never the sampler defaults) ----
    def _vision_settings(self) -> dict:
        ve = self._model.config.vision_encoder
        if ve is None:
            raise ValueError(
                f"{self.model_class} was loaded text_only (no vision tower); pass "
                "RuntimeConfig(engine_kwargs={'text_only': False}) for image inputs"
            )
        return {
            "patch_size": int(ve.patch_size),
            "max_soft_tokens": int(ve.num_mm_tokens_per_image),
            "pooling_kernel_size": int(ve.pooling_kernel_size),
        }

    def _require_audio_tower(self) -> None:
        if self._model.config.audio_encoder is None:
            raise ValueError(
                f"{self.model_class} was loaded text_only (no audio tower); pass "
                "RuntimeConfig(engine_kwargs={'text_only': False}) for audio inputs"
            )

    def _audio_soft_tokens(self, wav) -> int:
        n = int(len(wav))
        count = audio_soft_token_count(n)
        if count > self.audio_seq_length:
            window_s = self.audio_seq_length * 4 * (AUDIO_SAMPLE_RATE // 100) / AUDIO_SAMPLE_RATE
            raise ValueError(
                f"{self.spec.key}: audio is {n / AUDIO_SAMPLE_RATE:.1f}s, longer than this "
                f"checkpoint's ~{window_s:.0f}s encoder window ({self.audio_seq_length} soft tokens) — "
                "it would be silently truncated. Chunk the audio yourself before calling."
            )
        return max(1, count)

    # -- text rendering ---------------------------------------------------
    def _thinking_requested(self) -> bool:
        kwargs = dict(getattr(self.spec, "chat_template_kwargs", None) or {})
        return bool(kwargs.get("enable_thinking", False))

    def render_messages(self, messages: list, *, tools: list | None = None) -> str:
        """Gemma 4 turn format via the ``dialog`` package (``Format.GEMMA4``),
        ending with an open ``<|turn>model\\n`` for the model to fill. Media
        blocks render as ``<|image|>`` / ``<|audio|>`` placeholders."""
        import dialog

        if tools:
            raise NotImplementedError("GemmaJaxAdapter: tool rendering is not implemented")
        if self._thinking_requested():
            raise NotImplementedError(
                "GemmaJaxAdapter renders thinking OFF only: where Gemma 4's official template "
                "places the <|think|> control token for enable_thinking=True is not verified here; "
                "run with chat_template_kwargs={'enable_thinking': False} (the specs' default)"
            )
        turns = []
        for m in messages:
            role = str(m.get("role", "user")).lower()
            text, _images, _audios = _content_blocks(m.get("content"))
            if role == "system":
                turns.append(dialog.System(text))
            elif role in ("assistant", "model"):
                turns.append(dialog.Model(text))
            else:
                turns.append(dialog.User(text))
        return dialog.Conversation(*turns).as_text(format=dialog.Format.GEMMA4)

    def _token_str(self, i: int) -> str:
        s = self._tok.decode([int(i)])
        return s if s else self._pieces[int(i)]

    def _text(self, ids: list[int]) -> str:
        return self._tok.decode([int(i) for i in ids if int(i) not in self._special_ids])

    # -- encoding -----------------------------------------------------------
    def _preprocess_images(self, images: list):
        """``(PreprocessedVisionInput, soft_token_counts, grids)`` the library way,
        with the image budget read from the vision encoder."""
        import jax.numpy as jnp
        import numpy as np
        from gemma.gm.nn.gemma4 import _transformer as g4
        from gemma.gm.nn.gemma4.vision import _preprocessing as vp

        settings = self._vision_settings()
        arrays = [np.asarray(im.convert("RGB")) if hasattr(im, "convert") else np.asarray(im) for im in images]
        patches, positions_xy, counts = vp.preprocess_and_patchify(arrays, **settings)
        n_images, max_patches = patches.shape[0], patches.shape[1]
        vision_input = g4.PreprocessedVisionInput(
            patches=jnp.reshape(patches, (1, n_images * max_patches, patches.shape[2])),
            positions_xy=jnp.reshape(positions_xy, (1, n_images * max_patches, positions_xy.shape[2])),
            soft_token_counts=tuple(int(c) for c in counts),
        )
        grids = []
        for a, c in zip(arrays, counts):
            rows, cols = image_grid(a.shape[0], a.shape[1], **settings)
            if rows * cols != int(c):  # the encoder changed its pooling rule: fall back to "unknown"
                logger.warning("GemmaJaxAdapter: image grid %dx%d != %d soft tokens; grids left empty",
                               rows, cols, c)
                grids = []
                break
            grids.append((1, rows, cols))
        return vision_input, [int(c) for c in counts], grids, arrays

    def _encode_text(self, text: str, images: list | None = None, audios: list | None = None) -> Encoding:
        import numpy as np
        from gemma.gm.vision import _token_utils as tu

        row = np.asarray([int(i) for i in self._tok.encode(text, add_bos=True)], dtype=np.int32)
        media: dict[str, Any] = {}
        grids: list[tuple[int, int, int]] = []
        if images:
            vision_input, counts, grids, arrays = self._preprocess_images(images)
            row = tu.add_variable_extra_tokens_for_images(row[None], soft_token_counts=counts)[0]
            media["vision"] = vision_input
            media["images"] = arrays
        if audios:
            import jax.numpy as jnp

            self._require_audio_tower()
            wavs = [np.asarray(a, dtype=np.float32).reshape(-1) for a in audios]
            counts_a = [self._audio_soft_tokens(w) for w in wavs]
            row = tu.add_variable_extra_tokens_for_audio(row[None], soft_token_counts=counts_a)[0]
            longest = max(len(w) for w in wavs)
            padded = np.zeros((len(wavs), longest), dtype=np.float32)
            for i, w in enumerate(wavs):
                padded[i, : len(w)] = w
            media["audio"] = jnp.asarray(padded)[None]                                  # (1, N, S)
            media["audio_lengths"] = jnp.asarray([len(w) for w in wavs], dtype=jnp.int32)[None]  # (1, N)
            media["audio_soft_token_counts"] = tuple(counts_a)
            media["audios"] = wavs
        image_mask = row == IMAGE_SOFT_PLACEHOLDER
        audio_mask = row == AUDIO_SOFT_PLACEHOLDER
        public = row.copy()
        public[image_mask] = IMAGE_PLACEHOLDER_ID
        public[audio_mask] = AUDIO_PLACEHOLDER_ID
        ids = [int(i) for i in public.tolist()]
        return Encoding(
            ids=ids,
            tokens=[self._token_str(i) for i in ids],
            text=text,
            media=media,
            image_token_mask=[bool(v) for v in image_mask] if images else None,
            audio_token_mask=[bool(v) for v in audio_mask] if audios else None,
            grids=grids,
            image_token_id=IMAGE_PLACEHOLDER_ID if images else None,
        )

    def _check_modalities(self, images: list, audios: list) -> None:
        if images and "image" not in self.modalities:
            raise ValueError(
                f"{self.spec.key}: image inputs need the vision tower; this adapter was built "
                f"with modalities={sorted(self.modalities)} (text_only={self.text_only})"
            )
        if audios and "audio" not in self.modalities:
            raise ValueError(
                f"{self.spec.key}: audio inputs need the audio tower; this adapter was built "
                f"with modalities={sorted(self.modalities)} (text_only={self.text_only})"
            )

    # -- protocol: encoding ---------------------------------------------------
    def encode(self, inputs: Any, *, chat_template: bool) -> Encoding:
        self._ensure_loaded()
        inputs = inputs if isinstance(inputs, Inputs) else Inputs(prompt=str(inputs))
        images, audios = media_lists(inputs)
        self._check_modalities(images, audios)
        # hf_local's block order inside the one user turn: audio, images, text
        content: list = (
            [{"type": "audio"} for _ in audios] + [{"type": "image"} for _ in images]
            + [{"type": "text", "text": inputs.prompt}]
        )
        if chat_template:
            text = self.render_messages([{"role": "user", "content": content}])
        else:
            text, _, _ = _content_blocks(content)
        return self._encode_text(text, images=images, audios=audios)

    def render_chat(self, messages: list, tools: list | None = None) -> Encoding:
        self._ensure_loaded()
        from evalrx.models._media import resolve_audio, resolve_image

        images: list = []
        audios: list = []
        for m in messages:
            _t, ims, auds = _content_blocks(m.get("content"))
            images.extend(resolve_image(i) for i in ims)
            audios.extend(resolve_audio(a) for a in auds)
        self._check_modalities(images, audios)
        return self._encode_text(self.render_messages(messages, tools=tools), images=images, audios=audios)

    def decode(self, ids: list[int]) -> str:
        """One id -> its surface form (specials kept visible, for token labels);
        several -> text with specials skipped, like ``skip_special_tokens=True``."""
        self._ensure_loaded()
        ids = [int(i) for i in ids]
        if len(ids) == 1:
            return self._token_str(ids[0])
        return self._text(ids)

    # -- protocol: read internals ----------------------------------------------
    def _capture_filter(self, want_h: bool, want_a: bool, layer_ids: tuple, hidden_ids: tuple):
        n = self.n_layers
        layer_set, hidden_set = set(layer_ids), set(hidden_ids)

        def filt(mdl, method: str) -> bool:
            path = tuple(mdl.path)
            if method == "_encode_and_get_inputs":          # root: merged block-0 input = hidden[0]
                return want_h and path == () and 0 in hidden_set
            if method != "__call__":
                return False
            if len(path) == 1:
                name = path[0]
                if name.startswith("layer_"):
                    return want_h and (int(name[6:]) + 1) in hidden_set
                if name == "final_norm":
                    return want_h and n in hidden_set
                return False
            return (
                want_a and len(path) == 3 and path[2] == "attention_weights" and path[1] == "attn"
                and path[0].startswith("layer_") and int(path[0][6:]) in layer_set
            )

        return filt

    def _forward_fn(self, want_h: bool, want_a: bool, layer_ids: tuple, hidden_ids: tuple, media: bool):
        """A ``jax.jit``-ed forward for one capture configuration (cached)."""
        key = (want_h, want_a, layer_ids if want_a else (), hidden_ids if want_h else (), media)
        fn = self._forward_cache.get(key)
        if fn is not None:
            return fn
        import jax

        model, raw = self._model, self._raw_call
        filt = self._capture_filter(want_h, want_a, layer_ids, hidden_ids)
        capture = {"capture_intermediates": filt, "mutable": ["intermediates"]} if (want_h or want_a) else {}

        if media:
            def fwd(params, tokens, images, audio, audio_lengths, audio_soft_token_counts):
                out = model.apply(
                    {"params": params}, tokens, images, audio, audio_lengths, audio_soft_token_counts,
                    method=_media_forward, **capture,
                )
                (logits, embeddings), state = out if capture else (out, {})
                return logits, embeddings, state.get("intermediates", {}) if state else {}

            fn = jax.jit(fwd, static_argnames=("audio_soft_token_counts",))
        else:
            def fwd(params, tokens):
                out = model.apply({"params": params}, tokens, return_last_only=False, method=raw, **capture)
                out, state = out if capture else (out, {})
                return out.logits, None, state.get("intermediates", {}) if state else {}

            fn = jax.jit(fwd)
        self._forward_cache[key] = fn
        return fn

    def _model_tokens(self, enc: Encoding, length: int):
        """The library's token row: public ids with the soft positions set back to
        the internal placeholders, right-padded with PAD to *length*."""
        import numpy as np

        seq = len(enc.ids)
        tokens = np.full((1, length), PAD_ID, dtype=np.int32)
        tokens[0, :seq] = np.asarray(enc.ids, dtype=np.int32)
        if enc.image_token_mask is not None:
            tokens[0, :seq][np.asarray(enc.image_token_mask, dtype=bool)] = IMAGE_SOFT_PLACEHOLDER
        if enc.audio_token_mask is not None:
            tokens[0, :seq][np.asarray(enc.audio_token_mask, dtype=bool)] = AUDIO_SOFT_PLACEHOLDER
        return tokens

    def forward(
        self,
        enc: Encoding,
        *,
        capture: frozenset[str],
        layers: tuple[int, ...] | None = None,
    ) -> ForwardOut:
        import jax.numpy as jnp
        from flax.traverse_util import flatten_dict

        self._ensure_loaded()
        n = self.n_layers
        seq = len(enc.ids)
        length = bucket_length(seq, self.buckets)
        tokens = jnp.asarray(self._model_tokens(enc, length))
        want_h = CAPTURE_HIDDEN in capture
        want_a = CAPTURE_ATTN in capture and self.reference_attention
        if layers is None:
            layer_ids, hidden_ids = tuple(range(n)), tuple(range(n + 1))
        else:
            layer_ids = tuple(sorted({int(i) for i in layers if 0 <= int(i) < n}))
            hidden_ids = tuple(sorted({int(i) for i in layers if 0 <= int(i) <= n}))
        media = enc.media or {}
        has_media = "vision" in media or "audio" in media
        fn = self._forward_fn(want_h, want_a, layer_ids, hidden_ids, has_media)
        if has_media:
            logits_all, embeddings, inter = fn(
                self._params, tokens, media.get("vision"), media.get("audio"), media.get("audio_lengths"),
                media.get("audio_soft_token_counts"),
            )
        else:
            logits_all, embeddings, inter = fn(self._params, tokens)
        inter = flatten_dict(inter) if inter else {}

        def pick(key):
            v = inter.get(key)
            return None if v is None else v[0]

        logits = logits_all[0, :seq] if CAPTURE_LOGITS in capture else None
        hidden = attn = None
        if want_h:
            hidden = [None] * (n + 1)
            emb = pick(("_encode_and_get_inputs",))
            if emb is not None:
                hidden[0] = emb.embeddings[0, :seq]
            elif embeddings is not None and 0 in hidden_ids:
                hidden[0] = embeddings[0, :seq]
            for i in range(n):
                v = pick((f"layer_{i}", "__call__"))          # (cache, x)
                if v is not None:
                    hidden[i + 1] = v[1][0, :seq]
            normed = pick(("final_norm", "__call__"))
            if normed is not None:
                hidden[n] = normed[0, :seq]                  # HF layout: last entry is post final-norm
        if want_a:
            attn = [None] * n
            for i in range(n):
                p = pick((f"layer_{i}", "attn", "attention_weights", "__call__"))   # (1, T, H, S)
                if p is not None:
                    attn[i] = jnp.transpose(p[0, :seq, :, :seq], (1, 0, 2))       # -> (H, S, S)
        extras: dict[str, Any] = {}
        if enc.grids and len({(g[1], g[2]) for g in enc.grids}) == 1:
            extras["image_spatial_shape"] = (int(enc.grids[0][1]), int(enc.grids[0][2]))
        return ForwardOut(logits=logits, hidden=hidden, attn=attn, extras=extras)

    def unembed(self):
        self._ensure_loaded()
        return self._params["embedder"]["input_embedding"]          # (V, D); decode is x @ E.T

    def final_norm_params(self) -> NormParams:
        self._ensure_loaded()
        return NormParams(scale=self._params["final_norm"]["scale"], eps=1e-6, plus_one=False)

    # -- protocol: generation ---------------------------------------------------
    @staticmethod
    def _sampling_method(p: SamplingParams):
        from gemma import gm

        if p.greedy:
            return gm.text.Greedy()
        # the library has no combined top-p + top-k method: nucleus wins when both
        # are set (top_k is then ignored), top-k alone otherwise
        if 0.0 < p.top_p < 1.0:
            return gm.text.TopPSampling(p=float(p.top_p), temperature=float(p.temperature))
        if p.top_k and int(p.top_k) > 0:
            return gm.text.TopkSampling(k=int(p.top_k), temperature=float(p.temperature))
        return gm.text.RandomSampling(temperature=float(p.temperature))

    def _rng_seed(self, params: SamplingParams) -> int:
        """Sampler seed: the caller's ``seed`` when given, else a fresh draw. A
        constant default made every sampled call return the same text, so M1's
        self-consistency / coverage probes saw zero variance and M3 diagnosed
        "the harness is not sampling" (chartqa shakedown, 2026-10-02)."""
        if params.seed is not None:
            return int(params.seed)
        if params.greedy:
            return 0                                   # unused by Greedy; keep the trace stable
        return self._seed_source.randrange(1 << 31)

    def _strip_generated(self, predicted: list[int]) -> list[int]:
        st = self._tok.special_tokens
        end = {PAD_ID, int(st.EOS), int(st.END_OF_TURN), int(st.BEGIN_OF_TOOL_RESPONSE)}
        out: list[int] = []
        for t in predicted:
            if t in end:
                break
            out.append(int(t))
        return out

    def generate(self, enc: Encoding, params: SamplingParams) -> GenerateOut:
        import numpy as np
        from gemma import gm

        self._ensure_loaded()
        # the sampler tokenises text itself (add_bos=True, like encode) and expands
        # the media placeholders with the same functions, so feed it the very
        # string and media the forward pass saw; enc.ids is already expanded
        text = enc.text if enc.text is not None else self._tok.decode(enc.ids)
        media = enc.media or {}
        max_new = max(1, int(params.max_new_tokens))
        pad_len = bucket_length(len(enc.ids), self.buckets)
        # max_out_length sizes the output buffer, a static shape in the library's
        # prefill and decode jits; max_new_tokens is dynamic. Bucketing the first
        # keeps every max_tokens a stage asks for (24, 64, 256, ...) on a few
        # compiled programs instead of one compile per distinct value
        max_out = bucket_length(max_new, OUT_BUCKETS)
        cache_len = bucket_length(pad_len + max_out + 1, self.buckets)
        extra: dict[str, Any] = {}
        if "vision" in media:
            extra.update(self._vision_settings())
        sampler = gm.text.Gemma4Sampler(
            model=self._model, params=self._params, tokenizer=self._tok,
            sampling=self._sampling_method(params),
            cache_length=cache_len, max_out_length=max_out, pad_length=pad_len,
            audio_seq_length=self.audio_seq_length, **extra,
        )
        rng = self._rng_seed(params)
        out = sampler.sample(
            text, images=media.get("images") or None, audio=media.get("audios") or None,
            max_new_tokens=max_new, rng=rng, return_state=True,
        )
        predicted = [int(t) for t in np.asarray(out.state.predicted_tokens[0]).tolist()]
        ids = self._strip_generated(predicted)
        return GenerateOut(ids=ids, text=self._text(ids) if ids else "")

    def __repr__(self) -> str:
        return f"GemmaJaxAdapter({self.model_class}, {'loaded' if self.loaded else 'lazy'})"
