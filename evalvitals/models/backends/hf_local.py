"""HF-local backend — the only InternalsHandle path.

Loads any spec via ``transformers`` and captures internals.  Two-tier capture:

  * **HF flags** (``output_attentions`` / ``output_hidden_states`` / logits) cover
    attention, residual-stream hidden states and logits with NO module-path
    surgery — this is the high-value common path.
  * **hooks + runtime path discovery** (:mod:`evalvitals.models._discover`) are
    only needed beyond what flags give (activation patching, MoE routing,
    gradients) — reserved for Stage 2.

Attention capture REQUIRES eager attention (sdpa/flash silently return ``None``),
so when the model declares ``eager_required_for_attn`` we force
``attn_implementation="eager"``.  ``torch`` / ``transformers`` are imported
lazily so this module imports on a torch-free install.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.core.case import Inputs
from evalvitals.core.model import Model, TokenLogprob, Trace
from evalvitals.core.spec import AttnSemantics
from evalvitals.core.tool import ChatTurn
from evalvitals.models.backends.base import Backend, RuntimeConfig

logger = logging.getLogger(__name__)

# capability -> HF forward flag
_CAPTURE_FLAGS = {
    Capability.ATTENTION: "output_attentions",
    Capability.HIDDEN_STATES: "output_hidden_states",
}


def _read_nested_attr(obj: Any, attr_path: "str | None", *, default: Any) -> Any:
    """Walk a dotted attribute path on *obj*, returning *default* if any step is missing."""
    if not attr_path:
        return default
    for part in attr_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return default
    return obj


def _populate_vision_extras(
    extras: dict,
    input_ids: Any,  # CPU torch.Tensor
    proc_in: dict,
    model_config: Any,
    vision: Any,  # VisionSpec — avoid circular import; duck-typed
) -> None:
    """Fill *extras* with image-token mask and spatial layout for analyzers.

    Called from ``HFLocalModel._vlm_forward`` after the processor runs.
    Writes:
      ``image_token_mask``    — bool tensor (seq_len,) marking image-pad positions.
      ``image_spatial_shape`` — (H, W) patch grid after spatial merge, for reshaping.
      ``image_grid_thw``      — raw (T, H, W) tensor if grid_source=="grid_thw".
    """
    # A dotted attr (e.g. Omni's "thinker_config.image_token_id") needs the
    # nested walker; a bare name behaves identically to getattr(), so no VLM
    # spec's resolution changes.
    image_token_id = _read_nested_attr(model_config, vision.image_token_id_attr, default=None)
    if image_token_id is not None:
        extras["image_token_mask"] = input_ids == image_token_id

    merge = int(_read_nested_attr(model_config, vision.merge_size_attr, default=1) or 1)

    if vision.grid_source == "grid_thw":
        grid_t = proc_in.get("image_grid_thw")
        if grid_t is not None:
            grid = grid_t.cpu() if hasattr(grid_t, "cpu") else grid_t
            extras["image_grid_thw"] = grid
            _, h, w = int(grid[0, 0]), int(grid[0, 1]), int(grid[0, 2])
            extras["image_spatial_shape"] = (h // merge, w // merge)
    elif vision.grid_source == "grid_hw":
        grid_t = proc_in.get("image_grid_hw")
        if grid_t is not None:
            grid = grid_t.cpu() if hasattr(grid_t, "cpu") else grid_t
            extras["image_grid_hw"] = grid
            h, w = int(grid[0, 0]), int(grid[0, 1])
            extras["image_spatial_shape"] = (h // merge, w // merge)


def _populate_audio_extras(
    extras: dict,
    input_ids: Any,  # CPU torch.Tensor
    model_config: Any,
    audio: Any,  # AudioSpec — avoid circular import; duck-typed
) -> None:
    """Fill *extras* with the audio-token mask (the audio TokenTypeMap analog).

    Symmetric to :func:`_populate_vision_extras`, minus the spatial-grid part —
    audio placeholders are a flat run of one token per encoded frame-group, no
    2D reshape. Writes ``audio_token_mask`` — bool tensor (seq_len,) marking
    audio-placeholder positions — which downstream paper methods (e.g. a
    contrastive-decoding audio-reliance signal) read to isolate how much of the
    decoder's attention lands on audio vs. text/image tokens.
    """
    audio_token_id = _read_nested_attr(model_config, audio.audio_token_id_attr, default=None)
    if audio_token_id is not None:
        extras["audio_token_mask"] = input_ids == audio_token_id


def _resolve_image(obj: Any) -> Any:
    """Return a PIL image for *obj* (PIL passes through; str/Path is opened)."""
    if hasattr(obj, "size") and hasattr(obj, "mode"):  # already PIL-like
        return obj
    from PIL import Image

    return Image.open(obj).convert("RGB")


# Sampling rate contract for ``Inputs.audio``: an ndarray is expected to already
# be mono float32 at this rate (matches the WhisperFeatureExtractor every
# audio-capable spec in this codebase uses). A path/URL is decoded to it here,
# so callers never have to think about resampling.
AUDIO_SAMPLE_RATE = 16000


def _resolve_audio(obj: Any) -> Any:
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


def _check_audio_duration(audios: list, processor: Any, model_key: str) -> None:
    """Raise before the processor silently truncates audio past its encoder window.

    ``WhisperFeatureExtractor.chunk_length`` (seconds) differs per checkpoint —
    Qwen2-Audio's is 30s, Qwen2.5-Omni's is 300s — and padding beyond it is
    silently dropped rather than erroring (verified empirically: token count
    and ``feature_attention_mask`` both cap at the window with no warning).
    Read live from the processor, never baked, per this module's convention.
    """
    feature_extractor = getattr(processor, "feature_extractor", None)
    chunk_length = getattr(feature_extractor, "chunk_length", None)
    if chunk_length is None:
        return  # nothing to check this checkpoint's contract against
    # ``audios`` are already resolved to AUDIO_SAMPLE_RATE by _resolve_audio, but
    # the limit itself is read from the feature extractor's own rate rather than
    # the module constant, so this stays correct if that ever diverges.
    sampling_rate = getattr(feature_extractor, "sampling_rate", AUDIO_SAMPLE_RATE)
    limit_samples = int(chunk_length) * int(sampling_rate)
    for i, wav in enumerate(audios):
        if len(wav) > limit_samples:
            duration_sec = len(wav) / sampling_rate
            logger.warning(
                "%s: audio[%d] is %.1fs, longer than the %ds encoder window — refusing rather "
                "than letting the processor silently truncate it",
                model_key, i, duration_sec, chunk_length,
            )
            raise ValueError(
                f"{model_key}: audio[{i}] is {duration_sec:.1f}s, longer than "
                f"this checkpoint's {chunk_length}s encoder window — it would be silently "
                "truncated rather than raising inside the processor. Chunk the audio yourself "
                "before calling, or accept a documented context window in the caller."
            )


def _collect_message_images(messages: list) -> list:
    """Extract images from chat messages in appearance order.

    Messages follow the transformers content-block convention: ``content`` is a
    plain string OR a list of blocks, where an image block is
    ``{"type": "image", "image": <PIL | path>}``.  The block order across the
    whole conversation must match the ``images=`` list handed to the processor.
    """
    images: list = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                img = block.get("image")
                if img is not None:
                    images.append(_resolve_image(img))
    return images


def _format_vlm_input(processor: Any, tok: Any, prompt: str, image: Any) -> "tuple[str, list]":
    """Build the formatted text string and image list for the VLM processor.

    Uses the processor's ``apply_chat_template`` (or tokenizer's) when available
    so that image placeholder tokens are inserted at the correct position.
    Returns ``(text, [image])`` or ``(prompt, [])`` when no image is given.
    """
    apply_fn = getattr(processor, "apply_chat_template", None) or getattr(
        tok, "apply_chat_template", None
    )
    if apply_fn is None:
        return prompt, ([image] if image is not None else [])

    content: list = []
    images = (
        list(image) if isinstance(image, (list, tuple)) else ([image] if image is not None else [])
    )
    for item in images:
        content.append({"type": "image", "image": item})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = apply_fn(messages, tokenize=False, add_generation_prompt=True)
    return text, images


class HFLocalModel(Model):
    """A locally-loaded HF model, constructed from a :class:`ModelSpec`."""

    def __init__(self, spec, runtime: RuntimeConfig) -> None:
        self.spec = spec
        self.runtime = runtime
        self._hf = None  # (model, processor) — lazy
        self._ifcd_editors: dict[str, Any] = {}
        caps = {
            Capability.GENERATE,
            Capability.LOGITS,
            Capability.LOGPROBS,
            Capability.HIDDEN_STATES,
        }
        if spec.attn_semantics is not AttnSemantics.NONE:
            caps.add(Capability.ATTENTION)
        # TOOL_CALLS is a CONDITIONAL capability for local models: the backend
        # provides the channel, but tool-calling only works if the model's chat
        # template renders tools (declared per-model via spec.tool_calling).
        if spec.tool_calling:
            caps.add(Capability.TOOL_CALLS)
        self.capabilities = frozenset(caps)
        self.modalities = frozenset(spec.modalities)  # text / image / audio / video, from the spec

    def paper_method_fidelity(self, method: str) -> str:
        """Declare whether a paper-method adapter is exact or architecture-adapted.

        ``FixAgent`` admits architecture-native paper routes by default.  An
        experiment may explicitly opt into ``"adapted"`` methods, but reports
        must retain that distinction rather than treating a same-formula port
        to a different model architecture as a paper reproduction.
        """
        fidelity = self._paper_method_fidelity_impl(method)
        logger.debug("%s: paper_method_fidelity(%r) -> %r", self.spec.key, method, fidelity)
        return fidelity

    def _paper_method_fidelity_impl(self, method: str) -> str:
        if method == "vcd":
            # The executor uses the released corruption, plausibility cutoff,
            # and per-token sampler. Image-specific seeding keeps paired
            # evaluation independent of iteration order.
            return "per_item_seeded_sampler_specialization"
        if method == "icd":
            return (
                "native_binary_specialization"
                if self.spec.model_type == "instructblip"
                else "adapted"
            )
        if method == "vicrop":
            return (
                "native_selector_specialization" if self.spec.model_type == "llava" else "adapted"
            )
        if method == "opera":
            # This preserves OPERA's first-token over-trust penalty for POPE's
            # binary task. Its beam rollback is not meaningful when exactly
            # one output token is evaluated.
            return (
                "native_binary_specialization" if self.spec.model_type == "llava" else "unavailable"
            )
        if method == "ifcd":
            checkpoint = self.runtime.engine_kwargs.get("ifcd_checkpoint")
            if self.spec.model_type == "llava" and checkpoint and Path(str(checkpoint)).is_file():
                # The public Vicuna TruthX artifact is compatible with
                # LLaVA-1.5's decoder but is not the paper's MSCOCO-trained
                # editor, and modern HF hooks after ``o_proj``. Never promote
                # this to an exact IFCD reproduction.
                return "adapted_truthx_artifact"
            return "unavailable"
        if method == "pai":
            return (
                "native_attention_cfg_specialization"
                if self.spec.model_type == "llava"
                else "unavailable"
            )
        return "unavailable"

    @classmethod
    def from_loaded(
        cls, model, tokenizer, spec=None, runtime: "RuntimeConfig | None" = None
    ) -> "HFLocalModel":
        """Wrap an ALREADY-LOADED HF model + tokenizer (the ``wrap()`` on-ramp).

        Unlike the spec-driven path, no weights are fetched: we inject the live
        ``(model, processor)`` directly and skip :meth:`load`.  The spec is inferred
        from the model when not given, and the attention capability is verified
        against the live ``attn_implementation`` (eager is required, else the model
        returns ``None`` attentions) — flipping it to eager when possible.
        """
        from evalvitals.models.inference import infer_spec

        spec = spec or infer_spec(model, tokenizer)
        self = cls(spec, runtime or RuntimeConfig())
        self._hf = (model, tokenizer)  # bypass lazy load(); a bare tokenizer acts as processor
        if Capability.ATTENTION in self.capabilities:
            self._ensure_eager_attention(model)
        return self

    @staticmethod
    def _ensure_eager_attention(model) -> None:
        """Best-effort switch to eager attention so ``output_attentions`` is populated.

        sdpa / flash_attention_2 silently return ``None`` attentions.  Newer
        transformers expose ``set_attn_implementation``; otherwise we set the config
        flag and warn that a reload may be required for it to take effect.
        """
        config = getattr(model, "config", None)
        current = getattr(config, "_attn_implementation", None) if config is not None else None
        if current == "eager":
            return
        if hasattr(model, "set_attn_implementation"):
            try:
                model.set_attn_implementation("eager")
                return
            except Exception:  # pragma: no cover - falls through to the config flag
                pass
        if config is not None:
            config._attn_implementation = "eager"
            import warnings

            msg = (
                f"wrapped model used attn_implementation={current!r}; set it to 'eager' for "
                "attention capture. If attentions come back empty, reload the model with "
                "from_pretrained(..., attn_implementation='eager')."
            )
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)

    def unembed_weight(self):
        """The lm_head / unembedding weight ``(vocab, dim)`` for logit-lens."""
        from evalvitals.models._discover import get_unembed

        model, _ = self._loaded
        head = get_unembed(model)
        return getattr(head, "weight", None)

    def final_norm(self):
        """The final normalization module before the unembed (``None`` if not found).

        RMSNorm-family models need ``lm_head(norm(h_i))``, not ``lm_head(h_i)``,
        for faithful intermediate-layer readout (DeCo reference implementation).
        """
        from evalvitals.models._discover import get_final_norm

        model, _ = self._loaded
        return get_final_norm(model)

    # -- lazy load -----------------------------------------------------
    def load(self) -> None:
        import time

        import torch
        import transformers

        start = time.monotonic()
        logger.info(
            "loading %s from %s (backend=hf_local, dtype=%s, device=%s)",
            self.spec.key, self.spec.hf_repo, self.runtime.dtype, self.runtime.device,
        )
        auto_cls = getattr(transformers, self.spec.auto_class)
        proc_cls = getattr(transformers, self.spec.processor_class, transformers.AutoProcessor)

        attn_impl = self.runtime.attn_impl
        if (
            attn_impl is None
            and self.spec.eager_required_for_attn
            and Capability.ATTENTION in self.capabilities
        ):
            attn_impl = "eager"  # sdpa/flash return None attentions

        # Use `dtype` (the current transformers param; `torch_dtype` is deprecated).
        kwargs: dict[str, Any] = dict(
            dtype=getattr(torch, self.runtime.dtype),
            trust_remote_code=self.spec.trust_remote_code,
        )
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        device = self.runtime.device
        if device in (None, "auto") or isinstance(device, dict):
            # device_map path (multi-GPU / sharded) — needs accelerate
            model = auto_cls.from_pretrained(
                self.spec.hf_repo, device_map=device or "auto", **kwargs
            )
        else:
            # explicit single device ("cuda" / "cuda:0" / "cpu") — no accelerate dependency
            model = auto_cls.from_pretrained(self.spec.hf_repo, **kwargs).to(device)
        model.eval()
        processor = proc_cls.from_pretrained(
            self.spec.hf_repo, trust_remote_code=self.spec.trust_remote_code
        )

        # Verify the declared TOOL_CALLS capability against the actual template.
        if self.spec.tool_calling:
            tok = getattr(processor, "tokenizer", processor)
            template = getattr(tok, "chat_template", None) or ""
            if "tools" not in template:
                import warnings

                msg = (
                    f"{self.spec.key!r}: spec.tool_calling=True but the chat template has no "
                    "'tools' handling — tool-calling may not render. Verify the checkpoint."
                )
                logger.warning(msg)
                warnings.warn(msg)
        self._hf = (model, processor)
        logger.info(
            "loaded %s in %.1fs (capabilities=%s)",
            self.spec.key, time.monotonic() - start, sorted(c.value for c in self.capabilities),
        )

    @property
    def _loaded(self):
        if self._hf is None:
            self.load()
        return self._hf

    @staticmethod
    def _as_prompt(inputs: Any) -> str:
        return inputs.prompt if isinstance(inputs, Inputs) else str(inputs)

    def _encode(self, prompt: str):
        import torch  # noqa: F401

        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        enc = tok(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        return {k: v.to(device) for k, v in enc.items()}

    # -- interface -----------------------------------------------------
    def generate(self, inputs: Any, **kwargs) -> str:
        import torch

        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        if self.spec.needs_multimodal_encode:
            enc, _, _, _ = self._encode_vlm(inputs, model, processor)
        else:
            prompt = self._as_prompt(inputs)
            enc = self._encode(prompt)
        enc.pop("token_type_ids", None)  # some VLM processors emit this; generate() rejects it
        # "max_tokens" is the OpenAI-style name FixAgent's judge-proposed L2
        # PipelineSpecs use (see fix_tools._safe_generation_kwargs and the
        # _L2_PROMPT schema); transformers' generate() only recognises
        # max_new_tokens and raises ValueError on an unrecognised kwarg
        # rather than ignoring it, so every call in a judge-proposed
        # pipeline silently failed (caught by run_pipeline's per-call
        # try/except) and every case came back unscoreable. max_new_tokens
        # wins if a caller passes both.
        max_new = kwargs.pop("max_new_tokens", None)
        if max_new is None:
            max_new = kwargs.pop("max_tokens", self.runtime.max_new_tokens)
        else:
            kwargs.pop("max_tokens", None)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, **kwargs)
        new = out[0][enc["input_ids"].shape[1] :]
        return tok.decode(new, skip_special_tokens=True)

    def logprobs(
        self, inputs: Any, max_new_tokens: int = 64, top_k: int = 5, **kwargs
    ) -> list[TokenLogprob]:
        """Per-output-token logprobs via greedy generate with output_scores."""
        import torch

        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        if self.spec.needs_multimodal_encode:
            enc, _, _, _ = self._encode_vlm(inputs, model, processor)
        else:
            enc = self._encode(self._as_prompt(inputs))
        enc.pop("token_type_ids", None)
        n_in = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[0][n_in:].tolist()
        result: list[TokenLogprob] = []
        for i, score in enumerate(out.scores):
            lp = torch.log_softmax(score[0].float(), dim=-1)
            tid = gen_ids[i]
            topk = torch.topk(lp, min(top_k, lp.shape[-1]))
            top = {tok.decode([int(j)]): float(v) for v, j in zip(topk.values, topk.indices)}
            result.append(TokenLogprob(token=tok.decode([tid]), logprob=float(lp[tid]), top=top))
        return result

    def _vcd_encodings(
        self,
        inputs: Any,
        *,
        noise_step: int,
        noise_seed: int,
    ) -> tuple[Any, dict[str, Any]]:
        """Build clean/noisy VCD inputs at the published tensor boundary.

        The VCD release calls ``add_diffusion_noise`` *after* its image
        processor: it perturbs the model's normalized image tensor, not an RGB
        image that will subsequently be normalized again.  Matching that
        boundary is material; image-space noise has a different distribution.
        A content-derived seed preserves paired-test reproducibility without
        coupling one example's corruption to the iteration order of another.
        """
        import hashlib

        import torch

        model, processor = self._loaded
        enc, _, _, _ = self._encode_vlm(inputs, model, processor)
        enc.pop("token_type_ids", None)
        pixels = enc.get("pixel_values")
        if pixels is None:
            raise ValueError(f"{self.spec.key}: VCD requires processor pixel_values")
        step = max(0, min(999, int(noise_step)))
        # This is vcd_utils/vcd_add_noise.py verbatim in numerical form.
        betas = torch.sigmoid(torch.linspace(-6, 6, 1000, device=pixels.device))
        betas = betas * (0.5e-2 - 1e-5) + 1e-5
        alpha_bar = torch.cumprod(1 - betas, dim=0)[step].to(dtype=pixels.dtype)
        digest = hashlib.sha256(pixels.detach().float().cpu().numpy().tobytes()).digest()
        item_seed = (int(noise_seed) + int.from_bytes(digest[:8], "little")) % (2**63 - 1)
        generator = torch.Generator(device=pixels.device).manual_seed(item_seed)
        noise = torch.randn(
            pixels.shape, device=pixels.device, dtype=pixels.dtype, generator=generator
        )
        noisy_enc = dict(enc)
        noisy_enc["pixel_values"] = alpha_bar.sqrt() * pixels + (1 - alpha_bar).sqrt() * noise
        return enc, noisy_enc

    def _vcd_next_token_logits(
        self,
        inputs: Any,
        *,
        noise_step: int,
        noise_seed: int,
    ) -> tuple[Any, Any]:
        """Return clean/noisy next-token logits for the legacy binary route."""
        import torch

        model, _ = self._loaded
        enc, noisy_enc = self._vcd_encodings(
            inputs, noise_step=noise_step, noise_seed=noise_seed
        )
        with torch.no_grad():
            clean = model(**enc).logits[0, -1]
            noisy = model(**noisy_enc).logits[0, -1]
        return clean, noisy

    def generate_vcd(
        self,
        inputs: Any,
        *,
        alpha: float = 0.5,
        beta: float = 0.1,
        noise_step: int = 500,
        noise_seed: int = 55,
    ) -> str:
        """Run VCD's released contrastive formula through every output token.

        The source sampler uses temperature-one multinomial sampling.  Here the
        diffusion noise is seeded per image content, rather than a global loop
        RNG, so a frozen selection/confirmation split remains reproducible if
        case order changes.  That seed policy is recorded as a specialization.
        """
        logger.debug(
            "%s: generate_vcd(alpha=%s, beta=%s, noise_step=%s, noise_seed=%s)",
            self.spec.key, alpha, beta, noise_step, noise_seed,
        )
        if not self.spec.is_vlm:
            raise ValueError("VCD visual contrast requires a VLM")
        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            raise ValueError("VCD requires one image and a binary answer task")
        import hashlib

        import torch
        from transformers.generation.logits_process import LogitsProcessorList

        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        clean_enc, noisy_enc = self._vcd_encodings(
            inputs, noise_step=noise_step, noise_seed=noise_seed
        )
        from evalvitals.models.paper_methods.vcd import VCDLogitsProcessor

        # Match the release's temperature=1, top_p=1, no-top-k multinomial
        # branch while preventing evaluation-order-dependent randomness.
        fingerprint = hashlib.sha256(
            clean_enc["pixel_values"].detach().float().cpu().numpy().tobytes()
        ).digest()
        item_seed = (int(noise_seed) + int.from_bytes(fingerprint[:8], "little")) % (2**63 - 1)
        processor_list = LogitsProcessorList(
            [VCDLogitsProcessor(model, noisy_enc, alpha=alpha, beta=beta)]
        )
        cuda_devices = [clean_enc["input_ids"].device.index] if clean_enc["input_ids"].is_cuda else []
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(item_seed)
            if clean_enc["input_ids"].is_cuda:
                torch.cuda.manual_seed(item_seed)
            out = model.generate(
                **clean_enc,
                max_new_tokens=self.runtime.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                use_cache=True,
                logits_processor=processor_list,
            )
        return tok.decode(out[0][clean_enc["input_ids"].shape[1] :], skip_special_tokens=True)

    def generate_vcd_baseline(self, inputs: Any, *, noise_seed: int = 55) -> str:
        """Sample the clean VCD control with the candidate's per-image RNG.

        VCD evaluates both arms with temperature-one multinomial sampling. A
        greedy baseline paired with a sampled contrastive arm is not a valid
        paper-method comparison, so the white-box runner calls this method
        whenever it freezes the VCD candidate.
        """
        logger.debug("%s: generate_vcd_baseline(noise_seed=%s)", self.spec.key, noise_seed)
        if not self.spec.is_vlm:
            raise ValueError("VCD clean control requires a VLM")
        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            raise ValueError("VCD clean control requires one image")
        import hashlib

        import torch

        model, processor = self._loaded
        clean_enc, _ = self._vcd_encodings(inputs, noise_step=999, noise_seed=noise_seed)
        fingerprint = hashlib.sha256(
            clean_enc["pixel_values"].detach().float().cpu().numpy().tobytes()
        ).digest()
        item_seed = (int(noise_seed) + int.from_bytes(fingerprint[:8], "little")) % (2**63 - 1)
        cuda_devices = [clean_enc["input_ids"].device.index] if clean_enc["input_ids"].is_cuda else []
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(item_seed)
            if clean_enc["input_ids"].is_cuda:
                torch.cuda.manual_seed(item_seed)
            out = model.generate(
                **clean_enc,
                max_new_tokens=self.runtime.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                use_cache=True,
            )
        tok = getattr(processor, "tokenizer", processor)
        return tok.decode(out[0][clean_enc["input_ids"].shape[1] :], skip_special_tokens=True)

    def generate_instruction_cd(
        self,
        inputs: Any,
        *,
        alpha: float = 1.0,
        beta: float = 0.1,
        qformer_mode: str = "normal",
        disturbance: str = (
            "You are a confused objects detector to provide a fuzzy overview "
            "or impression of the image."
        ),
    ) -> str:
        """ICD for a binary visual-grounding decision.

        ICD (Wang et al., ACL 2024) contrasts a normal forward pass with a
        *disturbance-instruction* pass.  For InstructBLIP, the disturbance
        replaces only ``qformer_input_ids`` while the decoder prompt remains
        intact, matching the released ``normal.json`` route; ``qformer_mode``
        can also run the released disturbed-question variant.  Decoder-only
        VLMs such as Qwen have no Q-Former, so their text-prefix fallback is
        explicitly architecture-adapted.

        As with :meth:`generate_vcd`, this intentionally supports only a
        one-token Yes/No task.  Applying a first-token shortcut to free-form
        generation would not implement ICD's token-by-token sampler.
        """
        logger.debug(
            "%s: generate_instruction_cd(alpha=%s, beta=%s, qformer_mode=%r)",
            self.spec.key, alpha, beta, qformer_mode,
        )
        if not self.spec.is_vlm:
            raise ValueError("instruction contrast requires a VLM")
        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            raise ValueError("instruction contrast requires one image and a binary answer task")
        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)

        def token_id(answer: str) -> int:
            for spelling in (" " + answer, answer):
                encoded = tok(spelling, add_special_tokens=False)["input_ids"]
                if len(encoded) == 1:
                    return int(encoded[0])
            raise ValueError(
                f"{self.spec.key}: {answer!r} is not one token; ICD binary mode unavailable"
            )

        if self.spec.model_type == "instructblip":
            import torch

            if qformer_mode not in {"normal", "question"}:
                raise ValueError("qformer_mode must be 'normal' or 'question'")
            enc, _, _, _ = self._encode_vlm(inputs, model, processor)
            clean_enc = dict(enc)
            dirty_enc = dict(enc)
            disturbed_qformer_prompt = str(disturbance)
            if qformer_mode == "question":
                disturbed_qformer_prompt += self._as_prompt(inputs)
            qformer = processor.qformer_tokenizer(
                disturbed_qformer_prompt, return_tensors="pt", padding="longest", truncation=True
            ).to(next(model.parameters()).device)
            dirty_enc["qformer_input_ids"] = qformer["input_ids"]
            dirty_enc["qformer_attention_mask"] = qformer["attention_mask"]
            with torch.no_grad():
                clean = model(**clean_enc).logits[0, -1]
                disturbed = model(**dirty_enc).logits[0, -1]
        else:
            clean = self.forward(inputs, capture={Capability.LOGITS}).require(Capability.LOGITS)[-1]
            disturbed_inputs = Inputs(
                prompt=str(disturbance) + self._as_prompt(inputs), image=image
            )
            disturbed = self.forward(disturbed_inputs, capture={Capability.LOGITS}).require(
                Capability.LOGITS
            )[-1]
        ids = {answer: token_id(answer) for answer in ("Yes", "No")}
        clean_scores = {answer: float(clean[token].float()) for answer, token in ids.items()}
        disturbed_scores = {
            answer: float(disturbed[token].float()) for answer, token in ids.items()
        }
        cutoff = max(clean_scores.values()) + math.log(float(beta))
        scores = {
            answer: (1.0 + float(alpha)) * clean_scores[answer]
            - float(alpha) * disturbed_scores[answer]
            for answer in ids
            if clean_scores[answer] >= cutoff
        }
        return max(scores or clean_scores, key=(scores or clean_scores).get)

    def generate_vicrop(self, inputs: Any, *, layer: int | float = 14) -> str:
        """Run the architecture-native LLaVA ViCrop paper executor."""
        logger.debug("%s: generate_vicrop(layer=%s)", self.spec.key, layer)
        if self.spec.model_type != "llava":
            raise ValueError(f"{self.spec.key}: ViCrop is only native on the LLaVA executor")
        from evalvitals.models.paper_methods.vicrop import generate

        return generate(self, inputs, layer=layer)

    def generate_opera_binary(
        self,
        inputs: Any,
        *,
        num_attn_candidates: int = 5,
        penalty_weight: float = 1.0,
    ) -> str:
        """Run OPERA's first-token over-trust penalty for a binary VQA task.

        OPERA scores each likely continuation by the image attention of the
        *candidate token* and subtracts ``-image_attention`` from its logit.
        This is its published early-response penalty verbatim. POPE evaluates
        a single Yes/No token, therefore the later multi-token rollback branch
        is intentionally out of scope and this method must stay labelled a
        binary specialization.
        """
        import torch

        logger.debug(
            "%s: generate_opera_binary(num_attn_candidates=%s, penalty_weight=%s)",
            self.spec.key, num_attn_candidates, penalty_weight,
        )
        if self.spec.model_type != "llava":
            raise ValueError(f"{self.spec.key}: OPERA binary route is only native on LLaVA")
        if int(num_attn_candidates) < 1:
            raise ValueError("num_attn_candidates must be positive")
        model, processor = self._loaded
        enc, _, _, _ = self._encode_vlm(inputs, model, processor)
        enc.pop("token_type_ids", None)
        image_id = getattr(model.config, "image_token_index", None)
        if image_id is None:
            raise ValueError(f"{self.spec.key}: OPERA requires config.image_token_index")
        image_positions = (enc["input_ids"][0] == int(image_id)).nonzero().flatten()
        if image_positions.numel() == 0 or not bool(
            (image_positions[1:] == image_positions[:-1] + 1).all()
        ):
            raise ValueError(f"{self.spec.key}: OPERA requires one contiguous image-token span")

        with torch.no_grad():
            prefill = model(**enc, return_dict=True, output_attentions=True, use_cache=False)
        if not getattr(prefill, "attentions", None):
            raise ValueError(f"{self.spec.key}: OPERA requires eager self-attention outputs")
        raw_logits = prefill.logits[:, -1, :]
        k = min(int(num_attn_candidates), int(raw_logits.shape[-1]))
        candidate_scores, candidate_tokens = torch.topk(raw_logits, k, dim=-1, largest=True, sorted=True)
        adjusted_scores = candidate_scores.clone()
        for candidate_index in range(k):
            candidate_enc = dict(enc)
            candidate_enc["input_ids"] = torch.cat(
                (enc["input_ids"], candidate_tokens[:, candidate_index : candidate_index + 1]), dim=1
            )
            if "attention_mask" in candidate_enc:
                candidate_enc["attention_mask"] = torch.cat(
                    (enc["attention_mask"], torch.ones_like(enc["attention_mask"][:, :1])), dim=1
                )
            with torch.no_grad():
                candidate_output = model(
                    **candidate_enc, return_dict=True, output_attentions=True, use_cache=False
                )
            attentions = getattr(candidate_output, "attentions", None)
            if not attentions:
                raise ValueError(f"{self.spec.key}: OPERA candidate forward returned no attentions")
            # Reference OPERA maximises heads then sums the candidate's image
            # attention. With one beam, selecting the adjusted top candidate
            # is identical to its first beam-search step.
            last_attention = attentions[-1].amax(dim=1)[:, -1, :]
            image_attention = last_attention[:, image_positions].sum(dim=-1)
            adjusted_scores[:, candidate_index] += float(penalty_weight) * image_attention
        selected = candidate_tokens.gather(1, adjusted_scores.argmax(dim=-1, keepdim=True)).squeeze(1)
        tok = getattr(processor, "tokenizer", processor)
        return tok.decode([int(selected[0])], skip_special_tokens=True)

    def generate_ifcd(
        self,
        inputs: Any,
        *,
        alpha: float = 0.1,
        beta: float = 0.1,
        edit_strength: float = 0.5,
        top_layers: int = 15,
        max_new_tokens: int | None = None,
    ) -> str:
        """Run an explicitly adapted TruthX-backed IFCD decoder on LLaVA.

        The checkpoint path is a required runtime artifact, rather than an
        implicit download: its provenance controls whether an experiment can
        compare itself with IFCD's MSCOCO-trained editor.  This adapter uses
        modern HF output hooks, so even a matching artifact remains labelled
        adapted until its pre-``o_proj`` boundary is ported.
        """
        import torch
        from transformers.generation.logits_process import LogitsProcessorList

        logger.debug(
            "%s: generate_ifcd(alpha=%s, beta=%s, edit_strength=%s, top_layers=%s)",
            self.spec.key, alpha, beta, edit_strength, top_layers,
        )
        if self.spec.model_type != "llava":
            raise ValueError(f"{self.spec.key}: IFCD is only wired for LLaVA's Vicuna decoder")
        checkpoint = self.runtime.engine_kwargs.get("ifcd_checkpoint")
        if not checkpoint or not Path(str(checkpoint)).is_file():
            raise ValueError("IFCD requires runtime.engine_kwargs['ifcd_checkpoint']")
        model, processor = self._loaded
        enc, _, _, _ = self._encode_vlm(inputs, model, processor)
        enc.pop("token_type_ids", None)
        key = f"{Path(str(checkpoint)).resolve()}:{int(top_layers)}"
        editor = self._ifcd_editors.get(key)
        if editor is None:
            from evalvitals.models.paper_methods.ifcd import TruthXEditor

            hidden_size = int(getattr(model.config.text_config, "hidden_size", 0) or model.config.hidden_size)
            editor = TruthXEditor(checkpoint, hidden_size=hidden_size, top_layers=top_layers)
            self._ifcd_editors[key] = editor
        from evalvitals.models.paper_methods.ifcd import IFCDLogitsProcessor, truthx_editing

        language_model = getattr(model, "language_model", model)
        editor.strength = float(edit_strength)
        processor_list = LogitsProcessorList(
            [
                IFCDLogitsProcessor(
                    model,
                    dict(enc),
                    editor,
                    alpha=alpha,
                    beta=beta,
                    edit_strength=edit_strength,
                )
            ]
        )
        with truthx_editing(language_model, editor), torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens or self.runtime.max_new_tokens,
                do_sample=False,
                logits_processor=processor_list,
            )
        tok = getattr(processor, "tokenizer", processor)
        return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)

    def generate_vicrop_consensus(
        self,
        inputs: Any,
        *,
        baseline_answer: str,
        layer: int | float = 14,
    ) -> str:
        """Use ViCrop only when independent crop and fused-view answers agree.

        This is a deployment safety guard, not part of the source ViCrop
        method. It prevents a single misleading attention crop from replacing
        an otherwise stable baseline answer.
        """
        import re

        logger.debug("%s: generate_vicrop_consensus(layer=%s)", self.spec.key, layer)
        if self.spec.model_type != "llava":
            raise ValueError(f"{self.spec.key}: ViCrop is only native on the LLaVA executor")
        from evalvitals.models.paper_methods.vicrop import prepare_views

        image, crop, fused_prompt = prepare_views(self, inputs, layer=layer)
        fused = self.generate(Inputs(fused_prompt, [image, crop]))
        crop_only_prompt = (
            "The image is a task-relative crop selected from a larger scene. "
            "Answer only from visible evidence in this crop.\n\n"
            + self._as_prompt(inputs)
        )
        crop_only = self.generate(Inputs(crop_only_prompt, crop))

        def decision(text: str) -> str:
            lowered = str(text).lower()
            yes_no = re.search(r"\b(yes|no)\b", lowered)
            if yes_no:
                return yes_no.group(1)
            choices = re.findall(r"\b([a-d])\b", lowered)
            if choices:
                return choices[-1]
            return re.sub(r"\W+", "", lowered)

        return fused if decision(fused) and decision(fused) == decision(crop_only) else baseline_answer

    def generate_pai(
        self,
        inputs: Any,
        *,
        alpha: float = 0.2,
        guidance_scale: float = 2.0,
        start_layer: int = 2,
        end_layer: int = 32,
        max_new_tokens: int | None = None,
    ) -> str:
        """Run PAI's attention and classifier-free-guidance route on LLaVA.

        The attention boost and the image-free classifier-free-guidance cache
        follow the released PAI decoding path.  It remains an architecture
        specialization because the source uses its pinned LLaVA stack.
        """
        import torch

        logger.debug(
            "%s: generate_pai(alpha=%s, guidance_scale=%s, start_layer=%s, end_layer=%s)",
            self.spec.key, alpha, guidance_scale, start_layer, end_layer,
        )
        if self.spec.model_type != "llava":
            raise ValueError(f"{self.spec.key}: PAI is only native on the LLaVA executor")
        model, processor = self._loaded
        enc, _, _, _ = self._encode_vlm(inputs, model, processor)
        enc.pop("token_type_ids", None)
        image_id = getattr(model.config, "image_token_index", None)
        if image_id is None:
            raise ValueError(f"{self.spec.key}: PAI requires config.image_token_index")
        positions = (enc["input_ids"][0] == int(image_id)).nonzero().flatten()
        if positions.numel() == 0 or not bool((positions[1:] == positions[:-1] + 1).all()):
            raise ValueError(f"{self.spec.key}: PAI requires one contiguous image-token span")
        from transformers.generation.logits_process import LogitsProcessorList

        from evalvitals.models.paper_methods.pai import (
            PAICFGLogitsProcessor,
            image_attention_boost,
        )

        # PAI's reference code patches the LLaMA language model, whereas HF
        # LLaVA wraps it in ``LlavaForConditionalGeneration``.
        language_model = getattr(model, "language_model", model)
        unconditional_ids = torch.cat(
            (enc["input_ids"][:, : positions[0]], enc["input_ids"][:, positions[-1] + 1 :]),
            dim=1,
        )
        cfg = PAICFGLogitsProcessor(
            model,
            unconditional_ids,
            guidance_scale=guidance_scale,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        with image_attention_boost(
            language_model,
            image_start=int(positions[0]),
            image_end=int(positions[-1]) + 1,
            alpha=alpha,
            start_layer=start_layer,
            end_layer=end_layer,
        ):
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens or self.runtime.max_new_tokens,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([cfg]),
                )
        tok = getattr(processor, "tokenizer", processor)
        return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)

    def chat(self, messages: list, tools=None) -> ChatTurn:
        """Tool-aware turn via the model's chat template.

        transformers' ``apply_chat_template(tools=...)`` accepts OpenAI-format tool
        schemas and renders them into the prompt; the model emits the call as text
        which the (Qwen/Hermes) codec parses out — so ``raw_tool_calls`` is None.

        Multimodal: messages may carry transformers-style content blocks
        (``{"type": "image", "image": ...}``); on a VLM the images are routed
        through the processor so each block's placeholder tokens line up with
        its pixels — this is what lets the agent loop feed tool-returned crops
        back to the model.
        """
        import torch

        if Capability.TOOL_CALLS not in self.capabilities:
            raise CapabilityError(
                analyzer="chat", model=repr(self), missing={Capability.TOOL_CALLS}
            )
        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        images = _collect_message_images(messages) if self.spec.is_vlm else []
        if images:
            try:
                text = processor.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.spec.chat_template_kwargs,
                )
            except (
                TypeError
            ):  # older processors don't take tools= — same jinja template lives on the tokenizer
                text = tok.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.spec.chat_template_kwargs,
                )
            enc = processor(text=[text], images=images, return_tensors="pt")
            enc.pop("token_type_ids", None)  # some VLM processors emit this; generate() rejects it
            enc = enc.to(next(model.parameters()).device)
        else:
            text = tok.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **self.spec.chat_template_kwargs,  # e.g. {"enable_thinking": False} for Qwen3
            )
            enc = tok(text, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=self.runtime.max_new_tokens)
        n_in = enc["input_ids"].shape[1]
        gen = tok.decode(out[0][n_in:], skip_special_tokens=True)
        usage = {"prompt_tokens": int(n_in), "completion_tokens": int(out.shape[1] - n_in)}
        return ChatTurn(text=gen, raw_tool_calls=None, usage=usage)

    def _encode_vlm(self, inputs, model, processor):
        """Encode an (image/video/audio, text) input and build its TokenTypeMap.

        Despite the name (kept to avoid touching every call site), this is the
        general multimodal encode path: it also carries ``inputs.audio`` for
        omni/audio-only specs. Builds the TokenTypeMap from the processor output
        BEFORE moving to device, using the live config (image_token_id,
        vision_config.spatial_merge_size) + the spec's VisionSpec — so token ids /
        merge sizes are never hard-coded. ``self.spec.vision is None`` (an
        audio-only spec) skips the TokenTypeMap entirely — it is VLM-specific.

        ``inputs.video`` (list of PIL frames) takes priority over ``inputs.image``:
        each frame gets its own ``{"type": "image"}`` content slot so the processor
        inserts a separate image-token block per frame, enabling multi-frame /
        temporal inputs without a native video tower.

        Audio is independent of the image/video branch: any spec with
        ``self.spec.audio is not None`` that receives ``inputs.audio`` gets an
        ``{"type": "audio"}`` content block plus the decoded waveform(s) passed
        through the processor's ``audio=`` kwarg at :data:`AUDIO_SAMPLE_RATE`.
        """
        from evalvitals.core.tokentype import build_token_type_map

        tok = getattr(processor, "tokenizer", processor)
        prompt = self._as_prompt(inputs)
        image = getattr(inputs, "image", None) if isinstance(inputs, Inputs) else None
        video = getattr(inputs, "video", None) if isinstance(inputs, Inputs) else None
        audio = getattr(inputs, "audio", None) if isinstance(inputs, Inputs) else None

        if video is not None:
            # Multi-frame path: one <image> placeholder per frame, frames passed in order.
            content = [{"type": "image"} for _ in video] + [{"type": "text", "text": prompt}]
            images = list(video)
        elif image is not None:
            images = list(image) if isinstance(image, (list, tuple)) else [image]
            content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        else:
            content = [{"type": "text", "text": prompt}]
            images = []

        audios: list = []
        if self.spec.audio is not None and audio is not None:
            raw_audios = list(audio) if isinstance(audio, (list, tuple)) else [audio]
            audios = [_resolve_audio(a) for a in raw_audios]
            # Audio blocks precede the text block, matching the image/video
            # convention above; order must match the ``audio=`` list below.
            content = [{"type": "audio"} for _ in audios] + content

        if self.spec.model_type == "instructblip":
            if len(images) != 1:
                raise ValueError("InstructBLIP supports exactly one image per request")
            # InstructBLIP has no chat template or decoder image placeholder:
            # its processor returns vision pixels plus an independent Q-Former
            # text sequence.  The latter is what ICD perturbs.  Vicuna-based
            # InstructBLIP expects an explicit answer turn; without it the
            # checkpoint often terminates immediately after echoing the prompt.
            qformer_prompt = prompt
            decoder_prompt = f"Question: {prompt} Answer:"
            enc = processor(images=images[0], text=decoder_prompt, return_tensors="pt")
            # The published ICD implementation gives the Q-Former the raw
            # question while the Vicuna decoder receives its answer template.
            # Override the processor's coupled default to retain that split.
            qformer = processor.qformer_tokenizer(
                qformer_prompt, return_tensors="pt", padding="longest", truncation=True
            )
            enc["qformer_input_ids"] = qformer["input_ids"]
            enc["qformer_attention_mask"] = qformer["attention_mask"]
        else:
            text = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False,
                **self.spec.chat_template_kwargs,
            )
            proc_kwargs = {"text": [text], "return_tensors": "pt"}
            if images:
                proc_kwargs["images"] = images
            if audios:
                _check_audio_duration(audios, processor, self.spec.key)
                proc_kwargs["audio"] = audios
                proc_kwargs["sampling_rate"] = AUDIO_SAMPLE_RATE
            enc = processor(**proc_kwargs)
        ttm = (
            build_token_type_map(enc["input_ids"], enc, model.config, self.spec.vision)
            if self.spec.vision is not None
            else None
        )
        enc = enc.to(next(model.parameters()).device)
        ids = enc["input_ids"][0].tolist()
        tokens = [tok.decode([i]) for i in ids]
        return enc, ids, tokens, ttm

    def forward(self, inputs: Any, capture: set[Capability], spec=None) -> Trace:
        import torch

        model, processor = self._loaded
        if self.spec.needs_multimodal_encode:
            enc, token_ids, tokens, ttm = self._encode_vlm(inputs, model, processor)
        else:
            tok = getattr(processor, "tokenizer", processor)
            enc = self._encode(self._as_prompt(inputs))
            token_ids = enc["input_ids"][0].tolist()
            tokens = [tok.decode([t]) for t in token_ids]
            ttm = None

        extras: dict = {"attn_semantics": self.spec.attn_semantics.value}
        if self.spec.vision is not None:
            image = getattr(inputs, "image", None) if isinstance(inputs, Inputs) else None
            video = getattr(inputs, "video", None) if isinstance(inputs, Inputs) else None
            if image is not None or video is not None:
                _populate_vision_extras(
                    extras, torch.tensor(token_ids), enc, model.config, self.spec.vision
                )
        if self.spec.audio is not None:
            audio = getattr(inputs, "audio", None) if isinstance(inputs, Inputs) else None
            if audio is not None:
                _populate_audio_extras(extras, torch.tensor(token_ids), model.config, self.spec.audio)

        flags = {flag: True for cap, flag in _CAPTURE_FLAGS.items() if cap in capture}
        enc.pop("token_type_ids", None)  # some VLM processors emit this; forward() rejects it
        with torch.no_grad():
            outputs = model(**enc, **flags)

        layers = spec.layers if spec is not None else None
        to_cpu = spec.to_cpu if spec is not None else True

        def _maybe_subset(seq):
            return [seq[i] for i in layers] if layers is not None else list(seq)

        def _move(t):
            return t.cpu() if to_cpu else t

        provided: set[Capability] = set()
        attentions = hidden_states = logits = None
        if Capability.ATTENTION in capture:
            if getattr(outputs, "attentions", None) is None:
                raise RuntimeError(
                    f"{self!r}: ATTENTION was requested but the model returned no attentions. "
                    "Load it with attn_implementation='eager' (sdpa/flash silently return None)."
                )
            attentions = [_move(a.squeeze(0)) for a in _maybe_subset(outputs.attentions)]
            provided.add(Capability.ATTENTION)
        if (
            Capability.HIDDEN_STATES in capture
            and getattr(outputs, "hidden_states", None) is not None
        ):
            hidden_states = [_move(h.squeeze(0)) for h in _maybe_subset(outputs.hidden_states)]
            provided.add(Capability.HIDDEN_STATES)
        if Capability.LOGITS in capture:
            logits = _move(outputs.logits.squeeze(0))
            provided.add(Capability.LOGITS)

        return Trace(
            tokens=tokens,
            token_ids=token_ids,
            provided=provided,
            attentions=attentions,
            hidden_states=hidden_states,
            logits=logits,
            token_type_map=ttm,
            extras=extras,
        )

    def __repr__(self) -> str:
        status = "loaded" if self._hf else "lazy"
        return f"HFLocalModel(key={self.spec.key!r}, {status})"


class HFLocalBackend(Backend):
    kind = "hf_local"
    # Superset the backend CAN provide; the actual per-model set is computed in
    # HFLocalModel.__init__ (e.g. TOOL_CALLS only when the spec's template supports it).
    capabilities = frozenset(
        {
            Capability.GENERATE,
            Capability.TOOL_CALLS,
            Capability.LOGITS,
            Capability.LOGPROBS,
            Capability.HIDDEN_STATES,
            Capability.ATTENTION,
        }
    )

    def build(self, spec, runtime: RuntimeConfig) -> HFLocalModel:
        return HFLocalModel(spec, runtime)
