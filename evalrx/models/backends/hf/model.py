"""HF-local backend — the only InternalsHandle path.

Loads any spec via ``transformers`` and captures internals.  Two-tier capture:

  * **HF flags** (``output_attentions`` / ``output_hidden_states`` / logits) cover
    attention, residual-stream hidden states and logits with NO module-path
    surgery — this is the high-value common path.
  * **hooks + runtime path discovery** (:mod:`evalrx.models.backends.hf.discover`) are
    only needed beyond what flags give (activation patching, MoE routing,
    gradients) — reserved for Stage 2.

Attention capture REQUIRES eager attention (sdpa/flash silently return ``None``),
so when the model declares ``eager_required_for_attn`` we force
``attn_implementation="eager"``.  ``torch`` / ``transformers`` are imported
lazily so this module imports on a torch-free install.
"""

from __future__ import annotations

import logging
from typing import Any

from evalrx.core.capability import Capability, CapabilityError
from evalrx.core.case import Inputs
from evalrx.core.model import Model, TokenLogprob, Trace
from evalrx.core.spec import AttnSemantics
from evalrx.core.tool import ChatTurn
from evalrx.models._media import AUDIO_SAMPLE_RATE, resolve_audio, resolve_image
from evalrx.models.backends.base import Backend, RuntimeConfig
from evalrx.models.backends.hf._util import (  # noqa: F401
    _check_audio_duration,
    _collect_message_images,
    _new_tokens,
    _populate_audio_extras,
    _populate_vision_extras,
    _read_nested_attr,
)
from evalrx.models.backends.hf.decoding import ContrastiveDecodingMixin
from evalrx.models.backends.hf.specialists import SpecialistRepairsMixin

logger = logging.getLogger(__name__)

# capability -> HF forward flag
_CAPTURE_FLAGS = {
    Capability.ATTENTION: "output_attentions",
    Capability.HIDDEN_STATES: "output_hidden_states",
}


# Media resolution lives in evalrx.models._media (shared with jax_local); the
# private names stay bound here for existing callers and tests.
_resolve_image = resolve_image
_resolve_audio = resolve_audio


class HFLocalModel(ContrastiveDecodingMixin, SpecialistRepairsMixin, Model):
    """A locally-loaded HF model, constructed from a :class:`ModelSpec`."""

    def __init__(self, spec, runtime: RuntimeConfig) -> None:
        self.spec = spec
        self.runtime = runtime
        self._hf = None  # (model, processor) — lazy
        self._ifcd_editors: dict[str, Any] = {}
        self._detect_engines: dict[str, Any] = {}
        self._audio_text_engines: dict[tuple[str, str], Any] = {}
        self._audio_api_specialist_engines: dict[str, Any] = {}
        self._vision_api_specialist_engines: dict[str, Any] = {}
        self._vision_specialist_engines: dict[str, Any] = {}
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
        from evalrx.models.backends.hf.inference import infer_spec

        spec = spec or infer_spec(model, tokenizer)
        self = cls(spec, runtime or RuntimeConfig())
        self._apply_family_shims(model, tokenizer)
        self._hf = (model, tokenizer)  # bypass lazy load(); a bare tokenizer acts as processor
        if Capability.ATTENTION in self.capabilities:
            self._ensure_eager_attention(model)
        return self

    #: Families whose repo code targets an older transformers generate() loop.
    _GENERATE_SHIM_FAMILIES = frozenset({"nemotron_h", "nemotron_h_omni"})

    def _apply_family_shims(self, model, processor) -> None:
        """Per-family fixes so a loaded handle generates the way its authors intended.

        NemotronH remote code (Nemotron 3 Nano, model card "tested on 4.48.3"):

        * its ``prepare_inputs_for_generation`` builds the hybrid Mamba/attention
          cache only when ``past_key_values is None``, but transformers >= 4.5x
          pre-builds a ``DynamicCache`` for any class not named like a Mamba model
          -> the repo code then warns "no NemotronHHybridDynamicCache provided"
          and recomputes the whole sequence every step (1.9 tok/s on the 4B,
          2026-08-21). Telling generate() the model has no default dynamic cache
          restores the repo's own cache path.
        * its chat template closes every turn with ``<|im_end|>`` (the
          tokenizer's eos_token, id 11) while ``generation_config.eos_token_id``
          is ``</s>`` (2): generate() never stops and pads to the cap with
          ``<|im_end|>\n`` pairs. Stopping on the tokenizer's EOS too fixes it.

        Applied to the top-level model AND an embedded ``language_model`` (the
        Omni wrapper generates through it). No-op for every other family.
        """
        if self.spec.family in self._GENERATE_SHIM_FAMILIES:
            import types

            tok = getattr(processor, "tokenizer", processor)
            tok_eos = getattr(tok, "eos_token_id", None)
            for target in (model, getattr(model, "language_model", None)):
                if target is None:
                    continue
                if hasattr(target, "_supports_default_dynamic_cache"):
                    target._supports_default_dynamic_cache = types.MethodType(lambda _self: False, target)
                gen_cfg = getattr(target, "generation_config", None)
                if gen_cfg is None or tok_eos is None:
                    continue
                current = gen_cfg.eos_token_id
                ids = list(current) if isinstance(current, (list, tuple)) else ([current] if current is not None else [])
                if tok_eos not in ids:
                    gen_cfg.eos_token_id = ids + [int(tok_eos)]
                    logger.info("%s: generation stops on tokenizer eos %s as well (was %s)",
                                self.spec.key, tok_eos, current)
        self._suppress_talker_audio(model)

    def _suppress_talker_audio(self, model) -> None:
        """Never synthesize speech through an omni model's talker for a text benchmark.

        Qwen3-Omni-Instruct (config ``enable_audio_output=True`` on the
        published checkpoint) sets ``self.has_talker = True`` at load, and its
        own ``generate()`` defaults ``return_audio`` to ``self.has_talker``
        when the caller doesn't pass it. Every call then runs the full
        text-to-speech pipeline and returns ``(thinker_result.sequences,
        talker_wav)`` -- a 2-tuple whose first element is ``[batch, seq_len]``
        (batch first, not the single flat token sequence every other
        ``model.generate()`` call site in this file assumes) -- instead of
        just the thinker's text. Decoding that shape produced a batch-decode
        (a Python list containing one string) that downstream code treated as
        already-final text, so every generation surfaced as its own repr:
        seen live on Qwen3-Omni-30B-A3B-Instruct/MMAU, every one of 128
        cases, 2026-08-27 -- ``"['user\\n...\\nassistant\\nA']"``, not ``"A"``.

        Nothing in this codebase's analysis reads a talker's audio output, so
        request ``return_audio=False`` once here (read live off ``has_talker``
        rather than a hard-coded family list, matching this module's
        convention) instead of at all nine ``model.generate()`` call sites.
        """
        if not callable(getattr(model, "generate", None)):
            return
        original_generate = model.generate
        if getattr(original_generate, "_evalrx_talker_suppressed", False):
            return  # already wrapped (re-entrant load/wrap paths)

        def _generate_without_talker(*args, **kwargs):
            if getattr(model, "has_talker", False):
                kwargs.setdefault("return_audio", False)
            return original_generate(*args, **kwargs)

        _generate_without_talker._evalrx_talker_suppressed = True
        model.generate = _generate_without_talker

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
        from evalrx.models.backends.hf.discover import get_unembed

        model, _ = self._loaded
        head = get_unembed(model)
        return getattr(head, "weight", None)

    def final_norm(self):
        """The final normalization module before the unembed (``None`` if not found).

        RMSNorm-family models need ``lm_head(norm(h_i))``, not ``lm_head(h_i)``,
        for faithful intermediate-layer readout (DeCo reference implementation).
        """
        from evalrx.models.backends.hf.discover import get_final_norm

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
        self._apply_family_shims(model, processor)
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

    def _render_text_prompt(self, tok: Any, prompt: str) -> str:
        """The text a TEXT-ONLY spec feeds the tokenizer.

        With ``runtime.apply_chat_template`` the prompt becomes one user turn
        rendered through the chat template, carrying ``spec.chat_template_kwargs``
        (``enable_thinking=False`` for the reasoning checkpoints). Without it —
        the default — the prompt is tokenised verbatim: completion mode, where
        an instruct/thinking model writes its own ``<think>`` block and never
        sees the thinking switch (Qwen3.5-2B ran 8192 tokens that way on a
        causal-judgement item, 2026-08-21).
        """
        if not self.runtime.apply_chat_template:
            return prompt
        if not getattr(tok, "chat_template", None) or not hasattr(tok, "apply_chat_template"):
            logger.warning("%s: apply_chat_template requested but the tokenizer has no chat "
                           "template; tokenising the raw prompt", self.spec.key)
            return prompt
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            **self.spec.chat_template_kwargs,
        )

    def _encode(self, prompt: str):
        import torch  # noqa: F401

        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        text = self._render_text_prompt(tok, prompt)
        if text is prompt:
            enc = tok(text, return_tensors="pt")
        else:
            # a rendered template already carries BOS; a second one shifts every
            # position the white-box analyzers read
            enc = tok(text, return_tensors="pt", add_special_tokens=False)
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
        # PipelineSpec uses backend-neutral decoding controls. Translate the
        # two controls that Hugging Face does not accept verbatim instead of
        # letting run_pipeline swallow a ValueError and report 0 coverage.
        temperature = kwargs.get("temperature")
        if temperature is not None:
            if float(temperature) <= 0.0:
                kwargs.pop("temperature", None)
                kwargs["do_sample"] = False
                kwargs.pop("top_p", None)
                kwargs.pop("top_k", None)
            else:
                kwargs.setdefault("do_sample", True)
        stop = kwargs.pop("stop", None)
        if stop:
            kwargs["stop_strings"] = stop
            kwargs["tokenizer"] = tok
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, **kwargs)
        new = _new_tokens(out[0], enc["input_ids"][0])
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
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen_ids = _new_tokens(out.sequences[0], enc["input_ids"][0]).tolist()
        result: list[TokenLogprob] = []
        for i, score in enumerate(out.scores):
            lp = torch.log_softmax(score[0].float(), dim=-1)
            tid = gen_ids[i]
            topk = torch.topk(lp, min(top_k, lp.shape[-1]))
            top = {tok.decode([int(j)]): float(v) for v, j in zip(topk.values, topk.indices)}
            result.append(TokenLogprob(token=tok.decode([tid]), logprob=float(lp[tid]), top=top))
        return result

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
        new_ids = _new_tokens(out[0], enc["input_ids"][0])
        gen = tok.decode(new_ids, skip_special_tokens=True)
        usage = {"prompt_tokens": int(n_in), "completion_tokens": int(new_ids.shape[0])}
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
        from evalrx.core.tokentype import build_token_type_map

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
