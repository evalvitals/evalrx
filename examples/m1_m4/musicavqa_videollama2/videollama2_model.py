"""VideoLLaMA2.1-7B-AV as an evalrx ``Model`` — example-local, no framework changes.

VideoLLaMA2 (``DAMO-NLP-SG/VideoLLaMA2``, ``audio_visual`` branch) is not a
stock ``transformers`` causal LM and is not a registered EvalRX
``ModelSpec``; its own ``mm_infer()`` helper also hardcodes ``.cuda()`` in
three places, so it cannot run on CPU as shipped.  Rather than editing the
third-party package or `evalrx/`, this module:

  1. implements ``Model`` (the public ABC in ``evalrx.core.model`` —
     ``generate()`` + ``forward()``) directly, the same extension point the
     framework uses for ``api``/``hf_local`` backends themselves;
  2. keeps a device-parameterized COPY of ``mm_infer`` (``_mm_infer_on``
     below) so the same class runs on ``cuda`` (unmodified upstream path) or
     ``cpu`` (patched path) — a diff against upstream ``mm_infer``, not a
     rewrite of the model.

HIDDEN_STATES / ATTENTION capture (``forward()``): verified against the
upstream source (``videollama2/model/videollama2_qwen2.py``) —
``Videollama2Qwen2ForCausalLM.forward()`` genuinely threads
``output_hidden_states``/``output_attentions`` through to the underlying
Qwen2 stack after multimodal fusion, and ``from_pretrained`` accepts
``attn_implementation="eager"`` like any standard HF model (attention
capture needs eager; sdpa/flash return ``None``, same rule evalrx
applies to registered models). One real gap, left unfixed on purpose:
video/audio produce a *variable*, input-length-dependent number of fused
embedding slots (``prepare_inputs_labels_for_multimodal`` in
``videollama2_arch.py`` concatenates per-clip vision + audio features at
the ``<video>`` placeholder), so ``Trace.tokens``/``token_ids`` here
describe the PRE-fusion prompt only — they do not index 1:1 into the
returned hidden_states/attentions sequence dimension. Reproducing the
fusion boundary needs empirical verification against real weights (not
possible on this CPU-only, RAM-constrained host); analyzers that need a
``token_type_map`` (image/audio token span localization) will not have
one from this handle yet — everything else (representation-geometry/CKA,
attention-entropy, layer-wise probes) works off the raw tensors as-is.

``MockAVModel`` is a zero-weight stand-in with the same interface, for
smoke-testing the M1-M4 loop wiring on hosts that cannot fit the real 17GB
checkpoint in RAM (see ``mine_cases.py --mock`` / ``run.py --mock``).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from evalrx.core.capability import Capability
from evalrx.core.case import Inputs
from evalrx.core.model import Model, Trace


# ---------------------------------------------------------------------------
# Device-parameterized mm_infer (diff vs. videollama2/__init__.py::mm_infer:
# every hardcoded `.cuda()` -> `.to(device)`; `.half()` -> dtype param, since
# float16 on CPU is either unsupported or extremely slow for many ops).
# ---------------------------------------------------------------------------

def _build_multimodal_inputs(image_or_video, instruct, model, tokenizer, device, dtype,
                              modal="video"):
    """Shared prompt/tensor construction — the non-generation half of upstream
    ``mm_infer`` (text templating + tokenization + tensor placement), reused by
    both ``_mm_infer_on`` (generation) and ``_forward_on`` (internals capture)
    so the two paths can never drift on how the prompt is built."""
    from videollama2.constants import DEFAULT_AUDIO_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN
    from videollama2.mm_utils import tokenizer_multimodal_token

    modal_token = {"image": DEFAULT_IMAGE_TOKEN, "video": DEFAULT_VIDEO_TOKEN,
                   "text": "", "audio": DEFAULT_AUDIO_TOKEN}[modal]

    if modal == "text":
        tensor = None
    else:
        if isinstance(image_or_video, dict):
            tensor = {k: v.to(device=device, dtype=dtype) for k, v in image_or_video.items()}
        else:
            tensor = image_or_video.to(device=device, dtype=dtype)
        tensor = [(tensor, modal)]

    message = [{"role": "user", "content": modal_token + "\n" + instruct}]
    if model.config.model_type in ("videollama2", "videollama2_mistral", "videollama2_mixtral"):
        message = [{"role": "system", "content": (
            "<<SYS>>\nYou are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe.\n<</SYS>>"
        )}] + message

    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer_multimodal_token(
        prompt, tokenizer, modal_token, return_tensors="pt"
    ).unsqueeze(0).long().to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(device)
    return input_ids, attention_mask, tensor


def _mm_infer_on(image_or_video, instruct, model, tokenizer, device, dtype,
                  modal="video", return_meta: bool = False, **kwargs):
    import torch

    from videollama2.mm_utils import KeywordsStoppingCriteria

    input_ids, attention_mask, tensor = _build_multimodal_inputs(
        image_or_video, instruct, model, tokenizer, device, dtype, modal)

    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    do_sample = kwargs.get("do_sample", False)
    max_new_tokens = kwargs.get("max_new_tokens", 128)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=tensor,
            do_sample=do_sample,
            temperature=kwargs.get("temperature", 0.2 if do_sample else 0.0),
            max_new_tokens=max_new_tokens,
            top_p=kwargs.get("top_p", 0.9),
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if not return_meta:
        return text
    # HF's own finish_reason isn't surfaced by mm_infer's transformers.generate()
    # call above (this branch doesn't pass return_dict_in_generate=True, and
    # doing so changes output_ids' shape/type in ways the stopping-criteria +
    # batch_decode call above aren't written for) -- but "did it hit the token
    # cap" is exactly what FixAgent's L0 telemetry gate needs
    # (fix_agent.py:_l0_candidates -- metadata['finish_reason']=='length' +
    # metadata['generation_config']['max_tokens']), and that specific fact is
    # cheaply and reliably derivable here: output_ids includes the prompt, so
    # the number of NEWLY generated tokens is output_ids.shape[1] -
    # input_ids.shape[1]; reaching max_new_tokens (stopping_criteria never
    # fired) is the token-cap case, full stop.
    n_generated = output_ids.shape[1] - input_ids.shape[1]
    finish_reason = "length" if n_generated >= max_new_tokens else "stop"
    meta = {
        "finish_reason": finish_reason,
        "generation_config": {"max_tokens": int(max_new_tokens)},
        "n_generated_tokens": int(n_generated),
    }
    return text, meta


def _forward_on(image_or_video, instruct, model, tokenizer, device, dtype,
                 capture: "set[Capability]", modal="video") -> Trace:
    """One forward pass with internals capture — no generation. Mirrors
    ``_mm_infer_on``'s input construction, then calls the model directly with
    ``output_hidden_states``/``output_attentions`` (see module docstring for
    the token/hidden-state-length caveat this does NOT resolve)."""
    import torch

    input_ids, attention_mask, tensor = _build_multimodal_inputs(
        image_or_video, instruct, model, tokenizer, device, dtype, modal)
    want_hidden = Capability.HIDDEN_STATES in capture
    want_attn = Capability.ATTENTION in capture

    with torch.inference_mode():
        out = model(
            input_ids=input_ids, attention_mask=attention_mask, images=tensor,
            output_hidden_states=want_hidden, output_attentions=want_attn,
            use_cache=False, return_dict=True,
        )

    provided = {Capability.LOGITS}
    hidden_states = None
    if want_hidden and out.hidden_states is not None:
        hidden_states = [h[0].detach().cpu() for h in out.hidden_states]
        provided.add(Capability.HIDDEN_STATES)
    attentions = None
    if want_attn and out.attentions is not None:
        attentions = [a[0].detach().cpu() for a in out.attentions]
        provided.add(Capability.ATTENTION)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    return Trace(
        tokens=tokens, token_ids=input_ids[0].tolist(), provided=provided,
        attentions=attentions, hidden_states=hidden_states,
        logits=out.logits[0].detach().cpu(),
        extras={"note": "tokens/token_ids are PRE-fusion prompt tokens; the "
                         "video/audio placeholder expands to a variable number "
                         "of embeddings not reflected here — see module docstring"},
    )


# ---------------------------------------------------------------------------
# Real model
# ---------------------------------------------------------------------------

class VideoLLaMA2AVModel(Model):
    """GENERATE + HIDDEN_STATES + ATTENTION (when constructed with
    ``want_attention=True``, which forces eager attention at load time — the
    same eager-required-for-capture rule evalrx applies to registered
    models). See the module docstring for the one real gap: no
    ``token_type_map`` (video/audio token span localization) yet."""

    modalities = frozenset({"text", "video", "audio"})

    def __init__(self, model_path: str, device: str = "cuda",
                 max_new_tokens: int = 128, load_4bit: bool = False,
                 want_attention: bool = False):
        import torch
        from videollama2 import model_init

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.max_new_tokens = max_new_tokens
        self.capabilities = frozenset(
            {Capability.GENERATE, Capability.HIDDEN_STATES, Capability.LOGITS}
            | ({Capability.ATTENTION} if want_attention else set())
        )
        load_kwargs: dict[str, Any] = {"device": device}
        if load_4bit:
            load_kwargs["load_4bit"] = True
        if want_attention:
            # sdpa/flash silently return None for attentions — same rule
            # evalrx' own HFLocalModel applies to registered models.
            load_kwargs["attn_implementation"] = "eager"
        model, processor, tokenizer = model_init(model_path, **load_kwargs)
        if device != "cuda":
            model = model.to(device=device, dtype=self.dtype)
        self._model, self._processor, self._tokenizer = model, processor, tokenizer

    def generate(self, inputs: Any, **kwargs) -> str:
        prompt = inputs.prompt if isinstance(inputs, Inputs) else str(inputs)
        video_path = inputs.video if isinstance(inputs, Inputs) else None
        if not video_path:
            raise ValueError("VideoLLaMA2AVModel.generate requires Inputs.video (a video path)")
        # va=True: audio is auto-extracted from the mp4's own audio track —
        # no separate waveform file needed (Music-AVQA videos carry audio).
        tensor = self._processor["video"](str(video_path), va=True)
        return _mm_infer_on(
            tensor, prompt, model=self._model, tokenizer=self._tokenizer,
            device=self.device, dtype=self.dtype, modal="video",
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            do_sample=kwargs.get("do_sample", False),
        )

    def generate_with_meta(self, inputs: Any, **kwargs) -> "tuple[str, dict]":
        """generate() + (finish_reason, generation_config) -- NOT part of the
        Model ABC (generate() must return a bare str for every caller in the
        framework), used only by mine_cases.py's baseline pass so FixAgent's
        L0 telemetry gate (fix_agent.py:_l0_candidates) has real per-case
        evidence to propose the cheap, precise "raise max_tokens" repair from,
        instead of that hypothesis only ever reaching slow/unreliable
        LLM-authored L1/L2 coded pipelines."""
        prompt = inputs.prompt if isinstance(inputs, Inputs) else str(inputs)
        video_path = inputs.video if isinstance(inputs, Inputs) else None
        if not video_path:
            raise ValueError("VideoLLaMA2AVModel.generate requires Inputs.video (a video path)")
        tensor = self._processor["video"](str(video_path), va=True)
        return _mm_infer_on(
            tensor, prompt, model=self._model, tokenizer=self._tokenizer,
            device=self.device, dtype=self.dtype, modal="video",
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            do_sample=kwargs.get("do_sample", False),
            return_meta=True,
        )

    def forward(self, inputs: Any, capture: "set[Capability]", spec=None) -> Trace:
        prompt = inputs.prompt if isinstance(inputs, Inputs) else str(inputs)
        video_path = inputs.video if isinstance(inputs, Inputs) else None
        if not video_path:
            raise ValueError("VideoLLaMA2AVModel.forward requires Inputs.video (a video path)")
        if Capability.ATTENTION in capture and Capability.ATTENTION not in self.capabilities:
            raise ValueError(
                "ATTENTION was requested but this handle was built with "
                "want_attention=False (eager attention was not forced at load "
                "time) — construct VideoLLaMA2AVModel(..., want_attention=True)")
        tensor = self._processor["video"](str(video_path), va=True)
        return _forward_on(
            tensor, prompt, model=self._model, tokenizer=self._tokenizer,
            device=self.device, dtype=self.dtype, capture=capture, modal="video",
        )


# ---------------------------------------------------------------------------
# Mock model — proves the M1-M4 wiring without loading real weights
# ---------------------------------------------------------------------------

class MockAVModel(Model):
    """Deterministic-per-case fake answers, seeded by video_id/question_id so
    reruns are reproducible. Never reads the case's gold answer (that would be
    circular) — it draws from the dataset's small answer vocabulary and gets
    a realistic mix of PASS/FAIL by chance, which is all the loop wiring
    needs to exercise its code paths (M1 probes, M2 stats, M3 diagnosis, M4
    surgery/fix all need SOME fail/pass contrast to have something to say)."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "video", "audio"})

    _VOCAB = ["yes", "no", "one", "two", "three", "four", "guitar", "piano",
              "violin", "flute", "trumpet", "drum"]

    def generate(self, inputs: Any, **kwargs) -> str:
        prompt = inputs.prompt if isinstance(inputs, Inputs) else str(inputs)
        video = str(getattr(inputs, "video", "") or "")
        rng = random.Random(hash((video, prompt)) & 0xFFFFFFFF)
        return rng.choice(self._VOCAB)

    def forward(self, inputs: Any, capture: set[Capability], spec=None) -> Trace:
        raise NotImplementedError("MockAVModel is GENERATE-only.")
