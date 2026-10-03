"""``jax_local`` backend: white-box access to JAX models through an adapter.

Twin of ``hf_local`` for models that run under JAX (Gemma 4 through Google
DeepMind's ``gemma`` library is the reference; a user's own Flax / MaxText /
Penzai model plugs in through :func:`evalrx.wrap_jax`). The backend knows no
framework: it drives a :class:`~evalrx.models.backends.jax.protocol.JaxModelAdapter`
(built from ``ModelSpec.jax.adapter`` or passed in) and converts every captured
array to a CPU torch tensor at the ``Trace`` boundary, so every analyzer that
runs on ``hf_local`` text models runs here unchanged.

Capabilities: ``GENERATE`` / ``LOGITS`` / ``LOGPROBS`` / ``HIDDEN_STATES``
always; ``ATTENTION`` only when the adapter runs reference (materialised
softmax) attention; ``TOOL_CALLS`` when the spec's template renders tools.
Modalities come from the adapter (``Inputs.image`` / ``audio`` / ``video`` go
through ``adapter.encode``; the ``Encoding`` masks become ``image_token_mask``,
``audio_token_mask``, ``image_spatial_shape`` and the ``TokenTypeMap`` on the
Trace, the same fields hf_local fills). Not yet: ``GRADIENTS``, the L3a
executors and L3b interventions (``docs/design_jax_backend.md``).

Calls into the adapter are serialised on one re-entrant lock per model: the
analyzer stages run probes from a thread pool, and the JAX stack underneath is
not thread-safe (kauldron's ``ktyping`` keeps its type-check scopes on a
process-global stack, so two concurrent ``Gemma4Sampler.sample`` calls fail its
``assert s == self``; Flax tracing is no safer). hf_local needs no such lock
because torch ops are.

jax and torch are imported lazily inside ``load()`` / the boundary, so this
module imports on the light install and the registry stays torch-free.
"""

from __future__ import annotations

import functools
import importlib
import logging
import os
import sys
import threading
from typing import Any

from evalrx.core.capability import Capability, CapabilityError
from evalrx.core.case import Inputs
from evalrx.core.model import Model, TokenLogprob, Trace
from evalrx.core.tokentype import TokenTypeMap
from evalrx.core.tool import ChatTurn
from evalrx.models.backends.base import Backend, RuntimeConfig
from evalrx.models.backends.jax.protocol import (
    CAPTURE_ATTN,
    CAPTURE_HIDDEN,
    CAPTURE_LOGITS,
    Encoding,
    JaxModelAdapter,
    SamplingParams,
)

logger = logging.getLogger(__name__)

_CAPTURE_KEYS = {
    Capability.LOGITS: CAPTURE_LOGITS,
    Capability.HIDDEN_STATES: CAPTURE_HIDDEN,
    Capability.ATTENTION: CAPTURE_ATTN,
}
_UNSET = object()


def load_adapter_factory(path: str):
    """Resolve ``JaxSpec.adapter`` (``"pkg.module:factory"``) to the callable."""
    module, sep, name = path.partition(":")
    if not sep or not module or not name:
        raise ValueError(f"JaxSpec.adapter must look like 'pkg.module:factory', got {path!r}")
    return getattr(importlib.import_module(module), name)


def configure_jax_runtime(device: str) -> None:
    """Process-level JAX knobs from ``RuntimeConfig.device``.

    They only take effect before jax initialises its backend, so ``load()``
    calls this first. ``XLA_PYTHON_CLIENT_PREALLOCATE`` is turned off because
    XLA's default grabs 75 % of a card up front, which on a shared GPU box
    kills the load; allocation then happens on demand like torch.
    """
    jax_loaded = "jax" in sys.modules
    if not jax_loaded:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    kind = str(device).lower().split(":", 1)[0]          # "cuda:0" / "tpu:0" pick the platform
    platform = {"cpu": "cpu", "cuda": "cuda", "gpu": "cuda", "tpu": "tpu"}.get(kind)
    if platform is None:  # "auto" and friends: let jax pick
        return
    if jax_loaded:
        import jax

        have = jax.default_backend()
        want = "gpu" if platform == "cuda" else platform
        if have != want:
            logger.warning(
                "jax is already initialised on %s; RuntimeConfig.device=%r cannot be applied "
                "(set JAX_PLATFORMS before importing jax)", have, device,
            )
        return
    # cpu stays initialised next to the accelerator: adapters restore and cast
    # checkpoints in host memory first (the first platform listed is the default)
    os.environ.setdefault("JAX_PLATFORMS", platform if platform == "cpu" else f"{platform},cpu")


def _serialized(method):
    """Run ``method`` under the model's lock (re-entrant: ``logprobs`` calls
    ``_encode`` and the adapter twice, ``chat`` renders then generates)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class JaxLocalModel(Model):
    """A JAX model behind a :class:`JaxModelAdapter`, lazily loaded."""

    def __init__(self, spec, runtime: RuntimeConfig, adapter: JaxModelAdapter | None = None) -> None:
        if adapter is None:
            if getattr(spec, "jax", None) is None:
                raise ValueError(
                    f"{spec.key!r} has no JaxSpec: the jax_local backend needs ModelSpec.jax "
                    "(checkpoint, tokenizer, adapter); see docs/design_jax_backend.md section 3.2"
                )
            adapter = load_adapter_factory(spec.jax.adapter)(spec, runtime)
        self.spec = spec
        self.runtime = runtime
        self.adapter = adapter
        self._lock = threading.RLock()
        self._loaded = False
        self._unembed: Any = _UNSET
        self._final_norm: Any = _UNSET
        caps = {Capability.GENERATE, Capability.LOGITS, Capability.LOGPROBS, Capability.HIDDEN_STATES}
        if getattr(adapter, "reference_attention", False):
            caps.add(Capability.ATTENTION)
        if getattr(spec, "tool_calling", False):
            caps.add(Capability.TOOL_CALLS)
        self.capabilities = frozenset(caps)
        self.modalities = frozenset(getattr(adapter, "modalities", frozenset({"text"})))

    # -- lazy load -----------------------------------------------------
    @_serialized
    def load(self) -> None:
        if self._loaded:
            return
        configure_jax_runtime(self.runtime.device)
        logger.info(
            "loading %s (backend=jax_local, adapter=%s, device=%s, dtype=%s)",
            self.spec.key, type(self.adapter).__name__, self.runtime.device, self.runtime.dtype,
        )
        self.adapter.load()
        self._loaded = True

    def _ensure(self) -> None:
        if not self._loaded:
            self.load()

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _as_inputs(inputs: Any) -> Inputs:
        return inputs if isinstance(inputs, Inputs) else Inputs(prompt=str(inputs))

    @_serialized
    def _encode(self, inputs: Any) -> Encoding:
        self._ensure()
        return self.adapter.encode(self._as_inputs(inputs), chat_template=self.runtime.apply_chat_template)

    def _sampling(self, kwargs: dict) -> SamplingParams:
        """hf_local's kwarg contract: ``max_new_tokens`` wins over the OpenAI-style
        ``max_tokens``; ``temperature <= 0`` (or ``do_sample=False``) is greedy;
        ``do_sample=True`` without a temperature samples at 1.0."""
        kwargs = dict(kwargs)
        max_new = kwargs.pop("max_new_tokens", None)
        if max_new is None:
            max_new = kwargs.pop("max_tokens", self.runtime.max_new_tokens)
        else:
            kwargs.pop("max_tokens", None)
        do_sample = kwargs.pop("do_sample", None)
        temperature = kwargs.pop("temperature", None)
        if temperature is None:
            temperature = 1.0 if do_sample else 0.0
        elif do_sample is False:
            temperature = 0.0
        stop = kwargs.pop("stop", None) or []
        if isinstance(stop, str):
            stop = [stop]
        return SamplingParams(
            max_new_tokens=int(max_new),
            temperature=float(temperature),
            top_p=float(kwargs.pop("top_p", 1.0) or 1.0),
            top_k=int(kwargs.pop("top_k", 0) or 0),
            seed=kwargs.pop("seed", None),
            stop=[str(s) for s in stop],
        )

    @staticmethod
    def _truncate(text: str, stop: list[str]) -> str:
        cut = len(text)
        for s in stop:
            i = text.find(s)
            if i >= 0:
                cut = min(cut, i)
        return text[:cut]

    # -- interface -----------------------------------------------------
    @_serialized
    def generate(self, inputs: Any, **kwargs) -> str:
        enc = self._encode(inputs)
        params = self._sampling(kwargs)
        out = self.adapter.generate(enc, params)
        return self._truncate(out.text, params.stop)

    @_serialized
    def logprobs(
        self, inputs: Any, max_new_tokens: int = 64, top_k: int = 5, **kwargs
    ) -> list[TokenLogprob]:
        """Per-output-token logprobs: greedy decode, then ONE teacher-forced forward
        over prompt + continuation (exact for greedy, one extra pass)."""
        import numpy as np

        from evalrx.models.backends.jax.boundary import to_numpy

        enc = self._encode(inputs)
        gen = self.adapter.generate(enc, SamplingParams(max_new_tokens=int(max_new_tokens)))
        if not gen.ids:
            return []
        gen_tokens = [self.adapter.decode([int(t)]) for t in gen.ids]
        full = enc.extended([int(t) for t in gen.ids], gen_tokens)
        out = self.adapter.forward(full, capture=frozenset({CAPTURE_LOGITS}))
        logits = to_numpy(out.logits).astype(np.float32)  # (S, V)
        n_prompt = len(enc.ids)
        result: list[TokenLogprob] = []
        for i, tid in enumerate(gen.ids):
            row = logits[n_prompt - 1 + i]
            m = row.max()
            lp = row - m - np.log(np.exp(row - m).sum())
            k = min(int(top_k), lp.shape[-1])
            idx = np.argpartition(-lp, k - 1)[:k]
            idx = idx[np.argsort(-lp[idx])]
            top = {self.adapter.decode([int(j)]): float(lp[j]) for j in idx}
            result.append(TokenLogprob(token=gen_tokens[i], logprob=float(lp[int(tid)]), top=top))
        return result

    @_serialized
    def forward(self, inputs: Any, capture: set[Capability], spec=None) -> Trace:
        import torch

        from evalrx.models.backends.jax.boundary import to_torch

        if Capability.ATTENTION in capture and Capability.ATTENTION not in self.capabilities:
            raise CapabilityError(analyzer="forward", model=repr(self), missing={Capability.ATTENTION})
        enc = self._encode(inputs)
        want = frozenset(_CAPTURE_KEYS[c] for c in capture if c in _CAPTURE_KEYS)
        layers = list(spec.layers) if spec is not None and spec.layers is not None else None
        heads = list(spec.heads) if spec is not None and spec.heads is not None else None
        out = self.adapter.forward(enc, capture=want, layers=None if layers is None else tuple(layers))

        def _maybe_subset(seq):
            return [seq[i] for i in layers] if layers is not None else list(seq)

        provided: set[Capability] = set()
        attentions = hidden_states = logits = None
        if Capability.ATTENTION in capture:
            if out.attn is None:
                raise RuntimeError(
                    f"{self!r}: ATTENTION was requested but the adapter returned no attention "
                    "probabilities. Run reference attention (JaxSpec.reference_attention=True and "
                    "RuntimeConfig.attn_impl in (None, 'eager')); fused kernels never materialise them."
                )
            attentions = [to_torch(a) for a in _maybe_subset(out.attn)]
            if heads is not None:
                attentions = [a[heads] for a in attentions]
            provided.add(Capability.ATTENTION)
        if Capability.HIDDEN_STATES in capture and out.hidden is not None:
            hidden_states = [to_torch(h) for h in _maybe_subset(out.hidden)]
            provided.add(Capability.HIDDEN_STATES)
        if Capability.LOGITS in capture:
            if out.logits is None:
                raise RuntimeError(f"{self!r}: LOGITS was requested but the adapter returned none")
            logits = to_torch(out.logits)
            provided.add(Capability.LOGITS)

        extras: dict = {"attn_semantics": self.spec.attn_semantics.value, **dict(out.extras or {})}
        if enc.image_token_mask is not None:
            extras["image_token_mask"] = torch.tensor(enc.image_token_mask, dtype=torch.bool)
        if enc.audio_token_mask is not None:
            extras["audio_token_mask"] = torch.tensor(enc.audio_token_mask, dtype=torch.bool)
        # hf_local's image_spatial_shape (post-merge (H, W) grid) when the adapter
        # knows one grid for every image; relative-attention overlays reshape by it
        if "image_spatial_shape" not in extras and enc.grids and len({(g[1], g[2]) for g in enc.grids}) == 1:
            extras["image_spatial_shape"] = (int(enc.grids[0][1]), int(enc.grids[0][2]))
        ttm = None
        if enc.image_token_mask is not None and any(enc.image_token_mask):
            image_pos = [i for i, v in enumerate(enc.image_token_mask) if v]
            image_set = set(image_pos)
            ttm = TokenTypeMap(
                seq_len=len(enc.ids),
                image_pos=image_pos,
                text_pos=[i for i in range(len(enc.ids)) if i not in image_set],
                grids=list(enc.grids),
                image_token_id=enc.image_token_id,
            )
        return Trace(
            tokens=list(enc.tokens),
            token_ids=[int(t) for t in enc.ids],
            provided=provided,
            attentions=attentions,
            hidden_states=hidden_states,
            logits=logits,
            token_type_map=ttm,
            extras=extras,
        )

    @_serialized
    def chat(self, messages: list, tools=None) -> ChatTurn:
        if Capability.TOOL_CALLS not in self.capabilities:
            raise CapabilityError(analyzer="chat", model=repr(self), missing={Capability.TOOL_CALLS})
        self._ensure()
        enc = self.adapter.render_chat(messages, tools)
        out = self.adapter.generate(enc, SamplingParams(max_new_tokens=self.runtime.max_new_tokens))
        usage = {"prompt_tokens": len(enc.ids), "completion_tokens": len(out.ids)}
        return ChatTurn(text=out.text, raw_tool_calls=None, usage=usage)  # codec parses the text

    # -- lens accessors --------------------------------------------------
    @_serialized
    def unembed_weight(self):
        """The ``(vocab, dim)`` unembedding the model really applies, converted once."""
        self._ensure()
        if self._unembed is _UNSET:
            from evalrx.models.backends.jax.boundary import to_torch

            W = self.adapter.unembed()
            self._unembed = None if W is None else to_torch(W)
        return self._unembed

    @_serialized
    def final_norm(self):
        """The head-side norm as a small torch module (``None`` if the adapter has none)."""
        self._ensure()
        if self._final_norm is _UNSET:
            from evalrx.models.backends.jax.boundary import make_torch_rmsnorm

            params = self.adapter.final_norm_params()
            self._final_norm = None if params is None else make_torch_rmsnorm(params)
        return self._final_norm

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "lazy"
        return f"JaxLocalModel(key={self.spec.key!r}, adapter={type(self.adapter).__name__}, {status})"


class JaxLocalBackend(Backend):
    kind = "jax_local"
    # Superset the backend CAN provide; the per-model set is computed in
    # JaxLocalModel.__init__ (ATTENTION needs reference attention, TOOL_CALLS a
    # tool-rendering template), the same shape as hf_local.
    capabilities = frozenset({
        Capability.GENERATE,
        Capability.TOOL_CALLS,
        Capability.LOGPROBS,
        Capability.LOGITS,
        Capability.HIDDEN_STATES,
        Capability.ATTENTION,
    })

    def build(self, spec, runtime: RuntimeConfig) -> JaxLocalModel:
        return JaxLocalModel(spec, runtime)
