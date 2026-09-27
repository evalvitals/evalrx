"""Gemma 4 on JAX: the reference adapter for the ``jax_local`` backend.

Runs the Gemma 4 E2B / E4B checkpoints through Google DeepMind's ``gemma``
library (Flax Linen) behind the :class:`~evalrx.models.jax.protocol.JaxModelAdapter`
contract. Phase 1 of ``docs/design_jax_backend.md``: text prompts, read access
(logits, hidden states, attention probabilities), greedy / sampled generation,
teacher-forced logprobs. Image and audio inputs raise ``NotImplementedError``
until phase 3.

Facts this file depends on, verified against gemma 4.0.1 / flax 0.12.10 on
2026-09-26 (design doc, section 4):

* blocks are ``layer_{i}`` (35 on E2B); ``final_norm`` is a plain RMSNorm,
  ``x * rsqrt(mean(x^2) + 1e-6) * scale``; logits are
  ``x @ embedder.input_embedding.T`` followed by a tanh soft-cap at 30;
* attention is a materialised softmax whose probabilities pass through an
  identity module named ``attention_weights`` inside each block's ``attn``, so
  Flax ``capture_intermediates`` reads them without touching the model code;
  ``return_hidden_states`` returns exactly the ``final_norm`` output;
* the library's ``Transformer.__call__`` is wrapped in ``nn.jit``, whose cached
  trace bakes in the FIRST capture filter it sees (a second call with another
  filter silently returns the first filter's intermediates). This adapter
  therefore calls the layer beneath that wrapper through its own
  per-configuration ``jax.jit``;
* right-padding with PAD (0) leaves valid positions' outputs unchanged (max
  abs diff 1e-7 on a random model), so lengths are bucketed to bound recompiles;
* chat format (``dialog.Format.GEMMA4``): ``<|turn>user\\n...<turn|>\\n<|turn>model\\n``.
  Thinking is switched on by the ``<|think|>`` control token (id 98); the
  specs' ``enable_thinking=False`` therefore renders none of it. The exact
  placement the HF ``chat_template.jinja`` uses for ``enable_thinking=True`` was
  not verifiable here (gated repo), so that setting raises for now;
* checkpoints: public ``gs://gemma-data/checkpoints/gemma4-{e2b,e4b}-it``
  (Orbax; 18 GB for E2B including the vision and audio towers, which
  ``text_only=True`` skips); tokenizer ``gs://gemma-data/tokenizers/tokenizer_gemma4.model``
  (SentencePiece, 4.5 MB). Both read anonymously; a local mirror is picked up
  through ``RuntimeConfig.engine_kwargs["checkpoint"]`` (a ``tokenizer_gemma4.model``
  next to it, or in it, is used automatically).

jax and gemma are imported lazily inside ``load()``.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

from evalrx.core.case import Inputs
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
TOKENIZER_FILENAME = "tokenizer_gemma4.model"
_DTYPES = {
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float32": "float32", "fp32": "float32",
    "float16": "float16", "fp16": "float16",
}


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


def _content_text(content: Any) -> str:
    """The text of an OpenAI / transformers-style message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in ("image", "audio", "video", "image_url", "input_audio"):
                    continue
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return "" if content is None else str(content)


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


class GemmaJaxAdapter:
    """Gemma 4 E2B / E4B through the ``gemma`` library, text modality (phase 1)."""

    modalities = frozenset({"text"})

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
        self.text_only = bool(kw.pop("text_only", True))
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
        params = gm.ckpts.load_params(self.checkpoint, text_only=self.text_only)
        # the public checkpoints store float32 (19.8 GB for text-only E2B); honour
        # RuntimeConfig.dtype the way hf_local's torch_dtype does (bf16 = 9.9 GB)
        params = jax.tree.map(
            lambda a: a.astype(dtype)
            if jnp.issubdtype(a.dtype, jnp.floating) and a.dtype != dtype else a,
            params,
        )
        self._params = params
        self._forward_cache.clear()
        n_bytes = sum(int(a.nbytes) for a in jax.tree.leaves(params))
        logger.info(
            "GemmaJaxAdapter: %s loaded from %s in %.0fs (%.1f GB params, %s, %d layers, "
            "reference_attention=%s, jax backend=%s)",
            self.model_class, self.checkpoint, time.monotonic() - t0, n_bytes / 1e9, dtype_name,
            self._n_layers, self.reference_attention, jax.default_backend(),
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

    # -- text rendering ---------------------------------------------------
    def _thinking_requested(self) -> bool:
        kwargs = dict(getattr(self.spec, "chat_template_kwargs", None) or {})
        return bool(kwargs.get("enable_thinking", False))

    def render_messages(self, messages: list, *, tools: list | None = None) -> str:
        """Gemma 4 turn format via the ``dialog`` package (``Format.GEMMA4``),
        ending with an open ``<|turn>model\\n`` for the model to fill."""
        import dialog

        if tools:
            raise NotImplementedError("GemmaJaxAdapter: tool rendering is not implemented (phase 3)")
        if self._thinking_requested():
            raise NotImplementedError(
                "GemmaJaxAdapter renders thinking OFF only: where Gemma 4's official template "
                "places the <|think|> control token for enable_thinking=True is not verified here; "
                "run with chat_template_kwargs={'enable_thinking': False} (the specs' default)"
            )
        turns = []
        for m in messages:
            role = str(m.get("role", "user")).lower()
            text = _content_text(m.get("content"))
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

    def _encode_text(self, text: str) -> Encoding:
        ids = [int(i) for i in self._tok.encode(text, add_bos=True)]
        return Encoding(ids=ids, tokens=[self._token_str(i) for i in ids], text=text)

    # -- protocol: encoding ---------------------------------------------------
    def encode(self, inputs: Any, *, chat_template: bool) -> Encoding:
        self._ensure_loaded()
        inputs = inputs if isinstance(inputs, Inputs) else Inputs(prompt=str(inputs))
        if inputs.image is not None or inputs.audio is not None or inputs.video is not None:
            raise NotImplementedError(
                "GemmaJaxAdapter (phase 1) is text-only: image / audio / video inputs arrive with "
                "phase 3 of docs/design_jax_backend.md"
            )
        text = (
            self.render_messages([{"role": "user", "content": inputs.prompt}])
            if chat_template else inputs.prompt
        )
        return self._encode_text(text)

    def render_chat(self, messages: list, tools: list | None = None) -> Encoding:
        self._ensure_loaded()
        return self._encode_text(self.render_messages(messages, tools=tools))

    def decode(self, ids: list[int]) -> str:
        """One id -> its surface form (specials kept visible, for token labels);
        several -> text with specials skipped, like ``skip_special_tokens=True``."""
        self._ensure_loaded()
        ids = [int(i) for i in ids]
        if len(ids) == 1:
            return self._token_str(ids[0])
        return self._text(ids)

    # -- protocol: read internals ----------------------------------------------
    def _forward_fn(self, want_h: bool, want_a: bool, layer_ids: tuple, hidden_ids: tuple):
        """A ``jax.jit``-ed forward for one capture configuration (cached)."""
        key = (want_h, want_a, layer_ids if want_a else (), hidden_ids if want_h else ())
        fn = self._forward_cache.get(key)
        if fn is not None:
            return fn
        import jax

        model, raw, n = self._model, self._raw_call, self.n_layers
        layer_set, hidden_set = set(layer_ids), set(hidden_ids)

        def filt(mdl, method: str) -> bool:
            name = mdl.name or ""
            if method == "encode":
                return want_h and name == "embedder" and 0 in hidden_set
            if method != "__call__":
                return False
            if name.startswith("layer_"):
                return want_h and (int(name[6:]) + 1) in hidden_set
            if name == "final_norm":
                return want_h and n in hidden_set
            if name == "attention_weights" and want_a:
                block = getattr(getattr(mdl, "parent", None), "parent", None)   # attention_weights -> attn -> layer_i
                bname = getattr(block, "name", "") or ""
                return bname.startswith("layer_") and int(bname[6:]) in layer_set
            return False

        if want_h or want_a:
            def fwd(params, tokens):
                out, state = model.apply(
                    {"params": params}, tokens, return_last_only=False, method=raw,
                    capture_intermediates=filt, mutable=["intermediates"],
                )
                return out.logits, state["intermediates"]
        else:
            def fwd(params, tokens):
                out = model.apply({"params": params}, tokens, return_last_only=False, method=raw)
                return out.logits, {}

        fn = jax.jit(fwd)
        self._forward_cache[key] = fn
        return fn

    def forward(
        self,
        enc: Encoding,
        *,
        capture: frozenset[str],
        layers: tuple[int, ...] | None = None,
    ) -> ForwardOut:
        import jax.numpy as jnp
        import numpy as np
        from flax.traverse_util import flatten_dict

        self._ensure_loaded()
        n = self.n_layers
        seq = len(enc.ids)
        length = bucket_length(seq, self.buckets)
        tokens = np.full((1, length), PAD_ID, dtype=np.int32)
        tokens[0, :seq] = np.asarray(enc.ids, dtype=np.int32)
        want_h = CAPTURE_HIDDEN in capture
        want_a = CAPTURE_ATTN in capture and self.reference_attention
        if layers is None:
            layer_ids, hidden_ids = tuple(range(n)), tuple(range(n + 1))
        else:
            layer_ids = tuple(sorted({int(i) for i in layers if 0 <= int(i) < n}))
            hidden_ids = tuple(sorted({int(i) for i in layers if 0 <= int(i) <= n}))
        logits_all, inter = self._forward_fn(want_h, want_a, layer_ids, hidden_ids)(
            self._params, jnp.asarray(tokens)
        )
        inter = flatten_dict(inter) if inter else {}

        def pick(key):
            v = inter.get(key)
            return None if v is None else v[0]

        logits = logits_all[0, :seq] if CAPTURE_LOGITS in capture else None
        hidden = attn = None
        if want_h:
            hidden = [None] * (n + 1)
            emb = pick(("embedder", "encode"))
            if emb is not None:
                hidden[0] = emb[0, :seq]
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
        return ForwardOut(logits=logits, hidden=hidden, attn=attn, extras={})

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
        if 0.0 < p.top_p < 1.0:
            return gm.text.TopPSampling(p=float(p.top_p), temperature=float(p.temperature))
        if p.top_k and int(p.top_k) > 0:
            return gm.text.TopkSampling(k=int(p.top_k), temperature=float(p.temperature))
        return gm.text.RandomSampling(temperature=float(p.temperature))

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
        # the sampler tokenises text itself (add_bos=True, like encode), so feed it
        # the very string the forward pass saw
        text = enc.text if enc.text is not None else self._tok.decode(enc.ids)
        max_new = max(1, int(params.max_new_tokens))
        pad_len = bucket_length(len(enc.ids), self.buckets)
        cache_len = bucket_length(pad_len + max_new + 1, self.buckets)
        sampler = gm.text.Gemma4Sampler(
            model=self._model, params=self._params, tokenizer=self._tok,
            sampling=self._sampling_method(params),
            cache_length=cache_len, max_out_length=max_new, pad_length=pad_len,
        )
        rng = int(params.seed) if params.seed is not None else 0
        out = sampler.sample(text, max_new_tokens=max_new, rng=rng, return_state=True)
        predicted = [int(t) for t in np.asarray(out.state.predicted_tokens[0]).tolist()]
        ids = self._strip_generated(predicted)
        return GenerateOut(ids=ids, text=self._text(ids) if ids else "")

    def __repr__(self) -> str:
        return f"GemmaJaxAdapter({self.model_class}, {'loaded' if self.loaded else 'lazy'})"
