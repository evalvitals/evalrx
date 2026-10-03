"""jax_local backend on a toy JAX model: CPU only, no downloads, no gemma library.

The toy adapter exercises the same ``JaxLocalModel`` path the Gemma adapter
uses (encode -> adapter.forward -> torch at the Trace boundary -> analyzers),
mirroring ``test_wrap.py``'s FakeCausalLM for hf_local. Real-weights parity
lives in the GPU-marked test at the bottom.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
import jax.numpy as jnp  # noqa: E402

import evalrx  # noqa: E402
from evalrx.analyzers.lens.logit_lens import LogitLensAnalyzer  # noqa: E402
from evalrx.core.capability import Capability, CapabilityError  # noqa: E402
from evalrx.core.case import Inputs  # noqa: E402
from evalrx.core.model import CaptureSpec  # noqa: E402
from evalrx.models import RuntimeConfig, compose  # noqa: E402
from evalrx.models.backends import BACKENDS  # noqa: E402
from evalrx.models.backends.jax_local import JaxLocalBackend, JaxLocalModel  # noqa: E402
from evalrx.models.jax._boundary import make_torch_rmsnorm, to_torch  # noqa: E402
from evalrx.models.jax.protocol import (  # noqa: E402
    CAPTURE_ATTN,
    CAPTURE_HIDDEN,
    CAPTURE_LOGITS,
    Encoding,
    ForwardOut,
    GenerateOut,
    NormParams,
    SamplingParams,
)


# ----------------------------------------------------------------------
# A toy adapter: tiny causal "transformer" with real jax arrays
# ----------------------------------------------------------------------
class ToyAdapter:
    modalities = frozenset({"text"})

    def __init__(self, n_layers=2, dim=8, vocab=16, heads=2, reference_attention=True, seed=0):
        k1, k2 = jax.random.split(jax.random.key(seed))
        self.n_layers, self.dim, self.vocab, self.heads = n_layers, dim, vocab, heads
        self.reference_attention = reference_attention
        self.E = jax.random.normal(k1, (vocab, dim), dtype=jnp.float32)
        self.W = jax.random.normal(k2, (n_layers, dim, dim), dtype=jnp.float32) * 0.1
        self.loaded = False
        self.forward_calls = 0
        self.last_layers = "unset"

    def load(self):
        self.loaded = True

    # deterministic whitespace "tokenizer": id in 1..vocab-1 (0 is the end token)
    def _ids(self, text):
        return [1 + (sum(map(ord, w)) % (self.vocab - 1)) for w in text.split()] or [1]

    def encode(self, inputs, *, chat_template):
        text = f"<user> {inputs.prompt} <model>" if chat_template else inputs.prompt
        ids = self._ids(text)
        return Encoding(ids=ids, tokens=[f"t{i}" for i in ids], text=text)

    def render_chat(self, messages, tools=None):
        text = " ".join(f"<{m['role']}> {m['content']}" for m in messages) + " <model>"
        ids = self._ids(text)
        return Encoding(ids=ids, tokens=[f"t{i}" for i in ids], text=text)

    def decode(self, ids):
        return "".join(f"t{int(i)}" for i in ids)

    def _run(self, ids):
        x = self.E[jnp.asarray(ids)]
        S = len(ids)
        mask = jnp.tril(jnp.ones((S, S)))
        hidden, attn = [x], []
        for layer in range(self.n_layers):
            scores = (x @ self.W[layer]) @ x.T / math.sqrt(self.dim)
            per_head = [
                jax.nn.softmax(jnp.where(mask > 0, scores + 0.1 * h, -1e9), axis=-1)
                for h in range(self.heads)
            ]
            A = jnp.stack(per_head)                      # (H, S, S)
            x = x + A.mean(0) @ x
            attn.append(A)
            hidden.append(x)
        return x @ self.E.T, hidden, attn

    def forward(self, enc, *, capture, layers=None):
        self.forward_calls += 1
        self.last_layers = layers
        logits, hidden, attn = self._run(enc.ids)
        return ForwardOut(
            logits=logits if CAPTURE_LOGITS in capture else None,
            hidden=hidden if CAPTURE_HIDDEN in capture else None,
            attn=attn if (CAPTURE_ATTN in capture and self.reference_attention) else None,
            extras={"toy": True},
        )

    def generate(self, enc, params: SamplingParams):
        ids, new = list(enc.ids), []
        for _ in range(params.max_new_tokens):
            logits, _, _ = self._run(ids)
            nxt = int(jnp.argmax(logits[-1]))
            if nxt == 0:
                break
            new.append(nxt)
            ids.append(nxt)
        return GenerateOut(ids=new, text=self.decode(new))

    def unembed(self):
        return self.E

    def final_norm_params(self):
        return NormParams(scale=jnp.ones((self.dim,)), eps=1e-6)


# ----------------------------------------------------------------------
# registry / compose
# ----------------------------------------------------------------------
def test_backend_registered_with_white_box_capabilities():
    assert BACKENDS["jax_local"] is JaxLocalBackend
    for cap in (Capability.ATTENTION, Capability.HIDDEN_STATES, Capability.LOGITS, Capability.LOGPROBS):
        assert cap in JaxLocalBackend.capabilities


def test_compose_gemma_e2b_builds_lazily_without_loading():
    """The spec route: the gemma adapter module imports without jax/gemma being
    exercised, and no weights load at compose time."""
    m = compose("gemma-4-e2b-it", "jax_local", RuntimeConfig(max_new_tokens=8))
    assert isinstance(m, JaxLocalModel) and not m._loaded
    assert {Capability.ATTENTION, Capability.HIDDEN_STATES, Capability.LOGPROBS} <= m.capabilities
    # the spec declares vision + audio, so the towers load by default (phase 3)...
    assert m.modalities == frozenset({"text", "image", "audio"})
    assert "GemmaJaxAdapter" in repr(m)
    # ...and a text cell opts out of them (what the benchmark runner does for llm tasks)
    text_only = compose("gemma-4-e2b-it", "jax_local", RuntimeConfig(engine_kwargs={"text_only": True}))
    assert text_only.modalities == frozenset({"text"}) and text_only.adapter.text_only is True


def test_compose_refuses_spec_without_jax_twin():
    with pytest.raises(ValueError, match="JaxSpec"):
        compose("gemma-4-12b-it", "jax_local")


def test_reference_attention_off_drops_attention_capability():
    m = evalrx.wrap_jax(ToyAdapter(reference_attention=False))
    assert Capability.ATTENTION not in m.capabilities
    with pytest.raises(CapabilityError):
        m.forward("a b", capture={Capability.ATTENTION})
    with pytest.raises(CapabilityError):
        evalrx.wrap_jax(ToyAdapter(reference_attention=False), want={Capability.ATTENTION})


# ----------------------------------------------------------------------
# wrap_jax -> forward: Trace shapes, dtypes, subsetting
# ----------------------------------------------------------------------
def test_wrap_jax_forward_produces_hf_shaped_trace():
    ad = ToyAdapter(n_layers=3, dim=8, vocab=16, heads=4)
    m = evalrx.wrap_jax(ad)
    assert isinstance(m, JaxLocalModel) and m.adapter is ad
    trace = m.forward("hello brave new world", capture={Capability.ATTENTION, Capability.HIDDEN_STATES, Capability.LOGITS})
    assert ad.loaded                                   # lazy load happened on first use
    S = len(trace.token_ids)
    assert S == 4 and trace.tokens == [f"t{i}" for i in trace.token_ids]
    assert trace.provided == {Capability.ATTENTION, Capability.HIDDEN_STATES, Capability.LOGITS}
    assert len(trace.hidden_states) == 4 and all(isinstance(h, torch.Tensor) for h in trace.hidden_states)
    assert tuple(trace.hidden_states[0].shape) == (S, 8)
    assert len(trace.attentions) == 3 and tuple(trace.attentions[0].shape) == (4, S, S)
    assert tuple(trace.logits.shape) == (S, 16) and trace.logits.dtype == torch.float32
    assert trace.extras["attn_semantics"] == "standard" and trace.extras["toy"] is True
    assert trace.token_type_map is None                # text-only encoding
    # attention rows are probabilities
    assert torch.allclose(trace.attentions[0].sum(-1), torch.ones(4, S), atol=1e-5)


def test_forward_only_fills_requested_captures():
    m = evalrx.wrap_jax(ToyAdapter())
    trace = m.forward("a b c", capture={Capability.HIDDEN_STATES})
    assert trace.provided == {Capability.HIDDEN_STATES}
    assert trace.attentions is None and trace.logits is None
    with pytest.raises(ValueError, match="attention"):
        trace.require(Capability.ATTENTION)


def test_capture_spec_layers_and_heads_subset_like_hf_local():
    ad = ToyAdapter(n_layers=3, heads=4)
    m = evalrx.wrap_jax(ad)
    spec = CaptureSpec(layers=[0, 2], heads=[1, 3])
    trace = m.forward("a b c", capture={Capability.ATTENTION, Capability.HIDDEN_STATES}, spec=spec)
    assert ad.last_layers == (0, 2)                    # passed through so adapters can skip work
    assert len(trace.attentions) == 2 and len(trace.hidden_states) == 2
    assert tuple(trace.attentions[0].shape) == (2, 3, 3)


def test_forward_uses_chat_template_when_runtime_asks():
    ad = ToyAdapter()
    plain = evalrx.wrap_jax(ad).forward("x y", capture={Capability.LOGITS})
    templated = evalrx.wrap_jax(ad, apply_chat_template=True).forward("x y", capture={Capability.LOGITS})
    assert len(templated.token_ids) == len(plain.token_ids) + 2


# ----------------------------------------------------------------------
# generate / logprobs / chat
# ----------------------------------------------------------------------
def test_generate_is_greedy_and_deterministic_by_default():
    m = evalrx.wrap_jax(ToyAdapter(), max_new_tokens=5)
    a, b = m.generate("the cat sat"), m.generate("the cat sat")
    assert isinstance(a, str) and a == b and 0 < len(a) <= 5 * 3
    assert m.generate("the cat sat", max_tokens=2) == a[: len(m.generate("the cat sat", max_new_tokens=2))]


def test_generate_kwargs_map_to_sampling_params():
    m = evalrx.wrap_jax(ToyAdapter(), max_new_tokens=7)
    p = m._sampling({"max_tokens": 3, "temperature": 0.6, "top_p": 0.9, "top_k": 20, "stop": "STOP"})
    assert (p.max_new_tokens, p.temperature, p.top_p, p.top_k, p.stop) == (3, 0.6, 0.9, 20, ["STOP"])
    assert m._sampling({}).max_new_tokens == 7 and m._sampling({}).greedy
    assert m._sampling({"do_sample": True}).temperature == 1.0
    assert m._sampling({"temperature": 0.7, "do_sample": False}).greedy
    assert m._sampling({"max_new_tokens": 2, "max_tokens": 9}).max_new_tokens == 2


class ScopedToyAdapter(ToyAdapter):
    """Mimics the JAX stack's process-global state: like kauldron's ktyping
    scopes, every call pushes onto one shared stack and asserts it pops what
    it pushed, so two interleaved calls fail the way the real sampler does."""

    stack: list = []

    def _scoped(self, fn):
        import time

        token = object()
        self.stack.append(token)
        time.sleep(0.005)                      # widen the race window
        out = fn()
        assert self.stack.pop() is token, "interleaved call on the shared scope stack"
        return out

    def forward(self, enc, *, capture, layers=None):
        return self._scoped(lambda: super(ScopedToyAdapter, self).forward(enc, capture=capture, layers=layers))

    def generate(self, enc, params):
        return self._scoped(lambda: super(ScopedToyAdapter, self).generate(enc, params))


def test_adapter_calls_are_serialised_across_threads():
    """M1 runs probes from a thread pool; concurrent Gemma4Sampler.sample calls
    tripped kauldron's `assert s == self` (chartqa shakedown, 2026-10-02)."""
    from concurrent.futures import ThreadPoolExecutor

    m = evalrx.wrap_jax(ScopedToyAdapter(), max_new_tokens=4)
    prompts = [f"prompt number {i} words" for i in range(12)]

    def job(i):
        text = m.generate(prompts[i], temperature=1.0)
        trace = m.forward(prompts[i], {Capability.LOGITS})
        return text, trace.logits.shape[-1]

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(job, range(len(prompts))))
    assert len(results) == 12 and all(v == 16 for _, v in results)
    # the lock is re-entrant: logprobs encodes, generates and forwards under it
    assert m.logprobs("the cat sat", max_new_tokens=2)


def test_logprobs_are_teacher_forced_and_consistent_with_logits():
    ad = ToyAdapter(vocab=16)
    m = evalrx.wrap_jax(ad)
    lps = m.logprobs("a b c", max_new_tokens=3, top_k=4)
    assert 1 <= len(lps) <= 3
    for lp in lps:
        assert lp.logprob <= 0.0 and len(lp.top) == 4
        assert max(lp.top.values()) >= lp.logprob      # the chosen token is greedy -> it IS the top-1
        assert lp.token in lp.top and math.isclose(lp.top[lp.token], lp.logprob, rel_tol=1e-5)
    # cross-check the first step against a direct forward over the prompt
    trace = m.forward("a b c", capture={Capability.LOGITS})
    ref = torch.log_softmax(trace.logits[-1], dim=-1)
    tid = int(torch.argmax(ref))
    assert lps[0].token == ad.decode([tid]) and math.isclose(lps[0].logprob, float(ref[tid]), rel_tol=1e-4)


def test_chat_requires_tool_calls_capability():
    m = evalrx.wrap_jax(ToyAdapter())
    with pytest.raises(CapabilityError):
        m.chat([{"role": "user", "content": "hi"}])


def test_inputs_object_and_str_encode_the_same():
    m = evalrx.wrap_jax(ToyAdapter())
    assert m.forward(Inputs(prompt="a b"), capture={Capability.LOGITS}).token_ids == \
        m.forward("a b", capture={Capability.LOGITS}).token_ids


# ----------------------------------------------------------------------
# lens accessors + an analyzer end to end
# ----------------------------------------------------------------------
def test_unembed_and_final_norm_are_torch_objects_cached_once():
    m = evalrx.wrap_jax(ToyAdapter(dim=8, vocab=16))
    W = m.unembed_weight()
    assert isinstance(W, torch.Tensor) and tuple(W.shape) == (16, 8)
    assert m.unembed_weight() is W
    norm = m.final_norm()
    assert isinstance(norm, torch.nn.Module) and next(norm.parameters()).dtype == torch.float32
    x = torch.randn(3, 8)
    y = norm(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3), atol=1e-4)   # unit RMS with scale=1


def test_logit_lens_runs_on_wrapped_jax_model():
    m = evalrx.wrap_jax(ToyAdapter(n_layers=3, dim=8, vocab=16))
    result = LogitLensAnalyzer(top_k=3).run(m, "the capital of france is")
    assert result.findings["n_layers"] == 4            # n_layers + 1 hidden states, as on hf_local
    assert result.findings["final_norm_applied"] is True
    assert all(len(layer["top"]) == 3 for layer in result.findings["per_layer_top"])


# ----------------------------------------------------------------------
# the boundary itself
# ----------------------------------------------------------------------
def test_to_torch_preserves_bfloat16_bit_exactly():
    x = (jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 7.0).astype(jnp.bfloat16)
    t = to_torch(x)
    assert t.dtype == torch.bfloat16 and tuple(t.shape) == (2, 3)
    np.testing.assert_array_equal(t.float().numpy(), np.asarray(x.astype(jnp.float32)))
    f = to_torch(jnp.ones((2,), dtype=jnp.float32))
    assert f.dtype == torch.float32
    f[0] = 5.0                                          # writable, not a read-only view


def test_torch_rmsnorm_matches_plus_one_convention():
    scale = jnp.full((4,), 0.5, dtype=jnp.float32)
    plain = make_torch_rmsnorm(NormParams(scale=scale, eps=1e-6, plus_one=False))
    offset = make_torch_rmsnorm(NormParams(scale=scale, eps=1e-6, plus_one=True))
    x = torch.randn(2, 4)
    assert torch.allclose(offset(x), plain(x) * 3.0, atol=1e-5)


# ----------------------------------------------------------------------
# media: masks / grids from the adapter become hf_local's Trace fields
# ----------------------------------------------------------------------
class MediaToyAdapter(ToyAdapter):
    """ToyAdapter that reserves 4 soft tokens (a 2x2 grid) per image and 3 per
    audio clip, the way the Gemma adapter expands <|image|> / <|audio|>."""

    modalities = frozenset({"text", "image", "audio"})
    IMAGE_ID, AUDIO_ID = 14, 15

    def encode(self, inputs, *, chat_template):
        ids = self._ids(inputs.prompt)
        image_mask = audio_mask = None
        grids = []
        if getattr(inputs, "audio", None) is not None:
            ids = [self.AUDIO_ID] * 3 + ids
            audio_mask = [True] * 3 + [False] * (len(ids) - 3)
        if getattr(inputs, "image", None) is not None:
            n = len(ids)
            ids = [self.IMAGE_ID] * 4 + ids
            image_mask = [True] * 4 + [False] * n
            audio_mask = None if audio_mask is None else [False] * 4 + audio_mask
            grids = [(1, 2, 2)]
        return Encoding(
            ids=ids, tokens=[f"t{i}" for i in ids], text=inputs.prompt,
            image_token_mask=image_mask, audio_token_mask=audio_mask, grids=grids,
            image_token_id=self.IMAGE_ID if image_mask else None,
        )


def test_media_encoding_fills_token_type_map_and_extras():
    m = evalrx.wrap_jax(MediaToyAdapter(n_layers=2, dim=8, vocab=16, heads=2))
    assert m.modalities == frozenset({"text", "image", "audio"})
    trace = m.forward(Inputs(prompt="a b", image="img.png", audio=np.zeros(8, np.float32)),
                      capture={Capability.ATTENTION, Capability.HIDDEN_STATES, Capability.LOGITS})
    S = len(trace.token_ids)
    assert S == 4 + 3 + 2
    ttm = trace.token_type_map
    assert ttm is not None and ttm.n_images == 1 and ttm.grids == [(1, 2, 2)]
    assert ttm.image_pos == [0, 1, 2, 3] and ttm.text_pos == list(range(4, S))
    assert ttm.image_token_id == MediaToyAdapter.IMAGE_ID
    assert trace.extras["image_token_mask"].dtype == torch.bool and int(trace.extras["image_token_mask"].sum()) == 4
    assert int(trace.extras["audio_token_mask"].sum()) == 3
    assert trace.extras["image_spatial_shape"] == (2, 2)   # hf_local's reshape hint, exact on jax_local
    assert tuple(trace.hidden_states[0].shape) == (S, 8) and tuple(trace.attentions[0].shape) == (2, S, S)


def test_text_only_encoding_leaves_media_fields_empty():
    trace = evalrx.wrap_jax(MediaToyAdapter()).forward("a b", capture={Capability.LOGITS})
    assert trace.token_type_map is None
    assert "image_token_mask" not in trace.extras and "image_spatial_shape" not in trace.extras


def test_logprobs_keep_media_masks_through_teacher_forcing():
    ad = MediaToyAdapter(n_layers=2, dim=8, vocab=16, heads=2)
    m = evalrx.wrap_jax(ad)
    lp = m.logprobs(Inputs(prompt="a b", image="img.png"), max_new_tokens=3, top_k=2)
    assert lp                                          # generated something
    enc = ad.encode(Inputs(prompt="a b", image="img.png"), chat_template=False)
    ext = enc.extended([5, 6], ["t5", "t6"])
    assert ext.image_token_mask == enc.image_token_mask + [False, False]
    assert ext.grids == enc.grids and ext.image_token_id == enc.image_token_id and ext.text is None


# ----------------------------------------------------------------------
# Gemma adapter helpers (no weights): token budgets, grids, rendering
# ----------------------------------------------------------------------
def _sampler_audio_count(length, sample_rate=16000):
    """Verbatim copy of Gemma4Sampler.sample's per-clip soft-token arithmetic."""
    frame_length = int(round(sample_rate * 20.0 / 1000.0))
    hop_length = int(round(sample_rate * 10.0 / 1000.0))
    t = (length - (frame_length + 1)) // hop_length + 1
    for _ in range(2):
        t = ((t + 2) - 3) // 2 + 1
    return t


def test_audio_soft_token_count_matches_the_library_sampler():
    from evalrx.models.jax.gemma import audio_soft_token_count

    for n in (16000, 16001, 16321, 160000, int(28.22 * 16000), 30 * 16000, 31 * 16000):
        assert audio_soft_token_count(n) == _sampler_audio_count(n)
    assert audio_soft_token_count(30 * 16000) == 750      # the sampler's audio_seq_length cap


def test_image_grid_product_equals_library_soft_token_count():
    pytest.importorskip("gemma")
    from gemma.gm.nn.gemma4.vision import _preprocessing as vp

    from evalrx.models.jax.gemma import image_grid

    for settings in ({"patch_size": 16, "max_soft_tokens": 280, "pooling_kernel_size": 3},
                     {"patch_size": 16, "max_soft_tokens": 1120, "pooling_kernel_size": 3}):
        for h, w in ((600, 850), (404, 310), (788, 840), (48, 48), (20, 3000), (1, 1), (4000, 30)):
            rows, cols = image_grid(h, w, **settings)
            assert rows * cols == vp.predict_soft_token_count(
                h, w, settings["patch_size"], settings["max_soft_tokens"], settings["pooling_kernel_size"])


def test_content_blocks_render_placeholders_in_order_and_collect_payloads():
    from evalrx.models.jax.gemma import _content_blocks

    text, images, audios = _content_blocks([
        {"type": "audio", "audio": "clip.wav"}, {"type": "image", "image": "a.png"},
        {"type": "image"}, {"type": "text", "text": "Describe."},
    ])
    assert text == "<|audio|><|image|><|image|>Describe."
    assert images == ["a.png"] and audios == ["clip.wav"]        # payload-less blocks are placeholders only
    assert _content_blocks("plain") == ("plain", [], [])


def test_gemma_adapter_draws_a_fresh_sampler_seed_per_unseeded_call():
    from evalrx.models.jax.gemma import GemmaJaxAdapter
    from evalrx.specs import get_spec

    ad = GemmaJaxAdapter(get_spec("gemma-4-e2b-it"), RuntimeConfig())     # lazy: no jax / gemma import
    sampled = SamplingParams(max_new_tokens=4, temperature=1.0)
    seeds = {ad._rng_seed(sampled) for _ in range(8)}
    assert len(seeds) > 1 and all(0 <= s < 2**31 for s in seeds)
    assert ad._rng_seed(SamplingParams(max_new_tokens=4, temperature=1.0, seed=7)) == 7
    assert ad._rng_seed(SamplingParams(max_new_tokens=4)) == 0           # greedy: no randomness consumed


def test_configure_jax_runtime_keeps_cpu_next_to_the_accelerator(monkeypatch):
    """The Gemma adapter restores and casts on the host, so an accelerator
    platform must not drop jax's cpu backend; indexed devices pick the platform."""
    import os
    import sys

    from evalrx.models.backends.jax_local import configure_jax_runtime

    monkeypatch.delitem(sys.modules, "jax")                  # as if jax were not initialised yet
    for device, want in (("tpu", "tpu,cpu"), ("cuda", "cuda,cpu"), ("gpu", "cuda,cpu"),
                         ("cuda:1", "cuda,cpu"), ("cpu", "cpu")):
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)
        configure_jax_runtime(device)
        assert os.environ["JAX_PLATFORMS"] == want, device
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    configure_jax_runtime("auto")
    assert "JAX_PLATFORMS" not in os.environ
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")               # a caller's explicit choice wins
    configure_jax_runtime("cuda")
    assert os.environ["JAX_PLATFORMS"] == "tpu"


def test_placement_device_honours_an_index():
    from evalrx.models.jax.gemma import placement_device

    devs = ["d0", "d1"]
    assert placement_device(devs, "tpu:1") == "d1" and placement_device(devs, "cuda:0") == "d0"
    assert placement_device(devs, "auto") == "d0" and placement_device(devs, "tpu") == "d0"
    with pytest.raises(ValueError, match="2 device"):
        placement_device(devs, "cuda:2")


_CKPT_SHAPES = {"layer_0": {"w": (4, 4)}, "final_norm": {"scale": (4,)},
                "vision_encoder": {"w": (2, 2)}, "embedder": {"input_embedding": (8, 4)}}


def _fake_gemma_load(monkeypatch, *, fake_load_params, runtime):
    """A GemmaJaxAdapter whose model class, tokenizer and checkpoint IO are fakes."""
    from gemma import gm
    from gemma.gm.ckpts import _checkpoint as ck

    import evalrx.models.jax.gemma as gj
    from evalrx.specs import get_spec

    class FakeModel:
        config = type("C", (), {"num_layers": 1})()

        def __init__(self, text_only, dtype):
            pass

    class FakeTok:
        tokens = ["<pad>", "a"]
        special_tokens = type("S", (), {"__members__": {"PAD": 0}})

        def __init__(self, path):
            pass

    meta = jax.tree.map(lambda shape: jax.ShapeDtypeStruct(shape, jnp.float32), _CKPT_SHAPES,
                        is_leaf=lambda x: isinstance(x, tuple))
    monkeypatch.setattr(ck, "_get_metadata_and_path", lambda checkpointer, path: (meta, path))
    monkeypatch.setattr(gm.ckpts, "load_params", fake_load_params)
    monkeypatch.setattr(gm.text, "Gemma4Tokenizer", FakeTok)
    monkeypatch.setattr(gj, "_below_flax", lambda fn: fn)
    ad = gj.GemmaJaxAdapter(get_spec("gemma-4-e2b-it"), runtime)
    monkeypatch.setattr(ad, "_model_cls", lambda: FakeModel)
    return ad


def test_gemma_adapter_load_restores_straight_into_the_target_dtype_and_device(monkeypatch):
    """Orbax gets a typed, placed target: text weights in the runtime dtype,
    media towers float32, every leaf on the chosen device, so the float32
    checkpoint never materialises on the host or the accelerator."""
    pytest.importorskip("gemma")
    seen = {}

    def fake_load_params(path, *, params, text_only):
        seen.update(target=params, text_only=text_only)
        return jax.tree.map(lambda s: jax.device_put(jnp.zeros(s.shape, s.dtype), s.sharding), params)

    ad = _fake_gemma_load(monkeypatch, fake_load_params=fake_load_params, runtime=RuntimeConfig(dtype="bfloat16"))
    ad.load()
    t = seen["target"]
    assert t["layer_0"]["w"].dtype == jnp.bfloat16 and t["final_norm"]["scale"].dtype == jnp.bfloat16
    assert t["vision_encoder"]["w"].dtype == jnp.float32                   # towers keep float32
    assert all(s.sharding.device_set == {jax.devices()[0]} for s in jax.tree.leaves(t))
    assert ad._params["layer_0"]["w"].dtype == jnp.bfloat16


def test_gemma_adapter_text_only_target_drops_the_towers(monkeypatch):
    pytest.importorskip("gemma")
    seen = {}

    def fake_load_params(path, *, params, text_only):
        seen.update(target=params, text_only=text_only)
        return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), params)

    ad = _fake_gemma_load(monkeypatch, fake_load_params=fake_load_params,
                          runtime=RuntimeConfig(engine_kwargs={"text_only": True}))
    ad.load()
    assert "vision_encoder" not in seen["target"] and seen["text_only"] is True


def test_gemma_adapter_load_falls_back_to_host_cast_when_the_private_api_moves(monkeypatch):
    """If gemma's checkpoint helpers change, restore float32 on the host, cast
    with numpy there and place the cast tree (same dtypes, same device)."""
    pytest.importorskip("gemma")
    seen = {}

    def fake_load_params(path, *, text_only, sharding):
        seen.update(sharding=sharding)
        return jax.tree.map(lambda shape: jax.device_put(jnp.ones(shape, jnp.float32), sharding), _CKPT_SHAPES,
                            is_leaf=lambda x: isinstance(x, tuple))

    ad = _fake_gemma_load(monkeypatch, fake_load_params=fake_load_params, runtime=RuntimeConfig(dtype="bfloat16"))

    def moved(*a, **k):
        raise AttributeError("_CheckpointTree")

    monkeypatch.setattr(ad, "_restore_target", moved)
    ad.load()
    assert seen["sharding"].device_set == {jax.local_devices(backend="cpu")[0]}
    p = ad._params
    assert p["layer_0"]["w"].dtype == jnp.bfloat16 and p["vision_encoder"]["w"].dtype == jnp.float32
    assert all(a.devices() == {jax.devices()[0]} for a in jax.tree.leaves(p))


def test_gemma_adapter_sharding_knob_validates_and_picks_fsdp(monkeypatch):
    from evalrx.models.jax.gemma import GemmaJaxAdapter
    from evalrx.specs import get_spec

    spec = get_spec("gemma-4-e2b-it")
    with pytest.raises(ValueError, match="sharding"):
        GemmaJaxAdapter(spec, RuntimeConfig(engine_kwargs={"sharding": "zero3"}))
    pytest.importorskip("gemma")
    ad = GemmaJaxAdapter(spec, RuntimeConfig(engine_kwargs={"sharding": "fsdp"}))
    assert type(ad._param_sharding(jax)).__name__ == "FSDPSharding"
    one = GemmaJaxAdapter(spec, RuntimeConfig())._param_sharding(jax)     # one CPU device here
    assert one.device_set == {jax.devices()[0]}


def test_gemma_adapter_generate_buckets_the_static_output_length(monkeypatch):
    """max_out_length is a static shape in the library's prefill / decode jits:
    distinct max_tokens must share a bucket (one compile), while the exact
    budget still goes in as the dynamic max_new_tokens."""
    pytest.importorskip("gemma")
    from gemma import gm

    from evalrx.models.jax.gemma import OUT_BUCKETS, GemmaJaxAdapter, bucket_length
    from evalrx.specs import get_spec

    built, sampled = [], []

    class FakeSampler:
        def __init__(self, **kw):
            built.append(kw)

        def sample(self, text, *, max_new_tokens, **kw):
            sampled.append(max_new_tokens)
            predicted = np.zeros((1, built[-1]["max_out_length"]), dtype=np.int32)
            predicted[0, :2] = [5, 6]
            return type("Out", (), {"state": type("St", (), {"predicted_tokens": predicted})()})()

    class FakeTok:
        special_tokens = type("S", (), {"EOS": 1, "END_OF_TURN": 106, "BEGIN_OF_TOOL_RESPONSE": 50})

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    monkeypatch.setattr(gm.text, "Gemma4Sampler", FakeSampler)
    ad = GemmaJaxAdapter(get_spec("gemma-4-e2b-it"), RuntimeConfig())
    ad._params, ad._model, ad._tok = {"loaded": True}, object(), FakeTok()
    enc = Encoding(ids=[2, 7, 8], tokens=["<bos>", "a", "b"], text="a b")
    for n in (5, 24, 31, 64, 200):
        assert ad.generate(enc, SamplingParams(max_new_tokens=n)).ids == [5, 6]
    assert sampled == [5, 24, 31, 64, 200]
    assert [b["max_out_length"] for b in built] == [32, 32, 32, 64, 256]
    assert all(b["max_out_length"] in OUT_BUCKETS for b in built)
    assert all(b["cache_length"] == bucket_length(b["pad_length"] + b["max_out_length"] + 1, ad.buckets)
               for b in built)


def test_gemma_adapter_model_tokens_restore_internal_placeholders():
    from evalrx.models.jax.gemma import (
        AUDIO_PLACEHOLDER_ID,
        AUDIO_SOFT_PLACEHOLDER,
        IMAGE_PLACEHOLDER_ID,
        IMAGE_SOFT_PLACEHOLDER,
        GemmaJaxAdapter,
    )
    from evalrx.specs import get_spec

    ad = GemmaJaxAdapter(get_spec("gemma-4-e2b-it"), RuntimeConfig())     # lazy: no jax / gemma import
    assert ad.modalities == frozenset({"text", "image", "audio"}) and ad.text_only is False
    enc = Encoding(ids=[2, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, 7, AUDIO_PLACEHOLDER_ID, 9],
                   tokens=["<bos>", "<|image|>", "<|image|>", "x", "<|audio|>", "y"],
                   image_token_mask=[False, True, True, False, False, False],
                   audio_token_mask=[False, False, False, False, True, False])
    row = ad._model_tokens(enc, 8)
    assert row.shape == (1, 8)
    assert row[0].tolist() == [2, IMAGE_SOFT_PLACEHOLDER, IMAGE_SOFT_PLACEHOLDER, 7, AUDIO_SOFT_PLACEHOLDER, 9, 0, 0]
    text_only = GemmaJaxAdapter(get_spec("gemma-4-e2b-it"), RuntimeConfig(engine_kwargs={"text_only": True}))
    assert text_only.modalities == frozenset({"text"})
    with pytest.raises(ValueError, match="vision tower"):
        text_only._check_modalities(["img"], [])


# ----------------------------------------------------------------------
# real weights (opt-in): Gemma 4 E2B through the gemma library
# ----------------------------------------------------------------------
@pytest.mark.gpu
def test_gemma4_e2b_real_forward_shapes():
    pytest.importorskip("gemma")
    m = compose("gemma-4-e2b-it", "jax_local", RuntimeConfig(max_new_tokens=4, apply_chat_template=True,
                                                             engine_kwargs={"text_only": True}))
    trace = m.forward("What is the capital of France?", capture={Capability.HIDDEN_STATES, Capability.LOGITS})
    assert len(trace.hidden_states) == 36 and trace.logits.shape[-1] == 262144
    assert m.generate("What is the capital of France?").strip()


@pytest.mark.gpu
def test_gemma4_e2b_real_image_and_audio_inputs():
    """Image and audio rows through the towers: the soft-token reservation equals
    the encoder's pooled grid, masks and TokenTypeMap land on the Trace, and
    generation answers."""
    pytest.importorskip("gemma")
    from PIL import Image

    m = compose("gemma-4-e2b-it", "jax_local", RuntimeConfig(max_new_tokens=8, apply_chat_template=True))
    img = Image.new("RGB", (850, 600), (200, 30, 30))
    trace = m.forward(Inputs(prompt="What colour is this? Answer with one word.", image=img),
                      capture={Capability.HIDDEN_STATES, Capability.LOGITS})
    ttm = trace.token_type_map
    assert ttm.grids == [(1, 14, 19)] and len(ttm.image_pos) == 14 * 19       # 672 x 912 px, 48 px per token
    assert trace.extras["image_spatial_shape"] == (14, 19)
    assert int(trace.extras["image_token_mask"].sum()) == 14 * 19 and len(trace.hidden_states) == 36
    assert "red" in m.generate(Inputs(prompt="What colour is this? Answer with one word.", image=img)).lower()
    wav = np.zeros(16000 * 2, dtype=np.float32)                               # 2 s of silence
    trace = m.forward(Inputs(prompt="Is there speech in this clip? Answer yes or no.", audio=wav),
                      capture={Capability.LOGITS})
    assert int(trace.extras["audio_token_mask"].sum()) == 49 and trace.token_type_map is None
    assert m.generate(Inputs(prompt="Is there speech in this clip? Answer yes or no.", audio=wav)).strip()
