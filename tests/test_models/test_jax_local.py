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
    assert m.modalities == frozenset({"text"})        # phase 1: text
    assert "GemmaJaxAdapter" in repr(m)


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
# real weights (opt-in): Gemma 4 E2B through the gemma library
# ----------------------------------------------------------------------
@pytest.mark.gpu
def test_gemma4_e2b_real_forward_shapes():
    pytest.importorskip("gemma")
    m = compose("gemma-4-e2b-it", "jax_local", RuntimeConfig(max_new_tokens=4, apply_chat_template=True))
    trace = m.forward("What is the capital of France?", capture={Capability.HIDDEN_STATES, Capability.LOGITS})
    assert len(trace.hidden_states) == 36 and trace.logits.shape[-1] == 262144
    assert m.generate("What is the capital of France?").strip()
