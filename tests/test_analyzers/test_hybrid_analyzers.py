"""The write-side analyzers on HYBRID stacks: softmax + gated-delta-rule linear attention.

Qwen3.5 (and Qwen3-Next) interleave softmax attention with GatedDeltaNet
layers that have no attention matrix. The attention tap alone would leave those
layers unedited, so ``_internals/linear_mix`` handles them:

* media edits re-run a linear layer on the sequence WITHOUT the media tokens
  and blend its output at the text positions (``w = 0`` = knockout);
* relevance materialises the delta rule's token-mixing matrix ``A`` (the rule
  is linear in its values) with the module's own kernel.

Checked here on a tiny random Qwen3-Next (any transformers with it) and, when
transformers 5 is installed, a tiny Qwen3.5 VLM — the invariants first, then
every analyzer's result contract on the hybrid stack.
"""

from __future__ import annotations

import json
import math
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("PIL")
if not hasattr(transformers, "Qwen3NextForCausalLM"):  # pragma: no cover - old transformers
    pytest.skip("transformers has no Qwen3-Next", allow_module_level=True)

from evalrx.analyzers._internals.access import (  # noqa: E402
    DecoderStack,
    check_editable_attention,
    decoder_stack,
)
from evalrx.analyzers._internals.linear_mix import (  # noqa: E402
    _torch_kernel,
    check_linear_fidelity,
    delta_rule_relevance,
    linear_media_edit,
    mixing_matrix,
)
from evalrx.core.capability import Capability  # noqa: E402
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label  # noqa: E402
from evalrx.core.result import Result  # noqa: E402
from tests.tiny_whitebox import HYBRID_FAMILIES, TinyVLM, image  # noqa: E402

MEDIA = list(range(3, 19))  # the 16 image-token positions TinyVLM lays out


@pytest.fixture(scope="module", params=HYBRID_FAMILIES)
def hybrid(request):
    return TinyVLM(request.param)


def _batch(n: int = 4) -> CaseBatch:
    return CaseBatch([
        FailureCase(inputs=Inputs(prompt=f"what is in picture {i}?", image=image(i)),
                    expected="cat", label=Label.FAIL if i % 2 else Label.PASS)
        for i in range(n)
    ])


def _logits(model, inputs) -> "torch.Tensor":
    return model.forward(inputs, capture={Capability.LOGITS}).logits


# ── stack discovery ───────────────────────────────────────────────────────────

def test_stack_sees_both_layer_kinds(hybrid):
    stack = decoder_stack(hybrid._loaded[0])
    assert stack.kinds.count("softmax") == 2 and stack.kinds.count("linear") == 4
    assert stack.editable and not stack.fully_attentional
    assert all(m is not None for m in stack.mixers)


def test_a_layer_with_an_unknown_mixer_is_refused(hybrid):
    stack = decoder_stack(hybrid._loaded[0])
    broken = DecoderStack(layers=stack.layers, path=stack.path, attn=stack.attn,
                          parent=stack.parent, linear=[None] * stack.n_layers)
    with pytest.raises(ValueError, match="neither softmax attention nor"):
        check_editable_attention(hybrid, "test", broken)


# ── delta-rule mixing matrix ──────────────────────────────────────────────────

def test_mixing_matrix_reproduces_the_kernel_and_is_causal(hybrid):
    mod = next(m for m in decoder_stack(hybrid._loaded[0]).linear if m is not None)
    kernel = _torch_kernel(mod)
    torch.manual_seed(1)
    b, t, h, dk, dv = 1, 70, 4, 8, 8  # 70 > one 64-token chunk: crosses a chunk boundary
    q, k, v = torch.randn(b, t, h, dk), torch.randn(b, t, h, dk), torch.randn(b, t, h, dv)
    g, beta = -torch.rand(b, t, h), torch.rand(b, t, h)
    out, _ = kernel(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    a = mixing_matrix(kernel, q, k, g, beta, use_qk_l2norm_in_kernel=True)
    assert a.shape == (b, h, t, t)
    assert torch.allclose(torch.einsum("bhtj,bjhd->bthd", a, v), out, atol=1e-5)
    assert float(a.triu(1).abs().max()) == 0.0  # strictly causal
    # Chefer's sum rule for a value-linear mixer: Σ_tj (∇A ⊙ A) = Σ_t ∂y/∂o_t · o_t
    grad = torch.randn_like(out)
    grad_a = torch.einsum("bthd,bjhd->bhtj", grad, v)
    assert torch.allclose((grad_a * a).sum(), (grad * out).sum(), rtol=1e-4)


# ── media-absent re-run ───────────────────────────────────────────────────────

def test_linear_edit_at_weight_one_changes_nothing(hybrid):
    inputs = Inputs(prompt="what?", image=image(0))
    with torch.no_grad():
        base = _logits(hybrid, inputs)
        with linear_media_edit(hybrid, MEDIA, weight=1.0) as state:
            same = _logits(hybrid, inputs)
    assert torch.equal(base, same) and state.calls == []


def test_linear_knockout_keeps_the_prefix_and_moves_the_answer(hybrid):
    inputs = Inputs(prompt="what?", image=image(0))
    with torch.no_grad():
        base = _logits(hybrid, inputs)
        with linear_media_edit(hybrid, MEDIA, weight=0.0) as state:
            cut = _logits(hybrid, inputs)
    assert sorted(set(state.calls)) == [i for i, k in enumerate(decoder_stack(hybrid._loaded[0]).kinds)
                                        if k == "linear"]
    assert max(state.prefix_rel_diff.values()) < 1e-4  # re-run reproduces pre-media outputs
    assert torch.allclose(cut[:3], base[:3], atol=1e-6)  # nothing before the image changes
    assert not torch.allclose(cut[-1], base[-1])


def test_linear_fidelity_verdict(hybrid):
    verdict = check_linear_fidelity(hybrid, Inputs(prompt="x", image=image(1)), MEDIA)
    assert verdict["linear_faithful"] is True
    assert verdict["linear_layers_engaged"] == verdict["linear_layers_expected"] == 4


# ── analyzers on the hybrid stack ─────────────────────────────────────────────

def _contract(result, name: str, n: int) -> dict:
    assert isinstance(result, Result) and result.analyzer == name
    json.dumps(result.findings)
    rows = result.findings["per_case"]
    assert len(rows) == n
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float):
                assert math.isfinite(value), (key, value)
    return result.findings


def test_attention_knockout_on_hybrid(hybrid):
    from evalrx.analyzers.patching.attention_knockout import AttentionKnockoutAnalyzer

    f = _contract(AttentionKnockoutAnalyzer().run(hybrid, _batch()), "attention_knockout", 4)
    assert f["layer_kinds"] == {"softmax": 2, "linear": 4}
    fid = f["tap_fidelity"]
    assert fid["faithful"] and fid["tap_layers_engaged"] == 2 and fid["linear_layers_engaged"] == 4
    assert fid["n_checked"] == 4 and all("linear_faithful" in c for c in fid["cases"])  # every case
    assert any(abs(r["ko_delta_full"]) > 0 for r in f["per_case"])


def test_knockout_in_a_band_edits_only_that_band(hybrid):
    # The early third of a 6-layer [lin, lin, soft] x2 stack is two linear
    # layers: only the linear mechanism may engage, and the result must differ
    # from the full-stack knockout.
    from evalrx.analyzers._internals.edits import media_flow_edit

    inputs = Inputs(prompt="what?", image=image(0))
    text_q = [i for i in range(40) if i not in MEDIA]
    with torch.no_grad():
        with media_flow_edit(hybrid, MEDIA, queries=text_q, layers=[0, 1]):
            early = _logits(hybrid, inputs)
        with media_flow_edit(hybrid, MEDIA, queries=text_q):
            full = _logits(hybrid, inputs)
    assert not torch.allclose(early[-1], full[-1])


def test_attention_sensitivity_on_hybrid(hybrid):
    from evalrx.analyzers.patching.attention_sensitivity import AttentionSensitivityAnalyzer

    f = _contract(AttentionSensitivityAnalyzer().run(hybrid, _batch()), "attention_sensitivity", 4)
    span = math.log(2.0) - math.log(0.5)
    for row in f["per_case"]:
        assert row["rw_slope"] == pytest.approx((row["rw_delta_up"] - row["rw_delta_down"]) / span, abs=2e-4)


def test_generic_attention_on_hybrid_reaches_every_layer(hybrid):
    from evalrx.analyzers.attribution.generic_attn import GenericAttentionExplainability

    result = GenericAttentionExplainability().run(hybrid, _batch())
    f = _contract(result, "generic_attention", 4)
    assert f["tap_fidelity"]["softmax_layers"] == 2 and f["tap_fidelity"]["linear_layers"] == 4
    assert all(0.0 <= r["rel_media_share"] <= 1.0 for r in f["per_case"])
    assert all(p.grad is None for p in hybrid._loaded[0].parameters())


def test_head_attribution_on_hybrid_covers_linear_heads(hybrid):
    from evalrx.analyzers.lens.head_attribution import HeadAttributionAnalyzer

    result = HeadAttributionAnalyzer().run(hybrid, _batch())
    f = _contract(result, "head_attribution", 4)
    assert f["layer_kinds"].count("linear") == 4
    stack = decoder_stack(hybrid._loaded[0])
    expected = [m.num_v_heads if k == "linear" else m.config.num_attention_heads
                for m, k in zip(stack.mixers, stack.kinds)]
    assert f["heads_per_layer"] == expected
    assert f["linearization_max_error"] < 1e-3
    assert all(0.0 <= r["dla_linear_share"] <= 1.0 for r in f["per_case"])


@pytest.mark.parametrize("name", ["causal_trace", "attribution_patching", "gradcam", "tuned_lens",
                                  "patchscope"])
def test_architecture_agnostic_analyzers_run_on_hybrid(hybrid, name):
    from evalrx.analyzers.attribution.gradcam import GradCAMAnalyzer
    from evalrx.analyzers.lens.patchscope import PatchscopeAnalyzer
    from evalrx.analyzers.lens.tuned_lens import TunedLensAnalyzer
    from evalrx.analyzers.patching.attribution_patching import AttributionPatchingAnalyzer
    from evalrx.analyzers.patching.causal_trace import CausalTraceAnalyzer

    analyzer = {"causal_trace": CausalTraceAnalyzer(min_effect=0.0, n_points=3),
                "attribution_patching": AttributionPatchingAnalyzer(min_effect=0.0),
                "gradcam": GradCAMAnalyzer(), "tuned_lens": TunedLensAnalyzer(steps=10),
                "patchscope": PatchscopeAnalyzer(media_layers=2)}[name]
    _contract(analyzer.run(hybrid, _batch()), name, 4)


def test_attribution_patching_is_exact_in_the_small_corruption_limit_on_hybrid(hybrid):
    # The gradient runs back through the gated-delta-rule kernel and its conv;
    # with a tiny corruption the estimates must still meet the real patches.
    from evalrx.analyzers.patching.attribution_patching import AttributionPatchingAnalyzer

    for steps in (1, 5):
        f = AttributionPatchingAnalyzer(noise_scale=1e-3, min_effect=0.0, ig_steps=steps,
                                        n_verify=2).run(hybrid, _batch()).findings
        assert f["verification"]["all"]["mae"] < 0.02 and f["verification"]["all"]["pearson"] > 0.99


def test_path_patching_freezes_and_patches_the_linear_heads_too(hybrid):
    # Every head's direct path together must carry the whole media effect; a
    # linear-attention layer left unfrozen or unpatched would break the identity.
    from evalrx.analyzers.patching.path_patching import PathPatchingAnalyzer

    f = PathPatchingAnalyzer(min_effect=0.0, top_k=0).run(hybrid, _batch()).findings
    stack = decoder_stack(hybrid._loaded[0])
    assert f["layer_kinds"] == stack.kinds and "linear" in f["layer_kinds"]
    for row in f["per_case"]:
        assert row["pp_topk_direct_share"] == pytest.approx(1.0, abs=1e-3)
        assert 0.0 <= row["pp_linear_share"] <= 1.0


def test_attention_temperature_edits_only_the_softmax_layers(hybrid):
    from evalrx.analyzers.patching.attention_temperature import AttentionTemperatureAnalyzer

    stack = decoder_stack(hybrid._loaded[0])
    f = AttentionTemperatureAnalyzer().run(hybrid, _batch(2)).findings
    assert f["tt_edited_layers"] == stack.n_softmax < stack.n_layers
    assert f["layer_kinds"] == {"softmax": stack.n_softmax, "linear": stack.n_linear}
    early_softmax = [i for i in range(round(stack.n_layers / 3)) if stack.kinds[i] == "softmax"]
    if not early_softmax:  # a band of linear layers only has no temperature to set
        with pytest.raises(ValueError, match="no softmax attention layer"):
            AttentionTemperatureAnalyzer(band="early").run(hybrid, _batch(1))


def test_embed_lens_runs_on_hybrid(hybrid):
    from evalrx.analyzers.lens.embed_lens import EmbedLensAnalyzer

    _contract(EmbedLensAnalyzer().run(hybrid, _batch()), "embed_lens", 4)


def test_embed_lens_reads_a_placeholder_token_as_itself():
    # The text-only Qwen3-Next fixture feeds an "image" as 16 copies of the
    # placeholder token: each media token IS that token's embedding, so its
    # nearest word is itself at cosine 1 — and every case shows the same
    # "image", which counts once.
    from evalrx.analyzers.lens.embed_lens import EmbedLensAnalyzer
    from tests.tiny_whitebox import IMAGE_TOKEN

    result = EmbedLensAnalyzer().run(TinyVLM("qwen3_next"), _batch())
    f = result.findings
    assert f["n_cases_analyzed"] == 4 and f["n_distinct_media"] == 1 and f["recurring_inferred"] is False
    for row in f["per_case"]:
        assert row["el_mean_max_sim"] == pytest.approx(1.0, abs=1e-4)
    assert all(set(t["top1"]) == {IMAGE_TOKEN} for t in result.artifacts["tokens"].values())


# The kernel tap must preserve the native computation and restore both API
# styles, including when a forward fails. Backward runs after the tap exits.
def test_delta_rule_tap_preserves_logits_gradients_and_layer_identity(hybrid):
    from evalrx.core.model import CaptureSpec

    module = hybrid._loaded[0]
    linear = {i: m for i, m in enumerate(decoder_stack(module).linear) if m is not None}
    kernels = {i: _torch_kernel(m) for i, m in linear.items()}
    attributes = {i: dict(m.__dict__) for i, m in linear.items()}
    inputs = Inputs(prompt="what?", image=image(0))
    spec = CaptureSpec(with_grad=True)
    weight = module.get_input_embeddings().weight
    base = hybrid.forward(inputs, capture={Capability.LOGITS}, spec=spec).logits
    base_grad, = torch.autograd.grad(base[-1, 0], weight)
    other = TinyVLM(hybrid.family)
    abar = {}
    with delta_rule_relevance(hybrid, lambda layer, value: abar.update({layer: value})) as recorded:
        other.forward(inputs, capture={Capability.LOGITS}, spec=spec)
        assert not recorded  # another model of the same family is not tapped
        logits = hybrid.forward(inputs, capture={Capability.LOGITS}, spec=spec).logits
        assert set(recorded) == set(linear)
    assert torch.equal(logits, base)
    grad, = torch.autograd.grad(logits[-1, 0], weight)
    torch.testing.assert_close(grad, base_grad)
    assert set(abar) == set(linear)
    assert all(a.shape == (logits.shape[0], logits.shape[0]) and torch.isfinite(a).all()
               for a in abar.values())
    for i, mod in linear.items():
        assert _torch_kernel(mod) is kernels[i]
        for name in ("forward", "chunk_gated_delta_rule"):
            assert (name in mod.__dict__) == (name in attributes[i])
            assert mod.__dict__.get(name) is attributes[i].get(name)


def test_delta_rule_tap_restores_after_forward_error(hybrid, monkeypatch):
    linear = [m for m in decoder_stack(hybrid._loaded[0]).linear if m is not None]
    mod = linear[0]
    forward = mod.forward
    kernel = _torch_kernel(mod)

    def failing_forward(*args, **kwargs):
        forward(*args, **kwargs)
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(mod, "forward", failing_forward)
    attributes = [dict(m.__dict__) for m in linear]
    with pytest.raises(RuntimeError, match="injected forward failure"):
        with delta_rule_relevance(hybrid, lambda *_: None):
            hybrid.forward(Inputs(prompt="what?", image=image(0)), capture={Capability.LOGITS})
    assert sys.modules[type(mod).__module__].torch_chunk_gated_delta_rule is kernel
    for m, saved in zip(linear, attributes):
        for name in ("forward", "chunk_gated_delta_rule"):
            assert (name in m.__dict__) == (name in saved)
            assert m.__dict__.get(name) is saved.get(name)


def test_module_kernel_dispatch_does_not_tap_another_thread(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from evalrx.analyzers._internals.linear_mix import _module_kernel_forward

    owner = sys.modules[__name__]

    def original(value):
        return value * 2

    monkeypatch.setattr(owner, "torch_chunk_gated_delta_rule", original, raising=False)
    calls = []

    def wrap_kernel(kernel):
        def tapped(value):
            calls.append(value)
            return kernel(value)
        return tapped

    with ThreadPoolExecutor(max_workers=1) as pool:
        class Mixer:
            def forward(self, value):
                background = pool.submit(owner.torch_chunk_gated_delta_rule, value).result(timeout=5)
                return owner.torch_chunk_gated_delta_rule(value) + background

        forward = _module_kernel_forward(Mixer(), wrap_kernel)
        assert forward(3) == 12
    assert calls == [3]
    assert owner.torch_chunk_gated_delta_rule is original
