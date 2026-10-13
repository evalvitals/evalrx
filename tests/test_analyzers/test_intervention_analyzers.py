"""Write-side white-box analyzers on a tiny real Qwen-VL (CPU, random init).

Two layers of checks:

* invariants of the shared plumbing (``analyzers/_internals``): the edit-free
  attention tap is exact, keeps the causal mask, never retargets the vision
  tower, and refuses when it cannot engage; edits touch exactly the cells they
  name; residual hooks capture/replace what they say;
* the analyzers' result contract: a ``Result`` with JSON-serialisable findings,
  one ``per_case`` row per analysable case, and loud refusal (never a silent
  zero) on architectures or batches they cannot measure.
"""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("PIL")

from evalrx.analyzers._internals import access, attn_tap  # noqa: E402
from evalrx.analyzers._internals.attn_tap import (  # noqa: E402
    attention_tap,
    check_fidelity,
    knockout,
    require_fidelity,
    reweight,
)
from evalrx.analyzers._internals.residual import residual_hooks  # noqa: E402
from evalrx.analyzers._internals.targets import build_margin, gold_text  # noqa: E402
from evalrx.core.capability import Capability  # noqa: E402
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label  # noqa: E402
from evalrx.core.result import Result  # noqa: E402
from evalrx.core.spec import AttnSemantics  # noqa: E402
from tests.tiny_whitebox import IMAGE_TOKEN, VEND, VSTART, TinyVLM, image  # noqa: E402

FAMILIES = ("qwen2_5_vl", "qwen3_vl")


@pytest.fixture(scope="module", params=FAMILIES)
def vlm(request):
    return TinyVLM(request.param)


def _batch(n: int = 4, with_gold: bool = True) -> CaseBatch:
    return CaseBatch([
        FailureCase(
            inputs=Inputs(prompt=f"what is in picture {i}?", image=image(i)),
            expected="cat" if with_gold else None,
            label=Label.FAIL if i % 2 else Label.PASS,
        )
        for i in range(n)
    ])


def _raw_inputs(seq_tail: list[int]) -> dict:
    """Direct module inputs (bypassing the Model) for mask-level checks."""
    from tests.tiny_whitebox import GRID, _pixel_values

    ids = [1, 2, VSTART] + [IMAGE_TOKEN] * 16 + [VEND] + seq_tail
    return dict(input_ids=torch.tensor([ids]), pixel_values=_pixel_values(image(0)),
                image_grid_thw=torch.tensor([[1, GRID, GRID]]))


# ── attention tap ──────────────────────────────────────────────────────────────

def test_edit_free_tap_is_exact_and_engages_every_layer(vlm):
    verdict = check_fidelity(vlm, Inputs(prompt="describe", image=image(0)))
    assert verdict["tap_faithful"] is True
    assert verdict["tap_layers_engaged"] == verdict["tap_layers_expected"] == 6
    assert verdict["tap_max_logit_diff"] == 0.0  # same ops as eager, bit for bit


def test_tap_keeps_the_causal_mask(vlm):
    # transformers builds NO mask for an unknown attention name; the tap must
    # register one, or earlier positions would see a changed future token.
    module = vlm._loaded[0]
    a, b = _raw_inputs([7, 8, 9, 10]), _raw_inputs([7, 8, 9, 11])
    with torch.no_grad(), attention_tap(vlm):
        la, lb = module(**a).logits, module(**b).logits
    assert torch.equal(la[0, :-1], lb[0, :-1])
    assert not torch.equal(la[0, -1], lb[0, -1])


def test_tap_retargets_only_the_decoder_and_restores_on_error(vlm):
    cfg = vlm._loaded[0].config
    vision_before, text_before = cfg.vision_config._attn_implementation, cfg.text_config._attn_implementation
    with pytest.raises(RuntimeError, match="boom"):
        with attention_tap(vlm):
            assert cfg.text_config._attn_implementation == attn_tap.TAP_NAME
            assert cfg.vision_config._attn_implementation == vision_before
            raise RuntimeError("boom")
    assert cfg.text_config._attn_implementation == text_before
    assert attn_tap.active_state() is None


def test_knockout_touches_only_downstream_rows(vlm):
    module = vlm._loaded[0]
    enc = _raw_inputs([7, 8, 9, 10])
    image_pos = list(range(3, 19))
    text_q = [i for i in range(enc["input_ids"].shape[1]) if i not in image_pos]
    with torch.no_grad():
        base = module(**enc).logits
        with attention_tap(vlm, [knockout(image_pos, queries=text_q)]):
            cut = module(**enc).logits
    assert torch.equal(cut[0, :3], base[0, :3])          # before the image: nothing to cut
    assert not torch.allclose(cut[0, -1], base[0, -1])   # the answer position lost the image


def test_reweight_equivalences(vlm):
    module = vlm._loaded[0]
    enc = _raw_inputs([7, 8, 9, 10])
    keys = list(range(3, 19))
    with torch.no_grad():
        base = module(**enc).logits
        with attention_tap(vlm, [reweight(keys, 1.0)]):
            same = module(**enc).logits
        with attention_tap(vlm, [reweight(keys, 0.0)]):
            zero = module(**enc).logits
        with attention_tap(vlm, [knockout(keys)]):
            cut = module(**enc).logits
    assert torch.equal(same, base)
    assert torch.equal(zero, cut)  # scale by 0 IS a knockout


def test_scale_raw_does_not_renormalise(vlm):
    rows = {}

    def grab(layer, probs):
        rows[layer] = probs[0, :, -1].sum(-1)  # per-head row sums at the last query

    with torch.no_grad(), attention_tap(vlm, [reweight(range(3, 19), 3.0, renormalize=False)],
                                        on_probs=grab):
        vlm.forward(Inputs(prompt="hi", image=image(0)), capture={Capability.LOGITS})
    assert set(rows) == set(range(6))
    assert all(bool((r > 1.0 + 1e-3).all()) for r in rows.values())


def test_require_fidelity_refuses_a_bypassed_layer(vlm, monkeypatch):
    real = access.decoder_stack

    def with_phantom(module):
        stack = real(module)
        stack.attn = stack.attn + [torch.nn.Identity()]  # a layer the tap never sees
        stack.layers = list(stack.layers) + [torch.nn.Identity()]
        return stack

    monkeypatch.setattr(attn_tap, "decoder_stack", with_phantom)
    module = vlm._loaded[0]
    saved = module.__dict__.pop("_evalrx_tap_fidelity", None)
    try:
        with pytest.raises(RuntimeError, match="does not faithfully reproduce"):
            require_fidelity(vlm, Inputs(prompt="x", image=image(0)))
    finally:
        module.__dict__.pop("_evalrx_tap_fidelity", None)
        if saved is not None:
            module.__dict__["_evalrx_tap_fidelity"] = saved


# ── residual hooks ─────────────────────────────────────────────────────────────

def test_residual_replace_with_own_capture_is_a_noop(vlm):
    inputs = Inputs(prompt="what?", image=image(1))
    pos = list(range(3, 19))
    with torch.no_grad():
        with residual_hooks(vlm, capture={2: pos}) as s:
            base = vlm.forward(inputs, capture={Capability.LOGITS}).logits
        with residual_hooks(vlm, replace={2: (pos, s.captured[2])}):
            again = vlm.forward(inputs, capture={Capability.LOGITS}).logits
    assert s.captured[2].shape == (16, 32)
    assert torch.equal(base, again)


def test_residual_replace_changes_only_downstream(vlm):
    inputs = Inputs(prompt="what?", image=image(1))
    with torch.no_grad():
        base = vlm.forward(inputs, capture={Capability.LOGITS}).logits
        with residual_hooks(vlm, replace={3: ([5], torch.zeros(1, 32))}):
            patched = vlm.forward(inputs, capture={Capability.LOGITS}).logits
    assert torch.equal(patched[:5], base[:5])
    assert not torch.allclose(patched[-1], base[-1])


# ── targets ────────────────────────────────────────────────────────────────────

def test_gold_text_shapes():
    assert gold_text("cat") == "cat"
    assert gold_text(["cat", "kitty"]) == "cat"
    assert gold_text({"all_of": ["yes"], "none_of": ["no"]}) == "yes"
    assert gold_text(None) is None and gold_text("  ") is None


def test_prediction_margin_is_top1_minus_top2():
    logits = torch.tensor([[0.0, 3.0, 1.0, 2.5]])
    margin, why = build_margin("prediction", logits)
    assert why == "" and (margin.positive, margin.negative) == (1, 3)
    assert math.isclose(margin.value(logits), 0.5)


def test_gold_margin_needs_gold_and_tokenizer(vlm):
    logits = torch.randn(1, 128)
    no_gold = FailureCase(inputs=Inputs(prompt="x"))
    assert build_margin("gold", logits, model=vlm, case=no_gold) == (None, "no_gold")
    margin, why = build_margin("gold", logits, model=vlm, case=FailureCase(inputs=Inputs(prompt="x"),
                                                                            expected="cat"))
    assert why == "" and margin.positive == ord("c") % 90 + 1 and margin.negative != margin.positive


# ── analyzers ─────────────────────────────────────────────────────────────────

def _assert_result_contract(result, name: str, n_rows: int) -> dict:
    assert isinstance(result, Result) and result.analyzer == name
    json.dumps(result.findings)
    assert isinstance(result.summary(), str)
    rows = result.findings["per_case"]
    assert len(rows) == n_rows
    for row in rows:
        assert isinstance(row["sample_id"], str)
        for key, value in row.items():
            if isinstance(value, float):
                assert math.isfinite(value), (key, value)
    return result.findings


def test_attention_knockout_contract(vlm):
    from evalrx.analyzers.patching.attention_knockout import AttentionKnockoutAnalyzer

    f = _assert_result_contract(AttentionKnockoutAnalyzer().run(vlm, _batch()), "attention_knockout", 4)
    assert f["tap_fidelity"]["tap_faithful"] is True
    assert f["media_slot"] == "image" and f["n_layers"] == 6
    assert f["layer_bands"] == {"early": [0, 2], "mid": [2, 4], "late": [4, 6]}
    for row in f["per_case"]:
        assert row["ko_margin_clean"] >= 0  # prediction target: top1 - top2
        assert isinstance(row["ko_flips_top1"], bool)
        assert {"ko_delta_full", "ko_delta_early", "ko_delta_mid", "ko_delta_late"} <= row.keys()
    assert any(abs(r["ko_delta_full"]) > 0 for r in f["per_case"])


def test_attention_knockout_last_query_and_gold_target(vlm):
    from evalrx.analyzers.patching.attention_knockout import AttentionKnockoutAnalyzer

    batch = CaseBatch(list(_batch(2)) + list(_batch(2, with_gold=False)))
    f = AttentionKnockoutAnalyzer(target="gold", queries="last").run(vlm, batch).findings
    assert f["n_cases_analyzed"] == 2
    assert f["skip_reasons"] == {"no_gold": 2}


def test_attention_knockout_refuses_text_only_batch(vlm):
    from evalrx.analyzers.patching.attention_knockout import AttentionKnockoutAnalyzer

    with pytest.raises(ValueError, match="no_media_tokens"):
        AttentionKnockoutAnalyzer().run(vlm, CaseBatch([FailureCase(inputs=Inputs(prompt="hi"))]))


def test_attention_knockout_refuses_hybrid_attention():
    from evalrx.analyzers.patching.attention_knockout import AttentionKnockoutAnalyzer

    hybrid = TinyVLM("qwen2_5_vl", attn_semantics=AttnSemantics.HYBRID_SPARSE)
    with pytest.raises(ValueError, match="hybrid_sparse"):
        AttentionKnockoutAnalyzer().run(hybrid, _batch(1))


def test_attention_summary_reports_media_mass(vlm):
    from evalrx.analyzers.attention.summary import AttentionAnalyzer

    f = _assert_result_contract(AttentionAnalyzer().run(vlm, _batch(3)), "attention", 3)
    assert all(0.0 <= r["media_mass_mid"] <= 1.0 for r in f["per_case"])


# ── gradient path ──────────────────────────────────────────────────────────────

def test_with_grad_forward_keeps_a_graph_and_writes_no_parameter_grads(vlm):
    from evalrx.core.model import CaptureSpec

    module = vlm._loaded[0]
    with residual_hooks(vlm, capture={3: None}, keep_grad=True) as s:
        trace = vlm.forward(Inputs(prompt="q", image=image(0)), capture={Capability.LOGITS},
                            spec=CaptureSpec(with_grad=True, to_cpu=False))
    (grad,) = torch.autograd.grad(trace.logits[-1].sum(), s.captured[3])
    assert grad.shape == s.captured[3].shape and bool(grad.abs().sum() > 0)
    assert all(p.grad is None for p in module.parameters())
    plain = vlm.forward(Inputs(prompt="q", image=image(0)), capture={Capability.LOGITS})
    assert not plain.logits.requires_grad


def test_hf_local_honours_with_grad():
    # The backend contract the gradient analyzers rely on (no weights loaded).
    import inspect

    from evalrx.models.backends.hf.model import HFLocalBackend, HFLocalModel

    assert Capability.GRADIENTS in HFLocalBackend.capabilities
    assert "set_grad_enabled(with_grad)" in inspect.getsource(HFLocalModel.forward)


def test_gradcam_contract(vlm):
    from evalrx.analyzers.attribution.gradcam import GradCAMAnalyzer

    result = GradCAMAnalyzer().run(vlm, _batch())
    f = _assert_result_contract(result, "gradcam", 4)
    assert f["layer_used"] == 2 and f["map_shape"] == [4, 4]
    for row in f["per_case"]:
        assert 0.0 <= row["gc_media_grad_share"] <= 1.0
        assert isinstance(row["gc_empty"], bool)
        if not row["gc_empty"]:
            assert 0.0 <= row["gc_entropy"] <= 1.0 and 0.0 < row["gc_top5_share"] <= 1.0
    assert set(result.findings["per_case"][0]) >= {"gc_margin", "gc_media_grad_share"}
    assert {"spatial_map", "per_case_maps", "fail_mean_map", "pass_mean_map",
            "diff_map_fail_minus_pass"} <= set(result.artifacts)
    assert result.findings["spatial_map_case_id"] == f["per_case"][0]["sample_id"]


def test_gradcam_overlays_land_on_the_case_image(vlm, tmp_path):
    pytest.importorskip("matplotlib")
    from evalrx.analyzers.attribution.gradcam import GradCAMAnalyzer

    paths = GradCAMAnalyzer().run(vlm, _batch()).image_overlays(tmp_path, "c0_gradcam")
    assert {p.name for p in paths} >= {"c0_gradcam_spatial_map_overlay.png",
                                       "c0_gradcam_diff_map_fail_minus_pass_overlay.png"}


def test_generic_attention_contract(vlm):
    from evalrx.analyzers.attribution.generic_attn import GenericAttentionExplainability

    result = GenericAttentionExplainability().run(vlm, _batch())
    f = _assert_result_contract(result, "generic_attention", 4)
    assert f["tap_fidelity"]["tap_faithful"] is True
    assert len(f["top_relevant_tokens"]) == 10
    for row in f["per_case"]:
        assert 0.0 <= row["rel_media_share"] <= 1.0
        assert 0.0 <= row["rel_input_entropy"] <= 1.0
    assert "spatial_map" in result.artifacts


def test_generic_attention_runs_text_only(vlm):
    from evalrx.analyzers.attribution.generic_attn import GenericAttentionExplainability

    f = GenericAttentionExplainability().run(vlm, CaseBatch([FailureCase(inputs=Inputs(prompt="hello there"))])).findings
    row = f["per_case"][0]
    assert "rel_media_share" not in row and "rel_input_entropy" in row
    assert f["mean_media_share"] is None


# ── causal tracing ─────────────────────────────────────────────────────────────

def test_full_restore_after_deepstack_returns_the_clean_logits():
    # On Qwen3-VL the visual features are re-injected after layers 0..k-1; a
    # complete residual restore at layer >= k must still give back the clean run.
    from evalrx.analyzers.patching.causal_trace import _noised, deepstack_mergers

    vlm = TinyVLM("qwen3_vl")
    mergers = deepstack_mergers(vlm._loaded[0])
    assert len(mergers) == 2
    inputs = Inputs(prompt="what?", image=image(2))
    with torch.no_grad():
        with residual_hooks(vlm, capture={2: None}) as clean:
            base = vlm.forward(inputs, capture={Capability.LOGITS}).logits[-1]
        gen = torch.Generator().manual_seed(0)
        store: dict = {}
        with _noised(mergers, store, 3.0, gen):
            corrupt = vlm.forward(inputs, capture={Capability.LOGITS}).logits[-1]
            with residual_hooks(vlm, replace={2: (list(range(clean.captured[2].shape[0])), clean.captured[2])}):
                restored = vlm.forward(inputs, capture={Capability.LOGITS}).logits[-1]
    assert len(store) == 2
    assert not torch.allclose(corrupt, base)
    assert torch.allclose(restored, base, atol=1e-6)


@pytest.mark.parametrize("corruption", ["noise", "gray"])
def test_causal_trace_contract(vlm, corruption):
    from evalrx.analyzers.patching.causal_trace import CausalTraceAnalyzer

    f = _assert_result_contract(
        CausalTraceAnalyzer(corruption=corruption, min_effect=0.0, n_points=3).run(vlm, _batch()),
        "causal_trace", 4,
    )
    assert f["layers_probed"] == [0, 2, 5] and f["n_traced"] == 4
    assert f["deepstack_layers"] == (2 if vlm.spec.key.endswith("qwen3_vl") else 0)
    assert set(f["mean_recovery_curves"]) == {"fail_media", "fail_last", "pass_media", "pass_last"}
    for row in f["per_case"]:
        assert row["ct_effect_traceable"] is True
        assert 0.0 <= row["ct_media_handoff_frac"] <= 1.0
        assert 0.0 <= row["ct_last_rise_frac"] <= 1.0
        if f["deepstack_layers"]:  # restores before the last re-injection are partial by construction
            assert row["ct_media_handoff_frac"] >= 2 / 5


def test_causal_trace_noise_restore_at_layer0_is_exact_without_deepstack():
    from evalrx.analyzers.patching.causal_trace import CausalTraceAnalyzer

    vlm = TinyVLM("qwen2_5_vl")
    result = CausalTraceAnalyzer(min_effect=0.0, n_points=3).run(vlm, _batch(2))
    for curve in result.artifacts["recovery_curves"].values():
        assert curve["media"][0] == pytest.approx(1.0, abs=1e-3)
        assert curve["last"][0] == pytest.approx(0.0, abs=1e-3)


def test_causal_trace_small_effects_keep_the_row_but_skip_the_trace(vlm):
    from evalrx.analyzers.patching.causal_trace import CausalTraceAnalyzer

    f = CausalTraceAnalyzer(n_points=3).run(vlm, _batch(2)).findings  # tiny model: effects << 1 logit
    assert f["n_cases_analyzed"] == 2 and f["n_traced"] == 0
    for row in f["per_case"]:
        assert row["ct_effect_traceable"] is False
        assert "ct_media_recovery_mid" not in row  # no recovery number made up


def test_gray_inputs_keep_size_and_flatten_content():
    import numpy as np

    from evalrx.analyzers.patching.causal_trace import gray_inputs

    grayed = gray_inputs(Inputs(prompt="x", image=image(3)))
    arr = np.asarray(grayed.image)
    assert grayed.image.size == image(3).size and (arr == arr[0, 0]).all()
    assert gray_inputs(Inputs(prompt="x")) is None


# ── head attribution ──────────────────────────────────────────────────────────

def test_per_head_writes_sum_to_the_output_projection(vlm):
    from evalrx.analyzers._internals.heads import head_capture, n_heads_of, out_proj

    stack = access.decoder_stack(vlm._loaded[0])
    outputs = {}
    handles = [out_proj(a).register_forward_hook(
        lambda mod, args, out, i=i: outputs.__setitem__(i, out[0, -1].detach()))
        for i, a in enumerate(stack.attn)]
    try:
        with torch.no_grad(), head_capture(vlm) as cap:
            vlm.forward(Inputs(prompt="q", image=image(0)), capture={Capability.LOGITS})
    finally:
        for h in handles:
            h.remove()
    assert cap.final_residual is not None and len(cap.head_inputs) == 6
    for layer, attn in enumerate(stack.attn):
        proj = out_proj(attn)
        heads = n_heads_of(attn, proj)
        w = proj.weight.view(proj.weight.shape[0], heads, -1)
        x = cap.head_inputs[layer].view(heads, -1)
        per_head = torch.einsum("dhk,hk->hd", w, x)
        bias = proj.bias if proj.bias is not None else 0
        assert torch.allclose(per_head.sum(0) + bias, outputs[layer], atol=1e-5)


def test_head_attribution_contract(vlm):
    from evalrx.analyzers.lens.head_attribution import HeadAttributionAnalyzer

    result = HeadAttributionAnalyzer().run(vlm, _batch())
    f = _assert_result_contract(result, "head_attribution", 4)
    assert f["linearization_max_error"] < 1e-3  # RMSNorm, no soft-capping: frozen norm is exact
    assert f["n_layers"] == 6 and f["n_heads"] == 4 and len(f["top_heads_first_case"]) == 10
    for row in f["per_case"]:
        assert 0.0 < row["dla_top_head_share"] <= 1.0
        assert 0.0 <= row["dla_late_share"] <= 1.0 and 0.0 <= row["dla_negative_share"] <= 1.0
    assert result.artifacts["head_dla_mean_map"].shape == (6, 4)
    assert result.artifacts["head_dla_diff_map_fail_minus_pass"].shape == (6, 4)


def test_head_attribution_runs_text_only(vlm):
    from evalrx.analyzers.lens.head_attribution import HeadAttributionAnalyzer

    f = HeadAttributionAnalyzer().run(vlm, "plain text prompt").findings
    assert f["n_cases_analyzed"] == 1


# ── attention sensitivity ─────────────────────────────────────────────────────

def test_attention_sensitivity_contract(vlm):
    from evalrx.analyzers.patching.attention_sensitivity import AttentionSensitivityAnalyzer

    f = _assert_result_contract(AttentionSensitivityAnalyzer(band="mid").run(vlm, _batch()),
                                "attention_sensitivity", 4)
    assert f["tap_fidelity"]["tap_faithful"] is True and f["band"] == "mid"
    span = math.log(2.0) - math.log(0.5)
    for row in f["per_case"]:
        assert row["rw_slope"] == pytest.approx((row["rw_delta_up"] - row["rw_delta_down"]) / span, abs=2e-4)
        assert isinstance(row["rw_flips_up"], bool)


def test_attention_sensitivity_validates_params():
    from evalrx.analyzers.patching.attention_sensitivity import AttentionSensitivityAnalyzer

    with pytest.raises(ValueError, match="band"):
        AttentionSensitivityAnalyzer(band="middle")
    with pytest.raises(ValueError, match="down"):
        AttentionSensitivityAnalyzer(down=2.0, up=3.0)


# ── tuned lens ────────────────────────────────────────────────────────────────

def test_tuned_lens_fit_beats_the_identity_translator_and_leaves_weights_alone(vlm, tmp_path):
    from evalrx.analyzers.lens.tuned_lens import TunedLensAnalyzer

    module = vlm._loaded[0]
    identity = TunedLensAnalyzer(steps=0).run(vlm, _batch()).findings
    path = tmp_path / "lens.pt"
    tuned_result = TunedLensAnalyzer(steps=150, lr=1e-2, translators_path=str(path)).run(vlm, _batch())
    tuned = _assert_result_contract(tuned_result, "tuned_lens", 4)
    assert tuned["fit_kl_mean"] < identity["fit_kl_mean"]
    assert all(p.grad is None for p in module.parameters())
    assert path.exists()
    # A fresh analyzer with the same configuration reuses the cached fit.
    again = TunedLensAnalyzer(steps=150, lr=1e-2, translators_path=str(path)).run(vlm, _batch()).findings
    assert again["fit_kl_per_layer"] == tuned["fit_kl_per_layer"]


def test_tuned_lens_loads_saved_translators(tmp_path):
    from evalrx.analyzers.lens.tuned_lens import TunedLensAnalyzer

    path = tmp_path / "lens.pt"
    first = TunedLensAnalyzer(steps=30, translators_path=str(path)).run(TinyVLM("qwen2_5_vl"), _batch(2)).findings
    fresh_model = TinyVLM("qwen2_5_vl")  # no in-memory cache: must come from disk
    loaded = TunedLensAnalyzer(steps=30, translators_path=str(path)).run(fresh_model, _batch(2)).findings
    assert loaded["fit_kl_per_layer"] == first["fit_kl_per_layer"]
    strip = lambda rows: [{k: v for k, v in r.items() if k != "sample_id"} for r in rows]  # noqa: E731
    assert strip(loaded["per_case"]) == strip(first["per_case"])  # case ids are per-batch uuids
