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

import contextlib
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


def _raw_inputs(seq_tail: list[int], module=None) -> dict:
    """Direct module inputs (bypassing the Model) for mask-level checks."""
    from tests.tiny_whitebox import GRID, _accepts, _pixel_values

    ids = [1, 2, VSTART] + [IMAGE_TOKEN] * 16 + [VEND] + seq_tail
    enc = dict(input_ids=torch.tensor([ids]), pixel_values=_pixel_values(image(0)),
               image_grid_thw=torch.tensor([[1, GRID, GRID]]))
    if module is not None and _accepts(module, "mm_token_type_ids"):  # transformers 5 M-RoPE
        enc["mm_token_type_ids"] = (enc["input_ids"] == IMAGE_TOKEN).long()
    return enc


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
    a, b = _raw_inputs([7, 8, 9, 10], module), _raw_inputs([7, 8, 9, 11], module)
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
    enc = _raw_inputs([7, 8, 9, 10], module)
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
    enc = _raw_inputs([7, 8, 9, 10], module)
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
    with pytest.raises(attn_tap.FidelityError, match="does not faithfully reproduce"):
        require_fidelity(vlm, Inputs(prompt="x", image=image(0)))


def test_fidelity_is_measured_on_every_call_not_cached(vlm):
    inputs = Inputs(prompt="describe", image=image(0))
    before = vlm.n_forwards
    check_fidelity(vlm, inputs)
    check_fidelity(vlm, inputs)
    assert vlm.n_forwards - before == 4  # native + tapped, twice
    base = vlm.forward(inputs, capture={Capability.LOGITS}).require(Capability.LOGITS)
    before = vlm.n_forwards
    verdict = check_fidelity(vlm, inputs, base=base)
    assert vlm.n_forwards - before == 1  # the caller's native logits are reused
    assert verdict["tap_faithful"] and verdict["tap_argmax_same"]
    assert verdict["tap_rel_diff"] == pytest.approx(verdict["tap_max_logit_diff"] / verdict["tap_scale"], abs=1e-5)


def test_knockout_checks_every_analysed_case(vlm):
    from evalrx.analyzers.patching.attention_knockout import AttentionKnockoutAnalyzer

    batch = CaseBatch(list(_batch(3)) + list(_batch(1, with_gold=False)))
    fid = AttentionKnockoutAnalyzer(target="gold").run(vlm, batch).findings["tap_fidelity"]
    assert fid["checked"] == "every analysed case" and fid["n_checked"] == 3  # the no-gold case is skipped
    assert [c["case"] for c in fid["cases"]] == [c.id for c in list(batch)[:3]]
    assert fid["faithful"] and fid["tap_worst_case"] in {c.id for c in batch}


def test_a_later_unfaithful_case_refuses_the_whole_batch(vlm, monkeypatch):
    from evalrx.analyzers._internals import edits
    from evalrx.analyzers.patching.attention_knockout import AttentionKnockoutAnalyzer

    real = edits.check_fidelity
    calls = []

    def flaky(*args, **kwargs):
        verdict = real(*args, **kwargs)
        calls.append(1)
        if len(calls) == 3:  # the third case's input is off by more than rtol
            verdict = {**verdict, "tap_faithful": False, "tap_rel_diff": 0.02}
        return verdict

    monkeypatch.setattr(edits, "check_fidelity", flaky)
    batch = _batch(4)
    with pytest.raises(attn_tap.FidelityError) as info:
        AttentionKnockoutAnalyzer().run(vlm, batch)
    verdict = info.value.verdict
    assert verdict["n_checked"] == 3 and verdict["failed_case"] == list(batch)[2].id
    assert verdict["faithful"] is False and verdict["tap_max_rel_diff"] == 0.02


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


class _Vocab:
    """A BPE-like vocabulary: an answer, its spaced and its lower-case form are three ids."""

    pieces = ["<pad>", "Yes", " Yes", "yes", " yes", "No", " No", "no", " no", " a", "A", " A", "?"]

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            hits = [j for j, p in enumerate(self.pieces) if p and text.startswith(p, i)]
            best = max(hits, key=lambda j: len(self.pieces[j])) if hits else len(self.pieces) - 1
            ids.append(best)
            i += len(self.pieces[best]) if hits else 1
        return {"input_ids": ids}

    def decode(self, ids):
        return self.pieces[int(ids[0])]


def test_margins_compare_an_answer_with_other_answers_not_with_its_own_spellings(monkeypatch):
    # Qwen3.5-0.8B on POPE: " Yes" 0.80, "Yes" 0.20 on a correct case. The old
    # comparison token was " Yes" -- a gold "Yes" margin was negative there.
    from evalrx.analyzers._internals.targets import answer_variants

    monkeypatch.setattr(access, "tokenizer_of", lambda model: _Vocab())
    tok = _Vocab()
    assert answer_variants(tok, "Yes") == {1, 2, 3, 4}
    assert answer_variants(tok, "A") == {10, 11}            # never the article " a"
    logits = torch.zeros(1, len(_Vocab.pieces))
    logits[0, [2, 1, 5]] = torch.tensor([3.0, 2.0, 1.0])     # " Yes" > "Yes" > "No"
    gold, _ = build_margin("gold", logits, model=object(), case=FailureCase(inputs=Inputs(prompt="q"),
                                                                             expected="Yes"))
    assert (gold.positive, gold.negative, gold.variants) == (2, 5, (1, 2, 3, 4))
    assert math.isclose(gold.value(logits), 2.0)
    pred, _ = build_margin("prediction", logits, model=object())
    assert (pred.positive, pred.negative) == (2, 5)

    class _NoDecoder(_Vocab):
        decode = None

    assert answer_variants(_NoDecoder(), "Yes") == {1}     # nothing confirms the other forms


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


def test_attention_knockout_refuses_latent_attention():
    from evalrx.analyzers.patching.attention_knockout import AttentionKnockoutAnalyzer

    latent = TinyVLM("qwen2_5_vl", attn_semantics=AttnSemantics.MLA_LATENT)
    with pytest.raises(ValueError, match="mla_latent"):
        AttentionKnockoutAnalyzer().run(latent, _batch(1))


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


# ── attribution patching ──────────────────────────────────────────────────────

def test_residual_keep_grad_survives_frozen_weights():
    # A model loaded with requires_grad_(False) has no graph upstream of the
    # residual stream; keep_grad makes the captured state a leaf so the gradient
    # analyzers still work — and without grad mode it changes nothing.
    from evalrx.core.model import CaptureSpec

    frozen = TinyVLM("qwen2_5_vl")
    frozen._loaded[0].requires_grad_(False)
    inputs = Inputs(prompt="q", image=image(0))
    with residual_hooks(frozen, capture={0: None, 3: None}, keep_grad=True) as s:
        trace = frozen.forward(inputs, capture={Capability.LOGITS},
                               spec=CaptureSpec(with_grad=True, to_cpu=False))
    grads = torch.autograd.grad(trace.logits[-1].sum(), [s.captured[0], s.captured[3]])
    assert all(bool(g.abs().sum() > 0) for g in grads)
    with residual_hooks(frozen, capture={0: None}, keep_grad=True) as s:
        frozen.forward(inputs, capture={Capability.LOGITS})
    assert not s.captured[0].requires_grad


@pytest.mark.parametrize("direction, ig_steps", [("denoise", 1), ("noise", 1), ("denoise", 5)])
def test_attribution_patching_matches_exact_patches_for_a_small_corruption(vlm, direction, ig_steps):
    # First order is exact in the limit: with a tiny corruption every reported
    # single-token estimate must agree with the real patch of that token.
    from evalrx.analyzers.patching.attribution_patching import AttributionPatchingAnalyzer

    f = AttributionPatchingAnalyzer(noise_scale=1e-3, min_effect=0.0, direction=direction,
                                    ig_steps=ig_steps, n_verify=2).run(vlm, _batch()).findings
    checked = f["verification"]
    assert checked["last"]["n_sites"] == 4 * 2 and checked["relay"]["n_sites"] >= 1
    assert checked["all"]["mae"] < 0.02 and checked["all"]["pearson"] > 0.99


@pytest.mark.parametrize("corruption", ["noise", "gray"])
def test_integration_path_runs_from_the_corrupted_to_the_clean_run(corruption):
    # The path moves only the decoder's media input — the layer-0 media
    # embeddings and every DeepStack re-injection — so its ends must be the
    # corrupted run and the clean run exactly, pixels untouched.
    from evalrx.analyzers.patching.attribution_patching import _recorded, path_point
    from evalrx.analyzers.patching.causal_trace import (
        _noised,
        corrupted_variants,
        deepstack_mergers,
    )

    vlm = TinyVLM("qwen3_vl")
    mergers = deepstack_mergers(vlm._loaded[0])
    inputs = Inputs(prompt="what?", image=image(1))

    def forward(variant, record=None, capture=False):
        run_inputs, plan, (store, gen) = variant
        noised = _noised(mergers, store, 3.0, gen) if store is not None else contextlib.nullcontext()
        recording = _recorded(mergers, record) if record is not None else contextlib.nullcontext()
        with torch.no_grad(), recording, noised, residual_hooks(
                vlm, capture={0: None} if capture else None, replace=plan) as s:
            logits = vlm.forward(run_inputs, capture={Capability.LOGITS}).logits[-1]
        return logits, s.captured

    clean_rec, corrupt_rec = {}, {}
    clean, states = forward((inputs, {}, (None, None)), record=clean_rec, capture=True)
    media = list(range(3, 19))
    variants, _ = corrupted_variants(inputs, corruption, slot="image", media=media,
                                     embeddings=states[0][media], deepstack=True,
                                     noise_scale=3.0, n_noise=1, seed=0)
    corrupt, corrupt_states = forward(variants[0], record=corrupt_rec, capture=True)
    store = variants[0][2][0]
    deltas = ({i: corrupt_rec[i] - clean_rec[i] for i in corrupt_rec} if corruption == "gray"
              else dict(store))
    assert len(deltas) == 2 and not torch.allclose(corrupt, clean)
    ends = {alpha: forward(path_point(alpha, inputs=inputs, media=media, clean_media=states[0][media],
                                      corrupt_media=corrupt_states[0][media], deltas=deltas))[0]
            for alpha in (0.0, 1.0)}
    assert torch.allclose(ends[0.0], corrupt, atol=1e-5)
    assert torch.allclose(ends[1.0], clean, atol=1e-5)


def test_attribution_patching_exact_check_is_causal_traces_measurement(vlm):
    # Same corruption, same noise draws: the exact patch of the last position it
    # runs to verify itself is causal_trace's recovery at that layer, digit for digit.
    from evalrx.analyzers.patching.attribution_patching import AttributionPatchingAnalyzer
    from evalrx.analyzers.patching.causal_trace import CausalTraceAnalyzer

    batch = _batch()
    ap = AttributionPatchingAnalyzer(min_effect=0.0, n_verify=1).run(vlm, batch)
    ct = CausalTraceAnalyzer(min_effect=0.0, n_points=3).run(vlm, batch)
    assert ap.findings["verify_layers"] == [2] and ct.findings["layers_probed"][1] == 2
    for cid, pairs in ap.artifacts["verification"].items():
        (last,) = [p for p in pairs if p["site"] == "last"]
        assert last["exact"] == ct.artifacts["recovery_curves"][cid]["last"][1]


@pytest.mark.parametrize("corruption", ["noise", "gray"])
def test_attribution_patching_contract(vlm, corruption):
    from evalrx.analyzers.patching.attribution_patching import AttributionPatchingAnalyzer

    result = AttributionPatchingAnalyzer(corruption=corruption, min_effect=0.0).run(vlm, _batch())
    f = _assert_result_contract(result, "attribution_patching", 4)
    assert f["n_traced"] == 4 and f["direction"] == "denoise" and f["ig_steps"] == 5
    assert f["deepstack_layers"] == (2 if vlm.spec.key.endswith("qwen3_vl") else 0)
    assert set(f["verification"]) == {"last", "relay", "all"}
    assert set(f["mean_share_curves"]) == {f"{label}_{site}" for label in ("fail", "pass")
                                           for site in ("last", "relay")}
    assert all(len(curve) == 6 for curve in f["mean_share_curves"].values())
    assert f["relay_tokens"] == []  # the char tokenizer cannot decode: no token text made up
    for row in f["per_case"]:
        assert 0.0 <= row["ap_last_rise_frac"] <= 1.0 and row["ap_verify_mae"] >= 0.0
        assert "ap_relay_token" not in row
    for curve in result.artifacts["share_curves"].values():
        # Nothing has left the media tokens yet at layer 0: no text token carries any of it.
        assert curve["last"][0] == 0.0 and curve["relay"][0] == 0.0


def test_attribution_patching_names_the_relay_token(vlm, monkeypatch):
    from evalrx.analyzers._internals import access as access_mod
    from evalrx.analyzers.patching.attribution_patching import AttributionPatchingAnalyzer

    class Decoder:
        def decode(self, ids):
            return f"<{ids[0]}>"

    monkeypatch.setattr(access_mod, "tokenizer_of", lambda model: Decoder())
    batch = _batch()
    f = AttributionPatchingAnalyzer(min_effect=0.0, n_verify=0).run(vlm, batch).findings
    for case, row in zip(batch, f["per_case"]):
        ids = vlm.forward(case.inputs, capture={Capability.LOGITS}).token_ids
        assert row["ap_relay_token"] == f"<{ids[len(ids) - 1 - row['ap_relay_pos_from_end']]}>"
    assert sum(count for _, count in f["relay_tokens"]) == 4


def test_cases_failing_their_exact_check_stay_out_of_the_aggregates(vlm, monkeypatch):
    from evalrx.analyzers._internals import access as access_mod
    from evalrx.analyzers.patching.attribution_patching import AttributionPatchingAnalyzer

    class Decoder:
        def decode(self, ids):
            return f"<{ids[0]}>"

    monkeypatch.setattr(access_mod, "tokenizer_of", lambda model: Decoder())
    strict = AttributionPatchingAnalyzer(min_effect=0.0, verify_tol=-1.0).run(vlm, _batch()).findings
    assert strict["n_traced"] == 4 and strict["n_verified"] == 0
    assert all(row["ap_verified"] is False and "ap_relay_token" in row for row in strict["per_case"])
    assert strict["mean_share_curves"] == {} and strict["relay_tokens"] == []
    loose = AttributionPatchingAnalyzer(min_effect=0.0, verify_tol=1e9).run(vlm, _batch()).findings
    assert loose["n_verified"] == 4 and sum(count for _, count in loose["relay_tokens"]) == 4


def test_attribution_patching_small_effects_keep_the_row_but_skip_the_estimate(vlm):
    from evalrx.analyzers.patching.attribution_patching import AttributionPatchingAnalyzer

    result = AttributionPatchingAnalyzer().run(vlm, _batch(2))  # tiny model: effects << 1 logit
    f = result.findings
    assert f["n_cases_analyzed"] == 2 and f["n_traced"] == 0 and f["verification"] is None
    for row in f["per_case"]:
        assert row["ap_effect_traceable"] is False and "ap_last_rise_frac" not in row
    assert result.artifacts["share_curves"] == {}


def test_attribution_patching_writes_no_parameter_grads(vlm):
    from evalrx.analyzers.patching.attribution_patching import AttributionPatchingAnalyzer

    for direction, steps in (("denoise", 1), ("noise", 1), ("denoise", 3)):
        AttributionPatchingAnalyzer(min_effect=0.0, direction=direction, ig_steps=steps,
                                    n_verify=0).run(vlm, _batch(1))
    assert all(p.grad is None for p in vlm._loaded[0].parameters())


def test_attribution_patching_argument_checks_and_token_text():
    from evalrx.analyzers.patching.attribution_patching import (
        AttributionPatchingAnalyzer,
        _token_text,
    )

    with pytest.raises(ValueError, match="direction"):
        AttributionPatchingAnalyzer(direction="sideways")
    with pytest.raises(ValueError, match="corruption"):
        AttributionPatchingAnalyzer(corruption="blur")
    with pytest.raises(ValueError, match="ig_steps"):
        AttributionPatchingAnalyzer(ig_steps=0)

    class Decoder:
        def decode(self, ids):
            return f"<{ids[0]}>"

    assert _token_text(Decoder(), 7) == "<7>"
    assert _token_text(object(), 7) is None


# ── path patching ─────────────────────────────────────────────────────────────

def _clean_and_donor(vlm, inputs):
    """Projection inputs of the clean run and of the noise-corrupted run (seed 0)."""
    from evalrx.analyzers._internals.heads import mixer_hooks
    from evalrx.analyzers.patching.causal_trace import corrupted_variants

    with torch.no_grad():
        with residual_hooks(vlm, capture={0: None}) as res, mixer_hooks(vlm, capture=True) as clean:
            base = vlm.forward(inputs, capture={Capability.LOGITS}).logits[-1]
        media = list(range(3, 19))
        (variant,), _ = corrupted_variants(inputs, "noise", slot="image", media=media,
                                           embeddings=res.captured[0][media], deepstack=False,
                                           noise_scale=3.0, n_noise=1, seed=0)
        with residual_hooks(vlm, replace=variant[1]), mixer_hooks(vlm, capture=True) as donor:
            vlm.forward(inputs, capture={Capability.LOGITS})
    return base, clean.inputs, donor.inputs


def test_freezing_every_head_at_its_own_output_is_a_no_op(vlm):
    from evalrx.analyzers._internals.heads import mixer_hooks

    inputs = Inputs(prompt="what?", image=image(1))
    base, clean, _ = _clean_and_donor(vlm, inputs)
    with torch.no_grad(), mixer_hooks(vlm, replace=clean):
        frozen = vlm.forward(inputs, capture={Capability.LOGITS}).logits[-1]
    assert torch.equal(frozen, base)


def test_a_last_layer_heads_direct_path_is_its_whole_effect(vlm):
    # No head follows the last layer, so freezing the others cannot change what
    # its patch does: direct and total effects coincide exactly.
    from evalrx.analyzers._internals.heads import mixer_hooks, n_heads_of, out_proj
    from evalrx.analyzers.patching.path_patching import sender_inputs

    stack = access.decoder_stack(vlm._loaded[0])
    n_heads = [n_heads_of(m, out_proj(m)) for m in stack.mixers]
    inputs = Inputs(prompt="what?", image=image(1))
    base, clean, donor = _clean_and_donor(vlm, inputs)
    sender = [(stack.n_layers - 1, 0)]
    out = {}
    for freeze in (True, False):
        with torch.no_grad(), mixer_hooks(vlm, replace=sender_inputs(clean, donor, sender, n_heads, freeze=freeze)):
            out[freeze] = vlm.forward(inputs, capture={Capability.LOGITS}).logits[-1]
    assert torch.equal(out[True], out[False]) and not torch.equal(out[True], base)


def test_all_heads_together_carry_the_whole_media_effect(vlm):
    # MLPs are per-position, so the media reaches the answer position only
    # through token mixers: patching every head's direct path at once must
    # reproduce the corrupted run there, i.e. a share of exactly 1.
    from evalrx.analyzers.patching.path_patching import PathPatchingAnalyzer

    f = PathPatchingAnalyzer(min_effect=0.0, top_k=0).run(vlm, _batch()).findings
    for row in f["per_case"]:
        assert row["pp_topk_direct_share"] == pytest.approx(1.0, abs=1e-3)
        assert row["pp_topk_total_share"] == pytest.approx(1.0, abs=1e-3)


def test_path_patching_contract(vlm):
    from evalrx.analyzers.patching.path_patching import PathPatchingAnalyzer

    result = PathPatchingAnalyzer(min_effect=0.0, top_k=5).run(vlm, _batch())
    f = _assert_result_contract(result, "path_patching", 4)
    assert f["n_traced"] == 4 and f["heads_per_layer"] == [4] * 6 and f["top_k"] == 5
    assert 1 <= len(f["recurring_heads"]) <= 10
    for row in f["per_case"]:
        assert row["pp_top_head"].startswith("L") and isinstance(row["pp_top_head_linear"], bool)
        assert -1.0 <= row["pp_screen_spearman"] <= 1.0 and 0.0 <= row["pp_top_head_depth_frac"] <= 1.0
    for table in result.artifacts["head_tables"].values():
        assert len(table) == 5 and {"head", "screen", "direct", "kind"} <= set(table[0])
        screened = [abs(t["screen"]) for t in table]
        assert screened == sorted(screened, reverse=True)  # the top-k BY the screen were patched


def test_path_patching_small_effects_keep_the_row_but_skip_the_heads(vlm):
    from evalrx.analyzers.patching.path_patching import PathPatchingAnalyzer

    f = PathPatchingAnalyzer().run(vlm, _batch(2)).findings  # tiny model: effects << 1 logit
    assert f["n_traced"] == 0 and f["recurring_heads"] == []
    assert all(row["pp_effect_traceable"] is False and "pp_top_head" not in row for row in f["per_case"])


def test_path_patching_argument_checks_and_rank_correlation():
    from evalrx.analyzers.patching.path_patching import PathPatchingAnalyzer, _spearman

    with pytest.raises(ValueError, match="top_k"):
        PathPatchingAnalyzer(top_k=-1)
    with pytest.raises(ValueError, match="corruption"):
        PathPatchingAnalyzer(corruption="blur")
    assert _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert _spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert _spearman([1, 2], [1, 2]) is None


# ── patchscope ────────────────────────────────────────────────────────────────

def test_patchscope_layer0_readout_is_the_source_token_in_the_placeholder(vlm):
    # Layer 0 reads the embedding, so decoding the answer position there is the
    # target prompt with the source's last token written into the placeholder.
    from evalrx.analyzers.lens.patchscope import IDENTITY_PROMPT, PatchscopeAnalyzer
    from tests.tiny_whitebox import CharTokenizer

    case = _batch(1)
    result = PatchscopeAnalyzer().run(vlm, case)
    trace = vlm.forward(case[0].inputs, capture={Capability.LOGITS})
    target = int(trace.logits[-1].argmax())
    ids = CharTokenizer()(IDENTITY_PROMPT)["input_ids"]
    ids[-1] = trace.token_ids[-1]
    with torch.no_grad():
        plain = torch.softmax(vlm._loaded[0](input_ids=torch.tensor([ids])).logits[0, -1].float(), -1)
    curve = result.artifacts["readout_curves"][case[0].id]
    assert curve["p_target"][0] == pytest.approx(float(plain[target]), abs=1e-4)
    assert curve["top1"][0] == str(int(plain.argmax()))  # no decoder: the id, never a made-up word
    assert result.findings["patch_check_max_diff"] == 0.0


def test_patchscope_contract_with_media_words(vlm):
    from evalrx.analyzers.lens.patchscope import PatchscopeAnalyzer

    result = PatchscopeAnalyzer(media_layers=2).run(vlm, _batch())
    f = _assert_result_contract(result, "patchscope", 4)
    assert f["layers_read"] == list(range(6)) and f["media_layers_read"] == [2, 5]  # not the embedding
    for row in f["per_case"]:
        assert 0.0 <= row["ps_first_depth_frac"] <= 1.0 and 0.0 <= row["ps_agree_mid"] <= 1.0
        assert row["ps_late_drop"] == pytest.approx(row["ps_peak_prob"] - row["ps_final_prob"], abs=2e-4)
    for words in result.artifacts["media_words"].values():
        assert sorted(words) == [2, 5] and all(len(w) == 16 for w in words.values())
    assert set(f["mean_p_target_curves"]) == {"fail", "pass"}


def test_patchscope_reads_text_only_cases_and_skips_missing_gold(vlm):
    from evalrx.analyzers.lens.patchscope import PatchscopeAnalyzer

    text_only = CaseBatch([FailureCase(inputs=Inputs(prompt="two plus two is"), expected="4"),
                           FailureCase(inputs=Inputs(prompt="no gold here"))])
    f = PatchscopeAnalyzer(target="gold", n_points=3).run(vlm, text_only).findings
    assert f["n_cases_analyzed"] == 1 and f["skip_reasons"] == {"no_gold": 1}
    assert f["layers_read"] == [0, 2, 5]


class _CharLetters:
    """CharTokenizer ids with a decoder: every id reads as a distinct three-letter word."""

    def __call__(self, text, add_special_tokens=False):
        from tests.tiny_whitebox import CharTokenizer

        return CharTokenizer()(text, add_special_tokens)

    def decode(self, ids):
        return _LetterDecoder().decode(ids)


def test_specific_words_tell_a_case_from_the_template():
    from evalrx.analyzers._internals.words import specific_words

    words = {"a": ("p1", {"is", "snowboard", "image"}), "b": ("p2", {"is", "car", "image"}),
             "c": ("p3", {"is", "dog", "image"}), "d": ("p3", {"is", "dog", "image"})}  # d asks c's question
    got, inferred = specific_words(words)
    assert inferred and got == {"a": {"snowboard"}, "b": {"car"}, "c": {"dog"}, "d": {"dog"}}
    few, inferred = specific_words({"a": ("p1", {"x"}), "b": ("p2", {"y"})})
    assert not inferred and few == {"a": set(), "b": set()}


def test_patchscope_counts_the_question_word_on_media_and_reports_its_self_check(vlm, monkeypatch):
    from evalrx.analyzers.lens.patchscope import PatchscopeAnalyzer

    monkeypatch.setattr(access, "tokenizer_of", lambda model: _CharLetters())
    result = PatchscopeAnalyzer(media_layers=2).run(vlm, _batch())
    f = result.findings
    assert f["query_words_inferred"] is True and f["patch_fidelity"]["faithful"] is True
    for row in f["per_case"]:
        own = set(result.artifacts["query_words"][row["sample_id"]])
        assert len(own) == 1  # only the picture number differs between the prompts
        decoded = [w.strip().lower() for ws in result.artifacts["media_words"][row["sample_id"]].values() for w in ws]
        assert row["ps_media_query_hits"] == sum(w in own for w in decoded)
        assert row["ps_media_query_share"] == round(row["ps_media_query_hits"] / len(decoded), 4)


def test_patchscope_refuses_when_the_patch_is_not_inert(vlm, monkeypatch):
    from evalrx.analyzers._internals import residual
    from evalrx.analyzers.lens.patchscope import PatchscopeAnalyzer

    real = residual.batched_patch

    def scaled(model, patches, analyzer=None):
        return real(model, {layer: [(r, p, v * 3.0) for r, p, v in rows] for layer, rows in patches.items()},
                    analyzer=analyzer)

    monkeypatch.setattr(residual, "batched_patch", scaled)
    with pytest.raises(attn_tap.FidelityError) as info:
        PatchscopeAnalyzer().run(vlm, _batch(1))
    assert info.value.verdict["faithful"] is False and "layer" in info.value.verdict


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


# ── attention temperature ─────────────────────────────────────────────────────

def test_temperature_edits_are_exact(vlm):
    # Layer-0 attention depends on the input only, so edited and clean rows compare directly.
    from evalrx.analyzers._internals.attn_tap import temperature

    inputs = Inputs(prompt="what is here?", image=image(0))
    media = list(range(3, 19))
    probs: dict = {}

    def grab(tag):
        def record(layer, p):
            if layer == 0:
                probs[tag] = p.detach()[0].float()
        return record

    with torch.no_grad():
        with attention_tap(vlm, on_probs=grab("clean")):
            base = vlm.forward(inputs, capture={Capability.LOGITS}).logits
        with attention_tap(vlm, [temperature(media, 0.5, within=True)], on_probs=grab("within")):
            vlm.forward(inputs, capture={Capability.LOGITS})
        with attention_tap(vlm, [temperature(range(64), 2.0)], on_probs=grab("all")):
            vlm.forward(inputs, capture={Capability.LOGITS})
        with attention_tap(vlm, [temperature(media, 1.0, within=True)]):
            same = vlm.forward(inputs, capture={Capability.LOGITS}).logits
    assert torch.equal(same, base)                                  # T = 1 is no edit
    p, w, a = probs["clean"], probs["within"], probs["all"]
    last = p.shape[-1] - 1
    mass = p[:, last, media].sum(-1)
    assert torch.allclose(w[:, last, media].sum(-1), mass, atol=1e-6)  # how much it looks: kept
    sharpened = p[:, last, media] ** 2
    assert torch.allclose(w[:, last, media], sharpened / sharpened.sum(-1, keepdim=True) * mass[:, None], atol=1e-6)
    assert torch.allclose(w[:, last, 19:], p[:, last, 19:], atol=1e-6)  # text keys untouched
    flat = torch.softmax(torch.log(p[:, last, : last + 1]) / 2.0, -1)
    assert torch.allclose(a[:, last, : last + 1], flat, atol=1e-6)


def test_temperature_must_be_positive_and_finite():
    from evalrx.analyzers._internals.attn_tap import temperature

    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            temperature([1, 2], bad)


@pytest.mark.parametrize("scope", ["within", "all"])
def test_attention_temperature_contract(vlm, scope):
    from evalrx.analyzers.patching.attention_temperature import AttentionTemperatureAnalyzer

    f = _assert_result_contract(AttentionTemperatureAnalyzer(scope=scope).run(vlm, _batch()),
                                "attention_temperature", 4)
    assert f["tap_fidelity"]["tap_faithful"] is True and f["scope"] == scope
    assert f["tt_edited_layers"] == 6 and f["temperatures"] == [0.5, 2.0]
    for row in f["per_case"]:
        assert isinstance(row["tt_flips_sharp"], bool) and 0.0 <= row["tt_media_entropy"] <= 1.0
        assert row["tt_slope"] == pytest.approx(
            (row["tt_delta_flat"] - row["tt_delta_sharp"]) / math.log(4.0), abs=2e-4)


def test_attention_temperature_validates_params(vlm):
    from evalrx.analyzers.patching.attention_temperature import AttentionTemperatureAnalyzer

    with pytest.raises(ValueError, match="sharp"):
        AttentionTemperatureAnalyzer(sharp=1.5)
    with pytest.raises(ValueError, match="scope"):
        AttentionTemperatureAnalyzer(scope="text")
    with pytest.raises(ValueError, match="band"):
        AttentionTemperatureAnalyzer(band="top")
    assert AttentionTemperatureAnalyzer(band="late").run(vlm, _batch(1)).findings["tt_edited_layers"] == 2


# ── embed lens ────────────────────────────────────────────────────────────────

class _LetterDecoder:
    """Decodes every id to a distinct three-letter word, so content words exist."""

    def decode(self, ids):
        i = int(ids[0])
        return chr(ord("a") + i % 26) + chr(ord("a") + (i // 26) % 26) + "x"


def test_embed_lens_neighbours_match_brute_force(vlm):
    from evalrx.analyzers.lens.embed_lens import EmbedLensAnalyzer

    batch = _batch(2)
    result = EmbedLensAnalyzer(top_k=3, vocab_chunk=7).run(vlm, batch)  # many blocks to merge
    table = vlm._loaded[0].get_input_embeddings().weight.float()
    for case in batch:
        with torch.no_grad(), residual_hooks(vlm, capture={0: None}) as res:
            vlm.forward(case.inputs, capture={Capability.LOGITS})
        x = res.captured[0][3:19].float()
        cos = torch.nn.functional.normalize(x, dim=-1) @ torch.nn.functional.normalize(table, dim=-1).T
        want = cos.topk(3, dim=-1)
        got = result.artifacts["tokens"][case.id]
        assert got["top_ids"] == want.indices.tolist()
        assert got["max_sim"] == pytest.approx([round(v, 4) for v in want.values[:, 0].tolist()], abs=1e-4)


def test_embed_lens_contract_and_grounding(vlm, monkeypatch):
    from evalrx.analyzers._internals import access as access_mod
    from evalrx.analyzers.lens.embed_lens import EmbedLensAnalyzer

    monkeypatch.setattr(access_mod, "tokenizer_of", lambda model: _LetterDecoder())
    result = EmbedLensAnalyzer().run(vlm, _batch())
    f = _assert_result_contract(result, "embed_lens", 4)
    assert f["recurring_inferred"] is True
    for row in f["per_case"]:
        assert row["el_n_media_tokens"] == 16
        parts = row["el_frac_specific"] + row["el_frac_recurring"] + row["el_frac_high_norm"]
        assert parts == pytest.approx(1.0, abs=1e-3)
        sims = result.artifacts["prompt_sims"][row["sample_id"]]
        assert row["el_prompt_best_sim"] == max(sims.values()) and row["el_prompt_best_word"] in sims
        # the question word: only words most of the prompts do not share
        own = result.artifacts["query_words"][row["sample_id"]]
        assert own and row["el_query_word"] in own
        assert row["el_query_sim"] == max(sims[w] for w in own if w in sims)
    assert f["query_words_inferred"] is True


def test_embed_lens_reads_any_depth(vlm):
    from evalrx.analyzers.lens.embed_lens import EmbedLensAnalyzer

    batch = _batch(2)
    deep = EmbedLensAnalyzer(layer=0.6, top_k=2).run(vlm, batch)
    assert deep.findings["layer_used"] == 3  # round(0.6 * 5)
    table = vlm._loaded[0].get_input_embeddings().weight.float()
    with torch.no_grad(), residual_hooks(vlm, capture={3: None}) as res:
        vlm.forward(batch[0].inputs, capture={Capability.LOGITS})
    cos = (torch.nn.functional.normalize(res.captured[3][3:19].float(), dim=-1)
           @ torch.nn.functional.normalize(table, dim=-1).T)
    assert deep.artifacts["tokens"][batch[0].id]["top_ids"] == cos.topk(2, dim=-1).indices.tolist()


def test_embed_lens_counts_each_image_once(vlm):
    # Four questions about ONE image are one image's worth of evidence: every
    # nearest word of it "recurs" in 100% of the distinct images, but only one
    # image backs that, below min_images, so nothing is called recurring.
    from evalrx.analyzers.lens.embed_lens import EmbedLensAnalyzer

    same = CaseBatch([FailureCase(inputs=Inputs(prompt=f"question {i}?", image=image(0))) for i in range(4)])
    f = EmbedLensAnalyzer().run(vlm, same).findings
    assert f["n_cases_analyzed"] == 4 and f["n_distinct_media"] == 1
    assert f["recurring_inferred"] is False and f["recurring_words"] == []


def test_embed_lens_infers_nothing_recurring_from_too_few_images(vlm):
    from evalrx.analyzers.lens.embed_lens import EmbedLensAnalyzer

    f = EmbedLensAnalyzer(min_images=10).run(vlm, _batch()).findings
    assert f["recurring_inferred"] is False and f["recurring_words"] == []
    assert all(row["el_frac_recurring"] == 0.0 for row in f["per_case"])
    with pytest.raises(ValueError, match="recurring_frac"):
        EmbedLensAnalyzer(recurring_frac=0.0)


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
