"""Weight-edit continuation audit: the edit followed along recorded generations.

Checked on the planted-defect model, where a "generation" can be written by
hand: after a trigger token the defective model emits the wrong token, so a
recorded text that alternates trigger / wrong token is this model's loop. The
true edit must make that repeated text less likely and leave trigger-free text
alone; a matched random change must do neither.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evalrx.analyzers.weights import WeightEditContinuationAudit  # noqa: E402
from evalrx.analyzers.weights.continuation_audit import repetition_mask  # noqa: E402
from evalrx.core.capability import Capability, CapabilityError  # noqa: E402
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label  # noqa: E402
from evalrx.core.registry import registry  # noqa: E402
from evalrx.probing import ProbeToolbox, spec_for  # noqa: E402
from evalrx.wsp import TensorEdit, WeightEdit, applied, contrast_edit  # noqa: E402
from tests.conftest import FakeModel  # noqa: E402
from tests.toy_wsp import (  # noqa: E402
    DEFECT_LAYER,
    WRONG_TOKEN,
    D,
    ToyModel,
    echo_batch,
    snapshot,
    unchanged,
)

# Trigger-free text first (never repeats a bigram), then the model's own loop.
_PLAIN = "3 4 5 6 2 5 3 6 4 2 6 3"
_LOOP = " ".join(["0 7"] * 8)
# A prompt whose bigram occurs nowhere in the texts above.
_PROMPT = "2 2"


def _true_edit(model) -> WeightEdit:
    pool = echo_batch(4, 6)
    return contrast_edit(model, [c for c in pool if c.label == Label.FAIL],
                         [c for c in pool if c.label == Label.PASS], n_layers=1, n_columns=1)


def _gen(prompt: str, continuation) -> FailureCase:
    return FailureCase(inputs=Inputs(prompt=prompt), observed=continuation, label=Label.FAIL)


def _batch() -> CaseBatch:
    return CaseBatch([_gen(_PROMPT, f"{_PLAIN} {_LOOP}"),
                      _gen(_PROMPT, f"{_PLAIN} {' '.join(['1 7'] * 8)}")])


def _run(model, cases=None, **kwargs):
    kwargs.setdefault("repeat_ngram", 2)
    return WeightEditContinuationAudit(edit=_true_edit(model), **kwargs).run(
        model, cases or _batch()).findings


# ----------------------------------------------------------------------
# Repetition mask
# ----------------------------------------------------------------------

def test_repetition_mask_marks_contexts_already_seen():
    ids = [0, 7, 0, 7, 0, 7, 3, 4]
    assert repetition_mask(ids, 2) == [False, False, False, False, True, True, True, False]
    assert repetition_mask(ids, 9) == [False] * 8
    assert repetition_mask([], 2) == []
    assert repetition_mask([5, 5, 5, 5], 1) == [False, False, True, True]


# ----------------------------------------------------------------------
# Contract
# ----------------------------------------------------------------------

def test_registered_as_a_weight_edit_probe():
    assert registry.analyzers.has("weight_edit_continuation_audit")
    spec = spec_for("weight_edit_continuation_audit")
    assert spec.family == "weight_edit" and spec.intervenes is True
    assert spec.injected_params == ("edit",)
    assert {"groups", "repeat_ngram", "max_tokens"} <= set(spec.parameters_schema()["properties"])


def test_result_contract_and_restoration():
    model = ToyModel()
    snap = snapshot(model)
    f = _run(model, n_seeds=2)
    json.dumps(f)
    assert f["n_cases_analyzed"] == 2 and f["n_skipped"] == 0
    # 12 plain + 16 loop tokens per case, every continuation token scored once.
    assert f["n_positions"] == 56
    assert f["n_positions_loop"] + f["n_positions_productive"] == 56
    assert len(f["per_case"]) == 2
    row = f["per_case"][0]
    assert row["wc_n_positions"] == 28 and 0 < row["wc_frac_loop"] < 1
    assert set(WeightEditContinuationAudit.signal_docs) <= set(row)
    assert unchanged(model, snap)
    for layer in model.module.layers:
        assert not layer.mlp.down_proj._forward_pre_hooks
        assert not layer.mlp.down_proj._forward_hooks


def test_positions_are_grouped_by_the_repetition_mask():
    model = ToyModel()
    f = _run(model, controls=(), steering=False)
    # Per case: prompt (2) + plain (12) + "0 7" x 8. With bigram windows the
    # first four loop tokens are each predicted in a context not seen before;
    # the remaining twelve are predicted in a context that already occurred.
    assert f["n_positions_loop"] == 2 * 12
    assert f["n_positions_productive"] == 2 * 16


# ----------------------------------------------------------------------
# Reading the planted defect
# ----------------------------------------------------------------------

def test_the_edit_acts_in_the_loop_and_not_on_plain_text():
    model = ToyModel()
    f = _run(model, n_seeds=3)
    # The repeated text becomes much less likely...
    assert f["mean_dnll_edit_loop"] > 0.5
    # ...far more so than text outside the loop,
    # (the productive group still holds the loop's first cycle, which cannot
    # count as a repeat yet, so it is not exactly zero)
    assert f["dnll_edit_loop_minus_productive"] > 0.4
    assert f["mean_kl_edit_loop"] > 3 * f["mean_kl_edit_productive"]
    # a matched random direction or other neurons barely move the output,
    assert f["specificity_random_direction_loop"] > 10
    assert f["specificity_random_columns_loop"] > 10
    assert abs(f["mean_dnll_random_direction_loop"]) < 0.1
    # and one constant shift does not reproduce an input-dependent edit.
    assert f["steer_recovery_loop"] < 0.5


def test_plain_text_alone_is_left_untouched():
    model = ToyModel()
    cases = CaseBatch([_gen(_PROMPT, _PLAIN), _gen(_PROMPT, "6 5 4 3 2 3 5 2 4 6")])
    f = _run(model, cases, controls=(), steering=False)
    assert f["n_positions_loop"] == 0
    assert "mean_dnll_edit_loop" not in f and "dnll_edit_loop_minus_productive" not in f
    assert abs(f["mean_dnll_edit_productive"]) < 0.02
    assert f["frac_flips_edit_productive"] == 0.0


def test_per_position_values_match_an_independent_measurement():
    model = ToyModel()
    edit = _true_edit(model)
    case = _gen("4 2", "0 7 0 7 3")
    f = WeightEditContinuationAudit(edit=edit, groups="none", controls=(),
                                    steering=False).run(model, CaseBatch([case])).findings
    ids = torch.tensor([[4, 2, 0, 7, 0, 7, 3]])
    targets = [2, 3, 4, 5, 6]

    def nll(logits):
        lp = torch.log_softmax(logits[0].double(), dim=-1)
        return torch.stack([-lp[i - 1, int(ids[0, i])] for i in targets])

    with torch.no_grad():
        clean = model.module(input_ids=ids)
        with applied(model, edit):
            edited = model.module(input_ids=ids)
    assert f["n_positions_all"] == 5
    assert f["mean_nll_clean_all"] == pytest.approx(float(nll(clean).mean()), abs=1e-5)
    assert f["mean_dnll_edit_all"] == pytest.approx(float((nll(edited) - nll(clean)).mean()),
                                                    abs=1e-5)


def test_a_null_edit_measures_zero_everywhere():
    model = ToyModel()
    null = WeightEdit([TensorEdit.rank_one(
        f"layers.{DEFECT_LAYER}.mlp.down_proj.weight", np.eye(D)[WRONG_TOKEN], 0.0)])
    f = WeightEditContinuationAudit(edit=null, repeat_ngram=2, controls=(),
                                    steering=False).run(model, _batch()).findings
    for group in ("loop", "productive"):
        assert f[f"mean_dnll_edit_{group}"] == 0.0 and f[f"mean_kl_edit_{group}"] == 0.0


# ----------------------------------------------------------------------
# Options, skips, refusals
# ----------------------------------------------------------------------

def test_no_grouping_keeps_one_group():
    model = ToyModel()
    f = _run(model, groups="none", controls=("random_direction",), steering=False)
    assert f["n_positions_all"] == 56 and f["repeat_ngram"] is None
    assert "specificity_random_direction_all" in f
    assert "wc_frac_loop" not in f["per_case"][0]


def test_truncation_limits_the_scored_positions():
    model = ToyModel()
    f = _run(model, max_tokens=10, controls=(), steering=False)
    assert f["n_positions"] == 2 * (10 - 2)        # two prompt tokens per case


def test_chunking_does_not_change_the_result():
    model = ToyModel()
    a = _run(model, controls=(), steering=False, chunk=3)
    b = _run(model, controls=(), steering=False, chunk=256)
    for key in ("mean_dnll_edit_loop", "mean_kl_edit_loop", "mean_dnll_edit_productive"):
        assert a[key] == pytest.approx(b[key], abs=1e-9)


def test_cases_without_a_recorded_generation_are_skipped_and_counted():
    model = ToyModel()
    cases = CaseBatch([_gen("4 2", f"{_PLAIN} {_LOOP}"), _gen("4 2", None), _gen("4 2", "  ")])
    f = _run(model, cases, controls=(), steering=False)
    assert f["n_cases_analyzed"] == 1 and f["skip_reasons"] == {"no_continuation": 2}
    with pytest.raises(ValueError, match="no case could be analysed"):
        _run(model, CaseBatch([_gen("4 2", None)]))


def test_a_prompt_longer_than_the_limit_is_skipped():
    model = ToyModel()
    f = _run(model, CaseBatch([_gen("4 2 3 5 6", "0 7"), _gen("4 2", f"{_PLAIN} {_LOOP}")]),
             max_tokens=5, controls=(), steering=False)
    assert f["skip_reasons"] == {"no_continuation_tokens_after_truncation": 1}


def test_a_control_that_cannot_be_built_is_reported():
    model = ToyModel()
    whole = WeightEdit([TensorEdit.rank_one(
        f"layers.{DEFECT_LAYER}.mlp.down_proj.weight", np.eye(D)[WRONG_TOKEN], 1.0)])
    f = WeightEditContinuationAudit(edit=whole, repeat_ngram=2).run(model, _batch()).findings
    assert "random_columns" in f["controls_not_run"]
    assert "specificity_random_direction_loop" in f
    assert "mean_dnll_random_columns_loop" not in f


def test_steering_is_skipped_with_a_reason_for_non_linear_tensors():
    model = ToyModel()
    edit = WeightEdit([TensorEdit(name="embed.weight", delta=0.01 * np.ones((D, D)))])
    f = WeightEditContinuationAudit(edit=edit, repeat_ngram=2, controls=()).run(
        model, _batch()).findings
    assert "steering_not_run" in f and "steer_recovery_loop" not in f


@pytest.mark.parametrize("kwargs,message", [
    ({"groups": "paragraph"}, "groups must be one of"),
    ({"controls": ("shuffle",)}, "unknown control"),
    ({"n_seeds": 0}, "must be positive"),
    ({"repeat_ngram": 0}, "must be positive"),
    ({"max_tokens": 2}, "max_tokens"),
])
def test_bad_settings_are_refused(kwargs, message):
    model = ToyModel()
    with pytest.raises(ValueError, match=message):
        _run(model, **kwargs)


def test_the_model_is_restored_when_a_forward_fails_mid_audit():
    model = ToyModel()
    snap = snapshot(model)
    edit = _true_edit(model)
    module = model.module
    original = module.forward
    state = {"n": 0}

    def flaky(input_ids):
        state["n"] += 1
        if state["n"] == 3:        # case 1 clean, case 1 edited, then this one
            raise RuntimeError("device lost")
        return original(input_ids)

    module.forward = flaky
    try:
        with pytest.raises(RuntimeError, match="device lost"):
            WeightEditContinuationAudit(edit=edit, repeat_ngram=2, controls=(),
                                        steering=False).run(model, _batch())
    finally:
        module.forward = original
    assert unchanged(model, snap)


def test_models_without_internals_are_refused_clearly():
    edit = WeightEdit([TensorEdit(name="layers.0.mlp.down_proj.weight", delta=np.ones((D, D)))])
    with pytest.raises(CapabilityError):
        WeightEditContinuationAudit(edit=edit).run(
            FakeModel(capabilities={Capability.GENERATE}), "x")
    with pytest.raises(ValueError, match="white-box"):
        WeightEditContinuationAudit(edit=edit).run(
            FakeModel(capabilities={Capability.LOGITS}), "x")


def test_runs_as_a_probe_tool_with_json_arguments():
    model = ToyModel()
    box = ProbeToolbox(model, _batch(), overrides={
        "weight_edit_continuation_audit": WeightEditContinuationAudit(edit=_true_edit(model))})
    result = box.call("weight_edit_continuation_audit",
                      {"repeat_ngram": 2, "controls": ["random_direction"], "steering": False})
    assert result.findings["repeat_ngram"] == 2
    assert "specificity_random_direction_loop" in result.findings


# ----------------------------------------------------------------------
# A real local model handle
# ----------------------------------------------------------------------

def test_on_a_real_local_model_handle():
    """Mechanics on ``HFLocalModel`` over a from-scratch Llama: prompt tokens come
    from the handle, the continuation is appended, and the weights are restored.
    The weights are random, so only the bookkeeping is asserted."""
    pytest.importorskip("transformers")
    from evalrx.core.spec import ModelSpec
    from evalrx.models.backends.base import RuntimeConfig
    from evalrx.models.backends.hf.model import HFLocalModel
    from evalrx.wsp import write_sites
    from tests.test_eval_agent.test_fix_agent import _tiny_llama, _TinyWordTokenizer

    class _Tok(_TinyWordTokenizer):
        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in text.strip().lower().split()]
            if return_tensors == "pt":
                input_ids = torch.tensor([ids or [3]], dtype=torch.long)
                return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
            return {"input_ids": ids}

    spec = ModelSpec(key="tiny-llama-cont", family="fake", model_type="fake_llm", hf_repo="")
    model = HFLocalModel(spec, RuntimeConfig(device="cpu", dtype="float32", max_new_tokens=3))
    model._hf = (_tiny_llama(), _Tok())
    snap = snapshot(model)
    site = write_sites(model)[1]
    torch.manual_seed(0)
    edit = WeightEdit([TensorEdit.rank_one(site.name, torch.randn(16), 6.0, cols=[1, 5, 9])])
    cases = CaseBatch([_gen("classify alpha", "yes no yes no yes no yes no beta alpha")])
    f = WeightEditContinuationAudit(edit=edit, repeat_ngram=2, n_seeds=1).run(model, cases).findings
    json.dumps(f)
    assert f["n_positions"] == 10                 # ten continuation tokens after two prompt tokens
    assert f["n_positions_loop"] == 5 and f["n_positions_productive"] == 5
    assert f["mean_kl_edit_loop"] > 0
    assert "steer_recovery_loop" in f
    assert f["structure"][0]["alpha_estimate"] == pytest.approx(6.0, abs=1e-2)
    assert unchanged(model, snap)
