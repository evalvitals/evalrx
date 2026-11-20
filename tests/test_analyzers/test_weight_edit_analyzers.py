"""Weight-edit analyzers: structure from the weights, and the audit against controls.

The audit is checked on a model with a planted defect, where the right reading
is known in advance: the true edit moves the failing cases toward the gold
answer and leaves the passing ones alone, a random direction does not, and a
constant activation shift cannot stand in for an edit that only acts on the
inputs that trigger the defect.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evalrx.analyzers._internals.targets import build_margin  # noqa: E402
from evalrx.analyzers.weights import WeightEditAudit, WeightEditStructure  # noqa: E402
from evalrx.core.capability import Capability, CapabilityError  # noqa: E402
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label  # noqa: E402
from evalrx.core.registry import registry  # noqa: E402
from evalrx.core.result import Result  # noqa: E402
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


def _true_edit(model, alpha: float = 1.0) -> WeightEdit:
    pool = echo_batch(4, 6)
    return contrast_edit(model, [c for c in pool if c.label == Label.FAIL],
                         [c for c in pool if c.label == Label.PASS],
                         n_layers=1, n_columns=1, alpha=alpha)


def test_both_analyzers_are_registered_as_weight_edit_probes():
    for name in ("weight_edit_structure", "weight_edit_audit"):
        assert registry.analyzers.has(name)
        spec = spec_for(name)
        assert spec.family == "weight_edit"
        assert spec.injected_params == ("edit",)
        assert "edit" not in spec.parameters_schema()["properties"]
    assert spec_for("weight_edit_audit").intervenes is True
    assert spec_for("weight_edit_structure").intervenes is False


def test_structure_reports_the_rule_and_leaves_the_model_alone():
    model = ToyModel()
    snap = snapshot(model)
    result = WeightEditStructure(edit=_true_edit(model, alpha=6.0)).run(model, echo_batch())
    assert isinstance(result, Result) and result.analyzer == "weight_edit_structure"
    f = result.findings
    json.dumps(f)
    assert f["n_tensors"] == 1
    assert f["alpha_estimates"] == [6.0]
    assert f["max_rank_one_residual"] < 1e-5
    assert f["tensors"][0]["changed_columns"] == 1
    assert f["max_relative_change"] > 0
    assert unchanged(model, snap)


def test_structure_can_require_the_recorded_base():
    model = ToyModel()
    edit = WeightEdit([TensorEdit(name="layers.0.mlp.down_proj.weight",
                                  delta=np.ones((D, D)), base_sha256="0" * 64)])
    with pytest.raises(ValueError, match="differs from the one the edit was made on"):
        WeightEditStructure(edit=edit, verify_base=True).run(model, echo_batch())


def test_audit_result_contract():
    model = ToyModel()
    batch = echo_batch(4, 6)
    snap = snapshot(model)
    result = WeightEditAudit(edit=_true_edit(model), target="gold", n_seeds=2).run(model, batch)
    f = result.findings
    json.dumps(f)
    assert f["n_cases_analyzed"] == len(batch) and f["n_skipped"] == 0
    assert len(f["per_case"]) == len(batch)
    assert {r["sample_id"] for r in f["per_case"]} == {c.id for c in batch}
    for row in f["per_case"]:
        assert {"we_margin_clean", "we_delta_edit", "we_kl_edit", "we_flips_top1",
                "we_delta_random_direction", "we_kl_random_direction",
                "we_delta_random_columns", "we_kl_random_columns", "we_delta_steer"} <= set(row)
        assert row["we_kl_edit"] >= 0
    documented = set(WeightEditAudit.signal_docs)
    assert documented <= set(f["per_case"][0])
    assert f["n_fail"] == 4 and f["n_pass"] == 6
    assert unchanged(model, snap)


def test_audit_reads_the_planted_defect_correctly():
    model = ToyModel()
    f = WeightEditAudit(edit=_true_edit(model), target="gold", n_seeds=3).run(
        model, echo_batch(4, 6)).findings
    # The edit moves failing cases toward gold, flips every one of them...
    assert f["mean_delta_edit_fail"] > 1.0
    assert f["frac_flips_fail"] == 1.0
    # ...and has no side effect on the cases the model already got right.
    assert abs(f["mean_delta_edit_pass"]) < 0.05
    assert f["frac_flips_pass"] == 0.0
    # A matched random direction, or the same rule on other neurons, moves the
    # output far less than the edit does.
    assert f["specificity_random_direction"] > 2.0
    assert f["specificity_random_columns"] > 2.0
    # The defect is input-dependent (it fires on trigger tokens only), so one
    # constant shift for all cases cannot reproduce the edit.
    assert f["steer_recovery"] < 0.9


def test_a_random_edit_is_not_reported_as_specific():
    model = ToyModel()
    torch.manual_seed(0)
    arbitrary = WeightEdit([TensorEdit.rank_one(
        f"layers.{DEFECT_LAYER}.mlp.down_proj.weight", torch.randn(D), 1.0, cols=[0, 2])])
    f = WeightEditAudit(edit=arbitrary, target="gold", n_seeds=5).run(
        model, echo_batch(4, 6)).findings
    assert f["frac_flips_top1"] == 0.0
    assert f["specificity_random_direction"] < 5.0


def test_per_case_change_matches_an_independent_measurement():
    model = ToyModel()
    batch = echo_batch(2, 2)
    edit = _true_edit(model)
    rows = {r["sample_id"]: r for r in WeightEditAudit(
        edit=edit, target="gold", controls=(), steering=False).run(model, batch).findings["per_case"]}
    for case in batch:
        clean = model.forward(case.inputs, capture={Capability.LOGITS}).logits[-1:].float()
        margin, _ = build_margin("gold", clean, model=model, case=case)
        with applied(model, edit):
            edited = model.forward(case.inputs, capture={Capability.LOGITS}).logits[-1:].float()
        expected = margin.value(edited) - margin.value(clean)
        assert rows[case.id]["we_delta_edit"] == pytest.approx(expected, abs=1e-5)


def test_a_null_edit_measures_zero():
    model = ToyModel()
    null = _true_edit(model, alpha=0.0)
    f = WeightEditAudit(edit=null, controls=(), steering=False).run(model, echo_batch()).findings
    assert f["mean_kl_edit"] == 0.0 and f["mean_delta_edit"] == 0.0
    assert f["frac_flips_top1"] == 0.0


def test_controls_and_steering_can_be_switched_off():
    model = ToyModel()
    f = WeightEditAudit(edit=_true_edit(model), controls=("random_direction",),
                        steering=False, n_seeds=1).run(model, echo_batch()).findings
    assert "specificity_random_direction" in f
    assert "specificity_random_columns" not in f
    assert "steer_recovery" not in f and "steering_not_run" not in f
    assert "we_delta_steer" not in f["per_case"][0]


def test_a_control_that_cannot_be_built_is_reported():
    model = ToyModel()
    whole = WeightEdit([TensorEdit.rank_one(
        f"layers.{DEFECT_LAYER}.mlp.down_proj.weight", np.eye(D)[WRONG_TOKEN], 1.0)])
    f = WeightEditAudit(edit=whole, n_seeds=1).run(model, echo_batch()).findings
    assert "random_columns" in f["controls_not_run"]
    assert "specificity_random_direction" in f


def test_steering_is_skipped_with_a_reason_for_non_linear_tensors():
    model = ToyModel()
    edit = WeightEdit([TensorEdit(name="embed.weight", delta=0.01 * np.ones((D, D)))])
    f = WeightEditAudit(edit=edit, controls=()).run(model, echo_batch()).findings
    assert "steering_not_run" in f and "steer_recovery" not in f


def test_gold_target_skips_cases_without_a_gold_answer():
    model = ToyModel()
    batch = CaseBatch([*echo_batch(2, 2),
                       FailureCase(inputs=Inputs(prompt="3 4 5"), label=Label.PASS)])
    f = WeightEditAudit(edit=_true_edit(model), target="gold", controls=(),
                        steering=False).run(model, batch).findings
    assert f["n_cases_analyzed"] == 4 and f["skip_reasons"] == {"no_gold": 1}


def test_bad_settings_are_refused():
    model = ToyModel()
    edit = _true_edit(model)
    with pytest.raises(ValueError, match="unknown control"):
        WeightEditAudit(edit=edit, controls=("shuffle",)).run(model, echo_batch())
    with pytest.raises(ValueError, match="n_seeds"):
        WeightEditAudit(edit=edit, n_seeds=0).run(model, echo_batch())


def test_the_model_is_restored_when_a_forward_fails_mid_audit():
    model = ToyModel()
    snap = snapshot(model)
    edit = _true_edit(model)  # built before the forward pass is made flaky
    original = model.forward
    state = {"n": 0}

    def flaky(inputs, capture, spec=None):
        state["n"] += 1
        if state["n"] == 8:  # after the clean pass, inside the edited pass
            raise RuntimeError("device lost")
        return original(inputs, capture, spec)

    model.forward = flaky
    with pytest.raises(RuntimeError, match="device lost"):
        WeightEditAudit(edit=edit, controls=(), steering=False).run(model, echo_batch(3, 3))
    assert unchanged(model, snap)
    for layer in model.module.layers:
        assert not layer.mlp.down_proj._forward_hooks
        assert not layer.mlp.down_proj._forward_pre_hooks


def test_models_without_internals_are_refused_clearly():
    edit = WeightEdit([TensorEdit(name="layers.0.mlp.down_proj.weight", delta=np.ones((D, D)))])
    with pytest.raises(CapabilityError):
        WeightEditAudit(edit=edit).run(FakeModel(capabilities={Capability.GENERATE}), "x")
    with pytest.raises(ValueError, match="white-box"):
        WeightEditStructure(edit=edit).run(FakeModel(capabilities={Capability.LOGITS}), "x")


def test_audit_runs_as_a_probe_tool_with_json_arguments():
    model = ToyModel()
    box = ProbeToolbox(model, echo_batch(),
                       overrides={"weight_edit_audit": WeightEditAudit(edit=_true_edit(model))})
    report = box.describe(families=["weight_edit"])
    assert report["weight_edit_audit"]["available"] is True
    assert report["weight_edit_structure"]["codes"] == ["injection"]
    result = box.call("weight_edit_audit", {"target": "gold", "n_seeds": 1, "steering": False})
    assert result.findings["target"] == "gold" and result.findings["n_control_draws"] == 1


def test_audit_on_a_real_architecture():
    pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    from evalrx.wsp import write_sites
    from tests.tiny_whitebox import TinyVLM

    model = TinyVLM("qwen2_5_vl")
    snap = snapshot(model)
    site = write_sites(model)[3]
    torch.manual_seed(0)
    edit = WeightEdit([TensorEdit.rank_one(site.name, torch.randn(32), 6.0, cols=[1, 5, 9])])
    batch = CaseBatch([
        FailureCase(inputs=Inputs(prompt=f"what is {i}"), expected="b",
                    label=Label.FAIL if i % 2 else Label.PASS)
        for i in range(4)
    ])
    f = WeightEditAudit(edit=edit, n_seeds=1).run(model, batch).findings
    json.dumps(f)
    assert f["n_cases_analyzed"] == 4
    assert f["structure"][0]["alpha_estimate"] == pytest.approx(6.0, abs=1e-2)
    assert f["mean_kl_edit"] > 0
    assert "steer_recovery" in f
    assert unchanged(model, snap)
