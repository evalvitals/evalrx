"""L4 weight-edit repair: a training-free rank-one edit as a validated fix candidate.

Run against a model with one planted defect (``tests/toy_wsp.py``), so every
expectation here is known in advance rather than hoped for: the edit derived
from a disjoint diagnosis pool repairs the held-out failures, breaks nothing,
is validated by the same paired test as every other tier, and is recorded
beside a matched random-direction control that does not repair them.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from evalrx.core.capability import Capability  # noqa: E402
from evalrx.core.case import CaseBatch, Label  # noqa: E402
from evalrx.eval_agent.hypothesis import Hypothesis  # noqa: E402
from evalrx.eval_agent.stages.fix.agent import FixAgent  # noqa: E402
from evalrx.eval_agent.stages.fix.tiers import FixTier  # noqa: E402
from evalrx.eval_agent.stages.fix.weight_edit import (  # noqa: E402
    CANDIDATE_KIND,
    CANDIDATE_NAME,
    WeightEditSpec,
    run_weight_edit_repair,
    weight_edit_available,
)
from evalrx.wsp import TensorEdit, WeightEdit, contrast_edit  # noqa: E402
from tests.conftest import FakeModel  # noqa: E402
from tests.toy_wsp import (  # noqa: E402
    DEFECT_LAYER,
    ToyModel,
    echo_batch,
    exact_score,
    snapshot,
    unchanged,
)

_ONE = {"n_layers": 1, "n_columns": 1}


def _hyp() -> Hypothesis:
    return Hypothesis(statement="one neuron writes the wrong answer on trigger inputs",
                      target_model="toy", predicted_failure_mode="prior", test_design="")


def _weight_edit(out):
    found = [v for v in out.attempted if v.candidate.kind == CANDIDATE_KIND]
    assert len(found) <= 1
    return found[0] if found else None


# ----------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------

def test_executor_repairs_held_out_cases_and_restores_the_model():
    model = ToyModel()
    snap = snapshot(model)
    pool, val = echo_batch(4, 6), echo_batch(6, 8)
    result = run_weight_edit_repair(model, pool, val, dict(_ONE), exact_score)
    assert result.ok is True and result.error == ""
    assert set(result.scores) == {c.id for c in val}
    assert all(result.scores.values())                       # every held-out case passes
    assert set(result.outputs) == {c.id for c in val}
    assert unchanged(model, snap)
    json.dumps(result.audit)


def test_audit_records_what_was_applied_and_the_control():
    model = ToyModel()
    pool, val = echo_batch(4, 6), echo_batch(6, 8)
    audit = run_weight_edit_repair(model, pool, val, dict(_ONE), exact_score).audit
    assert audit["edit"]["tensors"][0]["name"] == f"layers.{DEFECT_LAYER}.mlp.down_proj.weight"
    assert audit["structure"][0]["changed_columns"] == 1
    assert audit["applied"]["max_relative_change"] > 0
    assert audit["edit_result"] == {"n_pass": 14, "n_scored": 14}
    # The matched random direction leaves the six failures failing.
    assert audit["control_random_direction"] == {"n_pass": 8, "n_scored": 14}
    assert audit["settings"]["alpha"] == 1.0


def test_control_can_be_switched_off():
    model = ToyModel()
    before = model.n_generate
    audit = run_weight_edit_repair(model, echo_batch(4, 6), echo_batch(2, 2),
                                   {**_ONE, "control": False}, exact_score).audit
    assert "control_random_direction" not in audit
    assert model.n_generate - before == 4  # one pass over the batch, not two


@pytest.mark.parametrize("pool_factory,payload,message", [
    (lambda val: None, {}, "no diagnosis pool"),
    (lambda val: CaseBatch([]), {}, "no diagnosis pool"),
    (lambda val: val, {}, "must come from disjoint cases"),
    (lambda val: CaseBatch([c for c in echo_batch(4, 6) if c.label == Label.FAIL]), {},
     "both failing and passing"),
    (lambda val: echo_batch(4, 6), {"n_columns": 0}, "invalid weight-edit settings"),
    (lambda val: echo_batch(4, 6), {"alpha": "strong"}, "invalid weight-edit settings"),
    (lambda val: echo_batch(4, 6), {"layers": [99]}, "could not build the edit"),
    (lambda val: echo_batch(4, 6), {"max_relative_change": 1e-9}, "could not be applied"),
    (lambda val: echo_batch(4, 6), {"edit_dir": "/no/such/dir"}, "could not load edit"),
])
def test_refusals_explain_themselves_and_touch_nothing(pool_factory, payload, message):
    model = ToyModel()
    snap = snapshot(model)
    val = echo_batch(3, 3)
    result = run_weight_edit_repair(model, pool_factory(val), val, payload, exact_score)
    assert result.ok is False
    assert message in result.error
    assert result.scores == {}
    assert model.n_generate == 0
    assert unchanged(model, snap)


def test_models_without_editable_weights_are_refused():
    api_like = FakeModel(capabilities={Capability.GENERATE})
    assert weight_edit_available(api_like) is False
    result = run_weight_edit_repair(api_like, echo_batch(2, 2), echo_batch(2, 2), {}, exact_score)
    assert result.ok is False and "white-box" in result.error
    assert weight_edit_available(ToyModel()) is True


def test_a_failing_generate_scores_none_and_the_model_is_still_restored():
    model = ToyModel()
    snap = snapshot(model)
    val = echo_batch(3, 3)
    doomed = list(val)[1].inputs.prompt
    original = model.generate

    def flaky(inputs, **kwargs):
        if inputs.prompt == doomed:
            raise RuntimeError("generation failed")
        return original(inputs, **kwargs)

    model.generate = flaky
    result = run_weight_edit_repair(model, echo_batch(4, 6), val, {**_ONE, "control": False},
                                    exact_score)
    assert result.ok is True
    assert list(result.scores.values()).count(None) == 1
    assert unchanged(model, snap)


def test_a_saved_edit_is_applied_as_written_without_a_pool(tmp_path):
    model = ToyModel()
    pool = echo_batch(4, 6)
    edit = contrast_edit(model, [c for c in pool if c.label == Label.FAIL],
                         [c for c in pool if c.label == Label.PASS], n_layers=1, n_columns=1)
    directory = edit.save(str(tmp_path / "edit"))
    val = echo_batch(5, 5)
    result = run_weight_edit_repair(model, None, val, {"edit_dir": directory}, exact_score)
    assert result.ok is True and all(result.scores.values())
    assert result.audit["edit"]["name"] == edit.name


def test_a_saved_edit_with_a_stale_base_hash_is_refused(tmp_path):
    model = ToyModel()
    snap = snapshot(model)
    stale = WeightEdit([TensorEdit.rank_one(
        f"layers.{DEFECT_LAYER}.mlp.down_proj.weight", [0, 0, 0, 0, 0, 0, 0, 1.0], 1.0)])
    stale.tensors[0].base_sha256 = "0" * 64
    directory = stale.save(str(tmp_path / "stale"))
    result = run_weight_edit_repair(model, None, echo_batch(2, 2),
                                    {"edit_dir": directory, "verify_base": True}, exact_score)
    assert result.ok is False and "differs from the one the edit was made on" in result.error
    assert unchanged(model, snap)


def test_settings_ignore_unknown_payload_keys():
    spec = WeightEditSpec.from_payload({"alpha": 3, "exec_error": "x", "audit": {}, "layers": (1,)})
    assert spec.alpha == 3.0 and spec.layers == [1]
    assert "exec_error" not in spec.to_dict()


# ----------------------------------------------------------------------
# Through the fix agent
# ----------------------------------------------------------------------

def test_fix_agent_validates_the_edit_with_the_shared_paired_test():
    model = ToyModel()
    snap = snapshot(model)
    agent = FixAgent(judge=None, max_tier="L4", allow_codegen=False, score_fn=exact_score,
                     finetune_pool=echo_batch(4, 6), weight_edit=dict(_ONE))
    out = agent.propose_and_validate(model, echo_batch(16, 10), [_hyp()])

    v = _weight_edit(out)
    assert v is not None
    assert v.candidate.tier is FixTier.L4_PARAMETERS and v.candidate.name == CANDIDATE_NAME
    assert v.exec_error == ""
    assert v.n_fixed == 16 and v.n_broken == 0
    assert v.verdict == "fixed" and v.e_value > v.e_threshold
    assert len(v.outputs) == 26                       # what it answered is kept per case
    assert out.fixed is True and out.best.candidate.name == CANDIDATE_NAME
    audit = v.candidate.payload["audit"]
    assert audit["edit_result"]["n_pass"] == 26
    assert audit["control_random_direction"]["n_pass"] == 10
    assert unchanged(model, snap)


def test_weight_edit_is_opt_in():
    agent = FixAgent(judge=None, max_tier="L4", allow_codegen=False, score_fn=exact_score,
                     finetune_pool=echo_batch(4, 6))
    out = agent.propose_and_validate(ToyModel(), echo_batch(8, 4), [_hyp()])
    assert _weight_edit(out) is None


def test_weight_edit_is_only_offered_at_the_parameter_tier():
    agent = FixAgent(judge=None, max_tier="L3b", allow_codegen=False, score_fn=exact_score,
                     finetune_pool=echo_batch(4, 6), weight_edit=True)
    out = agent.propose_and_validate(ToyModel(), echo_batch(8, 4), [_hyp()])
    assert _weight_edit(out) is None


def test_without_a_pool_the_candidate_is_recorded_not_executed():
    model = ToyModel()
    agent = FixAgent(judge=None, max_tier="L4", allow_codegen=False, score_fn=exact_score,
                     weight_edit=True)
    out = agent.propose_and_validate(model, echo_batch(8, 4), [_hyp()])
    v = _weight_edit(out)
    assert v is not None and v.fixed is False
    assert v.verdict == "not_executed"
    assert "no diagnosis pool" in v.summary


def test_a_pool_that_overlaps_the_validated_batch_is_never_used():
    model = ToyModel()
    snap = snapshot(model)
    batch = echo_batch(8, 4)
    agent = FixAgent(judge=None, max_tier="L4", allow_codegen=False, score_fn=exact_score,
                     finetune_pool=batch, weight_edit=dict(_ONE))
    out = agent.propose_and_validate(model, batch, [_hyp()])
    v = _weight_edit(out)
    assert v.verdict == "not_executed" and "disjoint" in v.summary
    assert out.fixed is False
    assert unchanged(model, snap)


def test_not_offered_for_a_model_without_internals():
    agent = FixAgent(judge=None, max_tier="L4", allow_codegen=False, weight_edit=True,
                     finetune_pool=echo_batch(4, 6))
    out = agent.propose_and_validate(
        FakeModel(capabilities={Capability.GENERATE}), echo_batch(8, 4), [_hyp()])
    assert _weight_edit(out) is None


def test_invalid_agent_settings_propose_nothing():
    agent = FixAgent(judge=None, max_tier="L4", allow_codegen=False, score_fn=exact_score,
                     finetune_pool=echo_batch(4, 6), weight_edit={"n_layers": 0})
    out = agent.propose_and_validate(ToyModel(), echo_batch(8, 4), [_hyp()])
    assert _weight_edit(out) is None


def test_the_candidate_has_a_plain_language_description():
    from evalrx.eval_agent.stages.fix.descriptions import BUILTIN_DESCRIPTIONS

    text = BUILTIN_DESCRIPTIONS[CANDIDATE_NAME]
    assert "without any training" in text and "random" in text


# ----------------------------------------------------------------------
# A real local model handle
# ----------------------------------------------------------------------

def test_executor_on_a_real_local_model_handle_restores_generation():
    """Mechanics on ``HFLocalModel`` over a from-scratch Llama: the edit is built,
    applied during generation and fully removed. Whether it helps is not at
    issue here — the weights are random."""
    pytest.importorskip("transformers")
    from evalrx.core.case import FailureCase, Inputs
    from evalrx.core.spec import ModelSpec
    from evalrx.models.backends.base import RuntimeConfig
    from evalrx.models.backends.hf.model import HFLocalModel
    from tests.test_eval_agent.test_fix_agent import _tiny_llama, _TinyWordTokenizer

    spec = ModelSpec(key="tiny-llama-wsp", family="fake", model_type="fake_llm", hf_repo="")
    model = HFLocalModel(spec, RuntimeConfig(device="cpu", dtype="float32", max_new_tokens=3))
    model._hf = (_tiny_llama(), _TinyWordTokenizer())
    assert weight_edit_available(model) is True

    def case(word, label):
        return FailureCase(inputs=Inputs(prompt=f"classify {word}"), expected="yes", label=label)

    pool = CaseBatch([case("alpha", Label.FAIL), case("alpha", Label.FAIL),
                      case("beta", Label.PASS), case("beta", Label.PASS)])
    val = CaseBatch([case("alpha", Label.FAIL), case("beta", Label.PASS)])
    snap = snapshot(model)
    before = [model.generate(c.inputs) for c in val]

    result = run_weight_edit_repair(
        model, pool, val, {"alpha": 6.0, "n_columns": 8},
        lambda c, out: str(c.expected) in str(out))
    assert result.ok is True, result.error
    assert set(result.scores) == {c.id for c in val}
    names = [t["name"] for t in result.audit["applied"]["tensors"]]
    assert names and all(n.endswith("mlp.down_proj.weight") for n in names)
    assert all(row["alpha_estimate"] == pytest.approx(6.0, abs=1e-2)
               for row in result.audit["structure"])
    assert "n_pass" in result.audit["control_random_direction"]
    assert unchanged(model, snap)
    assert [model.generate(c.inputs) for c in val] == before
