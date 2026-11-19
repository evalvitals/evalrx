"""The probe catalog: every analyzer described as a tool, with honest availability."""

from __future__ import annotations

import json

import pytest

import evalrx  # noqa: F401 - registers the analyzers
from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.registry import registry
from evalrx.core.result import Result
from evalrx.core.tool import Tool
from evalrx.probing import (
    FAMILY_QUESTIONS,
    availability,
    describe_probes,
    family_of,
    probe_specs,
    spec_for,
)
from evalrx.probing.catalog import _annotation_type, _params_of, is_stub
from tests.conftest import FakeModel


def _text_batch(n: int = 4) -> CaseBatch:
    return CaseBatch([
        FailureCase(inputs=Inputs(prompt=f"question {i}"),
                    label=Label.FAIL if i % 2 else Label.PASS)
        for i in range(n)
    ])


def test_every_registered_analyzer_has_a_spec():
    specs = probe_specs()
    assert set(specs) == set(registry.analyzers.list())
    for name, spec in specs.items():
        assert spec.name == name
        assert spec.summary, name
        assert spec.question, name
        assert spec.access in {"black_box", "white_box", "data_only"}, name
        assert spec.cost in {"light", "moderate", "heavy"}, name


def test_every_family_is_documented():
    families = {spec.family for spec in probe_specs().values()}
    assert families <= set(FAMILY_QUESTIONS), families - set(FAMILY_QUESTIONS)


def test_family_comes_from_the_package_with_renames():
    assert family_of(registry.analyzers.get("logit_lens")) == "lens"
    assert family_of(registry.analyzers.get("attention_knockout")) == "causal"
    assert family_of(registry.analyzers.get("weight_edit_audit")) == "weight_edit"

    class Outside(Analyzer):
        def _run(self, model, cases):  # pragma: no cover - never run
            return Result(analyzer="x", model="m")

    assert family_of(Outside) == "custom"
    Outside.probe_family = "lens"
    assert family_of(Outside) == "lens"


def test_access_follows_required_capabilities():
    specs = probe_specs()
    assert specs["self_consistency"].access == "black_box"
    assert specs["logit_lens"].access == "white_box"
    assert specs["loop_detect"].access == "data_only"
    assert specs["attention_knockout"].intervenes is True
    assert specs["logit_lens"].intervenes is False


def test_schemas_are_valid_json_and_reject_extra_arguments():
    for name, spec in probe_specs().items():
        schema = spec.parameters_schema()
        json.dumps(schema)  # every default must be JSON-serialisable
        assert schema["type"] == "object", name
        assert schema["additionalProperties"] is False, name
        for param in spec.callable_params:
            assert not param.injected, (name, param.name)


def test_host_only_arguments_stay_out_of_the_schema():
    spec = spec_for("first_error_judge")
    assert "judge" in {p.name for p in spec.params}
    assert "judge" not in spec.parameters_schema()["properties"]
    assert spec.injected_params == ()  # optional: the probe can run without it

    required = spec_for("trajectory_rubric")
    assert required.injected_params == ("judge",)


@pytest.mark.parametrize("annotation,expected", [
    ("int", ("integer", False, True)),
    ("float | None", ("number", True, True)),
    ("Optional[int]", ("integer", True, True)),
    ("dict[str, int | None] | None", ("object", True, True)),
    ("tuple[str, ...]", ("array", False, True)),
    ("int | float", ("number", False, True)),
    ("Optional[Callable[[Any], str]]", (None, True, True)),
    ("'Model | None'", (None, True, True)),
    ("Any", (None, True, False)),
])
def test_annotation_parsing(annotation, expected):
    assert _annotation_type(annotation) == expected


def test_parameter_types_fall_back_to_the_default_value():
    class Probe(Analyzer):
        def __init__(self, k=3, ratio=0.5, flag=False, tag="x", fn=len):
            super().__init__(k=k, ratio=ratio, flag=flag, tag=tag, fn=fn)

        def _run(self, model, cases):  # pragma: no cover
            return Result(analyzer="x", model="m")

    params = {p.name: p for p in _params_of(Probe)}
    assert params["k"].json_type == "integer"
    assert params["ratio"].json_type == "number"
    assert params["flag"].json_type == "boolean"
    assert params["tag"].json_type == "string"
    assert params["fn"].injected is True


def test_to_tool_produces_an_openai_schema():
    tool = spec_for("logit_lens").to_tool()
    assert isinstance(tool, Tool)
    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "logit_lens"
    assert schema["function"]["parameters"]["type"] == "object"


def test_failure_modes_are_the_inverse_of_the_routing_table():
    from evalrx.eval_agent.stages.probe import _FAILURE_MODE_TO_ANALYZERS

    specs = probe_specs()
    for mode, names in _FAILURE_MODE_TO_ANALYZERS.items():
        for name in names:
            if name in specs:
                assert mode in specs[name].failure_modes


def test_placeholders_are_reported_as_not_implemented():
    stubs = [name for name, spec in probe_specs().items() if not spec.implemented]
    for name in stubs:
        assert is_stub(registry.analyzers.get(name))
    model = FakeModel(capabilities=set(Capability))
    for name in stubs:
        verdict = availability(spec_for(name), model)
        assert verdict.available is False
        assert "not_implemented" in verdict.codes


def test_missing_capability_names_what_is_missing():
    model = FakeModel(capabilities={Capability.GENERATE})
    verdict = availability(spec_for("logit_lens"), model, _text_batch())
    assert verdict.available is False
    assert verdict.codes == ("capability",)
    assert "hidden_states" in verdict.missing[0]


def test_all_reasons_are_reported_not_just_the_first():
    model = FakeModel(capabilities={Capability.GENERATE})
    verdict = availability(spec_for("attention_knockout"), model, _text_batch())
    assert {"capability", "modality", "data_slot"} <= set(verdict.codes)
    assert len(verdict.missing) == len(verdict.codes)


def test_trajectory_probes_need_trajectories_in_the_batch():
    model = FakeModel(capabilities={Capability.GENERATE})
    verdict = availability(spec_for("loop_detect"), model, _text_batch())
    assert verdict.codes == ("trajectories",)
    # Without a batch there is no evidence either way: nothing is assumed missing.
    assert availability(spec_for("loop_detect"), model).available is True


def test_injected_argument_is_satisfied_by_an_override():
    model = FakeModel(capabilities={Capability.GENERATE})
    spec = spec_for("trajectory_rubric")
    assert "injection" in availability(spec, model).codes
    assert "injection" not in availability(spec, model, overrides=["trajectory_rubric"]).codes


def test_describe_probes_lists_everything_with_a_verdict():
    model = FakeModel(capabilities={Capability.GENERATE})
    report = describe_probes(model, _text_batch())
    assert set(report) == set(registry.analyzers.list())
    for name, row in report.items():
        assert isinstance(row["available"], bool)
        assert row["available"] == (not row["missing"]), name
        assert len(row["missing"]) == len(row["codes"]), name
    json.dumps(report)
    assert report["self_consistency"]["available"] is True
    assert report["logit_lens"]["available"] is False


def test_describe_probes_filters_by_family():
    report = describe_probes(FakeModel(), families=["lens"])
    assert report and all(row["family"] == "lens" for row in report.values())


def test_describe_runs_no_model_call():
    class Counting(FakeModel):
        calls = 0

        def generate(self, inputs, **kwargs):
            type(self).calls += 1
            return "x"

        def forward(self, inputs, capture, spec=None):
            type(self).calls += 1
            return super().forward(inputs, capture, spec)

    describe_probes(Counting(capabilities=set(Capability)), _text_batch())
    assert Counting.calls == 0
