"""Routing a question to probes, and calling a probe by name with JSON arguments."""

from __future__ import annotations

import pytest

import evalrx  # noqa: F401 - registers the analyzers
from evalrx.analyzers.agent.trajectory_rubric import TrajectoryRubricJudge
from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.result import Result
from evalrx.core.tool import ToolResult
from evalrx.eval_agent.stages.probe import StrategyProbe
from evalrx.eval_agent.stages.probe_agent import ProbeAgent
from evalrx.probing import (
    ProbeCallError,
    ProbeRouter,
    ProbeToolbox,
    build_probe,
    check_params,
    spec_for,
    summarize_result,
)
from tests.conftest import FakeModel

_WHITEBOX = {Capability.GENERATE, Capability.ATTENTION, Capability.HIDDEN_STATES,
             Capability.LOGITS, Capability.LOGPROBS}


def _batch(n: int = 6) -> CaseBatch:
    return CaseBatch([
        FailureCase(inputs=Inputs(prompt=f"question {i}"),
                    label=Label.FAIL if i % 2 else Label.PASS)
        for i in range(n)
    ])


# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------

def test_route_selects_only_runnable_probes():
    model = FakeModel(capabilities={Capability.GENERATE})
    route = ProbeRouter().route(model, _batch())
    assert route.selected
    for probe in route.selected:
        assert probe.spec.access != "white_box"


def test_route_order_matches_the_strategy_selector():
    model = FakeModel(capabilities=_WHITEBOX)
    data = _batch()
    route = ProbeRouter().route(model, data)
    expected = StrategyProbe().select(model, data=data)
    runnable = set(route.names)
    assert route.names == [n for n in expected if n in runnable]


def test_failure_modes_lead_and_carry_the_reason():
    model = FakeModel(capabilities=_WHITEBOX)
    route = ProbeRouter().route(model, _batch(), failure_modes=["Overconfidence"])
    assert route.names[:2] == ["verbalized_confidence", "calibration"]
    assert "targets overconfidence" in route.selected[0].reasons[0]


def test_requested_but_unavailable_probes_are_reported_with_reasons():
    model = FakeModel(capabilities={Capability.GENERATE})
    route = ProbeRouter().route(model, _batch(), failure_modes=["logit_lens"])
    assert "logit_lens" in route.unavailable
    assert "hidden_states" in route.unavailable["logit_lens"][0]
    assert "logit_lens" not in route.names


def test_unknown_failure_modes_are_named_not_dropped():
    route = ProbeRouter().route(FakeModel(), _batch(), failure_modes=["no_such_mode"])
    assert route.unknown_modes == ("no_such_mode",)


def test_only_matching_keeps_just_the_targeted_probes():
    model = FakeModel(capabilities=_WHITEBOX)
    route = ProbeRouter().route(model, _batch(), failure_modes=["overconfidence"],
                                only_matching=True)
    assert set(route.names) == {"verbalized_confidence", "calibration"}


def test_filters_by_family_access_and_intervention():
    model = FakeModel(capabilities=_WHITEBOX)
    lens = ProbeRouter().route(model, _batch(), families=["lens"])
    assert lens.selected and all(p.spec.family == "lens" for p in lens.selected)

    black = ProbeRouter().route(model, _batch(), access="black_box")
    assert all(p.spec.access != "white_box" for p in black.selected)

    white = ProbeRouter().route(model, _batch(), access="white_box")
    assert white.selected and all(p.spec.access == "white_box" for p in white.selected)

    passive = ProbeRouter().route(model, _batch(), allow_interventions=False)
    assert all(not p.spec.intervenes for p in passive.selected)


def test_max_probes_caps_the_selection():
    route = ProbeRouter().route(FakeModel(capabilities=_WHITEBOX), _batch(), max_probes=3)
    assert len(route.selected) == 3


def test_prompt_rendering_lists_names_arguments_and_rejections():
    model = FakeModel(capabilities={Capability.GENERATE})
    text = ProbeRouter().route(model, _batch(), failure_modes=["overconfidence", "logit_lens"],
                               max_probes=4).to_prompt()
    assert "verbalized_confidence" in text
    assert "params:" in text
    assert "UNAVAILABLE:" in text and "logit_lens" in text


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------

def test_unknown_arguments_are_rejected_with_the_accepted_list():
    with pytest.raises(ProbeCallError) as err:
        check_params(spec_for("logit_lens"), {"topk": 3})
    assert "accepted" in str(err.value)
    assert err.value.code == "bad_params"


def test_wrong_types_are_rejected():
    with pytest.raises(ProbeCallError, match="expects integer"):
        check_params(spec_for("self_consistency"), {"n": "five"})
    with pytest.raises(ProbeCallError, match="expects integer"):
        check_params(spec_for("self_consistency"), {"n": True})


def test_whole_floats_are_accepted_for_integer_fields():
    cleaned = check_params(spec_for("self_consistency"), {"n": 4.0})
    assert cleaned == {"n": 4} and isinstance(cleaned["n"], int)
    with pytest.raises(ProbeCallError, match="expects integer"):
        check_params(spec_for("self_consistency"), {"n": 4.5})


def test_host_only_argument_in_a_call_is_explained():
    with pytest.raises(ProbeCallError) as err:
        check_params(spec_for("first_error_judge"), {"judge": "gpt"})
    assert "supplied by the host" in str(err.value)


def test_build_probe_constructs_with_the_given_arguments():
    analyzer = build_probe("self_consistency", {"n": 2})
    assert analyzer.name == "self_consistency"
    assert analyzer.get_params()["n"] == 2


def test_build_probe_unknown_name():
    with pytest.raises(ProbeCallError) as err:
        build_probe("no_such_probe")
    assert err.value.code == "unknown_probe"


def test_build_probe_needs_an_override_for_injected_arguments():
    with pytest.raises(ProbeCallError) as err:
        build_probe("trajectory_rubric")
    assert err.value.code == "unavailable"

    judge = FakeModel(capabilities={Capability.GENERATE})
    override = TrajectoryRubricJudge(judge=judge)
    assert build_probe("trajectory_rubric", overrides={"trajectory_rubric": override}) is override

    # Per-call arguments build a NEW instance; the override is not mutated.
    before = dict(override.get_params())
    rebuilt = build_probe("trajectory_rubric", {"max_cases": 2},
                          overrides={"trajectory_rubric": override})
    assert rebuilt is not override
    assert rebuilt.get_params()["max_cases"] == 2
    assert rebuilt.get_params()["judge"] is judge
    assert override.get_params() == before


# ----------------------------------------------------------------------
# Toolbox
# ----------------------------------------------------------------------

def test_call_runs_the_probe_and_returns_a_result():
    box = ProbeToolbox(FakeModel(capabilities=_WHITEBOX), _batch())
    result = box.call("logit_lens", {"top_k": 2})
    assert isinstance(result, Result)
    assert result.analyzer == "logit_lens"


def test_call_refuses_an_unavailable_probe_before_running():
    box = ProbeToolbox(FakeModel(capabilities={Capability.GENERATE}), _batch())
    with pytest.raises(ProbeCallError) as err:
        box.call("logit_lens")
    assert err.value.code == "unavailable"
    assert "hidden_states" in str(err.value)


def test_call_without_data_is_refused():
    with pytest.raises(ProbeCallError):
        ProbeToolbox(FakeModel()).call("attention")


def test_available_matches_describe():
    box = ProbeToolbox(FakeModel(capabilities={Capability.GENERATE}), _batch())
    report = box.describe()
    assert box.available() == sorted(n for n, row in report.items() if row["available"])


def test_tools_are_bound_and_callable():
    box = ProbeToolbox(FakeModel(capabilities=_WHITEBOX), _batch())
    tools = {tool.name: tool for tool in box.tools()}
    assert set(tools) == set(box.available())
    out = tools["logit_lens"].fn(top_k=2)
    assert isinstance(out, ToolResult)
    assert "logit_lens" in out.text
    assert isinstance(out.meta["result"], Result)


def test_tools_can_include_unavailable_probes_on_request():
    box = ProbeToolbox(FakeModel(capabilities={Capability.GENERATE}), _batch())
    names = {tool.name for tool in box.tools(["logit_lens"], only_available=False)}
    assert names == {"logit_lens"}
    assert box.tools(["logit_lens"]) == []


def test_max_cases_caps_what_a_probe_sees():
    seen = {}

    class Counting(FakeModel):
        def generate(self, inputs, **kwargs):
            seen[str(inputs)] = True
            return "x"

    box = ProbeToolbox(Counting(capabilities={Capability.GENERATE}), _batch(6), max_cases=2)
    result = box.call("verbalized_confidence")
    assert len(result.cases) == 2


def test_summarize_result_keeps_scalars_and_counts_rows():
    result = Result(analyzer="a", model="m", findings={
        "mean": 0.5, "_caveat": "hidden", "per_case": [{"x": 1}, {"x": 2}], "nested": {"k": 1},
    })
    text = summarize_result(result)
    assert "'mean': 0.5" in text and "2 per-case rows" in text
    assert "_caveat" not in text and "nested" not in text


# ----------------------------------------------------------------------
# ProbeAgent pass-through
# ----------------------------------------------------------------------

def test_probe_agent_runs_named_probes_with_arguments():
    agent = ProbeAgent(max_workers=1)
    model = FakeModel(capabilities=_WHITEBOX)
    results = agent.probe(model, _batch(), analyzers=["logit_lens"],
                          analyzer_params={"logit_lens": {"top_k": 2}})
    assert set(results) == {"logit_lens"}
    assert "Called by name" in agent.last_schema.rationale


def test_exact_calls_are_neither_capped_nor_backfilled():
    model = FakeModel(capabilities=_WHITEBOX)
    capped = ProbeAgent(max_workers=1, max_analyzers=1)
    both = capped.probe(model, _batch(), analyzers=["logit_lens", "attention"], exact=True)
    assert set(both) == {"logit_lens", "attention"}  # the cap does not apply

    roomy = ProbeAgent(max_workers=1, max_analyzers=3)
    only = roomy.probe(model, _batch(), analyzers=["logit_lens"], exact=True)
    assert set(only) == {"logit_lens"}  # nothing was added to fill the slots

    # An inapplicable named probe is dropped and reported, not replaced.
    none = roomy.probe(model, _batch(), analyzers=["loop_detect"], exact=True)
    assert none == {}
    assert "loop_detect" in roomy.last_schema.rationale


def test_probe_agent_surfaces_a_bad_call():
    agent = ProbeAgent(max_workers=1)
    with pytest.raises(ProbeCallError):
        agent.probe(FakeModel(capabilities=_WHITEBOX), _batch(), analyzers=["logit_lens"],
                    analyzer_params={"logit_lens": {"nope": 1}})


def test_pinned_rerun_replays_the_arguments_of_the_named_call():
    built = []
    agent = ProbeAgent(max_workers=1)
    original = agent._make_analyzer

    def spy(name, params=None):
        built.append((name, params))
        return original(name, params)

    agent._make_analyzer = spy
    model = FakeModel(capabilities=_WHITEBOX)
    agent.probe(model, _batch(), analyzers=["logit_lens"], exact=True,
                analyzer_params={"logit_lens": {"top_k": 2}})
    # The held-out confirmation pins the same names without arguments.
    agent.probe(model, _batch(), analyzers=["logit_lens"])
    assert built == [("logit_lens", {"top_k": 2}), ("logit_lens", {"top_k": 2})]

    # A later named call with no arguments goes back to the defaults.
    agent.probe(model, _batch(), analyzers=["logit_lens"], exact=True)
    agent.probe(model, _batch(), analyzers=["logit_lens"])
    assert built[2:] == [("logit_lens", None), ("logit_lens", None)]
