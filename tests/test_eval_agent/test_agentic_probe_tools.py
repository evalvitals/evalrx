"""The agentic loop's probe tools: list what can run, then call probes by name.

``run_probe`` with no arguments keeps its old meaning (the probe agent selects).
With a name it runs exactly that probe with schema-checked arguments, adds the
result to the evidence gathered so far, and rejects a bad call before any model
work — so a judge can correct itself without spending a probe pass.
"""

from __future__ import annotations

import json

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.eval_agent import DiagnosisAgent, HypothesisTester, JudgePolicy, VLDiagnoseLoop
from evalrx.eval_agent.loop.judge.actions import Action
from evalrx.eval_agent.loop.judge.board import EvidenceBoard
from evalrx.eval_agent.loop.judge.tools import (
    _requested_probes,
    build_default_registry,
)
from evalrx.eval_agent.loop.state import RunState
from evalrx.eval_agent.runlog.context import RunContext
from evalrx.eval_agent.stages.probe_agent import ProbeAgent
from evalrx.eval_agent.stages.protocol import ExperimentProtocol
from tests.conftest import FakeModel

_WHITEBOX = {Capability.GENERATE, Capability.ATTENTION, Capability.HIDDEN_STATES,
             Capability.LOGITS}


class _Judge(FakeModel):
    def __init__(self, responses=()):
        super().__init__(capabilities={Capability.GENERATE})
        self._responses = iter(responses)
        self.calls: list[str] = []

    def generate(self, inputs, **kwargs) -> str:
        self.calls.append(str(inputs))
        return next(self._responses)


class _CountingModel(FakeModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_forward = 0

    def forward(self, inputs, capture, spec=None):
        self.n_forward += 1
        return super().forward(inputs, capture, spec)


def _cases() -> CaseBatch:
    return CaseBatch([
        FailureCase(inputs=Inputs(prompt=f"q{i}"), label=Label.FAIL if i % 2 else Label.PASS)
        for i in range(4)
    ])


def _setup(tmp_path, model=None, judge=None):
    ctx = RunContext(tmp_path / "run", verbose=False)
    judge = judge or _Judge()
    loop = VLDiagnoseLoop(
        model=model or _CountingModel(capabilities=_WHITEBOX, modalities={"text"}),
        protocol=ExperimentProtocol(
            description="probe tool test", task_domain="test",
            failure_patterns="none", target_modalities=frozenset({"text"}),
        ),
        policy=JudgePolicy(judge),
        probe_agent=ProbeAgent(max_workers=1, max_analyzers=2),
        diagnosis_agent=DiagnosisAgent(judge=_Judge()),
        hypothesis_tester=HypothesisTester(min_effect=0.05),
        run_logger=ctx.logger,
    )
    data = _cases()
    state = RunState(original_data=data, data=data, stages=loop.stages(), judge=judge)
    return loop, state, build_default_registry(loop, state), EvidenceBoard()


def _call(registry, board, tool, params=None):
    outcome = registry.dispatch(Action(tool=tool, params=params or {}, rationale="t"), board)
    board.action_log.append({"step": len(board.action_log), "tool": tool,
                             "params": params or {}, "ok": outcome.ok,
                             "summary": outcome.summary})
    return outcome


def test_registry_offers_list_probes_and_run_probe(tmp_path):
    _, _, registry, board = _setup(tmp_path)
    assert {"list_probes", "run_probe"} <= registry.tool_names()
    catalog = registry.catalog_for_prompt(board)
    assert "list_probes" in catalog and "params.name" in catalog


def test_request_parsing_accepts_one_or_many():
    assert _requested_probes({}) == []
    assert _requested_probes({"name": "a", "params": {"k": 1}}) == [("a", {"k": 1})]
    assert _requested_probes({"probes": [{"name": "a"}, "b"]}) == [("a", {}), ("b", {})]
    assert isinstance(_requested_probes({"probes": "a"}), str)
    assert isinstance(_requested_probes({"probes": [{"params": {}}]}), str)
    assert isinstance(_requested_probes({"name": "a", "params": [1]}), str)


def test_list_probes_runs_no_model_and_fills_the_board(tmp_path):
    loop, _, registry, board = _setup(tmp_path)
    outcome = _call(registry, board, "list_probes", {"families": ["lens"], "max_probes": 5})
    assert outcome.ok is True
    assert loop.model.n_forward == 0
    assert "logit_lens" in board.probe_catalog
    assert "PROBE CATALOG" in board.to_prompt(registry)
    assert all(p.spec.family == "lens" for p in outcome.payload.selected)


def test_list_probes_defaults_to_the_current_hypotheses(tmp_path):
    _, _, registry, board = _setup(tmp_path)
    board.hypotheses = [{"statement": "s", "failure_mode": "logit_lens", "status": "proposed"}]
    outcome = _call(registry, board, "list_probes", {"only_matching": True})
    assert outcome.ok is True
    assert "logit_lens" in outcome.payload.names


def test_list_probes_reports_an_empty_route(tmp_path):
    _, _, registry, board = _setup(tmp_path)
    outcome = _call(registry, board, "list_probes", {"families": ["no_such_family"]})
    assert outcome.ok is False and outcome.error == "empty"


def test_named_run_probe_runs_exactly_that_probe(tmp_path):
    _, state, registry, board = _setup(tmp_path)
    outcome = _call(registry, board, "run_probe",
                    {"name": "logit_lens", "params": {"top_k": 2}})
    assert outcome.ok is True, outcome.summary
    assert set(state.probe_results) == {"logit_lens"}
    assert [row["analyzer"] for row in board.probe_findings] == ["logit_lens"]


def test_named_calls_accumulate_evidence(tmp_path):
    _, state, registry, board = _setup(tmp_path)
    assert _call(registry, board, "run_probe", {"name": "logit_lens"}).ok
    assert _call(registry, board, "run_probe", {"probes": [{"name": "attention"}]}).ok
    assert set(state.probe_results) == {"logit_lens", "attention"}
    assert {row["analyzer"] for row in board.probe_findings} == {"logit_lens", "attention"}
    assert state.cycle == 2


def test_bad_arguments_are_rejected_before_any_model_work(tmp_path):
    loop, state, registry, board = _setup(tmp_path)
    outcome = _call(registry, board, "run_probe",
                    {"probes": [{"name": "attention"},
                                {"name": "logit_lens", "params": {"nope": 1}}]})
    assert outcome.ok is False and outcome.error == "bad_params"
    assert "accepted" in outcome.summary
    assert loop.model.n_forward == 0
    assert state.probe_results == {} and state.cycle == 0


def test_unknown_and_unavailable_probes_are_rejected_with_reasons(tmp_path):
    model = _CountingModel(capabilities={Capability.GENERATE}, modalities={"text"})
    _, _, registry, board = _setup(tmp_path, model=model)
    unknown = _call(registry, board, "run_probe", {"name": "no_such_probe"})
    assert unknown.ok is False and unknown.error == "unknown_probe"
    blocked = _call(registry, board, "run_probe", {"name": "logit_lens"})
    assert blocked.ok is False and blocked.error == "unavailable"
    assert "hidden_states" in blocked.summary


def test_run_probe_without_a_name_still_lets_the_agent_select(tmp_path):
    _, state, registry, board = _setup(tmp_path)
    outcome = _call(registry, board, "run_probe")
    assert outcome.ok is True
    assert 1 <= len(state.probe_results) <= 2  # ProbeAgent(max_analyzers=2) chose


def test_stats_precondition_is_met_by_a_named_probe(tmp_path):
    _, _, registry, board = _setup(tmp_path)
    assert "BLOCKED" in [line for line in registry.catalog_for_prompt(board).splitlines()
                         if line.startswith("- run_stats")][0]
    assert _call(registry, board, "run_probe", {"name": "logit_lens"}).ok
    assert "BLOCKED" not in [line for line in registry.catalog_for_prompt(board).splitlines()
                             if line.startswith("- run_stats")][0]


def test_full_loop_drives_list_then_named_probe(tmp_path):
    judge = _Judge([
        json.dumps({"tool": "list_probes", "params": {"families": ["lens"]},
                    "rationale": "see what can run"}),
        json.dumps({"tool": "run_probe", "params": {"name": "logit_lens", "params": {"top_k": 2}},
                    "rationale": "read the layers"}),
        json.dumps({"tool": "stop", "params": {"resolved": False, "reason": "enough"},
                    "rationale": "done"}),
    ])
    loop, _, _, _ = _setup(tmp_path, judge=judge)
    report = loop.run(_cases())
    assert report.stopped_by == "agent_stop"
    assert len(judge.calls) == 3
    # The second decision saw the catalog produced by the first.
    assert "PROBE CATALOG" in judge.calls[1] and "logit_lens" in judge.calls[1]


def test_unnamed_runs_keep_their_own_limit_while_named_calls_continue(tmp_path):
    _, _, registry, board = _setup(tmp_path)
    for _ in range(3):
        assert _call(registry, board, "run_probe").ok
    fourth = _call(registry, board, "run_probe")
    assert fourth.ok is False and fourth.error == "max_calls"
    assert "name the probes" in fourth.summary
    assert _call(registry, board, "run_probe", {"name": "logit_lens"}).ok
