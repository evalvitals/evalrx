"""End-to-end smoke test for AgenticDiagnoseLoop — the judge-decided mode.

Mirrors test_agent_launch_smoke.py's fixtures (a synthetic attention signal
that discriminates FAIL/PASS perfectly), but drives the loop via a scripted
decision judge instead of a fixed M1->M2->M3->M4 cycle, and additionally
proves the host-enforced pre-registration gate: declaring success before a
hypothesis is actually tested is rejected, not honored.
"""

from __future__ import annotations

import json

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.result import Result
from evalrx.eval_agent import AgenticDiagnoseLoop, DiagnosisAgent, HypothesisTester
from evalrx.eval_agent.log_schema import iter_log_errors
from evalrx.eval_agent.run_context import RunContext
from evalrx.eval_agent.stages.probe_agent import ProbeAgent
from evalrx.eval_agent.stages.protocol import ExperimentProtocol
from tests.conftest import FakeModel


class _ScriptedM3Judge(FakeModel):
    """DiagnosisAgent's own judge (M3 propose + adversarial review) — same
    script as test_agent_launch_smoke.py's _ScriptedJudge."""

    def __init__(self) -> None:
        super().__init__(capabilities={Capability.GENERATE})
        self.calls: list[str] = []

    def generate(self, inputs, **kwargs) -> str:
        prompt = str(inputs)
        self.calls.append(prompt)
        if "adversarial ML reviewer" in prompt:
            return (
                "KEEP: Failure is driven by an attention signal present on failing cases.\n"
                "REASON: evidence directly supports this claim"
            )
        return json.dumps([{
            "hypothesis": "Failure is driven by an attention signal present on failing cases.",
            "failure_mode": "attention",
        }])


class _ScriptedActionJudge(FakeModel):
    """The agentic loop's own decision judge: a fixed, valid action sequence."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(capabilities={Capability.GENERATE})
        self.calls: list[str] = []
        self._responses = iter(responses)

    def generate(self, inputs, **kwargs) -> str:
        self.calls.append(str(inputs))
        return next(self._responses)


_HAPPY_PATH = [
    '{"tool": "run_probe", "params": {}, "rationale": "start with M1"}',
    '{"tool": "run_stats", "params": {"confirmatory": true}, "rationale": "quantify effect"}',
    '{"tool": "propose_hypotheses", "params": {}, "rationale": "generate hypotheses"}',
    '{"tool": "test_hypothesis", "params": {}, "rationale": "test it"}',
    '{"tool": "stop", "params": {"resolved": true, "reason": "supported"}, "rationale": "done"}',
]

_PREMATURE_STOP_THEN_HAPPY_PATH = [
    '{"tool": "stop", "params": {"resolved": true, "reason": "looks done"}, "rationale": "early"}',
    *_HAPPY_PATH,
]


class _SignalProbe(ProbeAgent):
    """M1 probe with a stable per-case signal exactly on failing examples."""

    def probe(self, model, data, **kwargs):
        fail_ids = [case.id for case in data if case.label == Label.FAIL]
        return {
            "attention": Result(
                analyzer="attention", model=repr(model), cases=data,
                findings={
                    "mean_attention_flag": 1.0,
                    "per_case": [
                        {"sample_id": case_id, "attention_flag": True} for case_id in fail_ids
                    ],
                },
            )
        }


def _cases() -> CaseBatch:
    return CaseBatch([
        FailureCase(inputs=Inputs(prompt="q_fail_1"), label=Label.FAIL),
        FailureCase(inputs=Inputs(prompt="q_fail_2"), label=Label.FAIL),
        FailureCase(inputs=Inputs(prompt="q_pass_1"), label=Label.PASS),
        FailureCase(inputs=Inputs(prompt="q_pass_2"), label=Label.PASS),
    ])


def _protocol() -> ExperimentProtocol:
    return ExperimentProtocol(
        description="Smoke test: failures correlate with a synthetic attention signal.",
        task_domain="agentic pipeline smoke test",
        failure_patterns="attention signal appears on failing examples only",
        target_modalities=frozenset({"text"}),
    )


def _build_loop(tmp_path, action_judge, *, max_actions=10):
    ctx = RunContext(tmp_path / "run", verbose=False)
    loop = AgenticDiagnoseLoop(
        model=FakeModel(
            capabilities={Capability.GENERATE, Capability.ATTENTION}, modalities={"text"},
        ),
        protocol=_protocol(),
        judge=action_judge,
        probe_agent=_SignalProbe(max_analyzers=1),
        diagnosis_agent=DiagnosisAgent(judge=_ScriptedM3Judge()),
        hypothesis_tester=HypothesisTester(min_effect=0.05),
        max_actions=max_actions,
        run_logger=ctx.logger,
    )
    return ctx, loop


def test_agentic_loop_reaches_a_supported_hypothesis_via_scripted_actions(tmp_path):
    action_judge = _ScriptedActionJudge(_HAPPY_PATH)
    ctx, loop = _build_loop(tmp_path, action_judge)

    report = loop.run(_cases())

    assert report.stopped_by == "agent_stop"
    assert len(action_judge.calls) == 5  # exactly the scripted sequence, no repairs
    assert [h.predicted_failure_mode for h in report.all_hypotheses] == ["attention"]
    assert len(report.all_test_results) == 1
    assert report.all_test_results[0].status.value == "supported"
    assert report.all_test_results[0].effect_size == 1.0
    assert len(report.verified_hypotheses) == 1

    # The two new agentic events (agent_decision/agent_tool) were logged and
    # validate against the published run_log schema, alongside the reused
    # probe/analysis/diagnosis/surgery events from the wrapped M1-M4 stages,
    # bracketed by run_start/loop_end so the dashboard can show run provenance.
    errors = list(iter_log_errors(ctx.log_path))
    assert errors == []
    lines = [json.loads(line) for line in ctx.log_path.read_text().splitlines()]
    events = {line["event"] for line in lines}
    assert {
        "run_start", "loop_end", "agent_decision", "agent_tool",
        "probe", "analysis", "diagnosis", "surgery",
    } <= events

    # run_start distinguishes the agentic decision judge from the M3 stage judge
    # (both are ClaudeModel repr strings in general, but only one field name is
    # right for each — the dashboard header needs both).
    run_start = next(line for line in lines if line["event"] == "run_start")
    assert run_start["decision_judge"] == repr(action_judge)
    assert run_start["max_actions"] == 10


def test_agentic_loop_wires_cluster_failures_into_the_m3_prompt(tmp_path):
    """cluster_failures runs before propose_hypotheses and its output reaches
    the M3 prompt via DiagnosisAgent's failure_modes= param (Phase 3 wiring)."""
    sequence = [
        '{"tool": "run_probe", "params": {}, "rationale": "start with M1"}',
        '{"tool": "cluster_failures", "params": {}, "rationale": "look for patterns in FAIL cases"}',
        '{"tool": "run_stats", "params": {"confirmatory": true}, "rationale": "quantify effect"}',
        '{"tool": "propose_hypotheses", "params": {}, "rationale": "generate hypotheses"}',
        '{"tool": "test_hypothesis", "params": {}, "rationale": "test it"}',
        '{"tool": "stop", "params": {"resolved": true, "reason": "supported"}, "rationale": "done"}',
    ]
    action_judge = _ScriptedActionJudge(sequence)
    m3_judge = _ScriptedM3Judge()
    ctx = RunContext(tmp_path / "run", verbose=False)
    loop = AgenticDiagnoseLoop(
        model=FakeModel(capabilities={Capability.GENERATE, Capability.ATTENTION}, modalities={"text"}),
        protocol=_protocol(),
        judge=action_judge,
        probe_agent=_SignalProbe(max_analyzers=1),
        diagnosis_agent=DiagnosisAgent(judge=m3_judge),
        hypothesis_tester=HypothesisTester(min_effect=0.05),
        max_actions=10,
        run_logger=ctx.logger,
    )

    report = loop.run(_cases())

    assert report.stopped_by == "agent_stop"
    assert len(action_judge.calls) == 6
    # m3_judge.calls[0] is the propose call — its prompt must carry the
    # clustered failure modes (even the tiny 2-FAIL-case batch here yields the
    # single_cluster fallback, which is still a real, non-empty cluster).
    assert "FAILURE MODES" in m3_judge.calls[0]

    # The cluster_failures tool persists its report so the dashboard can render
    # it without re-deriving it from the log (dispatch payload isn't logged).
    fm_path = ctx.root / "artifacts" / "failure_modes.json"
    assert fm_path.exists()
    saved = json.loads(fm_path.read_text())
    assert saved["clusters"]


def test_agentic_loop_rejects_premature_stop_and_continues_to_a_real_verdict(tmp_path):
    """stop(resolved=true) before test_hypothesis must be rejected, not honored —
    the loop keeps going and reaches the same verdict as the happy path."""
    action_judge = _ScriptedActionJudge(_PREMATURE_STOP_THEN_HAPPY_PATH)
    ctx, loop = _build_loop(tmp_path, action_judge, max_actions=10)

    report = loop.run(_cases())

    assert report.stopped_by == "agent_stop"
    assert len(action_judge.calls) == 6  # the rejected early stop + the full happy path
    assert len(report.verified_hypotheses) == 1
