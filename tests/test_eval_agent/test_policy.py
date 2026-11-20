"""The loop owns the run bracket; a policy only decides the order of the stages.

Also: a run that a policy stopped on M3 shows M3 as failed in the report.
"""

from __future__ import annotations

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.eval_agent import AutoDiagnoseReport, FixedOrderPolicy, JudgePolicy, VLDiagnoseLoop
from evalrx.eval_agent.run_context import RunContext
from evalrx.eval_agent.stages.protocol import ExperimentProtocol
from evalrx.reporting.run_events import read_run_events
from tests.conftest import FakeModel


def _loop(tmp_path, **kwargs) -> tuple[RunContext, VLDiagnoseLoop]:
    ctx = RunContext(tmp_path / "run", verbose=False)
    loop = VLDiagnoseLoop(
        model=FakeModel(capabilities={Capability.GENERATE}),
        protocol=ExperimentProtocol(description="policy test", target_modalities=frozenset({"text"})),
        run_logger=ctx.logger,
        **kwargs,
    )
    return ctx, loop


def test_the_fixed_order_is_the_default_policy(tmp_path):
    _, loop = _loop(tmp_path)
    assert isinstance(loop.policy, FixedOrderPolicy)
    _, judged = _loop(tmp_path, policy=JudgePolicy(FakeModel()))
    assert isinstance(judged.policy, JudgePolicy)


def test_a_custom_policy_gets_the_loop_and_the_run_bracket(tmp_path):
    seen = {}

    class OneStep:
        def run(self, loop, data, timings):
            seen["tokens_reset"] = loop._tokens_used == 0
            timings["m1"] = 1.5
            return AutoDiagnoseReport(cycles=1, stopped_by="custom", store=loop.store)

    ctx, loop = _loop(tmp_path, policy=OneStep())
    loop._tokens_used = 99
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="q"), label=Label.FAIL)])

    report = loop.run(data)

    assert report.stopped_by == "custom" and seen["tokens_reset"]
    ctx.finalize()
    (end,) = [e for e in read_run_events(ctx.root) if e["event"] == "loop_end"]
    assert end["stopped_by"] == "custom"


def test_a_run_that_stopped_on_m3_shows_m3_as_failed_in_the_report(tmp_path):
    from evalrx.reporting.dynamic import build_report_data

    class Unreadable:
        def diagnose(self, analysis, **kwargs):
            raise RuntimeError("the judge's answer could not be read")

    ctx, loop = _loop(tmp_path, diagnosis_agent=Unreadable())
    data = CaseBatch([
        FailureCase(inputs=Inputs(prompt=f"q{i}"), label=Label.FAIL if i % 2 else Label.PASS)
        for i in range(4)
    ])
    assert loop.run(data).stopped_by == "diagnosis_error"
    ctx.finalize()

    report = build_report_data(ctx.root)
    m3 = next(s for s in report["stages"] if s["id"] == "m3")
    assert m3["status"] == "failed"
    assert "could not be read" in report["stage_detail"]["m3"]["error"]


def test_a_later_m3_failure_keeps_the_earlier_hypotheses_in_the_report(tmp_path):
    from evalrx.eval_agent import DiagnosisResult, Hypothesis
    from evalrx.reporting.dynamic import build_report_data

    class FailsSecond:
        calls = 0

        def diagnose(self, analysis, **kwargs):
            FailsSecond.calls += 1
            if FailsSecond.calls > 1:
                raise RuntimeError("the judge's answer could not be read")
            h = Hypothesis(statement="The model guesses without looking.", target_model="m",
                           predicted_failure_mode="no_such_signal")
            return DiagnosisResult(model_name="m", hypotheses=[h], raw_judge_output="{}")

    ctx, loop = _loop(tmp_path, diagnosis_agent=FailsSecond(), max_cycles=2)
    data = CaseBatch([
        FailureCase(inputs=Inputs(prompt=f"q{i}"), label=Label.FAIL if i % 2 else Label.PASS)
        for i in range(4)
    ])
    report = loop.run(data)
    assert report.stopped_by == "diagnosis_error" and len(report.final_hypotheses) == 1
    ctx.finalize()

    data = build_report_data(ctx.root)
    assert next(s for s in data["stages"] if s["id"] == "m3")["status"] == "failed"
    detail = data["stage_detail"]["m3"]
    assert detail["error"] and len(detail["hypotheses"]) == 1
