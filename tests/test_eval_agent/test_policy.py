"""The loop owns the run bracket; a policy only decides the order of the stages."""

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
