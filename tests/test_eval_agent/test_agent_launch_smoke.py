from __future__ import annotations

import json

from evalrx.analysis.dashboard import load_run
from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.result import Result
from evalrx.eval_agent import (
    DiagnosisAgent,
    HypothesisTester,
    ProbeAgent,
    RunContext,
    VLDiagnoseLoop,
)
from evalrx.eval_agent.stages.protocol import ExperimentProtocol
from tests.conftest import FakeModel


class _ScriptedJudge(FakeModel):
    """Deterministic M3 judge for the launch smoke test."""

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
        return json.dumps(
            [
                {
                    "hypothesis": (
                        "Failure is driven by an attention signal present on failing cases."
                    ),
                    # Present because the prompt asks for it: a missing or
                    # jargon-y plain line costs one extra plain-language
                    # repair call, which would change the call count below.
                    "plain_statement": (
                        "The model looks at the wrong part of the input on the cases it gets wrong."
                    ),
                    "failure_mode": "attention",
                }
            ]
        )


class _SignalProbe(ProbeAgent):
    """M1 probe with a stable per-case signal exactly on failing examples."""

    def probe(self, model, data, **kwargs):
        fail_ids = [case.id for case in data if case.label == Label.FAIL]
        return {
            "attention": Result(
                analyzer="attention",
                model=repr(model),
                cases=data,
                findings={
                    "mean_attention_flag": 1.0,
                    "per_case": [
                        {"sample_id": case_id, "attention_flag": True}
                        for case_id in fail_ids
                    ],
                },
            )
        }


def _cases() -> CaseBatch:
    return CaseBatch(
        [
            FailureCase(inputs=Inputs(prompt="q_fail_1"), label=Label.FAIL),
            FailureCase(inputs=Inputs(prompt="q_fail_2"), label=Label.FAIL),
            FailureCase(inputs=Inputs(prompt="q_pass_1"), label=Label.PASS),
            FailureCase(inputs=Inputs(prompt="q_pass_2"), label=Label.PASS),
        ]
    )


def _protocol() -> ExperimentProtocol:
    return ExperimentProtocol(
        description="Smoke test: failures correlate with a synthetic attention signal.",
        task_domain="agent pipeline smoke test",
        failure_patterns="attention signal appears on failing examples only",
        target_modalities=frozenset({"text"}),
    )


def test_vl_agent_launches_m1_to_m4_and_dashboard_loader_reads_run(tmp_path):
    """Launch the real M1->M2->M3->M4 loop and verify the UI data contract."""
    data = _cases()
    judge = _ScriptedJudge()
    # Pinned: asserts write_diagnose_report's V1-only report/*.json output.
    ctx = RunContext(tmp_path / "run", verbose=False, logger_version="v1")
    loop = VLDiagnoseLoop(
        model=FakeModel(
            capabilities={Capability.GENERATE, Capability.ATTENTION},
            modalities={"text"},
        ),
        protocol=_protocol(),
        probe_agent=_SignalProbe(max_analyzers=1),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        hypothesis_tester=HypothesisTester(min_effect=0.05),
        max_cycles=3,
        run_logger=ctx.logger,
    )

    report = loop.run(data)
    m5_result = loop.run_m5(report, data)
    ctx.write_diagnose_report(report, list(data))

    assert report.stopped_by == "criteria_met"
    assert report.cycles == 1
    assert [h.predicted_failure_mode for h in report.all_hypotheses] == ["attention"]
    assert len(report.all_test_results) == 1
    assert report.all_test_results[0].status.value == "supported"
    assert report.all_test_results[0].effect_size == 1.0
    assert len(report.verified_hypotheses) == 1
    assert m5_result is not None
    assert len(judge.calls) == 2

    loaded = load_run(ctx.root)
    assert loaded["kind"] == "loop"
    story = loaded["story"]
    assert story is not None
    assert len(story["analyses"]) == 1
    assert len(story["diagnoses"]) == 1
    assert len(story["surgeries"]) >= 1
    assert story["diagnoses"][0]["hypotheses"][0]["failure_mode"] == "attention"

    assert (ctx.root / "report" / "summary.json").exists()
    assert (ctx.root / "report" / "hypotheses.json").exists()
    assert (ctx.root / "report" / "m4_results.json").exists()
