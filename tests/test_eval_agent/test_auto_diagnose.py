"""Tests for the AutoDiagnose pipeline.

M1 ProbeAgent, M2 AnalysisModule, M3 DiagnosisAgent, M4 SurgeryAgent,
and the full AutoDiagnoseLoop that ties them together.
"""

from __future__ import annotations

from typing import Any

from evalrx.analysis.analysis_module import AnalysisFinding
from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label, Step, StepRole, Trajectory
from evalrx.core.registry import registry
from evalrx.eval_agent import (
    AnalysisModule,
    AnalysisReport,
    AutoDiagnoseLoop,
    AutoDiagnoseReport,
    DiagnosisAgent,
    DiagnosisResult,
    HypothesisStatus,
    InterventionResult,
    ModelKind,
    ProbeAgent,
    SurgeryAgent,
)
from evalrx.eval_agent.hypothesis import Hypothesis
from tests.conftest import FakeModel

# ── helpers ────────────────────────────────────────────────────────────────────


class ScriptedModel(FakeModel):
    """FakeModel with deterministic generate() responses."""

    def __init__(self, answers: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._answers = answers
        self._i = 0

    def generate(self, inputs, **kwargs) -> str:
        answer = self._answers[self._i % len(self._answers)]
        self._i += 1
        return answer


def _vlm_model() -> FakeModel:
    return FakeModel(
        capabilities={Capability.GENERATE, Capability.ATTENTION},
        modalities={"text", "image"},
    )


def _agent_model() -> FakeModel:
    return FakeModel(capabilities={Capability.GENERATE, Capability.TOOL_CALLS})


def _llm_model() -> FakeModel:
    return FakeModel(
        capabilities={Capability.GENERATE, Capability.ATTENTION, Capability.HIDDEN_STATES}
    )


def _traj_batch(n_fail: int = 1, n_pass: int = 1) -> CaseBatch:
    cases = []
    for i in range(n_fail):
        traj = Trajectory(
            sample_id=f"fail_{i}",
            goal="open file",
            outcome=Label.FAIL,
            steps=[
                Step(idx=0, role=StepRole.USER, content="open file"),
                Step(idx=1, role=StepRole.ACTOR, tool_call={"name": "open", "args": {}}),
                Step(idx=2, role=StepRole.TOOL, observation="Error: denied"),
                Step(idx=3, role=StepRole.ACTOR, tool_call={"name": "open", "args": {}}),
            ],
        )
        cases.append(FailureCase(inputs=Inputs(prompt="open"), trajectory=traj, label=Label.FAIL))
    for i in range(n_pass):
        traj = Trajectory(
            sample_id=f"pass_{i}",
            goal="open file",
            outcome=Label.PASS,
            steps=[
                Step(idx=0, role=StepRole.USER, content="open file"),
                Step(idx=1, role=StepRole.ACTOR, tool_call={"name": "open", "args": {}}),
                Step(idx=2, role=StepRole.TOOL, observation="OK"),
            ],
        )
        cases.append(FailureCase(inputs=Inputs(prompt="open"), trajectory=traj, label=Label.PASS))
    return CaseBatch(cases)


# ══════════════════════════════════════════════════════════════════════════════
# M1 — ProbeAgent
# ══════════════════════════════════════════════════════════════════════════════

def test_probe_agent_detects_llm_kind():
    agent = ProbeAgent()
    assert agent.detect_kind(_llm_model()) == ModelKind.LLM


def test_probe_agent_detects_vlm_kind():
    assert ProbeAgent().detect_kind(_vlm_model()) == ModelKind.VLM


def test_probe_agent_detects_agent_kind():
    assert ProbeAgent().detect_kind(_agent_model()) == ModelKind.AGENT


def test_probe_agent_returns_results_dict():
    model = _llm_model()
    agent = ProbeAgent(max_analyzers=2)
    results = agent.probe(model, CaseBatch([FailureCase(inputs=Inputs(prompt="x"))]))
    assert isinstance(results, dict)
    assert len(results) <= 2
    assert all(name in registry.analyzers.list() for name in results)


def test_probe_agent_only_compatible_analyzers():
    model = _llm_model()
    agent = ProbeAgent()
    results = agent.probe(model, CaseBatch([FailureCase(inputs=Inputs(prompt="x"))]))
    compatible = set(registry.analyzers.names_compatible_with(model))
    assert set(results.keys()) <= compatible


def test_probe_agent_does_not_select_an_analyzer_it_cannot_build(recwarn):
    """It used to be selected and then dropped at instantiation with a warning.

    That spent a selection slot on something that could never run. Selection now
    rejects it up front, so no warning is reached — and, on a single-turn batch,
    `counterfactual` would be rejected on the data shape as well.
    """
    model = FakeModel(capabilities={Capability.GENERATE, Capability.TOOL_CALLS})
    agent = ProbeAgent()
    results = agent.probe(model, CaseBatch([FailureCase(inputs=Inputs(prompt="x"))]))
    assert "counterfactual" not in results
    assert not any("counterfactual" in str(w.message) for w in recwarn.list)


def test_make_analyzer_still_warns_when_called_directly(recwarn):
    """The instantiation guard stays as a backstop for callers that force a name."""
    assert ProbeAgent()._make_analyzer("counterfactual") is None
    assert any("counterfactual" in str(w.message) for w in recwarn.list)


def test_single_turn_batch_is_not_offered_trajectory_analyzers():
    """The bug this gate exists for: analyzers/agent/ declares no capability, so
    every one of them matched a plain QA batch. `first_error_judge` was selected,
    ran, and reported n_trajectories=0 -- which M2 listed as a healthy metric."""
    from evalrx.eval_agent.stages.probe_agent import (
        _analyzer_data_preconditions_met,
    )

    single_turn = CaseBatch([FailureCase(inputs=Inputs(prompt="2+2?"))])
    for name in ("first_error_judge", "ignored_obs", "loop_detect",
                 "trajectory_rubric", "tool_shap", "counterfactual"):
        assert not _analyzer_data_preconditions_met(name, single_turn), name


def test_trajectory_batch_still_admits_them():
    from evalrx.eval_agent.stages.probe_agent import (
        _analyzer_data_preconditions_met,
    )

    traj = _traj_batch(n_fail=1, n_pass=1)
    for name in ("first_error_judge", "loop_detect", "counterfactual"):
        assert _analyzer_data_preconditions_met(name, traj), name


def test_non_agent_analyzers_are_unaffected_by_the_gate():
    from evalrx.eval_agent.stages.probe_agent import (
        _analyzer_data_preconditions_met,
    )

    single_turn = CaseBatch([FailureCase(inputs=Inputs(prompt="2+2?"))])
    for name in ("self_consistency", "format_sensitivity", "arith_audit"):
        assert _analyzer_data_preconditions_met(name, single_turn), name


def test_probe_agent_uses_override():
    from evalrx.analyzers.agent.counterfactual import CounterfactualReplay

    model = FakeModel(capabilities={Capability.GENERATE, Capability.TOOL_CALLS})
    data = _traj_batch(n_fail=1, n_pass=0)
    rerun = CounterfactualReplay(rerun_fn=lambda t, i, s: True, n_replays=1)
    agent = ProbeAgent(analyzer_overrides={"counterfactual": rerun})
    results = agent.probe(model, data)
    assert "counterfactual" in results


def test_probe_agent_priority_ordering():
    model = FakeModel(
        capabilities={
            Capability.GENERATE,
            Capability.ATTENTION,
            Capability.HIDDEN_STATES,
            Capability.LOGITS,
        }
    )
    agent = ProbeAgent()
    results = agent.probe(model, CaseBatch([FailureCase(inputs=Inputs(prompt="x"))]))
    names = list(results.keys())
    # attention should appear before cka in LLM priority order
    if "attention" in names and "cka" in names:
        assert names.index("attention") < names.index("cka")


# ══════════════════════════════════════════════════════════════════════════════
# M2 — AnalysisModule
# ══════════════════════════════════════════════════════════════════════════════

def _fake_results_with_sink() -> dict:
    """AttentionSink result with mean_sink_mass above threshold (0.6)."""
    from evalrx.core.result import Result

    return {
        "attention_sink": Result(
            analyzer="attention_sink",
            model="fake",
            findings={"n_layers": 3, "mean_sink_mass": 0.85, "sink_token": "t0",
                      "per_layer_sink": [0.8, 0.85, 0.9]},
        )
    }


def _fake_results_healthy() -> dict:
    from evalrx.core.result import Result

    return {
        "attention_sink": Result(
            analyzer="attention_sink",
            model="fake",
            findings={"n_layers": 3, "mean_sink_mass": 0.2, "sink_token": "t0",
                      "per_layer_sink": [0.2, 0.2, 0.2]},
        )
    }


def test_analysis_module_flags_high_sink():
    report = AnalysisModule().analyze(_fake_results_with_sink(), "test-model")
    assert isinstance(report, AnalysisReport)
    assert report.severity == "high"
    assert len(report.findings) >= 1
    assert any(f.metric == "mean_sink_mass" for f in report.findings)


def test_analysis_module_clean_model_gives_none_severity():
    report = AnalysisModule().analyze(_fake_results_healthy(), "test-model")
    assert report.severity == "none"
    assert report.findings == []


def test_analysis_module_narrative_contains_model_name():
    report = AnalysisModule().analyze(_fake_results_with_sink(), "MyModel")
    assert "MyModel" in report.narrative


def test_analysis_module_narrative_mentions_finding():
    report = AnalysisModule().analyze(_fake_results_with_sink(), "m")
    assert "attention_sink" in report.narrative or "sink" in report.narrative.lower()


def test_analysis_module_to_dict():
    report = AnalysisModule().analyze(_fake_results_with_sink(), "m")
    d = report.to_dict()
    assert {"model_name", "severity", "n_findings", "findings", "narrative"} <= d.keys()


def test_analysis_module_extra_rules():
    from evalrx.analysis.analysis_module import _Rule
    from evalrx.core.result import Result

    results = {
        "my_analyzer": Result(
            analyzer="my_analyzer", model="m",
            findings={"my_metric": 99.0},
        )
    }
    extra = {"my_analyzer": [_Rule("my_metric", 50.0, "above", "high", "custom rule hit")]}
    report = AnalysisModule(extra_rules=extra).analyze(results, "m")
    assert report.severity == "high"
    assert any(f.metric == "my_metric" for f in report.findings)


def test_analysis_module_sorts_high_severity_first():
    from evalrx.analysis.analysis_module import _Rule
    from evalrx.core.result import Result

    results = {
        "a1": Result(analyzer="a1", model="m", findings={"m1": 10.0}),
        "a2": Result(analyzer="a2", model="m", findings={"m2": 10.0}),
    }
    extra = {
        "a1": [_Rule("m1", 5.0, "above", "low", "low issue")],
        "a2": [_Rule("m2", 5.0, "above", "high", "high issue")],
    }
    report = AnalysisModule(extra_rules=extra).analyze(results, "m")
    assert report.findings[0].severity == "high"


# ══════════════════════════════════════════════════════════════════════════════
# M3 — DiagnosisAgent (takes AnalysisReport)
# ══════════════════════════════════════════════════════════════════════════════

def _make_report(severity="high") -> AnalysisReport:

    f = AnalysisFinding(
        analyzer="attention_sink", metric="mean_sink_mass",
        value=0.85, threshold=0.6, direction="above",
        severity=severity, message="over-attends to sink",
    )
    return AnalysisReport(
        model_name="test-model",
        findings=[f],
        severity=severity,
        narrative="[HIGH] attention_sink.mean_sink_mass=0.85 > 0.6",
        raw_results={},
    )


def test_unparsed_judge_text_falls_back_loudly_but_no_issue_quietly(caplog):
    """gemma-4-e2b/bbh_causal_judgement (2026-08-22): the judge wrote three
    hypotheses in a label format the parser missed and the run silently diagnosed
    an analysis-module template. The fallback stays (M4 still needs a hypothesis)
    but a parse miss must be visible; a genuine NO_ISSUE verdict stays quiet."""
    import logging

    with caplog.at_level(logging.WARNING, logger="evalrx.eval_agent.stages.diagnosis"):
        judge = ScriptedModel(answers=["Three rich paragraphs of diagnosis without any label lines."],
                              capabilities={Capability.GENERATE})
        diag = DiagnosisAgent(judge=judge).diagnose(_make_report())
    assert len(diag.hypotheses) == 1 and diag.hypotheses[0].predicted_failure_mode == "attention_sink"
    assert any("parsed to zero hypotheses" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="evalrx.eval_agent.stages.diagnosis"):
        judge = ScriptedModel(answers=["NO_ISSUE"], capabilities={Capability.GENERATE})
        diag = DiagnosisAgent(judge=judge).diagnose(_make_report())
    assert len(diag.hypotheses) == 1                     # the NO_ISSUE fallback is by design
    assert not any("parsed to zero" in r.message for r in caplog.records)


def test_diagnosis_parses_hypothesis_from_report():
    judge = ScriptedModel(
        answers=["HYPOTHESIS: model over-attends to BOS\nFAILURE_MODE: attention_sink\n"],
        capabilities={Capability.GENERATE},
    )
    diag = DiagnosisAgent(judge=judge).diagnose(_make_report())
    assert isinstance(diag, DiagnosisResult)
    assert len(diag.hypotheses) == 1
    assert diag.hypotheses[0].predicted_failure_mode == "attention_sink"
    assert diag.hypotheses[0].target_model == "test-model"


def test_diagnosis_no_issue_returns_empty():
    judge = ScriptedModel(answers=["NO_ISSUE"], capabilities={Capability.GENERATE})
    diag = DiagnosisAgent(judge=judge).diagnose(_make_report(severity="none"))
    assert diag.hypotheses == []
    assert "NO_ISSUE" in diag.raw_judge_output


def test_diagnosis_backward_compat_accepts_results_dict():
    """Passing a raw results dict (old API) still works via AnalysisModule wrapping."""
    from evalrx.analyzers.attention.summary import AttentionAnalyzer
    model = FakeModel()
    results = {"attention": AttentionAnalyzer().run(model, "probe")}
    judge = ScriptedModel(answers=["NO_ISSUE"], capabilities={Capability.GENERATE})
    diag = DiagnosisAgent(judge=judge).diagnose(results, model_name="m")
    assert isinstance(diag, DiagnosisResult)


def test_diagnosis_prompt_includes_severity_and_narrative():
    captured: list[str] = []

    class CapturingModel(FakeModel):
        def generate(self, inputs, **kw):
            captured.append(str(inputs))
            return "NO_ISSUE"

    DiagnosisAgent(judge=CapturingModel(capabilities={Capability.GENERATE})).diagnose(
        _make_report("high")
    )
    assert captured
    assert "high" in captured[0].lower()
    assert "attention_sink" in captured[0]


# ══════════════════════════════════════════════════════════════════════════════
# M4 — SurgeryAgent (unchanged; smoke-tested here for integration)
# ══════════════════════════════════════════════════════════════════════════════

def _hypothesis(mode: str = "loop") -> Hypothesis:
    return Hypothesis(statement="test", target_model="m", predicted_failure_mode=mode)


def test_surgery_verify_fn_override():
    expected = InterventionResult(
        hypothesis=_hypothesis(),
        status=HypothesisStatus.SUPPORTED,
        fixed=True,
        evidence={"custom": True},
    )
    agent = SurgeryAgent(verify_fn=lambda h, m, r, d: expected)
    result = agent.operate(_hypothesis(), None, {}, CaseBatch([]))
    assert result is expected


def test_surgery_correlate_supported(recwarn):
    from evalrx.analyzers.agent.loop_detect import LoopDetector

    model = _agent_model()
    data = _traj_batch(n_fail=2, n_pass=2)
    results = {"loop_detect": LoopDetector().run(model, data)}
    iv = SurgeryAgent().operate(_hypothesis("loop"), model, results, data)
    assert iv.status == HypothesisStatus.SUPPORTED
    assert iv.evidence["fail_rate_signal"] > iv.evidence["fail_rate_control"]


def test_surgery_inconclusive_no_labels():
    from evalrx.analyzers.agent.loop_detect import LoopDetector

    model = _agent_model()
    unlabeled = _traj_batch(n_fail=1, n_pass=1)
    for c in unlabeled:
        c.label = None
    results = {"loop_detect": LoopDetector().run(model, unlabeled)}
    iv = SurgeryAgent().operate(_hypothesis(), model, results, unlabeled)
    assert iv.status == HypothesisStatus.INCONCLUSIVE


def test_surgery_param_sweep():
    model = FakeModel(capabilities={Capability.GENERATE, Capability.ATTENTION})
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    agent = SurgeryAgent(analyzer_params={"attention": {"top_k": 2}})
    iv = agent.operate(_hypothesis(), model, {}, data)
    assert iv.status == HypothesisStatus.INCONCLUSIVE
    assert "attention" in iv.evidence["param_sweep"]


# ── M4 per-trial output: each operate() call gets its own self-contained
# experiments/NN_.../ folder (code + a kept, non-overwritten sandbox) ────────


class _FakeExperimentWriter:
    """Stands in for ExperimentWriter — actually runs code in the sandbox it's
    given (like the real writer does) so cleanup=False can be verified, but
    skips the LLM/CLI machinery entirely."""

    def __init__(self) -> None:
        self.calls = 0

    def write_and_run(self, *, hypothesis, model_context, cases_json, sandbox):
        from evalrx.eval_agent.stages.experiment_writer import ExperimentWriterResult

        self.calls += 1
        sandbox.run(f"print('verdict: 1.0')  # call {self.calls}")
        return ExperimentWriterResult(
            files={"main.py": f"# experiment for {hypothesis.predicted_failure_mode}"},
            verdict=1.0, metrics={"confidence": 0.95},
            returncode=0, timed_out=False, workdir=str(sandbox.workdir),
        )


def test_m4_experiment_gets_its_own_trial_with_kept_sandbox(tmp_path):
    """The bug this feature exists to fix: M4 experiments used to share (and
    overwrite) one sandbox, and ExperimentSandbox deleted it on success —
    so a *successful* experiment left no runnable code behind at all."""
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1")
    agent = SurgeryAgent(judge=FakeModel(), run_context=ctx)
    agent._writer = _FakeExperimentWriter()  # bypass the real LLM-driven writer

    hyp1 = _hypothesis("attention_sink")
    hyp2 = _hypothesis("modality_gap")
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])

    iv1 = agent.operate(hyp1, FakeModel(), {}, data)
    iv2 = agent.operate(hyp2, FakeModel(), {}, data)

    t1, t2 = iv1.experiment["trial_root"], iv2.experiment["trial_root"]
    assert t1 is not None and t2 is not None and t1 != t2

    ctx.logger.log_experiment(0, hyp1, iv1)
    ctx.logger.log_experiment(0, hyp2, iv2)
    ctx.finalize()

    from pathlib import Path

    p1, p2 = Path(t1), Path(t2)
    for p in (p1, p2):
        assert (p / "main.py").exists()
        assert (p / "record.md").exists()
        # cleanup=False: the script the fake writer ran via sandbox.run()
        # stays on disk even though it "succeeded" (verdict line, rc=0) —
        # the whole point of giving the experiment its own durable folder.
        assert list((p / "workspace").glob("exp_*.py")), \
            f"sandbox script should be kept in {p / 'workspace'}"


# ══════════════════════════════════════════════════════════════════════════════
# AutoDiagnoseLoop — full M1→M2→M3→M4
# ══════════════════════════════════════════════════════════════════════════════

def test_loop_analysis_only_mode():
    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    loop = AutoDiagnoseLoop(model=model, probe_agent=ProbeAgent(max_analyzers=2))
    report = loop.run(data)
    assert isinstance(report, AutoDiagnoseReport)
    assert report.resolved is False
    assert report.final_hypotheses == []
    assert report.final_analysis is not None
    assert len(report.final_results) <= 2


def test_loop_analysis_report_populated():
    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    loop = AutoDiagnoseLoop(model=model, probe_agent=ProbeAgent(max_analyzers=1))
    report = loop.run(data)
    assert report.final_analysis is not None
    assert isinstance(report.final_analysis.narrative, str)
    assert len(report.final_analysis.narrative) > 0


def test_loop_resolves_when_surgery_returns_fixed():
    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])

    judge = ScriptedModel(
        answers=["HYPOTHESIS: sink\nFAILURE_MODE: attention_sink\n"],
        capabilities={Capability.GENERATE},
    )

    def always_fixed(h, m, r, d):
        return InterventionResult(h, HypothesisStatus.SUPPORTED, fixed=True, evidence={})

    loop = AutoDiagnoseLoop(
        model=model,
        probe_agent=ProbeAgent(max_analyzers=1),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        surgery_agent=SurgeryAgent(verify_fn=always_fixed),
        max_cycles=5,
    )
    report = loop.run(data)
    assert report.resolved is True
    assert report.cycles == 1


def test_loop_stops_when_no_hypotheses():
    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    judge = ScriptedModel(answers=["NO_ISSUE"], capabilities={Capability.GENERATE})
    loop = AutoDiagnoseLoop(
        model=model,
        probe_agent=ProbeAgent(max_analyzers=1),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        max_cycles=5,
    )
    report = loop.run(data)
    assert report.resolved is False
    assert report.final_hypotheses == []


def test_loop_max_cycles_respected():
    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    judge = ScriptedModel(
        answers=["HYPOTHESIS: h\nFAILURE_MODE: f\n"],
        capabilities={Capability.GENERATE},
    )
    loop = AutoDiagnoseLoop(
        model=model,
        probe_agent=ProbeAgent(max_analyzers=1),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        surgery_agent=SurgeryAgent(verify_fn=lambda h, m, r, d: InterventionResult(
            h, HypothesisStatus.INCONCLUSIVE, fixed=False, evidence={}
        )),
        max_cycles=3,
    )
    report = loop.run(data)
    assert report.resolved is False
    assert report.cycles <= 3


def test_loop_store_accumulates():
    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    judge = ScriptedModel(
        answers=["HYPOTHESIS: h\nFAILURE_MODE: f\n"],
        capabilities={Capability.GENERATE},
    )
    loop = AutoDiagnoseLoop(
        model=model,
        probe_agent=ProbeAgent(max_analyzers=1),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        surgery_agent=SurgeryAgent(verify_fn=lambda h, m, r, d: InterventionResult(
            h, HypothesisStatus.INCONCLUSIVE, fixed=False, evidence={}
        )),
        max_cycles=1,
    )
    report = loop.run(data)
    assert len(report.store.results) > 0
    assert len(report.store.hypotheses) > 0


def test_loop_docker_mode_falls_back_gracefully(recwarn):
    """When Docker is unavailable, ProbeAgent warns and falls back (no crash)."""
    model = _llm_model()
    data = CaseBatch([FailureCase(inputs=Inputs(prompt="x"))])
    agent = ProbeAgent(use_docker=True, docker_image="nonexistent:tag", max_analyzers=1)
    loop = AutoDiagnoseLoop(model=model, probe_agent=agent)
    report = loop.run(data)
    assert isinstance(report, AutoDiagnoseReport)


# ── the per-analyzer case ceiling ────────────────────────────────────────────
def _labelled_batch(n_pass, n_fail):
    from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
    cases = [FailureCase(inputs=Inputs(prompt=f"p{i}"), observed="o",
                         expected="e", label=Label.PASS) for i in range(n_pass)]
    cases += [FailureCase(inputs=Inputs(prompt=f"f{i}"), observed="o",
                          expected="e", label=Label.FAIL) for i in range(n_fail)]
    return CaseBatch(cases)


class _NoKnob:
    """Like first_error_judge / rise / self_consistency: no max_cases at all."""
    max_cases = None
    name = "no_knob"


class _OwnKnob:
    max_cases = 12
    name = "own_knob"


def _agent(cap):
    from evalrx.eval_agent.stages.probe_agent import ProbeAgent
    return ProbeAgent(max_cases_per_analyzer=cap)


def test_cap_bounds_an_analyzer_that_has_no_max_cases_argument():
    """The wall-clock case: analyzer_overrides cannot reach these at all."""
    agent = _agent(32)
    out = agent._cap_cases("no_knob", _NoKnob(), _labelled_batch(114, 158))
    assert len(out) == 32
    assert agent.capped_analyzers["no_knob"] == (272, 32)


def test_cap_keeps_both_label_classes_in_proportion():
    """M1 contrasts PASS against FAIL — a subset with one class is not a probe,
    it is a null result that looks like a measurement."""
    from evalrx.core.case import Label

    out = _agent(32)._cap_cases("no_knob", _NoKnob(), _labelled_batch(114, 158))
    seen = [c.label for c in out]
    assert seen.count(Label.FAIL) == 19 and seen.count(Label.PASS) == 13


def test_cap_survives_a_batch_that_opens_with_one_label():
    """Head truncation would hand this analyzer 32 PASS and zero FAIL."""
    from evalrx.core.case import Label

    out = _agent(32)._cap_cases("no_knob", _NoKnob(), _labelled_batch(100, 8))
    assert len(out) == 32
    assert min(sum(1 for c in out if c.label is lab) for lab in
               (Label.PASS, Label.FAIL)) > 0


def test_cap_does_not_override_a_tighter_knob_the_analyzer_set_itself():
    batch = _labelled_batch(114, 158)
    agent = _agent(32)
    assert len(agent._cap_cases("own_knob", _OwnKnob(), batch)) == len(batch)
    assert "own_knob" not in agent.capped_analyzers


def test_cap_is_off_by_default_and_a_noop_below_the_ceiling():
    from evalrx.eval_agent.stages.probe_agent import ProbeAgent

    batch = _labelled_batch(114, 158)
    assert len(ProbeAgent()._cap_cases("no_knob", _NoKnob(), batch)) == len(batch)
    small = _labelled_batch(5, 5)
    assert len(_agent(32)._cap_cases("no_knob", _NoKnob(), small)) == 10


def test_cap_picks_the_same_cases_every_run():
    """Two runs on one batch must be comparable, so no RNG in the selection."""
    batch = _labelled_batch(114, 158)
    first = [c.inputs.prompt for c in _agent(32)._cap_cases("n", _NoKnob(), batch)]
    second = [c.inputs.prompt for c in _agent(32)._cap_cases("n", _NoKnob(), batch)]
    assert first == second
    # and it strides rather than taking the head of each group
    assert first[:2] != [c.inputs.prompt for c in batch][:2]
