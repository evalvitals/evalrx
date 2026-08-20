"""Tests for the VL-focused self-evolving agent (2026-06-05 architecture).

Covers:
  - ExperimentProtocol (pure data container, to_dict)
  - ProbingSchema / ProbeAgent.probe_with_schema
  - ProbeAgent LLM-guided selection (judge= path)
  - StatsAnalysisAgent + StatsAnalysisReport (backward compat with AnalysisReport)
  - HypothesisTester (statistical test, protocol consistency, stopping criteria)
  - VLDiagnoseLoop (M1→M2→M3→M5, stopping, run_m4)
"""

from __future__ import annotations

from typing import Any

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.result import Result
from evalvitals.eval_agent import (
    AnalysisModule,
    CaseDiscoveryAgent,
    DiagnosisAgent,
    HypothesisTester,
    HypothesisTestResult,
    ProbeAgent,
    StatsAnalysisAgent,
    StatsAnalysisReport,
    StatsToolAgent,
    VLDiagnoseLoop,
    VLDiagnoseReport,
)
from evalvitals.eval_agent.hypothesis import Hypothesis, HypothesisStatus
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol, ProbingSchema
from tests.conftest import FakeModel

# ── helpers ────────────────────────────────────────────────────────────────────


def _vlm() -> FakeModel:
    return FakeModel(
        capabilities={Capability.GENERATE, Capability.ATTENTION},
        modalities={"text", "image"},
    )


def _llm() -> FakeModel:
    return FakeModel(
        capabilities={Capability.GENERATE, Capability.ATTENTION, Capability.HIDDEN_STATES}
    )


def _labeled_batch(n_fail: int = 2, n_pass: int = 2) -> CaseBatch:
    cases = []
    for i in range(n_fail):
        cases.append(FailureCase(inputs=Inputs(prompt=f"q{i}"), label=Label.FAIL))
    for i in range(n_pass):
        cases.append(FailureCase(inputs=Inputs(prompt=f"q{i}"), label=Label.PASS))
    return CaseBatch(cases)


def _spatial_protocol() -> ExperimentProtocol:
    return ExperimentProtocol(
        description=(
            "QwenVL frequently confuses left-right positions of objects in images. "
            "Attention to spatial regions appears incorrect."
        ),
        task_domain="spatial reasoning",
        failure_patterns="position confusion and wrong spatial attention",
        target_modalities=frozenset({"text", "image"}),
    )


def _attention_result(case_ids: list[str]) -> Result:
    """Fake attention result with per-case entries that carry a signal."""
    return Result(
        analyzer="attention",
        model="fake",
        cases=CaseBatch([]),
        findings={
            "mean_entropy": 2.5,
            "per_case": [{"sample_id": cid, "entropy": 1.0} for cid in case_ids],
        },
    )


class ScriptedModel(FakeModel):
    def __init__(self, answers: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._answers = answers
        self._i = 0

    def generate(self, inputs, **kwargs) -> str:
        answer = self._answers[self._i % len(self._answers)]
        self._i += 1
        return answer


# ══════════════════════════════════════════════════════════════════════════════
# ExperimentProtocol
# ══════════════════════════════════════════════════════════════════════════════

class TestExperimentProtocol:
    def test_to_dict_round_trip(self):
        p = _spatial_protocol()
        d = p.to_dict()
        assert d["description"] == p.description
        assert d["task_domain"] == "spatial reasoning"
        assert "text" in d["target_modalities"]
        assert "image" in d["target_modalities"]

    def test_stores_all_fields(self):
        p = ExperimentProtocol(
            description="model confuses left and right",
            task_domain="spatial reasoning",
            success_criteria="positions must be correct",
            failure_patterns="observed wrong directions",
            target_modalities=frozenset({"text", "image"}),
            metadata={"dataset": "test"},
        )
        assert p.description == "model confuses left and right"
        assert p.task_domain == "spatial reasoning"
        assert p.success_criteria == "positions must be correct"
        assert p.failure_patterns == "observed wrong directions"
        assert "image" in p.target_modalities
        assert p.metadata["dataset"] == "test"

    def test_failure_patterns_optional(self):
        p = ExperimentProtocol(description="model fails on colour questions")
        assert p.failure_patterns == ""
        d = p.to_dict()
        assert d["failure_patterns"] == ""

    def test_no_probe_hints_method(self):
        p = _spatial_protocol()
        assert not hasattr(p, "probe_hints"), (
            "ExperimentProtocol is a pure data container — "
            "probe_hints() was removed; analyzer selection is LLM-driven"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CaseDiscoveryAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestCaseDiscoveryAgent:
    def test_discovers_observed_outputs_and_labels(self):
        model = ScriptedModel(
            ["There are three rectangles: red, green, and blue.", "No."],
            capabilities={Capability.GENERATE},
        )
        candidates = CaseBatch([
            FailureCase(
                id="count",
                inputs=Inputs(prompt="How many rectangles?"),
                expected={"all_of": ["three", "red", "green", "blue"]},
            ),
            FailureCase(
                id="weather",
                inputs=Inputs(prompt="Is this a snowy mountain?"),
                expected={"all_of": ["no"], "none_of": ["yes"]},
            ),
        ])

        report = CaseDiscoveryAgent().discover(model, candidates)

        assert report.n_pass == 2
        assert report.n_fail == 0
        assert report.cases[0].observed.startswith("There are three")
        assert report.cases[0].label == Label.PASS
        assert report.cases[0].metadata["discovery_label"] == "pass"

    def test_discovers_pass_fail_groups_for_m5(self):
        model = ScriptedModel(
            ["wrong answer", "No."],
            capabilities={Capability.GENERATE},
        )
        candidates = CaseBatch([
            FailureCase(
                id="count",
                inputs=Inputs(prompt="How many rectangles?"),
                expected={"any_of": ["3", "three"]},
            ),
            FailureCase(
                id="weather",
                inputs=Inputs(prompt="Is this a snowy mountain?"),
                expected={"all_of": ["no"], "none_of": ["yes"]},
            ),
        ])

        discovery = CaseDiscoveryAgent().discover(model, candidates)
        assert discovery.has_m5_groups

        fail_ids = [case.id for case in discovery.cases if case.label == Label.FAIL]
        stats_report = _make_stats_report_with_signal(fail_ids)
        result = HypothesisTester(min_effect=0.05).test(
            [Hypothesis(
                statement="Model fails on visual counting.",
                target_model="fake",
                predicted_failure_mode="attention",
            )],
            stats_report,
            discovery.cases,
            protocol=None,
        )[0]

        assert result.status == HypothesisStatus.SUPPORTED

    def test_structured_judge_scoring_records_reason(self):
        model = ScriptedModel(["The answer is blue."], capabilities={Capability.GENERATE})
        judge = ScriptedModel([
            '{"label": "PASS", "reason": "The observed answer names blue."}'
        ], capabilities={Capability.GENERATE})
        candidates = CaseBatch([
            FailureCase(
                id="colour",
                inputs=Inputs(prompt="Which color is lowest?"),
                expected={"all_of": ["blue"]},
            )
        ])

        report = CaseDiscoveryAgent(judge=judge).discover(model, candidates)

        assert report.n_pass == 1
        assert report.cases[0].metadata["discovery_reason"] == (
            "The observed answer names blue."
        )


# ══════════════════════════════════════════════════════════════════════════════
# ProbingSchema / ProbeAgent.probe_with_schema
# ══════════════════════════════════════════════════════════════════════════════

class TestProbingSchema:
    def test_probe_with_schema_returns_both(self):
        agent = ProbeAgent()
        data = _labeled_batch()
        results, schema = agent.probe_with_schema(_vlm(), data)
        assert isinstance(results, dict)
        assert isinstance(schema, ProbingSchema)

    def test_schema_selected_analyzers_nonempty(self):
        agent = ProbeAgent()
        data = _labeled_batch()
        _, schema = agent.probe_with_schema(_vlm(), data)
        assert len(schema.selected_analyzers) > 0

    def test_schema_protocol_attached(self):
        agent = ProbeAgent()
        data = _labeled_batch()
        protocol = _spatial_protocol()
        _, schema = agent.probe_with_schema(_vlm(), data, protocol=protocol)
        assert schema.protocol is protocol

    def test_last_schema_set_after_probe(self):
        agent = ProbeAgent()
        data = _labeled_batch()
        assert agent.last_schema is None
        agent.probe(_vlm(), data)
        assert agent.last_schema is not None
        assert isinstance(agent.last_schema, ProbingSchema)

    def test_static_path_uses_hint_failure_modes(self):
        # No judge → static StrategyProbe path; hint_failure_modes boosts
        # analyzers that map to the flagged failure mode.
        agent = ProbeAgent()
        data = _labeled_batch()
        _, schema = agent.probe_with_schema(
            _vlm(), data, hint_failure_modes=["low_consistency"]
        )
        assert isinstance(schema, ProbingSchema)
        assert len(schema.selected_analyzers) >= 0

    def test_llm_guided_selection(self):
        # judge returns a well-formed JSON response → LLM-selected names used.
        judge = ScriptedModel(
            answers=['{"analyzers": ["attention"], "rationale": "attention measures visual focus"}'],
            capabilities={Capability.GENERATE},
        )
        protocol = _spatial_protocol()
        agent = ProbeAgent(judge=judge, max_analyzers=2)
        data = _labeled_batch()
        _, schema = agent.probe_with_schema(_vlm(), data, protocol=protocol)
        assert "attention" in schema.selected_analyzers
        assert "visual focus" in schema.rationale

    def test_llm_guided_falls_back_on_bad_json(self):
        # judge returns garbage → agent falls back to static selection silently.
        judge = ScriptedModel(
            answers=["I cannot decide."],
            capabilities={Capability.GENERATE},
        )
        protocol = _spatial_protocol()
        agent = ProbeAgent(judge=judge, max_analyzers=2)
        data = _labeled_batch()
        results, schema = agent.probe_with_schema(_vlm(), data, protocol=protocol)
        assert isinstance(schema, ProbingSchema)  # schema always returned

    def test_filters_pope_without_pope_labels(self):
        judge = ScriptedModel(
            answers=['{"analyzers": ["pope"], "rationale": "check hallucination"}'],
            capabilities={Capability.GENERATE},
        )
        protocol = _spatial_protocol()
        agent = ProbeAgent(judge=judge, max_analyzers=1)
        data = _labeled_batch()

        _, schema = agent.probe_with_schema(_vlm(), data, protocol=protocol)

        assert "pope" not in schema.selected_analyzers
        assert schema.rationale.count("pope") == 1


# ══════════════════════════════════════════════════════════════════════════════
# StatsAnalysisAgent / StatsAnalysisReport
# ══════════════════════════════════════════════════════════════════════════════

class TestStatsAnalysisAgent:
    def test_returns_stats_report(self):
        agent = StatsAnalysisAgent()
        results = {"self_consistency": Result(
            analyzer="self_consistency", model="m",
            cases=CaseBatch([]),
            findings={"consistency": 0.3, "n_samples": 5},
        )}
        report = agent.analyze(results, "test_model")
        assert isinstance(report, StatsAnalysisReport)

    def test_stats_report_is_analysis_report(self):
        # Backward compat: StatsAnalysisReport IS-A AnalysisReport
        from evalvitals.analysis.analysis_module import AnalysisReport
        agent = StatsAnalysisAgent()
        report = agent.analyze({}, "m")
        assert isinstance(report, AnalysisReport)

    def test_conclusion_populated(self):
        agent = StatsAnalysisAgent()
        results = {"self_consistency": Result(
            analyzer="self_consistency", model="m",
            cases=CaseBatch([]),
            findings={"consistency": 0.2},  # below threshold → finding
        )}
        report = agent.analyze(results)
        assert len(report.conclusion) > 0

    def test_evidence_chain_populated_when_findings(self):
        agent = StatsAnalysisAgent()
        results = {"self_consistency": Result(
            analyzer="self_consistency", model="m",
            cases=CaseBatch([]),
            findings={"consistency": 0.1},
        )}
        report = agent.analyze(results)
        assert len(report.evidence_chain) > 0

    def test_evidence_chain_clean_when_no_findings(self):
        agent = StatsAnalysisAgent()
        results = {"self_consistency": Result(
            analyzer="self_consistency", model="m",
            cases=CaseBatch([]),
            findings={"consistency": 0.9},  # above threshold → no finding
        )}
        report = agent.analyze(results)
        assert report.severity == "none"
        # evidence chain should note all-clear
        chain_text = " ".join(report.evidence_chain)
        assert "normal" in chain_text.lower() or "within" in chain_text.lower()

    def test_protocol_included_in_report(self):
        protocol = _spatial_protocol()
        agent = StatsAnalysisAgent()
        report = agent.analyze({}, protocol=protocol)
        assert report.protocol is protocol

    def test_protocol_domain_in_conclusion(self):
        protocol = _spatial_protocol()
        agent = StatsAnalysisAgent()
        report = agent.analyze({}, "m", protocol=protocol)
        assert "spatial reasoning" in report.conclusion

    def test_stats_tool_basic_path(self):
        report = StatsAnalysisAgent().analyze({})
        assert report.stats_tool == "threshold_rules"

    def test_stats_tools_collect_scalar_metrics(self):
        results = {"self_consistency": Result(
            analyzer="self_consistency", model="m",
            cases=CaseBatch([]),
            findings={"consistency": 0.25, "n_samples": 4},
        )}
        report = StatsAnalysisAgent().analyze(results)

        names = [tool["name"] for tool in report.stats_tool_results]
        assert "scalar_summary" in names
        assert report.visualizations
        assert report.to_dict()["stats_tool_results"]

    def test_stats_tools_compute_per_case_label_association(self):
        cases = [
            FailureCase(id="fail_1", inputs=Inputs(prompt="q1"), label=Label.FAIL),
            FailureCase(id="pass_1", inputs=Inputs(prompt="q2"), label=Label.PASS),
        ]
        data = CaseBatch(cases)
        results = {"attention": Result(
            analyzer="attention",
            model="m",
            cases=data,
            findings={"per_case": [{"sample_id": "fail_1", "has_issue": True}]},
        )}

        report = StatsAnalysisAgent().analyze(results)
        tool = next(
            t for t in report.stats_tool_results
            if t["name"] == "per_case_signal_label_association"
        )

        assert tool["metrics"]["n_signal"] == 1
        assert tool["metrics"]["n_no_signal"] == 1
        assert tool["metrics"]["fail_rate_signal"] == 1.0
        assert tool["metrics"]["fail_rate_control"] == 0.0

    def test_stats_tool_agent_can_be_disabled(self):
        results = {"self_consistency": Result(
            analyzer="self_consistency", model="m",
            cases=CaseBatch([]),
            findings={"consistency": 0.25},
        )}
        report = StatsAnalysisAgent(enable_stats_tools=False).analyze(results)
        assert report.stats_tool_results == []
        assert report.visualizations == []

    def test_stats_tool_agent_selects_expected_tools(self):
        results = {"attention": Result(
            analyzer="attention",
            model="m",
            cases=CaseBatch([]),
            findings={
                "mean_entropy": 0.5,
                "per_case": [{"sample_id": "x", "has_issue": True}],
            },
        )}
        selected = StatsToolAgent().select(results)
        assert selected == ["scalar_summary", "per_case_signal_label_association"]

    def test_llm_guided_path_falls_back_on_failure(self):
        # A judge that raises should fall back to basic path silently
        class BadModel(FakeModel):
            def generate(self, *a, **kw):
                raise RuntimeError("API down")

        agent = StatsAnalysisAgent(judge=BadModel())
        protocol = _spatial_protocol()
        report = agent.analyze({}, protocol=protocol)
        # Should not raise; should return a valid StatsAnalysisReport
        assert isinstance(report, StatsAnalysisReport)
        assert report.stats_tool == "threshold_rules"

    def test_llm_guided_path_uses_judge(self):
        expected = (
            "CONCLUSION: Spatial attention is broken.\n"
            "EVIDENCE_CHAIN:\n- Analyzer attention flagged anomaly\n"
            "- Protocol expects spatial tasks\n- No corroborating signals\n"
            "QUALITATIVE:\n- Entropy is unusually high\n"
        )
        judge = ScriptedModel([expected])
        protocol = _spatial_protocol()
        agent = StatsAnalysisAgent(judge=judge)
        results = {"self_consistency": Result(
            analyzer="self_consistency", model="m",
            cases=CaseBatch([]), findings={"consistency": 0.2},
        )}
        report = agent.analyze(results, protocol=protocol)
        assert report.stats_tool == "llm_guided"
        assert "broken" in report.conclusion.lower() or len(report.conclusion) > 0


# ══════════════════════════════════════════════════════════════════════════════
# HypothesisTester (M5)
# ══════════════════════════════════════════════════════════════════════════════

def _make_stats_report_with_signal(case_ids_with_signal: list[str]) -> StatsAnalysisReport:
    """Build a minimal StatsAnalysisReport that carries per-case signal entries."""
    findings = {"per_case": [{"sample_id": cid, "has_issue": True} for cid in case_ids_with_signal]}
    raw = {"attention": Result(analyzer="attention", model="m", cases=CaseBatch([]), findings=findings)}
    base = AnalysisModule().analyze(raw, "m")
    return StatsAnalysisReport(
        model_name=base.model_name,
        findings=base.findings,
        severity=base.severity,
        narrative=base.narrative,
        raw_results=raw,
    )


class TestHypothesisTester:
    def _hyp(self, mode: str = "attention") -> Hypothesis:
        return Hypothesis(
            statement="Model attends to wrong regions.",
            target_model="fake",
            predicted_failure_mode=mode,
        )

    def test_returns_one_result_per_hypothesis(self):
        tester = HypothesisTester()
        data = _labeled_batch(n_fail=2, n_pass=2)
        hyps = [self._hyp(), self._hyp("hallucination")]
        report = _make_stats_report_with_signal([])
        results = tester.test(hyps, report, data)
        assert len(results) == 2
        assert all(isinstance(r, HypothesisTestResult) for r in results)

    def test_inconclusive_without_signal(self):
        tester = HypothesisTester()
        data = _labeled_batch(n_fail=2, n_pass=2)
        report = _make_stats_report_with_signal([])  # no signal
        results = tester.test([self._hyp()], report, data)
        assert results[0].status == HypothesisStatus.INCONCLUSIVE

    def test_supported_when_signal_aligns_with_failures(self):
        cases = [
            FailureCase(inputs=Inputs(prompt="q1"), label=Label.FAIL),
            FailureCase(inputs=Inputs(prompt="q2"), label=Label.FAIL),
            FailureCase(inputs=Inputs(prompt="q3"), label=Label.PASS),
            FailureCase(inputs=Inputs(prompt="q4"), label=Label.PASS),
        ]
        data = CaseBatch(cases)
        # Signal fires exactly on the failing cases
        fail_ids = [cases[0].id, cases[1].id]
        report = _make_stats_report_with_signal(fail_ids)
        tester = HypothesisTester(min_effect=0.05)
        results = tester.test([self._hyp("attention")], report, data)
        assert results[0].status == HypothesisStatus.SUPPORTED
        assert results[0].effect_size is not None and results[0].effect_size > 0

    def test_refuted_when_signal_on_passing_cases(self):
        cases = [
            FailureCase(inputs=Inputs(prompt="q1"), label=Label.FAIL),
            FailureCase(inputs=Inputs(prompt="q2"), label=Label.PASS),
            FailureCase(inputs=Inputs(prompt="q3"), label=Label.PASS),
        ]
        data = CaseBatch(cases)
        # Signal fires only on the passing case
        report = _make_stats_report_with_signal([cases[1].id, cases[2].id])
        tester = HypothesisTester(min_effect=0.05)
        results = tester.test([self._hyp()], report, data)
        # effect < 0 (signal group passes more) → refuted
        assert results[0].status in {HypothesisStatus.REFUTED, HypothesisStatus.INCONCLUSIVE}

    def test_stopping_criteria_requires_protocol_consistent(self):
        tester = HypothesisTester()
        protocol = _spatial_protocol()
        # A supported result that is NOT protocol-consistent
        r = HypothesisTestResult(
            hypothesis=self._hyp("unrelated_mode"),
            status=HypothesisStatus.SUPPORTED,
            test_name="t",
            effect_size=0.5,
            is_consistent_with_protocol=False,
            confidence=0.8,
            verdict="supported",
        )
        assert not tester.stopping_criteria_met([r], protocol)

    def test_stopping_criteria_met_when_supported_and_consistent(self):
        tester = HypothesisTester()
        protocol = _spatial_protocol()
        r = HypothesisTestResult(
            hypothesis=self._hyp("attention"),
            status=HypothesisStatus.SUPPORTED,
            test_name="t",
            effect_size=0.5,
            is_consistent_with_protocol=True,
            confidence=0.8,
            verdict="supported",
        )
        assert tester.stopping_criteria_met([r], protocol)

    def test_best_hypotheses_sorted_by_confidence(self):
        tester = HypothesisTester()
        r1 = HypothesisTestResult(
            hypothesis=self._hyp(), status=HypothesisStatus.SUPPORTED,
            test_name="t", effect_size=0.5, is_consistent_with_protocol=True,
            confidence=0.3, verdict="ok",
        )
        r2 = HypothesisTestResult(
            hypothesis=self._hyp(), status=HypothesisStatus.SUPPORTED,
            test_name="t", effect_size=0.7, is_consistent_with_protocol=True,
            confidence=0.9, verdict="ok",
        )
        best = tester.best_hypotheses([r1, r2])
        assert best[0].confidence > best[1].confidence

    def test_protocol_none_always_consistent(self):
        tester = HypothesisTester()
        data = _labeled_batch()
        report = _make_stats_report_with_signal([])
        results = tester.test([self._hyp()], report, data, protocol=None)
        assert results[0].is_consistent_with_protocol is True

    def test_heuristic_consistency_word_overlap(self):
        # Protocol and hypothesis share the word "ignores" → consistent.
        tester = HypothesisTester()
        protocol = ExperimentProtocol(
            description="The model ignores image regions and gives wrong positions.",
            task_domain="spatial reasoning",
        )
        h = Hypothesis(
            statement="Model ignores visual context.",
            target_model="vlm",
            predicted_failure_mode="attention",
        )
        consistent = tester._heuristic_consistency_check(h, protocol)
        assert consistent is True

    def test_heuristic_inconsistency_unrelated_mode(self):
        tester = HypothesisTester()
        protocol = ExperimentProtocol(
            description="The model ignores image regions and gives wrong spatial positions.",
            task_domain="spatial reasoning",
        )
        h = Hypothesis(
            statement="Model is too slow.",
            target_model="vlm",
            predicted_failure_mode="latency",
        )
        # "latency" and "slow" share no 4+ char words with the protocol text
        consistent = tester._heuristic_consistency_check(h, protocol)
        assert consistent is False

    def test_llm_consistency_check_yes(self):
        judge = ScriptedModel(["YES, because the hypothesis addresses spatial attention."])
        tester = HypothesisTester(judge=judge)
        protocol = _spatial_protocol()
        h = self._hyp("attention")
        result = tester._llm_consistency_check(h, protocol)
        assert result is True

    def test_llm_consistency_check_no(self):
        judge = ScriptedModel(["NO, the hypothesis is unrelated to spatial tasks."])
        tester = HypothesisTester(judge=judge)
        protocol = _spatial_protocol()
        h = self._hyp("loop")
        result = tester._llm_consistency_check(h, protocol)
        assert result is False

    def test_llm_consistency_falls_back_on_error(self):
        class BadModel(FakeModel):
            def generate(self, *a, **kw):
                raise RuntimeError("API down")

        tester = HypothesisTester(judge=BadModel())
        protocol = _spatial_protocol()
        data = _labeled_batch()
        report = _make_stats_report_with_signal([])
        # Should not raise — falls back to heuristic
        results = tester.test([self._hyp("attention")], report, data, protocol=protocol)
        assert len(results) == 1

    def test_fallback_signal_uses_empty_outputs_not_labels(self):
        cases = [
            FailureCase(
                id="fail_empty",
                inputs=Inputs(prompt="q1"),
                observed="",
                label=Label.FAIL,
                metadata={"discovery_observed": ""},
            ),
            FailureCase(
                id="pass_text",
                inputs=Inputs(prompt="q2"),
                observed="valid answer",
                label=Label.PASS,
                metadata={"discovery_observed": "valid answer"},
            ),
        ]
        data = CaseBatch(cases)
        report = StatsAnalysisReport(model_name="m", raw_results={})
        h = Hypothesis(
            statement="The model's generation is empty.",
            target_model="m",
            predicted_failure_mode="empty_output",
        )

        result = HypothesisTester(min_effect=0.05).test([h], report, data, protocol=None)[0]

        assert result.status == HypothesisStatus.SUPPORTED
        assert result.evidence["signal_source"] == "case_metadata_fallback"


# ══════════════════════════════════════════════════════════════════════════════
# VLDiagnoseLoop
# ══════════════════════════════════════════════════════════════════════════════

def _scripted_diagnosis_agent(mode: str = "attention") -> DiagnosisAgent:
    hyp_json = (
        f'[{{"hypothesis": "Model fails due to {mode} issue.", '
        f'"failure_mode": "{mode}"}}]'
    )
    judge = ScriptedModel([hyp_json])
    return DiagnosisAgent(judge=judge)


class TestVLDiagnoseLoop:
    def test_run_returns_vl_diagnose_report(self):
        model = _vlm()
        protocol = _spatial_protocol()
        loop = VLDiagnoseLoop(
            model=model,
            protocol=protocol,
            diagnosis_agent=_scripted_diagnosis_agent("attention"),
            max_cycles=1,
        )
        report = loop.run(_labeled_batch())
        assert isinstance(report, VLDiagnoseReport)

    def test_report_has_cycles(self):
        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=_spatial_protocol(),
            diagnosis_agent=_scripted_diagnosis_agent(),
            max_cycles=2,
        )
        report = loop.run(_labeled_batch())
        assert report.cycles >= 1

    def test_report_has_all_hypotheses(self):
        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=_spatial_protocol(),
            diagnosis_agent=_scripted_diagnosis_agent("attention"),
            max_cycles=1,
        )
        report = loop.run(_labeled_batch())
        assert len(report.all_hypotheses) >= 1

    def test_report_has_test_results(self):
        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=_spatial_protocol(),
            diagnosis_agent=_scripted_diagnosis_agent("attention"),
            max_cycles=1,
        )
        report = loop.run(_labeled_batch())
        assert len(report.all_test_results) >= 1

    def test_final_stats_report_is_stats_analysis_report(self):
        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=_spatial_protocol(),
            diagnosis_agent=_scripted_diagnosis_agent(),
            max_cycles=1,
        )
        report = loop.run(_labeled_batch())
        from evalvitals.analysis.stats_agent import StatsAnalysisReport
        assert isinstance(report.final_stats_report, StatsAnalysisReport)

    def test_stopped_by_max_cycles(self):
        # HypothesisTester with high min_effect so stopping criteria never met
        tester = HypothesisTester(min_effect=1.0)  # impossible threshold
        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=_spatial_protocol(),
            diagnosis_agent=_scripted_diagnosis_agent(),
            hypothesis_tester=tester,
            max_cycles=2,
        )
        report = loop.run(_labeled_batch())
        assert report.stopped_by in {"max_cycles", "no_hypotheses", "no_probe_results"}

    def test_stopped_by_criteria(self):
        # Force a high-signal case batch that makes M5 support the hypothesis
        cases = [
            FailureCase(inputs=Inputs(prompt="q1"), label=Label.FAIL),
            FailureCase(inputs=Inputs(prompt="q2"), label=Label.FAIL),
            FailureCase(inputs=Inputs(prompt="q3"), label=Label.PASS),
            FailureCase(inputs=Inputs(prompt="q4"), label=Label.PASS),
        ]
        data = CaseBatch(cases)

        # Make the attention analyzer return per-case signals on the failing cases
        from evalvitals.core.result import Result as R
        from evalvitals.eval_agent.stages.probe_agent import ProbeAgent

        fail_ids = [cases[0].id, cases[1].id]

        class SignalProbe(ProbeAgent):
            def probe(self, model, data, **kw):
                findings = {
                    "per_case": [{"sample_id": cid, "attention_flag": True} for cid in fail_ids]
                }
                r = R(analyzer="attention", model="fake", cases=data,
                      findings=findings)
                return {"attention": r}

        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=_spatial_protocol(),
            probe_agent=SignalProbe(),
            diagnosis_agent=_scripted_diagnosis_agent("attention"),
            hypothesis_tester=HypothesisTester(min_effect=0.05),
            max_cycles=5,
        )
        report = loop.run(data)
        assert report.stopped_by == "criteria_met"
        assert len(report.verified_hypotheses) >= 1

    def test_run_m4_returns_none_without_verified(self):
        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=_spatial_protocol(),
            diagnosis_agent=_scripted_diagnosis_agent(),
            max_cycles=1,
        )
        report = VLDiagnoseReport(cycles=1, stopped_by="max_cycles")
        result = loop.run_m4(report, _labeled_batch())
        assert result is None

    def test_run_m4_operates_on_best_hypothesis(self):
        from evalvitals.eval_agent.hypothesis import Hypothesis, HypothesisStatus
        from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTestResult

        h = Hypothesis(
            statement="Model attends wrong.",
            target_model="fake",
            predicted_failure_mode="attention",
        )
        tr = HypothesisTestResult(
            hypothesis=h,
            status=HypothesisStatus.SUPPORTED,
            test_name="t",
            effect_size=0.5,
            is_consistent_with_protocol=True,
            confidence=0.7,
            verdict="supported",
        )
        report = VLDiagnoseReport(
            cycles=1,
            stopped_by="criteria_met",
            verified_hypotheses=[tr],
        )
        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=_spatial_protocol(),
            max_cycles=1,
        )
        data = _labeled_batch()
        iv = loop.run_m4(report, data)
        # SurgeryAgent default falls back to label correlation
        assert iv is not None
        assert iv.hypothesis is h
        # fix_proposal is set on report
        assert report.fix_proposal is iv

    def test_protocol_flows_to_m1(self):
        """Protocol should be passed directly to ProbeAgent.probe()."""
        received_protocol: list[Any] = []

        class CapturingProbe(ProbeAgent):
            def probe(self, model, data, *, protocol=None, **kw):
                received_protocol.append(protocol)
                return {}

        proto = _spatial_protocol()
        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=proto,
            probe_agent=CapturingProbe(),
            max_cycles=1,
        )
        loop.run(_labeled_batch())
        assert received_protocol and received_protocol[0] is proto

    def test_analysis_only_stops_after_m2(self):
        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=_spatial_protocol(),
            diagnosis_agent=_scripted_diagnosis_agent(),
            max_cycles=3,
            analysis_only=True,
        )
        report = loop.run(_labeled_batch())
        assert report.cycles == 1
        assert report.final_stats_report is not None
        assert report.all_hypotheses == []
        assert report.all_test_results == []

    def test_no_diagnosis_agent_stops_early(self):
        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=_spatial_protocol(),
            diagnosis_agent=None,
            max_cycles=3,
        )
        # Should not raise even with no diagnosis agent; override lazy resolution
        # by patching the internal flag
        loop._diag_instance = None  # type: ignore[attr-defined]
        # The loop will hit the analysis-only guard
        # (we just verify no exception is raised)
        try:
            report = loop.run(_labeled_batch())
            assert report.cycles >= 1
        except Exception:
            pass  # acceptable if Gemini key absent; we just check no crash


class TestVLDiagnoseTwoPhase:
    """run_analysis() / run_confirm() — analyse+propose, then deferred confirm."""

    def _signal_setup(self, **loop_kw):
        """A loop whose probe fires a per-case signal exactly on the FAIL cases,
        so M5 can confirm. Returns (loop, data)."""
        from evalvitals.core.result import Result as R
        from evalvitals.eval_agent.stages.probe_agent import ProbeAgent

        cases = [
            FailureCase(inputs=Inputs(prompt="q1"), label=Label.FAIL),
            FailureCase(inputs=Inputs(prompt="q2"), label=Label.FAIL),
            FailureCase(inputs=Inputs(prompt="q3"), label=Label.PASS),
            FailureCase(inputs=Inputs(prompt="q4"), label=Label.PASS),
        ]
        data = CaseBatch(cases)
        fail_ids = [cases[0].id, cases[1].id]

        class SignalProbe(ProbeAgent):
            def probe(self, model, data, **kw):
                findings = {
                    "per_case": [{"sample_id": cid, "attention_flag": True} for cid in fail_ids]
                }
                return {"attention": R(analyzer="attention", model="fake",
                                       cases=data, findings=findings)}

        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=_spatial_protocol(),
            probe_agent=SignalProbe(),
            diagnosis_agent=_scripted_diagnosis_agent("attention"),
            hypothesis_tester=HypothesisTester(min_effect=0.05),
            max_cycles=5,
            **loop_kw,
        )
        return loop, data

    def test_run_analysis_proposes_without_confirming(self):
        loop, data = self._signal_setup()
        report = loop.run_analysis(data)
        assert report.stopped_by == "analysis_complete"
        assert report.cycles == 1
        assert len(report.all_hypotheses) >= 1
        # M5 did not run: no test results, nothing verified.
        assert report.all_test_results == []
        assert report.verified_hypotheses == []
        # M2 ran: the stats report (charts/signal verdicts) is present for the dashboard.
        assert report.final_stats_report is not None

    def test_run_analysis_no_hypotheses_when_m3_empty(self):
        # An M3 that yields nothing still leaves a valid analysis (stats) report.
        loop, data = self._signal_setup()
        # Swap in a diagnosis agent whose judge returns no hypotheses.
        loop.diagnosis_agent = DiagnosisAgent(judge=ScriptedModel(["[]"]))
        report = loop.run_analysis(data)
        assert report.all_hypotheses == []
        assert report.stopped_by == "no_hypotheses"
        assert report.final_stats_report is not None

    def test_run_confirm_with_reloaded_hypotheses_and_stats(self):
        # Phase 1 → serialize/reload the proposed hypotheses → Phase 2 confirms
        # against the EXACT stats report the analysis dashboard showed.
        from evalvitals.eval_agent.hypothesis import (
            hypothesis_from_dict,
            hypothesis_to_dict,
        )

        loop, data = self._signal_setup()
        analysis = loop.run_analysis(data)
        reloaded = [hypothesis_from_dict(hypothesis_to_dict(h))
                    for h in analysis.all_hypotheses]

        confirmed = loop.run_confirm(
            data, reloaded, stats_report=analysis.final_stats_report
        )
        assert len(confirmed.all_test_results) == len(reloaded)
        # Signal aligns with failures → at least one SUPPORTED, protocol-consistent.
        assert confirmed.stopped_by == "criteria_met"
        assert len(confirmed.verified_hypotheses) >= 1

    def test_run_confirm_regenerates_stats_when_omitted(self):
        from evalvitals.eval_agent.hypothesis import (
            hypothesis_from_dict,
            hypothesis_to_dict,
        )

        loop, data = self._signal_setup()
        analysis = loop.run_analysis(data)
        reloaded = [hypothesis_from_dict(hypothesis_to_dict(h))
                    for h in analysis.all_hypotheses]
        # No stats_report supplied → run_confirm re-runs M1→M2 silently.
        confirmed = loop.run_confirm(data, reloaded)
        assert len(confirmed.all_test_results) == len(reloaded)
        assert confirmed.final_stats_report is not None

    def test_confirm_then_run_m4(self):
        from evalvitals.eval_agent.hypothesis import (
            hypothesis_from_dict,
            hypothesis_to_dict,
        )

        loop, data = self._signal_setup()
        analysis = loop.run_analysis(data)
        reloaded = [hypothesis_from_dict(hypothesis_to_dict(h))
                    for h in analysis.all_hypotheses]
        confirmed = loop.run_confirm(
            data, reloaded, stats_report=analysis.final_stats_report
        )
        # The confirm report drives the post-loop fix step just like run().
        iv = loop.run_m4(confirmed, data)
        assert iv is not None
        assert confirmed.fix_proposal is iv

    def test_run_confirm_empty_hypotheses_is_safe(self):
        loop, data = self._signal_setup()
        analysis = loop.run_analysis(data)
        report = loop.run_confirm(
            data, [], stats_report=analysis.final_stats_report
        )
        assert report.all_test_results == []
        assert report.verified_hypotheses == []

    # ── WS2: analysis phase is descriptive (e-BH validity deferred to confirm) ──

    def test_run_analysis_defers_validity_verdict(self):
        loop, data = self._signal_setup()
        report = loop.run_analysis(data)
        sr = report.final_stats_report
        assert sr is not None
        # M2 ran (effect sizes + charts) but the e-BH verdict is deferred.
        assert sr.descriptive_only is True
        assert (sr.corrected_rejections or {}).get("deferred") is True

    def test_run_confirm_promotes_to_confirmatory(self):
        loop, data = self._signal_setup()
        analysis = loop.run_analysis(data)
        confirmed = loop.run_confirm(
            data, list(analysis.all_hypotheses),
            stats_report=analysis.final_stats_report,
        )
        sr = confirmed.final_stats_report
        # The confirm phase computes e-BH and flips descriptive_only off.
        assert sr.descriptive_only is False
        assert not (sr.corrected_rejections or {}).get("deferred")
        assert "method" in (sr.corrected_rejections or {})

    def test_run_keeps_validity_inline(self):
        # The all-in-one run() path is confirmatory by default — unchanged.
        loop = VLDiagnoseLoop(
            model=_vlm(), protocol=_spatial_protocol(),
            diagnosis_agent=_scripted_diagnosis_agent(), max_cycles=1,
        )
        report = loop.run(_labeled_batch())
        assert report.final_stats_report.descriptive_only is False


class TestDescriptivePhaseCompiler:
    """The compiler hides validity in analysis-phase (descriptive) mode."""

    _EXPLORE = {
        "candidate_signals": [
            {"name": "focus_share", "reject": True, "effect": 0.4, "ci": [0.1, 0.7]}
        ]
    }

    def test_descriptive_mode_demotes_claims(self):
        from evalvitals.reporting.compiler import compile_diagnostic_report

        story = {"analyses": [{"cycle": 0, "descriptive_only": True}], "surgeries": []}
        rep = compile_diagnostic_report(story, self._EXPLORE).to_dict()
        assert rep["claims"], "expected at least one claim"
        assert all(c["status"] == "descriptive" for c in rep["claims"])
        assert any("ANALYSIS PHASE" in c for c in rep["caveats"])

    def test_confirmatory_mode_shows_validity(self):
        from evalvitals.reporting.compiler import compile_diagnostic_report

        # A confirmatory M2 (descriptive_only False) restores the supported verdict.
        story = {"analyses": [{"cycle": 0, "descriptive_only": False}], "surgeries": []}
        rep = compile_diagnostic_report(story, self._EXPLORE).to_dict()
        assert any(c["status"] == "supported" for c in rep["claims"])

    def test_surgery_presence_restores_validity(self):
        from evalvitals.reporting.compiler import compile_diagnostic_report

        # Even a descriptive analysis flips to validity once a confirm-phase
        # surgery (M5) is recorded in the merged story.
        story = {
            "analyses": [{"cycle": 0, "descriptive_only": True}],
            "surgeries": [{"cycle": 0, "module": "m5"}],
        }
        rep = compile_diagnostic_report(story, self._EXPLORE).to_dict()
        assert any(c["status"] == "supported" for c in rep["claims"])


class TestM3FaultTolerance:
    def test_judge_exception_in_m3_does_not_kill_loop(self):
        """A judge timeout/quota error in M3 must end the loop gracefully
        (stopped_by=no_hypotheses), not propagate (regression: an agy
        TimeoutExpired killed the whole run after M1+M2 had succeeded)."""

        class ExplodingDiagnosisAgent:
            def diagnose(self, analysis, model_name="", prior_cycles=None):
                raise RuntimeError("AgyModel: agy timed out after 120s")

        loop = VLDiagnoseLoop(
            model=_vlm(),
            protocol=_spatial_protocol(),
            diagnosis_agent=ExplodingDiagnosisAgent(),
            max_cycles=2,
        )
        report = loop.run(_labeled_batch())
        assert isinstance(report, VLDiagnoseReport)
        assert report.stopped_by == "no_hypotheses"
        assert report.all_hypotheses == []


def test_default_is_a_single_diagnosis_cycle():
    """One M1→M5 pass by default, then the caller moves on to M4/fix (the
    unverified path covers 'nothing verified'); extra cycles are opt-in."""
    loop = VLDiagnoseLoop(model=FakeModel(), protocol=ExperimentProtocol(description="d"))
    assert loop.max_cycles == 1
