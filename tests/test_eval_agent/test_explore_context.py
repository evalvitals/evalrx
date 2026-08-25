"""Phase B/C — explorer mechanism notes (charts/observations) flow into M3 only.

Covers the core guardrail (DESIGN_m3_charts.md §5): an ``ExploreContext`` is
DESCRIPTIVE and UNCONFIRMED — it enters the M3 hypothesis-proposal prompt (and
its charts are attached as images), but it NEVER reaches the M2 confirmatory
family, M5 testing, or the fix gate.
"""

from __future__ import annotations

import inspect

from evalvitals.analysis.stats_agent import StatsAnalysisReport
from evalvitals.core.capability import Capability
from evalvitals.eval_agent.stages.diagnosis import (
    DiagnosisAgent,
    ExploreContext,
    _extract_referenced,
    _format_explore_section,
)
from tests.conftest import FakeModel

# ---------------------------------------------------------------------------
# ExploreContext construction
# ---------------------------------------------------------------------------

def test_from_report_returns_none_when_no_descriptive_content():
    assert ExploreContext.from_report(None) is None
    assert ExploreContext.from_report({}) is None
    assert ExploreContext.from_report({"candidate_signals": [{"name": "x"}]}) is None


def test_from_report_collects_observations_charts_caveats():
    ctx = ExploreContext.from_report({
        "observations": ["fail cases have small objects"],
        "charts": [{"title": "Size by label", "figure_path": "/tmp/x.png"},
                   {"title": "no-fig", "kind": "bar"}],
        "caveats": ["probe1 ~ label, circular"],
    })
    assert ctx is not None
    assert ctx.observations and ctx.caveats
    assert ctx.figure_paths == ["/tmp/x.png"]   # only charts with a rendered PNG
    assert not ctx.is_empty


def test_format_section_is_strongly_labelled_unconfirmed():
    ctx = ExploreContext(observations=["obs A"], charts=[{"title": "C1", "figure_path": "/p.png"}],
                         caveats=["be careful"])
    sec = _format_explore_section(ctx)
    assert "UNCONFIRMED" in sec
    assert "obs A" in sec and "C1" in sec and "be careful" in sec
    # empty context contributes nothing to the prompt
    assert _format_explore_section(None) == ""
    assert _format_explore_section(ExploreContext()) == ""


# ---------------------------------------------------------------------------
# M3 consumes the explore context (prompt + images + provenance)
# ---------------------------------------------------------------------------

class _ImageJudge(FakeModel):
    """A judge whose generate() explicitly declares images= (so diagnose attaches
    figures) and records every call."""

    def __init__(self) -> None:
        super().__init__(capabilities={Capability.GENERATE})
        self.calls: list[dict] = []

    def generate(self, inputs, images=None, **kw):  # noqa: D401
        self.calls.append({"prompt": str(inputs), "images": list(images or [])})
        # M3 cites the explore chart it drew the hypothesis from (by title); the
        # critic pass keeps it. _extract_referenced reads this output for provenance.
        return ("HYPOTHESIS: model ignores the image, consistent with the ObjSize by "
                "label chart\nFAILURE_MODE: weak_visual_grounding\n"
                "KEEP: model ignores the image, consistent with the ObjSize by label chart")


def _stats_report_with_figure(fig: str) -> StatsAnalysisReport:
    return StatsAnalysisReport(
        model_name="vlm",
        findings=[],
        severity="none",
        narrative="No anomalies.",
        raw_results={},
        conclusion="Answers from language priors.",
        evidence_chain=["attention near-uniform"],
        stats_results=[],
        figures=[fig],
    )


def test_diagnose_injects_explore_section_and_merges_images(tmp_path):
    m2_fig = tmp_path / "m2_forest.png"
    ex_fig = tmp_path / "explore_size.png"
    m2_fig.write_bytes(b"\x89PNG\r\n")
    ex_fig.write_bytes(b"\x89PNG\r\n")

    ctx = ExploreContext(
        observations=["FAIL cases skew to small objects"],
        charts=[{"title": "ObjSize by label", "figure_path": str(ex_fig),
                 "description": "bar of size by label"}],
        caveats=["explore split only"],
    )
    judge = _ImageJudge()
    diag = DiagnosisAgent(judge=judge).diagnose(
        _stats_report_with_figure(str(m2_fig)), explore_context=ctx
    )

    prompt = judge.calls[0]["prompt"]
    assert "EXPLORATORY MECHANISM NOTES" in prompt
    assert "FAIL cases skew to small objects" in prompt
    assert "ObjSize by label" in prompt
    # both the M2 confirmatory figure and the explore chart are attached to M3
    imgs = [str(p) for p in judge.calls[0]["images"]]
    assert str(m2_fig) in imgs and str(ex_fig) in imgs
    # provenance recorded on the result
    assert diag.explore_context_used is True
    assert "ObjSize by label" in diag.referenced_charts


def test_diagnose_without_explore_context_is_unchanged():
    judge = _ImageJudge()
    diag = DiagnosisAgent(judge=judge).diagnose(_stats_report_with_figure("/nonexistent.png"))
    assert "EXPLORATORY MECHANISM NOTES" not in judge.calls[0]["prompt"]
    assert diag.explore_context_used is False
    assert diag.referenced_charts == []


def test_extract_referenced_only_matches_mentioned_titles():
    ctx = ExploreContext(charts=[{"title": "Alpha plot"}, {"title": "Beta plot"}])
    assert _extract_referenced("I will test the Alpha plot idea", ctx) == ["Alpha plot"]
    assert _extract_referenced("nothing here", ctx) == []


# ---------------------------------------------------------------------------
# Double-blind guardrail: explore_context reaches M3 ONLY
# ---------------------------------------------------------------------------

def test_only_m3_diagnose_accepts_explore_context():
    from evalvitals.analysis.stats_agent import StatsAnalysisAgent
    from evalvitals.eval_agent.stages.fix_agent import FixAgent
    from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTester

    assert "explore_context" in inspect.signature(DiagnosisAgent.diagnose).parameters

    # M2 / M5 / fix must NOT take an explore_context anywhere in their public API.
    for cls in (StatsAnalysisAgent, HypothesisTester, FixAgent):
        for _name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
            params = inspect.signature(member).parameters
            assert "explore_context" not in params, f"{cls.__name__}.{_name} leaks explore_context"


# ---------------------------------------------------------------------------
# Loop wiring: coercion + optional-context dispatch
# ---------------------------------------------------------------------------

def test_coerce_explore_context_accepts_dict_ctx_and_none():
    from evalvitals.eval_agent.run_metadata import _coerce_explore_context

    ctx = _coerce_explore_context({"observations": ["a"]})
    assert isinstance(ctx, ExploreContext)
    assert _coerce_explore_context(ctx) is ctx                 # passthrough
    assert _coerce_explore_context(None) is None
    assert _coerce_explore_context({}) is None                 # empty -> None
    assert _coerce_explore_context(ExploreContext()) is None   # empty ctx -> None
    assert _coerce_explore_context(12345) is None              # garbage -> None


def test_dispatch_passes_context_only_to_agents_that_accept_it():
    from evalvitals.eval_agent.loop import _diagnose_with_optional_context

    ctx = ExploreContext(observations=["x"])

    class _Modern:
        def __init__(self): self.kw = None
        def diagnose(self, report, prior_cycles=None, explore_context=None):
            self.kw = {"prior_cycles": prior_cycles, "explore_context": explore_context}
            return "ok"

    class _Legacy:
        def __init__(self): self.kw = None
        def diagnose(self, report, prior_cycles=None):
            self.kw = {"prior_cycles": prior_cycles}
            return "ok"

    modern, legacy = _Modern(), _Legacy()
    _diagnose_with_optional_context(modern, "rep", None, ctx)
    _diagnose_with_optional_context(legacy, "rep", None, ctx)   # must not raise
    assert modern.kw["explore_context"] is ctx
    assert "explore_context" not in legacy.kw


def test_loop_init_stores_coerced_explore_context():
    from evalvitals.eval_agent.loop import VLDiagnoseLoop

    loop = VLDiagnoseLoop.__new__(VLDiagnoseLoop)  # avoid heavy __init__ deps
    # exercise the coercion path used by __init__
    from evalvitals.eval_agent.run_metadata import _coerce_explore_context
    loop._explore_context = _coerce_explore_context({"charts": [{"title": "C", "figure_path": "/p.png"}]})
    assert isinstance(loop._explore_context, ExploreContext)
    assert loop._explore_context.figure_paths == ["/p.png"]


# ---------------------------------------------------------------------------
# Phase C: M3 provenance is logged
# ---------------------------------------------------------------------------

def test_log_diagnosis_records_explore_provenance(tmp_path):
    import json

    from evalvitals.eval_agent.run_logger import RunLogger
    from evalvitals.eval_agent.stages.diagnosis import DiagnosisResult

    logger = RunLogger(run_dir=tmp_path / "run1")
    diag = DiagnosisResult(
        model_name="vlm",
        hypotheses=[],
        raw_judge_output="...",
        referenced_charts=["ObjSize by label"],
        explore_context_used=True,
    )
    logger.log_diagnosis(1, diag, explore_figures=["/tmp/explore_size.png"])

    lines = (tmp_path / "run1" / "run_log.jsonl").read_text().splitlines()
    entry = next(json.loads(x) for x in lines if json.loads(x).get("event") == "diagnosis")
    assert entry["referenced_charts"] == ["ObjSize by label"]
    assert entry["explore_context_used"] is True
    assert entry["explore_figures"] == ["/tmp/explore_size.png"]


def test_log_diagnosis_persists_the_critic_prompt_beside_its_response(tmp_path):
    """The critic now judges against the proposer's context + a label summary;
    a reviewer must be able to read what it was shown, not only what it said."""
    import json

    from evalvitals.eval_agent.hypothesis import Hypothesis
    from evalvitals.eval_agent.run_logger import RunLogger
    from evalvitals.eval_agent.stages.diagnosis import DiagnosisResult

    logger = RunLogger(run_dir=tmp_path / "run2")
    h1 = Hypothesis(statement="h1", target_model="vlm", predicted_failure_mode="x")
    diag = DiagnosisResult(
        model_name="vlm", hypotheses=[h1], raw_judge_output="...",
        critic_raw_output="REJECT: h1\nREASON: n=30",
        critic_prompt="adversarial reviewer ... LABEL SUMMARY ... gold=no answered=yes",
        n_critic_rejected=1,
        proposed_hypotheses=[h1],
        review_decisions=[{"statement": "h1", "decision": "reject", "reason": "n=30"}],
    )
    logger.log_diagnosis(0, diag)
    lines = (tmp_path / "run2" / "run_log.jsonl").read_text().splitlines()
    entry = next(json.loads(x) for x in lines if json.loads(x).get("event") == "diagnosis")
    # both record shapes travel together: the critic io and the review list
    assert entry["proposed_hypotheses"][0]["statement"] == "h1"
    assert entry["review"]["n_rejected"] == 1 and entry["review"]["decisions"][0]["decision"] == "reject"
    io = entry["critic_io"]
    assert io["raw_path"].endswith("c0_m3_critic.response.txt")
    assert io["prompt_path"].endswith("c0_m3_critic.prompt.txt")
    assert "LABEL SUMMARY" in (tmp_path / "run2" / io["prompt_path"]).read_text()
