"""What the report's full-record view shows for each stage.

``ReportData.stage_detail`` is compiled by :mod:`evalrx.reporting.dynamic` from
a run's logs, one block per stage, and the report UI's per-stage views read it.
These models are its shape; the TypeScript the UI imports is generated from
them (``python -m evalrx.contract.export``).

The compiler builds plain dicts and never validates against these models at
run time -- a report must still render when the ``contract`` extra is not
installed. ``tests/test_contract/test_stage_detail.py`` validates real and
synthetic reports against them instead, so a key the compiler adds without
declaring it here fails there.

Two kinds of model:

* closed (:class:`WireModel`) -- shapes the compiler builds itself. Every key it
  writes is declared, so an undeclared one is a test failure, not drift.
* open (:class:`OpenWireModel`) -- records copied through from the run's logs
  (hypotheses, held-out results, repair recommendations, ...). Their keys are
  whatever the producer wrote; only the ones the report reads are declared.

Older payloads (an uploaded archive, a report published before a field
existed) can lack any of these keys, so a reader must treat each block as
partial.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from evalrx.contract.common import OpenWireModel, WireModel

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class FigureView(WireModel):
    """A chart the run saved, with the reading written for it."""

    id: str
    title: str
    question: str
    path: str
    reading: str
    do_not_infer: str
    disposition: str
    not_promoted_reason: str
    data_uri: str | None = Field(
        default=None, description="The image inlined, on a portable export only (`evalrx report`).")


class StatView(WireModel):
    """One statistical screen from M2, labelled for a reader."""

    label: str
    raw_signal: str
    groups: list[str]
    effect: float | None
    ci: list[float | None] | None
    reject: bool
    underpowered: bool
    summary: str
    p_value: float | None
    tool: str | None
    e_value: float | None
    fail_rate_signal: float | None
    fail_rate_control: float | None


class CandidateSignalView(OpenWireModel):
    """A signal the explorer proposed for confirmation, as it recorded it."""


class OperationStep(WireModel):
    """One tool call from an agent's recorded trajectory."""

    order: int
    action: str
    raw_action: str
    parameters: dict[str, Any]
    thought: str


class OperationExample(WireModel):
    """The tool calls one case's agent trajectory actually made."""

    id: str
    case_id: str
    input: str
    expected: Any
    observed: Any
    outcome: str
    media_ids: list[str]
    steps: list[OperationStep]
    plain_reading: str


# ---------------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------------


class ProbeMetric(WireModel):
    """One headline number an analyzer reported, already formatted."""

    label: str
    value: str | None


class ProbeView(WireModel):
    """One M1 check (analyzer) and what it measured."""

    id: str
    title: str
    display_name: str
    raw_name: str
    question: str
    description: str
    n_cases: int | None = Field(
        description="None when the analyzer reported no case count at all: unknown, not zero.")
    metrics: list[ProbeMetric]
    finding_summary: dict[str, Any]
    raw_finding_summary: dict[str, Any]
    sample_rows: list[dict[str, Any]]
    n_sample_rows: int


class ModelCallView(WireModel):
    """One model call an M1 analyzer made: what it sent and what came back."""

    seq: int | None
    cycle: int | None
    method: str
    case_id: str | None
    batch_case_ids: list[str]
    duration_sec: float | None
    prompt: str
    output: str
    error: str | None


class ProbeExample(OpenWireModel):
    """One case walked through an M1 check -- logged by the probe, or rebuilt
    from a per-case measurement on older runs."""

    id: str | None = None
    kind: str | None = None
    case_id: str | None = None
    probe_title: str | None = None
    probe_question: str | None = None
    input: str | None = None
    baseline_output: Any = None
    expected: Any = None
    outcome: str | None = None
    media_ids: list[str] | None = None
    check_result: dict[str, Any] | None = None
    plain_reading: str | None = None
    evidence_scope: str | None = None


class M1StageDetail(WireModel):
    duration: float | None
    probes: list[ProbeView]
    n_probes: int
    n_measured: int
    examples: list[ProbeExample]
    operations: list[OperationExample]
    calls: dict[str, list[ModelCallView]] = Field(
        description="Each analyzer's model calls, oldest first, capped per analyzer.")
    n_calls: int


# ---------------------------------------------------------------------------
# M2
# ---------------------------------------------------------------------------


class TakeawayView(WireModel):
    """One ranked finding the explorer wrote, and the charts it cites."""

    plain_title: str | None
    title: str | None
    analysis: str | None
    caveat: str | None
    chart_names: list[str] | None
    table_names: list[str] | None


class ExploreAgentView(WireModel):
    """What the explore coding agent left: the analysis it wrote and ran.

    ``transcript`` is its CLI trajectory, cut to the report's budget; the run
    directory keeps the whole file.
    """

    code: str | None
    stdout: str | None
    stderr: str | None
    transcript: str | None
    transcript_truncated: bool
    audit: Any


class M2StageDetail(WireModel):
    mode: str
    conclusion: str
    narrative: str
    stats: list[StatView]
    takeaways: list[TakeawayView]
    figures: list[FigureView]
    observations: list[Any]
    caveats: list[Any]
    candidate_signals: list[CandidateSignalView]
    recommended_tests: list[Any]
    agent: ExploreAgentView | None


# ---------------------------------------------------------------------------
# M3
# ---------------------------------------------------------------------------


class HypothesisView(OpenWireModel):
    """A hypothesis M3 proposed, as the run recorded it."""

    id: str | None = None
    statement: str | None = None
    plain_statement: str | None = None
    hypothesis: str | None = None
    failure_mode: str | None = None
    basis: str | None = None
    test_design: str | None = None
    expected_association: str | None = None
    status: str | None = None


class M3StageDetail(WireModel):
    mode: str
    hypotheses: list[HypothesisView]
    candidate_signals: list[CandidateSignalView]
    recommended_tests: list[Any]
    evidence_figures: list[FigureView]
    evidence_stats: list[StatView]


# ---------------------------------------------------------------------------
# M4
# ---------------------------------------------------------------------------


class ValidationEvidence(OpenWireModel):
    """The statistics behind one held-out verdict, as the run recorded them."""

    ci: list[float | None] | None = None
    effect_size: float | None = None
    evidence_grade: str | None = None
    m4_verdict: str | None = None
    fdr: dict[str, Any] | None = None


class ValidationEventView(OpenWireModel):
    """M4's raw log record. Verdict cards fall back to it when a result
    lacks a field."""

    hypothesis: str | None = None
    status: str | None = None
    confidence_score: float | None = None
    evidence: ValidationEvidence | None = None


class ValidationResultView(OpenWireModel):
    """One held-out verdict, as the run recorded it."""

    hypothesis_id: str | None = None
    hypothesis: str | None = None
    statement: str | None = None
    plain_statement: str | None = None
    status: str | None = None
    verdict: str | None = None
    effect_size: float | None = None
    confidence: float | None = None
    ci: list[float | None] | None = None
    evidence_grade: str | None = None
    protocol_consistent: bool | None = None
    evidence: ValidationEvidence | None = None


class ValidationExample(OpenWireModel):
    """One held-out check made concrete: a logged validation case, or the
    aggregate test itself when the run kept no case-level example."""

    id: str | None = None
    kind: str | None = None
    hypothesis: str | None = None
    test: str | None = None
    status: str | None = None
    effect: float | None = None
    interval: list[float | None] | None = None
    verdict: str | None = None
    case_id: str | None = None
    input: str | None = None
    baseline_output: Any = None
    expected: Any = None
    media_ids: list[str] | None = None
    plain_reading: str | None = None
    evidence_scope: str | None = None


class M4StageDetail(WireModel):
    ran: bool
    mode: str
    results: list[ValidationResultView]
    event: ValidationEventView
    examples: list[ValidationExample]


# ---------------------------------------------------------------------------
# M5
# ---------------------------------------------------------------------------


class RepairCandidateView(WireModel):
    """One repair candidate and its paired comparison against the baseline.
    Every field is null when the run recorded no such candidate."""

    tier: str | None
    name: str | None = Field(
        description="A join key, not language; `ref` and `headline` are what a reader sees.")
    kind: str | None
    ref: str
    headline: str
    source: str | None
    verdict: str | None
    n_pairs: int | None
    n_baseline_correct: int | None
    n_candidate_correct: int | None
    n_fixed: int | None
    n_broken: int | None
    effect: float | None
    e_value: float | None
    e_threshold: float | None
    coverage: float | None
    reject: bool | None
    fixed: bool | None
    n_model_independent: int | None
    n_unstable: int | None
    summary: str | None
    payload: dict[str, Any]
    fixed_cases: list[str]
    broken_cases: list[str]


class RepairRecommendation(OpenWireModel):
    """The fix agent's own verdict on the run, and the sentence behind it."""

    action: str | None = None
    reason: str | None = None


class RepairExample(WireModel):
    """One case before and after the strongest repair -- confirmed or not."""

    id: str
    case_id: str
    kind: str = Field(
        description='"fixed", or "broken" when the attempt flipped no case to correct.')
    confirmed: bool
    repair_name: str
    input: str
    expected: Any
    baseline_output: Any
    repaired_output: Any
    media_ids: list[str]
    plain_reading: str
    baseline_available: bool


class RepairOperation(WireModel):
    action: str
    raw_action: str
    parameters: dict[str, Any]


class RepairOperationPreview(WireModel):
    """The image operations a repair declares, shown on a real case. A preview
    of the transform, not a recorded before/after."""

    id: str
    candidate: str | None
    case_id: str
    input: str
    expected: Any
    observed: Any
    media_ids: list[str]
    executed: bool
    operations: list[RepairOperation]


class M5StageDetail(WireModel):
    ran: bool
    fixed: bool
    skipped: bool
    skip_reason: str
    skip_detail: str
    candidates: list[RepairCandidateView]
    confirmation: RepairCandidateView
    best: RepairCandidateView
    recommendation: RepairRecommendation | None
    examples: list[RepairExample]
    operation_previews: list[RepairOperationPreview]
    surgeries: list[dict[str, Any]]
    prompt_template: str


class StageDetail(WireModel):
    """``ReportData.stage_detail``: the full record behind each stage."""

    m1: M1StageDetail
    m2: M2StageDetail
    m3: M3StageDetail
    m4: M4StageDetail
    m5: M5StageDetail


__all__ = [
    "CandidateSignalView",
    "ExploreAgentView",
    "FigureView",
    "HypothesisView",
    "M1StageDetail",
    "M2StageDetail",
    "M3StageDetail",
    "M4StageDetail",
    "M5StageDetail",
    "ModelCallView",
    "OperationExample",
    "OperationStep",
    "ProbeExample",
    "ProbeMetric",
    "ProbeView",
    "RepairCandidateView",
    "RepairExample",
    "RepairOperation",
    "RepairOperationPreview",
    "RepairRecommendation",
    "StageDetail",
    "StatView",
    "TakeawayView",
    "ValidationEventView",
    "ValidationEvidence",
    "ValidationExample",
    "ValidationResultView",
]
