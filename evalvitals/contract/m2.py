"""M2 — analysis: join measurements to outcomes, test which signals track failure.

M2 answers *what correlates*, never *why*. That boundary is load-bearing: an
in-sample association rendered with the same language as a corrected verdict is
the single most damaging mistake this pipeline can make, so the descriptive and
confirmatory states are distinguishable here at the type level
(:attr:`StatsReportWire.descriptive_only`), not by convention.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from evalvitals.contract.common import (
    ArtifactRef, CaseBatchRef, ExternalRef, JoinReport, OpenWireModel,
    StageEnvelope, WireModel,
)
from evalvitals.contract.m1 import ProtocolWire


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class AnalysisInput(WireModel):
    """M2 needs BOTH halves of the join: M1's numbers and the cases' labels.

    ``data`` is not optional in practice — without labels there is nothing to
    correlate against and M2 degrades to threshold rules only.
    """

    results_ref: str = Field(description="Path to the M1 ProbeOutput this analyzes.")
    data: CaseBatchRef
    model_name: str
    protocol: ProtocolWire | None = None
    confirmatory: bool = Field(
        default=True,
        description="False defers the e-BH validity verdict to the confirm phase; "
                    "the report then carries effect sizes and charts but NO reject decision.",
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class AnalysisFindingWire(WireModel):
    """One threshold-rule violation. The no-judge baseline signal."""

    analyzer: str
    metric: str
    value: float
    threshold: float
    direction: Literal["above", "below"]
    severity: Literal["high", "medium", "low"]
    message: str


class StatsToolResultWire(WireModel):
    """One statistical test's verdict.

    ``reject`` is the RAW, uncorrected decision and must never drive a verdict on
    its own. ``fdr_corrected`` is the family-level decision after multiplicity
    control and is the only field a downstream verdict may read. They are kept as
    separate fields precisely because collapsing them once let a p=0.14 result on
    n=3 "support" a hypothesis.
    """

    tool: str = Field(description="e.g. signal_label_assoc / mcnemar_evalue / bootstrap_diff / rank_corr")
    measured: str | None = Field(
        default=None,
        description="WHAT this result is about, in words, and unique within the report — "
                    "the label a chart axis or a table row should carry.\n\n"
                    "`tool` is the procedure, not the subject, and using it as a label "
                    "put two rows reading 'Mcnemar evalue' on a chart with no way to "
                    "tell them apart. `config['signal']` is the subject but is a machine "
                    "name, and is absent on paired tools entirely. Neither can label a "
                    "row on its own, so the producer states the label here.",
    )
    means: str | None = Field(
        default=None,
        description="What the measured quantity IS, in one sentence, taken from the "
                    "producing analyzer's `signal_docs`. Null means the analyzer did not "
                    "document that metric — render it as undocumented rather than "
                    "inventing a gloss, because a plausible wrong explanation of a "
                    "statistic is worse than an admitted missing one.",
    )
    config: dict[str, Any] = Field(default_factory=dict, description="Includes 'signal' — the routing key.")
    ok: bool
    error: str | None = None

    effect: float | None = None
    ci: tuple[float, float] | None = None
    p_value: float | None = None
    e_value: float | None = None
    underpowered: bool = False

    reject: bool = Field(default=False, description="RAW per-test decision. Not a verdict.")
    fdr_corrected: bool | None = Field(
        default=None,
        description="Family-level decision after correction. None = correction not yet run "
                    "(deferred descriptive phase) — which is NOT the same as False.",
    )
    correction_method: str | None = Field(default=None, description="e.g. 'bh' / 'ebh'.")
    correction_family: str | None = None
    analysis_key: str | None = Field(
        default=None, description="Stable per-result key, e.g. 'signal_label_assoc:attention.entropy'. "
                                  "Required for correction: one tool NAME covers many tests."
    )

    raw_reject: bool | None = Field(
        default=None, description="Pre-correction decision, kept for audit alongside `reject`."
    )
    figure_path: str | None = None

    n_signal: int | None = Field(default=None, description="Cases in the signal-present group.")
    n_control: int | None = Field(default=None, description="Cases in the signal-absent group.")
    n_measured: int | None = Field(
        default=None,
        description="Cases the producing analyzer actually measured. Analyzers measure "
                    "every case unless capped (an analyzer's max_cases or the probe "
                    "agent's max_cases_per_analyzer), so compare this with the batch size.",
    )
    n_imputed_absent: int | None = Field(
        default=None,
        description="Labeled cases with no measurement that were counted as signal-ABSENT. "
                    "Valid for a genuinely sparse flag that only lists where it fired; invalid "
                    "when the analyzer simply never looked. Nothing in the value distinguishes "
                    "the two, so the count must travel with the result.",
    )

    summary: str = ""
    details: dict[str, Any] = Field(
        default_factory=dict, description="fail rates / permutation_p / CI internals / ..."
    )

    @property
    def imputation_share(self) -> float | None:
        """Share of the tested sample that was never measured.

        Observed live (bbh_tracking7): ``step_rollout_value.recoverable`` was
        measured on 8 cases, tested on 125, and survived e-BH — 94% of its
        control group is imputed absence. A reader cannot see that from the
        effect size, the CI, or the corrected flag.
        """
        total = (self.n_signal or 0) + (self.n_control or 0)
        if not total or self.n_imputed_absent is None:
            return None
        return self.n_imputed_absent / total

    def is_mostly_imputed(self, threshold: float = 0.5) -> bool:
        share = self.imputation_share
        return share is not None and share > threshold

    def is_decisive(self, report_survivors: set[str] | None = None) -> bool:
        """Whether this result may decide a hypothesis's direction.

        Mirrors the M5 gate: raw rejection is necessary but never sufficient.
        """
        if not self.reject:
            return False
        if self.correction_method:
            return bool(self.fdr_corrected)
        if self.e_value is not None:
            return bool(self.analysis_key) and self.analysis_key in (report_survivors or set())
        return True


#: Legacy spellings this contract shipped before it was checked against the
#: producer. ``evalvitals.stats.multiplicity`` writes the hyphenated names and
#: has a fourth value the contract had no member for.
_LEGACY_CORRECTION_METHODS = {"ebh": "e-BH", "bh": "BH"}


class CorrectedRejections(WireModel):
    """Family-level multiplicity control across every tested signal.

    Testing 40 signals at alpha=0.05 yields ~2 "significant" results from luck
    alone. Without this the screen is not a screen.

    ``mixed-BH/e-BH`` is its own member rather than being folded into either
    neighbour: it says the family contained both p-values and e-values, which
    carry different validity guarantees, and a reader weighing the verdict needs
    to know that the guarantee is the weaker of the two.
    """

    method: Literal["e-BH", "BH", "mixed-BH/e-BH", "none"] = "none"
    alpha: float = 0.05
    deferred: bool = Field(
        default=False, description="True while the analysis phase withholds the verdict."
    )
    n_tested: int = 0
    rejected_result_keys: list[str] = Field(
        default_factory=list, description="analysis_key values that survived. The source of truth."
    )

    @field_validator("method", mode="before")
    @classmethod
    def _normalise_method(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _LEGACY_CORRECTION_METHODS.get(v.lower(), v)
        return v


class StatsReportWire(StageEnvelope):
    """M2's serialized report.

    Two field groups with different authority: everything above
    ``descriptive_only`` is computed by code and is authoritative; ``conclusion``
    and ``evidence_chain`` are written by the judge and are presentation only.
    Empty narrative is valid and must never be backfilled with an invented
    summary. The LLM may not write a number, and keeping the groups visibly
    apart is what makes that reviewable.

    Absent by design: the judge's prompt and raw response (already
    ``prompts/c<cycle>_m2_analysis.{prompt,response}.txt``), the tool-selection
    plan (already ``artifacts/c<cycle>_m2_stats_plan.json``), the model name and
    protocol (both in the run config and in :class:`AnalysisInput`), and the
    threshold-rule narrative (a rendering of ``findings``).
    """

    findings: list[AnalysisFindingWire] = Field(default_factory=list)
    stats_tool: Literal["threshold_rules", "llm_guided", "selected_tools", "generated"] = "threshold_rules"
    stats_results: list[StatsToolResultWire] | ExternalRef = Field(default_factory=list)
    corrected_rejections: CorrectedRejections | ExternalRef = Field(default_factory=CorrectedRejections)
    figures: list[ArtifactRef] = Field(default_factory=list)
    descriptive_only: bool = Field(
        default=False,
        description="True = effect sizes and charts only, e-BH deferred. A reader MUST NOT "
                    "render supported/rejected language while this is true.",
    )

    conclusion: str = Field(default="", description="Judge-written. Presentation only.")
    evidence_chain: list[str] = Field(default_factory=list, description="Judge-written derivation.")
    llm_fallback_reason: str = Field(
        default="",
        description="Non-empty only when the judge path was attempted AND raised. Not recoverable "
                    "from the prompt files: a fallback still writes both of them.",
    )

    raw_results_ref: str | None = Field(
        default=None,
        description="Path to the M1 ProbeOutput. The in-memory report carries the Result objects "
                    "themselves, but to_dict() drops them — M4/M5 still need per-case signals, so "
                    "the serialized contract carries a reference instead of pretending they survive.",
    )
    joins: list[JoinReport] = Field(
        default_factory=list, description="Signal<->label join health. Zero coverage is a plumbing failure."
    )

    @property
    def severity(self) -> Literal["high", "medium", "low", "none"]:
        """Worst flagged finding. Derived — a stored copy can contradict the list."""
        for level in ("high", "medium", "low"):
            if any(f.severity == level for f in self.findings):
                return level  # type: ignore[return-value]
        return "none"

    @model_validator(mode="after")
    def _measured_labels_are_distinct(self) -> "StatsReportWire":
        """Two rows labelled the same are two rows a reader cannot tell apart.

        Deduplicated rather than rejected: a colliding label is a presentation
        fault, and dropping the whole M2 payload over one would lose the
        statistics too. The suffix is the tool, then an ordinal — enough to
        separate them without inventing meaning.
        """
        rows = self.stats_results
        if not isinstance(rows, list):
            return self
        seen: dict[str, int] = {}
        for r in rows:
            if not r.measured:
                continue
            if r.measured in seen:
                seen[r.measured] += 1
                r.measured = f"{r.measured} ({r.tool}, {seen[r.measured]})"
            else:
                seen[r.measured] = 1
        return self

    @model_validator(mode="after")
    def _descriptive_has_no_verdict(self) -> "StatsReportWire":
        corr = self.corrected_rejections
        if self.descriptive_only and isinstance(corr, CorrectedRejections):
            if corr.rejected_result_keys and not corr.deferred:
                raise ValueError(
                    "descriptive_only=True but corrected_rejections carries survivors; "
                    "a descriptive report must not ship a validity verdict"
                )
        return self


class ExploreContextWire(WireModel):
    """Descriptive EDA notes from the optional explore side-path.

    Never authoritative. It enters the M3 prompt and the dashboard and nothing
    else — not M2's tested family, not M5, not the fix gate. It shapes WHICH
    hypotheses get proposed, never WHETHER one is true.
    """

    observations: list[str] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    source: str = "lambda_explorer"
    authoritative: Literal[False] = False


__all__ = [
    "AnalysisInput", "AnalysisFindingWire", "StatsToolResultWire",
    "CorrectedRejections", "StatsReportWire", "ExploreContextWire",
]
