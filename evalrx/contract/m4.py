"""M4 — hypothesis testing: adjudicate M3's claims against corrected statistics.

M4 mostly does not compute. It routes a hypothesis's ``test_design`` to the M2
results that bear on it, admits only results that survived multiplicity
correction, reads the sign of the effect, and applies a second, independent gate:
is this claim even about what the user asked to investigate.

SUPPORTED requires BOTH gates. Either alone is a different, weaker statement.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from evalrx.contract.common import (
    CaseBatchRef,
    EvidenceGrade,
    HypothesisStatus,
    StageEnvelope,
    WireModel,
)
from evalrx.contract.m1 import ProtocolWire

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class HypothesisTestInput(WireModel):
    """M4's inputs.

    ``split`` is recorded because it changes what the verdict MEANS: testing on
    the same cases the hypotheses were generated from is an internal consistency
    check, not out-of-sample evidence. A reader cannot infer this from the numbers
    and must not be left to guess.
    """

    hypotheses_ref: str = Field(description="Path to the M3 DiagnosisOutput.")
    stats_report_ref: str = Field(description="Path to the M2 StatsReportWire (post-correction).")
    data: CaseBatchRef
    protocol: ProtocolWire | None = Field(
        default=None, description="None => every hypothesis is assumed protocol-consistent."
    )
    split: Literal["explore", "confirm", "all"] = Field(
        default="explore",
        description="Which partition M4 actually ran on. 'explore' means the hypotheses were "
                    "generated on these same cases — an in-sample adjudication, not held-out.",
    )
    alpha: float = 0.05


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class TestEvidence(WireModel):
    """The statistics behind one verdict, with its provenance."""

    source: Literal["m2_stats_results", "fallback_per_case", "none"] = "none"
    chosen_tool: str | None = None
    routed_by: str = Field(default="", description="How test_design resolved to this result.")
    consulted_tools: list[str] = Field(default_factory=list)
    ci: tuple[float, float] | None = None
    e_value: float | None = None
    fdr_corrected: bool | None = Field(default=None, description="The field that actually decided this.")
    underpowered: bool = False
    n_signal: int | None = None
    n_control: int | None = None
    fail_rate_signal: float | None = None
    fail_rate_control: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class HypothesisTestResultWire(WireModel):
    """One hypothesis's verdict.

    ``effect_size`` is signal-group minus control-group fail rate, so its SIGN is
    the verdict's direction: positive supports the claim, negative refutes it.
    A refutation is a real, useful outcome — not a failure to conclude.
    """

    hypothesis_id: str = Field(min_length=1, description="Points into the M3 DiagnosisOutput.")
    status: HypothesisStatus
    test_name: str = Field(description="e.g. 'signal_label_assoc' / 'fail_rate_comparison'.")
    effect_size: float | None = None
    confidence: float = Field(ge=0.0, le=1.0, description="Geometric mean of evidence gap, "
                                                          "sample adequacy, control cleanliness.")
    evidence_grade: EvidenceGrade = EvidenceGrade.OBSERVATIONAL
    is_consistent_with_protocol: bool
    verdict: str = Field(description="One-line natural-language summary.")
    evidence: TestEvidence = Field(default_factory=TestEvidence)

    @model_validator(mode="after")
    def _supported_needs_both_gates(self) -> "HypothesisTestResultWire":
        # The whole point of the second gate: a statistically real claim about
        # something the user never asked about is not a supported diagnosis.
        if self.status == HypothesisStatus.SUPPORTED and not self.is_consistent_with_protocol:
            raise ValueError(
                "SUPPORTED requires both the statistical test AND protocol consistency; "
                "mark it INCONCLUSIVE and record the inconsistency in `verdict`"
            )
        if self.status == HypothesisStatus.SUPPORTED and self.evidence_grade == EvidenceGrade.NONE:
            raise ValueError("SUPPORTED with evidence_grade=none has nothing backing it")
        return self


class HypothesisTestOutput(StageEnvelope):
    """M4's serialized output — one result per hypothesis, plus the loop gate."""

    split: Literal["explore", "confirm", "all"] = "explore"
    results: list[HypothesisTestResultWire] = Field(default_factory=list)
    stopping_criteria_met: bool = Field(
        default=False,
        description="Whether the loop may stop on this cycle's evidence. Not simply "
                    "`verified() != []` — the depth-tiered rule also weighs evidence grade.",
    )

    def verified(self) -> list[str]:
        """Hypothesis ids M5 may operate on, strongest first.

        Derived, never stored: a persisted copy can disagree with `results`, and
        then nothing tells a reader which one the loop actually acted on.
        """
        ok = [r for r in self.results
              if r.status == HypothesisStatus.SUPPORTED and r.is_consistent_with_protocol]
        return [r.hypothesis_id for r in sorted(ok, key=lambda r: -r.confidence)]


__all__ = ["HypothesisTestInput", "TestEvidence", "HypothesisTestResultWire", "HypothesisTestOutput"]
