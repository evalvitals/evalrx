"""M3 — diagnosis: turn correlations into falsifiable mechanism claims.

This is the one stage where the LLM produces the payload rather than the prose.
That is exactly why its output must be machine-checkable: a hypothesis carries a
``test_design`` naming the signal that would confirm or refute it, so M5 can
route deterministically instead of re-reading the sentence.
"""

from __future__ import annotations

import re

from typing import Any, Literal

from pydantic import Field, field_validator

from evalvitals.contract.common import HypothesisStatus, StageEnvelope, WireModel
from evalvitals.contract.m2 import ExploreContextWire


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class DiagnosisInput(WireModel):
    """M3 reads M2's report — both the numbers and the narrative."""

    analysis_ref: str = Field(description="Path to the M2 StatsReportWire.")
    prior_cycles: list[dict[str, Any]] = Field(
        default_factory=list, description="Summaries of earlier cycles, so the judge avoids re-proposing."
    )
    explore_context: ExploreContextWire | None = None
    failure_modes: list[str] = Field(
        default_factory=list, description="Descriptive-only clustering tags."
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

#: A ``<analyzer>.<metric>`` reference — the form M5 resolves against M2's signals.
_SIGNAL_REF = re.compile(r"\b[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\b")

#: Non-signal test designs M5 knows how to route. Extend this together with the
#: router; a directive listed here and unhandled there is worse than a rejection.
TEST_DESIGN_DIRECTIVES = frozenset({
    "prompt_contrast",
    "analyzer_params",
    "strategy_contrast",
    "paired_rerun",
})


class HypothesisWire(WireModel):
    """One falsifiable claim about *why* the model fails.

    ``statement`` is the mechanism; ``test_design`` is the machine-readable
    commitment to how it could be wrong. A hypothesis without the latter cannot
    be tested and is therefore not a hypothesis — M5 has nothing to route on and
    M1 has nothing to re-probe.
    """

    id: str = Field(min_length=1, description="Stable id. Never slugify the statement and assume uniqueness.")
    statement: str = Field(min_length=10, description="Technical mechanism claim.")
    target_model: str
    predicted_failure_mode: str = Field(min_length=2)
    test_design: str = Field(
        min_length=1,
        description="Signal or contrast that would decide this, e.g. 'attention.image_token_ratio' "
                    "or 'prompt_contrast describe_first'. M5 routes on it; M1 re-probes on it.",
    )
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    parent_id: str | None = Field(default=None, description="Set when mutated from an earlier hypothesis.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("test_design")
    @classmethod
    def _must_be_routable(cls, v: str) -> str:
        """Reject a test_design M5 cannot route on.

        Routability is the whole point: M5 resolves this string against M2's
        signals and M1 re-probes on it. "investigate further" resolves to
        nothing, and a hypothesis nothing can decide is not falsifiable — it
        will come back INCONCLUSIVE forever and quietly hold the loop open.
        """
        if _SIGNAL_REF.search(v):
            return v
        if any(v.strip().startswith(d) for d in TEST_DESIGN_DIRECTIVES):
            return v
        raise ValueError(
            f"test_design must name a signal ('<analyzer>.<metric>') or start with one of "
            f"{sorted(TEST_DESIGN_DIRECTIVES)}; got {v[:60]!r}"
        )


class DiagnosisOutput(StageEnvelope):
    """M3's serialized output: the hypotheses, and nothing that is already on disk.

    Zero hypotheses is a completed, empty result (state EMPTY or ABSTAINED) — not
    an error. Rendering it red teaches readers to distrust a correct abstention.

    Judge I/O is deliberately absent. M3 is the one stage an LLM writes the
    payload for, so its prompt and verbatim response do have to be recoverable —
    but they are already persisted per cycle as
    ``prompts/c<cycle>_m3_diagnosis.{prompt,response}.txt``, and the findings the
    judge was shown are the M2 report named by :attr:`DiagnosisInput.analysis_ref`.
    Carrying either inline here would be a second copy that can disagree with the
    first.
    """

    hypotheses: list[HypothesisWire] = Field(default_factory=list)
    @field_validator("hypotheses")
    @classmethod
    def _ids_unique(cls, v: list[HypothesisWire]) -> list[HypothesisWire]:
        ids = [h.id for h in v]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate hypothesis ids make M5 verdicts ambiguous")
        return v


__all__ = ["TEST_DESIGN_DIRECTIVES", "DiagnosisInput", "HypothesisWire", "DiagnosisOutput"]
