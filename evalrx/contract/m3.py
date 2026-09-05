"""M3 — diagnosis: turn correlations into falsifiable mechanism claims.

This is the one stage where the LLM produces the payload rather than the prose.
That is exactly why its output must be machine-checkable: a hypothesis carries a
``test_design`` naming the signal that would confirm or refute it, so M4 can
route deterministically instead of re-reading the sentence.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import Field, field_validator

from evalrx.contract.common import HypothesisStatus, StageEnvelope, WireModel
from evalrx.contract.m2 import ExploreContextWire

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

#: A ``<analyzer>.<metric>`` reference — the form M4 resolves against M2's signals.
_SIGNAL_REF = re.compile(r"\b[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\b")

#: Non-signal test designs M4 knows how to route. Extend this together with the
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
    be tested and is therefore not a hypothesis — M4 has nothing to route on and
    M1 has nothing to re-probe.
    """

    id: str = Field(min_length=1, description="Stable id. Never slugify the statement and assume uniqueness.")
    statement: str = Field(min_length=10, description="Technical mechanism claim.")
    plain_statement: str = Field(
        default="",
        description="The SAME claim in one everyday sentence, for a reader who runs "
                    "evaluations and does no statistics. Not a second, softer claim — a "
                    "second rendering of this one, checked host-side against a jargon "
                    "list before it is accepted (see analysis.plain_language). Empty "
                    "when the producer wrote none; a consumer should then show "
                    "`statement` rather than paraphrase it into something the run "
                    "never said.",
    )
    target_model: str
    predicted_failure_mode: str = Field(min_length=2)
    test_design: str = Field(
        default="",
        description="Signal or contrast that would decide this, e.g. 'attention.image_token_ratio' "
                    "or 'prompt_contrast describe_first'. M4 routes on it; M1 re-probes on it. "
                    "EMPTY means the judge proposed none — see `is_routable`.",
    )
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    parent_id: str | None = Field(default=None, description="Set when mutated from an earlier hypothesis.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_routable(self) -> bool:
        """Whether M4 can resolve this design against measured signals.

        Mirrors what ``hypothesis_tester`` actually extracts — an identifier
        anywhere in the text, or one of the directives — rather than demanding a
        shape. The strict version required the string to BE a bare
        ``<analyzer>.<metric>`` or to START with a directive, and a strong judge
        does not write like that: on the Music-AVQA run, Opus at high effort
        produced three designs that named their analyzers and metrics inside a
        paragraph of interventional protocol. M4 routed all three
        (``routed_by="test_design"``, two at INTERVENTION grade) and the contract
        rejected the whole M3 payload for the formatting.

        A validator stricter than the consumer it protects does not prevent
        anything; it just discards good work. So routability is now REPORTED, and
        the three states stay distinguishable: empty (no test proposed),
        non-empty but nothing resolvable (a human can act on it, M4 cannot), and
        routable.
        """
        text = self.test_design.strip()
        if not text:
            return False
        if _SIGNAL_REF.search(text.lower()):
            return True
        return any(d in text for d in TEST_DESIGN_DIRECTIVES)

    @property
    def is_proposed(self) -> bool:
        """Whether the judge proposed a test at all, routable or not."""
        return bool(self.test_design.strip())


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

    @property
    def untestable(self) -> list[str]:
        """Ids of hypotheses carrying no test_design at all.

        These come back INCONCLUSIVE no matter how much evidence the next cycle
        gathers, because nothing was ever named that could decide them.
        """
        return [h.id for h in self.hypotheses if not h.is_proposed]

    @property
    def unroutable(self) -> list[str]:
        """Ids whose design a human can act on but M4 cannot resolve.

        Distinct from :attr:`untestable`: a test WAS proposed, it just names no
        measured signal, so it is work for the next M1 cycle rather than a claim
        with no falsifier. Collapsing the two would report a real experimental
        plan as an empty one.
        """
        return [h.id for h in self.hypotheses if h.is_proposed and not h.is_routable]

    @field_validator("hypotheses")
    @classmethod
    def _ids_unique(cls, v: list[HypothesisWire]) -> list[HypothesisWire]:
        ids = [h.id for h in v]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate hypothesis ids make M4 verdicts ambiguous")
        return v


__all__ = ["TEST_DESIGN_DIRECTIVES", "DiagnosisInput", "HypothesisWire", "DiagnosisOutput"]
