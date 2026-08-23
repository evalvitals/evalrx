"""M4 — intervention and repair. Two different jobs under one number.

``SurgeryAgent`` INTERVENES: it changes one variable, re-runs, and reads the
result as causal evidence about *why*. Its output is a verdict, not a deliverable
-- the modified inputs are discarded.

``FixAgent`` REPAIRS: it proposes changes intended to survive, and validates them
paired against the unmodified baseline on the same cases.

Conflating them produces the pipeline's worst failure mode: presenting a
diagnostic manipulation as a shippable fix.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from evalvitals.contract.common import (
    ArtifactRef, CaseBatchRef, HypothesisStatus, StageEnvelope, WireModel,
)
from evalvitals.contract.methodology import MethodologyWire


# ===========================================================================
# M4a — SurgeryAgent: verify the mechanism
# ===========================================================================

class SurgeryInput(WireModel):
    """One hypothesis at a time. Surgery is per-claim by construction."""

    hypothesis_id: str
    results_ref: str = Field(description="Path to the M1 ProbeOutput, for per-case signal extraction.")
    data: CaseBatchRef
    model_name: str


class InterventionOutput(StageEnvelope):
    """``SurgeryAgent.operate`` result — causal evidence, not a repair.

    ``fixed`` here means "the intervention completely separated failing from
    passing cases", i.e. the mechanism is confirmed. It does NOT mean a usable
    repair exists; that claim belongs to :class:`FixOutput` alone.
    """

    hypothesis_id: str = Field(min_length=1, description="Points into the M3 DiagnosisOutput.")
    hypothesis_status: HypothesisStatus = Field(
        description="Verdict on the hypothesis. Named apart from the envelope's `status`, "
                    "which is this STAGE's execution state — they answer different questions."
    )
    strategy: Literal["verify_fn", "analyzer_params", "experiment_writer", "passive_correlation"] = (
        "passive_correlation"
    )
    fixed: bool = Field(
        default=False, description="Mechanism fully separates the groups. NOT a shippable repair."
    )
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_dimensions: dict[str, float] = Field(
        default_factory=dict, description="Breakdown: evidence gap / sample adequacy / control cleanliness."
    )
    evidence: dict[str, Any] = Field(default_factory=dict)
    methodology: MethodologyWire | None = Field(
        default=None,
        description="The manipulation performed: which variable was changed, what was held fixed, "
                    "what was compared. This is what makes the causal claim auditable.",
    )
    trial_root: str | None = Field(
        default=None,
        description="Attempt folder when the experiment-writer strategy ran: generated code, "
                    "stdout/stderr, workspace and record all sit there.",
    )
    new_data: CaseBatchRef | None = Field(
        default=None,
        description="When SUPPORTED: the cases NOT in the signal group — the refined subset for the "
                    "next M1 cycle. null means no refinement, which differs from an empty batch.",
    )

    @property
    def is_intervention(self) -> bool:
        """Whether this run can ground a causal claim.

        ``passive_correlation`` only observes, so it cannot — which makes this a
        function of ``strategy``, not an independent fact worth storing.
        """
        return self.strategy != "passive_correlation"


# ===========================================================================
# M4b — FixAgent: propose and validate repairs
# ===========================================================================

class FixInput(WireModel):
    """Repairs are attempted only on hypotheses M5 verified. ``max_tier`` bounds
    how invasive the search may get and is the caller's decision, never the
    agent's."""

    hypothesis_ids: list[str]
    data: CaseBatchRef
    model_name: str
    max_tier: Literal["L0", "L1", "L2", "L3a", "L3b", "L4"] = Field(
        description="L0 runtime config / L1 prompt / L2 scaffold / L3a internals-read / "
                    "L3b internals-write / L4 params.\n\n"
                    "L0 is in this list because `FixTier` has it — decoding settings are a "
                    "real, and the least invasive, place to intervene. Omitting it made a "
                    "tier the pipeline can produce unrepresentable."
    )
    max_repair_rounds: int = Field(
        default=1, ge=1, description="Feedback-driven propose->validate rounds. 1 = no loop."
    )
    prior_attempts_ref: str | None = None


class FixAttemptWire(WireModel):
    """One candidate validated against the unmodified baseline on the SAME cases.

    Every count here is a PAIRED flip, never a rate difference. "repaired 18 /
    broke 1" is a statement about individual cases changing sign; a pass-rate
    delta is not the same claim and must never be displayed as if it were.
    """

    tier: Literal["L0", "L1", "L2", "L3a", "L3b", "L4"]
    name: str
    kind: str | None = None
    source: str | None = None
    methodology: MethodologyWire | None = Field(
        default=None,
        description="What this candidate DOES, as a graph plus a rendered draw.io diagram. The "
                    "payload is the implementation; this is the explanation. Required for any "
                    "candidate a reader is asked to accept — a repair nobody can follow cannot "
                    "be reviewed, only trusted.",
    )
    trial_root: str | None = Field(
        default=None, description="Self-contained attempt folder. Preferred join key across rounds — "
                                  "name alone is not unique."
    )

    n_pairs: int = 0
    n_baseline_correct: int = 0
    n_fixed: int = 0
    n_broken: int = 0
    fixed_cases: list[str] = Field(default_factory=list)
    broken_cases: list[str] = Field(default_factory=list)

    n_applicable: int = Field(default=0, description="Cases the candidate actually touched.")
    coverage: float | None = Field(default=None, description="applicable FAILs / total FAILs.")
    n_unstable: int = Field(
        default=0, description="Dropped as baseline-unstable, so sampling noise is not read as regression."
    )
    n_model_independent: int = Field(
        default=0,
        description="Cases the scaffold solved by its own computation under frozen_model_control — "
                    "excluded from the paired test rather than counted as fixed.",
    )

    effect: float | None = None
    e_value: float | None = None
    reject: bool = Field(default=False, description="This candidate's own gate. Not sufficient to win.")
    verdict: Literal[
        "fixed", "partial", "unsafe", "regressed", "no_effect", "not_executed", "model_independent"
    ] = Field(
        description="The candidate's outcome. Subsumes a separate `fixed` boolean "
                    "(verdict == 'fixed') and a separate exec_error ('not_executed')."
    )
    summary: str = ""


class FixOutput(StageEnvelope):
    """``FixAgent.propose_and_validate`` result.

    What each candidate literally applied — the prompt text, the generated code —
    is not here. It is the ``result.json`` under ``trial_root``, which is also
    where the per-case outputs and the execution record sit; inlining a code blob
    would put a second, drifting copy in every report a reader loads.
    """

    max_tier: Literal["L0", "L1", "L2", "L3a", "L3b", "L4"]
    routed: list[dict[str, str]] = Field(
        default_factory=list, description="Which hypothesis went to which tier, and why."
    )

    attempted: list[FixAttemptWire] = Field(
        default_factory=list, description="Every candidate validated against the unmodified baseline."
    )

    best: str | None = Field(
        default=None, description="Winning candidate NAME. The row itself is in `attempted`."
    )
    fixed: bool = False
    ebh_survivors: list[str] = Field(
        default_factory=list,
        description="Candidate names surviving e-BH across the whole tested family. A winner must be "
                    "BOTH individually `fixed` AND in this set — best-of-N needs its own correction.",
    )
    repair_rounds: int = Field(default=0, ge=0)
    recommendation: dict[str, Any] | None = Field(
        default=None,
        description="{recommend_tier, reason} when nothing validated. Escalation is NEVER automatic.",
    )
    refine_signal: dict[str, Any] | None = Field(
        default=None,
        description="Set when a candidate helped one subset and hurt another — evidence the hypothesis "
                    "should be re-scoped, not that no fix exists. Feedback edge back into M3.",
    )

    @model_validator(mode="after")
    def _fixed_requires_confirmed_survivor(self) -> "FixOutput":
        if not self.fixed:
            return self
        if self.best is None:
            raise ValueError("fixed=True requires a `best` candidate")
        row = next((a for a in self.attempted if a.name == self.best), None)
        if row is None:
            raise ValueError(f"best={self.best!r} is not among the validated attempts")
        if row.methodology is None:
            raise ValueError(
                "fixed=True without a methodology on the winning candidate: a shipped repair "
                "must carry an explanation a reviewer can read"
            )
        if self.ebh_survivors and self.best not in self.ebh_survivors:
            raise ValueError(
                f"best={self.best!r} is not an e-BH survivor; best-of-N selection without "
                "family correction is not a validated fix"
            )
        return self


__all__ = [
    "SurgeryInput", "InterventionOutput",
    "FixInput", "FixAttemptWire", "FixOutput",
]
