"""Hypotheses — the unit the self-evolving agent proposes, tests, and mutates.

A hypothesis is a falsifiable claim about *when/why* a model fails (e.g. "Qwen
mis-binds entities when two names share a surname"). The loop turns it into
cases + an experiment, runs it, and updates its status from the findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"        # generated, not yet tested
    TESTING = "testing"          # experiment in flight
    SUPPORTED = "supported"      # evidence backs it
    REFUTED = "refuted"          # evidence contradicts it
    INCONCLUSIVE = "inconclusive"


@dataclass
class Hypothesis:
    """A falsifiable claim about a model's failure behaviour.

    Attributes:
        statement:              Natural-language claim (LLM-generated/readable).
        target_model:           Registered model name the claim is about.
        predicted_failure_mode: Tag/description of the expected failure.
        test_design:            How to verify this claim — analyzer / per-case
                                signal / strategy-contrast keywords proposed by
                                M3 (e.g. ``"relative_attention.max_relative_weight"``,
                                ``"prompt_contrast describe_first"``).  M5 uses
                                it to route evidence deterministically and M1
                                uses it for cycle-2 targeted probing.
        expected_association:   Pre-registered direction for the named test:
                                ``higher_on_failures`` or ``lower_on_failures``.
                                This prevents a protective-valued signal (for
                                example ``n_correct``) from being interpreted
                                backwards after its effect is observed.
        status:                 Lifecycle state (:class:`HypothesisStatus`).
        parent_id:              Hypothesis this one was mutated from, if any.
        id:                     Stable identifier.
        evidence:               Result/case ids accumulated while testing.
        metadata:               Free-form extras.
    """

    statement: str
    target_model: str
    predicted_failure_mode: str
    test_design: str = ""
    expected_association: str = ""
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    parent_id: str | None = None
    id: str = ""
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class HypothesisGenerator:
    """Proposes new hypotheses — from scratch or by mutating prior ones.

    Stage 2: an LLM-backed generator that reads the store (past supported/refuted
    hypotheses and their evidence) and proposes the next ones to test.
    """

    def propose(self, context: Any = None) -> list[Hypothesis]:
        raise NotImplementedError("HypothesisGenerator is planned for Stage 2.")

    def mutate(self, hypothesis: Hypothesis, feedback: Any = None) -> list[Hypothesis]:
        """Derive refined/adjacent hypotheses from one that was tested."""
        raise NotImplementedError("HypothesisGenerator.mutate is planned for Stage 2.")


# ---------------------------------------------------------------------------
# Serialization helpers (needed by JsonlStore and loop checkpointing)
# ---------------------------------------------------------------------------


def hypothesis_to_dict(h: Hypothesis) -> dict[str, Any]:
    """Serialize a Hypothesis to a JSON-compatible dict."""
    return {
        "statement": h.statement,
        "target_model": h.target_model,
        "predicted_failure_mode": h.predicted_failure_mode,
        "test_design": h.test_design,
        "expected_association": h.expected_association,
        "status": h.status.value if h.status else HypothesisStatus.PROPOSED.value,
        "parent_id": h.parent_id,
        "id": h.id,
        "evidence": list(h.evidence),
        "metadata": dict(h.metadata),
    }


def hypothesis_from_dict(data: dict[str, Any]) -> Hypothesis:
    """Deserialize a Hypothesis from a dict (e.g., loaded from JSONL)."""
    raw_status = data.get("status", HypothesisStatus.PROPOSED.value)
    try:
        status = HypothesisStatus(raw_status)
    except ValueError:
        status = HypothesisStatus.PROPOSED
    return Hypothesis(
        statement=str(data.get("statement", "")),
        target_model=str(data.get("target_model", "")),
        predicted_failure_mode=str(data.get("predicted_failure_mode", "")),
        test_design=str(data.get("test_design", "")),
        expected_association=str(data.get("expected_association", "")),
        status=status,
        parent_id=data.get("parent_id"),
        id=str(data.get("id", "")),
        evidence=list(data.get("evidence", [])),
        metadata=dict(data.get("metadata", {})),
    )


class ManualHypothesisGenerator(HypothesisGenerator):
    """A non-LLM generator: drains a fixed queue, or calls an injected ``proposer``.

    Lets the loop run + be unit-tested without an LLM; swap in an LLM-backed
    generator later without touching the loop.
    """

    def __init__(self, hypotheses: list[Hypothesis] | None = None, proposer: Any = None) -> None:
        self._queue: list[Hypothesis] = list(hypotheses or [])
        self._proposer = proposer

    def propose(self, context: Any = None) -> list[Hypothesis]:
        if self._proposer is not None:
            return list(self._proposer(context))
        out, self._queue = self._queue, []
        return out

    def mutate(self, hypothesis: Hypothesis, feedback: Any = None) -> list[Hypothesis]:
        return []
