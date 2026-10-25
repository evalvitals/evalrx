"""One step's evidence on a growing sample, assembled from chunk observations.

A step that ran on the 8-FAIL sample and then on the cases the 16-FAIL sample
added holds two observations over disjoint case sets; which one measured a
case is a lookup, so the triage reads each case from the observation that
measured it and nothing is merged or recomputed by hand.

* ``reuse`` steps (a case's row does not depend on the rest of the batch)
  run only the cases not yet covered; the old rows stay current.
* other steps re-run on the whole current sample, and the latest run that
  covers a case is the one read: an earlier batch's rows were computed
  against a different batch and are kept on disk, not read.
* a step with an edit self-check that refused on any chunk is refused on
  every case: keeping the chunks that happened to pass would select on
  numerical noise, as the per-case gate inside one batch already rules out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from evalrx.eval_agent.probing.actions import OK, UNFAITHFUL, Observation, PlanStep


@dataclass
class StepEvidence:
    step: PlanStep
    chunks: list[Observation] = field(default_factory=list)

    def add(self, obs: Observation) -> None:
        self.chunks.append(obs)

    def refused(self) -> "Observation | None":
        if self.step.fidelity_check is None:
            return None
        return next((c for c in self.chunks if c.status == UNFAITHFUL), None)

    def for_case(self, cid: str) -> "Observation | None":
        """The observation that measured *cid* (``None`` = never planned on it)."""
        bad = self.refused()
        if bad is not None:
            return bad
        for obs in reversed(self.chunks):
            if cid in obs.planned_ids:
                return obs
        return None

    def covered(self) -> set[str]:
        return {cid for obs in self.chunks for cid in obs.planned_ids}

    def todo(self, target: Iterable[str]) -> list[str]:
        """The cases of *target* this step still has to run on (target order).

        ``reuse``: the ones no chunk planned. Otherwise: all of them, unless the
        latest chunk ran on exactly this case set.
        """
        target = list(target)
        if self.step.reuse:
            done = self.covered()
            return [cid for cid in target if cid not in done]
        if self.chunks and set(self.chunks[-1].planned_ids) == set(target):
            return []
        return target

    def valid_on(self, ids: Iterable[str]) -> bool:
        """Some case of *ids* has an ``ok`` measurement the step's self-check did not refuse."""
        if self.refused() is not None:
            return False
        return any((obs := self.for_case(cid)) is not None and obs.status == OK for cid in ids)

    def statuses_on(self, ids: Iterable[str]) -> list[str]:
        return sorted({obs.status for cid in ids if (obs := self.for_case(cid)) is not None})


def pick(evidence: "dict[str, Any]", key: str, cid: str) -> "Observation | None":
    """The observation for *cid* under *key*: a plain :class:`Observation` (one batch) or a
    :class:`StepEvidence` (chunks)."""
    value = evidence.get(key)
    if isinstance(value, StepEvidence):
        return value.for_case(cid)
    return value
