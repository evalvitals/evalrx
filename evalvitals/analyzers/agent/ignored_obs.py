"""Ignored-observation detection — deterministic trajectory heuristic.

Flags the pattern: a tool returns an error/empty signal, yet the next action
repeats the *same* tool call unchanged (the agent didn't adapt to the
observation).  MAST "ignored tool output" / failure-to-recover mode.

References:
- Failure-mode taxonomy — Why Do Multi-Agent LLM Systems Fail? (MAST)
  Cemri et al., 2025 — arXiv:2503.13657  (ignored-tool-output / fail-to-recover modes)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.case import StepRole
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model


@register_analyzer("ignored_obs")
class IgnoredObservationDetector(Analyzer):
    """Detect error observations that the agent then ignored (repeated the same call)."""

    name = "ignored_obs"
    requires = frozenset()
    applies_to_modalities = frozenset({"text", "image"})  # trajectory analysis is modality-agnostic

    DEFAULT_MARKERS = ("error", "not found", "failed", "no result", "none", "invalid", "empty", "exception")

    def __init__(self, markers: tuple[str, ...] | None = None) -> None:
        super().__init__(markers=markers or self.DEFAULT_MARKERS)

    def _is_error(self, observation) -> bool:
        text = str(observation).lower()
        return any(m in text for m in self.markers)

    @staticmethod
    def _sig(tool_call: dict | None) -> str | None:
        if not tool_call:
            return None
        return json.dumps([tool_call.get("name"), tool_call.get("args", {})], sort_keys=True, default=str)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case = []
        for case in cases:
            traj = case.trajectory
            if traj is None:
                continue
            steps = traj.steps
            flags = []
            for i, s in enumerate(steps):
                if s.role is not StepRole.TOOL or s.observation is None or not self._is_error(s.observation):
                    continue
                # the actor that produced this observation is the previous step;
                # the next actor that repeats the same call ignored the error.
                prev_sig = self._sig(steps[i - 1].tool_call) if i > 0 else None
                nxt = next((x for x in steps[i + 1:] if x.role is StepRole.ACTOR), None)
                if nxt is not None and prev_sig is not None and self._sig(nxt.tool_call) == prev_sig:
                    flags.append({"error_obs_step": s.idx, "repeated_at_step": nxt.idx})
            per_case.append({"sample_id": traj.sample_id, "n_ignored": len(flags), "ignored": flags})
        return Result(
            analyzer=self.name,
            model=repr(model),
            cases=cases,
            findings={
                "n_trajectories": len(per_case),
                "n_with_ignored_obs": sum(1 for c in per_case if c["n_ignored"]),
                "per_case": per_case,
            },
        )
