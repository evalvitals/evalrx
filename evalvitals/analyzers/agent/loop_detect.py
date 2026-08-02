"""Loop detection — a deterministic trajectory heuristic (no model, no LLM).

Flags repeated ``(tool, normalised args)`` actions in an agent trajectory — the
MAST "step repetition" failure mode.  Cheap, deterministic, and the highest-ROI
agent signal to compute first.

References:
- Failure-mode taxonomy — Why Do Multi-Agent LLM Systems Fail? (MAST)
  Cemri et al., 2025 — arXiv:2503.13657  (step-repetition mode)
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model


def _sig(tool_call: dict) -> tuple[str, str]:
    return (tool_call.get("name", ""), json.dumps(tool_call.get("args", {}), sort_keys=True, default=str))


@register_analyzer("loop_detect")
class LoopDetector(Analyzer):
    """Detect repeated tool actions in a trajectory.

    Hyper-parameters:
        min_repeats: how many times an identical action must occur to count as a loop.
    """

    name = "loop_detect"
    requires = frozenset()  # pure heuristic over Trajectory — no model needed
    applies_to_modalities = frozenset({"text", "image"})  # trajectory analysis is modality-agnostic

    def __init__(self, min_repeats: int = 2) -> None:
        super().__init__(min_repeats=min_repeats)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case = []
        for case in cases:
            traj = case.trajectory
            if traj is None:
                continue
            sigs = [_sig(s.tool_call) for s in traj.steps if s.tool_call]
            counts = Counter(sigs)
            loops = [
                {"action": name, "args": args, "count": n}
                for (name, args), n in counts.items()
                if n >= self.min_repeats
            ]
            consecutive = any(sigs[i] == sigs[i + 1] for i in range(len(sigs) - 1))
            per_case.append({
                "sample_id": traj.sample_id,
                "has_loop": bool(loops),
                "consecutive_repeat": consecutive,
                "loops": loops,
            })
        return Result(
            analyzer=self.name,
            model=repr(model),
            cases=cases,
            findings={
                "n_trajectories": len(per_case),
                "n_with_loops": sum(c["has_loop"] for c in per_case),
                "per_case": per_case,
            },
        )
