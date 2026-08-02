"""First-error attribution — LLM-as-judge (the Who&When 'all-at-once' protocol).

Given a trajectory, ask a JUDGE model which step first introduced the error.
The judge is a separate handle (judge != system-under-test, to avoid
self-preference bias).  Writes ``is_first_error`` onto the implicated step.

Reliability caveats (bake into any reporting): LLM judges show position/length
bias and self-preference; calibrate against a human-gold subset and report
step-accuracy/F1.  As reported by Zhang et al. (Who&When), even strong judges
reach only ~14% step-level accuracy — treat outputs as weak signal, not truth.

References:
- Which Agent Causes Task Failures and When? Automated Failure Attribution (Who&When)
  Zhang et al., ICML 2025 — arXiv:2505.00212
- Failure-mode vocabulary — MAST, Cemri et al., 2025 — arXiv:2503.13657
- LLM-as-judge biases — Judging LLM-as-a-Judge, Zheng et al., NeurIPS 2023 — arXiv:2306.05685
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, Trajectory
    from evalvitals.core.model import Model

_PROMPT = """You are debugging an AI agent trajectory. The agent failed to achieve the goal.
Goal: {goal}

Steps:
{steps}

Which step index FIRST introduced the error that led to failure?
Answer with exactly one line: "STEP: <index>"  (use -1 if no step is at fault)."""


def _render(traj: "Trajectory") -> str:
    lines = []
    for s in traj.steps:
        piece = f"[{s.idx}] {s.role.value}"
        if s.tool_call:
            piece += f" calls {s.tool_call.get('name')}({s.tool_call.get('args')})"
        if s.observation is not None:
            piece += f" -> obs: {str(s.observation)[:200]}"
        if s.content:
            piece += f" : {str(s.content)[:200]}"
        lines.append(piece)
    return "\n".join(lines)


@register_analyzer("first_error_judge")
class FirstErrorJudge(Analyzer):
    """Attribute the first-erroneous step via an LLM judge (all-at-once protocol)."""

    name = "first_error_judge"
    requires = frozenset()  # the analysed model is irrelevant; the judge does the work
    applies_to_modalities = frozenset({"text", "image"})  # trajectory analysis is modality-agnostic

    def __init__(self, judge: Optional["Model"] = None) -> None:
        super().__init__(judge=judge)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        judge = self.judge if self.judge is not None else model
        if judge is None or not hasattr(judge, "generate"):
            raise ValueError("FirstErrorJudge needs a judge model with generate(); pass judge=...")
        per_case = []
        for case in cases:
            traj = case.trajectory
            if traj is None:
                continue
            raw = judge.generate(_PROMPT.format(goal=traj.goal, steps=_render(traj)))
            m = re.search(r"STEP:\s*(-?\d+)", str(raw))
            idx = int(m.group(1)) if m else -1
            if 0 <= idx < len(traj.steps):
                traj.steps[idx].is_first_error = True
            per_case.append({"sample_id": traj.sample_id, "first_error_step": idx, "judge_raw": str(raw)[:200]})
        return Result(
            analyzer=self.name,
            model=repr(model),
            cases=cases,
            findings={
                "n_trajectories": len(per_case),
                "judge": repr(judge),
                "per_case": per_case,
                "_caveat": "LLM-judge: position/length/self-preference bias; calibrate vs human gold (Who&When).",
            },
        )
