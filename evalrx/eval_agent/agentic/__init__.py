"""Judge-decided next steps for VLDiagnoseLoop, in place of its fixed
M1->M2->M3->M4 cycle: ``VLDiagnoseLoop(..., policy=JudgePolicy(judge))``.

  actions.py  Action, decide — the judge's typed choice of the next tool
  board.py    EvidenceBoard, BudgetState — what the judge sees each turn
  tools.py    ToolSpec, ToolRegistry, build_default_registry — the M1-M4 tools
  policy.py   JudgePolicy
"""

from __future__ import annotations

from evalrx.eval_agent.agentic.actions import Action, decide
from evalrx.eval_agent.agentic.board import BudgetState, EvidenceBoard
from evalrx.eval_agent.agentic.policy import JudgePolicy
from evalrx.eval_agent.agentic.tools import (
    ToolOutcome,
    ToolRegistry,
    ToolSpec,
    build_default_registry,
)

__all__ = [
    "JudgePolicy",
    "Action",
    "decide",
    "EvidenceBoard",
    "BudgetState",
    "ToolSpec",
    "ToolOutcome",
    "ToolRegistry",
    "build_default_registry",
]
