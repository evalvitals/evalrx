"""Reasoning analyzers — text-chain diagnostics (black-box, ``GENERATE`` only).

Two of these are HYGIENE probes that belong before the rest, not beside them:
``answer_extraction_audit`` (is the FAIL label real, or did the grader miss the
answer?) and ``termination_audit`` (did the generation stop cleanly, or hit the
token budget / a repetition loop?).  Both produce confounds that mimic every
other probe's "bad" column, so a mechanism finding read before them is not
interpretable.

The rest decompose *how* a chain failed: ``arith_audit`` (computation slip vs
chain break, at zero extra cost), ``self_repair`` (detect / correct / damage),
``step_rollout_value`` (where the chain broke, by Math-Shepherd rollouts),
``knowledge_reasoning_split`` (missing fact vs broken composition), and
``contamination_score`` (is the benchmark measuring recall?).
"""

from evalrx.analyzers.reasoning.answer_extraction_audit import AnswerExtractionAudit
from evalrx.analyzers.reasoning.arith_audit import ArithmeticAudit
from evalrx.analyzers.reasoning.contamination import ContaminationProbe
from evalrx.analyzers.reasoning.knowledge_split import KnowledgeReasoningSplit
from evalrx.analyzers.reasoning.self_repair import SelfRepairAnalyzer
from evalrx.analyzers.reasoning.step_rollout_value import StepRolloutValueAnalyzer
from evalrx.analyzers.reasoning.termination_audit import TerminationAudit

__all__ = [
    "AnswerExtractionAudit",
    "TerminationAudit",
    "ArithmeticAudit",
    "SelfRepairAnalyzer",
    "StepRolloutValueAnalyzer",
    "KnowledgeReasoningSplit",
    "ContaminationProbe",
]
