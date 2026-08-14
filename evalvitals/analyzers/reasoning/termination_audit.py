"""Termination audit — *why* did the generation stop?

A missing answer has at least four unrelated causes: the decode hit the token
budget (truncated), the model explicitly declined (gave up), it fell into a
repetition loop (degenerate), or it finished cleanly and was simply wrong.
These need different fixes — a bigger budget, a prompt change, a decoding
change, a capability change — and only the last one is a reasoning failure.

This is a **confound control**, not a headline metric.  Truncated and degenerate
cases carry short, tag-less outputs, which is exactly the signature that
correlates with *every* other probe's "bad" column, so leaving them in the pool
makes M2 attribute a decoding-budget problem to whatever mechanism is being
tested.  Stratify on ``termination_class`` before reading any other probe.

One optional generation per non-clean case continues the truncated text: if the
continuation lands on an answer, the case was budget-limited, not stuck.

References:
- The Curious Case of Neural Text Degeneration — Holtzman et al., ICLR 2020 —
  arXiv:1904.09751 (repetition as a decoding pathology, not a knowledge one)
- Measuring Faithfulness in Chain-of-Thought Reasoning — Lanham et al., 2023 —
  arXiv:2307.13702 (truncated-reasoning protocol)
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.analyzers.reasoning._text import (
    answer_equal,
    extract_answer,
    has_answer_tag,
    looks_like_give_up,
    looks_truncated,
    repetition_score,
)
from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model

_CONTINUE_SUFFIX = (
    "Continue the reasoning above from where it stops and finish it. "
    "End with the final answer on its own last line as 'Answer: <answer>'."
)

CLEAN = "clean"
TRUNCATED = "truncated"
DEGENERATE = "degenerate"
GAVE_UP = "gave_up"
NO_ANSWER = "no_answer"


@register_analyzer("termination_audit")
class TerminationAudit(Analyzer):
    """Classify how each generation ended and test whether continuing rescues it.

    Hyper-parameters:
        continue_non_clean: spend one generation continuing each non-clean case.
        repetition_n:       word n-gram width for the degeneration score.
        repetition_thresh:  repetition_score above this ⇒ ``degenerate``.
        max_cases:          label-stratified cap.
        answer_fn/match_fn: answer extraction / equality (see ``_text``).
    """

    name = "termination_audit"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        continue_non_clean: bool = True,
        repetition_n: int = 8,
        repetition_thresh: float = 0.5,
        max_cases: int = 64,
        answer_fn: Optional[Callable[[Any], str]] = None,
        match_fn: Optional[Callable[[Any, Any], bool]] = None,
    ) -> None:
        super().__init__(
            continue_non_clean=continue_non_clean,
            repetition_n=repetition_n,
            repetition_thresh=repetition_thresh,
            max_cases=max_cases,
        )
        self.answer_fn = answer_fn or extract_answer
        self.match_fn = match_fn or answer_equal

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case: list[dict[str, Any]] = []
        for case in cases.stratified_head(self.max_cases):
            text = case.observed
            if not str(text or "").strip():
                # no stored output — generate one so the class is about THIS model
                text = str(model.generate(case.inputs))
            entry = self._classify(case, text)
            if self.continue_non_clean and entry["termination_class"] != CLEAN:
                entry.update(self._continue(model, case, text))
            per_case.append(entry)

        n = len(per_case)
        counts = {
            cls: sum(1 for c in per_case if c["termination_class"] == cls)
            for cls in (CLEAN, TRUNCATED, DEGENERATE, GAVE_UP, NO_ANSWER)
        }
        rescued = [c for c in per_case if "recovered_by_continuation" in c]
        findings: dict[str, Any] = {
            "n_cases": n,
            "class_counts": counts,
            "clean_rate": round(counts[CLEAN] / n, 4) if n else None,
            "non_clean_rate": round(1 - counts[CLEAN] / n, 4) if n else None,
            "truncation_rate": round(counts[TRUNCATED] / n, 4) if n else None,
            "degenerate_rate": round(counts[DEGENERATE] / n, 4) if n else None,
            "giveup_rate": round(counts[GAVE_UP] / n, 4) if n else None,
            "recovered_rate": (
                round(sum(c["recovered_by_continuation"] for c in rescued) / len(rescued), 4)
                if rescued
                else None
            ),
            "per_case": per_case,
            "_caveat": (
                "CONFOUND CONTROL — read this before any other probe. Cases with "
                "termination_class != 'clean' have short, tag-less outputs, the "
                "same signature every other probe scores as 'bad'; leaving them "
                "pooled makes M2 attribute a decoding-budget or repetition "
                "problem to whichever mechanism is under test. Stratify on "
                "termination_class (or exclude non-clean cases) first. A high "
                "recovered_by_continuation rate means the budget is the defect: "
                "raise max_tokens and re-collect before diagnosing anything else. "
                "The continuation columns are INTERVENTIONAL — held-out "
                "verification must re-run them."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)

    # ------------------------------------------------------------------
    def _classify(self, case: "FailureCase", text: Any) -> dict[str, Any]:
        raw = str(text or "")
        repetition = repetition_score(raw, self.repetition_n)
        tagged = has_answer_tag(raw)
        truncated = looks_truncated(raw)
        gave_up = looks_like_give_up(raw)

        # Order matters: degeneration is checked first because a looping
        # generation also LOOKS truncated (it is cut off mid-loop), and the fix
        # for a loop is a decoding change, not a bigger budget.
        if repetition >= self.repetition_thresh:
            cls = DEGENERATE
        elif gave_up:
            cls = GAVE_UP
        elif truncated:
            cls = TRUNCATED
        elif tagged:
            cls = CLEAN
        else:
            cls = NO_ANSWER
        return {
            "sample_id": case.id,
            "termination_class": cls,
            "repetition_score": repetition,
            "has_answer_tag": int(tagged),
            "looks_truncated": int(truncated),
            "gave_up": int(gave_up),
            "output_chars": len(raw),
            "output_words": len(raw.split()),
        }

    def _continue(self, model: "Model", case: "FailureCase", text: Any) -> dict[str, Any]:
        prompt = (
            f"{case.inputs.prompt or ''}\n\nPartial response:\n{text}\n\n{_CONTINUE_SUFFIX}"
        )
        continuation = str(model.generate(dataclasses.replace(case.inputs, prompt=prompt)))
        out: dict[str, Any] = {
            "continuation_has_answer": int(has_answer_tag(continuation)),
            "continuation_chars": len(continuation),
        }
        graded = (
            self.match_fn(self.answer_fn(continuation), case.expected)
            if case.expected is not None
            else None
        )
        # "Recovered" means the model could finish given more room: it produced
        # an answer AND — when a gold exists — the right one.
        out["recovered_by_continuation"] = int(
            out["continuation_has_answer"] == 1 and graded is not False
        )
        if graded is not None:
            out["continuation_correct"] = int(graded)
        return out
