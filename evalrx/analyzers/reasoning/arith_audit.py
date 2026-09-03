"""Arithmetic audit — computation slip or broken reasoning chain?

"The model is bad at math" is two failures wearing one label.  Either the plan
was right and one multiplication came out wrong (a *computation slip*, fixed by
a calculator tool or a verification pass), or every stated computation checks
out and the answer is still wrong (a *chain break*, fixed by better
decomposition — a calculator changes nothing).  M4 picks opposite tiers for the
two, so collapsing them wastes the whole arc.

The split needs **no extra generations**: every ``a op b = c`` statement in the
stored reasoning is re-evaluated, and the first wrong one is forward-substituted
— if repairing that single value reproduces the gold answer (directly, or by
propagating the difference/ratio to the model's own final number), the error is
a slip; if the arithmetic is clean and the answer is wrong, the chain broke.

References:
- Training Verifiers to Solve Math Word Problems — Cobbe et al., 2021 —
  arXiv:2110.14168 (calculator annotations: arithmetic vs planning errors)
- Let's Verify Step by Step — Lightman et al., ICLR 2024 — arXiv:2305.20050
- GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs
  — Mirzadeh et al., ICLR 2025 — arXiv:2410.05229
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalrx.analyzers.reasoning._text import (
    answer_equal,
    as_number,
    extract_answer,
    find_equations,
    numbers_in,
)
from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability, CapabilityError
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model

_COT_SUFFIX = (
    "Show every calculation as an explicit 'a op b = c' line. After the "
    "reasoning, give the final answer on its own last line as 'Answer: <answer>'."
)

SLIP = "computation_slip"
CHAIN_BREAK = "chain_break"
COMPOUND = "compound"
CLEAN = "clean"
LUCKY = "errors_but_correct"
UNKNOWN = "unknown"


def _close(a: float, b: float, rel_tol: float) -> bool:
    return abs(a - b) <= rel_tol * max(1.0, abs(b))


@register_analyzer("arith_audit")
class ArithmeticAudit(Analyzer):
    """Re-check every arithmetic step in the reasoning and classify the failure.

    Hyper-parameters:
        rel_tol:          relative tolerance for "the stated value is right".
        generate_missing: generate a chain when ``case.observed`` is empty
                          (1 generation; needs ``GENERATE``).  Off by default —
                          the probe is meant to be free.
        max_cases:        label-stratified cap; 0 (the default) = every case.
        answer_fn/match_fn: answer extraction / equality (see ``_text``).
    """

    name = "arith_audit"
    # Free and capability-free on the default path (it re-reads stored chains);
    # the generate_missing path checks GENERATE at run time.
    requires = frozenset()
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        rel_tol: float = 1e-6,
        generate_missing: bool = False,
        max_cases: int = 0,
        answer_fn: Optional[Callable[[Any], str]] = None,
        match_fn: Optional[Callable[[Any, Any], bool]] = None,
    ) -> None:
        super().__init__(
            rel_tol=rel_tol, generate_missing=generate_missing, max_cases=max_cases
        )
        self.answer_fn = answer_fn or extract_answer
        self.match_fn = match_fn or answer_equal

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        if self.generate_missing and Capability.GENERATE not in getattr(
            model, "capabilities", frozenset()
        ):
            raise CapabilityError(
                analyzer=self.name, model=repr(model), missing={Capability.GENERATE}
            )
        per_case: list[dict[str, Any]] = []
        for case in cases.stratified_head(self.max_cases):
            text = case.observed
            if not str(text or "").strip() and self.generate_missing:
                prompt = f"{case.inputs.prompt or ''}\n\n{_COT_SUFFIX}"
                text = str(model.generate(dataclasses.replace(case.inputs, prompt=prompt)))
            per_case.append(self._audit(case, text))

        classified = [c for c in per_case if c.get("error_class") not in (None, UNKNOWN)]
        wrong = [c for c in classified if c["answer_correct"] == 0]
        checked = [c for c in per_case if c.get("n_equations", 0) > 0]

        def _rate(cls: str) -> Optional[float]:
            return round(sum(1 for c in wrong if c["error_class"] == cls) / len(wrong), 4) \
                if wrong else None

        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "n_with_equations": len(checked),
            "n_wrong_answers": len(wrong),
            "mean_arith_error_rate": (
                round(sum(c["arith_error_rate"] for c in checked) / len(checked), 4)
                if checked
                else None
            ),
            "computation_slip_rate": _rate(SLIP),
            "chain_break_rate": _rate(CHAIN_BREAK),
            "compound_rate": _rate(COMPOUND),
            "n_errors_but_correct": sum(1 for c in classified if c["error_class"] == LUCKY),
            "per_case": per_case,
            "_caveat": (
                "The slip/chain-break split is only defined over cases whose "
                "reasoning actually WRITES its arithmetic: n_equations==0 means "
                "'not measured', not 'no arithmetic errors' — a model that "
                "computes silently is unmeasurable here, so read "
                "n_with_equations before the rates. computation_slip ⇒ an L1 "
                "tool/verification fix; chain_break ⇒ decomposition, where a "
                "calculator changes nothing. errors_but_correct is its own "
                "signal: arithmetic that is wrong yet lands on the gold answer "
                "means the written chain is decorative (cross-check with "
                "cot_faithfulness). OBSERVATIONAL — it re-reads stored outputs "
                "and adds no sampling, so it needs no re-run to confirm."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)

    # ------------------------------------------------------------------
    def _audit(self, case: "FailureCase", text: Any) -> dict[str, Any]:
        equations = find_equations(text)
        errors = [
            (i, expr, stated, computed)
            for i, (expr, stated, computed) in enumerate(equations)
            if not _close(stated, computed, self.rel_tol)
        ]
        entry: dict[str, Any] = {
            "sample_id": case.id,
            "n_equations": len(equations),
            "n_arith_errors": len(errors),
            "arith_error_rate": (
                round(len(errors) / len(equations), 4) if equations else 0.0
            ),
        }
        if errors:
            idx, expr, stated, computed = errors[0]
            entry.update(
                {
                    "first_error_idx": idx,
                    "first_error_expr": expr[:120],
                    "first_error_stated": stated,
                    "first_error_correct": computed,
                    # how deep into the written chain the first slip happened
                    "first_error_depth": round(idx / len(equations), 4),
                }
            )

        predicted = self.answer_fn(text)
        graded = (
            bool(self.match_fn(predicted, case.expected))
            if case.expected is not None
            else None
        )
        if graded is None:
            entry["error_class"] = UNKNOWN
            entry["_note"] = "no gold answer: arithmetic checked, failure not classified"
            return entry

        entry["answer_correct"] = int(graded)
        entry["predicted_answer"] = str(predicted)[:120]
        if graded:
            entry["error_class"] = LUCKY if errors else CLEAN
            return entry
        if not errors:
            entry["error_class"] = CHAIN_BREAK if equations else UNKNOWN
            return entry

        explains = self._slip_explains_final(errors[0], predicted, case.expected)
        entry["slip_explains_final"] = int(explains)
        entry["error_class"] = SLIP if explains else COMPOUND
        return entry

    def _slip_explains_final(
        self, error: tuple[int, str, float, float], predicted: Any, gold: Any
    ) -> bool:
        """Would repairing the FIRST wrong value have produced the gold answer?

        Three propagation paths cover the realistic cases without re-running the
        model: the repaired value IS the answer; the error carried through
        additively; or it carried through multiplicatively.
        """
        _, _, stated, computed = error
        gold_num = as_number(gold)
        if gold_num is None:
            return False
        tol = self.rel_tol
        if _close(computed, gold_num, tol):
            return True
        final_numbers = numbers_in(predicted)
        if not final_numbers:
            return False
        final = final_numbers[-1]
        if _close(final + (computed - stated), gold_num, tol):
            return True
        if stated != 0 and _close(final * (computed / stated), gold_num, tol):
            return True
        return False
