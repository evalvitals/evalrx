"""Self-repair — can the model detect its own error, and can it fix it without breaking what worked?

Three capabilities get conflated as "self-correction", and they dissociate:

  * **detection** — does the model know the answer is wrong when asked?
  * **correction** — given that it says so, does the revision land on the gold?
  * **damage** — does the same revision pass BREAK answers that were already right?

The third is the one that decides deployment.  A self-critique loop with a 40%
repair rate and a 25% damage rate is a net loss on any batch that is mostly
passing, and reporting repair alone hides that completely — which is why this
probe deliberately spends its generations on PASS cases too.

Cost: 2 generations per case (critique + blind revision), 3 with
``revise_with_critique=True``.

References:
- Large Language Models Cannot Self-Correct Reasoning Yet — Huang et al.,
  ICLR 2024 — arXiv:2310.01798 (intrinsic self-correction degrades accuracy)
- Self-Refine: Iterative Refinement with Self-Feedback — Madaan et al.,
  NeurIPS 2023 — arXiv:2303.17651
- Teaching Large Language Models to Self-Debug — Chen et al., ICLR 2024 —
  arXiv:2304.05128
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.analyzers.reasoning._text import (
    answer_equal,
    extract_answer,
    normalize_answer,
)
from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model

_CRITIQUE = (
    "Here is a candidate answer to the question above:\n\n{answer}\n\n"
    "Is this answer correct? Reply with exactly one word: CORRECT or INCORRECT."
)
_REVISE_BLIND = (
    "Here is a candidate answer to the question above:\n\n{answer}\n\n"
    "Review it and produce your best final answer. Give the answer on its own "
    "last line as 'Answer: <answer>'."
)
_REVISE_TOLD = (
    "Here is a candidate answer to the question above:\n\n{answer}\n\n"
    "This answer contains an error. Find it and produce a corrected final "
    "answer on its own last line as 'Answer: <answer>'."
)


@register_analyzer("self_repair")
class SelfRepairAnalyzer(Analyzer):
    """Measure self-detection, repair rate, and — the part that decides deployment — damage rate.

    Hyper-parameters:
        revise_with_critique: also run the "you made an error" revision
                              (+1 generation; measures repair under an oracle
                              error signal the deployed loop will not have).
        max_cases:            label-stratified cap — PASS cases are REQUIRED
                              here, they are what makes damage measurable.
        generate_missing:     produce a baseline answer when ``observed`` is empty.
        grader:               ``callable(prediction, case) -> bool | None``.
        answer_fn:            ``callable(text) -> str`` answer extractor.
    """

    name = "self_repair"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        revise_with_critique: bool = False,
        max_cases: int = 32,
        generate_missing: bool = True,
        grader: Optional[Callable[[Any, "FailureCase"], Optional[bool]]] = None,
        answer_fn: Optional[Callable[[Any], str]] = None,
    ) -> None:
        super().__init__(
            revise_with_critique=revise_with_critique,
            max_cases=max_cases,
            generate_missing=generate_missing,
        )
        # ctor names, so sklearn-style get_params() reflection works
        self.grader = grader or _default_grader
        self.answer_fn = answer_fn or extract_answer

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case: list[dict[str, Any]] = []
        for case in cases.stratified_head(self.max_cases):
            baseline = case.observed
            if not str(baseline or "").strip():
                if not self.generate_missing:
                    per_case.append(
                        {"sample_id": case.id, "skipped": "no stored answer to revise"}
                    )
                    continue
                baseline = str(model.generate(case.inputs))
            entry = self._probe_case(model, case, str(baseline))
            per_case.append(entry)

        graded = [c for c in per_case if "baseline_correct" in c]
        failed = [c for c in graded if c["baseline_correct"] == 0]
        passed = [c for c in graded if c["baseline_correct"] == 1]
        # A real model sometimes answers the critique in prose that parses to
        # nothing; those cases carry self_says_incorrect=None and must be
        # dropped from the detection rates rather than counted as "said correct".
        detected = [c for c in graded if c.get("detection_correct") is not None]
        verdicts = [c for c in passed if c.get("self_says_incorrect") is not None]

        repair_rate = (
            round(sum(c["repaired"] for c in failed) / len(failed), 4) if failed else None
        )
        damage_rate = (
            round(sum(c["damaged"] for c in passed) / len(passed), 4) if passed else None
        )
        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "n_graded": len(graded),
            "n_baseline_fail": len(failed),
            "n_baseline_pass": len(passed),
            "detection_accuracy": (
                round(sum(c["detection_correct"] for c in detected) / len(detected), 4)
                if detected
                else None
            ),
            "n_critique_unparsed": sum(
                1 for c in graded if c.get("self_says_incorrect") is None
            ),
            "false_alarm_rate": (
                round(sum(c["self_says_incorrect"] for c in verdicts) / len(verdicts), 4)
                if verdicts
                else None
            ),
            "repair_rate": repair_rate,
            "damage_rate": damage_rate,
            "net_revision_gain": (
                round(
                    (sum(c["repaired"] for c in failed) - sum(c["damaged"] for c in passed))
                    / len(graded),
                    4,
                )
                if graded
                else None
            ),
            "per_case": per_case,
            "_caveat": (
                "repair_rate WITHOUT damage_rate is not a result: a revision "
                "pass is only a fix if net_revision_gain > 0 on the deployed "
                "PASS/FAIL mix, and this batch is label-stratified, so its mix "
                "is NOT the deployment mix — re-weight before promoting an L1 "
                "self-critique fix. Cases needing PASS mass: damage_rate is "
                "None when the batch has no passing cases, and a repair-only "
                "reading is then unsafe. With revise_with_critique=True the "
                "told_* columns use an ORACLE error signal (we told it there is "
                "an error) and overstate what a deployed loop achieves. "
                "INTERVENTIONAL — held-out confirmation must re-run every call."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)

    # ------------------------------------------------------------------
    def _probe_case(self, model: "Model", case: "FailureCase", baseline: str) -> dict[str, Any]:
        entry: dict[str, Any] = {"sample_id": case.id}
        base_grade = self.grader(baseline, case)

        critique = str(model.generate(self._ask(case, _CRITIQUE, baseline)))
        says_incorrect = _parse_verdict(critique)
        revised = str(model.generate(self._ask(case, _REVISE_BLIND, baseline)))
        revised_grade = self.grader(revised, case)

        entry["self_says_incorrect"] = (
            int(says_incorrect) if says_incorrect is not None else None
        )
        entry["critique_parsed"] = int(says_incorrect is not None)
        entry["changed_answer"] = int(
            normalize_answer(self.answer_fn(revised))
            != normalize_answer(self.answer_fn(baseline))
        )
        if base_grade is None or revised_grade is None:
            entry["skipped"] = "ungradable (no gold answer)"
            return entry

        entry.update(
            {
                "baseline_correct": int(base_grade),
                "revised_correct": int(revised_grade),
                "repaired": int(not base_grade and revised_grade),
                "damaged": int(base_grade and not revised_grade),
            }
        )
        if says_incorrect is not None:
            entry["detection_correct"] = int(says_incorrect == (not base_grade))

        if self.revise_with_critique:
            told = str(model.generate(self._ask(case, _REVISE_TOLD, baseline)))
            told_grade = self.grader(told, case)
            if told_grade is not None:
                entry["told_correct"] = int(told_grade)
                entry["told_repaired"] = int(not base_grade and told_grade)
                entry["told_damaged"] = int(base_grade and not told_grade)
        return entry

    @staticmethod
    def _ask(case: "FailureCase", template: str, answer: str):
        prompt = f"{case.inputs.prompt or ''}\n\n{template.format(answer=answer)}"
        return dataclasses.replace(case.inputs, prompt=prompt)


def _default_grader(prediction: Any, case: "FailureCase") -> Optional[bool]:
    if case.expected is None:
        return None
    return answer_equal(extract_answer(prediction), case.expected)


#: Negated forms matter: "not correct" is a verdict of INCORRECT, and a plain
#: search for "correct" reads it backwards.  ``\bcorrect\b`` cannot match inside
#: "incorrect" (no word boundary there), so the two patterns stay disjoint.
_SAYS_INCORRECT = re.compile(
    r"\b(?:incorrect|wrong|false|not\s+(?:correct|right)|isn'?t\s+(?:correct|right))\b"
)
_SAYS_CORRECT = re.compile(r"\b(?:correct|right|yes)\b")


def _parse_verdict(text: Any) -> Optional[bool]:
    """``True`` = the model called the answer incorrect; ``None`` = unparseable."""
    lowered = str(text or "").lower()
    # Take whichever verdict appears FIRST — models restate the question
    # ("is this correct? ...") before delivering the verdict.
    hits: list[tuple[int, bool]] = [
        (m.start(), True) for m in _SAYS_INCORRECT.finditer(lowered)
    ] + [(m.start(), False) for m in _SAYS_CORRECT.finditer(lowered)]
    if not hits:
        return None
    return min(hits)[1]
