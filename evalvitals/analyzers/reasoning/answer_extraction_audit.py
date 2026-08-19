"""Answer-extraction audit — is this a wrong answer, or an unparsed one?

**Run this probe first.**  Every other per-case column in M1 is conditioned on
the PASS/FAIL label, and that label comes from a grader that has to find the
answer inside free-form text.  When the grader misses — the model wrote
``\\boxed{18}`` and the grader wanted ``Answer: 18``, or it answered in a
sentence, or the generation was cut off before the tag — the case is labelled
FAIL for a reason that has nothing to do with the model's reasoning.  Those
cases then enter every downstream correlation as noise pointing at whatever the
other probes happen to measure, and M2 reports a real-looking effect for a
harness bug.

Costs nothing by default (it reads ``observed``); with ``reask=True`` it spends
one strict-format generation per suspect case to settle whether the model
actually had the answer.

References:
- Answer-extraction sensitivity in LLM math evaluation — Language Model
  Evaluation Harness, answer-extraction regex family (EleutherAI, 2023-2025)
- Let's Verify Step by Step — Lightman et al., ICLR 2024 — arXiv:2305.20050
  (grading protocol for free-form numeric answers)
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.analyzers.reasoning._text import (
    _ANSWER_TAG,
    _BOXED,
    answer_equal,
    extract_answer,
    has_answer_tag,
    looks_like_give_up,
    looks_truncated,
    normalize_answer,
)
from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model

_REASK_SUFFIX = (
    "Reply with ONLY the final answer, no explanation, no units, no punctuation."
)


@register_analyzer("answer_extraction_audit")
class AnswerExtractionAudit(Analyzer):
    """Split FAIL cases into *really wrong* and *merely unparsed*.

    Hyper-parameters:
        reask:      re-ask each suspect case with a strict output format
                    (1 generation per suspect; needs ``GENERATE``).
        max_cases:  label-stratified cap.
        tail_chars: size of the trailing window treated as the answer region.
        answer_fn:  ``callable(text) -> str`` answer extractor.
        match_fn:   ``callable(prediction, gold) -> bool`` equality test.
    """

    name = "answer_extraction_audit"
    # Capability-free on the default (read-only) path so it can audit an
    # offline record dump; the re-ask path checks GENERATE at run time.
    requires = frozenset()
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        reask: bool = False,
        max_cases: int = 200,
        tail_chars: int = 200,
        answer_fn: Optional[Callable[[Any], str]] = None,
        match_fn: Optional[Callable[[Any, Any], bool]] = None,
    ) -> None:
        super().__init__(reask=reask, max_cases=max_cases, tail_chars=tail_chars)
        # ctor names, so sklearn-style get_params() reflection works
        self.answer_fn = answer_fn or extract_answer
        self.match_fn = match_fn or answer_equal

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        if self.reask and Capability.GENERATE not in getattr(
            model, "capabilities", frozenset()
        ):
            raise CapabilityError(
                analyzer=self.name, model=repr(model), missing={Capability.GENERATE}
            )
        per_case: list[dict[str, Any]] = []
        n_reasked = 0
        for case in cases.stratified_head(self.max_cases):
            entry = self._audit_case(case)
            if self.reask and entry.get("extraction_suspect") == 1:
                entry.update(self._reask(model, case))
                n_reasked += 1
            per_case.append(entry)

        gradable = [c for c in per_case if "extraction_suspect" in c]
        failing = [c for c in gradable if c["labelled_fail"] == 1]
        suspects = [c for c in failing if c["extraction_suspect"] == 1]
        confirmed = [c for c in suspects if c.get("reask_correct") == 1]
        suspect_rate = round(len(suspects) / len(failing), 4) if failing else None

        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "n_gradable": len(gradable),
            "n_labelled_fail": len(failing),
            "n_extraction_suspect": len(suspects),
            "suspect_rate": suspect_rate,
            "n_reasked": n_reasked,
            "n_reask_confirmed": len(confirmed) if self.reask else None,
            "n_extraction_point_miss": sum(c["extraction_point_miss"] for c in gradable),
            "missing_tag_rate": (
                round(sum(1 - c["has_answer_tag"] for c in gradable) / len(gradable), 4)
                if gradable
                else None
            ),
            "per_case": per_case,
            "_caveat": (
                "suspect_rate is the fraction of FAIL cases whose gold answer "
                "sits in the ANSWER REGION of the output (tail window or a "
                "tagged/boxed span) — HARNESS failures wearing a model-failure "
                "label. gold_in_output is the loose upper bound and over-counts: "
                "it fires on a gold that only appears as an intermediate "
                "quantity, so never read it as the suspect rate. Above ~5% the "
                "labels feeding M2 are contaminated and every other probe's "
                "correlations should be re-read after fixing extraction (or "
                "after dropping extraction_suspect==1 cases). This probe is "
                "OBSERVATIONAL by default: it re-reads stored outputs and "
                "introduces no new sampling. With reask=True the reask_* columns "
                "are interventional and must be re-run for held-out confirmation."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)

    # ------------------------------------------------------------------
    def _audit_case(self, case: "FailureCase") -> dict[str, Any]:
        from evalvitals.core.case import Label

        text = case.observed
        truncated = looks_truncated(text)
        finish_reason = (getattr(case, "metadata", {}) or {}).get("finish_reason")
        if finish_reason == "stop":
            truncated = False
        elif finish_reason == "length":
            truncated = True
        entry: dict[str, Any] = {
            "sample_id": case.id,
            "has_output": int(bool(str(text or "").strip())),
            "has_answer_tag": int(has_answer_tag(text)),
            "output_truncated": int(truncated),
            "gave_up": int(looks_like_give_up(text)),
            "output_chars": len(str(text or "")),
        }
        if case.expected is None or not str(text or "").strip():
            entry["skipped"] = "no gold answer or no stored output"
            return entry

        raw = str(text)
        extracted = self.answer_fn(text)
        strict = bool(self.match_fn(extracted, case.expected))
        gold = normalize_answer(case.expected)
        # Anywhere in the output — the loose UPPER bound; a gold that merely
        # appears as an intermediate quantity counts here and should not.
        anywhere = bool(self.match_fn(raw, case.expected)) or gold in normalize_answer(raw)
        # In the answer REGION: the tail of the generation, or any tagged /
        # boxed span. This is the one that means "the model did answer this".
        tail = raw[-self.tail_chars :]
        tagged_spans = " ".join(_BOXED.findall(raw) + _ANSWER_TAG.findall(raw))
        in_tail = bool(self.match_fn(tail, case.expected)) or bool(
            tagged_spans and self.match_fn(tagged_spans, case.expected)
        )
        entry.update(
            {
                "labelled_fail": int(case.label == Label.FAIL),
                "strict_match": int(strict),
                "gold_in_output": int(anywhere),
                "gold_in_answer_region": int(in_tail),
                "extracted_answer": str(extracted)[:120],
                # THE diagnosis: the harness called it a failure, yet the gold
                # answer sits in the answer region of the output.
                "extraction_suspect": int(case.label == Label.FAIL and in_tail),
                # narrower, grader-independent: present in the text but not at
                # the point the extractor reads
                "extraction_point_miss": int(in_tail and not strict),
                # the stored label disagrees with re-grading the stored output
                "label_disagrees": int((case.label == Label.FAIL) == strict),
            }
        )
        return entry

    def _reask(self, model: "Model", case: "FailureCase") -> dict[str, Any]:
        prompt = f"{case.inputs.prompt or ''}\n\n{_REASK_SUFFIX}"
        answer = str(model.generate(dataclasses.replace(case.inputs, prompt=prompt)))
        return {
            "reask_answer": answer[:120],
            "reask_correct": int(bool(self.match_fn(answer, case.expected))),
        }
