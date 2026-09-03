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
import re
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalrx.analyzers.reasoning._text import (
    _ANSWER_TAG,
    _BOXED,
    answer_equal,
    binary_direction,
    binary_gold,
    extract_answer,
    has_answer_tag,
    looks_like_give_up,
    looks_truncated,
    normalize_answer,
)
from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability, CapabilityError
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model

_REASK_SUFFIX = (
    "Reply with ONLY the final answer, no explanation, no units, no punctuation."
)


def _gold_scalar(expected: Any) -> str:
    if isinstance(expected, (list, tuple)):
        return str(expected[0]) if expected else ""
    return str(expected)


def _contract_answer(text: Any, contract: dict[str, Any]) -> str:
    """Return a committed short answer under a benchmark output contract.

    A single-letter gold must never match the first character of prose such as
    ``Audio Analysis``.  We accept an explicit Answer/Final marker or a final
    line containing only the answer, matching the benchmark's response shape.
    """
    raw = str(text or "")
    kind = str(contract.get("kind") or "").lower()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    final_line = lines[-1] if lines else raw.strip()

    if kind == "multiple_choice_letter":
        choices = "".join(str(c).strip()[:1].upper() for c in contract.get("choices", [])) or "ABCD"
        cls = re.escape(choices)
        marked = re.findall(
            rf"(?:\banswer|\bfinal(?:\s+answer)?)\s*(?:is\b\s*|[:=\-]\s*)"
            rf"[\(\[]?([{cls}])[\)\]]?\b",
            raw,
            flags=re.IGNORECASE,
        )
        boxed = re.findall(rf"\\boxed\s*\{{\s*([{cls}])\s*\}}", raw, flags=re.IGNORECASE)
        if marked or boxed:
            return (marked + boxed)[-1].upper()
        bare = re.fullmatch(rf"\s*[\(\[]?([{cls}])[\)\]]?[\s.!]*", final_line, re.IGNORECASE)
        return bare.group(1).upper() if bare else ""

    if kind == "yes_no":
        marked = re.findall(
            r"(?:\banswer|\bfinal(?:\s+answer)?)\s*(?:is\b\s*|[:=\-]\s*)"
            r"(yes|no)\b",
            raw,
            flags=re.IGNORECASE,
        )
        if marked:
            return marked[-1].capitalize()
        bare = re.fullmatch(r"\s*(yes|no)[\s.!]*", final_line, re.IGNORECASE)
        return bare.group(1).capitalize() if bare else ""

    return ""


def _contract_anywhere(text: Any, expected: Any, contract: dict[str, Any]) -> bool:
    """Loose upper bound for contracted answers, without substring matching."""
    raw = str(text or "")
    gold = _gold_scalar(expected).strip()
    kind = str(contract.get("kind") or "").lower()
    if _contract_answer(raw, contract).lower() == gold.lower():
        return True
    if kind == "multiple_choice_letter" and len(gold) == 1:
        return bool(re.search(rf"(?<![A-Za-z]){re.escape(gold)}(?![A-Za-z])", raw, re.IGNORECASE))
    if kind == "yes_no":
        return bool(re.search(rf"\b{re.escape(gold)}\b", raw, re.IGNORECASE))
    return False


@register_analyzer("answer_extraction_audit")
class AnswerExtractionAudit(Analyzer):
    """Split FAIL cases into *really wrong* and *merely unparsed*.

    Hyper-parameters:
        reask:      re-ask each suspect case with a strict output format
                    (1 generation per suspect; needs ``GENERATE``).
        max_cases:  label-stratified cap; 0 (the default) = every case.
        tail_chars: size of the trailing window treated as the answer region.
        answer_fn:  ``callable(text) -> str`` answer extractor.
        match_fn:   ``callable(prediction, gold) -> bool`` equality test.
    """

    name = "answer_extraction_audit"
    # Capability-free on the default (read-only) path so it can audit an
    # offline record dump; the re-ask path checks GENERATE at run time.
    requires = frozenset()
    applies_to_modalities = frozenset({"text", "image"})
    signal_docs = {
        'answered_yes': ('Model said yes', 'On a yes/no question, whether the MODEL answered yes. Its direction, not whether it was right.'),
        'gold_yes': ('Correct answer is yes', 'On a yes/no question, whether the CORRECT answer is yes. A property of the question, not the model.'),
        'extracted_answer': 'The answer the grader pulled out of the text.',
        'extraction_point_miss': 'The right answer was in the right place and still did not match — a formatting mismatch.',
        'extraction_suspect': ('Possible grading miss', 'This case was marked wrong, but the right answer is there — a likely grading mistake.'),
        'gave_up': ('Model gave up', 'Whether the model said it could not answer.'),
        'gold_in_answer_region': ('Right answer in place', 'Whether the right answer appears in the part of the text the grader reads.'),
        'gold_in_output': ('Right answer somewhere', 'Whether the right answer appears anywhere in the text, even if not where it was asked for.'),
        'has_answer_tag': ('Used the answer format', 'Whether the model used the answer format it was asked for.'),
        'has_output': ('Answered at all', 'Whether the model wrote anything at all.'),
        'label_disagrees': "The benchmark's verdict and a strict re-check disagree about this case.",
        'labelled_fail': 'Whether the benchmark marked this answer wrong.',
        'missing_tag_rate': ('Wrong answer format', 'Share of answers that never used the requested answer format.'),
        'n_cases': 'How many answers were inspected.',
        'n_extraction_point_miss': 'Cases where the answer was in the right place but the exact-match check still failed — usually formatting.',
        'n_extraction_suspect': 'Wrong answers where the right answer was in fact present in the text — the grader may have missed it.',
        'n_gradable': 'How many answers had a gold answer to check against.',
        'n_labelled_fail': "How many were marked wrong by the benchmark's own grader.",
        'n_reask_confirmed': 'How many of those re-asks confirmed the model really was wrong.',
        'n_reasked': 'How many suspect cases were asked again to check.',
        'output_chars': ('Answer length', 'How long the answer was, in characters. Very long usually means the model rambled instead of answering.'),
        'output_truncated': ('Answer was cut off', 'Whether the answer was cut off by the length limit rather than finished.'),
        'reask_answer': 'What the model said when asked the same question again.',
        'strict_match': ('Exactly matched gold', 'Whether the answer matched the gold answer exactly.'),
        'suspect_rate': ('Possible grading misses', 'Share of wrong answers that may be grading mistakes rather than model mistakes. High means the failure numbers are not trustworthy yet.'),
    }

    def __init__(
        self,
        reask: bool = False,
        max_cases: int = 0,
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
                "are interventional and must be re-run for held-out confirmation. "
                "On yes/no (true/false) tasks answered_yes and gold_yes are the "
                "answer's and the gold's DIRECTION, two separate marginals (not "
                "re-grades): a directional hypothesis is tested on them, e.g. "
                "answered_yes HIGHER on failing cases."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)

    # ------------------------------------------------------------------
    def _audit_case(self, case: "FailureCase") -> dict[str, Any]:
        from evalrx.core.case import Label

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
        metadata = getattr(case, "metadata", {}) or {}
        contract = metadata.get("output_contract") or {}
        contract_kind = str(contract.get("kind") or "").lower()
        contracted = contract_kind in {"multiple_choice_letter", "yes_no"}
        extracted = _contract_answer(raw, contract) if contracted else self.answer_fn(text)
        strict = (
            extracted.lower() == _gold_scalar(case.expected).strip().lower()
            if contracted else bool(self.match_fn(extracted, case.expected))
        )
        gold = normalize_answer(case.expected)
        # Anywhere in the output — the loose UPPER bound; a gold that merely
        # appears as an intermediate quantity counts here and should not.
        anywhere = (
            _contract_anywhere(raw, case.expected, contract)
            if contracted
            else bool(self.match_fn(raw, case.expected)) or gold in normalize_answer(raw)
        )
        # In the answer REGION: the tail of the generation, or any tagged /
        # boxed span. This is the one that means "the model did answer this".
        tail = raw[-self.tail_chars :]
        tagged_spans = " ".join(_BOXED.findall(raw) + _ANSWER_TAG.findall(raw))
        in_tail = (
            _contract_answer(tail, contract).lower() == _gold_scalar(case.expected).strip().lower()
            if contracted
            else bool(self.match_fn(tail, case.expected)) or bool(
                tagged_spans and self.match_fn(tagged_spans, case.expected)
            )
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
        # Binary tasks only: the DIRECTION of the gold and of the answer, as two
        # separate marginals. Neither is a function of the label -- answered_yes
        # is model behaviour, gold_yes is a question covariate -- so both may
        # enter M2's tested family, and M5 can check a directional hypothesis
        # ("a Yes prior: answered_yes HIGHER on failures; failures concentrate
        # on gold_yes=0"). Their CONJUNCTION (answered Yes on a gold No) is a
        # subset of FAIL by construction and must never be a column here.
        # Live motivation: audiocaps_hallucination 2026-08-20, where M3 named
        # extracted_answer / labelled_fail and M5 had nothing numeric to test.
        gold_dir = binary_gold(case.expected)
        if gold_dir is not None:
            entry["gold_yes"] = int(gold_dir == "yes")
            answer_dir = binary_direction(extracted) or binary_direction(tail, last=True)
            if answer_dir is not None:
                entry["answered_yes"] = int(answer_dir == "yes")
        return entry

    def _reask(self, model: "Model", case: "FailureCase") -> dict[str, Any]:
        prompt = f"{case.inputs.prompt or ''}\n\n{_REASK_SUFFIX}"
        answer = str(model.generate(dataclasses.replace(case.inputs, prompt=prompt)))
        return {
            "reask_answer": answer[:120],
            "reask_correct": int(bool(self.match_fn(answer, case.expected))),
        }
