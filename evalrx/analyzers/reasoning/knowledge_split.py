"""Knowledge/reasoning split — is the fact missing, or is the chaining broken?

A wrong multi-hop answer has two incompatible causes.  Either the model never
knew the bridging fact (a *knowledge* deficit — retrieval, tools, or a bigger
model), or it knew every fact and still failed to combine them (a *reasoning*
deficit — decomposition scaffolds, which do nothing for missing knowledge).
Picking the wrong one costs a whole M4 tier.

The discriminating move needs no dataset annotation: ask the model to state the
facts it needs, then hand *its own* stated facts back and ask again.  If it now
answers correctly, the knowledge was present all along and the composition
failed.  When the case carries gold context (``metadata['context']`` /
``'facts'``) an open-book arm is added, which separates "does not know" from
"cannot retrieve from its own weights".

Cost: 4 generations per case, 5 with gold context.

References:
- MuSiQue: Multihop Questions via Single-hop Question Composition — Trivedi et
  al., TACL 2022 — arXiv:2108.00573 (composability vs single-hop knowledge)
- Measuring and Narrowing the Compositionality Gap in Language Models — Press et
  al., EMNLP Findings 2023 — arXiv:2210.03350
- Chain-of-Thought Prompting Elicits Reasoning — Wei et al., NeurIPS 2022 —
  arXiv:2201.11903 (decomposition arm)
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalrx.analyzers.reasoning._text import answer_equal, extract_answer
from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model

_ANSWER_FORMAT = "Give the final answer on its own last line as 'Answer: <answer>'."
_BASELINE = _ANSWER_FORMAT
_DECOMPOSE = (
    "Break this into the sub-questions you must answer first, answer each in "
    f"turn, then combine them. {_ANSWER_FORMAT}"
)
_RECALL = (
    "Do NOT answer the question yet. List only the specific facts you would "
    "need in order to answer it, one per line."
)
_FROM_FACTS = (
    "Facts you listed:\n{facts}\n\nUsing only these facts, answer the question. "
    f"{_ANSWER_FORMAT}"
)
_OPEN_BOOK = (
    "Reference material:\n{context}\n\nUsing this material, answer the question. "
    f"{_ANSWER_FORMAT}"
)

KNOWLEDGE = "knowledge"
REASONING = "reasoning"
BOTH = "both"
NONE = "none"
UNKNOWN = "unknown"


@register_analyzer("knowledge_reasoning_split")
class KnowledgeReasoningSplit(Analyzer):
    """Attribute each failure to a missing fact or a broken composition.

    Hyper-parameters:
        use_context:  add the open-book arm when the case carries gold context.
        context_keys: metadata keys searched for that context, in order.
        max_cases:    label-stratified cap (4–5 generations each); 0 (the default) = every case.
        grader:       ``callable(prediction, case) -> bool | None``.
    """

    name = "knowledge_reasoning_split"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        use_context: bool = True,
        context_keys: tuple = ("context", "facts", "supporting_facts", "passage"),
        max_cases: int = 0,
        grader: Optional[Callable[[Any, "FailureCase"], Optional[bool]]] = None,
    ) -> None:
        super().__init__(
            use_context=use_context,
            context_keys=tuple(context_keys),
            max_cases=max_cases,
        )
        self.grader = grader or _default_grader

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case: list[dict[str, Any]] = []
        for case in cases.stratified_head(self.max_cases):
            if case.expected is None:
                per_case.append({"sample_id": case.id, "skipped": "no gold answer"})
                continue
            per_case.append(self._probe_case(model, case))

        scored = [c for c in per_case if "deficit_class" in c]
        failed = [c for c in scored if c["baseline_correct"] == 0]

        # UNKNOWN is a residual, not a class: with no gold context the open-book
        # arm never runs and every failure the scaffolds did not rescue lands
        # there. Dividing by all failures would then report a MEASURED 0.0 for
        # knowledge_deficit_share when nothing was measured at all — the same
        # trap arith_audit avoids by excluding its own UNKNOWN from the
        # denominator.
        classified = [c for c in failed if c["deficit_class"] != UNKNOWN]

        def _share(cls: str) -> Optional[float]:
            return (
                round(sum(1 for c in classified if c["deficit_class"] == cls)
                      / len(classified), 4)
                if classified
                else None
            )

        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "n_scored": len(scored),
            "n_baseline_fail": len(failed),
            "n_classified": len(classified),
            "n_unclassified": len(failed) - len(classified),
            "unclassified_share": (
                round((len(failed) - len(classified)) / len(failed), 4) if failed else None
            ),
            "knowledge_deficit_share": _share(KNOWLEDGE),
            "reasoning_deficit_share": _share(REASONING),
            "both_deficit_share": _share(BOTH),
            "decomposition_gain": _gain(scored, "decomposed_correct"),
            "own_facts_gain": _gain(scored, "own_facts_correct"),
            "open_book_gain": _gain(scored, "open_book_correct"),
            "per_case": per_case,
            "_caveat": (
                "The three deficit shares are over CLASSIFIED failures only — "
                "read n_classified and unclassified_share first. Separating "
                "'knowledge' from 'both' REQUIRES gold context in "
                "case.metadata; without it the open-book arm never runs, every "
                "failure the scaffolds did not rescue is unclassified, and "
                "knowledge_deficit_share is None (not 0.0). A 0.0 here means "
                "'measured, none were knowledge deficits'; None means 'this "
                "batch cannot answer the question'. "
                "The arms differ in PROMPT as well as in information, so a gain "
                "is 'this scaffold helps', not proof of where the knowledge "
                "lives — own_facts_correct in particular re-states the question "
                "with a cleaner context, which alone helps some models. Read "
                "reasoning_deficit_share against a same-length no-op scaffold "
                "before promoting a decomposition fix. Cases where the model "
                "lists NO usable facts fall out as own_facts_parsed=0 and are "
                "excluded from own_facts_gain rather than counted as failures. "
                "open_book_* exists only for cases carrying gold context; its "
                "absence is 'not measured'. INTERVENTIONAL — re-run to confirm."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)

    # ------------------------------------------------------------------
    def _probe_case(self, model: "Model", case: "FailureCase") -> dict[str, Any]:
        entry: dict[str, Any] = {"sample_id": case.id}
        baseline = self.grader(model.generate(self._ask(case, _BASELINE)), case)
        decomposed = self.grader(model.generate(self._ask(case, _DECOMPOSE)), case)

        facts = str(model.generate(self._ask(case, _RECALL)))
        fact_lines = [ln.strip() for ln in facts.splitlines() if ln.strip()]
        entry["n_listed_facts"] = len(fact_lines)
        entry["own_facts_parsed"] = int(bool(fact_lines))
        own_facts = None
        if fact_lines:
            own_facts = self.grader(
                model.generate(
                    self._ask(case, _FROM_FACTS.format(facts="\n".join(fact_lines)))
                ),
                case,
            )

        context = self._context_for(case)
        open_book = None
        if context:
            open_book = self.grader(
                model.generate(self._ask(case, _OPEN_BOOK.format(context=context))), case
            )

        if baseline is None:
            entry["skipped"] = "ungradable baseline"
            return entry
        entry["baseline_correct"] = int(baseline)
        for key, value in (
            ("decomposed_correct", decomposed),
            ("own_facts_correct", own_facts),
            ("open_book_correct", open_book),
        ):
            if value is not None:
                entry[key] = int(value)

        entry["deficit_class"] = _classify(baseline, own_facts, decomposed, open_book)
        return entry

    def _ask(self, case: "FailureCase", instruction: str):
        prompt = f"{case.inputs.prompt or ''}\n\n{instruction}"
        return dataclasses.replace(case.inputs, prompt=prompt)

    def _context_for(self, case: "FailureCase") -> str:
        if not self.use_context:
            return ""
        for key in self.context_keys:
            value = case.metadata.get(key)
            if isinstance(value, (list, tuple)):
                value = "\n".join(str(v) for v in value)
            if value and str(value).strip():
                return str(value).strip()
        return ""


def _classify(
    baseline: bool,
    own_facts: Optional[bool],
    decomposed: Optional[bool],
    open_book: Optional[bool],
) -> str:
    """Which deficit does the arm pattern imply for a failing baseline?"""
    if baseline:
        return NONE
    # Scaffolding its own knowledge fixed it ⇒ the facts were there, the
    # composition was not.
    if own_facts or decomposed:
        return REASONING
    # Only the gold context fixed it ⇒ the facts were genuinely missing.
    if open_book:
        return KNOWLEDGE
    if open_book is False:
        # even with the facts handed over it still fails: both halves are broken
        return BOTH
    return UNKNOWN


def _gain(entries: list[dict[str, Any]], key: str) -> Optional[float]:
    """Mean arm-minus-baseline accuracy over the cases where the arm ran."""
    pairs = [(c["baseline_correct"], c[key]) for c in entries if key in c]
    if not pairs:
        return None
    return round(sum(arm - base for base, arm in pairs) / len(pairs), 4)


def _default_grader(prediction: Any, case: "FailureCase") -> Optional[bool]:
    if case.expected is None:
        return None
    return answer_equal(extract_answer(prediction), case.expected)
