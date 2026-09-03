"""Coverage-vs-verification gap — did the model *have* the answer and fail to pick it?

pass@k and accuracy answer different questions, and the difference decides the
fix tier.  If the gold answer appears in *some* of k samples but the majority
vote lands elsewhere, the capability is present and the SELECTION is broken —
a reranker, a verifier, or a better aggregation (L1/L2) recovers it outright.
If no sample is ever right, no amount of scaffolding helps and the fix has to
change the model (L3/L4).  Reporting one accuracy number cannot tell these
apart, and the second is by far the more expensive mistake.

The optional self-verification arm goes one step further: it hands the model
its own k candidates and asks it to choose.  The distance between ``pass_at_k``
and ``verify_correct`` is the headroom a self-verifier can actually reach, as
opposed to the headroom that exists in principle.

Cost: k generations per case (+1 with ``self_verify=True``).

References:
- Evaluating Large Language Models Trained on Code — Chen et al., 2021 —
  arXiv:2107.03374 (the unbiased pass@k estimator)
- Self-Consistency Improves Chain of Thought Reasoning — Wang et al., ICLR 2023
  — arXiv:2203.11171 (majority@k)
- Large Language Monkeys: Scaling Inference Compute with Repeated Sampling —
  Brown et al., 2024 — arXiv:2407.21787 (coverage vs verification framing)
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalrx.analyzers.reasoning._text import (
    answer_equal,
    extract_answer,
    normalize_answer,
)
from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model

_VERIFY = (
    "Here are candidate answers produced for the question above:\n\n{candidates}\n\n"
    "Decide which one is correct. Give only that answer on its own last line as "
    "'Answer: <answer>'."
)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for *c* correct out of *n* samples (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return round(1.0 - math.comb(n - c, k) / math.comb(n, k), 4)


@register_analyzer("coverage_verification_gap")
class CoverageVerificationGap(Analyzer):
    """Sample k answers and split "cannot solve" from "cannot select".

    Hyper-parameters:
        k:           samples per case.
        self_verify: +1 generation asking the model to choose among its own k.
        max_cases:   label-stratified cap (k generations each); 0 (the default) = every case.
        gen_kwargs:  passed to ``model.generate`` — REQUIRES temperature > 0;
                     at temperature 0 all k samples are identical and the gap is
                     structurally 0.
        grader:      ``callable(prediction, case) -> bool | None`` — injected the
                     same way as the agent-side ``reliability_probe``.
        answer_fn:   ``callable(text) -> str`` answer extractor.
    """

    name = "coverage_verification_gap"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        k: int = 5,
        self_verify: bool = False,
        max_cases: int = 0,
        gen_kwargs: Optional[dict] = None,
        grader: Optional[Callable[[Any, "FailureCase"], Optional[bool]]] = None,
        answer_fn: Optional[Callable[[Any], str]] = None,
    ) -> None:
        super().__init__(
            k=max(int(k), 1),
            self_verify=self_verify,
            max_cases=max_cases,
            gen_kwargs=dict(gen_kwargs or {}),
        )
        # ctor names, so sklearn-style get_params() reflection works
        self.grader = grader or _default_grader
        self.answer_fn = answer_fn or extract_answer

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case: list[dict[str, Any]] = []
        for case in cases.stratified_head(self.max_cases):
            if case.expected is None:
                per_case.append({"sample_id": case.id, "skipped": "no gold answer"})
                continue
            per_case.append(self._probe_case(model, case))

        scored = [c for c in per_case if "coverage_gap" in c]
        degenerate = bool(scored) and all(c["n_unique"] == 1 for c in scored)
        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "n_scored": len(scored),
            "k": self.k,
            "gen_kwargs": dict(self.gen_kwargs),
            "mean_pass_at_k": _mean([c["pass_at_k"] for c in scored]),
            "mean_majority_correct": _mean([c["majority_correct"] for c in scored]),
            "coverage_gap_rate": _mean([c["coverage_gap"] for c in scored]),
            "mean_verify_correct": _mean(
                [c["verify_correct"] for c in scored if "verify_correct" in c]
            ),
            "no_coverage_rate": _mean([1 - c["pass_at_k"] for c in scored]),
            "degenerate_sampling": degenerate,
            "per_case": per_case,
            "_caveat": (
                "coverage_gap = 'the answer was in the pool, the vote missed it' "
                "⇒ a selection defect, fixable at L1/L2 with a verifier or "
                "reranker. no_coverage_rate = 'never produced once in k' ⇒ a "
                "capability defect no scaffold reaches; do not let a headline "
                "pass@k hide which of the two dominates. REQUIRES sampling: "
                "degenerate_sampling=true means every sample was identical "
                "(temperature 0 or a deterministic backend) and the gap is "
                "structurally zero, not measured. pass_at_k here uses the "
                "unbiased estimator over exactly k draws, so with k small its "
                "variance is large per case — read the batch rate. "
                "INTERVENTIONAL: held-out confirmation must re-sample."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)

    # ------------------------------------------------------------------
    def _probe_case(self, model: "Model", case: "FailureCase") -> dict[str, Any]:
        samples = [
            str(model.generate(case.inputs, **self.gen_kwargs)) for _ in range(self.k)
        ]
        answers = [self.answer_fn(s) for s in samples]
        grades = [self.grader(s, case) for s in samples]
        graded = [g for g in grades if g is not None]
        entry: dict[str, Any] = {"sample_id": case.id, "n_samples": len(samples)}
        if not graded:
            entry["skipped"] = "no sample could be graded"
            return entry

        n_correct = sum(graded)
        counts = Counter(normalize_answer(a) for a in answers)
        modal, modal_n = counts.most_common(1)[0]
        modal_correct = bool(answer_equal(modal, case.expected))

        entry.update(
            {
                "n_graded": len(graded),
                "n_correct": n_correct,
                "n_unique": len(counts),
                "pass_at_k": pass_at_k(len(graded), n_correct, len(graded)),
                "any_correct": int(n_correct > 0),
                "majority_correct": int(modal_correct),
                "majority_share": round(modal_n / len(answers), 4),
                # the diagnostic column: present in the pool, missed by the vote
                "coverage_gap": int(n_correct > 0 and not modal_correct),
                "modal_answer": str(modal)[:80],
            }
        )
        if self.self_verify:
            listing = "\n".join(
                f"{i + 1}. {a}" for i, a in enumerate(dict.fromkeys(answers))
            )
            chosen = model.generate(
                dataclasses.replace(
                    case.inputs,
                    prompt=f"{case.inputs.prompt or ''}\n\n{_VERIFY.format(candidates=listing)}",
                )
            )
            verdict = self.grader(chosen, case)
            if verdict is not None:
                entry["verify_correct"] = int(verdict)
                # how much of the available headroom self-verification captured
                entry["verifier_gap"] = int(n_correct > 0 and not verdict)
        return entry


def _mean(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _default_grader(prediction: Any, case: "FailureCase") -> Optional[bool]:
    if case.expected is None:
        return None
    return answer_equal(extract_answer(prediction), case.expected)
