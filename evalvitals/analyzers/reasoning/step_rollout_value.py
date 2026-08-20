"""Step-value rollouts — *where* did the reasoning chain break?

Knowing that a chain failed is not actionable; knowing it was already doomed at
step 2 is.  Math-Shepherd's insight is that a step's value can be estimated
without any human step labels: continue from that prefix ``n`` times and count
how often the completion reaches the gold answer.  The step where that value
collapses is the break point — everything after it is downstream noise, and a
fix targeted at the last visible mistake is aimed at the wrong place.

This is the text counterpart of the agent-side ``first_error_judge``: same
question (which step is the first error?), answered by rollout statistics
instead of an LLM judge, so it carries no judge bias — at the cost of
``n_steps × n_rollouts`` generations, which is why ``max_cases`` defaults low.

References:
- Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human
  Annotations — Wang et al., ACL 2024 — arXiv:2312.08935
- Let's Verify Step by Step — Lightman et al., ICLR 2024 — arXiv:2305.20050
- Improve Mathematical Reasoning with Process Supervision — Luo et al., 2024 —
  arXiv:2406.06592 (binary-search variant of the same estimator)
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.analyzers.reasoning._text import answer_equal, extract_answer
from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model

_COT_SUFFIX = (
    "Think step by step, one step per line. After the reasoning, give the final "
    "answer on its own last line as 'Answer: <answer>'."
)
_CONTINUE = (
    "Reasoning so far:\n{prefix}\n\nContinue from here and finish the solution. "
    "Give the final answer on its own last line as 'Answer: <answer>'."
)
_ANSWER_LINE = re.compile(r"^\s*(?:final\s+)?answer\s*[:=]", re.IGNORECASE)


def split_steps(text: Any, max_steps: int) -> list[str]:
    """Split a chain into reasoning steps (lines first, sentences as fallback)."""
    raw = str(text or "")
    lines = [ln.strip() for ln in raw.splitlines()]
    steps = [ln for ln in lines if ln and not _ANSWER_LINE.match(ln)]
    if len(steps) < 2:  # single-paragraph chain — fall back to sentences
        body = _ANSWER_LINE.split(raw)[0]
        steps = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    if len(steps) <= max_steps:
        return steps
    # keep the shape of the chain: evenly spaced cut points, always incl. the last
    idx = sorted({round(i * (len(steps) - 1) / (max_steps - 1)) for i in range(max_steps)})
    return [steps[i] for i in idx]


@register_analyzer("step_rollout_value")
class StepRolloutValueAnalyzer(Analyzer):
    """Estimate a per-step success value by rollouts and locate the break step.

    Hyper-parameters:
        n_rollouts:     completions sampled per step prefix.
        max_steps:      cap on probed steps (the chain is subsampled evenly).
        drop_threshold: value drop between consecutive steps that counts as the break.
        max_cases:      label-stratified cap — cost is n_steps × n_rollouts each.
        gen_kwargs:     passed to ``model.generate`` (rollouts NEED temperature > 0;
                        a deterministic model makes every rollout identical and the
                        values collapse to 0/1 — the caveat says so).
        grader/answer_fn: grading of each rollout against ``case.expected``.
    """

    name = "step_rollout_value"
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        n_rollouts: int = 3,
        max_steps: int = 7,
        drop_threshold: float = 0.34,
        max_cases: int = 8,
        gen_kwargs: Optional[dict] = None,
        grader: Optional[Callable[[Any, "FailureCase"], Optional[bool]]] = None,
        answer_fn: Optional[Callable[[Any], str]] = None,
    ) -> None:
        super().__init__(
            n_rollouts=n_rollouts,
            max_steps=max_steps,
            drop_threshold=drop_threshold,
            max_cases=max_cases,
            gen_kwargs=dict(gen_kwargs or {}),
        )
        self.grader = grader or _default_grader
        self.answer_fn = answer_fn or extract_answer

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        per_case: list[dict[str, Any]] = []
        step_values_by_case: dict[str, list[float]] = {}
        for case in cases.stratified_head(self.max_cases):
            if case.expected is None:
                per_case.append({"sample_id": case.id, "skipped": "no gold answer"})
                continue
            chain = case.observed
            if not str(chain or "").strip():
                chain = str(
                    model.generate(
                        dataclasses.replace(
                            case.inputs, prompt=f"{case.inputs.prompt or ''}\n\n{_COT_SUFFIX}"
                        )
                    )
                )
            entry = self._probe_case(model, case, str(chain))
            # Contract: a per-case row carries one level of scalars — a numeric
            # vector there looks like a signal and reaches no statistic. The
            # trajectory moves to findings["step_values_by_case"]; every scalar
            # the stats read (initial/final/min/max_value_drop/break_step_idx)
            # stays on the row.
            values = entry.pop("step_values", None)
            if values is not None:
                step_values_by_case[entry["sample_id"]] = values
            per_case.append(entry)

        scored = [c for c in per_case if "break_step_idx" in c]
        breaks = [c["break_step_idx"] for c in scored if c["break_step_idx"] is not None]
        findings: dict[str, Any] = {
            "n_cases": len(per_case),
            "n_scored": len(scored),
            "n_rollouts": self.n_rollouts,
            "gen_kwargs": dict(self.gen_kwargs),
            "mean_initial_value": _mean([c["initial_value"] for c in scored]),
            "mean_final_value": _mean([c["final_value"] for c in scored]),
            "mean_break_depth": _mean([c["break_depth"] for c in scored
                                       if c.get("break_depth") is not None]),
            "n_with_break": len(breaks),
            "per_case": per_case,
            "step_values_by_case": step_values_by_case,
            "_caveat": (
                "Step values are Monte-Carlo estimates from n_rollouts "
                "completions: with n_rollouts=3 a value is one of {0, .33, .67, "
                "1} and a single-step 'drop' of 0.33 is within sampling noise — "
                "raise n_rollouts before trusting a break_step_idx on an "
                "individual case, and prefer the batch-level mean_break_depth. "
                "The estimator REQUIRES sampling: at temperature 0 every rollout "
                "from a prefix is identical, values are 0/1, and break_step_idx "
                "degenerates into 'the first step whose continuation is wrong'. "
                "initial_value is the model's own unconditional success rate — a "
                "chain that starts near 0 has no break point to find, it never "
                "worked. INTERVENTIONAL: held-out confirmation must re-roll."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)

    # ------------------------------------------------------------------
    def _probe_case(self, model: "Model", case: "FailureCase", chain: str) -> dict[str, Any]:
        steps = split_steps(chain, self.max_steps)
        entry: dict[str, Any] = {"sample_id": case.id, "n_steps": len(steps)}
        if not steps:
            entry["skipped"] = "no reasoning steps to roll out from"
            return entry

        values: list[float] = []
        for i in range(len(steps)):
            prefix = "\n".join(steps[: i + 1])
            prompt = f"{case.inputs.prompt or ''}\n\n{_CONTINUE.format(prefix=prefix)}"
            hits = 0
            graded = 0
            for _ in range(self.n_rollouts):
                out = model.generate(
                    dataclasses.replace(case.inputs, prompt=prompt), **self.gen_kwargs
                )
                verdict = self.grader(out, case)
                if verdict is None:
                    continue
                graded += 1
                hits += int(verdict)
            values.append(round(hits / graded, 4) if graded else 0.0)

        entry["step_values"] = values
        entry["initial_value"] = values[0]
        entry["final_value"] = values[-1]
        entry["min_value"] = min(values)
        drops = [values[i - 1] - values[i] for i in range(1, len(values))]
        worst = max(drops) if drops else 0.0
        entry["max_value_drop"] = round(worst, 4)
        break_idx = next(
            (i + 1 for i, d in enumerate(drops) if d >= self.drop_threshold), None
        )
        entry["break_step_idx"] = break_idx
        entry["break_depth"] = (
            round(break_idx / max(len(steps) - 1, 1), 4) if break_idx is not None else None
        )
        entry["break_step_text"] = steps[break_idx][:200] if break_idx is not None else None
        # A chain whose value never recovers is unsalvageable by continuation;
        # one that dips and recovers is a detour, not a break.
        entry["recoverable"] = int(values[-1] > 0)
        return entry


def _mean(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _default_grader(prediction: Any, case: "FailureCase") -> Optional[bool]:
    if case.expected is None:
        return None
    return answer_equal(extract_answer(prediction), case.expected)
