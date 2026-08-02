"""Reliability probe — pass@k / pass^k over repeated agent runs.

A single run hides the capability/stability split: "reliably fails" and
"flaky" are different diagnoses with different fixes (prompt/tool surgery vs
decoding/consistency scaffolds).  This probe re-runs each case *k* times
through an injected runner and reports, per case, the pass@k family plus
answer- and trajectory-consistency measures.

Reference: Pass@k (capability: at least one of k passes) vs Pass^k
(reliability: all k pass) as popularised by agent-eval harnesses.

The runner is INJECTED — the analyzer never builds an agent itself::

    def runs_fn(case, k):
        # e.g. a temperature>0 endpoint, or per-rep seeds; use run_batch for concurrency
        return [Agent(handle, tools_factory(case), system=SYS).run(case) for _ in range(k)]

    probe = ReliabilityProbe(runs_fn=runs_fn, k=5)
    result = probe.run(model, cases)
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase, Trajectory
    from evalvitals.core.model import Model

_YESNO = re.compile(r"\b(yes|no)\b")


def default_grader(trajectory: "Trajectory", case: "FailureCase") -> Optional[bool]:
    """Grade one run against ``case.expected`` (``None`` = ungradable).

    yes/no expectations match the FIRST yes/no token in the answer; other
    expectations match as a case-insensitive substring.  Inject a task-specific
    grader for anything richer.
    """
    expected = case.expected
    if expected is None or trajectory.final_answer is None:
        return None
    answer = str(trajectory.final_answer).strip().lower()
    exp = str(expected).strip().lower()
    if exp in ("yes", "no"):
        m = _YESNO.search(answer)
        return bool(m and m.group(1) == exp)
    return exp in answer


def _normalized_answer(trajectory: "Trajectory") -> str:
    return " ".join(str(trajectory.final_answer or "").lower().split())[:200]


def _tool_sequence(trajectory: "Trajectory") -> tuple:
    return tuple(s.tool_call.get("name", "") for s in trajectory.steps if s.tool_call)


@register_analyzer("reliability_probe")
class ReliabilityProbe(Analyzer):
    """Re-run each case k times and measure pass@k / pass^k / consistency.

    Hyper-parameters:
        runs_fn:   ``callable(case, k) -> list[Trajectory]`` — REQUIRED.  The
                   caller owns sampling variation (temperature/seeds) and
                   concurrency; deterministic runners make every rep identical
                   and the probe degenerates (it will say so in the caveat).
        k:         repetitions per case.
        grader:    ``callable(trajectory, case) -> bool | None``; defaults to
                   :func:`default_grader`.
        max_cases: label-balanced cap on probed cases (k runs each is costly).
    """

    name = "reliability_probe"
    requires = frozenset()  # the injected runner encapsulates the model
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        runs_fn: Callable[["FailureCase", int], "list[Trajectory]"],
        k: int = 5,
        grader: Optional[Callable[["Trajectory", "FailureCase"], Optional[bool]]] = None,
        max_cases: Optional[int] = None,
    ) -> None:
        super().__init__(k=k, max_cases=max_cases)
        # stored under the ctor names so sklearn-style get_params() reflection works
        self.runs_fn = runs_fn
        self.grader = grader or default_grader

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        selected = cases.stratified_head(self.max_cases) if self.max_cases else list(cases)
        per_case: list[dict[str, Any]] = []
        for case in selected:
            trajectories = list(self.runs_fn(case, self.k) or [])
            if not trajectories:
                continue
            n_runs = len(trajectories)
            grades = [self.grader(t, case) for t in trajectories]
            graded = [g for g in grades if g is not None]

            answers = Counter(_normalized_answer(t) for t in trajectories)
            agreement = answers.most_common(1)[0][1] / n_runs
            sequences = Counter(_tool_sequence(t) for t in trajectories)
            calls = [
                int(t.metrics.get("n_tool_calls", sum(1 for s in t.steps if s.tool_call)))
                for t in trajectories
            ]

            entry: dict[str, Any] = {
                "sample_id": case.id,
                "n_runs": n_runs,
                "answer_agreement": round(agreement, 4),
                "answer_consistent": 1 if len(answers) == 1 else 0,
                "tool_seq_diversity": round(len(sequences) / n_runs, 4),
                "n_tool_calls_mean": round(statistics.fmean(calls), 3) if calls else 0.0,
                "n_tool_calls_std": round(statistics.pstdev(calls), 3) if len(calls) > 1 else 0.0,
            }
            if graded:
                n_pass = sum(graded)
                entry.update(
                    {
                        "n_graded": len(graded),
                        "n_pass": n_pass,
                        "success_rate": round(n_pass / len(graded), 4),
                        "pass_at_k": 1 if n_pass > 0 else 0,
                        "pass_all_k": 1 if n_pass == len(graded) else 0,
                        "flaky": 1 if 0 < n_pass < len(graded) else 0,
                    }
                )
            per_case.append(entry)

        rates = [c["success_rate"] for c in per_case if "success_rate" in c]
        all_consistent = per_case and all(c["answer_consistent"] == 1 for c in per_case)
        findings: dict[str, Any] = {
            "n_trajectories": len(per_case),
            "k": self.k,
            "n_graded_cases": len(rates),
            "mean_success_rate": round(statistics.fmean(rates), 4) if rates else None,
            "frac_flaky": (
                round(sum(c.get("flaky", 0) for c in per_case) / len(rates), 4) if rates else None
            ),
            "per_case": per_case,
            "_caveat": (
                "Ungraded cases (no expected answer) report consistency only. "
                + (
                    "Every case produced k identical answers — the runner looks "
                    "deterministic; use temperature>0 or per-rep seeds for a "
                    "meaningful reliability read. "
                    if all_consistent
                    else ""
                )
                + "pass@k measures capability, pass^k reliability; the gap is the "
                "instability mass."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
