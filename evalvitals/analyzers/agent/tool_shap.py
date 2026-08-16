"""Tool-ablation Shapley — which tools does the outcome actually depend on?

AgentSHAP (TokenSHAP, arXiv:2407.10114) attributes an agent's response to its
TOOLS by re-running with tool subsets and Shapley-averaging the marginal
contributions.  This port changes the value function for diagnosis: the
primary signal is the **outcome** (did the run pass, via an injected grader),
not answer similarity — "the wording changed" is not "it failed".  Answer
similarity to the all-tools baseline is kept as a secondary value.

Sits beside :class:`CounterfactualReplay` in the intervention family: that one
perturbs a *step inside* the trajectory, this one perturbs the *tool
configuration* — attribution to steps vs attribution to the agent's setup
(which is exactly the L2 fix surface: tool subset selection).

Cost: with the small tool kits diagnosis uses (N<=4), all 2^N subsets are
enumerated and the Shapley values are EXACT; beyond that, leave-one-out plus
sampled subsets give a Monte-Carlo estimate (reported in the caveat).

The runner is INJECTED::

    def run_with_tools(case, tool_names):
        tools = [by_name[n](case) for n in tool_names]   # rebind per case
        return Agent(handle, tools, system=SYS).run(case)

    probe = ToolShap(run_with_tools=run_with_tools,
                     tool_names=["image_zoom_in", "image_detect"])
"""

from __future__ import annotations

import difflib
import itertools
import math
import random
import statistics
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.analyzers.agent.reliability import default_grader
from evalvitals.core.analyzer import Analyzer
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase, Trajectory
    from evalvitals.core.model import Model

_FULL_ENUM_MAX_TOOLS = 4  # 2^4 = 16 runs/case — exact Shapley up to here


def _answer_similarity(a: str, b: str) -> float:
    """Dependency-free answer similarity (difflib ratio on normalized text)."""
    na = " ".join(str(a or "").lower().split())
    nb = " ".join(str(b or "").lower().split())
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _exact_shapley(values: "dict[frozenset, float]", names: list[str]) -> dict[str, float]:
    """Exact Shapley over a fully enumerated subset->value table."""
    n = len(names)
    phi: dict[str, float] = {}
    for tool in names:
        others = [t for t in names if t != tool]
        total = 0.0
        for r in range(len(others) + 1):
            weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for combo in itertools.combinations(others, r):
                s = frozenset(combo)
                total += weight * (values[s | {tool}] - values[s])
        phi[tool] = total
    return phi


def _sampled_shapley(values: "dict[frozenset, float]", names: list[str]) -> dict[str, float]:
    """Monte-Carlo estimate: mean marginal over the (S, S+tool) pairs we ran."""
    phi: dict[str, float] = {}
    for tool in names:
        margins = [
            values[s | {tool}] - values[s]
            for s in values
            if tool not in s and (s | {tool}) in values
        ]
        phi[tool] = statistics.fmean(margins) if margins else 0.0
    return phi


@register_analyzer("tool_shap")
class ToolShap(Analyzer):
    """Shapley attribution of the agent's outcome (and answer) to its tools.

    Hyper-parameters:
        run_with_tools:   ``callable(case, tuple[tool_names]) -> Trajectory`` —
                          REQUIRED.  Must honor the subset (only those tools
                          available) and stay deterministic per subset.
        tool_names:       the full tool kit being attributed.
        grader:           ``callable(trajectory, case) -> bool | None``
                          (default :func:`default_grader`); grades each subset
                          run for the outcome value function.
        max_combinations: cap on non-essential subset runs per case; beyond
                          full enumeration the estimate is Monte-Carlo.
        max_cases:        label-balanced cap on probed cases.
        seed:             sampling seed for the Monte-Carlo path.
    """

    name = "tool_shap"
    requires = frozenset()  # the injected runner encapsulates the model
    #: Reads agent runs. The model is not what makes this applicable —
    #: the DATA is, which is why `requires` stays empty (these run on
    #: trajectories loaded from disk, with no model at all).
    requires_trajectories = True
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        run_with_tools: Callable[["FailureCase", tuple], "Trajectory"],
        tool_names: list[str],
        grader: Optional[Callable[["Trajectory", "FailureCase"], Optional[bool]]] = None,
        max_combinations: Optional[int] = None,
        max_cases: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            tool_names=list(tool_names),
            max_combinations=max_combinations,
            max_cases=max_cases,
            seed=seed,
        )
        if not tool_names:
            raise ValueError("tool_shap needs at least one tool name")
        # stored under the ctor names so sklearn-style get_params() reflection works
        self.run_with_tools = run_with_tools
        self.grader = grader or default_grader

    # -- subset schedule ------------------------------------------------
    def _subsets(self) -> "tuple[list[frozenset], bool]":
        """All subsets to run (full set included) and whether enumeration is exact."""
        names = self.tool_names
        n = len(names)
        full = [frozenset(c) for r in range(n + 1) for c in itertools.combinations(names, r)]
        if n <= _FULL_ENUM_MAX_TOOLS and (
            self.max_combinations is None or len(full) <= self.max_combinations
        ):
            return full, True
        essential = [frozenset(names)]  # baseline
        essential.append(frozenset())  # no tools at all
        essential += [frozenset(names) - {t} for t in names]  # leave-one-out
        rest = [s for s in full if s not in essential]
        budget = max(0, (self.max_combinations or len(full)) - len(essential))
        rng = random.Random(self.seed)
        return essential + rng.sample(rest, min(budget, len(rest))), False

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        selected = cases.stratified_head(self.max_cases) if self.max_cases else list(cases)
        subsets, exact = self._subsets()
        names = self.tool_names
        per_case: list[dict[str, Any]] = []

        for case in selected:
            answers: dict[frozenset, str] = {}
            outcome_v: dict[frozenset, float] = {}
            sim_v: dict[frozenset, float] = {}
            runs_audit: list[dict] = []
            gradable = True
            for subset in subsets:
                traj = self.run_with_tools(case, tuple(sorted(subset)))
                answer = str(traj.final_answer or "")
                answers[subset] = answer
                grade = self.grader(traj, case)
                if grade is None:
                    gradable = False
                outcome_v[subset] = 1.0 if grade else 0.0
                runs_audit.append(
                    {
                        "tools": sorted(subset),
                        "passed": grade,
                        "n_tool_calls": int(traj.metrics.get("n_tool_calls", 0)),
                        "terminated": str(traj.metrics.get("terminated", "")),
                    }
                )
            baseline = answers[frozenset(names)]
            for subset, answer in answers.items():
                sim_v[subset] = _answer_similarity(answer, baseline)

            shap = _exact_shapley if exact else _sampled_shapley
            entry: dict[str, Any] = {
                "sample_id": case.id,
                "n_subset_runs": len(subsets),
                "baseline_pass": (
                    int(outcome_v[frozenset(names)]) if gradable else None
                ),
                "no_tools_pass": int(outcome_v[frozenset()]) if gradable else None,
                "runs": runs_audit,
            }
            if gradable:
                entry["tools_needed"] = int(
                    outcome_v[frozenset(names)] > outcome_v[frozenset()]
                )
                for tool, phi in shap(outcome_v, names).items():
                    entry[f"shap_outcome_{tool}"] = round(phi, 4)
            for tool, phi in shap(sim_v, names).items():
                entry[f"shap_answer_{tool}"] = round(phi, 4)
            per_case.append(entry)

        findings: dict[str, Any] = {
            "n_trajectories": len(per_case),
            "tool_names": names,
            "exact": exact,
            "runs_per_case": len(subsets),
            "per_case": per_case,
            "_caveat": (
                ("Exact Shapley (all 2^N subsets enumerated). " if exact else
                 "Monte-Carlo Shapley over sampled subsets — treat values as estimates. ")
                + "shap_outcome_* attributes PASSING to each tool (needs a gradable "
                "expected answer); shap_answer_* attributes the produced answer "
                "(similarity to the all-tools baseline) and is defined even without "
                "labels. Deterministic runners are assumed: each subset is run once. "
                "These are INTERVENTIONAL columns: held-out verification must "
                "RE-RUN the subset ablation on the held-out cases — never reuse "
                "exploration-set values. baseline_pass mechanically tracks the "
                "case label when labels come from the same runner config; treat "
                "it as sanity, not as a candidate signal."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
