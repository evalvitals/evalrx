"""Next experiments the fixed plan did not run: proposed by rule, checked, never run.

A run without internal state must still be able to name its next behavioural
experiment. Every route with FAIL cases on this sample gets the follow-ups of
the method table, behavioural ones first; each is built with the host's
extras and gated on the actual model and sample, so the report says which
can run here and why the others cannot. Nothing is executed or scored:
choosing among them is the adaptive controller's job (MVP-3), and only
configurations the tools already support are proposed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from evalrx.analyzers.perturbation.prompt_contrast import _DEFAULT_STRATEGIES as _PROMPT_STRATEGIES
from evalrx.eval_agent.probing.actions import canonical
from evalrx.eval_agent.probing.executor import gate

DESCRIBE_STRATEGIES = {"baseline": "{prompt}", "describe_first": _PROMPT_STRATEGIES["describe_first"]}


@dataclass(frozen=True)
class FollowUp:
    """One candidate: the route it serves, the tool and configuration, the question it asks."""

    route: str
    analyzer: str
    config: dict[str, Any]
    question: str
    task_kinds: tuple[str, ...] = ()
    caveat: str = ""


FOLLOW_UPS: tuple[FollowUp, ...] = (
    FollowUp("H", "answer_extraction_audit", {"reask": True},
             "does a re-ask with a stricter answer request recover the answer the output lost"),
    FollowUp("H", "termination_audit", {"continue_non_clean": True},
             "continued past where it stopped, does the output reach an answer"),
    FollowUp("N", "prompt_contrast", {"strategies": DESCRIBE_STRATEGIES},
             "asked to describe the media first, does the answer start to follow it",
             caveat="the runtime's output cap must leave room for the description"),
    FollowUp("N", "causal_trace", {"target": "gold"},
             "which layers and positions carry the media's effect on the gold margin"),
    FollowUp("U", "self_consistency", {"n": 10},
             "with twice the samples, how often does the answer move"),
    FollowUp("U", "logprob_entropy", {},
             "is the answer that moves a near tie in the answer-token probabilities"),
    FollowUp("U", "format_sensitivity", {},
             "with the options rotated, does the answer follow content or position",
             task_kinds=("multiple_choice_letter",)),
    FollowUp("S", "prompt_contrast", {"strategies": DESCRIBE_STRATEGIES},
             "does a stable error survive when the model first states what it sees",
             caveat="the runtime's output cap must leave room for the description"),
    FollowUp("S", "gradcam", {"target": "gold"},
             "which image regions the gold margin depends on (no reference regions: "
             "concentration only, not 'mislocalised')"),
    FollowUp("S", "tuned_lens", {},
             "a fitted per-layer readout of the answer position: at what depth the wrong answer forms"),
)

#: Not a tool call: extending the nested sample is an action of its own that
#: the controller cannot take yet.
EXPAND = {"analyzer": "(nested sample expansion)", "status": "not_implemented",
          "reason": "expansion is not a controller action yet (MVP-2); rerun with a larger --triage-n-fail"}


def _route_cases(records: list[dict[str, Any]], route: str) -> list[str]:
    return [r["case_id"] for r in records if r["label"] == "fail" and route in r["routes"]]


def propose(records: list[dict[str, Any]], model: Any, sample: Any, *, task_kind: "str | None",
            extras: "dict[str, dict[str, Any]] | None" = None,
            executed: "set[tuple[str, str]] | None" = None) -> dict[str, Any]:
    """Follow-ups per route with FAIL cases, each with ``status``:
    ``executable`` / ``unsupported`` / ``not_applicable`` / ``already_run`` / ``could_not_build``.

    *executed* holds ``(analyzer, canonical config)`` pairs this run already
    executed (the plan's own steps). *extras* are the host's constructor extras
    per analyzer, as for the plan.
    """
    from evalrx.core.registry import registry

    extras = extras or {}
    executed = executed or set()
    out: list[dict[str, Any]] = []
    for route in ("H", "N", "U", "S"):
        cases = _route_cases(records, route)
        if not cases:
            continue
        for f in (x for x in FOLLOW_UPS if x.route == route):
            config = {**f.config, **extras.get(f.analyzer, {})}
            row: dict[str, Any] = {"route": route, "n_fail_cases": len(cases), "analyzer": f.analyzer,
                                   "config": canonical(f.config), "question": f.question}
            if f.caveat:
                row["caveat"] = f.caveat
            if f.task_kinds and task_kind not in f.task_kinds:
                out.append({**row, "status": "not_applicable", "reason": f"for task kinds {list(f.task_kinds)}"})
                continue
            if (f.analyzer, str(canonical(config))) in executed:
                out.append({**row, "status": "already_run", "reason": "the plan ran this configuration"})
                continue
            try:
                analyzer = registry.analyzers.get(f.analyzer)(**config)
            except Exception as exc:  # noqa: BLE001 - a proposal that cannot be built says so
                out.append({**row, "status": "could_not_build", "reason": f"{type(exc).__name__}: {exc}"})
                continue
            why = gate(analyzer, model, sample)
            out.append({**row, "status": "unsupported" if why else "executable", "reason": why or ""})
        out.append({"route": route, "n_fail_cases": len(cases), **EXPAND})
    untested = Counter(reason.split(":")[0] for r in records if r["routes"] == ["X"] for reason in r["x_reasons"])
    return {"candidates": out, "x_untested": dict(untested),
            "_note": "proposed by rule from the routes on this sample, gated on this model; none was run"}
