"""MVP-3: the value of a candidate action and the scored scheduler's choice.

    U(a | E) = q(a) · [w_D·D(a) + w_T·T(a) + w_C·C(a) − w_R·R(a)]₊ / (c(a) + ε)

All gains are in [0, 1], ``c`` is on the run's primary budget axis (wall
seconds, or USD when every executed action has a price), and ``ε`` is that
axis' metering floor. The definitions are the design's first executable ones,
written down so a run can show why each action won:

* **D** (prediction disagreement): with an explainer, the share of pairs of the
  route's live competing explanations whose predictions for the step conflict
  (:mod:`.explanations`); the explanations come once per sample revision, and
  an explanation a prediction already missed on leaves the live set at the next
  decision. Without one, or with fewer than two live explanations, ``D = 0``
  and actions earn budget through T and C. An expansion's D is 0.
* **T** (evidence gap): 1 completes a recorded gap — a layer done on a smaller
  sample whose step only has to run on the route's new cases (a ``reuse``
  step), or a contradiction between two readings of the same check; 0.5 is an
  expansion that may close a recorded uncertainty (an uncertain routing
  signal, a small S cohort); 0 otherwise. A batch-dependent step that went
  stale re-runs on the whole sample: its value is the coverage it restores (C),
  not a gap of its own. Every T names the gap ids it serves.
* **C** (coverage): units are ``layer × gold stratum`` for every route with FAIL
  cases there. A unit's gap is 1 untested, 0.5 measured on fewer than
  ``min(min_cover, route FAIL cases in the stratum)`` cases, else 0. ``C(a)`` is
  the gap the action is expected to remove, over the number of units.
* **R** (redundancy): an observation that already exists is filtered out (R = 1);
  another tool for a layer that is already answered gets 0.5 (same purpose,
  same kind of evidence); otherwise 0.
* **q** (execution availability): ``(successes + 1) / (attempts + 2)`` over the
  distinct observations of the analyzer in this run's evidence (a read-back
  counts once, so a resumed run decides the same way); ``ok`` and not refused
  by its own check is a success, and ``unsupported`` (a gate result, filtered
  before scoring) is not an attempt.
* **c** (marginal cost): the step's own earlier cost per case, else the per-call
  rate of each method it calls (measured on this run), times the new cases.
  ``backward`` has no measured rate until a gradient tool runs; it is priced at
  three forwards (a prior, stated here).

Candidates are each route's open or stale layers (the layer's next candidate in
its fallback order, as in the rules), the expansion, and the batch-dependent
battery steps not yet run on the current sample. No batch-dependent candidate
(battery step or follow-up) is taken while an expansion can still follow: any
expansion supersedes its reading, and no route reads it (the rules' deferral,
for the same reason); its layer waits instead of falling back. Unlike the rules, the
scored scheduler may take a later layer first, and says so when it does. Every
``explore_every``-th decision picks, with a seed, among the feasible candidates
other than the top one. A gain below ``min_gain`` is not worth an action: when
no candidate reaches it, the run stops.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from evalrx.eval_agent.probing.actions import OK, UNSUPPORTED, PlanStep
from evalrx.eval_agent.probing.evidence import StepEvidence

SCORING_VERSION = "mvp3-score-2"
BACKWARD_FORWARDS = 3.0


@dataclass(frozen=True)
class Weights:
    d: float = 1.0
    t: float = 1.0
    c: float = 1.0
    r: float = 1.0


@dataclass
class Scored:
    """One candidate with its value components."""

    kind: str                          # "follow_up" | "extend" | "battery" | "expand"
    key: str                           # step key, or "expand"
    route: "str | None" = None
    rule_id: "str | None" = None
    order: tuple[int, int] = (99, 99)  # soft order: (route, layer); ties go to the earlier
    n_new_cases: int = 0
    D: float = 0.0
    T: float = 0.0
    C: float = 0.0
    R: float = 0.0
    q: float = 0.5
    cost: "float | None" = None
    gain: float = 0.0
    U: float = 0.0
    gaps: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    d_detail: dict[str, Any] = field(default_factory=dict)   # which live explanations D separates
    note: str = ""
    payload: Any = None                # the proposal / expansion plan; not serialised

    def row(self) -> dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()
                if k not in ("payload", "order")}


# -- q and c ---------------------------------------------------------------------------


def availability(analyzer: str, evidence: dict[str, StepEvidence]) -> float:
    """``(successes + 1) / (attempts + 2)`` over the analyzer's distinct observations."""
    seen: dict[str, bool] = {}
    for ev in evidence.values():
        if ev.step.analyzer != analyzer:
            continue
        for obs in ev.chunks:
            if obs.status == UNSUPPORTED:   # a gate result, filtered before scoring: not an attempt
                continue
            refused = (obs.fidelity or {}).get("status") == "failed"
            seen[obs.action_id] = obs.status == OK and not refused
    return (sum(seen.values()) + 1) / (len(seen) + 2)


def axis_of(evidence: dict[str, StepEvidence]) -> str:
    """``usd`` when every executed observation carries a price, else ``wall_sec``."""
    costs = [o.cost for ev in evidence.values() for o in ev.chunks if (o.cost or {}).get("calls")]
    if costs and all(c.get("usd") is not None for c in costs):
        return "usd"
    return "wall_sec"


EPSILON = {"wall_sec": 1.0, "usd": 1e-4}


def _amount(cost: dict[str, Any], axis: str) -> float:
    return float(cost.get(axis) or 0.0)


def method_rates(evidence: dict[str, StepEvidence], axis: str) -> dict[str, float]:
    """Cost per call by method, from observations that called one method only."""
    spent: dict[str, float] = {}
    calls: dict[str, int] = {}
    seen = set()
    for ev in evidence.values():
        for obs in ev.chunks:
            used = {k: int(v) for k, v in (obs.cost.get("calls") or {}).items() if v}
            if obs.action_id in seen or len(used) != 1:
                continue
            seen.add(obs.action_id)
            (method, n), = used.items()
            spent[method] = spent.get(method, 0.0) + _amount(obs.cost, axis)
            calls[method] = calls.get(method, 0) + n
    rates = {m: spent[m] / calls[m] for m in calls if calls[m]}
    if "backward" not in rates and "forward" in rates:
        rates["backward"] = BACKWARD_FORWARDS * rates["forward"]
    if "logprobs" not in rates and "generate" in rates:
        rates["logprobs"] = rates["generate"]
    return rates


def estimate_cost(step: PlanStep, n_cases: int, evidence: dict[str, StepEvidence], axis: str,
                  rates: dict[str, float]) -> "float | None":
    """Marginal cost of running *step* on *n_cases* new cases (``None`` = no basis yet)."""
    ev = evidence.get(step.name)
    rows = [(_amount(o.cost, axis), len(o.planned_ids)) for o in (ev.chunks if ev else [])
            if (o.cost or {}).get("calls") and o.planned_ids]
    if rows and sum(k for _, k in rows):
        return sum(s for s, _ in rows) / sum(k for _, k in rows) * n_cases
    if step.calls_per_case and all(m in rates for m in step.calls_per_case):
        return sum(rates[m] * k for m, k in step.calls_per_case.items()) * n_cases
    return None


# -- C: coverage units -------------------------------------------------------------------


def coverage_units(states_by_route: dict[str, list[dict[str, Any]]], records: list[dict[str, Any]],
                   evidence: dict[str, StepEvidence], layers: Any, *, min_cover: int) -> dict[str, dict[str, Any]]:
    """``{unit id: {layer, stratum, need, measured, gap, layer_state}}`` for every layer of every route
    with FAIL cases. A unit of an ``exhausted`` layer keeps its gap: no tool can close it on this model."""
    by_rule = {x.rule_id: x for x in layers}
    stratum_of = {r["case_id"]: r["stratum"] for r in records}
    units: dict[str, dict[str, Any]] = {}
    for route, states in states_by_route.items():
        for st in states:
            layer = by_rule[st["rule_id"]]
            keys = [*layer.served_by, *(c.name for c in layer.candidates)]
            for stratum in sorted({stratum_of[c] for c in st["_fail_ids"]}):
                ids = [c for c in st["_fail_ids"] if stratum_of[c] == stratum]
                measured = sum(1 for cid in ids if any(
                    (ev := evidence.get(k)) is not None and ev.refused() is None
                    and (o := ev.for_case(cid)) is not None and o.status == OK for k in keys))
                need = min(min_cover, len(ids))
                gap = 1.0 if measured == 0 else 0.5 if measured < need else 0.0
                units[f"{layer.rule_id}:{stratum}"] = {"rule_id": layer.rule_id, "stratum": stratum, "need": need,
                                                       "measured": measured, "gap": gap,
                                                       "layer_state": st["state"], "_ids": ids}
    return units


def coverage_gain(rule_id: str, new_ids: set[str], units: dict[str, dict[str, Any]]) -> tuple[float, list[str]]:
    """Gap removed if the action measures *new_ids*, over all units; and the units it touches."""
    if not units:
        return 0.0, []
    removed, touched = 0.0, []
    for uid, u in units.items():
        if u["rule_id"] != rule_id:
            continue
        added = sum(1 for cid in u["_ids"] if cid in new_ids)
        if not added:
            continue
        after = u["measured"] + added
        gap_after = 1.0 if after == 0 else 0.5 if after < u["need"] else 0.0
        if u["gap"] > gap_after:
            removed += u["gap"] - gap_after
            touched.append(uid)
    return removed / len(units), touched


def public_units(units: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {uid: {k: v for k, v in u.items() if not k.startswith("_")} for uid, u in units.items()}


# -- choosing ------------------------------------------------------------------------------


def value(s: Scored, weights: Weights, axis: str) -> Scored:
    s.gain = s.q * max(0.0, weights.d * s.D + weights.t * s.T + weights.c * s.C - weights.r * s.R)
    s.U = s.gain / ((s.cost if s.cost is not None else 0.0) + EPSILON[axis])
    return s


def choose(candidates: list[Scored], *, decision_no: int, seed: int, explore_every: int,
           min_gain: float) -> "tuple[Scored | None, str]":
    """The candidate to run, and why: ``best`` / ``explore`` / ``below_threshold`` / ``none``."""
    if not candidates:
        return None, "none"
    ranked = sorted(candidates, key=lambda s: (-s.U, s.order))
    if explore_every and (decision_no + 1) % explore_every == 0 and len(ranked) > 1:
        rng = random.Random(f"{seed}:{decision_no}")
        return rng.choice(ranked[1:]), "explore"
    best = ranked[0]
    if best.gain < min_gain:
        return None, "below_threshold"
    return best, "best"
