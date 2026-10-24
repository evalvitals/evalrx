"""MVP-3: the value of a candidate action and the scored scheduler's choice.

    U(a | E) = q(a) · [w_D·D(a) + w_T·T(a) + w_C·C(a) − w_R·R(a)]₊ / (c(a) + ε)

All gains are in [0, 1], ``c`` is on the run's primary budget axis (wall
seconds, or USD when every executed action has a price), and ``ε`` is that
axis' metering floor. The definitions are the design's first executable ones,
written down so a run can show why each action won:

* **D** (prediction disagreement): with an explainer, the pairs of the route's
  live competing explanations whose predictions for the step conflict, each
  counted by how decidable the conflict is on the route's cases, over all pairs
  (:mod:`.explanations`); the explanations come once the sample has settled,
  and an explanation a prediction already missed on leaves the live set at the
  next decision. Without one, or with fewer than two live explanations,
  ``D = 0`` and actions earn budget through T and C. An expansion's D is 0.
* **T** (evidence gap): 1 completes a recorded gap — a ``reuse`` step measured
  on part of the route's FAIL cases (a smaller sample, or earlier chunks) that
  still has to run on the rest, or a contradiction between two readings of the
  same check; 0.5 is an expansion that may close a recorded uncertainty (an
  uncertain routing signal, a small S cohort); 0 otherwise. A batch-dependent
  step that went stale re-runs on the whole sample: its value is the coverage
  it restores (C), not a gap of its own. Every T names the gap ids it serves.

A ``reuse`` step runs one chunk per decision (a fixed number of cases, FAIL and
PASS alternating), so a decision spends at most one chunk before the scheduler
sees the step's measured cost. T, D and R are about the step on the whole
route, so a chunk earns them in proportion to the share of the route's
remaining FAIL cases it covers, and the chunks of a step add up to what it
earns unchunked; C is counted per chunk already.
* **C** (coverage): units are ``layer × gold stratum`` for every route with FAIL
  cases there. A unit's gap is 1 untested, 0.5 measured on fewer than
  ``min(min_cover, route FAIL cases in the stratum)`` cases, else 0. ``C(a)`` is
  the gap the action is expected to remove, over the number of units.
* **R** (redundancy): an observation that already exists is filtered out (R = 1).
  A layer that is already answered offers its other candidates as
  *alternatives*: 0.5 when the alternative gives the kind of evidence the tool
  that answered the layer gave (the same question asked the same way, e.g. a
  second attention rescaling), 0 when it gives another kind (a different
  intervention). An alternative closes no gap and covers no unit, so it is
  only worth its D: a same-kind alternative runs when it tells live
  explanations apart by more than its redundancy.
* **q** (execution availability): ``(successes + 1) / (attempts + 2)`` over the
  distinct observations of the analyzer in this run's evidence (a read-back
  counts once, so a resumed run decides the same way); ``ok`` and not refused
  by its own check is a success, and ``unsupported`` (a gate result, filtered
  before scoring) is not an attempt.
* **c** (marginal cost): the step's own measured cost per case in this run,
  times the new cases; a step that has not run yet is priced at the run's
  median cost per case (budget checks use the maximum). There is no
  per-analyzer prior: measured costs do not transfer across input sizes,
  models or backends, and a reusable step's first chunk shows its real cost.

Candidates are each route's open or stale layers (the layer's next candidate in
its fallback order, as in the rules), the alternatives of its answered layers,
the expansion, and the batch-dependent battery steps not yet run on the
current sample. No batch-dependent candidate (battery step or follow-up) is
taken while an expansion can still follow: any expansion supersedes its
reading, and no route reads it (the rules' deferral, for the same reason); its
layer waits instead of falling back. Unlike the rules, the scored scheduler may
take a later layer first, and says so when it does.

**Exploration** has a share of the evidence spending (the judge's calls are not
an opportunity to explore), not of the decisions (a decision may be one
chunk). It draws from its own targets, the actions the rules and the value
never take: the next candidate of each route with FAIL cases too few to be
active (a low-frequency problem), and on each active route the representative
of every other route's layer (outside the route's method family). A target fits
when what exploration has spent, plus its conservative estimate (a budget
check's), stays within ``explore_share`` of the evidence spent; whenever one
fits, a seeded draw among them runs instead of the top candidate. Completing a
target on its route is exploration too, until the cases measured so far show a
reading outside its band: then the step has found something, and its
completion is an incomplete step that T values like any other. A gain below
``min_gain`` is not worth an action: the run stops once no candidate reaches it
and no target fits (the minimum exploration is met). ``explore_from="candidates"``
(an ablation) draws from the candidates other than the top one instead,
alternatives included.

**Controls.** Two schedulers take the scored scheduler's candidates and stop
rules and differ only in the pick, so a comparison with them isolates the
value ordering: ``random`` draws uniformly (seeded) among the candidates at or
above the gain floor, ``cheapest`` takes the lowest-cost one (an unpriced
candidate last, ties by soft order). Neither explores or reads explanations
(D is 0): their nearest value arm is the scored scheduler without exploration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from evalrx.eval_agent.probing.actions import OK, UNSUPPORTED, PlanStep
from evalrx.eval_agent.probing.evidence import StepEvidence

SCORING_VERSION = "mvp3-score-5"


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
    basis: str = ""                    # what the cost rests on: "own" / "typical" / "none"
    alternative_to: str = ""           # the step that already answered the layer (an alternative)
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


def per_case_costs(evidence: dict[str, StepEvidence], axis: str) -> list[float]:
    """Cost per planned case of every distinct observation that called the model."""
    seen: dict[str, float] = {}
    for ev in evidence.values():
        for obs in ev.chunks:
            cost = obs.cost or {}
            if obs.action_id in seen or not cost.get("calls") or not obs.planned_ids or cost.get(axis) is None:
                continue
            seen[obs.action_id] = _amount(cost, axis) / len(obs.planned_ids)
    return list(seen.values())


def step_cost(step: PlanStep, n_cases: int, evidence: dict[str, StepEvidence], axis: str, *,
              unmeasured: str = "typical") -> "tuple[float | None, str]":
    """Cost of running *step* on *n_cases* new cases, and its basis.

    No analyzer-specific prior: a step that has run in this run is priced at its
    own measured cost per case (``own``) over every observation that called the
    model or completed without calling it (an audit of stored outputs costs what
    it measured: next to nothing). One that has not run is priced at the run's
    ``typical`` (median) cost per case of the model-calling observations when
    ranking, or their ``max`` when checking a budget, so an unknown tool cannot
    look cheap; ``None`` before anything was measured.
    """
    ev = evidence.get(step.name)
    rows = [(_amount(o.cost, axis), len(o.planned_ids)) for o in (ev.chunks if ev else [])
            if ((o.cost or {}).get("calls") or o.status == OK) and o.planned_ids
            and (o.cost or {}).get(axis) is not None]
    if rows and sum(k for _, k in rows):
        return sum(s for s, _ in rows) / sum(k for _, k in rows) * n_cases, "own"
    pool = per_case_costs(evidence, axis)
    if not pool:
        return None, "none"
    pool.sort()
    per_case = pool[-1] if unmeasured == "max" else (pool[(len(pool) - 1) // 2] + pool[len(pool) // 2]) / 2
    return per_case * n_cases, unmeasured


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


#: How a decision picks among the scored candidates: the value (the scored scheduler) or a control.
POLICIES = ("value", "random", "cheapest")


def choose(candidates: list[Scored], *, decision_no: int, seed: int, explore_share: float, explored: float,
           spent: float, min_gain: float, targets: "list[Scored] | None" = None,
           policy: str = "value") -> "tuple[Scored | None, str]":
    """The candidate to run, and why: ``best`` / ``explore`` / ``random`` / ``cheapest`` /
    ``below_threshold`` / ``none``.

    *explored* and *spent* are what exploration and the run's evidence have cost so far, on the
    candidates' cost axis. Exploration draws from *targets*, the candidates only it takes; with
    ``targets=None`` it draws from the other candidates instead (below the gain floor, from all
    of them). A seeded draw among those that keep exploration within *explore_share* of the
    spending runs instead of the best.

    *policy* ``"value"`` is the scored scheduler. The controls take the candidates at or above the
    gain floor and never explore: ``"random"`` a seeded uniform draw among them, ``"cheapest"``
    the lowest-cost one (an unpriced candidate last, ties by soft order)."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}; got {policy!r}")
    if policy != "value":
        eligible = [s for s in candidates if s.gain >= min_gain]
        if not eligible:
            return None, "below_threshold" if candidates else "none"
        if policy == "random":
            return random.Random(f"{seed}:{decision_no}").choice(eligible), "random"
        return min(eligible, key=lambda s: (s.cost if s.cost is not None else math.inf, s.order)), "cheapest"
    ranked = sorted(candidates, key=lambda s: (-s.U, s.order))
    best = ranked[0] if ranked else None
    if targets is None:
        others = ranked[1:] if best is not None and best.gain >= min_gain else ranked
    else:
        others = sorted(targets, key=lambda s: s.order)
    fits = [s for s in others if explored + (s.cost or 0.0) <= explore_share * (spent + (s.cost or 0.0))]
    if explore_share > 0 and fits:
        return random.Random(f"{seed}:{decision_no}").choice(fits), "explore"
    if best is None:
        return None, "none"
    if best.gain < min_gain:
        return None, "below_threshold"
    return best, "best"
