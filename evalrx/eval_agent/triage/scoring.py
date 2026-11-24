"""The value of a candidate action and the scored scheduler's choice.

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
* **c** (marginal cost): with a :class:`CostTable` (``--triage-cost-table``),
  the table's price: the step's fixed seconds per case times the new cases,
  plus its one-time cost (a fit, a calibration) while the step has not run in
  this run; a step the table does not list is priced at the table's median
  price per case (exploration's share and budget checks use its maximum). What
  the evidence and exploration have spent is counted in the same prices, so a
  decision reads no measured time: two runs with the same table, cases and
  model outputs decide alike whatever the machine's load. Without a table, the
  step's own measured cost per case in this run, times the new cases; a step
  that has not run yet is priced at the run's median cost per case (budget
  checks use the maximum), and its first chunk shows its real cost. Measured
  costs enter the value and exploration's share, so such a run's decisions
  depend on its timings: a repeat may order its follow-ups differently. A
  table is data the host supplies for a model and a backend (costs do not
  transfer across input sizes, models or backends); the package holds no
  per-analyzer prior of its own. A budget is a real limit: it is always checked
  against the measured spending, with the table's price as the estimate.

Candidates are each route's open or stale layers (the layer's next candidate in
its fallback order, as in the rules), the alternatives of its answered layers,
the expansion, and the batch-dependent battery steps not yet run on the
current sample. No batch-dependent candidate (battery step or follow-up) is
taken while an expansion can still follow: any expansion supersedes its
reading, and no route reads it (the rules' deferral, for the same reason); its
layer waits instead of falling back. Unlike the rules, the scored scheduler may
take a later layer first, and says so when it does.

**Exploration** has a share of the evidence spending (the judge's calls are not
an opportunity to explore; with a cost table the spending is counted at the
table's prices), not of the decisions (a decision may be one chunk). It draws from its own targets, the actions the rules and the value
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
and no target fits (the minimum exploration is met).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalrx.eval_agent.triage.actions import OK, UNSUPPORTED, PlanStep
from evalrx.eval_agent.triage.evidence import StepEvidence

SCORING_VERSION = "score-6"


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
    basis: str = ""                    # what the cost rests on: "own" / "typical" / "max" / "none", or the
                                       # table's "table" / "table+fixed" / "table_typical" / "table_max"
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


def _costed(obs: Any) -> bool:
    """An observation that cost something to make: it called the model, or completed without
    calling it (an audit of stored outputs). A gate result (``unsupported``) is neither."""
    return bool(((obs.cost or {}).get("calls") or obs.status == OK) and obs.planned_ids)


@dataclass(frozen=True)
class CostTable:
    """Fixed prices for the scheduler's decisions, on one cost axis.

    ``steps`` maps a step key to ``(per_case, fixed)``: the price of one case, and a one-time
    price the step pays with its first action of a run (a tuned-lens fit; 0 for most steps).
    The prices are whatever the host measured once for this model and backend and then froze:
    a decision made from them does not depend on how long anything took in this run.
    """

    steps: dict[str, tuple[float, float]]
    axis: str = "wall_sec"
    source: Any = None                 # provenance, recorded as given
    sha1: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostTable":
        """``{"axis": "wall_sec", "steps": {key: {"per_case": s, "fixed": s}}, "source": ...}``;
        ``fixed`` may be left out (0)."""
        axis = data.get("axis", "wall_sec")
        if axis not in EPSILON:
            raise ValueError(f"cost table axis must be one of {sorted(EPSILON)}; got {axis!r}")
        rows = data.get("steps")
        if not isinstance(rows, dict) or not rows:
            raise ValueError("a cost table needs a non-empty 'steps' mapping")
        steps: dict[str, tuple[float, float]] = {}
        for key, row in rows.items():
            if not isinstance(row, dict) or "per_case" not in row:
                raise ValueError(f"cost table step {key!r} needs a 'per_case' price")
            per_case, fixed = float(row["per_case"]), float(row.get("fixed") or 0.0)
            if not (per_case >= 0 and fixed >= 0):   # also refuses NaN
                raise ValueError(f"cost table step {key!r}: prices must be non-negative numbers")
            steps[str(key)] = (per_case, fixed)
        body = json.dumps({"axis": axis, "steps": {k: list(v) for k, v in sorted(steps.items())}}, sort_keys=True)
        return cls(steps=steps, axis=axis, source=data.get("source"),
                   sha1=hashlib.sha1(body.encode()).hexdigest()[:16])

    @classmethod
    def load(cls, path: "str | Path") -> "CostTable":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def unlisted(self, unmeasured: str) -> float:
        """The price per case of a step the table does not list: the median of its model-calling
        steps' prices, or their maximum (``unmeasured="max"``), so an unknown tool cannot look cheap."""
        pool = sorted(per_case for per_case, _ in self.steps.values() if per_case > 0) or [0.0]
        return pool[-1] if unmeasured == "max" else (pool[(len(pool) - 1) // 2] + pool[len(pool) // 2]) / 2

    def price(self, key: str, n_cases: int, *, first: bool, unmeasured: str = "typical") -> tuple[float, str]:
        """The price of *key* on *n_cases* cases and its basis; *first* adds the one-time price."""
        if key not in self.steps:
            return self.unlisted(unmeasured) * n_cases, f"table_{unmeasured}"
        per_case, fixed = self.steps[key]
        if first and fixed:
            return fixed + per_case * n_cases, "table+fixed"
        return per_case * n_cases, "table"

    def identity(self) -> dict[str, Any]:
        return {"sha1": self.sha1, "axis": self.axis, "n_steps": len(self.steps), "source": self.source}


def priced(evidence: dict[str, StepEvidence], table: CostTable) -> float:
    """What the run's evidence costs at the table's prices: every distinct observation that cost
    something, at its planned cases, and each step's one-time price once. The scheduler's
    measure of spending when it decides from a table (a read-back counts once, as in ``q``)."""
    total = 0.0
    for key, ev in evidence.items():
        ids = {o.action_id: len(o.planned_ids) for o in ev.chunks if _costed(o)}
        if ids:
            total += table.price(key, sum(ids.values()), first=True)[0]
    return total


def step_cost(step: PlanStep, n_cases: int, evidence: dict[str, StepEvidence], axis: str, *,
              unmeasured: str = "typical", table: "CostTable | None" = None) -> "tuple[float | None, str]":
    """Cost of running *step* on *n_cases* new cases, and its basis.

    With a *table* on this axis, its price (``table``; ``table+fixed`` while the step has not
    run in this run and has a one-time price; ``table_typical`` / ``table_max`` for a step it
    does not list): nothing measured is read.

    Without one there is no analyzer-specific prior: a step that has run in this run is priced
    at its own measured cost per case (``own``) over every observation that called the
    model or completed without calling it (an audit of stored outputs costs what
    it measured: next to nothing). One that has not run is priced at the run's
    ``typical`` (median) cost per case of the model-calling observations when
    ranking, or their ``max`` when checking a budget, so an unknown tool cannot
    look cheap; ``None`` before anything was measured.
    """
    ev = evidence.get(step.name)
    if table is not None and table.axis == axis:
        return table.price(step.name, n_cases, first=not any(_costed(o) for o in (ev.chunks if ev else [])),
                           unmeasured=unmeasured)
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


def choose(candidates: list[Scored], *, decision_no: int, seed: int, explore_share: float, explored: float,
           spent: float, min_gain: float, targets: "list[Scored]") -> "tuple[Scored | None, str]":
    """The candidate to run, and why: ``best`` / ``explore`` / ``below_threshold`` / ``none``.

    *explored* and *spent* are what exploration and the run's evidence have cost so far, on the
    candidates' cost axis. Exploration draws from *targets*, the candidates only it takes: a
    seeded draw among those that keep exploration within *explore_share* of the spending runs
    instead of the best."""
    ranked = sorted(candidates, key=lambda s: (-s.U, s.order))
    best = ranked[0] if ranked else None
    others = sorted(targets, key=lambda s: s.order)
    fits = [s for s in others if explored + (s.cost or 0.0) <= explore_share * (spent + (s.cost or 0.0))]
    if explore_share > 0 and fits:
        return random.Random(f"{seed}:{decision_no}").choice(fits), "explore"
    if best is None:
        return None, "none"
    if best.gain < min_gain:
        return None, "below_threshold"
    return best, "best"
