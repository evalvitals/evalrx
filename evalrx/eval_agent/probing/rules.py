"""MVP-2 rules: what each route runs next, and when the sample is expanded.

Fixed rules, no scores (scoring candidates against each other is MVP-3). They
write down the cold-start prior of the design's method table, so a different
choice is a new rule version rather than an unrecorded preference.

**Layers.** Each route has ordered layers (the soft layer order). A layer is
one question; its candidates are the default representative, then its
fallbacks. A layer is *done* once a candidate, or a battery step that already
serves it, gave a valid observation (``ok`` and not refused by its own
self-check) on the route's FAIL cases. It is *exhausted* once every candidate
is gated off (capability, media, task kind), failed or refused. A failed or
refused candidate moves on to the next fallback of the same layer; a valid
negative moves on to the next layer. No rule keeps trying tools until one fires.

**Rounds.** A route is active with at least ``min_route_fail`` FAIL cases.
Each round, every active route proposes one candidate. H goes first, because
output problems come before mechanisms. The other routes follow by number of
FAIL cases, with ties in the order H, N, U, S, X. A layer that was done on a
smaller sample is first extended to the route's new cases, before the route
moves on to its next layer.

**Case sets.** A candidate runs on the route's FAIL cases plus as many PASS
cases of the same stratum, taken in the sampler's rank order. This is the
sample's own matching, and it nests the same way, so a grown route only adds
cases.

**Expansion.** At the start of a round, the nested sample doubles (up to
``max_fail`` FAIL cases) while a gap holds that more cases can close. Two gaps
count:

* a route whose FAIL-minus-PASS difference is positive but whose 95% interval
  still includes zero;
* a stratum whose S cohort, or its stable PASS controls, is smaller than
  ``min_cohort``. On POPE the S readings had the opposite sign at 16 cases.

Expansion comes before the routes' candidates, because it revises the routes
those candidates are drawn from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from evalrx.analyzers.perturbation.prompt_contrast import _DEFAULT_STRATEGIES as _PROMPT_STRATEGIES
from evalrx.eval_agent.probing.actions import PlanStep
from evalrx.eval_agent.probing.evidence import StepEvidence

RULES_VERSION = "mvp4-rules-2"
ROUTE_ORDER = ("H", "N", "U", "S", "X")
DESCRIBE_STRATEGIES = {"baseline": "{prompt}", "describe_first": _PROMPT_STRATEGIES["describe_first"]}

#: Kinds of evidence (``PlanStep.evidence``) that more than one tool of a layer gives:
#: once one of them has answered the layer, another of the same kind is redundant.
MEDIA_ATTENTION_CUT = "text-to-media attention cut, answer margin"
MEDIA_NOISE = "media embeddings noised and restored, answer margin"
ATTENTION_RESCALING = "attention to the media rescaled, answer margin"
TOKEN_PROBABILITIES = "output-token probabilities"
RESAMPLING = "re-sampled answers"
RELEVANCE_MAP = "relevance map of the answer margin over the media"
LAYER_READOUT = "answer position read out layer by layer"

#: Room for a description before the answer, asked per call (the task's own cap
#: is sized for the answer alone: 16 tokens on POPE).
DESCRIBE_OUTPUT_TOKENS = 256


def _describe(route: str, question: str) -> PlanStep:
    # one key for N3 and Sb: the routes share its observations (and its PASS controls)
    return PlanStep("prompt_contrast", {"strategies": DESCRIBE_STRATEGIES,
                                        "gen_kwargs": {"max_new_tokens": DESCRIBE_OUTPUT_TOKENS}}, route, question,
                    key="describe_first", calls_per_case={"generate": 2}, keep_artifacts=("outputs",),
                    caveat="the runtime's output cap must leave room for the description",
                    min_output_tokens=128)


@dataclass(frozen=True)
class Layer:
    """One question of a route, its candidates in fallback order, and where the prior comes from.

    ``served_by`` names battery steps that already answer it. ``case_filter``
    narrows the route's FAIL cases to the ones the question is about.
    """

    rule_id: str
    route: str
    question: str
    candidates: tuple[PlanStep, ...]
    served_by: tuple[str, ...] = ()
    case_filter: "Callable[[dict[str, Any]], bool] | None" = None
    filter_note: str = ""
    source: str = "probing_design_mix.md method table (cold-start prior)"


LAYERS: tuple[Layer, ...] = (
    Layer("H1", "H", "does a re-ask with a stricter answer request recover the answer the output lost", (
        PlanStep("answer_extraction_audit", {"reask": True}, "H1", "re-ask the cases whose answer was not read",
                 key="extraction_reask", calls_per_case={"generate": 1}),)),
    Layer("H2", "H", "continued past where it stopped, does the output reach an answer", (
        PlanStep("termination_audit", {"continue_non_clean": True}, "H2", "continue the outputs that did not end cleanly",
                 key="termination_continue", calls_per_case={"generate": 1}),)),
    Layer("N1", "N", "does the media path carry a measurable effect on the answer", (
        PlanStep("causal_trace", {"target": "prediction"}, "N1", "corrupt the media, restore one site at a time",
                 calls_per_case={"forward": 20}, evidence=MEDIA_NOISE),),
        served_by=("attention_knockout",)),
    Layer("N2", "N", "does reading the media with sharper or flatter attention change the answer margin", (
        PlanStep("attention_temperature", {"target": "prediction"}, "N2", "rescale the media attention temperature",
                 fidelity_check="edit", calls_per_case={"forward": 4}, evidence=ATTENTION_RESCALING),
        PlanStep("attention_sensitivity", {"target": "prediction"}, "N2", "scale attention to the media down and up",
                 fidelity_check="edit", calls_per_case={"forward": 4}, evidence=ATTENTION_RESCALING))),
    Layer("N3", "N", "asked to describe the media first, does the answer start to follow it", (
        _describe("N3", "describe the media, then answer"),)),
    Layer("U1", "U", "is the answer that moves close to a tie", (
        PlanStep("logprob_entropy", {}, "U1", "answer-token probabilities of the output",
                 calls_per_case={"logprobs": 1}, per_case=True, evidence=TOKEN_PROBABILITIES),
        PlanStep("self_consistency", {"n": 10}, "U1", "twice the samples: how often the answer moves",
                 key="self_consistency_n10", calls_per_case={"generate": 10}, per_case=True,
                 keep_artifacts=("samples",), evidence=RESAMPLING))),
    Layer("U2", "U", "with the options rotated, does the answer follow content or position", (
        PlanStep("format_sensitivity", {}, "U2", "rotate the options", calls_per_case={"generate": 3},
                 task_kinds=("multiple_choice_letter",)),)),
    Layer("Sb", "S", "does a stable error survive when the model first states what it sees", (
        _describe("Sb", "describe the media, then answer"),)),
    Layer("S1", "S", "which image regions the answer margin depends on (no reference regions: "
                     "concentration only, not 'mislocalised')", (
        PlanStep("gradcam", {"target": "prediction"}, "S1", "gradient x activation map of the answer margin",
                 calls_per_case={"backward": 1}, evidence=RELEVANCE_MAP),
        PlanStep("generic_attention", {"target": "prediction"}, "S1", "attention relevance of the answer margin",
                 fidelity_check="edit", calls_per_case={"backward": 1}, evidence=RELEVANCE_MAP))),
    Layer("S2", "S", "is the case's own question word readable off the media tokens", (), served_by=("embed_lens",)),
    Layer("S3", "S", "how early in the stack the answer was readable", (
        PlanStep("tuned_lens", {}, "S3", "fitted per-layer readout of the answer position (translators "
                 "fit once on a fixed set of EXPLORE cases)", calls_per_case={"forward": 1},
                 evidence=LAYER_READOUT),
        PlanStep("layer_contrast", {}, "S3", "early-vs-late layer contrast at the answer position",
                 calls_per_case={"forward": 1}, evidence=LAYER_READOUT)),
        served_by=("patchscope",)),
    Layer("X1", "X", "with the media removed instead of swapped, does the answer change "
                     "(a different intervention: it does not test N)", (
        PlanStep("modality_ablation", {"mode": "drop"}, "X1", "re-ask without the media",
                 key="media_drop", calls_per_case={"generate": 2}),),
        case_filter=lambda r: r["N"]["state"] == "untested",
        filter_note="X cases whose N check is missing (a case without media has N not applicable)"),
)


def candidate_steps(layers: "tuple[Layer, ...]" = LAYERS) -> list[PlanStep]:
    return [c for layer in layers for c in layer.candidates]


# -- case sets -----------------------------------------------------------------------


def route_fail_ids(records: list[dict[str, Any]], route: str,
                   case_filter: "Callable[[dict[str, Any]], bool] | None" = None) -> list[str]:
    return [r["case_id"] for r in records if r["label"] == "fail" and route in r["routes"]
            and (case_filter is None or case_filter(r))]


def matched_controls(records: list[dict[str, Any]], fail_ids: list[str],
                     rank: Callable[[str], Any]) -> tuple[list[str], dict[str, int]]:
    """As many PASS cases per stratum as *fail_ids* has there, by sampler rank; and any shortfall."""
    by_id = {r["case_id"]: r for r in records}
    need: dict[str, int] = {}
    for cid in fail_ids:
        need[by_id[cid]["stratum"]] = need.get(by_id[cid]["stratum"], 0) + 1
    controls, shortfall = [], {}
    for stratum, k in sorted(need.items()):
        pool = sorted((r["case_id"] for r in records if r["label"] == "pass" and r["stratum"] == stratum), key=rank)
        controls += pool[:k]
        if len(pool) < k:
            shortfall[stratum] = k - len(pool)
    return controls, shortfall


# -- routes ------------------------------------------------------------------------------


def active_routes(records: list[dict[str, Any]], min_route_fail: int) -> list[dict[str, Any]]:
    """Every route with FAIL cases, ``active`` when it has enough, in round order."""
    rows = []
    for route in ROUTE_ORDER:
        n = len(route_fail_ids(records, route))
        if n:
            rows.append({"route": route, "n_fail": n, "active": n >= min_route_fail})
    return sorted(rows, key=lambda r: (r["route"] != "H", -r["n_fail"], ROUTE_ORDER.index(r["route"])))


@dataclass
class Proposal:
    layer: Layer
    step: PlanStep
    kind: str                 # "open" (a new layer or fallback) | "extend" (a done layer, new cases)
    fail_ids: list[str]
    control_ids: list[str]
    todo: list[str]
    tried: list[dict[str, Any]]
    before: list[dict[str, Any]]   # the earlier layers' states

    target_override: "list[str] | None" = None   # a battery step runs on the whole sample

    @property
    def target_ids(self) -> list[str]:
        return list(self.target_override) if self.target_override is not None else [*self.fail_ids, *self.control_ids]


Gate = Callable[[PlanStep, list[str]], "tuple[str, str] | None"]


def layer_state(layer: Layer, evidence: dict[str, StepEvidence], fail_ids: list[str], target_ids: list[str],
                gate: Gate, blocked: "dict[str, str] | None" = None,
                pending: "dict[str, tuple[PlanStep, list[str]]] | None" = None) -> dict[str, Any]:
    """``done`` / ``extend`` / ``open`` / ``exhausted`` for *layer* on these cases, with what was passed over.

    *pending* names battery steps that serve a layer but have not run on the
    current sample yet (with that sample): the layer is open on them first,
    unless the model cannot run them (the battery records that when it settles).
    """
    blocked = blocked or {}
    if not fail_ids:
        return {"rule_id": layer.rule_id, "state": "no_cases"}
    for name in layer.served_by:
        if name in (pending or {}) and gate(pending[name][0], pending[name][1]) is None:
            step, sample_ids = pending[name]
            ev = evidence.get(name)
            return {"rule_id": layer.rule_id, "state": "open", "step": step, "battery": True,
                    "stale": bool(ev is not None and ev.chunks), "target": list(sample_ids),
                    "todo": ev.todo(sample_ids) if ev is not None else list(sample_ids), "tried": []}
    for name in layer.served_by:
        ev = evidence.get(name)
        if ev is not None and ev.valid_on(fail_ids):
            return {"rule_id": layer.rule_id, "state": "done", "via": name, "battery": True}
    tried: list[dict[str, Any]] = []
    for name in layer.served_by:
        ev = evidence.get(name)
        obs = next((o for cid in fail_ids if ev is not None and (o := ev.for_case(cid)) is not None), None)
        tried.append({"key": name, "status": ",".join(ev.statuses_on(fail_ids)) if obs is not None else "not run",
                      "battery": True, **({"reason": obs.error} if obs is not None and obs.error else {})})
    for step in layer.candidates:
        ev = evidence.get(step.name)
        if ev is not None and ev.valid_on(fail_ids):
            todo = ev.todo(target_ids)
            if todo and step.name not in blocked:
                return {"rule_id": layer.rule_id, "state": "extend", "via": step.name, "step": step,
                        "todo": todo, "tried": tried}
            return {"rule_id": layer.rule_id, "state": "done", "via": step.name, "tried": tried,
                    **({"incomplete": blocked[step.name]} if todo else {})}
        if ev is not None and any(cid in ev.covered() for cid in fail_ids):
            tried.append({"key": step.name, "status": ",".join(ev.statuses_on(fail_ids))})
            continue
        if step.name in blocked:
            tried.append({"key": step.name, "status": "blocked", "reason": blocked[step.name]})
            continue
        why = gate(step, target_ids)
        if why is not None:
            tried.append({"key": step.name, "status": why[0], "reason": why[1]})
            continue
        return {"rule_id": layer.rule_id, "state": "open", "step": step,
                "todo": ev.todo(target_ids) if ev is not None else list(target_ids), "tried": tried}
    return {"rule_id": layer.rule_id, "state": "exhausted", "tried": tried}


def route_states(route: str, records: list[dict[str, Any]], evidence: dict[str, StepEvidence], gate: Gate,
                 rank: Callable[[str], Any], *, layers: "tuple[Layer, ...]" = LAYERS,
                 blocked: "dict[str, str] | None" = None,
                 pending: "dict[str, tuple[PlanStep, list[str]]] | None" = None) -> list[dict[str, Any]]:
    """Every layer of *route*, in order, with its state on the route's current cases."""
    out = []
    for layer in (x for x in layers if x.route == route):
        fail_ids = route_fail_ids(records, route, layer.case_filter)
        controls, shortfall = matched_controls(records, fail_ids, rank)
        state = layer_state(layer, evidence, fail_ids, [*fail_ids, *controls], gate, blocked, pending)
        out.append({**state, "question": layer.question, "n_fail": len(fail_ids), "n_controls": len(controls),
                    "control_shortfall": shortfall, "_fail_ids": fail_ids, "_control_ids": controls})
    return out


def next_for_route(route: str, records: list[dict[str, Any]], evidence: dict[str, StepEvidence], gate: Gate,
                   rank: Callable[[str], Any], *, layers: "tuple[Layer, ...]" = LAYERS,
                   blocked: "dict[str, str] | None" = None) -> "Proposal | None":
    """The route's first layer that is open or needs extending, as a proposal; ``None`` when none is."""
    by_id = {x.rule_id: x for x in layers}
    states = route_states(route, records, evidence, gate, rank, layers=layers, blocked=blocked)
    for i, st in enumerate(states):
        if st["state"] in ("open", "extend"):
            # what this choice passed over: the exhausted layers before it, and this layer's own
            tried = [{**t, "rule_id": s["rule_id"]} for s in [*states[:i], st]
                     if s is st or s["state"] == "exhausted" for t in s.get("tried", [])]
            return Proposal(layer=by_id[st["rule_id"]], step=st["step"], kind=st["state"],
                            fail_ids=st["_fail_ids"], control_ids=st["_control_ids"], todo=st["todo"],
                            tried=tried, before=[public(s) for s in states[:i]])
    return None


def public(state: dict[str, Any]) -> dict[str, Any]:
    """A layer state without its case lists and step object, for the trace and the report."""
    out = {k: v for k, v in state.items() if not k.startswith("_") and k not in ("step", "target")}
    if "todo" in out:
        out["todo"] = len(out["todo"])
    if "step" in state:
        out["key"] = state["step"].name
    return out


# -- expansion -------------------------------------------------------------------------


def expansion_gaps(summary: dict[str, Any], records: list[dict[str, Any]], *, min_cohort: int) -> list[dict[str, Any]]:
    """The evidence gaps a larger nested sample could close (see the module docstring)."""
    gaps = []
    for tag in ("H_label_free", "N", "U"):
        row = summary.get(tag) or {}
        ci = row.get("signal_ci95")
        if row.get("routing_signal") and ci is not None and ci[0] <= 0.0:
            gaps.append({"gap": "signal_uncertain", "route": tag, "routing_signal": row["routing_signal"],
                         "ci95": ci, "n_tested": [row["fail"]["n_tested"], row["pass"]["n_tested"]]})
    for stratum in sorted({r["stratum"] for r in records if r["label"] == "fail" and r["stable"]}):
        n_s = sum(1 for r in records if r["label"] == "fail" and r["stable"] and r["stratum"] == stratum)
        n_c = sum(1 for r in records if r["label"] == "pass" and r["stable"] and r["stratum"] == stratum)
        if min(n_s, n_c) < min_cohort:
            gaps.append({"gap": "s_cohort_small", "stratum": stratum, "n_S": n_s, "n_stable_pass": n_c,
                         "min_cohort": min_cohort})
    return gaps


def next_size(n_fail: int, max_fail: int, n_available: int) -> "int | None":
    """The doubled FAIL count, capped; ``None`` when the sample cannot grow."""
    n = min(2 * n_fail, max_fail, n_available)
    return n if n > n_fail else None
