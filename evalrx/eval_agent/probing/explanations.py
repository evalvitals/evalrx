"""MVP-3b: competing explanations, their predictions per tool, and the D they give a candidate.

The chain's judge (the model ``runner.build_judge`` builds for M1-M5) is asked,
for every active route, for up to ``max_per_route`` local explanations of why
that route's FAIL cases fail. It is asked once the sample has settled (an
expansion revises the routes, as it supersedes batch-dependent readings), and
again, up to ``max_asks`` times in all, when the evidence has decided a
prediction and a route with tools left has fewer than two live explanations;
each ask sees the previous explanations' verdicts. For each tool still to run
on the route, every explanation predicts the direction of the tool's readings
on the route's FAIL cases against their PASS controls:

* ``up``: FAIL mean - control mean > the reading's band;
* ``down``: < -band;
* ``same``: within +-band;
* ``unknown``: the explanation does not say.

The readings, their meaning, sign and band are fixed here, per step, in
``CARDS``: the judge picks a direction, never the metric, the comparison or
the threshold. Code, not the judge, checks the predictions against the
evidence (``verdict``): a prediction is a hit when the 95% interval of the
difference lies inside its region, a miss when it lies outside, and
undetermined otherwise. An explanation with a miss is contradicted and leaves
the live set; one without any explicit prediction is untestable and is not in
it either.

``D(a)`` for a candidate on route r sums, over the pairs of r's live
explanations whose explicit predictions for that step conflict, how decidable
the conflict is on the route's cases, and divides by the number of pairs; a
pair with an ``unknown`` counts in the denominator only. Zero or one live
explanation gives ``D = 0``. Confidence the judge states is not asked for and
never weights anything. D is a disagreement heuristic, not a posterior or an
information gain.

**Decidability** (``decidability``): a conflict is only worth something if the
route's sample can settle it. Take one prediction as true at its reference
difference: 0 for ``same``, one standard deviation of the reading beyond the
band for ``up``/``down``. The cases needed are the fewest per side at which the
verdict's interval would then contradict the other prediction. With *n* route
FAIL cases (and as many controls), each side scores ``min(1, n / needed)``,
and the conflict scores the mean of the two sides; a pair conflicting on
several readings takes its most decidable one. The spread comes from every
case the step measured in this run. Until then a share's spread is taken at
its largest (0.5) and a mean's band as negligible next to its spread: the
least favourable values, so that a "same" prediction about an unmeasured mean
reading counts as unsettleable.

The judge sees the route summaries and readings of EXPLORE only, never a
case's content, media or CONFIRM. Each call's prompt, reply and cost are
stored under ``explanations/<key>.json``, keyed by the prompt and the judge's
identity, so a resumed run reads the same reply back instead of asking again
(a call that raised is not stored: it is asked again).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from evalrx.eval_agent.probing.metering import usage_counters, usage_delta
from evalrx.eval_agent.probing.triage import difference_interval

EXPLAIN_VERSION = "mvp3-explain-2"
DIRECTIONS = ("up", "down", "same", "unknown")
_ALIASES = {"↑": "up", "↓": "down", "≈": "same", "=": "same", "~": "same", "?": "unknown",
            "higher": "up", "lower": "down", "increase": "up", "decrease": "down", "none": "unknown"}
MIN_N = 2   # cases per side before an interval is computed


@dataclass(frozen=True)
class Metric:
    """One reading a prediction may be about: the step's per-case field, compared FAIL - control."""

    field: str
    meaning: str
    band: float
    kind: str = "mean"   # "mean" (Welch interval) | "rate" (0/1 per case, Newcombe interval)

    def card(self) -> dict[str, Any]:
        return {"field": self.field, "meaning": self.meaning, "band": self.band,
                "kind": "share of cases" if self.kind == "rate" else "mean"}


def _rate(field_: str, meaning: str, band: float = 0.1) -> Metric:
    return Metric(field_, meaning, band, "rate")


_MARGIN = "the gold first-token logit margin (gold vs its strongest competitor, fixed on the clean run)"

#: The readings each step's predictions may name, by step key (``PlanStep.name``).
#: Bands are fixed before any run: 0.1 for shares of cases, 0.5 for logit
#: margins, otherwise in the reading's own units. Steps without a card get no
#: predictions (D = 0): the H re-ask and continuation run on FAIL cases only,
#: so their readings have no PASS control to compare with.
CARDS: dict[str, tuple[Metric, ...]] = {
    # N: does the media path carry the gold answer
    "attention_knockout": (
        Metric("ko_delta_full", f"{_MARGIN}, clean minus with text-to-media attention cut in every layer; "
                                ">0 = the media reading supported gold", 0.5),
        Metric("ko_delta_late", "the same with the cut in the last third of layers only", 0.5)),
    "causal_trace": (
        Metric("ct_media_effect", f"{_MARGIN}, clean minus with the media embeddings noised; >0 = the media "
                                  "supported gold", 0.5),
        _rate("ct_effect_traceable", "the noising moved the margin by at least 1 logit")),
    "attention_temperature": (
        Metric("tt_slope", "change of the gold margin per log-unit of attention temperature over the media "
                           "(flat minus sharp over ln 4); <0 = sharper focus on the media helps gold", 0.25),
        Metric("tt_delta_sharp", f"{_MARGIN} with media attention sharpened (temperature 0.5) minus clean", 0.5),
        Metric("tt_media_entropy", "normalised entropy of text-to-media attention (1 = spread evenly)", 0.05)),
    "attention_sensitivity": (
        Metric("rw_slope", "change of the gold margin per log-unit of media attention scaling (x2 vs x0.5); "
                           ">0 = more attention to the media helps gold", 0.25),
        Metric("rw_delta_up", f"{_MARGIN} with media attention doubled minus clean", 0.5)),
    "describe_first": (
        _rate("changed_describe_first", "the answer changes when asked to describe the media first"),
        _rate("fixed_describe_first", "wrong before, right after describing first")),
    # U: is the answer near a tie, does it follow content
    "logprob_entropy": (
        Metric("mean_top_entropy", "entropy (nats) over the top alternatives per output token, mean; "
                                   "higher = less certain", 0.05),
        Metric("min_token_logprob", "lowest output-token log-probability; lower = a closer call", 0.1)),
    "self_consistency_n10": (
        Metric("consistency", "share of 10 samples giving the modal answer; lower = less stable", 0.05),),
    "format_sensitivity": (
        Metric("format_flip_rate", "share of option rotations whose chosen content differs from the modal "
                                   "content; higher = the answer does not follow content", 0.1),
        Metric("positional_bias", "share of rotations choosing the same letter; higher = position, not "
                                  "content", 0.1)),
    # S: where the stable error sits
    "prompt_threshold": (
        _rate("changed_sensitive", "the answer changes when told not to default to a negative answer"),
        _rate("fixed_sensitive", "wrong before, right after that instruction")),
    "gradcam": (
        Metric("gc_media_grad_share", "share of gradient x activation on the media tokens (at mid depth) of "
                                      "the gold margin; low = the margin hinges on the text", 0.05),
        Metric("gc_top5_share", "share of the media map's positive mass in its top 5% of positions; "
                                "higher = more concentrated", 0.05),
        Metric("gc_entropy", "normalised entropy of the media map (1 = diffuse)", 0.05)),
    "generic_attention": (
        Metric("rel_media_share", "share of propagated attention relevance of the gold margin on the media "
                                  "tokens; low = built from the text", 0.05),
        Metric("rel_top5_share", "share of the media relevance map in its top 5% of positions", 0.05),
        Metric("rel_entropy", "normalised entropy of the media relevance map (1 = diffuse)", 0.05)),
    "embed_lens": (
        Metric("el_query_sim", "cosine of the case's own question word to its nearest media token, late in "
                               "the stack; higher = the object is readable off the media", 0.02),
        Metric("el_frac_specific", "share of media tokens decoding to case-specific (not recurring, not "
                                   "high-norm) words", 0.05)),
    "patchscope": (
        Metric("ps_peak_prob", "highest probability of the gold answer read off the answer position at any "
                               "layer", 0.05),
        Metric("ps_late_drop", "that peak minus the output's gold probability; >0 = gold was more readable "
                               "mid-stack than at the output", 0.05),
        Metric("ps_media_query_share", "share of media-token readouts that are the case's question word", 0.05)),
    "tuned_lens": (
        Metric("tl_late_drop", "peak minus final probability of the model's own final answer across fitted "
                               "layer readouts (not gold)", 0.05),
        Metric("tl_decision_frac", "relative depth from which every readout agrees with the final answer; "
                                   "later = decided near the top", 0.05)),
    "layer_contrast": (
        Metric("contrast_margin", "log-prob of the final answer at the output minus at the most divergent "
                                  "earlier layer; <0 = stronger earlier, suppressed late", 0.25),
        Metric("layer_agreement_frac", "share of upper layers whose top-1 already is the final answer", 0.05)),
    "media_drop": (
        _rate("grounded_in_image", "the answer changes when the image is removed"),
        _rate("grounded_in_audio", "the answer changes when the audio is removed")),
}


# -- checking predictions --------------------------------------------------------------------

_T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
         2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
         2.045, 2.042)


def _t975(df: float) -> float:
    """Two-sided 95% Student-t quantile at ``floor(df)`` (conservative), normal beyond 120."""
    k = max(1, int(math.floor(df)))
    if k <= 30:
        return _T975[k - 1]
    return 2.021 if k < 60 else 2.000 if k < 120 else 1.960


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def difference(fail: list[float], control: list[float], kind: str) -> "tuple[float, list[float]] | None":
    """``(FAIL mean - control mean, 95% interval)``; ``None`` below ``MIN_N`` cases a side."""
    if len(fail) < MIN_N or len(control) < MIN_N:
        return None
    diff = _mean(fail) - _mean(control)
    if kind == "rate":
        ci = difference_interval(sum(1 for x in fail if x >= 0.5), len(fail),
                                 sum(1 for x in control if x >= 0.5), len(control))
        return diff, list(ci or [diff, diff])
    vf = sum((x - _mean(fail)) ** 2 for x in fail) / (len(fail) - 1) / len(fail)
    vc = sum((x - _mean(control)) ** 2 for x in control) / (len(control) - 1) / len(control)
    se = math.sqrt(vf + vc)
    if se == 0.0:
        return diff, [diff, diff]
    df = (vf + vc) ** 2 / ((vf ** 2 / (len(fail) - 1) if vf else 0.0) + (vc ** 2 / (len(control) - 1) if vc else 0.0))
    half = _t975(df) * se
    return diff, [diff - half, diff + half]


def observed(ci: list[float], band: float) -> str:
    """The direction the whole interval lies in, else ``undetermined``."""
    lo, hi = ci
    if lo > band:
        return "up"
    if hi < -band:
        return "down"
    if lo >= -band and hi <= band:
        return "same"
    return "undetermined"


def verdict(direction: str, ci: "list[float] | None", band: float) -> str:
    """``hit`` (the interval inside the predicted region), ``miss`` (outside it) or ``undetermined``."""
    if ci is None or direction not in ("up", "down", "same"):
        return "undetermined"
    lo, hi = ci
    if direction == "up":
        return "hit" if lo > band else "miss" if hi <= band else "undetermined"
    if direction == "down":
        return "hit" if hi < -band else "miss" if lo >= -band else "undetermined"
    return "hit" if -band <= lo and hi <= band else "miss" if lo > band or hi < -band else "undetermined"


def outcome(values: "dict[str, dict[str, Any]]", fail_ids: list[str], control_ids: list[str],
            metric: Metric) -> dict[str, Any]:
    """The reading's FAIL-vs-control difference on these cases, from per-case ``follow_up_values``."""
    def take(ids: list[str]) -> list[float]:
        return [float(values[c]["values"][metric.field]) for c in ids
                if values.get(c, {}).get("state") == "measured" and metric.field in values[c]["values"]]

    f, c = take(fail_ids), take(control_ids)
    out: dict[str, Any] = {"n": [len(f), len(c)]}
    if f:
        out["fail_mean"] = round(_mean(f), 4)
    if c:
        out["control_mean"] = round(_mean(c), 4)
    res = difference(f, c, metric.kind)
    if res is None:
        out["observed"] = "not_measured" if not f and not c else "undetermined"
        return out
    diff, ci = res
    out.update(diff=round(diff, 4), ci95=[round(ci[0], 4), round(ci[1], 4)], observed=observed(ci, metric.band))
    return out


# -- decidability -------------------------------------------------------------------------------

_MAX_CASES = 1 << 20


def half_width(n: int) -> float:
    """The verdict interval's half-width in standard deviations of the reading, with *n* cases on
    each side of equal spread (Welch's degrees of freedom are then ``2n - 2``)."""
    return _t975(2 * n - 2) * math.sqrt(2.0 / n)


def spread(groups: "list[list[float]]", kind: str) -> "float | None":
    """The reading's standard deviation within groups (FAIL, PASS), pooled; ``None`` without data.

    A share's spread is that of its smoothed rate ``(k + 2) / (m + 4)``, so a share
    measured at 0 still has one (the interval around 0/16 is not empty)."""
    if kind == "rate":
        rows = [(len(g), (sum(1 for x in g if x >= 0.5) + 2) / (len(g) + 4)) for g in groups if g]
        if not rows:
            return None
        return math.sqrt(sum(m * p * (1 - p) for m, p in rows) / sum(m for m, _ in rows))
    rows = [g for g in groups if len(g) >= 2]
    if not rows:
        return None
    ss = sum(sum((x - _mean(g)) ** 2 for x in g) for g in rows)
    return math.sqrt(ss / sum(len(g) - 1 for g in rows))


def cases_needed(true: str, other: str, z: float) -> float:
    """Fewest cases a side at which, with *true* holding at its reference difference, the
    interval contradicts *other*; ``inf`` when no sample would. *z* is the band in standard
    deviations of the reading.

    In units of the spread, with half-width ``h``: ``same`` misses under a true ``up``
    (one past the band) once ``h < 1``; the opposite direction once ``h <= 2z + 1``; and
    ``up``/``down`` misses under a true ``same`` only once the whole interval fits in
    the band, ``h <= z``."""
    limit = z if true == "same" else 1.0 if other == "same" else 2.0 * z + 1.0
    if limit <= 0.0:
        return math.inf
    if math.isinf(limit) or half_width(MIN_N) <= limit:
        return float(MIN_N)
    lo, hi = MIN_N, MIN_N
    while half_width(hi) > limit:
        lo, hi = hi, hi * 2
        if hi > _MAX_CASES:
            return math.inf
    while hi - lo > 1:   # half_width falls with n
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if half_width(mid) > limit else (lo, mid)
    return float(hi)


def decidability(a: str, b: str, n: int, metric: Metric, sd: "float | None") -> dict[str, Any]:
    """How far *n* route FAIL cases (and as many controls) go towards settling predictions *a*
    against *b* on *metric*, from 0 to 1; with the cases each side would need."""
    if sd is None:
        # least favourable: a share at its largest spread, a mean's band negligible next to it
        z, basis = (metric.band / 0.5 if metric.kind == "rate" else 0.0), "unmeasured"
    else:
        z, basis = (math.inf if sd == 0.0 else metric.band / sd), "measured"
    needed = (cases_needed(a, b, z), cases_needed(b, a, z))
    score = [0.0 if n < MIN_N or math.isinf(k) else min(1.0, n / k) for k in needed]
    return {"decidable": round(sum(score) / 2, 4), "n": n, "needed": [None if math.isinf(k) else int(k)
                                                                      for k in needed],
            "spread": None if sd is None else round(sd, 4), "basis": basis}


# -- explanations -------------------------------------------------------------------------------


@dataclass
class Hypothesis:
    id: str
    route: str
    claim: str
    predictions: dict[str, dict[str, str]]   # step key -> reading -> direction (explicit ones only)
    basis: str = ""

    def explicit(self) -> list[tuple[str, str, str]]:
        return [(step, f, d) for step, by in sorted(self.predictions.items()) for f, d in sorted(by.items())]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "route": self.route, "claim": self.claim, "basis": self.basis,
                "predictions": self.predictions}


@dataclass
class ExplanationSet:
    """One judge call's explanations for every active route at one sample revision."""

    revision: int
    key: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    offered: dict[str, dict[str, list[str]]] = field(default_factory=dict)   # route -> step -> readings
    dropped: list[dict[str, Any]] = field(default_factory=list)
    seen: dict[str, list[str]] = field(default_factory=dict)   # route -> steps with readings shown when asked
    reason: str = ""                                            # why the judge was asked
    decided_at_ask: set = field(default_factory=set)            # predictions already decided when asked
    source: str = "judge"            # "judge" | "read_back" | "failed"
    cost: dict[str, Any] = field(default_factory=dict)
    error: "str | None" = None

    def to_dict(self) -> dict[str, Any]:
        return {"revision": self.revision, "key": self.key, "source": self.source, "reason": self.reason,
                "error": self.error,
                "cost": self.cost, "offered": self.offered, "seen": self.seen, "dropped": self.dropped,
                "hypotheses": [h.to_dict() for h in self.hypotheses]}


Outcomes = Callable[[str, str, Metric], dict[str, Any]]   # (route, step key, metric) -> outcome


def check(hypotheses: list[Hypothesis], outcome_of: Outcomes) -> dict[str, dict[str, Any]]:
    """Every hypothesis' state on the current evidence: ``live`` / ``contradicted`` / ``untestable``,
    with each explicit prediction's verdict."""
    out = {}
    for h in hypotheses:
        rows = []
        for step, name, direction in h.explicit():
            metric = next(m for m in CARDS[step] if m.field == name)
            o = outcome_of(h.route, step, metric)
            rows.append({"step": step, "reading": name, "predicted": direction, "band": metric.band,
                         "outcome": o, "verdict": verdict(direction, o.get("ci95"), metric.band)})
        state = ("untestable" if not rows else
                 "contradicted" if any(r["verdict"] == "miss" for r in rows) else "live")
        out[h.id] = {"state": state, "predictions": rows}
    return out


Decidable = Callable[[str, str, str], dict[str, Any]]   # (reading, direction a, direction b) -> decidability


def disagreement(route: str, step: str, live: list[Hypothesis],
                 decidable: "Decidable | None" = None) -> tuple[float, dict[str, Any]]:
    """``D`` of running *step* on *route*, and which pairs of live explanations it separates.

    Each conflicting pair counts by how decidable its conflict is on the route's cases
    (:func:`decidability`; its most decidable reading); without *decidable* every conflict
    counts in full. ``conflict_share`` is the share of conflicting pairs, undiscounted."""
    hyps = [h for h in live if h.route == route]
    if len(hyps) < 2:
        return 0.0, {"n_live": len(hyps), "n_pairs": 0, "conflicting": [], "conflict_share": 0.0}
    pairs = list(combinations(hyps, 2))
    conflicting = []
    for a, b in pairs:
        pa, pb = a.predictions.get(step, {}), b.predictions.get(step, {})
        on = sorted(f for f in set(pa) & set(pb) if pa[f] != pb[f])
        if on:
            row: dict[str, Any] = {"pair": [a.id, b.id], "readings": on}
            if decidable is not None:
                row["by_reading"] = {f: decidable(f, pa[f], pb[f]) for f in on}
                row["decidable"] = max(r["decidable"] for r in row["by_reading"].values())
            conflicting.append(row)
    D = sum(c.get("decidable", 1.0) for c in conflicting) / len(pairs)
    return D, {"n_live": len(hyps), "n_pairs": len(pairs), "conflicting": conflicting,
               "conflict_share": len(conflicting) / len(pairs)}


# -- the judge call ---------------------------------------------------------------------------------

_INSTRUCTIONS = """\
You are helping to triage the errors of a model under test on a benchmark. A
controller has run checks on a sample of the model's EXPLORE cases and grouped
the wrong (FAIL) cases into routes; each route's FAIL cases are compared with
correct (PASS) cases of the same gold answer (its controls). Some tools have not
run on a route yet. The controller will run them one at a time and wants to run
first the ones whose result would tell competing explanations apart.

For every route listed under "routes", propose at most {max_per_route} competing
local explanations of why that route's FAIL cases fail. Local: about these cases
and the mechanisms the tools can see, not the model in general. Competing: they
should differ in what at least one listed tool would show. Do not split one
explanation into variants only to disagree, and propose fewer (or none) when you
have no real alternative.

A tool with "alternative_to" asks a question that another tool (named there)
has already answered on the route: it runs only if it tells explanations apart.

For each explanation and each tool listed under the route's "to_run", predict
for each of its readings the direction of (FAIL mean - control mean):
  "up"      the difference is above +band
  "down"    the difference is below -band
  "same"    the difference is within +-band
  "unknown" the explanation does not imply a direction (the default)
Only predict a direction the explanation implies; a wrong confident prediction
counts against it. Readings measured so far are under "measured"; use them, and
the previous explanations' verdicts under "previous", to revise.

Answer with JSON only, no prose, in this shape:
{{"routes": {{"<route>": [{{"id": "h1", "claim": "<one sentence>",
  "basis": "<which readings it rests on>",
  "predictions": {{"<step>": {{"<reading>": "up|down|same|unknown"}}}}}}]}}}}

Evidence (JSON):
"""


def prompt(context: dict[str, Any], max_per_route: int) -> str:
    return _INSTRUCTIONS.format(max_per_route=max_per_route) + json.dumps(context, indent=1, sort_keys=True,
                                                                           ensure_ascii=False)


def _json_object(text: str) -> Any:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def parse(reply: str, offered: dict[str, dict[str, list[str]]],
          max_per_route: int) -> tuple[list[Hypothesis], list[dict[str, Any]]]:
    """The valid explanations in *reply*, and what was dropped and why.

    Kept: routes and steps that were offered, readings on the step's card,
    the four directions (arrows accepted). A second explanation with the same
    explicit predictions as an earlier one is a duplicate (it can never be told
    apart from it); explanations beyond *max_per_route* are dropped.
    """
    dropped: list[dict[str, Any]] = []
    try:
        data = _json_object(reply)
    except (json.JSONDecodeError, ValueError) as exc:
        return [], [{"what": "reply", "reason": f"not JSON: {exc}"}]
    routes = data.get("routes") if isinstance(data, dict) else None
    if not isinstance(routes, dict):
        return [], [{"what": "reply", "reason": "no 'routes' object"}]
    out: list[Hypothesis] = []
    for route, items in sorted(routes.items()):
        if route not in offered:
            dropped.append({"what": f"route {route}", "reason": "not offered"})
            continue
        if not isinstance(items, list):
            dropped.append({"what": f"route {route}", "reason": "not a list"})
            continue
        kept: list[Hypothesis] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict) or not str(item.get("claim") or "").strip():
                dropped.append({"what": f"{route}[{i}]", "reason": "no claim"})
                continue
            hid = f"{route}.{str(item.get('id') or f'h{i + 1}').strip()}"
            if any(h.id == hid for h in kept):
                hid = f"{route}.h{i + 1}"
            preds: dict[str, dict[str, str]] = {}
            for step, by in (item.get("predictions") or {}).items():
                if step not in offered[route]:
                    dropped.append({"what": f"{hid} {step}", "reason": "step not offered on this route"})
                    continue
                for reading, raw in (by or {}).items() if isinstance(by, dict) else ():
                    d = str(raw).strip().lower()
                    d = _ALIASES.get(d, d)
                    if reading not in offered[route][step]:
                        dropped.append({"what": f"{hid} {step}.{reading}", "reason": "reading not on the card"})
                    elif d not in DIRECTIONS:
                        dropped.append({"what": f"{hid} {step}.{reading}", "reason": f"direction {raw!r}"})
                    elif d != "unknown":
                        preds.setdefault(step, {})[reading] = d
            h = Hypothesis(id=hid, route=route, claim=str(item["claim"]).strip(), predictions=preds,
                           basis=str(item.get("basis") or "").strip())
            if preds and any(k.predictions == preds for k in kept):
                dropped.append({"what": hid, "reason": "same explicit predictions as an earlier explanation"})
                continue
            if len(kept) >= max_per_route:
                dropped.append({"what": hid, "reason": f"more than {max_per_route} explanations"})
                continue
            kept.append(h)
        out += kept
    return out, dropped


class Explainer:
    """Asks a text judge (``generate(prompt) -> str``) for explanations, through the run's store.

    *identity* names the judge (provider, model, effort) and is part of the
    store key; *price_fn* turns the judge's usage counters into USD when it
    reports them (an API judge). A CLI judge reports neither: its calls are
    charged in wall seconds only.
    """

    def __init__(self, judge: Any, *, identity: dict[str, Any], max_per_route: int = 3, max_asks: int = 3,
                 price_fn: "Callable[[dict[str, Any] | None], float | None] | None" = None) -> None:
        self.judge, self.identity, self.price_fn = judge, identity, price_fn
        self.max_per_route, self.max_asks = max_per_route, max_asks

    def propose(self, root: "str | Path", revision: int, context: dict[str, Any],
                offered: dict[str, dict[str, list[str]]]) -> ExplanationSet:
        text = prompt(context, self.max_per_route)
        key = hashlib.sha1(json.dumps({"version": EXPLAIN_VERSION, "judge": self.identity, "prompt": text},
                                      sort_keys=True).encode()).hexdigest()[:16]
        store = Path(root) / "explanations"
        store.mkdir(parents=True, exist_ok=True)
        path = store / f"{key}.json"
        if path.is_file():
            rec = json.loads(path.read_text(encoding="utf-8"))
            hyps, dropped = parse(rec["reply"], offered, self.max_per_route)
            return ExplanationSet(revision, key, hyps, offered=offered, dropped=dropped, source="read_back",
                                  cost=rec["cost"])
        before = usage_counters(self.judge)
        started = time.monotonic()
        try:
            reply = str(self.judge.generate(text))
        except Exception as exc:  # noqa: BLE001 - a judge failure leaves D at 0 for this revision
            cost = {"wall_sec": round(time.monotonic() - started, 3), "calls": {"judge": 1}}
            return ExplanationSet(revision, key, offered=offered, source="failed", cost=cost,
                                  error=f"{type(exc).__name__}: {exc}")
        cost: dict[str, Any] = {"wall_sec": round(time.monotonic() - started, 3), "calls": {"judge": 1}}
        usage = usage_delta(before, usage_counters(self.judge))
        if usage is not None:
            cost["api_usage"] = usage
            cost["usd"] = self.price_fn(usage) if self.price_fn is not None else None
        rec = {"version": EXPLAIN_VERSION, "judge": self.identity, "revision": revision, "prompt": text,
               "reply": reply, "cost": cost}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        hyps, dropped = parse(reply, offered, self.max_per_route)
        return ExplanationSet(revision, key, hyps, offered=offered, dropped=dropped, source="judge", cost=cost)
