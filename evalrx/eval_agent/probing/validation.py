"""MVP-4: issues frozen from a diagnosis, validated once on cases the diagnosis never saw.

**Partition.** :func:`partition_bundle` fixes the data roles before the
controller sees any case. DISCOVERY is the EXPLORE partition the controller
works on; DIAGNOSTIC_VALIDATION is the CONFIRM partition minus every case that
shares a source with DISCOVERY (a source is the media content: a question about
an image the diagnosis saw is not a new image), plus, when a validation-only
source is given (``--triage-validation-source``: cases labelled by their own
Stage 0 pass, never seen by the controller or the judge), its cases under the
same rule. Repair CONFIRM is not part of this protocol (repair is out of
scope), so the bundle says so.

**Issues.** :func:`freeze_issues` turns a triage report into issue versions
whose cohort, signal, direction, band and test can be applied to new cases:

* ``route_signal``: a label-free check (H, N or U) fires more on FAIL than on
  stratum-matched PASS cases by more than ``ROUTE_BAND`` (discovery: the whole
  95% interval of the difference above it);
* ``reading``: on a route's FAIL cases (its layer's filter applied) a step's
  reading differs from the matched PASS controls in a direction, beyond the
  reading's band (discovery: the whole 95% interval outside it). One issue per
  route and step: the reading furthest outside its band, in band units; the
  step's other readings outside their bands are listed, not tested. A step that
  re-measures the check that defines the route, or reads what that check
  implies (``DEFINITIONAL``), is not an issue: the route was cut on it.

No reading reads the label: the controller refuses a carded step whose target is
the gold answer (a FAIL output is not gold by definition, so a gold margin lower
on FAIL restates the label), and the validator refuses a report from another
rules version (it re-runs the steps as they are defined now).

The judge's explanations are not issues: the ones live at an arm's stop are
frozen with the family, and their predictions are checked on the validation
cases of their route, the validator running the steps they name that no issue
ran there (mechanism claims: reported apart, not counted).

**Plan.** :func:`plan_validation` estimates, from DISCOVERY alone (before any
validation output is read), the validation cases each issue will have and the
cases it would need for its interval to clear the band at the discovery effect,
and the cost.

**Validation.** :func:`validate` runs once, on the union of the frozen families
of the given arms: the route checks on the validation FAIL cases and their
matched PASS controls, then each reading issue's step on its route. The family
is written to ``frozen.json`` before anything runs. Every issue claims a
direction, so its verdict is a one-sided test at ``ALPHA / K`` on each side
(Bonferroni over the K issues of the family): its interval is taken at
``1 - 2 * ALPHA / K``, and the verdict is ``supported`` (inside the predicted
region), ``refuted`` (outside it), ``inconclusive`` (neither) or
``unavailable`` (the step could not run, or too few cases). A false
``supported`` anywhere in the family then has probability at most ALPHA, and so
does a false ``refuted``: the side that can only refute no longer spends half of
the error the supported side may.

**Phenomena.** Several readings can say one thing (an attention cut and a noised
image both measuring how much the media carries the answer). Two reading issues
of one route are one phenomenon when their per-case values on the validation
cases go together within the FAIL cases and within the controls (pooled
within-group Spearman, ``|rho| >= SAME_PHENOMENON_RHO``; a difference between
the two groups does not count, since every issue has one); phenomena are the
connected groups. The rule is frozen with the family. Each arm is credited with
its own issues, grouped into phenomena: a phenomenon is supported for an arm
when one of the arm's own issues in it is; the per-issue tally is kept beside it.

**Main metric.** The number of supported phenomena depends on where the
threshold sits when a correlation lies near it, so it is reported over a grid
of thresholds (:data:`RHO_GRID`, :func:`supported_curve`), and an arm found
more than another only where it has more supported phenomena at every
threshold of the grid. The reference threshold only names the phenomena table.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from evalrx.eval_agent.probing.actions import PlanStep
from evalrx.eval_agent.probing.explanations import (
    CARDS,
    MIN_N,
    Metric,
    label_bound,
    outcome,
    primary_readings,
    restates_route,
    t_quantile,
    verdict,
    z_quantile,
)
from evalrx.eval_agent.probing.ledger import Ledger
from evalrx.eval_agent.probing.rules import (
    LAYERS,
    RULES_VERSION,
    Layer,
    candidate_steps,
    matched_controls,
    route_fail_ids,
)
from evalrx.eval_agent.probing.sampler import matched_sample, rank
from evalrx.eval_agent.probing.triage import (
    ROUTE_STEPS,
    difference_interval,
    follow_up_values,
    triage_cases,
)

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch

VALIDATION_VERSION = "mvp4-validate-5"
ALPHA = 0.05
#: A route signal is a difference of shares of cases: the band the CARDS use for shares.
ROUTE_BAND = 0.1
ROUTE_CHECKS = {"H_label_free": "H", "N": "N", "U": "U"}
#: Two reading issues of one route are one phenomenon at this pooled within-group |Spearman rho| (the
#: reference grouping, which names the phenomena table).
SAME_PHENOMENON_RHO = 0.5
#: The grouping thresholds the main metric is reported over.
RHO_GRID = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
#: Paired cases a group needs before it enters that correlation.
MIN_PAIRED = 5
VERDICT_ORDER = ("supported", "refuted", "inconclusive", "unavailable")


# -- partition --------------------------------------------------------------------------------------


def source_key(case: Any) -> str:
    """The case's source: a hash of its media content (two cases on the same image share it),
    or the case itself when it has no media."""
    inputs = getattr(case, "inputs", None)
    h = hashlib.sha1()
    found = False
    for name in ("image", "audio", "video"):
        media = getattr(inputs, name, None)
        if media is None:
            continue
        found = True
        if isinstance(media, (str, os.PathLike)) and Path(media).is_file():
            h.update(Path(media).read_bytes())
        elif hasattr(media, "tobytes"):
            h.update(media.tobytes())
        else:
            h.update(str(media).encode())
    return f"media:{h.hexdigest()[:16]}" if found else f"case:{getattr(case, 'id', '')}"


@dataclass
class PartitionBundle:
    """The data roles of one protocol, fixed before the controller sees a case."""

    diagnostic_val_source: str          # "internal" (CONFIRM) | "internal+source" (plus a validation-only manifest)
    generalisation_unit: str            # what a validated issue generalises to
    discovery_ids: list[str]
    validation_ids: list[str]
    excluded: dict[str, int]            # validation candidates left out, by reason
    n_sources: dict[str, int]
    rule: str
    seed: int
    confirm_split: float
    manifest: dict[str, Any] = field(default_factory=dict)
    validation_from: dict[str, int] = field(default_factory=dict)   # validation cases by origin (confirm, source)
    source_manifest: dict[str, Any] = field(default_factory=dict)   # the validation-only manifest, when given
    roles: dict[str, Any] = field(default_factory=lambda: {
        "controller": "DISCOVERY", "judge": "DISCOVERY (route summaries only)",
        "validator": "DIAGNOSTIC_VALIDATION", "repair_confirm": None})

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["sha1"] = hashlib.sha1(json.dumps({k: v for k, v in out.items() if k != "sha1"}, sort_keys=True,
                                              default=str).encode()).hexdigest()[:16]
        return out


def _file_info(path: "str | Path | None") -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {}
    return {"path": str(path), "sha1": hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]}


def partition_bundle(discovery: "CaseBatch", confirm: "CaseBatch | None", *, seed: int, confirm_split: float,
                     manifest: "str | Path | None" = None, source: "CaseBatch | None" = None,
                     source_manifest: "str | Path | None" = None) -> "tuple[PartitionBundle, CaseBatch]":
    """DIAGNOSTIC_VALIDATION = CONFIRM minus the cases sharing a source with DISCOVERY, plus the cases of
    a validation-only *source* (a manifest the controller never sees) that share none with it either."""
    from evalrx.core.case import CaseBatch

    seen = {source_key(c) for c in discovery}
    taken = {str(c.id) for c in discovery} | {str(c.id) for c in confirm or []}
    kept: list = []
    excluded: dict[str, int] = {}
    origin: dict[str, int] = {}
    for name, batch in (("confirm", confirm), ("source", source)):
        if batch is None:
            continue
        origin[name] = 0
        shared = "shares_a_source_with_discovery" if name == "confirm" else "source_shares_a_source_with_discovery"
        excluded[shared] = 0
        for c in batch:
            if source_key(c) in seen:
                excluded[shared] += 1
            elif name == "source" and str(c.id) in taken:   # a row the pool already holds, or a repeat: one id, one case
                excluded["source_repeats_an_id"] = excluded.get("source_repeats_an_id", 0) + 1
            else:
                kept.append(c)
                origin[name] += 1
                if name == "source":
                    taken.add(str(c.id))
    bundle = PartitionBundle(
        diagnostic_val_source="internal" if source is None else "internal+source",
        generalisation_unit="new media (no source shared with DISCOVERY)",
        discovery_ids=[str(c.id) for c in discovery], validation_ids=[str(c.id) for c in kept],
        excluded=excluded,
        n_sources={"discovery": len(seen), "validation": len({source_key(c) for c in kept})},
        rule=("DISCOVERY = EXPLORE; DIAGNOSTIC_VALIDATION = CONFIRM minus every case whose media content "
              "also appears in DISCOVERY"
              + ("; plus every case of the validation-only source whose media content does not"
                 if source is not None else "")
              + "; repair CONFIRM: none (repair is out of scope)"),
        seed=int(seed), confirm_split=float(confirm_split), manifest=_file_info(manifest),
        validation_from=origin, source_manifest=_file_info(source_manifest))
    return bundle, CaseBatch(kept)


# -- issues --------------------------------------------------------------------------------------------


@dataclass
class IssueRecord:
    """One frozen issue version: what is claimed, on which cases, and how it is tested."""

    kind: str                       # "route_signal" | "reading"
    route: str
    direction: str                  # "up" | "down"
    band: float
    metric_kind: str                # "rate" | "mean"
    claim: str
    check: str = ""                 # route_signal: the summary tag (H_label_free, N, U)
    step: str = ""                  # reading: the step key
    reading: str = ""               # reading: the field
    rule_id: "str | None" = None    # the layer whose case filter narrows the route's FAIL cases
    secondary: list[str] = field(default_factory=list)
    discovery: dict[str, Any] = field(default_factory=dict)
    spread: "float | None" = None   # a mean reading's within-group spread in DISCOVERY (for the plan)

    @property
    def issue_id(self) -> str:
        key = {"version": VALIDATION_VERSION, "kind": self.kind, "route": self.route, "direction": self.direction,
               "band": self.band, "check": self.check, "step": self.step, "reading": self.reading,
               "rule_id": self.rule_id}
        return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {"issue_id": self.issue_id, **asdict(self)}


def _layer_for(route: str, key: str, layers: "tuple[Layer, ...]") -> "Layer | None":
    for x in layers:
        if x.route == route and (key in x.served_by or key in {c.name for c in x.candidates}):
            return x
    return None


def _margin(direction: str, ci: list[float], band: float) -> float:
    """How far the interval lies outside the band in the claimed direction, in band units."""
    return ((ci[0] - band) if direction == "up" else (-ci[1] - band)) / band


def _spread_from(o: dict[str, Any]) -> "float | None":
    """A mean reading's pooled spread, recovered from its Welch interval (approximate df)."""
    nf, nc = o["n"]
    if nf < MIN_N or nc < MIN_N or "ci95" not in o:
        return None
    se = (o["ci95"][1] - o["ci95"][0]) / 2 / t_quantile(0.95, nf + nc - 2)
    return se / math.sqrt(1 / nf + 1 / nc)


def freeze_issues(report: dict[str, Any], layers: "tuple[Layer, ...]" = LAYERS) -> list[IssueRecord]:
    """The issue versions a triage report supports on DISCOVERY (see the module docstring)."""
    issues: list[IssueRecord] = []
    summary = report.get("summary") or {}
    for tag, route in ROUTE_CHECKS.items():
        row = summary.get(tag) or {}
        ci = row.get("signal_ci95")
        if ci is None or ci[0] <= ROUTE_BAND:
            continue
        f, p = row["fail"], row["pass"]
        issues.append(IssueRecord(
            kind="route_signal", route=route, direction="up", band=ROUTE_BAND, metric_kind="rate", check=tag,
            claim=(f"the {route} check fires more often on FAIL cases than on stratum-matched PASS cases, by more "
                   f"than {ROUTE_BAND}"),
            discovery={"fail": f"{f['n_fired']}/{f['n_tested']}", "pass": f"{p['n_fired']}/{p['n_tested']}",
                       "rates": [f["rate"], p["rate"]], "n": [f["n_tested"], p["n_tested"]],
                       "diff": round((f["rate"] or 0.0) - (p["rate"] or 0.0), 4), "ci95": ci}))
    for route, by_key in sorted((report.get("outcomes") or {}).items()):
        for key, by_field in sorted(by_key.items()):
            if restates_route(route, key) or key not in CARDS:
                continue
            bands = {m.field: m for m in CARDS[key]}
            outside = [(f, o) for f, o in sorted(by_field.items())
                       if f in bands and o.get("observed") in ("up", "down")]
            if not outside:
                continue
            f, o = max(outside, key=lambda fo: _margin(fo[1]["observed"], fo[1]["ci95"], bands[fo[0]].band))
            metric, layer = bands[f], _layer_for(route, key, layers)
            issues.append(IssueRecord(
                kind="reading", route=route, direction=o["observed"], band=metric.band, metric_kind=metric.kind,
                step=key, reading=f,
                rule_id=layer.rule_id if layer is not None and layer.case_filter is not None else None,
                claim=(f"on route {route}'s FAIL cases, {key}.{f} ({metric.meaning}) is "
                       f"{'higher' if o['observed'] == 'up' else 'lower'} than on stratum-matched PASS controls "
                       f"by more than {metric.band}"),
                secondary=[f"{f2}: {o2['observed']}" for f2, o2 in outside if f2 != f],
                discovery={k: o[k] for k in ("n", "fail_mean", "control_mean", "diff", "ci95") if k in o},
                spread=_spread_from(o) if metric.kind == "mean" else None))
    return issues


# -- plan -----------------------------------------------------------------------------------------------


def _cases_needed(issue: IssueRecord, level: float) -> float:
    """Fewest cases a side for the interval at *level* around the discovery difference to clear the
    band in the claimed direction; ``inf`` when the discovery difference does not clear it."""
    d = abs(float(issue.discovery.get("diff") or 0.0))
    if d <= issue.band:
        return math.inf
    room = d - issue.band
    if issue.metric_kind == "rate":
        if issue.kind == "route_signal":
            pf, pc = issue.discovery["rates"]
        else:
            pf, pc = issue.discovery.get("fail_mean", 0.5), issue.discovery.get("control_mean", 0.5)
        var = (pf or 0.0) * (1 - (pf or 0.0)) + (pc or 0.0) * (1 - (pc or 0.0))
        if var <= 0:
            return float(MIN_N)
        return float(max(MIN_N, math.ceil(var * (z_quantile(level) / room) ** 2)))
    s = issue.spread
    if s is None:
        return math.inf
    if s == 0:
        return float(MIN_N)
    n = MIN_N
    while t_quantile(level, 2 * n - 2) * s * math.sqrt(2 / n) >= room:
        n += 1 if n < 64 else n // 8
        if n > 100000:
            return math.inf
    return float(n)


def _per_case_cost(report: dict[str, Any], key: str) -> "float | None":
    rows = [(float((e.get("cost") or {}).get("wall_sec") or 0.0), int(e.get("n_cases") or 0))
            for e in report.get("executions") or [] if e.get("key") == key and (e.get("cost") or {}).get("calls")]
    n = sum(k for _, k in rows)
    return sum(s for s, _ in rows) / n if n else None


def plan_validation(issues: list[IssueRecord], reports: list[dict[str, Any]], *, n_validation: int,
                    discovery_fail_rate: float, level: float) -> list[dict[str, Any]]:
    """Per issue, from DISCOVERY only: the validation cases it should have, the cases it needs, and the
    estimated seconds; nothing here reads a validation output."""
    expected_fail = n_validation * discovery_fail_rate
    rows = []
    for issue in issues:
        shares = []
        for r in reports:
            routes = (r.get("revisions") or [{}])[-1].get("fail_routes") or {}
            n_fail = len((r.get("sample") or {}).get("fail_ids") or [])
            if n_fail:
                shares.append(routes.get(issue.route, 0) / n_fail if issue.kind == "reading" else 1.0)
        share = max(shares) if shares else 0.0
        n_expected = int(round(expected_fail * share))
        needed = _cases_needed(issue, level)
        costs = [c for r in reports if (c := _per_case_cost(r, issue.step)) is not None] if issue.step else []
        rows.append({"issue_id": issue.issue_id, "n_expected": n_expected,
                     "n_needed": None if math.isinf(needed) else int(needed),
                     "expected_decidable": n_expected >= needed,
                     "est_sec": round(2 * n_expected * max(costs), 1) if costs else None})
    return rows


# -- validation ---------------------------------------------------------------------------------------------


def _verdict(direction: str, ci: "list[float] | None", band: float) -> str:
    return {"hit": "supported", "miss": "refuted", "undetermined": "inconclusive"}[verdict(direction, ci, band)]


def _route_signal(records: list[dict[str, Any]], tag: str, level: float) -> dict[str, Any]:
    state = {"H_label_free": lambda r: r["H"]["state_label_free"], "N": lambda r: r["N"]["state"],
             "U": lambda r: r["U"]["state"]}[tag]
    counts = {}
    for label in ("fail", "pass"):
        tested = [r for r in records if r["label"] == label and state(r) in ("fired", "clear")]
        counts[label] = (sum(1 for r in tested if state(r) == "fired"), len(tested))
    (kf, nf), (kp, np_) = counts["fail"], counts["pass"]
    out: dict[str, Any] = {"fail": f"{kf}/{nf}", "pass": f"{kp}/{np_}", "n": [nf, np_]}
    if nf < MIN_N or np_ < MIN_N:
        return {**out, "verdict": "unavailable", "reason": f"fewer than {MIN_N} tested cases a side"}
    ci = difference_interval(kf, nf, kp, np_, z=z_quantile(level))
    ci95 = difference_interval(kf, nf, kp, np_)
    return {**out, "diff": round(kf / nf - kp / np_, 4), "ci": ci, "ci95": ci95,
            "verdict": _verdict("up", ci, ROUTE_BAND)}


# -- phenomena ------------------------------------------------------------------------------------------


def _ranks(xs: list[float]) -> list[float]:
    """Ranks with ties averaged."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def within_group_rho(a: dict[str, float], b: dict[str, float], groups: list[list[str]]) -> "float | None":
    """Spearman correlation of two per-case readings inside each group of cases, pooled: ranks are
    taken and centred within a group, so a shift between the groups does not count. ``None`` when no
    group has ``MIN_PAIRED`` cases with both readings, or a reading is constant within every group."""
    sxy = sxx = syy = 0.0
    for ids in groups:
        both = [c for c in ids if c in a and c in b]
        if len(both) < MIN_PAIRED:
            continue
        ra, rb = _ranks([a[c] for c in both]), _ranks([b[c] for c in both])
        ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
        sxy += sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        sxx += sum((x - ma) ** 2 for x in ra)
        syy += sum((y - mb) ** 2 for y in rb)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def group_phenomena(family: list[IssueRecord], results: dict[str, dict[str, Any]],
                    per_case: dict[str, tuple[dict[str, float], list[str], list[str]]]
                    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Connected groups of a route's reading issues whose validation values go together (see the module
    docstring); a route signal, or a reading without validation values, is a phenomenon of its own.
    Returns the phenomena and every within-route correlation computed (joined or not)."""
    links = []
    readings = [i for i in family if i.kind == "reading" and i.issue_id in per_case]
    for n, a in enumerate(readings):
        for b in readings[n + 1:]:
            if a.route != b.route:
                continue
            va, fail_ids, controls = per_case[a.issue_id]
            rho = within_group_rho(va, per_case[b.issue_id][0], [fail_ids, controls])
            if rho is not None:
                links.append({"issues": [a.issue_id, b.issue_id], "rho": round(rho, 3)})
    root = _components([i.issue_id for i in family], links, SAME_PHENOMENON_RHO)
    groups: dict[str, list[IssueRecord]] = {}
    for i in family:
        groups.setdefault(root[i.issue_id], []).append(i)
    out = []
    for n, members in enumerate(groups.values(), 1):
        ids = {i.issue_id for i in members}
        out.append({"id": f"P{n}", "route": members[0].route, "issues": sorted(ids),
                    "label": "; ".join(i.check or f"{i.step}.{i.reading}" for i in members),
                    "verdict": _best(results[i.issue_id]["verdict"] for i in members),
                    "links": [lk for lk in links if set(lk["issues"]) <= ids]})
    return out, links


def _components(ids: list[str], links: list[dict[str, Any]], threshold: float) -> dict[str, str]:
    """Each issue's connected group, joining the issues of every link with ``|rho| >= threshold``."""
    parent = {i: i for i in ids}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for lk in links:
        if abs(lk["rho"]) >= threshold:
            a, b = lk["issues"]
            parent[root(b)] = root(a)
    return {i: root(i) for i in ids}


def supported_curve(arms: dict[str, list[str]], verdicts: dict[str, str], links: list[dict[str, Any]],
                    grid: tuple[float, ...] = RHO_GRID) -> dict[str, Any]:
    """Each arm's supported phenomena at every grouping threshold of *grid*, and the differences between
    arms that hold at all of them (*arms*: issue ids per arm; *verdicts*: per issue of the family;
    *links*: the within-route correlations). A difference that appears at some thresholds only says how
    readings were grouped, not which arm found more."""
    curves: dict[str, list[int]] = {arm: [] for arm in arms}
    for t in grid:
        root = _components(list(verdicts), links, t)
        for arm, ids in arms.items():
            own: dict[str, list[str]] = {}
            for i in ids:
                own.setdefault(root[i], []).append(verdicts[i])
            curves[arm].append(sum(1 for v in own.values() if "supported" in v))
    names = sorted(arms)
    holds = [{"higher": a, "lower": b, "by": [x - y for x, y in zip(curves[a], curves[b])]}
             for a in names for b in names if a != b and all(x > y for x, y in zip(curves[a], curves[b]))]
    return {"grid": list(grid), "supported": curves, "differences": holds}


def credit_arms(families: dict[str, list[IssueRecord]], results: dict[str, dict[str, Any]],
                phenomenon_of: dict[str, str], reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each arm's own issues, grouped into phenomena: a phenomenon takes the best verdict among the
    arm's own issues in it (another arm's issue in the same phenomenon lends it nothing)."""
    credit = {}
    for arm, issues in families.items():
        tally = Counter(results[i.issue_id]["verdict"] for i in issues)
        own: dict[str, list[str]] = {}
        for i in issues:
            own.setdefault(phenomenon_of[i.issue_id], []).append(results[i.issue_id]["verdict"])
        by_phenomenon = Counter(_best(v) for v in own.values())
        credit[arm] = {"n_phenomena": len(own), **{f"phenomena_{v}": by_phenomenon.get(v, 0) for v in VERDICT_ORDER},
                       "n_issues": len(issues), **{v: tally.get(v, 0) for v in VERDICT_ORDER},
                       "charged_sec": ((reports.get(arm) or {}).get("cost") or {}).get("charged", {}).get("wall_sec")}
    return credit


def _best(verdicts: Any) -> str:
    """A group's verdict: supported if any member is, else refuted if any is, and so on."""
    found = set(verdicts)
    return next((v for v in VERDICT_ORDER if v in found), "unavailable")


def validate(model: Any, discovery: "CaseBatch", validation: "CaseBatch", families: dict[str, list[IssueRecord]],
             reports: dict[str, dict[str, Any]], *, out_dir: "str | Path", model_identity: dict[str, Any],
             bundle: "PartitionBundle | None" = None, seed: int = 0, stratum: str = "gold",
             host_assumed_finish: bool = False, answer_contract: bool = False, n_needs_counterfactual: bool = False,
             task_kind: "str | None" = None, answer_fn: Any = None,
             analyzer_kwargs: "dict[str, dict[str, Any]] | None" = None, price_fn: Any = None,
             per_call_generation: bool = True, chunk: int = 8, budget_sec: float = 0.0,
             battery: "tuple[PlanStep, ...] | None" = None, layers: "tuple[Layer, ...]" = LAYERS,
             run_logger: Any = None, progress: "Callable[[str], None] | None" = print) -> dict[str, Any]:
    """Validate the union of the arms' frozen *families* once on *validation*; credit every arm."""
    from evalrx.eval_agent.probing.controller import BATTERY, POLICY_VERSION, _Run, analyzer_extras

    battery = battery or BATTERY
    stale = sorted(arm for arm, r in reports.items() if r.get("rules_version") != RULES_VERSION)
    if stale:
        raise ValueError(f"reports {stale} come from another rules version than {RULES_VERSION}: the "
                         "validator re-runs the steps as they are defined now")
    if bound := label_bound((*battery, *candidate_steps(layers)), analyzer_kwargs):
        raise ValueError(f"steps {bound} read the gold answer (target='gold'); use target='prediction'")
    say = progress or (lambda _msg: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    union: dict[str, IssueRecord] = {}
    for issues in families.values():
        for issue in issues:
            union.setdefault(issue.issue_id, issue)
    family = sorted(union.values(), key=lambda i: (i.kind != "route_signal", i.route, i.step, i.reading))
    k = max(1, len(family))
    level = 1 - 2 * ALPHA / k   # one-sided at ALPHA / K on the claimed side
    n_disc = len(discovery)
    fail_rate = sum(1 for c in discovery if getattr(getattr(c, "label", None), "value", None) == "fail") / max(1, n_disc)
    plan = plan_validation(family, list(reports.values()), n_validation=len(validation),
                           discovery_fail_rate=fail_rate, level=level)
    frozen = {"version": VALIDATION_VERSION, "alpha": ALPHA, "k": k, "level": level,
              "test": "one-sided at alpha/K on the claimed side of each issue (and of each primary prediction)",
              "family": [i.to_dict() for i in family], "plan": plan,
              "arms": {arm: [i.issue_id for i in issues] for arm, issues in families.items()},
              "phenomenon_rule": {"statistic": "pooled within-group Spearman rho of the per-case values on the "
                                               "validation FAIL cases and on their controls, same route",
                                  "threshold": SAME_PHENOMENON_RHO, "min_paired_per_group": MIN_PAIRED,
                                  "grid": list(RHO_GRID),
                                  "comparison": "an arm found more than another only where its supported "
                                                "phenomena are more at every threshold of the grid"},
              "partition": bundle.to_dict() if bundle is not None else None,
              "explanations": live_explanations(reports)}
    frozen["sha1"] = hashlib.sha1(json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()[:16]
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2, default=str) + "\n", encoding="utf-8")
    say(f"[validate] froze {k} issue(s) from {len(families)} arm(s) (sha1 {frozen['sha1']}); level {level:.4f}")

    # -- from here on the validation partition is read --------------------------------------------------
    extras = analyzer_extras(analyzer_kwargs, discovery, seed, donor_pool=validation)
    run = _Run(model, validation, Ledger(out), context={"policy": POLICY_VERSION, "model": model_identity,
                                                         "role": "validation"},
               extras=extras, task_kind=task_kind, budget_sec=budget_sec, price_fn=price_fn,
               run_logger=run_logger, say=say, per_call_generation=per_call_generation, chunk=chunk)
    n_fail = sum(1 for c in validation if getattr(getattr(c, "label", None), "value", None) == "fail")
    sample, draw = matched_sample(validation, n_fail, seed=seed, stratum=stratum)
    for step in battery:
        if step.name in ROUTE_STEPS:
            run.run_step(step, list(draw.case_ids), kind="validation", revision=0)
    records = triage_cases(sample, run.evidence, host_assumed_finish=host_assumed_finish,
                           answer_contract=answer_contract, n_needs_counterfactual=n_needs_counterfactual,
                           answer_fn=answer_fn, stratum=stratum)
    rank_of = partial(rank, seed)
    by_rule = {x.rule_id: x for x in layers}
    steps = {s.name: s for s in (*battery, *candidate_steps(layers))}
    results: dict[str, dict[str, Any]] = {}
    measured: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    per_case: dict[str, tuple[dict[str, float], list[str], list[str]]] = {}
    for issue in family:
        if issue.kind == "route_signal":
            results[issue.issue_id] = _route_signal(records, issue.check, level)
            continue
        layer = by_rule.get(issue.rule_id) if issue.rule_id else None
        fail_ids = route_fail_ids(records, issue.route, layer.case_filter if layer is not None else None)
        controls = matched_controls(records, fail_ids, rank_of)[0]
        row: dict[str, Any] = {"n_route_fail": len(fail_ids), "n_controls": len(controls)}
        step = steps.get(issue.step)
        if step is None:
            results[issue.issue_id] = {**row, "verdict": "unavailable", "reason": f"no step {issue.step!r}"}
            continue
        if len(fail_ids) < MIN_N or len(controls) < MIN_N:
            results[issue.issue_id] = {**row, "verdict": "unavailable",
                                       "reason": f"fewer than {MIN_N} route FAIL cases or controls"}
            continue
        record = run.run_step(step, [*fail_ids, *controls], kind="validation", revision=0, route=issue.route)
        ev = run.evidence.get(step.name)
        if ev is None or not ev.chunks or ev.refused() is not None:
            why = ("refused by its own fidelity check" if ev is not None and ev.refused() is not None else
                   (record or {}).get("reason") or (record or {}).get("error") or "not run")
            results[issue.issue_id] = {**row, "verdict": "unavailable", "reason": str(why)}
            continue
        measured[(issue.route, step.name)] = (fail_ids, controls)
        metric = Metric(issue.reading, "", issue.band, issue.metric_kind)
        values = {cid: follow_up_values(ev, run.by_id[cid], answer_fn) for cid in (*fail_ids, *controls)}
        o = outcome(values, fail_ids, controls, metric, level=level)
        o95 = outcome(values, fail_ids, controls, metric)
        per_case[issue.issue_id] = ({cid: float(v["values"][issue.reading]) for cid, v in values.items()
                                     if v.get("state") == "measured" and issue.reading in v["values"]},
                                    fail_ids, controls)
        if "ci95" not in o:
            results[issue.issue_id] = {**row, **o, "verdict": "unavailable",
                                       "reason": f"fewer than {MIN_N} measured cases a side"}
            continue
        results[issue.issue_id] = {**row, "n": o["n"], "fail_mean": o.get("fail_mean"),
                                   "control_mean": o.get("control_mean"), "diff": o["diff"], "ci": o["ci95"],
                                   "ci95": o95.get("ci95"), "verdict": _verdict(issue.direction, o["ci95"], issue.band)}
    say(f"[validate] verdicts: {dict(Counter(r['verdict'] for r in results.values()))}")

    # a live explanation's primary predictions can name steps no issue ran on its route: run those too,
    # on the route's validation FAIL cases and controls, so every mechanism claim meets new cases
    for route, key in sorted({(h["route"], step) for rows in frozen["explanations"].values() for h in rows
                              for step, _ in h["primary"]}):
        if (route, key) in measured or key not in steps or restates_route(route, key):
            continue
        layer = _layer_for(route, key, layers)
        fail_ids = route_fail_ids(records, route, layer.case_filter if layer is not None else None)
        controls = matched_controls(records, fail_ids, rank_of)[0]
        if len(fail_ids) < MIN_N or len(controls) < MIN_N:
            continue
        run.run_step(steps[key], [*fail_ids, *controls], kind="validation", revision=0, route=route)
        ev = run.evidence.get(key)
        if ev is not None and ev.chunks and ev.refused() is None:
            measured[(route, key)] = (fail_ids, controls)

    phenomena, links = group_phenomena(family, results, per_case)
    phenomenon_of = {i: ph["id"] for ph in phenomena for i in ph["issues"]}
    say(f"[validate] {len(phenomena)} phenomena: {dict(Counter(ph['verdict'] for ph in phenomena))}")
    credit = credit_arms(families, results, phenomenon_of, reports)
    curve = supported_curve(frozen["arms"], {i: r["verdict"] for i, r in results.items()}, links)
    report = {
        "version": VALIDATION_VERSION, "frozen_sha1": frozen["sha1"], "alpha": ALPHA, "k": k, "level": level,
        "partition": frozen["partition"],
        "sample": draw.to_dict(),
        "issues": [{**i.to_dict(), "plan": next(p for p in plan if p["issue_id"] == i.issue_id),
                    "validation": results[i.issue_id], "phenomenon": phenomenon_of[i.issue_id]} for i in family],
        "phenomena": phenomena,
        "phenomenon_links": links,
        "arms": credit,
        "phenomenon_curve": curve,
        "explanations": _check_explanations(frozen["explanations"], run, measured, answer_fn),
        "cost": {"charged_sec": round(run.charged("wall_sec"), 3), "executions": len(run.executions)},
        "_note": ("each issue's interval is at 1 - 2 alpha/K: a one-sided test at alpha/K on its claimed side "
                  "(Bonferroni over the K issues of the family); ci95 "
                  "is the unadjusted interval, for reference. Verdicts: supported = inside the claimed region, "
                  "refuted = outside it, inconclusive = neither, unavailable = the step or the cases were missing. "
                  "A phenomenon is supported for an arm when one of the arm's own issues in it is"),
    }
    (out / "validation_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "validation_report.md").write_text(markdown(report), encoding="utf-8")
    return report


def live_explanations(reports: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each arm's explanations still live at its stop, with their predictions (frozen with the family)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for arm, r in reports.items():
        sets = ((r.get("explanations") or {}).get("sets")) or []
        if not sets or not sets[-1]["hypotheses"]:
            continue
        last = sets[-1]
        out[arm] = [{**{k: h[k] for k in ("id", "route", "claim", "predictions")},
                     "primary": sorted(primary_readings(h))} for h in last["hypotheses"]
                    if last["final"][h["id"]]["state"] == "live"]
    return out


def _check_explanations(explanations: dict[str, list[dict[str, Any]]], run: Any,
                        measured: dict[tuple[str, str], Any], answer_fn: Any) -> dict[str, Any]:
    """The frozen live explanations, judged on the validation evidence of their route by their primary
    predictions (mechanism claims: reported, not counted).

    The distinct primary predictions of all arms are one family: each is a one-sided test at
    ``ALPHA / K_e`` (its interval at ``1 - 2 * ALPHA / K_e``). An explanation is ``supported`` when
    every primary prediction hits, ``refuted`` when one misses, else ``undecided``; ``untestable``
    without a primary prediction off its route's cut. Other predictions are reported at 95%,
    unadjusted, where their step ran anyway. A prediction on a reading the route is cut on is
    ``definitional`` (never measured, neither support nor contradiction); any other stays ``not
    measured`` only where its step could not run there (too few cases, the budget, or a refused
    fidelity check)."""
    tests = sorted({(h["route"], step, f, h["predictions"][step][f]) for hyps in explanations.values()
                    for h in hyps for step, f in h["primary"] if not restates_route(h["route"], step)})
    k_e = max(1, len(tests))
    level = 1 - 2 * ALPHA / k_e
    arms: dict[str, Any] = {}
    for arm, hyps in explanations.items():
        rows = []
        for h in hyps:
            primary = {tuple(x) for x in h["primary"]}
            preds = []
            for step, by in sorted(h["predictions"].items()):
                for f, d in sorted(by.items()):
                    row = {"step": step, "reading": f, "predicted": d, "primary": (step, f) in primary}
                    if restates_route(h["route"], step):
                        # the route is cut on this reading: it can neither support nor contradict
                        preds.append({**row, "verdict": "definitional"})
                        continue
                    if (h["route"], step) not in measured:
                        preds.append({**row, "verdict": "not measured"})
                        continue
                    fail_ids, controls = measured[(h["route"], step)]
                    metric = next(m for m in CARDS[step] if m.field == f)
                    ev = run.evidence[step]
                    values = {cid: follow_up_values(ev, run.by_id[cid], answer_fn) for cid in (*fail_ids, *controls)}
                    o = outcome(values, fail_ids, controls, metric, level=level if row["primary"] else 0.95)
                    preds.append({**row, "observed": o.get("observed"), "n": o.get("n"), "diff": o.get("diff"),
                                  "ci": o.get("ci95"), "verdict": verdict(d, o.get("ci95"), metric.band)})
            decisive = [p for p in preds if p["primary"] and p["verdict"] != "definitional"]
            state = ("untestable" if not decisive else
                     "refuted" if any(p["verdict"] == "miss" for p in decisive) else
                     "supported" if all(p["verdict"] == "hit" for p in decisive) else "undecided")
            rows.append({"id": h["id"], "route": h["route"], "claim": h["claim"], "predictions": preds,
                         "state": state})
        arms[arm] = rows
    return {"k": k_e, "level": level, "arms": arms}


def markdown(report: dict[str, Any]) -> str:
    part = report.get("partition") or {}
    lines = [f"# Diagnostic validation ({report['version']})", "",
             f"Frozen family sha1 `{report['frozen_sha1']}`: {report['k']} issue(s), intervals at "
             f"{report['level']:.4f} (one-sided Bonferroni, alpha {report['alpha']}).",
             f"Partition: {part.get('rule', '-')}; {len(part.get('validation_ids') or [])} validation cases "
             f"({(part.get('excluded') or {})}"
             + (f"; from {part['validation_from']}" if part.get("validation_from") else "")
             + f"); sample {len(report['sample']['fail_ids'])} FAIL + "
             f"{len(report['sample']['pass_ids'])} PASS. Validation cost {report['cost']['charged_sec']} s.", "",
             "## Issues", "", "| issue | phenomenon | claim | discovery | plan: expected / needed | validation "
             "| verdict |", "|---|---|---|---|---|---|---|"]
    for i in report["issues"]:
        d, v, p = i["discovery"], i["validation"], i["plan"]
        disc = f"diff {d.get('diff')} CI {d.get('ci95')} n {d.get('n')}"
        val = (f"diff {v.get('diff')} CI {v.get('ci')} n {v.get('n')}" if "diff" in v else v.get("reason", "-"))
        lines.append(f"| {i['issue_id']} | {i['phenomenon']} | {i['claim']} | {disc} | {p['n_expected']} / "
                     f"{p['n_needed']} | {val} | **{v['verdict']}** |")
    lines += ["", "## Phenomena", "", "| phenomenon | route | issues | within-group rho | verdict |",
              "|---|---|---|---|---|"]
    for ph in report["phenomena"]:
        rho = ", ".join(str(lk["rho"]) for lk in ph["links"]) or "-"
        lines.append(f"| {ph['id']} | {ph['route']} | {ph['label']} | {rho} | **{ph['verdict']}** |")
    lines += ["", "## Arms", "", "| arm | phenomena | supported | refuted | inconclusive | unavailable "
              "| issues | supported | charged s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, c in report["arms"].items():
        lines.append(f"| {arm} | {c['n_phenomena']} | {c['phenomena_supported']} | {c['phenomena_refuted']} "
                     f"| {c['phenomena_inconclusive']} | {c['phenomena_unavailable']} | {c['n_issues']} "
                     f"| {c['supported']} | {c['charged_sec']} |")
    curve = report.get("phenomenon_curve")
    if curve:
        lines += ["", "## Supported phenomena over the grouping threshold (main metric)", "",
                  "| arm | " + " | ".join(f"{t:g}" for t in curve["grid"]) + " |",
                  "|---|" + "---:|" * len(curve["grid"])]
        lines += [f"| {arm} | " + " | ".join(map(str, counts)) + " |" for arm, counts in curve["supported"].items()]
        higher: dict[str, list[str]] = {}
        for d in curve["differences"]:
            higher.setdefault(d["higher"], []).append(d["lower"])
        lines += ["", "Differences that hold at every threshold: "
                  + ("; ".join(f"{a} > {', '.join(sorted(bs))}" for a, bs in sorted(higher.items())) if higher
                     else "none (every difference between arms depends on how readings are grouped).")]
    expl = report.get("explanations") or {}
    if expl.get("arms"):
        lines += ["", "## Explanations (mechanism claims: reported, not counted)", "",
                  f"Judged by their primary predictions: {expl['k']} distinct test(s), one-sided at "
                  f"{expl['level']:.4f} (Bonferroni); other predictions at 95%, reported only.", ""]
        for arm, rows in expl["arms"].items():
            for h in rows:
                checked = [("**primary** " if p["primary"] else "") + f"{p['step']}.{p['reading']} {p['predicted']} "
                           f"-> {p['verdict']}" for p in h["predictions"] if p["verdict"] != "not measured"]
                lines.append(f"- {arm} {h['id']} [{h['state']}]: {h['claim']} ({'; '.join(checked) or 'unmeasured'})")
    lines += ["", f"_{report['_note']}._", ""]
    return "\n".join(lines)
