"""Fused explore -> confirm pipeline (DESIGN §4).

One pass that DISCOVERS freely and CONFIRMS rigorously, with the two split apart:

1. ``_split_records`` — deterministic, stratified (by label) EXPLORE / CONFIRM
   partition (mirrors ``VLDiagnoseLoop._split_explore_confirm``). The explorer and
   catalog planner see ONLY explore rows; every e-value/CI is computed only on the
   disjoint confirm rows. That is the double-dip firewall.
2. Discovery on EXPLORE — TWO sources, blind to each other:
   - the LAMBDA explorer (free codegen EDA) -> observations / charts /
     candidate_signals, each optionally carrying a deterministic ``recipe``;
   - the catalog planner -> the per-case signal columns already present.
   Their union is deduped by signal name (estimand proxy; value-level estimand
   identity is an open question, see DESIGN §10).
3. Operationalization bridge — each explorer recipe is compiled on the CONFIRM rows
   into a frozen per-case column (``compile_recipe``) and merged into the confirm
   ``StatsInput.per_case`` so it competes like any analyzer signal.
4. Confirmation on CONFIRM — the real M2 firewall (``StatsAnalysisAgent``) runs the
   validated catalog over the EXPANDED signal family and produces effect/CI/e-value
   verdicts + e-BH ``corrected_rejections`` + conclusion. The explorer never decides.
5. Output assembly — per-signal host verdicts (provenance-tagged), and a graceful
   ``recommended_confirmatory_tests`` channel for candidates that could not be
   operationalized / were underpowered / had no confirm data. Nothing is silently
   dropped and nothing is over-claimed.

Multiplicity note: paired/e-value tools use e-BH and marginal p-value tools use
BH. The report's ``rejected_result_keys`` name each surviving result, so the
exact tested family can be audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalrx.analysis.operationalize import (
    RecipeError,
    SignalRecipe,
    compile_recipe,
)
from evalrx.analysis.stats_tools import build_stats_input_from_records

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class FusedSignal:
    """One candidate signal confirmed (or attempted) on the held-out CONFIRM split."""

    name: str
    source: str                 # "explorer" | "catalog" | "both"
    description: str = ""
    suggested_test: str = ""
    recipe: dict[str, Any] | None = None  # the SignalRecipe dict (bridged signals); replayable in-loop
    operationalized: bool = True   # became a per-case column on CONFIRM
    # host verdict (from the M2 firewall on CONFIRM) — authoritative
    effect: float | None = None
    ci: tuple[float, float] | None = None
    e_value: float | None = None
    p_value: float | None = None
    reject: bool | None = None
    underpowered: bool = False
    host_adjudicated: bool = False
    fdr_corrected: bool = False
    correction_method: str | None = None
    confirmed_on: str = ""      # "held_out" | "in_sample"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "suggested_test": self.suggested_test,
            "recipe": self.recipe,
            "operationalized": self.operationalized,
            "effect": self.effect,
            "ci": list(self.ci) if self.ci is not None else None,
            "e_value": self.e_value,
            "p_value": self.p_value,
            "reject": self.reject,
            "underpowered": self.underpowered,
            "host_adjudicated": self.host_adjudicated,
            "fdr_corrected": self.fdr_corrected,
            "correction_method": self.correction_method,
            "confirmed_on": self.confirmed_on,
        }


@dataclass
class FusedReport:
    """Output of :func:`run_fused_analysis` — explore (descriptive) + confirm (verdicts)."""

    question: str = ""
    ok: bool = False
    observations: list[str] = field(default_factory=list)
    visual_plan: list[dict[str, Any]] = field(default_factory=list)
    chart_readings: list[dict[str, Any]] = field(default_factory=list)
    dashboard_storyboard: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    candidate_signals: list[FusedSignal] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    recommended_confirmatory_tests: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    critique: list[str] = field(default_factory=list)
    adjudication: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    conclusion: str = ""
    explore_report: dict[str, Any] = field(default_factory=dict)
    confirm_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def confirmed_signal_names(self) -> list[str]:
        return [s.name for s in self.candidate_signals if s.reject]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ok": self.ok,
            "observations": self.observations,
            "visual_plan": self.visual_plan,
            "chart_readings": self.chart_readings,
            "dashboard_storyboard": self.dashboard_storyboard,
            "claims": self.claims,
            "candidate_signals": [s.to_dict() for s in self.candidate_signals],
            "charts": self.charts,
            "recommended_confirmatory_tests": self.recommended_confirmatory_tests,
            "caveats": self.caveats,
            "critique": self.critique,
            "adjudication": self.adjudication,
            "split": self.split,
            "conclusion": self.conclusion,
        }


# ---------------------------------------------------------------------------
# Deterministic stratified EXPLORE / CONFIRM split (records level)
# ---------------------------------------------------------------------------

def _group_key(row: Any, group_col: str, index: int) -> Any:
    """A row's group, or a key unique to the row when the group is missing."""
    value = row.get(group_col) if isinstance(row, dict) else getattr(row, group_col, None)
    if value is None or value == "":
        return ("__row__", index)
    return ("v", str(value))


def _split_by_group(
    rows: list[dict[str, Any]],
    *,
    n_confirm: int,
    seed: int,
    label_col: str,
    group_col: str,
) -> "tuple[list[dict], list[dict]]":
    """Split so that every group lies wholly on one side.

    Groups are stratified by their most common label and, within a stratum,
    drawn in a seeded random order until the stratum's share of the confirm
    rows is reached. Group sizes differ, so the confirm share is approximate:
    the draw stops at whichever group count lands closest to the target.
    """
    import random
    from collections import Counter

    members: dict[Any, list[int]] = {}
    for i, row in enumerate(rows):
        members.setdefault(_group_key(row, group_col, i), []).append(i)

    strata: dict[str, list[Any]] = {}
    distinct = {str(r.get(label_col)) for r in rows}
    if len(distinct) > 10:
        # A many-valued outcome (a score) has no "most common label" worth
        # stratifying on. Stratify groups by the quartile of their mean when
        # the outcome is numeric, and not at all otherwise.
        means: dict[Any, float] = {}
        for key, idxs in members.items():
            nums = [
                float(v) for v in (rows[i].get(label_col) for i in idxs)
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            if nums:
                means[key] = sum(nums) / len(nums)
        ranked = sorted(means, key=lambda k: (means[k], str(k)))
        quartile = {k: f"q{min(3, 4 * pos // max(1, len(ranked)))}"
                    for pos, k in enumerate(ranked)}
        for key in members:
            strata.setdefault(quartile.get(key, "unscored"), []).append(key)
    else:
        for key, idxs in members.items():
            labels = Counter(str(rows[i].get(label_col)) for i in idxs)
            # Ties resolve to the label that sorts first, so the split is deterministic.
            top = min(labels, key=lambda lab: (-labels[lab], lab))
            strata.setdefault(top, []).append(key)

    rng = random.Random(seed)
    frac = n_confirm / len(rows)
    confirm_idx: set[int] = set()
    for label in sorted(strata):
        keys = strata[label]
        rng.shuffle(keys)
        target = frac * sum(len(members[k]) for k in keys)
        taken = 0
        for key in keys:
            size = len(members[key])
            # Take the group only if that brings the stratum closer to its target.
            if abs(taken + size - target) > abs(taken - target):
                continue
            confirm_idx.update(members[key])
            taken += size
    explore = [r for i, r in enumerate(rows) if i not in confirm_idx]
    confirm = [r for i, r in enumerate(rows) if i in confirm_idx]
    return explore, confirm


def _split_records(
    records: list[dict[str, Any]],
    *,
    frac: float,
    seed: int,
    label_col: str,
    group_col: str | None = None,
) -> "tuple[list[dict], list[dict] | None]":
    """Deterministic, label-stratified ``(explore, confirm)`` split.

    Returns ``(explore, confirm)``. Like the in-loop split, returns
    ``(records, None)`` when ``frac <= 0`` or the batch is too small to hold out —
    a no-op, so the caller falls back to in-sample confirmation with a caveat.

    With *group_col*, rows that share a value of that column are kept on one
    side. Use it when rows are not independent — several rows cut from the
    same source item, the same prompt answered by several models — because a
    row-level split would otherwise put near-copies of an explore row in the
    confirm half and the confirmation would not be held out in any useful sense.
    """
    from evalrx.stats.subset_sampling import stratified_subset

    rows = list(records)
    if frac <= 0.0 or len(rows) < 4:
        return rows, None
    n_confirm = round(len(rows) * frac)
    if n_confirm <= 0 or n_confirm >= len(rows):
        return rows, None

    if group_col:
        explore, confirm = _split_by_group(
            rows, n_confirm=n_confirm, seed=seed, label_col=label_col,
            group_col=group_col,
        )
        if not explore or not confirm:
            return rows, None
        return explore, confirm

    def _key(row: dict[str, Any]) -> Any:
        return str(row.get(label_col))

    confirm = stratified_subset(rows, _key, n_confirm, seed=seed)
    confirm_ids = {id(r) for r in confirm}
    explore = [r for r in rows if id(r) not in confirm_ids]
    if not explore or not confirm:
        return rows, None
    return explore, confirm


def group_overlap(
    explore: list[dict[str, Any]],
    confirm: list[dict[str, Any]],
    group_col: str,
) -> dict[str, Any]:
    """How many confirm rows share a *group_col* value with an explore row.

    A share near zero means the two halves are separated at the group level;
    a large share means the confirm half repeats groups the explorer has seen.
    Rows with a missing group value count as not shared.
    """
    seen = {
        str(r.get(group_col)) for r in explore
        if r.get(group_col) not in (None, "")
    }
    present = [r for r in confirm if r.get(group_col) not in (None, "")]
    shared = sum(1 for r in present if str(r.get(group_col)) in seen)
    return {
        "group_col": group_col,
        "n_confirm_rows": len(confirm),
        "n_confirm_rows_in_explore_groups": shared,
        "share": round(shared / len(confirm), 4) if confirm else 0.0,
        "n_explore_groups": len(seen),
        "n_confirm_groups": len({str(r.get(group_col)) for r in present}),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_fused_analysis(
    records: list[dict[str, Any]],
    *,
    question: str = "What distinguishes failures from passes?",
    explorer: Any,
    stats_agent: Any | None = None,
    confirm_split: float = 0.3,
    seed: int = 0,
    id_col: str = "case_id",
    label_col: str = "label",
    group_col: str | None = None,
) -> FusedReport:
    """Run discovery on EXPLORE, confirmation on the held-out CONFIRM split.

    Args:
        records:        list of row dicts (id/label/signal columns).
        explorer:       an object with ``explore_records(rows, question=...) ->
                        ExploratoryAnalysisReport`` (e.g. :class:`ExploratoryAnalysisAgent`).
        stats_agent:    optional :class:`StatsAnalysisAgent`; one is built with a
                        high signal cap when omitted so the whole family is tested.
        confirm_split:  fraction held out for confirmation (0 = in-sample fallback).
        group_col:      optional column naming the unit rows were cut from; rows
                        that share a value stay on one side of the split.
    """
    rows = list(records or [])
    explore_rows, confirm_rows = _split_records(
        rows, frac=confirm_split, seed=seed, label_col=label_col,
        group_col=group_col,
    )
    held_out = confirm_rows is not None
    confirm_label = "held_out" if held_out else "in_sample"
    if confirm_rows is None:
        confirm_rows = explore_rows  # in-sample fallback (small batch)

    report = FusedReport(question=question)
    report.split = {
        "mode": confirm_label,
        "n_total": len(rows),
        "n_explore": len(explore_rows),
        "n_confirm": len(confirm_rows),
        "frac": confirm_split,
        "seed": seed,
    }
    if group_col:
        report.split["group_col"] = group_col
        if held_out:
            report.split["group_overlap"] = group_overlap(
                explore_rows, confirm_rows, group_col
            )

    # ── 1. Discovery source A: the LAMBDA explorer (EXPLORE only) ──
    explore_report = explorer.explore_records(explore_rows, question=question)
    report.ok = bool(getattr(explore_report, "ok", False))
    report.observations = list(getattr(explore_report, "observations", []) or [])
    report.visual_plan = list(getattr(explore_report, "visual_plan", []) or [])
    report.chart_readings = list(getattr(explore_report, "chart_readings", []) or [])
    report.dashboard_storyboard = list(getattr(explore_report, "dashboard_storyboard", []) or [])
    report.claims = list(getattr(explore_report, "claims", []) or [])
    report.charts = list(getattr(explore_report, "charts", []) or [])
    report.caveats = list(getattr(explore_report, "caveats", []) or [])
    report.critique = list(getattr(explore_report, "critique", []) or [])
    report.explore_report = (
        explore_report.to_dict() if hasattr(explore_report, "to_dict") else {}
    )
    explorer_candidates = list(getattr(explore_report, "candidate_signals", []) or [])
    explorer_named = {c.name: c for c in explorer_candidates if getattr(c, "name", "")}

    # ── 2. Discovery source B: catalog columns present on EXPLORE ──
    explore_inp = build_stats_input_from_records(
        explore_rows, id_col=id_col, label_col=label_col
    )
    catalog_names = set(explore_inp.per_case)

    # ── 3. Operationalization bridge: compile explorer recipes on CONFIRM ──
    bridged: dict[str, dict[str, float]] = {}
    recipe_by_name: dict[str, dict[str, Any]] = {}  # recipe name -> recipe dict (for replay)
    bridge_failed: list[Any] = []
    for c in explorer_candidates:
        recipe_data = getattr(c, "recipe", None)
        if not isinstance(recipe_data, dict):
            continue
        recipe = SignalRecipe.from_dict(recipe_data)
        if not recipe.name:
            recipe.name = c.name
        try:
            values = compile_recipe(recipe, confirm_rows, id_col=id_col)
        except (RecipeError, NotImplementedError):
            values = {}
        if values:
            bridged[recipe.name] = values
            recipe_by_name[recipe.name] = {**recipe_data, "name": recipe.name}
        else:
            bridge_failed.append(c)

    # ── 4. Confirmation: build confirm StatsInput, inject bridged signals, run M2 ──
    confirm_inp = build_stats_input_from_records(
        confirm_rows, id_col=id_col, label_col=label_col
    )
    catalog_confirm = set(confirm_inp.per_case)  # real catalog columns on CONFIRM
    bridged_keys: set[str] = set()               # final per_case keys for bridged signals
    bridged_origin: dict[str, str] = {}          # final key -> originating recipe name
    collisions: list[str] = []
    for name, values in bridged.items():
        key = name
        if key in catalog_confirm:
            # A bridged recipe must NEVER silently overwrite a real catalog column —
            # that would test a DIFFERENT estimand under the same name. Namespace it.
            key = f"bridged.{name}"
            while key in confirm_inp.per_case:
                key = f"_{key}"
            collisions.append(name)
        confirm_inp.per_case[key] = values
        bridged_keys.add(key)
        bridged_origin[key] = name
    if collisions:
        report.caveats.append(
            "bridged recipe name(s) collided with catalog column(s) and were "
            f"namespaced to avoid overwriting a real signal: {sorted(collisions)}"
        )

    # The deferred "leak-1" check: a bridged recipe (or catalog column) whose
    # value RECONSTRUCTS the FAIL label is the label in disguise — it would
    # trivially "confirm" with effect→1 and pollute confirmed_recipes.json. Route
    # such columns to the sanity lane so they never enter the confirmatory family
    # or get reported as discriminators; record a caveat for the audit trail.
    # (build_stats_input_from_records already isolated raw leaks; this re-runs
    # over the just-injected bridged signals.)
    from evalrx.analysis.stats_tools import isolate_label_leaks

    leaked = isolate_label_leaks(confirm_inp)
    for key in leaked:
        bridged_keys.discard(key)
    if leaked:
        report.caveats.append(
            "label-reconstructing signal(s) held out of the confirmatory family "
            "(sanity/plumbing, not discriminators): "
            + ", ".join(f"{k} [{v}]" for k, v in sorted(leaked.items()))
        )

    if stats_agent is None:
        from evalrx.analysis.stats_agent import StatsAnalysisAgent

        stats_agent = StatsAnalysisAgent(
            max_signal_tools=max(8, len(confirm_inp.per_case))
        )

    confirm_report = stats_agent.analyze_input(confirm_inp)
    report.confirm_stats = (
        confirm_report.to_dict() if hasattr(confirm_report, "to_dict") else {}
    )
    report.conclusion = getattr(confirm_report, "conclusion", "")
    corrected = getattr(confirm_report, "corrected_rejections", {}) or {}
    rejected_result_keys = set(corrected.get("rejected_result_keys", []))

    # ── 5. Assemble per-signal verdicts (provenance-tagged) ──
    verdict_by_signal = _verdicts_by_signal(getattr(confirm_report, "stats_results", []))
    for name in confirm_inp.per_case:
        result = verdict_by_signal.get(name)
        origin = bridged_origin.get(name, name)  # recipe name for a (possibly renamed) bridged key
        report.candidate_signals.append(
            _fused_signal(
                name=name,
                source=_source_of(name, catalog_names, bridged_keys, explorer_named),
                description=_describe(name, origin, explorer_named, bridged_keys),
                suggested_test=_suggested_test(origin, explorer_named),
                recipe=recipe_by_name.get(origin),
                result=result,
                rejected_result_keys=rejected_result_keys,
                confirmed_on=confirm_label,
            )
        )

    # ── 6. Graceful degradation channel: nothing silently dropped ──
    rec = list(getattr(explore_report, "recommended_confirmatory_tests", []) or [])
    for c in bridge_failed:
        rec.append(
            f"{c.name}: {getattr(c, 'rationale', '')} "
            "(recipe could not be operationalized on the confirm split)"
        )
    for c in explorer_candidates:
        if (
            not getattr(c, "recipe", None)
            and c.name not in catalog_names
            and c.name not in bridged
        ):
            rec.append(
                f"{c.name}: {getattr(c, 'rationale', '')} "
                "(descriptive only — no host-testable recipe)"
            )
    for fs in report.candidate_signals:
        if fs.underpowered:
            rec.append(f"{fs.name}: underpowered on the confirm split")
    report.recommended_confirmatory_tests = rec

    # ── 7. Family metadata + honest caveats ──
    report.adjudication = {
        "method": corrected.get("method", "e-BH"),
        "alpha": corrected.get("alpha"),
        "n_in_family": corrected.get("n_tested", 0),
        "rejected_result_keys": sorted(rejected_result_keys),
        "families": corrected.get("families", {}),
        "n_signals_tested": len(confirm_inp.per_case),
        "n_signals_rejected": sum(1 for s in report.candidate_signals if s.reject),
        "split": confirm_label,
    }
    if not held_out:
        report.caveats.append(
            "CONFIRM split is IN-SAMPLE (batch too small to hold out); verdicts are "
            "not double-dip-protected — enlarge the batch or lower confirm_split"
        )
    if any(s.host_adjudicated and s.reject and not s.fdr_corrected
           for s in report.candidate_signals):
        report.caveats.append(
            "some rejected signal verdicts were not FDR-corrected; treat them as "
            "descriptive until rerun with a correction-capable test"
        )
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verdicts_by_signal(stats_results: list[Any]) -> dict[str, Any]:
    """Map ``signal name -> StatsToolResult`` from the per-case catalog tools.

    Prefers ``signal_label_assoc`` (carries a reject) over ``rank_corr``.
    """
    out: dict[str, Any] = {}
    for r in stats_results or []:
        if not getattr(r, "ok", False):
            continue
        signal = (getattr(r, "config", {}) or {}).get("signal")
        if not signal:
            continue
        if getattr(r, "tool", "") == "signal_label_assoc":
            out[signal] = r
        elif signal not in out:
            out[signal] = r
    return out


def _fused_signal(
    *,
    name: str,
    source: str,
    description: str,
    suggested_test: str,
    recipe: dict[str, Any] | None,
    result: Any,
    rejected_result_keys: set[str],
    confirmed_on: str,
) -> FusedSignal:
    fs = FusedSignal(
        name=name,
        source=source,
        description=description,
        suggested_test=suggested_test,
        recipe=recipe,
        confirmed_on=confirmed_on,
    )
    if result is None:
        # Still a per-case column, but no usable test ran on CONFIRM (e.g. a
        # degenerate one-sided split). Left host_adjudicated=False, reject=None.
        return fs
    fs.host_adjudicated = True
    fs.effect = getattr(result, "effect", None)
    fs.ci = getattr(result, "ci", None)
    fs.e_value = getattr(result, "e_value", None)
    fs.p_value = getattr(result, "p_value", None)
    fs.underpowered = bool(getattr(result, "underpowered", False))
    fs.reject = bool(getattr(result, "reject", False))
    fs.correction_method = getattr(result, "correction_method", None)
    fs.fdr_corrected = bool(getattr(result, "fdr_corrected", False)) or (
        getattr(result, "analysis_key", None) in rejected_result_keys
    )
    if fs.fdr_corrected:
        fs.reject = True
    return fs


def _source_of(
    name: str,
    catalog_names: set[str],
    bridged_keys: set[str],
    explorer_named: dict[str, Any],
) -> str:
    in_bridged = name in bridged_keys
    in_catalog = name in catalog_names
    pointed = name in explorer_named
    # A bridged key is namespaced away from catalog columns, so a bridged signal is
    # purely explorer-sourced; "both" is reserved for the explorer naming a real column.
    if in_bridged:
        return "both" if in_catalog else "explorer"
    if in_catalog and pointed:
        return "both"
    return "catalog"


def _describe(
    name: str,
    origin: str,
    explorer_named: dict[str, Any],
    bridged_keys: set[str],
) -> str:
    c = explorer_named.get(origin) or explorer_named.get(name)
    rationale = getattr(c, "rationale", "") if c is not None else ""
    if rationale:
        return rationale
    if name in bridged_keys:
        return "bridged explorer signal"
    return "existing per-case column"


def _suggested_test(name: str, explorer_named: dict[str, Any]) -> str:
    c = explorer_named.get(name)
    return getattr(c, "suggested_test", "") if c is not None else ""
