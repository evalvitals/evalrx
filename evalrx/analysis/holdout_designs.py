"""Comparison designs for re-testing a frozen recipe on held-out rows.

:func:`~evalrx.analysis.holdout.holdout_confirm` used to know one comparison:
the outcome rate among rows a 0/1 recipe flags against the rate among the rows
it does not. Many findings are not of that shape, and forcing them into it
either loses the finding or tests a different one. This module adds the shapes
that came up in practice, all estimated the same way:

- ``two_group``     — flagged against unflagged rows (the original).
- ``stratified``    — the same difference taken within strata and averaged
                      with the strata's sizes as weights, so a gap that is only
                      a difference in mix between the two groups shrinks.
- ``did``           — difference in differences: how much larger the recipe's
                      effect is where a second 0/1 condition holds.
- ``dose_response`` — a continuous recipe cut at fixed points chosen during
                      exploration; the estimate is the change in the outcome
                      per step up the bins.

Every design is a statistic of row weights, so one cluster bootstrap
(:func:`evalrx.stats.bootstrap.weighted_bootstrap`) gives each an interval and
a p-value, and rows that belong to one group are resampled together.

A recipe may also state what it expects (``expect``): that the effect is
positive, negative, or no larger than a bound. "No larger than a bound" is how
a control is written — a comparison that should come out small — and it holds
only when the whole interval lies inside the bound, not whenever the estimate
happens to be near zero.

Nothing here refits anything: the expression, the subgroup condition, the
strata and the cut points all come from the recipe as it was frozen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from evalrx.analysis.operationalize import (
    RECIPE_DESIGNS,
    RecipeError,
    SignalRecipe,
    evaluate_expr,
    parse_expectation,
)
from evalrx.stats.bootstrap import (
    bootstrap_p_value,
    mean_difference_statistic,
    weighted_bootstrap,
)
from evalrx.stats.exact import two_group_exact_p

#: Below this many independent units (groups, or rows when there are none) on
#: the smaller side, a bootstrap interval and p-value are not to be relied on.
MIN_UNITS_PER_SIDE = 10

#: At most this many strata / bins are written into a verdict's details.
_MAX_DETAIL_ROWS = 40


@dataclass
class DesignEstimate:
    """One recipe's held-out estimate, before any multiplicity correction."""

    design: str = ""
    ok: bool = False
    reason: str = ""                     # why it was skipped, when not ok
    effect: float | None = None
    ci: tuple[float, float] | None = None
    relation: str = "nonzero"            # the claim being tested
    value: float = 0.0
    p_value: float | None = None         # for the claim
    p_contrary: float | None = None      # for the opposite of a bounded claim
    n: int = 0
    n_clusters: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    # Bootstrap replicates and the rows used; kept for follow-up claims on the
    # same estimate and for picking representative rows. Never serialised.
    draws: Any = None
    rows_used: list[int] = field(default_factory=list)
    flag: list[float] = field(default_factory=list)

    def claim_text(self) -> str:
        if self.relation == "nonzero":
            return "effect is not zero"
        if self.relation == "abs<":
            return f"|effect| < {self.value:g}"
        return f"effect {self.relation} {self.value:g}"


def _as_flag(value: Any) -> float | None:
    """A recipe value as 0.0/1.0, or None when it is not a flag."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and round(float(value), 6) in (0.0, 1.0):
        return float(round(float(value)))
    return None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _stratified_statistic(
    y: np.ndarray, flag: np.ndarray, codes: np.ndarray, n_strata: int
) -> "Callable[[np.ndarray], float]":
    """Size-weighted average of the within-stratum difference of means.

    A stratum that loses one of its two groups in a replicate drops out of
    that replicate and the remaining weights are renormalised.
    """
    in_b, in_a = flag, 1.0 - flag
    y_b, y_a = y * in_b, y * in_a

    def statistic(w: np.ndarray) -> float:
        w_b = np.bincount(codes, weights=w * in_b, minlength=n_strata)
        w_a = np.bincount(codes, weights=w * in_a, minlength=n_strata)
        s_b = np.bincount(codes, weights=w * y_b, minlength=n_strata)
        s_a = np.bincount(codes, weights=w * y_a, minlength=n_strata)
        both = (w_b > 0) & (w_a > 0)
        if not both.any():
            return float("nan")
        diff = s_b[both] / w_b[both] - s_a[both] / w_a[both]
        size = w_b[both] + w_a[both]
        return float((diff * size).sum() / size.sum())

    return statistic


def _did_statistic(
    y: np.ndarray, flag: np.ndarray, modifier: np.ndarray
) -> "Callable[[np.ndarray], float]":
    """(flag effect where modifier is 1) minus (flag effect where it is 0)."""
    cells = [(flag * modifier), ((1 - flag) * modifier),
             (flag * (1 - modifier)), ((1 - flag) * (1 - modifier))]
    sums = [y * c for c in cells]

    def statistic(w: np.ndarray) -> float:
        means = []
        for cell, total in zip(cells, sums):
            weight = float(w @ cell)
            if weight <= 0.0:
                return float("nan")
            means.append(float(w @ total) / weight)
        return (means[0] - means[1]) - (means[2] - means[3])

    return statistic


def _slope_statistic(y: np.ndarray, x: np.ndarray) -> "Callable[[np.ndarray], float]":
    """Weighted least-squares slope of *y* on *x* (here: on the bin index)."""

    def statistic(w: np.ndarray) -> float:
        total = float(w.sum())
        if total <= 0.0:
            return float("nan")
        mx = float(w @ x) / total
        my = float(w @ y) / total
        var = float(w @ ((x - mx) ** 2))
        if var <= 0.0:
            return float("nan")
        return float(w @ ((x - mx) * (y - my))) / var

    return statistic


def _skip(design: str, reason: str) -> DesignEstimate:
    return DesignEstimate(design=design, ok=False, reason=reason)


def estimate_design(
    recipe: SignalRecipe,
    rows: "list[dict[str, Any]]",
    outcome: "list[float | None]",
    *,
    clusters: "list[Any] | None" = None,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    id_col: str = "case_id",
    reserved: "tuple[str, ...] | list[str]" = (),
) -> DesignEstimate:
    """Estimate one frozen recipe's effect on *rows* with a cluster bootstrap.

    *outcome* is one number per row (a 0/1 failure indicator or a numeric
    outcome), None where it is missing. *clusters* is one group id per row, or
    None when rows are independent. Raises :class:`RecipeError` for an invalid
    expression; returns a skipped estimate (``ok=False`` with a reason) when
    the recipe is valid but the design cannot be estimated on these rows.
    """
    design = (recipe.design or "").strip().lower()
    if design and design not in RECIPE_DESIGNS:
        return _skip(design, f"unknown design {design!r}; expected one of {list(RECIPE_DESIGNS)}")

    kwargs: dict[str, Any] = {"id_col": id_col, "reserved": reserved}
    x_raw = evaluate_expr(recipe.expr, rows, **kwargs)
    keep = [v is not None and y is not None for v, y in zip(x_raw, outcome)]
    n_where_out = 0
    if (recipe.where or "").strip():
        cond = evaluate_expr(recipe.where, rows, what="where", **kwargs)
        for i, c in enumerate(cond):
            if keep[i] and (c is None or not c):
                keep[i] = False
                n_where_out += 1

    flags = [_as_flag(v) for v in x_raw]
    numbers = [_as_number(v) for v in x_raw]
    idx = [i for i, k in enumerate(keep) if k and numbers[i] is not None]
    if not idx:
        return _skip(design or "two_group", "the recipe has no value on any held-out row")
    is_flag = all(flags[i] is not None for i in idx)

    if not design:
        if is_flag:
            design = "two_group"
        elif recipe.bins:
            design = "dose_response"
        else:
            return _skip(
                "two_group",
                "recipe is not a binary flag on held-out rows (no frozen threshold "
                "to reuse); give it fixed `bins` for a dose-response trend, or "
                "write the threshold into the expression",
            )
    if design != "dose_response" and not is_flag:
        return _skip(design, f"design {design!r} needs a 0/1 recipe; this one is continuous")

    details: dict[str, Any] = {}
    if n_where_out:
        details["n_excluded_by_where"] = n_where_out
    statistic: "Callable[[np.ndarray], float]"

    if design == "two_group":
        y = np.asarray([outcome[i] for i in idx], dtype=float)
        flag = np.asarray([flags[i] for i in idx], dtype=float)
        if flag.sum() == 0 or flag.sum() == len(flag):
            return _skip(design, "all held-out rows fall on one side of the recipe")
        statistic = mean_difference_statistic(y, flag)
        details.update(_two_group_details(y, flag))

    elif design == "stratified":
        if not recipe.strata:
            return _skip(design, "design 'stratified' needs `strata` expressions")
        labels = [evaluate_expr(e, rows, what="strata", **kwargs) for e in recipe.strata]
        idx = [i for i in idx if all(col[i] is not None for col in labels)]
        keys = [tuple(str(col[i]) for col in labels) for i in idx]
        flag_all = np.asarray([flags[i] for i in idx], dtype=float)
        # Only strata that hold both groups can contribute a difference.
        sides: dict[tuple, set[float]] = {}
        for key, f in zip(keys, flag_all):
            sides.setdefault(key, set()).add(float(f))
        usable = {k for k, v in sides.items() if len(v) == 2}
        sel = [j for j, k in enumerate(keys) if k in usable]
        if not sel:
            return _skip(design, "no stratum holds both flagged and unflagged rows")
        details["n_strata"] = len(sides)
        details["n_strata_used"] = len(usable)
        details["n_rows_in_one_sided_strata"] = len(idx) - len(sel)
        idx = [idx[j] for j in sel]
        keys = [keys[j] for j in sel]
        order = {k: c for c, k in enumerate(sorted(usable))}
        codes = np.asarray([order[k] for k in keys], dtype=np.int64)
        y = np.asarray([outcome[i] for i in idx], dtype=float)
        flag = np.asarray([flags[i] for i in idx], dtype=float)
        statistic = _stratified_statistic(y, flag, codes, len(order))
        details["unadjusted_difference"] = round(
            float(mean_difference_statistic(y, flag)(np.ones(len(y)))), 6)
        details["strata"] = _strata_details(y, flag, codes, sorted(usable), recipe.strata)

    elif design == "did":
        if not (recipe.modifier or "").strip():
            return _skip(design, "design 'did' needs a 0/1 `modifier` expression")
        mod_raw = evaluate_expr(recipe.modifier, rows, what="modifier", **kwargs)
        mods = [_as_flag(v) if v is not None else None for v in mod_raw]
        if any(mods[i] is None and mod_raw[i] is not None for i in idx):
            return _skip(design, "the `modifier` expression is not 0/1 on held-out rows")
        idx = [i for i in idx if mods[i] is not None]
        y = np.asarray([outcome[i] for i in idx], dtype=float)
        flag = np.asarray([flags[i] for i in idx], dtype=float)
        mod = np.asarray([mods[i] for i in idx], dtype=float)
        cells = {}
        for label, m in (("modifier_1", 1.0), ("modifier_0", 0.0)):
            for side, f in (("flagged", 1.0), ("unflagged", 0.0)):
                pick = (mod == m) & (flag == f)
                if not pick.any():
                    return _skip(design, f"no held-out row with {label} and {side}")
                cells[f"{label}_{side}"] = {"n": int(pick.sum()),
                                            "mean": round(float(y[pick].mean()), 6)}
        statistic = _did_statistic(y, flag, mod)
        details["cells"] = cells
        details["effect_where_modifier_1"] = round(
            cells["modifier_1_flagged"]["mean"] - cells["modifier_1_unflagged"]["mean"], 6)
        details["effect_where_modifier_0"] = round(
            cells["modifier_0_flagged"]["mean"] - cells["modifier_0_unflagged"]["mean"], 6)

    else:  # dose_response
        edges = sorted(set(recipe.bins))
        if not edges:
            return _skip(design, "design 'dose_response' needs fixed `bins` cut points")
        x = np.asarray([numbers[i] for i in idx], dtype=float)
        y = np.asarray([outcome[i] for i in idx], dtype=float)
        bin_index = np.searchsorted(np.asarray(edges), x, side="right").astype(float)
        occupied = sorted(set(bin_index.tolist()))
        if len(occupied) < 2:
            return _skip(design, "held-out rows fall into a single bin")
        statistic = _slope_statistic(y, bin_index)
        details["bins"] = [
            {"bin": int(b), "low": edges[int(b) - 1] if b >= 1 else None,
             "high": edges[int(b)] if b < len(edges) else None,
             "n": int((bin_index == b).sum()),
             "mean": round(float(y[bin_index == b].mean()), 6)}
            for b in occupied
        ][:_MAX_DETAIL_ROWS]
        details["effect_unit"] = "change in the outcome per step up the bins"
        flag = bin_index  # kept so representatives can be chosen per bin end

    cl = [clusters[i] for i in idx] if clusters is not None else None
    boot = weighted_bootstrap(statistic, len(idx), clusters=cl, n_boot=n_boot,
                              ci=1.0 - alpha, seed=seed)
    draws = boot["draws"]
    if draws.size < max(20, n_boot // 2):
        return _skip(design, "the estimate was undefined in most bootstrap replicates "
                             "(a group or cell is too small)")
    relation, value = parse_expectation(recipe.expect) or ("nonzero", 0.0)
    p_value = bootstrap_p_value(draws, relation, value)
    p_contrary = contrary_p_value(draws, relation, value)
    details["p_value_method"] = "percentile bootstrap"
    binary = bool(np.isin(y, (0.0, 1.0)).all())
    if design == "two_group" and cl is None and binary and (relation == "nonzero" or value == 0.0) and relation != "abs<":
        # Independent rows and a binary outcome: the exact test replaces the
        # bootstrap p-value, which degenerates on small, cleanly separated
        # samples. The interval stays the bootstrap one.
        counts = (int(y[flag == 0.0].sum()), int((flag == 0.0).sum()),
                  int(y[flag == 1.0].sum()), int((flag == 1.0).sum()))
        side = {"nonzero": "two-sided", ">": "greater", "<": "less"}[relation]
        p_value = two_group_exact_p(*counts, alternative=side)
        if relation != "nonzero":
            p_contrary = two_group_exact_p(
                *counts, alternative="less" if relation == ">" else "greater")
        details["p_value_method"] = "exact (Fisher)"
    units = _units_per_side(design, flag, cl)
    if units is not None:
        details["min_units_per_side"] = units
        if units < MIN_UNITS_PER_SIDE:
            details["small_sample_warning"] = (
                f"only {units} independent unit(s) on the smaller side; the interval "
                f"and p-value are unreliable below about {MIN_UNITS_PER_SIDE}"
            )
    estimate = DesignEstimate(
        design=design, ok=True, effect=float(boot["effect"]),
        ci=(float(boot["ci_low"]), float(boot["ci_high"])),
        relation=relation, value=value,
        p_value=p_value,
        p_contrary=p_contrary,
        n=len(idx), n_clusters=boot["n_clusters"] if clusters is not None else None,
        details=details, draws=draws, rows_used=idx,
        flag=[float(v) for v in np.asarray(flag).tolist()],
    )
    if boot["n_undefined"]:
        estimate.details["n_undefined_replicates"] = int(boot["n_undefined"])
    return estimate


def _units_per_side(design: str, flag: np.ndarray, clusters: "list[Any] | None") -> int | None:
    """Independent units (groups, else rows) on the smaller side of a comparison.

    For a dose-response design the two sides are the lowest and highest bin.
    Returns None for designs whose sides are not two sets of rows.
    """
    if design not in ("two_group", "stratified", "dose_response"):
        return None
    low, high = float(flag.min()), float(flag.max())
    counts = []
    for side in (low, high):
        members = np.flatnonzero(flag == side)
        if clusters is None:
            counts.append(len(members))
        else:
            keys = {("__row__", int(j)) if clusters[j] in (None, "") else ("v", clusters[j])
                    for j in members}
            counts.append(len(keys))
    return int(min(counts))


def contrary_p_value(draws: Any, relation: str, value: float) -> float | None:
    """p-value for the OPPOSITE of a bounded claim, or None when there is none.

    "The effect is not zero" has no testable opposite without a bound, so it
    can only hold or stay open. A bounded claim can also be contradicted: the
    data can show the effect lies on the other side of the bound.
    """
    d = np.asarray(draws, dtype=float)
    if d.size == 0 or relation == "nonzero":
        return None
    if relation == ">":
        return bootstrap_p_value(d, "<", value)
    if relation == "<":
        return bootstrap_p_value(d, ">", value)
    # "abs<": the opposite is |effect| > value.
    return float((1 + int(np.sum(np.abs(d) <= abs(value)))) / (d.size + 1))


def claim_status(holds: bool, p_contrary: float | None, alpha: float) -> str:
    """``holds`` / ``contradicted`` / ``inconclusive`` for one claim."""
    if holds:
        return "holds"
    if p_contrary is not None and p_contrary <= alpha:
        return "contradicted"
    return "inconclusive"


def _two_group_details(y: np.ndarray, flag: np.ndarray) -> dict[str, Any]:
    flagged, unflagged = y[flag == 1.0], y[flag == 0.0]
    return {
        "n_flagged": int(len(flagged)),
        "n_unflagged": int(len(unflagged)),
        "mean_flagged": round(float(flagged.mean()), 6),
        "mean_unflagged": round(float(unflagged.mean()), 6),
    }


def _strata_details(
    y: np.ndarray, flag: np.ndarray, codes: np.ndarray,
    keys: "list[tuple]", exprs: "list[str]",
) -> list[dict[str, Any]]:
    out = []
    for code, key in enumerate(keys[:_MAX_DETAIL_ROWS]):
        pick = codes == code
        a, b = y[pick & (flag == 0.0)], y[pick & (flag == 1.0)]
        out.append({
            "stratum": dict(zip(exprs, key)),
            "n_flagged": int(len(b)), "n_unflagged": int(len(a)),
            "difference": round(float(b.mean() - a.mean()), 6),
        })
    return out


def representative_rows(
    estimate: DesignEstimate,
    rows: "list[dict[str, Any]]",
    outcome: "list[float | None]",
    columns: "list[str]",
    *,
    k: int = 3,
) -> dict[str, list[int]]:
    """Typical rows on each side of a comparison, as indices into *rows*.

    "Typical" means close to the side's median on the outcome and on the
    numeric *columns* the recipe reads, each distance scaled by the side's
    spread. Rows at the median show what the comparison is mostly made of;
    the most extreme rows would show the cases least like the rest. For a
    dose-response design the two sides are the lowest and the highest bin.
    """
    if not estimate.ok or not estimate.rows_used:
        return {}
    flag = np.asarray(estimate.flag, dtype=float)
    low, high = float(flag.min()), float(flag.max())
    if low == high:
        return {}
    names = ("unflagged", "flagged") if estimate.design != "dose_response" else ("lowest_bin", "highest_bin")
    out: dict[str, list[int]] = {}
    for name, side in zip(names, (low, high)):
        members = [estimate.rows_used[j] for j in np.flatnonzero(flag == side)]
        features: list[list[float]] = [[float(outcome[i] or 0.0) for i in members]]
        for col in columns:
            numeric = [
                float(v) for v in (rows[i].get(col) for i in members)
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            # A column is used only when every row of the side has a number in it.
            if len(numeric) == len(members):
                features.append(numeric)
        score = np.zeros(len(members))
        for feature in features:
            arr = np.asarray(feature)
            centre = float(np.median(arr))
            spread = float(np.median(np.abs(arr - centre))) or float(arr.std()) or 1.0
            score += np.abs(arr - centre) / spread
        order = np.argsort(score, kind="mergesort")[:k]
        out[name] = [members[int(j)] for j in order]
    return out


__all__ = [
    "DesignEstimate",
    "RecipeError",
    "claim_status",
    "contrary_p_value",
    "estimate_design",
    "representative_rows",
]
