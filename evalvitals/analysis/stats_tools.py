"""M2 — statistical tool catalog: thin wrappers around :mod:`evalvitals.stats`.

This module turns the rigorous-but-low-level :mod:`evalvitals.stats` building
blocks (McNemar + e-value, clustered bootstrap, Friedman/Nemenyi, e-BH) into a
small **catalog of named tools** that
:class:`~evalvitals.analysis.stats_agent.StatsAnalysisAgent` can
*select* and call with a config dict — the "select stats tools" half of the M2
plan (Plan A, 2026-06-05).

The flow is:

1. :func:`build_stats_input` normalises ``{analyzer: Result}`` + a labeled
   ``CaseBatch`` into a single :class:`StatsInput` (per-case signals, labels,
   scalar metrics, optional strategy groups).
2. The agent picks tool names from :data:`STATS_TOOL_CATALOG` (LLM-guided) or
   falls back to :func:`default_plan` (deterministic).
3. Each ``(tool, config)`` runs via :func:`run_stats_tool`, returning a uniform
   :class:`StatsToolResult` (effect, CI, e-value, reject, underpowered).
4. :func:`fdr_correct` applies e-BH across all tools that produced an e-value so
   multiple-metric testing is FDR-controlled, not naive.
5. :func:`plot_effects` draws an optional forest plot of effect ± CI.

No tool ever returns a bare p-value: every verdict carries an effect size and a
corrected reject decision, inherited from :func:`evalvitals.stats.compare`.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from evalvitals.core.case import Label
from evalvitals.stats import (
    compare,
    compare_multiple,
    e_value_test,
    kendall_tau,
)
from evalvitals.stats.multiplicity import correct_results

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.result import Result

logger = logging.getLogger(__name__)

# Keys in a per_case finding entry that identify the case, not a signal.
_ID_KEYS = ("sample_id", "case_id", "id")


# ---------------------------------------------------------------------------
# Normalised input + uniform result
# ---------------------------------------------------------------------------

@dataclass
class StatsInput:
    """Normalised view of M1 results + labels, ready for statistical tests.

    Attributes:
        labels:    ``{case_id -> is_fail}`` for PASS/FAIL cases (UNKNOWN dropped).
        per_case:  ``{"analyzer.metric" -> {case_id -> value}}`` — per-case
                   numeric/boolean signals harvested from ``findings["per_case"]``.
        scalars:   ``{"analyzer.metric" -> value}`` — aggregate numeric findings.
        groups:    Optional ``{strategy -> {case_id -> success}}`` for
                   paired/omnibus strategy comparisons (from ``findings["by_strategy"]``).
    """

    labels: dict[str, bool] = field(default_factory=dict)
    per_case: dict[str, dict[str, float]] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    groups: dict[str, dict[str, float]] | None = None
    # Per-case signals that near-perfectly RECONSTRUCT the FAIL label (a probe
    # output equal to the label, a label-recomputing recipe, …). Moved here by
    # :func:`isolate_label_leaks` so they never enter the tested family / e-BH
    # multiplicity / candidate charts / hypothesis seeding — but are KEPT as a
    # pipeline self-check (the plumbing audit). ``{name -> {case_id -> value}}``.
    sanity: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-case VECTOR signals (e.g. a full attention map per case), harvested from
    # ``Result.artifacts["per_case_maps"]``. Consumed by the tensor-level
    # ``attention_decoding`` omnibus, NOT the scalar tools.
    # ``{name -> {case_id -> np.ndarray}}``.
    per_case_vectors: dict[str, dict[str, "Any"]] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        results: "dict[str, Result]",
        data: "CaseBatch | None" = None,
    ) -> "StatsInput":
        """Build statistical input from EvalVitals analyzer results."""
        return build_stats_input(results, data)

    @classmethod
    def from_records(
        cls,
        records: "Any",
        *,
        id_col: str = "case_id",
        label_col: str = "label",
        signal_cols: "list[str] | tuple[str, ...] | None" = None,
        scalar_cols: "list[str] | tuple[str, ...] | None" = None,
        signal_prefix: str = "",
    ) -> "StatsInput":
        """Build statistical input from plain row dictionaries.

        ``label_col`` accepts booleans, ``Label`` values, common strings
        (``"pass"``, ``"fail"``, ``"success"``, ``"error"``), or 0/1 values
        where 1 means FAIL.  Signal and scalar columns must be numeric/bool.
        """
        return build_stats_input_from_records(
            records,
            id_col=id_col,
            label_col=label_col,
            signal_cols=signal_cols,
            scalar_cols=scalar_cols,
            signal_prefix=signal_prefix,
        )


@dataclass
class StatsToolResult:
    """Uniform output of one statistical tool.

    Attributes:
        tool:         Catalog name of the tool that produced this.
        config:       The (resolved) config the tool ran with.
        ok:           ``True`` when the test ran; ``False`` when skipped/failed.
        summary:      Human-readable one-liner.
        effect:       Effect size (e.g. fail-rate difference, τ) when applicable.
        ci:           Confidence interval on the effect, when applicable.
        reject:       Corrected reject decision (e-value / CI), never a bare p.
        e_value:      Anytime-valid e-value, when the test produces one.
        p_value:      Raw p (diagnostic only — never the decision basis).
        underpowered: Inconclusive *and* CI too wide to rule out a real effect.
        details:      Tool-specific extras (group sizes, ranks, contingency …).
        figure_path:  Path to a per-tool figure, if any.
        error:        Reason string when ``ok`` is ``False``.
    """

    tool: str
    config: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    summary: str = ""
    effect: float | None = None
    ci: tuple[float, float] | None = None
    reject: bool | None = None
    e_value: float | None = None
    p_value: float | None = None
    underpowered: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    figure_path: str | None = None
    error: str | None = None
    # Generic-M2 metadata. Existing callers can ignore these fields; they let
    # planners/controllers identify one result inside a larger tested family.
    analysis_key: str | None = None
    correction_family: str | None = "auto"  # "e_bh" | "bh" | "auto" | None
    correction_method: str | None = None
    fdr_corrected: bool = False
    raw_reject: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "config": self.config,
            "ok": self.ok,
            "summary": self.summary,
            "effect": self.effect,
            "ci": list(self.ci) if self.ci is not None else None,
            "reject": self.reject,
            "e_value": self.e_value,
            "p_value": self.p_value,
            "underpowered": self.underpowered,
            "details": self.details,
            "figure_path": self.figure_path,
            "error": self.error,
            "analysis_key": self.analysis_key,
            "correction_family": self.correction_family,
            "correction_method": self.correction_method,
            "fdr_corrected": self.fdr_corrected,
            "raw_reject": self.raw_reject,
        }


EvidenceResult = StatsToolResult


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def _entry_id(entry: dict) -> str:
    for k in _ID_KEYS:
        v = entry.get(k)
        if v:
            return str(v)
    return ""


def build_stats_input(
    results: "dict[str, Result]",
    data: "CaseBatch | None" = None,
) -> StatsInput:
    """Normalise ``{analyzer: Result}`` + optional labeled *data* into a :class:`StatsInput`."""
    labels: dict[str, bool] = {}
    if data is not None:
        for c in data:
            lab = getattr(c, "label", None)
            if lab is None or lab == Label.UNKNOWN:
                continue
            is_fail = lab == Label.FAIL
            labels[c.id] = is_fail
            traj = getattr(c, "trajectory", None)
            sid = getattr(traj, "sample_id", "") if traj is not None else ""
            if sid:
                labels[sid] = is_fail

    per_case: dict[str, dict[str, float]] = {}
    scalars: dict[str, float] = {}
    groups: dict[str, dict[str, float]] = {}
    per_case_vectors: dict[str, dict[str, Any]] = {}

    for aname, res in results.items():
        findings = res.findings or {}
        for k, v in findings.items():
            if isinstance(v, (int, float, bool)):
                scalars[f"{aname}.{k}"] = float(v)
        for entry in findings.get("per_case", []) or []:
            if not isinstance(entry, dict):
                continue
            cid = _entry_id(entry)
            if not cid:
                continue
            for k, v in entry.items():
                if k in _ID_KEYS:
                    continue
                if isinstance(v, (int, float, bool)):
                    per_case.setdefault(f"{aname}.{k}", {})[cid] = float(v)
        by_strategy = findings.get("by_strategy")
        if isinstance(by_strategy, dict):
            for sname, vec in by_strategy.items():
                slot = groups.setdefault(str(sname), {})
                if isinstance(vec, dict):
                    for cid, val in vec.items():
                        if isinstance(val, (int, float, bool)):
                            slot[str(cid)] = float(bool(val))

        # Per-case VECTOR signals (full attention maps) for the tensor-level
        # omnibus — kept in artifacts (heavy), so read them off the Result here.
        maps = (getattr(res, "artifacts", None) or {}).get("per_case_maps")
        if isinstance(maps, dict) and maps:
            col = {str(cid): m for cid, m in maps.items() if m is not None}
            if col:
                per_case_vectors[f"{aname}.map"] = col

    out = StatsInput(
        labels=labels,
        per_case=per_case,
        scalars=scalars,
        groups=groups or None,
        per_case_vectors=per_case_vectors,
    )
    # Route label-reconstructing signals to the sanity lane so they never enter
    # the tested family / e-BH multiplicity / candidate charts.
    isolate_label_leaks(out)
    return out


def build_stats_input_from_records(
    records: "Any",
    *,
    id_col: str = "case_id",
    label_col: str = "label",
    signal_cols: "list[str] | tuple[str, ...] | None" = None,
    scalar_cols: "list[str] | tuple[str, ...] | None" = None,
    signal_prefix: str = "",
) -> StatsInput:
    """Normalise plain records into :class:`StatsInput`.

    This is the standalone on-ramp for users who have a table of cases and
    signals rather than EvalVitals ``Result`` objects.
    """
    labels: dict[str, bool] = {}
    per_case: dict[str, dict[str, float]] = {}
    scalars: dict[str, float] = {}

    rows = list(records or [])
    if signal_cols is None and rows:
        excluded = {id_col, label_col, *(scalar_cols or ())}
        signal_cols = [
            str(k) for k, v in _row_items(rows[0])
            if k not in excluded and isinstance(v, (int, float, bool))
        ]
    signal_cols = tuple(signal_cols or ())
    scalar_cols = tuple(scalar_cols or ())

    for i, row in enumerate(rows):
        cid = _row_get(row, id_col, None)
        if cid in (None, ""):
            cid = str(i)
        cid = str(cid)

        label = _parse_label(_row_get(row, label_col, None))
        if label is not None:
            labels[cid] = label

        for col in signal_cols:
            val = _row_get(row, col, None)
            if isinstance(val, (int, float, bool)):
                key = f"{signal_prefix}.{col}" if signal_prefix else str(col)
                per_case.setdefault(key, {})[cid] = float(val)

        for col in scalar_cols:
            val = _row_get(row, col, None)
            if isinstance(val, (int, float, bool)):
                scalars[str(col)] = float(val)

    out = StatsInput(labels=labels, per_case=per_case, scalars=scalars)
    isolate_label_leaks(out)
    return out


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _row_items(row: Any) -> list[tuple[str, Any]]:
    if isinstance(row, dict):
        return list(row.items())
    if hasattr(row, "_asdict"):
        return list(row._asdict().items())
    if hasattr(row, "__dict__"):
        return list(vars(row).items())
    return []


def _parse_label(value: Any) -> bool | None:
    if value is None or value == Label.UNKNOWN:
        return None
    if value == Label.FAIL:
        return True
    if value == Label.PASS:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"fail", "failed", "failure", "false", "incorrect", "error", "bad", "1"}:
        return True
    if text in {"pass", "passed", "success", "true", "correct", "ok", "0"}:
        return False
    return None


def _is_binary(values: "Any") -> bool:
    return all(float(v) in (0.0, 1.0) for v in values)


def _binarize(
    signal_map: dict[str, float],
    mode: str = "median",
    threshold: float | None = None,
) -> dict[str, bool]:
    """Binarise a continuous per-case signal (median split by default)."""
    vals = list(signal_map.values())
    if not vals or _is_binary(vals):
        return {cid: bool(v) for cid, v in signal_map.items()}
    if threshold is None:
        if mode == "median":
            s = sorted(vals)
            n = len(s)
            threshold = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
        else:
            threshold = 0.0
    return {cid: (v > threshold) for cid, v in signal_map.items()}


# ---------------------------------------------------------------------------
# Label-leak detection (the deferred "leak-1" check from operationalize.py): a
# per-case signal that RECONSTRUCTS the FAIL label carries no diagnostic info —
# it is the label in disguise (e.g. a probe whose output equals the failure
# definition). Detect such columns statistically and route them to a separate
# "sanity" lane instead of testing/charting them as discriminators.
# ---------------------------------------------------------------------------

# A leak is the label *in disguise* — NOT merely a strong predictor. The
# signature is a BINARY flag that ~equals the FAIL label (a recomputed outcome,
# e.g. a probe that re-derives "is this a false detection"). A CONTINUOUS feature
# that perfectly separates the classes (e.g. object size) is legitimate discovery,
# the very thing we want to find — so separation alone never flags it. Recipe-level
# label references are caught earlier by compile_recipe's G4 guard.
_LEAK_MIN_N = 10
# A binary signal matching the FAIL label at ≥0.95 is a recomputed outcome, not a
# mechanism — genuine binary mechanism signals are noisy, and a probe re-deriving
# the answer lands near 1.0 (minus a little label drift). 0.95 catches the latter
# robustly while leaving any merely-strong (≤0.9) binary feature in the family.
_LEAK_BINARY_ACC = 0.95


def _auc(scores: list[float], labels: list[int]) -> float:
    """ROC-AUC of *scores* vs binary *labels* (rank-based, ties averaged)."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank across the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_pos = sum(ranks[idx] for idx, y in enumerate(labels) if y)
    u = rank_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def label_leak_score(sigmap: dict[str, float], labels: dict[str, bool]) -> dict[str, Any]:
    """Score whether a per-case signal IS the FAIL label in disguise.

    Returns ``{n, leak, score, kind, reason}``. Only BINARY signals can be flagged
    (``score`` = best-split accuracy vs the label); a binary flag that matches the
    label to ``_LEAK_BINARY_ACC`` with ≥ ``_LEAK_MIN_N`` cases is a recomputed
    outcome, not a discriminator. CONTINUOUS signals are NEVER flagged — perfect
    separation by a real feature is the discovery we want, not leakage (their AUC
    margin is still reported as ``score`` for transparency).
    """
    vals = list(sigmap.values())
    binary = bool(vals) and _is_binary(vals)
    # Align to labeled cases. Sparse binary flags: a labeled case missing from the
    # map means the signal is ABSENT (mirrors _split_signal_groups); continuous: skip.
    xs: list[float] = []
    ys: list[int] = []
    for cid, is_fail in labels.items():
        if cid in sigmap:
            xs.append(float(sigmap[cid]))
        elif binary:
            xs.append(0.0)
        else:
            continue
        ys.append(int(is_fail))
    n = len(xs)
    kind = "binary" if binary else "continuous"
    if n < _LEAK_MIN_N or not any(ys) or all(ys):
        return {"n": n, "leak": False, "score": 0.0, "kind": kind, "reason": ""}
    if not binary:
        # Report rank separation but never flag it — a perfectly separating
        # continuous feature is a finding, not a leak.
        margin = abs(2.0 * _auc(xs, ys) - 1.0)
        return {"n": n, "leak": False, "score": round(margin, 4),
                "kind": "continuous", "reason": ""}
    agree = sum(1 for x, y in zip(xs, ys) if int(x > 0.5) == y) / n
    acc = max(agree, 1.0 - agree)  # the signal may track FAIL or track PASS
    leak = acc >= _LEAK_BINARY_ACC
    return {"n": n, "leak": leak, "score": round(acc, 4), "kind": "binary",
            "reason": (f"binary signal reconstructs the FAIL label "
                       f"(best-split accuracy {acc:.3f})") if leak else ""}


#: Per-case signals that are the OUTCOME re-graded, not a mechanism: "is the
#: model's (baseline or strategy) answer correct" by an analyzer's own matcher.
#: They agree with the official label at 80-95% — below the leak threshold
#: (the matchers differ), above anything a real mechanism signal reaches — so
#: left in the family they are the BH survivors every time (qwen3.5-2b/
#: minervamath: all 6 survivors of 41 tests were these, effect -0.63 each) and
#: M5 then "refutes" or "supports" hypotheses on a tautology. The mechanism
#: content of these analyzers lives in their DERIVED flags (extraction_suspect,
#: label_disagrees, coverage_gap, majority_share, changed_answer, …), which
#: stay; strategy outcomes are compared PAIRED through ``groups``.
#: Matched on the metric name (the part after the analyzer prefix).
OUTCOME_REGRADE_METRICS: frozenset = frozenset({
    # the baseline answer re-graded
    "gold_in_output", "gold_in_answer_region", "strict_match", "answer_correct",
    "baseline_correct", "final_correct", "is_correct",
    # an intervention arm's answer graded (same item, same model: the
    # association with the label is the baseline's; the arm's VALUE is the
    # analyzer's gain scalar / derived flag, or a paired test)
    "revised_correct", "decomposed_correct", "own_facts_correct",
    "open_book_correct", "direct_correct", "reask_correct", "verify_correct",
    "continuation_correct", "told_correct",
    "majority_correct", "any_correct", "pass_at_k",
    # the COUNT of correct samples among k (coverage_verification_gap): a
    # 0..k integer, so label_leak_score (binary-only) never flags it, and
    # under degenerate sampling it is exactly {0, k} = the label. It was the
    # lone BH survivor on spatial457/qwen2.5-vl (2026-08-20) and named as
    # M5 evidence. n_unique (sample diversity) stays: that is a mechanism.
    "n_correct",
})


def _is_outcome_regrade(name: str) -> bool:
    metric = str(name).rsplit(".", 1)[-1].lower()
    return metric in OUTCOME_REGRADE_METRICS


def isolate_label_leaks(inp: StatsInput, *, denylist: "tuple[str, ...]" = ()) -> dict[str, str]:
    """Move label-reconstructing per-case columns from ``per_case`` to ``sanity``.

    Idempotent. A column is isolated when :func:`label_leak_score` flags it (a
    near-perfect label stand-in), when it is an outcome re-grade
    (:data:`OUTCOME_REGRADE_METRICS`), or its name contains a *denylist*
    substring. Returns ``{name -> reason}`` for the moved columns so callers can
    audit them. Leak-free columns are untouched, so the tested family holds only
    genuine candidate discriminators — and the explorer (fed ``per_case``) won't
    chart the isolated ones either.
    """
    moved: dict[str, str] = {}
    for name in list(inp.per_case):
        reason = ""
        if denylist and any(d in name for d in denylist):
            reason = "name matches leak denylist"
        elif _is_outcome_regrade(name):
            reason = "outcome re-grade (the answer's correctness, not a mechanism signal)"
        else:
            sc = label_leak_score(inp.per_case[name], inp.labels)
            if sc["leak"]:
                reason = sc["reason"]
        if reason:
            inp.sanity[name] = inp.per_case.pop(name)
            moved[name] = reason
    if moved:
        logger.info("isolated %d label-reconstructing signal(s) to the sanity lane: %s",
                    len(moved), ", ".join(sorted(moved)))
    return moved


def describe_data(inp: StatsInput) -> dict[str, Any]:
    """Compact, LLM-friendly summary of what statistical tests are feasible."""
    n_fail = sum(1 for v in inp.labels.values() if v)
    n_labeled = len(inp.labels)
    continuous = [k for k, m in inp.per_case.items() if not _is_binary(m.values())]
    return {
        "n_labeled": n_labeled,
        "n_fail": n_fail,
        "n_pass": n_labeled - n_fail,
        "per_case_signals": list(inp.per_case),
        "continuous_signals": continuous,
        "scalar_metrics": list(inp.scalars),
        "n_strategy_groups": len(inp.groups) if inp.groups else 0,
        # Label-reconstructing signals held out of the tested family (audit only).
        "sanity_signals": list(inp.sanity),
    }


def has_testable_data(inp: StatsInput) -> bool:
    """True when at least one tool could run (labels or strategy groups present)."""
    return bool(inp.labels) or bool(inp.groups)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _split_signal_groups(
    inp: StatsInput,
    signal_key: str,
    binarize: str = "median",
    threshold: float | None = None,
) -> tuple[list[int], list[int]] | None:
    """Split labeled cases into (signal-present, signal-absent) fail indicators.

    A sparse binary flag only lists the cases where it fired, so a labeled case
    missing from the signal map means the signal was *absent* (control group).
    For a continuous signal we cannot assume a value, so missing cases are
    skipped rather than defaulted.
    """
    sigmap = inp.per_case.get(signal_key)
    if not sigmap:
        return None
    binar = _binarize(sigmap, binarize, threshold)
    treat_missing_as_absent = _is_binary(sigmap.values())
    signal_fail: list[int] = []
    control_fail: list[int] = []
    for cid, is_fail in inp.labels.items():
        if cid in binar:
            present = binar[cid]
        elif treat_missing_as_absent:
            present = False
        else:
            continue
        (signal_fail if present else control_fail).append(int(is_fail))
    return signal_fail, control_fail


def _mean(xs: list[int]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _two_group_permutation_p(control_fail: list[int], signal_fail: list[int]) -> float:
    """Exact two-sided permutation p-value for a binary outcome split.

    Conditions on the group sizes and total number of failures. This gives the
    marginal signal test a p-value family so generalized M2 can control BH across
    many candidate signals.
    """
    n_control = len(control_fail)
    n_signal = len(signal_fail)
    n = n_control + n_signal
    if not n_control or not n_signal:
        return 1.0
    total_fail = sum(control_fail) + sum(signal_fail)
    obs = abs(_mean(signal_fail) - _mean(control_fail))
    denom = math.comb(n, n_signal)
    if denom == 0:
        return 1.0
    lo = max(0, n_signal - (n - total_fail))
    hi = min(n_signal, total_fail)
    prob = 0.0
    eps = 1e-12
    for k in range(lo, hi + 1):
        signal_rate = k / n_signal
        control_rate = (total_fail - k) / n_control
        if abs(signal_rate - control_rate) + eps >= obs:
            prob += math.comb(total_fail, k) * math.comb(n - total_fail, n_signal - k) / denom
    return min(1.0, max(0.0, prob))


def _tool_signal_label_assoc(inp: StatsInput, config: dict) -> StatsToolResult:
    """Unpaired fail-rate difference between cases with/without a per-case signal."""
    key = config.get("signal") or next(iter(inp.per_case), None)
    cfg = {**config, "signal": key}
    if not key or key not in inp.per_case:
        return StatsToolResult(
            tool="signal_label_assoc", config=cfg, ok=False,
            error="no per-case signal available", summary="signal_label_assoc: no signal",
        )
    split = _split_signal_groups(
        inp, key, config.get("binarize", "median"), config.get("threshold")
    )
    assert split is not None
    signal_fail, control_fail = split
    if not signal_fail or not control_fail:
        return StatsToolResult(
            tool="signal_label_assoc", config=cfg, ok=False,
            error="one group empty (need both signal-present and signal-absent cases)",
            summary="signal_label_assoc: degenerate split",
            details={"n_signal": len(signal_fail), "n_control": len(control_fail)},
        )
    sr = compare(
        control_fail, signal_fail, paired=False,
        alpha=config.get("alpha", 0.05),
        min_effect=config.get("min_effect", 0.0),
        n_boot=config.get("n_boot", 2000),
    )
    p_value = _two_group_permutation_p(control_fail, signal_fail)
    return StatsToolResult(
        tool="signal_label_assoc", config=cfg, ok=True,
        effect=sr.effect, ci=sr.ci, reject=sr.reject, p_value=p_value,
        underpowered=sr.underpowered,
        summary=f"signal '{key}' vs FAIL: {sr.summary()}",
        analysis_key=f"signal_label_assoc:{key}",
        correction_family="bh",
        raw_reject=sr.reject,
        details={
            "n_signal": len(signal_fail), "n_control": len(control_fail),
            "fail_rate_signal": round(_mean(signal_fail), 4),
            "fail_rate_control": round(_mean(control_fail), 4),
            "permutation_p": round(p_value, 6),
            **sr.details,
        },
    )


def _aligned_groups(inp: StatsInput, names: list[str]) -> tuple[list[str], list[list[float]]]:
    sets = [set(inp.groups[n]) for n in names]  # type: ignore[index]
    common = sorted(set.intersection(*sets)) if sets else []
    vecs = [[inp.groups[n][cid] for cid in common] for n in names]  # type: ignore[index]
    return common, vecs


def _tool_bootstrap_diff(inp: StatsInput, config: dict) -> StatsToolResult:
    """Unpaired fail-rate difference between two strategy groups (bootstrap CI)."""
    if not inp.groups or len(inp.groups) < 2:
        return StatsToolResult(
            tool="bootstrap_diff", config=config, ok=False,
            error="need >=2 strategy groups", summary="bootstrap_diff: <2 groups",
        )
    names = config.get("strategies") or list(inp.groups)[:2]
    names = list(names)[:2]
    common, vecs = _aligned_groups(inp, names)
    if not common:
        return StatsToolResult(
            tool="bootstrap_diff", config={**config, "strategies": names}, ok=False,
            error="no shared cases between groups", summary="bootstrap_diff: no overlap",
        )
    a, b = vecs
    sr = compare(
        a, b, paired=False,
        alpha=config.get("alpha", 0.05), min_effect=config.get("min_effect", 0.0),
        n_boot=config.get("n_boot", 2000),
    )
    return StatsToolResult(
        tool="bootstrap_diff", config={**config, "strategies": names}, ok=True,
        effect=sr.effect, ci=sr.ci, reject=sr.reject, underpowered=sr.underpowered,
        summary=f"{names[1]} vs {names[0]}: {sr.summary()}",
        analysis_key=f"bootstrap_diff:{names[0]}:{names[1]}",
        correction_family=None,
        raw_reject=sr.reject,
        details={"n": len(common), "strategies": names, **sr.details},
    )


def _tool_mcnemar_evalue(inp: StatsInput, config: dict) -> StatsToolResult:
    """Paired binary comparison of two strategies (McNemar + anytime-valid e-value)."""
    if not inp.groups or len(inp.groups) < 2:
        return StatsToolResult(
            tool="mcnemar_evalue", config=config, ok=False,
            error="need >=2 strategy groups", summary="mcnemar_evalue: <2 groups",
        )
    names = config.get("strategies") or list(inp.groups)[:2]
    names = list(names)[:2]
    common, vecs = _aligned_groups(inp, names)
    if not common:
        return StatsToolResult(
            tool="mcnemar_evalue", config={**config, "strategies": names}, ok=False,
            error="no shared cases between groups", summary="mcnemar_evalue: no overlap",
        )
    a, b = vecs
    sr = compare(
        a, b, paired=True,
        alpha=config.get("alpha", 0.05), min_effect=config.get("min_effect", 0.0),
    )
    return StatsToolResult(
        tool="mcnemar_evalue", config={**config, "strategies": names}, ok=True,
        effect=sr.effect, ci=sr.ci, reject=sr.reject, e_value=sr.e_value,
        p_value=sr.details.get("p_value"), underpowered=sr.underpowered,
        summary=f"{names[1]} vs {names[0]} (paired): {sr.summary()}",
        analysis_key=f"mcnemar_evalue:{names[0]}:{names[1]}",
        correction_family="e_bh",
        raw_reject=sr.reject,
        details={"n": len(common), "strategies": names, **sr.details},
    )


def _tool_friedman_nemenyi(inp: StatsInput, config: dict) -> StatsToolResult:
    """Rank 3+ strategies across shared cases (Friedman omnibus + Nemenyi post-hoc)."""
    if not inp.groups or len(inp.groups) < 3:
        return StatsToolResult(
            tool="friedman_nemenyi", config=config, ok=False,
            error="need >=3 strategy groups", summary="friedman_nemenyi: <3 groups",
        )
    names = list(inp.groups)
    common, vecs = _aligned_groups(inp, names)
    if not common:
        return StatsToolResult(
            tool="friedman_nemenyi", config=config, ok=False,
            error="no shared cases across all groups", summary="friedman_nemenyi: no overlap",
        )
    by_strategy = dict(zip(names, vecs))
    mc = compare_multiple(by_strategy, alpha=config.get("alpha", 0.05))
    return StatsToolResult(
        tool="friedman_nemenyi", config=config, ok=True,
        reject=mc.reject_global, p_value=mc.p_value,
        summary=mc.summary(),
        analysis_key="friedman_nemenyi:global",
        correction_family="bh",
        raw_reject=mc.reject_global,
        details={
            "avg_ranks": mc.avg_ranks,
            "critical_difference": mc.critical_difference,
            "significant_pairs": mc.significant_pairs,
            "n": mc.n,
        },
    )


def _tool_single_rate_evalue(inp: StatsInput, config: dict) -> StatsToolResult:
    """Anytime-valid test that the overall FAIL rate differs from a baseline p0.

    DESCRIPTIVE ONLY. The result is meaningful only when the case batch is a
    REPRESENTATIVE sample and ``p0`` is a justified baseline (the model's
    natural fail rate on this task). On a curated/enriched batch — the norm for
    diagnosis, where failures are over-sampled so a mechanism can be tested —
    the rate is a sampling artifact and ``p0=0.5`` tests nothing. The verdict
    layer treats this tool as descriptive and never makes it a hypothesis
    headline; pass ``config["p0"]`` = the manifest's recorded base rate to make
    it interpretable.
    """
    if not inp.labels:
        return StatsToolResult(
            tool="single_rate_evalue", config=config, ok=False,
            error="no labeled cases", summary="single_rate_evalue: no labels",
        )
    p0 = config.get("p0", 0.5)
    p0_justified = "p0" in config  # explicit baseline vs the meaningless default
    alpha = config.get("alpha", 0.05)
    fails = sum(1 for v in inp.labels.values() if v)
    n = len(inp.labels)
    res = e_value_test(fails, n, p0=p0, alpha=alpha)
    rate = fails / n
    caveat = "" if p0_justified else " (descriptive only: p0=0.5 is not a justified baseline)"
    return StatsToolResult(
        tool="single_rate_evalue", config={**config, "p0": p0}, ok=True,
        # No reported effect when p0 is the unjustified default — its rate − 0.5
        # would otherwise pollute any |effect|-based ranking downstream.
        effect=round(rate - p0, 4) if p0_justified else None,
        e_value=res["e_value"], reject=res["reject"] if p0_justified else False,
        analysis_key="single_rate_evalue:fail_rate",
        correction_family="e_bh" if p0_justified else None,
        raw_reject=res["reject"] if p0_justified else False,
        summary=(
            f"FAIL rate {rate:.1%} ({fails}/{n}) vs p0={p0:.2f}: "
            f"e={res['e_value']:.2f} -> "
            f"{'reject' if (res['reject'] and p0_justified) else 'inconclusive'}{caveat}"
        ),
        details={"fails": fails, "n": n, "rate": round(rate, 4),
                 "p0_justified": p0_justified, **res},
    )


def _tool_rank_corr(inp: StatsInput, config: dict) -> StatsToolResult:
    """Kendall τ between a continuous per-case signal and FAIL (monotonic association)."""
    key = config.get("signal") or next(iter(inp.per_case), None)
    cfg = {**config, "signal": key}
    if not key or key not in inp.per_case:
        return StatsToolResult(
            tool="rank_corr", config=cfg, ok=False,
            error="no per-case signal available", summary="rank_corr: no signal",
        )
    sigmap = inp.per_case[key]
    xs: list[float] = []
    ys: list[float] = []
    for cid, is_fail in inp.labels.items():
        if cid in sigmap:
            xs.append(sigmap[cid])
            ys.append(float(is_fail))
    if len(xs) < 3:
        return StatsToolResult(
            tool="rank_corr", config=cfg, ok=False,
            error="need >=3 paired (signal, label) points",
            summary="rank_corr: too few points", details={"n": len(xs)},
        )
    tau = kendall_tau(xs, ys)
    return StatsToolResult(
        tool="rank_corr", config=cfg, ok=True, effect=round(tau, 4),
        summary=f"Kendall τ between '{key}' and FAIL = {tau:+.3f} (n={len(xs)})",
        analysis_key=f"rank_corr:{key}",
        correction_family=None,
        details={"n": len(xs), "tau": round(tau, 4), "signal": key},
    )


# ---------------------------------------------------------------------------
# Tensor-level omnibus: decode the FAIL label from the full per-case attention
# map (not a scalar reduction). "Do FAIL and PASS attend differently *anywhere*?"
# A cross-validated linear decoder's out-of-fold AUC, calibrated by a label-
# permutation null — valid under the dependence between map cells, and feature-
# agnostic (robust to which scalar reduction would have mattered). Pure numpy.
# ---------------------------------------------------------------------------

_DECODE_MIN_N = 12      # too few maps to cross-validate a decoder meaningfully
_DECODE_MIN_PER_CLASS = 3


def _resize2d(m: "np.ndarray", g: int) -> "np.ndarray":
    """Bilinear-resize a 2-D map to ``(g, g)`` (pure numpy; no PIL dep)."""
    h, w = m.shape
    if (h, w) == (g, g):
        return m.astype(np.float64)
    yi = np.linspace(0, h - 1, g)
    xi = np.linspace(0, w - 1, g)
    y0 = np.floor(yi).astype(int)
    x0 = np.floor(xi).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (yi - y0)[:, None]
    wx = (xi - x0)[None, :]
    m = m.astype(np.float64)
    top = m[y0][:, x0] * (1 - wx) + m[y0][:, x1] * wx
    bot = m[y1][:, x0] * (1 - wx) + m[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def _cv_oof_scores(X: "np.ndarray", y: "np.ndarray", folds: int, lam: float, seed: int) -> "np.ndarray":
    """Out-of-fold decision scores from a regularized (ridge) linear decoder.

    Ridge least-squares on ±1 labels — closed-form and stable when features
    outnumber samples (the attention-map regime). Features are standardized on
    each fold's train split; the intercept is dropped (AUC is rank-invariant)."""
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    oof = np.zeros(n, dtype=np.float64)
    sizes = np.full(folds, n // folds, dtype=int)
    sizes[: n % folds] += 1
    start = 0
    eye = None
    for fs in sizes:
        te = idx[start:start + fs]
        tr = np.concatenate([idx[:start], idx[start + fs:]])
        start += fs
        if len(tr) < 2 or len(np.unique(y[tr])) < 2:
            continue  # degenerate fold → leave OOF scores at 0
        mu = X[tr].mean(0)
        sd = X[tr].std(0) + 1e-8
        xtr = (X[tr] - mu) / sd
        xte = (X[te] - mu) / sd
        if eye is None:
            eye = np.eye(xtr.shape[1])
        yc = 2.0 * y[tr] - 1.0
        w = np.linalg.solve(xtr.T @ xtr + lam * eye, xtr.T @ yc)
        oof[te] = xte @ w
    return oof


def _energy_distance_test(X: "np.ndarray", y: "np.ndarray", *,
                          n_perm: int, alpha: float, seed: int) -> "tuple[float, float, bool]":
    """Two-sample ENERGY-DISTANCE permutation test: do the FAIL and PASS rows of
    *X* come from different distributions?

    E = 2·mean‖x_fail − x_pass‖ − mean‖x_fail − x_fail'‖ − mean‖x_pass − x_pass'‖
    (≥0; larger = more different). More powerful than linear CV-decoding at low n —
    parameter-free and sensitive to nonlinear / higher-moment differences a linear
    boundary misses. The pairwise distance matrix is precomputed ONCE; each
    permutation only re-indexes it (and class sizes are preserved, so the diagonal
    bias cancels), so cost is O(n_perm · n²). Returns ``(energy, perm_p, reject)``."""
    n = len(y)
    sq = (X * X).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    dist = np.sqrt(d2)
    yb = np.asarray(y, dtype=bool)

    def _estat(mask: "np.ndarray") -> float:
        na = int(mask.sum())
        nb = n - na
        if na == 0 or nb == 0:
            return 0.0
        nm = ~mask
        daa = dist[np.ix_(mask, mask)].sum() / (na * na)
        dbb = dist[np.ix_(nm, nm)].sum() / (nb * nb)
        dab = dist[np.ix_(mask, nm)].sum() / (na * nb)
        return 2.0 * dab - daa - dbb

    obs = _estat(yb)
    rng = np.random.default_rng(seed)
    ge = 1  # +1 (observed) in both numerator and denominator → a valid permutation p
    for _ in range(n_perm):
        if _estat(rng.permutation(yb)) >= obs:
            ge += 1
    p = ge / (n_perm + 1)
    return float(obs), float(p), bool(p < alpha)


def _tool_attention_decoding(inp: StatsInput, config: dict) -> StatsToolResult:
    """Tensor-level omnibus: do FAIL and PASS per-case attention maps differ?

    Primary test: a two-sample ENERGY-DISTANCE permutation test over the full
    (standardized, resized) maps — parameter-free and more powerful than linear
    decoding at low n. A cross-validated linear-decoder out-of-fold AUC is
    reported alongside as an interpretable (but weaker) companion."""
    key = config.get("signal") or next(iter(inp.per_case_vectors), None)
    cfg = {**config, "signal": key}
    if not key or key not in inp.per_case_vectors:
        return StatsToolResult(
            tool="attention_decoding", config=cfg, ok=False,
            error="no per-case map vectors available", summary="attention_decoding: no maps",
        )
    vecmap = inp.per_case_vectors[key]
    g = int(config.get("grid", 8))
    lam = float(config.get("lam", 1.0))
    n_perm = int(config.get("n_perm", 500))   # tighter p floor (1/(n_perm+1)) than the old 200
    alpha = float(config.get("alpha", 0.05))
    seed = int(config.get("seed", 0))

    xs: list = []
    ys: list[int] = []
    for cid, is_fail in inp.labels.items():
        m = vecmap.get(cid)
        if m is None:
            continue
        m = np.asarray(m, dtype=np.float64)
        if m.ndim == 1:
            s = int(round(float(np.sqrt(m.size))))
            m = m.reshape(s, s) if s * s == m.size else m.reshape(1, -1)
        if m.ndim != 2 or m.size < 2:
            continue
        xs.append(_resize2d(m, g).ravel())
        ys.append(int(is_fail))

    n = len(ys)
    n_fail = int(sum(ys))
    if n < _DECODE_MIN_N or n_fail < _DECODE_MIN_PER_CLASS or (n - n_fail) < _DECODE_MIN_PER_CLASS:
        return StatsToolResult(
            tool="attention_decoding", config=cfg, ok=False,
            error=f"insufficient maps for the omnibus (n={n}, fail={n_fail})",
            summary="attention_decoding: underpowered", underpowered=True,
            details={"n": n, "n_fail": n_fail},
        )

    X = np.vstack(xs)
    y = np.asarray(ys, dtype=np.float64)
    # Standardize each map cell so no single high-variance patch dominates the
    # distance (or the decoder); both tests then see the map's SHAPE, not scale.
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)

    energy, p, reject = _energy_distance_test(Xz, y, n_perm=n_perm, alpha=alpha, seed=seed + 1)
    folds = max(2, min(int(config.get("folds", 5)), n_fail, n - n_fail))
    cv_auc = _auc(_cv_oof_scores(Xz, y, folds, lam, seed).tolist(), [int(v) for v in y])

    return StatsToolResult(
        tool="attention_decoding",
        config={**cfg, "grid": g, "n_perm": n_perm, "method": "energy_distance"},
        ok=True, effect=round(float(energy), 4), reject=reject, p_value=round(float(p), 4),
        underpowered=bool(not reject and p > 0.2 and n < 60),
        analysis_key=f"attention_decoding:{key}",
        correction_family="bh",
        raw_reject=reject,
        summary=(f"FAIL/PASS attention maps differ: energy-distance={energy:.3f}, "
                 f"permutation p={p:.3f} → {'reject H0 (maps differ)' if reject else 'inconclusive'} "
                 f"(companion CV-AUC={cv_auc:.3f}, n={n})"),
        details={"n": n, "n_fail": n_fail, "energy_distance": round(float(energy), 4),
                 "perm_p": round(float(p), 4), "cv_auc": round(float(cv_auc), 4),
                 "grid": g, "n_perm": n_perm, "n_features": int(X.shape[1]),
                 "method": "energy_distance"},
    )


# Registry: name -> callable. Edit STATS_TOOL_CATALOG in lockstep.
STATS_TOOLS: dict[str, Callable[[StatsInput, dict], StatsToolResult]] = {
    "signal_label_assoc": _tool_signal_label_assoc,
    "bootstrap_diff": _tool_bootstrap_diff,
    "mcnemar_evalue": _tool_mcnemar_evalue,
    "friedman_nemenyi": _tool_friedman_nemenyi,
    "single_rate_evalue": _tool_single_rate_evalue,
    "rank_corr": _tool_rank_corr,
    "attention_decoding": _tool_attention_decoding,
}

# Catalog text shown to the LLM selector (name -> when to use it).
STATS_TOOL_CATALOG: dict[str, str] = {
    "signal_label_assoc": (
        "Unpaired fail-rate difference between cases that exhibit a per-case "
        "analyzer signal and those that don't (bootstrap CI). Use when you have "
        "per-case signals AND PASS/FAIL labels — this is the main M2 test."
    ),
    "bootstrap_diff": (
        "Unpaired fail-rate difference between two strategy groups (bootstrap CI). "
        "Needs >=2 strategy groups (findings['by_strategy'])."
    ),
    "mcnemar_evalue": (
        "Paired binary comparison of two strategies on the same cases "
        "(McNemar + anytime-valid e-value). Needs exactly 2 paired strategy groups."
    ),
    "friedman_nemenyi": (
        "Rank 3+ strategies across shared cases (Friedman omnibus + Nemenyi "
        "post-hoc). Needs >=3 strategy groups."
    ),
    "single_rate_evalue": (
        "DESCRIPTIVE context only: tests whether the overall FAIL rate differs "
        "from a baseline p0. Meaningful ONLY on a representative sample with a "
        "justified p0 (pass config['p0'] = the natural base rate); on a curated/"
        "enriched diagnosis batch it tests nothing. Never decides a hypothesis."
    ),
    "rank_corr": (
        "Kendall tau between a continuous per-case signal and FAIL (monotonic "
        "association). Needs a continuous per-case signal."
    ),
    "attention_decoding": (
        "Tensor-level OMNIBUS: a two-sample ENERGY-DISTANCE permutation test over "
        "the FULL per-case attention map (not a scalar reduction), with a CV "
        "linear-decoder AUC reported alongside. Answers 'do FAIL and PASS attend "
        "differently anywhere?' — feature-agnostic and sensitive to nonlinear / "
        "distributional differences. Needs per-case map vectors (findings carry "
        "the scalars; the maps come from artifacts['per_case_maps'])."
    ),
}


def run_stats_tool(name: str, inp: StatsInput, config: dict | None = None) -> StatsToolResult:
    """Run a single catalog tool by name. Raises KeyError for unknown names."""
    tool = STATS_TOOLS[name]
    return tool(inp, config or {})


# ---------------------------------------------------------------------------
# Deterministic planner (fallback when no judge / LLM selection fails)
# ---------------------------------------------------------------------------

def default_plan(
    inp: StatsInput,
    max_signals: int | None = None,
) -> list[tuple[str, dict, str]]:
    """Deterministic ``[(tool, config, rationale)]`` plan from the data shape.

    The implementation delegates to the generic M2 planner so per-case signals
    are ranked by testability instead of original column order. ``max_signals``
    remains for backward compatibility; ``None`` means test every ranked signal.
    """
    from evalvitals.analysis.planner import plan_stats_input

    return [item.as_legacy_tuple() for item in plan_stats_input(inp, max_signals=max_signals)]


# ---------------------------------------------------------------------------
# Multiple-testing correction + visualization
# ---------------------------------------------------------------------------

def fdr_correct(results: list[StatsToolResult], alpha: float = 0.05) -> dict[str, Any]:
    """Apply multiplicity correction across all supported result families.

    e-values use e-BH; p-values use BH. The returned ``rejected_tools`` field is
    preserved for existing M1-M5 consumers, while ``rejected_result_keys`` and
    ``families`` expose the precise generalized-M2 family membership.
    """
    return correct_results(results, alpha=alpha)


def plot_effects(results: list[StatsToolResult], out_path: str) -> str | None:
    """Forest plot of effect ± CI for tools that produced both. Returns path or None."""
    items = [
        (r.tool, r.effect, r.ci)
        for r in results
        if r.ok and r.effect is not None and r.ci is not None
    ]
    if not items:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labels = [t for t, _, _ in items]
    effects = [e for _, e, _ in items]
    lows = [e - ci[0] for _, e, ci in items]
    highs = [ci[1] - e for _, e, ci in items]
    ys = list(range(len(items)))

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(items) + 1.5))
    ax.errorbar(effects, ys, xerr=[lows, highs], fmt="o", capsize=4, color="#2b6cb0")
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("effect size (fail-rate difference)")
    ax.set_title("M2 statistical effects (± CI)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
