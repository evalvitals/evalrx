"""Dataset profiling primitives for generalized M2 analysis.

This layer is intentionally data-domain agnostic. It identifies column types,
likely roles, missingness, and row grain so downstream planners can choose
statistics from evidence rather than column order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ColumnProfile:
    """Profile for one column in a row-oriented dataset."""

    name: str
    dtype: str
    role: str = "predictor"
    non_null: int = 0
    missing: int = 0
    unique: int = 0
    numeric_min: float | None = None
    numeric_max: float | None = None
    numeric_mean: float | None = None
    is_constant: bool = False
    is_binary: bool = False
    leakage_suspect: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "role": self.role,
            "non_null": self.non_null,
            "missing": self.missing,
            "unique": self.unique,
            "numeric_min": self.numeric_min,
            "numeric_max": self.numeric_max,
            "numeric_mean": self.numeric_mean,
            "is_constant": self.is_constant,
            "is_binary": self.is_binary,
            "leakage_suspect": self.leakage_suspect,
        }


@dataclass
class DatasetProfile:
    """General profile consumed by M2 planners and dashboards."""

    n_rows: int
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    id_columns: list[str] = field(default_factory=list)
    outcome_columns: list[str] = field(default_factory=list)
    group_columns: list[str] = field(default_factory=list)
    time_columns: list[str] = field(default_factory=list)
    grain: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    # Columns that look like the unit rows were cut from, found from the data
    # rather than the column name; see :func:`suggest_group_columns`.
    suggested_group_columns: list[dict[str, Any]] = field(default_factory=list)
    # Columns that almost fully determine the outcome; see :func:`leakage_suspects`.
    leakage_suspects: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "id_columns": self.id_columns,
            "outcome_columns": self.outcome_columns,
            "group_columns": self.group_columns,
            "time_columns": self.time_columns,
            "grain": self.grain,
            "warnings": self.warnings,
            "suggested_group_columns": self.suggested_group_columns,
            "leakage_suspects": self.leakage_suspects,
        }


def _row_items(row: Any) -> list[tuple[str, Any]]:
    if isinstance(row, dict):
        return list(row.items())
    if hasattr(row, "_asdict"):
        return list(row._asdict().items())
    if hasattr(row, "__dict__"):
        return list(vars(row).items())
    return []


def _infer_role(name: str) -> str:
    low = name.lower()
    if low in {"id", "case_id", "sample_id", "question_id"} or low.endswith("_id"):
        return "id"
    if low in {"label", "outcome", "target", "gold", "is_fail", "is_correct", "success"}:
        return "outcome"
    if low in {"model", "source", "source_dir", "dataset", "split", "strategy", "group"}:
        return "group"
    if low.endswith("_group") or low.startswith("group_"):
        return "group"
    if low in {"time", "timestamp", "date", "datetime"} or low.endswith("_time"):
        return "time"
    # Calendar words as whole name parts ("month_reported", "created_at",
    # "report_date"), not as substrings, so "candidate" is not read as a date.
    tokens = set(re.split(r"[^a-z0-9]+", low))
    if tokens & _TIME_TOKENS or low.endswith("_at"):
        return "time"
    return "predictor"


_TIME_TOKENS = frozenset({"date", "datetime", "timestamp", "month", "year", "week", "quarter"})
_ISO_DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?([T ]\d{2}:\d{2}.*)?$")


def _looks_like_dates(values: list[Any]) -> bool:
    """True when nearly every non-null value is an ISO date or year-month string."""
    strings = [v for v in values if isinstance(v, str) and v.strip()]
    if len(strings) < 3 or len(strings) < 0.9 * sum(v is not None for v in values):
        return False
    hits = sum(1 for v in strings if _ISO_DATE.match(v.strip()))
    return hits >= 0.9 * len(strings)


def _numeric_outcome(values: list[Any]) -> "list[float | None] | None":
    """The outcome as numbers, or None when it is not numeric or two-valued.

    Booleans and numbers map to themselves. A two-valued string outcome maps
    to 0/1 in sorted order; the direction does not matter to the callers,
    which only measure agreement and association strength.
    """
    present = [v for v in values if v is not None]
    if not present:
        return None
    if all(isinstance(v, (bool, int, float)) for v in present):
        return [None if v is None else float(v) for v in values]
    levels = sorted({str(v).strip().lower() for v in present})
    if len(levels) != 2:
        return None
    return [None if v is None else float(levels.index(str(v).strip().lower()))
            for v in values]


def intraclass_correlation(groups: list[Any], y: list[float]) -> float | None:
    """One-way ICC(1): the share of outcome variance that lies between groups.

    Zero means rows of one group agree no more than unrelated rows; a value
    near one means rows of a group are near-copies. Returns None when it is
    not defined (a single group, no group with two rows, or no variance).
    """
    by_group: dict[Any, list[float]] = {}
    for g, value in zip(groups, y):
        by_group.setdefault(g, []).append(value)
    k, n = len(by_group), len(y)
    if k < 2 or n <= k:
        return None
    grand = sum(y) / n
    ss_between = sum(len(v) * (sum(v) / len(v) - grand) ** 2 for v in by_group.values())
    ss_within = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in by_group.values())
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    n0 = (n - sum(len(v) ** 2 for v in by_group.values()) / n) / (k - 1)
    denom = ms_between + (n0 - 1.0) * ms_within
    if denom <= 0:
        return None
    return float((ms_between - ms_within) / denom)


def suggest_group_columns(
    rows: list[dict[str, Any]],
    *,
    outcome_col: str | None,
    exclude: "set[str] | frozenset[str]" = frozenset(),
    min_groups: int = 20,
    min_repeat_share: float = 0.2,
    min_icc: float = 0.1,
) -> list[dict[str, Any]]:
    """Columns that look like the unit several rows were cut from.

    A candidate has many distinct values, repeats (at least *min_repeat_share*
    of rows share their value with another row), and rows that share a value
    agree on the outcome far more than chance (ICC at least *min_icc*). Such a
    column should be passed as the group column: a row-level split would put
    rows of one group on both sides, and intervals that treat rows as
    independent would be too narrow.

    Only string columns and integer id columns are considered; a float
    column that repeats is a rounded measurement, not a sampling unit.
    Returns the candidates sorted by ICC, largest first.
    """
    if not rows or not outcome_col:
        return []
    y_all = _numeric_outcome([r.get(outcome_col) for r in rows])
    if y_all is None:
        return []
    names = sorted({str(k) for row in rows for k in row} - {outcome_col} - set(exclude))
    out: list[dict[str, Any]] = []
    for name in names:
        pairs = [
            (row.get(name), y) for row, y in zip(rows, y_all)
            if y is not None and row.get(name) not in (None, "")
        ]
        if len(pairs) < 2 * min_groups:
            continue
        values = [v for v, _ in pairs]
        is_string = all(isinstance(v, str) for v in values)
        is_int_id = (
            all(isinstance(v, int) and not isinstance(v, bool) for v in values)
            and _infer_role(name) in ("id", "group")
        )
        if not (is_string or is_int_id):
            continue
        counts: dict[Any, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        if len(counts) < min_groups or len(counts) == len(values):
            continue
        repeat_share = sum(c for c in counts.values() if c > 1) / len(values)
        if repeat_share < min_repeat_share:
            continue
        icc = intraclass_correlation(values, [y for _, y in pairs])
        if icc is None or icc < min_icc:
            continue
        out.append({
            "column": name,
            "n_groups": len(counts),
            "share_rows_in_repeated_groups": round(repeat_share, 4),
            "icc": round(icc, 4),
        })
    out.sort(key=lambda d: (-d["icc"], d["column"]))
    return out


def _auc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """Rank AUC of *scores* for the binary *y* (ties get the average rank)."""
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def leakage_suspects(
    rows: list[dict[str, Any]],
    *,
    outcome_col: str | None,
    exclude: "set[str] | frozenset[str]" = frozenset(),
    min_rows: int = 20,
    threshold: float = 0.98,
) -> list[dict[str, Any]]:
    """Columns that almost fully determine the outcome.

    Such a column is usually the outcome under another name, or was written
    after the outcome was known (a grader's note, a post-hoc flag). It would
    "explain" everything and mean nothing, so it is reported for the user to
    judge rather than silently used.

    For a two-valued outcome a column is suspect when guessing the more common
    outcome within each of its values is right at least *threshold* of the
    time (columns with few distinct values), or when its rank AUC is that far
    from 0.5 (numeric columns). For a numeric outcome it is suspect when the
    absolute correlation reaches *threshold*. Columns whose values are nearly
    unique per row are skipped: an id determines everything and tells nothing.
    """
    if not rows or not outcome_col:
        return []
    y_all = _numeric_outcome([r.get(outcome_col) for r in rows])
    if y_all is None:
        return []
    binary = {v for v in y_all if v is not None} <= {0.0, 1.0}
    names = sorted({str(k) for row in rows for k in row} - {outcome_col} - set(exclude))
    out: list[dict[str, Any]] = []
    for name in names:
        pairs = [
            (row.get(name), y) for row, y in zip(rows, y_all)
            if y is not None and row.get(name) is not None
        ]
        if len(pairs) < min_rows:
            continue
        values = [v for v, _ in pairs]
        y = np.asarray([t for _, t in pairs], dtype=float)
        if float(y.std()) == 0.0:
            continue
        numeric = all(isinstance(v, (bool, int, float)) for v in values)
        try:
            n_distinct = len(set(values))
        except TypeError:  # unhashable values (lists, dicts) are not predictors here
            continue
        base = max(float(y.mean()), 1.0 - float(y.mean())) if binary else None
        if binary and n_distinct <= max(2, min(20, len(values) // 5)):
            by_value: dict[Any, list[float]] = {}
            for v, t in pairs:
                by_value.setdefault(v, []).append(t)
            right = sum(max(sum(ts), len(ts) - sum(ts)) for ts in by_value.values())
            accuracy = right / len(pairs)
            if accuracy >= threshold and base is not None and base <= 0.95:
                out.append({"column": name, "measure": "accuracy_from_value",
                            "value": round(accuracy, 4), "baseline": round(base, 4)})
            continue
        if not numeric or n_distinct < 2:
            continue
        x = np.asarray([float(v) for v in values if isinstance(v, (bool, int, float))])
        if binary:
            auc = _auc(x, y)
            if auc is not None and max(auc, 1.0 - auc) >= threshold + (1.0 - threshold) / 2:
                out.append({"column": name, "measure": "auc", "value": round(auc, 4)})
        elif float(x.std()) > 0.0:
            r = float(np.corrcoef(x, y)[0, 1])
            if abs(r) >= threshold:
                out.append({"column": name, "measure": "correlation", "value": round(r, 4)})
    return out


def _infer_dtype(values: list[Any]) -> str:
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "empty"
    if all(isinstance(v, bool) for v in non_null):
        return "boolean"
    if all(isinstance(v, (int, float, bool)) for v in non_null):
        return "numeric"
    if all(isinstance(v, (list, tuple)) for v in non_null):
        return "vector"
    if all(isinstance(v, str) for v in non_null):
        lowered = {v.strip().lower() for v in non_null}
        if lowered <= {"pass", "fail", "passed", "failed", "success", "error", "correct",
                       "incorrect", "true", "false", "0", "1"}:
            return "categorical"
        return "text" if any(len(v) > 80 for v in non_null) else "categorical"
    return "mixed"


def profile_records(
    records: Any,
    *,
    outcome_col: str | None = None,
    group_cols: "list[str] | tuple[str, ...]" = (),
    time_cols: "list[str] | tuple[str, ...]" = (),
) -> DatasetProfile:
    """Profile a list/DataFrame-like object of row records.

    ``group_cols`` and ``time_cols`` name columns whose role the caller knows;
    they override the name heuristic, which only recognises a short list of
    English names. A named column that is absent from the data is reported in
    ``warnings`` instead of being ignored silently.

    ``outcome_col``, when given and present, overrides name-heuristic role
    inference: that column is forced to role ``"outcome"`` (and any other
    column the heuristic would have called ``"outcome"`` is demoted back to
    ``"predictor"`` so there is at most one outcome column). This lets a
    caller who knows their target column name (e.g. a continuous score with
    an arbitrary name) get correct framing without relying on
    :func:`_infer_role`'s English keyword list.
    """
    rows = []
    if records is not None and hasattr(records, "to_dict"):
        try:
            data = records.to_dict(orient="records")
            if isinstance(data, list):
                rows = [dict(r) for r in data if isinstance(r, dict)]
        except TypeError:
            rows = []
    if not rows:
        rows = [dict(_row_items(r)) for r in list(records or []) if _row_items(r)]

    names = sorted({str(k) for row in rows for k in row})
    columns: dict[str, ColumnProfile] = {}
    warnings: list[str] = []
    for name in names:
        vals = [row.get(name) for row in rows]
        non_null_vals = [v for v in vals if v is not None]
        dtype = _infer_dtype(vals)
        if outcome_col is not None and name in names:
            role = "outcome" if name == outcome_col else _infer_role(name)
            if role == "outcome" and name != outcome_col:
                role = "predictor"
        else:
            role = _infer_role(name)
        if role == "predictor" and _looks_like_dates(vals):
            role = "time"
        # An explicit role wins over anything inferred, except that the
        # outcome column stays the outcome.
        if role != "outcome":
            if name in group_cols:
                role = "group"
            elif name in time_cols:
                role = "time"
        unique_values = {str(v) for v in non_null_vals}
        numeric_vals = [
            float(v) for v in non_null_vals
            if isinstance(v, (int, float, bool))
        ]
        profile = ColumnProfile(
            name=name,
            dtype=dtype,
            role=role,
            non_null=len(non_null_vals),
            missing=len(vals) - len(non_null_vals),
            unique=len(unique_values),
            is_constant=len(unique_values) <= 1 and bool(non_null_vals),
            is_binary=bool(non_null_vals) and unique_values <= {"0", "1", "False", "True"},
            leakage_suspect=role == "outcome" or "label" in name.lower(),
        )
        if numeric_vals:
            profile.numeric_min = min(numeric_vals)
            profile.numeric_max = max(numeric_vals)
            profile.numeric_mean = sum(numeric_vals) / len(numeric_vals)
        if profile.is_constant and role == "predictor":
            warnings.append(f"column {name!r} is constant and unlikely to be testable")
        columns[name] = profile

    id_columns = [c.name for c in columns.values() if c.role == "id"]
    outcome_columns = [c.name for c in columns.values() if c.role == "outcome"]
    group_columns = [c.name for c in columns.values() if c.role == "group"]
    time_columns = [c.name for c in columns.values() if c.role == "time"]
    for kind, wanted in (("group", group_cols), ("time", time_cols)):
        for name in wanted:
            if name not in columns:
                warnings.append(f"{kind} column {name!r} was named but is not in the data")

    outcome_name = outcome_columns[0] if outcome_columns else None
    # A column the caller already named as the group needs no suggestion. One
    # that merely has a group-like name still does: the name alone does not
    # make the split or the intervals use it.
    suggested = suggest_group_columns(
        rows, outcome_col=outcome_name, exclude=set(group_cols)
    )
    for s in suggested:
        warnings.append(
            f"column {s['column']!r} looks like the unit rows were cut from "
            f"({s['n_groups']} groups, rows of one group share "
            f"{s['icc']:.0%} of the outcome variance); name it as the group column "
            "so the split and the intervals respect it"
        )
    suspects = leakage_suspects(
        rows, outcome_col=outcome_name, exclude=set(id_columns)
    )
    for s in suspects:
        if s["column"] in columns:
            columns[s["column"]].leakage_suspect = True
        warnings.append(
            f"column {s['column']!r} almost fully determines the outcome "
            f"({s['measure']}={s['value']}); check that it is not the outcome "
            "itself or something recorded after it"
        )
    grain = "case" if id_columns else "row"
    if id_columns and rows:
        id_col = id_columns[0]
        ids = [row.get(id_col) for row in rows if row.get(id_col) not in (None, "")]
        if len(set(ids)) < len(ids):
            grain = "repeated"
            warnings.append(f"id column {id_col!r} has repeated values")
    return DatasetProfile(
        n_rows=len(rows),
        columns=columns,
        id_columns=id_columns,
        outcome_columns=outcome_columns,
        group_columns=group_columns,
        time_columns=time_columns,
        grain=grain,
        warnings=warnings,
        suggested_group_columns=suggested,
        leakage_suspects=suspects,
    )


def describe_outcome(profile: DatasetProfile, *, continuous_unique_threshold: int = 8) -> dict[str, Any]:
    """Classify the dataset's outcome (if any) so callers can pick a framing.

    This is the piece that lets downstream prompts/plans stop assuming every
    dataset has a binary FAIL/PASS label. ``kind`` is one of:

    - ``"binary"``      — a two-valued outcome (booleans, pass/fail strings, 0/1).
    - ``"categorical"``  — 3+ discrete outcome values.
    - ``"continuous"``   — a numeric outcome with many distinct values.
    - ``"none"``         — no recognizable outcome column; treat as unsupervised EDA.

    Returns ``{"present", "column", "kind", "unique"}``. Only the first
    outcome column is described; ``profile_records(..., outcome_col=...)``
    is how a caller pins down which column that is.
    """
    if not profile.outcome_columns:
        return {"present": False, "column": None, "kind": "none", "unique": 0}
    name = profile.outcome_columns[0]
    col = profile.columns.get(name)
    if col is None:
        return {"present": False, "column": None, "kind": "none", "unique": 0}
    if col.dtype == "boolean" or col.is_binary or col.unique <= 2:
        kind = "binary"
    elif col.dtype == "numeric" and col.unique > continuous_unique_threshold:
        kind = "continuous"
    else:
        kind = "categorical"
    return {"present": True, "column": name, "kind": kind, "unique": col.unique}


def profile_stats_input(inp: Any) -> DatasetProfile:
    """Build a generic profile from the established M2 ``StatsInput`` contract."""
    columns: dict[str, ColumnProfile] = {}
    n_rows = len(getattr(inp, "labels", {}) or {})
    if n_rows:
        columns["label"] = ColumnProfile(
            name="label",
            dtype="boolean",
            role="outcome",
            non_null=n_rows,
            missing=0,
            unique=2 if any(inp.labels.values()) and not all(inp.labels.values()) else 1,
            is_binary=True,
            leakage_suspect=True,
        )
    for name, values in (getattr(inp, "per_case", {}) or {}).items():
        vals = list(values.values())
        nums = [float(v) for v in vals if isinstance(v, (int, float, bool))]
        cp = ColumnProfile(
            name=name,
            dtype=_infer_dtype(vals),
            role="predictor",
            non_null=len(vals),
            missing=max(0, n_rows - len(vals)),
            unique=len({str(v) for v in vals}),
            is_constant=len({str(v) for v in vals}) <= 1 and bool(vals),
            is_binary=bool(vals) and len(nums) == len(vals) and all(v in (0.0, 1.0) for v in nums),
        )
        if nums:
            cp.numeric_min = min(nums)
            cp.numeric_max = max(nums)
            cp.numeric_mean = sum(nums) / len(nums)
        columns[name] = cp
    for name in (getattr(inp, "scalars", {}) or {}):
        columns[name] = ColumnProfile(name=name, dtype="numeric", role="scalar", non_null=1)
    return DatasetProfile(
        n_rows=n_rows,
        columns=columns,
        outcome_columns=["label"] if n_rows else [],
        grain="case" if n_rows else "unknown",
    )
