"""Dataset profiling primitives for generalized M2 analysis.

This layer is intentionally data-domain agnostic. It identifies column types,
likely roles, missingness, and row grain so downstream planners can choose
statistics from evidence rather than column order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    if low in {"time", "timestamp", "date", "datetime"} or low.endswith("_time"):
        return "time"
    return "predictor"


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


def profile_records(records: Any, *, outcome_col: str | None = None) -> DatasetProfile:
    """Profile a list/DataFrame-like object of row records.

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
