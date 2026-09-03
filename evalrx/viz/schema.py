"""Typed contracts for EvalRX exploratory visualization artifacts."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

ChartKind = Literal["bar", "line", "scatter", "timeseries"]


class ChartSpec(TypedDict, total=False):
    name: str
    kind: ChartKind
    data: str | list[dict[str, Any]]
    x: str
    y: str
    title: str
    description: str
    figure_path: str
    render_skipped: str


class VisualPlan(TypedDict, total=False):
    name: str
    question: str
    data_shape: str
    plot_kind: str
    fallback_kind: ChartKind
    required_columns: list[str]
    rationale: str


class ChartReading(TypedDict, total=False):
    chart: str
    reading: str
    do_not_infer: str

