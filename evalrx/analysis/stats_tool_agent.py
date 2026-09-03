"""Stats-tool selection and execution for M2 exploratory analysis.

The goal is to make M2 more than a narrative wrapper around threshold rules:
given raw analyzer results, this module selects small, reproducible analysis
tools and returns quantitative tables plus visualization specs.  The specs are
JSON-safe so they can be logged, rendered later, or handed to an LLM without
shipping heavy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import TYPE_CHECKING, Any

from evalrx.core.case import CaseBatch, Label

if TYPE_CHECKING:
    from evalrx.core.result import Result
    from evalrx.eval_agent.stages.protocol import ExperimentProtocol


@dataclass
class StatsToolResult:
    """Output from one selected stats tool."""

    name: str
    rationale: str
    metrics: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    visualizations: list[dict[str, Any]] = field(default_factory=list)
    conclusion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rationale": self.rationale,
            "metrics": self.metrics,
            "tables": self.tables,
            "visualizations": self.visualizations,
            "conclusion": self.conclusion,
        }


class StatsToolAgent:
    """Select and run built-in exploratory stats tools.

    This is the deterministic baseline for the future self-extending path where
    an agent can generate new stats code.  It intentionally keeps tool outputs
    compact and JSON-serializable.
    """

    def __init__(self, max_tools: int = 3) -> None:
        self.max_tools = max_tools

    def analyze(
        self,
        results: dict[str, "Result"],
        *,
        protocol: "ExperimentProtocol | None" = None,
    ) -> list[StatsToolResult]:
        selected = self.select(results, protocol=protocol)
        outputs: list[StatsToolResult] = []
        for name in selected:
            if name == "scalar_summary":
                out = _scalar_summary(results)
            elif name == "per_case_signal_label_association":
                out = _per_case_signal_label_association(results)
            else:
                continue
            if out is not None:
                outputs.append(out)
        return outputs

    def select(
        self,
        results: dict[str, "Result"],
        *,
        protocol: "ExperimentProtocol | None" = None,
    ) -> list[str]:
        names: list[str] = []
        if _has_numeric_scalars(results):
            names.append("scalar_summary")
        if _has_per_case_entries(results):
            names.append("per_case_signal_label_association")
        return names[: self.max_tools]


def _has_numeric_scalars(results: dict[str, "Result"]) -> bool:
    for result in results.values():
        for key, value in result.findings.items():
            if key == "per_case":
                continue
            if isinstance(value, (int, float, bool)):
                return True
    return False


def _has_per_case_entries(results: dict[str, "Result"]) -> bool:
    return any(
        isinstance(result.findings.get("per_case"), list)
        and len(result.findings.get("per_case", [])) > 0
        for result in results.values()
    )


def _scalar_summary(results: dict[str, "Result"]) -> StatsToolResult | None:
    rows: list[dict[str, Any]] = []
    values: list[float] = []

    for analyzer_name, result in sorted(results.items()):
        for metric, raw in sorted(result.findings.items()):
            if metric == "per_case" or not isinstance(raw, (int, float, bool)):
                continue
            value = float(raw)
            values.append(value)
            rows.append({
                "analyzer": analyzer_name,
                "metric": metric,
                "value": round(value, 6),
            })

    if not rows:
        return None

    chart = {
        "kind": "bar",
        "title": "Analyzer scalar metrics",
        "x": [f"{r['analyzer']}.{r['metric']}" for r in rows],
        "y": [r["value"] for r in rows],
        "y_label": "value",
    }

    return StatsToolResult(
        name="scalar_summary",
        rationale="Summarize numeric analyzer findings before hypothesis generation.",
        metrics={
            "n_scalar_metrics": len(rows),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "mean": round(mean(values), 6),
        },
        tables={"scalar_metrics": rows},
        visualizations=[chart],
        conclusion=f"Collected {len(rows)} numeric metric(s) across analyzer outputs.",
    )


def _per_case_signal_label_association(
    results: dict[str, "Result"],
) -> StatsToolResult | None:
    cases = _find_cases(results)
    signal = _extract_signals(results)
    if not signal:
        return None

    labeled: dict[str, bool] = {}
    if cases is not None:
        for case in cases:
            if getattr(case, "label", None) in {Label.PASS, Label.FAIL}:
                labeled[case.id] = case.label == Label.FAIL
                signal.setdefault(case.id, False)

    rows: list[dict[str, Any]] = []
    for cid, has_signal in sorted(signal.items()):
        row: dict[str, Any] = {
            "case_id": cid,
            "has_signal": bool(has_signal),
        }
        if cid in labeled:
            row["label"] = "fail" if labeled[cid] else "pass"
        rows.append(row)

    n_signal = sum(1 for hit in signal.values() if hit)
    n_no_signal = len(signal) - n_signal
    metrics: dict[str, Any] = {
        "n_cases_with_signal_entries": len(signal),
        "n_signal": n_signal,
        "n_no_signal": n_no_signal,
    }

    labeled_rows = [r for r in rows if "label" in r]
    if labeled_rows:
        signal_labels = [r["label"] == "fail" for r in labeled_rows if r["has_signal"]]
        control_labels = [r["label"] == "fail" for r in labeled_rows if not r["has_signal"]]
        if signal_labels:
            metrics["fail_rate_signal"] = round(sum(signal_labels) / len(signal_labels), 6)
        if control_labels:
            metrics["fail_rate_control"] = round(sum(control_labels) / len(control_labels), 6)
        if signal_labels and control_labels:
            metrics["effect_size"] = round(
                metrics["fail_rate_signal"] - metrics["fail_rate_control"], 6
            )

    chart = {
        "kind": "bar",
        "title": "Per-case diagnostic signal split",
        "x": ["signal", "no_signal"],
        "y": [n_signal, n_no_signal],
        "y_label": "case count",
    }
    if "fail_rate_signal" in metrics or "fail_rate_control" in metrics:
        chart_rates = {
            "kind": "bar",
            "title": "Fail rate by diagnostic signal",
            "x": ["signal", "no_signal"],
            "y": [
                metrics.get("fail_rate_signal", 0.0),
                metrics.get("fail_rate_control", 0.0),
            ],
            "y_label": "fail rate",
        }
        visualizations = [chart, chart_rates]
    else:
        visualizations = [chart]

    conclusion = (
        "Computed per-case diagnostic signal coverage."
        if "effect_size" not in metrics
        else f"Signal fail-rate gap is {metrics['effect_size']:.3f}."
    )

    return StatsToolResult(
        name="per_case_signal_label_association",
        rationale="Check whether analyzer per-case signals align with PASS/FAIL outcomes.",
        metrics=metrics,
        tables={"per_case_signal": rows},
        visualizations=visualizations,
        conclusion=conclusion,
    )


def _find_cases(results: dict[str, "Result"]) -> CaseBatch | None:
    for result in results.values():
        cases = getattr(result, "cases", None)
        if cases is not None and len(cases) > 0:
            return cases
    return None


def _extract_signals(results: dict[str, "Result"]) -> dict[str, bool]:
    signal: dict[str, bool] = {}
    non_signal_keys = {"sample_id", "id", "step", "first_error_step", "action", "judge_raw"}

    for result in results.values():
        per_case = result.findings.get("per_case", [])
        if not isinstance(per_case, list):
            continue
        for entry in per_case:
            if not isinstance(entry, dict):
                continue
            cid = str(entry.get("sample_id") or entry.get("id") or "")
            if not cid:
                continue
            hit = any(
                isinstance(v, (int, float, bool)) and bool(v)
                for k, v in entry.items()
                if k not in non_signal_keys
            )
            signal[cid] = signal.get(cid, False) or hit
    return signal
