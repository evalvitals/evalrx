"""M2 — AnalysisModule: interpret raw analyzer results into structured findings.

The analysis module bridges M1 (raw Results) and M3 (hypothesis generation).
It applies threshold-based rules to flag anomalies, assigns severity, and
produces a human-readable narrative that the diagnosis agent (M3) uses as
context when generating hypotheses.

Usage::

    module = AnalysisModule()
    report = module.analyze(probe_results, model_name="qwen3-vl-8b")
    print(report.severity)    # "high" / "medium" / "low" / "none"
    print(report.narrative)   # human-readable summary for the LLM
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from evalrx.core.result import Result


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Rule:
    metric: str
    threshold: float
    direction: Literal["above", "below"]
    severity: Literal["high", "medium", "low"]
    message: str


# Image-related token name fragments — used to compute image_token_attention_ratio
# from the top_attended_tokens list produced by the attention analyzer.
_IMAGE_TOKEN_FRAGMENTS = frozenset(
    {"image_pad", "vision_start", "vision_end", "img", "<|vision", "<|image"}
)


def _compute_image_token_attention_ratio(findings: dict) -> float | None:
    """Sum attention weights of image-related tokens from top_attended_tokens.

    Returns the ratio (0–1) or None if the field is absent / not a list.
    """
    top = findings.get("top_attended_tokens")
    if not isinstance(top, list) or not top:
        return None
    total = 0.0
    image_total = 0.0
    for entry in top:
        w = entry.get("weight", 0.0)
        total += w
        tok = str(entry.get("token", "")).lower()
        if any(frag in tok for frag in _IMAGE_TOKEN_FRAGMENTS):
            image_total += w
    if total <= 0:
        return None
    return image_total / total


_RULES: dict[str, list[_Rule]] = {
    "attention": [
        _Rule(
            "image_token_attention_ratio", 0.05, "below", "medium",
            "VLM nearly ignores image tokens — attention dominated by text/structural tokens",
        ),
    ],
    "attention_sink": [
        _Rule("mean_sink_mass", 0.6, "above", "high",
              "model over-attends to the attention sink token"),
    ],
    "logprob_entropy": [
        _Rule("perplexity", 50.0, "above", "medium",
              "high perplexity — model is uncertain on this data"),
    ],
    "token_entropy": [
        _Rule("mean_entropy", 3.0, "above", "medium",
              "high mean token entropy — broad per-step uncertainty"),
    ],
    "self_consistency": [
        _Rule("consistency", 0.5, "below", "medium",
              "low self-consistency — model gives unstable answers"),
    ],
    "verbalized_confidence": [
        _Rule("verbalized_confidence", 0.4, "below", "low",
              "model expresses low confidence in its own outputs"),
    ],
    "pope": [
        _Rule("accuracy",  0.7, "below", "high",
              "low POPE accuracy — model frequently hallucinates objects"),
        _Rule("f1",        0.7, "below", "medium",
              "low POPE F1 — precision/recall imbalance in object detection"),
        _Rule("yes_rate",  0.7, "above", "high",
              "over-affirmation — model answers 'yes' regardless of image evidence"),
        _Rule("false_positive_rate", 0.5, "above", "high",
              "presence hallucination — model claims absent findings are present"),
    ],
    "chair": [
        _Rule("chair_i",   0.3, "above", "high",
              "high per-instance hallucination rate (CHAIR-I)"),
        _Rule("chair_s",   0.5, "above", "medium",
              "majority of captions contain at least one hallucination (CHAIR-S)"),
    ],
    "mm_shap": [
        _Rule("mm_score",  0.85, "above", "medium",
              "model over-relies on image, largely ignoring text tokens"),
        _Rule("mm_score",  0.05, "below", "medium",
              "model nearly ignores the image, relying only on text"),
    ],
    "loop_detect": [
        _Rule("n_with_loops", 0, "above", "high",
              "model enters repetitive action loops"),
    ],
    "ignored_obs": [
        _Rule("n_with_ignored_obs", 0, "above", "medium",
              "model ignores error observations returned by tools"),
    ],
    "cka": [
        _Rule("mean_offdiagonal_cka", 0.95, "above", "medium",
              "layers are near-identical — possible representational collapse"),
    ],
}

_SEVERITY_ORDER: dict[str, int] = {"high": 3, "medium": 2, "low": 1, "none": 0}


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class AnalysisFinding:
    """One flagged anomaly from a single analyzer metric.

    Attributes:
        analyzer:   Name of the analyzer that produced this finding.
        metric:     The specific metric key within ``findings``.
        value:      Observed value.
        threshold:  The threshold it crossed.
        direction:  ``"above"`` or ``"below"`` the threshold.
        severity:   ``"high"``, ``"medium"``, or ``"low"``.
        message:    Short human-readable description.
    """

    analyzer: str
    metric: str
    value: float
    threshold: float
    direction: Literal["above", "below"]
    severity: Literal["high", "medium", "low"]
    message: str

    def __str__(self) -> str:
        cmp = ">" if self.direction == "above" else "<"
        return (
            f"[{self.severity.upper()}] {self.analyzer}.{self.metric}="
            f"{self.value:.3g} {cmp} {self.threshold}: {self.message}"
        )


@dataclass
class AnalysisReport:
    """Structured output of :class:`AnalysisModule`.

    Attributes:
        model_name:  ``repr()`` of the analysed model.
        findings:    Flagged anomalies, sorted high-severity first.
        severity:    Overall severity (worst finding, or ``"none"`` if clean).
        narrative:   Multi-line human-readable summary forwarded to M3.
        raw_results: Original ``{analyzer: Result}`` from M1.
    """

    model_name: str
    findings: list[AnalysisFinding] = field(default_factory=list)
    severity: Literal["high", "medium", "low", "none"] = "none"
    narrative: str = ""
    raw_results: dict[str, "Result"] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "severity": self.severity,
            "n_findings": len(self.findings),
            "findings": [str(f) for f in self.findings],
            "narrative": self.narrative,
        }


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class AnalysisModule:
    """M2: apply threshold rules to raw analyzer results and build a report.

    Args:
        extra_rules: Additional ``{analyzer_name: [_Rule, ...]}`` entries
                     to merge with the built-in rules.
    """

    def __init__(self, extra_rules: dict[str, list[_Rule]] | None = None) -> None:
        self._rules: dict[str, list[_Rule]] = dict(_RULES)
        if extra_rules:
            for name, rules in extra_rules.items():
                self._rules.setdefault(name, []).extend(rules)

    def analyze(
        self,
        results: dict[str, "Result"],
        model_name: str = "",
    ) -> AnalysisReport:
        """Scan *results* for threshold violations and build an :class:`AnalysisReport`.

        Args:
            results:    ``{analyzer_name: Result}`` from M1 (ProbeAgent).
            model_name: Human-readable model identifier for the report.

        Returns:
            :class:`AnalysisReport` with flagged findings and a narrative.
        """
        findings: list[AnalysisFinding] = []

        for analyzer_name, result in results.items():
            # Derive scalar metrics from structured findings before applying rules.
            # This lets rules operate on computed values not directly in findings.
            derived: dict[str, float] = {}
            if analyzer_name == "attention":
                ratio = _compute_image_token_attention_ratio(result.findings)
                if ratio is not None:
                    derived["image_token_attention_ratio"] = ratio

            rules = self._rules.get(analyzer_name, [])
            for rule in rules:
                raw_val = derived.get(rule.metric) or result.findings.get(rule.metric)
                if raw_val is None:
                    continue
                try:
                    val = float(raw_val)
                except (TypeError, ValueError):
                    continue

                flagged = (
                    (rule.direction == "above" and val > rule.threshold)
                    or (rule.direction == "below" and val < rule.threshold)
                )
                if flagged:
                    findings.append(
                        AnalysisFinding(
                            analyzer=analyzer_name,
                            metric=rule.metric,
                            value=val,
                            threshold=rule.threshold,
                            direction=rule.direction,
                            severity=rule.severity,
                            message=f"{rule.message} (image_attn_ratio={val:.3f})"
                            if rule.metric == "image_token_attention_ratio"
                            else rule.message,
                        )
                    )

        # Sort: high → medium → low
        findings.sort(key=lambda f: -_SEVERITY_ORDER[f.severity])

        overall: Literal["high", "medium", "low", "none"] = "none"
        if findings:
            overall = findings[0].severity  # type: ignore[assignment]

        narrative = _build_narrative(model_name, findings, results)

        return AnalysisReport(
            model_name=model_name,
            findings=findings,
            severity=overall,
            narrative=narrative,
            raw_results=results,
        )


#: Metric names that count how many items an analyzer actually examined:
#: ``n_scored``, ``n_trajectories``, ``n_cases``, ``n_samples``, ``num_steps`` …
_COUNT_KEY = re.compile(r"(?:^n$|^n_|^num_|_count$)", re.IGNORECASE)


def _zero_counts(results: dict) -> list[str]:
    """``analyzer.metric`` keys whose sample count is zero.

    Deliberately restricted to count-like names. A zero elsewhere is usually a
    real measurement — ``flip_rate = 0`` means nothing flipped, which IS a
    finding — whereas a zero sample count means there was nothing to measure at
    all. Reporting the two the same way is how "examined 0 items" became a line
    under "healthy metrics".
    """
    out: list[str] = []
    for r_name, r in results.items():
        for key, value in (getattr(r, "findings", {}) or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value == 0 and _COUNT_KEY.search(key):
                out.append(f"{r_name}.{key}")
    return sorted(out)


def _build_narrative(
    model_name: str,
    findings: list[AnalysisFinding],
    results: dict[str, "Result"],
) -> str:
    lines: list[str] = [f"Model: {model_name}"]
    lines.append(f"Analyzers run: {', '.join(sorted(results)) or 'none'}")

    # Absence of evidence, before anything else. A zero count means the analyzer
    # examined nothing -- format_sensitivity found no options to rotate,
    # first_error_judge found no trajectories. Those used to be printed under
    # "healthy metrics (no threshold violations)", which reads as a clean bill of
    # health for a measurement that never happened.
    empty = _zero_counts(results)
    if empty:
        lines.append(
            f"\n{len(empty)} metric(s) measured NOTHING — absence of evidence, "
            f"NOT evidence of health:")
        for key in empty:
            lines.append(f"  {key} = 0  (analyzer ran but examined no items)")

    if not findings:
        lines.append(
            "No anomalies detected — all metrics within normal ranges."
            if not empty else
            "No anomalies detected among the metrics that DID measure something; "
            "the zero-count metrics above are not evidence either way.")
        return "\n".join(lines)

    lines.append(f"\n{len(findings)} anomalie(s) detected:")
    for f in findings:
        lines.append(f"  {f}")

    # Add a brief summary of healthy metrics for context
    all_metrics = {
        (r_name, k): v
        for r_name, r in results.items()
        for k, v in r.findings.items()
        if isinstance(v, (int, float))
    }
    if all_metrics:
        lines.append("\nSelected healthy metrics (no threshold violations):")
        flagged_keys = {(f.analyzer, f.metric) for f in findings}
        empty_keys = set(empty)
        shown = 0
        for (r_name, k), v in all_metrics.items():
            if (r_name, k) in flagged_keys or f"{r_name}.{k}" in empty_keys:
                continue
            if shown < 5:
                lines.append(f"  {r_name}.{k} = {v:.3g}")
                shown += 1

    return "\n".join(lines)
