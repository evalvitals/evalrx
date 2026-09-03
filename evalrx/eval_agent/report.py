"""Diagnostic report — turn a statistical verdict (+ its pre-registration) into output.

Carries the corrected decision, effect size + CI, e-value, the underpowered flag,
and the pre-registration hash that proves the hypothesis preceded the test.
"""

from __future__ import annotations

from typing import Optional

from evalrx.stats import StatResult


class DiagnosticReport:
    """Format an A/B / hypothesis-test conclusion as a dict or markdown."""

    @staticmethod
    def from_comparison(
        stat: StatResult,
        *,
        title: str = "A/B comparison",
        hypothesis=None,
        prereg_hash: Optional[str] = None,
        split: Optional[str] = None,
    ) -> dict:
        return {
            "title": title,
            "decision": "REJECT H0" if stat.reject else "inconclusive",
            "effect": stat.effect,
            "ci": list(stat.ci),
            "e_value": stat.e_value,
            "underpowered": stat.underpowered,
            "method": stat.method,
            "alpha": stat.alpha,
            "split": split,
            "preregistered": prereg_hash is not None,
            "prereg_hash": prereg_hash,
            "hypothesis": getattr(hypothesis, "statement", None) or getattr(hypothesis, "predicate", None),
            "details": stat.details,
        }

    @staticmethod
    def to_markdown(report: dict) -> str:
        lines = [
            f"## {report['title']}",
            f"- **Decision**: {report['decision']}"
            + ("  ⚠️ underpowered" if report.get("underpowered") else ""),
            f"- **Effect (B−A)**: {report['effect']:+.4f}  CI {report['ci'][0]:+.4f}..{report['ci'][1]:+.4f}",
            f"- **e-value**: {report['e_value']:.3g}" if report.get("e_value") is not None else "- e-value: n/a",
            f"- **Method**: {report['method']} (α={report['alpha']})",
        ]
        if report.get("hypothesis"):
            lines.append(f"- **Hypothesis**: {report['hypothesis']}")
        if report.get("preregistered"):
            lines.append(f"- **Pre-registered**: yes (hash `{report['prereg_hash']}`, split `{report.get('split')}`)")
        else:
            lines.append("- **Pre-registered**: NO — exploratory, not confirmatory")
        return "\n".join(lines)
