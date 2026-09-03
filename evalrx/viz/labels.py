"""Human-readable labels for internal EvalRX artifact identifiers."""

from __future__ import annotations

from typing import Any

DISPLAY_NAMES = {
    "generated_probe1_false_detection": "Sanity check: probe false-detection flag",
    "probe1_false_detection": "Sanity check: probe false-detection flag",
    "probe1_positive": "Probe says object is present",
    "relative_attention_focus_share": "Attention focus share",
    "relative_attention_max_relative_weight": "Maximum relative attention",
    "relative_attention_mean_relative_weight": "Mean relative attention",
    "low_focus": "High attention focus",
    "low_focus_share": "High attention focus",
}

COMPACT_NAMES = {
    "generated_probe1_false_detection": "Sanity check",
    "probe1_false_detection": "Sanity check",
    "probe1_positive": "Probe positive",
    "relative_attention_focus_share": "Attention focus",
    "relative_attention_max_relative_weight": "Max attention",
    "relative_attention_mean_relative_weight": "Mean attention",
    "low_focus": "High focus",
    "low_focus_share": "High focus",
}

PREFIXES = (
    ("groupstats_", "Group means for "),
    ("failrate_by_", "Failure rate by "),
    ("scatter_", "Scatter: "),
    ("corr_with_fail", "Correlation with failure"),
    ("correlations", "Signal correlations"),
    ("top_discriminators", "Top failure discriminators"),
    ("class_balance", "FAIL/PASS case balance"),
)


def raw_name(value: Any) -> str:
    return str(value or "").strip()


def display_name(value: Any, *, compact: bool = False) -> str:
    """Return a user-facing label for a raw signal/chart/table identifier."""
    text = raw_name(value)
    if not text:
        return ""
    base = text.rsplit("/", 1)[-1].removesuffix(".csv").removesuffix(".png")
    lookup = COMPACT_NAMES if compact else DISPLAY_NAMES
    if base in lookup:
        return lookup[base]
    for raw, label in sorted(lookup.items(), key=lambda item: len(item[0]), reverse=True):
        if raw in base:
            base = base.replace(raw, label)

    for prefix, phrase in PREFIXES:
        if base == prefix:
            return phrase.rstrip()
        if base.startswith(prefix):
            suffix = base[len(prefix):]
            return phrase + display_name(suffix, compact=compact)

    if "false_detection" in base:
        return "Sanity check: false-detection flag" if not compact else "Sanity check"
    if base.startswith("generated_"):
        return display_name(base.removeprefix("generated_"), compact=compact)
    if base.startswith("probe") and "positive" in base:
        return "Probe says object is present" if not compact else "Probe positive"

    cleaned = base.replace("_", " ").replace("-", " ")
    return cleaned[:1].upper() + cleaned[1:] if cleaned else text


def raw_hint(value: Any) -> str:
    text = raw_name(value)
    return f"Raw field: {text}" if text else ""
