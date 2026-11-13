"""Reader-facing names for the signals, tools and configs a run records."""

from __future__ import annotations

import re
from typing import Any, Mapping

# These labels are deliberately about the behaviour being measured, rather
# than the analyzer implementation that happened to measure it.  Unknown keys
# still get a deterministic readable fallback and preserve their raw key in
# the audit-only fields below.
PLAIN_LABELS = {
    "answer_extraction_audit": "Did the model give a usable answer?",
    "calibration": "Is the model confident when it is wrong?",
    "termination_audit": "Did the model stop before finishing?",
    "format_sensitivity": "Does changing answer order change its choice?",
    "selfcheck_consistency": "Does the model contradict itself?",
    "coverage_verification_gap": "Can repeated attempts ever find the right answer?",
    "logprob_entropy": "Does the model appear uncertain while answering?",
    "self_consistency": "Does the answer stay the same across retries?",
    "signal_label_assoc": "How strongly is this behavior linked to errors?",
    "rank_corr": "Do more of this behavior and more errors move together?",
    "single_rate_evalue": "Is this rate unlikely to be chance?",
    "n_correct": "Number of correct attempts",
    "any_correct": "At least one correct attempt",
    "majority_correct": "Most attempts were correct",
    "format_flip_rate": "Answer changed after reordering choices",
    "positional_bias": "Tendency to favor one answer position",
    "output_chars": "Length of the model answer",
    "output_truncated": "Answer appears cut short",
    "recovered_by_continuation": "Could continue to a usable answer",
    "pass_at_k": "Share of retries that were correct",
    "n_sentences": "Number of answer sentences checked",
    "conf_logprob": "How certain the model appeared",
    "label_disagrees": "Recorded answer disagrees with the label",
    "has_answer_tag": "Used the requested answer format",
    "extracted_answer": "Answer read from the output",
    "labelled_fail": "This case was an error",
}


def _plain_label(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Recorded measurement"
    if raw in PLAIN_LABELS:
        return PLAIN_LABELS[raw]
    return re.sub(r"\s+", " ", raw.replace("_", " ")).strip().capitalize()


def _stat_label(config: "Mapping[str, Any] | None", signal: Any, tool: Any) -> str:
    """A name for what one statistical test measured, unique within a report.

    A signal test names its signal. A PAIRED test names none — it compares two
    strategies — so the label used to fall through to the tool, and every paired
    row in the report came out reading "Mcnemar Evalue". The pair it compared is
    the thing that tells those rows apart, and it is already in the config.
    """
    cfg = config if isinstance(config, Mapping) else {}
    if cfg.get("signal"):
        return _plain_signal(cfg["signal"])
    groups = [str(name) for name in (cfg.get("strategies") or []) if name]
    if len(groups) >= 2:
        return f"{_plain_label(groups[-1])} vs {_plain_label(groups[0])}"
    return _plain_signal(signal or tool)


def _plain_signal(value: Any) -> str:
    """Convert ``analyzer.metric`` into a question a non-expert can read."""
    raw = str(value or "")
    if not raw:
        return "Recorded behavior"
    analyzer, _, metric = raw.partition(".")
    if analyzer and metric:
        return f"{_plain_label(analyzer)} — {_plain_label(metric)}"
    return _plain_label(raw)


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {_plain_label(key): item for key, item in value.items()}
