"""Question template for the loop's optional in-cycle explore step.

The explorer (:class:`~evalvitals.analysis.explorer.ExploratoryAnalysisAgent`)
is a free-form EDA coder: it only knows what the question tells it. This
template turns the run's :class:`~evalvitals.eval_agent.stages.protocol.ExperimentProtocol`
into a question that (a) names the outcome column, (b) says what the columns
are (M1 analyzer per-case signals), and (c) asks for FROZEN, threshold-explicit
recipes — the only form a candidate signal can later be re-evaluated in.
"""

from __future__ import annotations

from typing import Any

_EXPLORE_QUESTION = (
    "Which per-case signals distinguish the FAIL cases (label=fail) from the "
    "PASS cases (label=pass)?{context} Every other column is a per-case signal "
    "an M1 analyzer computed on the model under diagnosis (column name = "
    "analyzer.metric, sanitized). Compare each signal's distribution between "
    "FAIL and PASS, look for interactions and thresholds, and make every "
    "candidate signal a FROZEN, threshold-explicit recipe over the available "
    "numeric columns (explicit numeric cut-offs, no re-fitting) so it can be "
    "re-evaluated verbatim on held-out cases. Charts are leads for hypotheses, "
    "not proof: this is the exploration half of a diagnosis loop, and a "
    "separate confirmatory stage will test whatever you surface."
)


def default_explore_question(protocol: Any | None = None) -> str:
    """Build the explore question from the run's protocol (or a generic one)."""
    parts: list[str] = []
    for attr in ("description", "task_domain", "failure_patterns"):
        value = getattr(protocol, attr, None) if protocol is not None else None
        if value:
            parts.append(str(value).strip())
    context = (" Context: " + " ".join(parts)) if parts else ""
    return _EXPLORE_QUESTION.format(context=context)
