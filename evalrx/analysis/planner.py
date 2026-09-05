"""Generic planning utilities for M2 analysis.

The planner consumes a profile of the available data and emits a deterministic
tool plan. It is intentionally conservative: discovery may inspect everything,
but confirmatory plans should make tested families explicit and avoid silently
dropping columns because of their original order.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from evalrx.analysis.profile import DatasetProfile, profile_stats_input

logger = logging.getLogger(__name__)


@dataclass
class AnalysisPlanItem:
    """One planned statistical analysis."""

    tool: str
    config: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    estimand: str = ""
    family: str = "confirmatory"
    priority: float = 0.0

    def as_legacy_tuple(self) -> tuple[str, dict[str, Any], str]:
        return self.tool, self.config, self.rationale

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "config": self.config,
            "rationale": self.rationale,
            "estimand": self.estimand,
            "family": self.family,
            "priority": self.priority,
        }


def _signal_priority(inp: Any, signal: str, profile: DatasetProfile) -> float:
    col = profile.columns.get(signal)
    sigmap = (getattr(inp, "per_case", {}) or {}).get(signal, {})
    labels = getattr(inp, "labels", {}) or {}
    if not sigmap:
        return 0.0
    n_labeled = max(1, len(labels))
    coverage = min(1.0, len(sigmap) / n_labeled)
    values = [float(v) for v in sigmap.values()]
    unique = len(set(values))
    variance = 0.0
    if values:
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
    kind_bonus = 0.2 if col and col.is_binary else 0.1 if col and col.dtype == "numeric" else 0.0
    sparse_binary_bonus = 0.15 if col and col.is_binary and coverage < 0.95 else 0.0
    constant_penalty = -1.0 if unique <= 1 else 0.0
    return coverage + min(1.0, variance) + kind_bonus + sparse_binary_bonus + constant_penalty


#: Agreement above which a BINARY signal is the label wearing another name.
#: Not 1.0: one flipped case out of dozens is still a restatement, not a finding.
_LABEL_RESTATEMENT_AGREEMENT = 0.95
#: Below this many shared cases, perfect agreement is cheap and means nothing.
_MIN_CASES_TO_JUDGE_RESTATEMENT = 10
#: Names of fields an analyzer produces BY GRADING — i.e. by asking the same
#: question the label answers.  Matched on the last dotted segment.
_CORRECTNESS_NAME = re.compile(
    r"(?:^|_)(?:correct|incorrect|label|labelled|passed|is_pass)(?:_|$)"
    r"|_match$|^match$"
)
# These answer-audit fields are defined using the gold and/or the stored
# PASS/FAIL label.  A sparse flag can have low whole-batch agreement while
# still making ``P(FAIL | flag)=1`` by construction, so an association test is
# circular.  They remain useful descriptive sanity signals, never candidate
# discriminators for M2/M4.
_LABEL_DERIVED_SUFFIXES = frozenset({
    "answer_extraction_audit.extraction_suspect",
    "answer_extraction_audit.extraction_point_miss",
    "answer_extraction_audit.gold_in_output",
    "answer_extraction_audit.gold_in_answer_region",
    "answer_extraction_audit.strict_match",
    "answer_extraction_audit.label_disagrees",
    "answer_extraction_audit.labelled_fail",
    # gold x pred conjunctions: P(FAIL | flag) = 1 by construction. Sparse
    # enough to slip past label_leak_score, named nothing like "correct", so
    # they sat in the VLM family as guaranteed BH survivors. The marginals
    # (pope.answered_yes / pope.gold_yes) stay testable.
    "pope.false_positive",
    "pope.false_negative",
})


def restates_label(inp: Any, signal: str) -> bool:
    """True when *signal* reproduces the PASS/FAIL label, or its complement.

    Such a signal always rejects H0, always survives FDR, and carries no
    information — "does the label predict the label". Left in the pool it also
    CROWDS OUT the real ones, because the survivor list is what M4 draws on to
    verify a hypothesis. Measured on qwen3.5-2b / bbh_causal_judgement: of 10
    rejecting signals, 6 survived FDR and every one of those 6 was the label
    (``labelled_fail``, ``strict_match``, ``calibration.correct``,
    ``self_repair.baseline_correct`` / ``revised_correct``), while the two
    genuinely informative extraction flags did not survive. M2's own narrative
    called them out as "the label copied under another name"; M4 then verified a
    hypothesis about label REPRODUCIBILITY using ``labelled_fail``, at
    confidence 0.73.

    Requires BOTH a correctness-shaped NAME and label agreement in the DATA,
    because neither alone is safe and each fixes the other's failure:

    * Data alone suppresses real findings. A composite discovered over
      independent measurements — ``(obj_size < 40) and (attention < 0.3)`` —
      can separate the labels perfectly, and that is the most valuable result
      the loop can produce, not a tautology. It is statistically identical to
      ``calibration.correct``; only its provenance differs.
    * Name alone suppresses honest fields. A column called ``correct`` that
      tracks something other than this batch's label is a legitimate signal.

    Two further things this deliberately does NOT do:

    * It does not judge arbitrary signals by the conditional rate. A sparse,
      independently measured flag may legitimately have
      ``P(FAIL | signal)=1``. The explicit answer-audit exceptions above are
      removed by provenance because their definitions use gold/labels, not
      because of their observed conditional rate.
    * It does not touch continuous signals. Restatement is an identity claim,
      and a continuous measure binarised at some threshold can drift into high
      agreement without being the label at all.
    """
    labels = getattr(inp, "labels", {}) or {}
    if str(signal).lower() in _LABEL_DERIVED_SUFFIXES:
        return bool(labels)
    if not _CORRECTNESS_NAME.search(str(signal).rsplit(".", 1)[-1].lower()):
        return False
    sigmap = (getattr(inp, "per_case", {}) or {}).get(signal, {})
    shared = [key for key in sigmap if key in labels]
    if len(shared) < _MIN_CASES_TO_JUDGE_RESTATEMENT:
        return False
    try:
        values = {float(sigmap[key]) for key in shared}
    except (TypeError, ValueError):
        return False
    if not values <= {0.0, 1.0} or len(values) < 2:
        return False
    agree = sum(1 for key in shared
                if bool(float(sigmap[key])) == bool(labels[key]))
    rate = agree / len(shared)
    return (rate >= _LABEL_RESTATEMENT_AGREEMENT
            or rate <= 1.0 - _LABEL_RESTATEMENT_AGREEMENT)


def label_restating_signals(inp: Any) -> list[str]:
    """The signals :func:`ranked_signal_names` drops, so callers can report them.

    Dropped rather than tested-and-flagged on purpose: the harm is not that the
    result is wrong, it is that the result is RIGHT and meaningless, and every
    such test consumes a slot in the FDR family that a real signal needed.
    """
    return sorted(
        signal for signal in (getattr(inp, "per_case", {}) or {})
        if restates_label(inp, signal)
    )


def ranked_signal_names(inp: Any, *, max_signals: int | None = None) -> list[str]:
    """Rank per-case signals by testability rather than insertion order."""
    profile = profile_stats_input(inp)
    scored = [
        (_signal_priority(inp, signal, profile), signal)
        for signal in (getattr(inp, "per_case", {}) or {})
        if not restates_label(inp, signal)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    names = [name for score, name in scored if score > 0]
    if max_signals is not None:
        names = names[:max(0, int(max_signals))]
    return names


def plan_stats_input(
    inp: Any,
    *,
    max_signals: int | None = None,
) -> list[AnalysisPlanItem]:
    """Create a deterministic confirmatory plan for a ``StatsInput`` object."""
    labels = getattr(inp, "labels", {}) or {}
    groups = getattr(inp, "groups", None) or {}
    vectors = getattr(inp, "per_case_vectors", {}) or {}
    n_fail = sum(1 for value in labels.values() if value)
    n_pass = len(labels) - n_fail
    plan: list[AnalysisPlanItem] = []

    if n_pass > 0 and n_fail > 0 and getattr(inp, "per_case", None):
        # Dropped, not silently: a suppressed signal that turns out to be real
        # has to leave a trace somewhere, and this is the only place that knows.
        dropped = label_restating_signals(inp)
        if dropped:
            logger.info(
                "planner: dropped %d signal(s) that restate the PASS/FAIL label "
                "and would crowd the FDR family: %s",
                len(dropped), ", ".join(dropped))
        ranked = ranked_signal_names(inp, max_signals=max_signals)
        for key in ranked:
            plan.append(AnalysisPlanItem(
                tool="signal_label_assoc",
                config={"signal": key},
                rationale=f"test whether per-case signal '{key}' predicts FAIL",
                estimand=f"P(FAIL | {key}=present/high) - P(FAIL | {key}=absent/low)",
                priority=1.0,
            ))
        continuous = [
            key for key in ranked
            if key in (getattr(inp, "per_case", {}) or {})
            and len({float(v) for v in inp.per_case[key].values()}) > 2
        ]
        for key in continuous:
            plan.append(AnalysisPlanItem(
                tool="rank_corr",
                config={"signal": key},
                rationale=f"monotonic association between continuous '{key}' and FAIL",
                estimand=f"Kendall tau({key}, FAIL)",
                family="descriptive",
                priority=0.5,
            ))

    if n_pass > 0 and n_fail > 0 and vectors:
        vector_names = list(vectors)
        if max_signals is not None:
            vector_names = vector_names[:max(0, int(max_signals))]
        for key in vector_names:
            plan.append(AnalysisPlanItem(
                tool="attention_decoding",
                config={"signal": key},
                rationale=f"omnibus: do FAIL/PASS per-case maps '{key}' differ?",
                estimand=f"distributional difference in vector signal {key}",
                priority=0.8,
            ))

    if labels:
        plan.append(AnalysisPlanItem(
            tool="single_rate_evalue",
            config={},
            rationale="describe the overall FAIL rate against an explicit baseline if provided",
            estimand="P(FAIL) - p0",
            family="descriptive",
            priority=0.1,
        ))

    n_groups = len(groups)
    if n_groups >= 3:
        plan.append(AnalysisPlanItem(
            tool="friedman_nemenyi",
            config={},
            rationale="rank 3+ strategies on shared cases",
            estimand="strategy rank differences",
            priority=0.9,
        ))
        names = list(groups)
        base = names[0]
        for variant in names[1:]:
            plan.append(AnalysisPlanItem(
                tool="mcnemar_evalue",
                config={"strategies": [base, variant]},
                rationale=f"paired contrast: does '{variant}' repair '{base}' failures?",
                estimand=f"{variant} success - {base} success on paired cases",
                priority=1.0,
            ))
    elif n_groups == 2:
        plan.append(AnalysisPlanItem(
            tool="mcnemar_evalue",
            config={},
            rationale="paired two-strategy comparison",
            estimand="strategy success difference on paired cases",
            priority=1.0,
        ))

    return plan
