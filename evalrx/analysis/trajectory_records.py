"""Trajectory → flat records — the M1→M2 bridge for agent runs.

`evalrx explore` and :func:`~evalrx.analysis.stats_tools.build_stats_input_from_records`
consume flat rows (``case_id`` + ``label`` + numeric signal columns).  This
module flattens each agent :class:`~evalrx.core.case.Trajectory` into such
a row of trajectory-level features, so agent runs plug into the SAME M2/M3
statistical machinery as single-step probes.

Feature semantics (all deterministic, recomputed from steps when metrics are
missing):

``n_steps`` / ``n_turns`` / ``n_tool_calls``
    loop volume — how much work the agent did.
``n_calls_<tool>``
    per-tool call counts (tool-selection profile), one column per tool seen
    anywhere in the batch (0-filled elsewhere).
``n_tool_errors`` / ``tool_error_rate``
    observations matching the executor's ``"[tool error"`` / ``"[error"``
    envelope.
``n_repeated_calls`` / ``repeated_call_frac`` / ``max_consecutive_repeat``
    duplicate ``(tool, args)`` signatures — the loop-failure family
    (``max_consecutive_repeat >= 2`` is what LoopDetector flags).
``n_images_returned``
    images tools handed back (visual-evidence volume).
``first_call_turn``
    turn of the first tool call (0 = never called a tool).
``distinct_tools``, ``terminated_final``, ``answer_len_chars``
    coverage, clean termination (1/0), and answer volume.

No imports from ``eval_agent`` (the analysis layer stays standalone).
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from evalrx.core.case import FailureCase, Label, StepRole, Trajectory

_ERROR_MARKERS = ("[tool error", "[error")

#: The standard M2 question for agent-trajectory records.  It declares the
#: column-family semantics up front so the analysis treats each family for
#: what it is (causal attribution vs stability vs judge labels vs cost) and
#: steers explanations toward the black-box fix surface (prompt/tool changes).
#: Use as-is or as a skeleton; the workbench and docs reference it.
AGENT_QUESTION_TEMPLATE = (
    "What predicts failures (label=fail) in this agent run? Records are one row "
    "per case with trajectory features: tool-use volume (n_tool_calls, "
    "n_calls_<tool>), loop structure (max_consecutive_repeat, "
    "repeated_call_frac), tool errors (tool_error_rate), cost "
    "(total_completion_tokens, total_latency_ms), stability from repeated runs "
    "(success_rate, flaky, pass_at_k vs pass_all_k), causal tool attribution "
    "from subset ablation (shap_outcome_<tool>, no_tools_pass), and a "
    "judge-assigned failure_mode (a hypothesis label, not ground truth). "
    "Compare FAIL vs PASS, say which signal families carry independent "
    "information, and prefer explanations that point at fixable causes — "
    "prompt wording, tool descriptions, tool availability, or loop policy."
)


def _observation_text(observation: Any) -> str:
    if isinstance(observation, dict):
        return str(observation.get("text", ""))
    return "" if observation is None else str(observation)


def _call_signature(tool_call: dict) -> "tuple[str, str]":
    name = str(tool_call.get("name", ""))
    try:
        args = json.dumps(tool_call.get("args", {}), sort_keys=True, default=str)
    except (TypeError, ValueError):
        args = str(tool_call.get("args"))
    return name, args


def trajectory_features(trajectory: Trajectory) -> dict[str, Any]:
    """Flatten one trajectory into a dict of scalar features (see module doc)."""
    tool_calls: list[tuple[str, str]] = []
    n_tool_errors = 0
    n_images = 0
    first_call_turn = 0
    max_consecutive = 0
    consecutive = 0
    prev_sig: Optional[tuple[str, str]] = None

    for step in trajectory.steps:
        if step.tool_call:
            sig = _call_signature(step.tool_call)
            tool_calls.append(sig)
            if first_call_turn == 0:
                first_call_turn = int(step.span.get("turn", len(tool_calls)))
            if sig == prev_sig:
                consecutive += 1
            else:
                consecutive = 1
            prev_sig = sig
            max_consecutive = max(max_consecutive, consecutive)
        if step.role is StepRole.TOOL:
            text = _observation_text(step.observation)
            if text.startswith(_ERROR_MARKERS):
                n_tool_errors += 1
            if isinstance(step.observation, dict):
                n_images += int(step.observation.get("n_images") or 0)

    n_calls = len(tool_calls)
    seen: set[tuple[str, str]] = set()
    n_repeated = 0
    for sig in tool_calls:
        if sig in seen:
            n_repeated += 1
        seen.add(sig)

    metrics = trajectory.metrics or {}
    actor_spans = [s.span for s in trajectory.steps if s.role is StepRole.ACTOR and s.span]
    latencies = [float(sp["latency_ms"]) for sp in actor_spans if sp.get("latency_ms")]
    total_latency = sum(latencies)
    total_prompt_toks = sum(int(sp.get("prompt_tokens") or 0) for sp in actor_spans)
    total_completion_toks = sum(int(sp.get("completion_tokens") or 0) for sp in actor_spans)

    features: dict[str, Any] = {
        "n_steps": int(metrics.get("n_steps", len(trajectory.steps))),
        "n_turns": int(metrics.get("n_turns", 0)) or sum(
            1 for s in trajectory.steps if s.role is StepRole.ACTOR
        ),
        "n_tool_calls": n_calls,
        "distinct_tools": len({name for name, _ in tool_calls}),
        "n_tool_errors": n_tool_errors,
        "tool_error_rate": round(n_tool_errors / n_calls, 4) if n_calls else 0.0,
        "n_repeated_calls": n_repeated,
        "repeated_call_frac": round(n_repeated / n_calls, 4) if n_calls else 0.0,
        "max_consecutive_repeat": max_consecutive,
        "n_images_returned": n_images,
        "first_call_turn": first_call_turn,
        "terminated": str(metrics.get("terminated", "unknown")),
        "terminated_final": 1 if metrics.get("terminated") == "final" else 0,
        "answer_len_chars": len(str(trajectory.final_answer)) if trajectory.final_answer else 0,
        # cost accounting (0 when the backend reports no usage/latency)
        "total_latency_ms": round(float(metrics.get("total_latency_ms") or total_latency), 1),
        "mean_turn_latency_ms": round(total_latency / len(latencies), 1) if latencies else 0.0,
        "total_prompt_tokens": int(metrics.get("total_prompt_tokens") or total_prompt_toks),
        "total_completion_tokens": int(
            metrics.get("total_completion_tokens") or total_completion_toks
        ),
    }
    for name in sorted({name for name, _ in tool_calls}):
        features[f"n_calls_{name}"] = sum(1 for n, _ in tool_calls if n == name)
    return features


def trajectories_to_records(
    items: Iterable["FailureCase | Trajectory"],
    *,
    id_col: str = "case_id",
    label_col: str = "label",
) -> list[dict[str, Any]]:
    """Flatten cases/trajectories into ``records.json``-shaped rows.

    Each row carries *id_col*, *label_col* (the case label, falling back to
    the trajectory outcome), the trajectory features, and any scalar
    ``case.metadata`` entries (provenance columns like model/source).  Per-tool
    count columns are 0-filled across the batch so every row has the same
    schema.  Items without a trajectory are skipped.
    """
    rows: list[dict[str, Any]] = []
    tool_cols: set[str] = set()
    for item in items:
        if isinstance(item, Trajectory):
            case, trajectory = None, item
        else:
            case, trajectory = item, item.trajectory
        if trajectory is None:
            continue
        label = trajectory.outcome
        if case is not None and case.label is not Label.UNKNOWN:
            label = case.label
        row: dict[str, Any] = {
            id_col: case.id if case is not None else trajectory.sample_id,
            label_col: label.value if isinstance(label, Label) else str(label),
        }
        if case is not None:
            for key, value in (case.metadata or {}).items():
                if isinstance(value, (str, int, float, bool)) and key not in (id_col, label_col):
                    row.setdefault(key, value)
        features = trajectory_features(trajectory)
        tool_cols.update(k for k in features if k.startswith("n_calls_"))
        row.update(features)
        rows.append(row)

    for row in rows:  # uniform schema: 0-fill per-tool columns
        for col in tool_cols:
            row.setdefault(col, 0)
    return rows
