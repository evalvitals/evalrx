"""Shared run-provenance and logging-wiring helpers used by both
``VLDiagnoseLoop`` (loop.py) and ``AutoDiagnoseLoop`` (legacy.py).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _attach_run_logger(run_logger: Any, *agents: Any) -> None:
    """Point each agent's ``run_logger`` at *run_logger* when it is unset.

    Lets the probe / stats generators emit ``tool_codegen`` events without the
    caller having to wire the logger into every agent by hand.  Respects an
    explicitly-set logger (only fills in ``None``).
    """
    if run_logger is None:
        return
    for agent in agents:
        if agent is not None and getattr(agent, "run_logger", None) is None:
            try:
                agent.run_logger = run_logger
            except Exception:  # noqa: BLE001 - never let logging wiring break a run
                pass


def _data_provenance(data: Any) -> dict[str, Any]:
    """Fingerprint the case batch + tally its labels for the ``run_start`` record.

    Returns ``{"data_fingerprint": <12-hex>, "label_distribution": {...}}`` — the
    fingerprint is a SHA-1 over each case's ``id`` (or its prompt when no id),
    order-independent, so the same batch always hashes the same regardless of
    iteration order, and a different batch (even same size) hashes differently.
    Best-effort: returns ``{}`` if *data* isn't iterable.
    """
    import hashlib
    from collections import Counter

    try:
        cases = list(data)
    except Exception:  # noqa: BLE001
        return {}

    keys: list[str] = []
    labels: Counter = Counter()
    for c in cases:
        cid = getattr(c, "id", None)
        if not cid:
            prompt = getattr(getattr(c, "inputs", None), "prompt", None)
            cid = str(prompt) if prompt is not None else repr(c)
        keys.append(str(cid))
        label = getattr(c, "label", None)
        labels[getattr(label, "name", "UNKNOWN")] += 1

    out: dict[str, Any] = {}
    if keys:
        digest = hashlib.sha1("\n".join(sorted(keys)).encode("utf-8")).hexdigest()
        out["data_fingerprint"] = digest[:12]
    if labels:
        out["label_distribution"] = dict(labels)
    return out


def _run_config(loop: Any, data: Any, *, loop_name: str) -> dict[str, Any]:
    """Build the ``run_start`` provenance dict from a loop instance + its data.

    Robust to both loop types and missing agents — every field is best-effort so
    a partially-configured loop still produces a useful config record.
    """
    cfg: dict[str, Any] = {"loop": loop_name}
    try:
        cfg["model"] = repr(loop.model)
    except Exception:  # noqa: BLE001
        pass
    cfg["max_cycles"] = getattr(loop, "max_cycles", None)
    cfg["token_budget"] = getattr(loop, "token_budget", None)
    cfg["analysis_only"] = getattr(loop, "analysis_only", None)
    try:
        cfg["n_cases"] = len(data)
    except Exception:  # noqa: BLE001
        pass
    # Dataset provenance: a stable fingerprint over the cases plus their label
    # breakdown.  n_cases alone says how many; this says *which* (so two runs
    # can be confirmed to use the same batch) and the base failure rate the
    # whole diagnosis is conditioned on — both essential to interpret the run.
    _data_prov = _data_provenance(data)
    if _data_prov:
        cfg.update(_data_prov)
    # How the batch was partitioned before anything ran. ``n_cases`` above is
    # the explore partition (``data`` has already been narrowed to it); these
    # say whether a confirm / test partition was withheld and at what fraction,
    # so a report can tell a two-way run from a train/val/test one.
    for attr in ("confirm_split", "test_split", "confirm_split_seed"):
        value = getattr(loop, attr, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cfg[attr] = value

    protocol = getattr(loop, "protocol", None)
    if protocol is not None:
        cfg["protocol"] = {
            "description": getattr(protocol, "description", ""),
            "task_domain": getattr(protocol, "task_domain", None),
        }

    # Judge — read off the M3 agent (or its lazy default) when present.
    diag_agent = getattr(loop, "diagnosis_agent", None)
    judge = getattr(diag_agent, "judge", None) if diag_agent is not None else None
    if judge is not None:
        cfg["judge"] = repr(judge)

    # Coder — the M5 surgery writer's CLI provider/model, when configured.
    surgery = getattr(loop, "surgery_agent", None)
    writer = getattr(surgery, "_writer", None) if surgery is not None else None
    cli = getattr(getattr(writer, "_cfg", None), "cli_agent", None)
    if cli is not None:
        provider = getattr(cli, "provider", None)
        model = getattr(cli, "model", "")
        if provider:
            cfg["coder"] = f"{provider}:{model}" if model else provider

    # Whether tool synthesis (codegen) is enabled on either agent.
    cfg["allow_codegen"] = bool(
        getattr(getattr(loop, "probe_agent", None), "allow_codegen", False)
        or getattr(getattr(loop, "stats_agent", None), "_allow_codegen", False)
        or getattr(getattr(loop, "fix_agent", None), "_allow_codegen", False)
    )
    # Whether the in-cycle explore step (free-form EDA beside M2) is configured —
    # so a run with explorer notes in M3 is distinguishable from one without.
    explorer = getattr(loop, "explorer", None)
    if explorer is not None:
        cli = getattr(explorer, "_cli_config", None)
        provider = getattr(cli, "provider", None)
        model = getattr(cli, "model", "")
        cfg["explorer"] = (
            (f"{provider}:{model}" if model else provider) if provider
            else type(explorer).__name__
        )
    return cfg


def _log_generated_tools(run_logger: Any, cycle: int, module: str, agent: Any) -> None:
    """Snapshot an agent's active generated-tool registry for *cycle*.

    Reads ``_generated_probes`` (list of ``(generator, probe)`` tuples) for the
    probe agent or ``_generated_tools`` (list of tools) for the stats agent and
    forwards the bare tool objects to :meth:`RunLogger.log_tool_registry`.
    Best-effort: silently does nothing for agents without those attributes.
    """
    if agent is None:
        return
    raw = getattr(agent, "_generated_probes", None)
    if raw is None:
        raw = getattr(agent, "_generated_tools", None)
    if not raw:
        return
    tools = [t[1] if isinstance(t, tuple) else t for t in raw]
    try:
        run_logger.log_tool_registry(cycle, module, tools)
    except Exception as exc:  # noqa: BLE001
        logger.debug("log_tool_registry failed: %s", exc)


def _coerce_explore_context(value: "Any | None") -> "Any | None":
    """Normalise an explore_report argument into an ExploreContext or None.

    Accepts an ``ExploreContext``, a report dict (e.g. ``fused_report.json``),
    or ``None``. Never raises — a malformed value degrades to ``None``."""
    if value is None:
        return None
    try:
        from evalrx.eval_agent.stages.diagnosis import ExploreContext

        if isinstance(value, ExploreContext):
            return None if value.is_empty else value
        if isinstance(value, dict):
            return ExploreContext.from_report(value)
    except Exception as exc:  # explore context is optional, never fatal
        logger.warning("could not build ExploreContext from explore_report: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Internal adapter: convert HypothesisTestResult → InterventionResult-like
# object for RunLogger.log_surgery (which expects an InterventionResult).
# ---------------------------------------------------------------------------

def _make_intervention_result_from_test(tr: "Any") -> Any:
    """Wrap a HypothesisTestResult as a minimal InterventionResult."""
    from evalrx.eval_agent.stages.surgery import InterventionResult

    return InterventionResult(
        hypothesis=tr.hypothesis,
        status=tr.status,
        fixed=False,
        evidence={
            "m4_test_name": tr.test_name,
            "m4_effect_size": tr.effect_size,
            "m4_confidence": tr.confidence,
            "m4_protocol_consistent": tr.is_consistent_with_protocol,
            "m4_verdict": tr.verdict,
            "m4_evidence_grade": tr.evidence_grade,
            **tr.evidence,
        },
        confidence_score=tr.confidence,
    )
