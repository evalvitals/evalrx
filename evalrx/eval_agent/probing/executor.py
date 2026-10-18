"""Strict execution of one :class:`~evalrx.eval_agent.probing.actions.ProbeAction`.

Unlike ``ProbeAgent.probe``, nothing here selects, filters, backfills, caps or
generates: the analyzer runs once on exactly the planned cases, and the
outcome is typed instead of swallowed —

* ``unsupported`` — a required capability or media slot is missing; checked
  before anything runs, so no model call is spent;
* ``unfaithful`` — the edit mechanism's own self-check failed
  (:class:`~evalrx.analyzers._internals.attn_tap.FidelityError`);
* ``failed`` — any other exception;
* ``ok`` — findings, plus which planned cases actually produced a row.

Only ``ok`` carries findings; the other outcomes carry the reason, never a zero.

An analyzer that measures one case per call (``self_consistency`` reads only
``cases[0]``) runs with ``per_case=True``: once per planned case, each call's
findings becoming that case's ``per_case`` row. Artifacts named in
``keep_artifacts`` (e.g. the raw samples) are kept with the findings, since
the triage reads answers out of them.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from evalrx.eval_agent.probing.actions import (
    FAILED,
    OK,
    UNFAITHFUL,
    UNSUPPORTED,
    Observation,
    ProbeAction,
    canonical,
)
from evalrx.eval_agent.probing.metering import (
    MeteredModel,
    gpu_peak_mb,
    gpu_peak_reset,
    usage_counters,
    usage_delta,
)

if TYPE_CHECKING:
    from evalrx.core.analyzer import Analyzer
    from evalrx.core.case import CaseBatch


def gate(analyzer: "Analyzer", model: Any, cases: "CaseBatch") -> "str | None":
    """Why *analyzer* cannot run here, or ``None`` when its preconditions hold."""
    have = getattr(model, "capabilities", frozenset()) or frozenset()
    missing = sorted(c.value for c in set(getattr(analyzer, "requires", ())) - set(have))
    if missing:
        return f"model lacks capabilities {missing}"
    slots = set(getattr(analyzer, "requires_modalities", ()) or ())
    if slots and not any(getattr(c.inputs, s, None) is not None for c in cases for s in slots):
        return f"no planned case fills a {sorted(slots)} slot"
    return None


#: Where each kind of self-check reports its verdict in the findings.
FIDELITY_KEYS = {"edit": "tap_fidelity", "patch": "patch_fidelity"}


def _fidelity(check: "str | None", findings: dict[str, Any]) -> dict[str, Any]:
    if check is None:
        return {"status": "not_applicable"}
    verdict = findings.get(FIDELITY_KEYS[check])
    if not isinstance(verdict, dict):
        return {"status": "unchecked", "reason": "analyzer reported no fidelity verdict"}
    faithful = verdict.get("faithful", verdict.get("tap_faithful"))
    return {"status": "passed" if faithful else "failed", "verdict": canonical(verdict)}


def _kept(result: Any, names: "tuple[str, ...]") -> dict[str, Any]:
    artifacts = getattr(result, "artifacts", None) or {}
    return {name: canonical(artifacts[name]) for name in names if name in artifacts}


def _run_per_case(analyzer: "Analyzer", model: Any, cases: "CaseBatch",
                  keep_artifacts: "tuple[str, ...]") -> dict[str, Any]:
    """One call per case; each call's findings (minus ``_`` notes) become its row."""
    from evalrx.core.case import CaseBatch

    rows, notes = [], {}
    for case in cases:
        result = analyzer.run(model, CaseBatch([case]))
        findings = canonical(result.findings or {})
        row = {k: v for k, v in findings.items() if not k.startswith("_") and k != "per_case"}
        row["sample_id"] = str(case.id)
        if keep_artifacts:
            row["artifacts"] = _kept(result, keep_artifacts)
        notes.update({k: v for k, v in findings.items() if k.startswith("_") and k not in notes})
        rows.append(row)
    return {"per_case": rows, "per_case_mode": True, **notes}


def execute(action: ProbeAction, action_id: str, attempt: int, analyzer: "Analyzer",
            model: Any, cases: "CaseBatch", *, fidelity_check: "str | None" = None,
            per_case: bool = False, keep_artifacts: "tuple[str, ...]" = (),
            run_logger: Any = None) -> Observation:
    """Run *analyzer* on *cases* (= ``action.case_ids``) once and type the outcome.

    With *per_case*, the analyzer runs once per case instead (see the module
    docstring); a raise on any case types the whole action, as in batch mode.
    """
    from evalrx.analyzers._internals.attn_tap import FidelityError
    from evalrx.core.capability import CapabilityError

    planned = [str(c.id) for c in cases]
    if planned != list(action.case_ids):
        raise ValueError(f"{action.analyzer}: executor got cases {planned}, planned {list(action.case_ids)}")
    obs = Observation(
        action_id=action_id, analyzer=action.analyzer, attempt=attempt, status=FAILED,
        params=canonical(analyzer.get_params()), planned_ids=planned,
        fidelity={"status": "not_applicable"} if fidelity_check is None
        else {"status": "unchecked", "reason": "action did not reach the check"},
    )
    why = gate(analyzer, model, cases)
    if why is not None:
        obs.status, obs.error = UNSUPPORTED, why
        obs.cost = {"wall_sec": 0.0, "calls": {}, "model_sec": 0.0, "gpu_peak_mb": None}
        return obs

    target = model
    if run_logger is not None:
        from evalrx.eval_agent.model_instrumentation import InstrumentedModel

        target = InstrumentedModel(
            model, run_logger, cycle=0, analyzer=action.analyzer,
            case_prompts={str(c.inputs.prompt): str(c.id) for c in cases},
            batch_case_ids=planned,
        )
    metered = MeteredModel(target)
    measured_gpu = gpu_peak_reset()
    usage_before = usage_counters(model)
    t0 = time.monotonic()
    try:
        if per_case:
            raw = _run_per_case(analyzer, metered, cases, tuple(keep_artifacts))
        else:
            result = analyzer.run(metered, cases)
            raw = result.findings or {}
            if keep_artifacts:
                raw = {**raw, "artifacts": _kept(result, tuple(keep_artifacts))}
    except CapabilityError as exc:
        obs.status, obs.error_type, obs.error = UNSUPPORTED, type(exc).__name__, str(exc)[:500]
    except FidelityError as exc:
        obs.status, obs.error_type, obs.error = UNFAITHFUL, type(exc).__name__, str(exc)[:500]
        obs.fidelity = {"status": "failed", "verdict": canonical(exc.verdict)}
    except Exception as exc:  # noqa: BLE001 - typed outcome, not a crash of the controller
        obs.status, obs.error_type, obs.error = FAILED, type(exc).__name__, str(exc)[:500]
    else:
        findings = canonical(raw)
        obs.status, obs.findings = OK, findings
        obs.measured_ids = [i for i in planned if i in obs.per_case()]
        skips = findings.get("skip_reasons") or {}
        note = f"no per_case row (analyzer skip_reasons: {skips})" if skips else "no per_case row"
        obs.missing = {i: note for i in planned if i not in set(obs.measured_ids)}
        obs.fidelity = _fidelity(fidelity_check, findings)
    obs.cost = {
        "wall_sec": round(time.monotonic() - t0, 3),
        "calls": dict(metered.calls),
        "call_errors": dict(metered.errors),
        "model_sec": round(metered.model_sec, 3),
        "gpu_peak_mb": gpu_peak_mb() if measured_gpu else None,
        # what the provider counted (API models): requests, retries, tokens
        "api_usage": usage_delta(usage_before, usage_counters(model)),
    }
    return obs
