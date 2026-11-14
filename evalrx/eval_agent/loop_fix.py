"""The fix-stage orchestration behind :meth:`VLDiagnoseLoop.run_fix`.

Which hypotheses a fix targets, the tier ladder, the EXPLORE→CONFIRM
split, and the run-log record of the outcome. Free functions over the loop's
report and fix agent; ``VLDiagnoseLoop.run_fix`` / ``run_m5`` call them.
This module does not import :mod:`evalrx.eval_agent.loop`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from evalrx.core.case import CaseBatch

logger = logging.getLogger(__name__)


def _call_supported(fn: "Any", *args: Any, required: "tuple[str, ...]" = (),
                    **kwargs: Any) -> "Any":
    """Call *fn* with the non-None *kwargs* it accepts.

    Custom/stub fix agents keep working with optional extras such as
    ``prior_attempts`` / ``context`` / ``record`` (same pattern as
    ``_diagnose_with_optional_context``).  A *required* keyword the function
    cannot take raises instead of being dropped: running a tier ladder on an
    agent that ignores the tier window would silently test the wrong tier.
    """
    import inspect as _inspect

    params: "Mapping[str, _inspect.Parameter]"
    try:
        params = _inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    accepts_any = any(p.kind == p.VAR_KEYWORD for p in params.values())
    passed = {k: v for k, v in kwargs.items()
              if v is not None and (accepts_any or k in params)}
    missing = [k for k in required if kwargs.get(k) is not None and k not in passed]
    if missing:
        raise TypeError(
            f"{getattr(fn, '__qualname__', fn)} does not accept {', '.join(missing)}; "
            "the fix loop needs it to restrict the tier window"
        )
    return fn(*args, **passed)


def _propose_and_validate(agent: "Any", model: "Any", data: "Any", hypotheses: "Any",
                          **kwargs: Any) -> "Any":
    """``agent.propose_and_validate`` with a per-call tier window (required when
    given) and optional extras (see :func:`_call_supported`)."""
    return _call_supported(agent.propose_and_validate, model, data, hypotheses,
                           required=("max_tier", "min_tier"), **kwargs)


def _unverified_hypotheses(report: "Any") -> "list[Any]":
    """Best-first UNVERIFIED hypotheses: M4-tested and not refuted, highest
    confidence first; then untested proposals from the last cycle."""
    seen: set[str] = set()
    out: list[Any] = []
    tested = list(getattr(report, "all_test_results", None) or [])

    def _critic_ok(tr: "Any") -> int:
        meta = getattr(getattr(tr, "hypothesis", None), "metadata", None) or {}
        return 0 if meta.get("critic") == "reject" else 1

    ranked = sorted(
        tested,
        key=lambda tr: (float(getattr(tr, "confidence", 0.0) or 0.0), _critic_ok(tr)),
        reverse=True,
    )
    for tr in ranked:
        h = getattr(tr, "hypothesis", None)
        if h is None or _hyp_key(h) in seen:
            continue
        seen.add(_hyp_key(h))  # tested (in any way) -> never re-added as "untested"
        status = getattr(tr, "status", None)
        status_s = str(getattr(status, "value", status) or "").lower()
        if status_s == "refuted":
            continue
        out.append(h)
    for h in reversed(list(getattr(report, "final_hypotheses", None) or [])):
        if _hyp_key(h) not in seen:
            seen.add(_hyp_key(h))
            out.append(h)
    return out


def _m5_supported_key(report: "Any") -> "str | None":
    """Key of the hypothesis M5's experiment SUPPORTED, if any."""
    iv = getattr(report, "fix_proposal", None)
    status = getattr(iv, "status", None)
    status_s = str(getattr(status, "value", status) or "").lower()
    hyp = getattr(iv, "hypothesis", None)
    if iv is None or hyp is None or status_s != "supported":
        return None
    return _hyp_key(hyp)


def _hyp_key(hypothesis: "Any") -> str:
    """Identity of a hypothesis for matching across M4/M5 results."""
    hid = str(getattr(hypothesis, "id", "") or "")
    return hid or str(getattr(hypothesis, "statement", hypothesis))


def _m5_refuted(report: "Any") -> "tuple[set[str], list[str]]":
    """Hypotheses M5's intervention experiment REFUTED, with a one-line why.

    Reads ``report.fix_proposal`` (an ``InterventionResult`` from ``run_m5``).
    Returns ``(keys, notes)``; both empty when M5 did not run or did not refute.
    """
    iv = getattr(report, "fix_proposal", None)
    if iv is None:
        return set(), []
    status = getattr(iv, "status", None)
    status_s = str(getattr(status, "value", status) or "").lower()
    if status_s != "refuted":
        return set(), []
    hyp = getattr(iv, "hypothesis", None)
    if hyp is None:
        return set(), []
    statement = str(getattr(hyp, "statement", hyp))
    evidence = getattr(iv, "evidence", None) or {}
    scalars = []
    if isinstance(evidence, dict):
        for k, v in evidence.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                scalars.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
            if len(scalars) >= 8:
                break
    why = f" [experiment: {', '.join(scalars)}]" if scalars else ""
    return {_hyp_key(hyp)}, [f"{statement}{why}"]


def _fix_context_from_report(
    report: "Any",
    *,
    example_cases: "Any | None",
    explore_context: "Any | None",
    protocol: "Any | None",
    refuted: "list[str]",
    refuted_ids: "set[str] | None" = None,
) -> "Any":
    """Build the :class:`~evalrx.eval_agent.stages.fix.agent.FixContext`
    the proposer sees: M4 verdicts + M2 conclusion/tests + exploratory notes,
    the M5-refuted hypotheses, and (when a confirm split is in play) the
    EXPLORE cases in full."""
    from evalrx.eval_agent.stages.fix.types import FixContext

    lines: list[str] = []
    refuted_ids = set(refuted_ids or ())
    verified = [
        tr for tr in list(getattr(report, "verified_hypotheses", None) or [])
        if _hyp_key(getattr(tr, "hypothesis", None)) not in refuted_ids
    ]
    if verified:
        lines.append("  M4 verified hypotheses (statistical tests on the diagnosis split):")
        for tr in verified[:6]:
            stmt = str(getattr(getattr(tr, "hypothesis", None), "statement", ""))[:220]
            verdict = str(getattr(tr, "verdict", "") or "")[:300]
            conf = getattr(tr, "confidence", None)
            grade = getattr(tr, "evidence_grade", "")
            conf_s = f" conf={conf:.2f}" if isinstance(conf, (int, float)) else ""
            lines.append(f"    - {stmt}{conf_s} grade={grade}")
            if verdict:
                lines.append(f"        evidence: {verdict}")
    stats = getattr(report, "final_stats_report", None)
    if stats is not None:
        conclusion = str(getattr(stats, "conclusion", "") or getattr(stats, "narrative", "") or "")
        if conclusion:
            lines.append("  M2 conclusion:")
            lines.append("    " + conclusion.strip()[:1200])
        chain = list(getattr(stats, "evidence_chain", None) or [])
        if chain:
            lines.append("  M2 evidence chain:")
            lines += [f"    - {str(step)[:300]}" for step in chain[:8]]
        results = list(getattr(stats, "stats_results", None) or [])
        sig = [r for r in results if getattr(r, "ok", False) and getattr(r, "reject", False)]
        shown = sig[:8] or [r for r in results if getattr(r, "ok", False)][:5]
        if shown:
            lines.append("  M2 statistical tests" + (" (rejecting H0):" if sig else ":"))
            lines += [f"    - {str(getattr(r, 'summary', ''))[:300]}" for r in shown]
    if explore_context is not None and not getattr(explore_context, "is_empty", True):
        obs = list(getattr(explore_context, "observations", None) or [])
        cav = list(getattr(explore_context, "caveats", None) or [])
        if obs:
            lines.append("  Exploratory notes (free-form EDA, UNCONFIRMED — hints only):")
            lines += [f"    - {str(o)[:300]}" for o in obs[:10]]
        if cav:
            lines.append("  Explorer caveats:")
            lines += [f"    - {str(c)[:200]}" for c in cav[:5]]
    task_bits = []
    for attr in ("description", "task_domain", "failure_patterns"):
        value = str(getattr(protocol, attr, "") or "").strip()
        if value:
            task_bits.append(value)
    return FixContext(
        example_cases=example_cases,
        evidence="\n".join(lines),
        refuted=list(refuted),
        task_note=" — ".join(task_bits),
    )


def _fix_hypotheses(
    report: "Any", allow_unverified: bool
) -> "tuple[list[Any], str, set[str], list[str]]":
    """What the fix stage may build on: ``(hypotheses, note, refuted_ids, refuted_notes)``.

    M5's intervention experiment (run_m5) can REFUTE the very hypothesis M4
    verified. A refuted hypothesis must not reach the proposer as "verified":
    on qwen3.5-2b/bbh_tracking7 the fix judge/coder were told the M5-refuted
    grading-mismatch hypothesis was verified and half the coded pipeline's
    design served it. Refuted ones are dropped and passed to the proposer as
    "do not build on these".

    With *allow_unverified*, strong inconclusive leads are kept even when M4
    verified a different, narrower hypothesis, so a secondary supported
    symptom cannot crowd the plausible causal mechanisms out of the search.
    They stay marked as leads in the note; only candidate validation decides.
    """
    refuted_ids, refuted_notes = _m5_refuted(report)
    hypotheses = [
        tr.hypothesis for tr in report.verified_hypotheses
        if _hyp_key(tr.hypothesis) not in refuted_ids
    ]
    note = ""
    if allow_unverified:
        leads = [
            h for h in _unverified_hypotheses(report)
            if _hyp_key(h) not in refuted_ids
            and _hyp_key(h) not in {_hyp_key(v) for v in hypotheses}
        ][:3]
        had_verified = bool(hypotheses)
        hypotheses.extend(leads)
        supported = _m5_supported_key(report)
        if leads:
            note = (
                ("MIXED EVIDENCE: verified hypotheses come first; the remaining "
                 if had_verified else
                 "UNVERIFIED: M4 found no statistically significant evidence for these ")
                + "hypotheses; they are best-scoring leads, not established "
                "mechanisms"
                + ("; the M5 intervention experiment SUPPORTED the first lead"
                   if supported and _hyp_key(leads[0]) == supported else "")
                + ". Treat inconclusive leads only as hints about WHERE to intervene; "
                "the final candidate validation decides."
            )
    return hypotheses, note, refuted_ids, refuted_notes


def _climb_ladder(
    agent: "Any",
    model: "Any",
    data: "CaseBatch",
    hypotheses: "list[Any]",
    context: "Any",
    ceiling: "Any",
    label: str,
) -> "list[Any]":
    """Propose-and-validate one tier at a time, L0 up to *ceiling* (at most L3b).

    Each rung sees only its own tier (cheaper candidates are not pooled with a
    newly opened, more invasive tier) plus every failed attempt so far, so the
    judge can try something fundamentally different.  Stops at the first fix.
    Rungs record nothing; the caller records one combined outcome.  An opaque
    custom-agent result ends the climb as the last rung.
    """
    from evalrx.eval_agent.stages.fix.tiers import FixTier

    ladder = [
        FixTier.L0_RUNTIME_CONFIG,
        FixTier.L1_PROMPT,
        FixTier.L2_SCAFFOLD,
        FixTier.L3A_INTERNALS_READ,
        FixTier.L3B_INTERNALS_WRITE,
    ]
    rungs: "list[Any]" = []
    prior: "list[Any]" = []
    for tier in ladder:
        if tier > ceiling:
            break
        logger.info("run_fix: %s trying tier %s (%d prior attempt(s))",
                    label, tier.label, len(prior))
        rung = _propose_and_validate(
            agent, model, data, hypotheses,
            prior_attempts=prior if prior else None,
            context=context,
            max_tier=tier,
            min_tier=tier,
            record=False,
        )
        rungs.append(rung)
        if not hasattr(rung, "attempted"):
            break
        if rung.fixed:
            logger.info("run_fix: %s fixed at tier %s", label, tier.label)
            break
        prior.extend(v for v in rung.attempted if not v.fixed)
        logger.info("run_fix: %s tier %s exhausted — escalating", label, tier.label)
    return rungs


def _merge_rungs(rungs: "list[Any]", ceiling: "Any", alpha: float) -> "Any":
    """One outcome for a climb with no held-out partition.

    The merged set spans every escalated tier, so it is a LARGER best-of-N
    family than any single tier — e-BH FDR control is re-applied over the
    whole union (mirrors FixAgent.propose_and_validate) instead of an
    uncorrected max, or escalation would re-open the multiplicity leak.
    The last rung's outcome carries the merge.
    """
    from evalrx.eval_agent.stages.fix.verdict import ebh_survivors

    outcome = rungs[-1]
    outcome.attempted = [v for rung in rungs for v in rung.attempted]
    outcome.max_tier = ceiling
    tested = [v for v in outcome.attempted if v.e_value is not None]
    survivors = ebh_survivors(tested, alpha)
    outcome.ebh_survivors = sorted(v.candidate.name for v in tested if id(v) in survivors)
    winners = [v for v in outcome.attempted if v.fixed and id(v) in survivors]
    outcome.best = max(winners, key=lambda v: (v.effect or 0.0, -v.n_broken), default=None)
    outcome.fixed = outcome.best is not None
    return outcome


def _log_fix(run_logger: "Any | None", outcome: "Any") -> None:
    """Record the one combined fix outcome the unrecorded rounds left out."""
    if run_logger is None:
        return
    try:
        run_logger.log_fix(outcome)
    except Exception as exc:  # logging must never break the fix step
        logger.debug("run_fix: log_fix failed: %s", exc)


def _select_on_explore(attempted: "list[Any]") -> "Any | None":
    """The strongest EXPLORE improvement with enough support, or None."""
    executed = [
        validation
        for validation in attempted
        if validation.n_pairs > 0 and validation.effect is not None
    ]
    improving = [
        validation
        for validation in executed
        if validation.n_fixed > validation.n_broken
    ]
    # A spectacular percentage on two applicable examples is a fragile lead,
    # not a better selection than a supported improvement on dozens.  Require
    # up to eight EXPLORE pairs when that support exists anywhere in the
    # candidate family; small experiments still select from what they have.
    support_floor = min(8, max((v.n_pairs for v in improving), default=0))
    supported = [v for v in improving if v.n_pairs >= support_floor]
    return max(
        supported,
        key=lambda validation: (
            validation.e_value or 0.0,
            validation.n_fixed - validation.n_broken,
            validation.n_pairs,
            validation.effect or 0.0,
            -validation.n_broken,
            -int(validation.candidate.tier),
        ),
        default=None,
    )


def _selection_audit(attempted: "list[Any]") -> "list[dict[str, Any]]":
    """One row per EXPLORE candidate, as the contract emitter reads it."""
    from evalrx.eval_agent.stages.fix.descriptions import plain_description

    # `headline` travels with the audit row because the candidate object does
    # not: the contract emitter sees only these dicts, and a slug is the one
    # thing it must not show a reader.
    return [
        {
            "name": validation.candidate.name,
            "headline": plain_description(validation.candidate),
            "tier": validation.candidate.tier.label,
            "n_pairs": validation.n_pairs,
            "n_fixed": validation.n_fixed,
            "n_broken": validation.n_broken,
            "effect": validation.effect,
            "e_value": validation.e_value,
            "coverage": validation.coverage,
            "verdict": validation.verdict,
        }
        for validation in attempted
    ]


def _recommend_tier(
    agent: "Any",
    model: "Any",
    confirm: "CaseBatch",
    validations: "list[Any]",
    *,
    reason: str,
    max_tier: "Any",
) -> "dict[str, Any] | None":
    """The agent's next-tier recommendation, or the ceiling with *reason*."""
    routed = sorted({v.candidate.tier for v in validations})
    recommend = getattr(agent, "recommend", None)
    if callable(recommend):
        try:
            return _call_supported(
                recommend,
                routed,
                model=model,
                data=confirm,
                reason_prefix=reason,
                max_tier=max_tier,
            )
        except TypeError:
            # Backward compatibility for application-specific FixAgents.
            pass
    return {"recommend_tier": max_tier.label, "reason": reason}


def _confirm_from_explore(
    agent: "Any",
    model: "Any",
    confirm: "CaseBatch",
    attempted: "list[Any]",
    *,
    max_tier: "Any",
    repair_rounds: int,
) -> "Any":
    """Freeze the strongest EXPLORE improvement and test it once on CONFIRM."""
    from evalrx.eval_agent.stages.fix.types import FixOutcome
    from evalrx.eval_agent.stages.fix.verdict import ebh_survivors, refine_signal

    selected = _select_on_explore(attempted)
    audit = _selection_audit(attempted)
    if selected is None:
        return FixOutcome(
            max_tier=max_tier,
            repair_rounds=repair_rounds,
            selection_attempted=audit,
            recommendation=_recommend_tier(
                agent, model, confirm, attempted,
                reason="no EXPLORE candidate had positive net repairs; CONFIRM was left untouched",
                max_tier=max_tier,
            ),
        )

    # The candidate is frozen; validate_candidate scores it on the complete
    # CONFIRM partition (the search budget ``max_validation_cases`` does not
    # apply to confirmation).
    validation = agent.validate_candidate(model, confirm, selected.candidate)
    survivors = ebh_survivors(
        [validation] if validation.e_value is not None else [], agent.alpha
    )
    survived = id(validation) in survivors
    outcome = FixOutcome(
        max_tier=max_tier,
        attempted=[validation],
        best=validation if validation.fixed and survived else None,
        fixed=bool(validation.fixed and survived),
        repair_rounds=repair_rounds,
        ebh_survivors=[validation.candidate.name] if survived else [],
        selection_attempted=audit,
        selected_on_explore=selected.candidate.name,
    )
    if not outcome.fixed:
        no_fix = getattr(agent, "no_fix_recommendation", None)
        if callable(no_fix):
            outcome.recommendation = _call_supported(
                no_fix, [validation], [selected.candidate.tier], confirm, model,
                max_tier=max_tier,
            )
            if outcome.recommendation is not None:
                outcome.recommendation["reason"] = (
                    "the candidate selected on EXPLORE did not validate on untouched CONFIRM; "
                    + outcome.recommendation["reason"]
                )
        else:
            outcome.recommendation = _recommend_tier(
                agent, model, confirm, [validation],
                reason="the candidate selected on EXPLORE did not validate on untouched CONFIRM",
                max_tier=max_tier,
            )
    outcome.refine_signal = refine_signal([validation], confirm)
    return outcome
