"""Verdicts and next-step recommendations for validated repair candidates.

Pure functions over :class:`~.types.FixValidation` results: the triage
label of one candidate, multiplicity control across the tested family, and
what to do when nothing validated (retry execution, gather more failures, or
escalate the intervention tier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from evalrx.eval_agent.stages.fix.internals import INTERNALS_PRIMITIVES
from evalrx.eval_agent.stages.fix.repair_catalog import discover_methods, supports_tier
from evalrx.eval_agent.stages.fix.tiers import FixTier
from evalrx.eval_agent.stages.fix.types import FixValidation
from evalrx.stats.ebh import ebh
from evalrx.stats.evalue import evalue_bernoulli

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


def verdict(v: FixValidation) -> str:
    """Coarse triage label (defect 4): why a candidate did/didn't pass."""
    if v.fixed:
        return "fixed"
    if v.reject and (v.effect or 0.0) < 0:
        return "regressed"  # significantly worse
    net = v.n_fixed - v.n_broken
    if net > 0:
        return "partial"  # helped more than hurt, not significant
    if v.n_broken > v.n_fixed:
        return "unsafe"  # breaks more than it fixes
    return "no_effect"


def ebh_survivors(tested: "list[FixValidation]", alpha: float) -> "set[int]":
    """id()s of validations whose e-value survives e-BH across the family.

    The e-values come from compare() (the validated core), never from an
    LLM, so e-BH is sound. With m candidates, the sole-survivor bar rises to
    m/alpha (vs the per-candidate 1/alpha) — the correct best-of-N tax.
    A single tested candidate (m=1) reduces to the per-candidate gate, so
    non-competitive runs are unaffected.
    """
    if not tested:
        return set()
    # Callers pass only validations that carry an e-value.
    evalues = [float(cast(float, v.e_value)) for v in tested]
    return {id(tested[i]) for i in ebh(evalues, alpha=alpha)}


def refine_signal(
    attempted: "list[FixValidation]", data: "CaseBatch"
) -> "dict[str, Any] | None":
    """Heterogeneity feedback for re-diagnosis (defect 3).

    A candidate that repairs one subset while breaking another is evidence
    the failure mode is *not* homogeneous: the right next move is to split
    the population by sub-mechanism and re-diagnose, not to keep proposing
    whole-population transforms.  Surface the partition so the loop (or a
    human) can re-scope the hypothesis.
    """
    split = [v for v in attempted if v.n_fixed > 0 and v.n_broken > 0]
    if not split:
        return None
    v = max(split, key=lambda x: min(x.n_fixed, x.n_broken))
    return {
        "kind": "heterogeneous_failure_mode",
        "candidate": v.candidate.name,
        "helped_cases": list(v.fixed_cases),
        "hurt_cases": list(v.broken_cases),
        "message": (
            f"{v.candidate.name!r} repaired {v.n_fixed} case(s) but broke "
            f"{v.n_broken} — the failure mode is likely subset-specific. "
            "Re-diagnose: what distinguishes the helped cases from the hurt "
            "ones, and gate the fix on that predicate."
        ),
    }


def min_failures_for_power(alpha: float) -> int:
    """Smallest n where a flawless fix could clear the e-value gate."""
    threshold = 1.0 / alpha
    for n in range(1, 200):
        if evalue_bernoulli(n, n, p0=0.5) >= threshold:
            return n
    return 200


def no_fix_recommendation(
    attempted: "list[FixValidation]",
    routed_tiers: "list[FixTier]",
    data: "CaseBatch",
    model: "Model",
    *,
    max_tier: FixTier,
    alpha: float,
    baseline_rates: "dict[str, float | None]",
    paired_rates: bool,
    allow_adapted: bool = False,
    allow_codegen: bool = True,
) -> "dict[str, Any] | None":
    """Decide what 'no candidate validated' actually means.

    Three distinct causes the old single-path recommendation conflated:

    * **never executed** — engineering failure; retry within the tier, do
      not escalate (a crash is not evidence the tier is exhausted).
    * **underpowered by design** — even a flawless fix of every failure
      could not reach significance with this few failures; gather more
      failing cases instead of climbing the (more invasive) tier ladder.
    * **genuinely exhausted** — executed, powered, still no fix; escalate.
    """
    executed = [v for v in attempted if v.n_pairs > 0]
    # A model-independent candidate DID execute; it just is not a repair of
    # the model. It is neither an engineering failure nor evidence the tier
    # is exhausted, so it sits in neither list.
    never_ran = [v for v in attempted
                 if v.n_pairs == 0 and v.verdict != "model_independent"]
    if never_ran and not executed:
        return {
            "recommend_tier": max_tier.label,
            "action": "fix_execution",
            "reason": (
                "no candidate EXECUTED — escalating would be premature; "
                f"fix candidate execution and retry within {max_tier.label}. "
                "Failures: "
                + "; ".join(
                    f"{v.candidate.name}: {(v.exec_error or v.summary)[:120]}"
                    for v in never_ran[:3]
                )
            ),
        }

    # Underpowered-by-design: with n_fail failures the best possible result
    # (every failure repaired, nothing broken) yields an e-value ceiling of
    # evalue_bernoulli(n_fail, n_fail); if that cannot clear 1/alpha, no fix
    # of any tier can be certified here — the bottleneck is sample size, and
    # a "promising" candidate (helped more than it hurt) confirms the lead.
    n_fail = sum(1 for c in data if getattr(c.label, "value", None) == "fail")
    if paired_rates:
        # Paired rates: the best any candidate can do is lift every case's
        # rate to 1.0 -> the e-value of those differences is the ceiling.
        from evalrx.stats.evalue import evalue_bounded_mean

        diffs = [
            1.0 - r for c in data
            for r in [baseline_rates.get(c.id)] if r is not None
        ]
        ceiling = evalue_bounded_mean(diffs) if diffs else 1.0
    else:
        ceiling = evalue_bernoulli(n_fail, n_fail, p0=0.5) if n_fail > 0 else 1.0
    promising = [v for v in executed if (v.n_fixed - v.n_broken) > 0 and not v.reject]
    if ceiling < 1.0 / alpha:
        need = min_failures_for_power(alpha)
        evidence = ""
        if promising:
            best = max(
                promising,
                key=lambda v: (v.n_fixed - v.n_broken, v.effect or 0.0),
            )
            evidence = (
                f" {best.candidate.name!r} already helps net "
                f"{best.n_fixed - best.n_broken} case(s);"
            )
        return {
            "recommend_tier": None,
            "action": "gather_more_failures",
            "reason": (
                f"underpowered by design: only {n_fail} failure case(s) — even a "
                f"perfect fix tops out at e={ceiling:.1f} (< {1.0 / alpha:.0f} "
                f"needed).{evidence} collect >= {need} failing "
                "cases and re-validate before escalating the tier."
            ),
        }

    rec = recommend(
        routed_tiers, max_tier, model=model, data=data,
        allow_adapted=allow_adapted, allow_codegen=allow_codegen,
    )
    if promising and rec is not None:
        best = max(promising, key=lambda v: (v.n_fixed - v.n_broken, v.effect or 0.0))
        eff = f"{best.effect:+.3f}" if best.effect is not None else "n/a"
        ev = f"{best.e_value:.1f}" if best.e_value is not None else "n/a"
        rec["promising"] = {
            "candidate": best.candidate.name,
            "n_fixed": best.n_fixed,
            "n_broken": best.n_broken,
            "effect": best.effect,
            "e_value": best.e_value,
        }
        rec["reason"] += (
            f" (note: {best.candidate.name!r} is INCONCLUSIVE, not refuted — "
            f"{best.n_fixed} fixed / {best.n_broken} broken, effect {eff}, "
            f"e={ev} < {1.0 / alpha:.0f}; re-validating it on more or cleaner "
            "pairs (baseline_repeats>1 to drop sampling-unstable cases, or a larger "
            "confirm split) is a cheaper next step than escalating the tier)"
        )
    solo = [v for v in attempted if v.verdict == "model_independent"]
    if solo and rec is not None:
        rec["reason"] += (
            f" (note: {len(solo)} candidate(s) solved the task without the model "
            "and were not counted as repairs: "
            + ", ".join(v.candidate.name for v in solo[:3])
            + ")"
        )
    if never_ran and rec is not None:
        rec["reason"] += (
            f" (caveat: {len(never_ran)} candidate(s) never executed: "
            + ", ".join(v.candidate.name for v in never_ran[:3])
            + ")"
        )
    return rec


def recommend(
    routed: "list[FixTier]",
    max_tier: FixTier,
    *,
    model: "Model | None" = None,
    data: "CaseBatch | None" = None,
    reason_prefix: str = "",
    allow_adapted: bool = False,
    allow_codegen: bool = True,
) -> "dict[str, Any] | None":
    above = sorted(t for t in routed if t > max_tier)
    if above:
        target = above[0]
        reason = (
            f"verified hypotheses route to {target.label} "
            f"({target.describe()}), beyond the allowed {max_tier.label}"
        )
    elif max_tier < FixTier.L4_PARAMETERS:
        target = FixTier(max_tier + 1)
        reason = (
            f"no candidate within {max_tier.label} validated; the next "
            f"intervention space is {target.describe()}"
        )
    else:
        return None  # already at L4 — nothing above to recommend
    skipped: "list[str]" = []
    while (
        model is not None
        and target
        in {
            FixTier.L3A_INTERNALS_READ,
            FixTier.L3B_INTERNALS_WRITE,
        }
        and not tier_available(
            target, model, data=data,
            allow_adapted=allow_adapted, allow_codegen=allow_codegen,
        )
    ):
        skipped.append(target.label)
        if target >= FixTier.L4_PARAMETERS:
            return None
        target = FixTier(target + 1)
    if skipped:
        reason += (
            f"; skipped unsupported tier(s) {', '.join(skipped)} for this model "
            f"and routed to {target.describe()}"
        )
    if reason_prefix:
        reason = f"{reason_prefix}; {reason}"
    return {"recommend_tier": target.label, "reason": reason}


def tier_available(
    tier: FixTier,
    model: "Model",
    *,
    data: "CaseBatch | None" = None,
    allow_adapted: bool = False,
    allow_codegen: bool = True,
) -> bool:
    """Whether an invasive tier has an executor compatible with this task."""
    if data is not None and tier in {
        FixTier.L3A_INTERNALS_READ,
        FixTier.L3B_INTERNALS_WRITE,
    }:
        has_images = any(
            getattr(c.inputs, "image", None) is not None
            or getattr(c.inputs, "video", None) is not None
            for c in data
        )
        has_audio = any(getattr(c.inputs, "audio", None) is not None for c in data)
        tasks = {str((getattr(c, "metadata", {}) or {}).get("task", "")) for c in data}
        registered = any(
            method.tier == tier
            for method in discover_methods(
                model,
                max_tier=tier,
                has_images=has_images,
                has_audio=has_audio,
                tasks=tasks,
                allow_adapted=allow_adapted,
            )
        )
        if registered:
            return True
        if tier == FixTier.L3A_INTERNALS_READ:
            from evalrx.core.capability import Capability

            return bool(
                allow_codegen
                and Capability.ATTENTION in getattr(model, "capabilities", frozenset())
            )
        return bool(
            has_images
            and any(
                primitive.tier == tier and primitive.available(model)
                for primitive in INTERNALS_PRIMITIVES.values()
            )
        )
    if tier == FixTier.L3A_INTERNALS_READ:
        from evalrx.core.capability import Capability

        return (
            Capability.ATTENTION in getattr(model, "capabilities", frozenset())
            or supports_tier(model, tier)
        )
    if tier == FixTier.L3B_INTERNALS_WRITE:
        return any(
            primitive.tier == tier and primitive.available(model)
            for primitive in INTERNALS_PRIMITIVES.values()
        )
    return True
