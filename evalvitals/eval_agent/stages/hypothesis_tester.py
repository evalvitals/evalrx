"""M5 — HypothesisTester: verify hypotheses via statistical tests + protocol consistency.

M5 is the *gatekeeper* between M3 (hypothesis generation) and the stopping
decision.  It asks two questions for each hypothesis:

1. **Statistical support** — do cases that exhibit the hypothesised signal fail
   at a higher rate than cases that do not?  M5 now *consumes M2's rigorous
   ``stats_results``* (effect size + CI + e-value, FDR-corrected via e-BH) when
   present, deciding SUPPORTED/REFUTED only on a corrected rejection.  When M2
   ran without labeled data (no ``stats_results``), it falls back to a rigorous
   clustered-bootstrap :func:`~evalvitals.stats.compare` on the extracted
   per-case signal — never a hand-rolled proportion difference.

2. **Protocol consistency** — is the hypothesis consistent with what the user
   described in their experiment protocol?  Uses keyword-based heuristics by
   default; an optional LLM ``judge=`` runs a critic call for a richer check.

**Stopping criteria** (Plan A from the 2026-06-05 meeting):
:meth:`stopping_criteria_met` returns ``True`` when at least one hypothesis is
*both* statistically supported *and* protocol-consistent.  The loop calls this
after each M5 pass and breaks when it returns ``True``.

Usage::

    tester = HypothesisTester()
    results = tester.test(hypotheses, stats_report, data, protocol=protocol)

    if tester.stopping_criteria_met(results, protocol):
        best = tester.best_hypotheses(results)
        # pass best[0].hypothesis to M4

    # With LLM judge for richer protocol consistency:
    tester = HypothesisTester(judge=gemini)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.core.case import CaseBatch, Label
from evalvitals.eval_agent.hypothesis import Hypothesis, HypothesisStatus
from evalvitals.eval_agent.prompts.hypothesis_tester import _CONSISTENCY_PROMPT
from evalvitals.eval_agent.stages.surgery import _extract_per_case_signals
from evalvitals.stats import compare

if TYPE_CHECKING:
    from evalvitals.analysis.stats_agent import StatsAnalysisReport
    from evalvitals.analysis.stats_tools import StatsToolResult
    from evalvitals.core.model import Model
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

logger = logging.getLogger(__name__)

# M2 tools whose effect is a directional per-mechanism signal (effect>0 = harmful:
# the signal group fails MORE). These can decide SUPPORTED/REFUTED for a hypothesis.
_SIGNAL_TOOLS = frozenset({"signal_label_assoc", "bootstrap_diff", "mcnemar_evalue"})
# Tools that describe the run globally rather than a specific mechanism — used as
# corroborating evidence only, never sufficient to confirm a specific hypothesis.
# attention_decoding is a tensor-level OMNIBUS ("do the maps differ anywhere?") —
# corroborating, not a directional per-mechanism verdict.
_GLOBAL_TOOLS = frozenset({"single_rate_evalue", "friedman_nemenyi", "rank_corr",
                           "attention_decoding"})
# Descriptive-only tools whose "effect" is an artifact of the batch composition,
# not a mechanism signal (e.g. single_rate_evalue's rate − p0 on a curated/
# enriched batch). They are consulted for context but MUST NOT become a
# hypothesis verdict's headline result — otherwise an inconclusive run reads as
# "best: FAIL rate 7% vs p0=0.50 → reject", which sounds like a finding and is
# meaningless once the batch is not a representative sample.
_DESCRIPTIVE_TOOLS = frozenset({"single_rate_evalue"})


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class HypothesisTestResult:
    """Result of M5 testing one :class:`~evalvitals.eval_agent.hypothesis.Hypothesis`.

    Attributes:
        hypothesis:                  The hypothesis under test.
        status:                      Statistical verdict
                                     (:class:`~evalvitals.eval_agent.hypothesis.HypothesisStatus`).
        test_name:                   Which test was used (e.g. ``"fail_rate_comparison"``).
        effect_size:                 Observed fail-rate difference
                                     (signal group minus control group).
        is_consistent_with_protocol: ``True`` when the hypothesis is relevant
                                     to the user's experiment protocol.
        confidence:                  Combined score in [0, 1] — geometric mean
                                     of evidence gap, sample adequacy, and
                                     control cleanliness.
        verdict:                     NL one-liner for human consumption.
        evidence_grade:              Strength tier of the deciding evidence:
                                     ``"intervention"`` (paired strategy contrast
                                     or intervention-derived signal — causal),
                                     ``"observational"`` (signal/label
                                     association), or ``"none"``.
        evidence:                    Supporting statistics and group sizes.
    """

    hypothesis: Hypothesis
    status: HypothesisStatus
    test_name: str
    effect_size: float | None
    is_consistent_with_protocol: bool
    confidence: float
    verdict: str
    evidence_grade: str = "observational"
    evidence: dict[str, Any] = field(default_factory=dict)


# Evidence-strength ordering for the P4 depth-tiered stopping criterion.
_GRADE_ORDER = {"none": 0, "observational": 1, "intervention": 2}


def _evidence_grade(tool: str, signal: str) -> str:
    """Grade the deciding evidence: paired contrasts and intervention-derived
    per-case signals are causal ("intervention"); plain analyzer-signal
    associations are correlational ("observational")."""
    if tool in {"mcnemar_evalue", "friedman_nemenyi"}:
        return "intervention"
    if str(signal).startswith("prompt_contrast."):
        return "intervention"
    return "observational"


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------

class HypothesisTester:
    """M5: test hypotheses using statistical methods and protocol consistency.

    Args:
        judge:       Optional LLM for protocol consistency checks.
                     Any :class:`~evalvitals.core.model.Model` with
                     ``Capability.GENERATE``.  When ``None``, uses a
                     keyword-based heuristic instead.
        alpha:       Statistical significance level used for the e-value /
                     bootstrap-CI decisions inherited from
                     :func:`~evalvitals.stats.compare` and for confidence
                     scaling.
        min_effect:  Minimum fail-rate difference to consider a finding
                     meaningful (default 0.10 = 10 pp).  Used by the rigorous
                     fallback path when M2 supplied no ``stats_results``.
        min_evidence_grade: Depth tier required for the stopping criterion
                     (P4).  ``"observational"`` (default) stops on any
                     statistically supported + protocol-consistent hypothesis;
                     ``"intervention"`` keeps the loop running until a
                     hypothesis is verified by intervention-grade evidence
                     (paired strategy contrast / intervention-derived signal),
                     letting cycle 2 collect the targeted evidence that M3's
                     test designs ask for.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        alpha: float = 0.05,
        min_effect: float = 0.10,
        min_evidence_grade: str = "observational",
    ) -> None:
        self._judge = judge
        self.alpha = alpha
        self.min_effect = min_effect
        if min_evidence_grade not in _GRADE_ORDER:
            raise ValueError(
                f"min_evidence_grade must be one of {sorted(_GRADE_ORDER)}, "
                f"got {min_evidence_grade!r}"
            )
        self.min_evidence_grade = min_evidence_grade

    def test(
        self,
        hypotheses: list[Hypothesis],
        stats_report: "StatsAnalysisReport",
        data: CaseBatch,
        protocol: "ExperimentProtocol | None" = None,
    ) -> list[HypothesisTestResult]:
        """Test each hypothesis against *data* and *protocol*.

        Args:
            hypotheses:   Hypotheses produced by M3 (DiagnosisAgent).
            stats_report: M2's report — provides ``raw_results`` for
                          per-case signal extraction.
            data:         Cases to test against (must carry labels for
                          fail-rate comparison).
            protocol:     Experiment protocol for consistency checks.
                          ``None`` → all hypotheses are assumed consistent.

        Returns:
            One :class:`HypothesisTestResult` per hypothesis, in the
            same order as *hypotheses*.
        """
        # The whole cycle shares one stats_report, so either ALL hypotheses take
        # the primary (M2 e-BH-corrected) path or ALL take the label-free
        # fallback. On the fallback path the family is every hypothesis, so each
        # directional verdict must clear a Bonferroni-corrected significance
        # level (alpha / family) — otherwise best-of-N over the fallback would
        # manufacture SUPPORTED verdicts the primary path's e-BH would have
        # blocked.
        on_fallback = not list(getattr(stats_report, "stats_results", None) or [])
        family_size = len(hypotheses) if on_fallback else 1

        results: list[HypothesisTestResult] = []
        for h in hypotheses:
            result = self._test_one(h, stats_report, data, protocol, family_size)
            results.append(result)
        return results

    def stopping_criteria_met(
        self,
        test_results: list[HypothesisTestResult],
        protocol: "ExperimentProtocol | None" = None,
    ) -> bool:
        """Return ``True`` when at least one verified, protocol-consistent hypothesis exists.

        This is the **Plan A stopping criterion** from the 2026-06-05
        architecture meeting: the loop should stop once we have a
        statistically supported hypothesis that addresses what the
        user's protocol was testing — further cycling would only
        rediscover the same root cause.

        With ``min_evidence_grade="intervention"`` (P4 depth tier), an
        observationally supported hypothesis does NOT stop the loop — the next
        cycle can collect the targeted (intervention) evidence that M3's test
        designs call for.
        """
        need = _GRADE_ORDER[self.min_evidence_grade]
        return any(
            r.status == HypothesisStatus.SUPPORTED
            and r.is_consistent_with_protocol
            and _GRADE_ORDER.get(r.evidence_grade, 0) >= need
            for r in test_results
        )

    def best_hypotheses(
        self,
        test_results: list[HypothesisTestResult],
    ) -> list[HypothesisTestResult]:
        """Return verified, protocol-consistent hypotheses, strongest evidence first.

        Sorted by evidence grade (intervention > observational), then
        confidence.  The first element is the candidate to hand to M4.
        """
        supported = [
            r for r in test_results
            if r.status == HypothesisStatus.SUPPORTED and r.is_consistent_with_protocol
        ]
        return sorted(
            supported,
            key=lambda r: (_GRADE_ORDER.get(r.evidence_grade, 0), r.confidence),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _test_one(
        self,
        hypothesis: Hypothesis,
        stats_report: "StatsAnalysisReport",
        data: CaseBatch,
        protocol: "ExperimentProtocol | None",
        family_size: int = 1,
    ) -> HypothesisTestResult:
        """Test a single hypothesis.

        Primary path: consume M2's rigorous ``stats_results`` (effect + CI +
        e-value, FDR-aware).  Fallback (when M2 supplied none, e.g. it ran
        without labeled data): a rigorous ``compare()`` on the extracted
        per-case signal — *not* a hand-rolled proportion difference.
        """
        stats_results = list(getattr(stats_report, "stats_results", None) or [])
        if stats_results:
            core = self._verdict_from_stats_results(hypothesis, stats_report, stats_results)
            if core.get("evidence_grade") == "none":
                core = self._verdict_fallback(hypothesis, stats_report, data, family_size)
            elif core.get("evidence_grade") == "unmet":
                # The design named evidence nobody measured: stay inconclusive
                # rather than let the generic per-case fallback (any truthy
                # analyzer value vs FAIL) manufacture a verdict about it.
                core["evidence_grade"] = "none"
        else:
            core = self._verdict_fallback(hypothesis, stats_report, data, family_size)

        status: HypothesisStatus = core["status"]
        verdict: str = core["verdict"]

        # ── Protocol consistency check ────────────────────────────────
        is_consistent = self._check_protocol_consistency(
            hypothesis, stats_report, protocol
        )
        if not is_consistent and status == HypothesisStatus.SUPPORTED:
            verdict += " (note: not protocol-consistent — excluded from best candidates)"

        return HypothesisTestResult(
            hypothesis=hypothesis,
            status=status,
            test_name=core["test_name"],
            effect_size=core["effect_size"],
            is_consistent_with_protocol=is_consistent,
            confidence=core["confidence"],
            verdict=verdict,
            evidence_grade=core.get("evidence_grade", "observational"),
            evidence=core["evidence"],
        )

    # ------------------------------------------------------------------
    # Primary path: consume M2's rigorous stats_results
    # ------------------------------------------------------------------

    def _verdict_from_stats_results(
        self,
        hypothesis: Hypothesis,
        stats_report: "StatsAnalysisReport",
        stats_results: "list[StatsToolResult]",
    ) -> dict[str, Any]:
        """Derive a verdict from M2's effect-sized, FDR-aware tool results."""
        corrected = getattr(stats_report, "corrected_rejections", None) or {}
        fdr_rejected_keys = set(corrected.get("rejected_result_keys") or [])

        def _decisive(r: "StatsToolResult") -> bool:
            # A result decides direction only if it rejected H0 AFTER multiplicity
            # correction. correct_results() files every p- or e-valued result into
            # a BH / e-BH family and writes the family's verdict to fdr_corrected;
            # `reject` is deliberately left raw for BH members (loop evidence, see
            # multiplicity.py), so reading it here was the defect: on
            # qwen3.5-2b/bbh_word_sorting a signal with p=0.14 over n_signal=3
            # "supported" a hypothesis because a bootstrap CI over three cases
            # cannot straddle zero. Per result, not per tool name — every
            # signal_label_assoc test shares one name, so a tool-level set said
            # "rejected" for all of them once any one survived.
            if not r.reject:
                return False
            if r.correction_method:
                return bool(r.fdr_corrected)
            if r.e_value is not None:
                # e-valued but never corrected (deferred report): only the
                # report's own survivor list can vouch for it.
                return bool(r.analysis_key) and r.analysis_key in fdr_rejected_keys
            return True

        # Outcome re-grades ("is the answer correct" by an analyzer's own
        # matcher) are tautological evidence in either direction; M2 isolates
        # them to the sanity lane since 2026-08-19, but a confirm-only run
        # reloads an older M2's results, so they are dropped here as well.
        from evalvitals.analysis.stats_tools import _is_outcome_regrade

        stats_results = [r for r in stats_results
                         if not _is_outcome_regrade(_tool_signal(r))]
        relevant, global_res, routed_by = self._select_results(hypothesis, stats_results)
        decisive = [r for r in relevant if (r.effect or 0.0) != 0 and _decisive(r)]
        # A signal's effect > 0 means "the signal group fails MORE". Whether that
        # SUPPORTS the hypothesis depends on what the hypothesis predicted: one
        # that says a signal is LOW / absent / "sits at 0" on failures (minervamath:
        # "step_rollout_value.initial_value ... 5/8 chains sit at 0.0 from step
        # 0") predicts a NEGATIVE effect, and reading every negative effect as
        # "protective -> refuted" rejected hypotheses the data agreed with. The
        # predicted direction is read from the hypothesis text (TEST line first);
        # absent any cue the classic reading (signal marks the failure) stands.
        expected = {id(r): _expected_sign(hypothesis, _tool_signal(r)) for r in decisive}

        def _agrees(r: "StatsToolResult") -> bool:
            exp = expected[id(r)]
            eff = float(r.effect or 0.0)
            return eff > 0 if exp is None else (eff > 0) == (exp > 0)

        supporting = [r for r in decisive if _agrees(r)]
        contradicting = [r for r in decisive if not _agrees(r)]

        consulted = [r.tool for r in relevant] + [r.tool for r in global_res]

        if supporting:
            chosen = max(supporting, key=lambda r: abs(r.effect or 0.0))
            status = HypothesisStatus.SUPPORTED
        elif contradicting:
            chosen = max(contradicting, key=lambda r: abs(r.effect or 0.0))
            status = HypothesisStatus.REFUTED
        else:
            # Descriptive tools (single_rate_evalue's rate − p0) must not be the
            # headline: their effect is a batch-composition artifact, so on an
            # enriched batch they would always win max(|effect|) and print a
            # meaningless "vs p0=0.50 → reject" as the verdict.
            pool = [r for r in (relevant or global_res)
                    if r.tool not in _DESCRIPTIVE_TOOLS]
            chosen = max(pool, key=lambda r: abs(r.effect or 0.0)) if pool else None
            status = HypothesisStatus.INCONCLUSIVE

        if routed_by == "test_design_unmet":
            named = sorted(_identifiers(hypothesis.test_design or ""))[:6]
            return {
                "status": HypothesisStatus.INCONCLUSIVE,
                "effect_size": None,
                "confidence": 0.0,
                "verdict": (
                    "designated evidence not measured this cycle: the test design names "
                    + (", ".join(named) if named else "signals")
                    + " but M2 has no result for them — not judged on unrelated signals "
                    "(next cycle's M1 targets the design)"
                ),
                "test_name": "stats_results",
                "evidence_grade": "unmet",
                "evidence": {"source": "m2_stats_results", "consulted_tools": consulted,
                             "routed_by": routed_by, "designated": named},
            }
        if chosen is None:
            return {
                "status": HypothesisStatus.INCONCLUSIVE,
                "effect_size": None,
                "confidence": 0.0,
                "verdict": ("No discriminating M2 result for this hypothesis"
                            + (f" (consulted {len(consulted)} tool(s); only "
                               "batch-descriptive statistics available)"
                               if consulted else "")),
                "test_name": "stats_results",
                "evidence_grade": "none",
                "evidence": {"source": "m2_stats_results", "consulted_tools": consulted,
                             "routed_by": routed_by},
            }

        grade = _evidence_grade(chosen.tool, chosen.config.get("signal", ""))
        confidence = _confidence_from_stat(
            chosen.effect, chosen.ci, chosen.e_value, chosen.underpowered, self.alpha
        )
        # summary is baked at tool-run time, before correction, so it can still
        # say "REJECT H0" on a result BH killed; carry the family verdict along.
        from evalvitals.analysis.stats_agent import _multiplicity_note

        note = _multiplicity_note(chosen).strip()
        note = f" {note}" if note else ""
        exp_sign = _expected_sign(hypothesis, _tool_signal(chosen))
        direction_note = ""
        if exp_sign is not None and status != HypothesisStatus.INCONCLUSIVE:
            direction_note = (
                f" [hypothesis predicts the signal {'HIGHER' if exp_sign > 0 else 'LOWER'}"
                f" on failures; observed effect {float(chosen.effect or 0.0):+.3f} -> "
                + ("consistent" if status == HypothesisStatus.SUPPORTED else "opposite")
                + "]"
            )
        if status == HypothesisStatus.INCONCLUSIVE:
            verdict = f"No significant M2 result (best: {chosen.summary}{note})"
        else:
            verdict = f"{chosen.tool}: {chosen.summary}{note}{direction_note}"

        evidence = {
            "source": "m2_stats_results",
            "chosen_tool": chosen.tool,
            "expected_direction": (None if exp_sign is None
                                   else ("higher_in_fail" if exp_sign > 0 else "lower_in_fail")),
            "effect_size": chosen.effect,
            "ci": list(chosen.ci) if chosen.ci is not None else None,
            "e_value": chosen.e_value,
            "reject": chosen.reject,
            "underpowered": chosen.underpowered,
            "consulted_tools": consulted,
            "routed_by": routed_by,
            "evidence_grade": grade,
            "fdr": corrected,
        }
        return {
            "status": status,
            "effect_size": chosen.effect,
            "confidence": confidence,
            "verdict": verdict,
            "test_name": chosen.tool,
            "evidence_grade": grade,
            "evidence": evidence,
        }

    def _select_results(
        self,
        hypothesis: Hypothesis,
        stats_results: "list[StatsToolResult]",
    ) -> "tuple[list[StatsToolResult], list[StatsToolResult], str]":
        """Split tool results into (relevant signal tools, global tools, routed_by).

        Routing priority (P3):
        1. **test_design** — when M3 attached an explicit test design to the
           hypothesis, tools whose signal/strategies overlap it are the
           designated evidence (deterministic routing).
        2. **keywords** — overlap between the hypothesis text and the tool's
           signal key/name.
        3. **shared** — all signal tools as shared evidence (historical
           aggregated-signal behaviour).
        """
        signal_res = [
            r for r in stats_results
            if r.ok and r.tool in _SIGNAL_TOOLS and r.effect is not None
        ]
        global_res = [r for r in stats_results if r.ok and r.tool in _GLOBAL_TOOLS]

        def _tool_text(r: "StatsToolResult") -> str:
            strategies = " ".join(r.config.get("strategies", []) or [])
            return (f"{r.tool} {r.config.get('signal', '')} {strategies} "
                    f"{r.details.get('signal', '')}")

        # 1. Identifiers first. M3 is told to name analyzers / per-case signals
        #    ("cot_faithfulness.drift_away", "perturbation_battery"); those are
        #    the underscore/dotted tokens of the design. Matching on them, and
        #    only them, keeps a design that says "... the answer ..." from
        #    routing to answer_extraction_audit.gold_in_answer_region — which
        #    is exactly what happened on qwen3.5-2b/bbh_causal_judgement: six
        #    hypotheses about overthinking / counterfactuals / prompt encoding
        #    all "refuted" by that one unrelated (protective) signal.
        design_raw = hypothesis.test_design or ""
        design_ids = _identifiers(design_raw)
        if design_ids:
            exact = [r for r in signal_res if _tool_ids(r, level="signal") & design_ids]
            if exact:
                return exact, global_res, "test_design"
            by_analyzer = [r for r in signal_res if _tool_ids(r, level="analyzer") & design_ids]
            if by_analyzer:
                return by_analyzer, global_res, "test_design"
        # 2. Words of the design (generic tokens removed) against the SIGNAL's
        #    own words — never the tool name ("signal_label_assoc" is in every
        #    tool text) and never a stop word.
        design_kw = _signal_keywords(design_raw)
        if design_kw:
            designed = [r for r in signal_res if design_kw & _tool_keywords(r)]
            if designed:
                return designed, global_res, "test_design"
        if design_ids or design_kw:
            # The design named evidence M2 did not measure this cycle. Judging
            # the hypothesis on whatever else M2 happened to test is not a
            # test of THIS hypothesis; say so (M1's next cycle targets the
            # design) instead of borrowing an unrelated verdict.
            return [], global_res, "test_design_unmet"
        kw = _signal_keywords(hypothesis.statement + " "
                              + hypothesis.predicted_failure_mode.replace("_", " "))
        stmt_ids = _identifiers(hypothesis.statement)
        matched = [
            r for r in signal_res
            if (_tool_ids(r, level="signal") & stmt_ids)
            or (_tool_ids(r, level="analyzer") & stmt_ids)
            or (kw & _tool_keywords(r))
        ]
        if matched:
            return matched, global_res, "keywords"
        return signal_res, global_res, "shared"

    # ------------------------------------------------------------------
    # Fallback path: rigorous compare() on the extracted per-case signal
    # ------------------------------------------------------------------

    def _verdict_fallback(
        self,
        hypothesis: Hypothesis,
        stats_report: "StatsAnalysisReport",
        data: CaseBatch,
        family_size: int = 1,
    ) -> dict[str, Any]:
        """Rigorous bootstrap comparison when M2 supplied no stats_results.

        ``family_size`` is the number of hypotheses sharing this fallback cycle;
        a directional verdict must be significant at the Bonferroni-corrected
        level ``alpha / family_size`` (FWER control over the fallback family,
        the unpaired-CI analogue of the primary path's e-BH). At ``family_size=1``
        the correction is a no-op.
        """
        signal = _extract_per_case_signals(stats_report.raw_results)
        signal_source = "analyzer_per_case"
        if not signal:
            signal = _fallback_case_signals(data, hypothesis)
            signal_source = "case_metadata_fallback"

        fail_signal: list[int] = []
        fail_control: list[int] = []
        for case in data:
            if getattr(case, "label", None) is None:
                continue
            is_fail = int(case.label == Label.FAIL)
            (fail_signal if signal.get(case.id, False) else fail_control).append(is_fail)

        if not fail_signal or not fail_control:
            return {
                "status": HypothesisStatus.INCONCLUSIVE,
                "effect_size": None,
                "confidence": 0.0,
                "verdict": "Insufficient labeled data to test statistically.",
                "test_name": "fail_rate_compare",
                "evidence": {
                    "source": "fallback_compare",
                    "signal_source": signal_source,
                    "reason": "no signal-present and signal-absent split available",
                    "n_signal": len(fail_signal),
                    "n_control": len(fail_control),
                },
            }

        # Rigorous, effect-sized verdict (clustered bootstrap CI) — replaces the
        # old hand-computed proportion difference + geometric-mean confidence.
        # The CI is taken at the Bonferroni-corrected level so a directional
        # verdict is significant across the whole fallback family, not just on
        # its own (best-of-N over hypotheses would otherwise inflate SUPPORTED).
        corrected_alpha = self.alpha / max(1, family_size)
        sr = compare(
            fail_control, fail_signal, paired=False,
            alpha=corrected_alpha, min_effect=self.min_effect,
        )
        effect_size = round(sr.effect, 4)
        confidence = _confidence_from_stat(
            sr.effect, sr.ci, sr.e_value, sr.underpowered, corrected_alpha)
        # A directional verdict requires BOTH a meaningful effect AND significance
        # (CI excludes 0) at the family-corrected level.
        significant = sr.reject

        if effect_size > self.min_effect and significant:
            status = HypothesisStatus.SUPPORTED
            verdict = (
                f"Signal group fails {effect_size:.0%} more than control "
                f"({sr.summary()})."
            )
        elif effect_size < -self.min_effect and significant:
            status = HypothesisStatus.REFUTED
            verdict = (
                f"Signal group fails less than control "
                f"(effect={effect_size:.0%}); hypothesis refuted."
            )
        elif abs(effect_size) > self.min_effect and not significant:
            status = HypothesisStatus.INCONCLUSIVE
            verdict = (
                f"Effect {effect_size:.0%} but CI includes 0 at the "
                f"family-corrected level (alpha={corrected_alpha:.4g} over "
                f"{family_size} hypotheses) — best-of-N multiplicity, inconclusive."
            )
        else:
            status = HypothesisStatus.INCONCLUSIVE
            verdict = (
                f"Effect size {effect_size:.0%} below minimum {self.min_effect:.0%}; "
                f"inconclusive."
            )

        evidence = {
            "source": "fallback_compare",
            "signal_source": signal_source,
            "n_signal": len(fail_signal),
            "n_control": len(fail_control),
            "fail_rate_signal": round(sum(fail_signal) / len(fail_signal), 4),
            "fail_rate_control": round(sum(fail_control) / len(fail_control), 4),
            "effect_size": effect_size,
            "ci": list(sr.ci),
            "underpowered": sr.underpowered,
            "method": sr.method,
            "significant": significant,
            "family_size": family_size,
            "corrected_alpha": corrected_alpha,
        }
        return {
            "status": status,
            "effect_size": effect_size,
            "confidence": confidence,
            "verdict": verdict,
            "test_name": "fail_rate_compare",
            "evidence": evidence,
        }

    def _check_protocol_consistency(
        self,
        hypothesis: Hypothesis,
        stats_report: "StatsAnalysisReport",
        protocol: "ExperimentProtocol | None",
    ) -> bool:
        """Return True when the hypothesis is relevant to the protocol.

        Uses LLM critic when a judge is available; falls back to keyword
        overlap between the hypothesis failure mode and the protocol hints.
        """
        if protocol is None:
            return True

        if self._judge is not None:
            try:
                return self._llm_consistency_check(hypothesis, protocol)
            except Exception as exc:
                logger.debug("LLM consistency check failed, falling back: %s", exc)

        return self._heuristic_consistency_check(hypothesis, protocol)

    def _heuristic_consistency_check(
        self,
        hypothesis: Hypothesis,
        protocol: "ExperimentProtocol",
    ) -> bool:
        """Text overlap between protocol description and hypothesis."""
        proto_text = (protocol.description + " " + protocol.failure_patterns).lower()
        if not proto_text.strip():
            return True  # no prior to compare against

        hyp_text = (
            hypothesis.statement + " "
            + hypothesis.predicted_failure_mode.replace("_", " ")
        ).lower()

        # Any significant word (6+ chars) from hypothesis in protocol text?
        for word in hyp_text.split():
            if len(word) >= 6 and word in proto_text:
                return True

        # Any significant word (6+ chars) from protocol in hypothesis text?
        for word in proto_text.split():
            if len(word) >= 6 and word in hyp_text:
                return True

        return False

    def _llm_consistency_check(
        self,
        hypothesis: Hypothesis,
        protocol: "ExperimentProtocol",
    ) -> bool:
        """LLM critic: ask judge if the hypothesis is relevant to the protocol."""
        import inspect

        prompt = _CONSISTENCY_PROMPT.format(
            protocol_text=protocol.description,
            statement=hypothesis.statement,
            failure_mode=hypothesis.predicted_failure_mode,
        )
        sig = inspect.signature(self._judge.generate)  # type: ignore[union-attr]
        if "temperature" in sig.parameters:
            raw = self._judge.generate(prompt, temperature=0)  # type: ignore[union-attr]
        else:
            raw = self._judge.generate(prompt)  # type: ignore[union-attr]

        first_line = str(raw).strip().splitlines()[0].upper()
        return first_line.startswith("YES")


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _fallback_case_signals(data: CaseBatch, hypothesis: Hypothesis) -> dict[str, bool]:
    """Non-label fallback signals when analyzers expose no per-case entries.

    These are operational signals from discovery/generation, not PASS/FAIL
    labels: empty outputs, discovery exceptions, or UNKNOWN scorer verdicts.
    They keep M5 from becoming vacuous while avoiding label leakage.
    """
    hyp_text = (
        hypothesis.statement + " " + hypothesis.predicted_failure_mode
    ).lower()
    wants_empty = any(k in hyp_text for k in ("empty", "whitespace", "special token"))
    wants_parse = any(k in hyp_text for k in ("parse", "parser", "unparsed", "format"))

    signal: dict[str, bool] = {}
    for case in data:
        meta = getattr(case, "metadata", {}) or {}
        observed = str(getattr(case, "observed", "") or meta.get("discovery_observed", ""))
        hit = False
        if wants_empty and not observed.strip():
            hit = True
        if wants_parse and meta.get("discovery_label") == "unknown":
            hit = True
        if meta.get("discovery_error"):
            hit = True
        if hit:
            signal[case.id] = True
    return signal


def _keywords(text: str) -> set[str]:
    """Significant (4+ char) lowercase word tokens of *text*."""
    return set(re.findall(r"[a-z]{4,}", text.lower()))


#: Underscore / dotted names — how M3 refers to analyzers and per-case signals
#: ("perturbation_battery", "cot_faithfulness.drift_away", "noop_break_rate").
_IDENT_RE = re.compile(r"[a-z][a-z0-9]*(?:[._][a-z0-9]+)+")

#: Words that appear in nearly every signal name / test design and therefore
#: route nothing: matching on them is what sent unrelated hypotheses to
#: answer_extraction_audit.gold_in_answer_region ("answer") on causal_judgement.
_GENERIC_KW = frozenset({
    "answer", "answers", "output", "outputs", "case", "cases", "rate", "rates",
    "final", "score", "scores", "model", "prompt", "prompts", "label", "labels",
    "signal", "signals", "text", "value", "values", "step", "steps", "chain",
    "token", "tokens", "index", "count", "number", "numbers", "flag", "flags",
    "match", "matched", "matches", "held", "split", "test", "tests", "control",
    "cases", "correct", "incorrect", "true", "false", "fail", "fails", "failed",
    "pass", "passed", "with", "without", "that", "this", "from", "into", "only",
    "each", "every", "after", "before", "versus", "against", "across", "between",
    "using", "under", "over", "more", "less", "most", "least", "some", "none",
    "then", "than", "when", "while", "should", "would", "could", "must",
    "still", "also", "both", "same", "other", "their", "there", "these", "those",
    "which", "where", "what", "does", "will", "have", "been", "were", "they",
    "them", "much", "many", "very", "just", "like", "such", "here", "mean",
    "median", "batch", "data", "audit", "battery", "probe", "probes",
})


def _identifiers(text: str) -> set[str]:
    """Underscore / dotted identifiers in *text*, lower-cased (backticks ignored)."""
    return {m.group(0) for m in _IDENT_RE.finditer(str(text or "").lower())}


def _signal_keywords(text: str) -> set[str]:
    """Non-generic 4+ letter words of *text* (for word-level routing)."""
    return _keywords(text) - _GENERIC_KW


def _tool_signal(r: "StatsToolResult") -> str:
    cfg = getattr(r, "config", None) or {}
    det = getattr(r, "details", None) or {}
    return str(cfg.get("signal") or det.get("signal") or "").lower()


# Direction cues. Explicit "<higher|lower> on/in FAIL" phrases are read first
# (the M3 TEST line asks for them); otherwise cue words within a short window
# after the signal's mention ("sits at 0", "= 0", "absent", "low" -> LOWER on
# failures; "= 1", "elevated", "present", "high" -> HIGHER). Both / neither ->
# None, and the classic reading (signal marks the failure: effect > 0 supports)
# applies.
_DIR_EXPLICIT_HIGH = re.compile(
    r"\b(?:higher|high|elevated|greater|larger|more|raised|increased|present)\b"
    r"[^.;\n]{0,40}?\b(?:on|in|for|among|across)\s+(?:the\s+)?(?:fail\w*|failing|failures?)"
    r"|\b(?:lower|low|absent|smaller|less|reduced|decreased|missing|zero)\b"
    r"[^.;\n]{0,40}?\b(?:on|in|for|among|across)\s+(?:the\s+)?(?:pass\w*|passing|successes?)")
_DIR_EXPLICIT_LOW = re.compile(
    r"\b(?:lower|low|absent|smaller|less|reduced|decreased|missing|zero)\b"
    r"[^.;\n]{0,40}?\b(?:on|in|for|among|across)\s+(?:the\s+)?(?:fail\w*|failing|failures?)"
    r"|\b(?:higher|high|elevated|greater|larger|more|raised|increased|present)\b"
    r"[^.;\n]{0,40}?\b(?:on|in|for|among|across)\s+(?:the\s+)?(?:pass\w*|passing|successes?)")
_DIR_CUE_LOW = re.compile(
    r"(?:^|[\s(`])(?:=|==|is|are|at|of|sits?\s+at|sit\s+at|stuck\s+at|stays?\s+at|"
    r"equals?|reads?|drops?\s+to|falls?\s+to)\s*0(?:\.0+)?(?![.\d%])"
    r"|\b(?:low|lower|absent|zero|null|missing|lacking|drops?|falls?|"
    r"decreas\w*|never\s+fires?|does\s+not\s+fire|=\s*false)\b")
_DIR_CUE_HIGH = re.compile(
    r"(?:^|[\s(`])(?:=|==|is|are|at|of|sits?\s+at|equals?|reads?)\s*1(?:\.0+)?(?![.\d%])"
    r"|\b(?:high|higher|elevated|present|raised|increas\w*|spikes?|rises?|fires?|"
    r"=\s*true|dominat\w*)\b")
_DIR_WINDOW = 110
_DIR_EQ = re.compile(r"[`'\")\]]*\s*(?:==|=|is|equals?)\s*(0(?:\.0+)?|1(?:\.0+)?)(?![.\d%])(?!\s*[-–]\s*\d|\s*(?:to|vs|or)\b)")


def _expected_sign(hypothesis: "Hypothesis", signal: str) -> "int | None":
    """+1 if *hypothesis* predicts *signal* HIGHER on failing cases, -1 if LOWER,
    None when it says nothing legible about the direction (the caller then
    falls back to the classic reading: the signal marks the failure)."""
    if not signal:
        return None
    sig = str(signal).lower()
    names = {sig}
    if "." in sig:
        names.add(sig.split(".", 1)[1])  # bare metric name
    texts = [str(getattr(hypothesis, "test_design", "") or ""),
             str(getattr(hypothesis, "statement", "") or "")]
    for text in texts:
        low = text.lower()
        for name in sorted(names, key=len, reverse=True):
            start = 0
            votes_high = votes_low = 0
            while True:
                i = low.find(name, start)
                if i < 0:
                    break
                # whole-identifier match only (avoid "majority_correct" inside
                # "majority_correctness" etc.)
                before = low[i - 1] if i > 0 else " "
                after_i = i + len(name)
                after = low[after_i] if after_i < len(low) else " "
                if before.isalnum() or before == "_" or after.isalnum() or after == "_":
                    start = i + 1
                    continue
                window = low[max(0, i - 40):after_i + _DIR_WINDOW]
                tail = low[after_i:after_i + _DIR_WINDOW]
                # an equality right after the name ("`x` = 0", "x == 1", "x=0")
                # is decisive on its own and must not be diluted by a second
                # identifier's equality further along the sentence
                near = _DIR_EQ.match(low[after_i:after_i + 16])
                if near:
                    if near.group(1).startswith("0"):
                        votes_low += 1
                    else:
                        votes_high += 1
                elif _DIR_EXPLICIT_HIGH.search(window):
                    votes_high += 1
                elif _DIR_EXPLICIT_LOW.search(window):
                    votes_low += 1
                else:
                    lo = bool(_DIR_CUE_LOW.search(tail))
                    hi = bool(_DIR_CUE_HIGH.search(tail))
                    if lo and not hi:
                        votes_low += 1
                    elif hi and not lo:
                        votes_high += 1
                start = after_i
            if votes_high and not votes_low:
                return 1
            if votes_low and not votes_high:
                return -1
            if votes_high and votes_low:
                return None
    return None


def _tool_ids(r: "StatsToolResult", *, level: str = "signal") -> set[str]:
    """Identifiers a tool result answers to.

    ``level="signal"``: the full ``analyzer.signal`` key, the bare signal name
    and the strategy names; ``level="analyzer"``: the analyzer prefix only.
    """
    sig = _tool_signal(r)
    cfg = getattr(r, "config", None) or {}
    ids: set[str] = set()
    if level == "analyzer":
        if sig:
            ids.add(sig.split(".", 1)[0])
        return ids
    if sig:
        ids.add(sig)
        parts = sig.split(".", 1)
        if len(parts) > 1 and parts[1]:
            ids.add(parts[1])
    for strategy in cfg.get("strategies", []) or []:
        ids.add(str(strategy).lower())
    return ids


def _tool_keywords(r: "StatsToolResult") -> set[str]:
    """Non-generic words of the tool's signal key + strategies (NOT the tool name)."""
    cfg = getattr(r, "config", None) or {}
    words = " ".join([_tool_signal(r)] + [str(x) for x in (cfg.get("strategies", []) or [])])
    return _signal_keywords(words.replace("_", " ").replace(".", " "))


def _confidence_from_stat(
    effect: float | None,
    ci: tuple[float, float] | None,
    e_value: float | None,
    underpowered: bool,
    alpha: float = 0.05,
) -> float:
    """Map a rigorous statistical verdict to a confidence in [0, 1].

    Priority: an e-value (anytime-valid) over a CI.  An e-value at the rejection
    threshold ``1/alpha`` maps to 1.0; a CI that excludes 0 scales by how far it
    sits from 0 relative to its width.  ``underpowered`` halves the score.
    """
    c = 0.0
    if e_value is not None:
        c = min(1.0, e_value * alpha)  # e_value / (1/alpha)
    elif ci is not None:
        lo, hi = ci
        width = (hi - lo) or 1e-9
        if lo <= 0 <= hi:
            # CI includes 0 → not significant; small, capped contribution.
            c = min(0.49, abs(effect or 0.0) / width * 0.5)
        else:
            c = min(1.0, abs(effect or 0.0) / (abs(effect or 0.0) + width))
    if underpowered:
        c *= 0.5
    return round(c, 3)
