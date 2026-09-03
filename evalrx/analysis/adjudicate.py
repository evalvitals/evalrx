"""Host-side adjudication of explorer candidate signals (the firewall).

The :mod:`~evalrx.analysis.explorer` PROPOSES candidate signals; it has no
authority to declare significance. This module is the host firewall: it takes the
host-adjudicable ``sufficient`` statistics an explorer attached to a candidate and
recomputes the verdict with the SAME validated, multiplicity-aware core the M2
stats engine uses — never trusting a ``reject`` / ``e_value`` / ``p_value`` the
explorer self-declared (the explorer schema does not even carry one).

Mechanics (mirror ``stats_tools`` / ``stats_tool_generator`` exactly):

- ``_reconstruct_decision`` turns sufficient statistics into
  ``(reject, e_value, effect, ci)`` via the validated core.
- ``fdr_correct`` applies e-BH across every candidate that produced an e-value
  (the ``paired_binary`` shape), so proposing more candidates pays more
  multiplicity — exactly one e-BH family, no per-candidate self-rejection.
- A ``two_group`` candidate yields a bootstrap-CI reject but no e-value (matching
  the ``signal_label_assoc`` catalog tool). It is host-adjudicated but sits
  OUTSIDE the e-BH family and is flagged ``fdr_corrected=False`` — honest, not
  over-claimed.
- A candidate with no adjudicable ``sufficient`` is ``descriptive_only`` and never
  rejects.

IMPORTANT — split discipline: in the standalone chat path the candidate's
``sufficient`` is computed on the SAME rows used to discover the signal. That is
IN-SAMPLE; ``split_label`` records it so nothing over-claims, and an honest caveat
is appended. Held-out CONFIRM-split adjudication (selecting on EXPLORE, computing
each e-value on a disjoint CONFIRM split) is the job of the fused pipeline
(Phase B). This module supplies only the verdict mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evalrx.analysis.stats_tool_generator import _reconstruct_decision
from evalrx.analysis.stats_tools import StatsToolResult, fdr_correct

if TYPE_CHECKING:
    from evalrx.analysis.explorer import CandidateSignal, ExploratoryAnalysisReport


def _family_key(candidate: "CandidateSignal", index: int) -> str:
    """Stable, unique key so an e-BH rejection maps back to the right candidate
    even when two candidates share a name."""
    return f"{candidate.name or 'signal'}#{index}"


def adjudicate_signals(
    candidates: "list[CandidateSignal]",
    *,
    alpha: float = 0.05,
    split_label: str = "in_sample",
) -> dict[str, Any]:
    """Annotate each candidate IN PLACE with a host verdict; return family metadata.

    Reads only ``candidate.sufficient`` — never a self-declared verdict. Returns a
    metadata dict describing the single e-BH family the candidates competed in.
    """
    # 1) Host reconstruction per candidate; assemble the e-BH family from the
    #    candidates that produced an e-value (paired_binary).
    raw_reject: dict[int, bool] = {}
    family: list[StatsToolResult] = []
    family_key: dict[int, str] = {}
    for i, c in enumerate(candidates):
        recon = (
            _reconstruct_decision(c.sufficient, alpha)
            if isinstance(c.sufficient, dict)
            else None
        )
        if recon is None:
            c.host_adjudicated = False
            c.descriptive_only = True
            c.fdr_corrected = False
            c.e_value = None
            c.reject = False
            continue

        reject, e_value, effect, ci = recon
        c.host_adjudicated = True
        c.descriptive_only = False
        c.e_value = e_value
        if effect is not None:
            c.effect = effect
        if ci is not None:
            c.ci = ci
        raw_reject[i] = bool(reject)
        if e_value is not None:
            key = _family_key(c, i)
            family_key[i] = key
            family.append(
                StatsToolResult(
                    tool=key, e_value=e_value, effect=effect, ci=ci, reject=bool(reject)
                )
            )

    # 2) ONE e-BH family across the e-value-bearing candidates (the multiplicity
    #    firewall). `fdr_correct` only considers results with an e-value.
    fdr = fdr_correct(family, alpha=alpha)
    rejected_keys = set(fdr.get("rejected_tools", []))

    # 3) Final reject per candidate.
    for i, c in enumerate(candidates):
        if not c.host_adjudicated:
            continue
        if c.e_value is not None:
            # In the e-BH family. `reject` IS the e-BH outcome; `fdr_corrected`
            # means "this rejection is FDR-controlled" — True ONLY when it survived
            # e-BH (matches fused_pipeline; a family member that did not survive is
            # reject=False, fdr_corrected=False). Family membership is recoverable
            # from `e_value is not None`, so no information is lost.
            c.reject = family_key.get(i) in rejected_keys
            c.fdr_corrected = bool(c.reject)
        else:
            # two_group: CI-based reject, outside the e-BH family (mirrors M2's
            # signal_label_assoc). Honest about not being FDR-corrected.
            c.fdr_corrected = False
            c.reject = raw_reject.get(i, False)

    return {
        "method": "e-BH",
        "alpha": alpha,
        "split": split_label,
        "n_candidates": len(candidates),
        "n_host_adjudicated": sum(1 for c in candidates if c.host_adjudicated),
        "n_in_family": len(family),  # e-value-bearing candidates under e-BH
        "n_rejected": sum(1 for c in candidates if c.reject),
        "rejected": [c.name for c in candidates if c.reject],
    }


def adjudicate_report(
    report: "ExploratoryAnalysisReport",
    *,
    alpha: float = 0.05,
    split_label: str = "in_sample",
) -> "ExploratoryAnalysisReport":
    """Run :func:`adjudicate_signals` over a report's candidates, record metadata.

    Mutates and returns *report*. Appends an honest caveat when any verdict is
    in-sample (no held-out CONFIRM split was used to compute the e-values).
    """
    meta = adjudicate_signals(
        report.candidate_signals, alpha=alpha, split_label=split_label
    )
    report.adjudication = meta
    if meta["n_host_adjudicated"] and split_label == "in_sample":
        caveat = (
            "host verdicts are IN-SAMPLE (computed on the rows used to discover "
            "the signal); confirm on a held-out split before trusting"
        )
        if caveat not in report.caveats:
            report.caveats.append(caveat)
    return report
