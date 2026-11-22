"""Host-side adjudication of explorer candidate signals (the firewall).

The :mod:`~evalrx.analysis.explorer` PROPOSES candidate signals; it has no
authority to declare significance. This module is the host firewall: it takes the
host-adjudicable ``sufficient`` statistics an explorer attached to a candidate and
recomputes the verdict with the SAME validated, multiplicity-aware core the M2
stats engine uses — never trusting a ``reject`` / ``e_value`` / ``p_value`` the
explorer self-declared (the explorer schema does not even carry one).

Mechanics (mirror ``stats_tools`` / ``stats_tool_generator`` exactly):

- ``reconstruct_evidence`` turns sufficient statistics into a verdict via the
  validated core.
- A ``paired_binary`` candidate yields an e-value; those candidates share one
  e-BH family.
- A ``two_group`` candidate yields a bootstrap interval and p-value; those
  candidates share one Benjamini-Hochberg family. Its ``reject`` is survival of
  that correction, so proposing more two-group candidates pays more
  multiplicity, as it does for paired ones. ``raw_reject`` in the returned
  metadata lists the candidates whose own interval excluded zero.
- A candidate with no adjudicable ``sufficient`` is ``descriptive_only`` and never
  rejects.
- ``method`` in the returned metadata names the correction that was applied:
  ``"e-BH"``, ``"BH"``, both, or ``"none"`` when no candidate could be tested.

IMPORTANT — split discipline: in the standalone chat path the candidate's
``sufficient`` is computed on the SAME rows used to discover the signal. That is
IN-SAMPLE; ``split_label`` records it so nothing over-claims, and an honest caveat
is appended. Held-out CONFIRM-split adjudication (selecting on EXPLORE, computing
each e-value on a disjoint CONFIRM split) is the job of the fused pipeline
(Phase B). This module supplies only the verdict mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evalrx.analysis.stats_tool_generator import reconstruct_evidence
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
    # 1) Host reconstruction per candidate; candidates with an e-value form the
    #    e-BH family, candidates with a p-value the BH family.
    raw_reject: dict[int, bool] = {}
    family: list[StatsToolResult] = []
    family_key: dict[int, str] = {}
    for i, c in enumerate(candidates):
        evidence = (
            reconstruct_evidence(c.sufficient, alpha)
            if isinstance(c.sufficient, dict)
            else None
        )
        if evidence is None:
            c.host_adjudicated = False
            c.descriptive_only = True
            c.fdr_corrected = False
            c.e_value = None
            c.p_value = None
            c.reject = False
            continue

        c.host_adjudicated = True
        c.descriptive_only = False
        c.e_value = evidence["e_value"]
        c.p_value = evidence["p_value"]
        if evidence["effect"] is not None:
            c.effect = evidence["effect"]
        if evidence["ci"] is not None:
            c.ci = evidence["ci"]
        raw_reject[i] = bool(evidence["reject"])
        if c.e_value is not None or c.p_value is not None:
            key = _family_key(c, i)
            family_key[i] = key
            family.append(
                StatsToolResult(
                    tool="candidate_signal", analysis_key=key,
                    e_value=c.e_value,
                    p_value=c.p_value if c.e_value is None else None,
                    effect=c.effect, ci=c.ci, reject=raw_reject[i],
                    correction_family="e_bh" if c.e_value is not None else "bh",
                )
            )

    # 2) One e-BH family across the e-value-bearing candidates and one BH
    #    family across the p-value-bearing ones (the multiplicity firewall).
    fdr = fdr_correct(family, alpha=alpha)
    rejected_keys = set(fdr.get("rejected_result_keys", []))

    # 3) Final reject per candidate: survival of its family's correction.
    #    `fdr_corrected` means "this rejection is FDR-controlled" — True ONLY
    #    when it survived (matches fused_pipeline; a family member that did not
    #    survive is reject=False, fdr_corrected=False).
    for i, c in enumerate(candidates):
        if not c.host_adjudicated:
            continue
        c.reject = family_key.get(i) in rejected_keys
        c.fdr_corrected = bool(c.reject)

    n_e = sum(1 for c in candidates if c.host_adjudicated and c.e_value is not None)
    n_p = sum(
        1 for c in candidates
        if c.host_adjudicated and c.e_value is None and c.p_value is not None
    )
    method = " + ".join(
        name for name, size in (("e-BH", n_e), ("BH", n_p)) if size
    ) or "none"
    return {
        "method": method,
        "alpha": alpha,
        "split": split_label,
        "n_candidates": len(candidates),
        "n_host_adjudicated": sum(1 for c in candidates if c.host_adjudicated),
        "n_in_family": n_e,  # e-value-bearing candidates under e-BH
        "n_in_p_family": n_p,  # p-value-bearing (two_group) candidates under BH
        "n_rejected": sum(1 for c in candidates if c.reject),
        "rejected": [c.name for c in candidates if c.reject],
        "raw_reject": [c.name for i, c in enumerate(candidates) if raw_reject.get(i)],
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
