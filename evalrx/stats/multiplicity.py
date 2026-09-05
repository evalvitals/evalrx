"""Multiple-testing controllers for generic M2 evidence families.

The older M2 path only corrected e-values with e-BH. A generalized analysis
layer also needs ordinary p-value BH families while preserving the e-value path
used by the M1-M4 loop. This module keeps those families separate and writes the
family verdict back onto each result object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from evalrx.stats.ebh import ebh


def bh(pvalues: Sequence[float], alpha: float = 0.05) -> list[int]:
    """Benjamini-Hochberg FDR control for p-values under PRDS/independence.

    Returns indices into *pvalues* that survive the step-up procedure.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    k_star = 0
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= alpha * rank / m:
            k_star = rank
    return sorted(order[:k_star])


@dataclass
class MultiplicityReport:
    """Machine-readable summary of all correction families."""

    method: str
    alpha: float
    n_tested: int
    rejected_tools: list[str] = field(default_factory=list)
    rejected_result_keys: list[str] = field(default_factory=list)
    families: dict[str, dict[str, Any]] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = {
            "method": self.method,
            "alpha": self.alpha,
            "n_tested": self.n_tested,
            "rejected_tools": self.rejected_tools,
            "rejected_result_keys": self.rejected_result_keys,
            "families": self.families,
        }
        if self.note:
            out["note"] = self.note
        return out


def _result_key(result: Any, index: int) -> str:
    key = getattr(result, "analysis_key", None)
    if key:
        return str(key)
    config = getattr(result, "config", None) or {}
    signal = config.get("signal")
    tool = str(getattr(result, "tool", "result"))
    return f"{tool}:{signal}" if signal else f"{tool}#{index}"


def correct_results(results: list[Any], *, alpha: float = 0.05) -> dict[str, Any]:
    """Apply correction to supported result families and mutate result verdicts.

    Supported families:
    - ``e_bh``: results with e-values, corrected with e-BH.
    - ``bh``: results with p-values, corrected with Benjamini-Hochberg.

    Results with ``correction_family`` set to ``None`` are descriptive and are
    not entered into any family. If the field is absent, e-values default to
    ``e_bh`` and p-values default to ``bh``.
    """
    families: dict[str, list[tuple[int, Any]]] = {"e_bh": [], "bh": []}
    for i, result in enumerate(results):
        if not getattr(result, "ok", False):
            continue
        family = getattr(result, "correction_family", "auto")
        if family in ("", "none", None):
            continue
        if family == "auto":
            if getattr(result, "e_value", None) is not None:
                family = "e_bh"
            elif getattr(result, "p_value", None) is not None:
                family = "bh"
            else:
                continue
        if family == "e_bh" and getattr(result, "e_value", None) is not None:
            families["e_bh"].append((i, result))
        elif family == "bh" and getattr(result, "p_value", None) is not None:
            families["bh"].append((i, result))

    family_reports: dict[str, dict[str, Any]] = {}
    rejected_indices: set[int] = set()

    e_family = families["e_bh"]
    if e_family:
        keep = ebh([float(r.e_value) for _, r in e_family], alpha)
        idxs = {e_family[j][0] for j in keep}
        rejected_indices.update(idxs)
        family_reports["e_bh"] = {
            "method": "e-BH",
            "n_tested": len(e_family),
            "rejected_result_keys": [_result_key(results[i], i) for i in sorted(idxs)],
        }

    p_family = families["bh"]
    if p_family:
        keep = bh([float(r.p_value) for _, r in p_family], alpha)
        idxs = {p_family[j][0] for j in keep}
        rejected_indices.update(idxs)
        family_reports["bh"] = {
            "method": "BH",
            "n_tested": len(p_family),
            "rejected_result_keys": [_result_key(results[i], i) for i in sorted(idxs)],
        }

    e_indices = {idx for idx, _ in e_family}
    p_indices = {idx for idx, _ in p_family}
    for i, result in enumerate(results):
        if i in rejected_indices:
            result.reject = True
            result.fdr_corrected = True
            result.correction_method = "e-BH" if i in e_indices else "BH"
        elif i in e_indices:
            result.reject = False
            result.fdr_corrected = False
            result.correction_method = "e-BH"
        elif i in p_indices:
            # Keep the tool's raw effect/CI verdict available to the M1-M4 loop.
            # BH status is exposed separately for controlled downstream claims.
            result.fdr_corrected = False
            result.correction_method = "BH"

    rejected_tools = sorted({str(getattr(results[i], "tool", "")) for i in rejected_indices})
    rejected_keys = [_result_key(results[i], i) for i in sorted(rejected_indices)]
    n_tested = len(e_family) + len(p_family)
    method = (
        "mixed-BH/e-BH" if e_family and p_family
        else "e-BH" if e_family
        else "BH" if p_family
        else "none"
    )
    note = "" if n_tested else "no p-values or e-values to correct"
    return MultiplicityReport(
        method=method,
        alpha=alpha,
        n_tested=n_tested,
        rejected_tools=rejected_tools,
        rejected_result_keys=rejected_keys,
        families=family_reports,
        note=note,
    ).to_dict()
