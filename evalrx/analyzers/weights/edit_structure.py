"""Weight-edit structure — describe an edit from the weights alone.

No forward pass and no cases are needed: the analyzer applies the edit to a
copy of each tensor and reports where it moved, how large the move is, whether
it is low rank, and — for a rank-one change — the strength ``alpha`` of the
rule ``w' = w - alpha u (u^T w)`` that reproduces it. This is the first step of
auditing any edit, because every later explanation has to be an explanation of
THIS change.

Reference:
- Sparks of Weight-Space Programming — Takehi, Teoh, Zhang & Khattab, 2026.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


@register_analyzer("weight_edit_structure")
class WeightEditStructure(Analyzer):
    """Describe a weight edit from the weights: size, location, rank, and the rank-one rule it follows.

    Args:
        edit:         The :class:`~evalrx.wsp.edit.WeightEdit` to describe.
        verify_base:  Require the live tensors to match the hashes recorded in
                      the edit (an edit that overwrites entries is only
                      meaningful on the weights it was made on).
        top_singular: How many leading singular values to report per tensor.
    """

    name = "weight_edit_structure"
    # LOGITS stands in for "a local model whose parameters can be read": the
    # capability vocabulary has no entry for parameter access, and every
    # backend that provides full logits holds the module in process.
    requires = frozenset({Capability.LOGITS})
    question = "What exactly did the edit change in the weights?"
    probe_cost = "light"
    probe_intervenes = False
    signal_docs = {
        "max_relative_change": (
            "Edit size",
            "Largest relative change of any edited tensor (norm of the change over norm of "
            "the original). Larger means a more invasive edit.",
        ),
        "max_rank_one_residual": (
            "Rank-one residual",
            "Share of the change NOT explained by a single direction; 0 means the edit is "
            "exactly rank one, higher means it has more structure than one direction.",
        ),
    }

    edit: Any
    verify_base: bool
    top_singular: int

    def __init__(self, edit: "Any", verify_base: bool = False, top_singular: int = 8) -> None:
        super().__init__(edit=edit, verify_base=verify_base, top_singular=top_singular)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        from evalrx.wsp.edit import applied
        from evalrx.wsp.structure import edit_structure

        structure = edit_structure(model, self.edit, top_singular=self.top_singular)
        # Entering the context proves the edit resolves, fits and restores.
        with applied(model, self.edit, verify_base=self.verify_base) as report:
            applied_report = report.to_dict()
        rows = structure["tensors"]
        residuals = [r["rank_one_residual"] for r in rows if "rank_one_residual" in r]
        alphas = [r["alpha_estimate"] for r in rows if "alpha_estimate" in r]
        findings: dict[str, Any] = {
            "edit": self.edit.name,
            "n_tensors": len(rows),
            "changed_entries_total": sum(r.get("changed_entries", 0) for r in rows),
            "max_relative_change": applied_report["max_relative_change"],
            "max_rank_one_residual": max(residuals) if residuals else None,
            "alpha_estimates": [round(a, 4) for a in alphas],
            "base_verified": bool(self.verify_base),
            "tensors": rows,
            "direction_cosines": structure["direction_cosines"],
            "edit_metadata": self.edit.metadata,
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
