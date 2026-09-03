"""Regression tests for the adversarial-audit findings (wqxn21avv).

Each test would FAIL against the pre-fix code: they pin the silent-collision /
overwrite / self-prediction-leak guarantees the happy-path tests missed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evalrx.analysis import CandidateSignal, ExploratoryAnalysisReport
from evalrx.analysis.adjudicate import adjudicate_signals
from evalrx.analysis.fused_pipeline import run_fused_analysis
from evalrx.analysis.operationalize import (
    RecipeError,
    SignalRecipe,
    bridge_recipes_to_result,
    compile_recipe,
    compile_recipes,
    per_case_to_records,
)
from evalrx.eval_agent.loop import VLDiagnoseLoop


class _FakeResult:
    def __init__(self, per_case):
        self.findings = {"per_case": per_case}


# ---------------------------------------------------------------------------
# G4 leak — a recipe may not reference the outcome/id columns (self-prediction)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr", [
    "label == 1",
    "is_fail",
    "case_id",
    "(obj_size < 40) and is_correct",
])
def test_recipe_cannot_reference_label_or_id_columns(expr):
    rows = [{"case_id": "c0", "label": 1, "is_fail": 1, "is_correct": 0, "obj_size": 20}]
    with pytest.raises(RecipeError):
        compile_recipe(SignalRecipe(name="leak", kind="expr", expr=expr), rows)


def test_bridge_does_not_expose_label_to_recipe_compilation():
    # Even though labels exist, a recipe referencing 'label' is rejected -> dropped.
    per_case = [{"case_id": f"c{i}", "obj_size": 20 + i} for i in range(4)]
    probe_results = {"saliency": _FakeResult(per_case)}
    data = None
    leak = SignalRecipe(name="leak", kind="expr", expr="label == 'fail'")
    synth = bridge_recipes_to_result([leak], probe_results, data)
    assert synth is None  # leak recipe rejected at compile time -> nothing bridged


# ---------------------------------------------------------------------------
# #2/#6 — safe_ident collisions exclude (not first-wins-overwrite) the signals
# ---------------------------------------------------------------------------

def test_per_case_to_records_excludes_safe_ident_collisions():
    # "a.b" and "a_b" both sanitize to "a_b" — ambiguous, so BOTH are excluded.
    per_case = {
        "a.b": {"c0": 1.0},
        "a_b": {"c0": 2.0},
        "clean": {"c0": 9.0},
    }
    records = per_case_to_records(per_case)
    (row,) = records
    assert "a_b" not in row          # neither colliding signal leaks in
    assert row["clean"] == 9.0       # non-colliding signal survives
    assert "c0" == row["case_id"]


# ---------------------------------------------------------------------------
# #4 — duplicate recipe.name keeps the first, never silently clobbers
# ---------------------------------------------------------------------------

def test_compile_recipes_duplicate_name_keeps_first():
    rows = [{"case_id": "c0", "x": 5}]
    first = SignalRecipe(name="dup", kind="expr", expr="x > 0")    # -> 1.0
    second = SignalRecipe(name="dup", kind="expr", expr="x > 100")  # -> 0.0
    out = compile_recipes([first, second], rows)
    assert out["dup"]["c0"] == 1.0  # the FIRST recipe's value, not the second's


# ---------------------------------------------------------------------------
# #3 — in-loop bridge never overwrites a real analyzer named 'explored'
# ---------------------------------------------------------------------------

def test_loop_bridge_does_not_overwrite_real_analyzer():
    real = _FakeResult([{"case_id": "c0", "obj_size": 20}])
    probe_results = {"explored": real, "saliency": _FakeResult([{"case_id": "c0", "obj_size": 20}])}
    recipe = SignalRecipe(name="small", kind="expr", expr="saliency_obj_size < 40")
    me = SimpleNamespace(
        _signal_recipes=[recipe],
        _bridge_analyzer_name="explored",
        model="m",
        store=SimpleNamespace(add_result=lambda r: None),
    )
    VLDiagnoseLoop._bridge_signals(me, probe_results, None)

    assert probe_results["explored"] is real  # the real analyzer is untouched
    # the bridge landed under a deduped, non-colliding key
    bridged = [k for k in probe_results if k.startswith("explored_bridge")]
    assert len(bridged) == 1


# ---------------------------------------------------------------------------
# #5 — fused: bridged recipe name never overwrites a real catalog column
# ---------------------------------------------------------------------------

class _FakeExplorer:
    def __init__(self, candidates):
        self._candidates = candidates

    def explore_records(self, rows, *, question=""):
        return ExploratoryAnalysisReport(ok=True, candidate_signals=list(self._candidates))


def test_fused_bridged_name_does_not_overwrite_catalog_column():
    # records carry a real catalog column 'obj_size'; explorer recipe is ALSO named 'obj_size'
    rows = (
        [{"case_id": f"f{i}", "label": "fail", "obj_size": 20} for i in range(12)]
        + [{"case_id": f"p{i}", "label": "pass", "obj_size": 80} for i in range(12)]
    )
    explorer = _FakeExplorer([CandidateSignal(
        name="obj_size",
        recipe={"name": "obj_size", "kind": "expr", "expr": "obj_size > 1000"},  # always 0
    )])
    rep = run_fused_analysis(rows, explorer=explorer, confirm_split=0.3, seed=0)
    by_name = {s.name: s for s in rep.candidate_signals}

    # the real catalog 'obj_size' is preserved AND the bridged one is namespaced apart
    assert "obj_size" in by_name
    assert "bridged.obj_size" in by_name
    assert any("collided with catalog" in c for c in rep.caveats)
    # provenance is correct: the real column the explorer ALSO named -> "both";
    # the bridged recipe -> "explorer" (kept as a distinct estimand, not merged).
    assert by_name["obj_size"].source == "both"
    assert by_name["bridged.obj_size"].source == "explorer"


# ---------------------------------------------------------------------------
# #1 — fdr_corrected is consistent: True only for an FDR-controlled rejection
# ---------------------------------------------------------------------------

def test_fdr_corrected_only_for_surviving_family_members():
    strong = CandidateSignal(name="a", sufficient={"kind": "paired_binary", "b": 20, "c": 0})
    weak = CandidateSignal(name="b", sufficient={"kind": "paired_binary", "b": 5, "c": 4})
    adjudicate_signals([strong, weak])

    assert strong.reject is True and strong.fdr_corrected is True
    # weak is IN the family (has an e-value) but did NOT survive e-BH:
    assert weak.e_value is not None       # family membership recoverable
    assert weak.reject is False
    assert weak.fdr_corrected is False    # never claims FDR control it lacks
