"""Phase C — in-loop operationalization bridge (synthetic-Result injection)."""

from __future__ import annotations

from types import SimpleNamespace

from evalrx.analysis.operationalize import (
    SignalRecipe,
    bridge_recipes_to_result,
    per_case_to_records,
    safe_ident,
)
from evalrx.analysis.stats_tools import build_stats_input
from evalrx.eval_agent.loop import VLDiagnoseLoop


class _FakeResult:
    """Duck-typed analyzer Result carrying per_case findings."""

    def __init__(self, per_case):
        self.findings = {"per_case": per_case}


def _probe_results():
    # Two analyzer signals per case; keys become dotted "saliency.obj_size" etc.
    per_case = [
        {"case_id": "c0", "obj_size": 20, "attention": 0.1},
        {"case_id": "c1", "obj_size": 80, "attention": 0.6},
        {"case_id": "c2", "obj_size": 25, "attention": 0.2},
    ]
    return {"saliency": _FakeResult(per_case)}


# ---------------------------------------------------------------------------
# transpose + sanitization
# ---------------------------------------------------------------------------

def test_safe_ident_sanitizes_dotted_keys():
    assert safe_ident("saliency.obj_size") == "saliency_obj_size"
    assert safe_ident("3bad") == "_3bad"


def test_per_case_to_records_transposes_and_sanitizes():
    per_case = {"saliency.obj_size": {"c0": 20, "c1": 80}}
    labels = {"c0": True, "c1": False}
    records = per_case_to_records(per_case, labels)

    assert records == [
        {"case_id": "c0", "saliency_obj_size": 20, "label": "fail"},
        {"case_id": "c1", "saliency_obj_size": 80, "label": "pass"},
    ]


def test_per_case_to_records_omits_missing_signals():
    per_case = {"a.x": {"c0": 1.0}, "a.y": {"c1": 2.0}}
    records = per_case_to_records(per_case)
    by_id = {r["case_id"]: r for r in records}
    assert "a_x" in by_id["c0"] and "a_y" not in by_id["c0"]
    assert "a_y" in by_id["c1"] and "a_x" not in by_id["c1"]


# ---------------------------------------------------------------------------
# bridge_recipes_to_result: recipes over existing analyzer signals -> synthetic
# ---------------------------------------------------------------------------

def test_bridge_builds_synthetic_result_referencing_analyzer_signals():
    recipe = SignalRecipe(
        name="small_and_peripheral",
        kind="expr",
        # references the SANITIZED analyzer signal names
        expr="(saliency_obj_size < 40) and (saliency_attention < 0.3)",
    )
    synth = bridge_recipes_to_result(
        [recipe], _probe_results(), data=None, model_repr="m"
    )
    assert synth is not None
    assert synth.analyzer == "explored"
    rows = {r["case_id"]: r for r in synth.findings["per_case"]}
    assert rows["c0"]["small_and_peripheral"] == 1.0  # small + low attention
    assert rows["c1"]["small_and_peripheral"] == 0.0  # large
    assert rows["c2"]["small_and_peripheral"] == 1.0


def test_bridged_signal_enters_stats_input_family():
    recipe = SignalRecipe(
        name="small_and_peripheral", kind="expr",
        expr="(saliency_obj_size < 40) and (saliency_attention < 0.3)",
    )
    probe_results = _probe_results()
    synth = bridge_recipes_to_result([recipe], probe_results, data=None)
    probe_results["explored"] = synth

    inp = build_stats_input(probe_results, data=None)
    # both the raw analyzer signals AND the bridged composite are in the family
    assert "saliency.obj_size" in inp.per_case
    assert "explored.small_and_peripheral" in inp.per_case


def test_bridge_returns_none_when_nothing_compiles():
    recipe = SignalRecipe(name="x", kind="expr", expr="nonexistent_signal > 0")
    synth = bridge_recipes_to_result([recipe], _probe_results(), data=None)
    assert synth is None


# ---------------------------------------------------------------------------
# VLDiagnoseLoop._bridge_signals wiring (tested on a lightweight fake self)
# ---------------------------------------------------------------------------

def _fake_loop_self(recipes):
    added = []
    return SimpleNamespace(
        _signal_recipes=recipes,
        _bridge_analyzer_name="explored",
        model="m",
        store=SimpleNamespace(add_result=added.append),
        _added=added,
    )


def test_loop_bridge_signals_injects_synthetic_analyzer():
    recipe = SignalRecipe(
        name="small_and_peripheral", kind="expr",
        expr="(saliency_obj_size < 40) and (saliency_attention < 0.3)",
    )
    probe_results = _probe_results()
    me = _fake_loop_self([recipe])

    VLDiagnoseLoop._bridge_signals(me, probe_results, None)

    assert "explored" in probe_results
    assert len(me._added) == 1  # also recorded into the store
    inp = build_stats_input(probe_results, data=None)
    assert "explored.small_and_peripheral" in inp.per_case


def test_loop_bridge_signals_is_noop_without_recipes():
    probe_results = _probe_results()
    me = _fake_loop_self([])

    VLDiagnoseLoop._bridge_signals(me, probe_results, None)

    assert "explored" not in probe_results
    assert me._added == []


def test_loop_bridge_raises_stats_signal_cap_so_bridged_signals_are_tested():
    """Regression: a low max_signal_tools must not silently cap the bridged signals
    (they are appended last to per_case). The bridge raises the cap to cover them."""
    from evalrx.analysis.stats_agent import StatsAnalysisAgent

    probe_results = _probe_results()  # saliency.obj_size + saliency.attention
    recipe = SignalRecipe(name="small", kind="expr", expr="saliency_obj_size < 40")
    agent = StatsAnalysisAgent(max_signal_tools=1)  # deliberately too small
    me = _fake_loop_self([recipe])
    me.stats_agent = agent

    VLDiagnoseLoop._bridge_signals(me, probe_results, None)

    # 3 signals now in the family: saliency.obj_size, saliency.attention, explored.small
    assert agent._max_signal_tools >= 3
