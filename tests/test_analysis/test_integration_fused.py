"""Phase D — cross-phase integration: real explorer codegen -> bridge -> M2 confirm.

Ties the per-phase units together end to end and pins the guarantees that only
emerge across phases (real sandbox explorer, in-loop verdict, no e-value merging).
"""

from __future__ import annotations

from evalrx.analysis import ExploratoryAnalysisAgent, run_fused_analysis
from evalrx.analysis.adjudicate import adjudicate_signals
from evalrx.analysis.explorer import CandidateSignal
from evalrx.analysis.operationalize import SignalRecipe, bridge_recipes_to_result
from evalrx.analysis.stats_agent import StatsAnalysisAgent
from evalrx.analysis.stats_tools import build_stats_input

# A real explorer script: reads records.json, proposes a candidate that carries a
# deterministic recipe over the existing column. The fused pipeline runs this in a
# sandbox on EXPLORE, then compiles the recipe on the disjoint CONFIRM split.
_RECIPE_SCRIPT = """
import json
from pathlib import Path

rows = json.loads(Path("records.json").read_text())
payload = {
    "observations": ["small objects appear to fail more"],
    "candidate_signals": [{
        "name": "explored.small",
        "rationale": "small objects fail",
        "suggested_test": "signal_label_assoc",
        "recipe": {"name": "explored.small", "kind": "expr", "expr": "obj_size < 40"},
    }],
    "plots": [], "tables": {}, "charts": [], "caveats": [],
    "recommended_confirmatory_tests": [],
}
print("EXPLORATORY_RESULT_JSON=" + json.dumps(payload))
"""


class _ScriptedJudge:
    def __init__(self, code: str) -> None:
        self._code = code

    def generate(self, prompt: str, **kwargs) -> str:
        return f"```python\n{self._code}\n```"


def _dataset(n_each: int = 15) -> list[dict]:
    fails = [{"case_id": f"f{i}", "label": "fail", "obj_size": 20} for i in range(n_each)]
    passes = [{"case_id": f"p{i}", "label": "pass", "obj_size": 80} for i in range(n_each)]
    return fails + passes


def test_real_explorer_recipe_is_bridged_and_confirmed_on_held_out(tmp_path):
    from evalrx.agent_runtime.sandbox import ExperimentSandbox

    explorer = ExploratoryAnalysisAgent(
        judge=_ScriptedJudge(_RECIPE_SCRIPT),
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
    )
    rep = run_fused_analysis(_dataset(), explorer=explorer, confirm_split=0.3, seed=0)

    sig = next(s for s in rep.candidate_signals if s.name == "explored.small")
    assert sig.source == "explorer"
    assert sig.host_adjudicated is True
    assert sig.reject is True              # confirmed on the held-out split
    assert sig.confirmed_on == "held_out"
    assert rep.split["mode"] == "held_out"


def test_inloop_bridged_signal_gets_a_real_m2_verdict():
    """Phase C -> M2: a synthetic-Result composite signal is tested by M2."""
    class _FakeResult:
        def __init__(self, per_case):
            self.findings = {"per_case": per_case}

    per_case = (
        [{"case_id": f"f{i}", "obj_size": 20, "attention": 0.1} for i in range(10)]
        + [{"case_id": f"p{i}", "obj_size": 80, "attention": 0.6} for i in range(10)]
    )
    probe_results = {"saliency": _FakeResult(per_case)}

    recipe = SignalRecipe(
        name="small_and_peripheral", kind="expr",
        expr="(saliency_obj_size < 40) and (saliency_attention < 0.3)",
    )
    probe_results["explored"] = bridge_recipes_to_result([recipe], probe_results, data=None)

    inp = build_stats_input(probe_results, data=None)
    # attach labels (fails are the small/peripheral ones)
    inp.labels = {f"f{i}": True for i in range(10)} | {f"p{i}": False for i in range(10)}

    report = StatsAnalysisAgent(max_signal_tools=16).analyze_input(inp)
    verdicts = {
        (r.config or {}).get("signal"): r
        for r in report.stats_results
        if r.ok and r.tool == "signal_label_assoc"
    }
    assert "explored.small_and_peripheral" in verdicts
    assert verdicts["explored.small_and_peripheral"].reject is True


def test_distinct_signals_are_distinct_family_members_not_merged():
    """No e-value merging: each estimand is its own e-BH member (DESIGN §3.3)."""
    strong = CandidateSignal(name="a", sufficient={"kind": "paired_binary", "b": 20, "c": 0})
    also = CandidateSignal(name="b", sufficient={"kind": "paired_binary", "b": 18, "c": 0})
    meta = adjudicate_signals([strong, also])

    # two separate members competing in ONE e-BH family — never combined into one
    assert meta["n_in_family"] == 2
    assert {"a", "b"} == set(meta["rejected"])  # both strong enough to survive e-BH
