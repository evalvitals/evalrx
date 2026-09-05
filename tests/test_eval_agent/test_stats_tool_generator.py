"""Tests for M2 tier (b) StatsToolGenerator (sandbox code generation).

These use a scripted "LLM" that returns a real, runnable statistics script, so
the full generate -> sandbox-execute -> parse-result path is exercised
deterministically (no network, no real coding agent).
"""

from __future__ import annotations

from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.result import Result
from evalrx.eval_agent import (
    StatsAnalysisAgent,
    StatsToolGenerator,
    build_stats_input,
)

# A minimal, valid generated tool: reads the input file, computes the FAIL-rate
# difference between signal-present and signal-absent cases, prints the contract.
_GOOD_SCRIPT = '''
import json
data = json.load(open("m2_stats_input.json"))
labels = data["labels"]
sig = next(iter(data["per_case"].values()), {})
s = [int(labels[c]) for c in labels if sig.get(c, 0)]
ctl = [int(labels[c]) for c in labels if not sig.get(c, 0)]
rs = sum(s)/len(s) if s else 0.0
rc = sum(ctl)/len(ctl) if ctl else 0.0
eff = rs - rc
print("computed effect", eff)
print('STATS_RESULT_JSON=' + json.dumps({
    "summary": f"custom fail-rate diff = {eff:.2f}",
    "effect": eff, "ci": None, "underpowered": False,
    "details": {"n_signal": len(s), "n_control": len(ctl)},
    "sufficient": {"kind": "two_group", "a": ctl, "b": s},
}))
'''

_NO_MARKER_SCRIPT = 'print("I forgot the contract line")'

_USES_EVALRX = '''
import json
from evalrx.stats import compare
data = json.load(open("m2_stats_input.json"))
labels = data["labels"]
sig = next(iter(data["per_case"].values()), {})
s = [int(labels[c]) for c in labels if sig.get(c, 0)]
ctl = [int(labels[c]) for c in labels if not sig.get(c, 0)]
r = compare(ctl, s, paired=False)   # generated code may use evalrx.stats
print('STATS_RESULT_JSON=' + json.dumps({
    "summary": r.summary(), "effect": r.effect, "ci": list(r.ci),
    "underpowered": r.underpowered, "details": {},
    "sufficient": {"kind": "two_group", "a": ctl, "b": s},
}))
'''

# A tool that does NO real test but self-declares a rejection with a fabricated
# e-value: the host must IGNORE the self-declared verdict (descriptive only).
_LIES_SCRIPT = '''
import json
print('STATS_RESULT_JSON=' + json.dumps({
    "summary": "I claim significance with no statistic",
    "effect": 0.9, "ci": [0.8, 1.0], "reject": True, "e_value": 9999.0,
    "p_value": 0.0001, "underpowered": False, "details": {}, "sufficient": None,
}))
'''


class ScriptedJudge:
    """A fake judge whose generate() returns a fixed code block."""

    def __init__(self, code: str) -> None:
        self._code = code

    def generate(self, prompt: str, **kw) -> str:
        return f"```python\n{self._code}\n```"


def _inp():
    cases = [
        FailureCase(id="c0", inputs=Inputs(prompt="a"), label=Label.FAIL),
        FailureCase(id="c1", inputs=Inputs(prompt="b"), label=Label.FAIL),
        FailureCase(id="c2", inputs=Inputs(prompt="c"), label=Label.PASS),
        FailureCase(id="c3", inputs=Inputs(prompt="d"), label=Label.PASS),
    ]
    per_case = [{"sample_id": f"c{i}", "flag": 1 if i < 2 else 0} for i in range(4)]
    res = {"a": Result(analyzer="a", model="m", findings={"per_case": per_case})}
    return build_stats_input(res, CaseBatch(cases))


# ── generator unit tests ────────────────────────────────────────────────────


def test_generate_runs_and_parses_contract():
    gen = StatsToolGenerator(judge=ScriptedJudge(_GOOD_SCRIPT))
    result, tool = gen.generate("test signal vs fail", _inp(), name="diff")
    assert result.ok, result.error
    assert result.tool == "generated:diff"
    assert result.effect == 1.0 and result.reject is True
    assert result.details["n_signal"] == 2 and result.details["n_control"] == 2
    assert tool is not None and tool.code


def test_generated_tool_can_import_evalrx_stats():
    gen = StatsToolGenerator(judge=ScriptedJudge(_USES_EVALRX))
    result, tool = gen.generate("rigorous compare", _inp(), name="cmp")
    assert result.ok, result.error
    assert result.effect == 1.0 and result.ci == (1.0, 1.0)
    assert result.reject is True


def test_run_cached_reuses_without_llm():
    gen = StatsToolGenerator(judge=ScriptedJudge(_GOOD_SCRIPT))
    _, tool = gen.generate("first", _inp(), name="diff")
    assert tool is not None
    # Re-run on fresh data, no judge call needed.
    again = gen.run_cached(tool, _inp())
    assert again.ok and again.effect == 1.0


def test_generate_missing_marker_is_not_ok():
    gen = StatsToolGenerator(judge=ScriptedJudge(_NO_MARKER_SCRIPT))
    result, tool = gen.generate("bad", _inp(), name="bad")
    assert not result.ok and "STATS_RESULT_JSON" in (result.error or "")
    assert tool is None


def test_self_declared_reject_is_ignored():
    """A generated tool that self-declares reject/e_value but supplies no
    adjudicable sufficient statistic is treated as DESCRIPTIVE: the host never
    trusts the LLM's verdict, so it cannot reach M4's headline."""
    gen = StatsToolGenerator(judge=ScriptedJudge(_LIES_SCRIPT))
    result, _ = gen.generate("lie", _inp(), name="liar")
    assert result.ok
    assert result.reject is False          # self-declared True was ignored
    assert result.e_value is None          # fabricated e-value dropped
    assert result.p_value is None
    assert result.details.get("descriptive_only") is True


def test_host_reconstructs_e_value_from_paired_binary():
    """A paired_binary sufficient statistic is adjudicated host-side: the host
    computes the e-value from (b, c) and decides reject — not the script."""
    script = '''
import json
print('STATS_RESULT_JSON=' + json.dumps({
    "summary": "paired flip", "effect": None, "ci": None,
    "underpowered": False, "details": {},
    "sufficient": {"kind": "paired_binary", "b": 20, "c": 0},
}))
'''
    gen = StatsToolGenerator(judge=ScriptedJudge(script))
    result, _ = gen.generate("paired", _inp(), name="mc")
    assert result.ok
    assert result.reject is True
    assert result.e_value is not None and result.e_value > 20  # host-computed
    assert result.details.get("host_adjudicated") is True


def test_generator_unavailable_without_backend():
    gen = StatsToolGenerator()  # no judge, no cli
    assert gen.available is False
    result, tool = gen.generate("x", _inp())
    assert not result.ok and tool is None


# ── integration with StatsAnalysisAgent ─────────────────────────────────────


def _no_signal_inp_results():
    """Results+data where catalog tools cannot produce a usable signal verdict."""
    # No per_case signals and no PASS cases → signal_label_assoc N/A; only
    # single_rate runs.  Force codegen by having all-FAIL (single_rate still ok),
    # so instead drop labels entirely to make everything unusable.
    cases = [FailureCase(id="c0", inputs=Inputs(prompt="a"), label=Label.UNKNOWN)]
    res = {"a": Result(analyzer="a", model="m",
                       findings={"by_strategy": {"s0": {"c0": 1}}})}
    return res, CaseBatch(cases)


def test_agent_codegen_fires_when_catalog_empty():
    # Build a scenario where the catalog yields nothing ok, so codegen kicks in.
    res, data = _no_signal_inp_results()
    agent = StatsAnalysisAgent(
        judge=ScriptedJudge(_GOOD_SCRIPT),
        allow_codegen=True,
    )
    rep = agent.analyze(res, model_name="m", data=data)
    # A generated tool should appear in the plan/results.
    assert any(p["tool"].startswith("generated:") for p in rep.stats_plan)


def test_agent_no_codegen_when_disabled():
    res, data = _no_signal_inp_results()
    agent = StatsAnalysisAgent(judge=ScriptedJudge(_GOOD_SCRIPT), allow_codegen=False)
    rep = agent.analyze(res, model_name="m", data=data)
    assert not any(p["tool"].startswith("generated:") for p in rep.stats_plan)
    assert not any(r.tool.startswith("generated:") for r in rep.stats_results)
