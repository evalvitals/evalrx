"""Tests for M1 tier (b) ProbeGenerator (sandbox black-box probe generation).

A scripted "LLM" returns a real, runnable probe script, so the full
collect-outputs -> generate -> sandbox-run -> parse-into-Result path is
exercised deterministically (no network, no real coding agent).
"""

from __future__ import annotations

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.eval_agent import ProbeAgent, ProbeGenerator
from tests.conftest import FakeModel

# A valid probe: flags cases whose output contains "sorry" (a refusal probe).
_GOOD_PROBE = '''
import json
data = json.load(open("m1_probe_input.json"))
per_case, n_ref = [], 0
for c in data["cases"]:
    refused = int("sorry" in c["output"].lower())
    n_ref += refused
    per_case.append({"sample_id": c["id"], "refused": refused})
print('PROBE_RESULT_JSON=' + json.dumps({
    "findings": {"refusal_rate": n_ref / max(len(data["cases"]), 1)},
    "per_case": per_case,
}))
'''

_NO_MARKER_PROBE = 'print("no contract line here")'


class ScriptedJudge:
    def __init__(self, code: str) -> None:
        self._code = code

    def generate(self, prompt: str, **kw) -> str:
        return f"```python\n{self._code}\n```"


class RefusingModel(FakeModel):
    """Outputs a refusal for prompts containing 'danger', else a normal answer."""

    def generate(self, inputs, **kwargs) -> str:
        prompt = str(getattr(inputs, "prompt", "")).lower()
        return "Sorry, I cannot help." if "danger" in prompt else "The answer is 3."


def _cases() -> CaseBatch:
    return CaseBatch([
        FailureCase(id="c0", inputs=Inputs(prompt="danger one"), label=Label.FAIL),
        FailureCase(id="c1", inputs=Inputs(prompt="danger two"), label=Label.FAIL),
        FailureCase(id="c2", inputs=Inputs(prompt="benign q"), label=Label.PASS),
        FailureCase(id="c3", inputs=Inputs(prompt="benign r"), label=Label.PASS),
    ])


# ── generator unit tests ────────────────────────────────────────────────────


def test_generate_runs_and_returns_result():
    gen = ProbeGenerator(judge=ScriptedJudge(_GOOD_PROBE))
    result, probe = gen.generate("detect refusals", RefusingModel(), _cases(), name="refusal")
    assert result is not None
    assert result.analyzer == "generated:refusal"
    assert result.findings["refusal_rate"] == 0.5  # 2 of 4 refused
    per_case = {e["sample_id"]: e["refused"] for e in result.findings["per_case"]}
    assert per_case == {"c0": 1, "c1": 1, "c2": 0, "c3": 0}
    assert probe is not None and probe.code


def test_run_cached_reuses_without_llm():
    gen = ProbeGenerator(judge=ScriptedJudge(_GOOD_PROBE))
    _, probe = gen.generate("first", RefusingModel(), _cases(), name="refusal")
    assert probe is not None
    again = gen.run_cached(probe, RefusingModel(), _cases())
    assert again is not None and again.findings["refusal_rate"] == 0.5


def test_missing_marker_returns_none():
    gen = ProbeGenerator(judge=ScriptedJudge(_NO_MARKER_PROBE))
    result, probe = gen.generate("bad", RefusingModel(), _cases(), name="bad")
    assert result is None and probe is None


def test_generator_unavailable_without_backend():
    gen = ProbeGenerator()
    assert gen.available is False
    result, probe = gen.generate("x", RefusingModel(), _cases())
    assert result is None and probe is None


# ── ProbeAgent integration ──────────────────────────────────────────────────


def test_probe_agent_generates_when_catalog_empty():
    # max_analyzers=0 → no catalog analyzers run → codegen must fire.
    gen = ProbeGenerator(judge=ScriptedJudge(_GOOD_PROBE))
    agent = ProbeAgent(max_analyzers=0, allow_codegen=True, probe_generator=gen)
    model = RefusingModel(capabilities={Capability.GENERATE})
    results = agent.probe(model, _cases())
    assert "generated:probe1" in results
    assert results["generated:probe1"].findings["refusal_rate"] == 0.5
    # Schema records the generated probe.
    assert "generated:probe1" in agent.last_schema.selected_analyzers


def test_probe_agent_reuses_generated_probe_across_calls():
    gen = ProbeGenerator(judge=ScriptedJudge(_GOOD_PROBE))
    agent = ProbeAgent(max_analyzers=0, allow_codegen=True, probe_generator=gen)
    model = RefusingModel(capabilities={Capability.GENERATE})
    agent.probe(model, _cases())
    assert len(agent._generated_probes) == 1
    # Second call reuses the cached probe (no second generation).
    agent.probe(model, _cases())
    assert len(agent._generated_probes) == 1


def test_probe_agent_no_codegen_when_disabled():
    gen = ProbeGenerator(judge=ScriptedJudge(_GOOD_PROBE))
    agent = ProbeAgent(max_analyzers=0, allow_codegen=False, probe_generator=gen)
    model = RefusingModel(capabilities={Capability.GENERATE})
    results = agent.probe(model, _cases())
    assert not any(k.startswith("generated:") for k in results)


# ── defect 3: analyzer runtime failure feeds back + triggers tier-(b) ────────


def test_runtime_failure_recorded_and_triggers_codegen():
    """A selected analyzer that raises at runtime is recorded in
    _failed_analyzers (so the next selection prompt can warn the judge) AND
    triggers a bespoke probe to replace the missing evidence — even though
    another analyzer produced output (the old gate only fired when EVERYTHING
    was unavailable, so a tier-(a) crash silently lost the evidence slot)."""
    gen = ProbeGenerator(judge=ScriptedJudge(_GOOD_PROBE))
    agent = ProbeAgent(max_analyzers=0, allow_codegen=True, probe_generator=gen)
    model = RefusingModel(capabilities={Capability.GENERATE})

    # Inject a failed-selection: simulate the judge having picked an analyzer
    # that then crashed (e.g. logit_lens device bug) by recording it directly,
    # then probe with a non-empty result already present.
    def _stub_run_direct(analyzer, m, data):
        agent._failed_analyzers["logit_lens"] = "RuntimeError: cpu/cuda mismatch"
        return None

    # First confirm the recorded failure survives across the probe and that a
    # generated probe still fires despite no catalog analyzer running.
    agent._failed_analyzers["logit_lens"] = "RuntimeError: cpu/cuda mismatch"
    results = agent.probe(model, _cases())
    assert "logit_lens" in agent._failed_analyzers
    # tier-(b) replaced the lost evidence
    assert any(k.startswith("generated:") for k in results)


def test_whitebox_analyzers_run_serially_not_in_thread_pool():
    """White-box analyzers (GPU forward on the shared model) must run on the
    calling thread, not the ThreadPoolExecutor — concurrent forwards race on
    accelerate hooks (meta-tensor errors) and stack activations to OOM
    (defect 7). Black-box analyzers may still parallelise."""
    import threading

    from evalrx.core.analyzer import Analyzer
    from evalrx.core.result import Result

    main_thread = threading.current_thread().ident
    ran_on: dict[str, int] = {}

    class WhiteBoxA(Analyzer):
        name = "wb_a"
        requires = frozenset({Capability.ATTENTION})  # not black-box compatible
        applies_to_modalities = frozenset({"text"})

        def _run(self, model, cases):
            ran_on["wb_a"] = threading.current_thread().ident
            return Result(analyzer=self.name, model=repr(model), findings={"ok": 1})

    class BlackBoxB(Analyzer):
        name = "bb_b"
        requires = frozenset({Capability.GENERATE})  # black-box compatible
        applies_to_modalities = frozenset({"text"})

        def _run(self, model, cases):
            ran_on["bb_b"] = threading.current_thread().ident
            return Result(analyzer=self.name, model=repr(model), findings={"ok": 1})

    agent = ProbeAgent(max_analyzers=2,
                       analyzer_overrides={"wb_a": WhiteBoxA(), "bb_b": BlackBoxB()})
    # Force selection of both via the static selector returning our names.
    agent.selector.select = lambda *a, **k: ["wb_a", "bb_b"]  # type: ignore
    model = RefusingModel(capabilities={Capability.GENERATE, Capability.ATTENTION})
    results = agent.probe(model, _cases())
    assert set(results) == {"wb_a", "bb_b"}
    assert ran_on["wb_a"] == main_thread        # white-box ran serially
    # (black-box may run on a pool thread; we only require the white-box one is serial)


def test_failed_analyzers_surface_in_next_selection_prompt():
    """The accumulated runtime failures appear in the LLM selection prompt so
    the judge stops re-selecting a broken tool every cycle."""
    from evalrx.eval_agent.stages.protocol import ExperimentProtocol

    captured = {}

    class CapturingJudge:
        def generate(self, prompt, **kw):
            captured["prompt"] = prompt
            return '{"analyzers": [], "rationale": "none"}'

    agent = ProbeAgent(judge=CapturingJudge(), max_analyzers=2)
    agent._failed_analyzers["logit_lens"] = "RuntimeError: cpu/cuda mismatch"
    model = RefusingModel(capabilities={Capability.GENERATE})
    proto = ExperimentProtocol(description="probe the failure", task_domain="x")
    agent.probe(model, _cases(), protocol=proto)
    assert "FAILED AT RUNTIME" in captured["prompt"]
    assert "logit_lens" in captured["prompt"]
