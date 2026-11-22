"""M1 stays bounded: an analyzer cap, a time budget, and no analyzers that read one case."""

from __future__ import annotations

import time

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.registry import registry
from evalrx.eval_agent.stages.probe_agent import ProbeAgent
from tests.conftest import FakeModel


def _batch(n: int) -> CaseBatch:
    return CaseBatch([
        FailureCase(inputs=Inputs(prompt=f"What is {i}+{i}?"), expected=str(2 * i),
                    observed=str(2 * i), label=Label.FAIL if i % 2 else Label.PASS)
        for i in range(n)
    ])


def _single_case_names() -> set[str]:
    return {name for name in registry.analyzers.list()
            if getattr(registry.analyzers.get(name), "single_case", False)}


def test_an_analyzer_that_reads_one_case_is_not_selected_for_a_batch():
    model = FakeModel(capabilities={Capability.GENERATE, Capability.LOGPROBS})
    assert {"self_consistency", "verbalized_confidence"} <= _single_case_names()

    agent = ProbeAgent(max_analyzers=None)
    agent.probe(model, _batch(4))
    assert not set(agent.last_schema.selected_analyzers) & _single_case_names()

    one = ProbeAgent(max_analyzers=None)
    one.probe(model, _batch(1))
    assert set(one.last_schema.selected_analyzers) & _single_case_names()


def test_a_caller_that_names_a_single_case_analyzer_still_gets_it():
    model = FakeModel(capabilities={Capability.GENERATE})
    results = ProbeAgent().probe(model, _batch(3), analyzers=["self_consistency"])
    assert set(results) == {"self_consistency"}   # and nothing was added to reach the cap


def test_the_default_cap_bounds_how_many_analyzers_run():
    model = FakeModel(capabilities={Capability.GENERATE, Capability.LOGPROBS})
    agent = ProbeAgent()
    agent.probe(model, _batch(4))
    assert 0 < len(agent.last_schema.selected_analyzers) <= 6


def test_analyzers_that_have_not_started_when_the_budget_is_spent_are_skipped():
    class Slow(FakeModel):
        def generate(self, inputs, **kwargs) -> str:
            time.sleep(0.05)
            return "Answer: 1"

    lines: list[str] = []

    class Console:
        current_cycle = 0

        def progress(self, stage: str, text: str) -> None:
            lines.append(f"{stage} {text}")

        def log_model_call(self, **record) -> None:
            pass

    agent = ProbeAgent(time_budget_sec=0.01, max_workers=1, run_logger=Console())
    results = agent.probe(Slow(capabilities={Capability.GENERATE}), _batch(2),
                          analyzers=["prompt_contrast", "termination_audit"])

    assert set(results) == {"prompt_contrast"}       # the first one always runs
    assert "termination_audit" in agent.last_schema.rationale
    assert "time budget" in agent.last_schema.rationale
    assert any("prompt_contrast: done" in line for line in lines)
    assert any("termination_audit: skipped" in line for line in lines)


def test_an_api_model_never_sees_more_than_its_concurrency_in_flight():
    """Analyzers batch their questions; running analyzers in threads on top of
    that would multiply the requests an endpoint receives."""
    import threading

    from evalrx.models import RuntimeConfig, compose

    lock = threading.Lock()
    live, peak = [0], [0]

    def _generate(prompt, model="", **kw):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.01)
        with lock:
            live[0] -= 1
        return "Answer: 1"

    served = compose("qwen3.5-2b", "api", RuntimeConfig(generate_fn=_generate, concurrency=3), set())
    results = ProbeAgent(max_workers=8).probe(
        served, _batch(6), analyzers=["prompt_contrast", "perturbation_battery", "cot_faithfulness"])

    assert len(results) == 3
    assert peak[0] == 3


def test_the_cap_does_not_shorten_a_list_the_caller_named():
    model = FakeModel(capabilities={Capability.GENERATE})
    named = ["prompt_contrast", "termination_audit", "self_repair"]
    results = ProbeAgent(max_analyzers=2).probe(model, _batch(4), analyzers=named)
    assert set(results) == set(named)
