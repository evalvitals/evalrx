"""Batch generation: the Model default, the per-case probe driver, and the run log."""

from __future__ import annotations

import pytest

from evalrx.core.analyzer import Ask, labelled_answers, run_probes
from evalrx.core.case import FailureCase, Inputs
from evalrx.eval_agent.runlog.context import RunContext
from evalrx.eval_agent.runlog.instrumentation import InstrumentedModel
from evalrx.reporting.run_events import read_run_events
from tests.conftest import FakeModel


class Echo(FakeModel):
    """Answers ``<prompt>``; fails on a prompt containing ``boom``."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, dict]] = []

    def generate(self, inputs, **kwargs) -> str:
        prompt = inputs.prompt if isinstance(inputs, Inputs) else str(inputs)
        self.calls.append((prompt, kwargs))
        if "boom" in prompt:
            raise TimeoutError(prompt)
        return f"<{prompt}>"


class Batched(Echo):
    """A backend with a real batch call, which fails as a whole."""

    def generate_batch(self, inputs, **kwargs):
        self.batches = getattr(self, "batches", 0) + 1
        return [self.generate(item, **kwargs) for item in inputs]


def test_the_default_batch_asks_one_at_a_time_in_order():
    model = Echo()
    assert model.generate_batch(["a", "b"], temperature=1.0) == ["<a>", "<b>"]
    assert model.calls == [("a", {"temperature": 1.0}), ("b", {"temperature": 1.0})]
    with pytest.raises(TimeoutError):
        model.generate_batch(["a", "boom"])


def test_a_tolerant_batch_returns_the_error_in_place_without_repeating_calls():
    model = Echo()
    first, failed, last = model.try_generate_batch(["a", "boom", "c"])
    assert (first, last) == ("<a>", "<c>") and isinstance(failed, TimeoutError)
    assert [prompt for prompt, _ in model.calls] == ["a", "boom", "c"]


def test_a_tolerant_batch_tries_a_real_batch_call_first():
    model = Batched()
    assert model.try_generate_batch(["a", "b"]) == ["<a>", "<b>"] and model.batches == 1
    _, failed = model.try_generate_batch(["a", "boom"])  # the batch raised: asked one by one
    assert isinstance(failed, TimeoutError)


def test_probes_advance_together_and_each_gets_its_own_answers():
    model = Echo()

    def probe(name: str):
        first, second = yield Ask([f"{name}1", f"{name}2"])
        if name == "b":
            return [first, second]
        (third,) = yield Ask([f"{name}3 after {first}"], {"temperature": 1.0})
        return [first, second, third]

    rows = run_probes(model, [probe(name) for name in "abc"])

    assert rows == [["<a1>", "<a2>", "<a3 after <a1>>"], ["<b1>", "<b2>"],
                    ["<c1>", "<c2>", "<c3 after <c1>>"]]
    # one round per Ask: every case's first questions, then the follow-ups
    assert [prompt for prompt, _ in model.calls] == [
        "a1", "a2", "b1", "b2", "c1", "c2", "a3 after <a1>", "c3 after <c1>"]
    assert model.calls[-1][1] == {"temperature": 1.0}


def test_a_probe_that_asks_nothing_returns_its_row():
    def probe():
        return {"skipped": "no gold answer"}
        yield  # pragma: no cover - makes this a generator

    assert run_probes(Echo(), [probe()]) == [{"skipped": "no gold answer"}]


def test_a_tolerant_run_hands_the_probe_its_error():
    def probe(prompt: str):
        (answer,) = yield Ask([prompt])
        return "failed" if isinstance(answer, Exception) else answer

    assert run_probes(Echo(), [probe("a"), probe("boom")], tolerant=True) == ["<a>", "failed"]
    with pytest.raises(TimeoutError):
        run_probes(Echo(), [probe("boom")])


def test_labelled_answers_reads_the_stored_answer_and_asks_only_for_the_rest():
    model = Echo()
    cases = [FailureCase(inputs=Inputs(prompt="q1"), observed="stored"),
             FailureCase(inputs=Inputs(prompt="q2"))]
    assert labelled_answers(model, cases) == ["stored", "<q2>"]
    assert [prompt for prompt, _ in model.calls] == ["q2"]


def test_a_batch_is_logged_as_one_call_per_input_sharing_its_time(tmp_path):
    ctx = RunContext(tmp_path / "run", verbose=False)
    model = InstrumentedModel(Echo(), ctx.logger, cycle=0, analyzer="probe")

    assert model.generate_batch(["a", "b"]) == ["<a>", "<b>"]
    _, failed = model.try_generate_batch(["c", "boom"])
    assert isinstance(failed, TimeoutError)
    ctx.finalize()

    calls = [e for e in read_run_events(ctx.root) if e["event"] == "model_call"]
    assert [c["output"] for c in calls] == ["<a>", "<b>", "<c>", None]
    assert all(c["batch_size"] == 2 for c in calls) and "TimeoutError" in calls[-1]["error"]
    assert [c["call_index"] for c in calls] == [1, 2, 3, 4]


def test_a_model_runs_its_concurrency_of_generate_calls_at_once_and_keeps_the_order():
    import threading
    import time

    class Endpoint(Echo):
        concurrency = 4

        def __init__(self) -> None:
            super().__init__()
            self.now = self.peak = 0
            self._lock = threading.Lock()

        def generate(self, inputs, **kwargs) -> str:
            with self._lock:
                self.now += 1
                self.peak = max(self.peak, self.now)
            time.sleep(0.05)
            with self._lock:
                self.now -= 1
            return super().generate(inputs, **kwargs)

    model = Endpoint()
    prompts = [str(i) for i in range(12)]

    assert model.generate_batch(prompts) == [f"<{p}>" for p in prompts]
    assert model.peak == 4
    out = model.try_generate_batch(["a", "boom", "b"])
    assert out[0] == "<a>" and isinstance(out[1], TimeoutError) and out[2] == "<b>"


def test_logprobs_batch_scores_each_input_and_logs_one_call_each(tmp_path):
    ctx = RunContext(tmp_path / "run", verbose=False)
    model = InstrumentedModel(FakeModel(), ctx.logger, cycle=0, analyzer="probe")

    rows = model.logprobs_batch(["a", "b", "c"])
    ctx.finalize()

    assert len(rows) == 3 and all(row and row[0].token for row in rows)
    calls = [e for e in read_run_events(ctx.root) if e["event"] == "model_call"]
    assert [c["method"] for c in calls] == ["logprobs"] * 3
    assert all(c["batch_size"] == 3 for c in calls)
