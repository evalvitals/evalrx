

# ── discovery hands the model one batch; the backend decides how many at once ─

def test_discovery_keeps_runtime_concurrency_requests_in_flight_on_an_api_handle():
    """A served model can only batch the requests it has been handed.

    Discovery issued one generate at a time, so a local `vllm serve` sat at
    "Running: 1 reqs" for the whole baseline — the backend was swapped for
    throughput and then fed serially. The API backend now sends
    ``runtime.concurrency`` requests at once, and never more.
    """
    import threading

    from evalrx.core.case import FailureCase, Inputs
    from evalrx.eval_agent.stages.case_discovery import CaseDiscoveryAgent
    from evalrx.models import RuntimeConfig, compose

    peak = {"n": 0}
    live = {"n": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(4, timeout=5)

    def _generate(prompt, model="", **kw):
        with lock:
            live["n"] += 1
            peak["n"] = max(peak["n"], live["n"])
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        with lock:
            live["n"] -= 1
        return f"answer to {prompt}"

    cases = [
        FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"q{i}"), expected="answer to q%d" % i)
        for i in range(4)
    ]
    served = compose("qwen3.5-2b", "api", RuntimeConfig(generate_fn=_generate, concurrency=4), set())
    report = CaseDiscoveryAgent(
        scorer=lambda case, observed: observed == case.expected,
    ).discover(served, cases)

    assert peak["n"] == 4, f"expected 4 concurrent generates, saw {peak['n']}"
    # Order and labels are unchanged by the fan-out: only generation is parallel.
    assert [c.id for c in report.cases] == ["c0", "c1", "c2", "c3"]
    assert report.n_pass == 4 and report.n_fail == 0


def test_a_model_without_its_own_batch_call_is_asked_one_case_at_a_time():
    import threading

    from evalrx.core.case import FailureCase, Inputs
    from evalrx.eval_agent.stages.case_discovery import CaseDiscoveryAgent
    from tests.conftest import FakeModel

    threads: set[int] = set()

    class _Local(FakeModel):
        def generate(self, inputs, **kw):
            threads.add(threading.get_ident())
            return "x"

    cases = [FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"q{i}"), expected="x") for i in range(5)]
    report = CaseDiscoveryAgent(scorer=lambda case, observed: observed == case.expected).discover(
        _Local(), cases)
    assert report.n_pass == 5 and threads == {threading.get_ident()}


def test_a_case_whose_generation_raises_still_lands_as_unknown():
    """The fan-out carries the exception instead of raising it, so the batch
    survives — the same guarantee the sequential loop always gave."""
    from evalrx.core.case import FailureCase, Inputs, Label
    from evalrx.eval_agent.stages.case_discovery import CaseDiscoveryAgent
    from evalrx.models import RuntimeConfig, compose

    def _generate(prompt, model="", **kw):
        if prompt == "q1":
            raise RuntimeError("endpoint refused")
        return "ok"

    cases = [FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"q{i}"), expected="ok") for i in range(3)]
    served = compose("qwen3.5-2b", "api", RuntimeConfig(generate_fn=_generate, concurrency=3), set())
    report = CaseDiscoveryAgent(
        scorer=lambda case, observed: observed == case.expected, include_unknown=True,
    ).discover(served, cases)

    labels = {c.id: c.label for c in report.cases}
    assert labels["c0"] is Label.PASS and labels["c2"] is Label.PASS
    assert labels["c1"] is Label.UNKNOWN
    assert any("endpoint refused" in e for e in report.errors)
