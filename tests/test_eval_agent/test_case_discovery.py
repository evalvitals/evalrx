

# ── discovery may fan out, but only where fanning out is safe ────────────────

def test_discovery_fans_out_only_for_an_api_handle():
    """A served model can only batch the requests it has been handed.

    Discovery issued one generate at a time, so a local `vllm serve` sat at
    "Running: 1 reqs" for the whole baseline — the backend was swapped for
    throughput and then fed serially. A local backend must still be forced to
    1: it shares one GPU and is not thread-safe.
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
    served = compose("qwen3.5-2b", "api", RuntimeConfig(generate_fn=_generate), set())
    report = CaseDiscoveryAgent(
        scorer=lambda case, observed: observed == case.expected, concurrency=4,
    ).discover(served, cases)

    assert peak["n"] == 4, f"expected 4 concurrent generates, saw {peak['n']}"
    # Order and labels are unchanged by the fan-out: only generation is parallel.
    assert [c.id for c in report.cases] == ["c0", "c1", "c2", "c3"]
    assert report.n_pass == 4 and report.n_fail == 0


def test_a_local_backend_is_forced_back_to_one():
    from evalrx.eval_agent.stages.case_discovery import CaseDiscoveryAgent

    class _Local:
        def generate(self, inputs, **kw):
            return "x"

    agent = CaseDiscoveryAgent(concurrency=8)
    assert agent._workers(_Local(), 20) == 1


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
    served = compose("qwen3.5-2b", "api", RuntimeConfig(generate_fn=_generate), set())
    report = CaseDiscoveryAgent(
        scorer=lambda case, observed: observed == case.expected,
        concurrency=3, include_unknown=True,
    ).discover(served, cases)

    labels = {c.id: c.label for c in report.cases}
    assert labels["c0"] is Label.PASS and labels["c2"] is Label.PASS
    assert labels["c1"] is Label.UNKNOWN
    assert any("endpoint refused" in e for e in report.errors)
