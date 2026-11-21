

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


# ── a sampling model: the label is the majority of several answers ───────────

def _scripted(answers_by_prompt: "dict[str, list[str]]"):
    """A model that gives each prompt its scripted answers in turn, and the batches it saw."""
    from tests.conftest import FakeModel

    class _Scripted(FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.batches: list[list[str]] = []
            self._left = {prompt: list(answers) for prompt, answers in answers_by_prompt.items()}

        def generate(self, inputs, **kw):
            answer = self._left[inputs.prompt].pop(0)
            if answer == "RAISE":
                raise RuntimeError("endpoint refused")
            return answer

        def try_generate_batch(self, inputs, **kw):
            self.batches.append([item.prompt for item in inputs])
            answers: list = []
            for item in inputs:
                try:
                    answers.append(self.generate(item, **kw))
                except RuntimeError as exc:
                    answers.append(exc)
            return answers

    return _Scripted()


def _discover(model, prompts, n_samples, **kwargs):
    from evalrx.core.case import FailureCase, Inputs
    from evalrx.eval_agent.stages.case_discovery import CaseDiscoveryAgent

    cases = [FailureCase(id=f"c{i}", inputs=Inputs(prompt=p), expected="yes") for i, p in enumerate(prompts)]
    return CaseDiscoveryAgent(
        scorer=lambda case, observed: observed.startswith(case.expected),
        n_samples=n_samples, **kwargs,
    ).discover(model, cases)


def test_a_case_is_labelled_by_the_majority_of_its_sampled_answers():
    from evalrx.core.case import Label

    model = _scripted({"q0": ["no", "yes 1", "yes 2"], "q1": ["yes", "no 1", "no 2"]})

    report = _discover(model, ["q0", "q1"], n_samples=3)

    # Every sample of every case in one batch, a case's samples side by side.
    assert model.batches == [["q0", "q0", "q0", "q1", "q1", "q1"]]
    c0, c1 = report.cases
    assert (c0.label, c1.label) == (Label.PASS, Label.FAIL)
    assert (report.n_pass, report.n_fail, report.n_unknown) == (1, 1, 0)
    # The recorded answer is the first one that earned the label, not the first drawn.
    assert (c0.observed, c1.observed) == ("yes 1", "no 1")
    assert c0.metadata["discovery_votes"] == {"pass": 2, "fail": 1}
    assert c1.metadata["discovery_votes"] == {"pass": 1, "fail": 2}
    assert c0.metadata["discovery_reason"].startswith("2 of 3 sampled answers passed")


def test_one_answer_per_case_records_no_votes():
    report = _discover(_scripted({"q0": ["yes"]}), ["q0"], n_samples=1)

    assert "discovery_votes" not in report.cases[0].metadata
    assert report.cases[0].metadata["discovery_reason"] == "scored by injected scorer"


def test_a_failed_sample_does_not_vote_and_an_even_split_is_unknown():
    from evalrx.core.case import Label

    model = _scripted({"q0": ["RAISE", "yes", "yes"], "q1": ["RAISE", "yes", "no"]})

    report = _discover(model, ["q0", "q1"], n_samples=3, include_unknown=True)

    c0, c1 = report.cases
    assert c0.label is Label.PASS and c0.observed == "yes"
    assert "discovery_error" not in c0.metadata  # the case has a label; the error is in the report
    assert c1.label is Label.UNKNOWN and "endpoint refused" in c1.metadata["discovery_error"]
    assert sum("endpoint refused" in e for e in report.errors) == 2


def test_fewer_than_one_sample_is_refused():
    import pytest

    from evalrx.eval_agent.stages.case_discovery import CaseDiscoveryAgent

    with pytest.raises(ValueError, match="n_samples"):
        CaseDiscoveryAgent(n_samples=0)
