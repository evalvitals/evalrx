"""The fix stage asks the model in batches and owns no thread pool.

What runs at once is the model's business (``Model.generate_batch``): the
baseline samples, a template or spec candidate, and the calls a coded pipeline
makes at the same moment each reach the model as one batch.
"""

from __future__ import annotations

import threading

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model
from evalrx.eval_agent.stages.fix.agent import FixAgent, FixCandidate
from evalrx.eval_agent.stages.fix.pipeline import run_coded_pipeline
from evalrx.eval_agent.stages.fix.tiers import FixTier
from evalrx.eval_agent.stages.fix.tools import PipelineSpec, SpecPipelineModel, run_pipelines


class Batched(Model):
    """Answers "<prompt> -> yes" and records every batch it was given."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self, fail_on: str = "") -> None:
        self.batches: list[list[str]] = []
        self.threads: set[int] = set()
        self.fail_on = fail_on

    def generate(self, inputs, **kwargs):
        prompt = str(getattr(inputs, "prompt", inputs))
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("adapter hiccup")
        return f"{prompt} -> yes"

    def generate_batch(self, inputs, **kwargs):
        self.threads.add(threading.get_ident())
        self.batches.append([str(getattr(item, "prompt", item)) for item in inputs])
        return [self.generate(item, **kwargs) for item in inputs]

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class Exchanges:
    """A run logger that keeps the fix stage's model exchanges."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log_model_exchange(self, stage, **fields):
        self.calls.append(fields)


def _cases(n: int = 4, observed: "str | None" = "no") -> CaseBatch:
    return CaseBatch([
        FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"Q{i}"), observed=observed,
                    label=Label.FAIL, expected="yes")
        for i in range(n)
    ])


def _says_yes(case, output):
    return "yes" in output


def _agent(**kwargs) -> FixAgent:
    return FixAgent(judge=None, max_tier="L2", score_fn=_says_yes, floor_candidates=(), **kwargs)


def test_a_template_candidate_is_one_batch_and_one_exchange_per_case():
    log = Exchanges()
    agent = _agent(run_logger=log)
    model = Batched()
    cand = FixCandidate(tier=FixTier.L1_PROMPT, name="t", kind="template",
                        payload={"prompt_template": "Think. {prompt}"})

    scores = agent.validator.scores(cand, model, _cases())

    assert model.batches == [["Think. Q0", "Think. Q1", "Think. Q2", "Think. Q3"]]
    assert scores == {f"c{i}": True for i in range(4)}
    assert [call["metadata"]["case_id"] for call in log.calls] == ["c0", "c1", "c2", "c3"]
    assert all(call["operation"] == "template_candidate" for call in log.calls)


def test_a_candidate_that_leaves_a_case_untouched_is_not_run_on_it():
    model = Batched()
    cand = FixCandidate(tier=FixTier.L1_PROMPT, name="identity", kind="template",
                        payload={"prompt_template": "{prompt}"})

    scores = _agent().validator.scores(cand, model, _cases())

    assert model.batches == [] and set(scores.values()) == {None}


def test_one_failed_case_scores_none_and_the_rest_keep_their_answers():
    model = Batched(fail_on="Q2")
    for kind, payload in (("template", {"prompt_template": "x {prompt}"}),
                          ("spec", {"name": "s", "prompt_template": "x {prompt}"})):
        cand = FixCandidate(tier=FixTier.L2_SCAFFOLD, name="t", kind=kind, payload=payload)
        scores = _agent().validator.scores(cand, model, _cases())
        assert scores == {"c0": True, "c1": True, "c2": None, "c3": True}, kind


def test_a_multi_step_spec_advances_every_case_and_sample_together():
    model = Batched()
    spec = PipelineSpec(name="refine", strategy="self_refine", n_samples=3)
    cases = list(_cases(4))

    runs = run_pipelines(model, cases, spec)

    # Three rounds (draft, feedback, final); each holds every case x sample.
    assert [len(batch) for batch in model.batches] == [12, 12, 12]
    assert model.batches[0].count("Q0") == 3  # identical prompts stay separate rows
    assert all(len(run.calls) == 9 and len(run.outputs) == 3 for run in runs)
    assert all("Attempt:" in prompt for prompt in model.batches[1])
    assert all(f"Q{i}" in run.winner for i, run in enumerate(runs))


def test_a_deployed_spec_answers_a_batch_through_the_same_path():
    model = Batched()
    deployed = SpecPipelineModel(model, PipelineSpec(name="s", prompt_template="x {prompt}",
                                                     n_samples=2))

    answers = deployed.generate_batch([Inputs(prompt="A"), Inputs(prompt="B")])

    assert model.batches == [["x A", "x A", "x B", "x B"]]
    assert answers == ["x A -> yes", "x B -> yes"]
    assert deployed.generate(Inputs(prompt="C")) == "x C -> yes"


def test_the_fresh_baseline_samples_are_one_batch():
    log = Exchanges()
    agent = _agent(run_logger=log, baseline_repeats=3)
    model = Batched()

    baseline = agent.validator.baseline(model, _cases(2))

    # The recorded answer is sample 1; two fresh samples per case, all at once.
    assert model.batches == [["Q0", "Q0", "Q1", "Q1"]]
    assert baseline.counts == {"c0": 3, "c1": 3}
    assert baseline.rates == {"c0": 2 / 3, "c1": 2 / 3}
    assert [call["operation"] for call in log.calls] == ["fresh_baseline"] * 4


_ONE_THREAD_PER_CASE = """
import json
from concurrent.futures import ThreadPoolExecutor
cases = json.load(open("fix_cases.json"))["cases"]
def solve(c):
    first = model_generate(c["id"], prompt="A " + c["prompt"])
    second = model_generate(c["id"], prompt="B " + first)
    return {"sample_id": c["id"], "output": second}
with ThreadPoolExecutor(max_workers=len(cases)) as pool:
    rows = list(pool.map(solve, cases))
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": rows}))
"""


def test_the_bridge_answers_simultaneous_calls_as_one_batch_from_one_thread(tmp_path):
    model = Batched()
    cases = _cases(8)
    logged: list[dict] = []

    result = run_coded_pipeline(_ONE_THREAD_PER_CASE, model, cases, workdir=tmp_path,
                                timeout_sec=60, call_logger=logged.append)

    assert result.ok, result.error
    assert result.outputs == {f"c{i}": f"B A Q{i} -> yes -> yes" for i in range(8)}
    assert sum(len(batch) for batch in model.batches) == 16 == result.n_calls == len(logged)
    # Two rounds of eight; a slow machine may split a round, never one call each.
    assert len(model.batches) <= 6 and max(len(batch) for batch in model.batches) > 1
    assert len(model.threads) == 1 and threading.get_ident() not in model.threads


def test_the_bridge_batches_by_decoding_setting_and_replays_the_recorded_answer(tmp_path):
    code = """
import json
from concurrent.futures import ThreadPoolExecutor
cases = json.load(open("fix_cases.json"))["cases"]
def solve(c):
    kw = {"temperature": 0.7} if c["id"] in ("c0", "c1") else None
    return {"sample_id": c["id"],
            "output": model_generate(c["id"]) + "|" + model_generate(
                c["id"], prompt="P " + c["prompt"], generation_kwargs=kw)}
with ThreadPoolExecutor(max_workers=len(cases)) as pool:
    rows = list(pool.map(solve, cases))
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": rows}))
"""
    model = Batched()

    result = run_coded_pipeline(code, model, _cases(4), workdir=tmp_path, timeout_sec=60)

    assert result.ok, result.error
    assert result.outputs == {f"c{i}": f"no|P Q{i} -> yes" for i in range(4)}
    assert result.n_replayed == 4  # the plain calls never reached the model
    asked = sorted(prompt for batch in model.batches for prompt in batch)
    assert asked == ["P Q0", "P Q1", "P Q2", "P Q3"]
    # Sampled and default decoding never share a batch.
    assert not any({"P Q0", "P Q1"} & set(batch) and {"P Q2", "P Q3"} & set(batch)
                   for batch in model.batches)


def test_the_baseline_is_measured_once_for_a_model_and_its_cases():
    agent = _agent(baseline_repeats=3)
    model, cases = Batched(), _cases(2)

    first = agent.validator.baseline(model, cases)
    again = agent.validator.baseline(model, cases)     # the next tier of the same search
    other = agent.validator.baseline(model, _cases(3))  # another partition

    assert again is first and other is not first
    assert [len(batch) for batch in model.batches] == [4, 6]
