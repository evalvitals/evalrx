"""Every target-model call an analyzer makes must be durably recorded.

Before InstrumentedModel existed, an analyzer like self_consistency could call
``model.generate()`` N times per case and reduce that to one derived score
(``consistency``) with the N raw generations gone — no way to see what the
model was actually asked or actually said at each of those calls. This is the
gap RunLoggerV2.log_model_call / model_instrumentation.InstrumentedModel
closes; see run_logger_v2.py's log_probe (drains + nests into Langfuse) and
model_instrumentation.py.
"""

from __future__ import annotations

import json

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs
from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
from evalrx.eval_agent.stages.probe_agent import ProbeAgent
from tests.conftest import FakeModel


def _load(run_dir, *parts: str) -> dict:
    from pathlib import Path

    return json.loads(Path(run_dir, *parts).read_text())


class _CountingModel(FakeModel):
    """Returns a distinct answer each call so N samples are distinguishable."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.n_calls = 0

    def generate(self, inputs, **kwargs) -> str:
        self.n_calls += 1
        return f"answer-{self.n_calls}"


def _batch(n: int = 2) -> CaseBatch:
    return CaseBatch([
        FailureCase(inputs=Inputs(prompt=f"question {i}"), expected="42")
        for i in range(n)
    ])


def test_self_consistency_calls_are_recorded_and_survive_the_analyzer(tmp_path):
    model = _CountingModel(capabilities={Capability.GENERATE})
    run_logger = RunLoggerV2(run_dir=tmp_path / "run1")
    run_logger.current_cycle = 0
    agent = ProbeAgent(run_logger=run_logger)
    batch = _batch()

    results = agent.probe(model, batch, analyzers=["self_consistency"])
    run_logger.log_probe(0, results)
    run_logger.close()

    # RunLoggerV2 inlines model_call records into M1/log.json's "model_calls"
    # array rather than a sibling model_calls.jsonl file.
    m1 = _load(tmp_path / "run1", "M1", "log.json")
    records = m1["model_calls"]
    assert records, "no model_call records were written"
    assert all(r["event"] == "model_call" for r in records)
    assert all(r["analyzer"] == "self_consistency" for r in records)
    assert all(r["cycle"] == 0 for r in records)
    assert {r["method"] for r in records} == {"generate"}
    # call_index is scoped per analyzer invocation and starts at 1, not reused.
    assert sorted(r["call_index"] for r in records) == list(range(1, len(records) + 1))
    # The generated text is what a full backstop must preserve — self_consistency
    # itself only keeps a derived "consistency" score, never the raw samples.
    assert all(r["output"].startswith("answer-") for r in records)
    assert model.n_calls == len(records)
    # The headline claim: you can see what the model was actually ASKED, not
    # just what it said. self_consistency reuses each case's own prompt
    # unmodified, so every call should resolve to its originating case.
    # self_consistency only ever reads cases[0] (a separate, pre-existing
    # analyzer quirk unrelated to this fix — noted, not addressed here).
    assert {r["inputs"]["prompt"] for r in records} == {"question 0"}
    assert {r["case_id"] for r in records} == {batch[0].id}

    # The probe entry references the count, not a sibling file (there is none).
    assert m1["probe"][0]["n_model_calls"] == len(records)


def test_perturbation_analyzer_calls_carry_the_rewritten_prompt(tmp_path):
    """format_sensitivity rewrites the prompt for each variant — the exact
    evidence class self_consistency's identical-prompt resampling can't
    exercise. A rewritten call won't exact-match the case's own prompt, so it
    should fall back to batch_case_ids (scoped) rather than case_id (exact)."""
    model = _CountingModel(capabilities={Capability.GENERATE})
    run_logger = RunLoggerV2(run_dir=tmp_path / "run3")
    run_logger.current_cycle = 0
    agent = ProbeAgent(run_logger=run_logger)
    batch = CaseBatch([FailureCase(
        inputs=Inputs(prompt="What is 2+2?\nA. 3\nB. 4\nC. 5"), expected="B",
    )])

    agent.probe(model, batch, analyzers=["format_sensitivity"])
    run_logger.close()

    records = _load(tmp_path / "run3", "M1", "log.json")["model_calls"]
    assert records
    assert all(r["analyzer"] == "format_sensitivity" for r in records)
    # Distinct rewritten prompts actually reached the log, not one repeated string.
    assert len({r["inputs"]["prompt"] for r in records}) > 1
    # batch_case_ids is the honest fallback for every call, matched or not.
    expected_ids = {c.id for c in batch}
    assert all(set(r["batch_case_ids"]) == expected_ids for r in records)


def test_wrapping_does_not_change_result_model_repr(tmp_path):
    """Experiment.fingerprint() and Result.model both key off repr(model);
    InstrumentedModel must forward it unchanged or every call fragments the
    experiment cache and mislabels its Result."""
    model = FakeModel(capabilities={Capability.GENERATE})
    run_logger = RunLoggerV2(run_dir=tmp_path / "run2")
    run_logger.current_cycle = 0
    agent = ProbeAgent(run_logger=run_logger)

    results = agent.probe(model, _batch(1), analyzers=["self_consistency"])
    run_logger.close()

    assert results["self_consistency"].model == repr(model)


def test_probe_still_works_when_run_logger_is_absent(tmp_path):
    """No run logger means no instrumentation, not a crash."""
    model = FakeModel(capabilities={Capability.GENERATE})
    agent = ProbeAgent()  # run_logger defaults to None

    results = agent.probe(model, _batch(1), analyzers=["self_consistency"])
    assert "self_consistency" in results


def test_log_probe_drains_calls_tagged_under_a_stale_cycle(tmp_path):
    """Regression pin for the bug the advisor's review caught live: a caller
    (VLDiagnoseLoop._m4_holdout_pass, before it was fixed) can stamp
    current_cycle AFTER probe() already ran, so every call this round made
    gets tagged under the PREVIOUS cycle's number — a different key than the
    one log_probe is about to be called with. log_probe must drain the whole
    buffer regardless, not `pop(cycle, [])` keyed to the number it was called
    with, or these calls silently never reach Langfuse and the buffer leaks
    forever (they stay durably in M1/log.json's model_calls either way — this
    is specifically about the buffer/Langfuse side)."""
    model = _CountingModel(capabilities={Capability.GENERATE})
    run_logger = RunLoggerV2(run_dir=tmp_path / "run4")
    agent = ProbeAgent(run_logger=run_logger)

    run_logger.current_cycle = 7  # stale: set as if a PRIOR cycle never advanced
    results = agent.probe(model, _batch(1), analyzers=["self_consistency"])
    # Called with a DIFFERENT cycle number than what current_cycle was during
    # probe() above — exactly the M4 holdout shape (cycle=-1) before the fix.
    run_logger.log_probe(-1, results)
    run_logger.close()

    probe_entry = _load(tmp_path / "run4", "M1", "log.json")["probe"][0]
    assert probe_entry["n_model_calls"] > 0
    assert run_logger._pending_model_calls == {}
