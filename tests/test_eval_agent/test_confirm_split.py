"""Held-out CONFIRM split (leak #3, Phase 1).

When confirm_split>0, M1-M5 hypothesis generation runs on the EXPLORE partition
and the post-loop fix/surgery validate on the frozen CONFIRM partition — so the
deployed fix is confirmed on data the loop never mined (selection independent of
confirmation, the one guarantee e-values cannot provide). confirm_split=0 is a
byte-for-byte no-op.
"""

from __future__ import annotations

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model
from evalvitals.eval_agent.hypothesis import Hypothesis
from evalvitals.eval_agent.loop import VLDiagnoseLoop
from evalvitals.eval_agent.loop_reports import VLDiagnoseReport
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol


class _M(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def generate(self, inputs, **kwargs):
        return ""

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _batch(n=24):
    # mixed labels + probe_type so stratification has structure to preserve
    cases = []
    for i in range(n):
        cases.append(FailureCase(
            id=f"c{i}", inputs=Inputs(prompt="q"),
            label=Label.FAIL if i % 3 == 0 else Label.PASS,
            metadata={"probe_type": "adversarial" if i % 2 else "present"}))
    return CaseBatch(cases)


def _loop(**kw):
    return VLDiagnoseLoop(model=_M(), protocol=ExperimentProtocol(description="d"), **kw)


def test_split_off_is_noop():
    batch = _batch()
    explore, confirm = _loop()._split_explore_confirm(batch)
    assert confirm is None
    assert explore is batch  # same object — zero change to existing runs


def test_split_disjoint_stratified_deterministic():
    batch = _batch()
    loop = _loop(confirm_split=0.5)
    explore, confirm = loop._split_explore_confirm(batch)
    ex = {id(c) for c in explore}
    co = {id(c) for c in confirm}
    assert ex.isdisjoint(co)                       # disjoint
    assert len(ex | co) == len(list(batch))        # complete cover
    assert len(list(confirm)) == 12                # 50%
    # both labels survive in BOTH partitions (stratified, not a lucky draw)
    for part in (explore, confirm):
        labels = {c.label for c in part}
        assert Label.FAIL in labels and Label.PASS in labels
    # deterministic: a re-split (what run_fix does) reproduces the partition
    _, confirm2 = loop._split_explore_confirm(batch)
    assert {id(c) for c in confirm2} == co


class _RecordingFixAgent:
    """Captures which cases reach the fix module."""

    run_logger = None

    def __init__(self):
        self.seen_ids = None

    def propose_and_validate(self, model, data, hypotheses):
        self.seen_ids = {id(c) for c in data}
        return object()


def _report():
    h = Hypothesis(statement="x", target_model="m", predicted_failure_mode="")
    return VLDiagnoseReport(cycles=1, stopped_by="max_cycles", final_hypotheses=[h])


def test_run_fix_validates_on_confirm_partition():
    batch = _batch()
    stub = _RecordingFixAgent()
    loop = _loop(fix_agent=stub, confirm_split=0.5)
    _, confirm = loop._split_explore_confirm(batch)
    confirm_ids = {id(c) for c in confirm}

    loop.run_fix(_report(), batch)
    # the fix saw ONLY the held-out confirm cases (disjoint from explore)
    assert stub.seen_ids == confirm_ids
    assert len(stub.seen_ids) == 12


def test_run_fix_off_uses_full_batch():
    batch = _batch()
    stub = _RecordingFixAgent()
    loop = _loop(fix_agent=stub)  # confirm_split defaults to 0
    loop.run_fix(_report(), batch)
    assert stub.seen_ids == {id(c) for c in batch}  # unchanged: full batch
