"""Held-out CONFIRM split (leak #3, Phase 1).

When confirm_split>0, M1-M5 hypothesis generation runs on the EXPLORE partition
and the post-loop fix/surgery validate on the frozen CONFIRM partition — so the
deployed fix is confirmed on data the loop never mined (selection independent of
confirmation, the one guarantee e-values cannot provide). confirm_split=0 is a
byte-for-byte no-op.
"""

from __future__ import annotations

from types import SimpleNamespace

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model
from evalvitals.eval_agent.hypothesis import Hypothesis
from evalvitals.eval_agent.loop import VLDiagnoseLoop
from evalvitals.eval_agent.loop_reports import VLDiagnoseReport
from evalvitals.eval_agent.stages.fix_agent import FixCandidate, FixOutcome, FixValidation
from evalvitals.eval_agent.stages.fix_tiers import FixTier
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
    max_tier = FixTier.L2_SCAFFOLD
    max_validation_cases = 5

    def __init__(self):
        self.seen_ids = None
        self.proposal_ids = None
        self.confirm_ids = None
        self.calls = 0
        self.tiers = []
        self.min_tiers = []
        self.confirm_cap = None

    def propose_and_validate(self, model, data, hypotheses, proposal_data=None):
        self.calls += 1
        self.tiers.append(self.max_tier)
        self.min_tiers.append(getattr(self, "min_tier", None))
        self.seen_ids = {id(c) for c in data}
        self.proposal_ids = (
            {id(c) for c in proposal_data} if proposal_data is not None else None
        )
        candidate = FixCandidate(
            tier=FixTier.L2_SCAFFOLD,
            name="stub",
            payload={"prompt_template": "{prompt}"},
        )
        validation = FixValidation(
            candidate=candidate,
            n_pairs=len(list(data)),
            n_fixed=1,
            n_broken=0,
            effect=0.1,
        )
        return FixOutcome(
            max_tier=self.max_tier,
            attempted=[validation],
            repair_rounds=1,
        )

    def validate_candidate(self, model, data, candidate):
        self.confirm_cap = self.max_validation_cases
        self.confirm_ids = {id(c) for c in data}
        return FixValidation(
            candidate=candidate,
            n_pairs=len(list(data)),
            n_fixed=1,
            n_broken=0,
            effect=0.1,
            e_value=1.0,
        )

    def _ebh_survivors(self, tested):
        return set()

    def _refine_signal(self, attempted, data):
        return None


def _report():
    h = Hypothesis(statement="x", target_model="m", predicted_failure_mode="")
    return VLDiagnoseReport(cycles=1, stopped_by="max_cycles", final_hypotheses=[h])


# The reports below carry proposals only (nothing M5-verified): since the
# 2026-08-21 merge of main, run_fix records a skipped stage for that unless the
# caller opts in with allow_unverified=True — the split mechanics are the
# subject here, so every call opts in.
def test_run_fix_validates_on_confirm_partition():
    batch = _batch()
    stub = _RecordingFixAgent()
    loop = _loop(fix_agent=stub, confirm_split=0.5)
    explore, confirm = loop._split_explore_confirm(batch)
    explore_ids = {id(c) for c in explore}
    confirm_ids = {id(c) for c in confirm}

    outcome = loop.run_fix(_report(), batch, allow_unverified=True)
    # Candidate iteration/selection sees ONLY explore; the one frozen candidate
    # is then scored ONLY on confirm.
    assert stub.seen_ids == explore_ids
    assert stub.proposal_ids is None
    assert stub.confirm_ids == confirm_ids
    assert stub.seen_ids.isdisjoint(stub.confirm_ids)
    assert len(stub.confirm_ids) == 12
    assert stub.confirm_cap == 0  # selection cap is disabled for final confirmation
    assert stub.max_validation_cases == 5  # and restored afterwards
    assert outcome.selected_on_explore == "stub"
    assert outcome.selection_attempted[0]["n_fixed"] == 1
    assert len(outcome.attempted) == 1  # only the CONFIRM validation is final


def test_run_fix_off_uses_full_batch():
    batch = _batch()
    stub = _RecordingFixAgent()
    loop = _loop(fix_agent=stub)  # confirm_split defaults to 0
    loop.run_fix(_report(), batch, allow_unverified=True)
    assert stub.seen_ids == {id(c) for c in batch}  # unchanged: full batch
    assert stub.proposal_ids is None
    assert stub.confirm_ids is None


def test_run_m4_adapts_on_explore_not_final_confirm():
    batch = _batch()
    seen = set()

    class _Surgery:
        def operate(self, hypothesis, model, results, data):
            seen.update(id(case) for case in data)
            return SimpleNamespace(status="refuted", evidence={})

    loop = _loop(confirm_split=0.5)
    loop.surgery_agent = _Surgery()
    explore, confirm = loop._split_explore_confirm(batch)
    loop.run_m4(_report(), batch, allow_unverified=True)

    assert seen == {id(case) for case in explore}
    assert seen.isdisjoint(id(case) for case in confirm)


def test_run_fix_escalates_on_explore_then_confirms_once():
    batch = _batch()
    stub = _RecordingFixAgent()
    loop = _loop(fix_agent=stub, confirm_split=0.5)

    loop.run_fix(
        _report(), batch, auto_escalate=True, max_tier="L3a", allow_unverified=True
    )

    # The full ladder is authored using EXPLORE only. CONFIRM is touched once,
    # after the strongest improving candidate has been frozen.
    assert stub.calls == 4  # L0 -> L1 -> L2 -> L3a (configured ceiling)
    assert stub.tiers == [
        FixTier.L0_RUNTIME_CONFIG,
        FixTier.L1_PROMPT,
        FixTier.L2_SCAFFOLD,
        FixTier.L3A_INTERNALS_READ,
    ]
    assert stub.min_tiers == stub.tiers
    assert stub.seen_ids is not None
    assert stub.confirm_ids is not None
    explore, confirm = loop._split_explore_confirm(batch)
    assert stub.seen_ids == {id(case) for case in explore}
    assert stub.confirm_ids == {id(case) for case in confirm}
    assert stub.seen_ids.isdisjoint(stub.confirm_ids)


def test_explore_selection_rejects_tiny_high_effect_candidate():
    """A 2/5 swing must not outrank a supported improvement on 64 pairs."""
    class SelectionAgent(_RecordingFixAgent):
        def propose_and_validate(self, model, data, hypotheses, proposal_data=None):
            tiny = FixValidation(
                candidate=FixCandidate(FixTier.L2_SCAFFOLD, "tiny", payload={}),
                n_pairs=5, n_fixed=2, n_broken=0, effect=0.4, e_value=2.0,
            )
            supported = FixValidation(
                candidate=FixCandidate(FixTier.L2_SCAFFOLD, "supported", payload={}),
                n_pairs=64, n_fixed=15, n_broken=6, effect=0.140625, e_value=1.76,
            )
            return FixOutcome(
                max_tier=self.max_tier,
                attempted=[tiny, supported],
                repair_rounds=1,
            )

        def validate_candidate(self, model, data, candidate):
            self.confirm_ids = {id(c) for c in data}
            return FixValidation(
                candidate=candidate, n_pairs=len(data), n_fixed=1, n_broken=0,
                effect=0.1, e_value=1.0,
            )

    outcome = _loop(fix_agent=SelectionAgent(), confirm_split=0.5).run_fix(
        _report(), _batch(128), allow_unverified=True
    )
    assert outcome.selected_on_explore == "supported"
    assert outcome.selection_attempted[0]["e_value"] == 2.0
    assert "coverage" in outcome.selection_attempted[0]
