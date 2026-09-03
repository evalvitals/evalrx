"""Closed-loop orchestration: data split, pre-registration, A/B, report, loop."""

from __future__ import annotations

from evalrx.core.case import FailureCase, Label
from evalrx.eval_agent import (
    ABRunner,
    DataSplit,
    DiagnosticReport,
    EvalOrchestrator,
    Hypothesis,
    HypothesisStatus,
    InMemoryStore,
    ManualHypothesisGenerator,
    PreregisteredHypothesis,
    PreregistrationLog,
    SelfEvolveLoop,
    Split,
)


def _cases(n):
    return [type("C", (), {"id": f"c{i}"})() for i in range(n)]


# ---------------- data split ----------------
def test_datasplit_is_deterministic_and_partitions_all():
    split = DataSplit()
    cs = _cases(120)
    p = split.partition(cs)
    assert sum(len(v) for v in p.values()) == 120
    assert all(len(p[s]) > 0 for s in (Split.EXPLORE, Split.VALIDATE, Split.CONFIRM))
    # same id -> same split, every time
    assert split.assign("c1") == split.assign("c1")


# ---------------- pre-registration ----------------
def test_prereg_hash_stable_and_sensitive():
    h1 = PreregisteredHypothesis(predicate="small objects", min_effect=0.05)
    h2 = PreregisteredHypothesis(predicate="small objects", min_effect=0.05)
    h3 = PreregisteredHypothesis(predicate="small objects", min_effect=0.10)
    assert h1.contract_hash() == h2.contract_hash()
    assert h1.contract_hash() != h3.contract_hash()


def test_prereg_log_records_before_test():
    log = PreregistrationLog()
    h = PreregisteredHypothesis(predicate="p")
    assert not log.is_registered(h)
    log.register(h, timestamp="2026-01-01T00:00:00")
    assert log.is_registered(h) and len(log) == 1


# ---------------- A/B runner ----------------
def test_ab_runner_strong_effect():
    ab = ABRunner(lambda c: False, lambda c: True).run(_cases(30), alpha=0.05)
    assert ab.stat.effect == 1.0 and ab.stat.reject is True
    assert ab.n == 30


# ---------------- orchestrator (pre-registered A/B) ----------------
def test_orchestrator_preregisters_then_tests_and_reports():
    orch = EvalOrchestrator()
    hyp = PreregisteredHypothesis(predicate="prompt B helps", statement="B beats A", split="validate")
    report = orch.run(_cases(120), hyp, strategy_a=lambda c: False, strategy_b=lambda c: True,
                      timestamp="2026-01-01T00:00:00")
    assert report["preregistered"] is True and report["prereg_hash"]
    assert report["decision"] == "REJECT H0"
    assert report["split"] == "validate"
    assert orch.prereg_log.is_registered(hyp)
    md = DiagnosticReport.to_markdown(report)
    assert "Pre-registered" in md and report["prereg_hash"] in md


# ---------------- store ----------------
def test_store_query_by_kind_and_status():
    store = InMemoryStore()
    store.add_case(FailureCase.from_prompt("x", label=Label.FAIL, tags={"hallucination"}))
    store.add_hypothesis(Hypothesis(statement="s", target_model="m", predicted_failure_mode="f",
                                    status=HypothesisStatus.SUPPORTED))
    assert len(store.query(kind="cases", label=Label.FAIL)) == 1
    assert len(store.query(kind="cases", tags={"hallucination"})) == 1
    assert len(store.query(kind="cases", label=Label.PASS)) == 0
    assert len(store.query(kind="hypotheses", status=HypothesisStatus.SUPPORTED)) == 1


# ---------------- self-evolve loop ----------------
def test_loop_records_proposals_and_converges():
    gen = ManualHypothesisGenerator(hypotheses=[
        Hypothesis(statement="h1", target_model="m", predicted_failure_mode="a"),
        Hypothesis(statement="h2", target_model="m", predicted_failure_mode="b"),
    ])
    loop = SelfEvolveLoop(generator=gen)
    history = loop.run(max_cycles=5)
    assert len(history[0]) == 2          # first cycle proposes 2
    assert history[-1] == []             # converged (queue drained)
    assert len(loop.store.hypotheses) == 2
