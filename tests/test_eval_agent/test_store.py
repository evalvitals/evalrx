"""The in-memory store the diagnosis loops record cases and hypotheses into."""

from __future__ import annotations

from evalrx.core.case import FailureCase, Label
from evalrx.eval_agent import Hypothesis, HypothesisStatus, InMemoryStore


def test_store_query_by_kind_and_status():
    store = InMemoryStore()
    store.add_case(FailureCase.from_prompt("x", label=Label.FAIL, tags={"hallucination"}))
    store.add_hypothesis(Hypothesis(statement="s", target_model="m", predicted_failure_mode="f",
                                    status=HypothesisStatus.SUPPORTED))
    assert len(store.query(kind="cases", label=Label.FAIL)) == 1
    assert len(store.query(kind="cases", tags={"hallucination"})) == 1
    assert len(store.query(kind="cases", label=Label.PASS)) == 0
    assert len(store.query(kind="hypotheses", status=HypothesisStatus.SUPPORTED)) == 1
