"""Experiment caching is now CONTENT-addressed (not id()-addressed)."""

from __future__ import annotations

from evalrx.analyzers.attention.summary import AttentionAnalyzer
from evalrx.core import Experiment, ExperimentRunner
from tests.conftest import FakeModel


def test_equivalent_experiments_share_fingerprint():
    model = FakeModel()
    e1 = Experiment(model=model, analyzer=AttentionAnalyzer(layer=-1), data="same prompt")
    e2 = Experiment(model=model, analyzer=AttentionAnalyzer(layer=-1), data="same prompt")
    assert e1.fingerprint() == e2.fingerprint()


def test_different_params_differ():
    model = FakeModel()
    e1 = Experiment(model=model, analyzer=AttentionAnalyzer(layer=-1), data="p")
    e2 = Experiment(model=model, analyzer=AttentionAnalyzer(layer=0), data="p")
    assert e1.fingerprint() != e2.fingerprint()


def test_different_data_differ():
    model = FakeModel()
    e1 = Experiment(model=model, analyzer=AttentionAnalyzer(), data="a")
    e2 = Experiment(model=model, analyzer=AttentionAnalyzer(), data="b")
    assert e1.fingerprint() != e2.fingerprint()


def test_runner_dedupes_distinct_but_equivalent_objects():
    model = FakeModel()
    runner = ExperimentRunner()
    r1 = runner.run(Experiment(model=model, analyzer=AttentionAnalyzer(), data="x"))
    r2 = runner.run(Experiment(model=model, analyzer=AttentionAnalyzer(), data="x"))
    assert r1 is r2  # content-addressed cache hit across two distinct Experiment objects
