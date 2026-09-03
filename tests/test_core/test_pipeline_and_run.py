"""Tests for Pipeline composition and the config-driven run() entrypoint."""

from __future__ import annotations

from evalrx.analyzers.attention.summary import AttentionAnalyzer, AttentionResult
from evalrx.core import Experiment, ExperimentRunner, Pipeline
from tests.conftest import FakeModel


def test_pipeline_runs_named_steps():
    model = FakeModel()
    pipe = Pipeline([("attention", AttentionAnalyzer(top_k=3))])
    results = pipe.run(model, "the capital of france is")
    assert set(results) == {"attention"}
    assert isinstance(results["attention"], AttentionResult)


def test_experiment_runner_caches():
    model = FakeModel()
    exp = Experiment(model=model, analyzer=AttentionAnalyzer(), data="x")
    runner = ExperimentRunner()
    r1 = runner.run(exp)
    r2 = runner.run(exp)
    assert r1 is r2  # cached by experiment identity


def test_run_entrypoint_with_config(tmp_path, monkeypatch):
    import evalrx

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("model: qwen\nanalysis: attention\n")
    config = evalrx.load_config(cfg_file)

    # Swap the real Qwen for a FakeModel so no weights are loaded.
    monkeypatch.setattr(evalrx, "load_model", lambda cfg: FakeModel())
    result = evalrx.run(config, "the capital of france is")
    assert isinstance(result, AttentionResult)


def test_run_strips_legacy_call_prefix(tmp_path, monkeypatch):
    import evalrx

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("model: qwen\nanalysis: call_attention\n")  # old style
    config = evalrx.load_config(cfg_file)
    monkeypatch.setattr(evalrx, "load_model", lambda cfg: FakeModel())
    result = evalrx.run(config, "x")
    assert isinstance(result, AttentionResult)
