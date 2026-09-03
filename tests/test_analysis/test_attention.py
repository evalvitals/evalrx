"""Tests for AttentionAnalyzer and AttentionResult (core.Analyzer-based)."""

from __future__ import annotations

import torch

from evalrx.analyzers.attention.summary import AttentionAnalyzer, AttentionResult
from evalrx.core import Capability, CaseBatch
from tests.conftest import FakeModel

# ------------------------------------------------------------------
# Running the analyzer
# ------------------------------------------------------------------

def test_run_returns_attention_result():
    model = FakeModel(n_layers=2, n_heads=4, seq_len=5)
    result = AttentionAnalyzer().run(model, "a prompt")
    assert isinstance(result, AttentionResult)


def test_run_accepts_casebatch():
    model = FakeModel()
    batch = CaseBatch.from_prompts(["a prompt"])
    result = AttentionAnalyzer().run(model, batch)
    assert isinstance(result, AttentionResult)


def test_dimensions_from_trace():
    model = FakeModel(n_layers=3, n_heads=8, seq_len=6)
    result = AttentionAnalyzer().run(model, "x")
    assert result.num_layers == 3
    assert result.num_heads == 8
    assert result.seq_len == 6


# ------------------------------------------------------------------
# findings (agent-readable)
# ------------------------------------------------------------------

def test_findings_shape():
    model = FakeModel()
    result = AttentionAnalyzer(top_k=3).run(model, "x")
    f = result.findings
    assert f["num_layers"] == 3
    assert len(f["top_attended_tokens"]) == 3
    assert "mean_attention_entropy" in f


def test_findings_json_serialisable():
    import json

    model = FakeModel()
    result = AttentionAnalyzer().run(model, "x")
    json.dumps(result.to_dict())  # must not raise


# ------------------------------------------------------------------
# AttentionResult convenience methods (backed by artifacts)
# ------------------------------------------------------------------

def _result(n_layers=2, n_heads=4, seq_len=5) -> AttentionResult:
    model = FakeModel(n_layers=n_layers, n_heads=n_heads, seq_len=seq_len)
    return AttentionAnalyzer().run(model, "x")


def test_aggregate_mean_shape():
    r = _result()
    assert r.aggregate(layer=0, head="mean").shape == torch.Size([5, 5])


def test_aggregate_single_head_shape():
    r = _result(n_heads=4)
    assert r.aggregate(layer=0, head=2).shape == torch.Size([5, 5])


def test_aggregate_negative_layer():
    r = _result(n_layers=3)
    assert torch.allclose(r.aggregate(layer=-1), r.aggregate(layer=2))


def test_layer_head_matrix_shape():
    r = _result(n_layers=2, n_heads=4, seq_len=5)
    assert r.layer_head_matrix().shape == torch.Size([2, 4, 5, 5])


def test_to_numpy_dtype():
    r = _result()
    arr = r.to_numpy()
    assert arr.dtype.name == "float32"
    assert arr.shape == (5, 5)


def test_top_attended_tokens_sorted_and_typed():
    r = _result()
    top = r.top_attended_tokens(query_pos=0, k=5)
    vals = [v for _, v in top]
    assert vals == sorted(vals, reverse=True)
    for tok, val in top:
        assert isinstance(tok, str)
        assert isinstance(val, float)


def test_top_attended_tokens_k_clamp():
    r = _result(seq_len=3)
    assert len(r.top_attended_tokens(k=100)) == 3


def test_attention_entropy_shape():
    r = _result(seq_len=5)
    ent = r.attention_entropy()
    assert ent.shape == torch.Size([5])
    assert (ent >= 0).all()


# ------------------------------------------------------------------
# Analyzer metadata
# ------------------------------------------------------------------

def test_analyzer_declares_attention_requirement():
    assert AttentionAnalyzer.requires == frozenset({Capability.ATTENTION})
    assert AttentionAnalyzer.name == "attention"
