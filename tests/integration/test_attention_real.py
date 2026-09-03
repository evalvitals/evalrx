"""Integration tests: AttentionAnalyzer on a real Qwen2.5-7B model.

Verifies end-to-end that the analyzer produces correctly shaped results,
populated findings, and serialisable output — using real weights, not FakeModel.
"""

from __future__ import annotations

import json

import pytest

from evalrx.analyzers.attention.summary import AttentionAnalyzer, AttentionResult

pytestmark = pytest.mark.gpu

QWEN_LAYERS = 28
QWEN_HEADS = 28


def test_result_type(qwen_model):
    result = AttentionAnalyzer().run(qwen_model, "The Eiffel Tower is in")
    assert isinstance(result, AttentionResult)


def test_dimensions_match_model(qwen_model):
    result = AttentionAnalyzer().run(qwen_model, "hello world")
    assert result.num_layers == QWEN_LAYERS
    assert result.num_heads == QWEN_HEADS


def test_findings_top_k_respected(qwen_model):
    result = AttentionAnalyzer(top_k=5).run(qwen_model, "The capital of France is")
    assert len(result.findings["top_attended_tokens"]) == 5


def test_findings_weights_sum_to_one(qwen_model):
    result = AttentionAnalyzer(top_k=10).run(qwen_model, "hello")
    total = sum(t["weight"] for t in result.findings["top_attended_tokens"])
    assert abs(total - 1.0) < 0.01


def test_findings_json_serialisable(qwen_model):
    result = AttentionAnalyzer(top_k=3).run(qwen_model, "hello")
    json.dumps(result.to_dict())  # must not raise


def test_entropy_is_non_negative(qwen_model):
    result = AttentionAnalyzer().run(qwen_model, "hello world")
    assert result.findings["mean_attention_entropy"] >= 0.0
