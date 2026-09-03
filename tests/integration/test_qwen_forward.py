"""Integration tests: real Qwen2.5-7B forward pass.

Validates that the HFLocalModel forward() captures attention and hidden states
with the correct shapes for Qwen2.5-7B-Instruct (28 layers, 28 heads).
"""

from __future__ import annotations

import pytest
import torch

from evalrx.core.capability import Capability

pytestmark = pytest.mark.gpu

QWEN_LAYERS = 28
QWEN_HEADS = 28


def test_forward_provides_attention(qwen_model):
    trace = qwen_model.forward("The Eiffel Tower is in", capture={Capability.ATTENTION})
    assert Capability.ATTENTION in trace.provided
    assert trace.attentions is not None
    assert len(trace.attentions) == QWEN_LAYERS


def test_attention_tensor_shape(qwen_model):
    trace = qwen_model.forward("hello world", capture={Capability.ATTENTION})
    for layer_attn in trace.attentions:
        assert layer_attn.ndim == 3                              # (heads, seq, seq)
        assert layer_attn.shape[0] == QWEN_HEADS
        assert layer_attn.shape[1] == layer_attn.shape[2]       # square


def test_attention_weights_sum_to_one(qwen_model):
    trace = qwen_model.forward("hello", capture={Capability.ATTENTION})
    last = trace.attentions[-1]  # (heads, seq, seq)
    row_sums = last.sum(dim=-1)  # (heads, seq)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_forward_provides_hidden_states(qwen_model):
    trace = qwen_model.forward("hello", capture={Capability.HIDDEN_STATES})
    assert Capability.HIDDEN_STATES in trace.provided
    assert trace.hidden_states is not None
    assert len(trace.hidden_states) == QWEN_LAYERS + 1          # embedding + each layer


def test_tokens_match_input_length(qwen_model):
    prompt = "The capital of France is"
    trace = qwen_model.forward(prompt, capture={Capability.ATTENTION})
    assert len(trace.tokens) == len(trace.token_ids)
    assert len(trace.tokens) > 0
