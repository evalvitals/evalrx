"""Qwen3.5 specs: thinking is OFF on every chat-template render.

The two released checkpoints disagree on the template default when
``enable_thinking`` is absent (Qwen3.5-2B: off, Qwen3.5-9B: on), so the specs
send the kwarg explicitly; hf_local / vllm_offline forward
``spec.chat_template_kwargs`` into every ``apply_chat_template`` call.
"""

from __future__ import annotations

import pytest

from evalvitals.specs import get_spec

TEXT_KEYS = ("qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b")


@pytest.mark.parametrize("key", TEXT_KEYS)
def test_text_specs_disable_thinking_explicitly(key):
    spec = get_spec(key)
    assert spec.chat_template_kwargs == {"enable_thinking": False}
    assert spec.is_reasoning and not spec.is_vlm
    assert spec.auto_class == "AutoModelForCausalLM"
