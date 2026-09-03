"""ModelSpec x Backend composition + capability negotiation."""

from __future__ import annotations

import pytest

from evalrx.core.capability import Capability, CapabilityError
from evalrx.models import RuntimeConfig, compose
from evalrx.models.backends import BACKENDS
from evalrx.models.backends.api import APIModel, call_vision_api_generate_fn


def test_backends_registered():
    assert set(BACKENDS) == {"api", "hf_local", "vllm_offline"}


def test_capability_from_backend_not_spec():
    # hf_local can provide attention; api cannot — same spec, different caps.
    assert Capability.ATTENTION in BACKENDS["hf_local"].capabilities
    assert Capability.ATTENTION not in BACKENDS["api"].capabilities


def test_negotiation_raises_before_load():
    # Requesting ATTENTION against the api backend fails up front (no weights load).
    with pytest.raises(CapabilityError) as ei:
        compose("qwen3-vl-8b-instruct", "api", want={Capability.ATTENTION})
    assert Capability.ATTENTION in ei.value.missing


def test_api_only_spec_refuses_local_backend():
    with pytest.raises(ValueError):
        compose("step-1o-vision", "hf_local")


def test_api_model_generate_via_injected_fn():
    rt = RuntimeConfig(generate_fn=lambda prompt, model, **k: f"[{model}] {prompt}")
    m = compose("qwen3-8b", "api", rt)
    assert isinstance(m, APIModel)
    assert m.generate("hi") == "[Qwen/Qwen3-8B] hi"
    assert m.capabilities == frozenset({Capability.GENERATE, Capability.TOOL_CALLS})


def test_logprobs_capability_requires_logprobs_fn():
    # client_kwargs alone does NOT grant LOGPROBS — we only claim it if we can
    # actually retrieve them (a logprobs_fn is wired).
    m1 = compose("qwen3-8b", "api",
                 RuntimeConfig(generate_fn=lambda *a, **k: "x", client_kwargs={"logprobs": True}))
    assert Capability.LOGPROBS not in m1.capabilities
    m2 = compose("qwen3-8b", "api", RuntimeConfig(logprobs_fn=lambda *a, **k: []))
    assert Capability.LOGPROBS in m2.capabilities


def test_api_model_forward_refuses_internals():
    m = compose("qwen3-8b", "api", RuntimeConfig(generate_fn=lambda *a, **k: "x"))
    with pytest.raises(CapabilityError):
        m.forward("hi", capture={Capability.ATTENTION})


def test_call_vision_api_adapter_builds_messages():
    seen = {}

    def fake_engine(model, messages, sampling):
        seen["model"], seen["messages"] = model, messages
        return "engine-reply"

    fn = call_vision_api_generate_fn(fake_engine)
    out = fn("hello", model="gpt-x")
    assert out == "engine-reply"
    assert seen["messages"] == [{"role": "user", "content": "hello"}]


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        compose("qwen3-8b", "nope")


def test_local_tool_calls_is_conditional_on_spec():
    # qwen3-8b declares tool_calling=True -> local handle gains TOOL_CALLS;
    # gemma-3-1b-it does not -> handle lacks it. (Build is lazy: no weights loaded.)
    tool_model = compose("qwen3-8b", "hf_local")
    assert Capability.TOOL_CALLS in tool_model.capabilities
    plain_model = compose("gemma-3-1b-it", "hf_local")
    assert Capability.TOOL_CALLS not in plain_model.capabilities


def test_local_negotiation_uses_actual_handle_caps():
    # Requesting TOOL_CALLS from a non-tool local model fails (precise per-model negotiation).
    with pytest.raises(CapabilityError):
        compose("gemma-3-1b-it", "hf_local", want={Capability.TOOL_CALLS})


def test_omni_modalities_propagate_to_handle():
    # An omni spec carries audio/video; the handle's modality set reflects it
    # regardless of backend (capabilities still come from the backend).
    m = compose("qwen3-omni-30b-a3b-instruct", "api",
                RuntimeConfig(generate_fn=lambda *a, **k: "x"))
    assert m.modalities == frozenset({"text", "image", "audio", "video"})
    assert m.capabilities == frozenset({Capability.GENERATE, Capability.TOOL_CALLS})
