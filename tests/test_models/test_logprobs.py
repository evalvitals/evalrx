"""LOGPROBS retrieval over the API backend + the black-box LogprobEntropy analyzer."""

from __future__ import annotations

import pytest

from evalrx.analyzers.uncertainty.logprob_entropy import LogprobEntropyAnalyzer
from evalrx.core.capability import Capability, CapabilityError
from evalrx.core.model import TokenLogprob
from evalrx.core.registry import registry
from evalrx.core.spec import ModelSpec
from evalrx.models import RuntimeConfig, compose
from evalrx.models.backends.api import parse_openai_logprobs
from tests.conftest import FakeModel

_SPEC = ModelSpec(key="gpt-x", family="openai", model_type="gpt-x")


def _fake_logprobs_fn(prompt, model, **kw):
    return [TokenLogprob(token="Paris", logprob=-0.05, top={"Paris": -0.05, "Lyon": -3.0})]


def test_logprobs_capability_gated_on_logprobs_fn():
    no_lp = compose(_SPEC, "api", RuntimeConfig(generate_fn=lambda *a, **k: "x"))
    assert Capability.LOGPROBS not in no_lp.capabilities
    with pytest.raises(CapabilityError):
        no_lp.logprobs("hello")

    with_lp = compose(_SPEC, "api", RuntimeConfig(logprobs_fn=_fake_logprobs_fn))
    assert Capability.LOGPROBS in with_lp.capabilities
    toks = with_lp.logprobs("the capital of france is")
    assert toks[0].token == "Paris" and toks[0].logprob == -0.05


def test_parse_openai_logprobs():
    raw = [{"token": "Pa", "logprob": -0.1, "top_logprobs": [{"token": "Pa", "logprob": -0.1},
                                                             {"token": "Lo", "logprob": -2.0}]}]
    toks = parse_openai_logprobs(raw)
    assert toks[0].token == "Pa" and toks[0].top["Lo"] == -2.0


def test_logprob_entropy_analyzer():
    model = FakeModel(capabilities={Capability.GENERATE, Capability.LOGPROBS})
    res = LogprobEntropyAnalyzer().run(model, "x")
    f = res.findings
    assert f["n_tokens"] == 4
    assert f["perplexity"] is not None and f["perplexity"] >= 1.0  # exp(-mean_logprob) >= 1 for negative lp
    assert f["mean_top_entropy"] is not None
    assert "token_logprobs" in res.artifacts and "token_logprobs" not in f


def test_logprob_entropy_refused_without_capability():
    model = FakeModel(capabilities={Capability.GENERATE})  # no LOGPROBS
    with pytest.raises(CapabilityError):
        LogprobEntropyAnalyzer().run(model, "x")


def test_logprob_entropy_registered_and_matched():
    lp_model = FakeModel(capabilities={Capability.GENERATE, Capability.LOGPROBS})
    gen_only = FakeModel(capabilities={Capability.GENERATE})
    assert "logprob_entropy" in registry.analyzers.names_compatible_with(lp_model)
    assert "logprob_entropy" not in registry.analyzers.names_compatible_with(gen_only)
