"""TokenEntropyAnalyzer — a free LOGITS-only analyzer (no attention/eager)."""

from __future__ import annotations

import pytest

from evalrx.analyzers.uncertainty.entropy import TokenEntropyAnalyzer, UncertaintyResult
from evalrx.core.capability import Capability, CapabilityError
from tests.conftest import FakeModel


def test_runs_on_logits_capable_model():
    model = FakeModel(capabilities={Capability.GENERATE, Capability.LOGITS})
    result = TokenEntropyAnalyzer(top_k=3).run(model, "the capital of france is")
    assert isinstance(result, UncertaintyResult)
    f = result.findings
    assert f["seq_len"] == 5
    assert "mean_entropy" in f and "final_token_entropy" in f
    assert len(f["top_next_tokens"]) == 3
    # heavy tensor stays in artifacts, not findings
    assert "logits" in result.artifacts
    assert "logits" not in f


def test_refused_without_logits_capability():
    model = FakeModel(capabilities={Capability.GENERATE})  # no LOGITS
    with pytest.raises(CapabilityError):
        TokenEntropyAnalyzer().run(model, "x")


def test_registered_and_capability_matched():
    from evalrx.core.registry import registry

    assert registry.analyzers.has("token_entropy")
    logits_model = FakeModel(capabilities={Capability.GENERATE, Capability.LOGITS})
    assert "token_entropy" in registry.analyzers.names_compatible_with(logits_model)
    gen_only = FakeModel(capabilities={Capability.GENERATE})
    assert "token_entropy" not in registry.analyzers.names_compatible_with(gen_only)
