"""Tests for capability declaration, matching, and enforcement."""

from __future__ import annotations

import pytest

import evalrx.analyzers  # noqa: F401  (populate analyzer registry)
from evalrx.core import Capability, CapabilityError, registry
from tests.conftest import FakeModel


def test_model_supports_subset():
    model = FakeModel(capabilities={Capability.GENERATE, Capability.ATTENTION})
    assert model.supports({Capability.ATTENTION})
    assert model.supports({Capability.GENERATE, Capability.ATTENTION})
    assert not model.supports({Capability.GRADIENTS})


def test_registry_compatible_with_includes_attention():
    model = FakeModel(capabilities={Capability.ATTENTION})
    names = registry.analyzers.names_compatible_with(model)
    assert "attention" in names


def test_registry_compatible_with_excludes_gradient_analyzer():
    # Qwen-like local model: no GRADIENTS → gradcam excluded; no LOGPROBS+image → vl_shap excluded.
    model = FakeModel(
        capabilities={Capability.GENERATE, Capability.ATTENTION, Capability.HIDDEN_STATES}
    )
    names = registry.analyzers.names_compatible_with(model)
    assert "attention" in names
    assert "gradcam" not in names      # requires GRADIENTS
    assert "vl_shap" not in names      # requires LOGPROBS + image modality


def test_capability_error_on_missing():
    from evalrx.analyzers.attribution.gradcam import GradCAMAnalyzer

    model = FakeModel(capabilities={Capability.GENERATE})
    with pytest.raises(CapabilityError) as exc:
        GradCAMAnalyzer().run(model, "x")
    assert Capability.GRADIENTS in exc.value.missing


def test_capability_error_message_is_actionable():
    from evalrx.analyzers.geometry.linear_probe import LinearProbeAnalyzer

    model = FakeModel(capabilities={Capability.GENERATE})
    with pytest.raises(CapabilityError, match="hidden_states"):
        LinearProbeAnalyzer().run(model, "x")
