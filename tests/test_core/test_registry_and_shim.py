"""Tests for the registry, analyzer introspection, Result, and the call_x shim."""

from __future__ import annotations

import pytest

import evalrx.analyzers  # noqa: F401  (populate registries)
import evalrx.models  # noqa: F401
from evalrx.core import Result, registry
from tests.conftest import FakeModel

# -- registry ----------------------------------------------------------

def test_models_live_in_specs():
    # specs.REGISTRY is the single source of truth for model identity now.
    from evalrx.specs import list_specs

    assert "qwen2.5-7b-instruct" in list_specs()


def test_registry_models_is_deprecated_shim():
    # registry.models still works but warns and delegates to specs.
    with pytest.warns(DeprecationWarning):
        names = registry.models.list()
    assert "qwen2.5-7b-instruct" in names


def test_analyzers_registered():
    names = registry.analyzers.list()
    assert "attention" in names
    assert "rise" in names          # perturbation
    assert "loop_detect" in names   # agent heuristic


def test_registry_get_unknown_raises():
    with pytest.raises(KeyError):
        registry.analyzers.get("does_not_exist")


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        registry.analyzers.register("attention")(object)


# -- sklearn-style introspection --------------------------------------

def test_get_set_params():
    from evalrx.analyzers.attention.summary import AttentionAnalyzer

    a = AttentionAnalyzer(layer=-1, top_k=5)
    assert a.get_params() == {"layer": -1, "head": "mean", "top_k": 5}
    a.set_params(top_k=2)
    assert a.get_params()["top_k"] == 2
    assert a.top_k == 2


# -- call_x shim -------------------------------------------------------

def test_call_attention_shim_dispatches():
    from evalrx.analyzers.attention.summary import AttentionResult

    model = FakeModel()
    result = model.call_attention("the capital of france is")
    assert isinstance(result, AttentionResult)


def test_call_x_passes_kwargs():
    model = FakeModel()
    result = model.call_attention("x", top_k=2)
    assert len(result.findings["top_attended_tokens"]) == 2


def test_unknown_call_attr_raises():
    model = FakeModel()
    with pytest.raises(AttributeError):
        model.call_nonexistent_analysis  # noqa: B018


def test_non_call_missing_attr_raises():
    model = FakeModel()
    with pytest.raises(AttributeError):
        model.totally_unknown_attr  # noqa: B018


# -- Result ------------------------------------------------------------

def test_result_to_dict_drops_artifacts():
    r = Result(
        analyzer="x",
        model="m",
        findings={"k": 1},
        artifacts={"heavy": [1, 2, 3]},
    )
    d = r.to_dict()
    assert "artifacts" not in d
    assert d["findings"] == {"k": 1}


def test_result_summary_contains_findings():
    r = Result(analyzer="attention", model="m", findings={"seq_len": 5})
    s = r.summary()
    assert "attention" in s
    assert "seq_len" in s
