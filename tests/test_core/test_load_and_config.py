"""Unified model construction: evalrx.load(), load_model(), config routing.

All of these funnel through ``compose(spec, backend)``.  Construction is
weight-free (HFLocalModel computes capabilities in __init__; APIModel needs no
weights), so these run on CPU with no models downloaded.
"""

from __future__ import annotations

import pytest

from evalrx.config import ModelConfig, load_config
from evalrx.core import Capability, CapabilityError
from evalrx.models import load, load_model, resolve_spec_key
from evalrx.models.backends.api import APIModel
from evalrx.models.backends.hf_local import HFLocalModel

# -- evalrx.load() -------------------------------------------------

def test_load_spec_key_returns_hf_local():
    model = load("qwen2.5-7b-instruct")
    assert isinstance(model, HFLocalModel)
    assert {Capability.GENERATE, Capability.ATTENTION, Capability.HIDDEN_STATES} <= model.capabilities


def test_load_alias_resolves():
    assert resolve_spec_key("qwen") == "qwen2.5-7b-instruct"
    model = load("qwen")
    assert model.spec.key == "qwen2.5-7b-instruct"


def test_load_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        load("not-a-real-model")


def test_load_want_negotiation_ok():
    model = load("qwen2.5-7b-instruct", want={Capability.ATTENTION})
    assert isinstance(model, HFLocalModel)


def test_load_want_negotiation_rejected_on_api():
    # api backend cannot provide attention -> fail up front, before any load.
    with pytest.raises(CapabilityError):
        load("qwen2.5-7b-instruct", backend="api", want={Capability.ATTENTION})


def test_load_api_only_spec_forces_api_backend():
    # step-1o-vision is api_only; even the default hf_local request yields an APIModel.
    model = load("step-1o-vision")
    assert isinstance(model, APIModel)


def test_load_checkpoint_override():
    model = load("qwen2.5-7b-instruct", checkpoint="my/fork")
    assert model.spec.hf_repo == "my/fork"


# -- load_model(ModelConfig) ------------------------------------------

def test_load_model_from_config():
    model = load_model(ModelConfig(name="qwen"))
    assert isinstance(model, HFLocalModel)
    assert model.spec.key == "qwen2.5-7b-instruct"


def test_load_model_backend_and_want_from_config():
    cfg = ModelConfig(name="qwen2.5-7b-instruct", backend="hf_local", want=["attention"])
    model = load_model(cfg)
    assert isinstance(model, HFLocalModel)


def test_load_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        load_model(ModelConfig(name="nope"))


# -- config-driven run() routes through compose ------------------------

def test_config_short_form_parses(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("model: qwen2.5-7b-instruct\nanalysis: attention\n")
    cfg = load_config(cfg_file)
    assert cfg.model.name == "qwen2.5-7b-instruct"
    assert cfg.model.backend == "hf_local"
    assert cfg.analysis == "attention"


def test_config_full_form_parses(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "model:\n"
        "  name: qwen2.5-7b-instruct\n"
        "  backend: hf_local\n"
        "  device: cpu\n"
        "  dtype: float32\n"
        "  want: [attention]\n"
        "analysis: attention\n"
        "analysis_kwargs:\n"
        "  top_k: 5\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.model.backend == "hf_local"
    assert cfg.model.want == ["attention"]
    assert cfg.analysis_kwargs["top_k"] == 5
