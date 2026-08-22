"""hf_local post-load shims for remote-code families whose generate() assumptions
predate the installed transformers (NemotronH: own hybrid cache only when no
past_key_values is pre-built; turn-end <|im_end|> is not in generation_config)."""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from evalvitals.core.spec import ModelSpec
from evalvitals.models.backends.base import RuntimeConfig
from evalvitals.models.backends.hf_local import HFLocalModel


class _Tok:
    chat_template = None
    eos_token_id = 11


class _Gen(nn.Module):
    def __init__(self, eos=2):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.generation_config = SimpleNamespace(eos_token_id=eos)

    def _supports_default_dynamic_cache(self):
        return True


def _spec(family):
    return ModelSpec(key=f"fake-{family}", family=family, model_type="fake", hf_repo="fake/x",
                     auto_class="AutoModelForCausalLM", processor_class="AutoTokenizer")


def test_nemotron_h_gets_no_prebuilt_cache_and_stops_on_the_tokenizer_eos():
    model = _Gen(eos=2)
    HFLocalModel.from_loaded(model, _Tok(), spec=_spec("nemotron_h"), runtime=RuntimeConfig(device="cpu"))
    assert model._supports_default_dynamic_cache() is False
    assert model.generation_config.eos_token_id == [2, 11]


def test_omni_wrapper_shims_its_embedded_language_model_too():
    inner = _Gen(eos=[2])
    outer = _Gen(eos=None)
    outer.language_model = inner
    HFLocalModel.from_loaded(outer, _Tok(), spec=_spec("nemotron_h_omni"), runtime=RuntimeConfig(device="cpu"))
    assert inner._supports_default_dynamic_cache() is False and inner.generation_config.eos_token_id == [2, 11]
    assert outer.generation_config.eos_token_id == [11]


def test_other_families_are_untouched():
    model = _Gen(eos=2)
    HFLocalModel.from_loaded(model, _Tok(), spec=_spec("qwen3_5"), runtime=RuntimeConfig(device="cpu"))
    assert model._supports_default_dynamic_cache() is True and model.generation_config.eos_token_id == 2
