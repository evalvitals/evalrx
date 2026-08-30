"""hf_local post-load shims for remote-code families whose generate() assumptions
predate the installed transformers (NemotronH: own hybrid cache only when no
past_key_values is pre-built; turn-end <|im_end|> is not in generation_config)
and for omni checkpoints whose generate() silently runs an unwanted talker
(Qwen3-Omni-Instruct: has_talker=True from the checkpoint's own
enable_audio_output, so generate() defaults to synthesizing speech and
returning a (sequences, wav) tuple no caller in this file expects)."""

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


class _Omni(_Gen):
    """A generate() that records the kwargs it was actually called with."""

    def __init__(self, has_talker=True):
        super().__init__(eos=2)
        self.has_talker = has_talker
        self.calls: list = []

    def generate(self, *args, **kwargs):
        self.calls.append(kwargs)
        return "sequences"


def test_a_talker_model_gets_return_audio_false_by_default():
    model = _Omni(has_talker=True)
    HFLocalModel.from_loaded(model, _Tok(), spec=_spec("qwen3_omni_moe"), runtime=RuntimeConfig(device="cpu"))
    model.generate(input_ids="x")
    assert model.calls[-1]["return_audio"] is False


def test_an_explicit_return_audio_request_is_not_overridden():
    model = _Omni(has_talker=True)
    HFLocalModel.from_loaded(model, _Tok(), spec=_spec("qwen3_omni_moe"), runtime=RuntimeConfig(device="cpu"))
    model.generate(input_ids="x", return_audio=True)
    assert model.calls[-1]["return_audio"] is True


def test_a_model_without_has_talker_is_untouched():
    model = _Omni(has_talker=False)
    del model.has_talker  # e.g. a checkpoint whose config never set enable_audio_output
    HFLocalModel.from_loaded(model, _Tok(), spec=_spec("qwen3_omni_moe"), runtime=RuntimeConfig(device="cpu"))
    model.generate(input_ids="x")
    assert "return_audio" not in model.calls[-1]


def test_wrapping_is_idempotent_across_repeated_loads():
    # from_loaded / load can shim the same live handle more than once (a
    # retry, a second wrap() call); double-wrapping would call generate()
    # through two layers of the same kwarg-setting closure, which is
    # harmless here but is exactly the shape of bug that silently breaks
    # when a shim isn't idempotent, so pin it.
    model = _Omni(has_talker=True)
    spec = _spec("qwen3_omni_moe")
    HFLocalModel.from_loaded(model, _Tok(), spec=spec, runtime=RuntimeConfig(device="cpu"))
    once_wrapped = model.generate
    HFLocalModel.from_loaded(model, _Tok(), spec=spec, runtime=RuntimeConfig(device="cpu"))
    assert model.generate is once_wrapped
