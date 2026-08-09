"""Back-compat tests for the deprecated ``QwenLLM`` shim.

The concrete Qwen class is gone — identity lives in ``evalvitals.specs`` and
construction goes through ``compose``.  ``QwenLLM(...)`` is kept only as a
deprecated alias that builds an ``hf_local`` model.  (HF-local forward/capture
mechanics are covered by ``test_models/test_discover.py`` and the analyzer tests;
spec×backend composition by ``test_models/test_compose.py``.)
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch

from evalvitals.core import Capability
from evalvitals.core.case import Inputs
from evalvitals.core.model import Trace
from evalvitals.core.spec import ModelSpec, VisionSpec
from evalvitals.models.backends.base import RuntimeConfig
from evalvitals.models.backends.hf_local import HFLocalModel
from evalvitals.models.whitebox.qwen import QwenLLM


def test_qwenllm_warns_deprecation():
    with pytest.warns(DeprecationWarning, match="evalvitals.load"):
        QwenLLM()


def test_qwenllm_returns_hf_local_model_with_caps():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = QwenLLM()
    assert isinstance(model, HFLocalModel)
    assert model.spec.key == "qwen2.5-7b-instruct"
    # capabilities come from the hf_local backend + spec (no weights loaded)
    assert Capability.ATTENTION in model.capabilities
    assert Capability.HIDDEN_STATES in model.capabilities
    assert Capability.GENERATE in model.capabilities
    assert Capability.GRADIENTS not in model.capabilities


def test_qwenllm_checkpoint_override():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = QwenLLM(checkpoint="some/other-qwen")
    assert model.spec.hf_repo == "some/other-qwen"


def test_hf_vcd_processor_contrasts_clean_and_noisy_scores_each_step():
    spec = ModelSpec(
        key="fake-vlm", family="fake", model_type="fake_vlm", hf_repo="",
        auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
        vision=VisionSpec(image_token_id_attr="image_token_id"),
    )
    model = HFLocalModel(spec, RuntimeConfig())
    assert model.paper_method_fidelity("vcd") == "per_item_seeded_sampler_specialization"
    assert model.paper_method_fidelity("icd") == "adapted"

    from types import SimpleNamespace

    from evalvitals.models.paper_methods.vcd import VCDLogitsProcessor

    class NoisyPath:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                logits=torch.tensor([[[0.0, 0.0, 9.0]]]),
                past_key_values="cached-noisy-path",
            )

    noisy_path = NoisyPath()
    processor = VCDLogitsProcessor(noisy_path, {"pixel_values": "noisy"}, alpha=1, beta=0.1)
    clean_scores = torch.tensor([[0.0, 10.0, 9.0]])
    result = processor(torch.tensor([[1, 2]]), clean_scores)

    # Clean prefers Yes only moderately; VCD's noisy-image contrast makes
    # that visual evidence decisively stronger than the language prior.
    assert int(result.argmax(dim=-1).item()) == 1
    processor(torch.tensor([[1, 2, 3]]), clean_scores)
    assert noisy_path.calls[0]["pixel_values"] == "noisy"
    assert noisy_path.calls[1]["past_key_values"] == "cached-noisy-path"


def test_hf_instruction_cd_contrasts_disturbed_first_token(monkeypatch):
    pytest.importorskip("PIL")
    spec = ModelSpec(
        key="fake-vlm", family="fake", model_type="fake_vlm", hf_repo="",
        auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
        vision=VisionSpec(image_token_id_attr="image_token_id"),
    )
    model = HFLocalModel(spec, RuntimeConfig())

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1 if text.strip() == "Yes" else 2]}

    model._hf = (object(), Tokenizer())
    from PIL import Image

    image = Image.new("RGB", (8, 8), color="white")

    def fake_forward(inputs, capture, spec=None):
        # The disturbed instruction amplifies the language-prior No logit;
        # ICD removes that component and preserves the grounded Yes choice.
        logits = (
            torch.tensor([[0.0, 10.0, 9.0]])
            if not inputs.prompt.startswith("DISTURB")
            else torch.tensor([[0.0, 0.0, 12.0]])
        )
        return Trace(tokens=[], token_ids=[], provided={Capability.LOGITS}, logits=logits)

    monkeypatch.setattr(model, "forward", fake_forward)

    assert model.generate_instruction_cd(
        Inputs("Is the object present?", image), disturbance="DISTURB\n"
    ) == "Yes"


def test_hf_instructblip_icd_disturbs_only_qformer(monkeypatch):
    pytest.importorskip("PIL")
    spec = ModelSpec(
        key="fake-instructblip", family="instructblip", model_type="instructblip", hf_repo="",
        auto_class="InstructBlipForConditionalGeneration", processor_class="InstructBlipProcessor",
        vision=VisionSpec(),
    )
    model = HFLocalModel(spec, RuntimeConfig())
    assert model.paper_method_fidelity("icd") == "native_binary_specialization"

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1 if text.strip() == "Yes" else 2]}

    class QFormerTokenizer:
        def __call__(self, text, **kwargs):
            assert text == "DISTURB"
            class TokenBatch(dict):
                def to(self, device):
                    return self

            return TokenBatch(input_ids=torch.tensor([[99]]), attention_mask=torch.tensor([[1]]))

    class Processor:
        tokenizer = Tokenizer()
        qformer_tokenizer = QFormerTokenizer()

    class FakeInstructBlip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.qformer_inputs = []

        def forward(self, **kwargs):
            qformer_ids = kwargs["qformer_input_ids"]
            self.qformer_inputs.append(qformer_ids.clone())
            logits = torch.tensor([[[0.0, 10.0, 9.0]]])
            if int(qformer_ids[0, 0]) == 99:
                logits = torch.tensor([[[0.0, 0.0, 12.0]]])
            return SimpleNamespace(logits=logits)

    backend = FakeInstructBlip()
    model._hf = (backend, Processor())
    monkeypatch.setattr(
        model,
        "_encode_vlm",
        lambda inputs, _model, _processor: (
            {"pixel_values": torch.zeros((1, 3, 2, 2)), "qformer_input_ids": torch.tensor([[7]]),
             "qformer_attention_mask": torch.tensor([[1]]), "input_ids": torch.tensor([[4]])},
            [4], ["question"], None,
        ),
    )
    from PIL import Image

    assert model.generate_instruction_cd(Inputs("Is it present?", Image.new("RGB", (2, 2))), disturbance="DISTURB") == "Yes"
    assert [int(item[0, 0]) for item in backend.qformer_inputs] == [7, 99]


def test_hf_instructblip_generate_keeps_continuation_ids(monkeypatch):
    spec = ModelSpec(
        key="fake-instructblip", family="instructblip", model_type="instructblip", hf_repo="",
        auto_class="InstructBlipForConditionalGeneration", processor_class="InstructBlipProcessor",
        vision=VisionSpec(),
    )
    local = HFLocalModel(spec, RuntimeConfig())

    class Tokenizer:
        def decode(self, ids, **kwargs):
            assert list(ids) == [9]
            return "Yes"

    class Processor:
        tokenizer = Tokenizer()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 9]])

    local._hf = (Model(), Processor())
    monkeypatch.setattr(
        local,
        "_encode_vlm",
        lambda *args: ({"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.zeros((1, 3, 2, 2))}, [], [], None),
    )

    assert local.generate(Inputs("Is it present?", object())) == "Yes"
