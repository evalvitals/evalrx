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
from evalvitals.core.spec import AudioSpec, ModelSpec, VisionSpec
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


def test_hf_detector_grounded_presence_requires_allowlist_and_score():
    pytest.importorskip("PIL")
    spec = ModelSpec(
        key="fake-vlm", family="fake", model_type="fake_vlm", hf_repo="",
        auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
        vision=VisionSpec(image_token_id_attr="image_token_id"),
    )
    model = HFLocalModel(spec, RuntimeConfig())
    from PIL import Image

    image = Image.new("RGB", (8, 8), color="white")
    calls = []

    def detector(_image, query):
        calls.append(query)
        return [{"label": query, "score": 0.26, "box_px": [0, 0, 4, 4]}]

    kwargs = {
        "baseline_answer": "No",
        "objects": ["traffic light"],
        "detector_threshold": 0.25,
        "detector_engine": detector,
    }
    assert model.generate_detector_grounded_presence(
        Inputs("Is there a traffic light in the image?", image), **kwargs
    ) == "Yes"
    assert model.generate_detector_grounded_presence(
        Inputs("Is there a dog in the image?", image), **kwargs
    ) == "No"
    assert model.generate_detector_grounded_presence(
        Inputs("Is there a traffic light in the image?", image),
        **{**kwargs, "baseline_answer": "Yes"},
    ) == "Yes"
    assert calls == ["traffic light"]


def test_hf_clap_grounded_presence_uses_two_sided_gate():
    spec = ModelSpec(
        key="fake-audio", family="fake", model_type="fake_audio", hf_repo="",
        auto_class="AutoModelForAudioTextToText", processor_class="AutoProcessor",
        audio=AudioSpec(audio_token_id_attr="audio_token_id"),
    )
    model = HFLocalModel(spec, RuntimeConfig())
    waveform = torch.zeros(16000).numpy()
    prompt = "Can you detect the sound of a train in the audio?"

    assert model.generate_clap_grounded_presence(
        Inputs(prompt, audio=waveform), baseline_answer="No",
        similarity_engine=lambda _audio, query: 0.28,
    ) == "Yes"
    assert model.generate_clap_grounded_presence(
        Inputs(prompt, audio=waveform), baseline_answer="Yes",
        similarity_engine=lambda _audio, query: -0.06,
    ) == "No"
    assert model.generate_clap_grounded_presence(
        Inputs(prompt, audio=waveform), baseline_answer="No",
        similarity_engine=lambda _audio, query: 0.20,
    ) == "No"


def test_hf_audio_api_specialist_preserves_prompt_and_audio():
    spec = ModelSpec(
        key="fake-audio", family="fake", model_type="fake_audio", hf_repo="",
        auto_class="AutoModelForAudioTextToText", processor_class="AutoProcessor",
        audio=AudioSpec(audio_token_id_attr="audio_token_id"),
    )
    model = HFLocalModel(spec, RuntimeConfig())
    waveform = torch.zeros(16000).numpy()
    calls = []

    result = model.generate_audio_api_specialist(
        Inputs("Which option matches the audio?", audio=waveform),
        baseline_answer="B",
        specialist_engine=lambda inputs: calls.append(inputs) or "C",
    )

    assert result == "C" and len(calls) == 1
    assert calls[0].prompt == "Which option matches the audio?"
    assert torch.equal(torch.from_numpy(calls[0].audio), torch.from_numpy(waveform))

    accepted = model.generate_audio_api_specialist(
        Inputs("Which option matches the audio?", audio=waveform),
        baseline_answer="A",
        allowed_disagreements=["AC"],
        specialist_engine=lambda _inputs: "C",
    )
    rejected = model.generate_audio_api_specialist(
        Inputs("Which option matches the audio?", audio=waveform),
        baseline_answer="B",
        allowed_disagreements=["AC"],
        specialist_engine=lambda _inputs: "C",
    )
    assert accepted == "C" and rejected == "B"

    routed = model.generate_audio_api_specialist(
        Inputs("What words did the speaker say?", audio=waveform),
        baseline_answer="D",
        allowed_disagreements_by_route={"speech": ["DB"], "sound": ["AC"]},
        specialist_engine=lambda _inputs: "B",
    )
    music_texture = model.generate_audio_api_specialist(
        Inputs("How would you describe the texture of the sound?", audio=waveform),
        baseline_answer="C",
        allowed_disagreements_by_route={"sound": ["CD"]},
        specialist_engine=lambda _inputs: "D",
    )
    assert (routed, music_texture) == ("B", "C")


def test_hf_noncolor_spatial_specialist_preserves_color_baseline():
    spec = ModelSpec(
        key="fake-vlm", family="fake", model_type="fake_vlm", hf_repo="",
        auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
        vision=VisionSpec(),
    )
    model = HFLocalModel(spec, RuntimeConfig())
    calls = []
    engine = lambda inputs: calls.append(inputs) or "suv"

    shape = model.generate_noncolor_spatial_specialist(
        Inputs(prompt="What shape is left of the gray bus?", image="scene.png"),
        baseline_answer="rectangle",
        specialist_engine=engine,
    )
    color = model.generate_noncolor_spatial_specialist(
        Inputs(prompt="What color is the bus?", image="scene.png"),
        baseline_answer="red",
        specialist_engine=lambda _inputs: (_ for _ in ()).throw(
            AssertionError("must not run")
        ),
    )

    assert shape == "suv" and color == "red"
    assert len(calls) == 1 and calls[0].image == "scene.png"


def test_hf_chart_vision_specialist_preserves_prompt_and_image():
    spec = ModelSpec(
        key="fake-vlm", family="fake", model_type="fake_vlm", hf_repo="",
        auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
        vision=VisionSpec(),
    )
    model = HFLocalModel(spec, RuntimeConfig())
    calls = []

    result = model.generate_chart_vision_specialist(
        Inputs(prompt="What is the total?", image="chart.png"),
        baseline_answer="10",
        specialist_engine=lambda inputs: calls.append(inputs) or "42",
    )

    assert result == "42"
    assert len(calls) == 1
    assert calls[0].prompt == "What is the total?" and calls[0].image == "chart.png"


def test_hf_vision_api_specialist_preserves_prompt_and_image():
    spec = ModelSpec(
        key="fake-vlm", family="fake", model_type="fake_vlm", hf_repo="",
        auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
        vision=VisionSpec(),
    )
    model = HFLocalModel(spec, RuntimeConfig())
    calls = []

    result = model.generate_vision_api_specialist(
        Inputs(prompt="What is the average?", image="chart.png"),
        baseline_answer="10",
        specialist_engine=lambda inputs: calls.append(inputs) or "153:97",
    )

    assert float(result) == pytest.approx(153 / 97) and len(calls) == 1
    assert calls[0].prompt.startswith("What is the average?")
    assert "no explanation" in calls[0].prompt and calls[0].image == "chart.png"


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
