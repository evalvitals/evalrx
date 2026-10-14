"""Tiny offline VLM/ViT checks of real image-side probes through EvalRX."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evalrx.analyzers.vlmprobe import VLMProbeAnalyzer


@pytest.fixture
def tiny_vlm():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    package = pytest.importorskip("vlmprobe")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            vision = transformers.CLIPVisionConfig(
                hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                num_attention_heads=2, image_size=8, patch_size=4,
            )
            language = transformers.LlamaConfig(
                hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=2, vocab_size=32,
                max_position_embeddings=32, attention_dropout=0,
            )
            config = transformers.LlavaConfig(
                vision_config=vision.to_dict(), text_config=language.to_dict(),
                image_token_index=31, image_seq_length=4, vision_feature_layer=-1,
            )
            config._attn_implementation = "eager"
            raw = transformers.LlavaForConditionalGeneration(config).eval()
            inputs = {
                "input_ids": torch.tensor([[1, 31, 31, 31, 31, 2, 3]]),
                "attention_mask": torch.ones(1, 7, dtype=torch.long),
                "pixel_values": torch.randn(1, 3, 8, 8),
            }
        try:
            probe = package.Prober(raw)
        except package.CapabilityError as exc:
            pytest.skip(f"installed Transformers is outside the native VLM adapter contract: {exc}")
        # Transformers 5.3 installs its native vision-output capture hooks on
        # the first forward, independent of probing. Establish that baseline.
        with torch.no_grad():
            raw(**inputs, use_cache=False)
        yield SimpleNamespace(torch=torch, package=package, probe=probe, inputs=inputs, raw=raw,
                              metric=package.TokenMargin(5, 6))
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("method,config", [
    ("lens.logit", {"tokens": "last_prompt"}),
    ("lens.embed", {"tokens": "visual"}),
    ("attention.profile", {"include_attention": True}),
    ("attention.rollout", {}),
    ("attention.grad_cam", {"queries": "last_prompt", "keys": "visual"}),
    ("attention.attribution", {"queries": "last_prompt", "keys": "visual", "steps": 2}),
    ("attention.tam", {"queries": "last_prompt", "keys": "visual", "steps": 2}),
    ("attention.beyond_intuition", {"steps": 2, "variant": "head"}),
    ("attention.beyond_intuition", {"steps": 2, "variant": "token"}),
    ("attention.relevance", {}),
    ("attention.head_logits", {}),
    ("attention.reweight", {"queries": "last_prompt", "keys": "visual", "weight": 1.2}),
    ("attention.temperature", {"queries": "last_prompt", "temperature": 1.5}),
    ("causal.knockout", {"queries": "last_prompt", "keys": "visual"}),
])
def test_image_probe_tensors_match_direct_library(tiny_vlm, method, config):
    t = tiny_vlm
    # Transformers' native vision feature capture may already own hooks. Probe
    # cleanup must preserve those exact hooks, rather than remove user/backend hooks.
    before_hooks = {
        name: (dict(module._forward_hooks), dict(module._forward_pre_hooks))
        for name, module in t.raw.named_modules()
    }
    family, factory = method.split(".")
    bound = getattr(getattr(t.probe, family), factory)(**config)
    needs_metric = method not in {
        "lens.logit", "lens.embed", "attention.profile", "attention.rollout", "attention.head_logits",
    }
    options = {"metric": t.metric} if needs_metric else {}
    direct = bound.run(t.inputs, **options)
    result = VLMProbeAnalyzer(
        method, method_config=config, prober=t.probe, metric=t.metric,
        input_builder=lambda model, case: t.inputs,
    ).run(None, "image test")
    assert set(result.probe_results[0].tensors) == set(direct.tensors)
    for key, value in direct.tensors.items():
        t.torch.testing.assert_close(result.probe_results[0].tensors[key], value)
    assert {
        name: (dict(module._forward_hooks), dict(module._forward_pre_hooks))
        for name, module in t.raw.named_modules()
    } == before_hooks


def test_visual_residual_direction_and_measured_steering_match_direct(tiny_vlm):
    t = tiny_vlm
    negative = {
        "input_ids": t.torch.tensor([[1, 2, 3]]),
        "attention_mask": t.torch.ones(1, 3, dtype=t.torch.long),
    }
    visual = t.probe.causal.vsv(t.inputs, negative)
    direct = visual.configure(strength=0.02).run(t.inputs, direction=visual.directions, metric=t.metric)
    result = VLMProbeAnalyzer(
        "causal.vsv", method_config={"strength": 0.02}, prober=t.probe,
        input_builder=lambda model, case: t.inputs, metric=t.metric,
        run_config={"positive": t.inputs, "negative": negative},
    ).run(None, "image test")
    for key, value in direct.tensors.items():
        t.torch.testing.assert_close(result.probe_results[0].tensors[key], value)
    assert result.probe_results[0].metadata["visual_direction"]["trained"] is False
    assert result.metadata["automatic_calibration"] is False


def test_real_image_reference_patching_preserves_input_coordinates(tiny_vlm):
    t = tiny_vlm
    reference = {**t.inputs, "pixel_values": t.torch.zeros_like(t.inputs["pixel_values"])}
    for method in ("patch", "attribute"):
        direct = getattr(t.probe.causal, method)(tokens="visual").run(
            t.inputs, source=reference, metric=t.metric,
        )
        result = VLMProbeAnalyzer(
            f"causal.{method}", method_config={"tokens": "visual"}, prober=t.probe,
            input_builder=lambda model, case: t.inputs, metric=t.metric,
            run_config={"source": reference},
        ).run(None, "image test")
        for key, value in direct.tensors.items():
            t.torch.testing.assert_close(result.probe_results[0].tensors[key], value)


def test_automatic_hf_multimodal_encoding_preserves_template_and_visual_inputs(tiny_vlm):
    from transformers import BatchEncoding

    from evalrx.core.case import FailureCase, Inputs
    from evalrx.core.spec import ModelSpec, VisionSpec
    from evalrx.models.backends.hf.model import HFLocalModel

    t = tiny_vlm
    observed = {}

    class Processor:
        tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))

        def apply_chat_template(self, messages, **kwargs):
            observed["messages"] = messages
            observed["template_options"] = kwargs
            return "native image-and-text chat template"

        def __call__(self, **kwargs):
            observed["processor_options"] = kwargs
            return BatchEncoding({key: value.clone() for key, value in t.inputs.items()})

    spec = ModelSpec(
        key="tiny-offline-llava", family="llava", model_type="llava",
        vision=VisionSpec(image_token_id_attr="image_token_index", grid_source="fixed",
                          fixed_tokens_per_tile=4),
    )
    processor = Processor()
    model = HFLocalModel.from_loaded(t.raw, processor, spec=spec)
    image = object()  # Processor contract check; native pixel tensor above is real.
    case = FailureCase(inputs=Inputs(prompt="What is in the image?", image=image))
    direct = t.probe.attention.grad_cam().run(t.inputs, metric=t.metric)
    result = VLMProbeAnalyzer("attention.grad_cam", metric=t.metric).run(model, [case])
    for key, value in direct.tensors.items():
        t.torch.testing.assert_close(result.probe_results[0].tensors[key], value)
    assert observed["messages"][0]["content"] == [
        {"type": "image"}, {"type": "text", "text": "What is in the image?"},
    ]
    assert observed["template_options"]["add_generation_prompt"] is True
    assert observed["processor_options"]["text"] == ["native image-and-text chat template"]
    assert observed["processor_options"]["images"] == [image]


def test_genuine_vit_lrp_special_dispatch_and_caller_state():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    package = pytest.importorskip("vlmprobe")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            config = transformers.ViTConfig(
                image_size=8, patch_size=4, hidden_size=8, intermediate_size=16,
                num_hidden_layers=1, num_attention_heads=2, num_labels=3, hidden_act="gelu",
            )
            config._attn_implementation = "eager"
            raw = transformers.ViTForImageClassification(config).train()
            raw.vit.encoder.layer[0].eval()
            inputs = {"pixel_values": torch.randn(1, 3, 8, 8)}
        try:
            probe = package.Prober(raw)
        except package.CapabilityError as exc:
            pytest.skip(f"installed Transformers is outside the ViT adapter contract: {exc}")
        before = [(module, module.training) for module in raw.modules()]
        direct = probe.attention.dtd_lrp().run(inputs["pixel_values"], target=1)
        result = VLMProbeAnalyzer(
            "attention.dtd_lrp", prober=probe, run_config={"target": 1},
            input_builder=lambda model, case: inputs,
        ).run(None, "image classification")
        for key, value in direct.tensors.items():
            torch.testing.assert_close(result.probe_results[0].tensors[key], value)
        assert result.probe_results[0].tensors["patch_relevance"].shape == (1, 2, 2)
        assert all(module.training is training for module, training in before)
        assert all(parameter.grad is None for parameter in raw.parameters())
    finally:
        torch.set_num_threads(previous)
