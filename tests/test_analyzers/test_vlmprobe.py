"""VLMProbe integration: bounded dispatch, native contracts, and CPU numerics.

The real-model tests initialize tiny models locally and never download weights.
Optional dependencies are skipped at the fixture boundary; importing EvalRX and
discovering the analyzer remain testable without the VLMProbe extra.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from evalrx.analyzers.vlmprobe import METHOD_SPECS, VLMProbeAnalyzer, VLMProbeResult
from evalrx.core.case import CaseBatch, FailureCase, Inputs
from evalrx.core.registry import registry
from evalrx.core.result import Result
from tests.conftest import FakeModel

_METHOD_NAMES = {
    "lens.logit", "lens.tuned", "lens.jacobian", "lens.attention",
    "lens.embed", "lens.patchscope", "attention.profile", "attention.rollout",
    "attention.grad_cam", "attention.attribution", "attention.tam",
    "attention.beyond_intuition", "attention.dtd_lrp", "attention.relevance",
    "attention.head_logits", "attention.reweight", "attention.temperature",
    "causal.patch", "causal.path", "causal.attribute", "causal.steer",
    "causal.knockout", "causal.eap_ig", "causal.vsv",
}


@dataclass
class _ProbeResult:
    method: str
    tensors: dict
    metadata: dict = field(default_factory=dict)


class _BoundMethod:
    def __init__(self, parent, method, config):
        self.parent, self.method, self.config = parent, method, config
        self.name, self.probe = method.split(".")[-1], parent
        self.directions = "caller-owned-direction"
        self.metadata = {"method": "visual_direction"}

    def configure(self, **kwargs):
        self.config.update(kwargs)
        return self

    def run(self, inputs, **kwargs):
        self.parent.calls.append((self.method, "run", inputs, self.config, kwargs))
        torch = pytest.importorskip("torch")
        return _ProbeResult(self.method, {"effect": torch.tensor([0.125])}, {"actual": True})

    def sweep(self, inputs, **kwargs):
        self.parent.calls.append((self.method, "sweep", inputs, self.config, kwargs))
        torch = pytest.importorskip("torch")
        return _ProbeResult(self.method, {"effect": torch.tensor([0.125])})

    def evaluate(self, inputs, **kwargs):
        self.parent.calls.append((self.method, "evaluate", inputs, self.config, kwargs))
        torch = pytest.importorskip("torch")
        return _ProbeResult(self.method, {"recovery": torch.tensor([0.5, 1.0])})

    def generate(self, inputs, **kwargs):
        self.parent.calls.append((self.method, "generate", inputs, self.config, kwargs))
        torch = pytest.importorskip("torch")
        return _ProbeResult(self.method, {"token_ids": torch.tensor([[1, 2, 3]])})

    def load(self, directory):
        self.parent.loads.append(str(directory))
        return self


class _FakeProber:
    """Record family dispatch without importing the actual VLMProbe package."""

    def __init__(self):
        self.calls, self.loads, self.constructed = [], [], []
        self.model = SimpleNamespace()
        self.adapter = SimpleNamespace(model_id="fake-prober")
        self.spec = SimpleNamespace(output_kind="language")
        for family in ("lens", "attention", "causal"):
            methods = {}
            for full_name in _METHOD_NAMES:
                if full_name.startswith(family + "."):
                    def factory(*args, _method=full_name, **kwargs):
                        self.constructed.append((_method, args, kwargs))
                        return _BoundMethod(self, _method, kwargs)
                    methods[full_name.split(".")[1]] = factory
            setattr(self, family, SimpleNamespace(**methods))

    def describe(self):
        return {"model_id": "fake-prober", "methods": {
            name: {"available": True, "sites": [0, 1]}
            for name in _METHOD_NAMES
        }}


def _build_inputs(_model, case):
    return {"marker": case.inputs.prompt}


def test_public_inventory_and_required_constructor_keep_selection_explicit():
    assert set(METHOD_SPECS) == _METHOD_NAMES
    assert registry.analyzers.get("vlmprobe") is VLMProbeAnalyzer
    with pytest.raises(TypeError):
        VLMProbeAnalyzer()
    analyzer = VLMProbeAnalyzer(method="lens.logit", max_cases=2)
    assert analyzer.get_params()["method"] == "lens.logit"
    assert analyzer.set_params(max_cases=3) is analyzer
    assert analyzer.get_params()["max_cases"] == 3


@pytest.mark.parametrize("kwargs,error", [
    ({"method": "attention.imaginary"}, ValueError),
    ({"method": "lens.logit", "max_cases": -1}, ValueError),
    ({"method": "lens.logit", "max_cases": True}, ValueError),
    ({"method": "lens.logit", "operation": "generate"}, ValueError),
    ({"method": "lens.logit", "operation": "fit"}, ValueError),
    ({"method": "lens.logit", "method_config": []}, TypeError),
    ({"method": "lens.logit", "run_config": []}, TypeError),
])
def test_invalid_configuration_fails_before_model_execution(kwargs, error):
    with pytest.raises(error):
        VLMProbeAnalyzer(**kwargs)


@pytest.mark.parametrize("method,runtime,message", [
    ("causal.patch", {}, "reference|source"),
    ("causal.patch", {"source": {}, "donor": {}}, "only one|not both"),
    ("lens.patchscope", {}, "target_inputs"),
    ("causal.vsv", {}, "positive|negative"),
    ("attention.dtd_lrp", {}, "classification target|target"),
])
def test_missing_or_ambiguous_experiment_inputs_are_rejected(method, runtime, message):
    probe = _FakeProber()
    with pytest.raises(ValueError, match=message):
        VLMProbeAnalyzer(method=method, run_config=runtime, metric=lambda logits: logits,
            prober=probe, input_builder=_build_inputs).run(FakeModel(), "input")
    assert not probe.calls


def test_visualization_delegates_original_signed_results_and_user_geometry(monkeypatch):
    from evalrx.analyzers.vlmprobe import bridge

    native = _ProbeResult("attention_attribution", {})
    result = VLMProbeResult(analyzer="vlmprobe", model="test",
                           artifacts={"probe_results": [native]})
    calls = []
    def overlay(image, result, **kwargs):
        calls.append((image, result, kwargs))
        return "figure"
    monkeypatch.setattr(bridge, "import_vlmprobe", lambda: SimpleNamespace(
        visualization=SimpleNamespace(plot_attention_overlay=overlay)))
    image = object()
    assert result.visualize(plot="attention_overlay", image=image,
        grid=(2, 2), original_image_alpha=0.2, signed=True) == "figure"
    assert calls == [(image, native, {"grid": (2, 2), "original_image_alpha": 0.2, "signed": True})]
    with pytest.raises(ValueError, match="image"):
        result.visualize(plot="attention_overlay")
    with pytest.raises(ValueError, match="Unknown plot"):
        result.visualize(plot="invented")


def test_configured_method_requires_matching_prober_and_avoids_loading():
    pytest.importorskip("torch")
    probe = _FakeProber()
    bound = _BoundMethod(probe, "lens.tuned", {})
    bound.probe = probe
    result = VLMProbeAnalyzer(method="lens.tuned", configured_method=bound,
        prober=probe, input_builder=_build_inputs).run(FakeModel(), "input")
    assert len(result.artifacts["probe_results"]) == 1
    assert not probe.loads
    assert not probe.constructed
    other = _FakeProber()
    with pytest.raises(ValueError, match="same Prober"):
        VLMProbeAnalyzer(method="lens.tuned", configured_method=bound,
            prober=other, input_builder=_build_inputs).run(FakeModel(), "input")
    with pytest.raises(ValueError, match="name|method|match"):
        VLMProbeAnalyzer(method="lens.jacobian", configured_method=bound)
    with pytest.raises(ValueError, match="trained|calibrat|lens"):
        VLMProbeAnalyzer(method="lens.logit", configured_method=bound)
    with pytest.raises(ValueError, match="only one|ambiguous|calibration_dir|both"):
        VLMProbeAnalyzer(method="lens.tuned", configured_method=bound, calibration_dir="artifact")


@pytest.mark.parametrize("method", sorted(_METHOD_NAMES))
def test_every_public_factory_dispatches_and_keeps_native_artifacts(method, tmp_path):
    pytest.importorskip("torch")
    probe = _FakeProber()
    runtime = {"source": {"marker": "reference"}}
    config = {"layers": [0]}
    if method == "causal.vsv":
        runtime = {"positive": {"marker": "positive"}, "negative": {"marker": "negative"}}
        config["strength"] = 0.02
    if method == "attention.dtd_lrp":
        config, runtime = {}, {"target": 2}
    if method == "lens.patchscope":
        config.update(source_position=0, target_layer=0, target_position=0)
        runtime["target_inputs"] = {"marker": "target"}
    if method == "causal.path":
        config = {"senders": [(0, 0)]}
    if method == "causal.steer":
        runtime["direction"] = "caller-owned-direction"
    result = VLMProbeAnalyzer(
        method=method, method_config=config, run_config=runtime,
        prober=probe, input_builder=(lambda model, case: {"pixel_values":
            pytest.importorskip("torch").zeros(1, 3, 4, 4)}) if method == "attention.dtd_lrp" else _build_inputs,
        metric=lambda logits: logits,
        calibration_dir=tmp_path if method in {"lens.tuned", "lens.jacobian", "lens.attention"} else None,
    ).run(FakeModel(), "input")
    assert isinstance(result, Result)
    assert result.analyzer == "vlmprobe"
    assert len(result.artifacts["probe_results"]) == 1
    assert any(call[0] == method for call in probe.constructed)
    if method == "attention.dtd_lrp":
        assert tuple(probe.calls[-1][2].shape) == (1, 3, 4, 4)
        assert probe.calls[-1][4]["target"] == 2
    else:
        assert probe.calls[-1][2]["marker"] == "input"
    json.dumps(result.to_dict(), allow_nan=False)


def test_per_case_pair_callback_and_case_limit_preserve_order():
    pytest.importorskip("torch")
    probe = _FakeProber()
    cases = CaseBatch([FailureCase(inputs=Inputs(prompt=f"case-{i}")) for i in range(3)])
    seen = []

    def paired(case):
        seen.append(case.inputs.prompt)
        return {"source": {"marker": "reference-" + case.inputs.prompt}}

    result = VLMProbeAnalyzer(
        method="causal.patch", prober=probe, input_builder=_build_inputs,
        run_config=paired, metric=lambda logits: logits, max_cases=2,
    ).run(FakeModel(), cases)
    assert seen == ["case-0", "case-1"]
    assert [call[2]["marker"] for call in probe.calls] == seen
    assert [call[4]["source"]["marker"] for call in probe.calls] == [
        "reference-case-0", "reference-case-1",
    ]
    assert len(result.artifacts["probe_results"]) == 2
    assert len(result.findings["per_case"]) == 2
    assert len(result.to_json()) < 20_000


@pytest.mark.parametrize("method,operation,runtime", [
    ("causal.path", "sweep", {"donor": {"marker": "reference"}}),
    ("causal.eap_ig", "evaluate", {"source": {"marker": "reference"}, "budgets": [0, 1]}),
    ("causal.steer", "generate", {"direction": "direction", "max_new_tokens": 3}),
])
def test_nondefault_operations_dispatch(method, operation, runtime):
    pytest.importorskip("torch")
    probe = _FakeProber()
    config = {"senders": [(0, 0)]} if method == "causal.path" else {}
    if method == "causal.eap_ig":
        config["graph"] = "transformer"
    result = VLMProbeAnalyzer(
        method=method, operation=operation, method_config=config,
        run_config=runtime, metric=lambda logits: logits,
        prober=probe, input_builder=_build_inputs,
    ).run(FakeModel(), "input")
    assert probe.calls[-1][1] == operation
    assert len(result.artifacts["probe_results"]) == 1


def test_torch_free_import_registry_and_missing_optional_dependency():
    script = r'''
import importlib.abc
import sys
blocked = {"torch", "transformers", "vlmprobe", "vlm_probing"}
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError("blocked optional dependency", name=fullname.split(".")[0])
sys.meta_path.insert(0, BlockOptional())
import evalrx
from evalrx.analyzers.vlmprobe import VLMProbeAnalyzer, METHOD_SPECS
from evalrx.analyzers.vlmprobe.bridge import import_vlmprobe
assert evalrx.registry.analyzers.has("vlmprobe")
assert len(METHOD_SPECS) == 24
assert not blocked.intersection(sys.modules)
try:
    import_vlmprobe()
except (ImportError, RuntimeError) as exc:
    assert "vlmprobe" in str(exc).lower(), str(exc)
else:
    raise AssertionError("missing VLMProbe dependency was not reported")
'''
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                               text=True, env=environment, timeout=30)
    assert completed.returncode == 0, completed.stderr


@pytest.fixture(params=["gpt2", "llama"])
def tiny_local(request):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vlmprobe = pytest.importorskip("vlmprobe")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        if request.param == "gpt2":
            config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=8,
                vocab_size=23, n_positions=12, attn_pdrop=0, resid_pdrop=0,
                embd_pdrop=0, _attn_implementation="eager")
            raw = transformers.GPT2LMHeadModel(config)
        else:
            config = transformers.LlamaConfig(num_hidden_layers=2, num_attention_heads=2,
                num_key_value_heads=2, hidden_size=8, intermediate_size=16,
                vocab_size=23, max_position_embeddings=12, attention_dropout=0,
                _attn_implementation="eager")
            raw = transformers.LlamaForCausalLM(config)
    class Tokenizer:
        chat_template = None
        def __call__(self, text, return_tensors="pt", **kwargs):
            ids = [int(token) for token in text.split()]
            return {"input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        def decode(self, ids, **kwargs):
            return " ".join(str(int(token)) for token in ids)
    import evalrx
    local = evalrx.wrap(raw, Tokenizer())
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 2]]),
              "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    reference = {**inputs, "input_ids": torch.tensor([[1, 2, 3, 4]])}
    try:
        probe = vlmprobe.Prober(raw)
    except vlmprobe.CapabilityError as exc:
        torch.set_num_threads(previous_threads)
        pytest.skip(f"installed Transformers version is outside VLMProbe adapter contract: {exc}")
    try:
        yield SimpleNamespace(torch=torch, package=vlmprobe, raw=raw, local=local,
                              probe=probe, inputs=inputs, reference=reference,
                              metric=vlmprobe.TokenMargin(5, 6))
    finally:
        torch.set_num_threads(previous_threads)


def _snapshot(raw):
    return ([module.training for module in raw.modules()],
            [(tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
             for module in raw.modules()])


def test_native_text_encoding_and_final_norm_readout(tiny_local):
    t = tiny_local
    t.raw.train()
    next(iter(t.raw.modules())).training = False
    before = _snapshot(t.raw)
    result = VLMProbeAnalyzer(method="lens.logit", method_config={"layers": [0, 1]}).run(
        t.local, "1 2 3 2")
    native = result.artifacts["probe_results"][0]
    expected = t.probe.lens.logit(layers=[0, 1]).run(t.inputs)
    t.torch.testing.assert_close(native.tensors["logits"], expected.tensors["logits"])
    t.torch.testing.assert_close(native.tensors["logits"][-1], t.probe._run(t.inputs).logits[:, -1])
    assert _snapshot(t.raw) == before
    assert all(parameter.grad is None for parameter in t.raw.parameters())
    assert "logits" in result.findings["per_case"][0]["tensor_shapes"]
    assert len(result.to_json()) < 20_000


@pytest.mark.parametrize("method", ["causal.patch", "causal.attribute"])
def test_pairwise_patch_matches_native_and_identical_pair_is_zero(tiny_local, method):
    t = tiny_local
    before = _snapshot(t.raw)
    options = {"layers": [0, 1], "tokens": "last_prompt"}
    runtime = {"source": t.reference, "alignment": "position"}
    result = VLMProbeAnalyzer(
        method=method, method_config=options, run_config=runtime,
        metric={"kind": "token_margin", "positive": 5, "negative": 6},
        prober=t.probe, input_builder=lambda model, case: t.inputs,
    ).run(t.local, "test")
    expected = getattr(t.probe.causal, method.split(".")[1])(**options).run(
        t.inputs, metric=t.metric, **runtime)
    native = result.artifacts["probe_results"][0]
    for name, value in expected.tensors.items():
        t.torch.testing.assert_close(native.tensors[name], value)
    effect_name = "estimated_effect" if method == "causal.attribute" else "effect"
    assert native.tensors[effect_name].abs().max() > 1e-8
    zero = VLMProbeAnalyzer(
        method=method, method_config=options, run_config={"source": t.inputs},
        metric=t.metric, prober=t.probe, input_builder=lambda model, case: t.inputs,
    ).run(t.local, "same input").artifacts["probe_results"][0]
    t.torch.testing.assert_close(zero.tensors[effect_name],
                               t.torch.zeros_like(zero.tensors[effect_name]), atol=1e-7, rtol=0)
    assert _snapshot(t.raw) == before
    assert all(parameter.grad is None for parameter in t.raw.parameters())


@pytest.mark.parametrize("method", ["attention.rollout", "attention.grad_cam"])
def test_attention_maps_match_native_without_leaking_hooks(tiny_local, method):
    t = tiny_local
    before = _snapshot(t.raw)
    options = {"layers": [0, 1]}
    analyzer = VLMProbeAnalyzer(method=method, method_config=options,
        metric=t.metric, prober=t.probe, input_builder=lambda model, case: t.inputs)
    native = analyzer.run(t.local, "test").artifacts["probe_results"][0]
    factory = getattr(t.probe.attention, method.split(".")[1])(**options)
    expected = factory.run(t.inputs, **({"metric": t.metric} if method != "attention.rollout" else {}))
    assert native.tensors
    for name, value in expected.tensors.items():
        t.torch.testing.assert_close(native.tensors[name], value)
        assert t.torch.isfinite(value).all()
    assert _snapshot(t.raw) == before


def test_metric_failure_cleans_hooks_and_preserves_training_flags(tiny_local):
    t = tiny_local
    t.raw.train()
    if hasattr(t.raw, "transformer"):
        t.raw.transformer.h[0].ln_1.eval()
    else:
        t.raw.model.layers[0].input_layernorm.eval()
    before = _snapshot(t.raw)
    def fail(logits):
        raise RuntimeError("intentional metric failure")
    analyzer = VLMProbeAnalyzer(method="attention.grad_cam", metric=fail,
        prober=t.probe, input_builder=lambda model, case: t.inputs)
    with pytest.raises(RuntimeError, match="intentional metric failure"):
        analyzer.run(t.local, "test")
    assert _snapshot(t.raw) == before
    assert all(parameter.grad is None for parameter in t.raw.parameters())


def test_native_probe_save_roundtrip_and_no_implicit_calibration(tiny_local, tmp_path):
    t = tiny_local
    result = VLMProbeAnalyzer(method="lens.logit", prober=t.probe,
        input_builder=lambda model, case: t.inputs).run(t.local, "test")
    result.save_probes(tmp_path / "saved")
    tensors = list((tmp_path / "saved").rglob("tensors.pt"))
    metadata = list((tmp_path / "saved").rglob("metadata.json"))
    assert len(tensors) == len(metadata) == 1
    loaded = t.torch.load(tensors[0], weights_only=True)
    t.torch.testing.assert_close(loaded["logits"], result.artifacts["probe_results"][0].tensors["logits"])
    with pytest.raises((ValueError, RuntimeError), match="calibrat|fit|load|artifact"):
        VLMProbeAnalyzer(method="lens.tuned", prober=t.probe,
            input_builder=lambda model, case: t.inputs).run(t.local, "test")


def test_actual_path_sweep_matches_independent_sender_paths(tiny_local):
    t = tiny_local
    senders = [(0, 0), (0, 1)]
    runtime = {"donor": t.reference, "alignment": "position"}
    before = _snapshot(t.raw)
    native = VLMProbeAnalyzer(method="causal.path", operation="sweep",
        method_config={"senders": senders}, run_config=runtime,
        metric=t.metric, prober=t.probe,
        input_builder=lambda model, case: t.inputs).run(t.local, "test").artifacts["probe_results"][0]
    for index, sender in enumerate(senders):
        separate = t.probe.causal.path(senders=[sender]).run(t.inputs, metric=t.metric, **runtime)
        t.torch.testing.assert_close(native.tensors["effect"][index], separate.tensors["effect"])
    assert native.metadata["actual_forward_passes"] == 2 + 2 * len(senders)
    assert _snapshot(t.raw) == before


def test_actual_eap_ig_evaluation_has_native_graph_endpoints(tiny_local):
    t = tiny_local
    if type(t.raw).__name__ != "GPT2LMHeadModel":
        pytest.skip("VLMProbe transformer edge graph is audited for GPT-2 only")
    before = _snapshot(t.raw)
    result = VLMProbeAnalyzer(method="causal.eap_ig", operation="evaluate",
        method_config={"graph": "transformer", "steps": 3},
        run_config={"donor": t.reference, "budgets": [0, 20, 46],
                    "rankings": ("eap", "eap_ig"), "epsilon": 1e-8},
        metric=t.metric, prober=t.probe,
        input_builder=lambda model, case: t.inputs).run(t.local, "test")
    attribution = result.artifacts["probe_results"][0]
    assert attribution.tensors["edge_scores"].numel() == 46
    native = result.artifacts["evaluations"][0]
    assert native.tensors["all_retained_error"].abs().max() < 1e-6
    assert native.tensors["none_retained_error"].abs().max() < 1e-6
    assert t.torch.isfinite(native.tensors["circuit_score"]).all()
    assert (native.tensors["retained_edges"][:, 1] <= 20).all()
    assert _snapshot(t.raw) == before
    assert all(parameter.grad is None for parameter in t.raw.parameters())


@pytest.mark.parametrize("method,extra", [
    ("attention.reweight", {"keys": [0], "weight": 2.0}),
    ("attention.temperature", {"temperature": 1.5}),
    ("causal.knockout", {"keys": [0]}),
])
def test_real_attention_interventions_match_native_scores(tiny_local, method, extra):
    t = tiny_local
    options = {"layers": [0, 1], "queries": "last_prompt", **extra}
    before = _snapshot(t.raw)
    available = t.probe.describe()["methods"][method]["available"]
    if not available:
        # GPT-2's diagnostic returned attention is not an editable consumed
        # tensor. The integration must preserve that physical restriction.
        with pytest.raises(t.package.CapabilityError, match="editable|attention_scores"):
            VLMProbeAnalyzer(method=method, method_config=options, metric=t.metric,
                prober=t.probe, input_builder=lambda model, case: t.inputs).run(t.local, "test")
        assert _snapshot(t.raw) == before
        return
    native = VLMProbeAnalyzer(method=method, method_config=options, metric=t.metric,
        prober=t.probe, input_builder=lambda model, case: t.inputs).run(
            t.local, "test").artifacts["probe_results"][0]
    family, factory = method.split(".")
    expected = getattr(getattr(t.probe, family), factory)(**options).run(t.inputs, metric=t.metric)
    for key, value in expected.tensors.items():
        t.torch.testing.assert_close(native.tensors[key], value)
    assert native.tensors["effect"].abs().max() > 1e-9
    assert _snapshot(t.raw) == before
    assert all(parameter.grad is None for parameter in t.raw.parameters())


@pytest.mark.parametrize("method", ["lens.tuned", "lens.attention", "lens.jacobian"])
def test_explicit_calibration_artifact_load_matches_fitted_readout(tiny_local, method, tmp_path):
    """A few CPU updates test serialization; they make no quality claim."""
    t = tiny_local
    options = {"layers": [0], "tokens": "last_prompt", "binding": {
        "tokenizer_id": "integer-vocabulary@1", "calibration_id": "separate-test-calibration@1",
    }}
    before = _snapshot(t.raw)
    parameters = {key: value.detach().clone() for key, value in t.raw.named_parameters()}
    analyzer = VLMProbeAnalyzer(method=method, method_config=options, prober=t.probe,
                               input_builder=lambda model, case: t.inputs)
    fitted = analyzer.bind(t.local)
    # Different input from the evaluation case: calibration is caller-owned.
    if method == "lens.jacobian":
        fitted.fit(t.reference, skip_first=0, exclude_last=False)
    else:
        fitted.fit(t.reference, steps=3, lr=0.01)
    directory = tmp_path / "calibration"
    fitted.save(directory)
    result = VLMProbeAnalyzer(method=method, method_config=options,
        calibration_dir=directory, prober=t.probe,
        input_builder=lambda model, case: t.inputs).run(t.local, "evaluation")
    native = result.artifacts["probe_results"][0]
    expected = fitted.run(t.inputs)
    for key, value in expected.tensors.items():
        t.torch.testing.assert_close(native.tensors[key], value)
    assert result.metadata["automatic_calibration"] is False
    assert _snapshot(t.raw) == before
    for key, value in t.raw.named_parameters():
        t.torch.testing.assert_close(value, parameters[key], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in t.raw.parameters())
    wrong_options = {**options, "binding": {**options["binding"], "tokenizer_id": "wrong-vocab"}}
    with pytest.raises(ValueError, match="binding"):
        VLMProbeAnalyzer(method=method, method_config=wrong_options,
            calibration_dir=directory, prober=t.probe,
            input_builder=lambda model, case: t.inputs).run(t.local, "evaluation")


def test_automatic_binding_rejects_unsupported_backend():
    with pytest.raises(TypeError, match="HF.local|backend|explicit|provide prober"):
        VLMProbeAnalyzer(method="lens.logit").run(FakeModel(), "input")


def test_explicit_prober_must_match_exact_wrapped_model(tiny_local):
    t = tiny_local
    from evalrx.models.backends.hf.model import HFLocalModel

    other = HFLocalModel.from_loaded(copy.deepcopy(t.raw), t.local._loaded[1], spec=t.local.spec)
    with pytest.raises(ValueError, match="exact model"):
        VLMProbeAnalyzer(method="lens.logit", prober=t.probe,
            input_builder=lambda model, case: t.inputs).run(other, "input")


def test_unaudited_transformers_version_reports_explicit_adapter_requirement(tiny_local, monkeypatch):
    t = tiny_local
    adapters = importlib.import_module("vlmprobe.adapters.huggingface")
    monkeypatch.setattr(adapters, "version", lambda distribution: "5.15.0")
    with pytest.raises(t.package.CapabilityError, match="5.3.*explicit adapter"):
        VLMProbeAnalyzer(method="lens.logit").run(t.local, "1 2 3 2")
