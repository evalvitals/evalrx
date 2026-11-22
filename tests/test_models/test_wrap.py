"""evalrx.wrap(model, tokenizer) — the public on-ramp for user-supplied models.

These run on CPU with NO downloads: a tiny ``nn.Module`` + fake tokenizer stand in
for a real HF causal LM, exercising the same ``HFLocalModel`` capture path that the
curated ``load("qwen...")`` route uses.  (Real-weights parity lives behind ``--run-gpu``.)
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import evalrx
from evalrx.analyzers.lens.logit_lens import LogitLensAnalyzer
from evalrx.core.capability import Capability
from evalrx.core.case import Inputs
from evalrx.core.spec import ModelSpec, VisionSpec
from evalrx.models.backends.base import RuntimeConfig
from evalrx.models.backends.hf.inference import infer_spec
from evalrx.models.backends.hf.model import HFLocalModel


# ----------------------------------------------------------------------
# Minimal fakes mimicking the transformers surface wrap() touches
# ----------------------------------------------------------------------
class _Cfg:
    def __init__(self, model_type="fakellm", n_layers=3, attn_impl="eager", name="fake/model"):
        self.model_type = model_type
        self.num_hidden_layers = n_layers
        self._name_or_path = name
        self._attn_implementation = attn_impl
        # NOTE: deliberately no `vision_config` -> looks like a text decoder-only model.


class _Outputs:
    def __init__(self, hidden_states=None, attentions=None, logits=None):
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.logits = logits


class _Layer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)  # makes _discover see a decoder layer


class FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template

    def __call__(self, text, return_tensors="pt"):
        ids = list(range(1, len(text.split()) + 1)) or [1]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, skip_special_tokens=False):
        return f"t{ids[0]}"


class FakeCausalLM(nn.Module):
    """A tiny decoder-only stand-in: real params, HF-shaped forward outputs."""

    def __init__(self, n_layers=3, dim=8, vocab=16, n_heads=4, attn_impl="eager", emit_attn=True):
        super().__init__()
        self.config = _Cfg(n_layers=n_layers, attn_impl=attn_impl)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(dim) for _ in range(n_layers)])
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self._dim, self._vocab, self._n_layers, self._n_heads = dim, vocab, n_layers, n_heads
        self._emit_attn = emit_attn

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, output_hidden_states=False, output_attentions=False, **kw):
        seq = input_ids.shape[1]
        hs = tuple(torch.rand(1, seq, self._dim) for _ in range(self._n_layers + 1)) if output_hidden_states else None
        attns = (
            tuple(torch.rand(1, self._n_heads, seq, seq) for _ in range(self._n_layers))
            if output_attentions and self._emit_attn
            else None
        )
        return _Outputs(hidden_states=hs, attentions=attns, logits=torch.rand(1, seq, self._vocab))


# ----------------------------------------------------------------------
# infer_spec
# ----------------------------------------------------------------------
def test_infer_spec_reads_model_type_and_no_tools():
    spec = infer_spec(FakeCausalLM(), FakeTokenizer())
    assert spec.model_type == "fakellm"
    assert spec.hf_repo == ""          # already in memory -> never reloaded
    assert spec.tool_calling is False
    assert spec.is_vlm is False


def test_infer_spec_detects_tool_calling_from_template():
    spec = infer_spec(FakeCausalLM(), FakeTokenizer(chat_template="...{{ tools }}..."))
    assert spec.tool_calling is True


def test_infer_spec_rejects_vlm():
    model = FakeCausalLM()
    model.config.vision_config = object()  # now looks like a VLM
    with pytest.raises(NotImplementedError, match="VLM"):
        infer_spec(model, FakeTokenizer())


# ----------------------------------------------------------------------
# wrap() construction + capabilities
# ----------------------------------------------------------------------
def test_wrap_returns_hflocal_model():
    m = evalrx.wrap(FakeCausalLM(), FakeTokenizer())
    assert isinstance(m, HFLocalModel)


def test_wrap_infers_internals_capabilities():
    m = evalrx.wrap(FakeCausalLM(), FakeTokenizer())
    for cap in (Capability.GENERATE, Capability.LOGITS, Capability.HIDDEN_STATES, Capability.ATTENTION):
        assert cap in m.capabilities
    assert Capability.TOOL_CALLS not in m.capabilities  # no tools in template


def test_wrap_grants_tool_calls_when_template_supports_it():
    m = evalrx.wrap(FakeCausalLM(), FakeTokenizer(chat_template="{{ tools }}"))
    assert Capability.TOOL_CALLS in m.capabilities


def test_wrap_want_negotiation_passes_for_available_cap():
    m = evalrx.wrap(FakeCausalLM(), FakeTokenizer(), want={Capability.ATTENTION})
    assert Capability.ATTENTION in m.capabilities


def test_wrap_want_negotiation_rejects_missing_cap():
    from evalrx.core.capability import CapabilityError

    with pytest.raises(CapabilityError):
        evalrx.wrap(FakeCausalLM(), FakeTokenizer(), want={Capability.EMBEDDINGS})


# ----------------------------------------------------------------------
# attention fix-up
# ----------------------------------------------------------------------
def test_wrap_flips_non_eager_attention_and_warns():
    model = FakeCausalLM(attn_impl="sdpa")
    with pytest.warns(UserWarning, match="eager"):
        evalrx.wrap(model, FakeTokenizer())
    assert model.config._attn_implementation == "eager"


def test_wrap_leaves_eager_attention_untouched():
    model = FakeCausalLM(attn_impl="eager")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning expected
        evalrx.wrap(model, FakeTokenizer())


def test_forward_raises_actionable_error_when_attentions_missing():
    # Model never emits attentions (e.g. sdpa that didn't actually switch) -> clear error.
    m = evalrx.wrap(FakeCausalLM(emit_attn=False), FakeTokenizer())
    with pytest.raises(RuntimeError, match="eager"):
        m.forward("hello world", capture={Capability.ATTENTION})


# ----------------------------------------------------------------------
# end-to-end: wrap -> analyze, no weights
# ----------------------------------------------------------------------
def test_wrap_then_logit_lens_runs():
    m = evalrx.wrap(FakeCausalLM(n_layers=3, dim=8, vocab=16), FakeTokenizer())
    result = LogitLensAnalyzer(top_k=3).run(m, "the capital of france is")
    assert result.findings["n_layers"] == 4  # n_layers + 1 hidden states
    assert all(len(layer["top"]) == 3 for layer in result.findings["per_layer_top"])


def test_wrap_then_forward_captures_attention():
    m = evalrx.wrap(FakeCausalLM(n_layers=2, n_heads=4), FakeTokenizer())
    trace = m.forward("hello world", capture={Capability.ATTENTION})
    assert Capability.ATTENTION in trace.provided
    assert len(trace.attentions) == 2                 # one per layer
    assert trace.attentions[0].shape[0] == 4          # heads, after squeeze(0)


def test_wrap_unembed_weight_exposed():
    m = evalrx.wrap(FakeCausalLM(dim=8, vocab=16), FakeTokenizer())
    W = m.unembed_weight()
    assert tuple(W.shape) == (16, 8)  # (vocab, dim)


def test_hf_local_generate_uses_vlm_encoder_for_image_inputs(monkeypatch):
    spec = ModelSpec(
        key="fake-vlm",
        family="fake",
        model_type="fake_vlm",
        hf_repo="",
        auto_class="AutoModelForImageTextToText",
        processor_class="AutoProcessor",
        vision=VisionSpec(image_token_id_attr="image_token_id"),
    )
    m = HFLocalModel(spec, RuntimeConfig(max_new_tokens=4))

    class Processor:
        tokenizer = FakeTokenizer()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            assert "pixel_values" in kwargs
            return torch.tensor([[1, 2, 99]])

    called = {"vlm": False}

    def fake_encode_vlm(inputs, model, processor):
        called["vlm"] = True
        enc = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros(1, 3, 2, 2),
        }
        return enc, [1, 2], ["t1", "t2"], None

    def fail_text_encode(prompt):
        raise AssertionError("text-only encoder should not be used for VLM generate")

    m._hf = (Model(), Processor())
    monkeypatch.setattr(m, "_encode_vlm", fake_encode_vlm)
    monkeypatch.setattr(m, "_encode", fail_text_encode)

    out = m.generate(Inputs(prompt="what is here?", image=object()))
    assert called["vlm"] is True
    assert out == "t99"


def test_hf_local_generate_treats_max_tokens_as_max_new_tokens_alias():
    """FixAgent's judge-proposed L2 PipelineSpecs carry ``generation_kwargs``
    with an OpenAI-style ``max_tokens`` key (see fix_tools._safe_generation_kwargs
    and the _L2_PROMPT schema). transformers' generate() only recognises
    max_new_tokens and raises ValueError on any other kwarg it doesn't
    recognise, rather than ignoring it — so every judge-proposed L2 pipeline
    silently produced zero output on the HF-local backend. This is a
    regression test for aliasing max_tokens -> max_new_tokens."""
    m = evalrx.wrap(FakeCausalLM(), FakeTokenizer())

    class RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
            self.captured: "dict | None" = None

        def generate(self, **kwargs):
            self.captured = kwargs
            return torch.tensor([[1, 2, 99]])

    recorder = RecordingModel()
    m._hf = (recorder, FakeTokenizer())

    out = m.generate(Inputs(prompt="hi there"), max_tokens=30, temperature=0.1)
    assert recorder.captured["max_new_tokens"] == 30
    assert "max_tokens" not in recorder.captured
    assert out == "t99"

    # max_new_tokens wins and the stray max_tokens is dropped, not forwarded.
    out2 = m.generate(Inputs(prompt="hi there"), max_new_tokens=7, max_tokens=999)
    assert recorder.captured["max_new_tokens"] == 7
    assert "max_tokens" not in recorder.captured
    assert out2 == "t99"

    # Backend-neutral PipelineSpec controls must not leak invalid HF kwargs.
    m.generate(Inputs(prompt="hi"), temperature=0.0, stop=[])
    assert recorder.captured["do_sample"] is False
    assert "temperature" not in recorder.captured
    assert "stop" not in recorder.captured

    m.generate(Inputs(prompt="hi"), temperature=0.7, stop=["DONE"])
    assert recorder.captured["do_sample"] is True
    assert recorder.captured["temperature"] == 0.7
    assert recorder.captured["stop_strings"] == ["DONE"]
    assert recorder.captured["tokenizer"] is m._hf[1]


class _PaddingTokenizer(FakeTokenizer):
    pad_token_id = 0


class _RecordingLM(nn.Module):
    """Appends one token to each row: 100 + the row's prompt length."""

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs["input_ids"]
        mask = kwargs.get("attention_mask", torch.ones_like(ids))
        return torch.cat([ids, 100 + mask.sum(dim=1, keepdim=True)], dim=1)


def test_hf_local_generate_batch_pads_longest_first_and_keeps_input_order():
    m = evalrx.wrap(FakeCausalLM(), FakeTokenizer())
    m.runtime.batch_size = 2
    lm = _RecordingLM()
    m._hf = (lm, _PaddingTokenizer())

    out = m.generate_batch(["one", "one two three", "one two"], max_new_tokens=5)

    assert out == ["t101", "t103", "t102"]                   # answers follow the inputs
    first, second = lm.calls                                 # longest first
    assert first["input_ids"].tolist() == [[1, 2, 3], [0, 1, 2]]   # padded on the left
    assert first["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]
    assert second["input_ids"].tolist() == [[1]]
    assert all(call["max_new_tokens"] == 5 for call in lm.calls)


def test_hf_local_batches_a_multimodal_spec_by_stacking_single_encodings(monkeypatch):
    spec = ModelSpec(key="fake-vlm", family="fake", model_type="qwen3_5", hf_repo="",
                     auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
                     vision=VisionSpec(image_token_id_attr="image_token_id"))
    m = HFLocalModel(spec, RuntimeConfig(max_new_tokens=4))
    lm = _RecordingLM()

    class Processor:
        tokenizer = _PaddingTokenizer()

    def encode(inputs, model, processor):
        n = len(inputs.prompt.split())
        enc = {"input_ids": torch.arange(1, n + 1).unsqueeze(0),
               "attention_mask": torch.ones(1, n, dtype=torch.long),
               "pixel_values": torch.full((inputs.image, 3), float(n))}  # one row per patch
        return enc, [], [], None

    m._hf = (lm, Processor())
    monkeypatch.setattr(m, "_encode_vlm", encode)

    out = m.generate_batch([Inputs(prompt="a b", image=2), Inputs(prompt="a b c d", image=1),
                            Inputs(prompt="a", image=3)])

    assert out == ["t102", "t104", "t101"]
    (call,) = lm.calls
    assert call["input_ids"].tolist() == [[1, 2, 3, 4], [0, 0, 1, 2], [0, 0, 0, 1]]
    # Per-patch rows stay flat, in the order of the batch rows.
    assert call["pixel_values"][:, 0].tolist() == [4.0, 2.0, 2.0, 1.0, 1.0, 1.0]


def test_hf_local_runs_inputs_one_at_a_time_when_their_media_do_not_stack(monkeypatch):
    spec = ModelSpec(key="fake-vlm", family="fake", model_type="qwen3_5", hf_repo="",
                     auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
                     vision=VisionSpec(image_token_id_attr="image_token_id"))
    m = HFLocalModel(spec, RuntimeConfig(max_new_tokens=4))
    lm = _RecordingLM()

    class Processor:
        tokenizer = _PaddingTokenizer()

    def encode(inputs, model, processor):
        enc = {"input_ids": torch.tensor([[1, 2]]),
               "pixel_values": torch.zeros(1, 3, inputs.image, inputs.image)}  # per-input size
        return enc, [], [], None

    m._hf = (lm, Processor())
    monkeypatch.setattr(m, "_encode_vlm", encode)

    out = m.generate_batch([Inputs(prompt="a b", image=4), Inputs(prompt="a b", image=8)])

    assert out == ["t102", "t102"]
    assert [call["input_ids"].shape[0] for call in lm.calls] == [1, 1]


def test_hf_local_batches_inputs_with_and_without_an_image_separately(monkeypatch):
    spec = ModelSpec(key="fake-vlm", family="fake", model_type="qwen3_5", hf_repo="",
                     auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
                     vision=VisionSpec(image_token_id_attr="image_token_id"))
    m = HFLocalModel(spec, RuntimeConfig(max_new_tokens=4))
    lm = _RecordingLM()

    class Processor:
        tokenizer = _PaddingTokenizer()

    def encode(inputs, model, processor):
        n = len(inputs.prompt.split())
        enc = {"input_ids": torch.arange(1, n + 1).unsqueeze(0)}
        if inputs.image is not None:
            enc["pixel_values"] = torch.zeros(1, 3)
        return enc, [], [], None

    m._hf = (lm, Processor())
    monkeypatch.setattr(m, "_encode_vlm", encode)

    out = m.generate_batch([Inputs(prompt="a", image=1), Inputs(prompt="a b"),
                            Inputs(prompt="a b c", image=1), Inputs(prompt="a b c d")])

    assert out == ["t101", "t102", "t103", "t104"]
    assert [("pixel_values" in call, call["input_ids"].shape[0]) for call in lm.calls] == [
        (True, 2), (False, 2)]


def test_hf_local_encodes_a_long_list_a_window_at_a_time():
    m = evalrx.wrap(FakeCausalLM(), FakeTokenizer())
    m.runtime.batch_size = 2
    events: list[str] = []

    class Tok(_PaddingTokenizer):
        def __call__(self, text, return_tensors="pt"):
            events.append("encode")
            return super().__call__(text, return_tensors)

    class LM(_RecordingLM):
        def generate(self, **kwargs):
            events.append(f"generate{kwargs['input_ids'].shape[0]}")
            return super().generate(**kwargs)

    m._hf = (LM(), Tok())

    out = m.generate_batch(["one"] * 20)

    assert out == ["t101"] * 20
    # Eight encodings, their four batches, then the next eight: never all twenty at once.
    assert events[:12] == ["encode"] * 8 + ["generate2"] * 4
    assert events.count("encode") == 20 and events.count("generate2") == 10


def test_hf_local_does_not_batch_a_multimodal_family_that_was_not_checked(monkeypatch):
    spec = ModelSpec(key="fake-vlm", family="fake", model_type="some_new_vlm", hf_repo="",
                     auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
                     vision=VisionSpec(image_token_id_attr="image_token_id"))
    m = HFLocalModel(spec, RuntimeConfig(max_new_tokens=4))
    lm = _RecordingLM()

    class Processor:
        tokenizer = _PaddingTokenizer()

    def encode(inputs, model, processor):
        return ({"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.zeros(1, 3)},
                [], [], None)

    m._hf = (lm, Processor())
    monkeypatch.setattr(m, "_encode_vlm", encode)

    assert m.generate_batch([Inputs(prompt="a b", image=1)] * 3) == ["t102"] * 3
    assert [call["input_ids"].shape[0] for call in lm.calls] == [1, 1, 1]


def test_hf_local_logprobs_batch_ends_each_row_at_its_own_stop_token():
    m = evalrx.wrap(FakeCausalLM(), FakeTokenizer())

    class Out:
        # Row 0 stops after one token (7 = end of sequence) and is padded.
        sequences = torch.tensor([[1, 2, 7, 0], [0, 1, 5, 6]])
        scores = (torch.zeros(2, 16), torch.zeros(2, 16))

    class LM(nn.Module):
        generation_config = type("G", (), {"eos_token_id": 7})()

        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            assert kwargs["output_scores"] and not kwargs["do_sample"]
            return Out()

    m._hf = (LM(), _PaddingTokenizer())

    first, second = m.logprobs_batch(["one two", "one"])

    assert [t.token for t in first] == ["t7"]
    assert [t.token for t in second] == ["t5", "t6"]


def test_hf_local_uses_the_runtime_sampling_unless_the_call_passes_its_own():
    m = evalrx.wrap(FakeCausalLM(), FakeTokenizer())
    m.runtime.sampling = {"do_sample": True, "temperature": 0.6, "top_p": 0.95}
    lm = _RecordingLM()
    m._hf = (lm, FakeTokenizer())

    m.generate(Inputs(prompt="hi"))
    assert {k: lm.calls[-1][k] for k in ("do_sample", "temperature", "top_p")} == {
        "do_sample": True, "temperature": 0.6, "top_p": 0.95}

    m.generate(Inputs(prompt="hi"), temperature=1.0)
    assert lm.calls[-1]["temperature"] == 1.0 and lm.calls[-1]["do_sample"] is True

    m.generate(Inputs(prompt="hi"), do_sample=False)     # greedy: no sampling knobs reach HF
    assert lm.calls[-1]["do_sample"] is False
    assert not {"temperature", "top_p"} & set(lm.calls[-1])


def test_vllm_offline_hands_the_engine_every_prompt_at_once(monkeypatch):
    import sys
    import types

    from evalrx.models.backends.vllm_offline import VLLMOfflineModel

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=SamplingParams))

    class Engine:
        def __init__(self):
            self.calls: list[tuple[list[str], dict]] = []

        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompts, params):
            self.calls.append((list(prompts), params.kwargs))
            completion = lambda p: types.SimpleNamespace(  # noqa: E731
                text=p.upper(), token_ids=[len(p)], logprobs=[{len(p): -0.5, 99: -2.0}])
            return [types.SimpleNamespace(outputs=[completion(p)]) for p in prompts]

    spec = ModelSpec(key="fake-lm", family="fake", model_type="fake", hf_repo="",
                     auto_class="AutoModelForCausalLM", processor_class="AutoTokenizer")
    m = VLLMOfflineModel(spec, RuntimeConfig(max_new_tokens=8))
    m._llm = engine = Engine()

    assert m.generate_batch(["ab", "cde"], do_sample=False) == ["AB", "CDE"]
    rows = m.logprobs_batch(["ab", "cde"], top_k=2)

    assert [prompts for prompts, _ in engine.calls] == [["ab", "cde"], ["ab", "cde"]]
    assert engine.calls[1][1]["logprobs"] == 2
    assert [[(t.token, t.logprob) for t in row] for row in rows] == [[("t2", -0.5)], [("t3", -0.5)]]
    assert rows[0][0].top == {"t2": -0.5, "t99": -2.0}
