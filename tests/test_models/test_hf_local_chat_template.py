"""hf_local has TWO encode paths, selected by the SPEC (``needs_multimodal_encode``:
vision or audio present), never by the input:

* text path ``_encode``      — text-only specs (Qwen3.5 LLM, Nemotron-4B). With
  ``RuntimeConfig(apply_chat_template=True)`` the prompt is rendered as one user
  turn through the TOKENIZER's chat template; the default tokenises it verbatim.
* multimodal path ``_encode_vlm`` — Qwen3.5-VL, Gemma 4, the omni models, even on
  a text-only task. Always renders through the PROCESSOR's chat template.

Both must carry ``spec.chat_template_kwargs`` (``enable_thinking=False``) the
same way: on 2026-08-21 thinking-off held on the multimodal path only, and
Qwen3.5 in the LLM cell ran to the token cap inside ``<think>``. The tests at
the bottom pin the two paths to one contract."""

from __future__ import annotations

import torch
from torch import nn

from evalvitals.core.spec import ModelSpec
from evalvitals.models.backends.base import RuntimeConfig
from evalvitals.models.backends.hf_local import HFLocalModel


class _Tok:
    def __init__(self, chat_template="{{ messages }}"):
        self.chat_template = chat_template
        self.calls: list = []

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        self.calls.append({"messages": messages, "add_generation_prompt": add_generation_prompt,
                           "tokenize": tokenize, **kwargs})
        return f"<bos>[user]{messages[0]['content']}[/user][assistant]"

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        self.last_text, self.last_add_special = text, add_special_tokens
        ids = [1] * add_special_tokens + [ord(c) % 50 + 2 for c in text]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i) - 2 + 48) for i in ids)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=4, **kwargs):
        extra = torch.tensor([[50, 51, 52, 53][:max_new_tokens]])
        seq = torch.cat([input_ids, extra], dim=1)
        if kwargs.get("return_dict_in_generate"):
            from types import SimpleNamespace

            return SimpleNamespace(sequences=seq, scores=[torch.zeros(1, 60) for _ in range(extra.shape[1])])
        return seq


def _spec(**kw):
    return ModelSpec(key="fake-text", family="fake", model_type="fake", hf_repo="fake/text",
                     auto_class="AutoModelForCausalLM", processor_class="AutoTokenizer", **kw)


def test_default_keeps_the_raw_prompt_verbatim():
    tok = _Tok()
    m = HFLocalModel.from_loaded(_Model(), tok, spec=_spec(), runtime=RuntimeConfig(device="cpu"))
    m.generate("Solve it.", max_new_tokens=2)
    assert tok.calls == [] and tok.last_text == "Solve it." and tok.last_add_special is True


def test_opt_in_renders_one_user_turn_with_the_spec_kwargs():
    tok = _Tok()
    spec = _spec(chat_template_kwargs={"enable_thinking": False})
    m = HFLocalModel.from_loaded(_Model(), tok, spec=spec,
                                 runtime=RuntimeConfig(device="cpu", apply_chat_template=True))
    m.generate("Solve it.", max_new_tokens=2)
    assert tok.calls == [{"messages": [{"role": "user", "content": "Solve it."}],
                          "add_generation_prompt": True, "tokenize": False, "enable_thinking": False}]
    assert tok.last_text.startswith("<bos>[user]Solve it.") and tok.last_add_special is False
    # logprobs takes the same path
    m.logprobs("Another.", max_new_tokens=1)
    assert tok.calls[-1]["messages"][0]["content"] == "Another."


def test_opt_in_without_a_template_falls_back_to_the_raw_prompt():
    tok = _Tok(chat_template=None)
    m = HFLocalModel.from_loaded(_Model(), tok, spec=_spec(),
                                 runtime=RuntimeConfig(device="cpu", apply_chat_template=True))
    m.generate("Solve it.", max_new_tokens=2)
    assert tok.calls == [] and tok.last_text == "Solve it."


# ---------------------------------------------------------------------------
# multimodal path + the cross-path lock
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from transformers import BatchFeature  # noqa: E402

from evalvitals.core.case import Inputs  # noqa: E402
from evalvitals.core.spec import AudioSpec, VisionSpec  # noqa: E402

_THINK_OFF = {"enable_thinking": False}


class _Proc:
    """Processor fake: chat template over content blocks + tokenisation of the
    rendered text into a real ``BatchFeature`` (what ``_encode_vlm`` consumes)."""

    def __init__(self):
        self.tokenizer = _Tok()
        self.calls: list = []
        self.last_inputs: dict = {}

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        self.calls.append({"messages": messages, "add_generation_prompt": add_generation_prompt,
                           "tokenize": tokenize, **kwargs})
        blocks = messages[0]["content"]
        rendered = "".join(b["text"] if b["type"] == "text" else f"<{b['type']}>" for b in blocks)
        return f"<bos>[user]{rendered}[/user][assistant]"

    def __call__(self, text, return_tensors="pt", **kwargs):
        self.last_inputs = {"text": text, **kwargs}
        ids = [ord(c) % 50 + 2 for c in text[0]]
        return BatchFeature({"input_ids": torch.tensor([ids]),
                             "attention_mask": torch.ones(1, len(ids), dtype=torch.long)})


class _MMModel(_Model):
    def __init__(self):
        super().__init__()
        # placeholder ids outside the fake tokeniser's range so no text token
        # is mistaken for an image/audio position by the TokenTypeMap builder
        self.config = SimpleNamespace(image_token_id=999, audio_token_id=998)


def _mm_spec(**kw):
    # Gemma 4 shape: vision + audio on ONE spec, fixed grid, no merge size
    return ModelSpec(key="fake-omni", family="fake", model_type="fake", hf_repo="fake/omni",
                     auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
                     vision=VisionSpec(merge_size_attr=None, grid_source="fixed"),
                     audio=AudioSpec(), **kw)


@pytest.mark.parametrize("apply_flag", [False, True])
def test_multimodal_spec_renders_a_text_only_input_through_the_processor_template(apply_flag):
    """Gemma 4 / Qwen3.5-VL on an LLM dataset: no image, no audio — still the
    processor template with the spec kwargs, whatever the runtime flag says
    (that flag governs the text path only)."""
    proc = _Proc()
    m = HFLocalModel.from_loaded(_MMModel(), proc, spec=_mm_spec(chat_template_kwargs=_THINK_OFF),
                                 runtime=RuntimeConfig(device="cpu", apply_chat_template=apply_flag))
    m.generate(Inputs(prompt="Solve it."), max_new_tokens=2)
    assert proc.calls == [{"messages": [{"role": "user", "content": [{"type": "text", "text": "Solve it."}]}],
                           "add_generation_prompt": True, "tokenize": False, "enable_thinking": False}]
    assert proc.tokenizer.calls == []  # the tokenizer template belongs to the text path
    assert "images" not in proc.last_inputs and "audio" not in proc.last_inputs


@pytest.mark.parametrize("slot", ["image", "audio"])
def test_multimodal_inputs_keep_the_spec_kwargs_with_the_media_block_first(slot):
    proc = _Proc()
    m = HFLocalModel.from_loaded(_MMModel(), proc, spec=_mm_spec(chat_template_kwargs=_THINK_OFF),
                                 runtime=RuntimeConfig(device="cpu"))
    media = object() if slot == "image" else np.zeros(16000, dtype=np.float32)
    m.generate(Inputs(prompt="Describe.", **{slot: media}), max_new_tokens=2)
    call = proc.calls[-1]
    assert call["enable_thinking"] is False and call["add_generation_prompt"] is True
    assert call["messages"][0]["content"] == [{"type": slot}, {"type": "text", "text": "Describe."}]
    if slot == "image":
        assert proc.last_inputs["images"] == [media]
    else:
        assert proc.last_inputs["sampling_rate"] == 16000 and len(proc.last_inputs["audio"][0]) == 16000


def test_both_encode_paths_hand_the_template_identical_kwargs():
    """The lock for the 2026-08-21 bug class: for one ``chat_template_kwargs``,
    the text path (tokenizer template) and the multimodal path (processor
    template) must pass byte-identical kwargs — for generate AND logprobs."""
    kwargs = {"enable_thinking": False, "reasoning_effort": "none"}
    tok = _Tok()
    text_model = HFLocalModel.from_loaded(_Model(), tok, spec=_spec(chat_template_kwargs=kwargs),
                                          runtime=RuntimeConfig(device="cpu", apply_chat_template=True))
    proc = _Proc()
    mm_model = HFLocalModel.from_loaded(_MMModel(), proc, spec=_mm_spec(chat_template_kwargs=kwargs),
                                        runtime=RuntimeConfig(device="cpu", apply_chat_template=True))
    inputs = Inputs(prompt="Solve it.")
    for run in (lambda m: m.generate(inputs, max_new_tokens=2),
                lambda m: m.logprobs(inputs, max_new_tokens=1)):
        run(text_model)
        run(mm_model)
        text_call, mm_call = tok.calls[-1], proc.calls[-1]
        without_messages = lambda c: {k: v for k, v in c.items() if k != "messages"}  # noqa: E731
        assert without_messages(text_call) == without_messages(mm_call) == {
            "add_generation_prompt": True, "tokenize": False, **kwargs}
        assert text_call["messages"] == [{"role": "user", "content": "Solve it."}]
        assert mm_call["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Solve it."}]}]
    assert len(tok.calls) == len(proc.calls) == 2
