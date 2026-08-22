"""hf_local text path: ``RuntimeConfig(apply_chat_template=True)`` renders a text-only
spec's prompt as one user turn through the tokenizer's chat template, carrying
``spec.chat_template_kwargs``; the default tokenises the raw prompt verbatim."""

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
