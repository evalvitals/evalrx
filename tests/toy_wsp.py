"""A hand-built model with one planted rank-one defect, behind the ``Model`` contract.

The weight-edit code has to be tested where the right answer is known. This
model is small enough to reason about by hand:

* tokens ``0..7`` embed as scaled basis vectors of an 8-dimensional stream;
* each decoder layer adds ``down_proj(relu(up_proj(h)))`` to the stream, with
  no attention, so the output at a position depends on that position's token;
* the unembedding is the identity, so the predicted token is the largest
  stream coordinate. The clean model echoes its last input token.

The defect lives in layer :data:`DEFECT_LAYER`: neuron :data:`DEFECT_NEURON`
fires on the trigger tokens :data:`TRIGGERS` and writes strongly along
coordinate :data:`WRONG_TOKEN`. On a trigger token the model therefore answers
:data:`WRONG_TOKEN` instead of echoing; on every other token it is correct.

Removing that neuron's write along coordinate 7 (a rank-one edit with
``alpha = 1`` on one column) repairs the model, and nothing else in the model
is needed to explain the failure. Small random weights everywhere else keep
the contrast from being trivially exact.
"""

from __future__ import annotations

from typing import Any

import torch

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model, Trace

D = 8
VOCAB = 8
N_LAYERS = 3
DEFECT_LAYER = 1
DEFECT_NEURON = 5
TRIGGERS = (0, 1)
WRONG_TOKEN = 7
SAFE_TOKENS = (2, 3, 4, 5, 6)


class _Mlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(D, D, bias=False)
        self.down_proj = torch.nn.Linear(D, D, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.up_proj(h)))


class ToyDecoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _Mlp()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.mlp(h)


class ToyLM(torch.nn.Module):
    def __init__(self, noise: float = 0.02, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = torch.nn.Embedding(VOCAB, D)
        self.layers = torch.nn.ModuleList([ToyDecoderLayer() for _ in range(N_LAYERS)])
        with torch.no_grad():
            self.embed.weight.copy_(2.0 * torch.eye(D))
            for layer in self.layers:
                layer.mlp.up_proj.weight.mul_(0).add_(noise * torch.randn(D, D))
                layer.mlp.down_proj.weight.mul_(0).add_(noise * torch.randn(D, D))
            mlp = self.layers[DEFECT_LAYER].mlp
            mlp.up_proj.weight[DEFECT_NEURON].zero_()
            for token in TRIGGERS:
                mlp.up_proj.weight[DEFECT_NEURON, token] = 1.0
            mlp.down_proj.weight[:, DEFECT_NEURON].zero_()
            mlp.down_proj.weight[WRONG_TOKEN, DEFECT_NEURON] = 3.0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)           # [1, seq, D]
        for layer in self.layers:
            h = layer(h)
        return h                            # identity unembedding: logits are the stream


class _DigitTokenizer:
    """``"3 0 5"`` -> ids ``[3, 0, 5]``; the gold answer of a case is a digit."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [int(w) % VOCAB for w in str(text).split()]}


class ToyModel(Model):
    """:class:`ToyLM` under the evalrx ``Model`` contract (white-box, text only)."""

    def __init__(self, **kwargs: Any) -> None:
        self.capabilities = frozenset({Capability.GENERATE, Capability.LOGITS})
        self.modalities = frozenset({"text"})
        self._hf = (ToyLM(**kwargs).eval(), _DigitTokenizer())
        self.n_generate = 0

    @property
    def _loaded(self):
        return self._hf

    @property
    def module(self) -> ToyLM:
        return self._hf[0]

    def _ids(self, inputs: Any) -> list[int]:
        prompt = inputs.prompt if isinstance(inputs, Inputs) else str(inputs)
        return self._hf[1](prompt)["input_ids"]

    def forward(self, inputs: Any, capture: "set[Capability]", spec: Any = None) -> Trace:
        ids = self._ids(inputs)
        with torch.no_grad():
            logits = self.module(torch.tensor([ids]))[0]
        return Trace(tokens=[str(i) for i in ids], token_ids=ids,
                     provided={Capability.LOGITS}, logits=logits)

    def generate(self, inputs: Any, **kwargs: Any) -> str:
        self.n_generate += 1
        trace = self.forward(inputs, capture={Capability.LOGITS})
        return str(int(trace.logits[-1].argmax()))


def echo_case(last: int, context: tuple[int, ...] = (4, 2)) -> FailureCase:
    """A case whose correct answer is its last token; FAIL iff that token is a trigger."""
    prompt = " ".join(str(t) for t in (*context, last))
    return FailureCase(inputs=Inputs(prompt=prompt), expected=str(last),
                       label=Label.FAIL if last in TRIGGERS else Label.PASS)


def echo_batch(n_fail: int = 4, n_pass: int = 6) -> CaseBatch:
    """*n_fail* trigger cases and *n_pass* safe ones, each a distinct case object."""
    fails = [echo_case(TRIGGERS[i % len(TRIGGERS)], context=(3 + i % 4, 2 + i % 5))
             for i in range(n_fail)]
    passes = [echo_case(SAFE_TOKENS[i % len(SAFE_TOKENS)], context=(2 + i % 5, 6 - i % 4))
              for i in range(n_pass)]
    return CaseBatch(fails + passes)


def exact_score(case: FailureCase, output: str) -> bool:
    return str(output).strip() == str(case.expected).strip()


def snapshot(model: Any) -> dict:
    module = model._loaded[0] if hasattr(model, "_loaded") else model
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def unchanged(model: Any, snap: dict) -> bool:
    module = model._loaded[0] if hasattr(model, "_loaded") else model
    return all(torch.equal(v, snap[k]) for k, v in module.state_dict().items())
