"""vLLM offline backend — in-process, high-throughput generation + logprobs.

The *distinct* role of vLLM (NOT the same as ``api``): an in-process engine you
submit batches to, with continuous batching and ``prompt_logprobs`` — the
throughput sweet spot for perturbation sweeps (RISE / SHAP) and logprob scoring
on open weights.  It is a **black-box** backend: it exposes outputs only
(``GENERATE`` / ``TOOL_CALLS`` / ``LOGPROBS``) and **never** attention / hidden
states / gradients — paged/fused kernels don't materialise them.  White-box
capture is ``hf_local``'s job.

(A ``vllm serve`` HTTP endpoint, by contrast, is reached via the ``api`` backend.)

``vllm`` is imported lazily inside ``load()`` so this module imports without it.
"""

from __future__ import annotations

from typing import Any

from evalrx.core.capability import Capability, CapabilityError
from evalrx.core.case import Inputs
from evalrx.core.model import Model, TokenLogprob, Trace
from evalrx.core.tool import ChatTurn
from evalrx.models.backends.base import Backend, RuntimeConfig


class VLLMOfflineModel(Model):
    """A locally-loaded vLLM engine driven offline (batched)."""

    def __init__(self, spec, runtime: RuntimeConfig) -> None:
        self.spec = spec
        self.runtime = runtime
        self._llm = None  # lazy vllm.LLM
        caps = {Capability.GENERATE, Capability.LOGPROBS}
        if spec.tool_calling:  # conditional, like hf_local: needs a tool chat template
            caps.add(Capability.TOOL_CALLS)
        self.capabilities = frozenset(caps)
        self.modalities = frozenset(spec.modalities)  # text / image / audio / video, from the spec

    # -- lazy load -----------------------------------------------------
    def load(self) -> None:
        from vllm import LLM

        self._llm = LLM(
            model=self.spec.hf_repo,
            trust_remote_code=self.spec.trust_remote_code,
            dtype=self.runtime.dtype,
            **self.runtime.engine_kwargs,
        )

    @property
    def _engine(self):
        if self._llm is None:
            self.load()
        return self._llm

    @staticmethod
    def _as_prompt(inputs: Any) -> str:
        return inputs.prompt if isinstance(inputs, Inputs) else str(inputs)

    def _sampling(self, *, max_tokens: int | None = None, temperature: float = 0.0, logprobs: int | None = None, **_):
        from vllm import SamplingParams

        return SamplingParams(
            max_tokens=max_tokens or self.runtime.max_new_tokens,
            temperature=temperature,
            logprobs=logprobs,
        )

    # -- interface -----------------------------------------------------
    def generate(self, inputs: Any, **kwargs) -> str:
        out = self._engine.generate([self._as_prompt(inputs)], self._sampling(**kwargs))
        return out[0].outputs[0].text

    def chat(self, messages: list, tools=None) -> ChatTurn:
        if Capability.TOOL_CALLS not in self.capabilities:
            raise CapabilityError(analyzer="chat", model=repr(self), missing={Capability.TOOL_CALLS})
        tok = self._engine.get_tokenizer()
        text = tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, **self.spec.chat_template_kwargs
        )
        out = self._engine.generate([text], self._sampling())
        return ChatTurn(text=out[0].outputs[0].text, raw_tool_calls=None)  # codec parses the text

    def logprobs(self, inputs: Any, top_k: int = 5, **kwargs) -> list[TokenLogprob]:
        out = self._engine.generate([self._as_prompt(inputs)], self._sampling(logprobs=top_k, **kwargs))
        comp = out[0].outputs[0]
        tok = self._engine.get_tokenizer()
        result: list[TokenLogprob] = []
        for tid, lp_dict in zip(comp.token_ids, comp.logprobs or []):
            entry = lp_dict.get(tid)
            chosen = float(getattr(entry, "logprob", entry)) if entry is not None else 0.0
            chosen_tok = getattr(entry, "decoded_token", None) or tok.decode([tid])
            top = {
                (getattr(v, "decoded_token", None) or tok.decode([k])): float(getattr(v, "logprob", v))
                for k, v in lp_dict.items()
            }
            result.append(TokenLogprob(token=chosen_tok, logprob=chosen, top=top))
        return result

    def forward(self, inputs: Any, capture: set[Capability], spec=None) -> Trace:
        raise CapabilityError(  # vLLM exposes no internals — use hf_local for white-box
            analyzer="forward", model=repr(self), missing=set(capture) or {Capability.HIDDEN_STATES},
        )

    def __repr__(self) -> str:
        status = "loaded" if self._llm is not None else "lazy"
        return f"VLLMOfflineModel(key={self.spec.key!r}, {status})"


class VLLMOfflineBackend(Backend):
    kind = "vllm_offline"
    capabilities = frozenset({Capability.GENERATE, Capability.TOOL_CALLS, Capability.LOGPROBS})

    def build(self, spec, runtime: RuntimeConfig) -> VLLMOfflineModel:
        return VLLMOfflineModel(spec, runtime)
