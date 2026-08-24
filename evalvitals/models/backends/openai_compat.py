"""Built-in OpenAI-compatible client — chat_fn/generate_fn factories for APIModel.

Covers every OpenAI-compatible endpoint with one code path: a local
``vllm serve`` (the scale path for open Qwen3-VL agents — served with
``--enable-auto-tool-choice --tool-call-parser hermes`` it returns native
structured ``tool_calls``), OpenAI itself, or any compatible gateway.

Wire conversion: agent messages use the transformers-style content-block
convention (``{"type": "image", "image": PIL | path | URL}``); this module
converts image blocks to OpenAI ``image_url`` blocks (data URLs for local
images).  Sampling defaults to ``temperature=0`` — diagnosed runs should be
as deterministic as the endpoint allows.

Torch-free; ``openai`` and ``PIL`` are imported lazily.

Usage::

    from evalvitals import compose
    from evalvitals.models.backends.openai_compat import openai_runtime

    rt = openai_runtime(base_url="http://localhost:8901/v1")   # vllm serve
    vlm = compose("qwen3-vl-2b-instruct", "api", runtime=rt)
    Agent(vlm, tools=[...]).run(case)
"""

from __future__ import annotations

import base64
import io
from typing import Any, Callable, Optional

from evalvitals.core.tool import ChatTurn
from evalvitals.models.backends.api import parse_openai_logprobs
from evalvitals.models.backends.base import RuntimeConfig


def _to_data_url(image: Any) -> str:
    """Encode *image* (PIL / path / URL) as an ``image_url`` value.

    http(s) URLs pass through; everything else becomes a base64 PNG data URL.
    """
    if isinstance(image, str) and image.startswith(("http://", "https://", "data:")):
        return image
    if not (hasattr(image, "save") and hasattr(image, "mode")):  # a path -> open it
        from PIL import Image

        image = Image.open(image).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def to_openai_messages(messages: list) -> list[dict]:
    """Convert internal content-block messages into OpenAI wire format.

    Plain-string content, assistant ``tool_calls`` and ``role="tool"`` results
    pass through unchanged; ``{"type": "image", "image": ...}`` blocks become
    ``image_url`` blocks.
    """
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        blocks: list[dict] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                blocks.append(
                    {"type": "image_url", "image_url": {"url": _to_data_url(block.get("image"))}}
                )
            else:
                blocks.append(block)
        out.append({**msg, "content": blocks})
    return out


def _raw_tool_calls(message: Any) -> Optional[list]:
    calls = getattr(message, "tool_calls", None)
    if not calls:
        return None
    return [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in calls]


def openai_chat_fn(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Any = None,
    timeout: float = 300.0,
    **sampling: Any,
) -> Callable[..., ChatTurn]:
    """Build a tool-aware ``chat_fn(messages, tools, model) -> ChatTurn``.

    *client* may be injected (tests, custom transports); otherwise a lazy
    ``openai.OpenAI(base_url=..., api_key=...)`` is created on first call —
    ``api_key`` defaults to ``"EMPTY"``, which is what a local ``vllm serve``
    expects.  Extra keyword arguments become sampling parameters on every
    request (``temperature`` defaults to 0).
    """
    sampling.setdefault("temperature", 0.0)
    state = {"client": client}

    def _client():
        if state["client"] is None:
            import openai

            state["client"] = openai.OpenAI(
                base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout
            )
        return state["client"]

    def _fn(messages: list, tools=None, model: str = "") -> ChatTurn:
        kwargs: dict[str, Any] = dict(sampling)
        if tools:
            kwargs["tools"] = tools
            kwargs.setdefault("tool_choice", "auto")
        resp = _client().chat.completions.create(
            model=model, messages=to_openai_messages(messages), **kwargs
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return ChatTurn(
            text=choice.message.content or "",
            raw_tool_calls=_raw_tool_calls(choice.message),
            finish_reason=choice.finish_reason,
            usage=(
                {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                }
                if usage is not None
                else None
            ),
        )

    return _fn


def openai_generate_fn(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Any = None,
    timeout: float = 300.0,
    **sampling: Any,
) -> Callable[..., str]:
    """Build a simple ``generate_fn(prompt, model=...) -> str`` on the same client."""
    chat = openai_chat_fn(
        base_url=base_url, api_key=api_key, client=client, timeout=timeout, **sampling
    )

    def _fn(prompt: str, model: str = "", **kw) -> str:
        return chat([{"role": "user", "content": prompt}], tools=None, model=model).text

    return _fn


def openai_logprobs_fn(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Any = None,
    timeout: float = 300.0,
    top_logprobs: int = 5,
    **sampling: Any,
) -> Callable[..., list]:
    """Build a ``logprobs_fn(prompt, model=...) -> list[TokenLogprob]``.

    An OpenAI-compatible server returns per-token logprobs when asked, and
    ``vllm serve`` is one. Without this the api backend declares only GENERATE,
    so every analyzer that reads answer-token uncertainty is skipped as
    unsupported -- on the LLM benchmark set that is `calibration` and
    `logprob_entropy`, two of the eight pinned probes, dropped not because the
    server cannot answer but because nobody asked it.
    """
    sampling.setdefault("temperature", 0.0)
    state = {"client": client}

    def _client():
        if state["client"] is None:
            import openai

            state["client"] = openai.OpenAI(
                base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout
            )
        return state["client"]

    def _fn(prompt: str, model: str = "", **kw) -> list:
        kwargs: dict[str, Any] = {**sampling, **kw}
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = int(kwargs.pop("top_logprobs", top_logprobs))
        resp = _client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        logprobs = getattr(resp.choices[0], "logprobs", None)
        content = getattr(logprobs, "content", None) if logprobs is not None else None
        # A server that accepts `logprobs=True` and returns nothing has told us
        # it does not support them. Returning [] would read as "this answer had
        # no tokens", so let the caller see the empty result for what it is.
        return parse_openai_logprobs(
            [item.model_dump() if hasattr(item, "model_dump") else item for item in (content or [])]
        )

    return _fn


def openai_runtime(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Any = None,
    timeout: float = 300.0,
    with_logprobs: bool = True,
    **sampling: Any,
) -> RuntimeConfig:
    """A ready :class:`RuntimeConfig` for ``compose(key, "api", runtime=...)``.

    ``with_logprobs`` wires the logprobs path, which is what makes the backend
    claim :attr:`Capability.LOGPROBS`. Pass False for an endpoint that rejects
    the parameter -- claiming a capability the server does not have turns a
    skipped analyzer into a failing one.
    """
    kw = dict(base_url=base_url, api_key=api_key, client=client, timeout=timeout, **sampling)
    return RuntimeConfig(
        chat_fn=openai_chat_fn(**kw),
        generate_fn=openai_generate_fn(**kw),
        logprobs_fn=openai_logprobs_fn(**kw) if with_logprobs else None,
    )
