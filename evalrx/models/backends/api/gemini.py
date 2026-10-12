"""Built-in Gemini client — chat_fn / generate_fn / logprobs_fn factories for APIModel.

The mirror of :mod:`openai_compat` on Google's official ``google-genai`` SDK,
so a closed-weight Gemini model rides the same ``api`` backend as a served
open model::

    from evalrx import compose
    from evalrx.models.backends.api.gemini import gemini_runtime

    rt = gemini_runtime(temperature=0.6, max_output_tokens=2048)   # key: GEMINI_API_KEY
    llm = compose("gemini-3.6-flash", "api", runtime=rt)

What the wrapper adds over the raw SDK, in the order it matters for a
diagnosis run:

* **Thinking floor.** The benchmark runs every model with thinking as far OFF
  as the model allows. 3.x models take ``thinking_level`` (3.7-flash and
  3.1-pro-preview bottom out at ``low``, the others at ``minimal``); 2.5 models take
  ``thinking_budget`` (``0`` on flash / flash-lite; 2.5-pro cannot switch it
  off, floor 128). :func:`thinking_config` resolves the floor per model id,
  ``ThinkingPolicy(level=..., budget=...)`` overrides it, and a model that
  rejects the config falls back to the API default ONCE per model, logged.
* **Thought parts are not answer text.** The answer is rebuilt from the
  non-``thought`` parts only (``response.text`` would fold summaries in).
* **OpenAI-named sampling kwargs are translated** (``max_tokens`` →
  ``max_output_tokens``, ``n`` → ``candidate_count``, ``stop`` →
  ``stop_sequences``): the fix stage's L0 candidates re-issue the baseline
  decoding controls under the names the endpoint path uses.
* **Retry with backoff** on 429 / 5xx (``RESOURCE_EXHAUSTED`` / ``UNAVAILABLE``)
  — the discovery pass fires ``--concurrency`` requests at once.
* **Media goes inline**: image → PNG part, audio → its own format (or 16-bit
  WAV for a waveform); the API takes up to 20 MB per request inline.
* **Logprobs are OFF by default in this factory.** The benchmark opts into
  ``with_logprobs="auto", logprobs_model=...``: a real request must return
  token probabilities before LOGPROBS is advertised. Unsupported responses
  are recorded; operational errors propagate. Raw logits are never claimed.

Torch-free; ``google-genai`` is imported lazily (``pip install 'evalrx[gemini]'``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from evalrx.core.model import TokenLogprob
from evalrx.core.tool import ChatTurn
from evalrx.models.backends.api.gemini_model import (
    _audio_bytes,
    _png_bytes,
    _to_genai_contents,
    _to_genai_tools,
)
from evalrx.models.backends.base import RuntimeConfig

logger = logging.getLogger(__name__)

API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
THINKING_LEVELS = ("minimal", "low", "medium", "high")

#: Lowest ``thinking_level`` each 3.x model accepts (model cards + the thinking
#: guide, read 2026-08-25). Keys match as prefixes, so a dated variant
#: (``gemini-3.5-flash-lite-preview-…``) resolves to its base model.
LEVEL_FLOOR: dict[str, str] = {
    "gemini-3.7-flash": "low",          # minimal is not offered on 3.7-flash
    "gemini-3.6-flash": "minimal",
    "gemini-3.5-flash": "minimal",
    "gemini-3.5-flash-lite": "minimal",
    "gemini-3.1-flash-lite": "minimal",
    "gemini-3.1-pro-preview": "low",     # thinking cannot be disabled
}
#: Lowest ``thinking_budget`` each 2.5 model accepts (0 = thinking off).
BUDGET_FLOOR: dict[str, int] = {
    "gemini-2.5-flash": 0,
    "gemini-2.5-flash-lite": 0,
    "gemini-2.5-pro": 128,
}
#: A ``--thinking-level`` asked of a budget model: rough token equivalents.
LEVEL_TO_BUDGET: dict[str, int] = {"minimal": 0, "low": 1024, "medium": 8192, "high": 24576}

#: Thought tokens count against ``max_output_tokens`` (measured 2026-08-25 on
#: 3.6/3.7-flash: at a 16-token cap with thinking_level=low the answer is
#: truncated or empty, finish_reason=MAX_TOKENS). Whenever the effective
#: thinking config is not "off" (level above minimal, or a budget above 0),
#: the request cap is raised by this headroom so a short-answer task's 64-token
#: budget measures the answer, not the thoughts. 3.7-flash cannot go below
#: ``low``, so this is its normal operating mode.
THINKING_OUTPUT_HEADROOM = 1024


def thinking_spends_output_tokens(think: "dict | None") -> bool:
    """True when *think* lets the model emit thought tokens (they share the cap)."""
    if not think:
        return False
    if think.get("thinking_budget") is not None:
        return int(think["thinking_budget"]) > 0
    return str(think.get("thinking_level", "")).lower() != "minimal"

#: google-genai logs an "automatic function calling (AFC) … not recommended"
#: notice from ``Models.generate_content`` at WARNING; its once-per-process
#: guard races under concurrent discovery and the notice floods stderr (seen
#: 2026-08-25, SDK 2.20.0, concurrency 8). AFC is never enabled by this runtime
#: (tools go in as declarations, not callables), so the notice — and the "AFC is
#: enabled"/"AFC remote call" INFO chatter — carries no signal here. Drop exactly
#: those records; every other SDK warning stays visible.
_SDK_NOISE_MARKERS = ("automatic function calling", "AFC is enabled", "AFC remote call")


class _DropSdkNoise(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(marker in msg for marker in _SDK_NOISE_MARKERS)


_SDK_NOISE_FILTER = _DropSdkNoise()


def silence_sdk_chatter() -> None:
    """Keep google-genai AFC chatter off stderr (idempotent, per-logger)."""
    sdk_logger = logging.getLogger("google_genai.models")
    if _SDK_NOISE_FILTER not in sdk_logger.filters:
        sdk_logger.addFilter(_SDK_NOISE_FILTER)


_RETRY_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
_RETRY_MARKERS = ("resource_exhausted", "unavailable", "deadline_exceeded", "overloaded", "rate limit")

# OpenAI / transformers names the harness uses -> GenerateContentConfig names.
_RENAME = {"max_tokens": "max_output_tokens", "max_new_tokens": "max_output_tokens",
           "n": "candidate_count", "stop": "stop_sequences"}
_PASS = frozenset({
    "temperature", "top_p", "top_k", "seed", "max_output_tokens", "candidate_count",
    "stop_sequences", "response_logprobs", "logprobs", "safety_settings",
    "presence_penalty", "frequency_penalty",
})
_DROP = frozenset({"do_sample", "extra_body", "chat_template_kwargs", "top_logprobs", "timeout", "stream"})


# ----------------------------------------------------------------------
# Thinking policy
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ThinkingPolicy:
    """How much the model may think.

    ``level`` / ``budget`` are explicit overrides (``--thinking-level`` /
    ``--thinking-budget``); with neither, ``floor=True`` sends the lowest
    setting the model accepts and ``floor=False`` leaves the API default
    (``--enable-thinking``).
    """

    level: Optional[str] = None
    budget: Optional[int] = None
    floor: bool = True


def _base_model(model: str) -> str:
    return model.split("/", 1)[1] if model.startswith("models/") else model


def _lookup(table: dict, model: str) -> Optional[str]:
    """Longest key that is *model* or a dashed prefix of it."""
    name = _base_model(model)
    best = None
    for key in table:
        if name == key or name.startswith(key + "-"):
            if best is None or len(key) > len(best):
                best = key
    return best


def thinking_config(model: str, policy: ThinkingPolicy = ThinkingPolicy()) -> Optional[dict]:
    """``ThinkingConfig`` kwargs for *model* under *policy*; None = the API default."""
    if policy.budget is not None:
        return {"thinking_budget": int(policy.budget)}
    level_key = _lookup(LEVEL_FLOOR, model)
    budget_key = _lookup(BUDGET_FLOOR, model)
    if policy.level is not None:
        level = str(policy.level).lower()
        if level not in THINKING_LEVELS:
            raise ValueError(f"unknown thinking level {policy.level!r}; one of {THINKING_LEVELS}")
        if budget_key is not None:
            return {"thinking_budget": max(BUDGET_FLOOR[budget_key], LEVEL_TO_BUDGET[level])}
        if level_key is not None:
            floor = LEVEL_FLOOR[level_key]
            if THINKING_LEVELS.index(level) < THINKING_LEVELS.index(floor):
                logger.warning("%s does not go below thinking_level=%s; sending it instead of %s",
                               model, floor, level)
                level = floor
        return {"thinking_level": level}
    if not policy.floor:
        return None
    if level_key is not None:
        return {"thinking_level": LEVEL_FLOOR[level_key]}
    if budget_key is not None:
        return {"thinking_budget": BUDGET_FLOOR[budget_key]}
    logger.warning("%s: no thinking floor on record; thinking stays at the API default", model)
    return None


# ----------------------------------------------------------------------
# Request shaping
# ----------------------------------------------------------------------
def translate_sampling(kwargs: dict) -> dict:
    """OpenAI / transformers sampling names -> ``GenerateContentConfig`` fields.

    Unknown names are dropped (logged at debug) rather than sent: the SDK
    raises on them, and the harness passes decoding controls under the
    endpoint path's names (``max_tokens``, ``do_sample``, ``extra_body``).
    """
    out: dict[str, Any] = {}
    unknown = []
    for key, value in kwargs.items():
        name = _RENAME.get(key, key)
        if name == "logprobs" and isinstance(value, bool):
            continue  # OpenAI's flag; Gemini's `logprobs` is the top-k count (set by logprobs_fn)
        if name in _PASS:
            if name == "stop_sequences" and isinstance(value, str):
                value = [value]
            if name in ("max_output_tokens", "candidate_count", "top_k", "logprobs", "seed"):
                value = int(value)
            out[name] = value
        elif name in _DROP:
            continue
        else:
            unknown.append(key)
    if unknown:
        logger.debug("gemini: ignoring unsupported sampling kwargs %s", unknown)
    return out


def _parts(types: Any, prompt: str, image: Any = None, audio: Any = None) -> list:
    """One user turn as genai parts: media first, then the prompt."""
    parts = []
    if audio is not None:
        data, mime = _audio_bytes(audio)
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    if image is not None:
        parts.append(types.Part.from_bytes(data=_png_bytes(image), mime_type="image/png"))
    parts.append(types.Part(text=prompt))
    return parts


def _retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _RETRY_CODES:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _RETRY_MARKERS)


class _Runtime:
    """One client + the per-model decisions the three factories share."""

    def __init__(self, *, api_key: Optional[str], client: Any, timeout: float, retries: int,
                 policy: ThinkingPolicy, top_logprobs: int, sampling: dict) -> None:
        self.api_key = api_key
        self._client = client
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.policy = policy
        self.top_logprobs = int(top_logprobs)
        self.sampling = dict(sampling)
        #: shared, readable through ``fn.state`` on every factory output
        self.state: dict[str, Any] = {
            "model_version": None, "calls": 0, "retries": 0, "thinking_fallback": [],
            "headroom_for": [],
        }
        self._lock = threading.Lock()
        silence_sdk_chatter()

    # -- plumbing ------------------------------------------------------
    def genai(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                "gemini_runtime needs the google-genai package: pip install 'evalrx[gemini]'"
            ) from exc
        with self._lock:
            if self._client is None:
                key = self.api_key or next((os.environ[e] for e in API_KEY_ENVS if os.environ.get(e)), None)
                if not key:
                    raise ValueError("No Gemini API key: pass api_key= or set GEMINI_API_KEY")
                self._client = genai.Client(
                    api_key=key, http_options=types.HttpOptions(timeout=int(self.timeout * 1000))
                )
        return self._client, types

    def thinking(self, model: str) -> Optional[dict]:
        if model in self.state["thinking_fallback"]:
            return None
        return thinking_config(model, self.policy)

    def config(self, types: Any, model: str, extra: dict, *, system: Any = None,
               tools: Any = None, logprobs: bool = False):
        kw = translate_sampling({**self.sampling, **extra})
        think = self.thinking(model)
        if think:
            kw["thinking_config"] = types.ThinkingConfig(**think)
            if thinking_spends_output_tokens(think) and kw.get("max_output_tokens"):
                # thought tokens share max_output_tokens: give them their own room
                kw["max_output_tokens"] = int(kw["max_output_tokens"]) + THINKING_OUTPUT_HEADROOM
                if model not in self.state["headroom_for"]:
                    self.state["headroom_for"].append(model)
                    logger.info("%s thinks at %s: max_output_tokens raised by %d for the thought tokens",
                                model, think, THINKING_OUTPUT_HEADROOM)
        if system is not None:
            kw["system_instruction"] = system
        if tools:
            kw["tools"] = tools
        if logprobs:
            kw["response_logprobs"] = True
            kw["logprobs"] = self.top_logprobs
        return types.GenerateContentConfig(**kw)

    def request(self, model: str, contents: Any, extra: dict, *, system: Any = None,
                tools: Any = None, logprobs: bool = False):
        client, types = self.genai()
        attempt = 0
        while True:
            config = self.config(types, model, extra, system=system, tools=tools, logprobs=logprobs)
            try:
                response = self._generate_with_deadline(
                    client, model=model, contents=contents, config=config,
                )
            except Exception as exc:
                text = str(exc)
                if (getattr(exc, "code", None) == 400 and "thinking" in text.lower()
                        and self.thinking(model) is not None):
                    logger.warning("%s rejected thinking config %s (%s); using the API default from now on",
                                   model, self.thinking(model), text[:160])
                    self.state["thinking_fallback"].append(model)
                    continue
                if attempt < self.retries and _retryable(exc):
                    delay = min(30.0, 1.5 * (2 ** attempt)) + random.uniform(0.0, 0.5)
                    self.state["retries"] += 1
                    logger.warning("gemini %s: %s; retry %d/%d in %.1fs",
                                   model, text[:120], attempt + 1, self.retries, delay)
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise
            self.state["calls"] += 1
            version = getattr(response, "model_version", None)
            if version:
                self.state["model_version"] = version
            return response

    def _generate_with_deadline(self, client: Any, *, model: str, contents: Any, config: Any):
        """Enforce a wall-clock deadline around a synchronous SDK request.

        ``google-genai``'s HTTP timeout is normally sufficient, but an audio
        upload can occasionally leave its synchronous call blocked beyond that
        setting.  Benchmark M1 fans those calls out, so one such request used
        to keep an entire diagnosis run alive indefinitely.  A daemon helper
        lets the caller recover at ``self.timeout``; it deliberately cannot
        join a wedged transport thread, so it never delays process shutdown.
        """
        done = threading.Event()
        result: dict[str, Any] = {}

        def _call() -> None:
            try:
                result["response"] = client.models.generate_content(
                    model=model, contents=contents, config=config,
                )
            except BaseException as exc:  # re-raised on the caller thread
                result["error"] = exc
            finally:
                done.set()

        worker = threading.Thread(target=_call, name="gemini-request", daemon=True)
        worker.start()
        if not done.wait(self.timeout):
            raise TimeoutError(f"Gemini request exceeded wall-clock timeout of {self.timeout:.1f}s")
        if "error" in result:
            raise result["error"]
        return result["response"]


# ----------------------------------------------------------------------
# Response reading
# ----------------------------------------------------------------------
def _candidate(response: Any):
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        feedback = getattr(response, "prompt_feedback", None)
        reason = getattr(feedback, "block_reason", None) if feedback is not None else None
        if reason:
            logger.warning("gemini returned no candidate (block_reason=%s)", reason)
        return None, []
    cand = candidates[0]
    content = getattr(cand, "content", None)
    return cand, (getattr(content, "parts", None) or [])


def answer_text(response: Any) -> str:
    """The model's answer: text parts that are not ``thought`` parts."""
    _cand, parts = _candidate(response)
    return "".join(p.text for p in parts
                   if getattr(p, "text", None) and not getattr(p, "thought", False))


def _finish_reason(cand: Any) -> Optional[str]:
    reason = getattr(cand, "finish_reason", None) if cand is not None else None
    if reason is None:
        return None
    name = (getattr(reason, "name", None) or str(reason)).rsplit(".", 1)[-1].upper()
    return {"STOP": "stop", "MAX_TOKENS": "length"}.get(name, name.lower())


def _usage(response: Any) -> Optional[dict]:
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return None
    return {
        "prompt_tokens": int(getattr(um, "prompt_token_count", 0) or 0),
        "completion_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
        "thought_tokens": int(getattr(um, "thoughts_token_count", 0) or 0),
    }


def parse_logprobs_result(result: Any) -> list[TokenLogprob]:
    """``candidates[0].logprobs_result`` -> ``list[TokenLogprob]``.

    ``chosen_candidates`` are the generated tokens; ``top_candidates[i]`` the
    top-k alternatives at step *i*. A missing result (the model returns none)
    is an empty list, which the caller must read as "no logprobs", not as
    "an empty answer".
    """
    if result is None:
        return []
    def probability(candidate: Any) -> float:
        value = getattr(candidate, "log_probability", None)
        if value is None or not math.isfinite(float(value)) or float(value) > 0:
            raise ValueError("Gemini returned a missing or invalid token log probability")
        return float(value)

    chosen = getattr(result, "chosen_candidates", None) or []
    tops = getattr(result, "top_candidates", None) or []
    out: list[TokenLogprob] = []
    for i, c in enumerate(chosen):
        top: dict[str, float] = {}
        if i < len(tops):
            for t in getattr(tops[i], "candidates", None) or []:
                top[getattr(t, "token", "") or ""] = probability(t)
        out.append(TokenLogprob(token=getattr(c, "token", "") or "",
                                logprob=probability(c), top=top))
    return out


# ----------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------
def _generate_fn(rt: _Runtime) -> Callable[..., str]:
    def _fn(prompt: str, model: str = "", *, image: Any = None, audio: Any = None, **kw) -> str:
        _client, types = rt.genai()
        return answer_text(rt.request(model, _parts(types, prompt, image, audio), kw))

    _fn.state = rt.state  # type: ignore[attr-defined]
    return _fn


def _chat_fn(rt: _Runtime) -> Callable[..., ChatTurn]:
    def _fn(messages: list, tools=None, model: str = "") -> ChatTurn:
        _client, types = rt.genai()
        system, contents = _to_genai_contents(messages, types)
        response = rt.request(model, contents, {}, system=system, tools=_to_genai_tools(tools, types))
        cand, parts = _candidate(response)
        texts: list[str] = []
        calls: list[dict] = []
        for part in parts:
            if getattr(part, "text", None) and not getattr(part, "thought", False):
                texts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                calls.append({"id": getattr(fc, "id", None),
                              "function": {"name": fc.name, "arguments": json.dumps(dict(fc.args or {}))}})
        return ChatTurn(text="".join(texts), raw_tool_calls=calls or None,
                        finish_reason=_finish_reason(cand), usage=_usage(response))

    _fn.state = rt.state  # type: ignore[attr-defined]
    return _fn


def _logprobs_fn(rt: _Runtime) -> Callable[..., list]:
    def _fn(prompt: str, model: str = "", *, image: Any = None, audio: Any = None, **kw) -> list:
        _client, types = rt.genai()
        response = rt.request(model, _parts(types, prompt, image, audio), kw, logprobs=True)
        cand, _parts_ = _candidate(response)
        tokens = parse_logprobs_result(getattr(cand, "logprobs_result", None) if cand is not None else None)
        if not tokens:
            raise RuntimeError(f"{model} returned no token logprobs; probability analysis is unavailable")
        return tokens

    _fn.state = rt.state  # type: ignore[attr-defined]
    return _fn


def _probe_logprobs(rt: _Runtime, model: str) -> bool:
    """Verify actual token probabilities, not just SDK parameter availability.

    Only explicit unsupported-feature errors are downgraded. Authentication,
    quota, transport, and unknown errors must remain visible to the caller.
    """
    evidence = {"model": model, "status": "probing", "top_k": rt.top_logprobs}
    rt.state["logprobs_probe"] = evidence
    try:
        response = rt.request(model, "Reply with exactly OK.",
                              {"max_output_tokens": 32, "temperature": 0.0}, logprobs=True)
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if (getattr(exc, "code", None) == 400 and "logprob" in lowered
                and any(s in lowered for s in ("not supported", "unsupported", "not available", "not enabled"))):
            evidence.update(status="unsupported", reason=message)
            return False
        evidence.update(status="error", error_type=type(exc).__name__)
        raise
    cand, _ = _candidate(response)
    result = getattr(cand, "logprobs_result", None) if cand is not None else None
    chosen = getattr(result, "chosen_candidates", None) or []
    valid = bool(chosen) and all(
        getattr(c, "log_probability", None) is not None
        and math.isfinite(float(c.log_probability)) and float(c.log_probability) <= 0
        for c in chosen
    )
    evidence.update(status="available" if valid else "not_returned",
                    model_version=getattr(response, "model_version", None),
                    token_count=len(chosen), finish_reason=_finish_reason(cand))
    if valid:
        evidence["tokens"] = [vars(t) for t in parse_logprobs_result(result)]
    return valid


def _runtime(*, api_key, client, timeout, retries, thinking, top_logprobs, sampling) -> _Runtime:
    sampling = dict(sampling)
    sampling.setdefault("temperature", 0.0)
    return _Runtime(api_key=api_key, client=client, timeout=timeout, retries=retries,
                    policy=thinking or ThinkingPolicy(), top_logprobs=top_logprobs, sampling=sampling)


def gemini_generate_fn(
    *,
    api_key: Optional[str] = None,
    client: Any = None,
    timeout: float = 300.0,
    retries: int = 5,
    thinking: Optional[ThinkingPolicy] = None,
    **sampling: Any,
) -> Callable[..., str]:
    """Build ``generate_fn(prompt, model=..., image=, audio=) -> str``."""
    return _generate_fn(_runtime(api_key=api_key, client=client, timeout=timeout, retries=retries,
                                 thinking=thinking, top_logprobs=5, sampling=sampling))


def gemini_chat_fn(
    *,
    api_key: Optional[str] = None,
    client: Any = None,
    timeout: float = 300.0,
    retries: int = 5,
    thinking: Optional[ThinkingPolicy] = None,
    **sampling: Any,
) -> Callable[..., ChatTurn]:
    """Build a tool-aware ``chat_fn(messages, tools, model) -> ChatTurn`` (native function calling)."""
    return _chat_fn(_runtime(api_key=api_key, client=client, timeout=timeout, retries=retries,
                             thinking=thinking, top_logprobs=5, sampling=sampling))


def gemini_logprobs_fn(
    *,
    api_key: Optional[str] = None,
    client: Any = None,
    timeout: float = 300.0,
    retries: int = 5,
    thinking: Optional[ThinkingPolicy] = None,
    top_logprobs: int = 5,
    **sampling: Any,
) -> Callable[..., list]:
    """Build ``logprobs_fn(prompt, model=...) -> list[TokenLogprob]`` (``response_logprobs``)."""
    return _logprobs_fn(_runtime(api_key=api_key, client=client, timeout=timeout, retries=retries,
                                 thinking=thinking, top_logprobs=top_logprobs, sampling=sampling))


def gemini_runtime(
    *,
    api_key: Optional[str] = None,
    client: Any = None,
    timeout: float = 300.0,
    retries: int = 5,
    thinking: Optional[ThinkingPolicy] = None,
    with_logprobs: bool | str = False,
    logprobs_model: Optional[str] = None,
    top_logprobs: int = 5,
    **sampling: Any,
) -> RuntimeConfig:
    """A ready :class:`RuntimeConfig` for ``compose(key, "api", runtime=...)``.

    The three functions share one client and one ``state`` dict
    (``model_version`` of the last reply, call / retry counts, models that
    fell back to the default thinking config). ``with_logprobs`` is False by
    default. Pass ``with_logprobs="auto", logprobs_model=model_id`` to
    probe the endpoint and expose LOGPROBS only when token probabilities
    really arrive. True with a model id requires a successful probe.
    Probing never enables LOGITS: top-k log probabilities are not raw logits.
    """
    rt = _runtime(api_key=api_key, client=client, timeout=timeout, retries=retries,
                  thinking=thinking, top_logprobs=top_logprobs, sampling=sampling)
    if with_logprobs not in (True, False, "auto"):
        raise ValueError("with_logprobs must be True, False, or 'auto'")
    if not 0 <= top_logprobs <= 20:
        raise ValueError("top_logprobs must be between 0 and 20")
    if with_logprobs == "auto" and not logprobs_model:
        raise ValueError("logprobs_model is required for automatic capability probing")
    enabled = bool(with_logprobs)
    rt.state["logits"] = {"status": "unavailable", "reason": "Gemini API does not expose raw logits"}
    if with_logprobs and logprobs_model:
        enabled = _probe_logprobs(rt, logprobs_model)
        if not enabled and with_logprobs is True:
            raise RuntimeError(f"{logprobs_model} did not provide usable logprobs: "
                               f"{rt.state['logprobs_probe']}")
    elif not enabled:
        rt.state["logprobs_probe"] = {"status": "disabled"}
    return RuntimeConfig(
        chat_fn=_chat_fn(rt),
        generate_fn=_generate_fn(rt),
        logprobs_fn=_logprobs_fn(rt) if enabled else None,
    )


def runtime_state(model: Any) -> dict:
    """The shared ``state`` behind an APIModel built by :func:`gemini_runtime` (else ``{}``)."""
    fn = getattr(getattr(model, "runtime", None), "generate_fn", None)
    return dict(getattr(fn, "state", None) or {})
