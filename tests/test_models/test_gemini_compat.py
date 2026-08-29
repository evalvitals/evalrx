"""gemini_compat: the Gemini API as an ``api``-backend runtime, on a fake google-genai client.

The real SDK is optional (absent on CI), so every test injects a fake
``client`` whose ``models.generate_content`` records the request and replies
in the SDK's shape. What is pinned here is the behaviour a diagnosis run
depends on, not the SDK: thinking sent at each model's floor, thought parts
kept out of the answer, OpenAI-named decoding kwargs translated, media only
when a case carries it, retries on 429, logprobs claimed only when wired.
"""

from __future__ import annotations

import base64
import io
import threading
import wave
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from evalvitals.core.capability import Capability
from evalvitals.core.case import Inputs
from evalvitals.models.backends import gemini_compat as gc
from evalvitals.models.backends.gemini_compat import (
    ThinkingPolicy,
    answer_text,
    gemini_runtime,
    parse_logprobs_result,
    thinking_config,
    translate_sampling,
)


# -- a minimal fake of google.genai.types ------------------------------
@dataclass
class Part:
    text: Optional[str] = None
    thought: bool = False
    function_call: Any = None
    inline: Any = None

    @classmethod
    def from_bytes(cls, *, data: bytes = b"", mime_type: str = ""):
        return cls(inline=(mime_type, data))


@dataclass
class Content:
    role: str = "user"
    parts: list = field(default_factory=list)


@dataclass
class ThinkingConfig:
    thinking_level: Optional[str] = None
    thinking_budget: Optional[int] = None
    include_thoughts: Optional[bool] = None


@dataclass
class GenerateContentConfig:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None
    seed: Optional[int] = None
    candidate_count: Optional[int] = None
    stop_sequences: Optional[list] = None
    response_logprobs: Optional[bool] = None
    logprobs: Optional[int] = None
    thinking_config: Any = None
    system_instruction: Any = None
    tools: Any = None


@dataclass
class FunctionDeclaration:
    name: str = ""
    description: str = ""
    parameters_json_schema: Any = None
    parameters: Any = None


@dataclass
class GTool:
    function_declarations: list = field(default_factory=list)


@dataclass
class FunctionCall:
    name: str = ""
    args: dict = field(default_factory=dict)
    id: Optional[str] = None


@dataclass
class FunctionResponse:
    name: str = ""
    response: dict = field(default_factory=dict)


class FakeTypes:
    Part = Part
    Content = Content
    ThinkingConfig = ThinkingConfig
    GenerateContentConfig = GenerateContentConfig
    FunctionDeclaration = FunctionDeclaration
    Tool = GTool
    FunctionCall = FunctionCall
    FunctionResponse = FunctionResponse


class _Finish:
    def __init__(self, name):
        self.name = name


@dataclass
class Candidate:
    content: Content
    finish_reason: Any = None
    logprobs_result: Any = None


@dataclass
class Usage:
    prompt_token_count: int = 10
    candidates_token_count: int = 4
    thoughts_token_count: int = 0


@dataclass
class Response:
    candidates: list
    model_version: str = "gemini-3.6-flash-001"
    usage_metadata: Any = field(default_factory=Usage)
    prompt_feedback: Any = None


class APIError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class FakeClient:
    """Replies with the queued responses (or raises the queued exceptions), recording requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        outer = self

        class _Models:
            def generate_content(self, *, model, contents, config=None):
                outer.requests.append({"model": model, "contents": contents, "config": config})
                reply = outer.replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return reply

        self.models = _Models()


def _reply(*parts, finish="STOP", logprobs=None):
    return Response(candidates=[Candidate(content=Content(role="model", parts=list(parts)),
                                          finish_reason=_Finish(finish), logprobs_result=logprobs)])


def _runtime(client, **kw):
    """A gemini_runtime on the fake client; the fake types replace the SDK import."""
    rt = gemini_runtime(client=client, api_key="k", **kw)
    for fn in (rt.chat_fn, rt.generate_fn, rt.logprobs_fn):
        if fn is not None:
            fn.__closure__  # noqa: B018 - closures exist
    return rt


@pytest.fixture(autouse=True)
def _fake_sdk(monkeypatch):
    """``_Runtime.genai()`` returns the injected client with the fake types (no SDK import)."""
    def genai(self):
        return self._client, FakeTypes
    monkeypatch.setattr(gc._Runtime, "genai", genai)
    monkeypatch.setattr(gc.time, "sleep", lambda s: None)


def test_sdk_request_has_a_wall_clock_deadline():
    """A wedged audio upload must not strand a whole M1 batch forever."""
    class SlowModels:
        def generate_content(self, **kwargs):
            threading.Event().wait(0.2)
            return _reply(Part(text="late"))

    class SlowClient:
        models = SlowModels()

    rt = _runtime(SlowClient(), timeout=0.02, retries=0)
    with pytest.raises(TimeoutError, match="wall-clock timeout"):
        rt.generate_fn("q", model="gemini-3.7-flash")


# ----------------------------------------------------------------------
# Thinking policy
# ----------------------------------------------------------------------
def test_thinking_floor_per_model_family():
    assert thinking_config("gemini-3.7-flash") == {"thinking_level": "low"}
    assert thinking_config("gemini-3.6-flash") == {"thinking_level": "minimal"}
    assert thinking_config("gemini-3.5-flash-lite") == {"thinking_level": "minimal"}
    assert thinking_config("gemini-3.1-flash-lite") == {"thinking_level": "minimal"}
    assert thinking_config("gemini-2.5-flash") == {"thinking_budget": 0}
    assert thinking_config("gemini-2.5-flash-lite") == {"thinking_budget": 0}
    assert thinking_config("gemini-2.5-pro") == {"thinking_budget": 128}
    # dated / prefixed ids resolve to their base model
    assert thinking_config("models/gemini-2.5-flash-lite-preview-09-2025") == {"thinking_budget": 0}
    # unknown model: nothing is sent (the API default), never a guess
    assert thinking_config("gemini-9-ultra") is None


def test_explicit_level_is_honoured_but_never_below_the_floor():
    assert thinking_config("gemini-3.6-flash", ThinkingPolicy(level="high")) == {"thinking_level": "high"}
    # 3.7-flash has no minimal: the floor is sent instead
    assert thinking_config("gemini-3.7-flash", ThinkingPolicy(level="minimal")) == {"thinking_level": "low"}
    # a level asked of a budget model maps to a budget, floored
    assert thinking_config("gemini-2.5-flash", ThinkingPolicy(level="medium")) == {"thinking_budget": 8192}
    assert thinking_config("gemini-2.5-pro", ThinkingPolicy(level="minimal")) == {"thinking_budget": 128}
    assert thinking_config("gemini-2.5-flash", ThinkingPolicy(budget=512)) == {"thinking_budget": 512}
    with pytest.raises(ValueError, match="unknown thinking level"):
        thinking_config("gemini-3.6-flash", ThinkingPolicy(level="max"))


def test_enable_thinking_leaves_the_api_default():
    assert thinking_config("gemini-3.6-flash", ThinkingPolicy(floor=False)) is None
    assert thinking_config("gemini-2.5-flash", ThinkingPolicy(floor=False)) is None


# ----------------------------------------------------------------------
# Request shaping
# ----------------------------------------------------------------------
def test_openai_named_decoding_kwargs_are_translated_and_the_rest_dropped():
    out = translate_sampling({
        "max_tokens": 2048, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "n": 5,
        "stop": "Answer:", "do_sample": True, "extra_body": {"x": 1}, "logprobs": True,
        "top_logprobs": 5, "seed": 3, "mystery": 1,
    })
    assert out == {"max_output_tokens": 2048, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                   "candidate_count": 5, "stop_sequences": ["Answer:"], "seed": 3}


def test_generate_sends_the_floor_and_the_sampling_and_reads_only_answer_parts():
    client = FakeClient([_reply(Part(text="hmm", thought=True), Part(text="Answer: B"))])
    rt = _runtime(client, temperature=0.6, top_p=0.95, top_k=20, max_output_tokens=2048)

    out = rt.generate_fn("Q?", model="gemini-3.6-flash")

    assert out == "Answer: B"                      # the thought part is not answer text
    req = client.requests[0]
    assert req["model"] == "gemini-3.6-flash"
    cfg = req["config"]
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.max_output_tokens) == (0.6, 0.95, 20, 2048)
    assert cfg.thinking_config == ThinkingConfig(thinking_level="minimal")
    assert cfg.response_logprobs is None          # not asked for
    # one text part, no media
    assert [p.text for p in req["contents"]] == ["Q?"]
    assert rt.generate_fn.state["model_version"] == "gemini-3.6-flash-001"
    assert rt.generate_fn.state["calls"] == 1


def test_thinking_that_spends_output_tokens_gets_headroom_on_the_cap():
    """Measured: thought tokens count against max_output_tokens (3.6-flash at a
    16-token cap with level=low truncates; 3.7-flash returns an empty answer).
    A config that lets the model think therefore raises the request cap; one
    that turns thinking off (minimal / budget 0) must NOT touch it."""
    from evalvitals.models.backends.gemini_compat import (
        THINKING_OUTPUT_HEADROOM,
        thinking_spends_output_tokens,
    )

    assert not thinking_spends_output_tokens({"thinking_level": "minimal"})
    assert not thinking_spends_output_tokens({"thinking_budget": 0})
    assert not thinking_spends_output_tokens(None)
    assert thinking_spends_output_tokens({"thinking_level": "low"})
    assert thinking_spends_output_tokens({"thinking_budget": 128})

    # 3.6-flash floors at minimal -> the 64-token cap goes through untouched
    client = FakeClient([_reply(Part(text="A"))])
    _runtime(client, max_output_tokens=64).generate_fn("Q?", model="gemini-3.6-flash")
    assert client.requests[0]["config"].max_output_tokens == 64

    # 3.7-flash floors at low -> the cap gains the thought headroom (logged once)
    client = FakeClient([_reply(Part(text="A")), _reply(Part(text="B"))])
    rt = _runtime(client, max_output_tokens=64)
    rt.generate_fn("Q?", model="gemini-3.7-flash")
    assert client.requests[0]["config"].max_output_tokens == 64 + THINKING_OUTPUT_HEADROOM
    assert rt.generate_fn.state["headroom_for"] == ["gemini-3.7-flash"]
    # ...including on a per-call L0 override of the cap
    rt.generate_fn("Q?", model="gemini-3.7-flash", max_tokens=128)
    assert client.requests[1]["config"].max_output_tokens == 128 + THINKING_OUTPUT_HEADROOM

    # 2.5-pro cannot switch thinking off (budget floor 128) -> headroom too
    client = FakeClient([_reply(Part(text="A"))])
    _runtime(client, max_output_tokens=64).generate_fn("Q?", model="gemini-2.5-pro")
    assert client.requests[0]["config"].max_output_tokens == 64 + THINKING_OUTPUT_HEADROOM


def test_per_call_kwargs_override_the_baseline_decoding_under_openai_names():
    """The fix stage's L0 candidates re-issue ``max_tokens`` / ``temperature``."""
    client = FakeClient([_reply(Part(text="x"))])
    rt = _runtime(client, temperature=0.6, max_output_tokens=2048)
    rt.generate_fn("Q?", model="gemini-2.5-flash-lite", max_tokens=8192, temperature=0.0)
    cfg = client.requests[0]["config"]
    assert (cfg.max_output_tokens, cfg.temperature) == (8192, 0.0)
    assert cfg.thinking_config == ThinkingConfig(thinking_budget=0)


def test_media_is_sent_only_when_the_case_carries_it():
    class _Img:
        mode = "RGB"

        def save(self, buf, format=None):
            buf.write(b"PNGBYTES")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16)
    wav = buf.getvalue()

    client = FakeClient([_reply(Part(text="a")), _reply(Part(text="b")), _reply(Part(text="c"))])
    rt = _runtime(client)
    rt.generate_fn("see", model="gemini-3.6-flash", image=_Img())
    rt.generate_fn("hear", model="gemini-3.6-flash", audio=wav)
    rt.generate_fn("read", model="gemini-3.6-flash")

    img_parts = client.requests[0]["contents"]
    assert img_parts[0].inline == ("image/png", b"PNGBYTES") and img_parts[1].text == "see"
    aud_parts = client.requests[1]["contents"]
    assert aud_parts[0].inline[0] == "audio/wav" and aud_parts[0].inline[1] == wav
    assert aud_parts[1].text == "hear"
    assert [p.text for p in client.requests[2]["contents"]] == ["read"]


def test_api_model_forwards_only_the_modalities_the_spec_declares():
    """Through compose(): a Gemini spec declares image + audio, so a VLM/ALM case
    reaches the API with its media (the earlier api backend answered from the
    prompt alone)."""
    from evalvitals.models.compose import compose

    client = FakeClient([_reply(Part(text="Answer: B"))])
    model = compose("gemini-3.5-flash-lite", "api", _runtime(client))

    assert model.capabilities == frozenset({Capability.GENERATE, Capability.TOOL_CALLS})
    assert {"image", "audio", "text", "video"} <= set(model.modalities)
    model.generate(Inputs(prompt="which?", audio=b"RIFFfake"))
    parts = client.requests[0]["contents"]
    assert parts[0].inline[0] == "audio/wav" and parts[1].text == "which?"
    assert client.requests[0]["model"] == "gemini-3.5-flash-lite"


# ----------------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------------
def test_rate_limits_are_retried_with_backoff_then_succeed():
    client = FakeClient([APIError(429, "RESOURCE_EXHAUSTED: quota"), APIError(503, "UNAVAILABLE"),
                         _reply(Part(text="ok"))])
    rt = _runtime(client, retries=3)
    assert rt.generate_fn("Q?", model="gemini-3.6-flash") == "ok"
    assert len(client.requests) == 3
    assert rt.generate_fn.state["retries"] == 2


def test_a_non_retryable_error_propagates_and_retries_are_bounded():
    client = FakeClient([APIError(400, "INVALID_ARGUMENT: bad request")])
    rt = _runtime(client, retries=3)
    with pytest.raises(APIError, match="bad request"):
        rt.generate_fn("Q?", model="gemini-3.6-flash")
    assert len(client.requests) == 1

    client = FakeClient([APIError(429, "quota")] * 3)
    rt = _runtime(client, retries=2)
    with pytest.raises(APIError):
        rt.generate_fn("Q?", model="gemini-3.6-flash")
    assert len(client.requests) == 3


def test_a_rejected_thinking_config_falls_back_to_the_api_default_once_and_is_recorded():
    client = FakeClient([APIError(400, "INVALID_ARGUMENT: thinking_level is not supported for this model"),
                         _reply(Part(text="ok")), _reply(Part(text="again"))])
    rt = _runtime(client)
    assert rt.generate_fn("Q?", model="gemini-3.1-flash-lite") == "ok"
    assert client.requests[0]["config"].thinking_config == ThinkingConfig(thinking_level="minimal")
    assert client.requests[1]["config"].thinking_config is None
    # remembered per model: the next call does not pay the rejected request again
    rt.generate_fn("Q2?", model="gemini-3.1-flash-lite")
    assert client.requests[2]["config"].thinking_config is None
    assert rt.generate_fn.state["thinking_fallback"] == ["gemini-3.1-flash-lite"]


def test_no_candidate_is_an_empty_answer_not_a_crash():
    resp = Response(candidates=[])
    assert answer_text(resp) == ""


# ----------------------------------------------------------------------
# Logprobs (opt-in) and chat
# ----------------------------------------------------------------------
def test_logprobs_are_not_claimed_unless_wired():
    rt = gemini_runtime(client=FakeClient([]), api_key="k")
    assert rt.logprobs_fn is None
    rt = gemini_runtime(client=FakeClient([]), api_key="k", with_logprobs=True)
    assert rt.logprobs_fn is not None


def test_logprobs_fn_asks_for_response_logprobs_and_parses_the_result():
    @dataclass
    class Cand:
        token: str
        log_probability: float
        token_id: int = 0

    @dataclass
    class Top:
        candidates: list

    @dataclass
    class LPResult:
        chosen_candidates: list
        top_candidates: list

    result = LPResult(chosen_candidates=[Cand("B", -0.1), Cand(".", -0.5)],
                      top_candidates=[Top([Cand("B", -0.1), Cand("C", -2.4)]), Top([])])
    client = FakeClient([_reply(Part(text="B."), logprobs=result)])
    rt = _runtime(client, with_logprobs=True, top_logprobs=3)
    toks = rt.logprobs_fn("Q?", model="gemini-2.5-flash")
    cfg = client.requests[0]["config"]
    assert cfg.response_logprobs is True and cfg.logprobs == 3
    assert [t.token for t in toks] == ["B", "."]
    assert toks[0].logprob == pytest.approx(-0.1) and toks[0].top["C"] == pytest.approx(-2.4)
    assert toks[1].top == {}
    # a model that returns no logprobs_result yields [] -- "no logprobs", not "no tokens"
    assert parse_logprobs_result(None) == []


def test_chat_fn_uses_native_function_calling_and_reports_finish_reason():
    client = FakeClient([_reply(Part(text="zooming "),
                                Part(function_call=FunctionCall(name="zoom", args={"bbox": [0, 0, 1, 1]})),
                                finish="MAX_TOKENS")])
    rt = _runtime(client)
    turn = rt.chat_fn([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
                      tools=[{"type": "function", "function": {"name": "zoom", "parameters": {}}}],
                      model="gemini-3.6-flash")
    assert turn.text == "zooming "
    assert turn.raw_tool_calls[0]["function"]["name"] == "zoom"
    assert turn.finish_reason == "length"
    assert turn.usage == {"prompt_tokens": 10, "completion_tokens": 4, "thought_tokens": 0}
    cfg = client.requests[0]["config"]
    assert cfg.system_instruction == "s" and cfg.tools[0].function_declarations[0].name == "zoom"
    assert cfg.thinking_config == ThinkingConfig(thinking_level="minimal")


def test_audio_blocks_in_chat_history_become_inline_audio_parts():
    from evalvitals.models.blackbox.gemini import _to_genai_contents

    _system, contents = _to_genai_contents(
        [{"role": "user", "content": [{"type": "audio", "audio": b"RIFFfake"}, {"type": "text", "text": "?"}]}],
        FakeTypes,
    )
    assert contents[0].parts[0].inline == ("audio/wav", b"RIFFfake")
    assert contents[0].parts[1].text == "?"


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
def test_gemini_specs_are_api_only_omni_and_named_by_model_id():
    from evalvitals.specs import get_spec

    for key in ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"):
        spec = get_spec(key)
        assert spec.api_only and spec.hf_repo == "" and spec.family == "gemini"
        assert spec.modalities == frozenset({"text", "image", "audio", "video"})
        assert any("L2" in c for c in spec.caveats)


def test_wav_encoding_helper_round_trips_a_waveform():
    from evalvitals.models.blackbox.gemini import _audio_bytes

    data, mime = _audio_bytes(([0.0] * 32, 16000))
    assert mime == "audio/wav"
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == 16000 and w.getnframes() == 32
    assert base64.b64encode(data)  # bytes, not a data URL


# ----------------------------------------------------------------------
# SDK log noise
# ----------------------------------------------------------------------
def test_afc_chatter_is_filtered_off_the_sdk_logger_but_real_warnings_pass():
    import logging

    from evalvitals.models.backends import gemini_compat as gc

    sdk_logger = logging.getLogger("google_genai.models")
    records = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    sdk_logger.addHandler(handler)
    sdk_logger.setLevel(logging.INFO)
    try:
        gemini_runtime(client=FakeClient([]), api_key="k")  # __init__ installs the filter
        gemini_runtime(client=FakeClient([]), api_key="k")  # idempotent: no duplicate
        assert sdk_logger.filters.count(gc._SDK_NOISE_FILTER) == 1
        sdk_logger.warning(
            "Direct use of automatic function calling (AFC) in Models.generate_content"
            " is not recommended. Instead, we recommend to use AFC in Chat.send_message."
        )
        sdk_logger.info("AFC is enabled with max remote calls: 10.")
        sdk_logger.info("AFC remote call 1 is done.")
        sdk_logger.warning("there are non-text parts in the response")
        assert [r.getMessage() for r in records] == ["there are non-text parts in the response"]
    finally:
        sdk_logger.removeHandler(handler)
        sdk_logger.removeFilter(gc._SDK_NOISE_FILTER)
        sdk_logger.setLevel(logging.NOTSET)
