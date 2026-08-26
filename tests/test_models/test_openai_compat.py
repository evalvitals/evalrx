"""OpenAI-compatible client factories + the batch runner."""

from __future__ import annotations

import warnings

import pytest

from evalvitals.core.capability import Capability
from evalvitals.core.case import FailureCase, Inputs
from evalvitals.core.model import Model
from evalvitals.core.tool import ChatTurn, Tool
from evalvitals.models import RuntimeConfig, compose
from evalvitals.models.agent import run_batch
from evalvitals.models.backends.openai_compat import (
    openai_chat_fn,
    to_openai_messages,
)


class _FakeImg:
    mode = "RGB"

    def save(self, buf, format=None):
        buf.write(b"\x89PNGfake")


# ----------------------------------------------------------------------
# Message conversion
# ----------------------------------------------------------------------
def test_image_blocks_become_data_urls():
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _FakeImg()},
                {"type": "text", "text": "what is this?"},
            ],
        },
    ]
    out = to_openai_messages(messages)
    assert out[0] == {"role": "system", "content": "sys"}
    img_block, text_block = out[1]["content"]
    assert img_block["type"] == "image_url"
    assert img_block["image_url"]["url"].startswith("data:image/png;base64,")
    assert text_block == {"type": "text", "text": "what is this?"}


def test_http_image_urls_pass_through():
    messages = [{"role": "user", "content": [{"type": "image", "image": "https://x/y.png"}]}]
    out = to_openai_messages(messages)
    assert out[0]["content"][0]["image_url"]["url"] == "https://x/y.png"


def test_tool_and_assistant_messages_pass_through():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "5"},
    ]
    assert to_openai_messages(messages) == messages


# ----------------------------------------------------------------------
# chat_fn against a fake client
# ----------------------------------------------------------------------
class _FakeToolCall:
    def model_dump(self):
        return {"id": "c1", "function": {"name": "add", "arguments": '{"a": 1}'}}


class _FakeClient:
    def __init__(self, content="hi", tool_calls=None):
        self.last_kwargs = None
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.last_kwargs = kwargs

                class _Msg:
                    pass

                msg = _Msg()
                msg.content = content
                msg.tool_calls = tool_calls

                class _Choice:
                    pass

                choice = _Choice()
                choice.message = msg
                choice.finish_reason = "stop"

                class _Resp:
                    pass

                resp = _Resp()
                resp.choices = [choice]
                return resp

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_chat_fn_parses_native_tool_calls_and_defaults_temperature_zero():
    client = _FakeClient(content=None, tool_calls=[_FakeToolCall()])
    fn = openai_chat_fn(client=client)
    turn = fn([{"role": "user", "content": "add"}], tools=[{"type": "function"}], model="m")
    assert turn.raw_tool_calls == [{"id": "c1", "function": {"name": "add", "arguments": '{"a": 1}'}}]
    assert turn.text == ""
    assert client.last_kwargs["temperature"] == 0.0
    assert client.last_kwargs["tool_choice"] == "auto"
    assert client.last_kwargs["model"] == "m"


def test_chat_fn_plain_answer():
    fn = openai_chat_fn(client=_FakeClient(content="done"))
    turn = fn([{"role": "user", "content": "q"}], tools=None, model="m")
    assert turn.text == "done" and turn.raw_tool_calls is None


# ----------------------------------------------------------------------
# run_batch
# ----------------------------------------------------------------------
def _echo_tool(tag):
    return Tool(
        name="echo",
        description="echo",
        parameters={"type": "object", "properties": {}},
        fn=lambda: tag,
    )


def test_run_batch_on_api_handle_keeps_order_and_binds_tools_per_case():
    def chat_fn(messages, tools=None, model=""):
        return ChatTurn(text="ok")

    handle = compose("qwen3-8b", "api", RuntimeConfig(chat_fn=chat_fn))
    cases = [FailureCase(inputs=Inputs(prompt=f"q{i}")) for i in range(3)]
    seen: list[str] = []

    def factory(case):
        seen.append(case.inputs.prompt)
        return [_echo_tool(case.inputs.prompt)]

    trajs = run_batch(handle, cases, tools_factory=factory, concurrency=3)
    assert [t.goal for t in trajs] == ["q0", "q1", "q2"]
    assert sorted(seen) == ["q0", "q1", "q2"]
    assert all(t.metrics["terminated"] == "final" for t in trajs)


def test_run_batch_error_yields_stub_trajectory():
    class Boom(Model):
        capabilities = frozenset({Capability.GENERATE, Capability.TOOL_CALLS})

        def generate(self, inputs, **kw):
            return ""

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

        def chat(self, messages, tools=None):
            raise RuntimeError("endpoint down")

    trajs = run_batch(Boom(), ["a", "b"], tools_factory=lambda c: [], concurrency=1)
    assert len(trajs) == 2
    assert all(t.metrics["terminated"] == "error" for t in trajs)
    assert "endpoint down" in trajs[0].metrics["error"]


def test_run_batch_forces_sequential_for_local_handles():
    class Local(Model):
        capabilities = frozenset({Capability.GENERATE, Capability.TOOL_CALLS})

        def generate(self, inputs, **kw):
            return ""

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

        def chat(self, messages, tools=None):
            return ChatTurn(text="ok")

    with pytest.warns(UserWarning, match="forcing"):
        trajs = run_batch(Local(), ["x"], tools_factory=lambda c: [], concurrency=8)
    assert trajs[0].final_answer == "ok"


def test_run_batch_no_warning_when_sequential():
    class Local(Model):
        capabilities = frozenset({Capability.GENERATE, Capability.TOOL_CALLS})

        def generate(self, inputs, **kw):
            return ""

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

        def chat(self, messages, tools=None):
            return ChatTurn(text="ok")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_batch(Local(), ["x"], tools_factory=lambda c: [], concurrency=1)


# ----------------------------------------------------------------------
# logprobs: what makes a served model claim Capability.LOGPROBS
# ----------------------------------------------------------------------

class _FakeLogprobClient:
    """Records the request and replies in vLLM's OpenAI-compatible shape."""

    def __init__(self, content=None):
        self.seen = {}
        self._content = content if content is not None else [
            {"token": "apple", "logprob": -0.005,
             "top_logprobs": [{"token": "apple", "logprob": -0.005},
                              {"token": "Apple", "logprob": -5.3}]},
            {"token": " fig", "logprob": -0.012, "top_logprobs": []},
        ]
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.seen = kw
                return type("R", (), {"choices": [type("C", (), {
                    "logprobs": type("L", (), {"content": outer._content})(),
                })()]})()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_a_served_model_can_report_token_logprobs():
    """`vllm serve` returns them; the endpoint path was not asking.

    The api backend claims Capability.LOGPROBS only when a `logprobs_fn` is
    wired, so without one every analyzer reading answer-token uncertainty is
    skipped as unsupported -- `calibration` and `logprob_entropy` on the LLM
    benchmark set. They were dropped because nobody asked the server, not
    because it could not answer.
    """
    from evalvitals.models.backends.openai_compat import openai_logprobs_fn

    client = _FakeLogprobClient()
    fn = openai_logprobs_fn(client=client, temperature=0.0, max_tokens=16)
    out = fn("Sort: pear apple fig", model="qwen3.5-2b")

    assert client.seen["logprobs"] is True
    assert client.seen["top_logprobs"] == 5
    assert client.seen["model"] == "qwen3.5-2b"
    assert [t.token for t in out] == ["apple", " fig"]
    assert out[0].logprob == pytest.approx(-0.005)
    assert out[0].top["Apple"] == pytest.approx(-5.3)


def test_wiring_logprobs_is_what_grants_the_capability():
    """And opting out must actually withhold it: claiming LOGPROBS against an
    endpoint that rejects the parameter turns a skipped analyzer into a failing
    one, which is the worse outcome."""
    from evalvitals.models.backends.openai_compat import openai_runtime

    with_lp = openai_runtime(client=_FakeLogprobClient(), base_url="http://127.0.0.1:8020/v1")
    without = openai_runtime(
        client=_FakeLogprobClient(), base_url="http://127.0.0.1:8020/v1", with_logprobs=False
    )
    assert with_lp.logprobs_fn is not None
    assert without.logprobs_fn is None

    spec_key = "qwen3.5-2b"
    served = compose(spec_key, "api", with_lp, set())
    blind = compose(spec_key, "api", without, set())
    assert Capability.LOGPROBS in served.capabilities
    assert Capability.LOGPROBS not in blind.capabilities


def test_an_endpoint_that_returns_no_logprobs_yields_no_tokens():
    """Not an exception, and not a fabricated zero -- an empty list, which is
    what "this server answered without them" honestly looks like."""
    from evalvitals.models.backends.openai_compat import openai_logprobs_fn

    fn = openai_logprobs_fn(client=_FakeLogprobClient(content=[]))
    assert fn("hi", model="m") == []


# ----------------------------------------------------------------------
# Audio blocks + media slots on the generate / logprobs paths
# ----------------------------------------------------------------------
def _wav_bytes(n=16, sr=16000):
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x01" * n)
    return buf.getvalue()


def test_audio_path_becomes_an_input_audio_block(tmp_path):
    import base64

    wav = tmp_path / "clip.wav"
    wav.write_bytes(_wav_bytes())
    out = to_openai_messages([{"role": "user", "content": [
        {"type": "audio", "audio": str(wav)}, {"type": "text", "text": "what sound?"}]}])
    audio_block, text_block = out[0]["content"]
    assert audio_block["type"] == "input_audio"
    assert audio_block["input_audio"]["format"] == "wav"
    assert base64.b64decode(audio_block["input_audio"]["data"]) == _wav_bytes()
    assert text_block == {"type": "text", "text": "what sound?"}


def test_waveform_arrays_are_encoded_as_pcm16_wav():
    import base64
    import io
    import wave

    import numpy as np

    from evalvitals.models.backends.openai_compat import _to_input_audio

    enc = _to_input_audio((np.array([0.0, 0.5, -0.5], dtype=np.float32), 8000))
    assert enc["format"] == "wav"
    with wave.open(io.BytesIO(base64.b64decode(enc["data"])), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 8000, 3)


def test_generate_fn_sends_the_media_slots_as_content_blocks(tmp_path):
    from evalvitals.models.backends.openai_compat import openai_generate_fn

    wav = tmp_path / "clip.wav"
    wav.write_bytes(_wav_bytes())
    client = _FakeClient(content="a dog")
    fn = openai_generate_fn(client=client)
    assert fn("what?", model="m", audio=str(wav)) == "a dog"
    (msg,) = client.last_kwargs["messages"]
    assert [b["type"] for b in msg["content"]] == ["input_audio", "text"]
    # a text-only call is still a plain string turn (nothing changes for LLMs)
    fn("hi", model="m")
    assert client.last_kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_api_model_forwards_only_the_slots_the_spec_declares():
    from evalvitals.core.spec import AudioSpec, ModelSpec

    seen = {}

    def gen(prompt, model="", **kw):
        seen.update(kw)
        return "ok"

    audio_spec = ModelSpec(key="t-alm", family="x", model_type="x", hf_repo="x/alm",
                           auto_class="AutoModelForCausalLM",
                           audio=AudioSpec(audio_token_id_attr="a", audio_tower="t"))
    m = compose(audio_spec, "api", RuntimeConfig(generate_fn=gen), set())
    m.generate(Inputs(prompt="p", audio="clip.wav", image="img.png"))
    assert seen == {"audio": "clip.wav"}          # image: not a declared modality

    seen.clear()
    text_spec = ModelSpec(key="t-llm", family="x", model_type="x", hf_repo="x/llm",
                          auto_class="AutoModelForCausalLM")
    m = compose(text_spec, "api", RuntimeConfig(generate_fn=gen), set())
    m.generate(Inputs(prompt="p", audio="clip.wav"))
    assert seen == {}                              # text-only spec never forwards media
    m.generate("bare string prompt")
    assert seen == {}
