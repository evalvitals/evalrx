"""Agent(wraps=handle) — backend-agnostic tool loop + codecs + conditional caps."""

from __future__ import annotations

import pytest

from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.core.case import Label, StepRole
from evalvitals.core.model import Model
from evalvitals.core.tool import ChatTurn, Tool, ToolCall
from evalvitals.models import RuntimeConfig, compose
from evalvitals.models.agent import Agent, ToolExecutor
from evalvitals.models.toolcodec import OpenAIToolCodec, QwenToolCodec, codec_for


def _add_tool() -> Tool:
    return Tool(
        name="add",
        description="add two numbers",
        parameters={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
        fn=lambda a, b: a + b,
    )


class FakeChatHandle(Model):
    """A scripted tool-capable handle (local-style: tool call embedded in text)."""

    def __init__(self, script, caps=frozenset({Capability.GENERATE, Capability.TOOL_CALLS})):
        self.capabilities = caps
        self._script = list(script)
        self._i = 0
        self.seen_messages: list = []  # snapshot of the history at each chat() call

    def generate(self, inputs, **kwargs) -> str:
        return "noop"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError

    def chat(self, messages, tools=None) -> ChatTurn:
        self.seen_messages.append([dict(m) for m in messages])
        turn = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return turn


# ----------------------------------------------------------------------
# Codecs
# ----------------------------------------------------------------------
def test_qwen_codec_parses_tool_call_from_text():
    turn = ChatTurn(text='<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 3}}\n</tool_call>')
    call = QwenToolCodec().decode(turn)
    assert call.name == "add" and call.args == {"a": 2, "b": 3}


def test_qwen_codec_returns_none_on_plain_text():
    assert QwenToolCodec().decode(ChatTurn(text="just a final answer")) is None


def test_qwen_codec_handles_nested_arguments():
    # nested JSON in arguments — a naive {...} regex would truncate at the first brace
    turn = ChatTurn(text='<tool_call>\n{"name": "q", "arguments": {"filter": {"city": "NYC", "n": 2}}}\n</tool_call>')
    call = QwenToolCodec().decode(turn)
    assert call.name == "q" and call.args == {"filter": {"city": "NYC", "n": 2}}


def test_qwen_codec_decodes_multiple_parallel_calls():
    turn = ChatTurn(text=(
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    ))
    calls = QwenToolCodec().decode_all(turn)
    assert [c.name for c in calls] == ["a", "b"]


def test_qwen_codec_strips_think_from_final_text():
    turn = ChatTurn(text="<think>let me reason...</think>\nThe answer is 42.")
    assert QwenToolCodec().final_text(turn) == "The answer is 42."


def test_openai_codec_parses_structured_tool_calls():
    turn = ChatTurn(raw_tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": '{"x": 1}'}}])
    call = OpenAIToolCodec().decode(turn)
    assert call.name == "f" and call.args == {"x": 1} and call.id == "c1"


def test_encode_is_shared_openai_schema():
    schema = QwenToolCodec().encode([_add_tool()])
    assert schema[0]["type"] == "function" and schema[0]["function"]["name"] == "add"


# ----------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------
def test_executor_runs_and_handles_errors():
    ex = ToolExecutor([_add_tool()])
    assert ex.execute(ToolCall(name="add", args={"a": 2, "b": 3})) == 5
    assert "unknown tool" in ex.execute(ToolCall(name="nope"))


# ----------------------------------------------------------------------
# Agent loop — LOCAL-style handle (Qwen text codec)
# ----------------------------------------------------------------------
def test_agent_runs_tool_loop_then_final():
    handle = FakeChatHandle([
        ChatTurn(text='<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 3}}\n</tool_call>'),
        ChatTurn(text="the answer is 5"),
    ])
    traj = Agent(handle, tools=[_add_tool()], max_turns=5).run("compute 2+3")
    roles = [s.role for s in traj.steps]
    assert roles == [StepRole.USER, StepRole.ACTOR, StepRole.TOOL, StepRole.ACTOR]
    assert traj.steps[1].tool_call["name"] == "add"
    assert traj.steps[2].observation == 5
    assert traj.final_answer == "the answer is 5"
    assert traj.metrics["terminated"] == "final"
    assert traj.metrics["n_tool_calls"] == 1
    assert traj.outcome is Label.UNKNOWN  # correctness is a separate analyzer


def test_agent_stops_at_max_turns():
    looping = FakeChatHandle([ChatTurn(text='<tool_call>{"name": "add", "arguments": {"a": 1, "b": 1}}</tool_call>')])
    traj = Agent(looping, tools=[_add_tool()], max_turns=3).run("loop")
    assert traj.metrics["terminated"] == "max_turns"
    assert traj.metrics["n_turns"] == 3


def test_agent_requires_tool_calls_capability():
    weak = FakeChatHandle([ChatTurn(text="hi")], caps=frozenset({Capability.GENERATE}))
    with pytest.raises(CapabilityError):
        Agent(weak, tools=[_add_tool()])


# ----------------------------------------------------------------------
# Same Agent on an API handle (OpenAI structured codec) — backend-agnostic
# ----------------------------------------------------------------------
def test_agent_runs_on_api_handle_with_chat_fn():
    calls = {"n": 0}

    def scripted_chat_fn(messages, tools=None, model=""):
        calls["n"] += 1
        if calls["n"] == 1:
            return ChatTurn(raw_tool_calls=[{"id": "c1", "function": {"name": "add", "arguments": '{"a": 4, "b": 1}'}}])
        return ChatTurn(text="done: 5")

    handle = compose("qwen3-8b", "api", RuntimeConfig(chat_fn=scripted_chat_fn))
    assert isinstance(codec_for(handle), OpenAIToolCodec)
    traj = Agent(handle, tools=[_add_tool()]).run("add 4 and 1")
    assert traj.steps[2].observation == 5
    assert traj.final_answer == "done: 5"
    assert traj.metrics["n_tool_calls"] == 1


def test_codec_for_routes_local_to_qwen():
    handle = FakeChatHandle([ChatTurn(text="x")])
    assert isinstance(codec_for(handle), QwenToolCodec)


# ----------------------------------------------------------------------
# Multimodal loop — image cases and image-bearing tool results
# ----------------------------------------------------------------------
def _image_blocks(message):
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "image"]


def test_case_image_goes_into_user_content_blocks():
    from evalvitals.core.case import FailureCase, Inputs

    sentinel = object()
    handle = FakeChatHandle([ChatTurn(text="an answer")])
    traj = Agent(handle, tools=[_add_tool()]).run(
        FailureCase(inputs=Inputs(prompt="what is this?", image=sentinel))
    )
    user_msg = handle.seen_messages[0][0]
    blocks = _image_blocks(user_msg)
    assert len(blocks) == 1 and blocks[0]["image"] is sentinel
    assert {"type": "text", "text": "what is this?"} in user_msg["content"]
    assert traj.goal == "what is this?"
    assert traj.steps[0].span["has_image"] is True


def test_text_only_case_keeps_plain_string_content():
    handle = FakeChatHandle([ChatTurn(text="ok")])
    Agent(handle, tools=[_add_tool()]).run("just text")
    assert handle.seen_messages[0][0]["content"] == "just text"


def _crop_tool(sentinel):
    from evalvitals.core.tool import ToolResult

    return Tool(
        name="crop",
        description="crop",
        parameters={"type": "object", "properties": {}},
        fn=lambda: ToolResult(text="cropped it", images=[sentinel], meta={"k": "v"}),
    )


def test_toolresult_images_are_reinjected_as_a_user_message():
    sentinel = object()
    handle = FakeChatHandle([
        ChatTurn(text='<tool_call>{"name": "crop", "arguments": {}}</tool_call>'),
        ChatTurn(text="done"),
    ])
    Agent(handle, tools=[_crop_tool(sentinel)]).run("zoom please")
    history = handle.seen_messages[1]  # what the model saw on turn 2
    assert history[-1]["role"] == "user"
    blocks = _image_blocks(history[-1])
    assert len(blocks) == 1 and blocks[0]["image"] is sentinel
    assert history[-2] == {"role": "tool", "content": "cropped it"}  # text-only observation


def test_toolresult_observation_is_recorded_structured():
    sentinel = object()
    handle = FakeChatHandle([
        ChatTurn(text='<tool_call>{"name": "crop", "arguments": {}}</tool_call>'),
        ChatTurn(text="done"),
    ])
    traj = Agent(handle, tools=[_crop_tool(sentinel)]).run("zoom please")
    assert traj.steps[2].observation == {"text": "cropped it", "n_images": 1, "meta": {"k": "v"}}


def test_collect_message_images_orders_across_messages():
    from evalvitals.models.backends.hf_local import _collect_message_images

    class FakeImg:
        size, mode = (2, 2), "RGB"  # PIL-like so it passes through unopened

    a, b = FakeImg(), FakeImg()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "image", "image": a}, {"type": "text", "text": "q"}]},
        {"role": "assistant", "content": "calling tool"},
        {"role": "user", "content": [{"type": "image", "image": b}, {"type": "text", "text": "crop"}]},
    ]
    assert _collect_message_images(messages) == [a, b]
