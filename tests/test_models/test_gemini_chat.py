"""GeminiModel.chat — native function calling through a fake google-genai client.

The real SDK is optional (and absent on CI), so these tests inject a fake
``(client, types)`` pair through ``_genai`` and verify the two conversions:
loop messages -> genai contents, and genai response -> ChatTurn with
OpenAI-shaped ``raw_tool_calls`` (what ``OpenAIToolCodec`` decodes).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from evalrx.core.capability import Capability
from evalrx.models.blackbox.gemini import GeminiModel, _to_genai_contents, _to_genai_tools
from evalrx.models.toolcodec import OpenAIToolCodec, codec_for


# -- a minimal fake of google.genai.types ------------------------------
@dataclass
class FunctionCall:
    name: str = ""
    args: dict = field(default_factory=dict)
    id: Optional[str] = None


@dataclass
class FunctionResponse:
    name: str = ""
    response: dict = field(default_factory=dict)


@dataclass
class Part:
    text: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    function_response: Optional[FunctionResponse] = None
    inline: Any = None

    @classmethod
    def from_bytes(cls, data: bytes = b"", mime_type: str = ""):
        return cls(inline=(mime_type, len(data)))


@dataclass
class Content:
    role: str = "user"
    parts: list = field(default_factory=list)


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
class GenerateContentConfig:
    system_instruction: Any = None
    tools: Any = None
    temperature: float = 0.0


class FakeTypes:
    Part = Part
    Content = Content
    FunctionCall = FunctionCall
    FunctionResponse = FunctionResponse
    FunctionDeclaration = FunctionDeclaration
    Tool = GTool
    GenerateContentConfig = GenerateContentConfig


class _FakeImg:
    mode = "RGB"

    def save(self, buf, format=None):
        buf.write(b"12345")


# ----------------------------------------------------------------------
# Conversions
# ----------------------------------------------------------------------
def test_contents_conversion_covers_all_roles():
    messages = [
        {"role": "system", "content": "be careful"},
        {"role": "user", "content": [{"type": "image", "image": _FakeImg()},
                                     {"type": "text", "text": "what?"}]},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "function": {"name": "zoom", "arguments": '{"bbox": [0, 0, 1, 1]}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "zoomed"},
        {"role": "user", "content": "and now?"},
    ]
    system, contents = _to_genai_contents(messages, FakeTypes)
    assert system == "be careful"
    assert [c.role for c in contents] == ["user", "model", "user", "user"]
    assert contents[0].parts[0].inline == ("image/png", 5)
    assert contents[0].parts[1].text == "what?"
    fc = contents[1].parts[0].function_call
    assert fc.name == "zoom" and fc.args == {"bbox": [0, 0, 1, 1]}
    fr = contents[2].parts[0].function_response
    assert fr.name == "zoom" and fr.response == {"result": "zoomed"}
    assert contents[3].parts[0].text == "and now?"


def test_tools_conversion_uses_raw_json_schema():
    tools = [{"type": "function", "function": {
        "name": "zoom", "description": "z",
        "parameters": {"type": "object", "properties": {"bbox": {"type": "array"}}},
    }}]
    out = _to_genai_tools(tools, FakeTypes)
    decl = out[0].function_declarations[0]
    assert decl.name == "zoom"
    assert decl.parameters_json_schema == {"type": "object", "properties": {"bbox": {"type": "array"}}}


# ----------------------------------------------------------------------
# chat() end to end on a fake client
# ----------------------------------------------------------------------
class _FakeModels:
    def __init__(self, parts):
        self._parts = parts
        self.last = None

    def generate_content(self, *, model, contents, config=None, **kw):
        self.last = {"model": model, "contents": contents, "config": config}

        class _Cand:
            pass

        cand = _Cand()
        cand.content = Content(role="model", parts=self._parts)

        class _Resp:
            pass

        resp = _Resp()
        resp.candidates = [cand]
        return resp


class _FakeClient:
    def __init__(self, parts):
        self.models = _FakeModels(parts)


def _patched(model: GeminiModel, client) -> GeminiModel:
    model._genai = lambda: (client, FakeTypes)  # type: ignore[method-assign]
    return model


def test_chat_decodes_function_call_into_openai_shape():
    client = _FakeClient([Part(text="let me zoom. "),
                          Part(function_call=FunctionCall(name="zoom", args={"bbox": [0, 0, 1, 1]}))])
    model = _patched(GeminiModel(client=object()), client)
    turn = model.chat(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        tools=[{"type": "function", "function": {"name": "zoom", "parameters": {}}}],
    )
    assert turn.text == "let me zoom. "
    (call,) = turn.raw_tool_calls
    assert call["function"]["name"] == "zoom"
    assert json.loads(call["function"]["arguments"]) == {"bbox": [0, 0, 1, 1]}
    assert client.models.last["config"].temperature == 0.0
    assert client.models.last["config"].system_instruction == "s"


def test_chat_plain_text_turn():
    model = _patched(GeminiModel(client=object()), _FakeClient([Part(text="a cat")]))
    turn = model.chat([{"role": "user", "content": "q"}])
    assert turn.text == "a cat" and turn.raw_tool_calls is None


def test_gemini_declares_tool_calls_and_routes_to_openai_codec():
    model = GeminiModel(client=object())  # no SDK import on construction
    assert Capability.TOOL_CALLS in model.capabilities
    assert isinstance(codec_for(model), OpenAIToolCodec)
