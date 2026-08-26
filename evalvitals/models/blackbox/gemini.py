"""Gemini API model — black-box GENERATE + native tool-calling via Google's Gen AI SDK.

Install the optional dependency::

    pip install evalvitals[gemini]

Two roles:

* judge — ``GeminiModel(api_key=...)`` as before (``generate()``), and
* **agent base** — ``chat(messages, tools=...)`` speaks the agent loop's
  protocol with Gemini's NATIVE function calling (not an OpenAI-compat
  bridge, whose tool-call shims are historically quirky).  Images in
  content-block messages are sent as inline PNG parts, so multimodal
  tool loops (zoom crops fed back) work like on the other backends.

The loop threads history in OpenAI-ish form (assistant ``tool_calls`` /
``role="tool"`` results, via :class:`~evalvitals.models.toolcodec.OpenAIToolCodec`
— ``tool_call_style = "native"`` routes it); this module converts that history
to ``google-genai`` contents per call.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from evalvitals.core.capability import Capability
from evalvitals.core.case import Inputs
from evalvitals.core.tool import ChatTurn
from evalvitals.models.blackbox.base import BlackboxModel, Trace


def _png_bytes(image: Any) -> bytes:
    """Encode a PIL image (or open a path) as PNG bytes."""
    import io

    if not (hasattr(image, "save") and hasattr(image, "mode")):
        from PIL import Image

        image = Image.open(image).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


_AUDIO_MIME = {"wav": "audio/wav", "mp3": "audio/mp3", "flac": "audio/flac",
               "ogg": "audio/ogg", "m4a": "audio/mp4"}


def _audio_bytes(audio: Any) -> "tuple[bytes, str]":
    """``(bytes, mime_type)`` for an audio path / raw bytes / waveform.

    Reuses the endpoint path's encoder (a path keeps its own format; bytes are
    taken as WAV; a waveform is written as 16-bit PCM WAV), so the two API
    backends send the same bytes for the same case.
    """
    import base64

    from evalvitals.models.backends.openai_compat import _to_input_audio

    encoded = _to_input_audio(audio)
    return base64.b64decode(encoded["data"]), _AUDIO_MIME.get(encoded["format"], "audio/wav")


def _to_genai_contents(messages: list, types: Any) -> "tuple[Optional[str], list]":
    """Convert loop messages (OpenAI-ish history + content blocks) into
    ``(system_instruction, genai contents)``.

    ``role="tool"`` results become ``function_response`` parts; the function
    name comes from the preceding assistant turn's tool call (the loop is
    single-call-per-turn, so pairing by order is exact).
    """
    system: Optional[str] = None
    contents: list = []
    last_call_name = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            system = content if isinstance(content, str) else str(content)
            continue
        if role == "tool":
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=last_call_name or "tool",
                                response={"result": content},
                            )
                        )
                    ],
                )
            )
            continue
        if role == "assistant":
            parts = []
            if isinstance(content, str) and content:
                parts.append(types.Part(text=content))
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                last_call_name = fn.get("name", "")
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(name=last_call_name, args=args)
                    )
                )
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            continue
        # user message: plain text or content blocks (text / image)
        parts = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    parts.append(
                        types.Part.from_bytes(
                            data=_png_bytes(block.get("image")), mime_type="image/png"
                        )
                    )
                elif isinstance(block, dict) and block.get("type") == "audio":
                    data, mime = _audio_bytes(block.get("audio"))
                    parts.append(types.Part.from_bytes(data=data, mime_type=mime))
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(types.Part(text=block.get("text", "")))
        else:
            parts.append(types.Part(text=content if isinstance(content, str) else str(content)))
        contents.append(types.Content(role="user", parts=parts))
    return system, contents


def _to_genai_tools(tools: Optional[list], types: Any) -> Optional[list]:
    """OpenAI tool schemas -> genai Tool with FunctionDeclarations.

    Prefers ``parameters_json_schema`` (raw JSON schema passthrough, newer
    SDKs); falls back to ``parameters`` coercion on older ones.
    """
    if not tools:
        return None
    decls = []
    for t in tools:
        fn = t.get("function", t)
        try:
            decls.append(
                types.FunctionDeclaration(
                    name=fn.get("name", ""),
                    description=fn.get("description", ""),
                    parameters_json_schema=fn.get("parameters"),
                )
            )
        except Exception:  # older SDK without parameters_json_schema
            decls.append(
                types.FunctionDeclaration(
                    name=fn.get("name", ""),
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters"),
                )
            )
    return [types.Tool(function_declarations=decls)]


class GeminiModel(BlackboxModel):
    """Black-box wrapper around the Google Gemini API.

    Args:
        model_id:  Gemini model name (default: ``"gemini-2.5-flash"``).
        api_key:   API key.  Falls back to the ``GEMINI_API_KEY`` environment
                   variable when ``None``.
        client:    An injected ``genai.Client``-shaped object (tests, custom
                   transports); lazily created from *api_key* when ``None``.

    Requires ``pip install google-genai`` (``evalvitals[gemini]``).
    """

    capabilities = frozenset({Capability.GENERATE, Capability.TOOL_CALLS})
    modalities   = frozenset({"text", "image"})
    tool_call_style = "native"  # codec_for -> OpenAIToolCodec (structured calls)

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
        )
        self._client = client

    # -- plumbing ------------------------------------------------------
    def _genai(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "GeminiModel requires the google-genai package. "
                "Install it with: pip install 'evalvitals[gemini]'"
            ) from exc
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "No Gemini API key found. Pass api_key= or set GEMINI_API_KEY."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client, types

    # -- interface -----------------------------------------------------
    def generate(self, inputs: Any, **kwargs) -> str:
        client, types = self._genai()
        image = getattr(inputs, "image", None) if isinstance(inputs, Inputs) else None
        prompt = inputs.prompt if isinstance(inputs, Inputs) else str(inputs)
        contents: list = []
        if image is not None:
            contents.append(types.Part.from_bytes(data=_png_bytes(image), mime_type="image/png"))
        contents.append(prompt)
        response = client.models.generate_content(
            model=self.model_id, contents=contents, **kwargs
        )
        return response.text or ""

    def chat(self, messages: list, tools=None) -> ChatTurn:
        """One tool-aware turn via native Gemini function calling."""
        client, types = self._genai()
        system, contents = _to_genai_contents(messages, types)
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=_to_genai_tools(tools, types),
            temperature=0.0,  # diagnosed runs should be as deterministic as the API allows
        )
        response = client.models.generate_content(
            model=self.model_id, contents=contents, config=config
        )

        texts: list[str] = []
        raw_calls: list[dict] = []
        candidates = getattr(response, "candidates", None) or []
        parts = getattr(candidates[0].content, "parts", None) or [] if candidates else []
        for part in parts:
            if getattr(part, "text", None):
                texts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                raw_calls.append(
                    {
                        "id": getattr(fc, "id", None),
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(dict(fc.args or {})),
                        },
                    }
                )
        um = getattr(response, "usage_metadata", None)
        usage = (
            {
                "prompt_tokens": int(getattr(um, "prompt_token_count", 0) or 0),
                "completion_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
            }
            if um is not None
            else None
        )
        return ChatTurn(text="".join(texts), raw_tool_calls=raw_calls or None, usage=usage)

    def forward(self, inputs: Any, capture: set[Capability], spec: Any = None) -> Trace:  # type: ignore[override]
        raise NotImplementedError(
            "GeminiModel is a black-box model; only generate()/chat() are available."
        )

    def __repr__(self) -> str:
        return f"GeminiModel(model_id={self.model_id!r})"
