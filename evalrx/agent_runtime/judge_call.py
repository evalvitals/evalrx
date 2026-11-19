"""One typed judge call: a prompt in, a validated pydantic object out.

Every stage that needs structured output from a judge calls :func:`ask_judge`
instead of parsing text itself. PydanticAI runs the exchange:

* the reply's first JSON value is validated against ``output_type``;
* an invalid reply goes back to the judge with the validation errors, up to
  ``retries`` more times;
* when that budget runs out, :class:`JudgeOutputError` is raised;
* a judge that fails outright (timeout, a CLI limit notice) raises
  :class:`JudgeCallError`.

Both errors carry every exchange, so the caller can log what was asked and
what came back. A failed call is never turned into an empty answer here.

Judges are text in, text out (CLI agents, or API models behind
:class:`~evalrx.core.model.Judge`), so each attempt is rendered as one
self-contained prompt: the output schema, the request, and every earlier
answer with the feedback it got.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Sequence, TypeVar

import pydantic_ai
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import TextOutput

# A library does not print a first-run banner into its users' terminals.
pydantic_ai.BANNER_ENABLED = False

T = TypeVar("T", bound=BaseModel)

_SCHEMA_INSTRUCTIONS = (
    "Answer with one JSON value that matches this JSON Schema. Put the JSON "
    "in your reply; nothing else in the reply is read.\n\n{schema}"
)


@dataclass(frozen=True)
class Exchange:
    """One prompt sent to the judge and what came back."""

    role: str
    attempt: int
    prompt: str
    output: str | None
    error: str | None
    duration_sec: float
    images: tuple[str, ...] = ()

    def record(self) -> dict[str, Any]:
        """The model-call record a stage log stores."""
        inputs: Any = {"prompt": self.prompt, "images": list(self.images)} if self.images else self.prompt
        return {
            "role": self.role, "operation": "generate", "inputs": inputs,
            "output": self.output, "error": self.error,
            "duration_sec": self.duration_sec, "metadata": {"attempt": self.attempt},
        }


@dataclass(frozen=True)
class JudgeReply(Generic[T]):
    """A validated answer and the exchanges that produced it."""

    output: T
    exchanges: tuple[Exchange, ...]

    @property
    def raw(self) -> str:
        return self.exchanges[-1].output or ""


class JudgeError(RuntimeError):
    """A judge call that did not produce a valid answer."""

    def __init__(self, message: str, exchanges: Sequence[Exchange]) -> None:
        super().__init__(message)
        self.exchanges = tuple(exchanges)

    @property
    def raw(self) -> str:
        return (self.exchanges[-1].output or "") if self.exchanges else ""


class JudgeCallError(JudgeError):
    """The judge itself failed: it raised instead of answering."""


class JudgeOutputError(JudgeError):
    """The judge answered, but no answer was valid within the retry budget."""


def first_json_value(text: str) -> Any:
    """The first complete JSON object or array in *text*.

    Judges wrap JSON in prose and code fences; this skips anything before
    the first ``{`` or ``[`` that starts a valid JSON value.
    """
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char in "{[":
            try:
                return decoder.raw_decode(text, i)[0]
            except json.JSONDecodeError:
                continue
    raise ValueError("the reply contains no JSON object or array")


def _part_text(part: Any) -> str:
    if isinstance(part, RetryPromptPart):
        return part.model_response()
    content = getattr(part, "content", "")
    if isinstance(content, str):
        return content
    return "\n".join(str(c) for c in content if isinstance(c, str))


def render_prompt(messages: Sequence[ModelMessage], instructions: str | None) -> str:
    """Flatten a PydanticAI conversation into one prompt for a stateless judge."""
    blocks: list[str] = [instructions] if instructions else []
    for message in messages:
        texts = [t for t in (_part_text(p) for p in message.parts) if t]
        if not texts:
            continue
        if isinstance(message, ModelResponse):
            blocks.append("Your previous answer:\n" + "\n".join(texts))
        elif isinstance(message, ModelRequest) and any(
            isinstance(p, RetryPromptPart) for p in message.parts
        ):
            blocks.append("That answer could not be used:\n" + "\n".join(texts))
        else:
            blocks.append("\n".join(texts))
    return "\n\n".join(blocks)


def _accepts_images(judge: Any) -> bool:
    try:
        return "images" in inspect.signature(judge.generate).parameters
    except (TypeError, ValueError):
        return False


def _run_sync(agent: Agent[None, Any], prompt: str) -> Any:
    """``agent.run_sync``, also from a thread whose event loop is running (Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return agent.run_sync(prompt)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(agent.run_sync, prompt).result()


def ask_judge(
    judge: Any,
    prompt: str,
    output_type: type[T],
    *,
    role: str,
    retries: int = 1,
    images: Sequence[str | Path] = (),
    check: Callable[[T], str | None] | None = None,
) -> JudgeReply[T]:
    """Ask *judge* for one ``output_type`` answer to *prompt*.

    Args:
        judge:       Anything with ``generate(prompt, **kwargs) -> str``.
        prompt:      The request. The JSON Schema of ``output_type`` is
                     added by this function; the prompt need not describe it.
        output_type: The pydantic model the answer must validate as.
        role:        Name recorded on every exchange (``diagnosis_judge``, ...).
        retries:     How many more times an invalid answer is sent back.
        images:      Figure paths for a judge whose ``generate`` takes ``images``;
                     ignored for one that does not.
        check:       Optional check on a valid answer. A message asks the judge
                     to revise; on the last attempt the answer is kept anyway.

    Raises:
        JudgeCallError:   the judge raised.
        JudgeOutputError: no valid answer within ``1 + retries`` attempts.
    """
    figures = tuple(str(p) for p in images) if images and _accepts_images(judge) else ()
    exchanges: list[Exchange] = []

    def call_judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        text = render_prompt(messages, info.instructions)
        started = time.perf_counter()
        try:
            raw = judge.generate(text, images=[Path(f) for f in figures]) if figures else judge.generate(text)
        except Exception as exc:
            exchanges.append(Exchange(role, len(exchanges) + 1, text, None, str(exc),
                                      time.perf_counter() - started, figures))
            raise JudgeCallError(f"{role}: the judge failed: {exc}", exchanges) from exc
        exchanges.append(Exchange(role, len(exchanges) + 1, text, str(raw), None,
                                  time.perf_counter() - started, figures))
        return ModelResponse(parts=[TextPart(str(raw))])

    def parse(ctx: RunContext[None], text: str) -> T:
        try:
            answer = output_type.model_validate(first_json_value(text))
        except (ValueError, ValidationError) as exc:
            raise ModelRetry(str(exc)) from exc
        problem = check(answer) if check is not None else None
        if problem and ctx.retry < ctx.max_retries:
            raise ModelRetry(problem)
        return answer

    agent: Agent[None, T] = Agent(
        FunctionModel(call_judge),
        output_type=TextOutput(parse),
        instructions=_SCHEMA_INSTRUCTIONS.format(
            schema=json.dumps(output_type.model_json_schema(), indent=1)
        ),
        retries=retries,
    )
    try:
        result = _run_sync(agent, prompt)
    except JudgeCallError:
        raise
    except UnexpectedModelBehavior as exc:
        raise JudgeOutputError(
            f"{role}: no valid answer after {len(exchanges)} attempt(s)", exchanges
        ) from exc
    return JudgeReply(output=result.output, exchanges=tuple(exchanges))
