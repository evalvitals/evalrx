"""Agent — a backend-AGNOSTIC tool-calling loop over any handle.

``Agent(wraps=handle)`` works identically on an API model and a local model: the
loop only needs ``GENERATE`` + ``TOOL_CALLS`` (verified up front), never model
internals.  The single backend/model-specific piece is the
:class:`~evalvitals.models.toolcodec.ToolCallCodec` (auto-selected).  Tool
execution goes through a pluggable :class:`ToolExecutor` — swap in the existing
``APIToolHandler`` for production (image handling, etc.).

White-box backends additionally let an analyzer capture ONE step's internals via
``handle.forward(...)``, but the trajectory production here is the same loop.
Torch-free.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Optional

from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.core.case import (
    FailureCase,
    Inputs,
    Label,
    Step,
    StepRole,
    Trajectory,
)
from evalvitals.core.tool import Tool, ToolCall, ToolResult
from evalvitals.models.toolcodec import ToolCallCodec, codec_for


def _user_content(case: FailureCase) -> "str | list":
    """Build the first user message's content: plain text, or content blocks
    (transformers-style ``{"type": "image", "image": ...}``) when the case
    carries an image — the multimodal chat convention every ``Model.chat``
    implementation speaks (hf_local consumes it natively; API adapters convert
    image blocks to their wire format).
    """
    image = case.inputs.image
    if image is None:
        return case.inputs.prompt
    return [
        {"type": "image", "image": image},
        {"type": "text", "text": case.inputs.prompt},
    ]


def _observation_record(observation: Any) -> Any:
    """What lands in ``Step.observation`` — structured for :class:`ToolResult`
    (text + image count + host meta), the raw value otherwise."""
    if isinstance(observation, ToolResult):
        record: dict = {"text": observation.text, "n_images": len(observation.images)}
        if observation.meta:
            record["meta"] = observation.meta
        return record
    return observation


class ToolExecutor:
    """Run a decoded :class:`ToolCall` against the registered tools.

    The simple default; production swaps in the existing ``APIToolHandler``
    (which also handles tool-output images, retries, etc.).
    """

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._by_name = {t.name: t for t in tools}

    def execute(self, call: ToolCall) -> Any:
        tool = self._by_name.get(call.name)
        if tool is None:
            return f"[error: unknown tool {call.name!r}]"
        if tool.fn is None:
            return f"[error: tool {call.name!r} has no implementation]"
        try:
            return tool.fn(**(call.args or {}))
        except Exception as exc:  # surface tool errors to the model, don't crash the loop
            return f"[tool error in {call.name!r}: {exc}]"


class APIToolHandlerExecutor:
    """Bridge the Agent's tool execution to the XSkill engine's ``APIToolHandler``.

    The handler is INJECTED (not imported here), keeping evalvitals decoupled from
    the engine repo.  It faithfully calls the real signature
    ``execute_tool_call(tool_name, parameters, node, turn_idx, tool_call_id)`` and
    returns the processed text; tool-output images accumulate in ``self.new_images``
    for the caller to attach to the trajectory.

    Usage::

        from engine.api_tool_handler import APIToolHandler          # your repo
        handler = APIToolHandler(args, save_dir)
        agent = Agent(handle, tools, executor=APIToolHandlerExecutor(handler, node))

    ``node`` is the engine's ``SearchNode`` (the handler mutates it, e.g. image_map);
    pass the same node the engine drives so multi-turn image/state threading works.
    """

    def __init__(self, handler: Any, node: Any = None, *, result_key: str = "processed_result") -> None:
        self.handler = handler
        self.node = node
        self.result_key = result_key
        self._turn = 0
        self.new_images: list = []
        self.feedback_messages: list = []

    def execute(self, call: ToolCall) -> Any:
        self._turn += 1
        out = self.handler.execute_tool_call(
            call.name, call.args or {}, self.node, self._turn, getattr(call, "id", None)
        )
        if isinstance(out, dict):
            self.new_images.extend(out.get("new_images") or [])
            self.feedback_messages.extend(out.get("feedback_messages") or [])
            return out.get(self.result_key) or out.get("tool_result") or ""
        return out


def _as_case(data: Any) -> FailureCase:
    if isinstance(data, FailureCase):
        return data
    if isinstance(data, Inputs):
        return FailureCase(inputs=data)
    if isinstance(data, str):
        return FailureCase.from_prompt(data)
    raise TypeError(f"Agent.run expects str | Inputs | FailureCase, got {type(data).__name__}")


class Agent:
    """A tool-calling agent composed over a model handle (any backend)."""

    requires = frozenset({Capability.GENERATE, Capability.TOOL_CALLS})

    def __init__(
        self,
        handle,
        tools: Iterable[Tool],
        *,
        codec: Optional[ToolCallCodec] = None,
        executor: Optional[ToolExecutor] = None,
        max_turns: int = 10,
        system: Optional[str] = None,
    ) -> None:
        missing = self.requires - set(getattr(handle, "capabilities", frozenset()))
        if missing:
            raise CapabilityError(analyzer="Agent", model=repr(handle), missing=missing)
        if not hasattr(handle, "chat"):
            raise TypeError(
                f"{type(handle).__name__} has no chat(); agent mode needs a tool-aware chat method."
            )
        self.handle = handle
        self.tools = list(tools)
        self.codec = codec or codec_for(handle)
        self.executor = executor or ToolExecutor(self.tools)
        self.max_turns = max_turns
        self.system = system
        # An Agent still exposes the underlying model's capabilities (pure-model
        # analysis remains available on self.handle).
        self.capabilities = handle.capabilities

    def run(self, data: Any) -> Trajectory:
        """Drive the tool loop to completion and return a :class:`Trajectory`.

        Multimodal cases work end-to-end: ``case.inputs.image`` goes into the
        first user message as a content block, and a :class:`ToolResult` whose
        ``images`` are non-empty gets them re-injected as a follow-up user
        message — so the model *sees* what its tool produced (zoom crops etc.).
        """
        case = _as_case(data)
        goal = case.inputs.prompt
        encoded = self.codec.encode(self.tools)

        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": _user_content(case)})

        steps: list[Step] = [
            Step(
                idx=0,
                role=StepRole.USER,
                content=goal,
                span={"has_image": case.inputs.image is not None},
            )
        ]
        final_answer: Optional[str] = None
        terminated = "max_turns"

        turn = 0
        for turn in range(1, self.max_turns + 1):
            t0 = time.perf_counter()
            chat = self.handle.chat(messages, tools=encoded)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            call = self.codec.decode(chat)
            span: dict = {"turn": turn, "latency_ms": latency_ms}
            if chat.usage:
                span.update(chat.usage)
            steps.append(
                Step(
                    idx=len(steps),
                    role=StepRole.ACTOR,
                    content=chat.text,
                    tool_call=call.to_dict() if call else None,
                    span=span,
                )
            )
            messages.append(self.codec.assistant_message(chat, call))

            if call is None:
                final_answer = self.codec.final_text(chat)  # strips <think> etc.
                terminated = "final"
                break

            observation = self.executor.execute(call)
            steps.append(
                Step(
                    idx=len(steps),
                    role=StepRole.TOOL,
                    content=call.name,
                    observation=_observation_record(observation),
                    span={"turn": turn},
                )
            )
            messages.append(self.codec.tool_message(call, str(observation)))
            images = observation.images if isinstance(observation, ToolResult) else []
            if images:
                blocks: list = [{"type": "image", "image": im} for im in images]
                blocks.append(
                    {
                        "type": "text",
                        "text": f"The image{'s' if len(images) > 1 else ''} above "
                        f"{'were' if len(images) > 1 else 'was'} returned by the "
                        f"{call.name!r} tool call.",
                    }
                )
                messages.append({"role": "user", "content": blocks})

        metrics: dict = {
            "n_steps": len(steps),
            "n_turns": turn,
            "n_tool_calls": sum(1 for s in steps if s.tool_call),
            "terminated": terminated,
            "total_latency_ms": round(
                sum(s.span.get("latency_ms", 0.0) for s in steps if s.role is StepRole.ACTOR), 1
            ),
        }
        for key in ("prompt_tokens", "completion_tokens"):
            total = sum(int(s.span.get(key) or 0) for s in steps if s.role is StepRole.ACTOR)
            if total:
                metrics[f"total_{key}"] = total
        return Trajectory(
            sample_id=case.id,
            goal=goal,
            steps=steps,
            final_answer=final_answer,
            ground_truth=case.expected,
            outcome=Label.UNKNOWN,  # correctness is a separate analyzer's job
            metrics=metrics,
        )

    def __repr__(self) -> str:
        return f"Agent(handle={self.handle!r}, tools={[t.name for t in self.tools]}, codec={self.codec.name})"


def run_batch(
    handle,
    cases: Iterable[Any],
    *,
    tools_factory: "Callable[[FailureCase], Iterable[Tool]]",
    system: Optional[str] = None,
    max_turns: int = 10,
    codec: Optional[ToolCallCodec] = None,
    concurrency: int = 4,
    on_result: "Optional[Callable[[FailureCase, Trajectory], None]]" = None,
) -> list[Trajectory]:
    """Run the tool loop over many cases — the M1-scale batch driver.

    *tools_factory* builds the per-case tool set (visual tools bind to the
    case's image, so each case needs its own instances).  Concurrency uses
    threads and is only safe for API-backed handles (HTTP is reentrant); for
    local backends the loop is forced sequential — one GPU forward at a time.

    A case whose run raises (endpoint down, tool crash outside the executor
    envelope) yields a stub trajectory with ``metrics["terminated"] == "error"``
    instead of killing the batch; ``on_result`` fires per finished case (e.g.
    incremental JSONL persistence).  Results keep the input order.
    """
    from concurrent.futures import ThreadPoolExecutor

    from evalvitals.models.backends.api import APIModel

    normalized = [_as_case(c) for c in cases]
    if concurrency > 1 and not isinstance(handle, APIModel):
        import warnings

        warnings.warn(
            f"run_batch: {type(handle).__name__} is not an API handle; forcing "
            "concurrency=1 (local backends are not thread-safe).",
            stacklevel=2,
        )
        concurrency = 1

    def _one(case: FailureCase) -> Trajectory:
        try:
            agent = Agent(
                handle, tools_factory(case), codec=codec, max_turns=max_turns, system=system
            )
            trajectory = agent.run(case)
        except Exception as exc:  # keep the batch alive; the error IS the observation
            trajectory = Trajectory(
                sample_id=case.id,
                goal=case.inputs.prompt,
                steps=[Step(idx=0, role=StepRole.USER, content=case.inputs.prompt)],
                ground_truth=case.expected,
                metrics={"terminated": "error", "error": repr(exc)},
            )
        if on_result is not None:
            on_result(case, trajectory)
        return trajectory

    if concurrency <= 1:
        return [_one(c) for c in normalized]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(_one, normalized))
