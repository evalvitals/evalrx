"""The fix stage's run-log writer.

FixRecorder is the only part of the fix stage that talks to the run logger.
Every write is best-effort: a logging failure never breaks a repair.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model
    from evalrx.eval_agent.run_context import Trial
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.eval_agent.stages.fix.types import FixOutcome

logger = logging.getLogger(__name__)


class FixRecorder:
    """Records fix-stage events on an optional run logger."""

    def __init__(self, run_logger: "Any | None" = None) -> None:
        self.run_logger = run_logger

    @contextmanager
    def muted(self) -> "Iterator[None]":
        """Write nothing inside the block (the caller records a combined outcome)."""
        run_logger, self.run_logger = self.run_logger, None
        try:
            yield
        finally:
            self.run_logger = run_logger

    def outcome(self, outcome: "FixOutcome") -> None:
        """Record the stage outcome as a ``fix`` event."""
        if self.run_logger is None:
            return
        try:
            self.run_logger.log_fix(outcome)
        except Exception as exc:  # logging must never break the fix step
            logger.debug("FixAgent: log_fix failed: %s", exc)

    def judge_exchange(
        self, operation: str, prompt: str, output: str, *, error: "str | None" = None,
    ) -> None:
        """Best-effort detailed V2 model-call logging without changing V1."""
        fn = getattr(self.run_logger, "log_model_exchange", None)
        if not callable(fn):
            return
        try:
            fn(
                "M5", role="fix_judge", operation=operation,
                inputs=prompt, output=output, error=error,
            )
        except Exception as exc:  # logging must never break repair
            logger.debug("FixAgent: model exchange logging failed: %s", exc)

    def target_exchange(
        self,
        operation: str,
        inputs: Any,
        output: Any,
        *,
        case_id: str,
        error: "str | None" = None,
        duration_sec: "float | None" = None,
        metadata: "dict[str, Any] | None" = None,
    ) -> None:
        fn = getattr(self.run_logger, "log_model_exchange", None)
        if not callable(fn):
            return
        try:
            fn(
                "M5", role="target_model", operation=operation,
                inputs=inputs, output=output, error=error, duration_sec=duration_sec,
                metadata={"case_id": case_id, **(metadata or {})},
            )
        except Exception as exc:
            logger.debug("FixAgent: target exchange logging failed: %s", exc)

    def bridge_exchange(self, record: "dict[str, Any]") -> None:
        response = record.get("response") or {}
        self.target_exchange(
            "coded_pipeline_bridge", record.get("request"), response.get("output"),
            case_id=str(record.get("case_id") or ""), error=response.get("error"),
            duration_sec=record.get("duration_sec"),
            metadata={
                "replayed_from_recorded_baseline": bool(
                    record.get("replayed_from_recorded_baseline")
                )
            },
        )

    def codegen(
        self,
        name: str,
        prompt: str,
        source: str,
        code: str,
        raw: str,
        *,
        ok: bool,
        trial: "Trial | None" = None,
        usage: "dict | None" = None,
        raw_stream: str = "",
    ) -> None:
        """Log one codegen attempt; persist its prompt/code/thinking.

        With a *trial*, the files live under ``trial.root`` (alongside the
        rest of that attempt's record) instead of the run-global ``tools/`` —
        only a lean event (paths via ``extra["trial_root"]``) goes through
        :meth:`RunLogger.log_tool_codegen`, with no duplicate file copy.
        """
        extra: "dict[str, Any] | None" = (
            {"cli_usage": usage}
            if source.startswith("cli:") and usage
            else None
        )
        inline = bool(getattr(self.run_logger, "inline_text_artifacts", False))
        if trial is not None and not inline:
            if prompt:
                trial.write(f"{name}_prompt.txt", prompt)
            if code:
                trial.write(f"{name}_code.py", code)
            if raw:
                trial.write(f"{name}_agent_thinking.txt", raw)
            if raw_stream:
                trial.write(f"{name}_agent_raw_stream.txt", raw_stream)
            extra = {**(extra or {}), "trial_root": str(trial.root)}
            prompt, code, raw = "", "", ""
        elif trial is not None:
            # V2 keeps the exact prompt/code/response in M5/log.json.  The
            # trial workspace is ephemeral execution state and is snapshotted
            # separately by log_experiment/log_fix.
            extra = {**(extra or {}), "trial_root": str(trial.root)}
        if self.run_logger is None:
            return
        try:
            self.run_logger.log_tool_codegen(
                module="fix_pipeline",
                name=name,
                need="L2 coded repair pipeline",
                source=source,
                ok=ok,
                code=code,
                prompt=prompt,
                raw_output=raw,
                raw_stream=raw_stream,
                error="" if ok else "no code produced",
                extra=extra,
            )
        except Exception as exc:  # logging must never break the fix step
            logger.debug("FixAgent: log_tool_codegen failed: %s", exc)

    def instrument(self, model: "Model", data: "CaseBatch", analyzer: str) -> "Model":
        """Wrap *model* so every call is logged, when the run keeps full model I/O."""
        if not getattr(self.run_logger, "preserve_full_model_io", False):
            return model
        from evalrx.eval_agent.model_instrumentation import InstrumentedModel

        return InstrumentedModel(
            model, cast("RunLoggerV2", self.run_logger), stage="M5",  # only RunLoggerV2 sets preserve_full_model_io
            cycle=int(getattr(self.run_logger, "current_cycle", -1)),
            analyzer=analyzer,
            case_prompts={case.inputs.prompt: case.id for case in data},
            batch_case_ids=[case.id for case in data],
        )
