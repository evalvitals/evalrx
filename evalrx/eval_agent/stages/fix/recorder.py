"""The fix stage's run-log writer.

FixRecorder is the only part of the fix stage that talks to the run logger.
Every write is best-effort: a logging failure never breaks a repair.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from evalrx.agent_runtime.judge_call import Exchange
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model
    from evalrx.eval_agent.runlog.context import Trial
    from evalrx.eval_agent.stages.fix.types import FixOutcome

logger = logging.getLogger(__name__)


class FixRecorder:
    """Records fix-stage events on an optional run logger."""

    def __init__(self, run_logger: "Any | None" = None) -> None:
        self.run_logger = run_logger
        self._outcomes = True

    @contextmanager
    def without_outcome(self) -> "Iterator[None]":
        """Record no ``fix`` outcome inside the block; the caller records one
        that combines several searches.

        Every model and judge exchange is still logged: they are the only
        record of what a candidate asked and what it got back.
        """
        self._outcomes = False
        try:
            yield
        finally:
            self._outcomes = True

    def outcome(self, outcome: "FixOutcome") -> None:
        """Record the stage outcome as a ``fix`` event."""
        if self.run_logger is None or not self._outcomes:
            return
        try:
            self.run_logger.log_fix(outcome)
        except Exception as exc:  # logging must never break the fix step
            logger.debug("FixAgent: log_fix failed: %s", exc)

    def judge_exchanges(self, operation: str, exchanges: "Sequence[Exchange]") -> None:
        """Log every attempt of one fix-judge call; logging must never break repair."""
        if self.run_logger is None:
            return
        for exchange in exchanges:
            call = exchange.record()
            try:
                self.run_logger.log_model_exchange(
                    "M5", role="fix_judge", operation=operation,
                    inputs=call["inputs"], output=call["output"], error=call["error"],
                    duration_sec=call["duration_sec"], metadata=call["metadata"],
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
        if self.run_logger is None:
            return
        try:
            self.run_logger.log_model_exchange(
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
        """Log one codegen attempt: its prompt, code and the coder's narration.

        The text goes into the event. With a *trial*, the event also names
        ``trial.root`` so the attempt joins the workspace it ran in.
        """
        extra: "dict[str, Any] | None" = (
            {"cli_usage": usage}
            if source.startswith("cli:") and usage
            else None
        )
        if trial is not None:
            # The trial workspace is ephemeral execution state, snapshotted
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
        """Wrap *model* so every call is logged, when the run has a log."""
        if self.run_logger is None:
            return model
        from evalrx.eval_agent.runlog.instrumentation import InstrumentedModel

        return InstrumentedModel(
            model, self.run_logger, stage="M5",
            cycle=int(getattr(self.run_logger, "current_cycle", -1)),
            analyzer=analyzer,
            case_prompts={case.inputs.prompt: case.id for case in data},
            batch_case_ids=[case.id for case in data],
        )
