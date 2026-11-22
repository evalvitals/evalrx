"""Per-action resource metering.

``MeteredModel`` wraps the model for one action and counts every
generate/forward/logprobs/chat call and the seconds spent inside them; the
wall clock and the GPU peak (when torch is already loaded) bracket the whole
action. Calls and seconds are reported side by side, never added together.

An API model also reports what the provider counted — requests (retries
included separately) and prompt / completion / thought tokens — through
``usage_counters()``; :func:`usage_delta` differences two reads of it. A model
without counters reports ``None``, not zeros.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import Counter
from typing import Any


class MeteredModel:
    """Transparent proxy that counts model calls (repr and attributes forward)."""

    def __init__(self, model: Any) -> None:
        self._metered = model
        self.calls: Counter = Counter()
        self.errors: Counter = Counter()
        self.model_sec = 0.0
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._metered, name)

    def __repr__(self) -> str:
        return repr(self._metered)  # Result.model and cache keys stay unchanged

    def _timed(self, method: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return fn(*args, **kwargs)
        except Exception:
            with self._lock:
                self.errors[method] += 1
            raise
        finally:
            with self._lock:
                self.calls[method] += 1
                self.model_sec += time.monotonic() - t0

    def generate(self, inputs: Any, **kwargs: Any) -> Any:
        return self._timed("generate", self._metered.generate, inputs, **kwargs)

    def forward(self, inputs: Any, capture: Any, spec: Any = None) -> Any:
        return self._timed("forward", self._metered.forward, inputs, capture, spec)

    def logprobs(self, inputs: Any, **kwargs: Any) -> Any:
        return self._timed("logprobs", self._metered.logprobs, inputs, **kwargs)

    def chat(self, messages: list, tools: "list | None" = None) -> Any:
        return self._timed("chat", self._metered.chat, messages, tools)

    # The batch calls are counted per input under the single call's name, so a
    # step's call counts read the same whether its analyzer batches or not.
    def _timed_batch(self, method: str, fn: Any, inputs: Any, **kwargs: Any) -> Any:
        items = list(inputs)
        t0 = time.monotonic()
        failed = len(items)
        try:
            outs = fn(items, **kwargs)
            failed = sum(1 for out in outs if isinstance(out, Exception))
            return outs
        finally:
            with self._lock:
                self.calls[method] += len(items)
                if failed:
                    self.errors[method] += failed
                self.model_sec += time.monotonic() - t0

    def generate_batch(self, inputs: Any, **kwargs: Any) -> Any:
        return self._timed_batch("generate", self._metered.generate_batch, inputs, **kwargs)

    def try_generate_batch(self, inputs: Any, **kwargs: Any) -> Any:
        return self._timed_batch("generate", self._metered.try_generate_batch, inputs, **kwargs)

    def logprobs_batch(self, inputs: Any, **kwargs: Any) -> Any:
        return self._timed_batch("logprobs", self._metered.logprobs_batch, inputs, **kwargs)


def usage_counters(model: Any) -> "dict[str, int] | None":
    """The model's cumulative provider counters (through any proxy), or ``None``."""
    read = getattr(model, "usage_counters", None)
    if not callable(read):
        return None
    try:
        counters = read()
    except Exception:  # noqa: BLE001 - metering never fails the action
        return None
    return dict(counters) if isinstance(counters, dict) else None


def usage_delta(before: "dict[str, int] | None", after: "dict[str, int] | None") -> "dict[str, int] | None":
    """``after - before`` per counter; ``None`` unless both reads exist."""
    if before is None or after is None:
        return None
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in after}


def gpu_peak_reset() -> bool:
    """Reset the CUDA peak counter; ``False`` when there is no CUDA to measure."""
    torch = sys.modules.get("torch")  # never import torch just to meter an API run
    if torch is None or not torch.cuda.is_available():
        return False
    torch.cuda.reset_peak_memory_stats()
    return True


def gpu_peak_mb() -> "float | None":
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
