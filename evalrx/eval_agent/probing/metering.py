"""Per-action resource metering.

``MeteredModel`` wraps the model for one action and counts every
generate/forward/logprobs/chat call and the seconds spent inside them; the
wall clock and the GPU peak (when torch is already loaded) bracket the whole
action. Calls and seconds are reported side by side, never added together.
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
