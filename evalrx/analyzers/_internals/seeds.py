"""Per-case seeds: the same draws for a case whatever batch it is in, and in every rerun.

A seed taken from a case's position in its batch changes when the batch does,
and the default case id is a random uuid; the case's content (prompt and
media) is neither.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import Any, Iterator


def case_seed(seed: int, case: Any) -> int:
    """From the run *seed* and the case's content (prompt and media), never its position or id."""
    h = hashlib.sha1(str(seed).encode())
    inputs = getattr(case, "inputs", None)
    h.update(str(getattr(inputs, "prompt", "")).encode())
    for slot in ("image", "audio", "video"):
        media = getattr(inputs, slot, None)
        if media is None:
            continue
        h.update(slot.encode())
        h.update(media.tobytes() if hasattr(media, "tobytes") else str(media).encode())
    return int(h.hexdigest()[:12], 16)


@contextmanager
def seeded(seed: int) -> Iterator[None]:
    """Torch's random state seeded with *seed* inside the block (CPU and every CUDA device),
    restored afterwards, so nothing outside sees the reseed. Without torch it does nothing:
    an API model samples on the provider's side."""
    try:
        import torch
    except ImportError:
        yield
        return
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield
