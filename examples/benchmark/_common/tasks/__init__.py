"""Task registry: ``get(name)`` / ``names(modality)`` over every dataset module."""

from __future__ import annotations

from . import (
    af_reasoning_mcq,
    audiocaps_hallu,
    chartqa,
    gsm8k,
    hotpotqa,
    llm,
    mmau,
    mmsu,
    pope,
    spatial457,
)
from .base import (  # noqa: F401
    Task,
    build_cases,
    label_case,
    load_rows,
    manifest_path,
    score_case,
    write_manifest,
)

TASKS: dict[str, Task] = {}
for _task in (chartqa.TASK, spatial457.TASK, *pope.TASKS, mmau.TASK, mmsu.TASK, audiocaps_hallu.TASK,
              af_reasoning_mcq.TASK, *llm.TASKS, hotpotqa.TASK, gsm8k.TASK):
    TASKS[_task.name] = _task

DEFAULT_TASK = {"vlm": "chartqa", "llm": "bbh_causal_judgement", "alm": "mmau"}


def get(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(TASKS)}")
    return TASKS[name]


def names(modality: str | None = None) -> list[str]:
    return [n for n, t in TASKS.items() if modality is None or t.modality == modality]
