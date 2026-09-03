"""GUI / OS task loader → CaseBatch (agent tasks scored on final env state).

Records carry a ``goal`` (the instruction), an optional ``screenshot`` image, and
an ``expected_state`` (the annotated goal state — τ-bench style, compared to the
final env state rather than string-matching an answer).  Tagged ``gui_os``;
these cases are typically driven by an Agent and analysed at the trajectory level.
"""

from __future__ import annotations

from evalrx.core.case import CaseBatch
from evalrx.datasets.base import Dataset, cases_from_records, read_jsonl

_SAMPLE = [
    {"goal": "Open Settings and enable dark mode.", "expected_state": {"dark_mode": True}, "app": "settings"},
    {"goal": "Create a new file named notes.txt on the desktop.",
     "expected_state": {"file_exists": "~/Desktop/notes.txt"}, "app": "files"},
]


class GUIOSDataset(Dataset):
    """GUI / OS-level agent tasks (goal + optional screenshot + expected state)."""

    def __init__(
        self,
        records: list[dict] | None = None,
        path: str | None = None,
        *,
        prompt_key: str = "goal",
        image_key: str = "screenshot",
    ) -> None:
        self._records = records
        self._path = path
        self._keys = dict(prompt_key=prompt_key, answer_key="expected_state", image_key=image_key)

    @classmethod
    def from_records(cls, records: list[dict], **keys) -> "GUIOSDataset":
        return cls(records=records, **keys)

    @classmethod
    def from_jsonl(cls, path: str, **keys) -> "GUIOSDataset":
        return cls(path=path, **keys)

    @classmethod
    def sample(cls) -> "GUIOSDataset":
        return cls(records=_SAMPLE)

    def load(self) -> CaseBatch:
        records = self._records if self._records is not None else (
            read_jsonl(self._path) if self._path else _SAMPLE
        )
        return cases_from_records(records, tags={"gui_os"}, **self._keys)
