"""The in-container runner rebuilds the cases ProbeAgent ships to Docker."""

from __future__ import annotations

from evalrx.agent_runtime._docker_runner import _build_cases
from evalrx.core.case import Label


def test_a_case_without_a_usable_label_is_unknown():
    batch = _build_cases([
        {"prompt": "a", "label": "fail"},
        {"prompt": "b"},
        {"prompt": "c", "label": "not-a-label"},
    ])
    assert [case.label for case in batch] == [Label.FAIL, Label.UNKNOWN, Label.UNKNOWN]
    assert [case.to_dict()["label"] for case in batch] == ["fail", "unknown", "unknown"]
