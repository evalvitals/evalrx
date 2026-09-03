"""Judge liveness-probe autodetection (agent_runtime.judges.autodetect).

A quota-exhausted CLI model typically exits 0 with an empty response rather
than raising, so these tests pin the "empty means dead, try the next
candidate" behavior alongside the more obvious "raises means dead" case.
"""

from __future__ import annotations

import pytest

from evalrx.agent_runtime.judges import autodetect
from evalrx.agent_runtime.judges.autodetect import (
    ResolvedJudge,
    pick_live_model,
    resolve_cli_judge,
)


class _FakeModel:
    def __init__(self, response: str = "OK", *, raises: bool = False) -> None:
        self._response = response
        self._raises = raises

    def generate(self, prompt, **kwargs) -> str:
        if self._raises:
            raise RuntimeError("binary not found")
        return self._response


# ---------------------------------------------------------------------------
# pick_live_model
# ---------------------------------------------------------------------------

def test_pick_live_model_returns_first_responder():
    calls = []

    def factory(name):
        calls.append(name)
        return _FakeModel("OK")

    result = pick_live_model(factory, ["a", "b", "c"])
    assert result == "a"
    assert calls == ["a"]  # never tries b/c once a responds


def test_pick_live_model_skips_empty_response_candidates():
    def factory(name):
        return _FakeModel("" if name == "a" else "OK")

    assert pick_live_model(factory, ["a", "b"]) == "b"


def test_pick_live_model_skips_raising_candidates():
    def factory(name):
        return _FakeModel("OK", raises=(name == "a"))

    assert pick_live_model(factory, ["a", "b"]) == "b"


def test_pick_live_model_raises_with_aggregated_errors_when_all_dead():
    def factory(name):
        return _FakeModel("" if name == "a" else "OK", raises=(name == "b"))

    with pytest.raises(RuntimeError) as excinfo:
        pick_live_model(factory, ["a", "b"])
    msg = str(excinfo.value)
    assert "'a'" in msg and "empty response" in msg
    assert "'b'" in msg and "binary not found" in msg


# ---------------------------------------------------------------------------
# resolve_cli_judge
# ---------------------------------------------------------------------------

def test_resolve_cli_judge_agy_succeeds(monkeypatch):
    monkeypatch.setattr(autodetect, "AgyModel", lambda model, timeout_sec: _FakeModel("OK"))
    monkeypatch.setattr(
        autodetect, "pick_agy_model", lambda *a, **k: "some-agy-model"
    )

    resolved = resolve_cli_judge(provider="agy")
    assert isinstance(resolved, ResolvedJudge)
    assert resolved.provider == "agy"
    assert resolved.model == "some-agy-model"


def test_resolve_cli_judge_auto_falls_through_agy_to_claude(monkeypatch):
    def _dead_agy(model, timeout_sec):
        raise RuntimeError("agy binary not found")

    monkeypatch.setattr(autodetect, "AgyModel", _dead_agy)
    monkeypatch.setattr(autodetect, "ClaudeModel", lambda model, timeout_sec, effort: _FakeModel("OK"))
    monkeypatch.setattr(autodetect, "pick_claude_model", lambda *a, **k: "sonnet")

    resolved = resolve_cli_judge(provider="auto")
    assert resolved.provider == "claude"
    assert resolved.model == "sonnet"


def test_resolve_cli_judge_raises_aggregated_error_when_nothing_available(monkeypatch):
    def _dead(*a, **k):
        raise RuntimeError("not found")

    monkeypatch.setattr(autodetect, "AgyModel", _dead)
    monkeypatch.setattr(autodetect, "ClaudeModel", _dead)

    with pytest.raises(RuntimeError) as excinfo:
        resolve_cli_judge(provider="auto")
    assert "no CLI judge available" in str(excinfo.value)


def test_resolve_cli_judge_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        resolve_cli_judge(provider="bogus")


def test_resolve_cli_judge_skips_autoprobe_for_explicit_model(monkeypatch):
    seen = {}

    def _agy(model, timeout_sec):
        seen["model"] = model
        return _FakeModel("OK")

    def _boom(*a, **k):
        raise AssertionError("pick_agy_model should not run for an explicit model name")

    monkeypatch.setattr(autodetect, "AgyModel", _agy)
    monkeypatch.setattr(autodetect, "pick_agy_model", _boom)

    resolved = resolve_cli_judge(provider="agy", model="my-explicit-model")
    assert resolved.model == "my-explicit-model"
    assert seen["model"] == "my-explicit-model"
