from __future__ import annotations

from evalrx.agent_runtime.cli_types import CliAgentConfig, CliAgentResult
from evalrx.agent_runtime.codegen import CodegenRunner


class _FakeCliAgent:
    def __init__(self, result: CliAgentResult) -> None:
        self._result = result

    def run(self, prompt, *, workdir, timeout_sec=None):
        return self._result


def test_codegen_runner_prefers_named_file(monkeypatch, tmp_path):
    result = CliAgentResult(
        files={"other.py": "x = 1\n", "analysis.py": "answer = 1\n"},
        provider_name="fake",
        elapsed_sec=0.1,
        raw_output="trajectory",
    )
    monkeypatch.setattr(
        "evalrx.agent_runtime.providers.registry.create_cli_agent",
        lambda config: _FakeCliAgent(result),
    )

    out = CodegenRunner(CliAgentConfig(provider="claude_code")).write_code(
        "prompt",
        workdir=tmp_path,
        preferred_filenames=("analysis.py",),
    )

    assert out.ok
    assert out.code == "answer = 1\n"
    assert out.raw_output == "trajectory"
    assert out.files == result.files


def test_codegen_runner_falls_back_to_largest_py(monkeypatch, tmp_path):
    result = CliAgentResult(
        files={"small.py": "x\n", "large.py": "x = 1\nprint(x)\n", "note.txt": "ignore"},
        provider_name="fake",
        elapsed_sec=0.1,
    )
    monkeypatch.setattr(
        "evalrx.agent_runtime.providers.registry.create_cli_agent",
        lambda config: _FakeCliAgent(result),
    )

    out = CodegenRunner(CliAgentConfig(provider="claude_code")).write_code(
        "prompt",
        workdir=tmp_path,
        preferred_filenames=("missing.py",),
    )

    assert out.code == "x = 1\nprint(x)\n"


def test_codegen_runner_can_surface_error_as_raw(monkeypatch, tmp_path):
    result = CliAgentResult(
        files={},
        provider_name="fake",
        elapsed_sec=0.1,
        raw_output="",
        error="boom",
    )
    monkeypatch.setattr(
        "evalrx.agent_runtime.providers.registry.create_cli_agent",
        lambda config: _FakeCliAgent(result),
    )

    out = CodegenRunner(CliAgentConfig(provider="claude_code")).write_code(
        "prompt",
        workdir=tmp_path,
        include_error_in_raw=True,
    )

    assert not out.ok
    assert out.code == ""
    assert out.raw_output == "boom"
    assert out.error == "boom"


def test_codegen_runner_run_returns_cli_result(monkeypatch, tmp_path):
    result = CliAgentResult(
        files={"run.py": "print('hi')\n"},
        provider_name="fake",
        elapsed_sec=0.1,
        usage={"input_tokens": 1},
    )
    monkeypatch.setattr(
        "evalrx.agent_runtime.providers.registry.create_cli_agent",
        lambda config: _FakeCliAgent(result),
    )

    out = CodegenRunner(CliAgentConfig(provider="claude_code")).run(
        "prompt",
        workdir=tmp_path,
        timeout_sec=3,
    )

    assert out is result
    assert out.usage == {"input_tokens": 1}
