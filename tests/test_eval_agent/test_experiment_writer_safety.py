"""M5 generated diagnostics must not duplicate the resident model."""

from __future__ import annotations

from evalrx.core.capability import Capability
from evalrx.eval_agent.hypothesis import Hypothesis
from evalrx.eval_agent.stages.experiment_writer import (
    ExperimentWriter,
    ExperimentWriterConfig,
    build_model_context,
)
from tests.conftest import FakeModel


class _Judge(FakeModel):
    def __init__(self, code: str) -> None:
        super().__init__(capabilities={Capability.GENERATE})
        self.code = code
        self.prompts: list[str] = []

    def generate(self, inputs, **kwargs):
        self.prompts.append(str(inputs))
        return f"```python\n{self.code}\n```"


class _NeverRunSandbox:
    def __init__(self) -> None:
        self.called = False

    def run(self, *args, **kwargs):
        self.called = True
        raise AssertionError("unsafe generated experiment must not execute")

    def run_project(self, *args, **kwargs):
        self.called = True
        raise AssertionError("unsafe generated experiment must not execute")


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        statement="the stored answer was misparsed",
        target_model="resident",
        predicted_failure_mode="answer_extraction",
    )


def test_artifact_only_model_context_exposes_no_loader():
    context = build_model_context(FakeModel(), allow_reconstruction=False)
    assert context["load_expr"] == "None"
    assert context["access_mode"] == "artifacts_only"
    assert "evalrx.load" in context["access_note"]


def test_writer_blocks_generated_second_model_load_before_execution():
    judge = _Judge(
        "import evalrx\n"
        "def main():\n"
        "    model = evalrx.load('qwen3-8b')\n"
        "    print('verdict: 1.0')\n"
        "if __name__ == '__main__':\n"
        "    main()"
    )
    sandbox = _NeverRunSandbox()
    writer = ExperimentWriter(
        judge,
        ExperimentWriterConfig(hard_validation_max_repairs=0),
    )
    result = writer.write_and_run(
        _hypothesis(),
        build_model_context(FakeModel(), allow_reconstruction=False),
        "[]",
        sandbox,
    )
    assert result.returncode == -1
    assert "second model" in result.stderr
    assert result.total_sandbox_runs == 0
    assert sandbox.called is False
    assert "stored observed output" in judge.prompts[0]


def test_json_load_is_not_mistaken_for_model_loading():
    code = "import json\nwith open('cases.json') as f:\n    cases = json.load(f)\n"
    assert ExperimentWriter._contains_forbidden_model_load({"experiment.py": code}) is False
