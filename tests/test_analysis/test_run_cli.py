"""The installed CLI runs the same benchmark path as legacy entry points."""
from types import SimpleNamespace

import pytest

from evalrx.benchmark import runner
from evalrx.benchmark.run import build_parser
from evalrx.cli import main


def test_run_dispatch_preserves_tpu_and_dataset_settings(monkeypatch):
    observed = {}

    def run(args, task, resolved):
        observed.update(device=args.device, backend=resolved.backend, spec=resolved.spec_key,
                        task=task.name, limit=args.limit, baseline_only=args.baseline_only)
        return 7

    monkeypatch.setattr(runner, 'run', run)
    assert main(['run', '--modality', 'vlm', '--model', 'gemma-4-e2b', '--dataset', 'chartqa',
                 '--backend', 'jax_local', '--device', 'tpu', '--limit', '64', '--baseline-only']) == 7
    assert observed == {'device': 'tpu', 'backend': 'jax_local', 'spec': 'gemma-4-e2b-it',
                        'task': 'chartqa', 'limit': 64, 'baseline_only': True}


def test_api_judge_rejects_coding_agent_stages_before_api_call():
    args = build_parser().parse_args(['--judge-provider', 'gemini'])
    with pytest.raises(ValueError, match='coding-agent stages'):
        runner.build_judge(args)


def test_api_judge_uses_model_api_without_coding_cli(monkeypatch):
    import evalrx.models
    from evalrx.models.backends.api import gemini

    observed = {}
    judge = SimpleNamespace(generate=lambda prompt: 'OK')

    def compose(name, backend, runtime):
        observed.update(name=name, backend=backend, runtime=runtime)
        return judge

    monkeypatch.setattr(evalrx.models, 'compose', compose)
    monkeypatch.setattr(gemini, 'gemini_runtime', lambda **kwargs: kwargs)
    args = build_parser().parse_args(['--judge-provider', 'gemini', '--no-allow-codegen',
                                     '--no-explore', '--no-m2-codegen', '--skip-surgery'])
    result = runner.build_judge(args)
    assert result == (judge, 'llm', 'gemini-2.5-flash', ())
    assert observed['backend'] == 'api'
    assert observed['runtime']['max_output_tokens'] == 8192


def test_api_judge_can_launch_separate_gemini_coding_agent(monkeypatch):
    import evalrx.models
    from evalrx.models.backends.api import gemini

    judge = SimpleNamespace(generate=lambda prompt: 'OK')
    monkeypatch.setattr(evalrx.models, 'compose', lambda *args: judge)
    monkeypatch.setattr(gemini, 'gemini_runtime', lambda **kwargs: kwargs)
    args = build_parser().parse_args(['--judge-provider', 'gemini',
                                     '--coder-provider', 'gemini_cli',
                                     '--coder-model', 'gemini-2.5-flash', '--m2-codegen'])
    assert args.allow_codegen and args.explore
    assert runner.build_judge(args) == (judge, 'gemini_cli', 'gemini-2.5-flash', ())


def test_agy_effort_is_forwarded_to_both_judge_and_coder(monkeypatch):
    observed = {}
    judge = SimpleNamespace(generate=lambda prompt: 'OK')

    def factory(**kwargs):
        observed.update(kwargs)
        return judge

    monkeypatch.setattr('evalrx.agent_runtime.judges.AgyModel', factory)
    args = build_parser().parse_args(['--judge-provider', 'agy', '--judge-model',
                                     'gemini-3.6-flash', '--judge-effort', 'low'])
    assert runner.build_judge(args) == (judge, 'antigravity', 'gemini-3.6-flash', ('--effort', 'low'))
    assert observed['effort'] == 'low'


@pytest.mark.parametrize('mode,expected', [('auto', 'auto'), ('on', True), ('off', False)])
def test_gemini_benchmark_forwards_probability_policy(monkeypatch, mode, expected):
    from importlib import import_module

    from evalrx.benchmark.models import resolve
    from evalrx.benchmark.tasks import get
    from evalrx.models.backends.api import gemini

    observed = {}
    def runtime(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(gemini, 'gemini_runtime', runtime)
    monkeypatch.setattr(import_module('evalrx.models.compose'), 'compose', lambda *args: 'model')
    args = build_parser().parse_args(['--modality', 'vlm', '--model', 'gemini-3.6-flash',
                                     '--gemini-logprobs', mode, '--gemini-top-logprobs', '10'])
    runner.load_model(resolve(args.model, 'vlm', 'gemini'), args, get('chartqa'))
    assert observed['with_logprobs'] == expected
    assert observed['logprobs_model'] == 'gemini-3.6-flash'
    assert observed['top_logprobs'] == 10


def test_agentic_cli_exposes_bounded_controller():
    args = build_parser().parse_args(["--diagnosis-mode", "agentic", "--max-actions", "8",
                                     "--no-fix-floor", "--m1-codegen"])
    assert (args.diagnosis_mode, args.max_actions, args.fix_floor, args.m1_codegen) == (
        "agentic", 8, False, True)
    assert args.diagnosis_retries == 1


def test_agentic_rejects_unsupported_holdout_before_model_load():
    from evalrx.benchmark.run import run_args
    args = build_parser().parse_args(["--modality", "llm", "--model", "gemini-3.6-flash",
                                     "--diagnosis-mode", "agentic", "--held-out"])
    with pytest.raises(ValueError, match="EXPLORE"):
        run_args(args)
