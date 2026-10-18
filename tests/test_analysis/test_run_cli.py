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


def test_gemini_benchmark_probes_probability_support_automatically(monkeypatch):
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
    args = build_parser().parse_args(['--modality', 'vlm', '--model', 'gemini-3.6-flash'])
    runner.load_model(resolve(args.model, 'vlm', 'gemini'), args, get('chartqa'))
    assert observed['with_logprobs'] == 'auto'
    assert observed['logprobs_model'] == 'gemini-3.6-flash'
    assert observed['top_logprobs'] == 5


def test_agentic_is_default_without_extra_flags():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.diagnosis_mode == "agentic"
    assert args.max_actions == 12
    help_text = parser.format_help()
    for removed in ("--diagnosis-mode", "--diagnosis-retries", "--m1-codegen",
                    "--fix-floor", "--gemini-logprobs", "--gemini-top-logprobs"):
        assert removed not in help_text
    assert parser.parse_args(["--max-actions", "8"]).max_actions == 8


def test_legacy_fixed_protocol_remains_replayable():
    assert build_parser().parse_args(["--diagnosis-mode", "fixed"]).diagnosis_mode == "fixed"


def test_agentic_rejects_unsupported_holdout_before_model_load():
    from evalrx.benchmark.run import run_args
    args = build_parser().parse_args(["--modality", "llm", "--model", "gemini-3.6-flash",
                                     "--diagnosis-mode", "agentic", "--held-out"])
    with pytest.raises(ValueError, match="EXPLORE"):
        run_args(args)


@pytest.mark.parametrize('modality,dataset', [
    ('llm', 'bbh_causal_judgement'), ('vlm', 'chartqa'), ('alm', 'mmau'),
])
def test_default_runner_constructs_agentic_pipeline_for_every_modality(monkeypatch, tmp_path,
                                                                     modality, dataset):
    from evalrx.benchmark.models import resolve
    from evalrx.benchmark.tasks import get
    from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
    from evalrx.eval_agent import AgenticDiagnoseLoop, CaseDiscoveryAgent
    from evalrx.specs import get_spec
    from tests.conftest import FakeModel

    cases = CaseBatch([
        FailureCase(id=str(i), inputs=Inputs(prompt=f'question {i}'), expected='A',
                    observed='A' if i % 2 else 'B', label=Label.PASS if i % 2 else Label.FAIL)
        for i in range(8)
    ])
    args = build_parser().parse_args(['--modality', modality, '--model', 'gemini-3.6-flash',
                                     '--dataset', dataset, '--run-dir', str(tmp_path)])
    resolved = resolve(args.model, modality, 'gemini')
    monkeypatch.setattr(runner, 'ensure_manifest', lambda *a: tmp_path / 'manifest.json')
    monkeypatch.setattr(runner.T, 'build_cases', lambda *a: (cases, []))
    monkeypatch.setattr(runner, 'build_judge', lambda *a: (FakeModel(), 'claude', None, ()))
    monkeypatch.setattr(runner, 'load_model', lambda *a: (
        FakeModel(), {'do_sample': True, 'temperature': 0.6, 'top_p': 0.95},
        get_spec(resolved.spec_key)))
    monkeypatch.setattr(CaseDiscoveryAgent, 'discover', lambda *a, **k: SimpleNamespace(
        cases=cases, n_pass=4, n_fail=4, n_unknown=0, has_m4_groups=True))

    class ReachedController(Exception):
        pass

    def run(loop, data):
        assert loop.probe_agent.allow_codegen
        assert loop.max_actions == 12
        assert loop.fix_agent._floor_candidates == ()
        assert len(data) == 8
        raise ReachedController

    monkeypatch.setattr(AgenticDiagnoseLoop, 'run', run)
    with pytest.raises(ReachedController):
        runner.run(args, get(dataset), resolved)


def test_triage_explainer_is_the_chain_judge_and_never_the_model_under_test(monkeypatch):
    from evalrx.eval_agent.probing.explanations import Explainer

    seen = {}

    def build(args):
        seen.update(allow_codegen=args.allow_codegen, skip_m5=args.skip_m5)
        return SimpleNamespace(generate=lambda prompt: 'OK'), 'llm', args.judge_model or 'gemini-2.5-flash', ()

    monkeypatch.setattr(runner, 'build_judge', build)
    spec = SimpleNamespace(key='qwen3.5-0.8b', hf_repo='Qwen/Qwen3.5-0.8B')
    args = build_parser().parse_args(['--judge-provider', 'gemini', '--m1-policy', 'triage_active',
                                     '--triage-scheduler', 'scored', '--triage-explanations', 'judge'])
    explainer = runner._triage_explainer(args, spec)
    assert isinstance(explainer, Explainer) and seen == {'allow_codegen': False, 'skip_m5': True}
    assert explainer.identity == {'provider': 'gemini', 'model': 'gemini-2.5-flash', 'effort': 'high'}
    rules = build_parser().parse_args(['--triage-explanations', 'judge'])
    with pytest.raises(SystemExit, match='--triage-scheduler scored'):
        runner._triage_explainer(rules, spec)
    same = build_parser().parse_args(['--judge-provider', 'gemini', '--judge-model', 'Qwen/Qwen3.5-0.8B',
                                     '--triage-scheduler', 'scored', '--triage-explanations', 'judge'])
    with pytest.raises(SystemExit, match='model under test'):
        runner._triage_explainer(same, spec)
