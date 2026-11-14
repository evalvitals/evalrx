"""runner.run end to end, with the model, the judge and the stage calls faked.

What is real: the argument parser, the RunContext and its logger, the label
quarantine around the fix stage, and every agent the runner constructs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evalrx.benchmark import runner
from evalrx.benchmark.models import resolve
from evalrx.benchmark.run import build_parser
from evalrx.benchmark.tasks import get
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.eval_agent import AgenticDiagnoseLoop, CaseDiscoveryAgent, VLDiagnoseLoop
from evalrx.eval_agent.loop_reports import AutoDiagnoseReport
from evalrx.eval_agent.stages.fix.tiers import FixTier
from evalrx.eval_agent.stages.fix.types import FixCandidate, FixOutcome, FixValidation
from evalrx.reporting.run_events import read_v2_events
from evalrx.specs import get_spec
from tests.conftest import FakeModel

GEN = {"do_sample": True, "temperature": 0.6, "top_p": 0.95}


def _batch(prefix: str, n: int) -> CaseBatch:
    return CaseBatch([
        FailureCase(id=f"{prefix}{i}", inputs=Inputs(prompt=f"{prefix} question {i}"), expected="A",
                    observed="A" if i % 2 else "B", label=Label.PASS if i % 2 else Label.FAIL)
        for i in range(n)
    ])


class Cell:
    """One faked benchmark cell; ``calls`` is the order the runner reached things in."""

    def __init__(self, monkeypatch, tmp_path):
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.calls: list[str] = []
        self.loop = None
        self.fix_kwargs: dict = {}
        self.hidden_during_fix: bool | None = None
        self.report = AutoDiagnoseReport(cycles=1, stopped_by="agent_stop")
        self.outcome = FixOutcome(
            max_tier=FixTier.L2_SCAFFOLD,
            attempted=[FixValidation(
                candidate=FixCandidate(tier=FixTier.L1_PROMPT, name="think_first", payload={}),
                n_fixed=2, n_broken=1, effect=0.125, fixed=True)],
            recommendation={"recommend_tier": "L3a"},
        )
        self.m5_proposal = SimpleNamespace(status="supported")
        self.has_m4_groups = True

    def run(self, *argv: str, modality: str = "llm", dataset: str = "bbh_causal_judgement",
            model: str = "gemini-3.6-flash", backend: str | None = "gemini", model_obj=None) -> int:
        cell = self
        patch = self.monkeypatch.setattr
        args = build_parser().parse_args([
            "--modality", modality, "--model", model, "--dataset", dataset,
            "--run-dir", str(self.tmp_path / "runs"), *argv])
        resolved = resolve(args.model, modality, backend)
        self.run_dir = runner.run_dir_for(args, get(dataset))
        target = model_obj or FakeModel()

        def build_judge(_args):
            cell.calls.append("judge")
            return FakeModel(), "claude", None, ()

        def load_model(*_a):
            cell.calls.append("load_model")
            return target, dict(GEN), get_spec(resolved.spec_key)

        def discover(_agent, _model, candidates, protocol=None):
            cell.calls.append(f"discover:{len(candidates)}")
            batch = CaseBatch(list(candidates))
            return SimpleNamespace(cases=batch, n_pass=len(batch) // 2, n_fail=len(batch) - len(batch) // 2,
                                   n_unknown=0, has_m4_groups=cell.has_m4_groups)

        def loop_run(loop, data):
            cell.calls.append(f"loop.run:{type(loop).__name__}:{len(data)}")
            cell.loop = loop
            return cell.report

        def run_m5(_loop, _report, _cases, allow_unverified=False):
            cell.calls.append(f"run_m5:allow_unverified={allow_unverified}")
            return cell.m5_proposal

        def run_fix(_loop, _report, _cases, **kwargs):
            cell.calls.append("run_fix")
            cell.fix_kwargs = kwargs
            cell.hidden_during_fix = not (cell.run_dir / "baseline.json").exists()
            return cell.outcome

        patch(runner, "ensure_manifest", lambda *a: self.tmp_path / "manifest.json")
        patch(runner, "ensure_val_manifest", lambda *a: self.tmp_path / "val_manifest.json")
        patch(runner.T, "build_cases", lambda _task, manifest, _limit: (
            _batch("v" if "val" in manifest.name else "c", 4 if "val" in manifest.name else 8),
            [{"sample_seed": 11}]))
        patch(runner, "build_judge", build_judge)
        patch(runner, "load_model", load_model)
        patch(CaseDiscoveryAgent, "discover", discover)
        patch(AgenticDiagnoseLoop, "run", loop_run)
        patch(VLDiagnoseLoop, "run", loop_run)
        patch(VLDiagnoseLoop, "run_m5", run_m5)
        patch(VLDiagnoseLoop, "run_fix", run_fix)
        return runner.run(args, get(dataset), resolved)

    def read(self, name: str) -> dict:
        return json.loads((self.run_dir / name).read_text())

    def config(self) -> dict:
        """The run configuration as the finished log records it."""
        return self.read("logs/run.json")["manifest"]["config"]

    def events(self) -> list[str]:
        return [e["event"] for e in read_v2_events(self.run_dir / "logs")]


@pytest.fixture
def cell(monkeypatch, tmp_path):
    return Cell(monkeypatch, tmp_path)


def test_a_full_cell_runs_discovery_diagnosis_m5_and_the_fix_then_closes_the_log(cell, capsys):
    assert cell.run() == 0

    assert cell.calls == ["judge", "load_model", "discover:8", "loop.run:AgenticDiagnoseLoop:8",
                          "run_m5:allow_unverified=True", "run_fix"]
    # The gemini backend exposes no internals, so the default L3b ladder stops at L2.
    assert cell.fix_kwargs == {"max_tier": "L2", "auto_escalate": cell.fix_kwargs["auto_escalate"],
                               "allow_unverified": True}
    assert cell.hidden_during_fix is True
    assert (cell.run_dir / "baseline.json").exists()

    baseline = cell.read("baseline.json")
    assert (baseline["n"], baseline["n_pass"], baseline["n_fail"], baseline["accuracy"]) == (8, 4, 4, 0.5)
    assert baseline["cases"][0] == {"id": "c0", "expected": "A", "observed": "B", "label": "fail"}
    assert baseline["generation_kwargs"] == GEN
    assert cell.read("summary.json") == {
        "model": "gemini-3.6-flash", "spec": baseline["spec"], "dataset": "bbh_causal_judgement",
        "modality": "llm", "backend": "gemini", "n_cases": 8, "baseline_accuracy": 0.5, "cycles": 1,
        "stopped_by": "agent_stop", "n_verified": 0,
        "fix": {"recommendation": "{'recommend_tier': 'L3a'}",
                "attempted": [{"tier": "L1", "name": "think_first", "fixed": True, "repairs": 2,
                               "breaks": 1, "effect": 0.125}]},
        "diagnosis_status": "completed", "fix_tier": "L2",
    }
    assert (cell.run_dir / "api_capabilities.json").exists()

    # The loop is faked, so the log holds only what the runner itself writes.
    assert cell.events() == ["diagnose_report", "run_end"]
    assert cell.config()["n_cases"] == 8

    out = capsys.readouterr().out
    assert "Baseline: PASS=4, FAIL=4, UNKNOWN=0, accuracy=0.500" in out
    assert "Diagnosis: stopped_by=agent_stop, cycles=1, verified=0" in out
    assert "[fix] --fix-tier L3b clamped to L2" in out
    assert "M5 experiment on the UNVERIFIED (best lead) hypothesis: status=supported" in out
    assert "Fix [L1] think_first: fixed=True, repairs=2, breaks=1, effect=+0.125" in out
    assert "Done. Tidy M1-M5 logs:" in out


def test_the_loop_is_wired_from_the_arguments(cell):
    cell.run("--max-actions", "5", "--confirm-split", "0.25")

    loop = cell.loop
    assert (loop.max_actions, loop.max_cycles, loop.confirm_split, loop.m4_holdout) == (5, 1, 0.25, False)
    assert loop.val_data is None and loop.explorer is not None
    assert loop.probe_agent.allow_codegen is True
    assert loop.fix_agent.proposer._floor_candidates == ()
    config = cell.config()
    assert (config["m1_workers"], config["m1_codegen"], config["m1_selection"], config["m2_codegen"]) == (
        8, True, "judge", True)
    assert (config["split_mode"], config["n_cases"], config["manifest_seed"]) == ("explore_confirm", 8, 11)


def test_a_diagnosis_error_still_writes_the_summary_and_returns_2(cell):
    cell.report = AutoDiagnoseReport(cycles=2, stopped_by="diagnosis_error")

    assert cell.run("--skip-fix") == 2
    assert cell.calls[-1] == "loop.run:AgenticDiagnoseLoop:8"
    summary = cell.read("summary.json")
    assert (summary["diagnosis_status"], summary["fix"], summary["cycles"]) == ("error", None, 2)
    assert cell.events()[-1] == "run_end"


def test_skip_m5_goes_straight_to_the_fix(cell, capsys):
    assert cell.run("--skip-m5") == 0
    assert cell.calls[-2:] == ["loop.run:AgenticDiagnoseLoop:8", "run_fix"]
    assert "M5: skipped by --skip-m5" in capsys.readouterr().out


@pytest.mark.parametrize("flags, scope", [
    (["--fix-candidate", "self_consistency_5"], "self_consistency_5"),
    (["--code-only"], "coded_pipeline"),
    (["--registered-repairs-only"], "registered methods"),
])
def test_a_pre_registered_fix_scope_skips_m5(cell, capsys, flags, scope):
    assert cell.run(*flags) == 0
    assert "run_m5:allow_unverified=True" not in cell.calls and cell.calls[-1] == "run_fix"
    assert f"M5: skipped for pre-registered fix scope {scope!r}" in capsys.readouterr().out


def test_m5_with_nothing_to_test_says_so(cell, capsys):
    cell.m5_proposal = None
    assert cell.run() == 0
    assert "M5: no hypothesis to experiment on" in capsys.readouterr().out


def test_baseline_only_needs_no_judge_and_writes_no_log(cell, capsys):
    assert cell.run("--baseline-only") == 0
    assert cell.calls == ["load_model", "discover:8"]
    assert cell.read("baseline.json")["n"] == 8
    assert not (cell.run_dir / "logs").exists() and not (cell.run_dir / "summary.json").exists()
    assert "Done (baseline only)." in capsys.readouterr().out


def test_download_only_stops_after_the_manifest(cell):
    assert cell.run("--download-only") == 0
    assert cell.calls == []


def test_a_one_sided_baseline_cannot_be_diagnosed(cell):
    cell.has_m4_groups = False
    with pytest.raises(SystemExit, match="M4 needs both PASS and FAIL cases"):
        cell.run()
    assert cell.calls == ["judge", "load_model", "discover:8"]


def test_held_out_labels_a_separate_validation_set_for_the_fixed_loop(cell, capsys):
    assert cell.run("--diagnosis-mode", "fixed", "--held-out", "--skip-fix") == 0

    assert cell.calls == ["judge", "load_model", "discover:8", "discover:4", "loop.run:VLDiagnoseLoop:8"]
    assert cell.loop.m4_holdout is True and [c.id for c in cell.loop.val_data] == ["v0", "v1", "v2", "v3"]
    val = cell.read("baseline_val.json")
    assert (val["role"], val["n"], val["n_pass"], val["accuracy"]) == ("validation", 4, 2, 0.5)
    config = cell.config()
    assert (config["split_mode"], config["n_val_cases"], config["held_out"]) == ("held_out_val", 4, True)
    assert config["m1_selection"] == "pinned"
    assert "Validation baseline: PASS=2, FAIL=2, accuracy=0.500" in capsys.readouterr().out


def test_weights_are_loaded_before_the_timed_discovery_pass(cell, capsys):
    class Loadable(FakeModel):
        runtime = SimpleNamespace(attn_impl="sdpa")

        def load(self):
            cell.calls.append("load_weights")

    assert cell.run("--baseline-only", model_obj=Loadable()) == 0
    assert cell.calls == ["load_model", "load_weights", "discover:8"]
    assert "(attn_impl=sdpa); generation=" in capsys.readouterr().out


def test_a_deployed_baseline_spec_becomes_the_model_under_test(cell, tmp_path):
    spec_file = tmp_path / "winner.json"
    spec_file.write_text(json.dumps({"payload": {"name": "vote3", "n_samples": 3, "strategy": "majority"}}))

    assert cell.run("--baseline-spec", str(spec_file)) == 0
    assert type(cell.loop.model).__name__ == "SpecPipelineModel"
    assert cell.loop.fix_agent.proposer._deployed_spec["name"] == "vote3"
