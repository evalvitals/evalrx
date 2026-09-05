"""run_log.jsonl has a published schema that is drift-free and matches reality.

These tests are the teeth behind log_schema.py: they prove the committed
``run_log.schema.json`` is (1) in sync with the schema-as-code builder,
(2) a well-formed Draft 2020-12 schema, and — most importantly — (3) that what
``RunLogger`` actually emits for *every* event type validates against it. (3) is
what stops the schema from silently drifting away from the producer.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

jsonschema = pytest.importorskip("jsonschema")


def test_committed_schema_matches_builder():
    """The shipped JSON file must equal build_schema() — re-render after edits."""
    from evalrx.eval_agent.log_schema import SCHEMA_PATH, build_schema

    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == build_schema(), (
        "run_log.schema.json is stale. Re-render it:\n"
        "  python -c \"import json; from evalrx.eval_agent.log_schema import "
        "build_schema, SCHEMA_PATH; SCHEMA_PATH.write_text(json.dumps(build_schema(), "
        "indent=2)+chr(10))\""
    )


def test_schema_is_valid_draft_2020_12():
    from evalrx.eval_agent.log_schema import build_schema

    schema = build_schema()
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    assert schema["$schema"].endswith("2020-12/schema")


def _emit_every_event_type(run_dir) -> list[dict]:
    """Drive every RunLogger.log_* method with real domain objects.

    Returns the decoded JSONL events. Uses the actual report/result/hypothesis
    classes (not bespoke fakes) wherever cheap, so the lines are exactly what a
    real run produces — the whole point of the conformance check.
    """
    from evalrx.analysis.stats_agent import StatsAnalysisReport
    from evalrx.core.result import Result
    from evalrx.eval_agent.hypothesis import Hypothesis, HypothesisStatus
    from evalrx.eval_agent.run_logger import RunLogger
    from evalrx.eval_agent.stages.diagnosis import DiagnosisResult
    from evalrx.eval_agent.stages.surgery import InterventionResult

    logger = RunLogger(run_dir=run_dir)
    logger.current_cycle = 0

    logger.log_run_start({"model": "fake", "n_cases": 3})
    from evalrx.core import CaseBatch, FailureCase
    logger.log_cases(CaseBatch([FailureCase.from_prompt("example", id="case-1")]))

    res = Result(
        analyzer="self_consistency", model="fake",
        findings={"n_samples": 5, "consistency": 0.2, "n_unique": 5, "gen_kwargs": {}},
    )
    logger.log_probe(0, {"self_consistency": res})

    # In-cycle explore step: the real ExploratoryAnalysisReport shape (charts
    # carry figure_path once rendered; adjudication is the host's in-sample dict).
    from evalrx.analysis.explorer import ExploratoryAnalysisReport

    logger.log_explore(
        0,
        ExploratoryAnalysisReport(
            question="q", ok=True, observations=["fail cases have long chains"],
            charts=[{"name": "c", "title": "C", "kind": "bar",
                     "figure_path": str(run_dir / "explore" / "figures" / "c.png")}],
            tables={"t": "tables/t.csv"}, caveats=["in-sample"],
            adjudication={"method": "e-BH", "alpha": 0.05, "split": "in_sample",
                          "n_host_adjudicated": 1, "n_rejected": 0},
        ),
        out_dir=run_dir / "explore", duration_sec=1.5,
    )
    logger.log_explore(0, None, out_dir=None)   # the failed-before-a-report shape

    logger.log_analysis(
        0,
        StatsAnalysisReport(
            model_name="fake", findings=[], severity="medium",
            narrative="...", raw_results={}, stats_plan=[],
        ),
    )

    hyp = Hypothesis(statement="s", target_model="fake", predicted_failure_mode="m")
    logger.log_diagnosis(
        0, DiagnosisResult(model_name="fake", hypotheses=[hyp], raw_judge_output="raw"),
    )

    iv = InterventionResult(
        hypothesis=hyp, status=HypothesisStatus.REFUTED, fixed=False,
        evidence={"verdict": 0.0}, confidence_score=0.0,
        experiment={"provider": "llm", "verdict": 0.0, "metrics": {}, "returncode": 0},
    )
    logger.log_surgery(0, hyp, iv)
    logger.log_experiment(0, hyp, iv, module="m5")

    logger.log_tool_codegen(
        module="m1_probe", name="t", need="x", source="llm", ok=True, code="print(1)",
    )
    logger.log_tool_registry(
        0, "m1_probe",
        [SimpleNamespace(name="gen_probe", code="print(1)", need="x", source="llm")],
    )

    logger.log_agent_decision(
        0, action="run_probe", params={}, rationale="need M1 findings first", valid=True,
    )
    logger.log_agent_tool(0, tool="run_probe", ok=True, summary="1 analyzer ran")

    logger.log_fix(SimpleNamespace(to_dict=lambda: {"attempted": [], "recommendation": None}))
    logger.log_stage_skipped("M5", "no_accepted_hypothesis")

    logger.log_loop_end(
        SimpleNamespace(cycles=1, resolved=True, final_hypotheses=[hyp]),
        tokens_used=10, timings={"m1": 1.0},
    )
    logger.log_report_published({
        "schema_version": 1,
        "catalog_version": "evalrx-report@1",
        "json_render_version": "0.19.0",
        "source_event_seq": 1,
        "sha256": "abc",
        "generated_by": {"mode": "deterministic", "model": None},
    })
    logger.close()

    return [json.loads(line) for line in (run_dir / "run_log.jsonl").read_text().splitlines()]


def test_real_log_output_conforms_to_schema(tmp_path):
    """Every event RunLogger emits must validate — and every type must be covered."""
    from evalrx.eval_agent.log_schema import EVENT_TYPES, iter_log_errors

    run_dir = tmp_path / "run1"
    events = _emit_every_event_type(run_dir)

    covered = {e["event"] for e in events}
    assert covered == set(EVENT_TYPES), f"uncovered event types: {set(EVENT_TYPES) - covered}"

    errors = list(iter_log_errors(run_dir / "run_log.jsonl"))
    assert not errors, f"real log output violates the published schema: {errors}"


def test_fix_outcome_markdown_includes_explore_selection_audit(tmp_path):
    from evalrx.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path)
    logger.log_fix(SimpleNamespace(to_dict=lambda: {
        "attempted": [],
        "fixed": False,
        "max_tier": "L3a",
        "best": None,
        "recommendation": None,
        "selection_attempted": [{
            "tier": "L2", "name": "coded_pipeline", "verdict": "unsafe",
            "n_fixed": 2, "n_broken": 9, "effect": -0.0275,
        }],
        "selected_on_explore": None,
    }))
    logger.close()

    text = (tmp_path / "fixes" / "outcome.md").read_text(encoding="utf-8")
    assert "EXPLORE selection attempts (1)" in text
    assert "2 | 9 | -0.0275" in text
    assert "Selected on EXPLORE:** none" in text
    assert "Attempts (0)" in text


def test_validation_rejects_malformed_events():
    """The schema must actually reject the breakage it claims to catch."""
    from evalrx.eval_agent.log_schema import validate_event
    from evalrx.eval_agent.run_logger import RUN_LOG_SCHEMA_VERSION

    base = {
        "event": "probe", "schema_version": RUN_LOG_SCHEMA_VERSION,
        "ts": "2026-06-22T18:57:33.296014+00:00", "trace_id": "t", "cycle": 0,
        "analyzers": [], "findings": {}, "artifact_paths": {},
    }
    validate_event(base)  # the valid baseline must pass

    # unknown event name
    with pytest.raises(jsonschema.ValidationError):
        validate_event({**base, "event": "not_a_real_event"})
    # missing a required field (probe needs findings)
    bad = {k: v for k, v in base.items() if k != "findings"}
    with pytest.raises(jsonschema.ValidationError):
        validate_event(bad)
    # malformed timestamp
    with pytest.raises(jsonschema.ValidationError):
        validate_event({**base, "ts": "not-a-timestamp"})
    # wrong type for a core field
    with pytest.raises(jsonschema.ValidationError):
        validate_event({**base, "cycle": "zero"})


def test_optin_runtime_validation_warns_not_raises(tmp_path, monkeypatch):
    """EVALRX_VALIDATE_LOG warns on a bad event but never breaks the run."""
    from evalrx.eval_agent.run_logger import RunLogger

    monkeypatch.setenv("EVALRX_VALIDATE_LOG", "1")
    logger = RunLogger(run_dir=tmp_path / "run1")
    # Inject a structurally invalid event straight through _log; logging must
    # still succeed (warn-only), and the line must still be written.
    with pytest.warns(UserWarning, match="violates run_log schema"):
        logger._log({"event": "probe", "cycle": 0})  # missing required findings etc.
    logger.close()
    assert (tmp_path / "run1" / "run_log.jsonl").read_text().strip()
