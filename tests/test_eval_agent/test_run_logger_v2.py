"""RunLoggerV2 conformance: every log_* method, exercised end to end, must
produce the tidy M1..M5 layout described in RUN_LOGGER_V2.md — few files,
same-type-in-one-json, one folder per stage, nothing but JSON except real
binary media.

``_emit_every_method`` mirrors ``test_log_schema.py``'s
``_emit_every_event_type`` almost line for line (same real domain objects:
``StatsAnalysisReport``, ``DiagnosisResult``, ``InterventionResult``,
``Hypothesis``, ``ExploratoryAnalysisReport``, ``Result``) so this is a
faithful "does V2 handle the same real calls V1 does" check, not a test
written against V2's own assumptions.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def _emit_every_method(run_dir) -> "RunLoggerV2":  # noqa: F821
    from evalrx.analysis.explorer import ExploratoryAnalysisReport
    from evalrx.analysis.stats_agent import StatsAnalysisReport
    from evalrx.core import CaseBatch, FailureCase
    from evalrx.core.result import Result
    from evalrx.eval_agent.hypothesis import Hypothesis, HypothesisStatus
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.eval_agent.stages.diagnosis import DiagnosisResult
    from evalrx.eval_agent.stages.surgery import InterventionResult

    logger = RunLoggerV2(run_dir=run_dir)
    logger.current_cycle = 0

    logger.log_run_start({"model": "fake", "n_cases": 3})
    logger.log_cases(CaseBatch([FailureCase.from_prompt("example", id="case-1")]))

    res = Result(
        analyzer="self_consistency", model="fake",
        findings={"n_samples": 5, "consistency": 0.2, "n_unique": 5, "gen_kwargs": {}},
    )
    logger.log_probe(0, {"self_consistency": res}, judge_prompt="pick analyzers", judge_raw="self_consistency")

    logger.log_explore(
        0,
        ExploratoryAnalysisReport(
            question="q", ok=True, observations=["fail cases have long chains"],
            charts=[{"name": "c", "title": "C", "kind": "bar",
                     "figure_path": str(Path(run_dir) / "explore" / "figures" / "c.png")}],
            tables={"t": "tables/t.csv"}, caveats=["in-sample"],
            adjudication={"method": "e-BH", "alpha": 0.05, "split": "in_sample",
                          "n_host_adjudicated": 1, "n_rejected": 0},
        ),
        out_dir=Path(run_dir) / "explore", duration_sec=1.5,
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

    # M5 (intervention) shape — no "m4_test_name" in evidence.
    iv5 = InterventionResult(
        hypothesis=hyp, status=HypothesisStatus.REFUTED, fixed=False,
        evidence={"verdict": 0.0}, confidence_score=0.0,
        experiment={
            "provider": "llm", "verdict": 0.0, "metrics": {}, "returncode": 0,
            "code": "print('patched')", "stdout": "ok", "cli_raw_output": "thinking...",
        },
    )
    logger.log_surgery(0, hyp, iv5)
    logger.log_experiment(0, hyp, iv5, module="m5")

    # M4 (hypothesis-verification) shape — "m4_test_name" present in evidence.
    iv4 = InterventionResult(
        hypothesis=hyp, status=HypothesisStatus.SUPPORTED, fixed=False,
        evidence={"m4_test_name": "paired_t_test", "verdict": 1.0}, confidence_score=0.9,
    )
    logger.log_surgery(0, hyp, iv4, judge_prompt="protocol check", judge_raw="consistent")

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

    logger.log_fix(SimpleNamespace(to_dict=lambda: {
        "attempted": [{"name": "cand1", "tier": "L1", "outputs": {"case-1": "42"}}],
        "best": "cand1",
    }))
    # A module value that has no "m<N>" substring at all — must still land
    # somewhere findable (M5, via _STAGE_ALIASES), never silently drop.
    logger.log_tool_codegen(
        module="fix_pipeline", name="cmp", need="stats", source="llm", ok=True, code="print(2)",
    )
    logger.log_stage_skipped("M5", "no_accepted_hypothesis")

    logger.log_loop_end(
        SimpleNamespace(cycles=1, resolved=True, final_hypotheses=[hyp]),
        tokens_used=10, timings={"m1": 1.0},
    )
    logger.log_report_published({
        "schema_version": 1, "catalog_version": "evalrx-report@1",
        "json_render_version": "0.19.0", "source_event_seq": 1, "sha256": "abc",
        "generated_by": {"mode": "deterministic", "model": None},
    })
    logger.close()
    return logger


def _load(run_dir: Path, *parts: str) -> dict:
    return json.loads(Path(run_dir, *parts).read_text())


def test_layout_is_run_json_plus_one_folder_per_stage(tmp_path):
    """Rule 3 (M1..M5 each in their own folder) + rule 1 (few files)."""
    run_dir = tmp_path / "run1"
    _emit_every_method(run_dir)

    top_level = {p.name for p in run_dir.iterdir()}
    assert {"run.json", "M1", "M2", "M3", "M4", "M5"} <= top_level
    # Nothing loose at the top level besides run.json, the M-folders, the
    # media/artifacts dirs this run actually used, and the pre-existing
    # Langfuse/observability sidecars (.evalrx/, langfuse_trace.json) —
    # reused unchanged from RunLogger, orthogonal to the four layout rules.
    assert top_level <= {
        "run.json", "M1", "M2", "M3", "M4", "M5", "media", "artifacts",
        ".evalrx", "langfuse_trace.json",
    }

    for stage in ("M1", "M2", "M3", "M4", "M5"):
        stage_dir = run_dir / stage
        assert (stage_dir / "log.json").is_file()
        # Only log.json (+ an optional artifacts/ dir) live directly in a stage
        # folder — no stray .txt/.py/.md files.
        entries = {p.name for p in stage_dir.iterdir()}
        assert entries <= {"log.json", "artifacts"}
        for f in stage_dir.rglob("*"):
            if f.is_file():
                assert f.suffix in {".json", ".npy", ".png"}, f"non-json/media file: {f}"


def test_run_json_holds_run_wide_events_not_stage_content(tmp_path):
    run_dir = tmp_path / "run1"
    _emit_every_method(run_dir)
    doc = _load(run_dir, "run.json")

    assert doc["run_start"]["model"] == "fake"
    assert len(doc["cases"]) == 1
    assert doc["cases"][0]["case_id"] == "case-1"
    assert len(doc["report_published"]) == 1
    assert len(doc["loop_end"]) == 1
    assert doc["loop_end"][0]["cycles"] == 1
    assert len(doc["agent_decisions"]) == 1
    assert len(doc["agent_tool_calls"]) == 1
    # Every module/stage tag this run used was routable — nothing dropped.
    assert doc["unrouted"] == []


def test_same_type_events_share_one_json_array(tmp_path):
    """Rule 2: same-type logging in one json — every 'probe' event is one
    entry in M1/log.json's "probe" list, not its own file."""
    run_dir = tmp_path / "run1"
    _emit_every_method(run_dir)

    m1 = _load(run_dir, "M1", "log.json")
    assert isinstance(m1["probe"], list) and len(m1["probe"]) == 1
    assert m1["probe"][0]["findings"]["self_consistency"]["consistency"] == 0.2
    # Judge prompt/response for the M1 selection judge live INLINE — no
    # prompts/*.txt sibling file, per rule 4.
    assert m1["probe"][0]["judge_prompt"] == "pick analyzers"
    assert m1["probe"][0]["judge_response"] == "self_consistency"
    assert isinstance(m1["tool_codegen"], list) and len(m1["tool_codegen"]) == 1
    assert isinstance(m1["tool_registry"], list) and len(m1["tool_registry"]) == 1
    # tool_codegen's code is a plain string field, not a code_path.
    assert m1["tool_codegen"][0]["code"] == "print(1)"

    m2 = _load(run_dir, "M2", "log.json")
    assert len(m2["analysis"]) == 1
    assert len(m2["explore"]) == 2  # one real report + the failed-before-report shape
    assert m2["explore"][1]["error"] == "explorer produced no report"


def test_m4_m5_surgery_split_matches_the_post_swap_evidence_key(tmp_path):
    """log_surgery must route on "m4_test_name" (hypothesis-verification is
    M4 post stage-id-swap), not on the pre-swap "m5_test_name"."""
    run_dir = tmp_path / "run1"
    _emit_every_method(run_dir)

    m4 = _load(run_dir, "M4", "log.json")
    m5 = _load(run_dir, "M5", "log.json")
    assert len(m4["surgery"]) == 1
    assert m4["surgery"][0]["evidence"]["m4_test_name"] == "paired_t_test"
    assert m4["surgery"][0]["judge_prompt"] == "protocol check"
    assert len(m5["surgery"]) == 1
    assert "m4_test_name" not in m5["surgery"][0]["evidence"]


def test_experiment_and_fix_content_is_inlined_not_written_as_files(tmp_path):
    """Rule 4: code/stdout/cli narration are JSON string values, not
    experiments/*.py + *.txt siblings."""
    run_dir = tmp_path / "run1"
    _emit_every_method(run_dir)

    m5 = _load(run_dir, "M5", "log.json")
    exp = m5["experiment"][0]
    assert exp["code"] == {"main.py": "print('patched')"}
    assert exp["stdout"] == "ok"
    assert exp["cli_raw_output"] == "thinking..."

    fix = m5["fix"][0]
    # Per-case outputs stay inline (rule 1: fewer files beats a lean event).
    assert fix["attempted"][0]["outputs"] == {"case-1": "42"}
    assert fix["best"]["name"] == "cand1"

    # No experiments/, tools/, prompts/, or workspace/ directories anywhere.
    for forbidden in ("experiments", "tools", "prompts", "workspace", "fixes"):
        assert not (run_dir / forbidden).exists()


def test_unroutable_tag_falls_back_through_aliases_then_unrouted(tmp_path):
    """"fix_pipeline" has no "m<N>" substring — must resolve via
    _STAGE_ALIASES to M5, not silently vanish."""
    run_dir = tmp_path / "run1"
    _emit_every_method(run_dir)

    m5 = _load(run_dir, "M5", "log.json")
    codegen_names = [c["tool_name"] for c in m5["tool_codegen"]]
    assert "cmp" in codegen_names


def test_unroutable_tag_with_no_alias_warns_and_lands_in_run_json(tmp_path):
    """A *module* that matches neither the "m<N>" regex nor _STAGE_ALIASES is
    the one path the alias-fallback test above never exercises: it must warn
    (so a bad call site is visible) and still be recoverable from
    run.json["unrouted"] rather than silently dropped."""
    import pytest

    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2

    run_dir = tmp_path / "run1"
    logger = RunLoggerV2(run_dir=run_dir)
    logger.log_run_start()
    with pytest.warns(UserWarning, match="totally_new_stage"):
        logger.log_tool_codegen(
            module="totally_new_stage", name="mystery_tool", need="?",
            source="deterministic", ok=True, code="print(1)",
        )
    logger.close()

    run_doc = _load(run_dir, "run.json")
    assert len(run_doc["unrouted"]) == 1
    unrouted = run_doc["unrouted"][0]
    assert unrouted["tag"] == "totally_new_stage"
    assert unrouted["key"] == "tool_codegen"
    assert unrouted["tool_name"] == "mystery_tool"

    # And it must not have also been filed under some stage by fallback.
    for stage in ("M1", "M2", "M3", "M4", "M5"):
        doc = _load(run_dir, stage, "log.json")
        assert not doc.get("tool_codegen")


def test_stage_skipped_routes_to_the_named_stage(tmp_path):
    run_dir = tmp_path / "run1"
    _emit_every_method(run_dir)
    m5 = _load(run_dir, "M5", "log.json")
    assert m5["stage_skipped"][0]["reason_code"] == "no_accepted_hypothesis"


def test_probe_artifacts_land_under_that_stages_artifacts_dir(tmp_path):
    """Numeric M1 artifacts (e.g. attention tensors) are the one exception to
    "nothing but json" — they must still be scoped under M1/, not run-global."""
    import numpy as np

    from evalrx.core.result import Result
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2

    run_dir = tmp_path / "run1"
    logger = RunLoggerV2(run_dir=run_dir)
    logger.log_run_start()
    res = Result(
        analyzer="attention_sink", model="fake", findings={"n_cases": 1},
        artifacts={"attn_weights": np.zeros((2, 3, 3))},
    )
    logger.log_probe(0, {"attention_sink": res})
    logger.close()

    npy_files = list((run_dir / "M1" / "artifacts").glob("*.npy"))
    assert npy_files, "no .npy artifact written under M1/artifacts/"
    m1 = _load(run_dir, "M1", "log.json")
    rel = m1["probe"][0]["artifact_paths"]["attention_sink/attn_weights"]
    assert rel.startswith("M1/artifacts/")
    assert (run_dir / rel).is_file()


def test_model_calls_are_recorded_and_probe_events_stay_intact_alongside(tmp_path):
    """Wiring parity with the V1 InstrumentedModel fix: ProbeAgent + RunLoggerV2
    together must still capture every target-model call."""
    from evalrx.core.capability import Capability
    from evalrx.core.case import CaseBatch, FailureCase, Inputs
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.eval_agent.stages.probe_agent import ProbeAgent
    from tests.conftest import FakeModel

    run_dir = tmp_path / "run1"
    logger = RunLoggerV2(run_dir=run_dir)
    logger.current_cycle = 0
    model = FakeModel(capabilities={Capability.GENERATE})
    agent = ProbeAgent(run_logger=logger)
    batch = CaseBatch([FailureCase(inputs=Inputs(prompt="q0"), expected="42")])

    results = agent.probe(model, batch, analyzers=["self_consistency"])
    logger.log_probe(0, results)
    logger.close()

    m1 = _load(run_dir, "M1", "log.json")
    assert m1["model_calls"], "no model_call entries recorded"
    assert all(c["analyzer"] == "self_consistency" for c in m1["model_calls"])
    assert m1["probe"][0]["n_model_calls"] == len(m1["model_calls"])


def test_atomic_write_leaves_a_valid_file_after_every_event(tmp_path):
    """Every flush must be all-or-nothing: no reader ever sees a half-written
    file, and no stray .tmp* files survive a clean close()."""
    run_dir = tmp_path / "run1"
    _emit_every_method(run_dir)

    for f in run_dir.rglob("*.json"):
        json.loads(f.read_text())  # raises if truncated/partial
    assert not list(run_dir.rglob(".*.tmp*")), "leftover temp file after close()"


def test_close_ends_the_langfuse_trace(tmp_path):
    run_dir = tmp_path / "run1"
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2

    logger = RunLoggerV2(run_dir=run_dir)
    logger.log_run_start({"model": "fake"})
    logger.close()
    assert (run_dir / "langfuse_trace.json").exists()


def test_full_loop_run_is_a_drop_in_replacement_for_run_logger(tmp_path):
    """The strongest check available without a real model: run an actual
    VLDiagnoseLoop.run() — the same real integration test_holdout_cases_logged
    exercises against RunLogger — with RunLoggerV2 constructed directly
    (bypassing RunContext, a known, documented scope cut) instead of
    ctx.logger, and confirm no other loop/stage code needed to change AND the
    held-out split's cases still get logged (the exact regression that test
    guards against for V1)."""
    from evalrx.core.capability import Capability
    from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
    from evalrx.eval_agent import DiagnosisAgent, VLDiagnoseLoop
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.eval_agent.stages.protocol import ExperimentProtocol
    from tests.conftest import FakeModel
    from tests.test_eval_agent.test_vl_diagnose import ScriptedModel

    cases = CaseBatch([
        FailureCase(
            inputs=Inputs(prompt=f"question {i}"), expected="yes",
            observed="yes" if i % 2 else "no",
            label=Label.PASS if i % 2 else Label.FAIL,
        )
        for i in range(20)
    ])
    all_ids = {c.id for c in cases}
    run_dir = tmp_path / "run1"
    logger = RunLoggerV2(run_dir=run_dir)
    VLDiagnoseLoop(
        model=FakeModel(capabilities={Capability.GENERATE}, modalities={"text"}),
        protocol=ExperimentProtocol(description="does it answer?", task_domain="qa"),
        diagnosis_agent=DiagnosisAgent(judge=ScriptedModel([
            '[{"hypothesis":"it is unstable","failure_mode":"x","test":"attention.entropy"}]'
        ])),
        max_cycles=1, run_logger=logger, confirm_split=0.3,
    ).run(cases)
    logger.close()

    doc = _load(run_dir, "run.json")
    logged_ids = {c["case"]["id"] for c in doc["cases"]}
    missing = all_ids - logged_ids
    assert not missing, (
        f"{len(missing)} of {len(all_ids)} cases were never logged — the same "
        "held-out-split regression test_holdout_cases_logged.py guards against."
    )
    # M1 actually ran and produced a probe entry — the loop drove RunLoggerV2
    # through log_probe (and, transitively, InstrumentedModel/log_model_call)
    # exactly as it drives RunLogger.
    m1 = _load(run_dir, "M1", "log.json")
    assert m1.get("probe")


def test_run_context_v2_keeps_only_json_and_media_and_inlines_reports(tmp_path):
    """The real integration boundary is RunContext, not a standalone logger."""
    from types import SimpleNamespace

    from evalrx.eval_agent.run_context import RunContext

    root = tmp_path / "run"
    ctx = RunContext(
        root, logger_version="v2", config={"model": "fake"},
        observability_mode="offline",
    )
    runtime = ctx.runtime_root
    trial = ctx.new_trial("fixes", "candidate")
    trial.write("prompt.txt", "exact coder prompt")
    trial.write("pipeline.py", "print('candidate')")
    ctx.logger.log_run_start({"model": "fake"})
    ctx.logger.log_tool_codegen(
        module="fix_pipeline", name="candidate", need="repair", source="judge",
        ok=True, prompt="exact coder prompt", raw_output="exact coder response",
        code="print('candidate')",
    )
    report = SimpleNamespace(
        cycles=1, stopped_by="done", resolved=False, all_hypotheses=[],
        final_hypotheses=[], verified_hypotheses=[], all_test_results=[],
    )
    ctx.write_diagnose_report(report, [], discovery=[{"id": "case-1"}])
    ctx.finalize()
    ctx.finalize()  # lifecycle is explicitly idempotent

    assert not runtime.exists()
    assert not (root / "README.txt").exists()
    assert not (root / "report").exists()
    assert not (root / ".evalrx").exists()
    allowed_media = {
        ".npy", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
        ".wav", ".mp3", ".flac", ".ogg", ".mp4", ".avi", ".mov",
    }
    leaked = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() not in allowed_media | {".json"}
    ]
    assert leaked == []

    run_doc = _load(root, "run.json")
    assert run_doc["diagnose_reports"][0]["discovery"] == [{"id": "case-1"}]
    assert run_doc["manifest"]["run_id"] == "run"
    assert "langfuse_trace.json" in run_doc["manifest"]["files"]
    assert {f"M{i}/log.json" for i in range(1, 6)} <= set(run_doc["manifest"]["files"])
    m5 = _load(root, "M5", "log.json")
    call = m5["model_calls"][0]
    assert call["inputs"] == "exact coder prompt"
    assert call["output"] == "exact coder response"


def test_run_context_v2_snapshots_explore_text_and_media(tmp_path):
    from types import SimpleNamespace

    from evalrx.eval_agent.run_context import RunContext

    root = tmp_path / "run"
    ctx = RunContext(root, logger_version="v2", observability_mode="offline")
    explore = ctx.explore_dir
    (explore / "analysis.py").write_text("print('eda')")
    (explore / "table.csv").write_text("name,value\na,1\n")
    (explore / "plot.png").write_bytes(b"png-bytes")
    report = SimpleNamespace(
        ok=True, observations=["signal"], charts=[{"figure_path": str(explore / "plot.png")}],
        tables={"t": str(explore / "table.csv")}, adjudication={}, caveats=[],
        candidate_signals=[], hypotheses=[], attempts=1, error="", code="print('eda')",
        raw_outputs=["agent response"],
    )
    ctx.logger.log_explore(0, report, out_dir=explore)
    runtime = ctx.runtime_root
    ctx.finalize()

    assert not runtime.exists()
    m2 = _load(root, "M2", "log.json")["explore"][0]
    assert m2["workspace_snapshot"]["files"]["analysis.py"] == "print('eda')"
    assert m2["workspace_snapshot"]["files"]["table.csv"].startswith("name,value")
    assert len(m2["workspace_snapshot"]["media"]) == 1
    assert (root / m2["workspace_snapshot"]["media"][0]).is_file()


def test_run_context_v2_root_holds_no_v1_named_directories(tmp_path):
    """Root-purity check for the RunContext default flip (logger_version="v2"
    is now RunContext's default — see run_context.py). The file-suffix check
    in test_run_context_v2_keeps_only_json_and_media_and_inlines_reports would
    not catch an *empty* directory created by a stray V1-style ``_sub()`` call
    (mkdir happens regardless of whether anything is later written into it),
    so this asserts directory names directly."""
    from types import SimpleNamespace

    from evalrx.eval_agent.run_context import RunContext

    root = tmp_path / "run"
    ctx = RunContext(root, observability_mode="offline")  # default: v2
    assert ctx.is_v2

    # Touch every producer path a real run exercises: M2 artifacts, an
    # explore pass, and a fix trial — the three that differ from V1 (item A
    # in the migration plan).
    (ctx.figures_dir / "effect.png").write_bytes(b"png-bytes")
    (ctx.explore_dir / "table.csv").write_text("name,value\na,1\n")
    trial = ctx.new_trial("fixes", "candidate")
    trial.write("pipeline.py", "print('candidate')")

    ctx.logger.log_run_start({"model": "fake"})
    report = SimpleNamespace(
        cycles=1, stopped_by="done", resolved=False, all_hypotheses=[],
        final_hypotheses=[], verified_hypotheses=[], all_test_results=[],
    )
    ctx.write_diagnose_report(report, [], discovery=[])
    ctx.finalize()

    top_level = {p.name for p in root.iterdir()}
    v1_only = {"explore", "figures", "report", "tools", "workspace", "fixes",
               "experiments", "run_log.jsonl", "manifest.json", "README.txt"}
    assert not (top_level & v1_only), top_level
    assert top_level <= {"run.json", "M1", "M2", "M3", "M4", "M5", "artifacts", "contract",
                          "langfuse_trace.json"}
    # figures_dir/artifacts_dir's file did land under the V2 mapping (M2/artifacts),
    # not disappear — this isn't just an absence check.
    assert (root / "M2" / "artifacts" / "effect.png").is_file()


def test_reporting_reader_and_server_discover_v2_run(tmp_path):
    from evalrx.analysis.dashboard import load_loop_story
    from evalrx.observability.tracer import backfill_run_to_langfuse
    from evalrx.reporting.run_events import read_v2_events, resolve_v2_root
    from evalrx.reporting.server import find_run_root

    root = tmp_path / "archive" / "logs"
    _emit_every_method(root)
    events = read_v2_events(root.parent)

    assert resolve_v2_root(root.parent) == root.resolve()
    assert find_run_root(tmp_path / "archive") == root
    assert any(event["event"] == "run_start" for event in events)
    assert any(event["event"] == "case_record" for event in events)
    assert any(event["event"] == "probe" and event["stage"] == "M1" for event in events)
    assert any(event["event"] == "fix" and event["stage"] == "M5" for event in events)
    assert all(event.get("trace_id") for event in events)
    story = load_loop_story(root.parent)
    assert story is not None and story["probes"] and story["diagnoses"]
    backfill = backfill_run_to_langfuse(root.parent, dry_run=True)
    assert backfill["trace_id"] == events[0]["trace_id"]
    assert backfill["events"] == len(events)


def test_non_numeric_probe_artifacts_survive_without_sidecar_files(tmp_path):
    from evalrx.core.result import Result
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2

    artifacts = {"details": {"answer": "yes"}, "samples": ["yes", "no"],
                 "rows": [{"case_id": "a", "score": 0.5}]}
    with RunLoggerV2(tmp_path, observability_mode="offline") as logger:
        logger.log_probe(0, {"probe": Result(
            analyzer="probe", model="stub", findings={}, artifacts=artifacts,
        )})
    probe = _load(tmp_path, "M1", "log.json")["probe"][0]
    assert probe["artifacts"] == {f"probe/{key}": value for key, value in artifacts.items()}
    assert not (tmp_path / "M1" / "artifacts").exists()


def test_fix_trials_and_remaining_runtime_survive_cleanup(tmp_path):
    from evalrx.eval_agent.run_context import RunContext

    with RunContext(tmp_path, logger_version="v2", observability_mode="offline") as ctx:
        trials = [ctx.new_trial("fixes", name) for name in ("first", "second")]
        attempts = []
        for index, trial in enumerate(trials):
            (trial.workspace / "helper.py").write_text(f"VALUE = {index}")
            (trial.workspace / "result.png").write_bytes(bytes([index, 2, 3]))
            attempts.append({"name": str(index), "trial_root": str(trial.root), "outputs": {}})
        ctx.logger.log_fix(SimpleNamespace(to_dict=lambda: {
            "attempted": attempts, "selection_attempted": [], "best": "0",
        }))
        # Files created after log_fix and a discarded trial must also survive.
        leftover = ctx.new_trial("fixes", "discarded")
        leftover.write("run.sh", "echo retained")
        leftover.write("opaque.bin", b"\x00\xff\x01")
        leftover.write("large.txt", "x" * 2_000_001)
        runtime = ctx.runtime_root
    assert not runtime.exists()
    fix = _load(tmp_path, "M5", "log.json")["fix"][0]
    media = []
    for index, attempt in enumerate(fix["attempted"]):
        snapshot = attempt["workspace_snapshot"]
        assert snapshot["files"]["workspace/helper.py"] == f"VALUE = {index}"
        path = snapshot["media"][0]
        assert snapshot["media_files"]["workspace/result.png"] == path
        assert (tmp_path / path).read_bytes() == bytes([index, 2, 3])
        media.append(path)
    assert len(set(media)) == 2
    snapshot = _load(tmp_path, "run.json")["runtime_snapshot"]
    assert "echo retained" in snapshot["files"].values()
    assert "x" * 2_000_001 in snapshot["files"].values()
    assert any((tmp_path / path).read_bytes() == b"\x00\xff\x01" for path in snapshot["media"])
    assert not any(path.suffix in {".py", ".sh", ".txt"} for path in tmp_path.rglob("*"))
    assert "workspace_snapshot" not in attempts[0]  # no mutation of producer data


def test_failed_runtime_archive_does_not_delete_source(tmp_path, monkeypatch):
    import shutil

    import pytest

    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path, logger_version="v2", observability_mode="offline")
    ctx.logger.log_run_start({})
    runtime = ctx.runtime_root
    (runtime / "image.png").write_bytes(b"evidence")
    def fail_copy(*args, **kwargs):
        raise OSError("archive unavailable")
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copy2", fail_copy)
        with pytest.raises(OSError, match="archive unavailable"):
            ctx.finalize()
    assert (runtime / "image.png").read_bytes() == b"evidence"
    ctx.finalize()
    assert not runtime.exists()


def test_m4_case_study_accepts_new_and_legacy_v2_bundles(tmp_path):
    from evalrx.reporting.case_study import _m4
    from evalrx.reporting.run_events import read_v2_events

    _emit_every_method(tmp_path)
    for legacy in (False, True):
        if legacy:
            path = tmp_path / "M4" / "log.json"
            doc = _load(tmp_path, "M4", "log.json")
            for event in doc["surgery"]:
                event.pop("module")
            path.write_text(json.dumps(doc))
        events = read_v2_events(tmp_path)
        rows = _m4(SimpleNamespace(events_of=lambda name: [e for e in events if e["event"] == name]))
        assert len(rows) == 1
        assert rows[0]["status"] == "supported"
        assert rows[0]["test_name"] == "paired_t_test"


def test_loop_summary_keeps_verified_hypothesis_evidence(tmp_path):
    from evalrx.eval_agent.hypothesis import Hypothesis, HypothesisStatus
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.eval_agent.stages.hypothesis_tester import HypothesisTestResult

    hypothesis = Hypothesis(statement="why", target_model="stub", predicted_failure_mode="mode")
    result = HypothesisTestResult(
        hypothesis=hypothesis, status=HypothesisStatus.SUPPORTED, test_name="audit",
        effect_size=0.2, is_consistent_with_protocol=True, confidence=0.9, verdict="supported",
    )
    with RunLoggerV2(tmp_path, observability_mode="offline") as logger:
        logger.log_loop_end(SimpleNamespace(
            cycles=1, stopped_by="verified", all_hypotheses=[hypothesis], verified_hypotheses=[result],
        ))
    assert _load(tmp_path, "run.json")["loop_end"][0]["verified_hypotheses"] == [{
        "statement": "why", "failure_mode": "mode", "status": "supported",
        "confidence": 0.9, "protocol_consistent": True, "verdict": "supported",
    }]


def test_persisted_event_identity_orders_concurrent_calls_and_validates(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from evalrx.eval_agent.log_schema import _validator, build_v2_schema
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.reporting.run_events import read_v2_events

    monkeypatch.setenv("EVALRX_VALIDATE_LOG", "1")
    _emit_every_method(tmp_path)
    validator = _validator(build_v2_schema())
    for event in read_v2_events(tmp_path):
        validator.validate(event)
    root = tmp_path / "concurrent"
    with RunLoggerV2(root, observability_mode="offline") as logger:
        monkeypatch.setattr(logger, "_ts", lambda: "2026-09-07T00:00:00+00:00")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda index: logger.log_model_exchange(
                f"M{index % 5 + 1}", role="test", operation="generate", inputs=index, output=index,
            ), range(30)))
    events = read_v2_events(root)
    assert [event["event_seq"] for event in events] == list(range(1, 31))
    assert len({event["span_id"] for event in events}) == 30
    for event in events:
        validator.validate(event)
    assert [event["event_seq"] for event in read_v2_events(root)] == list(range(1, 31))


def test_v2_schema_validation_warns_but_preserves_invalid_event(tmp_path, monkeypatch):
    import pytest

    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2

    monkeypatch.setenv("EVALRX_VALIDATE_LOG", "1")
    with RunLoggerV2(tmp_path, observability_mode="offline") as logger:
        with pytest.warns(UserWarning, match="violates log schema"):
            logger.log_stage_skipped("M4", "reason", cycle="invalid")
    assert _load(tmp_path, "M4", "log.json")["stage_skipped"][0]["cycle"] == "invalid"

def test_hypothesis_id_joins_m3_to_m4_and_m5(tmp_path):
    """The lineage key a frontend needs to draw "M5 came from this M3
    hypothesis, M4 verified it": the SAME Hypothesis object logged by
    log_diagnosis (M3), log_surgery (M4-shaped and M5-shaped), and
    log_experiment (M5) must carry one consistent id, computed the same way
    evalrx.contract.emit.hypothesis_id does (delegates to the same function)."""
    from evalrx.eval_agent.hypothesis import hypothesis_id

    run_dir = tmp_path / "run1"
    _emit_every_method(run_dir)

    m3 = _load(run_dir, "M3", "log.json")["diagnosis"][0]
    hyps = m3["hypotheses"]
    assert len(hyps) == 1 and hyps[0]["id"]
    hid = hyps[0]["id"]

    surgeries = _load(run_dir, "M4", "log.json")["surgery"] + _load(run_dir, "M5", "log.json")["surgery"]
    assert len(surgeries) == 2  # one M4-shaped, one M5-shaped (see _emit_every_method)
    assert all(s["hypothesis_id"] == hid for s in surgeries)

    experiment = _load(run_dir, "M5", "log.json")["experiment"][0]
    assert experiment["hypothesis_id"] == hid

    # And it's exactly what a consumer re-deriving the id from the same
    # statement (id="", per _emit_every_method's `hyp`) would compute —
    # no drift between the log and an independent join.
    assert hid == hypothesis_id({"id": "", "statement": "s"})
